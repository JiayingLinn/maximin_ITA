"""Offline checks using fixture text and randomly initialized tiny local models."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, LlamaForSequenceClassification, PreTrainedTokenizerFast

from shp_rm.data import _load_one_split, load_and_tokenize_splits, preference_texts
from shp_rm.trainer import bradley_terry_loss, pairwise_metrics
from train_rm import main as train_main
from train_rm_fsdp import parse_args as parse_fsdp_args
from response_pool import main as pool_main


def make_tiny_model(path):
    vocab = {word: index for index, word in enumerate(
        ["<pad>", "<unk>", "<eos>", "<user>", "<assistant>",
         "question", "good", "bad", "answer", "fixture", "Prompt", "Response", ":"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
        eos_token="<eos>", additional_special_tokens=["<user>", "<assistant>"])
    tokenizer.chat_template = "{% for message in messages %}{{ '<' + message['role'] + '> ' + message['content'] + ' ' }}{% endfor %}{{ eos_token }}"
    tokenizer.save_pretrained(path)
    model = LlamaForSequenceClassification(LlamaConfig(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, num_labels=1, pad_token_id=0,
        eos_token_id=2, bos_token_id=2))
    model.save_pretrained(path)
    return tokenizer


def make_pairs(root):
    domain = root / "fixture_domain"
    domain.mkdir(parents=True)
    for split in ("train", "validation"):
        rows = [{"domain": f"fixture_domain_{split}", "history": "fixture question",
                 "human_ref_A": "good answer" if index % 2 else "bad answer",
                 "human_ref_B": "bad answer" if index % 2 else "good answer",
                 "labels": index % 2} for index in range(4)]
        (domain / f"{split}.json").write_text(json.dumps(rows))
    # An invalid test file proves this path never reads or parses the test split.
    (domain / "test.json").write_text("THIS FILE MUST NEVER BE LOADED")


class RewardTrainingTests(unittest.TestCase):
    def test_preference_mapping_and_gradient_direction(self):
        example = dict(history="question", human_ref_A="A", human_ref_B="B", labels=1)
        self.assertEqual(preference_texts(example), ("question", "A", "B", "A"))
        example["labels"] = 0
        self.assertEqual(preference_texts(example), ("question", "B", "A", "B"))
        chosen = torch.tensor([0.0, 1.0], requires_grad=True)
        rejected = torch.tensor([1.0, 0.0], requires_grad=True)
        loss = bradley_terry_loss(chosen, rejected)
        loss.backward()
        self.assertTrue(torch.all(chosen.grad < 0))
        self.assertTrue(torch.all(rejected.grad > 0))
        metrics = pairwise_metrics(SimpleNamespace(predictions=np.array([[0., 1.], [1., 0.]])))
        self.assertEqual(metrics["pairwise_accuracy"], 0.5)
        self.assertAlmostEqual(float(loss.detach()), metrics["bradley_terry_loss"], places=6)

    def test_local_loading_excludes_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = make_tiny_model(root / "model")
            make_pairs(root / "data")
            args = SimpleNamespace(dataset_path=str(root / "data"), domain="fixture_domain",
                                   dataset_cache_dir=str(root / "cache"), seq_length=8,
                                   preprocessing_num_workers=None, max_train_samples=None,
                                   max_validation_samples=None, overwrite_cache=False)
            data, info = load_and_tokenize_splits(args, tokenizer, ("train", "validation"))
            self.assertEqual(set(data), {"train", "validation"})
            self.assertFalse(info["test_split_loaded"])
            self.assertEqual(len(data["train"]), 4)
            with self.assertRaises(ValueError):
                _load_one_split(args, "test")

    def test_fsdp_arguments_have_no_resource_defaults(self):
        with self.assertRaises(SystemExit):
            parse_fsdp_args([])
        args = parse_fsdp_args(["--domain", "fixture_domain", "--model_path", "MODEL",
                                "--dataset_path", "DATA", "--output_dir", "OUTPUT"])
        self.assertEqual(args.report_to, "none")
        self.assertEqual(args.optim, "adamw_torch")

    def test_tiny_lora_train_save_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tiny_model(root / "model")
            make_pairs(root / "data")
            train_main(["--domain", "fixture_domain", "--model_path", str(root / "model"),
                        "--dataset_path", str(root / "data"), "--output_dir", str(root / "output"),
                        "--dataset_cache_dir", str(root / "cache"), "--use_cpu",
                        "--no-load_in_4bit", "--no-gradient_checkpointing",
                        "--max_steps", "2", "--eval_steps", "1", "--save_steps", "1",
                        "--logging_steps", "1", "--batch_size", "2", "--eval_batch_size", "2",
                        "--gradient_accumulation_steps", "1", "--seq_length", "32",
                        "--dataloader_num_workers", "0", "--lora_r", "2", "--lora_alpha", "4",
                        "--lora_dropout", "0"])
            gradients = json.loads((root / "output" / "gradient_check.json").read_text())
            reloaded = json.loads((root / "output" / "reload_verification.json").read_text())
            self.assertTrue(gradients["passed"])
            self.assertTrue(reloaded["passed"])
            self.assertTrue((root / "output" / "adapter_model.safetensors").exists())

            # Exercise actual generation and adapter inference through the public CLI.
            tokenizer = make_tiny_model(root / "generator")
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=64, pad_token_id=0, eos_token_id=2, bos_token_id=2))
            model.save_pretrained(root / "generator")
            test_rows = [{"domain": "fixture_domain_test", "history": "question answer",
                          "human_ref_A": "good", "human_ref_B": "bad", "labels": 1}]
            (root / "data/fixture_domain/test.json").write_text(json.dumps(test_rows))
            pool_main(["generate", "--model_path", str(root / "generator"),
                       "--dataset_path", str(root / "data"), "--domain", "fixture_domain",
                       "--dataset_cache_dir", str(root / "cache"),
                       "--split", "test", "--output", str(root / "raw.json"),
                       "--dtype", "float32", "--num_prompts", "1", "--responses_per_prompt", "3",
                       "--batch_size", "2", "--max_new_tokens", "3", "--max_prompt_tokens", "16"])
            for role, source, destination in (("proxy", "raw.json", "proxy.json"),
                                              ("judge", "proxy.json", "scored.json")):
                pool_main(["score", "--role", role, "--model_path", str(root / "model"),
                           "--adapter_path", str(root / "output"), "--batch_size", "2",
                           "--input", str(root / source), "--output", str(root / destination)])
            scored = json.loads((root / "scored.json").read_text())
            self.assertEqual(len(scored["prompts"][0]["candidates"]), 3)
            for row in scored["prompts"][0]["candidates"]:
                self.assertTrue(np.isfinite(row["proxy_raw"]))
                self.assertAlmostEqual(row["proxy_raw"], row["judge_raw"], places=6)


if __name__ == "__main__":
    unittest.main()
