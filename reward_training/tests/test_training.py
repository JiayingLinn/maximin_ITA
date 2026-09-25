"""Offline checks using synthetic text and randomly initialized tiny local models."""

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
from transformers import LlamaConfig, LlamaForSequenceClassification, PreTrainedTokenizerFast

from shp_rm.data import _load_one_split, load_and_tokenize_splits, preference_texts
from shp_rm.trainer import bradley_terry_loss, pairwise_metrics
from train_criteria_rm import CRITERIA, criterion_labels, compute_eval_metrics, main as criteria_main
from train_rm import main as train_main
from train_rm_fsdp import parse_args as parse_fsdp_args


def make_tiny_model(path):
    vocab = {word: index for index, word in enumerate(
        ["<pad>", "<unk>", "<eos>", "<user>", "<assistant>",
         "question", "good", "bad", "answer", "synthetic", "Prompt", "Response", ":"])}
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
    domain = root / "synthetic_domain"
    domain.mkdir(parents=True)
    for split in ("train", "validation"):
        rows = [{"domain": f"synthetic_domain_{split}", "history": "synthetic question",
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
            args = SimpleNamespace(dataset_path=str(root / "data"), domain="synthetic_domain",
                                   dataset_cache_dir=str(root / "cache"), seq_length=8,
                                   preprocessing_num_workers=None, max_train_samples=None,
                                   max_validation_samples=None, overwrite_cache=False)
            data, info = load_and_tokenize_splits(args, tokenizer, ("train", "validation"))
            self.assertEqual(set(data), {"train", "validation"})
            self.assertFalse(info["test_split_loaded"])
            self.assertEqual(len(data["train"]), 4)
            with self.assertRaises(ValueError):
                _load_one_split(args, "test")

    def test_regression_scale_and_metrics(self):
        self.assertEqual(criterion_labels(dict(zip(CRITERIA, range(5)))), [0., .25, .5, .75, 1.])
        with self.assertRaises(ValueError):
            criterion_labels(dict.fromkeys(CRITERIA, 5))
        metrics = compute_eval_metrics((np.ones((2, 5)), np.zeros((2, 5))))
        self.assertEqual(metrics["mse_avg"], 1.)
        self.assertEqual(metrics["mae_verbosity"], 1.)

    def test_fsdp_arguments_have_no_resource_defaults(self):
        with self.assertRaises(SystemExit):
            parse_fsdp_args([])
        args = parse_fsdp_args(["--domain", "synthetic_domain", "--model_path", "MODEL",
                                "--dataset_path", "DATA", "--output_dir", "OUTPUT"])
        self.assertEqual(args.report_to, "none")
        self.assertEqual(args.optim, "adamw_torch")

    def test_tiny_criteria_regression_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tiny_model(root / "model")
            rows = [{"prompt": "synthetic question", "response": "good answer",
                     **dict.fromkeys(CRITERIA, float(index))} for index in range(4)]
            for split in ("train", "validation"):
                (root / f"{split}.json").write_text(json.dumps(rows))
            criteria_main(["--model_path", str(root / "model"),
                           "--train_file", str(root / "train.json"),
                           "--validation_file", str(root / "validation.json"),
                           "--dataset_cache_dir", str(root / "cache"),
                           "--output_dir", str(root / "output"), "--use_cpu",
                           "--max_steps", "2", "--eval_steps", "1", "--save_steps", "1",
                           "--batch_size", "2", "--eval_batch_size", "2",
                           "--warmup_steps", "0", "--seq_length", "32"])
            config = json.loads((root / "output" / "config.json").read_text())
            self.assertEqual(config["problem_type"], "regression")
            self.assertEqual(len(config["id2label"]), 5)
            metrics = json.loads((root / "output" / "validation_results.json").read_text())
            self.assertTrue(np.isfinite(metrics["eval_mse_avg"]))

    def test_tiny_lora_train_save_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tiny_model(root / "model")
            make_pairs(root / "data")
            train_main(["--domain", "synthetic_domain", "--model_path", str(root / "model"),
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


if __name__ == "__main__":
    unittest.main()
