"""Strict SHP loading, A/B mapping, tokenization, and pairwise collation."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from datasets import Dataset, load_dataset

from .chat_format import tokenize_pair_text
from .constants import ALLOWED_SPLITS, MODEL_COLUMNS, REQUIRED_COLUMNS


LOGGER = logging.getLogger("fair_bon")


def valid_pair(example: dict[str, Any]) -> bool:
    """Keep all and only examples with usable pairwise fields."""
    try:
        label = int(example["labels"])
    except (KeyError, TypeError, ValueError):
        return False
    if label not in (0, 1):
        return False
    return all(
        isinstance(example.get(name), str) and bool(example[name].strip())
        for name in ("history", "human_ref_A", "human_ref_B")
    )


def preference_texts(example: dict[str, Any]) -> tuple[str, str, str, str]:
    """Map the randomized SHP label to prompt/chosen/rejected text."""
    label = int(example["labels"])
    if label == 1:
        return example["history"], example["human_ref_A"], example["human_ref_B"], "A"
    if label == 0:
        return example["history"], example["human_ref_B"], example["human_ref_A"], "B"
    raise ValueError(f"Expected labels in {{0, 1}}, received {example['labels']!r}")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping_checks(dataset: Dataset, count: int = 8) -> list[dict[str, Any]]:
    checks = []
    for index in range(min(count, len(dataset))):
        example = dataset[index]
        _, chosen, rejected, chosen_source = preference_texts(example)
        rejected_source = "B" if chosen_source == "A" else "A"
        check = {
            "index": index,
            "label": int(example["labels"]),
            "chosen_source": chosen_source,
            "rejected_source": rejected_source,
            "chosen_matches_source": chosen == example[f"human_ref_{chosen_source}"],
            "rejected_matches_source": rejected == example[f"human_ref_{rejected_source}"],
            "chosen_sha256": _digest(chosen),
            "rejected_sha256": _digest(rejected),
        }
        if not check["chosen_matches_source"] or not check["rejected_matches_source"]:
            raise AssertionError(f"Bad SHP A/B mapping: {check}")
        checks.append(check)
    return checks


def _validate_raw_split(dataset: Dataset, domain: str, split: str) -> dict[str, Any]:
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(f"{domain}/{split} is missing columns: {sorted(missing)}")
    if not len(dataset):
        raise ValueError(f"{domain}/{split} is empty")
    domain_values = sorted(str(value) for value in dataset.unique("domain"))
    if domain_values != [f"{domain}_{split}"]:
        raise ValueError(f"Expected domain={domain}_{split}; received {domain_values}")
    return {"split": split, "raw_examples": len(dataset), "domain_values": domain_values}


def _load_one_split(args: Any, split: str) -> Dataset:
    """Read one explicitly named local JSON file; training never loads test."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Only {ALLOWED_SPLITS} may be loaded")
    if not args.domain or Path(args.domain).name != args.domain or args.domain in (".", ".."):
        raise ValueError("Domain must be a single directory name")
    path = Path(args.dataset_path) / args.domain / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required local dataset split not found: {path}")
    return load_dataset("json", data_files={split: str(path)}, split=split,
                        cache_dir=args.dataset_cache_dir)


def load_validated_raw_split(args: Any, split: str) -> tuple[Dataset, dict[str, Any]]:
    dataset = _load_one_split(args, split)
    return dataset, _validate_raw_split(dataset, args.domain, split)


def load_and_tokenize_splits(
    args: Any,
    tokenizer: Any,
    requested_splits: Sequence[str],
) -> tuple[dict[str, Dataset], dict[str, Any]]:
    """Load only requested official splits and return model-only columns."""
    if not requested_splits:
        raise ValueError("At least one split must be requested")
    if len(set(requested_splits)) != len(requested_splits):
        raise ValueError(f"Duplicate requested splits: {requested_splits}")
    if any(split not in ALLOWED_SPLITS for split in requested_splits):
        raise ValueError(f"Only {ALLOWED_SPLITS} may be loaded")

    tokenized: dict[str, Dataset] = {}
    dataset_info: dict[str, Any] = {
        "dataset": args.dataset_path,
        "data_dir": args.domain,
        "requested_splits": list(requested_splits),
        "test_split_loaded": False,
        "splits": {},
    }

    for split in requested_splits:
        raw_dataset = _load_one_split(args, split)
        split_info = _validate_raw_split(
            raw_dataset,
            args.domain,
            split,
        )
        valid_dataset = raw_dataset.filter(
            valid_pair,
            num_proc=args.preprocessing_num_workers,
            desc=f"Filtering valid {args.domain}/{split} pairs",
        )
        split_info["effective_examples"] = len(valid_dataset)
        split_info["invalid_examples_removed"] = len(raw_dataset) - len(valid_dataset)
        split_info["mapping_checks"] = _mapping_checks(valid_dataset)

        limit: Optional[int] = (
            args.max_train_samples
            if split == "train"
            else args.max_validation_samples
        )
        if limit is not None:
            valid_dataset = valid_dataset.select(range(min(limit, len(valid_dataset))))
        if not len(valid_dataset):
            raise ValueError(f"No usable pairs in {args.domain}/{split}")
        split_info["examples_used"] = len(valid_dataset)

        def preprocess(example: dict[str, Any]) -> dict[str, list[int]]:
            prompt, chosen, rejected, _ = preference_texts(example)
            chosen_tokens = tokenize_pair_text(
                tokenizer, prompt, chosen, args.seq_length
            )
            rejected_tokens = tokenize_pair_text(
                tokenizer, prompt, rejected, args.seq_length
            )
            return {
                "input_ids_chosen": chosen_tokens["input_ids"],
                "attention_mask_chosen": chosen_tokens["attention_mask"],
                "input_ids_rejected": rejected_tokens["input_ids"],
                "attention_mask_rejected": rejected_tokens["attention_mask"],
            }

        processed = valid_dataset.map(
            preprocess,
            remove_columns=valid_dataset.column_names,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Tokenizing {args.domain}/{split}",
        )
        if set(processed.column_names) != MODEL_COLUMNS:
            raise AssertionError(
                f"Unexpected post-tokenization columns: {processed.column_names}"
            )

        tokenized[split] = processed
        dataset_info["splits"][split] = split_info
        LOGGER.info(
            "%s/%s raw=%d effective=%d used=%d",
            args.domain,
            split,
            split_info["raw_examples"],
            split_info["effective_examples"],
            split_info["examples_used"],
        )

    return tokenized, dataset_info


@dataclass
class PairwiseCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}
        for side in ("chosen", "rejected"):
            side_features = [
                {
                    "input_ids": feature[f"input_ids_{side}"],
                    "attention_mask": feature[f"attention_mask_{side}"],
                }
                for feature in features
            ]
            padded = self.tokenizer.pad(
                side_features,
                padding=True,
                return_tensors="pt",
            )
            batch[f"input_ids_{side}"] = padded["input_ids"]
            batch[f"attention_mask_{side}"] = padded["attention_mask"]

        # Trainer needs a label key to invoke compute_metrics. These values are
        # never passed to the model and are not pointwise reward supervision.
        batch["labels"] = torch.ones(len(features), dtype=torch.long)
        return batch
