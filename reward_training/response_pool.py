"""Generate responses, score them with trained adapters, and export algorithm inputs."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def require_new_output(path):
    if Path(path).exists():
        raise ValueError(f"Output already exists; choose a new output location: {path}")


def validate_pool(pool, *, scored=False):
    if not isinstance(pool, dict):
        raise ValueError("A response pool must be a JSON object")
    if pool.get("split") not in ("validation", "test"):
        raise ValueError("Response pools must use validation or test prompts")
    if not isinstance(pool.get("domain"), str) or not pool["domain"]:
        raise ValueError("A pool must identify its domain")
    if not isinstance(pool.get("prompts"), list) or not pool["prompts"]:
        raise ValueError("A pool must contain prompts")
    seen = set()
    for prompt in pool["prompts"]:
        if not isinstance(prompt.get("prompt"), str) or not prompt["prompt"].strip():
            raise ValueError("Each prompt must contain nonempty text")
        if prompt.get("id") != digest(prompt["prompt"]) or prompt["id"] in seen:
            raise ValueError("Prompt IDs must be unique SHA-256 hashes of prompt text")
        seen.add(prompt["id"])
        if not prompt.get("candidates"):
            raise ValueError("Each prompt must contain responses")
        ids = set()
        for candidate in prompt["candidates"]:
            if not isinstance(candidate.get("id"), str) or not candidate["id"] or candidate["id"] in ids:
                raise ValueError("Candidate IDs must be nonempty and unique within a prompt")
            ids.add(candidate["id"])
            if not isinstance(candidate.get("text"), str):
                raise ValueError("Each candidate must contain response text")
            for role in ("proxy", "judge"):
                key = f"{role}_raw"
                if scored or key in candidate:
                    value = candidate.get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"Missing or nonfinite {key}")
    if scored and set(pool.get("scorers", {})) != {"proxy", "judge"}:
        raise ValueError("Scored pools must record both reward-model configurations")
    return pool


def select_prompts(rows, domain, split, count, seed):
    from shp_rm.data import valid_pair

    unique = set()
    for row in rows:
        if row.get("domain") != f"{domain}_{split}":
            raise ValueError(f"Expected domain field {domain}_{split}")
        if valid_pair(row):
            unique.add(row["history"])
    ranked = sorted(unique, key=lambda text: digest(f"{seed}\0{domain}\0{split}\0{text}"))
    if len(ranked) < count:
        raise ValueError(f"Requested {count} prompts, but only {len(ranked)} unique valid histories exist")
    return ranked[:count]


def generate(args):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, set_seed
    from shp_rm.modeling import input_device, load_tokenizer

    require_new_output(args.output)
    if Path(args.domain).name != args.domain or args.domain in (".", ".."):
        raise ValueError("Domain must be a single directory name")
    if args.dtype != "float32" and not torch.cuda.is_available():
        raise ValueError("FP16/BF16 generation requires CUDA; use --dtype float32 for CPU")
    path = args.dataset_path / args.domain / f"{args.split}.json"
    rows = load_dataset("json", data_files=str(path), split="train", cache_dir=args.dataset_cache_dir)
    prompts = select_prompts(rows, args.domain, args.split, args.num_prompts, args.seed)
    tokenizer = load_tokenizer(args.model_path)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    settings = {key: getattr(args, key) for key in (
        "num_prompts", "responses_per_prompt", "batch_size", "seed", "dtype",
        "max_prompt_tokens", "max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty",
    )}
    pool = {"domain": args.domain, "split": args.split, "generation": settings, "prompts": []}
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            )[-args.max_prompt_tokens:]
            inputs = torch.tensor([tokens], dtype=torch.long, device=input_device(model))
            candidates = []
            for start in range(0, args.responses_per_prompt, args.batch_size):
                count = min(args.batch_size, args.responses_per_prompt - start)
                batch_seed = int(digest(f"{args.seed}\0{args.domain}\0{args.split}\0{prompt}\0{start}")[:8], 16)
                set_seed(batch_seed)
                sequences = model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    do_sample=True, num_beams=1, num_return_sequences=count,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    min_new_tokens=1, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, use_cache=True,
                )
                texts = tokenizer.batch_decode(sequences[:, inputs.shape[1]:], skip_special_tokens=True)
                candidates.extend({"id": f"response_{start + j:05d}", "text": text}
                                  for j, text in enumerate(texts))
            pool["prompts"].append({"id": digest(prompt), "prompt": prompt, "candidates": candidates})
            print(f"Generated {args.domain}/{args.split}: prompt {index + 1}/{len(prompts)}", flush=True)
    write_json(args.output, validate_pool(pool))


def score(args):
    import torch
    from shp_rm.chat_format import tokenize_pair_text
    from shp_rm.modeling import extract_scalar_rewards, input_device, load_model_and_tokenizer

    require_new_output(args.output)
    pool = validate_pool(read_json(args.input))
    if args.role in pool.get("scorers", {}):
        raise ValueError(f"This pool already has {args.role} scores")
    saved = read_json(args.adapter_path / "training_arguments.json")
    if saved["domain"] != pool["domain"]:
        raise ValueError("The reward-model domain does not match the response pool")
    if (saved["load_in_4bit"] or saved["fp16"] or saved["bf16"]) and not torch.cuda.is_available():
        raise ValueError("This adapter's saved precision settings require CUDA")
    loading = SimpleNamespace(**{**saved, "model_path": args.model_path})
    model, tokenizer, _ = load_model_and_tokenizer(loading, str(args.adapter_path), training=False)
    if not saved["load_in_4bit"]:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if saved["bf16"] else torch.float16 if saved["fp16"] else None
    context = torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()
    with torch.inference_mode(), context:
        for prompt in pool["prompts"]:
            candidates = prompt["candidates"]
            for start in range(0, len(candidates), args.batch_size):
                batch = candidates[start:start + args.batch_size]
                encoded = [tokenize_pair_text(tokenizer, prompt["prompt"], row["text"], saved["seq_length"])
                           for row in batch]
                inputs = tokenizer.pad(encoded, padding=True, return_tensors="pt")
                inputs = {key: tensor.to(input_device(model)) for key, tensor in inputs.items()}
                values = extract_scalar_rewards(model(**inputs)).float().cpu().tolist()
                for row, value in zip(batch, values):
                    row[f"{args.role}_raw"] = value
    pool.setdefault("scorers", {})[args.role] = {
        "adapter_sha256": file_digest(args.adapter_path / "adapter_model.safetensors"),
        **{key: saved[key] for key in ("seq_length", "load_in_4bit", "fp16", "bf16")},
    }
    write_json(args.output, validate_pool(pool))


def score_array(pool, role):
    return np.asarray([row[f"{role}_raw"] for prompt in pool["prompts"]
                       for row in prompt["candidates"]], dtype=np.float64)


def normalize(values, scale):
    low, high = float(scale["min"]), float(scale["max"])
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("Calibration needs finite, nonconstant scores")
    return np.clip((np.asarray(values) - low) / (high - low), 0.0, 1.0)


def calibrate(args):
    require_new_output(args.output)
    groups = {}
    for path in args.inputs:
        pool = validate_pool(read_json(path), scored=True)
        if pool["split"] != "validation":
            raise ValueError("Calibration must use validation pools, never test pools")
        domain = pool["domain"]
        if domain in groups:
            raise ValueError(f"Duplicate calibration domain: {domain}")
        scales, normalized = {}, {}
        for role in ("proxy", "judge"):
            values = score_array(pool, role)
            scales[role] = {"min": float(values.min()), "max": float(values.max())}
            normalized[role] = normalize(values, scales[role])
        groups[domain] = {
            "scales": scales, "scorers": pool["scorers"],
            "error_bound": float(np.sqrt(np.mean((normalized["proxy"] - normalized["judge"]) ** 2))),
            "prompt_ids": [prompt["id"] for prompt in pool["prompts"]],
            "num_candidates": len(normalized["proxy"]),
        }
    write_json(args.output, {
        "r_max": 1.0, "source_split": "validation", "normalization": "per_domain_per_scorer_minmax_clip",
        "error_estimator": "validation_rmse", "groups": groups,
    })


def export(args):
    # Import only the CPU algorithm schema; generation/scoring dependencies are optional here.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pessimism.input_data import parse_pool

    require_new_output(args.output_dir)
    calibration = read_json(args.calibration)
    if calibration.get("source_split") != "validation":
        raise ValueError("Expected calibration fitted on validation pools")
    pools, seen, prompt_count = [], set(), None
    for path in args.inputs:
        pool = validate_pool(read_json(path), scored=True)
        domain = pool["domain"]
        if pool["split"] != "test" or domain in seen:
            raise ValueError("Export requires one test pool per domain")
        seen.add(domain)
        group = calibration["groups"][domain]
        if pool["scorers"] != group["scorers"]:
            raise ValueError("Test and validation pools must use the same reward models and settings")
        if set(group["prompt_ids"]) & {prompt["id"] for prompt in pool["prompts"]}:
            raise ValueError("Calibration and test pools contain overlapping prompt texts")
        if prompt_count is not None and prompt_count != len(pool["prompts"]):
            raise ValueError("All domains must supply the same number of test prompts")
        prompt_count = len(pool["prompts"])
        pools.append(pool)
    documents, manifest = [], []
    for index in range(prompt_count):
        groups, ids = [], {}
        for pool in pools:
            domain = pool["domain"]
            prompt = pool["prompts"][index]
            calibration_group = calibration["groups"][domain]
            values = {role: normalize([row[f"{role}_raw"] for row in prompt["candidates"]],
                                      calibration_group["scales"][role])
                      for role in ("proxy", "judge")}
            groups.append({
                "id": domain, "error_bound": calibration_group["error_bound"],
                "candidates": [{"id": row["id"], "proxy_score": float(values["proxy"][j]),
                                "judge_score": float(values["judge"][j])}
                               for j, row in enumerate(prompt["candidates"])],
            })
            ids[domain] = prompt["id"]
        document = {"r_max": 1.0, "groups": groups}
        parse_pool(document)
        filename = f"pool_{index:05d}.json"
        documents.append((filename, document))
        manifest.append({"file": filename, "prompt_ids": ids})
    for filename, document in documents:
        write_json(args.output_dir / filename, document)
    write_json(args.output_dir / "manifest.json", {
        "pools": manifest, "calibration": calibration,
        "grouping": "one test prompt per domain at each index; budget shared across domains",
    })


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Generate responses from local SHP prompt splits")
    gen.add_argument("--model_path", required=True)
    gen.add_argument("--dataset_path", type=Path, required=True)
    gen.add_argument("--dataset_cache_dir", default=None)
    gen.add_argument("--domain", required=True)
    gen.add_argument("--split", choices=("validation", "test"), required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--num_prompts", type=positive_int, default=50)
    gen.add_argument("--responses_per_prompt", type=positive_int, default=256)
    gen.add_argument("--batch_size", type=positive_int, default=8)
    gen.add_argument("--seed", type=int, default=2026)
    gen.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    gen.add_argument("--max_prompt_tokens", type=positive_int, default=1024)
    gen.add_argument("--max_new_tokens", type=positive_int, default=512)
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top_p", type=float, default=0.8)
    gen.add_argument("--top_k", type=int, default=20)
    gen.add_argument("--repetition_penalty", type=float, default=1.05)
    gen.set_defaults(func=generate)
    scorer = commands.add_parser("score", help="Apply a trained proxy or judge adapter")
    scorer.add_argument("--input", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    scorer.add_argument("--role", choices=("proxy", "judge"), required=True)
    scorer.add_argument("--model_path", required=True)
    scorer.add_argument("--adapter_path", type=Path, required=True)
    scorer.add_argument("--batch_size", type=positive_int, default=8)
    scorer.set_defaults(func=score)
    cal = commands.add_parser("calibrate", help="Fit scales and empirical RMSE on validation pools")
    cal.add_argument("--inputs", type=Path, nargs="+", required=True)
    cal.add_argument("--output", type=Path, required=True)
    cal.set_defaults(func=calibrate)
    exp = commands.add_parser("export", help="Convert test pools into the allocation algorithm schema")
    exp.add_argument("--inputs", type=Path, nargs="+", required=True)
    exp.add_argument("--calibration", type=Path, required=True)
    exp.add_argument("--output_dir", type=Path, required=True)
    exp.set_defaults(func=export)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            if args.seed < 0 or not math.isfinite(args.temperature) or args.temperature <= 0:
                raise ValueError("seed must be nonnegative and temperature finite and positive")
            if not 0 < args.top_p <= 1 or args.top_k < 0 or not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0:
                raise ValueError("Invalid sampling settings")
        args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
