#!/usr/bin/env python3
"""Print a generated portable candidate pool, or save to an explicit output."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pessimism.input_data import synthetic_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        data = synthetic_document(args.groups, args.candidates, args.seed)
        text = json.dumps(data, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(text, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
