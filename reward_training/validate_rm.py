#!/usr/bin/env python3
"""Validation-only wrapper: pass explicit model, data, output and adapter locations."""
import sys
from train_rm import main

if __name__ == "__main__":
    if "--mode" in sys.argv[1:]:
        raise SystemExit("This entry point is validation-only; omit --mode")
    main([*sys.argv[1:], "--mode", "eval"])
