# Allocation algorithms

Greedy-BoN and Batched Greedy Pessimistic (BGP) for adaptive inference budget
allocation, with comparison baselines.

Run from the repository root, using `RUN_ROOT` and the prepared pools from the
[main pipeline](../README.md).

```bash
# Install allocation dependencies.
python -m pip install -r requirements.txt

# Run Greedy-BoN and BGP on one prepared pool.
python -m pessimism.run --input "$RUN_ROOT/algorithm_inputs/pool_00000.json" \
  --methods greedy_argmax lcb_greedy_auto --budget 255 --seed 2026 \
  --output "$RUN_ROOT/results/main_algorithms.json"

# Compare all algorithms and baselines on the same pool.
python -m pessimism.run --input "$RUN_ROOT/algorithm_inputs/pool_00000.json" \
  --budget 255 --seed 2026 --output "$RUN_ROOT/results/comparison.json"

# Show available methods and options.
python -m pessimism.run --help

# Check allocation algorithms and pool preparation.
python -m unittest discover -s tests -v
```
