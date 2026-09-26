# Allocation algorithms

NumPy implementations of uniform and greedy Best-of-N, and uniform and
LCB-Greedy (BGP) allocation with fixed or automatic ITP beta selection.
The budget is shared across groups. Judge scores are used only for evaluation.

The [main pipeline](../README.md) generates and scores real model responses,
then exports the JSON inputs consumed here. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pessimism.run --input "$RUN_ROOT/algorithm_inputs/pool_00000.json" \
  --budget 255 --seed 2026 --output "$RUN_ROOT/results/comparison.json"

# Compare fixed and automatic beta on the same scored pool.
python -m pessimism.run --input "$RUN_ROOT/algorithm_inputs/pool_00000.json" \
  --methods lcb_greedy_auto lcb_greedy_fixed \
  --budget 255 --initial-count 2 --alpha 1 --fixed-beta 0.1 \
  --beta-grid 0.01,0.03,0.1,0.3,1,3 \
  --output "$RUN_ROOT/results/beta_comparison.json"

python -m pessimism.run --help
python -m unittest discover -s tests -v
```

Methods: `uniform_argmax`, `greedy_argmax`, `uniform_itp_fixed`,
`uniform_itp_auto`, `lcb_greedy_fixed`, and `lcb_greedy_auto`.
Omitting `--methods` runs all six. Fixed-beta runs are empirical comparisons.

Input JSON has `r_max` and a `groups` array. Each group contains a unique `id`,
`error_bound`, and ordered `candidates`. Each candidate has a unique `id`,
`proxy_score`, and optional `judge_score`. Scores must lie in `[0, r_max]`;
judge scores must be present for every candidate or absent everywhere.
Error parameters must be supplied before allocation. The pipeline estimates
them using validation pools, which gives empirical RMSE rather than certified
population bounds. Candidate order is preserved and must not be sorted by score.

Results include allocations, selected candidate IDs, selection probabilities,
expected scores over each revealed prefix, and group reward summaries. Look up
the selected candidate IDs in the scored response files for their text.
