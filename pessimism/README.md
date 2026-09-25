# Allocation algorithms

NumPy implementations of uniform and greedy Best-of-N, and uniform and
LCB-Greedy (BGP) allocation with fixed or automatic ITP beta selection.
The budget is shared across groups. Judge scores are used only for evaluation.

Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -B -m pessimism.run --synthetic --budget 60 --seed 0
python -B -m pessimism.run --help

# Compare fixed and automatic beta.
python -B -m pessimism.run --synthetic \
  --methods lcb_greedy_auto lcb_greedy_fixed \
  --budget 60 --initial-count 2 --alpha 1 --fixed-beta 0.1 \
  --beta-grid 0.01,0.03,0.1,0.3,1,3

# Generate synthetic input or run on caller-supplied scores.
python -B scripts/make_synthetic_pool.py --output '<SYNTHETIC_JSON>'
python -B -m pessimism.run --input '<SCORED_POOL_JSON>' \
  --budget 255 --seed 2026 --output '<RESULT_JSON>'

# Algorithm tests.
PYTHON_BIN=python bash scripts/check_release.sh
```

Available methods: `uniform_argmax`, `greedy_argmax`, `uniform_itp_fixed`,
`uniform_itp_auto`, `lcb_greedy_fixed`, and `lcb_greedy_auto`.
Omitting `--methods` runs all six. Fixed-beta runs are empirical comparisons.

Input JSON contains ordered candidates with scores in `[0, r_max]` and an
externally supplied error bound per group. Candidate IDs are unique within
each group. Optional `judge_score` must be present for every candidate or none.
Supply enough candidates for the requested budget; this example uses budget 2:

```json
{
  "r_max": 1.0,
  "groups": [{
    "id": "group_0",
    "error_bound": 0.1,
    "candidates": [
      {"id": "candidate_0", "proxy_score": 0.3, "judge_score": 0.4},
      {"id": "candidate_1", "proxy_score": 0.6, "judge_score": 0.5}
    ]
  }]
}
```

Results include allocations, selected candidates, selection probabilities,
expected scores over each revealed prefix, and group reward summaries.
