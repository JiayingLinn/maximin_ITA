# Portable allocation runner

The runner needs Python 3.10+ and NumPy. The numerical core and baseline
functions are copied from the research implementation; the portable JSON
adapter and CLI replace the environment-specific model/data integration.

## Methods

| CLI method | Allocation | Within-group selection |
| --- | --- | --- |
| `uniform_argmax` (Uniform-BoN) | Equal integer split, remainder in input group order | Largest proxy score; earliest tie |
| `greedy_argmax` | One initial response per group, then one response for the group with the smallest current maximum | Largest proxy score |
| `uniform_itp_auto` | Equal integer split | ITP; maximize certificate over currently certified beta grid |
| `uniform_itp_fixed` | Equal integer split | ITP at `--fixed-beta` |
| `lcb_greedy_auto` (BGP-auto) | Minimum certificate, doubling batches | ITP with certified automatic beta |
| `lcb_greedy_fixed` (BGP-fixed) | Minimum certificate, doubling batches | ITP at `--fixed-beta` |

The budget is global and includes initialization. Every method starts at the
beginning of exactly the same supplied candidate order. A fresh generator
uses `--seed` for each method, so adding/reordering requested methods does not
change results for another method. Greedy-argmax always starts with one draw;
`--initial-count` sets the minimum initial count for LCB-Greedy. The original
schedule rejects an initial count below its smallest certifiable count.

## Commands

From the release folder, after installing its NumPy dependency:

```bash
python -B -m pessimism.run --synthetic --budget 60 --seed 0
bash scripts/run_algorithms.sh
python -B -m pessimism.run --synthetic --methods lcb_greedy_auto lcb_greedy_fixed \
  --budget 60 --initial-count 2 --alpha 0.5 --fixed-beta 0.1 \
  --beta-grid 0.01,0.03,0.1,0.3,1,3
python -B scripts/make_synthetic_pool.py --output /tmp/synthetic_pool.json
python -B -m pessimism.run --input /tmp/synthetic_pool.json --budget 60 \
  --output /tmp/allocation_results.json
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p 'test_*.py' -v
```

`PYTHON_BIN=python` (or `PYTHON=python`) can be set when invoking the shell script. With no arguments
the shell script uses synthetic data; otherwise it passes arguments through
and an explicit `--synthetic` or `--input` is required. There is no default
model, Judge, dataset, or input path. Output is JSON on stdout unless an
explicit `--output` is supplied. Demo and tests require no network/GPU/data.

## Caller-supplied JSON

```json
{
  "r_max": 1.0,
  "groups": [
    {
      "id": "group_0",
      "error_bound": 0.1,
      "candidates": [
        {"id": "candidate_0", "proxy_score": 0.3, "judge_score": 0.4},
        {"id": "candidate_1", "proxy_score": 0.6, "judge_score": 0.5}
      ]
    }
  ]
}
```

This two-candidate schema example can run with `--budget 2`. Real runs must
supply enough ordered candidates for every method. Group IDs must be unique;
candidate IDs must be unique within each group. Scores must already be
finite and on a common `[0, r_max]` scale. The runner never normalizes or clips
caller data. Group `error_bound` is a nonnegative externally provided L2 RM
error bound. The runner never estimates it from Judge scores. Judge scores
are optional, but must occur on every candidate or none; they only enter
post-selection evaluation. Candidate IDs and proxy scores alone are exposed
to the allocation stream. Unknown JSON fields are rejected.

## Output and interpretation

Each method reports group counts, selected candidate ID/index, selection
probabilities, selected proxy/Judge scores, and expected proxy/Judge scores.
An expected score is the final probability-weighted score over that group's
revealed prefix, conditional on the realized allocation. It is not an
expectation over allocation randomness or a population reward estimate.
Min/mean/max and gap (`max - min`) are aggregated across groups for both
expected and sampled scores. ITP methods additionally report beta,
certificate, certificate minus slack, effective error, and the schedule.
LCB methods record the allocation history.

`--alpha` / `--error-scale` multiplies the L2 error before both linear and
quadratic penalty terms. Default 1 matches the original bound.
`--slack` sets the original certification slack explicitly. When omitted,
the runner derives it from `--initial-count` and the largest beta using
`delta_total`; the exact value and its source are recorded. Such large slacks
are often vacuous at demo sample sizes. Neither a stored pool nor a numeric
certificate establishes the distributional and error-bound assumptions.

Fixed beta bypasses the certified beta set and is an empirical comparison,
although the original certification schedule is still needed for initial
counts. Rescaled error bounds need separate justification. Optional
`--allocation-batch-size` and `--index-error-scale` expose the existing
constant-batch and decoupled-index empirical ablations. The core also retains
other research knobs through its Python API; they are not needed by the
basic runner. The default keeps doubling and coupled error scaling.

`--capacity-mode capped` (default) skips exhausted groups and truncates batches
for adaptive methods, recording affected steps. This finite-pool adaptation
differs from unlimited sampling. `uncapped` disables those algorithm-level
caps, and requesting unavailable candidates fails without reusing data.
Uniform methods always use the exact equal split; if a group lacks its share,
the runner rejects the input instead of redistributing it. The run spends
exactly the global budget or fails.
