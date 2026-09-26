#!/usr/bin/env bash
# Local model/data paths are supplied by the caller. Run one stage or all stages.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1
PYTHON_BIN="${PYTHON_BIN:-python3}"
stage="${1:-help}"

if [[ "$stage" == help || "$stage" == --help || "$stage" == -h ]]; then
  cat <<'HELP'
Usage: bash scripts/run_pipeline.sh {train|generate|score|prepare|run|all}

Set RUN_ROOT to a new output directory for the full pipeline.
train:    also requires MODEL_3B, MODEL_7B, DATA_ROOT.
generate: also requires MODEL_3B, DATA_ROOT.
score:    also requires MODEL_3B, MODEL_7B.
prepare:  fit validation calibration and export test algorithm inputs.
run:      run all six algorithms on each exported pool.

Optional: PYTHON_BIN, DOMAINS (space separated), SEED (2026), BUDGET (255),
TEST_PROMPTS (50), TEST_RESPONSES (256), CALIBRATION_PROMPTS (20),
CALIBRATION_RESPONSES (32), GENERATION_BATCH_SIZE (8), SCORE_BATCH_SIZE (8).
HELP
  exit 0
fi
case "$stage" in train|generate|score|prepare|run|all) ;; *) echo "Unknown stage: $stage" >&2; exit 2 ;; esac
: "${RUN_ROOT:?Set RUN_ROOT to your output directory}"
read -r -a domains <<< "${DOMAINS:-askacademia askculinary askengineers asksciencefiction changemyview}"
SEED="${SEED:-2026}"
pool_cli=("$PYTHON_BIN" reward_training/response_pool.py)

train_models() {
  : "${MODEL_3B:?Set MODEL_3B to your local Qwen2.5-3B-Instruct directory}"
  : "${MODEL_7B:?Set MODEL_7B to your local Qwen2.5-7B-Instruct directory}"
  : "${DATA_ROOT:?Set DATA_ROOT to your local SHP dataset root}"
  local domain
  for domain in "${domains[@]}"; do
    common=(--dataset_path "$DATA_ROOT" --dataset_cache_dir "$RUN_ROOT/cache/datasets" --domain "$domain"
      --learning_rate 5e-6 --num_epochs 2 --seq_length 1024
      --weight_decay 0.05 --lr_scheduler_type cosine --warmup_ratio 0.03
      --lora_r 64 --lora_alpha 32 --lora_dropout 0.1
      --lora_target_modules q_proj k_proj v_proj o_proj
      --load_in_4bit --fp16 --no-bf16 --no-tf32 --gradient_checkpointing
      --eval_steps 300 --save_steps 100 --save_limit 2 --seed "$SEED")
    "$PYTHON_BIN" reward_training/train_rm.py "${common[@]}" \
      --model_path "$MODEL_3B" --output_dir "$RUN_ROOT/rm_3b/$domain" \
      --batch_size 4 --eval_batch_size 4 --gradient_accumulation_steps 4
    "$PYTHON_BIN" reward_training/train_rm.py "${common[@]}" \
      --model_path "$MODEL_7B" --output_dir "$RUN_ROOT/judge_7b/$domain" \
      --batch_size 2 --eval_batch_size 2 --gradient_accumulation_steps 8
  done
}

generate_pools() {
  : "${MODEL_3B:?Set MODEL_3B to your local Qwen2.5-3B-Instruct directory}"
  : "${DATA_ROOT:?Set DATA_ROOT to your local SHP dataset root}"
  local domain split prompts responses
  for domain in "${domains[@]}"; do
    for split in validation test; do
      if [[ "$split" == validation ]]; then
        prompts="${CALIBRATION_PROMPTS:-20}"; responses="${CALIBRATION_RESPONSES:-32}"
      else
        prompts="${TEST_PROMPTS:-50}"; responses="${TEST_RESPONSES:-256}"
      fi
      "${pool_cli[@]}" generate --model_path "$MODEL_3B" --dataset_path "$DATA_ROOT" \
        --dataset_cache_dir "$RUN_ROOT/cache/datasets" \
        --domain "$domain" --split "$split" --num_prompts "$prompts" \
        --responses_per_prompt "$responses" --batch_size "${GENERATION_BATCH_SIZE:-8}" \
        --dtype float16 --seed "$SEED" --max_prompt_tokens 1024 --max_new_tokens 512 \
        --temperature 0.7 --top_p 0.8 --top_k 20 --repetition_penalty 1.05 \
        --output "$RUN_ROOT/pools/$domain/$split.raw.json"
    done
  done
}

score_pools() {
  : "${MODEL_3B:?Set MODEL_3B to your local Qwen2.5-3B-Instruct directory}"
  : "${MODEL_7B:?Set MODEL_7B to your local Qwen2.5-7B-Instruct directory}"
  local domain split prefix
  for domain in "${domains[@]}"; do
    for split in validation test; do
      prefix="$RUN_ROOT/pools/$domain/$split"
      "${pool_cli[@]}" score --role proxy --model_path "$MODEL_3B" \
        --adapter_path "$RUN_ROOT/rm_3b/$domain" --batch_size "${SCORE_BATCH_SIZE:-8}" \
        --input "$prefix.raw.json" --output "$prefix.proxy.json"
      "${pool_cli[@]}" score --role judge --model_path "$MODEL_7B" \
        --adapter_path "$RUN_ROOT/judge_7b/$domain" --batch_size "${SCORE_BATCH_SIZE:-8}" \
        --input "$prefix.proxy.json" --output "$prefix.scored.json"
    done
  done
}

prepare_inputs() {
  local domain
  local validation=() test=()
  for domain in "${domains[@]}"; do
    validation+=("$RUN_ROOT/pools/$domain/validation.scored.json")
    test+=("$RUN_ROOT/pools/$domain/test.scored.json")
  done
  "${pool_cli[@]}" calibrate --inputs "${validation[@]}" --output "$RUN_ROOT/calibration.json"
  "${pool_cli[@]}" export --inputs "${test[@]}" --calibration "$RUN_ROOT/calibration.json" \
    --output_dir "$RUN_ROOT/algorithm_inputs"
}

run_algorithms() {
  local pool name
  for pool in "$RUN_ROOT"/algorithm_inputs/pool_*.json; do
    [[ -f "$pool" ]] || { echo "No algorithm inputs; run the prepare stage first." >&2; return 1; }
    name="$(basename "$pool")"
    "$PYTHON_BIN" -m pessimism.run --input "$pool" --budget "${BUDGET:-255}" \
      --seed "$SEED" --initial-count 2 --alpha 1 --fixed-beta 0.1 \
      --beta-grid 0.01,0.03,0.1,0.3,1,3 \
      --output "$RUN_ROOT/results/seed_$SEED/$name"
  done
}

case "$stage" in
  train) train_models ;;
  generate) generate_pools ;;
  score) score_pools ;;
  prepare) prepare_inputs ;;
  run) run_algorithms ;;
  all) train_models; generate_pools; score_pools; prepare_inputs; run_algorithms ;;
esac
