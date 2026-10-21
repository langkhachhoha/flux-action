#!/usr/bin/env bash
# Export a training checkpoint to a bf16 inference package and score it on the held-out episodes.
#
#   bash scripts/astribot/export_eval.sh                       # newest checkpoint, EMA sigma_rel 0.10
#   STEP=1500 PROFILE=model bash scripts/astribot/export_eval.sh
#   STEP=best PROFILE=model bash scripts/astribot/export_eval.sh  # lowest val loss (OUTPUT_DIR/best)
#
# PROFILE: model (raw weights) | ema_0p10 | ema_0p05. Compare several before choosing one for the robot.
# The evaluation samples each held-out window's action chunk exactly as inference does (history of 8
# ticks, 4 Euler steps, shift 6.93, guidance 3) and reports MSE in the normalized action space and in
# radians per joint. Offline error is a proxy; it does not establish task success on the robot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
BIN="$PROJECT_ROOT/.venv/bin"

DATASET_PATH=${DATASET_PATH:-"/home/hieuhm19/datasets/serving_brewed_coffee_lrb_annotated"}
INDEX_DIR=${INDEX_DIR:-"$PROJECT_ROOT/outputs/astribot/index"}
RUN_NAME=${RUN_NAME:-"flux3_astri_coffee_full_ft"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs/astribot/$RUN_NAME"}
PROFILE=${PROFILE:-"ema_0p10"}
WINDOWS_PER_EPISODE=${WINDOWS_PER_EPISODE:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

if [ -z "${STEP:-}" ]; then
    CKPT=$(ls -d "$OUTPUT_DIR"/step-* 2>/dev/null | while read -r d; do [ -f "$d/COMPLETE" ] && echo "$d"; done \
        | sort -t- -k2 -n | tail -n1)
    [ -n "$CKPT" ] || { echo "no complete checkpoint in $OUTPUT_DIR" >&2; exit 1; }
    STEP=${CKPT##*step-}
elif [ "$STEP" = "best" ]; then
    CKPT="$OUTPUT_DIR/best"
    STEP="$("$BIN/python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["step"])' "$CKPT/best.json")-best"
else
    CKPT="$OUTPUT_DIR/step-$STEP"
fi
EXPORT_DIR=${EXPORT_DIR:-"$OUTPUT_DIR/export/step-$STEP-$PROFILE-bf16"}

if [ ! -f "$EXPORT_DIR/manifest.json" ]; then
    "$BIN/flux-action" export-checkpoint --checkpoint "$CKPT" --output "$EXPORT_DIR" \
        --profile "$PROFILE" --dtype bfloat16
fi
"$BIN/flux-action" evaluate --checkpoint "$EXPORT_DIR" \
    --source-root "$DATASET_PATH" --index-dir "$INDEX_DIR" \
    --frame-hw 360,640 --windows-per-episode "$WINDOWS_PER_EPISODE" \
    --output "$EXPORT_DIR/eval_val.json"
echo "Export: $EXPORT_DIR   Report: $EXPORT_DIR/eval_val.json"
