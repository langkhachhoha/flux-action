#!/usr/bin/env bash
# FLUX 3 Action full fine-tuning on the Astribot S1 "serving brewed coffee" LeRobot dataset.
#
#   bash scripts/astribot/train_astri_coffee.sh                 # 1 GPU (GPU 0), foreground
#   DETACH=1 bash scripts/astribot/train_astri_coffee.sh        # survives SSH/network loss (setsid + nohup)
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,3 bash scripts/astribot/train_astri_coffee.sh   # 2 GPUs (FSDP2)
#
# Every hyperparameter below can be overridden from the environment, e.g. LR=5e-5 MAX_STEPS=5000 bash ...
# Running the same command again resumes from the newest complete checkpoint in OUTPUT_DIR (same W&B run).
# Stop cleanly (checkpoint first, no auto-restart):   touch "$OUTPUT_DIR/STOP"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
VENV="$PROJECT_ROOT/.venv"
TORCHRUN="$VENV/bin/torchrun"

# ---------------------------------------------------------------- data, weights, output
DATASET_PATH=${DATASET_PATH:-"/home/hieuhm19/datasets/serving_brewed_coffee_lrb_annotated"}
INDEX_DIR=${INDEX_DIR:-"$PROJECT_ROOT/outputs/astribot/index"}
WEIGHTS_DIR=${WEIGHTS_DIR:-"$PROJECT_ROOT/outputs/weights"}
CONFIG=${CONFIG:-"$PROJECT_ROOT/configs/astribot/train.json"}
RUN_NAME=${RUN_NAME:-"flux3_astri_coffee_full_ft"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs/astribot/$RUN_NAME"}

# ---------------------------------------------------------------- GPUs
# GPU 0 is the only idle card at the time of writing; the others run other users' jobs.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c -v '^$')
fi
# FSDP2 CPU offload of masters/gradients/AdamW/EMAs: required to fit the 7B full fine-tune (1 or 2 GPUs).
CPU_OFFLOAD=${CPU_OFFLOAD:-1}
# The CPU runs the optimizer step under offload; torchrun would otherwise pin every rank to 1 thread.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( $(nproc) / (2 * NUM_GPUS) > 4 ? $(nproc) / (2 * NUM_GPUS) : 4 ))}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

# ---------------------------------------------------------------- batch and schedule (optimizer updates)
WINDOWS_PER_GPU=${WINDOWS_PER_GPU:-8}   # windows per micro-batch per GPU (~62 GB peak at 8; lower it on OOM)
GLOBAL_BATCH=${GLOBAL_BATCH:-32}        # windows per optimizer update = NUM_GPUS x WINDOWS_PER_GPU x accumulation
MAX_STEPS=${MAX_STEPS:-3000}            # optimizer updates
LR=${LR:-3e-5}                          # peak trunk learning rate
HEADS_LR_MULT=${HEADS_LR_MULT:-5}       # fresh 16-D action/conditioning heads train at LR x this
FROZEN_STEPS=${FROZEN_STEPS:-200}       # trunk LR is 0 for these updates (heads only) ...
TRUNK_WARMUP_STEPS=${TRUNK_WARMUP_STEPS:-600}  # ... then warms up linearly over these
HEADS_WARMUP_STEPS=${HEADS_WARMUP_STEPS:-200}
COOLDOWN_STEPS=${COOLDOWN_STEPS:-500}   # linear decay to 0 over the last updates (0 = constant LR)
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
SEED=${SEED:-42}

# ---------------------------------------------------------------- data loading
NUM_WORKERS=${NUM_WORKERS:-6}           # decoding workers per GPU
VISITS_PER_EPOCH=${VISITS_PER_EPOCH:-8} # random windows drawn per episode per epoch

# ---------------------------------------------------------------- checkpoints, logging, evaluation
SAVE_STEPS=${SAVE_STEPS:-100}           # a checkpoint is ~140 GB (fp32 model + AdamW + 2 EMAs)
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-1} # newest complete checkpoints kept on disk (resume uses the newest)
KEEP_BEST=${KEEP_BEST:-loss}            # also keep the weights with the lowest val/<metric> in OUTPUT_DIR/best
                                        # (no optimizer state); "null" turns it off
LOG_STEPS=${LOG_STEPS:-1}
EVAL_STEPS=${EVAL_STEPS:-100}           # held-out loss (+ sampled-action error) every N updates
VAL_WINDOWS_PER_EPISODE=${VAL_WINDOWS_PER_EPISODE:-2}
VAL_SAMPLE_WINDOWS=${VAL_SAMPLE_WINDOWS:-8}

# ---------------------------------------------------------------- Weights & Biases
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"astribot_making_coffee"}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"$RUN_NAME"}
# online: a separate uploader process (wandb_sync.py) streams metrics.jsonl to W&B and fills any gap after a
#         network/proxy outage by itself; training never talks to W&B, so the office gateway cannot stall it.
# offline: the trainer logs to <output_dir>/wandb only; upload later with `wandb sync`.
WANDB_MODE=${WANDB_MODE:-"online"}

EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}    # extra "key=value" pairs, e.g. "policy.augment=true policy.caption_dropout=0.2"
MAX_RESTARTS=${MAX_RESTARTS:-20}        # automatic resumes after a crash
RESTART_DELAY=${RESTART_DELAY:-60}

# ---------------------------------------------------------------- checks
die() { echo "ERROR: $*" >&2; exit 1; }
[ -x "$TORCHRUN" ] || die "missing $VENV; run scripts/astribot/prepare.sh first"
[ -f "$INDEX_DIR/manifest.json" ] || die "missing index $INDEX_DIR; run scripts/astribot/prepare.sh first"
for f in flux-3-action-base.safetensors video_vae.safetensors text_encoder/config.json; do
    [ -e "$WEIGHTS_DIR/$f" ] || die "missing $WEIGHTS_DIR/$f; run scripts/astribot/prepare.sh first"
done
PER_UPDATE=$(( NUM_GPUS * WINDOWS_PER_GPU ))
[ $(( GLOBAL_BATCH % PER_UPDATE )) -eq 0 ] || die "GLOBAL_BATCH $GLOBAL_BATCH is not a multiple of NUM_GPUS x WINDOWS_PER_GPU = $PER_UPDATE"
GRAD_ACCUM=$(( GLOBAL_BATCH / PER_UPDATE ))
if [ "$COOLDOWN_STEPS" -gt 0 ]; then
    COOLDOWN_START=$(( MAX_STEPS - COOLDOWN_STEPS ))
    [ "$COOLDOWN_START" -ge 0 ] || die "COOLDOWN_STEPS exceeds MAX_STEPS"
else
    COOLDOWN_START=null
fi
bool() { [ "$1" = "1" ] && echo true || echo false; }

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"

# ---------------------------------------------------------------- detach (survive SSH / network loss)
if [ "${DETACH:-0}" = "1" ] && [ -z "${FLUX_DETACHED:-}" ]; then
    FLUX_DETACHED=1 NUM_GPUS=$NUM_GPUS setsid nohup bash "${BASH_SOURCE[0]}" "$@" >> "$LOG_FILE" 2>&1 < /dev/null &
    echo "Training detached (launcher PID $!). It keeps running if the SSH session or network drops."
    echo "  follow the log:   tail -f $LOG_FILE"
    echo "  stop cleanly:     touch $OUTPUT_DIR/STOP"
    exit 0
fi

OVERRIDES=(
    "source_root=$DATASET_PATH"
    "index_dir=$INDEX_DIR"
    "output_dir=$OUTPUT_DIR"
    "policy.trunk_weights=$WEIGHTS_DIR/flux-3-action-base.safetensors"
    "policy.video_vae_id=$WEIGHTS_DIR/video_vae.safetensors"
    "policy.text_encoder_id=$WEIGHTS_DIR/text_encoder"
    "policy.optimizer_lr=$LR"
    "policy.optimizer_lr_heads_multiplier=$HEADS_LR_MULT"
    "steps=$MAX_STEPS"
    "cooldown_start=$COOLDOWN_START"
    "cooldown_steps=$COOLDOWN_STEPS"
    "frozen_steps=$FROZEN_STEPS"
    "trunk_warmup_steps=$TRUNK_WARMUP_STEPS"
    "heads_warmup_steps=$HEADS_WARMUP_STEPS"
    "weight_decay=$WEIGHT_DECAY"
    "max_grad_norm=$MAX_GRAD_NORM"
    "seed=$SEED"
    "windows_per_rank=$WINDOWS_PER_GPU"
    "grad_accumulation=$GRAD_ACCUM"
    "num_workers=$NUM_WORKERS"
    "visits_per_epoch=$VISITS_PER_EPOCH"
    "checkpoint_every=$SAVE_STEPS"
    "keep_checkpoints=$KEEP_CHECKPOINTS"
    "keep_best=$KEEP_BEST"
    "log_every=$LOG_STEPS"
    "val_every=$EVAL_STEPS"
    "val_windows_per_episode=$VAL_WINDOWS_PER_EPISODE"
    "val_sample_windows=$VAL_SAMPLE_WINDOWS"
    "cpu_offload=$(bool "$CPU_OFFLOAD")"
)
if [ "$USE_WANDB" = "1" ] && [ "$WANDB_MODE" = "offline" ]; then
    OVERRIDES+=("wandb_project=$WANDB_PROJECT" "wandb_run_name=$WANDB_RUN_NAME" "wandb_mode=offline")
    [ -n "$WANDB_ENTITY" ] && OVERRIDES+=("wandb_entity=$WANDB_ENTITY")
else
    OVERRIDES+=("wandb_project=null")
fi
# shellcheck disable=SC2206
OVERRIDES+=($EXTRA_OVERRIDES)
ARGS=()
for o in "${OVERRIDES[@]}"; do ARGS+=(--override "$o"); done

echo "=========================================================="
echo "FLUX 3 Action FULL FINE-TUNING - Astribot serving brewed coffee"
echo "Project root:          $PROJECT_ROOT"
echo "Dataset / index:       $DATASET_PATH  /  $INDEX_DIR"
echo "Output directory:      $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES:  $CUDA_VISIBLE_DEVICES  ($NUM_GPUS GPU, CPU offload: $(bool "$CPU_OFFLOAD"), OMP threads: $OMP_NUM_THREADS)"
echo "Batch:                 $GLOBAL_BATCH windows/update = $NUM_GPUS GPU x $WINDOWS_PER_GPU x accumulation $GRAD_ACCUM"
echo "Updates:               $MAX_STEPS (trunk frozen $FROZEN_STEPS, warmup $TRUNK_WARMUP_STEPS, cooldown from $COOLDOWN_START)"
echo "LR trunk / heads:      $LR / x$HEADS_LR_MULT"
echo "Checkpoint / eval:     every $SAVE_STEPS / $EVAL_STEPS updates (keep $KEEP_CHECKPOINTS + best val/$KEEP_BEST)"
echo "W&B:                   $( [ "$USE_WANDB" = "1" ] && echo "$WANDB_PROJECT / $WANDB_RUN_NAME ($WANDB_MODE)" || echo disabled)"
echo "=========================================================="

rm -f "$OUTPUT_DIR/STOP" "$OUTPUT_DIR/STOPPED"
if [ "$USE_WANDB" = "1" ] && [ "$WANDB_MODE" != "offline" ]; then
    # uploads every 2 minutes while this launcher lives, once more after it ends, then exits
    UPLOADER=("$VENV/bin/python" "$SCRIPT_DIR/wandb_sync.py" --output-dir "$OUTPUT_DIR" --project "$WANDB_PROJECT"
        --name "$WANDB_RUN_NAME" --follow-pid $$)
    [ -n "$WANDB_ENTITY" ] && UPLOADER+=(--entity "$WANDB_ENTITY")
    nohup "${UPLOADER[@]}" >> "$OUTPUT_DIR/wandb_sync.log" 2>&1 < /dev/null &
    echo "W&B uploader started (PID $!, log $OUTPUT_DIR/wandb_sync.log)"
fi
trap 'echo "Interrupted; resume later with the same command."; exit 130' INT TERM

attempt=0
while true; do
    set +e
    if [ -n "${FLUX_DETACHED:-}" ]; then
        "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" -m flux_action.cli train --config "$CONFIG" "${ARGS[@]}"
        code=$?
    else
        "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" -m flux_action.cli train --config "$CONFIG" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
        code=${PIPESTATUS[0]}
    fi
    set -e
    if [ "$code" -eq 0 ]; then
        if [ -f "$OUTPUT_DIR/STOPPED" ]; then
            echo "Stopped on request after a checkpoint; run the same command to continue."
        else
            echo "Training finished."
        fi
        break
    fi
    attempt=$(( attempt + 1 ))
    if [ "$attempt" -gt "$MAX_RESTARTS" ]; then
        echo "Training failed (exit $code) $attempt times; giving up." >&2
        exit "$code"
    fi
    echo "Training exited with code $code; resuming from the newest checkpoint in ${RESTART_DELAY}s (restart $attempt/$MAX_RESTARTS)."
    sleep "$RESTART_DELAY"
done
