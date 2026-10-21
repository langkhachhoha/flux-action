#!/usr/bin/env bash
# Same run on two GPUs (FSDP2 shards the DiT across them; CPU offload stays on).
#
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/astribot/train_astri_coffee_2gpu.sh
#   DETACH=1 CUDA_VISIBLE_DEVICES=0,2 bash scripts/astribot/train_astri_coffee_2gpu.sh
#
# The global batch stays GLOBAL_BATCH (default 32: 2 GPUs x 8 windows x accumulation 2), so a run started on
# one GPU can continue here and back again from its newest checkpoint: the model, optimizer, EMAs and
# schedule continue exactly; only the data order restarts at a fresh epoch because the loader topology
# changed. Check `nvidia-smi` first and pick two idle GPUs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
export NUM_GPUS=2
exec bash "$SCRIPT_DIR/train_astri_coffee.sh" "$@"
