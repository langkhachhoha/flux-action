#!/usr/bin/env bash
# One-time preparation for scripts/astribot/train_astri_coffee.sh:
#   1. Python environment (.venv from uv.lock) + NATTEN (video VAE) + wandb
#   2. FLUX 3 Action base trunk, video VAE and text encoder from Hugging Face (gated: needs a token)
#   3. Index of the LeRobot dataset: manifest, state/action rows, normalization statistics, 20 held-out episodes
#
#   HF_TOKEN=hf_xxx bash scripts/astribot/prepare.sh      (or run `.venv/bin/hf auth login` once before)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
UV=${UV:-"$HOME/.local/bin/uv"}
PY="$PROJECT_ROOT/.venv/bin/python"

DATASET_PATH=${DATASET_PATH:-"/home/hieuhm19/datasets/serving_brewed_coffee_lrb_annotated"}
INDEX_DIR=${INDEX_DIR:-"$PROJECT_ROOT/outputs/astribot/index"}
WEIGHTS_DIR=${WEIGHTS_DIR:-"$PROJECT_ROOT/outputs/weights"}
VAL_EPISODES=${VAL_EPISODES:-20}    # the last N episodes (by index) are held out for validation
CHUNK_SIZE=${CHUNK_SIZE:-42}        # must equal policy.chunk_size in configs/astribot/train.json
NATTEN_VERSION=${NATTEN_VERSION:-"0.21.6"}

# ---------------------------------------------------------------- 1. environment
if [ ! -x "$PY" ]; then
    [ -x "$UV" ] || { curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh; }
    "$UV" sync --locked --extra encoders --extra data
fi
TORCH_TAG=$("$PY" -c 'import torch; v = torch.__version__.split("+")[0].split("."); print(f"torch{v[0]}{v[1]}{v[2]}" + "cu" + torch.version.cuda.replace(".", ""))')
if ! "$PY" -c 'import natten' 2>/dev/null; then
    "$UV" pip install --python "$PY" "natten==${NATTEN_VERSION}+${TORCH_TAG}" -f https://whl.natten.org/
fi
if ! "$PY" -c 'import wandb' 2>/dev/null; then
    "$UV" pip install --python "$PY" wandb
fi
"$PY" -c 'import torch, natten, wandb; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "natten", natten.__version__, "wandb", wandb.__version__)'

# ---------------------------------------------------------------- 2. weights
if [ ! -f "$WEIGHTS_DIR/flux-3-action-base.safetensors" ] || [ ! -f "$WEIGHTS_DIR/video_vae.safetensors" ] \
    || [ ! -f "$WEIGHTS_DIR/text_encoder/model.safetensors.index.json" ]; then
    "$PROJECT_ROOT/.venv/bin/hf" download black-forest-labs/flux-3-action-base \
        --include 'flux-3-action-base.safetensors' --include 'video_vae.safetensors' --include 'text_encoder/*' \
        --local-dir "$WEIGHTS_DIR"
fi
ls -la "$WEIGHTS_DIR"

# ---------------------------------------------------------------- 3. index
if [ ! -f "$INDEX_DIR/manifest.json" ]; then
    "$PROJECT_ROOT/.venv/bin/flux-action" index-lerobot \
        --source-root "$DATASET_PATH" --output-dir "$INDEX_DIR" \
        --camera head=observation.images.cam_head \
        --camera left_wrist=observation.images.cam_left_wrist \
        --camera right_wrist=observation.images.cam_right_wrist \
        --action-parameterization joint_delta --absolute-action-dims 14,15 \
        --chunk-size "$CHUNK_SIZE" --val-episodes "$VAL_EPISODES" \
        --dataset-id "$(basename "$DATASET_PATH")"
fi
"$PY" - "$INDEX_DIR" <<'EOF'
import json, sys
index = sys.argv[1]
manifest = json.load(open(f"{index}/manifest.json"))
stats = json.load(open(f"{index}/statistics.json"))
print("episodes:", manifest["counts"])
print("cameras:", manifest["camera_order"], manifest["camera_hw"])
print("action:", stats["action"]["parameterization"], "absolute dims", stats["action"]["absolute_dims"])
EOF
echo "Preparation done. Next: bash scripts/astribot/train_astri_coffee.sh"
