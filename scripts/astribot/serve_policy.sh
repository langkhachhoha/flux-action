#!/usr/bin/env bash
# Serve an Astribot export to the ROS 2 robot client (scripts/astribot/run_robot.sh).
#
#   bash scripts/astribot/serve_policy.sh [EXPORT_DIR] [policy_server.py options]
#   CUDA_DEVICE=cuda:1 PORT=5556 bash scripts/astribot/serve_policy.sh checkpoint/infer/step-2300-model-bf16
#
# Needs outputs/weights/video_vae.safetensors and outputs/weights/text_encoder (the frozen encoders of
# flux-3-action-base; see scripts/astribot/README.md). Run it in tmux: loading takes about a minute and
# the server must stay up while the robot client runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT=${1:-"$PROJECT_ROOT/checkpoint/infer/step-2300-model-bf16"}
shift || true
CUDA_DEVICE=${CUDA_DEVICE:-"cuda:0"}
PORT=${PORT:-5555}
WEIGHTS_DIR=${WEIGHTS_DIR:-"$PROJECT_ROOT/outputs/weights"}

for path in "$WEIGHTS_DIR/video_vae.safetensors" "$WEIGHTS_DIR/text_encoder/model.safetensors.index.json"; do
    if [ ! -f "$path" ] && [[ " $* " != *" --mock-encoders "* ]]; then
        echo "Missing $path. Download the frozen encoders first (scripts/astribot/README.md)." >&2
        exit 1
    fi
done

exec "$PROJECT_ROOT/.venv/bin/python" "$SCRIPT_DIR/policy_server.py" \
    --checkpoint "$CHECKPOINT" --device "$CUDA_DEVICE" --port "$PORT" \
    --video-vae "$WEIGHTS_DIR/video_vae.safetensors" --text-encoder "$WEIGHTS_DIR/text_encoder" "$@"
