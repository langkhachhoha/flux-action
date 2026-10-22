#!/usr/bin/env bash
# End-to-end test of the robot client against fake_robot.py (no hardware). The policy server must be running.
#
#   bash scripts/astribot/test_fake_robot.sh --max-cycles 3                      # dry-run
#   FAKE_EXECUTE=1 bash scripts/astribot/test_fake_robot.sh --prepare-ready-pose --max-cycles 3
#
# The fake robot replays a held-out coffee episode (FAKE_EPISODE, default 376) on an isolated localhost
# domain (FAKE_ROS_DOMAIN_ID, default 77; the real robot uses 25). FAKE_EXECUTE=1 also exercises --execute
# (ready pose, limiter, tracking guard, 250 Hz stream, final hold): the fake robot echoes the commands as its
# joint states. Passing --execute directly is refused, so this script never commands a real robot.
set -euo pipefail
for arg in "$@"; do
    if [[ "$arg" == "--execute" ]]; then
        echo "--execute is not allowed here; use FAKE_EXECUTE=1" >&2
        exit 2
    fi
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -eq 0 ]] && set -- --max-cycles 3

export ROS_DOMAIN_ID="${FAKE_ROS_DOMAIN_ID:-77}"
export ROS_LOCALHOST_ONLY=1
export ASTRIBOT_QUIET=1
source "$SCRIPT_DIR/ros2_env.sh"
if [[ "$ROS_DOMAIN_ID" == 25 ]]; then
    echo "refusing to run the fake robot on the real robot's domain 25" >&2
    exit 2
fi

FAKE_ARGS=(--episode "${FAKE_EPISODE:-376}")
if [[ "${FAKE_EXECUTE:-0}" == 1 ]]; then
    FAKE_ARGS+=(--follow-commands)
    set -- "$@" --execute --yes
fi
echo "[fake-robot] ROS_DOMAIN_ID=$ROS_DOMAIN_ID ${FAKE_ARGS[*]}"
/usr/bin/python3 "$SCRIPT_DIR/fake_robot.py" "${FAKE_ARGS[@]}" &
FAKE_PID=$!
trap 'kill -INT "$FAKE_PID" 2>/dev/null || true; wait "$FAKE_PID" 2>/dev/null || true' EXIT
sleep "${FAKE_STARTUP_S:-8}"  # decoding and JPEG-encoding the episode takes a few seconds

/usr/bin/python3 "$SCRIPT_DIR/robot_client.py" "$@"
echo "[fake-robot] robot client finished successfully"
