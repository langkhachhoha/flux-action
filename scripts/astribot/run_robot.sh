#!/usr/bin/env bash
# Astribot S1 robot client (ROS 2 Humble, system Python 3.10). Start the policy server first:
#   bash scripts/astribot/serve_policy.sh checkpoint/infer/step-2300-model-bf16
#
#   bash scripts/astribot/run_robot.sh --max-cycles 3                                   # dry-run
#   bash scripts/astribot/run_robot.sh --execute --prepare-ready-pose --max-cycles 3    # moves the robot
#
# Options: python3 scripts/astribot/robot_client.py --help. Logs go to ~/Desktop/astribot_logs/flux_action.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ros2_env.sh"
exec "${ROBOT_PYTHON:-/usr/bin/python3}" "$SCRIPT_DIR/robot_client.py" "$@"
