#!/usr/bin/env bash
# Move Astribot S1 smoothly to the HOME (rest) pose, or the ready pose (ROS 2 Humble, system Python 3.10).
# Stop run_robot.sh first.
#
#   bash scripts/astribot/move_home.sh --dry-run        # motion plan only, nothing moves
#   bash scripts/astribot/move_home.sh                  # asks for 's' before moving (4 s)
#   bash scripts/astribot/move_home.sh --duration 8     # slower
#   bash scripts/astribot/move_home.sh --pose ready
#
# Options: python3 scripts/astribot/move_home.py --help.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ros2_env.sh"
exec "${ROBOT_PYTHON:-/usr/bin/python3}" "$SCRIPT_DIR/move_home.py" "$@"
