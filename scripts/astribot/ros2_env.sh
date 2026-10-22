#!/usr/bin/env bash
# Source (do not run) before any ROS 2 program of scripts/astribot: ROS 2 Humble, the DDS settings of the
# robot bridge and the astribot_msgs overlay. Override with ROS_SETUP, ASTRIBOT_WS, ROS_DOMAIN_ID.
_ASTRIBOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ASTRIBOT_NOUNSET=0
[[ $- == *u* ]] && _ASTRIBOT_NOUNSET=1
set +u  # the ROS setup scripts read unset variables
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ -z "${ASTRIBOT_WS:-}" ]]; then
    for _candidate in "$_ASTRIBOT_DIR/../../../vm_astribot" "$HOME/Documents/vm_astribot" \
        "$HOME/workspace/projects/vm_astribot"; do
        if [[ -f "$_candidate/install/setup.bash" ]]; then
            ASTRIBOT_WS="$(cd "$_candidate" && pwd)"
            break
        fi
    done
fi
if [[ ! -f "${ASTRIBOT_WS:-}/install/setup.bash" ]]; then
    echo "astribot_msgs overlay not found; build vm_astribot or set ASTRIBOT_WS" >&2
    return 1
fi
source "$ASTRIBOT_WS/install/setup.bash"
[[ $_ASTRIBOT_NOUNSET == 1 ]] && set -u
# The bridge container is ROS 2 Foxy; Fast-DDS over UDP only avoids its incompatible shared-memory layout.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-25}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$_ASTRIBOT_DIR/fastdds_udp.xml"
unset CYCLONEDDS_URI
if [[ -z "${ASTRIBOT_QUIET:-}" ]]; then
    echo "[ros2_env] ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW=$RMW_IMPLEMENTATION ASTRIBOT_WS=$ASTRIBOT_WS"
fi
unset _ASTRIBOT_DIR _ASTRIBOT_NOUNSET _candidate
