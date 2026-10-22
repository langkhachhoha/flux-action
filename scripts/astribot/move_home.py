#!/usr/bin/env python3
"""Move Astribot S1 smoothly to the HOME (rest) or the ready pose (vmo_diffusion_policy_vla's move_to_home.py).

    bash scripts/astribot/move_home.sh --dry-run          # print the motion plan; nothing is published
    bash scripts/astribot/move_home.sh                    # asks for 's', then a 4 s cosine motion
    bash scripts/astribot/move_home.sh --duration 8       # slower, for changes above ~30 degrees
    bash scripts/astribot/move_home.sh --pose ready

Moves torso, both arms, both grippers (opened: release any object first) and head from their measured
positions; stop robot_client.py first, two programs must never command the robot at once. The running client
has the same motion on its ``h`` key.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import yaml
from robot_runtime import HOME_POSE, READY_POSE, pose_plan

SCRIPT_DIR = Path(__file__).resolve().parent
POSES = {"home": HOME_POSE, "ready": READY_POSE}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pose", choices=sorted(POSES), default="home")
    p.add_argument("--duration", "-d", type=float, default=4.0, help="motion time in seconds")
    p.add_argument("--rate", type=float, default=100.0, help="command rate (Hz)")
    p.add_argument("--dry-run", action="store_true", help="print the plan without publishing anything")
    p.add_argument("--yes", "-y", action="store_true", help="move without the 's' confirmation")
    p.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the joint states")
    p.add_argument("--max-sensor-age", type=float, default=1.5, help="stop when a joint state is older (s)")
    p.add_argument("--topics", default=str(SCRIPT_DIR / "ros2_astribot.yaml"), help="ROS 2 topic config")
    args = p.parse_args()
    if args.duration <= 0 or args.rate <= 0 or args.max_sensor_age <= 0:
        p.error("--duration, --rate and --max-sensor-age must be positive")
    return args


def main() -> None:
    args = parse_args()
    topics = yaml.safe_load(Path(args.topics).read_text())
    target = POSES[args.pose]

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from robot_ros2 import AstribotNode

    rclpy.init()
    node = AstribotNode(topics, (), node_name="flux_action_astribot_pose")
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, name="astribot-pose", daemon=True)
    spin.start()
    moving = False
    try:
        print("Waiting for the joint states of all six body parts ...", flush=True)
        deadline = time.monotonic() + args.timeout
        while not node.readiness()[0]:
            if time.monotonic() > deadline:
                raise TimeoutError(f"joint states not ready after {args.timeout:.0f}s: {node.readiness()[1]}")
            time.sleep(0.05)
        current = {part: node.part_positions(part) for part in target}
        for line in pose_plan(current, target, args.pose):
            print(line, flush=True)
        if args.dry_run:
            print("Dry-run: nothing is published.", flush=True)
            return
        if not args.yes:
            prompt = f"Enter 's' to move to the {args.pose.upper()} pose ({args.duration:.1f}s): "
            try:
                answer = input(prompt).strip().lower()
            except EOFError:
                answer = ""
            if answer != "s":
                print("Aborted; nothing is published.", flush=True)
                return
        moving = True
        node.move_to_pose(
            target, duration_s=args.duration, rate_hz=args.rate, max_state_age=args.max_sensor_age
        )
        time.sleep(0.3)  # the 250 Hz stream keeps holding the pose while the robot settles
        print(f"{args.pose.upper()} pose reached.", flush=True)
    except KeyboardInterrupt:
        print("Interrupted; holding the last command.", flush=True)
    finally:
        if moving:
            try:
                node.publish_hold()
            except Exception as error:
                print(f"WARNING: could not publish the final hold: {error}", flush=True)
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin.join(timeout=2.0)


if __name__ == "__main__":
    main()
