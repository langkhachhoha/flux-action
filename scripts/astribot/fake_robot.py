#!/usr/bin/env python3
"""Pretend to be an Astribot S1: replay one episode of the coffee dataset on the robot's ROS 2 topics.

Publishes the episode's three camera streams (JPEG, 360x640 as stored) and its 16-D state on the topics of
ros2_astribot.yaml, with torso and head at the ready pose, so robot_client.py runs end to end without
hardware. It never commands anything. With ``--follow-commands`` it reports each commanded part's latest
command as that part's state (a robot that tracks perfectly), which exercises the ``--execute`` path; on exit
it prints how many commands each part received and the largest step between two of them.

Refuses ROS_DOMAIN_ID 25 (the real robot's); test_fake_robot.sh runs it on an isolated localhost domain.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from astribot_msgs.msg import RobotJointController, RobotJointState
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from robot_runtime import READY_POSE
from sensor_msgs.msg import CompressedImage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "/nvme/hieuhm19/datasets/serving_brewed_coffee_lrb_annotated"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="LeRobot v3 coffee dataset")
    p.add_argument("--index-dir", default=str(PROJECT_ROOT / "outputs/astribot/index"))
    p.add_argument(
        "--episode", type=int, default=376, help="episode_index (376: held out, starts at the ready pose)"
    )
    p.add_argument("--max-frames", type=int, default=900, help="frames to replay before looping")
    p.add_argument("--topics", default=str(Path(__file__).resolve().parent / "ros2_astribot.yaml"))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--follow-commands", action="store_true", help="echo commands as joint states")
    return p.parse_args()


def load_episode(dataset: str, index_dir: str, episode_index: int, max_frames: int):
    """``(state (T, 16), {camera: [jpeg bytes] * T})`` of one episode, from the index and the dataset videos."""
    manifest = json.loads((Path(index_dir) / "manifest.json").read_text())
    episode = next((e for e in manifest["episodes"] if e["episode_index"] == episode_index), None)
    if episode is None:
        raise SystemExit(f"episode {episode_index} is not in {index_dir}/manifest.json")
    count = min(episode["n_frames"], max_frames)
    rows = np.load(Path(index_dir) / manifest["rows"]["file"], mmap_mode="r")
    state = np.asarray(rows[episode["from_index"] : episode["from_index"] + count, : manifest["state_dim"]])
    jpegs = {}
    for camera, video in episode["videos"].items():
        capture = cv2.VideoCapture(str(Path(dataset) / video["file"]))
        capture.set(cv2.CAP_PROP_POS_FRAMES, video["first_frame"])
        frames = []
        for _ in range(count):
            ok, bgr = capture.read()
            if not ok:
                raise SystemExit(
                    f"cannot decode {video['file']} at frame {video['first_frame'] + len(frames)}"
                )
            frames.append(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
        capture.release()
        jpegs[camera] = frames
    return state, jpegs


class FakeAstribot(Node):
    def __init__(self, state: np.ndarray, jpegs: dict, topics: dict, fps: float, follow: bool) -> None:
        super().__init__("fake_astribot_s1")
        self._state, self._jpegs = state, jpegs
        self._cameras = {c: self.create_publisher(CompressedImage, topics["cameras"][c], 10) for c in jpegs}
        self._states = {p: self.create_publisher(RobotJointState, t, 10) for p, t in topics["states"].items()}
        self._commanded: dict = {}
        self.command_stats: dict = {}
        if follow:
            for part, topic in topics["commands"].items():
                self.create_subscription(
                    RobotJointController, topic, lambda msg, key=part: self._on_command(msg, key), 10
                )
        self._frame = 0
        self.create_timer(1.0 / fps, self._tick)
        self.get_logger().info(f"replaying {len(state)} frames at {fps:.0f}Hz, follow_commands={follow}")

    def _on_command(self, message, part: str) -> None:
        value = np.asarray(message.command, dtype=np.float64)
        count, max_step = self.command_stats.get(part, (0, 0.0))
        previous = self._commanded.get(part)
        if previous is not None and previous.shape == value.shape:
            max_step = max(max_step, float(np.max(np.abs(value - previous))))
        self.command_stats[part] = (count + 1, max_step)
        self._commanded[part] = value

    def _tick(self) -> None:
        i = self._frame % len(self._state)
        stamp = self.get_clock().now().to_msg()
        for camera, publisher in self._cameras.items():
            message = CompressedImage()
            message.header.stamp = stamp
            message.format = "jpeg"
            message.data = self._jpegs[camera][i]
            publisher.publish(message)
        s = self._state[i]
        positions = {
            "left_arm": s[0:7],
            "right_arm": s[7:14],
            "left_gripper": s[14:15],
            "right_gripper": s[15:16],
            "torso": READY_POSE["torso"],
            "head": READY_POSE["head"],
        }
        positions.update(self._commanded)
        for part, publisher in self._states.items():
            message = RobotJointState()
            message.header.stamp = stamp
            message.position = [float(v) for v in positions[part]]
            publisher.publish(message)
        self._frame += 1


def main() -> None:
    args = parse_args()
    if os.environ.get("ROS_DOMAIN_ID", "0") == "25":
        raise SystemExit("ROS_DOMAIN_ID=25 is the real robot's domain; use an isolated one (e.g. 77)")
    topics = yaml.safe_load(Path(args.topics).read_text())
    state, jpegs = load_episode(args.dataset, args.index_dir, args.episode, args.max_frames)
    rclpy.init()
    node = FakeAstribot(state, jpegs, topics, args.fps, args.follow_commands)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():  # a SIGTERM shuts the context down underneath spin(); that is a normal stop
            raise
    finally:
        for part, (count, max_step) in sorted(node.command_stats.items()):
            print(f"[fake-robot] {part}: {count} commands, max step {max_step:.4f}", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
