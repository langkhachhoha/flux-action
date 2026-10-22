"""ROS 2 side of the Astribot S1 robot client: camera/joint subscriptions and the joint command stream.

Ported from vmo_diffusion_policy_vla's ``AstribotS1Node`` (same topics, QoS, 250 Hz rate-limited arm stream,
grippers sent on change, cosine ready/home-pose motion, no-jump hold). The node only keeps the newest decoded
frame of each camera and the newest state of each body part; the control loop samples them at its own
ticks, which is where the policy history advances (as in ``FluxActionPolicy.select_action``). An attached
video recorder gets the newest frames from a timer of its own, so the video runs on while the loop waits.

Needs ROS 2 Humble (Python 3.10) and the ``astribot_msgs`` overlay of the vm_astribot workspace.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from astribot_msgs.msg import RobotJointController, RobotJointState
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from robot_runtime import (
    FRAME_HW,
    POLICY_PARTS,
    decode_compressed_rgb,
    pack_parts,
    rate_limit,
    resize_rgb,
    safe_hold_parts,
    unpack_vector,
)
from sensor_msgs.msg import CompressedImage

PART_DOF = {"torso": 4, "left_arm": 7, "right_arm": 7, "left_gripper": 1, "right_gripper": 1, "head": 2}
ARM_JOINT_NAMES = {
    "left_arm": [f"astribot_arm_left_joint_{i}" for i in range(1, 8)],
    "right_arm": [f"astribot_arm_right_joint_{i}" for i in range(1, 8)],
}


class AstribotNode(Node):
    """Newest camera frames (RGB, resized to 360x640) and joint states; publishes joint commands."""

    def __init__(
        self,
        topics: Mapping,
        cameras: Sequence[str],
        *,
        frame_hw: Tuple[int, int] = FRAME_HW,
        node_name: str = "flux_action_astribot",
    ) -> None:
        super().__init__(node_name)
        self._lock = threading.RLock()
        self._frame_hw = tuple(frame_hw)
        self.cameras = tuple(cameras)
        self.required_parts = tuple(topics.get("required_state_parts", PART_DOF))
        unknown = set(self.required_parts) - set(PART_DOF)
        if unknown:
            raise ValueError(f"unknown required_state_parts: {sorted(unknown)}")
        self._state_parts = tuple(dict.fromkeys((*POLICY_PARTS, *self.required_parts)))
        stream = dict(topics.get("command_stream", {}))
        self.command_hz = float(stream.get("publish_hz", 250.0))
        self.arm_velocity_limit = float(stream.get("arm_velocity_limit", 8.0))
        if self.command_hz <= 0 or self.arm_velocity_limit <= 0:
            raise ValueError("command_stream publish_hz and arm_velocity_limit must be positive")
        self._arm_max_step = self.arm_velocity_limit / self.command_hz

        self._frames: Dict[str, np.ndarray] = {}
        self._frame_times: Dict[str, float] = {}
        self.native_hw: Dict[str, Tuple[int, int]] = {}
        self._decode_errors: Dict[str, str] = {}
        self._parts: Dict[str, np.ndarray] = {}
        self._part_times: Dict[str, float] = {}
        self._target: Optional[Dict[str, np.ndarray]] = None  # the command the 250 Hz stream moves toward
        self._published_arms: Dict[str, np.ndarray] = {}
        self._published_grippers: Dict[str, np.ndarray] = {}
        self._published_body: Dict[str, np.ndarray] = {}  # torso and head, commanded only by pose motions
        self._recorder = None

        sensors = ReentrantCallbackGroup()
        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE, depth=15
        )
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE, depth=10
        )
        camera_topics, state_topics = dict(topics["cameras"]), dict(topics["states"])
        command_topics = dict(topics["commands"])
        missing = [c for c in self.cameras if c not in camera_topics]
        missing += [p for p in self._state_parts if p not in state_topics]
        missing += [p for p in POLICY_PARTS if p not in command_topics]
        if missing:
            raise ValueError(f"topic config lacks: {missing}")
        for camera in self.cameras:
            self.create_subscription(
                CompressedImage,
                camera_topics[camera],
                lambda msg, key=camera: self._on_image(msg, key),
                qos_sub,
                callback_group=sensors,
            )
        for part in self._state_parts:
            self.create_subscription(
                RobotJointState,
                state_topics[part],
                lambda msg, key=part: self._on_joint(msg, key),
                qos_sub,
                callback_group=sensors,
            )
        self._command_pubs = {
            part: self.create_publisher(RobotJointController, topic, qos_pub)
            for part, topic in command_topics.items()
            if part in PART_DOF
        }
        self.create_timer(
            1.0 / self.command_hz, self._stream, callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.get_logger().info(
            f"cameras={list(self.cameras)} states={list(self._state_parts)} "
            f"arm stream {self.command_hz:.0f}Hz, limit {self.arm_velocity_limit:.1f}rad/s"
        )

    # ------------------------------------------------------------------------------------------ inputs
    def _on_image(self, message: CompressedImage, key: str) -> None:
        try:
            rgb = decode_compressed_rgb(message.data)
            native = rgb.shape[:2]
            rgb = resize_rgb(rgb, self._frame_hw)
        except Exception as error:  # a callback must not kill the executor
            with self._lock:
                self._decode_errors[key] = str(error)
            return
        with self._lock:
            self._frames[key] = rgb
            self._frame_times[key] = time.monotonic()
            self.native_hw.setdefault(key, native)
            self._decode_errors.pop(key, None)

    def _on_joint(self, message: RobotJointState, part: str) -> None:
        positions = np.asarray(message.position, dtype=np.float64)
        dof = PART_DOF[part]
        if positions.size < dof:
            self.get_logger().error(f"{part} state has {positions.size} positions, expected {dof}")
            return
        names = ARM_JOINT_NAMES.get(part)
        if names and message.name and all(n in message.name for n in names):
            by_name = dict(zip(message.name, positions))
            positions = np.asarray([by_name[n] for n in names])
        with self._lock:
            self._parts[part] = positions[:dof].copy()
            self._part_times[part] = time.monotonic()

    def readiness(self, since: float = 0.0) -> Tuple[bool, str]:
        """Every camera and required part has delivered a message after ``since`` (monotonic seconds)."""
        with self._lock:
            cameras = [c for c in self.cameras if self._frame_times.get(c, -1.0) < since]
            parts = [p for p in self._state_parts if self._part_times.get(p, -1.0) < since]
            errors = dict(self._decode_errors)
        reasons = []
        if cameras:
            reasons.append(f"waiting for cameras {cameras}")
        if parts:
            reasons.append(f"waiting for joint states {parts}")
        if errors:
            reasons.append(f"image decode errors {errors}")
        return not reasons, "; ".join(reasons) or "ready"

    def ages(self) -> Dict[str, float]:
        now = time.monotonic()
        with self._lock:
            ages = {f"camera:{c}": now - self._frame_times.get(c, -np.inf) for c in self.cameras}
            ages.update({f"state:{p}": now - self._part_times.get(p, -np.inf) for p in self.required_parts})
        return ages

    def sample(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Newest frames (keyed like the ROS config) and the 16-D measured state."""
        with self._lock:
            if any(c not in self._frames for c in self.cameras) or any(
                p not in self._parts for p in POLICY_PARTS
            ):
                raise RuntimeError("no complete Astribot sample yet")
            return {c: self._frames[c] for c in self.cameras}, pack_parts(self._parts)

    def part_positions(self, part: str) -> np.ndarray:
        with self._lock:
            return self._parts[part].copy()

    def last_command(self, part: str) -> np.ndarray:
        """The part's latest published command; its measured position when it was never commanded."""
        with self._lock:
            for published in (self._published_arms, self._published_grippers, self._published_body):
                if part in published:
                    return published[part].copy()
            if part not in self._parts:
                raise RuntimeError(f"no joint state yet for {part}")
            return self._parts[part].copy()

    # ------------------------------------------------------------------------------------------ video
    def attach_recorder(self, recorder, fps: float) -> None:
        """Feed ``recorder.submit`` the newest frames and state at ``fps`` while it records."""
        self._recorder = recorder
        self.create_timer(1.0 / fps, self._record, callback_group=MutuallyExclusiveCallbackGroup())

    def _record(self) -> None:
        recorder = self._recorder
        if recorder is None or not recorder.is_recording:
            return
        with self._lock:  # frames are replaced, never written into, so references suffice
            frames = {c: self._frames[c] for c in self.cameras if c in self._frames}
            state = pack_parts(self._parts) if all(p in self._parts for p in POLICY_PARTS) else None
        recorder.submit(frames, state)

    # ------------------------------------------------------------------------------------------ commands
    def _publish(self, part: str, positions: Sequence[float]) -> None:
        message = RobotJointController()
        message.header.stamp = self.get_clock().now().to_msg()
        message.mode = 1
        # The production LeRobot daemon leaves ``name`` empty; the S1 controller reads ``command`` in order.
        message.command = [float(v) for v in positions]
        self._command_pubs[part].publish(message)

    def move_to_pose(
        self, pose: Mapping[str, Sequence[float]], *, duration_s: float, rate_hz: float, max_state_age: float
    ) -> None:
        """Cosine-interpolate the listed parts from their latest command (the measured position of a part
        never commanded, so a lagging measurement cannot make the first step jump); afterwards the command
        stream holds the pose. Call only after the operator has authorized motion."""
        targets = {part: np.asarray(value, dtype=np.float64) for part, value in pose.items()}
        for part, value in targets.items():
            if (
                part not in self._command_pubs
                or value.shape != (PART_DOF[part],)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"invalid pose target for {part}: {value}")
        with self._lock:
            missing = [part for part in targets if part not in self._parts]
            if missing:
                raise RuntimeError(f"no joint state yet for {missing}")
            starts = {part: self.last_command(part) for part in targets}
            self._target = None  # the 250 Hz stream pauses: this loop is the only command source
        steps = max(1, int(round(duration_s * rate_hz)))
        deadline = time.monotonic()
        for index in range(steps):
            stale = {k: a for k, a in self.ages().items() if k.startswith("state:") and a > max_state_age}
            if stale:
                raise RuntimeError(f"joint state went stale during the pose motion: {stale}")
            alpha = 0.5 * (1.0 - np.cos(np.pi * (index + 1) / steps))
            step = {part: starts[part] + alpha * (targets[part] - starts[part]) for part in targets}
            for part, value in step.items():
                self._publish(part, value)
            with self._lock:  # an interrupted motion holds what was last published
                self._published_arms.update({p: step[p].copy() for p in step if p.endswith("_arm")})
                self._published_grippers.update({p: step[p].copy() for p in step if p.endswith("_gripper")})
                self._published_body.update({p: step[p].copy() for p in step if p in ("torso", "head")})
            deadline += 1.0 / rate_hz
            time.sleep(max(0.0, deadline - time.monotonic()))
        with self._lock:  # the stream starts from the pose, so the first policy target cannot snap back
            if all(part in targets for part in POLICY_PARTS):
                self._target = {part: targets[part].copy() for part in POLICY_PARTS}

    def set_target(self, action: Sequence[float]) -> None:
        """Next absolute 16-D command; the 250 Hz stream rate-limits the arms toward it."""
        parts = unpack_vector(action)
        if any(not np.isfinite(v).all() for v in parts.values()):
            raise ValueError("command contains NaN or infinity")
        with self._lock:
            for arm in ("left_arm", "right_arm"):
                if arm not in self._published_arms:
                    if arm not in self._parts:
                        raise RuntimeError("arm state is not ready")
                    self._published_arms[arm] = self._parts[arm].copy()
            self._target = parts

    def _stream(self) -> None:
        with self._lock:
            if self._target is None:
                return
            out = {}
            for arm in ("left_arm", "right_arm"):
                current = self._published_arms.get(arm, self._target[arm])
                out[arm] = rate_limit(current, self._target[arm], self._arm_max_step)
                self._published_arms[arm] = out[arm].copy()
            for gripper in ("left_gripper", "right_gripper"):
                previous = self._published_grippers.get(gripper)
                if previous is None or not np.allclose(previous, self._target[gripper]):
                    out[gripper] = self._target[gripper].copy()
                    self._published_grippers[gripper] = out[gripper].copy()
        for part, value in out.items():
            self._publish(part, value)

    def publish_hold(self) -> None:
        """Hold the latest published command (never a lagging measurement) and stop following targets."""
        with self._lock:
            if any(p not in self._parts for p in POLICY_PARTS):
                raise RuntimeError("joint state is not ready")
            held = safe_hold_parts(
                {p: self._parts[p] for p in POLICY_PARTS}, self._published_arms, self._published_grippers
            )
            self._target = {p: v.copy() for p, v in held.items()}
            self._published_arms.update({p: held[p].copy() for p in ("left_arm", "right_arm")})
            self._published_grippers.update({p: held[p].copy() for p in ("left_gripper", "right_gripper")})
        for part, value in held.items():
            self._publish(part, value)
