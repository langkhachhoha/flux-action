"""ROS-free pieces of the Astribot S1 robot client: data contract, poses, history, safety, operator keys,
server connection.

The robot client (``robot_client.py``) runs on the Python 3.10 of ROS 2 Humble, while the model needs this
repository's Python 3.12, so everything here sticks to the standard library, NumPy and (lazily) OpenCV, and
imports nothing from ``flux_action``. It is tested without ROS or a robot (``tests/test_astribot_runtime.py``).

Data contract (the coffee dataset ``serving_brewed_coffee_lrb_annotated``, LeRobot v3, 30 Hz):

* state / action: 16 values ``arm_left_j0..6, arm_right_j0..6, gripper_left, gripper_right``; arm joints in
  radians. The measured gripper state is the finger position (about 0 open, 35-42 closed on an object); the
  gripper command is 0 (open) .. 100 (closed). Both arrive as the robot's ``joint_space_states`` report them.
* cameras ``images.head``, ``images.left_wrist``, ``images.right_wrist``: RGB uint8 360x640 (16:9).
* the policy history mirrors ``flux_action.processing.history.ObservationHistory``: tick ``t`` records the
  measured state and frames of ``t`` and the absolute command issued before ``t``; the first tick of an
  episode fills the whole history, the measured state standing in for the commands never issued.
"""

from __future__ import annotations

import os
import queue
import select
import socket
import sys
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from astribot_wire import recv_message, send_message

STATE_DIM = 16
STATE_NAMES = (
    *(f"arm_left_j{i}" for i in range(7)),
    *(f"arm_right_j{i}" for i in range(7)),
    "gripper_left",
    "gripper_right",
)
PART_SLICES = {
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
    "left_gripper": slice(14, 15),
    "right_gripper": slice(15, 16),
}
POLICY_PARTS = tuple(PART_SLICES)
ARM_DIMS = 14
GRIPPER_RANGE = (0.0, 100.0)
FRAME_HW = (360, 640)
FLUX_CAMERA_PREFIX = "images."

# Full-body pose the coffee demonstrations start from (VMO teleop ready pose; 28 of the 394 episodes start
# within 0.1 rad of it). Torso and head are not part of the policy state, but they set the camera views.
READY_POSE = {
    "torso": (0.5962085805053711, -1.1908595955810546, 0.5908210523071289, -0.008323674865722656),
    "left_arm": (
        0.16365501190185547, -0.023521346862792968, -1.4246325209960937, 1.6550282841796875,
        -0.3483923372192383, 0.12443570697021485, 0.1267342858276367,
    ),
    "right_arm": (
        -0.15411713177490236, -0.02150647967529297, 1.411213769165039, 1.6472044661865235,
        0.34210998944091797, 0.12159181854248047, -0.11975973211669921,
    ),
    "left_gripper": (0.0,),
    "right_gripper": (0.0,),
    "head": (-0.002846445861816406, 0.8870121532592773),
}  # fmt: skip

# Resting pose (vmo_diffusion_policy_vla's S1_HOME_POSE): torso upright, arms folded, grippers open, head level.
HOME_POSE = {
    "torso": (0.28698922, -0.58628635, 0.29730871, -0.00562478),
    "left_arm": (0.15609237, -0.02107994, -1.42101810, 1.65760320, -0.34594139, 0.11592893, 0.12616208),
    "right_arm": (-0.16180375, -0.02177351, 1.41792764, 1.65719897, 0.34370263, 0.11559321, -0.12362211),
    "left_gripper": (0.0,),
    "right_gripper": (0.0,),
    "head": (0.00077752, 0.03296819),
}  # fmt: skip
POSE_PARTS = ("torso", "left_arm", "right_arm", "left_gripper", "right_gripper", "head")

# Arm limits of the production Astribot runtime (vm_lerobot eval_robot/configs/ASTRIBOT_S1.yaml); the
# shoulder yaw range differs between the arms.
ARM_LIMITS = {
    "left_arm": (
        np.array([-3.08, -1.51, -3.08, -0.0, -2.37, -0.78, -1.55]),
        np.array([3.08, 0.62, 1.51, 2.55, 2.37, 0.78, 1.55]),
    ),
    "right_arm": (
        np.array([-3.08, -1.51, -1.51, 0.0, -2.37, -0.78, -1.55]),
        np.array([3.08, 0.62, 3.08, 2.55, 2.37, 0.78, 1.55]),
    ),
}


# ---------------------------------------------------------------------------------------------- 16-D vectors
def unpack_vector(vector: Sequence[float]) -> Dict[str, np.ndarray]:
    """16-D state/action -> ``left_arm (7), right_arm (7), left_gripper (1), right_gripper (1)``."""
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (STATE_DIM,):
        raise ValueError(f"Astribot state/action must have shape ({STATE_DIM},), got {value.shape}")
    return {part: value[index].copy() for part, index in PART_SLICES.items()}


def pack_parts(parts: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Named parts -> the 16-D float32 vector in dataset order."""
    out = np.empty(STATE_DIM, dtype=np.float32)
    for part, index in PART_SLICES.items():
        value = np.asarray(parts[part], dtype=np.float32).reshape(-1)
        if value.size != index.stop - index.start:
            raise ValueError(f"{part} has {value.size} values, expected {index.stop - index.start}")
        out[index] = value
    return out


def arm_max_abs(a: Sequence[float], b: Sequence[float]) -> float:
    """Largest arm-joint difference (radians) between two 16-D vectors."""
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64)[:ARM_DIMS] - np.asarray(b)[:ARM_DIMS])))


def pose_plan(
    current: Mapping[str, Sequence[float]], target: Mapping[str, Sequence[float]], name: str
) -> List[str]:
    """The lines of a motion-plan table (largest change per body part), printed before a pose motion."""
    lines = [f"Motion plan to the {name.upper()} pose:"]
    largest = 0.0
    for part in (p for p in POSE_PARTS if p in target):
        start = np.asarray(current[part], dtype=np.float64).reshape(-1)
        goal = np.asarray(target[part], dtype=np.float64).reshape(-1)
        if part.endswith("_gripper"):
            lines.append(f"  {part:<14} {start[0]:6.1f} -> {goal[0]:.1f}")
            continue
        delta = float(np.max(np.abs(goal - start)))
        largest = max(largest, delta)
        lines.append(f"  {part:<14} max change {delta:.3f} rad ({np.rad2deg(delta):5.1f} deg)")
    lines.append(f"  largest joint change {largest:.3f} rad ({np.rad2deg(largest):.1f} deg)")
    return lines


# ---------------------------------------------------------------------------------------------- images
def camera_key(flux_key: str) -> str:
    """``images.head`` (export) -> ``head`` (ROS topic config key)."""
    if not flux_key.startswith(FLUX_CAMERA_PREFIX):
        raise ValueError(f"camera key must start with {FLUX_CAMERA_PREFIX!r}: {flux_key!r}")
    return flux_key[len(FLUX_CAMERA_PREFIX) :]


def decode_compressed_rgb(data: bytes) -> np.ndarray:
    """JPEG/PNG bytes of a ``sensor_msgs/CompressedImage`` -> RGB uint8 (OpenCV decodes BGR)."""
    import cv2

    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV could not decode the compressed image")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def resize_rgb(image: np.ndarray, hw: Tuple[int, int] = FRAME_HW) -> np.ndarray:
    """RGB uint8 -> ``hw``; area filter when shrinking (the head camera sends 1280x720)."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 uint8 RGB image, got {image.shape} {image.dtype}")
    if image.shape[:2] == tuple(hw):
        return np.ascontiguousarray(image)
    import cv2

    shrink = hw[0] <= image.shape[0] and hw[1] <= image.shape[1]
    interpolation = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    return np.ascontiguousarray(cv2.resize(image, (hw[1], hw[0]), interpolation=interpolation))


def aspect_warning(native_hw: Tuple[int, int], hw: Tuple[int, int] = FRAME_HW) -> Optional[str]:
    """A message when a camera's aspect ratio differs from the training frames' (resizing would distort)."""
    native, expected = native_hw[1] / native_hw[0], hw[1] / hw[0]
    if abs(native / expected - 1) < 0.02:
        return None
    return f"aspect {native:.3f} != {expected:.3f} (16:9); the resize distorts the image"


# ---------------------------------------------------------------------------------------------- format check
@dataclass(frozen=True)
class RangeIssue:
    name: str
    value: float
    q01: float
    q99: float
    spans_outside: float  # distance beyond [q01, q99] in units of (q99 - q01)


def state_range_issues(
    state: Sequence[float], stats: Mapping[str, Sequence[float]], tolerance: float = 0.25
) -> List[RangeIssue]:
    """Dimensions of a measured state outside the training range ``[q01, q99]`` by more than ``tolerance``
    spans. Wrong units (degrees, a 0-100 gripper reading) or a wrong joint order show up here."""
    value = np.asarray(state, dtype=np.float64)
    q01, q99 = np.asarray(stats["q01"], dtype=np.float64), np.asarray(stats["q99"], dtype=np.float64)
    span = np.maximum(q99 - q01, 1e-6)
    outside = np.maximum(q01 - value, value - q99) / span
    return [
        RangeIssue(STATE_NAMES[i], float(value[i]), float(q01[i]), float(q99[i]), float(outside[i]))
        for i in range(STATE_DIM)
        if outside[i] > tolerance
    ]


# ---------------------------------------------------------------------------------------------- safety
class ActionLimiter:
    """Clamp an absolute 16-D command against joint limits and a per-tick step from ``reference``."""

    def __init__(
        self,
        max_joint_delta: Optional[float] = 0.15,
        max_gripper_delta: Optional[float] = 20.0,
        enforce_joint_limits: bool = True,
    ) -> None:
        for name, value in (("max_joint_delta", max_joint_delta), ("max_gripper_delta", max_gripper_delta)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        self.max_joint_delta = max_joint_delta
        self.max_gripper_delta = max_gripper_delta
        self.enforce_joint_limits = enforce_joint_limits

    def limit(self, action: Sequence[float], reference: Sequence[float]) -> np.ndarray:
        target, current = unpack_vector(action), unpack_vector(reference)
        if any(not np.isfinite(v).all() for v in target.values()):
            raise ValueError("Policy produced NaN or infinity; refusing to command the robot")
        if any(not np.isfinite(v).all() for v in current.values()):
            raise ValueError("Reference state contains NaN or infinity")
        for arm in ("left_arm", "right_arm"):
            lower, upper = ARM_LIMITS[arm]
            if self.enforce_joint_limits:
                target[arm] = np.clip(target[arm], lower, upper)
            if self.max_joint_delta is not None:
                step = np.clip(target[arm] - current[arm], -self.max_joint_delta, self.max_joint_delta)
                target[arm] = current[arm] + step
            if self.enforce_joint_limits:  # the reference itself may sit marginally outside a bound
                target[arm] = np.clip(target[arm], lower, upper)
        for gripper in ("left_gripper", "right_gripper"):
            if self.max_gripper_delta is not None:
                step = np.clip(
                    target[gripper] - current[gripper], -self.max_gripper_delta, self.max_gripper_delta
                )
                target[gripper] = current[gripper] + step
            target[gripper] = np.clip(target[gripper], *GRIPPER_RANGE)
        return pack_parts(target).astype(np.float64)


def rate_limit(current: Sequence[float], target: Sequence[float], max_step: float) -> np.ndarray:
    """Advance ``current`` toward ``target`` by at most ``max_step`` per joint (the 250 Hz arm stream)."""
    current_np, target_np = np.asarray(current, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if current_np.shape != target_np.shape or max_step <= 0:
        raise ValueError("current/target must match and max_step must be positive")
    return current_np + np.clip(target_np - current_np, -max_step, max_step)


def safe_hold_parts(
    measured: Mapping[str, Sequence[float]],
    published_arms: Mapping[str, Sequence[float]],
    published_grippers: Mapping[str, Sequence[float]],
) -> Dict[str, np.ndarray]:
    """Hold targets that never jump: the latest published command of each part, the measurement only for a
    part never commanded (a lagging measurement can trip the S1 EtherCAT sudden-change guard)."""
    held = {}
    for part in POLICY_PARTS:
        source = (published_arms if part.endswith("_arm") else published_grippers).get(part, measured[part])
        value = np.asarray(source, dtype=np.float64).reshape(-1)
        if not np.isfinite(value).all():
            raise ValueError(f"hold value for {part} is not finite")
        held[part] = value.copy()
    return held


@dataclass(frozen=True)
class Tracking:
    """Previous arm command versus the fresh measurement (the LeRobot S1 runtime's diagnostic)."""

    error: np.ndarray
    maximum: float
    rms: float
    worst_joint: str
    severity: str  # ok | warning | critical


def arm_tracking(previous_command, measured, warn: float = 0.15, critical: float = 0.30) -> Tracking:
    if not 0 < warn < critical:
        raise ValueError("tracking thresholds must satisfy 0 < warn < critical")
    error = np.asarray(previous_command, dtype=np.float64)[:ARM_DIMS] - np.asarray(measured)[:ARM_DIMS]
    if not np.isfinite(error).all():
        raise ValueError("tracking inputs contain NaN or infinity")
    magnitude = np.abs(error)
    maximum = float(magnitude.max())
    severity = "critical" if maximum > critical else "warning" if maximum > warn else "ok"
    return Tracking(
        error, maximum, float(np.sqrt(np.mean(error**2))), STATE_NAMES[int(magnitude.argmax())], severity
    )


class CriticalStreak:
    """Stop after ``limit`` consecutive critical tracking samples (0 disables); one spike is tolerated."""

    def __init__(self, limit: int = 3) -> None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        self.limit, self.count = limit, 0

    def observe(self, severity: str) -> bool:
        self.count = self.count + 1 if severity == "critical" else 0
        return self.limit > 0 and self.count >= self.limit


# ---------------------------------------------------------------------------------------------- operator
class OperatorKeys:
    """Lines the operator types while the robot runs (``h`` + Enter: go home), read by a thread so the
    control loop never waits for the terminal. Reads the raw descriptor: a daemon thread blocked inside
    ``sys.stdin`` can abort the interpreter at exit."""

    def __init__(self, fd: Optional[int] = None) -> None:
        self._fd = sys.stdin.fileno() if fd is None else fd
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._read, name="operator-keys", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        pending = b""
        while not self._closed.is_set():
            try:
                if not select.select([self._fd], [], [], 0.1)[0]:
                    continue
                data = os.read(self._fd, 1024)
            except (OSError, ValueError):
                break
            if not data:
                break
            *lines, pending = (pending + data).split(b"\n")
            for line in lines:
                self._lines.put(line.decode(errors="replace").strip().lower())
        self._lines.put(None)  # end of input

    def _next(self, timeout: Optional[float]) -> Optional[str]:
        line = self._lines.get(timeout=timeout) if timeout else self._lines.get_nowait()
        if line is None:
            self._lines.put(None)  # stays at end of input
        return line

    def poll(self) -> Optional[str]:
        """The next typed line (lower case, stripped), or None when there is none (yet)."""
        try:
            return self._next(None)
        except queue.Empty:
            return None

    def wait(self) -> Optional[str]:
        """Block for the next typed line; None at end of input. Ctrl-C still interrupts."""
        while True:
            try:
                return self._next(0.1)
            except queue.Empty:
                continue

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------------------------- history
class TickHistory:
    """The last ``n_obs_steps`` control ticks of one episode: frames, measured states, preceding commands."""

    def __init__(self, n_obs_steps: int, camera_keys: Sequence[str]) -> None:
        if n_obs_steps < 1 or not camera_keys:
            raise ValueError("need n_obs_steps >= 1 and at least one camera")
        self.n_obs_steps = n_obs_steps
        self.camera_keys = tuple(camera_keys)
        self._frames = {key: deque(maxlen=n_obs_steps) for key in self.camera_keys}
        self._states: deque = deque(maxlen=n_obs_steps)
        self._commands: deque = deque(maxlen=n_obs_steps)

    def __len__(self) -> int:
        return len(self._states)

    def append(
        self,
        frames: Mapping[str, np.ndarray],
        state: Sequence[float],
        last_command: Optional[Sequence[float]],
    ) -> None:
        """Record one control tick; ``last_command`` is None when no command preceded it."""
        state_np = np.asarray(state, dtype=np.float32).copy()
        if state_np.shape != (STATE_DIM,) or not np.isfinite(state_np).all():
            raise ValueError(f"state must be a finite ({STATE_DIM},) vector, got {state_np.shape}")
        command = (
            state_np.copy() if last_command is None else np.asarray(last_command, dtype=np.float32).copy()
        )
        if command.shape != (STATE_DIM,) or not np.isfinite(command).all():
            raise ValueError(f"command must be a finite ({STATE_DIM},) vector")
        current = {}
        for key in self.camera_keys:
            frame = np.asarray(frames[key])
            if frame.shape != (*FRAME_HW, 3) or frame.dtype != np.uint8:
                raise ValueError(f"{key} must be uint8 {(*FRAME_HW, 3)} RGB, got {frame.shape} {frame.dtype}")
            current[key] = frame.copy()
        for _ in range(self.n_obs_steps if not self._states else 1):
            for key in self.camera_keys:
                self._frames[key].append(current[key])
            self._states.append(state_np)
            self._commands.append(command)

    @property
    def last_command(self) -> np.ndarray:
        if not self._commands:
            raise RuntimeError("history is empty")
        return self._commands[-1].copy()

    def window(self) -> Dict[str, np.ndarray]:
        """One server request: ``state`` and ``command_history`` (n, 16) float32, ``images.*`` (n, H, W, 3)."""
        if len(self._states) != self.n_obs_steps:
            raise RuntimeError("history is empty")
        window = {
            "state": np.stack(self._states).astype(np.float32),
            "command_history": np.stack(self._commands).astype(np.float32),
        }
        for key in self.camera_keys:
            window[key] = np.stack(self._frames[key])
        return window


# ---------------------------------------------------------------------------------------------- server
def parse_address(address: str) -> Tuple[str, int]:
    host, separator, port = address.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError(f"server address must be HOST:PORT, got {address!r}")
    return host, int(port)


class PolicyClient:
    """Blocking connection to ``policy_server.py``; ``metadata`` is the server's hello message."""

    def __init__(self, host: str, port: int, timeout: float = 120.0) -> None:
        self._sock = socket.create_connection((host, port), timeout=10.0)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(timeout)
        meta, _ = recv_message(self._sock)
        if meta.get("type") != "hello":
            raise RuntimeError(f"unexpected greeting from the policy server: {meta}")
        if int(meta["action_dim"]) != STATE_DIM or tuple(meta["frame_hw"]) != FRAME_HW:
            raise ValueError(f"server policy is not the Astribot contract: {meta}")
        names = meta.get("state_names")
        if names is not None and tuple(names) != STATE_NAMES:
            raise ValueError(f"server state order {names} differs from {STATE_NAMES}")
        self.metadata = meta

    @property
    def camera_keys(self) -> List[str]:
        return list(self.metadata["camera_keys"])

    def infer(
        self, window: Mapping[str, np.ndarray], task: Optional[str] = None, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """One history window -> ``(chunk_size, 16)`` absolute commands (float64) and the reply metadata."""
        request: dict = {"type": "infer"}
        if task:
            request["task"] = task
        if seed is not None:
            request["seed"] = int(seed)
        send_message(self._sock, request, dict(window))
        meta, arrays = recv_message(self._sock)
        if meta.get("type") == "error":
            raise RuntimeError(f"policy server failed:\n{meta.get('message')}")
        if meta.get("type") != "actions" or "actions" not in arrays:
            raise RuntimeError(f"unexpected policy server reply: {meta}")
        actions = np.asarray(arrays["actions"], dtype=np.float64)
        expected = (int(self.metadata["chunk_size"]), STATE_DIM)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise RuntimeError(f"policy returned invalid actions {actions.shape}, expected {expected}")
        return actions, meta

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
