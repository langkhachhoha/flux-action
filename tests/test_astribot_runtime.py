# Copyright 2026 Black Forest Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROS-free parts of the Astribot robot client (scripts/astribot/robot_runtime.py).

Runs under this repository's environment and under ROS 2 Humble's Python 3.10, where the client runs:
    /usr/bin/python3 -m pytest -p no:cacheprovider --noconftest tests/test_astribot_runtime.py
"""

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/astribot"))

import robot_runtime as rt  # noqa: E402
from astribot_wire import recv_message, send_message  # noqa: E402

HW = rt.FRAME_HW
CAMERAS = ("images.head", "images.left_wrist", "images.right_wrist")
READY = rt.pack_parts({p: rt.READY_POSE[p] for p in rt.POLICY_PARTS})
# The coffee export's state quantiles (checkpoint/infer/step-2300-model-bf16/config.json).
STATE_STATS = {
    "q01": [-0.669, -0.500, -1.668, 0.273, -0.685, -0.392, -0.848, -0.519, -0.270, 0.862, 0.239, -0.571,
            -0.448, -0.805, -1.284, -0.133],
    "q99": [0.474, 0.236, -1.102, 1.973, -0.137, 0.390, 0.394, 0.867, 0.402, 1.595, 2.049, 0.659, 0.656,
            0.650, 39.851, 42.448],
}  # fmt: skip


def frames(value: int = 0) -> dict:
    return {key: np.full((*HW, 3), value, dtype=np.uint8) for key in CAMERAS}


def test_vector_layout_matches_the_dataset():
    assert rt.STATE_NAMES[:2] == ("arm_left_j0", "arm_left_j1")
    assert rt.STATE_NAMES[7] == "arm_right_j0" and rt.STATE_NAMES[14:] == ("gripper_left", "gripper_right")
    vector = np.arange(16, dtype=np.float32)
    parts = rt.unpack_vector(vector)
    assert parts["left_arm"].tolist() == list(range(7)) and parts["right_gripper"].tolist() == [15]
    np.testing.assert_array_equal(rt.pack_parts(parts), vector)
    with pytest.raises(ValueError):
        rt.unpack_vector(np.zeros(15))


def test_ready_pose_lies_in_the_training_range():
    assert rt.state_range_issues(READY, STATE_STATS) == []


def test_home_pose_is_complete_and_within_the_arm_limits():
    dof = {"torso": 4, "left_arm": 7, "right_arm": 7, "left_gripper": 1, "right_gripper": 1, "head": 2}
    for pose in (rt.HOME_POSE, rt.READY_POSE):
        assert {part: len(value) for part, value in pose.items()} == dof
    for arm in ("left_arm", "right_arm"):
        lower, upper = rt.ARM_LIMITS[arm]
        assert np.all(lower <= rt.HOME_POSE[arm]) and np.all(rt.HOME_POSE[arm] <= upper)
    assert rt.HOME_POSE["left_gripper"] == (0.0,) and rt.HOME_POSE["right_gripper"] == (0.0,)


def test_pose_plan_reports_the_largest_change():
    current = {part: np.asarray(value) for part, value in rt.READY_POSE.items()}
    current["left_gripper"] = np.array([37.0])
    lines = rt.pose_plan(current, rt.HOME_POSE, "home")
    assert lines[0] == "Motion plan to the HOME pose:" and len(lines) == 8
    head = next(line for line in lines if line.strip().startswith("head"))
    assert "0.854 rad" in head  # head pitch 0.887 -> 0.033
    assert any("37.0 -> 0.0" in line for line in lines)
    assert lines[-1].startswith("  largest joint change 0.854 rad")


def test_operator_keys_read_lines_until_end_of_input():
    read_fd, write_fd = os.pipe()
    keys = rt.OperatorKeys(read_fd)
    try:
        assert keys.poll() is None
        os.write(write_fd, b"H\n  x \n\nh")
        assert keys.wait() == "h" and keys.wait() == "x" and keys.wait() == ""
        time.sleep(0.3)
        assert keys.poll() is None  # "h" without Enter is not a line yet
        os.close(write_fd)
        assert keys.wait() is None and keys.poll() is None  # end of input stays
    finally:
        keys.close()
        os.close(read_fd)


def test_state_range_flags_wrong_units():
    degrees = READY.copy()
    degrees[:14] = np.rad2deg(degrees[:14])
    assert {i.name for i in rt.state_range_issues(degrees, STATE_STATS)} >= {"arm_left_j3", "arm_right_j2"}
    command_scale = READY.copy()
    command_scale[14] = 100.0  # a 0-100 command reading where the dataset stores the 0-40 finger position
    (issue,) = rt.state_range_issues(command_scale, STATE_STATS)
    assert issue.name == "gripper_left" and issue.spans_outside > 1.0


def test_history_fills_the_first_tick_and_then_rolls():
    history = rt.TickHistory(8, CAMERAS)
    state0 = READY.copy()
    history.append(frames(0), state0, None)
    window = history.window()
    assert window["state"].shape == (8, 16) and window["state"].dtype == np.float32
    np.testing.assert_array_equal(window["command_history"], np.tile(state0, (8, 1)))
    assert window["images.head"].shape == (8, *HW, 3) and window["images.head"].dtype == np.uint8
    command = state0 + 0.01
    history.append(frames(1), state0 + 0.005, command)
    window = history.window()
    np.testing.assert_allclose(window["command_history"][-1], command)
    np.testing.assert_allclose(window["state"][-1], state0 + 0.005)
    assert window["images.left_wrist"][-1].max() == 1 and window["images.left_wrist"][-2].max() == 0
    np.testing.assert_allclose(history.last_command, command)


def test_history_rejects_wrong_frames_and_states():
    history = rt.TickHistory(8, CAMERAS)
    bad = frames()
    bad["images.head"] = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        history.append(bad, READY, None)
    with pytest.raises(ValueError):
        history.append(frames(), np.full(16, np.nan), None)
    with pytest.raises(RuntimeError):
        history.window()


def test_history_matches_flux_observation_history():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from flux_action.processing.history import ObservationHistory

    config = SimpleNamespace(n_obs_steps=8, action_dim=16, camera_keys=CAMERAS)
    reference, ours = ObservationHistory(8), rt.TickHistory(8, CAMERAS)
    rng = np.random.default_rng(0)
    last = None
    for tick in range(11):
        state = READY + rng.normal(0, 0.01, 16).astype(np.float32)
        images = {k: rng.integers(0, 255, (*HW, 3), dtype=np.uint8) for k in CAMERAS}
        batch = {"state": torch.from_numpy(state)[None]}
        batch.update({k: torch.from_numpy(v).permute(2, 0, 1)[None] for k, v in images.items()})
        expected = reference.append(batch, config, None if last is None else torch.from_numpy(last)[None])
        ours.append(images, state, last)
        window = ours.window()
        np.testing.assert_array_equal(window["state"], expected["state"][0].numpy())
        np.testing.assert_array_equal(window["command_history"], expected["command_history"][0].numpy())
        for key in CAMERAS:
            np.testing.assert_array_equal(window[key], expected[key][0].permute(0, 2, 3, 1).numpy())
        last = state + 0.02  # the command the loop issues at this tick


def test_limiter_clamps_steps_limits_and_grippers():
    limiter = rt.ActionLimiter(max_joint_delta=0.15, max_gripper_delta=20.0)
    action = READY.astype(np.float64).copy()
    action[0] += 1.0  # a 1 rad jump
    action[3] = 9.0  # beyond the elbow limit
    action[14], action[15] = 150.0, -5.0
    out = limiter.limit(action, READY)
    assert out[0] == pytest.approx(READY[0] + 0.15)
    assert out[3] <= 2.55 and out[3] == pytest.approx(min(READY[3] + 0.15, 2.55))
    assert out[14] == pytest.approx(20.0) and out[15] == 0.0
    with pytest.raises(ValueError):
        limiter.limit(np.full(16, np.nan), READY)


def test_rate_limit_and_safe_hold():
    np.testing.assert_allclose(rt.rate_limit(np.zeros(7), np.ones(7), 0.032), np.full(7, 0.032))
    measured = {p: np.zeros(7 if p.endswith("arm") else 1) for p in rt.POLICY_PARTS}
    held = rt.safe_hold_parts(measured, {"left_arm": np.ones(7)}, {"right_gripper": np.array([50.0])})
    assert held["left_arm"].tolist() == [1.0] * 7 and held["right_arm"].tolist() == [0.0] * 7
    assert held["right_gripper"].tolist() == [50.0] and held["left_gripper"].tolist() == [0.0]


def test_tracking_severity_and_critical_streak():
    measured = READY.astype(np.float64)
    assert rt.arm_tracking(measured, measured).severity == "ok"
    lagging = measured.copy()
    lagging[9] += 0.2
    tracking = rt.arm_tracking(lagging, measured)
    assert tracking.severity == "warning" and tracking.worst_joint == "arm_right_j2"
    lagging[9] += 0.2
    assert rt.arm_tracking(lagging, measured).severity == "critical"
    streak = rt.CriticalStreak(3)
    assert [
        streak.observe(s) for s in ("critical", "critical", "ok", "critical", "critical", "critical")
    ] == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_aspect_warning():
    assert rt.aspect_warning((720, 1280)) is None and rt.aspect_warning((360, 640)) is None
    assert "16:9" in rt.aspect_warning((240, 320))


def test_jpeg_decode_is_rgb_and_resize_keeps_size():
    cv2 = pytest.importorskip("cv2")
    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # red
    data = cv2.imencode(".jpg", rgb[:, :, ::-1])[1].tobytes()
    decoded = rt.decode_compressed_rgb(data)
    assert decoded[..., 0].mean() > 190 and decoded[..., 2].mean() < 10
    assert rt.resize_rgb(decoded).shape == (*HW, 3)


def test_policy_client_round_trip():
    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]
    seen = {}

    def serve():
        conn, _ = server.accept()
        with conn:
            hello = {
                "type": "hello",
                "action_dim": 16,
                "frame_hw": list(HW),
                "camera_keys": list(CAMERAS),
                "chunk_size": 42,
                "state_names": list(rt.STATE_NAMES),
            }
            send_message(conn, hello)
            meta, arrays = recv_message(conn)
            seen.update(meta=meta, arrays=arrays)
            chunk = np.tile(arrays["command_history"][-1], (42, 1)).astype(np.float32)
            send_message(
                conn, {"type": "actions", "inference_s": 0.1, "seed": meta.get("seed")}, {"actions": chunk}
            )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = rt.PolicyClient("127.0.0.1", port, timeout=5.0)
    history = rt.TickHistory(8, client.camera_keys)
    history.append(frames(3), READY, None)
    actions, reply = client.infer(history.window(), "serving brewed coffee", seed=7)
    client.close()
    thread.join(timeout=5.0)
    server.close()
    assert actions.shape == (42, 16) and actions.dtype == np.float64
    np.testing.assert_allclose(actions[0], READY, atol=1e-6)
    assert (
        seen["meta"] == {"type": "infer", "task": "serving brewed coffee", "seed": 7} and reply["seed"] == 7
    )
    assert seen["arrays"]["images.right_wrist"].shape == (8, *HW, 3)


def test_parse_address():
    assert rt.parse_address("127.0.0.1:5555") == ("127.0.0.1", 5555)
    with pytest.raises(ValueError):
        rt.parse_address("5555")


def test_video_recorder_writes_cameras_grid_and_session_info(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from robot_video import VideoRecorder

    cameras = ("head", "left_wrist", "right_wrist")
    recorder = VideoRecorder(tmp_path / "video", cameras, 30.0, title="step-test", mode="EXECUTE")
    recorder.submit({c: np.zeros((*HW, 3), np.uint8) for c in cameras})  # before start: ignored
    recorder.start()
    assert recorder.is_recording and recorder.codec in ("avc1", "mp4v")
    for i in range(12):
        recorder.set_status(phase="RUN", cycle=i // 4, note=f"chunk motion 0.0{i} rad")
        shot = {c: np.full((*HW, 3), 20 * i, np.uint8) for c in cameras[: 2 if i == 5 else 3]}
        recorder.submit(shot, READY)
    summary = recorder.stop()
    assert recorder.stop() == summary and not recorder.is_recording
    assert summary["frames"] == 12 and summary["dropped_frames"] == 0 and summary["errors"] == []
    info = json.loads((tmp_path / "video/session_info.json").read_text())
    assert info["files"] == ["grid.mp4", "head.mp4", "left_wrist.mp4", "right_wrist.mp4"]
    for name, size in (("head", (640, 360)), ("grid", (1280, 720))):
        capture = cv2.VideoCapture(str(tmp_path / f"video/{name}.mp4"))
        shape = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        count = 0
        while capture.read()[0]:
            count += 1
        capture.release()
        assert shape == size and count == 12


def test_episode_log_shares_the_run_directory(tmp_path):
    pytest.importorskip("cv2")
    from robot_log import EpisodeLog

    (tmp_path / "video").mkdir()  # the recorder starts first
    log = EpisodeLog(tmp_path, CAMERAS, 30.0, {}, windows=False)
    log.tick(frames(1), READY, None)
    assert log.close() == tmp_path and (tmp_path / "meta.json").exists()
    with pytest.raises(FileExistsError):
        EpisodeLog(tmp_path, CAMERAS, 30.0, {}, windows=False)


def test_console_tee_copies_prints(tmp_path, capsys):
    from robot_log import ConsoleTee

    tee = ConsoleTee(tmp_path / "console.log")
    print("cycle=0000", flush=True)
    print("WARNING x", file=sys.stderr)
    tee.close()
    tee.close()
    print("after close")
    captured = capsys.readouterr()
    assert "cycle=0000" in captured.out and "after close" in captured.out
    assert (tmp_path / "console.log").read_text() == "cycle=0000\nWARNING x\n"
