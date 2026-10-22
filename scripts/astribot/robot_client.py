#!/usr/bin/env python3
"""Run a FLUX Action export on Astribot S1 over ROS 2; the model runs in ``policy_server.py``.

    bash scripts/astribot/run_robot.sh --max-cycles 3                     # dry-run: nothing is published
    bash scripts/astribot/run_robot.sh --execute --prepare-ready-pose     # moves the robot

While it runs, ``h`` + Enter stops the policy, holds, prints the motion plan and (after a second ``h``, or at
once with ``--yes``) moves the robot to the HOME pose; Ctrl-C stops and holds. Each run gets one directory
(default ``~/Desktop/astribot_logs/flux_action/<time>_<execute|dry>``): the episode log, the real-time video
of every camera plus a status grid in ``video/`` (from the start confirmation to the end, like vmo's
recorder), and ``console.log``, a copy of everything printed.

ROS 2 Humble's rclpy is Python 3.10 and the model needs Python 3.12, so this process (system Python 3.10)
owns the topics, the 30 Hz control loop and the safety layers, and asks the policy server (``.venv``, GPU)
for a 42-step chunk whenever its action queue is empty. At every tick it records the camera frames, the
measured state and the command issued before the tick, exactly as ``FluxActionPolicy.select_action`` does;
the server integrates the predicted arm deltas onto the last command. Without ``--execute`` nothing is
published and the measured state stands in for the command (the robot keeps holding).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from robot_log import ConsoleTee, EpisodeLog
from robot_runtime import (
    HOME_POSE,
    READY_POSE,
    ActionLimiter,
    CriticalStreak,
    OperatorKeys,
    PolicyClient,
    TickHistory,
    arm_max_abs,
    arm_tracking,
    aspect_warning,
    camera_key,
    pack_parts,
    parse_address,
    pose_plan,
    state_range_issues,
)
from robot_video import VideoRecorder

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TOPICS = SCRIPT_DIR / "ros2_astribot.yaml"
REFUSE_SPANS = 1.0  # a state this far outside the training range means wrong units or joint order


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--server", default="127.0.0.1:5555", help="policy server HOST:PORT")
    p.add_argument("--server-timeout", type=float, default=120.0, help="seconds to wait per prediction")
    p.add_argument(
        "--task", default=None, help="instruction (default: the server's, 'serving brewed coffee')"
    )
    p.add_argument("--topics", default=str(DEFAULT_TOPICS), help="ROS 2 topic config (YAML)")
    p.add_argument("--execute", action="store_true", help="publish commands; without it this is a dry-run")
    p.add_argument("--yes", action="store_true", help="skip the interactive start confirmation of --execute")
    p.add_argument(
        "--prepare-ready-pose",
        action="store_true",
        help="first move torso/arms/grippers/head to the ready pose the demonstrations start from",
    )
    p.add_argument("--ready-pose-duration", type=float, default=3.0)
    p.add_argument("--ready-pose-rate", type=float, default=100.0, help="command rate of pose motions (Hz)")
    p.add_argument("--home-duration", type=float, default=4.0, help="seconds of the motion to HOME ('h')")
    p.add_argument(
        "--execute-steps",
        type=int,
        default=None,
        help="actions run per prediction (default n_action_steps=32)",
    )
    p.add_argument(
        "--max-cycles", type=int, default=None, help="stop after N predictions (and their actions)"
    )
    p.add_argument(
        "--seed-mode",
        choices=("cycle", "fixed"),
        default="cycle",
        help="cycle: sampler seed = --seed + prediction index (a new noise draw every prediction); "
        "fixed: the export's inference_seed for every prediction",
    )
    p.add_argument("--seed", type=int, default=0, help="first sampler seed of --seed-mode cycle")
    p.add_argument("--no-warmup", action="store_true", help="skip one discarded prediction before the loop")
    p.add_argument("--ready-timeout", type=float, default=30.0)
    p.add_argument("--max-sensor-age", type=float, default=1.0, help="stop when a stream is older (s)")
    p.add_argument("--max-joint-delta", type=float, default=0.15, help="arm step limit per tick (rad)")
    p.add_argument("--max-gripper-delta", type=float, default=20.0, help="gripper step limit per tick")
    p.add_argument("--disable-joint-limits", action="store_true", help="not recommended on hardware")
    p.add_argument("--tracking-warn-threshold", type=float, default=0.15)
    p.add_argument("--tracking-critical-threshold", type=float, default=0.30)
    p.add_argument("--tracking-critical-limit", type=int, default=3, help="stop after N critical in a row")
    p.add_argument("--tracking-log-interval", type=int, default=30)
    p.add_argument(
        "--allow-out-of-range-state",
        action="store_true",
        help=f"allow --execute although the measured state is > {REFUSE_SPANS} span outside the training range",
    )
    p.add_argument(
        "--log-dir", default=None, help="run directories (default ~/Desktop/astribot_logs/flux_action)"
    )
    p.add_argument("--no-log", action="store_true", help="do not write the episode log")
    p.add_argument(
        "--no-log-windows",
        action="store_true",
        help="do not keep the lossless request frames (~5 MB per prediction) needed for an exact replay",
    )
    p.add_argument("--no-record", action="store_true", help="do not record the real-time camera video")
    p.add_argument("--no-record-grid", action="store_true", help="record the cameras without grid.mp4")
    p.add_argument("--record-fps", type=float, default=None, help="video rate (default: control rate)")
    args = p.parse_args()
    if args.prepare_ready_pose and not args.execute:
        p.error("--prepare-ready-pose requires --execute")
    if args.max_cycles is not None and args.max_cycles < 1:
        p.error("--max-cycles must be >= 1")
    if not 0 < args.tracking_warn_threshold < args.tracking_critical_threshold:
        p.error("tracking thresholds must satisfy 0 < warn < critical")
    positive = ("max_sensor_age", "ready_pose_duration", "ready_pose_rate", "home_duration")
    if any(getattr(args, name) <= 0 for name in positive) or (args.record_fps or 1) <= 0:
        p.error(
            "--max-sensor-age, --ready-pose-duration, --ready-pose-rate, --home-duration, --record-fps "
            "must be positive"
        )
    return args


def wait_ready(node, timeout: float, since: float = 0.0) -> None:
    deadline, last = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        ready, reason = node.readiness(since)
        if ready:
            return
        if reason != last:
            print(f"Waiting for Astribot inputs: {reason}", flush=True)
            last = reason
        time.sleep(0.05)
    raise TimeoutError(f"Astribot inputs not ready after {timeout:.0f}s: {node.readiness(since)[1]}")


def default_log_root() -> Path:
    desktop = Path.home() / "Desktop/astribot_logs"
    return (
        desktop / "flux_action" if desktop.is_dir() else SCRIPT_DIR.parents[1] / "outputs/astribot/robot_logs"
    )


def check_state(state: np.ndarray, meta: dict, execute: bool, allow: bool) -> None:
    """Print the measured state against the training range; refuse --execute on a units/order mismatch."""
    left, right, grippers = (
        np.round(state[s], 3).tolist() for s in (slice(0, 7), slice(7, 14), slice(14, 16))
    )
    print(f"Measured state: left_arm={left} right_arm={right} grippers={grippers}", flush=True)
    stats = meta.get("state_normalization")
    if not stats:
        return
    issues = state_range_issues(state, stats)
    if not issues:
        print("State check: every value lies within the training range.", flush=True)
        return
    for issue in issues:
        print(
            f"WARNING state {issue.name}={issue.value:.3f} outside the training range "
            f"[{issue.q01:.3f}, {issue.q99:.3f}] by {issue.spans_outside:.2f} span(s)",
            flush=True,
        )
    worst = max(issue.spans_outside for issue in issues)
    if execute and worst > REFUSE_SPANS and not allow:
        raise RuntimeError(
            "measured state is far outside the training data (wrong units, joint order or pose); refusing "
            "--execute. Fix it, or pass --allow-out-of-range-state after checking."
        )


def operator_keys(execute: bool):
    """Start reading the operator's keys from the terminal (None without one)."""
    try:
        keys = OperatorKeys()
    except (AttributeError, OSError, ValueError):  # stdin missing or closed
        return None
    action = "stop the policy and move the robot to HOME" if execute else "stop (dry-run: nothing moves)"
    print(f"Type h + Enter to {action}; Ctrl-C stops and holds.", flush=True)
    return keys


def go_home(node, args: argparse.Namespace, keys, status) -> None:
    """The operator typed h: stop the policy, hold, print the plan, then (confirmed) move to HOME."""
    print("HOME requested: the policy is stopped.", flush=True)
    if args.execute:
        node.publish_hold()
    status(phase="HOLD")
    current = {part: node.last_command(part) for part in HOME_POSE}
    for line in pose_plan(current, HOME_POSE, "home"):
        print(line, flush=True)
    if not args.execute:
        print("Dry-run: nothing is published; the robot is not moved.", flush=True)
        return
    if not args.yes:
        print(
            f"Enter 'h' to move to HOME now ({args.home_duration:.1f}s); anything else keeps holding: ",
            end="",
            flush=True,
        )
        if keys.wait() != "h":
            print("Not moving; the robot holds.", flush=True)
            return
    print(f"Moving to the HOME pose ({args.home_duration:.1f}s) ...", flush=True)
    status(phase="HOME POSE")
    node.move_to_pose(
        HOME_POSE,
        duration_s=args.home_duration,
        rate_hz=args.ready_pose_rate,
        max_state_age=args.max_sensor_age,
    )
    status(phase="HOME")
    print("HOME pose reached.", flush=True)


def main() -> None:
    args = parse_args()
    run_dir = None
    if not (args.no_log and args.no_record):
        root = Path(args.log_dir).expanduser() if args.log_dir else default_log_root()
        run_dir = root / (time.strftime("%Y%m%d_%H%M%S") + ("_execute" if args.execute else "_dry"))
        run_dir.mkdir(parents=True)
    console = ConsoleTee(run_dir / "console.log") if run_dir else None
    try:
        run(args, run_dir)
    except Exception:
        traceback.print_exc()  # into console.log as well
        sys.exit(1)
    finally:
        if console:
            console.close()


def run(args: argparse.Namespace, run_dir: Optional[Path]) -> None:
    if run_dir:
        print(f"Run directory: {run_dir}", flush=True)
    topics = yaml.safe_load(Path(args.topics).read_text())
    host, port = parse_address(args.server)
    print(f"Connecting to the policy server {host}:{port} ...", flush=True)
    client = PolicyClient(host, port, timeout=args.server_timeout)
    meta = client.metadata
    if meta.get("mock_encoders"):
        if args.execute:
            raise RuntimeError(
                "the policy server runs --mock-encoders (meaningless actions); refusing --execute"
            )
        print("WARNING: the server runs mock encoders; this dry-run only tests the plumbing.", flush=True)
    n_obs, n_action = int(meta["n_obs_steps"]), int(meta["n_action_steps"])
    execute_steps = n_action if args.execute_steps is None else args.execute_steps
    if not 1 <= execute_steps <= n_action:
        raise ValueError(f"--execute-steps must be in [1, {n_action}], got {execute_steps}")
    fps = float(meta["fps"])
    task = args.task or meta["default_task"]
    flux_keys = client.camera_keys
    cameras = {key: camera_key(key) for key in flux_keys}  # images.head -> head
    print(
        f"Policy server ready: checkpoint={meta['checkpoint']} gpu={meta['gpu']} n_obs_steps={n_obs} "
        f"n_action_steps={n_action} chunk={meta['chunk_size']} sampler_steps={meta['num_inference_steps']} "
        f"guidance={meta['guidance_scale']} seed_mode={args.seed_mode} task={task!r}",
        flush=True,
    )

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from robot_ros2 import AstribotNode

    rclpy.init()
    node = AstribotNode(topics, list(cameras.values()))
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, name="astribot-ros2", daemon=True)
    spin.start()
    recorder = None
    if run_dir and not args.no_record:
        recorder = VideoRecorder(
            run_dir / "video",
            list(cameras.values()),
            args.record_fps or fps,
            grid=not args.no_record_grid,
            title=Path(meta["checkpoint"]).name,
            mode="EXECUTE" if args.execute else "DRY-RUN",
            metadata={
                "checkpoint": meta["checkpoint"],
                "task": task,
                "server": args.server,
                "execute": args.execute,
                "execute_steps": execute_steps,
                "seed_mode": args.seed_mode,
            },
        )
        node.attach_recorder(recorder, recorder.fps)

    def status(**fields) -> None:
        if recorder:
            recorder.set_status(**fields)

    limiter = ActionLimiter(
        args.max_joint_delta, args.max_gripper_delta, enforce_joint_limits=not args.disable_joint_limits
    )
    streak = CriticalStreak(args.tracking_critical_limit)
    log = keys = None
    commands_started = False
    inference_times, tracking_max, warnings, criticals, tracked = [], 0.0, 0, 0, 0

    def observe():
        stale = {k: a for k, a in node.ages().items() if a > args.max_sensor_age}
        if stale:
            raise RuntimeError(
                "stale Astribot input(s): " + ", ".join(f"{k}={a:.2f}s" for k, a in stale.items())
            )
        frames, state = node.sample()
        return {flux: frames[ros] for flux, ros in cameras.items()}, state

    def seed_for(cycle: int):
        return args.seed + cycle if args.seed_mode == "cycle" else None

    print(
        f"Mode={'EXECUTE' if args.execute else 'DRY-RUN'} control={fps:.0f}Hz execute_steps={execute_steps} "
        f"arm_stream={node.command_hz:.0f}Hz arm_velocity_limit={node.arm_velocity_limit:.1f}rad/s "
        f"ready_pose={'yes' if args.prepare_ready_pose else 'no'} critical_stop={args.tracking_critical_limit}",
        flush=True,
    )
    if not args.execute:
        print("No command will be published. Add --execute only after a clean dry-run.", flush=True)
    try:
        wait_ready(node, args.ready_timeout)
        for ros_key, hw in sorted(node.native_hw.items()):
            note = aspect_warning(hw)
            print(
                f"Camera {ros_key}: native {hw[1]}x{hw[0]} -> policy 640x360"
                + (f"  WARNING: {note}" if note else ""),
                flush=True,
            )
        check_state(node.sample()[1], meta, args.execute, args.allow_out_of_range_state)

        if args.execute and not args.yes:
            extra = (
                " It first moves torso, arms, grippers and head to the ready pose."
                if args.prepare_ready_pose
                else ""
            )
            try:
                answer = (
                    input(f"Enter 's' to start FLUX inference and publish commands.{extra} ").strip().lower()
                )
            except EOFError:
                answer = ""
            if answer != "s":
                print("Aborted before publishing any command.", flush=True)
                return
        if recorder:
            recorder.start()
            print(
                f"Recording video: {recorder.directory} ({recorder.codec}, {recorder.fps:.0f} fps)",
                flush=True,
            )
        keys = operator_keys(args.execute)

        first_command = None
        if args.prepare_ready_pose:
            print(f"Moving to the ready pose ({args.ready_pose_duration:.1f}s) ...", flush=True)
            status(phase="READY POSE")
            commands_started = True
            node.move_to_pose(
                READY_POSE,
                duration_s=args.ready_pose_duration,
                rate_hz=args.ready_pose_rate,
                max_state_age=args.max_sensor_age,
            )
            settled = time.monotonic()
            wait_ready(node, args.ready_timeout, since=settled)  # fresh frames/states from the pose
            first_command = pack_parts(
                {p: READY_POSE[p] for p in ("left_arm", "right_arm", "left_gripper", "right_gripper")}
            )
            print("Ready pose reached; the policy history starts now.", flush=True)
            check_state(node.sample()[1], meta, args.execute, args.allow_out_of_range_state)

        if not args.no_warmup:
            status(phase="WARM-UP")
            warm = TickHistory(n_obs, flux_keys)
            frames, state = observe()
            warm.append(frames, state, first_command)
            started = time.monotonic()
            _, reply = client.infer(warm.window(), task, seed_for(0))
            print(
                f"Warm-up prediction discarded (server {reply['inference_s']:.2f}s, "
                f"round trip {time.monotonic() - started:.2f}s, peak GPU {reply['peak_gpu_gb']:.1f}GB)",
                flush=True,
            )

        if not args.no_log:
            log_meta = {"server": meta, "args": vars(args), "task": task, "cameras": cameras}
            log = EpisodeLog(run_dir, list(flux_keys), fps, log_meta, windows=not args.no_log_windows)

        history = TickHistory(n_obs, flux_keys)
        queue: deque = deque()
        last_command = first_command  # the command in force before the coming tick
        previous_command = None  # for the tracking diagnostic
        cycle, dt = 0, 1.0 / fps
        home = False
        deadline = time.monotonic()
        while rclpy.ok():
            key = keys.poll() if keys else None
            if key == "h":
                home = True
                break
            if key is not None:
                print(
                    f"Ignored input {key!r}: h + Enter stops and goes HOME, Ctrl-C stops and holds.",
                    flush=True,
                )
            frames, state = observe()
            # Dry-run publishes nothing: the measured state (the robot holds still) is the truthful command.
            command_before = last_command if args.execute else None
            history.append(frames, state, command_before)
            tick = log.tick(frames, state, command_before) if log else None

            if not queue:
                if args.max_cycles is not None and cycle >= args.max_cycles:
                    break
                window = history.window()
                anchor = window["command_history"][-1]
                seed = seed_for(cycle)
                status(phase="INFER", cycle=cycle)
                started = time.monotonic()
                actions, reply = client.infer(window, task, seed)
                round_trip = time.monotonic() - started
                inference_times.append(reply["inference_s"])
                chunk = actions[:execute_steps]
                if log:
                    log.cycle(tick, window, actions, seed, reply["inference_s"], round_trip)
                motion = max(arm_max_abs(a, anchor) for a in chunk)
                seed_text = seed if seed is not None else "export"
                print(
                    f"cycle={cycle:04d} inference={reply['inference_s']:.2f}s round_trip={round_trip:.2f}s "
                    f"seed={seed_text} first_step={arm_max_abs(chunk[0], anchor):.3f}rad "
                    f"chunk_arm_motion={motion:.3f}rad "
                    f"grippers=[{chunk[0][14]:.1f}, {chunk[0][15]:.1f}]->[{chunk[-1][14]:.1f}, {chunk[-1][15]:.1f}]",
                    flush=True,
                )
                status(phase="RUN", note=f"chunk motion {motion:.3f} rad   seed {seed_text}")
                queue.extend(chunk)
                cycle += 1
                observe()  # a stream may have stopped while the server was sampling
                deadline = time.monotonic()  # the robot held its target meanwhile; restart the tick clock

            raw = queue.popleft()
            if args.execute:
                measured = node.sample()[1].astype(np.float64)
                command = limiter.limit(raw, measured if last_command is None else last_command)
                if previous_command is not None:
                    tracking = arm_tracking(
                        previous_command,
                        measured,
                        args.tracking_warn_threshold,
                        args.tracking_critical_threshold,
                    )
                    tracked += 1
                    tracking_max = max(tracking_max, tracking.maximum)
                    stop = streak.observe(tracking.severity)
                    detail = f"max={tracking.maximum:.3f}rad joint={tracking.worst_joint} rms={tracking.rms:.3f}rad"
                    if tracking.severity == "critical":
                        criticals += 1
                        print(
                            f"CRITICAL tracking lag step={tracked}: {detail} streak={streak.count}",
                            flush=True,
                        )
                    elif tracking.severity == "warning":
                        warnings += 1
                        print(f"WARNING tracking lag step={tracked}: {detail}", flush=True)
                    elif tracked % args.tracking_log_interval == 0:
                        print(f"tracking step={tracked}: {detail}", flush=True)
                    if stop:
                        raise RuntimeError(f"stopped: {streak.count} consecutive critical tracking samples")
                node.set_target(command)
                commands_started = True
                previous_command = command.copy()
                last_command = command
                if log:
                    log.executed(tick, command)
            deadline += dt
            if time.monotonic() - deadline > dt:
                deadline = time.monotonic()  # a slow tick must not be followed by a burst of commands
            time.sleep(max(0.0, deadline - time.monotonic()))
        if home:
            commands_started = commands_started or args.execute
            go_home(node, args, keys, status)
    except KeyboardInterrupt:
        print("Interrupted; stopping.", flush=True)
    finally:
        if commands_started:
            try:
                node.publish_hold()
            except Exception as error:
                print(f"WARNING: could not publish the final hold: {error}", flush=True)
        if keys:
            keys.close()
        if inference_times:
            print(
                f"Inference summary: cycles={len(inference_times)} mean={np.mean(inference_times):.2f}s "
                f"max={np.max(inference_times):.2f}s",
                flush=True,
            )
        if tracked:
            print(
                f"Tracking summary: samples={tracked} warnings={warnings} critical={criticals} "
                f"max_seen={tracking_max:.3f}rad",
                flush=True,
            )
        if recorder and recorder.is_recording:
            video = recorder.stop()
            print(
                f"Video saved: {video['directory']} ({video['duration_s']:.0f}s, {video['frames']} frames, "
                f"{video['dropped_frames']} dropped, {video['codec']})",
                flush=True,
            )
        if log:
            print(f"Episode log saved: {log.close()}", flush=True)
        client.close()
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin.join(timeout=2.0)


if __name__ == "__main__":
    main()
