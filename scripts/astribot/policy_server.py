#!/usr/bin/env python3
"""Serve an Astribot FLUX Action export to the ROS 2 client over a local socket.

    .venv/bin/python scripts/astribot/policy_server.py \
        --checkpoint checkpoint/infer/step-2300-model-bf16 --device cuda:0

The robot client (robot_client.py, ROS 2 Humble's Python 3.10) keeps the control loop, the ROS topics and
every safety check; this process only holds the model. Each request is one history window, exactly what
``FluxActionPolicy.select_action`` would assemble at a replanning tick:

    state            float32 (n_obs_steps, 16)   measured states of ticks t-7..t
    command_history  float32 (n_obs_steps, 16)   absolute commands issued before each of those ticks
    images.<camera>  uint8   (n_obs_steps, 360, 640, 3) RGB frames of ticks t-7..t

plus optional ``task`` and ``seed`` (the sampler's noise seed; default: the export's ``inference_seed``).
The reply is ``actions`` float32 (chunk_size, 16): absolute commands in dataset units (arm deltas integrated
onto the last command, grippers absolute). See ``astribot_wire`` for the framing.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from astribot_wire import recv_message, send_message
from robot_runtime import FRAME_HW, STATE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flux_action.models.text_encoder import load_text_encoder  # noqa: E402
from flux_action.models.video_vae import load_video_vae  # noqa: E402
from flux_action.policy import FluxActionPolicy  # noqa: E402

DEFAULT_TASK = "serving brewed coffee"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", required=True, help="Export directory (config.json, model.safetensors)"
    )
    parser.add_argument("--video-vae", default=str(PROJECT_ROOT / "outputs/weights/video_vae.safetensors"))
    parser.add_argument("--text-encoder", default=str(PROJECT_ROOT / "outputs/weights/text_encoder"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--task", default=DEFAULT_TASK, help="Instruction used when a request names none")
    parser.add_argument(
        "--num-inference-steps", type=int, default=None, help="Override the export's sampler steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override the export's guidance (1.0 = one DiT pass per step, half the time)",
    )
    parser.add_argument(
        "--keep-text-encoder-on-gpu",
        action="store_true",
        help="Keep Qwen3-VL on the GPU (+8.9 GB); by default it visits the GPU only for a new instruction",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip the synthetic warm-up prediction")
    parser.add_argument(
        "--mock-encoders",
        action="store_true",
        help="Random video VAE and mock text encoder: tests plumbing, memory and latency without the "
        "encoder weights. Actions are meaningless; the robot client refuses --execute.",
    )
    return parser.parse_args()


def load_policy(
    checkpoint: str,
    device: str,
    video_vae: str,
    text_encoder: str,
    *,
    mock_encoders: bool = False,
    keep_text_encoder_on_gpu: bool = False,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
) -> FluxActionPolicy:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{device} requested but CUDA is unavailable")
    started = time.perf_counter()
    # The export's config names the encoders by the training machine's paths; load them from here.
    policy = FluxActionPolicy.from_pretrained(
        checkpoint,
        video_vae=load_video_vae(None if mock_encoders else video_vae),
        text_encoder=load_text_encoder(None if mock_encoders else text_encoder),
        device="cpu",
    )
    if not keep_text_encoder_on_gpu:
        policy.set_text_encoder_offload(True)
    policy.to(device).eval()
    if num_inference_steps is not None:
        policy.config.num_inference_steps = num_inference_steps
    if guidance_scale is not None:
        policy.config.guidance_scale = guidance_scale
    policy.config.validate_inference()
    cfg = policy.config
    if cfg.inference_profile != "history" or cfg.action_dim != 16 or cfg.camera_layout != "droid":
        raise ValueError("this server expects the Astribot history export (16-D actions, droid layout)")
    if mock_encoders:
        print("WARNING: mock encoders loaded; predicted actions are meaningless.", flush=True)
    print(
        f"Loaded {checkpoint} in {time.perf_counter() - started:.1f}s on {device}: "
        f"cameras={list(cfg.camera_keys)} n_obs_steps={cfg.n_obs_steps} chunk={cfg.chunk_size} "
        f"n_action_steps={cfg.n_action_steps} sampler={cfg.sampler}x{cfg.num_inference_steps} "
        f"guidance={cfg.guidance_scale} dtype={cfg.torch_dtype}",
        flush=True,
    )
    return policy


def metadata(policy: FluxActionPolicy, args: argparse.Namespace) -> dict:
    cfg = policy.config
    device = torch.device(args.device)
    return {
        "type": "hello",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "camera_keys": list(cfg.camera_keys),
        "frame_hw": list(FRAME_HW),
        "n_obs_steps": cfg.n_obs_steps,
        "n_action_steps": cfg.n_action_steps,
        "chunk_size": cfg.chunk_size,
        "action_dim": cfg.action_dim,
        "fps": cfg.fps,
        "action_parameterization": cfg.action_parameterization,
        "absolute_action_dims": list(cfg.absolute_action_dims),
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "inference_seed": cfg.inference_seed,
        "state_names": list(STATE_NAMES),
        "state_normalization": {q: list(v) for q, v in cfg.state_normalization.items()},
        "action_normalization": {q: list(v) for q, v in cfg.action_normalization.items()},
        "default_task": args.task,
        "mock_encoders": args.mock_encoders,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def predict(
    policy: FluxActionPolicy, arrays: dict[str, np.ndarray], task: str, seed: int | None = None
) -> tuple[np.ndarray, float]:
    cfg = policy.config
    window = (cfg.n_obs_steps, cfg.action_dim)
    for key in ("state", "command_history"):
        value = arrays.get(key)
        if value is None or value.shape != window or value.dtype != np.float32:
            raise ValueError(f"{key} must be float32 {window}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or infinity")
    device = policy.device
    batch = {
        "state": torch.from_numpy(arrays["state"])[None].to(device),
        "command_history": torch.from_numpy(arrays["command_history"])[None].to(device),
        "task": [task],
    }
    for key in cfg.camera_keys:
        frames = arrays.get(key)
        expected = (cfg.n_obs_steps, *FRAME_HW, 3)
        if frames is None or frames.shape != expected or frames.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8 {expected}")
        batch[key] = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    default_seed = cfg.inference_seed
    if seed is not None:
        cfg.inference_seed = int(seed)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch)[0]
    finally:
        cfg.inference_seed = default_seed
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    actions = actions.float().cpu().numpy()
    if actions.shape != (cfg.chunk_size, cfg.action_dim) or not np.isfinite(actions).all():
        raise RuntimeError(f"policy returned invalid actions {actions.shape}")
    return actions, elapsed


def warmup(policy: FluxActionPolicy, task: str) -> None:
    """One synthetic prediction: encodes and caches the instruction and allocates the CUDA workspaces."""
    cfg = policy.config
    state = np.tile(
        (np.asarray(cfg.state_normalization["q01"]) + np.asarray(cfg.state_normalization["q99"])) / 2,
        (cfg.n_obs_steps, 1),
    ).astype(np.float32)
    arrays = {"state": state, "command_history": state.copy()}
    for key in cfg.camera_keys:
        arrays[key] = np.full((cfg.n_obs_steps, *FRAME_HW, 3), 128, dtype=np.uint8)
    started = time.perf_counter()
    _, elapsed = predict(policy, arrays, task)
    print(
        f"Warm-up done in {time.perf_counter() - started:.1f}s (last sampling {elapsed:.2f}s)",
        flush=True,
    )


def serve_connection(conn: socket.socket, policy: FluxActionPolicy, args: argparse.Namespace) -> None:
    send_message(conn, metadata(policy, args))
    while True:
        try:
            meta, arrays = recv_message(conn)
        except ConnectionError:
            return
        kind = meta.get("type")
        if kind == "ping":
            send_message(conn, {"type": "pong"})
            continue
        if kind != "infer":
            send_message(conn, {"type": "error", "message": f"unknown request type {kind!r}"})
            continue
        task = str(meta.get("task") or args.task)
        seed = meta.get("seed")
        try:
            actions, elapsed = predict(policy, arrays, task, None if seed is None else int(seed))
        except Exception:
            message = traceback.format_exc()
            print(message, file=sys.stderr, flush=True)
            send_message(conn, {"type": "error", "message": message})
            continue
        peak = torch.cuda.max_memory_allocated(policy.device) / 2**30 if policy.device.type == "cuda" else 0.0
        send_message(
            conn,
            {"type": "actions", "inference_s": elapsed, "task": task, "seed": seed, "peak_gpu_gb": peak},
            {"actions": actions},
        )


def main() -> None:
    args = parse_args()
    policy = load_policy(
        args.checkpoint,
        args.device,
        args.video_vae,
        args.text_encoder,
        mock_encoders=args.mock_encoders,
        keep_text_encoder_on_gpu=args.keep_text_encoder_on_gpu,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    if not args.no_warmup:
        warmup(policy, args.task)
    with socket.create_server((args.host, args.port)) as server:
        print(f"Policy server listening on {args.host}:{args.port} (task={args.task!r})", flush=True)
        while True:
            conn, address = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Client connected from {address[0]}:{address[1]}", flush=True)
            with conn:
                try:
                    serve_connection(conn, policy, args)
                except Exception:  # a broken or malformed client must not stop the server
                    print(f"Connection error:\n{traceback.format_exc()}", flush=True)
            print("Client disconnected; waiting for the next one.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Policy server stopped.", flush=True)
