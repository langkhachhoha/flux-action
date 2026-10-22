#!/usr/bin/env python3
"""Score an Astribot export on the held-out coffee episodes, with the frozen encoders of this machine.

Same windows and metrics as ``flux-action evaluate`` in export_eval.sh (history of 8 ticks, the export's
sampler, original 360x640 videos), but the video VAE and text encoder are loaded from local paths: the
export's config names the training machine's paths. Builds the dataset index on first use.

    .venv/bin/python scripts/astribot/evaluate_export.py \
        --checkpoint checkpoint/infer/step-2300-model-bf16 --windows-per-episode 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from policy_server import PROJECT_ROOT, load_policy

from flux_action.inference.evaluate import evaluate_policy  # noqa: E402  (path set by policy_server)

DEFAULT_DATASET = "/nvme/hieuhm19/datasets/serving_brewed_coffee_lrb_annotated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-root", default=DEFAULT_DATASET, help="LeRobot v3 coffee dataset")
    parser.add_argument("--index-dir", default=str(PROJECT_ROOT / "outputs/astribot/index"))
    parser.add_argument("--val-episodes", type=int, default=20, help="Held out when building the index")
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--num-inference-steps", type=int, default=None, help="Override the export's sampler steps"
    )
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override the export's guidance")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-vae", default=str(PROJECT_ROOT / "outputs/weights/video_vae.safetensors"))
    parser.add_argument("--text-encoder", default=str(PROJECT_ROOT / "outputs/weights/text_encoder"))
    parser.add_argument(
        "--output", default=None, help="Report JSON (default: <checkpoint>/eval_val_local.json)"
    )
    return parser.parse_args()


def ensure_index(args: argparse.Namespace, chunk_size: int) -> None:
    if (Path(args.index_dir) / "manifest.json").is_file():
        return
    # The arguments of scripts/astribot/prepare.sh, so the split equals the training run's.
    command = [
        str(PROJECT_ROOT / ".venv/bin/flux-action"), "index-lerobot",
        "--source-root", args.source_root, "--output-dir", args.index_dir,
        "--camera", "head=observation.images.cam_head",
        "--camera", "left_wrist=observation.images.cam_left_wrist",
        "--camera", "right_wrist=observation.images.cam_right_wrist",
        "--action-parameterization", "joint_delta", "--absolute-action-dims", "14,15",
        "--chunk-size", str(chunk_size), "--val-episodes", str(args.val_episodes),
        "--dataset-id", Path(args.source_root).name,
    ]  # fmt: skip
    print("Building the dataset index:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    ensure_index(args, config["chunk_size"])
    policy = load_policy(
        args.checkpoint,
        args.device,
        args.video_vae,
        args.text_encoder,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    report = evaluate_policy(
        policy,
        args.source_root,
        args.index_dir,
        split="val",
        windows_per_episode=args.windows_per_episode,
        max_windows=args.max_windows,
        frame_hw=(360, 640),
    )
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    output = Path(args.output) if args.output else Path(args.checkpoint) / "eval_val_local.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n")
    rmse = np.sqrt(np.asarray(report["action_mse_raw_per_channel"]))
    names = report.get("action_names") or [str(i) for i in range(len(rmse))]
    print(
        f"Held-out windows={report['n_windows']} episodes={report['episodes']} "
        f"action_mse_normalized={report['action_mse_normalized']:.4f} "
        f"(median {report['action_mse_normalized_median']:.4f}) in {report['seconds']:.0f}s"
    )
    print(
        f"Arm RMSE over the {config['chunk_size']}-step chunk: mean {rmse[:14].mean():.4f}rad, "
        f"max {rmse[:14].max():.4f}rad ({names[int(np.argmax(rmse[:14]))]}); "
        f"gripper RMSE {rmse[14]:.1f} / {rmse[15]:.1f} (0-100)"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    sys.exit(main())
