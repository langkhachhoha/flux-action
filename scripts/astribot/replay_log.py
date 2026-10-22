#!/usr/bin/env python3
"""Replay an episode log of robot_client.py through the model, and compare its scene with the training data.

    .venv/bin/python scripts/astribot/replay_log.py --log ~/Desktop/astribot_logs/flux_action/<run> \
        --checkpoint checkpoint/infer/step-2300-model-bf16 --device cuda:1 --seeds 0,1,2,3

For every logged prediction the request is rebuilt from the log (state/command windows and the lossless
request frames of ``windows/``; logs without them fall back to the MP4s, whose compression alone can change a
prediction near a decision such as when to start moving) and sampled again: with the logged seed it must
reproduce the robot's chunk (a mismatch means the client and the model disagree on the data format); with
other seeds it shows how much the motion depends on the noise draw. ``scene.jpg`` puts the first logged frames next
to the first frames of the held-out episodes that start closest to the logged state: the table, objects and
camera angles should look alike, otherwise the policy sees a scene it was not trained on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from policy_server import PROJECT_ROOT, load_policy, predict

from flux_action.data.droid.video import decode_window  # noqa: E402  (path set by policy_server)

DEFAULT_DATASET = "/nvme/hieuhm19/datasets/serving_brewed_coffee_lrb_annotated"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", required=True, help="episode log directory written by robot_client.py")
    p.add_argument("--checkpoint", default=None, help="export (default: the one the log was recorded with)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="", help="extra sampler seeds to try, e.g. 0,1,2,3")
    p.add_argument("--max-cycles", type=int, default=None)
    p.add_argument("--no-model", action="store_true", help="only write scene.jpg")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--index-dir", default=str(PROJECT_ROOT / "outputs/astribot/index"))
    p.add_argument("--video-vae", default=str(PROJECT_ROOT / "outputs/weights/video_vae.safetensors"))
    p.add_argument("--text-encoder", default=str(PROJECT_ROOT / "outputs/weights/text_encoder"))
    return p.parse_args()


def read_frames(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def arm_motion(chunk: np.ndarray, anchor: np.ndarray, steps: int) -> float:
    return float(np.abs(chunk[:steps, :14] - anchor[:14]).max())


def scene_sheet(log: Path, cameras: list[str], first_state: np.ndarray, dataset: str, index_dir: str) -> Path:
    """First logged frames (top row) above the first frames of the two closest-starting held-out episodes."""
    from PIL import Image

    manifest = json.loads((Path(index_dir) / "manifest.json").read_text())
    rows = np.load(Path(index_dir) / manifest["rows"]["file"], mmap_mode="r")
    episodes = [e for e in manifest["episodes"] if e["split"] == "val"]
    distance = [float(np.abs(rows[e["from_index"], :14] - first_state[:14]).max()) for e in episodes]
    closest = [episodes[i] for i in np.argsort(distance)[:2]]
    sheet = [
        np.concatenate(
            [np.asarray(Image.open(log / f"first_frames_{c}.png").convert("RGB")) for c in cameras], 1
        )
    ]
    for episode in closest:
        views = []
        for camera in cameras:
            video = episode["videos"][camera.removeprefix("images.")]
            views.append(
                decode_window(f"{dataset}/{video['file']}", video["first_frame"], 1, fps=manifest["fps"],
                              frame_hw=(360, 640), decoder="pyav", resize=True)[0]
            )  # fmt: skip
        sheet.append(np.concatenate(views, 1))
    path = log / "scene.jpg"
    Image.fromarray(np.concatenate(sheet, 0)).resize((1440, 270 * len(sheet))).save(path, quality=90)
    names = ", ".join(
        f"episode {e['episode_index']} ({distance[episodes.index(e)]:.3f} rad from the logged start)"
        for e in closest
    )
    print(f"Scene sheet: {path} (top: robot; below: {names})")
    return path


def main() -> None:
    args = parse_args()
    log = Path(args.log).expanduser()
    meta = json.loads((log / "meta.json").read_text())
    server = meta["server"]
    cameras = list(server["camera_keys"])
    ticks = np.load(log / "ticks.npz")
    scene_sheet(log, cameras, ticks["state"][0], args.dataset, args.index_dir)
    if args.no_model:
        return
    cycles = np.load(log / "cycles.npz")
    frames = None  # MP4 frames, only for cycles without a lossless window
    n_obs, steps = int(server["n_obs_steps"]), int(meta["args"]["execute_steps"] or server["n_action_steps"])
    policy = load_policy(
        args.checkpoint or server["checkpoint"],
        args.device,
        args.video_vae,
        args.text_encoder,
        num_inference_steps=server["num_inference_steps"],
        guidance_scale=server["guidance_scale"],
    )
    extra = [int(s) for s in args.seeds.split(",") if s]
    report = []
    for c in range(len(cycles["tick"]))[: args.max_cycles]:
        tick = int(cycles["tick"][c])
        arrays = {"state": cycles["state_window"][c], "command_history": cycles["command_window"][c]}
        window = log / "windows" / f"cycle_{c:04d}.npz"
        if window.is_file():
            with np.load(window) as saved:
                arrays.update({key: saved[key] for key in cameras})
            source = "exact"
        else:
            frames = frames or {key: read_frames(log / f"{key}.mp4") for key in cameras}
            picks = [max(0, t) for t in range(tick - n_obs + 1, tick + 1)]  # the first tick fills the history
            if any(t >= len(frames[k]) for k in cameras for t in picks):
                print(f"cycle {c}: frames of tick {tick} are missing from the MP4s; stopping")
                break
            arrays.update({key: np.ascontiguousarray(frames[key][picks]) for key in cameras})
            source = "mp4"
        anchor = arrays["command_history"][-1]
        logged = cycles["chunk"][c]
        seed = int(cycles["seed"][c])
        replay, _ = predict(policy, arrays, meta["task"], None if seed < 0 else seed)
        row = {
            "cycle": c,
            "tick": tick,
            "frames": source,
            "logged_motion": arm_motion(logged, anchor, steps),
            "replay_motion": arm_motion(replay, anchor, steps),
            "replay_vs_logged_max_abs": float(np.abs(replay[:steps, :14] - logged[:steps, :14]).max()),
            "seeds": {
                s: arm_motion(predict(policy, arrays, meta["task"], s)[0], anchor, steps) for s in extra
            },
        }
        report.append(row)
        seeds = " ".join(f"seed{s}={m:.3f}" for s, m in row["seeds"].items())
        print(
            f"cycle={c:04d} tick={tick} frames={source} logged={row['logged_motion']:.3f}rad "
            f"replay={row['replay_motion']:.3f}rad "
            f"|replay-logged|max={row['replay_vs_logged_max_abs']:.4f}rad {seeds}",
            flush=True,
        )
    (log / "replay.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"Report: {log / 'replay.json'}")


if __name__ == "__main__":
    main()
