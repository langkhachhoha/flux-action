"""Upload <output_dir>/metrics.jsonl to Weights & Biases from a process separate from training.

Training writes every record to metrics.jsonl, the source of truth. This uploader pushes them to the W&B run
whose id is in <output_dir>/wandb_run_id (created if missing), resuming that run. Every cycle asks the W&B API
what the run already holds (the last train / val / checkpoint step), uploads only the rest in a child process
with a timeout, and repeats. A proxy or network outage (the office gateway asking to log in again, expired
sessions, SSL interception) therefore costs nothing: the next cycle after the network is back fills the gap,
and training never waits on W&B.

    python scripts/astribot/wandb_sync.py --output-dir outputs/astribot/<run> --project astribot_making_coffee
        [--entity ...] [--name ...] [--interval 120] [--follow-pid <launcher pid>] [--once]

With --follow-pid it uploads a last time and exits once that process has ended.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from flux_action.training.trainer import WANDB_ID_FILENAME, wandb_kind, wandb_payload

KIND_KEYS = {"train": "train/loss", "val": "val/loss", "checkpoint": "train/checkpoint_seconds"}


def log(message: str, **fields) -> None:
    print(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "msg": message, **fields}), flush=True)


def run_id(output_dir: Path) -> str:
    path = output_dir / WANDB_ID_FILENAME
    if path.is_file() and path.read_text().strip():
        return path.read_text().strip()
    output_dir.mkdir(parents=True, exist_ok=True)
    value = secrets.token_hex(4)
    path.write_text(value + "\n")
    return value


def records(output_dir: Path) -> list[dict]:
    path = output_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line being written right now
            break
    return out


def remote_progress(entity: str | None, project: str, rid: str) -> dict[str, int]:
    """Last ``train/step`` the run holds per series; zeros for a run that does not exist yet."""
    import wandb

    api = wandb.Api(timeout=60)
    path = f"{entity or api.default_entity}/{project}/{rid}"
    try:
        run = api.run(path)
    except wandb.errors.CommError as error:
        if "not found" in str(error).lower() or "404" in str(error):
            return dict.fromkeys(KIND_KEYS, 0)
        raise
    progress = dict.fromkeys(KIND_KEYS, 0)
    for kind, key in KIND_KEYS.items():
        for row in run.scan_history(keys=["train/step", key], page_size=2000):
            if row.get(key) is not None and row.get("train/step") is not None:
                progress[kind] = max(progress[kind], int(row["train/step"]))
    return progress


def run_config(output_dir: Path) -> dict:
    """Training and policy configuration from the newest checkpoint (the trainer's resolved settings)."""
    steps = sorted(
        (int(p.name.removeprefix("step-")), p) for p in output_dir.glob("step-*") if (p / "COMPLETE").exists()
    )
    if not steps:
        return {}
    path = steps[-1][1]
    state = json.loads((path / "state.json").read_text())
    return {
        **state.get("extra", {}).get("train_config", {}),
        "policy_resolved": json.loads((path / "config.json").read_text()),
        "global_batch": state.get("extra", {}).get("global_batch"),
        "world_size": state.get("world_size"),
    }


def upload_once(args) -> int:
    """One cycle: compare with the run, upload the missing records, finish. Returns the records sent."""
    import wandb

    output_dir = Path(args.output_dir)
    rid = run_id(output_dir)
    done = remote_progress(args.entity, args.project, rid)
    pending = [
        r for r in records(output_dir) if (k := wandb_kind(r)) is not None and int(r["step"]) > done[k]
    ]
    if not pending:
        log("up to date", run_id=rid, remote=done)
        return 0
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        id=rid,
        resume="allow",
        dir=str(output_dir),
        config=run_config(output_dir),
        settings=wandb.Settings(init_timeout=120, console="off", x_disable_stats=True, save_code=False),
    )
    run.define_metric("train/step")
    run.define_metric("*", step_metric="train/step")
    for record in pending:
        run.log(wandb_payload(record))
    run.finish()
    log("uploaded", run_id=rid, records=len(pending), remote_before=done, last_step=int(pending[-1]["step"]))
    return len(pending)


def alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--interval", type=float, default=120.0, help="seconds between upload cycles")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds before a stuck cycle is killed")
    parser.add_argument("--follow-pid", type=int, default=None, help="exit after this process ends")
    parser.add_argument("--once", action="store_true", help="run a single cycle in this process")
    args = parser.parse_args()
    os.environ.setdefault("WANDB_SILENT", "true")
    if args.once:
        upload_once(args)
        return
    child = [sys.executable, os.path.abspath(__file__), "--once"]
    for name in ("output_dir", "project", "entity", "name"):
        value = getattr(args, name)
        if value is not None:
            child += [f"--{name.replace('_', '-')}", str(value)]
    log("uploader started", output_dir=args.output_dir, project=args.project, follow_pid=args.follow_pid)
    while True:
        last_round = not alive(args.follow_pid)
        try:
            result = subprocess.run(child, timeout=args.timeout, capture_output=True, text=True)
            for line in (result.stdout + result.stderr).splitlines()[-5:]:
                print(line, flush=True)
            if result.returncode != 0:
                log("cycle failed; retrying next cycle", returncode=result.returncode)
        except subprocess.TimeoutExpired:
            log("cycle timed out (network?); retrying next cycle", timeout=args.timeout)
        if last_round:
            log("followed process ended; uploader exits")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
