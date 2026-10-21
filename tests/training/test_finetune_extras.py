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
"""Held-out evaluation, the STOP file and W&B logging of the trainer, on a history policy over LeRobot."""

import json
import math

import pytest
import torch
from conftest import TINY_DIT, FakeVideoVAE
from synthetic_lerobot import CAMERAS, DIM, build_v3_dataset

from flux_action.data.lerobot import index as lerobot_index
from flux_action.models.text_encoder import MockTextEncoder
from flux_action.policy import FluxActionPolicy
from flux_action.training.checkpoint import export_policy, latest_checkpoint
from flux_action.training.trainer import STOP_FILENAME, STOPPED_FILENAME, TrainConfig, Trainer

CHUNK = 8
EPISODES = [
    {"n_frames": n, "caption": c} for n, c in ((40, "pick"), (38, "place"), (44, "pick"), (41, "stack"))
]
POLICY = dict(
    camera_layout="side_by_side",
    camera_keys=["images.top", "images.wrist"],
    canvas_hw=[64, 128],
    action_dim=DIM,
    gripper_flip_dims=[],
    action_parameterization="joint_delta",
    absolute_action_dims=[-1],
    inference_profile="history",
    n_obs_steps=3,
    history_snapshots=2,
    condition_on_past_actions=True,
    chunk_size=CHUNK,
    n_action_steps=4,
    text_fixed_length=16,
    separate_timesteps=True,
    conditioning_noise_max=0.2,
    loss_reduction="modalities",
    action_loss_weight=0.5,
    dit_config=TINY_DIT,
    augment=False,
    sampler="euler",
    num_inference_steps=2,
    guidance_scale=2.0,
    sampler_shift=5.0,
)


def tiny_policy(config):
    return FluxActionPolicy(config, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32))


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("finetune_extras")
    source = build_v3_dataset(root / "source", [dict(e) for e in EPISODES])
    lerobot_index.index_dataset(source, root / "index", CAMERAS, chunk_size=CHUNK, val_episodes=1)
    return source, root / "index"


def make_config(indexed, output, **overrides):
    source, index = indexed
    values = dict(
        source_root=str(source),
        index_dir=str(index),
        output_dir=str(output),
        policy=POLICY,
        steps=4,
        cooldown_start=None,
        cooldown_steps=0,
        frozen_steps=0,
        windows_per_rank=2,
        visits_per_epoch=2,
        num_workers=1,
        in_process_loader=True,
        decoder="pyav",
        frame_hw=(32, 32),
        param_dtype="float32",
        compute_dtype="float32",
        activation_checkpointing=False,
        checkpoint_every=2,
        log_every=1,
        max_grad_norm=1.0,
        ema_sigma_rels=(0.10,),
        reseed_on_resume=False,
    )
    values.update(overrides)
    return TrainConfig(**values)


def records(output):
    return [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]


def losses(output):
    return [(r["step"], r["loss"]) for r in records(output) if "loss" in r and "event" not in r]


def test_validation_is_deterministic_and_leaves_training_untouched(indexed, tmp_path):
    Trainer(make_config(indexed, tmp_path / "plain"), policy_factory=tiny_policy).run()
    for name in ("val_a", "val_b"):
        config = make_config(
            indexed, tmp_path / name, val_every=2, val_windows_per_episode=3, val_sample_windows=2
        )
        Trainer(config, policy_factory=tiny_policy).run()
    # evaluation forks and restores the generators: the training losses are bit-identical
    assert losses(tmp_path / "val_a") == losses(tmp_path / "plain")
    vals = [r for r in records(tmp_path / "val_a") if r.get("event") == "val"]
    assert [r["step"] for r in vals] == [2, 4]
    for r in vals:
        assert r["windows"] == 3 and r["sample_windows"] == 2
        for key in (
            "loss",
            "video_mse",
            "action_mse",
            "sample_action_mse_normalized",
            "sample_action_mse_raw",
        ):
            assert math.isfinite(r[key]) and r[key] >= 0
        assert len(r["sample_action_mse_raw_per_channel"]) == DIM
    again = [r for r in records(tmp_path / "val_b") if r.get("event") == "val"]
    strip = ("time", "seconds")
    assert [{k: v for k, v in r.items() if k not in strip} for r in vals] == [
        {k: v for k, v in r.items() if k not in strip} for r in again
    ]


def test_stop_file_checkpoints_and_the_resume_continues_exactly(indexed, tmp_path):
    Trainer(make_config(indexed, tmp_path / "full"), policy_factory=tiny_policy).run()
    part = tmp_path / "part"
    part.mkdir()
    (part / STOP_FILENAME).touch()
    result = Trainer(make_config(indexed, part), policy_factory=tiny_policy).run()
    assert result["stopped"] and result["step"] == 1
    assert latest_checkpoint(part) == part / "step-1"
    assert (part / STOPPED_FILENAME).exists() and not (part / STOP_FILENAME).exists()
    (part / STOPPED_FILENAME).unlink()  # the launcher clears it before relaunching
    result = Trainer(make_config(indexed, part), policy_factory=tiny_policy).run()
    assert result["step"] == 4 and not result["stopped"]
    assert losses(part) == losses(tmp_path / "full")


def test_keep_best_holds_the_lowest_val_loss_beside_the_latest_checkpoint(indexed, tmp_path):
    output = tmp_path / "best"
    config = make_config(
        indexed, output, steps=4, checkpoint_every=1, val_every=1, keep_checkpoints=1, keep_best="loss"
    )
    Trainer(config, policy_factory=tiny_policy).run()
    vals = {r["step"]: r["loss"] for r in records(output) if r.get("event") == "val"}
    best_step = min(vals, key=vals.get)
    best = json.loads((output / "best" / "best.json").read_text())
    assert (best["step"], best["metric"], best["value"]) == (best_step, "loss", vals[best_step])
    kept = [r for r in records(output) if r.get("event") == "best"]
    assert kept[-1]["step"] == best_step and kept[0]["previous"] is None
    assert all(a["value"] > b["value"] for a, b in zip(kept, kept[1:]))
    # only the newest checkpoint stays; the best keeps its weights (no optimizer) after pruning removed it
    assert [p.name for p in output.glob("step-*")] == ["step-4"]
    assert (output / "best" / "model").is_dir() and not (output / "best" / "optimizer").exists()
    assert export_policy(output / "best", tmp_path / "export")["step"] == best_step
    with pytest.raises(ValueError, match="keep_best needs val_every"):
        make_config(indexed, output, keep_best="loss")


def test_wandb_offline_run_is_continued_on_resume(indexed, tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    output = tmp_path / "wandb_run"
    config = dict(wandb_project="flux-action-test", wandb_mode="offline", steps=2, val_every=2)
    Trainer(make_config(indexed, output, **config), policy_factory=tiny_policy).run()
    run_id = (output / "wandb_run_id").read_text().strip()
    Trainer(make_config(indexed, output, **{**config, "steps": 4}), policy_factory=tiny_policy).run()
    assert (output / "wandb_run_id").read_text().strip() == run_id
    events = [r for r in records(output) if r.get("event") == "wandb"]
    assert len(events) == 2 and all(e["run_id"] == run_id and e["mode"] == "offline" for e in events)
    # one directory per launch, named by start second: two quick launches may share it
    assert len(list((output / "wandb").glob(f"offline-run-*-{run_id}"))) >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CPU offload offloads a GPU's training state")
def test_cpu_offload_on_one_gpu_matches_the_plain_run_and_resumes_exactly(indexed, tmp_path):
    common = dict(compute_dtype="float32", activation_checkpointing=True, val_every=2, val_sample_windows=1)
    Trainer(make_config(indexed, tmp_path / "plain", **common), policy_factory=tiny_policy).run()
    offload = dict(common, cpu_offload=True, offload_text_encoder=True, reshard_after_forward=False)
    trainer = Trainer(make_config(indexed, tmp_path / "offload", **offload), policy_factory=tiny_policy)
    trainer.run()
    assert trainer.sharded and trainer.policy.device.type == "cuda"
    assert next(trainer.policy.dit.parameters()).device.type == "cpu"  # fp32 masters stay on the host
    assert all(s.device.type == "cpu" for ema in trainer.emas.values() for s in ema.shadow)
    plain, offloaded = losses(tmp_path / "plain"), losses(tmp_path / "offload")
    assert [s for s, _ in plain] == [s for s, _ in offloaded] == [1, 2, 3, 4]
    for (_, a), (_, b) in zip(plain, offloaded, strict=True):
        assert b == pytest.approx(a, rel=2e-3, abs=1e-5)  # same math; only kernels and reduction order differ
    part = tmp_path / "part"
    Trainer(make_config(indexed, part, **{**offload, "steps": 2}), policy_factory=tiny_policy).run()
    Trainer(make_config(indexed, part, **offload), policy_factory=tiny_policy).run()
    assert losses(part) == offloaded
    vals = [r for r in records(tmp_path / "offload") if r.get("event") == "val"]
    assert len(vals) == 2 and all(math.isfinite(r["sample_action_mse_raw"]) for r in vals)
