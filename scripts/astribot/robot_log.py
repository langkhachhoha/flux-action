"""Episode log of the Astribot robot client, in the training data's format, for offline replay.

Per control tick it keeps what entered the policy history: the three 360x640 RGB frames (one MP4 per camera),
the measured 16-D state and the command issued before the tick (NaN when none was). Per prediction it keeps
the request window itself (frames losslessly, since the policy can react to compression-level differences
near a decision such as when to start moving), the returned 42x16 chunk, the seed and the timings.
``replay_log.py`` feeds a log back through the model. Files are written by a background thread.

    <dir>/images.<camera>.mp4    one frame per control tick, 30 fps (lossy; for viewing)
    <dir>/ticks.npz      time (T,), state (T, 16), command_before (T, 16), executed (T, 16; NaN in dry-run)
    <dir>/cycles.npz     tick (C,), state_window (C, 8, 16), command_window (C, 8, 16), chunk (C, 42, 16),
                         seed (C,; -1 = the export's), inference_s (C,), round_trip_s (C,)
    <dir>/windows/cycle_NNNN.npz   images.<camera> uint8 (8, 360, 640, 3): the exact request frames
    <dir>/meta.json      server metadata, arguments, start time, counts
    <dir>/first_frames_images.<camera>.png   the first policy frame (compare the scene with the training data)

The client puts the real-time video of ``robot_video.py`` in ``<dir>/video/`` and copies everything it prints
to ``<dir>/console.log`` (``ConsoleTee``), so one directory holds the whole run.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

MAX_PENDING_FRAMES = (
    600  # queued jobs (20 s of ticks); beyond that ticks are dropped from the MP4s, never the loop
)


class EpisodeLog:
    def __init__(
        self, directory: Path, cameras: Sequence[str], fps: float, meta: Mapping, windows: bool = True
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)  # the client creates it for the video first
        if (self.directory / "meta.json").exists():
            raise FileExistsError(f"{self.directory} already holds an episode log")
        if windows:
            (self.directory / "windows").mkdir()
        self.cameras = tuple(cameras)
        self.fps = float(fps)
        self.windows = windows
        self._meta = dict(meta, start_time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._ticks: dict = {"time": [], "state": [], "command_before": [], "executed": []}
        names = ("tick", "state_window", "command_window", "chunk", "seed", "inference_s", "round_trip_s")
        self._cycles: dict = {name: [] for name in names}
        self._jobs: "queue.Queue" = queue.Queue()
        self._writers: dict = {}
        self.dropped = 0
        self.errors: list = []
        self._thread = threading.Thread(target=self._work, name="episode-log", daemon=True)
        self._thread.start()

    def tick(
        self,
        frames: Mapping[str, np.ndarray],
        state: Sequence[float],
        command_before: Optional[Sequence[float]],
    ) -> int:
        """Record one control tick (where the policy history is appended); returns its index."""
        index = len(self._ticks["time"])
        self._ticks["time"].append(time.monotonic())
        self._ticks["state"].append(np.asarray(state, dtype=np.float32))
        nan = np.full(len(self._ticks["state"][-1]), np.nan, dtype=np.float32)
        before = nan if command_before is None else np.asarray(command_before, np.float32)
        self._ticks["command_before"].append(before)
        self._ticks["executed"].append(nan)
        if self._jobs.qsize() < MAX_PENDING_FRAMES:
            self._jobs.put(("frames", index, {c: frames[c] for c in self.cameras}))
        else:  # never stall the control loop for the log
            self.dropped += 1
        return index

    def executed(self, tick: int, command: Sequence[float]) -> None:
        self._ticks["executed"][tick] = np.asarray(command, dtype=np.float32)

    def cycle(
        self, tick: int, window: Mapping[str, np.ndarray], chunk, seed, inference_s, round_trip_s
    ) -> None:
        index = len(self._cycles["tick"])
        self._cycles["tick"].append(tick)
        self._cycles["state_window"].append(np.asarray(window["state"], dtype=np.float32))
        self._cycles["command_window"].append(np.asarray(window["command_history"], dtype=np.float32))
        self._cycles["chunk"].append(np.asarray(chunk, dtype=np.float32))
        self._cycles["seed"].append(-1 if seed is None else int(seed))
        self._cycles["inference_s"].append(float(inference_s))
        self._cycles["round_trip_s"].append(float(round_trip_s))
        if self.windows:
            self._jobs.put(("window", index, {c: window[c] for c in self.cameras}))

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            kind, index, images = job
            try:
                if kind == "frames":
                    self._write_frames(index, images)
                else:
                    np.savez_compressed(self.directory / "windows" / f"cycle_{index:04d}.npz", **images)
            except Exception as error:  # report at close; the robot loop must keep running
                if len(self.errors) < 5:
                    self.errors.append(f"{kind} {index}: {error}")

    def _write_frames(self, index: int, frames: Mapping[str, np.ndarray]) -> None:
        cv2 = self._cv2
        for camera, frame in frames.items():
            bgr = np.ascontiguousarray(frame[:, :, ::-1])
            if index == 0:
                cv2.imwrite(str(self.directory / f"first_frames_{camera}.png"), bgr)
            writer = self._writers.get(camera)
            if writer is None:
                path = str(self.directory / f"{camera}.mp4")
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, bgr.shape[1::-1])
                if not writer.isOpened():
                    raise RuntimeError(f"OpenCV cannot write {path}")
                self._writers[camera] = writer
            writer.write(bgr)

    def close(self, **extra) -> Path:
        self._jobs.put(None)
        self._thread.join(timeout=120.0)
        for writer in self._writers.values():
            writer.release()
        if self._ticks["time"]:
            np.savez(self.directory / "ticks.npz", **{k: np.stack(v) for k, v in self._ticks.items()})
        if self._cycles["tick"]:
            np.savez(self.directory / "cycles.npz", **{k: np.stack(v) for k, v in self._cycles.items()})
        meta = dict(self._meta, ticks=len(self._ticks["time"]), cycles=len(self._cycles["tick"]))
        meta.update(dropped_frames=self.dropped, log_errors=self.errors, **extra)
        (self.directory / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        for error in self.errors:
            print(f"WARNING: episode log: {error}", flush=True)
        return self.directory


class ConsoleTee:
    """Copy everything written to ``sys.stdout`` and ``sys.stderr`` into ``path`` as well, until ``close``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._file = open(self.path, "a", encoding="utf-8", buffering=1)
        self._saved = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(sys.stdout, self._file), _Tee(sys.stderr, self._file)

    def close(self) -> None:
        if self._file.closed:
            return
        sys.stdout, sys.stderr = self._saved
        self._file.close()


class _Tee:
    def __init__(self, stream, file) -> None:
        self._stream, self._file = stream, file

    def write(self, text: str) -> int:
        count = self._stream.write(text)
        try:
            self._file.write(text)
        except ValueError:  # closed; the terminal still gets the text
            pass
        return count

    def flush(self) -> None:
        self._stream.flush()
        if not self._file.closed:
            self._file.flush()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)
