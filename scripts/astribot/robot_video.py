"""Continuous camera video of an Astribot run (vmo_diffusion_policy_vla's AstribotVideoRecorder, ported).

The episode log keeps one frame per control tick, and there are no ticks while the server samples a chunk.
This recorder instead takes the newest frame of every camera at a fixed rate from a ROS timer of its own, from
the operator's start through the ready pose, warm-up, every prediction and the home motion, so the video
shows the run in real time. The grid video adds a status panel (mode, phase, cycle, chunk motion, grippers,
elapsed time). A background thread encodes (H.264 when OpenCV's FFmpeg has it, else MPEG-4); when it falls
behind, video frames are dropped, never control ticks. ROS-free; the client calls ``submit`` from the node.

    <dir>/<camera>.mp4      one per camera, 640x360
    <dir>/grid.mp4          head | status panel, left wrist | right wrist (1280x720)
    <dir>/session_info.json duration, frame count, drops, codec, run metadata
"""

from __future__ import annotations

import atexit
import json
import queue
import threading
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from robot_runtime import FRAME_HW

CODECS = ("avc1", "mp4v")  # H.264 first; mp4v plays in fewer viewers but every OpenCV build writes it
MAX_PENDING_SECONDS = 10.0
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, YELLOW, GREEN, RED, BLUE = (
    (235, 235, 235),
    (160, 160, 160),
    (0, 220, 255),
    (110, 230, 120),
    (70, 70, 240),
    (255, 200, 150),
)  # BGR


class VideoRecorder:
    def __init__(
        self,
        directory: Path,
        cameras: Sequence[str],
        fps: float = 30.0,
        *,
        grid: bool = True,
        title: str = "",
        mode: str = "",
        metadata: Optional[Mapping] = None,
        frame_hw: Sequence[int] = FRAME_HW,
    ) -> None:
        if fps <= 0 or not cameras:
            raise ValueError("need fps > 0 and at least one camera")
        self.directory = Path(directory)
        self.cameras = tuple(cameras)
        self.fps = float(fps)
        self.grid = grid
        self.title, self.mode = title, mode
        self.metadata = dict(metadata or {})
        self.frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
        self.codec: Optional[str] = None
        self.frames = 0
        self.dropped = 0
        self.errors: list = []
        self._status = {"phase": "START", "cycle": None, "note": ""}
        self._lock = threading.Lock()
        self._jobs: "queue.Queue" = queue.Queue(maxsize=max(1, int(self.fps * MAX_PENDING_SECONDS)))
        self._writers: dict = {}
        self._thread: Optional[threading.Thread] = None
        self._recording = False
        self._started_at = 0.0
        self._start_time = ""
        self._summary: Optional[dict] = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        import cv2

        if self._thread is not None:
            return
        self._cv2 = cv2
        self.directory.mkdir(parents=True, exist_ok=True)
        height, width = self.frame_hw
        for camera in self.cameras:
            self._writers[camera] = self._open(f"{camera}.mp4", (width, height))
        if self.grid:
            rows = (len(self.cameras) + 2) // 2  # the cameras plus the status panel, two per row
            self._writers["grid"] = self._open("grid.mp4", (2 * width, rows * height))
        self._started_at = time.monotonic()
        self._start_time = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._thread = threading.Thread(target=self._work, name="video-recorder", daemon=True)
        self._thread.start()
        self._recording = True
        atexit.register(self.stop)  # finalize the MP4s even when the client dies

    def _open(self, name: str, size):
        cv2 = self._cv2
        path = str(self.directory / name)
        for codec in CODECS if self.codec is None else (self.codec,):
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), self.fps, size)
            if writer.isOpened():
                self.codec = codec
                return writer
            writer.release()
        raise RuntimeError(f"OpenCV cannot write {path} with any of {CODECS}")

    def set_status(
        self, phase: Optional[str] = None, cycle: Optional[int] = None, note: Optional[str] = None
    ) -> None:
        """Update the status panel; arguments left as None keep their value."""
        with self._lock:
            for key, value in (("phase", phase), ("cycle", cycle), ("note", note)):
                if value is not None:
                    self._status[key] = value

    def submit(self, frames: Mapping[str, np.ndarray], state: Optional[Sequence[float]] = None) -> None:
        """One video frame: the newest RGB frame of each camera (missing ones are drawn black)."""
        if not self._recording:
            return
        with self._lock:
            status = dict(self._status)
        item = (dict(frames), None if state is None else np.asarray(state), status, time.monotonic())
        try:
            self._jobs.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _work(self) -> None:
        while True:
            item = self._jobs.get()
            if item is None:
                return
            try:
                self._write(*item)
            except Exception as error:  # report at stop; the robot loop must keep running
                if len(self.errors) < 5:
                    self.errors.append(str(error))

    def _write(self, frames: Mapping, state, status: dict, captured: float) -> None:
        cv2 = self._cv2
        elapsed = captured - self._started_at
        tiles = []
        for camera in self.cameras:
            rgb = frames.get(camera)
            if rgb is None:
                bgr = np.zeros((*self.frame_hw, 3), dtype=np.uint8)
                cv2.putText(bgr, "no signal", (20, 60), FONT, 1.0, GREY, 2, cv2.LINE_AA)
            else:
                bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
                if bgr.shape[:2] != self.frame_hw:
                    bgr = cv2.resize(bgr, self.frame_hw[::-1], interpolation=cv2.INTER_AREA)
            self._writers[camera].write(bgr)
            tiles.append(self._banner(bgr, camera, elapsed) if self.grid else None)
        self.frames += 1
        if self.grid:
            tiles.insert(1, self._panel(state, status, elapsed))
            if len(tiles) % 2:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
            self._writers["grid"].write(np.vstack(rows))

    def _banner(self, bgr: np.ndarray, camera: str, elapsed: float) -> np.ndarray:
        cv2 = self._cv2
        out = bgr.copy()
        out[:30] = (out[:30] * 0.35).astype(np.uint8)
        cv2.putText(out, camera, (10, 21), FONT, 0.6, YELLOW, 1, cv2.LINE_AA)
        stamp = f"{elapsed:6.1f}s"
        width = cv2.getTextSize(stamp, FONT, 0.55, 1)[0][0]
        cv2.putText(out, stamp, (out.shape[1] - width - 10, 21), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        return out

    def _panel(self, state, status: dict, elapsed: float) -> np.ndarray:
        cv2 = self._cv2
        panel = np.full((*self.frame_hw, 3), 28, dtype=np.uint8)
        cycle = status["cycle"]
        lines = [
            ("FLUX ACTION - ASTRIBOT S1", YELLOW, 0.8),
            (self.title[:44], WHITE, 0.6),
            (f"mode   {self.mode}", RED if self.mode == "EXECUTE" else GREEN, 0.7),
            (f"phase  {status['phase']}", GREEN, 0.7),
            (f"cycle  {'--' if cycle is None else f'{cycle:04d}'}", BLUE, 0.7),
            (status["note"][:44], WHITE, 0.6),
        ]
        if state is not None and len(state) >= 16:
            lines.append((f"grippers  L {state[14]:5.1f}   R {state[15]:5.1f}", WHITE, 0.6))
        lines.append((f"time {elapsed:7.1f}s   frame {self.frames}", GREY, 0.6))
        y = 40
        for text, color, scale in lines:
            if text:
                cv2.putText(panel, text, (20, y), FONT, scale, color, 1 if scale < 0.7 else 2, cv2.LINE_AA)
            y += 42
        return panel

    def stop(self) -> dict:
        """Finish encoding, close the files and write ``session_info.json``; repeated calls are no-ops."""
        if self._summary is not None or self._thread is None:
            return self._summary or {}
        self._recording = False
        atexit.unregister(self.stop)
        self._jobs.put(None)
        self._thread.join(timeout=60.0)
        for writer in self._writers.values():
            writer.release()
        duration = time.monotonic() - self._started_at
        self._summary = {
            "directory": str(self.directory),
            "start_time": self._start_time,
            "duration_s": round(duration, 2),
            "frames": self.frames,
            "fps": self.fps,
            "dropped_frames": self.dropped,
            "codec": self.codec,
            "cameras": list(self.cameras),
            "files": sorted(f"{name}.mp4" for name in self._writers),
            "errors": self.errors,
            "metadata": self.metadata,
        }
        info = self.directory / "session_info.json"
        info.write_text(json.dumps(self._summary, indent=1, default=str) + "\n")
        for error in self.errors:
            print(f"WARNING: video recorder: {error}", flush=True)
        return self._summary
