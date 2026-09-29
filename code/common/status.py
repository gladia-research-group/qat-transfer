"""Atomic run lifecycle and progress signaling."""

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path


class StatusWriter:
    def __init__(self, eval_dir: str | Path):
        self.path = Path(eval_dir) / ".status.json"
        self.t0 = None
        self.status = {}

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.status, sort_keys=True) + "\n")
        tmp.replace(self.path)

    def __enter__(self):
        self.t0 = time.monotonic()
        self.status = {
            "state": "running",
            "started": self._now(),
            "ended": None,
            "elapsed_s": None,
            "heartbeat": self._now(),
            "progress": None,
            "wave_id": os.environ.get("WAVE_ID"),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "source_revision": os.environ.get("SOURCE_REVISION"),
            "source_tag": os.environ.get("SOURCE_TAG"),
        }
        self._write()
        print(
            f"[status] RUN START {self.status['started']} "
            f"wave={self.status['wave_id']} gpu={self.status['gpu']} "
            f"source={self.status['source_revision']}",
            flush=True,
        )
        return self

    def heartbeat(self, progress: str | None = None) -> None:
        self.status["heartbeat"] = self._now()
        if progress is not None:
            self.status["progress"] = progress
        self._write()

    def __exit__(self, exc_type, exc, tb):
        self.status.update(
            state="failed" if exc_type else "done",
            ended=self._now(),
            elapsed_s=round(time.monotonic() - self.t0, 1),
        )
        self._write()
        print(
            f"[status] RUN END {self.status['ended']} "
            f"state={self.status['state']} elapsed={self.status['elapsed_s']}s",
            flush=True,
        )
        return False
