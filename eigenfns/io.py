"""Checkpointing and result I/O.

Long solves checkpoint after every locked block and auto-resume; multi-hour
CUDA runs on this machine can die (driver crashes documented in the parent
project), so everything is regenerable from the latest checkpoint.

Blocks are stored append-only (block_000.npy, block_001.npy, ...) so a 128³
run never rewrites the accumulated locked set (~22 GB) on each save.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class BlockCheckpointer:
    """Persist locked eigenpairs incrementally; resume a partial solve."""

    def __init__(self, directory: str | Path, tag: str, meta: dict | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.meta = dict(meta or {})

    def _meta_path(self) -> Path:
        return self.dir / f"{self.tag}_meta.json"

    def _block_paths(self, i: int) -> tuple[Path, Path]:
        return (self.dir / f"{self.tag}_block{i:03d}_vals.npy",
                self.dir / f"{self.tag}_block{i:03d}_vecs.npy")

    def load(self):
        """Return (vals, vecs, n_blocks, meta); vals/vecs None if no checkpoint."""
        pm = self._meta_path()
        if not pm.exists():
            return None, None, 0, self.meta
        meta = json.loads(pm.read_text())
        n_blocks = int(meta.get("n_blocks", 0))
        vals, vecs = [], []
        for i in range(n_blocks):
            pv, px = self._block_paths(i)
            if not (pv.exists() and px.exists()):
                raise RuntimeError(f"checkpoint corrupt: missing block {i} files")
            vals.append(np.load(pv))
            vecs.append(np.load(px))
        if not vals:
            return None, None, 0, meta
        return np.concatenate(vals), np.concatenate(vecs), n_blocks, meta

    def save_block(self, i: int, vals: np.ndarray, vecs: np.ndarray,
                   extra: dict | None = None):
        """Append block i (write-then-rename for crash safety)."""
        pv, px = self._block_paths(i)
        for path, arr in ((pv, vals), (px, vecs)):
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            tmp.replace(path)
        meta = dict(self.meta)
        meta.update(extra or {})
        meta["n_blocks"] = i + 1
        meta["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self._meta_path().with_suffix(".tmp.json")
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(self._meta_path())
