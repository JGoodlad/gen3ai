"""Self-play snapshot pool.

Manages a directory of frozen model checkpoints used as training opponents.
Pool state is derived entirely from the directory — no separate manifest file.
Reconstructs on every __init__ so launcher restarts are transparent.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from sb3_contrib import MaskablePPO

from agents.model.snapshot import load_model_snapshot
from agents.model.model_version import ModelVersion
from main.launcher.ipc import emit


@dataclass
class SnapshotEntry:
    path: Path
    step: int
    pinned: bool = False  # pinned entries are never evicted


def heuristic_fraction(win_rate_vs_bots: float) -> float:
    """Fraction of training envs that use heuristic opponents (vs. pool snapshots).

    Smoothstep interpolation from 0.80 (agent weak, needs bot guidance) to 0.10
    (agent dominant, mostly self-play with a floor to prevent forgetting basics).
    The ramp activates once win rate vs. strategic bots passes 50%.

    ``GEN3_FORCE_SELFPLAY=1`` overrides this to 0% — EVERY training env uses a pool
    (self-play) opponent. This is the faithful self-play stress mode: only the RLPlayer
    self-play path exercises the snapshot→serialize decision the stale-decision race lives in
    (heuristic bots never touch it), so a fresh run at the natural ~80% bots barely tests it.
    Use it to reproduce the race and to verify the settle fix (see
    ``single_agent_wrapper._settle_opponent_battle``).
    """
    import os
    if os.environ.get("GEN3_FORCE_SELFPLAY"):
        return 0.0
    t = (win_rate_vs_bots - 0.50) / (0.85 - 0.50)
    t = max(0.0, min(1.0, t))
    t_smooth = t * t * (3.0 - 2.0 * t)
    return 0.80 * (1.0 - t_smooth) + 0.10 * t_smooth


class SnapshotPool:
    """Directory-backed pool of frozen model checkpoints for self-play opponents.

    Filenames encode the training step: ``snapshot_<step:012d>.zip``.
    Sorted directory scan restores the pool on every startup — no JSON manifest.

    The step-0 seed is always pinned and never evicted. All other entries are
    evicted oldest-first when the pool exceeds ``max_snapshots``.
    """

    _WIN_RATE_FILE = "win_rate_vs_bots.txt"

    def __init__(
        self,
        pool_dir: Path,
        current_version: ModelVersion,
        device: str = "auto",
        max_snapshots: int = 20,
        recency_weight: float = 0.3,
        lru_cache_size: int = 3,
    ):
        self.pool_dir = Path(pool_dir)
        self._current_version = current_version
        self._device = device
        self.max_snapshots = max_snapshots
        self.recency_weight = recency_weight
        self._cache_size = lru_cache_size
        self._entries: list[SnapshotEntry] = []
        self._model_cache: OrderedDict[str, MaskablePPO] = OrderedDict()
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self._scan()

    # ── Pool population ────────────────────────────────────────────────────

    def seed(self, model: MaskablePPO) -> SnapshotEntry:
        """Save the step-0 seed (pinned, never evicted).

        Idempotent — safe to call on every startup; skips the write if the seed
        zip already exists on disk.
        """
        existing = self._find_step(0)
        if existing:
            return existing
        entry = self._write(model, step=0, pinned=True)
        emit(f"🌱 [SELFPLAY] Pool seeded at step 0 → {entry.path.name}")
        return entry

    def add(self, model: MaskablePPO, step: int) -> SnapshotEntry:
        """Promote the current model as a new pool snapshot.

        Evicts the oldest non-pinned entry if the pool would exceed
        ``max_snapshots`` after the addition.
        """
        entry = self._write(model, step=step, pinned=False)
        self._evict()
        emit(f"📦 [SELFPLAY] Snapshot promoted at step {step:,} → {entry.path.name} "
             f"(pool size: {len(self._entries)})")
        return entry

    # ── Sampling ───────────────────────────────────────────────────────────

    def sample(self) -> SnapshotEntry:
        """Return one entry weighted toward recent snapshots.

        ``recency_weight=0`` → uniform; ``recency_weight=1`` → strongly recent.
        """
        if not self._entries:
            raise RuntimeError("Pool is empty — call seed() first")
        steps = [e.step for e in self._entries]
        span = max(steps[-1] - steps[0], 1)
        weights = [
            1.0 + self.recency_weight * (e.step - steps[0]) / span
            for e in self._entries
        ]
        return random.choices(self._entries, weights=weights, k=1)[0]

    def entry_weight(self, entry: SnapshotEntry) -> float:
        """Sampling weight for a specific entry (same formula used in sample())."""
        if not self._entries:
            return 1.0
        steps = [e.step for e in self._entries]
        span = max(steps[-1] - steps[0], 1)
        return 1.0 + self.recency_weight * (entry.step - steps[0]) / span

    def sentinel_entries(self, n: int = 5) -> list[SnapshotEntry]:
        """Return up to ``n`` evenly spaced entries, newest first.

        Used for sentinel eval to compute the monotonicity score.
        """
        entries = list(reversed(self._entries))
        if len(entries) <= n:
            return entries
        step_size = (len(entries) - 1) / (n - 1)
        return [entries[round(i * step_size)] for i in range(n)]

    # ── Model loading ──────────────────────────────────────────────────────

    def load_model(self, entry: SnapshotEntry) -> MaskablePPO:
        """Return an LRU-cached model for inference (env=None, no training state)."""
        key = str(entry.path)
        if key in self._model_cache:
            self._model_cache.move_to_end(key)
        else:
            if len(self._model_cache) >= self._cache_size:
                self._model_cache.popitem(last=False)
            loaded = load_model_snapshot(
                str(entry.path),
                env=None,
                current_version=self._current_version,
                device=self._device,
            )
            self._model_cache[key] = loaded
        return self._model_cache[key]

    # ── Win-rate persistence ───────────────────────────────────────────────

    def persist_win_rate(self, win_rate: float) -> None:
        """Write win_rate_vs_bots to disk so the next launcher restart picks it up."""
        (self.pool_dir / self._WIN_RATE_FILE).write_text(f"{win_rate:.6f}\n")

    def load_persisted_win_rate(self) -> float:
        """Read the last-persisted win rate, or 0.0 if not available."""
        path = self.pool_dir / self._WIN_RATE_FILE
        if path.exists():
            try:
                return float(path.read_text().strip())
            except (ValueError, OSError) as e:
                print(f"[SELFPLAY] Warning: could not read {path}: {e} — defaulting to 0.0")
        return 0.0

    # ── Convenience ────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def __len__(self) -> int:
        return len(self._entries)

    # ── Internals ──────────────────────────────────────────────────────────

    def _scan(self) -> None:
        """Reconstruct pool state from the directory on disk."""
        paths = sorted(self.pool_dir.glob("snapshot_*.zip"))
        self._entries = []
        for p in paths:
            try:
                step = int(p.stem.split("_")[1])
                self._entries.append(SnapshotEntry(path=p, step=step, pinned=(step == 0)))
            except (IndexError, ValueError):
                pass  # ignore malformed filenames

    def _write(self, model: MaskablePPO, step: int, pinned: bool) -> SnapshotEntry:
        path = self.pool_dir / f"snapshot_{step:012d}.zip"
        model.save(str(path))
        # Drop a shared model_config.json next to the snapshots so load_model_snapshot()
        # — used by BOTH eval sentinels and the training-env opponents — performs a REAL
        # architecture compatibility check instead of silently skipping it (every snapshot
        # in a pool shares this run's current_version). Without it, a stale-arch snapshot
        # would load with mismatched weights instead of a clean ModelVersionError.
        (self.pool_dir / "model_config.json").write_text(self._current_version.to_json())
        entry = SnapshotEntry(path=path, step=step, pinned=pinned)
        # Replace any existing entry at this step (idempotent re-seed)
        self._entries = [e for e in self._entries if e.step != step]
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.step)
        return entry

    def _evict(self) -> None:
        unpinned = [e for e in self._entries if not e.pinned]
        while len(self._entries) > self.max_snapshots and unpinned:
            oldest = unpinned.pop(0)
            self._entries.remove(oldest)
            oldest.path.unlink(missing_ok=True)
            self._model_cache.pop(str(oldest.path), None)

    def _find_step(self, step: int) -> SnapshotEntry | None:
        return next((e for e in self._entries if e.step == step), None)
