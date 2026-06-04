"""Self-play snapshot pool.

Manages a directory of frozen model checkpoints used as training opponents.
Pool state is derived entirely from the directory — no separate manifest file.
Reconstructs on every __init__ so launcher restarts are transparent.
"""

from __future__ import annotations

import random
import shutil
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


# Self-play ramp anchors (win rate vs. strategic bots, Random excluded):
#   < SELF_PLAY_START         → NO self-play (100% heuristics) — don't burn cycles on a weak
#                               self-opponent before the model can reliably beat the bots.
#   SELF_PLAY_START→FULL      → smoothstep ramp up.
#   ≥ SELF_PLAY_FULL          → max self-play, keeping HEURISTIC_FLOOR vs bots (anti-forgetting).
SELF_PLAY_START = 0.55
SELF_PLAY_FULL = 0.80
HEURISTIC_FLOOR = 0.10


def heuristic_fraction(win_rate_vs_bots: float) -> float:
    """Fraction of training envs that use heuristic opponents (vs. pool snapshots).

    1.0 (0% self-play) below ``SELF_PLAY_START``; smoothstep down to ``HEURISTIC_FLOOR``
    (→ 90% self-play) as win rate climbs ``SELF_PLAY_START``→``SELF_PLAY_FULL``; flat
    ``HEURISTIC_FLOOR`` above, so the agent always sees a few real bots (prevents forgetting
    basics). Self-play is therefore *gated* on competence — a weak model trains entirely vs
    bots, and self-play only engages (and the pool is only seeded) once win rate clears
    ``SELF_PLAY_START``, so the seed is always captured from a competent model.

    ``GEN3_FORCE_SELFPLAY=1`` overrides this to 0% — EVERY training env uses a pool
    (self-play) opponent. This is the faithful self-play stress mode: only the RLPlayer
    self-play path exercises the snapshot→serialize decision the stale-decision race lives in
    (heuristic bots never touch it), so a fresh run barely tests it. Use it to reproduce the
    race and to verify the settle fix (see ``single_agent_wrapper._settle_opponent_battle``).
    """
    import os
    if os.environ.get("GEN3_FORCE_SELFPLAY"):
        return 0.0
    t = (win_rate_vs_bots - SELF_PLAY_START) / (SELF_PLAY_FULL - SELF_PLAY_START)
    t = max(0.0, min(1.0, t))
    t_smooth = t * t * (3.0 - 2.0 * t)
    return 1.0 * (1.0 - t_smooth) + HEURISTIC_FLOOR * t_smooth


class SnapshotPool:
    """Directory-backed pool of frozen model checkpoints for self-play opponents.

    Filenames encode the training step: ``snapshot_<step:012d>.zip``.
    Sorted directory scan restores the pool on every startup — no JSON manifest.

    Nothing is pinned: the pool is a **sliding window** of the ``max_snapshots`` most
    recent snapshots (oldest evicted first), so an old/weak seed ages out instead of
    lingering as a trivially-easy floor. Anti-forgetting is handled by the heuristic floor
    in ``heuristic_fraction`` (the agent always trains a few % vs real bots), not a pinned
    seed. Seeding is gated on competence by the caller (only seed once win rate clears
    ``SELF_PLAY_START``), so the seed is captured from a competent model.
    """

    _WIN_RATE_FILE = "win_rate_vs_bots.txt"
    _SUMMARY_FILE = "summary.json"

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
        self._distilled_cache: OrderedDict[int, object] = OrderedDict()   # step -> DistilledOpponentModel
        self._distill_layout: dict | None = None
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self._scan()

    # ── Pool population ────────────────────────────────────────────────────

    def seed(self, model: MaskablePPO) -> SnapshotEntry:
        """Save the step-0 seed (NOT pinned — it ages out as the window slides).

        Idempotent — safe to call on every startup; skips the write if the seed
        zip already exists on disk. Gate the CALL on competence (only seed once win rate
        clears ``SELF_PLAY_START``) so the seed is captured from a competent model.
        """
        existing = self._find_step(0)
        if existing:
            return existing
        entry = self._write(model, step=0, pinned=False)
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

    def add_from_path(self, src_zip: "str | Path", step: int) -> SnapshotEntry:
        """Promote an already-saved snapshot file (frozen weights) as a new pool entry.

        Like ``add()``, but COPIES an existing ``.zip`` instead of re-saving a live model.
        The non-blocking self-play eval freezes the trainee's weights to disk at the
        trigger step, then promotes THAT bit-exact snapshot after the (later) collect —
        by which point the live model has advanced, so re-saving it would promote the
        wrong weights. Evicts the oldest non-pinned entry if the pool would exceed
        ``max_snapshots``.
        """
        src = Path(src_zip)
        dst = self.pool_dir / f"snapshot_{step:012d}.zip"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        # Shared arch tag next to the snapshots, so load_model_snapshot() does a REAL
        # compatibility check (every snapshot in this pool shares current_version).
        (self.pool_dir / "model_config.json").write_text(self._current_version.to_json())
        entry = SnapshotEntry(path=dst, step=step, pinned=False)
        # Replace any existing entry at this step (idempotent), keep sorted by step.
        self._entries = [e for e in self._entries if e.step != step]
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.step)
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

    # ── Distilled-opponent variants (all-or-nothing; see distill_integration.md §8) ────────
    # The distilled `.pt` + gate `.json` manifest live in `<pool>/distilled/` next to the
    # snapshots, so they survive restarts (reconstructed by scanning the dir) and slide out with
    # the window. The manager (reconcile loop) uses gate_passed_steps/distilled_artifact_steps/
    # remove_distilled; the env loads the distilled adapter via load_distilled_opponent.
    @property
    def distilled_dir(self) -> Path:
        # Sibling of snapshots/ in the run output dir (models/<run>/distilled/), next to
        # best_model/ + eval_traces/ — not buried under snapshots/.
        return self.pool_dir.parent / "distilled"

    def distilled_pt(self, step: int) -> Path:
        return self.distilled_dir / f"snapshot_{step:012d}.distilled.pt"

    def distilled_manifest(self, step: int) -> Path:
        return self.distilled_dir / f"snapshot_{step:012d}.distilled.json"

    def set_distill_layout(self, layout: dict) -> None:
        """Provide the obs layout so distilled students can be rebuilt on load (env-side)."""
        self._distill_layout = layout

    def gate_passed_steps(self) -> set[int]:
        """Steps whose distilled variant exists AND passed the gate (manifest ``passed: true``)."""
        import json
        out: set[int] = set()
        if not self.distilled_dir.exists():
            return out
        for mf in self.distilled_dir.glob("snapshot_*.distilled.json"):
            try:
                m = json.loads(mf.read_text())
            except (ValueError, OSError):
                continue
            step = int(m.get("step", -1))
            if m.get("passed") and step >= 0 and self.distilled_pt(step).exists():
                out.add(step)
        return out

    def distilled_artifact_steps(self) -> set[int]:
        """Every step with a distilled .pt on disk (passed or not) — for eviction cleanup."""
        if not self.distilled_dir.exists():
            return set()
        return {int(p.name.split("_")[1].split(".")[0])
                for p in self.distilled_dir.glob("snapshot_*.distilled.pt")}

    def distilled_log(self, step: int) -> Path:
        return self.distilled_dir / f"distill_{step:012d}.log"

    def remove_distilled(self, step: int) -> None:
        """Delete every artifact for a snapshot (.pt + manifest + log + any stale .tmp). Called by
        the reconcile loop when the snapshot slides out of the pool window — so the distilled set
        is bounded by the pool size with no separate GC."""
        for p in (self.distilled_pt(step), self.distilled_manifest(step), self.distilled_log(step),
                  Path(str(self.distilled_pt(step)) + ".tmp"),
                  Path(str(self.distilled_manifest(step)) + ".tmp")):
            p.unlink(missing_ok=True)
        self._distilled_cache.pop(step, None)

    def failed_distill_manifests(self) -> dict[int, dict]:
        """{step: manifest} for distilled attempts that did NOT pass the gate — lets a restarted
        manager recover its escalation state (which ladder rung was tried) from disk, so it doesn't
        re-distill a known-unfit snapshot from scratch."""
        import json
        out: dict[int, dict] = {}
        if not self.distilled_dir.exists():
            return out
        for mf in self.distilled_dir.glob("snapshot_*.distilled.json"):
            try:
                m = json.loads(mf.read_text())
            except (ValueError, OSError):
                continue
            if not m.get("passed"):
                out[int(m.get("step", -1))] = m
        return out

    def load_distilled_opponent(self, entry: SnapshotEntry):
        """LRU-cached DistilledOpponentModel adapter for a snapshot (duck-typed like MaskablePPO
        for RLPlayer). Requires ``set_distill_layout`` first."""
        from agents.training.distill.student import load_distilled, DistilledOpponentModel
        if self._distill_layout is None:
            raise RuntimeError("set_distill_layout() must be called before load_distilled_opponent()")
        step = entry.step
        if step in self._distilled_cache:
            self._distilled_cache.move_to_end(step)
            return self._distilled_cache[step]
        if len(self._distilled_cache) >= self._cache_size:
            self._distilled_cache.popitem(last=False)
        student = load_distilled(str(self.distilled_pt(step)), self._distill_layout, device=self._device)
        adapter = DistilledOpponentModel(student, device=self._device)
        self._distilled_cache[step] = adapter
        return adapter

    def sample_from(self, steps) -> "SnapshotEntry | None":
        """Recency-weighted sample restricted to ``steps`` (the distilled-deployable set). None if
        none of those steps are in the pool."""
        allowed = [e for e in self._entries if e.step in set(steps)]
        if not allowed:
            return None
        ss = [e.step for e in allowed]
        span = max(ss[-1] - ss[0], 1)
        weights = [1.0 + self.recency_weight * (e.step - ss[0]) / span for e in allowed]
        return random.choices(allowed, weights=weights, k=1)[0]

    # ── Resume state persistence (summary.json) ────────────────────────────

    def persist_summary(self, **fields) -> None:
        """Merge ``fields`` into ``<pool_dir>/summary.json`` — the self-play resume state.

        Keys we write each eval: ``win_rate_vs_bots``, ``self_play_fraction``,
        ``last_eval_step``, ``seeded``, ``pool_generation``, ``updated_at``. A merge (not a
        rewrite) so partial updates don't drop keys. Distinct from the prober's
        ``eval_traces/*/summary.json``. Mirrors win_rate to the legacy ``.txt`` for any
        in-flight reader.
        """
        import json
        data = self.load_summary()
        data.update(fields)
        (self.pool_dir / self._SUMMARY_FILE).write_text(json.dumps(data, indent=2))
        if "win_rate_vs_bots" in fields:
            try:
                (self.pool_dir / self._WIN_RATE_FILE).write_text(f"{float(fields['win_rate_vs_bots']):.6f}\n")
            except (ValueError, TypeError):
                pass

    def load_summary(self) -> dict:
        """Read ``summary.json`` (the resume state), or ``{}`` if missing/corrupt."""
        import json
        path = self.pool_dir / self._SUMMARY_FILE
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (ValueError, OSError) as e:
                print(f"[SELFPLAY] Warning: could not read {path}: {e} — ignoring")
        return {}

    def persist_win_rate(self, win_rate: float) -> None:
        """Back-compat shim — prefer ``persist_summary``. Records win_rate into summary.json."""
        self.persist_summary(win_rate_vs_bots=float(win_rate))

    def load_persisted_win_rate(self) -> float:
        """Last-persisted win_rate_vs_bots (drives the ramp on resume), or 0.0.

        Prefers ``summary.json``; falls back to the legacy ``win_rate_vs_bots.txt`` for runs
        whose pool predates summary.json.
        """
        summary = self.load_summary()
        if "win_rate_vs_bots" in summary:
            try:
                return float(summary["win_rate_vs_bots"])
            except (ValueError, TypeError):
                pass
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
                # Nothing is pinned — the pool is a sliding window (oldest evicted first),
                # so an old/weak seed ages out rather than lingering as a trivial floor.
                self._entries.append(SnapshotEntry(path=p, step=step, pinned=False))
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
