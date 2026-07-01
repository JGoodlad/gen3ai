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


def heuristic_fraction(win_rate_vs_bots: float, *, floor: float = HEURISTIC_FLOOR,
                       start: float = SELF_PLAY_START, full: float = SELF_PLAY_FULL) -> float:
    """Fraction of training episodes that use heuristic (bot) opponents vs. pool snapshots.

    1.0 (0% self-play) below ``start``; smoothstep down to ``floor`` as win rate climbs
    ``start``→``full``; flat ``floor`` above, so the agent always sees a few real bots
    (prevents forgetting basics). Self-play is therefore *gated* on competence — a weak model
    trains entirely vs bots, and self-play only engages (and the pool is only seeded) once win
    rate clears ``start``, so the seed is always captured from a competent model.

    The three anchors are **configurable** (defaults = the module constants
    ``HEURISTIC_FLOOR`` / ``SELF_PLAY_START`` / ``SELF_PLAY_FULL``), threaded from
    ``--heuristic-floor`` / ``--self-play-start-wr`` / ``--self-play-full-wr`` so a run can keep
    the coverage-punishing bots in the training mix for longer (raise ``full`` to ramp slower,
    raise ``floor`` to keep a bigger permanent bot slice). Passing the defaults reproduces the
    original curriculum exactly. Called only in the trainer process (startup + each eval's live
    push); env workers receive the computed fraction via ``env_method``.

    ``GEN3_FORCE_SELFPLAY=1`` overrides this to 0% — EVERY training env uses a pool
    (self-play) opponent. This is the faithful self-play stress mode: only the RLPlayer
    self-play path exercises the snapshot→serialize decision the stale-decision race lives in
    (heuristic bots never touch it), so a fresh run barely tests it. Use it to reproduce the
    race and to verify the settle fix (see ``single_agent_wrapper._settle_opponent_battle``).
    """
    import os
    if os.environ.get("GEN3_FORCE_SELFPLAY"):
        return 0.0
    span = max(full - start, 1e-9)  # guard a degenerate start==full config
    t = (win_rate_vs_bots - start) / span
    t = max(0.0, min(1.0, t))
    t_smooth = t * t * (3.0 - 2.0 * t)
    return 1.0 * (1.0 - t_smooth) + floor * t_smooth


class SnapshotPool:
    """Directory-backed pool of frozen model checkpoints for self-play opponents.

    Filenames encode the training step: ``snapshot_<step:012d>.zip``.
    Sorted directory scan restores the pool on every startup — no JSON manifest.

    By default nothing is pinned: the pool is a **sliding window** of the ``max_snapshots``
    most recent snapshots (oldest evicted first), so an old/weak seed ages out instead of
    lingering as a trivially-easy floor. Anti-forgetting is handled by the heuristic floor
    in ``heuristic_fraction`` (the agent always trains a few % vs real bots), not a pinned
    seed. Seeding is gated on competence by the caller (only seed once win rate clears
    ``SELF_PLAY_START``), so the seed is captured from a competent model.

    Two opt-in **PFSP / league-lite** modes layer on top (both OFF → byte-identical):

    - ``pfsp_scale > 0`` — blend per-snapshot HARDNESS into ``sample()``: oversample the selves
      the trainee is losing to (win-rates pushed via ``set_win_rates`` each eval), without
      starving the ones it beats. Turns the pool from a uniform-ish recency window into a
      prioritised curriculum.
    - ``pool_spread`` — replace the oldest-evicted sliding window with **spread retention** (keep
      newest + oldest + an even interior spread), so PFSP has a genuinely diverse ladder of selves
      to up-weight instead of a recent-selves echo chamber.
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
        pfsp_scale: float = 0.0,
        pool_spread: bool = False,
    ):
        self.pool_dir = Path(pool_dir)
        self._current_version = current_version
        self._device = device
        self.max_snapshots = max_snapshots
        self.recency_weight = recency_weight
        # PFSP (prioritized fictitious self-play): when > 0, blend a per-entry HARDNESS factor
        # into the sampling weight — oversample snapshots the trainee is LOSING to (low win-rate)
        # while never starving the ones it beats. 0.0 → pure recency (byte-identical). The win-rates
        # are pushed in each eval cycle via set_win_rates(); an entry with no measured rate uses the
        # mean of the known rates (so a fresh/unmeasured snapshot is treated as average difficulty).
        self.pfsp_scale = max(0.0, float(pfsp_scale))
        self._win_rates: dict[int, float] = {}  # step → P(trainee beats this snapshot), EMA-smoothed
        # Spread retention: keep a temporally-DIVERSE ladder (newest + oldest + an even interior
        # spread) instead of the oldest-evicted sliding window, so PFSP has a real range of selves
        # to up-weight (a recent-selves-only window is a near-50% echo chamber). Off → byte-identical.
        self._pool_spread = bool(pool_spread)
        self._cache_size = lru_cache_size
        self._entries: list[SnapshotEntry] = []
        self._model_cache: OrderedDict[str, MaskablePPO] = OrderedDict()
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

    def set_win_rates(self, rates: "dict | None") -> None:
        """Store the trainee's per-snapshot win-rates (``{step: P(win)}``) for PFSP weighting.

        Pushed by the self-play eval callback each cycle (EMA-smoothed there). A no-op on the
        sampling weight while ``pfsp_scale == 0``. Keys are coerced to int / values to float so
        a JSON-round-tripped dict (string keys, on resume) is accepted transparently."""
        self._win_rates = {int(k): float(v) for k, v in (rates or {}).items()}

    def _pfsp_default_p(self, entries: list[SnapshotEntry]) -> float:
        """Win-rate to assume for an entry with no measured rate: the mean of the known rates
        over ``entries`` (so an unmeasured snapshot is treated as average difficulty), or 1.0 when
        nothing is known (→ every PFSP factor collapses to 1 ⇒ pure recency at cold start)."""
        known = [self._win_rates[e.step] for e in entries if e.step in self._win_rates]
        return sum(known) / len(known) if known else 1.0

    def _entry_weight_with(self, entry: SnapshotEntry, entries: list[SnapshotEntry],
                           default_p: "float | None" = None) -> float:
        """Sampling weight = recency × PFSP-hardness, computed over the ``entries`` cohort.

        ``pfsp_scale == 0`` returns exactly the recency weight (byte-identical to the legacy
        sliding-window formula). With PFSP on, ``factor = 1 + pfsp_scale·(1 − p)`` where ``p`` is
        the trainee's win-rate vs this snapshot — so a snapshot it loses to (``p→0``) is sampled
        up to ``1 + pfsp_scale``× more, while one it dominates (``p→1``) keeps factor 1 (never
        starved → coverage preserved)."""
        steps = [e.step for e in entries]
        span = max(steps[-1] - steps[0], 1)
        recency = 1.0 + self.recency_weight * (entry.step - steps[0]) / span
        if self.pfsp_scale <= 0.0:
            return recency
        p = self._win_rates.get(entry.step)
        if p is None:
            p = self._pfsp_default_p(entries) if default_p is None else default_p
        p = min(max(p, 0.0), 1.0)
        return recency * (1.0 + self.pfsp_scale * (1.0 - p))

    def sample(self) -> SnapshotEntry:
        """Return one entry weighted toward recent snapshots (× PFSP hardness when enabled).

        ``recency_weight=0`` → uniform; ``recency_weight=1`` → strongly recent. With
        ``pfsp_scale > 0`` the weight is additionally scaled by per-snapshot hardness (see
        ``_entry_weight_with``), oversampling the selves the trainee is losing to.
        """
        if not self._entries:
            raise RuntimeError("Pool is empty — call seed() first")
        default_p = self._pfsp_default_p(self._entries) if self.pfsp_scale > 0.0 else None
        weights = [self._entry_weight_with(e, self._entries, default_p) for e in self._entries]
        return random.choices(self._entries, weights=weights, k=1)[0]

    def entry_weight(self, entry: SnapshotEntry) -> float:
        """Sampling weight for a specific entry (same formula used in sample())."""
        if not self._entries:
            return 1.0
        return self._entry_weight_with(entry, self._entries)

    def sentinel_entries(self, n: int = 5) -> list[SnapshotEntry]:
        """Return up to ``n`` evenly spaced entries, newest first.

        Used for sentinel eval to compute the monotonicity score.
        """
        entries = list(reversed(self._entries))
        if n <= 1:
            return entries[:1]          # n=1 (or clamped 0/neg): just the newest — avoids /0 below
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

    def steps(self) -> list[int]:
        """The training steps of the snapshots currently in the pool (sorted ascending)."""
        return [e.step for e in self._entries]

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
        if self._pool_spread:
            self._evict_spread()
            return
        unpinned = [e for e in self._entries if not e.pinned]
        while len(self._entries) > self.max_snapshots and unpinned:
            oldest = unpinned.pop(0)
            self._entries.remove(oldest)
            oldest.path.unlink(missing_ok=True)
            self._model_cache.pop(str(oldest.path), None)

    def _evict_spread(self) -> None:
        """Spread-retention eviction: keep a temporally-DIVERSE ladder instead of the most-recent
        ``max_snapshots``. Retains the newest (the freshest self) and the oldest (a weak early self —
        a forgetting tripwire PFSP can up-weight if the trainee starts losing to it); thins the most
        REDUNDANT interior snapshot (the one whose neighbours are closest together, i.e. the smallest
        step-gap to its two neighbours) until at capacity. Parameter-free and deterministic. Pinned
        entries are exempt; the two endpoints are scored ``+inf`` redundancy so they're evicted only
        when they are literally the sole droppable entries (so a future pinning scheme can't strand
        the ladder by forcing an endpoint out while a thinnable interior exists)."""
        while len(self._entries) > self.max_snapshots:
            n = len(self._entries)
            droppable = [i for i, e in enumerate(self._entries) if not e.pinned]
            if not droppable:
                return  # everything pinned — cannot evict
            def _redundancy(i: int) -> float:
                if i == 0 or i == n - 1:
                    return float("inf")  # structural anchors — never thinned while an interior remains
                return float(self._entries[i + 1].step - self._entries[i - 1].step)
            victim = self._entries.pop(min(droppable, key=_redundancy))
            victim.path.unlink(missing_ok=True)
            self._model_cache.pop(str(victim.path), None)

    def _find_step(self, step: int) -> SnapshotEntry | None:
        return next((e for e in self._entries if e.step == step), None)
