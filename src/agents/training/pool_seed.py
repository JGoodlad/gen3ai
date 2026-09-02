"""SEED A FORK'S SELF-PLAY POOL FROM ITS PARENT — and REFUSE a poolless `--self-play` fork.

THE DEFECT THIS MAKES UNREPRESENTABLE. :class:`~agents.training.snapshot_pool.SnapshotPool`
derives its entire state from a directory, and a FORK starts in a NEW run dir whose
``snapshots/`` is empty. With ``--self-play`` on, an empty pool is not an error: the trainer
warns and every training episode falls back to the BOT pool. So a fold launched to replicate a
parent that trained at 90% self-play silently trains at ~0% self-play — an opponent-distribution
confound sitting underneath the very contrast the fold measures, invisible in every metric.

It has now cost or nearly cost two cells:

* **2026-08-18** — three 3M-step ``ai_v9_17_tdaux_*`` forks off a 25M base each started empty and
  ended with 1, 2 and 1 snapshots against the base's 12. Essentially all 9M fork-steps were bot
  games; a three-arm A/B whose gates were defined on the self-play regime was voided.
* **2026-09-02** — the three-dose cell. Caught at launch, before any GPU-hour, by reading the
  startup line. Fixed by hand.

**THE MANUAL FIX WAS HALF A FIX THE FIRST TIME, and that half is the reason this module exists.**
Copying the parent's ``snapshot_*.zip`` files alone still produced ``self_play_fraction=0%``: the
STARTING fraction is computed from ``SnapshotPool.load_persisted_win_rate()``, which reads the
pool's METADATA (``summary.json``, falling back to ``win_rate_vs_bots.txt``) — not the zips. A
pool with 14 snapshots and no metadata reads as a competent-model pool that the ramp has not
opened yet, which is exactly as wrong as an empty one and looks healthier.

THE FILE SET (audited against ``SnapshotPool``, 2026-09-02 — this is the complete list of paths
the pool's loader reads out of its own directory):

===============================  ===============================================================
``snapshot_*.zip``               ``_scan()`` globs them; they ARE the pool's entries.
``summary.json``                 ``_SUMMARY_FILE`` — ``load_summary`` / ``load_persisted_win_rate``.
                                 Carries ``win_rate_vs_bots`` (the ramp input), plus
                                 ``self_play_fraction`` / ``last_eval_step`` / ``seeded`` /
                                 ``pool_generation`` for the resume.
``win_rate_vs_bots.txt``         ``_WIN_RATE_FILE`` — the legacy single-float fallback
                                 ``load_persisted_win_rate`` reads when ``summary.json`` has no
                                 ``win_rate_vs_bots`` key.
``model_config.json``            NOT read by ``SnapshotPool`` itself: ``load_model_snapshot``
                                 looks for it beside the ``.zip`` and then one directory up, so
                                 without it every pool opponent silently arch-checks against the
                                 RUN ROOT's config instead of the pool's own.
===============================  ===============================================================

Nothing else in the pool directory is read, and there is no manifest — that is the whole reason
the class is directory-derived.

THE RULE. On a genuine FORK (`main.train.fork_lr.is_same_run_checkpoint` says the `--model` is not
this run's own checkpoint — IMPORTED, never re-derived, because a second predicate for the same
question is a second answer waiting to disagree) with `--self-play` on and an EMPTY pool, copy the
parent's pool. A launcher RESTART never re-seeds (the pool is the run's own by then, and re-seeding
would overwrite a grown pool with the parent's stale one every few hours). A non-empty pool is
NEVER touched, so a hand-seeded arm that later syncs this code keeps exactly the pool it was given.
A FRESH run keeps today's behaviour exactly — a fresh run legitimately starts poolless and grows
one, and the win-rate gate is what keeps it from seeding a random-weights opponent.

THE REFUSAL. If, after all of that, `--self-play` is on and the pool is still empty on a FORK, the
run exits ``FATAL_CONFIG`` instead of quietly training on bots. The three ways out are printed.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

#: The pool's metadata files, in the order they are copied. See the table in the module docstring.
POOL_METADATA_FILES = ("summary.json", "win_rate_vs_bots.txt", "model_config.json")

#: How a pool entry is named on disk (``SnapshotPool._scan``).
SNAPSHOT_GLOB = "snapshot_*.zip"

#: Written INTO the seeded pool directory. Self-describing beside the thing it describes, and the
#: source `_run_lineage` reads to put `pool_seeded_from` into the run's immutable lineage block.
#: Deliberately NOT a metadata.json write at startup: `run_io._resolve_fresh_model_dir` treats an
#: existing metadata.json as "this name is already a run", so writing one before the first
#: checkpoint would make a crashed pre-save fork un-relaunchable under its own name.
SEED_RECORD_FILE = "pool_seed.json"

#: Bumped if the record's shape changes.
SEED_RECORD_SCHEMA = 1


@dataclasses.dataclass(frozen=True)
class PoolSeedDecision:
    """What to do about the pool on this process, and WHY. `reason` is printed, always."""

    seed: bool                      # copy the parent's pool into this run's pool dir?
    refuse: bool                    # exit FATAL_CONFIG (only ever set when `seed` is False)
    is_fork: bool
    pool_dir: str
    parent_run_dir: Optional[str]
    reason: str


# --------------------------------------------------------------------------------------------
# plain filesystem questions
# --------------------------------------------------------------------------------------------
def pool_dir_for(args, model_dir: str) -> str:
    """The pool directory this run uses — ``--snapshot-dir`` if given, else ``<run>/snapshots``.

    Mirrors `train_rl_agent`'s own resolution exactly; both call this so they cannot drift."""
    snapshot_dir = getattr(args, "snapshot_dir", None)
    return str(snapshot_dir) if snapshot_dir else os.path.join(model_dir, "snapshots")


def snapshot_zips(pool_dir: str) -> List[str]:
    """The pool's snapshot files, sorted — the same glob ``SnapshotPool._scan`` uses."""
    try:
        return sorted(glob.glob(os.path.join(pool_dir, SNAPSHOT_GLOB)))
    except OSError:
        return []


def is_pool_empty(pool_dir: str) -> bool:
    """True when the directory holds no ``snapshot_*.zip`` (missing directory included)."""
    return not snapshot_zips(pool_dir)


def parent_run_dir_for(args, model_dir: str) -> Optional[str]:
    """The RUN DIRECTORY of this fork's parent, or None.

    Prefers the run's own RECORDED lineage block (immutable, so it names the original parent even
    after the launcher swaps `--model` to a drifted checkpoint), and falls back to the run
    directory of the `--model` path itself — which is the only answer available on the FIRST
    process of a fork, before any save has written a lineage block.
    """
    from agents.training import lineage

    try:
        recorded = lineage.fork_parent(model_dir, warn=False)
    except Exception:  # noqa: BLE001 — a provenance read must never break a launch
        recorded = None
    if recorded is not None and recorded.run_dir and os.path.isdir(recorded.run_dir):
        return recorded.run_dir
    model_path = getattr(args, "model", None)
    if not model_path:
        return None
    try:
        return lineage.run_dir_of(model_path)
    except Exception:  # noqa: BLE001 — same
        return None


def parent_pool_dir(parent_run_dir: Optional[str]) -> Optional[str]:
    """The parent's pool directory, or None when it has no populated one."""
    if not parent_run_dir:
        return None
    cand = os.path.join(parent_run_dir, "snapshots")
    return cand if snapshot_zips(cand) else None


# --------------------------------------------------------------------------------------------
# the decision (pure)
# --------------------------------------------------------------------------------------------
def decide(*, self_play: bool, model_path: Optional[str], model_dir: str, pool_dir: str,
           parent_run_dir: Optional[str], seed_enabled: bool, allow_empty: bool,
           ) -> PoolSeedDecision:
    """Seed / refuse / do nothing — pure, so the whole rule is unit-testable without a run.

    The six outcomes:

      * self-play OFF                       → nothing (the pool is not read at all).
      * FRESH run (no ``--model``)          → nothing, and NEVER a refusal. A fresh run
                                              legitimately starts poolless and grows one.
      * RESTART (same-run checkpoint)       → nothing. Re-seeding here would overwrite the run's
                                              own grown pool with the parent's stale one on every
                                              periodic restart.
      * FORK, pool NOT empty                → nothing. Idempotent: a hand-seeded arm is untouched.
      * FORK, pool empty, a parent pool     → SEED (unless ``--no-fork-pool-seed``).
      * FORK, pool empty, nothing to seed   → REFUSE, unless ``--allow-empty-pool``.
    """
    from main.train.fork_lr import is_same_run_checkpoint

    if not self_play:
        return PoolSeedDecision(False, False, False, pool_dir, parent_run_dir,
                                "--self-play is off — the pool is not read")
    if not model_path:
        return PoolSeedDecision(False, False, False, pool_dir, parent_run_dir,
                                "fresh run (no --model) — a fresh run starts poolless by design "
                                "and seeds itself once it clears the win-rate gate")
    if is_same_run_checkpoint(model_path, model_dir):
        return PoolSeedDecision(False, False, False, pool_dir, parent_run_dir,
                                "same-run restart — the pool on disk is this run's own")
    if not is_pool_empty(pool_dir):
        n = len(snapshot_zips(pool_dir))
        return PoolSeedDecision(False, False, True, pool_dir, parent_run_dir,
                                f"fork, but the pool already holds {n} snapshot(s) — untouched")
    if not seed_enabled:
        return PoolSeedDecision(False, not allow_empty, True, pool_dir, parent_run_dir,
                                "fork with an EMPTY pool and --no-fork-pool-seed")
    if parent_pool_dir(parent_run_dir) is None:
        return PoolSeedDecision(False, not allow_empty, True, pool_dir, parent_run_dir,
                                "fork with an EMPTY pool, and the fork parent has no pool to "
                                "seed from")
    return PoolSeedDecision(True, False, True, pool_dir, parent_run_dir,
                            "fork with an EMPTY pool and a parent pool to seed from")


# --------------------------------------------------------------------------------------------
# the copy
# --------------------------------------------------------------------------------------------
def seed_pool(parent_run_dir: str, pool_dir: str) -> Dict[str, Any]:
    """Copy the parent's pool — every ``snapshot_*.zip`` PLUS every metadata file that exists.

    Returns the record (also written into the pool dir as ``pool_seed.json``). A metadata file the
    parent does not have is simply absent from ``files``; only ``snapshot_*.zip`` is mandatory, and
    ``decide`` has already established that the parent has some.
    """
    src = os.path.join(parent_run_dir, "snapshots")
    os.makedirs(pool_dir, exist_ok=True)
    copied: List[str] = []
    zips = snapshot_zips(src)
    for z in zips:
        shutil.copy2(z, os.path.join(pool_dir, os.path.basename(z)))
        copied.append(os.path.basename(z))
    metadata: List[str] = []
    for name in POOL_METADATA_FILES:
        p = os.path.join(src, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(pool_dir, name))
            metadata.append(name)
    record: Dict[str, Any] = {
        "schema": SEED_RECORD_SCHEMA,
        "parent_run_dir": os.path.abspath(parent_run_dir),
        "parent_run_name": os.path.basename(os.path.normpath(parent_run_dir)),
        "parent_pool_dir": os.path.abspath(src),
        "n_snapshots": len(copied),
        "snapshots": copied,
        "files": metadata,
        "win_rate_vs_bots": _read_win_rate(pool_dir),
    }
    write_seed_record(pool_dir, record)
    return record


def _read_win_rate(pool_dir: str) -> Optional[float]:
    """``win_rate_vs_bots`` as the pool's own loader will read it — ``summary.json`` first, then
    the legacy ``.txt``. Reported in the seeded line so the fix's HALF-DONE form is visible: a
    ``None`` here is the 14-zips-and-no-metadata pool that reads ``self_play_fraction=0%``."""
    summary = os.path.join(pool_dir, "summary.json")
    if os.path.isfile(summary):
        try:
            val = json.loads(open(summary, encoding="utf-8").read()).get("win_rate_vs_bots")
            if isinstance(val, (int, float)):
                return float(val)
        except (ValueError, OSError):
            pass
    txt = os.path.join(pool_dir, "win_rate_vs_bots.txt")
    if os.path.isfile(txt):
        try:
            return float(open(txt, encoding="utf-8").read().strip())
        except (ValueError, OSError):
            pass
    return None


def write_seed_record(pool_dir: str, record: Dict[str, Any]) -> None:
    try:
        with open(os.path.join(pool_dir, SEED_RECORD_FILE), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    except OSError:  # noqa: BLE001 — provenance must never break a launch
        pass


def read_seed_record(pool_dir: str) -> Optional[Dict[str, Any]]:
    """The seed record left in a pool directory, or None. Total by design."""
    try:
        with open(os.path.join(pool_dir, SEED_RECORD_FILE), encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:  # noqa: BLE001 — absent, truncated, or not ours
        return None
    return obj if isinstance(obj, dict) else None


# --------------------------------------------------------------------------------------------
# the refusal
# --------------------------------------------------------------------------------------------
def refusal_message(decision: PoolSeedDecision) -> str:
    """The FATAL text. Names the three ways out — a refusal that does not is a wall."""
    parent = decision.parent_run_dir or "(unknown — --model resolved to no run directory)"
    return (
        "\n[SELFPLAY] FATAL: --self-play on a FORK whose snapshot pool is EMPTY.\n"
        f"  reason        : {decision.reason}\n"
        f"  pool dir      : {decision.pool_dir}\n"
        f"  fork parent   : {parent}\n"
        "  An empty pool does NOT disable --self-play — it silently falls back to the BOT pool,\n"
        "  so this run would train at ~0% self-play against a parent that trained at its ramped\n"
        "  fraction. That confound voided a three-arm A/B on 2026-08-18 and nearly voided the\n"
        "  three-dose cell on 2026-09-02.\n"
        "  Three ways out, and any one of them is enough:\n"
        f"    * SEED IT BY HAND — copy the parent's pool (zips AND metadata; the zips alone still\n"
        f"      read self_play_fraction=0%):\n"
        f"        cp <parent_run>/snapshots/snapshot_*.zip {decision.pool_dir}/\n"
        f"        cp <parent_run>/snapshots/{{{','.join(POOL_METADATA_FILES)}}} {decision.pool_dir}/\n"
        "    * --allow-empty-pool   consent to the bot fallback explicitly (the pool then grows\n"
        "      from this run's own first promotion, as a fresh run's would)\n"
        "    * drop --self-play     train against bots and say so\n"
    )


# --------------------------------------------------------------------------------------------
# THE ENTRY POINT
# --------------------------------------------------------------------------------------------
def prepare_pool(args, model_dir: str, *, emit_fn=None, exit_fn=None) -> Optional[Dict[str, Any]]:
    """Seed the pool if this is a genuine fork with an empty one; refuse a poolless fork.

    Called ONCE, before the run's :class:`SnapshotPool` is constructed — the starting
    ``self_play_fraction`` is read off the pool's metadata at construction, so seeding afterwards
    would land the files and still announce 0%.

    Returns the seed record when it seeded, else None. Exits ``FATAL_CONFIG`` on a refusal.
    """
    from main.exit_codes import TrainExitCode

    if emit_fn is None:
        from main.launcher.ipc import emit as emit_fn  # type: ignore[assignment]
    if exit_fn is None:
        exit_fn = sys.exit

    pool_dir = pool_dir_for(args, model_dir)
    parent = parent_run_dir_for(args, model_dir)
    decision = decide(
        self_play=bool(getattr(args, "self_play", False)),
        model_path=getattr(args, "model", None),
        model_dir=model_dir,
        pool_dir=pool_dir,
        parent_run_dir=parent,
        # `--no-fork-pool-seed` is BoolFlag's generated negation of `--fork-pool-seed`
        # (default True), so the dest is the POSITIVE name.
        seed_enabled=bool(getattr(args, "fork_pool_seed", True)),
        allow_empty=bool(getattr(args, "allow_empty_pool", False)),
    )
    if decision.refuse:
        print(refusal_message(decision), file=sys.stderr, flush=True)
        exit_fn(int(TrainExitCode.FATAL_CONFIG))
        return None
    if not decision.seed:
        return None
    record = seed_pool(str(decision.parent_run_dir), pool_dir)
    wr = record.get("win_rate_vs_bots")
    wr_txt = f"{wr:.2%}" if isinstance(wr, float) else "UNKNOWN (no pool metadata on the parent)"
    emit_fn(
        f"🌱 [SELFPLAY] [pool] seeded {record['n_snapshots']} snapshots + metadata from "
        f"{record['parent_run_name']} (win_rate_vs_bots={wr_txt}) "
        f"[files: {', '.join(record['files']) or 'none'}]"
    )
    return record
