"""The eval cycle's COLLECT side — split out of ``eval_callback.py`` (2026-10-01).

Shared by BOTH eval callbacks: pooling a cycle's shard results (``merge_eval_results``), stating the
trace SELECTION in the cycle's manifest (``record_eval_selection``), retaining / grooming the weight
snapshots and the trace dirs, the best-model copies, and the cycle's wall clock
(``record_cycle_wall``). ``eval_callback`` re-exports every public name here.
"""
import glob
import json
import os
import re
import shutil
import time
from typing import Any

from agents.training.eval_launch import EVAL_MANIFEST_NAME, EVAL_SNAPSHOT_NAME
from agents.training.eval_player import ForensicQuota, _rule_for
from agents.training.eval_sharding import PLAN_NAME, ShardedEvalPool
from agents.training.trace_selection import SELECTION_SCHEMA, build_selection
from main.launcher.ipc import send_metrics
from utils.contention import describe_contention


def merge_eval_results(run_dir: str, names: list[str]) -> tuple[dict, list]:
    """Pool the cycle's shard results into the per-opponent ``merged`` dict + the fully-missing names.

    Battle-level work-stealing writes one ``shard__<unit_id>.json`` per played shard (an opponent
    is split into chunks any idle worker can drain). This reads the cycle's ``plan.json`` — the
    launcher writes it before spawning workers, so it is the single source of truth for which
    shards each opponent expects — groups the shards by opponent and aggregates **exactly**
    (Σwon/Σfinished for win_rate; count-weighted reward/ep_len; raw δ pooled then ONE CVaR, since a
    CVaR can't be averaged). The returned ``merged`` is shape-compatible with every downstream
    consumer (record_per_opponent / build_bot_eval_block / record_elo / the pool & externals
    blocks) and additionally carries ``counts`` (exact W/L per opponent — the rating-fidelity
    record) and ``coverage``. ``missing`` = names with ZERO shards present (a whole opponent lost
    to a worker crash), logged by the caller exactly as a missing opponent always was. Shared by
    both eval callbacks. (Per-cycle ``run_dir`` is wiped at cleanup, so no shard/lock ever leaks
    across cycles.)
    """
    empty = {"win_rates": {}, "reward_means": {}, "ep_lens": {},
             "td_resid_tails": {}, "durations_sec": {}, "counts": {}, "coverage": {}}
    if not os.path.exists(os.path.join(run_dir, PLAN_NAME)):
        return empty, list(names)  # no plan written (shouldn't happen live) → nothing to merge
    merged_all, missing_all = ShardedEvalPool.from_plan(run_dir).collect(run_dir)
    wanted = set(names)
    merged = {block: {k: v for k, v in vals.items() if k in wanted}
              for block, vals in merged_all.items()}
    missing = [n for n in names if n in set(missing_all)]
    # Partial coverage (some-but-not-all shards of an opponent reported) → it was measured over
    # fewer games; surface it so a crashed shard doesn't silently degrade win-rate / ELO unnoticed.
    partial = {k: c for k, c in merged.get("coverage", {}).items() if c < 1.0}
    if partial:
        worst = ", ".join(f"{k} {c * 100:.0f}%" for k, c in sorted(partial.items(), key=lambda x: x[1]))
        # Do NOT assert a cause here: a worker crash and a cycle killed for overrunning produce
        # the identical shortfall, and naming only the crash sent readers hunting the wrong one
        # (the same misattribution that let the parity test blame the Rust port for the load
        # average). State the fact, then hand over the evidence that separates the two.
        print(f"⚠️ [EVAL] partial shard coverage — these win-rates/ELO are measured over FEWER "
              f"games (a worker crashed, or the cycle was killed for overrunning): {worst}. "
              f"{describe_contention()}")
    return merged, missing


def prune_eval_traces(model_dir: str | None, keep_n: int) -> None:
    """Keep only the N most-recent eval step dirs under ``<model_dir>/eval_traces``.

    0 = keep all. Older dirs are removed whole; the current cycle's dir is the newest so
    it is never touched. ``python -m main.prober.groom`` is the manual fallback.
    """
    if keep_n <= 0 or not model_dir:
        return
    base = os.path.join(model_dir, "eval_traces")
    if not os.path.isdir(base):
        return
    dirs = []
    for name in os.listdir(base):
        m = re.match(r"step_(\d+)$", name)
        full = os.path.join(base, name)
        if m and os.path.isdir(full):
            dirs.append((int(m.group(1)), full))
    for _step, path in sorted(dirs, reverse=True)[keep_n:]:
        shutil.rmtree(path, ignore_errors=True)


def prune_eval_snapshots(model_dir: str | None, keep_n: int) -> None:
    """Keep only the N most-recent persisted eval snapshots (``eval_traces/step_*/snapshot.zip``)."""
    if keep_n <= 0 or not model_dir:
        return
    snaps = glob.glob(os.path.join(model_dir, "eval_traces", "step_*", EVAL_SNAPSHOT_NAME))

    def _stepof(p: str) -> int:
        m = re.search(r"step_(\d+)", p)
        return int(m.group(1)) if m else 0

    for p in sorted(snaps, key=_stepof, reverse=True)[keep_n:]:
        try:
            os.remove(p)
        except OSError:
            pass


def record_eval_selection(model_dir: str | None, step: int, merged: dict,
                          quota: "ForensicQuota | dict | None" = None) -> dict | None:
    """Patch this cycle's manifest with the per-opponent TRACE SELECTION (`gen3_trace_selection_manifest_v1`).

    Called at COLLECT, because the counts do not exist at launch: `write_eval_manifest` runs before
    a single battle is played. Same patch-the-manifest shape as `persist_eval_snapshot`.

    ``merged`` is the pooled eval result. `counts` gives (n_won, n_finished) per opponent — what
    was PLAYED — `traces` gives (traces_won, traces_written[, traces_drawn]) — what the forensic
    quota KEPT — and `draws` gives the DRAWN battles played (tie or 250-turn timeout), which
    poke-env counts nowhere. Together they are the whole record; the derived capture rates are
    computed by `trace_selection.selection_entry` so the producer and every consumer share one
    arithmetic.

    🚨 The `traces` tuple GREW a third element on 2026-09-07. It is unpacked BY LENGTH, so a
    2-tuple (an older parent, or a caller written before the draw bucket) still records — with
    its draws left ABSENT rather than written as a zero nobody measured.

    Returns the block it wrote, or ``None`` when there was nothing to record or the manifest could
    not be updated. **A failure is a warning, never a raise** — this is provenance for an offline
    reader, and it must not be able to take down a training run at an eval boundary. The block then
    stays ``null``, which every consumer reads as SELECTION UNKNOWN.
    """
    if not model_dir:
        return None
    counts = merged.get("counts") or {}
    traces = merged.get("traces") or {}
    draws = merged.get("draws") or {}
    if not counts:
        return None
    per_opponent = {}
    for key, (n_won, n_finished) in counts.items():
        t = tuple(traces.get(key) or (0, 0))
        t_won, t_written = t[0], t[1]
        t_drawn = t[2] if len(t) > 2 else 0
        per_opponent[str(key)] = {
            "battles_played": n_finished, "battles_won": n_won,
            "battles_drawn": int(draws.get(key) or 0),
            "traces_written": t_written, "traces_won": t_won, "traces_drawn": t_drawn,
        }
    q = ForensicQuota.coerce(quota)
    block = build_selection(per_opponent, win_quota=q.win,
                            loss_quota=q.loss,
                            draw_quota=q.draw)
    mpath = os.path.join(model_dir, "eval_traces", f"step_{step}", EVAL_MANIFEST_NAME)
    try:
        with open(mpath) as f:
            m = json.load(f)
        m["selection_schema"] = SELECTION_SCHEMA
        m["selection_rule"] = _rule_for(q)
        m["selection"] = block
        with open(mpath, "w") as f:
            json.dump(m, f, indent=2)
    except (OSError, ValueError) as e:
        print(f"⚠️ [EVAL] could not record trace selection for step {step:,}: {e}")
        return None
    return block


def persist_eval_snapshot(model_dir: str | None, step: int, snapshot_path: str, keep_n: int) -> None:
    """Copy a cycle's weight snapshot into ``eval_traces/step_<N>/snapshot.zip`` (next to
    its traces), patch that step's manifest to point at it, then prune to the N most-recent.

    No-op when ``keep_n<=0`` (traces still carry the identity manifest). Lets the prober
    reload the bit-exact model that produced a cycle's traces. Shared by both eval callbacks.
    """
    if keep_n <= 0 or not model_dir:
        return
    dst_dir = os.path.join(model_dir, "eval_traces", f"step_{step}")
    os.makedirs(dst_dir, exist_ok=True)
    try:
        shutil.copy2(snapshot_path, os.path.join(dst_dir, EVAL_SNAPSHOT_NAME))
    except OSError as e:
        print(f"⚠️ [EVAL] could not persist snapshot for step {step:,}: {e}")
        return
    mpath = os.path.join(dst_dir, EVAL_MANIFEST_NAME)
    try:
        with open(mpath) as f:
            m = json.load(f)
        m["snapshot"] = EVAL_SNAPSHOT_NAME
        with open(mpath, "w") as f:
            json.dump(m, f, indent=2)
    except (OSError, ValueError):
        pass
    prune_eval_snapshots(model_dir, keep_n)


def copy_run_config_to_best_model(model_dir: "str | None", best_model_dir: "str | None") -> None:
    """Copy the run-level ``model_config.json`` into ``best_model/`` whenever the best model is saved,
    so ``best_model/`` is a SELF-CONTAINED snapshot (weights + arch sidecar in one dir) — the unified
    place a stable-opponent consumer looks first for the arch gate. (The eval/ELO sidecar is the
    separate ``best_model.json`` written by ``write_best_model_sidecar``.) Best-effort."""
    if not model_dir or not best_model_dir:
        return
    src = os.path.join(model_dir, "model_config.json")
    if not os.path.exists(src):
        return
    try:
        os.makedirs(best_model_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(best_model_dir, "model_config.json"))
    except OSError as e:
        print(f"⚠️ [EVAL] could not copy model_config.json into best_model/: {e}")


def write_best_model_sidecar(model_dir: "str | None", best_zip_path: str, model) -> None:
    """Write ``best_model/best_model.json`` — the per-checkpoint-style sidecar for the best snapshot,
    REUSING ``snapshot.write_checkpoint_metadata`` (which writes ``<zip − .zip>.json``). It carries
    the current ``latest_eval`` block — **including the run's ELO** — so ``best_model/`` is a
    self-contained, ELO-bearing snapshot a stable-opponent consumer can read directly (no parent
    search). Best-effort — never raise into the best-save path."""
    if not model_dir:
        return
    try:
        from agents.model.snapshot import (
            write_checkpoint_metadata, _read_latest_eval, _read_pin_history,
        )
        lr = float(model.policy.optimizer.param_groups[0]["lr"])
        # git_hash=None on purpose: `snapshot.resolve_git_hash` owns that decision for EVERY
        # sidecar (launcher pin → the imported checkout's HEAD, raising if they disagree).
        # Passing `get_git_hash()` here used to hand it a pre-resolved ambient value.
        write_checkpoint_metadata(best_zip_path, lr=lr, n_epochs=int(model.n_epochs),
                                  eval_block=_read_latest_eval(model_dir),
                                  pin_history=_read_pin_history(model_dir))
    except Exception as e:  # noqa: BLE001 — best-effort sidecar; must NEVER break the best-save path
        print(f"⚠️ [EVAL] could not write best_model.json sidecar: {e}")


def record_cycle_wall(cb: Any, pending: dict, merged: dict, *, tag: str) -> float:
    """`eval/wall_sec` (gen3_eval_wall_sec_v1): the eval cycle's WALL time — from the top of
    `_launch_eval` (the snapshot save included) to the end of its collection (records, best-model
    save, trace manifest, snapshot persist, pruning) — recorded at the eval step beside
    `eval/duration_sec`, which is the SUMMED UNIT TIME (every shard's own duration, added up: on the
    Rust eval core the units play CONCURRENTLY, so it read ~350-390 s for a ~14 s cycle).
    🚨 On the RUST core the cycle is blocking and in-process, so `wall_sec` IS its cost to training;
    on the PYTHON core the workers play beside training, so it is the cycle's LATENCY. Called from
    inside `_collect_pending`'s `isolated_dump` scope, so its own dump carries only these scalars."""
    t0 = pending.get("t_launch", pending.get("launched_at"))
    wall = max(0.0, time.monotonic() - float(t0)) if t0 is not None else float("nan")
    summed = float(sum(merged.get("durations_sec", {}).values()))
    step = int(pending["step"])
    cb.logger.record("eval/wall_sec", wall)
    cb.logger.dump(step)
    send_metrics({"eval/wall_sec": wall, "_step": step})
    print(f"[{tag}] step {step:,}: cycle wall {wall:.1f}s (launch -> collected) · "
          f"summed unit time {summed:.0f}s (eval/duration_sec)", flush=True)
    return wall
