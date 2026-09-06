#!/usr/bin/env python3
"""The TIERED retention policy — ``archive_grooming_dryrun.py --policy tiered``.

The standing policy (``--policy standing``, still the default) asks one question of
every run — *is it CLOSED?* — and applies one rule.  That is the right shape while
every era is still in play, and the wrong shape once whole eras are not: it frees
18.8 GB of a 257.1 GB archive, and it cannot free the 20.8 GB of **legacy
root-level checkpoints** that the pre-``checkpoints/`` runs carry, because it only
ever looks inside ``checkpoints/``.

The tiered policy asks a second question — *which era, and is anything still
reaching for it?* — and grades the answer:

===== =============================================== ==========================
tier  who                                             what happens
===== =============================================== ==========================
0     LIVE, or anything a live run reaches for        nothing
1     REFERENCED in any era                           standing + snapshots rule
2     ai_v9+ CLOSED                                   standing + snapshots rule
3     ai_v8 CLOSED, unreferenced                      first+last+pin, no every-10th
4     pre-ai_v8 (v5/v6/v7 and the run_2026* dirs)     AGGRESSIVE keep-list
===== =============================================== ==========================

**The owner's reason for tier 4, verbatim (2026-09-06):** "Yes, please work on a
reasonable retention policy, especially pre ai_v8 eras, as we are unlikely to need
anything from them as there wasn't a 'novel' outcome, more getting the pattern
established and us able to make meaningful progress."

Two mechanisms here are NEW and are not merely a re-grading of the standing rules:

* **The reference graph reads ``original_command`` as well as ``lineage``.**  The
  ``lineage`` block postdates most of the archive — the live ``v8rep_*``
  replication arms carry ``lineage: null`` — so a lineage-only graph cannot see
  that they fork ``ai_v8_04`` and distil from ``ai_v8_09``/``_06``/``_13``.  The
  ledger tail does not name those teachers either (measured: 12 runs in the last
  1500 lines, none of them the teachers).  Under the standing policy they were
  saved by the blanket "v8-era" rule; the tiered policy retires that blanket, so
  it has to see the edges properly instead.
* **Tier 4 has a KEEP-LIST, not a delete-list**, and it is enforced twice — once
  when the plan is built and once by :func:`assert_safe_tiered` before the plan is
  reported or executed.  A tier-4 run whose final model cannot be resolved gets
  **no plan at all** rather than a best guess.

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterable

# --------------------------------------------------------------- constants ----

#: the ledger window the tiered policy reads (the standing policy uses 1000)
TIERED_LEDGER_TAIL_LINES = 1500

#: the first ai_v8 run is ``ai_v8_01_zarch_film_0717``; anything whose era resolves
#: only through a date is pre-v8 below this.  Kept as (month, day) of 2026.
PRE_V8_CUTOFF_MMDD = (7, 17)

#: tier 3 keeps the ends and the pin only — no every-Nth stride.
TIER3_CHECKPOINT_EVERY = 0

#: proposed (never applied) thinning for a POOL that is kept.
SNAPSHOT_THIN_EVERY = 4

#: tier 4 keeps these subdirectories whole — the era's training record.
TIER4_KEEP_SUBDIRS = frozenset({"tb", "tb_imgs", "snapshot_ladder", "elo"})

#: tier 4 keeps these run-root files by name, always.
TIER4_KEEP_ROOT_FILES = frozenset({
    "metadata.json", "model_config.json", "latest.txt", "eval_results.jsonl",
})

#: tier 4 deletes run-root files with these extensions (except the resolved model).
TIER4_ROOT_DELETE_EXT = frozenset({".zip", ".log"})

#: the pool directory, and the metadata a fork's auto-seed needs alongside the zips.
SNAPSHOT_DIR = "snapshots"
SNAPSHOT_META = frozenset({"model_config.json", "summary.json", "win_rate_vs_bots.txt"})

_SNAP_RE = re.compile(r"^snapshot_(\d+)\.zip$")
_CKPT_SIDECAR_RE = re.compile(r"^checkpoint_\d+_steps\.json$")

#: argv flags whose value is a model/pool reference, for the snapshots rule.
_POOL_FLAGS = ("--stable-opponents", "--exploiter", "--exploiter-ladder",
               "--warmstart-consensus")
_MODEL_FLAGS = ("--model", "--distill-anchor-parent")
_TEACHER_FLAGS = ("--distill-teacher", "--win-prob-pbrs-source")

#: committed files under this prefix count as a MEASUREMENT ARTIFACT — naming a run
#: there protects it (tier 1), unlike ordinary prose.
MEASUREMENT_PREFIX = "designs/research_state/measurements/"

#: 🚨 **AN ARTIFACT THAT ENUMERATES THE ARCHIVE IS NOT A REFERENCE TO ANYTHING IN
#: IT.**  This is not a nicety — the first tiered run put **all 118** non-tier-0
#: runs in tier 1 and graded nothing, because the committed census
#: ``archive_grooming_dryrun_2026-09-06.md`` names every run in ``models/`` by
#: construction, and so does every lineage / sidecar sweep.  A retention policy that
#: reads its own previous report as evidence can never close a run.  Matched on the
#: BASENAME, so a future dated copy is covered without an edit.
BOOKKEEPING_ARTIFACT_PREFIXES = (
    "archive_grooming_",        # this tool's own reports
    "fh_lineage",               # the folding-history lineage sweep
    "folding_history_",         # ditto
    "sidecar_audit",            # the per-checkpoint commit audit
    "run_inventory",            # any future whole-archive inventory
)


def is_bookkeeping(rel: str) -> bool:
    """True for a committed file that names runs because it ENUMERATES the archive."""
    base = os.path.basename(rel)
    return any(base.startswith(p) for p in BOOKKEEPING_ARTIFACT_PREFIXES)

#: **REVIEW HOLDS** — runs pinned to tier 1 by a HUMAN READING, recorded here with
#: the finding that forced it.  The standing policy's ``needs_review`` flag makes a
#: run visible; it does not decide.  This is where a decision lands, so that "we
#: looked at it" is a fact in the code rather than a claim in a report.
#:
#: The five runs the 2026-09-06 dry run flagged were read against the ledger and the
#: committed tree.  Three are held; two are released, and WHY is stated for each.
#:
#: A HOLD SUPPRESSES THE PLAN ENTIRELY — it does not merely downgrade the tier.  The
#: reason a run is held is that we are not certain which of its files its banked
#: claim rests on; grooming "only" its ``eval_traces`` while that is unresolved is
#: the same bet with a smaller stake.  A hold is released by BANKING the artifact,
#: not by re-reading the ledger.
REVIEW_HOLDS: "dict[str, str]" = {
    "ai_v9_48_G1_action_0826":
        "REVIEW HOLD — the program's first POSITIVE distill arm (pooled +0.0398 "
        "[+0.016,+0.064] z=+3.29; G2-fdB +0.0762 z=+6.01, ledger.md:4943ff). NO "
        "committed artifact carries the per-arm numbers: fold_capacity_telemetry.md "
        "has fdA/fdB/fdC/fdE rows and no G1/G2 row, and no ai_v9_48_*_endofrun.json "
        "exists — the claim rests on this run's eval_results.jsonl + eval_traces. "
        "Bank an endofrun artifact and this hold can be released.",
    "ai_v9_26_baitent_probe_0823":
        "REVIEW HOLD (partial) — the capacity baseline IS banked "
        "(designs/research_state/capacity_battery.md:153ff), but the P2 bait-entropy "
        "per-leg result (boost_eff 3.0, flagged 5.9%, B1 0.056 -> 0.229, leg-vs-leg "
        "z=-2.55, ledger.md:3722) is in no committed artifact, and the Baton Pass "
        "GIGO reproducer decodes loss_s0_003_states.npz from this run's traces "
        "(ledger.md:3595). ladder_readiness.md:269 also loads its legB_final_model.zip.",
    "ai_v9_45_fdF_p1_0826":
        "REVIEW HOLD — the NUMBERS are banked "
        "(designs/ai_v10/design_advantage_gated_distillation.md:459-467 carries the "
        "entropy 0.892 -> 1.354 dissolution and the subtraction rule), so this is not "
        "a data dependency; it is held because the ledger records an explicit owner "
        "decision to preserve it as the entropy-dissolution SPECIMEN (ledger.md:4937).",
    # RELEASED, and why — kept here so the reading is visible, not just its outcome:
    # ai_v5_11_tail2_53m_0611: its only ledger claim (value_share ~0.6, ELO ahead of
    #   ai_v5_10, ledger.md:85) lives in tb/ and the checkpoint sidecars — and tier 4
    #   KEEPS tb/ and metadata.json, so the claim survives its own tier.  The two
    #   downstream findings are already banked in
    #   designs/research_state/levers/attack_type_mismatch.md:15-31 and
    #   designs/ai_v6/design_differentiable_damage_op.md:39-48.
    # ai_v5_12_bias_05_N_0612: ledger.md:87 asserts no number at all ("LIVE
    #   (bias-redesign)"), and its one measured contribution is banked at
    #   levers/attack_type_mismatch.md:33-34 (immune-pick V>=0 delta +0.0158 z=6.8).
    #   The cleanest release of the five.
}


class TieredRefusal(RuntimeError):
    """A tier-4 plan would delete something the keep-list names, or the run has no
    resolvable final model.  The run's plan is dropped; nothing is guessed at."""


# --------------------------------------------------------- reference graph ----


def _argv_text(meta: dict) -> str:
    """The provenance strings a run records about what it LOADED."""
    parts = []
    for key in ("original_command", "launcher_command"):
        v = meta.get(key)
        if isinstance(v, str):
            parts.append(v)
    cli = meta.get("cli_args")
    if isinstance(cli, dict):
        parts.append(json.dumps(cli))
    return "\n".join(parts)


def _flag_before(text: str, pos: int, window: int = 220) -> str:
    """The nearest ``--flag`` token to the left of ``pos`` (``""`` if none near)."""
    left = text[max(0, pos - window):pos]
    hits = re.findall(r"(--[a-z0-9][a-z0-9-]*)", left)
    return hits[-1] if hits else ""


def argv_refs(meta: dict, run_names: "Iterable[str]") -> "list[list[str]]":
    """``[kind, run_name, resolved_file]`` for every run this run's ARGV names.

    The ``lineage`` block is the authority when it exists; this is what stands in
    for it on every pre-``lineage`` run on disk — which is most of the archive, and
    includes the live ``v8rep_*`` arms whose block is ``null``.  ``kind`` is derived
    from the nearest preceding flag, so the snapshots rule can tell a FORK PARENT
    (``--model``) from a teacher or a pool source.
    """
    names = [n for n in run_names if n]
    if not names:
        return []
    text = _argv_text(meta)
    if not text:
        return []
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    rx = re.compile(r"(" + alt + r")((?:/[A-Za-z0-9_.\-]+)*\.zip)?")
    out: dict[tuple, list[str]] = {}
    for m in rx.finditer(text):
        name, tail = m.group(1), m.group(2) or ""
        flag = _flag_before(text, m.start())
        if flag in _MODEL_FLAGS:
            kind = "argv fork_parent"
        elif flag in _POOL_FLAGS:
            kind = "argv pool_source"
        elif flag in _TEACHER_FLAGS:
            kind = "argv teacher"
        else:
            kind = "argv mention"
        key = (kind, name, os.path.basename(tail))
        out.setdefault(key, [kind, name, os.path.basename(tail)])
    return list(out.values())


def build_model_graph(runs: dict) -> dict:
    """Who reaches for whom, over ``lineage`` UNION ``original_command``.

    Returns ``{"refs_out", "referenced_by", "pinned_files", "fork_parents"}``:

    * ``refs_out[child]``  — every ``[kind, parent, file]`` the child names;
    * ``referenced_by[parent]`` — human strings naming the children;
    * ``pinned_files[parent]`` — bare filenames some child resolved to;
    * ``fork_parents`` — parents named by a FORK edge (``lineage`` ``fork_parent``
      or an argv ``--model``), which is the set the snapshots rule keeps, because a
      fork auto-seeds its parent's pool.
    """
    names = list(runs)
    refs_out: dict[str, list[list[str]]] = {}
    referenced_by: dict[str, list[str]] = {}
    pinned: dict[str, set[str]] = {}
    fork_parents: set[str] = set()

    for child, r in runs.items():
        edges = [list(e) for e in r.get("lineage_refs") or []]
        meta = r.get("_metadata")
        if meta is None:
            meta = read_metadata(r["run_dir"])
        edges += argv_refs(meta, names)
        seen: set[tuple] = set()
        keep_edges: list[list[str]] = []
        for kind, parent, resolved in edges:
            if parent == child or parent not in runs:
                continue
            key = (kind, parent, os.path.basename(resolved or ""))
            if key in seen:
                continue
            seen.add(key)
            keep_edges.append([kind, parent, resolved])
            referenced_by.setdefault(parent, []).append(f"{child} ({kind})")
            if resolved:
                pinned.setdefault(parent, set()).add(os.path.basename(resolved))
            if kind in ("fork_parent", "argv fork_parent", "argv pool_source"):
                fork_parents.add(parent)
        refs_out[child] = keep_edges

    return {"refs_out": refs_out, "referenced_by": referenced_by,
            "pinned_files": pinned, "fork_parents": fork_parents}


def read_metadata(run_dir: str) -> dict:
    try:
        with open(os.path.join(run_dir, "metadata.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------ era + tiers ----


_DATE_TAIL = re.compile(r"_(\d{2})(\d{2})$")
_RUN_TS = re.compile(r"^run_(\d{4})(\d{2})(\d{2})")


def era_of(name: str, generation: str) -> "tuple[str, str]":
    """``(era, how)`` — ``era`` is ``ai_vN`` or ``unknown``; ``how`` names the source.

    A run whose generation does not resolve from its name or its lineage is dated
    from the name instead (``run_YYYYMMDD…`` or a trailing ``_MMDD``), because the
    ``run_2026*`` dirs carry no generation at all and the owner's tier 4 names them
    explicitly.  A date this cannot read is ``unknown`` — never assumed pre-v8.
    """
    m = re.search(r"ai_v(\d+)", generation)
    if m:
        return f"ai_v{m.group(1)}", f"generation ({generation})"
    if name.startswith("v8rep"):
        return "ai_v8", "v8 replication family"
    m = _RUN_TS.match(name)
    if m:
        mo, day = int(m.group(2)), int(m.group(3))
        side = "pre-v8" if (mo, day) < PRE_V8_CUTOFF_MMDD else "v8+"
        return ("pre_v8_by_date" if side == "pre-v8" else "unknown",
                f"date {m.group(1)}-{m.group(2)}-{m.group(3)} ({side} of "
                f"{PRE_V8_CUTOFF_MMDD[0]:02d}-{PRE_V8_CUTOFF_MMDD[1]:02d})")
    m = _DATE_TAIL.search(name)
    if m:
        mo, day = int(m.group(1)), int(m.group(2))
        side = "pre-v8" if (mo, day) < PRE_V8_CUTOFF_MMDD else "v8+"
        return ("pre_v8_by_date" if side == "pre-v8" else "unknown",
                f"name date {m.group(1)}-{m.group(2)} ({side})")
    return "unknown", "no generation and no readable date"


def is_pre_v8(era: str) -> bool:
    return era in ("ai_v5", "ai_v6", "ai_v7", "pre_v8_by_date")


def is_v8(era: str) -> bool:
    return era == "ai_v8"


def is_v9_plus(era: str) -> bool:
    m = re.match(r"ai_v(\d+)$", era)
    return bool(m) and int(m.group(1)) >= 9


def assign_tiers(runs: dict, live: dict, refs: dict, graph: dict,
                 recent_days: int, ledger_tail_lines: int,
                 follow_symlinked_runs: bool, now: float) -> None:
    """Stamp ``tier`` / ``tier_name`` / ``tier_reasons`` onto every run, in place.

    First match wins, and the order is the safety order: a run that is LIVE-adjacent
    is never graded on its era.
    """
    # tier 0, pass 1 — the direct signals
    tier0: dict[str, list[str]] = {}
    for n, r in runs.items():
        why: list[str] = []
        if n in live:
            why.append("a LIVE launcher/trainer process names this run dir")
        if recent_days and r["mtime"] and (now - r["mtime"]) < recent_days * 86400:
            why.append(f"training output written within {recent_days} days "
                       f"({r['mtime_source']}/, {r['mtime_iso']}) — an arm launched "
                       "between ledger updates has no other signal")
        if r["is_symlink"] and not follow_symlinked_runs:
            why.append("run dir is a SYMLINK into a launcher worktree — the bytes are "
                       f"at {r['realpath']}, outside models/ "
                       "(--follow-symlinked-runs to include it)")
        if why:
            tier0[n] = why

    # tier 0, pass 2 — everything a tier-0 run reaches for, transitively
    frontier = set(tier0)
    seen = set(frontier)
    while frontier:
        nxt = set()
        for child in frontier:
            for _kind, parent, _f in graph["refs_out"].get(child, []):
                if parent in runs and parent not in seen:
                    seen.add(parent)
                    nxt.add(parent)
                    tier0.setdefault(parent, []).append(
                        f"model-graph ancestor (transitively) of the LIVE/recent run "
                        f"{child}")
        frontier = nxt

    for n, r in runs.items():
        era, era_how = era_of(n, r["generation"])
        r["era"] = era
        r["era_source"] = era_how

        if n in tier0:
            r["tier"], r["tier_name"] = 0, "LIVE"
            r["tier_reasons"] = tier0[n]
            continue

        why = []
        if n in REVIEW_HOLDS:
            r["review_hold"] = REVIEW_HOLDS[n]
            r["tier"], r["tier_name"] = 1, "REFERENCED"
            r["tier_reasons"] = [REVIEW_HOLDS[n]]
            continue
        if n in refs["ledger_runs"]:
            why.append(f"named in the ledger's last {ledger_tail_lines} lines")
        origins = [o for o in refs["by_run"].get(n, [])
                   if o != "ledger.md (tail)" and not is_bookkeeping(o)]
        scripts = [o for o in origins if _is_script(o)]
        artifacts = [o for o in origins
                     if o.startswith(MEASUREMENT_PREFIX) and not _is_script(o)]
        if scripts:
            why.append(f"loaded by {len(scripts)} committed script(s): "
                       + ", ".join(scripts[:3]) + (" …" if len(scripts) > 3 else ""))
        if artifacts:
            why.append(f"named by {len(artifacts)} committed measurement artifact(s): "
                       + ", ".join(artifacts[:3]) + (" …" if len(artifacts) > 3 else ""))
        by_runs = graph["referenced_by"].get(n) or []
        if by_runs:
            why.append("named by another run's model graph: "
                       + ", ".join(sorted(by_runs)[:3])
                       + (" …" if len(by_runs) > 3 else ""))
        if why:
            r["tier"], r["tier_name"] = 1, "REFERENCED"
            r["tier_reasons"] = why
            continue

        if is_pre_v8(era):
            r["tier"], r["tier_name"] = 4, "PRE-v8"
            r["tier_reasons"] = [f"{era} via {era_how}; nothing reaches for it"]
        elif is_v8(era):
            r["tier"], r["tier_name"] = 3, "v8 CLOSED"
            r["tier_reasons"] = [f"{era} via {era_how}; nothing reaches for it"]
        elif is_v9_plus(era):
            r["tier"], r["tier_name"] = 2, "v9+ CLOSED"
            r["tier_reasons"] = [f"{era} via {era_how}; nothing reaches for it"]
        else:
            # an era we cannot read is graded as the GENTLEST closed tier, never
            # as pre-v8 — an unreadable name must not buy an aggressive plan.
            r["tier"], r["tier_name"] = 2, "v9+ CLOSED"
            r["tier_reasons"] = [f"era {era_how} — graded as the gentlest closed tier"]


_SCRIPT_EXT = frozenset({".py", ".sh", ".bash", ".ipynb"})


def _is_script(rel: str) -> bool:
    return os.path.splitext(rel)[1].lower() in _SCRIPT_EXT


# ------------------------------------------------------- the snapshots rule ----


def plan_snapshots(run_dir: str, keep: bool, keep_reason: str) -> dict:
    """The pool: kept whole, or deleted whole.  Thinning is PROPOSED, never planned.

    A self-play pool is kept only when some run forks this run (a fork auto-seeds
    its parent's pool) or a committed script names the run.  When it is kept, the
    report carries what an every-``SNAPSHOT_THIN_EVERY``th + newest thinning WOULD
    free — a number to decide on, not an action.
    """
    d = os.path.join(run_dir, SNAPSHOT_DIR)
    out = {"action": "absent", "reason": "no snapshots/ dir", "delete": [],
           "bytes": 0, "thin_proposal_bytes": 0, "n_snapshots": 0,
           "n_thin_would_drop": 0}
    if not os.path.isdir(d):
        return out

    snaps: list[tuple[int, str, int]] = []
    total = 0
    for e in sorted(os.listdir(d)):
        p = os.path.join(d, e)
        if not os.path.isfile(p):
            continue
        try:
            size = os.path.getsize(p)
        except OSError:
            continue
        total += size
        m = _SNAP_RE.match(e)
        if m:
            snaps.append((int(m.group(1)), e, size))
    snaps.sort()
    out["n_snapshots"] = len(snaps)
    out["bytes"] = total

    if not keep:
        out["action"] = "delete"
        out["reason"] = keep_reason
        out["delete"] = [SNAPSHOT_DIR]
        return out

    out["action"] = "keep"
    out["reason"] = keep_reason
    # A pool seeds a fork only if the METADATA rides with the zips — the zips alone
    # read `self_play_fraction=0%`.  Report a kept pool that is missing any of it.
    present = {e for e in os.listdir(d)}
    missing = sorted(SNAPSHOT_META - present)
    if missing:
        out["reason"] += (f" ⚠️ but the pool is MISSING {', '.join(missing)} — "
                          "a fork seeded from it would start at 0% self-play")
    # PROPOSAL only: every 4th, ascending, plus the newest.
    keep_idx = {i for i in range(len(snaps)) if i % SNAPSHOT_THIN_EVERY == 0}
    if snaps:
        keep_idx.add(len(snaps) - 1)
    drop = [s for i, s in enumerate(snaps) if i not in keep_idx]
    out["thin_proposal_bytes"] = sum(s[2] for s in drop)
    out["n_thin_would_drop"] = len(drop)
    return out


def snapshots_verdict(name: str, graph: dict, refs: dict) -> "tuple[bool, str]":
    """Does this run's pool survive?  ``(keep, reason)`` — one sentence per run."""
    if name in graph["fork_parents"]:
        who = [w for w in graph["referenced_by"].get(name, [])
               if "fork_parent" in w or "pool_source" in w]
        return True, ("KEPT — a fork parent / pool source: "
                      + ", ".join(sorted(who)[:3]) + " (a fork auto-seeds its "
                      "parent's pool, so the zips AND the metadata are load-bearing)")
    scripts = [o for o in refs["by_run"].get(name, [])
               if _is_script(o) and not is_bookkeeping(o)]
    if scripts:
        return True, ("KEPT — a committed script names this run: "
                      + ", ".join(scripts[:2]))
    return False, ("FREED — no run forks it, no committed script names it as a "
                   "--stable-opponents / --exploiter / pool source")


# ------------------------------------------------------------ tier 4 plan ----


def resolve_final_model(run_dir: str) -> dict:
    """The ONE model file tier 4 keeps, through the run-spec choke point.

    ``agents.training.fixed_opponent_pool.resolve_model_ref`` is CALLED, never
    re-derived, so the file kept is by construction the file a bare
    ``--distill-teacher <run>`` / ``--stable-opponents <run>`` would load today
    (``gen3_last_snapshot_resolution_v1``: latest.txt → highest checkpoint →
    final_model → best_model LAST).
    """
    out = {"ok": False, "zip_path": "", "rel": "", "rung": "", "rule": "",
           "num_timesteps": None, "error": ""}
    try:
        from agents.training.fixed_opponent_pool import resolve_model_ref
    except Exception as exc:                                    # pragma: no cover
        out["error"] = f"resolve_model_ref unavailable: {exc}"
        return out
    try:
        rm = resolve_model_ref(run_dir, warn=False)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    run_abs = os.path.realpath(run_dir)
    zp = os.path.realpath(rm.zip_path)
    rel = os.path.relpath(zp, run_abs)
    if rel.startswith(".."):
        out["error"] = f"resolved {rm.zip_path} is outside {run_dir}"
        return out
    out.update({"ok": True, "zip_path": rm.zip_path, "rel": rel, "rung": rm.rung,
                "rule": rm.rule, "num_timesteps": rm.num_timesteps})
    return out


def tier4_keep_set(run_dir: str, resolved_rel: str, latest_pin: "str | None",
                   pinned_files: "Iterable[str]") -> "dict[str, str]":
    """Run-relative paths tier 4 keeps, each with its reason.  The keep-list IS the
    policy — everything not named here is deleted."""
    keep: dict[str, str] = {}
    for f in sorted(TIER4_KEEP_ROOT_FILES):
        if os.path.exists(os.path.join(run_dir, f)):
            keep[f] = "the run's identity + result record"
    for d in sorted(TIER4_KEEP_SUBDIRS):
        if os.path.isdir(os.path.join(run_dir, d)):
            keep[d] = "the era's training record — never thinned"
    # EVERY checkpoint SIDECAR survives, at either layout.  A sidecar is kilobytes
    # and carries the per-checkpoint record — step count, git hash, pin history,
    # the eval at that step — which is precisely what makes the era's training
    # record readable after its weights are gone.  Tier 4 deletes the WEIGHTS; it
    # is not supposed to delete the LOG of them, and the ai_v5_11 review turned on
    # exactly this distinction (its one ledger claim lives in tb/ + the sidecars).
    for d in (".", "checkpoints"):
        sub = os.path.join(run_dir, d)
        if not os.path.isdir(sub):
            continue
        for e in sorted(os.listdir(sub)):
            if not _CKPT_SIDECAR_RE.match(e):
                continue
            rel = e if d == "." else os.path.join(d, e)
            keep[rel] = "checkpoint sidecar — the record, not the weights"

    for e in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, e)
        if not os.path.isfile(p) or e in keep:
            continue
        if os.path.splitext(e)[1].lower() not in TIER4_ROOT_DELETE_EXT:
            keep[e] = "small run-root bookkeeping (not a .zip or a .log)"
    if resolved_rel:
        keep[resolved_rel] = "THE final model — what resolve_model_ref picks"
    if latest_pin:
        pin = latest_pin.strip()
        if pin and os.path.isfile(os.path.join(run_dir, pin)):
            keep.setdefault(pin, "latest.txt's target — so the pin still resolves")
    for f in sorted(set(pinned_files or ())):
        for cand in (os.path.join("checkpoints", f), f):
            if os.path.isfile(os.path.join(run_dir, cand)):
                keep.setdefault(cand, "another run's model graph resolved to this file")
                break
    return keep


def plan_tier4(run_dir: str, latest_pin: "str | None",
               pinned_files: "Iterable[str]") -> dict:
    """The aggressive plan: everything not on the keep-list, deleted.

    Raises :class:`TieredRefusal` when no final model resolves — a run whose one
    keeper cannot be identified is left alone entirely.
    """
    resolved = resolve_final_model(run_dir)
    if not resolved["ok"]:
        raise TieredRefusal(
            f"no final model resolves for {run_dir}: {resolved['error']}")

    keep = tier4_keep_set(run_dir, resolved["rel"], latest_pin, pinned_files)
    keep_tops = {k.split(os.sep)[0] for k in keep}

    delete: list[str] = []
    freed = 0
    for e in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, e)
        if e in keep:
            continue
        if os.path.isdir(p):
            if e in keep_tops:
                # a kept file lives inside — descend and delete around it
                for sub in sorted(os.listdir(p)):
                    rel = os.path.join(e, sub)
                    if rel in keep:
                        continue
                    delete.append(rel)
                    freed += _size(os.path.join(run_dir, rel))
            else:
                delete.append(e)
                freed += _size(p)
        else:
            delete.append(e)
            freed += _size(p)

    return {"delete": delete, "keep": keep, "bytes_freed": freed,
            "resolved": resolved}


def _size(path: str) -> int:
    if os.path.isdir(path):
        total = 0
        for root, _d, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def assert_safe_tiered(run_dir: str, delete_rel: "Iterable[str]",
                       keep_rel: "Iterable[str]") -> None:
    """The tier-4 safety net: containment, plus the keep-list re-checked.

    The standing ``_assert_safe`` cannot serve here — tier 4 deliberately reaches
    outside ``checkpoints/`` and ``eval_traces/`` — so the guarantee is restated
    positively instead: nothing escapes the run dir, and nothing the keep-list
    names (or lives under a kept path) is in the deletion set.
    """
    run_abs = os.path.realpath(run_dir)
    keep = set(keep_rel)
    for rel in delete_rel:
        p_abs = os.path.realpath(os.path.join(run_dir, rel))
        back = os.path.relpath(p_abs, run_abs)
        if back.startswith(".."):
            raise TieredRefusal(f"{rel} escapes the run dir {run_dir}")
        if rel in keep:
            raise TieredRefusal(f"{rel} is on the keep-list")
        for k in keep:
            if rel == k or k.startswith(rel.rstrip("/") + os.sep):
                raise TieredRefusal(f"{rel} contains the kept path {k}")


# --------------------------------------------------------------- reporting ----

TIER_ORDER = [
    (0, "LIVE", "live, or reached for by something live — untouched"),
    (1, "REFERENCED", "named by a script, a measurement artifact, the ledger tail, "
                      "or another run's model graph — standing policy + snapshots rule"),
    (2, "v9+ CLOSED", "standing policy + snapshots rule"),
    (3, "v8 CLOSED", "first + last + latest.txt pin (no every-10th) + snapshots rule"),
    (4, "PRE-v8", "AGGRESSIVE keep-list — the era's record survives, the weights do not"),
]


def tier_table(runs: dict) -> "list[dict]":
    rows = []
    for tier, tname, what in TIER_ORDER:
        members = [r for r in runs.values() if r.get("tier") == tier]
        rows.append({
            "tier": tier, "name": tname, "policy": what,
            "n_runs": len(members),
            "gb_now": round(sum(r["sizes"]["total"] for r in members) / 1e9, 3),
            "gb_freed": round(
                sum(r["plan"]["bytes_freed"] for r in members) / 1e9, 3),
        })
    return rows


def snapshot_counts(runs: dict) -> dict:
    out = {"kept": 0, "deleted": 0, "absent": 0, "skipped": 0,
           "gb_kept": 0.0, "gb_freed": 0.0, "gb_thin_proposal": 0.0}
    for r in runs.values():
        s = r.get("plan", {}).get("snapshots")
        if not s:
            out["skipped"] += 1
            continue
        act = s["action"]
        out[{"keep": "kept", "delete": "deleted", "absent": "absent"}.get(
            act, "skipped")] += 1
        if act == "keep":
            out["gb_kept"] += s["bytes"] / 1e9
            out["gb_thin_proposal"] += s["thin_proposal_bytes"] / 1e9
        elif act == "delete":
            out["gb_freed"] += s["bytes"] / 1e9
    for k in ("gb_kept", "gb_freed", "gb_thin_proposal"):
        out[k] = round(out[k], 3)
    return out


def render_tiered(c: dict) -> str:
    """The tiered-policy sections of the markdown report."""
    runs = c["runs"]
    L: list[str] = []
    A = L.append

    A("## The TIERED policy\n")
    A("> **The owner's reason for tier 4, verbatim (2026-09-06):** *\"Yes, please "
      "work on a reasonable retention policy, especially pre ai_v8 eras, as we are "
      "unlikely to need anything from them as there wasn't a 'novel' outcome, more "
      "getting the pattern established and us able to make meaningful progress.\"*\n")
    A("| tier | who | GB now | GB freed | runs | what happens |")
    A("|---:|---|---:|---:|---:|---|")
    for row in c["tiers"]["table"]:
        A(f"| {row['tier']} | {row['name']} | {row['gb_now']} | {row['gb_freed']} | "
          f"{row['n_runs']} | {row['policy']} |")
    A("")

    s = c["tiers"]["snapshots"]
    A("### The snapshots rule\n")
    A("A self-play pool is kept **only** when some run forks this run — a fork "
      "auto-seeds its parent's pool, so the zips *and* `summary.json` / "
      "`win_rate_vs_bots.txt` / `model_config.json` are load-bearing — or a "
      "committed script names the run as a `--stable-opponents` / `--exploiter` / "
      "pool source. Otherwise `snapshots/` goes whole.\n")
    A("| | |")
    A("|---|---:|")
    A(f"| pools KEPT | {s['kept']} ({s['gb_kept']} GB) |")
    A(f"| pools FREED | {s['deleted']} ({s['gb_freed']} GB) |")
    A(f"| runs with no pool | {s['absent']} |")
    A(f"| tier-0 runs (rule not applied) | {s['skipped']} |")
    A(f"| **PROPOSED** further thinning of the KEPT pools "
      f"(every {SNAPSHOT_THIN_EVERY}th + newest) | **{s['gb_thin_proposal']} GB** |")
    A("")
    A(f"The thinning is a **proposal, not a plan** — no kept pool loses a byte in "
      f"this policy. It is reported so the {s['gb_thin_proposal']} GB is a number "
      "the owner can decide on rather than a discovery made later.\n")

    A("### Every pool decision, per run\n")
    A("| run | tier | pool GB | decision | why |")
    A("|---|---:|---:|---|---|")
    for n in sorted(runs, key=lambda k: -(runs[k]["plan"].get("snapshots") or
                                          {"bytes": 0})["bytes"]):
        sp = runs[n]["plan"].get("snapshots")
        if not sp or sp["action"] == "absent":
            continue
        A(f"| `{n}` | {runs[n]['tier']} | {round(sp['bytes'] / 1e9, 3)} | "
          f"{sp['action'].upper()} | {sp['reason']} |")
    A("")

    A("### Tier 4 — what each pre-v8 run keeps\n")
    A("The keep-list is the policy. `resolve_model_ref` picks the ONE model file, "
      "and the **rung** it fired on is recorded, because a bare run dir has meant "
      "different files at different times (`gen3_last_snapshot_resolution_v1`).\n")
    A("| run | GB freed | the kept model | rung | steps |")
    A("|---|---:|---|---|---:|")
    t4 = [n for n in sorted(runs) if runs[n].get("tier") == 4]
    for n in sorted(t4, key=lambda k: -runs[k]["plan"]["bytes_freed"]):
        r = runs[n]
        res = r["plan"].get("resolved") or {}
        if not res.get("ok"):
            A(f"| `{n}` | 0 | **REFUSED** — {res.get('error', 'unresolved')} | — | — |")
            continue
        ts = res.get("num_timesteps")
        A(f"| `{n}` | {round(r['plan']['bytes_freed'] / 1e9, 3)} | "
          f"`{res['rel']}` | {res['rung']} | {ts if ts is not None else '—'} |")
    if not t4:
        A("| — | | *(no run graded tier 4)* | | |")
    A("")
    A("**Consequence, stated plainly:** a tier-4 run becomes un-probeable except at "
      "its final checkpoint. That costs less than it sounds: root `CLAUDE.md` records "
      "that on 2026-08-13 **79 of 79 archived runs could not be re-loaded** at the "
      "then-current architecture, and the drift has only grown since — so every "
      "model-loading prober view (`analyze` / `lookahead` / `better-line` / "
      "`replay-counterfactual` / `probe`) already returns an `ArchDriftError` on "
      "these runs. What survives is exactly what still works on them: `tb/`, the ELO "
      "ladder, `eval_results.jsonl`, and the model-free prober views.\n")
    return "\n".join(L)
