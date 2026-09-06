#!/usr/bin/env python3
"""Archive-grooming census for ``models/`` — DRY RUN unless ``--apply`` is passed.

The archive is the one thing in this project that is never reproducible, so this
tool is built to REFUSE rather than to reclaim.  It reports, per run:

* generation (``ai_vN`` from the name, else the lineage), ``config_version``;
* status — ``LIVE`` (a launcher/trainer process names the run dir), ``REFERENCED``
  (the ledger tail, a committed *script*, the v8 era, recent activity, or a fork
  link into a live run), or ``CLOSED``;
* per-subdir sizes;
* what the retention policy would DELETE and what it would KEEP *with the reason*.

**The policy is applied to CLOSED runs only**, and it touches exactly two places:

* ``checkpoints/`` — keep the FIRST, the LAST, every 10th, whatever ``latest.txt``
  pins, and any checkpoint another run's ``lineage`` block resolved to.  Each
  ``.zip`` travels with its ``.json`` sidecar.
* ``eval_traces/`` — ``main.prober.groom`` at ``3/1`` (the standing policy), and
  it is the groomer's OWN planner that is called, never a re-implementation.

Everything else is out of scope by construction: ``tb/`` is never thinned,
``best_model/`` / ``snapshots/`` / ``snapshot_ladder/`` / ``cf_*`` / ``elo/`` are
never touched, and ``metadata.json`` / ``model_config.json`` / ``latest.txt`` /
``eval_results.jsonl`` are never candidates.  ``_assert_safe`` re-checks every
planned path against those rules before anything is reported or removed, so a
future edit to the planner cannot quietly widen the blast radius.

A run is dropped from the deletion set entirely — and listed separately — when a
committed file (of ANY kind, prose included) or the ledger tail names one of the
exact files the plan would delete.

**Prose that merely names a run does not protect the run.** The historical record
names nearly every run forever, so treating a `.md` mention as a live reference
would make nothing closable and the census useless.  A committed *script* does
protect it — a script names a run dir in order to load it — and prose still
vetoes when it names an exact planned path.

**TWO POLICIES live here.** ``--policy standing`` (the DEFAULT) is everything above
and is unchanged. ``--policy tiered`` grades each run by ERA and by whether anything
still reaches for it, and adds a rule for ``snapshots/`` — the archive's second
largest consumer, which the standing policy leaves entirely alone. The tiers, the
reference graph they are built on, and the owner's reason for the aggressive pre-v8
tier live in the sibling module ``archive_grooming_tiers.py``.

Run:
    python designs/research_state/measurements/archive_grooming_dryrun.py
    python designs/research_state/measurements/archive_grooming_dryrun.py --policy tiered

    # what the owner would run to actually delete (NOT this pass):
    python designs/research_state/measurements/archive_grooming_dryrun.py --apply
    python designs/research_state/measurements/archive_grooming_dryrun.py \
        --policy tiered --apply

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

# --- bootstrap: put src/ on the path so main.prober.groom imports.  This cannot
# go through utils.paths, which is itself only importable once src/ is there.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SRC = os.path.join(_REPO, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

if _HERE not in sys.path:                    # so the sibling tiers module imports
    sys.path.insert(0, _HERE)

import archive_grooming_tiers as tiers  # noqa: E402  (after the sys.path bootstrap)
from main.prober.groom import groom_run  # noqa: E402  (after the sys.path bootstrap)

# ---------------------------------------------------------------- policy ----

KEEP_TRACE_STEPS = 3          # `prober.groom 3/1`, the standing policy
KEEP_SNAPSHOTS = 1
CHECKPOINT_EVERY = 10         # keep first + last + every 10th
RECENT_DAYS = 7               # a run touched this recently is treated as REFERENCED

#: subdirectories of a run the planner may ever propose deleting inside.
TOUCHABLE_SUBDIRS = ("checkpoints", "eval_traces")

#: subdirectories that must never appear in a deletion set, at any depth.
PROTECTED_SUBDIRS = frozenset({
    "tb", "best_model", "snapshots", "snapshot_ladder", "elo",
    "cf_records", "cf_labels", "stalls", "crashes", "tb_imgs",
})

#: run-root files that must never appear in a deletion set.
PROTECTED_FILES = frozenset({
    "metadata.json", "model_config.json", "latest.txt", "eval_results.jsonl",
    "command.txt", "team_winrates.json", "team_winrates_history.jsonl",
    "capacity_battery.json", "launcher_child.log",
})

#: directory names under models/ that are not runs at all.
NON_RUN_DIRS = frozenset({"_arch", "_goldens", "saved_work"})

_CKPT_RE = re.compile(r"^checkpoint_(\d+)_steps\.(?:zip|json)$")
_GEN_RE = re.compile(r"ai_v(\d+)")

# ------------------------------------------------------------- utilities ----


def dir_stat(path: str) -> "tuple[int, float]":
    """``(bytes, newest file mtime)`` under ``path``; ``(0, 0.0)`` if absent."""
    total = 0
    newest = 0.0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                st = os.stat(os.path.join(root, f))
            except OSError:
                continue
            total += st.st_size
            if st.st_mtime > newest:
                newest = st.st_mtime
    return total, newest


def dir_size(path: str) -> int:
    """Bytes under ``path`` (0 when it does not exist)."""
    return dir_stat(path)[0]


def _gb(n: int) -> float:
    return round(n / 1e9, 3)


def read_json(path: str) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------ checkpoint policy ----


def checkpoint_step(name: str) -> "int | None":
    m = _CKPT_RE.match(name)
    return int(m.group(1)) if m else None


def plan_checkpoints(
    names: Iterable[str],
    latest_pin: "str | None" = None,
    pinned_files: "Iterable[str] | None" = None,
    every: int = CHECKPOINT_EVERY,
) -> "tuple[dict[str, str], list[str]]":
    """Split ``checkpoints/`` filenames into keep-with-reason and delete.

    ``names`` are bare filenames.  ``latest_pin`` is ``latest.txt``'s content (a
    run-relative path).  ``pinned_files`` are bare filenames another run's
    lineage resolved to.  A ``.json`` sidecar always follows its ``.zip``,
    because they are kept or dropped by STEP, not by filename.
    """
    names = list(names)
    pinned = set(pinned_files or ())
    steps = sorted({s for s in (checkpoint_step(n) for n in names) if s is not None})
    if not steps:
        # anything unrecognised in checkpoints/ is kept, never guessed at
        return ({n: "unrecognised name — kept" for n in sorted(names)}, [])

    step_set = set(steps)
    keep_steps: dict[int, str] = {}

    def note(step: int, reason: str) -> None:
        if step in keep_steps:
            if reason not in keep_steps[step]:
                keep_steps[step] += f", {reason}"
        else:
            keep_steps[step] = reason

    note(steps[0], "first")
    note(steps[-1], "last")
    # ``every=0`` means NO stride — the ends and the pins only.  That is tier 3's
    # rule, and it must not be spelled as a huge stride: `i % 10**9` still keeps
    # index 0, which reads as "first" twice and hides the difference.
    if every:
        for i, s in enumerate(steps):
            if i % every == 0:
                note(s, f"every-{every}th")

    if latest_pin:
        pin_step = checkpoint_step(os.path.basename(latest_pin.strip()))
        if pin_step is not None and pin_step in step_set:
            note(pin_step, "latest.txt pin")

    for fname in sorted(pinned):
        s = checkpoint_step(os.path.basename(fname))
        if s is not None and s in step_set:
            note(s, "referenced by another run's lineage")

    keep: dict[str, str] = {}
    delete: list[str] = []
    for n in sorted(names):
        s = checkpoint_step(n)
        if s is None:
            keep[n] = "unrecognised name — kept"
        elif s in keep_steps:
            keep[n] = keep_steps[s]
        else:
            delete.append(n)
    return keep, delete


# ----------------------------------------------------------- trace policy ----


def plan_traces(run_dir: str) -> dict:
    """The groomer's OWN dry-run plan at the standing 3/1 retention."""
    return groom_run(run_dir, keep_trace_steps=KEEP_TRACE_STEPS,
                     keep_snapshots=KEEP_SNAPSHOTS, apply=False)


# ------------------------------------------------------------ safety net ----


class UnsafeDeletion(RuntimeError):
    """A planned path violates the never-delete rules."""


def _assert_safe(run_dir: str, paths: Iterable[str]) -> None:
    """Raise unless every path is inside a touchable subdir of ``run_dir``.

    Containment is checked on the REALPATH of both sides.  Eight run dirs in the
    live archive are symlinks into launcher worktrees, so an ``abspath``-only
    check would compare a path that reads ``models/<run>/checkpoints/x.zip``
    against a file that physically lives under ``.claude/worktrees/`` — the guard
    would pass while describing the wrong location.
    """
    run_abs = os.path.realpath(run_dir)
    for p in paths:
        p_abs = os.path.realpath(p)
        rel = os.path.relpath(p_abs, run_abs)
        if rel.startswith(".."):
            raise UnsafeDeletion(f"{p} escapes the run dir {run_dir}")
        parts = rel.split(os.sep)
        if len(parts) < 2:
            raise UnsafeDeletion(f"{p} is a run-root entry — never a candidate")
        if parts[0] not in TOUCHABLE_SUBDIRS:
            raise UnsafeDeletion(f"{p} is outside {TOUCHABLE_SUBDIRS}")
        if PROTECTED_SUBDIRS.intersection(parts):
            raise UnsafeDeletion(f"{p} touches a protected subdir")
        if parts[-1] in PROTECTED_FILES:
            raise UnsafeDeletion(f"{p} is a protected file")


# ------------------------------------------------------ reference scanning ----


def live_run_dirs(models_dir: str) -> "dict[str, str]":
    """Run names named by a live launcher/trainer process argv → that argv."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args", "--no-headers"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    names = set(os.listdir(models_dir)) if os.path.isdir(models_dir) else set()
    hits: dict[str, str] = {}
    for line in out.splitlines():
        if not any(k in line for k in ("main.launcher", "train_rl_agent", "main.train")):
            continue
        for n in names:
            if n and n in line:
                hits[n] = line.strip()[:300]
    return hits


_SKIP_EXT = frozenset({
    ".npz", ".npy", ".png", ".jpg", ".jpeg", ".gz", ".zip", ".pyc", ".pt",
    ".bin", ".ico", ".woff", ".woff2", ".pdf", ".webp",
})
_SKIP_PREFIX = ("data/", "deps/", "src/poke_env/", "src/rust_sim/tests/vectors/")

#: extensions whose files can EXECUTE against a run dir.  A run one of these
#: names is protected whole; a run named only in prose is not — the historical
#: record names nearly every run forever, so treating prose as a reference makes
#: nothing closable.  Prose still VETOES when it names an exact planned path.
_SCRIPT_EXT = frozenset({".py", ".sh", ".bash", ".ipynb"})


def _is_script(rel: str) -> bool:
    return os.path.splitext(rel)[1].lower() in _SCRIPT_EXT


def committed_files(repo_root: str) -> "list[str]":
    """Committed text files worth scanning for run references."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "ls-files"],
            capture_output=True, text=True, timeout=180, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    keep = []
    for rel in out.splitlines():
        if rel.startswith(_SKIP_PREFIX):
            continue
        if os.path.splitext(rel)[1].lower() in _SKIP_EXT:
            continue
        keep.append(rel)
    return keep


def build_reference_index(
    repo_root: str, run_names: "list[str]", ledger_tail_lines: int = 1000,
) -> dict:
    """Which committed files (and the ledger tail) name each run, and which exact
    ``<run>/<path>`` strings they name."""
    empty = {"by_run": {}, "named_paths": {}, "ledger_runs": set(),
             "ledger_anywhere": {}}
    if not run_names:
        return empty

    # longest first, so `X_exploiter_v1` wins over its own prefix `X`
    alt = "|".join(re.escape(n) for n in sorted(run_names, key=len, reverse=True))
    rx = re.compile(r"(" + alt + r")(/[A-Za-z0-9_.\-/]*)?")

    by_run: dict[str, list[str]] = {}
    named_paths: dict[str, set[str]] = {}

    def scan(text: str, origin: str, into_run: dict) -> None:
        for name, rest in rx.findall(text):
            into_run.setdefault(name, [])
            if origin not in into_run[name]:
                into_run[name].append(origin)
            if rest and len(rest) > 1:
                named_paths.setdefault(name, set()).add(
                    rest.lstrip("/").rstrip("/.,;:)\"'`"))

    for rel in committed_files(repo_root):
        full = os.path.join(repo_root, rel)
        try:
            if os.path.getsize(full) > 8_000_000:
                continue
            with open(full, "rb") as fh:
                text = fh.read().decode("utf-8", "ignore")
        except OSError:
            continue
        scan(text, rel, by_run)

    ledger_runs: dict[str, list[str]] = {}
    ledger_anywhere: dict[str, int] = {}
    ledger = os.path.join(repo_root, "designs", "research_state", "ledger.md")
    if os.path.isfile(ledger):
        try:
            with open(ledger, "rb") as fh:
                lines = fh.read().decode("utf-8", "ignore").splitlines()
            scan("\n".join(lines[-ledger_tail_lines:]), "ledger.md (tail)", ledger_runs)
            # the WHOLE ledger, for the review flag — a run the tail does not name
            # can still be the subject of a banked result further up
            for i, line in enumerate(lines, 1):
                for name, _rest in rx.findall(line):
                    ledger_anywhere.setdefault(name, i)
        except OSError:
            pass

    return {"by_run": by_run, "named_paths": named_paths,
            "ledger_runs": set(ledger_runs), "ledger_anywhere": ledger_anywhere}


# ----------------------------------------------------------- run scanning ----

_KNOWN_SUBDIRS = ("checkpoints", "best_model", "snapshots", "eval_traces", "tb",
                  "snapshot_ladder", "cf_records", "cf_labels", "stalls",
                  "crashes", "elo", "tb_imgs")

#: Recency is read from TRAINING OUTPUT only.  The run-root bookkeeping files are
#: rewritten wholesale by maintenance passes — a `main.lineage --backfill` on
#: 2026-09-01 restamped 153 of 217 `metadata.json` in one go — so a run-root
#: mtime reads the entire archive as fresh and the recency guard protects
#: everything.  Measured: checkpoint mtimes span 2026-06 … 2026-09 and do
#: separate the runs; metadata.json mtimes do not.
_ACTIVITY_SUBDIRS = ("checkpoints", "tb", "eval_traces", "snapshots", "best_model")


def discover_runs(models_dir: str) -> "list[str]":
    """Directories under ``models_dir`` that look like runs."""
    out = []
    for name in sorted(os.listdir(models_dir)):
        full = os.path.join(models_dir, name)
        if not os.path.isdir(full) or name in NON_RUN_DIRS:
            continue
        if not (os.path.isfile(os.path.join(full, "model_config.json"))
                or os.path.isfile(os.path.join(full, "metadata.json"))
                or os.path.isdir(os.path.join(full, "checkpoints"))):
            continue
        out.append(name)
    return out


def generation_of(name: str, lineage: dict) -> str:
    m = _GEN_RE.search(name)
    if m:
        return f"ai_v{m.group(1)}"
    if name.startswith("v8rep"):
        return "ai_v8 (replication)"
    for entry in reversed(lineage.get("ancestry") or []):
        m = _GEN_RE.search(entry.get("run_name") or "")
        if m:
            return f"ai_v{m.group(1)} (via lineage)"
    m = _GEN_RE.search((lineage.get("fork_parent") or {}).get("run_name") or "")
    if m:
        return f"ai_v{m.group(1)} (via parent)"
    return "unknown"


def is_v8_era(name: str, generation: str) -> bool:
    return name.startswith("v8rep") or "ai_v8" in name or generation.startswith("ai_v8")


def lineage_refs(lineage: dict) -> "list[list[str]]":
    """``[kind, run_name, resolved_file]`` for every model this run's lineage names."""
    out: list[list[str]] = []

    def one(kind: str, blk: Any) -> None:
        if not isinstance(blk, dict):
            return
        rn = blk.get("run_name") or ""
        f = blk.get("resolved_file") or blk.get("resolved_path") or blk.get("path") or ""
        if rn:
            out.append([kind, rn, f])

    one("fork_parent", lineage.get("fork_parent"))
    for t in lineage.get("teachers") or []:
        one("teacher", t)
    one("exploiter_target", lineage.get("exploiter_target"))
    return out


def scan_run(models_dir: str, name: str) -> dict:
    run_dir = os.path.join(models_dir, name)
    meta = read_json(os.path.join(run_dir, "metadata.json"))
    mc = read_json(os.path.join(run_dir, "model_config.json"))
    lineage = meta.get("lineage") or {}

    entries = sorted(os.listdir(run_dir))
    stats = {k: dir_stat(os.path.join(run_dir, k)) for k in _KNOWN_SUBDIRS}
    sizes = {k: v[0] for k, v in stats.items()}
    other = 0
    for e in entries:
        if e in _KNOWN_SUBDIRS:
            continue
        p = os.path.join(run_dir, e)
        if os.path.isdir(p):
            other += dir_size(p)
        elif os.path.isfile(p):
            try:
                other += os.path.getsize(p)
            except OSError:
                pass
    sizes["other"] = other
    sizes["total"] = sum(v for k, v in sizes.items() if k != "total")

    latest_pin = None
    lp = os.path.join(run_dir, "latest.txt")
    if os.path.isfile(lp):
        try:
            with open(lp) as fh:
                latest_pin = fh.read().strip()
        except OSError:
            pass

    # recency from TRAINING OUTPUT only — see _ACTIVITY_SUBDIRS
    mtime = 0.0
    mtime_source = ""
    for k in _ACTIVITY_SUBDIRS:
        if stats[k][1] > mtime:
            mtime, mtime_source = stats[k][1], k

    ck_dir = os.path.join(run_dir, "checkpoints")
    return {
        "name": name,
        "run_dir": run_dir,
        "is_symlink": os.path.islink(run_dir),
        "realpath": os.path.realpath(run_dir),
        "generation": generation_of(name, lineage),
        "config_version": mc.get("config_version"),
        "arch_signature": mc.get("arch_signature"),
        "role": lineage.get("role"),
        "num_timesteps": meta.get("num_timesteps"),
        "lineage_refs": lineage_refs(lineage),
        "latest_pin": latest_pin,
        "sizes": sizes,
        "mtime": mtime,
        "mtime_source": mtime_source,
        "mtime_iso": time.strftime("%Y-%m-%d", time.localtime(mtime)) if mtime else "",
        "checkpoint_names": sorted(os.listdir(ck_dir)) if os.path.isdir(ck_dir) else [],
    }


# ------------------------------------------------------------- the census ----


def _empty_plan() -> dict:
    return {"delete": [], "keep": {}, "bytes_freed": 0, "gb_freed": 0.0,
            "checkpoints_deleted": 0, "trace_steps_deleted": 0,
            "snapshots_dropped": 0}


def build_census(models_dir: str, repo_root: str, recent_days: int = RECENT_DAYS,
                 ledger_tail_lines: int = 1000,
                 follow_symlinked_runs: bool = False) -> dict:
    names = discover_runs(models_dir)
    runs = {n: scan_run(models_dir, n) for n in names}

    live = live_run_dirs(models_dir)
    refs = build_reference_index(repo_root, names, ledger_tail_lines)
    now = time.time()

    # who points at whom (parent -> children, parent -> the exact files pinned)
    pinned_by_run: dict[str, set[str]] = {}
    referenced_by: dict[str, list[str]] = {}
    for child, r in runs.items():
        for kind, parent, resolved in r["lineage_refs"]:
            if parent not in runs:
                continue
            referenced_by.setdefault(parent, []).append(f"{child} ({kind})")
            if resolved:
                pinned_by_run.setdefault(parent, set()).add(os.path.basename(resolved))

    # ---- status pass 1: direct signals
    for n, r in runs.items():
        reasons: list[str] = []
        if n in live:
            reasons.append("LIVE launcher process")
        if n in refs["ledger_runs"]:
            reasons.append(f"named in the ledger's last {ledger_tail_lines} lines")
        origins = [o for o in refs["by_run"].get(n, []) if o != "ledger.md (tail)"]
        scripts = [o for o in origins if _is_script(o)]
        prose = [o for o in origins if not _is_script(o)]
        r["named_by_committed_scripts"] = scripts
        r["named_by_committed_prose"] = prose[:12]
        r["n_named_by_committed_prose"] = len(prose)
        if scripts:
            shown = ", ".join(scripts[:4]) + (" …" if len(scripts) > 4 else "")
            reasons.append(f"loaded by {len(scripts)} committed script(s): {shown}")
        if is_v8_era(n, r["generation"]):
            reasons.append("v8-era (the era replication + head-to-head load these)")
        if recent_days and r["mtime"] and (now - r["mtime"]) < recent_days * 86400:
            reasons.append(f"training output written within {recent_days} days "
                           f"({r['mtime_source']}/, {r['mtime_iso']})")
        if r["is_symlink"] and not follow_symlinked_runs:
            reasons.append("run dir is a SYMLINK — the data lives at "
                           f"{r['realpath']}, outside models/ "
                           "(--follow-symlinked-runs to include it)")
        r["_reasons"] = reasons
        r["live_argv"] = live.get(n)

    # ---- status pass 2: fork parents of LIVE runs, transitively
    frontier = {n for n in runs if n in live}
    seen = set(frontier)
    while frontier:
        nxt = set()
        for child in frontier:
            for _kind, parent, _f in runs[child]["lineage_refs"]:
                if parent in runs and parent not in seen:
                    seen.add(parent)
                    nxt.add(parent)
                    runs[parent]["_reasons"].append(
                        "fork parent (transitively) of a LIVE run")
        frontier = nxt

    for n, r in runs.items():
        if n in live:
            r["status"] = "LIVE"
        elif r["_reasons"]:
            r["status"] = "REFERENCED"
        else:
            r["status"] = "CLOSED"
        r["status_reasons"] = r.pop("_reasons")
        r["referenced_by_runs"] = sorted(set(referenced_by.get(n, [])))

    # ---- the plan, for CLOSED runs only
    excluded_by_named_file: list[dict] = []
    for n, r in runs.items():
        r["plan"] = _empty_plan()
        if r["status"] != "CLOSED":
            r["plan"]["skipped"] = f"status={r['status']}"
            continue

        run_dir = r["run_dir"]
        ck_dir = os.path.join(run_dir, "checkpoints")
        keep, delete = plan_checkpoints(
            r["checkpoint_names"], r["latest_pin"], pinned_by_run.get(n, set()))
        del_paths = [os.path.join("checkpoints", d) for d in delete]
        freed = 0
        for d in delete:
            try:
                freed += os.path.getsize(os.path.join(ck_dir, d))
            except OSError:
                pass

        tr = plan_traces(run_dir)
        trace_paths = []
        for e in tr["plan"]:
            if e["action"] == "remove_step":
                trace_paths.append(os.path.join("eval_traces", f"step_{e['step']}"))
            elif e["action"] == "drop_snapshot":
                trace_paths.append(
                    os.path.join("eval_traces", f"step_{e['step']}", "snapshot.zip"))
        freed += tr["bytes_reclaimed"]

        all_rel = del_paths + trace_paths
        _assert_safe(run_dir, [os.path.join(run_dir, p) for p in all_rel])

        # a committed file / the ledger naming one of these exact paths vetoes the run
        named = refs["named_paths"].get(n, set())
        collisions = sorted(
            p for p in all_rel
            if any(nm == p or nm.startswith(p + "/")
                   or p.startswith(nm.rstrip("/") + "/")
                   or os.path.basename(nm) == os.path.basename(p)
                   for nm in named)
        )
        if collisions:
            excluded_by_named_file.append({
                "run": n,
                "collisions": collisions[:20],
                "n_collisions": len(collisions),
                "would_have_freed_gb": _gb(freed),
                "named_by": refs["by_run"].get(n, [])[:6],
            })
            r["plan"] = _empty_plan()
            r["plan"]["skipped"] = "a committed file or the ledger names a file in the plan"
            r["status"] = "REFERENCED"
            r["status_reasons"].append(
                "a committed file / the ledger names a file the plan would delete")
            continue

        r["plan"] = {
            "delete": all_rel,
            "keep": keep,
            "bytes_freed": freed,
            "gb_freed": _gb(freed),
            "checkpoints_deleted": len(delete),
            "trace_steps_deleted": len(tr["removed_steps"]),
            "snapshots_dropped": len(tr["dropped_snapshots"]),
        }

    # REVIEW FLAG — a CLOSED run the ledger names somewhere OUTSIDE its tail.  The
    # tail rule is what protects; this is what makes its edge visible instead of
    # silent, so a banked result further up the ledger gets a human look before
    # its checkpoints are thinned.  It does NOT change the plan.
    needs_review = []
    for n, r in sorted(runs.items()):
        if r["status"] != "CLOSED" or not r["plan"]["delete"]:
            continue
        line = refs["ledger_anywhere"].get(n)
        if line is not None:
            needs_review.append({
                "run": n, "ledger_line": line,
                "gb_freed": _gb(r["plan"]["bytes_freed"]),
                "generation": r["generation"],
                "n_named_by_committed_prose": r.get("n_named_by_committed_prose", 0),
            })
            r["review_flag"] = f"named at ledger.md:{line}, outside the tail window"

    symlinked = [
        {"run": n, "realpath": r["realpath"], "gb": _gb(r["sizes"]["total"]),
         "status": r["status"], "generation": r["generation"]}
        for n, r in sorted(runs.items()) if r["is_symlink"]
    ]

    total_bytes = sum(r["sizes"]["total"] for r in runs.values())
    symlinked_bytes = sum(r["sizes"]["total"] for r in runs.values() if r["is_symlink"])
    freed_bytes = sum(r["plan"]["bytes_freed"] for r in runs.values())
    by_freed = sorted(runs.values(), key=lambda r: -r["plan"]["bytes_freed"])

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models_dir": os.path.abspath(models_dir),
        "repo_root": os.path.abspath(repo_root),
        "applied": False,
        "policy": {
            "keep_trace_steps": KEEP_TRACE_STEPS,
            "keep_snapshots": KEEP_SNAPSHOTS,
            "checkpoint_every": CHECKPOINT_EVERY,
            "recent_days": recent_days,
            "ledger_tail_lines": ledger_tail_lines,
            "touchable_subdirs": list(TOUCHABLE_SUBDIRS),
            "protected_subdirs": sorted(PROTECTED_SUBDIRS),
            "protected_files": sorted(PROTECTED_FILES),
        },
        "totals": {
            "n_runs": len(runs),
            "n_live": sum(1 for r in runs.values() if r["status"] == "LIVE"),
            "n_referenced": sum(1 for r in runs.values() if r["status"] == "REFERENCED"),
            "n_closed": sum(1 for r in runs.values() if r["status"] == "CLOSED"),
            "n_with_plan": sum(1 for r in runs.values() if r["plan"]["delete"]),
            "archive_bytes": total_bytes,
            "archive_gb": _gb(total_bytes),
            "n_symlinked_runs": len(symlinked),
            "symlinked_gb": _gb(symlinked_bytes),
            "in_models_gb": _gb(total_bytes - symlinked_bytes),
            "freed_bytes": freed_bytes,
            "freed_gb": _gb(freed_bytes),
            "freed_pct": round(100.0 * freed_bytes / total_bytes, 1) if total_bytes else 0.0,
            "n_files_planned": sum(len(r["plan"]["delete"]) for r in runs.values()),
            "n_excluded_by_named_file": len(excluded_by_named_file),
            "n_needs_review": len(needs_review),
        },
        "needs_review": needs_review,
        "top20_by_gb_freed": [
            {"run": r["name"], "gb_freed": _gb(r["plan"]["bytes_freed"]),
             "generation": r["generation"], "status": r["status"],
             "checkpoints_deleted": r["plan"]["checkpoints_deleted"],
             "trace_steps_deleted": r["plan"]["trace_steps_deleted"]}
            for r in by_freed[:20] if r["plan"]["bytes_freed"] > 0
        ],
        "excluded_by_named_file": excluded_by_named_file,
        "symlinked_runs": symlinked,
        "runs": {n: {k: v for k, v in r.items() if k != "checkpoint_names"}
                 for n, r in runs.items()},
    }


# ------------------------------------------------------- the TIERED census ----


def _named_path_collisions(name: str, all_rel: "list[str]", refs: dict) -> "list[str]":
    """Exact paths a committed file / the ledger names that this plan would delete.

    Unchanged from the standing policy and deliberately shared: the veto is the
    last-resort net under EVERY policy, so it must be one implementation.
    """
    named = refs["named_paths"].get(name, set())
    return sorted(
        p for p in all_rel
        if any(nm == p or nm.startswith(p + "/")
               or p.startswith(nm.rstrip("/") + "/")
               or os.path.basename(nm) == os.path.basename(p)
               for nm in named)
    )


def _standing_style_plan(r: dict, every: int, pinned: "set[str]") -> "tuple[list, dict, int, dict]":
    """`checkpoints/` at the given stride + `eval_traces/` at the standing 3/1."""
    run_dir = r["run_dir"]
    ck_dir = os.path.join(run_dir, "checkpoints")
    keep, delete = plan_checkpoints(r["checkpoint_names"], r["latest_pin"], pinned,
                                    every=every)
    del_paths = [os.path.join("checkpoints", d) for d in delete]
    freed = 0
    for d in delete:
        try:
            freed += os.path.getsize(os.path.join(ck_dir, d))
        except OSError:
            pass
    tr = plan_traces(run_dir)
    trace_paths = []
    for e in tr["plan"]:
        if e["action"] == "remove_step":
            trace_paths.append(os.path.join("eval_traces", f"step_{e['step']}"))
        elif e["action"] == "drop_snapshot":
            trace_paths.append(
                os.path.join("eval_traces", f"step_{e['step']}", "snapshot.zip"))
    freed += tr["bytes_reclaimed"]
    return del_paths + trace_paths, keep, freed, tr


def build_census_tiered(models_dir: str, repo_root: str,
                        recent_days: int = RECENT_DAYS,
                        ledger_tail_lines: int = tiers.TIERED_LEDGER_TAIL_LINES,
                        follow_symlinked_runs: bool = False) -> dict:
    """The tiered policy.  The standing :func:`build_census` is untouched by it.

    Discovery, the reference index, the symlink hold-out and the named-path veto
    are the SAME code the standing policy runs — only the grading and the per-tier
    plan differ, so a divergence between the two policies can only ever be a
    deliberate one.
    """
    names = discover_runs(models_dir)
    runs = {n: scan_run(models_dir, n) for n in names}
    live = live_run_dirs(models_dir)
    refs = build_reference_index(repo_root, names, ledger_tail_lines)
    now = time.time()

    graph = tiers.build_model_graph(runs)
    tiers.assign_tiers(runs, live, refs, graph, recent_days, ledger_tail_lines,
                       follow_symlinked_runs, now)

    excluded_by_named_file: list[dict] = []
    refusals: list[dict] = []

    for n, r in runs.items():
        r["plan"] = _empty_plan()
        r["live_argv"] = live.get(n)
        r["status"] = {0: "LIVE", 1: "REFERENCED"}.get(r["tier"], "CLOSED")
        r["status_reasons"] = list(r["tier_reasons"])
        r["referenced_by_runs"] = sorted(set(graph["referenced_by"].get(n, [])))
        r["model_refs"] = graph["refs_out"].get(n, [])
        if r["tier"] == 0:
            r["plan"]["skipped"] = "tier 0 — LIVE or reached for by something live"
            continue
        if r.get("review_hold"):
            # A HOLD suppresses the plan, it does not merely soften it — see
            # tiers.REVIEW_HOLDS.  Held runs still appear in the review table.
            r["plan"]["skipped"] = "REVIEW HOLD — " + r["review_hold"]
            continue

        pinned = graph["pinned_files"].get(n, set())
        snap_keep, snap_reason = tiers.snapshots_verdict(n, graph, refs)

        if r["tier"] == 4:
            try:
                p4 = tiers.plan_tier4(r["run_dir"], r["latest_pin"], pinned)
            except tiers.TieredRefusal as exc:
                refusals.append({"run": n, "reason": str(exc)})
                r["plan"]["skipped"] = f"tier 4 REFUSED — {exc}"
                r["plan"]["resolved"] = {"ok": False, "error": str(exc)}
                continue
            tiers.assert_safe_tiered(r["run_dir"], p4["delete"], p4["keep"])
            all_rel, keep, freed = p4["delete"], p4["keep"], p4["bytes_freed"]
            snap = tiers.plan_snapshots(r["run_dir"], keep=False,
                                        keep_reason="tier 4 — the whole run is "
                                                    "reduced to its record + final model")
            snap["action"] = "delete" if snap["n_snapshots"] or snap["bytes"] else "absent"
            resolved = p4["resolved"]
        else:
            every = (tiers.TIER3_CHECKPOINT_EVERY if r["tier"] == 3
                     else CHECKPOINT_EVERY)
            all_rel, keep, freed, _tr = _standing_style_plan(r, every, pinned)
            _assert_safe(r["run_dir"],
                         [os.path.join(r["run_dir"], p) for p in all_rel])
            snap = tiers.plan_snapshots(r["run_dir"], snap_keep, snap_reason)
            if snap["delete"]:
                all_rel = all_rel + snap["delete"]
                freed += snap["bytes"]
            resolved = None

        collisions = _named_path_collisions(n, all_rel, refs)
        if collisions:
            excluded_by_named_file.append({
                "run": n, "collisions": collisions[:20],
                "n_collisions": len(collisions),
                "would_have_freed_gb": _gb(freed),
                "named_by": refs["by_run"].get(n, [])[:6],
            })
            r["plan"] = _empty_plan()
            r["plan"]["skipped"] = "a committed file or the ledger names a file in the plan"
            r["status"] = "REFERENCED"
            r["status_reasons"].append(
                "a committed file / the ledger names a file the plan would delete")
            continue

        # `keep` keys are BARE filenames for the standing-shaped tiers (that is
        # `plan_checkpoints`' contract, which the 34 standing tests pin) and
        # run-relative paths for tier 4.  `keep_rel` is the normalised form, so
        # the execution-time guard never has to know which tier it is holding.
        keep_rel = (sorted(keep) if r["tier"] == 4
                    else sorted(os.path.join("checkpoints", k) for k in keep))
        r["plan"] = {
            "delete": all_rel, "keep": keep, "keep_rel": keep_rel,
            "bytes_freed": freed, "gb_freed": _gb(freed),
            "checkpoints_deleted": sum(1 for p in all_rel
                                       if p.startswith("checkpoint")),
            "trace_steps_deleted": sum(1 for p in all_rel
                                       if p.startswith("eval_traces")),
            "snapshots_dropped": snap["n_snapshots"] if snap["action"] == "delete" else 0,
            "snapshots": snap,
        }
        if resolved is not None:
            r["plan"]["resolved"] = resolved

    # the review flag is the SAME edge-visibility device the standing policy uses
    needs_review = []
    for n, r in sorted(runs.items()):
        if not r["plan"]["delete"]:
            continue
        line = refs["ledger_anywhere"].get(n)
        if line is not None:
            needs_review.append({
                "run": n, "ledger_line": line, "tier": r["tier"],
                "gb_freed": _gb(r["plan"]["bytes_freed"]),
                "generation": r["generation"],
                "n_named_by_committed_prose": len(
                    [o for o in refs["by_run"].get(n, [])
                     if o != "ledger.md (tail)" and not _is_script(o)]),
            })
            r["review_flag"] = f"named at ledger.md:{line}, outside the tail window"

    symlinked = [
        {"run": n, "realpath": r["realpath"], "gb": _gb(r["sizes"]["total"]),
         "status": r["status"], "generation": r["generation"], "tier": r["tier"]}
        for n, r in sorted(runs.items()) if r["is_symlink"]
    ]
    total_bytes = sum(r["sizes"]["total"] for r in runs.values())
    symlinked_bytes = sum(r["sizes"]["total"] for r in runs.values() if r["is_symlink"])
    freed_bytes = sum(r["plan"]["bytes_freed"] for r in runs.values())
    by_freed = sorted(runs.values(), key=lambda r: -r["plan"]["bytes_freed"])

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models_dir": os.path.abspath(models_dir),
        "repo_root": os.path.abspath(repo_root),
        "applied": False,
        "policy_name": "tiered",
        "policy": {
            "keep_trace_steps": KEEP_TRACE_STEPS,
            "keep_snapshots": KEEP_SNAPSHOTS,
            "checkpoint_every": CHECKPOINT_EVERY,
            "tier3_checkpoint_every": tiers.TIER3_CHECKPOINT_EVERY,
            "snapshot_thin_every": tiers.SNAPSHOT_THIN_EVERY,
            "recent_days": recent_days,
            "ledger_tail_lines": ledger_tail_lines,
            "pre_v8_cutoff_mmdd": list(tiers.PRE_V8_CUTOFF_MMDD),
            "tier4_keep_subdirs": sorted(tiers.TIER4_KEEP_SUBDIRS),
            "tier4_keep_root_files": sorted(tiers.TIER4_KEEP_ROOT_FILES),
            "touchable_subdirs": list(TOUCHABLE_SUBDIRS),
            "protected_subdirs": sorted(PROTECTED_SUBDIRS),
            "protected_files": sorted(PROTECTED_FILES),
        },
        "tiers": {
            "table": tiers.tier_table(runs),
            "snapshots": tiers.snapshot_counts(runs),
            "refusals": refusals,
        },
        "totals": {
            "n_runs": len(runs),
            "n_live": sum(1 for r in runs.values() if r["status"] == "LIVE"),
            "n_referenced": sum(1 for r in runs.values() if r["status"] == "REFERENCED"),
            "n_closed": sum(1 for r in runs.values() if r["status"] == "CLOSED"),
            "n_with_plan": sum(1 for r in runs.values() if r["plan"]["delete"]),
            "archive_bytes": total_bytes,
            "archive_gb": _gb(total_bytes),
            "n_symlinked_runs": len(symlinked),
            "symlinked_gb": _gb(symlinked_bytes),
            "in_models_gb": _gb(total_bytes - symlinked_bytes),
            "freed_bytes": freed_bytes,
            "freed_gb": _gb(freed_bytes),
            "freed_pct": round(100.0 * freed_bytes / total_bytes, 1) if total_bytes else 0.0,
            "n_files_planned": sum(len(r["plan"]["delete"]) for r in runs.values()),
            "n_excluded_by_named_file": len(excluded_by_named_file),
            "n_needs_review": len(needs_review),
            "n_tier4_refused": len(refusals),
        },
        "needs_review": needs_review,
        "top20_by_gb_freed": [
            {"run": r["name"], "gb_freed": _gb(r["plan"]["bytes_freed"]),
             "generation": r["generation"], "status": r["status"],
             "tier": r["tier"],
             "checkpoints_deleted": r["plan"]["checkpoints_deleted"],
             "trace_steps_deleted": r["plan"]["trace_steps_deleted"]}
            for r in by_freed[:20] if r["plan"]["bytes_freed"] > 0
        ],
        "excluded_by_named_file": excluded_by_named_file,
        "symlinked_runs": symlinked,
        "runs": {n: {k: v for k, v in r.items()
                     if k not in ("checkpoint_names", "_metadata")}
                 for n, r in runs.items()},
    }


# --------------------------------------------------------------- rendering ----

APPLY_CMD = (
    "export PYTHONPATH=$PYTHONPATH:src && \\\n"
    "/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \\\n"
    "  designs/research_state/measurements/archive_grooming_dryrun.py --apply"
)


def apply_cmd(policy: str = "standing") -> str:
    """The exact command that would EXECUTE this policy.  Run it from the MAIN
    checkout — ``models/`` exists only there."""
    if policy == "tiered":
        return (
            "cd /home/goodlad/dev/gen3ai && \\\n"
            "export PYTHONPATH=$PYTHONPATH:src && \\\n"
            "/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \\\n"
            "  designs/research_state/measurements/archive_grooming_dryrun.py \\\n"
            "  --policy tiered --apply"
        )
    return APPLY_CMD


def render_markdown(c: dict) -> str:
    t = c["totals"]
    p = c["policy"]
    policy_name = c.get("policy_name", "standing")
    L: list[str] = []
    A = L.append

    A("# Archive-grooming DRY RUN — `models/`\n")
    A(f"*policy: **{policy_name}***\n")
    A(f"*Generated {c['generated_at']} · `{c['models_dir']}`*\n")
    A("> **NOTHING WAS DELETED IN THIS PASS.** This is a census; the plan below is what "
      "the retention policy *would* do, and it was produced with `--apply` absent.\n")

    A("## Headline\n")
    A("| | |")
    A("|---|---|")
    A(f"| runs in the archive | **{t['n_runs']}** |")
    A(f"| total size | **{t['archive_gb']} GB** |")
    A(f"| … of which physically under `models/` | {t['in_models_gb']} GB |")
    A(f"| … in {t['n_symlinked_runs']} SYMLINKED run dirs (elsewhere on disk) "
      f"| {t['symlinked_gb']} GB |")
    A(f"| the policy would free | **{t['freed_gb']} GB** ({t['freed_pct']}%) |")
    A(f"| entries in the plan | {t['n_files_planned']} |")
    A(f"| runs with a non-empty plan | {t['n_with_plan']} |")
    A(f"| LIVE / REFERENCED / CLOSED | {t['n_live']} / {t['n_referenced']} / {t['n_closed']} |")
    A(f"| runs vetoed by a named file | {t['n_excluded_by_named_file']} |")
    A(f"| **CLOSED runs needing review** | **{t['n_needs_review']}** |")
    if "n_tier4_refused" in t:
        A(f"| tier-4 runs REFUSED (no resolvable final model) | {t['n_tier4_refused']} |")
    A("")

    if policy_name == "tiered":
        A(tiers.render_tiered(c))

    A("## The policy\n")
    if policy_name == "tiered":
        A("Applied per TIER (above). Tiers 1-3 stay inside "
          f"`{'`, `'.join(p['touchable_subdirs'])}` plus `snapshots/`; tier 4 works "
          "from a KEEP-LIST instead and is guarded by `assert_safe_tiered`. The rules "
          "below describe the tier-1/2 body of the policy, which is the standing one "
          f"verbatim (tier 3 differs only in taking no every-{p['checkpoint_every']}th "
          "stride).\n")
    else:
        A(f"Applied to **CLOSED runs only**, and only inside "
          f"`{'`, `'.join(p['touchable_subdirs'])}`.\n")
    A(f"- **`checkpoints/`** — keep the FIRST, the LAST, every {p['checkpoint_every']}th, "
      "whatever `latest.txt` pins, and any checkpoint another run's `lineage` block "
      "resolved to. A `.json` sidecar is kept or dropped with its `.zip`, by STEP.")
    A(f"- **`eval_traces/`** — `main.prober.groom` at "
      f"{p['keep_trace_steps']}/{p['keep_snapshots']}. The groomer's own planner is "
      "called, not re-implemented, so the two can never drift.")
    A(f"- **Never touched**: `{'`, `'.join(p['protected_subdirs'])}`, and the run-root "
      f"files `{'`, `'.join(p['protected_files'])}`. `_assert_safe` re-checks every "
      "planned path against these before the plan is reported or executed.")
    if policy_name == "tiered":
        A(f"- A run is **tier 0** if a launcher process names it, its training output "
          f"was written within {p['recent_days']} days, its run dir is a symlink, or it "
          "is a (transitive) model-graph ancestor of any of those. It is **tier 1** if "
          f"the ledger's last {p['ledger_tail_lines']} lines name it, a committed "
          "**script** names it, a committed **measurement artifact** names it, or "
          "another run's model graph names it. The v8-era blanket is RETIRED — the "
          "model graph replaces it, and reads `original_command` as well as `lineage`.")
    else:
        A(f"- A run is REFERENCED — and therefore untouched — if a launcher process names it, "
          f"the ledger's last {p['ledger_tail_lines']} lines name it, a committed **script** "
          f"names it, it is v8-era, it was touched within {p['recent_days']} days, or it is "
          "a (transitive) fork parent of a LIVE run.")
    A("- Prose that merely *mentions* a run does **not** protect it — the historical "
      "record names nearly every run forever, so a `.md` mention as a live reference "
      "would close nothing. A committed script does protect it (a script names a run "
      "dir in order to load it), and prose still **vetoes** when it names an exact "
      "path the plan would delete.")
    if policy_name == "tiered":
        A("- `snapshots/` (the self-play pool) HAS a rule here — see *The snapshots "
          "rule* above. It is the second-largest consumer in the archive and the "
          "standing policy leaves it entirely alone.\n")
    else:
        A("- `snapshots/` (the self-play pool) is **out of scope** even though it is the "
          "second-largest consumer — the standing policy says nothing about it, so this "
          "pass measures it and proposes nothing.\n")

    A("## Top 20 runs by GB freed\n")
    A("| # | run | generation | GB freed | ckpts deleted | trace steps deleted |")
    A("|---|---|---|---:|---:|---:|")
    if c["top20_by_gb_freed"]:
        for i, r in enumerate(c["top20_by_gb_freed"], 1):
            A(f"| {i} | `{r['run']}` | {r['generation']} | {r['gb_freed']} | "
              f"{r['checkpoints_deleted']} | {r['trace_steps_deleted']} |")
    else:
        A("| — | *(the policy proposes no deletions)* | | | | |")
    A("")

    A("## Runs vetoed because a committed file or the ledger names a file in the plan\n")
    A("These are excluded from the deletion set automatically.\n")
    if c["excluded_by_named_file"]:
        A("| run | GB it would have freed | example named path | named by |")
        A("|---|---:|---|---|")
        for e in c["excluded_by_named_file"]:
            ex = e["collisions"][0] if e["collisions"] else ""
            A(f"| `{e['run']}` | {e['would_have_freed_gb']} | `{ex}` | "
              f"{', '.join(e['named_by'][:2])} |")
    else:
        A("*(none — every run whose plan collided with a named path was already "
          "REFERENCED on a stronger signal)*")
    A("")

    A("## ⚠️ SYMLINKED run dirs — the data is NOT under `models/`\n")
    A("These entries are symlinks into launcher worktrees, so `du -sh models/` does "
      "not see them and a deletion \"in `models/`\" would physically land under "
      "`.claude/worktrees/`. They are held out of the plan by default; "
      "`--follow-symlinked-runs` opts in after you have confirmed the targets are "
      "still the ones you mean.\n")
    if c.get("symlinked_runs"):
        A("| run | generation | GB | status | data actually lives at |")
        A("|---|---|---:|---|---|")
        for e in c["symlinked_runs"]:
            A(f"| `{e['run']}` | {e['generation']} | {e['gb']} | {e['status']} | "
              f"`{e['realpath']}` |")
    else:
        A("*(none)*")
    A("")

    A("## ⚠️ REVIEW BEFORE APPLYING — CLOSED runs the ledger names outside its tail\n")
    A("The tail window is what protects a run; this section makes its EDGE visible "
      "rather than silent. Each of these has a non-empty plan **and** is named "
      "somewhere higher up `ledger.md`, so a banked result may still rest on it. "
      "They are still in the deletion set — read them before running `--apply`, and "
      "widen `--ledger-tail-lines` (or delete the run's entry from the plan) if any "
      "should be kept.\n")
    if c.get("needs_review"):
        A("| run | generation | GB freed | named at | prose mentions |")
        A("|---|---|---:|---|---:|")
        for e in c["needs_review"]:
            A(f"| `{e['run']}` | {e['generation']} | {e['gb_freed']} | "
              f"`ledger.md:{e['ledger_line']}` | {e['n_named_by_committed_prose']} |")
    else:
        A("*(none)*")
    A("")

    A("## Per-run census\n")
    A("Sizes in GB. `plan GB` is 0 for every run that is not CLOSED.\n")
    A("| run | gen | cfg | " + ("tier | " if policy_name == "tiered" else "")
      + "status | ckpts | best | snaps | traces | tb | other | total | plan GB |")
    A("|---|---|---:|" + ("---:|" if policy_name == "tiered" else "")
      + "---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in sorted(c["runs"], key=lambda k: -c["runs"][k]["sizes"]["total"]):
        r = c["runs"][n]
        s = r["sizes"]
        cfg = r["config_version"] if r["config_version"] is not None else "—"
        tcol = f"{r.get('tier', '')} | " if policy_name == "tiered" else ""
        A(f"| `{n}` | {r['generation']} | {cfg} | {tcol}{r['status']} | "
          f"{_gb(s['checkpoints'])} | {_gb(s['best_model'])} | {_gb(s['snapshots'])} | "
          f"{_gb(s['eval_traces'])} | {_gb(s['tb'])} | {_gb(s['other'])} | "
          f"{_gb(s['total'])} | {_gb(r['plan']['bytes_freed'])} |")
    A("")

    A("## Why each non-CLOSED run is protected\n")
    for n in sorted(c["runs"]):
        r = c["runs"][n]
        if r["status"] == "CLOSED":
            continue
        why = "; ".join(r["status_reasons"]) or "(no reason recorded)"
        A(f"- **`{n}`** — {r['status']}: {why}")
    A("")

    A("## What would be KEPT, and why (every CLOSED run with a plan)\n")
    any_plan = False
    for n in sorted(c["runs"]):
        r = c["runs"][n]
        if not r["plan"]["delete"]:
            continue
        any_plan = True
        A(f"<details><summary><code>{n}</code> — {_gb(r['plan']['bytes_freed'])} GB freed, "
          f"{len(r['plan']['delete'])} entries deleted</summary>\n")
        A("**KEEP**\n")
        tier4 = r.get("tier") == 4
        for f, reason in sorted(r["plan"]["keep"].items()):
            A(f"- `{f if tier4 else 'checkpoints/' + f}` — {reason}")
        if tier4:
            A("- everything else in the run dir is DELETED — the keep-list above IS "
              "the policy\n")
        else:
            A("- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, "
              "`metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` "
              "— never candidates")
            A(f"- the {KEEP_TRACE_STEPS} most-recent `eval_traces/step_*` "
              f"(+ `snapshot.zip` on the newest {KEEP_SNAPSHOTS}) — `prober.groom` "
              "retention\n")
        A("**DELETE**\n")
        for d in r["plan"]["delete"]:
            A(f"- `{d}`")
        A("\n</details>\n")
    if not any_plan:
        A("*(no CLOSED run has a non-empty plan)*\n")

    A("## To actually apply this\n")
    A("```bash")
    A(apply_cmd(policy_name))
    A("```\n")
    if policy_name == "tiered":
        A("Run it from the **main checkout** — `models/` exists only there — and read "
          "*REVIEW BEFORE APPLYING* first. `--policy standing` (the default) is the "
          "gentler pass and is still available unchanged.\n")
    A("**Nothing was deleted in this pass — this was a dry run, and it wrote only the "
      "two report files.**\n")
    return "\n".join(L)


# ------------------------------------------------------------------ apply ----


def apply_plan(census: dict) -> dict:
    """Execute the plan.  Only reachable via an explicit ``--apply``."""
    removed: list[str] = []
    errors: list[str] = []
    freed = 0
    for n, r in census["runs"].items():
        run_dir = r["run_dir"]
        rels = r["plan"]["delete"]
        if not rels:
            continue
        # RE-CHECK at execution time, under the rules of the tier that produced the
        # plan — never under a weaker set.  Tier 4 reaches outside the two touchable
        # subdirs by design, so it is guarded positively by its own keep-list; every
        # other plan is still held to `_assert_safe`.
        keep_rel = r["plan"].get("keep_rel") or []
        if r.get("tier") == 4:
            tiers.assert_safe_tiered(run_dir, rels, keep_rel)
        else:
            pool = [p for p in rels if p == tiers.SNAPSHOT_DIR]
            rest = [p for p in rels if p != tiers.SNAPSHOT_DIR]
            if pool:
                tiers.assert_safe_tiered(run_dir, pool, keep_rel)
            _assert_safe(run_dir, [os.path.join(run_dir, p) for p in rest])
        for rel in rels:
            full = os.path.join(run_dir, rel)
            try:
                if os.path.islink(full):
                    errors.append(f"{os.path.join(n, rel)}: is a symlink — refused")
                    continue
                if os.path.isdir(full):
                    freed += dir_size(full)
                    shutil.rmtree(full)
                elif os.path.exists(full):
                    freed += os.path.getsize(full)
                    os.remove(full)
                else:
                    continue
                removed.append(os.path.join(n, rel))
            except OSError as exc:
                errors.append(f"{os.path.join(n, rel)}: {exc}")
    census["applied"] = True
    census["apply_result"] = {"n_removed": len(removed), "bytes_freed": freed,
                              "gb_freed": _gb(freed), "errors": errors}
    return census


# ------------------------------------------------------------------- main ----


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(
        prog="archive_grooming_dryrun.py",
        description="Census (and, with --apply, execution) of the models/ retention policy.",
    )
    p.add_argument("--models-dir", default=None,
                   help="the run archive (default: $GEN3AI_MODELS_DIR, else <repo>/models)")
    p.add_argument("--repo-root", default=_REPO, help="checkout to scan for references")
    p.add_argument("--out-prefix", default=None,
                   help="write <prefix>.md and <prefix>.json (default: named after "
                        "the POLICY, so a tiered pass can never clobber the standing "
                        "report or the other way round)")
    p.add_argument("--policy", choices=("standing", "tiered"), default="standing",
                   help="standing (the DEFAULT, unchanged): one rule for every CLOSED "
                        "run. tiered: grade by era + who still reaches for the run, "
                        "and add a snapshots rule (see archive_grooming_tiers.py)")
    p.add_argument("--recent-days", type=int, default=RECENT_DAYS,
                   help="a run touched this recently counts as REFERENCED (0 disables)")
    p.add_argument("--ledger-tail-lines", type=int, default=None,
                   help="default: 1000 under --policy standing, "
                        f"{tiers.TIERED_LEDGER_TAIL_LINES} under --policy tiered")
    p.add_argument("--follow-symlinked-runs", action="store_true",
                   help="include run dirs that are SYMLINKS into launcher worktrees "
                        "(their data is not under models/; held out by default)")
    p.add_argument("--no-write", action="store_true", help="print the summary only")
    p.add_argument("--apply", action="store_true",
                   help="ACTUALLY DELETE the planned entries (default: dry run)")
    args = p.parse_args(argv)

    models_dir = (args.models_dir or os.environ.get("GEN3AI_MODELS_DIR")
                  or os.path.join(args.repo_root, "models"))
    if not os.path.isdir(models_dir):
        print(f"no run archive at {models_dir} — nothing to census", file=sys.stderr)
        return 2

    out_prefix = args.out_prefix
    if out_prefix is None:
        out_prefix = os.path.join(
            _HERE, "archive_grooming_dryrun_2026-09-06" if args.policy == "standing"
            else "archive_grooming_tiered_2026-09-06")

    tail = args.ledger_tail_lines
    if tail is None:
        tail = tiers.TIERED_LEDGER_TAIL_LINES if args.policy == "tiered" else 1000
    if args.policy == "tiered":
        census = build_census_tiered(models_dir, args.repo_root, args.recent_days,
                                     tail, args.follow_symlinked_runs)
    else:
        census = build_census(models_dir, args.repo_root, args.recent_days,
                              tail, args.follow_symlinked_runs)
    if args.apply:
        census = apply_plan(census)

    if not args.no_write:
        with open(out_prefix + ".json", "w") as fh:
            json.dump(census, fh, indent=1, default=str)
        with open(out_prefix + ".md", "w") as fh:
            fh.write(render_markdown(census))

    t = census["totals"]
    print(f"policy                  {census.get('policy_name', 'standing')}")
    print(f"runs                    {t['n_runs']}")
    print(f"archive                 {t['archive_gb']} GB")
    print(f"policy would free       {t['freed_gb']} GB ({t['freed_pct']}%)")
    print(f"LIVE/REFERENCED/CLOSED  {t['n_live']}/{t['n_referenced']}/{t['n_closed']}")
    print(f"vetoed by a named file  {t['n_excluded_by_named_file']}")
    if "tiers" in census:
        print()
        print("per tier:   tier  who              runs    GB now   GB freed")
        for row in census["tiers"]["table"]:
            print(f"            {row['tier']:>4}  {row['name']:<15} {row['n_runs']:>4} "
                  f"{row['gb_now']:>9} {row['gb_freed']:>10}")
        s = census["tiers"]["snapshots"]
        print(f"snapshots:  {s['kept']} pools KEPT ({s['gb_kept']} GB) · "
              f"{s['deleted']} FREED ({s['gb_freed']} GB) · "
              f"proposed thinning of the kept ones would free {s['gb_thin_proposal']} GB")
        if census["tiers"]["refusals"]:
            print(f"tier-4 REFUSED (no resolvable final model): "
                  f"{len(census['tiers']['refusals'])}")
            for e in census["tiers"]["refusals"][:10]:
                print(f"    {e['run']}: {e['reason'][:120]}")
        print()
    print(f"NEEDS REVIEW            {t['n_needs_review']} runs with a plan the ledger "
          f"names outside its tail")
    for e in census["needs_review"]:
        print(f"    {e['run']:<44} {e['gb_freed']:>7} GB  ledger.md:{e['ledger_line']}")
    print()
    print("top 10 by GB freed:")
    for i, r in enumerate(census["top20_by_gb_freed"][:10], 1):
        print(f"  {i:2d}. {r['run']:<50} {r['gb_freed']:>8} GB")
    print()
    if census["applied"]:
        ar = census["apply_result"]
        print(f"APPLIED: {ar['n_removed']} entries removed, {ar['gb_freed']} GB freed")
        for e in ar["errors"][:20]:
            print(f"  ERROR {e}")
    else:
        print("NOTHING WAS DELETED IN THIS PASS — dry run.")
        print("To apply:\n" + apply_cmd(census.get("policy_name", "standing")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
