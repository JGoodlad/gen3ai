"""`python -m main.tb_curate` — THE CURATED TENSORBOARD LOGDIR.

THE PROBLEM. `tensorboard --logdir models/` shows all **217** runs that carry a `tb/`. Most of them
are two-hour ablation arms, dose cells and exploiter fleets; the handful of long runs the owner
actually reads are lost among them, every chart's legend is unusable, and the origin process holds
every one of them in memory.

THE FIX, and why it is a symlink farm rather than a filter. TensorBoard has no "show only these
runs" server-side option — the logdir IS the selection. So this maintains a second directory of
symlinks:

    tb_curated/<run>  ->  models/<run>/tb

and TensorBoard is pointed at `tb_curated/`. Three properties come free: the run appears under its
own NAME (not `<run>/tb`, which is what `--logdir models/` shows); the data is never copied, so
`models/` stays the single source of truth and this directory can be deleted at any time; and a run
that is grooming-thinned or archived simply becomes a dangling link rather than a stale copy.

WHAT GETS IN. Three sources, unioned:

1. **`designs/tb_curated_runs.json`** — the committed list, with a one-line `why` per entry. Edit it
    by hand; that file is the owner's dial.
2. **Every LIVE run** — detected by scanning `ps` for a launcher (or a bare `train_rl_agent.py`) and
    reading the run name out of its argv. This is the half that matters operationally: the arm
    running right now always shows up, without anyone remembering to add it, and drops back out of
    the union when it finishes (though a run worth keeping should then be added to the list).
3. **The BASELINE REGISTRY's `tb_curated` list** (`designs/baselines.json`,
    `gen3_baselines_registry_v1`) — the NAMED reference curves. A baseline whose curve is not on the
    board is a baseline nobody can look at, so the registry's declared set is unioned in rather than
    hand-copied here. `--no-baselines` opts out; a broken registry degrades to no union rather than
    taking TensorBoard curation down with it (`python -m main.baselines check` is what reports it).

SAFETY. This tool creates and removes **symlinks inside the curated directory and nothing else**. It
refuses to delete anything there that is not a symlink, and it never opens, moves or writes a single
byte under `models/`.

    python -m main.tb_curate                 # print the proposal + a diff (same as --check)
    python -m main.tb_curate --check         # what the dir WOULD hold vs what it does
    python -m main.tb_curate --apply         # rebuild the symlink dir idempotently
    python -m main.tb_curate --propose       # survey the archive's long runs, for amending the list
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

#: The committed list. Relative to the repo root.
LIST_RELPATH = os.path.join("designs", "tb_curated_runs.json")

#: The curated logdir. Deliberately OUTSIDE `models/` (which is not committed and is read-mostly)
#: and gitignored — it is a derived view, rebuildable from the list in one command.
CURATED_DIRNAME = "tb_curated"

#: How a live run is recognised in `ps` output.
_LAUNCHER_MARKERS = ("main.launcher", "train_rl_agent.py")
_RUN_NAME_RE = re.compile(r"--run[-_]name(?:=|\s+)([^\s]+)")
_RUN_DIR_RE = re.compile(r"--run[-_]dir(?:=|\s+)([^\s]+)")


# --------------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------------
def repo_root() -> str:
    from utils.paths import repo_root as _rr
    return str(_rr())


def curated_dir(explicit: Optional[str] = None) -> str:
    return os.path.abspath(explicit) if explicit else os.path.join(repo_root(), CURATED_DIRNAME)


def list_path(explicit: Optional[str] = None) -> str:
    return os.path.abspath(explicit) if explicit else os.path.join(repo_root(), LIST_RELPATH)


def models_dir(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return os.path.abspath(explicit)
    from utils.paths import main_models_dir
    d = main_models_dir()
    return str(d) if d else None


# --------------------------------------------------------------------------------------------
# the committed list
# --------------------------------------------------------------------------------------------
def load_list(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """The committed entries, or `[]` when the file is absent.

    Absent is a legitimate state (a fresh clone that has not curated anything), and it must not be
    an error: the union with the LIVE runs still produces a usable logdir.
    """
    p = list_path(path)
    try:
        with open(p, encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001 — a malformed dial is worth a loud message
        raise SystemExit(f"{p}: could not be read as JSON — {exc}")
    runs = obj.get("runs") if isinstance(obj, dict) else obj
    if not isinstance(runs, list):
        raise SystemExit(f"{p}: expected a top-level 'runs' list")
    out = []
    for r in runs:
        if isinstance(r, str):
            out.append({"run": r, "why": ""})
        elif isinstance(r, dict) and r.get("run"):
            out.append({"run": str(r["run"]), "why": str(r.get("why", ""))})
    return out


# --------------------------------------------------------------------------------------------
# live runs
# --------------------------------------------------------------------------------------------
def live_runs() -> List[Tuple[str, int]]:
    """`(run_name, pid)` for every training process currently alive, newest-PID last.

    Reads `ps -eo pid=,args=` and matches on the MODULE/SCRIPT the process is running, then takes
    the run name from `--run-name` or the basename of `--run-dir`. A launcher and its child both
    name the same run, so the result is de-duplicated by name.

    This deliberately does not match on the word "gen3ai" or a path: an agent's own `ps` scan or a
    grep would otherwise match itself, which is the failure mode `pkill -f` has already caused in
    this project.
    """
    try:
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:  # noqa: BLE001 — no ps, a container, a locked-down box
        return []
    found: Dict[str, int] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str, _, args = line.partition(" ")
        if not any(m in args for m in _LAUNCHER_MARKERS):
            continue
        m = _RUN_NAME_RE.search(args)
        name = m.group(1) if m else None
        if not name:
            m = _RUN_DIR_RE.search(args)
            if m:
                name = os.path.basename(os.path.normpath(m.group(1)))
        if not name:
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        found.setdefault(name, pid)
    return sorted(found.items(), key=lambda kv: kv[1])


# --------------------------------------------------------------------------------------------
# the desired state
# --------------------------------------------------------------------------------------------
def baseline_runs() -> List[Tuple[str, str]]:
    """`(run_name, why)` for every member of the registry's `tb_curated` baseline LIST.

    `designs/baselines.json` is where the NAMED baselines live (`gen3_baselines_registry_v1`), and
    its `tb_curated` list is the declared set of reference curves. Unioning it in here means the
    two files cannot drift apart in the one direction that matters: a baseline whose curve is not
    on the board is a baseline nobody can look at. `tb_curated_runs.json` stays the dial for
    everything else — it curates far more than the registry names, with a `why` per entry.

    Degrades to `[]` on any registry problem: TensorBoard curation must never be what a broken
    registry takes down, and `python -m main.baselines check` is what reports that.
    """
    try:
        from agents.training import baselines as reg
        li = reg.get_list("tb_curated")
        return [(reg.get(m).run, f"baseline `{m}` (registry tb_curated list)") for m in li.members]
    except Exception:  # noqa: BLE001 — see the docstring: this is a union, not a gate
        return []


def desired(models: str, *, list_file: Optional[str] = None,
            include_live: bool = True,
            include_baselines: bool = True) -> Tuple[List[Dict[str, Any]], List[str]]:
    """`(entries, problems)` — what the curated dir should hold, and what could not be honoured.

    An entry is `{run, why, source, target, exists}`. A listed run that has no `tb/` is reported as
    a problem and NOT linked: a dangling link in the logdir makes TensorBoard log an error on every
    reload, and a silently-missing run is worse than a named one.

    Three sources union: the committed list, the LIVE runs, and the baseline registry's
    `tb_curated` list. A run named by more than one carries them all in `source`.
    """
    entries: Dict[str, Dict[str, Any]] = {}
    problems: List[str] = []

    def add(run: str, why: str, source: str, *, report_missing: bool = True) -> None:
        tb = os.path.join(models, run, "tb")
        if run in entries:
            entries[run]["source"] = entries[run]["source"] + "+" + source
            return
        if not os.path.isdir(tb):
            if report_missing:
                problems.append(f"{run}: no {os.path.join('models', run, 'tb')} ({source})")
            return
        entries[run] = {"run": run, "why": why, "source": source, "target": tb}

    for item in load_list(list_file):
        add(item["run"], item["why"], "list")
    if include_baselines:
        # A baseline whose run is not in THIS archive is SILENT here, unlike a listed one. The
        # committed list is a per-archive dial and a missing entry there is the user's own stale
        # edit; the baseline registry is a global declaration, so an absent run means "this box
        # does not have that run" — a fact `python -m main.baselines check` reports properly and
        # this tool could not act on. Reporting it here would duplicate a finding into a tool with
        # no remedy, and would fire on every foreign or synthetic models/ tree.
        for name, why in baseline_runs():
            add(name, why, "baseline", report_missing=False)
    if include_live:
        for name, pid in live_runs():
            add(name, f"LIVE (pid {pid})", "live")

    return [entries[k] for k in sorted(entries)], problems


def current(cdir: str) -> Dict[str, str]:
    """`{name: link target}` for every symlink currently in the curated dir."""
    out: Dict[str, str] = {}
    if not os.path.isdir(cdir):
        return out
    for name in sorted(os.listdir(cdir)):
        p = os.path.join(cdir, name)
        if os.path.islink(p):
            out[name] = os.readlink(p)
    return out


# --------------------------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------------------------
def apply(cdir: str, entries: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    """Make the curated dir hold exactly `entries`. Returns `(added, removed, kept)`.

    ONLY symlinks are created and removed, and only inside `cdir`. A non-symlink found there is left
    alone and reported — this tool has no business deleting a real file someone put in its way.
    """
    os.makedirs(cdir, exist_ok=True)
    want = {e["run"]: e["target"] for e in entries}
    have = current(cdir)
    added, removed, kept = [], [], []

    for name, target in want.items():
        p = os.path.join(cdir, name)
        if name in have:
            if os.path.realpath(have[name]) == os.path.realpath(target):
                kept.append(name)
                continue
            os.unlink(p)          # retarget: the run moved (e.g. promoted to _goldens/)
        elif os.path.exists(p):
            continue              # a real file/dir squatting the name — reported by `check`
        os.symlink(target, p)
        added.append(name)

    for name in have:
        if name not in want:
            os.unlink(os.path.join(cdir, name))
            removed.append(name)

    return added, removed, kept


def squatters(cdir: str) -> List[str]:
    """Names in the curated dir that are NOT symlinks — this tool will never touch them."""
    if not os.path.isdir(cdir):
        return []
    return sorted(n for n in os.listdir(cdir)
                  if not os.path.islink(os.path.join(cdir, n)))


# --------------------------------------------------------------------------------------------
# the proposal survey
# --------------------------------------------------------------------------------------------
def _max_step(run_dir: str) -> int:
    best = 0
    for z in glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint_*_steps.zip")):
        m = re.search(r"checkpoint_(\d+)_steps", os.path.basename(z))
        if m:
            best = max(best, int(m.group(1)))
    try:
        from agents.training.lineage import read_num_timesteps
        best = max(best, read_num_timesteps(run_dir) or 0)
    except Exception:  # noqa: BLE001
        pass
    return best


def propose(models: str, *, top: int = 25) -> List[Dict[str, Any]]:
    """The archive's longest runs, with their lineage role — the pool to amend the list FROM.

    Ranked by max recorded step, which for a fork is its ABSOLUTE step (a fork continues the
    parent's counter), so a long chain's later links naturally rank above the short ablation arms
    that fork off them.
    """
    from agents.training import lineage as L
    rows = []
    for name in sorted(os.listdir(models)):
        rd = os.path.join(models, name)
        if not os.path.isdir(os.path.join(rd, "tb")):
            continue
        block = None
        try:
            block = L.read_block(rd)
        except Exception:  # noqa: BLE001
            pass
        rows.append({
            "run": name,
            "max_step": _max_step(rd),
            "role": (block or {}).get("role"),
            "fork_step": (block or {}).get("fork_step"),
            "parent": (((block or {}).get("fork_parent") or {}) or {}).get("run_name"),
            "events": len(glob.glob(os.path.join(rd, "tb", "**", "events.out.tfevents.*"),
                                    recursive=True)),
        })
    rows.sort(key=lambda r: r["max_step"], reverse=True)
    return rows[:top]


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _print_check(cdir: str, entries: List[Dict[str, Any]], problems: List[str]) -> None:
    have = current(cdir)
    want = {e["run"] for e in entries}
    print(f"curated logdir : {cdir}")
    print(f"proposed       : {len(entries)} run(s)\n")
    for e in entries:
        state = "ok" if e["run"] in have else "ADD"
        why = f"  — {e['why']}" if e["why"] else ""
        print(f"  [{state:>3}] {e['run']:<46} ({e['source']}){why}")
    stale = [n for n in have if n not in want]
    for n in stale:
        print(f"  [DEL] {n:<46} (no longer listed and not live)")
    for s in squatters(cdir):
        print(f"  [!!!] {s:<46} (not a symlink — left alone)")
    for p in problems:
        print(f"  [ ?? ] {p}")
    if not stale and all(e["run"] in have for e in entries):
        print("\nUp to date.")
    else:
        print(f"\n{sum(1 for e in entries if e['run'] not in have)} to add, {len(stale)} to remove."
              "  Run with --apply.")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m main.tb_curate",
        description="Maintain the curated TensorBoard logdir (a symlink view over models/*/tb).")
    p.add_argument("--check", action="store_true", help="Show the proposal and the diff (default)")
    p.add_argument("--apply", action="store_true", help="Rebuild the symlink dir idempotently")
    p.add_argument("--propose", action="store_true",
                   help="Survey the archive's longest runs, to amend the committed list from")
    p.add_argument("--list-file", default=None, help=f"Override {LIST_RELPATH}")
    p.add_argument("--curated-dir", default=None, help=f"Override <repo>/{CURATED_DIRNAME}")
    p.add_argument("--models-dir", default=None, help="Override the run archive")
    p.add_argument("--no-live", action="store_true", help="List only; do not union the live runs")
    p.add_argument("--no-baselines", action="store_true",
                   help="Do not union the baseline registry's `tb_curated` list "
                        "(designs/baselines.json)")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    args = p.parse_args(argv)

    models = models_dir(args.models_dir)
    if not models:
        print("No models/ archive found. Pass --models-dir (or set $GEN3AI_MODELS_DIR).",
              file=sys.stderr)
        return 2
    cdir = curated_dir(args.curated_dir)

    if args.propose:
        rows = propose(models)
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        print(f"Longest runs under {models} (rank by max recorded step):\n")
        for r in rows:
            print(f"  {r['max_step']:>13,}  {r['run']:<46} role={str(r['role']):<8} "
                  f"ev={r['events']:<3} parent={r['parent']}")
        print(f"\nEdit {LIST_RELPATH} to change what TensorBoard shows.")
        return 0

    entries, problems = desired(models, list_file=args.list_file, include_live=not args.no_live,
                                include_baselines=not args.no_baselines)

    if args.json:
        print(json.dumps({"curated_dir": cdir, "entries": entries, "problems": problems,
                          "current": current(cdir)}, indent=2))
        return 0

    if not args.apply:
        _print_check(cdir, entries, problems)
        return 0

    added, removed, kept = apply(cdir, entries)
    print(f"curated logdir : {cdir}")
    for n in added:
        print(f"  + {n}")
    for n in removed:
        print(f"  - {n}")
    print(f"\n{len(added)} added, {len(removed)} removed, {len(kept)} unchanged "
          f"({len(entries)} total).")
    for s in squatters(cdir):
        print(f"  [!!!] {s} is not a symlink — left alone")
    for pr in problems:
        print(f"  [ ?? ] {pr}")
    print("\nTensorBoard serves this dir via ~/.config/systemd/user/tensorboard.service "
          "(`systemctl --user restart tensorboard` after changing the LIST is not needed — "
          "TensorBoard rescans the logdir; a restart is only needed if the dir itself moved).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
