"""FORK / MANUAL-RESUME RECIPE AUDIT (2026-10-14) — read-only, re-runnable.

THE BUG AUDITED (GIGO, P1). A `--model` launch that was a FORK (checkpoint outside the new run dir),
or a MANUAL same-run resume of a run whose `original_command` did not carry `--arch production`
(or its alias `--arch endstate`), did not inherit the parent's / run's recorded TRAINING RECIPE:
every recipe knob the argv did not type fell to the PARSER DEFAULT. The launcher's own
auto-restarts were safe (they re-pass the original typed argv).

WHAT THIS SCRIPT READS (never writes anything under `models/`):

1. FORK vs PARENT. For a run whose lineage role is `fork` (or any role with a recorded fork parent —
   via `agents.training.lineage.role_of` / `fork_parent`, never a path regex), every recipe row
   (`recipe_surface.ROWS + FORK_ROWS`, minus `INERT_ON_RESUME`, minus `fork_lr` / `fork_lr_freeze`)
   is compared between the run's `metadata.json:cli_args` and the PARENT's. A difference is TYPED if
   its flag appears in the run's `original_command` or `launcher_command`, else UNTYPED; an UNTYPED
   difference whose run value equals the parser default is the BUG SHAPE. `trunk_precision` None is
   the untyped tri-state (it resolves by inheritance), reported as TRISTATE, never BUG SHAPE.
   As an independent cross-check the TRAINER's own recorded values (the checkpoint sidecar JSON) are
   compared between the parent's resolved file and the fork's FIRST checkpoint.
2. MID-RUN CHANGES. For every run, the per-checkpoint sidecars (`checkpoints/checkpoint_*_steps.json`
   + `final_model.json`), ordered by `num_timesteps`, are walked for changes in the trainer knobs
   (`SIDECAR_KEYS`), plus `metadata.json:pin_history` length, `launcher_command != original_command`,
   whether the LAST launch was a manual same-run `--model` resume and whether it recorded
   `_recipe_restart_inherited`, and `matchup_history` changes (self-play / opponent mix per launch).
3. RESEARCH DEPENDENCE. For every flagged run, `designs/research_state/ledger.md` entries and
   `UNDERSTANDING.md` sections mentioning the run name or its seed-group prefix are listed.

CAVEAT: `metadata.json` is overwritten on every save, so `cli_args` / `launcher_command` describe the
run's LAST launch; earlier launches are visible only through the sidecars and `pin_history`.

Usage (rule 5b while a run is live):
    nice -n 19 taskset -c 5-7,13-15 python audit.py [--models-dir DIR] [--json OUT] [--md OUT]
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))

from agents.training import lineage  # noqa: E402
from main.train import recipe_surface as rs  # noqa: E402
from main.train.fork_lr import is_same_run_checkpoint  # noqa: E402
from utils.paths import repo_path  # noqa: E402

SKIP = set(rs.INERT_ON_RESUME) | {"fork_lr", "fork_lr_freeze"}
AUDIT_ROWS = [r for r in rs.ROWS + rs.FORK_ROWS if r.dest not in SKIP]
#: Trainer values recorded per checkpoint (dotted = nested).
SIDECAR_KEYS = ("ent_coef", "grad_accum_steps", "n_epochs", "clip_range", "clip_range_vf",
                "weight_decay", "batch_size", "n_steps", "vf_coef", "gae_lambda",
                "dose.rollout_rows", "dose.effective_batch", "dose.kl_controller.min_lr",
                "dose.kl_controller.max_lr", "env_core.n_envs", "env_core.target_samples",
                "env_core.trainee_slots", "dose.fork_lr", "dose.lr_frozen", "matchup_hash")
#: Rows a difference in which cannot move training, with the reason (verified at the code).
INERT_NOTES = {"clip_range_vf": "INERT under the win-prob critic: instrumented_ppo/train_setup.py takes "
               "value_mode='plain' when critic_winprob (the only critic); it is only logged"}
#: Seed-group prefix -> other names the research docs use for the group (name-grep misses them).
ALIASES = {"rb_hib_": ["knock-in fork", "knock-in forks"]}
ARCH_FLAGS_RECIPE = ("production", "endstate")


def _json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _get(d: Any, dotted: str) -> Any:
    for k in dotted.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def parser_defaults() -> Dict[str, Any]:
    from main.train.parser import build_parser
    with contextlib.redirect_stdout(io.StringIO()):
        p = build_parser()
    return {r.dest: p.get_default(r.dest) for r in rs.ALL_ROWS}


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
            and not isinstance(b, bool):
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def _typed(flag: str, *cmds: Optional[str]) -> bool:
    return any(lineage.command_has_flag(c, flag) for c in cmds if c)


def sidecars(run_dir: str) -> List[Tuple[int, str, Dict[str, Any]]]:
    out = []
    paths = glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint_*_steps.json"))
    paths.append(os.path.join(run_dir, "final_model.json"))
    for p in paths:
        j = _json(p)
        if j is None or not isinstance(j.get("num_timesteps"), (int, float)):
            continue
        out.append((int(j["num_timesteps"]), os.path.relpath(p, run_dir), j))
    out.sort(key=lambda t: (t[0], t[1].startswith("final")))
    return out


def sidecar_for_zip(zip_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not zip_path:
        return None
    return _json(os.path.splitext(zip_path)[0] + ".json")


def mid_run_changes(scs: List[Tuple[int, str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    changes = []
    prev: Dict[str, Tuple[Any, int]] = {}
    for step, name, j in scs:
        for k in SIDECAR_KEYS:
            v = _get(j, k)
            if k in prev and not _eq(prev[k][0], v):
                changes.append({"knob": k, "from": prev[k][0], "to": v, "prev_step": prev[k][1],
                                "at_step": step, "file": name})
            prev[k] = (v, step)
    return changes


def matchup_changes(meta: Dict[str, Any]) -> Dict[str, Any]:
    mh = meta.get("matchup_history") or []
    if not isinstance(mh, list):
        return {"entries": None, "changes": []}
    changes = []
    for a, b in zip(mh, mh[1:]):
        if a.get("hash") != b.get("hash"):
            sa, sb = a.get("spec") or {}, b.get("spec") or {}
            diff = {k: [sa.get(k), sb.get(k)] for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)}
            changes.append({"at": b.get("recorded_at"), "from_hash": a.get("hash"),
                            "to_hash": b.get("hash"), "diff": diff})
    return {"entries": len(mh), "mix_kinds": [(e.get("spec") or {}).get("mix_kind") for e in mh],
            "changes": changes}


def fork_compare(run_dir: str, meta: Dict[str, Any], parent_dir: str,
                 defaults: Dict[str, Any]) -> List[Dict[str, Any]]:
    pmeta = _json(os.path.join(parent_dir, "metadata.json")) or {}
    cli, pcli = meta.get("cli_args") or {}, pmeta.get("cli_args") or {}
    oc, lc = meta.get("original_command"), meta.get("launcher_command")
    rows = []
    for r in AUDIT_ROWS:
        if r.dest not in cli and r.dest not in pcli:
            continue
        v, pv = cli.get(r.dest), pcli.get(r.dest)
        if _eq(v, pv):
            continue
        typed = _typed(r.flag, oc, lc)
        at_default = _eq(v, defaults.get(r.dest))
        tristate = defaults.get(r.dest) is None and v is None
        cls = ("TYPED" if typed else "TRISTATE" if tristate
               else "BUG_SHAPE" if at_default else "UNTYPED")
        rows.append({"knob": r.dest, "run": v, "parent": pv, "typed": typed,
                     "eq_parser_default": at_default, "parser_default": defaults.get(r.dest),
                     "class": cls, "inert": INERT_NOTES.get(r.dest)})
    return rows


def trainer_compare(parent_zip: Optional[str], scs: List[Tuple[int, str, Dict[str, Any]]]
                    ) -> Dict[str, Any]:
    psc = sidecar_for_zip(parent_zip)
    first = next((j for _, n, j in scs if n.startswith("checkpoints")), scs[0][2] if scs else None)
    if psc is None or first is None:
        return {"available": False, "parent_sidecar": bool(psc), "fork_sidecar": bool(first)}
    diffs = {k: [_get(first, k), _get(psc, k)] for k in SIDECAR_KEYS
             if k not in ("matchup_hash", "dose.fork_lr") and not _eq(_get(first, k), _get(psc, k))}
    return {"available": True, "diffs_fork_first_vs_parent": diffs}


def _arch_recipe(cmd: Optional[str]) -> bool:
    return (lineage.command_flag_value(cmd, "--arch") or "") in ARCH_FLAGS_RECIPE


def audit_run(run_dir: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    name = os.path.basename(run_dir)
    meta = _json(os.path.join(run_dir, "metadata.json")) or {}
    role = lineage.role_of(run_dir)
    parent = lineage.fork_parent(run_dir, warn=False)
    oc, lc = meta.get("original_command"), meta.get("launcher_command")
    cli = meta.get("cli_args") or {}
    last_model = lineage.command_flag_value(lc, "--model")
    # A MANUAL same-run resume: the last launch's --model is this run's own checkpoint and the argv is
    # not the original one (the launcher's auto-restarts re-pass the original argv).
    manual_same_run = bool(last_model) and is_same_run_checkpoint(
        lineage.resolve_model_path(last_model), run_dir) and (lc or "") != (oc or "")
    scs = sidecars(run_dir)
    rec: Dict[str, Any] = {
        "run": name, "role": role,
        "parent_run": parent.run_name if parent else None,
        "parent_file": (parent.resolved_file or parent.resolved_path) if parent else None,
        "original_has_arch_recipe": _arch_recipe(oc),
        "launcher_differs_from_original": (lc or "") != (oc or ""),
        "last_launch_manual_same_run_resume": manual_same_run,
        "recipe_restart_inherited_recorded": "_recipe_restart_inherited" in cli,
        "pin_history_len": len(meta.get("pin_history") or []),
        "n_sidecars": len(scs),
        "mid_run_changes": mid_run_changes(scs),
        "matchup": matchup_changes(meta),
        "fork_diffs": [], "trainer_vs_parent": None,
    }
    if parent and parent.run_dir and os.path.isdir(parent.run_dir):
        rec["fork_diffs"] = fork_compare(run_dir, meta, parent.run_dir, defaults)
        rec["trainer_vs_parent"] = trainer_compare(parent.resolved_file or parent.resolved_path, scs)
    elif parent:
        rec["fork_diffs_error"] = f"parent run dir not found: {parent.run_dir!r}"
    if manual_same_run:
        # Recipe rows the RELAUNCH argv did not type whose recorded value is the parser default.
        rec["relaunch_untyped_at_parser_default"] = [
            r.dest for r in AUDIT_ROWS if r.dest in cli and not _typed(r.flag, lc)
            and _eq(cli.get(r.dest), defaults.get(r.dest)) and defaults.get(r.dest) is not None]
    bug = [d for d in rec["fork_diffs"] if d["class"] == "BUG_SHAPE"]
    rec["bug_shape"] = bool(bug)
    rec["bug_shape_training_affecting"] = [d["knob"] for d in bug if not d["inert"]]
    rec["manual_resume_uninherited"] = (manual_same_run and not rec["original_has_arch_recipe"]
                                        and not rec["recipe_restart_inherited_recorded"])
    rec["flagged"] = bool(bug or rec["mid_run_changes"] or rec["matchup"]["changes"]
                          or rec["manual_resume_uninherited"]
                          or (rec["trainer_vs_parent"] or {}).get("diffs_fork_first_vs_parent")
                          or [d for d in rec["fork_diffs"] if d["class"] == "UNTYPED"])
    return rec


def group_prefix(name: str) -> str:
    """The ARM prefix (`rb_hib_ctrl_s3101` -> `rb_hib_ctrl_`)."""
    return re.sub(r"_s\d+[a-z]?$", "_", name)


def family_prefix(name: str) -> str:
    """The seed-GROUP family prefix (`rb_hib_ctrl_s3101` -> `rb_hib_`)."""
    parts = name.split("_")
    return "_".join(parts[:2]) + "_" if len(parts) > 2 else name


def research_refs(names: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    ledger = repo_path("designs", "research_state", "ledger.md")
    und = repo_path("designs", "research_state", "UNDERSTANDING.md")
    entries: List[Tuple[str, str]] = []  # (doc:heading, body)
    for path, pat, tag in ((ledger, r"^### ", "ledger"), (und, r"^#{2,3} ", "UNDERSTANDING")):
        try:
            lines = open(path).read().splitlines()
        except OSError:
            continue
        head, body = None, []
        for ln in lines + ["### END"]:
            if re.match(pat, ln):
                if head:
                    entries.append((f"{tag}: {head[:200]}", "\n".join(body)))
                head, body = ln.lstrip("# ").strip(), []
            else:
                body.append(ln)
    for n in names:
        keys = (n, group_prefix(n), family_prefix(n), *ALIASES.get(family_prefix(n), []))
        out[n] = [h for h, b in entries if any(k in b or k in h for k in keys)]
    return out


def _fmt(v: Any) -> str:
    return json.dumps(v) if not isinstance(v, str) else v


def table(recs: List[Dict[str, Any]], refs: Dict[str, List[str]]) -> str:
    n = len(recs)
    forks = sum(1 for r in recs if r["parent_run"])
    flagged = [r for r in recs if r["flagged"]]
    lines = [f"# Fork / manual-resume recipe audit — {n} runs scanned, {forks} forks, "
             f"{len(flagged)} with a finding, {n - len(flagged)} clean",
             "", f"Bug-shape forks: {sum(r['bug_shape'] for r in recs)} (with a TRAINING-AFFECTING "
             f"bug-shape knob: {sum(bool(r['bug_shape_training_affecting']) for r in recs)}); "
             f"manual same-run resumes: {sum(r['last_launch_manual_same_run_resume'] for r in recs)}, "
             f"of which without recorded inheritance "
             f"{sum(r['manual_resume_uninherited'] for r in recs)}; runs with a "
             f"mid-run trainer-knob change: {sum(bool(r['mid_run_changes']) for r in recs)}.", "",
             "| run | role | parent | fork vs parent (cli_args) | trainer: fork 1st ckpt vs parent | "
             "mid-run changes | launch | research refs |",
             "|---|---|---|---|---|---|---|---|"]
    for r in flagged:
        fd = "; ".join(f"{d['knob']}: {_fmt(d['run'])} vs {_fmt(d['parent'])} ({d['class']}"
                       f"{', =parser default' if d['eq_parser_default'] else ''}"
                       f"{', INERT' if d['inert'] else ''})"
                       for d in r["fork_diffs"]) or "-"
        tv = (r["trainer_vs_parent"] or {}).get("diffs_fork_first_vs_parent") or {}
        tvs = "; ".join(f"{k}: {_fmt(a)} vs {_fmt(b)}" for k, (a, b) in tv.items()) or "-"
        mc = "; ".join(f"{c['knob']}: {_fmt(c['from'])}→{_fmt(c['to'])} @ {c['at_step']}"
                       for c in r["mid_run_changes"])
        mc += "".join(f"; matchup {c['from_hash']}→{c['to_hash']} @ {c['at']}"
                      for c in r["matchup"]["changes"])
        launch = (f"pins={r['pin_history_len']}, relaunch={'Y' if r['launcher_differs_from_original'] else 'N'}"
                  f"{', MANUAL same-run resume' if r['last_launch_manual_same_run_resume'] else ''}"
                  f"{', inherited' if r['recipe_restart_inherited_recorded'] else ''}"
                  + (f", relaunch untyped@default={r['relaunch_untyped_at_parser_default']}"
                     if r.get("relaunch_untyped_at_parser_default") else ""))
        rf = "<br>".join(refs.get(r["run"], [])) or "none found"
        lines.append(f"| {r['run']} | {r['role']} | {r['parent_run'] or '-'} | {fd} | {tvs} | "
                     f"{mc.lstrip('; ') or '-'} | {launch} | {rf} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models-dir", default="/home/goodlad/dev/gen3ai/models")
    ap.add_argument("--json", default=os.path.join(HERE, "audit.json"))
    ap.add_argument("--md", default=os.path.join(HERE, "audit_table.md"))
    a = ap.parse_args(argv)
    defaults = parser_defaults()
    runs = sorted(d for d in glob.glob(os.path.join(a.models_dir, "rb_*")) if os.path.isdir(d))
    recs = [audit_run(d, defaults) for d in runs]
    refs = research_refs([r["run"] for r in recs if r["flagged"]])
    for r in recs:
        r["research_refs"] = refs.get(r["run"], [])
    with open(a.json, "w") as f:
        json.dump({"parser_defaults": defaults, "audit_rows": [r.dest for r in AUDIT_ROWS],
                   "runs": recs}, f, indent=1, default=str)
    with open(a.md, "w") as f:
        f.write(table(recs, refs))
    print(open(a.md).read().splitlines()[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
