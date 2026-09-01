"""FOLDING-HISTORY / POST-FOLD-CONSOLIDATION PROBE (mission M6).

Two "history of the parent" hypotheses that no probe has tested:

  (a) ACCUMULATED FOLDING — a network that has absorbed teachers before may be organized to
      absorb them again. The distillability index gestures at it (`R2ACTION`, already folded
      once, posts the highest a0 AND the highest a_max of any lr-3e-4 cell) but never tested
      PRIOR-FOLD COUNT as the variable.
  (b) POST-FOLD CONSOLIDATION — distilled content may need training time after the fold to
      settle into a form that generalizes; if v8's fold had far more post-fold steps than the
      gen-era folds, "transfer quality" could partly be "consolidation time".

Instrument: the ADMITTED `distillability_index_probe.py` (2026-08-28), mechanics unchanged —
full-policy Adam on masked CE to a fixed teacher's argmax, batch 256, 400 steps, 14 log-spaced
eval points; ABSORPTION = held-out on-slice top-1 agreement (held out BY BATTLE FILE);
COLLATERAL = off-slice masked KL / top-1 agreement / |dV| vs the student's OWN pre-probe policy.
The step-1 shock is recorded but reported as an ordering only (it failed value-level admission
in the source battery and nothing here re-admits it).

What is NEW here is the STUDENT axis: the student is an argument, and the roster is chosen to
vary prior-fold count while holding total training steps as constant as the archive allows.

============================================================================================
REGISTERED PREDICTIONS — declared before any cell ran, scored in `aggregate`, never tuned.
============================================================================================
  M1 (mission prediction 1)  Prior-fold count is CONFOUNDED with age in EVERY cell the archive
     supports, so the honest deliverable is the confounded table plus the named cell that would
     break it.  SCORING: the archive is searched for any pair of generalist checkpoints with
     DIFFERENT fold-ancestry counts at IDENTICAL total steps; M1 fails if one exists.
  M2 (mission prediction 2)  Post-fold consolidation does NOT order with the measured untaught
     outcome (v8 +5.42pp / rev-3 -0.75pp / rev-2 -7.06pp).  SCORING: Spearman rho between
     post-fold steps and untaught outcome over the three folds; M2 passes iff |rho| < 1.0
     (with n=3, a perfect ordering is the only non-trivial signal available and it happens by
     chance with probability 1/6 -- so a PASS is weak and a FAIL is reported as an ORDERING,
     never as a coefficient).
  A1 (the (a) test at matched age)  At IDENTICAL total steps (28,067,760) and identical parent,
     the 1-fold student's absorption ceiling a_max exceeds BOTH 0-fold arms by more than the
     instrument's measured seed noise (0.018), on both probe seeds, against the primary teacher.
  A2 (the (a) test across the ladder)  a_max is monotone non-decreasing in fold-ancestry count
     across {0,1,2} on the rev-lineage roster.

Run (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src):

    python folding_history_probe.py lineage
    python folding_history_probe.py build-states
    python folding_history_probe.py probe <cell> <student> <teacher_set> <seed> <steps> <lr>
    python folding_history_probe.py aggregate
    python folding_history_probe.py report

`<student>` is a key of STUDENTS below; `<teacher_set>` is `f` / `a`, or `f*` / `a*` for the
CONTENT CONTROL on that set's states (targets become the student's own argmax -- same optimizer,
same states, zero new behavioural content). QUOTE the control token: bare `f*` is glob-expanded
by the shell and the cell dies on argv parsing (it killed four cells on the lr battery).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import glob
import json
import re
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("PROBE_TORCH_THREADS", "1")))

NEG = -1e8
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = "/home/goodlad/dev/gen3ai/models"
RESULTS = os.path.join(HERE, "fh_results")
OBS_DIM = 2501
BATCH = 256
SEED_NOISE_GAIN = 0.018  # measured seed-to-seed |d gain@400| bound, 2026-08-28 battery §2

# ---------------------------------------------------------------------------
# THE STUDENT ROSTER
#
# `folds` = number of runs in the checkpoint's ANCESTRY (root -> this run, inclusive) that
# carried a --distill-teacher list with a NON-ZERO --distill-coef.  Computed independently by
# `lineage` from metadata alone; the values below are asserted against it, so a roster that
# drifts from the recorded commands fails loudly rather than being believed.
#
# ARM A is the confound-BREAKING cell: three checkpoints at byte-identical total steps
# (28,067,760), all forked from the same rev-1 final, differing only in the distill term.
# ARM B is the fold-count ladder above 1, which the archive supports only WITH the age confound.
# ARM C is a second, independent lineage (tick-1) with the same ladder shape.
# ARM D is the ancestry-free control (gen-17 shares no weights with the rev lineage).
# ---------------------------------------------------------------------------
STUDENTS = {
    # key            run                            arm  expected folds
    "root":       ("ai_v9_29_rev1_0823",            "A", 0),
    "plain0":     ("ai_v9_62_R2PLAIN_0827",         "A", 0),
    # `ecol0` was PLANNED as the teacher-ECOLOGY arm (teachers listed at --distill-coef 0, so the
    # team distribution is biased toward the teacher teams while no loss folds). It is NOT one.
    # `apply_distill_team_bias`'s own docstring records that R2CTRL "got an effective bias of 0.0
    # (the pairs were parsed only above coef 0)" — the defect `gen3_distill_bias_at_coef0_v1`
    # later fixed — and its metadata confirms it: `cli_args._distill_pairs` is EMPTY, against 5
    # for `fold1`. So `ecol0` is a SECOND REPLICATE of the plain +3M continuation, and that is
    # better for A1 than the arm it was meant to be: `plain0` vs `ecol0` measures the RUN-TO-RUN
    # spread of the 0-fold condition at matched steps, which is the yardstick a 1-fold delta has
    # to clear. Do not describe this cell as an ecology control.
    "ecol0":      ("ai_v9_58_R2CTRL_0827",          "A", 0),
    "fold1":      ("ai_v9_59_R2ACTION_0827",        "A", 1),
    "fold2":      ("ai_v9_76_R4ACTION_0830",        "B", 2),
    "fold2self":  ("ai_v9_72_R3SELF_0828",          "B", 2),
    "tick1":      ("ai_v9_34_tick1_0824",           "C", 1),
    "tick1x2":    ("ai_v9_37_tick1_dosext_0825",    "C", 2),
    "gen17":      ("ai_v9_21_gen17_pfspoff_0820",   "D", 0),
}
STUDENT_PATHS = {k: f"{MODELS}/{v[0]}/final_model.zip" for k, v in STUDENTS.items()}

# ---------------------------------------------------------------------------
# TEACHERS
#
# Both are rev-1-final-initialised exploiters targeting `ai_v9_59_R2ACTION_0827`, so they sit at
# the SAME DAG distance from every ARM-A student (each is rev-1-final + ~3M).  They differ in
# SLICE EXPOSURE, which is the confound a single teacher cannot separate:
#   f = ai_v9_68_R3F6f  teams {460b8c99, 7036a0a1} -- UNTAUGHT to root/plain0/ecol0/fold1/tick*,
#       taught to fold2 (via R4S3c, which pins a superset).  PRIMARY, because the arm-A cell --
#       the one that breaks the age confound -- is slice-clean for every one of its three arms.
#   a = ai_v9_63_R3F6a  teams {eccfe630, 023a2d47} -- taught to fold1 (via R2F5a, the same two
#       teams) and to fold2 (via R4S3a).  SECONDARY: it is the teacher-independence check AND
#       the deliberate slice-exposed contrast against `f`.
# ---------------------------------------------------------------------------
TEACHER_RUNS = {"f": "ai_v9_68_R3F6f_0828", "a": "ai_v9_63_R3F6a_0828"}
TEACHERS = {k: f"{MODELS}/{r}/final_model.zip" for k, r in TEACHER_RUNS.items()}

# OFF-slice source: rev-1's traces (2456 files / 281 teams / 12 eval steps) -- the admitted
# instrument's own off-source and the widest in the archive.  Collateral is drift from the
# STUDENT's own pre-probe policy on these states, so what the meter needs is BREADTH, not the
# student's exact on-policy distribution; caveat 3 of the 2026-08-28 instrument applies unchanged
# and is NOT repaired by this choice (and applies harder to `gen17`, which is not in this lineage).
OFF_RUN = "ai_v9_29_rev1_0823"

# The three folds whose untaught-team outcome has been MEASURED, with provenance.  Used only by
# the consolidation table; never by the probe cells.
FOLD_OUTCOMES = {
    "ai_v8_14_distill3_0725": dict(
        label="v8", untaught_pp=+5.42, ci=[+3.44, +7.42], z=+4.83,
        source="v8_redistribution_pfsp_2026-08-30.md §4 (16 untaught teams, 7,680 battles)"),
    "ai_v9_59_R2ACTION_0827": dict(
        label="rev-2", untaught_pp=-7.06, ci=[-10.56, -3.50], z=-3.86,
        source="rev3_untaught_pulldown_2026-08-30.md row B-C (8 sample teams, this instrument)"),
    "ai_v9_70_R3ACTION_0828": dict(
        label="rev-3", untaught_pp=-0.75, ci=[-4.56, +3.00], z=-0.39,
        source="rev3_untaught_pulldown_2026-08-30.md row A-B"),
}


# ============================================================================
# 1. THE LINEAGE DAG
# ============================================================================

def _flag(cmd: str, f: str):
    m = re.search(rf'{re.escape(f)}\s+(\S+)', cmd or "")
    return m.group(1) if m else None


def _run_of(path: str | None):
    """A checkpoint path -> its run directory NAME, or None."""
    if not path:
        return None
    p = path.replace("/home/goodlad/dev/gen3ai/", "")
    if p.startswith("models/"):
        p = p[len("models/"):]
    return p.split("/")[0] or None


def _max_ckpt_steps(run: str):
    cks = glob.glob(f"{MODELS}/{run}/checkpoints/checkpoint_*_steps.zip") + \
          glob.glob(f"{MODELS}/{run}/checkpoint_*_steps.zip")
    vals = [int(re.search(r'checkpoint_(\d+)_steps', c).group(1)) for c in cks]
    return max(vals) if vals else None


def scan_runs():
    """Every run in models/ with a metadata.json, keyed by run name.

    Lineage is read from `original_command` ONLY.  `cli_args["model"]` is overwritten by each
    RESUMING process and points at the run's own `final_model_interrupted.zip` -- self-referential
    and useless for ancestry (CLAUDE.md: `original_command` is the immutable original invocation).
    A run whose `original_command` is absent or names no `--model` is recorded as a ROOT with
    `parent_resolved=False` and is never guessed at.
    """
    runs = {}
    for d in sorted(os.listdir(MODELS)):
        md = os.path.join(MODELS, d, "metadata.json")
        if not os.path.isfile(md):
            continue
        try:
            m = json.load(open(md))
        except Exception as exc:  # noqa: BLE001
            runs[d] = dict(run=d, error=f"unreadable metadata: {exc}")
            continue
        oc = m.get("original_command") or ""
        ca = m.get("cli_args") or {}
        model_arg = _flag(oc, "--model")
        # A fork of `init_model.zip` inherits the parent's INITIALISATION, not its training —
        # `ai_v8_03_zarch_control_0718` does exactly this off `ai_v8_01`. The DAG edge is real
        # (it is where the weights came from) but the parent contributes ZERO trained steps, so
        # `own_steps` must not be differenced against it. Flagged, never silently differenced.
        parent_is_init = bool(model_arg and os.path.basename(model_arg) == "init_model.zip")
        teach = _flag(oc, "--distill-teacher")
        coef = ca.get("distill_coef")
        # `--distill-teacher <run>:<teams>` — `*` means "every team that teacher was pinned to",
        # an explicit comma list means exactly those. Both forms are live (rev-2/3/4 use `*`,
        # COMPFOLD names teams), so the SLICE a fold taught is only recoverable by parsing both.
        teacher_specs = []
        for t in (teach.split(";") if teach else []):
            run_, _, teams_ = t.partition(":")
            teacher_specs.append(dict(
                run=_run_of(run_),
                teams=None if teams_.strip() in ("*", "") else
                      [x.split("/")[-1].replace(".txt", "") for x in teams_.split(",") if x]))
        teachers = [s["run"] for s in teacher_specs]
        runs[d] = dict(
            run=d,
            parent=_run_of(model_arg),
            parent_is_init=parent_is_init,
            parent_resolved=bool(oc),
            exploiter_target=_run_of(_flag(oc, "--exploiter")),
            is_exploiter=bool(_flag(oc, "--exploiter")),
            teachers=teachers,
            teacher_specs=teacher_specs,
            distill_coef=coef,
            distill_target=ca.get("distill_target"),
            distill_team_bias=ca.get("distill_team_bias"),
            # A FOLD is a teacher list AND a non-zero coefficient.  `ai_v9_58_R2CTRL_0827` lists
            # five teachers at coef 0.0 -- it is the ECOLOGY control (team sampling biased toward
            # the teacher teams, no distillation term), and counting it as a fold would be wrong.
            is_fold=bool(teachers) and bool(coef),
            # A SELF-fold (`ai_v9_72_R3SELF_0828`: teacher == its own parent) is a fold in every
            # optimizer sense and carries ZERO external content. It must count for fold COUNT and
            # must NOT count as having been taught a teacher's behaviour on those teams.
            is_self_fold=bool(teachers) and bool(coef) and teachers == [_run_of(model_arg)],
            lr=ca.get("lr"),
            steps_flag=ca.get("steps"),
            final_steps=_max_ckpt_steps(d),
            git_hash=m.get("git_hash"),
            trace_files=len(glob.glob(f"{MODELS}/{d}/eval_traces/*/*/*_summary.json")),
            trainee_teams=[t.split("/")[-1].replace(".txt", "")
                           for t in (_flag(oc, "--trainee-teams") or "").split(",") if t],
        )
    return runs


def ancestry(runs, run, _seen=None):
    """[root, ..., run].  Cycles and unresolvable parents terminate the walk, never loop."""
    _seen = _seen or set()
    if run in _seen or run not in runs:
        return []
    _seen.add(run)
    par = runs[run].get("parent")
    if par == run:                     # self-reference (a resumed run's own interrupted ckpt)
        par = None
    return (ancestry(runs, par, _seen) if par else []) + [run]


def lineage(write=True):
    runs = scan_runs()
    for r in runs.values():
        if r.get("error"):
            continue
        chain = ancestry(runs, r["run"])
        r["chain"] = chain
        r["folds_in_ancestry"] = sum(1 for c in chain if runs[c].get("is_fold"))
        r["fold_runs_in_ancestry"] = [c for c in chain if runs[c].get("is_fold")]
        par = runs.get(r["parent"] or "")
        if r.get("parent_is_init"):
            r["own_steps"] = r["final_steps"]      # forked an INIT: every step is its own
        else:
            r["own_steps"] = (r["final_steps"] - par["final_steps"]
                              if r["final_steps"] and par and par.get("final_steps") else None)
        # An unresolved ancestor makes the fold count a LOWER BOUND — flagged, never smoothed.
        #
        # ⚠️ The obvious form of this check is VACUOUS and was written that way first:
        # `parent_resolved or not parent` can never be False, because `parent` is READ FROM
        # `original_command`, so a run with no recorded command has no parent by construction and
        # passes trivially. It reported 0 unresolved lineages out of 160 while 21 ai_v5/ai_v6-era
        # runs carry no `original_command` at all — their parent is UNKNOWN, not absent, and the
        # walk was silently calling each of them a root. The honest predicate is whether every
        # run in the chain STATES its invocation.
        # `parent_resolved` IS "this run states its invocation" (it is `bool(original_command)`),
        # set for every run in scan_runs, so it is order-independent here.
        r["ancestry_complete"] = (all(runs[c]["parent_resolved"] for c in chain)
                                  and (r["parent"] is None or r["parent"] in runs))

    # roster cross-check: the hand-written folds in STUDENTS must equal the derived count
    mism = {}
    for k, (rname, _arm, exp) in STUDENTS.items():
        got = runs.get(rname, {}).get("folds_in_ancestry")
        if got != exp:
            mism[k] = dict(run=rname, roster=exp, derived=got)

    # ---- SLICE EXPOSURE: which TEAMS has each student already been taught? -----------------
    # A fold-count contrast is also a slice-exposure contrast unless the probe teacher's teams
    # are untaught to every arm. This computes it from metadata rather than asserting it in
    # prose: taught(student) = union over the folds in its ancestry of each teacher spec's teams
    # (the spec's explicit list, else that teacher's own `--trainee-teams`).
    def taught_teams(run, content_only=True):
        out = set()
        for c in runs[run].get("chain") or []:
            if not runs[c].get("is_fold"):
                continue
            if content_only and runs[c].get("is_self_fold"):
                continue          # a self-fold biased the ECOLOGY, it taught no new behaviour
            for spec in runs[c].get("teacher_specs") or []:
                if spec["teams"]:
                    out |= set(spec["teams"])
                elif spec["run"] in runs:
                    out |= set(runs[spec["run"]].get("trainee_teams") or [])
        return out

    exposure = {}
    for k, (rname, _arm, _f) in STUDENTS.items():
        tt = taught_teams(rname)                       # CONTENT: external behaviour absorbed
        te = taught_teams(rname, content_only=False)   # + ECOLOGY: teams a self-fold also biased
        exposure[k] = dict(n_taught_teams=len(tt), taught_teams=sorted(tt),
                           n_ecology_teams=len(te), per_teacher={})
        for tk, trun in TEACHER_RUNS.items():
            pins = set(runs.get(trun, {}).get("trainee_teams") or [])
            exposure[k]["per_teacher"][tk] = dict(
                teacher_run=trun, teacher_teams=sorted(pins),
                overlap=sorted(pins & tt), slice_taught=bool(pins & tt),
                slice_in_ecology=bool(pins & te))

    # the consolidation table
    consolidation = []
    for frun, o in FOLD_OUTCOMES.items():
        r = runs.get(frun, {})
        par = runs.get(r.get("parent") or "", {})
        kids = [k for k, v in runs.items() if v.get("parent") == frun]
        try:
            sig = json.load(open(f"{MODELS}/{frun}/model_config.json")).get("arch_signature")
        except Exception:  # noqa: BLE001
            sig = None
        consolidation.append(dict(
            fold_run=frun, label=o["label"], arch_signature=sig,
            parent=r.get("parent"), parent_steps=par.get("final_steps"),
            fold_end_steps=r.get("final_steps"),
            steps_under_distill=r.get("own_steps"),
            distill_coef=r.get("distill_coef"), lr=r.get("lr"),
            n_teachers=len(r.get("teachers") or []),
            prior_folds=(r.get("folds_in_ancestry") or 0) - 1,
            descendants=kids,
            untaught_pp=o["untaught_pp"], ci=o["ci"], z=o["z"], outcome_source=o["source"]))
    consolidation.sort(key=lambda c: -(c["steps_under_distill"] or 0))

    # M1: does the archive hold a fold-count contrast at IDENTICAL total steps?
    by_steps = {}
    for r in runs.values():
        if r.get("error") or not r.get("final_steps") or r.get("is_exploiter"):
            continue
        by_steps.setdefault(r["final_steps"], []).append(r)
    m1_breakers = []
    for st, group in sorted(by_steps.items()):
        counts = {g["folds_in_ancestry"] for g in group}
        if len(counts) > 1:
            m1_breakers.append(dict(total_steps=st, fold_counts=sorted(counts),
                                    runs={g["run"]: g["folds_in_ancestry"] for g in group}))

    out = dict(
        generated="folding_history_probe.py lineage",
        n_runs=len(runs),
        roster_mismatches=mism,
        students={k: dict(run=v[0], arm=v[1], roster_folds=v[2],
                          derived_folds=runs.get(v[0], {}).get("folds_in_ancestry"),
                          final_steps=runs.get(v[0], {}).get("final_steps"),
                          own_steps=runs.get(v[0], {}).get("own_steps"),
                          parent=runs.get(v[0], {}).get("parent"),
                          chain=runs.get(v[0], {}).get("chain"),
                          fold_runs=runs.get(v[0], {}).get("fold_runs_in_ancestry"),
                          ancestry_complete=runs.get(v[0], {}).get("ancestry_complete"),
                          lr=runs.get(v[0], {}).get("lr"))
                  for k, v in STUDENTS.items()},
        consolidation=consolidation,
        slice_exposure=exposure,
        m1_matched_step_fold_contrasts=m1_breakers,
        unresolved_lineages=sorted(r["run"] for r in runs.values()
                                   if not r.get("error") and not r.get("ancestry_complete")),
        no_original_command=sorted(r["run"] for r in runs.values()
                                   if not r.get("error") and not r["parent_resolved"]),
        runs=runs,
    )
    if write:
        json.dump(out, open(os.path.join(HERE, "fh_lineage.json"), "w"), indent=1)

    print(f"[lineage] {len(runs)} runs scanned")
    if mism:
        print(f"  !! ROSTER MISMATCH (hand-written vs derived): {json.dumps(mism)}")
    print(f"\n{'student':11s} {'run':30s} {'arm':4s} {'folds':6s} {'steps':>11s} {'own':>10s}  chain")
    for k, v in STUDENTS.items():
        s = out["students"][k]
        print(f"{k:11s} {s['run']:30s} {s['arm']:4s} {str(s['derived_folds']):6s} "
              f"{str(s['final_steps']):>11s} {str(s['own_steps']):>10s}  "
              f"{' -> '.join(s['chain'] or [])}")
    print("\n[consolidation]")
    for c in consolidation:
        print(f"  {c['label']:6s} {c['fold_run']:30s} under-distill {str(c['steps_under_distill']):>10s} "
              f"coef {c['distill_coef']} lr {c['lr']}  untaught {c['untaught_pp']:+.2f}pp")
    print(f"\n[M1] matched-total-step fold-count contrasts found: {len(m1_breakers)}")
    for b in m1_breakers[:10]:
        print(f"  {b['total_steps']}: {b['runs']}")
    print(f"\n[unresolved ancestry, never guessed] {len(out['unresolved_lineages'])} of {len(runs)} runs"
          f" ({len(out['no_original_command'])} carry no `original_command` at all): "
          f"{', '.join(out['unresolved_lineages'][:8])}"
          f"{' ...' if len(out['unresolved_lineages']) > 8 else ''}")
    bad = [k for k, v in STUDENTS.items() if not runs[v[0]]["ancestry_complete"]] + \
          [c["fold_run"] for c in consolidation if not runs[c["fold_run"]]["ancestry_complete"]]
    print(f"[roster/consolidation members with an unresolved ancestry] "
          f"{bad if bad else 'NONE — every fold count below is exact, not a lower bound'}")
    return out


# ============================================================================
# 2. THE INSTRUMENT (verbatim mechanics from distillability_index_probe.py)
# ============================================================================

def load_policy(ckpt_path: str, device: str = "cpu"):
    from sb3_contrib import MaskablePPO
    from main.prober.model import sanitized_load_custom_objects, peek_checkpoint, _arch_drift_error
    peek = peek_checkpoint(ckpt_path)
    custom_objects, dropped = sanitized_load_custom_objects(ckpt_path, device)
    try:
        model = MaskablePPO.load(ckpt_path, device=device, custom_objects=custom_objects)
    except Exception as exc:  # noqa: BLE001
        raise _arch_drift_error(ckpt_path, peek, dropped, exc) from exc
    _silence(model.policy)
    return model, tuple(dropped)


def _silence(policy):
    policy.set_training_mode(False)
    for m in policy.modules():
        if hasattr(m, "_debugger"):
            m._debugger = None


def masked_logits(policy, obs_t, mask_t):
    d = policy.get_distribution({"observation": obs_t, "action_mask": mask_t})
    lg = d.distribution.logits
    return torch.where(mask_t.bool(), lg, torch.full_like(lg, NEG))


@torch.no_grad()
def eval_probs(policy, obs, mask, bs=256):
    out = []
    for i in range(0, len(obs), bs):
        ml = masked_logits(policy, torch.as_tensor(obs[i:i + bs]), torch.as_tensor(mask[i:i + bs]))
        out.append(torch.log_softmax(ml, 1).double().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def eval_values(policy, obs, mask, bs=256):
    out = []
    for i in range(0, len(obs), bs):
        ot, mt = torch.as_tensor(obs[i:i + bs]), torch.as_tensor(mask[i:i + bs])
        out.append(policy.predict_values({"observation": ot, "action_mask": mt}).flatten().double().numpy())
    return np.concatenate(out, 0)


def kl_rows(logp_new, logp_ref, mask):
    m = mask.astype(bool)
    p = np.exp(logp_new)
    return np.where(m, p * (logp_new - logp_ref), 0.0).sum(1)


def harvest(run):
    recs = []
    for sm in sorted(glob.glob(f"{MODELS}/{run}/eval_traces/*/*/*_summary.json")):
        npz = sm.replace("_summary.json", "_states.npz")
        if not os.path.exists(npz):
            continue
        try:
            s = json.load(open(sm))
        except Exception:  # noqa: BLE001
            continue
        team = tuple(sorted(m["species"] for m in s.get("teams", {}).get("ours", [])))
        if len(team) != 6:
            continue
        z = np.load(npz)
        obs, mask = z["obs"], z["action_mask"]
        if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
            continue
        keep = mask.sum(1) >= 2
        if not keep.any():
            continue
        parts = sm.split("/")
        recs.append(dict(file=sm, team=team, step=parts[-3], opp=parts[-2],
                         obs=obs[keep].astype(np.float32), mask=mask[keep].astype(bool)))
    return recs


def pick(recs, n, seed, per_file_cap=None):
    rng = np.random.default_rng(seed)
    order = list(range(len(recs)))
    rng.shuffle(order)
    picked_o, picked_m, prov = [], [], []
    idx = {i: rng.permutation(len(recs[i]["obs"])).tolist() for i in order}
    total = 0
    while total < n:
        progressed = False
        for i in order:
            q = idx[i]
            if not q:
                continue
            if per_file_cap is not None and sum(1 for p in prov if p[0] == i) >= per_file_cap:
                continue
            j = q.pop()
            picked_o.append(recs[i]["obs"][j]); picked_m.append(recs[i]["mask"][j])
            prov.append((i, j)); total += 1; progressed = True
            if total >= n:
                break
        if not progressed:
            break
    return np.stack(picked_o), np.stack(picked_m), prov


def build_states():
    p_recs_all = harvest(OFF_RUN)
    print(f"[off-source] {OFF_RUN}: {len(p_recs_all)} files, "
          f"{sum(len(r['obs']) for r in p_recs_all)} rows, {len({r['team'] for r in p_recs_all})} teams")
    prov = dict(off_run=OFF_RUN, teachers={}, students=STUDENT_PATHS,
                filters=f"action_mask.sum(1) >= 2; obs dim {OBS_DIM}; ON split by BATTLE FILE (75/25)")
    for i, (key, run) in enumerate(sorted(TEACHER_RUNS.items())):
        t_recs = harvest(run)
        teams = sorted({r["team"] for r in t_recs})
        assert 1 <= len(teams) <= 12, (run, len(teams))   # a generalist census => wrong run
        print(f"[on:{key}] {run}: {len(t_recs)} files, {sum(len(r['obs']) for r in t_recs)} rows, "
              f"{len(teams)} pinned teams")
        rng = np.random.default_rng(20260831 + i)
        fidx = rng.permutation(len(t_recs))
        held = set(fidx[:max(1, len(t_recs) // 4)].tolist())
        tr = [r for j, r in enumerate(t_recs) if j not in held]
        he = [r for j, r in enumerate(t_recs) if j in held]
        on_tr_o, on_tr_m, _ = pick(tr, 3000, 11 + 10 * i)
        on_he_o, on_he_m, _ = pick(he, 1200, 12 + 10 * i)
        p_recs = [r for r in p_recs_all if r["team"] not in set(teams)]
        off_o, off_m, off_prov = pick(p_recs, 1500, 13 + 10 * i, per_file_cap=2)
        off_teams = len({p_recs[j]["team"] for j, _ in off_prov})
        off_steps = sorted({p_recs[j]["step"] for j, _ in off_prov})
        print(f"[off:{key}] {len(p_recs)} files after pin-exclusion; {len(off_o)} rows / "
              f"{off_teams} teams / {len(off_steps)} eval steps")
        np.savez_compressed(os.path.join(HERE, f"fh_states_{key}.npz"),
                            on_train_obs=on_tr_o, on_train_mask=on_tr_m,
                            on_held_obs=on_he_o, on_held_mask=on_he_m,
                            off_obs=off_o, off_mask=off_m)
        prov["teachers"][key] = dict(run=run, pinned_teams=[list(t) for t in teams],
                                     trace_files=len(t_recs), train_files=len(tr), held_files=len(he),
                                     n_on_train=int(len(on_tr_o)), n_on_held=int(len(on_he_o)),
                                     off=dict(n=int(len(off_o)), distinct_teams=off_teams,
                                              eval_steps=len(off_steps),
                                              files_after_exclusion=len(p_recs)))
    json.dump(prov, open(os.path.join(HERE, "fh_state_provenance.json"), "w"), indent=2)
    print("[done] fh_states_{a,f}.npz + fh_state_provenance.json")


def eval_points(n):
    pts, v = {0, n}, 1
    while v <= n:
        pts.add(v); v = int(np.ceil(v * 1.6))
    return sorted(pts)


def probe(argv):
    cell, skey, tset, seed, nsteps, LR = (argv[0], argv[1], argv[2],
                                          int(argv[3]), int(argv[4]), float(argv[5]))
    if skey not in STUDENTS:
        raise SystemExit(f"unknown student {skey!r}; known: {sorted(STUDENTS)}")
    student = STUDENT_PATHS[skey]
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"{cell}.json")
    t_start = time.time()

    self_target = tset.endswith("*")
    tset = tset.rstrip("*")
    S = np.load(os.path.join(HERE, f"fh_states_{tset}.npz"))
    on_tr_o, on_tr_m = S["on_train_obs"], S["on_train_mask"]
    on_he_o, on_he_m = S["on_held_obs"], S["on_held_mask"]
    off_o, off_m = S["off_obs"], S["off_mask"]

    tcache = os.path.join(HERE, f"fh_teacher_targets_{tset}.npz")
    if os.path.exists(tcache):
        T = np.load(tcache)
        tgt_tr, tgt_he, tprob_he = T["tgt_train"], T["tgt_held"], T["tprob_held"]
    else:
        tm, tdrop = load_policy(TEACHERS[tset])
        if tdrop:
            raise SystemExit(f"REFUSED: teacher {TEACHERS[tset]} dropped kwargs {tdrop}")
        lp_tr = eval_probs(tm.policy, on_tr_o, on_tr_m)
        lp_he = eval_probs(tm.policy, on_he_o, on_he_m)
        tgt_tr, tgt_he, tprob_he = lp_tr.argmax(1), lp_he.argmax(1), np.exp(lp_he)
        # ATOMIC: two cells on the same teacher run concurrently in the battery, and a torn .npz
        # is a SILENT corruption (np.load would raise, but a half-written file that happens to
        # parse would not). Write private, then rename — rename within a directory is atomic.
        tmp = f"{tcache}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, tgt_train=tgt_tr, tgt_held=tgt_he, tprob_held=tprob_he)
        os.replace(tmp, tcache)
        del tm

    m, dropped = load_policy(student)
    policy = m.policy
    if dropped:
        raise SystemExit(f"REFUSED: {student} dropped kwargs {dropped} — not a faithful rebuild")

    if self_target:
        tgt_tr = eval_probs(policy, on_tr_o, on_tr_m).argmax(1)
        lp0 = eval_probs(policy, on_he_o, on_he_m)
        tgt_he, tprob_he = lp0.argmax(1), np.exp(lp0)

    ref_off_lp = eval_probs(policy, off_o, off_m)
    ref_off_arg = ref_off_lp.argmax(1)
    ref_off_val = eval_values(policy, off_o, off_m)

    tr_t, trm_t = torch.as_tensor(on_tr_o), torch.as_tensor(on_tr_m)
    tgt_t = torch.as_tensor(tgt_tr.astype(np.int64))
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    rng = np.random.default_rng(seed)     # SAME batch sequence across cells at a given seed
    pts = set(eval_points(nsteps))
    curve, losses = [], []

    def snapshot(step):
        lp_he = eval_probs(policy, on_he_o, on_he_m)
        lp_tr = eval_probs(policy, on_tr_o[:1200], on_tr_m[:1200])
        lp_off = eval_probs(policy, off_o, off_m)
        val_off = eval_values(policy, off_o, off_m)
        kl = kl_rows(lp_off, ref_off_lp, off_m)
        rec = dict(step=step,
                   on_held_agree=float((lp_he.argmax(1) == tgt_he).mean()),
                   on_train_agree=float((lp_tr.argmax(1) == tgt_tr[:1200]).mean()),
                   on_held_teacher_ce=float(-np.take_along_axis(lp_he, tgt_he[:, None], 1).mean()),
                   on_held_tv=float(0.5 * np.abs(np.exp(lp_he) - tprob_he).sum(1).mean()),
                   off_kl=float(kl.mean()), off_kl_median=float(np.median(kl)),
                   off_agree=float((lp_off.argmax(1) == ref_off_arg).mean()),
                   off_value_mad=float(np.abs(val_off - ref_off_val).mean()),
                   off_value_corr=float(np.corrcoef(val_off, ref_off_val)[0, 1]),
                   elapsed_s=round(time.time() - t_start, 1))
        curve.append(rec)
        print(f"  step {step:5d}  on_held {rec['on_held_agree']:.4f}  offKL {rec['off_kl']:.4f}  "
              f"offAgree {rec['off_agree']:.4f}  Vmad {rec['off_value_mad']:.3f}  "
              f"[{rec['elapsed_s']}s]", flush=True)

    snapshot(0)
    for step in range(1, nsteps + 1):
        idx = torch.as_tensor(rng.choice(len(on_tr_o), BATCH, replace=False))
        ml = masked_logits(policy, tr_t[idx], trm_t[idx])
        loss = torch.nn.functional.cross_entropy(ml, tgt_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        if step in pts:
            snapshot(step)

    json.dump(dict(cell=cell, student_key=skey, student=student,
                   student_run=STUDENTS[skey][0], arm=STUDENTS[skey][1],
                   roster_folds=STUDENTS[skey][2],
                   teacher_set=tset,
                   teacher=("SELF(content control)" if self_target else TEACHERS[tset]),
                   self_target=self_target, probe_seed=seed, n_steps=nsteps, lr=LR, batch=BATCH,
                   n_on_train=int(len(on_tr_o)), n_on_held=int(len(on_he_o)), n_off=int(len(off_o)),
                   dropped_kwargs=list(dropped), curve=curve,
                   loss_first10=losses[:10], loss_last10=losses[-10:],
                   wall_s=round(time.time() - t_start, 1)),
              open(out_path, "w"), indent=2)
    print(f"[done] {cell} -> {out_path}  ({time.time()-t_start:.0f}s)")


# ============================================================================
# 3. AGGREGATION + REGISTERED-PREDICTION SCORING
# ============================================================================

def crossing(curve, thresh, use_gain):
    a = np.maximum.accumulate(np.array([c["on_held_agree"] for c in curve], float))
    lvl = a - (a[0] if use_gain else 0.0)
    if lvl.max() < thresh:
        return None
    j = int(np.argmax(lvl >= thresh))
    if j == 0:
        return {k: curve[0][k] for k in ("off_kl", "off_agree", "off_value_mad")} | {"step": 0, "exact": True}
    lo, hi = lvl[j - 1], lvl[j]
    w = 0.0 if hi == lo else (thresh - lo) / (hi - lo)
    out = {"step": curve[j - 1]["step"] + w * (curve[j]["step"] - curve[j - 1]["step"]), "exact": False}
    for k in ("off_kl", "off_agree", "off_value_mad"):
        out[k] = float(curve[j - 1][k] + w * (curve[j][k] - curve[j - 1][k]))
    return out


def summarize(d):
    c = d["curve"]
    a = np.array([x["on_held_agree"] for x in c], float)
    s = dict(cell=d["cell"], student_key=d["student_key"], student_run=d["student_run"],
             arm=d["arm"], roster_folds=d["roster_folds"], teacher_set=d["teacher_set"],
             self_target=d.get("self_target", False), seed=d["probe_seed"], lr=d["lr"],
             a0=float(a[0]), a_max=float(a.max()), gain_max=float(a.max() - a[0]),
             step1_shock_kl=float(c[1]["off_kl"]) if len(c) > 1 else None,
             final_off_kl=float(c[-1]["off_kl"]), final_off_agree=float(c[-1]["off_agree"]),
             final_off_value_mad=float(c[-1]["off_value_mad"]),
             final_off_value_corr=float(c[-1]["off_value_corr"]))
    for g in (0.03, 0.05, 0.10):
        s[f"idx_gain_{g:.2f}"] = crossing(c, g, True)
    for A in (0.70, 0.78, 0.80):
        s[f"idx_abs_{A:.2f}"] = crossing(c, A, False)
    by_step = {x["step"]: x for x in c}
    for S in (32, 135, 400):
        x = by_step.get(S)
        if x is None:
            s[f"at_{S}"] = None
            continue
        gain = x["on_held_agree"] - s["a0"]
        s[f"at_{S}"] = dict(gain=float(gain), off_kl=float(x["off_kl"]),
                            off_agree=float(x["off_agree"]),
                            off_value_mad=float(x["off_value_mad"]),
                            on_held=float(x["on_held_agree"]),
                            eff=float(gain / x["off_kl"]) if x["off_kl"] > 1e-9 else None)
    return s


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    if len(x) < 2 or rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def aggregate():
    cells = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        d = json.load(open(f)); d["cell"] = os.path.basename(f)[:-5]
        cells[d["cell"]] = d
    summ = {k: summarize(v) for k, v in cells.items()}
    lin = lineage(write=True)

    def get(skey, tset, lr, seed, ctrl=False):
        for s in summ.values():
            if (s["student_key"] == skey and s["teacher_set"] == tset
                    and abs(s["lr"] - lr) < 1e-12 and s["seed"] == seed
                    and s["self_target"] == ctrl):
                return s
        return None

    PRIMARY_T, PRIMARY_LR = "f", 1e-4
    seeds = sorted({s["seed"] for s in summ.values()
                    if s["teacher_set"] == PRIMARY_T and not s["self_target"]
                    and abs(s["lr"] - PRIMARY_LR) < 1e-12})

    # ---- A1: the matched-age cell (28,067,760 steps; plain0 / ecol0 vs fold1) ----
    a1 = {}
    for sd in seeds:
        f1 = get("fold1", PRIMARY_T, PRIMARY_LR, sd)
        p0 = get("plain0", PRIMARY_T, PRIMARY_LR, sd)
        e0 = get("ecol0", PRIMARY_T, PRIMARY_LR, sd)
        if not (f1 and p0 and e0):
            continue
        d_plain = f1["a_max"] - p0["a_max"]
        d_ecol = f1["a_max"] - e0["a_max"]
        a1[f"s{sd}"] = dict(
            seed=sd,
            amax_fold1=f1["a_max"], amax_plain0=p0["a_max"], amax_ecol0=e0["a_max"],
            a0_fold1=f1["a0"], a0_plain0=p0["a0"], a0_ecol0=e0["a0"],
            gain_fold1=f1["gain_max"], gain_plain0=p0["gain_max"], gain_ecol0=e0["gain_max"],
            kl400_fold1=f1["final_off_kl"], kl400_plain0=p0["final_off_kl"],
            kl400_ecol0=e0["final_off_kl"],
            delta_vs_plain=d_plain, delta_vs_ecol=d_ecol,
            # `plain0` and `ecol0` are two 0-fold replicates of the SAME +3M continuation (see the
            # STUDENTS note): their gap is the RUN-TO-RUN noise floor, and it is the honest scale
            # for this comparison. SEED_NOISE_GAIN is the PROBE-seed bound and is the scale the
            # prediction was registered against — both are reported, and if the run-to-run gap is
            # the larger one then the registered threshold understates the noise. Stated, not
            # silently swapped: the registered verdict is scored on what was registered.
            run_to_run_gap=abs(p0["a_max"] - e0["a_max"]),
            margin_over_run_to_run=min(d_plain, d_ecol) - abs(p0["a_max"] - e0["a_max"]),
            exceeds_noise=bool(d_plain > SEED_NOISE_GAIN and d_ecol > SEED_NOISE_GAIN))
    # TRI-STATE, deliberately. `all([])` is True and `bool({}) and all(...)` is False; both are
    # verdicts a battery with MISSING cells has not earned. An unrun A1 must read INSUFFICIENT,
    # never PASS and never FAIL — the vacuous-guard class this repo has been bitten by.
    a1_pass = (None if len(a1) < 2 else all(v["exceeds_noise"] for v in a1.values()))

    # ---- A2: monotone in fold count on each lineage ----
    a2 = {}
    for name, keys in (("rev_lineage", ["plain0", "fold1", "fold2"]),
                       ("rev_lineage_self2", ["plain0", "fold1", "fold2self"]),
                       ("tick_lineage", ["root", "tick1", "tick1x2"])):
        for sd in seeds:
            vals = [get(k, PRIMARY_T, PRIMARY_LR, sd) for k in keys]
            if any(v is None for v in vals):
                continue
            folds = [lin["students"][k]["derived_folds"] for k in keys]
            amax = [v["a_max"] for v in vals]
            steps = [lin["students"][k]["final_steps"] for k in keys]
            a2[f"{name}_s{sd}"] = dict(
                keys=keys, folds=folds, steps=steps, a_max=amax,
                a0=[v["a0"] for v in vals], kl400=[v["final_off_kl"] for v in vals],
                rho_folds=_spearman(folds, amax), rho_steps=_spearman(steps, amax),
                monotone=bool(all(amax[i] <= amax[i + 1] + 1e-12 for i in range(len(amax) - 1))))
    # ⚠️ EXACT lineage match, not a prefix: `"rev_lineage_self2_s1".startswith("rev_lineage_s")` is
    # TRUE, so the obvious filter silently folded the ZERO-CONTENT self-fold ladder into A2 and
    # scored the prediction FAIL when the ladder it was registered on PASSES. The self2 arm is a
    # deliberate contrast (§4.4), not a member of the registered ladder.
    _rev = [v for k, v in a2.items() if re.fullmatch(r"rev_lineage_s\d+", k)]
    a2_pass = (None if not _rev else all(v["monotone"] for v in _rev))

    # ---- item 4: absorption gained PER STEP, in a lineage with a fold vs one without ----
    # Both children continue the SAME parent (`root`) for the SAME 3,078,768 steps, so the
    # denominator is identical and the ratio is a difference in numerators. `gen17` is an
    # ancestry-free LEVEL anchor at one age, not a slope — a slope needs two of its checkpoints
    # and only its final is in this battery.
    slopes = {}
    for sd in seeds:
        base = get("root", PRIMARY_T, PRIMARY_LR, sd)
        if base is None:
            continue
        for k in ("plain0", "ecol0", "fold1"):
            c = get(k, PRIMARY_T, PRIMARY_LR, sd)
            if c is None:
                continue
            n = lin["students"][k]["own_steps"]
            slopes[f"{k}_s{sd}"] = dict(
                student=k, seed=sd, folds=lin["students"][k]["derived_folds"], own_steps=n,
                a_max_root=base["a_max"], a_max=c["a_max"],
                d_a_max=c["a_max"] - base["a_max"],
                d_a_max_per_Mstep=(c["a_max"] - base["a_max"]) / (n / 1e6) if n else None,
                d_a0=c["a0"] - base["a0"])
        g = get("gen17", PRIMARY_T, PRIMARY_LR, sd)
        if g is not None:
            slopes[f"gen17_anchor_s{sd}"] = dict(
                student="gen17", seed=sd, folds=0, own_steps=lin["students"]["gen17"]["final_steps"],
                a_max_root=base["a_max"], a_max=g["a_max"], d_a_max=g["a_max"] - base["a_max"],
                d_a_max_per_Mstep=None, d_a0=g["a0"] - base["a0"],
                note="ANCESTRY-FREE level anchor, not a slope — shares no weights with root")

    # ---- M1: is fold count confounded with age in EVERY supported cell? ----
    matched = lin["m1_matched_step_fold_contrasts"]
    probed_matched = [b for b in matched
                      if len({lin["students"][k]["derived_folds"]
                              for k in STUDENTS
                              if lin["students"][k]["final_steps"] == b["total_steps"]}) > 1]
    m1 = dict(
        statement="prior-fold count is CONFOUNDED with age in every cell the archive supports",
        matched_step_contrasts_in_archive=matched,
        matched_step_contrasts_actually_probed=probed_matched,
        passed=bool(not probed_matched),
        note=("M1 FAILS when a matched-total-step fold-count contrast exists AND was measured; "
              "the archive is searched over non-exploiter runs only"))

    # ---- M2: post-fold consolidation vs untaught outcome ----
    con = lin["consolidation"]
    xs = [c["steps_under_distill"] for c in con]
    ys = [c["untaught_pp"] for c in con]
    rho = _spearman(xs, ys) if all(x is not None for x in xs) else None
    # Companion, NOT a second prediction: the same three folds ranked by (a)'s variable instead.
    # It is reported beside rho because the two are mutually confounded across these three points
    # (the two folds with a prior fold in their ancestry are also the two with more steps).
    pf = [c["prior_folds"] for c in con]
    rho_pf = _spearman(pf, ys)
    m2 = dict(
        statement="post-fold consolidation does NOT order with the measured untaught outcome",
        table=con, steps_under_distill=xs, untaught_pp=ys, spearman_rho=rho,
        prior_folds=pf, spearman_rho_prior_folds=rho_pf, n=len(con),
        passed=bool(rho is not None and abs(rho) < 1.0),
        refuse_coefficient=("n=3: an ORDERING is reported, never a fitted coefficient. A perfect "
                            "ordering arises by chance with probability 1/6 under the null."),
        confound=("the distill term is ACTIVE FOR THE WHOLE fold run, so `steps_under_distill` is "
                  "simultaneously the DOSE and the consolidation window -- this axis cannot "
                  "separate them, and the v8 fold additionally differs in lr (7e-5 vs 3e-4), "
                  "coef (1.0 vs 0.176) and absolute age (292M vs ~29-33M)"))

    # The verdict string must not read as a DIRECTION claim when A1 is a MAGNITUDE test. A1 asks
    # whether the effect clears 0.018; failing that is not the same as the effect being absent, and
    # a reader who quotes only this line must not be misled about which was measured.
    n_pos = sum(1 for v in a1.values()
                for d in (v["delta_vs_plain"], v["delta_vs_ecol"]) if d > 0)
    n_cmp = 2 * len(a1)
    verdict_bits = []
    verdict_bits.append("(a) ACCUMULATED FOLDING: " + (
        "INSUFFICIENT CELLS" if a1_pass is None else
        f"A1 PASSES its registered 0.018 bar ({n_pos}/{n_cmp} comparisons positive)" if a1_pass else
        f"A1 FAILS its registered 0.018 MAGNITUDE bar, but the DIRECTION is "
        f"{n_pos}/{n_cmp} positive at matched age"))
    verdict_bits.append("(b) POST-FOLD CONSOLIDATION: " + (
        "does not order with outcome" if m2["passed"] else
        f"ORDERS PERFECTLY with outcome (rho={rho}), n=3, dose-confounded"))

    out = dict(
        generated="folding_history_probe.py aggregate",
        cells=summ, curves={k: v["curve"] for k, v in cells.items()},
        lineage=dict(students=lin["students"], consolidation=lin["consolidation"],
                     slice_exposure=lin["slice_exposure"],
                     roster_mismatches=lin["roster_mismatches"],
                     matched_step_fold_contrasts=lin["m1_matched_step_fold_contrasts"],
                     unresolved_lineages=lin["unresolved_lineages"]),
        predictions=dict(M1=m1, M2=m2,
                         A1=dict(statement="at 28,067,760 steps the 1-fold student's a_max exceeds "
                                           "BOTH 0-fold arms by > 0.018 on both seeds",
                                 per_seed=a1, passed=a1_pass),
                         A2=dict(statement="a_max monotone non-decreasing in fold count",
                                 per_lineage=a2, passed=a2_pass)),
        absorption_per_step=slopes,
        verdict="; ".join(verdict_bits),
        meta={k: {kk: v[kk] for kk in ("student", "student_run", "teacher", "teacher_set",
                                       "probe_seed", "lr", "batch", "n_steps", "n_on_train",
                                       "n_on_held", "n_off", "dropped_kwargs", "wall_s")}
              for k, v in cells.items()},
        state_provenance=json.load(open(os.path.join(HERE, "fh_state_provenance.json"))))
    json.dump(out, open(os.path.join(HERE, "folding_history_consolidation_2026-08-31.json"), "w"),
              indent=2)

    print(f"\n{'cell':26s} {'stu':10s} {'T':2s} {'lr':>7s} {'sd':>2s} {'a0':>6s} {'amax':>6s} "
          f"{'gain':>7s} {'KL@400':>7s} {'agr':>6s} {'Vmad':>6s}")
    for k in sorted(summ, key=lambda k: (summ[k]["self_target"], summ[k]["teacher_set"],
                                         -summ[k]["lr"], summ[k]["student_key"], summ[k]["seed"])):
        s = summ[k]
        print(f"{k:26s} {s['student_key']:10s} {s['teacher_set']:2s} {s['lr']:7.0e} {s['seed']:2d} "
              f"{s['a0']:6.3f} {s['a_max']:6.3f} {s['gain_max']:+7.3f} {s['final_off_kl']:7.3f} "
              f"{s['final_off_agree']:6.3f} {s['final_off_value_mad']:6.2f}")
    print()
    print(json.dumps(dict(A1=a1, A1_passed=a1_pass, A2=a2, A2_passed=a2_pass,
                          M1_passed=m1["passed"], M2_rho=rho, M2_passed=m2["passed"],
                          verdict=out["verdict"]), indent=2))


def report():
    """Emit the report's markdown tables FROM the aggregate, so no number is ever retyped."""
    A = json.load(open(os.path.join(HERE, "folding_history_consolidation_2026-08-31.json")))
    summ, lin = A["cells"], A["lineage"]

    def f(v, d=3):
        return "—" if v is None else f"{v:.{d}f}"

    print("### Lineage — the student roster\n")
    print("| student | run | arm | folds in ancestry | total steps | own steps | chain |")
    print("|---|---|---|---|---|---|---|")
    for k, s in lin["students"].items():
        own = "— (root)" if s["own_steps"] is None else f"{s['own_steps']:,}"
        print(f"| `{k}` | `{s['run']}` | {s['arm']} | **{s['derived_folds']}** | "
              f"{s['final_steps']:,} | {own} | "
              f"{' → '.join(x.replace('ai_v9_', '') for x in (s['chain'] or []))} |")

    if lin.get("slice_exposure"):
        print("\n### Slice exposure — has this student already been TAUGHT the probe teacher's teams?\n")
        print("| student | folds | teams taught (content) | teams in fold ecology | "
              "teacher `f` slice taught | teacher `a` slice taught |")
        print("|---|---|---|---|---|---|")
        for k, s in lin["students"].items():
            e = lin["slice_exposure"][k]

            def _x(tk):
                p = e["per_teacher"][tk]
                return ("**YES**" if p["slice_taught"] else
                        "ecology only" if p["slice_in_ecology"] else "no")
            print(f"| `{k}` | {s['derived_folds']} | {e['n_taught_teams']} | "
                  f"{e['n_ecology_teams']} | {_x('f')} | {_x('a')} |")

    print("\n### Absorption cells\n")
    print("| cell | student | folds | steps | T | lr | seed | a0 | a_max | gain | KL@400 | "
          "off-agree@400 | \\|dV\\|@400 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(summ, key=lambda k: (summ[k]["self_target"], summ[k]["teacher_set"],
                                         -summ[k]["lr"], summ[k]["arm"],
                                         summ[k]["student_key"], summ[k]["seed"])):
        s = summ[k]
        st = lin["students"][s["student_key"]]
        print(f"| `{k}` | `{s['student_key']}` | {st['derived_folds']} | {st['final_steps']/1e6:.2f}M | "
              f"{s['teacher_set']} | {s['lr']:.0e} | {s['seed']} | {s['a0']:.3f} | "
              f"**{s['a_max']:.3f}** | {s['gain_max']:+.3f} | {s['final_off_kl']:.3f} | "
              f"{s['final_off_agree']:.3f} | {s['final_off_value_mad']:.2f} |")

    print("\n### A1 — the matched-age cell (28,067,760 steps, same parent, same era)\n")
    print("| seed | a_max plain0 (0 folds) | a_max ecol0 (0 folds, 2nd replicate) | "
          "a_max fold1 (1 fold) | Δ vs plain0 | Δ vs ecol0 | run-to-run gap | "
          "margin over it | > registered 0.018 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(A["predictions"]["A1"]["per_seed"].items()):
        print(f"| {v['seed']} | {v['amax_plain0']:.3f} | {v['amax_ecol0']:.3f} | "
              f"**{v['amax_fold1']:.3f}** | {v['delta_vs_plain']:+.3f} | {v['delta_vs_ecol']:+.3f} | "
              f"{v['run_to_run_gap']:.3f} | {v['margin_over_run_to_run']:+.3f} | "
              f"{'**YES**' if v['exceeds_noise'] else 'NO'} |")

    print("\n### A2 — the fold-count ladder (confounded with age above 1 fold)\n")
    print("| lineage × seed | students | folds | steps (M) | a_max | ρ(folds) | ρ(steps) | monotone |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in sorted(A["predictions"]["A2"]["per_lineage"].items()):
        print(f"| `{k}` | {' / '.join(v['keys'])} | {v['folds']} | "
              f"{[round(s/1e6, 2) for s in v['steps']]} | {[round(x, 3) for x in v['a_max']]} | "
              f"{f(v['rho_folds'], 2)} | {f(v['rho_steps'], 2)} | "
              f"{'YES' if v['monotone'] else 'NO'} |")

    if A.get("absorption_per_step"):
        print("\n### Absorption gained per step of continuation (mission item 4)\n")
        print("| arm | folds | own steps | a_max(root) | a_max | Δa_max | Δa_max per 1M steps | Δa0 |")
        print("|---|---|---|---|---|---|---|---|")
        for k, v in sorted(A["absorption_per_step"].items()):
            print(f"| `{k}` | {v['folds']} | {v['own_steps']:,} | {v['a_max_root']:.3f} | "
                  f"{v['a_max']:.3f} | {v['d_a_max']:+.3f} | "
                  f"{f(v['d_a_max_per_Mstep'])} | {v['d_a0']:+.3f} |")

    print("\n### Post-fold consolidation (hypothesis b)\n")
    print("| fold | run | parent | steps under the distill term | coef | lr | teachers | "
          "prior folds | arch signature | untaught outcome | CI | z |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in lin["consolidation"]:
        print(f"| **{c['label']}** | `{c['fold_run']}` | `{c['parent']}` | "
              f"**{c['steps_under_distill']:,}** | {c['distill_coef']} | {c['lr']} | "
              f"{c['n_teachers']} | {c['prior_folds']} | `{c['arch_signature']}` | "
              f"**{c['untaught_pp']:+.2f}pp** | "
              f"[{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] | {c['z']:+.2f} |")
    M2 = A["predictions"]["M2"]
    print(f"\nSpearman ρ(steps under the distill term, untaught outcome) = "
          f"**{f(M2['spearman_rho'], 2)}**, n={M2['n']}.  "
          f"Companion on (a)'s variable: ρ(prior folds, untaught outcome) = "
          f"**{f(M2['spearman_rho_prior_folds'], 2)}**.")

    print(f"\n**VERDICT** — {A['verdict']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "aggregate"
    if cmd == "lineage":
        lineage()
    elif cmd == "build-states":
        build_states()
    elif cmd == "probe":
        probe(sys.argv[2:])
    elif cmd == "aggregate":
        aggregate()
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
