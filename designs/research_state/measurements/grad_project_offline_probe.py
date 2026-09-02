"""GRAD-PROJECT OFFLINE PROBE — `distill_grad_project`'s projection measured on the REAL rev-4
fold ingredients, at production-shaped gradients, in the ADMITTED licensing-probe instrument.

WHY. `distill_grad_project.py` landed 2026-09-01 (`2e99e6fe`) with a smoke that read
`proj_removed_frac` ~= 0.75-0.89 on a `--debug` toy config (`--n-steps 512 --batch-size 128`), and
that smoke **could not tell "the projection removed the leak" from "the projection removed the
teaching"** — it has no absorption meter at all. This probe supplies the missing meter: the same
micro-instrument the LR licensing verdict was written on (`lr_licensing_probe.py`, 2026-08-31),
with the projection wired into the Adam step, so absorption and collateral are read on the same
curve.

REGISTERED QUESTIONS (see the .md's pre-registration block; nothing here was tuned after numbers):
  Q1  what fraction of the distillation gradient's energy lies in the span of the off-slice
      `grad log pi(a*|s)` directions (`removed_frac`), vs `m` (8/16/32/64) and vs training step?
  Q2  with the projection applied every step, what is the absorption-vs-collateral Pareto curve
      against the unprojected run at the same lr, and at MATCHED absorption is collateral lower?
  Q3  does the projection reduce absorption by LESS than it reduces collateral (separable) or by
      about the same (not separable at first order — the method's ceiling)?

REGISTERED PREDICTIONS:
  P1  `removed_frac` at m=16 on production gradients is BELOW 0.5 (the smoke's 0.80 was a
      toy-config artefact).
  P2  at matched absorption 0.70, projected collateral is at least 30% LOWER than unprojected.
  P3  the absorption ceiling under projection is within 0.05 of unprojected.

INSTRUMENT. `lr_licensing_probe` is IMPORTED, never edited: its `load_policy`, `masked_logits`,
`eval_probs`, `eval_values`, `kl_rows`, `eval_points`, `crossing`, its student/teacher identities
and its committed `lr_states_{a,b,c}.npz` / `lr_teacher_targets_{a,b,c}.npz` caches are used
verbatim. The projection uses `agents.training.instrumented_ppo.distill_grad_project`'s own pure
functions (`behaviour_constraints`, `orthonormalize`, `project_out`, `flatten_grads`) — the maths
is NOT re-implemented here, and that module needed no change: those four are already pure
module-level functions with no dependence on the PPO object.

THE ONE DELIBERATE DEVIATION FROM THE LICENSING PROBE, and it is required by the question. The
1,500-row OFF-SLICE pool is SPLIT, by a fixed permutation shared by every arm, into

    CONSTRAINT pool  500 rows  — the projection samples its `m` constraint states from here
    EVALUATION pool 1,000 rows — collateral (KL / agreement / |dV|) is measured only here

so no state ever both constrains the update and scores it. The split is at the STATE level, not the
team level: teams appear in both pools, which is what PRODUCTION does (the live projector draws its
constraint rows from the same off-slice distribution the collateral meter reads), so a team-level
split would measure a different operator than the one that shipped. Collateral is therefore read on
1,000 rows here against 1,500 in the licensing probe; every arm shares the identical pool, so the
comparison is matched, but the absolute KL is NOT numerically interchangeable with that record's.

MODES
  none        the unprojected control — byte-identical to `lr_licensing_probe`'s training loop
              except for the smaller collateral evaluation pool.
  proj:M      the projection applied to EVERY step at `m = M`; `removed_frac` / `rank` logged per
              step.
  monitor:    unprojected TRAINING, but at each eval point `removed_frac` is computed for every
              `m` in {8,16,32,64} on that step's real gradient and thrown away. This is the Q1
              table: `removed_frac` vs m along the trajectory the fold actually takes.

Run (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src):

    python grad_project_offline_probe.py probe <cell> <tset> <seed> <steps> <lr> <mode>
    python grad_project_offline_probe.py aggregate

`<tset>` is `a`/`b`/`c`. `<mode>` is `none`, `proj:16`, or `monitor`. There is no control-token
glob hazard here (no `*` form); the zero-content control is out of scope — this probe compares
PROJECTED against UNPROJECTED at fixed content, not content against no-content.
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
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lr_licensing_probe as LP                                    # noqa: E402  the instrument
from agents.training.instrumented_ppo.distill_grad_project import (  # noqa: E402  the operator
    behaviour_constraints, flatten_grads, orthonormalize, project_out,
)

torch.set_num_threads(int(os.environ.get("PROBE_TORCH_THREADS", "1")))

RESULTS = os.path.join(HERE, "gp_results")
INPUTS = os.path.join(HERE, "grad_project_offline_2026-09-01_inputs")

#: The off-slice split. FIXED — never the cell seed — so every arm constrains and scores on the
#: identical two pools and an arm difference cannot be a split difference.
OFF_SPLIT_SEED = 20260901
N_CONSTRAINT = 500

#: `monitor` mode's m ladder.
MONITOR_MS = (8, 16, 32, 64)

BATCH = LP.BATCH          # 256, the admitted instrument's batch


def split_off(off_o, off_m):
    """The fixed CONSTRAINT / EVALUATION split of the licensing probe's 1,500-row off pool."""
    n = len(off_o)
    perm = np.random.default_rng(OFF_SPLIT_SEED).permutation(n)
    c, e = perm[:N_CONSTRAINT], perm[N_CONSTRAINT:]
    return (off_o[c], off_m[c]), (off_o[e], off_m[e])


def removal_for(policy, params, g, con_o_t, con_m_t, m, gen):
    """`(removal_vector, rank, removed_frac)` for one step at constraint width `m`.

    Every line of maths here is `distill_grad_project`'s; this function only chooses the rows and
    reports the diagnostics, exactly as `DistillGradProjector.before_backward` does.
    """
    n = len(con_o_t)
    rows = torch.as_tensor(gen.choice(n, min(m, n), replace=False).astype(np.int64))
    cons = behaviour_constraints(policy, {"observation": con_o_t, "action_mask": con_m_t},
                                 con_m_t, rows, params)
    basis = orthonormalize(cons)
    del cons
    removal = project_out(g, basis)
    g_sq = float(g.pow(2).sum())
    frac = float(removal.pow(2).sum()) / g_sq if g_sq > 0 else 0.0
    rank = len(basis)
    del basis
    return removal, rank, frac


def apply_removal(params, removal):
    """`.grad -= removal`, materialising a `None` grad the removal actually touches."""
    off = 0
    with torch.no_grad():
        for p in params:
            k = p.numel()
            chunk = removal[off:off + k].view_as(p)
            off += k
            if p.grad is None:
                if float(chunk.abs().max()) == 0.0:
                    continue
                p.grad = torch.zeros_like(p)
            p.grad.sub_(chunk)


# ============================================================================
# ONE PROBE CELL
# ============================================================================

def probe(argv):
    cell, tset, seed, nsteps, LR, mode = (argv[0], argv[1], int(argv[2]),
                                          int(argv[3]), float(argv[4]), argv[5])
    proj_m = None
    if mode.startswith("proj:"):
        proj_m = int(mode.split(":", 1)[1])
    elif mode not in ("none", "monitor"):
        raise SystemExit(f"unknown mode {mode!r}; expected none | proj:M | monitor")
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"{cell}.json")
    t_start = time.time()

    S = np.load(os.path.join(HERE, f"lr_states_{tset}.npz"))
    on_tr_o, on_tr_m = S["on_train_obs"], S["on_train_mask"]
    on_he_o, on_he_m = S["on_held_obs"], S["on_held_mask"]
    (con_o, con_m), (ev_o, ev_m) = split_off(S["off_obs"], S["off_mask"])

    tcache = os.path.join(HERE, f"lr_teacher_targets_{tset}.npz")
    if not os.path.exists(tcache):
        raise SystemExit(f"missing committed teacher-target cache {tcache} — run "
                         "`python lr_licensing_probe.py build-states` first")
    T = np.load(tcache)
    tgt_tr, tgt_he, tprob_he = T["tgt_train"], T["tgt_held"], T["tprob_held"]

    m_model, dropped = LP.load_policy(LP.STUDENT)
    policy = m_model.policy
    if dropped:
        raise SystemExit(f"REFUSED: {LP.STUDENT} dropped kwargs {dropped} — not a faithful rebuild")
    params = [p for p in policy.parameters() if p.requires_grad]

    # reference = the student BEFORE any distillation, on the EVALUATION half of the off pool
    ref_lp = LP.eval_probs(policy, ev_o, ev_m)
    ref_arg = ref_lp.argmax(1)
    ref_val = LP.eval_values(policy, ev_o, ev_m)

    tr_t = torch.as_tensor(on_tr_o); trm_t = torch.as_tensor(on_tr_m)
    con_o_t = torch.as_tensor(con_o); con_m_t = torch.as_tensor(con_m)
    tgt_t = torch.as_tensor(tgt_tr.astype(np.int64))
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    rng = np.random.default_rng(seed)               # SAME batch sequence across cells at a seed
    #: the projector's own stream, never the batch stream — a projected and an unprojected arm at
    #: one seed therefore see the IDENTICAL minibatch sequence.
    pgen = np.random.default_rng(900000 + seed)
    pts = set(LP.eval_points(nsteps))
    curve, losses, proj_log, monitor_log = [], [], [], []

    def snapshot(step):
        lp_he = LP.eval_probs(policy, on_he_o, on_he_m)
        lp_tr = LP.eval_probs(policy, on_tr_o[:1200], on_tr_m[:1200])
        lp_off = LP.eval_probs(policy, ev_o, ev_m)
        val_off = LP.eval_values(policy, ev_o, ev_m)
        kl = LP.kl_rows(lp_off, ref_lp, ev_m)
        rec = dict(
            step=step,
            on_held_agree=float((lp_he.argmax(1) == tgt_he).mean()),
            on_train_agree=float((lp_tr.argmax(1) == tgt_tr[:1200]).mean()),
            on_held_teacher_ce=float(-np.take_along_axis(lp_he, tgt_he[:, None], 1).mean()),
            on_held_tv=float(0.5 * np.abs(np.exp(lp_he) - tprob_he).sum(1).mean()),
            off_kl=float(kl.mean()), off_kl_median=float(np.median(kl)),
            off_agree=float((lp_off.argmax(1) == ref_arg).mean()),
            off_value_mad=float(np.abs(val_off - ref_val).mean()),
            off_value_corr=float(np.corrcoef(val_off, ref_val)[0, 1]),
            elapsed_s=round(time.time() - t_start, 1),
        )
        curve.append(rec)
        print(f"  step {step:5d}  on_held {rec['on_held_agree']:.4f}  "
              f"offKL {rec['off_kl']:.4f}  offAgree {rec['off_agree']:.4f}  "
              f"Vmad {rec['off_value_mad']:.4f}  [{rec['elapsed_s']}s]", flush=True)

    snapshot(0)
    for step in range(1, nsteps + 1):
        idx = rng.choice(len(on_tr_o), BATCH, replace=False)
        it = torch.as_tensor(idx)
        ml = LP.masked_logits(policy, tr_t[it], trm_t[it])
        loss = torch.nn.functional.cross_entropy(ml, tgt_t[it])
        opt.zero_grad(); loss.backward()

        if proj_m is not None:
            g = flatten_grads([p.grad for p in params], params)
            t0 = time.perf_counter()
            removal, rank, frac = removal_for(policy, params, g, con_o_t, con_m_t, proj_m, pgen)
            apply_removal(params, removal)
            del removal, g
            proj_log.append(dict(step=step, m=proj_m, rank=rank, removed_frac=frac,
                                 ms=round(1000 * (time.perf_counter() - t0), 1)))
        elif mode == "monitor" and step in pts:
            g = flatten_grads([p.grad for p in params], params)
            row = dict(step=step, g_norm=float(g.norm()))
            for mm in MONITOR_MS:
                _rem, rank, frac = removal_for(policy, params, g, con_o_t, con_m_t, mm, pgen)
                del _rem
                row[f"removed_frac_m{mm}"] = frac
                row[f"rank_m{mm}"] = rank
            monitor_log.append(row)
            print(f"    [monitor {step:5d}] " + "  ".join(
                f"m{mm}={row[f'removed_frac_m{mm}']:.4f}" for mm in MONITOR_MS), flush=True)
            del g

        opt.step()
        losses.append(float(loss.detach()))
        if step in pts:
            snapshot(step)

    json.dump(dict(cell=cell, student=LP.STUDENT, teacher_set=tset,
                   teacher=LP.TEACHERS[tset], mode=mode, proj_m=proj_m,
                   probe_seed=seed, n_steps=nsteps, lr=LR, batch=BATCH,
                   off_split=dict(seed=OFF_SPLIT_SEED, n_constraint=int(len(con_o)),
                                  n_eval=int(len(ev_o)), level="state"),
                   n_on_train=int(len(on_tr_o)), n_on_held=int(len(on_he_o)),
                   n_params=int(sum(p.numel() for p in params)),
                   dropped_kwargs=list(dropped), curve=curve,
                   proj_log=proj_log, monitor_log=monitor_log,
                   loss_first10=losses[:10], loss_last10=losses[-10:],
                   wall_s=round(time.time() - t_start, 1)),
              open(out_path, "w"), indent=2)
    print(f"[done] {cell} -> {out_path}  ({time.time() - t_start:.0f}s)")


# ============================================================================
# AGGREGATION + REGISTERED-PREDICTION SCORING
# ============================================================================

ABS_LEVELS = (0.65, 0.70, 0.72)
GAIN_LEVELS = (0.03, 0.05, 0.10)

#: 🚨 P1-P3 were REGISTERED on the lr-1e-4 arms ("12 cells ... all at lr 1e-4"). The lr-3e-4
#: robustness cells and the m=8/32 sweep cells are reported in their own sections and must NEVER be
#: pooled into a scored prediction — a pre-registration that silently grows its own sample when a
#: side-arm lands is not a pre-registration. Scoring is therefore gated on BOTH `m == 16` and this
#: lr; the first draft gated only on m and quietly scored a 7th arm at the wrong step size.
SCORED_LR = 1e-4


def _degenerate(cross):
    """A crossing already satisfied at STEP 0 is not a measurement of what the level COST.

    `LP.crossing` returns the step-0 record when the curve already sits above the level, and that
    record's `off_kl` is 0.0 by construction (collateral is measured against the step-0 policy
    itself). Reading it as "this arm reached absorption A at zero collateral" would be a
    -100%-shaped artefact, and teacher `b`'s a0 = 0.681 makes it real at the 0.65 level. Flagged
    here so every consumer prints DEGENERATE instead.
    """
    return bool(cross is not None and cross.get("step") == 0)


def summarize(d):
    c = d["curve"]
    a = np.array([x["on_held_agree"] for x in c], float)
    s = dict(cell=d["cell"], teacher_set=d["teacher_set"], mode=d["mode"],
             proj_m=d.get("proj_m"), seed=d["probe_seed"], lr=d["lr"],
             n_steps=d["n_steps"],
             a0=float(a[0]), a_max=float(a.max()), gain_max=float(a.max() - a[0]),
             final_off_kl=float(c[-1]["off_kl"]),
             final_off_agree=float(c[-1]["off_agree"]),
             final_off_value_mad=float(c[-1]["off_value_mad"]),
             wall_s=d.get("wall_s"))
    for A in ABS_LEVELS:
        s[f"idx_abs_{A:.2f}"] = LP.crossing(c, A, False)
    for g in GAIN_LEVELS:
        s[f"idx_gain_{g:.2f}"] = LP.crossing(c, g, True)
    pl = d.get("proj_log") or []
    if pl:
        fr = np.array([r["removed_frac"] for r in pl], float)
        rk = np.array([r["rank"] for r in pl], float)
        n = len(fr)
        s["proj"] = dict(
            m=d.get("proj_m"), n_steps_logged=n,
            removed_frac_mean=float(fr.mean()),
            removed_frac_first32=float(fr[:32].mean()),
            removed_frac_last32=float(fr[-32:].mean()),
            removed_frac_min=float(fr.min()), removed_frac_max=float(fr.max()),
            rank_mean=float(rk.mean()),
            ms_mean=float(np.mean([r["ms"] for r in pl])),
        )
    ml = d.get("monitor_log") or []
    if ml:
        s["monitor"] = {f"m{mm}": dict(
            mean=float(np.mean([r[f"removed_frac_m{mm}"] for r in ml])),
            early=float(np.mean([r[f"removed_frac_m{mm}"] for r in ml if r["step"] <= 32])),
            late=float(np.mean([r[f"removed_frac_m{mm}"] for r in ml if r["step"] >= 135])),
            rank_mean=float(np.mean([r[f"rank_m{mm}"] for r in ml])),
            by_step={str(r["step"]): r[f"removed_frac_m{mm}"] for r in ml},
        ) for mm in MONITOR_MS}
    return s


def _pct(new, old):
    return None if old in (None, 0) else 100.0 * (old - new) / old


def aggregate():
    cells = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        d = json.load(open(f)); d["cell"] = os.path.basename(f)[:-5]
        cells[d["cell"]] = d
    summ = {k: summarize(v) for k, v in cells.items()}

    def get(tset, seed, lr, mode):
        for s in summ.values():
            if (s["teacher_set"] == tset and s["seed"] == seed
                    and abs(s["lr"] - lr) < 1e-12 and s["mode"] == mode):
                return s
        return None

    arms = sorted({(s["teacher_set"], s["seed"], s["lr"]) for s in summ.values()})

    # ---- Q1: removed_frac vs m vs step, from every monitor cell + every projected arm ----
    q1 = {"monitor": {}, "projected_arms": {}}
    for mm in MONITOR_MS:
        rows = [s["monitor"][f"m{mm}"] for s in summ.values() if "monitor" in s]
        if rows:
            q1["monitor"][f"m{mm}"] = dict(
                n_cells=len(rows),
                mean=float(np.mean([r["mean"] for r in rows])),
                early=float(np.mean([r["early"] for r in rows])),
                late=float(np.mean([r["late"] for r in rows])),
                per_cell=[round(r["mean"], 4) for r in rows],
                rank_mean=float(np.mean([r["rank_mean"] for r in rows])),
            )
    for k, s in summ.items():
        if "proj" in s:
            q1["projected_arms"][k] = s["proj"]

    # ---- P1 ----
    m16 = q1["monitor"].get("m16")
    p16 = [s["proj"] for s in summ.values()
           if s.get("proj", {}).get("m") == 16 and abs(s["lr"] - SCORED_LR) < 1e-12]
    p1_vals = ([m16["mean"]] if m16 else []) + [p["removed_frac_mean"] for p in p16]
    P1 = dict(
        statement="removed_frac at m=16 on production gradients is below 0.5",
        monitor_mean=(m16 or {}).get("mean"),
        monitor_per_cell=(m16 or {}).get("per_cell"),
        projected_arm_means=[round(p["removed_frac_mean"], 4) for p in p16],
        all_values=[round(v, 4) for v in p1_vals],
        max_observed=(round(max(p1_vals), 4) if p1_vals else None),
        verdict=("PASS" if p1_vals and max(p1_vals) < 0.5
                 else ("FAIL" if p1_vals else "NOT MEASURED")),
    )

    # ---- P2 / P3 / Q2 / Q3: projected vs unprojected, arm by arm ----
    pareto, p2_rows, p3_rows = [], [], []
    for tset, seed, lr in arms:
        base = get(tset, seed, lr, "none")
        if base is None:
            continue
        for s in summ.values():
            if not (s["teacher_set"] == tset and s["seed"] == seed
                    and abs(s["lr"] - lr) < 1e-12 and s["mode"].startswith("proj:")):
                continue
            row = dict(teacher=tset, seed=seed, lr=lr, m=s["proj_m"],
                       ceiling_unproj=base["a_max"], ceiling_proj=s["a_max"],
                       d_ceiling=s["a_max"] - base["a_max"],
                       gain_unproj=base["gain_max"], gain_proj=s["gain_max"],
                       kl400_unproj=base["final_off_kl"], kl400_proj=s["final_off_kl"],
                       kl400_drop_pct=_pct(s["final_off_kl"], base["final_off_kl"]),
                       vmad_unproj=base["final_off_value_mad"],
                       vmad_proj=s["final_off_value_mad"],
                       removed_frac_mean=s.get("proj", {}).get("removed_frac_mean"))
            for A in ABS_LEVELS:
                cu, cp = base[f"idx_abs_{A:.2f}"], s[f"idx_abs_{A:.2f}"]
                deg = _degenerate(cu) or _degenerate(cp)
                row[f"abs{A:.2f}"] = dict(
                    reached_unproj=cu is not None, reached_proj=cp is not None,
                    degenerate=deg,
                    kl_unproj=(cu or {}).get("off_kl"), kl_proj=(cp or {}).get("off_kl"),
                    drop_pct=(None if (deg or not (cu and cp))
                              else _pct(cp["off_kl"], cu["off_kl"])),
                    step_unproj=(cu or {}).get("step"), step_proj=(cp or {}).get("step"))
            for g in GAIN_LEVELS:
                cu, cp = base[f"idx_gain_{g:.2f}"], s[f"idx_gain_{g:.2f}"]
                deg = _degenerate(cu) or _degenerate(cp)
                row[f"gain{g:.2f}"] = dict(
                    reached_unproj=cu is not None, reached_proj=cp is not None,
                    degenerate=deg,
                    kl_unproj=(cu or {}).get("off_kl"), kl_proj=(cp or {}).get("off_kl"),
                    drop_pct=(None if (deg or not (cu and cp))
                              else _pct(cp["off_kl"], cu["off_kl"])))
            # Q3 separability: relative reduction in ABSORPTION GAIN vs in COLLATERAL
            gu, gp = base["gain_max"], s["gain_max"]
            row["q3"] = dict(
                gain_reduction_pct=_pct(gp, gu),
                collateral_reduction_pct=_pct(s["final_off_kl"], base["final_off_kl"]),
            )
            if row["q3"]["gain_reduction_pct"] is not None:
                row["q3"]["separability_ratio"] = (
                    row["q3"]["collateral_reduction_pct"] / row["q3"]["gain_reduction_pct"]
                    if row["q3"]["gain_reduction_pct"] not in (0.0, None) else None)
            pareto.append(row)
            if s["proj_m"] == 16 and abs(lr - SCORED_LR) < 1e-12:
                d = row["abs0.70"]
                p2_rows.append(dict(
                    arm=f"{tset}_s{seed}", **d,
                    verdict=("DEGENERATE" if d.get("degenerate")
                             else "FAIL(ceiling not reached)" if not d["reached_proj"]
                             else "PASS" if (d["drop_pct"] is not None and d["drop_pct"] >= 30.0)
                             else "FAIL")))
                p3_rows.append(dict(
                    arm=f"{tset}_s{seed}", ceiling_unproj=base["a_max"],
                    ceiling_proj=s["a_max"], d_ceiling=s["a_max"] - base["a_max"],
                    verdict=("PASS" if abs(s["a_max"] - base["a_max"]) <= 0.05 else "FAIL")))

    def _verdict(rows):
        if not rows:
            return "NOT MEASURED"
        return "PASS" if all(r["verdict"] == "PASS" for r in rows) else (
            "PARTIAL" if any(r["verdict"] == "PASS" for r in rows) else "FAIL")

    def _span(vals, label):
        """mean + full RANGE over arms. Two seeds is not enough for a CI and this says so."""
        v = [x for x in vals if x is not None]
        if not v:
            return {"label": label, "n": 0, "note": "NOT MEASURED"}
        return {"label": label, "n": len(v), "mean": float(np.mean(v)),
                "min": float(np.min(v)), "max": float(np.max(v)),
                "per_arm": [round(x, 4) for x in v],
                "note": "RANGE over arms, not a CI — 2 seeds x <=3 teachers"}

    P2 = dict(statement="at matched absorption 0.70, projected (m=16) collateral is >= 30% lower",
              rows=p2_rows, verdict=_verdict(p2_rows),
              interval=_span([r["drop_pct"] for r in p2_rows], "KL drop % at absorption 0.70"))
    P3 = dict(statement="the absorption ceiling under projection (m=16) is within 0.05 of "
                        "unprojected",
              rows=p3_rows, verdict=_verdict(p3_rows),
              interval=_span([r["d_ceiling"] for r in p3_rows], "delta ceiling (proj - unproj)"))

    m16 = [r for r in pareto if r["m"] == 16]
    Q3 = dict(
        statement="does the projection cut absorption LESS than collateral (separable) or the "
                  "same (first-order ceiling)?",
        gain_reduction=_span([r["q3"]["gain_reduction_pct"] for r in m16],
                             "absorption-gain reduction %, m=16"),
        collateral_reduction=_span([r["q3"]["collateral_reduction_pct"] for r in m16],
                                   "collateral KL@400 reduction %, m=16"),
        ratio=_span([r["q3"].get("separability_ratio") for r in m16],
                    "collateral reduction / gain reduction, m=16"),
    )
    # The separability RATIO is only interpretable when its denominator has a consistent sign.
    # Here it does not: the absorption-gain change straddles zero across arms, so a ratio of
    # "collateral removed per unit teaching removed" divides by something indistinguishable from
    # nothing and its sign flips arm to arm. Say so in the artefact rather than letting a reader
    # quote a 1.34x or a -4.05x as if it measured separability.
    gr = [r["q3"]["gain_reduction_pct"] for r in m16
          if r["q3"]["gain_reduction_pct"] is not None]
    Q3["ratio_interpretable"] = bool(gr and (all(x > 0 for x in gr) or all(x < 0 for x in gr)))
    Q3["note"] = (
        "the ratio column is NOT interpretable on this data: the absorption-gain reduction "
        "straddles zero across arms, so the ratio divides by a quantity indistinguishable from "
        "zero and flips sign arm to arm. Read the two reduction columns separately."
        if not Q3["ratio_interpretable"] else
        "the gain reduction has a consistent sign across arms, so the ratio is interpretable.")

    out = dict(
        title="GRAD-PROJECT OFFLINE — distill_grad_project's projection on the rev-4 ingredients",
        date="2026-09-01",
        student=LP.STUDENT, teachers=LP.TEACHERS, off_run=LP.OFF_RUN,
        off_split=dict(seed=OFF_SPLIT_SEED, n_constraint=N_CONSTRAINT, n_eval=1500 - N_CONSTRAINT,
                       level="state",
                       note="constraint and evaluation pools are DISJOINT SETS OF STATES drawn "
                            "from the licensing probe's 1500-row off pool by a fixed permutation "
                            "shared by every arm; teams overlap between the pools, matching "
                            "production, where the live projector's constraint rows come from the "
                            "same off-slice distribution the collateral meter reads"),
        cells={k: summ[k] for k in sorted(summ)},
        Q1_removed_frac=q1,
        Q2_Q3_pareto=pareto,
        Q3_separability=Q3,
        P1=P1, P2=P2, P3=P3,
    )
    p = os.path.join(HERE, "grad_project_offline_2026-09-01.json")
    json.dump(out, open(p, "w"), indent=2)
    print(json.dumps(dict(P1=P1["verdict"], P2=P2["verdict"], P3=P3["verdict"],
                          n_cells=len(summ)), indent=2))
    print(f"[done] -> {p}")


def _f(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def _pctf(x):
    return "—" if x is None else f"{x:+.1f}%"


def report():
    """Emit the record's markdown tables from the aggregate JSON.

    A MISSING cell is never interpolated: an arm that was not run prints an explicit gap row, and
    an absorption level a projected arm never reached prints `NOT REACHED`, never a blank.
    """
    p = os.path.join(HERE, "grad_project_offline_2026-09-01.json")
    d = json.load(open(p))
    out = []

    out.append("### removed_frac vs m (monitor cells — UNPROJECTED trajectory)\n")
    out.append("| m | mean | early (step ≤ 32) | late (step ≥ 135) | rank / m | per-cell means |")
    out.append("|---|---|---|---|---|---|")
    for mm in MONITOR_MS:
        r = d["Q1_removed_frac"]["monitor"].get(f"m{mm}")
        if r is None:
            out.append(f"| {mm} | — | — | — | — | NOT MEASURED |")
            continue
        out.append(f"| {mm} | **{_f(r['mean'])}** | {_f(r['early'])} | {_f(r['late'])} | "
                   f"{r['rank_mean']:.1f} / {mm} | {r['per_cell']} |")

    out.append("\n### removed_frac on the PROJECTED arms (its own trajectory, every step)\n")
    out.append("| arm | m | mean | first 32 steps | last 32 steps | min | max | rank | ms/step |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for k, r in sorted(d["Q1_removed_frac"]["projected_arms"].items()):
        out.append(f"| `{k}` | {r['m']} | **{_f(r['removed_frac_mean'])}** | "
                   f"{_f(r['removed_frac_first32'])} | {_f(r['removed_frac_last32'])} | "
                   f"{_f(r['removed_frac_min'])} | {_f(r['removed_frac_max'])} | "
                   f"{r['rank_mean']:.1f} | {r['ms_mean']:.0f} |")

    out.append("\n### Pareto — projected vs unprojected, per arm\n")
    out.append("| teacher | seed | lr | m | ceil unproj | ceil proj | Δceil | gain unproj | "
               "gain proj | KL@400 unproj | KL@400 proj | KL@400 drop | "
               "KL@abs.70 unproj | KL@abs.70 proj | drop |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in d["Q2_Q3_pareto"]:
        a = r["abs0.70"]
        drop70 = ("NOT REACHED" if not a["reached_proj"]
                  else ("DEGENERATE" if a.get("degenerate") else _pctf(a["drop_pct"])))
        out.append(
            f"| {r['teacher']} | {r['seed']} | {r['lr']:g} | {r['m']} | "
            f"{_f(r['ceiling_unproj'], 3)} | {_f(r['ceiling_proj'], 3)} | "
            f"{r['d_ceiling']:+.3f} | {_f(r['gain_unproj'], 3)} | {_f(r['gain_proj'], 3)} | "
            f"{_f(r['kl400_unproj'], 3)} | {_f(r['kl400_proj'], 3)} | "
            f"{_pctf(r['kl400_drop_pct'])} | "
            f"{_f(a['kl_unproj'], 3)} | {_f(a['kl_proj'], 3)} | {drop70} |")

    out.append("\n### Q3 separability — how much of each the projection removed\n")
    out.append("| teacher | seed | m | absorption-gain reduction | collateral reduction | "
               "ratio (collateral / gain) |")
    out.append("|---|---|---|---|---|---|")
    for r in d["Q2_Q3_pareto"]:
        q = r["q3"]
        ratio = q.get("separability_ratio")
        rs = "—" if ratio is None else f"{ratio:.2f}x"
        out.append(f"| {r['teacher']} | {r['seed']} | {r['m']} | "
                   f"{_pctf(q['gain_reduction_pct'])} | {_pctf(q['collateral_reduction_pct'])} | "
                   f"{rs} |")

    for pid in ("P1", "P2", "P3"):
        pr = d[pid]
        out.append(f"\n### {pid} — {pr['statement']}  →  **{pr['verdict']}**\n")
        if pid == "P1":
            out.append(f"monitor mean {_f(pr['monitor_mean'])} · per-cell {pr['monitor_per_cell']} "
                       f"· projected-arm means {pr['projected_arm_means']} · "
                       f"**max observed {_f(pr['max_observed'])}** (bar: < 0.5)")
        elif pid == "P2":
            out.append("| arm | reached 0.70 unproj | reached 0.70 proj | KL unproj | KL proj | "
                       "drop | verdict |")
            out.append("|---|---|---|---|---|---|---|")
            for r in pr["rows"]:
                out.append(f"| `{r['arm']}` | {r['reached_unproj']} | {r['reached_proj']} | "
                           f"{_f(r['kl_unproj'], 3)} | {_f(r['kl_proj'], 3)} | "
                           f"{_pctf(r['drop_pct'])} | **{r['verdict']}** |")
        else:
            out.append("| arm | ceiling unproj | ceiling proj | Δ | verdict |")
            out.append("|---|---|---|---|---|")
            for r in pr["rows"]:
                out.append(f"| `{r['arm']}` | {_f(r['ceiling_unproj'], 3)} | "
                           f"{_f(r['ceiling_proj'], 3)} | {r['d_ceiling']:+.3f} | "
                           f"**{r['verdict']}** |")
    print("\n".join(out))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "aggregate"
    if cmd == "probe":
        probe(sys.argv[2:])
    elif cmd == "aggregate":
        aggregate()
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
