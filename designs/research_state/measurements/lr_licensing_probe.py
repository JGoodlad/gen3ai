"""LR LICENSING PROBE — the distillability micro-instrument on the ACTUAL rev-4 fold ingredients.

Question (registered, ledger 38fa4eb + ac40230): does lowering the distill-term step size from
3e-4 to 1e-4 Pareto-dominate on the REAL ingredients of the upcoming revolution fold — the actual
fold parent (`ai_v9_59_R2ACTION_0827`, what the live `ai_v9_76_R4ACTION_0830` forked) distilled
toward each of the three rev-4 teachers (`ai_v9_73/74/75_R4S3{a,b,c}_0829`)?

Registered predictions (scored in aggregate, never tuned):
  P1  lr 1e-4 Pareto-dominates lr 3e-4 on every teacher: absorption ceiling not lower
      (>= ceiling@3e-4 − 0.018, the instrument's measured seed noise) AND off-slice collateral
      KL@400 lower.
  P2  The zero-content control (student onto its OWN argmax) at 3e-4 shows collateral comparable
      to the with-content cells (the Adam-overshoot account); at 1e-4 it shrinks by >= 40%.

Instrument is the ADMITTED `distillability_index_probe.py` (2026-08-28) verbatim in its
mechanics: full-policy Adam on masked CE to the teacher argmax, batch 256, 400 steps, 14
log-spaced eval points; ABSORPTION = held-out on-slice top-1 agreement (held out by BATTLE FILE),
COLLATERAL = off-slice masked KL / top-1 agreement / |dV| vs the student's own pre-probe policy.
The step-1 shock is reported as an ordering only (it failed value-level admission).

Run (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src):

    python lr_licensing_probe.py build-states
    python lr_licensing_probe.py probe <cell> <teacher_set> <seed> <steps> <lr>
    python lr_licensing_probe.py aggregate

`<teacher_set>` is `a` / `b` / `c` (the three rev-4 teachers), or `a*` / `b*` / `c*` for the
CONTENT CONTROL on that set's states (targets become the student's own argmax — same optimizer,
same states, zero new behavioural content). The student is always the fold parent; there is no
student argument by design — this probe measures ONE student on its real menu.

QUOTE the control token: bare `a*` is glob-expanded by the shell against the working directory
and the cell dies on argv parsing (it did, four times, on this battery's first run). Write
`probe ctrl_a_3e4_s1 'a*' 1 400 3e-4`.
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

torch.set_num_threads(int(os.environ.get("PROBE_TORCH_THREADS", "1")))

NEG = -1e8
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = "/home/goodlad/dev/gen3ai/models"

# The ACTUAL rev-4 fold ingredients, read from ai_v9_76_R4ACTION_0830/metadata.json
# original_command: --model models/ai_v9_59_R2ACTION_0827/final_model.zip
#                   --distill-teacher models/ai_v9_73:*;models/ai_v9_74:*;models/ai_v9_75:*
# The fold-arm table forks a COMMON base: rev-2 (ai_v9_59), rev-3 (ai_v9_70) and the live rev-4
# (ai_v9_76) fold all record `--model models/ai_v9_59_R2ACTION_0827/final_model.zip`, so the
# arms stay comparable. The student here is therefore what the LIVE rev-4 fold actually forked.
PARENT = "ai_v9_59_R2ACTION_0827"
# $GEN3_LR_PROBE_STUDENT overrides it — used to re-run the verdict against a DIFFERENT candidate
# parent (e.g. the rev-4 fold OUTPUT) should the revolution fold not fork the common base. The
# per-cell JSON records the resolved path, so a mixed results dir stays self-describing.
STUDENT = os.environ.get("GEN3_LR_PROBE_STUDENT") or f"{MODELS}/{PARENT}/final_model.zip"
# OFF-slice source: the parent's OWN traces are too narrow to measure collateral breadth
# (measured 2026-08-30: 474 files / 9 distinct teams / 2 eval steps — a fold run's eval traces
# cluster on the fold teams). rev-1 — the student's own lineage ancestor, and the admitted
# instrument's off-source — has 2456 files / 281 teams / 12 eval steps. Collateral is drift from
# the STUDENT's own pre-probe policy on these states, so state provenance needs breadth, not
# the parent's exact on-policy distribution (caveat 3 of the 2026-08-28 instrument applies).
OFF_RUN = "ai_v9_29_rev1_0823"
TEACHER_RUNS = {
    "a": "ai_v9_73_R4S3a_0829",
    "b": "ai_v9_74_R4S3b_0829",
    "c": "ai_v9_75_R4S3c_0829",
}
TEACHERS = {k: f"{MODELS}/{r}/final_model.zip" for k, r in TEACHER_RUNS.items()}
BATCH = 256
RESULTS = os.path.join(HERE, "lr_results")
OBS_DIM = 2501
SEED_NOISE_GAIN = 0.018  # measured seed-to-seed |Δgain@400| bound from the 2026-08-28 battery


def load_policy(ckpt_path: str, device: str = "cpu"):
    """The project's read-only offline load path (same as the prober's ProbeModel.load)."""
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
        ot = torch.as_tensor(obs[i:i + bs])
        mt = torch.as_tensor(mask[i:i + bs])
        ml = masked_logits(policy, ot, mt)
        out.append(torch.log_softmax(ml, 1).double().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def eval_values(policy, obs, mask, bs=256):
    out = []
    for i in range(0, len(obs), bs):
        ot = torch.as_tensor(obs[i:i + bs])
        mt = torch.as_tensor(mask[i:i + bs])
        out.append(policy.predict_values({"observation": ot, "action_mask": mt}).flatten().double().numpy())
    return np.concatenate(out, 0)


def kl_rows(logp_new, logp_ref, mask):
    m = mask.astype(bool)
    p = np.exp(logp_new)
    d = np.where(m, p * (logp_new - logp_ref), 0.0)
    return d.sum(1)


# ============================================================================
# STATE-SET BUILDER
# ============================================================================

def harvest(run):
    recs = []
    for sm in sorted(glob.glob(f"{MODELS}/{run}/eval_traces/*/*/*_summary.json")):
        npz = sm.replace("_summary.json", "_states.npz")
        if not os.path.exists(npz):
            continue
        try:
            s = json.load(open(sm))
        except Exception:
            continue
        team = tuple(sorted(m["species"] for m in s.get("teams", {}).get("ours", [])))
        if len(team) != 6:
            continue
        z = np.load(npz)
        obs, mask = z["obs"], z["action_mask"]
        if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
            continue
        keep = mask.sum(1) >= 2            # a 1-legal-action state carries no policy signal
        if not keep.any():
            continue
        parts = sm.split("/")
        recs.append(dict(file=sm, team=team, step=parts[-3], opp=parts[-2],
                         obs=obs[keep].astype(np.float32), mask=mask[keep].astype(bool)))
    return recs


def pick(recs, n, seed, per_file_cap=None):
    """Deterministic stratified draw: round-robin over source FILES, seeded."""
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
    print(f"[off-source] {OFF_RUN}: {len(p_recs_all)} trace files, "
          f"{sum(len(r['obs']) for r in p_recs_all)} usable rows, "
          f"{len({r['team'] for r in p_recs_all})} teams")
    prov = dict(student=STUDENT, parent_run=PARENT, off_run=OFF_RUN, teachers={},
                off_source_note="parent's own traces measured too narrow (9 teams / 2 eval steps); "
                                "rev-1 lineage traces used instead (281 teams / 12 steps)",
                filters=f"action_mask.sum(1) >= 2; obs dim {OBS_DIM}; ON split by BATTLE FILE (75/25)")

    for i, (key, run) in enumerate(sorted(TEACHER_RUNS.items())):
        t_recs = harvest(run)
        teams = sorted({r["team"] for r in t_recs})
        # The rev-4 teachers are exploiters pinned to a small team menu (the fold spec lists 8
        # per teacher). A generalist-sized team census here would mean we harvested the wrong run.
        assert 1 <= len(teams) <= 12, (run, len(teams))
        print(f"[on:{key}] teacher {run}: {len(t_recs)} files, "
              f"{sum(len(r['obs']) for r in t_recs)} usable rows, {len(teams)} pinned teams")

        rng = np.random.default_rng(20260830 + i)
        fidx = rng.permutation(len(t_recs))
        held_files = set(fidx[:max(1, len(t_recs) // 4)].tolist())
        tr = [r for j, r in enumerate(t_recs) if j not in held_files]
        he = [r for j, r in enumerate(t_recs) if j in held_files]
        on_tr_o, on_tr_m, _ = pick(tr, 3000, 11 + 10 * i)
        on_he_o, on_he_m, _ = pick(he, 1200, 12 + 10 * i)

        # OFF: the parent's own traces, MINUS any state on this teacher's pinned teams.
        p_recs = [r for r in p_recs_all if r["team"] not in set(teams)]
        off_o, off_m, off_prov = pick(p_recs, 1500, 13 + 10 * i, per_file_cap=2)
        off_teams = len({p_recs[j]["team"] for j, _ in off_prov})
        off_steps = sorted({p_recs[j]["step"] for j, _ in off_prov})
        print(f"[off:{key}] {len(p_recs)} files after pin-exclusion; drew {len(off_o)} rows "
              f"from {off_teams} distinct teams across {len(off_steps)} eval steps")

        np.savez_compressed(
            os.path.join(HERE, f"lr_states_{key}.npz"),
            on_train_obs=on_tr_o, on_train_mask=on_tr_m,
            on_held_obs=on_he_o, on_held_mask=on_he_m,
            off_obs=off_o, off_mask=off_m,
        )
        prov["teachers"][key] = dict(
            run=run, pinned_teams=[list(t) for t in teams],
            trace_files=len(t_recs), usable_rows=int(sum(len(r["obs"]) for r in t_recs)),
            train_files=len(tr), held_files=len(he),
            n_on_train=int(len(on_tr_o)), n_on_held=int(len(on_he_o)),
            off=dict(files_after_exclusion=len(p_recs), n=int(len(off_o)),
                     distinct_teams=off_teams, eval_steps=len(off_steps)),
        )
    json.dump(prov, open(os.path.join(HERE, "lr_state_provenance.json"), "w"), indent=2)
    print("[done] wrote lr_states_{a,b,c}.npz + lr_state_provenance.json")


# ============================================================================
# ONE PROBE CELL
# ============================================================================

def eval_points(n):
    pts = {0, n}
    v = 1
    while v <= n:
        pts.add(v); v = int(np.ceil(v * 1.6))
    return sorted(pts)


def probe(argv):
    cell, tset, seed, nsteps, LR = argv[0], argv[1], int(argv[2]), int(argv[3]), float(argv[4])
    out_path = os.path.join(RESULTS, f"{cell}.json")
    os.makedirs(RESULTS, exist_ok=True)
    t_start = time.time()

    self_target = tset.endswith("*")          # content control: targets = student's OWN argmax
    tset = tset.rstrip("*")
    S = np.load(os.path.join(HERE, f"lr_states_{tset}.npz"))
    on_tr_o, on_tr_m = S["on_train_obs"], S["on_train_mask"]
    on_he_o, on_he_m = S["on_held_obs"], S["on_held_mask"]
    off_o, off_m = S["off_obs"], S["off_mask"]

    # ---- teacher targets (computed ONCE per teacher set, cached) ----
    tcache = os.path.join(HERE, f"lr_teacher_targets_{tset}.npz")
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
        np.savez_compressed(tcache, tgt_train=tgt_tr, tgt_held=tgt_he, tprob_held=tprob_he)
        del tm

    # ---- student = the ACTUAL fold parent, always ----
    m, dropped = load_policy(STUDENT)
    policy = m.policy
    if dropped:
        raise SystemExit(f"REFUSED: {STUDENT} dropped kwargs {dropped} — not a faithful rebuild")

    if self_target:
        tgt_tr = eval_probs(policy, on_tr_o, on_tr_m).argmax(1)
        lp0 = eval_probs(policy, on_he_o, on_he_m)
        tgt_he, tprob_he = lp0.argmax(1), np.exp(lp0)

    # ---- reference = the student BEFORE any distillation ----
    ref_off_lp = eval_probs(policy, off_o, off_m)
    ref_off_arg = ref_off_lp.argmax(1)
    ref_off_val = eval_values(policy, off_o, off_m)

    tr_t = torch.as_tensor(on_tr_o); trm_t = torch.as_tensor(on_tr_m)
    tgt_t = torch.as_tensor(tgt_tr.astype(np.int64))
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    rng = np.random.default_rng(seed)          # SAME batch sequence across cells at a given seed
    pts = set(eval_points(nsteps))
    curve, losses = [], []

    def snapshot(step):
        lp_he = eval_probs(policy, on_he_o, on_he_m)
        lp_tr = eval_probs(policy, on_tr_o[:1200], on_tr_m[:1200])
        lp_off = eval_probs(policy, off_o, off_m)
        val_off = eval_values(policy, off_o, off_m)
        kl = kl_rows(lp_off, ref_off_lp, off_m)
        rec = dict(
            step=step,
            on_held_agree=float((lp_he.argmax(1) == tgt_he).mean()),
            on_train_agree=float((lp_tr.argmax(1) == tgt_tr[:1200]).mean()),
            on_held_teacher_ce=float(-np.take_along_axis(lp_he, tgt_he[:, None], 1).mean()),
            on_held_tv=float(0.5 * np.abs(np.exp(lp_he) - tprob_he).sum(1).mean()),
            off_kl=float(kl.mean()), off_kl_median=float(np.median(kl)),
            off_agree=float((lp_off.argmax(1) == ref_off_arg).mean()),
            off_value_mad=float(np.abs(val_off - ref_off_val).mean()),
            off_value_corr=float(np.corrcoef(val_off, ref_off_val)[0, 1]),
            elapsed_s=round(time.time() - t_start, 1),
        )
        curve.append(rec)
        print(f"  step {step:5d}  on_held {rec['on_held_agree']:.4f}  on_train {rec['on_train_agree']:.4f}"
              f"  offKL {rec['off_kl']:.4f}  offAgree {rec['off_agree']:.4f}"
              f"  Vmad {rec['off_value_mad']:.4f}  [{rec['elapsed_s']}s]", flush=True)

    snapshot(0)
    for step in range(1, nsteps + 1):
        idx = rng.choice(len(on_tr_o), BATCH, replace=False)
        it = torch.as_tensor(idx)
        ml = masked_logits(policy, tr_t[it], trm_t[it])
        loss = torch.nn.functional.cross_entropy(ml, tgt_t[it])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        if step in pts:
            snapshot(step)

    json.dump(dict(cell=cell, student=STUDENT, teacher_set=tset,
                   teacher=("SELF(content control)" if self_target else TEACHERS[tset]),
                   self_target=self_target,
                   probe_seed=seed, n_steps=nsteps, lr=LR, batch=BATCH,
                   n_on_train=int(len(on_tr_o)), n_on_held=int(len(on_he_o)), n_off=int(len(off_o)),
                   dropped_kwargs=list(dropped), curve=curve,
                   loss_first10=losses[:10], loss_last10=losses[-10:],
                   wall_s=round(time.time() - t_start, 1)),
              open(out_path, "w"), indent=2)
    print(f"[done] {cell} -> {out_path}  ({time.time()-t_start:.0f}s)")


# ============================================================================
# AGGREGATION + REGISTERED-PREDICTION SCORING
# ============================================================================

def crossing(curve, thresh, use_gain):
    a = np.array([c["on_held_agree"] for c in curve], float)
    a = np.maximum.accumulate(a)
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
    s = dict(cell=d["cell"], student=d["student"], teacher_set=d["teacher_set"],
             self_target=d.get("self_target", False),
             seed=d["probe_seed"], lr=d["lr"], n_steps=d["n_steps"],
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


def _cellkey(tset, lr, seed, self_target):
    return f"{'ctrl_' if self_target else ''}{tset}_lr{lr:g}_s{seed}"


def aggregate():
    cells = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        d = json.load(open(f)); d["cell"] = os.path.basename(f)[:-5]
        cells[d["cell"]] = d
    summ = {k: summarize(v) for k, v in cells.items()}

    # ---- registered-prediction scoring ----
    # Scored on the PRIMARY student only (the parent the live rev-4 fold forked). Cells run
    # against an alternate candidate parent via $GEN3_LR_PROBE_STUDENT are reported separately
    # under `alternate_student_arms` and never pooled into P1/P2.
    PRIMARY = f"{MODELS}/{PARENT}/final_model.zip"

    def get(tset, lr, seed=1, ctrl=False, student=PRIMARY):
        for s in summ.values():
            if (s["teacher_set"] == tset and abs(s["lr"] - lr) < 1e-12
                    and s["seed"] == seed and s["self_target"] == ctrl
                    and s["student"] == student):
                return s
        return None

    seeds = sorted({s["seed"] for s in summ.values()
                    if not s["self_target"] and s["student"] == PRIMARY})
    p1 = {}
    for t in sorted(TEACHER_RUNS):
        for sd in seeds:
            hi, lo = get(t, 3e-4, seed=sd), get(t, 1e-4, seed=sd)
            if hi is None or lo is None:
                continue
            # Ceiling comparison at MATCHED absorption effort, tolerant of the instrument's own
            # seed noise (|d gain@400| <= 0.018 over 22 paired cells, 2026-08-28 §2).
            ceiling_ok = lo["a_max"] >= hi["a_max"] - SEED_NOISE_GAIN
            collat_ok = lo["final_off_kl"] < hi["final_off_kl"]
            # The fairer half: collateral at MATCHED absorption gain (+0.05), not matched steps.
            m_hi, m_lo = hi["idx_gain_0.05"], lo["idx_gain_0.05"]
            matched_ok = (None if (m_hi is None or m_lo is None)
                          else bool(m_lo["off_kl"] < m_hi["off_kl"]))
            p1[f"{t}_s{sd}"] = dict(
                teacher=t, seed=sd,
                ceiling_3e4=hi["a_max"], ceiling_1e4=lo["a_max"],
                kl400_3e4=hi["final_off_kl"], kl400_1e4=lo["final_off_kl"],
                offagree400_3e4=hi["final_off_agree"], offagree400_1e4=lo["final_off_agree"],
                vmad400_3e4=hi["final_off_value_mad"], vmad400_1e4=lo["final_off_value_mad"],
                kl_at_gain05_3e4=None if m_hi is None else m_hi["off_kl"],
                kl_at_gain05_1e4=None if m_lo is None else m_lo["off_kl"],
                ceiling_not_lower=bool(ceiling_ok), collateral_lower=bool(collat_ok),
                collateral_lower_at_matched_gain=matched_ok,
                pareto=bool(ceiling_ok and collat_ok))
    p1_pass = bool(p1) and all(v["pareto"] for v in p1.values())

    # P2 is scored PER SEED, control against the with-content cells at the SAME seed and lr.
    # "comparable" is operationalized as the control carrying >= 60% of the with-content
    # collateral (the 2026-08-28 lineage battery measured 79%); the threshold is declared here
    # rather than chosen after seeing these numbers.
    p2 = {}
    for sd in seeds:
        ctrl_hi, ctrl_lo = get("a", 3e-4, seed=sd, ctrl=True), get("a", 1e-4, seed=sd, ctrl=True)
        content_hi = [c for c in (get(t, 3e-4, seed=sd) for t in sorted(TEACHER_RUNS)) if c]
        if not (ctrl_hi and ctrl_lo and content_hi):
            continue
        mean_content_kl = float(np.mean([s["final_off_kl"] for s in content_hi]))
        ratio = ctrl_hi["final_off_kl"] / mean_content_kl if mean_content_kl > 0 else None
        shrink = (1.0 - ctrl_lo["final_off_kl"] / ctrl_hi["final_off_kl"]
                  if ctrl_hi["final_off_kl"] > 0 else None)
        p2[f"s{sd}"] = dict(
            seed=sd, n_content_cells=len(content_hi),
            ctrl_kl400_3e4=ctrl_hi["final_off_kl"], ctrl_kl400_1e4=ctrl_lo["final_off_kl"],
            mean_content_kl400_3e4=mean_content_kl,
            overshoot_share_3e4=ratio, ctrl_shrink_1e4=shrink,
            comparable_at_3e4=bool(ratio is not None and ratio >= 0.60),
            shrinks_ge_40pct=bool(shrink is not None and shrink >= 0.40))
    p2_pass = bool(p2) and all(v["comparable_at_3e4"] and v["shrinks_ge_40pct"] for v in p2.values())

    # ---- ALTERNATE-STUDENT arms: same comparison, a different candidate fold parent ----
    alt = {}
    for st in sorted({s["student"] for s in summ.values() if s["student"] != PRIMARY}):
        for t in sorted(TEACHER_RUNS):
            for sd in seeds:
                hi, lo = get(t, 3e-4, seed=sd, student=st), get(t, 1e-4, seed=sd, student=st)
                if hi is None or lo is None:
                    continue
                alt[f"{os.path.basename(os.path.dirname(st))}_{t}_s{sd}"] = dict(
                    student=st, teacher=t, seed=sd,
                    ceiling_3e4=hi["a_max"], ceiling_1e4=lo["a_max"],
                    kl400_3e4=hi["final_off_kl"], kl400_1e4=lo["final_off_kl"],
                    offagree400_3e4=hi["final_off_agree"], offagree400_1e4=lo["final_off_agree"],
                    vmad400_3e4=hi["final_off_value_mad"], vmad400_1e4=lo["final_off_value_mad"],
                    pareto=bool(lo["a_max"] >= hi["a_max"] - SEED_NOISE_GAIN
                                and lo["final_off_kl"] < hi["final_off_kl"]))

    # ---- content-minus-overshoot NET per with-content cell (control-matched, same lr/step) ----
    net = {}
    for s in summ.values():
        if s["self_target"] or s["student"] != PRIMARY:
            continue
        ctrl = get("a", s["lr"], seed=s["seed"], ctrl=True)
        if ctrl is None or s["at_400"] is None or ctrl["at_400"] is None:
            continue
        net[s["cell"]] = dict(
            content_gain400=s["at_400"]["gain"],
            kl400=s["at_400"]["off_kl"], ctrl_kl400=ctrl["at_400"]["off_kl"],
            net_kl400=s["at_400"]["off_kl"] - ctrl["at_400"]["off_kl"],
            vmad400=s["at_400"]["off_value_mad"], ctrl_vmad400=ctrl["at_400"]["off_value_mad"],
        )

    n_arms = len(p1)
    verdict = (f"LICENSED: lr 1e-4 Pareto-dominates 3e-4 on all {n_arms} teacher x seed arms "
               f"of the real rev-4 fold menu"
               if p1_pass else
               "NOT LICENSED as registered — see the per-arm table")

    out = dict(cells=summ, curves={k: v["curve"] for k, v in cells.items()},
               predictions=dict(
                   P1=dict(statement="1e-4 Pareto-dominates on the real ingredients "
                                     "(ceiling not lower AND collateral lower, per teacher x seed)",
                           per_arm=p1, passed=bool(p1_pass)),
                   P2=dict(statement="zero-content control at 3e-4 comparable to with-content cells "
                                     "(>= 60% of their collateral); at 1e-4 it shrinks >= 40%",
                           per_seed=p2, passed=bool(p2_pass))),
               net_content_minus_overshoot=net,
               alternate_student_arms=alt,
               primary_student=PRIMARY,
               verdict=verdict,
               meta={k: {kk: v[kk] for kk in ("student", "teacher", "teacher_set", "probe_seed",
                                              "lr", "batch", "n_steps", "n_on_train", "n_on_held",
                                              "n_off", "dropped_kwargs", "wall_s")}
                     for k, v in cells.items()},
               state_provenance=json.load(open(os.path.join(HERE, "lr_state_provenance.json"))))
    json.dump(out, open(os.path.join(HERE, "lr_licensing_probe_2026-08-31.json"), "w"), indent=2)

    def fmt(v, d=3):
        return "MISS" if v is None else f"{v:.{d}f}"

    print(f"{'cell':22s} {'a0':>6s} {'amax':>6s} {'gain':>7s} {'KL@400':>7s} {'agr@400':>8s} "
          f"{'Vmad':>6s} {'KL@d05':>7s}")
    for k in sorted(summ):
        s = summ[k]
        i = s["idx_gain_0.05"]
        print(f"{k:22s} {s['a0']:6.3f} {s['a_max']:6.3f} {s['gain_max']:+7.3f} "
              f"{s['final_off_kl']:7.3f} {s['final_off_agree']:8.3f} "
              f"{s['final_off_value_mad']:6.2f} {fmt(None if i is None else i['off_kl']):>7s}")
    print()
    print(json.dumps(dict(P1=p1, P1_passed=p1_pass, P2=p2, P2_passed=p2_pass, verdict=verdict),
                     indent=2))


def report():
    """Emit the report's markdown tables FROM the aggregate, so no number is ever retyped."""
    A = json.load(open(os.path.join(HERE, "lr_licensing_probe_2026-08-31.json")))
    summ, p1, p2 = A["cells"], A["predictions"]["P1"]["per_arm"], A["predictions"]["P2"]["per_seed"]

    def f(v, d=3):
        return "—" if v is None else f"{v:.{d}f}"

    print("### Per-arm cells\n")
    print("| arm | lr | a0 | a_max | gain | KL@400 | off-agree@400 | \\|dV\\|@400 | KL@gain+0.05 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(summ, key=lambda k: (summ[k]["self_target"], summ[k]["student"],
                                         summ[k]["teacher_set"], summ[k]["seed"], -summ[k]["lr"])):
        s = summ[k]
        i = s["idx_gain_0.05"]
        print(f"| `{k}` | {s['lr']:.0e} | {s['a0']:.3f} | **{s['a_max']:.3f}** | "
              f"{s['gain_max']:+.3f} | **{s['final_off_kl']:.3f}** | {s['final_off_agree']:.3f} | "
              f"{s['final_off_value_mad']:.2f} | {f(None if i is None else i['off_kl'])} |")

    print("\n### P1 — per teacher x seed\n")
    print("| arm | ceiling 3e-4 | ceiling 1e-4 | KL@400 3e-4 | KL@400 1e-4 | "
          "KL@matched gain 3e-4 | 1e-4 | ceiling not lower | collateral lower | PARETO |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(p1):
        v = p1[k]
        print(f"| `{k}` | {v['ceiling_3e4']:.3f} | **{v['ceiling_1e4']:.3f}** | "
              f"{v['kl400_3e4']:.3f} | **{v['kl400_1e4']:.3f}** | "
              f"{f(v['kl_at_gain05_3e4'])} | {f(v['kl_at_gain05_1e4'])} | "
              f"{'YES' if v['ceiling_not_lower'] else 'NO'} | "
              f"{'YES' if v['collateral_lower'] else 'NO'} | "
              f"{'**PASS**' if v['pareto'] else '**FAIL**'} |")

    print("\n### P2 — the zero-content control\n")
    print("| seed | ctrl KL@400 3e-4 | mean with-content KL@400 3e-4 | overshoot share | "
          "ctrl KL@400 1e-4 | shrink | comparable (>=60%) | shrink >=40% |")
    print("|---|---|---|---|---|---|---|---|")
    for k in sorted(p2):
        v = p2[k]
        print(f"| {v['seed']} | {v['ctrl_kl400_3e4']:.3f} | {v['mean_content_kl400_3e4']:.3f} | "
              f"**{v['overshoot_share_3e4']:.1%}** | {v['ctrl_kl400_1e4']:.3f} | "
              f"**{v['ctrl_shrink_1e4']:.1%}** | "
              f"{'YES' if v['comparable_at_3e4'] else 'NO'} | "
              f"{'YES' if v['shrinks_ge_40pct'] else 'NO'} |")

    if A.get("alternate_student_arms"):
        print("\n### Alternate-student arms (a different candidate fold parent)\n")
        print("| arm | ceiling 3e-4 | ceiling 1e-4 | KL@400 3e-4 | KL@400 1e-4 | PARETO |")
        print("|---|---|---|---|---|---|")
        for k, v in sorted(A["alternate_student_arms"].items()):
            print(f"| `{k}` | {v['ceiling_3e4']:.3f} | **{v['ceiling_1e4']:.3f}** | "
                  f"{v['kl400_3e4']:.3f} | **{v['kl400_1e4']:.3f}** | "
                  f"{'**PASS**' if v['pareto'] else '**FAIL**'} |")

    print(f"\n**VERDICT** — {A['verdict']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "aggregate"
    if cmd == "build-states":
        build_states()
    elif cmd == "probe":
        probe(sys.argv[2:])
    elif cmd == "aggregate":
        aggregate()
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
