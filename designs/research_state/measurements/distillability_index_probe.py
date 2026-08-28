"""DISTILLABILITY INDEX — micro-distill probe (CPU, offline, `models/` read-only).

Producer for `distillability_index_gen_2026-08-28.{md,json}`.

Instrument: fine-tune a STUDENT checkpoint's FULL policy on a fixed TEACHER's argmax over a fixed
ON-SLICE state set, and track (a) ABSORPTION = held-out on-slice top-1 agreement with the teacher,
(b) COLLATERAL = off-slice divergence from the student's OWN pre-probe policy (masked KL, top-1
agreement, and |ΔV|).

Run (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src):

    python distillability_index_probe.py build-states
    python distillability_index_probe.py probe <cell> <student> <teacher_set> <seed> <steps> [lr]
    python distillability_index_probe.py aggregate

`<student>` is a checkpoint path, `FRESH:<seed>` (fresh-init at the current architecture), or
`TEACHER`. `<teacher_set>` is `A` / `B`, or `A*` for the CONTENT CONTROL (targets are the
student's own argmax — same optimizer, zero new behavioural content).
"""
from __future__ import annotations

import sys

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("PROBE_TORCH_THREADS", "1")))

NEG = -1e8


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


def fresh_policy_like(model, seed: int):
    """A FRESH-INIT policy at the identical architecture (SB3's own construction path)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pol = model.policy_class(
        model.observation_space, model.action_space, model.lr_schedule, **model.policy_kwargs
    )
    _silence(pol)
    return pol


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
    """Masked log-probs (N,11) float64 for a whole state set."""
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
    """KL(new || ref) per row, over legal actions only. Illegal entries are exactly 0 prob."""
    m = mask.astype(bool)
    p = np.exp(logp_new)
    d = np.where(m, p * (logp_new - logp_ref), 0.0)
    return d.sum(1)


# ============================================================================
# STATE-SET BUILDER
# ============================================================================
import json, glob, os, sys, hashlib
import numpy as np

MODELS = "/home/goodlad/dev/gen3ai/models"
OUT = os.path.dirname(os.path.abspath(__file__))
TEACHER = "ai_v9_53_R2F5a_0826"
TEACHER_B = "ai_v9_54_R2F5b_0826"
PARENT = "ai_v9_29_rev1_0823"


def harvest(run):
    """[(file, team_key, step, opponent, obs, mask)] over a run's eval traces."""
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
        if obs.ndim != 2 or obs.shape[1] != 2501:
            continue
        keep = mask.sum(1) >= 2            # a 1-legal-action state carries no policy signal
        if not keep.any():
            continue
        parts = sm.split("/")
        recs.append(dict(file=sm, team=team, step=parts[-3], opp=parts[-2],
                         obs=obs[keep].astype(np.float32), mask=mask[keep].astype(bool)))
    return recs


def pick(recs, n, seed, per_file_cap=None):
    """Deterministic stratified draw: round-robin over source FILES (so no single battle
    dominates), then a seeded shuffle of the resulting rows."""
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
    t_recs = harvest(TEACHER)
    teams = sorted({r["team"] for r in t_recs})
    assert len(teams) == 2, teams
    print(f"[on] teacher {TEACHER}: {len(t_recs)} files, {sum(len(r['obs']) for r in t_recs)} usable rows, "
          f"{len(teams)} pinned teams")

    # Split ON by FILE (battle) so held-out states are from battles never trained on.
    rng = np.random.default_rng(20260828)
    fidx = rng.permutation(len(t_recs))
    n_held_files = max(1, len(t_recs) // 4)
    held_files = set(fidx[:n_held_files].tolist())
    tr = [r for i, r in enumerate(t_recs) if i not in held_files]
    he = [r for i, r in enumerate(t_recs) if i in held_files]

    on_tr_o, on_tr_m, _ = pick(tr, 3000, 11)
    on_he_o, on_he_m, _ = pick(he, 1200, 12)

    # OFF: the parent run's own traces, MINUS any state on either pinned team.
    p_recs = [r for r in harvest(PARENT) if r["team"] not in set(teams)]
    off_o, off_m, off_prov = pick(p_recs, 1500, 13, per_file_cap=2)
    off_teams = len({p_recs[i]["team"] for i, _ in off_prov})
    off_steps = sorted({p_recs[i]["step"] for i, _ in off_prov})
    print(f"[off] parent {PARENT}: {len(p_recs)} files after pin-exclusion; drew {len(off_o)} rows "
          f"from {off_teams} distinct teams across {len(off_steps)} eval steps")

    # Second teacher (bonus cell)
    tb_recs = harvest(TEACHER_B)
    tb_teams = sorted({r["team"] for r in tb_recs})
    tb_fidx = np.random.default_rng(20260829).permutation(len(tb_recs))
    tb_held = set(tb_fidx[:max(1, len(tb_recs) // 4)].tolist())
    tb_tr_o, tb_tr_m, _ = pick([r for i, r in enumerate(tb_recs) if i not in tb_held], 3000, 21)
    tb_he_o, tb_he_m, _ = pick([r for i, r in enumerate(tb_recs) if i in tb_held], 1200, 22)
    # OFF for teacher B must exclude ITS pins too
    pb = [r for r in harvest(PARENT) if r["team"] not in set(tb_teams)]
    offb_o, offb_m, _ = pick(pb, 1500, 23, per_file_cap=2)

    np.savez_compressed(
        os.path.join(OUT, "states_A.npz"),
        on_train_obs=on_tr_o, on_train_mask=on_tr_m,
        on_held_obs=on_he_o, on_held_mask=on_he_m,
        off_obs=off_o, off_mask=off_m,
    )
    np.savez_compressed(
        os.path.join(OUT, "states_B.npz"),
        on_train_obs=tb_tr_o, on_train_mask=tb_tr_m,
        on_held_obs=tb_he_o, on_held_mask=tb_he_m,
        off_obs=offb_o, off_mask=offb_m,
    )
    prov = dict(
        teacher_A=dict(run=TEACHER, pinned_teams=[list(t) for t in teams],
                       trace_files=len(t_recs), usable_rows=int(sum(len(r["obs"]) for r in t_recs)),
                       train_files=len(tr), held_files=len(he),
                       n_on_train=int(len(on_tr_o)), n_on_held=int(len(on_he_o))),
        teacher_B=dict(run=TEACHER_B, pinned_teams=[list(t) for t in tb_teams],
                       trace_files=len(tb_recs), n_on_train=int(len(tb_tr_o)), n_on_held=int(len(tb_he_o))),
        off_slice=dict(run=PARENT, files_after_exclusion=len(p_recs), n=int(len(off_o)),
                       distinct_teams=off_teams, eval_steps=off_steps),
        filters="action_mask.sum(1) >= 2; obs dim 2501; ON split by BATTLE FILE (75/25)",
    )
    json.dump(prov, open(os.path.join(OUT, "state_provenance.json"), "w"), indent=2)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'eval_steps'})
                      for k, v in prov.items()}, indent=2)[:1600])


# ============================================================================
# ONE PROBE CELL
# ============================================================================
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = "/home/goodlad/dev/gen3ai/models"
TEACHERS = {"A": f"{MODELS}/ai_v9_53_R2F5a_0826/final_model.zip",
            "B": f"{MODELS}/ai_v9_54_R2F5b_0826/final_model.zip"}
BATCH = 256


def eval_points(n):
    pts = {0, n}
    v = 1
    while v <= n:
        pts.add(v); v = int(np.ceil(v * 1.6))
    return sorted(pts)


def probe(argv):
    cell, spec, tset, seed, nsteps = argv[0], argv[1], argv[2], int(argv[3]), int(argv[4])
    LR = float(argv[5]) if len(argv) > 5 else 3e-4
    out_path = os.path.join(HERE, "results", f"{cell}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t_start = time.time()

    self_target = tset.endswith("*")          # "A*" = targets are the STUDENT's OWN argmax
    tset = tset.rstrip("*")
    S = np.load(os.path.join(HERE, f"states_{tset}.npz"))
    on_tr_o, on_tr_m = S["on_train_obs"], S["on_train_mask"]
    on_he_o, on_he_m = S["on_held_obs"], S["on_held_mask"]
    off_o, off_m = S["off_obs"], S["off_mask"]

    # ---- teacher targets (computed ONCE per teacher set, cached) ----
    tcache = os.path.join(HERE, f"teacher_targets_{tset}.npz")
    if os.path.exists(tcache):
        T = np.load(tcache)
        tgt_tr, tgt_he, tprob_he = T["tgt_train"], T["tgt_held"], T["tprob_held"]
    else:
        tm, _ = load_policy(TEACHERS[tset])
        lp_tr = eval_probs(tm.policy, on_tr_o, on_tr_m)
        lp_he = eval_probs(tm.policy, on_he_o, on_he_m)
        tgt_tr, tgt_he, tprob_he = lp_tr.argmax(1), lp_he.argmax(1), np.exp(lp_he)
        np.savez_compressed(tcache, tgt_train=tgt_tr, tgt_held=tgt_he, tprob_held=tprob_he)
        del tm

    # ---- student ----
    dropped = ()
    if spec.startswith("FRESH:"):
        base, _ = load_policy(TEACHERS[tset])          # architecture donor only
        policy = fresh_policy_like(base, int(spec.split(":")[1]))
        del base
        student_path = f"fresh-init(seed={spec.split(':')[1]}) at current arch"
    else:
        path = TEACHERS[tset] if spec == "TEACHER" else spec
        m, dropped = load_policy(path)
        policy = m.policy
        student_path = path
    if dropped:
        raise SystemExit(f"REFUSED: {student_path} dropped kwargs {dropped} — not a faithful rebuild")

    if self_target:
        # CONTENT CONTROL: same optimizer, same states, ZERO new behavioural content.
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

    json.dump(dict(cell=cell, student=student_path, teacher_set=tset, teacher=TEACHERS[tset],
                   probe_seed=seed, n_steps=nsteps, lr=LR, batch=BATCH,
                   n_on_train=int(len(on_tr_o)), n_on_held=int(len(on_he_o)), n_off=int(len(off_o)),
                   dropped_kwargs=list(dropped), curve=curve,
                   loss_first10=losses[:10], loss_last10=losses[-10:],
                   wall_s=round(time.time() - t_start, 1)),
              open(out_path, "w"), indent=2)
    print(f"[done] {cell} -> {out_path}  ({time.time()-t_start:.0f}s)")


# ============================================================================
# AGGREGATION
# ============================================================================
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AGE = {"age_02M": 2_000_016, "age_06M": 6_000_000, "age_12M": 12_000_000,
       "age_18M": 18_000_016, "age_24M": 24_000_000, "age_25M_final": 25_000_000}


def crossing(curve, key, thresh, use_gain):
    """Collateral readouts at the FIRST point the (running-max) absorption crosses `thresh`.
    Linear interpolation between the two bracketing eval points. None if never reached."""
    a = np.array([c["on_held_agree"] for c in curve], float)
    a = np.maximum.accumulate(a)                       # monotone envelope: noise must not decide
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


def load_all():
    cells = {}
    for f in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
        name = os.path.basename(f)[:-5]
        d = json.load(open(f)); d["cell"] = name
        cells[name] = d
    return cells


def summarize(d):
    c = d["curve"]
    a = np.array([x["on_held_agree"] for x in c], float)
    s = dict(cell=d["cell"], student=d["student"], teacher_set=d["teacher_set"], seed=d["probe_seed"],
             lr=d["lr"], n_steps=d["n_steps"],
             a0=float(a[0]), a_max=float(a.max()), gain_max=float(a.max() - a[0]),
             step1_shock_kl=float(c[1]["off_kl"]) if len(c) > 1 else None,
             final_off_kl=float(c[-1]["off_kl"]), final_off_agree=float(c[-1]["off_agree"]),
             final_off_value_mad=float(c[-1]["off_value_mad"]))
    for g in (0.03, 0.05, 0.10):
        s[f"idx_gain_{g:.2f}"] = crossing(c, "on_held_agree", g, True)
    for A in (0.60, 0.70, 0.78, 0.80):
        s[f"idx_abs_{A:.2f}"] = crossing(c, "on_held_agree", A, False)
    # MATCHED-EFFORT readouts: no interpolation, no ceiling assumption.
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


def aggregate():
    cells = load_all()
    summ = {k: summarize(v) for k, v in cells.items()}
    out = dict(cells=summ, curves={k: v["curve"] for k, v in cells.items()},
               meta={k: {kk: v[kk] for kk in ("student", "teacher", "teacher_set", "probe_seed",
                                              "lr", "batch", "n_steps", "n_on_train", "n_on_held",
                                              "n_off", "dropped_kwargs", "wall_s")}
                     for k, v in cells.items()},
               state_provenance=json.load(open(os.path.join(HERE, "state_provenance.json"))))
    json.dump(out, open(os.path.join(HERE, "aggregate.json"), "w"), indent=2)

    def fmt(v, d=3):
        return "MISS" if v is None else f"{v:.{d}f}"

    hdr = (f"{'cell':30s} {'a0':>6s} {'amax':>6s} {'gain':>7s} {'shock':>7s} "
           f"{'KL@d05':>7s} {'AGR@d05':>8s} {'Vmad@d05':>9s} {'step@d05':>9s} {'KL@0.70':>8s}")
    print(hdr)
    for k in sorted(summ):
        s_ = summ[k]
        i = s_["idx_gain_0.05"]; ia = s_["idx_abs_0.70"]
        print(f"{k:30s} {s_['a0']:6.3f} {s_['a_max']:6.3f} {s_['gain_max']:+7.3f} "
              f"{s_['step1_shock_kl']:7.3f} "
              f"{fmt(None if i is None else i['off_kl']):>7s} "
              f"{fmt(None if i is None else i['off_agree']):>8s} "
              f"{fmt(None if i is None else i['off_value_mad'], 2):>9s} "
              f"{fmt(None if i is None else i['step'], 0):>9s} "
              f"{fmt(None if ia is None else ia['off_kl']):>8s}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "aggregate"
    if cmd == "build-states":
        build_states()
    elif cmd == "probe":
        probe(sys.argv[2:])
    elif cmd == "aggregate":
        aggregate()
    else:
        raise SystemExit(__doc__)
