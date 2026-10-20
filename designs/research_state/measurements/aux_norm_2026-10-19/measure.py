"""F-LG-12 (TECH_DEBT P1) — the MEASURED bias of the learner's per-micro SUBSET normalisers under gradient
accumulation, on real Rust-collector rows, CPU only.

Usage (from a worktree root, PYTHONPATH=$PWD/src, CUDA_VISIBLE_DEVICES=""):

    python designs/research_state/measurements/aux_norm_2026-10-19/measure.py collect --out <dir>
    python designs/research_state/measurements/aux_norm_2026-10-19/measure.py atoms   --out <dir>
    python designs/research_state/measurements/aux_norm_2026-10-19/measure.py simulate --out <dir>

``collect``: a real complete-game rollout on the Rust collector at the PRODUCTION surface (the K9(a) golden's
seeded learner, `learner_golden.build_learner`'s surface, p2 a seeded random external route), saved as an npz.

``atoms``: the rollout is cut, after the learner's own intent-label alignment, into ATOMS of ``ATOM`` rows (one
`rollout_buffer.get` permutation). For every atom the REAL region R1 (`micro_step.micro_step`) runs and every
subset-normalised term is split into its NUMERATOR S (the summed per-element loss) and its COUNT n (the labelled
elements), read back from R1's own metrics; ``S / n`` is checked against R1's loss value for the atom (the split is
self-validating). Additive in S and n, so a micro-batch of M rows is the union of M / ATOM random atoms.

``simulate``: for each learner SHAPE (micro rows, accumulation K, rollout rows) many accumulation groups are drawn
(atoms with replacement — the live rollout is ~12x the pool) and each term's two normalisations are compared:

  * per-micro (TODAY): ``(1/k) * sum_m S_m / n_m`` (a micro with n_m = 0 contributes 0, as R1's ``present`` gate);
  * whole-group: ``sum_m S_m / sum_m n_m`` (the batch the accumulated step stands for).

Reported per term: the label rate, the per-micro count's CV (the per-row weight dispersion n_bar / n_m), the share
of EMPTY micros, and the relative difference of the two normalisations (mean = the systematic effective-coefficient
bias; sd = the extra noise).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ATOM = 16
#: (name, micro rows, K, rollout rows) — the shapes a live run and its gates train at.
SHAPES = (
    ("production / X26 / X35 / toy (2048 x 32, 98,304 rows)", 2048, 32, 98304),
    ("K9(a) learner golden (16 x 3, 64 rows)", 16, 3, 64),
    ("--debug smoke (2304-row update, 1 group of micro 256 x K 9)", 256, 9, 2304),
)
COLLECT_ENVS, COLLECT_STEPS, COLLECT_SEED, COLLECT_P2 = 32, 256, 41, 9


def _collect(out: Path) -> None:
    import torch as th

    from agents.training import learner_golden as LG
    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.build import RustEnvDecl
    from agents.training.rust_vec_env import RustVecEnv

    TK.build_selfcheck()
    _args, obs, act = TK.production_spaces()
    decl = RustEnvDecl(n_envs=COLLECT_ENVS, threads=2, front="ffi", profile="selfcheck", n_steps=COLLECT_STEPS,
                       micro_batch=COLLECT_STEPS, device="cpu", backend="eager", run_seed=COLLECT_SEED,
                       gamma=1.0, gae_lambda=0.8)
    p2 = TK.RandomP2(COLLECT_P2)
    env = RustVecEnv(n_envs=COLLECT_ENVS, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl, p2=p2, builder=TK.pool_builder()))
    try:
        with LG._one_thread():
            model = TK.fresh_model(env, n_steps=COLLECT_STEPS, batch_size=COLLECT_STEPS, seed=LG.MODEL_SEED,
                                   perturb_seed=LG.PERTURB_SEED, policy_args=_args, perturb_keyed=True)
        th.set_num_threads(4)
        col = env.startup(model)
        if not col.collect(model, TK.NullCallback(), model.rollout_buffer):
            raise SystemExit("the collector stopped")
        rb = model.rollout_buffer
        data = {f"obs:{k}": np.ascontiguousarray(v) for k, v in rb.observations.items()}
        for f in LG._BUFFER_FIELDS:
            data[f] = np.ascontiguousarray(getattr(rb, f))
    finally:
        env.close()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "pool.npz", **data)
    print(f"wrote {out / 'pool.npz'}: {COLLECT_STEPS} x {COLLECT_ENVS} rows")


#: term -> (metric key holding the per-micro LOSS value S/n, how n is read). Every term R1 normalises by a
#: per-micro subset count at the production surface (`belief_bank_static`, `flat_intent_fold`, `win_prob_terms`).
def _split(mets: Dict[str, Tuple[float, float]], obs: Dict[str, Any], B: int) -> Dict[str, Tuple[float, float]]:
    """{term: (S, n)} for one atom from R1's (value, weight) metrics; n = 0 => S = 0."""
    from agents.training.belief_bank import _EV_LOSS_WEIGHT, _NATURE_CE_WEIGHT

    def v(k: str) -> float:
        val, w = mets[k]                    # a missing key is a wrong name, never a zero (KeyError)
        return float(val) if w > 0.5 else 0.0

    out: Dict[str, Tuple[float, float]] = {}
    # hidden_team_set: presence set BCE and BeliefHead set BCE share the SAME scored rows (same cand / k);
    # n is not exported, so it is recomputed below by the caller (`set_rows`).
    out["set_presence_bce"] = (v("belief/set_presence_bce"), float("nan"))
    out["set_beliefhead_bce"] = (v("belief/set_beliefhead_set_bce"), float("nan"))
    m_n = v("belief/set_hyp_moves_supervised")
    out["set_hyp_moves_bce"] = (v("belief/set_hyp_moves_bce") * m_n, m_n)
    n_mb = v("belief/move_revealed_slots") + v("belief/move_unrevealed_slots")
    out["move_belief"] = (v("belief/move_bce") * n_mb, n_mb)
    n_lat = v("belief/movelatent_slots")
    out["move_latent_cos"] = ((1.0 - v("belief/movelatent_cosine")) * n_lat, n_lat)
    n_sp = v("belief/spread_n_slots")
    out["spread"] = (v("belief/spread_loss") * n_sp, n_sp)
    n_nat = v("belief/natureev_n_slots")
    nat_ce = v("belief/natureev_nature_ce")
    out["nature_ce"] = (nat_ce * n_nat, n_nat)
    ev_m = np.asarray(obs["belief_ev_mask"]) > 0.5
    n_ev = float(ev_m.sum()) if n_nat > 0 else 0.0
    ev_loss = (v("belief/natureev_loss") - _NATURE_CE_WEIGHT * nat_ce) / _EV_LOSS_WEIGHT if n_nat > 0 else 0.0
    out["ev"] = (ev_loss * n_ev, n_ev)
    n_hp = v("belief/hptype_n_slots")
    out["hp_type"] = (v("belief/hptype_loss") * n_hp, n_hp)
    n_it = v("belief/item_n_slots")
    out["item"] = (v("belief/item_loss") * n_it, n_it)
    n_in = v("aux/opp_intent/flat_n_supervised")
    out["opp_intent"] = (v("aux/opp_intent/flat_loss") * n_in, n_in)
    n_wp = v("win_prob/coverage") * B
    out["win_prob"] = (v("win_prob/loss") * n_wp, n_wp)
    return out


def _atoms(out: Path) -> None:
    import torch as th

    from agents.model.hypothesis_set import label_multi_hot, set_bce
    from agents.training import learner_golden as LG
    from agents.training.instrumented_ppo.micro_step import micro_step
    from agents.training.rust_vec_env import RustVecEnv

    th.set_num_threads(4)
    with np.load(out / "pool.npz") as z:
        data = {k: z[k] for k in z.files}
    n_steps, n_envs = data["actions"].shape[:2]
    _args, obs_sp, act_sp = __import__("agents.training.rust_rollout.testkit", fromlist=["x"]).production_spaces()
    env = RustVecEnv(n_envs=n_envs, observation_space=obs_sp, action_space=act_sp, build=lambda m: None)
    LG.N_STEPS = n_steps                      # the golden's builder at THIS buffer's shape
    model = LG.build_learner(env=env, trunk_precision="fp32")
    rb = model.rollout_buffer
    rb.reset()
    for k in rb.observations:
        rb.observations[k][...] = data["obs:" + k]
    for f in LG._BUFFER_FIELDS:
        getattr(rb, f)[...] = data[f]
    rb.full, rb.pos = True, n_steps
    model._current_progress_remaining = 1.0
    model.policy.set_training_mode(True)
    model._align_opp_intent_labels()
    st = model._micro_static(model._resolve_fold_flags())
    print("MicroStatic:", st)
    np.random.seed(7)
    rows: List[Dict[str, Tuple[float, float]]] = []
    max_rel = 0.0
    # GRAD ENABLED, as in train(): the extractor stashes some supervision targets (the move-latent table) only
    # under `torch.is_grad_enabled()`; nothing is back-propagated and each atom's graph is dropped.
    with th.enable_grad():
        for i, rd in enumerate(rb.get(ATOM)):
            mo = micro_step(model.policy, rd.observations, rd.actions, rd.action_masks, rd.old_log_prob,
                            rd.old_values, rd.advantages, rd.returns, st)
            mets = {k: (float(a), float(b)) for k, (a, b) in mo.metrics.items()}
            if i == 0:
                print("R1 metric keys:", sorted(k for k in mets if not k.startswith("aux/opp_intent/flat_")))
                print("R1 terms:", sorted(mo.terms))
            obs = {k: t.cpu().numpy() for k, t in rd.observations.items()}
            sp = _split(mets, obs, ATOM)
            fe = model.policy.features_extractor
            hs = fe.last_hypothesis
            if hs is not None:
                y = label_multi_hot(rd.observations["belief_species"], hs.species.logits.shape[-1],
                                    hs.species.logits)
                _l, p_n, _b = set_bce(hs.species.logits, hs.species, y)
                p_n_f = float(p_n)
                sp["set_presence_bce"] = (sp["set_presence_bce"][0] * p_n_f, p_n_f)
                sp["set_beliefhead_bce"] = (sp["set_beliefhead_bce"][0] * p_n_f, p_n_f)
            # self-check: the split reproduces R1's per-term loss value (S / n) for the parts with a direct metric
            for term, key in (("move_belief", "belief/move_bce"), ("spread", "belief/spread_loss"),
                              ("hp_type", "belief/hptype_loss"), ("item", "belief/item_loss"),
                              ("opp_intent", "aux/opp_intent/flat_loss"), ("win_prob", "win_prob/loss"),
                              ("set_hyp_moves_bce", "belief/set_hyp_moves_bce")):
                s, n = sp[term]
                if n > 0:
                    ref = mets[key][0]
                    max_rel = max(max_rel, abs(s / n - ref) / max(abs(ref), 1e-12))
            rows.append(sp)
            if i % 64 == 0:
                print(f"atom {i}", flush=True)
    terms = sorted(rows[0])
    S = np.array([[r[t][0] for t in terms] for r in rows])
    N = np.array([[r[t][1] for t in terms] for r in rows])
    np.savez(out / "atoms.npz", S=S, N=N, terms=np.array(terms), atom=ATOM)
    print(f"wrote {out / 'atoms.npz'}: {len(rows)} atoms x {len(terms)} terms; split self-check max rel err "
          f"{max_rel:.2e}")


def _simulate(out: Path, n_groups: int = 4000, seed: int = 11) -> Dict[str, Any]:
    z = np.load(out / "atoms.npz")
    S, N, terms, atom = z["S"], z["N"], [str(t) for t in z["terms"]], int(z["atom"])
    rng = np.random.default_rng(seed)
    res: Dict[str, Any] = {"atoms": int(S.shape[0]), "atom_rows": atom, "pool_rows": int(S.shape[0] * atom),
                           "shapes": {}}
    for name, micro, K, rollout in SHAPES:
        a_per_micro = micro // atom
        n_micro = rollout // micro
        groups = [K] * (n_micro // K) + ([n_micro % K] if n_micro % K else [])
        per_term: Dict[str, Any] = {}
        for ti, t in enumerate(terms):
            rel, empt, cv = [], [], []
            for _ in range(n_groups):
                k = groups[rng.integers(len(groups))] if len(groups) > 1 else groups[0]
                idx = rng.integers(S.shape[0], size=(k, a_per_micro))
                s_m = S[idx, ti].sum(1)
                n_m = N[idx, ti].sum(1)
                tot_n = n_m.sum()
                if tot_n <= 0:
                    continue
                per_micro = np.where(n_m > 0, s_m / np.maximum(n_m, 1e-30), 0.0).mean()
                whole = s_m.sum() / tot_n
                rel.append(per_micro / whole - 1.0 if whole != 0 else 0.0)
                empt.append(float((n_m <= 0).mean()))
                if k > 1 and n_m.mean() > 0:
                    cv.append(float(n_m.std() / n_m.mean()))
            rate = float(N[:, ti].sum() / (S.shape[0] * atom))
            r = np.asarray(rel)
            per_term[t] = {"labelled_per_row": rate,
                           "count_cv_per_micro": float(np.mean(cv)) if cv else None,
                           "empty_micro_share": float(np.mean(empt)) if empt else None,
                           "rel_diff_mean": float(r.mean()) if r.size else None,
                           "rel_diff_sd": float(r.std()) if r.size else None,
                           "rel_diff_mean_se": float(r.std() / np.sqrt(r.size)) if r.size else None,
                           "groups": int(r.size)}
        res["shapes"][name] = {"micro": micro, "K": K, "rollout": rollout, "groups_per_epoch": groups,
                               "terms": per_term}
    (out / "simulate.json").write_text(json.dumps(res, indent=1))
    for name, sh in res["shapes"].items():
        print(f"\n== {name}  groups/epoch {sh['groups_per_epoch']}")
        print(f"{'term':<20}{'lab/row':>9}{'CV n_m':>9}{'empty':>8}{'bias':>11}{'sd':>10}")
        for t, d in sh["terms"].items():
            def f(x: Any, p: str) -> str:
                return format(x, p) if x is not None else "-"
            print(f"{t:<20}{f(d['labelled_per_row'], '9.3f')}{f(d['count_cv_per_micro'], '9.4f')}"
                  f"{f(d['empty_micro_share'], '8.3f')}{f(d['rel_diff_mean'], '+11.2e')}{f(d['rel_diff_sd'], '10.2e')}")
    return res


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("collect", "atoms", "simulate", "all"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    steps = ("collect", "atoms", "simulate") if a.cmd == "all" else (a.cmd,)
    for c in steps:                         # `all`: one heavy slot for the three steps
        {"collect": _collect, "atoms": _atoms, "simulate": _simulate}[c](a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
