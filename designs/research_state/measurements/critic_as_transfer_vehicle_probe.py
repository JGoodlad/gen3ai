"""M5 — IS THE CRITIC THE OFF-SLICE VEHICLE?

Question: after a distillation fold, what changes on UNTAUGHT-team states — the POLICY's action
distribution, or the CRITIC's valuation — and which of the two ORDERS with the measured untaught
win-rate gift?

Two eras, each a (parent, fold) pair plus a FIXED reference opponent that is an ancestor of both:

  v8   parent ai_v8_04_distill_4teacher_0722   fold ai_v8_14_distill3_0725   ref ai_v8_03_zarch_control_0718
       -> the fold that GIFTED untaught teams (+5.42pp, v8_redistribution_pfsp_2026-08-30)
  gen  parent ai_v9_59_R2ACTION_0827           fold ai_v9_70_R3ACTION_0828   ref ai_v9_29_rev1_0823
       -> the fold that did NOT gift (-0.75pp null, rev3_untaught_pulldown_2026-08-30)

METHOD.  For each era we PLAY battles on each probe team (untaught + a taught contrast) against
the fixed reference opponent, and at every decision we score BOTH arms on the IDENTICAL observation
and the IDENTICAL legal mask, in one process:

  * masked action distribution   -> KL(fold||parent), TV, argmax agreement            [POLICY change]
  * critic scalar V              -> rank correlation, standardized |dV|, level shift  [CRITIC change]
  * win-prob head                -> calibration (LEVEL) and AUC/resolution (RESOLUTION)

The battle's realized outcome is the label.  Both state sets are collected: `--actor parent`
(the parent's own distribution) and `--actor fold`.  Cross-scoring on a common state set is the
paired change measurement; each arm on its OWN generated states is the on-policy resolution read.

WHY realized outcomes and not tight-MC labels: a battle outcome IS one draw from the true value
distribution at every state it contains, so discrimination (AUC), Murphy reliability/resolution and
the reliability curve are all consistently estimable from single draws — what tight MC buys is
per-state resolution of an individual anchor, which is what the truthcheck needed and this probe
does not.  We spend the budget on MANY states with 1-draw labels + battle-clustered inference
rather than few states with R=40 labels; see the write-up's Limits section.

MASKS ARE REAL, NEVER INFERRED.  The recorded eval traces do not carry the action mask and their
logits are already-normalized log-probs, so `logits > -1e8` recovers ALL-LEGAL (the documented
vacuous-guard trap).  This probe therefore generates its own states and takes the mask straight
from `embed_battle`.

Run (v8 arm must run in the ERA-PINNED tree; gen arm in the current one):

  export PYTHONPATH=/tmp/probeP_v8era/src
  nice -n 15 python critic_as_transfer_vehicle_probe.py collect --era v8  --actor parent --n 10
  export PYTHONPATH=$PYTHONPATH:src
  nice -n 15 python critic_as_transfer_vehicle_probe.py collect --era gen --actor parent --n 10
  nice -n 15 python critic_as_transfer_vehicle_probe.py analyze
  nice -n 15 python critic_as_transfer_vehicle_probe.py report
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("POKESIM_SIM_BRIDGE_BIN",
                      "/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge")

import numpy as np  # noqa: E402

MAIN = "/home/goodlad/dev/gen3ai"
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("M5_SCRATCH", "/tmp/m5_critic_vehicle")
STEM = os.path.join(HERE, "critic_as_transfer_vehicle_2026-08-31")

# --- ERA CONFIG ------------------------------------------------------------------------------
ERAS = {
    "v8": {
        "parent": ("ai_v8_04_distill_4teacher_0722", "final_model_interrupted.zip"),
        "fold": ("ai_v8_14_distill3_0725", "final_model_interrupted.zip"),
        "ref": ("ai_v8_03_zarch_control_0718", "final_model_interrupted.zip"),
        # MATCHED-NOISE CONTROL: an EARLIER checkpoint of the parent's OWN run.  parent-vs-control
        # is ordinary training with NO fold, so it calibrates "how much do these two meters move
        # per unit of training" and makes 'the policy moved more than the critic' a scale-free
        # claim instead of a comparison of two different units.
        "control": ("ai_v8_04_distill_4teacher_0722",
                    "checkpoints/checkpoint_269716291_steps.zip"),
        "control_span_steps": 7.46e6,   # 269.72M -> 277.18M; the fold spans ~14.8M (see Limits)
        "impl": "node",  # the era's rust bridge predates the seedless-seed fix (bc00d4d)
        "gift_json": f"{HERE}/v8_redistribution_pfsp_2026-08-30.json",
    },
    "gen": {
        "parent": ("ai_v9_59_R2ACTION_0827", "final_model.zip"),
        "fold": ("ai_v9_70_R3ACTION_0828", "final_model.zip"),
        "ref": ("ai_v9_29_rev1_0823", "final_model.zip"),
        "control": ("ai_v9_59_R2ACTION_0827", "snapshots/snapshot_000024000000.zip"),
        "control_span_steps": 4.07e6,   # 24.00M -> 28.07M; the fold spans ~4.55M — well matched
        "impl": "rust",
        "gift_json": f"{HERE}/rev3_untaught_pulldown_2026-08-30.json",
    },
}

# rev-3's OWN taught teams: one per teacher cluster F6a..F6f (the taught contrast for the gen era),
# read from the recorded --trainee-teams of each fleet arm and pinned here after that read.
GEN_TAUGHT_RUNS = ["ai_v9_63_R3F6a_0828", "ai_v9_64_R3F6b_0828", "ai_v9_65_R3F6c_0828",
                   "ai_v9_66_R3F6d_0828", "ai_v9_67_R3F6e_0828", "ai_v9_68_R3F6f_0828"]

SEED = 20260831


# --- team resolution -------------------------------------------------------------------------
def _sha(team_str: str) -> str:
    return hashlib.sha1(team_str.strip().encode()).hexdigest()[:10]


def probe_teams(era: str) -> list[dict]:
    """[{key, team_str, kind, gift}] — kind in {untaught, taught}; gift = measured per-team
    fold-minus-parent win-rate delta (untaught only, from the published probe)."""
    from utils.team_loader import TeamLoader
    if era == "v8":
        d = json.load(open(ERAS["v8"]["gift_json"]))["per_team"]
        by_sha = {_sha(t): t for t in TeamLoader().get_all_teams()}
        out = []
        for sha, rec in d.items():
            if sha not in by_sha:
                raise SystemExit(f"[m5] v8 team {sha} not in this tree's pool — wrong tree?")
            out.append({"key": sha, "team_str": by_sha[sha], "kind": rec["kind"],
                        "arch": rec["arch"],
                        "gift": round(rec["fold_wr"] - rec["parent_wr"], 6)})
        return sorted(out, key=lambda r: (r["kind"], r["key"]))
    # gen
    q = json.load(open(ERAS["gen"]["gift_json"]))
    gift = {r["team"]: r["A_minus_B"]["mean"] for r in q["per_team"]}
    arch = {t["basename"]: t["archetype"] for t in q["selection"]["teams"]}
    out = []
    for bn, g in gift.items():
        out.append({"key": bn, "team_str": open(f"{MAIN}/data/teams/sample/{bn}.txt").read(),
                    "kind": "untaught", "arch": arch.get(bn, "?"), "gift": g})
    for run in GEN_TAUGHT_RUNS:
        md = json.load(open(f"{MAIN}/models/{run}/metadata.json"))
        tt = (md.get("cli_args") or {}).get("trainee_teams")
        if not tt:
            raise SystemExit(f"[m5] {run}: no recorded --trainee-teams; refusing to guess")
        bn = os.path.basename(tt.split(",")[0].strip())[:-4]   # one per teacher cluster
        out.append({"key": bn, "team_str": open(f"{MAIN}/data/teams/sample/{bn}.txt").read(),
                    "kind": "taught", "arch": "?", "gift": None})
    return sorted(out, key=lambda r: (r["kind"], r["key"]))


# --- collection ------------------------------------------------------------------------------
def _load(run: str, zipname: str, cv):
    from agents.model.snapshot import load_foreign_opponent
    m, _ = load_foreign_opponent(f"{MAIN}/models/{run}/{zipname}", current_version=cv,
                                 device="cpu", config_path=f"{MAIN}/models/{run}/model_config.json")
    fe = m.policy.features_extractor
    if hasattr(fe, "_debugger"):
        fe._debugger = None
    m.policy.set_training_mode(False)
    return m


def _score(model, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(log-softmax over the 11 RAW logits, critic V, win-prob) for one state.

    The mask is applied by the analysis, not here, so the stored row keeps the model's own
    unmasked logits and the mask is a separate column — a mask bug can then be seen, not baked in.
    """
    import torch as th
    ob = th.as_tensor(obs[None, :])
    mk = th.as_tensor(mask[None, :])
    pin = {"observation": ob, "action_mask": mk}
    with th.no_grad():
        dist = model.policy.get_distribution(pin)
        logits = dist.distribution.logits[0].cpu().numpy().astype(np.float32)
        v = float(model.policy.predict_values(pin)[0].item())
        fe = model.policy.features_extractor
        wl = getattr(fe, "last_win_prob_logits", None)
        wp = float(th.sigmoid(wl[0, 0]).item()) if wl is not None else float("nan")
    return logits, v, wp


def collect(era: str, actor: str, n: int, shard: str, kinds: str) -> int:
    from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration
    from poke_env.teambuilder import Teambuilder
    from agents.inference.player import RLPlayer
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import load_mappings
    from utils.bridge.local_battle_runner import run_local_battles
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder

    cfg = ERAS[era]
    mappings = load_mappings()
    cv = current_model_version(mappings)
    models = {k: _load(*cfg[k], cv) for k in ("parent", "fold", "ref", "control")}
    obs_dim = int(models["parent"].policy.observation_space["observation"].shape[0])
    print(f"[m5] era={era} actor={actor} obs_dim={obs_dim}", flush=True)

    # ACID: the two arms must be DISTINCT networks, else every delta below is a structural zero.
    sdp, sdf = models["parent"].policy.state_dict(), models["fold"].policy.state_dict()
    keys = sorted(set(sdp) & set(sdf))
    l2 = sum(float((sdp[k] - sdf[k]).pow(2).sum()) for k in keys
             if sdp[k].shape == sdf[k].shape and sdp[k].is_floating_point()) ** 0.5
    if l2 <= 1e-3:
        raise SystemExit(f"[m5] ACID FAILED: parent and fold are the same network (L2={l2})")
    print(f"[m5] ACID param L2(parent,fold) = {l2:.3f}", flush=True)

    teams = [t for t in probe_teams(era) if t["kind"] in kinds.split(",")]
    si, sk = (int(x) for x in shard.split("/"))
    if sk > 1:
        teams = [t for i, t in enumerate(teams) if i % sk == si]
    print(f"[m5] {len(teams)} teams: {[t['key'] for t in teams]}", flush=True)

    pool = TeamLoader().get_all_teams()
    base_tb = Gen3Teambuilder(pool)
    rng = random.Random(SEED)
    order = [rng.randrange(0, len(base_tb.packed_teams)) for _ in range(n)]
    seeds = [[rng.randrange(0, 0x10000) for _ in range(4)] for _ in range(n)]

    class _Seq(Teambuilder):
        def __init__(self, base, order):
            self._packed, self._order, self.i = base.packed_teams, order, 0

        def reset(self):
            self.i = 0

        def yield_team(self):
            t = self._packed[self._order[self.i % len(self._order)]]
            self.i += 1
            return t

    acct = [0]

    def _a(tag):
        acct[0] += 1
        return AccountConfiguration(f"M5{tag[:2]}{acct[0]:05d}", "pw")

    scored: list[dict] = []

    class _Rec(RLPlayer):
        def embed_battle(self, b):
            d = super().embed_battle(b)
            mask = np.asarray(d["action_mask"], np.float32)
            if mask.sum() <= 1:          # no decision, or a forced single legal action
                return d
            obs = np.asarray(d["observation"], np.float32)
            lp, vp, wpp = _score(models["parent"], obs, mask)
            lf, vf, wpf = _score(models["fold"], obs, mask)
            lc, vc, wpc = _score(models["control"], obs, mask)
            scored.append({"tag": getattr(b, "battle_tag", ""), "turn": int(getattr(b, "turn", 0)),
                           "mask": mask.astype(np.int8).tolist(),
                           "lp_parent": [round(float(x), 5) for x in lp],
                           "lp_fold": [round(float(x), 5) for x in lf],
                           "lp_control": [round(float(x), 5) for x in lc],
                           "v_parent": vp, "v_fold": vf, "v_control": vc,
                           "wp_parent": wpp, "wp_fold": wpf, "wp_control": wpc})
            return d

    os.makedirs(SCRATCH, exist_ok=True)
    out = f"{SCRATCH}/rows_{era}_{actor}_{si}of{sk}.jsonl"
    done = set()
    if os.path.exists(out):
        for ln in open(out):
            done.add(json.loads(ln)["team"])
        print(f"[m5] resume: {len(done)} teams already done", flush=True)
    sink = open(out, "a")

    for t in teams:
        if t["key"] in done:
            continue
        t0 = time.time()
        opp_tb = _Seq(base_tb, order)
        p = _Rec(model=models[actor], team=Gen3Teambuilder([t["team_str"]]),
                 battle_format="gen3ou", server_configuration=LocalhostServerConfiguration,
                 mappings=mappings, account_configuration=_a("P"), stochastic=False,
                 start_listening=False)
        o = RLPlayer(model=models["ref"], team=opp_tb, battle_format="gen3ou",
                     server_configuration=LocalhostServerConfiguration, mappings=mappings,
                     account_configuration=_a("O"), stochastic=False, start_listening=False)
        battles = []
        for i in range(n):
            scored.clear()
            w0, f0 = p.n_won_battles, p.n_finished_battles
            try:
                asyncio.run(run_local_battles(p, o, 1, seed=seeds[i], impl=cfg["impl"]))
            except Exception as e:
                print(f"    [drop] {t['key']} battle {i}: {type(e).__name__}: {e}", flush=True)
                continue
            if p.n_finished_battles == f0:
                print(f"    [drop] {t['key']} battle {i}: unfinished", flush=True)
                continue
            won = 1 if p.n_won_battles > w0 else 0
            rows = list(scored)
            for j, r in enumerate(rows):
                r["battle"] = i
                r["won"] = won
                r["pos"] = (j + 1) / len(rows)
            battles.append({"i": i, "won": won, "n_states": len(rows), "rows": rows})
        if opp_tb.i != len(battles) and opp_tb.i != n:
            print(f"    [warn] {t['key']}: opp teambuilder yielded {opp_tb.i} for {n} battles",
                  flush=True)
        ns = sum(b["n_states"] for b in battles)
        wr = np.mean([b["won"] for b in battles]) if battles else float("nan")
        sink.write(json.dumps({"era": era, "actor": actor, "team": t["key"], "kind": t["kind"],
                               "arch": t["arch"], "gift": t["gift"], "obs_dim": obs_dim,
                               "battles": battles}) + "\n")
        sink.flush()
        print(f"  {t['key']:18s} {t['kind']:8s} wr={wr:.3f} n_battles={len(battles)} "
              f"states={ns}  {time.time() - t0:.0f}s", flush=True)
    sink.close()
    return 0


# --- statistics ------------------------------------------------------------------------------
def _masked_logsoftmax(lp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask > 0, lp, -1e9)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z) * (mask > 0)
    return e / e.sum(axis=-1, keepdims=True)


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC with ties handled; nan if a class is empty."""
    pos, neg = y > 0.5, y <= 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    r = np.empty(len(score), float)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    i = 0
    ranks = np.empty(len(score), float)
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r[order] = ranks
    n1, n0 = pos.sum(), neg.sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def _murphy(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Brier = reliability - resolution + uncertainty.  reliability = LEVEL error (calibration),
    resolution = how far the conditional outcome rates move away from the base rate = the
    discriminating content."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ybar = y.mean()
    rel = res = 0.0
    for k in range(bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        rel += nk * (p[m].mean() - y[m].mean()) ** 2
        res += nk * (y[m].mean() - ybar) ** 2
    n = len(y)
    return {"brier": float(np.mean((p - y) ** 2)), "reliability": rel / n,
            "resolution": res / n, "uncertainty": float(ybar * (1 - ybar)),
            "bias": float(p.mean() - ybar)}


def _cluster_boot(units: list[np.ndarray], stat, reps: int = 4000, seed: int = 7) -> dict:
    """Bootstrap a statistic over CLUSTERS (battles).  `stat` maps a concatenated array-of-rows
    to a float; each unit is one battle's rows."""
    rng = np.random.default_rng(seed)
    K = len(units)
    if K < 2:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "z": float("nan")}
    point = stat(np.concatenate(units))
    draws = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, K, K)
        draws[r] = stat(np.concatenate([units[j] for j in idx]))
    draws = draws[np.isfinite(draws)]
    # A cell whose clusters all share one outcome makes AUC undefined on every resample; that is a
    # MISSING value, never an interpolated one.
    if len(draws) < 3 or not np.isfinite(point):
        return {"point": float(point) if np.isfinite(point) else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "z": float("nan"),
                "n_finite_draws": int(len(draws))}
    sd = float(draws.std(ddof=1))
    return {"point": float(point), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "z": float(point / sd) if sd > 0 else float("nan"),
            "n_finite_draws": int(len(draws))}


def _load_rows(era: str, actor: str) -> list[dict]:
    out = []
    for fn in sorted(os.listdir(SCRATCH)) if os.path.isdir(SCRATCH) else []:
        if not fn.startswith(f"rows_{era}_{actor}_"):
            continue
        for ln in open(f"{SCRATCH}/{fn}"):
            out.append(json.loads(ln))
    return out


def _pack(team_rec: dict) -> dict:
    """Flatten one team record into aligned arrays + a battle-index vector."""
    L = {k: [] for k in ("mask", "lp_parent", "lp_fold", "lp_control",
                         "v_parent", "v_fold", "v_control",
                         "wp_parent", "wp_fold", "wp_control", "won", "pos", "battle")}
    for b in team_rec["battles"]:
        for r in b["rows"]:
            for k in L:
                L[k].append(r[k])
    A = {k: np.asarray(v, dtype=np.float32) for k, v in L.items()}
    A["battle"] = np.asarray(L["battle"], dtype=np.int32)
    return A


def _pair_change(A: dict, a: str, b: str, units, mask) -> dict:
    """How far did arm `b` move from arm `a` on these states — on the POLICY meter and on the
    CRITIC meter, both computed on the identical states with the identical legal mask."""
    eps = 1e-12
    pa = _masked_logsoftmax(A[f"lp_{a}"], mask)
    pb = _masked_logsoftmax(A[f"lp_{b}"], mask)
    kl = np.sum(pb * (np.log(pb + eps) - np.log(pa + eps)), axis=1)
    tv = 0.5 * np.abs(pb - pa).sum(axis=1)
    agree = (np.argmax(np.where(mask > 0, A[f"lp_{a}"], -1e9), axis=1)
             == np.argmax(np.where(mask > 0, A[f"lp_{b}"], -1e9), axis=1)).astype(float)
    # PopArt means/scales differ between arms, so a raw |ΔV| conflates an affine re-scaling with a
    # change of shape.  Each arm is z-scored WITHIN the cell first: what survives is the part of
    # the critic's landscape that actually moved.
    va, vb = A[f"v_{a}"], A[f"v_{b}"]
    za = (va - va.mean()) / (va.std() or 1.0)
    zb = (vb - vb.mean()) / (vb.std() or 1.0)
    dv = np.abs(zb - za)

    def clustered(vals):
        return _cluster_boot([vals[u] for u in units], lambda x: float(np.mean(x)))

    wb_v, wb_td = [], []
    for u in units:
        if len(u) >= 4:
            wb_v.append(_spearman(va[u], vb[u]))
            ta, tb = np.diff(va[u]), np.diff(vb[u])
            if len(ta) >= 3:
                wb_td.append(_spearman(ta, tb))
    return {
        "kl": clustered(kl), "tv": clustered(tv), "argmax_agree": clustered(agree),
        "dv_std": clustered(dv),
        "v_spearman": _spearman(va, vb),
        "v_spearman_within_battle": float(np.nanmean(wb_v)) if wb_v else float("nan"),
        "td_spearman_within_battle": float(np.nanmean(wb_td)) if wb_td else float("nan"),
        "wp_level_shift": clustered(A[f"wp_{b}"] - A[f"wp_{a}"]),
        "wp_spearman": _spearman(A[f"wp_{a}"], A[f"wp_{b}"]),
    }


def team_stats(A: dict) -> dict:
    """Every per-state quantity for one team, plus the battle-clustered aggregates."""
    if len(A["won"]) == 0:
        return {}
    mask = A["mask"]
    y = A["won"]
    bat = A["battle"]
    units = [np.where(bat == b)[0] for b in np.unique(bat)]

    def auc_of(col):
        return _cluster_boot([np.stack([col[u], y[u]], 1) for u in units],
                             lambda x: _auc(x[:, 0], x[:, 1]))

    out = {"n_states": int(len(y)), "n_battles": int(len(units)), "win_rate": float(y.mean())}
    out["fold_vs_parent"] = _pair_change(A, "parent", "fold", units, mask)
    out["parent_vs_control"] = _pair_change(A, "control", "parent", units, mask)
    # the FOLD change expressed in units of an equal-ish stretch of ordinary no-fold training
    fp, pc = out["fold_vs_parent"], out["parent_vs_control"]
    out["fold_over_control"] = {
        "kl": fp["kl"]["point"] / pc["kl"]["point"] if pc["kl"]["point"] else float("nan"),
        "tv": fp["tv"]["point"] / pc["tv"]["point"] if pc["tv"]["point"] else float("nan"),
        "disagree": ((1 - fp["argmax_agree"]["point"]) / (1 - pc["argmax_agree"]["point"])
                     if (1 - pc["argmax_agree"]["point"]) else float("nan")),
        "dv_std": fp["dv_std"]["point"] / pc["dv_std"]["point"] if pc["dv_std"]["point"] else float("nan"),
        "v_derank": ((1 - fp["v_spearman_within_battle"]) / (1 - pc["v_spearman_within_battle"])
                     if (1 - pc["v_spearman_within_battle"]) else float("nan")),
    }
    # flatten the fold-vs-parent meters to the top level for backwards-compatible readers
    out.update({k: fp[k] for k in ("kl", "tv", "argmax_agree", "dv_std", "v_spearman",
                                   "v_spearman_within_battle", "td_spearman_within_battle",
                                   "wp_level_shift", "wp_spearman")})
    out["wp_spearman_within_battle"] = float(np.nanmean(
        [_spearman(A["wp_parent"][u], A["wp_fold"][u]) for u in units if len(u) >= 4]))
    # CRITIC QUALITY vs the realized outcome — the resolution-vs-level split
    for arm in ("parent", "fold", "control"):
        out[f"auc_v_{arm}"] = auc_of(A[f"v_{arm}"])
        out[f"auc_wp_{arm}"] = auc_of(A[f"wp_{arm}"])
        out[f"murphy_{arm}"] = _murphy(A[f"wp_{arm}"], y)
    # EARLY-GAME stratum.  A state in the last third of a decided battle is near-terminal and both
    # critics call it correctly, which inflates every AUC toward a shared ceiling; the early stratum
    # is where a resolution difference can exist at all.
    early = A["pos"] < 0.5
    if early.sum() >= 50:
        eu = [u[early[u]] for u in units]
        eu = [u for u in eu if len(u) > 0]
        for arm in ("parent", "fold", "control"):
            out[f"auc_wp_early_{arm}"] = _cluster_boot(
                [np.stack([A[f"wp_{arm}"][u], y[u]], 1) for u in eu],
                lambda x: _auc(x[:, 0], x[:, 1]))
            out[f"murphy_early_{arm}"] = _murphy(A[f"wp_{arm}"][early], y[early])
        out["n_states_early"] = int(early.sum())
    return out


def analyze() -> int:
    res = {"probe": "M5 critic-as-transfer-vehicle", "date": "2026-08-31",
           "scratch": SCRATCH, "cells": {}}
    for era in ERAS:
        for actor in ("parent", "fold"):
            recs = _load_rows(era, actor)
            if not recs:
                continue
            cell = {"per_team": {}, "by_kind": {}}
            packed = {}
            for r in recs:
                A = _pack(r)
                packed[r["team"]] = (r, A)
                st = team_stats(A)
                st.update({"kind": r["kind"], "arch": r["arch"], "gift": r["gift"]})
                cell["per_team"][r["team"]] = st
            for kind in ("untaught", "taught"):
                sel = [(r, A) for r, A in packed.values() if r["kind"] == kind]
                if not sel:
                    continue
                M = {k: np.concatenate([A[k] for _, A in sel]) for k in sel[0][1]}
                # make battle ids globally unique so clustering is by (team, battle)
                offs, cur = [], 0
                for _, A in sel:
                    offs.append(A["battle"] + cur)
                    cur += int(A["battle"].max()) + 1 if len(A["battle"]) else 0
                M["battle"] = np.concatenate(offs)
                st = team_stats(M)
                st["n_teams"] = len(sel)
                cell["by_kind"][kind] = st
            # co-occurrence: which change ORDERS with the per-team gift?
            gifts, kls, dvs, aucd, wpd, tvs = [], [], [], [], [], []
            for k, st in cell["per_team"].items():
                if st.get("gift") is None or st["kind"] != "untaught":
                    continue
                gifts.append(st["gift"]); kls.append(st["kl"]["point"])
                tvs.append(st["tv"]["point"]); dvs.append(st["dv_std"]["point"])
                aucd.append(st["auc_wp_fold"]["point"] - st["auc_wp_parent"]["point"])
                wpd.append(st["murphy_fold"]["resolution"] - st["murphy_parent"]["resolution"])
            if len(gifts) >= 4:
                g = np.asarray(gifts, float)
                cell["gift_cooccurrence"] = {
                    "n_teams": len(gifts), "gift_mean": float(g.mean()),
                    "spearman_gift_vs_kl": _spearman(g, np.asarray(kls)),
                    "spearman_gift_vs_tv": _spearman(g, np.asarray(tvs)),
                    "spearman_gift_vs_dv": _spearman(g, np.asarray(dvs)),
                    "spearman_gift_vs_d_auc_wp": _spearman(g, np.asarray(aucd)),
                    "spearman_gift_vs_d_resolution": _spearman(g, np.asarray(wpd)),
                    "rows": [{"team": k, "gift": st["gift"], "kl": st["kl"]["point"],
                              "tv": st["tv"]["point"], "dv_std": st["dv_std"]["point"],
                              "d_auc_wp": st["auc_wp_fold"]["point"] - st["auc_wp_parent"]["point"],
                              "d_resolution": (st["murphy_fold"]["resolution"]
                                               - st["murphy_parent"]["resolution"]),
                              "d_reliability": (st["murphy_fold"]["reliability"]
                                                - st["murphy_parent"]["reliability"])}
                             for k, st in cell["per_team"].items()
                             if st.get("gift") is not None and st["kind"] == "untaught"],
                }
            res["cells"][f"{era}/{actor}"] = cell
    def _plain(o):
        if isinstance(o, dict):
            return {k: _plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_plain(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    json.dump(_plain(res), open(f"{STEM}.json", "w"), indent=1)
    print(f"[m5] wrote {STEM}.json ({len(res['cells'])} cells)")
    return 0


def _f(x, nd=4):
    try:
        return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except Exception:
        return "n/a"


def _ci(d, nd=4, scale=1.0):
    if not d or not np.isfinite(d.get("point", float("nan"))):
        return "n/a"
    return (f"{d['point'] * scale:.{nd}f} [{d['lo'] * scale:.{nd}f}, "
            f"{d['hi'] * scale:.{nd}f}]")


def report() -> int:
    R = json.load(open(f"{STEM}.json"))
    L = []
    for cell_name, cell in R["cells"].items():
        L.append(f"\n### cell `{cell_name}` (states generated by the {cell_name.split('/')[1]} arm)\n")
        L.append("| class | teams | battles | states | WR | KL(fold‖parent) | TV | argmax-agree "
                 "| \\|ΔV\\|/sd | ρ(V) pooled / within-battle | Δwin-prob level |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for kind, st in cell.get("by_kind", {}).items():
            L.append(f"| {kind} | {st.get('n_teams')} | {st['n_battles']} | {st['n_states']} | "
                     f"{_f(st['win_rate'],3)} | {_ci(st['kl'])} | {_ci(st['tv'])} | "
                     f"{_ci(st['argmax_agree'],3)} | {_ci(st['dv_std'],3)} | "
                     f"{_f(st['v_spearman'],4)} / {_f(st['v_spearman_within_battle'],4)} | "
                     f"{_ci(st['wp_level_shift'],4)} |")
        L.append("")
        L.append("| class | arm | AUC(V) | AUC(win-prob) | Murphy RESOLUTION | Murphy RELIABILITY "
                 "(level err) | bias | Brier |")
        L.append("|---|---|---|---|---|---|---|---|")
        for kind, st in cell.get("by_kind", {}).items():
            for arm in ("control", "parent", "fold"):
                m = st[f"murphy_{arm}"]
                L.append(f"| {kind} | {arm} | {_ci(st[f'auc_v_{arm}'],3)} | "
                         f"{_ci(st[f'auc_wp_{arm}'],3)} | {_f(m['resolution'],4)} | "
                         f"{_f(m['reliability'],4)} | {_f(m['bias'],3)} | {_f(m['brier'],4)} |")
        if any("auc_wp_early_parent" in st for st in cell.get("by_kind", {}).values()):
            L.append("")
            L.append("EARLY stratum (first half of each battle) — where a resolution difference "
                     "can exist at all")
            L.append("")
            L.append("| class | states | arm | AUC(win-prob) | RESOLUTION | RELIABILITY | bias |")
            L.append("|---|---|---|---|---|---|---|")
            for kind, st in cell.get("by_kind", {}).items():
                if "auc_wp_early_parent" not in st:
                    continue
                for arm in ("control", "parent", "fold"):
                    m = st[f"murphy_early_{arm}"]
                    L.append(f"| {kind} | {st['n_states_early']} | {arm} | "
                             f"{_ci(st[f'auc_wp_early_{arm}'],3)} | {_f(m['resolution'],4)} | "
                             f"{_f(m['reliability'],4)} | {_f(m['bias'],3)} |")
        L.append("")
        L.append("**Fold change vs the matched no-fold control** (parent ← an earlier checkpoint of "
                 "the parent's OWN run; ratio > 1 = the fold moved this meter more than ordinary "
                 "training did over a comparable stretch)")
        L.append("")
        L.append("| class | KL fold / control | TV | argmax-DISagreement | \\|Δz(V)\\| | "
                 "V within-battle DE-ranking |")
        L.append("|---|---|---|---|---|---|")
        for kind, st in cell.get("by_kind", {}).items():
            fo = st.get("fold_over_control", {})
            L.append(f"| {kind} | {_f(fo.get('kl'),2)}x | {_f(fo.get('tv'),2)}x | "
                     f"{_f(fo.get('disagree'),2)}x | {_f(fo.get('dv_std'),2)}x | "
                     f"{_f(fo.get('v_derank'),2)}x |")
        L.append("")
        L.append("| class | pair | KL | TV | argmax-agree | \\|Δz(V)\\| | ρ(V) within-battle | "
                 "ρ(TD) within-battle |")
        L.append("|---|---|---|---|---|---|---|---|")
        for kind, st in cell.get("by_kind", {}).items():
            for pair in ("fold_vs_parent", "parent_vs_control"):
                p = st.get(pair)
                if not p:
                    continue
                L.append(f"| {kind} | {pair} | {_ci(p['kl'])} | {_ci(p['tv'])} | "
                         f"{_ci(p['argmax_agree'],3)} | {_ci(p['dv_std'],3)} | "
                         f"{_f(p['v_spearman_within_battle'],4)} | "
                         f"{_f(p['td_spearman_within_battle'],4)} |")
        co = cell.get("gift_cooccurrence")
        if co:
            L.append("")
            L.append(f"co-occurrence with per-team gift (n={co['n_teams']} untaught teams, Spearman): "
                     f"KL {_f(co['spearman_gift_vs_kl'],3)} · TV {_f(co['spearman_gift_vs_tv'],3)} · "
                     f"|ΔV| {_f(co['spearman_gift_vs_dv'],3)} · "
                     f"ΔAUC(win-prob) {_f(co['spearman_gift_vs_d_auc_wp'],3)} · "
                     f"Δresolution {_f(co['spearman_gift_vs_d_resolution'],3)}")
            L.append("")
            L.append("| team | gift | KL | TV | \\|ΔV\\|/sd | ΔAUC(wp) | Δresolution | Δreliability |")
            L.append("|---|---|---|---|---|---|---|---|")
            for r in sorted(co["rows"], key=lambda r: -r["gift"]):
                L.append(f"| `{r['team']}` | {r['gift']:+.4f} | {_f(r['kl'],4)} | {_f(r['tv'],4)} | "
                         f"{_f(r['dv_std'],3)} | {r['d_auc_wp']:+.4f} | "
                         f"{r['d_resolution']:+.5f} | {r['d_reliability']:+.5f} |")
    txt = "\n".join(L)
    open(f"{STEM}_tables.md", "w").write(txt + "\n")
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--era", required=True, choices=list(ERAS))
    c.add_argument("--actor", required=True, choices=["parent", "fold"])
    c.add_argument("--n", type=int, default=10)
    c.add_argument("--shard", default="0/1")
    c.add_argument("--kinds", default="untaught,taught")
    sub.add_parser("analyze")
    sub.add_parser("report")
    t = sub.add_parser("teams")
    t.add_argument("--era", required=True, choices=list(ERAS))
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        return collect(a.era, a.actor, a.n, a.shard, a.kinds)
    if a.cmd == "analyze":
        return analyze()
    if a.cmd == "report":
        return report()
    if a.cmd == "teams":
        for t_ in probe_teams(a.era):
            print(t_["key"], t_["kind"], t_["arch"], t_["gift"])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
