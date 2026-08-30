"""PROBE Q — rev-3's OWN untaught-team pull-down (P3's missing third point).

Question: the rev-2 fold LOST −5.9pp (z=2.5) on teams it never taught while gaining on the
teams it did. Does rev-3's fold do the same? Registered before any data:

  (H1) share-constant   : R3-ACTION loses ≈ rev-2's −5.9pp analogue on never-taught teams.
  (H2) content-externality (breadth-determined): rev-3's narrow 2-team-per-teacher fleet ALSO
       robs, −4..−6pp.
  (H3) ≈0 or positive   : refutes both, reopens the mechanism.

ARMS (all pilot the SAME pinned team, all vs the SAME fixed reference opponent):
  A = R3-ACTION final     models/ai_v9_70_R3ACTION_0828/final_model.zip
  B = R2-ACTION final     models/ai_v9_59_R2ACTION_0827/final_model.zip   (A's PARENT)
  C = rev-1 final         models/ai_v9_29_rev1_0823/final_model.zip       (B's parent)
A−B is the one-hop redistribution read; A−C the two-hop cumulative.

METER CONVENTION (the standing per-team piloting meter): the arm pilots ONE pinned team; the
opponent is a FIXED reference model drawing from the 719-team pool; greedy (stochastic=False);
in-process rust bridge; CPU.

PAIRING: battle i of every (team, arm) cell uses the SAME opponent pool team and the SAME
gen-5 sim seed. The three arms therefore face an identical draw sequence — the paired design
the meter registers. Divergence after ply 1 is intrinsic (different actions consume different
dice); pairing removes the team-draw and opening-dice variance, which is the dominant term.

CIs: battle-CLUSTER bootstrap over TEAMS (the cluster is the team, not the battle — per-team
rows are strongly correlated within a team). Per-team rows get a paired-battle normal CI.

Run:
  nice -n 15 python designs/research_state/measurements/rev3_untaught_pulldown.py --n 200
  (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import asyncio
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
import torch as th  # noqa: E402

th.set_num_threads(1)

from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration  # noqa: E402
from poke_env.teambuilder import Teambuilder  # noqa: E402

from agents.inference.player import RLPlayer  # noqa: E402
from agents.model.snapshot import current_model_version, load_foreign_opponent  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402
from utils.bridge.local_battle_runner import run_local_battles  # noqa: E402
from utils.team_loader import TeamLoader  # noqa: E402
from utils.teambuilder import Gen3Teambuilder  # noqa: E402

MAIN = "/home/goodlad/dev/gen3ai"

ARMS = [
    ("R3ACTION", f"{MAIN}/models/ai_v9_70_R3ACTION_0828/final_model.zip",
     f"{MAIN}/models/ai_v9_70_R3ACTION_0828/model_config.json"),
    ("R2ACTION", f"{MAIN}/models/ai_v9_59_R2ACTION_0827/final_model.zip",
     f"{MAIN}/models/ai_v9_59_R2ACTION_0827/model_config.json"),
    ("REV1", f"{MAIN}/models/ai_v9_29_rev1_0823/final_model.zip",
     f"{MAIN}/models/ai_v9_29_rev1_0823/model_config.json"),
]
# The fixed reference opponent: rev-1 final — the era base, held constant across arms so the
# per-team rows of all three arms are on ONE scale.
REF_OPPONENT = "REV1"

# --- the TAUGHT UNION (read from run metadata `--trainee-teams`, never hand-copied) -----------
TAUGHT_RUNS = [  # F5a-e (rev-2 fleet) + F6a-f + F6CURR (rev-3 fleet)
    "ai_v9_53_R2F5a_0826", "ai_v9_54_R2F5b_0826", "ai_v9_55_R2F5c_0826",
    "ai_v9_56_R2F5d_0826", "ai_v9_57_R2F5e_0826",
    "ai_v9_63_R3F6a_0828", "ai_v9_64_R3F6b_0828", "ai_v9_65_R3F6c_0828",
    "ai_v9_66_R3F6d_0828", "ai_v9_67_R3F6e_0828", "ai_v9_68_R3F6f_0828",
    "ai_v9_69_R3F6CURR_0828",
]


def taught_union() -> set[str]:
    """Every team basename ANY fleet arm pinned, read from recorded metadata."""
    out: set[str] = set()
    for run in TAUGHT_RUNS:
        md = json.load(open(f"{MAIN}/models/{run}/metadata.json"))
        tt = (md.get("cli_args") or {}).get("trainee_teams")
        if not tt:
            raise SystemExit(f"[probeQ] {run}: no recorded --trainee-teams; refusing to guess")
        for p in tt.split(","):
            out.add(os.path.basename(p.strip())[:-4])
    return out


class FixedSequenceTeambuilder(Teambuilder):
    """Yields packed teams from a caller-controlled index sequence — the pairing instrument.

    Wraps an already-validated ``Gen3Teambuilder`` so the pool is byte-identical to the one the
    meter's opponent normally draws from; only the DRAW ORDER is made deterministic.
    """

    def __init__(self, base: Gen3Teambuilder, order: list[int]):
        self._packed = base.packed_teams
        self._keys = base._pool_keys
        self._order = order
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def yield_team(self):
        idx = self._order[self.i % len(self._order)]
        self.i += 1
        return self._packed[idx]

    def key_at(self, i: int) -> str:
        return self._keys[self._order[i % len(self._order)]]


_ACCT = [0]


def _acct(tag: str) -> AccountConfiguration:
    _ACCT[0] += 1
    return AccountConfiguration(f"Q{tag[:3]}{_ACCT[0]:05d}", "pw")


def load(zip_path: str, cfg: str, cv):
    m, _ = load_foreign_opponent(zip_path, current_version=cv, device="cpu", config_path=cfg)
    fe = m.policy.features_extractor
    if hasattr(fe, "_debugger"):
        fe._debugger = None
    m.policy.set_training_mode(False)
    return m


def acid_test(models: dict, mappings) -> dict:
    """The standing load-path acid test, in the form this probe can afford.

    Three facts, each of which would invalidate every number downstream if false:
      1. all three arms LOAD at the current architecture (no silent ArchDrift / random init);
      2. the three are DISTINCT networks — pairwise max|Δ| over a shared forward is > 0 (a
         mis-resolved path that loads the same zip three times reads as a perfect null);
      3. the lineage ORDER shows in function space — A(child) is closer to B(parent) than to
         C(grandparent). A wrong parent assignment would break this.
    """
    sp = next(iter(models.values())).policy.observation_space
    B = 8
    g = th.Generator().manual_seed(20260830)
    obs = {}
    for k, s in sp.spaces.items():
        dt = th.float32 if np.issubdtype(s.dtype, np.floating) else th.int64
        obs[k] = th.zeros((B, *s.shape), dtype=dt)
    # The flat `observation` block carries EMBEDDED IDS as well as continuous features, so a
    # random vector indexes an embedding table out of range. Zeros are a legal (if degenerate)
    # input; the forward's job here is only "loads and produces DIFFERENT logits", and the
    # parameter-space + in-situ checks below carry the real weight.
    del g
    obs["action_mask"] = th.ones((B, 11), dtype=th.int64)
    mask = np.ones((B, 11), dtype=bool)
    obs_dim = int(sp["observation"].shape[0])
    feats = {}
    for name, m in models.items():
        # The arms' obs SPACES are not identical (a distill arm carries `distill_mask`; rev-1
        # does not), so each model is fed exactly the keys its own space declares.
        own = {k: v for k, v in obs.items() if k in m.policy.observation_space.spaces}
        for k, s in m.policy.observation_space.spaces.items():
            if k not in own:
                dt = th.float32 if np.issubdtype(s.dtype, np.floating) else th.int64
                own[k] = th.zeros((B, *s.shape), dtype=dt)
        with th.no_grad():
            dist = m.policy.get_distribution(own, action_masks=mask)
            feats[name] = dist.distribution.probs.detach().clone()
    names = list(models)
    dmat = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dmat[f"{a}|{b}"] = float((feats[a] - feats[b]).abs().max())
    # Parameter-space companion: the most direct "these are three different networks" check,
    # and the one that survives any doubt about a synthetic forward.
    sds = {n: m.policy.state_dict() for n, m in models.items()}
    keys = sorted(set.intersection(*[set(s) for s in sds.values()]))
    pmat = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            tot = 0.0
            for k in keys:
                ta, tb = sds[a][k], sds[b][k]
                if ta.shape == tb.shape and ta.is_floating_point():
                    tot += float((ta - tb).pow(2).sum())
            pmat[f"{a}|{b}"] = tot ** 0.5
    ordered = dmat.get("R3ACTION|R2ACTION", 0.0) < dmat.get("R3ACTION|REV1", 0.0)
    p_ordered = pmat.get("R3ACTION|R2ACTION", 0.0) < pmat.get("R3ACTION|REV1", 0.0)
    ok = all(v > 1e-6 for v in dmat.values()) and all(v > 1e-3 for v in pmat.values())
    return {"pairwise_max_abs_dp": dmat, "pairwise_param_l2": pmat, "all_distinct": bool(ok),
            "fn_lineage_order_child_nearer_parent": bool(ordered),
            "param_lineage_order_child_nearer_parent": bool(p_ordered), "obs_dim": obs_dim}


def battle_seed(rng: random.Random) -> list[int]:
    return [rng.randrange(0, 0x10000) for _ in range(4)]


def run_cell(model, team_str, opp_model, opp_tb: FixedSequenceTeambuilder, n: int,
             seeds: list[list[int]], mappings, tag: str, impl: str) -> list[int]:
    """Play ``n`` paired battles; return the per-battle win indicator (1/0), index-aligned."""
    wins: list[int] = []
    opp_tb.reset()
    pilot_tb = Gen3Teambuilder([team_str])
    p = RLPlayer(model=model, team=pilot_tb, battle_format="gen3ou",
                 server_configuration=LocalhostServerConfiguration, mappings=mappings,
                 account_configuration=_acct(tag), stochastic=False, start_listening=False)
    o = RLPlayer(model=opp_model, team=opp_tb, battle_format="gen3ou",
                 server_configuration=LocalhostServerConfiguration, mappings=mappings,
                 account_configuration=_acct("OPP"), stochastic=False, start_listening=False)
    for i in range(n):
        w0, f0 = p.n_won_battles, p.n_finished_battles
        try:
            asyncio.run(run_local_battles(p, o, 1, seed=seeds[i], impl=impl))
        except Exception as e:  # a wedged/errored battle is DROPPED, never scored
            print(f"    [drop] battle {i}: {type(e).__name__}: {e}", flush=True)
            wins.append(-1)
            continue
        if p.n_finished_battles == f0:
            wins.append(-1)  # unfinished => dropped
        else:
            wins.append(1 if p.n_won_battles > w0 else 0)
    return wins


def paired_ci(a: list[int], b: list[int]) -> tuple[float, float, float, float, int]:
    """Paired difference of two aligned win vectors → (mean, lo, hi, z, n_used)."""
    d = [x - y for x, y in zip(a, b) if x >= 0 and y >= 0]
    n = len(d)
    if n < 2:
        return (float("nan"),) * 4 + (n,)
    m = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    return m, m - 1.96 * se, m + 1.96 * se, (m / se if se > 0 else float("nan")), n


def cluster_bootstrap(per_team: list[list[float]], reps: int = 20000, seed: int = 7) -> tuple[float, float, float, float]:
    """Bootstrap over TEAMS (the cluster). ``per_team[k]`` = that team's paired per-battle diffs."""
    rng = np.random.default_rng(seed)
    teams = [np.asarray(t, dtype=float) for t in per_team if len(t) > 0]
    if not teams:
        return (float("nan"),) * 4
    point = float(np.mean([t.mean() for t in teams]))
    K = len(teams)
    draws = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, K, K)
        draws[r] = np.mean([teams[j].mean() for j in idx])
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    sd = float(draws.std(ddof=1))
    return point, lo, hi, (point / sd if sd > 0 else float("nan"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="battles per team per arm")
    ap.add_argument("--teams", type=int, default=8, help="how many untaught teams")
    ap.add_argument("--pilot", action="store_true", help="1 team, --n games, print throughput and exit")
    ap.add_argument("--impl", default="rust")
    ap.add_argument("--out", default="designs/research_state/measurements/rev3_untaught_pulldown_2026-08-30")
    ap.add_argument("--teams-json", default="", help="explicit comma-list of team basenames (overrides selection)")
    ap.add_argument("--arms", default="R3ACTION,R2ACTION,REV1")
    ap.add_argument("--resume", default="", help="jsonl of already-played cells to skip")
    ap.add_argument("--shard", default="0/1", help="i/k — play only teams with index %% k == i")
    a = ap.parse_args(argv)

    t_start = time.time()
    mappings = load_mappings()
    cv = current_model_version(mappings)

    want = set(a.arms.split(","))
    models = {}
    for name, z, cfg in ARMS:
        if name in want or name == REF_OPPONENT:
            models[name] = load(z, cfg, cv)
    print(f"[probeQ] loaded {list(models)}", flush=True)

    acid = acid_test({k: models[k] for k in ("R3ACTION", "R2ACTION", "REV1") if k in models}, mappings)
    print(f"[probeQ] ACID {json.dumps(acid)}", flush=True)
    if not acid["all_distinct"]:
        raise SystemExit("[probeQ] ACID FAILED — arms are not distinct networks")

    taught = taught_union()
    print(f"[probeQ] taught union ({len(taught)}): {sorted(taught)}", flush=True)

    sel = json.load(open(a.teams_json)) if a.teams_json.endswith(".json") else None
    if sel is None:
        raise SystemExit("[probeQ] --teams-json (the pre-registered selection) is required")
    picks = sel["teams"]
    bad = [t for t in picks if t["basename"] in taught]
    if bad:
        raise SystemExit(f"[probeQ] SELECTION GIGO: {[b['basename'] for b in bad]} are in the taught union")
    if a.pilot:
        picks = picks[:1]
    else:
        picks = picks[: a.teams]
        si, sk = (int(x) for x in a.shard.split("/"))
        if sk > 1:
            picks = [p for i, p in enumerate(picks) if i % sk == si]
    print(f"[probeQ] {len(picks)} untaught teams: {[p['basename'] for p in picks]}", flush=True)

    pool = TeamLoader().get_all_teams()
    base_tb = Gen3Teambuilder(pool)
    print(f"[probeQ] opponent pool = {len(base_tb.packed_teams)} validated teams", flush=True)

    rng = random.Random(20260830)
    order = [rng.randrange(0, len(base_tb.packed_teams)) for _ in range(a.n)]
    seeds = [battle_seed(rng) for _ in range(a.n)]
    opp_tb = FixedSequenceTeambuilder(base_tb, order)

    arms = [x for x in a.arms.split(",") if x in models]
    rows = {}
    done = {}
    if a.resume and os.path.exists(a.resume):
        for ln in open(a.resume):
            r = json.loads(ln)
            done[(r["team"], r["arm"])] = r["wins"]
        print(f"[probeQ] resumed {len(done)} cells", flush=True)
    sink = open(a.resume or (a.out + "_cells.jsonl"), "a")

    for pk in picks:
        tname = pk["basename"]
        team_str = open(f"{MAIN}/data/teams/sample/{tname}.txt").read()
        for arm in arms:
            if (tname, arm) in done:
                rows[(tname, arm)] = done[(tname, arm)]
                continue
            t0 = time.time()
            w = run_cell(models[arm], team_str, models[REF_OPPONENT], opp_tb, a.n, seeds,
                         mappings, arm, a.impl)
            rows[(tname, arm)] = w
            ok = [x for x in w if x >= 0]
            dt = time.time() - t0
            # PAIRING GUARD: the opponent teambuilder must have been consulted exactly once per
            # battle. If it was not, every arm faced whatever poke-env drew on its own and the
            # "paired" claim is false — a silent GIGO that reads as a plausible win rate.
            if opp_tb.i != a.n:
                raise SystemExit(f"[probeQ] PAIRING GIGO: opp teambuilder yielded {opp_tb.i} "
                                 f"teams for {a.n} battles")
            print(f"  {tname} {arm:9s} wr={np.mean(ok) if ok else float('nan'):.3f} "
                  f"n={len(ok)}/{a.n} drop={a.n - len(ok)}  {dt:.0f}s ({dt / max(1, a.n):.2f}s/battle)",
                  flush=True)
            sink.write(json.dumps({"team": tname, "arm": arm, "wins": w}) + "\n")
            sink.flush()
        if a.pilot:
            break

    if a.pilot:
        print(f"[probeQ] PILOT done in {time.time() - t_start:.0f}s", flush=True)
        return 0

    # ---------------- report ----------------
    out = {"probe": "Q", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "acid": acid, "n_per_cell": a.n, "impl": a.impl,
           "ref_opponent": REF_OPPONENT, "taught_union": sorted(taught),
           "selection": sel, "per_team": [], "pooled": {}}
    ab_clusters, ac_clusters, bc_clusters = [], [], []
    for pk in picks:
        t = pk["basename"]
        A, B, C = rows.get((t, "R3ACTION")), rows.get((t, "R2ACTION")), rows.get((t, "REV1"))
        if A is None or B is None:
            continue
        mab, lab, hab, zab, nab = paired_ci(A, B)
        rec = {"team": t, "archetype": pk.get("archetype"), "tags": pk.get("tags"),
               "wr_R3ACTION": float(np.mean([x for x in A if x >= 0])),
               "wr_R2ACTION": float(np.mean([x for x in B if x >= 0])),
               "A_minus_B": {"mean": mab, "lo": lab, "hi": hab, "z": zab, "n": nab}}
        ab_clusters.append([x - y for x, y in zip(A, B) if x >= 0 and y >= 0])
        if C is not None:
            mac, lac, hac, zac, nac = paired_ci(A, C)
            mbc, lbc, hbc, zbc, nbc = paired_ci(B, C)
            rec["wr_REV1"] = float(np.mean([x for x in C if x >= 0]))
            rec["A_minus_C"] = {"mean": mac, "lo": lac, "hi": hac, "z": zac, "n": nac}
            # B − C is REV-2's OWN untaught pull-down, re-measured on THIS team set with THIS
            # instrument — so the P3 row's rev-2 point does not have to be imported from another
            # harness's scale. The 3-coverage-team −5.9pp stays the registered reference.
            rec["B_minus_C"] = {"mean": mbc, "lo": lbc, "hi": hbc, "z": zbc, "n": nbc}
            ac_clusters.append([x - y for x, y in zip(A, C) if x >= 0 and y >= 0])
            bc_clusters.append([x - y for x, y in zip(B, C) if x >= 0 and y >= 0])
        out["per_team"].append(rec)
        # IN-SITU ACID — the strongest of the three, and free. Greedy policies on identical
        # (team, opponent-team, sim-seed) triples are DETERMINISTIC, so two arms that were
        # secretly the same weights would produce BYTE-IDENTICAL win vectors. Any identical pair
        # here means a model path was mis-resolved.
        for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
            u, v = {"A": A, "B": B, "C": C}[x], {"A": A, "B": B, "C": C}[y]
            if v is not None and u == v:
                raise SystemExit(f"[probeQ] IN-SITU ACID FAILED on {t}: arms {x} and {y} produced "
                                 "identical per-battle outcomes — they are the same network")

    p, lo, hi, z = cluster_bootstrap(ab_clusters)
    out["pooled"]["A_minus_B"] = {"mean": p, "lo": lo, "hi": hi, "z": z, "n_teams": len(ab_clusters)}
    if ac_clusters:
        p2, lo2, hi2, z2 = cluster_bootstrap(ac_clusters)
        out["pooled"]["A_minus_C"] = {"mean": p2, "lo": lo2, "hi": hi2, "z": z2, "n_teams": len(ac_clusters)}
        p3, lo3, hi3, z3 = cluster_bootstrap(bc_clusters)
        out["pooled"]["B_minus_C"] = {"mean": p3, "lo": lo3, "hi": hi3, "z": z3, "n_teams": len(bc_clusters)}
        out["pooled"]["mean_wr"] = {
            k: float(np.mean([r["wr_" + k] for r in out["per_team"]])) for k in ("R3ACTION", "R2ACTION", "REV1")}
    out["wall_s"] = time.time() - t_start

    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["pooled"], indent=1), flush=True)
    print(f"[probeQ] wrote {a.out}.json in {out['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
