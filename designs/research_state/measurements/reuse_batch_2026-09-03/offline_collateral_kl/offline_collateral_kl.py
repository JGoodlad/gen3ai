"""OFFLINE off-slice displacement — the anchor's collateral_kl, recomputed from saved checkpoints.

WHY THIS EXISTS: --distill-anchor-monitor was carried by the DOSE argvs only, so C1 and B2 emit no
distill/* tags at all. Their displacement is UNMEASURED, not small. This recovers a comparable
quantity from artifacts that already exist, with no training.

WHAT IT IS, precisely: the SAME statistic the callback logs — `masked_kl_rows` imported from the
live module, forward KL(parent || arm) over LEGAL actions with illegal logits driven to -inf on both
sides — averaged over OFF-SLICE rows. Offline the off-slice set is exact by construction: every
state comes from a battle piloted on one of the 8 UNTAUGHT teams, which are disjoint from every
arm's taught slice (verified: overlap 0).

WHAT IT IS NOT: the callback accumulates over the fold's OWN rollout states as training proceeds;
this reads one FIXED batch drawn from a single behaviour policy at the end. Same formula, different
state distribution. It belongs in its OWN column and must never be merged with the logged
0.545/0.583/0.605.

THE CALIBRATION IS THE POINT (Model Review 2's design): the three dose arms carry BOTH numbers, so
their offline ordering is checked against their logged ordering FIRST. If the offline statistic does
not reproduce dose-monotonicity, C1's offline number is uninterpretable and we learn that before
reading it — not after.

Run: python offline_collateral_kl.py <out.json> [n_battles=24] [concurrency=3]
"""
import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import asyncio, json, random, sys, time
import numpy as np
import torch as th
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from agents.inference.player import RLPlayer
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
from agents.training.instrumented_ppo.distill_anchor import masked_kl_rows
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

REV1 = "models/ai_v9_29_rev1_0823"
CFG = f"{REV1}/snapshots/model_config.json"
PARENT = "models/ai_v9_59_R2ACTION_0827/final_model.zip"
OPPONENT = f"{REV1}/snapshots/snapshot_000024000000.zip"
ARMS = [("R4DOSE12","models/ai_v9_150_R4DOSE12_0901/final_model.zip","0.53x", 0.5446),
        ("R4DOSE6", "models/ai_v9_151_R4DOSE6_0901/final_model.zip", "1.06x", 0.5832),
        ("R4DOSE3", "models/ai_v9_152_R4DOSE3_0901/final_model.zip", "2.12x", 0.6047),
        ("B2",      "models/ai_v9_140_B2_0901/final_model.zip",      "coef .1761", None),
        ("C1",      "models/ai_v9_141_C1_0901/final_model.zip",      "coef 0",     None)]
import re
# The untaught-8 list, IN SEED ORDER, lives in untaught_teams.json beside this file. It used to be
# regex-read from an `untaught_probe.py` that is not in the tree (found 2026-09-05 by the
# content_locality probe, which recovered the order from untaught_C1_end.json and reproduced the
# canonical 1100 states exactly). The old path is kept as a fallback for a checkout that has it.
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_here, "untaught_teams.json")):
    import json as _json
    UNTAUGHT = _json.load(open(os.path.join(_here, "untaught_teams.json")))["untaught"]
else:
    UNTAUGHT = re.findall(r'"(data/teams/sample/[0-9a-f]+\.txt)"',
                          open(os.path.join(_here, "untaught_probe.py")).read())
assert len(UNTAUGHT) == 8, UNTAUGHT


def _strip_debugger(m):
    obj = getattr(m, "policy", m)
    for mod in (obj.modules() if hasattr(obj, "modules") else []):
        if getattr(mod, "_debugger", None) is not None:
            mod._debugger = None
    return m


class PinnedTeam(Gen3Teambuilder):
    def __init__(self, path): super().__init__([open(path).read()])
    def yield_team(self): return self.packed_teams[0]


class PairedPool(Gen3Teambuilder):
    def __init__(self, teams):
        super().__init__(teams); self._seq, self._i = [], 0
    def set_sequence(self, seq):
        self._seq, self._i = seq, 0; return self
    def yield_team(self):
        t = self.packed_teams[self._seq[self._i % len(self._seq)]]; self._i += 1; return t


class Capturing(RLPlayer):
    """Records every observation the pilot actually acted on — the off-slice state batch."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.captured = []
    def embed_battle(self, battle):
        d = super().embed_battle(battle)
        if d is not None and d.get("action_mask") is not None and int(d["action_mask"].sum()) > 0:
            self.captured.append({"observation": d["observation"].copy(),
                                  "action_mask": d["action_mask"].copy()})
        return d


def main(out_path, n_battles=24, conc=3):
    maps = load_mappings(); cv = current_model_version(maps)
    parent, _ = load_foreign_opponent(PARENT, current_version=cv, device="cpu", config_path=CFG)
    _strip_debugger(parent)
    opp_model, _ = load_foreign_opponent(OPPONENT, current_version=cv, device="cpu", config_path=CFG)
    _strip_debugger(opp_model)
    if conc != 1 and os.environ.get("OKL_ALLOW_CONCURRENCY") != "1":
        raise SystemExit(
            f"REFUSING concurrency={conc}: this measurement is only reproducible at concurrency=1.\n"
            "  Seeds pin the dice and both players' sampling, but interleaved battles still consume\n"
            "  the shared streams in a scheduling-dependent order -- measured 2026-09-03, seeded at\n"
            "  concurrency=3 two runs differed by 52 states and up to +0.043 in level.\n"
            "  Re-run with concurrency 1, or set OKL_ALLOW_CONCURRENCY=1 to accept unquotable levels.")

    loader = TeamLoader(); pool = PairedPool(loader.get_all_teams()); n_pool = len(pool.packed_teams)

    # --- collect a FIXED off-slice state batch: the PARENT piloting the 8 untaught teams ---
    t0 = time.time(); states = []; team_of = []
    per_team = max(1, n_battles // len(UNTAUGHT))
    for ti, team in enumerate(UNTAUGHT):
        rng = random.Random(61000 + ti)
        seq = [rng.randrange(n_pool) for _ in range(per_team)]
        # SEEDED (2026-09-03): unseeded, a re-run played different battles -- 948 vs 1213 states
        # and every arm's LEVEL moving +0.04..+0.075, so only orderings were quotable. Three
        # sources have to be pinned together or the run still wanders: the sim dice (seed= below),
        # and BOTH players' action sampling, which otherwise draw from the process-wide RNG that
        # the bridge interleaves between them.
        pilot = Capturing(model=parent, team=PinnedTeam(team), battle_format="gen3ou",
                          server_configuration=LocalhostServerConfiguration, mappings=maps,
                          account_configuration=AccountConfiguration(f"OKL{ti}a", "pw"),
                          stochastic=True, start_listening=False, policy_seed=71000 + ti)
        opp = RLPlayer(model=opp_model, team=pool.set_sequence(seq), battle_format="gen3ou",
                       server_configuration=LocalhostServerConfiguration, mappings=maps,
                       account_configuration=AccountConfiguration(f"OKL{ti}b", "pw"),
                       stochastic=True, start_listening=False, policy_seed=72000 + ti)
        pilot.reset_battles(); opp.reset_battles()
        asyncio.run(run_local_battles(pilot, opp, per_team, concurrency=conc, impl="rust",
                                      seed=[ti + 1, 2, 3, 4]))
        states.extend(pilot.captured)
        team_of.extend([ti] * len(pilot.captured))   # cluster label: states repeat within a team
        print(f"  team {ti}: +{len(pilot.captured)} states (total {len(states)})", flush=True)
    print(f"  collected {len(states)} off-slice states in {time.time()-t0:.0f}s", flush=True)
    if len(states) < 200:
        raise SystemExit(f"FATAL: only {len(states)} states — too few to average a KL over")

    obs = {"observation": th.as_tensor([s["observation"] for s in states], dtype=th.float32),
           "action_mask": th.as_tensor([s["action_mask"] for s in states], dtype=th.float32)}

    def logits_of(model):
        sp = model.observation_space.spaces
        o = {k: v for k, v in obs.items() if k in sp}
        with th.no_grad():
            return model.policy.get_distribution(o).distribution.logits

    p_log = logits_of(parent)
    cl = np.array(team_of)
    # ONE fixed resampling index set shared by every arm, so an arm-vs-arm difference is PAIRED on
    # the same team draws instead of each arm carrying its own independent noise.
    bs_idx = np.random.default_rng(20260903).integers(0, len(UNTAUGHT), (20000, len(UNTAUGHT)))
    per_team_kl = {}
    res = {"_meta": {"statistic": "forward KL(parent||arm) over legal actions, mean over off-slice rows",
                     "same_formula_as": "distill_anchor.masked_kl_rows (imported, not reimplemented)",
                     "state_source": "PARENT piloting the 8 UNTAUGHT teams — off-slice by construction",
                     "batch_is_fixed": "YES as of 2026-09-03, and it takes BOTH halves: the four "
                                       "seeds below AND concurrency=1. Seeds alone are NOT enough -- "
                                       "measured, seeded at concurrency=3 two runs still gave 1193 vs "
                                       "1141 states and levels apart by up to +0.043, because "
                                       "interleaved battles consume the shared streams in a "
                                       "scheduling-dependent order. At concurrency=1 two runs are "
                                       "bit-identical (1100 states, every arm equal to 6 decimals). "
                                       "The two artifacts dated before this are UNSEEDED draws, kept "
                                       "deliberately: their +0.04..+0.075 level shift is the evidence "
                                       "for the old levels-are-a-draw caveat.",
                     "read_the_CLUSTER_mean": "cluster_mean/cluster_ci95 weight each of the 8 teams "
                                              "equally and are the program's unit. offline_collateral_kl "
                                              "is the STATE-weighted mean and can disagree in SIGN: on "
                                              "2026-09-03 C1-B2 was +0.0309 pooled but -0.0188 clustered "
                                              "(6 of 8 teams negative, two big-count teams carrying the "
                                              "pooled sign).",
                     "NOT_the_same_as": "the callback's logged value (fold's own rollout states, accumulated)",
                     "n_states": len(states), "parent": PARENT, "opponent": OPPONENT,
                     "seeds": {"sim": "[team_index+1, 2, 3, 4] per team",
                               "pilot_policy": "71000 + team_index",
                               "opponent_policy": "72000 + team_index",
                               "pool_sequence": "61000 + team_index",
                               "note": "pinned 2026-09-03; the two artifacts dated before that are "
                                       "UNSEEDED draws and are kept deliberately -- their +0.04..+0.075 "
                                       "level shift is the evidence for the levels-are-a-draw caveat"}}}
    for tag, path, desc, logged in ARMS:
        if not os.path.exists(path):
            print(f"  {tag}: MODEL MISSING — UNCOVERED", flush=True); res[tag] = None; continue
        m, _ = load_foreign_opponent(path, current_version=cv, device="cpu", config_path=CFG)
        _strip_debugger(m)
        rows = masked_kl_rows(p_log, logits_of(m), obs["action_mask"])
        v = float(rows.mean())
        r_np = rows.detach().cpu().numpy()
        # A mean over ~950 states is a POINT with no interval, and reading one arm against another
        # off two such points is exactly the vacuous comparison this program retired. Cluster over
        # TEAMS, as everywhere else -- between-team variance dominates and more states per team
        # does not shrink it.
        per_team = np.array([r_np[cl == t].mean() for t in range(len(UNTAUGHT))])
        bstat = per_team[bs_idx].mean(axis=1)
        res[tag] = {"offline_collateral_kl": v, "desc": desc, "logged_callback_value": logged,
                    "per_team": [float(x) for x in per_team],
                    "cluster_mean": float(per_team.mean()),
                    "cluster_ci95": [float(np.percentile(bstat, 2.5)),
                                     float(np.percentile(bstat, 97.5))]}
        per_team_kl[tag] = per_team
        print(f"  {tag:9s} {desc:11s} offline {v:.4f}   (logged {logged if logged is not None else '—'})", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)

    # --- PAIRED arm-vs-arm differences (same team draws) — the quantity a reader compares ---
    pairs = {}
    for a, b in (("C1", "B2"), ("R4DOSE3", "R4DOSE12"), ("B2", "R4DOSE3")):
        if a in per_team_kl and b in per_team_kl:
            d = per_team_kl[a] - per_team_kl[b]
            bd = d[bs_idx].mean(axis=1)
            lo, hi = float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))
            pairs[f"{a}-{b}"] = {"delta": float(d.mean()), "ci95": [lo, hi],
                                 "separates_from_zero": not (lo <= 0 <= hi)}
            print(f"  PAIRED {a}-{b}: {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] "
                  f"{'SEPARATES' if not (lo <= 0 <= hi) else 'SPANS ZERO'}", flush=True)
    res["_meta"]["paired_differences"] = pairs

    # --- CALIBRATION GATE: does the offline statistic reproduce the dose ordering? ---
    d = [res[t]["offline_collateral_kl"] for t in ("R4DOSE12","R4DOSE6","R4DOSE3") if res.get(t)]
    if len(d) == 3:
        ok = d[0] < d[1] < d[2]
        res["_meta"]["calibration"] = {
            "offline_dose_ordering": d, "logged_dose_ordering": [0.5446, 0.5832, 0.6047],
            "reproduces_logged_monotonicity": bool(ok),
            "verdict": ("offline statistic tracks the logged one on the arms that carry both — "
                        "C1/B2 offline values are interpretable")
                       if ok else
                       ("offline statistic does NOT reproduce the logged dose ordering — "
                        "C1/B2 offline values are UNINTERPRETABLE, do not read them")}
        print(f"\n  CALIBRATION: offline dose ordering {d[0]:.4f} / {d[1]:.4f} / {d[2]:.4f}"
              f"  -> {'REPRODUCES' if ok else '** DOES NOT REPRODUCE **'} the logged ordering", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24,
         int(sys.argv[3]) if len(sys.argv) > 3 else 3)
