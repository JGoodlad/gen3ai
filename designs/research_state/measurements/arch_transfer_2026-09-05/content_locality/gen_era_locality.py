"""CONTENT LOCALITY, GEN ERA — is a specialist teacher's divergence LOCAL or GLOBAL?

Statistic: forward KL(teacher || parent) over LEGAL actions, per decision state, using
`masked_kl_rows` IMPORTED from agents.training.instrumented_ppo.distill_anchor (never
reimplemented -- the same formula the live anchor monitor logs).

State batch: the FOLD PARENT (ai_v9_59_R2ACTION_0827/final_model.zip) pilots each of 24 teams --
the 16 taught teams of the teacher-content 2x2 plus the 8 untaught teams of the reuse batch --
against rev-1's 24M snapshot. The recipe (seeds, pinned team, paired pool, concurrency=1) is
COPIED VERBATIM from reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py and
merely extended to more teams. The 8 untaught teams keep indices 0..7 and 3 battles each, so
their states are the SAME states that artifact scored -- a free cross-check, printed as
UNTAUGHT STATE CROSS-CHECK.

Two locality reads (both pre-registered in PREREGISTRATION.md):
  A  per teacher: mean KL over ITS 2 taught teams  /  mean KL over the 8 untaught teams
  B  per taught TEAM: the KL of the teacher that taught it / the mean KL of its 7 same-half
     siblings on the SAME team and the SAME states  (state distribution held fixed)

Matched-noise floor: the identical statistic between nearby same-run checkpoints of the parent.

Run: python gen_era_locality.py <out.json> [battles_per_team=3]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import asyncio, json, random, sys, time
import numpy as np
import torch as th
th.set_num_threads(1)
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from agents.inference.player import RLPlayer
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
from agents.training.instrumented_ppo.distill_anchor import masked_kl_rows
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

MAIN = "/home/goodlad/dev/gen3ai"
MD = f"{MAIN}/models"
REV1 = f"{MD}/ai_v9_29_rev1_0823"
CFG = f"{REV1}/snapshots/model_config.json"
PARENT = f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip"
OPPONENT = f"{REV1}/snapshots/snapshot_000024000000.zip"

# --- MATCHED-NOISE FLOOR: two nearby same-run checkpoints of the PARENT'S OWN run -------------
PAR_CKPT_A = f"{MD}/ai_v9_59_R2ACTION_0827/checkpoints/checkpoint_28067760_steps.zip"   # nearest
PAR_CKPT_B = f"{MD}/ai_v9_59_R2ACTION_0827/checkpoints/checkpoint_27917760_steps.zip"   # -150k


def T(b):
    return f"{MAIN}/data/teams/sample/{b}.txt"


# indices 0..7 -- the reuse batch's UNTAUGHT 8, IN ITS ORDER (read off untaught_C1_end.json's key
# order, since untaught_probe.py is not in the tree). Keeping the order and 3 battles/team makes
# these states byte-identical to the canonical offline_collateral_kl batch.
UNTAUGHT = ["61590463ee85d456", "9283210847f806ee", "ce35b7368c3d692e", "9909f2e98e981ccc",
            "9d5f845869e899ee", "f7ba5702fe856292", "90b94599967c6b77", "dbf81d8ecae51c39"]

# The 8 teacher PAIRS of the teacher-content 2x2. Each pair (unfunded R5F parent, funded R5FUND
# fork) was trained on the SAME 2 teams -- read from each run's recorded --trainee-teams, never
# hand-typed (verified against teacher_content_2x2_2026-09-04/taught_probe.py's 16-team union).
PAIRS = [
    ("00", f"{MD}/ai_v9_92_R5F00_0831",  f"{MD}/ai_v9_120_R5FUND00_0901", ["8bdb5796b9", "436335607f"]),
    ("02", f"{MD}/ai_v9_94_R5F02_0831",  f"{MD}/ai_v9_122_R5FUND02_0901", ["6916e13879", "e702a104eb"]),
    ("04", f"{MD}/ai_v9_96_R5F04_0831",  f"{MD}/ai_v9_124_R5FUND04_0901", ["64a691c473", "e28069562f"]),
    ("06", f"{MD}/ai_v9_98_R5F06_0831",  f"{MD}/ai_v9_126_R5FUND06_0901", ["9e95fb59d7", "9d8391d864"]),
    ("08", f"{MD}/ai_v9_100_R5F08_0831", f"{MD}/ai_v9_128_R5FUND08_0901", ["750d056194", "a9831a5f5e"]),
    ("10", f"{MD}/ai_v9_102_R5F10_0831", f"{MD}/ai_v9_130_R5FUND10_0901", ["6ebe9ebc4d", "9b454d9ea7"]),
    ("12", f"{MD}/ai_v9_104_R5F12_0831", f"{MD}/ai_v9_132_R5FUND12_0901", ["009e3d0244", "b904dbe059"]),
    ("14", f"{MD}/ai_v9_106_R5F14_0831", f"{MD}/ai_v9_134_R5FUND14_0901", ["aea50f207e", "37d717a93a"]),
]
TAUGHT = [b for _, _, _, bs in PAIRS for b in bs]          # 16, indices 8..23
TEAMS = [(b, "untaught") for b in UNTAUGHT] + [(b, "taught") for b in TAUGHT]
IDX = {b: i for i, (b, _) in enumerate(TEAMS)}


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
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.captured = []
    def embed_battle(self, battle):
        d = super().embed_battle(battle)
        if d is not None and d.get("action_mask") is not None and int(d["action_mask"].sum()) > 0:
            self.captured.append({"observation": d["observation"].copy(),
                                  "action_mask": d["action_mask"].copy()})
        return d


def boot_ci(vals, idx):
    b = np.asarray(vals)[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main(out_path, per_team=3):
    maps = load_mappings(); cv = current_model_version(maps)

    def load(p):
        m, _ = load_foreign_opponent(p, current_version=cv, device="cpu", config_path=CFG)
        return _strip_debugger(m)

    parent = load(PARENT)
    opp_model = load(OPPONENT)

    loader = TeamLoader(); pool = PairedPool(loader.get_all_teams())
    n_pool = len(pool.packed_teams)

    t0 = time.time(); states = []; team_of = []
    for ti, (b, kind) in enumerate(TEAMS):
        rng = random.Random(61000 + ti)
        seq = [rng.randrange(n_pool) for _ in range(per_team)]
        pilot = Capturing(model=parent, team=PinnedTeam(T(b)), battle_format="gen3ou",
                          server_configuration=LocalhostServerConfiguration, mappings=maps,
                          account_configuration=AccountConfiguration(f"CL{ti}a", "pw"),
                          stochastic=True, start_listening=False, policy_seed=71000 + ti)
        opp = RLPlayer(model=opp_model, team=pool.set_sequence(seq), battle_format="gen3ou",
                       server_configuration=LocalhostServerConfiguration, mappings=maps,
                       account_configuration=AccountConfiguration(f"CL{ti}b", "pw"),
                       stochastic=True, start_listening=False, policy_seed=72000 + ti)
        pilot.reset_battles(); opp.reset_battles()
        asyncio.run(run_local_battles(pilot, opp, per_team, concurrency=1, impl="rust",
                                      seed=[ti + 1, 2, 3, 4]))
        states.extend(pilot.captured); team_of.extend([ti] * len(pilot.captured))
        print(f"  [{ti:2d}] {kind:8s} {b} +{len(pilot.captured):4d} states "
              f"(total {len(states)}, {time.time()-t0:.0f}s)", flush=True)
    n_untaught_states = sum(1 for t in team_of if t < 8)
    print(f"  UNTAUGHT STATE CROSS-CHECK: {n_untaught_states} states on indices 0..7 "
          f"(canonical offline_collateral_kl batch = 1100)", flush=True)
    if len(states) < 500:
        raise SystemExit(f"FATAL: only {len(states)} states")

    obs = {"observation": th.as_tensor(np.array([s["observation"] for s in states]), dtype=th.float32),
           "action_mask": th.as_tensor(np.array([s["action_mask"] for s in states]), dtype=th.float32)}
    cl = np.array(team_of)

    def logits_of(model, chunk=256):
        sp = model.observation_space.spaces
        out = []
        with th.no_grad():
            for i in range(0, obs["observation"].shape[0], chunk):
                o = {k: v[i:i + chunk] for k, v in obs.items() if k in sp}
                out.append(model.policy.get_distribution(o).distribution.logits)
        return th.cat(out, 0)

    p_log = logits_of(parent)

    def per_team_kl(model_path, tag):
        m = load(model_path)
        q = logits_of(m)
        # PRIMARY DIRECTION (pre-registered): forward KL(teacher || parent).
        fwd = masked_kl_rows(q, p_log, obs["action_mask"]).detach().cpu().numpy()
        # the anchor monitor's own direction, kept so the two are never confused
        rev = masked_kl_rows(p_log, q, obs["action_mask"]).detach().cpu().numpy()
        del m
        f = np.array([fwd[cl == t].mean() for t in range(len(TEAMS))])
        r = np.array([rev[cl == t].mean() for t in range(len(TEAMS))])
        print(f"  {tag:16s} KL_t||p  untaught {f[:8].mean():.4f}  taught16 {f[8:].mean():.4f}",
              flush=True)
        return f, r

    res = {"_meta": {
        "statistic": "forward KL(teacher||parent) over legal actions; masked_kl_rows IMPORTED "
                     "from agents.training.instrumented_ppo.distill_anchor",
        "also_reported": "KL(parent||teacher), the anchor monitor's own direction",
        "state_source": "PARENT pilots each of 24 teams vs rev-1's 24M snapshot, "
                        f"{per_team} battles/team, concurrency=1, seeded",
        "seeds": {"sim": "[team_index+1,2,3,4]", "pilot_policy": "71000+team_index",
                  "opponent_policy": "72000+team_index", "pool_sequence": "61000+team_index",
                  "note": "verbatim from reuse_batch offline_collateral_kl.py; the untaught 8 hold "
                          "indices 0..7 so their states reproduce that artifact's batch"},
        "teams": [{"i": i, "basename": b, "kind": k} for i, (b, k) in enumerate(TEAMS)],
        "n_states": len(states), "n_untaught_states": int(n_untaught_states),
        "states_per_team": [int((cl == t).sum()) for t in range(len(TEAMS))],
        "parent": PARENT, "opponent": OPPONENT, "config": CFG,
        "wall_s_states": round(time.time() - t0, 1)}}

    kl = {}
    for tag, path in [("FLOOR_ckptA", PAR_CKPT_A), ("FLOOR_ckptB", PAR_CKPT_B)]:
        kl[tag] = per_team_kl(path, tag)
    for pid, unf, fund, _bs in PAIRS:
        kl[f"UNF{pid}"] = per_team_kl(f"{unf}/final_model.zip", f"UNF{pid}")
        kl[f"FUND{pid}"] = per_team_kl(f"{fund}/final_model.zip", f"FUND{pid}")

    # ACID: every teacher must be a DISTINCT network from every other, else a mis-resolved path
    # reads as a perfect null. Cheap surrogate: no two per-team KL vectors may be identical.
    vecs = {k: v[0] for k, v in kl.items()}
    dup = [(a, b) for i, a in enumerate(vecs) for b in list(vecs)[i + 1:]
           if np.allclose(vecs[a], vecs[b], atol=1e-9)]
    res["_meta"]["acid_all_distinct"] = not dup
    res["_meta"]["acid_duplicates"] = [f"{a}|{b}" for a, b in dup]
    if dup:
        print(f"  !! ACID: duplicate KL vectors {dup}", flush=True)

    res["per_team_kl_fwd"] = {k: [float(x) for x in v[0]] for k, v in kl.items()}
    res["per_team_kl_rev"] = {k: [float(x) for x in v[1]] for k, v in kl.items()}
    json.dump(res, open(out_path, "w"), indent=1)

    # ---------------------------------------------------------------- floor ------------------
    rng = np.random.default_rng(20260905)
    bs8 = rng.integers(0, 8, (20000, 8))            # over the 8 untaught teams
    bs16 = rng.integers(0, 16, (20000, 16))         # over the 16 taught teams
    bs8t = rng.integers(0, 8, (20000, 8))           # over the 8 teachers of a half

    fl = {}
    for tag in ("FLOOR_ckptA", "FLOOR_ckptB"):
        f = kl[tag][0]
        fl[tag] = {"untaught_mean": float(f[:8].mean()),
                   "untaught_ci95": list(boot_ci(f[:8], bs8)),
                   "taught16_mean": float(f[8:].mean()),
                   "taught16_ci95": list(boot_ci(f[8:], bs16)),
                   "L_floor": float(f[8:].mean() / f[:8].mean())}
    res["floor"] = fl
    print(f"\n  FLOOR ckpt_28067760 vs parent_final: untaught {fl['FLOOR_ckptA']['untaught_mean']:.4f} "
          f" taught {fl['FLOOR_ckptA']['taught16_mean']:.4f}", flush=True)
    print(f"  FLOOR ckpt_27917760 vs parent_final: untaught {fl['FLOOR_ckptB']['untaught_mean']:.4f} "
          f" taught {fl['FLOOR_ckptB']['taught16_mean']:.4f}", flush=True)

    # -------------------------------------------------------- PRIMARY A ----------------------
    per_teacher = {}
    for half, pref in (("unfunded", "UNF"), ("funded", "FUND")):
        for pid, _u, _f, bs in PAIRS:
            f = kl[f"{pref}{pid}"][0]
            own = np.array([f[IDX[b]] for b in bs])
            unt = f[:8]
            per_teacher[f"{pref}{pid}"] = {
                "half": half, "taught_teams": bs,
                "kl_taught": float(own.mean()), "kl_taught_per_team": [float(x) for x in own],
                "kl_untaught": float(unt.mean()), "kl_untaught_per_team": [float(x) for x in unt],
                "L": float(own.mean() / unt.mean())}
    res["primary_A_per_teacher"] = per_teacher

    groups = {}
    for half, pref in (("unfunded", "UNF"), ("funded", "FUND")):
        Ls = np.array([per_teacher[f"{pref}{p}"]["L"] for p, *_ in PAIRS])
        kt = np.array([per_teacher[f"{pref}{p}"]["kl_taught"] for p, *_ in PAIRS])
        ku = np.array([per_teacher[f"{pref}{p}"]["kl_untaught"] for p, *_ in PAIRS])
        groups[half] = {"L_mean": float(Ls.mean()), "L_ci95": list(boot_ci(Ls, bs8t)),
                        "L_per_teacher": [float(x) for x in Ls],
                        "kl_taught_mean": float(kt.mean()), "kl_taught_ci95": list(boot_ci(kt, bs8t)),
                        "kl_untaught_mean": float(ku.mean()), "kl_untaught_ci95": list(boot_ci(ku, bs8t))}
    dL = np.array([per_teacher[f"FUND{p}"]["L"] - per_teacher[f"UNF{p}"]["L"] for p, *_ in PAIRS])
    lo, hi = boot_ci(dL, bs8t)
    groups["FUNDED_minus_UNFUNDED_L"] = {"delta": float(dL.mean()), "ci95": [lo, hi],
                                         "paired_on": "the 8 teacher PAIRS (same 2 teams each)",
                                         "separates_from_zero": not (lo <= 0 <= hi)}
    dU = np.array([per_teacher[f"FUND{p}"]["kl_untaught"] - per_teacher[f"UNF{p}"]["kl_untaught"]
                   for p, *_ in PAIRS])
    lo2, hi2 = boot_ci(dU, bs8t)
    groups["FUNDED_minus_UNFUNDED_kl_untaught"] = {"delta": float(dU.mean()), "ci95": [lo2, hi2],
                                                   "separates_from_zero": not (lo2 <= 0 <= hi2)}
    dT = np.array([per_teacher[f"FUND{p}"]["kl_taught"] - per_teacher[f"UNF{p}"]["kl_taught"]
                   for p, *_ in PAIRS])
    lo3, hi3 = boot_ci(dT, bs8t)
    groups["FUNDED_minus_UNFUNDED_kl_taught"] = {"delta": float(dT.mean()), "ci95": [lo3, hi3],
                                                 "separates_from_zero": not (lo3 <= 0 <= hi3)}
    res["primary_A_groups"] = groups

    # -------------------------------------------------------- PRIMARY B ----------------------
    # For each taught team: the KL of the teacher that TAUGHT it vs the mean KL of the 7 same-half
    # siblings on the SAME states. State distribution, team and recipe held fixed.
    B = {}
    for half, pref in (("unfunded", "UNF"), ("funded", "FUND")):
        own, sib = [], []
        for pid, _u, _f, bs in PAIRS:
            for b in bs:
                i = IDX[b]
                own.append(kl[f"{pref}{pid}"][0][i])
                sib.append(np.mean([kl[f"{pref}{q}"][0][i] for q, *_ in PAIRS if q != pid]))
        own, sib = np.array(own), np.array(sib)
        d = own - sib
        r = own / sib
        lo, hi = boot_ci(d, bs16); rlo, rhi = boot_ci(r, bs16)
        B[half] = {"kl_own_mean": float(own.mean()), "kl_siblings_mean": float(sib.mean()),
                   "delta_own_minus_siblings": float(d.mean()), "delta_ci95": [lo, hi],
                   "delta_separates_from_zero": not (lo <= 0 <= hi),
                   "R_ratio_mean": float(r.mean()), "R_ci95": [rlo, rhi],
                   "per_team_own": [float(x) for x in own],
                   "per_team_siblings": [float(x) for x in sib]}
    rf = np.array(B["funded"]["per_team_own"]) / np.array(B["funded"]["per_team_siblings"])
    ru = np.array(B["unfunded"]["per_team_own"]) / np.array(B["unfunded"]["per_team_siblings"])
    lo, hi = boot_ci(rf - ru, bs16)
    B["FUNDED_minus_UNFUNDED_R"] = {"delta": float((rf - ru).mean()), "ci95": [lo, hi],
                                    "separates_from_zero": not (lo <= 0 <= hi)}
    res["primary_B_sibling_control"] = B

    json.dump(res, open(out_path, "w"), indent=1)
    print("\n  === PRIMARY A (per-teacher L = KL_taught / KL_untaught) ===", flush=True)
    for half in ("unfunded", "funded"):
        g = groups[half]
        print(f"  {half:9s} KL_taught {g['kl_taught_mean']:.4f}  KL_untaught {g['kl_untaught_mean']:.4f}"
              f"  L {g['L_mean']:.4f} CI [{g['L_ci95'][0]:.4f},{g['L_ci95'][1]:.4f}]", flush=True)
    g = groups["FUNDED_minus_UNFUNDED_L"]
    print(f"  FUNDED-UNFUNDED L: {g['delta']:+.4f} CI [{g['ci95'][0]:+.4f},{g['ci95'][1]:+.4f}] "
          f"{'SEPARATES' if g['separates_from_zero'] else 'SPANS ZERO'}", flush=True)
    print("\n  === PRIMARY B (sibling control, 16 taught teams) ===", flush=True)
    for half in ("unfunded", "funded"):
        b = B[half]
        print(f"  {half:9s} own {b['kl_own_mean']:.4f}  siblings {b['kl_siblings_mean']:.4f}  "
              f"delta {b['delta_own_minus_siblings']:+.4f} CI "
              f"[{b['delta_ci95'][0]:+.4f},{b['delta_ci95'][1]:+.4f}]  R {b['R_ratio_mean']:.4f} "
              f"CI [{b['R_ci95'][0]:.4f},{b['R_ci95'][1]:.4f}]", flush=True)
    b = B["FUNDED_minus_UNFUNDED_R"]
    print(f"  FUNDED-UNFUNDED R: {b['delta']:+.4f} CI [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] "
          f"{'SEPARATES' if b['separates_from_zero'] else 'SPANS ZERO'}", flush=True)
    print(f"\n  wrote {out_path}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
