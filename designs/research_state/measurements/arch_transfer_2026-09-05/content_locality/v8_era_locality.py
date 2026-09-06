"""CONTENT LOCALITY, v8 ERA — the same measurement as gen_era_locality.py, at commit b13b30b2.

MUST BE RUN FROM AN ERA-PINNED CHECKOUT (`PYTHONPATH=<era>/src`). The v8 checkpoints are
config_version 45 / obs 2992 and current code REFUSES them.

Statistic: forward KL(teacher || parent) over LEGAL actions. The era predates
`instrumented_ppo.distill_anchor`, so the function comes from `era_kl.masked_kl_rows_era` -- a
verbatim copy whose bit-identity with the gen-era import is gated by `kl_unit_test.py`.

State batch: the FOLD PARENT (ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip) pilots
each team against the fixed reference opponent (ai_v8_03_zarch_control_0718, an ancestor of both
arms and equal to neither -- probe P's reference, reused).

ERA DETERMINISM. `$GEN3AI_{PLAYER,TEAM,POLICY,POOL,STALLER}_SEED` DO NOT EXIST at b13b30b2 (they
landed 2026-08-30). Determinism here comes from the same three things every era probe uses:
`stochastic=False` on both sides (no policy draw), a PINNED single team on the pilot, and an
EXPLICIT 4-int sim seed per team. The opponent's team sequence is drawn from a per-team
`random.Random(61000+i)`, which is process-local and unaffected by the missing env vars. The era's
rust bridge predates the seedless-seed fix (`bc00d4d`), so the NODE bridge is mandatory.

TEAMS. Taught = the union of the three v8_14 teachers' recorded `--trainee-teams`, resolved by
sha10 against the era pool (never hand-typed content). Untaught = the first 8 of probe P's
pre-registered `probe_untaught` set, asserted disjoint from that union.

Run (from the era tree):
  PYTHONPATH=/tmp/v8rep_era/src PYTHONDONTWRITEBYTECODE=1 nice -n 10 \
      python v8_era_locality.py <out.json> [battles_per_team=3]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import asyncio, hashlib, itertools, json, random, sys, time
import numpy as np
import torch as th
th.set_num_threads(1)
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from agents.inference.player import RLPlayer
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from era_kl import masked_kl_rows_era as masked_kl_rows   # noqa: E402

MAIN = "/home/goodlad/dev/gen3ai"
# The era tree, whose data/ the era pool comes from. TeamLoader resolves "data/teams" relative to
# the CWD, so this must be the tree the process was started in.
ERA_ROOT = os.environ.get("ERA_ROOT", os.getcwd())
MD = f"{MAIN}/models"
PAR_RUN = f"{MD}/ai_v8_04_distill_4teacher_0722"
PARENT = f"{PAR_RUN}/final_model_interrupted.zip"
PAR_CFG = f"{PAR_RUN}/model_config.json"
REF = f"{MD}/ai_v8_03_zarch_control_0718/final_model_interrupted.zip"
REF_CFG = f"{MD}/ai_v8_03_zarch_control_0718/model_config.json"

# MATCHED-NOISE FLOOR: the parent run's own two nearest retained checkpoints. The parent's final is
# 277,583,267 steps, so these are -405k and -1.82M from it.
FLOORS = [("FLOOR_c277178", f"{PAR_RUN}/checkpoints/checkpoint_277178472_steps.zip", PAR_CFG),
          ("FLOOR_c275758", f"{PAR_RUN}/checkpoints/checkpoint_275758296_steps.zip", PAR_CFG)]

# v8_14's THREE teachers, with each one's recorded --trainee-teams resolved to sha10.
# `pool10` / `defensive10` carry the sha10 in the filename their argv names; `semistall3` names
# three data/teams/sample files, whose sha10 is computed from the file at run time.
TEACHERS = [
    ("pool10", f"{MD}/ai_v8_09_pool10_exploiter_0723/final_model_interrupted.zip",
     f"{MD}/ai_v8_09_pool10_exploiter_0723/model_config.json",
     {"sha": ["564b9be3ae", "d3c1cd0952", "7594a34f82", "47dc388b25", "4c9552cd01",
              "552d5857a3", "f5d46ca0fc", "3a83154c2a", "9b454d9ea7", "24db4aacdd"]}),
    ("semistall3", f"{MD}/ai_v8_06_semistall_3team_exploiter_0722/final_model_interrupted.zip",
     f"{MD}/ai_v8_06_semistall_3team_exploiter_0722/model_config.json",
     {"files": [f"{ERA_ROOT}/data/teams/sample/9d5f845869e899ee.txt",
                f"{ERA_ROOT}/data/teams/sample/f7ba5702fe856292.txt",
                f"{ERA_ROOT}/data/teams/sample/0972146213a667c9.txt"]}),
    ("defensive10", f"{MD}/ai_v8_13_defensive10_exploiter_0725/final_model_interrupted.zip",
     f"{MD}/ai_v8_13_defensive10_exploiter_0725/model_config.json",
     {"sha": ["9278913bce", "fc908f1bf4", "83aee9db7e", "044da80d78", "bef089d2cf",
              "3f95b25e9a", "8b6b8c8f52", "5c88ff9ca5", "3e9bdcee48", "65bfb2e8b4"]}),
]

# probe P's pre-registered untaught set (first 8), verbatim from v8_gift_timing_probe.py.
UNTAUGHT_SHA = ["d0a4d2bcb8", "c90e782cad", "a6b630e6b4", "a577a735b7",
                "9292a21833", "eaa88395e7", "7c2cb5cec1", "89fcef3b53"]

_ACCT = itertools.count(1)


def sha10(s):
    return hashlib.sha1(s.strip().encode()).hexdigest()[:10]


def _acct(tag):
    return AccountConfiguration(f"CL{tag[:2]}{next(_ACCT):05d}", "pw")


def _strip_debugger(m):
    obj = getattr(m, "policy", m)
    for mod in (obj.modules() if hasattr(obj, "modules") else []):
        if getattr(mod, "_debugger", None) is not None:
            mod._debugger = None
    if hasattr(m, "policy"):
        m.policy.set_training_mode(False)
    return m


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
            self.captured.append({"observation": np.asarray(d["observation"]).copy(),
                                  "action_mask": np.asarray(d["action_mask"]).copy()})
        return d


def boot_ci(vals, idx):
    b = np.asarray(vals)[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main(out_path, per_team=3):
    maps = load_mappings(); cv = current_model_version(maps)

    def load(p, cfg):
        m, _ = load_foreign_opponent(p, current_version=cv, device="cpu", config_path=cfg)
        return _strip_debugger(m)

    pool_teams = TeamLoader().get_all_teams()
    by_sha = {sha10(t): t for t in pool_teams}

    taught_of = {}
    for name, _p, _c, spec in TEACHERS:
        if "sha" in spec:
            shas = list(spec["sha"])
        else:
            shas = [sha10(open(f).read()) for f in spec["files"]]
        missing = [s for s in shas if s not in by_sha]
        if missing:
            raise SystemExit(f"[GIGO] teacher {name}: {missing} not resolvable in the era pool")
        taught_of[name] = shas
    # MEASURED, not assumed: the three teachers' taught sets OVERLAP. ai_v8_06's
    # data/teams/sample/9d5f845869e899ee.txt hashes to 564b9be3ae, which is also ai_v8_09's t00.
    # So the union is DEDUPED here, and the sibling control below uses only SINGLY-taught teams --
    # a team two of the three teachers saw has no clean "did not teach it" sibling on this side.
    seen, taught_union = set(), []
    for n in taught_of:
        for s in taught_of[n]:
            if s not in seen:
                seen.add(s); taught_union.append(s)
    shared = sorted({s for s in seen if sum(s in taught_of[n] for n in taught_of) > 1})
    if shared:
        print(f"[v8] NOTE: {len(shared)} team(s) taught by >1 teacher: {shared} "
              f"-- excluded from the sibling control", flush=True)
    bad = [s for s in UNTAUGHT_SHA if s in set(taught_union)]
    if bad:
        raise SystemExit(f"[GIGO] untaught set overlaps the taught union: {bad}")
    miss = [s for s in UNTAUGHT_SHA if s not in by_sha]
    if miss:
        raise SystemExit(f"[GIGO] untaught {miss} not in the era pool")

    TEAMS = [(s, "untaught") for s in UNTAUGHT_SHA] + [(s, "taught") for s in taught_union]
    IDX = {s: i for i, (s, _) in enumerate(TEAMS)}
    n_unt = len(UNTAUGHT_SHA)
    print(f"[v8] {len(TEAMS)} teams = {n_unt} untaught + {len(taught_union)} taught "
          f"({ {k: len(v) for k, v in taught_of.items()} }), pool {len(pool_teams)}", flush=True)

    parent = load(PARENT, PAR_CFG)
    ref = load(REF, REF_CFG)
    pool = PairedPool(pool_teams); n_pool = len(pool.packed_teams)

    t0 = time.time(); states = []; team_of = []
    for ti, (s, kind) in enumerate(TEAMS):
        rng = random.Random(61000 + ti)
        seq = [rng.randrange(n_pool) for _ in range(per_team)]
        pilot = Capturing(model=parent, team=Gen3Teambuilder([by_sha[s]]), battle_format="gen3ou",
                          server_configuration=LocalhostServerConfiguration, mappings=maps,
                          account_configuration=_acct("pi"), stochastic=False,
                          start_listening=False)
        opp = RLPlayer(model=ref, team=pool.set_sequence(seq), battle_format="gen3ou",
                       server_configuration=LocalhostServerConfiguration, mappings=maps,
                       account_configuration=_acct("op"), stochastic=False,
                       start_listening=False)
        pilot.reset_battles(); opp.reset_battles()
        asyncio.run(run_local_battles(pilot, opp, per_team, concurrency=1, impl="node",
                                      seed=[ti + 1, 2, 3, 4]))
        states.extend(pilot.captured); team_of.extend([ti] * len(pilot.captured))
        print(f"  [{ti:2d}] {kind:8s} {s} +{len(pilot.captured):4d} states "
              f"(total {len(states)}, {time.time()-t0:.0f}s)", flush=True)
    if len(states) < 400:
        raise SystemExit(f"FATAL: only {len(states)} states")

    obs = {"observation": th.as_tensor(np.array([x["observation"] for x in states]), dtype=th.float32),
           "action_mask": th.as_tensor(np.array([x["action_mask"] for x in states]), dtype=th.float32)}
    cl = np.array(team_of)

    def logits_of(model, chunk=128):
        sp = model.observation_space.spaces
        out = []
        with th.no_grad():
            for i in range(0, obs["observation"].shape[0], chunk):
                o = {k: v[i:i + chunk] for k, v in obs.items() if k in sp}
                out.append(model.policy.get_distribution(o).distribution.logits)
        return th.cat(out, 0)

    p_log = logits_of(parent)

    kl = {}

    def score(tag, path, cfg):
        m = load(path, cfg)
        q = logits_of(m)
        fwd = masked_kl_rows(q, p_log, obs["action_mask"]).detach().cpu().numpy()
        rev = masked_kl_rows(p_log, q, obs["action_mask"]).detach().cpu().numpy()
        del m
        f = np.array([fwd[cl == t].mean() for t in range(len(TEAMS))])
        r = np.array([rev[cl == t].mean() for t in range(len(TEAMS))])
        kl[tag] = (f, r)
        print(f"  {tag:16s} KL_t||p  untaught {f[:n_unt].mean():.4f}  "
              f"taught{len(taught_union)} {f[n_unt:].mean():.4f}", flush=True)

    for tag, path, cfg in FLOORS:
        score(tag, path, cfg)
    for name, path, cfg, _spec in TEACHERS:
        score(name, path, cfg)

    vecs = {k: v[0] for k, v in kl.items()}
    dup = [(a, b) for i, a in enumerate(vecs) for b in list(vecs)[i + 1:]
           if np.allclose(vecs[a], vecs[b], atol=1e-9)]
    if dup:
        print(f"  !! ACID: duplicate KL vectors {dup}", flush=True)

    res = {"_meta": {
        "era": "v8 (b13b30b289c5eaba136a930a4ab63451e209fbe5)",
        "statistic": "forward KL(teacher||parent) over legal actions; era_kl.masked_kl_rows_era, "
                     "gated bit-identical to the gen-era import by kl_unit_test.py",
        "also_reported": "KL(parent||teacher)",
        "state_source": f"PARENT pilots each of {len(TEAMS)} teams vs the fixed reference "
                        f"ai_v8_03_zarch_control final, {per_team} battles/team, GREEDY both "
                        "sides, concurrency=1, node bridge",
        "determinism": "stochastic=False both sides + pinned pilot team + explicit 4-int sim "
                       "seed; the five GEN3AI_*_SEED env vars DO NOT EXIST at this commit",
        "seeds": {"sim": "[team_index+1,2,3,4]", "pool_sequence": "61000+team_index"},
        "teams": [{"i": i, "sha10": s, "kind": k} for i, (s, k) in enumerate(TEAMS)],
        "taught_of": taught_of, "n_states": len(states),
        "states_per_team": [int((cl == t).sum()) for t in range(len(TEAMS))],
        "parent": PARENT, "reference_opponent": REF,
        "acid_all_distinct": not dup, "acid_duplicates": [f"{a}|{b}" for a, b in dup],
        "wall_s_states": round(time.time() - t0, 1)}}
    res["per_team_kl_fwd"] = {k: [float(x) for x in v[0]] for k, v in kl.items()}
    res["per_team_kl_rev"] = {k: [float(x) for x in v[1]] for k, v in kl.items()}
    json.dump(res, open(out_path, "w"), indent=1)

    rng = np.random.default_rng(20260905)
    nT = len(taught_union)
    bsU = rng.integers(0, n_unt, (20000, n_unt))
    bsT = rng.integers(0, nT, (20000, nT))

    fl = {}
    for tag, _p, _c in FLOORS:
        f = kl[tag][0]
        fl[tag] = {"untaught_mean": float(f[:n_unt].mean()),
                   "untaught_ci95": list(boot_ci(f[:n_unt], bsU)),
                   "taught_mean": float(f[n_unt:].mean()),
                   "taught_ci95": list(boot_ci(f[n_unt:], bsT)),
                   "L_floor": float(f[n_unt:].mean() / f[:n_unt].mean())}
    res["floor"] = fl
    print(f"\n  FLOOR {FLOORS[0][0]}: untaught {fl[FLOORS[0][0]]['untaught_mean']:.4f}  "
          f"taught {fl[FLOORS[0][0]]['taught_mean']:.4f}", flush=True)
    print(f"  FLOOR {FLOORS[1][0]}: untaught {fl[FLOORS[1][0]]['untaught_mean']:.4f}  "
          f"taught {fl[FLOORS[1][0]]['taught_mean']:.4f}", flush=True)

    # --------------------------------------------------------- PRIMARY A ---------------------
    per_teacher = {}
    for name, _p, _c, _s in TEACHERS:
        f = kl[name][0]
        own = np.array([f[IDX[s]] for s in taught_of[name]])
        unt = f[:n_unt]
        bso = rng.integers(0, len(own), (20000, len(own)))
        Lb = own[bso].mean(axis=1) / unt[bsU].mean(axis=1)
        per_teacher[name] = {"n_taught": len(own),
                             "kl_taught": float(own.mean()),
                             "kl_taught_per_team": [float(x) for x in own],
                             "kl_untaught": float(unt.mean()),
                             "kl_untaught_per_team": [float(x) for x in unt],
                             "L": float(own.mean() / unt.mean()),
                             "L_ci95": [float(np.percentile(Lb, 2.5)),
                                        float(np.percentile(Lb, 97.5))]}
    res["primary_A_per_teacher"] = per_teacher

    own_all = np.array([kl[n][0][IDX[s]] for n, _p, _c, _sp in TEACHERS for s in taught_of[n]])
    unt_all = np.array([np.mean([kl[n][0][i] for n, _p, _c, _sp in TEACHERS])
                        for i in range(n_unt)])
    Lb = own_all[bsT].mean(axis=1) / unt_all[bsU].mean(axis=1)
    res["primary_A_era"] = {"kl_taught_mean": float(own_all.mean()),
                            "kl_taught_ci95": list(boot_ci(own_all, bsT)),
                            "kl_untaught_mean": float(unt_all.mean()),
                            "kl_untaught_ci95": list(boot_ci(unt_all, bsU)),
                            "L": float(own_all.mean() / unt_all.mean()),
                            "L_ci95": [float(np.percentile(Lb, 2.5)),
                                       float(np.percentile(Lb, 97.5))]}

    # --------------------------------------------------------- PRIMARY B ---------------------
    own, sib, used = [], [], []
    for name, _p, _c, _sp in TEACHERS:
        for s in taught_of[name]:
            if s in shared:            # no clean sibling exists for a doubly-taught team
                continue
            i = IDX[s]
            own.append(kl[name][0][i])
            sib.append(np.mean([kl[o][0][i] for o, _q, _r, _t in TEACHERS if o != name]))
            used.append(s)
    own, sib = np.array(own), np.array(sib)
    bsB = rng.integers(0, len(own), (20000, len(own)))
    d, r = own - sib, own / sib
    lo, hi = boot_ci(d, bsB); rlo, rhi = boot_ci(r, bsB)
    res["primary_B_sibling_control"] = {
        "n_taught_teams": len(own), "excluded_shared": shared,
        "teams_used": used, "n_siblings": len(TEACHERS) - 1,
        "kl_own_mean": float(own.mean()), "kl_siblings_mean": float(sib.mean()),
        "delta_own_minus_siblings": float(d.mean()), "delta_ci95": [lo, hi],
        "delta_separates_from_zero": not (lo <= 0 <= hi),
        "R_ratio_mean": float(r.mean()), "R_ci95": [rlo, rhi],
        "per_team_own": [float(x) for x in own], "per_team_siblings": [float(x) for x in sib]}

    json.dump(res, open(out_path, "w"), indent=1)
    print("\n  === PRIMARY A (per-teacher L = KL_taught / KL_untaught) ===", flush=True)
    for name in per_teacher:
        p = per_teacher[name]
        print(f"  {name:12s} n={p['n_taught']:2d}  KL_taught {p['kl_taught']:.4f}  "
              f"KL_untaught {p['kl_untaught']:.4f}  L {p['L']:.4f} "
              f"CI [{p['L_ci95'][0]:.4f},{p['L_ci95'][1]:.4f}]", flush=True)
    e = res["primary_A_era"]
    print(f"  ERA POOLED   KL_taught {e['kl_taught_mean']:.4f}  KL_untaught "
          f"{e['kl_untaught_mean']:.4f}  L {e['L']:.4f} "
          f"CI [{e['L_ci95'][0]:.4f},{e['L_ci95'][1]:.4f}]", flush=True)
    b = res["primary_B_sibling_control"]
    print(f"\n  === PRIMARY B (sibling control, {b['n_taught_teams']} singly-taught teams, "
          f"{b['n_siblings']} siblings) ===")
    print(f"  own {b['kl_own_mean']:.4f}  siblings {b['kl_siblings_mean']:.4f}  delta "
          f"{b['delta_own_minus_siblings']:+.4f} CI [{b['delta_ci95'][0]:+.4f},"
          f"{b['delta_ci95'][1]:+.4f}]  R {b['R_ratio_mean']:.4f} "
          f"CI [{b['R_ci95'][0]:.4f},{b['R_ci95'][1]:.4f}]", flush=True)
    print(f"\n  wrote {out_path}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
