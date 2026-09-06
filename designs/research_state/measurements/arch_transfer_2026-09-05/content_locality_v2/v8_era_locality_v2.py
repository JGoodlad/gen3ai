"""CONTENT LOCALITY v2, v8 ERA — content_locality/v8_era_locality.py with the same corrections.

MUST BE RUN FROM THE ERA-PINNED CHECKOUT (`PYTHONPATH=<era>/src`). The v8 checkpoints are
config_version 45 / obs 2992 and current code REFUSES them.

CORRECTION 1 — CHECKPOINT RESOLUTION. v1 scored `{run}/final_model_interrupted.zip`, which is not
  a rung of the training path's resolver at all. The era checkout carries the SAME
  `agents.training.fixed_opponent_pool._resolve_zip_and_config` as the gen tree (verified by
  diffing the function), and `ai_v8_14_distill3_0725`'s recorded `--distill-teacher` names three
  RUN DIRECTORIES, so the fold loaded `best_model/best_model.zip` for all three. That resolver is
  IMPORTED here. It also returns the config next to the zip (`best_model/model_config.json`), which
  is what the fold used; the config only feeds the arch-signature check, never the network.

CORRECTION 2 — REFERENCE. v8's three teachers fork FROM `ai_v8_04_distill_4teacher_0722/
  final_model_interrupted.zip` — the fold parent itself (verified with `python -m main.lineage`).
  So the fold parent IS the true origin here and the two references of the gen arm COINCIDE. One
  column, stated as such.

CORRECTION 3 — the cluster bootstrap is sized from its own array (see boot.py). v1's `own_all`
  held 23 (teacher, taught team) cells while its `bsT` drew in [0, 22), so `primary_A_era`'s
  pooled-L CI silently dropped `defensive10`'s last team. The headline sibling-control R was
  correctly sized in v1 and is unaffected in point estimate.

Everything else — teams, seeds, greedy play, node bridge, concurrency=1, the era KL copy — is
verbatim from v1.

Run (from the era tree):
  cd /tmp/v8rep_era && PYTHONPATH=/tmp/v8rep_era/src PYTHONDONTWRITEBYTECODE=1 \
    ERA_ROOT=/tmp/v8rep_era GEN3AI_TIMEOUT_SCALE=12 nice -n 10 \
    python <this> <out.json> [battles_per_team=3]
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
from agents.training.fixed_opponent_pool import _resolve_zip_and_config
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "content_locality"))
from era_kl import masked_kl_rows_era as masked_kl_rows   # noqa: E402
from boot import Boot                                     # noqa: E402

MAIN = "/home/goodlad/dev/gen3ai"
ERA_ROOT = os.environ.get("ERA_ROOT", os.getcwd())
MD = f"{MAIN}/models"
PAR_RUN = f"{MD}/ai_v8_04_distill_4teacher_0722"
PARENT = f"{PAR_RUN}/final_model_interrupted.zip"      # REF — the fold's --model AND the origin
PAR_CFG = f"{PAR_RUN}/model_config.json"
REF = f"{MD}/ai_v8_03_zarch_control_0718/final_model_interrupted.zip"
REF_CFG = f"{MD}/ai_v8_03_zarch_control_0718/model_config.json"

FLOORS = [("FLOOR_c277178", f"{PAR_RUN}/checkpoints/checkpoint_277178472_steps.zip", PAR_CFG),
          ("FLOOR_c275758", f"{PAR_RUN}/checkpoints/checkpoint_275758296_steps.zip", PAR_CFG)]

# v8_14's THREE teachers as RUN DIRECTORIES (v1 named .zip files), plus each one's recorded
# --trainee-teams resolved to sha10 exactly as v1 did.
TEACHERS = [
    ("pool10", f"{MD}/ai_v8_09_pool10_exploiter_0723",
     {"sha": ["564b9be3ae", "d3c1cd0952", "7594a34f82", "47dc388b25", "4c9552cd01",
              "552d5857a3", "f5d46ca0fc", "3a83154c2a", "9b454d9ea7", "24db4aacdd"]}),
    ("semistall3", f"{MD}/ai_v8_06_semistall_3team_exploiter_0722",
     {"files": [f"{ERA_ROOT}/data/teams/sample/9d5f845869e899ee.txt",
                f"{ERA_ROOT}/data/teams/sample/f7ba5702fe856292.txt",
                f"{ERA_ROOT}/data/teams/sample/0972146213a667c9.txt"]}),
    ("defensive10", f"{MD}/ai_v8_13_defensive10_exploiter_0725",
     {"sha": ["9278913bce", "fc908f1bf4", "83aee9db7e", "044da80d78", "bef089d2cf",
              "3f95b25e9a", "8b6b8c8f52", "5c88ff9ca5", "3e9bdcee48", "65bfb2e8b4"]}),
]

UNTAUGHT_SHA = ["d0a4d2bcb8", "c90e782cad", "a6b630e6b4", "a577a735b7",
                "9292a21833", "eaa88395e7", "7c2cb5cec1", "89fcef3b53"]
# The published per-team untaught state counts of this batch, asserted rather than hoped for.
# Read from content_locality/v8_era_n{3,9}.json["_meta"]["states_per_team"][:8].
EXPECT = {3: [109, 104, 98, 96, 88, 80, 92, 78],
          9: [266, 255, 260, 312, 270, 265, 303, 259]}

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


def main(out_path, per_team=3):
    maps = load_mappings(); cv = current_model_version(maps)

    def load(p, cfg):
        m, _ = load_foreign_opponent(p, current_version=cv, device="cpu", config_path=cfg)
        return _strip_debugger(m)

    # CORRECTION 1: the training path's own resolver, IMPORTED.
    resolved = {}
    for name, run, _spec in TEACHERS:
        z, c, _b = _resolve_zip_and_config(run, None)
        v1 = os.path.join(run, "final_model_interrupted.zip")
        resolved[name] = {"run": os.path.relpath(run, MD), "zip": os.path.relpath(z, MD),
                          "config": os.path.relpath(c, MD),
                          "v1_scored": os.path.relpath(v1, MD)}
        print(f"[v8] RESOLVE {name}: {resolved[name]['zip']}  (v1 scored "
              f"{os.path.basename(v1)})", flush=True)

    pool_teams = TeamLoader().get_all_teams()
    by_sha = {sha10(t): t for t in pool_teams}

    taught_of = {}
    for name, _run, spec in TEACHERS:
        shas = list(spec["sha"]) if "sha" in spec else [sha10(open(f).read())
                                                        for f in spec["files"]]
        missing = [s for s in shas if s not in by_sha]
        if missing:
            raise SystemExit(f"[GIGO] teacher {name}: {missing} not resolvable in the era pool")
        taught_of[name] = shas
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
    per_team_counts = [int(sum(1 for t in team_of if t == i)) for i in range(len(TEAMS))]
    if per_team in EXPECT:
        got, exp = per_team_counts[:n_unt], EXPECT[per_team]
        print(f"  UNTAUGHT CROSS-CHECK  got {got}\n"
              f"                        exp {exp}  "
              f"{'REPRODUCED' if got == exp else 'MISMATCH'}", flush=True)
        assert got == exp, "did NOT reproduce content_locality's v8 untaught batch"
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
    for name, _run, _spec in TEACHERS:
        score(name, f"{MD}/{resolved[name]['zip']}", f"{MD}/{resolved[name]['config']}")

    vecs = {k: v[0] for k, v in kl.items()}
    dup = [(a, b) for i, a in enumerate(vecs) for b in list(vecs)[i + 1:]
           if np.allclose(vecs[a], vecs[b], atol=1e-9)]
    if dup:
        print(f"  !! ACID: duplicate KL vectors {dup}", flush=True)

    res = {"_meta": {
        "era": "v8 (b13b30b289c5eaba136a930a4ab63451e209fbe5)",
        "corrections_vs_v1": [
            "CHECKPOINT: teachers resolved by the era tree's own agents.training."
            "fixed_opponent_pool._resolve_zip_and_config(run_dir, None)",
            "REFERENCE: the fold parent IS the teachers' fork origin here (main.lineage), so the "
            "gen arm's REF-A and REF-B coincide — one column",
            "BOOTSTRAP: every CI resamples over the clusters of its own array (boot.py)"],
        "statistic": "forward KL(teacher||parent) over legal actions; era_kl.masked_kl_rows_era, "
                     "gated bit-identical to the gen-era import by kl_unit_test.py",
        "also_reported": "KL(parent||teacher)",
        "state_source": f"PARENT pilots each of {len(TEAMS)} teams vs the fixed reference "
                        f"ai_v8_03_zarch_control final, {per_team} battles/team, GREEDY both "
                        "sides, concurrency=1, node bridge (VERBATIM from v1)",
        "seeds": {"sim": "[team_index+1,2,3,4]", "pool_sequence": "61000+team_index"},
        "teams": [{"i": i, "sha10": s, "kind": k} for i, (s, k) in enumerate(TEAMS)],
        "taught_of": taught_of, "n_states": len(states),
        "states_per_team": per_team_counts,
        "parent_and_origin": PARENT, "reference_opponent": REF,
        "resolved_teachers": resolved,
        "acid_all_distinct": not dup, "acid_duplicates": [f"{a}|{b}" for a, b in dup],
        "wall_s_states": round(time.time() - t0, 1)}}
    res["per_team_kl_fwd"] = {k: [float(x) for x in v[0]] for k, v in kl.items()}
    res["per_team_kl_rev"] = {k: [float(x) for x in v[1]] for k, v in kl.items()}
    json.dump(res, open(out_path, "w"), indent=1)

    boot = Boot()
    fl = {}
    for tag, _p, _c in FLOORS:
        f = kl[tag][0]
        fl[tag] = {"untaught_mean": float(f[:n_unt].mean()),
                   "untaught_ci95": list(boot.ci(f[:n_unt])),
                   "taught_mean": float(f[n_unt:].mean()),
                   "taught_ci95": list(boot.ci(f[n_unt:])),
                   "L_floor": float(f[n_unt:].mean() / f[:n_unt].mean())}
        print(f"\n  FLOOR {tag}: untaught {fl[tag]['untaught_mean']:.4f}  "
              f"taught {fl[tag]['taught_mean']:.4f}  floor L {fl[tag]['L_floor']:.4f}", flush=True)
    res["floor"] = fl

    # --------------------------------------------------------- PRIMARY A ---------------------
    per_teacher = {}
    for name, _run, _spec in TEACHERS:
        f = kl[name][0]
        own = np.array([f[IDX[s]] for s in taught_of[name]])
        unt = f[:n_unt]
        Lb = own[boot.idx(len(own))].mean(axis=1) / unt[boot.idx(len(unt))].mean(axis=1)
        per_teacher[name] = {"n_taught": len(own),
                             "kl_taught": float(own.mean()),
                             "kl_taught_per_team": [float(x) for x in own],
                             "kl_untaught": float(unt.mean()),
                             "kl_untaught_per_team": [float(x) for x in unt],
                             "L": float(own.mean() / unt.mean()),
                             "L_ci95": [float(np.percentile(Lb, 2.5)),
                                        float(np.percentile(Lb, 97.5))]}
    res["primary_A_per_teacher"] = per_teacher

    own_all = np.array([kl[n][0][IDX[s]] for n, _r, _sp in TEACHERS for s in taught_of[n]])
    unt_all = np.array([np.mean([kl[n][0][i] for n, _r, _sp in TEACHERS]) for i in range(n_unt)])
    # CORRECTION 3: own_all has one cell per (teacher, taught team) = 23, NOT len(taught_union)=22.
    assert len(own_all) == sum(len(v) for v in taught_of.values())
    Lb = own_all[boot.idx(len(own_all))].mean(axis=1) / unt_all[boot.idx(len(unt_all))].mean(axis=1)
    res["primary_A_era"] = {"n_cells": int(len(own_all)),
                            "n_taught_union": len(taught_union),
                            "bootstrap_note": "resampled over all 23 (teacher, team) cells; v1 "
                                              "drew indices in [0,22) and dropped the last",
                            "kl_taught_mean": float(own_all.mean()),
                            "kl_taught_ci95": list(boot.ci(own_all)),
                            "kl_untaught_mean": float(unt_all.mean()),
                            "kl_untaught_ci95": list(boot.ci(unt_all)),
                            "L": float(own_all.mean() / unt_all.mean()),
                            "L_ci95": [float(np.percentile(Lb, 2.5)),
                                       float(np.percentile(Lb, 97.5))]}

    # --------------------------------------------------------- PRIMARY B ---------------------
    own, sib, used = [], [], []
    for name, _run, _spec in TEACHERS:
        for s in taught_of[name]:
            if s in shared:
                continue
            i = IDX[s]
            own.append(kl[name][0][i])
            sib.append(np.mean([kl[o][0][i] for o, _r, _t in TEACHERS if o != name]))
            used.append(s)
    own, sib = np.array(own), np.array(sib)
    d, r = own - sib, own / sib
    lo, hi = boot.ci(d); rlo, rhi = boot.ci(r)
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
          f"CI [{e['L_ci95'][0]:.4f},{e['L_ci95'][1]:.4f}]  ({e['n_cells']} cells)", flush=True)
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
