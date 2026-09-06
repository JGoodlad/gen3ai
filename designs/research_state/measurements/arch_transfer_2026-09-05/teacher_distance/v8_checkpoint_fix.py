"""v8 D_off ON THE CHECKPOINT THE FOLD ACTUALLY LOADED.

content_locality's v8 arm scored `final_model_interrupted.zip`. The training path resolves a
`--distill-teacher` run-dir through `fixed_opponent_pool._resolve_zip_and_config`, whose rungs are
`best_model/best_model.zip` -> `final_model.zip` -> `best_model.zip`. `final_model_interrupted.zip`
is NOT a rung, and all three v8 teachers have a `best_model/best_model.zip` with a different
sha256 -- so v8_14 distilled from `best_model/`, not from what was scored.

This regenerates ONLY the untaught-8 states (indices 0..7, 9 battles, greedy, node bridge -- the
era recipe verbatim) and scores BOTH variants of each teacher on them, so the correction is
MEASURED rather than inferred. The floor checkpoints are re-scored as a reproduction check.

MUST be run from the era checkout (b13b30b2):
  cd /tmp/v8rep_era && PYTHONPATH=/tmp/v8rep_era/src PYTHONDONTWRITEBYTECODE=1 \
    ERA_ROOT=/tmp/v8rep_era GEN3AI_TIMEOUT_SCALE=8 nice -n 10 python <this> <out.json>
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "content_locality"))
from era_kl import masked_kl_rows_era as masked_kl_rows   # noqa: E402

MAIN = "/home/goodlad/dev/gen3ai"
MD = f"{MAIN}/models"
PAR_RUN = f"{MD}/ai_v8_04_distill_4teacher_0722"
PARENT = f"{PAR_RUN}/final_model_interrupted.zip"
PAR_CFG = f"{PAR_RUN}/model_config.json"
REF = f"{MD}/ai_v8_03_zarch_control_0718/final_model_interrupted.zip"
REF_CFG = f"{MD}/ai_v8_03_zarch_control_0718/model_config.json"

# name -> run dir. Both checkpoint variants are scored.
TEACHER_RUNS = {
    "pool10": f"{MD}/ai_v8_09_pool10_exploiter_0723",
    "semistall3": f"{MD}/ai_v8_06_semistall_3team_exploiter_0722",
    "defensive10": f"{MD}/ai_v8_13_defensive10_exploiter_0725",
}
FLOORS = {"FLOOR_c277178": f"{PAR_RUN}/checkpoints/checkpoint_277178472_steps.zip",
          "FLOOR_c275758": f"{PAR_RUN}/checkpoints/checkpoint_275758296_steps.zip"}
UNTAUGHT_SHA = ["d0a4d2bcb8", "c90e782cad", "a6b630e6b4", "a577a735b7",
                "9292a21833", "eaa88395e7", "7c2cb5cec1", "89fcef3b53"]
EXPECT = [266, 255, 260, 312, 270, 265, 303, 259]      # v8_era_n9.json states_per_team[:8]

_ACCT = itertools.count(1)


def sha10(s):
    return hashlib.sha1(s.strip().encode()).hexdigest()[:10]


def _acct(t):
    return AccountConfiguration(f"V8{t[:2]}{next(_ACCT):05d}", "pw")


def _strip(m):
    for mod in getattr(m, "policy", m).modules():
        if getattr(mod, "_debugger", None) is not None:
            mod._debugger = None
    if hasattr(m, "policy"):
        m.policy.set_training_mode(False)
    return m


class PairedPool(Gen3Teambuilder):
    def __init__(self, t): super().__init__(t); self._seq, self._i = [], 0
    def set_sequence(self, s): self._seq, self._i = s, 0; return self
    def yield_team(self):
        t = self.packed_teams[self._seq[self._i % len(self._seq)]]; self._i += 1; return t


class Capturing(RLPlayer):
    def __init__(self, *a, **kw): super().__init__(*a, **kw); self.captured = []
    def embed_battle(self, b):
        d = super().embed_battle(b)
        if d is not None and d.get("action_mask") is not None and int(d["action_mask"].sum()) > 0:
            self.captured.append({"observation": np.asarray(d["observation"]).copy(),
                                  "action_mask": np.asarray(d["action_mask"]).copy()})
        return d


def main(out_path):
    maps = load_mappings(); cv = current_model_version(maps)

    def load(p, cfg):
        m, _ = load_foreign_opponent(p, current_version=cv, device="cpu", config_path=cfg)
        return _strip(m)

    by_sha = {sha10(t): t for t in TeamLoader().get_all_teams()}
    miss = [s for s in UNTAUGHT_SHA if s not in by_sha]
    if miss:
        raise SystemExit(f"[GIGO] untaught {miss} not in the era pool")

    parent = load(PARENT, PAR_CFG); ref = load(REF, REF_CFG)
    pool = PairedPool(list(by_sha.values())); n_pool = len(pool.packed_teams)

    t0 = time.time(); states = []; team_of = []
    for ti, s in enumerate(UNTAUGHT_SHA):
        rng = random.Random(61000 + ti)
        seq = [rng.randrange(n_pool) for _ in range(9)]
        pilot = Capturing(model=parent, team=Gen3Teambuilder([by_sha[s]]), battle_format="gen3ou",
                          server_configuration=LocalhostServerConfiguration, mappings=maps,
                          account_configuration=_acct("pi"), stochastic=False,
                          start_listening=False)
        opp = RLPlayer(model=ref, team=pool.set_sequence(seq), battle_format="gen3ou",
                       server_configuration=LocalhostServerConfiguration, mappings=maps,
                       account_configuration=_acct("op"), stochastic=False, start_listening=False)
        pilot.reset_battles(); opp.reset_battles()
        asyncio.run(run_local_battles(pilot, opp, 9, concurrency=1, impl="node",
                                      seed=[ti + 1, 2, 3, 4]))
        states.extend(pilot.captured); team_of.extend([ti] * len(pilot.captured))
        print(f"  [{ti}] {s} +{len(pilot.captured)} (total {len(states)}, "
              f"{time.time()-t0:.0f}s)", flush=True)

    counts = [int(sum(1 for t in team_of if t == i)) for i in range(8)]
    ok = counts == EXPECT
    print(f"  per-team counts {counts}\n  expected        {EXPECT}  "
          f"{'REPRODUCED' if ok else 'MISMATCH'}", flush=True)
    assert ok, "did NOT reproduce content_locality's v8 untaught batch -- refusing to report"

    obs = {"observation": th.as_tensor(np.array([s["observation"] for s in states]), dtype=th.float32),
           "action_mask": th.as_tensor(np.array([s["action_mask"] for s in states]), dtype=th.float32)}
    cl = np.array(team_of)

    def logits_of(m, chunk=256):
        sp = m.observation_space.spaces; out = []
        with th.no_grad():
            for i in range(0, obs["observation"].shape[0], chunk):
                o = {k: v[i:i + chunk] for k, v in obs.items() if k in sp}
                out.append(m.policy.get_distribution(o).distribution.logits)
        return th.cat(out, 0)

    p_log = logits_of(parent)

    def score(path, cfg):
        m = load(path, cfg); q = logits_of(m)
        f = masked_kl_rows(q, p_log, obs["action_mask"]).detach().cpu().numpy()
        del m
        per = [float(f[cl == i].mean()) for i in range(8)]
        return per, float(np.mean(per))

    res = {"teachers": {}, "floors": {}}
    for name, run in TEACHER_RUNS.items():
        cfg = f"{run}/model_config.json"
        row = {}
        for variant, rel in (("best_model (WHAT THE FOLD LOADS)", "best_model/best_model.zip"),
                             ("final_model_interrupted (content_locality)",
                              "final_model_interrupted.zip")):
            p = os.path.join(run, rel)
            if not os.path.isfile(p):
                row[variant] = None; continue
            per, mu = score(p, cfg)
            row[variant] = {"path": os.path.relpath(p, MAIN), "per_team": per, "D_off": mu}
            print(f"  {name:12s} {variant:44s} D_off {mu:.4f}", flush=True)
        res["teachers"][name] = row
    for name, p in FLOORS.items():
        per, mu = score(p, PAR_CFG)
        res["floors"][name] = {"per_team": per, "D_off": mu}
        print(f"  {name:12s} D_off {mu:.4f}", flush=True)

    best = [res["teachers"][n]["best_model (WHAT THE FOLD LOADS)"]["D_off"] for n in TEACHER_RUNS]
    fin = [res["teachers"][n]["final_model_interrupted (content_locality)"]["D_off"]
           for n in TEACHER_RUNS]
    fl = float(np.mean([v["D_off"] for v in res["floors"].values()]))
    res["summary"] = {"D_off_best_model": float(np.mean(best)),
                      "D_off_final_model_interrupted": float(np.mean(fin)),
                      "floor_mean": fl,
                      "floor_units_best": float(np.mean(best)) / fl,
                      "floor_units_final": float(np.mean(fin)) / fl,
                      "reproduces_content_locality_batch": ok,
                      "n_states": len(states), "per_team_counts": counts}
    print(f"\n  v8 SET D_off  best_model {np.mean(best):.4f} ({np.mean(best)/fl:.1f}x floor)   "
          f"final_model_interrupted {np.mean(fin):.4f} ({np.mean(fin)/fl:.1f}x floor)   "
          f"floor {fl:.4f}")
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v8_checkpoint_fix.json")
