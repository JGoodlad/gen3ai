"""TEACHER DISTANCE — D_off and D_on for every gen-era fold's teacher set, against ITS OWN parent.

D_off = mean forward KL(teacher || parent) over legal actions on the UNTAUGHT-8 states.
D_on  = the same statistic on that teacher's OWN taught teams.

`masked_kl_rows` is IMPORTED from agents.training.instrumented_ppo.distill_anchor -- the same
formula the live --distill-anchor-monitor logs, never reimplemented.

STATE RECIPE, copied verbatim from content_locality/gen_era_locality.py (itself copied from
reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py): the FOLD PARENT pilots each
team against rev-1's 24M snapshot, stochastic, per-player policy seeds 71000/72000+team_index, sim
seed [team_index+1,2,3,4], pool sequence random.Random(61000+team_index), concurrency=1, rust.

The UNTAUGHT 8 hold indices 0..7 at 9 battles each -- IDENTICAL to content_locality's n=9 run, so
the gen-era D_off for the FUND/UNF sets must REPRODUCE that artifact's untaught column exactly.
That reproduction is asserted, not hoped for. Taught teams run at 3 battles each (D_on is the
secondary prediction; the cost of 40 more teams at 9 is not worth it).

Two arms:
  gen  -- parent ai_v9_59_R2ACTION_0827/final_model.zip, teacher sets R4set / R3set / FUND / UNF
  rev2 -- parent ai_v9_29_rev1_0823/final_model.zip,     teacher set  R2set

Run: python teacher_distance.py <arm: gen|rev2> <out.json>
     (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
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

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = "/home/goodlad/dev/gen3ai"
MD = f"{MAIN}/models"
REV1 = f"{MD}/ai_v9_29_rev1_0823"
CFG = f"{REV1}/snapshots/model_config.json"
OPPONENT = f"{REV1}/snapshots/snapshot_000024000000.zip"

UNTAUGHT = ["61590463ee85d456", "9283210847f806ee", "ce35b7368c3d692e", "9909f2e98e981ccc",
            "9d5f845869e899ee", "f7ba5702fe856292", "90b94599967c6b77", "dbf81d8ecae51c39"]
N_UNTAUGHT_BATTLES = 9      # matches content_locality n=9 exactly on indices 0..7
N_TAUGHT_BATTLES = 3

ARMS = {
    "gen":  {"parent": f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip",
             "sets": ["R4set", "R3set", "FUND", "UNF"]},
    "rev2": {"parent": f"{MD}/ai_v9_29_rev1_0823/final_model.zip",
             "sets": ["R2set"]},
}


def T(b):
    return f"{MAIN}/data/teams/sample/{b}.txt"


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


def main(arm, out_path):
    spec = ARMS[arm]
    sets = json.load(open(f"{HERE}/teacher_sets.json"))
    used = {k: sets[k] for k in spec["sets"]}
    for v in used.values():
        assert v["parent"] == spec["parent"], f"arm {arm}: set parent {v['parent']} != {spec['parent']}"

    taught_union = sorted({b for v in used.values() for b in v["taught_union"]})
    assert not (set(taught_union) & set(UNTAUGHT))
    TEAMS = ([(b, "untaught", N_UNTAUGHT_BATTLES) for b in UNTAUGHT]
             + [(b, "taught", N_TAUGHT_BATTLES) for b in taught_union])
    IDX = {b: i for i, (b, _, _) in enumerate(TEAMS)}
    n_un = len(UNTAUGHT)

    maps = load_mappings(); cv = current_model_version(maps)

    def load(p):
        m, _ = load_foreign_opponent(p, current_version=cv, device="cpu", config_path=CFG)
        return _strip_debugger(m)

    parent = load(spec["parent"])
    opp_model = load(OPPONENT)

    loader = TeamLoader(); pool = PairedPool(loader.get_all_teams())
    n_pool = len(pool.packed_teams)

    t0 = time.time(); states = []; team_of = []
    for ti, (b, kind, nb) in enumerate(TEAMS):
        rng = random.Random(61000 + ti)
        seq = [rng.randrange(n_pool) for _ in range(nb)]
        pilot = Capturing(model=parent, team=PinnedTeam(T(b)), battle_format="gen3ou",
                          server_configuration=LocalhostServerConfiguration, mappings=maps,
                          account_configuration=AccountConfiguration(f"TD{arm}{ti}a", "pw"),
                          stochastic=True, start_listening=False, policy_seed=71000 + ti)
        opp = RLPlayer(model=opp_model, team=pool.set_sequence(seq), battle_format="gen3ou",
                       server_configuration=LocalhostServerConfiguration, mappings=maps,
                       account_configuration=AccountConfiguration(f"TD{arm}{ti}b", "pw"),
                       stochastic=True, start_listening=False, policy_seed=72000 + ti)
        pilot.reset_battles(); opp.reset_battles()
        asyncio.run(run_local_battles(pilot, opp, nb, concurrency=1, impl="rust",
                                      seed=[ti + 1, 2, 3, 4]))
        states.extend(pilot.captured); team_of.extend([ti] * len(pilot.captured))
        print(f"  [{ti:2d}] {kind:8s} {b} n{nb} +{len(pilot.captured):4d} states "
              f"(total {len(states)}, {time.time()-t0:.0f}s)", flush=True)

    n_un_states = sum(1 for t in team_of if t < n_un)
    print(f"  UNTAUGHT STATE COUNT: {n_un_states} on indices 0..{n_un-1}", flush=True)
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

    per_teacher = {}
    for tag, v in used.items():
        for t in v["teachers"]:
            if t in per_teacher:
                continue
            path = (f"{MD}/{t}/best_model/best_model.zip"
                    if os.path.exists(f"{MD}/{t}/best_model/best_model.zip")
                    else f"{MD}/{t}/final_model.zip")
            m = load(path)
            q = logits_of(m)
            fwd = masked_kl_rows(q, p_log, obs["action_mask"]).detach().cpu().numpy()
            rev = masked_kl_rows(p_log, q, obs["action_mask"]).detach().cpu().numpy()
            del m
            f = np.array([fwd[cl == i].mean() for i in range(len(TEAMS))])
            r = np.array([rev[cl == i].mean() for i in range(len(TEAMS))])
            own = [IDX[b] for v2 in used.values()
                   for tt, bs in v2["per_teacher_taught"].items() if tt == t for b in bs]
            own = sorted(set(own))
            per_teacher[t] = {"path": os.path.relpath(path, MAIN),
                              "per_team_fwd": [float(x) for x in f],
                              "per_team_rev": [float(x) for x in r],
                              "own_taught_idx": own,
                              "d_off": float(f[:n_un].mean()),
                              "d_on": float(f[own].mean())}
            print(f"  {t:26s} D_off {f[:n_un].mean():.4f}  D_on(own {len(own):2d}) "
                  f"{f[own].mean():.4f}", flush=True)

    # ACID: no two teachers may share a per-team KL vector (a mis-resolved path masquerading)
    vecs = {t: tuple(round(x, 9) for x in d["per_team_fwd"]) for t, d in per_teacher.items()}
    assert len(set(vecs.values())) == len(vecs), "ACID FAIL: two teachers produced identical KL"

    sets_out = {}
    for tag, v in used.items():
        offs = [per_teacher[t]["d_off"] for t in v["teachers"]]
        ons = [per_teacher[t]["d_on"] for t in v["teachers"]]
        sets_out[tag] = {"teachers": v["teachers"], "n_teachers": len(v["teachers"]),
                         "parent": v["parent"], "n_taught_teams": len(v["taught_union"]),
                         "D_off": float(np.mean(offs)), "D_on": float(np.mean(ons)),
                         "per_teacher_D_off": offs, "per_teacher_D_on": ons}
        print(f"SET {tag:7s} n={len(offs)}  D_off {np.mean(offs):.4f}  D_on {np.mean(ons):.4f}",
              flush=True)

    out = {"_meta": {
        "arm": arm, "parent": spec["parent"], "opponent": OPPONENT, "config": CFG,
        "statistic": "forward KL(teacher||parent) over legal actions; masked_kl_rows IMPORTED "
                     "from agents.training.instrumented_ppo.distill_anchor",
        "state_source": f"PARENT pilots each of {len(TEAMS)} teams vs rev-1's 24M snapshot, "
                        f"{N_UNTAUGHT_BATTLES} battles/untaught team and {N_TAUGHT_BATTLES}/taught, "
                        "stochastic, concurrency=1, rust",
        "seeds": {"sim": "[team_index+1,2,3,4]", "pilot_policy": "71000+team_index",
                  "opponent_policy": "72000+team_index", "pool_sequence": "61000+team_index"},
        "teams": [{"i": i, "basename": b, "kind": k, "battles": nb}
                  for i, (b, k, nb) in enumerate(TEAMS)],
        "n_states": len(states), "n_untaught_states": int(n_un_states),
        "states_per_team": [int((cl == i).sum()) for i in range(len(TEAMS))],
        "wall_s": round(time.time() - t0, 1)},
        "per_teacher": per_teacher, "sets": sets_out}
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
