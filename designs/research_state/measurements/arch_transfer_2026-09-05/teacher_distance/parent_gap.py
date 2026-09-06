"""THE PARENT GAP — how much of a gen-era teacher's D_off is inherited rather than earned?

Every gen-era teacher (R2F5, R3F6, R4S3, R5F) was forked from `ai_v9_29_rev1_0823/final_model.zip`.
For the rev-2 fold that IS the fold parent, so its D_off is pure teacher displacement. For every
other fold the parent is R2ACTION, which is itself a fold OFF rev-1 — so those teachers start at
whatever distance rev-1 already sits from R2ACTION, before they train at all.

This measures that baseline: KL(REV1FIN || R2ACTION) on the SAME untaught-8 states, plus the two
matched-noise floor checkpoints of R2ACTION's own run (a reproduction check against
content_locality's floor row).

State recipe identical to teacher_distance.py's untaught half -- same seeds, same 9 battles/team,
same indices 0..7 -- so the states reproduce (per-team counts are asserted against the gen arm).

Run: python parent_gap.py <out.json>
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
PARENT = f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip"

PROBES = {
    "REV1FIN (the gen-era teachers' own fork parent)": f"{REV1}/final_model.zip",
    "FLOOR_ckpt_28067760": f"{MD}/ai_v9_59_R2ACTION_0827/checkpoints/checkpoint_28067760_steps.zip",
    "FLOOR_ckpt_27917760": f"{MD}/ai_v9_59_R2ACTION_0827/checkpoints/checkpoint_27917760_steps.zip",
}
UNTAUGHT = ["61590463ee85d456", "9283210847f806ee", "ce35b7368c3d692e", "9909f2e98e981ccc",
            "9d5f845869e899ee", "f7ba5702fe856292", "90b94599967c6b77", "dbf81d8ecae51c39"]
EXPECT = [280, 399, 333, 458, 714, 592, 391, 301]      # the gen arm's per-team counts


def _strip(m):
    for mod in getattr(m, "policy", m).modules():
        if getattr(mod, "_debugger", None) is not None:
            mod._debugger = None
    return m


class PinnedTeam(Gen3Teambuilder):
    def __init__(self, p): super().__init__([open(p).read()])
    def yield_team(self): return self.packed_teams[0]


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
            self.captured.append({"observation": d["observation"].copy(),
                                  "action_mask": d["action_mask"].copy()})
        return d


def main(out_path):
    maps = load_mappings(); cv = current_model_version(maps)

    def load(p):
        m, _ = load_foreign_opponent(p, current_version=cv, device="cpu", config_path=CFG)
        return _strip(m)

    parent = load(PARENT); opp_model = load(OPPONENT)
    pool = PairedPool(TeamLoader().get_all_teams()); n_pool = len(pool.packed_teams)

    t0 = time.time(); states = []; team_of = []
    for ti, b in enumerate(UNTAUGHT):
        rng = random.Random(61000 + ti)
        seq = [rng.randrange(n_pool) for _ in range(9)]
        pilot = Capturing(model=parent, team=PinnedTeam(f"{MAIN}/data/teams/sample/{b}.txt"),
                          battle_format="gen3ou", mappings=maps,
                          server_configuration=LocalhostServerConfiguration,
                          account_configuration=AccountConfiguration(f"PG{ti}a", "pw"),
                          stochastic=True, start_listening=False, policy_seed=71000 + ti)
        opp = RLPlayer(model=opp_model, team=pool.set_sequence(seq), battle_format="gen3ou",
                       mappings=maps, server_configuration=LocalhostServerConfiguration,
                       account_configuration=AccountConfiguration(f"PG{ti}b", "pw"),
                       stochastic=True, start_listening=False, policy_seed=72000 + ti)
        pilot.reset_battles(); opp.reset_battles()
        asyncio.run(run_local_battles(pilot, opp, 9, concurrency=1, impl="rust",
                                      seed=[ti + 1, 2, 3, 4]))
        states.extend(pilot.captured); team_of.extend([ti] * len(pilot.captured))
        print(f"  [{ti}] {b} +{len(pilot.captured)} (total {len(states)}, {time.time()-t0:.0f}s)",
              flush=True)

    counts = [int(sum(1 for t in team_of if t == i)) for i in range(8)]
    print(f"  per-team counts {counts}  expected {EXPECT}  "
          f"{'REPRODUCED' if counts == EXPECT else 'MISMATCH'}", flush=True)
    assert counts == EXPECT, "state batch did NOT reproduce the gen arm -- refusing to report"

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
    res = {}
    for tag, path in PROBES.items():
        m = load(path); q = logits_of(m)
        f = masked_kl_rows(q, p_log, obs["action_mask"]).detach().cpu().numpy()
        del m
        per = [float(f[cl == i].mean()) for i in range(8)]
        res[tag] = {"path": os.path.relpath(path, MAIN), "per_team": per,
                    "D_off": float(np.mean(per))}
        print(f"  {tag:52s} D_off {np.mean(per):.4f}", flush=True)

    json.dump({"_meta": {"parent": PARENT, "opponent": OPPONENT,
                         "states": len(states), "per_team_counts": counts,
                         "reproduces_gen_arm_untaught_batch": counts == EXPECT,
                         "wall_s": round(time.time() - t0, 1)},
               "probes": res}, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/parent_gap.json")
