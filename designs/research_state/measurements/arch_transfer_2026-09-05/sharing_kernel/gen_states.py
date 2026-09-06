"""SHARING-KERNEL state generator — one era-agnostic recipe, run once per era.

WHAT IT PRODUCES: a fixed batch of decision states the PARENT actually acted on, piloting each of
24 probe teams (16 TAUGHT by the teacher-content 2x2 fleet, 8 UNTAUGHT), against a FIXED opponent
model on 3 FIXED opponent teams. Every state carries its team sha10 so the kernel can be clustered
and permuted at the TEAM level.

WHY ONE RECIPE FOR BOTH ERAS. The v8 era predates the five `GEN3AI_*_SEED` variables (they landed
2026-08-30, well after `b13b30b2`), so the only determinism available there is the one probe P and
`v8_gift_timing_probe.py` used: `stochastic=False` on BOTH sides (no policy draw), a PINNED single
team per side (no team draw), and an EXPLICIT 4-int sim seed per battle (no dice draw). Rather than
run the two eras on two different protocols and then compare them, this script uses the ERA-
COMPATIBLE protocol in BOTH. That is a DELIBERATE DEVIATION from
`reuse_batch_2026-09-03/offline_collateral_kl.py`, which pilots stochastically with `policy_seed`:
here the measured quantity is the gradient of log pi at the ARGMAX action, so greedy piloting is
also the coherent behaviour policy for it. The five seed vars are exported by the caller anyway and
are inert at the era commit (recorded, not assumed).

TEAMS ARE RESOLVED BY CONTENT SHA, NEVER BY FILENAME. The 16 taught teams were promoted into
`data/teams/sample/` with sha10 filenames AFTER `b13b30b2`; at the era they live under
`data/teams/others/giraffe/` with opaque basenames. Both trees' pools hold all 24 team STRINGS
(verified: 770 files each, 24/24 shas present, 0 substitutions).

Run:
  gen:  PYTHONPATH=<worktree>/src python gen_states.py --era gen --out states_gen.npz
  v8 :  PYTHONPATH=/tmp/v8rep_era/src python gen_states.py --era v8  --out states_v8.npz
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import itertools
import json
import os
import random
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

th.set_num_threads(1)

from poke_env.ps_client import AccountConfiguration  # noqa: E402
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration  # noqa: E402

from agents.inference.player import RLPlayer  # noqa: E402
from agents.model.snapshot import current_model_version, load_foreign_opponent  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402
from utils.bridge.local_battle_runner import run_local_battles  # noqa: E402
from utils.teambuilder import Gen3Teambuilder  # noqa: E402

MD = "/home/goodlad/dev/gen3ai/models"

# --- THE 16 TAUGHT TEAMS: copied VERBATIM from
# designs/research_state/measurements/teacher_content_2x2_2026-09-04/taught_probe.py::SLICES,
# which resolved them at run time from each TC arm's own recorded --distill-teacher. Never
# hand-typed from an argv.
TAUGHT_SHA = ["009e3d0244", "37d717a93a", "436335607f", "64a691c473", "6916e13879",
              "6ebe9ebc4d", "750d056194", "8bdb5796b9", "9b454d9ea7", "9d8391d864",
              "9e95fb59d7", "a9831a5f5e", "aea50f207e", "b904dbe059", "e28069562f",
              "e702a104eb"]
# --- THE 8 UNTAUGHT TEAMS: the "sha" field of every row in
# designs/research_state/measurements/rev3_untaught_pulldown_selection.json, i.e. the same 8 teams
# offline_collateral_kl.py uses as its off-slice set.
UNTAUGHT_SHA = ["a7406f6c97", "55ff6899a2", "3495ef83ef", "1c4e182530",
                "564b9be3ae", "21022d30fb", "6a49f096f0", "324235812b"]
# --- 3 FIXED OPPONENT TEAMS: the first three of v8_gift_timing_probe.py::V8_SELECTION["opponents"]
# (probe P's pre-registered opponent set). Present in both trees' pools.
OPP_SHA = ["c1fc379c85", "8a29df031e", "6c129cb50c"]

N_PER_TEAM = 19          # EQUAL across all 24 teams -- required for the team-level permutation
BATTLES_PER_CELL = 1     # 24 teams x 3 opponents x 1 = 72 battles

ERAS = {
    "gen": dict(
        parent=f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip",
        parent_cfg=f"{MD}/ai_v9_29_rev1_0823/snapshots/model_config.json",
        opp=f"{MD}/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip",
        opp_cfg=f"{MD}/ai_v9_29_rev1_0823/snapshots/model_config.json",
        impl="rust",
    ),
    "v8": dict(
        parent=f"{MD}/ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip",
        parent_cfg=f"{MD}/ai_v8_04_distill_4teacher_0722/model_config.json",
        opp=f"{MD}/ai_v8_03_zarch_control_0718/final_model_interrupted.zip",
        opp_cfg=f"{MD}/ai_v8_03_zarch_control_0718/model_config.json",
        # The era's rust bridge predates the seedless-seed fix (bc00d4d) -- node is MANDATORY here.
        impl="node",
    ),
}

_ACCT = itertools.count(1)


def sha10(s: str) -> str:
    return hashlib.sha1(s.strip().encode()).hexdigest()[:10]


def repo_root() -> str:
    """The tree this code was IMPORTED from -- the era tree under the era PYTHONPATH.

    `utils.paths` (the project's one place that knows the tree depth) does not exist at
    `b13b30b2`, so the era falls back to the same arithmetic against the imported `utils`
    package: <root>/src/utils/__init__.py -> parents[2]. Never `os.getcwd()` -- this script is
    run from the deliverable directory in the CURRENT tree while importing the ERA tree."""
    try:
        import utils.paths as P
        return str(P.repo_root())
    except ModuleNotFoundError:
        import pathlib
        import utils
        return str(pathlib.Path(utils.__file__).resolve().parents[2])


def build_pool() -> dict:
    root = repo_root()
    pool = {}
    for f in sorted(glob.glob(f"{root}/data/teams/**/*.txt", recursive=True)):
        try:
            s = open(f).read()
        except Exception:
            continue
        pool.setdefault(sha10(s), (s, f))
    return pool


class PinnedTeam(Gen3Teambuilder):
    """MUST subclass Gen3Teambuilder -- yield_team has to return a PACKED team."""

    def __init__(self, team_str: str):
        super().__init__([team_str])

    def yield_team(self):
        return self.packed_teams[0]


def _strip_debugger(m):
    """A --log-level periodic checkpoint carries an ObservationDebugger that print()s a full board
    on EVERY forward. Verified output-neutral upstream (taught_probe.py / cf_producer.py do the
    same); here it is also removed because dynamo/graph-free gradient work must not depend on it."""
    obj = getattr(m, "policy", m)
    for mod in (obj.modules() if hasattr(obj, "modules") else []):
        if getattr(mod, "_debugger", None) is not None:
            mod._debugger = None
    fe = getattr(getattr(m, "policy", m), "features_extractor", None)
    if fe is not None and hasattr(fe, "_debugger"):
        fe._debugger = None
    return m


class Capturing(RLPlayer):
    """Records every observation the pilot actually acted on."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.captured = []

    def embed_battle(self, battle):
        d = super().embed_battle(battle)
        # >= 2 LEGAL ACTIONS, not >= 1. With a single legal action the masked policy puts all mass
        # on it, so log pi(a*|s) == 0 IDENTICALLY and grad log pi(a*|s) is the ZERO VECTOR -- a
        # state with no direction, whose cosine against anything is undefined. Measured on the
        # first smoke: such states appear and made the whole kernel NaN via a 0/0. Filtering here
        # rather than downstream keeps the per-team state count EQUAL, which the team-level
        # permutation null requires.
        if d is not None and d.get("action_mask") is not None and int(d["action_mask"].sum()) >= 2:
            self.captured.append((d["observation"].copy(), d["action_mask"].copy()))
        return d


def evenly(seq, k):
    """k evenly-spaced picks from seq (no RNG -- the subsample must not add a draw)."""
    n = len(seq)
    if n <= k:
        return list(seq)
    idx = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
    return [seq[i] for i in idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", required=True, choices=sorted(ERAS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--teams", default="all", help="all | taught | untaught (for sharding)")
    a = ap.parse_args(argv)
    E = ERAS[a.era]
    t0 = time.time()

    maps = load_mappings()
    cv = current_model_version(maps)
    parent, _ = load_foreign_opponent(E["parent"], current_version=cv, device="cpu",
                                      config_path=E["parent_cfg"])
    _strip_debugger(parent)
    parent.policy.set_training_mode(False)
    opp_model, _ = load_foreign_opponent(E["opp"], current_version=cv, device="cpu",
                                         config_path=E["opp_cfg"])
    _strip_debugger(opp_model)
    opp_model.policy.set_training_mode(False)

    # ACID: pilot and opponent must be DISTINCT networks; a mis-resolved path that loads one zip
    # twice would read as a spuriously self-consistent state stream.
    sa = parent.policy.state_dict()
    sb = opp_model.policy.state_dict()
    l2 = sum(float((sa[k] - sb[k]).pow(2).sum()) for k in sorted(set(sa) & set(sb))
             if sa[k].shape == sb[k].shape and sa[k].is_floating_point()) ** 0.5
    print(f"[gs] ACID pilot-vs-opponent param L2 = {l2:.4f}", flush=True)
    if l2 < 1e-3:
        raise SystemExit("[gs] ACID FAILED -- pilot and opponent are the same network")

    pool = build_pool()
    want = TAUGHT_SHA + UNTAUGHT_SHA + OPP_SHA
    missing = [s for s in want if s not in pool]
    if missing:
        raise SystemExit(f"[gs] selection GIGO: {missing} not in the pool at {repo_root()}")
    print(f"[gs] pool {len(pool)} teams at {repo_root()}; all {len(want)} selected shas present",
          flush=True)

    probes = ([(s, "taught") for s in TAUGHT_SHA] if a.teams in ("all", "taught") else [])
    probes += ([(s, "untaught") for s in UNTAUGHT_SHA] if a.teams in ("all", "untaught") else [])

    obs_rows, mask_rows, team_rows, grp_rows = [], [], [], []
    prov = []
    for team_sha, grp in probes:
        team_str = pool[team_sha][0]
        got = []
        for opp_sha in OPP_SHA:
            rng = random.Random(f"{team_sha}:{opp_sha}")     # probe P's CRN construction, verbatim
            seeds = [[rng.randrange(0, 65536) for _ in range(4)]
                     for _ in range(BATTLES_PER_CELL)]
            p = Capturing(model=parent, team=PinnedTeam(team_str), battle_format="gen3ou",
                          server_configuration=LocalhostServerConfiguration, mappings=maps,
                          account_configuration=AccountConfiguration(
                              f"SK{next(_ACCT):05d}", "pw"),
                          stochastic=False, start_listening=False)
            o = RLPlayer(model=opp_model, team=PinnedTeam(pool[opp_sha][0]),
                         battle_format="gen3ou",
                         server_configuration=LocalhostServerConfiguration, mappings=maps,
                         account_configuration=AccountConfiguration(f"SK{next(_ACCT):05d}", "pw"),
                         stochastic=False, start_listening=False)
            for s in seeds:
                try:
                    asyncio.run(run_local_battles(p, o, 1, seed=list(s), concurrency=1,
                                                  impl=E["impl"]))
                except Exception as e:
                    print(f"    !! {team_sha}/{opp_sha} {type(e).__name__}: {str(e)[:140]}",
                          flush=True)
            got.extend(p.captured)
            prov.append({"team": team_sha, "opp": opp_sha, "seeds": seeds,
                         "captured": len(p.captured)})
        picks = evenly(got, N_PER_TEAM)
        if len(picks) < N_PER_TEAM:
            raise SystemExit(f"[gs] team {team_sha}: only {len(picks)} states (< {N_PER_TEAM}); "
                             "the per-team count must be EQUAL across teams or the team-level "
                             "permutation null is not exchangeable")
        for ob, mk in picks:
            obs_rows.append(ob)
            mask_rows.append(mk)
            team_rows.append(team_sha)
            grp_rows.append(grp)
        print(f"  {grp:8s} {team_sha}: captured {len(got)} -> kept {len(picks)}   "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    obs = np.asarray(obs_rows, dtype=np.float32)
    mask = np.asarray(mask_rows, dtype=np.float32)
    np.savez_compressed(a.out, observation=obs, action_mask=mask,
                        team=np.array(team_rows), group=np.array(grp_rows))
    meta = {"era": a.era, "impl": E["impl"], "repo_root": repo_root(),
            "parent": E["parent"], "opponent": E["opp"],
            "n_states": int(obs.shape[0]), "obs_dim": int(obs.shape[1]),
            "n_per_team": N_PER_TEAM, "battles_per_cell": BATTLES_PER_CELL,
            "taught_sha": TAUGHT_SHA, "untaught_sha": UNTAUGHT_SHA, "opp_sha": OPP_SHA,
            "determinism": "stochastic=False both sides; pinned single team per side; explicit "
                           "4-int sim seed per battle from Random(f'{team}:{opp}')",
            "acid_pilot_vs_opp_l2": l2, "provenance": prov,
            "wall_s": round(time.time() - t0, 1)}
    with open(a.out.replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[gs] wrote {obs.shape} to {a.out} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
