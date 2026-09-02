"""v8 GIFT TIMING — does the untaught gain ACCRUE over the fold's PPO time?

QUESTION. Probe P (`v8_redistribution_pfsp_2026-08-30.md`) measured that v8's fold
(`ai_v8_14_distill3_0725`, forked from `ai_v8_04_distill_4teacher_0722` at 277,583,267 steps and
run to 292,100,648) GIFTED **+5.42pp [+3.44, +7.42]** to 16 pool teams it never taught. It
measured the ENDPOINT only. This probe measures the SAME quantity at six points ALONG the fold, so
the gift's TIME COURSE becomes an observable.

  hypothesis (owner, 2026-09-01): the gift is off-slice PPO RE-OPTIMISING a distillation
  perturbation, which needs PPO TIME  =>  the untaught gain ACCRUES.
  competing: the gift is the leak's direct content  =>  the gain is present EARLY and flat.

INSTRUMENT — REUSED, NOT REBUILT. Every cell-defining constant below is copied VERBATIM from
`v8_fold_behavioral_fingerprint_probe.py` (which itself copied them from probe P's
`/tmp/probeP/selection.json`): the 16 untaught probe team sha10s, the 8 fixed opponent team
sha10s, the fixed reference opponent (`ai_v8_03_zarch_control_0718` — an ancestor of both arms,
equal to neither), the per-cell CRN seed construction `random.Random(f"{team}:{opp}")` drawing
4-int sim seeds, greedy play on both sides, the node bridge, CPU. Game index i is therefore the
SAME battle for every arm, and this probe's battles are a CRN PREFIX SUBSAMPLE of probe P's own.

ERA PIN. The v8 arms load only under the v8-era code (`b13b30b289c5eaba136a930a4ab63451e209fbe5`).
Run from an era-pinned checkout; the era's rust bridge predates the seedless-seed fix (`bc00d4d`),
so the NODE bridge is mandatory here.

THE FIVE REPRODUCIBILITY SEEDS DO NOT EXIST AT THIS COMMIT. `$GEN3AI_{PLAYER,TEAM,POLICY,POOL,
STALLER}_SEED` landed 2026-08-30, well after `b13b30b`; grep over the era tree finds zero
references. Determinism here comes from the same three things it came from for probe P and M4:
`stochastic=False` on both sides (no policy draw), a PINNED single team per side (no team draw),
and an EXPLICIT 4-int sim seed per battle (no dice draw). The env vars are exported anyway so the
command is copy-pasteable forward, and their inertness at this commit is recorded, not assumed.

Run (from the era-pinned tree):
  PYTHONPATH=<era>/src GEN3AI_TIMEOUT_SCALE=12 nice -n 15 python v8_gift_timing_probe.py \
      --arm <name> --games 16 --shard 0/4 --out /tmp/v8t/rows
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import random
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch as th  # noqa: E402

th.set_num_threads(1)

from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration  # noqa: E402

from agents.inference.player import RLPlayer  # noqa: E402
from agents.model.snapshot import current_model_version, load_foreign_opponent  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402
from utils.bridge.local_battle_runner import run_local_battles  # noqa: E402
from utils.team_loader import TeamLoader  # noqa: E402
from utils.teambuilder import Gen3Teambuilder  # noqa: E402

MAIN = "/home/goodlad/dev/gen3ai"
MD = f"{MAIN}/models"
FOLD_RUN = f"{MD}/ai_v8_14_distill3_0725"
CFG = f"{FOLD_RUN}/model_config.json"

# The fork point, from v8_14's metadata: the parent's `final_model_interrupted.zip` step count.
FORK_STEP = 277_583_267

# THE ARMS. `parent` and `foldfinal` are probe P's OWN two arms, byte-for-byte (the exact fork
# source and the exact fold endpoint), so both ends of this curve are direct replications of the
# +5.42pp headline. The five interior arms are the retained checkpoints nearest the
# pre-registered +1/+3/+6/+9/+12M grid; v8_14 retains 14 checkpoints (not 28 — models/ retention
# thinned the run), all listed in `v8_gift_timing_inputs/`.
#
# WHY `foldfinal` AND NOT `checkpoint_292100648`: the run's last checkpoint is 292,100,648
# (+14.52M) but `final_model_interrupted.zip` is 292,623,779 (+15.04M) — 0.52M further on, and
# ‖ckpt − final‖₂ = 17.15 against the fold's 238.9 total travel, so they are NOT the same weights.
# Probe P's fold arm is the final; using the checkpoint instead would put this probe's endpoint on
# a different model from the headline it is timing.
ARMS = {
    "parent":    (f"{MD}/ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip",
                  f"{MD}/ai_v8_04_distill_4teacher_0722/model_config.json", FORK_STEP),
    "c278672":   (f"{FOLD_RUN}/checkpoints/checkpoint_278671945_steps.zip", CFG, 278_671_945),
    "c280656":   (f"{FOLD_RUN}/checkpoints/checkpoint_280656375_steps.zip", CFG, 280_656_375),
    "c283636":   (f"{FOLD_RUN}/checkpoints/checkpoint_283635665_steps.zip", CFG, 283_635_665),
    "c287136":   (f"{FOLD_RUN}/checkpoints/checkpoint_287136098_steps.zip", CFG, 287_136_098),
    "c290116":   (f"{FOLD_RUN}/checkpoints/checkpoint_290115536_steps.zip", CFG, 290_115_536),
    "foldfinal": (f"{FOLD_RUN}/final_model_interrupted.zip", CFG, 292_623_779),
    # UNREGISTERED FOLLOW-UP arms, added AFTER the six-cell grid returned a HUMPED curve
    # (+9.67pp at +12.53M falling to +4.98pp at the endpoint). They exist only to LOCALISE that
    # decline between +12.5M and the end — they are not part of P1/P2 and are labelled
    # exploratory wherever they are reported.
    "c291106":   (f"{FOLD_RUN}/checkpoints/checkpoint_291106373_steps.zip", CFG, 291_106_373),
    "c292101":   (f"{FOLD_RUN}/checkpoints/checkpoint_292100648_steps.zip", CFG, 292_100_648),
}
REF = (f"{MD}/ai_v8_03_zarch_control_0718/final_model_interrupted.zip",
       f"{MD}/ai_v8_03_zarch_control_0718/model_config.json")

# VERBATIM from `v8_fold_behavioral_fingerprint_probe.py::V8_SELECTION` (probe P's pre-registered
# selection). Only the 16 untaught probe teams and the 8 fixed opponents are used here.
V8_SELECTION = {
    "probe_untaught": ["d0a4d2bcb8", "c90e782cad", "a6b630e6b4", "a577a735b7",
                       "9292a21833", "eaa88395e7", "7c2cb5cec1", "89fcef3b53",
                       "32f549483f", "593d7fb8a8", "7163ad9387", "dd460484fc",
                       "b26ed9c8e1", "c84f2b64a2", "f593373169", "048182d1e9"],
    "control_taught": ["564b9be3ae", "7594a34f82", "044da80d78",
                       "5c88ff9ca5", "4771662cf7", "45995e432f"],
    "opponents": ["c1fc379c85", "8a29df031e", "6c129cb50c", "511c359e2e",
                  "0af19b638a", "c0e60903e8", "2c5b1d6abb", "f7e46432b9"],
    "labels": {
        "d0a4d2bcb8": "balance", "c90e782cad": "balance", "a6b630e6b4": "balance",
        "a577a735b7": "balance", "9292a21833": "hyper_offense", "eaa88395e7": "hyper_offense",
        "7c2cb5cec1": "hyper_offense", "89fcef3b53": "offense", "32f549483f": "offense",
        "593d7fb8a8": "offense", "7163ad9387": "semi_stall", "dd460484fc": "semi_stall",
        "b26ed9c8e1": "semi_stall", "c84f2b64a2": "stall", "f593373169": "stall",
        "048182d1e9": "stall", "564b9be3ae": "semi_stall", "7594a34f82": "semi_stall",
        "044da80d78": "stall", "5c88ff9ca5": "stall", "4771662cf7": "stall",
        "45995e432f": "stall",
    },
}

_ACCT = itertools.count(1)


def _acct(tag: str) -> AccountConfiguration:
    return AccountConfiguration(f"VT{tag[:2]}{next(_ACCT):05d}", "pw")


def sha10(s: str) -> str:
    return hashlib.sha1(s.strip().encode()).hexdigest()[:10]


def load(zip_path: str, cfg: str, cv):
    m, _ = load_foreign_opponent(zip_path, current_version=cv, device="cpu", config_path=cfg)
    fe = m.policy.features_extractor
    if hasattr(fe, "_debugger"):
        fe._debugger = None
    m.policy.set_training_mode(False)
    return m


def acid(models: dict) -> dict:
    """The arms LOAD and are DISTINCT networks. A mis-resolved path that loads one zip twice
    reads as a perfect null, so distinctness is a gate, not a nicety."""
    sds = {n: m.policy.state_dict() for n, m in models.items()}
    names = list(models)
    keys = sorted(set.intersection(*[set(s) for s in sds.values()]))
    pmat = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            tot = 0.0
            for k in keys:
                ta, tb = sds[a][k], sds[b][k]
                if ta.shape == tb.shape and ta.is_floating_point():
                    tot += float((ta - tb).pow(2).sum())
            pmat[f"{a}|{b}"] = round(tot ** 0.5, 4)
    return {"pairwise_param_l2": pmat, "all_distinct": all(v > 1e-3 for v in pmat.values()),
            "n_params": {n: int(sum(p.numel() for p in m.policy.parameters()))
                         for n, m in models.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--games", type=int, default=16, help="battles per (team, opp) cell")
    ap.add_argument("--opps", type=int, default=8)
    ap.add_argument("--shard", default="0/1", help="i/k over the 16 untaught probe teams")
    ap.add_argument("--impl", default="node", choices=("node", "rust"))
    ap.add_argument("--out", default="/tmp/v8t/rows")
    a = ap.parse_args(argv)
    if a.impl != "node":
        raise SystemExit("[vt] the v8 era predates the rust seedless-seed fix — node is mandatory")

    t0 = time.time()
    mappings = load_mappings()
    cv = current_model_version(mappings)
    models = {"arm": load(*ARMS[a.arm][:2], cv), "ref": load(*REF, cv)}
    ac = acid(models)
    print(f"[vt] ACID {json.dumps(ac)}", flush=True)
    if not ac["all_distinct"]:
        raise SystemExit("[vt] ACID FAILED — arm and reference are not distinct networks")

    pool = {sha10(t): t for t in TeamLoader().get_all_teams()}
    sel = V8_SELECTION
    missing = [s for s in sel["probe_untaught"] + sel["opponents"] if s not in pool]
    if missing:
        raise SystemExit(f"[vt] selection GIGO: {missing} not in the pool")

    probes = list(sel["probe_untaught"])
    si, sk = (int(x) for x in a.shard.split("/"))
    probes = [p for i, p in enumerate(probes) if i % sk == si]
    opps = sel["opponents"][:a.opps]
    print(f"[vt] arm={a.arm} step={ARMS[a.arm][2]} {len(probes)} teams x {len(opps)} opps x "
          f"{a.games}g = {len(probes) * len(opps) * a.games} battles", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cell_path = f"{a.out}_{a.arm}_s{si}of{sk}_cells.jsonl"
    done = set()
    if os.path.exists(cell_path):
        # RESUME. Every cell is INDEPENDENT: its seeds come from `Random(f"{team}:{opp}")`, an
        # identity-derived generator, and each battle is played on an explicit seed under greedy
        # play — so re-entering with earlier cells present reproduces the uninterrupted run.
        for ln in open(cell_path):
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("requested") == d.get("finished"):
                done.add((d["team"], d["opp"]))
    cellsink = open(cell_path, "a")

    for team_sha in probes:
        team_str = pool[team_sha]
        for opp_sha in opps:
            if (team_sha, opp_sha) in done:
                print(f"  skip (done) {team_sha} vs {opp_sha}", flush=True)
                continue
            rng = random.Random(f"{team_sha}:{opp_sha}")   # probe P's CRN construction, verbatim
            seeds = [[rng.randrange(0, 65536) for _ in range(4)] for _ in range(a.games)]
            ts = time.time()
            p = RLPlayer(model=models["arm"], team=Gen3Teambuilder([team_str]),
                         battle_format="gen3ou",
                         server_configuration=LocalhostServerConfiguration,
                         mappings=mappings, account_configuration=_acct(a.arm),
                         stochastic=False, start_listening=False)
            o = RLPlayer(model=models["ref"], team=Gen3Teambuilder([pool[opp_sha]]),
                         battle_format="gen3ou",
                         server_configuration=LocalhostServerConfiguration,
                         mappings=mappings, account_configuration=_acct("rf"),
                         stochastic=False, start_listening=False)
            per_game = []
            for s in seeds:
                w0, f0 = p.n_won_battles, p.n_finished_battles
                try:
                    asyncio.run(run_local_battles(p, o, 1, seed=list(s), concurrency=1,
                                                  impl=a.impl))
                except Exception as e:
                    print(f"    !! {a.arm}/{team_sha}/{opp_sha} {type(e).__name__}: "
                          f"{str(e)[:140]}", flush=True)
                per_game.append(1 if p.n_won_battles > w0
                                else (0 if p.n_finished_battles > f0 else -1))
            rec = {"arm": a.arm, "step": ARMS[a.arm][2], "team": team_sha, "opp": opp_sha,
                   "arch": sel["labels"].get(team_sha), "kind": "untaught",
                   "wins": p.n_won_battles, "finished": p.n_finished_battles,
                   "requested": a.games, "per_game": per_game,
                   "secs": round(time.time() - ts, 1)}
            cellsink.write(json.dumps(rec) + "\n")
            cellsink.flush()
            print(f"  {team_sha} vs {opp_sha} [{a.arm}] {p.n_won_battles}/{p.n_finished_battles}"
                  f" {rec['secs']:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    cellsink.close()
    with open(f"{a.out}_{a.arm}_s{si}of{sk}_meta.json", "w") as f:
        json.dump({"acid": ac, "arm": a.arm, "step": ARMS[a.arm][2], "games": a.games,
                   "opps": opps, "shard": a.shard, "impl": a.impl,
                   "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print(f"[vt] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
