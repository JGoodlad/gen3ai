"""M1 AXIS SPLIT — the UNTAUGHT arm of the fold-shape table.

WHAT THIS IS: a VERBATIM re-use of the training session's untaught instrument
(`~/.claude/jobs/1046b1d6/tmp/probes/untaught_probe.py`), extended to the two fold arms it
never covered — COMPFOLD (3x4) and R3ACTION (6x2) — plus the rev-1 baseline that turns the
already-measured R2ACTION row into rev-2's OWN untaught hop.

WHY VERBATIM MATTERS: `untaught_R2ACTION.json` and `untaught_R4ACTION.json` already exist on
this exact 8-team set with this exact seed family (`1000 + 9 + slice_index`), this exact fixed
target (rev-1 @24M snapshot), n = 200, stochastic=False, rust bridge. Re-implementing the meter
would have put the two new arms on a second scale and made the cross-arm differences
uninterpretable. The SLICES list, the seed base, the target, the player construction and the
Wilson interval below are byte-for-byte the originals; only the CLI and this docstring differ.

THE SET: the 8 teams screened by the training session's `headroom_screen.py` and NOT selected
into rev-4's fleet. Verified here (see the report script) to be untaught by EVERY fold in the
table: rev-2's F5 union (9), rev-3's F6 union (12), COMPFOLD's explicit 12, rev-4's R4S3 union
(24). Untaught by construction, chosen before any of these folds existed.

n IS FIXED PRE-DATA at 200 games/team/arm — inherited from the instrument, not re-chosen.

Run (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src):
  python axis_split_untaught_arm.py <model.zip> <tag> <out.json> [n=200] [concurrency=3]
"""
import asyncio
import json
import math
import os
import random
import sys
import time

from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from agents.inference.player import RLPlayer
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

REV1 = "models/ai_v9_29_rev1_0823"
TARGET = f"{REV1}/final_model.zip"
CFG = f"{REV1}/model_config.json"

# ORDER IS THE SEED — never reorder or insert.
SLICES = [
    ("U_61590463", "data/teams/sample/61590463ee85d456.txt"),
    ("U_92832108", "data/teams/sample/9283210847f806ee.txt"),
    ("U_ce35b736", "data/teams/sample/ce35b7368c3d692e.txt"),
    ("U_9909f2e9", "data/teams/sample/9909f2e98e981ccc.txt"),
    ("U_9d5f8458", "data/teams/sample/9d5f845869e899ee.txt"),
    ("U_f7ba5702", "data/teams/sample/f7ba5702fe856292.txt"),
    ("U_90b94599", "data/teams/sample/90b94599967c6b77.txt"),
    ("U_dbf81d8e", "data/teams/sample/dbf81d8ecae51c39.txt"),
]
_SEED_BASE = 9  # the standing meter's seed family, continued at indices 9/10/11


class PinnedTeam(Gen3Teambuilder):
    """MUST subclass Gen3Teambuilder — yield_team has to return a PACKED team."""

    def __init__(self, path):
        super().__init__([open(path).read()])

    def yield_team(self):
        return self.packed_teams[0]


class PairedPool(Gen3Teambuilder):
    """Indices are into `packed_teams`, NOT the raw list — Gen3Teambuilder SKIPS invalid teams."""

    def __init__(self, teams):
        super().__init__(teams)
        self._seq, self._i = [], 0

    def set_sequence(self, seq):
        self._seq, self._i = seq, 0
        return self

    def yield_team(self):
        t = self.packed_teams[self._seq[self._i % len(self._seq)]]
        self._i += 1
        return t


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - hw, c + hw)


def run(model_path, tag, out_path, n=200, conc=3):
    loader = TeamLoader()
    pool = loader.get_all_teams()
    opp_pool = PairedPool(pool)
    n_pool = len(opp_pool.packed_teams)
    maps = load_mappings()
    cv = current_model_version(maps)
    tgt, _ = load_foreign_opponent(TARGET, current_version=cv, device="cpu", config_path=CFG)
    arm, _ = load_foreign_opponent(model_path, current_version=cv, device="cpu", config_path=CFG)
    res = {
        "_meta": {
            "tag": tag,
            "model": model_path,
            "target": TARGET,
            "n_per_team": n,
            "seed_convention": "1000 + slice_index",
            "pool": n_pool,
        }
    }
    # RESUME (gen3_untaught_arm_resume_v1, 2026-08-31). Every team is INDEPENDENT — its opponent
    # draw comes from `random.Random(1000 + _SEED_BASE + si)`, an index-derived seed, and the sim
    # dice are minted per battle — so re-entering the loop with earlier teams already present
    # reproduces exactly the run that would have happened uninterrupted. This is NOT an optimization:
    # at `conc >= 2` the bridge runner takes its BOUNDED-CONCURRENCY path, whose per-battle bound is
    # a deliberate TOTAL-DURATION cap, and CLAUDE.md is explicit that scaling does not rescue a cap.
    # Beside a saturated box that cap fires, and without this an 8-team arm lost every completed team.
    if os.path.exists(out_path):
        prior = json.load(open(out_path))
        done = {k: v for k, v in prior.items() if k not in ("_meta", "POOLED")}
        if done:
            res.update(done)
            print(f"  [{tag}] RESUMING — {len(done)} team(s) already collected: "
                  f"{', '.join(sorted(done))}", flush=True)
    print(f"  [{tag}] {model_path}  |  {len(SLICES)} teams x {n} games  |  pool {n_pool}", flush=True)
    tw = sum(v["wins"] for k, v in res.items() if k != "_meta")
    tn = sum(v["games"] for k, v in res.items() if k != "_meta")
    for si, (sname, spath) in enumerate(SLICES):
        if sname in res:
            continue
        rng = random.Random(1000 + _SEED_BASE + si)
        seq = [rng.randrange(n_pool) for _ in range(n)]
        pilot = RLPlayer(
            model=arm,
            team=PinnedTeam(spath),
            battle_format="gen3ou",
            server_configuration=LocalhostServerConfiguration,
            mappings=maps,
            account_configuration=AccountConfiguration(f"AX{tag[:2]}{si}a", "pw"),
            stochastic=False,
            start_listening=False,
        )
        opp = RLPlayer(
            model=tgt,
            team=opp_pool.set_sequence(seq),
            battle_format="gen3ou",
            server_configuration=LocalhostServerConfiguration,
            mappings=maps,
            account_configuration=AccountConfiguration(f"AX{tag[:2]}{si}b", "pw"),
            stochastic=False,
            start_listening=False,
        )
        pilot.reset_battles()
        opp.reset_battles()
        t0 = time.time()
        asyncio.run(run_local_battles(pilot, opp, n, concurrency=conc, impl="rust"))
        fin = max(1, pilot.n_finished_battles)
        lo, hi = wilson(pilot.n_won_battles, fin)
        res[sname] = {
            "wins": pilot.n_won_battles,
            "games": pilot.n_finished_battles,
            "wr": pilot.n_won_battles / fin,
            "ci95": [lo, hi],
            "secs": round(time.time() - t0, 1),
        }
        tw += pilot.n_won_battles
        tn += pilot.n_finished_battles
        print(
            f"  [{tag}] {sname:14} {pilot.n_won_battles:4}/{fin:4} = {pilot.n_won_battles/fin:.4f} "
            f"[{lo:.3f},{hi:.3f}]  [{res[sname]['secs']}s]",
            flush=True,
        )
        json.dump(res, open(out_path, "w"), indent=1)
    lo, hi = wilson(tw, tn)
    res["POOLED"] = {"wins": tw, "games": tn, "wr": tw / max(1, tn), "ci95": [lo, hi]}
    print(f"\n  [{tag}] UNTAUGHT POOLED {tw}/{tn}={tw/max(1,tn):.4f} [{lo:.3f},{hi:.3f}]", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    run(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]) if len(sys.argv) > 4 else 200,
        int(sys.argv[5]) if len(sys.argv) > 5 else 3,
    )
