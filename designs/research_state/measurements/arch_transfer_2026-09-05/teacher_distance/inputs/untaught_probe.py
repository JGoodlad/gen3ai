"""PROBE-Q INSTRUMENT — rev-4's UNTAUGHT PULL-DOWN (scorecard REPRO-1 / REPRO-2).

WHY A NEW METER: rev-4's fleet taught ALL 24 teams the existing meters cover, so neither the
9-slice nor the coverage cut contains a single untaught team. REPRO-1 asks what the fold did to
teams the fleet NEVER SAW — the gift sign. That set has to come from outside the 24.

THE SET: the 8 teams screened by headroom_screen.py but NOT selected into rev-4's fleet (they fell
below the 0.15 headroom cut). They were chosen by a rule fixed BEFORE rev-4 launched and were never
trained on, so they are untaught by construction rather than by after-the-fact selection.

REPRO-2's FLOOR DISCRIMINATOR is satisfiable here: 6 of the 8 have parent WR > 0.55 (0.553, 0.573,
0.580, 0.580, 0.593, 0.613), so there is competence to rob and the binding read is available. The
2 sub-0.55 teams are reported but are NOT the binding stratum — a null on floor-level teams proves
nothing, which is exactly what REPRO-2 exists to prevent.

n IS FIXED PRE-DATA at 200 games/team/arm, per the scorecard. Do not raise it after seeing a result.

ARMS: the rev-4 fold vs its own base (the rev-2 fold, which is also what rev-4's fleet
best-responded to). PULL-DOWN = fold - base on untaught teams. REPRO-1 PASSES if the CI upper
bound is >= 0; the reproduction signature is a sign flip from rev-2's -7.1.

Run: python untaught_probe.py <model.zip> <tag> <out.json> [n_per_team=200] [concurrency]
"""
import asyncio, json, math, random, sys, time
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from agents.inference.player import RLPlayer
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


def _strip_debugger(m):
    """Remove the ObservationDebugger a --log-level periodic checkpoint carries; it print()s a full
    DEEP TRACE board on EVERY forward. VERIFIED OUTPUT-NEUTRAL on 2026-09-02 (actions and values
    bit-identical with and without), which matters because these numbers are compared against a
    baseline measured BEFORE the strip existed. cf_producer.py does the same for the same reason."""
    obj = getattr(m, "policy", m)
    for mod in (obj.modules() if hasattr(obj, "modules") else []):
        if getattr(mod, "_debugger", None) is not None:
            mod._debugger = None
    return m

REV1 = "models/ai_v9_29_rev1_0823"
TARGET = f"{REV1}/snapshots/snapshot_000024000000.zip"
CFG = f"{REV1}/snapshots/model_config.json"

# ORDER IS THE SEED — never reorder or insert; si 0-6 match probes/piloting.json exactly.
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
_SEED_BASE = 90   # distinct seed family: the untaught probe is its own instrument
_SEED_BASE = 9          # continue the standing meter's seed family at indices 9/10/11


class PinnedTeam(Gen3Teambuilder):
    """MUST subclass Gen3Teambuilder — yield_team has to return a PACKED team."""
    def __init__(self, path): super().__init__([open(path).read()])
    def yield_team(self): return self.packed_teams[0]


class PairedPool(Gen3Teambuilder):
    """Indices are into `packed_teams`, NOT the raw list — Gen3Teambuilder SKIPS invalid teams."""
    def __init__(self, teams):
        super().__init__(teams); self._seq, self._i = [], 0
    def set_sequence(self, seq):
        self._seq, self._i = seq, 0; return self
    def yield_team(self):
        t = self.packed_teams[self._seq[self._i % len(self._seq)]]; self._i += 1; return t


def wilson(k, n, z=1.96):
    if not n: return (0.0, 0.0)
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    hw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (c-hw, c+hw)


def run(model_path, tag, out_path, n=100, conc=4):
    loader = TeamLoader(); pool = loader.get_all_teams()
    opp_pool = PairedPool(pool); n_pool = len(opp_pool.packed_teams)
    maps = load_mappings(); cv = current_model_version(maps)
    tgt, _ = load_foreign_opponent(TARGET, current_version=cv, device="cpu", config_path=CFG)
    _strip_debugger(tgt)
    arm, _ = load_foreign_opponent(model_path, current_version=cv, device="cpu", config_path=CFG)
    _strip_debugger(arm)
    res = {"_meta": {"tag": tag, "model": model_path, "target": TARGET, "n_per_team": n,
                     "seed_convention": "1000 + slice_index", "pool": n_pool}}
    print(f"  [{tag}] {model_path}  |  {len(SLICES)} teams x {n} games  |  pool {n_pool}", flush=True)
    tw = tn = 0
    for si, (sname, spath) in enumerate(SLICES):
        rng = random.Random(1000 + _SEED_BASE + si)
        seq = [rng.randrange(n_pool) for _ in range(n)]
        pilot = RLPlayer(model=arm, team=PinnedTeam(spath),
                         battle_format="gen3ou", server_configuration=LocalhostServerConfiguration,
                         mappings=maps, account_configuration=AccountConfiguration(f"AP{tag[:2]}{si}a", "pw"),
                         stochastic=True, start_listening=False)
        opp = RLPlayer(model=tgt, team=opp_pool.set_sequence(seq),
                       battle_format="gen3ou", server_configuration=LocalhostServerConfiguration,
                       mappings=maps, account_configuration=AccountConfiguration(f"AP{tag[:2]}{si}b", "pw"),
                       stochastic=True, start_listening=False)
        pilot.reset_battles(); opp.reset_battles()
        t0 = time.time()
        asyncio.run(run_local_battles(pilot, opp, n, concurrency=conc, impl="rust"))
        fin = max(1, pilot.n_finished_battles)
        lo, hi = wilson(pilot.n_won_battles, fin)
        res[sname] = {"wins": pilot.n_won_battles, "games": pilot.n_finished_battles,
                      "wr": pilot.n_won_battles/fin, "ci95": [lo, hi],
                      "secs": round(time.time()-t0, 1)}
        tw += pilot.n_won_battles; tn += pilot.n_finished_battles
        print(f"  [{tag}] {sname:14} {pilot.n_won_battles:4}/{fin:4} = {pilot.n_won_battles/fin:.4f} "
              f"[{lo:.3f},{hi:.3f}]  [{res[sname]['secs']}s]", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
    lo, hi = wilson(tw, tn)
    res["POOLED"] = {"wins": tw, "games": tn, "wr": tw/max(1, tn), "ci95": [lo, hi]}
    print(f"\n  [{tag}] ARM PILOTING POOLED {tw}/{tn}={tw/max(1,tn):.4f} [{lo:.3f},{hi:.3f}]", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3],
        int(sys.argv[4]) if len(sys.argv) > 4 else 100,
        int(sys.argv[5]) if len(sys.argv) > 5 else 4)
