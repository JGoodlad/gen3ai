"""BaitBot VALIDATION BY INSTRUMENT — real bridge battles, no mocks.

The units pin the predicate and the arithmetic. They cannot tell you the bot actually baits when
plugged into a live battle: the trigger reads poke-env state that the units fake. This plays real
games and checks the bot's own counters against its dial, and that p_bait=0 vs p_bait=1 are
behaviourally different — the minimum for calling p_bait a CONTROLLED VARIABLE.

Run directly (bridge-backed, no server), the project's fuzz convention:
    python src/agents/baitbot_instrument_test.py [n_battles]
"""
import asyncio
import sys

from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.baitbot import Gen3BaitBotPlayer
from agents.opponents import Gen3AggressivePlayer
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


def _mk(cls, tag, tb, **kw):
    return cls(battle_format="gen3ou", team=tb,
               server_configuration=LocalhostServerConfiguration,
               account_configuration=AccountConfiguration(tag, "pw"),
               start_listening=False, **kw)


def run(n_battles: int = 30) -> int:
    loader = TeamLoader()
    all_t, samp = loader.get_all_teams(), loader.get_sample_teams()
    rc = 0
    for p_bait in (0.0, 0.6, 1.0):
        tb_a = Gen3Teambuilder(all_t, bias_teams=samp, bias_prob=0.1)
        tb_b = Gen3Teambuilder(all_t, bias_teams=samp, bias_prob=0.1)
        # An ALWAYS-ATTACKING opponent maximises revealed attacks, so the trigger is exercised.
        bait = _mk(Gen3BaitBotPlayer, f"BB{int(p_bait*10)}", tb_a, p_bait=p_bait, seed=11)
        foe = _mk(Gen3AggressivePlayer, f"AG{int(p_bait*10)}", tb_b)
        asyncio.run(run_local_battles(bait, foe, n_battles, concurrency=4, impl="rust"))
        opp, took, rate = bait.n_bait_opportunities, bait.n_baits_taken, bait.realized_bait_rate
        print(f"  p_bait={p_bait:<4} opportunities={opp:<5} taken={took:<5} realized={rate:.3f}", flush=True)
        if p_bait == 0.0 and took != 0:
            print("  ✗ p_bait=0 took a bait — the dial does not gate the pivot"); rc = 1
        if p_bait == 1.0 and opp > 0 and took != opp:
            print("  ✗ p_bait=1 skipped a bait — the dial does not gate the pivot"); rc = 1
        if opp == 0:
            print("  ✗ ZERO bait opportunities in real play — the trigger never fires"); rc = 1
        elif 0 < p_bait < 1 and abs(rate - p_bait) > 0.15:
            print(f"  ✗ realized {rate:.3f} strays from the dial {p_bait}"); rc = 1
    print("BAITBOT INSTRUMENT VALIDATION:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(run(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
