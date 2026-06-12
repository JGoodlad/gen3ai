"""Reconstruction fuzz — real battles, then offline replay + re-roll invariants.

Plays N random battles through the bridge (no server), captures each battle's
``__RECON__`` record from the registry, and checks the offline primitives:

replay (the reproducibility contract):
- ``replay_battle`` ends, and its omniscient outcome matches the live result
  (same final turn, same winner);
- replay is DETERMINISTIC: a second replay regenerates byte-identical per-side
  chunk sequences, modulo ``|t:|`` wall-clock timestamp lines (the sim stamps
  those with Date.now at emission; they are in poke-env's MESSAGES_TO_IGNORE,
  so they are invisible to battle state and obs).

reroll (the counterfactual primitive):
- the reconstruction point sits at the requested turn with both requests open;
- re-rolling is a deterministic function of the seed (same seed → identical
  outcome + identical suffix protocol);
- distinct seeds explore (over the battle's turns, at least one turn shows >1
  distinct outcome — dice exist);
- every re-rolled timeline advances (turn+1 or battle end), even when a faint
  forces a mid-turn switch the original timeline never had.

Run directly (no server needed):
    python src/utils/bridge/reconstruction_fuzz_test.py [n_battles]
"""

from __future__ import annotations

import asyncio
import sys
import time

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import pop_record, replay_battle, reroll_turn
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"
REROLL_SEEDS = [f"{7 * i + 11},{131 * i + 3},{977 * i + 17},{4099 * i + 5}" for i in range(6)]


def _strip_ts(chunks) -> list:
    """Mask |t:| wall-clock lines — emission-time stamps, invisible to state/obs."""
    return ["\n".join("|t:|*" if l.startswith("|t:|") else l for l in c.split("\n"))
            for c in chunks]


def _verify_replay(record, battle, p1_name: str) -> None:
    rep = replay_battle(record)
    assert rep.outcome["ended"], f"{record.battle_tag}: replay did not end"
    live_winner = p1_name if battle.won else (None if battle.won is None else "other")
    if battle.won is True:
        assert rep.outcome["winner"] == p1_name, (
            f"{record.battle_tag}: live winner {p1_name} vs replay {rep.outcome['winner']}"
        )
    elif battle.won is False:
        assert rep.outcome["winner"] not in (None, p1_name), (
            f"{record.battle_tag}: live loss but replay winner={rep.outcome['winner']}"
        )
    assert rep.outcome["turn"] == battle.turn, (
        f"{record.battle_tag}: live final turn {battle.turn} vs replay {rep.outcome['turn']}"
    )
    # Determinism: byte-identical regeneration (modulo |t:| emission timestamps).
    rep2 = replay_battle(record)
    assert _strip_ts(rep2.p1_chunks) == _strip_ts(rep.p1_chunks), (
        f"{record.battle_tag}: replay is not deterministic (p1 stream)"
    )
    assert _strip_ts(rep2.p2_chunks) == _strip_ts(rep.p2_chunks), (
        f"{record.battle_tag}: replay is not deterministic (p2 stream)"
    )


def _verify_reroll(record, final_turn: int) -> bool:
    """Re-roll one mid-battle turn; returns True if outcomes varied across seeds."""
    t = max(2, min(4, final_turn - 1))
    rr = reroll_turn(record, t, seeds=REROLL_SEEDS)
    assert rr.pre_state["turn"] == t
    assert rr.requests["p1"] and rr.requests["p2"], "decision-point requests missing"
    assert rr.prefix_p1_chunks and rr.prefix_p2_chunks
    assert len(rr.rerolls) == len(REROLL_SEEDS)
    for r in rr.rerolls:
        assert not r.outcome.get("stuck"), f"turn {t} re-roll stuck (seed {r.seed})"
        assert r.outcome["ended"] or r.outcome["turn"] == t + 1, (
            f"re-roll did not advance: turn {r.outcome['turn']} (seed {r.seed})"
        )
        assert r.turn_log, "empty omniscient turn log"
        assert r.p1_chunks and r.p2_chunks, "empty one-sided suffix protocol"
    # Determinism: same seed → identical outcome and identical suffix protocol
    # (modulo |t:| emission timestamps).
    again = reroll_turn(record, t, seeds=[REROLL_SEEDS[0]]).rerolls[0]
    first = rr.rerolls[0]
    assert again.outcome == first.outcome, "re-roll outcome not deterministic"
    assert _strip_ts(again.p1_chunks) == _strip_ts(first.p1_chunks), (
        "re-roll protocol not deterministic")
    outs = {str(sorted(r.outcome.items())) for r in rr.rerolls}
    return len(outs) > 1


async def main(n_battles: int) -> None:
    ts = int(time.time()) % 100000
    pool = TeamLoader().get_all_teams()
    p1_name, p2_name = f"RFz{ts}", f"RFo{ts}"
    p1 = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(p1_name, "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1,
    )
    p2 = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(p2_name, "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1,
    )

    print(f"Reconstruction fuzz — {n_battles} battles", flush=True)
    await run_local_battles(p1, p2, n_battles)

    n_varied = 0
    n_checked = 0
    for tag, battle in p1._battles.items():
        record = pop_record(tag)
        assert record is not None, f"no reconstruction record captured for {tag}"
        assert record.username("p1") == p1_name and record.username("p2") == p2_name
        _verify_replay(record, battle, p1_name)
        if battle.turn >= 3:
            if _verify_reroll(record, battle.turn):
                n_varied += 1
        n_checked += 1
        print(f"  {tag}: replay reproduced (turn {battle.turn}, "
              f"winner={'p1' if battle.won else 'p2/tie'}), re-roll OK", flush=True)

    assert n_checked == n_battles
    # Dice exist: across the run, at least one re-rolled turn must split outcomes.
    assert n_varied >= 1, "no re-rolled turn produced distinct outcomes across 6 seeds"
    print(f"\nPASS — {n_checked} battles replayed bit-deterministically; "
          f"{n_varied} re-rolled turns showed outcome variance", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    asyncio.run(main(n))
