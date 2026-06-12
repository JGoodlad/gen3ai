"""Obs round-trip fuzz — the reconstruction layer's load-bearing correctness proof.

Plays REAL battles through the local bridge with a trainee that records, at every
decision, exactly what the live eval player records (the full ``embed_battle``
observation + the chosen action, persisted through the real ``BattleRecorder`` →
``states.npz`` path, with the battle's reconstruction record joined alongside as
``*_reconstruction.json``). Then, for every battle, rebuilds the trainee's
one-sided observations OFFLINE — ``reconstruction.replay_battle`` regenerates the
per-side protocol from the omniscient record, ``obs_materializer`` replays it
through the real obs pipeline — and requires every materialized obs row to equal
the live-saved row BIT-FOR-BIT.

That equality proves, in one shot:
- the bridge ``__RECON__`` capture is complete (seed + teams + every command);
- the sim replay regenerates the exact one-sided protocol the agent saw;
- the materializer's decision cadence mirrors the live player exactly
  (stall check → embed → zero-mask deferral → tracker.advance with the live
  action), including every stateful tracker the obs depends on.

Runs TWO phases: sequential (eval's default ``--eval-concurrency-per-worker 1``)
and **concurrent** (``concurrency=3`` — overlapping battles whose feeds
interleave on POKE_LOOP, the latency-hiding eval mode). The concurrent phase
pins the structural claim that per-battle state (tracker, recorder, packed
team under the runner's start_lock) is independent, so interleaving doesn't
perturb any battle's obs.

Run directly (no server needed):
    python src/agents/training/obs_roundtrip_fuzz_test.py [n_sequential] [n_concurrent]
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.gen3_battle import Gen3Battle
from agents.inference.player import Gen3Player
from agents.training.battle_recorder import BattleRecorder, write_battle_record
from agents.training.obs_materializer import materialize_from_record
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord, register_trace_prefix
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"


class RecordingFuzzPlayer(Gen3Player):
    """Live-side stand-in for EvalRLPlayer: identical decision cadence and
    identical forensic persistence, with a seeded-random policy in place of the
    model (the obs pipeline neither knows nor cares which policy picks)."""

    def __init__(self, *, out_dir: str, rng_seed: int, **kwargs):
        super().__init__(battle_class=Gen3Battle, **kwargs)
        self._out_dir = out_dir
        self._rng = np.random.RandomState(rng_seed)
        self._recorders: dict[str, BattleRecorder] = {}
        self._trace_idx = 0
        self.trace_prefixes: list[str] = []

    def choose_move(self, battle):
        forfeit = self._handle_stall(battle, "FUZZ_STALL")
        if forfeit:
            return forfeit
        obs_dict = self.embed_battle(battle)
        mask = obs_dict["action_mask"]
        if int(mask.sum()) == 0:
            return self.choose_default_move()
        idx = int(self._rng.choice(np.flatnonzero(mask)))

        tag = battle.strict_view().battle_tag
        rec = self._recorders.get(tag)
        if rec is None:
            rec = BattleRecorder(tag)
            self._recorders[tag] = rec
        # The same state dict RLPlayer._predict_best_action captures — obs is the
        # real thing; logits/value are placeholders (no model in this fuzz).
        rec.record(battle, idx, probs=np.full(11, 1 / 11), mask=mask,
                   state={"obs": np.asarray(obs_dict["observation"], dtype=np.float32),
                          "logits": np.zeros(11, dtype=np.float32), "value": 0.0})
        self._get_tracker(battle).advance(idx)
        return self.action_to_order(idx, battle)

    def _battle_finished_callback(self, battle) -> None:
        super()._battle_finished_callback(battle)
        tag = battle.strict_view().battle_tag
        rec = self._recorders.pop(tag, None)
        if rec is None:
            return
        self._trace_idx += 1
        prefix = os.path.join(self._out_dir, f"battle_{self._trace_idx:03d}")
        write_battle_record(prefix, rec, battle, step=0)
        # Same join the eval path uses: the __RECON__ frame arrives after this
        # callback (the streams close after |win|), so registering the prefix
        # here lets the registry complete the pair and write the artifact.
        register_trace_prefix(tag, prefix, extra={"trainee_username": self.username})
        self.trace_prefixes.append(prefix)


def _verify_battle(prefix: str) -> int:
    """Materialize the battle's obs offline and compare to the live rows."""
    recon_path = f"{prefix}_reconstruction.json"
    assert os.path.exists(recon_path), f"missing reconstruction artifact: {recon_path}"
    record = ReconstructionRecord.load(recon_path)
    assert record.trainee_username, "artifact lacks trainee_username"

    npz = np.load(f"{prefix}_states.npz")
    live_obs, actions = npz["obs"], npz["actions"]
    assert len(actions) == len(live_obs)

    trace = materialize_from_record(record, actions=actions)
    assert trace.actions_complete, "materializer ran out of actions mid-battle"
    assert len(trace.decisions) == len(live_obs), (
        f"{prefix}: decision count mismatch — live {len(live_obs)} vs "
        f"materialized {len(trace.decisions)}"
    )
    for i, dec in enumerate(trace.decisions):
        if not np.array_equal(dec.obs, live_obs[i]):
            bad = np.flatnonzero(dec.obs != live_obs[i])
            raise AssertionError(
                f"{prefix}: obs row {i} (turn {dec.turn}) differs at {len(bad)} dims; "
                f"first at dim {bad[0]}: live={live_obs[i][bad[0]]} "
                f"materialized={dec.obs[bad[0]]}"
            )
    return len(live_obs)


async def run_phase(n_battles: int, concurrency: int, tag: str) -> int:
    ts = int(time.time()) % 100000
    pool = TeamLoader().get_all_teams()
    with tempfile.TemporaryDirectory(prefix="obs_roundtrip_") as out_dir:
        trainee = RecordingFuzzPlayer(
            out_dir=out_dir, rng_seed=ts,
            battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
            account_configuration=AccountConfiguration(f"ORz{tag}{ts}", "pw"),
            server_configuration=LocalhostServerConfiguration,
            start_listening=False, max_concurrent_battles=max(1, concurrency),
        )
        opp = RandomPlayer(
            battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
            account_configuration=AccountConfiguration(f"ORo{tag}{ts}", "pw"),
            server_configuration=LocalhostServerConfiguration,
            start_listening=False, max_concurrent_battles=max(1, concurrency),
        )

        print(f"Obs round-trip fuzz — {n_battles} battles, concurrency={concurrency}",
              flush=True)
        await run_local_battles(trainee, opp, n_battles, concurrency=concurrency)
        assert len(trainee.trace_prefixes) == n_battles, (
            f"expected {n_battles} traces, got {len(trainee.trace_prefixes)}"
        )

        total = 0
        for prefix in trainee.trace_prefixes:
            n = _verify_battle(prefix)
            total += n
            print(f"  {os.path.basename(prefix)}: {n} decisions bit-identical", flush=True)
    return total


async def main(n_sequential: int, n_concurrent: int) -> None:
    total = await run_phase(n_sequential, concurrency=1, tag="s")
    total += await run_phase(n_concurrent, concurrency=3, tag="c")
    print(f"\nPASS — {n_sequential} sequential + {n_concurrent} concurrent battles, "
          f"{total} decisions: every materialized obs row equals the live row "
          f"bit-for-bit", flush=True)


if __name__ == "__main__":
    n_seq = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_conc = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    asyncio.run(main(n_seq, n_conc))
