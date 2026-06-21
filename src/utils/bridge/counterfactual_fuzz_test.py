"""Counterfactual replay-to-end faithfulness — REAL bridge battles, no server. Run directly:

    export PYTHONPATH=$PYTHONPATH:src && python3 src/utils/bridge/counterfactual_fuzz_test.py [n]

Two decisive oracles + a divergence check, per recorded battle:

1. FULL-REPLAY OUTCOME ORACLE — scripting BOTH sides to the whole recorded command log (no divergence)
   and re-seeding with the recorded resolved seed reproduces the recorded WINNER. If the scripting sent
   a wrong choice the game would desync and the winner would flip.
2. FULL-REPLAY OBS ORACLE — the scripted trainee's per-decision obs (built LIVE through the real
   ``embed_battle`` / tracker during the scripted replay) equals the recorded ``states.npz`` obs
   BIT-FOR-BIT. This is the ``obs_roundtrip`` guarantee carried through the live ``run_local_battles``
   path, and the proof that the prefix ``_invert`` → ``tracker.advance`` keeps the post-divergence
   turn-history faithful.
3. DIVERGENCE-TO-TERMINAL — substituting a move at an anchor turn and continuing live (random policy
   here) reaches a real win/loss; the prefix obs (before the divergence) still match the recording.

No ``test_*`` functions (so ``pytest`` collects nothing — this is a script), mirroring the other
``utils/bridge`` fuzz tests.
"""

import asyncio
import json
import sys
import tempfile
import time

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.gen3_battle import Gen3Battle
from agents.inference.player import Gen3Player
from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from utils.bridge.counterfactual import replay_counterfactual
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


class _ReplayTrainee(Gen3Player):
    """A model-free Gen3Player: live decisions pick a random legal action (used only AFTER the
    scripted handoff). Same obs pipeline / tracker as the real trainee, so the obs oracle is exact."""

    def __init__(self, *, rng_seed: int = 0, **kwargs):
        super().__init__(battle_class=Gen3Battle, **kwargs)
        self._rng = np.random.RandomState(rng_seed)

    def choose_move(self, battle):
        forfeit = self._handle_stall(battle, "CF_STALL")
        if forfeit:
            return forfeit
        obs_dict = self.embed_battle(battle)
        mask = obs_dict["action_mask"]
        if int(mask.sum()) == 0:
            return self.choose_default_move()
        idx = int(self._rng.choice(np.flatnonzero(mask)))
        self._get_tracker(battle).advance(idx)
        return self.action_to_order(idx, battle)


def _record_one_battle(out_dir: str):
    ts = int(time.time() * 1000) % 1_000_000
    pool = TeamLoader().get_all_teams()
    trainee = RecordingFuzzPlayer(
        out_dir=out_dir, rng_seed=ts, battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"CFt{ts % 100000}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"CFo{ts % 100000}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(trainee, opp, 1))
    prefix = trainee.trace_prefixes[0]
    record = ReconstructionRecord.load(f"{prefix}_reconstruction.json")
    with open(f"{prefix}_summary.json") as f:
        summary = json.load(f)
    with np.load(f"{prefix}_states.npz") as z:
        npz = {k: z[k] for k in z.files}
    return record, summary, npz


def _fresh_players(record, tag, *, opp_seed: int = 7):
    """A scripted-then-live trainee + a DETERMINISTIC (seeded random-legal) opponent. Both policies are
    seeded so the only post-divergence randomness is the SIM PRNG — which makes the reseed determinism
    check exact (the recorded opponent was a RandomPlayer, but the counterfactual opponent is the
    caller's; here a deterministic Gen3Player stands in)."""
    our = record.side_of(record.trainee_username)
    trainee = _ReplayTrainee(
        rng_seed=0, battle_format=record.format_id, team=record.packed_team(our),
        account_configuration=AccountConfiguration(f"CFr{tag}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = _ReplayTrainee(
        rng_seed=opp_seed, battle_format=record.format_id, team="x",
        account_configuration=AccountConfiguration(f"CFp{tag}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    return trainee, opp


def _check_battle(i: int) -> None:
    with tempfile.TemporaryDirectory(prefix="cf_fuzz_") as out_dir:
        record, summary, npz = _record_one_battle(out_dir)
    recorded_result = ((summary.get("meta") or {}).get("result") or "").lower()
    n_decisions = int(npz["obs"].shape[0])

    # --- 1 + 2: full-replay outcome + obs oracle ---
    trainee, opp = _fresh_players(record, f"a{i}")
    out = replay_counterfactual(record, trainee=trainee, opponent=opp,
                                divergence_turn=None, capture_obs=True)
    assert out["ended"], f"[{i}] full replay did not finish: {out}"
    assert out["outcome"] == recorded_result, (
        f"[{i}] full-replay winner {out['outcome']!r} != recorded {recorded_result!r}")

    captured = out["trainee_obs"]
    assert len(captured) == n_decisions, (
        f"[{i}] captured {len(captured)} obs, recorded {n_decisions} decisions")
    for k, (got, want) in enumerate(zip(captured, npz["obs"])):
        if not np.array_equal(got, np.asarray(want, dtype=np.float32)):
            diff = int(np.argmax(np.abs(got - want) > 0))
            raise AssertionError(
                f"[{i}] scripted obs at decision {k} differs from recorded npz "
                f"(first diff at dim {diff}: {got[diff]} vs {want[diff]})")

    # --- 3: divergence-to-terminal (substitute a move mid-game, continue live) ---
    invs = summary["invocations"]
    # A MID-game move anchor (real prefix to validate + a real continuation), turn > 1.
    move_anchors = [j for j, inv in enumerate(invs)
                    if inv.get("phase") == "move_selection" and int(inv["turn"]) > 1 and j + 3 < len(invs)]
    anchor = move_anchors[len(move_anchors) // 2] if move_anchors else None
    if anchor is not None:
        from agents.training.obs_materializer import materialize_from_record
        actions = np.asarray(npz["actions"], dtype=int)
        trace = materialize_from_record(record, actions=actions, map_actions_at=anchor,
                                        stop_after_decision=anchor)
        choice_map = trace.action_choices or {}
        chosen = int(actions[anchor])
        # an ALTERNATIVE move if one exists, else the recorded one (still exercises the handoff)
        alts = [a for a in choice_map if a != chosen]
        sub_idx = alts[0] if alts else chosen
        turn = int(invs[anchor]["turn"])
        trainee, opp = _fresh_players(record, f"b{i}")
        dout = replay_counterfactual(
            record, trainee=trainee, opponent=opp, divergence_turn=turn,
            substitute_choice=choice_map[sub_idx], capture_obs=True)
        assert dout["ended"], f"[{i}] divergence replay did not finish: {dout}"
        assert dout["outcome"] in ("win", "loss", "tie"), dout
        # The prefix (decisions strictly before the divergence turn) must still match the recording.
        pre = [k for k, inv in enumerate(invs) if int(inv["turn"]) < turn and k < n_decisions]
        for k in pre[: len(dout["trainee_obs"])]:
            if k < len(dout["trainee_obs"]):
                assert np.array_equal(dout["trainee_obs"][k],
                                      np.asarray(npz["obs"][k], dtype=np.float32)), (
                    f"[{i}] divergence prefix obs at decision {k} differs from recorded")

        # --- 4: Monte-Carlo reseed — post-divergence dice resample; the prefix stays FIXED ---
        from main.prober.falsifier import fresh_seeds
        ps_a, ps_b = fresh_seeds(2, salt=f"cf{i}")
        reseed_outs = []
        for k, ps in enumerate((ps_a, ps_b)):
            tr, op = _fresh_players(record, f"c{i}{k}")
            r = replay_counterfactual(record, trainee=tr, opponent=op, divergence_turn=turn,
                                      substitute_choice=choice_map[sub_idx], post_t_seed=ps,
                                      capture_obs=True)
            assert r["ended"], f"[{i}] reseed line did not finish (seed {ps}): {r}"
            for k2 in pre[: len(r["trainee_obs"])]:               # reseed must NOT disturb the prefix
                assert np.array_equal(r["trainee_obs"][k2], np.asarray(npz["obs"][k2], dtype=np.float32)), (
                    f"[{i}] reseed disturbed prefix obs at decision {k2}")
            reseed_outs.append(r["outcome"])
        # Determinism: the SAME post_t_seed (+ deterministic policies both sides) reproduces the outcome.
        tr, op = _fresh_players(record, f"d{i}")
        r2 = replay_counterfactual(record, trainee=tr, opponent=op, divergence_turn=turn,
                                   substitute_choice=choice_map[sub_idx], post_t_seed=ps_a)
        assert r2["outcome"] == reseed_outs[0], (
            f"[{i}] reseed not deterministic: {r2['outcome']} != {reseed_outs[0]}")

    print(f"  battle {i}: outcome={out['outcome']} decisions={n_decisions} "
          f"divergence@turn={turn if anchor is not None else 'n/a'}  OK")


def main(n: int = 3) -> None:
    for i in range(n):
        _check_battle(i)
    print(f"counterfactual_fuzz_test: {n} battles PASSED")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
