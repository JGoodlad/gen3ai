"""reroll_many parity fuzz — the BATCHED re-roll ≡ per-call ``reroll_turn``, bit-for-bit at the obs.

``reroll_many`` resolves N candidate-action arms of one turn in a SINGLE Node process (so the one-ply
lookahead pays the ~677 ms spawn / pokemon-showdown ``require`` cost ONCE instead of once per
candidate). It is purely a PERFORMANCE batching of :func:`reroll_turn` — the per-arm behaviour must not
drift. This fuzz pins that.

For an anchored move decision in each of N real bridge battles (no server) it builds the SAME
``(candidate action × seed)`` arms the lookahead builds, then resolves them TWO ways — one batched
``reroll_many`` call vs a separate ``reroll_turn`` per ``(candidate, seed)`` — and asserts they are
equivalent:

1. the per-side SUFFIX chunks are byte-identical MODULO the ``|t:|`` wall-clock line, and — airtight —
   the ONLY lines that ever differ are ``|t:|`` stamps (emission-time ``Date.now`` stamps, in poke-env's
   ``MESSAGES_TO_IGNORE``, invisible to battle state and obs); the per-arm ``outcome`` /
   ``choices_used`` match exactly; and
2. the MATERIALIZED successor obs (prefix+suffix → the REAL encoder) is BIT-FOR-BIT identical — the
   proof that the ``|t:|`` difference is truly obs-invisible and the batched path feeds the model an
   identical ``V(s′)`` input.

Run directly (no server needed — the bridge is in-process):
    python src/utils/bridge/reroll_many_parity_fuzz_test.py [n_battles]
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.training.obs_materializer import (
    materialize_decisions, materialize_from_record)
from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from main.prober.engine import _has_state
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord, reroll_many, reroll_turn
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"
N_SEEDS = 2                # fresh seeds per arm; "original" (the CRN realized-dice line) is always added
MAX_ALTS = 2              # alternative candidates beyond the chosen action


def _record_one_battle(out_dir: str):
    ts = int(time.time() * 1000) % 100000
    pool = TeamLoader().get_all_teams()
    trainee = RecordingFuzzPlayer(
        out_dir=out_dir, rng_seed=ts, battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"RMt{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"RMo{ts}", "pw"),
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


def _first_anchor_with_successor(summary, npz):
    invs = summary["invocations"]
    for i, inv in enumerate(invs):
        if (inv.get("phase") == "move_selection" and i + 1 < len(invs)
                and _has_state(npz, i) and _has_state(npz, i + 1)):
            return i
    return None


def _assert_chunk_parity(batched, single, *, tag, a, s):
    """Airtight: the batched and single per-side suffixes differ ONLY in |t:| timestamp lines."""
    assert len(batched) == len(single), (
        f"{tag} a={a} s={s}: chunk COUNT differs (batched {len(batched)} vs single {len(single)})")
    for cb, cs in zip(batched, single):
        if cb == cs:
            continue
        lb, ls = cb.split("\n"), cs.split("\n")
        assert len(lb) == len(ls), f"{tag} a={a} s={s}: line COUNT differs within a chunk"
        for x, y in zip(lb, ls):
            if x != y:
                assert x.startswith("|t:|") and y.startswith("|t:|"), (
                    f"{tag} a={a} s={s}: a NON-timestamp line differs:\n"
                    f"  batched={x!r}\n  single ={y!r}")


def _succ_obs(prefix_chunks, suffix_chunks, side, record, prefix_actions, a, anchor):
    """Materialize the successor obs (one decision past the anchor) the same way lookahead does."""
    mt = materialize_decisions(
        list(prefix_chunks) + list(suffix_chunks), username=record.username(side),
        packed_team=record.packed_team(side), side=side, actions=prefix_actions + [a],
        battle_format=record.format_id, battle_tag=record.battle_tag,
        stop_after_decision=anchor + 1)
    if len(mt.decisions) <= anchor + 1:
        return None
    return mt.decisions[anchor + 1].obs


def _check_battle(record, summary, npz) -> int:
    """Verify reroll_many ≡ per-call reroll_turn for one battle. Returns the # of obs-gated arms."""
    anchor = _first_anchor_with_successor(summary, npz)
    if anchor is None:
        return 0
    invs = summary["invocations"]
    turn = int(invs[anchor]["turn"])
    side = record.side_of(record.trainee_username)
    other = "p2" if side == "p1" else "p1"
    actions = np.asarray(npz["actions"], dtype=int)
    chosen_idx = int(actions[anchor])

    # Map legal actions → sim choice strings via the SAME real mapper the lookahead uses.
    trace = materialize_from_record(
        record, actions=actions, map_actions_at=anchor, stop_after_decision=anchor)
    choice_map = trace.action_choices or {}
    assert chosen_idx in choice_map, f"{record.battle_tag}: chosen action not legal in replay"
    cand = [chosen_idx] + [c for c in choice_map if c != chosen_idx][:MAX_ALTS]

    seeds = [f"{13 * i + 5},{29 * i + 7},{101 * i + 3},{200 * i + 9}" for i in range(N_SEEDS)]
    seed_list = seeds + ["original"]
    prefix_actions = [int(x) for x in actions[:anchor]]

    # Batched: resolve every (candidate × seed) arm in ONE process (exactly the lookahead's arm spec).
    arms = []
    for a in cand:
        our = "recorded" if a == chosen_idx else choice_map[a]
        for s in seed_list:
            arms.append({f"{side}_action": our, f"{other}_action": "recorded",
                         "seed": s, "label": int(a)})
    many = reroll_many(record, turn, arms)
    many_prefix = many.prefix_p1_chunks if side == "p1" else many.prefix_p2_chunks
    by_arm: dict = {}
    for arm in many.arms:
        by_arm.setdefault(int(arm.label), {})[arm.seed] = arm

    n_obs_gated = 0
    for a in cand:
        our = "recorded" if a == chosen_idx else choice_map[a]
        for s in seed_list:
            single_rr = reroll_turn(
                record, turn, seeds=[s],
                **{f"{side}_action": our, f"{other}_action": "recorded"})
            single = single_rr.rerolls[0]
            single_prefix = single_rr.prefix_p1_chunks if side == "p1" else single_rr.prefix_p2_chunks
            arm = by_arm[a][s]

            # (1) per-side suffix parity modulo |t:|, + exact outcome / chosen-actions.
            _assert_chunk_parity(arm.p1_chunks, single.p1_chunks, tag=record.battle_tag, a=a, s=s)
            _assert_chunk_parity(arm.p2_chunks, single.p2_chunks, tag=record.battle_tag, a=a, s=s)
            assert arm.outcome == single.outcome, (
                f"{record.battle_tag} a={a} s={s}: outcome differs\n"
                f"  batched={arm.outcome}\n  single ={single.outcome}")
            assert arm.choices_used == single.choices_used, (
                f"{record.battle_tag} a={a} s={s}: chosen actions differ")

            # (2) materialized successor obs BIT-FOR-BIT (both paths) — the obs-invisibility proof.
            if arm.outcome.get("ended") or arm.outcome.get("stuck"):
                continue
            ob_b = _succ_obs(many_prefix, arm.p1_chunks if side == "p1" else arm.p2_chunks,
                             side, record, prefix_actions, a, anchor)
            ob_s = _succ_obs(single_prefix, single.p1_chunks if side == "p1" else single.p2_chunks,
                             side, record, prefix_actions, a, anchor)
            if ob_b is None or ob_s is None:
                continue
            assert np.array_equal(ob_b, ob_s), (
                f"{record.battle_tag} a={a} s={s}: materialized successor obs differ between "
                f"reroll_many and reroll_turn (NOT bit-for-bit)")
            n_obs_gated += 1
    return n_obs_gated


def main(n_battles: int) -> None:
    print(f"reroll_many parity fuzz — {n_battles} battles", flush=True)
    total_obs_gated = 0
    n_checked = 0
    for b in range(n_battles):
        with tempfile.TemporaryDirectory(prefix="reroll_many_parity_") as out_dir:
            record, summary, npz = _record_one_battle(out_dir)
            gated = _check_battle(record, summary, npz)
            total_obs_gated += gated
            n_checked += 1
            print(f"  {record.battle_tag}: reroll_many ≡ reroll_turn "
                  f"({gated} arms obs-gated bit-for-bit)", flush=True)
    assert n_checked == n_battles
    assert total_obs_gated >= 1, (
        "no arm reached the obs bit-for-bit gate — every anchor's turn ended? re-run / raise n_battles")
    print(f"\nPASS — {n_checked} battles: batched reroll_many is byte-identical to per-call reroll_turn "
          f"(modulo |t:|) and produces bit-for-bit identical obs ({total_obs_gated} arms gated)",
          flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(n)
