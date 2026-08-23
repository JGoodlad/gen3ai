"""search-server clone parity fuzz — the warm serializeBattle clone-and-branch ≡ the trusted
reroll_many path, bit-for-bit at the obs, plus the value_crn faithfulness anchor.

The ``better_line`` search branches a tree by CLONING mid-battle states
(``State.serializeBattle`` / ``deserializeBattle`` in ``search_driver.js``) — a path NO prior code
exercised. This fuzz pins it against the already-trusted ``reroll_many`` re-roll (parity-fuzzed
bit-for-bit vs per-call ``reroll_turn``) over real bridge battles (no server):

For an anchored move decision in each battle it expands the SAME (candidate, opponent-recorded, CRN)
arms two ways — once through the ``SearchSession`` clone, once through ``reroll_many`` — and asserts:

1. **value_crn ANCHOR** — the clone's ``recorded_exact`` child (the chosen action, realized dice)
   materializes a successor obs BIT-FOR-BIT equal to the recorded ``states.npz`` next obs. So with the
   real model the chosen line's V(s′) == the trace's ``recorded_next_value`` (the lookahead anchor).
2. **clone ≡ reroll_many** — every candidate's clone successor obs equals the ``reroll_many`` successor
   obs BIT-FOR-BIT (the two engines must agree; |t:| is obs-invisible so this is exact).
3. **depth-2 composition** — cloning a depth-1 child and expanding again composes a valid
   (root-prefix + suffix₁ + suffix₂) chain that materializes a depth-2 successor obs.

Run directly (no server needed — the bridge is in-process):
    python src/utils/bridge/search_clone_parity_fuzz_test.py [n_battles]
    python src/utils/bridge/search_clone_parity_fuzz_test.py 4 --impl rust
    python src/utils/bridge/search_clone_parity_fuzz_test.py 4 --impl rust --record-impl rust

``--impl`` selects which OFFLINE driver serves both halves of the comparison (the warm
clone-and-branch ``SearchSession`` **and** the ``reroll_many`` oracle it is checked against);
``--record-impl`` selects the LIVE bridge that plays the battle the record comes from. They are
deliberately independent: a mixed run is the realistic case (training on rust, forensics on
node), and a record produced by one engine must replay on the other for that to hold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.training.obs_materializer import (
    infer_action_indices, materialize_decisions, materialize_from_record)
from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from main.prober.engine import _has_state
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord, reroll_many
from utils.bridge.search_session import SearchSession
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"
MAX_ALTS = 3


def _record_one_battle(out_dir: str, record_impl: str = "node"):
    ts = int(time.time() * 1000) % 100000
    pool = TeamLoader().get_all_teams()
    trainee = RecordingFuzzPlayer(
        out_dir=out_dir, rng_seed=ts, battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"SCt{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"SCo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(trainee, opp, 1, impl=record_impl))
    prefix = trainee.trace_prefixes[0]
    record = ReconstructionRecord.load(f"{prefix}_reconstruction.json")
    with open(f"{prefix}_summary.json") as f:
        summary = json.load(f)
    with np.load(f"{prefix}_states.npz") as z:
        npz = {k: z[k] for k in z.files}
    return record, summary, npz


def _mid_anchor(summary, npz):
    invs = summary["invocations"]
    cand = [i for i, inv in enumerate(invs)
            if inv.get("phase") == "move_selection" and i + 1 < len(invs)
            and _has_state(npz, i) and _has_state(npz, i + 1)]
    return cand[len(cand) // 2] if cand else None


def _check_battle(record, summary, npz, impl: str = "node") -> int:
    anchor = _mid_anchor(summary, npz)
    if anchor is None:
        return 0
    invs = summary["invocations"]
    turn = int(invs[anchor]["turn"])
    side = record.side_of(record.trainee_username)
    other = "p2" if side == "p1" else "p1"
    username = record.username(side)
    actions = np.asarray(npz["actions"], dtype=int)
    chosen = int(actions[anchor])
    prefix_actions = [int(x) for x in actions[:anchor]]
    trace = materialize_from_record(record, actions=actions, map_actions_at=anchor,
                                    stop_after_decision=anchor, impl=impl)
    cmap = trace.action_choices or {}
    assert chosen in cmap, f"{record.battle_tag}: chosen not legal in replay"

    def succ_obs(prefix, suffix, a, decision=None, encode_only_at=None):
        d = anchor + 1 if decision is None else decision
        mt = materialize_decisions(
            list(prefix) + list(suffix), username=username, packed_team=record.packed_team(side),
            side=side, actions=prefix_actions + a, battle_format=record.format_id,
            battle_tag=record.battle_tag, stop_after_decision=d, encode_only_at=encode_only_at)
        return mt.decisions[d].obs if len(mt.decisions) > d else None

    # infer_action_indices (the interior-opponent action-history primitive) must recover OUR side's
    # recorded actions bit-for-bit — the same inversion drives the opponent's obs in the search.
    inferred = np.asarray(infer_action_indices(record, side, impl=impl), dtype=int)
    m = min(len(inferred), len(actions))
    assert m > 0 and np.array_equal(inferred[:m], actions[:m]), (
        f"{record.battle_tag}: infer_action_indices != recorded trainee actions (inversion desync)")

    ob_recorded = np.asarray(npz["obs"][anchor + 1], dtype=np.float32)
    alts = [a for a in cmap if a != chosen][:MAX_ALTS]
    # Counts OBS comparisons only, and therefore starts at ZERO. It was initialised to 1 (for the
    # two pre-session asserts above), which made the caller's `total >= 1` floor — whose message
    # reads "no obs check ran" — impossible to fire: every non-terminal-anchor battle returned at
    # least 1 whether or not a single obs was ever compared. Every obs check below sits behind an
    # `ended`/`None` guard, so all four CAN be skipped at once.
    n_checked = 0

    with SearchSession(record, impl=impl) as ss:
        root = ss.open_root(turn)
        pfx = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
        opp_rec = root.recorded_choices.get(other) or "default"
        arms = [{"node_id": root.node_id, "recorded_exact": True, "seed": "original", "label": chosen}]
        for a in alts:
            arms.append({"node_id": root.node_id, f"{side}_action": cmap[a],
                         f"{other}_action": opp_rec, "seed": "original", "label": a})
        nodes = {n.label: n for n in ss.expand_many(arms)}

        # (1) value_crn ANCHOR — the chosen recorded_exact clone reproduces the recorded next obs.
        ch = nodes[chosen]
        if not ch.ended:
            ch_chunks = ch.p1_chunks if side == "p1" else ch.p2_chunks
            ob_clone = succ_obs(pfx, ch_chunks, [chosen])
            assert ob_clone is not None and np.array_equal(ob_clone, ob_recorded), (
                f"{record.battle_tag}: clone recorded_exact successor obs != recorded next obs "
                "BIT-FOR-BIT (the value_crn anchor)")
            n_checked += 1
            # (1b) LEVER-3 faithfulness — encoding ONLY the target decision (track-only prefix, the
            # better-line speedup) yields a BIT-FOR-BIT identical successor obs to a full encode.
            ob_lean = succ_obs(pfx, ch_chunks, [chosen], encode_only_at={anchor + 1})
            assert ob_lean is not None and np.array_equal(ob_lean, ob_clone), (
                f"{record.battle_tag}: encode_only_at successor obs != full-encode obs BIT-FOR-BIT "
                "(track-only prefix corrupted the target obs)")
            n_checked += 1

        # (2) clone ≡ reroll_many for each candidate (chosen + alts), bit-for-bit.
        rr_arms = [{f"{side}_action": "recorded", f"{other}_action": "recorded",
                    "seed": "original", "label": chosen}]
        for a in alts:
            rr_arms.append({f"{side}_action": cmap[a], f"{other}_action": opp_rec,
                            "seed": "original", "label": a})
        rr = reroll_many(record, turn, rr_arms, impl=impl)
        rr_pfx = rr.prefix_p1_chunks if side == "p1" else rr.prefix_p2_chunks
        rr_by = {arm.label: arm for arm in rr.arms}
        for a in [chosen] + alts:
            cl, rrm = nodes[a], rr_by[a]
            if cl.ended or rrm.outcome.get("ended") or rrm.outcome.get("stuck"):
                continue
            ob_cl = succ_obs(pfx, cl.p1_chunks if side == "p1" else cl.p2_chunks, [a])
            ob_rr = succ_obs(rr_pfx, rrm.p1_chunks if side == "p1" else rrm.p2_chunks, [a])
            if ob_cl is None or ob_rr is None:
                continue
            assert np.array_equal(ob_cl, ob_rr), (
                f"{record.battle_tag} action {a}: clone obs != reroll_many obs BIT-FOR-BIT")
            n_checked += 1

        # (3) depth-2 composition — clone a depth-1 child, expand again, materialize a depth-2 obs.
        ext = next((nodes[a] for a in alts if not nodes[a].ended and nodes[a].node_id), None)
        if ext is not None:
            d2 = ss.expand_many([{"node_id": ext.node_id, f"{side}_action": "default",
                                  f"{other}_action": "default", "seed": "original", "label": "d2"}])[0]
            if not d2.ended:
                suffix2 = (ext.p1_chunks if side == "p1" else ext.p2_chunks) + \
                          (d2.p1_chunks if side == "p1" else d2.p2_chunks)
                ob = succ_obs(pfx, suffix2, [ext.label, 0], decision=anchor + 2)
                assert ob is not None, (
                    f"{record.battle_tag}: depth-2 composed chunks failed to materialize a successor obs")
                n_checked += 1

        # (4) WARM-SESSION REUSE — re-open the SAME process's root (the driver clears its node cache)
        # and re-expand the chosen arm; it must reproduce the first result (no cross-search leakage).
        if not ch.ended:
            root2 = ss.open_root(turn, record=record)
            assert root2.recorded_choices == root.recorded_choices, (
                f"{record.battle_tag}: re-open_root on the warm session changed the decision point")
            ch2 = ss.expand_many([{"node_id": root2.node_id, "recorded_exact": True,
                                   "seed": "original", "label": chosen}])[0]
            ob_reuse = succ_obs(root2.prefix_p1_chunks if side == "p1" else root2.prefix_p2_chunks,
                                ch2.p1_chunks if side == "p1" else ch2.p2_chunks, [chosen])
            assert ob_reuse is not None and np.array_equal(ob_reuse, ob_recorded), (
                f"{record.battle_tag}: warm-session reuse (2nd open_root) != first result BIT-FOR-BIT")
            n_checked += 1
    return n_checked


def main(n_battles: int, impl: str = "node", record_impl: str = "node") -> None:
    print(f"search-server clone parity fuzz — {n_battles} battles "
          f"[driver={impl}, recorded on {record_impl}]", flush=True)
    total, n = 0, 0
    for _ in range(n_battles):
        with tempfile.TemporaryDirectory(prefix="search_clone_parity_") as out_dir:
            record, summary, npz = _record_one_battle(out_dir, record_impl=record_impl)
            c = _check_battle(record, summary, npz, impl=impl)
            total += c
            n += 1
            print(f"  {record.battle_tag}: clone ≡ reroll_many + value_crn anchor + depth-2 "
                  f"({c} obs checks bit-for-bit)", flush=True)
    assert n == n_battles
    assert total >= 1, ("no obs check ran — every anchor terminal? re-run / raise n_battles "
                        "(this floor counts OBS comparisons only; it was uncountable until "
                        "n_checked stopped being seeded at 1)")
    print(f"\nPASS — {n} battles: serializeBattle clone is byte-identical to reroll_many at the obs, "
          f"reproduces the recorded next state, and composes to depth 2 ({total} checks)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n_battles", nargs="?", type=int, default=4)
    ap.add_argument("--impl", choices=("node", "rust"), default="node",
                    help="Which OFFLINE driver serves the clone-and-branch search + the "
                         "reroll_many oracle it is checked against (default node — the "
                         "historical behavior). 'rust' runs the same assertions through "
                         "src/rust_sim/src/bin/search_driver.rs.")
    ap.add_argument("--record-impl", choices=("node", "rust"), default="node",
                    help="Which LIVE bridge plays the battle the record comes from. Kept "
                         "separate from --impl on purpose: a rust-recorded battle replayed by "
                         "the node driver (and vice versa) is the cross-impl case a mixed "
                         "run actually hits — training on rust, forensics on node.")
    a = ap.parse_args()
    main(a.n_battles, impl=a.impl, record_impl=a.record_impl)
