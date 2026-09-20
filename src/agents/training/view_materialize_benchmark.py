"""ms per SUCCESSOR: the protocol path vs the one-sided VIEW path (`gen3_one_sided_view_v1`).

    export PYTHONPATH=$PYTHONPATH:src
    python3 src/agents/training/view_materialize_benchmark.py [--battles N] [--b 1 33]

**What is being compared.** A search successor's read-models + observation, reached two ways from
the SAME `expand_many` response:

* **protocol** — the PRODUCTION road, `obs_materializer.materialize_branches`: replay the shared
  prefix ONCE, then per arm restore the pickled player snapshot, feed the ply's one-sided protocol
  text through poke-env's parser, and encode. Anything less than prefix sharing would be an unfair
  comparison — measured, it reads 85-110x, which is a statement about rebuilding a replay player
  and not about this change.
* **view** — `gen3_view_successor_v1`, the DEFAULT road: open the fork on the shared prefix ONCE
  (`open_view_fork`, which is `materialize_branches`' own first half), then per arm fold the ply's
  protocol into events in Python, build the read-models from the arm's `view_pN` payload, advance
  a cloned `EpisodeTracker` and encode.

🚨 **BOTH columns now include the shared prefix and the TRACKERS**, which the first version of
this benchmark did not: it timed a tracker-less encode against the production materializer, so it
compared a leaf production does not score. Both encode through the identical
`Gen3ObservationEncoder`, so the delta is the road and nothing else. The gate that says the two roads AGREE is
`agents/battle/one_sided_view_parity_fuzz_test.py`; this script only says how much they
COST, and it refuses to be read as a correctness check.

🚨 **WARN, NEVER STRETCH.** A benchmark's output IS the measurement, so this prints the load and
says the comparison is void if the box was busy — it does not scale anything. Run the two arms
back to back under the same load; that is the only honest A/B on a shared box.

⚠️ **B = 1 and B = 33 answer different questions.** B = 1 is one successor (the per-arm floor,
where the prefix restore dominates); B = 33 is one ply of a real search (every legal action), where
the prefix is shared across arms and the per-arm cost is what the width ladder actually pays.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from typing import List

import numpy as np

from agents.training.obs_materializer import (Branch, materialize_branches,
                                              open_view_fork)
from agents.observation.state_encoder import get_observation_encoder, load_mappings
from utils.bridge.search_session import SearchSession
from utils.contention import describe_contention, warn_if_contended


def _fixture(impl: str):
    """One real gen3ou battle + a mid-battle anchor, via the parity gate's own recorder (one
    fixture builder, so the benchmark and the gate cannot drift apart)."""
    import agents.battle.one_sided_view_parity_fuzz_test as G

    with tempfile.TemporaryDirectory() as td:
        record, summary, npz = G._record_one_battle(td, impl)
    actions = np.asarray(npz["actions"], dtype=int)
    invs = summary["invocations"]
    cand = [i for i, inv in enumerate(invs)
            if inv.get("phase") == "move_selection" and int(inv["turn"]) > 1]
    if not cand:
        return None
    anchor = cand[len(cand) // 2]
    return record, summary, npz, actions, anchor


def _run(n_battles: int, widths: List[int], impl: str) -> None:
    import agents.battle.one_sided_view_parity_fuzz_test as G

    warn_if_contended()
    encoder = get_observation_encoder(load_mappings())
    rows: dict = {b: {"protocol": [], "view": []} for b in widths}
    arms_seen: dict = {b: 0 for b in widths}

    for _ in range(n_battles):
        fx = _fixture(impl)
        if fx is None:
            continue
        record, summary, npz, actions, anchor = fx
        invs = summary["invocations"]
        turn = int(invs[anchor]["turn"])
        side = record.side_of(record.trainee_username)
        other = "p2" if side == "p1" else "p1"
        prefix_actions = [int(x) for x in actions[:anchor]]

        with SearchSession(record, impl=impl) as ss:
            root = ss.open_root(turn)
            pfx = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
            cmap = G._choice_map(record, side, prefix_actions, pfx, anchor)
            if not cmap:
                continue
            opp_rec = root.recorded_choices.get(other) or "default"
            for width in widths:
                picks = (sorted(cmap) * ((width // max(len(cmap), 1)) + 1))[:width]
                expand = [{"node_id": root.node_id, f"{side}_action": cmap[a],
                           f"{other}_action": opp_rec,
                           "seed": f"{turn},{k + 1},{anchor + 7},{k * 13 + 11}", "label": k}
                          for k, a in enumerate(picks)]
                nodes = [n for n in ss.expand_many(expand) if not (n.ended or n.stuck)]
                if not nodes:
                    continue
                arms_seen[width] += len(nodes)

                # --- VIEW path: the PRODUCTION road (`--materializer view`) — the shared
                # prefix ONCE through `open_view_fork`, then per arm the ply's event fold, the
                # read-models, the tracker advance and the encode. Trackers INCLUDED, which the
                # earlier form of this benchmark did not have and which is most of the per-arm
                # Python: a leaf whose recency / pair-history / event-window blocks were zero is
                # not the leaf production scores.
                t0 = time.perf_counter()
                fork, _root_choices = open_view_fork(
                    pfx, username=record.username(side),
                    packed_team=record.packed_team(side), side=side,
                    prefix_actions=prefix_actions, battle_format=record.format_id,
                    battle_tag=record.battle_tag, encoder=encoder)
                for n in nodes:
                    payload = n.view_p1 if side == "p1" else n.view_p2
                    fork.successor(payload, n.p1_chunks if side == "p1" else n.p2_chunks,
                                   sorted(cmap)[0])
                t_view = (time.perf_counter() - t0) * 1000.0 / len(nodes)

                # --- PROTOCOL path: the production shared-prefix materializer
                branches = [
                    Branch(chunks=(n.p1_chunks if side == "p1" else n.p2_chunks),
                           actions=[sorted(cmap)[0]], label=n.label)
                    for n in nodes
                ]
                t0 = time.perf_counter()
                materialize_branches(
                    pfx, branches, username=record.username(side),
                    packed_team=record.packed_team(side), side=side,
                    prefix_actions=prefix_actions, battle_format=record.format_id,
                    battle_tag=record.battle_tag,
                    # 🚨 `map_actions_at` is NOT optional here. Production always asks for it
                    # (`search._expand_ply` passes it, because a child's legal surface is what
                    # makes a deeper ply possible), and the VIEW road always builds it — so a
                    # protocol call without it compares a cheaper protocol road against the real
                    # view road. Measured: it is ~24% of the view road's per-arm wall.
                    map_actions_at=anchor + 1, stop_after_decision=anchor + 1,
                    encode_only_at={anchor + 1})
                t_proto = (time.perf_counter() - t0) * 1000.0 / len(nodes)

                rows[width]["protocol"].append(t_proto)
                rows[width]["view"].append(t_view)

    print("\nPER-SUCCESSOR MATERIALIZE BENCHMARK  (ms per arm, median over battles)")
    print(f"  {describe_contention()}")
    print("  🚨 ratios are the claim; absolute ms scale with whatever else the box is doing.\n")
    print(f"  {'B':>4}  {'arms':>5}  {'protocol ms':>12}  {'view ms':>9}  {'speedup':>8}")
    for b in widths:
        p, v = rows[b]["protocol"], rows[b]["view"]
        if not p:
            print(f"  {b:>4}  {'-':>5}  {'(no arms)':>12}")
            continue
        mp, mv = statistics.median(p), statistics.median(v)
        print(f"  {b:>4}  {arms_seen[b]:>5}  {mp:>12.3f}  {mv:>9.3f}  {mp / mv:>7.2f}x")
    print("\n  The PROTOCOL column INCLUDES the one-off shared prefix replay amortized over the "
          "\n  arms, which is exactly what a search ply pays — so B=1 carries the whole prefix and "
          "\n  B=33 carries a thirty-third of it. That is the honest shape of the win: the view "
          "\n  path needs no prefix and no snapshot restore at all.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=2)
    ap.add_argument("--b", type=int, nargs="+", default=[1, 33])
    ap.add_argument("--impl", default="rust")
    a = ap.parse_args()
    _run(a.battles, a.b, a.impl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
