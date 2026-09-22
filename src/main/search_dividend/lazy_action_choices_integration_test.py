"""A successor's token map is built ONLY when something reads it — and is the same map when it is.

`gen3_lazy_action_choices_v1`. ``action_choices`` is the REAL action mapper run over every legal
index of a successor, and the only readers are :meth:`~main.search_dividend.deepen.TreeNode.
expandable`, :func:`~main.search_dividend.deepen.plan_beam`'s arm-count estimate and
``search._expand_ply``'s own loop — all three inside ``_score_world``'s ``while ply < md``. A
depth-1 decision therefore built one per arm and read none.

**Two claims, and the second is why this is not just a perf note.** A lazy map that is never read
saves the work; a lazy map that is read must produce EXACTLY what the eager one did, or the
deepener branches on different tokens and the search decides something else. So this gate asserts
both: unread at depth 1, and byte-equal to the eager map at depth 2.

⚠️ **The VIEW road only.** The protocol road's map is built by the replay player at the decision
itself (``map_actions_at``), where the poke-env battle still stands; by the time a caller could ask
for it that battle has been restored out from under it. Deferring there is not a lazy read, it is
a second replay.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from agents.training.view_successor import LazyTokens
from main.search_dividend.budget import WidthCaps
from main.search_dividend.search import SearchConfig, SearchEngine

pytestmark = [pytest.mark.sim, pytest.mark.integration]

#: 🚨 PINNED — see ``fork_sharing_parity_integration_test._KEY``. A collected test takes the SAME
#: battle every run; only the sweeps record a random one.
_KEY = 3


def _decision():
    """One mid-battle move selection off the reproducible fixture battle."""
    import agents.battle.one_sided_view_parity_fuzz_test as G
    from main.search_dividend import determinize as dz
    from utils.bridge.search_session import SearchSession

    with tempfile.TemporaryDirectory() as td:
        record, summary, npz = G._record_one_battle(td, "rust", _KEY)
    actions = np.asarray(npz["actions"], dtype=int)
    invs = summary["invocations"]
    cand = [i for i, iv in enumerate(invs)
            if iv.get("phase") == "move_selection" and int(iv["turn"]) > 1]
    assert cand, "the fixture battle produced no mid-battle move selection"
    side = record.side_of(record.trainee_username)
    anchor = cand[len(cand) // 3]
    turn = int(invs[anchor]["turn"])
    our_history = [int(x) for x in actions[:anchor]]
    with SearchSession(record, impl="rust") as ss:
        pfx = getattr(ss.open_root(turn), f"prefix_{side}_chunks")
    tokens = G._choice_map(record, side, our_history, pfx, anchor)
    assert tokens, "the anchor decision offered no legal action"
    opp_true = record.packed_team("p2" if side == "p1" else "p1")
    return record, side, turn, our_history, tokens, dz.chunks_to_lines(pfx), opp_true


def _run(fx, *, max_depth: int):
    """``(engine result, [every successor's token map, in order])`` — the maps captured as the
    objects themselves, so the test can ask whether each was ever BUILT."""
    import main.search_dividend.search as S

    record, side, turn, our_history, tokens, observed, opp_true = fx
    seen = []
    orig = S.SearchEngine._materialize

    def recording(self, ctx, branches, branch_of, parents, acts, arm_suffix, arm_view,
                  dec_i, ply, widths):
        out = orig(self, ctx, branches, branch_of, parents, acts, arm_suffix, arm_view,
                   dec_i, ply, widths)
        for leaf in out.values():
            seen.append(leaf.action_choices)
        return out

    S.SearchEngine._materialize = recording
    eng = SearchEngine(
        model=None, mappings=None,
        cfg=SearchConfig(arm="oracle", budget_s=1e9, seed=7, max_depth=max_depth,
                         search_impl="rust", materializer="view",
                         caps=WidthCaps(m_opp=2, k_worlds=1, r_dice=1)),
        pool_packed=[])
    eng._score_batch = lambda obs, masks: (                      # type: ignore[assignment]
        np.asarray(obs, dtype=np.float64).sum(axis=1), "fake")
    try:
        res = eng.choose(record=record, side=side, turn=turn, our_history=list(our_history),
                         our_tokens=dict(tokens), observed_our_lines=list(observed),
                         pub=None, policy_action=next(iter(tokens)),
                         opp_true_packed=opp_true)
    finally:
        S.SearchEngine._materialize = orig
        eng.close()
    return res, seen


def test_a_depth_one_search_never_builds_a_successor_token_map():
    fx = _decision()
    res, maps = _run(fx, max_depth=1)
    assert res.fallback is None, f"the search declined ({res.fallback}) — the gate is vacuous"
    lazy = [m for m in maps if isinstance(m, LazyTokens)]
    assert lazy, ("no successor came back with a LazyTokens — either every arm fell back to the "
                  "protocol road or the laziness is gone; either way this proves nothing")
    built = [m for m in lazy if m.materialized]
    assert not built, (
        f"{len(built)} of {len(lazy)} successor token maps were BUILT at depth 1, where nothing "
        f"reads one — the deferral is not reaching the mapper")
    print(f"  depth 1: {len(lazy)} lazy maps, {len(built)} built, "
          f"arms_scored={res.widths.arms_scored}")


def test_a_deepened_search_builds_the_same_tokens_the_eager_map_had():
    """The read path must be identical, not merely present.

    The comparison is against ``_choice_map`` run directly on the successor's own read-models —
    the same primitive the eager code called at materialization time — so a deferral that captured
    the wrong ``vbattle`` / ``mask`` / ``legal`` fails here rather than silently offering a
    deeper ply a different action set."""
    from agents.training.view_successor import _choice_map

    fx = _decision()
    res, maps = _run(fx, max_depth=2)
    assert res.fallback is None, f"the search declined ({res.fallback}) — the gate is vacuous"
    lazy = [m for m in maps if isinstance(m, LazyTokens)]
    assert lazy, "no successor came back with a LazyTokens"
    built = [m for m in lazy if m.materialized]
    assert built, ("depth 2 was asked for and NO token map was built — the search never deepened, "
                   "so the read path is untested")
    for m in built:
        assert dict(m) == dict(m._fn()), (
            "a deferred token map does not equal what its own producer returns — the captured "
            "read-models are not the successor's")
    assert _choice_map is not None
    print(f"  depth 2: {len(lazy)} lazy maps, {len(built)} built, "
          f"depth_realized={res.widths.depth_realized}")
