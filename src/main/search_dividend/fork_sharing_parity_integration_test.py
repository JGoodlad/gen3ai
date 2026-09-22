"""ONE view fork per DECISION must produce exactly what one fork per WORLD produced.

`gen3_one_fork_per_decision_v1`. The ply-1 ``ViewSuccessorFactory`` is ``materialize_branches``'
first half over the one-sided PREFIX, our action history and OUR packed team — and a determinized
world changes only the OPPONENT's team, so the K worlds of one decision were each replaying one
identical prefix. :meth:`~main.search_dividend.search.SearchEngine._root_fork` now builds it once
and keys the cache on the prefix BYTES.

🚨 **What could go wrong is not "the prefixes differ" — the cache key catches that.** It is that a
factory is not actually reusable: if ``successor()`` mutated the event folder, the frozen tracker
or the prior-event list, world 2 would inherit world 1's ply and every arm after the first would
be scored at a board nobody was ever on. That failure is SILENT — the obs stays well-formed, the
scores stay plausible, and only a byte comparison against the un-shared road can see it. So this
gate runs the real :class:`~main.search_dividend.search.SearchEngine` over one seeded decision
twice, with sharing ON and with it FORCED OFF, and compares every successor's observation bytes in
order.

The scorer is the pure ``obs.sum``, the same choice
``materializer_parity_integration_test`` makes and for the same reason: a trained net is a
contraction and can map two different vectors onto one score, while a sum cannot.
"""

from __future__ import annotations

import tempfile
from typing import List

import numpy as np
import pytest

from main.search_dividend.budget import WidthCaps
from main.search_dividend.search import SearchConfig, SearchEngine

pytestmark = [pytest.mark.sim, pytest.mark.integration]

#: 🚨 PINNED, and the default is NOT. ``_record_one_battle``'s ``fixed_key`` defaults to ``None``,
#: which records a FRESH RANDOM battle — right for the sweeps, wrong for a collected gate, which
#: can then ride main red on a draw nobody has seen (the project's fuzz rule,
#: ``designs/rust_sim/one_sided_view.md`` §4). Pass it explicitly, every time.
_KEY = 5


class _NeverHits(dict):
    """A fork cache that stores but never serves — the CONTROL, i.e. one fork per world.

    A dict subclass rather than a stub on ``_root_fork``: the control must run the very code the
    experiment runs, differing only in whether the lookup succeeds."""

    def get(self, key, default=None):          # noqa: D102 - see the class docstring
        return default


def _engine(pool, *, share: bool, materializer: str = "view") -> SearchEngine:
    cfg = SearchConfig(
        arm="honest", budget_s=1e9, seed=7, max_depth=1, search_impl="rust",
        materializer=materializer, caps=WidthCaps(m_opp=2, k_worlds=3, r_dice=1))
    eng = SearchEngine(model=None, mappings=None, cfg=cfg, pool_packed=list(pool))
    eng._score_batch = lambda obs, masks: (                      # type: ignore[assignment]
        np.asarray(obs, dtype=np.float64).sum(axis=1), "fake")
    if not share:
        eng._fork_cache = _NeverHits()
    return eng


def _run(pool, record, side, turn, our_history, tokens, observed, *, share: bool,
         materializer: str = "view"):
    """``(per-action scores, chosen action, widths, [successor obs bytes, in order])``."""
    import agents.training.view_successor as VS

    seen: List[bytes] = []
    orig = VS.ViewSuccessorFactory.successor

    def recording(self, payload, chunks, action):
        got = orig(self, payload, chunks, action)
        if got is not None:
            seen.append(np.asarray(got.obs, dtype=np.float32).tobytes())
        return got

    VS.ViewSuccessorFactory.successor = recording
    eng = _engine(pool, share=share, materializer=materializer)
    try:
        res = eng.choose(record=record, side=side, turn=turn, our_history=list(our_history),
                         our_tokens=dict(tokens), observed_our_lines=list(observed),
                         pub=None, policy_action=next(iter(tokens)), opp_true_packed=None)
    finally:
        VS.ViewSuccessorFactory.successor = orig
        eng.close()
    return dict(res.scores or {}), int(res.action), res.widths, seen


def test_one_shared_fork_scores_every_world_exactly_as_a_per_world_fork_did():
    import agents.battle.one_sided_view_parity_fuzz_test as G
    from main.search_dividend import determinize as dz
    from main.search_dividend.__main__ import _pool
    from utils.bridge.search_session import SearchSession

    with tempfile.TemporaryDirectory() as td:
        record, summary, npz = G._record_one_battle(td, "rust", _KEY)
    actions = np.asarray(npz["actions"], dtype=int)
    invs = summary["invocations"]
    cand = [i for i, iv in enumerate(invs)
            if iv.get("phase") == "move_selection" and int(iv["turn"]) > 1]
    assert cand, "the fixture battle produced no mid-battle move selection"
    side = record.side_of(record.trainee_username)
    # THE FULL POOL, and an EARLY anchor. `build_determinizations` needs pool-consistent
    # completions for the opponent's still-hidden slots: late in a battle most of the team has
    # been revealed and K collapses to 1, which makes this gate vacuous rather than failing it.
    pool = _pool(0)

    compared = 0
    for frac in (0.0, 0.15, 0.3):
        anchor = cand[int(len(cand) * frac)]
        turn = int(invs[anchor]["turn"])
        our_history = [int(x) for x in actions[:anchor]]
        with SearchSession(record, impl="rust") as ss:
            pfx = getattr(ss.open_root(turn), f"prefix_{side}_chunks")
        tokens = G._choice_map(record, side, our_history, pfx, anchor)
        if not tokens:
            continue
        observed = dz.chunks_to_lines(pfx)

        ctl_scores, ctl_act, ctl_w, ctl_obs = _run(
            pool, record, side, turn, our_history, tokens, observed, share=False)
        exp_scores, exp_act, exp_w, exp_obs = _run(
            pool, record, side, turn, our_history, tokens, observed, share=True)

        print(f"  turn {turn}: gated={exp_w.worlds_gated_ok} requested={exp_w.worlds_requested} "
              f"open_failed={exp_w.worlds_open_failed} gate_failed={exp_w.worlds_gate_failed} "
              f"successors={len(exp_obs)} arms={exp_w.arms_scored} tokens={len(tokens)}")
        if int(exp_w.worlds_gated_ok) < 2 or not exp_obs:
            continue                       # one world, or no view arm — nothing to share
        compared += 1

        # NON-VACUITY: the experiment must really have shared, the control must really not have.
        assert exp_w.fork_cache_hit > 0, (
            f"turn {turn}: {exp_w.worlds_gated_ok} worlds and NO cache hit — the shared run "
            f"built a fork per world, so this comparison proves nothing")
        assert ctl_w.fork_cache_hit == 0, "the control served a cached fork"
        assert ctl_w.fork_cache_miss == exp_w.fork_cache_hit + exp_w.fork_cache_miss

        assert len(exp_obs) == len(ctl_obs), (
            f"turn {turn}: {len(exp_obs)} shared successors vs {len(ctl_obs)} un-shared")
        for i, (a, b) in enumerate(zip(ctl_obs, exp_obs)):
            assert a == b, (
                f"turn {turn}: successor #{i} differs between the shared fork and the per-world "
                f"fork — the factory is NOT reusable across worlds")
        assert exp_scores == ctl_scores, f"turn {turn}: per-action scores differ"
        assert exp_act == ctl_act, f"turn {turn}: chose {exp_act} shared vs {ctl_act} un-shared"
        assert exp_w.arms_scored == ctl_w.arms_scored
        assert exp_w.view_arms == ctl_w.view_arms
        print(f"  turn {turn}: worlds={exp_w.worlds_gated_ok} successors={len(exp_obs)} "
              f"fork_hit={exp_w.fork_cache_hit} fork_miss={exp_w.fork_cache_miss} "
              f"view_arms={exp_w.view_arms}")

    assert compared >= 1, (
        "no decision produced two gated worlds with a view arm — the gate is vacuous")


def test_one_shared_prefix_replay_serves_every_world_on_the_protocol_road():
    """The same claim for ``BranchFork`` — the PROTOCOL road's shared-prefix fork.

    🚨 It is not a second copy of the test above dressed differently. The view fork carries a
    frozen tracker and an event folder and is never mutated; this one carries a LIVE poke-env
    replay player that every arm mutates and ``_PlayerSnapshot.restore`` resets. Reusing it
    across worlds asserts that the restore is complete — a field the snapshot forgets would leak
    world 1's last arm into world 2's first, and the obs would still be well-formed.

    It matters to the VIEW road too: every D10 fallback arm runs through here."""
    import agents.battle.one_sided_view_parity_fuzz_test as G
    from main.search_dividend import determinize as dz
    from main.search_dividend.__main__ import _pool
    from utils.bridge.search_session import SearchSession

    with tempfile.TemporaryDirectory() as td:
        record, summary, npz = G._record_one_battle(td, "rust", _KEY)
    actions = np.asarray(npz["actions"], dtype=int)
    invs = summary["invocations"]
    cand = [i for i, iv in enumerate(invs)
            if iv.get("phase") == "move_selection" and int(iv["turn"]) > 1]
    assert cand, "the fixture battle produced no mid-battle move selection"
    side = record.side_of(record.trainee_username)
    pool = _pool(0)

    compared = 0
    for frac in (0.0, 0.15, 0.3):
        anchor = cand[int(len(cand) * frac)]
        turn = int(invs[anchor]["turn"])
        our_history = [int(x) for x in actions[:anchor]]
        with SearchSession(record, impl="rust") as ss:
            pfx = getattr(ss.open_root(turn), f"prefix_{side}_chunks")
        tokens = G._choice_map(record, side, our_history, pfx, anchor)
        if not tokens:
            continue
        observed = dz.chunks_to_lines(pfx)

        ctl_scores, ctl_act, ctl_w, _ = _run(
            pool, record, side, turn, our_history, tokens, observed,
            share=False, materializer="protocol")
        exp_scores, exp_act, exp_w, _ = _run(
            pool, record, side, turn, our_history, tokens, observed,
            share=True, materializer="protocol")
        if int(exp_w.worlds_gated_ok) < 2 or exp_w.arms_scored <= 0:
            continue
        compared += 1
        assert exp_w.branch_fork_cache_hit > 0, (
            f"turn {turn}: {exp_w.worlds_gated_ok} worlds and NO branch-fork hit — the shared "
            f"run replayed the prefix per world, so this comparison proves nothing")
        assert ctl_w.branch_fork_cache_hit == 0, "the control served a cached branch fork"
        assert exp_scores == ctl_scores, (
            f"turn {turn}: per-action scores differ between the shared prefix replay and the "
            f"per-world one — `_PlayerSnapshot.restore` does not fully reset the player")
        assert exp_act == ctl_act
        assert exp_w.arms_scored == ctl_w.arms_scored
        print(f"  turn {turn}: worlds={exp_w.worlds_gated_ok} arms={exp_w.arms_scored} "
              f"branch_hit={exp_w.branch_fork_cache_hit} "
              f"branch_miss={exp_w.branch_fork_cache_miss}")

    assert compared >= 1, "no decision produced two gated worlds — the gate is vacuous"
