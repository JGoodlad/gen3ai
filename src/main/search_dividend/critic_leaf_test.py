"""THE SEARCH LEAF UNDER `--critic winprob` — `gen3_winprob_critic_mode_v1`, design gap **B10**.

The design's §3.10 asks for three things and this file pins all three, plus the one that decides
whether the battery RUNS at all on a winprob checkpoint.

1. `--defensive-leaf`'s legal set collapses to `winprob` (probe G's `value` control arm does not
   exist on a critic whose two readouts are the same number).
2. `--score auto` "dies" — as a RESOLUTION to `win_prob`, announced, not as a refusal. The
   difference is the whole design decision and is asserted here rather than described: `auto` is
   the CLI default, so refusing it would make the flagless invocation a usage error on every
   winprob checkpoint, and the battery is required to run end to end.
3. `check_leaf` is KEPT and still guards, even though it becomes vacuous — deleting a guard
   because it currently has nothing to catch is how the choice-reject allowlist entry outlived
   its own fix.

And the one that is not in the design: **the refusal has to happen at STARTUP.**
`SearchEngine.choose` catches every exception as a counted `search_error` fallback, so
`batch_scores`'s deep `ValueError` on the flagless default would turn every decision into a
policy fallback and report the arm's dividend as ~0 with nothing in the log saying the battery
never ran. That is exactly the "a default that can quietly become the losing arm" failure
`defensive.py` was written against, arriving through the error path instead of the value path.
"""
from __future__ import annotations

import pytest

from agents.model.critic_mode import CRITIC_SHAPED, CRITIC_WINPROB
from main.search_dividend import defensive as dfn


# ---------------------------------------------------------------------------------------------
# 1. the legal sets
# ---------------------------------------------------------------------------------------------

def test_shaped_keeps_both_leaves_and_all_three_scores():
    assert dfn.leaves_for_critic(CRITIC_SHAPED) == dfn.LEAVES == ("winprob", "value")
    assert dfn.scores_for_critic(CRITIC_SHAPED) == dfn.SCORES == ("auto", "value", "win_prob")


def test_winprob_collapses_both_to_the_one_readout_that_exists():
    assert dfn.leaves_for_critic(CRITIC_WINPROB) == ("winprob",)
    assert dfn.scores_for_critic(CRITIC_WINPROB) == ("win_prob",)


def test_an_unknown_mode_string_is_treated_as_shaped_not_as_an_error():
    """`is_winprob` is a `== "winprob"` test on anything stringable, so a checkpoint predating the
    flag (no `_critic_mode` at all) reads as the default and is untouched. That is the property
    that keeps every archived run loadable, and it is asserted rather than assumed."""
    assert dfn.leaves_for_critic(None) == dfn.LEAVES
    assert dfn.leaves_for_critic("some_future_mode") == dfn.LEAVES


# ---------------------------------------------------------------------------------------------
# 2. the resolution
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("score", ["auto", "value", "win_prob"])
@pytest.mark.parametrize("leaf", ["winprob", "value", None])
def test_shaped_is_returned_untouched_with_no_notes(score, leaf):
    """Not "narrowed to the same values" — never consulted. The `shaped` path must be able to run
    probe G's `value` control arm exactly as it always could."""
    assert dfn.resolve_for_critic(CRITIC_SHAPED, score, leaf) == (score, leaf, [])


def test_auto_resolves_to_win_prob_and_SAYS_SO():
    score, leaf, notes = dfn.resolve_for_critic(CRITIC_WINPROB, "auto", "winprob")
    assert score == "win_prob"
    assert leaf == "winprob"
    assert notes, "a resolution a reader has to infer from silence is not a resolution"
    assert "auto" in notes[0] and "win_prob" in notes[0]


def test_an_explicit_win_prob_resolves_to_itself_and_says_nothing():
    assert dfn.resolve_for_critic(CRITIC_WINPROB, "win_prob", "winprob") == (
        "win_prob", "winprob", [])


def test_an_explicit_value_score_is_REFUSED_on_a_winprob_policy():
    with pytest.raises(ValueError, match="in no loss graph"):
        dfn.resolve_for_critic(CRITIC_WINPROB, "value", "winprob")


def test_an_explicit_value_LEAF_is_REFUSED_on_a_winprob_policy():
    with pytest.raises(ValueError, match="only leaf"):
        dfn.resolve_for_critic(CRITIC_WINPROB, "auto", "value")


def test_a_leaf_of_None_is_the_non_defensive_path_and_is_never_judged():
    """`--root-strategy grid`/`racing` never build a `DefensiveConfig`, so there is no leaf to
    narrow — and inventing one would refuse a grid arm over a flag it does not use."""
    assert dfn.resolve_for_critic(CRITIC_WINPROB, "win_prob", None)[:2] == ("win_prob", None)


# ---------------------------------------------------------------------------------------------
# 3. check_leaf is KEPT, and is still a real guard on the shaped path
# ---------------------------------------------------------------------------------------------

def test_check_leaf_still_catches_the_silent_substitution_it_was_written_for():
    """Vacuous under `winprob` (one readout ⇒ no substitution is representable) — but the function
    must still EXIST and still fire, because the second readout may come back."""
    with pytest.raises(dfn.DefensiveLeafError):
        dfn.check_leaf("value", dfn.DefensiveConfig())
    dfn.check_leaf("win_prob", dfn.DefensiveConfig())          # the winprob arm passes


def test_the_resolved_pair_satisfies_check_leaf_by_construction():
    """The resolution and the guard must AGREE: whatever `resolve_for_critic` hands the CLI has to
    be a pair the seam accepts, or the battery would refuse its own resolved config per decision."""
    score, leaf, _ = dfn.resolve_for_critic(CRITIC_WINPROB, "auto", "winprob")
    assert dfn.resolve_score_mode(leaf) == score
    dfn.check_leaf(score, dfn.DefensiveConfig(leaf=leaf))


# ---------------------------------------------------------------------------------------------
# 4. the STARTUP seam — why the deep refusal is not enough on its own
# ---------------------------------------------------------------------------------------------

def test_the_deep_refusal_is_swallowed_by_the_engines_error_fallback():
    """THE REASON THE CLI RESOLVES AT STARTUP, demonstrated on the real code path rather than
    argued. `SearchEngine.choose` turns ANY exception from the search into a counted
    `search_error` that plays the policy action — correct (a search failure must never cost the
    battle) and, for a config error, silent: the arm would report a dividend of ~0 having never
    searched. So `batch_scores` refusing is the library backstop; the CLI must refuse first."""
    import inspect

    from main.search_dividend import search as srch

    src = inspect.getsource(srch.SearchEngine.choose)
    assert "except Exception" in src and "search_error" in src, (
        "if `choose` stopped swallowing exceptions this test's premise is gone — re-read whether "
        "the startup resolution is still the load-bearing one")


def test_the_cli_resolves_the_score_before_it_builds_a_SearchConfig():
    """A source pin on the ORDER: the resolution must land on `args` before `SearchConfig(...)`
    reads `args.score` / `args.defensive_leaf`, or the narrowed value would never reach the run."""
    import inspect

    from main.search_dividend import __main__ as m

    src = inspect.getsource(m.main)
    i_resolve = src.index("resolve_for_critic")
    i_cfg = src.index("SearchConfig(")
    assert i_resolve < i_cfg, "the critic resolution must precede SearchConfig construction"
    assert "model.policy" in src[i_resolve - 400:i_resolve], (
        "the mode must be read off the LIVE policy, never a config file")
