"""Gates for the FORK ARM's COMMON RANDOM NUMBERS and its substitute resolution (`gen3_fork_v1`).

The pairing is the arm's answer to a banked finding: *"`cf_q_labels` pairs only the SIM DICE and
leaves both sides sampling at temperature 1.0 … A label factory that pairs the dice but not the
policy draws may be teaching the head noise"*
(`designs/research_state/measurements/paired_refit_discrimination_2026-09-14/`). These are the
arithmetic gates; `fork_crn_sim_test` proves the same claim on the real bridge.
"""
from __future__ import annotations

import torch as th

from agents.training.fork_crn import branch_policy_seed, player_seeds


def test_every_branch_of_one_fork_gets_the_SAME_stream():
    """The whole pairing: the seed depends on the fork and the side, NOT on the branch."""
    a = player_seeds("battle7:12:3", "dice_and_draws")
    b = player_seeds("battle7:12:3", "dice_and_draws")
    assert a == b


def test_the_two_SIDES_get_different_streams():
    """Both sides of one branch sharing a stream would couple the trainee's coin to the
    opponent's — a correlation that is not in the training ecology, i.e. a second confound
    smuggled in under a fix for the first."""
    t, o = player_seeds("battle7:12:3", "dice_and_draws")
    assert t != o


def test_two_different_forks_get_different_streams():
    assert player_seeds("a", "dice_and_draws") != player_seeds("b", "dice_and_draws")
    assert branch_policy_seed("a", "trainee") != branch_policy_seed("b", "trainee")


def test_dice_mode_leaves_the_shared_default_generator_alone():
    """`dice` is the `cf_q_labels` regime and the CONTROL — `RLPlayer` documents `policy_seed
    None` as 'use torch's shared default generator', i.e. today's behaviour."""
    assert player_seeds("battle7:12:3", "dice") == (None, None)


def test_seeds_are_inside_torchs_generator_range():
    for salt in ("x", "y" * 200, "battle_9:41:12"):
        for side in ("trainee", "opponent"):
            s = branch_policy_seed(salt, side)
            assert 0 <= s < (1 << 62)
            th.Generator().manual_seed(int(s))          # must not raise


def test_an_identically_seeded_generator_draws_the_identical_action_sequence():
    """The mechanism, exercised through the exact call `RLPlayer._predict_best_action` makes:
    `torch.multinomial(cat.probs, 1, True, generator=gen)`. One uniform per decision, whatever the
    state, so two branches' streams cannot slip relative to each other."""
    probs = th.tensor([[0.2, 0.3, 0.5]])
    seed = branch_policy_seed("fork:1", "trainee")

    def draw(n, s):
        g = th.Generator(); g.manual_seed(int(s))
        return [int(th.multinomial(probs, 1, True, generator=g).item()) for _ in range(n)]

    assert draw(20, seed) == draw(20, seed)
    assert draw(20, seed) != draw(20, branch_policy_seed("fork:2", "trainee"))


# ── the substitute, resolved against the LIVE legal set ──────────────────────────────────────
class _FakeOrder:
    def __init__(self, msg):
        self.message = msg


class _FakePlayer:
    """A NON-gen3 player (no `embed_battle`), so `install_scripted_prefix` takes its simple path."""

    def __init__(self):
        self.calls = []
        self.choose_move = lambda battle: _FakeOrder("/choose LIVE")

    def action_to_order(self, idx, battle):
        self.calls.append((int(idx), battle))
        return _FakeOrder(f"/choose move {idx}")


class _FakeBattle:
    def __init__(self, turn):
        self.turn = turn
        self.force_switch = False


class _FakeRecord:
    commands = (("p1", "move 1"), ("p2", "move 2"), ("p1", "move 3"))


def test_a_CALLABLE_substitute_is_resolved_through_the_players_own_action_mapper():
    """A buffer row knows its action as an INDEX, and the index -> choice mapping is a function of
    the LIVE legal set. Re-deriving it from the record would be a second implementation of
    `action_to_order` that could disagree with the one the trainee itself used."""
    from utils.bridge.counterfactual import install_scripted_prefix

    p = _FakePlayer()
    state = install_scripted_prefix(
        p, side="p1", record=_FakeRecord(), divergence_turn=4,
        substitute_choice=lambda player, battle: player.action_to_order(7, battle).message[
            len("/choose "):],
        is_our_side=True)
    assert state["substitute_resolved"] is None           # not resolved until the fork decision
    order = p.choose_move(_FakeBattle(turn=4))
    assert order.message == "/choose move 7"
    assert state["substitute_resolved"] == "move 7"
    assert p.calls and p.calls[0][0] == 7


def test_a_STRING_substitute_is_reported_unchanged_before_the_fork_is_reached():
    from utils.bridge.counterfactual import install_scripted_prefix

    p = _FakePlayer()
    state = install_scripted_prefix(p, side="p1", record=_FakeRecord(), divergence_turn=4,
                                    substitute_choice="move 2", is_our_side=True)
    assert state["substitute_resolved"] == "move 2"


def test_a_substitute_that_cannot_be_resolved_RAISES_rather_than_replaying_the_recorded_move():
    """Deliberately unguarded: a swallowed failure would fall through to the recorded choice and
    silently produce a line that is NOT the counterfactual the caller asked for."""
    from utils.bridge.counterfactual import install_scripted_prefix

    p = _FakePlayer()

    def boom(_player, _battle):
        raise RuntimeError("no such legal action")

    install_scripted_prefix(p, side="p1", record=_FakeRecord(), divergence_turn=4,
                            substitute_choice=boom, is_our_side=True)
    try:
        p.choose_move(_FakeBattle(turn=4))
    except RuntimeError as exc:
        assert "no such legal action" in str(exc)
    else:                                                  # pragma: no cover
        raise AssertionError("the failing substitute was swallowed")
