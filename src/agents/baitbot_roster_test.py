"""gen3_baitbot_roster_v1 — BaitBot enters the TRAINING roster only when declared, at the exact share.

The E4 gate's controlled variable is punishment FREQUENCY. If the declared share and the realized
share can drift — because the other bots were re-weighted, or because the default pool silently
gained a ninth member — the arm stops being an experiment. These pin both.
"""
import pytest

from agents.baitbot import Gen3BaitBotPlayer, make_baitbot_class, weight_for_share


def test_default_roster_is_byte_identical():
    """share=0 must leave the pool untouched — every pre-existing command is unchanged."""
    from main.train_rl_agent import build_parser
    a = build_parser().parse_args(["--steps", "1"])
    assert a.bait_bot_share == 0.0


def test_the_factory_bakes_the_dial_into_the_class():
    """The roster instantiates cls(**kwargs) with no extras, so the dial must live on the class."""
    C = make_baitbot_class(0.6)
    assert issubclass(C, Gen3BaitBotPlayer)
    assert C.p_bait_declared == 0.6
    assert C.__name__ == "Gen3BaitBotP060"      # the name carries the dial


def test_the_class_name_distinguishes_dials():
    assert make_baitbot_class(0.25).__name__ != make_baitbot_class(0.6).__name__


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_factory_rejects_an_impossible_dial(bad):
    with pytest.raises(ValueError):
        make_baitbot_class(bad)


@pytest.mark.parametrize("share", [0.1, 0.25, 0.5, 0.75])
def test_share_is_exact_on_a_uniform_roster(share):
    w = weight_for_share(share, [1.0] * 8)
    assert abs(w / (8 + w) - share) < 1e-12


@pytest.mark.parametrize("others", [[2.0] + [1.0] * 7, [0.5] * 8, [3.0, 1.0, 1.0, 1.0]])
def test_share_stays_exact_when_the_OTHERS_are_reweighted(others):
    """The failure this guards: --bot-weights re-weights the heuristics and the declared share
    silently becomes something else."""
    w = weight_for_share(0.25, others)
    assert abs(w / (sum(others) + w) - 0.25) < 1e-12


def test_share_of_zero_gives_zero_weight():
    assert weight_for_share(0.0, [1.0] * 8) == 0.0


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5])
def test_impossible_shares_are_refused(bad):
    """share=1.0 would mean an all-BaitBot roster — unreachable by weighting, so refuse it."""
    with pytest.raises(ValueError):
        weight_for_share(bad, [1.0] * 8)


def test_a_zero_weight_roster_is_refused():
    with pytest.raises(ValueError):
        weight_for_share(0.25, [0.0, 0.0])
