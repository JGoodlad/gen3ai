"""The trace-SELECTION contract (`gen3_trace_selection_manifest_v1`).

Pure — no torch, no filesystem, no simulator. The recorder-side and consumer-side tests live with
their own subjects (`eval_manifest_test.py`, `main/scaffolding_gauge_test.py`,
`main/prober/session_test.py`); this file pins the arithmetic and the ABSENT-is-UNKNOWN rule that
all three depend on.
"""

import pytest

from agents.training.trace_selection import (
    SELECTION_SCHEMA, UNKNOWN_LABEL, build_selection, capture_rates, describe_selection,
    forensic_selection_rule, manifest_win_rates, read_selection, selection_entry, selection_rule,
)


def _entry(played, won, written, t_won):
    return selection_entry(battles_played=played, battles_won=won,
                           traces_written=written, traces_won=t_won)


# ─────────────────────────────────────────────────────────── the derived capture rates

def test_the_rates_are_traces_per_battle_PLAYED_split_by_outcome():
    """The whole point: 5 of 90 wins traced and 10 of 10 losses traced is a LOSS-ENRICHED sample,
    and the two rates are what say so. A single pooled 'capture rate' would read 0.15 and hide it."""
    e = _entry(100, 90, 15, 5)
    assert e["capture_rate_win"] == pytest.approx(5 / 90)
    assert e["capture_rate_loss"] == pytest.approx(10 / 10)
    assert e["capture_rate_loss"] > e["capture_rate_win"] * 15


def test_a_zero_denominator_reads_None_and_never_zero():
    """An opponent that lost nothing has an UNDEFINED loss-capture rate. A 0.0 there would claim
    'we captured none of its losses' — a different, false statement, and one a reweighting would
    happily divide by."""
    swept = _entry(10, 10, 5, 5)
    assert swept["capture_rate_loss"] is None
    assert swept["capture_rate_win"] == pytest.approx(0.5)
    lost_every = _entry(10, 0, 5, 0)
    assert lost_every["capture_rate_win"] is None
    assert lost_every["capture_rate_loss"] == pytest.approx(0.5)
    nothing = _entry(0, 0, 0, 0)
    assert nothing["capture_rate_win"] is None and nothing["capture_rate_loss"] is None


@pytest.mark.parametrize("played,won,written,t_won", [
    (100, 90, 15, 5), (10, 10, 10, 10), (10, 0, 3, 0), (1, 1, 1, 1), (250, 3, 13, 3),
])
def test_the_sums_reconcile_and_the_rates_are_in_the_unit_interval(played, won, written, t_won):
    e = _entry(played, won, written, t_won)
    assert e["traces_won"] <= e["battles_won"]
    assert e["traces_written"] <= e["battles_played"]
    assert e["battles_won"] <= e["battles_played"]
    for k in ("capture_rate_win", "capture_rate_loss"):
        if e[k] is not None:
            assert 0.0 <= e[k] <= 1.0


def test_inconsistent_counts_are_CLAMPED_rather_than_trusted():
    """A partial cycle can report a shard's traces against battle counts from elsewhere. A capture
    rate above 1 is an arithmetic impossibility that turns into a NEGATIVE importance weight
    downstream, so the counts are clamped into consistency here, at the one place they are read."""
    e = _entry(10, 4, 99, 77)          # more traces than battles, more trace-wins than wins
    assert e["traces_written"] == 10 and e["traces_won"] == 4
    assert e["capture_rate_win"] == pytest.approx(1.0)
    assert e["capture_rate_loss"] == pytest.approx(1.0)
    assert _entry(10, 99, 3, 1)["battles_won"] == 10          # won > played
    assert _entry(-5, -5, -5, -5)["battles_played"] == 0      # negatives floor at 0


def test_traces_lost_can_never_exceed_battles_lost():
    """The loss rate's numerator is derived (written − won), so an over-reported `traces_written`
    must not push it past the losses actually played."""
    e = _entry(10, 8, 10, 0)   # 10 traces, 0 of them wins => 10 'losses' against only 2 played
    assert e["capture_rate_loss"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────── the block and its reader

def test_build_and_read_round_trip_through_a_manifest_shaped_dict():
    block = build_selection(
        {"heuristic": dict(battles_played=100, battles_won=90, traces_written=15, traces_won=5)},
        win_quota=5, loss_quota=10)
    manifest = {"selection": block, "selection_rule": forensic_selection_rule(5, 10)}
    assert read_selection(manifest) is not None
    assert capture_rates(manifest)["heuristic"]["battles_played"] == 100
    assert manifest_win_rates(manifest) == {"heuristic": pytest.approx(0.9)}
    assert "quota" in selection_rule(manifest).lower()


def test_the_rule_states_the_QUOTA_NUMBERS_not_just_its_name():
    """A reader of an old trace tree has the recorder's constants nowhere to hand."""
    rule = forensic_selection_rule(win_quota=5, loss_quota=10)
    assert "5" in rule and "10" in rule
    assert "LOSS-ENRICHED" in rule
    assert forensic_selection_rule(1, 2) != rule       # it tracks the quotas it is given


@pytest.mark.parametrize("manifest", [
    None, {}, {"step": 1}, {"selection": None}, {"selection": {}}, {"selection": []},
    {"selection": {"schema": SELECTION_SCHEMA, "opponents": {}}},
    {"selection": {"schema": SELECTION_SCHEMA + 99, "opponents": {"x": {}}}},
    "not a dict",
])
def test_ABSENT_or_UNREADABLE_reads_UNKNOWN_and_never_uniform(manifest):
    """🚨 THE RULE THIS MODULE EXISTS FOR. Every way the record can be missing — a legacy tree, a
    cycle that crashed before collecting (the block is written as null at launch), a schema this
    build does not know — means the same thing: the selection is UNKNOWN. Silently treating it as
    an unbiased sample is the defect, and it would be a worse one for being invisible."""
    assert read_selection(manifest) is None
    assert capture_rates(manifest) is None
    assert manifest_win_rates(manifest) == {}
    assert describe_selection(manifest) == UNKNOWN_LABEL
    assert "UNKNOWN" in describe_selection(manifest)


def test_manifest_win_rates_omits_an_opponent_with_no_battles_played():
    """An undefined rate is omitted, not reported as 0.0 — a 0.0 is a claim about a lost cycle."""
    block = build_selection(
        {"played": dict(battles_played=4, battles_won=3, traces_written=2, traces_won=1),
         "never": dict(battles_played=0, battles_won=0, traces_written=0, traces_won=0)},
        win_quota=5, loss_quota=10)
    rates = manifest_win_rates({"selection": block})
    assert rates == {"played": pytest.approx(0.75)}


def test_the_known_label_reports_the_two_rates_and_names_the_skew():
    block = build_selection(
        {"h": dict(battles_played=100, battles_won=90, traces_written=15, traces_won=5)},
        win_quota=5, loss_quota=10)
    label = describe_selection({"selection": block})
    assert "SELECTION RECORDED" in label and "UNKNOWN" not in label
    assert "0.056" in label and "1.000" in label       # per WIN vs per LOSS


def test_a_label_survives_an_opponent_whose_rates_are_both_undefined():
    """`n/a`, not a crash and not a fabricated mean."""
    block = build_selection({"x": dict(battles_played=0, battles_won=0,
                                       traces_written=0, traces_won=0)},
                            win_quota=5, loss_quota=10)
    assert "n/a" in describe_selection({"selection": block})
