"""Unit tests for the multi-month aggregator.

Network paths (fetch_month, sync_stats) are exercised by the integration
sanity run in the CLI; this file pins the pure-Python logic.
"""
from sync import (
    iter_months_backward,
    merge_chaos,
    _merge_species,
    _parse_end_month,
    _previous_month,
)


# ---------------------------------------------------------------------------
# Month iteration
# ---------------------------------------------------------------------------

def test_previous_month_wraps_across_year_boundary():
    assert _previous_month(2026, 1) == (2025, 12)


def test_previous_month_within_year():
    assert _previous_month(2026, 5) == (2026, 4)


def test_iter_months_backward_count():
    months = list(iter_months_backward(2026, 4, 12))
    assert len(months) == 12
    assert months[0] == (2026, 4)
    assert months[-1] == (2025, 5)


def test_iter_months_backward_no_duplicates():
    months = list(iter_months_backward(2026, 4, 24))
    assert len(set(months)) == 24


def test_parse_end_month_explicit():
    assert _parse_end_month("2026-04") == (2026, 4)


def test_parse_end_month_default_is_previous_month():
    """Default with no argument should be some valid previous month, not raise."""
    y, m = _parse_end_month(None)
    assert 1 <= m <= 12
    assert y >= 2025


# ---------------------------------------------------------------------------
# Species merge
# ---------------------------------------------------------------------------

def _month(raw, moves, items=None, abilities=None, viability=None, usage=None):
    """Build a minimal species record for tests."""
    rec = {"Raw count": raw, "Moves": dict(moves)}
    if items is not None:
        rec["Items"] = dict(items)
    if abilities is not None:
        rec["Abilities"] = dict(abilities)
    if viability is not None:
        rec["Viability Ceiling"] = list(viability)
    if usage is not None:
        rec["usage"] = usage
    return rec


def test_merge_species_sums_raw_count_and_moves():
    dst = {}
    _merge_species(dst, _month(100, {"surf": 50.0, "icebeam": 30.0}), is_first_month_for_species=True)
    _merge_species(dst, _month(200, {"surf": 40.0, "thunderbolt": 10.0}), is_first_month_for_species=False)

    assert dst["Raw count"] == 300
    assert dst["Moves"]["surf"] == 90.0       # summed across months
    assert dst["Moves"]["icebeam"] == 30.0    # only in month 1
    assert dst["Moves"]["thunderbolt"] == 10.0  # only in month 2


def test_merge_species_latest_wins_for_viability_and_usage():
    """is_first_month_for_species=True should pin Viability/usage from the first call only."""
    dst = {}
    _merge_species(
        dst, _month(100, {"surf": 1.0}, viability=[88, 84, 80, 76], usage=0.20),
        is_first_month_for_species=True,
    )
    _merge_species(
        dst, _month(50, {"surf": 1.0}, viability=[80, 76, 72, 68], usage=0.10),
        is_first_month_for_species=False,  # NOT first — should NOT overwrite
    )

    assert dst["Viability Ceiling"] == [88, 84, 80, 76]
    assert dst["usage"] == 0.20


def test_merge_species_handles_missing_fields():
    """Months that omit a summable field should not crash."""
    dst = {}
    _merge_species(dst, {"Raw count": 100, "Moves": {"surf": 1.0}}, is_first_month_for_species=True)
    # Second month has no Moves key at all
    _merge_species(dst, {"Raw count": 50}, is_first_month_for_species=False)
    assert dst["Raw count"] == 150
    assert dst["Moves"] == {"surf": 1.0}


# ---------------------------------------------------------------------------
# Full merge_chaos
# ---------------------------------------------------------------------------

def _chaos(year, month, battles, species_map):
    return ({
        "info": {
            "metagame": "gen3ou",
            "cutoff": 1500,
            "cutoff deviation": 0,
            "team type": None,
            "number of battles": battles,
        },
        "data": species_map,
    })


def test_merge_chaos_sums_battles_and_lists_window_months():
    months = [
        ((2026, 4), _chaos(2026, 4, 100, {
            "Tyranitar": _month(50, {"crunch": 25.0}),
        })),
        ((2026, 3), _chaos(2026, 3, 200, {
            "Tyranitar": _month(100, {"crunch": 60.0, "rockslide": 40.0}),
        })),
        ((2026, 2), _chaos(2026, 2, 300, {
            "Tyranitar": _month(150, {"crunch": 90.0}),
            "Salamence": _month(80, {"dragonclaw": 70.0}),
        })),
    ]
    merged = merge_chaos(months, "gen3ou-1500")

    assert merged["info"]["number of battles"] == 600
    assert merged["info"]["aggregated"] is True
    assert merged["info"]["window_months"] == ["2026-04", "2026-03", "2026-02"]
    assert merged["info"]["format"] == "gen3ou-1500"

    tyr = merged["data"]["Tyranitar"]
    assert tyr["Raw count"] == 300
    assert tyr["Moves"]["crunch"] == 175.0
    assert tyr["Moves"]["rockslide"] == 40.0

    # Salamence only appears in the oldest month, but should still land in the
    # merged output with the correct counts.
    sal = merged["data"]["Salamence"]
    assert sal["Raw count"] == 80
    assert sal["Moves"]["dragonclaw"] == 70.0


def test_merge_chaos_hidden_power_use_case():
    """End-to-end shape for the Hidden Power priors path: typed HP move counts
    should sum across months as the priors script expects."""
    months = [
        ((2026, 4), _chaos(2026, 4, 100, {
            "Jolteon": _month(50, {"hiddenpowerice": 30.0, "hiddenpowergrass": 5.0, "thunderbolt": 50.0}),
        })),
        ((2026, 3), _chaos(2026, 3, 200, {
            "Jolteon": _month(100, {"hiddenpowerice": 80.0, "hiddenpowergrass": 10.0}),
        })),
    ]
    merged = merge_chaos(months, "gen3ou-1500")

    jol = merged["data"]["Jolteon"]["Moves"]
    assert jol["hiddenpowerice"] == 110.0
    assert jol["hiddenpowergrass"] == 15.0
    assert jol["thunderbolt"] == 50.0
