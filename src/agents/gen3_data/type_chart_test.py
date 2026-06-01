from agents.gen3_data import type_chart


def test_super_effective_and_immune_cells():
    # Keyed {DEFENDING: {ATTACKING: multiplier}}, poke-env enum-name keys.
    assert type_chart.multiplier("FIRE", "WATER") == 2      # water beats fire
    assert type_chart.multiplier("WATER", "FIRE") == 0.5    # fire resisted by water
    assert type_chart.multiplier("GROUND", "ELECTRIC") == 0  # ground immune to electric
    assert type_chart.multiplier("NORMAL", "NORMAL") == 1


def test_chart_covers_all_18_types():
    chart = type_chart.chart()
    assert len(chart) == 18
    # Square: every defending row carries an entry for every attacking type.
    types = set(chart)
    for row in chart.values():
        assert types.issubset(set(row))
