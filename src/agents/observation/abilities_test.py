from .abilities import AbilitiesEncoder


_BASE_MAPPING = {
    "intimidate": {"num": 1},
    "levitate":   {"num": 2},
    "thickfat":   {"num": 3},
    "immunity":   {"num": 4},
    "wonderguard": {"num": 5},
    "voltabsorb": {"num": 6},
    "illuminate": {"num": 7},
}


# ---------------------------------------------------------------------------
# Dimension and basic shape
# ---------------------------------------------------------------------------

def test_abilities_encoder_dimension():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    assert encoder.dimension == 4   # [ability1_id, ability2_id, dominance, known]


def test_abilities_encoder_layout():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    layout = encoder.get_layout()
    assert layout["id1"]["offset"] == 0
    assert layout["id2"]["offset"] == 1
    assert layout["dominance"]["offset"] == 2
    assert layout["known"]["offset"] == 3


# ---------------------------------------------------------------------------
# Revealed path (own team always; opp once an ability fires)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Unrevealed path — Smogon priors
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
