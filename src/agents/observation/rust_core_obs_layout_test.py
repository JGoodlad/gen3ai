"""The committed Rust observation layout equals what the Python encoder's constants render today.

``src/rust_sim/src/encoder/layout.rs`` is GENERATED (``gen3_core_obs_layout_v1``) from
``agents/observation/constants.py`` and the sub-encoders' index maps, so a dim change is a
one-place edit: this routine pin fails the day the committed file is stale, BEFORE slice O's byte
comparison could fail on it.
"""

from agents.observation.rust_core_obs_layout import OUT, render


def test_the_committed_obs_layout_is_current():
    assert OUT.read_text() == render(), (
        "src/rust_sim/src/encoder/layout.rs is STALE — run "
        "`python -m agents.observation.rust_core_obs_layout --write` and rebuild the rust core")


def test_the_layout_dim_is_the_encoders_dim():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    dim = Gen3ObservationEncoder(load_mappings()).dimension
    assert f"pub const OBS_DIM: usize = {dim};" in render()


def test_the_layout_is_not_vacuous():
    text = render()
    for needle in ("pub const POKEMON_FULL_DIM", "pub static VOLATILE_TO_SLOT", "pub static SAT_LUT",
                   "pub const EV_ITEM_TRANSITION", '"partiallytrapped"', '"light_screen"'):
        assert needle in text, needle
