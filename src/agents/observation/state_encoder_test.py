import numpy as np
import torch
import gymnasium as gym
from .state_encoder import Gen3ObservationEncoder, load_mappings

# Computed from live constants so they track architecture changes (no magic numbers).
from agents.observation.constants import OFFSET_EVENT_WINDOW, EVENT_WINDOW_DIM, OBS_FACTS_DIM

# gen3_pair_history_v1: base ends after the H-A2 pair block.
EXPECTED_BASE_DIM = OFFSET_EVENT_WINDOW + EVENT_WINDOW_DIM + OBS_FACTS_DIM  # gen3_obs_facts_v1
# gen3_frame_deletion_v1: the prev-mask + lag-frame tail is DELETED, so the full obs
# IS the base. Kept as a separate name so a future appended block has an obvious
# home, and so its absence reads as a visible equality rather than a silent one.
EXPECTED_OBS_DIM = EXPECTED_BASE_DIM


def test_encoder_dimension():
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    assert encoder.base_dimension == EXPECTED_BASE_DIM
    assert encoder.dimension == EXPECTED_OBS_DIM


def test_encoder_and_features_extractor_are_compatible():
    """
    Verifies the encoder dimension and features extractor architecture agree.
    Catches obs-space/architecture mismatches before they surface at checkpoint load.
    """
    from agents.model.features_extractor import Gen3FeaturesExtractor

    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    layout = encoder.get_layout()

    obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(encoder.dimension,), dtype=np.float32),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(11,), dtype=np.int8),
    })

    model = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings)
    model.eval()

    dummy_obs = {
        "observation": torch.zeros(1, encoder.dimension),
        "action_mask": torch.ones(1, 11, dtype=torch.int8),
    }
    with torch.no_grad():
        pi_features, vf_features = model(dummy_obs)

    # Dual-head extractor: policy and value features, both PROJECTION_DIM-wide.
    assert pi_features.shape == (1, model.features_dim)
    assert vf_features.shape == (1, model.vf_features_dim)   # value_pooled (no value projection: F1)


def test_layout_parts_are_their_own_spans_and_tile_the_row():
    """Every `parts[...]` span is its OWN [start, end) (reactive's `end` once read `total_dim`), and the
    parts + event window + obs-facts tile [0, total_dim) exactly. Fails on a revert of the reactive fix."""
    from agents.observation.constants import OFFSET_REACTIVE, REACTIVE_DIM, OFFSET_PAIR_HISTORY
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    parts = layout["parts"]
    assert parts["reactive"]["end"] == parts["reactive"]["start"] + REACTIVE_DIM == OFFSET_REACTIVE + REACTIVE_DIM
    assert parts["reactive"]["end"] == OFFSET_PAIR_HISTORY == parts["pair_history"]["start"]
    assert parts["reactive"]["end"] != layout["total_dim"]
    names = list(parts)
    for a, b in zip(names, names[1:]):
        assert parts[a]["end"] == parts[b]["start"], (a, b)
    assert parts[names[0]]["start"] == 0
    assert parts[names[-1]]["end"] == layout["event_window_offset"]
    assert layout["event_window_offset"] + layout["event_window_dim"] == layout["obs_facts_offset"]
    assert layout["obs_facts_offset"] + layout["obs_facts_dim"] == layout["total_dim"]


def test_validate_layout_raises_on_every_kind_of_wrong_span():
    import copy
    import pytest
    from .state_encoder import validate_layout
    good = Gen3ObservationEncoder(load_mappings()).get_layout()
    validate_layout(good)

    def mutated(fn):
        lay = copy.deepcopy(good)
        fn(lay)
        return lay

    bad = {
        "reactive end = total (the original bug)": lambda l: l["parts"]["reactive"].__setitem__("end", l["total_dim"]),
        "gap": lambda l: l["parts"]["global"].__setitem__("start", l["parts"]["global"]["start"] + 1),
        "overlap": lambda l: l["parts"]["opp_team"].__setitem__("start", l["parts"]["opp_team"]["start"] - 1),
        "width != dim": lambda l: l["parts"]["global"].__setitem__("end", l["parts"]["global"]["end"] - 1),
        "width != reshape": lambda l: l["parts"]["pair_history"].__setitem__("end", l["parts"]["pair_history"]["end"] + 1),
        "short of total": lambda l: l.__setitem__("total_dim", l["total_dim"] + 1),
        "event window moved": lambda l: l.__setitem__("event_window_offset", l["event_window_offset"] + 1),
    }
    for label, fn in bad.items():
        with pytest.raises(ValueError):
            validate_layout(mutated(fn))
            pytest.fail(f"validate_layout accepted: {label}")
