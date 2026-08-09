"""Tests for the declarative obs schema (Stage-3 groundwork).

The schema is a validated VIEW over the live layout: these tests prove it tiles the real
encoder's vector exactly and that its self-validation actually catches gaps/overlaps."""
import pytest

from agents.observation.schema import Block, ObsSchema, build_schema
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_enc = Gen3ObservationEncoder(load_mappings())
_layout = _enc.get_layout()


def test_schema_tiles_the_live_encoder_exactly():
    sch = build_schema(_layout)                      # validate() runs inside
    assert sch.total_dim == _enc.dimension == _layout["total_dim"]
    assert sch.block("turn_history").offset == _layout["turn_history_offset"]
    assert sch.block("prev_action_mask").offset == _layout["base_dim"]
    # The reactive children must tile the whole reactive block (the sub-layout is complete).
    r = sch.block("reactive")
    assert sum(c.dim for c in r.children) == r.dim


def test_generated_slices_and_space_match_the_live_encoder():
    """The generator half: `slices()` must agree with every offset the live layout publishes
    (the drift guard consumers rely on), and `gym_space()` must reproduce the env's vector
    space exactly — one source for the obs dim."""
    import numpy as np

    sch = build_schema(_layout)
    sl = sch.slices()
    assert sl["turn_history"] == slice(_layout["turn_history_offset"], _enc.dimension)
    assert sl["prev_action_mask"].start == _layout["base_dim"]
    assert sl["reactive.active_req_moves"].start == (
        _layout["parts"]["reactive"]["start"] + _layout["reactive_layout"]["active_req_moves"]["offset"])
    # Every slice must sit inside the vector and child slices inside their parent.
    for name, s in sl.items():
        assert 0 <= s.start < s.stop <= sch.total_dim, name
    space = sch.gym_space()
    assert space.shape == (_enc.dimension,)
    assert space.dtype == np.float32 and space.low[0] == -np.inf


def test_validation_catches_gaps_and_overlaps():
    with pytest.raises(ValueError, match="gap or overlap"):
        ObsSchema(total_dim=10, blocks=[Block("a", 0, 4), Block("b", 5, 5)]).validate()
    with pytest.raises(ValueError, match="tile 9 dims"):
        ObsSchema(total_dim=10, blocks=[Block("a", 0, 4), Block("b", 4, 5)]).validate()
    with pytest.raises(ValueError, match="children tile"):
        ObsSchema(total_dim=4, blocks=[
            Block("a", 0, 4, children=[Block("c", 0, 3)]),
        ]).validate()
