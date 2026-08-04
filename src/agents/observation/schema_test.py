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


def test_validation_catches_gaps_and_overlaps():
    with pytest.raises(ValueError, match="gap or overlap"):
        ObsSchema(total_dim=10, blocks=[Block("a", 0, 4), Block("b", 5, 5)]).validate()
    with pytest.raises(ValueError, match="tile 9 dims"):
        ObsSchema(total_dim=10, blocks=[Block("a", 0, 4), Block("b", 4, 5)]).validate()
    with pytest.raises(ValueError, match="children tile"):
        ObsSchema(total_dim=4, blocks=[
            Block("a", 0, 4, children=[Block("c", 0, 3)]),
        ]).validate()
