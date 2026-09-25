"""The bridge transport's `__OBS__` frame (gen3_core_obs_source_v1): stashed raw per side before
the chunk it precedes, refused when nobody asked for it, and never decoded by the transport."""
import pytest

from utils.bridge.bridge_session import BridgeSession


def _session(spec):
    s = BridgeSession.__new__(BridgeSession)
    s._closing, s._tag, s._core_obs_spec, s.core_obs = False, "battle-gen3ou-9", spec, {}
    return s


def test_an_obs_frame_is_stashed_raw_under_the_current_battle_tag():
    s = _session({"sides": ["p1"], "decision_tense": False, "switch_freeze": False})
    s._dispatch('__OBS__ p1 {"n": 0}')
    assert s.core_obs == {"p1": ("battle-gen3ou-9", '{"n": 0}')}
    s._dispatch('__OBS__ p1 {"n": 1}')
    assert s.core_obs["p1"][1] == '{"n": 1}'          # the latest wins


def test_an_obs_frame_nobody_asked_for_is_a_protocol_violation():
    with pytest.raises(RuntimeError, match="nobody asked"):
        _session(None)._dispatch('__OBS__ p1 {"n": 0}')


def test_core_obs_needs_the_rust_bridge():
    from utils.bridge.bridge_session import attach_bridge_transport

    with pytest.raises(ValueError, match="rust"):
        attach_bridge_transport(object(), battle_format="gen3ou", impl="node", core_obs=True)
