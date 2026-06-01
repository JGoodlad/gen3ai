"""
Shared harness for the feature-coverage suite.

Every test in this folder answers the same two-part question for one battle
edge case:

  1. CAPTURE  — is the edge case encoded into the observation vector at the
                expected offset (right one-hot bit / scalar / embedded id)?
  2. NETWORK  — does that encoded signal actually flow through the *real*
                `Gen3FeaturesExtractor` and move its policy/value output,
                rather than being silently dropped (e.g. by `embed_delta_slot`
                or a key-padding mask)?

The suite is deterministic and server-free: it drives the real
`TurnDeltaEncoder` and the real `Gen3FeaturesExtractor` (live layout + mappings)
on hand-built `TurnDelta`s / observation regions. Real-battle *capture* of these
same signals is covered by the bridge-backed fuzz tests under
`training/poke_env_gaps/`; this suite closes the last hop those don't exercise —
that the captured dims reach the network.

Helpers
-------
`feature_model()`      -> (model, layout, mappings), built once (extractor init
                          runs a dummy forward, so reuse it).
`td_encoder()`         -> the shared TurnDeltaEncoder.
`make_delta(**f)`      -> a TurnDelta.empty() with fields overridden.
`anchor_delta(**f)`    -> like make_delta but with a default move set so the
                          history slot is non-empty (NOT treated as padding),
                          letting you isolate a single field against a baseline.
`encode_delta(d)`      -> [TURN_DELTA_DIM] encoded slot vector.
`obs_zero(layout)`     -> [1, total_dim] zero observation.
`obs_with_delta(...)`  -> zero obs with `delta` in a history slot.
`history_slot_offset`  -> absolute obs offset of a history slot.
`set_region(...)`      -> write raw values into a static obs region by offset.
`forward(model, obs)`  -> (pi, vf) under no_grad.
`policy_value_changed` -> (pi_changed, vf_changed) bools between two obs.
`assert_reaches_network(model, base, variant, msg)` -> asserts BOTH heads move.
`read_block` / `onehot_idx` / `multihot_idxs` -> vector inspection helpers.
"""
import dataclasses
from typing import Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
import torch

from agents.model.features_extractor import Gen3FeaturesExtractor, N_HISTORY_TURNS
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.turn_delta_encoder import TurnDeltaEncoder, TURN_DELTA_DIM
from agents.training.turn_delta import TurnDelta

_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Model / encoder (built once — extractor __init__ runs a dummy forward pass)
# ---------------------------------------------------------------------------

def feature_model():
    """Return (model, layout, mappings). Cached; safe to call per-test."""
    if "model" not in _CACHE:
        mappings = load_mappings()
        encoder = Gen3ObservationEncoder(mappings)
        layout = encoder.get_layout()
        space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(layout["total_dim"],), dtype=np.float32
        )
        model = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings)
        model.eval()
        _CACHE["model"] = (model, layout, mappings)
    return _CACHE["model"]


def td_encoder() -> TurnDeltaEncoder:
    if "td" not in _CACHE:
        m = load_mappings()
        _CACHE["td"] = TurnDeltaEncoder(m.get("moves", {}), m.get("species", {}))
    return _CACHE["td"]


# ---------------------------------------------------------------------------
# TurnDelta construction
# ---------------------------------------------------------------------------

def make_delta(**fields) -> TurnDelta:
    """A `TurnDelta.empty()` with the given fields overridden."""
    return dataclasses.replace(TurnDelta.empty(), **fields)


def anchor_delta(**fields) -> TurnDelta:
    """A non-empty TurnDelta whose history slot is guaranteed present (unmasked).

    Defaults `our_move_id="tackle"` + a real `our_prev_active` so the encoded
    slot is non-zero even before your edge field is added — so a baseline and a
    variant that differ in ONE field are both 'present' history, and the output
    delta isolates that field (not slot presence). Override any default freely."""
    base = dict(our_move_id="tackle", our_prev_active="snorlax")
    base.update(fields)
    return make_delta(**base)


def encode_delta(delta: TurnDelta) -> np.ndarray:
    """Encode a TurnDelta to its [TURN_DELTA_DIM] slot vector (float32)."""
    vec = td_encoder().encode(delta)
    assert vec.shape == (TURN_DELTA_DIM,), vec.shape
    return vec


# ---------------------------------------------------------------------------
# Observation assembly
# ---------------------------------------------------------------------------

def obs_zero(layout) -> torch.Tensor:
    return torch.zeros(1, layout["total_dim"], dtype=torch.float32)


def history_slot_offset(layout, slot_from_recent: int = 0) -> int:
    """Absolute obs offset of a turn-history slot.

    `slot_from_recent=0` is the MOST-RECENT slot (the one whose output flows
    into the projection); higher values walk back in time. Mirrors
    `Gen3Env.embed_battle`: [base encoder | 11 prev_mask | N history slots]."""
    assert 0 <= slot_from_recent < N_HISTORY_TURNS
    start = layout["base_dim"] + 11
    idx = N_HISTORY_TURNS - 1 - slot_from_recent
    return start + idx * TURN_DELTA_DIM


def obs_with_delta(layout, delta: TurnDelta, slot_from_recent: int = 0) -> torch.Tensor:
    """Zero observation with `delta` encoded into a history slot."""
    obs = obs_zero(layout)
    off = history_slot_offset(layout, slot_from_recent)
    obs[0, off:off + TURN_DELTA_DIM] = torch.from_numpy(encode_delta(delta))
    return obs


def set_region(obs: torch.Tensor, abs_offset: int, values) -> torch.Tensor:
    """Write `values` into obs[0, abs_offset:abs_offset+len]. Returns obs."""
    t = torch.as_tensor(np.asarray(values, dtype=np.float32))
    obs[0, abs_offset:abs_offset + t.numel()] = t.reshape(-1)
    return obs


# ---------------------------------------------------------------------------
# Forward / assertions
# ---------------------------------------------------------------------------

def forward(model, obs: torch.Tensor):
    """Return (pi_features, vf_features) under no_grad."""
    with torch.no_grad():
        return model({"observation": obs})


def policy_value_changed(model, obs_a: torch.Tensor, obs_b: torch.Tensor) -> Tuple[bool, bool]:
    pa, va = forward(model, obs_a)
    pb, vb = forward(model, obs_b)
    return (not torch.allclose(pa, pb)), (not torch.allclose(va, vb))


def assert_reaches_network(model, base: torch.Tensor, variant: torch.Tensor, msg: str = ""):
    """Assert the difference between `base` and `variant` moves BOTH heads.

    The transformer body is shared and history/static signals feed both the
    policy and value pools, so a live signal should move both. If a specific
    edge case legitimately only touches one head, use `policy_value_changed`
    and assert the relevant one instead."""
    pi_changed, vf_changed = policy_value_changed(model, base, variant)
    assert pi_changed, f"policy head did not move — signal not reaching network: {msg}"
    assert vf_changed, f"value head did not move — signal not reaching network: {msg}"


def assert_delta_reaches_network(model, layout, base_delta: TurnDelta,
                                 variant_delta: TurnDelta, msg: str = ""):
    """Convenience: place two deltas in the most-recent slot and assert the
    network output differs. Use anchor_delta for both so only the edge field
    differs and the comparison isolates that field."""
    base = obs_with_delta(layout, base_delta)
    variant = obs_with_delta(layout, variant_delta)
    assert_reaches_network(model, base, variant, msg)


# ---------------------------------------------------------------------------
# Vector inspection
# ---------------------------------------------------------------------------

def read_block(vec, offset: int, dim: int) -> np.ndarray:
    return np.asarray(vec)[offset:offset + dim]


def onehot_idx(vec, offset: int, dim: int) -> Optional[int]:
    """Index of the set bit in vec[offset:offset+dim], or None if all < 0.5."""
    block = read_block(vec, offset, dim)
    if block.max() <= 0.5:
        return None
    return int(np.argmax(block))


def multihot_idxs(vec, offset: int, dim: int) -> Sequence[int]:
    """Indices set (>0.5) in vec[offset:offset+dim]."""
    block = read_block(vec, offset, dim)
    return [i for i in range(dim) if block[i] > 0.5]
