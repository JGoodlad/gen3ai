"""
Integration test: a FAILED Protect is captured *and* reaches the network.

The pipeline for a move outcome is:

    protocol (|-nothing| on a repeat Protect)
        -> TurnView fold  (SideTurn.failed -> outcome == "fail")
        -> TurnDelta.our_move_outcome == "fail"
        -> TurnDeltaEncoder.encode  (3-dim one-hot [hit, miss, fail] -> [0, 0, 1])
        -> Gen3ObservationEncoder turn-history block (most-recent slot)
        -> Gen3FeaturesExtractor  (pass-through scalar, NOT an embedded id)

`move_outcome_fuzz_test.py` already validates the protocol -> ... -> encoded-vector
half of that chain end-to-end on real battles (its EDGE/VARIANCE teams carry
Protect/Substitute and it asserts `fail_seen > 0` with the fail one-hot at
OFFSET_OUR_MOVE_OUTCOME). What it does NOT exercise is the *last* hop: that the
encoded fail dims actually flow through `Gen3FeaturesExtractor` and move the
policy/value output, rather than being silently dropped by `embed_delta_slot`.

This test closes that gap. It is deterministic and server-free — it drives the
real `TurnDeltaEncoder` and the real `Gen3FeaturesExtractor` (live layout +
mappings), so it integrates the encoder and the network across the seam that
matters here. The capture side is asserted at the encoded-vector level; the
network side is asserted by a differential forward pass (fail vs hit vs absent).

Run (collected by default pytest — no Node bridge, no live server):
    export PYTHONPATH=$PYTHONPATH:src
    python -m pytest src/agents/model/failed_protect_feature_test.py -q
"""
import dataclasses

import numpy as np
import gymnasium as gym
import torch

from agents.model.features_extractor import Gen3FeaturesExtractor, N_HISTORY_TURNS
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.turn_delta_encoder import (
    TurnDeltaEncoder,
    TURN_DELTA_DIM,
    OUTCOME_DIM,
    OFFSET_OUR_MOVE_OUTCOME,
    OFFSET_OPP_MOVE_OUTCOME,
    OFFSET_OUR_MOVE_BLOCK,
    TURN_DELTA_EMBEDDED_IDS,
    TURN_DELTA_SCALAR_OFFSETS,
    _OUTCOME_TO_IDX,
)
from agents.training.turn_delta import TurnDelta


# ---------------------------------------------------------------------------
# Shared fixtures (built once — the extractor init does a dummy forward pass)
# ---------------------------------------------------------------------------

_MODEL = None
_LAYOUT = None
_TD_ENC = None


def _model_and_layout():
    """The real feature extractor + live observation layout, built lazily once."""
    global _MODEL, _LAYOUT
    if _MODEL is None:
        mappings = load_mappings()
        encoder = Gen3ObservationEncoder(mappings)
        _LAYOUT = encoder.get_layout()
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_LAYOUT["total_dim"],), dtype=np.float32
        )
        _MODEL = Gen3FeaturesExtractor(obs_space, layout=_LAYOUT, mappings=mappings)
        _MODEL.eval()
    return _MODEL, _LAYOUT


def _td_encoder():
    global _TD_ENC
    if _TD_ENC is None:
        m = load_mappings()
        _TD_ENC = TurnDeltaEncoder(m.get("moves", {}), m.get("species", {}))
    return _TD_ENC


def _protect_delta(outcome):
    """A TurnDelta for 'we pressed Protect', with the given resolved outcome.

    `outcome` is "fail" (Protect on repeat did nothing), "hit" (Protect went up),
    or None (move did not resolve). Everything else is held identical so the only
    thing distinguishing two of these deltas is the outcome one-hot."""
    return dataclasses.replace(
        TurnDelta.empty(),
        our_move_id="protect",
        our_attempted_move_id="protect",
        our_prev_active="blastoise",
        our_move_outcome=outcome,
    )


def _history_start(layout):
    # Mirrors Gen3Env.embed_battle: [base encoder | 11 prev_mask | N history slots].
    return layout["base_dim"] + 11


def _obs_with_recent_delta(layout, delta):
    """A full zero observation with `delta` encoded into the MOST-RECENT history slot.

    The base/team/global regions are left zero — irrelevant here; we isolate the
    history slot exactly as features_extractor_test.py does. Returns [1, total_dim]."""
    obs = torch.zeros(1, layout["total_dim"], dtype=torch.float32)
    slot = _td_encoder().encode(delta)
    assert slot.shape == (TURN_DELTA_DIM,)
    start = _history_start(layout) + (N_HISTORY_TURNS - 1) * TURN_DELTA_DIM
    obs[0, start:start + TURN_DELTA_DIM] = torch.from_numpy(slot)
    return obs


# ---------------------------------------------------------------------------
# 1. Capture: a failed Protect encodes the "fail" one-hot
# ---------------------------------------------------------------------------

def test_failed_protect_encodes_fail_onehot():
    """delta.our_move_outcome == 'fail' -> [0, 0, 1] at OFFSET_OUR_MOVE_OUTCOME."""
    enc = _td_encoder().encode(_protect_delta("fail"))
    block = enc[OFFSET_OUR_MOVE_OUTCOME:OFFSET_OUR_MOVE_OUTCOME + OUTCOME_DIM]
    np.testing.assert_array_equal(block, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert int(np.argmax(block)) == _OUTCOME_TO_IDX["fail"]

    # Sanity: a Protect that went up encodes "hit", not "fail" — same move,
    # different outcome dim, so the two are genuinely distinguishable on-vector.
    hit = _td_encoder().encode(_protect_delta("hit"))
    hit_block = hit[OFFSET_OUR_MOVE_OUTCOME:OFFSET_OUR_MOVE_OUTCOME + OUTCOME_DIM]
    np.testing.assert_array_equal(hit_block, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    # And the move id itself is carried (Protect, not the unknown sentinel), so
    # the network can attribute the failure to Protect specifically.
    assert enc[OFFSET_OUR_MOVE_BLOCK] != 0.0


# ---------------------------------------------------------------------------
# 2. Structural: the outcome dims are pass-through scalars (not dropped)
# ---------------------------------------------------------------------------

def test_outcome_dims_reach_extractor_as_scalars():
    """The outcome one-hot positions are pass-through scalars in the extractor.

    `embed_delta_slot` routes only the positions in TURN_DELTA_EMBEDDED_IDS to
    embedding tables; every other position is gathered into the scalar index and
    fed to the history projection. If an outcome dim were (mis)declared as an
    embedded id it would be quantized to an embedding lookup and the continuous
    one-hot would never reach the network. Assert it is a scalar, both via the
    public manifest and via the live extractor's gather index."""
    embedded = frozenset(pos for pos, _ in TURN_DELTA_EMBEDDED_IDS)
    outcome_positions = list(range(OFFSET_OUR_MOVE_OUTCOME, OFFSET_OUR_MOVE_OUTCOME + OUTCOME_DIM)) + \
        list(range(OFFSET_OPP_MOVE_OUTCOME, OFFSET_OPP_MOVE_OUTCOME + OUTCOME_DIM))
    for pos in outcome_positions:
        assert pos not in embedded, f"outcome dim {pos} wrongly declared as an embedded id"
        assert pos in TURN_DELTA_SCALAR_OFFSETS, f"outcome dim {pos} missing from scalar offsets"

    # The extractor's actual gather index must contain every outcome position.
    # `_td_scalar_idx` is the buffer on the shared Embeddings module that
    # `embed_delta_slot` uses to pick the pass-through scalars.
    model, _ = _model_and_layout()
    scalar_idx = set(model.embeddings._td_scalar_idx.tolist())
    for pos in outcome_positions:
        assert pos in scalar_idx, f"outcome dim {pos} not in extractor scalar gather"


# ---------------------------------------------------------------------------
# 3. Network: a failed Protect moves the policy/value output
# ---------------------------------------------------------------------------

def test_failed_protect_changes_network_output():
    """A failed-Protect history slot produces different features than empty history."""
    model, layout = _model_and_layout()
    obs_empty = torch.zeros(1, layout["total_dim"], dtype=torch.float32)
    obs_fail = _obs_with_recent_delta(layout, _protect_delta("fail"))

    with torch.no_grad():
        pi0, vf0 = model({"observation": obs_empty})
        pi1, vf1 = model({"observation": obs_fail})

    assert not torch.allclose(pi0, pi1), "failed Protect did not reach the policy head"
    assert not torch.allclose(vf0, vf1), "failed Protect did not reach the value head"


def test_failed_vs_successful_protect_distinguishable_by_network():
    """The strongest claim: the FAIL signal *specifically* reaches the network.

    Two observations identical except the most-recent history slot's outcome
    one-hot (fail vs hit) — same Protect, same everything else. If the network
    output differs, the failed-vs-succeeded distinction is genuinely visible to
    the model, not collapsed away upstream."""
    model, layout = _model_and_layout()
    obs_fail = _obs_with_recent_delta(layout, _protect_delta("fail"))
    obs_hit = _obs_with_recent_delta(layout, _protect_delta("hit"))

    # Precondition: the two obs differ ONLY within the outcome one-hot block.
    # (fail = [0,0,1] vs hit = [1,0,0] differ at the hit and fail dims; the
    # middle "miss" dim is zero in both — so the diff is a non-empty subset of
    # the 3-dim block, not the whole block.)
    diff = (obs_fail - obs_hit).abs().squeeze(0).numpy()
    nonzero = set(np.where(diff > 0)[0].tolist())
    slot_start = _history_start(layout) + (N_HISTORY_TURNS - 1) * TURN_DELTA_DIM
    outcome_block = set(
        slot_start + OFFSET_OUR_MOVE_OUTCOME + i for i in range(OUTCOME_DIM)
    )
    assert nonzero and nonzero <= outcome_block, (
        "fail vs hit obs differ outside the outcome one-hot block"
    )

    with torch.no_grad():
        pi_fail, vf_fail = model({"observation": obs_fail})
        pi_hit, vf_hit = model({"observation": obs_hit})

    assert not torch.allclose(pi_fail, pi_hit), \
        "fail vs successful Protect is invisible to the policy head"
    assert not torch.allclose(vf_fail, vf_hit), \
        "fail vs successful Protect is invisible to the value head"
