"""Unit tests for the POINTER ACTION HEAD (`pointer_head`, v49, gen3_pointer_head_v1).

The claim: score each action FROM THE TOKEN OF THE ENTITY IT SELECTS, so two measured defects become
structurally impossible rather than defended:
  * **F2** — switch logits are read from a permutation-INVARIANT CLS pool, so a bench mon's own token
    can never reach its own switch logit.
  * **the ordering bug class** — the extractor reads a mon's moves SORTED BY ID while the action space
    uses REQUEST order, so `action_mask[6+k]` and move-token *k* can refer to different moves.
    `agents/action/ordering_integrity.py` exists only to permute one into the other.

The permutation test below is the load-bearing one: it is the single place that bug class is
dissolved, and it is deliberately given a SCRAMBLED order (an alphabetical moveset would pass even if
the permutation were a no-op).
"""
import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.action.constants import ACTION_SPACE_SIZE, MOVE_START, N_MOVE_SLOTS, STRUGGLE, SWITCH_START
from agents.model.features_extractor import (
    Gen3FeaturesExtractor, MOVE_NET_HIDDEN, _request_order_move_tokens,
)
from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()
_SIG = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)


def _make(**kw):
    space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(space, layout=_layout, mappings=_mappings,
                                 **{k: v for k, v in kw.items() if k in _SIG})


def _obs(batch=4, seed=5):
    return {"observation": torch.rand(batch, _layout["total_dim"],
                                      generator=torch.Generator().manual_seed(seed))}


# ------------------------------------------------------- the permutation (the crux)
def _ctx(active_idx, sorted_ids, req_ids, tok):
    import types
    B = len(active_idx)
    c = types.SimpleNamespace(batch_size=B, device=torch.device("cpu"),
                              our_active_idx=torch.tensor(active_idx),
                              all_move_ids=torch.zeros(B, 12, N_MOVE_SLOTS, dtype=torch.long),
                              our_active_req_move_ids=torch.zeros(B, N_MOVE_SLOTS))
    for b, a in enumerate(active_idx):
        c.all_move_ids[b, a] = torch.tensor(sorted_ids[b])
        c.our_active_req_move_ids[b] = torch.tensor(req_ids[b], dtype=torch.float32)
    return c, tok


def test_permutation_maps_request_slots_by_move_num_not_position():
    """A SCRAMBLED request order must be resolved by move-num identity. Token value == its sorted slot,
    so the output reads back the sorted slot each request slot resolved to."""
    D = 3
    tok = torch.zeros(1, 12, N_MOVE_SLOTS, D)
    for s in range(N_MOVE_SLOTS):
        tok[0, 2, s] = float(s)
    ctx, tok = _ctx([2], [[10, 20, 30, 40]], [[30., 10., 40., 20.]], tok)
    out, valid = _request_order_move_tokens(tok, ctx)
    assert [float(out[0, k, 0]) for k in range(4)] == [2.0, 0.0, 3.0, 1.0]
    assert valid[0].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_permutation_zeroes_unresolved_slots():
    """A mon with <4 moves (or a forced-Struggle decision) has request slots that resolve to nothing:
    they must be zeroed AND marked invalid, so the head contributes exactly 0 there rather than
    scoring a garbage vector into a live logit."""
    D = 2
    tok = torch.zeros(1, 12, N_MOVE_SLOTS, D)
    for s in range(N_MOVE_SLOTS):
        tok[0, 0, s] = float(10 + s)
    ctx, tok = _ctx([0], [[7, 8, 0, 0]], [[8., 7., 0., 0.]], tok)
    out, valid = _request_order_move_tokens(tok, ctx)
    assert [float(out[0, k, 0]) for k in range(4)] == [11.0, 10.0, 0.0, 0.0]
    assert valid[0].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_permutation_is_identity_on_an_already_sorted_moveset():
    """The degenerate case that would hide a broken permutation — pinned so a regression can't pass
    by only ever being tested on alphabetical movesets."""
    D = 2
    tok = torch.zeros(1, 12, N_MOVE_SLOTS, D)
    for s in range(N_MOVE_SLOTS):
        tok[0, 1, s] = float(s)
    ctx, tok = _ctx([1], [[5, 6, 7, 8]], [[5., 6., 7., 8.]], tok)
    out, _ = _request_order_move_tokens(tok, ctx)
    assert [float(out[0, k, 0]) for k in range(4)] == [0.0, 1.0, 2.0, 3.0]


# ------------------------------------------------------- structure / stash
def test_off_builds_nothing_and_leaves_the_stash_cold():
    off = _make(pointer_head=False)
    assert off.pointer_head is None
    assert off.pokemon_encoder.stash_move_tokens is False
    with torch.no_grad():
        off(_obs())
    assert off.pokemon_encoder.last_move_tokens is None
    assert off.last_pointer_delta is None


def test_on_stashes_per_move_tokens_at_the_right_shape():
    on = _make(pointer_head=True).eval()
    with torch.no_grad():
        on(_obs(batch=3))
    t = on.pokemon_encoder.last_move_tokens
    assert tuple(t.shape) == (3, 12, N_MOVE_SLOTS, MOVE_NET_HIDDEN[1])


def test_delta_shape_matches_the_action_space():
    on = _make(pointer_head=True).eval()
    with torch.no_grad():
        on(_obs())
    d = on.last_pointer_delta
    assert tuple(d.shape) == (4, ACTION_SPACE_SIZE)
    assert SWITCH_START == 0 and MOVE_START == 6 and STRUGGLE == 10   # layout the head assumes


def test_delta_is_exactly_zero_at_init():
    """Identity-at-init on the BARE extractor. The same property THROUGH a real SB3 policy — where it
    is only true because of the M1 ortho-init fix — is pinned in identity_init_test.py."""
    on = _make(pointer_head=True).eval()
    with torch.no_grad():
        on(_obs())
    assert float(on.last_pointer_delta.abs().max()) == 0.0


def test_trained_scorers_move_only_their_own_action_block():
    """Perturb one scorer at a time; only its slice of the 11-dim delta may become nonzero. Guards the
    concat order (a swapped switch/move block would be silent otherwise)."""
    on = _make(pointer_head=True).eval()
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        on.pointer_head.switch_score.weight.copy_(
            torch.rand(on.pointer_head.switch_score.weight.shape, generator=g) - 0.5)
        on(_obs())
        d = on.last_pointer_delta
    assert bool((d[:, SWITCH_START:6].abs() > 0).any()), "switch scorer did not reach the switch block"
    assert float(d[:, MOVE_START:MOVE_START + 4].abs().max()) == 0.0, "leaked into the move block"
    assert float(d[:, STRUGGLE].abs().max()) == 0.0, "leaked into struggle"


# ------------------------------------------------------- versioning
def _mv(**ov):
    return ModelVersion.from_layout_and_policy_kwargs(_layout, {"features_extractor_kwargs": ov})


def test_version_gate_and_migration():
    assert MODEL_CONFIG_VERSION >= 49
    on, off = _mv(pointer_head=True), _mv(pointer_head=False)
    for a, b in ((on, off), (off, on)):
        with pytest.raises(ModelVersionError, match="pointer_head"):
            a.check_compatible(b)
    on.check_compatible(_mv(pointer_head=True))
    migrated = _migrate_config({"config_version": 48})
    assert migrated["pointer_head"] is False and migrated["config_version"] >= 49
