"""gen3_move_legality_by_id_v1 — our active's request-order legality reaches its SORTED per-mon move slots
by move-num IDENTITY, never by position.

From gen3_frame_deletion_v1 (bcdd868b) the legacy `PokemonEncoder` wrote `our_active_req_move_legal`
(request order) onto the move slots (sorted by `Move.id`) BY POSITION: on 6.8 % of real move-bearing
decisions a choosable move's token read "illegal". Every test here FAILS on a revert to the positional
write: the synthetic table pins the rule, and the two real-row tests pin the CONSUMER (the move network's
input) — the move a legality bit lands on, and the per-mon move tokens' invariance to request order.
"""
import types

import numpy as np
import pytest
import torch

from agents.model.extractor_ctx import active_move_legality_sorted, active_request_sorted_match


def _ctx(sorted_ids, req_ids, legal):
    B = len(sorted_ids)
    c = types.SimpleNamespace(batch_size=B, device=torch.device("cpu"),
                              our_active_idx=torch.zeros(B, dtype=torch.long),
                              all_move_ids=torch.zeros(B, 12, 4, dtype=torch.long),
                              our_active_req_move_ids=torch.tensor(req_ids, dtype=torch.float32),
                              our_active_req_move_legal=torch.tensor(legal, dtype=torch.float32))
    c.all_move_ids[:, 0] = torch.tensor(sorted_ids)
    return c


def test_legality_follows_the_move_not_the_position():
    # A REAL golden row (blob buffer): Choice-locked into num 251 (request slot 3, sorted slot 1).
    c = _ctx([[332, 251, 89, 355]], [[89, 355, 332, 251]], [[0.0, 0.0, 0.0, 1.0]])
    assert active_move_legality_sorted(c).tolist() == [[0.0, 1.0, 0.0, 0.0]]   # positional read [0,0,0,1]
    m = active_request_sorted_match(c)
    assert m.sum().item() == 4 and m[0, 3, 1].item()


def test_unmatched_and_empty_slots_read_illegal():
    # forced switch / Struggle: an all-zero request block → our active's slots all 0 (as before the fix);
    # a sorted slot the request does not name → 0.
    c = _ctx([[34, 89, 0, 0], [34, 89, 92, 0]], [[0, 0, 0, 0], [89, 34, 0, 0]],
             [[0, 0, 0, 0], [1.0, 1.0, 0, 0]])
    assert active_move_legality_sorted(c).tolist() == [[0, 0, 0, 0], [1.0, 1.0, 0.0, 0.0]]


@pytest.fixture(scope="module")
def fe_rows():
    """The production-surface extractor (eval mode) + the REAL golden-buffer rows."""
    from agents.action.ordering_integrity import row_offsets
    from agents.training import learner_golden as LG
    from main.train.production_args import production_args

    with np.load(LG.BUFFER_PATH) as z:
        obs = z["obs:observation"]
    obs = torch.as_tensor(obs.reshape(-1, obs.shape[-1]).astype(np.float32))
    fe = LG.build_learner(args=production_args()).policy.eval().features_extractor
    o = row_offsets()
    rows = []                                  # misaligned rows: non-alphabetical order AND an illegal move
    for i in range(obs.shape[0]):
        r = obs[i].numpy()
        team = r[o.team0:o.team0 + 6 * o.mon_dim].reshape(6, o.mon_dim)
        act = np.flatnonzero(team[:, o.active_col] > 0.5)
        req, legal = r[o.req_ids], r[o.req_legal]
        if len(act) == 1 and (req > 0).sum() == 4 and (legal < 0.5).any() and \
                not np.array_equal(team[act[0], o.slot_id_cols], req):
            rows.append(i)
    assert rows, "no misaligned real row in the golden buffer — these tests would be vacuous"
    return fe, obs, rows, o


def _move_net_input(fe, obs):
    """The move network's input, [B, 12 mons, 4 sorted slots, F] (the legality bit is one of its columns)."""
    got = {}

    def hook(mod, args, out):                  # returns None: a forward hook's return REPLACES the output
        got["x"] = args[0].clone()

    h = fe.pokemon_encoder.move_network.register_forward_hook(hook)
    try:
        with torch.no_grad():
            fe({"observation": obs})
    finally:
        h.remove()
    return got["x"].reshape(obs.shape[0], 12, 4, -1)


def test_a_legality_bit_lands_on_its_own_moves_slot(fe_rows):
    """Flip request slot k's legality: the ONLY move-network input that moves is our active's SORTED slot
    holding request move k (the positional write moved sorted slot k instead)."""
    fe, obs, rows, o = fe_rows
    i = rows[0]
    row = obs[i:i + 1]
    team = row[0, o.team0:o.team0 + 6 * o.mon_dim].reshape(6, o.mon_dim)
    act = int(torch.nonzero(team[:, o.active_col] > 0.5)[0, 0])
    sorted_ids = team[act, torch.as_tensor(o.slot_id_cols)].tolist()
    req_ids = row[0, o.req_ids].tolist()
    k = next(k for k in range(4) if sorted_ids.index(req_ids[k]) != k)   # a slot whose position differs
    flipped = row.clone()
    flipped[0, o.req_legal.start + k] = 1.0 - flipped[0, o.req_legal.start + k]
    # the move network's input is a function of the per-mon block, board context and this bit only (the
    # op reads the request legality downstream), so exactly one (mon, slot) row may move.
    a, b = _move_net_input(fe, row), _move_net_input(fe, flipped)
    moved = (a != b).any(-1)[0]                                            # [12, 4]
    assert moved.sum().item() == 1, moved.nonzero().tolist()
    assert moved[act, sorted_ids.index(req_ids[k])].item()


def test_per_mon_move_tokens_are_invariant_to_request_order(fe_rows):
    """Permute the request block (ids, types, legality together — the same moves, listed in another order):
    the per-mon move tokens depend on the moveset, never on the request's listing order."""
    fe, obs, rows, o = fe_rows
    x = obs[rows]
    y = x.clone()
    perm = [2, 0, 3, 1]
    for blk in range(3):
        s = o.req0 + 4 * blk
        y[:, s:s + 4] = x[:, s:s + 4][:, perm]
    with torch.no_grad():
        fe({"observation": x})
        tx = fe.pokemon_encoder.last_move_tokens.clone()
        fe({"observation": y})
        ty = fe.pokemon_encoder.last_move_tokens.clone()
    assert torch.equal(tx, ty), (tx - ty).abs().max().item()
