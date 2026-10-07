"""X5 U3 part 2: the opponent ACTIVE's move axis under `--belief-tokens fixed_mass`
(`gen3_x5_belief_tokens_v1`; design_x5_belief_tokens.md §3.1 / §3.2 "Moves", §3.5 classes S and M).

* **Class S — ONE order.** The E4 seats, the op's top-K seat axis (its pair cells, α's seats), the D3
  edge cells and the intent operands all read the move group's seats (revealed first, then the top
  unrevealed by π_m, ties to the lower num); no `torch.topk` selects them (F-X5-13). Reverting the E4
  or the op side to its own top-K fails the agreement test.
* **The revealed Hidden Power seat** is priced as its typed mixture E_t[f(HP_t)] (the extended axis ⊕
  `mix`): the mix rows, the weights, and the contraction.
* **Class M — the presence-scaled max (§9 M2 = C).** The op's incoming maxes weight each candidate by
  its fixed-mass presence (1 revealed): I1 (π = 0 ≡ the candidate absent) holds exactly; I2 is
  DECLARED FALSE (two half-presence copies read as ONE half threat) and pinned; on revealed-only
  inputs the weights are exactly 1 on the revealed moves (today's formula at the pinned 1).
* **OTHER_move** is the active's E5 seat: its mass Σ_beyond π_m (no clamp) and a presence-scaled worst
  case over its members.
"""
from __future__ import annotations

import copy

import pytest
import torch

from agents.model.hypothesis_set import HypothesisBuilder
from agents.model.hypothesis_tokens import fixed_mass_moves, other_move_cells
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM


@pytest.fixture(scope="module")
def layout():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    return Gen3ObservationEncoder(load_mappings()).get_layout()


@pytest.fixture(scope="module")
def builder(layout):
    return HypothesisBuilder(layout, global_input_dim=50, n_move_seats=6)


def _species_with_hp(builder):
    """A species num whose legal moves include Hidden Power (237)."""
    legal = builder.move_legal
    nums = torch.nonzero(legal[:, HIDDEN_POWER_MOVE_NUM]).flatten()
    return int(nums[nums > 0][0])


def _group(builder, revealed_rows, seed=0):
    """A move group for B rows of one HP-capable species, with the given revealed ids per row."""
    sp = _species_with_hp(builder)
    B = len(revealed_rows)
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, builder.n_moves, generator=g) * 2.0
    rev = torch.tensor([r + [0] * (4 - len(r)) for r in revealed_rows], dtype=torch.long)
    mg = builder.move_group(logits, torch.full((B,), sp, dtype=torch.long), rev)
    return mg, logits


def test_the_revealed_hidden_power_seat_is_its_typed_mixture(builder):
    legal_row = builder.move_legal[_species_with_hp(builder)]
    other = int(torch.nonzero(legal_row & builder.move_valid).flatten()[0])
    mg, logits = _group(builder, [[HIDDEN_POWER_MOVE_NUM, other], [other]])
    fm = fixed_mass_moves(mg, logits)
    K = mg.seat_nums.shape[1]
    hp_seat = (mg.seat_nums[0] == HIDDEN_POWER_MOVE_NUM).nonzero().flatten()
    assert hp_seat.numel() == 1
    k = int(hp_seat[0])
    p_t = torch.sigmoid(logits[0, 355:371])
    p_t = p_t / p_t.sum()
    # the HP row of `mix` is the typed weights on the 16 extended columns, nothing on the seats
    assert torch.allclose(fm.mix[0, k, K:], p_t) and float(fm.mix[0, k, :K].abs().sum()) == 0.0
    # every other seat (and every seat of the HP-free row) is the identity
    eye = torch.eye(K, K + 16)
    rows = [i for i in range(K) if i != k]
    assert torch.equal(fm.mix[0, rows], eye[rows]) and torch.equal(fm.mix[1], eye)
    # the candidate weights: 1 on a revealed move, the typed weights on 355..370, 0 on the bare 237
    assert float(fm.w_all[0, other]) == 1.0 and float(fm.w_all[0, HIDDEN_POWER_MOVE_NUM]) == 0.0
    assert torch.allclose(fm.w_all[0, 355:371], p_t)
    # contraction: a per-num quantity on the extended axis lands on the HP seat as E_t[f(HP_t)]
    f = torch.arange(builder.n_moves, dtype=torch.float32)[fm.idx_ext]            # [B,K+16]
    out = fm.mix_seats(f, dim=1)
    assert torch.allclose(out[0, k], (p_t * torch.arange(355, 371, dtype=torch.float32)).sum())
    assert torch.equal(out[1], fm.seat_nums[1].float())


def test_seat_presence_and_structural_liveness(builder):
    legal_row = builder.move_legal[_species_with_hp(builder)]
    lm = torch.nonzero(legal_row & builder.move_valid).flatten()
    four = [int(x) for x in lm[:4]]
    mg, logits = _group(builder, [four, four[:1], []])
    fm = fixed_mass_moves(mg, logits)
    # all four revealed: k_m = 0, so only the four revealed seats carry mass
    assert fm.seat_on[0].tolist() == [True] * 4 + [False] * 2
    assert torch.equal(fm.seat_w[0, :4], torch.ones(4))
    # one revealed: seat 0 pinned at 1 (log 0), the rest π_m ∈ (0,1) with their exact log
    assert bool(fm.seat_on[1].all()) and float(fm.seat_logp[1, 0]) == 0.0
    assert bool((fm.seat_w[1, 1:] < 1).all()) and bool((fm.seat_w[1, 1:] > 0).all())
    assert torch.allclose(fm.seat_logp[1, 1:].exp(), fm.seat_w[1, 1:], rtol=1e-5)
    # OTHER_move: everything past the seats, its mass = Σ π_m there (an expected count, unclamped)
    assert torch.allclose(fm.other_mass, torch.where(fm.beyond, mg.presence.pi, 0.0).sum(-1))
    assert not bool(fm.other_live[0])


def test_other_move_seat_is_its_mass_and_a_presence_scaled_worst_case(builder):
    mg, logits = _group(builder, [[]], seed=3)
    fm = fixed_mass_moves(mg, logits)
    from agents.model.damage_tables import build_damage_buffers
    bufs = build_damage_buffers(builder.n_moves, builder.n_species, 100)
    bp, acc, phys = bufs["MOVE_BP"], bufs["MOVE_ACCURACY"], bufs["MOVE_PHYS"]
    c = other_move_cells(fm, bp, acc, phys)
    assert torch.allclose(c[0, 0], fm.other_mass[0]) and float(c[0, 3]) == 1.0
    score = torch.where(fm.beyond, fm.w_all, 0.0) * bp / 150.0 * acc
    assert torch.allclose(c[0, 1], (score * phys).amax(-1)[0])
    assert torch.allclose(c[0, 2], (score * (1 - phys)).amax(-1)[0])


# ----------------------------------------------------------------------------------- class M (moves)
def test_class_m_presence_scaled_max_I1_exact_I2_declared_false():
    """The op's reduction site (`DamageOperator._chan_max`, hard_max) under fixed-mass weights:
    max_m (π_m · v_m). I1: a candidate at π = 0 reads exactly as absent. I2 is FALSE by declaration
    (§9 M2 = C): two copies at π/2 read as π/2, not π — pinned so a future 'fix' is a decision."""
    from agents.model.damage_op import DamageOperator
    torch.manual_seed(0)
    v = torch.rand(4, 6, 10, dtype=torch.float64)                                # values >= 0
    pi = torch.rand(4, 1, 10, dtype=torch.float64)
    mask = (torch.rand(1, 1, 10) < 0.6).to(torch.float64)
    chan = DamageOperator._chan_max
    base = chan(None, pi * v, mask)                                              # type: ignore[arg-type]
    pi0 = pi.clone()
    pi0[..., 3] = 0.0
    drop = chan(None, (pi0 * v), mask)                                           # type: ignore[arg-type]
    absent = chan(None, torch.cat([(pi * v)[..., :3], (pi * v)[..., 4:]], -1),   # type: ignore[arg-type]
                  torch.cat([mask[..., :3], mask[..., 4:]], -1))
    assert torch.equal(drop, absent)                                             # I1, bit-exact
    # I2 declared false: split candidate 0 (on-channel, the maximiser) into two halves
    m1 = torch.ones(1, 1, 1, dtype=torch.float64)
    one = chan(None, pi[..., :1] * v[..., :1], m1)                               # type: ignore[arg-type]
    halves = chan(None, torch.cat([pi[..., :1] / 2 * v[..., :1]] * 2, -1),       # type: ignore[arg-type]
                  torch.ones(1, 1, 2, dtype=torch.float64))
    assert torch.allclose(halves, one / 2) and not torch.allclose(halves, one)
    assert base.shape == (4, 6)


# ----------------------------------------------------------------------------------- the real policy
@pytest.fixture(scope="module")
def fm_policy():
    from agents.model.hypothesis_set_test import _unperturbed_learner
    from main.train.production_args import production_args
    a = production_args()
    return _unperturbed_learner(a)


def _golden(n=64):
    from agents.model.hypothesis_set_test import _obs_from_golden
    return _obs_from_golden(n)


def test_one_order_feeds_the_e4_seats_the_op_seat_axis_and_alphas_seats(fm_policy):
    pol = copy.deepcopy(fm_policy.policy).eval()
    fe = pol.features_extractor
    obs = _golden()
    with torch.no_grad():
        fe(obs)
    hs = fe.last_hypothesis
    seats = hs.moves.seat_nums
    assert torch.equal(fe.entity_seats.last_cand[0], seats)
    assert torch.equal(fe.damage_op.last_topk_idx, seats)
    # X5 U4: α is retired under fixed_mass — the FLAT pointer's move columns are the same seats
    assert not hasattr(fe.stash, "alpha_seat_nums")             # the blob α stash is DELETED (part 2)
    assert torch.equal(fe.last_flat_intent.seat_nums, seats)
    assert torch.equal(fe.last_flat_intent.cand_ids[:, :seats.shape[1]], seats)
    assert torch.equal(fe.damage_op.last_topk_w, hs.moves.seat_pi)
    assert fe.damage_op.stash.seat_mix is not None
    # the op's class-M weights ARE the fixed-mass presence (detached), not sigmoid(logits)
    bi = torch.arange(seats.shape[0])
    fmv = fixed_mass_moves(hs.moves, fe.last_move_belief_logits[bi, fe.unpack(obs).opp_active_local])
    assert torch.equal(fe.damage_op.last_w_all, fmv.w_all * fe.damage_op.HP_CAND_MASK[None, :])
    assert not fe.damage_op.last_w_all.requires_grad


def test_the_op_seat_axis_contracts_the_extended_axis(fm_policy):
    """Where no Hidden Power is revealed the contraction is the identity: each seat's pair cell is
    the raw cell at its own num (the blob's gather), so the extension changes nothing there."""
    pol = copy.deepcopy(fm_policy.policy).eval()
    fe = pol.features_extractor
    obs = _golden()
    with torch.no_grad():
        fe(obs)
    op = fe.damage_op
    mix = op.stash.seat_mix
    K = mix.shape[1]
    ident = (mix == torch.eye(K, mix.shape[2])).all(-1).all(-1)                  # rows w/o a revealed HP
    assert bool(ident.any())
    ext = op.stash.seat_ext_idx
    assert torch.equal(ext[:, :K], op.last_topk_idx)
    assert torch.equal(ext[:, K:], torch.arange(355, 371).expand(ext.shape[0], -1))


def test_fixed_mass_refuses_fewer_than_four_move_seats():
    import gymnasium as gym
    import numpy as np
    from agents.model.extractor_arch import build_extractor_arch_kwargs
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from main.train.production_args import production_args
    a = production_args()
    enc = Gen3ObservationEncoder(load_mappings())
    kw = build_extractor_arch_kwargs(a, base=enc.get_features_extractor_kwargs())
    kw["entity_topk_seats"] = 3
    kw["damage_topk_k"] = 3
    space = gym.spaces.Box(0.0, 1.0, shape=(kw["layout"]["total_dim"],), dtype=np.float32)
    with pytest.raises(ValueError, match="entity_topk_seats >= 4"):
        Gen3FeaturesExtractor(space, **kw)
