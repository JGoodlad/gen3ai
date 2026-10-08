"""`gen3_static_tokens_v1` — `--token-encoding static` (`designs/endstate/design_static_tokens.md`).

Each test names what a revert would break:

* the STATIC identity S of a mon is a function of its own (species, set) fields alone: the same slot gives
  the same S on another row, in another seat, under another board (fails if any board / dynamic / position
  input enters S);
* the four moves are a SET: permuting a mon's move slots leaves S, D and the token unchanged (fails on a
  concatenation, legacy's sorted-by-id Linear);
* the DYNAMIC state reaches the token (HP, status, the active context) and leaves S untouched (fails if D is
  dropped or folded into S);
* the BOARD context (clock, weather, faint counts, Spikes, screens) does not enter the per-mon encoder
  (fails when the static encoder reads it; the legacy control shows the same perturbation moves legacy);
* X5's hypothesis tokens are the dex table encoded once and GATHERED, and
  equal the per-row pass on every hypothesis slot (fp32 within 1e-5; fp64 within 1e-12), on both static
  branches; the forward takes that path once (fails on revert to the legacy split);
* `legacy` is the default and builds `PokemonEncoder` (its byte identity is the K9 golden's and the obs
  golden's, unchanged);
* the opponent stat prior's non-HP columns ARE the damage operator's `SPECIES_SPREAD_PRIOR`, and our team's
  actual stats are the gen-3 level-100 formula.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict, List, Tuple

import pytest
import torch

import agents.model.extractor_forward as EF
from agents.model.encoders import PokemonEncoder
from agents.model.extractor_ctx import ExtractorContext, slice_pokemon_categoricals
from agents.model.hypothesis_tokens import hypothesis_ctx
from agents.model.static_tokens import StaticTokenEncoder, static_hypothesis_tokens
from agents.observation.constants import (POKEMON_CONDITION_OFFSET, POKEMON_HP_OFFSET, POKEMON_MOVES_OFFSET,
                                          POKEMON_SPECIES_OFFSET, POKEMON_SPREAD_OFFSET, TEAM_SIZE)

#: fp32 bound on two evaluations of the same function that differ only in summation order / batch shape.
FP32_ATOL = 1e-5
#: fp64 bound for the same.
FP64_ATOL = 1e-12
#: A perturbation "moves" a token when its max |Δ| exceeds this (the effects measured here are O(0.1)).
MOVES = 1e-3


def _rows(model: Any, n: int = 0) -> Dict[str, torch.Tensor]:
    from stable_baselines3.common.utils import obs_as_tensor
    rb = model.rollout_buffer
    obs = obs_as_tensor({k: v.reshape((-1,) + v.shape[2:]) for k, v in rb.observations.items()}, "cpu")
    if n:
        N = next(iter(obs.values())).shape[0]
        idx = (torch.arange(n) * max(1, N // n) + torch.arange(n) // N) % N
        obs = {k: v[idx] for k, v in obs.items()}
    return obs


def _static_args() -> Any:
    from main.train.production_args import production_args
    a = production_args()
    a.token_encoding = "static"
    return a


@pytest.fixture(scope="module")
def static_model() -> Any:
    """Production (X5's hypothesis tokens, the only belief representation) + `--token-encoding static`."""
    from agents.training import learner_golden as LG
    m = LG.build_learner(args=_static_args())
    LG.load_buffer_into(m)
    fe = m.policy.features_extractor
    assert isinstance(fe.pokemon_encoder, StaticTokenEncoder) and fe.hypothesis_builder is not None   # PRECONDITION
    return m


@pytest.fixture(scope="module")
def legacy() -> Any:
    from agents.training import learner_golden as LG
    m = LG.build_learner()
    LG.load_buffer_into(m)
    return m


def _reslice(ctx: ExtractorContext, pp: torch.Tensor, layout: Dict[str, Any]) -> ExtractorContext:
    """``ctx`` with its per-mon block replaced by ``pp`` and every categorical re-sliced from it."""
    ids = slice_pokemon_categoricals(pp, layout)
    return dataclasses.replace(ctx, pokemon_part=pp, **ids)


def _board_perturbed(ctx: ExtractorContext, seed: int = 3) -> ExtractorContext:
    """The five board fields legacy's per-mon encoder reads, each replaced by noise."""
    g = torch.Generator().manual_seed(seed)

    def noise(t: torch.Tensor) -> torch.Tensor:
        return torch.rand(t.shape, generator=g, dtype=t.dtype)
    return dataclasses.replace(ctx, turn_feature=noise(ctx.turn_feature), weather_feature=noise(ctx.weather_feature),
                               fainted_feature=noise(ctx.fainted_feature), spikes_feature=noise(ctx.spikes_feature),
                               screen_feature=noise(ctx.screen_feature))


# ---------------------------------------------------------------------------------------------- legacy
def test_legacy_is_the_default_and_builds_the_pokemon_encoder(legacy):
    fe = legacy.policy.features_extractor
    assert fe.token_encoding == "legacy"
    assert type(fe.pokemon_encoder) is PokemonEncoder


def test_the_flag_is_recorded_and_read_back(static_model, legacy):
    from agents.model.snapshot import arch_toggles_from_model
    assert arch_toggles_from_model(static_model)["token_encoding"] == "static"
    assert arch_toggles_from_model(legacy)["token_encoding"] == "legacy"


# ------------------------------------------------------------------------------------ board context
def test_board_context_does_not_enter_the_static_encoder(static_model, legacy):
    for model, static in ((static_model, True), (legacy, False)):
        fe = model.policy.features_extractor
        with torch.no_grad():
            ctx = fe.unpack(_rows(model))
            a = fe.pokemon_encoder(ctx, fe.embeddings)
            b = fe.pokemon_encoder(_board_perturbed(ctx), fe.embeddings)
        d = (a - b).abs().max().item()
        if static:
            assert d == 0.0, f"a board field reached the static per-mon encoder (max |Δtoken| {d:.3g})"
        else:   # the CONTROL: the same perturbation moves legacy's tokens, so the static check is not vacuous
            assert d > MOVES, f"PRECONDITION: the board perturbation must move legacy's tokens ({d:.3g})"


# ------------------------------------------------------------------------------------ static identity
def test_the_same_species_and_set_gives_the_same_static_identity_anywhere(static_model):
    """Copy mons between rows and seats (other boards, other teams, our side vs theirs): S follows the slot."""
    fe = static_model.policy.features_extractor
    pe = fe.pokemon_encoder
    with torch.no_grad():
        ctx = fe.unpack(_rows(static_model))
        B = ctx.batch_size
        pp = ctx.pokemon_part.clone()
        # four REAL mons from four different rows, two ours and two theirs, each copied into a seat of the
        # other side on a row from the far end of the buffer (a different battle state)
        src: List[Tuple[int, int]] = []
        for r, side in ((0, 0), (B // 4, 1), (B // 2, 0), (3 * B // 4, 1)):
            m = next(side * TEAM_SIZE + j for j in range(TEAM_SIZE) if ctx.species_ids[r, side * TEAM_SIZE + j] > 0)
            src.append((r, m))
        dst: List[Tuple[int, int]] = [(B - 1 - i, (m + TEAM_SIZE + i) % (2 * TEAM_SIZE)) for i, (_, m) in enumerate(src)]
        for (r0, m0), (r1, m1) in zip(src, dst):
            pp[r1, m1] = pp[r0, m0]
        ctx2 = _board_perturbed(_reslice(ctx, pp, fe.layout))
        s1 = pe.parts(ctx, fe.embeddings).static
        s2 = pe.parts(ctx2, fe.embeddings).static
    assert len({r for r, _ in src} | {r for r, _ in dst}) == 8, "PRECONDITION: eight distinct rows"
    for (r0, m0), (r1, m1) in zip(src, dst):
        d = (s1[r0, m0] - s2[r1, m1]).abs().max().item()
        assert d <= FP32_ATOL, f"S of the same slot differs between ({r0},{m0}) and ({r1},{m1}): {d:.3g}"
    # and NOT a constant: two different mons have different identities
    assert (s1[0, 0] - s1[0, 1]).abs().max().item() > MOVES


def test_the_move_set_is_permutation_invariant(static_model, legacy):
    perm = [2, 0, 3, 1]
    for model, static in ((static_model, True), (legacy, False)):
        fe = model.policy.features_extractor
        pe = fe.pokemon_encoder
        slot = fe.layout['pokemon']['moves']['layout']['slots']
        width = slot[1]['offset'] - slot[0]['offset']
        with torch.no_grad():
            ctx = fe.unpack(_rows(model))
            pp = ctx.pokemon_part
            blocks = [pp[..., POKEMON_MOVES_OFFSET + k * width:POKEMON_MOVES_OFFSET + (k + 1) * width]
                      for k in range(4)]
            pp2 = torch.cat([pp[..., :POKEMON_MOVES_OFFSET]] + [blocks[k] for k in perm]
                            + [pp[..., POKEMON_MOVES_OFFSET + 4 * width:]], dim=-1)
            ctx2 = _reslice(ctx, pp2, fe.layout)
            a = pe(ctx, fe.embeddings)
            b = pe(ctx2, fe.embeddings)
            moved = (pe.last_move_tokens is not None)
        d = (a - b).abs().max().item()
        assert moved
        if static:
            assert d <= FP32_ATOL, f"permuting the move slots moved the static token by {d:.3g}"
            p1, p2 = pe.parts(ctx, fe.embeddings), pe.parts(ctx2, fe.embeddings)
            assert (p1.static - p2.static).abs().max().item() <= FP32_ATOL
            # the per-move tokens PERMUTE with the slots (equivariant)
            assert (p1.move_tokens[:, :, perm] - p2.move_tokens).abs().max().item() <= FP32_ATOL
        else:   # CONTROL: legacy concatenates in slot order
            assert d > MOVES, f"PRECONDITION: legacy must be order-sensitive ({d:.3g})"


# ------------------------------------------------------------------------------------- dynamic state
def test_the_dynamic_state_reaches_the_token_and_not_the_identity(static_model):
    fe = static_model.policy.features_extractor
    pe = fe.pokemon_encoder
    with torch.no_grad():
        ctx = fe.unpack(_rows(static_model))
        base = pe.parts(ctx, fe.embeddings)
        pp = ctx.pokemon_part.clone()
        pp[..., POKEMON_HP_OFFSET] = pp[..., POKEMON_HP_OFFSET] * 0.5                     # HP
        pp[..., POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 3] = torch.tensor([0., 1., 0.])  # burned
        hp = pe.parts(_reslice(ctx, pp, fe.layout), fe.embeddings)
        ctx_act = dataclasses.replace(ctx, our_ctx_raw=ctx.our_ctx_raw + 1.0)               # boosts / volatiles
        act = pe.parts(ctx_act, fe.embeddings)
    real = ctx.species_ids > 0
    assert (base.static - hp.static).abs().max().item() == 0.0, "a dynamic field reached S"
    assert (base.tokens - hp.tokens)[real].abs().amax(-1).min().item() > MOVES, "HP / status did not reach a token"
    ar = torch.arange(ctx.batch_size)
    d_act = (base.tokens - act.tokens).abs().amax(-1)                                    # [B,12]
    assert d_act[ar, ctx.our_active_idx].min().item() > MOVES, "our active's context did not reach its token"
    bench = torch.ones_like(d_act, dtype=torch.bool)
    bench[ar, ctx.our_active_idx] = False
    assert d_act[bench].max().item() == 0.0, "the active context reached a benched mon"


# -------------------------------------------------------------------------------- the X5 table gather
def _both(fe: Any, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
    ctx = fe.unpack(obs)
    hs = fe._build_hypothesis_species(ctx, fe.pokemon_encoder(ctx, fe.embeddings))
    ref = fe.pokemon_encoder(hypothesis_ctx(ctx, hs, fe.layout), fe.embeddings)[:, TEAM_SIZE:2 * TEAM_SIZE]
    new = static_hypothesis_tokens(fe.pokemon_encoder, fe.embeddings, hs.slot_species,
                                   fe.hypothesis_builder.dex_rows)
    return ref, new, hs, ctx


#: Both static branches: B·6 < 400 encodes the slots' own dex rows (T2's small buckets); B·6 ≥ 400 the table.
BRANCHES = [pytest.param(8, id="per_slot_B8"), pytest.param(128, id="table_B128")]


@pytest.mark.parametrize("n", BRANCHES)
def test_the_gathered_hypothesis_tokens_are_the_per_row_pass(static_model, n):
    fe = static_model.policy.features_extractor
    with torch.no_grad():
        ref, new, hs, ctx = _both(fe, _rows(static_model, n))
    hyp = hs.slot_is_hypothesis
    assert int(hyp.sum()) >= 4, "PRECONDITION: the rows hold hypothesis slots"
    act = torch.nn.functional.one_hot(ctx.opp_active_local, TEAM_SIZE).bool()
    assert not (hyp & act).any(), "PRECONDITION: a hypothesis slot is never the opponent's active"
    d32 = (ref - new)[hyp].abs().max().item()
    assert d32 <= FP32_ATOL, f"fp32 |gathered - per-row| = {d32:.3g}"
    pol = copy.deepcopy(static_model.policy).double()
    obs64 = {k: (v.double() if v.is_floating_point() else v) for k, v in _rows(static_model, n).items()}
    with torch.no_grad():
        ref64, new64, hs64, _ = _both(pol.features_extractor, obs64)
    d64 = (ref64 - new64)[hs64.slot_is_hypothesis].abs().max().item()
    assert d64 <= FP64_ATOL, f"fp64 |gathered - per-row| = {d64:.3g}: not the same function"


def test_the_fixed_mass_forward_takes_the_static_gather_once(static_model, monkeypatch):
    fe = static_model.policy.features_extractor
    calls = [0]
    real = static_hypothesis_tokens

    def counted(*a: Any, **k: Any) -> torch.Tensor:
        calls[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(EF, "static_hypothesis_tokens", counted, raising=False)
    with torch.no_grad():
        fe(_rows(static_model))
    assert calls[0] == 1, f"the static gather ran {calls[0]}x (expected once per forward)"


def test_the_static_hypothesis_token_is_a_function_of_the_species_alone(static_model):
    """Two batches with the same hypothesis species but different boards / teams give the same tokens."""
    fe = static_model.policy.features_extractor
    species = torch.tensor([[5, 248, 227, 0, 0, 0], [0, 0, 0, 248, 5, 227]])
    with torch.no_grad():
        t = static_hypothesis_tokens(fe.pokemon_encoder, fe.embeddings, species, fe.hypothesis_builder.dex_rows)
    assert (t[0, 0] - t[1, 4]).abs().max().item() <= FP32_ATOL
    assert (t[0, 1] - t[1, 3]).abs().max().item() <= FP32_ATOL
    assert (t[0, 0] - t[0, 1]).abs().max().item() > MOVES


# ----------------------------------------------------------------------------------------- the stats
def test_the_opponent_stat_prior_is_the_operators_and_adds_hp(static_model):
    fe = static_model.policy.features_extractor
    prior = fe.pokemon_encoder.STAT_PRIOR
    assert torch.equal(prior[:, 1:, :], fe.damage_op.SPECIES_SPREAD_PRIOR)
    valid = fe.damage_op.SPECIES_SPREAD_PRIOR[:, 0, 0] > 0
    shedinja = 292                                                    # its HP is always 1 (base HP 1)
    assert prior[shedinja, 0, 0].item() == 1.0
    valid[shedinja] = False
    # the L100 HP floor is 2·base + 31 + 110 >= 2·20 + 141 = 181 for every other gen-3 species
    assert (prior[valid, 0, 0] >= 181).all(), "an HP mean below the level-100 floor"


def test_our_actual_stats_are_the_level_100_formula(static_model):
    pe = static_model.policy.features_extractor.pokemon_encoder
    pp = torch.zeros(1, 1, 122)
    base = torch.tensor([80., 100., 90., 60., 70., 110.])
    pp[0, 0, POKEMON_SPECIES_OFFSET + 1:POKEMON_SPECIES_OFFSET + 7] = base / 255.0
    sp = POKEMON_SPREAD_OFFSET
    pp[0, 0, sp:sp + 6] = 1.0                                       # IV 31
    pp[0, 0, sp + 6:sp + 12] = torch.tensor([4., 252., 0., 0., 0., 252.]) / 252.0
    pp[0, 0, sp + 12] = 1.0                                         # spread known
    pp[0, 0, sp + 13:sp + 18] = torch.tensor([1.1, 1.0, 0.9, 1.0, 1.0])   # Adamant
    feat = pe._actual_stats(pp, torch.zeros(1, 1, dtype=torch.long))[0, 0]
    want = torch.tensor([2 * 80 + 31 + 1 + 110, (2 * 100 + 31 + 63 + 5) * 1.1, 2 * 90 + 31 + 5,
                         (2 * 60 + 31 + 5) * 0.9, 2 * 70 + 31 + 5, 2 * 110 + 31 + 63 + 5]) / 500.0
    assert torch.allclose(feat[:6], want, atol=1e-6)
    assert torch.equal(feat[6:12], torch.zeros(6)) and feat[12].item() == 1.0


def _layout_containers(obj: Any, out: Dict[int, str], path: str = "layout") -> None:
    if isinstance(obj, (dict, list)):
        out[id(obj)] = path
        for k, v in (obj.items() if isinstance(obj, dict) else enumerate(obj)):
            _layout_containers(v, out, f"{path}[{k!r}]")


def test_the_static_encoder_forward_holds_no_reference_into_the_layout(static_model):
    """gen3_static_layout_ints_v1 (F-ST-9): the encoder's forward state is PLAIN INTS, never a sub-dict of the
    layout (`ObsUnpack.layout` reaches the same object, and dynamo guarded the two paths' identity — the
    OBJECT_ALIASING guard that fired after the compile lock on the CUDA launch, 2026-10-07). Only the inert
    `layout` record itself may alias it; the move columns are exactly the slot layout's."""
    fe = static_model.policy.features_extractor
    pe = fe.pokemon_encoder
    held: Dict[int, str] = {}
    _layout_containers(fe.unpack.layout, held)
    assert len(held) > 10                                                  # PRECONDITION: the walk saw the tree
    bad = [(name, held[id(v)]) for name, v in vars(pe).items()
           if name != "layout" and not name.startswith("_modules") and id(v) in held]
    assert not bad, f"the static encoder holds layout containers: {bad}"
    pk = fe.unpack.layout['pokemon']
    msl, mo = pk['moves']['layout']['slot_layout'], pk['moves']['offset']
    for (spans, k, c), slot in zip(pe._move_cols, pk['moves']['layout']['slots']):
        s0 = mo + slot['offset']
        assert spans == ((s0 + msl['power']['offset'], s0 + msl['type']['offset']),
                         (s0 + msl['type']['offset'] + msl['type']['dim'], s0 + msl['known']['offset']),
                         (s0 + msl['max_pp']['offset'], s0 + msl['max_pp']['offset'] + msl['max_pp']['dim']),
                         (s0 + msl['accuracy']['offset'], s0 + msl['never_miss']['offset'] + msl['never_miss']['dim']))
        assert (k, c) == (s0 + msl['known']['offset'], s0 + msl['current_pp']['offset'])
    assert len(pe._move_cols) == len(pk['moves']['layout']['slots']) == pe.num_moves
