"""`gen3_static_board_v1` — `--token-encoding static`, build STAGE 2 (`designs/endstate/design_static_tokens.md` §4).

Each test names what a revert would break:

* the two SIDE tokens read the SAME columns of their own side (one shared `side_proj`): swapping the two sides'
  content swaps the two side tokens' content (fails if a side reads the other's facts, or the sides get
  different columns);
* the COUNTS reach a side token: a fainted count, an alive count (a mon at HP 0) and a revealed count each move
  their own side's token and nothing else (fails if a count is dropped: a masked key cannot be counted by
  attention, audit §4 A1);
* the per-mon OP CONTENT reaches every mon on BOTH sides as content: THEIR Spikes reach their mons' trunk input
  only through `op_content` (the control: zeroing it removes the effect), and our active's outgoing cells reach
  each of their mons (fails if OPC is dropped or one-sided, audit §4 A3);
* no board fact reaches the per-mon encoder, including the board scalars of `non_matchup_rest`;
* the `x` edge writes each mon to its OWN side's seat and `g` / `c4` to FIELD (fails on revert to the global seat);
* the readouts: `tower` drops the `non_matchup_rest` bypass (width and value); `trunk`'s state query attends over
  the three refined board tokens in the trunk's own seat order; the critic's pool reads the three board rows;
* `legacy` builds none of it (its byte identity is the K9 goldens' and the compiled-region goldens', unchanged).
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict, List

import pytest
import torch

from agents.model.arch_constants import D_MODEL, TRANSFORMER_N_HEADS
from agents.model.board_tokens import (BOARD_SEATS_LEGACY, BOARD_SEATS_STATIC, SIDE_FACTS, OpContent,
                                       field_features, side_features)
from agents.model.extractor_ctx import TOKEN_TYPE_OUR_SIDE, TOKEN_TYPE_THEIR_SIDE, ExtractorContext
from agents.model.projection import compute_projection_widths
from agents.model.team_transformer import EdgeBias
from agents.observation.constants import POKEMON_SPECIES_KNOWN_OFFSET, TEAM_SIZE

#: A perturbation "moves" a token when its max |Δ| exceeds this (the effects measured here are O(0.1)).
MOVES = 1e-4
#: fp32 bound on the same arithmetic reassociated.
FP32_ATOL = 1e-6


def _rows(model: Any, n: int = 0) -> Dict[str, torch.Tensor]:
    from stable_baselines3.common.utils import obs_as_tensor
    rb = model.rollout_buffer
    obs = obs_as_tensor({k: v.reshape((-1,) + v.shape[2:]) for k, v in rb.observations.items()}, "cpu")
    if n:
        N = next(iter(obs.values())).shape[0]
        obs = {k: v[torch.arange(n) * max(1, N // n) % N] for k, v in obs.items()}
    return obs


def _args(readout: str = "tower", encoding: str = "static") -> Any:
    from main.train.production_args import production_args
    a = production_args()
    a.token_encoding = encoding
    a.policy_readout = readout
    return a


def _learner(readout: str, encoding: str = "static") -> Any:
    from agents.training import learner_golden as LG
    m = LG.build_learner(args=_args(readout, encoding))
    LG.load_buffer_into(m)
    return m


@pytest.fixture(scope="module")
def static_tower() -> Any:
    m = _learner("tower")
    fe = m.policy.features_extractor
    assert fe.team_transformer.static_board and fe.op_content is not None      # PRECONDITION
    return m


@pytest.fixture(scope="module")
def static_trunk() -> Any:
    m = _learner("trunk")
    assert m.policy.features_extractor.policy_query is not None                   # PRECONDITION
    return m


@pytest.fixture(scope="module")
def legacy() -> Any:
    from agents.training import learner_golden as LG
    m = LG.build_learner()
    LG.load_buffer_into(m)
    return m


def _with_op_content(model: Any, amount: bool = False, outgoing: bool = False, bias: bool = False) -> Any:
    """A deep copy of the extractor whose op content is ZERO (its init) except the named projections, which get
    random weights (and random biases with ``bias``). The golden learner's build perturbs every parameter, so the
    fixture's op content is not at its zero init; each routing test sets it explicitly."""
    fe = copy.deepcopy(model.policy.features_extractor)
    g = torch.Generator().manual_seed(7)
    oc = fe.op_content
    for lin, live in ((oc.amount_proj, amount), (oc.outgoing_proj, outgoing)):
        with torch.no_grad():
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * 0.3 if live else torch.zeros_like(lin.weight))
            lin.bias.copy_(torch.randn(lin.bias.shape, generator=g) * 0.3 if (live and bias)
                           else torch.zeros_like(lin.bias))
    fe.eval()
    return fe


def _op_content_args(fe: Any, obs: Dict[str, torch.Tensor]) -> Any:
    seen: List[Any] = []
    h = fe.op_content.register_forward_pre_hook(lambda _m, args: seen.append(args))
    try:
        with torch.no_grad():
            fe(obs)
    finally:
        h.remove()
    assert len(seen) == 1
    return seen[0]


def _trunk_input(fe: Any, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """[B,12,D] the mon tokens as they ENTER the trunk (after every pre-trunk injection)."""
    seen: List[torch.Tensor] = []
    h = fe.team_transformer.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            fe(obs)
    finally:
        h.remove()
    assert len(seen) == 1
    return seen[0]


# ------------------------------------------------------------------------------------------- legacy
def test_legacy_builds_no_board_tokens_and_no_op_content(legacy):
    fe = legacy.policy.features_extractor
    tt = fe.team_transformer
    assert not tt.static_board and tt._total_tokens == 2 * TEAM_SIZE + 1 and tt.board_seats == BOARD_SEATS_LEGACY
    assert hasattr(tt, "global_proj") and not hasattr(tt, "side_proj") and fe.op_content is None
    assert fe.assembler.board_bypass


def test_static_builds_three_board_seats(static_tower):
    tt = static_tower.policy.features_extractor.team_transformer
    assert tt._total_tokens == 2 * TEAM_SIZE + 3 and tt.board_seats == BOARD_SEATS_STATIC
    assert not hasattr(tt, "global_proj") and tt.token_type_emb.num_embeddings == 9


def test_the_op_content_zero_init_survives_a_real_policy_build():
    """On the construction path training uses (MaskablePPO → SB3's orthogonal re-init → the guard)."""
    from agents.model.identity_init_test import _build_real_policy
    model, _ = _build_real_policy(token_encoding="static", edge_bias_families="x,g")
    oc = model.policy.features_extractor.op_content
    assert oc is not None and oc.outgoing_proj is not None
    for lin in (oc.amount_proj, oc.outgoing_proj):
        assert float(lin.weight.detach().abs().max()) == 0.0 and float(lin.bias.detach().abs().max()) == 0.0


# --------------------------------------------------------------------------------------- the sides
def _all_revealed(ctx: ExtractorContext) -> ExtractorContext:
    """Every mon revealed, so 'alive' means HP > 0 on both sides (the swap below is then an exact involution)."""
    pp = ctx.pokemon_part.clone()
    pp[:, :, POKEMON_SPECIES_KNOWN_OFFSET] = 1.0
    return dataclasses.replace(ctx, pokemon_part=pp, opp_addressable=ctx.hp_and_active[:, TEAM_SIZE:, 0] > 0)


def _distinct_sides(ctx: ExtractorContext, off: Any) -> ExtractorContext:
    """The board columns made different between the two sides on every row (the buffer's real boards mostly
    hold no Spikes / screens / Wish, which would make a side mix-up invisible)."""
    B = ctx.batch_size
    ar = torch.arange(B, dtype=ctx.non_matchup_rest.dtype)
    nmr = ctx.non_matchup_rest.clone()
    nmr[:, off.wish_our], nmr[:, off.wish_opp] = 0.25, 0.75
    sc = torch.stack([(ar % 2), 1 - (ar % 2)], -1).repeat(1, 4)                        # ours ≠ theirs per screen
    return dataclasses.replace(ctx, non_matchup_rest=nmr, screen_feature=sc,
                               spikes_feature=torch.stack([ar * 0 + 1 / 3, ar * 0 + 1.0], -1),
                               fainted_feature=torch.stack([ar * 0 + 1 / 6, ar * 0 + 3 / 6], -1))


def _swap_sides(ctx: ExtractorContext, off: Any) -> ExtractorContext:
    """The board with OUR side and THEIR side exchanged: per-mon halves, the paired board columns, the counts."""
    def halves(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t[:, TEAM_SIZE:], t[:, :TEAM_SIZE]], dim=1)
    nmr = ctx.non_matchup_rest.clone()
    nmr[:, off.wish_our], nmr[:, off.wish_opp] = ctx.non_matchup_rest[:, off.wish_opp], ctx.non_matchup_rest[:, off.wish_our]
    sc = ctx.screen_feature.reshape(-1, 4, 2).flip(-1).reshape(-1, 8)
    hpa = halves(ctx.hp_and_active)
    return dataclasses.replace(
        ctx, pokemon_part=halves(ctx.pokemon_part), hp_and_active=hpa, non_matchup_rest=nmr, screen_feature=sc,
        spikes_feature=ctx.spikes_feature.flip(-1), fainted_feature=ctx.fainted_feature.flip(-1),
        opp_addressable=hpa[:, TEAM_SIZE:, 0] > 0)


def test_the_side_tokens_are_side_relative(static_tower):
    fe = static_tower.policy.features_extractor
    tt = fe.team_transformer
    off = tt._board_offsets
    with torch.no_grad():
        ctx = _distinct_sides(_all_revealed(fe.unpack(_rows(static_tower))), off)
        sw = _swap_sides(ctx, off)
        f0, f1 = side_features(ctx, off), side_features(sw, off)
        b0, b1 = tt.board_tokens(ctx), tt.board_tokens(sw)
        types = tt.token_type_emb.weight
    for c in (0, 1, 2, 3, 4, 5, 7):     # every board-column fact differs between the sides on every row
        assert (f0[:, 0, c] != f0[:, 1, c]).all(), f"PRECONDITION: {SIDE_FACTS[c]} differs between the sides"
    assert torch.equal(f1[:, 0], f0[:, 1]) and torch.equal(f1[:, 1], f0[:, 0]), "swapping the sides did not swap the facts"
    assert torch.equal(field_features(sw, off), field_features(ctx, off)), "the field is not a side fact"
    # one SHARED projection: our side's token after the swap is their side's before, up to the type embedding
    t_our, t_their = types[TOKEN_TYPE_OUR_SIDE], types[TOKEN_TYPE_THEIR_SIDE]
    d = ((b1[:, 0] - t_our) - (b0[:, 1] - t_their)).abs().max().item()
    assert d <= FP32_ATOL, f"the sides do not share one projection over the same columns ({d:.3g})"
    assert (b0[:, 0] - b0[:, 1]).abs().max().item() > MOVES


def test_the_counts_reach_their_own_side_token(static_tower):
    fe = static_tower.policy.features_extractor
    tt = fe.team_transformer
    with torch.no_grad():
        ctx = fe.unpack(_rows(static_tower))
        base = tt.board_tokens(ctx)
        fainted = dataclasses.replace(ctx, fainted_feature=ctx.fainted_feature + torch.tensor([1 / 6, 0.0]))
        hpa = ctx.hp_and_active.clone()
        alive_ours = hpa[:, :TEAM_SIZE, 0] > 0
        assert alive_ours.any(1).all(), "PRECONDITION: every row has a live mon of ours"
        first = alive_ours.float().argmax(1)
        hpa[torch.arange(ctx.batch_size), first, 0] = 0.0
        dead = dataclasses.replace(ctx, hp_and_active=hpa)
        pp = ctx.pokemon_part.clone()
        pp[:, TEAM_SIZE:, POKEMON_SPECIES_KNOWN_OFFSET] = 0.0      # their active is always revealed: the count moves
        revealed = dataclasses.replace(ctx, pokemon_part=pp)
        cases = {"fainted": (fainted, 0), "alive": (dead, 0), "revealed": (revealed, 1)}
        for name, (c, side) in cases.items():
            d = (tt.board_tokens(c) - base).abs().amax(-1)                               # [B,3]
            assert d[:, side].min().item() > MOVES, f"the {name} count did not reach its side token"
            assert d[:, 1 - side].max().item() == 0.0 and d[:, 2].max().item() == 0.0, \
                f"the {name} count reached the other side / the field"
    assert {"alive_count", "fainted_count", "revealed_count"} <= set(SIDE_FACTS)


def test_a_fact_moves_the_refined_board_and_the_critic_pool_reads_three_board_rows(static_tower):
    fe = static_tower.policy.features_extractor
    obs = _rows(static_tower, 16)
    seen: Dict[str, Any] = {}
    h = fe.value_entity_pool.register_forward_hook(lambda _m, a, kw, out: seen.update(kw), with_kwargs=True)
    try:
        with torch.no_grad():
            fe(obs)
            a = fe.team_transformer.board_rows().clone()
            sl = _slices(fe)
            obs2 = dict(obs)
            o = obs["observation"].clone()
            o[:, sl["reactive.fainted"].start] += 1 / 6
            obs2["observation"] = o
            fe(obs2)
            b = fe.team_transformer.board_rows()
    finally:
        h.remove()
    assert a.shape[1:] == (3, D_MODEL)
    assert (a - b).abs().max().item() > MOVES, "a side count did not reach the refined board tokens"
    assert seen.get("global_row") is None and tuple(seen["board_rows"].shape[1:]) == (3, D_MODEL)


def _slices(fe: Any) -> Dict[str, slice]:
    from agents.observation.schema import build_schema
    return build_schema(fe.layout).slices()


# ------------------------------------------------------------------------ no board in the per-mon encoder
def test_no_board_fact_reaches_the_static_per_mon_encoder(static_tower):
    fe = static_tower.policy.features_extractor
    with torch.no_grad():
        ctx = fe.unpack(_rows(static_tower))
        g = torch.Generator().manual_seed(5)

        def noise(t: torch.Tensor) -> torch.Tensor:
            return torch.rand(t.shape, generator=g, dtype=t.dtype)
        board = dataclasses.replace(
            ctx, non_matchup_rest=noise(ctx.non_matchup_rest), turn_feature=noise(ctx.turn_feature),
            weather_feature=noise(ctx.weather_feature), fainted_feature=noise(ctx.fainted_feature),
            spikes_feature=noise(ctx.spikes_feature), screen_feature=noise(ctx.screen_feature))
        d = (fe.pokemon_encoder(ctx, fe.embeddings) - fe.pokemon_encoder(board, fe.embeddings)).abs().max().item()
    assert d == 0.0, f"a board scalar reached the static per-mon encoder ({d:.3g})"


# --------------------------------------------------------------------------------- the op content
def test_their_spikes_reach_their_mons_as_content_only_through_op_content(static_tower):
    obs = _rows(static_tower, 32)
    sl = _slices(static_tower.policy.features_extractor)
    hz = sl["global_env.hazards"].start + 1                                        # THEIR side's layers
    o0, o3 = obs["observation"].clone(), obs["observation"].clone()
    o0[:, hz], o3[:, hz] = 0.0, 1.0
    obs0, obs3 = dict(obs, observation=o0), dict(obs, observation=o3)
    live = _with_op_content(static_tower, amount=True)
    # The SIDE FACT itself: their Spikes is THEIR mons' entry chip (the `x` cell's column 0), never ours.
    a0, a3 = _op_content_args(live, obs0), _op_content_args(live, obs3)
    assert (a3[0][0][..., 0] - a0[0][0][..., 0]).abs().max().item() == 0.0, "THEIR Spikes became OUR entry chip"
    assert (a3[0][1][..., 0] - a0[0][1][..., 0]).abs().max().item() > MOVES
    # Under X5 (the only belief representation since the version break) OUR mons' `pursuit_p` (column 1: P(they
    # carry Pursuit), composed from the move belief) is a function of the BOARD — the hypothesis set's δ_θ reads
    # the board context — so any board fact, their Spikes included, legitimately moves it. That is a BELIEF
    # conditioned on the board, not a side mix-up of the hazard fact; drop its column from the projection so the
    # trunk-input read below isolates the hazard's own route.
    with torch.no_grad():
        live.op_content.amount_proj.weight[:, 1] = 0.0
    d = (_trunk_input(live, obs3) - _trunk_input(live, obs0)).abs().amax(-1)          # [B,12]
    assert d[:, TEAM_SIZE:].gt(MOVES).any(1).float().mean().item() >= 0.5, \
        "THEIR Spikes reached their mons on fewer than half the rows"
    assert d[:, :TEAM_SIZE].max().item() == 0.0, "THEIR Spikes reached OUR mons pre-trunk"
    # the CONTROL: with the op content at its zero init, nothing pre-trunk reads their Spikes on their mons
    dead = _with_op_content(static_tower)
    d0 = (_trunk_input(dead, obs3) - _trunk_input(dead, obs0)).abs().amax(-1)
    assert d0.max().item() == 0.0, "a route other than op_content carried their Spikes to a mon token"


def test_the_op_content_reaches_every_mon_on_both_sides(static_tower):
    """Bias 0, random weights: a mon's trunk input moves EXACTLY where its op cells are non-zero — the amounts
    (x / g) on all twelve, our active's outgoing cells on their six only."""
    obs = _rows(static_tower, 32)
    base = _trunk_input(_with_op_content(static_tower), obs)
    fe_a = _with_op_content(static_tower, amount=True)
    x, g, _d1 = _op_content_args(fe_a, obs)
    has = torch.cat([torch.cat([x[0], g[0]], -1), torch.cat([x[1], g[1]], -1)], 1).abs().amax(-1) > 0   # [B,12]
    assert has[:, :TEAM_SIZE].sum().item() >= 32 and has[:, TEAM_SIZE:].sum().item() >= 32, \
        "PRECONDITION: mons with non-zero x / g cells on both sides"
    d = (_trunk_input(fe_a, obs) - base).abs().amax(-1)
    assert d[has].min().item() > MOVES, "an x / g amount did not reach its mon's token"
    assert d[~has].max().item() == 0.0, "a mon with zero cells moved: the content is not per-mon"
    fe_o = _with_op_content(static_tower, outgoing=True)
    _x, _g, d1 = _op_content_args(fe_o, obs)
    has_o = d1.abs().amax(dim=(1, 3)) > 0                                                          # [B,6] their mons
    assert has_o.sum().item() >= 16, "PRECONDITION: their mons with non-zero outgoing cells"
    d_o = (_trunk_input(fe_o, obs) - base).abs().amax(-1)
    assert d_o[:, :TEAM_SIZE].max().item() == 0.0, "the outgoing cells reached OUR mons"
    assert d_o[:, TEAM_SIZE:][has_o].min().item() > MOVES, "our active's outgoing damage did not reach their mon"
    assert d_o[:, TEAM_SIZE:][~has_o].max().item() == 0.0


def test_op_content_refuses_a_missing_outgoing_cell():
    oc = OpContent(outgoing=True)
    z = torch.zeros(1, TEAM_SIZE, 4)
    with pytest.raises(ValueError, match="no d1 cells"):
        oc((z, z), (z, z), None)


# ------------------------------------------------------------------------------------------- edges
@pytest.mark.parametrize("seats", [BOARD_SEATS_LEGACY, BOARD_SEATS_STATIC], ids=["legacy", "static"])
def test_x_writes_to_the_own_side_seat_and_g_c4_to_the_field(seats):
    eb = EdgeBias("x,g,c4")
    g = torch.Generator().manual_seed(1)
    for lin in (eb.x_map, eb.g_map, eb.c4_map):
        with torch.no_grad():
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g))
    base = 2 * TEAM_SIZE + (3 if seats == BOARD_SEATS_STATIC else 1)
    n = base + 4
    B, H = 2, TRANSFORMER_N_HEADS

    def touched(cells: Dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            out = eb(torch.zeros(B, H, n, n), base, cells, board_seats=seats)
        return out.abs().sum((0, 1)) > 0                                                # [n,n]
    r = lambda *s: torch.rand(*s, generator=g) + 0.1                                       # noqa: E731
    s_our, s_opp, s_field = seats
    x = touched({"x": (r(B, TEAM_SIZE, 4), r(B, TEAM_SIZE, 4))})
    assert x[:TEAM_SIZE, s_our].all() and x[TEAM_SIZE:2 * TEAM_SIZE, s_opp].all()
    assert x[s_our, :TEAM_SIZE].all() and x[s_opp, TEAM_SIZE:2 * TEAM_SIZE].all()
    if seats == BOARD_SEATS_STATIC:
        assert not x[:TEAM_SIZE, s_opp].any() and not x[TEAM_SIZE:2 * TEAM_SIZE, s_our].any(), "x crossed sides"
        assert not x[:, s_field].any(), "x reached FIELD"
    assert int(x.sum()) == 2 * 2 * TEAM_SIZE
    gg = touched({"g": (r(B, TEAM_SIZE, 4), r(B, TEAM_SIZE, 4))})
    assert gg[:2 * TEAM_SIZE, s_field].all() and int(gg.sum()) == 2 * 2 * TEAM_SIZE
    c4 = touched({"c4": r(B, 4, 4)})
    assert c4[base:base + 4, s_field].all() and c4[s_field, base:base + 4].all() and int(c4.sum()) == 8


# ---------------------------------------------------------------------------------------- readouts
def test_the_tower_drops_the_board_bypass(static_tower, legacy):
    fe_s, fe_l = static_tower.policy.features_extractor, legacy.policy.features_extractor
    pi_s, _ = compute_projection_widths(fe_s.layout, token_encoding="static")
    pi_l, _ = compute_projection_widths(fe_l.layout)
    nmr = fe_l.team_transformer._non_matchup_rest_dim
    assert pi_l - pi_s == nmr == fe_l.projection_input_dim - fe_s.projection_input_dim
    for fe, reads in ((fe_s, False), (fe_l, True)):
        ctx = fe.unpack(_rows(static_tower, 4))
        nan = dataclasses.replace(ctx, non_matchup_rest=torch.full_like(ctx.non_matchup_rest, float("nan")))
        z = torch.zeros(4, D_MODEL)
        with torch.no_grad():
            pi, _ = fe.assembler(z, z, z, z, nan)
        assert bool(torch.isnan(pi).any()) is reads, "the tower's board bypass is " + ("missing" if reads else "live")


def test_the_trunk_state_query_attends_over_the_three_board_tokens(static_trunk):
    fe = static_trunk.policy.features_extractor
    seen: List[Any] = []
    h = fe.policy_query.register_forward_pre_hook(lambda _m, args: seen.append(args))
    try:
        with torch.no_grad():
            fe(_rows(static_trunk, 8))
    finally:
        h.remove()
    tokens, pad = seen[0][0], seen[0][1]
    tt = fe.team_transformer
    board = tt.board_rows()
    assert torch.equal(tokens[:, 2 * TEAM_SIZE:tt._total_tokens], board), "the board keys are not the board tokens"
    assert not pad[:, 2 * TEAM_SIZE:tt._total_tokens].any(), "a board token is masked"
    # the trunk's seat order: the entity / event seats follow the board (the per-key log-presence aligns)
    assert tokens.shape[1] >= tt._total_tokens + 4


def test_the_board_tokens_move_the_trunk_policy_context(static_trunk):
    fe = static_trunk.policy.features_extractor
    obs = _rows(static_trunk, 8)
    sl = _slices(fe)
    o = obs["observation"].clone()
    o[:, sl["global_env.clock"]] = 1.0 - o[:, sl["global_env.clock"]]
    with torch.no_grad():
        a, _ = fe.forward_internal(obs)
        b, _ = fe.forward_internal(dict(obs, observation=o))
    assert (a - b).abs().max().item() > MOVES, "the FIELD clock did not reach the trunk policy context"


# ---------------------------------------------------------------------------------------- versioning
def test_a_stage_one_static_config_is_refused_and_a_current_record_stamps_through():
    """v140: a pre-v140 `static` record is the stage-1 layout (one global token) — no home in this code. Since
    the X5 version break raised MIGRATION_FLOOR to 144, EVERY pre-v144 config (stage-1 static or legacy) is
    refused at the floor before the v140 branch could run (that branch is unreachable, left in place); a record
    at the current version carries its encoding through verbatim."""
    from agents.model.critic_mode_test import _fresh_current_config
    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersionError
    from agents.model.model_version.migrations import MIGRATION_FLOOR, _migrate_config
    assert MODEL_CONFIG_VERSION >= 140 and MIGRATION_FLOOR > 139
    for enc in ("static", "legacy"):
        with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
            _migrate_config(_fresh_current_config(config_version=139, token_encoding=enc))
    for enc in ("static", "legacy"):
        out = _migrate_config(_fresh_current_config(token_encoding=enc))
        assert out["token_encoding"] == enc and out["config_version"] == MODEL_CONFIG_VERSION
