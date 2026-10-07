"""gen3_speed_physics_v1 (v143, `--speed-physics`; architecture audit F7b) inside a REAL production-config extractor.

What must hold, each failing on revert:

  * **`off` runs nothing new** — no buffer, no stash, the same state_dict as `on` (no parameters), and the speed
    rule is never called (a raising stub on `damage_op_speed.p_first_same_priority` lets `off` through);
  * **`on` replaces the logistic at the op's sites** — every outspeed read differs from `off` and is the ONE rule;
  * on a REAL observation edited to a constructed board: their paralysis raises P(we first), our +2 stage raises
    it, a Choice Band on our active changes nothing, a mixture exactly at our speed reads ½ (the coin flip), the
    learned spread belief is not read, and Quick Claw is OFF in gen3ou (and priced when a format allows it);
  * the speed mixture is the FULL Smogon spreads mixture (gen3_speed_mixture_v1): Blissey's max-Speed sets keep
    their mass (the Gaussian's 2026-10-07 certain-and-wrong row);
  * the version machinery (migration default, `check_compatible`, snapshot kwargs) and the dependency refusal.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
from typing import Any, Dict

import numpy as np
import pytest
import torch

from agents.model.damage_op_layout import _COND_PAR_IDX
from agents.model.damage_tables import CHOICE_BAND_ITEM_NUM, QUICK_CLAW_ITEM_NUM
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.constants import POKEMON_CONDITION_OFFSET, TEAM_SIZE
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.paths import repo_path, src_path


def _prod_kwargs(**over: Any) -> Dict[str, Any]:
    from agents.model.damage_tables import sanitize_historical_move_floor
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {k: v for k, v in cfg.items() if k in sig}
    sanitize_historical_move_floor(kw)
    kw.update(over)
    return kw


def _build(**over: Any) -> Gen3FeaturesExtractor:
    import gymnasium as gym
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(0)
    return Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **_prod_kwargs(**over)).eval()


@pytest.fixture(scope="module")
def obs() -> torch.Tensor:
    return torch.as_tensor(np.load(src_path("agents", "model", "compile_parity_obs.npz"))["obs"])


@pytest.fixture(scope="module")
def fe_on() -> Gen3FeaturesExtractor:
    return _build(speed_physics="on")


def _p_out_preGain(fe: Gen3FeaturesExtractor, obs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        fe({"observation": obs})
    op = fe.damage_op
    return op.tensors_from_block(op.last_raw_block).out_p_outspeed[:, 0].clone()


# ------------------------------------------------------------------------------------------------ off
def test_off_builds_nothing_and_never_calls_the_rule(obs: torch.Tensor, monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.model.damage_op_speed as dos

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("the speed rule ran under --speed-physics off")
    monkeypatch.setattr(dos, "p_first_same_priority", boom)
    fe = _build()
    assert fe.speed_physics == "off" and fe.damage_op.speed_physics is False
    assert not hasattr(fe.damage_op, "SPECIES_QC_PRIOR")
    assert not hasattr(fe.damage_op, "SPEED_MIX")
    with torch.no_grad():
        fe({"observation": obs[:8]})                      # off: the stub is never reached
    assert fe.damage_op.stash.speed_fast_pair is None and fe.damage_op.stash.item_qc_prob is None
    on = _build(speed_physics="on")
    with pytest.raises(AssertionError, match="speed rule ran"):
        with torch.no_grad():
            on({"observation": obs[:8]})


def test_on_adds_no_parameter(fe_on: Gen3FeaturesExtractor) -> None:
    off = _build()
    a, b = off.state_dict(), fe_on.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_off_is_the_fixed_logistic(obs: torch.Tensor) -> None:
    """`off` P(outspeed) for our active is sigmoid(Δspeed / 15) — the production read, untouched."""
    from agents.model.damage_op_layout import _DMG_SPEED_SCALE
    assert _DMG_SPEED_SCALE == 15.0
    p = _p_out_preGain(_build(), obs)
    assert ((p > 0) & (p < 1)).any()


# ------------------------------------------------------------------------------------------------- on
def test_on_replaces_the_logistic_and_stays_a_probability(obs: torch.Tensor, fe_on: Gen3FeaturesExtractor) -> None:
    off = _p_out_preGain(_build(), obs)
    on = _p_out_preGain(fe_on, obs)
    assert torch.isfinite(on).all() and (on >= 0).all() and (on <= 1).all()
    assert (on - off).abs().max() > 0.05
    with torch.no_grad():
        out = fe_on({"observation": obs})
    assert all(torch.isfinite(t).all() for t in out)


# --------------------------------------------------------------------------- edited real observations
def _ctx(fe: Gen3FeaturesExtractor, obs: torch.Tensor) -> Any:
    with torch.no_grad():
        return fe.unpack({"observation": obs})


def _p_active(fe: Gen3FeaturesExtractor, ctx: Any, spread_belief: Any = None) -> torch.Tensor:
    op = fe.damage_op
    ar = torch.arange(ctx.batch_size)
    with torch.no_grad():
        return op._p_first_vs_opp_active(ctx, spread_belief, op._our_speeds_exact(ctx)[ar, ctx.our_active_idx],
                                         op._our_quick_claw(ctx)[ar, ctx.our_active_idx])


def _with(ctx: Any, **edits: torch.Tensor) -> Any:
    return dataclasses.replace(ctx, **edits)


def _rows_with_both_actives(ctx: Any) -> torch.Tensor:
    ar = torch.arange(ctx.batch_size)
    return (ctx.hp_and_active[ar, ctx.our_active_idx, 0] > 0) & \
        (ctx.hp_and_active[ar, TEAM_SIZE + ctx.opp_active_local, 0] > 0)


def test_their_paralysis_raises_p_first(obs: torch.Tensor, fe_on: Gen3FeaturesExtractor) -> None:
    ctx = _ctx(fe_on, obs)
    ar = torch.arange(ctx.batch_size)
    pp = ctx.pokemon_part.clone()
    col = POKEMON_CONDITION_OFFSET + _COND_PAR_IDX
    pp[ar, TEAM_SIZE + ctx.opp_active_local, col] = 0.0
    pp[ar, ctx.our_active_idx, col] = 0.0
    base = _p_active(fe_on, _with(ctx, pokemon_part=pp))
    pp2 = pp.clone()
    pp2[ar, TEAM_SIZE + ctx.opp_active_local, col] = 1.0
    para = _p_active(fe_on, _with(ctx, pokemon_part=pp2))
    keep = _rows_with_both_actives(ctx) & (base > 0.01) & (base < 0.99)
    assert keep.sum() >= 4
    assert (para[keep] > base[keep]).all()


def test_our_plus_two_stage_raises_p_first(obs: torch.Tensor, fe_on: Gen3FeaturesExtractor) -> None:
    ctx = _ctx(fe_on, obs)
    our = ctx.our_ctx_raw.clone()
    our[:, 8], our[:, 9] = 0.0, 0.0
    base = _p_active(fe_on, _with(ctx, our_ctx_raw=our))
    our2 = our.clone()
    our2[:, 8] = 2.0 / 6.0
    up = _p_active(fe_on, _with(ctx, our_ctx_raw=our2))
    keep = _rows_with_both_actives(ctx) & (base > 0.01) & (base < 0.99)
    assert keep.sum() >= 4
    assert (up[keep] > base[keep]).all()


def test_choice_band_does_not_touch_speed(obs: torch.Tensor, fe_on: Gen3FeaturesExtractor) -> None:
    """`data/items.ts` choiceband has no ``onModifySpe``."""
    ctx = _ctx(fe_on, obs)
    ar = torch.arange(ctx.batch_size)
    items = ctx.item_ids.clone()
    items[ar, ctx.our_active_idx] = CHOICE_BAND_ITEM_NUM
    assert torch.equal(_p_active(fe_on, ctx), _p_active(fe_on, _with(ctx, item_ids=items)))


def test_a_mixture_at_our_exact_speed_is_a_coin_flip(obs: torch.Tensor) -> None:
    """Neither side boosted nor paralysed, their species' mixture set to a POINT at our active's exact speed: ½."""
    fe = _build(speed_physics="on")
    op = fe.damage_op
    ctx = _ctx(fe, obs)
    ar = torch.arange(ctx.batch_size)
    col = POKEMON_CONDITION_OFFSET + _COND_PAR_IDX
    pp = ctx.pokemon_part.clone()
    pp[:, :, col] = 0.0
    our, opp = ctx.our_ctx_raw.clone(), ctx.opp_ctx_raw.clone()
    our[:, 8:10], opp[:, 8:10] = 0.0, 0.0
    ctx = _with(ctx, pokemon_part=pp, our_ctx_raw=our, opp_ctx_raw=opp)
    ours = op._our_speeds_exact(ctx)[ar, ctx.our_active_idx]
    their = ctx.species_ids[ar, TEAM_SIZE + ctx.opp_active_local]
    keep = their.unique(return_counts=True)
    keep = torch.isin(their, keep[0][keep[1] == 1])                 # one species = one point: unique rows only
    assert keep.sum() >= 4
    for b in torch.nonzero(keep).flatten().tolist():                # this test's own op: the mixture edited
        op.SPEED_MIX[their[b]] = 0.0
        op.SPEED_MIX[their[b], int(ours[b].item())] = 1.0
    p = _p_active(fe, ctx)
    assert torch.equal(p[keep], torch.full_like(p[keep], 0.5))


def test_the_learned_spread_belief_is_not_read(obs: torch.Tensor, fe_on: Gen3FeaturesExtractor) -> None:
    """gen3_speed_mixture_v1: their speed is the Smogon mixture alone — a believed speed changes nothing (the
    Gaussian it replaced centred on it)."""
    ctx = _ctx(fe_on, obs)
    sb = torch.full((ctx.batch_size, TEAM_SIZE, 5), 999.0)
    assert torch.equal(_p_active(fe_on, ctx), _p_active(fe_on, ctx, sb))


def test_the_speed_mix_is_the_full_smogon_spreads_mixture() -> None:
    """Every row is a distribution; Blissey — the 2026-10-07 certain-and-wrong row (a Timid 252-Speed Blissey,
    229, against our 214, read P = 1.0 by the Gaussian at 148 ± 5) — keeps its max-Speed sets' mass, so our 214
    is NOT certain to move first; and the mixture's support is lumpy (most mass at the uninvested speeds)."""
    from agents import gen3_data
    from agents.model.belief_tables import SPEED_LATTICE, build_species_speed_mix
    n = int(_build(speed_physics="on").damage_op.BASE_STATS.shape[0])
    mix = build_species_speed_mix(n)
    assert mix.shape == (n, SPEED_LATTICE)
    assert torch.allclose(mix.sum(-1), torch.ones(n), atol=1e-5)
    bl = mix[gen3_data.species.get("blissey").num]
    assert bl[229].item() > 0.0                                     # Timid 252 Speed: (110+31+63+5)·1.1
    above = bl[215:].sum().item()
    assert 0.0 < above < 0.05                                       # rare but real: P(we first) < 1
    assert bl[:214].sum().item() > 0.95


def test_quick_claw_is_off_in_gen3ou_and_priced_where_a_format_allows_it(obs: torch.Tensor) -> None:
    fe = _build(speed_physics="on")
    op = fe.damage_op
    assert op.quick_claw_live is False                          # BANNED in gen3ou (owner + Showdown master)
    ctx = _ctx(fe, obs)
    ar = torch.arange(ctx.batch_size)
    items = ctx.item_ids.clone()
    items[ar, ctx.our_active_idx] = QUICK_CLAW_ITEM_NUM
    qc = _with(ctx, item_ids=items)
    base = _p_active(fe, ctx)
    assert torch.equal(_p_active(fe, qc), base)                 # a banned item never fires
    op.quick_claw_live = True                                   # a format that allows it (this test's own op)
    from agents.model.damage_tables import build_species_qc_prior
    op.register_buffer("SPECIES_QC_PRIOR", torch.zeros_like(build_species_qc_prior(op.BASE_STATS.shape[0])),
                       persistent=False)
    items2 = items.clone()
    items2[:, TEAM_SIZE:] = 1                                    # every opp item revealed as a non-Quick-Claw
    p0 = _p_active(fe, _with(ctx, item_ids=torch.where(items2 == QUICK_CLAW_ITEM_NUM, 1, items2)))
    p1 = _p_active(fe, _with(ctx, item_ids=items2))
    assert torch.allclose(p1, 0.8 * p0 + 0.2, atol=1e-6)


# ------------------------------------------------------------------------------- versions / refusals
def test_requires_the_op_and_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="requires damage_op"):
        _build(speed_physics="on", damage_op=False, **{k: False for k in (
            "damage_outgoing", "damage_matrices_outgoing", "damage_matrices_incoming", "op_believed_lean")})
    with pytest.raises(ValueError, match="speed_physics must be one of"):
        _build(speed_physics="yes")


def test_version_machinery() -> None:
    from agents.model.model_version import ModelVersion, ModelVersionError
    from agents.model.model_version.migrations import MIGRATION_FLOOR, _migrate_config
    # the v143 branch (a pre-v143 config defaults to `off`) is unreachable since the X5 version break raised
    # MIGRATION_FLOOR to 144: such a config is refused at the floor; a RECORDED mode migrates verbatim
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 142})
    for mode in ("off", "on"):
        m = _migrate_config({"config_version": MIGRATION_FLOOR, "speed_physics": mode})
        assert m["speed_physics"] == mode and m["config_version"] >= 143
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    a = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    assert a.speed_physics == "off"
    on = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {"speed_physics": "on"}})
    assert on.speed_physics == "on"
    b = dataclasses.replace(a, speed_physics="on")
    with pytest.raises(ModelVersionError, match="speed_physics mismatch"):
        b.check_compatible(a)
    from agents.model.snapshot import current_model_version
    sig = inspect.signature(current_model_version).parameters
    assert sig["speed_physics"].default == "off"
