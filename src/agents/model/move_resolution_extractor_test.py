"""gen3_move_resolution_v1 (v141, `--move-resolution`) — the family inside a REAL extractor and a REAL policy.

What must hold, each failing on revert:

  * **`off` builds nothing** — no module, no op seam, no state_dict key, the production cell widths.
  * **`on`, built the way training builds it** (`MaskablePPO` → `_build`): the seven per-action blocks are
    RETIRED (no state_dict key, no optimizer slot), the family is in the optimizer, its projections are exactly
    zero at init, and EVERY OTHER parameter's initial bytes equal the `off` build's (the one-lever property).
  * **`gather_ops` reads the right observation fields** — a real observation edited to a constructed board
    (their Substitute up, our Confuse Ray) reaches the fact.
  * the dependency refusals and the version machinery (migration default, `check_compatible`, snapshot kwargs).
"""
from __future__ import annotations

import dataclasses
import inspect
import json
from typing import Any, Dict

import numpy as np
import pytest
import torch

from agents import gen3_data
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.move_resolution_rules import MOVE_RESOLUTION_MOVE_IDX as MI
from agents.observation.gen3_effects import VOLATILE_SLOTS
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.paths import repo_path, src_path

SEVEN = ("intent_move_cell", "intent_threshold_move", "intent_conditional", "pair_outcome_move", "switch_branch",
         "pair_outcome_switch", "conditional_threat")


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


# ------------------------------------------------------------------------------------------- off
def test_off_builds_nothing_and_sets_no_op_seam():
    fe = _build()
    assert fe.move_resolution == "off" and fe.move_resolution_cell is None
    assert fe.damage_op.stash_species_post is False
    assert not any("move_resolution" in k for k in fe.state_dict())
    assert all(getattr(fe, n) is not None for n in SEVEN)        # production keeps the seven blocks
    assert fe.retire_superseded_action_cells() is False
    assert all(getattr(fe, n) is not None for n in SEVEN)


def test_off_forward_leaves_no_species_posterior_stash(obs):
    fe = _build()
    with torch.no_grad():
        fe({"observation": obs[:4]})
    assert fe.damage_op.stash.opp_species_post is None


# -------------------------------------------------------------------------------------------- on
def test_on_widens_both_cells_and_runs_finite(obs):
    fe = _build(move_resolution="on")
    with torch.no_grad():
        fe({"observation": obs})
    pi = fe.last_pointer_inputs
    assert pi.move_cells.shape[2] == fe.pointer_move_cell_dim
    assert pi.switch_cells.shape[2] == fe.pointer_switch_cell_dim
    assert torch.isfinite(pi.move_cells).all() and torch.isfinite(pi.switch_cells).all()


def _policy(move_resolution: str):
    from gymnasium import spaces
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.model.identity_init_test import _Env
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    enc = Gen3ObservationEncoder(load_mappings())
    kw = {**enc.get_features_extractor_kwargs(), **_prod_kwargs(move_resolution=move_resolution)}
    assert isinstance(spaces, object)
    torch.manual_seed(0)
    return MaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env(enc.dimension)]),
        n_steps=16, batch_size=16, n_epochs=1, device="cpu",
        policy_kwargs={"features_extractor_class": Gen3FeaturesExtractor,
                       "features_extractor_kwargs": kw, "net_arch": dict(pi=[64], vf=[64])})


@pytest.fixture(scope="module")
def policies():
    return _policy("off").policy, _policy("on").policy


def test_on_retires_the_seven_blocks_on_a_real_policy(policies):
    _off, on = policies
    fe = on.features_extractor
    assert all(getattr(fe, n) is None for n in SEVEN)
    keys = on.state_dict().keys()
    assert not any(k.split(".")[1] in SEVEN for k in keys if k.startswith("features_extractor."))
    assert any(k.startswith("features_extractor.move_resolution_cell.") for k in keys)
    opt = {id(p) for g in on.optimizer.param_groups for p in g["params"]}
    assert all(id(p) in opt for p in fe.move_resolution_cell.parameters())
    from agents.model.damage_op_layout import _PTR_MOVE_CELL, _PTR_SWITCH_CELL_IN
    assert fe.pointer_move_cell_dim == _PTR_MOVE_CELL + 38
    assert fe.pointer_switch_cell_dim == _PTR_SWITCH_CELL_IN + 18


def test_on_is_identity_at_init(policies):
    _off, on = policies
    cell = on.features_extractor.move_resolution_cell
    for p in cell.parameters():
        assert float(p.abs().max()) == 0.0


def test_every_other_parameter_starts_bit_identical_to_production(policies):
    """The ONE-lever property: the screen arm differs from production by the family (and the pointer head that
    reads the wider cells), never by an init draw elsewhere."""
    off, on = policies
    a, b = off.state_dict(), on.state_dict()
    shared = [k for k in b if k in a and not k.startswith("pointer_head.")]
    assert len(shared) > 100
    moved = [k for k in shared if not torch.equal(a[k], b[k])]
    assert not moved, f"the family moved other parameters' initial bytes: {moved[:5]}"
    only_off = {k.split(".")[1] for k in a if k not in b and k.startswith("features_extractor.")}
    assert only_off == set(SEVEN)


# -------------------------------------------------------------------- gather on an edited real board
_POK = 122
_OPP0 = 732
_CTX_OPP = 1524
_REACTIVE = 1604


def _first_live_row(obs: torch.Tensor, fe: Gen3FeaturesExtractor) -> int:
    ctx = fe.unpack({"observation": obs})
    live = (ctx.hp_and_active[torch.arange(obs.shape[0]), ctx.our_active_idx, 0] > 0) & \
        ctx.hp_and_active[:, 6:12, -1].any(dim=1) & (ctx.our_active_req_move_ids[:, 0] > 0)
    return int(torch.nonzero(live)[0, 0])


def _facts(fe: Gen3FeaturesExtractor, x: torch.Tensor) -> torch.Tensor:
    cell = fe.move_resolution_cell
    cap: Dict[str, torch.Tensor] = {}
    orig = cell.forward

    def hook(ops):
        cap["m"] = cell.raw(ops)[0]
        return orig(ops)
    cell.forward = hook
    try:
        with torch.no_grad():
            fe({"observation": x})
    finally:
        cell.forward = orig
    return cap["m"]


def test_gather_reads_their_substitute_and_our_request_moves(obs):
    fe = _build(move_resolution="on")
    b = _first_live_row(obs, fe)
    x = obs[b:b + 1].clone()
    md = gen3_data.moves.get("confuseray")
    x[0, _REACTIVE + 5] = float(md.num)                                   # our request slot 0: Confuse Ray
    x[0, _REACTIVE + 5 + 8] = 1.0                                         # legal
    clean = _facts(fe, x)[0, 0, MI["p_lands_stay"]].item()
    x[0, _CTX_OPP + 14 + list(VOLATILE_SLOTS).index("substitute")] = 1.0   # their Substitute goes up
    x[0, _CTX_OPP + 14 + list(VOLATILE_SLOTS).index("confusion")] = 0.0
    subbed = _facts(fe, x)[0, 0, MI["p_lands_stay"]].item()
    assert clean > 0.5 and subbed == 0.0


def test_gather_reads_our_hp_for_substitute(obs):
    fe = _build(move_resolution="on")
    b = _first_live_row(obs, fe)
    x = obs[b:b + 1].clone()
    ctx = fe.unpack({"observation": x})
    our = int(ctx.our_active_idx[0])
    x[0, _REACTIVE + 5] = float(gen3_data.moves.get("substitute").num)
    x[0, list(VOLATILE_SLOTS).index("substitute") + 1464 + 14] = 0.0
    x[0, our * _POK + 67] = 0.2
    assert _facts(fe, x)[0, 0, MI["p_lands_stay"]].item() == 0.0
    x[0, our * _POK + 67] = 0.8
    assert _facts(fe, x)[0, 0, MI["p_lands_stay"]].item() == pytest.approx(1.0)


# --------------------------------------------------------------------------------- refusals
def test_on_requires_the_intent_head():
    with pytest.raises(ValueError, match="opp_intent"):
        _build(move_resolution="on", opp_intent=False)


def test_on_builds_under_fixed_mass():
    """gen3_move_resolution_x5_v1: the family no longer refuses X5 — it reads the flat pointer (the fixed_mass forward,
    OTHER's pricing and the one-lever property are `move_resolution_x5_test.py`)."""
    fe = _build(move_resolution="on", belief_tokens="fixed_mass")
    assert fe.move_resolution_cell is not None and fe.flat_intent_head is not None


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="move_resolution"):
        _build(move_resolution="yes")


# ------------------------------------------------------------------------------ version machinery
def test_a_pre_v141_config_migrates_to_off():
    from agents.model.model_version import _migrate_config
    out = _migrate_config({"config_version": 139})
    assert out["move_resolution"] == "off" and out["config_version"] >= 141


def test_check_compatible_gates_the_flag():
    from agents.model.model_version import ModelVersion, ModelVersionError
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    b = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    a = dataclasses.replace(b, move_resolution="on")
    with pytest.raises(ModelVersionError, match="move_resolution"):
        a.check_compatible(b)


def test_the_flag_round_trips_through_the_snapshot_kwargs():
    from agents.model.snapshot import current_model_version
    assert current_model_version(load_mappings(), move_resolution="on").move_resolution == "on"
    assert current_model_version(load_mappings()).move_resolution == "off"


def test_an_unrevealed_ability_is_read_from_the_known_flag_not_the_id(obs):
    """The observation writes an UNREVEALED opponent's most likely ability into id1 with known = 0. The family must
    price Toxic on an unrevealed Snorlax through the Smogon prior (Immunity ~0.86), never as a certain Immunity —
    and so must the production op (gen3_op_ability_known_v1, 2026-10-07: its `_status_landing` read `id > 0` as
    revealed and said 0; it now reads the `known` flag). With the ability REVEALED it is exactly 0."""
    from agents.observation.types import TypeEncoder
    fe = _build(move_resolution="on")
    b = _first_live_row(obs, fe)
    x = obs[b:b + 1].clone()
    ctx = fe.unpack({"observation": x})
    oa = int(ctx.opp_active_local[0])
    base = _OPP0 + oa * _POK
    x[0, base + 0] = float(gen3_data.species.get("snorlax").num)
    x[0, base + 10] = float(TypeEncoder.TYPE_TO_IDX["NORMAL"])
    x[0, base + 11] = 0.0
    x[0, base + 12] = float(gen3_data.abilities.get("immunity").num)
    x[0, base + 13] = float(gen3_data.abilities.get("thickfat").num)
    x[0, base + 14] = 0.86
    x[0, base + 15] = 0.0                                                   # known = 0: unrevealed
    x[0, base + 16:base + 23] = 0.0                                         # no status
    for v in ("substitute", "taunt"):
        x[0, _CTX_OPP + 14 + list(VOLATILE_SLOTS).index(v)] = 0.0
    x[0, 1584 + 12 + 5] = 0.0                                               # no Safeguard on their side
    x[0, _REACTIVE + 5] = float(gen3_data.moves.get("toxic").num)
    fam = _facts(fe, x)[0, 0, MI["p_lands_stay"]].item()
    assert 0.0 < fam < 0.2
    op_says = fe.damage_op._status_landing(fe.unpack({"observation": x}))[0, 0].item()
    p_thickfat = float(gen3_data.priors.ability("snorlax")["thickfat"])
    acc = float(fe.damage_op.MOVE_ACCURACY[gen3_data.moves.get("toxic").num])
    assert op_says == pytest.approx(acc * p_thickfat, abs=1e-5)             # ≈ 0.85 · 0.14 (was a certain 0)
    assert fe.damage_op._status_landing(fe.unpack({"observation": x}))[0, 4].item() == 0.0   # known = 0
    x[0, base + 15] = 1.0                                                   # REVEALED Immunity
    assert fe.damage_op._status_landing(fe.unpack({"observation": x}))[0, 0].item() == 0.0
    assert fe.damage_op._status_landing(fe.unpack({"observation": x}))[0, 4].item() == 1.0
