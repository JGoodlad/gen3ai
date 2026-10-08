"""The MON-TIED ``out_gain`` (config v145, ``gen3_mon_tied_gain_v1``; owner 2026-10-07: "fix those non-equivariant
knobs").

The op's learned gain is ONE scalar per (block region, channel) with NO position in its key: the X5 version break's
part 4 tied the request slots / move seats; v145 ties our TEAM SLOTS and their mons as well (the incoming per-mon
rows 72 -> 12, the Choice-Band tail 12 -> 2, the render matrices' per-mon cells), with no lead-mon special case.
Each test FAILS ON REVERT:

* the counts — production 29 / the render arm 92, counted from the layout constants, and the extractor holds
  exactly 70 parameters fewer than the per-team-slot (v144) keying builds;
* permuting OUR TEAM SLOTS (the incoming rows + the CB tail) of a real raw block permutes the gained block exactly,
  under planted non-uniform gains; the same for their mons / our mons in the render matrices;
* every team-slot copy of a channel accumulates into ONE parameter;
* the version gate — the live stamps, and a v144 config REFUSED naming the tie and the pinned commit.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from agents.model.damage_op_layout import (_DMG_CB_PER_MON, _DMG_IMX_CELL, _DMG_IMX_HEADER, _DMG_OMX_CELL,
                                           _DMG_OUT_N_MOVES, _DMG_OUT_PER_MOVE, _DMG_PER_MON, _N_OUT_SECONDARY)
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.constants import TEAM_SIZE
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.paths import repo_path, src_path

K = 6                                       # the render arm's incoming-matrix K


def _prod_kwargs() -> Dict[str, Any]:
    from agents.model.damage_tables import sanitize_historical_move_floor
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {k: v for k, v in cfg.items() if k in sig}
    sanitize_historical_move_floor(kw)
    return kw


def _build() -> Gen3FeaturesExtractor:
    import gymnasium as gym
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(0)
    return Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **_prod_kwargs()).eval()


def _render_op() -> Any:
    from agents.model.damage_op import DamageOperator
    return DamageOperator(Gen3ObservationEncoder(load_mappings()).get_layout(), outgoing=True, topk_k=K,
                          matrices_outgoing=True, matrices_incoming=True)


def _v144_keys(*, outgoing: bool, matrices_outgoing: bool, matrices_incoming_k: int) -> List[tuple]:
    """The v144 keying (per-team-slot / per-mon keys kept) — the revert, for the parameter-count delta."""
    from agents.model.damage_op_layout import out_gain_channel_keys
    keys = out_gain_channel_keys(outgoing=outgoing, matrices_outgoing=matrices_outgoing,
                                 matrices_incoming_k=matrices_incoming_k)
    out, seen = [], 0
    for pos, k in enumerate(keys):
        if k[0] == "incoming_row":
            out.append((k[0], pos // _DMG_PER_MON, k[1]))
        elif k[0] in ("cb_high", "cb_pko"):
            out.append((k[0], seen % TEAM_SIZE))
            seen += 1
        else:
            out.append(k)
    return out


@pytest.fixture(scope="module")
def obs() -> torch.Tensor:
    return torch.as_tensor(np.load(src_path("agents", "model", "compile_parity_obs.npz"))["obs"])


@pytest.fixture(scope="module")
def fe() -> Gen3FeaturesExtractor:
    return _build()


def _positions(op: Any) -> Any:
    return op.tensors_from_block(torch.arange(op.out_dim, dtype=torch.float32)[None])


def _team_slot_groups(op: Any) -> List[torch.Tensor]:
    """Every OUR-TEAM-SLOT-indexed region of the block as ``[6, F]`` flat positions (slot-major)."""
    t = _positions(op)
    return [t.incoming_rows[0].long(), t.cb_high[0].long()[:, None], t.cb_pko[0].long()[:, None]]


def _gain_index(op: Any, pos: int) -> int:
    return int(op._out_gain_tie[:, pos].argmax())


def _plant_random_gains(op: Any, seed: int = 11) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        op.out_gain.copy_(0.5 + 1.5 * torch.rand(op.out_gain.shape, generator=g))


def _raw_block(fe: Gen3FeaturesExtractor, obs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        fe({"observation": obs})
    raw = fe.damage_op.last_raw_block
    assert raw is not None and bool((raw != 0).any())
    return raw.clone()


# ------------------------------------------------------------------------------------------------ counts
def test_the_production_gain_is_one_scalar_per_channel_counted_from_the_layout(fe: Gen3FeaturesExtractor) -> None:
    op = fe.damage_op
    assert op.drop_renders and op.outgoing, "PRECONDITION: the production op (lean forward, outgoing on)"
    expected = (_DMG_PER_MON + _DMG_CB_PER_MON + 1                      # incoming row, CB high / pko, p_cb
                + _DMG_OUT_PER_MOVE + 1 + _N_OUT_SECONDARY + 2)         # out_move, outspeed, secondaries, status
    assert op.out_gain.numel() == expected == 29
    assert len(op.out_gain_keys) == len(set(op.out_gain_keys)) == 29
    assert all(len(k) <= 2 for k in op.out_gain_keys), "a key carries a position beside its channel"


def test_the_render_arm_ties_the_per_mon_cells() -> None:
    op = _render_op()
    assert op.matrices_incoming_k == K
    assert op.out_gain.numel() == 29 + (_DMG_OMX_CELL + 1) + (_DMG_IMX_HEADER + _DMG_IMX_CELL) == 92


def test_the_extractor_holds_exactly_the_tied_away_parameters_fewer(fe: Gen3FeaturesExtractor,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """The revert (the v144 per-team-slot keying) builds 70 more parameters — 60 incoming-row + 10 CB-tail gains —
    and every other parameter's shape is the same."""
    import agents.model.damage_op as DO
    monkeypatch.setattr(DO, "out_gain_channel_keys", _v144_keys)
    old = _build()
    monkeypatch.undo()
    assert old.damage_op.out_gain.numel() == 99
    n_new = sum(p.numel() for p in fe.parameters())
    n_old = sum(p.numel() for p in old.parameters())
    assert n_old - n_new == (TEAM_SIZE - 1) * (_DMG_PER_MON + _DMG_CB_PER_MON) == 70
    new_shapes = {n: tuple(p.shape) for n, p in fe.named_parameters() if not n.endswith("out_gain")}
    old_shapes = {n: tuple(p.shape) for n, p in old.named_parameters() if not n.endswith("out_gain")}
    assert new_shapes == old_shapes


def test_the_tied_init_is_the_per_slot_init(fe: Gen3FeaturesExtractor) -> None:
    full = fe.damage_op.expanded_out_gain().detach()
    rows, cb_high, cb_pko = _team_slot_groups(fe.damage_op)
    assert torch.equal(full[rows], full[rows[:1]].expand_as(rows))
    assert torch.equal(full[rows[0]], torch.tensor([1 / 1.5, 1 / 1.5, 1 / 3.0, 1.0, 1.0,
                                                    1 / 1.5, 1 / 1.5, 1 / 3.0, 1.0, 1.0, 1.0, 1.0]))
    assert bool((full[cb_high] == 1 / 1.5).all()) and bool((full[cb_pko] == 1.0).all())


# ------------------------------------------------------------------------------------------- equivariance
def test_permuting_our_team_slots_of_a_real_raw_block_permutes_the_gained_block(
        fe: Gen3FeaturesExtractor, obs: torch.Tensor) -> None:
    op = fe.damage_op
    raw = _raw_block(fe, obs)
    saved = op.out_gain.detach().clone()
    _plant_random_gains(op)
    try:
        for sigma in ([5, 4, 3, 2, 1, 0], [1, 2, 3, 4, 5, 0], [0, 2, 1, 3, 5, 4], [3, 0, 1, 2, 4, 5]):
            perm = torch.arange(op.out_dim)
            for grp in _team_slot_groups(op):
                perm[grp.reshape(-1)] = grp[sigma].reshape(-1)
            assert not torch.equal(perm, torch.arange(op.out_dim))
            with torch.no_grad():
                assert torch.equal(op.apply_out_gain(raw[:, perm]), op.apply_out_gain(raw)[:, perm]), sigma
    finally:
        with torch.no_grad():
            op.out_gain.copy_(saved)


def test_permuting_mons_in_the_render_matrices_permutes_the_gained_block() -> None:
    """The outgoing matrix's THEIR-mon axis (cells + revealed bits) and the incoming matrix's OUR-mon axis."""
    op = _render_op()
    t = _positions(op)
    omx = t.outgoing_matrix[0].long()
    omx_cells = omx[:_DMG_OUT_N_MOVES * TEAM_SIZE * _DMG_OMX_CELL].reshape(_DMG_OUT_N_MOVES, TEAM_SIZE, _DMG_OMX_CELL)
    omx_rev = omx[_DMG_OUT_N_MOVES * TEAM_SIZE * _DMG_OMX_CELL:]
    assert omx_rev.numel() == TEAM_SIZE
    imx_cells = t.incoming_matrix[0].long()[K * _DMG_IMX_HEADER:].reshape(TEAM_SIZE, K, _DMG_IMX_CELL)
    _plant_random_gains(op, seed=13)
    raw = torch.rand(4, op.out_dim, generator=torch.Generator().manual_seed(2))
    for sigma in ([5, 4, 3, 2, 1, 0], [1, 2, 3, 4, 5, 0]):
        perm = torch.arange(op.out_dim)
        perm[omx_cells.reshape(-1)] = omx_cells[:, sigma].reshape(-1)
        perm[omx_rev] = omx_rev[sigma]
        perm[imx_cells.reshape(-1)] = imx_cells[sigma].reshape(-1)
        with torch.no_grad():
            assert torch.equal(op.apply_out_gain(raw[:, perm]), op.apply_out_gain(raw)[:, perm]), sigma


def test_every_team_slot_copy_of_a_channel_accumulates_into_one_parameter(
        fe: Gen3FeaturesExtractor, obs: torch.Tensor) -> None:
    op = fe.damage_op
    raw = _raw_block(fe, obs)
    w = torch.randn(raw.shape, generator=torch.Generator().manual_seed(5))
    op.out_gain.grad = None
    (op.apply_out_gain(raw) * w).sum().backward()
    grad = op.out_gain.grad.detach().clone()
    op.out_gain.grad = None
    contrib = (w * raw).sum(0)
    for grp in _team_slot_groups(op):
        for f in range(grp.shape[1]):
            cols = {_gain_index(op, int(p)) for p in grp[:, f]}
            assert len(cols) == 1, f"the 6 team-slot copies of one channel map to {len(cols)} parameters"
            j = cols.pop()
            torch.testing.assert_close(grad[j], contrib[grp[:, f]].sum(), rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------------------------- the gate
def test_the_live_stamps_are_the_mon_tie() -> None:
    from agents.model.model_version import ARCH_SIGNATURE, MIGRATION_FLOOR, MODEL_CONFIG_VERSION, SIGNATURE_FIRST_VERSION
    from agents.model.model_version.version_break import MON_TIE_CONFIG, MON_TIE_SIGNATURE
    assert MODEL_CONFIG_VERSION == MIGRATION_FLOOR == MON_TIE_CONFIG == 145
    assert ARCH_SIGNATURE == MON_TIE_SIGNATURE == "gen3_mon_tied_gain_v1"
    assert SIGNATURE_FIRST_VERSION[ARCH_SIGNATURE] == 145
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    assert (cfg["config_version"], cfg["arch_signature"]) == (145, "gen3_mon_tied_gain_v1")


def test_a_v144_config_is_refused_naming_the_tie_and_the_pin(tmp_path: Any) -> None:
    from agents.model.model_version import ModelVersionError, _migrate_config
    from agents.model.model_version.version_break import (LAST_V144_COMMIT, PreBreakCheckpointError,
                                                          check_post_break)
    data = {"config_version": 144, "arch_signature": "gen3_x5_version_break_v1"}
    with pytest.raises(ModelVersionError) as exc:
        _migrate_config(dict(data))
    msg = str(exc.value)
    assert "PRE-GENERATION" in msg and "out_gain" in msg and "TEAM SLOTS" in msg and "99 -> 29" in msg
    assert LAST_V144_COMMIT in msg and "PINNED" in msg
    assert "belief_tokens" not in msg, "a v144 config is X5-only: the belief paragraph does not apply"
    cfg = tmp_path / "model_config.json"
    cfg.write_text(json.dumps(data))
    with pytest.raises(PreBreakCheckpointError) as pexc:
        check_post_break(str(cfg))
    assert pexc.value.last_commit == LAST_V144_COMMIT and pexc.value.config_version == 144
