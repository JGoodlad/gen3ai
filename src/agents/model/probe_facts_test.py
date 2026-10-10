"""`gen3_probe_facts_v1` (config v153) — the representation probe battery's two candidates
(`designs/endstate/design_hand_computed_features.md` §4 rows 12-13, `measurements/probe_battery_2026-10-09/`), each its
own flag, OFF in production:

* `--effective-stats on` (`static_facts.effective_stat_features`): each side's ACTIVE mon's stage-applied Atk / Def /
  SpA / SpD / Spe (Spe after paralysis) / 500 and its accuracy / evasion multipliers, as token content;
* `--move-target-state on` (`static_facts.move_target_features`): their active's HP + status onto our 4 E3 seats.

What a revert breaks is in each docstring.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any, Dict, List

import pytest
import torch

from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.identity_init_test import _build_real_policy
from agents.model.static_facts import (EFFECTIVE_STATS_DIM, EFFECTIVE_STATS_FACTS, MOVE_TARGET_DIM,
                                       _acc_stage_mult, _stat_stage_mult, effective_stat_features,
                                       move_target_features)
from agents.model.static_port_test import _permute_ours
from agents.observation.constants import (CONDITION_DIM, POKEMON_CONDITION_OFFSET, POKEMON_HP_OFFSET,
                                          POKEMON_SPREAD_DIM, POKEMON_SPREAD_OFFSET, TEAM_SIZE)
from utils.paths import repo_path

EI = {n: i for i, n in enumerate(EFFECTIVE_STATS_FACTS)}


def _toggles(**over: Any) -> Dict[str, Any]:
    with open(repo_path("designs", "production_config.json")) as fh:
        cfg = json.load(fh)
    tog = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    tog.update(over)
    return tog


_CACHE: Dict[str, Any] = {}


def _policy(**over: Any) -> Any:
    key = json.dumps(over, sort_keys=True)
    if key not in _CACHE:
        _CACHE[key] = _build_real_policy(**_toggles(**over))[0].policy
    return _CACHE[key]


@pytest.fixture(scope="module")
def rows() -> torch.Tensor:
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    obs, _mask = load_parity_rows(Gen3ObservationEncoder(load_mappings()).dimension)
    return torch.as_tensor(obs)


def _record(pol: Any, x: torch.Tensor) -> List[torch.Tensor]:
    fe = pol.features_extractor
    with torch.no_grad():
        pi, vf = fe({"observation": x})
        p = fe.last_pointer_inputs
        lat = pol.mlp_extractor.forward_actor(pi)
        logits = pol.pointer_head(lat, p.move_tokens, p.move_valid, p.team_tokens, p.move_cells, p.switch_cells)
    return [pi, vf, p.team_tokens, p.move_tokens, p.move_cells, p.switch_cells, fe.last_win_prob_logits, logits]


def _stage_cols(stages: Dict[int, int]) -> torch.Tensor:
    """An active context's 14 boost columns for {stat index: stage} (atk def spa spd spe acc eva)."""
    v = torch.zeros(14)
    for i, s in stages.items():
        v[2 * i] = max(0, s) / 6.0
        v[2 * i + 1] = max(0, -s) / 6.0
    return v


def test_the_stage_tables_are_showdowns() -> None:
    """`sim/pokemon.ts` boostTable [1, 1.5, 2, 2.5, 3, 3.5, 4] (+s; 1/x for −s) and `data/mods/gen3/scripts.ts`'s
    accuracy table [1, 4/3, 5/3, 2, 7/3, 8/3, 3]."""
    st = torch.arange(-6, 7, dtype=torch.float32)
    table = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    acc = [1, 4 / 3, 5 / 3, 2, 7 / 3, 8 / 3, 3]
    for s, m, a in zip(st.tolist(), _stat_stage_mult(st).tolist(), _acc_stage_mult(st).tolist()):
        k = int(abs(s))
        assert m == pytest.approx(table[k] if s >= 0 else 1.0 / table[k])
        assert a == pytest.approx(acc[k] if s >= 0 else 1.0 / acc[k])


def test_the_effective_stats_on_a_constructed_board(rows: torch.Tensor) -> None:
    """Our active at +2 Atk, −1 SpD, paralysed, +1 accuracy: its Atk ×2, SpD ×2/3, Spe ×1/4 (the stage-free Spe), the
    accuracy multiplier 4/3, from the OBSERVED spread; their active at −2 Def and +2 evasion priced on the believed
    spread (planted); every benched mon 0. Fails on a stage table off by a step, a dropped paralysis, the belief read
    for our mon, or a benched mon carrying an active's stages."""
    fe = _policy(effective_stats="on").features_extractor
    op = fe.damage_op
    with torch.no_grad():
        ctx = fe.unpack({"observation": rows[:4]})
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:, -1].any(dim=1)
        assert bool(has_opp.all()), "the first four fixture rows need an opponent active"
        B = ctx.batch_size
        ar = torch.arange(B)
        our_raw, opp_raw = ctx.our_ctx_raw.clone(), ctx.opp_ctx_raw.clone()
        our_raw[:, :14] = _stage_cols({0: 2, 3: -1, 5: 1})
        opp_raw[:, :14] = _stage_cols({1: -2, 6: 2})
        pp = ctx.pokemon_part.clone()
        hp = ctx.hp_and_active.clone()
        oa, ta = ctx.our_active_idx, ctx.opp_active_local
        hp[ar, oa, 0] = 0.7
        hp[ar, TEAM_SIZE + ta, 0] = 0.5
        pp[ar, oa, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + CONDITION_DIM] = 0.0
        pp[ar, oa, POKEMON_CONDITION_OFFSET + 2] = 1.0                    # PAR
        pp[ar, TEAM_SIZE + ta, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + CONDITION_DIM] = 0.0
        c = dataclasses.replace(ctx, our_ctx_raw=our_raw, opp_ctx_raw=opp_raw, pokemon_part=pp, hp_and_active=hp)
        sb = torch.full((B, TEAM_SIZE, 5), 250.0)
        out = effective_stat_features(op, c, sb)                          # [B,12,7]
    assert out.shape == (B, 2 * TEAM_SIZE, EFFECTIVE_STATS_DIM)
    base = op.BASE_STATS[c.species_ids[ar, oa]]
    spr = pp[ar, oa, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
    raw = (2.0 * base[:, 1:6] + spr[:, 1:6] * 31.0 + spr[:, 7:12] * 252.0 / 4.0 + 5.0) * spr[:, 13:18]
    ours = out[ar, oa]
    assert torch.allclose(ours[:, EI["atk"]], raw[:, 0] * 2.0 / 500.0, rtol=1e-5)
    assert torch.allclose(ours[:, EI["def"]], raw[:, 1] / 500.0, rtol=1e-5)
    assert torch.allclose(ours[:, EI["spd"]], raw[:, 3] * (2.0 / 3.0) / 500.0, rtol=1e-5)
    assert torch.allclose(ours[:, EI["spe"]], raw[:, 4] * 0.25 / 500.0, rtol=1e-5)
    assert torch.allclose(ours[:, EI["accuracy_mult"]], torch.full((B,), 4.0 / 3.0))
    assert torch.allclose(ours[:, EI["evasion_mult"]], torch.ones(B))
    theirs = out[ar, TEAM_SIZE + ta]
    assert torch.allclose(theirs[:, EI["def"]], torch.full((B,), 250.0 * 0.5 / 500.0))
    assert torch.allclose(theirs[:, EI["atk"]], torch.full((B,), 250.0 / 500.0))
    assert torch.allclose(theirs[:, EI["evasion_mult"]], torch.full((B,), 5.0 / 3.0))
    active = torch.zeros(B, 2 * TEAM_SIZE, dtype=torch.bool)
    active[ar, oa] = True
    active[ar, TEAM_SIZE + ta] = True
    assert float(out[~active].abs().max()) == 0.0


def test_the_move_target_is_their_active_and_nothing_without_one(rows: torch.Tensor) -> None:
    """[their active's HP fraction, its status one-hot] (the TARGET of our moves, never our actor's); a board with no
    opponent active reads 0. Fails on the actor's row read instead, or a target invented for an empty field."""
    fe = _policy(move_target_state="on").features_extractor
    with torch.no_grad():
        ctx = fe.unpack({"observation": rows[:4]})
        f = move_target_features(ctx)
        ar = torch.arange(ctx.batch_size)
        row = ctx.pokemon_part[ar, TEAM_SIZE + ctx.opp_active_local]
        want = torch.cat([row[:, POKEMON_HP_OFFSET:POKEMON_HP_OFFSET + 1],
                          row[:, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + CONDITION_DIM]], dim=-1)
        has = ctx.hp_and_active[:, TEAM_SIZE:, -1].any(dim=1).float()[:, None]
        assert f.shape == (ctx.batch_size, MOVE_TARGET_DIM)
        assert torch.equal(f, want * has)
        hp = ctx.hp_and_active.clone()
        hp[:, TEAM_SIZE:, -1] = 0.0
        assert float(move_target_features(dataclasses.replace(ctx, hp_and_active=hp)).abs().max()) == 0.0


@pytest.mark.parametrize("flag,extra", [("effective_stats", "effective_stats_proj"),
                                        ("move_target_state", "move_target_proj")])
@pytest.mark.parametrize("encoding", ["legacy", "static"])
def test_one_lever_init_and_the_fact_reaches_the_trunk(rows: torch.Tensor, flag: str, extra: str,
                                                       encoding: str) -> None:
    """ON = OFF + exactly the lever's zero-init projection (every other byte equal) and the init forward bit-identical,
    in both encodings; with the projection planted, the pointer inputs move (the fact reaches the trunk)."""
    off, on = _policy(token_encoding=encoding), copy.deepcopy(_policy(token_encoding=encoding, **{flag: "on"}))
    so, sn = off.state_dict(), on.state_dict()
    new = sorted(set(sn) - set(so))
    assert new and all(extra in k for k in new), new
    assert all(float(sn[k].abs().max()) == 0.0 for k in new)
    assert all(torch.equal(so[k], sn[k]) for k in so)
    for a, b in zip(_record(off, rows), _record(on, rows)):
        assert torch.equal(a, b)
    proj = getattr(on.features_extractor, extra)
    with torch.no_grad():
        proj.weight.copy_(torch.randn(proj.weight.shape, generator=torch.Generator().manual_seed(2)) * 0.1)
    assert not torch.equal(_record(on, rows)[3], _record(off, rows)[3]) or \
        not torch.equal(_record(on, rows)[2], _record(off, rows)[2])


def test_the_effective_stats_permute_with_our_team_slots(rows: torch.Tensor) -> None:
    """One formula per mon: permuting our six slots permutes our rows (the active moves with its slot)."""
    fe = _policy(effective_stats="on").features_extractor
    sig = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        ctx = fe.unpack({"observation": rows[:16]})
        base = effective_stat_features(fe.damage_op, ctx)
        perm = effective_stat_features(fe.damage_op, _permute_ours(ctx, sig))
    assert torch.allclose(perm[:, :TEAM_SIZE], base[:, sig], atol=1e-6)
    assert torch.allclose(perm[:, TEAM_SIZE:], base[:, TEAM_SIZE:], atol=1e-6)
    assert float(base.abs().sum()) > 0.0


def test_versioning_migrates_and_gates_both() -> None:
    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError
    from agents.model.model_version.migrations import _migrate_config
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    assert MODEL_CONFIG_VERSION >= 153
    data = _migrate_config({"config_version": 152})
    assert (data["effective_stats"], data["move_target_state"]) == ("off", "off")
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    mv = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    for n in ("effective_stats", "move_target_state"):
        with pytest.raises(ModelVersionError, match=n):
            mv.check_compatible(dataclasses.replace(mv, **{n: "on"}))
        rec = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {n: "on"}})
        assert getattr(rec, n) == "on"


def test_production_builds_neither() -> None:
    fe = _policy().features_extractor
    assert fe.effective_stats_proj is None and fe.move_target_proj is None
    assert fe.effective_stats == "off" and fe.move_target_state == "off"
