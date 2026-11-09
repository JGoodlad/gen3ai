"""`gen3_static_recovery_v1` (config v150) — the three STATIC-RECOVERY facts (`designs/endstate/design_static_tokens.md`
§13): `--trunk-layers N`, the switch-cell hazard and the end-of-turn residual (the last two built whenever the damage
operator is since the legacy deletion's U1 deleted `--switch-hazard-cost` / `--eot-residual`).

Built through a REAL SB3 policy (the construction path training uses, the identity_init_test rule) on the LEGACY base
(`legacy_production_config`: the pre-adoption production toggles — every lever here is ON in production since the END
STATE's adoption, 2026-10-12, so each OFF path is reached only from the legacy base) with `token_encoding='static'` (and legacy where a lever composes with it), over REAL observation rows (the
compile parity fixture), and on CONSTRUCTED boards for the physics.

Each test names what a revert breaks:

* TRUNK DEPTH: `--trunk-layers 3` at init is the 2-round network BIT FOR BIT (every shared parameter byte, the
  extractor's outputs, the pointer inputs and the win-prob logits; static AND legacy) — fails on a post-LN extra round
  (LN of an LN output is not the identity: pinned separately), on an RNG-drawing init (every later byte moves) or on a
  non-zero output projection; the extra round still LEARNS (its out-proj gets a gradient at init and moves the output
  once non-zero); permuting our team slots permutes the trunk's output with the extra round live.
* SWITCH-CELL HAZARD on constructed boards (0–3 layers, Flying, Levitate known / by its species, our side's layers):
  the block equals [our layers / 3, the switch-in fraction]; it IS the op's ONE entry rule (a planted `spikes_entry`
  moves it) and our half of `--mon-hazard-cost`'s fact; it is APPENDED last to the switch cell and reaches the policy
  only through the pointer head's zero-init projection (planted, only SWITCH logits move); it permutes with our slots.
* END-OF-TURN RESIDUAL, each component on a constructed board in isolation: Leftovers (known / consumed / the species'
  prior), sand (types, Sand Veil), hail, a timed weather's LAST turn (no chip), Cloud Nine suppression, Rain Dish, burn /
  poison 1/8, Toxic (n+1)/16 for the active and 1/16 for a benched toxic mon, Shed Skin's 1/3 cure, Leech Seed drain on
  the seeded ACTIVE only and its heal to EVERY mon of the other side scaled by max HP (Liquid Ooze inverts it), Wish
  +1/2 to every mon of its side, Ingrain, Curse, Nightmare × P(stays asleep), the net's HP clamp, the alive and
  concreteness gates; the rule permutes with our slots; its projection is zero-init and reaches the trunk once live.
* one-lever init: each remaining lever ON adds only its own parameters (zero where the identity needs it), every
  other initial byte equals the OFF build's, and the init forward is bit-identical.
* VERSIONING: a pre-v150 config migrates to trunk depth 2; a depth mismatch is refused by `check_compatible`.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.damage_kinds import residual_points
from agents.model.eot_residual import EOT_DIM, EOT_FACTS, leech_seed_drained
from agents.model.extractor_ctx import POKEMON_ABILITY_KNOWN_OFFSET, ExtractorContext
from agents.model.identity_init_test import _build_real_policy
from agents.model.static_facts import SWITCH_HAZARD_DIM, mon_hazard_features, switch_hazard_features
from agents.model.static_port_test import _constructed_board, _permute_ours, _plant_live
from agents.model.trunk_depth import IdentityInitRound
from agents.observation.constants import (BOOSTS_DIM, POKEMON_CONDITION_OFFSET, POKEMON_COUNTER_OFFSET,
                                          POKEMON_ITEMS_OFFSET, POKEMON_SLEEP_BELIEF_OFFSET, POKEMON_SPREAD_OFFSET,
                                          TEAM_SIZE, TOXIC_STAGE_MAX)
from agents.observation.gen3_effects import VOLATILE_SLOTS
from main.train.arch_arms import legacy_production_config
from utils.paths import repo_path

#: fp32 bound on the same arithmetic in a different order (a permuted sum, a reassociated matmul).
FP32_ATOL = 1e-5
_CHIP = (0.0, 1.0 / 8, 1.0 / 6, 1.0 / 4)
_COL = {name: i for i, name in enumerate(EOT_FACTS)}


def _toggles(encoding: str = "static", **over: Any) -> Dict[str, Any]:
    # LEGACY-BASE re-point (the END STATE's adoption, 2026-10-12): "base ⊕ one lever" with every lever OFF in the base.
    with open(repo_path("designs", "production_config.json")) as fh:
        cfg = legacy_production_config(json.load(fh))
    tog = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    tog.update(token_encoding=encoding, **over)
    return tog


_CACHE: Dict[str, Any] = {}


def _policy(encoding: str = "static", **over: Any) -> Any:
    key = json.dumps([encoding, over], sort_keys=True)
    if key not in _CACHE:
        _CACHE[key] = _build_real_policy(**_toggles(encoding, **over))[0].policy
    return _CACHE[key]


@pytest.fixture(scope="module")
def rows() -> torch.Tensor:
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    obs, _mask = load_parity_rows(Gen3ObservationEncoder(load_mappings()).dimension)
    return torch.as_tensor(obs)


def _forward_record(pol: Any, x: torch.Tensor) -> List[torch.Tensor]:
    """Everything the init forward hands a head: the extractor's two outputs, the pointer inputs, the win-prob logits,
    and the policy's action logits."""
    fe = pol.features_extractor
    with torch.no_grad():
        pi, vf = fe({"observation": x})
        p = fe.last_pointer_inputs
        lat = pol.mlp_extractor.forward_actor(pi)
        logits = pol.pointer_head(lat, p.move_tokens, p.move_valid, p.team_tokens, p.move_cells, p.switch_cells)
    return [pi, vf, p.team_tokens, p.move_tokens, p.move_cells, p.switch_cells, fe.last_win_prob_logits, logits]


def _one_lever(off: Any, on: Any, extra_prefixes: Tuple[str, ...]) -> List[str]:
    """Assert ON = OFF + exactly the lever's new keys (each under ``extra_prefixes``); every shared key's bytes equal.
    Returns the new keys."""
    so, sn = off.state_dict(), on.state_dict()
    assert not set(so) - set(sn), sorted(set(so) - set(sn))[:5]
    extra = sorted(set(sn) - set(so))
    assert extra and all(any(p in k for p in extra_prefixes) for k in extra), extra
    moved = [k for k in so if not torch.equal(so[k], sn[k])]
    assert not moved, f"the lever moved other initial bytes: {moved[:5]}"
    return extra


# ===================================================================================================== TRUNK DEPTH
@pytest.mark.slow  # ship gate (2026-10-19): 41 s beside a live run
@pytest.mark.parametrize("encoding", ["static", "legacy"])
def test_trunk_depth_3_at_init_is_the_2_round_network_bit_for_bit(rows: torch.Tensor, encoding: str) -> None:
    two, three = _policy(encoding), _policy(encoding, trunk_layers=3)
    extra = _one_lever(two, three, ("team_transformer.extra_rounds.0.",))
    extra = [k for k in extra if k.startswith("features_extractor.")]       # (pi_ / vf_ are aliases of it)
    zeros = [k for k in extra if ".out_proj." in k or ".linear2." in k]
    assert len(zeros) == 4 and all(float(three.state_dict()[k].abs().max()) == 0.0 for k in zeros), zeros
    live = [k for k in extra if (".in_proj.weight" in k or ".linear1.weight" in k)]
    assert all(float(three.state_dict()[k].abs().max()) > 0.0 for k in live), "the extra round's in-proj is dead"
    for a, b in zip(_forward_record(two, rows), _forward_record(three, rows)):
        assert a is not None and b is not None
        assert torch.equal(a, b), "the 3-round network at init is not the 2-round one bit for bit"


def test_a_post_ln_round_with_zero_outputs_is_not_the_identity_but_the_pre_ln_round_is() -> None:
    """Why the extra round is PRE-LN: the trunk's rounds are post-LN, so their output is a LayerNorm output; a post-LN
    round with zero out-proj / FFN output computes LN(LN(x)) ≠ x (LayerNorm's eps), while the pre-LN residual round
    adds exactly 0."""
    from agents.model.team_transformer import BiasedEncoderLayer
    g = torch.Generator().manual_seed(3)
    x = torch.nn.functional.layer_norm(torch.randn(4, 15, 128, generator=g) * 0.3, (128,))
    post = BiasedEncoderLayer()
    pre = IdentityInitRound()
    with torch.no_grad():
        for lin in (post.out_proj, post.linear2):
            lin.weight.zero_()
            lin.bias.zero_()
        dev = float((post(x) - x).abs().max())
        assert dev > 0.0, "post-LN with zero outputs happened to be the identity on this input"
        assert torch.equal(pre(x), x), "the pre-LN identity-init round is not the identity"


def test_the_extra_round_learns_from_init(rows: torch.Tensor) -> None:
    pol = copy.deepcopy(_policy(trunk_layers=3))
    fe = pol.features_extractor
    rnd = fe.team_transformer.extra_rounds[0]
    pi, vf = fe({"observation": rows[:16]})
    (pi.square().mean() + vf.square().mean()).backward()
    assert rnd.out_proj.weight.grad is not None and float(rnd.out_proj.weight.grad.abs().max()) > 0.0
    with torch.no_grad():
        base = fe({"observation": rows[:16]})[0].clone()
        rnd.out_proj.weight.copy_(torch.randn(rnd.out_proj.weight.shape, generator=torch.Generator().manual_seed(1)))
        moved = fe({"observation": rows[:16]})[0]
    assert not torch.equal(base, moved), "a non-zero extra round did not change the forward"


def test_the_extra_round_is_equivariant_over_our_team_slots(rows: torch.Tensor) -> None:
    fe = copy.deepcopy(_policy(trunk_layers=3).features_extractor)
    _plant_live(fe)
    with torch.no_grad():
        for p in fe.team_transformer.extra_rounds.parameters():
            if p.dim() == 2:
                p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(7)) * 0.05)
    fe.eval()
    sig = torch.tensor([3, 0, 5, 1, 4, 2])

    def trunk_out(perm: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        h = (fe.unpack.register_forward_hook(lambda _m, _a, ctx: _permute_ours(ctx, sig)) if perm else None)
        try:
            with torch.no_grad():
                fe({"observation": rows})
        finally:
            if h is not None:
                h.remove()
        ours, theirs = fe.stash.trunk_tokens
        return ours.clone(), theirs.clone()

    o, t = trunk_out(False)
    op, tp = trunk_out(True)
    assert float(o.std()) > 0.0
    assert torch.allclose(op, o[:, sig], atol=FP32_ATOL), "the deeper trunk's our-mon outputs did not permute"
    assert torch.allclose(tp, t, atol=FP32_ATOL), "their tokens moved under a permutation of OUR slots"


def test_trunk_layers_refuses_an_unknown_depth() -> None:
    from agents.model.trunk_depth import build_extra_rounds
    assert build_extra_rounds(2) is None
    for bad in (1, 5):
        with pytest.raises(ValueError, match="trunk_layers"):
            build_extra_rounds(bad)


# ============================================================================================ SWITCH-CELL HAZARD
def test_the_switch_cell_hazard_on_constructed_boards(rows: torch.Tensor) -> None:
    fe = _policy().features_extractor
    ctx, grounded = _constructed_board(fe, rows)
    with torch.no_grad():
        f = switch_hazard_features(fe.damage_op, ctx)                                  # [4,6,2]
        per_mon = mon_hazard_features(fe.damage_op, ctx)[:, :TEAM_SIZE]
    assert f.shape == (4, TEAM_SIZE, SWITCH_HAZARD_DIM)
    assert torch.equal(f, per_mon), "the switch cell is not our half of the per-mon fact"
    layers = (ctx.spikes_feature * 3).round().long()
    for b in range(4):
        for j in range(TEAM_SIZE):
            assert float(f[b, j, 0]) == pytest.approx(float(ctx.spikes_feature[b, 0]), abs=1e-7), (b, j)
            want = _CHIP[int(layers[b, 0])] * float(grounded[j])
            assert float(f[b, j, 1]) == pytest.approx(want, abs=1e-7), (b, j, "chip")
    assert bool((f[:, :, 1] > 0).any()) and bool((f[:, :, 1] == 0).any()), "PRECONDITION: both immune and grounded"


def test_the_switch_cell_reads_the_ops_one_entry_rule(rows: torch.Tensor, monkeypatch: Any) -> None:
    fe = _policy().features_extractor
    ctx = fe.unpack({"observation": rows[:4]})
    planted = torch.full((4, TEAM_SIZE), 0.3125)
    monkeypatch.setattr(type(fe.damage_op), "spikes_entry",
                        lambda self, c: (planted, planted * 0, planted * 0, planted * 0))
    with torch.no_grad():
        f = switch_hazard_features(fe.damage_op, ctx)
    assert torch.equal(f[..., 1], planted), "the switch cell computes its own Spikes rule"


def test_the_switch_hazard_reaches_only_the_switch_logits_through_a_zero_init_projection(rows: torch.Tensor) -> None:
    on = _policy()
    key = "pointer_head.switch_extra_proj.weight"
    assert key in on.state_dict() and float(on.state_dict()[key].abs().max()) == 0.0
    b = _forward_record(on, rows)
    fe = on.features_extractor
    assert fe.pointer_switch_extra_dim == SWITCH_HAZARD_DIM
    with torch.no_grad():
        want = switch_hazard_features(fe.damage_op, fe.unpack({"observation": rows}))
    assert torch.equal(b[5][..., -SWITCH_HAZARD_DIM:], want), "the block is not APPENDED last"
    assert bool((want[..., 0] > 0).any()), "PRECONDITION: some row with Spikes on our side"
    pol = copy.deepcopy(on)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():                       # the three scorers are zero-init too: make every logit live first
        for sc in (pol.pointer_head.move_score, pol.pointer_head.switch_score, pol.pointer_head.struggle_score):
            sc.weight.copy_(torch.randn(sc.weight.shape, generator=g))
    live = _forward_record(pol, rows)
    with torch.no_grad():
        pol.pointer_head.switch_extra_proj.weight.fill_(0.5)
    c = _forward_record(pol, rows)
    moved = (c[-1] != live[-1]).any(0)
    assert bool(moved[:TEAM_SIZE].any()), "a live projection did not reach the switch logits"
    assert not bool(moved[TEAM_SIZE:].any()), "the switch block leaked into a move / struggle logit"


def test_the_switch_cell_hazard_permutes_with_our_slots(rows: torch.Tensor) -> None:
    fe = _policy().features_extractor
    ctx = fe.unpack({"observation": rows})
    sig = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        base = switch_hazard_features(fe.damage_op, ctx)
        perm = switch_hazard_features(fe.damage_op, _permute_ours(ctx, sig))
    assert torch.equal(perm, base[:, sig])


# ========================================================================================= END-OF-TURN RESIDUAL
def _abil(name: str) -> int:
    from agents import gen3_data
    return int(gen3_data.abilities.get(name).num)


def _item(name: str) -> int:
    from agents import gen3_data
    return int(gen3_data.items.get(name).num)


def _sp(name: str) -> int:
    from agents import gen3_data
    return int(gen3_data.species.get(name).num)


def _types(*names: str) -> Tuple[int, int]:
    from agents.model.damage_tables import _T2I
    tt = sorted(names)
    return _T2I[tt[0]], (_T2I[tt[1]] if len(tt) > 1 else 0)


def _ours(name: str, d: int) -> float:
    """OUR mon's exact end-of-turn amount as a fraction of its max HP (gen3_residual_exact_v1): Showdown's
    ``max(1, ⌊maxhp / d⌋) / maxhp`` at HP IV 31, EV 0 (the `_blank` spread). THEIR mon keeps the nominal ``1 / d``."""
    from agents import gen3_data
    base = gen3_data.species.get(name).base_stats["hp"]
    mhp = 1 if base == 1 else 2 * base + 31 + 110
    return max(1, mhp // d) / mhp


@pytest.fixture(scope="module")
def eot_fe() -> Any:
    return _policy().features_extractor


def _blank(fe: Any, rows: torch.Tensor) -> ExtractorContext:
    """ONE board where every mon is a revealed, ability-known (Thick Fat), Choice-Band-holding, status-free Snorlax at
    HP 0.5 with an exact max HP of 2·160 + 31 + 110 = 461 (HP IV 31, EV 0), slot 0 active on both sides, no weather,
    no Wish, no volatile — the rule reads 0 everywhere except where a test plants a component."""
    ctx = fe.unpack({"observation": rows[:1]})
    n = 2 * TEAM_SIZE
    pp = ctx.pokemon_part.clone()
    pp[..., POKEMON_ABILITY_KNOWN_OFFSET] = 1.0
    pp[..., POKEMON_ITEMS_OFFSET + 1] = 1.0
    pp[..., POKEMON_ITEMS_OFFSET + 2] = 0.0
    pp[..., POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7] = 0.0
    pp[..., POKEMON_CONDITION_OFFSET] = 1.0
    pp[..., POKEMON_COUNTER_OFFSET:POKEMON_COUNTER_OFFSET + 2] = 0.0
    pp[..., POKEMON_SLEEP_BELIEF_OFFSET:POKEMON_SLEEP_BELIEF_OFFSET + 3] = 0.0
    pp[..., POKEMON_SPREAD_OFFSET] = 1.0                                     # HP IV 31
    pp[..., POKEMON_SPREAD_OFFSET + 6] = 0.0                                 # HP EV 0
    ha = ctx.hp_and_active.clone()
    ha[..., 0] = 0.5
    ha[..., -1] = 0.0
    ha[:, 0, -1] = 1.0
    ha[:, TEAM_SIZE, -1] = 1.0
    t1, t2 = _types("NORMAL")
    w = torch.zeros_like(ctx.weather_feature)
    w[:, 0] = 1.0
    nmr = ctx.non_matchup_rest.clone()
    off = fe.eot_residual_rule
    nmr[:, off._wish_our] = 0.0
    nmr[:, off._wish_opp] = 0.0
    return dataclasses.replace(
        ctx, pokemon_part=pp, hp_and_active=ha,
        species_ids=torch.full((1, n), _sp("snorlax"), dtype=ctx.species_ids.dtype),
        type1_ids=torch.full((1, n), t1, dtype=ctx.type1_ids.dtype),
        type2_ids=torch.full((1, n), t2, dtype=ctx.type2_ids.dtype),
        ability1_ids=torch.full((1, n), _abil("thickfat"), dtype=ctx.ability1_ids.dtype),
        item_ids=torch.full((1, n), _item("choiceband"), dtype=ctx.item_ids.dtype),
        our_active_idx=torch.zeros(1, dtype=ctx.our_active_idx.dtype),
        opp_active_local=torch.zeros(1, dtype=ctx.opp_active_local.dtype),
        our_ctx_raw=torch.zeros_like(ctx.our_ctx_raw), opp_ctx_raw=torch.zeros_like(ctx.opp_ctx_raw),
        weather_feature=w, non_matchup_rest=nmr,
        opp_believed_mask=torch.zeros_like(ctx.opp_believed_mask))


def _mon(ctx: ExtractorContext, slot: int, *, species: str = "", types: Tuple[str, ...] = (), ability: str = "",
         ability_known: bool = True, item: str = "", item_known: bool = True, consumed: bool = False,
         status: str = "", tox_ticks: int = 0, p_wake: float = 0.0, hp: float = -1.0) -> ExtractorContext:
    pp, ha = ctx.pokemon_part.clone(), ctx.hp_and_active.clone()
    sp, t1, t2, ab, it = (ctx.species_ids.clone(), ctx.type1_ids.clone(), ctx.type2_ids.clone(),
                          ctx.ability1_ids.clone(), ctx.item_ids.clone())
    if species:
        sp[:, slot] = _sp(species)
    if types:
        a, b = _types(*types)
        t1[:, slot], t2[:, slot] = a, b
    if ability:
        ab[:, slot] = _abil(ability)
    pp[:, slot, POKEMON_ABILITY_KNOWN_OFFSET] = 1.0 if ability_known else 0.0
    if item:
        it[:, slot] = _item(item)
    pp[:, slot, POKEMON_ITEMS_OFFSET + 1] = 1.0 if item_known else 0.0
    pp[:, slot, POKEMON_ITEMS_OFFSET + 2] = 1.0 if consumed else 0.0
    if status:
        pp[:, slot, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7] = 0.0
        pp[:, slot, POKEMON_CONDITION_OFFSET + ["", "brn", "par", "slp", "frz", "psn", "tox"].index(status)] = 1.0
    pp[:, slot, POKEMON_COUNTER_OFFSET + 1] = min(tox_ticks, TOXIC_STAGE_MAX) / TOXIC_STAGE_MAX    # the cell's own scale
    pp[:, slot, POKEMON_SLEEP_BELIEF_OFFSET + 1] = p_wake
    if hp >= 0.0:
        ha[:, slot, 0] = hp
    return dataclasses.replace(ctx, pokemon_part=pp, hp_and_active=ha, species_ids=sp, type1_ids=t1, type2_ids=t2,
                               ability1_ids=ab, item_ids=it)


def _vol(ctx: ExtractorContext, side: int, name: str) -> ExtractorContext:
    col = BOOSTS_DIM + VOLATILE_SLOTS.index(name)
    raw = (ctx.our_ctx_raw if side == 0 else ctx.opp_ctx_raw).clone()
    raw[:, col] = 1.0
    return dataclasses.replace(ctx, **{("our_ctx_raw" if side == 0 else "opp_ctx_raw"): raw})


def _weather(ctx: ExtractorContext, kind: int, permanent: bool, turns_left: int = 0) -> ExtractorContext:
    w = torch.zeros_like(ctx.weather_feature)
    w[:, kind] = 1.0
    w[:, 5] = 1.0 if permanent else 0.0
    w[:, 6] = turns_left / 5.0
    return dataclasses.replace(ctx, weather_feature=w)


def _eot(fe: Any, ctx: ExtractorContext) -> torch.Tensor:
    with torch.no_grad():
        return fe.eot_residual_rule(ctx, fe.damage_op)[0]                               # [12, EOT_DIM]


def _only(out: torch.Tensor, col: str, want: Dict[int, float]) -> None:
    """Column ``col`` equals ``want`` at the named slots and 0 at every other slot; every OTHER component column is 0
    everywhere (the component is isolated); ``net`` equals the HP-clamped sum."""
    c = _COL[col]
    for slot in range(2 * TEAM_SIZE):
        assert float(out[slot, c]) == pytest.approx(want.get(slot, 0.0), abs=1e-6), (col, slot)
    others = [i for name, i in _COL.items() if name not in (col, "net")]
    assert float(out[:, others].abs().max()) == 0.0, (col, out[:, others])


def test_the_blank_board_reads_zero(eot_fe: Any, rows: torch.Tensor) -> None:
    out = _eot(eot_fe, _blank(eot_fe, rows))
    assert out.shape == (2 * TEAM_SIZE, EOT_DIM)
    assert float(out.abs().max()) == 0.0, out


def test_leftovers_known_consumed_and_by_the_species_prior(eot_fe: Any, rows: torch.Tensor) -> None:
    ctx = _blank(eot_fe, rows)
    ctx = _mon(ctx, 1, item="leftovers")                                        # ours, known: exact
    ctx = _mon(ctx, 2, item="leftovers", consumed=True)                          # knocked off / consumed: none
    ctx = _mon(ctx, 7, item="leftovers")                                        # theirs, revealed
    ctx = _mon(ctx, 8, species="blissey", item="choiceband", item_known=False)    # theirs, unknown: the prior
    out = _eot(eot_fe, ctx)
    p_bliss = float(eot_fe.eot_residual_rule.SPECIES_LEFTOVERS_PRIOR[_sp("blissey")])          # Smogon P(Leftovers)
    assert 0.5 < p_bliss < 1.0
    _only(out, "leftovers", {1: _ours("snorlax", 16), 7: 1 / 16, 8: p_bliss / 16})    # ours ⌊461/16⌋ = 28 HP
    assert float(out[1, _COL["net"]]) == pytest.approx(28 / 461)


def test_sand_and_hail_chip_by_type_and_sand_veil(eot_fe: Any, rows: torch.Tensor) -> None:
    ctx = _weather(_blank(eot_fe, rows), 3, permanent=True)                        # Sand Stream: permanent
    ctx = _mon(ctx, 1, species="tyranitar", types=("ROCK", "DARK"))
    ctx = _mon(ctx, 2, species="skarmory", types=("STEEL", "FLYING"))
    ctx = _mon(ctx, 3, species="cacturne", types=("GRASS", "DARK"), ability="sandveil")
    ctx = _mon(ctx, 9, species="dewgong", types=("WATER", "ICE"))
    out = _eot(eot_fe, ctx)
    sp = {1: "tyranitar", 2: "skarmory", 3: "cacturne"}
    nominal = {s: (-_ours(sp.get(s, "snorlax"), 16) if s < TEAM_SIZE else -1 / 16) for s in range(2 * TEAM_SIZE)}
    _only(out, "weather", {s: v for s, v in nominal.items() if s not in (1, 2, 3)})
    hail = _eot(eot_fe, _weather(ctx, 4, permanent=False, turns_left=3))
    _only(hail, "weather", {s: v for s, v in nominal.items() if s != 9})


def test_a_timed_weathers_last_turn_has_no_chip(eot_fe: Any, rows: torch.Tensor) -> None:
    base = _blank(eot_fe, rows)
    for left, chip in ((5, True), (2, True), (1, False)):
        out = _eot(eot_fe, _weather(base, 3, permanent=False, turns_left=left))
        want = {s: (-_ours("snorlax", 16) if s < TEAM_SIZE else -1 / 16) for s in range(2 * TEAM_SIZE)}
        _only(out, "weather", want if chip else {})


def test_cloud_nine_on_the_field_suppresses_the_weather(eot_fe: Any, rows: torch.Tensor) -> None:
    ctx = _weather(_blank(eot_fe, rows), 3, permanent=True)
    ctx = _mon(ctx, 2, species="golduck", types=("WATER",), ability="cloudnine")    # our BENCH Golduck
    ctx = _mon(ctx, TEAM_SIZE, species="golduck", types=("WATER",), ability="damp", ability_known=False)
    out = _eot(eot_fe, ctx)
    p_c9 = float(eot_fe.eot_residual_rule.SPECIES_EOT_PRIOR[_sp("golduck"), 4])                # Smogon P(Cloud Nine)
    assert 0.9 < p_c9 < 1.0
    want = {}
    for s in range(2 * TEAM_SIZE):
        own = 1.0 if s == 2 else (p_c9 if s == TEAM_SIZE else 0.0)
        other = p_c9 if s < TEAM_SIZE else 0.0                                         # the OTHER side's active
        want[s] = -(_ours("snorlax", 16) if s < TEAM_SIZE else 1 / 16) * (1 - own) * (1 - other)
    want[2] = 0.0
    _only(out, "weather", want)


def test_rain_dish(eot_fe: Any, rows: torch.Tensor) -> None:
    ctx = _weather(_blank(eot_fe, rows), 2, permanent=False, turns_left=4)
    ctx = _mon(ctx, 4, species="ludicolo", types=("WATER", "GRASS"), ability="raindish")
    _only(_eot(eot_fe, ctx), "rain_dish", {4: _ours("ludicolo", 16)})                # ⌊301/16⌋ = 18 HP
    _only(_eot(eot_fe, _weather(ctx, 2, permanent=False, turns_left=1)), "rain_dish", {})


def test_the_status_tick_burn_poison_toxic_and_shed_skin(eot_fe: Any, rows: torch.Tensor) -> None:
    ctx = _blank(eot_fe, rows)
    ctx = _mon(ctx, 0, status="tox", tox_ticks=3)                                 # our ACTIVE: 4th tick
    ctx = _mon(ctx, 1, status="tox", tox_ticks=0)                                 # BENCH toxic: resets to 1/16
    ctx = _mon(ctx, 2, status="brn")
    ctx = _mon(ctx, 3, status="psn")
    ctx = _mon(ctx, 4, status="par")                                              # no tick
    ctx = _mon(ctx, 7, species="dragonair", types=("DRAGON",), ability="shedskin", status="psn")
    out = _eot(eot_fe, ctx)
    s16, s8 = _ours("snorlax", 16), _ours("snorlax", 8)                           # 28 and 57 of 461 HP
    _only(out, "status", {0: -4 * s16, 1: -s16, 2: -s8, 3: -s8, 7: -(1 / 8) * (2 / 3)})


def test_the_toxic_tick_ramps_past_the_old_8_cap_to_the_engines_15(eot_fe: Any, rows: torch.Tensor) -> None:
    """GIGO (`gen3_toxic_stage_scale_v1`): the obs toxic cell saturated at 8 ticks while Showdown's `tox.onResidual`
    runs the stage to 15 (`if (stage < 15) stage++` BEFORE the damage `floor(maxhp/16) · stage`), so the 9th tick on
    read as the 9th/8th. With n ticks taken the next tick is min(n + 1, 15)/16."""
    ctx = _blank(eot_fe, rows)
    for slot, n in ((0, 8), (1, 11), (2, 13), (3, 14), (4, 15)):
        ctx = _mon(ctx, slot, status="tox", tox_ticks=n)
    out = _eot(eot_fe, ctx)
    s16 = _ours("snorlax", 16)                                                    # stage × ⌊461/16⌋
    _only(out, "status", {0: -9 * s16, 1: -12 * s16, 2: -14 * s16, 3: -15 * s16, 4: -15 * s16})


def test_leech_seed_drain_and_its_heal_to_the_other_side(eot_fe: Any, rows: torch.Tensor) -> None:
    """gen3_leech_seed_exact_v1: the drain is Showdown's INTEGER ``min(hp, max(1, ⌊maxhp / 8⌋))`` (`data/moves.ts`
    leechseed ``this.damage(baseMaxhp / 8)`` → ``clampIntRange(·, 1)``), and the other side heals that amount. Their
    Snorlax's 461 / 8 = 57.625 drains 57 — the old expected amount (57.625) fails every assertion below."""
    ctx = _vol(_blank(eot_fe, rows), 1, "leechseed")                              # THEIR active (Snorlax) seeded
    ctx = _mon(ctx, 1, species="blissey")
    out = _eot(eot_fe, ctx)
    drained = 57.0                                                                # ⌊461 / 8⌋; their HP 230.5 > 57
    bliss = 2 * 255 + 31 + 110                                                    # our Blissey's exact max HP
    gain = {s: drained / 461.0 for s in range(TEAM_SIZE)}
    gain[1] = drained / bliss
    assert float(out[TEAM_SIZE, _COL["leech_drain"]]) == pytest.approx(-drained / 461.0, rel=1e-6)
    for s in range(2 * TEAM_SIZE):
        assert float(out[s, _COL["leech_gain"]]) == pytest.approx(gain.get(s, 0.0), rel=1e-6), s
        if s != TEAM_SIZE:
            assert float(out[s, _COL["leech_drain"]]) == 0.0, s
    low = _eot(eot_fe, _mon(ctx, TEAM_SIZE, hp=0.05))                              # less HP than 1/8 left: capped
    assert float(low[0, _COL["leech_gain"]]) == pytest.approx(0.05, rel=1e-6)
    assert float(low[TEAM_SIZE, _COL["leech_drain"]]) == pytest.approx(-0.05, rel=1e-6)
    ooze = _eot(eot_fe, _mon(ctx, TEAM_SIZE, species="tentacruel", types=("WATER", "POISON"), ability="liquidooze"))
    tenta = 2 * 80 + 31 + 110                                                     # 301: drains ⌊301 / 8⌋ = 37
    assert float(ooze[0, _COL["leech_gain"]]) == pytest.approx(-(tenta // 8) / 461.0, rel=1e-6)


def test_leech_seed_on_our_1_hp_mon_drains_exactly_1_hp(eot_fe: Any, rows: torch.Tensor) -> None:
    """The brief's Showdown-verified case: OUR seeded active at 1 HP (a 714-HP Blissey, 252 HP EVs) loses exactly 1
    HP — `sim/pokemon.ts` ``damage`` stops the HP at 0, so the amount dealt (and healed, `if (damage) this.heal`)
    is 1 — and their slot heals 1 HP of its 461. The obs cell is ``f32(1 / 714)``; the HP points come from
    `our_hp_points`, so the drain is the integer, not ``f32(1/714) · 714``."""
    ctx = _vol(_mon(_blank(eot_fe, rows), 0, species="blissey", hp=float(np.float32(1.0 / 714.0))), 0, "leechseed")
    pp = ctx.pokemon_part.clone()
    pp[:, 0, POKEMON_SPREAD_OFFSET + 6] = 1.0                                    # 252 HP EVs → max HP 714
    out = _eot(eot_fe, dataclasses.replace(ctx, pokemon_part=pp))
    assert float(out[0, _COL["leech_drain"]]) == pytest.approx(-1.0 / 714.0, rel=1e-6)
    for s in range(TEAM_SIZE, 2 * TEAM_SIZE):
        assert float(out[s, _COL["leech_gain"]]) == pytest.approx(1.0 / 461.0, rel=1e-6), s
    full = _eot(eot_fe, dataclasses.replace(_mon(ctx, 0, hp=1.0), pokemon_part=pp))    # full HP: ⌊714 / 8⌋ = 89
    assert float(full[0, _COL["leech_drain"]]) == pytest.approx(-89.0 / 714.0, rel=1e-6)


def test_leech_seed_drains_a_1_max_hp_shedinja_by_its_1_hp(eot_fe: Any, rows: torch.Tensor) -> None:
    """``clampIntRange(1 / 8, 1)`` = 1: a seeded Shedinja (max HP 1) loses its one HP (the Rust engine's
    `apply_leech_seed` ``(maxhp / 8).max(1)``, `gen3_wonder_guard_v1`), where the old 1/8 read it as surviving."""
    ctx = _vol(_mon(_blank(eot_fe, rows), 0, species="shedinja", types=("BUG", "GHOST"), hp=1.0), 0, "leechseed")
    out = _eot(eot_fe, ctx)
    assert float(out[0, _COL["leech_drain"]]) == -1.0
    assert float(out[TEAM_SIZE, _COL["leech_gain"]]) == pytest.approx(1.0 / 461.0, rel=1e-6)


def test_an_85_hp_ev_blissey_reads_its_floored_max_hp(eot_fe: Any, rows: torch.Tensor) -> None:
    """gen3_stat_ev_points_v1 end to end: our seeded Blissey at full HP with 85 HP EVs has max HP 672 (``tr(85 / 4)``
    = 21), drains ⌊672 / 8⌋ = 84, i.e. exactly 1/8 of its max; the old ``ev / 4`` read 672.25 (-84 / 672.25)."""
    ctx = _vol(_mon(_blank(eot_fe, rows), 0, species="blissey", hp=1.0), 0, "leechseed")
    pp = ctx.pokemon_part.clone()
    pp[:, 0, POKEMON_SPREAD_OFFSET + 6] = float(np.float32(85.0 / 252.0))
    out = _eot(eot_fe, dataclasses.replace(ctx, pokemon_part=pp))
    assert float(out[0, _COL["leech_drain"]]) == pytest.approx(-84.0 / 672.0, rel=1e-6)


def test_leech_seed_drained_is_showdowns_integer_for_every_reachable_pair() -> None:
    """EXHAUSTIVE over max HP 1..714 and HP 0..max HP: ``min(hp, max(1, maxhp // 8))`` in exact integers."""
    for maxhp in range(1, 715):
        hp = torch.arange(0, maxhp + 1, dtype=torch.float32)
        got = leech_seed_drained(torch.tensor(float(maxhp)), hp)
        want = torch.minimum(hp, torch.tensor(float(max(1, maxhp // 8))))
        assert torch.equal(got, want), maxhp


def test_residual_points_is_showdowns_integer_for_every_max_hp_and_denominator() -> None:
    """gen3_residual_exact_v1, EXHAUSTIVE over max HP 1..714 and every gen-3 residual denominator: where exact,
    ``max(1, maxhp // d)`` (`sim/battle.ts` ``spreadDamage`` ``clampIntRange(damage, 1)``; ``heal``'s ``<= 1 → 1`` then
    ``trunc``); elsewhere the nominal ``maxhp / d``. A non-power-of-two denominator is REFUSED (its quotient is not
    exact in f32)."""
    m = torch.arange(1, 715, dtype=torch.float32)
    for d in (16, 8, 4, 2):
        want = torch.from_numpy(np.maximum(1, np.arange(1, 715) // d).astype(np.float32))
        assert torch.equal(residual_points(m, float(d), torch.ones_like(m, dtype=torch.bool)), want), d
        assert torch.equal(residual_points(m, float(d), torch.zeros_like(m, dtype=torch.bool)), m / d), d
    with pytest.raises(ValueError, match="not a gen-3 residual"):
        residual_points(m, 3.0, torch.ones_like(m, dtype=torch.bool))


def test_our_amounts_are_showdowns_integers_and_theirs_the_nominal_fraction(eot_fe: Any, rows: torch.Tensor) -> None:
    """Our 714-HP Blissey (252 HP EVs) under Toxic at stage 4 loses ``clampIntRange(714 / 16, 1) · 4`` = 44 · 4 = 176
    (`data/conditions.ts` tox), with Leftovers it heals ⌊714/16⌋ = 44 (`data/mods/gen4/items.ts`), a Wish heals
    ⌊714/2⌋ = 357 (`data/mods/gen4/moves.ts` wish ``target.baseMaxhp / 2``); their Snorlax (an ESTIMATED 461) keeps
    the nominal 1/16 · 4. The old nominal fractions (178.5, 44.625, 357) fail the first two."""
    ctx = _mon(_blank(eot_fe, rows), 0, species="blissey", status="tox", tox_ticks=3, item="leftovers", hp=0.25)
    ctx = _mon(ctx, TEAM_SIZE, status="tox", tox_ticks=3)
    pp = ctx.pokemon_part.clone()
    pp[:, 0, POKEMON_SPREAD_OFFSET + 6] = 1.0                                    # 252 HP EVs → 714
    nmr = ctx.non_matchup_rest.clone()
    nmr[:, eot_fe.eot_residual_rule._wish_our] = 0.5
    out = _eot(eot_fe, dataclasses.replace(ctx, pokemon_part=pp, non_matchup_rest=nmr))
    assert float(out[0, _COL["status"]]) == pytest.approx(-176 / 714, rel=1e-6)
    assert float(out[0, _COL["leftovers"]]) == pytest.approx(44 / 714, rel=1e-6)
    assert float(out[0, _COL["wish"]]) == pytest.approx(357 / 714, rel=1e-6)
    assert float(out[TEAM_SIZE, _COL["status"]]) == pytest.approx(-4 / 16, rel=1e-6)


def test_a_full_hp_poisoned_leftovers_holder_wastes_the_heal(eot_fe: Any, rows: torch.Tensor) -> None:
    """THE HP CHAIN: Leftovers (order 10 sub 4) runs before poison (sub 6), and at full HP it heals nothing
    (`sim/battle.ts` ``heal``: ``if (target.hp >= target.maxhp) return false``), so our full-HP poisoned Snorlax nets
    the whole −57 / 461; the clamp of the sum read −29 / 461. At half HP both apply (−29)."""
    ctx = _mon(_blank(eot_fe, rows), 1, item="leftovers", status="psn", hp=1.0)
    ctx = _mon(ctx, 2, item="leftovers", status="psn", hp=float(np.float32(230 / 461)))
    out = _eot(eot_fe, ctx)
    assert float(out[1, _COL["net"]]) == pytest.approx(-57 / 461, rel=1e-6)
    assert float(out[2, _COL["net"]]) == pytest.approx(-29 / 461, rel=1e-6)


def test_leech_seed_drains_the_hp_left_after_the_residuals_before_it(eot_fe: Any, rows: torch.Tensor) -> None:
    """gen3 residual order (`data/mods/gen4/items.ts` leftovers sub 4, `data/mods/gen4/moves.ts` leechseed sub 5;
    `sim/battle.ts` ``comparePriority`` sorts order, speed, then sub-order): our seeded Snorlax at 20 / 461 HP with
    Leftovers heals to 48 and THEN loses min(48, 57) = 48 — their slot heals 48, not the decision-time 20. Under sand
    (field order 8, before both) the 28 chip KOs it first: no Leftovers (a fainted mon heals nothing), no drain."""
    ctx = _vol(_mon(_blank(eot_fe, rows), 0, item="leftovers", hp=float(np.float32(20 / 461))), 0, "leechseed")
    out = _eot(eot_fe, ctx)
    assert float(out[0, _COL["leech_drain"]]) == pytest.approx(-48 / 461, rel=1e-6)
    assert float(out[TEAM_SIZE, _COL["leech_gain"]]) == pytest.approx(48 / 461, rel=1e-6)
    assert float(out[0, _COL["net"]]) == pytest.approx(-20 / 461, rel=1e-6)
    sand = _eot(eot_fe, _weather(ctx, 3, permanent=True))
    assert float(sand[0, _COL["leech_drain"]]) == 0.0 and float(sand[TEAM_SIZE, _COL["leech_gain"]]) == 0.0
    assert float(sand[0, _COL["net"]]) == pytest.approx(-20 / 461, rel=1e-6)


def test_wish_heals_before_the_weather_chips(eot_fe: Any, rows: torch.Tensor) -> None:
    """Wish (gen 4 mod order 7) before sand (field order 8): our Snorlax at 10 / 461 heals 230, then takes 28 —
    net +202. The reverse order would KO it at the chip and void the Wish (net −10)."""
    ctx = _weather(_mon(_blank(eot_fe, rows), 0, hp=float(np.float32(10 / 461))), 3, permanent=True)
    nmr = ctx.non_matchup_rest.clone()
    nmr[:, eot_fe.eot_residual_rule._wish_our] = 0.5
    out = _eot(eot_fe, dataclasses.replace(ctx, non_matchup_rest=nmr))
    assert float(out[0, _COL["net"]]) == pytest.approx(202 / 461, rel=1e-6)


def test_a_shedinja_on_either_side_loses_its_one_hp_to_sand(eot_fe: Any, rows: torch.Tensor) -> None:
    """``clampIntRange(1 / 16, 1)`` = 1: sand KOs a 1-max-HP Shedinja (Bug / Ghost, not sand-immune) at the residual,
    ours AND theirs (its max HP is the species' 1 on either side, not an estimate); the old nominal −1/16 read it
    surviving."""
    ctx = _weather(_blank(eot_fe, rows), 3, permanent=True)
    for slot in (1, TEAM_SIZE + 1):
        ctx = _mon(ctx, slot, species="shedinja", types=("BUG", "GHOST"), ability="wonderguard", hp=1.0)
    out = _eot(eot_fe, ctx)
    for slot in (1, TEAM_SIZE + 1):
        assert float(out[slot, _COL["weather"]]) == -1.0 and float(out[slot, _COL["net"]]) == -1.0, slot


def test_wish_heals_every_mon_of_its_side_by_half(eot_fe: Any, rows: torch.Tensor) -> None:
    ctx = _blank(eot_fe, rows)
    nmr = ctx.non_matchup_rest.clone()
    nmr[:, eot_fe.eot_residual_rule._wish_opp] = 0.5
    out = _eot(eot_fe, dataclasses.replace(ctx, non_matchup_rest=nmr))
    _only(out, "wish", {s: 0.5 for s in range(TEAM_SIZE, 2 * TEAM_SIZE)})
    assert float(out[TEAM_SIZE, _COL["net"]]) == pytest.approx(0.5)                # HP 0.5 + 0.5 = full


def test_ingrain_curse_and_nightmare_on_the_active_only(eot_fe: Any, rows: torch.Tensor) -> None:
    base = _blank(eot_fe, rows)
    _only(_eot(eot_fe, _vol(base, 0, "ingrain")), "ingrain", {0: _ours("snorlax", 16)})
    _only(_eot(eot_fe, _vol(base, 1, "curse")), "curse", {TEAM_SIZE: -1 / 4})          # theirs: nominal
    _only(_eot(eot_fe, _vol(base, 0, "curse")), "curse", {0: -_ours("snorlax", 4)})     # ours: ⌊461/4⌋ = 115 HP
    nm = _mon(_vol(base, 0, "nightmare"), 0, status="slp", p_wake=0.25)
    _only(_eot(eot_fe, nm), "nightmare", {0: -_ours("snorlax", 4) * 0.75})
    _only(_eot(eot_fe, _vol(base, 0, "nightmare")), "nightmare", {})                # awake: no Nightmare


def test_the_net_is_clamped_to_the_hp_the_mon_has_and_lacks(eot_fe: Any, rows: torch.Tensor) -> None:
    """Our Snorlax at 46 / 461 HP takes Curse's 115: it loses the 46 it has; at 452 / 461 Leftovers' 28 heals the 9
    it lacks."""
    ctx = _vol(_mon(_blank(eot_fe, rows), 0, hp=float(np.float32(46 / 461))), 0, "curse")
    ctx = _mon(ctx, 1, item="leftovers", hp=float(np.float32(452 / 461)))
    out = _eot(eot_fe, ctx)
    assert float(out[0, _COL["net"]]) == pytest.approx(-46 / 461, rel=1e-6)
    assert float(out[1, _COL["net"]]) == pytest.approx(9 / 461, rel=1e-6)


def test_fainted_and_unknown_slots_read_zero(eot_fe: Any, rows: torch.Tensor) -> None:
    ctx = _weather(_blank(eot_fe, rows), 3, permanent=True)
    ctx = _mon(ctx, 3, hp=0.0)
    with torch.no_grad():
        conc = torch.ones(1, TEAM_SIZE)
        conc[:, 2] = 0.0
        out = eot_fe.eot_residual_rule(ctx, eot_fe.damage_op, conc)[0]
    assert float(out[3].abs().max()) == 0.0 and float(out[TEAM_SIZE + 2].abs().max()) == 0.0
    assert float(out[4, _COL["weather"]]) == pytest.approx(-_ours("snorlax", 16))


def test_the_residual_permutes_with_our_slots(eot_fe: Any, rows: torch.Tensor) -> None:
    ctx = eot_fe.unpack({"observation": rows})
    sig = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        base = eot_fe.eot_residual_rule(ctx, eot_fe.damage_op)
        perm = eot_fe.eot_residual_rule(_permute_ours(ctx, sig), eot_fe.damage_op)
    assert float(base.abs().max()) > 0.0
    assert torch.allclose(perm[:, :TEAM_SIZE], base[:, :TEAM_SIZE][:, sig], atol=1e-7)
    assert torch.allclose(perm[:, TEAM_SIZE:], base[:, TEAM_SIZE:], atol=1e-7)


def test_the_residual_reaches_the_trunk_through_its_zero_init_projection(rows: torch.Tensor) -> None:
    on = copy.deepcopy(_policy())
    proj = on.features_extractor.eot_residual_proj
    assert proj is not None and proj.bias is None and float(proj.weight.abs().max()) == 0.0
    base = _forward_record(on, rows)
    with torch.no_grad():
        proj.weight.copy_(torch.randn(proj.weight.shape, generator=torch.Generator().manual_seed(2)) * 0.1)
    live = _forward_record(on, rows)
    assert not torch.equal(base[2], live[2]), "a live projection did not reach the team tokens"
    # (no op, no residual: `extractor_build` builds it `if damage_op`; the no-op build is the toy, whose load path
    # `adopted_levers_test.py` exercises)


# ============================================================================== THE COMBINED ARM'S LEVERS TOGETHER
@pytest.mark.slow  # ship gate (2026-10-19): 24 s beside a live run
def test_all_levers_together_start_as_the_static_network(rows: torch.Tensor) -> None:
    """The static-recovery levers that still have an OFF value (the two static facts and the trunk depth; the hazard
    block and the residual are built with the op since U1), on together, at init equal `static` with them off — each
    lever's parameters are its own, zero where its identity needs it."""
    off = _policy()
    on = _policy(mon_hazard_cost="on", move_actor_state="on", trunk_layers=3)
    _one_lever(off, on, ("mon_hazard_proj.", "move_actor_proj.", "extra_rounds.0."))
    a, b = _forward_record(off, rows), _forward_record(on, rows)
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x, y), f"record {i}"


# ===================================================================================================== VERSIONING
def test_versioning_migrates_and_gates_the_trunk_depth() -> None:
    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError
    from agents.model.model_version.migrations import _migrate_config
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    assert MODEL_CONFIG_VERSION >= 150
    out = _migrate_config({"config_version": 149, "token_encoding": "static"})
    assert out["trunk_layers"] == 2 and "switch_hazard_cost" not in out and "eot_residual" not in out
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    mv = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    assert mv.trunk_layers == 2
    for name, val in (("trunk_layers", 3),):
        with pytest.raises(ModelVersionError, match=name):
            mv.check_compatible(dataclasses.replace(mv, **{name: val}))
        rec = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {name: val}})
        assert getattr(rec, name) == val
        assert getattr(current_model_version(load_mappings(), **{name: val}), name) == val
