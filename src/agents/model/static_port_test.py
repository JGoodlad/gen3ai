"""`gen3_static_port_v1` (config v147) — `--token-encoding static` at HEAD: the two EQUIVARIANCE fixes and the two
NARROW facts (`--mon-hazard-cost`, `--move-actor-state`; `designs/endstate/design_static_tokens.md` §12).

Built through a REAL SB3 policy (the construction path training uses — the identity_init_test rule) on the
PRODUCTION toggles with `token_encoding='static'`, over REAL observation rows (the compile parity fixture).

Each test names what a revert breaks:

* the Spikes entry cost on CONSTRUCTED boards (0–3 layers; Flying; Levitate known on our side; an opponent's
  Levitate revealed, or unknown under its species' Smogon prior — and a top-1 prior id that is NOT a reveal), both
  sides reading their own side's Spikes (fails on a wrong fraction, a side swap, an ungrounded Flying / Levitate mon,
  or an unknown ability read as revealed);
* ONE rule: the `x` edge cell and the per-mon fact both read `DamageOperator.spikes_entry` (fails if either
  re-derives the rule — the two copies could drift);
* the move-token input carries our active's HP: changing it moves every VALID E3 seat by exactly the projection of
  the change, and never an invalid one (fails if the fact is dropped or lands on the wrong seats); with the flag off
  the E3 seats do not read HP at all (the control: static's move tokens carry none);
* EQUIVARIANCE over our team slots, with each flag on (every zero-init route planted live): permuting our six slots
  permutes the trunk's input mon tokens exactly and leaves THEIR tokens and the E3 seats unchanged (fails on any
  position-keyed weight on that axis);
* the two equivariance fixes: S reads the type PAIR as a set (swapping type1 / type2 leaves S bit-identical — fails on
  the concat), and the op content's outgoing route is a set function of our moves (permuting the request order
  leaves it unchanged; an all-zero move cell contributes exactly 0 — fails on the request-ordered Linear);
* the ONE-LEVER init: each flag ON adds exactly its one zero-init, bias-free projection, every other initial
  parameter byte equals the OFF build's at the same seed, and the forward at init is bit-identical (fails on an
  RNG-drawing init, a module built out of order, a non-zero projection);
* versioning: a v146 `static` record is refused; a v146 `legacy` one migrates with both facts `off`; a mismatch of
  either fact is refused by `check_compatible`.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Tuple

import pytest
import torch

from agents.model.board_tokens import OPC_OUTGOING_CELL, OpContent
from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.damage_tables import _T2I
from agents.model.extractor_ctx import POKEMON_ABILITY_KNOWN_OFFSET, ExtractorContext, active_request_sorted_match
from agents.model.identity_init_test import _build_real_policy
from agents.model.static_facts import MON_HAZARD_DIM, MOVE_ACTOR_DIM, mon_hazard_features, move_actor_features
from agents.observation.constants import POKEMON_HP_OFFSET, TEAM_SIZE
from utils.paths import repo_path

#: fp32 bound on the same arithmetic in a different order (a permuted sum, a reassociated matmul).
FP32_ATOL = 1e-5
_CHIP = (0.0, 1.0 / 8, 1.0 / 6, 1.0 / 4)
_FACT_PROJ = ("mon_hazard_proj", "move_actor_proj")


def _static_toggles(**over: Any) -> Dict[str, Any]:
    with open(repo_path("designs", "production_config.json")) as fh:
        cfg = json.load(fh)
    tog = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    tog.update(token_encoding="static", **over)
    return tog


_CACHE: Dict[str, Any] = {}


def _policy(**over: Any) -> Tuple[Any, Any]:
    key = json.dumps(over, sort_keys=True)
    if key not in _CACHE:
        _CACHE[key] = _build_real_policy(**_static_toggles(**over))
    return _CACHE[key]


@pytest.fixture(scope="module")
def rows() -> torch.Tensor:
    _m, enc = _policy()
    obs, _mask = load_parity_rows(enc.dimension)
    return torch.as_tensor(obs)


def _ctx(fe: Any, x: torch.Tensor) -> ExtractorContext:
    return fe.unpack({"observation": x})  # type: ignore[no-any-return]


# ------------------------------------------------------------------------------- fact A: the Spikes entry cost
def _species_num(name: str) -> int:
    from agents import gen3_data
    return int(gen3_data.species.get(name).num)


def _levitate() -> int:
    from agents import gen3_data
    return int(gen3_data.abilities.get("levitate").num)


def _plant(ctx: ExtractorContext, slot: int, species: str, types: Tuple[str, ...], ability: int,
           known: bool, revealed: bool = True) -> ExtractorContext:
    """Put ``species`` with ``types`` and ability id ``ability`` (``known`` = the reveal flag) at ``slot`` (0–11)."""
    sp, t1, t2, a1, pp = (ctx.species_ids.clone(), ctx.type1_ids.clone(), ctx.type2_ids.clone(),
                          ctx.ability1_ids.clone(), ctx.pokemon_part.clone())
    tt = sorted(types) + ["NONE"] * (2 - len(types))
    sp[:, slot] = _species_num(species)
    t1[:, slot] = _T2I[tt[0]]
    t2[:, slot] = _T2I.get(tt[1], 0) if tt[1] != "NONE" else 0
    a1[:, slot] = ability
    pp[:, slot, POKEMON_ABILITY_KNOWN_OFFSET] = 1.0 if known else 0.0
    bm = ctx.opp_believed_mask.clone()
    if slot >= TEAM_SIZE:
        bm[:, slot - TEAM_SIZE] = not revealed
    return dataclasses.replace(ctx, species_ids=sp, type1_ids=t1, type2_ids=t2, ability1_ids=a1, pokemon_part=pp,
                               opp_believed_mask=bm)


def _constructed_board(fe: Any, rows: torch.Tensor) -> Tuple[ExtractorContext, torch.Tensor]:
    """Four rows whose (our, their) Spikes layers are (0,1) (1,2) (2,3) (3,0), and twelve planted mons with their
    EXPECTED grounded bit (1 = takes the chip)."""
    ctx = _ctx(fe, rows[:4])
    snorlax_abil = 0      # any non-Levitate id; Snorlax's prior is Immunity / Thick Fat (P(Levitate) = 0)
    lev = _levitate()
    plan = [
        # ours: exact abilities
        (0, "skarmory", ("STEEL", "FLYING"), snorlax_abil, True, True, 0.0),       # Flying: immune
        (1, "snorlax", ("NORMAL",), snorlax_abil, True, True, 1.0),                 # grounded
        (2, "claydol", ("GROUND", "PSYCHIC"), lev, True, True, 0.0),                # Levitate known
        (3, "gengar", ("GHOST", "POISON"), lev, True, True, 0.0),                   # Levitate known
        (4, "snorlax", ("NORMAL",), snorlax_abil, True, True, 1.0),
        (5, "snorlax", ("NORMAL",), snorlax_abil, True, True, 1.0),
        # theirs: the belief where the ability is unrevealed
        (6, "gengar", ("GHOST", "POISON"), lev, False, True, 0.0),                  # unknown, prior P(Lev) = 1
        (7, "snorlax", ("NORMAL",), lev, False, True, 1.0),                         # a top-1 id is NOT a reveal
        (8, "skarmory", ("STEEL", "FLYING"), snorlax_abil, False, True, 0.0),       # Flying: immune
        (9, "weezing", ("POISON",), lev, True, True, 0.0),                          # Levitate revealed
        (10, "snorlax", ("NORMAL",), snorlax_abil, True, True, 1.0),                # revealed, grounded
        (11, "snorlax", ("NORMAL",), snorlax_abil, False, False, 0.0),              # UNREVEALED: fraction 0
    ]
    grounded = torch.zeros(2 * TEAM_SIZE)
    for slot, sp, ty, ab, known, revealed, g in plan:
        ctx = _plant(ctx, slot, sp, ty, ab, known, revealed)
        grounded[slot] = g
    layers = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=ctx.spikes_feature.dtype)
    return dataclasses.replace(ctx, spikes_feature=layers / 3.0), grounded


def test_the_spikes_entry_cost_on_constructed_boards(rows: torch.Tensor) -> None:
    fe = _policy(mon_hazard_cost="on")[0].policy.features_extractor
    ctx, grounded = _constructed_board(fe, rows)
    with torch.no_grad():
        f = mon_hazard_features(fe.damage_op, ctx)                                    # [4,12,2]
    assert f.shape == (4, 2 * TEAM_SIZE, MON_HAZARD_DIM)
    layers = (ctx.spikes_feature * 3).round().long()
    for b in range(4):
        for slot in range(2 * TEAM_SIZE):
            side = 0 if slot < TEAM_SIZE else 1
            want_layers = float(ctx.spikes_feature[b, side])
            want_chip = _CHIP[int(layers[b, side])] * float(grounded[slot])
            assert float(f[b, slot, 0]) == pytest.approx(want_layers, abs=1e-7), (b, slot, "layers column")
            assert float(f[b, slot, 1]) == pytest.approx(want_chip, abs=1e-7), (b, slot, "chip column")


def test_the_spikes_entry_reads_base_types_not_the_current_ones(rows: torch.Tensor) -> None:
    """GIGO fix 2026-10-09: the obs type columns hold a mon's CURRENT types (Color Change, Transform, ...), but a
    switch-IN runs after `clearVolatile` reverted them to the species' (`deps/pokemon-showdown` `sim/pokemon.ts`:
    `setSpecies(baseSpecies)`). Planted: a Kecleon whose CURRENT type is Flying (base Normal) PAYS the chip, on both
    sides; a Ditto Transformed into Skarmory (current Steel/Flying, base Normal) PAYS; a real Flying-base Skarmory does
    NOT even when its current columns read Normal -- the species decides. Fails on a revert to the `type1_ids` /
    `type2_ids` read."""
    fe = _policy(mon_hazard_cost="on")[0].policy.features_extractor
    ctx = _ctx(fe, rows[:1])
    plan = [  # (slot, species, CURRENT types, expected: pays the chip)
        (0, "kecleon", ("FLYING",), True),
        (1, "ditto", ("STEEL", "FLYING"), True),
        (2, "skarmory", ("NORMAL",), False),
        (3, "skarmory", ("STEEL", "FLYING"), False),
        (4, "snorlax", ("NORMAL",), True),
        (6, "kecleon", ("FLYING",), True),
        (7, "skarmory", ("NORMAL",), False),
        (8, "skarmory", ("STEEL", "FLYING"), False),
    ]
    for slot, sp, ty, _pays in plan:
        ctx = _plant(ctx, slot, sp, ty, ability=0, known=True)
    layers = torch.tensor([[2, 3]], dtype=ctx.spikes_feature.dtype)
    ctx = dataclasses.replace(ctx, spikes_feature=layers / 3.0)
    with torch.no_grad():
        f = mon_hazard_features(fe.damage_op, ctx)
    for slot, sp, ty, pays in plan:
        want = (_CHIP[2] if slot < TEAM_SIZE else _CHIP[3]) if pays else 0.0
        assert float(f[0, slot, 1]) == pytest.approx(want, abs=1e-7), (slot, sp, ty)


def test_the_spikes_entry_reads_the_species_levitate_not_the_current_ability(rows: torch.Tensor) -> None:
    """GIGO fix 2026-10-09 (Levitate half): a switch-in resets the ability to the BASE one (`clearVolatile`), while
    Trace / Role Play / Skill Swap / Transform change only the CURRENT one -- which is what the ability column holds.
    A Gardevoir that Traced a Levitate user (column reads Levitate) PAYS; a real Levitate SPECIES (Gengar, Latias,
    Flygon, Weezing, Misdreavus) does NOT, whatever its column reads, revealed or not, on both sides. Fails on a
    revert to the `ability1_ids` / revealed-ability read."""
    fe = _policy(mon_hazard_cost="on")[0].policy.features_extractor
    ctx = _ctx(fe, rows[:1])
    lev, other = _levitate(), 0
    plan = [  # (slot, species, types, ability column, known flag, revealed, expected: pays the chip)
        (0, "gardevoir", ("PSYCHIC",), lev, True, True, True),                 # Traced Levitate: pays
        (1, "gengar", ("GHOST", "POISON"), other, True, True, False),
        (2, "latias", ("DRAGON", "PSYCHIC"), other, True, True, False),
        (3, "flygon", ("DRAGON", "GROUND"), other, True, True, False),
        (4, "weezing", ("POISON",), other, True, True, False),
        (5, "misdreavus", ("GHOST",), other, True, True, False),
        (6, "gardevoir", ("PSYCHIC",), lev, True, True, True),                 # revealed Traced Levitate: pays
        (7, "gengar", ("GHOST", "POISON"), other, True, True, False),          # revealed non-Levitate column: immune
        (8, "weezing", ("POISON",), other, False, True, False),                # unrevealed: the species decides
        (9, "gardevoir", ("PSYCHIC",), lev, False, True, True),                # unrevealed id1 = a prior, not a reveal
    ]
    for slot, sp, ty, ab, known, revealed, _pays in plan:
        ctx = _plant(ctx, slot, sp, ty, ab, known, revealed)
    ctx = dataclasses.replace(ctx, spikes_feature=torch.tensor([[2, 3]], dtype=ctx.spikes_feature.dtype) / 3.0)
    with torch.no_grad():
        f = mon_hazard_features(fe.damage_op, ctx)
    for slot, sp, _ty, ab, known, _rev, pays in plan:
        want = (_CHIP[2] if slot < TEAM_SIZE else _CHIP[3]) if pays else 0.0
        assert float(f[0, slot, 1]) == pytest.approx(want, abs=1e-7), (slot, sp, ab, known)


def test_every_gen3_species_levitate_prior_is_zero_or_one() -> None:
    """The premise of the species read: in gen 3 Levitate is the SOLE ability of its species, so the species'
    Smogon P(Levitate) is exactly 0 or 1 (17 species). A species holding Levitate beside another ability would make the
    species read inexact for a revealed mon and need the known-ability fallback (fails here if the data ever gives one)."""
    from agents.model.damage_tables import build_trap_tables
    from agents import gen3_data
    t = build_trap_tables(1024, 512)["SPECIES_TRAP_PRIOR"][:, 3]
    assert set(t.tolist()) <= {0.0, 1.0}, sorted(set(t.tolist()))
    for sid in ("gengar", "latias", "flygon", "weezing", "misdreavus"):
        assert float(t[gen3_data.species.get(sid).num]) == 1.0, sid
    assert float(t[gen3_data.species.get("gardevoir").num]) == 0.0


def test_the_x_cell_and_the_per_mon_fact_read_one_entry_rule(rows: torch.Tensor) -> None:
    """Agreement on the constructed boards (the `x` cell gates by alive / revealed, the fact does not), and the
    single SOURCE: a planted `spikes_entry` reaches both."""
    fe = _policy(mon_hazard_cost="on")[0].policy.features_extractor
    op = fe.damage_op
    ctx, _g = _constructed_board(fe, rows)
    with torch.no_grad():
        fe({"observation": rows[:4]})                                                 # stash the belief logits
        logits = fe.last_move_belief_logits
        op.stash.x5 = None                                                            # price on ``ctx`` itself
        our_x, opp_x = op.pairwise_entry(ctx, logits)
        f = mon_hazard_features(op, ctx)
    alive_i = (ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0).float()
    alive_j = (ctx.hp_and_active[:, TEAM_SIZE:, 0] > 0).float()
    revealed_j = (~ctx.opp_believed_mask).float()
    assert torch.equal(our_x[..., 0], f[:, :TEAM_SIZE, 1] * alive_i)
    assert torch.equal(opp_x[..., 0], f[:, TEAM_SIZE:, 1] * alive_j * revealed_j)
    planted = tuple(torch.full((4, TEAM_SIZE), v) for v in (0.11, 0.07, 0.5, 0.25))
    op.spikes_entry = lambda _ctx: planted                                            # instance shadow
    try:
        with torch.no_grad():
            our_x2, _ = op.pairwise_entry(ctx, logits)
            f2 = mon_hazard_features(op, ctx)
    finally:
        del op.spikes_entry
    assert torch.equal(our_x2[..., 0], planted[0] * alive_i)
    assert torch.equal(f2[:, :TEAM_SIZE, 1], planted[0]) and torch.equal(f2[:, TEAM_SIZE:, 1], planted[1] * revealed_j)


# ------------------------------------------------------------------------------- fact B: the actor's state on E3
def _e3_seats(fe: Any, x: torch.Tensor, hp_scale: float = 1.0) -> torch.Tensor:
    """[B,4,D] the E3 seats as they ENTER the trunk, with our active's HP scaled by ``hp_scale``."""
    seen: List[torch.Tensor] = []

    def scaled(_m: Any, _a: Any, ctx: ExtractorContext) -> ExtractorContext:
        pp = ctx.pokemon_part.clone()
        ar = torch.arange(ctx.batch_size)
        pp[ar, ctx.our_active_idx, POKEMON_HP_OFFSET] = pp[ar, ctx.our_active_idx, POKEMON_HP_OFFSET] * hp_scale
        return dataclasses.replace(ctx, pokemon_part=pp)

    h = fe.team_transformer.register_forward_pre_hook(
        lambda _m, _a, kw: seen.append(kw["extra"][0][:, :4].detach().clone()), with_kwargs=True)
    hu = fe.unpack.register_forward_hook(scaled)
    try:
        with torch.no_grad():
            fe({"observation": x})
    finally:
        h.remove()
        hu.remove()
    assert len(seen) == 1
    return seen[0]


def test_the_move_tokens_carry_our_active_hp(rows: torch.Tensor) -> None:
    off = _policy()[0].policy.features_extractor
    assert off.move_actor_proj is None
    assert torch.equal(_e3_seats(off, rows), _e3_seats(off, rows, 0.5)), \
        "CONTROL: with the flag off, static's E3 seats read no HP"
    fe = _policy(move_actor_state="on")[0].policy.features_extractor
    w = fe.move_actor_proj.weight
    with torch.no_grad():
        w.copy_(torch.randn(w.shape, generator=torch.Generator().manual_seed(3)))
    try:
        base, half = _e3_seats(fe, rows), _e3_seats(fe, rows, 0.5)
        ctx = _ctx(fe, rows)
        ar = torch.arange(ctx.batch_size)
        hp = ctx.pokemon_part[ar, ctx.our_active_idx, POKEMON_HP_OFFSET]
        valid = active_request_sorted_match(ctx).any(-1)                            # [B,4] THE seat validity rule
        assert bool((~valid).any()) and bool(valid.any()), "PRECONDITION: valid AND invalid E3 seats"
        want = (-0.5 * hp)[:, None, None] * w[:, 0][None, None, :]                   # Δ = W[:,hp] · ΔHP
        d = half - base
        assert bool((hp > 0).any()), "PRECONDITION: a live active"
        for b in range(ctx.batch_size):
            for k in range(4):
                if valid[b, k]:
                    assert torch.allclose(d[b, k], want[b, 0], atol=FP32_ATOL), (b, k)
                else:
                    assert float(d[b, k].abs().max()) == 0.0, ("an invalid seat moved", b, k)
        f = move_actor_features(ctx)
        assert f.shape == (ctx.batch_size, MOVE_ACTOR_DIM) and torch.equal(f[:, 0], hp)
    finally:
        with torch.no_grad():
            w.zero_()


# --------------------------------------------------------------------------- equivariance over our team slots
_PER_MON = ("pokemon_part", "species_ids", "all_move_ids", "all_move_type_ids", "item_ids", "ability1_ids",
            "ability2_ids", "type1_ids", "type2_ids", "hp_probs", "hp_and_active", "last_move_ids")


def _permute_ours(ctx: ExtractorContext, sig: torch.Tensor) -> ExtractorContext:
    """New slot i holds old slot sig[i], on every per-mon field and every index into our team."""
    order = torch.cat([sig, torch.arange(TEAM_SIZE, 2 * TEAM_SIZE)])
    inv = torch.argsort(sig)
    ch: Dict[str, Any] = {n: getattr(ctx, n)[:, order] for n in _PER_MON}
    ch["our_active_idx"] = inv[ctx.our_active_idx]
    ch["fainted_mask_ours"] = ctx.fainted_mask_ours[:, sig]
    if ctx.pair_history is not None:
        ch["pair_history"] = ctx.pair_history[:, :, sig]                              # (opp i, our j)
    return dataclasses.replace(ctx, **ch)


def _plant_live(fe: Any, seed: int = 5) -> None:
    """Every all-zero (zero-init) Linear / IsolatedLinear in the extractor gets random weights, so every route —
    the op content, the two facts, prefuse — is live in the comparison."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _n, m in fe.named_modules():
            wt = getattr(m, "weight", None)
            if isinstance(wt, torch.nn.Parameter) and wt.dim() == 2 and not bool(wt.any()):
                wt.copy_(torch.randn(wt.shape, generator=g) * 0.1)


def _trunk_inputs(fe: Any, x: torch.Tensor, sig: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
    seen: List[Tuple[torch.Tensor, torch.Tensor]] = []
    hu = (fe.unpack.register_forward_hook(lambda _m, _a, ctx: _permute_ours(ctx, sig)) if sig is not None
          else None)
    h = fe.team_transformer.register_forward_pre_hook(
        lambda _m, a, kw: seen.append((a[0].detach().clone(), kw["extra"][0][:, :4].detach().clone())),
        with_kwargs=True)
    try:
        with torch.no_grad():
            fe({"observation": x})
    finally:
        h.remove()
        if hu is not None:
            hu.remove()
    assert len(seen) == 1
    return seen[0]


@pytest.mark.parametrize("flags", [{}, {"mon_hazard_cost": "on"}, {"move_actor_state": "on"},
                                   {"mon_hazard_cost": "on", "move_actor_state": "on"}],
                         ids=["static", "hazard", "actor", "both"])
def test_permuting_our_team_slots_permutes_the_trunk_input_exactly(rows: torch.Tensor, flags: Dict[str, str]) -> None:
    import copy
    fe = copy.deepcopy(_policy(**flags)[0].policy.features_extractor)
    _plant_live(fe)
    fe.eval()
    sig = torch.tensor([3, 0, 5, 1, 4, 2])
    base, e3 = _trunk_inputs(fe, rows)
    perm, e3p = _trunk_inputs(fe, rows, sig)
    assert float(base[:, :TEAM_SIZE].std()) > 0.0
    assert torch.allclose(perm[:, :TEAM_SIZE], base[:, :TEAM_SIZE][:, sig], atol=FP32_ATOL), \
        "our mon tokens did not permute with our slots"
    assert torch.allclose(perm[:, TEAM_SIZE:], base[:, TEAM_SIZE:], atol=FP32_ATOL), "their tokens moved"
    assert torch.allclose(e3p, e3, atol=FP32_ATOL), "the E3 seats moved"


# ------------------------------------------------------------------------------ the two equivariance fixes
def test_static_identity_reads_the_type_pair_as_a_set(rows: torch.Tensor) -> None:
    fe = _policy()[0].policy.features_extractor
    ctx = _ctx(fe, rows)
    sw = dataclasses.replace(ctx, type1_ids=ctx.type2_ids, type2_ids=ctx.type1_ids)
    assert bool((ctx.type1_ids != ctx.type2_ids).any()), "PRECONDITION: some dual-type / mono-type mon"
    with torch.no_grad():
        a = fe.pokemon_encoder.parts(ctx, fe.embeddings).static
        b = fe.pokemon_encoder.parts(sw, fe.embeddings).static
    assert torch.equal(a, b), "S read the type pair in order (the alphabetical concat)"


def test_the_outgoing_op_content_is_a_set_function_of_our_moves() -> None:
    oc = OpContent(outgoing=True)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        oc.outgoing_proj.weight.copy_(torch.randn(oc.outgoing_proj.weight.shape, generator=g))
    x = (torch.rand(2, TEAM_SIZE, 4, generator=g), torch.rand(2, TEAM_SIZE, 4, generator=g))
    gc = (torch.rand(2, TEAM_SIZE, 4, generator=g), torch.rand(2, TEAM_SIZE, 4, generator=g))
    d1 = torch.rand(2, 4, TEAM_SIZE, OPC_OUTGOING_CELL, generator=g)
    with torch.no_grad():
        base = oc(x, gc, d1)
        perm = oc(x, gc, d1[:, [2, 0, 3, 1]])
        three = oc(x, gc, torch.cat([d1[:, :3], torch.zeros_like(d1[:, 3:])], dim=1))
    assert torch.allclose(perm, base, atol=1e-6), "the outgoing route read our moves' request order"
    assert float((base[:, TEAM_SIZE:] - base[:, :TEAM_SIZE]).abs().max()) > 0.0
    # an all-zero (empty / illegal) request slot contributes EXACTLY nothing: the 3-move set == 4 with a zero cell
    with torch.no_grad():
        per = torch.relu(oc.outgoing_cell(d1[:, :3].permute(0, 2, 1, 3))).sum(dim=2)          # 3 moves only
        want = oc.amount_proj(torch.cat([x[1], gc[1]], -1)) + oc.outgoing_proj(per)
    assert torch.allclose(three[:, TEAM_SIZE:], want, atol=1e-6)


# --------------------------------------------------------------------------------------- the one-lever init
def test_each_fact_adds_only_its_zero_projection_and_the_init_forward_is_identical(rows: torch.Tensor) -> None:
    off = _policy()[0].policy
    on = _policy(mon_hazard_cost="on", move_actor_state="on")[0].policy
    so, sn = off.state_dict(), on.state_dict()
    extra = sorted(set(sn) - set(so))
    assert not set(so) - set(sn)
    assert extra and all(k.rsplit(".", 2)[-2] in _FACT_PROJ and k.endswith(".weight") for k in extra), extra
    for k in extra:
        assert float(sn[k].abs().max()) == 0.0, f"{k} is not zero at init"
    moved = [k for k in so if not torch.equal(so[k], sn[k])]
    assert not moved, f"a fact flag moved other initial bytes: {moved[:5]}"
    with torch.no_grad():
        a = off.features_extractor({"observation": rows})
        b = on.features_extractor({"observation": rows})
    for ta, tb in zip(a, b):
        assert torch.equal(ta, tb), "ON at init is not the OFF forward"


# ----------------------------------------------------------------------------------------------- versioning
def test_versioning_refuses_a_pre_port_static_record_and_gates_both_facts() -> None:
    from agents.model.model_version import ModelVersion, ModelVersionError, MODEL_CONFIG_VERSION
    from agents.model.model_version.migrations import _migrate_config
    assert MODEL_CONFIG_VERSION >= 147
    with pytest.raises(ModelVersionError, match="static port"):
        _migrate_config({"config_version": 146, "token_encoding": "static"})
    out = _migrate_config({"config_version": 146, "token_encoding": "legacy"})
    assert out["mon_hazard_cost"] == "off" and out["move_actor_state"] == "off" and out["config_version"] >= 147
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    mv = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    assert mv.mon_hazard_cost == "off" and mv.move_actor_state == "off"
    for name in ("mon_hazard_cost", "move_actor_state"):
        other = dataclasses.replace(mv, **{name: "on"})
        with pytest.raises(ModelVersionError, match=name):
            mv.check_compatible(other)
        rec = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {name: "on"}})
        assert getattr(rec, name) == "on"                                            # RECORDED from the kwargs
        assert getattr(current_model_version(load_mappings(), **{name: "on"}), name) == "on"  # the worker gate


def test_a_hidden_slot_is_priced_as_its_hypothesis(rows: torch.Tensor, monkeypatch: Any) -> None:
    """Under X5 the per-mon fact reads the context the op prices with: a HIDDEN opponent slot carries its hypothesis
    species (types, Levitate prior) and counts as concrete — never the empty row's unknown types."""
    import agents.model.extractor_forward as EF
    fe = _policy(mon_hazard_cost="on")[0].policy.features_extractor
    seen: List[Tuple[Any, Any]] = []
    real = EF.mon_hazard_features

    def rec(op: Any, ctx: Any, conc: Any = None) -> torch.Tensor:
        seen.append((ctx, conc))
        return real(op, ctx, conc)

    monkeypatch.setattr(EF, "mon_hazard_features", rec)
    with torch.no_grad():
        fe({"observation": rows})
    assert len(seen) == 1
    ctx, conc = seen[0]
    hs = fe.stash.hypothesis
    hidden = hs.slot_is_hypothesis
    assert bool(hidden.any()), "PRECONDITION: a hidden opponent slot"
    assert torch.equal(ctx.species_ids[:, TEAM_SIZE:][hidden], hs.slot_species[hidden].to(ctx.species_ids.dtype))
    assert bool((ctx.species_ids[:, TEAM_SIZE:][hidden] > 0).all())
    assert conc is not None and bool((conc[hidden] == 1).all())
