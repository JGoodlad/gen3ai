"""`gen3_endstate_facts_v1` (config v152) — the FACT-COMPLETION facts (`designs/endstate/design_hand_computed_features.md`
§1's FACT / JUDGMENT test, §4 ranks 3-5, §5 rank 1, finding 7). Three are the damage operator's only physics since the
legacy deletion's U1 deleted their flags and OFF paths (`designs/ops/legacy_deletion_manifest_2026-10-12.md` D1-D3);
two are still flags, OFF in the LEGACY base (`legacy_production_config`, the pre-adoption network; ON in production):

* the exact P(KO) (`ko_exact.py`; was `--ko-ramp exact`): over the 16 gen-3 rolls + the move's crit chance, each
  roll's KO resolved over the observed HP interval — at every site (`DamageOperator._rolls`, the Choice-Band `ko_cb`,
  intent_threshold's Substitute break, move resolution's Pursuit KO);
* the exact status facts (was `--status-facts exact`): the expected burn damage lost and the expected outspeed lost to
  paralysis in the pair-outcome grid's columns 12-13, and every mon's cure-availability flags as token content;
* `--move-resolution-facts full`: the facts move resolution dropped (Focus Punch / Substitute / Endure / Endeavor,
  `spin_value_lost`, the spin-denial stake), restored;
* `--drop-progress-clock on`: the model reads `turns_since_progress` as 0;
* the ONE end-of-turn rule (was `--g-ledger eot`): the op's `g` cell reads `eot_residual.EotResidualRule`.

What a revert breaks, test by test, is in each docstring. Real SB3 policies (the construction path training uses) over
the compile-parity fixture's real rows, plus constructed boards / constructed `MoveResolutionOps` for the physics.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
from typing import Any, Dict, List

import pytest
import torch

from agents import gen3_data
from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.identity_init_test import _build_real_policy
from agents.model.ko_exact import (CRIT_P_BASE, MEAN_ROLL, ROLLS, crit_p_table, ko_given_hit, opp_hp_bounds,
                                   ours_hp_bounds, roll_ko_prob)
from agents.model.move_resolution import (MoveResolutionCell, restored_move_facts, restored_switch_facts,
                                          status_move_facts, status_switch_facts)
from agents.model.move_resolution_rules import (MOVE_RESOLUTION_RESTORED_MOVE_COORDS as RMC,
                                                MOVE_RESOLUTION_RESTORED_SWITCH_COORDS as RSC)
from agents.model.move_resolution_test import FLAG_T, KIND_T, hit_grid, mf, num, ops, with_vol
from agents.model.move_resolution import move_facts, switch_facts
from agents.model.pair_outcome import PAIR_OUTCOME_IDX, STATUS_FACT_COORDS
from agents.model.static_port_test import _permute_ours
from agents.model.status_facts import CURE_DIM, CURE_FACTS, CureFlags
from agents.observation.constants import (POKEMON_CONDITION_OFFSET, POKEMON_ITEMS_OFFSET, TEAM_SIZE)
from main.train.arch_arms import legacy_production_config
from utils.paths import repo_path

_LEVERS = ("move_resolution_facts", "drop_progress_clock")
_ON = {"move_resolution_facts": "full", "drop_progress_clock": "on"}
_OFF = {"move_resolution_facts": "off", "drop_progress_clock": "off"}
#: the move-resolution bundle the restored facts live in (the legacy base + `--move-resolution on`)
_MR = {"move_resolution": "on"}
RMI = {n: i for i, n in enumerate(RMC)}
CI = {n: i for i, n in enumerate(CURE_FACTS)}


def _toggles(**over: Any) -> Dict[str, Any]:
    # LEGACY-BASE re-point (the END STATE's adoption, 2026-10-12): the flag tests are "base ⊕ one lever" with the lever
    # OFF in the base; production carries both ON, so the base is the PRE-ADOPTION network.
    with open(repo_path("designs", "production_config.json")) as fh:
        cfg = legacy_production_config(json.load(fh))
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


def _plant_live(mod: torch.nn.Module, seed: int = 5) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _n, m in mod.named_modules():
            wt = getattr(m, "weight", None)
            if isinstance(wt, torch.nn.Parameter) and wt.dim() == 2 and not bool(wt.any()):
                wt.copy_(torch.randn(wt.shape, generator=g) * 0.1)


# ====================================================================================== THE EXACT P(KO)
def _brute_ko(top: float, hp_points: int, crit_top: float, c: float) -> float:
    """The game's rule, enumerated: roll r deals floor(top · r / 100); a KO is damage ≥ HP (integers)."""
    def frac(t: float) -> float:
        return sum(math.floor(t * r / 100.0) >= hp_points for r in ROLLS) / 16.0
    return (1.0 - c) * frac(top) + c * frac(crit_top)


def _far_from_a_step(top: float, hp: int) -> bool:
    """Every roll's damage at least 0.5 HP from the integer HP threshold (standing rule 8: a case within the
    resolution of the observation is excluded, never tolerated by chance)."""
    return all(abs(top * r / 100.0 - hp) >= 0.5 for r in ROLLS)


def test_the_exact_ko_equals_the_enumerated_rolls_and_crit_on_our_side() -> None:
    """Over 4,000 seeded (damage, HP) pairs whose every roll sits at least ½ HP from the HP threshold, the exact rule
    (mean-roll damage in, ours ±½ HP) equals the ENUMERATION of the 16 floored rolls and the crit, for an ordinary
    move (1/16) and a high-crit one (1/8). Fails on the ramp (mean-anchored, no crit), on a mean/max mix-up (the
    0.925), on a missing crit term and on a crit chance from the wrong ratio."""
    g = torch.Generator().manual_seed(11)
    checked = 0
    for _ in range(4000):
        top = float(torch.empty(1).uniform_(40.0, 700.0, generator=g))
        hp = int(torch.randint(1, 600, (1,), generator=g))
        crit_top = 2.0 * top * float(torch.empty(1).uniform_(1.0, 2.0, generator=g))   # a crit skips the screens
        if not (_far_from_a_step(top, hp) and _far_from_a_step(crit_top, hp)):
            continue
        for c in (1.0 / 16.0, 1.0 / 8.0):
            lo, hi = ours_hp_bounds(torch.tensor(float(hp)), 1.0)
            got = float(ko_given_hit(torch.tensor(top * MEAN_ROLL), torch.tensor(crit_top * MEAN_ROLL), lo, hi,
                                     torch.tensor(c)))
            assert got == pytest.approx(_brute_ko(top, hp, crit_top, c), abs=1e-6), (top, hp, crit_top, c)
        checked += 1
    assert checked > 1000, checked


def test_the_ramp_was_wrong_where_the_mean_roll_equals_the_hp() -> None:
    """The finding the lever fixes: a hit whose MEAN roll equals the target's HP KOs on 8 of the 16 rolls (92.5 …
    100 of the top), plus the crit; the legacy ramp read 0 there."""
    top, hp = 400.0, 370.0                                          # mean 370: rolls 93 … 100 (372 … 400) KO
    lo, hi = ours_hp_bounds(torch.tensor(hp), 1.0)
    p = float(roll_ko_prob(torch.tensor(top * MEAN_ROLL), lo, hi))
    assert p == pytest.approx(8.0 / 16.0)
    ramp = max(0.0, min(1.0, (top * MEAN_ROLL - hp) / (0.15 * top * MEAN_ROLL)))
    assert ramp == 0.0
    full = float(ko_given_hit(torch.tensor(top * MEAN_ROLL), torch.tensor(2 * top * MEAN_ROLL), lo, hi))
    assert full == pytest.approx(15.0 / 16.0 * 0.5 + 1.0 / 16.0 * 1.0)


def test_their_hp_is_the_reported_percentage_bin() -> None:
    """HP Percentage Mod (`sim/pokemon.ts` getHealth: ceil(100 · hp / maxhp), 99 % until full): a reported p < 100 %
    is a true fraction in (p − 1 %, p]; a full one is exact (±½ HP). A hit landing in the middle of the bin KOs
    with probability ½ per roll that straddles it — the exact P(KO) given the observation."""
    lo, hi = opp_hp_bounds(torch.tensor(0.50), torch.tensor(0.50), 1.0, 1.0 / 300.0)
    assert float(lo) == pytest.approx(0.49) and float(hi) == pytest.approx(0.50)
    lo, hi = opp_hp_bounds(torch.tensor(1.0), torch.tensor(1.0), 1.0, 1.0 / 300.0)
    assert float(hi - lo) == pytest.approx(1.0 / 300.0)
    # a 100-roll damage of 0.495: roll 100 lands mid-bin (½), roll 99 (0.49005) just inside its floor (0.005)
    lo, hi = opp_hp_bounds(torch.tensor(0.50), torch.tensor(0.50), 1.0, 1.0 / 300.0)
    p = float(roll_ko_prob(torch.tensor(0.495 * MEAN_ROLL), lo, hi))
    want = sum(min(1.0, max(0.0, (0.495 * r / 100.0 - 0.49) / 0.01)) for r in ROLLS) / 16.0
    assert want == pytest.approx((0.5 + 0.005) / 16.0)
    assert p == pytest.approx(want, abs=1e-5)


def test_the_crit_table_reads_each_moves_gen3_crit_ratio() -> None:
    """`sim/battle-actions.ts`: gen ≤ 5 critMult [0, 16, 8, …] by `critRatio` (default 1): Slash 1/8, Tackle 1/16."""
    t = crit_p_table(400)
    assert float(t[num("slash")]) == pytest.approx(1.0 / 8.0)
    assert float(t[num("crosschop")]) == pytest.approx(1.0 / 8.0)
    assert float(t[num("tackle")]) == pytest.approx(CRIT_P_BASE)
    assert float(t[num("earthquake")]) == pytest.approx(1.0 / 16.0)


def test_every_op_ko_site_reads_the_one_rule(rows: torch.Tensor) -> None:
    """`DamageOperator._rolls` IS `ko_exact.ko_given_hit` (ours ±½ HP; theirs the percentage bin) — the op's only
    P(KO) since U1 — and the stashed incoming KO column is a probability on real rows. Fails if a site keeps a
    private ramp."""
    op = _policy().features_extractor.damage_op
    assert hasattr(op, "MOVE_CRIT_P") and not hasattr(op, "ko_exact")
    g = torch.Generator().manual_seed(3)
    dmg_ns = torch.rand(4, 6, 5, generator=g) * 400.0
    screen = torch.where(torch.rand(4, 1, 5, generator=g) > 0.5, 0.5, 1.0)
    maxhp = 250.0 + torch.rand(4, 6, 1, generator=g) * 200.0
    cur = maxhp * torch.rand(4, 6, 1, generator=g)
    acc = torch.rand(4, 1, 5, generator=g)
    crit = torch.full((4, 1, 5), 1.0 / 8.0)
    _h, _l, _c, ko = op._rolls(dmg_ns, screen, maxhp, cur, acc, crit_p=crit)
    lo, hi = ours_hp_bounds(cur, 1.0)
    want = acc * ko_given_hit(dmg_ns * screen, 2.0 * dmg_ns, lo, hi, crit)
    assert torch.allclose(ko, want, atol=1e-6)
    frac = cur / maxhp
    _h, _l, _c, ko_o = op._rolls(dmg_ns, screen, maxhp, cur, acc, crit_p=crit, opp_hp_frac=frac)
    lo, hi = opp_hp_bounds(cur, frac, maxhp, 1.0)
    assert torch.allclose(ko_o, acc * ko_given_hit(dmg_ns * screen, 2.0 * dmg_ns, lo, hi, crit), atol=1e-6)
    # the real forward: the stashed incoming KO column (acc · P(KO | hit)) is a live probability on real rows
    fe = _policy().features_extractor
    with torch.no_grad():
        fe({"observation": rows})
        ko_ex = fe.damage_op.last_pair_cells[..., PAIR_OUTCOME_IDX["ko_ramp"]]
    assert float(ko_ex.min()) >= 0.0 and float(ko_ex.max()) <= 1.0 and float(ko_ex.max()) > 0.0


def test_the_substitute_break_is_exact_at_the_subs_floor_hp() -> None:
    """intent_threshold (P8) with the exact P(KO): P(their hit breaks our sub | hit) at the sub's exact HP
    floor(maxhp / 4) over the rolls + crit. A 601-HP mon's sub holds 150 HP; a 171.5-HP top hit deals 149.2 on roll 87
    and 150.9 on roll 88 (both > ½ HP from the threshold), so it breaks on rolls 88 … 100 and on every crit. Fails on
    the ramp, on a sub at 25 % of max HP read as a fraction without the floor, and on a dropped crit."""
    from agents.model.intent_threshold import ExactKo, sub_break_given_hit
    maxhp = torch.tensor([601.0])
    top = 171.5
    assert _far_from_a_step(top, 150)
    high = torch.tensor([[top * MEAN_ROLL / 601.0]])
    crit = torch.tensor([[2.0 * top * MEAN_ROLL / 601.0]])
    p = float(sub_break_given_hit(high, crit, ExactKo(maxhp=maxhp, crit_p=torch.tensor([[1.0 / 16.0]]))))
    n = sum(math.floor(top * r / 100.0) >= 150 for r in ROLLS)
    assert n == 13
    assert p == pytest.approx(15.0 / 16.0 * n / 16.0 + 1.0 / 16.0, abs=1e-5)


def test_intent_threshold_reads_the_exact_break_when_given_the_operands() -> None:
    """`threshold_probs(..., exact=ExactKo(...))` (P8's site) prices the Substitute break by `sub_break_given_hit`,
    not the ramp; `exact=None` keeps the ramp bit for bit. Fails if the site keeps its private ramp under the flag."""
    from agents.model.intent_threshold import ExactKo, sub_break_given_hit, threshold_probs
    cells = torch.zeros(1, 6, 2, 6)
    cells[0, 0, 0, 1] = 171.5 * MEAN_ROLL / 601.0                     # high (mean roll)
    cells[0, 0, 0, 2] = 2.0 * 171.5 * MEAN_ROLL / 601.0               # crit
    cells[0, 0, 0, 4] = 0.9                                           # acc
    logits = torch.tensor([[2.0, -30.0, -30.0]])                     # seat 0 (+ a dead seat), switch
    gate = torch.ones(1, 6, 1)
    act = torch.tensor([0])
    ex = ExactKo(maxhp=torch.tensor([601.0]), crit_p=torch.full((1, 2), 1.0 / 16.0))
    a0 = float(torch.softmax(logits, -1)[0, 0])
    p_exact = float(threshold_probs(logits, cells, gate, act, exact=ex).p_sub_broken)
    want = a0 * 0.9 * float(sub_break_given_hit(cells[0, 0, :, 1][None], cells[0, 0, :, 2][None], ex)[0, 0])
    assert p_exact == pytest.approx(want, abs=1e-6)
    p_ramp = float(threshold_probs(logits, cells, gate, act).p_sub_broken)
    hk = float(cells[0, 0, 0, 1])
    assert p_ramp == pytest.approx(a0 * 0.9 * min(1.0, max(0.0, (hk - 0.25) / (0.15 * hk + 1e-6))), abs=1e-6)
    assert abs(p_exact - p_ramp) > 1e-3


# ====================================================================================== THE STATUS FACTS
def test_the_pair_status_columns_carry_the_two_facts(rows: torch.Tensor) -> None:
    """The op's unified outcome grid keeps its width; positions 12-13 hold the expected burn damage lost (= P(this
    seat's burn lands) × our mon's burn loss — one ratio per mon, whatever the seat) and the expected outspeed lost
    (= P(paralysis lands) × Δ P(we act first), the forward's own speed pair). The JUDGMENTS these columns held
    (`neutralization`, `tempo_cost`) were deleted by U1. Fails on a re-weighted sum, a severity, or a burn column that
    is not the per-mon loss."""
    with torch.no_grad():
        on = _policy().features_extractor
        on({"observation": rows})
        pin_on = on.damage_op.last_pair_in
        p_fast, p_fast_par = on.damage_op.stash.speed_fast_pair
    p_par, p_brn = pin_on[..., PAIR_OUTCOME_IDX["p_par"]], pin_on[..., PAIR_OUTCOME_IDX["p_brn"]]
    assert int((p_par > 1e-3).sum()) > 0, "no paralysis-landing seat in the fixture"
    d_fast = (p_fast - p_fast_par).clamp(0.0, 1.0)
    assert int(((p_par > 1e-3) & (d_fast[:, :, None] > 1e-3)).sum()) > 0, "PRECONDITION: a live paralysis cost"
    burned = p_brn > 1e-3
    assert int(burned.sum()) > 0, "no burn-landing seat in the fixture"
    with torch.no_grad():
        ctx = on.unpack({"observation": rows})
        loss = on.damage_op.our_burn_loss(ctx, on.last_spread_belief)                 # [B,6]
    assert float((loss[:, :, None] * burned).max()) > 0.0, "every burn-landing seat's mon loses nothing"
    assert torch.allclose(pin_on[..., 12], p_brn * loss[:, :, None], atol=1e-6)
    assert float(pin_on[..., 12].min()) >= 0.0 and float(pin_on[..., 13].min()) >= 0.0
    assert STATUS_FACT_COORDS == ("e_burn_dmg_lost", "e_par_outspeed_lost")


def _bench_slot(ctx: Any) -> int:
    a = int(ctx.our_active_idx[0])
    return (a + 1) % TEAM_SIZE


def _with_moves(ctx: Any, slot: int, mids: List[str]) -> Any:
    from agents.observation.types import TypeEncoder
    ids, tys = ctx.all_move_ids.clone(), ctx.all_move_type_ids.clone()
    ids[:, slot] = 0
    tys[:, slot] = 0
    for i, mid in enumerate(mids):
        ids[:, slot, i] = num(mid)
        tys[:, slot, i] = TypeEncoder.TYPE_TO_IDX[str(gen3_data.moves.raw()[mid]["type"]).upper()]
    return dataclasses.replace(ctx, all_move_ids=ids, all_move_type_ids=tys)


def test_the_burn_loss_on_a_mixed_attacker_is_its_physical_moves_halved(rows: torch.Tensor) -> None:
    """`our_burn_loss` on a constructed bench mon: a PHYSICAL move loses damage, a SPECIAL one nothing, a mixed set
    the expectation over its held moves (α = 1 / held: [Earthquake, Surf] reads exactly half of [Earthquake]); Guts
    and an already-burned mon lose nothing (Showdown gen-3 `modifyDamage`: the halving skips Guts; a burned mon is
    halved in both worlds). Fails if the special move were halved, if the expectation were a max, or on a dropped
    Guts / already-burned case."""
    fe = _policy().features_extractor
    op = fe.damage_op
    with torch.no_grad():
        ctx = fe.unpack({"observation": rows[:8]})
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:, -1].any(dim=1)
        keep = [i for i in range(8) if bool(has_opp[i])][:1]
        assert keep, "no row with an opponent active"
        ctx = fe.unpack({"observation": rows[keep]})
        j = _bench_slot(ctx)
        pp = ctx.pokemon_part.clone()
        pp[:, j, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7] = 0.0
        hp = ctx.hp_and_active.clone()
        hp[:, j, 0] = 1.0
        ctx = dataclasses.replace(ctx, pokemon_part=pp, hp_and_active=hp)
        phys = float(op.our_burn_loss(_with_moves(ctx, j, ["earthquake"]))[0, j])
        spec = float(op.our_burn_loss(_with_moves(ctx, j, ["surf"]))[0, j])
        mixed = float(op.our_burn_loss(_with_moves(ctx, j, ["earthquake", "surf"]))[0, j])
        assert phys > 0.01 and spec == 0.0
        assert mixed == pytest.approx(phys / 2.0, rel=1e-5)
        guts = ctx.ability1_ids.clone()
        guts[:, j] = int(gen3_data.abilities.get("guts").num)
        assert float(op.our_burn_loss(dataclasses.replace(_with_moves(ctx, j, ["earthquake"]),
                                                          ability1_ids=guts))[0, j]) == 0.0
        pb = ctx.pokemon_part.clone()
        pb[:, j, POKEMON_CONDITION_OFFSET + 1] = 1.0                    # [None, BRN, …]: already burned
        assert float(op.our_burn_loss(dataclasses.replace(_with_moves(ctx, j, ["earthquake"]),
                                                          pokemon_part=pb))[0, j]) == 0.0


def _cure_board(fe: Any, rows: torch.Tensor) -> Any:
    ctx = fe.unpack({"observation": rows[:1]})
    hp = ctx.hp_and_active.clone()
    hp[:, :, 0] = 1.0
    pp = ctx.pokemon_part.clone()
    pp[:, :, POKEMON_ITEMS_OFFSET + 1] = 1.0                          # every item known …
    pp[:, :, POKEMON_ITEMS_OFFSET + 2] = 0.0                          # … and not consumed
    items = torch.zeros_like(ctx.item_ids)
    abl = torch.zeros_like(ctx.ability1_ids)
    ids = torch.zeros_like(ctx.all_move_ids)
    bm = torch.zeros_like(ctx.opp_believed_mask)
    return dataclasses.replace(ctx, hp_and_active=hp, pokemon_part=pp, item_ids=items, ability1_ids=abl,
                               all_move_ids=ids, opp_believed_mask=bm)


def test_the_cure_flags_on_constructed_boards(rows: torch.Tensor) -> None:
    """Each flag in isolation: a live Heal Bell user marks its WHOLE side (a fainted one does not); Natural Cure,
    Rest, a known Lum / Chesto Berry on the mon itself (a consumed one reads 0); THEIR unrevealed item reads the
    species' Smogon prior. Fails on a side mix-up, a dropped alive gate, a consumed berry read as held."""
    fe = _policy().features_extractor
    rule: CureFlags = fe.status_cure_rule
    op = copy.copy(fe.damage_op)
    from agents.model.damage_op import OpStashes
    op.stash = OpStashes()                                           # the belief-off branch: logits below
    with torch.no_grad():
        ctx = _cure_board(fe, rows)
        ids = ctx.all_move_ids.clone()
        ids[0, 2, 0] = num("healbell")
        ids[0, 3, 0] = num("rest")
        abl = ctx.ability1_ids.clone()
        abl[0, 4] = int(gen3_data.abilities.get("naturalcure").num)
        items = ctx.item_ids.clone()
        items[0, 5] = int(gen3_data.items.get("lumberry").num)
        items[0, 1] = int(gen3_data.items.get("chestoberry").num)
        pp = ctx.pokemon_part.clone()
        pp[0, TEAM_SIZE + 1, POKEMON_ITEMS_OFFSET + 1] = 0.0            # their slot 1: item unrevealed
        sp = ctx.species_ids.clone()
        sp[0, TEAM_SIZE + 1] = int(gen3_data.species.get("snorlax").num)
        c = dataclasses.replace(ctx, all_move_ids=ids, ability1_ids=abl, item_ids=items, pokemon_part=pp,
                                species_ids=sp)
        logits = torch.full((1, TEAM_SIZE, op.MOVE_BP.shape[0]), -30.0)
        out = rule(c, op, logits)                                    # [1,12,CURE_DIM]
        assert out.shape == (1, 2 * TEAM_SIZE, CURE_DIM)
        assert torch.equal(out[0, :TEAM_SIZE, CI["cleric_on_side"]], torch.ones(TEAM_SIZE))
        assert float(out[0, TEAM_SIZE:, CI["cleric_on_side"]].max()) < 1e-6
        assert float(out[0, 3, CI["rest"]]) == 1.0 and float(out[0, 2, CI["rest"]]) == 0.0
        assert float(out[0, 4, CI["natural_cure"]]) == 1.0 and float(out[0, 0, CI["natural_cure"]]) == 0.0
        assert float(out[0, 5, CI["lum_berry"]]) == 1.0 and float(out[0, 1, CI["chesto_berry"]]) == 1.0
        prior = float(rule.SPECIES_CURE_ITEM_PRIOR[sp[0, TEAM_SIZE + 1], 1])
        assert float(out[0, TEAM_SIZE + 1, CI["chesto_berry"]]) == pytest.approx(prior)
        # the cleric faints: the side loses it; a consumed berry is gone
        hp = c.hp_and_active.clone()
        hp[0, 2, 0] = 0.0
        pc = c.pokemon_part.clone()
        pc[0, 5, POKEMON_ITEMS_OFFSET + 2] = 1.0
        out2 = rule(dataclasses.replace(c, hp_and_active=hp, pokemon_part=pc), op, logits)
        assert float(out2[0, :TEAM_SIZE, CI["cleric_on_side"]].max()) == 0.0
        assert float(out2[0, 5, CI["lum_berry"]]) == 0.0
        # their believed Heal Bell (the belief-off move presence) marks THEIR side
        lg = logits.clone()
        lg[0, 0, num("healbell")] = 30.0
        out3 = rule(c, op, lg)
        assert float(out3[0, TEAM_SIZE:, CI["cleric_on_side"]].min()) == pytest.approx(1.0)


def test_the_cure_flags_and_the_burn_loss_permute_with_our_team_slots(rows: torch.Tensor) -> None:
    """Equivariance: permuting our six slots permutes our cure flags and our burn losses (one formula per mon)."""
    fe = _policy().features_extractor
    sig = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        fe({"observation": rows[:16]})                               # leaves the X5 roster on the op's stash
        ctx = fe.unpack({"observation": rows[:16]})
        base = fe.status_cure_rule(ctx, fe.damage_op, fe.last_move_belief_logits)
        perm = fe.status_cure_rule(_permute_ours(ctx, sig), fe.damage_op, fe.last_move_belief_logits)
        assert torch.allclose(perm[:, :TEAM_SIZE], base[:, sig], atol=1e-6)
        assert torch.allclose(perm[:, TEAM_SIZE:], base[:, TEAM_SIZE:], atol=1e-6)
        bl = fe.damage_op.our_burn_loss(ctx)
        blp = fe.damage_op.our_burn_loss(_permute_ours(ctx, sig))
        assert torch.allclose(blp, bl[:, sig], atol=1e-5)
        assert float(bl.abs().sum()) > 0.0


# ====================================================================================== THE RESTORED FACTS
def _restored(o: Any) -> torch.Tensor:
    m = move_facts(o, KIND_T, FLAG_T)
    return restored_move_facts(o, m, KIND_T, FLAG_T, torch.tensor([num("endeavor")]))


def test_focus_punch_survives_unless_a_seat_hits_us_and_not_through_our_sub() -> None:
    """`fp_survives` = 1 − Σ α_k · P(seat k's damaging hit lands on us); a hit on OUR Substitute does not break the
    focus (`data/mods/gen4/moves.ts` focuspunch: lostFocus is set by damage to the user). Fails if the sub exception
    or the accuracy were dropped."""
    kw = dict(seats=("bodyslam", "growl", "growl"), alpha=(0.5, 0.2, 0.2), pair_in=hit_grid(seat=0, acc=0.8))
    r = _restored(ops(("focuspunch",), **kw))
    assert float(r[0, 0, RMI["fp_survives"]]) == pytest.approx(1.0 - 0.5 * 0.8)
    r = _restored(ops(("focuspunch",), **kw, **with_vol("our", "substitute")))
    assert float(r[0, 0, RMI["fp_survives"]]) == pytest.approx(1.0)
    assert float(_restored(ops(("tackle",), **kw))[0, 0, RMI["fp_survives"]]) == 0.0


def test_the_substitute_survives_the_exact_break() -> None:
    """`sub_survives` = 1 − Σ α_k · acc_k · P(dmg ≥ sub | hit): the exact roll count at floor(maxhp / 4) (the ramp
    re-thresholded at 25 % it replaced was deleted by U1)."""
    pin = hit_grid(seat=0, high=171.5 * MEAN_ROLL / 601.0, acc=1.0)
    pin[0, 0, 0, 2] = 2.0 * 171.5 * MEAN_ROLL / 601.0
    o = ops(("substitute",), seats=("bodyslam", "growl", "growl"), alpha=(0.5, 0.2, 0.2), pair_in=pin,
            our_maxhp=torch.tensor([601.0]), seat_crit=torch.full((1, 3), 1.0 / 16.0))
    brk = 15.0 / 16.0 * 13 / 16.0 + 1.0 / 16.0                    # rolls 88 … 100 of 171.5 reach 150; every crit
    assert float(_restored(o)[0, 0, RMI["sub_survives"]]) == pytest.approx(1.0 - 0.5 * brk, abs=1e-5)


def test_endure_and_endeavor_read_p_ko_with_no_threshold() -> None:
    """Endure × P(they KO us); Endeavor × P(we are NOT KO'd) — the family's own `p_ko_us`, no hand threshold."""
    kw = dict(seats=("bodyslam", "growl", "growl"), alpha=(0.6, 0.2, 0.1), pair_in=hit_grid(seat=0, ko=0.7))
    o = ops(("endure", "endeavor", "tackle"), **kw)
    r = _restored(o)
    p_ko = 0.6 * 0.7
    assert float(r[0, 0, RMI["endure_p_ko"]]) == pytest.approx(p_ko)
    assert float(r[0, 1, RMI["endeavor_survives"]]) == pytest.approx(1.0 - p_ko)
    assert float(r[0, 2, RMI["endure_p_ko"]]) == 0.0 and float(r[0, 2, RMI["endeavor_survives"]]) == 0.0


def test_spin_value_lost_is_the_expected_layers_a_failed_spin_leaves() -> None:
    """`spin_value_lost` = is Rapid Spin × (1 − p_resolve) × OUR side's Spikes / 3: a Ghost arrival (β) blocks the
    spin; with no Spikes on our side it costs nothing."""
    p_type = torch.zeros(1, 6, 19)
    imm = torch.zeros(1, 6, 19)
    from agents.observation.types import TypeEncoder
    imm[0, 1, TypeEncoder.TYPE_TO_IDX["NORMAL"]] = 1.0             # the arrival (β on slot 1) is a Ghost
    spikes = torch.tensor([[2.0 / 3.0, 0.0]])
    o = ops(("rapidspin",), a_switch=0.5, imm_dmg=imm, p_type=p_type, spikes=spikes)
    p_res = mf(o)
    assert p_res < 0.99
    r = _restored(o)
    assert float(r[0, 0, RMI["spin_value_lost"]]) == pytest.approx((1.0 - p_res) * 2.0 / 3.0)
    o0 = ops(("rapidspin",), a_switch=0.5, imm_dmg=imm, p_type=p_type, spikes=torch.zeros(1, 2))
    assert float(_restored(o0)[0, 0, RMI["spin_value_lost"]]) == 0.0


def test_the_spin_denial_stake_is_the_expected_layers_kept() -> None:
    """`spin_denied_stake` = P(their Rapid Spin fails on our Ghost j) × THEIR side's Spikes / 3 (the layers we laid,
    which our Ghost switch-in preserves) — the stake the base family dropped (owner re-classification 2026-10-09)."""
    ghost = torch.zeros(1, 6)
    ghost[0, 2] = 1.0
    o = ops(("tackle",), seats=("rapidspin", "growl", "growl"), alpha=(0.4, 0.2, 0.2), our_is_ghost=ghost,
            spikes=torch.tensor([[0.0, 1.0]]))
    s = restored_switch_facts(o, switch_facts(o))
    assert s.shape == (1, 6, len(RSC))
    assert float(s[0, 2, 0]) == pytest.approx(0.4 * 1.0)
    assert float(s[0, 1, 0]) == 0.0


def test_the_status_facts_reach_the_move_resolution_cells() -> None:
    """Under `--move-resolution on` the two status facts (pair_in 12-13) are α-reduced at our active (move cell) and
    at every mon (switch cell)."""
    pin = torch.zeros(1, 6, 3, 14)
    pin[0, 0, 0, 12] = 0.2
    pin[0, 0, 1, 13] = 0.5
    pin[0, 3, 0, 12] = 0.4
    o = ops(("tackle",), alpha=(0.5, 0.25, 0.0), pair_in=pin)
    sm = status_move_facts(o)
    assert float(sm[0, 0, 0]) == pytest.approx(0.1) and float(sm[0, 0, 1]) == pytest.approx(0.125)
    ss = status_switch_facts(o)
    assert float(ss[0, 3, 0]) == pytest.approx(0.2)


@pytest.mark.parametrize("lever", ["move_resolution_facts", "status_facts"])
def test_the_restored_and_status_facts_reach_the_pointer(rows: torch.Tensor, lever: str) -> None:
    """The new blocks reach the policy ONLY through their zero-init projections: the restored facts at init equal the
    lever-off build's pointer bit for bit; with either projection planted, the move cells (and logits) move. (The
    status facts have no off build since U1: their projection is zero-init and the planted copy moves the cells.)
    Fails if a block bypasses its projection or never reaches the pointer."""
    if lever == "move_resolution_facts":
        off = _policy(**_MR)
        on = copy.deepcopy(_policy(**_MR, **{lever: _ON[lever]}))
        for a, b in zip(_record(off, rows), _record(on, rows)):
            assert torch.equal(a, b)
    else:
        off = _policy(**_MR)
        on = copy.deepcopy(off)
    cell: MoveResolutionCell = on.features_extractor.move_resolution_cell
    proj = cell.restored_move_proj if lever == "move_resolution_facts" else cell.status_move_proj
    assert proj is not None and float(proj.weight.abs().max()) == 0.0
    _plant_live(proj)
    rec_on = _record(on, rows)
    assert not torch.equal(rec_on[4], _record(off, rows)[4]), "the planted block never reached the move cells"


# ====================================================================================== ONE RULE / NO CLOCK
def test_the_g_ledger_reads_the_one_end_of_turn_rule(rows: torch.Tensor) -> None:
    """The op's `g` cells ARE `eot_residual.g_cells` of THE rule the per-mon residual reads (its four groups summing to
    the rule's unclamped total) — the op's own coarse ledger was deleted by U1. Fails on a second ledger or a dropped
    component."""
    from agents.model.eot_residual import EOT_FACTS, g_cells
    fe = _policy().features_extractor
    with torch.no_grad():
        fe({"observation": rows})
        ctx = fe.unpack({"observation": rows})
        our_g, opp_g = fe.damage_op.pairwise_schedule(ctx)
        x5 = fe.damage_op.stash.x5
        eot = fe.damage_op.eot_rule(ctx, fe.damage_op, None if x5 is None else x5.concrete,
                                    None if x5 is None else x5.alive)
        want_our, want_opp = g_cells(eot)
        if x5 is not None:
            want_opp = want_opp * x5.alive[:, :, None]
    assert torch.allclose(our_g, want_our) and torch.allclose(opp_g, want_opp)
    net_unclamped = eot[..., :len(EOT_FACTS) - 1].sum(-1)
    assert torch.allclose(torch.cat([our_g, opp_g], 1).sum(-1), net_unclamped, atol=1e-6)
    assert float(our_g.abs().sum()) > 0.0, "PRECONDITION: some end-of-turn HP change on our side in the fixture"


def test_dropping_the_progress_clock_is_reading_it_as_zero(rows: torch.Tensor) -> None:
    """`--drop-progress-clock on` builds the SAME parameters (bytes equal) and its forward equals the legacy base's
    forward on rows whose `turns_since_progress` cell is zeroed — and differs from it on the raw rows (the cell is
    live in the fixture). The observation layout is untouched. Fails if the mask hits another column or none."""
    off, on = _policy(), _policy(drop_progress_clock="on")
    so, sn = off.state_dict(), on.state_dict()
    assert set(so) == set(sn) and all(torch.equal(so[k], sn[k]) for k in so)
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from agents.observation.schema import build_schema
    lay = Gen3ObservationEncoder(load_mappings()).get_layout()
    sl = build_schema(lay).slices()
    col = sl['global_env'].start + on.features_extractor.unpack._tsp
    assert float(rows[:, col].abs().sum()) > 0.0, "the fixture's progress clock is 0 everywhere"
    z = rows.clone()
    z[:, col] = 0.0
    for a, b in zip(_record(on, rows), _record(off, z)):
        assert torch.equal(a, b)
    assert not torch.equal(_record(on, rows)[0], _record(off, rows)[0])


# ====================================================================================== IDENTITY / INIT / VERSION
def test_flags_off_builds_nothing_and_the_op_builds_its_facts() -> None:
    """The legacy base builds neither remaining flag (no restored projection, the clock read live); the op always holds
    the crit table, the one end-of-turn rule and the status / cure facts (U1)."""
    fe = _policy().features_extractor
    assert hasattr(fe.damage_op, "MOVE_CRIT_P") and fe.damage_op.eot_rule is not None
    assert fe.status_cure_rule is not None and fe.status_cure_proj is not None
    assert fe.unpack.drop_progress_clock is False
    for n in _LEVERS:
        assert getattr(fe, n) == _OFF[n]
    mr = _policy(**_MR).features_extractor.move_resolution_cell
    assert mr.restored_move_proj is None and mr.status_move_proj is not None


@pytest.mark.parametrize("over,extra", [
    ({**_MR, "move_resolution_facts": "full"}, ("restored_move_proj", "restored_switch_proj")),
], ids=["facts"])
def test_one_lever_init(rows: torch.Tensor, over: Dict[str, str], extra: tuple) -> None:
    """ON = OFF + exactly the lever's zero-init keys; every other initial byte equal; the init forward bit-identical."""
    base = {k: v for k, v in over.items() if k in _MR}
    off, on = _policy(**base), _policy(**over)
    so, sn = off.state_dict(), on.state_dict()
    assert not set(so) - set(sn)
    new = sorted(set(sn) - set(so))
    assert new and all(any(e in k for e in extra) for k in new), new
    assert all(float(sn[k].abs().max()) == 0.0 for k in new)
    assert all(torch.equal(so[k], sn[k]) for k in so)
    for a, b in zip(_record(off, rows), _record(on, rows)):
        assert torch.equal(a, b)


def test_a_pre_v152_config_migrates_to_off_and_a_mismatch_is_refused() -> None:
    """A pre-v152 config migrates to the remaining levers' OFF values (the only possible past); each flip is refused by
    `check_compatible` naming it; the kwarg reaches the recorded field and `current_model_version`."""
    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError
    from agents.model.model_version.migrations import _migrate_config
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    assert MODEL_CONFIG_VERSION >= 152
    data = _migrate_config({"config_version": 151})
    for n in _LEVERS:
        assert data[n] == _OFF[n]
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    mv = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    for n in _LEVERS:
        assert getattr(mv, n) == _OFF[n]
        with pytest.raises(ModelVersionError, match=n):
            mv.check_compatible(dataclasses.replace(mv, **{n: _ON[n]}))
        rec = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {n: _ON[n]}})
        assert getattr(rec, n) == _ON[n]
        assert getattr(current_model_version(load_mappings(), **{n: _ON[n]}), n) == _ON[n]
