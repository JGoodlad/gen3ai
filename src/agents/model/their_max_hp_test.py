"""gen3_their_max_hp_belief_v1 (backlog E13) — THEIR max HP is a BELIEF from the species' Smogon spreads, held in one
place (`damage_kinds.their_max_hp` / `TheirHp` over `belief_tables.build_their_max_hp`), where the op read every
opponent as carrying 0 HP EVs (``2·base + 141``: a Skarmory 271 for a usual 334, so a 35 % Skarmory read as killed by
Seismic Toss when 95 % of them survive it).

The mechanics, each verified in `deps/pokemon-showdown`:

* max HP at level 100 = ``2·base + IV + tr(EV / 4) + 110`` (`sim/battle.ts` ``statModify``'s HP branch: ``tr(tr(2 * stat
  + set.ivs + tr(set.evs / 4) + 100) * set.level / 100 + 10)``); a Shedinja's is 1 whatever its spread
  (`data/pokedex.ts` shedinja ``maxHP: 1``; `sim/pokemon.ts` ``if (this.species.maxHP) stats.hp = this.species.maxHP``);
* the opponent's HP reaches us as a percent, ``ceil(100 · hp / maxhp)`` (`sim/pokemon.ts` ``getHealth``, HP Percentage
  Mod): a reported ``p`` < 100 % means ``hp`` in ``((p − 1 %)·M, p·M]``;
* a hit KOs at ``hp <= 0`` (`sim/pokemon.ts` ``damage``), so a fixed 100 HP (Seismic Toss) KOs iff ``hp <= 100``;
* an end-of-turn residual is ``max(1, ⌊M / d⌋)`` (`sim/battle.ts` ``spreadDamage`` ``clampIntRange(damage, 1)``).

What a revert breaks is in each docstring. The LEAK GUARD is three tests: the helper refuses anything but integer
species ids, no forward module spells their max HP outside the helper, and the op's whole output and the end-of-turn
rule are bit-identical whatever the opponent's spread cells hold."""
from __future__ import annotations

import dataclasses
import io
import json
import re
import tokenize
import types
from typing import Any, Dict, List, Tuple

import pytest
import torch

from agents import gen3_data
from agents.model import belief_tables as bt
from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.damage_kinds import (RESIDUAL_DENOMS, TheirHp, expected_their_max_hp, their_fixed_ko,
                                       their_ko_given_hit, their_max_hp, their_residual_frac, their_residual_points)
from agents.model.features_extractor import DamageOperator
from agents.model.identity_init_test import _build_real_policy
from agents.model.ko_exact import ko_given_hit, opp_hp_bounds
from agents.model.nonformula_damage_test import _ctx as _nf_ctx
from agents.model.damage_op_layout import _DMG_OAX_PER_MOVE
from agents.model.damage_op_test import _make_layout
from agents.model.selection_sites import FORWARD_MODULES
from agents.model.static_recovery_test import _COL, _blank, _eot, _mon, _vol
from agents.model.static_recovery_test import _policy as _legacy_policy
from agents.observation.constants import POKEMON_SPREAD_DIM, POKEMON_SPREAD_OFFSET, TEAM_SIZE
from utils.paths import repo_path, src_path

_N_SPECIES = 600                                    # > every gen-3 species num
_SKARMORY, _BLISSEY, _SHEDINJA, _SALAMENCE, _SNORLAX = (
    int(gen3_data.species.get(s).num) for s in ("skarmory", "blissey", "shedinja", "salamence", "snorlax"))


@pytest.fixture(scope="module")
def tables() -> Dict[str, torch.Tensor]:
    return bt.build_their_max_hp(_N_SPECIES)


@pytest.fixture(scope="module")
def top(tables: Dict[str, torch.Tensor]) -> Any:
    """A stand-in for the op that carries only its `THEIR_HP_*` buffers (all the helpers read)."""
    return types.SimpleNamespace(**tables)


@pytest.fixture(scope="module")
def op() -> DamageOperator:
    return DamageOperator(_make_layout(), outgoing=True)


@pytest.fixture(scope="module")
def rows() -> torch.Tensor:
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    obs, _mask = load_parity_rows(Gen3ObservationEncoder(load_mappings()).dimension)
    return torch.as_tensor(obs)


def _smogon_support(species: str) -> Tuple[List[int], List[float]]:
    """The species' max-HP distribution re-derived HERE from the facade, independently of `their_max_hp_support`:
    Showdown's L100 formula at IV 31 over every Smogon spread's HP EV."""
    base = int(gen3_data.species.get(species).base_stats["hp"])
    acc: Dict[int, float] = {}
    for _nature, evs, w in gen3_data.priors.all_spreads(species):
        m = 2 * base + 31 + int(evs[0]) // 4 + 110
        acc[m] = acc.get(m, 0.0) + float(w)
    tot = sum(acc.values())
    vals = sorted(acc)
    return vals, [acc[m] / tot for m in vals]


def _exact_fixed_ko(species: str, hp_frac: float, fixed: float) -> float:
    """P(a fixed ``fixed`` HP KOs) over the FULL support, in plain Python: their HP uniform on the reported percent
    bin ``((p − 1 %)·M, p·M]`` of each max HP ``M`` (a full mon: exactly ``M``, resolved over ±½ HP)."""
    tot = 0.0
    for m, w in zip(*_smogon_support(species)):
        lo, hi = (m - 0.5, m + 0.5) if hp_frac >= 1.0 else (hp_frac * m - 0.01 * m, hp_frac * m)
        tot += w * min(1.0, max(0.0, (fixed - lo) / (hi - lo)))
    return tot


# ------------------------------------------------------------------------------------------------ the tables
def test_the_support_is_showdowns_l100_hp_over_the_smogon_spreads() -> None:
    """Skarmory (base HP 65): 271 with 0 HP EVs, 334 with 252 — ``2·65 + 31 + tr(EV / 4) + 110``; Blissey (255): 651 to
    714. The support equals the independent re-derivation from the facade, value for value. A revert to the neutral
    point (271 / 651 alone) fails the max; a formula slip (``EV / 4`` unfloored, a missing IV) fails the lattice."""
    for species, lo, hi in (("skarmory", 271, 334), ("blissey", 651, 714), ("salamence", 331, 394)):
        vals, ws = bt.their_max_hp_support(species)
        want_v, want_w = _smogon_support(species)
        assert vals == want_v and ws == pytest.approx(want_w, rel=1e-12), species
        assert vals[0] == lo and vals[-1] == hi and sum(ws) == pytest.approx(1.0), (species, vals[0], vals[-1])
        assert all(isinstance(v, int) for v in vals)


def test_skarmory_is_usually_334_and_the_belief_says_so(tables: Dict[str, torch.Tensor]) -> None:
    """THE backlog case: Smogon's modal Skarmory carries 252 HP EVs (max HP 334), so the expected max HP is ~330, not
    the 271 the op assumed — a 22 % overstatement of every damage fraction against it. The headline figures the docs
    quote, pinned on the committed Smogon data (re-pin them with the data, for that cause only)."""
    vals, ws = bt.their_max_hp_support("skarmory")
    assert vals[max(range(len(ws)), key=ws.__getitem__)] == 334
    assert float(tables["THEIR_HP_MEAN"][_SKARMORY]) == pytest.approx(330.44, abs=0.05)
    assert 1.0 / float(tables["THEIR_HP_INV"][_SKARMORY]) == pytest.approx(329.79, abs=0.05)
    assert float(tables["THEIR_HP_MEAN"][_BLISSEY]) == pytest.approx(669.94, abs=0.05)


def test_a_species_whose_modal_set_has_no_hp_evs_keeps_its_mass_there(tables: Dict[str, torch.Tensor],
                                                                     top: Any) -> None:
    """Salamence's (and Blissey's) MODAL Smogon set has 0 HP EVs: the belief keeps that mass at the neutral 331 (its
    first atom, the majority weight) and puts the rest above — it is not a flat "+20 HP". Seismic Toss into a 30 %
    Salamence KOs the 331-HP majority (99.3 HP left at most) and not a 394-HP one: a probability strictly between,
    which no single assumed max HP gives (the old neutral point read 1; the mean 340 reads 0.41)."""
    for species, num, neutral in (("salamence", _SALAMENCE, 331), ("blissey", _BLISSEY, 651)):
        vals, ws = bt.their_max_hp_support(species)
        mode = max(range(len(ws)), key=ws.__getitem__)
        assert mode == 0 and ws[0] > 0.5 and vals[0] == neutral, (species, vals[mode], ws[0])
        assert float(tables["THEIR_HP_ATOMS"][num, 0]) == pytest.approx(vals[0], abs=0.5)
        assert float(tables["THEIR_HP_WEIGHTS"][num, 0]) >= ws[0] - 1e-6
        assert float(tables["THEIR_HP_MEAN"][num]) > vals[0] + 5.0
    got = float(their_fixed_ko(their_max_hp(top, torch.tensor([_SALAMENCE])), torch.tensor([0.30]),
                               torch.tensor([100.0])))
    assert got == pytest.approx(_exact_fixed_ko("salamence", 0.30, 100.0), abs=0.02)
    assert bt.their_max_hp_support("salamence")[1][0] <= got < 0.9


def test_shedinja_is_exactly_one_hp(tables: Dict[str, torch.Tensor], top: Any) -> None:
    """Species-determined (``maxHP: 1``), whatever Smogon's Shedinja spreads say: every table row is exactly 1, any
    hit KOs it and every residual takes its one HP. The old estimate read 2·1 + 141 = 143."""
    assert bt.their_max_hp_support("shedinja") == ([1], [1.0])
    assert float(tables["THEIR_HP_INV"][_SHEDINJA]) == 1.0 and float(tables["THEIR_HP_MEAN"][_SHEDINJA]) == 1.0
    assert torch.equal(tables["THEIR_HP_ATOMS"][_SHEDINJA], torch.ones(bt.THEIR_HP_ATOMS))
    assert tables["THEIR_HP_WEIGHTS"][_SHEDINJA].tolist() == [1.0] + [0.0] * (bt.THEIR_HP_ATOMS - 1)
    assert torch.equal(tables["THEIR_HP_RESIDUAL"][_SHEDINJA], torch.ones(len(RESIDUAL_DENOMS)))
    their = their_max_hp(top, torch.tensor([_SHEDINJA]))
    full = torch.tensor([1.0])
    assert float(their.maxhp) == 1.0 and float(their.points(full)) == 1.0
    assert float(their_ko_given_hit(torch.tensor([3.0]), torch.tensor([6.0]), their, full, None)) == 1.0
    assert float(their_fixed_ko(their, full, torch.tensor([20.0]))) == 1.0
    for d in RESIDUAL_DENOMS:
        assert float(their_residual_points(top, torch.tensor([_SHEDINJA]), d)) == 1.0


def test_a_base_hp_1_species_that_is_not_shedinja_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Base HP 1 is how the tables know the one ``maxHP: 1`` species; a second one would read 1 HP silently."""
    real = gen3_data.species.get("skarmory")
    monkeypatch.setattr(gen3_data.species, "get",
                        lambda _sid: dataclasses.replace(real, base_stats={**real.base_stats, "hp": 1}))
    with pytest.raises(ValueError, match="base HP 1 is how the op knows"):
        bt.their_max_hp_support("skarmory")


def test_every_table_is_its_supports_exact_moment(tables: Dict[str, torch.Tensor]) -> None:
    """For EVERY species: ``E[1 / M]``, ``E[M]`` and the four ``E[max(1, ⌊M / d⌋) / M]`` equal the support's sums; the
    atoms are a distribution (weights sum to 1, ascending, inside the support) whose HARMONIC identity ``Σ w_k / atom_k
    = E[1 / M]`` holds — the mixture's expected damage fraction is the full distribution's. Jensen: ``E[M] · E[1 / M]
    >= 1``, strictly for a species with more than one max HP (so the two reads are not interchangeable)."""
    spread_species = 0
    for sid in gen3_data.species.base_form_ids():
        num = int(gen3_data.species.get(sid).num)
        vals, ws = bt.their_max_hp_support(sid)
        inv = sum(w / m for m, w in zip(vals, ws))
        mean = sum(w * m for m, w in zip(vals, ws))
        assert float(tables["THEIR_HP_INV"][num]) == pytest.approx(inv, rel=1e-6), sid
        assert float(tables["THEIR_HP_MEAN"][num]) == pytest.approx(mean, rel=1e-6), sid
        for col, d in enumerate(bt.THEIR_HP_RESIDUAL_DENOMS):
            want = sum(w * max(1, m // d) / m for m, w in zip(vals, ws))
            assert float(tables["THEIR_HP_RESIDUAL"][num, col]) == pytest.approx(want, rel=1e-6), (sid, d)
        atoms, mass = tables["THEIR_HP_ATOMS"][num].double(), tables["THEIR_HP_WEIGHTS"][num].double()
        assert float(mass.sum()) == pytest.approx(1.0, abs=1e-6) and bool((mass >= 0).all()), sid
        assert bool((atoms[1:] >= atoms[:-1]).all()), sid
        assert vals[0] - 1e-3 <= float(atoms[0]) and float(atoms[-1]) <= vals[-1] + 1e-3, sid
        assert float((mass / atoms).sum()) == pytest.approx(inv, rel=1e-6), sid
        assert mean * inv >= 1.0 - 1e-12
        spread_species += int(len(vals) > 1 and mean * inv > 1.0 + 1e-6)
    assert spread_species > 100, spread_species


def test_the_unowned_row_and_a_species_without_a_usage_record(tables: Dict[str, torch.Tensor]) -> None:
    """Row 0 (no species: a pad / unknown id) is the point mass 141 — finite, so an empty slot never divides by 0. A
    species Smogon has no record of is uniform over the 64 lattice points (no usage, no preference)."""
    assert float(tables["THEIR_HP_MEAN"][0]) == float(bt.THEIR_HP_UNOWNED) == 141.0
    assert tables["THEIR_HP_WEIGHTS"][0].tolist() == [1.0] + [0.0] * (bt.THEIR_HP_ATOMS - 1)
    assert bool(torch.isfinite(torch.stack([t.flatten().sum() for t in tables.values()])).all())
    assert bool((tables["THEIR_HP_INV"] > 0).all())
    bare = [s for s in gen3_data.species.base_form_ids()
            if not gen3_data.priors.all_spreads(s) and gen3_data.species.get(s).base_stats["hp"] != 1]
    assert bare, "no species without a Smogon record: this pin has lost its fixture"
    vals, ws = bt.their_max_hp_support(bare[0])
    base = int(gen3_data.species.get(bare[0]).base_stats["hp"])
    assert vals == [2 * base + 141 + j for j in range(bt.THEIR_HP_LATTICE)]
    assert ws == pytest.approx([1.0 / bt.THEIR_HP_LATTICE] * bt.THEIR_HP_LATTICE)


def test_the_residual_columns_are_damage_kinds_denominators() -> None:
    """`belief_tables` cannot import the forward module that owns `RESIDUAL_DENOMS`; the two orders are pinned equal
    (a reorder would hand Leftovers the burn tick)."""
    assert tuple(float(d) for d in bt.THEIR_HP_RESIDUAL_DENOMS) == RESIDUAL_DENOMS


def _full_support_belief(species: str) -> TheirHp:
    vals, ws = bt.their_max_hp_support(species)
    m, w = torch.tensor([vals], dtype=torch.float64), torch.tensor([ws], dtype=torch.float64)
    return TheirHp((w / m).sum(-1), (w * m).sum(-1), m, w)


def test_eight_atoms_price_p_ko_within_the_declared_bound(tables: Dict[str, torch.Tensor], top: Any) -> None:
    """THE quantisation contract (`designs/model/op_contracts.md`): over every species with a usage record, every
    reported percent 1 … 100 and a grid of damages, the 8-atom mixture's P(KO) is within 0.03 of the FULL support's —
    for a fixed damage (20 / 40 / 100 HP) and for a formula hit (the 16 rolls + the crit). ONE atom at the harmonic
    mean misses by more than 0.5 on the same grid: a KO is a threshold in the max HP, so a point cannot price it."""
    fr = (torch.arange(1, 101, dtype=torch.float64) / 100.0)[:, None]                 # [100, 1]
    fixed = torch.tensor([20.0, 40.0, 100.0], dtype=torch.float64)[None, :]
    dmg = torch.tensor([30.0, 60.0, 90.0, 120.0, 180.0, 240.0, 320.0], dtype=torch.float64)[None, :]
    worst_fixed = worst_roll = worst_point = 0.0
    for sid in gen3_data.species.base_form_ids():
        if not gen3_data.priors.all_spreads(sid) or sid == "shedinja":
            continue
        num = int(gen3_data.species.get(sid).num)
        full = _full_support_belief(sid)
        got = TheirHp(tables["THEIR_HP_INV"][num].double()[None], tables["THEIR_HP_MEAN"][num].double()[None],
                      tables["THEIR_HP_ATOMS"][num].double()[None], tables["THEIR_HP_WEIGHTS"][num].double()[None])
        point = TheirHp(got.inv, got.mean, (1.0 / got.inv)[:, None], torch.ones(1, 1, dtype=torch.float64))
        want_f = their_fixed_ko(full, fr, fixed)
        worst_fixed = max(worst_fixed, float((their_fixed_ko(got, fr, fixed) - want_f).abs().max()))
        worst_point = max(worst_point, float((their_fixed_ko(point, fr, fixed) - want_f).abs().max()))
        want_r = their_ko_given_hit(dmg, 2.0 * dmg, full, fr, None)
        worst_roll = max(worst_roll, float((their_ko_given_hit(dmg, 2.0 * dmg, got, fr, None) - want_r).abs().max()))
    assert worst_fixed <= 0.03 and worst_roll <= 0.03, (worst_fixed, worst_roll)
    assert worst_point > 0.5, worst_point


# ------------------------------------------------------------------------------------------------ the op's kernels
def _outgoing(op: DamageOperator, ctx: Any) -> torch.Tensor:
    """``[low, high, crit, pko]`` of OUR move slot 0 into the opp active — from all THREE outgoing kernels, which
    must agree (one belief, one arithmetic)."""
    blk = op._outgoing_block(ctx)[0, 0:4]
    omx = op._outgoing_matrix(ctx)[0, 0:4]
    oax = op._outgoing_attacker_matrix(ctx)[0, 0:_DMG_OAX_PER_MOVE]
    for other in (omx, oax):
        assert torch.allclose(other, blk, atol=1e-6), (blk, other)
    return blk


def test_seismic_toss_into_a_35_percent_skarmory_is_a_small_probability(op: DamageOperator) -> None:
    """THE backlog case, end to end. A reported 35 % leaves a 334-HP Skarmory (252 HP EVs, the modal set) more than
    113 HP — Seismic Toss's 100 does not KO it; only a Skarmory under ~294 max HP can die, under 286 surely. So P(KO)
    is the Smogon mass of those sets, ~5 %, where the old op read 1 (35 % of an assumed 271 = 94.85 HP). The damage is
    the expected fraction ``100 · E[1 / M]`` (0.303), where it read 100 / 271 = 0.369. At 30 % nearly every Skarmory
    dies (100.2 HP left at most for 334): P(KO) ~0.96."""
    out = _outgoing(op, _nf_ctx("seismictoss", _SKARMORY, "STEEL", "FLYING", opp_hp=0.35))
    want = _exact_fixed_ko("skarmory", 0.35, 100.0)
    assert 0.02 < want < 0.10                                              # the fixture still means what it says
    assert float(out[3]) == pytest.approx(want, abs=0.02)
    inv = sum(w / m for m, w in zip(*_smogon_support("skarmory")))
    assert float(out[1]) == pytest.approx(100.0 * inv, rel=1e-5) and float(out[0]) == float(out[1])
    assert 100.0 * inv < 0.31 < 100.0 / 271.0
    low = _outgoing(op, _nf_ctx("seismictoss", _SKARMORY, "STEEL", "FLYING", opp_hp=0.30))
    assert float(low[3]) == pytest.approx(_exact_fixed_ko("skarmory", 0.30, 100.0), abs=0.02) and float(low[3]) > 0.9
    dead_sure = _outgoing(op, _nf_ctx("seismictoss", _SKARMORY, "STEEL", "FLYING", opp_hp=0.25))
    assert float(dead_sure[3]) == 1.0                                      # 83.5 HP at most, whatever its spread


def test_seismic_toss_into_blissey_reads_the_expected_fraction_and_a_graded_ko(op: DamageOperator) -> None:
    """Blissey (651 … 714): Seismic Toss is ``100 · E[1 / M]`` of its max HP (0.1495; the neutral 100 / 651 = 0.1536
    overstated it), a full one never dies to it, and at a reported 15 % the 651-HP majority does (97.65 HP left at
    most) while a 714-HP one almost never does (more than 99.96 left): P(KO) ~0.71, where the old cut read 1."""
    full = _outgoing(op, _nf_ctx("seismictoss", _BLISSEY, "NORMAL"))
    inv = sum(w / m for m, w in zip(*_smogon_support("blissey")))
    assert float(full[1]) == pytest.approx(100.0 * inv, rel=1e-5) and float(full[3]) == 0.0
    assert 100.0 / 714.0 < 100.0 * inv < 100.0 / 651.0
    part = _outgoing(op, _nf_ctx("seismictoss", _BLISSEY, "NORMAL", opp_hp=0.15))
    want = _exact_fixed_ko("blissey", 0.15, 100.0)
    assert 0.5 < want < 0.9 and float(part[3]) == pytest.approx(want, abs=0.02)


def test_a_formula_hits_ko_is_the_mixture_over_their_max_hp(op: DamageOperator) -> None:
    """`_rolls_their`: the damage fractions are the damage in HP × ``E[1 / M]``, and P(KO) is `ko_exact.ko_given_hit`
    over each max-HP atom's percent bin, weighted — equal to a hand-rolled sum over the op's own atoms, and different
    from the same closed form at ONE max HP (the neutral 271 the op assumed)."""
    their = their_max_hp(op, torch.tensor([_SKARMORY])).lift(slice(None), None)        # [1, 1]; atoms [1, 1, K]
    dmg_ns = torch.tensor([[60.0, 100.0, 118.0, 140.0]])
    screen, acc, frac = torch.ones(1, 4), torch.ones(1, 4), torch.tensor([[0.35]])
    high, _low, _crit, ko = op._rolls_their(dmg_ns, screen, their, frac, acc, crit_p=None)
    want = torch.zeros(1, 4)
    for k in range(bt.THEIR_HP_ATOMS):
        m = op.THEIR_HP_ATOMS[_SKARMORY, k]
        lo, hi = opp_hp_bounds(frac * m, frac, m, 1.0)
        want = want + op.THEIR_HP_WEIGHTS[_SKARMORY, k] * ko_given_hit(dmg_ns, 2.0 * dmg_ns, lo, hi, None)
    assert torch.allclose(ko, want, atol=1e-6)
    lo, hi = opp_hp_bounds(frac * 271.0, frac, 271.0, 1.0)
    neutral = ko_given_hit(dmg_ns, 2.0 * dmg_ns, lo, hi, None)
    assert float((ko - neutral).abs().max()) > 0.5                 # the 100-HP hit: sure at 271, rare at 334
    assert torch.allclose(high, (dmg_ns * op.THEIR_HP_INV[_SKARMORY]).clamp(max=float(high.max())), atol=1e-6)


def test_a_ko_every_max_hp_grants_is_exactly_one_and_one_none_grants_exactly_zero(top: Any) -> None:
    """Where the atoms AGREE the mixture returns their value EXACTLY (`_mix_atoms`): a full-HP mon of every species
    against 2000 fixed damage, or a hit whose lowest roll exceeds any max HP, is P(KO) == 1.0, and 1 HP of damage
    into a full mon == 0.0 — never the fp32 weights' own sum (1 ± 1.2e-7, which a plain ``Σ w·x`` returns for some
    species)."""
    sums = top.THEIR_HP_WEIGHTS.sum(-1)
    assert int((sums != 1.0).sum()) > 0                      # the premise: the plain sum is not 1 for some species
    their = their_max_hp(top, torch.arange(_N_SPECIES))
    full = torch.ones(_N_SPECIES)
    assert bool((their_fixed_ko(their, full, torch.full((_N_SPECIES,), 2000.0)) == 1.0).all())
    big, small = torch.full((_N_SPECIES,), 2000.0), torch.full((_N_SPECIES,), 1.0)
    assert bool((their_ko_given_hit(big, 2.0 * big, their, full, None) == 1.0).all())
    live = torch.ones(_N_SPECIES, dtype=torch.bool)
    live[_SHEDINJA] = False                                  # 1 damage is its whole HP: the boundary, not a miss
    assert bool((their_fixed_ko(their, full, small)[live] == 0.0).all())
    assert bool((their_ko_given_hit(small, 2.0 * small, their, full, None)[live] == 0.0).all())
    half = their_fixed_ko(their_max_hp(top, torch.tensor([_BLISSEY])), torch.tensor([0.15]), torch.tensor([100.0]))
    assert 0.0 < float(half) < 1.0                           # and a graded KO stays graded


def test_their_hp_in_points_is_the_expected_max_hp(top: Any) -> None:
    """THEIR HP as an operand in POINTS (their own Endeavor into us; Eruption's is a fraction) is ``fraction · E[M]``;
    on the scale a damage fraction divides by it is ``fraction / E[1 / M]`` — two different reads of one belief."""
    their = their_max_hp(top, torch.tensor([_SKARMORY]))
    half = torch.tensor([0.5])
    assert float(their.points(half)) == pytest.approx(0.5 * float(top.THEIR_HP_MEAN[_SKARMORY]), rel=1e-6)
    assert float(their.frac_points(half)) == pytest.approx(0.5 / float(top.THEIR_HP_INV[_SKARMORY]), rel=1e-6)
    assert float(their.points(half)) > float(their.frac_points(half)) > 0.5 * 271.0


def test_an_averaged_slot_reads_the_species_mixture_and_never_divides_by_zero(top: Any) -> None:
    """A slot whose species is a distribution (hidden under the prior; X5's OTHER): ``E[1 / M]`` and ``E[M]`` are the
    mixture's, the atoms collapse to the one point ``1 / E[1 / M]`` (no P(KO) is taken there); a concrete slot is
    untouched; an all-zero distribution reads the finite unowned row, not 1 / 0."""
    their = their_max_hp(top, torch.tensor([[_SKARMORY, _BLISSEY, 0, 0, 0, 0]]))
    p = torch.zeros(1, _N_SPECIES)
    p[0, _SKARMORY], p[0, _BLISSEY] = 0.25, 0.75
    where = torch.tensor([[False, True, True, False, False, False]])
    out = expected_their_max_hp(top, their, p, where)
    inv = 0.25 * float(top.THEIR_HP_INV[_SKARMORY]) + 0.75 * float(top.THEIR_HP_INV[_BLISSEY])
    assert out.inv[0, 1:3].tolist() == pytest.approx([inv, inv], rel=1e-6)
    assert float(out.inv[0, 0]) == float(top.THEIR_HP_INV[_SKARMORY])
    assert torch.equal(out.atoms[0, 0], top.THEIR_HP_ATOMS[_SKARMORY])
    assert out.atoms[0, 1].tolist() == pytest.approx([1.0 / inv] * bt.THEIR_HP_ATOMS, rel=1e-6)
    assert out.weights[0, 1].tolist() == [1.0] + [0.0] * (bt.THEIR_HP_ATOMS - 1)
    empty = expected_their_max_hp(top, their, torch.zeros(1, TEAM_SIZE, _N_SPECIES), torch.ones(1, TEAM_SIZE).bool())
    assert bool(torch.isfinite(empty.maxhp).all()) and float(empty.mean[0, 3]) == 141.0


# ------------------------------------------------------------------------------------------------ the end of turn
def test_their_end_of_turn_amounts_are_the_expected_integer_rule(rows: torch.Tensor) -> None:
    """THEIR burned Skarmory loses ``E[⌊M / 8⌋ / M]`` of its max HP (0.1229: 41 of 334 — the integer rule, over the
    belief), not the nominal 1/8; its Leftovers heal ``E[⌊M / 16⌋ / M]`` (0.0600, not 0.0625); a seeded one at 10 %
    drains that expectation too. Each equals the independent sum over the Smogon support."""
    fe = _legacy_policy().features_extractor
    slot = TEAM_SIZE + 1
    vals, ws = _smogon_support("skarmory")
    e8 = sum(w * (m // 8) / m for m, w in zip(vals, ws))
    e16 = sum(w * (m // 16) / m for m, w in zip(vals, ws))
    assert e8 < 0.1235 and e16 < 0.0605                                    # below the nominal 1/8 and 1/16
    burned = _mon(_blank(fe, rows), slot, species="skarmory", types=("STEEL", "FLYING"), status="brn")
    assert float(_eot(fe, burned)[slot, _COL["status"]]) == pytest.approx(-e8, rel=1e-5)
    lefties = _mon(_blank(fe, rows), slot, species="skarmory", types=("STEEL", "FLYING"), item="leftovers")
    assert float(_eot(fe, lefties)[slot, _COL["leftovers"]]) == pytest.approx(e16, rel=1e-5)
    seeded = _vol(_mon(_blank(fe, rows), TEAM_SIZE, species="skarmory", types=("STEEL", "FLYING"), hp=0.10), 1,
                  "leechseed")
    drain = float(_eot(fe, seeded)[TEAM_SIZE, _COL["leech_drain"]])
    assert drain == pytest.approx(-0.10, rel=1e-5)                              # capped at its HP
    seeded = _vol(_mon(_blank(fe, rows), TEAM_SIZE, species="skarmory", types=("STEEL", "FLYING"), hp=0.50), 1,
                  "leechseed")
    assert float(_eot(fe, seeded)[TEAM_SIZE, _COL["leech_drain"]]) == pytest.approx(-e8, rel=1e-5)


def test_the_residual_helpers_refuse_a_denominator_that_is_not_a_residuals(top: Any) -> None:
    ids = torch.tensor([_SKARMORY])
    assert float(their_residual_frac(top, ids, 8.0)) == float(top.THEIR_HP_RESIDUAL[_SKARMORY, 1])
    with pytest.raises(ValueError, match="not a gen-3 residual"):
        their_residual_frac(top, ids, 6.0)


# ------------------------------------------------------------------------------------------------ the leak guard
def test_the_helper_refuses_anything_but_integer_species_ids(top: Any, rows: torch.Tensor) -> None:
    """LEAK GUARD 1 (the throwing guard): their max HP is priced from the SPECIES alone. A float tensor — a spread
    block, a stat, an HP fraction, the opponent's true max HP — or a bool mask RAISES instead of indexing; a
    distribution that is not over the species axis (an 18-wide spread) RAISES in the averaging helper."""
    for bad in (torch.tensor([334.0]), torch.tensor([0.35]), torch.zeros(1, 6, POKEMON_SPREAD_DIM),
                torch.tensor([True, False])):
        with pytest.raises(TypeError, match="species_ids must be integer dex nums"):
            their_max_hp(top, bad)
        with pytest.raises(TypeError, match="species_ids must be integer dex nums"):
            their_residual_points(top, bad, 8.0)
    their = their_max_hp(top, torch.zeros(1, TEAM_SIZE, dtype=torch.long))
    with pytest.raises(ValueError, match="not the species axis"):
        expected_their_max_hp(top, their, torch.zeros(1, TEAM_SIZE, POKEMON_SPREAD_DIM),
                              torch.ones(1, TEAM_SIZE).bool())


def _production_policy() -> Any:
    with open(repo_path("designs", "production_config.json")) as fh:
        cfg = json.load(fh)
    return _build_real_policy(**{k: v for k, v in cfg.items() if not isinstance(v, (dict, list))})[0].policy


def test_the_op_and_the_end_of_turn_rule_never_read_the_opponents_spread(rows: torch.Tensor) -> None:
    """LEAK GUARD 2 (the PRODUCTION forward, real rows): the observation's opponent spread cells are the opponent's
    TRUE IVs / EVs / nature under the diagnostic oracle reveal. Whatever they hold — zeros, a max-HP spread, noise —
    the damage operator's whole output, EVERY kernel of it the extractor calls (each method of the op is wrapped: the
    pairwise kernels never pass through ``forward``) and the end-of-turn rule's are BIT-IDENTICAL: nothing of their
    true spread reaches a max HP, a damage fraction, a P(KO) or a residual. A site that prices their max HP from the
    spread cells (the leak) fails here; so does one that reads any of their stats from them."""
    fe = _production_policy().features_extractor
    seen: Dict[str, List[torch.Tensor]] = {"op": [], "eot": []}
    kernels: List[Tuple[str, torch.Tensor]] = []

    def _tensors(out: Any) -> List[torch.Tensor]:
        if isinstance(out, torch.Tensor):
            return [out.detach().clone()]
        if isinstance(out, (tuple, list)):
            return [t for o in out for t in _tensors(o)]
        return []

    def _recorded(name: str, fn: Any) -> Any:
        def call(*a: Any, **k: Any) -> Any:
            out = fn(*a, **k)
            kernels.extend((name, t) for t in _tensors(out))
            return out
        return call
    op_cls = type(fe.damage_op)
    for name in sorted({n for c in op_cls.__mro__ if c.__module__.startswith("agents.model.damage_op")
                        for n, v in vars(c).items() if isinstance(v, types.FunctionType) and not n.startswith("__")
                        and n != "forward"}):
        setattr(fe.damage_op, name, _recorded(name, getattr(fe.damage_op, name)))
    hooks = [fe.damage_op.register_forward_hook(lambda _m, _i, out: seen["op"].append(out.detach().clone())),
             fe.eot_residual_rule.register_forward_hook(lambda _m, _i, out: seen["eot"].append(out.detach().clone()))]
    sl = slice(POKEMON_SPREAD_OFFSET, POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM)
    gen = torch.Generator().manual_seed(13)
    kind = ["zeros"]

    def plant(_m: Any, _inp: Any, ctx: Any) -> Any:
        """Replace the unpacked context's opponent spread cells (a forward hook's return value IS the output)."""
        pp = ctx.pokemon_part.clone()
        block = pp[:, TEAM_SIZE:, sl]
        if kind[0] == "zeros":
            block.zero_()
        elif kind[0] == "max_hp":                        # IV 31, 252 HP EVs, a neutral nature: the bulkiest truth
            block.zero_()
            block[..., 0:6] = 1.0                        # the six IVs
            block[..., 6] = 1.0                          # the HP EV
            block[..., 12] = 1.0                         # spread_known, as the oracle reveal sets it
            block[..., 13:18] = 1.0                      # the nature modifiers
        else:
            block.copy_(torch.rand(block.shape, generator=gen))
        return dataclasses.replace(ctx, pokemon_part=pp)
    hooks.append(fe.unpack.register_forward_hook(plant))
    try:
        marks = []
        for kind[0] in ("zeros", "max_hp", "noise"):
            with torch.no_grad():
                fe({"observation": rows})
            marks.append(len(kernels))
    finally:
        for h in hooks:
            h.remove()
    assert len(seen["op"]) == 3 and len(seen["eot"]) >= 3, {k: len(v) for k, v in seen.items()}
    assert float(seen["op"][0].abs().sum()) > 0.0 and float(seen["eot"][0].abs().sum()) > 0.0
    per = len(seen["eot"]) // 3
    for i in (1, 2):
        assert torch.equal(seen["op"][0], seen["op"][i]), i
        for j in range(per):
            assert torch.equal(seen["eot"][j], seen["eot"][i * per + j]), (i, j)
    # every kernel: the same calls in the same order, each returned tensor bit-identical; the three that read
    # `their_hp_slots` / `their_residual_frac_slots` outside `forward` did run
    n = marks[0]
    assert n > 0 and marks == [n, 2 * n, 3 * n], marks
    names = {name for name, _t in kernels[:n]}
    assert {"_outgoing_matrix", "_believed_attackers", "pairwise_status_consequence"} <= names, sorted(names)
    for i in (1, 2):
        for j in range(n):
            assert kernels[j][0] == kernels[i * n + j][0], (i, j)
            assert torch.equal(kernels[j][1], kernels[i * n + j][1]), (i, kernels[j][0])


#: A forward-module spelling that prices THEIR max HP outside `damage_kinds` (comments / strings stripped first):
#: a direct read of the belief's tables, a hand-built `TheirHp`, the L100 HP constants (``+ 110`` of the formula, the
#: neutral ``141``), or a base-HP read (nothing but a max-HP formula needs a species' base HP).
_IDX = r"(?:[^\[\]]|\[[^\[\]]*\])*"          # an index expression, one level of nested brackets allowed
_FORBIDDEN = {
    "the belief's tables": re.compile(r"THEIR_HP_"),
    "a hand-built belief": re.compile(r"\bTheirHp\("),
    "the L100 HP constant": re.compile(r"(?<![\w.])1(10|41)(\.0)?\b"),
    "a base-HP read": re.compile(r"_BS_HP|\bbase_hp\b|BASE_STATS\[" + _IDX + r"\]\s*\[\s*(\.\.\.|:)\s*,\s*0\s*\]"
                                 r"|BASE_STATS\[" + _IDX + r",\s*0\s*\]|\b[ad]_base\[\s*(\.\.\.|:)\s*,\s*0\s*\]"),
}
#: Declared exceptions, (module, stripped line): OUR exact max HP (`eot_residual._our_max_hp`, our six slots only).
_ALLOWED = {
    ("eot_residual", "base_hp = op.BASE_STATS[our_ids][..., 0].to(ctx.pokemon_part.dtype)"),
    ("eot_residual", "ours = our_max_hp(base_hp, iv[..., 0], ev_points[..., 0])"),
}
#: The spellings this change REMOVED, verbatim: each must trip the guard (its teeth).
_REMOVED = (
    "opp_maxhp = 2.0 * d_base[:, 0] + 31.0 + 110.0",
    "opp_maxhp = 2.0 * d_base[..., 0] + 31.0 + 110.0",
    "e_maxhp = 2.0 * (sp_probs @ self.BASE_STATS[:, 0]) + 31.0 + 110.0",
    "opp_cur_hp_att = opp_hp_frac * (2.0 * self.BASE_STATS[ctx.species_ids[ar, opp_act], 0] + 31.0 + 110.0)",
    "atk_cur = opp_hp_frac * (2.0 * a_base[:, 0] + 31.0 + 110.0)",
    "atk_cur_j = hp_j * (2.0 * a_base[..., _BS_HP] + 31.0 + 110.0)",
    "theirs = 2.0 * base_hp[:, TEAM_SIZE:] + 31.0 + 110.0",
    "inv = self.THEIR_HP_INV[ctx.species_ids[:, TEAM_SIZE:]]",
    "their = TheirHp(1.0 / m, m, m[..., None], w)",
)


def _code_lines(path: Any) -> List[Tuple[int, str]]:
    out: Dict[int, List[str]] = {}
    for t in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        if t.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                      tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        out.setdefault(t.start[0], []).append(t.string)
    return [(n, " ".join(v)) for n, v in sorted(out.items())]


def _flat(code: str) -> str:
    return re.sub(r"\s+", " ", code.replace(" . ", ".").replace(" (", "(").replace("( ", "(").replace(" )", ")")
                  .replace(" [", "[").replace("[ ", "[").replace(" ]", "]").replace(" ,", ","))


def test_no_forward_module_prices_their_max_hp_outside_the_helper() -> None:
    """LEAK GUARD 3 / the one-spelling guard: every forward module but `damage_kinds` (which owns the helper) carries
    none of `_FORBIDDEN`'s spellings, so THEIR max HP is the Smogon belief BY CONSTRUCTION at every site — a new
    kernel that re-spells ``2·base + 141`` (the E13 bug), reads the tables around the integer-ids refusal, or builds
    a `TheirHp` from something else, fails here. The declared exceptions must each still exist."""
    hits = []
    for mod in FORWARD_MODULES:
        if mod == "damage_kinds":
            continue
        for n, code in _code_lines(src_path("agents", "model", f"{mod}.py")):
            flat = _flat(code)
            hits += [(mod, n, flat, why) for why, pat in _FORBIDDEN.items() if pat.search(flat)]
    allowed = {(m, re.sub(r"\s+", "", a)) for m, a in _ALLOWED}
    bad = [h for h in hits if (h[0], re.sub(r"\s+", "", h[2])) not in allowed]
    assert not bad, bad
    seen = {(h[0], re.sub(r"\s+", "", h[2])) for h in hits}
    assert allowed <= seen, ("a declared exception no longer exists", allowed - seen)


def test_the_guard_has_teeth_on_every_spelling_this_change_removed() -> None:
    """Each line E13 deleted (the op's seven neutral estimates) and each way around the helper trips `_FORBIDDEN`;
    the helper's own module is the one exemption, and it is a forward module (so the scan's list is the real one)."""
    for line in _REMOVED:
        assert any(p.search(_flat(line)) for p in _FORBIDDEN.values()), line
    base_hp_reads = ("b = self.BASE_STATS[ctx.species_ids[:, TEAM_SIZE:]][..., 0]", "b = self.BASE_STATS[ids][:, 0]",
                     "b = self.BASE_STATS[ctx.species_ids[ar, opp_act], 0]", "b = d_base[..., 0]")
    for line in base_hp_reads:                                  # a base-HP read alone, no constant beside it
        assert _FORBIDDEN["a base-HP read"].search(_flat(line)), line
    assert not _FORBIDDEN["a base-HP read"].search("opp_def = 2.0 * self.BASE_STATS[ids][:, 2] + 36.0")
    assert "damage_kinds" in FORWARD_MODULES and "damage_op_blocks" in FORWARD_MODULES
    assert not any(p.search("crit_p = 1.0 / 16.0 + 0.110 * x[..., 10]") for p in _FORBIDDEN.values())
