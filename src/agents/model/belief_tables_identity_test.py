"""BIT-IDENTITY of the belief-table builders against the per-element TENSOR code they replaced (2026-10-20, perf).

`build_opp_spread_prior`, `build_species_nature_prior`, `build_species_ev_prior`, `build_move_prior_logits` and
`build_species_cooccur_prior` and `build_item_prior` were rewritten to accumulate in Python floats (float64 tables) or numpy float32 scalars
(float32 accumulators) and build each tensor ONCE, instead of ~400k per-element tensor reads, writes and adds per
production model build. Every `_ref_*` below is the ORIGINAL code, verbatim but for its comments and its argument /
GIGO guards; each test compares the WHOLE table at the production sizes with `torch.equal`, so any change to a
single output bit (an add re-ordered, a float32 rounding moved, a cell written twice) fails here.
`build_species_speed_mix` (first rewritten in fa3b3f72, whose per-element-tensor reference test stays in
`belief_tables_test.py`) is compared here, whole, against that commit's Python-float version.
"""
from __future__ import annotations

import math
from typing import cast, Dict, Tuple

import pytest
import torch

from agents import gen3_data
from agents.gen3_data.species import SpeciesData
from agents.model import belief_tables as bt
from agents.model.belief_tables import (_COOCCUR_LIFT_CLAMP, _ILLEGAL_PROB, _NATURE_PRIOR_FLOOR, _PRIOR_FLOOR,
                                        _SPECIES_PRIOR_FLOOR, _SPREAD_BASE_IDX, N_NATURES, N_SPREAD_STATS,
                                        SPECIES_CLAUSE_LOGIT, SPEED_LATTICE, SPREAD_STAT_COLS,
                                        format_banned_move_cells)
from agents.model.dex_ids import (_belief_num, _hp_typed_nums, build_species_usage_prior,
                                  format_banned_species_nums)

#: The production table sizes (`state_encoder` layout: max_species, max_moves, max_items).
N_SPECIES = 400
N_MOVES = 400
N_ITEMS = 600


def _ref_opp_spread_prior(n_species: int) -> torch.Tensor:
    prior = torch.zeros(n_species, N_SPREAD_STATS, 2, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        snum = sd.num
        if not (0 <= snum < n_species):
            continue
        spr = gen3_data.priors.spreads(sid)
        for j, stat in enumerate(SPREAD_STAT_COLS):
            base = int(sd.base_stats.get(stat, 0))
            evi = _SPREAD_BASE_IDX[stat]
            m1 = m2 = wsum = 0.0
            for nature, evs, w in spr:
                nd = gen3_data.natures.get(str(nature).lower())
                mult = nd.multipliers.get(stat, 1.0) if nd is not None else 1.0
                val = float(gen3_data.priors.gen3_stat(base, int(evs[evi]), mult))
                m1 += w * val
                m2 += w * val * val
                wsum += float(w)
            if wsum <= 0.0:
                neutral = float(gen3_data.priors.gen3_stat(base, 0, 1.0))
                maxed = float(gen3_data.priors.gen3_stat(base, 252, 1.1))
                prior[snum, j, 0] = neutral
                prior[snum, j, 1] = max(1.0, (maxed - neutral) / 2.0)
            else:
                mean = m1 / wsum
                var = max(0.0, m2 / wsum - mean * mean)
                prior[snum, j, 0] = mean
                prior[snum, j, 1] = max(1.0, var ** 0.5)
    return prior


def _ref_species_nature_prior(n_species: int) -> torch.Tensor:
    logprior = torch.full((n_species, N_NATURES), 1.0 / N_NATURES, dtype=torch.float32).log()
    nat_raw = gen3_data.natures.raw()
    for sid in gen3_data.species.base_form_ids():
        snum = cast(SpeciesData, gen3_data.species.get(sid)).num
        if not (0 <= snum < n_species):
            continue
        counts = torch.zeros(N_NATURES, dtype=torch.float32)
        for nature, _evs, w in gen3_data.priors.spreads(sid):
            nd = nat_raw.get(str(nature).lower())
            if nd is None:
                continue
            counts[int(nd["num"])] += float(w)
        tot = float(counts.sum())
        if tot <= 0.0:
            continue
        p = counts / tot
        p = (1.0 - _NATURE_PRIOR_FLOOR) * p + _NATURE_PRIOR_FLOOR / N_NATURES
        logprior[snum] = p.log()
    return logprior


def _ref_species_ev_prior(n_species: int) -> torch.Tensor:
    ev = torch.zeros(n_species, N_SPREAD_STATS, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        snum = cast(SpeciesData, gen3_data.species.get(sid)).num
        if not (0 <= snum < n_species):
            continue
        acc = torch.zeros(N_SPREAD_STATS, dtype=torch.float32)
        wsum = 0.0
        for _nature, evs, w in gen3_data.priors.spreads(sid):
            for j, stat in enumerate(SPREAD_STAT_COLS):
                acc[j] += float(w) * float(evs[_SPREAD_BASE_IDX[stat]])
            wsum += float(w)
        if wsum > 0.0:
            ev[snum] = acc / wsum
    return ev


def _ref_move_prior_logits(n_species: int, n_moves: int, floor: float = _PRIOR_FLOOR) -> torch.Tensor:
    eps = _ILLEGAL_PROB
    prob = torch.full((n_species, n_moves), eps, dtype=torch.float64)
    covered = torch.zeros(n_species, dtype=torch.bool)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        snum = sd.num
        if not (0 <= snum < n_species):
            continue
        covered[snum] = True
        legal = gen3_data.learnset.get_legal_moves(sid)
        if legal is None:
            prob[snum, :] = floor
        else:
            for move_id in legal:
                md = gen3_data.moves.get(move_id)
                if md is not None:
                    bnum = _belief_num(move_id, md)
                    if 0 <= bnum < n_moves:
                        prob[snum, bnum] = floor
                    if move_id == "hiddenpower":
                        for tnum in _hp_typed_nums():
                            if 0 <= tnum < n_moves:
                                prob[snum, tnum] = floor
        usage: Dict[int, float] = {}
        for move_id, p in gen3_data.priors.moves(sid).items():
            md = gen3_data.moves.get(move_id)
            if md is None:
                continue
            bnum = _belief_num(move_id, md)
            if 0 <= bnum < n_moves:
                usage[bnum] = usage.get(bnum, 0.0) + float(p)
            if move_id.startswith("hiddenpower") and 0 <= md.num < n_moves and md.num != bnum:
                usage[md.num] = usage.get(md.num, 0.0) + float(p)
        for num, u in usage.items():
            if u > float(prob[snum, num]):
                prob[snum, num] = u
    prob[~covered, :] = floor
    for snum, mnum in format_banned_move_cells(n_species, n_moves):
        prob[snum, mnum] = eps
    prob = prob.clamp(eps, 1.0 - eps)
    return torch.logit(prob).to(torch.float32)


def _ref_species_cooccur_prior(n_species: int) -> Tuple[torch.Tensor, torch.Tensor]:
    usage_prior = build_species_usage_prior(n_species)
    log_marginal = usage_prior.clamp_min(_SPECIES_PRIOR_FLOOR).log()
    log_marginal[0] = math.log(_SPECIES_PRIOR_FLOOR)
    for snum in format_banned_species_nums(n_species):
        log_marginal[snum] = SPECIES_CLAUSE_LOGIT

    log_lift = torch.zeros((n_species, n_species), dtype=torch.float32)
    base_ids = set(gen3_data.species.base_form_ids())
    for tid in base_ids:
        td = gen3_data.species.get(tid)
        if td is None or not (0 < td.num < n_species):
            continue
        mates = gen3_data.priors.teammates(tid)
        if not mates:
            continue
        base_renorm = max(1.0 - float(usage_prior[td.num]), 1e-6)
        cond_by_num: dict = {}
        for sid, p_cond in mates.items():
            sd = cast(SpeciesData, gen3_data.species.get(sid))
            if sd is None or not (0 < sd.num < n_species) or sd.num == td.num:
                continue
            cond_by_num[sd.num] = cond_by_num.get(sd.num, 0.0) + float(p_cond)
        for snum, p_cond in cond_by_num.items():
            expected = max(float(usage_prior[snum]), _SPECIES_PRIOR_FLOOR) / base_renorm
            lift = math.log(max(p_cond, 1e-9) / expected)
            log_lift[snum, td.num] = max(-_COOCCUR_LIFT_CLAMP, min(_COOCCUR_LIFT_CLAMP, lift))
    return log_marginal, log_lift


def _ref_species_speed_mix(n_species: int) -> torch.Tensor:
    """fa3b3f72's Python-float speed mix (proven bit-identical to the per-element tensor code by
    `belief_tables_test::test_the_speed_mix_is_bit_identical_to_the_per_element_tensor_reference`), before the
    per-call nature / stat memos and the per-base uniform-fallback row."""
    acc = [[0.0] * SPEED_LATTICE for _ in range(n_species)]
    filled = [False] * n_species
    nature_spe = [float(m.get("spe", 1.0)) for m in gen3_data.natures.multipliers().values()]
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        snum = sd.num
        if not (0 <= snum < n_species):
            continue
        base = int(sd.base_stats.get("spe", 0))
        rows = []
        for n, evs, w in gen3_data.priors.all_spreads(sid):
            nd = gen3_data.natures.get(str(n).lower())
            rows.append((nd.multipliers.get("spe", 1.0) if nd is not None else 1.0,
                         int(evs[_SPREAD_BASE_IDX["spe"]]), float(w)))
        if not rows:
            rows = [(m, ev, 1.0) for m in nature_spe for ev in range(0, 253, 4)]
        tot = sum(w for _, _, w in rows)
        for mult, ev, w in rows:
            v = int(gen3_data.priors.gen3_stat(base, ev, mult))
            acc[snum][v] += w / tot
        filled[snum] = True
    for snum, f in enumerate(filled):
        if not f:
            acc[snum][0] = 1.0
    return torch.tensor(acc, dtype=torch.float64).to(torch.float32)


def _ref_item_prior(n_species: int, n_items: int) -> torch.Tensor:
    _FLOOR = 1e-5
    prior = torch.full((n_species, n_items), 1.0 / n_items, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        if not (0 <= sd.num < n_species):
            continue
        entry = gen3_data.priors.items(sid)
        if not entry:
            continue
        vec = torch.full((n_items,), _FLOOR, dtype=torch.float32)
        for item_id, p in entry.items():
            it = gen3_data.items.get(item_id)
            num = int(it.num) if it is not None else (0 if item_id == "nothing" else None)
            if num is not None and 0 <= num < n_items:
                vec[num] += float(p)
        prior[sd.num] = vec / float(vec.sum())
    from agents.gen3_data import format_spec
    for item_id in sorted(format_spec.active().banned_items):
        it = gen3_data.items.get(item_id)
        if it is not None and 0 <= int(it.num) < n_items:
            prior[:, int(it.num)] = 0.0
    prior = prior / prior.sum(dim=1, keepdim=True)
    return prior


def _assert_bit_identical(got: torch.Tensor, ref: torch.Tensor) -> None:
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert torch.equal(got, ref), f"{int((got != ref).sum())} cells differ"


def test_the_speed_mix_is_bit_identical_to_the_python_float_reference() -> None:
    _assert_bit_identical(bt.build_species_speed_mix(N_SPECIES), _ref_species_speed_mix(N_SPECIES))
    no_record = [sid for sid in gen3_data.species.base_form_ids() if not gen3_data.priors.all_spreads(sid)]
    assert len(no_record) >= 20          # precondition: the per-base uniform-fallback row is exercised


def test_the_item_prior_is_bit_identical_to_the_tensor_reference() -> None:
    _assert_bit_identical(bt.build_item_prior(N_SPECIES, N_ITEMS), _ref_item_prior(N_SPECIES, N_ITEMS))


def test_the_opp_spread_prior_is_bit_identical_to_the_tensor_reference() -> None:
    _assert_bit_identical(bt.build_opp_spread_prior(N_SPECIES), _ref_opp_spread_prior(N_SPECIES))


def test_the_static_stat_prior_is_bit_identical_to_the_tensor_reference() -> None:
    """`build_static_stat_prior` embeds the opp-spread prior: its non-HP block is the reference's."""
    _assert_bit_identical(bt.build_static_stat_prior(N_SPECIES)[:, 1:, :], _ref_opp_spread_prior(N_SPECIES))


def test_the_nature_prior_is_bit_identical_to_the_tensor_reference() -> None:
    _assert_bit_identical(bt.build_species_nature_prior(N_SPECIES), _ref_species_nature_prior(N_SPECIES))


def test_the_ev_prior_is_bit_identical_to_the_tensor_reference() -> None:
    _assert_bit_identical(bt.build_species_ev_prior(N_SPECIES), _ref_species_ev_prior(N_SPECIES))


@pytest.mark.parametrize("floor", [_PRIOR_FLOOR, 0.05])
def test_the_move_prior_is_bit_identical_to_the_tensor_reference(floor: float) -> None:
    _assert_bit_identical(bt.build_move_prior_logits(N_SPECIES, N_MOVES, floor=floor),
                          _ref_move_prior_logits(N_SPECIES, N_MOVES, floor=floor))


def test_the_cooccur_prior_is_bit_identical_to_the_tensor_reference() -> None:
    marginal, lift = bt.build_species_cooccur_prior(N_SPECIES)
    ref_marginal, ref_lift = _ref_species_cooccur_prior(N_SPECIES)
    _assert_bit_identical(marginal, ref_marginal)
    _assert_bit_identical(lift, ref_lift)
    assert int((lift != 0).sum()) > 1000             # precondition: the comparison covers a populated table


def test_the_float32_accumulators_are_exercised() -> None:
    """Precondition of the two float32-accumulator tests: rows whose float32 sum ROUNDS (a float64 sum would give
    a different float32), so an accumulator that silently switched precision could not pass them by luck."""
    rounds = 0
    for sid in gen3_data.species.base_form_ids():
        ws = [float(w) for _n, _e, w in gen3_data.priors.spreads(sid)]
        f32 = torch.zeros((), dtype=torch.float32)
        for w in ws:
            f32 += w
        rounds += int(float(f32) != float(torch.tensor(sum(ws), dtype=torch.float32)))
    assert rounds >= 10
