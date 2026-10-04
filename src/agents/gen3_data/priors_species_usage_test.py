"""F-X5-47 (`gen3_smogon_species_usage_weighted_v1`, 2026-10-04): the Smogon SPECIES-USAGE marginal is each
species' RATING-WEIGHTED set total W (``Σ Abilities`` = Smogon's ``p.raw.weight``), the same population every
other Smogon prior is weighted by — not the UNWEIGHTED ``Raw count`` (every rating weight 1), whose share is
off by ×0.19-×1.75 by species (Shuckle 5.2× too high; Tyranitar 10 % too low).

Every test here FAILS on a revert of the fix: the facade's values (W, never Raw count), the three consumers'
tensors (the op's ``SPECIES_USAGE_PRIOR`` / the T0 marginal through ``build_species_usage_prior``, and the
co-occurrence lift's independence baseline), and the THROWING guard (a planted Raw-count table, one planted
species, the latest-month ``usage`` share the old code fell back to).
"""
from __future__ import annotations

import math

import pytest

from agents import gen3_data
from agents.gen3_data import priors
from agents.model import belief_tables, dex_ids

#: The num axis the model builders are sized with in their own tests (any size past the dex works).
_N_SPECIES = 400


def _records() -> dict:
    return {priors._species_id(k): v for k, v in priors.smogon_stats_raw()["data"].items()}


def _num(sid: str) -> int:
    sd = gen3_data.species.get(sid)
    assert sd is not None
    return int(sd.num)


def test_species_usage_is_the_weighted_set_total_not_raw_count():
    """Every value is the species' W = ``Σ Abilities`` (= ``Σ Moves / 4``), and NONE is its ``Raw count``:
    W / Raw count runs 0.10-0.94 on the committed window, so every species is >= 6 % from its raw count."""
    u = priors.species_usage()
    recs = _records()
    assert len(u) == len(recs) == 216
    for sid, v in u.items():
        rec = recs[sid]
        assert v == pytest.approx(sum(rec["Abilities"].values()), rel=1e-12), sid
        assert v == pytest.approx(sum(rec["Moves"].values()) / 4.0, rel=1e-9), sid
        assert v / float(rec["Raw count"]) < 0.94, sid


def test_the_slot_prior_is_the_weighted_share():
    """The op's ``SPECIES_USAGE_PRIOR`` and the T0 marginal (both ``build_species_usage_prior``) carry the
    W share: a ratio of two covered species is their W ratio. Under Raw count Shuckle sat at 5.2x its
    weighted share and Tyranitar 10 % below it."""
    prior = dex_ids.build_species_usage_prior(_N_SPECIES).double()
    recs = _records()
    w = {s: sum(r["Abilities"].values()) for s, r in recs.items()}
    raw = {s: float(r["Raw count"]) for s, r in recs.items()}
    tt, sk, sh = _num("tyranitar"), _num("skarmory"), _num("shuckle")
    assert float(prior[tt] / prior[sk]) == pytest.approx(w["tyranitar"] / w["skarmory"], rel=1e-5)
    assert float(prior[sh] / prior[tt]) == pytest.approx(w["shuckle"] / w["tyranitar"], rel=1e-5)
    # the defect's own scale, far from either side of any rounding boundary
    share_w = w["shuckle"] / sum(w.values())
    share_raw = raw["shuckle"] / sum(raw.values())
    assert share_raw / share_w > 5.0
    assert float(prior[sh]) < 1.5 * share_w
    assert float(prior[tt]) > 0.085                       # W share 0.0863; the Raw count share was 0.0781


def test_the_cooccur_baseline_divides_by_the_weighted_share():
    """The co-occurrence prior's marginal is log(W share), and its lift's independence baseline divides the
    WEIGHTED teammate conditional by the W share renormalized without t — one population on both sides."""
    log_marginal, log_lift = belief_tables.build_species_cooccur_prior(_N_SPECIES)
    prior = dex_ids.build_species_usage_prior(_N_SPECIES).double()
    tt, sk = _num("tyranitar"), _num("skarmory")
    w = {s: sum(r["Abilities"].values()) for s, r in _records().items()}
    assert float(log_marginal[tt] - log_marginal[sk]) == pytest.approx(
        math.log(w["tyranitar"] / w["skarmory"]), abs=1e-5)     # Raw count: 0.074 nats off
    assert float(log_marginal[tt]) == pytest.approx(math.log(float(prior[tt])), abs=1e-6)
    p_cond = sum(p for s, p in priors.teammates("tyranitar").items()
                 if gen3_data.species.get(s) is not None and _num(s) == sk)
    expected = float(prior[sk]) / (1.0 - float(prior[tt]))
    assert float(log_lift[sk, tt]) == pytest.approx(math.log(p_cond / expected), abs=1e-5)


def test_the_guard_raises_on_a_planted_raw_count_table():
    recs = _records()
    planted = {s: float(r["Raw count"]) for s, r in recs.items()}
    with pytest.raises(priors.PriorInvariantError, match="F-X5-47"):
        priors._checked_species_usage(planted)
    # the latest-month weighted `usage` SHARE the old code fell back to is not a count either
    latest = {s: float(r["usage"]) for s, r in recs.items()}
    with pytest.raises(priors.PriorInvariantError, match="F-X5-47"):
        priors._checked_species_usage(latest)
    priors._checked_species_usage(priors.species_usage())  # the facade's own table passes


def test_the_guard_raises_on_a_single_raw_count_species():
    """One species at the SMALLEST gap the Raw count table carried (Raw count / W >= 1.067 on every species)
    still throws, and so does a table missing a species."""
    table = dict(priors.species_usage())
    table["tyranitar"] *= 1.06
    with pytest.raises(priors.PriorInvariantError, match="tyranitar"):
        priors._checked_species_usage(table)
    table = dict(priors.species_usage())
    del table["shuckle"]
    with pytest.raises(priors.PriorInvariantError, match="species set"):
        priors._checked_species_usage(table)
