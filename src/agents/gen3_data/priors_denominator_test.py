"""F-X5-41 (`gen3_smogon_prior_denominator_v1`, 2026-10-04): the Smogon MOVE prior is the chaos ``Moves``
over the species' RATING-WEIGHTED set total W (``Σ Abilities`` = Smogon's ``p.raw.weight``), not over the
UNWEIGHTED ``Raw count`` — which deflated every move prior by a species-dependent 0.10-0.94 (Skarmory Spikes
0.547, per-species sums 0.41-3.75 where a set runs ~4).

Every test here FAILS on a revert of the fix: the committed table read through the facade (Spikes, the
four-slot mass), the tool's derivation (it must reproduce the committed file), and the THROWING guards (a
planted deflated table, a planted inconsistent chaos record).
"""
from __future__ import annotations

import copy
import json
import statistics

import pytest

from agents.gen3_data import priors
from tools.smogon_stats_downloader import compute_priors as CP
from utils.paths import repo_path


def _chaos() -> dict:
    return copy.deepcopy(priors.smogon_stats_raw())


def _deflated_table(chaos: dict) -> dict:
    """The PRE-FIX derivation, verbatim: Moves[m] / Raw count, clamped at 1."""
    species = json.loads(repo_path("data", "pokemon", "gen3_species.json").read_text())
    out: dict = {}
    for name, rec in chaos["data"].items():
        if name.lower() not in species or not rec.get("Moves"):
            continue
        raw = float(rec["Raw count"])
        row = {CP._to_id(m): min(1.0, u / raw) for m, u in rec["Moves"].items() if CP._to_id(m) and u > 0}
        if row:
            out[name.lower()] = row
    return out


def test_skarmory_spikes_is_near_certain():
    """Skarmory runs Spikes on ~every set in gen3ou (Smogon's own moveset report: 99.7 % at 1500). The
    deflated table read 0.547."""
    assert priors.moves("skarmory")["spikes"] > 0.99
    assert priors.moves("metagross")["meteormash"] > 0.97
    assert priors.moves("blissey")["softboiled"] > 0.96


def test_every_species_move_mass_is_four_slots():
    """``Σ_m P(m in set) + empty-slot mass == 4`` for EVERY species, exactly (the chaos ``Moves`` count one
    weight per SLOT, an empty one under ``""``); and the bulk of species fill all four slots. The deflated
    table summed to 0.41-3.75 (median 1.47)."""
    chaos = {k.lower(): (k, v) for k, v in priors.smogon_stats_raw()["data"].items()}
    sums = []
    for sp, row in priors.move_raw().items():
        name, rec = chaos[sp]
        s = sum(row.values())
        sums.append(s)
        assert abs(s + CP.empty_slot_mass(name, rec) - 4.0) < 1e-9, sp
        assert all(0.0 < p <= 1.0 for p in row.values()), sp
    assert len(sums) > 200
    assert statistics.median(sums) > 3.99 and min(sums) > 3.7


def test_the_tool_reproduces_the_committed_move_prior():
    """The committed ``gen3_move_priors.json`` IS the tool's output on the committed stats (a hand edit, or a
    reverted tool, fails here)."""
    species = json.loads(repo_path("data", "pokemon", "gen3_species.json").read_text())
    derived = CP.compute_move_priors(_chaos(), species)
    committed = json.loads(repo_path("data", "pokemon", "gen3_move_priors.json").read_text())
    assert derived.keys() == committed.keys()
    for sp in derived:
        assert derived[sp].keys() == committed[sp].keys(), sp
        assert all(abs(derived[sp][m] - committed[sp][m]) < 1e-15 for m in derived[sp]), sp


def test_the_load_guard_raises_on_a_planted_deflated_table():
    deflated = _deflated_table(_chaos())
    assert abs(deflated["skarmory"]["spikes"] - 0.547) < 0.001          # the planted table IS the defect
    with pytest.raises(priors.PriorInvariantError, match="F-X5-41"):
        priors._checked_moves(deflated)
    with pytest.raises(CP.PriorInvariantError, match="F-X5-41"):
        CP.check_move_priors(deflated, _chaos())
    # the committed table passes both
    priors._checked_moves(priors.move_raw())
    CP.check_move_priors(priors.move_raw(), _chaos())


def test_the_load_guard_raises_on_a_single_deflated_species():
    """One species off by the SMALLEST deflation the old table carried (W / Raw count = 0.94) still throws."""
    table = copy.deepcopy(priors.move_raw())
    table["tyranitar"] = {m: p * 0.94 for m, p in table["tyranitar"].items()}
    with pytest.raises(priors.PriorInvariantError, match="tyranitar"):
        priors._checked_moves(table)


def test_weighted_count_refuses_an_inconsistent_record():
    """W is ONE total: a record whose Items (or Moves / 4) disagree with its Abilities is refused, never used."""
    chaos = _chaos()
    rec = chaos["data"]["Skarmory"]
    assert CP.weighted_count("Skarmory", rec) == pytest.approx(sum(rec["Abilities"].values()))
    bad = copy.deepcopy(rec)
    bad["Items"]["leftovers"] *= 1.01
    with pytest.raises(CP.PriorInvariantError, match="Items"):
        CP.weighted_count("Skarmory", bad)
    bad = copy.deepcopy(rec)
    bad["Moves"]["spikes"] *= 1.5
    with pytest.raises(CP.PriorInvariantError, match="Moves"):
        CP.weighted_count("Skarmory", bad)
    with pytest.raises(CP.PriorInvariantError):
        CP.weighted_count("Skarmory", {"Raw count": 10, "Moves": {"spikes": 1.0}})


def test_a_move_prior_above_one_throws_rather_than_clamping():
    chaos = {"data": {"Skarmory": copy.deepcopy(_chaos()["data"]["Skarmory"])}}
    rec = chaos["data"]["Skarmory"]
    w = sum(rec["Abilities"].values())
    # move one slot's weight from the empty key onto spikes beyond W (Σ Moves still 4 W, so W passes)
    extra = 1.2 * w - rec["Moves"]["spikes"]
    rec["Moves"]["spikes"] += extra
    rec["Moves"]["roar"] -= extra
    with pytest.raises(CP.PriorInvariantError, match="> 1"):
        CP.compute_move_priors(chaos, {"skarmory": {}})


def test_check_priors_raises_on_a_non_distribution():
    """The other five priors are self-normalised (their weighted numerator and denominator are the same
    field), and each must stay a distribution — checked by the tool before writing and by the facade at load."""
    item = {"skarmory": {"leftovers": 0.6, "nothing": 0.3}}
    with pytest.raises(CP.PriorInvariantError, match="item"):
        CP.check_priors(_chaos(), hp={}, ability={}, move={}, item=item, spread={}, teammate={})
    with pytest.raises(priors.PriorInvariantError, match="not a distribution"):
        priors._checked_distributions("x.json", item, dict.values)
    for loader in (priors.ability_raw, priors.hidden_power_raw, priors.item_raw, priors.spread_raw,
                   priors.teammate_raw):
        assert loader()                                      # the committed files pass their load guard
