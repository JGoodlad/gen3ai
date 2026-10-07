"""PRIOR = 0 for every entity the gen3ou format spec bans (`gen3_format_spec_priors_v1`, design_format_spec.md
§5.1). Each test FAILS on revert: the committed Smogon priors carry no banned mass and ARE the tool's output (a
reverted tool reproduces the 1 % Sand Veil floor and the Quick Claw mass), the load guard THROWS on planted
banned mass, the move prior keeps its slot identity, and every model table gives a banned entity the ILLEGAL
value instead of a liftable floor."""
from __future__ import annotations

import copy
import json
import math

import pytest
import torch

from agents import gen3_data
from agents.gen3_data import format_spec as fs
from agents.gen3_data import priors
from tools.smogon_stats_downloader import compute_priors as CP
from utils.paths import repo_path

SPEC = fs.GEN3OU


def _committed(name: str) -> dict:
    return json.loads(repo_path("data", "pokemon", name).read_text())


def _inputs():
    chaos = copy.deepcopy(priors.smogon_stats_raw())
    species = _committed("gen3_species.json")
    pokedex = json.loads(repo_path("src", "poke_env", "data", "static", "pokedex", "gen3pokedex.json").read_text())
    return chaos, species, pokedex, set(_committed("gen3_abilities.json"))


def test_no_committed_prior_gives_a_banned_entity_mass():
    for sp, row in priors.ability_raw().items():
        assert not set(row) & SPEC.banned_abilities, sp
    for sp, row in priors.item_raw().items():
        assert not set(row) & SPEC.banned_items, sp
    for sp, row in priors.move_raw().items():
        assert not [m for m in row if SPEC.is_banned("move", m, species=sp)], sp
    for sp, row in priors.teammate_raw().items():
        assert sp not in SPEC.banned_species and not set(row) & SPEC.banned_species, sp
    assert not set(priors.species_usage()) & SPEC.banned_species
    for sid, _ in fs.ABILITY_LOCKED_SPECIES:
        assert priors.ability(sid) == {}, sid


def test_the_former_floor_cases_now_read_their_one_legal_ability():
    assert priors.ability("dugtrio") == {"arenatrap": 1.0}
    assert priors.ability("gligar") == {"hypercutter": 1.0}
    assert priors.ability("electrode") == {"static": 1.0}
    assert priors.ability("voltorb") == {"static": 1.0}
    assert abs(sum(priors.items("hypno").values()) - 1.0) < 1e-9 and "quickclaw" not in priors.items("hypno")


def test_the_tool_reproduces_the_committed_ability_and_item_priors():
    chaos, species, pokedex, abilities = _inputs()
    ab, summary = CP.compute_ability_priors(chaos, species, pokedex, abilities)
    assert ab == _committed("gen3_ability_priors.json")
    assert sorted(summary["format_locked"]) == sorted(s for s, _ in fs.ABILITY_LOCKED_SPECIES)
    it = CP.compute_item_priors(chaos, species)
    committed = _committed("gen3_item_priors.json")
    assert it.keys() == committed.keys()
    for sp in it:
        assert it[sp].keys() == committed[sp].keys(), sp
        assert all(abs(it[sp][i] - committed[sp][i]) < 1e-15 for i in it[sp]), sp


@pytest.mark.parametrize("fname,kind,table", [
    ("gen3_ability_priors.json", "ability", {"dugtrio": {"arenatrap": 0.99, "sandveil": 0.01}}),
    ("gen3_item_priors.json", "item", {"hypno": {"leftovers": 0.5, "quickclaw": 0.5}}),
    ("gen3_move_priors.json", "move", {"swampert": {"earthquake": 0.9, "swagger": 0.01}}),
    ("gen3_move_priors.json", "move", {"smeargle": {"spore": 0.9, "ingrain": 0.1}}),
    ("gen3_teammate_priors.json", "teammate", {"tyranitar": {"mewtwo": 1.0}}),
    ("gen3_teammate_priors.json", "teammate", {"mewtwo": {"tyranitar": 1.0}}),
])
def test_the_load_guard_throws_on_planted_banned_mass(fname, kind, table):
    with pytest.raises(priors.PriorInvariantError, match="BANNED"):
        priors._checked_format_legal(fname, kind, table)
    kw = {"ability": {}, "item": {}, "move": {}, "teammate": {}}
    kw[kind] = table
    with pytest.raises(CP.PriorInvariantError, match="banned"):
        CP.check_format_legal(**kw)


def test_the_load_guard_passes_a_legal_species_move():
    priors._checked_format_legal("gen3_move_priors.json", "move", {"roselia": {"ingrain": 0.3}})


def test_species_usage_refuses_a_banned_species():
    table = dict(priors.species_usage())
    table["mewtwo"] = 1.0
    with pytest.raises(priors.PriorInvariantError, match="banned"):
        priors._checked_species_usage(table)


def _move_mass(rec: dict, frm: str, to: str, frac: float) -> None:
    """Move ``frac`` of a set's weight from move ``frm`` to ``to`` (Σ Moves stays 4 W, so W is unchanged)."""
    w = sum(rec["Abilities"].values())
    key = next(k for k in rec["Moves"] if CP._to_id(k) == frm)
    rec["Moves"][key] -= frac * w
    rec["Moves"][to] = rec["Moves"].get(to, 0.0) + frac * w


def test_the_tool_drops_banned_move_mass_and_keeps_the_slot_identity():
    chaos, species, _, _ = _inputs()
    rec = chaos["data"]["Swampert"]
    _move_mass(rec, "earthquake", "swagger", 0.02)
    out = CP.compute_move_priors(chaos, species)["swampert"]
    assert "swagger" not in out
    e = CP.empty_slot_mass("Swampert", rec)
    assert abs(sum(out.values()) + e - 4.0) < 1e-9
    CP.check_move_priors({"swampert": out}, chaos)
    # a banned mass so large a legal move (Skarmory's Spikes, ~0.997) would exceed P = 1: THROWS, never clamps
    chaos2, _, _, _ = _inputs()
    rec2 = chaos2["data"]["Skarmory"]
    second = sorted(((CP._to_id(k), v) for k, v in rec2["Moves"].items() if CP._to_id(k) != "spikes"),
                    key=lambda kv: -kv[1])[0][0]
    _move_mass(rec2, second, "swagger", 0.5)
    with pytest.raises(CP.PriorInvariantError, match="banned-move mass"):
        CP.compute_move_priors({"data": {"Skarmory": rec2}}, species)


# ------------------------------------------------------------------------------- the model tables
def test_the_move_prior_gives_a_banned_move_the_illegal_value():
    from agents.model.belief_tables import _ILLEGAL_PROB, _PRIOR_FLOOR, build_move_prior_logits
    lg = build_move_prior_logits(400, 380)
    p = torch.sigmoid(lg.double())
    for mid in ("swagger", "doubleteam", "minimize", "fissure", "sheercold", "assist"):
        assert p[:, gen3_data.moves.get(mid).num].max().item() == pytest.approx(_ILLEGAL_PROB, rel=1e-3), mid
    ing = gen3_data.moves.get("ingrain").num
    assert p[gen3_data.species.get("smeargle").num, ing].item() == pytest.approx(_ILLEGAL_PROB, rel=1e-3)
    assert p[gen3_data.species.get("roselia").num, ing].item() == pytest.approx(_PRIOR_FLOOR, rel=1e-3)


def test_the_item_prior_gives_a_banned_item_zero():
    from agents.model.belief_tables import build_item_prior
    prior = build_item_prior(400, 400)
    for iid in SPEC.banned_items:
        assert prior[:, gen3_data.items.get(iid).num].max().item() == 0.0, iid
    assert torch.allclose(prior.sum(1), torch.ones(400), atol=1e-5)


def test_the_species_priors_give_a_banned_species_the_illegal_value():
    from agents.model.belief_tables import SPECIES_CLAUSE_LOGIT, build_species_cooccur_prior
    from agents.model.dex_ids import build_species_usage_prior
    usage = build_species_usage_prior(400)
    log_marginal, _ = build_species_cooccur_prior(400)
    for sid in ("mewtwo", "wobbuffet", "deoxys", "mrmime", "sandslash"):
        n = gen3_data.species.get(sid).num
        assert usage[n].item() == 0.0, sid
        assert log_marginal[n].item() == pytest.approx(SPECIES_CLAUSE_LOGIT), sid
    assert log_marginal[gen3_data.species.get("tyranitar").num].item() > math.log(0.05)


def test_fixed_mass_hypothesis_moves_exclude_banned_moves():
    """X5 fixed_mass (production after the version break) builds each hypothesis mon's move set from the move
    prior's legality (`hypothesis_set.build_move_legality`): a format-banned move is never a candidate."""
    from agents.model.hypothesis_set import build_move_legality
    legal, valid = build_move_legality(400, 380)
    for mid in ("swagger", "doubleteam", "fissure", "assist"):
        n = gen3_data.moves.get(mid).num
        assert not bool(legal[:, n].any()) and not bool(valid[n]), mid
    ing = gen3_data.moves.get("ingrain").num
    assert not bool(legal[gen3_data.species.get("smeargle").num, ing])
    assert bool(legal[gen3_data.species.get("roselia").num, ing]) and bool(valid[ing])
