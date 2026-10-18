"""The team-set overlap check (``gen3_team_overlap_v1``) — what is matched, what is counted, and that EVERY read
writes and prints it.

The finding this exists for (2026-10-18): the ``away`` set was described as unseen by us, and 19 of its 20 teams
are exact copies of pool teams. Each test below fails if its rule is reverted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from main.anchors import team_overlap as ov
from main.anchors.results import render, summarize
from utils.gen3_utils import GEN3_HP_IVS

MON = """{head} @ Leftovers
Ability: {ability}
EVs: {evs}
Adamant Nature
{ivs}- {m1}
- {m2}
- Protect
- {hp}
"""
SPECIES = ("Snorlax", "Skarmory", "Metagross", "Salamence", "Tyranitar", "Swampert", "Gengar", "Starmie")


def mon(species: str, *, m1: str = "Earthquake", m2: str = "Rock Slide", hp: str = "Hidden Power [Fire]",
        evs: str = "252 HP / 252 Atk / 4 Def", ivs: str = "", ability: str = "Thick Fat", head: str = "") -> str:
    return MON.format(head=head or species, ability=ability, evs=evs, ivs=ivs, m1=m1, m2=m2, hp=hp)


def team(*species: str, **kw) -> str:
    return "\n".join(mon(s, **kw) for s in species)


# ----------------------------------------------------------------------------------------- what is matched
def test_the_hidden_power_formula_gives_every_spread_its_own_type() -> None:
    for typ, ivs in GEN3_HP_IVS.items():
        assert ov.hidden_power_type(ivs) == typ.lower()
    assert ov.hidden_power_type({}) == "dark"          # all 31


@pytest.mark.parametrize("spelling, ivs, expect", [
    ("Hidden Power [Fire]", "", "hiddenpowerfire"),
    ("Hidden Power Fire", "", "hiddenpowerfire"),
    ("hiddenpowerfire", "", "hiddenpowerfire"),
    # TYPELESS: the type the IVs give — the simulator's type, not a placeholder.
    ("Hidden Power", "IVs: 31 HP / 30 Atk / 31 Def / 30 SpA / 31 SpD / 30 Spe\n", "hiddenpowerfire"),
    ("Hidden Power", "", "hiddenpowerdark"),
])
def test_hidden_power_is_matched_by_its_TYPE(spelling: str, ivs: str, expect: str) -> None:
    m = ov.parse_mon(mon("Snorlax", hp=spelling, ivs=ivs))
    assert expect in m.moves and not {"hiddenpower"} & set(m.moves)


def test_two_hidden_power_types_are_two_different_mon_sets() -> None:
    a = ov.parse_mon(mon("Snorlax", hp="Hidden Power [Fire]"))
    b = ov.parse_mon(mon("Snorlax", hp="Hidden Power [Grass]"))
    assert a.key != b.key


def test_a_nickname_and_a_gender_tag_resolve_to_the_species() -> None:
    plain = ov.parse_mon(mon("Skarmory"))
    for head in ("Airmure (Skarmory)", "Skarmory (M)", "Airmure (Skarmory) (F)"):
        assert ov.parse_mon(mon("Skarmory", head=head)).key == plain.key


def test_a_cosmetic_forme_matches_its_base_and_a_stat_forme_does_not() -> None:
    """Formes share their base's num, so nothing is keyed by num; a forme with its base's stats and types (the
    Unown letters) IS the base mechanically, a forme with its own stats (Deoxys-Speed) is not."""
    assert ov.parse_mon(mon("Unown-B")).species == "unown"
    assert ov.parse_mon(mon("Deoxys-Speed")).species == "deoxysspeed"
    assert ov.parse_mon(mon("Deoxys")).species == "deoxys"


def test_trailing_blank_lines_add_no_mon() -> None:
    t = ov.parse_team(team(*SPECIES[:6]) + "\n\n\n")
    assert len(t.mons) == 6


def test_the_strict_tier_adds_item_ability_nature_and_evs() -> None:
    a = ov.parse_mon(mon("Snorlax"))
    b = ov.parse_mon(mon("Snorlax", evs="252 HP / 4 Atk / 252 Def"))
    assert a.key == b.key and a.strict_key != b.strict_key
    assert a.evs == (252, 252, 4, 0, 0, 0)


# ----------------------------------------------------------------------------------------- what is counted
def test_the_counts_against_a_reference() -> None:
    ref = ov.Reference(ov.parse_items([("r1", team(*SPECIES[:6])), ("r2", team(*SPECIES[2:8]))]))
    read = ov.parse_items([
        ("exact", team(*SPECIES[:6])),                                                 # 6/6 of r1
        ("evs_differ", team(*SPECIES[:6], evs="252 HP / 4 Atk / 252 Def")),            # exact, not strict
        ("same_species", team(*SPECIES[:6], m1="Body Slam")),                          # 0 mon sets, same 6 species
        ("four", team(*SPECIES[:4]) + "\n" + team("Jirachi", "Zapdos", m1="Body Slam")),  # 4 of r1 / r2
        ("none", team("Jirachi", "Zapdos", "Celebi", "Suicune", "Heracross", "Blissey")),
    ])
    o = ov.overlap(read, ref)
    assert o["exact_team"] == 2 and o["exact_team_strict"] == 1
    assert o["same_species6"] == 3
    assert o["max_shared_mons_hist"] == {"0": 2, "1": 0, "2": 0, "3": 0, "4": 1, "5": 0, "6": 2}
    assert o["teams_sharing_ge4"] == 3
    assert o["mons_seen"] == 6 + 6 + 0 + 4 + 0
    assert o["mons_seen_strict"] == 6 + 0 + 0 + 4 + 0
    assert o["exact_examples"][0] == ["exact", "r1"]


# ------------------------------------------------------------------------- every read writes and prints it
def test_the_summary_carries_the_overlap_and_render_prints_it() -> None:
    from main.anchors.results_test import _cell

    block = {"against": {"training_pool": {"ours": ov.overlap(
        ov.parse_items([("a", team(*SPECIES[:6]))]),
        ov.Reference(ov.parse_items([("p", team(*SPECIES[:6]))])))}}, "missing_references": ["x=/nope"]}
    s = summarize(_cell(), [], status="OK", team_overlap=block)
    assert s["team_overlap"] is block
    text = render(s)
    assert "ours vs training_pool (1): exact 1/1" in text and "reference not on this box: x=/nope" in text
    assert "not computed" in render(summarize(_cell(), [], status="OK"))


def test_EVERY_read_computes_writes_and_prints_its_overlap(tmp_path, monkeypatch, capsys) -> None:
    """End to end through ``main.anchors.cli.main`` with the games stubbed: the overlap is computed before any
    game, against OUR training pool and the opponent's configured team sets, and lands in summary.json."""
    from main.anchors import cli as cli_mod
    from main.anchors import runner as runner_mod
    from main.anchors.session import pool_team_texts
    import main.live.halt as halt_mod

    teams = tmp_path / "cache" / "teams" / "theirset" / "gen3ou"
    teams.mkdir(parents=True)
    (teams / "a.gen3ou_team").write_text(pool_team_texts()[0] + "\n")       # an exact pool copy
    (teams / "b.gen3ou_team").write_text(team("Jirachi", "Zapdos", "Celebi", "Suicune", "Heracross", "Blissey"))
    exe = tmp_path / "python"
    exe.write_text("")
    conf = {
        "showdown": {"node": "node", "port_range": [9500, 9599]},
        "opponents": {"metamon": {
            "checkout": str(tmp_path), "python": str(exe), "cache_dir": str(tmp_path / "cache"),
            "agents": {"SmallRL": {"checkpoint": 40, "params": 1}},
            "team_sets": {"home": "theirset", "away": "theirset"},
            "reference_team_dirs": {"theirset": str(teams), "gone": str(tmp_path / "nope")}},
            "foulplay": {"checkout": str(tmp_path), "python": str(exe), "team_dirs": {}}},
        "our_team_sources": {"home": {"kind": "dir", "path": str(teams)}},
    }
    cfg_path = tmp_path / "anchors.json"
    cfg_path.write_text(json.dumps(conf))
    monkeypatch.setenv("GEN3AI_ANCHORS_CONFIG", str(cfg_path))
    for var in ("GEN3AI_METAMON_DIR", "GEN3AI_METAMON_PYTHON", "GEN3AI_METAMON_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(halt_mod, "refuse_if_halted", lambda *_a, **_k: None)
    started = []

    async def no_games(plan, cfg):
        started.append(plan)
        return [], {}, None

    monkeypatch.setattr(runner_mod, "run_series", no_games)
    out = tmp_path / "out"
    rc = cli_mod.main(["--our-side", "bot:staller", "--opponent", "metamon:SmallRL", "--teamset", "home",
                       "--games", "2", "--server-uri", "ws://127.0.0.1:9543/showdown/websocket",
                       "--out", str(out)])
    assert rc == 2 and started                   # no games -> short_series, but the summary is written
    s = json.loads((out / "summary.json").read_text())
    block = s["team_overlap"]
    assert block["version"] == ov.OVERLAP_VERSION and block["theirs_same_as_ours"] is True
    pool = block["against"]["training_pool"]["ours"]
    assert (pool["n_read_teams"], pool["exact_team"]) == (2, 1)
    assert block["against"]["metamon:theirset"]["ours"]["exact_team"] == 2
    assert block["missing_references"] == [f"metamon:gone={tmp_path / 'nope'}"]
    printed = capsys.readouterr().out
    assert "ours vs training_pool (" in printed and "exact 1/2" in printed
    assert os.path.isfile(out / "games.jsonl")


def test_the_committed_config_names_metamon_s_known_team_sets() -> None:
    from main.anchors.config import load_config
    from utils.paths import repo_path

    cfg = load_config(Path(repo_path("designs", "ops", "anchors.json")))
    assert {"competitive", "hl_05_26"} <= set(cfg.metamon.reference_team_dirs)
