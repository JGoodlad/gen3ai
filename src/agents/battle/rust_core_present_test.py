"""Every reading rule of the Rust core's ``present()`` pinned against POKE-ENV ITSELF.

``src/rust_sim/src/present/`` mirrors poke-env's state tracker as named rules (V1–V17, R1–R3; the
table in its ``mod.rs``). Its own ``cargo test`` pins say what the rule IS; this file says the rule
is poke-env's: each constructed scenario is fed, as ONE side's protocol text, both to a
``Gen3Battle`` (through :mod:`agents.battle.offline_feed`, the live player's dispatch) and to the
core (``core_events --present-stream``, the parse path), and the two ``LiveView`` + ``LegalActions``
are compared field by field, TYPE-strict, with no allowlist — slice V's comparison, on the shapes a
recorded corpus may not reach.

🚨 **The core reads the TRUTH, and parity with poke-env is not the goal.** Where poke-env is WRONG
about a sim fact (a registered finding, :mod:`agents.battle.poke_env_findings` — PE-V10, PE-R1b,
PE-V16), the pin asserts the core's TRUE value, asserts that poke-env reads differently, and asserts
the difference is exactly that finding (value-aware) — so the pin fails the day poke-env is fixed
(delete the finding) or the day the core drifts off the truth.
"""

from __future__ import annotations

import json
import subprocess
from typing import Dict, FrozenSet, List, Optional

import pytest

from agents.battle import rust_core_parity_views as V
from agents.battle.core_view import legal_actions_from_core, live_view_from_core
from agents.battle.live_view import LegalActions
from agents.battle.offline_feed import feed_line, new_battle

pytestmark = [pytest.mark.sim]

TEAM = ("Metagross|||clearbody|meteormash,earthquake,explosion,agility|Adamant|252,252,,,4,|||||]"
        "Suicune|||pressure|calmmind,surf,rest,icebeam|Bold|252,,252,,4,|||||")


def _req(active: str = "Metagross", cond_m: str = "301/301", cond_s: str = "341/341", *,
         moves: Optional[List[dict]] = None, extra: str = "", trapped: bool = False,
         force: bool = False) -> str:
    """Our side's `|request|` (the two-mon TEAM), in the bridge's own key order."""
    am, as_ = ("true", "false") if active == "Metagross" else ("false", "true")
    mv = moves if moves is not None else [
        {"move": "Meteor Mash", "id": "meteormash", "pp": 16, "maxpp": 16, "target": "normal",
         "disabled": False}]
    act = {"moves": mv}
    if trapped:
        act["trapped"] = True
    head = '{"forceSwitch":[true],' if force else '{"active":[' + json.dumps(act, separators=(",", ":")) + "],"
    return ("|request|" + head + '"side":{"name":"me","id":"p1","pokemon":['
            '{"ident":"p1: Metagross","details":"Metagross","condition":"' + cond_m + '","active":' + am
            + ',"stats":{"atk":405,"def":296,"spa":203,"spd":216,"spe":214},"moves":["meteormash",'
            '"earthquake","explosion","agility"],"baseAbility":"clearbody","item":"leftovers",'
            '"pokeball":"pokeball"},'
            '{"ident":"p1: Suicune","details":"Suicune","condition":"' + cond_s + '","active":' + as_
            + ',"stats":{"atk":139,"def":266,"spa":260,"spd":308,"spe":213},"moves":["calmmind","surf",'
            '"rest","icebeam"],"baseAbility":"pressure","item":"leftovers","pokeball":"pokeball"}]}'
            + extra + "}")


PREFIX = ["|player|p1|me||", "|player|p2|foe||", "|teamsize|p1|2", "|teamsize|p2|3", "|gen|3",
          "|tier|[Gen 3] OU", "|start"]
BASE = PREFIX + [_req(), "|switch|p1a: Metagross|Metagross|301/301",
                 "|switch|p2a: Zapdos|Zapdos|100/100", "|turn|1"]


def core(lines: List[str]) -> dict:
    from utils.bridge.sim_bridge_bin import resolve_core_events_bin

    head = json.dumps({"viewer": 0, "username": "me", "team": TEAM})
    p = subprocess.run([resolve_core_events_bin(), "--present-stream"], input="\n".join([head] + lines) + "\n",
                       capture_output=True, text=True, check=True)
    out = json.loads(p.stdout)
    assert out["ok"], out["error"]
    return out


def reading(lines: List[str]):
    b = new_battle("p1", {"p1": "me", "p2": "foe"}, packed_team=TEAM)
    for ln in lines:
        feed_line(b, ln)
    return b


def assert_same(lines: List[str], known: FrozenSet[str] = frozenset()) -> Dict[str, object]:
    """The core's view + legality == poke-env's, type-strict, every field — except where a
    registered poke-env finding explains the difference; exactly the findings in ``known`` must
    fire (a finding that stops firing is as loud as a new divergence)."""
    out = core(lines)
    b = reading(lines)
    live_c = live_view_from_core(out["view"], battle_tag=b.battle_tag)
    legal_c = legal_actions_from_core(out["legal"], out["request"])
    census = V.ViewCensus()
    legal_p = LegalActions.from_battle(b) if b.last_request else None
    V.compare_decision(b.live_view(), legal_p, live_c, legal_c, census, "pin", column="core")
    assert not census.divergences, census.render()
    assert set(census.known) == set(known), f"findings fired {dict(census.known)}, expected {sorted(known)}"
    assert census.fields > 20
    return out


def opp(out: dict, sp: str) -> dict:
    return next(m for m in out["view"]["opp"]["mons"] if m["species"] == sp)


def ours(out: dict, sp: str) -> dict:
    return next(m for m in out["view"]["ours"]["mons"] if m["species"] == sp)


# ---------------------------------------------------------------------------- the rules

def test_v1_v2_order_and_reveal():
    out = assert_same(BASE + ["|switch|p2a: Snorlax|Snorlax, M|100/100",
                              "|switch|p2a: Fishy|Gyarados, F|100/100"])
    assert [m["species"] for m in out["view"]["opp"]["mons"]] == ["zapdos", "snorlax", "gyarados"]


def test_v3_pressure_by_inference_and_a_called_move():
    out = assert_same(BASE + ["|move|p2a: Zapdos|Thunderbolt|p1a: Metagross",
                              "|move|p1a: Metagross|Meteor Mash|p2a: Zapdos",
                              "|move|p2a: Zapdos|Sleep Talk|p2a: Zapdos",
                              "|move|p2a: Zapdos|Drill Peck|p1a: Metagross|[from]move: Sleep Talk"])
    assert opp(out, "zapdos")["ability"] == "pressure"


def test_v4_the_volatile_lifecycle():
    assert_same(BASE + ["|-singleturn|p2a: Zapdos|Protect", "|-start|p2a: Zapdos|move: Taunt",
                        "|-start|p2a: Zapdos|confusion", "|turn|2", "|turn|3",
                        "|-end|p2a: Zapdos|confusion", "|-start|p1a: Metagross|Substitute"])


def test_v4_baton_pass():
    out = assert_same(BASE + ["|-boost|p2a: Zapdos|spa|2", "|-start|p2a: Zapdos|Substitute",
                              "|-start|p2a: Zapdos|move: Taunt",
                              "|switch|p2a: Celebi|Celebi|100/100|[from] Baton Pass"])
    assert opp(out, "celebi")["boosts"] == {"spa": 2}


def test_v5_the_status_counter_with_the_r1_fix():
    out = assert_same(BASE + ["|-status|p2a: Zapdos|tox", "|turn|2", "|turn|3",
                              "|-status|p2a: Zapdos|slp|[from] move: Rest", "|cant|p2a: Zapdos|slp",
                              "|-cureteam|p2a: Zapdos|[from] move: Aromatherapy"])
    assert opp(out, "zapdos")["status_counter"] == 1


def test_v6_the_protect_streak():
    assert_same(BASE + ["|move|p2a: Zapdos|Protect|p2a: Zapdos", "|-singleturn|p2a: Zapdos|Protect",
                        "|turn|2", "|move|p2a: Zapdos|Detect|p2a: Zapdos"])


def test_v7_item_disclosure():
    lines = BASE + ["|-heal|p2a: Zapdos|100/100|[from] item: Leftovers",
                    "|switch|p2a: Snorlax|Snorlax, M|100/100",
                    "|-enditem|p2a: Snorlax|Salac Berry|[eat]",
                    "|-heal|p2a: Snorlax|100/100|[from] item: Salac Berry"]
    out = assert_same(lines)
    s = opp(out, "snorlax")
    assert (s["item"], s["consumed_item"]) == (None, "salacberry"), "no berry from a heal line"
    out = assert_same(lines + ["|-damage|p1a: Metagross|280/301|[from] item: Rocky Helmet|[of] p2a: Snorlax"])
    s = opp(out, "snorlax")
    assert (s["item"], s["consumed_item"]) == ("rockyhelmet", None), "a truthy item clears the consumed"


def test_v8_ability_slots_and_disclosures():
    assert_same(BASE + ["|switch|p2a: Porygon2|Porygon2|100/100",
                        "|-ability|p2a: Porygon2|Clear Body|[from] ability: Trace|[of] p1a: Metagross",
                        "|switch|p2a: Snorlax|Snorlax, M|100/100",
                        "|-immune|p2a: Snorlax|[from] ability: Immunity",
                        "|-activate|p2a: Snorlax|ability: Immunity"])


def test_v9_the_opponent_is_hidden():
    assert_same(BASE + ["|-damage|p2a: Zapdos|54/100", "|-damage|p1a: Metagross|200/301"])


def test_pe_v10_a_fainted_mon_holds_no_stages_where_poke_env_keeps_them():
    lines = BASE + ["|-boost|p2a: Zapdos|spa|1", "|faint|p2a: Zapdos"]
    out = assert_same(lines, known=frozenset({"PE-V10"}))
    assert opp(out, "zapdos")["boosts"] == {}, "the truth: the faint cleared them"
    assert dict(reading(lines).live_view().opp.active.boosts) == {"spa": 1}, "poke-env keeps them"
    out = assert_same(lines + ["|switch|p2a: Snorlax|Snorlax, M|100/100"])
    assert opp(out, "zapdos")["boosts"] == {}


def test_v11_screens_and_spikes():
    assert_same(BASE + ["|turn|4", "|-sidestart|p2: foe|Reflect", "|-sidestart|p1: me|Spikes",
                        "|-sidestart|p1: me|Spikes", "|-sidestart|p2: foe|move: Light Screen",
                        "|-sideend|p2: foe|Reflect"])


def test_v12_weather():
    assert_same(BASE + ["|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Zapdos", "|turn|2",
                        "|-weather|Sandstorm|[upkeep]", "|turn|3", "|-weather|RainDance", "|turn|4"])


def test_v13_legality_trapped_forced_and_struggle():
    assert_same(BASE + [_req(trapped=True)])
    assert_same(BASE + ["|faint|p1a: Metagross", _req(cond_m="0 fnt", force=True)])
    assert_same(BASE + [_req(moves=[{"move": "Struggle", "id": "struggle", "target": "randomNormal",
                                      "disabled": False}])])


def test_v14_transform_overlay():
    assert_same(BASE + ["|switch|p2a: Ditto|Ditto|100/100", "|move|p2a: Ditto|Transform|p1a: Metagross",
                        "|-transform|p2a: Ditto|p1a: Metagross"])


def test_v15_a_benched_mons_pp_is_the_sighting_count():
    """Only the ACTIVE mon is re-synced from the request (R3): our move's PP after a switch-out."""
    moves = [{"move": "Meteor Mash", "id": "meteormash", "pp": 14, "maxpp": 16, "target": "normal",
              "disabled": False}]
    surf = [{"move": "Surf", "id": "surf", "pp": 24, "maxpp": 24, "target": "allAdjacent",
             "disabled": False}]
    out = assert_same(BASE + ["|move|p1a: Metagross|Meteor Mash|p2a: Zapdos", _req(moves=moves),
                              "|switch|p1a: Suicune|Suicune|341/341", _req(active="Suicune", moves=surf)])
    mm = next(m for m in ours(out, "metagross")["moves"] if m["id"] == "meteormash")
    assert mm["current_pp"] == 14, "synced from the request while active, then kept"


def test_pe_v16_flash_fire_survives_its_holders_fire_move_where_poke_env_ends_it():
    lines = BASE + ["|switch|p2a: Houndoom|Houndoom, M|100/100",
                    "|-start|p2a: Houndoom|ability: Flash Fire",
                    "|move|p2a: Houndoom|Flamethrower|p1a: Metagross"]
    out = assert_same(lines, known=frozenset({"PE-V16"}))
    assert "flashfire" in opp(out, "houndoom")["volatiles"], "the truth: it lasts until switch-out"
    assert "flashfire" not in dict(reading(lines).live_view().opp.active.volatiles), "poke-env ends it"
    lines += ["|switch|p2a: Zapdos|Zapdos|100/100"]
    out = assert_same(lines)
    assert "flashfire" not in opp(out, "houndoom")["volatiles"], "cleared on switch-out"


def test_v17_a_request_that_contradicts_the_active_flag_resyncs_it():
    assert_same(BASE + ["|switch|p1a: Suicune|Suicune|341/341", _req(active="Metagross")])


def test_r2_psych_up_copies_onto_its_user():
    assert_same(BASE + ["|-boost|p1a: Metagross|atk|2", "|-copyboost|p2a: Zapdos|p1a: Metagross|[from] move: Psych Up"])


def test_r3_our_active_pp_is_the_requests():
    moves = [{"move": "Meteor Mash", "id": "meteormash", "pp": 13, "maxpp": 16, "target": "normal",
              "disabled": False}]
    out = assert_same(BASE + ["|move|p1a: Metagross|Meteor Mash|p2a: Zapdos", _req(moves=moves)])
    assert next(m for m in ours(out, "metagross")["moves"] if m["id"] == "meteormash")["current_pp"] == 13


def test_mimic_leppa_trick_conversion_and_forecast():
    assert_same(BASE + ["|move|p2a: Zapdos|Mimic|p1a: Metagross", "|-start|p2a: Zapdos|Mimic|Earthquake",
                        "|-activate|p2a: Zapdos|move: Trick|[of] p1a: Metagross",
                        "|switch|p2a: Porygon2|Porygon2|100/100",
                        "|-start|p2a: Porygon2|typechange|Water|[from] move: Conversion",
                        "|switch|p2a: Castform|Castform|100/100",
                        "|-formechange|p2a: Castform|Castform-Sunny|[msg]"])


def test_pe_r1b_the_toxic_counter_is_the_stage_where_poke_env_counts_turns():
    mid = BASE + ["|-status|p2a: Zapdos|tox", "|-damage|p2a: Zapdos|94/100 tox|[from] psn", "|turn|2",
                  "|-damage|p2a: Zapdos|82/100 tox|[from] psn"]
    # Between the residual and the next |turn| (an end-of-turn forced replacement's decision) the
    # sim is already at stage 2; poke-env, ticking at |turn|, still reads 1.
    out = assert_same(mid, known=frozenset({"PE-R1b"}))
    assert opp(out, "zapdos")["status_counter"] == 2
    lines = mid + ["|turn|3"]
    out = assert_same(lines)
    assert opp(out, "zapdos")["status_counter"] == 2, "two residual chips: stage 2, as poke-env reads"
    lines += ["|switch|p2a: Snorlax|Snorlax, M|100/100", "|switch|p2a: Zapdos|Zapdos|82/100 tox",
              "|turn|4"]
    out = assert_same(lines, known=frozenset({"PE-R1b"}))
    assert opp(out, "zapdos")["status_counter"] == 0, "the truth: no residual since re-entry"
    assert reading(lines).live_view().opp.active.status_counter == 1, "poke-env ticked at |turn|"
    # poke-env FREEZES that count at a faint; the finding follows it there (the slot no longer reads it).
    assert_same(lines + ["|faint|p2a: Zapdos"], known=frozenset({"PE-R1b"}))
