"""Every reading rule of the Rust core's ``present()`` on CONSTRUCTED scenarios, held to a frozen GOLDEN (P6 of the
poke-env retirement, 2026-10-08).

Until P6 this file fed each scenario, as ONE side's protocol text, both to the core (``core_events --present-stream``)
and to poke-env's ``Gen3Battle``, and compared the two ``LiveView`` + ``LegalActions`` field by field (slice V's
comparison on shapes a recorded corpus may not reach); the scenarios V17-V19, R2, R3 and the called-move / transform
/ Trick / Forecast shapes have no ``cargo test`` twin. With poke-env retired the comparison is FROZEN: every scenario's
whole core answer (the view, the legality, the raw request) is recorded in ``core_present_golden.json``, keyed by the
scenario's text, AT THE COMMIT WHERE BOTH READINGS STILL AGREED (the deleted test passed on it, with no poke-env
finding firing; the core binary is unchanged since). A reading change is therefore a reviewed golden re-record
(``GEN3AI_RECORD_PRESENT_GOLDEN=1``), never silent; the per-scenario asserts below (what the rule IS) stay as written.
The refusal tests keep the CLASS poke-env raised as the expected class of the core's refusal.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import FrozenSet, List, Optional

import pytest

pytestmark = [pytest.mark.sim]

#: Ships beside this test (a test locating its own data file, not repo-root discovery).
GOLDEN = Path(__file__).with_name("core_present_golden.json")
_RECORD = os.environ.get("GEN3AI_RECORD_PRESENT_GOLDEN") == "1"
_recorded: dict = {}

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


def _key(lines: List[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _golden() -> dict:
    return json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


def assert_same(lines: List[str], known: FrozenSet[str] = frozenset()) -> dict:
    """The core's whole answer for ``lines`` == the frozen golden (recorded where the core and poke-env agreed).
    ``known`` (the poke-env findings a scenario expected) is kept for the record: the registry was EMPTY."""
    assert not known, "no poke-env finding is registered any more"
    out = core(lines)
    canon = json.loads(json.dumps(out, sort_keys=True))
    k = _key(lines)
    if _RECORD:
        _recorded[k] = canon
        g = _golden()
        g[k] = canon
        GOLDEN.write_text(json.dumps(g, sort_keys=True, indent=1) + "\n")
        return out
    g = _golden()
    assert k in g, "an UNRECORDED scenario — record it deliberately: GEN3AI_RECORD_PRESENT_GOLDEN=1"
    assert canon == g[k], "the core's reading of this scenario moved from the golden"
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


def test_pe_v10_a_fainted_mon_holds_no_stages():
    """Fixed in the fork (`gen3_pe_reading_fixes_v1`): both readings hold none at the faint."""
    lines = BASE + ["|-boost|p2a: Zapdos|spa|1", "|faint|p2a: Zapdos"]
    out = assert_same(lines)
    assert opp(out, "zapdos")["boosts"] == {}, "the truth: the faint cleared them"
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


def test_v18_transform_overlay():
    assert_same(BASE + ["|switch|p2a: Ditto|Ditto|100/100", "|move|p2a: Ditto|Transform|p1a: Metagross",
                        "|-transform|p2a: Ditto|p1a: Metagross"])


def test_v19_a_benched_mons_pp_is_the_sighting_count():
    """Only the ACTIVE mon is re-synced from the request (R3): our move's PP after a switch-out."""
    moves = [{"move": "Meteor Mash", "id": "meteormash", "pp": 14, "maxpp": 16, "target": "normal",
              "disabled": False}]
    surf = [{"move": "Surf", "id": "surf", "pp": 24, "maxpp": 24, "target": "allAdjacent",
             "disabled": False}]
    out = assert_same(BASE + ["|move|p1a: Metagross|Meteor Mash|p2a: Zapdos", _req(moves=moves),
                              "|switch|p1a: Suicune|Suicune|341/341", _req(active="Suicune", moves=surf)])
    mm = next(m for m in ours(out, "metagross")["moves"] if m["id"] == "meteormash")
    assert mm["current_pp"] == 14, "synced from the request while active, then kept"


def test_pe_v16_flash_fire_survives_its_holders_fire_move():
    """Fixed in the fork (`gen3_pe_reading_fixes_v1`): both readings keep it to the switch-out."""
    lines = BASE + ["|switch|p2a: Houndoom|Houndoom, M|100/100",
                    "|-start|p2a: Houndoom|ability: Flash Fire",
                    "|move|p2a: Houndoom|Flamethrower|p1a: Metagross"]
    out = assert_same(lines)
    assert "flashfire" in opp(out, "houndoom")["volatiles"], "the truth: it lasts until switch-out"
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


def test_pe_r1b_the_toxic_counter_is_the_stage():
    """Fixed in the fork (`gen3_pe_reading_fixes_v1`): both readings count residual chips since
    the switch-in, at the three shapes where upstream's per-`|turn|` tick disagreed."""
    mid = BASE + ["|-status|p2a: Zapdos|tox", "|-damage|p2a: Zapdos|94/100 tox|[from] psn", "|turn|2",
                  "|-damage|p2a: Zapdos|82/100 tox|[from] psn"]
    # Between the residual and the next |turn| (an end-of-turn forced replacement's decision) the
    # sim is already at stage 2 (upstream poke-env read 1).
    out = assert_same(mid)
    assert opp(out, "zapdos")["status_counter"] == 2
    lines = mid + ["|turn|3"]
    out = assert_same(lines)
    assert opp(out, "zapdos")["status_counter"] == 2, "two residual chips: stage 2, as poke-env reads"
    lines += ["|switch|p2a: Snorlax|Snorlax, M|100/100", "|switch|p2a: Zapdos|Zapdos|82/100 tox",
              "|turn|4"]
    out = assert_same(lines)
    assert opp(out, "zapdos")["status_counter"] == 0, "the truth: no residual since re-entry"
    # A faint keeps the count it had (upstream froze its one-ahead count there).
    assert_same(lines + ["|faint|p2a: Zapdos"])


# ---------------------------------------------------------------------------
# REFUSALS — `gen3_core_error_v1`: where poke-env raises, the core refuses with the SAME class
# ---------------------------------------------------------------------------

def core_raw(lines: List[str]) -> dict:
    from utils.bridge.sim_bridge_bin import resolve_core_events_bin

    head = json.dumps({"viewer": 0, "username": "me", "team": TEAM})
    p = subprocess.run([resolve_core_events_bin(), "--present-stream"], input="\n".join([head] + lines) + "\n",
                       capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


#: One line poke-env REFUSES, appended to a valid opening, and the class it raises.
REFUSALS = {
    "unknown_keyword": ("|-dynamaxplus|p2a: Zapdos", "UnknownMessageType"),
    "non_gen3_keyword": ("|-mega|p2a: Zapdos|Zapdos|Zapdosite", "UnsupportedMessageType"),
    "gen_mismatch": ("|gen|4", "RuntimeError"),
    "turn_not_an_int": ("|turn|two", "ValueError"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_refusals_raise_the_same_class(case):
    """The core's REFUSAL class == the exception class poke-env raises on the same line — the
    parity gate compares the CLASS (the message is each side's own). A refusal in the core that
    poke-env accepts, or the reverse, fails here."""
    line, want = REFUSALS[case]
    lines = BASE + [line]
    out = core_raw(lines)
    assert not out["ok"], "the core accepted a line poke-env refuses"
    err = out["core_error"]
    assert (err["kind"], err["class"]) == ("refusal", want), err


def test_malformed_input_is_not_reported_as_a_refusal():
    """An undecodable `|request|` is MALFORMED — the core never dresses it up as a poke-env class."""
    out = core_raw(BASE + ['|request|{"side":'])
    assert not out["ok"]
    assert out["core_error"]["kind"] == "malformed" and out["core_error"]["class"] is None, out


@pytest.mark.parametrize("called", [
    "|move|p2a: Clefable|Thunderbolt|p1a: Metagross|[from] Metronome",
    "|move|p2a: Clefable|Swords Dance|p2a: Clefable|[from] Metronome",
    "|move|p2a: Clefable|Slack Off||[from] Metronome|[still]",
    "|move|p2a: Clefable|Super Fang|p1a: Metagross|[from] Metronome|[miss]|[miss]",
    "|move|p2a: Clefable|Nature Power|p2a: Clefable|[from] Metronome",
])
def test_called_move_class_metronome(called):
    """`gen3_called_move_reading_v1`: a Metronome-called move (gen3's BARE `[from] Metronome`,
    every tail shape the sim emits) — the core and poke-env agree the called move is neither
    revealed nor charged (it crashed poke-env's parse before the fix)."""
    out = assert_same(BASE + ["|switch|p2a: Clefable|Clefable, F|100/100",
                              "|move|p2a: Clefable|Metronome|p2a: Clefable", called])
    assert [(m["id"], m["current_pp"]) for m in opp(out, "clefable")["moves"]] == [("metronome", 15)]


@pytest.mark.parametrize("caller,called", [
    ("Assist", "|move|p2a: Clefable|Explosion|p1a: Metagross|[from] Assist"),
    ("Assist", "|move|p2a: Clefable|Roar||[from] Assist|[still]"),
    ("Nature Power", "|move|p2a: Clefable|Swift|p1a: Metagross|[from] Nature Power|[miss]"),
])
def test_called_move_class_assist_and_nature_power(caller, called):
    out = assert_same(BASE + ["|switch|p2a: Clefable|Clefable, F|100/100",
                              f"|move|p2a: Clefable|{caller}|p2a: Clefable", called])
    assert [m["id"] for m in opp(out, "clefable")["moves"]] == [caller.lower().replace(" ", "")]


# ------------------------------------------------- gen3_obs_facts_v1: V18 public reveal, V19 residual
def test_v18_what_the_opponent_has_seen_of_our_side():
    """The PUBLIC flags (`LiveMove.seen`, `item_public`, `ability_public`) are the reading's own
    reveal rules pointed at our side: a `|request|` reveals nothing, a public line reveals."""
    out = assert_same(BASE)
    m = ours(out, "metagross")
    assert not any(mv["seen"] for mv in m["moves"]) and not m["item_public"] and not m["ability_public"]
    out = assert_same(BASE + ["|move|p1a: Metagross|Meteor Mash|p2a: Zapdos",
                              "|-heal|p1a: Metagross|301/301|[from] item: Leftovers",
                              "|-immune|p1a: Metagross|[from] ability: Clear Body"])
    m = ours(out, "metagross")
    assert [mv["id"] for mv in m["moves"] if mv["seen"]] == ["meteormash"]
    assert m["item_public"] and m["ability_public"]
    out = assert_same(BASE + ["|move|p2a: Zapdos|Thunderbolt|p1a: Metagross"])
    assert [mv["seen"] for mv in opp(out, "zapdos")["moves"]] == [True]


def test_v18_an_item_gone_and_a_trick_are_public():
    out = assert_same(BASE + ["|-enditem|p1a: Metagross|Leftovers|[from] move: Knock Off|[of] p2a: Zapdos"])
    assert ours(out, "metagross")["item_public"]
    out = assert_same(BASE + ["|-activate|p1a: Metagross|move: Trick|[of] p2a: Zapdos"])
    assert ours(out, "metagross")["item_public"] and opp(out, "zapdos")["item_public"]


def test_v19_residual_done_is_the_current_turns_upkeep():
    assert assert_same(BASE)["view"]["residual_done"] is False
    assert assert_same(BASE + ["|", "|upkeep"])["view"]["residual_done"] is True
    assert assert_same(BASE + ["|", "|upkeep", "|turn|2"])["view"]["residual_done"] is False
