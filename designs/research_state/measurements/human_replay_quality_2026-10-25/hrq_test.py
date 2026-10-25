"""Tests of the data card's text pass, bot rule and set fit (``hrq.py``). Run: pytest <this file> (not in testpaths)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hrq  # noqa: E402

LOG = "\n".join([
    "|init|battle", "|title|A vs. B", "|j|☆Alice", "|t:|1779180344", "|gametype|singles",
    "|player|p1|Alice|lyra|1540", "|player|p2|PAC-Bot|n|1498", "|gen|3", "|tier|[Gen 3] OU", "|rated|",
    "|rule|Sleep Clause Mod: Limit one foe put to sleep", "|t:|1779180344", "|start",
    "|switch|p1a: Tyranitar|Tyranitar, M|100/100", "|switch|p2a: Zapdos|Zapdos|100/100", "|turn|1",
    "|c|☆Alice|gl", "|t:|1779180368", "|move|p1a: Tyranitar|Rock Slide|p2a: Zapdos",
    "|move|p2a: Zapdos|Thunderbolt|p1a: Tyranitar", "|-damage|p1a: Tyranitar|80/100", "|turn|2",
    "|t:|1779180400", "|-message|PAC-Bot forfeited.", "|win|Alice", "|raw|rating change",
])


def test_parse_reads_header_end_and_reveals():
    r = hrq.parse(LOG)
    assert (r["n1"], r["n2"], r["r1"], r["r2"], r["rated"]) == ("Alice", "PAC-Bot", 1540, 1498, True)
    assert (r["gen"], r["tier"], r["gametype"], r["turns"]) == ("3", "[Gen 3] OU", "singles", 2)
    assert (r["end"], r["win"], r["loser_quit"]) == ("forfeit", "p1", "p2")
    assert r["chat1"] == 1 and r["chat2"] == 0
    assert r["o1"] == ["tyranitar"] and r["o2"] == ["zapdos"]
    assert r["m1"]["tyranitar"][0] == ["rockslide"] and r["m2"]["zapdos"][0] == ["thunderbolt"]
    assert (r["t0"], r["t1"]) == (1779180344, 1779180400)


def test_body_hash_ignores_room_lines():
    a = hrq.parse(LOG)["body"]
    b = hrq.parse(LOG.replace("|c|☆Alice|gl", "|c|☆Alice|other").replace("|raw|rating change", ""))["body"]
    assert a == b


def test_forfeit_without_result_is_no_result():
    r = hrq.parse(LOG.replace("|win|Alice", ""))
    assert r["end"] == "none" and r["win"] is None


def test_bot_rule_name_volume_hours():
    import collections
    mk = lambda name, per_day, hours: {  # noqa: E731
        "name": name, "days": collections.Counter({"2026-07-01": per_day}),
        "hours": {"2026-07-01": set(range(hours))}}
    per = {"a": mk("PAC-MM-Alakazam", 1, 1), "b": mk("Human", hrq.BOT_GAMES_PER_DAY, 3),
           "c": mk("Human2", hrq.BOT_GAMES_PER_DAY - 1, hrq.BOT_HOURS_PER_DAY), "d": mk("Robotnik", 5, 5),
           "e": mk("Human3", hrq.BOT_GAMES_PER_DAY - 1, hrq.BOT_HOURS_PER_DAY - 1)}
    f = hrq.bot_flags(per)
    assert f["a"] == ["name"] and f["b"] == ["volume"] and f["c"] == ["hours"]
    assert "d" not in f and "e" not in f          # 'bot' inside a word is not a bot name; one under each bar is clean


def test_fit_rule_matches_memory_rule():
    from main.anchors.team_overlap import MonSet
    ms = MonSet("zapdos", ("hiddenpowergrass", "thunderbolt", "toxic", "batonpass"), "leftovers", "pressure")
    assert hrq.fit(["thunderbolt", "hiddenpower"], "leftovers", None, ms)
    assert not hrq.fit(["thunderbolt", "drillpeck"], None, None, ms)
    assert not hrq.fit(["thunderbolt"], "choiceband", None, ms)
    assert hrq.fit(["struggle"], None, "pressure", ms)
