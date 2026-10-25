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


# ---- the revision (REVIEW A M5 / m1, REVIEW C F13)
def _snapshot(tmp_path, population="spectator", body=LOG.encode()):
    import gzip
    import hashlib
    import json
    rel = "2026-07-01/battle-gen3ou-1.log"
    (tmp_path / "corpus" / "2026-07-01").mkdir(parents=True)
    (tmp_path / "corpus" / rel).write_bytes(body)
    tsv = f"{rel}\t{hashlib.sha256(body).hexdigest()}\t{len(body)}\n".encode()
    (tmp_path / "store").mkdir()
    with gzip.open(tmp_path / "store" / "snap.tsv.gz", "wb") as f:
        f.write(tsv)
    reg = {"snapshots": {"snap": {"population": population, "files": 1,
                                  "manifest_sha256": hashlib.sha256(tsv).hexdigest()}}}
    (tmp_path / "reg.json").write_text(json.dumps(reg))
    return rel, body


def test_read_manifest_checks_the_registry_digest(tmp_path):
    import json
    import pytest
    rel, body = _snapshot(tmp_path)
    rows = hrq.read_manifest("snap", tmp_path / "store", tmp_path / "reg.json")
    assert rows[0][0] == rel and rows[0][2] == len(body)
    reg = json.loads((tmp_path / "reg.json").read_text())
    reg["snapshots"]["snap"]["manifest_sha256"] = "0" * 64                    # planted mismatch
    (tmp_path / "reg.json").write_text(json.dumps(reg))
    with pytest.raises(hrq.SnapshotError):
        hrq.read_manifest("snap", tmp_path / "store", tmp_path / "reg.json")
    with pytest.raises(hrq.SnapshotError):
        hrq.registry_row("hr_not_registered", tmp_path / "reg.json")


def test_verify_bytes_refuses_a_changed_log_or_a_wrong_header(tmp_path):
    import hashlib
    import pytest
    rel, body = _snapshot(tmp_path)
    sha = hashlib.sha256(body).hexdigest()
    hrq.verify_bytes("snap", "spectator", rel, sha, len(body), body)
    with pytest.raises(hrq.SnapshotError):
        hrq.verify_bytes("snap", "spectator", rel, sha, len(body), body[:-1] + b"X")
    with pytest.raises(hrq.SnapshotError):                                    # an archive log never starts |init|battle
        hrq.verify_bytes("snap", "archive", rel, sha, len(body), body)


def test_account_ids_are_salted_and_never_the_name():
    a = hrq.acct("Alice", salt=b"s1")
    assert a.startswith("acct_") and "alice" not in a.lower()
    assert a == hrq.acct("ALICE", salt=b"s1")                                 # the Showdown id, not the display name
    assert a != hrq.acct("Alice", salt=b"s2")


def test_drift_by_account_has_no_excess_when_the_months_are_one_population():
    import random
    rng = random.Random(3)
    species = [f"sp{i}" for i in range(40)]
    accts = [f"u{i}" for i in range(400)]
    team = {u: tuple(rng.sample(species, 6)) for u in accts}            # each player keeps one team: repeat players

    def month(tag, shift):
        return [{"acct": u, "o": team[u] if not shift else tuple(sorted(team[u]))[:5] + ("new",)}
                for u in accts for _ in range(3)]
    marg = lambda ts: {k: v / sum(c.values()) for c in [__import__("collections").Counter(x for t in ts for x in t)]  # noqa: E731
                       for k, v in c.items()}
    tvd = lambda p, q: 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))  # noqa: E731
    same = hrq.drift_by_account(marg, tvd, {"spec 2026-06": month("a", False), "spec 2026-07": month("b", False)},
                                k_splits=5)
    moved = hrq.drift_by_account(marg, tvd, {"spec 2026-06": month("a", False), "spec 2026-07": month("b", True)},
                                 k_splits=5)
    assert abs(same["excess"]["2026-06->2026-07"]["mean"]) < 0.02
    assert moved["excess"]["2026-06->2026-07"]["min"] > 0.05
