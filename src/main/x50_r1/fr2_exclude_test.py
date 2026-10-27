"""FR-2's exclusion (X59-S Amendment 4c): the patterns on synthetic protocol lines, the per-battle scan of both seats'
streams, the list's round trip with its sha256, and G-ENC's selection by the list."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import List

import pytest

from main.x50_r1 import fr2_exclude as X
from main.x50_r1 import genc

FLAGGED = {
    "|-singleturn|p1a: Heracross|move: Endure": "-singleturn:Endure",
    "|-singleturn|p1a: Heracross|Endure": "-singleturn:Endure",
    "|-singleturn|p1a: Blissey|Snatch": "-singleturn:Snatch",
    "|-singleturn|p2a: Plusle|Helping Hand|[of] p2a: Minun": "-singleturn:Helping Hand",
    "|-singlemove|p1a: Gengar|Destiny Bond": "-singlemove:Destiny Bond",
    "|-singlemove|p2a: Banette|Grudge": "-singlemove:Grudge",
    "|-singlemove|p1a: Tauros|Rage": "-singlemove:Rage",
    "|-activate|p1a: Zapdos|trapped": "-activate:trapped",
}
NOT_FLAGGED = [
    "|move|p1a: Heracross|Endure|p1a: Heracross",          # the move line, not the volatile's start
    "|move|p2a: Misdreavus|Mean Look|p1a: Zapdos",
    "|-singleturn|p1a: Skarmory|Protect",
    "|-singleturn|p1a: Heracross|move: Endured",            # a longer effect name
    "|-singlemove|p1a: Gengar|Destiny Bonding",
    "|-activate|p1a: Heracross|move: Endure",               # the Endure HIT, not its start
    "|-activate|p1a: Zapdos|move: Protect",
    "|-start|p1a: Zapdos|trapped",                          # not the keyword the registration names
    "|-end|p1a: Gengar|Destiny Bond",
    "|-activate|p1a: Smeargle|Rage",
    "|turn|3",
]


@pytest.mark.parametrize("line,kind", sorted(FLAGGED.items()))
def test_every_registered_start_is_flagged(line, kind):
    assert X.match(line) == kind


@pytest.mark.parametrize("line", NOT_FLAGGED)
def test_other_lines_are_not(line):
    assert X.match(line) is None


def _trace(root: Path, name: str, p1: List[str], p2: List[str]) -> str:
    d = root / "eval_traces" / "step_1" / "sentinel_0"
    d.mkdir(parents=True, exist_ok=True)
    for seat, lines in (("p1", p1), ("p2", p2)):
        with gzip.open(d / f"{name}.{seat}.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({"header": True}) + "\n")
            fh.writelines(json.dumps({"text": ln}) + "\n" for ln in lines)
    s = d / f"{name}_summary.json"
    s.write_text(json.dumps({"meta": {"trace_source": {"trainee_side": "p1", "records": {
        "p1": f"{name}.p1.jsonl.gz", "p2": f"{name}.p2.jsonl.gz"}}}}))
    return str(s)


def test_a_battle_is_flagged_from_either_seats_stream_and_the_list_round_trips(tmp_path):
    clean = _trace(tmp_path, "a", ["|turn|1", "|move|p1a: X|Tackle|p2a: Y"], ["|turn|1"])
    p2_only = _trace(tmp_path, "b", ["|turn|1"], ["|turn|1", "|-singlemove|p2a: Gengar|Destiny Bond"])
    both = _trace(tmp_path, "c", ["|-activate|p1a: Z|trapped", "|-singleturn|p1a: B|Snatch"],
                  ["|-activate|p1a: Z|trapped"])
    body = X.build([clean, p2_only, both])
    assert body["battles"] == [p2_only, both] and body["count"] == 2 and body["scanned"] == 3
    assert body["kinds"] == {"-activate:trapped": 1, "-singlemove:Destiny Bond": 1, "-singleturn:Snatch": 1}
    f = tmp_path / "x.json"
    f.write_text(json.dumps(body))
    keys, sha = X.load(f)
    assert keys == {X.norm(p2_only), X.norm(both)} and sha == hashlib.sha256(f.read_bytes()).hexdigest()
    # G-ENC checks only the battles the list leaves, and records what it excluded
    items, n, rec = genc.select([clean, p2_only, both], str(f))
    assert items == [clean] and n == 2 and rec == {"path": str(f), "sha256": sha, "count": 2}
    assert genc.select([clean, both], None) == ([clean, both], 0, None)


def test_a_list_of_another_schema_or_with_a_wrong_count_is_refused(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"schema": "nope", "count": 0, "battles": []}))
    with pytest.raises(ValueError, match="schema"):
        X.load(f)
    f.write_text(json.dumps({"schema": X.SCHEMA, "count": 2, "battles": ["/a"]}))
    with pytest.raises(ValueError, match="count"):
        X.load(f)
