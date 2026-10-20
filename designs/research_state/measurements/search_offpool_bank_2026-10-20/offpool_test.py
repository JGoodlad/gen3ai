"""Tests of this unit's pure parts: the constants' rules, the registered read's selfcheck, the near-copy keys and the
registration's file digests. Run: ``python -m pytest designs/research_state/measurements/search_offpool_bank_2026-10-20/
offpool_test.py -q`` with ``PYTHONPATH=src``."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import offpool_common as E  # noqa: E402


def test_ids_sha_is_the_maturity_banks():
    from main.hidden_info_study import maturity_bank as MB

    ids = ["b#p1#3", "a#p2#1", "c#p1#9"]
    assert E.ids_sha(ids) == MB.ids_sha(ids)
    assert E.ids_sha(ids) == E.ids_sha(list(reversed(ids)))


def test_stratum_rule():
    assert E.stratum("sentinel_0") == "snapshot"
    assert all(E.stratum(b) == "bot" for b in E.BOTS)
    with pytest.raises(SystemExit):
        E.stratum("ext_metamon")


def test_parts_and_seeds_are_complete_and_distinct():
    assert set(E.PARTS) == set(E.GAMES)
    keys = [(c, p) for c in E.CYCLES + (E.EXTENSION,) for p in E.PARTS]
    assert sorted(keys) == sorted(E.SEEDS)
    assert len(set(E.SEEDS.values())) == len(E.SEEDS) and E.SMOKE_SEED not in E.SEEDS.values()
    covered = sorted(i for p, r in E.PARTS.items() if r for i in range(*r))
    assert covered == list(range(len(E.SENTINELS)))           # every sentinel in exactly one part


def test_read_selfcheck():
    import offpool_read as R

    R.selfcheck()


def test_near_copy_keys():
    import offpool_run as RUN

    a = "Snorlax||Leftovers|ThickFat|BodySlam,Curse,Rest,HiddenPower|Careful|252,,,,252,4|||||,Ghost,,,,"
    b = "Snorlax||ChestoBerry|Immunity|BodySlam,Curse,Rest,HiddenPower|Adamant|252,252,,,,4|||||,Fighting,,,,"
    assert RUN.loose_key(a) == RUN.loose_key(b)              # species + moves, Hidden Power untyped
    team_a = "]".join([a] * 2)
    team_b = "]".join([b] * 2)
    assert RUN.shared(RUN.loose_key, team_a, team_b) == 2


def test_registration_digests():
    reg = json.loads((HERE / "offpool_registration.json").read_text())
    for name, sha in reg["sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == sha, name
