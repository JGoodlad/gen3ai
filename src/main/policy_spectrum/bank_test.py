"""The bank's selection, stamps, storage — and gate ② on the COMMITTED v1 bank (unit; no core)."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict

import pytest

from main.policy_spectrum import bank as B
from main.policy_spectrum.categories import CATEGORIES
from utils.paths import repo_path

BANK_V1 = repo_path("designs", "research_state", "measurements", "m5_laneS", "bank_v1")


def test_pick_traces_quota_fill_and_determinism():
    prefixes = [f"loss_s0_{i:03d}" for i in range(9)] + ["win_s0_001", "draw_s1_002"]
    a = B.pick_traces(prefixes)
    assert a == B.pick_traces(list(reversed(prefixes)))
    outc = [B.outcome_of(p) for p in a]
    assert len(a) == 5
    assert outc.count("win") == 1 and outc.count("draw") == 1 and outc.count("loss") == 3
    assert B.pick_traces(["win_s0_001"]) == ["win_s0_001"]


def test_outcome_and_opponent_vocabulary_refuse_unknowns():
    with pytest.raises(ValueError):
        B.outcome_of("tie_s0_001")
    with pytest.raises(KeyError):
        B.opponent_class("metamon")
    assert B.opponent_class("sentinel_3") == "pool_snapshot"
    assert B.opponent_class("staller_v2") == "bot"


def test_phase_and_legal_buckets():
    assert B.phase_of(2, 6, 6) == "opening"
    assert B.phase_of(10, 2, 5) == "endgame"
    assert B.phase_of(10, 4, 3) == "midgame"
    assert [B.legal_bucket(n) for n in (2, 3, 4, 6, 7, 11)] == ["2-3", "2-3", "4-6", "4-6", "7+", "7+"]


def _toy():
    b = B.BankBattle(battle_id="r/step_1/random/win_s0_001", source={"label": "t", "opponent": "random"},
                     format_id="gen3ou", seed="1,2,3,4", p1={"name": "a", "team": "x"},
                     p2={"name": "b", "team": "y"}, commands=[["p1", "move 1"]], banked_side="p1",
                     traced_side="p1", outcome="win")
    d = {"id": "r#p1#0", "battle": b.battle_id, "n": 0, "cats": {"6": "attack"}}
    return b, d


def test_write_load_roundtrip_is_byte_stable_and_refuses_an_edit(tmp_path):
    b, d = _toy()
    man = {"schema": B.BANK_SCHEMA, "content_sha256": B.content_sha([b], [d])}
    B.write_bank(tmp_path / "a", man, [b], [d])
    B.write_bank(tmp_path / "b", man, [b], [d])
    for f in ("battles.jsonl.gz", "decisions.jsonl.gz", "manifest.json"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()
    got = B.load_bank(tmp_path / "a")
    assert asdict(got.battles[0]) == asdict(b) and got.decisions == [d]
    edited = dict(d, cats={"6": "status"})
    (tmp_path / "a" / "decisions.jsonl.gz").write_bytes(B._gz_bytes([B._dumps(edited)]))
    with pytest.raises(ValueError, match="edited"):
        B.load_bank(tmp_path / "a")


# ------------------------------------------------------------------------------------------------
# gate ② on the committed bank
# ------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v1():
    return B.load_bank(BANK_V1)


def test_v1_bank_size_and_gate_one_record(v1):
    m = v1.manifest
    assert len(v1.decisions) >= 10_000
    assert m["counts"] == B.summarize(v1.decisions)
    assert m["gate1"]["result"] == "PASS" and not m["gate1"]["failures"]
    assert m["gate1"]["decisions_checked"] >= m["counts"]["gate1_checkable"]


def test_v1_bank_every_decision_is_stamped_in_vocabulary(v1):
    ids = set()
    battles = {b.battle_id: b for b in v1.battles}
    for d in v1.decisions:
        assert d["id"] not in ids
        ids.add(d["id"])
        assert d["battle"] in battles and d["side"] == battles[d["battle"]].banked_side
        assert d["kind"] in ("free", "forced_switch", "switch_only")
        assert d["phase"] in ("opening", "midgame", "endgame")
        assert d["opp_class"] in ("bot", "pool_snapshot", "exploiter")
        assert d["outcome"] in ("win", "loss", "draw")
        assert d["n_legal"] >= 2 and d["n_legal"] == d["mask"].count("1") == len(d["tokens"])
        assert set(d["cats"]) == set(d["tokens"]) and set(d["cats"].values()) <= set(CATEGORIES)
        assert d["played"] in d["tokens"].values()
        if d["kind"] == "forced_switch":
            assert set(d["cats"].values()) == {"switch"}
        # the bank stores NO observation: the recording is kept as a hash and 11 logits
        assert not any(isinstance(v, list) and len(v) > 11 for v in d.values())
        if d["rec_obs_sha256"]:
            assert d["tokens"][str(d["rec_action"])] == d["played"]


def test_v1_bank_covers_every_stratum(v1):
    c = v1.manifest["counts"]
    assert set(c["kind"]) >= {"free", "forced_switch"}
    assert set(c["phase"]) == {"opening", "midgame", "endgame"}
    assert set(c["opp_class"]) == {"bot", "pool_snapshot", "exploiter"}
    assert set(c["category_legal"]) == set(CATEGORIES)
    assert min(c["category_legal"].values()) >= 1000
    assert set(c["opp_name"]) >= set(B.BOTS)
    assert c["teams"] >= 50


def test_v1_bank_files_are_gzip_mtime_zero():
    for f in ("battles.jsonl.gz", "decisions.jsonl.gz"):
        raw = (BANK_V1 / f).read_bytes()
        assert raw[4:8] == b"\x00\x00\x00\x00"          # the gzip header's MTIME
        with gzip.open(BANK_V1 / f, "rt") as fh:
            json.loads(fh.readline())
