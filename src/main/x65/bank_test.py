"""B1's label bank: π′, the keys, outcome-blindness, and — on the frame probe's one real miss (S′-LIVE
``pstar_s__forum72__340__g10``, our second decision inside request update 24 after an ``[Unavailable choice]``) — the
ordered rebuild and the identity guard (PREREG §9: "the identity guard on a constructed mis-ordered log")."""
from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from main.x65 import bank as B

FIXTURE = Path(__file__).parent / "testdata" / "misordered_cell"
#: the probe's miss: our decision n 23 (turn 23), the log handed to the search held 46 commands
MISS_N, MISS_AT = 23, 46


def _recs():
    return [json.loads(x) for x in (FIXTURE / "search_decisions.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------------------------------------ pure
def test_pi_prime_is_the_pikl_softmax_over_the_legal_actions():
    q = {"1": 0.40, "3": 0.50, "7": 0.45}
    lp = {"1": -0.5, "3": -2.0, "7": -1.2}
    p = B.pi_prime(q, lp, lam=0.015)
    s = np.array([lp[k] + q[k] / 0.015 for k in ("1", "3", "7")])
    e = np.exp(s - s.max())
    assert np.allclose(p[[1, 3, 7]], e / e.sum(), rtol=0, atol=1e-15)
    assert p.sum() == pytest.approx(1.0, abs=1e-15)
    assert (p[[0, 2, 4, 5, 6, 8, 9, 10]] == 0).all()
    assert int(np.argmax(p)) == 3                     # Q / λ dominates: the mode is S′'s pick
    with pytest.raises(B.BankError):
        B.pi_prime({"1": 0.1}, {"2": -0.1})


def test_inner_validation_is_a_function_of_the_deal_alone():
    import hashlib

    deals = [f"{i:020x}" for i in range(4000)]
    got = [B.inner_val(d) for d in deals]
    assert got == [hashlib.sha256(d.encode()).digest()[0] % 10 == 0 for d in deals]
    assert 0.08 < np.mean(got) < 0.12                 # 26 of 256 first-byte values


def test_the_builder_refuses_every_result_bearing_file(tmp_path):
    for name in ("games.jsonl", "summary.json", "forced_ends.jsonl", "peer_acceptor_report.json"):
        p = tmp_path / name
        p.write_text("{}")
        with pytest.raises(B.OutcomeBlindError):
            B.read_json(p)
        with pytest.raises(B.OutcomeBlindError):
            B.read_jsonl(p)


def test_counted_units_keep_only_the_counted_units_structure(tmp_path):
    rec = {"rule": "searched_play", "void": None, "inputs": {"raw": "pstar", "searched": "pstar_s", "deals": [
        {"deal": "forum72:300:0", "game": 0, "panel": "forum72", "unit_seed": 300,
         "raw": {"status": "counted", "unit": "pstar__forum72__300__g00"},
         "searched": {"status": "exhausted", "unit": "pstar_s__forum72__300__g00"}}]},
           "units": [{"games": [["x", 1, "t", "win", 9]]}]}
    (tmp_path / "record.json").write_text(json.dumps(rec))
    assert B.counted_units(tmp_path / "record.json") == [
        {"label": "pstar", "unit": "pstar__forum72__300__g00", "panel": "forum72", "unit_seed": 300, "game": 0}]


def test_the_capture_reader_never_keeps_the_chunks_and_stops_at_the_end(tmp_path):
    cell = tmp_path / "c"
    (cell / "capture").mkdir(parents=True)
    start = {"formatid": "gen3ou", "seed": [1, 2, 3, 4], "p1": {}, "p2": {}, "core_obs": {"sides": ["p1"]}}
    (cell / "capture" / "b.json").write_text(json.dumps({"seed": [1, 2, 3, 4], "commands": [
        "START " + json.dumps(start), "CHOOSE p1 move a", "CHOOSE p2 move b", "FORCELOSE p2", "END"],
        "chunks": [["p1", "|win|someone"]]}))
    st, cmds = B.read_capture(cell)
    assert cmds == ["CHOOSE p1 move a", "CHOOSE p2 move b"] and B.our_side(st) == "p1"


# ------------------------------------------------------------------------------------------------ the core
@pytest.mark.sim
@pytest.mark.integration
def test_the_naive_prefix_misses_the_probe_frame():
    """The hazard B1 removes (the frame probe's one miss): a naive ``cmds[:at]`` holds the opponent's update-24
    choice instead of our own refused first choice, and its replayed row is not the live frame."""
    from main.anchors.input_log import frame_sha256
    from main.value_search import worlds as W

    start, cmds = B.read_capture(FIXTURE)
    rec = _recs()[MISS_N]
    assert int(rec["at"]) == MISS_AT and rec["searched"] and cmds[MISS_AT - 1].startswith("CHOOSE p1 ")
    truth = W.prepare_root(B.naive_cut(start, cmds, MISS_AT), MISS_AT, "p2", rec["key"])
    assert frame_sha256(truth.ref[-1]) != rec["frame_sha256"]


@pytest.mark.sim
@pytest.mark.integration
def test_the_ordered_rebuild_matches_every_recorded_frame_including_the_miss():
    rows, st = B.game_rows(FIXTURE, "pstar_s")
    recs = _recs()
    assert st["drop"] == {} and st["labels_kept"] == st["labels_recorded"] == sum(1 for r in recs if r["searched"])
    assert st["frame_ok"] == st["frame_checked"] == sum(1 for r in recs if r.get("frame_sha256"))
    assert st["log_sha_bad"] == st["at_bad"] == st["turn_bad"] == st["xcheck_bad"] == 0
    miss = [r for r in rows if r["n"] == MISS_N]
    assert len(miss) == 1 and miss[0]["label"] and miss[0]["at"] == MISS_AT and miss[0]["update"] == 24
    assert [r["n"] for r in rows] == list(range(len(recs)))
    assert all(r["turn"] == int(rec["turn"]) for r, rec in zip(rows, recs))


@pytest.mark.sim
@pytest.mark.integration
def test_the_identity_guard_drops_a_misordered_log(monkeypatch):
    """A mis-ordered log (the naive prefix, substituted for the ordered cut) gives the miss a wrong row: the guard
    DROPS that label row and counts it; it never reaches the bank."""
    real = B.honest_cut

    def misordered(start, cmds, upd, side, p, u):
        return B.naive_cut(start, cmds, len(real(start, cmds, upd, side, p, u)["cmds"]))

    monkeypatch.setattr(B, "honest_cut", misordered)
    rows, st = B.game_rows(FIXTURE, "pstar_s")
    assert not any(r["n"] == MISS_N for r in rows)
    assert st["labels_kept"] == st["labels_recorded"] - 1
    lost = {k: v for k, v in st["drop"].items() if k.startswith("label:")}
    assert sum(lost.values()) == 1, st["drop"]


@pytest.mark.sim
@pytest.mark.integration
def test_a_corrupted_frame_digest_is_dropped_and_counted(tmp_path):
    cell = tmp_path / "cell"
    shutil.copytree(FIXTURE, cell)
    recs = _recs()
    recs[MISS_N - 1]["frame_sha256"] = "0" * 64
    (cell / "search_decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows, st = B.game_rows(cell, "pstar_s")
    assert st["drop"].get("label:frame_sha256") == 1
    assert not any(r["n"] == MISS_N - 1 for r in rows)
    del recs[MISS_N - 2]["frame_sha256"]                      # a label row with no digest is never kept either
    (cell / "search_decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows, st = B.game_rows(cell, "pstar_s")
    assert st["drop"].get("label:no_frame_sha256") == 1
    assert not any(r["n"] == MISS_N - 2 for r in rows)


@pytest.mark.sim
@pytest.mark.integration
def test_the_bank_builds_checks_and_never_opens_a_result_file(tmp_path, monkeypatch):
    state = tmp_path / "state"
    cells = state / "anchor" / "u" / "pstar_s" / "cells"
    cells.mkdir(parents=True)
    shutil.copytree(FIXTURE, cells / "pstar_s__forum72__340__g10")
    (cells / "pstar_s__forum72__340__g10" / "games.jsonl").write_text("THE RESULT\n")
    (cells / "pstar_s__forum72__340__g10" / "summary.json").write_text("{}")
    rec = {"rule": "searched_play", "void": None, "inputs": {"raw": "pstar", "searched": "pstar_s", "deals": [
        {"deal": "forum72:340:10", "game": 10, "panel": "forum72", "unit_seed": 340,
         "raw": {"status": "exhausted", "unit": "pstar__forum72__340__g10"},
         "searched": {"status": "counted", "unit": "pstar_s__forum72__340__g10"}}]}}
    (state / "record.json").write_text(json.dumps(rec))
    opened = []
    real_open = builtins.open

    def spy(file, *a, **k):
        opened.append(Path(str(file)).name)
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    out = tmp_path / "bank"
    man = B.build(state, out, workers=1, model=None, allow_dirty=True)
    monkeypatch.setattr(builtins, "open", real_open)
    assert not set(opened) & B.FORBIDDEN
    assert man["void"] is None and man["label_rows"] == sum(1 for r in _recs() if r["searched"])
    pp = man["per_panel"]["forum72"]
    assert pp["labels_recorded"] == pp["labels_kept"] and pp["label_drop_frac"] == 0
    sha = (out / "bank_manifest.sha256").read_text().strip()
    man["dirty"] = False
    (out / "bank_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    sha = B.sha256_file(out / "bank_manifest.json")
    assert B.check(out, sha)["label_rows"] == man["label_rows"]
    with pytest.raises(B.BankError):
        B.check(out, "f" * 64)
    z = B.load_rows(out, ["forum72"])
    assert z["obs"].shape == (len(_recs()), 2845) and int(z["label"].sum()) == man["label_rows"]
    lab = z["label"]
    assert np.allclose(z["pi_prime"][lab].sum(1), 1.0) and (z["pi_prime"][~lab] == 0).all()
    assert ((z["pi_prime"] > 0) <= (z["mask"] == 1)).all()
