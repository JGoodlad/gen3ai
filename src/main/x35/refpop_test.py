"""X35's node rules and the reference-population freeze (``main.x35.refpop``) on SYNTHETIC checkpoint directories:
the registered P* = 180M answer, ties down, the refusals, the band's rule-8 edges, and the write-once freeze with
its sha256s. No real checkpoint is read (the two FROZEN call files are read as JSON only)."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from main.x35 import refpop as RP
from utils.paths import repo_path

M = 1_000_000
STAGE = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17")


def make_archive(d: Path, last_nominal_m: int = 180, *, start_m: int = 1, overshoot: int = 55) -> Path:
    """A checkpoints/ dir with one tiny file per 1M step (its bytes name its step, so every sha256 differs)."""
    d.mkdir(parents=True, exist_ok=True)
    for m in range(start_m, last_nominal_m + 1):
        s = m * M + overshoot
        (d / f"checkpoint_{s}_steps.zip").write_bytes(f"ckpt {s}".encode())
        (d / f"checkpoint_{s}_steps.json").write_text("{}")
    return d


def test_a_node_is_the_checkpoint_within_one_interval_above_its_10m_multiple():
    assert RP.node_of(180_000_055) == 180 * M
    assert RP.node_of(180_999_999) == 180 * M
    assert RP.node_of(181_000_010) is None
    assert RP.node_of(179_999_999) is None


def test_the_registered_p_star_180m_answer():
    assert RP.archive_targets(180 * M) == [k * 20 * M for k in range(1, 9)]
    assert RP.a_block_targets(180 * M) == [("pstar", 180 * M), ("pstar-10M", 170 * M), ("mid(N2,N3)", 50 * M),
                                           ("mid(N4,N5)", 90 * M), ("mid(N6,N7)", 130 * M)]


def test_ties_round_down_and_a_non_multiple_midpoint_rounds_to_the_nearest_node():
    assert RP.nearest_multiple(Fraction(85 * M)) == 80 * M
    assert RP.nearest_multiple(Fraction(85 * M + 1)) == 90 * M
    # P* = 190M: k * 21.25M -> 20, 40, 60, 80 (85 is a tie: down), 110, 130, 150, 170
    assert RP.archive_targets(190 * M) == [x * M for x in (20, 40, 60, 80, 110, 130, 150, 170)]
    # mid(N4, N5) = 95M, a tie: down to 90M
    assert [x for _, x in RP.a_block_targets(190 * M)] == [x * M for x in (190, 180, 50, 90, 140)]


def test_too_early_a_p_star_or_a_colliding_a_block_is_refused():
    with pytest.raises(RP.RefPopError, match="too early"):
        RP.archive_targets(90 * M)
    with pytest.raises(RP.RefPopError, match="not disjoint"):
        RP.a_block_targets(100 * M)  # mid(N2,N3) = 25M ties down onto N2 = 20M
    with pytest.raises(RP.RefPopError, match="not a 10M node"):
        RP.archive_targets(185 * M)


def test_two_directories_claiming_one_node_are_refused(tmp_path):
    a = make_archive(tmp_path / "a" / "checkpoints", 20)
    b = make_archive(tmp_path / "b" / "checkpoints", 20, start_m=20, overshoot=99)
    with pytest.raises(RP.RefPopError, match="two checkpoints claim"):
        RP.scan_nodes([a, b])


@pytest.mark.parametrize("score,kept", [(0.5, True), (0.2001, True), (0.7999, True), (0.19, False), (0.81, False),
                                        (0.20, False), (0.80, False), (0.20 + 1e-12, False), (0.80 - 1e-12, False)])
def test_the_band_keeps_the_open_interior_and_excludes_its_rule_8_edges(score, kept):
    got, why = RP.band_decision(RP.BandCell(score=score, n_pairs=500, verdict="OK"))
    assert got is kept and (why is None) is kept


@pytest.mark.parametrize("cell", [None, RP.BandCell(0.5, 499, "OK"), RP.BandCell(0.5, 500, "INCONCLUSIVE"),
                                  RP.BandCell(None, 500, "OK")])
def test_a_band_cell_that_cannot_decide_refuses(cell):
    with pytest.raises(RP.RefPopError):
        RP.band_decision(cell)


def _freeze(tmp_path, band, out_name="R.json"):
    ck = make_archive(tmp_path / "rb_x26" / "checkpoints", 185)
    tstar = tmp_path / "tstar.zip"
    tstar.write_bytes(b"toy")
    prod = tmp_path / "prod.zip"
    prod.write_bytes(b"production")
    out = tmp_path / out_name
    doc = RP.freeze(side="production", pstar=str(ck / "checkpoint_180000055_steps.zip"), archive_dirs=[str(ck)],
                    label_prefix="x26", x_candidates=[("tstar", str(tstar)), ("production", str(prod))],
                    band=band, out=str(out), tool_commit="abc")
    return doc, out, ck, tstar


def test_the_freeze_writes_every_member_with_its_sha256_and_verifies(tmp_path):
    band = {"tstar": RP.BandCell(0.35, 500, "OK"), "production": RP.BandCell(0.86, 500, "OK")}
    doc, out, ck, tstar = _freeze(tmp_path, band)
    assert [e["label"] for e in doc["archive"]] == [f"x26_{20 * k:03d}m" for k in range(1, 9)]
    assert [m["label"] for m in doc["members"]] == ["x26_pstar", "x26_170m", "x26_050m", "x26_090m", "x26_130m",
                                                    "tstar"]
    x = {e["label"]: e for e in doc["x_block"]}
    assert x["tstar"]["kept"] and not x["production"]["kept"] and "outside" in x["production"]["excluded_because"]
    assert doc["pstar"]["sha256"] == RP.sha256_file(ck / "checkpoint_180000055_steps.zip")
    assert RP.verify(str(out)) == []
    with pytest.raises(RP.RefPopError, match="frozen ONCE"):
        _freeze(tmp_path / "again", band, out_name="../R.json")
    tstar.write_bytes(b"retrained")
    assert any("sha256 changed" in p for p in RP.verify(str(out)))


def test_an_edited_freeze_file_fails_its_content_hash(tmp_path):
    doc, out, _, _ = _freeze(tmp_path, {"tstar": RP.BandCell(0.5, 500, "OK"),
                                        "production": RP.BandCell(0.5, 500, "OK")})
    d = json.loads(out.read_text())
    d["members"] = d["members"][:-1]
    out.write_text(json.dumps(d))
    assert any("content_sha256" in p for p in RP.verify(str(out)))


def test_an_undecided_x_candidate_refuses_the_freeze(tmp_path):
    with pytest.raises(RP.RefPopError, match="undecided member"):
        _freeze(tmp_path, {"tstar": RP.BandCell(0.5, 500, "OK")})
    assert not (tmp_path / "R.json").exists()


def test_the_cli_freezes_from_a_band_json_and_verifies(tmp_path, capsys):
    ck = make_archive(tmp_path / "rb_x26" / "checkpoints", 180)
    t = tmp_path / "t.zip"
    t.write_bytes(b"t")
    bj = tmp_path / "band.json"
    bj.write_text(json.dumps({"tstar": {"score": 0.42, "n_pairs": 500, "verdict": "OK"}}))
    out = tmp_path / "R.json"
    argv = ["--side", "production", "--pstar", str(ck / "checkpoint_180000055_steps.zip"), "--archive-dir", str(ck),
            "--x", f"tstar={t}"]
    assert RP.main(["plan", *argv]) == 0
    assert RP.main(["freeze", *argv, "--band-json", str(bj), "--out", str(out)]) == 0
    assert RP.main(["verify", str(out)]) == 0
    assert RP.main(["freeze", *argv, "--band-json", str(bj), "--out", str(out)]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_the_registered_t_star_290m_answer():
    """The toy side's REAL start node (X34 @ 290,000,087, 2026-10-19), hand-computed from the rule: k · 270M / 8 =
    33.75, 67.5 (> half: up), 101.25, 135 (a tie: DOWN), 168.75, 202.5, 236.25, 270M, and the three midpoints
    (70+100)/2 = 85 (a tie: DOWN), (130+170)/2 = 150, (200+240)/2 = 220M."""
    assert RP.archive_targets(290 * M) == [30 * M, 70 * M, 100 * M, 130 * M, 170 * M, 200 * M, 240 * M, 270 * M]
    assert RP.a_block_targets(290 * M) == [("pstar", 290 * M), ("pstar-10M", 280 * M), ("mid(N2,N3)", 80 * M),
                                           ("mid(N4,N5)", 150 * M), ("mid(N6,N7)", 220 * M)]


@pytest.mark.parametrize("name,pstar_sha,step", [
    ("reference_population_production.json", "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed", 280_000_170),
    ("reference_population_toy.json", "cfa7f7bee7b7a43959f496ce6d5a34556bae914de9c80f90f9342a05a22965e7", 290_000_087),
])
def test_a_frozen_r_file_is_unedited_names_its_start_node_and_obeys_the_rule(name, pstar_sha, step):
    """R is write-once: its content hash holds, it names the registered start node (P* for production, the real T* for
    the toy: NOT the 200M stand-in), its archive and A-block are the rule's nodes for that start, and every X member's
    kept / excluded verdict is the band rule's own on its recorded score. (The files' sha256 are `refpop verify`'s,
    which needs the checkpoints and so runs at every launch.)"""
    doc = json.loads((STAGE / name).read_text())
    assert hashlib.sha256(RP._canon(doc)).hexdigest() == doc["content_sha256"]
    assert doc["schema"] == RP.SCHEMA and doc["side"] == name.split("_")[-1].split(".")[0]
    assert (doc["pstar"]["sha256"], doc["pstar"]["step"]) == (pstar_sha, step)
    pn = RP.node_of(step)
    assert [e["nominal"] for e in doc["archive"]] == RP.archive_targets(pn)
    assert [e["nominal"] for e in doc["a_block"]] == [t for _, t in RP.a_block_targets(pn)]
    assert doc["a_block"][0]["sha256"] == pstar_sha
    for x in doc["x_block"]:
        kept, _ = RP.band_decision(RP.BandCell(x["band_score"], x["band_n_pairs"], "OK"))
        assert kept == x["kept"], x["label"]
    assert [m["label"] for m in doc["members"] if m["block"] == "X"] == [x["label"] for x in doc["x_block"] if x["kept"]]
