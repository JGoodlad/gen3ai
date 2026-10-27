"""Foul Play's team pack under the front end's played-team rule (``main.anchors.frontend_pack``, SOP H22).

The front end plays the validator's re-pack (a real server's `valResult.slice(1)`); Foul Play's own packer writes no
Hidden Power IVs. Real Node, real Showdown, and Foul Play's own interpreter when the box has it."""
from __future__ import annotations

from pathlib import Path

import pytest

from main.anchors import frontend_pack as fpk
from main.anchors.config import AnchorConfigError, load_config

pytestmark = pytest.mark.integration

#: A mon with Hidden Power Bug and no IVs, packed the way Foul Play's `json_to_packed` writes it (blank IV slots).
_HP_BUG = ("Tyranitar||leftovers|sandstream|hiddenpowerbug,rockslide,earthquake,crunch|adamant|252,252,,,4,|"
           "||||,,,,,,|,,,,,")
_PLAIN = "Snorlax||leftovers|thickfat|bodyslam,curse,rest,sleeptalk|careful|252,4,,,252,||,,,,,|||,,,,,"
#: Metamon's files end in blank lines; Foul Play packs the trailing "\n\n" as a SEVENTH empty mon, which the validator refuses.
_SEVEN = "]".join([_PLAIN] * 1 + ["||||||,,,,,||,,,,,|||,,,,,"] * 6)


def test_a_hidden_power_without_ivs_plays_differently_re_packed_and_a_plain_set_does_not(tmp_path):
    files = {}
    for name, pack in (("hp.txt", _HP_BUG), ("plain.txt", _PLAIN), ("seven.txt", _SEVEN)):
        f = tmp_path / name
        f.write_text("x")
        files[str(f)] = pack
    rec = fpk.build(list(files), files)
    assert rec["schema"] == fpk.FRONTEND_PACK_SCHEMA and rec["n_files"] == 3
    assert [c["file"] for c in rec["changed"]] == ["hp.txt"]
    assert any("hiddenpower" in d for d in rec["changed"][0]["diffs"]), rec["changed"]
    assert rec["frontend_pack_changed"] is True
    assert rec["refused"] == ["seven.txt"], "a file the validator refuses never plays, before or after: not a change"
    only_plain = fpk.build([str(tmp_path / "plain.txt")], {str(tmp_path / "plain.txt"): _PLAIN})
    assert only_plain["frontend_pack_changed"] is False and only_plain["changed"] == [] and only_plain["refused"] == []


def test_foul_plays_walk_skips_hidden_files_and_dirs(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / ".hid").write_text("x")
    (tmp_path / ".d").mkdir()
    (tmp_path / ".d" / "b.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("x")
    assert [Path(f).name for f in fpk.fp_team_files(tmp_path)] == ["a.txt", "c.txt"]
    assert [Path(f).name for f in fpk.fp_team_files(tmp_path / "a.txt")] == ["a.txt"]


@pytest.mark.slow       # Foul Play's interpreter + two Node passes over its 72-file pool
def test_foul_plays_actual_team_sources_are_a_new_regime_under_the_validators_repack():
    """THE MEASUREMENT (2026-10-27): the files Foul Play really draws from (`anchors.json`'s `foulplay.team_dirs`). Its pool
    files with a Hidden Power play another set re-packed (Dark 70 -> the named type); `away`'s one validator-accepted
    file with a Hidden Power (team_23's Gyarados) does too. If this ever reads unchanged, the regime split is vacuous and
    the SOP note must say so."""
    cfg = load_config()
    fp = cfg.foulplay
    if not (Path(fp.python).is_file() and Path(fp.checkout).is_dir()):
        pytest.skip("no Foul Play checkout / interpreter on this box (config foulplay)")
    got = {}
    for teamset in ("home", "away"):
        try:
            d = fp.team_dir(teamset)
        except AnchorConfigError:
            continue
        if not d.exists():
            continue
        got[teamset] = fpk.cell_record(str(fp.python), Path(fp.checkout), d)
    assert got, "no Foul Play team directory exists on this box"
    for teamset, rec in got.items():
        assert rec["frontend_pack_changed"] is True, (teamset, rec["changed"][:2])
