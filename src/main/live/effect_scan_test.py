"""The drift gate on the RUST reader (P6 of the poke-env retirement): its two reader checks have teeth and pass.

* the ENCODER (source) check — every effect line the PINNED Showdown's gen3 source can announce reads and encodes clean
  on the live reader, and a fabricated effect (an unknown ``-start`` / ``-activate`` / ``-singleturn``) is a finding;
* the READ + ENCODER (replayed) check — a real public replay with a player DISCONNECT / REJOIN (``|player|p1|`` with an
  empty name, then the name again) reads clean from both seats (the retired Python scan reported it as a structural
  failure: its spectator ``Gen3Battle`` re-derived the role and put the other side's mons on its team), and the same
  replay with one planted unknown keyword FAILS.
"""
from pathlib import Path

import pytest

from main import ladder_drift_scan
from main.live import effect_scan
from utils.paths import repo_path

pytestmark = pytest.mark.integration

REJOIN = Path(__file__).parent / "testdata" / "player_rejoin.log"


def test_every_effect_line_of_the_pinned_showdown_reads_and_encodes():
    root = repo_path("deps", "pokemon-showdown")
    lines = effect_scan.source_lines(root)
    assert len(lines) >= 90, f"only {len(lines)} derived lines — the text scan lost its sources"
    assert effect_scan.probe_lines(sorted(lines)) == {}
    assert effect_scan.check(str(root)) == 0


def test_a_fabricated_effect_is_a_finding_and_a_real_one_is_not():
    fake = [effect_scan.line_text("-start", "bogusvolatile"), effect_scan.line_text("-activate", "move: Bogus Move"),
            effect_scan.line_text("-singleturn", "Bogus")]
    real = [effect_scan.line_text("-start", "Substitute"), effect_scan.line_text("-activate", "move: Heal Bell")]
    bad = effect_scan.probe_lines(fake + real)
    assert sorted(bad) == sorted(fake), bad
    assert all("UnknownVolatileError" in why for why in bad.values()), bad


def test_the_drift_scan_reads_a_rejoin_replay_clean_and_fails_a_planted_keyword(tmp_path):
    assert any(ln == "|player|p1|" for ln in REJOIN.read_text().splitlines()), "the fixture lost its rejoin line"
    assert ladder_drift_scan.scan([str(REJOIN)]) == 0
    lines = REJOIN.read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("|turn|2"))
    planted = tmp_path / "planted.log"
    planted.write_text("\n".join(lines[:k] + ["|bogusword|p1a: X"] + lines[k:]) + "\n")
    assert ladder_drift_scan.scan([str(planted)]) == 1
