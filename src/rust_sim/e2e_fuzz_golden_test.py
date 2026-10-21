"""`gen_e2e_fuzz.js` REPRODUCES the committed `e2e_fuzz_golden.txt` (and its taxonomy) byte for byte.

The golden's pairing indexes into the sorted valid-team list, so the generator used to drift with `data/teams/` (770 -> 773
-> 813 files; two INIT field counts) and a committed golden nobody could refresh. The fix pins the team list
(`tests/vectors/e2e_fuzz_teams.sha256`, sha256 per file); these tests hold it:

* the pin matches the files on disk (a changed or removed team is a failure here, not a silent shift of every later pair);
* the generator's first battles equal the golden's prefix, header included (`sim`, ~25 s);
* a FULL regeneration equals the golden and the taxonomy (`slow`, ~9 min);
* the golden's trailing Quick Claw LEAD battles carry the turn-0 `quickClawRoll` (5th INIT field) -- the one place it can
  matter, since no pinned team holds a Quick Claw (`tests/e2e_fuzz_test.rs` holds the Rust half: the roll reorders turn 1).

Every assertion fails on the drift it names. A deliberate re-record: `node src/rust_sim/harness/gen_e2e_fuzz.js
--repin-teams`, regenerate, and say why in the commit and the CHANGELOG.
"""
import hashlib
import os
import subprocess

import pytest

from utils.paths import repo_path

VECTORS = repo_path("src", "rust_sim", "tests", "vectors")
GENERATOR = repo_path("src", "rust_sim", "harness", "gen_e2e_fuzz.js")
GOLDEN = VECTORS / "e2e_fuzz_golden.txt"
TAXONOMY = VECTORS / "e2e_fuzz_taxonomy.txt"
MANIFEST = VECTORS / "e2e_fuzz_teams.sha256"
PREFIX_BATTLES = 15


def _pinned() -> list:
    return [ln.split() for ln in MANIFEST.read_text().splitlines() if ln and not ln.startswith("#")]


def _generate(out_dir, battles: int, unfiltered: int, qc_lead_battles: int = 1) -> None:
    if not (repo_path("deps", "pokemon-showdown", "dist").exists() and
            repo_path("deps", "pokemon-showdown", "node_modules").exists()):
        pytest.fail("deps/pokemon-showdown is not built (git submodule update --init; ./scripts/bootstrap.sh --skip-env)")
    env = dict(os.environ, E2E_OUT_DIR=str(out_dir), E2E_FILTERED_TARGET=str(battles), E2E_UNFILTERED=str(unfiltered))
    env["E2E_QC_LEAD_BATTLES"] = str(qc_lead_battles)
    env.pop("E2E_MASTER_SEED", None)
    env.pop("E2E_MAX_TRIES", None)
    env.pop("E2E_TEAMS_DIR", None)
    r = subprocess.run(["node", str(GENERATOR)], env=env, capture_output=True, text=True, timeout=3000)
    assert r.returncode == 0, r.stderr[-2000:]


def test_the_pinned_team_list_is_what_is_on_disk():
    pinned = _pinned()
    assert len(pinned) == 722, "the committed golden's header says filter-clean-teams 722/722"
    assert "filter-clean-teams 722/722" in GOLDEN.read_text().split("\n", 3)[2]
    assert [rel for _, rel in pinned] == sorted(rel for _, rel in pinned), "the pairing indexes the SORTED list"
    for sha, rel in pinned:
        f = repo_path("data", "teams", rel)
        assert f.exists(), f"pinned team {rel} left data/teams/ (the golden cannot be regenerated)"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sha, f"pinned team {rel} changed"


def _golden_cases() -> list:
    """[(scen, p1 lead item, p2 lead item, INIT fields)] in file order."""
    teams, inits = {}, {}
    for ln in GOLDEN.read_text().splitlines():
        f = ln.split("\t")
        if f[0] == "TEAM":
            teams.setdefault(f[1], {})[f[2]] = f[3].split("]")[0].split("|")[2].lower()
        elif f[0] == "INIT":
            inits[f[1]] = f[2:]
    return [(k, teams[k]["p1"], teams[k]["p2"], inits[k]) for k in inits]


def test_the_golden_carries_quick_claw_lead_battles_with_the_turn0_roll():
    """None of the pinned teams holds a Quick Claw (so the 220 random battles cannot exercise the turn-0 `quickClawRoll`):
    the golden's trailing battles are the pinned teams with the p1 LEAD's item set to Quick Claw and the roll TRUE, which
    the Rust test replays and proves the roll reorders turn 1 for. Drop them and the roll is untested again."""
    for _, rel in _pinned():
        assert "quick claw" not in repo_path("data", "teams", rel).read_text().lower(), (
            f"{rel} holds a Quick Claw: the random battles may now exercise the roll; revisit the lead battles")
    cases = _golden_cases()
    qc = [c for c in cases if c[1] == "quickclaw"]
    assert qc, "no golden battle has a Quick Claw lead: the turn-0 roll is untested by the e2e gate"
    assert all(len(c[3]) == 3 and c[3][2] == "1" for c in qc), "a Quick Claw lead battle must carry roll = 1 (5th INIT field)"
    assert all(len(c[3]) == 2 for c in cases if c[1] != "quickclaw"), "only the Quick Claw lead battles carry the 5th field"
    assert cases[-len(qc):] == qc, "the Quick Claw lead battles FOLLOW the random ones (the prefix test depends on it)"


@pytest.mark.sim
@pytest.mark.slow  # ~20 s of Node generation, over the routine tier's 8 s bar
def test_the_generators_first_battles_are_the_goldens_prefix(tmp_path):
    """The deterministic MASTER_SEED pairing makes battle e2e_N the N-th accepted battle: a short run is a byte prefix of
    the golden, header and INIT field count included (`tests/e2e_fuzz_test.rs` asserts 4 INIT fields)."""
    _generate(tmp_path, PREFIX_BATTLES, 0, qc_lead_battles=0)  # the Quick Claw battles FOLLOW the 220 random ones
    got = (tmp_path / "e2e_fuzz_golden.txt").read_text().splitlines()
    want = GOLDEN.read_text().splitlines()
    assert sum(ln.startswith("SCEN\t") for ln in got) == PREFIX_BATTLES
    assert got == want[:len(got)]


@pytest.mark.sim
@pytest.mark.slow
def test_a_full_regeneration_equals_the_golden_and_the_taxonomy_byte_for_byte(tmp_path):
    _generate(tmp_path, 220, 300)
    assert (tmp_path / "e2e_fuzz_golden.txt").read_bytes() == GOLDEN.read_bytes()
    assert (tmp_path / "e2e_fuzz_taxonomy.txt").read_bytes() == TAXONOMY.read_bytes()
