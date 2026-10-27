"""X35 batch 1's gates (``main.x35.batch``, the orchestrator's rulings of 2026-10-19 on the adversarial review and of
2026-10-20 after the GPU battery): AM-1's one pin PER SIDE (AM-8), the side's own required commit (production's
FP32_ALT), the POOL record, AM-2's variance gate G1, AM-9 / AM-3 / AM-4's conditional stages and AM-6's K9(b) arm-end
read. Synthetic files only; every case fails on revert of the rule it names, and every constructed number sits well
clear of its boundary except the deliberate rule-8 cases (standing rule 8)."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest

from main.x35 import batch as B
from main.x35 import kit as K
from utils.paths import repo_path

PIN = "a" * 40
OTHER = "b" * 40


def _run(models: Path, name: str, *hashes: str, final: bool = False) -> Path:
    d = models / name
    d.mkdir(parents=True)
    if hashes:
        meta = {"git_hash": hashes[0], "pin_history": [{"git_hash": h, "pin_source": "pin_commit"} for h in hashes]}
        (d / "metadata.json").write_text(json.dumps(meta))
    if final:
        (d / "final_model.zip").write_bytes(b"z")
    return d


# ------------------------------------------------------------------------------------------------------ AM-1 / AM-8
PROD, TOY = "production", "toy"
FP32 = "FP32_ALT"


def _pin_file(stage: Path, side: str) -> str:
    return B.pin_path(str(stage), side)


def test_the_first_go_of_a_side_writes_its_pin_once_and_every_later_launch_of_it_must_equal_it(tmp_path):
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    models.mkdir()
    ok, lines = B.pin_check(PIN, str(stage), str(models), TOY)
    assert ok and "the first toy GO writes" in lines[0]
    with pytest.raises(B.BatchError, match="needs --t62 <NA_TOY>"):
        B.pin_write_if_absent(PIN, str(stage), TOY, None)
    assert "WRITTEN" in B.pin_write_if_absent(PIN, str(stage), TOY, "NA_TOY")
    rec = B.read_pin_file(_pin_file(stage, TOY), TOY)
    assert rec is not None and rec["sha"] == PIN and rec["t62_verdict"] == "NA_TOY" and rec["side"] == TOY
    assert Path(_pin_file(stage, TOY)).name == "x35_pin_batch1_toy.txt"
    assert B.pin_check(PIN, str(stage), str(models), TOY)[0]
    ok, lines = B.pin_check(OTHER, str(stage), str(models), TOY)          # a launch at a LATER pin
    assert not ok and "!= X35_PIN_1_toy" in lines[0]
    with pytest.raises(B.BatchError, match="written ONCE"):
        B.write_pin_file(_pin_file(stage, TOY), PIN, "NA_TOY", TOY)
    with pytest.raises(B.BatchError, match="!= X35_PIN_1_toy"):
        B.pin_write_if_absent(OTHER, str(stage), TOY, "NA_TOY")


def test_am8_each_side_has_its_own_pin_and_the_production_pin_may_differ_from_the_toys(tmp_path):
    """The orchestrator's ruling (2026-10-20): ONE PIN PER SIDE. The toy pin being written must not bind production (it
    would, under the old single x35_pin_batch1.txt), and the production pin being a different sha is the point."""
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    models.mkdir()
    B.pin_write_if_absent(PIN, str(stage), TOY, "NA_TOY")
    ok, lines = B.pin_check(OTHER, str(stage), str(models), PROD)         # production: no record of its own yet
    assert ok and "the first production GO writes" in lines[0], lines
    assert "WRITTEN" in B.pin_write_if_absent(OTHER, str(stage), PROD, FP32)
    assert Path(_pin_file(stage, PROD)).name == "x35_pin_batch1_production.txt"
    assert B.pin_check(OTHER, str(stage), str(models), PROD)[0] and B.pin_check(PIN, str(stage), str(models), TOY)[0]
    ok, lines = B.pin_check(PIN, str(stage), str(models), PROD)           # the toy's pin is not production's
    assert not ok and "!= X35_PIN_1_production" in lines[0]
    ok, lines = B.pin_check(OTHER, str(stage), str(models), TOY)
    assert not ok and "!= X35_PIN_1_toy" in lines[0]


def test_am8_the_t62_token_is_the_sides_own(tmp_path):
    """Production: FP32_ALT (the battery DECIDED it; ADOPT / PENDING / INCONCLUSIVE no longer open a production GO).
    The toy: NA_TOY (no tau solve). A token of the other side, or an old one, is a refusal at write AND at the check."""
    stage = tmp_path / "stage"
    stage.mkdir()
    for side, bad in ((PROD, "NA_TOY"), (PROD, "ADOPT"), (PROD, "PENDING"), (TOY, FP32), (TOY, "ADOPT")):
        with pytest.raises(B.BatchError, match="is not a"):
            B.pin_write_if_absent(PIN, str(stage), side, bad)
        ok, lines = B.pin_check(PIN, str(stage), str(tmp_path / "m"), side, t62=bad)
        assert not ok and "is not a" in lines[0], (side, bad, lines)
    assert B.pin_check(PIN, str(stage), str(tmp_path / "m"), PROD, t62=FP32)[0]
    assert B.pin_check(PIN, str(stage), str(tmp_path / "m"), TOY, t62="NA_TOY")[0]
    assert not list(stage.iterdir())                                      # nothing was written by a refusal
    assert set(B.T62_VERDICTS) == {FP32, "NA_TOY"}


def test_a_pin_file_that_names_the_other_side_or_a_wrong_token_is_a_refusal(tmp_path):
    f = tmp_path / "x35_pin_batch1_production.txt"
    f.write_text(f"sha: {PIN}\nside: toy\nt62_verdict: NA_TOY\n")
    with pytest.raises(B.BatchError, match="says side 'toy'"):
        B.read_pin_file(str(f), PROD)
    f.write_text(f"sha: {PIN}\nside: production\nt62_verdict: ADOPT\n")
    with pytest.raises(B.BatchError, match="not a production token"):
        B.read_pin_file(str(f), PROD)
    with pytest.raises(B.BatchError, match="not one of"):
        B.pin_path(str(tmp_path), "both")


def test_every_batch1_runs_recorded_pin_must_equal_its_sides_pin(tmp_path):
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    _run(models, "rb_x35_a1arch_s3501", PIN)
    assert B.pin_check(PIN, str(stage), str(models), PROD)[0]
    _run(models, "rb_x35_c0_s3501", PIN, OTHER)                          # a restart that moved its pin
    ok, lines = B.pin_check(PIN, str(stage), str(models), PROD)
    assert not ok and any("rb_x35_c0_s3501" in x and "bbbbbbbbbbbb" in x for x in lines), lines


def test_the_other_sides_runs_are_listed_but_not_judged_by_this_sides_pin(tmp_path):
    """Toy runs at the toy pin, production launching at another pin: production's check must pass (under the single
    batch pin it would HOLD on every toy run), and each side still HOLDs on its OWN run recorded at the wrong pin."""
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    _run(models, "rb_x35t_c0_s3601", PIN)                                 # a toy run, at the toy pin
    _run(models, "rb_x35t_x47d4_s3601", PIN)
    ok, lines = B.pin_check(OTHER, str(stage), str(models), PROD)
    assert ok and any("judged at its own launches" in x and "rb_x35t_c0_s3601" in x for x in lines), lines
    ok, lines = B.pin_check(OTHER, str(stage), str(models), TOY)          # before any toy record: runs vs this launch's pin
    assert not ok and any("rb_x35t_c0_s3601" in x and "aaaaaaaaaaaa" in x for x in lines), lines
    B.pin_write_if_absent(PIN, str(stage), TOY, "NA_TOY")
    ok, lines = B.pin_check(PIN, str(stage), str(models), TOY)
    assert ok and any("2 side-pinned toy run(s) checked" in x for x in lines), lines
    _run(models, "rb_x35t_c0_s3602", OTHER)                               # a toy run on the wrong pin
    ok, lines = B.pin_check(PIN, str(stage), str(models), TOY)
    assert not ok and any("rb_x35t_c0_s3602" in x for x in lines), lines
    assert B.pin_check(OTHER, str(stage), str(models), PROD)[0]           # production does not care


def test_a_batch1_run_without_a_recorded_pin_holds_and_other_batches_are_not_judged(tmp_path):
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    _run(models, "rb_x35_c0_s3502")                                       # no metadata.json
    _run(models, "rb_x35_a3mx_s3501", OTHER)                              # a later batch's run (its own pin)
    ok, lines = B.pin_check(PIN, str(stage), str(models), PROD)
    assert not ok and any("no recorded pin" in x for x in lines)
    assert any("not a side-pinned arm" in x and "rb_x35_a3mx_s3501" in x for x in lines)
    (models / "rb_x35_c0_s3502").rename(models / "rb_x35_c0_s3502.POOLSEED_FAIL")   # moved aside: still batch 1
    ok, lines = B.pin_check(PIN, str(stage), str(models), PROD)
    assert not ok and any("POOLSEED_FAIL" in x for x in lines)


def test_a_corrupt_pin_file_is_a_refusal_not_no_pin(tmp_path):
    (tmp_path / "x35_pin_batch1_toy.txt").write_text("sha: abc\n")
    with pytest.raises(B.BatchError, match="40-hex"):
        B.pin_check(PIN, str(tmp_path), str(tmp_path / "models"), TOY)



# ---------------------------------------------- AM-D1-FIX (2026-10-22): D1's own corpus-only pin, the one AM-8 exception
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, files: dict, msg: str) -> str:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def d1_repo(tmp_path, monkeypatch):
    """A repo: the side pin ``base``; ``pin`` = base + one corpus-only commit; ``wide`` = base + a commit that also
    touches the trainer; ``two`` = pin + another corpus commit. B's D1 constants point at ``pin``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    base = _commit(repo, {"src/main/train.py": "x = 1\n", "src/utils/team_corpora/a.txt": "1982\n"}, "base")
    pin = _commit(repo, {"src/utils/team_corpora/a.txt": "1981\n"}, "guard")
    two = _commit(repo, {"src/utils/team_corpora/b.txt": "more\n"}, "more")
    _git(repo, "checkout", "-q", base)
    wide = _commit(repo, {"src/utils/team_corpora/a.txt": "1981\n", "src/main/train.py": "x = 2\n"}, "wide")
    monkeypatch.setattr(B, "D1_PIN", pin)
    monkeypatch.setattr(B, "D1_PIN_BASE", base)
    monkeypatch.setattr(B, "ARM_PINS", {"d1": pin})
    monkeypatch.setattr(B, "D1_CORPORA_TREE", _git(repo, "rev-parse", f"{pin}:src/utils/team_corpora"))
    return {"repo": str(repo), "base": base, "pin": pin, "wide": wide, "two": two}


def test_d1_pin_check_accepts_only_the_side_pin_plus_one_corpus_only_commit(d1_repo):
    r = d1_repo
    ok, lines = B.d1_pin_check(r["pin"], r["base"], r["repo"])
    assert ok and "one commit, 1 file(s)" in lines[0], lines
    for pin, side_pin, why in ((r["base"], r["base"], "launches only at D1_PIN"),       # d1 at the side pin
                               (r["pin"], r["two"], "sits on"),                        # another side pin
                               ):
        ok, lines = B.d1_pin_check(pin, side_pin, r["repo"])
        assert not ok and why in lines[0], (why, lines)
    B.D1_PIN = r["wide"]                                                               # a diff outside the corpus
    ok, lines = B.d1_pin_check(r["wide"], r["base"], r["repo"])
    assert not ok and "outside: ['src/main/train.py']" in lines[0], lines
    B.D1_PIN = r["two"]                                                                # two commits on top
    ok, lines = B.d1_pin_check(r["two"], r["base"], r["repo"])
    assert not ok and "parent(s)" in lines[0], lines


def test_d1_pin_check_refuses_a_corpus_tree_other_than_the_guard_commits(d1_repo, monkeypatch):
    monkeypatch.setattr(B, "D1_CORPORA_TREE", "0" * 40)
    ok, lines = B.d1_pin_check(d1_repo["pin"], d1_repo["base"], d1_repo["repo"])
    assert not ok and "tree" in lines[0] and "guard commit's" in lines[0], lines


def test_pin_check_holds_d1_to_its_own_pin_and_every_other_arm_to_the_sides(d1_repo, tmp_path):
    r = d1_repo
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    models.mkdir()
    ok, lines = B.pin_check(r["pin"], str(stage), str(models), PROD, arm="d1", repo=r["repo"])
    assert not ok and "no x35_pin_batch1_production.txt is written yet" in lines[0], lines
    B.pin_write_if_absent(r["base"], str(stage), PROD, FP32)
    assert B.pin_check(r["pin"], str(stage), str(models), PROD, arm="d1", repo=r["repo"])[0]
    ok, lines = B.pin_check(r["base"], str(stage), str(models), PROD, arm="d1", repo=r["repo"])
    assert not ok and "d1 launches only at D1_PIN" in lines[0], lines                   # d1 at the side pin: HOLD
    for arm in ("c0", "sprt", None):                                                 # nobody else at D1's pin
        ok, lines = B.pin_check(r["pin"], str(stage), str(models), PROD, arm=arm, repo=r["repo"])
        assert not ok and "!= X35_PIN_1_production" in lines[0], (arm, lines)
        assert B.pin_check(r["base"], str(stage), str(models), PROD, arm=arm, repo=r["repo"])[0]
    # the runs: d1 at the side pin is the listed PRE-FIX attempt, d1 at its pin passes, c0 at D1's pin HOLDs
    _run(models, "rb_x35_d1wide_s3501.PREFIX_LABEL_CRASH", r["base"])
    _run(models, "rb_x35_d1wide_s3501", r["pin"])
    ok, lines = B.pin_check(r["base"], str(stage), str(models), PROD, arm="c0", repo=r["repo"])
    assert ok and any("PRE-FIX d1 attempt" in x and "PREFIX_LABEL_CRASH" in x for x in lines), lines
    _run(models, "rb_x35_d1wide_s3502", r["two"])
    _run(models, "rb_x35_c0_s3501", r["pin"])
    ok, lines = B.pin_check(r["base"], str(stage), str(models), PROD, arm="c0", repo=r["repo"])
    assert not ok and any(x.startswith("rb_x35_d1wide_s3502:") for x in lines) and \
        any(x.startswith("rb_x35_c0_s3501:") for x in lines), lines


def test_pin_write_never_writes_for_d1_and_refuses_a_wrong_d1_pin(d1_repo, tmp_path):
    r = d1_repo
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(B.BatchError, match="AM-D1-FIX"):
        B.pin_write_if_absent(r["pin"], str(stage), PROD, FP32, arm="d1")             # no side record to rest on
    assert not list(stage.iterdir())
    B.pin_write_if_absent(r["base"], str(stage), PROD, FP32)
    before = Path(_pin_file(stage, PROD)).read_text()
    assert "unchanged" in B.pin_write_if_absent(r["pin"], str(stage), PROD, None, arm="d1")
    with pytest.raises(B.BatchError, match="AM-D1-FIX"):
        B.pin_write_if_absent(r["base"], str(stage), PROD, None, arm="d1")
    assert Path(_pin_file(stage, PROD)).read_text() == before


def test_the_registered_d1_pin_is_reachable_and_carries_a_main_commits_corpus_tree():
    """The real D1_PIN (local branch ``x35-d1-pin``) passes its own check against the production pin, and its corpus
    tree is the one a commit on this checkout's history holds (the shipped guard commit)."""
    from utils.paths import repo_root

    root = str(repo_root())
    ok, lines = B.d1_pin_check(B.D1_PIN, B.D1_PIN_BASE, root)
    assert ok, lines
    git = ["git", "-C", root]
    shas = subprocess.run(git + ["log", "--format=%H", "HEAD", "--", "src/utils/team_corpora"], check=True,
                          capture_output=True, text=True).stdout.split()
    trees = {subprocess.run(git + ["rev-parse", f"{c}:src/utils/team_corpora"], capture_output=True,
                            text=True).stdout.strip() for c in shas}
    assert B.D1_CORPORA_TREE in trees

# ------------------------------------------- AM-D3 (2026-10-25): D3's own corpus-only pin, checked as D1's is
@pytest.fixture
def d3_repo(d1_repo, monkeypatch):
    """D1's fixture repo, read as D3's: ``pin`` is D3's lever commit on the side pin."""
    r = d1_repo
    monkeypatch.setattr(B, "D3_PIN", r["pin"])
    monkeypatch.setattr(B, "D3_PIN_BASE", r["base"])
    monkeypatch.setattr(B, "D3_CORPORA_TREE", B.D1_CORPORA_TREE)
    monkeypatch.setattr(B, "ARM_PINS", {"d3": r["pin"]})
    return r


def test_d3_pin_check_accepts_only_the_side_pin_plus_one_corpus_only_commit(d3_repo, monkeypatch):
    r = d3_repo
    ok, lines = B.d3_pin_check(r["pin"], r["base"], r["repo"])
    assert ok and lines[0].startswith("AM-D3: D3's pin") and "lever commit's" in lines[0], lines
    ok, lines = B.d3_pin_check(r["base"], r["base"], r["repo"])
    assert not ok and "d3 launches only at D3_PIN" in lines[0], lines
    monkeypatch.setattr(B, "D3_PIN", r["wide"])
    ok, lines = B.d3_pin_check(r["wide"], r["base"], r["repo"])
    assert not ok and "outside: ['src/main/train.py']" in lines[0], lines
    monkeypatch.setattr(B, "D3_PIN", r["pin"])
    monkeypatch.setattr(B, "D3_CORPORA_TREE", "0" * 40)
    ok, lines = B.d3_pin_check(r["pin"], r["base"], r["repo"])
    assert not ok and "lever commit's" in lines[0], lines


def test_pin_check_holds_d3_to_its_own_pin_and_a_d3_run_at_the_side_pin_is_refused(d3_repo, tmp_path):
    r = d3_repo
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    models.mkdir()
    B.pin_write_if_absent(r["base"], str(stage), PROD, FP32)
    assert B.pin_check(r["pin"], str(stage), str(models), PROD, arm="d3", repo=r["repo"])[0]
    ok, lines = B.pin_check(r["base"], str(stage), str(models), PROD, arm="d3", repo=r["repo"])
    assert not ok and "d3 launches only at D3_PIN" in lines[0], lines
    assert "unchanged" in B.pin_write_if_absent(r["pin"], str(stage), PROD, None, arm="d3")
    with pytest.raises(B.BatchError, match="AM-D3"):
        B.pin_write_if_absent(r["base"], str(stage), PROD, None, arm="d3")
    _run(models, "rb_x35_d3mut_s3501", r["pin"])
    assert B.pin_check(r["base"], str(stage), str(models), PROD, arm="c0", repo=r["repo"])[0]
    _run(models, "rb_x35_d3mut_s3502", r["base"])                 # no pre-fix attempt exists for D3: refused
    ok, lines = B.pin_check(r["base"], str(stage), str(models), PROD, arm="c0", repo=r["repo"])
    assert not ok and any(x.startswith("rb_x35_d3mut_s3502:") for x in lines), lines


def test_d3_stages_look1_on_the_side_gate_and_look2_only_on_its_own_continue(tmp_path):
    stage = tmp_path / "stage"
    (stage / B.READS_DIR).mkdir(parents=True)
    st = str(stage)
    assert B.stage_status(st, "d3", 3501)[0] == B.PENDING                        # AM-8: no FP32_ALT named
    _fill_fp32(stage)
    for seed in (3501, 3502, 3503):
        status, why = B.stage_status(st, "d3", seed)
        assert status == B.ALLOWED and "AM-D3" in why and B.stage_of("d3", seed) == "D3", why
    assert B.stage_status(st, "d3", 3504)[0] == B.PENDING and B.stage_of("d3", 3504) == "D3b"
    rec = {"arm": "X41", "look": 1, "verdict": "CONTINUE", "h": {"final": False, "class": "UNDECIDED"}}
    (stage / B.READS_DIR / "d3_look1.json").write_text(json.dumps(rec))
    assert B.stage_status(st, "d3", 3505)[0] == B.ALLOWED
    (stage / B.READS_DIR / "d3_look1.json").write_text(json.dumps(dict(rec, verdict="NULL")))
    assert B.stage_status(st, "d3", 3504)[0] == B.REFUSED
    (stage / B.READS_DIR / "d3_look1.json").write_text(json.dumps(dict(rec, arm="X38")))
    with pytest.raises(B.BatchError, match="not X41"):
        B.stage_status(st, "d3", 3504)


def test_the_registered_d3_pin_is_reachable_and_carries_a_main_commits_corpus_tree():
    """The real D3_PIN (tag ``x35-d3-pin-<sha8>`` on origin) passes its own check against the production pin, and its
    corpus tree is the one a commit on this checkout's history holds (the shipped lever commit)."""
    from utils.paths import repo_root

    root = str(repo_root())
    ok, lines = B.d3_pin_check(B.D3_PIN, B.D3_PIN_BASE, root)
    assert ok, lines
    git = ["git", "-C", root]
    shas = subprocess.run(git + ["log", "--format=%H", "HEAD", "--", "src/utils/team_corpora"], check=True,
                          capture_output=True, text=True).stdout.split()
    trees = {subprocess.run(git + ["rev-parse", f"{c}:src/utils/team_corpora"], capture_output=True,
                            text=True).stdout.strip() for c in shas}
    assert B.D3_CORPORA_TREE in trees


# ------------------------------------------- AM-N1P (2026-10-26): N1-P's own corpus-only pin, checked as D3's is
@pytest.fixture
def n1p_repo(d1_repo, monkeypatch):
    """D1's fixture repo, read as N1-P's: ``pin`` is the corpus-only commit on the side pin."""
    r = d1_repo
    monkeypatch.setattr(B, "N1P_PIN", r["pin"])
    monkeypatch.setattr(B, "N1P_PIN_BASE", r["base"])
    monkeypatch.setattr(B, "N1P_CORPORA_TREE", B.D1_CORPORA_TREE)
    monkeypatch.setattr(B, "ARM_PINS", {"n1p": r["pin"]})
    return r


def test_n1p_pin_check_accepts_only_the_side_pin_plus_one_corpus_only_commit(n1p_repo, monkeypatch):
    r = n1p_repo
    ok, lines = B.n1p_pin_check(r["pin"], r["base"], r["repo"])
    assert ok and lines[0].startswith("AM-N1P: N1P's pin") and "registered N1-P tree" in lines[0], lines
    ok, lines = B.n1p_pin_check(r["base"], r["base"], r["repo"])
    assert not ok and "n1p launches only at N1P_PIN" in lines[0], lines
    monkeypatch.setattr(B, "N1P_PIN", r["wide"])
    ok, lines = B.n1p_pin_check(r["wide"], r["base"], r["repo"])
    assert not ok and "outside: ['src/main/train.py']" in lines[0], lines
    monkeypatch.setattr(B, "N1P_PIN", r["two"])
    ok, lines = B.n1p_pin_check(r["two"], r["base"], r["repo"])
    assert not ok and "parent(s)" in lines[0], lines
    monkeypatch.setattr(B, "N1P_PIN", r["pin"])
    monkeypatch.setattr(B, "N1P_CORPORA_TREE", "0" * 40)
    ok, lines = B.n1p_pin_check(r["pin"], r["base"], r["repo"])
    assert not ok and "registered N1-P tree" in lines[0], lines


def test_pin_check_holds_n1p_to_its_own_pin_and_an_n1p_run_at_the_side_pin_is_refused(n1p_repo, tmp_path):
    r = n1p_repo
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    models.mkdir()
    B.pin_write_if_absent(r["base"], str(stage), PROD, FP32)
    assert B.pin_check(r["pin"], str(stage), str(models), PROD, arm="n1p", repo=r["repo"])[0]
    ok, lines = B.pin_check(r["base"], str(stage), str(models), PROD, arm="n1p", repo=r["repo"])
    assert not ok and "n1p launches only at N1P_PIN" in lines[0], lines
    assert "unchanged" in B.pin_write_if_absent(r["pin"], str(stage), PROD, None, arm="n1p")
    with pytest.raises(B.BatchError, match="AM-N1P"):
        B.pin_write_if_absent(r["base"], str(stage), PROD, None, arm="n1p")
    _run(models, "rb_x35_n1p_s3501", r["pin"])
    assert B.pin_check(r["base"], str(stage), str(models), PROD, arm="c0", repo=r["repo"])[0]
    _run(models, "rb_x35_n1p_s3502", r["base"])                   # no pre-fix attempt exists for N1-P: refused
    ok, lines = B.pin_check(r["base"], str(stage), str(models), PROD, arm="c0", repo=r["repo"])
    assert not ok and any(x.startswith("rb_x35_n1p_s3502:") for x in lines), lines


def test_n1p_stages_look1_on_the_side_gate_and_look2_only_on_its_own_continue(tmp_path):
    stage = tmp_path / "stage"
    (stage / B.READS_DIR).mkdir(parents=True)
    st = str(stage)
    assert B.stage_status(st, "n1p", 3501)[0] == B.PENDING                       # AM-8: no FP32_ALT named
    _fill_fp32(stage)
    for seed in (3501, 3502, 3503):
        status, why = B.stage_status(st, "n1p", seed)
        assert status == B.ALLOWED and "AM-N1P" in why and B.stage_of("n1p", seed) == "N1P", why
    assert B.stage_status(st, "n1p", 3504)[0] == B.PENDING and B.stage_of("n1p", 3504) == "N1Pb"
    rec = {"arm": "X61P", "look": 1, "verdict": "CONTINUE", "h": {"final": False, "class": "UNDECIDED"}}
    (stage / B.READS_DIR / "n1p_look1.json").write_text(json.dumps(rec))
    assert B.stage_status(st, "n1p", 3505)[0] == B.ALLOWED
    (stage / B.READS_DIR / "n1p_look1.json").write_text(json.dumps(dict(rec, verdict="NULL")))
    assert B.stage_status(st, "n1p", 3504)[0] == B.REFUSED
    (stage / B.READS_DIR / "n1p_look1.json").write_text(json.dumps(dict(rec, arm="X41")))
    with pytest.raises(B.BatchError, match="not X61P"):
        B.stage_status(st, "n1p", 3504)


def test_the_registered_n1p_pin_adds_only_the_corpus_files_main_landed_and_its_registry_line():
    """The real N1P_PIN (tag ``x61p-n1p-pin-<sha8>`` on origin) passes its own check against the production pin; its
    ``n1_proc_v1`` subtree IS the one main's history holds (the de269480 bytes the soak read), and its
    ``__init__.py`` differs from the side pin's by the one ``CORPORA`` entry, nothing else (AM-N1P: no builder, no D3
    corpus, no label-guard or GIGO-fix code)."""
    from utils.paths import repo_root

    root = str(repo_root())
    ok, lines = B.n1p_pin_check(B.N1P_PIN, B.N1P_PIN_BASE, root)
    assert ok, lines

    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", root, *a], check=True, capture_output=True, text=True).stdout

    assert git("rev-parse", f"{B.N1P_PIN}:src/utils/team_corpora").strip() == B.N1P_CORPORA_TREE
    assert sorted(git("diff", "--name-only", B.N1P_PIN_BASE, B.N1P_PIN).split()) == [
        "src/utils/team_corpora/__init__.py", "src/utils/team_corpora/n1_proc_v1/manifest.json",
        "src/utils/team_corpora/n1_proc_v1/soak_selfcheck.json", "src/utils/team_corpora/n1_proc_v1/teams.jsonl.gz"]
    sub = "src/utils/team_corpora/n1_proc_v1"
    assert git("rev-parse", f"{B.N1P_PIN}:{sub}").strip() in {
        git("rev-parse", f"{c}:{sub}").strip()
        for c in git("log", "--format=%H", "HEAD", "--", sub).split()}
    diff = [ln for ln in git("diff", "-U0", B.N1P_PIN_BASE, B.N1P_PIN, "--",
                             "src/utils/team_corpora/__init__.py").splitlines()
            if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    assert diff == ['-CORPORA: Dict[str, Path] = {"w_forum_v1": HERE / "w_forum_v1"}',
                    '+CORPORA: Dict[str, Path] = {"w_forum_v1": HERE / "w_forum_v1", "n1_proc_v1": HERE / "n1_proc_v1"}']


# ------------------------------------------------------------------ AM-8: production's FP32_ALT commit, a placeholder
def _side_file(stage: Path, line: str) -> Path:
    f = stage / B.SIDE_REQUIRED_FILE.format(side=PROD)
    f.write_text("# header\n\n" + line + "\n")
    return f


def _fill_fp32(stage: Path, sha: str = "c" * 40) -> None:
    """The production side file with FP32_ALT NAMED (hex), as the follow-up commit will write it."""
    _side_file(stage, f"{sha} {FP32} the fp32 build")


def _git_repo(tmp_path: Path):
    """A throwaway repo: base <- fp32 <- head, and ``side`` branching off base (no fp32). ``(repo, {name: sha})``."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*a: str) -> str:
        r = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()

    git("init", "-q", "-b", "main")
    shas = {}
    for name in ("base", "fp32", "head"):
        (repo / name).write_text(name)
        git("add", name)
        git("commit", "-q", "-m", name)
        shas[name] = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "side", shas["base"])
    (repo / "side").write_text("side")
    git("add", "side")
    git("commit", "-q", "-m", "side")
    shas["side"] = git("rev-parse", "HEAD")
    return repo, shas


def test_am8_production_is_refused_while_the_fp32_alt_sha_is_a_placeholder(tmp_path):
    repo, shas = _git_repo(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    shipped = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17/"
                        + B.SIDE_REQUIRED_FILE.format(side=PROD))
    assert shipped.is_file()
    # the file as staged before 2026-10-21: the shipped one with its FP32_ALT sha put back to the placeholder
    text = "\n".join(("FP32_ALT_SHA_PENDING" + ln[40:]) if ln.split(" ", 2)[1:2] == [FP32] else ln
                     for ln in shipped.read_text().splitlines())
    (stage / shipped.name).write_text(text + "\n")
    rows = B.read_side_required(str(stage), PROD)
    assert [n for _, n in rows] == [FP32] and not B._is_commit_sha(rows[0][0])
    ok, lines = B.required_check(str(stage), PROD, shas["head"], str(repo))
    assert not ok and "no commit recorded yet" in lines[0], lines
    filled, why = B.side_gate(str(stage), PROD)
    assert not filled and "not recorded yet" in why
    assert B.stage_status(str(stage), "a1", 3501)[0] == B.PENDING        # every production fork waits
    assert B.stage_status(str(stage), "d1", 3501)[0] == B.PENDING
    assert B.stage_status(str(stage), "toy_c0", 3601)[0] == B.ALLOWED    # the toy side has no such condition
    assert K.main(["required-check", "--side", PROD, "--pin", shas["head"], "--stage", str(stage),
                   "--repo", str(repo)]) == 1


def test_am8_production_needs_the_fp32_alt_commit_in_its_pin(tmp_path):
    repo, shas = _git_repo(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    _fill_fp32(stage, shas["fp32"])
    ok, lines = B.required_check(str(stage), PROD, shas["head"], str(repo))
    assert ok and "is in the pin" in lines[0], lines
    assert B.required_check(str(stage), PROD, shas["fp32"], str(repo))[0]  # a pin AT the commit carries it
    ok, lines = B.required_check(str(stage), PROD, shas["side"], str(repo))  # a pin that branched before the build
    assert not ok and "lacks it" in lines[0], lines
    _fill_fp32(stage, "d" * 40)                                           # a sha that is no commit of the repo
    ok, lines = B.required_check(str(stage), PROD, shas["head"], str(repo))
    assert not ok and "does not name a commit" in lines[0], lines
    _fill_fp32(stage, shas["fp32"][:12])                                  # a short sha resolves too
    assert B.required_check(str(stage), PROD, shas["head"], str(repo))[0]
    assert B.side_gate(str(stage), PROD)[0] and B.stage_status(str(stage), "a1", 3501)[0] == B.ALLOWED
    _fill_fp32(stage, shas["fp32"])
    assert K.main(["required-check", "--side", PROD, "--pin", shas["head"], "--stage", str(stage),
                   "--repo", str(repo)]) == 0


def test_am8_deleting_the_production_file_or_its_line_is_never_a_way_past_the_check(tmp_path):
    repo, shas = _git_repo(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(B.BatchError, match="is missing"):
        B.required_check(str(stage), PROD, shas["head"], str(repo))
    assert not B.side_gate(str(stage), PROD)[0] and B.stage_status(str(stage), "c0", 3501)[0] == B.PENDING
    _side_file(stage, f"{shas['fp32']} OTHER_COMMIT some other commit")
    with pytest.raises(B.BatchError, match="names no"):
        B.required_check(str(stage), PROD, shas["head"], str(repo))
    assert K.main(["required-check", "--side", PROD, "--pin", shas["head"], "--stage", str(stage),
                   "--repo", str(repo)]) == 2                              # a typed refusal, not a pass
    assert B.required_check(str(stage), TOY, shas["head"], str(repo)) == (True, ["no toy-only required commits"])


# ------------------------------------------------------------------------------------------------------ POOL
def _parent(tmp_path: Path, steps, fork: int = 280) -> Path:
    p = tmp_path / "parent"
    (p / "snapshots").mkdir(parents=True)
    for s in steps:
        (p / "snapshots" / f"snapshot_{s:012d}.zip").write_bytes(f"s{s}".encode())
    (p / "snapshots" / "summary.json").write_text("{}")
    return p


def _call(step: int = 280, sha: str = "shaP"):
    return {"pstar": {"path": "x/checkpoints/c.zip", "step": step, "sha256": sha}}


def test_the_pool_record_is_t54s_filtered_set_and_a_moved_set_holds(tmp_path):
    parent = _parent(tmp_path, [164, 200, 268, 286, 376])
    e = B.expected_pool(str(parent), 280)
    assert [s["name"] for s in e["snapshots"]] == [f"snapshot_{s:012d}.zip" for s in (164, 200, 268)]
    stage = tmp_path / "stage"
    stage.mkdir()
    ok, line = B.pool_check(str(stage), "production", _call(), str(parent))
    assert ok and "the first production GO records 3" in line and not list(stage.iterdir())   # a dry run writes nothing
    assert B.pool_check(str(stage), "production", _call(), str(parent), write=True)[0]
    assert B.recorded_pool_names(str(stage), "production") == [s["name"] for s in e["snapshots"]]
    assert B.pool_check(str(stage), "production", _call(), str(parent))[0]
    (parent / "snapshots" / "snapshot_000000000164.zip").unlink()       # the parent's pool evicted a kept node
    ok, line = B.pool_check(str(stage), "production", _call(), str(parent))
    assert not ok and "gone ['snapshot_000000000164.zip']" in line
    (parent / "snapshots" / "snapshot_000000000164.zip").write_bytes(b"different bytes")
    ok, line = B.pool_check(str(stage), "production", _call(), str(parent))
    assert not ok and "changed ['snapshot_000000000164.zip']" in line
    ok, line = B.pool_check(str(stage), "production", _call(sha="other"), str(parent))
    assert not ok and "start node" in line


def test_the_pool_metadata_is_recorded_but_never_compared(tmp_path):
    parent = _parent(tmp_path, [164, 200])
    stage = tmp_path / "stage"
    stage.mkdir()
    B.pool_check(str(stage), "toy", _call(), str(parent), write=True)
    (parent / "snapshots" / "summary.json").write_text('{"win_rate_vs_bots": 0.9}')   # F31: the parent's scalars move
    assert B.pool_check(str(stage), "toy", _call(), str(parent))[0]


def test_the_post_spawn_check_compares_the_seeded_names_with_the_record(tmp_path):
    run = tmp_path / "rb_x35_c0_s3501"
    (run / "snapshots").mkdir(parents=True)
    rec = {"schema": 2, "fork_step": 280, "snapshots": ["snapshot_000000000164.zip"], "dropped_steps": []}
    (run / "snapshots" / "pool_seed.json").write_text(json.dumps(rec))
    line = "[step filter <= 280: none later]"
    assert K.check_pool_seed("c0", str(run), _call(), line, 280, ["snapshot_000000000164.zip"]) == ("PASS", [])
    st, probs = K.check_pool_seed("c0", str(run), _call(), line, 280,
                                  ["snapshot_000000000164.zip", "snapshot_000000000200.zip"])
    assert st == "FAIL" and any("missing ['snapshot_000000000200.zip']" in p for p in probs)


# ------------------------------------------------------------------------------------------------------ AM-2
def _g1(a1, c0):
    return {"a1": dict(zip(("3501", "3502", "3503"), a1)), "c0": dict(zip(("3501", "3502", "3503", "3504", "3505"), c0))}


def test_g1_is_the_pooled_within_arm_sd_with_4_df():
    g = B.gate_g1(_g1([60.0, 61.0, 62.0], [55.0, 57.0, 59.0]))    # SS = 2 + 8 = 10 over 4 df: sigma = sqrt(2.5)
    assert g.df == 4 and math.isclose(g.sigma_pp, math.sqrt(2.5), rel_tol=0, abs_tol=1e-12)
    assert g.verdict == B.HOLD                                       # 1.58 > 1.2
    g = B.gate_g1(_g1([60.0, 60.5, 61.0], [55.0, 55.4, 55.8]))     # SS = 0.5 + 0.32 = 0.82: sigma = 0.453
    assert g.verdict == B.PASS and g.reestimate is None


def test_g1_rule8_a_sigma_within_1e9_of_the_bar_counts_as_above():
    # one arm (x - h, x, x + h) contributes 2h^2; the other constant: sigma^2 = 2h^2 / 4  =>  h = 1.2 * sqrt(2)
    h = 1.2 * math.sqrt(2.0)
    g = B.gate_g1(_g1([50.0 - h, 50.0, 50.0 + h], [40.0, 40.0, 40.0]))
    assert abs(g.sigma_pp - 1.2) <= 1e-9 and g.near and g.verdict == B.HOLD
    h2 = 1.19 * math.sqrt(2.0)                                        # clearly below: PASS
    assert B.gate_g1(_g1([50.0 - h2, 50.0, 50.0 + h2], [40.0, 40.0, 40.0])).verdict == B.PASS


def test_g1_reports_the_6_df_reestimate_and_refuses_a_bad_input(tmp_path):
    g = B.gate_g1(_g1([60.0, 60.5, 61.0], [55.0, 55.4, 55.8, 56.0, 54.0]))
    assert g.reestimate is not None and g.reestimate["df"] == 6 and g.reestimate["reported_only"]
    with pytest.raises(B.BatchError, match="no S for seed"):
        B.gate_g1({"a1": {"3501": 1.0, "3502": 2.0}, "c0": {"3501": 1.0, "3502": 2.0, "3503": 3.0}})
    with pytest.raises(B.BatchError, match="not a finite"):
        B.gate_g1(_g1([60.0, float("nan"), 61.0], [55.0, 55.4, 55.8]))
    with pytest.raises(B.BatchError, match="look 2 is not an input"):
        B.gate_g1({"a1": {"3501": 1.0, "3502": 2.0, "3503": 3.0, "3504": 4.0},
                   "c0": {"3501": 1.0, "3502": 2.0, "3503": 3.0}})
    p = B.write_g1(str(tmp_path), g, {"x": 1})
    assert B.read_g1(str(tmp_path)) == B.PASS and Path(p).is_file()
    with pytest.raises(B.BatchError, match="written ONCE"):
        B.write_g1(str(tmp_path), g, {})


def test_the_kit_gate_subcommand_exits_on_the_verdict(tmp_path):
    f = tmp_path / "g1.json"
    f.write_text(json.dumps(_g1([60.0, 61.0, 62.0], [55.0, 57.0, 59.0])))
    assert K.main(["gate", str(f), "--stage", str(tmp_path), "--write"]) == 1           # HOLD
    assert B.read_g1(str(tmp_path)) == B.HOLD


# -------------------------------------------------------------------------- AM-9 / AM-3 / AM-4 (the order and its conditions)
def test_batch1_is_am9s_order_the_toy_block_first():
    """AM-9 (orchestrator, 2026-10-20): T1 (toy C0 + X47 s3601-s3603, AM-7's interleave), T1b on X47's look-1 CONTINUE,
    then production P1-P4 once FP32_ALT has landed, and the toy calibration T2 last (conditional on production)."""
    assert [n for n, _ in B.STAGES] == ["T1", "T1b", "P1", "P2", "P3", "P4", "T2"]
    assert list(K.BATCH1) == [
        ("toy_c0", 3601), ("toy_x47", 3601), ("toy_x47", 3602), ("toy_c0", 3602), ("toy_c0", 3603), ("toy_x47", 3603),
        ("toy_x47", 3604), ("toy_c0", 3604), ("toy_c0", 3605), ("toy_x47", 3605),
        ("a1", 3501), ("c0", 3501), ("c0", 3502), ("a1", 3502), ("a1", 3503), ("c0", 3503),
        ("c0", 3504), ("c0", 3505), ("a1", 3504), ("a1", 3505),
        ("d1", 3501), ("d1", 3502), ("d1", 3503),
        ("toy_a1", 3601), ("toy_a1", 3602), ("toy_a1", 3603)]                 # AM-RP-c: A2 / toy A2 DROPPED
    assert len({K.run_name(a, s) for a, s in K.BATCH1}) == 26
    first_prod = min(i for i, (a, _) in enumerate(K.BATCH1) if not a.startswith("toy_"))
    assert all(a.startswith("toy_") for a, _ in K.BATCH1[:first_prod]) and first_prod == 10   # T1 + T1b, then production


def _save(stage: Path, arm: str, look: int, **doc) -> None:
    (stage / B.READS_DIR).mkdir(exist_ok=True)
    (stage / B.READS_DIR / f"{arm}_look{look}.json").write_text(json.dumps(dict(arm=B.READ_LABEL[arm], look=look, **doc)))


def _g1_record(stage: Path, verdict: str) -> None:
    (stage / B.G1_FILE).write_text(json.dumps({"schema": B.G1_SCHEMA, "verdict": verdict}))


def test_t1_and_p1_are_unconditional_and_g1_gates_d1_and_c0s_look_2(tmp_path):
    st = str(tmp_path)
    _fill_fp32(tmp_path)                                                   # FP32_ALT named: production may be asked
    for a, s in [("a1", 3501), ("c0", 3503), ("toy_c0", 3601), ("toy_x47", 3603)]:
        assert B.stage_status(st, a, s)[0] == B.ALLOWED, (a, s)
    for a, s in [("d1", 3501), ("c0", 3504)]:
        assert B.stage_status(st, a, s)[0] == B.PENDING, (a, s)
    _g1_record(tmp_path, B.HOLD)
    assert B.stage_status(st, "c0", 3504)[0] == B.ALLOWED          # the HOLD branch runs C0's look-2 seeds next
    assert B.stage_status(st, "d1", 3501)[0] == B.REFUSED
    assert B.stage_status(st, "a2", 3501)[0] == B.REFUSED
    assert B.stage_status(st, "toy_a1", 3601)[0] == B.REFUSED
    (tmp_path / B.G1_FILE).unlink()
    _g1_record(tmp_path, B.PASS)
    assert B.stage_status(st, "d1", 3501)[0] == B.ALLOWED


def test_am_rp_c_a2_and_toy_a2_are_dropped_refused_at_every_seed_and_whatever_the_records_say(tmp_path):
    """AM-RP-c (orchestrator, 2026-10-20): A2 and toy A2 left the queue. ``stage-check`` REFUSES every seed of them
    with the ruling as its reason, even on the records that once ALLOWED them (G1 PASS, production A2 final); their
    runs would still be judged by the side's pin (they stay batch-1 arms). Fails on revert of the DROPPED rule."""
    st = str(tmp_path)
    _fill_fp32(tmp_path)
    _g1_record(tmp_path, B.PASS)
    _save(tmp_path, "a2", 1, verdict="NULL", r1={"final": True, "class": "NULL"})
    for a in ("a2", "toy_a2"):
        assert not any(a == arm for _, forks in B.STAGES for arm, _ in forks)
        assert a in B.BATCH1_ARMS
        for s in K.ARMS[a].seeds:
            status, why = B.stage_status(st, a, s)
            assert status == B.REFUSED and "AM-RP-c" in why, (a, s, why)
    assert B.stage_status(st, "d1", 3501)[0] == B.ALLOWED and B.stage_status(st, "toy_a1", 3601)[0] == B.PENDING


def test_a_look_2_bon_fork_needs_its_look_1_continue(tmp_path):
    st = str(tmp_path)
    _fill_fp32(tmp_path)
    assert B.stage_status(st, "a1", 3504)[0] == B.PENDING
    _save(tmp_path, "a1", 1, verdict="NULL", r1={"final": True, "class": "NULL"})
    assert B.stage_status(st, "a1", 3504)[0] == B.REFUSED
    (tmp_path / B.READS_DIR / "a1_look1.json").unlink()
    _save(tmp_path, "a1", 1, verdict="CONTINUE", r1={"final": False, "class": "UNDECIDED"})
    assert B.stage_status(st, "a1", 3505)[0] == B.ALLOWED


def test_am4_x47_look_2_and_its_toy_c0_seeds_need_x47_look_1_continue(tmp_path):
    st = str(tmp_path)
    assert B.stage_status(st, "toy_x47", 3604)[0] == B.PENDING
    assert B.stage_status(st, "toy_c0", 3604)[0] == B.PENDING
    _save(tmp_path, "toy_x47", 1, verdict="PER-STEP ONLY")
    assert B.stage_status(st, "toy_x47", 3604)[0] == B.REFUSED
    _save(tmp_path, "toy_x47", 1, verdict="CONTINUE")
    assert B.stage_status(st, "toy_x47", 3605)[0] == B.ALLOWED and B.stage_status(st, "toy_c0", 3605)[0] == B.ALLOWED
    assert K.build_argv("toy_x47", 3605, {"side": "toy", "archive": [],
                                          "pstar": {"path": "m/rb_t/checkpoints/c.zip", "step": 10, "sha256": "s"}})


def test_am3_the_toy_calibration_needs_productions_final_class(tmp_path):
    st = str(tmp_path)
    _fill_fp32(tmp_path)
    _g1_record(tmp_path, B.PASS)
    assert B.stage_status(st, "toy_a1", 3601)[0] == B.PENDING             # production A1 not read yet
    _save(tmp_path, "a1", 1, verdict="CONTINUE", r1={"final": False, "class": "UNDECIDED"})
    assert B.stage_status(st, "toy_a1", 3601)[0] == B.PENDING             # look 2 may still decide it
    _save(tmp_path, "a1", 2, verdict="UNDECIDED", r1={"final": True, "class": "UNDECIDED"})
    status, why = B.stage_status(st, "toy_a1", 3602)
    assert status == B.REFUSED and "NOT SHOWN" in why
    (tmp_path / B.READS_DIR / "a1_look2.json").unlink()
    _save(tmp_path, "a1", 2, verdict="ADOPT-CANDIDATE", r1={"final": True, "class": "GAIN"})
    assert B.stage_status(st, "toy_a1", 3603)[0] == B.ALLOWED
    assert B.stage_status(st, "toy_a2", 3601)[0] == B.REFUSED             # AM-RP-c: toy A2 is DROPPED


def test_a_read_record_for_another_arm_or_look_is_refused(tmp_path):
    _fill_fp32(tmp_path)
    (tmp_path / B.READS_DIR).mkdir()
    (tmp_path / B.READS_DIR / "a1_look1.json").write_text(json.dumps({"arm": "A2", "look": 1, "verdict": "CONTINUE"}))
    with pytest.raises(B.BatchError, match="not A1 look 1"):
        B.stage_status(str(tmp_path), "a1", 3504)


def test_the_disk_check_skips_the_forks_the_records_refused(tmp_path):
    _fill_fp32(tmp_path)
    ok, msg = K.disk_check("/nonexistent", 10.0, thinning=False, existing=[], free_gib=1000.0, stage=str(tmp_path))
    assert ok and "46 batch-1/2 fork(s)" in msg, msg                       # PENDING forks are counted (A2s dropped; 20 of batch 2)
    _g1_record(tmp_path, B.HOLD)                                          # D1 and toy A1 (3 + 3) refused
    ok, msg = K.disk_check("/nonexistent", 10.0, thinning=False, existing=[], free_gib=1000.0, stage=str(tmp_path))
    assert ok and "40 batch-1/2 fork(s)" in msg, msg


# ------------------------------------------------------------------------------------------------------ AM-6
FORK = 280_000_170


def _series(judged, p99, start=FORK + 100_000):
    return {"judged_max": [(start + i, v) for i, v in enumerate(judged)],
            "p99": [(start + i, v) for i, v in enumerate(p99)]}


def test_k9b_passes_rows_under_the_ceilings_and_voids_a_crossing():
    assert B.judge_k9b("r", FORK, _series([2.6e-4, 1e-5], [2.7e-5, 1e-6])).verdict == B.PASS
    k = B.judge_k9b("r", FORK, _series([2.6e-4, 2e-3], [2.7e-5, 1e-6]))
    assert k.verdict == B.VOID and any("judged_max" in f for f in k.fires)
    k = B.judge_k9b("r", FORK, _series([2.6e-4], [3e-4]))
    assert k.verdict == B.VOID and any("p99" in f for f in k.fires)
    k = B.judge_k9b("r", FORK, _series([float("nan")], [1e-6]))
    assert k.verdict == B.VOID and any("non-finite" in f for f in k.fires)
    k = B.judge_k9b("r", FORK, _series([1e-5], [1e-6]), log_text="x " + B.K9B_NONFINITE_LINE + " y")
    assert k.verdict == B.VOID


def test_k9b_reads_only_rows_after_the_fork_step():
    s = _series([1e-5], [1e-6])
    s["judged_max"].insert(0, (FORK, 5e-3))                               # an INHERITED row at the fork step
    assert B.judge_k9b("r", FORK, s).verdict == B.PASS
    assert B.judge_k9b("r", FORK - 1, s).verdict == B.VOID                 # the same row read as the fork's own


def test_k9b_rule8_within_1e9_relative_of_a_ceiling_is_recorded_not_fired():
    k = B.judge_k9b("r", FORK, _series([1e-3 * (1 + 5e-10)], [1e-6]))
    assert k.verdict == B.PASS and k.near
    assert B.judge_k9b("r", FORK, _series([1e-3 * (1 + 1e-6)], [1e-6])).verdict == B.VOID


def test_k9b_absence_and_an_unknown_fork_are_inconclusive_never_a_pass():
    assert B.judge_k9b("r", FORK, {"judged_max": [], "p99": [(FORK + 1, 1e-6)]}).verdict == B.INCONCLUSIVE
    assert B.judge_k9b("r", None, _series([1e-5], [1e-6])).verdict == B.INCONCLUSIVE


def _write_events(path: Path, rows) -> None:
    from torch.utils.tensorboard import SummaryWriter

    path.mkdir(parents=True, exist_ok=True)
    w = SummaryWriter(log_dir=str(path))
    for tag, step, v in rows:
        w.add_scalar(tag, v, step)
    w.close()


def test_k9b_read_on_a_run_dir_excludes_the_inherited_file(tmp_path):
    run = tmp_path / "rb_x35_c0_s3501"
    tb = run / "tb"
    _write_events(tmp_path / "inh", [("behaviour/max_abs_dlogp_judged", FORK - 10, 9e-3),
                                     ("behaviour/p99_abs_dlogp_current", FORK - 10, 9e-4)])
    tb.mkdir(parents=True)
    src = next((tmp_path / "inh").glob("*tfevents*"))
    src.rename(tb / "events.out.tfevents.0000000000.inherited.0.0")
    (tb / "INHERITED_FROM.json").write_text(json.dumps({"fork_step": FORK}))
    _write_events(tb, [("behaviour/max_abs_dlogp_judged", FORK + 5000, 1e-5),
                       ("behaviour/p99_abs_dlogp_current", FORK + 5000, 1e-6)])
    k = B.k9b_read(str(run))
    assert k.verdict == B.PASS and k.fork_step == FORK and k.n_rows == {"judged_max": 1, "p99": 1}, k
    stage = tmp_path / "stage"
    stage.mkdir()
    (run / "final_model.zip").write_bytes(b"z")
    out = tmp_path / "scratch"
    ok, lines = B.k9b_batch(str(stage), str(tmp_path), out_dir=str(out))
    assert ok and lines == ["rb_x35_c0_s3501: K9(b) arm end PASS"]
    # F1 (d4387711): the stage is only READ; the missing record lands in the caller's out dir
    assert (out / B.K9B_DIR / "rb_x35_c0_s3501.json").is_file() and not (stage / B.K9B_DIR).exists()


def _finished(models, name="rb_x35_c0_s3501"):
    (models / name).mkdir(parents=True)
    (models / name / "final_model.zip").write_bytes(b"z")
    return name


class _Reads:
    """A K9(b) reader that counts its calls and PASSes (the rows are covered above)."""
    def __init__(self, verdict="PASS"):
        self.calls, self.verdict = [], verdict

    def __call__(self, run_dir):
        self.calls.append(run_dir)
        return _k9b(os.path.basename(run_dir), self.verdict)


def _k9b(run, verdict):
    return B.judge_k9b(run, FORK, _series([1e-5], [1e-6])) if verdict == "PASS" else B.judge_k9b(run, FORK, _series([2e-3], [1e-6]))


def test_k9b_batch_default_output_is_a_scratch_dir_outside_every_checkout(tmp_path, monkeypatch):
    """F1 (d4387711): with no out dir the missing record goes to $GEN3AI_X35_K9B_SCRATCH, never into the stage."""
    name = _finished(tmp_path / "models")
    scratch, stage = tmp_path / "scr", tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setenv(B.K9B_SCRATCH_ENV, str(scratch))
    reader = _Reads()
    ok, _ = B.k9b_batch(str(stage), str(tmp_path / "models"), reader=reader)
    assert ok and (scratch / B.K9B_DIR / f"{name}.json").is_file() and not any(stage.iterdir())
    B.k9b_batch(str(stage), str(tmp_path / "models"), reader=reader)
    assert len(reader.calls) == 1                                  # the scratch record is read back, not re-read
    monkeypatch.delenv(B.K9B_SCRATCH_ENV)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert B.k9b_scratch_dir() == str(tmp_path / "home" / ".cache" / "gen3ai" / "x35_k9b")


def test_k9b_batch_reads_a_shipped_record_in_the_stage_without_reading_the_run(tmp_path):
    name = _finished(tmp_path / "models")
    stage = tmp_path / "stage"
    (stage / B.K9B_DIR).mkdir(parents=True)
    (stage / B.K9B_DIR / f"{name}.json").write_text(json.dumps({"verdict": "VOID"}))
    reader = _Reads()
    ok, lines = B.k9b_batch(str(stage), str(tmp_path / "models"), out_dir=str(tmp_path / "o"), reader=reader)
    assert not ok and lines == [f"{name}: K9(b) arm end VOID"] and not reader.calls


def test_a_k9b_record_inside_the_main_checkout_is_refused(tmp_path, monkeypatch):
    """F1 (d4387711): fails on revert. 'main' is a tmp dir here, so a revert writes into tmp, never the real main."""
    fake_main = tmp_path / "main"
    (fake_main / "designs").mkdir(parents=True)
    monkeypatch.setattr(B, "_main_checkout_roots", lambda: [os.path.realpath(fake_main)])
    name = _finished(tmp_path / "models")
    for bad in (fake_main / "designs", fake_main / "designs" / "new_dir"):
        with pytest.raises(B.BatchError, match="main checkout"):
            B.k9b_batch(str(tmp_path / "stage"), str(tmp_path / "models"), out_dir=str(bad), reader=_Reads())
        with pytest.raises(B.BatchError, match="main checkout"):
            B.k9b_record(str(bad), _k9b(name, "PASS"))
    assert not list(fake_main.rglob("*.json"))
    # a symlink into main is main too; a sibling that merely shares the name prefix is not
    link = tmp_path / "link"
    link.symlink_to(fake_main / "designs")
    with pytest.raises(B.BatchError, match="main checkout"):
        B.k9b_record(str(link), _k9b(name, "PASS"))
    B.k9b_record(str(tmp_path / "main2"), _k9b(name, "PASS"))
    assert (tmp_path / "main2" / B.K9B_DIR / f"{name}.json").is_file()


def test_the_real_main_checkout_is_a_refused_k9b_destination():
    """Detection only (nothing is written): the real main checkout resolves and its own stage is refused."""
    from utils.git import get_main_repo_root
    from utils.paths import repo_root

    main = get_main_repo_root(cwd=str(repo_root()))
    with pytest.raises(B.BatchError, match="main checkout"):
        B.refuse_inside_main_checkout(os.path.join(main, "designs", "research_state", "x"))
    B.refuse_inside_main_checkout(os.path.join(os.path.expanduser("~"), ".cache", "gen3ai", "x35_k9b"))


def test_kit_k9b_batch_out_dir_flag_and_main_refusal(tmp_path, monkeypatch, capsys):
    fake_main = tmp_path / "main"
    fake_main.mkdir()
    monkeypatch.setattr(B, "_main_checkout_roots", lambda: [os.path.realpath(fake_main)])
    models = tmp_path / "models"
    _finished(models)
    monkeypatch.setattr(B, "k9b_read", _Reads())
    base = ["k9b-batch", "--stage", str(tmp_path / "stage"), "--models-dir", str(models)]
    assert K.main(base + ["--out-dir", str(fake_main / "stage")]) == 2
    assert "REFUSED" in capsys.readouterr().err and not list(fake_main.rglob("*"))
    assert K.main(base + ["--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "k9b" / "rb_x35_c0_s3501.json").is_file()


# ---- an OVERLAP arm (X14, `--overlap-collect`): `behaviour_snapshot/*` is the REQUIRED family -------------------------
NAN = float("nan")


def _snap(judged, p99, start=FORK + 100_000):
    return {"snapshot_judged_max": [(start + i, v) for i, v in enumerate(judged)],
            "snapshot_p99": [(start + i, v) for i, v in enumerate(p99)]}


def _overlap_series(*, serial_judged=(5e-5, 6e-5), serial_p99=(1e-5, 1e-5), snap_judged=(1.26e-4, 7e-5, 8e-5),
                    snap_p99=(3.1e-5, 2e-5, 2e-5)):
    """The shape of the 2026-10-20 battery's X14 ON arm AFTER `gen3_k9b_no_row_no_tag_v1`: the learner's own probe has
    rows only at each process's first (serial) update; the snapshot family has one per overlapped update."""
    return {**_series(serial_judged, serial_p99), **_snap(snap_judged, snap_p99, start=FORK + 200_000)}


def test_an_overlap_arm_is_judged_on_the_snapshot_family_and_passes_on_it():
    """The battery's real ON arm: judged max 1.26e-4 (over the 1e-4 `fatal` bar, 8x under AM-6's 1e-3), p99 3.1e-5."""
    k = B.judge_k9b("r", FORK, _overlap_series(), overlap=True)
    assert k.verdict == B.PASS and k.overlap, k
    assert k.n_rows == {"judged_max": 2, "p99": 2, "snapshot_judged_max": 3, "snapshot_p99": 3}, k.n_rows
    assert k.max_value["snapshot_judged_max"] == 1.26e-4
    assert k.as_dict()["required_family"] == "behaviour_snapshot"


def test_an_overlap_arm_read_on_the_learners_probe_alone_is_inconclusive_never_a_near_vacuous_pass():
    """Fails on revert: before, the overlap flag did not exist and two serial rows PASSed an arm whose every stale row
    was judged elsewhere (TECH_DEBT 2(b))."""
    only_serial = _series([5e-5, 6e-5], [1e-5, 1e-5])
    assert B.judge_k9b("r", FORK, only_serial).verdict == B.PASS                      # a SERIAL arm: this is its read
    k = B.judge_k9b("r", FORK, only_serial, overlap=True)
    assert k.verdict == B.INCONCLUSIVE and any("behaviour_snapshot/" in r for r in k.reasons), k
    # a snapshot family with only one of its two statistics is not a read either
    half = {**only_serial, "snapshot_judged_max": [(FORK + 5, 1e-5)]}
    assert B.judge_k9b("r", FORK, half, overlap=True).verdict == B.INCONCLUSIVE


def test_an_overlap_arm_without_any_learner_probe_row_is_nothing_judged_there_not_inconclusive_and_not_void():
    """A missing `behaviour/*` current-row statistic is "nothing judged" (the snapshot is the required family)."""
    snap_only = _snap([1e-5, 2e-5], [1e-6, 2e-6])
    k = B.judge_k9b("r", FORK, snap_only, overlap=True)
    assert k.verdict == B.PASS and k.n_rows["judged_max"] == 0 and k.max_value["judged_max"] is None, k
    # the same series without the flag: the snapshot rows are themselves the evidence of an overlap arm
    assert B.judge_k9b("r", FORK, snap_only).verdict == B.PASS
    # judged max present at the process's first update, its p99 row absent: still not a reason, still not a fire
    lopsided = {**snap_only, "judged_max": [(FORK + 10, 5e-5)], "p99": []}
    assert B.judge_k9b("r", FORK, lopsided, overlap=True).verdict == B.PASS


def test_a_real_nan_or_a_crossing_in_either_family_of_an_overlap_arm_still_voids():
    ok = _overlap_series()
    nan_serial = _overlap_series(serial_p99=(NAN, 1e-5))                              # the archived battery's shape
    nan_snap = _overlap_series(snap_p99=(3.1e-5, NAN, 2e-5))
    over_snap = _overlap_series(snap_judged=(1.26e-4, 2e-3, 8e-5))
    over_serial = _overlap_series(serial_p99=(1e-5, 3e-4))
    assert B.judge_k9b("r", FORK, ok, overlap=True).verdict == B.PASS
    for bad in (nan_serial, nan_snap, over_snap, over_serial):
        k = B.judge_k9b("r", FORK, bad, overlap=True)
        assert k.verdict == B.VOID and k.fires, k
    assert any("non-finite" in f and f.startswith("snapshot_p99") for f in
               B.judge_k9b("r", FORK, nan_snap, overlap=True).fires)
    assert B.judge_k9b("r", FORK, _snap([1e-5], [1e-6]), log_text=B.K9B_NONFINITE_LINE, overlap=True).verdict == B.VOID


def test_the_snapshot_rows_obey_the_fork_step_and_rule_8_like_the_serial_ones():
    inh, fresh = _snap([5e-3], [9e-4], start=FORK), _snap([1e-5], [1e-6])             # an INHERITED row at the fork step
    mixed = {k: inh[k] + fresh[k] for k in inh}
    assert B.judge_k9b("r", FORK, mixed, overlap=True).verdict == B.PASS
    assert B.judge_k9b("r", FORK - 1, mixed, overlap=True).verdict == B.VOID          # the same row read as the fork's own
    near = B.judge_k9b("r", FORK, _snap([1e-3 * (1 + 5e-10)], [1e-6]), overlap=True)
    assert near.verdict == B.PASS and near.near
    assert B.judge_k9b("r", FORK, _snap([1e-3 * (1 + 1e-6)], [1e-6]), overlap=True).verdict == B.VOID


def test_rows_sharing_the_last_step_are_two_updates_both_counted_and_both_judged():
    """The final dump writes the last update's scalars at the SAME step as the one before (`loop._final_dump`): 15 updates,
    14 distinct steps, 15 rows (the battery's both arms). Deduplicating by step would drop a real update and could drop
    its violation: the SECOND row at the shared step is the one over the ceiling."""
    steps = [FORK + 100_000 + 100 * i for i in range(14)]
    judged = [(st, 5e-5) for st in steps] + [(steps[-1], 2e-3)]
    p99 = [(st, 1e-5) for st in steps] + [(steps[-1], 1e-5)]
    k = B.judge_k9b("r", FORK, {"judged_max": judged, "p99": p99})
    assert k.n_rows == {"judged_max": 15, "p99": 15}, k.n_rows
    assert len({s for s, _ in judged}) == 14
    assert k.verdict == B.VOID and any("0.002" in f and f.endswith(f"at step {steps[-1]}") for f in k.fires), k.fires


def _overlap_run(tmp_path: Path, *, meta=None, rows=()):
    run = tmp_path / "rb_x14ab_on_s2601"
    tb = run / "tb"
    tb.mkdir(parents=True)
    (tb / "INHERITED_FROM.json").write_text(json.dumps({"fork_step": FORK}))
    _write_events(tb, list(rows))
    if meta is not None:
        (run / "metadata.json").write_text(json.dumps(meta))
    return run


SNAP_J, SNAP_P = "behaviour_snapshot/max_abs_dlogp_judged", "behaviour_snapshot/p99_abs_dlogp_current"
SER_J, SER_P = "behaviour/max_abs_dlogp_judged", "behaviour/p99_abs_dlogp_current"
OVL = "overlap/join_wait_ms"


def test_k9b_read_of_an_overlap_run_dir_judges_the_snapshot_and_counts_the_shared_last_step(tmp_path):
    steps = [FORK + 100_000 + 100 * i for i in range(14)]
    rows = [(SER_J, steps[0], 5e-5), (SER_P, steps[0], 1e-5)]                         # the first, serial update
    for i, st in enumerate(steps[1:], 1):
        rows += [(SNAP_J, st, 5e-5 + 1e-6 * i), (SNAP_P, st, 1e-5), (OVL, st, 3.0)]
    rows += [(SNAP_J, steps[-1], 6e-5), (SNAP_P, steps[-1], 1e-5)]                  # the final dump: same step, 14th row
    run = _overlap_run(tmp_path, meta={"env_core": {"overlap_collect": True}, "cli_args": {"overlap_collect": True}},
                       rows=rows)
    k = B.k9b_read(str(run))
    assert k.verdict == B.PASS and k.overlap, k
    assert k.n_rows == {"judged_max": 1, "p99": 1, "snapshot_judged_max": 14, "snapshot_p99": 14}, k.n_rows
    assert "metadata.json env_core.overlap_collect" in k.overlap_evidence
    assert "metadata.json cli_args.overlap_collect" in k.overlap_evidence
    assert any(e.startswith("13 own overlap/join_wait_ms") for e in k.overlap_evidence)
    assert any(e.startswith("14 own behaviour_snapshot/*") for e in k.overlap_evidence)


def test_k9b_read_refuses_to_pass_an_arm_the_record_says_overlapped_on_the_learners_probe_alone(tmp_path):
    """The record is the witness: an arm that recorded `overlap_collect` but has no snapshot row (a vacuous overlap, or
    the snapshot pass missing) is INCONCLUSIVE even though its two serial `behaviour/*` rows look clean."""
    rows = [(SER_J, FORK + 100_000, 5e-5), (SER_P, FORK + 100_000, 1e-5)]
    run = _overlap_run(tmp_path, meta={"env_core": {"overlap_collect": True}}, rows=rows)
    k = B.k9b_read(str(run))
    assert k.verdict == B.INCONCLUSIVE and k.overlap and k.overlap_evidence == ("metadata.json env_core.overlap_collect",), k
    assert any("behaviour_snapshot/" in r for r in k.reasons)
    # the same rows with a record that says OFF (or none) and no overlap rows are a SERIAL arm's PASS
    run2 = _overlap_run(tmp_path / "s", meta={"env_core": {"overlap_collect": False}, "cli_args": {"overlap_collect": False}},
                        rows=rows)
    k2 = B.k9b_read(str(run2))
    assert k2.verdict == B.PASS and not k2.overlap and k2.n_rows == {"judged_max": 1, "p99": 1}, k2


def test_k9b_read_takes_an_arm_whose_record_is_missing_as_overlapped_when_its_own_rows_say_so(tmp_path):
    """No `metadata.json` at all, but the arm logged `overlap/*` rows and no snapshot row: still an overlap arm."""
    rows = [(SER_J, FORK + 100_000, 5e-5), (SER_P, FORK + 100_000, 1e-5), (OVL, FORK + 100_100, 2.0)]
    k = B.k9b_read(str(_overlap_run(tmp_path, meta=None, rows=rows)))
    assert k.verdict == B.INCONCLUSIVE and k.overlap and len(k.overlap_evidence) == 1, k


def test_k9b_read_of_the_archived_battery_shape_a_nan_current_row_statistic_still_voids(tmp_path):
    """An overlap arm that wrote the pre-fix NaN placeholder on `behaviour/*` is VOID (a real non-finite |d log pi| to
    the reader); an arm that wrote NO row there (the fix) and judged the snapshot passes."""
    snap = [(SNAP_J, FORK + 100_100, 5e-5), (SNAP_P, FORK + 100_100, 1e-5), (OVL, FORK + 100_100, 2.0)]
    meta = {"env_core": {"overlap_collect": True}}
    nan = _overlap_run(tmp_path / "nan", meta=meta, rows=snap + [(SER_P, FORK + 100_100, NAN)])
    assert B.k9b_read(str(nan)).verdict == B.VOID
    fixed = _overlap_run(tmp_path / "fixed", meta=meta, rows=snap)
    assert B.k9b_read(str(fixed)).verdict == B.PASS


def test_k9b_batch_holds_on_a_void_finished_arm(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    _run(tmp_path / "models", "rb_x35_a1arch_s3501", PIN, final=True)
    _run(tmp_path / "models", "rb_x35_c0_s3501", PIN)                   # not finished: not read
    void = B.judge_k9b("rb_x35_a1arch_s3501", FORK, _series([2e-3], [1e-6]))
    ok, lines = B.k9b_batch(str(stage), str(tmp_path / "models"), reader=lambda rd: void)
    assert not ok and lines == ["rb_x35_a1arch_s3501: K9(b) arm end VOID"]


# ------------------------------------------------------------------------------- the launch script calls them
def test_launch_x35_runs_every_amendment_pre_check():
    from utils.paths import repo_path

    sh = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17/launch_x35.sh").read_text()
    for needle in (
            # AM-1 as amended by AM-8: the pin is judged and written PER SIDE, and a typed --t62 is checked in the dry run
            # (AM-D1-FIX: the launching arm is passed, so d1 is held to its own corpus-only pin)
            "kit pin-check --pin \"$PIN\" --side \"$SIDE\" --arm \"$ARM\" --stage \"$STAGE\" --models-dir \"$MODELS\" "
            "${T62:+--t62 \"$T62\"} >",
            "PINF=$STAGE/x35_pin_batch1_$SIDE.txt", "[ -e \"$PINF\" ] || [ -n \"$T62\" ]",
            # AM-8: the side's own required commits (production: FP32_ALT)
            "kit required-check --side \"$SIDE\" --pin \"$PIN\" --stage \"$STAGE\" --repo \"$REPO\" >",
            "kit stage-check --arm", "kit pool-check --arm \"$ARM\" --call \"$R\" --stage \"$STAGE\" >",
            "kit k9b-batch", "--side \"$SIDE\" --arm \"$ARM\" --stage \"$STAGE\" --models-dir \"$MODELS\" --write ${T62:+--t62",
            "pool-check --arm \"$ARM\" --call \"$R\" --stage \"$STAGE\" --write",
            "pool-seed-check --arm \"$ARM\" --run-dir \"$MODELS/$RUN\" --call \"$R\" --stage \"$STAGE\"",
            "kit disk --required-gib \"$gib\" --models-dir \"$MODELS\" --stage \"$STAGE\"",
            # AMENDMENT D1-CORPUS (2026-10-20): the arm's team corpus at the pin
            "kit corpus-check --arm \"$ARM\" --pin \"$PIN\" --repo \"$REPO\" >"):
        assert needle in sh, needle
    assert "x35_pin_batch1.txt" not in sh.replace("x35_pin_batch1_", "")        # the single batch pin file is gone
    # the once-only writes happen AFTER the GO's pre-checks, never in a dry run
    assert sh.index("prechecks go\n") < sh.index("--write ${T62:+--t62")
    assert re.search(r"if \[ \"\$GO\" -ne 1 \]; then\n  prechecks dry\n", sh)
    # the required-commit check runs inside prechecks(), before the run-dir and R checks
    assert sh.index("kit required-check") < sh.index("# 3. the run dir") < sh.index("prechecks dry")
    assert sh.index("kit corpus-check") < sh.index("# 3. the run dir")


def test_launch_x35_pre_check_1b_refuses_an_empty_cuda_visible_devices():
    """Pre-check 1b (2026-10-20): an inherited CUDA_VISIBLE_DEVICES="" killed toy_x47 3602's first GO at startup
    ("No CUDA GPUs are available") and the launcher's --dry-run did not catch it. The check sits in prechecks(),
    right after pre-check 1's `kit live`, and is RUN here: set-and-empty HOLDs; unset or a device list passes."""
    from utils.paths import repo_path

    sh = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17/launch_x35.sh").read_text()
    line = next(l for l in sh.splitlines() if 'CUDA_VISIBLE_DEVICES+x' in l)
    body = sh[sh.index("prechecks() {"):]
    assert body.index("kit live") < body.index("CUDA_VISIBLE_DEVICES+x") < body.index("local lease")
    probe = 'hold() { echo "HOLD: $1"; exit 7; }\n' + line.strip() + '\necho PASS\n'
    for env_args, want in ((["CUDA_VISIBLE_DEVICES="], "HOLD"), (["-u", "CUDA_VISIBLE_DEVICES"], "PASS"),
                           (["CUDA_VISIBLE_DEVICES=0"], "PASS")):
        r = subprocess.run(["env", *env_args, "bash", "-c", probe], capture_output=True, text=True)
        assert r.stdout.startswith(want), (env_args, r.stdout, r.stderr)
        assert (r.returncode == 7) == (want == "HOLD")


def test_launch_x35_rotates_an_existing_launcher_log_at_go(tmp_path):
    """A re-launch of the same run name (after a failed GO) must not share the old launcher log: the watcher read the
    old attempt's "cannot restart" lines as the new run's failure and exited at once (toy_x47 3602, 2026-10-20). At
    GO, before the launcher starts, an existing <run>.launcher.log is RENAMED to <log>.<ts>.prev (never deleted)."""
    from utils.paths import repo_path

    sh = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17/launch_x35.sh").read_text()
    line = next(l for l in sh.splitlines() if l.startswith('if [ -e "$J/$RUN.launcher.log" ]; then PREV='))
    assert sh.index("batch records:") < sh.index(line) < sh.index('log "START $RUN') < sh.index(">> \"$J/$RUN.launcher.log\"")
    probe = ('J="$1"; RUN=r; log() { echo "LOG: $1"; }; hold() { echo "HOLD: $1"; exit 7; }\n' + line + '\necho DONE\n')
    log = tmp_path / "r.launcher.log"
    log.write_text("old attempt: cannot restart\n")
    r = subprocess.run(["bash", "-c", probe, "probe", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0 and "rotated the previous launcher log" in r.stdout, (r.stdout, r.stderr)
    prev = [p for p in tmp_path.iterdir() if p.name.startswith("r.launcher.log.") and p.name.endswith(".prev")]
    assert not log.exists() and len(prev) == 1 and prev[0].read_text() == "old attempt: cannot restart\n"
    r = subprocess.run(["bash", "-c", probe, "probe", str(tmp_path)], capture_output=True, text=True)    # no log: no-op
    assert r.returncode == 0 and "rotated" not in r.stdout and not log.exists()


def test_launch_x35_runs_the_launcher_dry_run_with_the_card_visible():
    """The launcher's CUDA-visibility preflight (d812e78c4) refuses a --device cuda launch under an empty
    CUDA_VISIBLE_DEVICES, dry run included, so pre-check 6 must not export one (it blocked toy_c0 3602's GO,
    2026-10-20); pre-check 1b, which refuses an inherited empty value, runs first in the same prechecks()."""
    from utils.paths import repo_path

    sh = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17/launch_x35.sh").read_text()
    body = sh[sh.index("prechecks() {"):]
    line = next(l for l in body.splitlines() if "-m main.launcher --dry-run" in l)
    assert "CUDA_VISIBLE_DEVICES" not in line, line
    assert body.index("CUDA_VISIBLE_DEVICES+x") < body.index("-m main.launcher --dry-run")


def test_am8_the_shipped_production_file_names_fp32_alt_d4af9f1e():
    """Filled 2026-10-21 after the GPU battery (tau_cuda_parity PASS, FP32_ALT cost 1.37-2.55 % <= 3 %)."""
    shipped = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17/"
                        + B.SIDE_REQUIRED_FILE.format(side=PROD))
    tmp_rows = [ln.split(" ", 2)[:2] for ln in shipped.read_text().splitlines() if ln and not ln.startswith("#")]
    assert tmp_rows == [["d4af9f1e26061681644473d22e2022dda2b43655", FP32]]
    assert B._is_commit_sha(tmp_rows[0][0])


def test_every_committed_read_record_carries_its_arms_registered_label():
    """Each saved ``reads/<arm>_look<k>.json`` at the real stage loads through ``read_record`` (its ``arm`` is the
    registered READ_LABEL, e.g. D1's is 'X38'), so stage-check can never refuse a launch on a mislabelled record (the
    2026-10-23 D1 look-1 record was written as 'D1' and held D1's look-2 seeds)."""
    stage = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17")
    found = 0
    for p in sorted((stage / B.READS_DIR).glob("*_look*.json")):
        m = re.fullmatch(r"(.+)_look(\d+)\.json", p.name)
        assert m, p.name
        arm, look = m.group(1), int(m.group(2))
        assert arm in B.READ_LABEL, f"{p.name}: no registered label for arm {arm!r}"
        assert B.read_record(str(stage), arm, look) is not None
        found += 1
    assert found >= 1, "no committed read record found: the precondition of this test does not hold"


# ------------------------------------------------- X48 (PREREG x48_kffn_production_2026-10-27): one look, side-pinned
def test_x48_is_side_pinned_and_one_look_on_the_side_gate_only(tmp_path):
    stage = tmp_path / "stage"
    (stage / B.READS_DIR).mkdir(parents=True)
    st = str(stage)
    assert "x48" in B.SIDE_PINNED_ARMS and "x48" not in B.ARM_PINS and B.READ_LABEL["x48"] == "X48"
    assert B.stage_status(st, "x48", 3501)[0] == B.PENDING                       # AM-8: no FP32_ALT named
    _fill_fp32(stage)
    for seed in (3501, 3502, 3503):
        status, why = B.stage_status(st, "x48", seed)
        assert status == B.ALLOWED and "X48 PREREG" in why and B.stage_of("x48", seed) == "X48", why
    assert B.stage_of("x48", 3504) is None and B.stage_of("x48", 3505) is None   # no look 2 is registered
    (stage / B.READS_DIR / "x48_look1.json").write_text(json.dumps({"arm": "X48", "look": 1, "verdict": "GO"}))
    assert B.read_record(st, "x48", 1)["verdict"] == "GO"
    (stage / B.READS_DIR / "x48_look1.json").write_text(json.dumps({"arm": "X61P", "look": 1}))
    with pytest.raises(B.BatchError, match="not X48"):
        B.read_record(st, "x48", 1)
    names = B.side_pinned_run_names()
    assert {n for n, (a, _) in names.items() if a == "x48"} == {f"rb_x35_x48kffn_s{s}" for s in (3501, 3502, 3503)}
