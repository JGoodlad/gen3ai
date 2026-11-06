"""AM-X60T (2026-11-06): X60-T's kit arm ``x60t`` (C0 + the human-data clause and nothing else), its own pin
``batch.X60T_PIN`` (D1's pin + ONE commit changing EXACTLY the enumerated paths, REVIEW F F-M3 fix 4), the human-corpus
check (the pin's stub and the LOCAL corpus by sha256, with its two soak records), and the stages (look 2 and C0's
top-up seeds 3506-3508 only on x60t's own look-1 CONTINUE).

The real-pin tests read this checkout's object store (the pin is the local branch ``x60t-pin``, tag
``x60t-pin-<sha8>`` on origin); a check that needs the LOCAL human corpus skips where it is absent (it never enters
git). No test reads or writes a human team text.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from main.x35 import batch as B
from main.x35 import kit as K
from main.x35.kit_test import standin_call
from utils.paths import human_teams_dir, repo_root

D1_PIN = "440b61c320efdd4e172f54a192a2c84630a4c796"
SIDE_PIN = "f4b25238def57900c0305bb2a3a5a8f9edaef285"
#: main's switch commit, whose files the pin carries byte for byte (but the two merged ones and the stub)
SWITCH_COMMIT = "fbd1afa38"
#: the pin's files that are NOT main's switch-commit bytes: the two merged onto the pin's older code, and the stub
MERGED = {"src/utils/paths.py", "src/utils/team_corpora/__init__.py", "designs/ops/flag_census.md",
          "src/utils/team_corpora/human_corpora.json"}
LOCAL = human_teams_dir() / K.X60T_CORPUS["corpus"]
needs_local = pytest.mark.skipif(not (LOCAL / "manifest.json").is_file(),
                                 reason=f"the LOCAL human corpus {LOCAL} is not on this machine (never in git)")


def _git(*a: str, check: bool = True) -> str:
    return subprocess.run(["git", "-C", str(repo_root()), *a], check=check, capture_output=True, text=True).stdout


def _has(sha: str) -> bool:
    return subprocess.run(["git", "-C", str(repo_root()), "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True).returncode == 0


needs_pin = pytest.mark.skipif(not _has(B.X60T_PIN), reason="X60T_PIN is not in this object store (fetch the tag)")


# ------------------------------------------------------------------------------------------ the kit arm
def test_x60t_is_c0_plus_the_clause_and_nothing_else():
    """b-ON = C0's argv token for token + the ONE clause (``--human-data teams:hr_spec_v1.r3@0.5``) in N1-P's slot,
    the same P*, seed and steps; only the run name differs. Fails if any other token is added, dropped or moved."""
    call = standin_call()
    for seed in (3501, 3502, 3503, 3504, 3505):
        on, off = K.build_argv("x60t", seed, call), K.build_argv("c0", seed, call)
        assert on[:-2] == off[:-2][:off.index("--seed")] + list(K.X60T_LEVER) + ["--seed", str(seed)]
        assert on[-2:] == ["--run-name", f"rb_x60t_s{seed}"] and off[-1] == f"rb_x35_c0_s{seed}"
        assert len(on) == len(off) + 2
    assert K.X60T_LEVER == ("--human-data", "teams:hr_spec_v1.r3@0.5")
    assert K.X60T_CLAUSE == f"teams:{K.X60T_CORPUS['corpus']}@0.5" and K.ARM_HUMAN_DATA == {"x60t": K.X60T_CLAUSE}
    n1p = K.build_argv("n1p", 3501, call)
    assert K.build_argv("x60t", 3501, call).index("--human-data") == n1p.index("--team-mix")   # N1-P's slot
    d = K.describe("x60t", 3501, call)
    assert d["delta_text"] == "+20,000,000 steps" and d["parent_run_dir"].endswith("rb_x26_s2601")
    assert d["behaviour"] == K.ARMS["c0"].behaviour == "warn"


def test_x60t_seeds_and_c0s_top_up():
    assert K.ARMS["x60t"].seeds == tuple(range(3501, 3509))
    assert K.ARMS["c0"].seeds == (3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508)
    call = standin_call()
    for seed in (3506, 3507, 3508):            # the top-up is C0's argv at a new seed, nothing else
        top, c0 = K.build_argv("c0", seed, call), K.build_argv("c0", 3501, call)
        assert [t for t in top if t not in (str(seed), f"rb_x35_c0_s{seed}")] == \
               [t for t in c0 if t not in ("3501", "rb_x35_c0_s3501")]
    with pytest.raises(K.KitError, match="not registered"):
        K.build_argv("x60t", 3509, call)


def test_the_clause_parses_as_the_switchs_grammar_and_names_the_registered_corpus():
    from agents.info_levers.human_data import parse

    c = parse(K.X60T_CLAUSE)
    assert (c.lever, c.rowset, c.param) == ("teams", K.X60T_CORPUS["corpus"], 0.5)
    from utils.team_corpora.human import stub
    s = stub(K.X60T_CORPUS["corpus"])                     # main's committed stub agrees with the kit's identity
    for k in ("n", "tier", "tier_sha256", "data_sha256", "manifest_sha256", "soak_sha256"):
        assert s[k] == K.X60T_CORPUS[k], k


def test_the_validation_verdict_requires_the_clause_on_x60t_and_off_elsewhere():
    call = standin_call()
    base = ("this command would launch\n  role        : FORK of {p}\n  +20,000,000 steps\n  ✓ fork arch : compatible\n"
            "  K9(b) check : warn\n  20.68 GiB REQUIRED\n")
    chk = "  K9(b) behaviour check          : warn  (typed)\n"
    on_line = (f"  human data                     : {K.X60T_CLAUSE}  hash x  (training honours teams only)\n"
               f"      [HUMAN-DATA] lever=teams rowset={K.X60T_CORPUS['corpus']} "
               f"rows_sha={K.X60T_CORPUS['data_sha256']} clause=x\n")
    off_line = "  human data                     : off (no human data)\n"
    x, c0 = K.describe("x60t", 3501, call), K.describe("c0", 3501, call)
    dry_x, dry_c = base.format(p=x["parent_run_dir"]), base.format(p=c0["parent_run_dir"])
    ok, line = K.check_validation(x, dry_x + on_line, chk + on_line, 0, 0)
    assert ok and "human_data=[teams:hr_spec_v1.r3@0.5]" in line, line
    assert not K.check_validation(x, dry_x + off_line, chk + on_line, 0, 0)[0]     # the dry-run lost the clause
    assert not K.check_validation(x, dry_x, chk, 0, 0)[0]                         # no human-data line at all
    assert K.check_validation(c0, dry_c + off_line, chk + off_line, 0, 0)[0]
    assert K.check_validation(c0, dry_c, chk, 0, 0)[0]                            # an older transcript: no line
    assert not K.check_validation(c0, dry_c + on_line, chk, 0, 0)[0]              # C0 must never carry a clause


# ------------------------------------------------------------------------------------------ the pin
@needs_pin
def test_the_pin_changes_exactly_the_enumerated_paths_on_d1s_pin():
    """F-M3 fix 4: X60T_PIN = D1's pin + ONE commit changing EXACTLY X60T_PIN_PATHS; D1's pin is the side pin + the
    AM-D1-FIX label guard (corpus package only). Fails on any extra, missing or renamed path."""
    assert B.X60T_PIN_BASE == B.D1_PIN == D1_PIN and B.D1_PIN_BASE == SIDE_PIN
    assert B.ARM_PINS["x60t"] == B.X60T_PIN and B.ARM_PIN_AMENDMENT["x60t"] == "AM-X60T"
    assert _git("rev-list", "--parents", "-n", "1", B.X60T_PIN).split()[1:] == [D1_PIN]
    assert sorted(_git("diff", "--name-only", D1_PIN, B.X60T_PIN).split()) == sorted(B.X60T_PIN_PATHS)
    assert len(B.X60T_PIN_PATHS) == len(set(B.X60T_PIN_PATHS)) == 19
    assert all(p.startswith("src/") for p in B.X60T_PIN_PATHS if p != "designs/ops/flag_census.md")
    assert not [p for p in B.X60T_PIN_PATHS if p.startswith(("src/rust", "src/agents/model", "src/agents/observation"))]


@needs_pin
def test_the_pin_carries_mains_switch_bytes_and_the_corpus_stub_only():
    """Every backported file but the merged ones is byte-for-byte main's switch commit; the stub registers exactly
    hr_spec_v1.r3 with the kit's identity and carries no team text."""
    if not _has(SWITCH_COMMIT):
        pytest.skip(f"{SWITCH_COMMIT} is not in this object store")
    def changed(a: str, b: str, p: str) -> list:
        return [ln for ln in _git("diff", "-U0", a, b, "--", p).splitlines()
                if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]

    for p in sorted(set(B.X60T_PIN_PATHS) - MERGED):     # the same added / removed lines as main's switch commit
        assert changed(D1_PIN, B.X60T_PIN, p) == changed(f"{SWITCH_COMMIT}^", SWITCH_COMMIT, p), p
    # paths.py: main's two functions appended (the pin predates gen3ai_archive_dir); __init__.py: the F-M7 guard only
    assert [ln for ln in changed(D1_PIN, B.X60T_PIN, "src/utils/paths.py") if ln.startswith("-")] == []
    assert [ln for ln in changed(D1_PIN, B.X60T_PIN, "src/utils/team_corpora/__init__.py")
            if ln.startswith("+") and ln[1:] not in _git("show", f"{SWITCH_COMMIT}:src/utils/team_corpora/__init__.py")
            .splitlines()] == []
    doc = json.loads(_git("show", f"{B.X60T_PIN}:src/utils/team_corpora/human_corpora.json"))
    assert list(doc["corpora"]) == [K.X60T_CORPUS["corpus"]]
    st = doc["corpora"][K.X60T_CORPUS["corpus"]]
    assert st["human_data"] is True and not {"text", "texts", "teams", "rows"} & set(st)
    for k in ("n", "tier", "tier_sha256", "data_sha256", "manifest_sha256", "soak_sha256"):
        assert st[k] == K.X60T_CORPUS[k], k


@needs_pin
@needs_local
def test_the_registered_pin_passes_its_own_check():
    ok, lines = B.x60t_pin_check(B.X60T_PIN, SIDE_PIN, str(repo_root()))
    assert ok and "exactly the 19 enumerated path(s)" in lines[0] and "LOCAL corpus agree" in lines[1], lines
    ok, line = K.corpus_check("x60t", B.X60T_PIN, str(repo_root()))
    assert ok and "22950 battles PASS at env f4b25238def5" in line and "soak_main.json" in line, line


@needs_pin
def test_x60t_pin_check_refuses_the_wrong_pin_a_moved_path_list_and_another_side_pin(monkeypatch):
    root = str(repo_root())
    ok, lines = B.x60t_pin_check(D1_PIN, SIDE_PIN, root)
    assert not ok and "x60t launches only at X60T_PIN" in lines[0], lines
    ok, lines = B.x60t_pin_check(B.X60T_PIN, D1_PIN, root)
    assert not ok and "the side's written pin" in lines[0], lines
    monkeypatch.setattr(B, "X60T_PIN_PATHS", B.X60T_PIN_PATHS[:-1])
    ok, lines = B.x60t_pin_check(B.X60T_PIN, SIDE_PIN, root)
    assert not ok and "EXACTLY" in lines[0] and "human_test.py" in lines[0], lines


@needs_pin
def test_the_human_corpus_check_refuses_a_changed_local_corpus_and_a_pin_without_the_stub(tmp_path):
    """A local corpus whose bytes differ from the stub (here: an empty directory, then a tampered soak record) and a
    pin without the stub (D1's) are refused. No human team text is written: only sha-bearing JSON files."""
    root = str(repo_root())
    ok, line = K.human_corpus_check("x60t", D1_PIN, root, local_dir=str(tmp_path))
    assert not ok and "holds no human_corpora.json" in line, line
    ok, line = K.human_corpus_check("x60t", B.X60T_PIN, root, local_dir=str(tmp_path))
    assert not ok and "is not x60t's registered" in line and "manifest.json None" in line, line


# ------------------------------------------------------------------------------------------ the stages
def _fill_fp32(stage: Path, sha: str = "c" * 40) -> None:
    (stage / B.SIDE_REQUIRED_FILE.format(side="production")).write_text(f"# header\n\n{sha} FP32_ALT the fp32 build\n")


def test_look2_and_c0s_top_up_run_only_on_x60ts_own_look1_continue(tmp_path):
    stage = tmp_path / "stage"
    (stage / B.READS_DIR).mkdir(parents=True)
    st = str(stage)
    assert B.stage_status(st, "x60t", 3501)[0] == B.PENDING                    # AM-8: no FP32_ALT named
    _fill_fp32(stage)
    for seed in (3501, 3502, 3503):
        status, why = B.stage_status(st, "x60t", seed)
        assert status == B.ALLOWED and "AM-X60T" in why and B.stage_of("x60t", seed) == "X60T", why
    for arm, seed, stg in [("x60t", s, "X60Tb") for s in range(3504, 3509)] + [("c0", s, "X60T-C0b")
                                                                                for s in (3506, 3507, 3508)]:
        assert B.stage_status(st, arm, seed)[0] == B.PENDING and B.stage_of(arm, seed) == stg, (arm, seed)
    rec = {"arm": "X60T", "look": 1, "verdict": "CONTINUE"}
    (stage / B.READS_DIR / "x60t_look1.json").write_text(json.dumps(rec))
    assert all(B.stage_status(st, "x60t", s)[0] == B.ALLOWED for s in range(3504, 3509))
    assert all(B.stage_status(st, "c0", s)[0] == B.ALLOWED for s in (3506, 3507, 3508))
    (stage / B.READS_DIR / "x60t_look1.json").write_text(json.dumps(dict(rec, verdict="FUTILE")))
    assert B.stage_status(st, "x60t", 3504)[0] == B.REFUSED and B.stage_status(st, "c0", 3506)[0] == B.REFUSED
    assert B.stage_status(st, "c0", 3504)[0] == B.PENDING                       # C0's batch-1 look 2 is still G1's
    assert "rb_x60t_s3501" in B.side_pinned_run_names() and "rb_x35_c0_s3508" in B.side_pinned_run_names()
