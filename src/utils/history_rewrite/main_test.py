"""The CLI: it refuses cleanly, speaks of identities by index, and (with git-filter-repo cached) runs the whole dry run."""
from __future__ import annotations

import os

import pytest

from utils.history_rewrite import __main__ as M
from utils.history_rewrite import names as N
from utils.history_rewrite import rewrite as W
from utils.history_rewrite.synthetic_repo_testkit import ALL_NAMES, make_repo


@pytest.fixture(autouse=True)
def _no_renice(monkeypatch):
    monkeypatch.setattr(M, "_low_priority", lambda cpus: None)       # a test process must not lower its own priority


def _no_names(text: str) -> None:
    assert not any(n.lower() in text.lower() for n in ALL_NAMES), "a name reached the output"


def test_help_lists_every_subcommand(capsys):
    with pytest.raises(SystemExit) as e:
        M.main(["--help"])
    assert e.value.code == 0
    text = capsys.readouterr().out
    for sub in ("fetch-filter-repo", "names", "audit", "rules", "plan", "rewrite", "map", "cite", "verify", "report", "gates", "dryrun"):
        assert sub in text


def test_plan_prints_the_three_steps_and_never_a_force_push(tmp_path, capsys):
    r = make_repo(tmp_path)
    rc = M.main(["plan", "--source", r["path"], "--work", str(tmp_path / "w"), "--rules", str(tmp_path / "rules.txt"),
                 "--filter-repo", "/x/git_filter_repo.py"])
    text = capsys.readouterr().out
    assert rc == 0 and text.index("1. clone") < text.index("2. keep-refs") < text.index("3. repack") < text.index("4. filter-repo")
    assert "--prune-empty never" in text and "--preserve-commit-encoding" in text and "--file-info-callback" in text
    assert "NOT executed by this tool, ever: the force-push" in text
    assert "push" not in text.replace("force-push", "").lower()


def test_a_refusal_is_redacted_and_exits_2(tmp_path, capsys):
    r = make_repo(tmp_path)
    names = tmp_path / "n.json"
    assert M.main(["names", "--source", r["path"], "--introducing", r["intro"], "--out", str(names)]) == 0
    capsys.readouterr()
    work = tmp_path / "used"
    work.mkdir()
    (work / "f").write_text("x")
    rc = M.main(["rewrite", "--source", r["path"], "--introducing", r["intro"], "--names", str(names), "--work", str(work),
                 "--filter-repo", "/does/not/matter.py"])
    err = capsys.readouterr().err
    assert rc == 2 and "REFUSED" in err and "not empty" in err
    _no_names(err)


def test_names_inside_the_repo_are_refused(tmp_path, capsys):
    r = make_repo(tmp_path)
    rc = M.main(["names", "--source", r["path"], "--introducing", r["intro"], "--out", os.path.join(r["path"], "names.json")])
    assert rc == 2 and "outside the repo" in capsys.readouterr().err


@pytest.mark.integration
def test_the_whole_dry_run_through_the_cli_on_a_synthetic_repo(tmp_path, capsys):
    try:
        fr = str(W.fetch_filter_repo(os.path.expanduser("~/.cache/gen3ai/history_rewrite/filter_repo")))
    except Exception as e:
        pytest.skip(f"git-filter-repo is not cached and cannot be downloaded: {type(e).__name__}")
    r = make_repo(tmp_path)
    names = tmp_path / "private" / "names.json"
    assert M.main(["names", "--source", r["path"], "--introducing", r["intro"], "--out", str(names)]) == 0
    rc = M.main(["dryrun", "--source", r["path"], "--introducing", r["intro"], "--names", str(names), "--work", str(tmp_path / "work"),
                 "--filter-repo", fr, "--models", str(tmp_path / "models")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "[PASS] names" in out and "[PASS] graph" in out and "[PASS] changed_set" in out and "[PASS] tip_trees" in out
    assert "every rewritten citation resolves in the clone: YES" in out and "second pass rewrites 0 more" in out
    _no_names(out)
    report = (tmp_path / "work" / "cannot_rewrite_report.md").read_text()
    assert "## 5. Outside this machine" in report
    _no_names(report)
    tsv = (tmp_path / "work" / "history_rewrite_sha_map.tsv").read_text().splitlines()
    assert sum(1 for line in tsv if not line.startswith("#")) == 4 and tsv[0].startswith("# DRY RUN map")
    assert N.load_identities(names)


def test_leakcheck_finds_a_name_in_a_tracked_file_and_redacts_the_report(tmp_path, capsys):
    r = make_repo(tmp_path)
    names = tmp_path / "private" / "names.json"
    assert M.main(["names", "--source", r["path"], "--introducing", r["intro"], "--out", str(names)]) == 0
    capsys.readouterr()
    tree = tmp_path / "wt"
    tree.mkdir()
    (tree / "ok.md").write_text("nothing here\n")
    (tree / "leak.md").write_text("the account ZorblaxPilot again\n")
    import subprocess
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    rc = M.main(["leakcheck", "--source", r["path"], "--tree", str(tree), "--names", str(names)])
    out = capsys.readouterr().out
    assert rc == 1 and "[FAIL] tree_names" in out and "leak.md: #04" in out
    _no_names(out)
    (tree / "leak.md").write_text("clean now\n")
    assert M.main(["leakcheck", "--source", r["path"], "--tree", str(tree), "--names", str(names)]) == 0


def test_an_informational_check_reads_review_not_fail_and_does_not_fail_the_run():
    from utils.history_rewrite import verify as V
    assert V.Check("variants", False, "d", informational=True).line().startswith("[REVIEW]")
    assert V.Check("names", False, "d").line().startswith("[FAIL]") and V.Check("names", True, "d").line().startswith("[PASS]")
