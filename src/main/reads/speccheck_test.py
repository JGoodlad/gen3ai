"""``check-spec`` and the pin (belief-prior-mix 71c76dfb FINDING 8: a spec's `pin` must exist before the spec, so a PREREG
and its specs landed in two commits). A pin is an EXISTING commit (the one that lands the code the read plays); the
registration commit (PREREG + spec) is a later one. `--pin-pending` reviews a spec before its pin is decided. A miniature
git repo carries the commits."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.reads import cli
from main.reads.playpath_test import BASE, base_of, commit_on, git, write
from main.reads.spec import SpecError
from main.reads.spec_test import base as spec_base
from main.reads.speccheck import check_spec


@pytest.fixture
def mini_repo(tmp_path) -> Path:
    """A miniature git repo shaped like the real layout (``playpath_test.BASE``): the commits a pin names."""
    repo = tmp_path / "mini"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "core.fileMode", "true")
    write(repo, dict(BASE))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def _write(tmp_path: Path, pin=None, **extra) -> Path:
    d = spec_base()
    if pin is not None:
        d["pin"] = pin
    d.update(extra)
    p = tmp_path / "read_spec.json"
    p.write_text(json.dumps(d))
    return p


def test_a_spec_without_a_pin_checks_as_unpinned_and_final(tmp_path, mini_repo):
    out = check_spec(_write(tmp_path), repo=mini_repo)
    assert out["pin"] is None and out["sha256_final"] is True and out["read"] == "demo_read_2026-10-25"
    assert out["allow_partial_at_stop"] is False
    assert (out["rederived_guards"], out["choices_guard"]) == ("count", "require")   # a spec this change never saw


def test_a_pin_must_name_a_commit_that_exists_before_the_spec(tmp_path, mini_repo):
    with pytest.raises(SpecError, match=r"pin deadbeef.*names no commit.*--pin-pending"):
        check_spec(_write(tmp_path, pin="deadbeef"), repo=mini_repo)
    base = base_of(mini_repo)
    out = check_spec(_write(tmp_path, pin=base[:12]), repo=mini_repo)
    assert out["pin"]["commit"] == base and out["pin"]["is_head"] is True and out["pin"]["play_path_differs_at_head"] == 0


def test_the_registration_commit_may_differ_from_the_pin_only_off_the_play_path(tmp_path, mini_repo):
    """The one-commit flow: the pin is the code commit; PREREG + spec land after it. The check says how far the play path
    at HEAD has moved from the pin (0 = reviewing HEAD's code is reviewing what plays)."""
    pin = base_of(mini_repo)
    reg = commit_on(mini_repo, pin, {"designs/tools/reads.md": "# registered\n", "src/main/reads/runner.py": "SCHED = 2\n"})
    out = check_spec(_write(tmp_path, pin=pin), repo=mini_repo)
    assert out["pin"]["head"] == reg and out["pin"]["is_head"] is False
    assert out["pin"]["play_path_differs_at_head"] == 0 and out["pin"]["files"] == []
    moved = commit_on(mini_repo, reg, {"src/utils/sibling.py": "S = 2\n"})
    out = check_spec(_write(tmp_path, pin=pin), repo=mini_repo)
    assert out["pin"]["head"] == moved and out["pin"]["play_path_differs_at_head"] == 1
    assert out["pin"]["files"] == ["changed src/utils/sibling.py"]


def test_pin_pending_reviews_a_spec_that_has_no_pin_yet_and_marks_it_not_final(tmp_path, mini_repo):
    out = check_spec(_write(tmp_path), pin_pending=True, repo=mini_repo)
    assert out["pin"]["state"] == "PENDING" and out["sha256_final"] is False
    assert out["pin"]["candidate_head"] == base_of(mini_repo) and "never the one that adds the spec" in out["pin"]["next"]
    assert out["read"] == "demo_read_2026-10-25" and out["anchor"] == ["u"], "everything else is validated as usual"
    pinned = _write(tmp_path, pin=base_of(mini_repo))
    with pytest.raises(SpecError, match="already carries pin"):
        check_spec(pinned, pin_pending=True, repo=mini_repo)


def test_pin_pending_still_refuses_a_malformed_spec(tmp_path, mini_repo):
    p = _write(tmp_path, decision={"rule": "vibes", "params": {}})
    with pytest.raises(SpecError, match="none of"):
        check_spec(p, pin_pending=True, repo=mini_repo)


def test_the_cli_check_spec_modes(tmp_path, capsys):
    p = _write(tmp_path)
    assert cli.main(["check-spec", "--spec", str(p), "--pin-pending"]) == 0
    assert json.loads(capsys.readouterr().out)["pin"]["state"] == "PENDING"
    assert cli.main(["check-spec", "--spec", str(p)]) == 0
    assert json.loads(capsys.readouterr().out)["pin"] is None
    assert cli.main(["check-spec", "--spec", str(_write(tmp_path, pin="abcdef0")), "--pin-pending"]) == 2
    assert "already carries pin" in capsys.readouterr().err
    assert cli.main(["check-spec", "--spec", str(_write(tmp_path, pin="abcdef0"))]) == 2
    assert "names no commit" in capsys.readouterr().err


def test_a_spec_pin_registered_at_an_old_sha_checks_through_the_history_rewrite_map(tmp_path, mini_repo, monkeypatch):
    """The 7 content-pinned `read_spec*.json` keep their one old sha: `check-spec` must find its commit."""
    from utils.git_sha_testkit import fake_old, use_map

    base = base_of(mini_repo)
    use_map(monkeypatch, tmp_path, {fake_old("a"): base})
    out = check_spec(_write(tmp_path, pin=fake_old("a")[:12]), repo=mini_repo)
    assert out["pin"]["commit"] == base and out["pin"]["is_head"] is True
