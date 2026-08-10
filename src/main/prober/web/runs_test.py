"""Path confinement for the run picker — the security-critical half of the web app.

The hosted instance is unauthenticated for reading, so the `run` value is attacker-controlled.
These tests are written as ATTACKS rather than as behaviour checks: each one is a thing a visitor
could type, and the assertion is that it selects nothing.

The design being tested is "membership in a server-enumerated set", not "sanitised input" — so the
traversal cases below should be boring, and that is the point. A regression that reintroduces path
joining would light most of this file up at once.
"""

from __future__ import annotations

import os

import pytest

from main.prober.web.runs import RunAccessError, RunStore


def _make_run(root, name, *, traces=True) -> str:
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    if traces:
        os.makedirs(os.path.join(path, "eval_traces", "step_100", "bot"), exist_ok=True)
    with open(os.path.join(path, "metadata.json"), "w") as fh:
        fh.write("{}")
    return path


@pytest.fixture()
def models(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    _make_run(str(root), "run_a")
    _make_run(str(root), "run_b")
    (tmp_path / "outside_secret").mkdir()
    (tmp_path / "outside_secret" / "loot.txt").write_text("do not read me")
    return str(root)


# -- enumeration ------------------------------------------------------------------------

def test_lists_the_runs_and_nothing_else(models, tmp_path):
    (tmp_path / "models" / "not_a_run").mkdir()          # no run markers
    (tmp_path / "models" / "stray.txt").write_text("x")  # not a directory
    names = {r["name"] for r in RunStore(models).list_runs()}
    assert names == {"run_a", "run_b"}


def test_a_run_without_traces_is_listed_but_flagged(models):
    _make_run(models, "run_empty", traces=False)
    rows = {r["name"]: r for r in RunStore(models).list_runs()}
    assert rows["run_empty"]["has_traces"] is False
    assert rows["run_a"]["has_traces"] is True


def test_newest_run_is_the_default(models):
    """Opening the site should land on what you were just training, not an alphabetical accident."""
    import time
    newest = _make_run(models, "run_z")
    future = time.time() + 10_000                        # unambiguously the most recent
    os.utime(newest, (future, future))
    assert RunStore(models).default_run() == "run_z"


# -- the attacks ------------------------------------------------------------------------

@pytest.mark.parametrize("attack", [
    "../outside_secret",
    "../../etc",
    "..",
    ".",
    "/etc/passwd",
    "/etc",
    "run_a/../../outside_secret",
    "run_a/eval_traces",
    "./run_a",
    "run_a/",
    "\\..\\outside_secret",
    "%2e%2e%2foutside_secret",
    "..%2Foutside_secret",
    "run_a\x00.txt",
    "",
    " ",
    "~",
    "~/",
])
def test_no_traversal_string_selects_anything(models, attack):
    """None of these appear in a directory listing, so none of them can resolve. The failure is
    a clean refusal, not an exception from deep inside os.path."""
    with pytest.raises(RunAccessError):
        RunStore(models).resolve(attack)


def test_the_error_never_echoes_the_rejected_input(models):
    """Error text is rendered to the visitor. Echoing the probe back turns the message into an
    oracle for mapping the filesystem."""
    with pytest.raises(RunAccessError) as exc:
        RunStore(models).resolve("../../etc/shadow")
    assert "etc" not in str(exc.value) and "shadow" not in str(exc.value)


def test_a_sibling_of_a_pinned_run_is_not_reachable(models):
    """Pointing the server at ONE run must not widen the root to its parent.

    This is the mistake that would be easy to make — resolving a run dir by taking its dirname as
    the models root — and it would silently expose every other run on the box.
    """
    store = RunStore(os.path.join(models, "run_a"))
    assert [r["name"] for r in store.list_runs()] == ["run_a"]
    assert store.resolve("run_a").endswith("run_a")
    with pytest.raises(RunAccessError):
        store.resolve("run_b")


# -- symlinks ---------------------------------------------------------------------------

def test_a_top_level_symlinked_run_is_followed_and_marked(models, tmp_path):
    """The owner's decision (2026-08-09): the launcher surfaces worktree runs into models/ as
    symlinks, so refusing them would hide the entire current generation. One hop, resolved here,
    at enumeration."""
    target = _make_run(str(tmp_path), "elsewhere_run")
    os.symlink(target, os.path.join(models, "linked_run"))

    rows = {r["name"]: r for r in RunStore(models).list_runs()}
    assert rows["linked_run"]["linked"] is True
    assert rows["run_a"]["linked"] is False
    assert RunStore(models).resolve("linked_run") == os.path.realpath(target)


def test_a_symlink_inside_a_run_refuses_the_whole_run(models, tmp_path):
    """The remaining escape: a link planted inside an otherwise-legitimate run. Refused loudly —
    silently skipping it would leave a run half-readable and nobody the wiser."""
    os.symlink(str(tmp_path / "outside_secret"),
               os.path.join(models, "run_a", "eval_traces", "sneaky"))
    store = RunStore(models)
    assert "run_a" in {r["name"] for r in store.list_runs()}   # still listed...
    with pytest.raises(RunAccessError) as exc:
        store.resolve("run_a")                                  # ...but not openable
    assert "symlink" in str(exc.value)
    assert store.resolve("run_b")                               # its neighbour is unaffected


def test_the_inner_symlink_message_stays_run_relative(models, tmp_path):
    """It names the offending entry so it can be fixed, but as a run-relative path — an absolute
    one would leak the box's layout into a rendered error."""
    os.symlink(str(tmp_path / "outside_secret"), os.path.join(models, "run_a", "sneaky"))
    with pytest.raises(RunAccessError) as exc:
        RunStore(models).resolve("run_a")
    assert "sneaky" in str(exc.value)
    assert str(tmp_path) not in str(exc.value)


def test_a_dangling_top_level_symlink_is_ignored(models, tmp_path):
    os.symlink(str(tmp_path / "nope"), os.path.join(models, "broken"))
    assert "broken" not in {r["name"] for r in RunStore(models).list_runs()}


def test_a_symlink_to_a_non_run_directory_is_ignored(models, tmp_path):
    """It resolves and it is a directory, but it has no run markers — so it is not a run."""
    os.symlink(str(tmp_path / "outside_secret"), os.path.join(models, "loot"))
    assert "loot" not in {r["name"] for r in RunStore(models).list_runs()}


def test_the_audit_is_cached_so_it_does_not_walk_per_request(models, monkeypatch):
    """The walk is ~7k entries on a real run. Doing it on every page load would be the slowest
    thing in the app."""
    import main.prober.web.runs as R

    calls = []
    real = R._first_symlink
    monkeypatch.setattr(R, "_first_symlink", lambda p: (calls.append(p), real(p))[1])
    store = RunStore(models)
    for _ in range(5):
        store.resolve("run_a")
    assert len(calls) == 1


# -- misc -------------------------------------------------------------------------------

def test_a_nonexistent_root_is_refused_at_construction(tmp_path):
    with pytest.raises(RunAccessError):
        RunStore(str(tmp_path / "nope"))


def test_an_empty_models_dir_has_no_default(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    store = RunStore(str(empty))
    assert store.list_runs() == [] and store.default_run() is None
    with pytest.raises(RunAccessError):
        store.resolve(None)


# -- regressions from the 2026-08-09 adversarial review --------------------------------------

def test_an_unreadable_subdirectory_refuses_the_run_rather_than_passing_it(models, tmp_path):
    """`os.walk`'s default silently DISCARDS an OSError and yields nothing for that subtree, so a
    symlink inside a mode-0o000 directory was hidden and the run was ACCEPTED — the audit failing
    open on precisely the input designed to defeat it."""
    hidden = os.path.join(models, "run_a", "hidden")
    os.makedirs(hidden)
    os.symlink(str(tmp_path / "outside_secret"), os.path.join(hidden, "leak"))
    os.chmod(hidden, 0o000)
    try:
        with pytest.raises(RunAccessError) as exc:
            RunStore(models).resolve("run_a")
        assert "symlink" in str(exc.value)
    finally:
        os.chmod(hidden, 0o755)          # so tmp cleanup can remove it


def test_a_walk_failure_is_not_cached_as_a_refusal(models, tmp_path):
    """An unreadable subtree can be transient (a directory mid-delete). Caching the refusal would
    make a recoverable condition permanent until restart."""
    blocked = os.path.join(models, "run_b", "blocked")
    os.makedirs(blocked)
    os.chmod(blocked, 0o000)
    store = RunStore(models)
    try:
        with pytest.raises(RunAccessError):
            store.resolve("run_b")
    finally:
        os.chmod(blocked, 0o755)
    assert store.resolve("run_b"), "the refusal was cached; the run never recovers"


def test_a_pinned_run_whose_name_fails_the_enumeration_pattern_still_resolves(tmp_path):
    """Pinned mode takes the name from the server's own basename, so a space or a leading
    underscore is legitimate. `list_runs()` advertised it while `resolve()` refused it."""
    for odd in ("my run (v8)", "_old_run", ".hidden_run"):
        root = tmp_path / odd
        os.makedirs(root / "eval_traces", exist_ok=True)
        (root / "metadata.json").write_text("{}")
        store = RunStore(str(root))
        name = store.default_run()
        assert name == odd
        assert store.resolve(name) == os.path.realpath(str(root)), odd


def test_enumeration_still_skips_odd_names_in_a_shared_models_dir(models):
    """The pattern was removed from resolve(), not from list_runs() — a shared models/ directory
    should still not advertise something with a newline or a control character in its name."""
    weird = os.path.join(models, "bad\nname")
    os.makedirs(os.path.join(weird, "eval_traces"), exist_ok=True)
    assert "bad\nname" not in {r["name"] for r in RunStore(models).list_runs()}
