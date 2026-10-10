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


# -- the picker's classification (2026-10-09): current / older / no traces ---------------------------------
# The owner: "clean up the model selection; 99 % are irrelevant". On a real archive 38 of 322 runs had eval
# traces and none was at the current architecture; the rest were skeletons. The store now classifies each run
# CHEAPLY (model_config.json + the newest eval manifest; no checkpoint is opened) so the picker, `/api/runs` and
# the default run share one answer.

import json  # noqa: E402
import time  # noqa: E402

from agents.model.model_version import (  # noqa: E402
    ARCH_SIGNATURE, MODEL_CONFIG_VERSION, OBS_SEMANTICS_VERSION,
)

_HEAD_REC = {"config_version": MODEL_CONFIG_VERSION, "arch_signature": ARCH_SIGNATURE, "total_dim": 2845}
_OLD_REC = {"config_version": 143, "arch_signature": "gen3_event_record_v2", "total_dim": 2761}


def _arch_run(root, name, rec=None, *, traces=True, age=0.0):
    """A run recording `rec` (its config version + signature) in model_config.json and in its eval manifest,
    with `traces` or as a SKELETON (weights and logs only), last touched `age` seconds ago."""
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    if rec is not None:
        with open(os.path.join(path, "model_config.json"), "w") as fh:
            json.dump(rec, fh)
    if traces:
        step = os.path.join(path, "eval_traces", "step_100")
        os.makedirs(os.path.join(step, "bot"), exist_ok=True)
        with open(os.path.join(step, "eval_manifest.json"), "w") as fh:
            json.dump(dict(rec or {}, step=100), fh)
    else:
        with open(os.path.join(path, "final_model.zip"), "wb") as fh:
            fh.write(b"not a zip: a skeleton's weights are never opened to list it")
    when = time.time() - age
    os.utime(path, (when, when))
    return path


@pytest.fixture()
def archive(tmp_path):
    """One of each: a HEAD-architecture run with traces, an older one with traces, a skeleton, a run that has
    not finished its first eval cycle, and a traced run that records nothing."""
    root = str(tmp_path / "archive")
    os.makedirs(root)
    _arch_run(root, "rb_head", _HEAD_REC, age=5000)
    _arch_run(root, "rb_old", _OLD_REC, age=4000)
    _arch_run(root, "ai_skeleton", {"config_version": 107, "arch_signature": "gen3_critic_route_wave_v1"},
              traces=False, age=3000)
    _arch_run(root, "rb_live", _HEAD_REC, traces=False, age=10)           # NEWEST: a launch with no traces yet
    _arch_run(root, "rb_unrecorded", None, age=6000)
    return root


def test_every_run_lands_in_one_of_three_tiers(archive):
    rows = {r["name"]: r for r in RunStore(archive).list_runs()}
    assert {n: r["tier"] for n, r in rows.items()} == {
        "rb_head": "current", "rb_old": "older", "rb_unrecorded": "older",
        "ai_skeleton": "no_traces", "rb_live": "no_traces"}
    assert rows["rb_head"]["model_views"] is True and rows["rb_head"]["arch"]["status"] == "current"
    assert rows["rb_old"]["model_views"] is False and rows["rb_old"]["arch"]["kind"] == "arch_signature"
    assert rows["rb_old"]["arch"]["config_version"] == 143
    assert rows["rb_unrecorded"]["arch"]["status"] == "unrecorded", "never counted as current"
    assert rows["ai_skeleton"]["has_traces"] is False and rows["rb_head"]["has_traces"] is True


def test_the_semantics_only_class_keeps_a_run_out_of_the_current_tier(tmp_path):
    """A run one version below the obs-semantics marker has HEAD's signature and HEAD's width — every shape
    fits — yet its model views cannot run. The picker must not offer it as 'current'."""
    root = str(tmp_path / "archive")
    os.makedirs(root)
    _arch_run(root, "rb_prev", dict(_HEAD_REC, config_version=OBS_SEMANTICS_VERSION - 1))
    (row,) = RunStore(root).list_runs()
    assert row["tier"] == "older" and row["arch"]["kind"] == "obs_semantics"


def test_an_empty_eval_traces_directory_is_not_traces(tmp_path):
    root = str(tmp_path / "archive")
    path = _arch_run(root, "rb_empty", _HEAD_REC, traces=False)
    os.makedirs(os.path.join(path, "eval_traces"))
    (row,) = RunStore(root).list_runs()
    assert row["has_traces"] is False and row["tier"] == "no_traces"


def test_the_default_is_the_newest_current_run_even_when_newer_runs_exist(archive):
    """`rb_live` (no traces yet) is the newest by mtime and `rb_old` is newer than `rb_head` — the old rule
    opened the former. The rule is: newest CURRENT run with traces."""
    assert RunStore(archive).default_run() == "rb_head"


def test_with_no_current_run_the_default_is_the_newest_run_that_has_traces(archive):
    import shutil
    shutil.rmtree(os.path.join(archive, "rb_head"))
    assert RunStore(archive).default_run() == "rb_old"


def test_with_no_traces_anywhere_the_default_is_the_newest_run(tmp_path):
    root = str(tmp_path / "archive")
    _arch_run(root, "a_skeleton", _OLD_REC, traces=False, age=100)
    _arch_run(root, "b_skeleton", _OLD_REC, traces=False, age=5)
    assert RunStore(root).default_run() == "b_skeleton"


def test_a_run_the_picker_hides_is_still_openable(archive):
    """Hiding is PRESENTATION. Membership in the server's own enumeration is the security gate, and it is
    over EVERY run — a deep link to a skeleton (or a run mid-launch) still resolves."""
    store = RunStore(archive)
    assert store.resolve("ai_skeleton").endswith("ai_skeleton")
    assert store.resolve("rb_live").endswith("rb_live")


def test_listing_never_opens_a_checkpoint(archive, monkeypatch):
    """The classification is two small JSON reads per run. The skeleton's `final_model.zip` is junk bytes and
    the zip machinery is booby-trapped, so a listing that touched either would raise."""
    import zipfile

    def trap(*a, **k):
        raise AssertionError("listing a run opened a zip")

    monkeypatch.setattr(zipfile, "ZipFile", trap)
    monkeypatch.setattr(zipfile, "is_zipfile", trap)
    assert len(RunStore(archive).list_runs()) == 5


def test_the_classification_is_cached_by_mtime(archive, monkeypatch):
    from main.prober import arch_status as A

    store = RunStore(archive)
    store.list_runs()
    reads = []
    real = A._read_json
    monkeypatch.setattr(A, "_read_json", lambda p: (reads.append(p), real(p))[1])
    store.list_runs()
    assert reads == [], "an untouched archive must be served from the cache, not re-read"
