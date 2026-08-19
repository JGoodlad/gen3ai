"""The staleness guard: templates pinned to the process, and the watchdog that replaces it.

THE FAILURE THESE PIN, measured 2026-08-18. The prober service had been up 5 days. Jinja reloads a
changed template from disk; Python cannot reload a changed module. So it was serving NEW templates
against OLD code, and every `/battle` for a fresh run returned HTTP 500 —
`UndefinedError: 'dict object' has no attribute 'win_prob'` — because the template asked for a key
the running `session.py` predated. `Restart=always` never fired: nothing had crashed, systemd
called the unit healthy, and the tunnel in front of it agreed.

Two halves, and both are needed. Pinning the templates means a stale process serves a COHERENT old
page instead of a broken hybrid. The watchdog means it does not stay stale.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap

import pytest
from fastapi.testclient import TestClient

from main.prober.web import fixture_run
from main.prober.web.app import SOURCE_REVISION, create_app

_WATCHDOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "..", "..", "scripts", "workstation",
                         "prober_web_watchdog.sh")
_WATCHDOG = os.path.normpath(_WATCHDOG)


# -- the app half: one revision, pinned ---------------------------------------------------------

def test_the_templates_are_pinned_to_the_process():
    """The defect itself. With `auto_reload` on, a deploy that changes a template AND the code it
    reads from leaves a live process rendering the new template against the old module — which is
    not a stale page but a 500."""
    app = create_app(None, password="test-only-password")
    env = app.state.templates.env
    assert env.auto_reload is False, (
        "Jinja would hot-reload templates from disk while Python stays pinned at process start — "
        "the exact hybrid that 500'd every /battle for five days")


def test_a_template_edited_after_startup_is_not_picked_up(tmp_path):
    """The behavioural half of the same claim: pinning has to be observable, not just configured.

    Renders a page, rewrites the template on disk, renders again — the output must not change.
    Asserting the flag alone would pass against a starlette that ignored it."""
    run = fixture_run.build(str(tmp_path))
    app = create_app(run, password="test-only-password")
    with TestClient(app) as client:
        before = client.get("/battles").text
        template = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "templates", "battles.html")
        original = open(template).read()
        try:
            with open(template, "w") as fh:
                fh.write(original.replace("</h1>", " EDITED-AFTER-STARTUP</h1>", 1))
            after = client.get("/battles").text
        finally:
            with open(template, "w") as fh:
                fh.write(original)
    assert "EDITED-AFTER-STARTUP" not in after
    assert after == before, "the template was re-read from disk mid-process"


def test_health_reports_the_revision_this_process_is_running(tmp_path):
    """What the watchdog compares. It must name the code SERVING requests, so it is captured at
    import and never re-read — a value that refreshed itself would report the repo's revision and
    the comparison would always agree with itself."""
    run = fixture_run.build(str(tmp_path))
    app = create_app(run, password="test-only-password")
    with TestClient(app) as client:
        health = client.get("/api/health").json()
    assert health["revision"] == SOURCE_REVISION
    assert health["jobs_running"] == 0
    # A real checkout resolves to a sha; 'unknown' is the honest answer elsewhere, never a fake one.
    assert health["revision"] == "unknown" or len(health["revision"]) == 40


def test_the_revision_is_keyed_on_the_SOURCE_not_the_working_directory(tmp_path, monkeypatch):
    """The service runs with `WorkingDirectory` at the repo that holds `models/`, which today is
    also where the code lives. Resolving the revision from the CWD would keep working right up
    until someone served a worktree, and then report the wrong repo with no symptom."""
    from main.prober.web import app as app_module

    monkeypatch.chdir(tmp_path)                     # a CWD that is not a git checkout at all
    assert app_module._source_revision() == SOURCE_REVISION


def test_running_jobs_are_counted_so_a_restart_can_defer():
    """`falsify_scan` and `calibration` are minutes of Node re-rolls and a restart kills them."""
    from main.prober.web.jobs import JobRegistry

    import threading

    registry = JobRegistry(max_workers=1)
    try:
        assert registry.n_running() == 0
        started, release = threading.Event(), threading.Event()

        def _slow():
            started.set()
            release.wait(timeout=10)
            return {"ok": True}

        registry.submit("falsify_scan", {}, _slow)
        assert started.wait(timeout=10)
        assert registry.n_running() == 1, "a job in flight must be visible to the watchdog"
        release.set()
    finally:
        registry.shutdown()


# -- the watchdog half: the decision logic ------------------------------------------------------
#
# Driven end to end through the real script — a fake health endpoint (a file served by python's
# http.server would be a second process; a `file://` URL is simpler and curl reads it happily) and
# a stubbed `systemctl` that records what it was asked to do.

def _stub_systemctl(tmp_path, *, is_active: str = "active") -> str:
    """A `systemctl` that reports a state and appends every invocation to a log."""
    path = tmp_path / "systemctl"
    path.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        echo "$@" >> "{tmp_path}/calls.log"
        for a in "$@"; do
          if [ "$a" = "is-active" ]; then echo "{is_active}"; exit 0; fi
        done
        exit 0
        """))
    path.chmod(0o755)
    return str(path)


def _health_file(tmp_path, **fields) -> str:
    path = tmp_path / "health.json"
    path.write_text(json.dumps(fields))
    return path.as_uri()


def _run_watchdog(tmp_path, health_url: str, repo: str, *, dry_run=True, is_active="active"):
    env = dict(os.environ,
               GEN3AI_PROBER_HEALTH=health_url,
               GEN3AI_REPO=repo,
               GEN3AI_SYSTEMCTL=_stub_systemctl(tmp_path, is_active=is_active),
               GEN3AI_PROBER_UNIT="unit-under-test.service")
    cmd = ["bash", _WATCHDOG] + (["--dry-run"] if dry_run else [])
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    calls = tmp_path / "calls.log"
    return proc, (calls.read_text() if calls.exists() else "")


@pytest.fixture()
def repo_at_head(tmp_path):
    """A throwaway git repo, so the test controls what HEAD is."""
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(a, cwd=repo, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")
    (repo / "f").write_text("x")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "one")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return str(repo), head


@pytest.mark.integration
def test_a_current_process_is_left_alone(tmp_path, repo_at_head):
    """The common case, and the one that must never restart: matching revisions."""
    repo, head = repo_at_head
    proc, calls = _run_watchdog(tmp_path, _health_file(tmp_path, revision=head, jobs_running=0),
                                repo)
    assert proc.returncode == 0
    assert "restart" not in calls, f"restarted a process that was already current: {calls!r}"


@pytest.mark.integration
def test_a_stale_process_is_restarted(tmp_path, repo_at_head):
    repo, head = repo_at_head
    stale = "0" * 40
    proc, calls = _run_watchdog(tmp_path, _health_file(tmp_path, revision=stale, jobs_running=0),
                                repo, dry_run=True)
    assert proc.returncode == 0
    assert "STALE" in proc.stdout
    assert head[:12] in proc.stdout, "the log must name the revisions it compared"


@pytest.mark.integration
def test_a_build_with_no_revision_field_counts_as_stale(tmp_path, repo_at_head):
    """A process predating the field is BY DEFINITION older than the code that added it. Reading a
    missing field as 'fine' would make the watchdog blind to exactly the build it was written for
    — the one that was live when this failure was found."""
    repo, _ = repo_at_head
    proc, _ = _run_watchdog(tmp_path, _health_file(tmp_path, ok=True), repo)
    assert proc.returncode == 0
    assert "STALE" in proc.stdout and "unknown" in proc.stdout


@pytest.mark.integration
def test_a_restart_defers_while_a_probe_is_running(tmp_path, repo_at_head):
    """A restart kills a multi-minute Node re-roll. The code is stale either way; discarding a
    probe someone is waiting on is the worse trade, and the next tick retries."""
    repo, _ = repo_at_head
    proc, calls = _run_watchdog(tmp_path,
                                _health_file(tmp_path, revision="0" * 40, jobs_running=1),
                                repo, dry_run=False)
    assert proc.returncode == 0
    assert "deferring" in proc.stdout
    assert "restart" not in calls, "killed a running probe instead of waiting a tick"


@pytest.mark.integration
def test_a_stopped_unit_is_left_stopped(tmp_path, repo_at_head):
    """Starting a unit the operator stopped is not this script's business — and `Restart=always`
    already owns the case where it died."""
    repo, _ = repo_at_head
    proc, calls = _run_watchdog(tmp_path, _health_file(tmp_path, revision="0" * 40), repo,
                                dry_run=False, is_active="inactive")
    assert proc.returncode == 0
    assert "leaving it to systemd" in proc.stdout
    assert "restart" not in calls


@pytest.mark.integration
def test_an_unreachable_service_fails_loudly(tmp_path, repo_at_head):
    """A watchdog that exits 0 when it cannot see the thing it guards is worse than none: the
    journal would show a clean run forever while the service was unreachable."""
    repo, _ = repo_at_head
    proc, _ = _run_watchdog(tmp_path, "http://127.0.0.1:1/api/health", repo)
    assert proc.returncode == 1
    assert "unreachable" in proc.stderr


@pytest.mark.integration
def test_the_units_reference_the_script_that_exists():
    """The committed unit files are the install source of truth; a path typo in them surfaces as a
    watchdog that has silently never run."""
    here = os.path.dirname(_WATCHDOG)
    service = open(os.path.join(here, "gen3ai-prober-web-watchdog.service")).read()
    exec_line = [ln for ln in service.splitlines() if ln.startswith("ExecStart=")]
    assert exec_line, "the unit has no ExecStart"
    script = exec_line[0].split("=", 1)[1].strip()
    assert os.path.basename(script) == os.path.basename(_WATCHDOG)
    assert shutil.which("bash"), "the script needs bash"
    timer = open(os.path.join(here, "gen3ai-prober-web-watchdog.timer")).read()
    assert "OnUnitActiveSec=" in timer and "WantedBy=timers.target" in timer
