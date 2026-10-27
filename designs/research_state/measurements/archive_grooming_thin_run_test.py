"""``archive_grooming_dryrun.py --thin-run``: the per-run F22 thinning, on a SYNTHETIC tree.

Nothing here touches the real ``models/`` archive.  Like its sibling test file this lives
under ``designs/`` (``pytest.ini`` collects only ``src`` and ``tools``), so it is run directly:

    python -m pytest designs/research_state/measurements/archive_grooming_thin_run_test.py -q

(in a linked worktree, first: export PYTHONPATH=$PWD/src)
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import archive_grooming_dryrun as ag  # noqa: E402

FORK = 380_000_195
RUN = "rb_x35_test_s3501"


def _w(path: str, blob: bytes = b"x") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)


def make_fork(root: str, name: str = RUN, *, final: bool = True, fork_step: int = FORK) -> str:
    """A finished 30M fork: checkpoints 1M apart at <k>M + a small offset, sidecars, the rest."""
    run = os.path.join(root, name)
    for k in range(1, 31):
        step = (fork_step // 1_000_000 + k) * 1_000_000 + (k * 7) % 200
        _w(os.path.join(run, "checkpoints", f"checkpoint_{step}_steps.zip"), f"z{step}".encode())
        _w(os.path.join(run, "checkpoints", f"checkpoint_{step}_steps.json"), b'{"lr": 1}')
    _w(os.path.join(run, "metadata.json"), json.dumps({"lineage": {"fork_step": fork_step}}).encode())
    _w(os.path.join(run, "tb", "events.out.tfevents.1"), b"tb")
    _w(os.path.join(run, "eval_traces", "step_1", "b0.json"), b"{}")
    _w(os.path.join(run, "snapshots", "snapshot_1.zip"), b"s")
    _w(os.path.join(run, "eval_results.jsonl"), b"{}\n")
    if final:
        _w(os.path.join(run, "final_model.zip"), b"final")
    return run


def zips(run: str) -> "list[int]":
    return sorted(ag.checkpoint_step(n) for n in os.listdir(os.path.join(run, "checkpoints"))
                  if n.endswith(".zip"))


def make_x35(root: str, protect: "list[str]" = ()) -> str:
    d = os.path.join(root, "x35")
    _w(os.path.join(d, "reference_population_production.json"),
       json.dumps({"a_block": [{"path": p} for p in protect], "sha": "ab"}).encode())
    _w(os.path.join(d, "x35_pool_batch1_toy.json"), json.dumps(
        {"parent_run_dir": "/x/models/rb_parent", "snapshots": [{"name": "snapshot_9.zip"}]}).encode())
    return d


def tree(run: str) -> "dict[str, str]":
    out = {}
    for r, _d, fs in os.walk(run):
        for f in fs:
            p = os.path.join(r, f)
            out[os.path.relpath(p, run)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def plan(run: str, protected: "set[str]" = frozenset(), **kw) -> dict:
    return ag.plan_thin_run(run, set(protected), **kw)


# ------------------------------------------------------------ selection ----


def test_keeps_exactly_the_10m_nodes_the_final_checkpoint_and_nothing_else(tmp_path):
    run = make_fork(str(tmp_path))
    p = plan(run)
    kept = sorted(ag.checkpoint_step(n) for n in p["keep"] if n.endswith(".zip"))
    # fork 380.0M, 30 checkpoints to 410M: the 390M, 400M, 410M nodes; 410M is also the last
    assert [k // 1_000_000 for k in kept] == [390, 400, 410]
    assert len(p["delete"]) == 27
    assert p["missing_nodes"] == []
    assert "390M node" in p["keep"][f"checkpoint_{kept[0]}_steps.zip"]
    assert "final checkpoint" in p["keep"][f"checkpoint_{kept[-1]}_steps.zip"]


def test_the_node_is_the_first_checkpoint_at_or_past_the_multiple(tmp_path):
    # 399000003 sits JUST BELOW 400M and 400000135 just past it: the node is the latter
    nodes, _ = ag.thin_nodes([399_000_003, 400_000_135, 401_000_221], 0)
    assert nodes == {400_000_000: 400_000_135}


def test_a_node_with_no_checkpoint_in_the_window_is_reported_not_guessed(tmp_path):
    steps = [s * 1_000_000 + 50 for s in range(381, 390)] + [s * 1_000_000 + 50 for s in range(391, 395)]
    nodes, missing = ag.thin_nodes(steps, FORK)
    assert 390_000_000 in missing and 390_000_000 not in nodes


def test_absolute_and_fork_relative_multiples_differ_for_an_unaligned_fork():
    steps = [s * 1_000_000 + 10 for s in range(386, 410)]
    a, _ = ag.thin_nodes(steps, 385_000_100, basis="absolute")
    f, _ = ag.thin_nodes(steps, 385_000_100, basis="fork")
    assert sorted(a) == [390_000_000, 400_000_000]
    assert sorted(f) == [395_000_000, 405_000_000]


def test_the_fork_start_is_never_this_runs_node():
    nodes, _ = ag.thin_nodes([380_000_195 + 1_000_000 * k for k in range(1, 5)], 380_000_195)
    assert 380_000_000 not in nodes


def test_every_checkpoint_sidecar_is_kept_even_when_its_zip_is_deleted(tmp_path):
    run = make_fork(str(tmp_path))
    p = plan(run)
    sidecars = [n for n in os.listdir(os.path.join(run, "checkpoints")) if n.endswith(".json")]
    assert len(sidecars) == 30
    assert all(p["keep"][n] == ag.SIDECAR_KEEP_REASON for n in sidecars)
    assert not any(n.endswith(".json") for n in p["delete"])
    ag.apply_thin_run(p, str(tmp_path / "log"))
    assert sorted(n for n in os.listdir(os.path.join(run, "checkpoints")) if n.endswith(".json")) \
        == sorted(sidecars)


def test_the_latest_txt_pin_is_kept(tmp_path):
    run = make_fork(str(tmp_path))
    pin = sorted(n for n in os.listdir(os.path.join(run, "checkpoints")) if n.endswith(".zip"))[4]
    _w(os.path.join(run, "latest.txt"), f"checkpoints/{pin}".encode())
    assert "latest.txt pin" in plan(run)["keep"][pin]


# -------------------------------------------------- protected references ----


def test_a_file_named_by_a_reference_population_is_kept_and_is_deleted_without_it(tmp_path):
    run = make_fork(str(tmp_path))
    victim = sorted(n for n in os.listdir(os.path.join(run, "checkpoints")) if n.endswith(".zip"))[2]
    assert victim in plan(run)["delete"]
    # the reference names it under ANOTHER models root: the <run>/checkpoints/<file> key matches
    x35 = make_x35(str(tmp_path), [f"/elsewhere/models/{RUN}/checkpoints/{victim}"])
    keys, files = ag.protected_reference_paths(x35)
    p = plan(run, keys)
    assert victim not in p["delete"] and "protected X35 reference" in p["keep"][victim]
    assert len(files) == 2                       # the reference population AND the pool record


def test_the_pool_records_snapshots_are_protected_paths(tmp_path):
    keys, _ = ag.protected_reference_paths(make_x35(str(tmp_path)))
    assert "rb_parent/snapshots/snapshot_9.zip" in keys


def test_missing_references_fail_closed(tmp_path):
    with pytest.raises(ag.ThinRefused, match="reference_population"):
        ag.protected_reference_paths(str(tmp_path / "nope"))
    bad = tmp_path / "bad"
    _w(str(bad / "reference_population_x.json"), b"not json")
    with pytest.raises(ag.ThinRefused, match="unreadable"):
        ag.protected_reference_paths(str(bad))


# ------------------------------------------------------------- refusals ----


def check(run: str, tmp, **kw):
    kw.setdefault("baseline_set", set())
    kw.setdefault("live", {})
    kw.setdefault("open_pids", [])
    ag.check_thinnable(run, str(tmp), **kw)


def test_a_finished_unreferenced_run_passes_every_check(tmp_path):
    check(make_fork(str(tmp_path)), tmp_path)


@pytest.mark.parametrize("name", ["rb_x26_s2601", "rb_x34_anything", "rb_toy_ent_s4002_deep"])
def test_x26_and_x34_runs_are_refused(tmp_path, name):
    with pytest.raises(ag.ThinRefused, match="X26 / X34"):
        check(make_fork(str(tmp_path), name), tmp_path)


def test_a_baseline_run_is_refused(tmp_path):
    run = make_fork(str(tmp_path))
    with pytest.raises(ag.ThinRefused, match="baselines.json"):
        check(run, tmp_path, baseline_set={RUN})


def test_the_real_registry_names_its_runs():
    names = ag.baseline_runs()
    assert "rb_es_end2_s2005" in names                       # `production`'s run


def test_a_run_named_by_a_live_process_is_refused(tmp_path):
    with pytest.raises(ag.ThinRefused, match="launcher / trainer"):
        check(make_fork(str(tmp_path)), tmp_path, live={RUN: "python -m main.launcher ..."})


def test_a_process_with_the_run_dir_open_is_refused_for_real(tmp_path):
    run = make_fork(str(tmp_path))
    proc = subprocess.Popen(["sleep", "30"], cwd=run)
    try:
        with pytest.raises(ag.ThinRefused, match=rf"pid\(s\) {proc.pid} have the run dir open: {proc.pid} \[sleep 30\]"):
            ag.check_thinnable(run, str(tmp_path), baseline_set=set(), live={})   # the holder's cmdline is named
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait()
    ag.check_thinnable(run, str(tmp_path), baseline_set=set(), live={})   # closed again: passes


def test_a_live_launcher_pid_file_is_refused_and_a_dead_one_is_not(tmp_path):
    run = make_fork(str(tmp_path))
    proc = subprocess.Popen(["sleep", "30"])
    pf = str(tmp_path / "run.pid")
    try:
        _w(pf, str(proc.pid).encode())
        with pytest.raises(ag.ThinRefused, match="is alive"):
            check(run, tmp_path, pid_files=[pf])
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait()
    check(run, tmp_path, pid_files=[pf])                     # reaped: the pid is dead


def test_an_unfinished_run_is_refused_unless_allowed(tmp_path):
    run = make_fork(str(tmp_path), final=False)
    with pytest.raises(ag.ThinRefused, match="FINISHED"):
        check(run, tmp_path)
    check(run, tmp_path, allow_no_final=True)


def test_a_dir_without_checkpoints_is_refused(tmp_path):
    os.makedirs(tmp_path / "empty")
    with pytest.raises(ag.ThinRefused, match="not a run dir"):
        check(str(tmp_path / "empty"), tmp_path)


# ---------------------------------------------------------------- apply ----


def _cli(tmp_path, *extra):
    x35 = make_x35(str(tmp_path))
    return ag.main(["--thin-run", RUN, "--models-dir", str(tmp_path), "--repo-root", str(tmp_path),
                    "--x35-dir", x35, "--log-dir", str(tmp_path / "log"), *extra])


def test_the_cli_is_a_dry_run_by_default(tmp_path, capsys):
    run = make_fork(str(tmp_path))
    before = tree(run)
    assert _cli(tmp_path) == 0
    assert tree(run) == before and not (tmp_path / "log").exists()
    assert "NOTHING WAS DELETED" in capsys.readouterr().out


def test_apply_deletes_exactly_the_planned_zips_and_shas_the_kept_files(tmp_path, capsys):
    run = make_fork(str(tmp_path))
    before = tree(run)
    p = plan(run)
    assert _cli(tmp_path, "--apply") == 0
    after = tree(run)
    gone = set(before) - set(after)
    assert gone == {os.path.join("checkpoints", n) for n in p["delete"]}
    assert all(after[k] == before[k] for k in after)         # nothing else changed
    assert all(before[os.path.join("checkpoints", n)] for n in p["keep"])
    log = json.load(open(next((tmp_path / "log").iterdir())))
    assert log["sha_mismatch"] == [] and log["leftovers"] == []
    assert log["sha256_kept"]["final_model.zip"] == before["final_model.zip"]
    assert len(log["removed"]) == 27


def test_apply_refuses_a_live_run_through_the_cli(tmp_path, monkeypatch, capsys):
    run = make_fork(str(tmp_path))
    before = tree(run)
    monkeypatch.setattr(ag, "live_run_dirs", lambda _m: {RUN: "python -m main.launcher"})
    assert _cli(tmp_path, "--apply") == 3
    assert tree(run) == before
    assert "THIN REFUSED" in capsys.readouterr().err


def test_apply_never_deletes_a_symlink_or_a_non_zip(tmp_path):
    run = make_fork(str(tmp_path))
    p = plan(run)
    ck = os.path.join(run, "checkpoints")
    victim = os.path.join(ck, p["delete"][0])
    os.remove(victim)
    os.symlink(os.path.join(run, "final_model.zip"), victim)
    with pytest.raises(ag.UnsafeDeletion, match="symlink"):
        ag.apply_thin_run(p, str(tmp_path / "log"))
    p["delete"] = [p["delete"][1].replace(".zip", ".json")]
    with pytest.raises(ag.UnsafeDeletion, match="not a checkpoint zip"):
        ag.apply_thin_run(p, str(tmp_path / "log"))


def test_a_replan_after_a_thinning_deletes_nothing_more(tmp_path):
    run = make_fork(str(tmp_path))
    ag.apply_thin_run(plan(run), str(tmp_path / "log"))
    again = plan(run)
    assert again["delete"] == [] and again["missing_nodes"] == []
    assert [s // 1_000_000 for s in zips(run)] == [390, 400, 410]
