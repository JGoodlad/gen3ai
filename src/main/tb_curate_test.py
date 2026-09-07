"""`gen3_tb_curate_v1` — the curated TensorBoard logdir (a symlink view over `models/*/tb`).

The safety properties are the ones worth pinning: this tool touches ONLY symlinks, ONLY inside the
curated directory, and never a byte under `models/`.
"""
from __future__ import annotations

import json
import os

from main import tb_curate as C


def _make_models(root: str, runs, *, with_tb=True) -> str:
    models = os.path.join(root, "models")
    for r in runs:
        d = os.path.join(models, r, "tb") if with_tb else os.path.join(models, r)
        os.makedirs(d, exist_ok=True)
        if with_tb:
            open(os.path.join(d, "events.out.tfevents.1700000000.host.1.0"), "wb").close()
    return models


def _write_list(root: str, runs) -> str:
    p = os.path.join(root, "tb_curated_runs.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"runs": [{"run": r, "why": "because"} for r in runs]}, f)
    return p


# --------------------------------------------------------------------------------------------
# the desired state
# --------------------------------------------------------------------------------------------
def test_desired_is_the_list_and_targets_the_runs_tb_dir(tmp_path):
    root = str(tmp_path)
    models = _make_models(root, ["alpha", "beta", "gamma"])
    lst = _write_list(root, ["alpha", "gamma"])

    entries, problems = C.desired(models, list_file=lst, include_live=False)

    assert [e["run"] for e in entries] == ["alpha", "gamma"]
    assert entries[0]["target"] == os.path.join(models, "alpha", "tb")
    assert problems == []


def test_a_listed_run_with_no_tb_is_a_reported_problem_not_a_dangling_link(tmp_path):
    """A dangling link makes TensorBoard log an error on every reload; a named gap does not."""
    root = str(tmp_path)
    models = _make_models(root, ["alpha"])
    os.makedirs(os.path.join(models, "no_tb_run"), exist_ok=True)
    lst = _write_list(root, ["alpha", "no_tb_run", "never_existed"])

    entries, problems = C.desired(models, list_file=lst, include_live=False)

    assert [e["run"] for e in entries] == ["alpha"]
    assert len(problems) == 2
    assert any("no_tb_run" in p for p in problems)
    assert any("never_existed" in p for p in problems)


def test_an_absent_list_file_is_not_an_error(tmp_path):
    """A fresh clone has curated nothing; the union with the LIVE runs is still usable."""
    models = _make_models(str(tmp_path), ["alpha"])
    entries, problems = C.desired(models, list_file=os.path.join(str(tmp_path), "nope.json"),
                                  include_live=False)
    assert entries == [] and problems == []


def test_a_bare_string_entry_is_accepted(tmp_path):
    root = str(tmp_path)
    models = _make_models(root, ["alpha"])
    p = os.path.join(root, "l.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"runs": ["alpha"]}, f)

    entries, _ = C.desired(models, list_file=p, include_live=False)

    assert [e["run"] for e in entries] == ["alpha"]


# --------------------------------------------------------------------------------------------
# apply — idempotent, and symlinks only
# --------------------------------------------------------------------------------------------
def test_apply_creates_symlinks_named_for_the_run(tmp_path):
    root = str(tmp_path)
    models = _make_models(root, ["alpha", "beta"])
    lst = _write_list(root, ["alpha", "beta"])
    cdir = os.path.join(root, "tb_curated")
    entries, _ = C.desired(models, list_file=lst, include_live=False)

    added, removed, kept = C.apply(cdir, entries)

    assert sorted(added) == ["alpha", "beta"] and removed == [] and kept == []
    assert os.path.islink(os.path.join(cdir, "alpha"))
    # the run shows in TensorBoard under its own NAME, and the link resolves to its tb/
    assert os.path.realpath(os.path.join(cdir, "alpha")) == \
        os.path.realpath(os.path.join(models, "alpha", "tb"))
    assert os.path.isfile(os.path.join(cdir, "alpha",
                                       "events.out.tfevents.1700000000.host.1.0"))


def test_apply_is_idempotent(tmp_path):
    root = str(tmp_path)
    models = _make_models(root, ["alpha"])
    lst = _write_list(root, ["alpha"])
    cdir = os.path.join(root, "tb_curated")
    entries, _ = C.desired(models, list_file=lst, include_live=False)

    C.apply(cdir, entries)
    added, removed, kept = C.apply(cdir, entries)

    assert added == [] and removed == [] and kept == ["alpha"]


def test_apply_removes_a_delisted_run(tmp_path):
    root = str(tmp_path)
    models = _make_models(root, ["alpha", "beta"])
    cdir = os.path.join(root, "tb_curated")

    entries, _ = C.desired(models, list_file=_write_list(root, ["alpha", "beta"]),
                           include_live=False)
    C.apply(cdir, entries)
    entries, _ = C.desired(models, list_file=_write_list(root, ["alpha"]), include_live=False)
    added, removed, kept = C.apply(cdir, entries)

    assert removed == ["beta"] and kept == ["alpha"]
    assert not os.path.exists(os.path.join(cdir, "beta"))
    # ...and the underlying run is untouched
    assert os.path.isfile(os.path.join(models, "beta", "tb",
                                       "events.out.tfevents.1700000000.host.1.0"))


def test_apply_retargets_a_run_that_moved(tmp_path):
    """Promotion to `_goldens/` moves a run; the link must follow rather than dangle."""
    root = str(tmp_path)
    models = _make_models(root, ["alpha"])
    cdir = os.path.join(root, "tb_curated")
    entries, _ = C.desired(models, list_file=_write_list(root, ["alpha"]), include_live=False)
    C.apply(cdir, entries)

    moved = os.path.join(root, "elsewhere", "alpha", "tb")
    os.makedirs(moved, exist_ok=True)
    added, removed, kept = C.apply(cdir, [{"run": "alpha", "target": moved}])

    assert added == ["alpha"] and kept == []
    assert os.path.realpath(os.path.join(cdir, "alpha")) == os.path.realpath(moved)


def test_apply_never_deletes_a_non_symlink(tmp_path):
    """A real file someone put in the way is none of this tool's business."""
    root = str(tmp_path)
    models = _make_models(root, ["alpha"])
    cdir = os.path.join(root, "tb_curated")
    os.makedirs(cdir, exist_ok=True)
    squatter = os.path.join(cdir, "NOTES.md")
    with open(squatter, "w", encoding="utf-8") as f:
        f.write("hand-written")
    real_dir = os.path.join(cdir, "a_real_dir")
    os.makedirs(real_dir)

    entries, _ = C.desired(models, list_file=_write_list(root, ["alpha"]), include_live=False)
    C.apply(cdir, entries)

    assert os.path.isfile(squatter)
    assert open(squatter, encoding="utf-8").read() == "hand-written"
    assert os.path.isdir(real_dir)
    assert sorted(C.squatters(cdir)) == ["NOTES.md", "a_real_dir"]


def test_apply_never_writes_under_models(tmp_path):
    """The archive is read-only to this tool. Asserted by comparing the whole tree before/after."""
    root = str(tmp_path)
    models = _make_models(root, ["alpha", "beta"])

    def snapshot():
        out = {}
        for dirpath, dirnames, filenames in os.walk(models):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                out[p] = (os.path.getsize(p), os.path.getmtime(p))
            for dn in dirnames:
                out[os.path.join(dirpath, dn)] = "dir"
        return out

    before = snapshot()
    entries, _ = C.desired(models, list_file=_write_list(root, ["alpha"]), include_live=False)
    C.apply(os.path.join(root, "tb_curated"), entries)

    assert snapshot() == before


# --------------------------------------------------------------------------------------------
# live-run detection
# --------------------------------------------------------------------------------------------
def test_live_runs_reads_the_run_name_out_of_a_launcher_argv(monkeypatch):
    import subprocess

    fake = (
        "  111 /usr/bin/python3 -m main.launcher --run-name ai_v12_01_winprob_critic "
        "--restart-interval-hours 3 --device cuda\n"
        "  112 /usr/bin/python3 /tmp/launcher-abc/src/main/train_rl_agent.py --steps 5 "
        "--run-dir models/ai_v12_01_winprob_critic\n"
        "  113 /usr/bin/python3 -m main.launcher --run_name other_run\n"
        "  114 vim notes.txt\n"
        "  115 grep -r main.launcher src/\n"
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=fake, stderr=""))

    live = C.live_runs()

    names = [n for n, _ in live]
    assert "ai_v12_01_winprob_critic" in names
    assert "other_run" in names
    assert len(names) == len(set(names)), "the launcher and its child name ONE run"
    # a grep whose ARGV mentions the marker is a process, not a run — it names no run and drops out
    assert len(names) == 2


def test_live_runs_survives_no_ps(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("ps")))
    assert C.live_runs() == []


def test_a_live_run_is_unioned_in_and_marked(tmp_path, monkeypatch):
    root = str(tmp_path)
    models = _make_models(root, ["alpha", "the_live_one"])
    monkeypatch.setattr(C, "live_runs", lambda: [("the_live_one", 4242)])

    entries, _ = C.desired(models, list_file=_write_list(root, ["alpha"]), include_live=True)

    assert [e["run"] for e in entries] == ["alpha", "the_live_one"]
    live = [e for e in entries if e["run"] == "the_live_one"][0]
    assert live["source"] == "live" and "4242" in live["why"]


def test_a_run_that_is_both_listed_and_live_appears_once(tmp_path, monkeypatch):
    root = str(tmp_path)
    models = _make_models(root, ["alpha"])
    monkeypatch.setattr(C, "live_runs", lambda: [("alpha", 7)])

    entries, _ = C.desired(models, list_file=_write_list(root, ["alpha"]), include_live=True)

    assert len(entries) == 1
    assert entries[0]["source"] == "list+live"


# --------------------------------------------------------------------------------------------
# the committed list itself
# --------------------------------------------------------------------------------------------
def test_the_committed_list_parses_and_every_entry_has_a_reason():
    """The list is the owner's dial — an entry with no `why` is an entry nobody can audit."""
    entries = C.load_list()
    assert entries, "designs/tb_curated_runs.json should not be empty"
    for e in entries:
        assert e["run"] and e["why"], f"{e['run']}: every curated run needs a one-line reason"
    assert len({e["run"] for e in entries}) == len(entries), "no duplicate runs"
