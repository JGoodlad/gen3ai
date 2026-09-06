"""THE LINEAGE BLOCK — who forked whom, recorded once and never re-derived.

The claims these tests exist for, in order of how much a regression would cost:

1. **A FORK writes the block.** Parent identity, teachers, target, ancestry.
2. **A RESTART preserves it BYTE-FOR-BYTE.** This is the whole feature. The launcher swaps
   `--model` to the fork's OWN drifted checkpoint on every relaunch, so a block re-derived on a
   restart would silently re-point the recorded parent at the student — the exact failure the
   distill anchor has a module of prose defending against.
3. **A FRESH run writes the explicit null form**, because "no block" and "no parent" are
   different facts.
4. **A LEGACY run derives, warns, and says it derived.** Every run on disk today is legacy.

Milliseconds — no torch, no model, nothing outside `tmp_path` (plus one `main_models_dir()` read
that skips when there is no archive).
"""
import json
import os
import zipfile

import pytest

from agents.model.snapshot import save_model_snapshot
from agents.training.lineage import (
    LINEAGE_SCHEMA, ForkParent, ancestry, ancestry_from_parent, ancestry_stop, build_lineage,
    build_lineage_from_command, check_links, checkpoint_num_timesteps, describe_model, fork_parent,
    parse_command, read_block, role_for, role_of, sha256_file, teacher_paths,
)


class _StubVersion:
    """`save_model_snapshot` calls exactly one method on its `version`."""

    def to_json(self) -> str:
        return json.dumps({"arch_signature": "test_arch_v1", "config_version": 7})


def _fake_checkpoint(path, num_timesteps: int) -> str:
    """An SB3-shaped zip: the python-level state is a plain-JSON `data` member."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", json.dumps({"num_timesteps": num_timesteps}))
        z.writestr("policy.pth", b"not-really-weights")
    return str(path)


def _run(root, name, *, command=None, lineage=None, arch="test_arch_v1", git="deadbeef",
         steps=1000):
    """A run directory with a metadata.json, a model_config.json and one final_model.zip."""
    d = os.path.join(str(root), name)
    os.makedirs(d, exist_ok=True)
    meta = {"git_hash": git}
    if command is not None:
        meta["original_command"] = command
    if lineage is not None:
        meta["lineage"] = lineage
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(d, "model_config.json"), "w") as f:
        json.dump({"arch_signature": arch, "config_version": 42}, f)
    _fake_checkpoint(os.path.join(d, "final_model.zip"), steps)
    return d


# ---------------------------------------------------------------------------
# 1 — a FORK writes the block
# ---------------------------------------------------------------------------


def test_a_fork_records_its_parent_teachers_and_target(tmp_path):
    parent = _run(tmp_path, "the_parent", steps=25_000_000)
    t1 = _run(tmp_path, "teacher_one")
    target = _run(tmp_path, "the_target")
    child = os.path.join(str(tmp_path), "the_child")
    os.makedirs(child)

    block = build_lineage(
        model_path=os.path.join(parent, "final_model.zip"), model_dir=child,
        exploiter=os.path.join(target, "final_model.zip"),
        distill_teacher=f"{os.path.join(t1, 'final_model.zip')}:*", fork_step=25_000_000)

    assert block["schema"] == LINEAGE_SCHEMA
    assert block["role"] == "exploiter"            # --exploiter wins over --distill-teacher
    assert block["fork_step"] == 25_000_000
    fp = block["fork_parent"]
    assert fp["run_name"] == "the_parent"
    assert fp["git_hash"] == "deadbeef"
    assert fp["arch_signature"] == "test_arch_v1"
    assert fp["model_config_version"] == 42
    assert fp["num_timesteps"] == 25_000_000
    assert fp["sha256"] == sha256_file(os.path.join(parent, "final_model.zip"))
    assert fp["created_at"]
    assert [t["run_name"] for t in block["teachers"]] == ["teacher_one"]
    assert block["exploiter_target"]["run_name"] == "the_target"


def test_role_follows_the_flags(tmp_path):
    assert role_for(model_path=None, exploiter=None, distill_teacher=None) == "fresh"
    assert role_for(model_path="p.zip", exploiter=None, distill_teacher=None) == "fork"
    assert role_for(model_path="p.zip", exploiter=None, distill_teacher="t:*") == "fold"
    assert role_for(model_path="p.zip", exploiter="x.zip", distill_teacher="t:*") == "exploiter"


def test_num_timesteps_comes_from_the_zip_and_falls_back_to_the_filename(tmp_path):
    z = _fake_checkpoint(os.path.join(str(tmp_path), "checkpoint_777_steps.zip"), 999)
    assert checkpoint_num_timesteps(z) == 999            # the zip wins
    junk = os.path.join(str(tmp_path), "checkpoint_1234_steps.zip")
    with open(junk, "wb") as f:
        f.write(b"not a zip at all")
    assert checkpoint_num_timesteps(junk) == 1234        # the filename convention is the fallback
    assert checkpoint_num_timesteps(os.path.join(str(tmp_path), "final_model.zip")) is None


def test_an_unresolvable_parent_still_records_the_path_it_was_given(tmp_path):
    got = describe_model("models/a_run_that_never_existed/final_model.zip", hash_file=False)
    assert got.path == "models/a_run_that_never_existed/final_model.zip"
    assert got.num_timesteps is None and got.sha256 is None


# ---------------------------------------------------------------------------
# 2 — a RESTART preserves it, byte for byte
# ---------------------------------------------------------------------------


def test_a_same_run_restart_builds_NOTHING(tmp_path):
    """The predicate is `fork_lr.is_same_run_checkpoint`, imported — not a second copy."""
    run = os.path.join(str(tmp_path), "my_run")
    os.makedirs(os.path.join(run, "checkpoints"))
    ckpt = _fake_checkpoint(os.path.join(run, "checkpoints", "checkpoint_5_steps.zip"), 5)
    assert build_lineage(model_path=ckpt, model_dir=run) is None


def test_the_block_survives_a_restart_byte_for_byte(tmp_path):
    """save_model_snapshot's existing-value-wins rule, the same one `original_command` uses."""
    run = str(tmp_path / "run")
    original = {"schema": LINEAGE_SCHEMA, "role": "fold", "fork_parent": {"path": "PARENT.zip"},
                "fork_step": 11, "teachers": [], "exploiter_target": None, "ancestry": []}
    save_model_snapshot(run, _StubVersion(), git_hash="a", lineage=original)
    # A restart: a DIFFERENT parent offered (the launcher's swapped --model), and no block at all.
    save_model_snapshot(run, _StubVersion(), git_hash="b",
                        lineage={"role": "fork", "fork_parent": {"path": "THE-STUDENT.zip"}})
    save_model_snapshot(run, _StubVersion(), git_hash="c", lineage=None)
    with open(os.path.join(run, "metadata.json")) as f:
        got = json.load(f)["lineage"]
    assert got == original


def test_a_run_with_no_block_stays_without_one(tmp_path):
    run = str(tmp_path / "run")
    save_model_snapshot(run, _StubVersion(), git_hash="a")
    with open(os.path.join(run, "metadata.json")) as f:
        assert "lineage" not in json.load(f)


# ---------------------------------------------------------------------------
# 3 — a FRESH run writes the explicit null form
# ---------------------------------------------------------------------------


def test_a_fresh_run_writes_the_null_form(tmp_path):
    block = build_lineage(model_path=None, model_dir=str(tmp_path / "fresh"), fork_step=0)
    assert block["role"] == "fresh"
    assert block["fork_parent"] is None
    assert block["ancestry"] == []
    assert "ancestry_stop" not in block


def test_the_null_form_round_trips_through_metadata(tmp_path):
    run = str(tmp_path / "run")
    block = build_lineage(model_path=None, model_dir=run, fork_step=0)
    save_model_snapshot(run, _StubVersion(), git_hash="a", lineage=block)
    assert read_block(run) == block
    assert fork_parent(run) is None
    assert role_of(run) == "fresh"


# ---------------------------------------------------------------------------
# 4 — ANCESTRY: two levels, a legacy stop, and cycle safety
# ---------------------------------------------------------------------------


def _chain(tmp_path):
    """grandparent (fresh, recorded) ← parent (fold, recorded) ← child (recorded)."""
    gp = _run(tmp_path, "gp", lineage={"schema": 1, "role": "fresh", "fork_parent": None,
                                       "fork_step": 0, "ancestry": []})
    p = _run(tmp_path, "parent", lineage={
        "schema": 1, "role": "fold", "fork_step": 100,
        "fork_parent": {"path": os.path.join(gp, "final_model.zip"), "run_name": "gp",
                        "run_dir": gp, "git_hash": "deadbeef", "arch_signature": "test_arch_v1"},
        "ancestry": []})
    c = _run(tmp_path, "child", lineage={
        "schema": 1, "role": "fork", "fork_step": 200,
        "fork_parent": {"path": os.path.join(p, "final_model.zip"), "run_name": "parent",
                        "run_dir": p, "git_hash": "deadbeef", "arch_signature": "test_arch_v1"},
        "ancestry": []})
    return gp, p, c


def test_ancestry_follows_two_levels_and_stops_at_the_fresh_root(tmp_path):
    gp, p, c = _chain(tmp_path)
    chain = ancestry(c)
    assert [n["run_name"] for n in chain] == ["parent", "gp"]
    assert [n["fork_step"] for n in chain] == [100, 0]
    assert all(n["source"] == "lineage" for n in chain)
    assert "fresh" in ancestry_stop(c)["reason"]


def test_ancestry_stops_CLEANLY_at_a_legacy_parent_and_says_where_it_went_dark(tmp_path):
    """A recorded child whose parent predates the block: the chain continues one derived step and
    then ends, with the reason recorded rather than the list simply being short."""
    legacy_root = _run(tmp_path, "legacy_root", command="train.py --steps 5")   # no --model
    legacy = _run(tmp_path, "legacy_mid",
                  command=f"train.py --model {os.path.join(legacy_root, 'final_model.zip')}")
    child = _run(tmp_path, "child", lineage={
        "schema": 1, "role": "fork", "fork_step": 1,
        "fork_parent": {"path": os.path.join(legacy, "final_model.zip"), "run_name": "legacy_mid",
                        "run_dir": legacy}, "ancestry": []})
    chain = ancestry(child)
    assert [n["run_name"] for n in chain] == ["legacy_mid", "legacy_root"]
    assert chain[1]["source"] == "original_command"
    assert ancestry_stop(child)["at"] == "legacy_root"


def test_ancestry_stops_when_the_parent_directory_is_not_on_disk(tmp_path):
    child = _run(tmp_path, "child", lineage={
        "schema": 1, "role": "fork", "fork_step": 1,
        "fork_parent": {"path": "/nowhere/at/all/final_model.zip",
                        "run_dir": "/nowhere/at/all"}, "ancestry": []})
    assert "not on disk" in ancestry_stop(child)["reason"]


def test_a_CYCLE_terminates_instead_of_recursing_forever(tmp_path):
    """A hand-edited or corrupted block can name its own descendant. Cycle safety is on realpaths."""
    a = os.path.join(str(tmp_path), "a")
    b = os.path.join(str(tmp_path), "b")
    _run(tmp_path, "a", lineage={"schema": 1, "role": "fork", "fork_step": 1, "ancestry": [],
                                 "fork_parent": {"path": os.path.join(b, "final_model.zip"),
                                                 "run_name": "b", "run_dir": b}})
    _run(tmp_path, "b", lineage={"schema": 1, "role": "fork", "fork_step": 2, "ancestry": [],
                                 "fork_parent": {"path": os.path.join(a, "final_model.zip"),
                                                 "run_name": "a", "run_dir": a}})
    chain = ancestry(a)
    assert [n["run_name"] for n in chain] == ["b", "a"]
    assert "CYCLE" in ancestry_stop(a)["reason"]


def test_the_depth_limit_terminates_a_very_long_chain(tmp_path):
    prev = None
    for i in range(6):
        lin = ({"schema": 1, "role": "fresh", "fork_parent": None, "fork_step": 0, "ancestry": []}
               if prev is None else
               {"schema": 1, "role": "fork", "fork_step": i, "ancestry": [],
                "fork_parent": {"path": os.path.join(prev, "final_model.zip"),
                                "run_name": os.path.basename(prev), "run_dir": prev}})
        prev = _run(tmp_path, f"gen{i}", lineage=lin)
    parent = fork_parent(prev)
    chain, stop = ancestry_from_parent(parent, max_depth=2)
    assert len(chain) == 2 and "depth limit 2" in stop["reason"]


# ---------------------------------------------------------------------------
# 5 — THE ACCESSOR: one call, and the legacy path warns
# ---------------------------------------------------------------------------


def test_the_accessor_prefers_the_RECORDED_block(tmp_path, capsys):
    run = _run(tmp_path, "r", command="train.py --model models/WRONG/final_model.zip",
               lineage={"schema": 1, "role": "fork", "fork_step": 3, "ancestry": [],
                        "fork_parent": {"path": "models/RIGHT/final_model.zip"}})
    got = fork_parent(run)
    assert got.path == "models/RIGHT/final_model.zip" and not got.derived
    assert "WARNING" not in capsys.readouterr().err


def test_the_accessor_DERIVES_from_original_command_and_WARNS(tmp_path, capsys):
    parent = _run(tmp_path, "the_parent", steps=42)
    run = _run(tmp_path, "legacy",
               command=f"python -m main.launcher --model {os.path.join(parent, 'final_model.zip')} "
                       f"--run-name legacy")
    got = fork_parent(run)
    err = capsys.readouterr().err
    assert "[lineage] WARNING: derived from original_command (legacy run, pre-lineage)" in err
    assert got.derived is True
    assert got.run_name == "the_parent"
    assert got.num_timesteps == 42
    assert got.sha256 is None          # the derive path never hashes; only a write does


def test_the_accessor_returns_None_when_nothing_names_a_parent(tmp_path):
    assert fork_parent(str(tmp_path / "not_a_run")) is None
    assert fork_parent(_run(tmp_path, "bare")) is None
    assert fork_parent(_run(tmp_path, "fresh_legacy", command="train.py --steps 5")) is None


@pytest.mark.parametrize("cmd,expect", [
    ("t.py --model A.zip", "A.zip"),
    ("t.py --model=A.zip", "A.zip"),
    ("t.py --model_path A.zip", "A.zip"),
    ("t.py --model 'a path/A.zip'", "a path/A.zip"),
    ("t.py --steps 5", None),
    ("t.py --model 'unterminated", None),          # shlex raises → total, never propagates
])
def test_parse_command_is_total(cmd, expect):
    assert parse_command(cmd)["model"] == expect


def test_teacher_paths_reuses_the_one_spec_parser():
    assert teacher_paths("A:*;B:x.txt,y.txt") == ["A", "B"]
    assert teacher_paths(None) == []
    assert teacher_paths("garbage-with-no-colon") == []     # malformed is not our error to raise


# ---------------------------------------------------------------------------
# 6 — INTEGRITY: the three checkable failures
# ---------------------------------------------------------------------------


def test_check_links_is_clean_on_a_healthy_link(tmp_path):
    parent = _run(tmp_path, "p")
    zip_path = os.path.join(parent, "final_model.zip")
    child = _run(tmp_path, "c", lineage={
        "schema": 1, "role": "fork", "fork_step": 1, "ancestry": [],
        "fork_parent": {"path": zip_path, "resolved_path": zip_path, "run_dir": parent,
                        "run_name": "p", "arch_signature": "test_arch_v1",
                        "sha256": sha256_file(zip_path)}})
    assert check_links(child) == []


def test_check_links_catches_a_REPLACED_parent_checkpoint(tmp_path):
    parent = _run(tmp_path, "p")
    zip_path = os.path.join(parent, "final_model.zip")
    child = _run(tmp_path, "c", lineage={
        "schema": 1, "role": "fork", "fork_step": 1, "ancestry": [],
        "fork_parent": {"path": zip_path, "resolved_path": zip_path, "run_dir": parent,
                        "sha256": "0" * 64}})
    assert any("sha256 MISMATCH" in p for p in check_links(child))


def test_check_links_catches_a_missing_parent_and_an_ARCH_CHANGE(tmp_path):
    parent = _run(tmp_path, "p", arch="OTHER_arch_v9")
    child = _run(tmp_path, "c", arch="test_arch_v1", lineage={
        "schema": 1, "role": "fork", "fork_step": 1, "ancestry": [],
        "fork_parent": {"path": os.path.join(parent, "final_model.zip"), "run_dir": parent,
                        "arch_signature": "OTHER_arch_v9"}})
    assert any("arch_signature changed" in p for p in check_links(child))

    gone = _run(tmp_path, "g", lineage={
        "schema": 1, "role": "fork", "fork_step": 1, "ancestry": [],
        "fork_parent": {"path": "/nowhere/final_model.zip", "run_dir": "/nowhere"}})
    problems = check_links(gone)
    assert any("parent run directory missing" in p for p in problems)
    assert any("parent checkpoint missing" in p for p in problems)


# ---------------------------------------------------------------------------
# 7 — the CLI, on a synthetic tree
# ---------------------------------------------------------------------------


def test_main_lineage_renders_a_synthetic_tree(tmp_path, capsys):
    from main.lineage import main as lineage_main
    gp, p, c = _chain(tmp_path)
    rc = lineage_main([c])
    out = capsys.readouterr().out
    assert rc == 0
    assert "child" in out and "parent" in out and "gp" in out
    assert "recorded" in out


def test_main_lineage_json_carries_the_chain(tmp_path, capsys):
    from main.lineage import main as lineage_main
    gp, p, c = _chain(tmp_path)
    assert lineage_main([c, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    row = payload["runs"][0]
    assert row["run"] == "child" and row["recorded"] is True
    assert [n["run_name"] for n in row["ancestry"]] == ["parent", "gp"]


def test_main_lineage_exits_nonzero_on_a_broken_link(tmp_path, capsys):
    from main.lineage import main as lineage_main
    broken = _run(tmp_path, "broken", lineage={
        "schema": 1, "role": "fork", "fork_step": 1, "ancestry": [],
        "fork_parent": {"path": "/nowhere/final_model.zip", "run_dir": "/nowhere"}})
    assert lineage_main([broken]) == 1
    assert "⚠" in capsys.readouterr().out


def test_main_lineage_reports_a_missing_run(tmp_path, capsys):
    from main.lineage import main as lineage_main
    assert lineage_main([str(tmp_path / "nope")]) == 1
    assert "no such run directory" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 8 — BACKFILL: dry-run by default, refuses to overwrite a recorded block
# ---------------------------------------------------------------------------


def test_backfill_is_a_DRY_RUN_by_default(tmp_path):
    from main.lineage import backfill
    parent = _run(tmp_path, "p", steps=7)
    legacy = _run(tmp_path, "legacy",
                  command=f"t.py --model {os.path.join(parent, 'final_model.zip')}")
    got = backfill(legacy)
    assert got["written"] is False
    assert got["block"]["derived"] is True
    assert got["block"]["fork_parent"]["run_name"] == "p"
    assert read_block(legacy) is None      # nothing on disk changed


def test_backfill_apply_writes_a_derived_block(tmp_path):
    from main.lineage import backfill
    parent = _run(tmp_path, "p", steps=7)
    legacy = _run(tmp_path, "legacy",
                  command=f"t.py --model {os.path.join(parent, 'final_model.zip')}")
    got = backfill(legacy, apply=True)
    assert got["written"] is True
    block = read_block(legacy)
    assert block["derived"] is True and block["fork_parent"]["run_name"] == "p"
    assert block["fork_step"] == 7
    # And the accessor now reports it as recorded-but-derived rather than re-parsing.
    assert fork_parent(legacy).derived is True


def test_backfill_REFUSES_a_run_that_already_records_a_block(tmp_path):
    from main.lineage import backfill
    run = _run(tmp_path, "r", command="t.py --model X.zip",
               lineage={"schema": 1, "role": "fork", "fork_parent": {"path": "KEEP.zip"},
                        "fork_step": 1, "ancestry": []})
    got = backfill(run, apply=True)
    assert "already records" in got["action"] and got["written"] is False
    assert read_block(run)["fork_parent"]["path"] == "KEEP.zip"


@pytest.mark.parametrize("cmd,needle", [
    (None, "no original_command"),
    ("t.py --steps 5", None),           # a FRESH legacy run backfills the null form
])
def test_backfill_skips_what_it_cannot_derive(tmp_path, cmd, needle):
    from main.lineage import backfill
    run = _run(tmp_path, "r", command=cmd)
    got = backfill(run)
    if needle:
        assert needle in got["action"]
    else:
        assert got["block"]["role"] == "fresh"


def test_backfill_skips_a_legacy_run_whose_model_is_its_own_checkpoint(tmp_path):
    """A recorded command naming a checkpoint INSIDE the run is a restart, not a fork — there is
    nothing to backfill, and inventing a self-parent would create a cycle."""
    from main.lineage import backfill
    run = _run(tmp_path, "r")
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    ck = _fake_checkpoint(os.path.join(run, "checkpoints", "checkpoint_9_steps.zip"), 9)
    with open(os.path.join(run, "metadata.json"), "w") as f:
        json.dump({"git_hash": "x", "original_command": f"t.py --model {ck}"}, f)
    assert "restart, not a fork" in backfill(run)["action"]


# ---------------------------------------------------------------------------
# 9 — the shapes themselves
# ---------------------------------------------------------------------------


def test_fork_parent_dataclass_round_trips_and_hides_a_false_derived():
    fp = ForkParent(path="a.zip", run_name="r")
    assert "derived" not in fp.to_dict()
    assert ForkParent.from_dict(fp.to_dict()).run_name == "r"
    assert ForkParent.from_dict({"path": "a.zip"}, derived=True).derived is True
    # Unknown keys from a future schema are ignored rather than raising.
    assert ForkParent.from_dict({"path": "a.zip", "some_future_field": 1}).path == "a.zip"


def test_build_lineage_from_command_marks_itself_derived(tmp_path):
    parent = _run(tmp_path, "p", steps=3)
    block = build_lineage_from_command(
        f"t.py --model {os.path.join(parent, 'final_model.zip')} --distill-teacher T:*",
        model_dir=str(tmp_path / "child"), hash_parent=False)
    assert block["derived"] is True and block["role"] == "fold"
    assert block["fork_parent"]["sha256"] is None


# ---------------------------------------------------------------------------
# 10 — the SEAM: both build paths must actually ask for the block, and pass it on
# ---------------------------------------------------------------------------


def test_both_model_build_paths_write_the_lineage():
    """A source pin. The block is useless if a build path stops asking for it, and the failure is
    silent — a run just quietly has no ancestry."""
    import inspect

    from agents.training import lineage as lineage_mod
    from main.train import model_build, run_io
    src = inspect.getsource(model_build)
    assert src.count("_run_lineage(") == 2, "the RESUME and the FRESH path each build it once"
    assert src.count("lineage=_lineage") == 6, "every save on both paths must carry it"
    assert "_fork_source_model = args.model" in src, (
        "the parent must be captured BEFORE --warmstart-consensus re-points args.model, or a "
        "warm-started exploiter records itself as its own ancestor")
    assert "is_same_run_checkpoint" in inspect.getsource(lineage_mod), (
        "fork-vs-restart is fork_lr\'s predicate, imported — never a second copy")
    assert "build_lineage" in inspect.getsource(run_io)


def test_dose_and_lineage_are_independent_blocks(tmp_path):
    """`dose` is written on every save; `lineage` is written once. Neither may eat the other."""
    run = str(tmp_path / "run")
    save_model_snapshot(run, _StubVersion(), git_hash="a",
                        hparams={"dose": {"dose_rate_now": 1.0}},
                        lineage=build_lineage(model_path=None, model_dir=run, fork_step=0))
    save_model_snapshot(run, _StubVersion(), git_hash="b",
                        hparams={"dose": {"dose_rate_now": 2.0}})
    with open(os.path.join(run, "metadata.json")) as f:
        meta = json.load(f)
    assert meta["dose"]["dose_rate_now"] == 2.0        # dose is CURRENT
    assert meta["lineage"]["role"] == "fresh"          # lineage is IMMUTABLE


# ---------------------------------------------------------------------------
# 9 — num_timesteps: how far the run GOT, beside where it STARTED (fork_step)
# ---------------------------------------------------------------------------


def test_read_num_timesteps_reads_the_top_level_key(tmp_path):
    from agents.training.lineage import read_num_timesteps

    run = _run(tmp_path, "r")
    meta_path = os.path.join(run, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["num_timesteps"] = 278_664_287
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    assert read_num_timesteps(run) == 278_664_287


def test_read_num_timesteps_is_UNKNOWN_on_a_legacy_run(tmp_path):
    """A run saved before the key existed. None, never 0 — this reader will not open a zip."""
    from agents.training.lineage import read_num_timesteps

    assert read_num_timesteps(_run(tmp_path, "legacy")) is None
    assert read_num_timesteps(str(tmp_path / "nope")) is None


def test_main_lineage_prints_the_step_count_beside_fork_step(tmp_path, capsys):
    from main.lineage import main as lineage_main

    gp, p, c = _chain(tmp_path)
    meta_path = os.path.join(c, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["num_timesteps"] = 5_000_000
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    assert lineage_main([c]) == 0
    out = capsys.readouterr().out
    assert "fork_step=" in out and "num_timesteps=5,000,000" in out


def test_main_lineage_says_unknown_rather_than_zero_for_a_legacy_run(tmp_path, capsys):
    from main.lineage import main as lineage_main, read_run as lineage_read_run

    gp, p, c = _chain(tmp_path)
    assert lineage_read_run(c)["num_timesteps"] is None
    assert lineage_main([c]) == 0
    assert "num_timesteps=unknown" in capsys.readouterr().out
