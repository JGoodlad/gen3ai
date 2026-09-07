"""`gen3_tb_inherit_v1` — a fork's inherited TensorBoard prefix.

Every test here builds a SYNTHETIC parent event file with known steps and tags, so the truncation,
the scalar filter, the idempotency guard and the fork-of-a-fork composition are each asserted
against a value this file wrote, not against whatever the archive happens to contain.
"""
from __future__ import annotations

import json
import os

import pytest

from agents.training import tb_inherit as TI


# --------------------------------------------------------------------------------------------
# helpers — write a synthetic tb/ the way a real trainer would
# --------------------------------------------------------------------------------------------
def _write_events(tb_dir: str, entries, *, name="events.out.tfevents.1700000000.host.1.0",
                  form="simple_value"):
    """`entries` = [(step, tag, value)] written the way a REAL writer writes them.

    🚨 `form` defaults to `simple_value` because that is what every writer in this tree actually
    puts on disk, and the distinction is the whole point: `EventFileLoader` migrates `simple_value`
    into a `scalars`-plugin tensor as it reads, so a copier built on that loader silently writes the
    migrated form, and `EventAccumulator` then files it under `tensors` — the curve vanishes from the
    scalars dashboard while every count and hash still looks right. A test that wrote the tensor form
    and read it back through the same migrating loader could not see that, which is exactly how it
    got past the first version of this suite. `form="tensor"` covers the other spelling.
    """
    from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter

    os.makedirs(tb_dir, exist_ok=True)
    header = event_pb2.Event(wall_time=0.0, file_version="brain.Event:2")
    with open(os.path.join(tb_dir, name), "wb") as fh:
        w = RecordWriter(fh)
        w.write(header.SerializeToString())
        for step, tag, val in entries:
            ev = event_pb2.Event(wall_time=1700000000.0 + step, step=step)
            if form == "simple_value":
                ev.summary.value.add(tag=tag, simple_value=float(val))
            else:
                meta = summary_pb2.SummaryMetadata()
                meta.plugin_data.plugin_name = "scalars"
                meta.data_class = summary_pb2.DATA_CLASS_SCALAR
                tensor = tensor_pb2.TensorProto(dtype=types_pb2.DT_FLOAT, float_val=[float(val)])
                ev.summary.value.add(tag=tag, metadata=meta, tensor=tensor)
            w.write(ev.SerializeToString())


def _on_disk_value_forms(tb_dir: str):
    """`{field_name: count}` over the RAW records — no migration. What is really on disk."""
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

    out = {}
    for path in TI.event_files(tb_dir):
        for ev in LegacyEventFileLoader(path).Load():
            if ev.WhichOneof("what") != "summary":
                continue
            for v in ev.summary.value:
                k = v.WhichOneof("value")
                out[k] = out.get(k, 0) + 1
    return out


def _dashboard_scalars(tb_dir: str, tag: str):
    """`[(step, value)]` as the SCALARS DASHBOARD would show them.

    Goes through `EventAccumulator.Scalars` — the accessor TensorBoard's scalars view uses — rather
    than a hand-rolled reader, so a value that lands in the wrong bucket fails the test instead of
    passing it.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0, "tensors": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def _write_histogram_event(tb_dir: str, step: int, tag: str,
                           name="events.out.tfevents.1700000001.host.1.0"):
    """One NON-scalar value — the thing the filter must drop."""
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter

    os.makedirs(tb_dir, exist_ok=True)
    with open(os.path.join(tb_dir, name), "wb") as fh:
        w = RecordWriter(fh)
        w.write(event_pb2.Event(wall_time=0.0, file_version="brain.Event:2").SerializeToString())
        ev = event_pb2.Event(wall_time=1700000000.0, step=step)
        histo = summary_pb2.HistogramProto(min=0.0, max=1.0, num=2, sum=1.0, sum_squares=1.0)
        histo.bucket_limit.extend([0.5, 1.0])
        histo.bucket.extend([1.0, 1.0])
        ev.summary.value.add(tag=tag, histo=histo)
        w.write(ev.SerializeToString())


def _read_back(tb_dir: str):
    """Every scalar in `tb_dir`, as `[(step, tag, value)]`, in BOTH on-disk spellings.

    Uses the RAW loader and handles `simple_value` and the tensor form explicitly, rather than
    letting `EventFileLoader` normalise them — normalising is what hid the value-form bug.
    """
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
    from tensorboard.util import tensor_util

    out = []
    for path in TI.event_files(tb_dir):
        for ev in LegacyEventFileLoader(path).Load():
            if ev.WhichOneof("what") != "summary":
                continue
            for v in ev.summary.value:
                which = v.WhichOneof("value")
                if which == "simple_value":
                    out.append((int(ev.step), v.tag, float(v.simple_value)))
                elif which == "tensor":
                    out.append((int(ev.step), v.tag, float(tensor_util.make_ndarray(v.tensor))))
    return out


def _make_run(root: str, name: str, entries=None, *, lineage=None) -> str:
    run = os.path.join(root, name)
    os.makedirs(run, exist_ok=True)
    if entries is not None:
        _write_events(TI.tb_dir_of(run), entries)
    if lineage is not None:
        with open(os.path.join(run, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"lineage": lineage}, f)
    return run


def _lineage_block(parent_run: str, fork_step: int, *, role="fork", derived=False):
    block = {
        "schema": 1, "role": role, "fork_step": fork_step,
        "fork_parent": {"run_dir": parent_run, "run_name": os.path.basename(parent_run)},
        "teachers": [], "exploiter_target": None, "ancestry": [],
    }
    if derived:
        block["derived"] = True
    return block


# --------------------------------------------------------------------------------------------
# truncation — the core contract
# --------------------------------------------------------------------------------------------
def test_only_events_at_or_below_fork_step_are_copied(tmp_path):
    """The parent trained PAST the fork; the copy must stop at fork_step exactly."""
    root = str(tmp_path)
    parent = _make_run(root, "parent", [(0, "a/x", 1.0), (100, "a/x", 2.0),
                                        (200, "a/x", 3.0), (300, "a/x", 4.0)])
    fork = _make_run(root, "fork")

    res = TI.inherit_tb(fork, parent_run_dir=parent, fork_step=200)

    assert res.written, res.reason
    assert res.events_copied == 3           # 0, 100, 200 — NOT 300
    assert res.first_step == 0 and res.last_step == 200
    steps = sorted(s for s, _, _ in _read_back(TI.tb_dir_of(fork)))
    assert steps == [0, 100, 200], "the parent's post-fork tail must never appear in the fork"


def test_fork_reads_as_one_continuous_series_from_step_zero(tmp_path):
    """The point of the whole feature: prefix + the fork's own events = one ordered curve.

    Read through `EventAccumulator.Scalars` — what the dashboard actually shows — because that is
    the only reader whose verdict matters, and the only one that catches a value written in the
    wrong form (see `_write_events`).
    """
    root = str(tmp_path)
    tag = "rollout/ep_rew_mean"
    parent = _make_run(root, "parent", [(0, tag, 1.0), (100, tag, 2.0),
                                        (200, tag, 3.0), (300, tag, 99.0)])
    fork = _make_run(root, "fork")
    TI.inherit_tb(fork, parent_run_dir=parent, fork_step=200)
    # ...then the fork trains, continuing the global step counter.
    _write_events(TI.tb_dir_of(fork), [(200, tag, 3.0), (250, tag, 4.0), (300, tag, 5.0)],
                  name="events.out.tfevents.1800000000.host.2.0")

    series = sorted(_dashboard_scalars(TI.tb_dir_of(fork), tag))

    assert series, "the inherited curve must appear in the SCALARS view, not just in the file"
    assert [s for s, _ in series] == [0, 100, 200, 200, 250, 300]
    assert series[0][0] == 0, "the curve must start at step 0"
    # the parent's step-300 value (99.0) must NOT have leaked in beside the fork's own 5.0
    assert 99.0 not in [v for _, v in series]


def test_the_on_disk_value_form_is_preserved(tmp_path):
    """THE REGRESSION GUARD for the bug the first version of this shipped.

    `EventFileLoader` migrates `simple_value` into a `scalars`-plugin tensor as it reads. Copying
    what it returns writes the migrated form; `EventAccumulator` then files those records under
    `tensors`, and the inherited curve is ABSENT from the scalars dashboard while the provenance,
    the event count and the sha all still look correct. Nothing but a raw read catches it.
    """
    root = str(tmp_path)
    parent = _make_run(root, "parent", [(0, "a/x", 1.0), (100, "a/x", 2.0)])
    assert _on_disk_value_forms(TI.tb_dir_of(parent)) == {"simple_value": 2}, "precondition"
    fork = _make_run(root, "fork")

    TI.inherit_tb(fork, parent_run_dir=parent, fork_step=100)

    assert _on_disk_value_forms(TI.tb_dir_of(fork)) == {"simple_value": 2}, (
        "the copy must preserve the parent's on-disk value form; a migrated tensor here means the "
        "curve will not appear in the scalars dashboard")
    assert _dashboard_scalars(TI.tb_dir_of(fork), "a/x") == [(0, 1.0), (100, 2.0)]


def test_a_tensor_form_parent_is_carried_too(tmp_path):
    """The other legal spelling must survive the copy unchanged as well."""
    root = str(tmp_path)
    parent = _make_run(root, "parent")
    _write_events(TI.tb_dir_of(parent), [(0, "a/x", 1.0), (100, "a/x", 2.0)], form="tensor")
    fork = _make_run(root, "fork")

    res = TI.inherit_tb(fork, parent_run_dir=parent, fork_step=100)

    assert res.written and res.events_copied == 2
    assert _on_disk_value_forms(TI.tb_dir_of(fork)) == {"tensor": 2}


def test_the_inherited_file_sorts_before_the_runs_own_events(tmp_path):
    """Name order is the order a directory walk reads in; the prefix must come first."""
    root = str(tmp_path)
    parent = _make_run(root, "parent", [(0, "a/x", 1.0)])
    fork = _make_run(root, "fork")
    TI.inherit_tb(fork, parent_run_dir=parent, fork_step=100)
    _write_events(TI.tb_dir_of(fork), [(100, "a/x", 2.0)],
                  name="events.out.tfevents.1800000000.host.2.0")

    names = [os.path.basename(f) for f in TI.event_files(TI.tb_dir_of(fork))]
    assert names[0] == TI.INHERITED_EVENTS_BASENAME


# --------------------------------------------------------------------------------------------
# the scalar filter
# --------------------------------------------------------------------------------------------
def test_non_scalar_values_are_skipped(tmp_path):
    root = str(tmp_path)
    parent = _make_run(root, "parent", [(10, "a/x", 1.0)])
    _write_histogram_event(TI.tb_dir_of(parent), 10, "a/hist")
    fork = _make_run(root, "fork")

    res = TI.inherit_tb(fork, parent_run_dir=parent, fork_step=100)

    assert res.tags == ("a/x",), "a histogram must not be carried into the fork"
    assert res.values_copied == 1


# --------------------------------------------------------------------------------------------
# provenance + idempotency
# --------------------------------------------------------------------------------------------
def test_provenance_is_written_and_describes_the_copy(tmp_path):
    root = str(tmp_path)
    parent = _make_run(root, "parent_run", [(0, "a/x", 1.0), (50, "b/y", 2.0)])
    fork = _make_run(root, "fork")

    res = TI.inherit_tb(fork, parent_run_dir=parent, fork_step=50)
    prov = TI.read_provenance(fork)

    assert prov is not None
    assert prov["marker"] == "gen3_tb_inherit_v1"
    assert prov["parent_run_name"] == "parent_run"
    assert prov["parent_run_dir"] == os.path.abspath(parent)
    assert prov["fork_step"] == 50
    assert prov["events_copied"] == 2
    assert sorted(prov["tags_copied"]) == ["a/x", "b/y"]
    assert prov["events_file"] == TI.INHERITED_EVENTS_BASENAME
    # the sha names the file that was actually written
    assert prov["sha256"] == res.sha256
    assert prov["sha256"] == TI._sha256_file(os.path.join(TI.tb_dir_of(fork),
                                                          TI.INHERITED_EVENTS_BASENAME))


def test_a_second_call_is_a_no_op(tmp_path):
    """THE LAUNCHER-RESTART GUARD. A restart that still names the parent must not double the prefix."""
    root = str(tmp_path)
    parent = _make_run(root, "parent", [(0, "a/x", 1.0), (100, "a/x", 2.0)])
    fork = _make_run(root, "fork")

    first = TI.inherit_tb(fork, parent_run_dir=parent, fork_step=100)
    before = _read_back(TI.tb_dir_of(fork))
    second = TI.inherit_tb(fork, parent_run_dir=parent, fork_step=100)

    assert first.written and not second.written
    assert "already inherited" in second.reason
    assert _read_back(TI.tb_dir_of(fork)) == before, "a re-copy would draw every series twice"


def test_force_rewrites_rather_than_appending(tmp_path):
    root = str(tmp_path)
    parent = _make_run(root, "parent", [(0, "a/x", 1.0), (100, "a/x", 2.0)])
    fork = _make_run(root, "fork")
    TI.inherit_tb(fork, parent_run_dir=parent, fork_step=100)

    again = TI.inherit_tb(fork, parent_run_dir=parent, fork_step=100, force=True)

    assert again.written
    assert len(_read_back(TI.tb_dir_of(fork))) == 2, "force replaces the file, never appends"


def test_dry_run_touches_nothing(tmp_path):
    root = str(tmp_path)
    parent = _make_run(root, "parent", [(0, "a/x", 1.0)])
    fork = _make_run(root, "fork")

    res = TI.inherit_tb(fork, parent_run_dir=parent, fork_step=100, dry_run=True)

    assert res.written and res.events_copied == 1
    assert TI.read_provenance(fork) is None
    assert not os.path.isdir(TI.tb_dir_of(fork))


# --------------------------------------------------------------------------------------------
# fork of a fork
# --------------------------------------------------------------------------------------------
def test_fork_of_a_fork_truncates_the_grandparent_prefix(tmp_path):
    """The grandparent's prefix rides in the parent's tb/, so re-truncating composes the chain."""
    root = str(tmp_path)
    gp = _make_run(root, "grandparent", [(0, "a/x", 0.0), (100, "a/x", 1.0),
                                         (200, "a/x", 2.0), (900, "a/x", 9.0)])
    parent = _make_run(root, "parent")
    TI.inherit_tb(parent, parent_run_dir=gp, fork_step=200)          # parent forked at 200
    _write_events(TI.tb_dir_of(parent), [(200, "a/x", 2.0), (300, "a/x", 3.0),
                                         (400, "a/x", 4.0)],
                  name="events.out.tfevents.1800000000.host.2.0")

    child = _make_run(root, "child")
    res = TI.inherit_tb(child, parent_run_dir=parent, fork_step=300)  # child forked at 300

    assert res.parent_inherited, "the parent's own prefix must be recognised"
    steps = sorted(s for s, _, _ in _read_back(TI.tb_dir_of(child)))
    # 0,100,200 from the GRANDPARENT (via the parent's prefix) + 200,300 from the parent itself.
    assert steps == [0, 100, 200, 200, 300]
    assert 400 not in steps, "the parent's post-fork tail must be truncated"
    assert 900 not in steps, "the grandparent's post-fork tail was already gone and stays gone"


# --------------------------------------------------------------------------------------------
# the training seam — `inherit_from_lineage`
# --------------------------------------------------------------------------------------------
def test_seam_copies_on_a_fork_block(tmp_path):
    root = str(tmp_path)
    parent = _make_run(root, "parent", [(0, "a/x", 1.0), (100, "a/x", 2.0), (200, "a/x", 3.0)])
    fork = _make_run(root, "fork")

    res = TI.inherit_from_lineage(fork, _lineage_block(parent, 100))

    assert res.written and res.events_copied == 2


def test_seam_is_a_no_op_on_a_restart_and_on_a_fresh_run(tmp_path):
    root = str(tmp_path)
    fork = _make_run(root, "fork")

    restart = TI.inherit_from_lineage(fork, None)          # build_lineage returns None on a restart
    fresh = TI.inherit_from_lineage(fork, {"role": "fresh", "fork_parent": None, "fork_step": 0})

    assert not restart.written and "same-run restart" in restart.reason
    assert not fresh.written and "fresh run" in fresh.reason
    assert not os.path.isdir(TI.tb_dir_of(fork))


def test_no_tb_inherit_writes_nothing(tmp_path):
    root = str(tmp_path)
    parent = _make_run(root, "parent", [(0, "a/x", 1.0)])
    fork = _make_run(root, "fork")

    res = TI.inherit_from_lineage(fork, _lineage_block(parent, 100), enabled=False)

    assert not res.written
    assert "--no-tb-inherit" in res.reason
    assert not os.path.isdir(TI.tb_dir_of(fork))


def test_seam_never_raises_when_the_parent_is_gone(tmp_path):
    """A cosmetic convenience must not be able to kill a training launch."""
    root = str(tmp_path)
    fork = _make_run(root, "fork")

    res = TI.inherit_from_lineage(fork, _lineage_block(os.path.join(root, "vanished"), 100))

    assert not res.written and "nothing to inherit" in res.reason


def test_a_fork_at_step_zero_inherits_nothing(tmp_path):
    root = str(tmp_path)
    parent = _make_run(root, "parent", [(0, "a/x", 1.0)])
    fork = _make_run(root, "fork")

    res = TI.inherit_tb(fork, parent_run_dir=parent, fork_step=0)

    assert not res.written and "step 0" in res.reason


# --------------------------------------------------------------------------------------------
# the backfill census
# --------------------------------------------------------------------------------------------
def test_forks_missing_prefix_lists_only_uninherited_forks(tmp_path):
    root = str(tmp_path)
    parent = _make_run(root, "parent", [(0, "a/x", 1.0), (100, "a/x", 2.0)])
    _make_run(root, "fresh_run", [(0, "a/x", 1.0)],
              lineage={"role": "fresh", "fork_parent": None, "fork_step": 0})
    _make_run(root, "a_fork", [(100, "a/x", 2.0)],
              lineage=_lineage_block(parent, 100, derived=True))
    done = _make_run(root, "already_done", [(100, "a/x", 2.0)],
                     lineage=_lineage_block(parent, 100))
    TI.inherit_tb(done, parent_run_dir=parent, fork_step=100)

    rows = TI.forks_missing_prefix(root)

    assert [r["run"] for r in rows] == ["a_fork"]
    assert rows[0]["parent_tb_exists"] is True
    assert rows[0]["derived"] is True, "a regex-derived parent claim must be flagged"


def test_forks_missing_prefix_flags_a_parent_with_no_tb(tmp_path):
    root = str(tmp_path)
    parent = _make_run(root, "parent")                     # no tb/ at all
    _make_run(root, "a_fork", [(100, "a/x", 2.0)], lineage=_lineage_block(parent, 100))

    rows = TI.forks_missing_prefix(root)

    assert rows[0]["parent_tb_exists"] is False


# --------------------------------------------------------------------------------------------
# the fork decision is IMPORTED, never re-derived
# --------------------------------------------------------------------------------------------
def test_the_seam_defers_to_build_lineages_fork_decision():
    """`inherit_from_lineage` must not own a second answer to 'is this a fork?'.

    `build_lineage` already returns None on a same-run restart via `fork_lr.is_same_run_checkpoint`;
    a second predicate here is a second answer waiting to disagree.

    Asserted over the AST's NAMES, not the source text — the module docstring *explains* the rule by
    naming the predicate, and a substring grep cannot tell an explanation from a call.
    """
    for name in _referenced_names(TI):
        assert name != "is_same_run_checkpoint", (
            "tb_inherit must take the fork decision from the lineage block it is handed, "
            "not re-derive it")


def _referenced_names(module):
    """Every identifier the module's CODE mentions (attributes, calls, imports) — docstrings out."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add(a.name.split(".")[0])
                names.add(a.name.rsplit(".", 1)[-1])
            if isinstance(node, ast.ImportFrom) and node.module:
                names.update(node.module.split("."))
    return names


def test_module_is_torch_free():
    """Like `main.lineage` / `main.dose`: it must read runs whose architecture has drifted.

    Two halves, because either alone is weak: nothing in the import graph may name torch or SB3
    (the AST check), and importing the module must not pull torch into `sys.modules` in a fresh
    interpreter (the behavioural check the AST cannot make).
    """
    names = _referenced_names(TI)
    assert "torch" not in names and "stable_baselines3" not in names

    import subprocess
    import sys
    from utils.paths import src_root
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import agents.training.tb_inherit; "
         "print('torch' in sys.modules or 'stable_baselines3' in sys.modules)"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(src_root())})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", "importing tb_inherit must not drag torch in"


@pytest.mark.parametrize("bad", ["", "   "])
def test_a_blank_parent_run_dir_is_a_no_op(tmp_path, bad):
    fork = _make_run(str(tmp_path), "fork")
    block = _lineage_block("x", 100)
    block["fork_parent"]["run_dir"] = bad
    assert not TI.inherit_from_lineage(fork, block).written
