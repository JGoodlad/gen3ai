"""``utils.gate_lock_status`` — the pure halves: the ``-n`` parse, the same-shape ETA, the slot record's
ownership rule and the history trim. The real-process halves (a live holder's record, the queue, the
history line a release writes) are in ``gate_lock_test.py``."""
from __future__ import annotations

import json
import os

from utils import gate_lock_status as GS


def test_workers_of_reads_every_spelling_of_n_and_refuses_auto():
    assert GS.workers_of(["pytest", "src/", "-n", "3"]) == 3
    assert GS.workers_of(["pytest", "-n6"]) == 6
    assert GS.workers_of(["pytest", "--numprocesses", "4"]) == 4
    assert GS.workers_of(["pytest", "--numprocesses=2"]) == 2
    assert GS.workers_of(["pytest", "-n", "auto"]) is None
    assert GS.workers_of(["pytest", "src/"]) is None
    assert GS.workers_of(["pytest", "-nofoo"]) is None


def _h(held, *, cpus=6, workers=3, rc=0, routine=True):
    return {"held_s": held, "cpus": cpus, "workers": workers, "rc": rc, "routine": routine}


def test_the_eta_is_the_median_of_the_last_gates_of_the_SAME_shape_only():
    hist = [_h(9999, cpus=16, workers=6),          # another shape: never mixed in
            _h(9999, workers=4),
            _h(9999, rc=2),                         # a usage error / interrupted: not a finished gate
            _h(9999, rc=-15),
            _h(9999, routine=False),                # not a routine gate
            _h(100), _h(300), _h(200)]
    assert GS.eta_s(hist, cpus=6, workers=3) == (200.0, 3)
    assert GS.eta_s(hist, cpus=6, workers=None) is None
    assert GS.eta_s(hist, cpus=8, workers=3) is None
    # failures (rc 1) ran to the end: they count; only the last ETA_SAMPLES are read
    many = [_h(1000.0)] * 20 + [_h(float(s), rc=1) for s in range(1, GS.ETA_SAMPLES + 1)]
    med, n = GS.eta_s(many, cpus=6, workers=3)
    assert n == GS.ETA_SAMPLES and med == float((GS.ETA_SAMPLES + 1) // 2)


def test_a_record_is_cleared_only_by_its_own_holder(tmp_path):
    rec = GS.SlotRecord.mine(0, ["pytest", "src/", "-m", "not slow and not e2e", "-n", "3"])
    GS.write_record(tmp_path, rec)
    got = GS.read_record(tmp_path, 0)
    assert got is not None and got.pid == os.getpid() and got.workers == 3 and got.cpus >= 1
    GS.clear_record(tmp_path, 0, pid=rec.pid + 1)        # a LATER holder's release must not remove it
    assert GS.read_record(tmp_path, 0) is not None
    GS.clear_record(tmp_path, 0, pid=rec.pid)
    assert GS.read_record(tmp_path, 0) is None


def test_a_corrupt_or_foreign_record_reads_as_none(tmp_path):
    GS.record_path(tmp_path, 0).write_text("{not json")
    assert GS.read_record(tmp_path, 0) is None
    GS.record_path(tmp_path, 0).write_text(json.dumps({"schema": "other", "pid": 1}))
    assert GS.read_record(tmp_path, 0) is None


def test_history_appends_skips_torn_lines_and_trims(tmp_path):
    rec = GS.SlotRecord.mine(0, ["pytest", "-m", "not slow and not e2e", "-n", "3"])
    GS.history_path(tmp_path).write_text("{torn\n")
    for i in range(2 * GS.HISTORY_KEEP + 1):
        GS.append_history(tmp_path, rec, held_s=float(i), rc=0)
    rows = GS.read_history(tmp_path)
    # the torn line + 2K appends pass 2K lines on the (2K)th append, which trims to the last K; one more
    # append follows: K + 1, the torn line gone
    assert len(rows) == GS.HISTORY_KEEP + 1
    assert rows[-1]["held_s"] == float(2 * GS.HISTORY_KEEP) and rows[-1]["routine"] is True
    assert rows[-1]["workers"] == 3


def test_cpu_list_compresses_ranges():
    assert GS.cpu_list([5, 6, 7, 13, 14, 15]) == "5-7,13-15"
    assert GS.cpu_list([0, 2, 3]) == "0,2-3"
    assert GS.cpu_list([]) == ""


def test_the_human_report_names_a_free_slot_and_an_empty_queue(tmp_path):
    state = {"slots": [None], "waiters": []}
    out = GS.format_human(state)
    assert "slot 0: FREE" in out and "nobody waiting" in out
    assert GS.format_compact(state) == "slot 0 free; 0 waiting"
