"""The cutover stress driver: resumability (a unit on disk is never re-run), the fair scheduler,
the pin's capabilities gating a stream, and a registration that cannot be silently edited."""
import json
import subprocess
import sys
import time

import pytest

from main.rust_core_cutover import driver as D
from main.rust_core_cutover import plan as PL

SIZES = dict(pool_n=719, ladder_full_n=22813, ladder_policy_n=800)


def test_next_unit_skips_done_and_running_units_and_balances_streams():
    streams = [s for s in PL.streams(**SIZES) if s.name in ("ladder_full_a", "corpora")]
    done = {PL.unit_id("ladder_full_a", 0), PL.unit_id("ladder_full_a", 1)}
    # corpora has 0/1 done, ladder 2/286: corpora is the least-advanced stream
    assert D.next_unit(streams, done, set(), None, []) == "corpora.00000"
    done.add("corpora.00000")
    assert D.next_unit(streams, done, set(), None, []) == "ladder_full_a.00002"
    assert D.next_unit(streams, done, {"ladder_full_a.00002"}, None, []) == "ladder_full_a.00003"
    assert D.next_unit(streams, done, set(), ["corpora"], []) is None


def test_a_stream_the_pin_cannot_run_is_never_scheduled():
    streams = [s for s in PL.streams(**SIZES) if s.name.startswith("envn_")]
    assert D.next_unit(streams, set(), set(), None, []) is None
    assert D.next_unit(streams, set(), set(), None, ["obs_source_core"]) is not None


def test_rows_are_atomic_and_define_done(tmp_path):
    D.write_row(tmp_path, {"unit": "corpora.00000", "stream": "corpora", "status": "ok"})
    assert D.done_units(tmp_path) == {"corpora.00000"}
    assert not list((tmp_path / "units").glob("*.tmp"))
    # a half-written temp file is NOT a done unit
    (tmp_path / "units" / "ladder_full_a.00000.json.tmp").write_text("{")
    assert D.done_units(tmp_path) == {"corpora.00000"}


def test_a_changed_registration_is_refused_unless_amended_visibly(tmp_path):
    D.write_registration(tmp_path, SIZES)
    D.write_registration(tmp_path, SIZES)                 # the same plan resumes
    with pytest.raises(SystemExit):
        D.write_registration(tmp_path, {**SIZES, "pool_n": 720})
    reg = D.write_registration(tmp_path, {**SIZES, "pool_n": 720}, amend="pool grew")
    assert reg["amendments"][0]["reason"] == "pool grew"
    assert "pool_random" in reg["amendments"][0]["changed"]
    assert (tmp_path / "registration.0.json").exists()
    assert D.write_registration(tmp_path, {**SIZES, "pool_n": 720}) == reg   # resumes amended


def test_an_orphaned_unit_process_is_recognised_by_its_argv():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
                          "main.rust_core_cutover", "unit", "x.00001"])
    try:
        time.sleep(0.2)
        assert D._alive_unit(p.pid, "x.00001")
        assert not D._alive_unit(p.pid, "x.00002")
        a = D._Adopted(p.pid)
        assert a.poll() is None
    finally:
        p.kill()
        p.wait()
    assert D._Adopted(p.pid).poll() == 0


def test_the_registration_on_disk_names_every_target(tmp_path):
    reg = D.write_registration(tmp_path, SIZES)
    assert json.loads((tmp_path / "registration.json").read_text()) == reg
    names = {s["name"] for s in reg["streams"]}
    assert {"ladder_full_a", "ladder_full_b", "pool_random", "soak_transport", "envn_pool_random",
            "fz_proto_ladder", "fz_sbdiff_ladder"} <= names
    assert all(s["target"] for s in reg["streams"])
