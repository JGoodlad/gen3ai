"""CHILD-session probe for ``pytest_session_tmp_test.py`` -- NOT collected by the gate (the name does not match
``python_files``); the parent runs it explicitly, several at once. What it does is chosen by ``$PROBE_MODE``:

* ``hold``  -- write a file in its ``tmp_path``, announce itself (``<$PROBE_OUT>.up``), wait for ``<$PROBE_OUT>.go``,
  then report whether its file is still there (``<$PROBE_OUT>``);
* ``quick`` -- report where its ``tmp_path`` is and pass;
* ``fail``  -- report, then fail (its tree is kept for debugging).
"""
import json
import os
import time


def test_probe(tmp_path):
    out = os.environ["PROBE_OUT"]
    mode = os.environ["PROBE_MODE"]
    mine = tmp_path / "keep.txt"
    mine.write_text("x")
    report = {"pid": os.getpid(), "tmp_path": str(tmp_path), "basetemp": str(tmp_path.parent)}
    if mode == "hold":
        with open(out + ".up", "w") as f:
            json.dump(report, f)
        deadline = time.time() + float(os.environ.get("PROBE_WAIT_S", "60"))
        while time.time() < deadline and not os.path.exists(out + ".go"):
            time.sleep(0.1)
        report["alive_at_end"] = mine.exists()
    with open(out, "w") as f:
        json.dump(report, f)
    assert mode != "fail", "a deliberately failing session"
    assert mine.exists(), "this session's own tmp_path vanished under it"
