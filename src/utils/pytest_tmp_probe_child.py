"""CHILD-session probe for ``pytest_tmp_on_disk_test.py`` — NOT collected by the gate (the name does
not match ``python_files``); the parent runs it explicitly in a subprocess. It writes where its
``tmp_path``, ``tempfile.gettempdir()`` and a GRANDCHILD process's temp dir landed."""
import json
import os
import subprocess
import sys
import tempfile


def test_report_where_temp_lands(tmp_path):
    (tmp_path / "payload.bin").write_bytes(b"x" * 100)
    child = subprocess.run([sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
                           capture_output=True, text=True, check=True).stdout.strip()
    out = os.environ["GEN3AI_TMP_PROBE_OUT"]
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    with open(os.path.join(out, f"{worker}.json"), "w") as f:
        json.dump({"tmp_path": str(tmp_path), "gettempdir": tempfile.gettempdir(),
                   "subprocess_gettempdir": child, "TMPDIR": os.environ.get("TMPDIR")}, f)


def test_second_so_both_workers_run(tmp_path):
    test_report_where_temp_lands(tmp_path)
