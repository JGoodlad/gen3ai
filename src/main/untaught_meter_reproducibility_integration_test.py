"""THE REPRODUCIBILITY GATE for ``python -m main.untaught_meter`` — real bridge battles.

The meter's whole claim is that a LEVEL it prints can be quoted. That needs two things at once and
this test proves both in one shot: every global-RNG seam pinned, and ``concurrency == 1``. Seeds
alone are not enough — measured 2026-09-03, seeded at concurrency 3 two runs of the offline
collateral-KL probe still produced 1193 vs 1141 states with arm levels up to +0.043 apart.

It also proves the thing that licenses SHARDING: two runs at ``--workers 2`` are byte-identical, so
a cell is a pure function of (ref, team index, battle index) and the split across worker PROCESSES
cannot move a number. ``exploiter_competence`` verified this by hand before it sharded 3200 battles
across six workers; here it is a standing gate.

Marked ``sim`` (bridge battles, no server) and ``slow`` (two full measurement runs, 24 battles plus
four model loads — cost tracks battle COUNT, and ``slow`` is the marker that means expensive).

Run it alone::

    export PYTHONPATH=$PYTHONPATH:src
    pytest src/main/untaught_meter_reproducibility_integration_test.py -q -m "sim and slow"
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from agents.training import untaught_meter as engine
from utils.paths import main_models_dir, repo_root, src_root

pytestmark = [pytest.mark.sim, pytest.mark.slow]

#: Two FROZEN, explicitly-named files. Naming the .zip (rather than a run dir) pins resolution, so
#: a run that is still training cannot move the file under the test between its two invocations.
_REF = "ai_v9_59_R2ACTION_0827/final_model.zip"
_BASELINE = "ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip"
_CONFIG = "ai_v9_29_rev1_0823/snapshots/model_config.json"


def _models_or_skip():
    models = main_models_dir()
    if models is None:
        pytest.skip("no models/ archive on this box ($GEN3AI_MODELS_DIR / the main checkout)")
    for rel in (_REF, _BASELINE, _CONFIG):
        if not os.path.exists(models / rel):
            pytest.skip(f"the meter's fixture checkpoint is absent: {rel}")
    return models


def _teams_manifest(tmp_path) -> str:
    """The FIRST TWO of the untaught 8, in seed order — a prefix of the real slice, not a reshuffle."""
    teams = engine.load_team_manifest(str(engine.DEFAULT_TEAMS_MANIFEST))[:2]
    p = tmp_path / "teams2.json"
    p.write_text(json.dumps({"note": "reproducibility gate: the first 2 of the untaught 8",
                             "untaught": [t.path for t in teams]}))
    return str(p)


def _run(models, manifest, out_path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [env.get("PYTHONPATH", ""), str(src_root())] if p)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env.setdefault(var, "1")
    env.setdefault("GEN3AI_TIMEOUT_SCALE", "8")
    prebuilt = repo_root() / "src" / "rust_sim" / "target" / "release" / "sim_bridge"
    if "POKESIM_SIM_BRIDGE_BIN" not in env and prebuilt.exists():
        env["POKESIM_SIM_BRIDGE_BIN"] = str(prebuilt)     # never pay a cargo build inside a test
    argv = [sys.executable, "-m", "main.untaught_meter",
            f"ARM={models / _REF}", "--baseline", f"BASE={models / _BASELINE}",
            "--opponent", str(models / _BASELINE), "--config", str(models / _CONFIG),
            "--teams", manifest, "--games-per-team", "3", "--workers", "2",
            "--quiet", "--json", out_path]
    proc = subprocess.run(argv, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"meter failed:\n{proc.stdout}\n{proc.stderr}"
    with open(out_path) as fh:
        return json.load(fh)


def _stable(doc: dict) -> str:
    """Everything but the wall clock and the cwd — the part a quoted level lives in."""
    doc = json.loads(json.dumps(doc))
    doc["_meta"].pop("volatile", None)
    return json.dumps(doc, sort_keys=True, indent=1)


def test_two_sharded_runs_of_the_meter_are_byte_identical(tmp_path):
    """Same argv, same output path, two processes: the JSON must match byte for byte.

    Both halves are under test at once — the five seeds AND ``concurrency=1``. If either regresses
    the levels start wandering and every delta read off them becomes a draw rather than a
    measurement.
    """
    models = _models_or_skip()
    manifest = _teams_manifest(tmp_path)
    out = str(tmp_path / "run.json")

    first = _stable(_run(models, manifest, out))
    second = _stable(_run(models, manifest, out))
    assert first == second

    doc = json.loads(second)
    assert doc["result"]["timeouts"]["timeouts"] == 0
    assert doc["_meta"]["concurrency"] == 1
    assert doc["_meta"]["workers"] == 2
    # A level worth quoting: 2 teams x 3 games x 2 pilots.
    assert doc["result"]["levels"]["ARM"]["attempted"] == 6
    assert len(doc["result"]["teams"]) == 2
