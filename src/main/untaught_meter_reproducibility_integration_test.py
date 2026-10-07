"""THE REPRODUCIBILITY GATE for ``python -m main.untaught_meter`` — real games on the Rust EVAL CORE.

The meter's whole claim is that a LEVEL it prints can be quoted. Since poke-env retirement P2 every game runs on the
Rust eval core (``agents.training.untaught_rust``), where a game is a pure function of its key (the cell's cycle
seed = f(``--seed``, team index), the game index) through the per-game seed rule and the keyed draw. Two things
follow, and this test proves both through the CLI exactly as a reader runs it:

* two runs of the same argv are BYTE-IDENTICAL (everything but the wall clock and the cwd);
* the SHARD split cannot move a number: ``--workers 2`` (two processes, each its own engine) and ``--workers 1``
  (one engine) give the same levels, at one compute (``--n-envs`` etc.: a decision within a rounding error of a tie
  may flip with the forward's batch shape, so the compute is held fixed — standing rule 8).

It also pins the TRANSPORT stamp the artifact carries (``rust_eval``, the encoder, the core build).

Marked ``sim`` + ``integration`` (real games, the in-process core) and ``slow`` (three CLI runs, five engine
startups). The checkpoints are two seeded PERTURBED-fresh production-architecture policies (the Lane H gate's
recipe), saved once to ``tmp_path``, so it needs no run archive.

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
from utils.paths import repo_root, src_root

pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]

COMPUTE = ["--n-envs", "4", "--threads", "1", "--torch-threads", "1", "--front", "ffi", "--profile", "selfcheck"]


def _models(tmp_path) -> dict:
    from agents.training.rust_eval import parity as PAR
    from agents.training.rust_rollout.testkit import build_selfcheck

    build_selfcheck()
    with PAR.declared_torch_state(1):
        arm, (base,), cfg = PAR.build_models(tmp_path / "run_untaught_repro", n_sentinels=1)
    return {"ref": arm, "baseline": base, "config": cfg}


def _teams_manifest(tmp_path) -> str:
    """The FIRST TWO of the untaught 8, in seed order — a prefix of the real slice, not a reshuffle."""
    teams = engine.load_team_manifest(str(engine.DEFAULT_TEAMS_MANIFEST))[:2]
    p = tmp_path / "teams2.json"
    p.write_text(json.dumps({"note": "reproducibility gate: the first 2 of the untaught 8",
                             "untaught": [t.path for t in teams]}))
    return str(p)


def _run(models, manifest, out_path, workers: int) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(src_root()), env.get("PYTHONPATH", "")] if p)
    argv = [sys.executable, "-m", "main.untaught_meter",
            f"ARM={models['ref']}", "--baseline", f"BASE={models['baseline']}",
            "--opponent", models["baseline"],
            "--teams", manifest, "--games-per-team", "3", "--workers", str(workers), *COMPUTE,
            "--quiet", "--json", out_path]
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, cwd=str(repo_root()), timeout=1800)
    assert proc.returncode == 0, f"meter failed:\n{proc.stdout}\n{proc.stderr}"
    with open(out_path) as fh:
        return json.load(fh)


def _stable(doc: dict) -> str:
    """Everything but the wall clock and the cwd — the part a quoted level lives in."""
    doc = json.loads(json.dumps(doc))
    doc["_meta"].pop("volatile", None)
    return json.dumps(doc, sort_keys=True, indent=1)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("untaught_repro")
    models = _models(tmp)
    manifest = _teams_manifest(tmp)
    out = str(tmp / "run.json")
    first = _stable(_run(models, manifest, out, 2))
    second = _stable(_run(models, manifest, out, 2))
    one = _run(models, manifest, str(tmp / "one.json"), 1)
    return first, second, one


def test_two_sharded_runs_of_the_meter_are_byte_identical(runs):
    first, second, _one = runs
    assert first == second
    doc = json.loads(second)
    assert doc["result"]["timeouts"]["timeouts"] == 0          # a turn-limit game is a DRAW on the core
    assert doc["_meta"]["workers"] == 2
    # A level worth quoting: 2 teams x 3 games x 2 pilots.
    assert doc["result"]["levels"]["ARM"]["attempted"] == 6
    assert len(doc["result"]["teams"]) == 2


def test_the_shard_split_cannot_move_a_number(runs):
    _first, second, one = runs
    two = json.loads(second)
    assert one["_meta"]["workers"] == 1
    for lab in ("ARM", "BASE"):
        assert one["result"]["levels"][lab]["per_team"] == two["result"]["levels"][lab]["per_team"]


def test_the_artifact_carries_the_rust_eval_transport_stamp(runs):
    _first, second, one = runs
    for doc in (json.loads(second), one):
        m = doc["_meta"]
        assert (m["transport"], m["encoder"]) == ("rust_eval", "rust") and m["core_stamp"]
        assert doc["result"]["transport"] == "rust_eval"
        assert all(c["transport"] == "rust_eval" for lv in doc["result"]["levels"].values()
                   for c in lv["per_team"].values())
