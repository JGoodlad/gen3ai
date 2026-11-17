"""The fresh-cell kit END TO END on CPU, on real BOUNDARY-1 runs posing as a fresh cell and a banked cell: plan (the
registered guards refuse the posing runs; the tests' override admits them), play at the pin, read, re-check, and the
REPORTED team bootstrap. A REHEARSAL ONLY: the first checkpoint (1M steps, never a B1 look step), a few mirrored pairs
a cell, a scratch ledger and work directory, no number quoted anywhere.

Opt-in: it needs the main run archive and a clean checkout of B1's launch pin with its Rust env core built
(``GEN3AI_FRESH_REHEARSAL_PIN_DIR``); ``GEN3AI_FRESH_REHEARSAL_OUT`` keeps the work directory (default: pytest's tmp).
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pytest

from main.b1 import arms as A
from main.b1 import cross as X
from main.b1 import fresh as F
from main.b1 import fresh_rule as FR
from main.b1 import look as K

PIN_DIR = os.environ.get("GEN3AI_FRESH_REHEARSAL_PIN_DIR")
#: Two C runs and two W runs: each side holds one of each, so no cell of the rehearsal is a B1 contrast on its own.
POSING = {"rb_b1c_s5101": "C", "rb_b1w_s5103": "W", "rb_b1c_s5102": "C", "rb_b1w_s5104": "W"}
STEP = 1_000_000
PAIRS = {p: 4 for p in X.PANELS}

pytestmark = [pytest.mark.e2e, pytest.mark.slow,
              pytest.mark.skipif(not PIN_DIR, reason="opt-in: set GEN3AI_FRESH_REHEARSAL_PIN_DIR to a clean "
                                                     "checkout of B1's launch pin with its Rust env core built")]


def _spec() -> F.ArmSpec:
    """Built directly: :func:`main.b1.fresh.arm_from_doc` refuses B1's own seeds and names as a fresh arm."""
    with pytest.raises(F.FreshError, match="BOUNDARY-1's own"):
        F.arm_from_doc({"schema": F.ARM_SCHEMA, "card": "rehearsal", "pin": F.B1_LAUNCH_PIN, "banked_arm": "C",
                        "lever": ["--n-epochs", "5"], "declared_keys": ["n_epochs"],
                        "fresh": [{"seed": 5101, "run": "rb_b1c_s5101"}, {"seed": 5103, "run": "rb_b1w_s5103"}],
                        "own_control": [{"seed": 5102, "run": "rb_b1c_s5102"}, {"seed": 5104, "run": "rb_b1w_s5104"}]})
    return F.ArmSpec(card="rehearsal", pin=F.B1_LAUNCH_PIN, banked_arm="C", lever=("--n-epochs", "5"),
                     declared_keys=("n_epochs",), fresh=(F.Run(5101, "rb_b1c_s5101"), F.Run(5103, "rb_b1w_s5103")),
                     own_control=(F.Run(5102, "rb_b1c_s5102"), F.Run(5104, "rb_b1w_s5104")))


def test_the_kit_plans_plays_and_reads_real_runs_end_to_end_on_cpu(tmp_path):
    from utils.paths import main_models_dir

    models = main_models_dir()
    if models is None or not all((models / r).is_dir() for r in POSING):
        pytest.skip("the main run archive with B1's runs is not here")
    pin_dir = Path(str(PIN_DIR))
    out = Path(os.environ.get("GEN3AI_FRESH_REHEARSAL_OUT") or tmp_path)
    work, ledger = out / "work", out / "ledger"
    spec = _spec()
    resolver = F.pin_tree_resolver(pin_dir, spec.pin)

    # 1. the registered guards refuse runs that are not the declared lever over their control
    doc, probs = F.plan(models, spec, "50M", resolver, step=STEP)
    assert doc is None
    assert any("rb_b1c_s5101: original_command is not the registered argv of rb_b1c_s5101" in p for p in probs)
    assert any("rb_b1c_s5101: cli_args.n_epochs 10 is not its argv's 5" in p for p in probs)
    assert any("rb_b1w_s5104: original_command is not the registered argv" in p for p in probs)

    # 2. the tests' override: each run held to B1's own guard for the arm it REALLY is (levers from the pin's parser)
    levers = {r: dict(zip(K.CONFIG_KEYS, (ns.get(k) for k in K.CONFIG_KEYS)))
              for r, ns in zip(POSING, resolver([A.build_argv(K.KIT_ARM[POSING[r]], int(r.split("_s")[1]), spec.pin)
                                                 for r in POSING]))}

    def as_b1(side: str, run_dir: Path, run: F.Run, step: int):
        return K.check_run(run_dir, POSING[run.run], run.seed, spec.pin, step, levers[run.run])

    def one_runtime(_spec: F.ArmSpec, facts: Mapping[str, Sequence[Dict[str, Any]]]) -> List[str]:
        every = [f for s in FR.SIDES for f in facts[s]]
        for f in every:
            f.pop("_cli", None)
            f.pop("_true", None)
        return [w for w in ("torch_version", "init_num_threads") if len({str(f.get(w)) for f in every}) > 1]

    doc, probs = F.plan(models, spec, "50M", resolver, guard=as_b1, cross=one_runtime, step=STEP)
    assert probs == [] and doc is not None and len(doc["cells"]) == 4
    assert all(f["steps"] < STEP + A.CHECKPOINT_EVERY for s in FR.SIDES for f in doc["runs"][s])
    work.mkdir(parents=True, exist_ok=True)
    (work / "cells_plan.json").write_text(json.dumps(doc["cells"], indent=1))
    (work / "look.json").write_text(json.dumps(doc, indent=1, default=str))

    # 3. play at the pin on CPU (rule 5b's shape), a scratch ledger, a few pairs a cell; a CPU panel takes over ten
    #    minutes at nice 19 (measured: ~125 s a cell at 4 pairs), so it is a normal heavy job, not the short lane
    assert F.play(work, pin_dir, "cpu", ledger=ledger, pairs=PAIRS) == 0, (work / "run.log").read_text()

    # 4. the registered read refuses the short cells; the tests' pairs read the cross
    _inp, short = F.read_ledger(ledger, doc)
    assert short["decision"]["outcome"] == FR.INCONCLUSIVE
    assert all("completed pairs < the registered" in r or "drawn" in r for r in short["decision"]["reasons"])
    inp, verdict = F.read_ledger(ledger, doc, pairs=PAIRS)
    (work / "result").mkdir(exist_ok=True)
    (work / "result" / "inputs.json").write_text(json.dumps(inp, indent=1, default=str))
    (work / "result" / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    outcome = verdict["decision"]["outcome"]
    assert outcome in (FR.CONTINUE, FR.STOPPED_AT_50M) or all("drawn" in r for r in verdict["decision"]["reasons"])
    same, _ = F.recheck(work)
    assert same
    if outcome != FR.INCONCLUSIVE:
        assert set(verdict["views"]) == set(X.VIEWS) and verdict["views"]["N"]["df"] == 2
        cells, rows = F.ledger_rows(ledger, doc)
        try:
            rep: Dict[str, Any] = F.bootstrap(doc, cells, rows, draws=500, pairs=PAIRS)
        except ValueError as e:  # a few teams a panel may not survive a resample: REPORTED, and said
            rep = {"status": "REPORTED", "refused": str(e)}
        (work / "result" / "team_bootstrap.json").write_text(json.dumps(rep, indent=1, default=str))
        assert rep["status"] == "REPORTED"
