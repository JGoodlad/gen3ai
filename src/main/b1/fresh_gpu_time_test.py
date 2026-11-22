"""The fresh kit's REPORTED matched-GPU-time cross (``python -m main.b1.fresh plan-gpu-time``, Q5-N PREREG §3): the
banked step from rho in exact arithmetic, the plan's guards and refusals, the step-time meter it is read by, and the
read that reports the statistic without a class. No game is played."""
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pytest

from main.b1 import arms as A
from main.b1 import cross as X
from main.b1 import fresh as F
from main.b1 import fresh_rule as FR
from main.b1 import look as K
from main.b1 import lr_report as LR
from main.b1.fresh_test import PIN, _stub_k9b, b1_look_of, make_cell_run, spec_of, stub_resolver
from main.b1.look_test import make_run
from main.b1.lr_report_meters_test import ROLL, timing, write_tags

M100 = 100_000_000
#: The banked runs' checkpoints: the two looks', and one just past each 1M multiple the tests' rhos name.
BANKED_STEPS: Tuple[int, ...] = (50_000_081, 72_000_030, 84_000_017, 100_000_044)


def build(tmp: Path, banked_steps: Sequence[int] = BANKED_STEPS) -> Tuple[Path, F.ArmSpec]:
    models = tmp / "models"
    spec = spec_of()
    for seed in A.SEEDS["c"]:
        make_run(models, "C", seed, pin=PIN, steps=tuple(banked_steps))
    for r in spec.fresh:
        make_cell_run(models, spec, r)
    return models, spec


def fresh_look(models: Path, spec: F.ArmSpec) -> Dict[str, Any]:
    """The card's own 100M look, as ``plan --look 100M`` writes it."""
    doc, probs = F.plan(models, spec, "100M", stub_resolver, b1_look=b1_look_of(models, "100M"), k9b=_stub_k9b)
    assert probs == [] and doc is not None
    return json.loads(json.dumps(doc, default=str))


def cross_plan(models: Path, spec: F.ArmSpec, look: Dict[str, Any], rho: Optional[Fraction] = Fraction(7126, 10000)):
    return F.plan_gpu_time(models, spec, stub_resolver, look, k9b=_stub_k9b,
                           rho=None if rho is None else (rho, {"meter": "test"}))


# ---------------------------------------------------------------------------------------------- the banked step
@pytest.mark.parametrize("rho,m", [
    (Fraction(59, 100), 59_000_000),                      # exactly on a multiple: that multiple, not the next
    (Fraction(59, 100) + Fraction(1, 10 ** 12), 60_000_000),
    (Fraction(59, 100) - Fraction(1, 10 ** 12), 59_000_000),
    (Fraction(59, 100) + Fraction(1, 10 ** 18), 60_000_000),      # below a float's resolution: float math says 59M
    (Fraction(7, 100), 7_000_000), (Fraction(55, 100), 55_000_000),  # float math rounds these UP a whole interval
    (Fraction(0.59), 59_000_000),                         # the float's exact value lies just below 0.59
    (Fraction(1, 2), 50_000_000), (Fraction(5, 6), 84_000_000), (Fraction(99, 100), 99_000_000),
    (Fraction(1, 10 ** 9), 1_000_000)])
def test_the_banked_step_is_the_first_1m_multiple_at_or_after_rho_times_100m_exactly(rho, m):
    assert F.matched_gpu_step(rho) == m


@pytest.mark.parametrize("rho", [Fraction(991, 1000), Fraction(1), Fraction(7, 5)])
def test_a_rho_whose_banked_step_reaches_the_look_has_no_cross(rho):
    with pytest.raises(F.FreshError, match="no matched-GPU-time cross to play"):
        F.matched_gpu_step(rho)


@pytest.mark.parametrize("rho", [Fraction(0), Fraction(-1, 2), 0.7, 1])
def test_rho_must_be_a_positive_exact_ratio(rho):
    with pytest.raises(F.FreshError, match="not a positive exact ratio"):
        F.matched_gpu_step(rho)


# ---------------------------------------------------------------------------------------------- the plan
def test_the_cross_plays_the_100m_looks_fresh_files_against_the_banked_runs_first_checkpoint_at_m(tmp_path):
    models, spec = build(tmp_path)
    look = fresh_look(models, spec)
    doc, probs = cross_plan(models, spec, look)
    assert probs == [] and doc is not None
    assert doc["look"] == "100M" and doc["step"] == M100 and doc["schema"] == F.LOOK_SCHEMA
    cx = doc["cross"]
    assert cx["kind"] == F.GPU_TIME and cx["banked_step_target"] == 72_000_000 and cx["rho_exact"] == "3563/5000"
    assert cx["banked_checkpoint_steps"] == {r: 72_000_030 for r in ("rb_b1c_s5101", "rb_b1c_s5102", "rb_b1c_s5110")}
    assert [(f["run"], f["steps"], f["sha256"]) for f in doc["runs"]["fresh"]] == \
        [(f["run"], f["steps"], f["sha256"]) for f in look["runs"]["fresh"]], "the 100M look's files, byte for byte"
    assert all(f["steps"] == 72_000_030 for f in doc["runs"]["banked"])
    assert len(doc["cells"]) == 9 and all("72000030" in o and "100000044" in p for p, o in doc["cells"])
    assert doc["names"]["home72"]["request"] == "b1fresh_q5n_100m_gputime_home72"
    assert {p: n["family"] for p, n in doc["names"].items()} == {p: n["family"] for p, n in look["names"].items()}
    assert all(doc["names"][p]["request"] != look["names"][p]["request"] for p in X.PANELS), "its own requests"
    assert doc["banked_identity"]["checked"] is False and doc["lever_check"]["ok"]


def test_the_cross_refuses_anything_but_the_cards_own_100m_look(tmp_path):
    models, spec = build(tmp_path)
    look = fresh_look(models, spec)
    for over, why in (({"look": "50M"}, "look is '50M'"), ({"card": "q6n"}, "card is 'q6n'"),
                      ({"pin": "d" * 40}, "pin is"), ({"arm_sha256": "x"}, "arm_sha256 is 'x'"),
                      ({"schema": "other"}, "schema is 'other'"), ({"cross": {"kind": F.GPU_TIME}}, "itself a cross")):
        doc, probs = cross_plan(models, spec, {**look, **over})
        assert doc is None and any(why in p for p in probs), (over, probs)


def test_a_fresh_file_that_changed_since_the_100m_look_refuses(tmp_path):
    models, spec = build(tmp_path)
    look = fresh_look(models, spec)
    z = Path(look["runs"]["fresh"][1]["checkpoint"])
    z.write_bytes(b"rewritten")
    doc, probs = cross_plan(models, spec, look)
    assert doc is None and len(probs) == 1 and probs[0].startswith("rb_q5n_s5202: its 100M file is step 100000044 sha")


def test_a_banked_run_without_a_checkpoint_within_an_interval_of_m_refuses_by_name(tmp_path):
    models, spec = build(tmp_path, banked_steps=(50_000_081, 73_500_000, 100_000_044))
    doc, probs = cross_plan(models, spec, fresh_look(models, spec))
    assert doc is None and all("more than one interval" in p for p in probs) and len(probs) == 3
    doc, probs = cross_plan(models, spec, fresh_look(models, spec), Fraction(1))
    assert doc is None and probs and "no matched-GPU-time cross" in probs[0]


def test_the_banked_runs_are_held_to_b1s_guard_at_m(tmp_path):
    models, spec = build(tmp_path)
    look = fresh_look(models, spec)
    side = models / "rb_b1c_s5110" / "checkpoints" / "checkpoint_72000030_steps.json"
    side.write_text(json.dumps({**json.loads(side.read_text()), "ent_coef": 0.05}))
    doc, probs = cross_plan(models, spec, look)
    assert doc is None and probs == ["rb_b1c_s5110: checkpoint_72000030_steps.zip ent_coef 0.05 != 0.0125"]


def test_rho_is_read_by_the_step_time_meter_on_all_six_runs_and_none_is_dropped(tmp_path):
    """Without the tests' rho, the plan reads every run's event files: fresh 25 s, banked 30 s a 1M steps, rho 5/6,
    the banked files at 84M."""
    models, spec = build(tmp_path)
    look = fresh_look(models, spec)
    for r in [*spec.fresh, *F.banked_runs(spec)]:
        write_tags(models / r.run, timing(ROLL, 30.0 if r.run.startswith("rb_q5n") else 40.0, 20_000.0, 1.0), 1)
    doc, probs = cross_plan(models, spec, look, rho=None)
    assert probs == [] and doc is not None
    cx = doc["cross"]
    assert (cx["rho_exact"], cx["banked_step_target"]) == ("5/6", 84_000_000)
    st = cx["step_time"]
    assert st["meter"] == "main.b1.lr_report.step_time" and st["window"] == list(LR.STEP_TIME_WINDOW)
    assert [r["sec_per_1M_steps"] for r in st["runs"]["fresh"]] == [25.0] * 3
    assert [r["run"] for r in st["runs"]["banked"]] == ["rb_b1c_s5101", "rb_b1c_s5102", "rb_b1c_s5110"]
    write_tags(models / "rb_b1c_s5102", {LR.COLLECT_TAG: {s: 1.0 for s in ROLL if s <= 30_000_000}}, 2)
    doc, probs = cross_plan(models, spec, look, rho=None)
    assert doc is None and probs[0].startswith("the step-time meter cannot be read, and no run is dropped: "
                                               "rb_b1c_s5102: its rollout rows")


def test_the_cli_plans_the_cross_into_its_own_work_directory_and_has_no_flag_for_the_tests_rho(tmp_path, capsys,
                                                                                                monkeypatch):
    models, spec = build(tmp_path)
    arm = tmp_path / "fresh_arm_q5n.json"
    arm.write_text(json.dumps(spec.as_dict()))
    spec = F.load_arm(arm)                                                   # the look records the file's sha256
    look = fresh_look(models, spec)
    w100 = tmp_path / "w100"
    w100.mkdir()
    (w100 / "look.json").write_text(json.dumps(look))
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(models))
    monkeypatch.setattr(F, "pin_tree_resolver", lambda tree, pin: stub_resolver)
    base = ["plan-gpu-time", "--arm", str(arm), "--pin-tree", str(tmp_path), "--fresh-look", str(w100 / "look.json")]
    assert F.main([*base, "--out", str(w100)]) == 2
    assert "the 100M look's own work directory" in capsys.readouterr().err
    assert F.main([*base, "--out", str(tmp_path / "wg")]) == 3                 # no event files: the meter refuses
    assert "the step-time meter cannot be read" in capsys.readouterr().err and not (tmp_path / "wg").exists()
    for r in [*spec.fresh, *F.banked_runs(spec)]:
        write_tags(models / r.run, timing(ROLL, 30.0 if r.run.startswith("rb_q5n") else 40.0, 20_000.0, 1.0), 1)
    monkeypatch.setattr(K, "k9b_read", _stub_k9b)
    assert F.main([*base, "--out", str(tmp_path / "wg")]) == 0
    io = capsys.readouterr()
    assert io.out.strip() == "9" and "REPORTED matched_gpu_time: rho 0.833333, banked files at or after 84,000,000" in io.err
    written = json.loads((tmp_path / "wg" / "look.json").read_text())
    assert written["cross"]["banked_step_target"] == 84_000_000
    assert json.loads((tmp_path / "wg" / "cells_plan.json").read_text()) == written["cells"]
    with pytest.raises(SystemExit):
        F.main(["plan-gpu-time", "--help"])
    text = capsys.readouterr().out
    assert "--fresh-look" in text and not any(f in text for f in ("--rho", "--guard", "--k9b", "--step "))


# ---------------------------------------------------------------------------------------------- the read
def test_a_cross_verdict_reports_the_statistic_and_reads_no_class():
    v = {"look": "100M", "views": {"N": {"estimate_pp": 1.0}},
         "decision": {"verdict": X.WIN, "outcome": X.WIN, "at_boundary": ["x"], "labels": ["y"], "reasons": [],
                      "words": f"{X.WIN} on N: fresh - banked = +4.00 pp [+1.00, +7.00] (90 %, 4 df)"}}
    got = F.cross_verdict(v, {"kind": F.GPU_TIME, "banked_step_target": 72_000_000})
    assert got["decision"] == {"verdict": "REPORTED", "outcome": "REPORTED", "at_boundary": [], "labels": [],
                               "reasons": [], "words": "REPORTED on N: fresh - banked = +4.00 pp [+1.00, +7.00] "
                                                       "(90 %, 4 df)"}
    assert got["decides"] is None and got["cross"]["banked_step_target"] == 72_000_000 and got["views"] == v["views"]
    short = {"decision": {"verdict": FR.INCONCLUSIVE, "outcome": FR.INCONCLUSIVE, "reasons": ["short"]}}
    assert F.cross_verdict(short, {"kind": F.GPU_TIME})["decision"] == short["decision"]


@pytest.mark.integration
def test_the_cross_is_read_from_a_real_ledger_reported_rechecked_and_never_bootstrapped(tmp_path, monkeypatch):
    from main.b1.look_test import _write_look

    models, spec = build(tmp_path)
    doc, _ = cross_plan(models, spec, fresh_look(models, spec))
    assert doc is not None
    root = tmp_path / "ledger"
    _write_look(root, doc, commit=PIN)
    _inp, v = F.read_ledger(root, doc, pairs={p: 2 for p in X.PANELS})
    got = F.cross_verdict(v, doc["cross"])
    assert v["decision"]["outcome"] == "NO-USEFUL-GAIN" and got["decision"]["outcome"] == F.REPORTED
    work = tmp_path / "wg"
    work.mkdir()
    (work / "look.json").write_text(json.dumps(doc))
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(models))
    assert F.main(["read", "--work", str(work), "--ledger", str(root)]) == 0
    verdict = json.loads((work / "result" / "verdict.json").read_text())
    assert verdict["decision"]["outcome"] == FR.INCONCLUSIVE and verdict["cross"]["kind"] == F.GPU_TIME
    assert json.loads((work / "result" / "inputs.json").read_text())["cross"] == doc["cross"]
    assert F.main(["recheck", "--work", str(work)]) == 0
    assert F.main(["bootstrap", "--work", str(work), "--ledger", str(root)]) == 2
