"""The fresh kit's banked cell at a look B1 played only as a SURVIVOR look (FRAME §2 item 6; ``main.b1.fresh.
b1_look_problems``): B1's 100M look read W and C alone, and cell C's 100M files are accepted from it iff the look's
rows for C are exactly the banked cell's three runs and disk, look and B1's committed record agree on every step and
sha256. Every other survivor look is refused, and a full look is read as before. No game is played."""
import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pytest

from main.b1 import arms as A
from main.b1 import fresh as F
from main.b1 import fresh_rule as FR
from main.b1 import look as K
from main.b1.fresh_test import PIN, _stub_k9b, b1_look_of, make_cell_run, spec_of, stub_resolver
from main.b1.look_test import make_run

M100 = 100_000_000
STEPS: Tuple[int, ...] = (50_000_081, 72_000_030, 100_000_044)
Record = Dict[str, Dict[str, Any]]


@pytest.fixture()
def world(tmp_path) -> Tuple[Path, F.ArmSpec]:
    """C's and W's three runs at 50M and 100M (B1's survivor arms) and the card's fresh cell."""
    models = tmp_path / "models"
    spec = spec_of()
    for arm in ("C", "W"):
        for seed in A.SEEDS[K.KIT_ARM[arm]]:
            make_run(models, arm, seed, pin=PIN, steps=STEPS)
    for r in spec.fresh:
        make_cell_run(models, spec, r)
    return models, spec


def survivor_look(models: Path, look: str = "100M", survivor: str = "width") -> Dict[str, Any]:
    """B1's survivor look.json as its kit writes one for ``--survivor width``: the rows of W and C, nothing else."""
    rows = {arm: b1_look_of(models, look, arm)["runs"][arm] for arm in ("C", "W")}
    return {"look": look, "pin": PIN, "survivor": survivor, "replaced_seed": None, "eq_step": None,
            "steps": {arm: F.LOOK_STEPS[look] for arm in rows}, "runs": rows}


def record_of(b1_look: Dict[str, Any], arm: str = "C") -> Record:
    """B1's committed record of that look, in :data:`main.b1.fresh.B1_SURVIVOR_RECORD`'s shape."""
    return {b1_look["look"]: {"survivor": b1_look["survivor"], "banked_arm": arm, "checkpoints": {
        r["run"]: (r["steps"], r["sha256"]) for r in b1_look["runs"][arm]}}}


def plan(models: Path, spec: F.ArmSpec, b1_look: Dict[str, Any], record: Optional[Record], look: str = "100M",
         **kw: Any):
    kw.setdefault("k9b", _stub_k9b)
    return F.plan(models, spec, look, stub_resolver, b1_look=b1_look, b1_record=record, **kw)


def rewrite(models: Path, run: str, step: int = 100_000_044) -> None:
    (models / run / "checkpoints" / f"checkpoint_{step}_steps.zip").write_bytes(b"other bytes")


# ------------------------------------------------------------------------------------------- the acceptance
def test_cell_cs_100m_files_are_accepted_from_b1s_survivor_look_when_disk_look_and_record_agree(world):
    models, spec = world
    look = survivor_look(models)
    doc, probs = plan(models, spec, look, record_of(look))
    assert probs == [] and doc is not None
    assert doc["banked_identity"] == {"checked": True, "b1_look": "100M", "pin": PIN, "survivor": "width"}
    assert [f["run"] for f in doc["runs"][FR.BANKED]] == ["rb_b1c_s5101", "rb_b1c_s5102", "rb_b1c_s5110"]
    assert [(f["steps"], f["sha256"]) for f in doc["runs"][FR.BANKED]] == [
        (r["steps"], r["sha256"]) for r in look["runs"]["C"]]
    full, probs = plan(models, spec, b1_look_of(models, "100M"), None)      # the same look doc as a full look's ...
    assert probs == [] and {k: v for k, v in doc.items() if k != "banked_identity"} == {
        k: v for k, v in full.items() if k != "banked_identity"}
    assert full["banked_identity"] == {"checked": True, "b1_look": "100M", "pin": PIN}   # ... whose record is unchanged


def test_the_matched_gpu_time_cross_plans_from_a_100m_look_read_off_the_survivor_look(world):
    from fractions import Fraction

    models, spec = world
    look = survivor_look(models)
    doc, probs = plan(models, spec, look, record_of(look))
    assert probs == []
    cross, probs = F.plan_gpu_time(models, spec, stub_resolver, json.loads(json.dumps(doc, default=str)),
                                   k9b=_stub_k9b, rho=(Fraction(7127, 10000), {"meter": "test"}))
    assert probs == [] and cross["cross"]["banked_step_target"] == 72_000_000
    assert sorted(cross["cross"]["banked_checkpoint_steps"].values()) == [72_000_030] * 3


# --------------------------------------------------------------------------------------------- the refusals
def _extra_row(look: Dict[str, Any], rec: Record) -> None:
    look["runs"]["C"].append({**look["runs"]["C"][0], "run": "rb_b1c_s5109"})


def _extra_row_and_record(look: Dict[str, Any], rec: Record) -> None:
    _extra_row(look, rec)
    rec["100M"]["checkpoints"]["rb_b1c_s5109"] = rec["100M"]["checkpoints"]["rb_b1c_s5101"]


def _row_twice(look: Dict[str, Any], rec: Record) -> None:
    look["runs"]["C"].append(dict(look["runs"]["C"][2]))


def _missing_row(look: Dict[str, Any], rec: Record) -> None:
    del look["runs"]["C"][1]


def _record_sha(look: Dict[str, Any], rec: Record) -> None:
    rec["100M"]["checkpoints"]["rb_b1c_s5102"] = (100_000_044, "0" * 64)


def _record_step(look: Dict[str, Any], rec: Record) -> None:
    rec["100M"]["checkpoints"]["rb_b1c_s5110"] = (100_000_045, rec["100M"]["checkpoints"]["rb_b1c_s5110"][1])


def _record_of_other_runs(look: Dict[str, Any], rec: Record) -> None:
    rec["100M"]["checkpoints"]["rb_b1c_s5109"] = rec["100M"]["checkpoints"].pop("rb_b1c_s5110")


def _record_short(look: Dict[str, Any], rec: Record) -> None:
    del rec["100M"]["checkpoints"]["rb_b1c_s5101"]


@pytest.mark.parametrize("mutate,want", [
    (_extra_row, ["B1's 100M survivor look reads ['rb_b1c_s5101', 'rb_b1c_s5102', 'rb_b1c_s5109', 'rb_b1c_s5110'] for "
                  "cell C, not exactly the banked cell's ['rb_b1c_s5101', 'rb_b1c_s5102', 'rb_b1c_s5110']"]),
    (_extra_row_and_record, ["B1's 100M survivor look reads", "B1's record of its 100M survivor look names "
                             "['rb_b1c_s5101', 'rb_b1c_s5102', 'rb_b1c_s5109', 'rb_b1c_s5110'] for cell C, not the "
                             "banked cell's ['rb_b1c_s5101', 'rb_b1c_s5102', 'rb_b1c_s5110']"]),
    (_row_twice, ["B1's 100M survivor look reads ['rb_b1c_s5101', 'rb_b1c_s5102', 'rb_b1c_s5110', 'rb_b1c_s5110']"]),
    (_missing_row, ["rb_b1c_s5102: not in B1's 100M look", "B1's 100M survivor look reads ['rb_b1c_s5101', "
                    "'rb_b1c_s5110'] for cell C, not exactly"]),
    (_record_sha, ["rb_b1c_s5102: B1's 100M survivor look says step 100000044 sha"]),
    (_record_step, ["rb_b1c_s5110: B1's 100M survivor look says step 100000044 sha"]),
    (_record_of_other_runs, ["B1's record of its 100M survivor look names ['rb_b1c_s5101', 'rb_b1c_s5102', "
                             "'rb_b1c_s5109'] for cell C"]),
    (_record_short, ["B1's record of its 100M survivor look names ['rb_b1c_s5102', 'rb_b1c_s5110'] for cell C"]),
], ids=lambda v: v.__name__.strip("_") if callable(v) else None)
def test_a_survivor_look_whose_rows_or_record_differ_from_the_banked_cell_is_refused(
        world, mutate: Callable[[Dict[str, Any], Record], None], want):
    models, spec = world
    look = survivor_look(models)
    rec = record_of(look)
    mutate(look, rec)
    doc, probs = plan(models, spec, look, rec)
    assert doc is None and len(probs) == len(want)
    for got, prefix in zip(probs, want):
        assert got.startswith(prefix)


def test_a_file_that_no_longer_hashes_to_the_look_or_to_b1s_record_is_refused(world):
    models, spec = world
    look = survivor_look(models)
    rec = record_of(look)
    rewrite(models, "rb_b1c_s5102")                       # the bytes on disk changed after B1's look
    doc, probs = plan(models, spec, look, rec)
    assert doc is None and len(probs) == 1 and probs[0].startswith("rb_b1c_s5102: its 100M file is step 100000044 sha")
    now = survivor_look(models)                           # a look.json rewritten to the new bytes: disk == look
    doc, probs = plan(models, spec, now, rec)             # ... but not B1's record
    assert doc is None and probs == [
        f"rb_b1c_s5102: B1's 100M survivor look says step 100000044 sha {now['runs']['C'][1]['sha256'][:12]}, B1's "
        f"record step 100000044 sha {look['runs']['C'][1]['sha256'][:12]}"]


def test_without_the_tests_record_the_kit_holds_the_look_to_b1s_real_record(world):
    """The CLI's path: no ``b1_record``, so :data:`F.B1_SURVIVOR_RECORD` decides, and these files are not B1's."""
    models, spec = world
    doc, probs = plan(models, spec, survivor_look(models), None)
    assert doc is None and len(probs) == 3
    for got, (run, (step, sha)) in zip(probs, F.B1_SURVIVOR_RECORD["100M"]["checkpoints"].items()):
        assert got.startswith(f"{run}: B1's 100M survivor look says step 100000044 sha")
        assert got.endswith(f"B1's record step {step} sha {sha[:12]}")


def test_a_survivor_look_of_another_factor_look_pin_or_arm_stays_refused(world):
    models, spec = world
    good = survivor_look(models)
    rec = record_of(good)
    breadth = {**good, "survivor": "breadth"}
    doc, probs = plan(models, spec, breadth, rec)
    assert doc is None and probs == [f"the B1 look given is the '100M' look at {PIN[:12]} (survivor 'breadth'), not "
                                     f"B1's full 100M look at {PIN[:12]}, nor the survivor look B1's record names for "
                                     "cell C at 100M"]
    doc, probs = plan(models, spec, {**good, "pin": "d" * 40}, rec)
    assert doc is None and "not B1's full 100M look" in probs[0]
    doc, probs = plan(models, spec, good, rec, look="50M")                  # the 100M survivor look at the 50M look
    assert doc is None and "the '100M' look" in probs[0] and "not B1's full 50M look" in probs[0]
    at50 = survivor_look(models, "50M")                    # a survivor look AT 50M: B1's record names none there
    for record in (None, rec, record_of(good, arm="W")):
        doc, probs = plan(models, spec, at50, record, look="50M")
        assert doc is None and probs[0].endswith("nor the survivor look B1's record names for cell C at 50M")
    doc, probs = plan(models, spec, good, record_of(good, arm="W"))        # a record that serves the other arm
    assert doc is None and probs[0].endswith("nor the survivor look B1's record names for cell C at 100M")
    banked_w = F.arm_from_doc({**spec.as_dict(), "banked_arm": "W"})
    rows = [dict(r, seed=0) for r in good["runs"]["W"]]
    assert F.b1_look_problems(good, banked_w, "100M", rows, record_of(good, arm="W")) == []   # only by a tests' record
    assert F.b1_look_problems(good, banked_w, "100M", rows)[0].endswith(           # the kit's own record: cell C only
        "nor the survivor look B1's record names for cell W at 100M")


def test_the_run_guard_still_refuses_a_banked_run_under_a_good_survivor_look(world):
    """K9(b) at the step is the run guard's, read before the look is compared: a FAIL refuses whatever the look says."""
    models, spec = world
    look = survivor_look(models)

    def k9b(run_dir: Path, upto: Optional[int]) -> Dict[str, Any]:
        rec = _stub_k9b(run_dir, upto)
        return {**rec, "verdict": "FAIL", "reasons": ["a canary fired"]} if run_dir.name == "rb_b1c_s5102" else rec

    doc, probs = plan(models, spec, look, record_of(look), k9b=k9b)
    assert doc is None and len(probs) == 1 and probs[0].startswith("rb_b1c_s5102: ") and "K9(b)" in probs[0]
    md = models / "rb_b1c_s5110" / "metadata.json"         # and B1's own guard: a run launched by another argv
    md.write_text(md.read_text().replace("--ent-coef 0.0125", "--ent-coef 0.05"))
    doc, probs = plan(models, spec, look, record_of(look))
    assert doc is None and probs[0].startswith("rb_b1c_s5110: original_command is not the registered argv")


# ------------------------------------------------------------------------------------------ a full look, unchanged
@pytest.mark.parametrize("look", ["50M", "100M"])
def test_a_full_look_is_read_as_before_and_its_look_doc_records_no_survivor(world, look):
    models, spec = world
    full = b1_look_of(models, look)
    doc, probs = plan(models, spec, full, None, look=look)
    assert probs == [] and doc["banked_identity"] == {"checked": True, "b1_look": look, "pin": PIN}
    extra = copy.deepcopy(full)                            # a full look's other rows are not this cell's business
    extra["runs"]["C"].append({**extra["runs"]["C"][0], "run": "rb_b1c_s5109"})
    assert plan(models, spec, extra, None, look=look)[1] == []
    other = copy.deepcopy(full)
    other["runs"]["C"][0]["sha256"] = "0" * 64
    doc, probs = plan(models, spec, other, None, look=look)
    assert doc is None and len(probs) == 1 and probs[0].startswith(f"rb_b1c_s5101: its {look} file is step")


# ------------------------------------------------------------------------------------- the record, and the CLI
def test_the_kits_survivor_record_is_b1s_committed_100m_record_for_cell_c():
    from utils.paths import repo_path

    verdict = json.loads(repo_path("designs/research_state/measurements/boundary1_scratch_2026-11-09/record/look_100M/"
                                   "verdict.json").read_text())
    assert set(F.B1_SURVIVOR_RECORD) == {"100M"}
    rec = F.B1_SURVIVOR_RECORD["100M"]
    assert (verdict["look"], verdict["pin"], verdict["survivor"]) == ("100M", F.B1_LAUNCH_PIN, rec["survivor"])
    assert rec["banked_arm"] == "C" and rec["banked_arm"] in F.BANKED_ARMS
    cell = [r.run for r in F.banked_runs(spec_of())]
    assert verdict["runs"]["C"] == cell == list(rec["checkpoints"])
    assert {run: (verdict["checkpoints"][run]["steps"], verdict["checkpoints"][run]["sha256"]) for run in cell} == {
        run: tuple(v) for run, v in rec["checkpoints"].items()}
    for step, sha in rec["checkpoints"].values():
        assert M100 <= step < M100 + A.CHECKPOINT_EVERY and len(sha) == 64


def test_the_cli_has_no_flag_for_the_survivor_record(capsys):
    with pytest.raises(SystemExit):
        F.main(["plan", "--help"])
    text = capsys.readouterr().out
    assert "--b1-look" in text and "record" not in text
