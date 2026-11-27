"""§9's two REPORTED end crosses (``main.x72.cross --end``): the restated registration, the plans (9 cells x 4,000
and 24 cells x 1,000; the one that IS round 1's cross), X70's ``rect_stat`` run from its frozen file against three
banked records, the report that is never a verdict, and the driver with a recorded runner. No game is played and
nothing touches ``models/``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.x72 import cross as X
from main.x72 import finals as F
from main.x72.cross_test import BANKED, Runner, _other_venue, play, register, rule  # noqa: F401  (rule: a fixture)
from main.x72.finals_test import DIR, c0_world, train_round, world  # noqa: F401  (fixtures)
from utils.paths import repo_path

M = repo_path("designs/research_state/measurements")
CE3D = M / "ce3d_magnet_screen_2026-11-11" / "record" / "deciding"
X70 = M / "x70_ent0_control_2026-11-13" / "record"
#: CE-3d's 24 banked deciding cells (swap finals x C0 finals, 1,000 pairs): their pentanomials, by cell index
CE3D_CELLS = {n: c["pairs"] for n, c in enumerate(json.loads((CE3D / "cells.json").read_text()))}
CE3D_H = json.loads((CE3D / "h.json").read_text())["h_pp"]
CE3D_V = json.loads((CE3D / "verdict.json").read_text())
ROUND_DOC_KEYS = {"cross", "round", "pairs", "batch_pairs", "family", "request", "rehearsal", "play_pin", "rows", "cols",
                  "row_seeds", "col_seeds", "row_runs", "col_runs", "row_files", "col_files", "row_steps", "col_steps"}


@pytest.fixture()
def ended(c0_world, tmp_path):  # noqa: F811
    """Rounds 1 and 2 archived, the eight C0 finals registered: the keyword arguments of ``drive``."""
    models, root = c0_world
    for k in (1, 2):
        train_round(models, k)
        F.archive(k, root=root, models_dir=models)
    return dict(ledger_root=tmp_path / "ledger", archive_root=root, models_dir=models, py="PY", cpus="5-7")


def planned(ended, tmp_path, end: str, k: int, **kw):
    work = tmp_path / f"work_{end}_{k}"
    cells, shas = X.plan_end(end, k, root=ended["archive_root"], models_dir=ended["models_dir"], **kw)
    X.write_plan(work, cells, shas)
    return work, cells, shas


def files(work: Path) -> set:
    return {p.name for p in work.iterdir()}


# ----------------------------------------------------------------------------------- the restated registration
def test_the_end_crosses_are_the_registrations():
    text = " ".join((DIR / "PREREG.md").read_text().split())
    assert ("**Two end crosses on the card**, X70's protocol, at the play pin, once each after the STOP, when the "
            "read round is ≥ 1: the read finals × the three banked swap finals, 9 cells × 4,000 pairs (it IS round "
            "1's cross when the read round is 1;") in text
    assert ("the read finals × the eight C0 finals, 24 cells × 1,000 pairs (≈ 0.27 GPU-h), beside the banked "
            "+3.97 pp. X70's `rect_stat` interval, no class.") in text
    assert {k: v[:2] for k, v in X.ENDS.items()} == {"swap": (4000, "x72_end_swap_h2h"), "c0": (1000, "x72_end_c0_h2h")}
    assert X.reuses("swap", 1) and not any(X.reuses(e, k) for e, k in (("swap", 2), ("c0", 1), (None, 1)))
    rule_text = X.family_rule({"rehearsal": False, "end": "c0", "reuses": None})
    assert "§9: REPORTED, decides nothing, no class" in rule_text and "the eight banked C0 finals, 1,000" in rule_text
    assert X.family_rule({"rehearsal": False, "end": "swap", "reuses": "x72_round_1"}) == X.FAMILY_RULE
    assert X.family_rule({"rehearsal": False}) == X.FAMILY_RULE and len({rule_text, X.FAMILY_RULE, X.REHEARSAL_RULE}) == 3


def test_the_banked_3_97_pp_is_ce3ds_deciding_cross_of_the_same_files_at_the_same_pin():
    """The REAL tables (no stand-in): the number printed beside the C0 end cross is the swap finals x the same
    eight C0 files, 1,000 pairs a cell, at the play pin; it was played on CPU, which the report says."""
    shas = json.loads((CE3D / "shas.json").read_text())
    assert shas["rows"] == [F.SWAP[s][1] for s in F.SEEDS] and shas["cols"] == list(F.C0.values())
    assert {c["n_pairs"] for c in json.loads((CE3D / "cells.json").read_text())} == {X.ENDS["c0"][0]}
    log = (CE3D / "run.log").read_text()
    assert f'"commit": "{X.PLAY_PIN}"' in log and '"family_id": "ce3d_swap_vs_c0_h2h"' in log
    assert log.count("engine up") == log.count("T2 eager on cpu") == 1
    b = X.banked_c0(X.x70_rule())
    assert round(b["delta_hat_pp"], 2) == 3.97 and b["delta_hat_pp"] == CE3D_V["delta_hat_pp"]
    assert b["interval90_pp"] == CE3D_V["interval90_pp"] and b["shape"] == [3, 8]
    assert b["source"] == f"{X.CE3D_DECIDING}/verdict.json" and b["played_on"].startswith("cpu / eager")
    assert repo_path(b["source"]).is_file() and CE3D == repo_path(X.CE3D_DECIDING)


# ---------------------------------------------------------------------------------------------- the statistic
def test_the_statistic_is_x70s_rect_stat_run_from_its_frozen_file(rule):  # noqa: F811
    x70 = X.x70_rule()
    assert repo_path(X.X70_RULE) == M / "x70_ent0_control_2026-11-13" / "screen_rule.py"
    # X70's own reported C0 cross (4 x 8 at 1,000 pairs), from its banked matrix
    h, rep = (json.loads((X70 / "c0" / f).read_text()) for f in ("h.json", "report_c0.json"))
    got = x70["rect_stat"](h["h_pp"], cv=x70["cell_var"](1000))
    assert got["interval90_pp"] == pytest.approx(rep["interval90_pp"], abs=1e-12) and got["shape"] == [4, 8]
    # the registration's own lineages-as-a-sample interval of a 3 x 3 at 4,000 pairs (x72_rule.py's copy)
    m = [[BANKED[3 * i + j]["score_pp"] for j in range(3)] for i in range(3)]
    want = rule["decide_round"](m)["reported_lineages_as_a_sample"]
    got = x70["rect_stat"](m, cv=x70["cell_var"](4000))
    assert got["interval90_pp"] == want["interval90_pp"] and got["df"] == want["df"]
    assert (x70["cell_var"](4000), x70["cell_var"](1000)) == (0.3125, 1.25)


@pytest.mark.parametrize("what, const", [("X70's banked interval", "X70_ANCHOR"), ("CE-3d's banked interval",
                                                                                 "CE3D_DECIDING")])
def test_a_rule_file_that_does_not_reproduce_a_banked_interval_refuses_the_plan(ended, tmp_path, monkeypatch, what,
                                                                                const):
    src = repo_path(getattr(X, const))
    bad = tmp_path / "banked"
    bad.mkdir()
    v = json.loads((src / "verdict.json").read_text())
    v["interval90_pp"][0] -= 1e-6
    (bad / "verdict.json").write_text(json.dumps(v))
    (bad / "h.json").write_text((src / "h.json").read_text())
    monkeypatch.setattr(X, const, str(bad))
    with pytest.raises(F.FinalsError, match=f"does not reproduce {what}"):
        X.plan_end("c0", 2, root=ended["archive_root"], models_dir=ended["models_dir"])
    run = Runner(ended["ledger_root"], rows={})
    assert X.drive(2, tmp_path / "pin", tmp_path / "w", "cuda", end="c0", run=run, **ended) == X.EXIT_REFUSED
    assert [c[0][0] for c in run.calls] == ["git", "git"]              # refused before the registration and the play


# ------------------------------------------------------------------------------------------------------ the plan
def test_the_swap_end_cross_is_the_read_finals_against_the_banked_swap_finals(ended, tmp_path):
    _work, cells, shas = planned(ended, tmp_path, "swap", 2)
    models, root = ended["models_dir"], ended["archive_root"]
    rows = [str(root / "r2" / F.run_name(s) / "final_model.zip") for s in F.SEEDS]
    cols = [str(models / F.SWAP[s][0] / "final_model.zip") for s in F.SEEDS]
    assert cells == [[r, c] for r in rows for c in cols] and len(cells) == 9
    assert (shas["cross"], shas["end"], shas["round"], shas["pairs"], shas["batch_pairs"], shas["family"],
            shas["request"], shas["rehearsal"], shas["reuses"], shas["beside"]) == (
        "x72_end_swap_r2", "swap", 2, 4000, 500, "x72_end_swap_h2h", "x72_end_swap_r2", False, None, None)
    assert shas["cols"] == [F.SWAP[s][1] for s in F.SEEDS] and shas["col_seeds"] == list(F.SEEDS)
    assert shas["rows"] == [f.sha256 for f in F.load_round(2, root=root, models_dir=models)]


def test_the_c0_end_cross_is_the_read_finals_against_the_eight_c0_finals(ended, tmp_path):
    _work, cells, shas = planned(ended, tmp_path, "c0", 1)
    models, root = ended["models_dir"], ended["archive_root"]
    rows = [str(root / "r1" / F.run_name(s) / "final_model.zip") for s in F.SEEDS]
    cols = [str(models / f"rb_x35_c0_s{s}" / "final_model.zip") for s in range(3501, 3509)]
    assert cells == [[r, c] for r in rows for c in cols] and len(cells) == 24
    assert (shas["cross"], shas["end"], shas["round"], shas["pairs"], shas["batch_pairs"], shas["family"],
            shas["request"], shas["reuses"]) == ("x72_end_c0_r1", "c0", 1, 1000, 500, "x72_end_c0_h2h",
                                                 "x72_end_c0_r1", None)
    assert shas["cols"] == list(F.C0.values()) and shas["col_seeds"] == list(range(3501, 3509))
    assert shas["beside"]["delta_hat_pp"] == CE3D_V["delta_hat_pp"]
    assert X.plan_end("c0", 2, root=root, models_dir=models)[1]["request"] == "x72_end_c0_r2"


def test_at_read_round_one_the_swap_end_cross_is_round_ones_cross(ended, tmp_path):
    _work, cells, shas = planned(ended, tmp_path, "swap", 1)
    rcells, rshas = X.plan(1, root=ended["archive_root"], models_dir=ended["models_dir"])
    assert cells == rcells and set(rshas) == ROUND_DOC_KEYS           # the round's plan carries no end key
    assert shas == dict(rshas, cross="x72_end_swap_r1", end="swap", reuses="x72_round_1", beside=None)
    assert (shas["family"], shas["request"], shas["pairs"]) == ("x72_round_h2h", "x72_r1", 4000)


def test_the_end_plan_refuses_what_is_not_registered_or_does_not_verify(ended):
    kw = dict(root=ended["archive_root"], models_dir=ended["models_dir"])
    for end, k, why in (("x67", 1, "not one of"), ("swap", 0, "round 0"), ("c0", 5, "round 5"),
                        ("swap", 3, "r3: no finals.json"), ("c0", True, "round True")):
        with pytest.raises(F.FinalsError, match=why):
            X.plan_end(end, k, **kw)
    with pytest.raises(F.FinalsError, match="nothing to rehearse"):
        X.plan_end("swap", 1, rehearsal_pairs=4, **kw)
    for bad in (0, 1000, 1001):
        with pytest.raises(F.FinalsError, match=r"rehearsal-pairs .* is not in 1\.\.999"):
            X.plan_end("c0", 1, rehearsal_pairs=bad, **kw)
    (ended["models_dir"] / "rb_x35_c0_s3504" / "final_model.zip").write_bytes(b"moved")
    with pytest.raises(F.FinalsError, match="rb_x35_c0_s3504: final_model.zip has sha256"):
        X.plan_end("c0", 1, **kw)
    X.plan_end("swap", 2, **kw)                                       # the swap end cross reads no C0 file


# ---------------------------------------------------------------------------------------------------- the report
def test_the_c0_report_reproduces_ce3ds_banked_cross_from_its_banked_cells(ended, tmp_path):
    """CE-3d's 24 banked pentanomials, read as this cross's cells: its matrix, its +3.97 pp, its interval."""
    work, _cells, shas = planned(ended, tmp_path, "c0", 2)
    play(ended["ledger_root"], shas, cells=CE3D_CELLS)
    rec = X.read(work, ended["ledger_root"])
    rep = json.loads((work / "report.json").read_text())
    assert rep["matrix_pp"] == CE3D_H and rep["shape"] == [3, 8] and rep["cell_var_pp2"] == 1.25
    assert rep["delta_hat_pp"] == pytest.approx(3.967708333333327, abs=1e-12)
    assert rep["interval90_pp"] == pytest.approx(CE3D_V["interval90_pp"], abs=1e-12)
    assert (rep["decides"], rep["read"], rep["inconclusive"], rep["read_round"], rep["end"]) == ("nothing", True, [], 2,
                                                                                                "c0")
    assert rep["beside"]["interval90_pp"] == CE3D_V["interval90_pp"] and rep["pairs"] == 1000
    assert rep["played_on"] == [{"device": "cuda", "backend": "graph", "rows": 24}]
    assert rep["cols"] == [f"rb_x35_c0_s{s}" for s in range(3501, 3509)] and rep["col_sha256"] == list(F.C0.values())
    # a report is never a verdict: no class, no reading, and nothing the round's rule could be handed
    assert not {"outcome", "class", "reading", "h_pp", "next"} & set(rep)
    assert files(work) == {"cells_plan.json", "shas.json", "cells.json", "report.json", "read.json"}
    assert (rec["outcome"], rec["n_cells_read"]) == ("REPORTED", 24) and len(rec["cells"]) == 24
    said = []
    assert X._report(rec, said.append) == X.EXIT_OK
    assert "D = +3.97 pp, 90 % interval [+3.31, +4.63]" in said[1] and "no class" in said[1]
    assert "beside: D = +3.97 pp, [+3.31, +4.63]" in said[2] and "cpu / eager" in said[2] and len(said) == 3


def test_the_swap_report_is_the_registrations_lineages_as_a_sample_interval(ended, tmp_path, rule):  # noqa: F811
    work, _cells, shas = planned(ended, tmp_path, "swap", 2)
    play(ended["ledger_root"], shas)
    rec = X.read(work, ended["ledger_root"])
    rep = json.loads((work / "report.json").read_text())
    assert rep["matrix_pp"] == [[BANKED[3 * i + j]["score_pp"] for j in range(3)] for i in range(3)]
    want = rule["decide_round"](rep["matrix_pp"])["reported_lineages_as_a_sample"]
    assert rep["interval90_pp"] == want["interval90_pp"] and rep["cell_var_pp2"] == 0.3125 and rep["beside"] is None
    assert not (work / "h.json").exists() and "decide" not in rec and rec["outcome"] == "REPORTED"


@pytest.mark.parametrize("kw, why", [
    ({"skip": (23,)}, "missing cell"),
    ({"cells": {**CE3D_CELLS, 0: [187, 3, 553, 5, 251]}}, "999 completed pairs < 1000"),
    ({"cells": {**CE3D_CELLS, 0: [187, 3, 555, 5, 251]}}, "1001 completed pairs > the registered 1000"),
])
def test_a_cell_missing_or_not_exactly_the_registered_pairs_is_inconclusive_with_no_statistic(ended, tmp_path, kw, why):
    work, _cells, shas = planned(ended, tmp_path, "c0", 1)
    play(ended["ledger_root"], shas, **{"cells": CE3D_CELLS, **kw})
    rec = X.read(work, ended["ledger_root"])
    rep = json.loads((work / "report.json").read_text())
    assert rec["outcome"] == "INCONCLUSIVE" and rep["read"] is False and any(why in r for r in rep["inconclusive"])
    assert not {"matrix_pp", "delta_hat_pp", "interval90_pp", "rehearsal_stat"} & set(rep)
    assert X._report(rec, lambda _m: None) == X.EXIT_INCONCLUSIVE


def test_the_end_read_refuses_other_rows_and_a_plan_that_is_not_the_registered_cross(ended, tmp_path):
    work, _cells, shas = planned(ended, tmp_path, "c0", 1)
    register(ended["ledger_root"], shas).close()
    with pytest.raises(X.CrossRefused) as exc:
        X.read(work, ended["ledger_root"])
    assert exc.value.code == X.EXIT_ROWS and "nothing was played" in str(exc.value)
    play(ended["ledger_root"], shas, cells=CE3D_CELLS, edit=_other_venue)
    with pytest.raises(X.CrossRefused) as exc:
        X.read(work, ended["ledger_root"])
    assert exc.value.code == X.EXIT_ROWS and "more than one device/backend" in str(exc.value)
    assert files(work) == {"cells_plan.json", "shas.json"}
    for edit in ({"pairs": 4000}, {"family": "x72_round_h2h"}, {"request": "x72_end_c0_r2"}):
        (work / "shas.json").write_text(json.dumps(dict(shas, **edit)))
        with pytest.raises(X.CrossRefused) as exc:
            X.read(work, ended["ledger_root"])
        assert exc.value.code == X.EXIT_REFUSED and "x72_end_c0_r1" in str(exc.value)


# ---------------------------------------------------------------------------------------------------- the driver
def test_drive_plays_the_c0_end_cross_at_the_pin_in_its_own_family_and_reports_once(ended, tmp_path):
    work, pin = tmp_path / "work", tmp_path / "pin"
    run = Runner(ended["ledger_root"], rows={"cells": CE3D_CELLS})
    assert X.drive(2, pin, work, "cuda", end="c0", run=run, **ended) == X.EXIT_OK
    argvs = [c[0] for c in run.calls]
    assert [a[0] for a in argvs] == ["git", "git", "PY", str(repo_path("scripts", "ops", "gpu_lock.sh"))]
    shas = json.loads((work / "shas.json").read_text())
    reg = argvs[2]
    assert reg[reg.index("--id") + 1] == "x72_end_c0_h2h" and reg[reg.index("--commit") + 1] == X.PLAY_PIN
    assert reg[reg.index("--rule") + 1] == X.family_rule(shas) != X.FAMILY_RULE
    assert argvs[3] == X.venue_argv(X.play_argv("PY", work, shas, "cuda", ended["ledger_root"]), "cuda", "")
    play_ = argvs[3]
    assert [play_[play_.index(f) + 1] for f in ("--pairs", "--batch-pairs", "--seed", "--family", "--request",
                                                "--label")] == ["1000", "500", "0", "x72_end_c0_h2h",
                                                                "x72_end_c0_r2", "x72_swap_plateau"]
    assert len(json.loads((work / "cells_plan.json").read_text())) == 24
    assert all(kw["cwd"] == str(pin) and kw["env"]["PYTHONPATH"] == str(pin / "src") for _a, kw in run.calls[2:])
    assert json.loads((work / "report.json").read_text())["delta_hat_pp"] == pytest.approx(CE3D_V["delta_hat_pp"])
    log = (work / "run.log").read_text()
    assert "x72_end_c0_r2 (REPORTED, decides nothing): 24 cells x 1000 pairs on cuda" in log and "beside: " in log
    again = Runner(ended["ledger_root"], rows={})
    assert X.drive(2, pin, work, "cuda", end="c0", run=again, **ended) == X.EXIT_HOLD and again.calls == []


def test_drive_at_read_round_one_reads_round_ones_cross_and_never_plays(ended, tmp_path):
    work, pin = tmp_path / "work", tmp_path / "pin"
    run = Runner(ended["ledger_root"], head="0" * 40, rows={})        # even the pin is not asked for: nothing plays
    assert X.drive(1, pin, work, "cuda", end="swap", run=run, **ended) == X.EXIT_ROWS      # round 1 was not played
    assert run.calls == [] and not (work / "read.json").exists()
    log = (work / "run.log").read_text()
    assert "family 'x72_round_h2h' is not registered" in log and "this never replays" in log
    register(ended["ledger_root"], json.loads((work / "shas.json").read_text())).close()
    assert X.drive(1, pin, work, "cuda", end="swap", run=run, **ended) == X.EXIT_ROWS      # registered, no row yet
    assert "request x72_r1 holds no row" in (work / "run.log").read_text() and run.calls == []
    # round 1's cross, played by the round's own driver in its own work dir
    assert X.drive(1, pin, tmp_path / "r1", "cuda", run=Runner(ended["ledger_root"], rows={}), **ended) == X.EXIT_OK
    assert X.drive(1, pin, work, "cuda", end="swap", run=run, **ended) == X.EXIT_OK and run.calls == []
    rep = json.loads((work / "report.json").read_text())
    assert rep["matrix_pp"] == json.loads((tmp_path / "r1" / "h.json").read_text())["h_pp"]
    assert (rep["reuses"], rep["family"], rep["request"], rep["cross"]) == ("x72_round_1", "x72_round_h2h", "x72_r1",
                                                                            "x72_end_swap_r1")
    assert "IS round 1's cross: nothing is registered or played" in (work / "run.log").read_text()
    assert not (work / "h.json").exists()
    assert X.drive(1, pin, work, "cuda", end="swap", run=run, **ended) == X.EXIT_HOLD and run.calls == []


@pytest.mark.parametrize("end, runner, code, n_calls", [
    ("swap", dict(head="0" * 40), X.EXIT_HOLD, 1),
    ("c0", dict(register_rc=2), X.EXIT_REFUSED, 3),
    ("swap", dict(play_rc=1, rows={}), X.EXIT_PLAY, 4),
    ("swap", dict(rows={"skip": (0,)}), X.EXIT_INCONCLUSIVE, 4),
    ("c0", dict(rows={"cells": CE3D_CELLS, "edit": _other_venue}), X.EXIT_ROWS, 4),
])
def test_the_end_drives_exit_codes_are_the_round_drives(ended, tmp_path, end, runner, code, n_calls):
    work = tmp_path / "work"
    run = Runner(ended["ledger_root"], **runner)
    assert X.drive(2, tmp_path / "pin", work, "cuda", end=end, run=run, **ended) == code
    assert len(run.calls) == n_calls and (work / "read.json").exists() == (code == X.EXIT_INCONCLUSIVE)
    assert not (work / "h.json").exists()
    if code == X.EXIT_PLAY:                                           # an interrupted play is RESUMED
        ok = Runner(ended["ledger_root"], rows={})
        assert X.drive(2, tmp_path / "pin", work, "cuda", end=end, run=ok, **ended) == X.EXIT_OK
    other = Runner(ended["ledger_root"], rows={})
    assert X.drive(2, tmp_path / "pin", tmp_path / "w2", "cuda", end="x67", run=other, **ended) == X.EXIT_REFUSED
    assert len(other.calls) == 2 and not (tmp_path / "w2" / "shas.json").exists()


def test_an_end_rehearsal_has_its_own_names_and_reports_no_statistic_of_record(ended, tmp_path):
    work = tmp_path / "work"
    tiny = {n: [0, 0, 3, 0, 1] for n in range(24)}
    run = Runner(ended["ledger_root"], rows={"cells": tiny})
    assert X.drive(2, tmp_path / "pin", work, "cuda", end="c0", run=run, rehearsal_pairs=4, **ended) == X.EXIT_OK
    shas, rep = json.loads((work / "shas.json").read_text()), json.loads((work / "report.json").read_text())
    assert (shas["family"], shas["request"], shas["pairs"], shas["batch_pairs"]) == (
        "x72_rehearsal_h2h", "x72_rehearsal_end_c0_r2", 4, 4)
    reg, played = run.calls[2][0], run.calls[3][0]
    assert reg[reg.index("--rule") + 1] == X.REHEARSAL_RULE and played[played.index("--label") + 1] == "x72_rehearsal"
    assert rep["read"] is False and rep["rehearsal"] is True and rep["rehearsal_matrix_pp"] == [[62.5] * 8] * 3
    assert rep["rehearsal_stat"]["delta_hat_pp"] == 12.5 and rep["rehearsal_stat"]["cell_var_pp2"] == 312.5
    assert not {"matrix_pp", "delta_hat_pp", "interval90_pp"} & set(rep)
    assert json.loads((work / "read.json").read_text())["outcome"] == "REHEARSAL"


def test_the_cli_plans_each_end_cross_and_says_which_one_is_round_ones(ended, tmp_path, capsys):
    args = ["--archive-root", str(ended["archive_root"]), "--models-dir", str(ended["models_dir"]), "--ledger-root",
            str(ended["ledger_root"])]
    assert X.main(["plan", "--end", "c0", "--round", "2", "--work-dir", str(tmp_path / "a"), *args]) == 0
    assert capsys.readouterr().out == f"24 cells planned in {tmp_path / 'a'} x 1000 pairs: x72_end_c0_r2\n"
    assert X.main(["plan", "--end", "swap", "--round", "1", "--work-dir", str(tmp_path / "b"), *args]) == 0
    assert "x72_end_swap_r1 IS x72_round_1 (request x72_r1); nothing to play" in capsys.readouterr().out
    assert X.main(["plan", "--end", "swap", "--round", "3", "--work-dir", str(tmp_path / "c"), *args]) == 3
    assert X.main(["plan", "--round", "1", "--work-dir", str(tmp_path / "d"), *args]) == 0
    assert capsys.readouterr().out == f"9 cells planned in {tmp_path / 'd'}\n"
    assert X.main(["read", "--work-dir", str(tmp_path / "b"), "--ledger-root", str(ended["ledger_root"])]) == X.EXIT_ROWS
    assert "this never replays" in capsys.readouterr().err
