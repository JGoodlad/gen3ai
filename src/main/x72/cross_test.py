"""X72's round cross (``main.x72.cross``): the restated protocol against the frozen registration and X70's driver,
the plan over archive copies, the read against a real ledger (X70's banked cells as the arithmetic's reference),
every INCONCLUSIVE and every refusal, the frozen rule taking the read's ``h.json``, and the driver with a recorded
runner. No game is played and nothing touches ``models/``."""
from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.training import eval_ledger as L
from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import testkit as K
from main.x72 import cross as X
from main.x72 import finals as F
from main.x72.finals_test import DIR, train_round, world  # noqa: F401  (world is a fixture)
from utils.paths import repo_path

X70 = repo_path("designs/research_state/measurements/x70_ent0_control_2026-11-13")
RULE = DIR / "x72_rule.py"
#: nine of X70's banked deciding cells (their pentanomials and the score X70's reader computed from each)
BANKED = json.loads((X70 / "record" / "anchor" / "cells.json").read_text())[:9]


@pytest.fixture()
def rule():
    return runpy.run_path(str(RULE))


@pytest.fixture()
def planned(world, tmp_path):  # noqa: F811
    """Round 1 archived and planned: ``(work dir, ledger root, shas, models, archive root)``."""
    models, root = world
    train_round(models, 1)
    F.archive(1, root=root, models_dir=models)
    work = tmp_path / "work"
    cells, shas = X.plan(1, root=root, models_dir=models)
    X.write_plan(work, cells, shas)
    return work, tmp_path / "ledger", shas, models, root


def register(ledger: Path, shas) -> L.LedgerWriter:
    w = L.LedgerWriter(ledger, producer="h2h")
    w.register_family(shas["family"], decision_kind="ab_verdict", protocol=X.PROTOCOL, commit=X.PLAY_PIN,
                      rule=X.family_rule(shas))
    return w


def play(ledger: Path, shas, *, cells=None, skip=(), edit=None, voided=None) -> None:
    """Record the cross's rows: one row a cell, cell ``n`` (rows outer) with pentanomial ``cells[n]`` (default: the
    ``n``-th banked X70 cell's). ``skip`` leaves cells out; ``voided[n]`` pairs of cell ``n`` are voided (two aborted
    games each); ``edit(n, row)`` changes a row before it is appended."""
    w = register(ledger, shas)
    req = w.open_request(shas["request"], kind="adhoc", purpose="ab", family=shas["family"])
    n = 0
    for r in shas["rows"]:
        for c in shas["cols"]:
            pc = (cells or {}).get(n) or BANKED[n]["pairs"]
            if n not in skip:
                v = (voided or {}).get(n, 0)
                row = K.make_row(w.next_row_id(), request=req, batch=0, pc=pc, p=r, o=c, purpose="ab", voided=v,
                                 aborted=2 * v, schedule_key=f"h2h:{r}:{c}", ts=S.utc_now())
                row["commit"] = X.PLAY_PIN
                row["compute"].update(device="cuda", backend="graph")
                if edit is not None:
                    edit(n, row)
                assert L.validate_row(row) == []
                claim = w.claim(req["request_id"], batch=0, player=r, opponent=c,
                                regime_id=row["regime"]["regime_id"], expected_wall_s=10)
                w.append_row(row, claim)
            n += 1
    w.close()


# -------------------------------------------------------------------------------------- the restated protocol
def test_the_protocol_is_the_registrations_and_x70s(rule):
    text = " ".join((DIR / "PREREG.md").read_text().split())
    assert f"**Play pin `{X.PLAY_PIN}`**" in text
    assert "**9 cells × 4,000 mirrored pairs = 72,000 games**" in text and rule["PAIRS"] == X.PAIRS == 4000
    assert ("both sides greedy, batches of 500 pairs, schedule seed 0, the players' eval team builder (no panel), "
            "`--oracle-reveal-mode off`, purpose `ab`, label `x72_swap_plateau`, family `x72_round_h2h`, request "
            "`x72_r<k>`") in text
    assert (X.BATCH_PAIRS, X.SCHEDULE_SEED, X.LABEL, X.FAMILY, X.request(3)) == (500, 0, "x72_swap_plateau",
                                                                                 "x72_round_h2h", "x72_r3")
    assert "`x72_rule.py decide-round <k> h.json`" in text and RULE == repo_path(X.RULE_FILE)
    # X70's driver, flag for flag: the play line and the two venues' wrappers
    drv = (X70 / "cross" / "drive_cross.sh").read_text()
    assert ('PLAY=("$PY" -m main.h2h play-many --cells "$W/cells_plan.json" --pairs "$PAIRS" --batch-pairs 500 '
            '--seed 0 --device "$DEV"\n      --purpose ab --family "$FAMILY" --request "$REQUEST" '
            '--label x70_ent0_control --oracle-reveal-mode off)') in drv
    assert '"$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 "${PLAY[@]}"' in drv
    assert 'nice -n 19 taskset -c "$CPUS" "$OPS/mem_cap.sh" 24 "${PLAY[@]}" --threads 3 --torch-threads 3' in drv
    assert X.PROTOCOL == K.PROTO and X.VENUES == {"cuda": ("32", ()), "cpu": ("24", ("--threads", "3",
                                                                                   "--torch-threads", "3"))}


def test_the_play_argv_is_x70s_with_this_crosss_names(planned):
    work, ledger, shas, _models, _root = planned
    play = X.play_argv("PY", work, shas, "cuda", ledger)
    assert play == ["PY", "-m", "main.h2h", "play-many", "--cells", str(work / "cells_plan.json"), "--pairs", "4000",
                    "--batch-pairs", "500", "--seed", "0", "--device", "cuda", "--purpose", "ab", "--family",
                    "x72_round_h2h", "--request", "x72_r1", "--label", "x72_swap_plateau", "--oracle-reveal-mode",
                    "off", "--out", str(ledger)]
    ops = repo_path("scripts", "ops")
    assert X.venue_argv(play, "cuda", "") == [str(ops / "gpu_lock.sh"), str(ops / "mem_cap.sh"), "32", *play]
    assert X.venue_argv(play, "cpu", "5-7") == ["nice", "-n", "19", "taskset", "-c", "5-7", str(ops / "mem_cap.sh"),
                                                "24", *play, "--threads", "3", "--torch-threads", "3"]
    assert (ops / "gpu_lock.sh").is_file() and (ops / "mem_cap.sh").is_file()


# ------------------------------------------------------------------------------------------------------ the plan
def test_the_plan_is_round_k_copies_against_round_k_minus_one_rows_outer(planned):
    work, _ledger, shas, models, root = planned
    cells = json.loads((work / "cells_plan.json").read_text())
    rows = [str(root / "r1" / F.run_name(s) / "final_model.zip") for s in F.SEEDS]
    cols = [str(models / F.SWAP[s][0] / "final_model.zip") for s in F.SEEDS]
    assert cells == [[r, c] for r in rows for c in cols]
    assert shas["rows"] == [F.load_final(1, s, root=root, models_dir=models).sha256 for s in F.SEEDS]
    assert shas["cols"] == [F.SWAP[s][1] for s in F.SEEDS]
    assert (shas["round"], shas["pairs"], shas["batch_pairs"], shas["family"], shas["request"], shas["rehearsal"]) == (
        1, 4000, 500, "x72_round_h2h", "x72_r1", False)
    assert shas["row_seeds"] == shas["col_seeds"] == list(F.SEEDS)
    # round 2: the columns are round 1's ARCHIVE copies, although the run dirs now hold round 2's finals
    train_round(models, 2)
    F.archive(2, root=root, models_dir=models)
    cells2, shas2 = X.plan(2, root=root, models_dir=models)
    assert [c[1] for c in cells2[:3]] == rows and shas2["cols"] == shas["rows"] and shas2["request"] == "x72_r2"
    assert all("/r2/" in c[0] for c in cells2)


def test_the_plan_refuses_an_unarchived_round_a_moved_copy_and_round_zero(planned):
    _work, _ledger, _shas, models, root = planned
    with pytest.raises(F.FinalsError, match="r2: no finals.json"):
        X.plan(2, root=root, models_dir=models)
    with pytest.raises(F.FinalsError):
        X.plan(0, root=root, models_dir=models)
    (root / "r1" / F.run_name(7343) / "final_model.zip").write_bytes(b"moved")
    with pytest.raises(F.FinalsError, match="no longer match"):
        X.plan(1, root=root, models_dir=models)


def test_a_work_dir_belongs_to_one_cross(planned):
    work, _ledger, shas, models, root = planned
    X.write_plan(work, json.loads((work / "cells_plan.json").read_text()), shas)        # the same plan: fine
    train_round(models, 2)
    F.archive(2, root=root, models_dir=models)
    with pytest.raises(X.CrossRefused) as exc:
        X.write_plan(work, *X.plan(2, root=root, models_dir=models))
    assert exc.value.code == X.EXIT_HOLD and json.loads((work / "shas.json").read_text()) == shas


# ------------------------------------------------------------------------------------------------------ the read
def test_the_read_scores_each_cell_as_x70s_reader_did_and_the_frozen_rule_takes_h_json(planned, rule):
    work, ledger, shas, _models, _root = planned
    play(ledger, shas)
    rec = X.read(work, ledger)
    h = json.loads((work / "h.json").read_text())
    assert h["h_pp"] == [[BANKED[3 * i + j]["score_pp"] for j in range(3)] for i in range(3)]
    assert h["h_pp"][0][0] == 50.20625 == 100.0 * (3996 + 41 / 2) / 8000
    assert (rec["outcome"], rec["inconclusive"], rec["n_cells_read"]) == ("READ", [], 9)
    assert h["played_on"] == [{"device": "cuda", "backend": "graph", "rows": 9}]
    assert (h["rows"], h["cols"]) == ([F.run_name(s) for s in F.SEEDS], [F.SWAP[s][0] for s in F.SEEDS])
    assert [c["n_pairs"] for c in json.loads((work / "cells.json").read_text())] == [4000] * 9
    assert json.loads((work / "read.json").read_text())["decide"] == f"python {X.RULE_FILE} decide-round 1 {work / 'h.json'}"
    assert X._report(rec, lambda _m: None) == X.EXIT_OK
    v = rule["decide_round"](h["h_pp"])
    assert v["outcome"] != "INCONCLUSIVE" and v["cells"] == 9
    # the registered CLI, on the file the reader wrote
    out = subprocess.run([sys.executable, str(RULE), "decide-round", "1", str(work / "h.json")], capture_output=True,
                         text=True, check=True)
    got = json.loads(out.stdout)
    assert got["outcome"] == v["outcome"] and got["delta_hat_pp"] == v["delta_hat_pp"] and "next" in got


@pytest.mark.parametrize("kw, why", [
    ({"skip": (4,)}, "missing cell"),
    ({"cells": {4: [865, 21, 2215, 12, 886]}}, "3999 completed pairs < 4000"),
    ({"cells": {4: [865, 21, 2217, 12, 886]}}, "4001 completed pairs > the registered 4000"),
    ({"skip": tuple(range(9))[1:]}, "missing cell"),
])
def test_a_cell_missing_or_not_exactly_4000_pairs_is_inconclusive_and_the_rule_holds(planned, rule, kw, why):
    work, ledger, shas, _models, _root = planned
    play(ledger, shas, **kw)
    rec = X.read(work, ledger)
    h = json.loads((work / "h.json").read_text())
    assert rec["outcome"] == "INCONCLUSIVE" and h["h_pp"] == [] and any(why in r for r in h["inconclusive"]), h
    assert X._report(rec, lambda _m: None) == X.EXIT_INCONCLUSIVE
    assert rule["decide_round"](h["h_pp"])["outcome"] == "INCONCLUSIVE"
    held = json.loads(subprocess.run([sys.executable, str(RULE), "decide-round", "1", str(work / "h.json")],
                                     capture_output=True, text=True, check=True).stdout)
    assert (held["outcome"], held["next"]["action"]) == ("INCONCLUSIVE", "HOLD")


def test_a_cell_that_is_not_ok_is_inconclusive(planned):
    work, ledger, shas, _models, _root = planned
    play(ledger, shas, voided={7: 1334})        # 2,668 aborted of 10,668 attempted games > 25 %, 4,000 pairs complete
    rec = X.read(work, ledger)
    h = json.loads((work / "h.json").read_text())
    assert rec["outcome"] == "INCONCLUSIVE" and h["h_pp"] == [] and any("> 25 %" in r for r in h["inconclusive"])
    assert [c["verdict"] for c in rec["cells"]].count("OK") == 8 and [c["n_pairs"] for c in rec["cells"]] == [4000] * 9


def _other_venue(n, row):
    if n == 8:
        row["compute"].update(device="cpu", backend="eager")


def _other_commit(n, row):
    if n == 2:
        row["commit"] = F.TRAIN_PIN


def _other_seed(n, row):
    if n == 5:
        row["seed"]["schedule_seed"] = 1


@pytest.mark.parametrize("edit, why", [(_other_venue, "more than one device/backend: cpu/eager x1 rows, cuda/graph x8"),
                                       (_other_commit, "not the play pin 333a73dd"),
                                       (_other_seed, "schedule seed ['1'], not 0")])
def test_rows_that_are_not_one_registered_cross_are_refused_and_nothing_is_written(planned, edit, why):
    work, ledger, shas, _models, _root = planned
    play(ledger, shas, edit=edit)
    with pytest.raises(X.CrossRefused) as exc:
        X.read(work, ledger)
    assert exc.value.code == X.EXIT_ROWS and why in str(exc.value)
    assert not any((work / f).exists() for f in ("h.json", "cells.json", "read.json"))
    with pytest.raises(X.CrossRefused):
        X.venue_check(shas, "cuda", ledger)


def test_a_request_with_no_row_is_not_a_look(planned):
    work, ledger, shas, _models, _root = planned
    register(ledger, shas).close()
    with pytest.raises(X.CrossRefused) as exc:
        X.read(work, ledger)
    assert exc.value.code == X.EXIT_ROWS and "nothing was played" in str(exc.value)
    assert not (work / "read.json").exists()
    X.venue_check(shas, "cuda", ledger)                    # no row yet: either venue may start it
    X.venue_check(shas, "cpu", ledger)


def test_a_cross_is_finished_on_the_device_it_started_on(planned):
    work, ledger, shas, _models, _root = planned
    play(ledger, shas, skip=(8,), edit=lambda n, row: row["compute"].update(device="cuda:0"))
    X.venue_check(shas, "cuda", ledger)
    with pytest.raises(X.CrossRefused) as exc:
        X.venue_check(shas, "cpu", ledger)
    assert exc.value.code == X.EXIT_ROWS and "never on 'cpu'" in str(exc.value)


def test_the_read_refuses_a_plan_that_is_not_the_registered_cross(planned):
    work, ledger, shas, _models, _root = planned
    (work / "shas.json").write_text(json.dumps(dict(shas, pairs=1000)))
    with pytest.raises(X.CrossRefused) as exc:
        X.read(work, ledger)
    assert exc.value.code == X.EXIT_REFUSED


# ---------------------------------------------------------------------------------------------------- the driver
class Runner:
    """``subprocess.run`` recorded: git answers as told, ``family-register`` registers in the test's ledger, the play
    records ``self.rows`` (keyword arguments of :func:`play`) and exits ``self.play_rc``."""

    def __init__(self, ledger: Path, shas=None, *, head=X.PLAY_PIN, dirty="", register_rc=0, play_rc=0, rows=None):
        self.ledger, self.shas, self.head, self.dirty = ledger, shas, head, dirty
        self.register_rc, self.play_rc, self.rows, self.calls = register_rc, play_rc, rows, []

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw))
        if argv[0] == "git":
            return SimpleNamespace(returncode=0, stdout=self.head + "\n" if "rev-parse" in argv else self.dirty)
        shas = self.shas or json.loads((Path(argv[argv.index("--cells") + 1]).parent / "shas.json").read_text()) \
            if "play-many" in argv else self.shas
        if "family-register" in argv:
            if self.register_rc == 0:
                w = L.LedgerWriter(self.ledger, producer="ledger_cli")
                w.register_family(argv[argv.index("--id") + 1], decision_kind="ab_verdict",
                                  rule=argv[argv.index("--rule") + 1], protocol=argv[argv.index("--protocol") + 1],
                                  commit=argv[argv.index("--commit") + 1])
                w.close()
            return SimpleNamespace(returncode=self.register_rc)
        if self.play_rc == 0 and self.rows is not None:
            play(self.ledger, shas, **self.rows)
        return SimpleNamespace(returncode=self.play_rc)


@pytest.fixture()
def archived(world, tmp_path):  # noqa: F811
    models, root = world
    train_round(models, 1)
    F.archive(1, root=root, models_dir=models)
    return dict(ledger_root=tmp_path / "ledger", archive_root=root, models_dir=models, py="PY", cpus="5-7")


def test_drive_plans_registers_checks_the_venue_plays_at_the_pin_and_reads_once(archived, tmp_path):
    work, pin = tmp_path / "work", tmp_path / "pin"
    run = Runner(archived["ledger_root"], rows={})
    assert X.drive(1, pin, work, "cuda", run=run, **archived) == X.EXIT_OK
    argvs = [c[0] for c in run.calls]
    assert [a[:4] for a in argvs[:2]] == [["git", "-C", str(pin), "rev-parse"], ["git", "-C", str(pin), "status"]]
    assert argvs[1][4:] == ["--porcelain", "--untracked-files=no", "--", "src", "data"]
    reg, kw = run.calls[2]
    assert reg == ["PY", "-m", "main.eval_ledger", "family-register", "--root", str(archived["ledger_root"]), "--id",
                   "x72_round_h2h", "--decision-kind", "ab_verdict", "--rule", X.FAMILY_RULE, "--protocol",
                   "gen3_eval_protocol_v1_h2h", "--commit", X.PLAY_PIN]
    played, pkw = run.calls[3]
    shas = json.loads((work / "shas.json").read_text())
    assert played == X.venue_argv(X.play_argv("PY", work, shas, "cuda", archived["ledger_root"]), "cuda", "")
    for k in (kw, pkw):                                  # the pin's code plays and registers, never this checkout's
        assert k["cwd"] == str(pin) and k["env"]["PYTHONPATH"] == str(pin / "src")
        assert k["env"]["GEN3AI_SHARED_COMPILE_CACHE"] == "0"
    assert len(run.calls) == 4 and json.loads((work / "h.json").read_text())["h_pp"][0][0] == 50.20625
    assert "play: " in (work / "run.log").read_text()
    # ONE look: a second drive plays nothing and holds
    again = Runner(archived["ledger_root"], rows={})
    assert X.drive(1, pin, work, "cuda", run=again, **archived) == X.EXIT_HOLD and again.calls == []
    assert "ONE look is spent" in (work / "run.log").read_text()


def test_drive_on_cpu_hides_the_card_and_runs_niced_on_the_agent_cores(archived, tmp_path):
    work = tmp_path / "work"
    run = Runner(archived["ledger_root"], rows={"edit": lambda n, row: row["compute"].update(device="cpu",
                                                                                           backend="eager")})
    assert X.drive(1, tmp_path / "pin", work, "cpu", run=run, **archived) == X.EXIT_OK
    played, kw = run.calls[3]
    assert played[:6] == ["nice", "-n", "19", "taskset", "-c", "5-7"] and played[-4:] == ["--threads", "3",
                                                                                         "--torch-threads", "3"]
    assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "" and played[played.index("--device") + 1] == "cpu"


@pytest.mark.parametrize("runner, code, n_calls", [
    (dict(head="0" * 40), X.EXIT_HOLD, 1),
    (dict(dirty=" M src/main/h2h/play.py\n"), X.EXIT_HOLD, 2),
    (dict(register_rc=2), X.EXIT_REFUSED, 3),
    (dict(play_rc=1, rows={}), X.EXIT_PLAY, 4),
    (dict(rows={"skip": (0,)}), X.EXIT_INCONCLUSIVE, 4),
    (dict(rows={"edit": _other_venue}), X.EXIT_ROWS, 4),
])
def test_drives_exit_codes(archived, tmp_path, runner, code, n_calls):
    work = tmp_path / "work"
    run = Runner(archived["ledger_root"], **runner)
    assert X.drive(1, tmp_path / "pin", work, "cuda", run=run, **archived) == code
    assert len(run.calls) == n_calls
    assert (work / "read.json").exists() == (code == X.EXIT_INCONCLUSIVE)
    if code == X.EXIT_PLAY:                               # an interrupted play is RESUMED by the next drive
        ok = Runner(archived["ledger_root"], rows={})
        assert X.drive(1, tmp_path / "pin", work, "cuda", run=ok, **archived) == X.EXIT_OK


def test_drive_refuses_before_any_command_when_a_final_does_not_verify(archived, tmp_path):
    run = Runner(archived["ledger_root"], rows={})
    assert X.drive(2, tmp_path / "pin", tmp_path / "work", "cuda", run=run, **archived) == X.EXIT_REFUSED
    assert [c[0][0] for c in run.calls] == ["git", "git"] and not (tmp_path / "work" / "shas.json").exists()
    assert X.drive(1, tmp_path / "pin", tmp_path / "work", "tpu", run=run, **archived) == X.EXIT_USAGE


def test_drive_refuses_to_start_on_the_other_venue(archived, tmp_path):
    work = tmp_path / "work"
    half = Runner(archived["ledger_root"], rows={"skip": (8,)}, play_rc=0)
    cells, shas = X.plan(1, root=archived["archive_root"], models_dir=archived["models_dir"])
    play(archived["ledger_root"], shas, skip=(8,))               # eight cells already on the card
    run = Runner(archived["ledger_root"], rows={})
    assert X.drive(1, tmp_path / "pin", work, "cpu", run=run, **archived) == X.EXIT_ROWS
    assert len(run.calls) == 3 and not (work / "read.json").exists() and half.calls == []


# ------------------------------------------------------------------------------------------------- the rehearsal
def test_a_rehearsal_has_its_own_names_and_never_yields_a_matrix_the_rule_can_read(archived, tmp_path, rule):
    work = tmp_path / "work"
    tiny = {n: [0, 0, 3, 0, 1] for n in range(9)}
    run = Runner(archived["ledger_root"], rows={"cells": tiny})
    assert X.drive(1, tmp_path / "pin", work, "cuda", run=run, rehearsal_pairs=4, **archived) == X.EXIT_OK
    shas, h = json.loads((work / "shas.json").read_text()), json.loads((work / "h.json").read_text())
    assert (shas["family"], shas["request"], shas["pairs"], shas["batch_pairs"]) == ("x72_rehearsal_h2h",
                                                                                   "x72_rehearsal_r1", 4, 4)
    played = run.calls[3][0]
    assert played[played.index("--label") + 1] == "x72_rehearsal" and played[played.index("--pairs") + 1] == "4"
    assert run.calls[2][0][run.calls[2][0].index("--rule") + 1] == X.REHEARSAL_RULE
    assert h["h_pp"] == [] and h["rehearsal_h_pp"] == [[62.5] * 3] * 3 and h["rehearsal"] is True
    assert json.loads((work / "read.json").read_text())["outcome"] == "REHEARSAL"
    assert rule["decide_round"](h["h_pp"])["outcome"] == "INCONCLUSIVE"
    for bad in (0, 4000, 4001):
        with pytest.raises(F.FinalsError, match="rehearsal-pairs"):
            X.plan(1, root=archived["archive_root"], models_dir=archived["models_dir"], rehearsal_pairs=bad)


def test_the_cli_plans_reads_and_reports_refusals(planned, world, capsys):  # noqa: F811
    work, ledger, shas, models, root = planned
    args = ["--work-dir", str(work), "--ledger-root", str(ledger)]
    assert X.main(["plan", "--round", "1", "--archive-root", str(root), "--models-dir", str(models), *args]) == 0
    assert X.main(["plan", "--round", "2", "--archive-root", str(root), "--models-dir", str(models), *args]) == 3
    register(ledger, shas).close()
    assert X.main(["venue-check", "--device", "cuda", *args]) == 0
    assert X.main(["read", *args]) == X.EXIT_ROWS
    play(ledger, shas, skip=(3,))
    assert X.main(["read", *args]) == X.EXIT_INCONCLUSIVE
    assert X.main(["read", "--work-dir", str(work / "nowhere"), "--ledger-root", str(ledger)]) == X.EXIT_USAGE
    assert "REFUSED" in capsys.readouterr().err


def test_the_rows_are_read_under_the_terms_the_cells_are_read_under():
    """The venue / pin / seed check must see exactly the rows the cells pool: one declaration's terms, two names."""
    import dataclasses

    import main.h2h.cross as HX

    assert dataclasses.replace(X.ROWS_READ, name=HX.FAMILY_READ_OFF.name) == HX.FAMILY_READ_OFF
    assert X.ROWS_READ.name == "main.x72.cross.rows" != HX.FAMILY_READ_OFF.name
    assert X.ROWS_READ.regime.protocol == X.PROTOCOL
