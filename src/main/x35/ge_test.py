"""X35's G-E READERS (``main.x35.ge``; F27 closed for the readers): the reader argv, its trigger (``kit stage-check``,
``launch_x35.sh`` pre-check 9), the side's pin over reader runs, the post-spawn pool exemption, and the ``r2`` read
``readout.decide`` / ``decide-d1`` need. Synthetic files only; every case fails on revert of the rule it names."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import pytest

from agents.training import best_response_gap as E
from main.x35 import batch as B
from main.x35 import ge as GE
from main.x35 import kit as K
from main.x35 import readout as R
from utils.paths import repo_path

PSTAR_STEP = 280_000_170
FINAL_STEP = 300_000_170
LEVERS = {"--stable-opponents", "--stable-opponent-selfplay-share", "--pfsp-scale", "--n-sentinels", "--team-mix",
          "--team-mix-sides", "--grow", "--self-play", "--exploiter-keep-bots"}


def flags(tokens: List[str]) -> Dict[str, Union[str, bool]]:
    out: Dict[str, Union[str, bool]] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        assert t.startswith("--"), tokens
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[t], i = tokens[i + 1], i + 2
        else:
            out[t], i = True, i + 1
    return out


def _call() -> Dict[str, Any]:
    return {"side": "production", "archive": [],
            "pstar": {"label": "x26_pstar", "path": "models/rb_x26_s2601/checkpoints/c.zip", "step": PSTAR_STEP,
                      "sha256": "shaP"}}


def _target(arm: str, step: int = FINAL_STEP) -> Dict[str, Any]:
    run = GE.target_run(arm, 3501)
    return {"run": run, "path": f"/m/{run}/final_model.zip", "step": step, "sha256": "s", "standin": False}


# ------------------------------------------------------------------------------------------------------- the argv
@pytest.mark.parametrize("arm", sorted(GE.READER_ARMS))
def test_a_reader_best_responds_to_its_targets_final_on_the_offense_archetype_at_a_pinned_dose(arm):
    t = _target(arm)
    f = flags(K.build_argv(arm, 3501, _call(), target=t))
    assert f["--model"] == f["--exploiter"] == t["path"]                       # a fork of the target, vs the target
    assert f["--no-self-play"] is True                                          # F16
    assert f["--behaviour-check"] == "warn"                                     # F14
    assert f["--steps"] == str(FINAL_STEP + 5_000_000)                          # §5.6.4: 5M
    assert f["--fork-lr"] == "5.6e-05" and f["--fork-lr-freeze"] is True        # the dose, pinned by construction
    assert f["--seed"] == "3531"                                                # §5.6.1: readers 3531+
    assert f["--run-name"] == "rb_x35_ge_" + GE.target_run(arm, 3501)[len("rb_x35_"):]
    teams = str(f["--trainee-teams"]).split(",")
    offense = E.load_teamsets()["offense"]
    assert teams == offense["paths"] and len(teams) == 5
    arch, _ = E.archetype_of(teams, E.load_teamsets())
    assert arch == "offense"                                                    # the meter will assign it (F17)
    assert not LEVERS & set(f)                                                  # no arm lever reaches a reader


def test_the_on_and_off_readers_differ_only_in_their_target_and_run_name():
    on = flags(K.build_argv("ge_a1", 3501, _call(), target=_target("ge_a1")))
    off = flags(K.build_argv("ge_c0", 3501, _call(), target=_target("ge_c0")))
    assert {k for k in on if on[k] != off.get(k)} == {"--model", "--exploiter", "--run-name"}
    assert set(on) == set(off)


def test_a_reader_needs_its_target_and_only_seed_3501_is_registered():
    with pytest.raises(K.KitError, match="needs its target"):
        K.build_argv("ge_a1", 3501, _call())
    with pytest.raises(K.KitError, match="not registered"):
        K.build_argv("ge_a1", 3502, _call(), target=_target("ge_a1"))
    assert all(K.ARMS[r].side == "production" and K.ARMS[r].phase == GE.READER_PHASE for r in GE.READER_ARMS)
    assert not {a for a, _ in K.BATCH1} & set(GE.READER_ARMS)                  # not batch-1 forks (order, disk)


def test_describe_names_the_targets_run_as_the_fork_parent_and_plus_5m():
    d = K.describe("ge_d1", 3501, _call(), target=_target("ge_d1"))
    assert d["parent_run_dir"] == "/m/rb_x35_d1wide_s3501"
    assert d["delta_text"] == "+5,000,000 steps" and d["run_name"] == "rb_x35_ge_d1wide_s3501"
    assert d["target"]["path"].endswith("rb_x35_d1wide_s3501/final_model.zip")


def test_the_target_is_the_arms_finished_final_at_its_registered_length(tmp_path, monkeypatch):
    models = tmp_path / "models"
    with pytest.raises(K.KitError, match="no final_model.zip"):
        K.reader_target("ge_a1", 3501, _call(), models_dir=str(models))
    run = models / "rb_x35_a1arch_s3501"
    run.mkdir(parents=True)
    (run / "final_model.zip").write_bytes(b"final")
    monkeypatch.setattr(K, "_final_step", lambda p: PSTAR_STEP + 19_000_000)
    with pytest.raises(K.KitError, match="did not reach its registered length"):
        K.reader_target("ge_a1", 3501, _call(), models_dir=str(models))
    monkeypatch.setattr(K, "_final_step", lambda p: FINAL_STEP)
    t = K.reader_target("ge_a1", 3501, _call(), models_dir=str(models))
    assert t["step"] == FINAL_STEP and t["path"] == str((run / "final_model.zip").resolve()) and not t["standin"]
    assert len(t["sha256"]) == 64


def test_the_post_spawn_pool_check_exempts_a_reader_it_seeds_no_pool(tmp_path):
    assert K.check_pool_seed("ge_c0", str(tmp_path / "nowhere"), _call(), "", None) == ("PASS", [])


# ---------------------------------------------------------------------------------------------------- the trigger
def _stage(tmp_path: Path, fp32: bool = True) -> str:
    sha = "c" * 40 if fp32 else "FP32_ALT_SHA_PENDING"
    (tmp_path / B.SIDE_REQUIRED_FILE.format(side="production")).write_text(f"{sha} FP32_ALT the fp32 build\n")
    (tmp_path / B.READS_DIR).mkdir(exist_ok=True)
    return str(tmp_path)


def _save(stage: str, arm: str, look: int, cls: str, final: bool = True, verdict: str = "INCONCLUSIVE") -> None:
    key = "h" if arm == "d1" else "r1"
    doc = {"arm": B.READ_LABEL[arm], "look": look, "verdict": verdict, key: {"final": final, "class": cls}}
    (Path(stage) / B.READS_DIR / f"{arm}_look{look}.json").write_text(json.dumps(doc))


def test_a_reader_launches_only_on_its_arms_final_gain(tmp_path):
    st = _stage(tmp_path)
    assert B.stage_status(st, "ge_a1", 3501)[0] == B.PENDING                 # no read yet
    _save(st, "a1", 1, "UNDECIDED", final=False, verdict="CONTINUE")
    assert B.stage_status(st, "ge_a1", 3501)[0] == B.PENDING                 # look 2 may still read GAIN
    _save(st, "a1", 2, "NULL", verdict="NULL")
    status, why = B.stage_status(st, "ge_a1", 3501)
    assert status == B.REFUSED and "buys nothing" in why
    _save(st, "a1", 2, "GAIN")                                               # R1 GAIN: the verdict is INCONCLUSIVE
    assert B.stage_status(st, "ge_a1", 3501)[0] == B.ALLOWED
    assert B.stage_status(st, "ge_a2", 3501)[0] == B.PENDING                 # A2 has its own read


def test_d1s_reader_reads_its_primary_h_not_r1(tmp_path):
    st = _stage(tmp_path)
    _save(st, "d1", 1, "GAIN")
    assert B.stage_status(st, "ge_d1", 3501)[0] == B.ALLOWED
    assert B.final_class(st, "d1")[0] == "GAIN"


def test_an_inconclusive_primary_with_no_class_is_pending_not_refused(tmp_path):
    st = _stage(tmp_path)
    (Path(st) / B.READS_DIR / "a2_look1.json").write_text(json.dumps(
        {"arm": "A2", "look": 1, "verdict": "INCONCLUSIVE", "r1": {"final": True, "class": None}}))
    assert B.stage_status(st, "ge_a2", 3501)[0] == B.PENDING


def test_the_shared_c0_reader_waits_for_any_gain_and_is_refused_when_none_can_come(tmp_path):
    st = _stage(tmp_path)
    assert B.stage_status(st, "ge_c0", 3501)[0] == B.PENDING
    _save(st, "a1", 1, "NULL", verdict="NULL")
    _save(st, "a2", 1, "LOSS", verdict="LOSS")
    assert B.stage_status(st, "ge_c0", 3501)[0] == B.PENDING                 # D1 not read yet
    _save(st, "d1", 1, "GAIN")
    assert B.stage_status(st, "ge_c0", 3501)[0] == B.ALLOWED
    _save(st, "d1", 1, "UNDECIDED", verdict="UNDECIDED")
    assert B.stage_status(st, "ge_c0", 3501)[0] == B.REFUSED


def test_production_readers_wait_for_the_fp32_alt_commit(tmp_path):
    st = _stage(tmp_path, fp32=False)
    _save(st, "a1", 1, "GAIN")
    status, why = B.stage_status(st, "ge_a1", 3501)
    assert status == B.PENDING and "AM-8" in why


def test_the_kit_stage_check_cli_holds_a_reader_without_a_gain(tmp_path):
    st = _stage(tmp_path)
    assert K.main(["stage-check", "--arm", "ge_c0", "--seed", "3501", "--stage", st]) == 1
    _save(st, "a2", 1, "GAIN")
    assert K.main(["stage-check", "--arm", "ge_c0", "--seed", "3501", "--stage", st]) == 0


# ----------------------------------------------------------------------------------------------------- the pin
def test_the_sides_pin_judges_the_reader_runs_too(tmp_path):
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    d = models / "rb_x35_ge_a1arch_s3501"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"git_hash": "b" * 40, "pin_history": []}))
    ok, lines = B.pin_check("a" * 40, str(stage), str(models), "production")
    assert not ok and any("rb_x35_ge_a1arch_s3501" in x for x in lines)
    (d / "metadata.json").write_text(json.dumps({"git_hash": "a" * 40, "pin_history": []}))
    assert B.pin_check("a" * 40, str(stage), str(models), "production")[0]


# -------------------------------------------------------------------------------------------------------- the read
def _reader_run(models: Path, arm: str, *, wins: int, games: int = 300, dose: float = 1.0e-8,
                archetype: str = "offense", target: str = "", budget: int = 5_000_064,
                final: bool = True) -> E.ExploiterRun:
    rd = models / GE.reader_run(arm, 3501)
    rd.mkdir(parents=True, exist_ok=True)
    if final:
        (rd / "final_model.zip").write_bytes(b"r")
    tz = target or str(models / GE.target_run(arm, 3501) / "final_model.zip")
    pts = [E.SeriesPoint(step=FINAL_STEP + 2_000_000 * (i + 1), wins=wins // 3, games=games // 3, post_fork=True)
           for i in range(3)]
    return E.ExploiterRun(
        run=rd.name, run_dir=str(rd), target_run=GE.target_run(arm, 3501), target_file=tz, target_step=FINAL_STEP,
        target_pins_own_teams=False, fork_step=FINAL_STEP, num_timesteps=FINAL_STEP + budget, budget=budget,
        dose_rate=dose, lr_median=5.6e-5, teams=[], archetype=archetype, membership="exact",
        regime={"exploiter_keep_bots": False, "cycle_games": games // 3}, series=pts, lineage_derived=False,
        dose_source="tb")


def _models(tmp_path: Path, monkeypatch) -> Path:
    models = tmp_path / "models"
    for arm in ("a1", "a2", "d1", "c0"):
        d = models / K.run_name(arm, 3501)
        d.mkdir(parents=True)
        (d / "final_model.zip").write_bytes(b"f")
    monkeypatch.setattr(K, "_final_step", lambda p: FINAL_STEP)
    return models


def _reader_of(runs: Dict[str, E.ExploiterRun]):
    return lambda rd, ts: runs[Path(rd).name]


def test_the_r2_block_is_the_two_readers_pooled_counts(tmp_path, monkeypatch):
    models = _models(tmp_path, monkeypatch)
    on, off = _reader_run(models, "ge_a1", wins=180), _reader_run(models, "ge_c0", wins=150)
    r2 = GE.r2_input("a1", models_dir=str(models), reader=_reader_of({on.run: on, off.run: off}))
    assert r2["on"] == [180, 300] and r2["off"] == [150, 300] and r2["stat"] == "pooled"


@pytest.mark.parametrize("what,on_kw,off_kw,match", [
    ("dose 10 % gate", {"dose": 1.0e-8}, {"dose": 1.3e-8}, "not matched"),
    ("budget 2 % gate", {"budget": 5_000_064}, {"budget": 5_400_000}, "not matched"),
    ("an unfinished reader", {"final": False}, {}, "unfinished"),
    ("the wrong archetype", {"archetype": "stall"}, {}, "archetype"),
    ("a short budget", {"budget": 4_000_000}, {"budget": 4_000_000}, "budget"),
    ("another target file", {"target": "/elsewhere/final_model.zip"}, {}, "recorded target"),
])
def test_the_r2_block_refuses_what_g_e_is_not_defined_on(tmp_path, monkeypatch, what, on_kw, off_kw, match):
    models = _models(tmp_path, monkeypatch)
    on = _reader_run(models, "ge_a1", wins=180, **on_kw)
    off = _reader_run(models, "ge_c0", wins=150, **off_kw)
    with pytest.raises(GE.ReaderError, match=match):
        GE.r2_input("a1", models_dir=str(models), reader=_reader_of({on.run: on, off.run: off}))


def _read_doc(r2: Any) -> Dict[str, Any]:
    """An A1 look-1 read whose R1 is a clear GAIN and whose other guards all PASS (each well clear of its bar)."""
    doc: Dict[str, Any] = {"arm": "A1", "look": 1,
                           "r1": {"on": {"3501": 60.0, "3502": 60.5, "3503": 61.0},
                                  "off": {"3501": 50.0, "3502": 50.5, "3503": 51.0}},
                           "g_f": {"h_pp": [[51.0, 50.5, 51.5], [50.8, 51.2, 50.9], [51.1, 50.7, 51.3]]},
                           "g_c": {"b1": [0.0, -0.01, 0.01], "b2": [0.0, -0.01, 0.01]},
                           "g_k": {ts: {"on": [250, 500], "off": [250, 500]} for ts in R.G_K_TEAMSETS}}
    if r2 is not None:
        doc["r2"] = r2
    return doc


def test_with_the_readers_an_a1_gain_reaches_a_verdict_and_g_e_decides_adopt_or_trade(tmp_path, monkeypatch):
    assert R.read(_read_doc(None)).verdict == R.INCONCLUSIVE                    # the gap this unit closes
    models = _models(tmp_path, monkeypatch)
    off = _reader_run(models, "ge_c0", wins=150)
    same = _reader_run(models, "ge_a1", wins=153)
    v = R.read(_read_doc(GE.r2_input("a1", models_dir=str(models), reader=_reader_of({same.run: same, off.run: off}))))
    assert v.verdict == R.ADOPT
    worse = _reader_run(models, "ge_a1", wins=240)                              # ON detected MORE exploitable
    v = R.read(_read_doc(GE.r2_input("a1", models_dir=str(models), reader=_reader_of({worse.run: worse, off.run: off}))))
    assert v.verdict == R.TRADE and "G-E" in v.reasons[0]


def test_r2_is_only_defined_for_the_gain_arms():
    with pytest.raises(GE.ReaderError, match="no G-E"):
        GE.r2_input("c0", models_dir="/nowhere")


# ------------------------------------------------------------------------------- the committed validation transcripts
VALIDATED = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17/ge_readers_validation_dfeb3d92")


@pytest.mark.parametrize("arm", sorted(GE.READER_ARMS))
def test_the_validated_reader_argv_is_the_kits_and_its_real_transcripts_pass_the_verdict(arm):
    """The four argvs validated at dfeb3d92 on the stand-in target P* (checkargs + launcher --dry-run) are exactly what
    the kit builds today, and the kit's own verdict passes their transcripts; the same transcripts FAIL as a batch-1 arm
    would read them (the wrong parent / +steps), so the verdict is not vacuous."""
    call = _call()
    saved = (VALIDATED / f"argv_{arm}_s3501.txt").read_text().split()
    pstar = saved[saved.index("--model") + 1]                   # the stand-in: P*'s file, as validated
    assert pstar.endswith("rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip")
    t = GE.standin_target(pstar, PSTAR_STEP)
    d = K.describe(arm, 3501, call, target=t)
    assert saved == d["argv"]
    dry = (VALIDATED / f"dryrun_{arm}_s3501.txt").read_text()
    chk = (VALIDATED / f"checkargs_{arm}_s3501.txt").read_text()
    ok, line = K.check_validation(d, dry, chk, 0, 0)
    assert ok, line
    assert "self_play                False      (TYPED)" in dry                 # F16 reached the child's surface
    wrong = dict(d, delta_text="+20,000,000 steps")
    assert not K.check_validation(wrong, dry, chk, 0, 0)[0]
