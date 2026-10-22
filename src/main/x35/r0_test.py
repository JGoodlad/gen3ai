"""X35 R0, the exploitability curve (``main.x35.r0``; launch sheet §0 AM-R0): the argv IS the G-E reader's with the
side's start node as the target and +20M, the kit argv equals the committed ``checkargs`` transcripts, the arms are
side-pinned outside the closed ``STAGES``, production waits for FP32_ALT, the read plan refuses a thinned run, and the
registered class is deterministic. Synthetic files only; nothing launches or plays. Every constructed number sits well
clear of its boundary except the deliberate rule-8 cases (standing rule 8)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

from main.x35 import batch as B
from main.x35 import kit as K
from main.x35 import r0 as R
from main.x35.batch_test import PIN, OTHER, _fill_fp32, _run
from main.x35.ge import standin_target
from main.x35.kit_test import flags, standin_call, toy_call
from utils.paths import repo_path

STAGE = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17")


def _call(arm: str) -> Dict:
    return toy_call() if K.ARMS[arm].side == "toy" else standin_call()


# ------------------------------------------------------------------------------------------------------ the argv
def test_r0_is_the_ge_readers_argv_with_the_start_node_as_target_and_plus_20m():
    """F-LS-19: "R0 is the SAME argv with P* as the target". Only --steps (+20M, not +5M) and the run name differ."""
    call = standin_call()
    p = call["pstar"]
    r0 = flags(K.build_argv("r0", 3531, call))
    ge = flags(K.build_argv("ge_c0", 3501, call, target=standin_target(p["path"], p["step"])))
    assert {k for k in r0 if r0[k] != ge.get(k)} == {"--steps", "--run-name"} and set(r0) == set(ge)
    assert r0["--model"] == r0["--exploiter"] == p["path"] and r0["--no-self-play"] is True
    assert r0["--steps"] == str(p["step"] + 20_000_000) and r0["--seed"] == "3531" == ge["--seed"]
    assert r0["--run-name"] == "rb_x35_r0_s3531"


def test_toy_r0_is_productions_form_plus_the_toy_side_extras():
    on, prod = flags(K.build_argv("toy_r0", 3631, toy_call())), flags(K.build_argv("r0", 3531, standin_call()))
    assert set(on) - set(prod) == {"--snapshot-ladder-games", "--supply-starve-cycles"} and set(prod) <= set(on)
    assert on["--steps"] == str(toy_call()["pstar"]["step"] + 20_000_000) and on["--run-name"] == "rb_x35t_r0_s3631"
    assert on["--model"] == on["--exploiter"] == toy_call()["pstar"]["path"]
    assert {k for k in prod if prod[k] != on[k]} == {"--model", "--exploiter", "--steps", "--seed", "--run-name"}
    with pytest.raises(K.KitError, match="reads the toy call record"):
        K.build_argv("toy_r0", 3631, standin_call())
    with pytest.raises(K.KitError, match="not registered"):
        K.build_argv("r0", 3501, standin_call())


def _transcript_argv(path: Path) -> List[str]:
    line = path.read_text().splitlines()[0]
    return line.split('--argv "', 1)[1].rstrip('"').split()


@pytest.mark.parametrize("arm,seed,transcript,start", [
    ("toy_r0", 3631, "r0/checkargs_toy_r0_s3631_at_toy_pin.txt",
     ("rb_toy_ent_s4002_deep/checkpoints/checkpoint_290000087_steps.zip", 290000087)),
    ("r0", 3531, "r0/checkargs_r0_s3531.txt", ("rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip", 280000170)),
    ("r0", 3531, "r0/checkargs_r0_s3531_at_head.txt",
     ("rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip", 280000170)),
])
def test_the_kit_argv_equals_the_committed_checkargs_transcripts_argv_and_each_exited_0(arm, seed, transcript, start):
    p = STAGE / transcript
    want = _transcript_argv(p)
    assert p.read_text().rstrip().endswith("[exit 0]") and "unrecognized                   : 0" in p.read_text()
    model = want[want.index("--model") + 1]
    assert model.endswith(f"/models/{start[0]}"), model
    call = {"side": K.ARMS[arm].side, "archive": [], "pstar": {"path": model, "step": start[1], "sha256": "s"}}
    assert K.build_argv(arm, seed, call) == want


# ------------------------------------------------------------------------------------------ the gates (AM-1 / AM-8)
def test_r0_is_side_pinned_and_outside_the_closed_batches():
    assert set(B.R0_ARMS) == set(R.R0_ARMS) and set(B.R0_ARMS) <= set(B.SIDE_PINNED_ARMS)
    assert not {a for _, forks in B.STAGES + B.BATCH2_STAGES for a, _ in forks} & set(R.R0_ARMS)
    assert not {a for a, _ in K.BATCH1 + K.BATCH2} & set(R.R0_ARMS)
    assert {(a, s) for _, forks in B.R0_STAGES for a, s in forks} == {("toy_r0", 3631), ("r0", 3531)}
    assert {K.ARMS[a].side for a in R.R0_ARMS} == {"toy", "production"} and K.ARMS["toy_r0"].run.startswith("rb_x35t_")
    assert all(K.ARMS[a].phase == R.R0_PHASE and K.ARMS[a].length == 20_000_000 for a in R.R0_ARMS)
    assert R.R0_ARMS.keys() <= K.NO_POOL_ARMS


@pytest.mark.parametrize("run,side", [("rb_x35t_r0_s3631", "toy"), ("rb_x35_r0_s3531", "production")])
def test_an_r0_run_on_a_wrong_pin_holds_its_sides_pin_check(tmp_path, run, side):
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    B.pin_write_if_absent(PIN, str(stage), side, "NA_TOY" if side == "toy" else "FP32_ALT")
    _run(models, run, PIN)
    ok, lines = B.pin_check(PIN, str(stage), str(models), side)
    assert ok and any("1 side-pinned" in x for x in lines), lines
    (models / run / "metadata.json").write_text(json.dumps({"git_hash": OTHER}))
    ok, lines = B.pin_check(PIN, str(stage), str(models), side)
    assert not ok and any(run in x and OTHER[:12] in x for x in lines), lines


def test_production_r0_waits_for_fp32_alt_and_toy_r0_has_no_condition(tmp_path):
    assert B.stage_status(str(tmp_path), "toy_r0", 3631)[0] == B.ALLOWED
    st, why = B.stage_status(str(tmp_path), "r0", 3531)
    assert st == B.PENDING and "FP32_ALT" in why
    _fill_fp32(tmp_path)
    assert B.stage_status(str(tmp_path), "r0", 3531)[0] == B.ALLOWED
    assert B.stage_of("r0", 3531) == "R0-P" and B.stage_of("toy_r0", 3631) == "R0-T"


def test_an_r0_fork_seeds_no_pool_so_the_post_spawn_check_passes(tmp_path):
    assert K.check_pool_seed("r0", str(tmp_path), standin_call(), "", None) == ("PASS", [])


# ------------------------------------------------------------------------------------------------------ the plan
def _ckpts(run: Path, steps: Sequence[int]) -> None:
    (run / "checkpoints").mkdir(parents=True)
    for s in steps:
        (run / "checkpoints" / f"checkpoint_{s}_steps.zip").write_bytes(b"z")


def test_the_plan_takes_the_start_file_at_0_and_the_first_checkpoint_in_each_window(tmp_path):
    fork = 290_000_087
    run = tmp_path / "rb_x35t_r0_s3631"
    _ckpts(run, [fork + m * 1_000_000 + 129 for m in range(1, 21)] + [fork + 4_000_000 + 900_000])
    plan = R.curve_plan(str(run), fork, "/start.zip")
    assert [k for k, _, _ in plan] == list(R.READ_POINTS) and plan[0] == (0, "/start.zip", fork)
    assert plan[2][2] == fork + 4_000_129                       # the FIRST checkpoint of [fork + 4M, fork + 5M)
    assert plan[-1][2] == fork + 20_000_129


def test_a_thinned_read_point_is_a_refusal_and_the_final_covers_the_cap(tmp_path):
    fork = 280_000_170
    run = tmp_path / "rb_x35_r0_s3531"
    _ckpts(run, [fork + m * 1_000_000 + 7 for m in range(1, 20) if m != 6])
    with pytest.raises(R.R0Error, match=r"read point \+6,000,000"):
        R.curve_plan(str(run), fork, "/s.zip")
    (run / "checkpoints" / f"checkpoint_{fork + 6_000_007}_steps.zip").write_bytes(b"z")
    with pytest.raises(R.R0Error, match=r"read point \+20,000,000"):
        R.curve_plan(str(run), fork, "/s.zip", final_step=fork + 19_999_000)     # short of the cap
    (run / "final_model.zip").write_bytes(b"f")
    assert R.curve_plan(str(run), fork, "/s.zip", final_step=fork + 20_000_200)[-1][1].endswith("final_model.zip")


def test_the_windows_sit_on_the_trainers_absolute_save_boundaries(tmp_path):
    """R0-toy's REAL grid (rb_x35t_r0_s3631): saves land just past each absolute 1M multiple, so the +6M save
    (296,000,066) is SHORT of fork + 6M = 296,000,087; a fork-anchored window refused it (the first read, 2026-10-21)."""
    fork = 290_000_087
    run = tmp_path / "rb_x35t_r0_s3631"
    grid = [291000121, 292000095, 293000366, 294000197, 295000107, 296000066, 297000177, 298000012, 299000054,
            300000063, 301000015, 302000218, 303000069, 304000041, 305000059, 306000306, 307000081, 308000096,
            309000170, 310000106]
    _ckpts(run, grid)
    plan = R.curve_plan(str(run), fork, "/start.zip", final_step=310_059_274)
    assert [s for _, _, s in plan] == [fork] + grid[1::2]


# ----------------------------------------------------------------------------------------------------- the class
def _curve(rates: Dict[int, float], n: int = 400, timeouts: int = 0) -> List[R.Point]:
    return [R.Point(k, round(rates[k] * n), n, 0, n + timeouts) for k in R.READ_POINTS]


def _sat(g0: float, h: float, t: int) -> Dict[int, float]:
    """gap(0) = g0, reaching the plateau h at read point t and flat after."""
    return {k: 0.5 + (h if k >= t else g0 + (h - g0) * k / t) for k in R.READ_POINTS}


def test_a_curve_flat_from_8m_reads_saturated_with_its_t90():
    c = R.classify(_curve(_sat(0.0, 0.16, 8_000_000), n=4000))
    assert c["cls"] == R.SATURATED and c["t90"] == 8_000_000 and not c["no_rise"]
    assert abs(c["headroom"] - 0.16) < 1e-3 and c["learned"][1] > 0.1


def test_a_curve_still_climbing_at_the_cap_reads_rising():
    rates = {k: 0.5 + 0.02 + 0.015 * k / 1_000_000 for k in R.READ_POINTS}       # +3 pp per 2M, 9 pp late - early
    c = R.classify(_curve(rates))
    assert c["cls"] == R.RISING and c["late_minus_early"][1] > 0


def test_a_noisy_curve_neither_rising_nor_flat_reads_unresolved():
    rates = _sat(0.0, 0.15, 8_000_000)
    rates.update({16_000_000: 0.68, 18_000_000: 0.67, 20_000_000: 0.68})           # late +3 pp over early, n = 400
    c = R.classify(_curve(rates))
    assert c["cls"] == R.UNRESOLVED, c["why"]


def test_no_learned_exploitation_reads_no_rise_and_no_t90():
    c = R.classify(_curve({k: 0.53 for k in R.READ_POINTS}, n=4000))
    assert c["no_rise"] and c["t90"] is None and c["cls"] == R.SATURATED and abs(c["gap0"] - 0.03) < 1e-9


def test_t90_within_1e9_of_its_bar_is_not_reached():
    """Rule 8: the bar is g0 + 0.9 (H - g0); a point AT the bar does not reach it, the next one above does."""
    rates = _sat(0.0, 0.20, 10_000_000)                    # bar = 0.18; 8M sits on 0.16, 10M on 0.20
    rates[8_000_000] = 0.68                                 # wins 272 / 400 -> gap exactly 0.18 = the bar
    c = R.classify(_curve(rates))
    assert abs(c["points"][4]["gap"] - 0.18) < 1e-12 and c["t90"] == 10_000_000


def test_a_missing_point_is_refused_and_heavy_timeouts_are_inconclusive():
    pts = _curve(_sat(0.0, 0.16, 8_000_000))
    with pytest.raises(R.R0Error, match="lacks read point"):
        R.classify(pts[:-1])
    c = R.classify(_curve(_sat(0.0, 0.16, 8_000_000), timeouts=200))               # 200 / 600 = 33 % > 25 %
    assert c["cls"] == R.INCONCLUSIVE and "timeouts" in c["why"]


def test_the_a6_prefix_cell_pools_only_the_first_5m_of_post_fork_rows():
    from agents.training.best_response_gap import SeriesPoint

    class Run:
        run, fork_step = "rb_x35_r0_s3531", 1_000
        post_fork = [SeriesPoint(1_000 + s, w, 100, True) for s, w in
                     ((2_000_000, 60), (4_000_000, 64), (5_000_000, 70), (6_000_000, 99))]

    assert R.prefix_cell(Run()) == (194, 300)
    Run.post_fork = []
    with pytest.raises(R.R0Error, match="no post-fork"):
        R.prefix_cell(Run())


def test_a_curve_record_is_written_once(tmp_path):
    out = tmp_path / "curve.json"
    out.write_text("{}")
    with pytest.raises(R.R0Error, match="written ONCE"):
        R.play_curve(str(tmp_path), str(out), call=standin_call())


def test_a_ledgered_read_banks_each_point_resumes_without_replaying_and_writes_the_record_once(tmp_path, monkeypatch):
    """The read runs in <= 45-min chunks (rule 5b): a banked point is reused, never replayed; ``max_points`` stops a
    chunk without a record; a banked point for another file is a refusal; the record equals one un-chunked read's."""
    from types import SimpleNamespace

    from agents.training import best_response_gap as E
    from agents.training import untaught_meter as um

    fork = 290_000_087
    run_dir = tmp_path / "rb_x35t_r0_s3631"
    grid = [290_000_000 + m * 1_000_000 + 33 for m in range(1, 21)]
    _ckpts(run_dir, grid)
    (run_dir / "final_model.zip").write_bytes(b"f")
    start = str(tmp_path / "start.zip")
    Path(start).write_bytes(b"s")
    run = SimpleNamespace(run=run_dir.name, target_file=start, archetype="offense", budget=R.L_R0, fork_step=fork,
                          num_timesteps=fork + R.L_R0 + 50, teams=["t"], post_fork=[])
    played: List[str] = []

    def play_cells(refs, teams, target, *, games_per_team, seed, stochastic, progress=None, game_log=None):
        out = {}
        for r in refs:
            played.append(r.label)
            w = 40 + int(r.label[1:3])            # the gap rises by 1 game per 1M of k
            out[r.label] = {"X0": um.Cell(wins=w, losses=80 - w, finished=80, attempted=80, transport="rust_eval")}
            if game_log is not None:
                game_log[(r.label, "X0")] = [{"game": 0}]
        return out

    monkeypatch.setattr(E, "read_exploiter", lambda *a, **k: run)
    monkeypatch.setattr(E, "load_teamsets", lambda: None)
    monkeypatch.setattr(um, "team_slices", lambda *a, **k: ["X0"])
    monkeypatch.setattr(um, "resolve_ref", lambda path, label, role=None: SimpleNamespace(label=label, path=path))
    monkeypatch.setattr(um, "play_cells", play_cells)
    call = {"pstar": {"path": start, "step": fork}}
    led, out = str(tmp_path / "ledger"), str(tmp_path / "curve.json")
    part = R.play_curve(str(run_dir), out, call=call, games=80, ledger=led, max_points=4)
    assert part["partial"] and part["remaining"] == 7 and played == ["k00M", "k02M", "k04M", "k06M"]
    assert not Path(out).exists()
    doc = R.play_curve(str(run_dir), out, call=call, games=80, ledger=led)
    assert played[4:] == [f"k{k // 1_000_000:02d}M" for k in R.READ_POINTS[4:]]       # nothing replayed
    ref = R.play_curve(str(run_dir), str(tmp_path / "once.json"), call=call, games=80)
    assert {k: v for k, v in doc.items()} == ref and doc["points"][-1]["wins"] == 60
    bad = json.loads((Path(led) / "k02M.json").read_text())
    bad["step"] += 1
    (Path(led) / "k02M.json").write_text(json.dumps(bad))
    with pytest.raises(R.R0Error, match="banked for"):
        R.play_curve(str(run_dir), str(tmp_path / "again.json"), call=call, games=80, ledger=led)
