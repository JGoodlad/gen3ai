"""X35 BATCH 2 in the kit (F-AS-8, STAGED_PLAN §4.1-§4.3): the SPRT and LR-F arms are their side's C0 argv token for
token plus ONE lever, their runs are judged against the side's written pin (the hole F-AS-8 found: a new arm's run on a
wrong pin passed), AM-6 and the disk pre-check count them, ``stage-check`` routes them through AM-8 and their own
registered conditions, the readout knows their labels, and the registered VOID guards read synthetic logs. Synthetic
files only; every case fails on revert of the rule it names, and every constructed number sits well clear of its
boundary except the deliberate rule-8 cases (standing rule 8)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from main.x35 import batch as B
from main.x35 import batch2_void as V
from main.x35 import kit as K
from main.x35 import readout as R
from main.x35.batch_test import FORK, OTHER, PIN, _fill_fp32, _g1_record, _run, _save, _series, _write_events
from main.x35.kit_test import standin_call, toy_call
from utils.paths import repo_path

STAGE = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17")
#: The ONE lever of each batch-2 arm, typed literally (STAGED_PLAN §4.1.1 / §4.1.2 / §4.2), never read from the kit.
LEVERS = {"sprt": ["--promotion-sprt"], "toy_sprt": ["--promotion-sprt"],
          "lrfloor": ["--min-lr", "1.7361e-4"], "toy_lrfloor": ["--min-lr", "2.0833e-4"]}
CONTROL = {"sprt": "c0", "lrfloor": "c0", "toy_sprt": "toy_c0", "toy_lrfloor": "toy_c0"}


def _call(arm: str) -> Dict:
    return toy_call() if K.ARMS[arm].side == "toy" else standin_call()


@pytest.mark.parametrize("arm,seed", [(a, s) for a in sorted(LEVERS) for s in K.ARMS[a].seeds])
def test_every_batch2_argv_is_its_sides_c0_token_for_token_plus_the_one_lever(arm, seed):
    """F-AS-8: one lever per A/B. Removing the lever's tokens leaves the side's C0 argv at the same seed, token for
    token, in order, but for the run name."""
    call = _call(arm)
    on, off = K.build_argv(arm, seed, call), K.build_argv(CONTROL[arm], seed, call)
    lever = LEVERS[arm]
    i = on.index(lever[0])
    assert on[i:i + len(lever)] == lever and on.count(lever[0]) == 1
    rest = on[:i] + on[i + len(lever):]
    assert rest[-2:] == ["--run-name", K.run_name(arm, seed)] and off[-2:] == ["--run-name", K.run_name(CONTROL[arm], seed)]
    assert rest[:-1] == off[:-1]
    assert len(on) == len(off) + len(lever)
    assert K.describe(arm, seed, call)["side"] == K.ARMS[CONTROL[arm]].side


def _transcript_argv(path: Path) -> List[str]:
    line = path.read_text().splitlines()[0]
    return line.split('--argv "', 1)[1].rstrip('"').split()


@pytest.mark.parametrize("arm,seed,transcript,pstar", [
    ("toy_sprt", 3601, "batch2_sprt/checkargs_toy_sprt_s3601_at_toy_pin.txt",
     ("rb_toy_ent_s4002_deep/checkpoints/checkpoint_290000087_steps.zip", 290000087)),
    ("toy_lrfloor", 3601, "batch2_lr_floor/checkargs_toy_lrfloor_s3601_at_toy_pin.txt",
     ("rb_toy_ent_s4002_deep/checkpoints/checkpoint_290000087_steps.zip", 290000087)),
    ("sprt", 3501, "batch2_sprt/checkargs_sprt_s3501.txt",
     ("rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip", 280000170)),
])
def test_the_kit_argv_equals_the_registered_checkargs_transcripts_argv(arm, seed, transcript, pstar):
    """The argv `checkargs` validated (AM-RP-f, at the toy pin 4609c155 for the toy) IS what `describe` builds."""
    want = _transcript_argv(STAGE / transcript)
    side = K.ARMS[arm].side
    call = {"side": side, "archive": [], "pstar": {"path": f"/home/goodlad/dev/gen3ai/models/{pstar[0]}",
                                                   "step": pstar[1], "sha256": "s"}}
    assert K.build_argv(arm, seed, call) == want


def test_batch2_arms_are_side_pinned_and_not_in_the_closed_stages():
    assert set(B.BATCH2_ARMS) == set(LEVERS) and set(B.BATCH2_ARMS) <= set(B.SIDE_PINNED_ARMS)
    assert not {a for _, forks in B.STAGES for a, _ in forks} & set(B.BATCH2_ARMS)       # STAGES is closed
    assert {(a, s) for _, forks in B.BATCH2_STAGES for a, s in forks} == \
        {(a, s) for a in LEVERS for s in K.ARMS[a].seeds}
    assert len(K.BATCH2) == 20 and not set(K.BATCH1) & set(K.BATCH2)


# --------------------------------------------------------------------------- AM-1: the side's pin covers batch 2
@pytest.mark.parametrize("run,side", [("rb_x35t_sprt_s3601", "toy"), ("rb_x35t_lrfloor_s3602", "toy"),
                                      ("rb_x35_sprt_s3501", "production"), ("rb_x35_lrfloor_s3503", "production")])
def test_a_batch2_run_on_a_wrong_pin_holds_the_sides_pin_check(tmp_path, run, side):
    """THE HOLE F-AS-8 found (f_as_8_kit_dryform.txt): a new arm's run recording a WRONG pin was listed "not batch 1,
    not judged" and the check PASSED. Now it is judged against its side's written pin and HOLDs."""
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    B.pin_write_if_absent(PIN, str(stage), side, "NA_TOY" if side == "toy" else "FP32_ALT")
    _run(models, run, PIN)
    ok, lines = B.pin_check(PIN, str(stage), str(models), side)
    assert ok and any("1 side-pinned" in x for x in lines), lines
    _run(models, run.replace("_s3", "_s9") + "x", OTHER)                   # an unregistered name: still not judged
    (models / run / "metadata.json").write_text(json.dumps({"git_hash": OTHER}))
    ok, lines = B.pin_check(PIN, str(stage), str(models), side)
    assert not ok and any(run in x and OTHER[:12] in x for x in lines), lines
    assert any("not a side-pinned arm" in x for x in lines), lines
    other_side = "production" if side == "toy" else "toy"
    ok, lines = B.pin_check(OTHER, str(stage), str(models), other_side)   # the other side lists it, does not judge it
    assert ok and any("judged at its own launches" in x and run in x for x in lines), lines


def test_the_k9b_pre_check_holds_on_a_void_finished_batch2_arm(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    _run(tmp_path / "models", "rb_x35t_sprt_s3601", PIN, final=True)
    void = B.judge_k9b("rb_x35t_sprt_s3601", FORK, _series([2e-3], [1e-6]))
    ok, lines = B.k9b_batch(str(stage), str(tmp_path / "models"), reader=lambda rd: void)
    assert not ok and lines == ["rb_x35t_sprt_s3601: K9(b) arm end VOID"]


def test_the_disk_check_counts_batch2s_forks(tmp_path):
    _fill_fp32(tmp_path)
    ok, msg = K.disk_check("/nonexistent", 10.0, thinning=False, existing=[], free_gib=1000.0, stage=str(tmp_path))
    assert ok and "46 batch-1/2 fork(s)" in msg, msg                     # 26 batch 1 + 20 batch 2
    have = [K.run_name("toy_sprt", s) for s in (3601, 3602, 3603)]
    ok, msg = K.disk_check("/nonexistent", 10.0, thinning=False, existing=have, free_gib=1000.0, stage=str(tmp_path))
    assert "43 batch-1/2 fork(s)" in msg, msg


# ------------------------------------------------------------------------------------- stage-check: AM-8 + batch 2
def test_production_batch2_is_pending_on_the_fp32_alt_placeholder_like_c0(tmp_path):
    """F-AS-8 item 3: `sprt` read ALLOWED while `c0` read PENDING on the FP32_ALT placeholder."""
    st = str(tmp_path)
    (tmp_path / B.SIDE_REQUIRED_FILE.format(side="production")).write_text("FP32_ALT_SHA_PENDING FP32_ALT x\n")
    for a in ("c0", "sprt", "lrfloor"):
        status, why = B.stage_status(st, a, 3501)
        assert status == B.PENDING and "FP32_ALT" in why, (a, why)
    for a in ("toy_sprt", "toy_lrfloor"):                                  # the toy has no AM-8 condition
        assert B.stage_status(st, a, 3601)[0] == B.ALLOWED
    _fill_fp32(tmp_path)
    for a in ("sprt", "lrfloor"):
        status, why = B.stage_status(st, a, 3502)
        assert status == B.ALLOWED and "B2-" in why, (a, why)


def test_batch2_look_2_needs_its_own_look_1_continue_and_licenses_toy_c0s_look_2(tmp_path):
    st = str(tmp_path)
    _fill_fp32(tmp_path)
    for a, s in (("sprt", 3504), ("lrfloor", 3505), ("toy_sprt", 3604), ("toy_lrfloor", 3605)):
        assert B.stage_status(st, a, s)[0] == B.PENDING, a
    _save(tmp_path, "toy_sprt", 1, verdict="NULL", r1={"final": True, "class": "NULL"})
    assert B.stage_status(st, "toy_sprt", 3604)[0] == B.REFUSED
    _save(tmp_path, "sprt", 1, verdict="CONTINUE", r1={"final": False, "class": "UNDECIDED"})
    assert B.stage_status(st, "sprt", 3505)[0] == B.ALLOWED
    assert B.stage_status(st, "toy_c0", 3604)[0] == B.PENDING             # nothing licenses toy C0's look 2 yet
    _save(tmp_path, "toy_lrfloor", 1, verdict="CONTINUE", r1={"final": False, "class": "UNDECIDED"})
    assert B.stage_status(st, "toy_lrfloor", 3604)[0] == B.ALLOWED
    status, why = B.stage_status(st, "toy_c0", 3605)                       # its shared control's seeds follow
    assert status == B.ALLOWED and "toy_lrfloor" in why, why


def test_production_lrf_is_refused_on_a_toy_lrf_final_loss_only(tmp_path):
    """STAGED_PLAN §4.1.2 (§4.1's licence table): toy LOSS ⇒ production LR-F is not run; NULL / unread ⇒ no gate."""
    st = str(tmp_path)
    _fill_fp32(tmp_path)
    assert B.stage_status(st, "lrfloor", 3501)[0] == B.ALLOWED                 # toy LR-F unread: AM-RP-e (8) first
    _save(tmp_path, "toy_lrfloor", 1, verdict="NULL", r1={"final": True, "class": "NULL"})
    assert B.stage_status(st, "lrfloor", 3501)[0] == B.ALLOWED
    (tmp_path / B.READS_DIR / "toy_lrfloor_look1.json").unlink()
    _save(tmp_path, "toy_lrfloor", 1, verdict="LOSS", r1={"final": True, "class": "LOSS"})
    status, why = B.stage_status(st, "lrfloor", 3502)
    assert status == B.REFUSED and "§4.1.2" in why, why
    assert B.stage_status(st, "sprt", 3501)[0] == B.ALLOWED                    # SPRT does not read LR-F


def test_g1_does_not_gate_batch2(tmp_path):
    st = str(tmp_path)
    _fill_fp32(tmp_path)
    _g1_record(tmp_path, B.HOLD)
    assert B.stage_status(st, "d1", 3501)[0] == B.REFUSED
    assert all(B.stage_status(st, a, 3501)[0] == B.ALLOWED for a in ("sprt", "lrfloor"))


def test_the_cli_admits_the_batch2_arms(tmp_path, capsys):
    call = tmp_path / "R.json"
    call.write_text(json.dumps(toy_call()))
    assert K.main(["stage-check", "--arm", "toy_sprt", "--seed", "3601", "--stage", str(tmp_path)]) == 0
    assert "ALLOWED" in capsys.readouterr().out
    assert K.main(["stage-check", "--arm", "toy_sprt", "--seed", "3606", "--stage", str(tmp_path)]) == 2   # unregistered
    assert K.main(["describe", "--arm", "toy_lrfloor", "--seed", "3601", "--call", str(call), "--field", "run_name"]) == 2
    assert "REFUSED" in capsys.readouterr().err                            # the stand-in call record does not verify


# --------------------------------------------------------------------------------------- the readout's labels
def test_the_readout_reads_sprt_and_lrf_and_their_records_round_trip(tmp_path):
    assert {"SPRT", "LRF"} <= R.ARMS and not {"SPRT", "LRF"} & R.R2_ARMS
    for label in ("SPRT", "LRF"):
        v = R.read({"arm": label, "look": 1, "r1": {"on": {"1": 40.0, "2": 41.0, "3": 39.0},
                                                   "off": {"1": 50.0, "2": 51.0, "3": 49.0}}})
        assert v.verdict == R.LOSS
    _save(tmp_path, "toy_sprt", 1, verdict="CONTINUE")
    assert B.read_record(str(tmp_path), "toy_sprt", 1)["arm"] == "SPRT"
    (tmp_path / B.READS_DIR / "sprt_look1.json").write_text(json.dumps({"arm": "LRF", "look": 1}))
    with pytest.raises(B.BatchError, match="not SPRT look 1"):
        B.read_record(str(tmp_path), "sprt", 1)


# ------------------------------------------------------------------------------------- the VOID guards: SPRT
def _jsonl(*rows: Dict) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def _start(s: int) -> Dict:
    return {"event": "start", "step": s}


def _verdict(s: int, promoted: bool, reason: str = "bound") -> Dict:
    return {"event": "verdict", "step": s, "promoted": promoted, "verdict": "accept" if promoted else "reject",
            "reason": reason}


def _abandoned(s: int) -> Dict:
    return {"event": "abandoned", "step": s, "promoted": False}


def test_sprt_void_v_s1_no_decided_verdict(tmp_path):
    rows, bad = V.parse_sprt_log(_jsonl(_start(2), _verdict(2, False, "abandoned")))
    r = V.judge_sprt("r", rows, finished=True, n_unparseable=bad)
    assert r.verdict == V.VOID and any("V-S1" in f for f in r.fires)
    r = V.judge_sprt("r", [], finished=True)                               # no log at all: the lever never acted
    assert r.verdict == V.VOID and any("V-S1" in f for f in r.fires)


def test_sprt_void_v_s2_is_strictly_above_a_quarter():
    def log(n_ok: int, n_ab: int):
        rows = []
        for i in range(n_ok):
            rows += [_start(i), _verdict(i, i % 2 == 0)]
        for j in range(n_ab):
            rows += [_start(100 + j), _abandoned(100 + j)]
        return V.parse_sprt_log(_jsonl(*rows))[0]

    r = V.judge_sprt("r", log(3, 1), finished=True)                        # 1 of 4 = exactly 25 %: not above
    assert r.verdict == V.PASS and r.detail == {"candidates": 4, "decided": 3, "abandoned": 1, "promoted": 2,
                                                "abandoned_steps": [100]}
    r = V.judge_sprt("r", log(5, 2), finished=True)                        # 2 of 7 = 28.6 %
    assert r.verdict == V.VOID and any("V-S2" in f for f in r.fires)


def test_sprt_void_counts_a_start_without_an_end_and_an_in_test_abandon_as_abandoned():
    rows = V.parse_sprt_log(_jsonl(_start(1), _verdict(1, True), _start(2), _verdict(2, False), _start(3),
                                   _verdict(3, False), _start(4), _start(5), _verdict(5, False, "abandoned")))[0]
    r = V.judge_sprt("r", rows, finished=True)
    assert r.detail["abandoned_steps"] == [4, 5] and r.verdict == V.VOID   # 2 of 5 > 25 %


def test_sprt_void_is_inconclusive_before_the_final_or_on_an_unparseable_line(tmp_path):
    ok = _jsonl(_start(1), _verdict(1, True))
    assert V.judge_sprt("r", V.parse_sprt_log(ok)[0], finished=False).verdict == V.INCONCLUSIVE
    rows, bad = V.parse_sprt_log(ok + '{"event": "verd')
    assert bad == 1 and V.judge_sprt("r", rows, finished=True, n_unparseable=bad).verdict == V.INCONCLUSIVE
    run = tmp_path / "rb_x35t_sprt_s3601"
    run.mkdir()
    (run / V.SPRT_LOG).write_text(ok)
    assert V.read_sprt(str(run)).verdict == V.INCONCLUSIVE                 # no final_model.zip yet
    (run / "final_model.zip").write_bytes(b"z")
    assert V.read_sprt(str(run)).verdict == V.PASS


# ------------------------------------------------------------------------------------- the VOID guards: LR-F
W = V.LRF_WINDOW_STEPS


def _kl(values, start=FORK + 100_000, every=100_000):
    return [(start + i * every, v) for i, v in enumerate(values)]


def test_lrf_kl_void_fires_on_a_trailing_2m_mean_above_008():
    calm = _kl([0.03] * 40)
    assert V.judge_lrf_kl("r", FORK, [calm], finished=True).verdict == V.PASS
    spike = _kl([0.03] * 20 + [0.15] * 15 + [0.03] * 5)                   # a 1.5M burst at 0.15: mean > 0.08
    r = V.judge_lrf_kl("r", FORK, [spike], finished=True)
    assert r.verdict == V.VOID and r.detail["max_window_mean"] > 0.09
    blip = _kl([0.03] * 20 + [0.5] + [0.03] * 19)                          # one high row: the 2M mean stays low
    assert V.judge_lrf_kl("r", FORK, [blip], finished=True).verdict == V.PASS


def test_lrf_kl_rule8_a_mean_on_the_bar_is_recorded_not_fired():
    on_bar = _kl([0.08] * 40)
    r = V.judge_lrf_kl("r", FORK, [on_bar], finished=True)
    assert r.verdict == V.PASS and r.near
    past = _kl([0.08 + 1e-6] * 40)
    assert V.judge_lrf_kl("r", FORK, [past], finished=True).verdict == V.VOID


def test_lrf_kl_reads_only_after_the_fork_and_a_replay_keeps_the_later_segment():
    inherited = [(FORK - 50_000, 0.5)]                                     # the parent's row: never read
    first = _kl([0.03] * 30 + [0.2] * 10)                                  # a crashed segment's high rows ...
    replay = _kl([0.03] * 10, start=FORK + 3_100_000)                      # ... replayed lower by the restart
    r = V.judge_lrf_kl("r", FORK, [inherited + first, replay], finished=True)
    assert r.verdict == V.PASS and r.detail["rows_after_fork"] == 40, r


def test_lrf_kl_absence_unknown_fork_non_finite_and_unfinished():
    assert V.judge_lrf_kl("r", FORK, [_kl([0.03] * 10)], finished=True).verdict == V.INCONCLUSIVE   # < one window
    assert V.judge_lrf_kl("r", None, [_kl([0.03] * 40)], finished=True).verdict == V.INCONCLUSIVE
    assert V.judge_lrf_kl("r", FORK, [_kl([0.03] * 39 + [float("nan")])], finished=True).verdict == V.VOID
    assert V.judge_lrf_kl("r", FORK, [_kl([0.03] * 40)], finished=False).verdict == V.INCONCLUSIVE


def test_lrf_kl_reads_a_synthetic_run_dir_and_the_kit_void_cli(tmp_path, capsys):
    run = tmp_path / "rb_x35t_lrfloor_s3601"
    tb = run / "tb"
    tb.mkdir(parents=True)
    (tb / "INHERITED_FROM.json").write_text(json.dumps({"fork_step": FORK}))
    _write_events(tb, [(V.LRF_KL_TAG, s, v) for s, v in _kl([0.03] * 20 + [0.12] * 25)])
    (run / "final_model.zip").write_bytes(b"z")
    r = V.read_lrf_kl(str(run))
    assert r.verdict == V.VOID and r.detail["fork_step"] == FORK and r.detail["rows_after_fork"] == 45, r
    assert K.main(["void", "--arm", "toy_lrfloor", "--run-dir", str(run)]) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == V.VOID
    srun = tmp_path / "rb_x35t_sprt_s3601"
    srun.mkdir()
    (srun / V.SPRT_LOG).write_text(_jsonl(_start(1), _verdict(1, True)))
    (srun / "final_model.zip").write_bytes(b"z")
    assert K.main(["void", "--arm", "toy_sprt", "--run-dir", str(srun)]) == 0
    capsys.readouterr()
