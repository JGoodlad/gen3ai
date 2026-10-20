"""X35's launch kit (``main.x35.kit``): every argv it builds from a call record EQUALS the staging's validated
stand-in (``checkargs`` / ``launcher --dry-run`` at 9ebda190) flag for flag, the two-phase records refuse what they
must, and the disk pre-check's arithmetic. Synthetic files only; nothing launches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import pytest

from main.x35 import kit as K
from utils.paths import repo_path

STAGE = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17")
VALIDATED = STAGE / "batch1_validation_803a13e3"   # the 24 argvs validated at 803a13e3 (the toy on the 200M stand-in)
C = "models/rb_x26_s2601/checkpoints"
ARCHIVE_STEPS = {20: 20000043, 40: 40000045, 60: 60000135, 80: 80000111, 100: 100000037, 120: 120000055,
                 140: 140000062, 160: 160000085}


def flags(tokens: List[str]) -> Dict[str, Union[str, bool]]:
    """argparse's view of an argv: flag -> value (a bare flag -> True; a repeated flag -> its last value)."""
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


def standin_call() -> Dict:
    """The stand-in P* (180,000,055) and its archive, in the frozen R file's shape (refpop.freeze)."""
    arch = [{"k": k, "label": f"x26_{n:03d}m", "path": f"{C}/checkpoint_{ARCHIVE_STEPS[n]}_steps.zip",
             "nominal": n * 1_000_000, "step": ARCHIVE_STEPS[n], "sha256": f"sha{n}"}
            for k, n in enumerate(sorted(ARCHIVE_STEPS), start=1)]
    return {"side": "production", "archive": arch,
            "pstar": {"label": "x26_pstar", "path": f"{C}/checkpoint_180000055_steps.zip", "step": 180000055,
                      "sha256": "shaP"}}


def standin(name: str) -> List[str]:
    return (STAGE / f"argv_{name}_standin.txt").read_text().split()


@pytest.mark.parametrize("arm,seed,name", [("c0", 3501, "c0_s3501"), ("a1", 3501, "a1_s3501"),
                                           ("a2", 3501, "a2_s3501"), ("a3_mx", 3511, "a3_main_exploiter"),
                                           ("a6_lx", 3521, "a6_league_exploiter"), ("d1", 3501, "d1_s3501")])
def test_every_production_argv_equals_its_validated_stand_in(arm, seed, name):
    assert flags(K.build_argv(arm, seed, standin_call())) == flags(standin(name))


def toy_call(path: str = "models/rb_toy_ent_s4002/final_model.zip", step: int = 100081428) -> Dict:
    return {"side": "toy", "archive": [],
            "pstar": {"label": "toy_pstar", "path": path, "step": step, "sha256": "shaT"}}


def test_the_toy_argv_equals_its_validated_stand_in_but_for_c3s_warn():
    """The staged stand-in (9ebda190) typed K9(b) `fatal`; C3 (2026-10-19) moved the toy arms to `warn`. Nothing else moved."""
    want = flags(standin("toy_c0"))
    assert want["--behaviour-check"] == "fatal"
    want["--behaviour-check"] = "warn"
    assert flags(K.build_argv("toy_c0", 3601, toy_call())) == want


def test_a_and_c_share_everything_but_the_lever():
    call = standin_call()
    on, off = flags(K.build_argv("a1", 3502, call)), flags(K.build_argv("c0", 3502, call))
    lever = {"--stable-opponents", "--stable-opponent-selfplay-share"}
    assert set(on) - set(off) == lever
    assert {k: v for k, v in on.items() if k not in lever | {"--run-name"}} == \
        {k: v for k, v in off.items() if k != "--run-name"}


def test_unregistered_seeds_arms_and_sides_are_refused():
    with pytest.raises(K.KitError, match="not registered"):
        K.build_argv("a1", 3511, standin_call())
    with pytest.raises(K.KitError, match="not one of"):
        K.build_argv("a7", 3501, standin_call())
    with pytest.raises(K.KitError, match="reads the toy call record"):
        K.build_argv("toy_a1", 3601, standin_call())
    with pytest.raises(K.KitError, match="needs the a3_mx phase-1 record"):
        K.build_argv("a3", 3501, standin_call())


def _exploiter_runs(tmp_path: Path, arm: str, step: int) -> Path:
    md = tmp_path / "models"
    for s in K.ARMS[arm].seeds:
        d = md / K.run_name(arm, s)
        d.mkdir(parents=True)
        (d / "final_model.zip").write_bytes(f"{arm}{s}".encode())
    return md


def test_the_phase_1_record_feeds_phase_2_and_refuses_a_short_final(tmp_path, monkeypatch):
    call = standin_call()
    md = _exploiter_runs(tmp_path, "a3_mx", 0)
    monkeypatch.setattr(K, "_final_step", lambda z: 185000055)
    rec = K.phase1_record("a3_mx", call, str(md), str(tmp_path / "p1.json"))
    assert [e["label"] for e in rec["exploiters"]] == ["mx_s3511", "mx_s3512", "mx_s3513"]
    assert K.verify_phase1(str(tmp_path / "p1.json")) == []
    f = flags(K.build_argv("a3", 3501, call, rec))
    assert f["--stable-opponent-selfplay-share"] == "0.4" and f["--stable-opponent-pfsp"] is True
    assert str(f["--stable-opponents"]).count(":mx_s35") == 3 and f["--run-name"] == "rb_x35_a3mx_s3501"
    with pytest.raises(K.KitError, match="phase-1 record is for a3_mx"):
        K.build_argv("a6", 3501, call, rec)
    with pytest.raises(K.KitError, match="written ONCE"):
        K.phase1_record("a3_mx", call, str(md), str(tmp_path / "p1.json"))
    monkeypatch.setattr(K, "_final_step", lambda z: 185000054)
    with pytest.raises(K.KitError, match="< the registered"):
        K.phase1_record("a3_mx", call, str(md), str(tmp_path / "p1b.json"))
    (md / "rb_x35_mx_s3512" / "final_model.zip").write_bytes(b"swapped")
    assert any("sha256 changed" in p for p in K.verify_phase1(str(tmp_path / "p1.json")))


def test_a_missing_exploiter_final_is_refused(tmp_path):
    md = _exploiter_runs(tmp_path, "a6_lx", 0)
    (md / "rb_x35_lx_s3523" / "final_model.zip").unlink()
    with pytest.raises(K.KitError, match="no final_model.zip"):
        K.phase1_record("a6_lx", standin_call(), str(md), str(tmp_path / "p1.json"))


def test_the_disk_check_counts_the_whole_remaining_batch():
    ok, msg = K.disk_check("/nonexistent", 20.68, thinning=False, existing=[], free_gib=318.0)
    assert not ok and "32 batch-1 fork(s)" in msg    # AM-7's 32 forks, the conditional stages counted (PENDING)
    ok, _ = K.disk_check("/nonexistent", 20.68, thinning=True, existing=[], free_gib=318.0)
    assert ok
    done = [K.run_name(a, s) for a, s in K.BATCH1[:7]]
    ok, msg = K.disk_check("/nonexistent", 20.68, thinning=False, existing=done, free_gib=600.0)   # 25 x 20.68 = 517.0
    assert ok and "25 batch-1" in msg
    ok, msg = K.disk_check("/nonexistent", 20.68, thinning=True, existing=[], free_gib=10.0)
    assert not ok and "not even this launch" in msg


def test_the_disk_check_counts_x47s_three_forks():
    """F-LS-1 / AM-4: X47's runs (look 1 and look 2) are batch-1 forks the disk check reserves room for
    (5 x 13.05 GiB = 65.25 GiB), and stops counting once they exist. Free space is well clear of the 65.25 boundary
    on both sides (standing rule 8)."""
    x47 = [K.run_name("toy_x47", s) for s in (3601, 3602, 3603, 3604, 3605)]
    assert x47 == [f"rb_x35t_x47d4_s{s}" for s in range(3601, 3606)]
    others = sorted({K.run_name(a, s) for a, s in K.BATCH1} - set(x47))
    ok, msg = K.disk_check("/nonexistent", 13.05, thinning=False, existing=others, free_gib=75.0)
    assert ok and "5 batch-1 fork(s)" in msg, msg
    ok, msg = K.disk_check("/nonexistent", 13.05, thinning=False, existing=others, free_gib=55.0)
    assert not ok and "5 batch-1 fork(s)" in msg, msg
    ok, msg = K.disk_check("/nonexistent", 13.05, thinning=False, existing=others + x47, free_gib=1.0)
    assert ok and "0 batch-1 fork(s)" in msg, msg


def test_describe_names_the_dry_run_expectations():
    d = K.describe("a6_lx", 3522, standin_call())
    assert d["parent_run_dir"] == "models/rb_x26_s2601" and d["delta_text"] == "+5,000,000 steps"
    assert d["run_name"] == "rb_x35_lx_s3522" and json.dumps(d)


# ------------------------------------------------------------------------------- T54's post-spawn pool-seed check
def _seeded(tmp_path: Path, **over) -> Path:
    run = tmp_path / "rb_x35_a1arch_s3501"
    (run / "snapshots").mkdir(parents=True)
    rec = {"schema": 2, "fork_step": 180000055, "snapshots": ["snapshot_176000000.zip", "snapshot_180000055.zip"],
           "dropped_steps": [184000013, 188000021], **over}
    (run / "snapshots" / "pool_seed.json").write_text(json.dumps(rec))
    return run


LINE = ("🌱 [SELFPLAY] [pool] seeded 2 snapshots + metadata from rb_x26_s2601 (win_rate_vs_bots=81.00%) "
        "[files: summary.json] [step filter <= 180000055: DROPPED 2 later snapshot(s) at steps "
        "[184000013, 188000021], pruned ['pfsp_win_rates']]")


def test_the_pool_seed_check_passes_a_filtered_seed_matching_its_startup_line(tmp_path):
    run = _seeded(tmp_path)
    assert K.check_pool_seed("a1", str(run), standin_call(), LINE, 180000055) == ("PASS", [])
    st, _ = K.check_pool_seed("a1", str(run), standin_call(), "", 180000055)
    assert st == K.PENDING
    assert K.check_pool_seed("a1", str(tmp_path / "nothing"), standin_call(), LINE, 180000055)[0] == K.PENDING
    assert K.check_pool_seed("a3_mx", str(tmp_path / "nothing"), standin_call(), "", None) == ("PASS", [])


@pytest.mark.parametrize("over,line,why", [
    ({"fork_step": None}, LINE.replace("step filter <= 180000055: DROPPED 2 later snapshot(s) at steps "
                                       "[184000013, 188000021], pruned ['pfsp_win_rates']",
                                       "step filter OFF: the fork checkpoint's num_timesteps is UNREADABLE"),
     "fork_step is NULL"),
    ({"schema": 1}, LINE, "schema 1"),
    ({"fork_step": 180000054}, LINE, "!= P*'s num_timesteps"),
    ({"snapshots": ["snapshot_176000000.zip", "snapshot_184000013.zip"]}, LINE, "later than fork_step"),
    ({}, LINE.replace("DROPPED 2 later snapshot(s) at steps [184000013, 188000021]",
                      "DROPPED 1 later snapshot(s) at steps [184000013]"), "dropped steps"),
    ({}, LINE.replace("<= 180000055", "<= 170000000"), "step filter 170000000"),
])
def test_the_pool_seed_check_fails_every_t54_hazard(tmp_path, over, line, why):
    st, probs = K.check_pool_seed("c0", str(_seeded(tmp_path, **over)), standin_call(), line, 180000055)
    assert st == "FAIL" and any(why in p for p in probs), probs


def test_a_none_later_seed_passes(tmp_path):
    run = _seeded(tmp_path, dropped_steps=[])
    line = "🌱 [SELFPLAY] [pool] seeded 2 snapshots + metadata from x [files: ] [step filter <= 180000055: none later]"
    assert K.check_pool_seed("a2", str(run), standin_call(), line, 180000055) == ("PASS", [])


def test_d1_is_c0_plus_the_team_mix_and_nothing_else():
    """X38 / D1 (§4.2 of the diversity doc): b-ON = C0's argv + the ONE lever, both sides, the same P*, seed, steps."""
    call = standin_call()
    on, off = flags(K.build_argv("d1", 3502, call)), flags(K.build_argv("c0", 3502, call))
    assert set(on) - set(off) == {"--team-mix", "--team-mix-sides"}
    assert on["--team-mix"] == "w_forum_v1@ABC:0.5" and on["--team-mix-sides"] == "both"
    assert {k: v for k, v in on.items() if k not in {"--team-mix", "--team-mix-sides", "--run-name"}} == \
        {k: v for k, v in off.items() if k != "--run-name"}
    assert on["--run-name"] == "rb_x35_d1wide_s3502"
    assert [s for a, s in K.BATCH1 if a == "d1"] == [3501, 3502, 3503]


# ------------------------------------------------------------------------------ X47 (F-LS-1) and C3 (toy K9(b) warn)
def _validated_x47_call() -> Dict:
    """The toy call the 803a13e3 validation built X47 from, recovered from its argv file (the 200M STAND-IN for T*)."""
    t = (VALIDATED / "argv_x47_s3601.txt").read_text().split()
    return toy_call(t[t.index("--model") + 1], int(t[t.index("--steps") + 1]) - K.L_TOY)


def test_x47_is_the_validated_x47_argv_token_for_token_but_for_c3s_warn():
    """The kit builds the line `validate_batch1.sh` validated (checkargs 0, dry-run `grow : DECLARED trunk_layers 2 -> 4`
    at 803a13e3), in the same token ORDER (`--grow` between `--device cuda` and `--seed`); only K9(b) moved (C3)."""
    for seed in (3601, 3602, 3603):
        want = (VALIDATED / f"argv_x47_s{seed}.txt").read_text().split()
        i = want.index("--behaviour-check")
        assert want[i + 1] == "fatal"
        want[i + 1] = "warn"
        assert K.build_argv("toy_x47", seed, _validated_x47_call()) == want


def test_x47_is_toy_c0_plus_the_grow_lever_and_nothing_else():
    call = toy_call()
    on, off = flags(K.build_argv("toy_x47", 3602, call)), flags(K.build_argv("toy_c0", 3602, call))
    assert set(on) - set(off) == {"--grow"} and set(off) <= set(on)
    assert on["--grow"] == "trunk_layers=4"
    assert {k: v for k, v in on.items() if k not in {"--grow", "--run-name"}} == \
        {k: v for k, v in off.items() if k != "--run-name"}
    assert on["--run-name"] == "rb_x35t_x47d4_s3602" and on["--steps"] == str(100081428 + 70_000_000)
    d = K.describe("toy_x47", 3602, call)
    assert d["side"] == "toy" and d["delta_text"] == "+70,000,000 steps" and d["parent_run_dir"] == "models/rb_toy_ent_s4002"
    with pytest.raises(K.KitError, match="not registered"):
        K.build_argv("toy_x47", 3606, call)               # X47 is registered at seeds 3601-3605 (AM-4) only
    for s in (3604, 3605):                                # AM-4: X47's look 2, paired with toy C0 at the same seed
        on, off = flags(K.build_argv("toy_x47", s, call)), flags(K.build_argv("toy_c0", s, call))
        assert set(on) - set(off) == {"--grow"} and on["--seed"] == off["--seed"] == str(s)
    with pytest.raises(K.KitError, match="reads the toy call record"):
        K.build_argv("toy_x47", 3601, standin_call())


def test_c3_the_toy_arms_run_k9b_warn_under_x34s_ceilings_and_production_is_unchanged():
    toys = [n for n, a in K.ARMS.items() if a.side == "toy"]
    assert sorted(toys) == ["toy_a1", "toy_a2", "toy_c0", "toy_x47"]
    for n in toys:
        assert flags(K.build_argv(n, 3601, toy_call()))["--behaviour-check"] == "warn", n
        d = K.describe(n, 3601, toy_call())
        assert d["behaviour"] == "warn"
        assert d["k9b_ceilings"] == {"judged_max": 1e-3, "p99": 1e-4, "non_finite": "STOP", "rule8_rel": 1e-9}
    for n, a in K.ARMS.items():   # production keeps what the kit said before C3: `warn` (X26's regime), no toy ceilings
        if a.side == "production":
            assert a.behaviour == "warn" and K.describe(n, a.seeds[0], standin_call(), _p1(n))["k9b_ceilings"] is None, n


def _p1(arm: str):
    """A phase-1 record for a phase-2 arm (the standin_call's P*), else None."""
    a = K.ARMS[arm]
    if a.phase != 2:
        return None
    return {"arm": a.phase1_of, "pstar_sha256": "shaP",
            "exploiters": [{"path": f"/x/{i}.zip", "label": f"e{i}"} for i in range(3)]}


def test_batch1_keeps_c2s_x47_interleave_and_the_toy_calibration_last():
    """C2 (orchestrator, 2026-10-19) inside AM-7's order (the whole list: ``batch_test.test_batch1_is_am7s_order``):
    X47 beside its toy C0 in counterbalanced pairs, at look 1 (ABA) and look 2 (BA, AB), and the toy calibration
    (toy A1 / toy A2, AM-3's conditional stage) last."""
    pos = {(a, s): i for i, (a, s) in enumerate(K.BATCH1)}
    first = []
    for s in (3601, 3602, 3603, 3604, 3605):
        assert abs(pos["toy_c0", s] - pos["toy_x47", s]) == 1, s    # X47 runs right beside its own C0
        first.append("toy_c0" if pos["toy_c0", s] < pos["toy_x47", s] else "toy_x47")
    assert first == ["toy_c0", "toy_x47", "toy_c0", "toy_x47", "toy_c0"]   # box drift lands on neither side
    assert {a for a, _ in K.BATCH1[-6:]} == {"toy_a1", "toy_a2"}
    assert max(pos[a, s] for a, s in K.BATCH1 if not a.startswith("toy_")) < min(
        pos[a, s] for a, s in K.BATCH1 if a.startswith("toy_"))           # production first


def test_a_grown_pool_seed_record_passes_the_check(tmp_path):
    """X47's seeded pool is GROWN at the child's start, which adds a `grown` block to pool_seed.json; T54's check
    (run on the toy_x47 arm after the spawn) must read that record as it reads any other."""
    run = _seeded(tmp_path, grown={"fields": {"trunk_layers": {"parent": 2, "child": 4}}, "snapshots": ["a.zip"],
                                   "skipped": []})
    assert K.check_pool_seed("toy_x47", str(run), standin_call(), LINE, 180000055) == ("PASS", [])


# ----------------------------------------------------- validate_batch1.sh's verdict on one argv's transcripts
def _transcripts(arm_file: str):
    return ((VALIDATED / f"dryrun_{arm_file}.txt").read_text(), (VALIDATED / f"checkargs_{arm_file}.txt").read_text())


def _fatal_era(d: Dict) -> Dict:
    """The 803a13e3 transcripts were taken BEFORE C3, so they read `K9(b) check : fatal`; the verdict compares the arm's
    mode with what the transcript printed, so these cases hand it the mode those transcripts were made under."""
    return dict(d, behaviour="fatal")


def test_the_verdict_passes_a_real_x47_transcript_and_shows_its_declared_growth():
    dry, chk = _transcripts("x47_s3601")
    ok, line = K.check_validation(_fatal_era(K.describe("toy_x47", 3601, _validated_x47_call())), dry, chk, 0, 0)
    assert ok, line
    assert "verdict=PASS" in line and "grow=[DECLARED trunk_layers 2 -> 4]" in line and "delta=[+70,000,000 steps]" in line
    assert "gib=[13.05]" in line and "would_launch=1" in line


def test_the_verdict_passes_a_real_toy_c0_transcript_with_no_growth_line():
    dry, chk = _transcripts("toy_c0_s3601")
    ok, line = K.check_validation(_fatal_era(K.describe("toy_c0", 3601, _validated_x47_call())), dry, chk, 0, 0)
    assert ok and "grow=[none]" in line, line      # the always-printed "(child-only: a declared growth's ...)" is no growth line


@pytest.mark.parametrize("what,arm,dry_edit,chk_edit,rcs,why", [
    ("x47 lost its growth line in the dry-run", "toy_x47", ("grow        : DECLARED", "grow        : (no)"), None, (0, 0),
     "dry-run declared growth None"),
    ("x47 lost its growth line in checkargs", "toy_x47", None, ("declared growth (--grow)", "declared growth (--no)"),
     (0, 0), "checkargs declared growth None"),
    ("a checkargs refusal", "toy_x47", None, None, (2, 0), "checkargs exit 2"),
    ("a dry-run refusal", "toy_x47", None, None, (0, 2), "dry-run exit 2"),
    ("no 'would launch'", "toy_x47", ("this command would launch", "this command would NOT launch"), None, (0, 0),
     "would launch"),
    ("the wrong step arithmetic", "toy_x47", ("+70,000,000 steps", "+20,000,000 steps"), None, (0, 0), "step arithmetic"),
    ("the wrong parent", "toy_x47", ("FORK of /", "FORK of /nowhere/"), None, (0, 0), "role is not FORK of"),
    ("a K9(b) mode that is not the arm's", "toy_x47", ("K9(b) check : fatal", "K9(b) check : off"), None, (0, 0),
     "dry-run K9(b) mode 'off'"),
    ("the trainer's arch check missing", "toy_x47", ("✓ fork arch", "? fork arch"), None, (0, 0), "fork arch"),
])
def test_the_verdict_fails_each_hazard(what, arm, dry_edit, chk_edit, rcs, why):
    dry, chk = _transcripts("x47_s3601")
    if dry_edit:
        assert dry_edit[0] in dry, what
        dry = dry.replace(*dry_edit)
    if chk_edit:
        assert chk_edit[0] in chk, what
        chk = chk.replace(*chk_edit)
    ok, line = K.check_validation(_fatal_era(K.describe(arm, 3601, _validated_x47_call())), dry, chk, *rcs)
    assert not ok and "verdict=FAIL" in line and why in line, (what, line)


def test_the_verdict_fails_a_growth_line_on_an_arm_that_declared_none_and_a_c3_mode_mismatch():
    dry, chk = _transcripts("x47_s3601")           # X47's transcripts read against the toy C0 arm: an unexpected growth
    ok, line = K.check_validation(_fatal_era(K.describe("toy_c0", 3601, _validated_x47_call())), dry, chk, 0, 0)
    assert not ok and "declared growth 'trunk_layers 2 -> 4' != None" in line
    ok, line = K.check_validation(K.describe("toy_x47", 3601, _validated_x47_call()), dry, chk, 0, 0)   # C3: warn
    assert not ok and "K9(b) mode 'fatal' != the arm's 'warn'" in line      # a pre-C3 (fatal) transcript no longer passes
