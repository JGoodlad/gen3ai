"""X35-A3 (= LG-1 = CE-3c, PREREG ``x35_a3_main_exploiters_2026-11-16``): the league's first arm with main exploiters
folded back in, as TWO kit arms at the production side pin -- ``a3_mx`` (three 5M main exploiters of P*) and ``a3``
(P* +20M with those three as 40 % of its self-play, hardest first). This file holds the registration's stage gate
(``batch._a3_status``: the registered seeds only, the PREREG, the phase-1 record, never a stand-in record), the two
argvs, and the committed registration files (argvs, transcripts, the draft chain items, the recorded kit checks and
plan, the rule).

The real-pin test reads this checkout's object store and skips where the pin was not fetched. Nothing here launches,
plays or touches ``models/``; the run directories and the records are synthetic.
"""
from __future__ import annotations

import json
import runpy
import subprocess
from pathlib import Path

import pytest

from main.x35 import batch as B
from main.x35 import kit as K
from main.x35.kit_test import STAGE, standin_call, toy_call
from utils.paths import repo_path, repo_root

DIR = repo_path("designs/research_state/measurements/x35_a3_main_exploiters_2026-11-16")
SIDE_PIN = "f4b25238def57900c0305bb2a3a5a8f9edaef285"
MX, A3 = "a3_mx", "a3"
MX_SEEDS = (3511, 3512, 3513)
A3_SEEDS = (3501, 3502, 3503)
#: The kit's own phase-2 seeds this PREREG does NOT register (one look, no look 2).
A3_UNREGISTERED = (3504, 3505)
MX_RUNS = [f"rb_x35_mx_s{s}" for s in MX_SEEDS]
#: The two banners as the pin's launcher events print them (PREREG §8); a stable opponent's label carries ``ext_``.
MX_BANNER = ("[EXPLOITER] training vs ext_rb_x26_s2601 as the SOLE opponent every episode "
             "(temp 1; no self-play/pool/bots)")
A3_BANNER = ("[STABLE] 3 cross-run opponent(s): ext_mx_s3511, ext_mx_s3512, ext_mx_s3513 — eval greedy; "
             "training ≤40% of self-play until mastered (win_rate ≥ 80%)")
FOLD = ("--stable-opponent-selfplay-share", "0.4", "--stable-opponent-pfsp")


def _real_call() -> dict:
    """The frozen production R's call record (P* = rb_x26_s2601 @280,000,170), as committed."""
    return json.loads((STAGE / "reference_population_production.json").read_text())


def _standin_record() -> dict:
    """The committed STAND-IN phase-1 record (three banked R0 checkpoints): validation only, never a launch value."""
    return json.loads((DIR / "phase1_a3_STANDIN.json").read_text())


def _visible(dry: str) -> str:
    """The registration's dry run ran with the card hidden at nice 19 (rule 5, 5b): its two refusals are the CUDA
    and the inherited-nice preflights. The chain re-runs it visible; model that one verdict line only."""
    return dry.replace("this command would NOT launch", "this command would launch")


def _has(sha: str) -> bool:
    return subprocess.run(["git", "-C", str(repo_root()), "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True).returncode == 0


needs_pin = pytest.mark.skipif(not _has(SIDE_PIN), reason="the production side pin is not in this object store")


def _fill_fp32(stage: Path, sha: str = "c" * 40) -> None:
    (stage / B.SIDE_REQUIRED_FILE.format(side="production")).write_text(f"# header\n\n{sha} FP32_ALT the fp32 build\n")


def _stage(tmp_path: Path, monkeypatch, prereg: bool = True) -> str:
    """A scratch stage with the AM-8 side gate filled and (by default) the A3 PREREG present."""
    stage = tmp_path / "stage"
    (stage / B.READS_DIR).mkdir(parents=True)
    _fill_fp32(stage)
    path = tmp_path / "a3" / "PREREG.md"
    monkeypatch.setattr(B, "A3_PREREG", str(path))                  # absolute: repo_path() returns it unchanged
    if prereg:
        path.parent.mkdir()
        path.write_text("# X35-A3\n")
    return str(stage)


def _real_record(tmp_path: Path, monkeypatch, out: Path) -> dict:
    """The record ``kit phase1-record`` writes for three (synthetic) finished exploiter runs."""
    md = tmp_path / "models"
    for s in MX_SEEDS:
        d = md / K.run_name(MX, s)
        d.mkdir(parents=True)
        (d / "final_model.zip").write_bytes(f"{MX}{s}".encode())
    monkeypatch.setattr(K, "_final_step", lambda z: 185000055)
    return K.phase1_record(MX, standin_call(), str(md), str(out))


# ------------------------------------------------------------------------------------------ the two kit arms
@pytest.mark.parametrize("seed", MX_SEEDS)
def test_the_kit_builds_the_validated_phase_1_argv_token_for_token(seed):
    """The kit builds exactly the line checkargs / the dry run validated at the side pin (PREREG §8)."""
    want = (DIR / f"argv_{MX}_s{seed}.txt").read_text().split()
    assert K.build_argv(MX, seed, _real_call()) == want
    pstar = want[want.index("--model") + 1]
    assert want[want.index("--exploiter") + 1] == pstar and pstar.endswith("checkpoint_280000170_steps.zip")
    assert "--no-self-play" in want and want[want.index("--steps") + 1] == "285000170"
    assert not [t for t in want if t.startswith("--stable") or t.startswith("--trainee-teams")]


@pytest.mark.parametrize("seed", A3_SEEDS)
def test_the_kit_builds_the_validated_phase_2_argv_from_the_committed_stand_in_record(seed):
    """Phase 2's argv is a function of the phase-1 record; the transcripts were taken on the committed STAND-IN
    record (the finals do not exist), and the kit rebuilds that line token for token from it."""
    want = (DIR / f"argv_{A3}_STANDIN_s{seed}.txt").read_text().split()
    rec = _standin_record()
    assert K.build_argv(A3, seed, _real_call(), rec) == want
    opp = want[want.index("--stable-opponents") + 1].split(",")
    assert opp == [f"{e['path']}:{e['label']}" for e in rec["exploiters"]] and len(opp) == 3
    i = want.index("--stable-opponent-selfplay-share")
    assert tuple(want[i:i + 3]) == FOLD and want[want.index("--steps") + 1] == "300000170"
    assert "--exploiter" not in want and "--no-self-play" not in want


def test_phase_2_is_c0_plus_the_three_exploiters_at_40_percent_hardest_first(tmp_path, monkeypatch):
    """With the REAL record's shape: C0's argv token for token + the three finals as stable opponents, share 0.4,
    PFSP; only the seed and the run name differ. The labels are what the registered banner prints."""
    call = standin_call()
    rec = _real_record(tmp_path, monkeypatch, tmp_path / "p1.json")
    assert [e["run"] for e in rec["exploiters"]] == MX_RUNS and rec["arm"] == MX
    assert [e["label"] for e in rec["exploiters"]] == ["mx_s3511", "mx_s3512", "mx_s3513"]
    for seed in A3_SEEDS:
        off, on = K.build_argv("c0", seed, call), K.build_argv(A3, seed, call, rec)
        cut = off.index("--seed")
        lever = on[cut:-4]
        assert on[:cut] == off[:cut] and lever[0] == "--stable-opponents" and tuple(lever[2:]) == FOLD
        assert lever[1] == ",".join(f"{e['path']}:{e['label']}" for e in rec["exploiters"])
        assert on[-4:] == ["--seed", str(seed), "--run-name", f"rb_x35_a3mx_s{seed}"]
    for e in rec["exploiters"]:
        assert f"ext_{e['label']}" in A3_BANNER


def test_phase_2_needs_the_record_and_neither_arm_takes_the_toy_call():
    with pytest.raises(K.KitError, match="needs the a3_mx phase-1 record"):
        K.build_argv(A3, 3501, standin_call())
    for arm, seed in ((MX, 3511), (A3, 3501)):
        with pytest.raises(K.KitError):
            K.build_argv(arm, seed, toy_call(), _standin_record())


def test_the_seeds_are_the_registered_ones():
    """Phase 1's three seeds are NEW (no other arm's: a seed names a run and a pool RNG stream); phase 2's are
    deliberately C0's first three, so each A3 fork has a same-seed C0 sibling."""
    assert K.ARMS[MX].seeds == MX_SEEDS and K.ARMS[A3].seeds == A3_SEEDS + A3_UNREGISTERED
    assert K.ARMS[MX].length == 5_000_000 and K.ARMS[A3].length == 20_000_000 and K.ARMS[A3].phase1_of == MX
    others = {s for name, arm in K.ARMS.items() if name != MX for s in arm.seeds}
    assert not others & set(MX_SEEDS), others & set(MX_SEEDS)
    assert set(A3_SEEDS) <= set(K.ARMS["c0"].seeds)
    assert [(n, forks) for n, forks in B.A3_STAGES] == [
        ("A3_MX", tuple((MX, s) for s in MX_SEEDS)), ("A3", tuple((A3, s) for s in A3_SEEDS))]
    assert B.A3_ARMS == (MX, A3) and B.A3_PHASE1_FILE == "phase1_a3.json"


# ------------------------------------------------------------------------------------------ the stage gate
def test_both_arms_are_side_pinned_and_wait_for_the_side_gate_and_the_prereg(tmp_path, monkeypatch):
    """Before this registration both arms read ``ALLOWED ... (not a staged arm)`` at any kit seed and no pin-check
    judged their runs."""
    assert MX in B.SIDE_PINNED_ARMS and A3 in B.SIDE_PINNED_ARMS
    assert MX not in B.ARM_PINS and A3 not in B.ARM_PINS            # the side pin, no arm pin of their own
    names = B.side_pinned_run_names()
    assert {n for n, (a, _) in names.items() if a == MX} == set(MX_RUNS)
    assert {f"rb_x35_a3mx_s{s}" for s in A3_SEEDS} <= {n for n, (a, _) in names.items() if a == A3}
    bare = tmp_path / "bare"
    (bare / B.READS_DIR).mkdir(parents=True)
    monkeypatch.setattr(B, "A3_PREREG", str(tmp_path / "nowhere" / "PREREG.md"))
    for arm, seed in ((MX, 3511), (A3, 3501)):
        status, why = B.stage_status(str(bare), arm, seed)
        assert status == B.PENDING and "PREREG" not in why, why    # AM-8: no FP32_ALT named
    _fill_fp32(bare)
    for arm, seed in ((MX, 3511), (A3, 3501)):
        status, why = B.stage_status(str(bare), arm, seed)
        assert status == B.PENDING and "PREREG" in why and "not in this checkout" in why, why


def test_phase_1_is_allowed_at_its_three_seeds_with_the_prereg(tmp_path, monkeypatch):
    st = _stage(tmp_path, monkeypatch)
    for seed in MX_SEEDS:
        status, why = B.stage_status(st, MX, seed)
        assert status == B.ALLOWED and "stage A3_MX, A3 PREREG present" in why, why


@pytest.mark.parametrize("arm,seed", [(A3, 3504), (A3, 3505), (MX, 3501), (MX, 3514), (A3, 3511)])
def test_an_unregistered_seed_is_refused_even_with_everything_else_in_place(tmp_path, monkeypatch, arm, seed):
    """One look, no look 2: the kit's own a3 seeds 3504 / 3505 are not this PREREG's, with or without a record."""
    st = _stage(tmp_path, monkeypatch)
    _real_record(tmp_path, monkeypatch, Path(st) / B.A3_PHASE1_FILE)
    status, why = B.stage_status(st, arm, seed)
    assert status == B.REFUSED and "not registered by the A3 PREREG" in why and "no look 2" in why, why


def test_phase_2_waits_for_the_record_and_refuses_a_stand_in(tmp_path, monkeypatch):
    """The phase boundary in code: PENDING with no record; REFUSED on the stand-in record (which HASHES, so
    ``verify-phase1`` alone passes it), on a record of another arm, of two runs or of garbage; ALLOWED only on a
    record naming the three registered exploiter runs."""
    st = _stage(tmp_path, monkeypatch)
    record = Path(st) / B.A3_PHASE1_FILE
    for seed in A3_SEEDS:
        status, why = B.stage_status(st, A3, seed)
        assert status == B.PENDING and "waits for the phase-1 record phase1_a3.json" in why, why
    record.write_text(json.dumps(_standin_record()))
    for seed in A3_SEEDS:
        status, why = B.stage_status(st, A3, seed)
        assert status == B.REFUSED and "a stand-in record never launches phase 2" in why and "STANDIN" in why, why
    assert B.stage_status(st, MX, 3511)[0] == B.ALLOWED            # phase 1 does not read the record
    real = _real_record(tmp_path, monkeypatch, tmp_path / "p1.json")
    for bad in (dict(real, arm="r0"), dict(real, exploiters=real["exploiters"][:2]),
                dict(real, exploiters=list(reversed(real["exploiters"])))):
        record.write_text(json.dumps(bad))
        status, why = B.stage_status(st, A3, 3501)
        assert status == B.REFUSED and "not the three registered exploiters" in why, why
    for garbage in ("{not json", json.dumps({"arm": MX}), json.dumps([1, 2])):
        record.write_text(garbage)
        status, why = B.stage_status(st, A3, 3501)
        assert status == B.REFUSED and "is unreadable" in why, why
    record.write_text(json.dumps(real))
    for seed in A3_SEEDS:
        status, why = B.stage_status(st, A3, seed)
        assert status == B.ALLOWED and "stage A3, A3 PREREG present" in why, why


def test_the_real_stage_folder_holds_no_phase_1_record():
    """No exploiter has trained: a ``phase1_a3.json`` in the committed stage folder would be a fabricated record."""
    assert not (STAGE / B.A3_PHASE1_FILE).exists()
    assert B.stage_status(str(STAGE), A3, 3501)[0] in (B.PENDING, B.REFUSED)


# ------------------------------------------------------------------------------------------ the validation verdict
@pytest.mark.parametrize("arm,seed,stem", [(MX, s, f"{MX}_s{s}") for s in MX_SEEDS]
                         + [(A3, s, f"{A3}_STANDIN_s{s}") for s in A3_SEEDS])
def test_the_verdict_passes_the_committed_transcripts(arm, seed, stem):
    dry, chk = (DIR / f"dryrun_{stem}.txt").read_text(), (DIR / f"checkargs_{stem}.txt").read_text()
    assert dry.count("✗ REFUSED") == 2 and "CUDA_VISIBLE_DEVICES=''" in dry and "nice 19" in dry    # the two expected
    assert f"pin         : {SIDE_PIN}" in dry
    d = K.describe(arm, seed, _real_call(), _standin_record() if arm == A3 else None)
    ok, line = K.check_validation(d, _visible(dry), chk, 0, 0)
    assert ok, line
    assert f"delta=[{'+5,000,000' if arm == MX else '+20,000,000'} steps]" in line and "grow=[none]" in line
    ok, line = K.check_validation(d, dry, chk, 0, 3)                # the hidden card's exit 3 is not a pass
    assert not ok and "dry-run exit 3" in line, line


def test_neither_arms_transcripts_pass_as_the_others():
    call, rec = _real_call(), _standin_record()
    mdry, mchk = (DIR / f"dryrun_{MX}_s3511.txt").read_text(), (DIR / f"checkargs_{MX}_s3511.txt").read_text()
    adry, achk = (DIR / f"dryrun_{A3}_STANDIN_s3501.txt").read_text(), \
        (DIR / f"checkargs_{A3}_STANDIN_s3501.txt").read_text()
    ok, line = K.check_validation(K.describe(A3, 3501, call, rec), _visible(mdry), mchk, 0, 0)
    assert not ok, line
    ok, line = K.check_validation(K.describe(MX, 3511, call), _visible(adry), achk, 0, 0)
    assert not ok, line


# ------------------------------------------------------------------------------------------ the draft chain items
def test_the_draft_chain_items_are_the_kits_runs_at_the_side_pin_with_the_registered_banners():
    """``chain_items_a3.json`` (the Training session's hand-off): three phase-1 items, then three phase-2 items whose
    verify step re-hashes the phase-1 record; each arm's banner is the other's VOID line."""
    doc = json.loads((DIR / "chain_items_a3.json").read_text())
    assert doc["x_pin"]["sha"] == SIDE_PIN
    assert "GO" not in doc and "go" not in doc and "NO GO" in doc["x_what"]
    assert len(doc["x_phase_boundary"]["steps"]) == 4 and "§5.3" in doc["x_phase_boundary"]["when"]
    items = doc["items"]
    assert [(i["arm"], i["seed"]) for i in items] == [(MX, s) for s in MX_SEEDS] + [(A3, s) for s in A3_SEEDS]
    for it in items:
        assert it["kind"] == "arm" and it["pin"] == SIDE_PIN and it["run"] == K.run_name(it["arm"], it["seed"])
        assert (it["arm"], it["seed"]) in [f for _, forks in B.A3_STAGES for f in forks]
        assert it["thin"] == "planner" and it["flags"] == ["--thinning"]
        assert it["on_void"] == "stop" and it["on_throughput_fail"] == "page"
        assert it["throughput"] == {"min_steps_per_s": 1000, "within_min": 60, "first_line_within_min": 90}
        assert it["go_by"].startswith("<orchestrator, date>")       # the GO is the orchestrator's to fill
        b = it["banner"]
        assert b["within_min"] == 30 and not any(v in b["expect"] for v in b["void"])
        if it["arm"] == MX:
            assert it["disk_need_gib"] == 12.0 and "verify" not in it        # the dry run's 11.22, rounded up
            assert b["expect"] == MX_BANNER and "[STABLE]" in b["void"]
            assert any(v in A3_BANNER for v in b["void"])           # a fold-in fork's banner VOIDs an exploiter
        else:
            assert it["disk_need_gib"] == 15.0                      # the dry run's 14.28, rounded up
            assert b["expect"] == A3_BANNER and "[EXPLOITER]" in b["void"]
            assert any(v in MX_BANNER for v in b["void"])
            assert [v[1:4] for v in it["verify"]] == [["-m", "main.x35.kit", "verify-phase1"]]
            assert it["verify"][0][4].endswith("x35_breakers_stage_2026-10-17/phase1_a3.json")
        assert "[ModelVersion] FATAL" in b["void"]


@needs_pin
def test_the_banners_are_the_pins_format_strings():
    """The two banner strings are the pin's launcher events with the registered values filled in (PREREG §8)."""
    setup = subprocess.run(["git", "-C", str(repo_root()), "show", f"{SIDE_PIN}:src/main/train/matchup_setup.py"],
                           check=True, capture_output=True, text=True).stdout
    assert "[STABLE] {len(_fixed_opponents)} cross-run opponent(s): {_stable_labels} — " in setup
    assert "f\"eval greedy; training ≤{args.stable_opponent_selfplay_share:.0%} of self-play until \"" in setup
    assert "[EXPLOITER] training vs {_exploiter_entry.label} as the SOLE opponent every episode " in setup
    assert f"{0.4:.0%}" == "40%" and "≤40% of self-play until mastered" in A3_BANNER


# ------------------------------------------------------------------------------------------ the recorded checks
def test_the_recorded_kit_checks_and_plan_say_what_the_prereg_says():
    checks = (DIR / "kit_checks_a3.txt").read_text()
    for seed in MX_SEEDS:
        assert f"ALLOWED: a3_mx s{seed}: no unmet condition (stage A3_MX, A3 PREREG present)" in checks
    for seed in A3_SEEDS:
        assert f"PENDING: a3 s{seed}: phase 2 waits for the phase-1 record phase1_a3.json" in checks
    for seed in A3_UNREGISTERED:
        assert f"REFUSED: a3 s{seed} is not registered by the A3 PREREG" in checks
    assert "REFUSED: seed 3514 is not registered for a3_mx ([3511, 3512, 3513])" in checks
    assert "REFUSED: a3 s3501: the phase-1 record phase1_a3.json names ['STANDIN" in checks
    assert "ALLOWED: a3 s" not in checks                           # no phase-2 fork was ever allowed
    assert checks.count(f"the pin equals X35_PIN_1_production {SIDE_PIN[:12]}") == 2 and "!= X35_PIN_1_pro" in checks
    plan = (DIR / "arm_chain_plan_a3.txt").read_text()
    for run in MX_RUNS:
        assert plan.count(f"arm:{run}: the kit agrees — FORK of") >= 1 and "+5,000,000 steps, --steps 285,000,170" in plan
    assert plan.index("\nPLAN OK\nrc=0\n") < plan.index("PLAN REFUSED")        # phase 1 plans first, and passes
    assert "REFUSED: seed 3514 is not registered for a3_mx ([3511, 3512, 3513])" in plan
    assert plan.count("arm a3 is phase 2: it needs the a3_mx phase-1 record") == 3     # §11.2: --describe, phase 2
    assert "PLAN REFUSED (3 problem(s))" in plan and "the kit agrees — FORK of" not in plan[plan.index("PHASE-2"):]


def test_the_plans_refusal_of_phase_2_is_the_kits_describe_without_a_record():
    """PREREG §11.2 (a kit gap, reported): ``arm_chain plan --describe`` asks ``kit describe`` with no ``--phase1``,
    and ``describe`` has no default; this pins the behaviour the recorded plan shows, so a fix updates both."""
    with pytest.raises(K.KitError, match="arm a3 is phase 2: it needs the a3_mx phase-1 record"):
        K.describe(A3, 3501, _real_call())


def test_the_stand_in_record_hashes_like_a_real_one_so_the_stage_gate_is_what_refuses_it():
    rec = _standin_record()
    assert rec["arm"] == MX and rec["schema"] == "x35_phase1_exploiters_v1" and "VALIDATION ONLY" in rec["x_STANDIN"]
    assert all("STANDIN" in e["run"] and "rb_x35_r0_s3531" in e["path"] for e in rec["exploiters"])
    assert not {e["run"] for e in rec["exploiters"]} & set(MX_RUNS)


# ------------------------------------------------------------------------------------------ the PREREG and the rule
def test_the_committed_prereg_is_where_the_batch_looks_and_names_the_arms_seeds_and_classes():
    assert repo_path(B.A3_PREREG).is_file(), B.A3_PREREG
    assert repo_path(B.A3_PREREG).parent == DIR
    text = repo_path(B.A3_PREREG).read_text()
    assert SIDE_PIN in text and "there is no GO marker" in text
    for word in ("ROBUSTNESS-GAIN", "TRADE", "NO-FALL", "INCONCLUSIVE", "EXPLOITS", "UNVERIFIED"):
        assert word in text, word
    assert all(str(s) in text for s in MX_SEEDS + A3_SEEDS + A3_UNREGISTERED)
    for name in ("a3_rule.py", "selfcheck.log", "power.log", "power_designs.log", "power_guards.log",
                 "power_gate.log", "chain_items_a3.json", "kit_checks_a3.txt", "arm_chain_plan_a3.txt",
                 "dose_c0.txt", "phase1_a3_STANDIN.json"):
        assert (DIR / name).is_file() and name in text, name
    assert not [p for p in DIR.rglob("*") if p.name.upper().startswith("GO")]   # registered, not launched
    assert not [p for p in DIR.rglob("phase1_a3.json")]             # only the stand-in, under its own name


@pytest.fixture(scope="module")
def rule():
    return runpy.run_path(str(DIR / "a3_rule.py"))


def test_the_rules_self_check_passes_and_is_the_committed_log(rule):
    res = rule["self_check"]()
    assert res["self_check"] == "OK" and res["problems"] == [], res["problems"]
    assert json.dumps(res, indent=1) + "\n" == (DIR / "selfcheck.log").read_text()


def test_the_phase_boundary_gate_is_one_sided_at_90_percent_with_the_binomial_floor(rule):
    """PREREG §5.3: three exploiter finals x P*, 1,000 mirrored pairs each; EXPLOITS iff the 90 % interval's lower
    end on the mean edge is strictly above 0. Three identical scores do not shrink the interval to nothing."""
    gate = rule["phase1_gate"]
    assert rule["GATE_PAIRS"] == 1000 and rule["GATE_LEVEL"] == 0.90
    tight = gate([54.0, 55.0, 56.0])
    assert tight["exploits"] and tight["df"] == 2 and tight["mean_edge_pp"] == pytest.approx(5.0)
    same = gate([53.0, 53.0, 53.0])
    assert same["floor_bound"] and same["se_pp"] > 0.5 and same["interval90_pp"][0] < 3.0
    assert not gate([50.0, 55.0, 60.0])["exploits"]                # a wide spread: the mean edge is not resolved
    assert not gate([50.0, 50.0, 50.0])["exploits"] and not gate([49.0, 50.0, 51.0])["exploits"]
    assert not gate([44.0, 45.0, 46.0])["exploits"]                # an exploiter that LOSES never passes
    for bad in ([55.0, 55.0], [55.0, 55.0, 55.0, 55.0], [55.0, 55.0, float("nan")], [55.0, 55.0, 101.0]):
        with pytest.raises(ValueError):
            gate(bad)
