"""X70 (PREREG ``x70_ent0_control_2026-11-13``): the ent-0 control of the trailing-magnet screen, ONE kit arm ``x70``
(C0 + ``--ent-coef 0``: the swap's argv minus ``--magnet-kl-coef 0.05``) at the swap's own training pin
``batch.CE3D_PIN`` -- its argv, seeds, stage and pin check, and the committed registration files the kit, the arm
chain and the cross read (argvs, transcripts, the draft chain items, the cross plan's constants, the rule).

The real-pin tests read this checkout's object store (tag ``ce3d-pin-9e97cad5`` on origin) and skip where it was not
fetched. Nothing here launches, plays or touches ``models/``; the run directories are synthetic.
"""
from __future__ import annotations

import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from main.x35 import batch as B
from main.x35 import kit as K
from main.x35.kit_test import STAGE, standin_call, toy_call
from utils.paths import repo_path, repo_root

DIR = repo_path("designs/research_state/measurements/x70_ent0_control_2026-11-13")
CE3D_DIR = repo_path("designs/research_state/measurements/ce3d_magnet_screen_2026-11-11")
SIDE_PIN = "f4b25238def57900c0305bb2a3a5a8f9edaef285"
ARM = "x70"
SEEDS = (7331, 7332, 7333, 7334)
LEVER = ("--ent-coef", "0")
STEM = "rb_x35_x70ent0_s"
SWAP_SEEDS = (7321, 7322, 7323)
#: The TYPED row as both tools print it (the argv's ``0`` prints as ``0.0``).
TYPED_ENT = "ent_coef                 0.0        (TYPED)"
#: The hyperparameter table's row at an entropy coefficient of 0, as the swap forks' logs print it (PREREG §7).
ROW_AT_ZERO = "|    ent_coef   | 0        |"
#: The same table's rows at the three coefficients a mistyped or inherited launch could carry.
ROWS_ELSEWHERE = ("|    ent_coef   | 0.05     |", "|    ent_coef   | 0.0125   |", "|    ent_coef   | 0.1      |")
BANNER = {"expect": "ent_coef   | 0 ",
          "void": ["ent_coef   | 0.05 ", "ent_coef   | 0.1 ", "ent_coef   | 0.0125 ", "[MAGNET] --magnet-kl-coef",
                   "[ModelVersion] FATAL"],
          "within_min": 30}


def _real_call() -> dict:
    """The frozen production R's call record (P* = rb_x26_s2601 @280,000,170), as committed."""
    return json.loads((STAGE / "reference_population_production.json").read_text())


def _transcripts(seed: int, arm: str = ARM, where: Path = DIR):
    stem = f"{arm}_s{seed}"
    return (where / f"dryrun_{stem}.txt").read_text(), (where / f"checkargs_{stem}.txt").read_text()


def _visible(dry: str) -> str:
    """The registration's dry run ran with the card hidden at nice 19 (rule 5, 5b): its two refusals are the CUDA
    and the inherited-nice preflights. The chain re-runs it visible; model that one verdict line only."""
    return dry.replace("this command would NOT launch", "this command would launch")


def _has(sha: str) -> bool:
    return subprocess.run(["git", "-C", str(repo_root()), "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True).returncode == 0


needs_pin = pytest.mark.skipif(not _has(B.CE3D_PIN), reason="CE3D_PIN is not in this object store (fetch the tag)")


# ------------------------------------------------------------------------------------------ the kit arm
@pytest.mark.parametrize("seed", SEEDS)
def test_the_kit_builds_the_validated_argv_token_for_token(seed):
    """The kit builds exactly the line checkargs / the dry run validated at CE3D_PIN (PREREG §7)."""
    want = (DIR / f"argv_{ARM}_s{seed}.txt").read_text().split()
    assert K.build_argv(ARM, seed, _real_call()) == want


def test_the_arm_is_c0_plus_one_flag_and_the_swap_minus_the_magnet():
    """C0's argv token for token + ``--ent-coef 0`` in the lever slot; the swap's with the magnet's two tokens
    removed. Only the seed and the run name differ from either."""
    assert K.X70_LEVER == LEVER == K.CE3D_SWAP_LEVER[:2] and K.X70_SEEDS == SEEDS
    assert K.CE3D_SWAP_LEVER[2:] == ("--magnet-kl-coef", "0.05")
    call = standin_call()
    off = K.build_argv("c0", 3501, call)
    swap = K.build_argv("ce3d_swap", 7321, call)
    cut = off.index("--seed")
    for seed in SEEDS:
        on = K.build_argv(ARM, seed, call)
        assert on[:-2] == off[:cut] + list(LEVER) + ["--seed", str(seed)]
        assert on[-2:] == ["--run-name", f"{STEM}{seed}"] and len(on) == len(off) + 2
        m = swap.index("--magnet-kl-coef")
        assert on[:cut + 2] == swap[:m] and swap[m:m + 2] == ["--magnet-kl-coef", "0.05"]
        assert swap[m + 2] == "--seed" and len(swap) == len(on) + 2
        assert not [t for t in on if "magnet" in t]                 # no magnet flag of any spelling
    assert K.ARMS[ARM].seeds == SEEDS and ARM not in K.ARM_GROWTH and ARM not in K.ARM_HUMAN_DATA
    d = K.describe(ARM, SEEDS[1], call)
    assert d["side"] == "production" and d["delta_text"] == "+20,000,000 steps" and d["behaviour"] == "warn"
    assert d["parent_run_dir"].endswith("rb_x26_s2601") and d["k9b_ceilings"] is None


def test_the_arm_declares_one_typed_knob_with_the_printed_spelling():
    assert K.ARM_TYPED_RECIPE[ARM] == (("ent_coef", "0.0", "--ent-coef"),)
    argv = K.build_argv(ARM, 7331, standin_call())
    assert argv[argv.index("--ent-coef") + 1] == "0" and argv.count("--ent-coef") == 1


def test_the_seeds_are_disjoint_from_every_other_arm_and_from_the_retired_six():
    """A seed names a run and a pool RNG stream; X70's four are no sibling's and none of CE-3d's retired six."""
    others = {s for name, arm in K.ARMS.items() if name != ARM for s in arm.seeds}
    assert not others & set(SEEDS), others & set(SEEDS)
    assert not set(SEEDS) & set(B.CE3D_RETIRED_SEEDS)


@pytest.mark.parametrize("seed", [7330, 7335, 7321, 7301, 3501, 6701])
def test_the_arm_refuses_every_seed_but_its_four(seed):
    with pytest.raises(K.KitError, match="not registered"):
        K.build_argv(ARM, seed, standin_call())


def test_the_arm_refuses_the_toy_call_and_no_alias_reaches_the_kit(capsys):
    with pytest.raises(K.KitError, match="reads the production call record"):
        K.build_argv(ARM, SEEDS[0], toy_call())
    for name in ("X70", "x70ent0", "toy_x70", "x70b", "ent0"):
        assert K.canonical_arm(name) == name
        with pytest.raises(K.KitError, match="is not one of"):
            K.build_argv(name, 7331, standin_call())
        with pytest.raises(SystemExit):
            K.main(["describe", "--arm", name, "--seed", "7331", "--call", "/nonexistent"])
        capsys.readouterr()


# ------------------------------------------------------------------------------------------ the validation verdict
@pytest.mark.parametrize("seed", SEEDS)
def test_the_verdict_passes_the_real_transcripts_and_shows_the_one_typed_lever(seed):
    dry, chk = _transcripts(seed)
    assert dry.count("✗ REFUSED") == 2 and "CUDA_VISIBLE_DEVICES=''" in dry and "nice 19" in dry    # the two expected
    assert f"pin         : {B.CE3D_PIN}" in dry and f"the PINNED commit {B.CE3D_PIN[:8]}" in chk
    assert TYPED_ENT in dry and TYPED_ENT in chk
    assert "magnet_kl_coef" not in chk and "(TYPED)" not in "".join(
        ln for ln in dry.splitlines() if "magnet" in ln)            # no typed magnet row in either tool
    assert "--magnet" not in (DIR / f"argv_{ARM}_s{seed}.txt").read_text()
    ok, line = K.check_validation(K.describe(ARM, seed, _real_call()), _visible(dry), chk, 0, 0)
    assert ok, line
    assert "typed=[ent_coef 0.0]" in line and "grow=[none]" in line and "human_data=[off]" in line


def test_the_verdict_fails_transcripts_where_the_lever_did_not_reach_the_config():
    """`It launches` is not `it is the experiment`: without the typed row, with another value, or with the row in
    one tool only, the verdict FAILS; so does the hidden card's exit 3."""
    d = K.describe(ARM, 7331, _real_call())
    dry, chk = _transcripts(7331)
    restored = TYPED_ENT.replace("(TYPED)", "(restored from the record)")
    for bad in (restored, "ent_coef                 0.05       (TYPED)"):
        ok, line = K.check_validation(d, _visible(dry).replace(TYPED_ENT, bad), chk.replace(TYPED_ENT, bad), 0, 0)
        assert not ok and "dry-run typed ent_coef" in line and "checkargs typed ent_coef" in line, line
    ok, line = K.check_validation(d, _visible(dry), chk.replace(TYPED_ENT, restored), 0, 0)
    assert not ok and "checkargs typed ent_coef [] != the arm's lever" in line, line
    ok, line = K.check_validation(d, dry, chk, 0, 3)
    assert not ok and "dry-run exit 3" in line, line


def test_no_other_arm_passes_on_x70s_transcripts_nor_x70_on_the_swaps():
    """X70's transcripts are not C0's, X67's or the swap's, and the swap's are not X70's: each reading shows a typed
    knob the arm does not declare, another value, or a declared lever that is missing."""
    call = _real_call()
    dry, chk = _transcripts(7331)
    ok, line = K.check_validation(K.describe("c0", 3501, call), _visible(dry), chk, 0, 0)
    assert not ok and "checkargs shows a TYPED ent_coef, but c0 declares none" in line, line
    ok, line = K.check_validation(K.describe("x67", 6701, call), _visible(dry), chk, 0, 0)
    assert not ok and "typed ent_coef ['0.0'] != the arm's lever ['0.0125']" in line, line
    ok, line = K.check_validation(K.describe("ce3d_swap", 7321, call), _visible(dry), chk, 0, 0)
    assert not ok and "typed magnet_kl_coef [] != the arm's lever ['0.05']" in line, line
    sdry, schk = _transcripts(7321, "ce3d_swap", CE3D_DIR)
    ok, line = K.check_validation(K.describe(ARM, 7331, call), _visible(sdry), schk, 0, 0)
    assert not ok and "checkargs shows a TYPED magnet_kl_coef, but x70 declares none" in line, line


# ------------------------------------------------------------------------------------------ the stage
def _fill_fp32(stage: Path, sha: str = "c" * 40) -> None:
    (stage / B.SIDE_REQUIRED_FILE.format(side="production")).write_text(f"# header\n\n{sha} FP32_ALT the fp32 build\n")


def test_the_arm_is_side_pinned_and_allowed_only_with_the_prereg(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    (stage / B.READS_DIR).mkdir(parents=True)
    st = str(stage)
    prereg = tmp_path / "x70" / "PREREG.md"
    monkeypatch.setattr(B, "X70_PREREG", str(prereg))               # absolute: repo_path() returns it unchanged
    assert B.X70_ARMS == (ARM,)
    assert ARM in B.SIDE_PINNED_ARMS and ARM in B.PREREG_GATED_ARMS and ARM not in B.READ_LABEL
    assert all(B.stage_of(ARM, s) == "X70" for s in SEEDS)
    for seed in (7335, 3501) + SWAP_SEEDS + B.CE3D_RETIRED_SEEDS:
        assert B.stage_of(ARM, seed) is None
    assert [f for _, forks in B.X70_STAGES for f in forks] == [(ARM, s) for s in SEEDS]
    assert not {a for _, forks in B.STAGES + B.CE3D_STAGES for a, _ in forks} & {ARM}   # neither list gained it
    assert B.stage_status(st, ARM, SEEDS[0])[0] == B.PENDING                    # AM-8: no FP32_ALT named
    _fill_fp32(stage)
    status, why = B.stage_status(st, ARM, 7331)
    assert status == B.PENDING and "PREREG" in why, why
    prereg.parent.mkdir()
    prereg.write_text("# X70\n")
    for seed in SEEDS:
        status, why = B.stage_status(st, ARM, seed)
        assert status == B.ALLOWED and "X70 PREREG present" in why, why
    names = B.side_pinned_run_names()
    assert {n for n, (a, _) in names.items() if a == ARM} == {f"{STEM}{s}" for s in SEEDS}


def test_the_committed_prereg_is_where_the_batch_looks_and_names_the_arm_and_the_pins():
    assert repo_path(B.X70_PREREG).is_file(), B.X70_PREREG
    assert repo_path(B.X70_PREREG).parent == DIR
    text = repo_path(B.X70_PREREG).read_text()
    assert B.CE3D_PIN in text and f"ce3d-pin-{B.CE3D_PIN[:8]}" in text and B.CE3D_MAGNET_PIN[:8] in text
    assert "ANCHOR-ADDS" in text and "ANCHOR-COSTS" in text and "NOT-SEPARATED" in text and "INCONCLUSIVE" in text
    assert all(str(s) in text for s in SEEDS)
    assert "there is no GO marker" in text
    for name in ("chain_items_x70.json", "kit_checks_x70.txt", "arm_chain_plan_x70.txt", "dose_swap.txt",
                 "screen_rule.py", "cross/plan_cross.py", "cross/read_cross.py", "cross/drive_cross.sh"):
        assert (DIR / name).is_file(), name
    assert not [p for p in DIR.rglob("*") if p.name.upper().startswith("GO")]   # registered, not launched


# ------------------------------------------------------------------------------------------ the pin
def test_the_arm_trains_at_the_swaps_pin_under_its_own_registration():
    assert B.ARM_PINS[ARM] == B.CE3D_PIN == B.ARM_PINS["ce3d_swap"]
    assert B.ARM_PIN_AMENDMENT[ARM] == "X70 PREREG §3" and B.ARM_PIN_AMENDMENT["ce3d_swap"] == "CE-3d PREREG §4"


@needs_pin
def test_the_pin_check_passes_the_training_pin_and_refuses_every_other_for_this_arm():
    root = str(repo_root())
    ok, lines = B.arm_pin_check(ARM, B.CE3D_PIN, SIDE_PIN, root)
    assert ok and lines[0].startswith("X70 PREREG §3: ") and "exactly the 3 enumerated path(s)" in lines[0], lines
    for pin in (SIDE_PIN, B.CE3D_MAGNET_PIN):       # the plain production pin; the first CE-3d pin (the play pin)
        ok, lines = B.arm_pin_check(ARM, pin, SIDE_PIN, root)
        assert not ok and "x70 (an arm at the CE-3d training pin) launches only at CE3D_PIN" in lines[0], lines
        assert lines[0].startswith("X70 PREREG §3: "), lines
    ok, lines = B.arm_pin_check("ce3d_swap", B.CE3D_PIN, SIDE_PIN, root)        # the swap's lines are unchanged
    assert ok and lines[0].startswith("CE-3d PREREG §4: "), lines


@needs_pin
def test_the_pin_assigns_a_typed_zero_and_prints_a_magnet_line_only_with_a_positive_coefficient():
    """The two pin-source facts the banner rests on (PREREG §7): a fork ASSIGNS the typed entropy coefficient
    unconditionally (a `0` is not dropped as falsy), and each of the pin's three ``[MAGNET]`` prints sits behind a
    positive coefficient, so the VOID substring cannot fire on an X70 fork that built no magnet."""
    def show(path: str) -> str:
        return subprocess.run(["git", "-C", str(repo_root()), "show", f"{B.CE3D_PIN}:{path}"], check=True,
                              capture_output=True, text=True).stdout

    assert "\n        model.ent_coef = args.ent_coef " in show("src/main/train/model_build.py")
    magnet = show("src/agents/training/magnet.py")
    attach = magnet[magnet.index("def attach("):]
    off, say = attach.index("    if coef == 0.0:\n        model._magnet = None\n        return None\n"), \
        attach.index("[MAGNET] --magnet-kl-coef")
    assert off < say and magnet.count("[MAGNET]") == 1              # the early return precedes the one print
    config = show("src/main/train/config.py")
    assert config.count("[MAGNET]") == 1
    assert "    if args.magnet_kl_coef > 0.0:\n        emit(f\"🧲 [MAGNET] --magnet-kl-coef " in config
    setup = show("src/main/train/rust_env_setup.py")
    assert setup.count("[MAGNET]") == 1
    assert setup.index("if col.magnet_slot is not None:") < setup.index("[MAGNET] T2 slot")
    assert "[MAGNET] T2 slot".find(BANNER["void"][3]) == -1         # the slot line is not the VOID substring anyway


# ------------------------------------------------------------------------------------------ the draft chain items
def test_the_draft_chain_items_are_the_kits_forks_at_the_pin_with_the_registered_banner():
    """``chain_items_x70.json`` (the Training session's hand-off): four arm items, each the kit's run at CE3D_PIN,
    whose banner is the hyperparameter table's row at an entropy coefficient of 0 and whose VOID lines are that
    table's rows at every coefficient a sibling arm runs, plus the magnet's attach line."""
    doc = json.loads((DIR / "chain_items_x70.json").read_text())
    assert doc["x_pin"]["sha"] == B.CE3D_PIN and doc["x_pin"]["base"] == B.CE3D_MAGNET_PIN
    assert "GO" not in doc and "go" not in doc and "NO GO" in doc["x_what"]
    items = doc["items"]
    assert [(i["arm"], i["seed"]) for i in items] == [(ARM, s) for s in SEEDS]
    call = _real_call()
    for it in items:
        d = K.describe(it["arm"], it["seed"], call)
        assert it["kind"] == "arm" and it["pin"] == B.CE3D_PIN and it["run"] == d["run_name"]
        assert it["thin"] == "planner" and it["flags"] == ["--thinning"]
        assert it["on_void"] == "stop" and it["on_throughput_fail"] == "page"
        assert it["throughput"] == {"min_steps_per_s": 1000, "within_min": 60, "first_line_within_min": 90}
        assert it["disk_need_gib"] == 15.0                          # the dry run's 14.27, rounded up; no sidecars
        assert it["go_by"].startswith("<orchestrator, date>")       # the GO is the orchestrator's to fill
        b = it["banner"]
        assert b == BANNER
        assert b["expect"] in ROW_AT_ZERO and not any(v in ROW_AT_ZERO for v in b["void"])
        for row in ROWS_ELSEWHERE:                                  # another coefficient: VOID, never the banner
            assert b["expect"] not in row and any(v in row for v in b["void"]), row
        assert not any(v in b["expect"] for v in b["void"])


def test_the_recorded_plan_and_kit_checks_say_what_the_prereg_says():
    plan = (DIR / "arm_chain_plan_x70.txt").read_text()
    assert plan.count("the kit agrees — FORK of") >= 4 and "\nPLAN OK\nrc=0\n" in plan
    assert "REFUSED: seed 7335 is not registered for x70 ([7331, 7332, 7333, 7334])" in plan
    checks = (DIR / "kit_checks_x70.txt").read_text()
    for seed in SEEDS:
        assert f"ALLOWED: x70 s{seed}: no unmet condition (stage X70, X70 PREREG present)" in checks
    assert checks.count("launches only at CE3D_PIN 9e97cad5709d, not ") == 2
    dose = (DIR / "dose_swap.txt").read_text()
    for seed in SWAP_SEEDS:
        row = next(ln for ln in dose.splitlines() if ln.startswith(f"rb_x35_ce3dswap_s{seed}")).split()
        assert row[2:5] == ["65,536", "10", "0.0001005"] and row[6] == "2.044e-08", row


# ------------------------------------------------------------------------------------------ the cross and its rule
@pytest.fixture(scope="module")
def plan():
    return runpy.run_path(str(DIR / "cross" / "plan_cross.py"))


def test_the_cross_plans_constants_are_the_kits_and_the_batchs(plan):
    arms = plan["ARMS"]
    assert plan["CE3D_PIN"] == B.CE3D_PIN and plan["SIDE_PIN"] == SIDE_PIN
    for name, kit_arm in (("x70", ARM), ("swap", "ce3d_swap"), ("c0", "c0"), ("x67", "x67")):
        fmt, seeds, pin, ent, magnet = arms[name]
        assert seeds == K.ARMS[kit_arm].seeds, name
        assert [fmt.format(s) for s in seeds] == [K.run_name(kit_arm, s) for s in seeds], name
        want_ent = dict((k, float(v)) for k, v, _ in K.ARM_TYPED_RECIPE.get(kit_arm, ())).get("ent_coef", 0.05)
        assert ent == want_ent, name
        assert pin == (B.CE3D_PIN if kit_arm in (ARM, "ce3d_swap") else SIDE_PIN), name
    assert arms["swap"][4] == 0.05 and arms["x70"][4] == 0.0 and arms["c0"][4] is None and arms["x67"][4] is None
    assert plan["CROSSES"] == {"anchor": ("swap", "x70", 4000), "c0": ("x70", "c0", 1000),
                               "x67": ("x70", "x67", 1000)}
    assert plan["BANKED_ARMS"] == ("swap", "c0", "x67")             # the X70 forks are the only runs not yet banked
    call = _real_call()
    assert K.build_argv(ARM, 7331, call)[K.build_argv(ARM, 7331, call).index("--steps") + 1] \
        == str(plan["MIN_LAST_STEP"])
    assert plan["FORK_STEP"] + 20_000_000 == plan["MIN_LAST_STEP"]


def _x70_run(root: Path, plan: dict, seed: int = 7331) -> Path:
    run = root / f"{STEM}{seed}"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({
        "pin_history": [{"git_hash": B.CE3D_PIN, "first_step": plan["FORK_STEP"], "last_step": 300_061_000}],
        "lineage": {"role": "fork", "fork_parent": {"sha256": plan["PSTAR_SHA256"]}},
        "cli_args": {"seed": seed, "n_epochs": 10, "ent_coef": 0.0}, "ent_coef": 0.0,
        "torch_version": "2.8.0", "init_num_threads": 1, "num_timesteps": 300_061_000}))
    (run / "model_config.json").write_text(json.dumps({"magnet_kl_coef": 0.0, "magnet_refresh_steps": 2_000_000}))
    return run


def _edit(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


def test_the_plan_admits_a_clean_x70_fork_and_refuses_each_way_it_can_be_another_run(plan, tmp_path, monkeypatch):
    """The plan's guard for an X70 fork (PREREG §5.2): one pin entry at the training pin from the fork step, a fork
    of P*, its seed, entropy coefficient 0 in both records, the magnet field PRESENT at 0.0, no sidecar, K9(b)
    PASS. Each single departure is named."""
    monkeypatch.setattr(B, "k9b_read", lambda run: SimpleNamespace(verdict="PASS", reasons=[], fires=[]))
    _, seeds, pin, ent, magnet = plan["ARMS"]["x70"]
    check = plan["check_run"]
    run = _x70_run(tmp_path / "ok", plan)
    probs, facts = check(run, 7331, pin, ent, magnet)
    assert probs == [] and facts["torch_version"] == "2.8.0"

    def one(tag, fn, needle, file="metadata.json"):
        bad = _x70_run(tmp_path / tag, plan)
        _edit(bad / file, fn)
        got, _ = check(bad, 7331, pin, ent, magnet)
        assert len(got) >= 1 and any(needle in g for g in got), (tag, got)

    one("pin", lambda d: d["pin_history"][0].update(git_hash=B.CE3D_MAGNET_PIN), "is not one entry at 9e97cad5")
    one("two", lambda d: d["pin_history"].append(dict(d["pin_history"][0])), "is not one entry at 9e97cad5")
    one("start", lambda d: d["pin_history"][0].update(first_step=0), "not the fork step")
    one("short", lambda d: d["pin_history"][0].update(last_step=299_000_000), "last_step 299000000 <")
    one("fresh", lambda d: d["lineage"].update(role="fresh"), "is not a FORK of P*")
    one("seed", lambda d: d["cli_args"].update(seed=7332), "seed 7332 != 7331")
    one("ent", lambda d: d.update(ent_coef=0.05), "metadata.ent_coef 0.05 != 0.0")
    one("cli", lambda d: d["cli_args"].update(ent_coef=0.0125), "cli_args.ent_coef 0.0125 != 0.0")
    one("swap", lambda d: d.update(magnet_kl_coef=0.05), "model_config.magnet_kl_coef 0.05 != 0.0", "model_config.json")
    one("nokey", lambda d: d.pop("magnet_kl_coef"), "model_config.magnet_kl_coef None != 0.0", "model_config.json")
    side = _x70_run(tmp_path / "side", plan)
    (side / "final_model.magnet.pt").write_bytes(b"m")
    assert any("sits beside the final of a run with no magnet" in g for g in check(side, 7331, pin, ent, magnet)[0])
    monkeypatch.setattr(B, "k9b_read", lambda run: SimpleNamespace(verdict="FIRE", reasons=[], fires=["approx_kl"]))
    assert check(run, 7331, pin, ent, magnet)[0] == ["K9(b) FIRE: approx_kl"]


@pytest.fixture(scope="module")
def rule():
    return runpy.run_path(str(DIR / "screen_rule.py"))


def test_the_rules_self_check_passes_and_names_the_banked_swap_entropies(rule):
    out = rule["self_check"]()
    assert out["self_check"] == "OK" and out["problems"] == [], out["problems"]
    assert [round(x, 4) for x in out["swap_end_entropy_banked"]] == [0.4838, 0.4962, 0.4921]


def test_the_rule_reads_only_the_registered_shape_and_classes_by_the_intervals_sign(rule):
    decide = rule["decide"]
    up = [[51.0, 51.2, 50.9, 51.1], [51.3, 50.8, 51.0, 51.2], [50.9, 51.1, 51.2, 51.0]]
    got = decide(up)
    assert got["outcome"] == "ANCHOR-ADDS" and got["shape"] == [3, 4] and got["interval90_pp"][0] > 0
    down = [[100.0 - x for x in row] for row in up]
    got = decide(down)
    assert got["outcome"] == "ANCHOR-COSTS" and got["interval90_pp"][1] < 0
    flat = [[50.1, 49.9, 50.0, 50.2], [49.8, 50.1, 50.0, 49.9], [50.0, 50.1, 49.9, 50.0]]
    assert decide(flat)["outcome"] == "NOT-SEPARATED"
    for wrong in ([[50.0] * 3] * 3, [[50.0] * 4] * 4, [[50.0] * 3] * 4):
        assert decide(wrong)["outcome"] == "INCONCLUSIVE"


def test_the_entropy_labels_are_ranges_against_the_swaps_and_collapse_is_a_label(rule):
    labels = rule["labels"]
    assert labels([0.30, 0.35, 0.40, 0.45])["labels"] == ["ENT-BELOW-SWAP"]
    assert labels([0.50, 0.55, 0.60, 0.65])["labels"] == ["ENT-ABOVE-SWAP"]
    got = labels([0.20, 0.49, 0.50, 0.60])
    assert got["labels"] == ["ENT-COLLAPSED"] and got["collapsed_forks"] == [1]
    assert abs(got["collapse_bar_nats"] - got["swap_lowest"] / 2) < 1e-15
    assert labels([0.47, 0.49, 0.50, 0.60])["labels"] == []         # the ranges overlap: no label
