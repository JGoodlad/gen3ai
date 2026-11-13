"""CE-3d (PREREG ``ce3d_magnet_screen_2026-11-11``, Amendments 1 and 3): the trailing-magnet screen's ONE kit arm,
``ce3d_swap`` (C0 + ``--ent-coef 0 --magnet-kl-coef 0.05``: the magnet in place of the entropy bonus) -- its own pin
``batch.CE3D_PIN`` (the production side pin + the first CE-3d pin's ONE commit + the fit-check fix's ONE commit, each
changing exactly its enumerated paths), its stage, the committed registration files the kit and the arm chain read
(argvs, transcripts, the draft chain items), and the first fork's startup-dead attempt (Amendment 3). The
registration's two arms (``ce3d_mag``, ``ce3d_ent``) were dropped before any launch: no name or seed of theirs
reaches the kit.

The real-pin tests read this checkout's object store (the pins are tags ``ce3d-pin-9e97cad5`` and
``ce3d-pin-333a73dd`` on origin) and skip where they were not fetched. Nothing here launches, plays or touches
``models/``; the run directories are synthetic.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from main.x35 import batch as B
from main.x35 import kit as K
from main.x35.kit_test import STAGE, standin_call, toy_call
from utils.paths import repo_path, repo_root

DIR = repo_path("designs/research_state/measurements/ce3d_magnet_screen_2026-11-11")
SIDE_PIN = "f4b25238def57900c0305bb2a3a5a8f9edaef285"
D1_PIN = "440b61c320efdd4e172f54a192a2c84630a4c796"
ARM = "ce3d_swap"
SEEDS = (7321, 7322, 7323)
LEVER = ("--ent-coef", "0", "--magnet-kl-coef", "0.05")
STEM = "rb_x35_ce3dswap_s"
#: The two TYPED rows as both tools print them (the argv's ``0`` prints as ``0.0``).
TYPED_ENT = "ent_coef                 0.0        (TYPED)"
TYPED_MAG = "magnet_kl_coef           0.05       (TYPED)"
#: The line ``magnet.attach`` prints for this arm (the pin's CPU fork smoke printed it, 2026-11-11).
BANNER = "beside the entropy bonus (ent_coef 0; the KL's own entropy part adds 0.05 to that weight)"


def _real_call() -> dict:
    """The frozen production R's call record (P* = rb_x26_s2601 @280,000,170), as committed."""
    return json.loads((STAGE / "reference_population_production.json").read_text())


def _transcripts(arm: str, seed: int):
    stem = f"{arm}_s{seed}"
    return (DIR / f"dryrun_{stem}.txt").read_text(), (DIR / f"checkargs_{stem}.txt").read_text()


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


def test_the_arm_is_c0_plus_the_swap_and_nothing_else():
    """C0's argv token for token + the swap's four tokens in the lever slot; only the seed and run name differ."""
    assert K.CE3D_SWAP_LEVER == LEVER and K.CE3D_SWAP_SEEDS == SEEDS
    call = standin_call()
    off = K.build_argv("c0", 3501, call)
    cut = off.index("--seed")
    for seed in SEEDS:
        on = K.build_argv(ARM, seed, call)
        assert on[:-2] == off[:cut] + list(LEVER) + ["--seed", str(seed)]
        assert on[-2:] == ["--run-name", f"{STEM}{seed}"] and len(on) == len(off) + 4
        assert "--magnet-refresh-steps" not in on               # the cadence is the pin's default, untyped
    assert K.ARMS[ARM].seeds == SEEDS and ARM not in K.ARM_GROWTH and ARM not in K.ARM_HUMAN_DATA
    d = K.describe(ARM, SEEDS[1], call)
    assert d["side"] == "production" and d["delta_text"] == "+20,000,000 steps" and d["behaviour"] == "warn"
    assert d["parent_run_dir"].endswith("rb_x26_s2601") and d["k9b_ceilings"] is None


def test_the_swap_declares_both_typed_knobs_with_the_printed_spelling():
    """One arm, TWO typed recipe rows: the entropy coefficient at 0 (printed ``0.0``) and the magnet's coefficient."""
    assert K.ARM_TYPED_RECIPE == {
        "x67": (("ent_coef", "0.0125", "--ent-coef"),),
        ARM: (("ent_coef", "0.0", "--ent-coef"), ("magnet_kl_coef", "0.05", "--magnet-kl-coef")),
        "x70": (("ent_coef", "0.0", "--ent-coef"),)}                # X70: the swap's first row alone (x70_test.py)
    x67 = K.build_argv("x67", 6701, standin_call())
    swap = K.build_argv(ARM, 7321, standin_call())
    assert x67[x67.index("--ent-coef") + 1] == "0.0125" and swap[swap.index("--ent-coef") + 1] == "0"
    assert "--magnet-kl-coef" not in x67


def test_the_seeds_are_disjoint_from_every_other_arm_and_from_the_retired_six():
    """A seed names a run and a pool RNG stream; the swap's three are no sibling's, and the six seeds of the two
    arms Amendment 1 dropped are never reused by ANY arm."""
    others = {s for name, arm in K.ARMS.items() if name != ARM for s in arm.seeds}
    assert not others & set(SEEDS), others & set(SEEDS)
    assert B.CE3D_RETIRED_SEEDS == (7301, 7302, 7303, 7311, 7312, 7313)
    every = {s for arm in K.ARMS.values() for s in arm.seeds}
    assert not every & set(B.CE3D_RETIRED_SEEDS), every & set(B.CE3D_RETIRED_SEEDS)


@pytest.mark.parametrize("seed", [7320, 7324, 7301, 7311, 3501, 6701])
def test_the_arm_refuses_every_seed_but_its_three(seed):
    with pytest.raises(K.KitError, match="not registered"):
        K.build_argv(ARM, seed, standin_call())


def test_the_arm_refuses_the_toy_call_and_no_alias_or_dropped_arm_reaches_the_kit(capsys):
    with pytest.raises(K.KitError, match="reads the production call record"):
        K.build_argv(ARM, SEEDS[0], toy_call())
    for name in ("ce3d", "CE3D_SWAP", "ce3dswap", "toy_ce3d_swap", "ce3d_swap2", "ce3d_mag", "ce3d_ent"):
        assert K.canonical_arm(name) == name
        with pytest.raises(K.KitError, match="is not one of"):
            K.build_argv(name, 7321, standin_call())
        with pytest.raises(SystemExit):
            K.main(["describe", "--arm", name, "--seed", "7321", "--call", "/nonexistent"])
        capsys.readouterr()


# ------------------------------------------------------------------------------------------ the validation verdict
@pytest.mark.parametrize("seed", SEEDS)
def test_the_verdict_passes_the_real_transcripts_and_shows_both_typed_levers(seed):
    dry, chk = _transcripts(ARM, seed)
    assert dry.count("✗ REFUSED") == 2 and "CUDA_VISIBLE_DEVICES=''" in dry and "nice 19" in dry    # the two expected
    assert f"pin         : {B.CE3D_PIN}" in dry and f"the PINNED commit {B.CE3D_PIN[:8]}" in chk
    for row in (TYPED_ENT, TYPED_MAG):
        assert row in dry and row in chk
    ok, line = K.check_validation(K.describe(ARM, seed, _real_call()), _visible(dry), chk, 0, 0)
    assert ok, line
    assert "typed=[ent_coef 0.0, magnet_kl_coef 0.05]" in line and "grow=[none]" in line
    assert "human_data=[off]" in line


def test_the_verdict_fails_transcripts_where_either_lever_did_not_reach_the_config():
    """`It launches` is not `it is the experiment`: a transcript without EITHER typed row, with another value, or
    with the row in one tool only FAILS (each lever alone: a swap that kept the entropy bonus is the dropped magnet
    arm, a swap without its magnet is an entropy-free fork); so does the hidden card's exit 3."""
    d = K.describe(ARM, 7321, _real_call())
    dry, chk = _transcripts(ARM, 7321)
    for typed, other in ((TYPED_ENT, "ent_coef                 0.05       (TYPED)"),
                         (TYPED_MAG, "magnet_kl_coef           0.1        (TYPED)")):
        knob = typed.split()[0]
        restored = typed.replace("(TYPED)", "(restored from the record)")
        for bad in (restored, other):
            ok, line = K.check_validation(d, _visible(dry).replace(typed, bad), chk.replace(typed, bad), 0, 0)
            assert not ok and f"dry-run typed {knob}" in line and f"checkargs typed {knob}" in line, line
            assert line.count(" typed ") == 2, line                             # the OTHER lever still reads as typed
        ok, line = K.check_validation(d, _visible(dry), chk.replace(typed, restored), 0, 0)
        assert not ok and f"checkargs typed {knob} [] != the arm's lever" in line, line
    ok, line = K.check_validation(d, dry, chk, 0, 3)
    assert not ok and "dry-run exit 3" in line, line


def test_no_other_arm_passes_on_the_swaps_transcripts_nor_the_swap_on_c0s_shape():
    """The swap's transcripts are not C0's or X67's (each shows a typed knob the arm does not declare, or another
    value), and a C0 argv that types either of the swap's flags is refused."""
    call = _real_call()
    dry, chk = _transcripts(ARM, 7321)
    c0, x67 = K.describe("c0", 3501, call), K.describe("x67", 6701, call)
    ok, line = K.check_validation(c0, _visible(dry), chk, 0, 0)
    assert not ok and "checkargs shows a TYPED ent_coef, but c0 declares none" in line, line
    assert "checkargs shows a TYPED magnet_kl_coef, but c0 declares none" in line, line
    ok, line = K.check_validation(x67, _visible(dry), chk, 0, 0)
    assert not ok and "typed ent_coef ['0.0'] != the arm's lever ['0.0125']" in line, line
    assert "checkargs shows a TYPED magnet_kl_coef, but x67 declares none" in line, line
    untyped_chk = chk.replace(TYPED_ENT, "ent_coef                 0.05       (x)").replace(
        TYPED_MAG, "magnet_kl_coef           0.0        (x)")
    for flag, knob in (("--ent-coef", "ent_coef"), ("--magnet-kl-coef", "magnet_kl_coef")):
        typing = dict(c0)
        typing["argv"] = list(c0["argv"]) + [flag, "0.05"]
        ok, line = K.check_validation(typing, _visible(dry), untyped_chk, 0, 0)
        assert not ok and f"types {flag} but the arm declares no typed {knob}" in line, line


# ------------------------------------------------------------------------------------------ the stage
def _fill_fp32(stage: Path, sha: str = "c" * 40) -> None:
    (stage / B.SIDE_REQUIRED_FILE.format(side="production")).write_text(f"# header\n\n{sha} FP32_ALT the fp32 build\n")


def test_the_arm_is_side_pinned_and_allowed_only_with_the_prereg(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    (stage / B.READS_DIR).mkdir(parents=True)
    st = str(stage)
    prereg = tmp_path / "ce3d" / "PREREG.md"
    monkeypatch.setattr(B, "CE3D_PREREG", str(prereg))              # absolute: repo_path() returns it unchanged
    assert B.CE3D_ARMS == (ARM,)
    assert ARM in B.SIDE_PINNED_ARMS and ARM in B.PREREG_GATED_ARMS and ARM not in B.READ_LABEL
    assert all(B.stage_of(ARM, s) == "CE3D" for s in SEEDS)
    for seed in (7324, 3501) + B.CE3D_RETIRED_SEEDS:
        assert B.stage_of(ARM, seed) is None
    assert [f for _, forks in B.CE3D_STAGES for f in forks] == [(ARM, s) for s in SEEDS]
    assert not {a for _, forks in B.STAGES for a, _ in forks} & {ARM}           # STAGES is closed
    assert B.stage_status(st, ARM, SEEDS[0])[0] == B.PENDING                    # AM-8: no FP32_ALT named
    _fill_fp32(stage)
    status, why = B.stage_status(st, ARM, 7321)
    assert status == B.PENDING and "PREREG" in why, why
    prereg.parent.mkdir()
    prereg.write_text("# CE-3d\n")
    for seed in SEEDS:
        status, why = B.stage_status(st, ARM, seed)
        assert status == B.ALLOWED and "CE-3d PREREG present" in why, why
    names = B.side_pinned_run_names()
    assert {n for n, (a, _) in names.items() if a == ARM} == {f"{STEM}{s}" for s in SEEDS}
    assert not [n for n in names if "ce3dmag" in n or "ce3dent" in n]


def test_the_committed_prereg_is_where_the_batch_looks_and_carries_the_amendment():
    assert repo_path(B.CE3D_PREREG).is_file(), B.CE3D_PREREG
    assert repo_path(B.CE3D_PREREG).parent == DIR
    text = repo_path(B.CE3D_PREREG).read_text()
    assert "Amendment 1" in text and "--ent-coef 0 --magnet-kl-coef 0.05" in text
    # Amendment 3 (the re-pin after fork 1's step-0 launch failure) names the pin the kit launches at and its base
    assert "## 15. Amendment 3" in text and B.CE3D_PIN in text and B.CE3D_MAGNET_PIN in text
    assert f"ce3d-pin-{B.CE3D_PIN[:8]}" in text


# ------------------------------------------------------------------------------------------ the pin
def test_the_arm_has_its_own_pin_on_the_production_side_pin():
    assert B.ARM_PINS[ARM] == B.CE3D_PIN and len(B.CE3D_PIN) == 40
    assert not {"ce3d_mag", "ce3d_ent"} & (set(B.ARM_PINS) | set(B.ARM_PIN_AMENDMENT) | set(K.ARMS))
    # a pin on a pin (Amendment 3): the training pin sits on the first CE-3d pin, which sits on the side pin
    assert B.CE3D_PIN_BASE == B.CE3D_MAGNET_PIN and B.CE3D_MAGNET_PIN_BASE == SIDE_PIN
    assert len({B.CE3D_PIN, B.CE3D_MAGNET_PIN, SIDE_PIN}) == 3 and len(B.CE3D_MAGNET_PIN) == 40
    assert B.ARM_PIN_AMENDMENT[ARM] == "CE-3d PREREG §4"
    assert len(set(B.CE3D_MAGNET_PIN_PATHS)) == len(B.CE3D_MAGNET_PIN_PATHS) == 44
    assert "src/agents/training/magnet.py" in B.CE3D_MAGNET_PIN_PATHS
    # the fix's commit: the fit check and its two test files, none of them a loss, collector, version or play path
    assert sorted(B.CE3D_PIN_PATHS) == ["src/agents/training/rust_rollout/magnet_integration_test.py",
                                        "src/agents/training/update_fit.py",
                                        "src/agents/training/update_fit_test.py"]
    assert set(B.CE3D_PIN_PATHS) - set(B.CE3D_MAGNET_PIN_PATHS) == {"src/agents/training/update_fit.py",
                                                                    "src/agents/training/update_fit_test.py"}
    for paths in (B.CE3D_PIN_PATHS, B.CE3D_MAGNET_PIN_PATHS):
        assert not [p for p in paths if p.startswith(("src/rust", "data/", "src/agents/observation/"))]


@needs_pin
def test_the_registered_pin_is_one_commit_on_the_first_pin_on_the_side_pin_and_passes_its_own_check():
    root = str(repo_root())

    def parents(sha: str) -> list:
        return subprocess.run(["git", "-C", root, "rev-list", "--parents", "-n", "1", sha], check=True,
                              capture_output=True, text=True).stdout.split()

    assert parents(B.CE3D_PIN) == [B.CE3D_PIN, B.CE3D_MAGNET_PIN]
    assert parents(B.CE3D_MAGNET_PIN) == [B.CE3D_MAGNET_PIN, SIDE_PIN]
    ok, lines = B.arm_pin_check(ARM, B.CE3D_PIN, SIDE_PIN, root)
    assert ok and "exactly the 3 enumerated path(s)" in lines[0] and "exactly the 44 enumerated path(s)" in lines[0], lines


@needs_pin
def test_the_pin_check_refuses_the_wrong_pin_another_side_pin_and_a_moved_path_list(monkeypatch):
    root = str(repo_root())
    ok, lines = B.ce3d_pin_check(SIDE_PIN, SIDE_PIN, root)          # the plain production pin has no magnet
    assert not ok and "the CE-3d arm launches only at CE3D_PIN" in lines[0], lines
    ok, lines = B.ce3d_pin_check(B.CE3D_MAGNET_PIN, SIDE_PIN, root)  # the first pin dies at the fit check (Amendment 3)
    assert not ok and "the CE-3d arm launches only at CE3D_PIN" in lines[0], lines
    ok, lines = B.ce3d_pin_check(B.CE3D_PIN, D1_PIN, root)
    assert not ok and "the side's written pin" in lines[0], lines
    for name in ("CE3D_PIN_PATHS", "CE3D_MAGNET_PIN_PATHS"):        # either commit's path list, one path short
        with monkeypatch.context() as m:
            m.setattr(B, name, getattr(B, name)[:-1])
            ok, lines = B.ce3d_pin_check(B.CE3D_PIN, SIDE_PIN, root)
            assert not ok and "EXACTLY" in lines[0] and "extra" in lines[0], (name, lines)
    with monkeypatch.context() as m:                                # the training pin directly on the side pin
        m.setattr(B, "CE3D_PIN_BASE", SIDE_PIN)
        ok, lines = B.ce3d_pin_check(B.CE3D_PIN, SIDE_PIN, root)
        assert not ok and "parent(s)" in lines[0], lines
    assert B.ce3d_pin_check(B.CE3D_PIN, SIDE_PIN, root)[0]


@needs_pin
def test_the_training_pin_carries_the_fit_checks_fill_inputs_and_the_first_pin_does_not():
    """Amendment 3's reason, read from the two commits: the first pin's fit check ran its dry update on a fixture
    buffer with no magnet column (fork 1 died there); the training pin's fills it the way the collector does."""
    def fit(sha: str) -> str:
        return subprocess.run(["git", "-C", str(repo_root()), "show", f"{sha}:src/agents/training/update_fit.py"],
                              check=True, capture_output=True, text=True).stdout

    new, old = fit(B.CE3D_PIN), fit(B.CE3D_MAGNET_PIN)
    for piece in ("FILL_INPUTS", "class FillInputMissing(FatalConfigError)", "fill_column"):
        assert piece in new and piece not in old, piece


# ------------------------------------------------------------------------------------------ the startup-dead attempt
DEAD = "rb_x35_ce3dswap_s7321"


def _stage_at_side_pin(tmp_path: Path) -> Path:
    stage = tmp_path / "stage"
    stage.mkdir()
    B.pin_write_if_absent(SIDE_PIN, str(stage), "production", "FP32_ALT")
    return stage


def _dead_dir(models: Path, name: str) -> Path:
    """What fork 1's attempt left: logs, a crash file, the seeded pool, a compile cache -- no metadata, no
    checkpoint, no model."""
    d = models / name
    (d / "crashes").mkdir(parents=True)
    (d / "snapshots").mkdir()
    (d / "compile_cache").mkdir()
    (d / "snapshots" / "snapshot_280000170.zip").write_bytes(b"z")
    (d / "crashes" / "restart_err.txt").write_text("MagnetError")
    (d / "launcher_child.log").write_text("FATAL")
    return d


def _pin_check(stage: Path, models: Path):
    return B.pin_check(SIDE_PIN, str(stage), str(models), "production")


def test_the_startup_dead_attempt_is_registered_at_the_pin_it_died_at():
    assert B.STARTUP_DEAD_ATTEMPTS == {DEAD: (B.CE3D_MAGNET_PIN, "CE-3d PREREG Amendment 3 (§15)")}
    assert DEAD == K.run_name(ARM, SEEDS[0])


def test_the_dead_attempt_in_place_holds_the_side_and_moved_aside_is_listed_not_judged(tmp_path):
    stage, models = _stage_at_side_pin(tmp_path), tmp_path / "models"
    _dead_dir(models, DEAD)
    ok, lines = _pin_check(stage, models)                           # in place: the SOP move has not happened
    assert not ok and any("no recorded pin" in x and "still IN PLACE" in x for x in lines), lines
    (models / DEAD).rename(models / f"{DEAD}.FITCHECK_NO_MAGNET_COLUMN")
    ok, lines = _pin_check(stage, models)
    assert ok, lines
    assert any("STARTUP-DEAD attempt" in x and B.CE3D_MAGNET_PIN[:12] in x and "Amendment 3" in x for x in lines), lines


@pytest.mark.parametrize("leave", ["metadata", "latest", "checkpoint", "final", "empty_checkpoints_dir"])
def test_a_moved_aside_dead_dir_that_holds_training_output_is_judged_like_any_run(tmp_path, leave):
    stage, models = _stage_at_side_pin(tmp_path), tmp_path / "models"
    d = _dead_dir(models, f"{DEAD}.FITCHECK_NO_MAGNET_COLUMN")
    if leave == "metadata":
        (d / "metadata.json").write_text("{}")                      # written, but with no git_hash
    elif leave == "latest":
        (d / "latest.txt").write_text("checkpoints/checkpoint_281000170_steps.zip")
    elif leave == "checkpoint":
        (d / "checkpoints").mkdir()
        (d / "checkpoints" / "checkpoint_281000170_steps.zip").write_bytes(b"z")
    elif leave == "final":
        (d / "final_model_interrupted.zip").write_bytes(b"z")
    else:
        (d / "checkpoints").mkdir()
    ok, lines = _pin_check(stage, models)
    if leave == "empty_checkpoints_dir":                            # an empty directory is not a checkpoint
        assert ok, lines
        return
    assert not ok and any("no recorded pin" in x and "registered as a startup-dead attempt" in x for x in lines), lines


def test_the_allowance_is_the_registered_run_only_and_the_relaunch_is_judged_by_its_recorded_pin(tmp_path):
    stage, models = _stage_at_side_pin(tmp_path), tmp_path / "models"
    _dead_dir(models, f"{DEAD}.FITCHECK_NO_MAGNET_COLUMN")
    _dead_dir(models, "rb_x35_ce3dswap_s7322.DIED")                 # the same remains, a run nobody registered
    ok, lines = _pin_check(stage, models)
    assert not ok and any(x.startswith("rb_x35_ce3dswap_s7322.DIED: no recorded pin") for x in lines), lines
    assert not any("registered as a startup-dead attempt" in x for x in lines), lines
    (models / "rb_x35_ce3dswap_s7322.DIED").rename(tmp_path / "gone")
    relaunch = models / DEAD                                        # the relaunch under the same name records its pin
    relaunch.mkdir()
    for sha, good in ((B.CE3D_PIN, True), (B.CE3D_MAGNET_PIN, False), (SIDE_PIN, False)):
        (relaunch / "metadata.json").write_text(json.dumps({"git_hash": sha}))
        ok, lines = _pin_check(stage, models)
        assert ok is good, (sha, lines)


@needs_pin
def test_the_pin_carries_the_lever_applies_a_typed_zero_and_keeps_the_config_version():
    """The pin's parser knows the flag, its config version is the side pin's (no bump), a fork ASSIGNS the typed
    entropy coefficient unconditionally (a `0` is not dropped as falsy), and the banner the chain waits for is the
    text the pin's code prints (``:g`` prints 0.0 as ``0``)."""
    def show(path: str) -> str:
        return subprocess.run(["git", "-C", str(repo_root()), "show", f"{B.CE3D_PIN}:{path}"], check=True,
                              capture_output=True, text=True).stdout

    def version(sha: str) -> str:
        text = subprocess.run(["git", "-C", str(repo_root()), "show",
                               f"{sha}:src/agents/model/model_version/constants.py"], check=True,
                              capture_output=True, text=True).stdout
        return next(ln for ln in text.splitlines() if ln.startswith("MODEL_CONFIG_VERSION"))

    assert version(B.CE3D_PIN) == version(SIDE_PIN)
    assert '"--magnet-kl-coef"' in show("src/main/train/parser/hyperparameters.py")
    assert "\n        model.ent_coef = args.ent_coef " in show("src/main/train/model_build.py")
    magnet = show("src/agents/training/magnet.py")
    for piece in ('over the legal actions, beside the "',
                  "f\"entropy bonus (ent_coef {float(getattr(model, 'ent_coef', 0.0)):g}; the KL's own entropy part "
                  "adds {coef:g} \"",
                  'f"to that weight). The magnet: '):
        assert piece in magnet, piece
    assert f"beside the entropy bonus (ent_coef {0.0:g}; the KL's own entropy part adds {0.05:g} to that weight)" \
        == BANNER


# ------------------------------------------------------------------------------------------ the draft chain items
def test_the_draft_chain_items_are_the_kits_forks_at_the_pin_with_the_swaps_banner():
    """``chain_items_ce3d.json`` (the Training session's hand-off): three arm items, each the kit's run at CE3D_PIN,
    whose banner is the magnet's attach line naming BOTH levers (entropy coefficient 0, the KL at 0.05) and whose
    VOID lines are the hyperparameter table's rows for every entropy coefficient a sibling arm runs."""
    items = json.loads((DIR / "chain_items_ce3d.json").read_text())["items"]
    assert [(i["arm"], i["seed"]) for i in items] == [(ARM, s) for s in SEEDS]
    call = _real_call()
    for it in items:
        d = K.describe(it["arm"], it["seed"], call)
        assert it["kind"] == "arm" and it["pin"] == B.CE3D_PIN and it["run"] == d["run_name"]
        assert it["thin"] == "planner" and it["flags"] == ["--thinning"]
        assert it["on_void"] == "stop" and it["on_throughput_fail"] == "page"
        assert it["throughput"] == {"min_steps_per_s": 1000, "within_min": 60, "first_line_within_min": 90}
        assert it["disk_need_gib"] == 16.0                          # the dry run's 14.33 + the magnet's sidecars
        b = it["banner"]
        assert b["within_min"] == 30 and b["expect"] == BANNER
        assert b["void"] == ["ent_coef   | 0.05 ", "ent_coef   | 0.1 ", "ent_coef   | 0.0125 ",
                             "[ModelVersion] FATAL"]
        assert not any(v in b["expect"] for v in b["void"])
        assert not any(v in "|    ent_coef   | 0        |" for v in b["void"])     # the pin's smoke row at 0
