"""CE-3d (PREREG ``ce3d_magnet_screen_2026-11-11``): the trailing-magnet screen's two kit arms -- ``ce3d_mag`` (C0 +
``--magnet-kl-coef 0.05``) and its entropy-matched control ``ce3d_ent`` (C0 + ``--ent-coef 0.1``) -- their ONE own pin
``batch.CE3D_PIN`` (the production side pin + ONE commit changing exactly the enumerated paths), their stage, and the
committed registration files the kit and the arm chain read (argvs, transcripts, the draft chain items).

The real-pin tests read this checkout's object store (the pin is tag ``ce3d-pin-333a73dd`` on origin) and skip where
it was not fetched. Nothing here launches, plays or touches ``models/``.
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
#: arm -> (seeds, the lever, the run-name stem, the typed knob as the tools print it)
ARMS = {"ce3d_mag": ((7301, 7302, 7303), ("--magnet-kl-coef", "0.05"), "rb_x35_ce3dmag_s", "magnet_kl_coef 0.05"),
        "ce3d_ent": ((7311, 7312, 7313), ("--ent-coef", "0.1"), "rb_x35_ce3dent_s", "ent_coef 0.1")}
FORKS = [(arm, seed) for arm, (seeds, *_rest) in ARMS.items() for seed in seeds]


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


# ------------------------------------------------------------------------------------------ the kit arms
@pytest.mark.parametrize("arm,seed", FORKS)
def test_the_kit_builds_the_validated_argv_token_for_token(arm, seed):
    """The kit builds exactly the line checkargs / the dry run validated at CE3D_PIN (PREREG §7)."""
    want = (DIR / f"argv_{arm}_s{seed}.txt").read_text().split()
    assert K.build_argv(arm, seed, _real_call()) == want


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_each_arm_is_c0_plus_its_one_lever_and_nothing_else(arm):
    """Both arms are C0's argv token for token + ONE lever in the same slot; only the seed and run name differ."""
    seeds, lever, stem, _ = ARMS[arm]
    call = standin_call()
    off = K.build_argv("c0", 3501, call)
    cut = off.index("--seed")
    for seed in seeds:
        on = K.build_argv(arm, seed, call)
        assert on[:-2] == off[:cut] + list(lever) + ["--seed", str(seed)]
        assert on[-2:] == ["--run-name", f"{stem}{seed}"] and len(on) == len(off) + 2
    assert K.ARMS[arm].seeds == seeds and arm not in K.ARM_GROWTH and arm not in K.ARM_HUMAN_DATA
    d = K.describe(arm, seeds[1], call)
    assert d["side"] == "production" and d["delta_text"] == "+20,000,000 steps" and d["behaviour"] == "warn"
    assert d["parent_run_dir"].endswith("rb_x26_s2601") and d["k9b_ceilings"] is None


def test_the_two_levers_are_the_registered_ones_and_differ_only_there():
    """The deciding contrast is magnet against control: their argvs differ by the lever, the seed and the name."""
    assert K.CE3D_MAG_LEVER == ("--magnet-kl-coef", "0.05") and K.CE3D_ENT_LEVER == ("--ent-coef", "0.1")
    assert K.ARM_TYPED_RECIPE == {"x67": ("ent_coef", "0.0125", "--ent-coef"),
                                  "ce3d_mag": ("magnet_kl_coef", "0.05", "--magnet-kl-coef"),
                                  "ce3d_ent": ("ent_coef", "0.1", "--ent-coef")}
    call = standin_call()
    mag, ent = K.build_argv("ce3d_mag", 7301, call), K.build_argv("ce3d_ent", 7311, call)
    assert [(a, b) for a, b in zip(mag, ent) if a != b] == [
        ("--magnet-kl-coef", "--ent-coef"), ("0.05", "0.1"), ("7301", "7311"),
        ("rb_x35_ce3dmag_s7301", "rb_x35_ce3dent_s7311")]
    assert "--ent-coef" not in mag and "--magnet-kl-coef" not in ent and "--magnet-refresh-steps" not in mag


def test_the_seeds_are_disjoint_from_every_other_arm_and_from_each_other():
    """A seed names a run and a pool RNG stream; CE-3d's six must be no sibling's (the PREREG's search, as a test)."""
    mine = {s for arm in ARMS for s in K.ARMS[arm].seeds}
    assert len(mine) == 6
    others = {s for name, arm in K.ARMS.items() if name not in ARMS for s in arm.seeds}
    assert not others & mine, others & mine


@pytest.mark.parametrize("arm,seed", [("ce3d_mag", 7304), ("ce3d_mag", 7311), ("ce3d_mag", 3501),
                                      ("ce3d_ent", 7301), ("ce3d_ent", 7314), ("ce3d_ent", 6701)])
def test_each_arm_refuses_every_seed_but_its_three(arm, seed):
    with pytest.raises(K.KitError, match="not registered"):
        K.build_argv(arm, seed, standin_call())


def test_the_arms_refuse_the_toy_call_and_no_alias_or_variant_reaches_them(capsys):
    for arm, (seeds, *_rest) in ARMS.items():
        with pytest.raises(K.KitError, match="reads the production call record"):
            K.build_argv(arm, seeds[0], toy_call())
    for name in ("ce3d", "CE3D_MAG", "ce3dmag", "ce3dent", "toy_ce3d_mag", "ce3d_swap", "ce3d_mag2"):
        assert K.canonical_arm(name) == name
        with pytest.raises(K.KitError, match="is not one of"):
            K.build_argv(name, 7301, standin_call())
        with pytest.raises(SystemExit):
            K.main(["describe", "--arm", name, "--seed", "7301", "--call", "/nonexistent"])
        capsys.readouterr()


# ------------------------------------------------------------------------------------------ the validation verdict
@pytest.mark.parametrize("arm,seed", FORKS)
def test_the_verdict_passes_the_real_transcripts_and_shows_the_typed_lever(arm, seed):
    dry, chk = _transcripts(arm, seed)
    assert dry.count("✗ REFUSED") == 2 and "CUDA_VISIBLE_DEVICES=''" in dry and "nice 19" in dry    # the two expected
    assert f"pin         : {B.CE3D_PIN}" in dry and f"the PINNED commit {B.CE3D_PIN[:8]}" in chk
    ok, line = K.check_validation(K.describe(arm, seed, _real_call()), _visible(dry), chk, 0, 0)
    assert ok, line
    assert f"typed=[{ARMS[arm][3]}]" in line and "grow=[none]" in line and "human_data=[off]" in line


def test_the_verdict_fails_transcripts_whose_lever_did_not_reach_the_config():
    """`It launches` is not `it is the experiment`: a transcript without the arm's TYPED row, with another value, or
    with the lever in one tool only FAILS; so does the hidden card's exit 3."""
    call = _real_call()
    for arm, seed, typed, other in (
            ("ce3d_mag", 7301, "magnet_kl_coef           0.05       (TYPED)", "magnet_kl_coef           0.1        (TYPED)"),
            ("ce3d_ent", 7311, "ent_coef                 0.1        (TYPED)", "ent_coef                 0.0125     (TYPED)")):
        dry, chk = _transcripts(arm, seed)
        d = K.describe(arm, seed, call)
        knob = typed.split()[0]
        assert typed in dry and typed in chk
        restored = typed.replace("(TYPED)", "(restored from the record)")
        for bad in (restored, other):
            ok, line = K.check_validation(d, _visible(dry).replace(typed, bad), chk.replace(typed, bad), 0, 0)
            assert not ok and f"dry-run typed {knob}" in line and f"checkargs typed {knob}" in line, line
        ok, line = K.check_validation(d, _visible(dry), chk.replace(typed, restored), 0, 0)
        assert not ok and f"checkargs typed {knob} [] != the arm's lever" in line, line
        ok, line = K.check_validation(d, dry, chk, 0, 3)
        assert not ok and "dry-run exit 3" in line, line


def test_no_arm_passes_on_another_arms_transcripts():
    """The magnet's transcripts are not the control's, C0's or X67's, and the reverse: each arm's typed knob must be
    its own, and an arm that declares none must show none in checkargs."""
    call = _real_call()
    mag_dry, mag_chk = _transcripts("ce3d_mag", 7301)
    ent_dry, ent_chk = _transcripts("ce3d_ent", 7311)
    mag, ent = K.describe("ce3d_mag", 7301, call), K.describe("ce3d_ent", 7311, call)
    c0, x67 = K.describe("c0", 3501, call), K.describe("x67", 6701, call)
    ok, line = K.check_validation(mag, _visible(ent_dry), ent_chk, 0, 0)
    assert not ok and "typed magnet_kl_coef [] != the arm's lever ['0.05']" in line, line
    assert "checkargs shows a TYPED ent_coef, but ce3d_mag declares none" in line, line
    ok, line = K.check_validation(ent, _visible(mag_dry), mag_chk, 0, 0)
    assert not ok and "typed ent_coef [] != the arm's lever ['0.1']" in line, line
    assert "checkargs shows a TYPED magnet_kl_coef, but ce3d_ent declares none" in line, line
    for other in (c0, x67):
        ok, line = K.check_validation(other, _visible(mag_dry), mag_chk, 0, 0)
        assert not ok and f"checkargs shows a TYPED magnet_kl_coef, but {other['arm']} declares none" in line, line
    ok, line = K.check_validation(x67, _visible(ent_dry), ent_chk, 0, 0)
    assert not ok and "typed ent_coef ['0.1'] != the arm's lever ['0.0125']" in line, line
    c0 = dict(c0)
    c0["argv"] = list(c0["argv"]) + ["--magnet-kl-coef", "0.05"]
    c0_chk = mag_chk.replace("magnet_kl_coef           0.05       (TYPED)", "magnet_kl_coef           0.0        (x)")
    ok, line = K.check_validation(c0, _visible(mag_dry), c0_chk, 0, 0)
    assert not ok and "types --magnet-kl-coef but the arm declares no typed magnet_kl_coef" in line, line


# ------------------------------------------------------------------------------------------ the stage
def _fill_fp32(stage: Path, sha: str = "c" * 40) -> None:
    (stage / B.SIDE_REQUIRED_FILE.format(side="production")).write_text(f"# header\n\n{sha} FP32_ALT the fp32 build\n")


def test_the_arms_are_side_pinned_and_allowed_only_with_the_prereg(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    (stage / B.READS_DIR).mkdir(parents=True)
    st = str(stage)
    prereg = tmp_path / "ce3d" / "PREREG.md"
    monkeypatch.setattr(B, "CE3D_PREREG", str(prereg))              # absolute: repo_path() returns it unchanged
    assert B.CE3D_ARMS == ("ce3d_mag", "ce3d_ent")
    for arm, (seeds, *_rest) in ARMS.items():
        assert arm in B.SIDE_PINNED_ARMS and arm in B.PREREG_GATED_ARMS and arm not in B.READ_LABEL
        assert all(B.stage_of(arm, s) == "CE3D" for s in seeds)
        assert B.stage_of(arm, seeds[-1] + 1) is None and B.stage_of(arm, 3501) is None
        assert B.stage_status(st, arm, seeds[0])[0] == B.PENDING                # AM-8: no FP32_ALT named
    assert B.stage_of("ce3d_mag", 7311) is None and B.stage_of("ce3d_ent", 7301) is None
    assert not {a for _, forks in B.STAGES for a, _ in forks} & set(ARMS)       # STAGES is closed
    _fill_fp32(stage)
    status, why = B.stage_status(st, "ce3d_mag", 7301)
    assert status == B.PENDING and "PREREG" in why, why
    prereg.parent.mkdir()
    prereg.write_text("# CE-3d\n")
    for arm, seed in FORKS:
        status, why = B.stage_status(st, arm, seed)
        assert status == B.ALLOWED and "CE-3d PREREG present" in why, why
    names = B.side_pinned_run_names()
    assert {n for n, (a, _) in names.items() if a in ARMS} == \
        {f"{stem}{s}" for seeds, _lever, stem, _typed in ARMS.values() for s in seeds}


def test_the_committed_prereg_is_where_the_batch_looks():
    assert repo_path(B.CE3D_PREREG).is_file(), B.CE3D_PREREG
    assert repo_path(B.CE3D_PREREG).parent == DIR


# ------------------------------------------------------------------------------------------ the pin
def test_both_arms_share_one_own_pin_on_the_production_side_pin():
    assert B.ARM_PINS["ce3d_mag"] == B.ARM_PINS["ce3d_ent"] == B.CE3D_PIN and len(B.CE3D_PIN) == 40
    assert B.CE3D_PIN_BASE == SIDE_PIN and B.CE3D_PIN != SIDE_PIN
    assert B.ARM_PIN_AMENDMENT["ce3d_mag"] == B.ARM_PIN_AMENDMENT["ce3d_ent"] == "CE-3d PREREG §4"
    assert len(set(B.CE3D_PIN_PATHS)) == len(B.CE3D_PIN_PATHS) == 44
    assert "src/agents/training/magnet.py" in B.CE3D_PIN_PATHS
    assert not [p for p in B.CE3D_PIN_PATHS if p.startswith(("src/rust", "data/", "src/agents/observation/"))]


@needs_pin
def test_the_registered_pin_is_one_commit_on_the_side_pin_and_passes_its_own_check():
    root = str(repo_root())
    parents = subprocess.run(["git", "-C", root, "rev-list", "--parents", "-n", "1", B.CE3D_PIN], check=True,
                             capture_output=True, text=True).stdout.split()
    assert parents == [B.CE3D_PIN, SIDE_PIN]
    for arm in ARMS:
        ok, lines = B.arm_pin_check(arm, B.CE3D_PIN, SIDE_PIN, root)
        assert ok and "exactly the 44 enumerated path(s)" in lines[0], lines


@needs_pin
def test_the_pin_check_refuses_the_wrong_pin_another_side_pin_and_a_moved_path_list(monkeypatch):
    root = str(repo_root())
    ok, lines = B.ce3d_pin_check(SIDE_PIN, SIDE_PIN, root)          # the plain production pin has no magnet
    assert not ok and "the CE-3d arms launch only at CE3D_PIN" in lines[0], lines
    ok, lines = B.ce3d_pin_check(B.CE3D_PIN, D1_PIN, root)
    assert not ok and "the side's written pin" in lines[0], lines
    monkeypatch.setattr(B, "CE3D_PIN_PATHS", B.CE3D_PIN_PATHS[:-1])
    ok, lines = B.ce3d_pin_check(B.CE3D_PIN, SIDE_PIN, root)
    assert not ok and "EXACTLY" in lines[0] and "extra" in lines[0], lines


@needs_pin
def test_the_pin_carries_the_lever_and_keeps_the_production_config_version():
    """The pin's parser knows the flag, its config version is the side pin's (no bump), and the banner the chain
    waits for is the text the pin's code prints."""
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
    magnet = show("src/agents/training/magnet.py")
    for piece in ('over the legal actions, beside the "',
                  "f\"entropy bonus (ent_coef {float(getattr(model, 'ent_coef', 0.0)):g}; the KL's own entropy part "
                  "adds {coef:g} \"",
                  'f"to that weight). The magnet: '):
        assert piece in magnet, piece


# ------------------------------------------------------------------------------------------ the draft chain items
def test_the_draft_chain_items_are_the_kits_forks_at_the_pin_with_each_arms_banner():
    """``chain_items_ce3d.json`` (the Training session's hand-off): six arm items, the arms alternating, each the
    kit's run at CE3D_PIN, with the arm's own banner -- the magnet's attach line, the control's 0.1 row -- and the
    other arm's lever among its VOID lines."""
    items = json.loads((DIR / "chain_items_ce3d.json").read_text())["items"]
    assert [(i["arm"], i["seed"]) for i in items] == [
        ("ce3d_mag", 7301), ("ce3d_ent", 7311), ("ce3d_mag", 7302), ("ce3d_ent", 7312), ("ce3d_mag", 7303),
        ("ce3d_ent", 7313)]
    call = _real_call()
    for it in items:
        d = K.describe(it["arm"], it["seed"], call)
        assert it["kind"] == "arm" and it["pin"] == B.CE3D_PIN and it["run"] == d["run_name"]
        assert it["thin"] == "planner" and it["flags"] == ["--thinning"]
        assert it["on_void"] == "stop" and it["on_throughput_fail"] == "page"
        assert it["throughput"] == {"min_steps_per_s": 1000, "within_min": 60, "first_line_within_min": 90}
        b = it["banner"]
        assert b["within_min"] == 30 and "[ModelVersion] FATAL" in b["void"]
        if it["arm"] == "ce3d_mag":
            assert b["expect"] == ("beside the entropy bonus (ent_coef 0.05; the KL's own entropy part adds 0.05 to "
                                   "that weight)")
            assert "ent_coef   | 0.1 " in b["void"] and it["disk_need_gib"] > 14.33
        else:
            assert b["expect"] == "ent_coef   | 0.1 " and {"ent_coef   | 0.05 ", "[MAGNET]"} <= set(b["void"])
        assert not any(v in b["expect"] for v in b["void"])
