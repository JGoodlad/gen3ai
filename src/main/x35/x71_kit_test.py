"""X71's LAUNCH KIT (PREREG ``x71_exploitability_probe_2026-11-16`` §4.3 "OWED"): the eight responders as kit arms
(``main.x35.x71``, ``kit.ARMS``), their stage and pin checks (``main.x35.batch``), the declared placeholder of X70's
unfinished target, the draft chain items and the banked plan / kit-check transcripts, and the read's driver
(``drive_read_x71.sh``).

The registration's own files (the argvs, the transcripts, the frozen rule and reader) are pinned by ``x71_test.py``;
this file pins the kit to them. Nothing here launches, plays a game or touches ``models/``: the run directories are
synthetic and the driver's reader is a stub or the frozen reader over a recorded player.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from main.ops.arm_chain import spec as chain_spec
from main.x35 import batch as B
from main.x35 import kit as K
from main.x35 import x71 as X
from main.x35.kit_test import STAGE, toy_call
from main.x35.x71_test import DIR, EXISTING, MODELS, TEAMS, _load, _write_run
from utils.paths import repo_path, repo_root

SIDE_PIN = "f4b25238def57900c0305bb2a3a5a8f9edaef285"
ARMS = ("x71_c0", "x71_x67", "x71_swap", "x71_x70")
SEEDS = (7111, 7112)
#: the registered launch order (PREREG §4.1): the three banked targets at 7111, then at 7112; X70's two later
ORDER = [(f"x71_{t}", s) for s in SEEDS for t in ("c0", "x67", "swap")] + [("x71_x70", s) for s in SEEDS]
X70_RUN = "rb_x35_x70ent0_s7331"
#: P*'s step + the X70 arm's length: the step a finished X70 fork reaches at least
X70_FLOOR = 300_000_170
#: The hyperparameter table's rows as a fork's log prints them (X70 PREREG §7): the typed 0.05, and the two values a
#: target's own record could win with (X70's 0, X67's 0.0125).
ROW_TYPED = "|    ent_coef   | 0.05     |"
ROWS_ELSEWHERE = ("|    ent_coef   | 0        |", "|    ent_coef   | 0.0125   |")
BANNER = {"expect": "ent_coef   | 0.05 ",
          "void": ["ent_coef   | 0 ", "ent_coef   | 0.0125 ", "[MAGNET] --magnet-kl-coef", "[ModelVersion] FATAL"],
          "within_min": 30}


def _real_call() -> dict:
    """The frozen production R's call record (P* = rb_x26_s2601 @280,000,170), as committed."""
    return json.loads((STAGE / "reference_population_production.json").read_text())


def _banked_target(tag: str) -> dict:
    """The target record ``x71.target`` returns for a banked final, built from the registration's own values."""
    run, step = EXISTING[tag]
    return {"run": run, "path": f"{MODELS}/{run}/final_model.zip", "step": step, "sha256": X.TARGET_SHA256[tag],
            "standin": False, "placeholder": False}


def _visible(dry: str) -> str:
    """The registration's dry run ran with the card hidden at nice 19 (rule 5, 5b): its two refusals are the CUDA
    and the inherited-nice preflights (``x71_test`` pins exactly those two). The launch re-runs it visible; model
    that one verdict line only."""
    return dry.replace("this command would NOT launch", "this command would launch")


def _has(sha: str) -> bool:
    return subprocess.run(["git", "-C", str(repo_root()), "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True).returncode == 0


needs_pin = pytest.mark.skipif(not _has(B.CE3D_PIN), reason="CE3D_PIN is not in this object store (fetch the tag)")


# ------------------------------------------------------------------------------- the kit's tables are the rule's
def test_the_kits_tables_are_the_frozen_rules():
    """``main.x35.x71`` restates ``x71_rule.py`` (a frozen record is never imported by a tool): every constant, the
    run names and the argv of all eight responders are the rule's."""
    rule = _load("x71_rule")
    assert tuple(X.X71_ARMS) == ARMS == B.X71_ARMS and X.X71_ARMS == {f"x71_{t}": t for t in rule.TARGETS}
    assert {t: (arm, run) for t, (arm, run, _lever) in rule.TARGETS.items()} == X.TARGETS
    assert X.TARGET_SHA256 == rule.TARGET_SHA256 and X.SEEDS == rule.SEEDS == SEEDS and X.L_RESP == rule.L_RESP
    assert X.PIN == rule.PIN == B.CE3D_PIN and X.FORK_LR == rule.FORK_LR and X.RECIPE == rule.RECIPE
    assert {t: s for t, s in X.TARGET_STEP.items() if s is not None} == {t: s for t, (_r, s) in EXISTING.items()}
    assert X.TARGET_STEP["x70"] is None and X.TARGET_SHA256["x70"] is None
    for tag in rule.TARGETS:
        arm = f"x71_{tag}"
        a = K.ARMS[arm]
        assert (a.side, a.length, a.seeds, a.phase, a.behaviour) == ("production", rule.L_RESP, SEEDS, X.X71_PHASE,
                                                                     "warn")
        # the target is the target ARM's lowest registered seed, in the kit's own tables
        t_arm = X.target_arm(arm)
        assert K.run_name(t_arm, min(K.ARMS[t_arm].seeds)) == X.target_run(arm) == rule.TARGETS[tag][1]
        for seed in SEEDS:
            assert K.run_name(arm, seed) == rule.run_name(tag, seed)
            t = {"run": X.target_run(arm), "path": "/m/T/final_model.zip", "step": 300_000_000,
                 "placeholder": False}
            assert K.build_argv(arm, seed, _real_call(), None, t) == rule.responder_argv(
                tag, seed, "/m/T/final_model.zip", 300_000_000)
    from main.x35 import ge

    assert ge.offense_teams().split(",") == TEAMS


@pytest.mark.parametrize("tag", sorted(EXISTING))
@pytest.mark.parametrize("seed", SEEDS)
def test_the_kit_builds_the_banked_argv_token_for_token(tag, seed):
    """The kit builds exactly the line checkargs and the dry run validated at the pin (PREREG §4.2)."""
    want = (DIR / f"argv_x71_{tag}_s{seed}.txt").read_text().split()
    d = K.describe(f"x71_{tag}", seed, _real_call(), None, _banked_target(tag))
    assert d["argv"] == want and len(want) == 26
    assert d["run_name"] == f"rb_x35_x71_{tag}_s{seed}" and d["delta_text"] == "+5,000,000 steps"
    assert d["parent_run_dir"] == f"{MODELS}/{EXISTING[tag][0]}"        # a FORK of the target, never of P*
    assert d["phase"] == X.X71_PHASE and d["side"] == "production" and d["behaviour"] == "warn"
    assert "phase1" not in d


@pytest.mark.parametrize("tag", sorted(EXISTING))
@pytest.mark.parametrize("seed", SEEDS)
def test_the_verdict_passes_the_banked_transcripts_once_the_card_is_visible(tag, seed):
    """``kit validate-run``'s verdict on the registration's own transcripts: PASS with the dry run's verdict line
    visible, FAIL as banked (exit 3, the registering shell's two refusals). The typed recipe is the arm's own: the
    two knobs are declared in ``ARM_TYPED_AT_PRODUCTION``, without which a production arm that types them fails."""
    dry = (DIR / f"dryrun_x71_{tag}_s{seed}.txt").read_text()
    chk = (DIR / f"checkargs_x71_{tag}_s{seed}.txt").read_text()
    d = K.describe(f"x71_{tag}", seed, _real_call(), None, _banked_target(tag))
    ok, line = K.check_validation(d, _visible(dry), chk, 0, 0)
    assert ok, line
    assert f"role=[FORK of {MODELS}/{EXISTING[tag][0]}]" in line and "delta=[+5,000,000 steps]" in line
    assert "gib=[7.91]" in line and "k9b=[warn]" in line and "grow=[none]" in line and "human_data=[off]" in line
    ok, line = K.check_validation(d, dry, chk, 0, 3)
    assert not ok and "dryrun=3" in line and "does not say 'would launch' exactly once" in line
    # another responder's transcripts are not this arm's
    other = "x67" if tag != "x67" else "c0"
    ok, line = K.check_validation(K.describe(f"x71_{other}", seed, _real_call(), None, _banked_target(other)),
                                  _visible(dry), chk, 0, 0)
    assert not ok, line


def test_the_typed_recipe_is_declared_for_the_responders_only(monkeypatch):
    assert K.ARM_TYPED_AT_PRODUCTION == {a: ("--ent-coef", "--magnet-kl-coef") for a in ARMS}
    dry = (DIR / "dryrun_x71_c0_s7111.txt").read_text()
    chk = (DIR / "checkargs_x71_c0_s7111.txt").read_text()
    d = K.describe("x71_c0", 7111, _real_call(), None, _banked_target("c0"))
    monkeypatch.setattr(K, "ARM_TYPED_AT_PRODUCTION", {})
    ok, line = K.check_validation(d, _visible(dry), chk, 0, 0)
    assert not ok, line                                 # undeclared, a typed recipe knob fails the verdict


@pytest.mark.parametrize("arm,seed", [("x71_c0", 7113), ("x71_c0", 3501), ("x71_x67", 6701), ("x71_swap", 7321),
                                      ("x71_x70", 7331), ("x71_x70", 7110)])
def test_a_responder_arm_refuses_every_seed_but_its_two(arm, seed):
    with pytest.raises(K.KitError, match=rf"seed {seed} is not registered for {arm} \(\[7111, 7112\]\)"):
        K.describe(arm, seed, _real_call(), None, _banked_target("c0"))


def test_a_responder_needs_its_own_target_and_the_production_call():
    call = _real_call()
    with pytest.raises(K.KitError, match="is an X71 responder: it needs its target final"):
        K.build_argv("x71_c0", 7111, call)
    with pytest.raises(K.KitError, match="x71_c0: its target is rb_x35_c0_s3501, not 'rb_x35_ce3dswap_s7321'"):
        K.build_argv("x71_c0", 7111, call, None, _banked_target("swap"))
    with pytest.raises(K.KitError, match="reads the production call record, got side 'toy'"):
        K.build_argv("x71_c0", 7111, toy_call(), None, _banked_target("c0"))
    with pytest.raises(K.KitError, match="is not an X71 responder"):
        K.x71_target("c0", 3501, call, models_dir="/nowhere")
    with pytest.raises(X.X71Error, match="is not an X71 responder arm"):
        X.tag_of("x70")


# ----------------------------------------------------------- the target: judged in full, or a declared placeholder
def _final(root: Path, run: str, body: bytes = b"final", **md) -> Path:
    d = root / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "final_model.zip").write_bytes(body)
    if md:
        (d / "metadata.json").write_text(json.dumps(md))
    return d / "final_model.zip"


def test_x70s_missing_target_is_a_placeholder_under_plan_only(tmp_path):
    """X70's final does not exist yet. A launch describe REFUSES; ``arm_chain plan``'s (``describe --plan``) builds
    the argv from the DECLARED PLACEHOLDER and says so (the pattern X35-A3's phase 2 used for its phase 1)."""
    call, md = _real_call(), str(tmp_path)
    with pytest.raises(K.KitError, match=r"rb_x35_x70ent0_s7331: no final_model.zip under .*`describe --plan` builds "
                                         r"it from a declared placeholder, never a launch"):
        K.x71_target("x71_x70", 7111, call, models_dir=md)
    t = K.x71_target("x71_x70", 7111, call, models_dir=md, plan=True)
    assert t == {"run": X70_RUN, "path": str(tmp_path.resolve() / X70_RUN / "final_model.zip"), "step": X70_FLOOR,
                 "sha256": X.PLACEHOLDER_SHA, "standin": False, "placeholder": True}
    assert X70_FLOOR == call["pstar"]["step"] + K.ARMS["x70"].length
    assert len(X.PLACEHOLDER_SHA) != 64 and X.PLACEHOLDER_SHA.startswith("<")      # a marker, never a hash
    for seed in SEEDS:
        d = K.describe_for_plan("x71_x70", seed, call, stage=str(STAGE), models_dir=md, target=t)
        assert d["phase1"]["mode"] == X.PLACEHOLDER_MODE == "TARGET-PLACEHOLDER"
        assert X70_RUN in d["phase1"]["note"] and "G5" in d["phase1"]["note"]
        argv = d["argv"]
        assert argv[argv.index("--steps") + 1] == str(X70_FLOOR + 5_000_000) == "305000170"
        assert argv[argv.index("--model") + 1] == argv[argv.index("--exploiter") + 1] == t["path"]
        assert argv[argv.index("--run-name") + 1] == f"rb_x35_x71_x70_s{seed}"
        assert d["parent_run_dir"] == str(tmp_path.resolve() / X70_RUN)
    # the CLI, as the launch script and the chain's plan call it
    r = str(STAGE / "reference_population_production.json")
    base = ["describe", "--arm", "x71_x70", "--seed", "7111", "--call", r, "--models-dir", md]
    assert K.main(base) == 2
    assert K.main(base + ["--plan", "--stage", str(STAGE), "--field", "run_name"]) == 0


def test_a_placeholder_is_never_a_launch_argv(tmp_path, capsys):
    call = _real_call()
    ph = K.x71_target("x71_x70", 7112, call, models_dir=str(tmp_path), plan=True)
    with pytest.raises(K.KitError, match="a PLACEHOLDER target is for planning only, never a launch argv"):
        K.build_argv("x71_x70", 7112, call, None, ph)
    with pytest.raises(K.KitError, match="for planning only"):
        K.describe("x71_x70", 7112, call, None, ph)
    with pytest.raises(X.X71Error, match="for planning only"):
        X.build_x71_argv("x71_x70", 7112, ph, run_name="rb_x35_x71_x70_s7112")
    assert K.describe("x71_x70", 7112, call, None, ph, plan=True)["argv"][-1] == "rb_x35_x71_x70_s7112"
    # the launch script's own call (no --plan) prints no argv for it
    capsys.readouterr()
    assert K.main(["describe", "--arm", "x71_x70", "--seed", "7112", "--models-dir", str(tmp_path), "--field",
                   "argv", "--call", str(STAGE / "reference_population_production.json")]) == 2
    got = capsys.readouterr()
    assert got.out == "" and "REFUSED" in got.err


def test_a_banked_target_never_takes_a_placeholder(tmp_path, monkeypatch):
    """C0's, X67's and the swap's finals are BANKED (a registered sha256 and step): a missing one is a refusal even
    under ``plan``, and one that exists is judged by its hash and step (PREREG G2)."""
    call, md = _real_call(), str(tmp_path)
    for arm in ARMS[:3]:
        with pytest.raises(X.X71Error, match="is BANKED .*it takes no placeholder"):
            X.placeholder_target(arm, md, X70_FLOOR)
        for plan in (False, True):
            with pytest.raises(K.KitError, match="no final_model.zip under") as exc:
                K.x71_target(arm, 7111, call, models_dir=md, plan=plan)
            assert "placeholder" not in str(exc.value)
    z = _final(tmp_path, "rb_x35_c0_s3501", b"not the registered file")
    monkeypatch.setattr(K, "_final_step", lambda p: 300_061_940)
    with pytest.raises(K.KitError, match=r"not the registered 09a6a2df4850 \(PREREG G2: the file moved\)"):
        K.x71_target("x71_c0", 7111, call, models_dir=md, plan=True)
    sha = hashlib.sha256(z.read_bytes()).hexdigest()
    monkeypatch.setattr(X, "TARGET_SHA256", dict(X.TARGET_SHA256, c0=sha))
    assert K.x71_target("x71_c0", 7111, call, models_dir=md) == {
        "run": "rb_x35_c0_s3501", "path": str(z.resolve()), "step": 300_061_940, "sha256": sha, "standin": False,
        "placeholder": False}
    monkeypatch.setattr(K, "_final_step", lambda p: 300_061_941)
    with pytest.raises(K.KitError, match=r"final at 300,061,941, not the registered 300,061,940 \(PREREG §2\)"):
        K.x71_target("x71_c0", 7111, call, models_dir=md)


def test_x70s_real_final_is_judged_by_its_record(tmp_path, monkeypatch):
    """Once X70's final exists the SAME call judges it in full, with or without ``plan`` (PREREG G5): the arm's
    registered length, a recorded ``ent_coef`` of 0 and no magnet. There is no placeholder left to fall back to."""
    call, md = _real_call(), str(tmp_path)
    step = [X70_FLOOR + 21_185]
    monkeypatch.setattr(K, "_final_step", lambda p: step[0])

    def judged(plan: bool = False) -> dict:
        return K.x71_target("x71_x70", 7111, call, models_dir=md, plan=plan)

    z = _final(tmp_path, X70_RUN, ent_coef=0.0, magnet_kl_coef=0.0)
    sha = hashlib.sha256(z.read_bytes()).hexdigest()
    want = {"run": X70_RUN, "path": str(z.resolve()), "step": step[0], "sha256": sha, "standin": False,
            "placeholder": False}
    assert judged() == want == judged(plan=True)
    d = K.describe_for_plan("x71_x70", 7111, call, stage=str(STAGE), models_dir=md, target=want)
    assert "phase1" not in d and d["argv"][d["argv"].index("--steps") + 1] == str(step[0] + 5_000_000)
    assert K.describe("x71_x70", 7111, call, None, want)["argv"] == d["argv"]         # now a launch argv
    _final(tmp_path, X70_RUN, ent_coef=0)                           # the magnet's field absent: no magnet
    assert judged()["placeholder"] is False
    for md_doc, needle in (({"ent_coef": 0.05}, r"recorded ent_coef 0.05, not 0 \(G5\)"),
                           ({"ent_coef": 0.0125}, r"recorded ent_coef 0.0125, not 0 \(G5\)"),
                           ({"magnet_kl_coef": 0.0}, r"recorded ent_coef None, not 0 \(G5\)"),
                           ({"ent_coef": 0.0, "magnet_kl_coef": 0.05},
                            r"recorded magnet_kl_coef 0.05, not 0 or absent \(G5\)")):
        _final(tmp_path, X70_RUN, **md_doc)
        for plan in (False, True):
            with pytest.raises(K.KitError, match=needle):
                judged(plan)
    (tmp_path / X70_RUN / "metadata.json").unlink()
    with pytest.raises(K.KitError, match="metadata.json is unreadable"):
        judged(plan=True)
    _final(tmp_path, X70_RUN, ent_coef=0.0)
    step[0] = X70_FLOOR - 1
    with pytest.raises(K.KitError, match=r"final at 300,000,169 < 300,000,170 \(the arm did not reach its "
                                         r"registered length\)"):
        judged(plan=True)


# ------------------------------------------------------------------------------------------ the stage and the pin
def _fill_fp32(stage: Path, sha: str = "c" * 40) -> None:
    (stage / B.SIDE_REQUIRED_FILE.format(side="production")).write_text(f"# header\n\n{sha} FP32_ALT the fp32 build\n")


def test_the_stage_admits_the_eight_registered_forks_in_the_registered_order(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    (stage / B.READS_DIR).mkdir(parents=True)
    st = str(stage)
    prereg = tmp_path / "x71" / "PREREG.md"
    monkeypatch.setattr(B, "X71_PREREG", str(prereg))               # absolute: repo_path() returns it unchanged
    assert [f for _, forks in B.X71_STAGES for f in forks] == ORDER
    assert [n for n, _ in B.X71_STAGES] == ["X71", "X71_X70"]
    assert not {a for _, forks in B.STAGES + B.CE3D_STAGES + B.X70_STAGES for a, _ in forks} & set(ARMS)
    assert not set(ARMS) & set(B.SIDE_PINNED_ARMS) and not set(ARMS) & set(B.PREREG_GATED_ARMS)
    assert not set(ARMS) & set(B.READ_LABEL) and set(ARMS) <= K.NO_POOL_ARMS
    assert B.stage_status(st, "x71_c0", 7111)[0] == B.PENDING                   # AM-8: no FP32_ALT named
    _fill_fp32(stage)
    status, why = B.stage_status(st, "x71_c0", 7111)
    assert status == B.PENDING and "PREREG" in why, why
    prereg.parent.mkdir()
    prereg.write_text("# X71\n")
    for arm, seed in ORDER:
        status, why = B.stage_status(st, arm, seed)
        name = "X71_X70" if arm == "x71_x70" else "X71"
        assert status == B.ALLOWED and f"(stage {name}, X71 PREREG present)" in why, why
    for arm, seed in (("x71_c0", 7113), ("x71_swap", 7321), ("x71_x70", 7331)):
        status, why = B.stage_status(st, arm, seed)
        assert status == B.REFUSED and "its seeds for" in why and "[7111, 7112]" in why and "one look" in why, why
    with_readers = B.side_pinned_run_names(readers=True)
    assert {n: v for n, v in with_readers.items() if v[0] in ARMS} == {
        f"rb_x35_x71_{a[4:]}_s{s}": (a, s) for a in ARMS for s in SEEDS}
    assert not [n for n, (a, _) in B.side_pinned_run_names().items() if a in ARMS]


def test_the_committed_prereg_is_where_the_batch_looks_and_the_registration_holds_no_go():
    assert repo_path(B.X71_PREREG).is_file() and repo_path(B.X71_PREREG).parent == DIR
    assert B.X71_PREREG == X.X71_PREREG
    for name in ("chain_items_x71.json", "kit_checks_x71.txt", "arm_chain_plan_x71.txt", "drive_read_x71.sh",
                 "read_path_smoke_x71.txt", "read_x71.py", "x71_rule.py"):
        assert (DIR / name).is_file(), name
    assert not [p.name for p in DIR.rglob("*") if p.name.upper().startswith(("GO", "VERDICT"))]


def test_the_responders_train_at_the_swaps_pin_under_x71s_registration():
    for arm in ARMS:
        assert B.ARM_PINS[arm] == B.CE3D_PIN == X.PIN and B.ARM_PIN_AMENDMENT[arm] == "X71 PREREG §3"
    assert B.ARM_PIN_AMENDMENT["x70"] == "X70 PREREG §3" and B.ARM_PIN_AMENDMENT["ce3d_swap"] == "CE-3d PREREG §4"


@needs_pin
def test_the_pin_check_passes_the_training_pin_and_refuses_every_other_for_a_responder():
    root = str(repo_root())
    for arm in ARMS:
        ok, lines = B.arm_pin_check(arm, B.CE3D_PIN, SIDE_PIN, root)
        assert ok and lines[0].startswith("X71 PREREG §3: ") and "exactly the 3 enumerated path(s)" in lines[0], lines
        for pin in (SIDE_PIN, B.CE3D_MAGNET_PIN):   # the plain production pin; the first CE-3d pin
            ok, lines = B.arm_pin_check(arm, pin, SIDE_PIN, root)
            assert not ok and lines[0].startswith("X71 PREREG §3: "), lines
            assert f"{arm} (an arm at the CE-3d training pin) launches only at CE3D_PIN" in lines[0], lines
    ok, lines = B.arm_pin_check("x70", B.CE3D_PIN, SIDE_PIN, root)             # X70's lines are unchanged
    assert ok and lines[0].startswith("X70 PREREG §3: "), lines


# ------------------------------------------------------------------------------------------ the draft chain items
def test_the_draft_chain_items_are_the_kits_forks_at_the_pin_in_the_registered_order():
    """``chain_items_x71.json`` (the Training session's hand-off): eight arm items in the registered order, each the
    kit's run at CE3D_PIN, every checkpoint KEPT, the banner the hyperparameter table's row at the TYPED entropy
    coefficient and its VOID lines the rows a target's own record would print, plus the magnet's attach line."""
    doc = json.loads((DIR / "chain_items_x71.json").read_text())
    assert doc["x_pin"]["sha"] == B.CE3D_PIN and doc["x_pin"]["base"] == B.CE3D_MAGNET_PIN
    assert doc["x_pin"]["base_base"] == SIDE_PIN
    assert "GO" not in doc and "go" not in doc and "NO GO" in doc["x_what"]
    assert "TARGET-PLACEHOLDER" in doc["x_x70_boundary"] and "e2c4498fb" in doc["x_x70_boundary"]
    assert "drive_read_x71.sh <1|2> <WORK_DIR> <PLAY_DIR>" in doc["x_read"]
    items = doc["items"]
    assert [(i["arm"], i["seed"]) for i in items] == ORDER
    for n, it in enumerate(items):
        parsed = chain_spec._arm(it, f"items[{n}]")                 # the chain file's own loader takes the item
        assert (parsed.arm, parsed.seed, parsed.pin, parsed.throughput) == (it["arm"], it["seed"], B.CE3D_PIN, None)
        assert it["kind"] == "arm" and it["run"] == K.run_name(it["arm"], it["seed"])
        assert (it["arm"], it["seed"]) in [f for _, forks in B.X71_STAGES for f in forks]
        assert it["thin"] == "none" and "KEPT" in it["thin_reason"] and it["flags"] == ["--thinning"]
        assert it["on_void"] == "stop" and it["on_throughput_fail"] == "page" and "throughput" not in it
        assert it["disk_need_gib"] == 8.0                           # the dry run's 7.91, rounded up
        assert it["go_by"].startswith("<orchestrator, date>") and X.target_run(it["arm"]) in it["go_by"]
        assert "verify" not in it
        b = it["banner"]
        assert b == BANNER
        assert b["expect"] in ROW_TYPED and not any(v in ROW_TYPED for v in b["void"])
        for row in ROWS_ELSEWHERE:                                  # a target's own coefficient: VOID, never the banner
            assert b["expect"] not in row and any(v in row for v in b["void"]), row
        assert not any(v in b["expect"] for v in b["void"])
    allowed = doc["off_main_pins"]
    assert set(allowed) == {i["pin"] for i in items} == {B.CE3D_PIN}
    assert doc["x_pin"]["tag"] in allowed[B.CE3D_PIN]                # the reason names the ref that holds the pin
    assert chain_spec._off_main_pins(doc) == allowed                 # the form the chain file's loader accepts
    assert all("off_main_pins" not in i for i in items)              # chain-level, never an arm field


def test_the_banked_plan_and_kit_checks_say_what_was_executed():
    plan = (DIR / "arm_chain_plan_x71.txt").read_text()
    read1, rest = plan.split("\n# a variant:", 1)
    assert read1.count("the kit agrees — FORK of") == 6 and read1.rstrip().endswith("PLAN OK\nrc=0")
    for tag, (run, step) in EXISTING.items():
        assert read1.count(f"FORK of {MODELS}/{run}, +5,000,000 steps, --steps {step + 5_000_000:,}") == 2
    assert "REFUSED: seed 7113 is not registered for x71_swap ([7111, 7112])" in rest
    assert f"at pin {SIDE_PIN}: main.checkargs --pin exits 3" in rest and "--magnet-kl-coef" in rest
    assert "the kit names arm x71_x67 s7111's run 'rb_x35_x71_x67_s7111', not 'rb_x35_x71_c0_s7111x'" in rest
    assert rest.count(f"at pin {B.CE3D_PIN}: is not an ancestor of origin/main, and the chain file's off_main_pins "
                      f"does not name it") == 6
    assert rest.count("PLAN REFUSED") == 4 and rest.count("rc=1") == 4          # the four variants, nothing else
    x70 = rest[rest.index("X70's target rb_x35_x70ent0_s7331 has no final yet"):]  # planned WITH --check-pins
    assert "> --check-pins --describe\n# X70's target" in rest
    assert x70.count(f"FORK of {MODELS}/{X70_RUN}, +5,000,000 steps, --steps 305,000,170; its target has no final "
                     f"yet: the argv is built from a declared PLACEHOLDER") == 2
    assert x70.count("its argv is NOT run through the pin's parser yet") == 2 and "REFUSED" not in x70
    assert x70.rstrip().endswith("PLAN OK\nrc=0")
    checks = (DIR / "kit_checks_x71.txt").read_text()
    for arm, seed in ORDER:
        name = "X71_X70" if arm == "x71_x70" else "X71"
        assert f"ALLOWED: {arm} s{seed}: no unmet condition (stage {name}, X71 PREREG present)" in checks
    assert checks.count("launches only at CE3D_PIN 9e97cad5709d, not ") == 2
    assert checks.count("IDENTICAL (26 tokens; run rb_x35_x71_") == 6
    assert checks.count("names no team corpus") == 4 and "'mode': 'TARGET-PLACEHOLDER'" in checks
    for run in ("rb_x35_c0_s3501", "rb_x35_x67ent_s6701", "rb_x35_ce3dswap_s7321"):
        assert f"k9b-batch: {run}: K9(b) arm end PASS" in checks
    assert "HOLD: a training run is LIVE" in checks and "NOT EXECUTED (PREREG G3 stays open)" in checks
    smoke = (DIR / "read_path_smoke_x71.txt").read_text()
    assert "NOT AN X71 CELL" in smoke and "HOLD: plan REFUSED" in smoke and "rust_eval" in smoke


# ------------------------------------------------------------------------------------------ the read's driver
DRIVER = DIR / "drive_read_x71.sh"
MEM_CAP_STUB = """#!/usr/bin/env bash
# the test's mem_cap: record the call, drop the options and the cap, run the command
echo "$*" >> "$X71_STUB_CALLS"
while [ $# -gt 0 ]; do case "$1" in --name|--max-run-s) shift 2 ;; --*) shift ;; *) break ;; esac; done
shift
exec "$@"
"""
#: A reader with ``read_x71.py``'s verbs and exit codes over a script of outcomes: ``play`` pops the next exit code
#: from ``script`` (3 banks one cell and says partial; 0 banks the last one), ``plan`` and ``read`` exit as told.
STUB_READER = """import json, os, sys
from pathlib import Path
state = Path(os.environ["X71_STUB_STATE"])
doc = json.loads(state.read_text())
verb, read = sys.argv[1], sys.argv[2]
doc["calls"].append(sys.argv[1:])
rc = 0
if verb == "plan":
    rc = doc.get("plan_rc", 0)
    print(json.dumps({"read": read}))
elif verb == "play":
    ledger = Path(sys.argv[sys.argv.index("--ledger") + 1])
    assert sys.argv[sys.argv.index("--max-chunks") + 1] == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and "GEN3AI_TEST_ALLOW_GPU" not in os.environ
    rc = doc["script"].pop(0) if doc["script"] else 0
    if rc in (0, 3) and not doc.get("bank_nothing"):
        ledger.mkdir(parents=True, exist_ok=True)
        n = len(list(ledger.glob("*__*.json")))
        if n < doc["chunks"]:
            (ledger / f"t{n}__k0.json").write_text("{}")
else:
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    rc = doc.get("read_rc", 0)
    if rc == 0:
        out.write_text(json.dumps({"cls": "SYNTHETIC"}))
        print(json.dumps({"cls": "SYNTHETIC"}))
state.write_text(json.dumps(doc))
sys.exit(rc)
"""


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                          capture_output=True, text=True).stdout.strip()


class _Rig:
    """A scratch venue for the driver: a play checkout (a one-commit git repository), a stub ``scripts/ops`` and a
    work directory, all under ``tmp_path``."""

    def __init__(self, tmp_path: Path, reader: Path):
        self.work, self.play, self.ops = tmp_path / "work", tmp_path / "play", tmp_path / "ops"
        self.calls, self.state, self.reader = tmp_path / "mem_cap_calls.txt", tmp_path / "state.json", reader
        (self.play / "src").mkdir(parents=True)
        (self.play / "src" / "a.txt").write_text("a\n")
        _git(self.play, "init", "-q")
        _git(self.play, "add", "-A")
        _git(self.play, "commit", "-q", "-m", "one")
        self.ops.mkdir()
        stub = self.ops / "mem_cap.sh"
        stub.write_text(MEM_CAP_STUB)
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)

    def run(self, read: str = "1", work: Path = None, **env: str):
        ambient = {k: v for k, v in os.environ.items() if not k.startswith("X71_")}     # no override leaks in
        full = dict(ambient, GEN3AI_PYTHON=sys.executable, X71_READER=str(self.reader), X71_OPS=str(self.ops),
                    X71_STUB_CALLS=str(self.calls), X71_STUB_STATE=str(self.state), **env)
        got = subprocess.run(["bash", str(DRIVER), read, str(work or self.work), str(self.play)], env=full,
                             capture_output=True, text=True, timeout=600)
        return got.returncode, got.stdout + got.stderr

    def script(self, **doc) -> None:
        self.state.write_text(json.dumps(dict({"calls": [], "script": [], "chunks": 9}, **doc)))

    def verbs(self) -> list:
        return [c[0] for c in json.loads(self.state.read_text())["calls"]]

    def log(self) -> str:
        return (self.work / "run.log").read_text()


@pytest.fixture()
def rig(tmp_path):
    reader = tmp_path / "stub_reader.py"
    reader.write_text(STUB_READER)
    return _Rig(tmp_path, reader)


def test_the_driver_plays_one_chunk_a_slot_until_all_are_banked_then_reads_once(rig):
    rig.script(script=[3] * 8 + [0])
    rc, out = rig.run()
    assert rc == 0, out
    assert rig.verbs() == ["plan"] + ["play"] * 9 + ["read"]
    assert len(list((rig.work / "ledger").glob("*__*.json"))) == 9
    assert json.loads((rig.work / "verdict_read1.json").read_text()) == {"cls": "SYNTHETIC"}
    assert (rig.work / "play_commit").read_text().strip() == _git(rig.play, "rev-parse", "HEAD")
    calls = rig.calls.read_text().splitlines()
    assert len(calls) == 9                                          # one heavy slot a chunk; plan and read take none
    assert all(c.startswith("--name x71_read1 --max-run-s 7200 8 ") and " play 1 --ledger " in c and
               c.endswith("--max-chunks 1") for c in calls), calls
    log = rig.log()
    assert "DONE (read 1)" in log and log.count("OVERRIDE X71_") == 2 and "banked 8 chunk(s)" in log
    # a second invocation plays and reads nothing: the verdict is written once
    rc, out = rig.run()
    assert rc == 0 and "DONE before" in out and rig.verbs().count("read") == 1 and rig.verbs().count("play") == 9
    # read 2 on the SAME work dir: the nine banked cells stay, three more are played, its own verdict file
    rig.script(script=[3, 3, 0], chunks=12)
    rc, out = rig.run("2")
    assert rc == 0, out
    assert rig.verbs() == ["plan", "play", "play", "play", "read"]
    assert len(list((rig.work / "ledger").glob("*__*.json"))) == 12
    assert (rig.work / "verdict_read2.json").is_file() and (rig.work / "verdict_read1.json").is_file()


def test_the_driver_resumes_a_partly_banked_ledger_without_replaying(rig):
    rig.script(script=[3, 3, 7])
    rc, out = rig.run()
    assert rc == 4 and "HOLD: a chunk failed with exit 7" in out
    assert len(list((rig.work / "ledger").glob("*__*.json"))) == 2 and not (rig.work / "verdict_read1.json").exists()
    rig.script(script=[3] * 6 + [0])
    rc, out = rig.run()
    assert rc == 0 and rig.verbs() == ["plan"] + ["play"] * 7 + ["read"], out
    assert len(list((rig.work / "ledger").glob("*__*.json"))) == 9


@pytest.mark.parametrize("doc,rc,needle,verbs", [
    ({"plan_rc": 2}, 3, "HOLD: plan REFUSED", ["plan"]),
    ({"script": [3, 2]}, 3, "HOLD: play REFUSED", ["plan", "play", "play"]),
    ({"script": [3, 89]}, 8, "ran past its bound of 7200s", ["plan", "play", "play"]),
    ({"script": [3, 3], "bank_nothing": True}, 8, "banked nothing", ["plan", "play"]),
    ({"script": [0], "read_rc": 2}, 6, "HOLD: the read failed", ["plan", "play", "read"]),
])
def test_the_driver_holds_on_each_refusal_and_never_writes_a_verdict(rig, doc, rc, needle, verbs):
    rig.script(**doc)
    got, out = rig.run()
    assert got == rc and needle in out, out
    assert rig.verbs() == verbs and not (rig.work / "verdict_read1.json").exists()
    assert not (rig.work / "plan_read1.json.tmp").exists()


def test_the_driver_holds_one_ledger_to_one_playing_commit_and_one_lane(rig, tmp_path):
    """The PREREG registers no playing commit; the driver refuses to mix two in one ledger, a dirty play tree, a
    work directory inside the checkout (a verdict is never written beside the registration) and a bad usage."""
    rig.script(script=[3, 7])
    assert rig.run()[0] == 4
    first = (rig.work / "play_commit").read_text().strip()
    (rig.play / "src" / "a.txt").write_text("b\n")
    rig.script(script=[0])
    rc, out = rig.run()
    assert rc == 5 and "uncommitted changes under src/ or data/" in out and rig.verbs() == []
    _git(rig.play, "commit", "-q", "-am", "two")
    rc, out = rig.run()
    assert rc == 5 and f"this ledger's cells were played at {first}" in out and rig.verbs() == []
    assert (rig.work / "play_commit").read_text().strip() == first
    rc, out = rig.run(work=DIR / "scratch_work")
    assert rc == 5 and "is inside the checkout" in out and not (DIR / "scratch_work").exists()
    for args in (["3", str(rig.work), str(rig.play)], ["1", str(rig.work)], []):
        got = subprocess.run(["bash", str(DRIVER), *args], capture_output=True, text=True, timeout=60)
        assert got.returncode == 2 and "usage: drive_read_x71.sh <1|2> <WORK_DIR> <PLAY_DIR>" in got.stderr
    # a second driver on the same work directory is held by the lock
    lock = tmp_path / "work" / "lock"
    holder = subprocess.Popen(["flock", "-n", str(lock), "sleep", "30"])
    try:
        for _ in range(200):                                        # the holder has the lock once flock exec'd sleep
            if subprocess.run(["flock", "-n", str(lock), "true"]).returncode != 0:
                break
        rc, out = rig.run()
        assert rc == 5 and "another driver holds" in out
    finally:
        holder.kill()
        holder.wait()


def test_the_driver_runs_the_frozen_reader_under_rule_5bs_shape():
    text = DRIVER.read_text()
    assert 'READER=${X71_READER:-$HERE/read_x71.py}' in text and 'OPS=${X71_OPS:-$REPO/scripts/ops}' in text
    assert 'LOW=(nice -n 19 taskset -c "$CPUS")' in text and "utils.core_split agent-cpus" in text
    assert 'export PYTHONPATH=$PLAY_DIR/src CUDA_VISIBLE_DEVICES=""' in text and "unset GEN3AI_TEST_ALLOW_GPU" in text
    assert '"${LOW[@]}" "$OPS/mem_cap.sh" --name "x71_read$READ" --max-run-s "$BOUND" "$GB"' in text
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("timeout " in ln or "gpu_lock" in ln or "--device" in ln for ln in code)   # no outer timeout, CPU
    assert os.access(DRIVER, os.X_OK)


WRAPPED_READER = """import importlib.util, json, os, sys
from pathlib import Path
from types import SimpleNamespace
frozen = Path(os.environ["X71_FROZEN_READER"])
spec = importlib.util.spec_from_file_location("x71_frozen_read", frozen)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
from agents.training import best_response_gap as E
from agents.training import untaught_meter as um
world = json.loads(Path(os.environ["X71_WORLD"]).read_text())
def read_exploiter(ref, _teamsets):
    r = world["records"][Path(ref).name]
    end = SimpleNamespace(wins=160, games=256, step=r["fork_step"] + 5_000_000)
    return SimpleNamespace(run=Path(ref).name, target_file=r["target_file"], fork_step=r["fork_step"],
                           archetype="offense", membership="exact", teams=r["teams"], pooled=lambda: (300, 512),
                           endpoint=lambda: end)
def play_chunk(ref_path, target_path, teams, games_per_team, *, label):
    w = round(world["rates"].get(label, 0.5) * games_per_team)
    got = {t.key: um.Cell(wins=w, losses=games_per_team - w, finished=games_per_team, attempted=games_per_team,
                          transport="rust_eval") for t in teams}
    return got, {t.key: [] for t in teams}, "rust_eval"
E.read_exploiter, E.load_teamsets = read_exploiter, (lambda path=None: {})
reader.R.TARGET_SHA256 = {t: None for t in reader.R.TARGETS}
reader.play_chunk = play_chunk
sys.exit(reader.main())
"""


@pytest.mark.slow
def test_the_driver_takes_the_frozen_reader_end_to_end_over_a_recorded_player(tmp_path):
    """The read path end to end, without a game: the driver, the FROZEN ``read_x71.py`` (its plan, its ledger, its
    verdict through ``x71_rule.classify``) on synthetic run directories, the playing function alone replaced by a
    recorded one. Read 1's nine cells, then read 2's three on the same ledger."""
    rule = _load("x71_rule")
    models, t_step, records = tmp_path / "models", 300_000_000, {}
    for tag, (_arm, run, _lever) in rule.TARGETS.items():
        t = _write_run(models, run, steps=t_step)
        for s in rule.SEEDS:
            name = rule.run_name(tag, s)
            _write_run(models, name, steps=t_step + rule.L_RESP + 61_000, seed=s)
            records[name] = {"target_file": str(t / "final_model.zip"), "fork_step": t_step,
                             "teams": [str(repo_path(x)) for x in TEAMS]}
    rates = {f"{t}_{c}": r for t in rule.TARGETS for c, r in (("k0", 0.475), ("r7111", 0.565), ("r7112", 0.565))}
    rates.update(x67_r7111=0.665, x67_r7112=0.665)
    (tmp_path / "world.json").write_text(json.dumps({"records": records, "rates": rates}))
    reader = tmp_path / "wrapped_reader.py"
    reader.write_text(WRAPPED_READER)
    rig = _Rig(tmp_path, reader)
    env = {"X71_MODELS_DIR": str(models), "X71_WORLD": str(tmp_path / "world.json"),
           "X71_FROZEN_READER": str(DIR / "read_x71.py"), "PYTHONPATH_PLAY": str(repo_root() / "src")}
    rc, out = _run_with_src(rig, "1", env)
    assert rc == 0, out + rig.log()
    v = json.loads((rig.work / "verdict_read1.json").read_text())
    assert v["cls"] == {"x67": "MORE-EXPLOITABLE", "swap": "NOT-MORE-BY-BAR"} and v["transport"] == "rust_eval"
    assert v["cells"]["c0"]["k0"] == {"wins": 3_800, "finished": 8_000, "attempted": 8_000, "ties": 0}
    assert sorted(p.name for p in (rig.work / "ledger").iterdir()) == sorted(
        f"{t}__{c}.json" for t in ("c0", "x67", "swap") for c in ("k0", "r7111", "r7112"))
    assert len(rig.calls.read_text().splitlines()) == 9
    rc, out = _run_with_src(rig, "2", env)
    assert rc == 0, out + rig.log()
    assert json.loads((rig.work / "verdict_read2.json").read_text())["cls"] == {"x70": "NOT-MORE-BY-BAR"}
    assert len(rig.calls.read_text().splitlines()) == 12            # read 1's nine cells were not replayed
    assert json.loads((rig.work / "verdict_read1.json").read_text()) == v


def _run_with_src(rig: "_Rig", read: str, env: dict):
    """Run the driver with the rig's play checkout standing for THIS checkout's code: the driver exports
    ``PYTHONPATH=PLAY_DIR/src``, so the scratch repository's ``src`` is a link to this checkout's."""
    src = rig.play / "src"
    if not src.is_symlink():
        _git(rig.play, "rm", "-rq", "src")
        src.symlink_to(Path(env["PYTHONPATH_PLAY"]), target_is_directory=True)
        _git(rig.play, "add", "src")
        _git(rig.play, "commit", "-q", "-m", "this checkout's src")
    return rig.run(read, **{k: v for k, v in env.items() if k != "PYTHONPATH_PLAY"})
