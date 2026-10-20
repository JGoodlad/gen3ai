"""X35 batch 1's gates (``main.x35.batch``, the orchestrator's rulings of 2026-10-19 on the adversarial review and of
2026-10-20 after the GPU battery): AM-1's one pin PER SIDE (AM-8), the side's own required commit (production's
FP32_ALT), the POOL record, AM-2's variance gate G1, AM-9 / AM-3 / AM-4's conditional stages and AM-6's K9(b) arm-end
read. Synthetic files only; every case fails on revert of the rule it names, and every constructed number sits well
clear of its boundary except the deliberate rule-8 cases (standing rule 8)."""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import pytest

from main.x35 import batch as B
from main.x35 import kit as K
from utils.paths import repo_path

PIN = "a" * 40
OTHER = "b" * 40


def _run(models: Path, name: str, *hashes: str, final: bool = False) -> Path:
    d = models / name
    d.mkdir(parents=True)
    if hashes:
        meta = {"git_hash": hashes[0], "pin_history": [{"git_hash": h, "pin_source": "pin_commit"} for h in hashes]}
        (d / "metadata.json").write_text(json.dumps(meta))
    if final:
        (d / "final_model.zip").write_bytes(b"z")
    return d


# ------------------------------------------------------------------------------------------------------ AM-1 / AM-8
PROD, TOY = "production", "toy"
FP32 = "FP32_ALT"


def _pin_file(stage: Path, side: str) -> str:
    return B.pin_path(str(stage), side)


def test_the_first_go_of_a_side_writes_its_pin_once_and_every_later_launch_of_it_must_equal_it(tmp_path):
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    models.mkdir()
    ok, lines = B.pin_check(PIN, str(stage), str(models), TOY)
    assert ok and "the first toy GO writes" in lines[0]
    with pytest.raises(B.BatchError, match="needs --t62 <NA_TOY>"):
        B.pin_write_if_absent(PIN, str(stage), TOY, None)
    assert "WRITTEN" in B.pin_write_if_absent(PIN, str(stage), TOY, "NA_TOY")
    rec = B.read_pin_file(_pin_file(stage, TOY), TOY)
    assert rec is not None and rec["sha"] == PIN and rec["t62_verdict"] == "NA_TOY" and rec["side"] == TOY
    assert Path(_pin_file(stage, TOY)).name == "x35_pin_batch1_toy.txt"
    assert B.pin_check(PIN, str(stage), str(models), TOY)[0]
    ok, lines = B.pin_check(OTHER, str(stage), str(models), TOY)          # a launch at a LATER pin
    assert not ok and "!= X35_PIN_1_toy" in lines[0]
    with pytest.raises(B.BatchError, match="written ONCE"):
        B.write_pin_file(_pin_file(stage, TOY), PIN, "NA_TOY", TOY)
    with pytest.raises(B.BatchError, match="!= X35_PIN_1_toy"):
        B.pin_write_if_absent(OTHER, str(stage), TOY, "NA_TOY")


def test_am8_each_side_has_its_own_pin_and_the_production_pin_may_differ_from_the_toys(tmp_path):
    """The orchestrator's ruling (2026-10-20): ONE PIN PER SIDE. The toy pin being written must not bind production (it
    would, under the old single x35_pin_batch1.txt), and the production pin being a different sha is the point."""
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    models.mkdir()
    B.pin_write_if_absent(PIN, str(stage), TOY, "NA_TOY")
    ok, lines = B.pin_check(OTHER, str(stage), str(models), PROD)         # production: no record of its own yet
    assert ok and "the first production GO writes" in lines[0], lines
    assert "WRITTEN" in B.pin_write_if_absent(OTHER, str(stage), PROD, FP32)
    assert Path(_pin_file(stage, PROD)).name == "x35_pin_batch1_production.txt"
    assert B.pin_check(OTHER, str(stage), str(models), PROD)[0] and B.pin_check(PIN, str(stage), str(models), TOY)[0]
    ok, lines = B.pin_check(PIN, str(stage), str(models), PROD)           # the toy's pin is not production's
    assert not ok and "!= X35_PIN_1_production" in lines[0]
    ok, lines = B.pin_check(OTHER, str(stage), str(models), TOY)
    assert not ok and "!= X35_PIN_1_toy" in lines[0]


def test_am8_the_t62_token_is_the_sides_own(tmp_path):
    """Production: FP32_ALT (the battery DECIDED it; ADOPT / PENDING / INCONCLUSIVE no longer open a production GO).
    The toy: NA_TOY (no tau solve). A token of the other side, or an old one, is a refusal at write AND at the check."""
    stage = tmp_path / "stage"
    stage.mkdir()
    for side, bad in ((PROD, "NA_TOY"), (PROD, "ADOPT"), (PROD, "PENDING"), (TOY, FP32), (TOY, "ADOPT")):
        with pytest.raises(B.BatchError, match="is not a"):
            B.pin_write_if_absent(PIN, str(stage), side, bad)
        ok, lines = B.pin_check(PIN, str(stage), str(tmp_path / "m"), side, t62=bad)
        assert not ok and "is not a" in lines[0], (side, bad, lines)
    assert B.pin_check(PIN, str(stage), str(tmp_path / "m"), PROD, t62=FP32)[0]
    assert B.pin_check(PIN, str(stage), str(tmp_path / "m"), TOY, t62="NA_TOY")[0]
    assert not list(stage.iterdir())                                      # nothing was written by a refusal
    assert set(B.T62_VERDICTS) == {FP32, "NA_TOY"}


def test_a_pin_file_that_names_the_other_side_or_a_wrong_token_is_a_refusal(tmp_path):
    f = tmp_path / "x35_pin_batch1_production.txt"
    f.write_text(f"sha: {PIN}\nside: toy\nt62_verdict: NA_TOY\n")
    with pytest.raises(B.BatchError, match="says side 'toy'"):
        B.read_pin_file(str(f), PROD)
    f.write_text(f"sha: {PIN}\nside: production\nt62_verdict: ADOPT\n")
    with pytest.raises(B.BatchError, match="not a production token"):
        B.read_pin_file(str(f), PROD)
    with pytest.raises(B.BatchError, match="not one of"):
        B.pin_path(str(tmp_path), "both")


def test_every_batch1_runs_recorded_pin_must_equal_its_sides_pin(tmp_path):
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    _run(models, "rb_x35_a1arch_s3501", PIN)
    assert B.pin_check(PIN, str(stage), str(models), PROD)[0]
    _run(models, "rb_x35_c0_s3501", PIN, OTHER)                          # a restart that moved its pin
    ok, lines = B.pin_check(PIN, str(stage), str(models), PROD)
    assert not ok and any("rb_x35_c0_s3501" in x and "bbbbbbbbbbbb" in x for x in lines), lines


def test_the_other_sides_runs_are_listed_but_not_judged_by_this_sides_pin(tmp_path):
    """Toy runs at the toy pin, production launching at another pin: production's check must pass (under the single
    batch pin it would HOLD on every toy run), and each side still HOLDs on its OWN run recorded at the wrong pin."""
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    _run(models, "rb_x35t_c0_s3601", PIN)                                 # a toy run, at the toy pin
    _run(models, "rb_x35t_x47d4_s3601", PIN)
    ok, lines = B.pin_check(OTHER, str(stage), str(models), PROD)
    assert ok and any("judged at its own launches" in x and "rb_x35t_c0_s3601" in x for x in lines), lines
    ok, lines = B.pin_check(OTHER, str(stage), str(models), TOY)          # before any toy record: runs vs this launch's pin
    assert not ok and any("rb_x35t_c0_s3601" in x and "aaaaaaaaaaaa" in x for x in lines), lines
    B.pin_write_if_absent(PIN, str(stage), TOY, "NA_TOY")
    ok, lines = B.pin_check(PIN, str(stage), str(models), TOY)
    assert ok and any("2 side-pinned toy run(s) checked" in x for x in lines), lines
    _run(models, "rb_x35t_c0_s3602", OTHER)                               # a toy run on the wrong pin
    ok, lines = B.pin_check(PIN, str(stage), str(models), TOY)
    assert not ok and any("rb_x35t_c0_s3602" in x for x in lines), lines
    assert B.pin_check(OTHER, str(stage), str(models), PROD)[0]           # production does not care


def test_a_batch1_run_without_a_recorded_pin_holds_and_other_batches_are_not_judged(tmp_path):
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    _run(models, "rb_x35_c0_s3502")                                       # no metadata.json
    _run(models, "rb_x35_a3mx_s3501", OTHER)                              # a later batch's run (its own pin)
    ok, lines = B.pin_check(PIN, str(stage), str(models), PROD)
    assert not ok and any("no recorded pin" in x for x in lines)
    assert any("not a side-pinned arm" in x and "rb_x35_a3mx_s3501" in x for x in lines)
    (models / "rb_x35_c0_s3502").rename(models / "rb_x35_c0_s3502.POOLSEED_FAIL")   # moved aside: still batch 1
    ok, lines = B.pin_check(PIN, str(stage), str(models), PROD)
    assert not ok and any("POOLSEED_FAIL" in x for x in lines)


def test_a_corrupt_pin_file_is_a_refusal_not_no_pin(tmp_path):
    (tmp_path / "x35_pin_batch1_toy.txt").write_text("sha: abc\n")
    with pytest.raises(B.BatchError, match="40-hex"):
        B.pin_check(PIN, str(tmp_path), str(tmp_path / "models"), TOY)


# ------------------------------------------------------------------ AM-8: production's FP32_ALT commit, a placeholder
def _side_file(stage: Path, line: str) -> Path:
    f = stage / B.SIDE_REQUIRED_FILE.format(side=PROD)
    f.write_text("# header\n\n" + line + "\n")
    return f


def _fill_fp32(stage: Path, sha: str = "c" * 40) -> None:
    """The production side file with FP32_ALT NAMED (hex), as the follow-up commit will write it."""
    _side_file(stage, f"{sha} {FP32} the fp32 build")


def _git_repo(tmp_path: Path):
    """A throwaway repo: base <- fp32 <- head, and ``side`` branching off base (no fp32). ``(repo, {name: sha})``."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*a: str) -> str:
        r = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()

    git("init", "-q", "-b", "main")
    shas = {}
    for name in ("base", "fp32", "head"):
        (repo / name).write_text(name)
        git("add", name)
        git("commit", "-q", "-m", name)
        shas[name] = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "side", shas["base"])
    (repo / "side").write_text("side")
    git("add", "side")
    git("commit", "-q", "-m", "side")
    shas["side"] = git("rev-parse", "HEAD")
    return repo, shas


def test_am8_production_is_refused_while_the_fp32_alt_sha_is_a_placeholder(tmp_path):
    repo, shas = _git_repo(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    shipped = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17/"
                        + B.SIDE_REQUIRED_FILE.format(side=PROD))
    assert shipped.is_file()                                             # the staged file ships with a placeholder
    (stage / shipped.name).write_text(shipped.read_text())
    rows = B.read_side_required(str(stage), PROD)
    assert [n for _, n in rows] == [FP32] and not B._is_commit_sha(rows[0][0])
    ok, lines = B.required_check(str(stage), PROD, shas["head"], str(repo))
    assert not ok and "no commit recorded yet" in lines[0], lines
    filled, why = B.side_gate(str(stage), PROD)
    assert not filled and "not recorded yet" in why
    assert B.stage_status(str(stage), "a1", 3501)[0] == B.PENDING        # every production fork waits
    assert B.stage_status(str(stage), "d1", 3501)[0] == B.PENDING
    assert B.stage_status(str(stage), "toy_c0", 3601)[0] == B.ALLOWED    # the toy side has no such condition
    assert K.main(["required-check", "--side", PROD, "--pin", shas["head"], "--stage", str(stage),
                   "--repo", str(repo)]) == 1


def test_am8_production_needs_the_fp32_alt_commit_in_its_pin(tmp_path):
    repo, shas = _git_repo(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    _fill_fp32(stage, shas["fp32"])
    ok, lines = B.required_check(str(stage), PROD, shas["head"], str(repo))
    assert ok and "is in the pin" in lines[0], lines
    assert B.required_check(str(stage), PROD, shas["fp32"], str(repo))[0]  # a pin AT the commit carries it
    ok, lines = B.required_check(str(stage), PROD, shas["side"], str(repo))  # a pin that branched before the build
    assert not ok and "lacks it" in lines[0], lines
    _fill_fp32(stage, "d" * 40)                                           # a sha that is no commit of the repo
    ok, lines = B.required_check(str(stage), PROD, shas["head"], str(repo))
    assert not ok and "does not name a commit" in lines[0], lines
    _fill_fp32(stage, shas["fp32"][:12])                                  # a short sha resolves too
    assert B.required_check(str(stage), PROD, shas["head"], str(repo))[0]
    assert B.side_gate(str(stage), PROD)[0] and B.stage_status(str(stage), "a1", 3501)[0] == B.ALLOWED
    _fill_fp32(stage, shas["fp32"])
    assert K.main(["required-check", "--side", PROD, "--pin", shas["head"], "--stage", str(stage),
                   "--repo", str(repo)]) == 0


def test_am8_deleting_the_production_file_or_its_line_is_never_a_way_past_the_check(tmp_path):
    repo, shas = _git_repo(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(B.BatchError, match="is missing"):
        B.required_check(str(stage), PROD, shas["head"], str(repo))
    assert not B.side_gate(str(stage), PROD)[0] and B.stage_status(str(stage), "c0", 3501)[0] == B.PENDING
    _side_file(stage, f"{shas['fp32']} OTHER_COMMIT some other commit")
    with pytest.raises(B.BatchError, match="names no"):
        B.required_check(str(stage), PROD, shas["head"], str(repo))
    assert K.main(["required-check", "--side", PROD, "--pin", shas["head"], "--stage", str(stage),
                   "--repo", str(repo)]) == 2                              # a typed refusal, not a pass
    assert B.required_check(str(stage), TOY, shas["head"], str(repo)) == (True, ["no toy-only required commits"])


# ------------------------------------------------------------------------------------------------------ POOL
def _parent(tmp_path: Path, steps, fork: int = 280) -> Path:
    p = tmp_path / "parent"
    (p / "snapshots").mkdir(parents=True)
    for s in steps:
        (p / "snapshots" / f"snapshot_{s:012d}.zip").write_bytes(f"s{s}".encode())
    (p / "snapshots" / "summary.json").write_text("{}")
    return p


def _call(step: int = 280, sha: str = "shaP"):
    return {"pstar": {"path": "x/checkpoints/c.zip", "step": step, "sha256": sha}}


def test_the_pool_record_is_t54s_filtered_set_and_a_moved_set_holds(tmp_path):
    parent = _parent(tmp_path, [164, 200, 268, 286, 376])
    e = B.expected_pool(str(parent), 280)
    assert [s["name"] for s in e["snapshots"]] == [f"snapshot_{s:012d}.zip" for s in (164, 200, 268)]
    stage = tmp_path / "stage"
    stage.mkdir()
    ok, line = B.pool_check(str(stage), "production", _call(), str(parent))
    assert ok and "the first production GO records 3" in line and not list(stage.iterdir())   # a dry run writes nothing
    assert B.pool_check(str(stage), "production", _call(), str(parent), write=True)[0]
    assert B.recorded_pool_names(str(stage), "production") == [s["name"] for s in e["snapshots"]]
    assert B.pool_check(str(stage), "production", _call(), str(parent))[0]
    (parent / "snapshots" / "snapshot_000000000164.zip").unlink()       # the parent's pool evicted a kept node
    ok, line = B.pool_check(str(stage), "production", _call(), str(parent))
    assert not ok and "gone ['snapshot_000000000164.zip']" in line
    (parent / "snapshots" / "snapshot_000000000164.zip").write_bytes(b"different bytes")
    ok, line = B.pool_check(str(stage), "production", _call(), str(parent))
    assert not ok and "changed ['snapshot_000000000164.zip']" in line
    ok, line = B.pool_check(str(stage), "production", _call(sha="other"), str(parent))
    assert not ok and "start node" in line


def test_the_pool_metadata_is_recorded_but_never_compared(tmp_path):
    parent = _parent(tmp_path, [164, 200])
    stage = tmp_path / "stage"
    stage.mkdir()
    B.pool_check(str(stage), "toy", _call(), str(parent), write=True)
    (parent / "snapshots" / "summary.json").write_text('{"win_rate_vs_bots": 0.9}')   # F31: the parent's scalars move
    assert B.pool_check(str(stage), "toy", _call(), str(parent))[0]


def test_the_post_spawn_check_compares_the_seeded_names_with_the_record(tmp_path):
    run = tmp_path / "rb_x35_c0_s3501"
    (run / "snapshots").mkdir(parents=True)
    rec = {"schema": 2, "fork_step": 280, "snapshots": ["snapshot_000000000164.zip"], "dropped_steps": []}
    (run / "snapshots" / "pool_seed.json").write_text(json.dumps(rec))
    line = "[step filter <= 280: none later]"
    assert K.check_pool_seed("c0", str(run), _call(), line, 280, ["snapshot_000000000164.zip"]) == ("PASS", [])
    st, probs = K.check_pool_seed("c0", str(run), _call(), line, 280,
                                  ["snapshot_000000000164.zip", "snapshot_000000000200.zip"])
    assert st == "FAIL" and any("missing ['snapshot_000000000200.zip']" in p for p in probs)


# ------------------------------------------------------------------------------------------------------ AM-2
def _g1(a1, c0):
    return {"a1": dict(zip(("3501", "3502", "3503"), a1)), "c0": dict(zip(("3501", "3502", "3503", "3504", "3505"), c0))}


def test_g1_is_the_pooled_within_arm_sd_with_4_df():
    g = B.gate_g1(_g1([60.0, 61.0, 62.0], [55.0, 57.0, 59.0]))    # SS = 2 + 8 = 10 over 4 df: sigma = sqrt(2.5)
    assert g.df == 4 and math.isclose(g.sigma_pp, math.sqrt(2.5), rel_tol=0, abs_tol=1e-12)
    assert g.verdict == B.HOLD                                       # 1.58 > 1.2
    g = B.gate_g1(_g1([60.0, 60.5, 61.0], [55.0, 55.4, 55.8]))     # SS = 0.5 + 0.32 = 0.82: sigma = 0.453
    assert g.verdict == B.PASS and g.reestimate is None


def test_g1_rule8_a_sigma_within_1e9_of_the_bar_counts_as_above():
    # one arm (x - h, x, x + h) contributes 2h^2; the other constant: sigma^2 = 2h^2 / 4  =>  h = 1.2 * sqrt(2)
    h = 1.2 * math.sqrt(2.0)
    g = B.gate_g1(_g1([50.0 - h, 50.0, 50.0 + h], [40.0, 40.0, 40.0]))
    assert abs(g.sigma_pp - 1.2) <= 1e-9 and g.near and g.verdict == B.HOLD
    h2 = 1.19 * math.sqrt(2.0)                                        # clearly below: PASS
    assert B.gate_g1(_g1([50.0 - h2, 50.0, 50.0 + h2], [40.0, 40.0, 40.0])).verdict == B.PASS


def test_g1_reports_the_6_df_reestimate_and_refuses_a_bad_input(tmp_path):
    g = B.gate_g1(_g1([60.0, 60.5, 61.0], [55.0, 55.4, 55.8, 56.0, 54.0]))
    assert g.reestimate is not None and g.reestimate["df"] == 6 and g.reestimate["reported_only"]
    with pytest.raises(B.BatchError, match="no S for seed"):
        B.gate_g1({"a1": {"3501": 1.0, "3502": 2.0}, "c0": {"3501": 1.0, "3502": 2.0, "3503": 3.0}})
    with pytest.raises(B.BatchError, match="not a finite"):
        B.gate_g1(_g1([60.0, float("nan"), 61.0], [55.0, 55.4, 55.8]))
    with pytest.raises(B.BatchError, match="look 2 is not an input"):
        B.gate_g1({"a1": {"3501": 1.0, "3502": 2.0, "3503": 3.0, "3504": 4.0},
                   "c0": {"3501": 1.0, "3502": 2.0, "3503": 3.0}})
    p = B.write_g1(str(tmp_path), g, {"x": 1})
    assert B.read_g1(str(tmp_path)) == B.PASS and Path(p).is_file()
    with pytest.raises(B.BatchError, match="written ONCE"):
        B.write_g1(str(tmp_path), g, {})


def test_the_kit_gate_subcommand_exits_on_the_verdict(tmp_path):
    f = tmp_path / "g1.json"
    f.write_text(json.dumps(_g1([60.0, 61.0, 62.0], [55.0, 57.0, 59.0])))
    assert K.main(["gate", str(f), "--stage", str(tmp_path), "--write"]) == 1           # HOLD
    assert B.read_g1(str(tmp_path)) == B.HOLD


# -------------------------------------------------------------------------- AM-9 / AM-3 / AM-4 (the order and its conditions)
def test_batch1_is_am9s_order_the_toy_block_first():
    """AM-9 (orchestrator, 2026-10-20): T1 (toy C0 + X47 s3601-s3603, AM-7's interleave), T1b on X47's look-1 CONTINUE,
    then production P1-P4 once FP32_ALT has landed, and the toy calibration T2 last (conditional on production)."""
    assert [n for n, _ in B.STAGES] == ["T1", "T1b", "P1", "P2", "P3", "P4", "T2"]
    assert list(K.BATCH1) == [
        ("toy_c0", 3601), ("toy_x47", 3601), ("toy_x47", 3602), ("toy_c0", 3602), ("toy_c0", 3603), ("toy_x47", 3603),
        ("toy_x47", 3604), ("toy_c0", 3604), ("toy_c0", 3605), ("toy_x47", 3605),
        ("a1", 3501), ("c0", 3501), ("c0", 3502), ("a1", 3502), ("a1", 3503), ("c0", 3503),
        ("c0", 3504), ("c0", 3505), ("a1", 3504), ("a1", 3505),
        ("d1", 3501), ("d1", 3502), ("d1", 3503),
        ("toy_a1", 3601), ("toy_a1", 3602), ("toy_a1", 3603)]                 # AM-RP-c: A2 / toy A2 DROPPED
    assert len({K.run_name(a, s) for a, s in K.BATCH1}) == 26
    first_prod = min(i for i, (a, _) in enumerate(K.BATCH1) if not a.startswith("toy_"))
    assert all(a.startswith("toy_") for a, _ in K.BATCH1[:first_prod]) and first_prod == 10   # T1 + T1b, then production


def _save(stage: Path, arm: str, look: int, **doc) -> None:
    (stage / B.READS_DIR).mkdir(exist_ok=True)
    (stage / B.READS_DIR / f"{arm}_look{look}.json").write_text(json.dumps(dict(arm=B.READ_LABEL[arm], look=look, **doc)))


def _g1_record(stage: Path, verdict: str) -> None:
    (stage / B.G1_FILE).write_text(json.dumps({"schema": B.G1_SCHEMA, "verdict": verdict}))


def test_t1_and_p1_are_unconditional_and_g1_gates_d1_and_c0s_look_2(tmp_path):
    st = str(tmp_path)
    _fill_fp32(tmp_path)                                                   # FP32_ALT named: production may be asked
    for a, s in [("a1", 3501), ("c0", 3503), ("toy_c0", 3601), ("toy_x47", 3603)]:
        assert B.stage_status(st, a, s)[0] == B.ALLOWED, (a, s)
    for a, s in [("d1", 3501), ("c0", 3504)]:
        assert B.stage_status(st, a, s)[0] == B.PENDING, (a, s)
    _g1_record(tmp_path, B.HOLD)
    assert B.stage_status(st, "c0", 3504)[0] == B.ALLOWED          # the HOLD branch runs C0's look-2 seeds next
    assert B.stage_status(st, "d1", 3501)[0] == B.REFUSED
    assert B.stage_status(st, "a2", 3501)[0] == B.REFUSED
    assert B.stage_status(st, "toy_a1", 3601)[0] == B.REFUSED
    (tmp_path / B.G1_FILE).unlink()
    _g1_record(tmp_path, B.PASS)
    assert B.stage_status(st, "d1", 3501)[0] == B.ALLOWED


def test_am_rp_c_a2_and_toy_a2_are_dropped_refused_at_every_seed_and_whatever_the_records_say(tmp_path):
    """AM-RP-c (orchestrator, 2026-10-20): A2 and toy A2 left the queue. ``stage-check`` REFUSES every seed of them
    with the ruling as its reason, even on the records that once ALLOWED them (G1 PASS, production A2 final); their
    runs would still be judged by the side's pin (they stay batch-1 arms). Fails on revert of the DROPPED rule."""
    st = str(tmp_path)
    _fill_fp32(tmp_path)
    _g1_record(tmp_path, B.PASS)
    _save(tmp_path, "a2", 1, verdict="NULL", r1={"final": True, "class": "NULL"})
    for a in ("a2", "toy_a2"):
        assert not any(a == arm for _, forks in B.STAGES for arm, _ in forks)
        assert a in B.BATCH1_ARMS
        for s in K.ARMS[a].seeds:
            status, why = B.stage_status(st, a, s)
            assert status == B.REFUSED and "AM-RP-c" in why, (a, s, why)
    assert B.stage_status(st, "d1", 3501)[0] == B.ALLOWED and B.stage_status(st, "toy_a1", 3601)[0] == B.PENDING


def test_a_look_2_bon_fork_needs_its_look_1_continue(tmp_path):
    st = str(tmp_path)
    _fill_fp32(tmp_path)
    assert B.stage_status(st, "a1", 3504)[0] == B.PENDING
    _save(tmp_path, "a1", 1, verdict="NULL", r1={"final": True, "class": "NULL"})
    assert B.stage_status(st, "a1", 3504)[0] == B.REFUSED
    (tmp_path / B.READS_DIR / "a1_look1.json").unlink()
    _save(tmp_path, "a1", 1, verdict="CONTINUE", r1={"final": False, "class": "UNDECIDED"})
    assert B.stage_status(st, "a1", 3505)[0] == B.ALLOWED


def test_am4_x47_look_2_and_its_toy_c0_seeds_need_x47_look_1_continue(tmp_path):
    st = str(tmp_path)
    assert B.stage_status(st, "toy_x47", 3604)[0] == B.PENDING
    assert B.stage_status(st, "toy_c0", 3604)[0] == B.PENDING
    _save(tmp_path, "toy_x47", 1, verdict="PER-STEP ONLY")
    assert B.stage_status(st, "toy_x47", 3604)[0] == B.REFUSED
    _save(tmp_path, "toy_x47", 1, verdict="CONTINUE")
    assert B.stage_status(st, "toy_x47", 3605)[0] == B.ALLOWED and B.stage_status(st, "toy_c0", 3605)[0] == B.ALLOWED
    assert K.build_argv("toy_x47", 3605, {"side": "toy", "archive": [],
                                          "pstar": {"path": "m/rb_t/checkpoints/c.zip", "step": 10, "sha256": "s"}})


def test_am3_the_toy_calibration_needs_productions_final_class(tmp_path):
    st = str(tmp_path)
    _fill_fp32(tmp_path)
    _g1_record(tmp_path, B.PASS)
    assert B.stage_status(st, "toy_a1", 3601)[0] == B.PENDING             # production A1 not read yet
    _save(tmp_path, "a1", 1, verdict="CONTINUE", r1={"final": False, "class": "UNDECIDED"})
    assert B.stage_status(st, "toy_a1", 3601)[0] == B.PENDING             # look 2 may still decide it
    _save(tmp_path, "a1", 2, verdict="UNDECIDED", r1={"final": True, "class": "UNDECIDED"})
    status, why = B.stage_status(st, "toy_a1", 3602)
    assert status == B.REFUSED and "NOT SHOWN" in why
    (tmp_path / B.READS_DIR / "a1_look2.json").unlink()
    _save(tmp_path, "a1", 2, verdict="ADOPT-CANDIDATE", r1={"final": True, "class": "GAIN"})
    assert B.stage_status(st, "toy_a1", 3603)[0] == B.ALLOWED
    assert B.stage_status(st, "toy_a2", 3601)[0] == B.REFUSED             # AM-RP-c: toy A2 is DROPPED


def test_a_read_record_for_another_arm_or_look_is_refused(tmp_path):
    _fill_fp32(tmp_path)
    (tmp_path / B.READS_DIR).mkdir()
    (tmp_path / B.READS_DIR / "a1_look1.json").write_text(json.dumps({"arm": "A2", "look": 1, "verdict": "CONTINUE"}))
    with pytest.raises(B.BatchError, match="not A1 look 1"):
        B.stage_status(str(tmp_path), "a1", 3504)


def test_the_disk_check_skips_the_forks_the_records_refused(tmp_path):
    _fill_fp32(tmp_path)
    ok, msg = K.disk_check("/nonexistent", 10.0, thinning=False, existing=[], free_gib=1000.0, stage=str(tmp_path))
    assert ok and "46 batch-1/2 fork(s)" in msg, msg                       # PENDING forks are counted (A2s dropped; 20 of batch 2)
    _g1_record(tmp_path, B.HOLD)                                          # D1 and toy A1 (3 + 3) refused
    ok, msg = K.disk_check("/nonexistent", 10.0, thinning=False, existing=[], free_gib=1000.0, stage=str(tmp_path))
    assert ok and "40 batch-1/2 fork(s)" in msg, msg


# ------------------------------------------------------------------------------------------------------ AM-6
FORK = 280_000_170


def _series(judged, p99, start=FORK + 100_000):
    return {"judged_max": [(start + i, v) for i, v in enumerate(judged)],
            "p99": [(start + i, v) for i, v in enumerate(p99)]}


def test_k9b_passes_rows_under_the_ceilings_and_voids_a_crossing():
    assert B.judge_k9b("r", FORK, _series([2.6e-4, 1e-5], [2.7e-5, 1e-6])).verdict == B.PASS
    k = B.judge_k9b("r", FORK, _series([2.6e-4, 2e-3], [2.7e-5, 1e-6]))
    assert k.verdict == B.VOID and any("judged_max" in f for f in k.fires)
    k = B.judge_k9b("r", FORK, _series([2.6e-4], [3e-4]))
    assert k.verdict == B.VOID and any("p99" in f for f in k.fires)
    k = B.judge_k9b("r", FORK, _series([float("nan")], [1e-6]))
    assert k.verdict == B.VOID and any("non-finite" in f for f in k.fires)
    k = B.judge_k9b("r", FORK, _series([1e-5], [1e-6]), log_text="x " + B.K9B_NONFINITE_LINE + " y")
    assert k.verdict == B.VOID


def test_k9b_reads_only_rows_after_the_fork_step():
    s = _series([1e-5], [1e-6])
    s["judged_max"].insert(0, (FORK, 5e-3))                               # an INHERITED row at the fork step
    assert B.judge_k9b("r", FORK, s).verdict == B.PASS
    assert B.judge_k9b("r", FORK - 1, s).verdict == B.VOID                 # the same row read as the fork's own


def test_k9b_rule8_within_1e9_relative_of_a_ceiling_is_recorded_not_fired():
    k = B.judge_k9b("r", FORK, _series([1e-3 * (1 + 5e-10)], [1e-6]))
    assert k.verdict == B.PASS and k.near
    assert B.judge_k9b("r", FORK, _series([1e-3 * (1 + 1e-6)], [1e-6])).verdict == B.VOID


def test_k9b_absence_and_an_unknown_fork_are_inconclusive_never_a_pass():
    assert B.judge_k9b("r", FORK, {"judged_max": [], "p99": [(FORK + 1, 1e-6)]}).verdict == B.INCONCLUSIVE
    assert B.judge_k9b("r", None, _series([1e-5], [1e-6])).verdict == B.INCONCLUSIVE


def _write_events(path: Path, rows) -> None:
    from torch.utils.tensorboard import SummaryWriter

    path.mkdir(parents=True, exist_ok=True)
    w = SummaryWriter(log_dir=str(path))
    for tag, step, v in rows:
        w.add_scalar(tag, v, step)
    w.close()


def test_k9b_read_on_a_run_dir_excludes_the_inherited_file(tmp_path):
    run = tmp_path / "rb_x35_c0_s3501"
    tb = run / "tb"
    _write_events(tmp_path / "inh", [("behaviour/max_abs_dlogp_judged", FORK - 10, 9e-3),
                                     ("behaviour/p99_abs_dlogp_current", FORK - 10, 9e-4)])
    tb.mkdir(parents=True)
    src = next((tmp_path / "inh").glob("*tfevents*"))
    src.rename(tb / "events.out.tfevents.0000000000.inherited.0.0")
    (tb / "INHERITED_FROM.json").write_text(json.dumps({"fork_step": FORK}))
    _write_events(tb, [("behaviour/max_abs_dlogp_judged", FORK + 5000, 1e-5),
                       ("behaviour/p99_abs_dlogp_current", FORK + 5000, 1e-6)])
    k = B.k9b_read(str(run))
    assert k.verdict == B.PASS and k.fork_step == FORK and k.n_rows == {"judged_max": 1, "p99": 1}, k
    stage = tmp_path / "stage"
    stage.mkdir()
    (run / "final_model.zip").write_bytes(b"z")
    ok, lines = B.k9b_batch(str(stage), str(tmp_path))
    assert ok and lines == ["rb_x35_c0_s3501: K9(b) arm end PASS"] and (stage / B.K9B_DIR / "rb_x35_c0_s3501.json").is_file()


def test_k9b_batch_holds_on_a_void_finished_arm(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    _run(tmp_path / "models", "rb_x35_a1arch_s3501", PIN, final=True)
    _run(tmp_path / "models", "rb_x35_c0_s3501", PIN)                   # not finished: not read
    void = B.judge_k9b("rb_x35_a1arch_s3501", FORK, _series([2e-3], [1e-6]))
    ok, lines = B.k9b_batch(str(stage), str(tmp_path / "models"), reader=lambda rd: void)
    assert not ok and lines == ["rb_x35_a1arch_s3501: K9(b) arm end VOID"]


# ------------------------------------------------------------------------------- the launch script calls them
def test_launch_x35_runs_every_amendment_pre_check():
    from utils.paths import repo_path

    sh = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17/launch_x35.sh").read_text()
    for needle in (
            # AM-1 as amended by AM-8: the pin is judged and written PER SIDE, and a typed --t62 is checked in the dry run
            "kit pin-check --pin \"$PIN\" --side \"$SIDE\" --stage \"$STAGE\" --models-dir \"$MODELS\" "
            "${T62:+--t62 \"$T62\"} >",
            "PINF=$STAGE/x35_pin_batch1_$SIDE.txt", "[ -e \"$PINF\" ] || [ -n \"$T62\" ]",
            # AM-8: the side's own required commits (production: FP32_ALT)
            "kit required-check --side \"$SIDE\" --pin \"$PIN\" --stage \"$STAGE\" --repo \"$REPO\" >",
            "kit stage-check --arm", "kit pool-check --arm \"$ARM\" --call \"$R\" --stage \"$STAGE\" >",
            "kit k9b-batch", "--side \"$SIDE\" --stage \"$STAGE\" --models-dir \"$MODELS\" --write ${T62:+--t62",
            "pool-check --arm \"$ARM\" --call \"$R\" --stage \"$STAGE\" --write",
            "pool-seed-check --arm \"$ARM\" --run-dir \"$MODELS/$RUN\" --call \"$R\" --stage \"$STAGE\"",
            "kit disk --required-gib \"$gib\" --models-dir \"$MODELS\" --stage \"$STAGE\"",
            # AMENDMENT D1-CORPUS (2026-10-20): the arm's team corpus at the pin
            "kit corpus-check --arm \"$ARM\" --pin \"$PIN\" --repo \"$REPO\" >"):
        assert needle in sh, needle
    assert "x35_pin_batch1.txt" not in sh.replace("x35_pin_batch1_", "")        # the single batch pin file is gone
    # the once-only writes happen AFTER the GO's pre-checks, never in a dry run
    assert sh.index("prechecks go\n") < sh.index("--write ${T62:+--t62")
    assert re.search(r"if \[ \"\$GO\" -ne 1 \]; then\n  prechecks dry\n", sh)
    # the required-commit check runs inside prechecks(), before the run-dir and R checks
    assert sh.index("kit required-check") < sh.index("# 3. the run dir") < sh.index("prechecks dry")
    assert sh.index("kit corpus-check") < sh.index("# 3. the run dir")


def test_launch_x35_pre_check_1b_refuses_an_empty_cuda_visible_devices():
    """Pre-check 1b (2026-10-20): an inherited CUDA_VISIBLE_DEVICES="" killed toy_x47 3602's first GO at startup
    ("No CUDA GPUs are available") and the launcher's --dry-run did not catch it. The check sits in prechecks(),
    right after pre-check 1's `kit live`, and is RUN here: set-and-empty HOLDs; unset or a device list passes."""
    from utils.paths import repo_path

    sh = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17/launch_x35.sh").read_text()
    line = next(l for l in sh.splitlines() if 'CUDA_VISIBLE_DEVICES+x' in l)
    body = sh[sh.index("prechecks() {"):]
    assert body.index("kit live") < body.index("CUDA_VISIBLE_DEVICES+x") < body.index("local lease")
    probe = 'hold() { echo "HOLD: $1"; exit 7; }\n' + line.strip() + '\necho PASS\n'
    for env_args, want in ((["CUDA_VISIBLE_DEVICES="], "HOLD"), (["-u", "CUDA_VISIBLE_DEVICES"], "PASS"),
                           (["CUDA_VISIBLE_DEVICES=0"], "PASS")):
        r = subprocess.run(["env", *env_args, "bash", "-c", probe], capture_output=True, text=True)
        assert r.stdout.startswith(want), (env_args, r.stdout, r.stderr)
        assert (r.returncode == 7) == (want == "HOLD")


def test_launch_x35_rotates_an_existing_launcher_log_at_go(tmp_path):
    """A re-launch of the same run name (after a failed GO) must not share the old launcher log: the watcher read the
    old attempt's "cannot restart" lines as the new run's failure and exited at once (toy_x47 3602, 2026-10-20). At
    GO, before the launcher starts, an existing <run>.launcher.log is RENAMED to <log>.<ts>.prev (never deleted)."""
    from utils.paths import repo_path

    sh = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17/launch_x35.sh").read_text()
    line = next(l for l in sh.splitlines() if l.startswith('if [ -e "$J/$RUN.launcher.log" ]; then PREV='))
    assert sh.index("batch records:") < sh.index(line) < sh.index('log "START $RUN') < sh.index(">> \"$J/$RUN.launcher.log\"")
    probe = ('J="$1"; RUN=r; log() { echo "LOG: $1"; }; hold() { echo "HOLD: $1"; exit 7; }\n' + line + '\necho DONE\n')
    log = tmp_path / "r.launcher.log"
    log.write_text("old attempt: cannot restart\n")
    r = subprocess.run(["bash", "-c", probe, "probe", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0 and "rotated the previous launcher log" in r.stdout, (r.stdout, r.stderr)
    prev = [p for p in tmp_path.iterdir() if p.name.startswith("r.launcher.log.") and p.name.endswith(".prev")]
    assert not log.exists() and len(prev) == 1 and prev[0].read_text() == "old attempt: cannot restart\n"
    r = subprocess.run(["bash", "-c", probe, "probe", str(tmp_path)], capture_output=True, text=True)    # no log: no-op
    assert r.returncode == 0 and "rotated" not in r.stdout and not log.exists()
