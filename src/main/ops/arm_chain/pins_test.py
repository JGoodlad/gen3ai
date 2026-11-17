"""``plan --check-pins`` judges each arm's OWN pin (``pins.py``): it resolves, it is on main or the chain file says why
not (and a ref holds it), and the PINNED parser accepts the arm's argv.

Every test pins one refusal (or the pass) and fails when its check is removed. The git cases run in a temp checkout
(``testing_fakes.make_world``: branch ``main``, no ``origin``, so they name ``--main-ref main``); the checkargs cases run
a scripted runner, with REAL transcripts of ``main.checkargs --pin`` (X70's argv at its own pin; the same argv plus
``--magnet-kl-coef`` at ``f4b25238``, which has no such flag) for the verdict reader. ``integration``: it runs git.
"""
from __future__ import annotations

import os
import shlex
from typing import Any, Dict, List

import pytest

from main.ops.arm_chain import cli, pins
from main.ops.arm_chain.launch import Describe
from main.ops.arm_chain.spec import ArmItem, as_dict
from main.ops.arm_chain.testing_fakes import (FakeRunner, World, checkargs_handler, commit_all, describe_handler, _git,
                                              make_world)

pytestmark = pytest.mark.integration

#: Real ``main.checkargs --pin 9e97cad5 --argv <X70 s7331's kit argv>`` (exit 0), trimmed (the box's absolute paths made relative:
#: ``utils.home_path_literal_gate_test`` forbids a home path in any string).
OK_TRANSCRIPT = """checked 8 flags from --argv
  parser                         : the PINNED commit 9e97cad5's — --pin
  ✅ argv validated against PINNED parser @9e97cad5 via the pinned build_parser() [mode=build_parser, 390 options, 2.5s]
  accepted by the PINNED parser  : 8
  ✓ fork arch  : compatible with models/rb_x26_s2601/model_config.json (check_compatible, the trainer's own check)
  ✓ this command still launches (its own commit's parser accepts every flag)
"""
#: Real ``main.checkargs --pin f4b25238 --argv "<the same argv> --magnet-kl-coef 0.05"`` (exit 3), trimmed.
REFUSED_TRANSCRIPT = """checked 8 flags from --argv
  parser                         : the PINNED commit f4b25238's — --pin
  ✗ NOT IN PINNED TREE @f4b25238: --magnet-kl-coef (exists only in the current tree)
      the child runs the PINNED tree, whose parser has no such flag — it would die at startup with a run dir already on disk
  ✗ argv REFUSED by the PINNED parser @f4b25238 via the pinned build_parser() [mode=build_parser]
      unknown flag at this commit: --magnet-kl-coef
  accepted by the PINNED parser  : 7
  in THIS tree but NOT at the pin : 1  ✗ WOULD FAIL AT LAUNCH
"""
SCAN_TRANSCRIPT = ("  ⚠️ argv questioned by the PINNED parser @abc via a static scan of its add_argument calls "
                   "[mode=ast_scan]\n  ✓ this command still launches\n")


@pytest.fixture
def world(tmp_path) -> World:
    return make_world(str(tmp_path))


def spec_of(w: World, arms: List[Dict[str, Any]], **over: Any):
    return w.spec(arms, **over)


def side_commit(w: World, branch: str = "side") -> str:
    """A commit that is NOT an ancestor of main (it sits on ``branch``, forked from ``pin_a``)."""
    _git(w.repo, "checkout", "-q", "-b", branch, w.pin_a)
    sha = commit_all(w.repo, f"off main: {branch}")   # a unique message: two commits in one second must differ
    _git(w.repo, "checkout", "-q", "main")
    return sha


def describe_of(it: ArmItem) -> Describe:
    return Describe(run_name=it.run, parent_run_dir="/p/parent", delta_text="+1,000 steps", steps=2000,
                    argv=["--model", "/p/parent/checkpoints/ck_1000_steps.zip", "--steps", "2000",
                          "--run-name", it.run])


def checkargs_calls(r: FakeRunner) -> List[List[str]]:
    return [a for a, _ in r.calls if "main.checkargs" in a]


# -- layers 1 and 2: git ---------------------------------------------------------------------------------------------

def test_a_pin_on_main_and_a_resolvable_window_commit_pass(world):
    win = {"kind": "window", "name": "between", "worktree": world.repo, "require_commit": world.pin_a, "budget_min": 5,
           "on_fail": "continue", "preempt_marker": "NO_GO",
           "steps": [{"name": "s", "argv": ["true"], "gpu": False, "mem_gb": 1, "timeout_s": 30}]}
    s = spec_of(world, [world.arm("a1", 1, world.pin_a), win, world.arm("a2", 2, world.pin_b)])
    assert pins.check_pins(s, main_ref="main") == []


def test_an_unresolvable_pin_is_refused_naming_the_arm_the_pin_and_the_reason(world):
    s = spec_of(world, [world.arm("a1", 1, world.pin_a), world.arm("a2", 2, "0" * 40)])
    bad = pins.check_pins(s, main_ref="main")
    assert len(bad) == 1
    assert "arm:rb_a2_s2" in bad[0] and "ARM a2 s2" in bad[0] and "0" * 40 in bad[0] and "is not a commit" in bad[0]


def test_a_window_commit_that_is_not_a_commit_is_still_refused(world):
    win = {"kind": "window", "name": "between", "worktree": world.repo, "require_commit": "deadbeef", "budget_min": 5,
           "on_fail": "continue", "preempt_marker": "NO_GO",
           "steps": [{"name": "s", "argv": ["true"], "gpu": False, "mem_gb": 1, "timeout_s": 30}]}
    bad = pins.check_pins(spec_of(world, [world.arm("a1", 1, world.pin_a), win]), main_ref="main")
    assert len(bad) == 1 and "window:between" in bad[0] and "deadbeef" in bad[0]


def test_a_pin_off_main_is_refused_unless_the_chain_file_allows_it(world):
    off = side_commit(world)
    s = spec_of(world, [world.arm("a1", 1, world.pin_a), world.arm("a2", 2, off)])
    bad = pins.check_pins(s, main_ref="main")
    assert len(bad) == 1
    assert "arm:rb_a2_s2" in bad[0] and off in bad[0] and "is not an ancestor of main" in bad[0] \
        and "off_main_pins" in bad[0]


def test_an_allowed_off_main_pin_held_by_a_branch_passes(world):
    off = side_commit(world)    # the branch `side` holds it
    s = spec_of(world, [world.arm("a1", 1, off)], off_main_pins={off: "the swap's training pin: main + the magnet"})
    assert pins.check_pins(s, main_ref="main") == []


def test_an_allowed_off_main_pin_that_no_ref_holds_is_refused(world):
    off = side_commit(world)
    _git(world.repo, "branch", "-D", "side")    # the commit is now referenced by nothing but the reflog
    s = spec_of(world, [world.arm("a1", 1, off)], off_main_pins={off: "kept off main"})
    bad = pins.check_pins(s, main_ref="main")
    assert len(bad) == 1 and off in bad[0] and "no branch or tag holds it" in bad[0]


def test_an_allowance_names_one_pin_only(world):
    """The reason is for THAT pin: another arm's off-main pin is still refused."""
    off, other = side_commit(world, "side1"), side_commit(world, "side2")
    s = spec_of(world, [world.arm("a1", 1, off), world.arm("a2", 2, other)], off_main_pins={off: "allowed"})
    bad = pins.check_pins(s, main_ref="main")
    assert len(bad) == 1 and "arm:rb_a2_s2" in bad[0] and other in bad[0]


def test_a_main_ref_that_does_not_resolve_refuses_the_chain_instead_of_skipping_the_ancestry(world):
    s = spec_of(world, [world.arm("a1", 1, world.pin_a)])
    bad = pins.check_pins(s)    # the default, origin/main: the temp checkout has no origin
    assert len(bad) == 1 and "origin/main is not a commit" in bad[0] and "cannot judge" in bad[0]


# -- layer 3: the pinned parser --------------------------------------------------------------------------------------

def test_the_verdict_reader_on_real_transcripts():
    assert pins.judge_checkargs(0, OK_TRANSCRIPT) is None
    why = pins.judge_checkargs(3, REFUSED_TRANSCRIPT)
    assert why is not None and why.startswith("main.checkargs --pin exits 3:")
    assert "NOT IN PINNED TREE @f4b25238: --magnet-kl-coef" in why
    scan = pins.judge_checkargs(0, SCAN_TRANSCRIPT)
    assert scan is not None and "mode=ast_scan" in scan and "UNVALIDATED" in scan
    assert "mode=unavailable" in (pins.judge_checkargs(0, "  [mode=unavailable, 0 options]\n") or "")
    # the pinned tree IS the checkout's HEAD: no pinned line at all, the current parser answered: the exit code decides
    assert pins.judge_checkargs(0, "  ✓ this command still launches\n") is None
    assert "exits 124" in (pins.judge_checkargs(124, "[arm_chain] TIMEOUT after 600 s\n") or "")


def test_the_argv_is_judged_the_way_the_kit_runs_it(world):
    """``cd <main> && PYTHONPATH=<main>/src python -m main.checkargs --pin <the ARM's pin> --argv "<describe argv>"``,
    on CPU (root rule 5); a second arm at ANOTHER pin is judged at its own."""
    s = spec_of(world, [world.arm("a1", 1, world.pin_a), world.arm("a2", 2, world.pin_b)])
    r = FakeRunner().on(*checkargs_handler())
    for it in s.arms():
        assert pins.check_argv_at_pin(s, it, describe_of(it), r, {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0"}) == []
    assert [(a[:5], shlex.split(a[-1])) for a in checkargs_calls(r)] == [
        ([s.python, "-m", "main.checkargs", "--pin", it.pin], describe_of(it).argv) for it in s.arms()]
    assert [a[-2] for a in checkargs_calls(r)] == ["--argv", "--argv"]
    env = r.calls[0][1]
    assert env["PYTHONPATH"] == os.path.join(world.repo, "src") and env["CUDA_VISIBLE_DEVICES"] == ""


def test_a_flag_the_pin_does_not_have_is_refused_naming_the_arm_the_pin_and_the_reason(world):
    s = spec_of(world, [world.arm("a1", 1, world.pin_a)])
    it = s.arms()[0]
    r = FakeRunner().on(*checkargs_handler(3, REFUSED_TRANSCRIPT))
    bad = pins.check_argv_at_pin(s, it, describe_of(it), r, {})
    assert len(bad) == 1
    assert "arm:rb_a1_s1" in bad[0] and "ARM a1 s1" in bad[0] and world.pin_a in bad[0]
    assert "exits 3" in bad[0] and "--magnet-kl-coef" in bad[0] and "plan_checkargs.a1.1.txt" in bad[0]


def test_a_parser_verdict_that_is_not_authoritative_is_a_refusal_not_a_pass(world):
    s = spec_of(world, [world.arm("a1", 1, world.pin_a)])
    it = s.arms()[0]
    r = FakeRunner().on(*checkargs_handler(0, SCAN_TRANSCRIPT))
    bad = pins.check_argv_at_pin(s, it, describe_of(it), r, {})
    assert len(bad) == 1 and "UNVALIDATED" in bad[0] and world.pin_a in bad[0]


# -- the verb --------------------------------------------------------------------------------------------------------

def write_chain(w: World, arms: List[Dict[str, Any]], **over: Any) -> str:
    import json
    p = os.path.join(w.root, "chain.json")
    with open(p, "w") as f:
        json.dump(w.chain_dict(arms, **over), f)
    return p


def plan(path: str, runner: FakeRunner, *flags: str) -> int:
    return cli.main(["plan", path, "--main-ref", "main", *flags], runner=runner)


def test_plan_check_pins_passes_when_every_arm_is_judged_ok(world, capsys):
    p = write_chain(world, [world.arm("a1", 1, world.pin_a), world.arm("a2", 2, world.pin_b)])
    r = FakeRunner().on(*describe_handler()).on(*checkargs_handler())
    assert plan(p, r, "--check-pins") == 0
    out = capsys.readouterr().out
    assert "PLAN OK" in out and "the kit agrees" not in out    # --describe's lines are --describe's
    assert [a[a.index("--pin") + 1] for a in checkargs_calls(r)] == [world.pin_a, world.pin_b]
    # each arm's describe is asked once, and its argv is the one checkargs is given
    assert sum("describe" in a for a, _ in r.calls) == 2


def test_plan_check_pins_exits_nonzero_naming_the_arm_the_pin_and_the_reason(world, capsys):
    p = write_chain(world, [world.arm("a1", 1, world.pin_a), world.arm("a2", 2, world.pin_b)])
    r = FakeRunner().on(*describe_handler()).on(*checkargs_handler(3, REFUSED_TRANSCRIPT))
    assert plan(p, r, "--check-pins") == 1
    out = capsys.readouterr().out
    assert "PLAN REFUSED (2 problem(s))" in out
    for key, pin in (("rb_a1_s1", world.pin_a), ("rb_a2_s2", world.pin_b)):
        line = next(ln for ln in out.splitlines() if ln.startswith("REFUSED:") and key in ln)
        assert pin in line and "--magnet-kl-coef" in line


def test_plan_check_pins_refuses_an_off_main_pin_and_does_not_run_checkargs_for_an_unresolvable_one(world, capsys):
    off = side_commit(world)
    p = write_chain(world, [world.arm("a1", 1, off), world.arm("a2", 2, "0" * 40)])
    r = FakeRunner().on(*describe_handler()).on(*checkargs_handler())
    assert plan(p, r, "--check-pins") == 1
    out = capsys.readouterr().out
    assert "is not an ancestor of main" in out and "0" * 40 + " is not a commit" in out
    # the off-main pin is still judged by its parser; the unresolvable one is not (checkargs would only repeat it)
    assert [a[a.index("--pin") + 1] for a in checkargs_calls(r)] == [off]


def test_plan_check_pins_honours_the_chain_files_off_main_allowance(world, capsys):
    off = side_commit(world)
    p = write_chain(world, [world.arm("a1", 1, off)], off_main_pins={off: "main + the magnet; tag ce3d-pin"})
    r = FakeRunner().on(*describe_handler()).on(*checkargs_handler())
    assert plan(p, r, "--check-pins") == 0 and "PLAN OK" in capsys.readouterr().out


def test_plan_check_pins_refuses_an_arm_whose_argv_the_kit_will_not_give(world, capsys):
    """No argv, no judgement: that is a refusal, never a silent skip of the arm's pin."""
    p = write_chain(world, [world.arm("a1", 1, world.pin_a)])
    r = FakeRunner().on(*describe_handler(refuse="arm a1 is unknown")).on(*checkargs_handler())
    assert plan(p, r, "--check-pins") == 1
    out = capsys.readouterr().out
    assert "arm:rb_a1_s1" in out and "arm a1 is unknown" in out and "pin was NOT judged" in out
    assert checkargs_calls(r) == []


def test_plan_describe_alone_does_not_judge_pins(world, capsys):
    """``--describe`` keeps its meaning (the kit's run name); the pin judgement is ``--check-pins``'s."""
    p = write_chain(world, [world.arm("a1", 1, "0" * 40)])
    r = FakeRunner().on(*describe_handler()).on(*checkargs_handler(3, REFUSED_TRANSCRIPT))
    assert plan(p, r, "--describe") == 0
    assert "the kit agrees" in capsys.readouterr().out and checkargs_calls(r) == []


def test_an_off_main_allowance_never_enters_an_items_saved_snapshot(world):
    """``chain._load_state`` refuses a STARTED item whose ``as_dict`` changed: a live supervisor restarted after this
    change must still load its chain, so the allowance lives on the chain, not on the arm."""
    off = side_commit(world)
    plain = spec_of(world, [world.arm("a1", 1, off)])
    allowed = spec_of(world, [world.arm("a1", 1, off)], off_main_pins={off: "kept off main"})
    assert allowed.off_main_pins == {off: "kept off main"} and plain.off_main_pins == {}
    assert as_dict(allowed.arms()[0]) == as_dict(plain.arms()[0])
    assert not any("off_main" in k for k in as_dict(allowed.arms()[0]))
