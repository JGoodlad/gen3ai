"""gen3_fork_lr_inherit_guard_v1 — a FORK of a FROZEN parent must NAME its own dose.

THE DEFECT THIS STANDS FOR. `--fork-lr-freeze` pins a rate and holds the KL controller at it for a
whole run. Fork that run and name no `--fork-lr` and SB3 restores the parent's optimizer state: the
frozen NUMBER arrives, the freeze does not, and a live controller starts annealing away from a
value that was chosen precisely because it should not move. The three era-2 exploiters did exactly
this on 2026-09-20 — 2.80e-05 → 8.36e-05, median 5.5e-05, a dose of 0.39× the v8 reference against
the era-1 exploiters' 1.78×, a 4.5× gap discovered only after the GPU time was spent, on argvs that
three gates had passed.

🚨 **WHY THE EVIDENCE COMES FROM `metadata.json` AND NOWHERE ELSE.** `model_config.json` is a
weight-SHAPE record: its 144 keys carry no `fork_lr`, `learning_rate`, `batch_size`, `n_epochs` or
`grad_accum_steps`. So the obvious shape of this guard — overlay the argv on the parent's recorded
config and read the resolved value — cannot work, and every test below builds its fixture parent
out of a `metadata.json`, which is where the answer actually lives.

Every test here is a REAL fixture on disk: a run directory with a real `metadata.json`, read by the
real reader. No monkeypatching, so nothing can pass because a stub reached nothing.
"""
from __future__ import annotations

import ast
import json
import os

import pytest

from agents.training import lineage
from main.train.fork_lr import (FrozenParent, check_inherited_fork_lr, frozen_parent_pin,
                                is_same_run_checkpoint)


def _run(tmp_path, name, *, command=None, dose=None, role=None):
    """A fixture RUN DIRECTORY holding a real metadata.json and a real checkpoint file."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    meta = {}
    if command is not None:
        meta["original_command"] = command
    if dose is not None:
        meta["dose"] = dose
    if role is not None:
        meta["lineage"] = {"schema": 1, "role": role, "fork_parent": None}
    (d / "metadata.json").write_text(json.dumps(meta))
    (d / "final_model.zip").write_bytes(b"not a real zip, and nothing here opens one")
    return d


FROZEN_CMD = ("launcher/__main__.py --device cuda --steps 100 --fork-lr 2.8e-5 --fork-lr-freeze "
              "--model models/parent/final_model.zip --run-name frozen_parent")


# --------------------------------------------------------------------------------------------
# frozen_parent_pin — WHICH recorded field answered, and when it must NOT answer
# --------------------------------------------------------------------------------------------
def test_freeze_is_read_from_the_immutable_original_command_with_its_value(tmp_path):
    d = _run(tmp_path, "frozen_parent", command=FROZEN_CMD, role="fork")
    pin = frozen_parent_pin(str(d))
    assert isinstance(pin, FrozenParent)
    assert pin.lr == pytest.approx(2.8e-5)
    assert pin.source == "original_command", "the IMMUTABLE record must win over the dose block"
    assert pin.run_name == "frozen_parent"


def test_the_equals_spelling_of_fork_lr_is_read_too(tmp_path):
    """`--fork-lr=2.8e-5` is the same flag. A reader that only knows `--flag V` is the shape of
    every regex this project has had to replace."""
    cmd = "launcher/__main__.py --fork-lr=2.8e-5 --fork-lr-freeze --model models/p/final_model.zip"
    pin = frozen_parent_pin(str(_run(tmp_path, "eq", command=cmd, role="fork")))
    assert pin is not None and pin.lr == pytest.approx(2.8e-5)


def test_a_restart_persisted_freeze_is_read_from_the_dose_block(tmp_path):
    """The freeze survives a restart by being re-applied from the recorded pin, so a run can have
    run FROZEN while its own command says nothing. The `dose` block is that second warrant."""
    d = _run(tmp_path, "restart_frozen",
             command="launcher/__main__.py --model models/p/final_model.zip",
             dose={"lr_frozen": True, "fork_lr": 5.5e-5, "lr_now": 5.5e-5}, role="fork")
    pin = frozen_parent_pin(str(d))
    assert pin is not None
    assert pin.lr == pytest.approx(5.5e-5)
    assert pin.source == "dose_block"


def test_a_FRESH_run_with_a_frozen_dose_block_is_NOT_a_frozen_parent(tmp_path):
    """🚨 THE FALSE-POSITIVE THIS GUARD MUST NOT HAVE. A fresh run cannot have inherited a pin, and
    treating one as frozen would refuse every ordinary fork of every ordinary run — 162 of them in
    the archive. The `dose` route therefore requires `lineage` to say FORK."""
    d = _run(tmp_path, "fresh", command="launcher/__main__.py --steps 100",
             dose={"lr_frozen": True, "fork_lr": 3e-4}, role="fresh")
    assert frozen_parent_pin(str(d)) is None


def test_an_unfrozen_parent_is_not_a_frozen_parent(tmp_path):
    d = _run(tmp_path, "unfrozen",
             command="launcher/__main__.py --model models/p/final_model.zip",
             dose={"lr_frozen": False, "lr_now": 8.36e-5}, role="exploiter")
    assert frozen_parent_pin(str(d)) is None


def test_a_parent_that_is_not_on_this_box_is_UNKNOWN_not_frozen(tmp_path):
    assert frozen_parent_pin(str(tmp_path / "no_such_run")) is None
    assert frozen_parent_pin(None) is None


# --------------------------------------------------------------------------------------------
# check_inherited_fork_lr — the five outcomes
# --------------------------------------------------------------------------------------------
def _fork_of(tmp_path, **kw):
    parent = _run(tmp_path, "frozen_parent", command=FROZEN_CMD, role="fork")
    return check_inherited_fork_lr(
        model_path=str(parent / "final_model.zip"),
        model_dir=str(tmp_path / "child"),
        fork_lr=kw.get("fork_lr"), fork_lr_freeze=kw.get("fork_lr_freeze", False),
        allow=kw.get("allow", False))


def test_a_fork_of_a_frozen_parent_naming_neither_flag_is_REFUSED(tmp_path):
    v = _fork_of(tmp_path)
    assert v.refuse is True and v.status == "REFUSE"
    assert v.line.startswith("[ForkLR] FATAL:")
    # the message must name the PARENT'S FROZEN VALUE and the exact fix — a refusal that does not
    # say what to type is a refusal the operator works around.
    assert "2.80e-05" in v.line
    assert "--fork-lr 2.8e-05 --fork-lr-freeze" in v.line
    assert "--allow-inherited-fork-lr" in v.line
    assert v.parent is not None and v.parent.run_name == "frozen_parent"


def test_naming_fork_lr_passes(tmp_path):
    v = _fork_of(tmp_path, fork_lr=2.5e-4)
    assert v.refuse is False and v.status == "named" and "0.00025" in v.line


def test_naming_only_the_freeze_passes(tmp_path):
    """`--fork-lr-freeze` alone is a deliberate statement about the dose too — the operator has
    said the rate must not move, and the guard's question is whether the argv is silent."""
    v = _fork_of(tmp_path, fork_lr_freeze=True)
    assert v.refuse is False and v.status == "named"


def test_the_override_warns_and_does_not_refuse(tmp_path):
    v = _fork_of(tmp_path, allow=True)
    assert v.refuse is False and v.status == "override"
    assert "2.80e-05" in v.line and "main.dose" in v.line


def test_a_same_run_RESTART_never_fires(tmp_path):
    """🚨 The launcher re-invokes the SAME argv every --restart-interval-hours. A guard that fired
    on a restart would kill every restart of every legitimately-inheriting run."""
    child = tmp_path / "child"
    (child / "checkpoints").mkdir(parents=True)
    ckpt = child / "checkpoints" / "checkpoint_100_steps.zip"
    ckpt.write_bytes(b"x")
    assert is_same_run_checkpoint(str(ckpt), str(child)) is True
    v = check_inherited_fork_lr(model_path=str(ckpt), model_dir=str(child),
                                fork_lr=None, fork_lr_freeze=False)
    assert v.refuse is False and v.status == "restart"


def test_a_fresh_run_never_fires(tmp_path):
    v = check_inherited_fork_lr(model_path=None, model_dir=str(tmp_path / "child"),
                                fork_lr=None, fork_lr_freeze=False)
    assert v.refuse is False and v.status == "fresh"


def test_an_unreadable_parent_REPORTS_and_does_not_refuse(tmp_path):
    """A guard that cannot see the evidence must not pretend it did — it says UNKNOWN out loud."""
    v = check_inherited_fork_lr(model_path=str(tmp_path / "gone" / "final_model.zip"),
                                model_dir=str(tmp_path / "child"),
                                fork_lr=None, fork_lr_freeze=False)
    assert v.refuse is False and v.status == "parent_unreadable"
    assert "UNKNOWN" in v.line


def test_a_fork_of_an_UNFROZEN_parent_passes_and_says_so(tmp_path):
    parent = _run(tmp_path, "ordinary",
                  command="launcher/__main__.py --model models/p/final_model.zip",
                  dose={"lr_frozen": False, "lr_now": 8.36e-5}, role="exploiter")
    v = check_inherited_fork_lr(model_path=str(parent / "final_model.zip"),
                                model_dir=str(tmp_path / "child"),
                                fork_lr=None, fork_lr_freeze=False)
    assert v.refuse is False and v.status == "parent_unfrozen"


# --------------------------------------------------------------------------------------------
# the wiring: the flag, and WHERE the launch path calls the guard
# --------------------------------------------------------------------------------------------
def test_the_override_flag_exists_and_defaults_off():
    from main.train_rl_agent import build_parser

    ns = build_parser().parse_args([])
    assert ns.allow_inherited_fork_lr is False
    ns = build_parser().parse_args(["--allow-inherited-fork-lr"])
    assert ns.allow_inherited_fork_lr is True


def test_the_launch_path_calls_the_guard_BEFORE_it_creates_the_run_dir():
    """🚨 A refusal must leave NOTHING behind — the same property the [Untaught] guard has.

    Read out of `train_rl_agent.main`'s own AST rather than asserted in prose: the guard call must
    appear before the `os.makedirs(model_dir, ...)` that creates the directory. A later edit that
    reorders them fails here instead of leaving orphan run dirs on every refused launch.
    """
    from utils.paths import src_path

    tree = ast.parse(src_path("main", "train_rl_agent.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main")
    guard_line = makedirs_line = None
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name == "enforce_inherited_fork_lr" and guard_line is None:
            guard_line = node.lineno
        if (name == "makedirs" and node.args
                and getattr(node.args[0], "id", None) == "model_dir"
                and makedirs_line is None):
            makedirs_line = node.lineno
    assert guard_line is not None, "train_rl_agent.main no longer calls enforce_inherited_fork_lr"
    assert makedirs_line is not None, "the os.makedirs(model_dir) this test keys off is gone"
    assert guard_line < makedirs_line, (
        f"the fork-lr guard runs at line {guard_line}, AFTER os.makedirs(model_dir) at "
        f"{makedirs_line} — a refusal would leave an orphan run directory behind")


def test_checkargs_prints_the_verdict_on_every_argv(tmp_path, capsys):
    """BOTH surfaces read ONE function. checkargs prints; the launch path refuses."""
    from main.checkargs import _print_fork_lr_inheritance

    parent = _run(tmp_path, "frozen_parent", command=FROZEN_CMD, role="fork")

    class _NS:
        run_dir = str(tmp_path / "child")
        run_name = None
        fork_lr = None
        fork_lr_freeze = False
        allow_inherited_fork_lr = False

    _print_fork_lr_inheritance({"ns": _NS(), "model": str(parent / "final_model.zip")})
    out = capsys.readouterr().out
    assert "[ForkLR] FATAL:" in out
    assert "WOULD FAIL AT LAUNCH" in out
    # and the closing parser verdict must be explicitly de-conflicted from this one
    assert "does NOT overrule" in out


# --------------------------------------------------------------------------------------------
# the recorded-command grammar these guards stand on
# --------------------------------------------------------------------------------------------
def test_lineage_command_helpers_know_both_spellings():
    cmd = "x.py --fork-lr 2.8e-5 --fork-lr-freeze --steps 10"
    assert lineage.command_flag_value(cmd, "--fork-lr") == "2.8e-5"
    assert lineage.command_has_flag(cmd, "--fork-lr-freeze") is True
    assert lineage.command_flag_value("x.py --fork-lr=9e-5", "--fork-lr") == "9e-5"
    assert lineage.command_has_flag("x.py --steps 10", "--fork-lr-freeze") is False
    # total: a malformed command answers, it does not raise
    assert lineage.command_tokens("x.py --a 'unterminated") == []
    assert lineage.command_flag_value(None, "--fork-lr") is None


def test_the_three_era2_exploiters_are_exactly_what_this_guard_catches():
    """THE REGRESSION, against the real archive when it is present.

    `ai_v13_13/14/15` forked the plateau parent's frozen 2.80e-05 and named neither flag. If this
    guard had existed they would have been refused at launch, before 24 GPU-hours. Skipped where
    there is no run archive (a worktree, CI) — `models/` lives only in the MAIN checkout.
    """
    from utils.paths import main_models_dir

    root = main_models_dir()
    if root is None:
        pytest.skip("no models/ archive on this box")
    parent = root / "ai_v13_12_plateau"
    if not (parent / "metadata.json").exists():
        pytest.skip("the plateau parent is not in this archive")
    pin = frozen_parent_pin(str(parent))
    assert pin is not None and pin.lr == pytest.approx(2.8e-5)
    for name in ("ai_v13_13_exploit5_offense", "ai_v13_14_exploit5_balance",
                 "ai_v13_15_exploit5_stall"):
        run = root / name
        if not (run / "metadata.json").exists():
            continue
        cmd = lineage.read_original_command(str(run))
        assert not lineage.command_has_flag(cmd, "--fork-lr", "--fork-lr-freeze"), (
            f"{name}'s recorded command names a fork-lr flag — the archaeology this test asserts "
            f"has changed")
        v = check_inherited_fork_lr(model_path=str(parent / "final_model.zip"),
                                    model_dir=str(root / name), fork_lr=None,
                                    fork_lr_freeze=False)
        assert v.refuse is True, f"{name} would not have been caught"
    assert os.path.isdir(str(root))
