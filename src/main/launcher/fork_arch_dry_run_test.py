"""F-BI-2: `main.checkargs` and `launcher --dry-run` judge a `--model` argv's ARCHITECTURE against the
checkpoint's recorded `model_config.json`, with the trainer's own check (`main.train.fork_arch`).

Before the fix a toy fork typing `--trunk-layers 4` read "✓ this command still launches" and the dry
run "would launch"; the child then died `[ModelVersion] FATAL: trunk_layers mismatch`. Each refusal
test below FAILS ON REVERT (drop the `fork_arch` verdict from `checkargs.check` and both refusals
read exit 0). The pass tests pin that the verdict is no false alarm on the two fork shapes in use: an
X26-style production fork (`--behaviour-check warn --team-mix …`) and an X34-style toy continuation
(the architecture inherited, nothing arch typed).

The parent configs are FIXTURES in a tmp run archive (`dry_run_test.isolated`), built through
`main.fresh_checkpoint`'s own fresh-run construction, never `main.train.fork_arch` — so the check is
not graded by the code it tests. Nothing under the real `models/` is read or written.
"""

import os

import pytest

from main.exit_codes import TrainExitCode
from main.launcher import dry_run_test as _drt
from main.launcher.dry_run_test import _dry_run, _make_run

#: `dry_run_test`'s tmp git repo + booby-trapped launcher, re-exported as this module's fixtures.
repo = _drt.repo
isolated = _drt.isolated

#: The X34 toy's FRESH arch argv (`designs/research_state/measurements/x34_toy_deep_stage_2026-10-17/
#: toy_argv_head_fresh.txt`, the arch half): production with every lever off and a depth-2 trunk.
TOY_ARCH = ("--arch production --allow-nonproduction-arch --unified-moves off --no-damage-op "
            "--no-damage-outgoing --damage-matrices off --damage-topk 0 --move-belief-mode off "
            "--no-move-prior-fusion --no-move-latent --opp-belief-cls-k 0 --no-item-belief --no-spread-belief "
            "--no-spread-belief-nature --no-t0-species-prior --no-species-prior-fusion --entity-topk-seats 0 "
            "--no-entity-tail-seats --edge-bias-families off --no-history-events --no-value-entity-pool "
            "--no-value-entity-pool-full --mon-hazard-cost off --trunk-layers 2 --move-set-closure off "
            "--no-op-drop-renders --no-op-believed-lean --opp-belief-aux-coef 0 --opp-intent-coef 0 "
            "--move-belief-coef 0 --move-belief-latent-coef 0 --spread-belief-coef 0 --item-belief-coef 0 "
            "--hp-type-belief-coef 0").split()

#: The X34 continuation (`fork_argv_minimal.txt`) and an X35 / X26-style production fork, minus `--model`.
X34_FORK = ("--steps 500000000 --seed 4002 --checkpoint-every-steps 1000000 --snapshot-ladder-games 0 "
            "--device cpu --run-name rb_toy_deep_fixture --supply-starve-cycles self_play_pool=0").split()
X26_FORK = ("--behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cpu "
            "--team-mix w_forum_v1@ABC:0.5 --team-mix-sides both --seed 3501 "
            "--run-name rb_x35_fixture").split()


def _write_parent_config(run_dir: str, arch_argv: "list[str] | None") -> None:
    """The parent's `model_config.json`, as a FRESH run at `arch_argv` records it (None = production)."""
    from agents.model.model_version import ModelVersion
    from main.fresh_checkpoint import _production_policy_kwargs

    args = None
    if arch_argv is not None:
        import contextlib
        import io

        from main.train.config import resolve_config
        from main.train.parser import build_parser
        parser = build_parser()
        args = parser.parse_args([*arch_argv, "--steps", "1000"])
        with contextlib.redirect_stdout(io.StringIO()):   # resolved as `production_args` resolves its argv
            resolve_config(args, parser)
    args, layout, pk = _production_policy_kwargs(args)
    version = ModelVersion.from_layout_and_policy_kwargs(
        layout, pk, vf_coef=args.vf_coef, move_belief_coef=args.move_belief_coef,
        move_belief_latent_coef=args.move_belief_latent_coef,
        spread_belief_coef=args.spread_belief_coef, opp_belief_aux_coef=args.opp_belief_aux_coef,
        hp_type_belief_coef=args.hp_type_belief_coef, item_belief_coef=args.item_belief_coef,
        opp_intent_coef=float(args.opp_intent_coef or 0.0), trunk_precision=args.trunk_precision)
    with open(os.path.join(run_dir, "model_config.json"), "w") as fh:
        fh.write(version.to_json())


@pytest.fixture
def toy_parent(isolated):
    _root, (first, _second), work = isolated
    run_dir, ckpt = _make_run(work, first, name="rb_toy_parent")
    _write_parent_config(run_dir, TOY_ARCH)
    return ckpt


@pytest.fixture
def prod_parent(isolated):
    _root, (first, _second), work = isolated
    run_dir, ckpt = _make_run(work, first, name="rb_prod_parent")
    _write_parent_config(run_dir, None)
    return ckpt


def _checkargs(argv):
    from main.checkargs import check
    return check(argv)["fork_arch"]


# --- checkargs.check: the verdict itself ---------------------------------------------------------

def test_a_toy_fork_typing_a_deeper_trunk_is_refused_with_the_trainers_reason(toy_parent):
    v = _checkargs(["--model", toy_parent, "--trunk-layers", "4", *X34_FORK])
    assert v is not None and v.checked and v.refused, v
    assert "trunk_layers mismatch: saved=2, current=4" in v.reason


def test_the_x34_style_toy_continuation_passes(toy_parent):
    v = _checkargs(["--model", toy_parent, *X34_FORK])
    assert v is not None and v.checked and not v.refused, v


def test_the_x26_style_production_fork_passes(prod_parent):
    v = _checkargs(["--model", prod_parent, *X26_FORK])
    assert v is not None and v.checked and not v.refused, v


def test_a_production_fork_typing_another_depth_is_refused(prod_parent):
    v = _checkargs(["--model", prod_parent, "--trunk-layers", "4", *X26_FORK])
    assert v is not None and v.checked and v.refused and "trunk_layers mismatch" in v.reason, v


def test_an_unreadable_parent_config_is_not_judged_and_not_refused(isolated):
    """`_make_run`'s stub config cannot be read as a ModelVersion: no verdict is guessed, nothing refuses."""
    from main.launcher.dry_run_test import _make_run as make_run
    from main.train.fork_arch import fork_arch_verdict
    _root, (first, _second), work = isolated
    _run_dir, ckpt = make_run(work, first, name="rb_stub_parent")
    assert _checkargs(["--model", ckpt, *X26_FORK]) is None, "checkargs skips a parent it cannot read"
    v = fork_arch_verdict(["--model", ckpt, *X26_FORK], ckpt)
    assert not v.checked and not v.refused and "not readable" in v.note, v


def test_a_declared_growth_option_lets_its_fields_change(toy_parent, monkeypatch):
    """The B-GROW seam: a SET growth option's declared fields are taken as current before the check;
    an unset one changes nothing. (The fixture keys the table on `trunk_layers` itself, which is set.)"""
    from main.train import fork_arch
    argv = ["--model", toy_parent, "--trunk-layers", "4", *X34_FORK]
    monkeypatch.setattr(fork_arch, "DECLARED_GROWTH", {"grow_fixture_unset": ("trunk_layers",)})
    assert _checkargs(argv).refused, "an unset growth option must not open the field"
    monkeypatch.setattr(fork_arch, "DECLARED_GROWTH", {"trunk_layers": ("trunk_layers",)})
    v = _checkargs(argv)
    assert v.checked and not v.refused, v


# --- the two front ends: exit codes ----------------------------------------------------------------

def test_dry_run_refuses_the_mismatched_fork(toy_parent, monkeypatch, capsys):
    _dry_run(["--no-pin", "--model", toy_parent, "--trunk-layers", "4", *X34_FORK], monkeypatch,
             expect=int(TrainExitCode.FATAL_CONFIG))
    out = capsys.readouterr().out
    assert "✗ REFUSED (architecture vs the --model checkpoint" in out
    assert "trunk_layers mismatch: saved=2, current=4" in out
    assert "would NOT launch" in out


def test_dry_run_passes_the_matching_fork(toy_parent, monkeypatch, capsys):
    _dry_run(["--no-pin", "--model", toy_parent, *X34_FORK], monkeypatch)
    out = capsys.readouterr().out
    assert "✓ fork arch  : compatible" in out and "would launch" in out


def test_checkargs_main_refuses_the_mismatched_fork_with_fatal_config(toy_parent, monkeypatch, capsys):
    from main import checkargs
    monkeypatch.setattr(checkargs, "_pinned_report", lambda argv, pin: (None, "this tree (fixture)"))
    argv = " ".join(["--model", toy_parent, "--trunk-layers", "4", *X34_FORK])
    assert checkargs.main(["--argv", argv]) == int(TrainExitCode.FATAL_CONFIG)
    assert "still launches" not in capsys.readouterr().out
    assert checkargs.main(["--argv", " ".join(["--model", toy_parent, *X34_FORK])]) == 0
