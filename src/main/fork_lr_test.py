"""`--fork-lr` / `--fork-lr-freeze` — the fork-vs-restart rule, the pin, and the freeze.

The claim these tests exist for is narrow and load-bearing: **a launcher PERIODIC RESTART must
never re-apply the pin.** The launcher re-invokes the same argv every `--restart-interval-hours`
into the same run dir, so a flag that fires "on resume" fires every few hours forever — which would
reset the KL controller's adapted rate on every restart, silently, for the life of the run.

Milliseconds — no torch model, no filesystem beyond `tmp_path`. Unmarked, fast inner loop.
"""
import json
from unittest.mock import MagicMock

import pytest

from agents.training.adaptive_lr_callback import AdaptivePPOCallback, TwoPhaseLRCallback
from main.train.fork_lr import (
    ForkLrDecision, apply_fork_lr_pin, build_pin_record, clamp_pin, is_same_run_checkpoint,
    read_recorded_pin, resolve_fork_lr,
)


# ---------------------------------------------------------------------------
# THE DISCRIMINATION RULE
# ---------------------------------------------------------------------------


def test_a_checkpoint_this_run_wrote_is_a_RESTART(tmp_path):
    run = tmp_path / "models" / "my_run"
    assert is_same_run_checkpoint(str(run / "checkpoints" / "checkpoint_5_steps.zip"), str(run))
    # The legacy layout kept checkpoints at the run root, and those runs still resume.
    assert is_same_run_checkpoint(str(run / "checkpoint_5_steps.zip"), str(run))
    assert is_same_run_checkpoint(str(run / "final_model.zip"), str(run))


def test_a_checkpoint_from_ANOTHER_run_is_a_FORK(tmp_path):
    run = tmp_path / "models" / "my_run"
    parent = tmp_path / "models" / "the_parent"
    assert not is_same_run_checkpoint(
        str(parent / "checkpoints" / "checkpoint_9_steps.zip"), str(run))
    # And a run whose NAME is a prefix of ours is not ours either.
    assert not is_same_run_checkpoint(str(tmp_path / "models" / "my_run_2" / "x.zip"), str(run))


def test_the_consensus_WARMSTART_is_a_fork_even_though_it_lives_inside_the_run_dir(tmp_path):
    """`--warmstart-consensus` re-points `--model` at `<run>/warmstart/warmstart_consensus.zip`.

    That is an INIT built from foreign teachers, not this run's own training progress, so it must
    read as a fork — otherwise a warm-started exploiter could never be pinned.
    """
    run = tmp_path / "models" / "my_run"
    assert not is_same_run_checkpoint(
        str(run / "warmstart" / "warmstart_consensus.zip"), str(run))


def test_a_fork_with_fork_lr_applies_the_pin(tmp_path):
    d = resolve_fork_lr(fork_lr=7e-5, fork_lr_freeze=False,
                        model_path=str(tmp_path / "parent" / "c.zip"), model_dir=str(tmp_path / "r"))
    assert d.apply and d.lr == pytest.approx(7e-5) and not d.frozen
    assert "FORK" in d.reason


def test_a_SAME_RUN_RESTART_does_NOT_re_apply_the_pin(tmp_path):
    """THE regression this whole module exists for."""
    run = tmp_path / "r"
    (run / "checkpoints").mkdir(parents=True)
    d = resolve_fork_lr(fork_lr=7e-5, fork_lr_freeze=False,
                        model_path=str(run / "checkpoints" / "checkpoint_5_steps.zip"),
                        model_dir=str(run))
    assert not d.apply and d.lr is None
    assert "restart" in d.reason and "NOT re-applied" in d.reason


def test_a_fork_WITHOUT_fork_lr_changes_nothing(tmp_path):
    d = resolve_fork_lr(fork_lr=None, fork_lr_freeze=False,
                        model_path=str(tmp_path / "p" / "c.zip"), model_dir=str(tmp_path / "r"))
    assert not d.apply and d.lr is None and not d.frozen


# ---------------------------------------------------------------------------
# THE FREEZE IS A PROPERTY OF THE RUN, so it DOES survive a restart
# ---------------------------------------------------------------------------


def _write_pin(run, lr, frozen):
    run.mkdir(parents=True, exist_ok=True)
    (run / "metadata.json").write_text(json.dumps(
        {"dose": {"fork_lr_pin": {"lr": lr, "frozen": frozen, "applied_at_step": 1}}}))


def test_a_frozen_run_re_applies_its_RECORDED_pin_on_a_restart(tmp_path):
    run = tmp_path / "r"
    _write_pin(run, 5e-5, frozen=True)
    d = resolve_fork_lr(fork_lr=None, fork_lr_freeze=False,   # a manual resume that dropped the flags
                        model_path=str(run / "checkpoints" / "c.zip"), model_dir=str(run))
    assert d.apply and d.lr == pytest.approx(5e-5) and d.frozen
    assert "FROZEN" in d.reason


def test_a_frozen_run_re_applies_from_the_ARGV_when_no_pin_is_on_record(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    d = resolve_fork_lr(fork_lr=5e-5, fork_lr_freeze=True,
                        model_path=str(run / "checkpoints" / "c.zip"), model_dir=str(run))
    assert d.apply and d.lr == pytest.approx(5e-5) and d.frozen


def test_an_UNFROZEN_recorded_pin_is_not_re_applied(tmp_path):
    run = tmp_path / "r"
    _write_pin(run, 5e-5, frozen=False)
    d = resolve_fork_lr(fork_lr=7e-5, fork_lr_freeze=False,
                        model_path=str(run / "checkpoints" / "c.zip"), model_dir=str(run))
    assert not d.apply


@pytest.mark.parametrize("payload", ["", "{", "{}", '{"dose": 3}', '{"dose": {}}'])
def test_an_unreadable_or_pinless_metadata_reads_as_NO_PIN(tmp_path, payload):
    run = tmp_path / "r"
    run.mkdir()
    (run / "metadata.json").write_text(payload)
    assert read_recorded_pin(str(run)) is None


def test_a_missing_metadata_reads_as_no_pin(tmp_path):
    assert read_recorded_pin(str(tmp_path / "nowhere")) is None


# ---------------------------------------------------------------------------
# THE PIN ITSELF — three sites must agree or it is a no-op somewhere
# ---------------------------------------------------------------------------


def _fake_model(lr=1e-4):
    model = MagicMock()
    model.policy.optimizer.param_groups = [{"lr": lr}]
    model.num_timesteps = 1234
    return model


def test_the_pin_moves_the_optimizer_the_schedule_and_the_controller():
    model = _fake_model(lr=2.8e-5)
    cb = AdaptivePPOCallback(initial_lr=2.8e-5, verbose=0)
    d = ForkLrDecision(True, 1e-4, False, "fork")
    rec = apply_fork_lr_pin(model, d, lr_callback=cb, min_lr=1e-5, max_lr=6e-4,
                            source_model="/parent/c.zip")
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert model.lr_schedule(0.5) == pytest.approx(1e-4)   # SB3 re-installs from this every train()
    assert cb.current_lr == pytest.approx(1e-4)            # the ladder starts HERE, not at 2.8e-5
    assert rec == {"lr": pytest.approx(1e-4), "frozen": False, "applied_at_step": 1234,
                   "source_model": "/parent/c.zip", "reason": "fork"}


def test_the_pin_still_respects_min_lr_and_max_lr():
    assert clamp_pin(1e-9, min_lr=1e-5, max_lr=6e-4) == pytest.approx(1e-5)
    assert clamp_pin(1.0, min_lr=1e-5, max_lr=6e-4) == pytest.approx(6e-4)
    model = _fake_model()
    apply_fork_lr_pin(model, ForkLrDecision(True, 1.0, False, "fork"),
                      lr_callback=None, min_lr=1e-5, max_lr=6e-4, source_model="x")
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(6e-4)


def test_not_applying_touches_nothing():
    model = _fake_model(lr=2.8e-5)
    cb = AdaptivePPOCallback(initial_lr=2.8e-5, verbose=0)
    assert apply_fork_lr_pin(model, ForkLrDecision(False, None, False, "restart"),
                             lr_callback=cb, min_lr=1e-5, max_lr=6e-4, source_model="x") is None
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(2.8e-5)
    assert cb.current_lr == pytest.approx(2.8e-5)


def test_the_pin_record_round_trips_through_metadata(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    rec = build_pin_record(ForkLrDecision(True, 9e-5, True, "why"), applied_lr=9e-5,
                           source_model="/p/c.zip", num_timesteps=7)
    (run / "metadata.json").write_text(json.dumps({"dose": {"fork_lr_pin": rec}}))
    assert read_recorded_pin(str(run)) == rec


# ---------------------------------------------------------------------------
# THE FREEZE holds the LR across a KL excursion that WOULD otherwise move it
# ---------------------------------------------------------------------------


def _attach(cb, num_timesteps=0, optimizer_lr=1e-4):
    cb.model = MagicMock()
    cb.model.num_timesteps = num_timesteps
    cb.model.logger.name_to_value = {}
    cb.model.n_epochs = 7
    cb.model.policy.optimizer.param_groups = [{"lr": optimizer_lr}]
    return cb


def _excursion(cb, kl):
    """One rollout at a KL far outside the band — the unfrozen controller MUST move on this."""
    cb.model.logger.name_to_value = {"train/approx_kl": kl}
    cb._on_rollout_end()


@pytest.mark.parametrize("kl", [0.30, 0.0001])   # far above the band, and far below it
def test_a_FROZEN_adaptive_controller_does_not_move_on_a_KL_excursion(kl):
    frozen = _attach(AdaptivePPOCallback(initial_lr=1e-4, cooldown_rollouts=0, verbose=0))
    frozen.freeze_at(1e-4)
    control = _attach(AdaptivePPOCallback(initial_lr=1e-4, cooldown_rollouts=0, verbose=0))
    for _ in range(5):
        _excursion(frozen, kl)
        _excursion(control, kl)
    assert control.current_lr != pytest.approx(1e-4), "the control arm must actually move"
    assert frozen.current_lr == pytest.approx(1e-4)
    assert frozen.model.lr_schedule(0.5) == pytest.approx(1e-4)


@pytest.mark.parametrize("kl", [0.30, 0.0001])
def test_a_FROZEN_two_phase_controller_does_not_move_in_PHASE_1(kl):
    frozen = _attach(TwoPhaseLRCallback(initial_lr=1e-4, total_steps=100, anneal_start_steps=90,
                                        anneal_min_lr=1e-5, cooldown_rollouts=0, verbose=0),
                     num_timesteps=10)
    frozen.freeze_at(1e-4)
    control = _attach(TwoPhaseLRCallback(initial_lr=1e-4, total_steps=100, anneal_start_steps=90,
                                         anneal_min_lr=1e-5, cooldown_rollouts=0, verbose=0),
                      num_timesteps=10)
    for _ in range(5):
        _excursion(frozen, kl)
        _excursion(control, kl)
    assert control.current_lr != pytest.approx(1e-4)
    assert frozen.current_lr == pytest.approx(1e-4)


def test_a_FROZEN_two_phase_controller_does_not_run_the_COSINE_either():
    """A freeze that only silenced the KL ladder would still be annealed away in Phase 2."""
    cb = _attach(TwoPhaseLRCallback(initial_lr=1e-4, total_steps=100, anneal_start_steps=10,
                                    anneal_min_lr=1e-6, handoff_lr=1e-4, verbose=0),
                 num_timesteps=55)
    cb.freeze_at(1e-4)
    cb._on_training_start()
    cb._on_rollout_end()
    assert cb.current_lr == pytest.approx(1e-4)
    assert cb.model.lr_schedule(0.5) == pytest.approx(1e-4)


def test_an_UNFROZEN_controller_is_byte_identical_to_before_the_flag_existed():
    """The default path must not have moved: `frozen` defaults False and changes nothing."""
    cb = _attach(AdaptivePPOCallback(initial_lr=1e-4, cooldown_rollouts=0, verbose=0))
    assert cb.frozen is False
    _excursion(cb, 0.30)
    assert cb.current_lr == pytest.approx(1e-4 / 1.2)


def test_apply_fork_lr_pin_freezes_the_controller_when_the_decision_says_so():
    model = _fake_model()
    cb = AdaptivePPOCallback(initial_lr=1e-4, cooldown_rollouts=0, verbose=0)
    apply_fork_lr_pin(model, ForkLrDecision(True, 5e-5, True, "fork"), lr_callback=cb,
                      min_lr=1e-5, max_lr=6e-4, source_model="x")
    assert cb.frozen is True and cb.current_lr == pytest.approx(5e-5)


# ---------------------------------------------------------------------------
# THE CONFIG REFUSALS
# ---------------------------------------------------------------------------


def _resolve(argv):
    from main.train_rl_agent import build_parser
    from main.train.config import resolve_config
    parser = build_parser()
    args = parser.parse_args(argv)
    resolve_config(args, parser)
    return args


@pytest.mark.parametrize("argv,needle", [
    (["--fork-lr", "7e-5"], "RESUME-ONLY"),
    (["--model", "x.zip", "--fork-lr-freeze"], "needs --fork-lr"),
    (["--model", "x.zip", "--fork-lr", "0"], "must be > 0"),
])
def test_the_refusals_fire_with_an_actionable_message(argv, needle, capsys):
    with pytest.raises(SystemExit):
        _resolve(argv)
    assert needle in capsys.readouterr().err


def test_a_fresh_run_with_no_fork_flags_still_resolves():
    args = _resolve([])
    assert args.fork_lr is None and args.fork_lr_freeze is False


def test_checkargs_accepts_both_flags():
    from main.checkargs import known_option_strings
    known = known_option_strings()
    assert known["--fork-lr"] == "fork_lr"
    assert known["--fork-lr-freeze"] == "fork_lr_freeze"


def test_the_flags_are_deliberately_NOT_extractor_toggles():
    """`flag_registry`'s scope is EXTRACTOR architecture. These reach the optimizer, not the model."""
    from agents.model.flag_registry import BY_NAME
    assert "fork_lr" not in BY_NAME and "fork_lr_freeze" not in BY_NAME


def test_the_model_build_resume_path_calls_the_resolver():
    """A source pin: the discrimination rule is useless if the resume path stops asking for it."""
    import inspect

    from main.train import model_build
    src = inspect.getsource(model_build)
    assert "resolve_fork_lr(" in src and "apply_fork_lr_pin(" in src
    assert "read_recorded_pin(" in src, (
        "a restart must carry the recorded pin forward, or the provenance block evaporates")
