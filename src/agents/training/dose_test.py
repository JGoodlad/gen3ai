"""The DOSE arithmetic and the `dose` provenance block.

The arithmetic is three lines and it is worth pinning anyway, because the whole point of the block
is that a reader trusts it without re-deriving it — and because `grad_accum_steps` belongs in the
DENOMINATOR (K micro-batches make ONE optimizer step), which is exactly the kind of fact a later
"simplification" gets backwards.

The reference row is v8's own recorded shape (`ai_v8_14_distill3_0725`: batch 2048, accum 16,
7 epochs, lr median 1.0040e-4 → 2.145e-8), so a change to this file's conventions fails against a
number the ledger already quotes rather than against one this file invented.
"""
from unittest.mock import MagicMock

import pytest

from agents.training.dose import (
    dose_block, dose_rate, effective_batch, kl_controller_block, kl_controller_snapshot,
    updates_per_env_step,
)


# ---------------------------------------------------------------------------
# THE ARITHMETIC
# ---------------------------------------------------------------------------


def test_grad_accum_multiplies_the_batch_and_DIVIDES_the_update_count():
    assert effective_batch(2048, 16) == 32768
    assert updates_per_env_step(batch_size=2048, grad_accum_steps=16, n_epochs=7) == \
        pytest.approx(7 / 32768)
    # The same LR at accum 2 vs accum 16 is an 8x dose difference — the thing the meter exists for.
    a = dose_rate(lr=1e-4, batch_size=2048, grad_accum_steps=2, n_epochs=7)
    b = dose_rate(lr=1e-4, batch_size=2048, grad_accum_steps=16, n_epochs=7)
    assert a / b == pytest.approx(8.0)


def test_the_v8_reference_row_reproduces_the_ledgers_number():
    assert dose_rate(lr=1.0039698565664741e-4, batch_size=2048, grad_accum_steps=16,
                     n_epochs=7) == pytest.approx(2.145e-8, rel=0.01)


@pytest.mark.parametrize("accum", [None, 0, 1])
def test_an_absent_or_degenerate_accum_means_ONE(accum):
    assert effective_batch(4096, accum) == 4096


def test_a_zero_batch_RAISES_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="positive"):
        updates_per_env_step(batch_size=0, grad_accum_steps=1, n_epochs=5)


def test_the_dose_is_linear_in_the_learning_rate():
    base = dose_rate(lr=1e-5, batch_size=4096, grad_accum_steps=2, n_epochs=10)
    assert dose_rate(lr=3e-5, batch_size=4096, grad_accum_steps=2,
                     n_epochs=10) == pytest.approx(3 * base)


# ---------------------------------------------------------------------------
# THE BLOCK
# ---------------------------------------------------------------------------


def _model(lr=1e-4, batch_size=2048, grad_accum_steps=16, n_epochs=7, **extra):
    m = MagicMock()
    m.policy.optimizer.param_groups = [{"lr": lr}]
    m.batch_size = batch_size
    m.grad_accum_steps = grad_accum_steps
    m.n_epochs = n_epochs
    m.num_timesteps = 42
    m._dose_lr_flag = 3e-4
    m._fork_lr_pin = None
    m._dose_kl = None
    for k, v in extra.items():
        setattr(m, k, v)
    return m


def test_the_block_states_what_the_flag_said_AND_what_the_optimizer_is_doing():
    """The whole reason the block exists: `--lr` is INERT on a resume, and that must be VISIBLE."""
    b = dose_block(_model(lr=1e-4))
    assert b["lr_now"] == pytest.approx(1e-4)
    assert b["lr_flag"] == pytest.approx(3e-4)      # what was typed, and ignored
    assert b["effective_batch"] == 32768
    assert b["updates_per_env_step"] == pytest.approx(7 / 32768)
    assert b["dose_rate_now"] == pytest.approx(1e-4 * 7 / 32768)
    assert b["fork_lr"] is None and b["lr_frozen"] is False


def test_dose_rate_now_is_lr_now_times_updates_per_env_step_exactly():
    b = dose_block(_model(lr=5.8e-5, batch_size=2048, grad_accum_steps=2, n_epochs=10))
    assert b["dose_rate_now"] == pytest.approx(b["lr_now"] * b["updates_per_env_step"])
    assert b["dose_rate_now"] == pytest.approx(1.419e-7, rel=0.01)   # the rev-2 row


def test_an_applied_pin_shows_up_as_fork_lr_and_the_pin_record():
    pin = {"lr": 7e-5, "frozen": True, "applied_at_step": 9, "source_model": "/p/c.zip"}
    b = dose_block(_model(lr=7e-5, _fork_lr_pin=pin))
    assert b["fork_lr"] == pytest.approx(7e-5)
    assert b["lr_frozen"] is True
    assert b["fork_lr_pin"] == pin


def test_the_kl_controller_config_rides_along():
    from agents.training.adaptive_lr_callback import AdaptivePPOCallback

    cb = AdaptivePPOCallback(initial_lr=1e-4, min_lr=1e-5, max_lr=6e-4, verbose=0)
    kl = dose_block(_model(_dose_kl=kl_controller_snapshot(cb)))["kl_controller"]
    assert kl == {"phase": "adaptive", "target_kl": 0.01, "kl_factor": 2.0, "lr_factor": 1.2,
                  "min_lr": 1e-5, "max_lr": 6e-4}


def test_a_two_phase_controller_reports_WHICH_PHASE_is_in_charge():
    from agents.training.adaptive_lr_callback import TwoPhaseLRCallback

    cb = TwoPhaseLRCallback(initial_lr=1e-4, total_steps=100, anneal_start_steps=50,
                            anneal_min_lr=1e-6, verbose=0)
    snap = kl_controller_snapshot(cb)
    assert kl_controller_block(snap, num_timesteps=10)["phase"] == "twophase_1"
    assert kl_controller_block(snap, num_timesteps=90)["phase"] == "twophase_2"
    cb.freeze_at(1e-4)
    assert kl_controller_block(kl_controller_snapshot(cb), num_timesteps=90)["phase"] == "frozen"


def test_no_controller_reports_None_rather_than_a_fabricated_default():
    assert kl_controller_snapshot(None) is None
    assert kl_controller_block(None) is None
    assert dose_block(_model())["kl_controller"] is None


def test_the_snapshot_is_PLAIN_DATA_and_survives_a_pickle():
    """The save hazard, as a test: a live callback on the model breaks every `model.save()`."""
    import pickle

    from agents.training.adaptive_lr_callback import AdaptivePPOCallback

    snap = kl_controller_snapshot(AdaptivePPOCallback(initial_lr=1e-4, verbose=0))
    assert pickle.loads(pickle.dumps(snap)) == snap
    assert all(isinstance(v, (int, float, bool, str)) for v in snap.values())


def test_model_build_stashes_the_SNAPSHOT_and_never_the_callback():
    import inspect

    from main.train import model_build
    src = inspect.getsource(model_build)
    assert "kl_controller_snapshot(lr_callback)" in src
    assert "_lr_callback = " not in src, (
        "a live LR callback on the model breaks cloudpickle at the first save")


def test_a_model_missing_every_optional_attribute_still_produces_a_block():
    """Provenance must never be what fails a save."""
    class Bare:
        batch_size = 4096
        n_epochs = 5

        class policy:  # noqa: N801 — a stand-in, not a class name anyone reads
            optimizer = None
    b = dose_block(Bare())
    assert b["lr_now"] is None and b["dose_rate_now"] is None
    assert b["effective_batch"] == 4096 and b["updates_per_env_step"] == pytest.approx(5 / 4096)
    assert b["lr_flag"] is None and b["fork_lr"] is None


def test_the_block_is_recorded_in_metadata_and_NEVER_in_model_config():
    """Provenance rule, root CLAUDE.md: `model_config.json` is the weight-shape record."""
    import inspect

    from agents.model.model_version import ModelVersion
    from main.train import run_io
    # It goes in via the ONE hparams dict, which `save_model_snapshot` merges into metadata.json.
    assert '"dose": dose_block(model)' in inspect.getsource(run_io._model_hparams)
    # And nowhere near the weight-shape record: no `ModelVersion` field is named for it, so
    # `check_compatible` can never reject a resume over a provenance number.
    assert not any("dose" in f for f in ModelVersion.__dataclass_fields__)


def test_the_tensorboard_callback_publishes_the_rate_and_the_effective_batch():
    from main.train.run_io import DoseLogCallback

    cb = DoseLogCallback()
    cb.model = _model(lr=1e-4)   # SB3's `logger` is a property over `self.model.logger`
    cb._on_rollout_end()
    recorded = dict(c.args for c in cb.model.logger.record.call_args_list)
    assert recorded["train/dose_rate"] == pytest.approx(1e-4 * 7 / 32768)
    assert recorded["train/effective_batch"] == 32768
