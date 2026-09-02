"""THE OFF-SLICE DISTILL ANCHOR (`gen3_distill_offslice_anchor_v1`) — the four things that must hold.

1. **OFF IS FREE.** No parent attached ⇒ the parameter update is bit-identical to the tree before
   the feature existed, and a parent attached at coefficient 0 (`--distill-anchor-monitor`) is
   bit-identical TOO — the monitor arm must be a pure instrument, not a slightly different run.
2. **THE KL IS A KL.** Zero when the student IS the parent, positive once it moves, masked to the
   legal set, and — under `off_slice` — carrying literally no gradient on the taught rows.
3. **THE SLICE SPLITS.** `collateral_kl` reads the off-slice rows and `on_slice_kl` the on-slice
   ones, on a batch where the two answers are constructed to differ.
4. **THE PARENT SURVIVES A RESTART.** The resolution prefers the run's IMMUTABLE `original_command`
   over this process's `--model`, because an idempotent fork's `--model` is swapped to the fork's
   own latest checkpoint on every relaunch. Getting this wrong makes the anchor a silent no-op that
   still reads as ON, which is why it is pinned by a test that reproduces the swap.
"""
import copy
import json

import numpy as np
import torch as th
from gymnasium import spaces

from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.instrumented_ppo.distill_anchor import (
    ANCHOR_MODES,
    anchor_loss_and_metrics,
    anchor_row_weights,
    distill_anchor_step,
    masked_kl_rows,
)
from agents.training.distill_anchor_callback import (
    ANCHOR_PARENT_ROUTES,
    DistillAnchorCallback,
    parse_model_arg,
    read_original_command,
    resolve_anchor_parent,
)
from agents.training.instrumented_ppo_test import (
    _CounterDictEnv, _build_tiny_ppo, _train_from_init,
)


# ======================================================================================
# 1. The pure KL + slice math
# ======================================================================================

def test_kl_is_zero_when_the_student_is_the_parent():
    th.manual_seed(0)
    logits = th.randn(6, 5)
    rows = masked_kl_rows(logits, logits.clone(), th.ones(6, 5))
    assert float(rows.abs().max()) < 1e-6


def test_kl_is_positive_once_the_student_moves():
    th.manual_seed(0)
    parent = th.randn(6, 5)
    rows = masked_kl_rows(parent, parent + th.randn(6, 5) * 1.5, th.ones(6, 5))
    assert float(rows.min()) >= 0.0            # a KL is never negative
    assert float(rows.mean()) > 1e-3


def test_kl_ignores_illegal_actions():
    """An illegal column may hold anything; both sides renormalise over the legal set, so a change
    confined to an illegal column must not move the KL by a single ulp."""
    th.manual_seed(1)
    parent, student = th.randn(4, 5), th.randn(4, 5)
    amask = th.ones(4, 5)
    amask[:, 3] = 0.0
    base = masked_kl_rows(parent, student, amask)
    parent2, student2 = parent.clone(), student.clone()
    parent2[:, 3] += 12.0
    student2[:, 3] -= 30.0
    assert th.allclose(base, masked_kl_rows(parent2, student2, amask), atol=1e-6)


def test_row_weights_split_the_batch_by_teacher_id():
    tid = th.tensor([[0.], [1.], [0.], [2.]])           # integer teacher-id: 0 = off-slice
    w, off = anchor_row_weights(tid, "off_slice", th.float32)
    assert w.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert off.tolist() == [1.0, 0.0, 1.0, 0.0]
    w_all, off_all = anchor_row_weights(tid, "all", th.float32)
    assert w_all.tolist() == [1.0, 1.0, 1.0, 1.0]       # `all` anchors every row...
    assert off_all.tolist() == [1.0, 0.0, 1.0, 0.0]     # ...but the METER is still the off-slice one


def test_modes_are_exactly_the_two_the_flag_offers():
    assert ANCHOR_MODES == ("off_slice", "all")


def _split_batch():
    """A batch whose two slices genuinely disagree: rows 0/2 (off-slice) drift hard from the
    parent, rows 1/3 (on-slice, teacher 1) sit on it. Constructed so `collateral_kl >> on_slice_kl`
    — a meter that silently pooled the two would read neither."""
    th.manual_seed(3)
    parent = th.randn(4, 5)
    student = parent.clone()
    student[0] += th.randn(5) * 3.0
    student[2] += th.randn(5) * 3.0
    tid = th.tensor([[0.], [1.], [0.], [1.]])
    return parent, student, th.ones(4, 5), tid


def test_metrics_split_the_slices():
    parent, student, amask, tid = _split_batch()
    kl, m = anchor_loss_and_metrics(parent, student, amask, tid, mode="off_slice")
    assert m["off_slice_frac"] == 0.5
    assert m["collateral_kl"] > 1e-2                 # the rows that moved
    assert m["on_slice_kl"] < 1e-6                   # the rows that did not
    assert abs(float(kl) - m["collateral_kl"]) < 1e-6   # off_slice ⇒ the loss IS the collateral meter
    assert m["anchor_n"] == 2.0


def test_mode_all_anchors_every_row_but_the_meters_do_not_move():
    parent, student, amask, tid = _split_batch()
    _, m_off = anchor_loss_and_metrics(parent, student, amask, tid, mode="off_slice")
    kl_all, m_all = anchor_loss_and_metrics(parent, student, amask, tid, mode="all")
    assert m_all["collateral_kl"] == m_off["collateral_kl"]   # the METER is mode-invariant...
    assert m_all["on_slice_kl"] == m_off["on_slice_kl"]
    assert m_all["anchor_n"] == 4.0                           # ...the LOSS's support is not
    rows = masked_kl_rows(parent, student, amask)
    assert abs(float(kl_all) - float(rows.mean())) < 1e-6


def test_empty_off_slice_yields_no_term_but_still_reports():
    """Every row on a teacher's team: the anchor has nothing to constrain, so the KL is None (the
    `_distill_loss` None-guard convention — an empty subset must never NaN-poison the loss), and
    `on_slice_kl` is still reported so the series does not gap."""
    th.manual_seed(4)
    parent = th.randn(3, 5)
    # A UNIFORM shift would be invisible (softmax is shift-invariant) — the student has to actually
    # hold a different distribution for `on_slice_kl` to have anything to report.
    kl, m = anchor_loss_and_metrics(parent, parent + th.randn(3, 5), th.ones(3, 5),
                                    th.ones(3, 1), mode="off_slice")
    assert kl is None
    assert m["off_slice_frac"] == 0.0
    assert "collateral_kl" not in m
    assert m["on_slice_kl"] > 0.0


def test_off_slice_mode_puts_zero_gradient_on_the_taught_rows():
    """THE pin for 'the anchor never fights the teacher'. Under `off_slice` the gradient w.r.t. the
    student's logits must be EXACTLY zero on every on-slice row — not merely small."""
    parent, student, amask, tid = _split_batch()
    student = student.detach().requires_grad_(True)
    kl, _ = anchor_loss_and_metrics(parent, student, amask, tid, mode="off_slice")
    kl.backward()
    g = student.grad
    assert float(g[1].abs().max()) == 0.0 and float(g[3].abs().max()) == 0.0
    assert float(g[0].abs().max()) > 0.0            # ...and non-zero where it does apply


def test_mode_all_does_carry_gradient_on_the_taught_rows():
    """The contrast that makes the test above mean something: `all` is a real alternative arm."""
    parent, student, amask, tid = _split_batch()
    student = student.detach().requires_grad_(True)
    kl, _ = anchor_loss_and_metrics(parent, student, amask, tid, mode="all")
    kl.backward()
    assert float(student.grad[1].abs().max()) > 0.0


# ======================================================================================
# 2. Parent resolution — the restart rule
# ======================================================================================

def test_parse_model_arg_handles_every_spelling():
    assert parse_model_arg("python x.py --model models/a/final_model.zip --steps 3") \
        == "models/a/final_model.zip"
    assert parse_model_arg("python x.py --model=models/b.zip") == "models/b.zip"
    assert parse_model_arg("python x.py --model_path models/c.zip") == "models/c.zip"
    assert parse_model_arg("python x.py --steps 3") is None
    assert parse_model_arg("") is None
    assert parse_model_arg("python 'unterminated") is None      # total, never raises


def test_resolve_prefers_an_explicit_pin(tmp_path):
    path, route = resolve_anchor_parent(explicit="models/pinned.zip",
                                        run_dir=str(tmp_path), cli_model="models/drifted.zip")
    assert (path, route) == ("models/pinned.zip", "explicit")


def test_resolve_falls_back_to_the_cli_model_on_a_forks_first_launch(tmp_path):
    """A fresh fork dir holds no metadata.json (`resolve_launch_run_dir` refuses to fork onto an
    existing run), so `--model` IS the fork parent and route 2 cannot fire with a wrong answer."""
    path, route = resolve_anchor_parent(explicit=None, run_dir=str(tmp_path),
                                        cli_model="models/parent/final_model.zip")
    assert (path, route) == ("models/parent/final_model.zip", "cli_model")


def test_resolve_reads_the_fork_parent_from_original_command_not_the_drifted_checkpoint(tmp_path):
    """🚨 THE RESTART PIN. Second launch of an idempotent fork: the launcher has swapped `--model`
    to the fork's OWN latest checkpoint, so `cli_model` is the DRIFTED policy. The anchor must
    still resolve the ORIGINAL fork parent, from the immutable `original_command`."""
    (tmp_path / "metadata.json").write_text(json.dumps({
        "original_command": ("python -m main.launcher --model "
                             "models/ai_v9_59_R2ACTION_0827/final_model.zip --run-name fold"),
        "cli_args": {"model": "models/fold/checkpoints/checkpoint_900_steps.zip"},
    }))
    path, route = resolve_anchor_parent(
        explicit=None, run_dir=str(tmp_path),
        cli_model="models/fold/checkpoints/checkpoint_900_steps.zip")
    assert path == "models/ai_v9_59_R2ACTION_0827/final_model.zip"
    assert route == "original_command"


def test_resolve_is_stable_across_many_restarts(tmp_path):
    """The same drifted `--model` at three successive restarts resolves to ONE parent — the failure
    this design exists to prevent is a trust region that ratchets along with the student."""
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"original_command": "python t.py --model models/base/final_model.zip"}))
    got = {resolve_anchor_parent(explicit=None, run_dir=str(tmp_path),
                                 cli_model=f"models/fold/checkpoints/checkpoint_{n}_steps.zip")[0]
           for n in (100, 200, 300)}
    assert got == {"models/base/final_model.zip"}


def test_resolve_unresolved_when_nothing_names_a_parent(tmp_path):
    assert resolve_anchor_parent(explicit=None, run_dir=str(tmp_path), cli_model=None) \
        == (None, "unresolved")


def test_read_original_command_tolerates_junk(tmp_path):
    assert read_original_command(str(tmp_path)) is None            # no file
    (tmp_path / "metadata.json").write_text("{not json")
    assert read_original_command(str(tmp_path)) is None            # unparseable
    (tmp_path / "metadata.json").write_text(json.dumps({"original_command": "   "}))
    assert read_original_command(str(tmp_path)) is None            # empty


def test_routes_are_the_documented_three():
    assert ANCHOR_PARENT_ROUTES == ("explicit", "original_command", "cli_model")


def test_the_callback_attaches_the_parent_and_the_hparams_on_every_launch():
    """`_on_training_start` — not model construction — is the hook, precisely because it runs again
    on every launcher restart. Here: it sets all three hparams and loads the parent FROM THE PATH."""
    class _Stub:
        pass

    seen = []

    def _loader(path):
        seen.append(path)
        return f"parent<{path}>"

    cb = DistillAnchorCallback(parent_path="models/base/final_model.zip", route="original_command",
                               coef=0.05, mode="off_slice", monitor=False, load_parent=_loader)
    cb.model = _Stub()
    cb._on_training_start()
    assert cb.model.distill_anchor_coef == 0.05
    assert cb.model.distill_anchor_mode == "off_slice"
    assert cb.model.distill_anchor_monitor is False
    assert cb.model._distill_anchor_parent == "parent<models/base/final_model.zip>"
    # A SECOND launch (the restart) re-loads from the same path rather than reusing anything.
    cb.model = _Stub()
    cb._on_training_start()
    assert seen == ["models/base/final_model.zip"] * 2


def test_monitor_only_still_attaches_the_parent():
    """coef 0 + monitor is the pure-instrument arm — it must LOAD (there is nothing to measure
    against otherwise) while folding no term."""
    class _Stub:
        pass

    cb = DistillAnchorCallback(parent_path="p", route="explicit", coef=0.0, mode="off_slice",
                               monitor=True, load_parent=lambda p: "loaded")
    cb.model = _Stub()
    cb._on_training_start()
    assert cb.model._distill_anchor_parent == "loaded"
    assert cb.model.distill_anchor_coef == 0.0
    assert cb.model.distill_anchor_monitor is True


# ======================================================================================
# 3. End to end through a real train()
# ======================================================================================

class _MixedDistillEnv(_CounterDictEnv):
    """`_CounterDictEnv` plus a `distill_mask` that ALTERNATES between teacher 1 and no-teacher, so
    every minibatch carries both slices — the only shape in which the anchor and its meters are
    both exercised."""

    def __init__(self, ep_len=1000):
        super().__init__(ep_len=ep_len)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1e4, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(2,), dtype=np.int8),
            "distill_mask": spaces.Box(0.0, 4.0, shape=(1,), dtype=np.float32),
        })

    def _obs(self):
        o = super()._obs()
        o["distill_mask"] = np.array([float(self._t % 2)], dtype=np.float32)
        return o


class _Rec:
    def __init__(self):
        self.vals = {}

    def record(self, k, v, *a, **kw):
        self.vals[k] = v

    def __getattr__(self, _n):
        return lambda *a, **kw: None


def _build_anchor_ppo(n_steps=8, n_envs=4):
    """A tiny PPO on the mixed-slice env, plus a frozen PARENT built on the PLAIN env — mirroring
    production, where the parent is a checkpoint whose obs space does not carry the training-only
    `distill_mask` key and the step filters the obs down to what it knows."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([(lambda: _MixedDistillEnv()) for _ in range(n_envs)])
    model = InstrumentedMaskablePPO(
        "MultiInputPolicy", venv, n_steps=n_steps, batch_size=4, n_epochs=1,
        normalize_advantage=False, ent_coef=0.0, vf_coef=0.5, device="cpu", seed=0)
    parent, _ = _build_tiny_ppo(n_steps=n_steps, n_envs=n_envs)
    th.manual_seed(11)                       # FIXED: the test must not flap
    with th.no_grad():
        for p in parent.policy.action_net.parameters():
            p.add_(th.randn_like(p) * 2.0)
    parent.policy.set_training_mode(False)
    return model, parent


def _anchor_arm(*, attach_parent, coef, mode="off_slice"):
    th.manual_seed(0)
    np.random.seed(0)
    model, parent = _build_anchor_ppo()
    model.learn(total_timesteps=8 * 4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    if attach_parent:
        model._distill_anchor_parent = parent
    model.distill_anchor_coef = coef
    model.distill_anchor_mode = mode
    model._logger = _Rec()
    sd = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    return sd, model.logger.vals


def test_no_parent_is_byte_identical_and_logs_nothing():
    """The default: the flag was never passed, so `train()` must be the function it was before."""
    base, log_base = _anchor_arm(attach_parent=False, coef=0.0)
    assert not [k for k in log_base if "collateral" in k or "anchor" in k]
    # A second identically-seeded arm reproduces it exactly — the equality below is meaningful only
    # because this toy IS deterministic across fresh, identically-seeded models.
    again, _ = _anchor_arm(attach_parent=False, coef=0.0)
    for k in base:
        assert th.equal(base[k], again[k]), f"the no-parent arm is not reproducible at {k}"


def test_monitor_only_is_byte_identical_and_emits_every_meter():
    """`--distill-anchor-coef 0 --distill-anchor-monitor`: the parent IS loaded and forwarded, and
    every meter appears — but not one parameter moves. That is what makes the monitored control arm
    comparable with the folded arm rather than a third condition."""
    base, _ = _anchor_arm(attach_parent=False, coef=0.0)
    mon, log = _anchor_arm(attach_parent=True, coef=0.0)
    for k in base:
        assert th.equal(base[k], mon[k]), f"monitor-only perturbed {k}"
    assert log["distill/collateral_kl"] > 0.0
    assert log["distill/on_slice_kl"] > 0.0
    assert log["distill/anchor_loss"] == 0.0        # measured zero, not a missing series
    assert 0.0 < log["distill/off_slice_frac"] < 1.0
    assert log["distill/anchor_kl"] > 0.0


def test_a_live_coefficient_changes_the_update():
    base, _ = _anchor_arm(attach_parent=False, coef=0.0)
    on, log = _anchor_arm(attach_parent=True, coef=0.5)
    assert any(not th.equal(base[k], on[k]) for k in base), \
        "distill_anchor_coef > 0 folded nothing into the update"
    assert log["distill/anchor_loss"] > 0.0


def test_a_real_fold_reports_content_and_damage_side_by_side():
    """THE point of the feature. One arm with a live TEACHER *and* a live anchor must emit, in the
    same `train()`, both halves of the trade: absorption (`teacher_agreement_on_slice`) and damage
    (`collateral_kl`). The third number — `grad/distill_anchor_share` — needs a shared TRUNK the
    grad-balance probe can measure, which this toy MultiInputPolicy does not have; its wiring is
    pinned by `test_the_anchor_is_registered_for_a_grad_share` instead."""
    th.manual_seed(0)
    np.random.seed(0)
    model, parent = _build_anchor_ppo()
    teacher, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    th.manual_seed(21)
    with th.no_grad():
        for p in teacher.policy.action_net.parameters():
            p.add_(th.randn_like(p) * 2.0)
    teacher.policy.set_training_mode(False)
    model.learn(total_timesteps=8 * 4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model._distill_teachers = [teacher]
    model.distill_coef = 0.2
    model._distill_anchor_parent = parent
    model.distill_anchor_coef = 0.05
    model._logger = _Rec()
    _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    log = model.logger.vals
    assert 0.0 <= log["distill/teacher_agreement_on_slice"] <= 1.0
    assert log["distill/collateral_kl"] > 0.0
    assert log["distill/kl"] > 0.0                       # the teacher term is live in the same call


def test_the_anchor_is_registered_for_a_grad_share():
    """`grad/distill_anchor_share` exists only if the term is handed to the grad-balance probe under
    that name, and only a run with a Gen3 shared trunk emits it — so the wiring is pinned at the
    source, beside the fold that must not silently lose it.

    It rides the `distill` NOISE group for the same reason it rides the aux probe: it is part of the
    fold's dose, not a supervised head."""
    import inspect
    from agents.training.instrumented_ppo import ppo as _ppo
    src = inspect.getsource(_ppo.InstrumentedMaskablePPO.train)
    assert 'aux_probe_terms["distill_anchor"] = anchor_term' in src
    assert '_ntg.add("distill", anchor_term)' in src


def test_the_step_skips_when_the_rollout_has_no_distill_mask():
    """A run whose env never emits `distill_mask` (no live distillation) has no slice to split, so
    the step must decline rather than guess — the config layer refuses this combination up front,
    and this is the belt to that brace."""
    class _NoMask:
        observations = {"observation": th.zeros(2, 1)}
        action_masks = th.ones(2, 2)

    class _Model:
        _distill_anchor_parent = object()
        distill_anchor_coef = 0.5

    class _Pi:
        class distribution:
            logits = th.zeros(2, 2)

    out = {}
    assert distill_anchor_step(_Model(), _NoMask(), _Pi(), out) is None
    assert out == {}


# ======================================================================================
# 4. The CLI surface
# ======================================================================================

def test_checkargs_accepts_the_anchor_flags():
    """`python -m main.checkargs` must recognise every new flag, so a recorded command carrying one
    still validates offline (the whole point of that tool is that a stale argv is found in one pass,
    not in a launch-crash-fix loop)."""
    from main.checkargs import check
    got = check(["--distill-teacher", "models/t:data/teams/sample/a.txt",
                 "--distill-coef", "0.1", "--distill-anchor-coef", "0.02",
                 "--distill-anchor-mode", "all", "--distill-anchor-monitor",
                 "--distill-anchor-parent", "models/base/final_model.zip"])
    assert got["unknown"] == []
    for f in ("--distill-anchor-coef", "--distill-anchor-mode",
              "--distill-anchor-monitor", "--distill-anchor-parent"):
        assert f in got["accepted"]


def test_config_refuses_an_anchor_without_a_live_distill():
    """The dependency is structural: the anchor's slice IS the `distill_mask` obs key, which the env
    emits only for a run with a live distill term."""
    import pytest
    from main.train.config import resolve_config
    from main.train_rl_agent import build_parser
    p = build_parser()
    args = p.parse_args(["--distill-anchor-coef", "0.02", "--steps", "10"])
    with pytest.raises(SystemExit):
        resolve_config(args, p)


def test_the_anchor_hparams_default_to_off_on_the_class():
    assert InstrumentedMaskablePPO.distill_anchor_coef == 0.0
    assert InstrumentedMaskablePPO.distill_anchor_mode == "off_slice"
    assert InstrumentedMaskablePPO.distill_anchor_monitor is False


def test_the_frozen_parent_is_never_pickled_into_a_checkpoint():
    """Re-loading from the fork-parent path on every restart is the whole correctness argument; a
    pickled parent would be a second, wrong answer that survives restarts."""
    model, _venv = _build_tiny_ppo()
    assert "_distill_anchor_parent" in model._excluded_save_params()
