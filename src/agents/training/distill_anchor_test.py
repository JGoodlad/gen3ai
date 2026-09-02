"""THE OFF-SLICE DISTILL ANCHOR (`gen3_distill_offslice_anchor_v1`) — the four things that must hold.

1. **OFF IS FREE.** No parent attached ⇒ the parameter update is bit-identical to the tree before
   the feature existed, and a parent attached at coefficient 0 (`--distill-anchor-monitor`) is
   bit-identical TOO — the monitor arm must be a pure instrument, not a slightly different run.
2. **THE KL IS A KL.** Zero when the student IS the parent, positive once it moves, masked to the
   legal set, and — under `off_slice` — carrying literally no gradient on the taught rows.
3. **THE SLICE SPLITS.** `collateral_kl` reads the off-slice rows and `on_slice_kl` the on-slice
   ones, on a batch where the two answers are constructed to differ.
4. **THE REFERENCE SURVIVES A RESTART.** Both halves: the FIXED parent is re-read from its own
   PATH (never from the drifted checkpoint), and a MOVING reference — which has no path, being a
   function of this run's own trajectory — is persisted beside the checkpoint and restored from
   there. Getting either wrong makes the anchor a silent no-op that still reads as ON, which is why
   both are pinned by tests that reproduce a launcher restart.
5. **THE PARENT SURVIVES A RESTART.** The resolution prefers the run's IMMUTABLE `original_command`
   over this process's `--model`, because an idempotent fork's `--model` is swapped to the fork's
   own latest checkpoint on every relaunch. Getting this wrong makes the anchor a silent no-op that
   still reads as ON, which is why it is pinned by a test that reproduces the swap.
"""
import copy
import json
import os

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


def test_modes_are_exactly_the_ones_the_flag_offers():
    """`grad_project` (gen3_distill_grad_project_v1) is a THIRD mode, and it is a different
    MECHANISM rather than a third row set — it projects the DISTILL gradient off the off-slice
    behaviour subspace at every step. Its OUTPUT half is `off_slice`'s, pinned just below; the
    projection itself is pinned by `instrumented_ppo_distill_grad_project_test.py`."""
    assert ANCHOR_MODES == ("off_slice", "all", "grad_project")


def test_grad_project_takes_the_off_slice_row_weights_for_its_output_half():
    tid = th.tensor([0.0, 1.0, 0.0, 2.0])
    w, off = anchor_row_weights(tid, "grad_project", th.float32)
    w_off, off_off = anchor_row_weights(tid, "off_slice", th.float32)
    assert w.tolist() == w_off.tolist() and off.tolist() == off_off.tolist()


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


def test_routes_are_the_documented_four():
    # gen3_run_lineage_v1 inserted "lineage" (metadata's first-class block) ahead of the legacy
    # "original_command" derivation, which is now the accessor's warn-path, not an inline parse.
    assert ANCHOR_PARENT_ROUTES == ("explicit", "lineage", "original_command", "cli_model")


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


# ======================================================================================
# 5. THE REFERENCE — `--distill-anchor-ref {parent,ema,periodic}`
#
# The anchor's reference was FIXED at the frozen fold parent, and the question that produced this
# section is why it is not refreshed the way PPO's clip reference is. The answer is that they bound
# different quantities: the clip bounds the per-update RATE against the data-collecting policy, and
# the anchor bounds the ACCUMULATED DISPLACEMENT from the fold start — which is what the licensing
# probe measured, and which is SYSTEMATIC (the same off-slice direction every step), so a following
# reference barely resists it. A fixed reference is Learning-without-Forgetting; a Polyak-averaged
# one is ACER's trust region. Both alternatives ship OPT-IN so one cell can compare all three, and
# what has to hold is: the default is unchanged, the two degenerate settings collapse onto the modes
# they should, the periodic cadence is the declared one, the state survives a restart, and a
# reference that cannot be restored says so instead of silently resetting to fold start.
# ======================================================================================

from agents.training.instrumented_ppo.distill_anchor import (      # noqa: E402
    ANCHOR_REFS, frozen_logits, offslice_kl,
)
from agents.training.distill_anchor_callback import (              # noqa: E402
    ANCHOR_REF_SCHEMA, ANCHOR_REF_SUFFIX, anchor_ref_path, assert_reference_matches_student,
    ema_window, polyak_update_, save_anchor_ref_beside,
)


def test_refs_are_exactly_the_three_the_flag_offers():
    assert ANCHOR_REFS == ("parent", "ema", "periodic")


def test_the_ema_window_arithmetic_is_the_one_the_help_string_quotes():
    """`1/(1-tau)` TRAIN() CALLS. The env-step figure a reader actually needs is that window times
    the rollout, and at the production shape a rollout is n_envs*n_steps = 48*2048 = 98,304 steps."""
    assert abs(ema_window(0.99) - 100.0) < 1e-6
    assert abs(ema_window(0.9) - 10.0) < 1e-9
    assert ema_window(1.0) == float("inf")             # tau 1.0 IS `parent`: the reference is fixed
    assert abs(ema_window(0.99) * 48 * 2048 - 9_830_400) < 1.0


def test_the_reference_sibling_path_is_derived_from_the_checkpoint():
    assert anchor_ref_path("m/run/checkpoints/checkpoint_900_steps.zip") == \
        "m/run/checkpoints/checkpoint_900_steps" + ANCHOR_REF_SUFFIX
    assert anchor_ref_path("m/run/final_model_interrupted.zip").endswith(ANCHOR_REF_SUFFIX)
    assert anchor_ref_path("m/run/x") == "m/run/x" + ANCHOR_REF_SUFFIX      # no extension: appended


def test_polyak_update_is_the_convex_combination_it_claims():
    import torch.nn as nn
    ref, stu = nn.Linear(3, 2), nn.Linear(3, 2)
    with th.no_grad():
        ref.weight.fill_(0.0)
        stu.weight.fill_(1.0)
    polyak_update_(ref, stu, 0.75)
    assert th.allclose(ref.weight, th.full_like(ref.weight, 0.25))
    polyak_update_(ref, stu, 0.75)                       # 0.75*0.25 + 0.25*1.0
    assert th.allclose(ref.weight, th.full_like(ref.weight, 0.4375))


def test_polyak_at_tau_one_never_moves_and_at_tau_zero_becomes_the_student():
    import torch.nn as nn
    ref, stu = nn.Linear(3, 2), nn.Linear(3, 2)
    with th.no_grad():
        ref.weight.fill_(0.0)
        stu.weight.fill_(7.0)
    polyak_update_(ref, stu, 1.0)
    assert float(ref.weight.abs().max()) == 0.0          # tau=1 IS `parent`
    polyak_update_(ref, stu, 0.0)
    assert th.equal(ref.weight, stu.weight)              # tau=0 IS the current student


def test_polyak_copies_non_float_entries_rather_than_averaging_them():
    """An average of two integer counters is not a counter. `num_batches_tracked` is the canonical
    one; averaging it would produce a value neither side ever held."""
    import torch.nn as nn
    ref, stu = nn.BatchNorm1d(3), nn.BatchNorm1d(3)
    with th.no_grad():
        ref.num_batches_tracked.fill_(4)
        stu.num_batches_tracked.fill_(10)
        ref.running_mean.fill_(0.0)
        stu.running_mean.fill_(1.0)
    polyak_update_(ref, stu, 0.5)
    assert int(ref.num_batches_tracked) == 10                       # COPIED
    assert th.allclose(ref.running_mean, th.full_like(ref.running_mean, 0.5))   # AVERAGED


def test_periodic_at_a_zero_cadence_collapses_to_parent():
    """Documented as '0 = never = parent'; collapsed at construction so nothing downstream carries
    the special case, and asserted here so the help string is a fact rather than a promise."""
    cb = DistillAnchorCallback(parent_path="p", route="explicit", coef=0.1, mode="off_slice",
                               monitor=False, load_parent=lambda _p: object(),
                               ref="periodic", refresh_every=0)
    assert cb.ref == "parent"


def test_an_unknown_reference_raises_at_construction():
    import pytest
    with pytest.raises(ValueError):
        DistillAnchorCallback(parent_path="p", route="explicit", coef=0.1, mode="off_slice",
                              monitor=False, load_parent=lambda _p: object(), ref="clip")


# --------------------------------------------------------------------------------------
# The moving reference on a REAL policy
# --------------------------------------------------------------------------------------

def _build_moving_pair(n_steps=8, n_envs=4):
    """``(student, frozen_parent, loader)`` — all arch-IDENTICAL, on the mixed-slice env.

    `loader` is the injected `load_parent`, and it returns a FRESH, INDEPENDENT model every call —
    which is production's shape exactly: `load_foreign_opponent` reads the zip again. That
    independence is load-bearing here, because the moving reference is a SECOND load of the parent
    and a loader that handed back one shared object would make every `polyak_update_` mutate the
    frozen parent the displacement meter reads.

    Arch-identical because a moving reference must share ONE `state_dict` with the student, which is
    true on any genuine fold. `_build_anchor_ppo`'s deliberately-narrower parent models the obs-key
    filtering instead, and both shapes are exercised.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _mk(seed):
        venv = DummyVecEnv([(lambda: _MixedDistillEnv()) for _ in range(n_envs)])
        return InstrumentedMaskablePPO(
            "MultiInputPolicy", venv, n_steps=n_steps, batch_size=4, n_epochs=1,
            normalize_advantage=False, ent_coef=0.0, vf_coef=0.5, device="cpu", seed=seed)

    model = _mk(0)
    parent = _mk(1)
    th.manual_seed(11)                       # FIXED: the test must not flap
    with th.no_grad():
        for p in parent.policy.action_net.parameters():
            p.add_(th.randn_like(p) * 2.0)
    parent.policy.set_training_mode(False)
    _weights = copy.deepcopy(parent.policy.state_dict())
    _calls = []

    def _loader(path):
        _calls.append(path)
        if len(_calls) == 1:
            return parent                    # the FROZEN parent: the first load
        fresh = _mk(2 + len(_calls))
        fresh.policy.load_state_dict(_weights)
        fresh.policy.set_training_mode(False)
        return fresh

    _loader.calls = _calls
    return model, parent, _loader


def _attach(model, loader, *, ref, tau=0.99, refresh_every=8, coef=0.5, run_dir=None,
            resume_model=None, expect_restore=False, parent_path="models/base/final_model.zip"):
    cb = DistillAnchorCallback(parent_path=parent_path, route="explicit",
                               coef=coef, mode="off_slice", monitor=False,
                               load_parent=loader, ref=ref, ema_tau=tau,
                               refresh_every=refresh_every, run_dir=run_dir,
                               resume_model=resume_model, expect_restore=expect_restore)
    cb.model = model
    # gen3_distill_stop_rule_v1: `_on_rollout_end` now records `distill/anchor_coef` every rollout
    # (a FLAT series under a static coefficient, on purpose — see `_step_dual`), so these
    # bare-model unit tests need a real Logger. A no-op logger here would make the one thing the
    # dual-ascent arm is read on untestable at this seam.
    from stable_baselines3.common.logger import Logger as _L
    model.set_logger(_L(folder=None, output_formats=[]))
    cb._on_training_start()
    return cb


def _sd_equal(a, b) -> bool:
    return set(a) == set(b) and all(th.equal(a[k], b[k]) for k in a)


def test_the_moving_reference_starts_AT_the_parent():
    """All three modes coincide at fold start — the ema/periodic arms are a claim about how the
    reference MOVES, not about where it begins, and beginning anywhere else would confound them."""
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="ema", tau=0.9)
    assert _sd_equal(cb._ref_policy.state_dict(), parent.policy.state_dict())
    assert model._distill_anchor_ref is not parent
    assert model._distill_anchor_ref.policy is cb._ref_policy


def test_parent_mode_aliases_the_frozen_parent_and_stores_no_policy():
    """The default arm must run exactly ONE frozen forward, which is what `ref is parent` in
    `distill_anchor_step` keys off — so `parent` mode may not build a copy at all."""
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="parent")
    assert model._distill_anchor_ref is parent
    assert cb._ref_policy is None
    assert len(loader.calls) == 1, "`parent` mode must not pay for a second load"
    assert cb.save_reference("/nonexistent/ckpt.zip") is None      # nothing to persist, no write
    assert save_anchor_ref_beside(model, "/nonexistent/ckpt.zip") is None


def test_the_moving_reference_is_a_SECOND_LOAD_and_never_touches_the_frozen_parent():
    """🚨 THE TWO REASONS IT IS NOT A `deepcopy(model.policy)`. The extractor carries a per-forward
    stash of NON-LEAF tensors, which `deepcopy` refuses outright; and `--compile-trainer` patches
    the BOUND `fe.forward` as an INSTANCE attribute, which `deepcopy` treats as ATOMIC — so the
    copy's forward would still be closed over the LIVE extractor and every "frozen reference" logit
    would silently be the student's own, reading a KL of exactly 0 forever.

    The second half is the one this test can measure: the reference and the parent must be DISTINCT
    objects, or the first Polyak update would drag the displacement meter's own baseline along."""
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="ema", tau=0.5)
    assert len(loader.calls) == 2, "the reference must be its own load of the parent"
    assert cb._ref_policy is not parent.policy
    frozen = {k: v.clone() for k, v in parent.policy.state_dict().items()}
    with th.no_grad():
        for p in model.policy.action_net.parameters():
            p.add_(10.0)
    cb._on_rollout_end()
    assert _sd_equal(parent.policy.state_dict(), frozen), \
        "moving the reference moved the FROZEN parent the displacement meter reads"
    assert not _sd_equal(cb._ref_policy.state_dict(), frozen)


def test_a_reference_that_does_not_match_the_student_is_REFUSED():
    """An average between two architectures is not a trust region. Caught at startup with the
    diagnosis, rather than as a `polyak_update_` that quietly skips the keys it cannot find."""
    import pytest
    import torch.nn as nn

    class _A(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(3, 2)

    class _B(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(3, 2)
            self.extra = nn.Linear(2, 2)

    class _C(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(4, 2)

    assert_reference_matches_student(_A(), _A())                     # identical: fine
    with pytest.raises(ValueError, match="only the student has"):
        assert_reference_matches_student(_A(), _B())
    with pytest.raises(ValueError, match="only the reference has"):
        assert_reference_matches_student(_B(), _A())
    with pytest.raises(ValueError, match="shape mismatch"):
        assert_reference_matches_student(_C(), _A())


def test_ema_moves_the_reference_toward_the_student_once_per_rollout():
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="ema", tau=0.5)
    with th.no_grad():
        for p in model.policy.action_net.parameters():
            p.add_(10.0)
    w0 = cb._ref_policy.action_net.weight.clone()
    cb._on_rollout_end()
    w1 = cb._ref_policy.action_net.weight.clone()
    target = model.policy.action_net.weight
    assert th.allclose(w1, 0.5 * w0 + 0.5 * target)
    cb._on_rollout_end()
    assert th.allclose(cb._ref_policy.action_net.weight, 0.5 * w1 + 0.5 * target)


def test_ema_at_tau_one_leaves_the_reference_at_the_parent_forever():
    """The degenerate end of the knob IS the default mode, which is what makes `ema` a superset of
    `parent` rather than a different feature."""
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="ema", tau=1.0)
    with th.no_grad():
        for p in model.policy.action_net.parameters():
            p.add_(10.0)
    for _ in range(5):
        cb._on_rollout_end()
    assert _sd_equal(cb._ref_policy.state_dict(), parent.policy.state_dict())


def test_ema_at_tau_zero_makes_the_reference_the_current_student():
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="ema", tau=0.0)
    with th.no_grad():
        for p in model.policy.action_net.parameters():
            p.add_(10.0)
    cb._on_rollout_end()
    assert _sd_equal(cb._ref_policy.state_dict(), model.policy.state_dict())


def test_periodic_refreshes_on_the_declared_rollout_and_NOT_before():
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="periodic", refresh_every=3)
    with th.no_grad():
        for p in model.policy.action_net.parameters():
            p.add_(10.0)
    for i in (1, 2):
        cb._on_rollout_end()
        assert _sd_equal(cb._ref_policy.state_dict(), parent.policy.state_dict()), \
            f"the reference moved at rollout {i} of 3"
        assert cb._rollouts == i
    cb._on_rollout_end()                                   # the third: refresh
    assert _sd_equal(cb._ref_policy.state_dict(), model.policy.state_dict())
    assert cb._rollouts == 0 and cb._refreshes == 1


def test_the_age_meter_says_what_the_anchor_is_anchored_to():
    """`parent` = rollouts since fold start (ever-rising, which is the point); `periodic` = rollouts
    since the last refresh (resets); `ema` = the NOMINAL WINDOW, because a geometric average has no
    age and a rollout counter there would be a number about nothing."""
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="parent")
    for _ in range(3):
        cb._on_rollout_end()
    assert model.distill_anchor_ref_age == 3.0
    cb = _attach(model, loader, ref="periodic", refresh_every=2)
    cb._on_rollout_end()
    assert model.distill_anchor_ref_age == 1.0
    cb._on_rollout_end()
    assert model.distill_anchor_ref_age == 0.0             # refreshed
    cb = _attach(model, loader, ref="ema", tau=0.99)
    cb._on_rollout_end()
    assert abs(model.distill_anchor_ref_age - 100.0) < 1e-6


# --------------------------------------------------------------------------------------
# Persistence: the reference is RUN STATE and must survive a launcher restart
# --------------------------------------------------------------------------------------

def _fake_ckpt(tmp_path, name="checkpoint_900_steps.zip"):
    d = tmp_path / "checkpoints"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_bytes(b"")            # the zip's CONTENT is irrelevant; only its path names the sibling
    return str(p)


def test_the_moving_reference_survives_a_save_then_restart(tmp_path):
    """🚨 The mirror of the parent's restart rule. The parent is re-read from a PATH; a moving
    reference has no path, so a restart that re-initialised it would reset the trust region to fold
    start every few hours while still reading as ON."""
    run_dir = str(tmp_path)
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="ema", tau=0.5, run_dir=run_dir)
    with th.no_grad():
        for p in model.policy.action_net.parameters():
            p.add_(10.0)
    cb._on_rollout_end()
    moved = {k: v.clone() for k, v in cb._ref_policy.state_dict().items()}
    ckpt = _fake_ckpt(tmp_path)
    written = cb.save_reference(ckpt)
    assert written == anchor_ref_path(ckpt) and os.path.exists(written)
    assert not os.path.exists(written + ".tmp")            # atomic: the temp never survives

    cb2 = _attach(model, loader, ref="ema", tau=0.5, run_dir=run_dir,
                  resume_model=ckpt, expect_restore=True)
    assert "RESTORED" in cb2._restore_note
    assert _sd_equal(cb2._ref_policy.state_dict(), moved)
    assert not _sd_equal(cb2._ref_policy.state_dict(), parent.policy.state_dict()), \
        "the restart silently re-anchored to the fold parent"


def test_the_periodic_cadence_survives_the_restart_too(tmp_path):
    """Not just the weights: a restored snapshot with a reset counter would refresh on a schedule
    the run never chose, which is a different cadence wearing the flag's name."""
    run_dir = str(tmp_path)
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="periodic", refresh_every=5, run_dir=run_dir)
    for _ in range(3):
        cb._on_rollout_end()
    assert cb._rollouts == 3
    ckpt = _fake_ckpt(tmp_path)
    cb.save_reference(ckpt)
    cb2 = _attach(model, loader, ref="periodic", refresh_every=5, run_dir=run_dir,
                  resume_model=ckpt, expect_restore=True)
    assert cb2._rollouts == 3
    cb2._on_rollout_end()
    assert cb2._rollouts == 4 and cb2._refreshes == 0      # ...and the 5th, not the 8th, refreshes
    cb2._on_rollout_end()
    assert cb2._refreshes == 1


def test_a_missing_sibling_on_a_RESTART_falls_back_to_the_parent_and_SAYS_SO(tmp_path):
    """Never silently: the note is what an operator reads to learn that a restart reset the trust
    region, and a reset that announced nothing is indistinguishable from one that never happened."""
    run_dir = str(tmp_path)
    model, parent, loader = _build_moving_pair()
    ckpt = _fake_ckpt(tmp_path)                            # no sibling beside it
    cb = _attach(model, loader, ref="ema", tau=0.5, run_dir=run_dir,
                 resume_model=ckpt, expect_restore=True)
    assert "NO reference sibling" in cb._restore_note and "RESET to fold start" in cb._restore_note
    assert _sd_equal(cb._ref_policy.state_dict(), parent.policy.state_dict())


def test_a_missing_sibling_on_a_FORKS_FIRST_LAUNCH_is_not_reported_as_a_reset(tmp_path):
    """The same absence means two different things, and only one of them is a problem."""
    model, parent, loader = _build_moving_pair()
    ckpt = _fake_ckpt(tmp_path)
    cb = _attach(model, loader, ref="ema", run_dir=str(tmp_path),
                 resume_model=ckpt, expect_restore=False)
    assert "fork's first launch" in cb._restore_note and "RESET" not in cb._restore_note


def test_a_sibling_from_ANOTHER_RUN_is_refused(tmp_path):
    """THE FORK GUARD. A fork off a fold's `final_model.zip` would otherwise inherit that fold's
    average as its own starting reference — a different run's trajectory wearing this run's name."""
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="ema", tau=0.5, run_dir=str(run_a))
    with th.no_grad():
        for p in model.policy.action_net.parameters():
            p.add_(10.0)
    cb._on_rollout_end()
    ckpt = _fake_ckpt(run_a)
    cb.save_reference(ckpt)
    cb2 = _attach(model, loader, ref="ema", tau=0.5, run_dir=str(run_b),
                  resume_model=ckpt, expect_restore=True)
    assert "REFUSED" in cb2._restore_note and "belongs to run" in cb2._restore_note
    assert _sd_equal(cb2._ref_policy.state_dict(), parent.policy.state_dict())


def test_a_sibling_saved_against_a_DIFFERENT_PARENT_is_refused(tmp_path):
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="ema", tau=0.5, run_dir=str(tmp_path))
    ckpt = _fake_ckpt(tmp_path)
    cb.save_reference(ckpt)
    _, _, loader2 = _build_moving_pair()
    cb2 = _attach(model, loader2, ref="ema", tau=0.5, run_dir=str(tmp_path),
                  resume_model=ckpt, expect_restore=True,
                  parent_path="models/OTHER/final_model.zip")
    assert "REFUSED" in cb2._restore_note and "anchors to" in cb2._restore_note


def test_a_sibling_saved_under_a_DIFFERENT_MODE_is_refused(tmp_path):
    """A `periodic` snapshot is not an `ema` average; loading one as the other would report the arm
    the run is not."""
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="periodic", refresh_every=4, run_dir=str(tmp_path))
    ckpt = _fake_ckpt(tmp_path)
    cb.save_reference(ckpt)
    cb2 = _attach(model, loader, ref="ema", tau=0.5, run_dir=str(tmp_path),
                  resume_model=ckpt, expect_restore=True)
    assert "REFUSED" in cb2._restore_note and "saved under ref" in cb2._restore_note


def test_an_unknown_schema_and_a_corrupt_blob_are_both_refused_not_half_read(tmp_path):
    model, parent, loader = _build_moving_pair()
    cb = _attach(model, loader, ref="ema", tau=0.5, run_dir=str(tmp_path))
    ckpt = _fake_ckpt(tmp_path)
    cb.save_reference(ckpt)
    blob = th.load(anchor_ref_path(ckpt), map_location="cpu", weights_only=False)
    blob["schema"] = ANCHOR_REF_SCHEMA + 1
    th.save(blob, anchor_ref_path(ckpt))
    cb2 = _attach(model, loader, ref="ema", tau=0.5, run_dir=str(tmp_path),
                  resume_model=ckpt, expect_restore=True)
    assert "REFUSED" in cb2._restore_note and "schema" in cb2._restore_note

    with open(anchor_ref_path(ckpt), "wb") as f:
        f.write(b"not a torch file")
    cb3 = _attach(model, loader, ref="ema", tau=0.5, run_dir=str(tmp_path),
                  resume_model=ckpt, expect_restore=True)
    assert "unreadable" in cb3._restore_note
    assert _sd_equal(cb3._ref_policy.state_dict(), parent.policy.state_dict())


def test_save_anchor_ref_beside_is_TOTAL(tmp_path):
    """It is called from the checkpoint sites, one of which is a SIGNAL HANDLER. A diagnostic's
    persistence must never be what loses a checkpoint."""
    model, parent, loader = _build_moving_pair()
    _attach(model, loader, ref="ema", tau=0.5, run_dir=str(tmp_path))
    assert save_anchor_ref_beside(model, "/no/such/dir/ckpt.zip") is None      # unwritable → None
    assert save_anchor_ref_beside(object(), "/tmp/x.zip") is None              # no anchor at all
    ok = save_anchor_ref_beside(model, _fake_ckpt(tmp_path))
    assert ok and os.path.exists(ok)


def test_the_reference_and_its_writer_are_never_pickled_into_a_checkpoint():
    """`_distill_anchor_ref_writer` is the CALLBACK, which back-references the model and SB3's
    `Logger` (a `_contextvars.Context`): pickling it would break EVERY save in the run."""
    model, _venv = _build_tiny_ppo()
    excluded = model._excluded_save_params()
    assert "_distill_anchor_ref" in excluded
    assert "_distill_anchor_ref_writer" in excluded


def test_every_resumable_checkpoint_site_writes_the_sibling():
    """The reference is only useful if it is beside the checkpoint a restart actually resumes from —
    and the launcher's periodic restart resumes from the SIGTERM save, not the periodic one."""
    import inspect
    from main.train import lifecycle, model_build, run_io
    assert "save_anchor_ref_beside(self.model, ckpt_path)" in inspect.getsource(
        run_io._TrackingCheckpointCallback._on_step)
    src = inspect.getsource(lifecycle._setup_signal_handlers)
    assert src.count("save_anchor_ref_beside(model,") == 2      # the SIGTERM abort + SIGUSR1 forced
    # BOTH final-save branches: the FRESH one and the RESUME one — and a fold IS a resume, so the
    # branch a fold actually takes is the second.
    assert inspect.getsource(model_build).count("save_anchor_ref_beside(model, final_path") == 2


# --------------------------------------------------------------------------------------
# The meters, and the loss, through a real train()
# --------------------------------------------------------------------------------------

def _moving_arm(*, ref, tau=0.99, refresh_every=8, coef=0.5, rollout_ends=1):
    """One `train()` call with a MOVING reference attached, returning (state_dict, metrics)."""
    th.manual_seed(0)
    np.random.seed(0)
    model, parent, loader = _build_moving_pair()
    model.learn(total_timesteps=8 * 4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    cb = _attach(model, loader, ref=ref, tau=tau, refresh_every=refresh_every, coef=coef)
    for _ in range(rollout_ends):
        cb._on_rollout_end()
    model._logger = _Rec()
    sd = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    return sd, model.logger.vals


def test_ema_at_tau_one_reproduces_the_parent_arm_exactly():
    """The strongest form of "the default is unchanged": the degenerate ema setting is not merely
    close to `parent`, it is the same update and the same KL."""
    base, log_base = _moving_arm(ref="parent")
    same, log_same = _moving_arm(ref="ema", tau=1.0)
    for k in base:
        assert th.equal(base[k], same[k]), f"tau=1.0 diverged from `parent` at {k}"
    assert abs(log_base["distill/anchor_kl"] - log_same["distill/anchor_kl"]) < 1e-9


def test_ema_at_tau_zero_drives_the_anchor_loss_to_nothing():
    """The other degenerate end: the reference IS the student, so there is no displacement to
    penalise and the trust region stops constraining. That it still emits `collateral_kl_vs_parent`
    is the point of separating the two meters."""
    _, log_parent = _moving_arm(ref="parent")
    _, log_zero = _moving_arm(ref="ema", tau=0.0)
    assert log_zero["distill/anchor_kl"] < 0.05 * log_parent["distill/anchor_kl"]
    assert log_zero["distill/collateral_kl_vs_parent"] > 10.0 * log_zero["distill/anchor_kl"], \
        "the displacement meter followed the moving reference instead of the frozen parent"


def test_the_displacement_meter_reads_the_FROZEN_parent_in_every_mode():
    """`collateral_kl` reads the ANCHOR's reference, so under a moving one it is a RATE. The
    displacement is the number the untaught-team meter correlates with, so it gets its own key —
    and in `parent` mode the two are the same number computed once (no second forward)."""
    _, log_parent = _moving_arm(ref="parent")
    assert log_parent["distill/collateral_kl_vs_parent"] == log_parent["distill/collateral_kl"]
    _, log_ema = _moving_arm(ref="ema", tau=0.0)
    assert log_ema["distill/collateral_kl_vs_parent"] != log_ema["distill/collateral_kl"]


def test_the_age_meter_reaches_tensorboard():
    _, log = _moving_arm(ref="periodic", refresh_every=8, rollout_ends=3)
    assert log["distill/anchor_ref_age_rollouts"] == 3.0
    _, log_ema = _moving_arm(ref="ema", tau=0.99)
    assert abs(log_ema["distill/anchor_ref_age_rollouts"] - 100.0) < 1e-6


def test_offslice_kl_is_the_collateral_meter_computed_against_any_reference():
    parent, student, amask, tid = _split_batch()
    _, m = anchor_loss_and_metrics(parent, student, amask, tid, mode="off_slice")
    assert abs(offslice_kl(parent, student, amask, tid) - m["collateral_kl"]) < 1e-6
    assert offslice_kl(parent, student, amask, th.ones(4, 1)) is None      # no off-slice row
    assert offslice_kl(None, student, amask, tid) is None


def test_frozen_logits_filters_the_obs_to_the_references_own_space():
    """A parent from an older obs generation has never seen the training-only `distill_mask`, and a
    reference is not allowed to be the thing that crashes a fold on a key it does not know."""
    model, parent = _build_anchor_ppo()
    obs = {"observation": th.zeros(3, 1), "action_mask": th.ones(3, 2),
           "distill_mask": th.zeros(3, 1)}
    assert "distill_mask" not in parent.observation_space.spaces
    assert tuple(frozen_logits(parent, obs).shape) == (3, 2)               # no KeyError


# --------------------------------------------------------------------------------------
# The CLI surface
# --------------------------------------------------------------------------------------

def test_checkargs_accepts_the_reference_flags():
    from main.checkargs import check
    got = check(["--distill-teacher", "models/t:data/teams/sample/a.txt",
                 "--distill-coef", "0.1", "--distill-anchor-coef", "0.02",
                 "--distill-anchor-ref", "ema", "--distill-anchor-ema-tau", "0.995",
                 "--distill-anchor-refresh-every", "4"])
    assert got["unknown"] == []
    for f in ("--distill-anchor-ref", "--distill-anchor-ema-tau",
              "--distill-anchor-refresh-every"):
        assert f in got["accepted"]


def _resolved(argv):
    from main.train.config import resolve_config
    from main.train_rl_agent import build_parser
    p = build_parser()
    args = p.parse_args(argv)
    resolve_config(args, p)
    return args


def test_the_reference_defaults_to_the_byte_identical_parent():
    args = _resolved(["--steps", "10"])
    assert args.distill_anchor_ref == "parent"
    assert args.distill_anchor_ema_tau == 0.99
    assert args.distill_anchor_refresh_every == 8


def test_config_refuses_a_reference_knob_with_no_anchor():
    import pytest
    for argv in (["--distill-anchor-ref", "ema"],
                 ["--distill-anchor-ema-tau", "0.9"],
                 ["--distill-anchor-refresh-every", "4"]):
        with pytest.raises(SystemExit):
            _resolved(argv + ["--steps", "10"])


def test_config_refuses_a_tau_outside_the_unit_interval():
    import pytest
    base = ["--distill-teacher", "models/t:data/teams/sample/a.txt", "--distill-coef", "0.1",
            "--distill-anchor-coef", "0.02", "--steps", "10"]
    for bad in ("-0.1", "1.5"):
        with pytest.raises(SystemExit):
            _resolved(base + ["--distill-anchor-ema-tau", bad])
    for good in ("0.0", "1.0", "0.99"):
        _resolved(base + ["--distill-anchor-ema-tau", good])          # the endpoints ARE legal


def test_config_refuses_a_negative_refresh_cadence():
    import pytest
    base = ["--distill-teacher", "models/t:data/teams/sample/a.txt", "--distill-coef", "0.1",
            "--distill-anchor-coef", "0.02", "--steps", "10"]
    with pytest.raises(SystemExit):
        _resolved(base + ["--distill-anchor-refresh-every", "-1"])
    _resolved(base + ["--distill-anchor-refresh-every", "0"])          # 0 = never = parent


def test_the_reference_hparams_default_to_the_fixed_parent_on_the_class():
    assert InstrumentedMaskablePPO.distill_anchor_ref == "parent"
    assert InstrumentedMaskablePPO.distill_anchor_ref_age == 0.0
