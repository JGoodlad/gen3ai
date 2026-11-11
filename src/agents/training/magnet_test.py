"""The trailing magnet (`agents.training.magnet`, CE-3d / T56, `gen3_magnet_kl_v1`) — the parts that need no learner.

* **the KL is over the LEGAL actions**: it equals the hand-computed ``Σ_legal π (log π − log π_mag)``, and an illegal
  entry contributes exactly nothing to the value or to the gradient whatever the column or the logits hold there;
* **the direction** is KL(π ‖ π_mag) (the learner's expectation), and it is ``−H(π) − Σ π log π_mag``;
* **the cadence**: the magnet refreshes at the first update boundary at or after `--magnet-refresh-steps` env steps,
  not before; a refresh is a frozen copy (later weight moves do not reach it) and reloads T2's magnet slot;
* **the sidecar**: the magnet round-trips bit for bit, `run_io.commit_checkpoint` writes it between the zip and
  ``latest.txt``, and `attach` reads it back on a resume, refuses a same-run resume that lost it, and seeds a fork;
* **the flags**: off by default, inherited on a flagless resume, recorded on `ModelVersion` (config v170), refused
  with the fork arm and with `--overlap-collect`.

Every test here fails if the behaviour it names is reverted.
"""
from __future__ import annotations

import inspect
import os
from types import SimpleNamespace
from typing import Any, List, Tuple

import numpy as np
import pytest
import torch as th

from agents.model import masked_categorical as MC
from agents.training import magnet as M

N_ACTIONS = 9


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _rows(seed: int = 0, n: int = 6) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """``(raw logits [requires grad], legal mask, the magnet's masked log-probs with 0.0 on illegal entries)``: every
    row has at least two legal and one illegal action."""
    g = th.Generator().manual_seed(seed)
    logits = th.randn(n, N_ACTIONS, generator=g, dtype=th.float32).requires_grad_(True)
    mask = th.rand(n, N_ACTIONS, generator=g) < 0.6
    mask[:, 0] = True
    mask[:, 1] = True
    mask[:, 2] = False
    mag_logits = th.randn(n, N_ACTIONS, generator=g, dtype=th.float32)
    mag = MC.masked_logits(mag_logits, mask, N_ACTIONS)
    return logits, mask, th.where(mask, mag, th.zeros(()))


def _kl(logits: th.Tensor, mask: th.Tensor, column: th.Tensor) -> th.Tensor:
    logp = MC.masked_logits(logits, mask, N_ACTIONS)
    return M.reverse_kl_rows(logp, MC.mask_bool(mask, logp), column)


class _Policy(th.nn.Module):
    """A stand-in policy: what the magnet needs of one is its state dict."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        th.manual_seed(seed)
        self.a = th.nn.Linear(4, 3)
        self.b = th.nn.Linear(3, 2)
        self.register_buffer("steps", th.zeros(1))


def _move(policy: th.nn.Module, by: float = 0.5) -> None:
    with th.no_grad():
        for p in policy.parameters():
            p.add_(by)


class _Svc:
    def __init__(self) -> None:
        self.loads: List[Tuple[int, str, th.Tensor]] = []

    def load(self, slot: int, policy: Any, model_id: str) -> None:
        self.loads.append((int(slot), str(model_id), policy.a.weight.detach().clone()))


def _model(policy: th.nn.Module, *, coef: float = 0.05, refresh: int = 1000, steps: int = 0) -> Any:
    return SimpleNamespace(policy=policy, magnet_kl_coef=coef, magnet_refresh_steps=refresh, num_timesteps=steps,
                           ent_coef=0.05)


# ── the term: legal actions only, the reverse direction ─────────────────────────────────────────────────────────
def test_the_KL_is_the_hand_computed_sum_over_the_LEGAL_actions():
    logits, mask, column = _rows()
    got = _kl(logits, mask, column).detach().numpy().astype(np.float64)
    lg, mk, col = logits.detach().numpy().astype(np.float64), mask.numpy(), column.numpy().astype(np.float64)
    for i in range(lg.shape[0]):
        legal = np.flatnonzero(mk[i])
        z = lg[i, legal] - lg[i, legal].max()
        logp = z - np.log(np.exp(z).sum())
        want = float((np.exp(logp) * (logp - col[i, legal])).sum())
        assert abs(got[i] - want) < 1e-5, (i, got[i], want)
        assert got[i] > 1e-3                      # two different distributions: a positive KL, not a vacuous zero


def test_an_ILLEGAL_entry_contributes_nothing_to_the_value_or_the_gradient_whatever_it_holds():
    """T2 serves −inf on an illegal action, and a column or logit that reaches the sum there is a NaN (``0 · inf``) in
    the value or in the backward. Revert either ``where`` of `reverse_kl_rows` and this fails."""
    logits, mask, column = _rows(1)
    base = _kl(logits, mask, column)
    (g_base,) = th.autograd.grad(base.sum(), logits)
    assert th.isfinite(base).all() and th.isfinite(g_base).all()
    # no gradient reaches an illegal logit: what is left is the first normalisation's rounding residue (its
    # log-sum-exp cancels exactly on paper), five orders below a legal logit's gradient
    assert float(g_base[~mask].abs().max()) < 1e-6
    assert g_base[mask].abs().max() > 1e-3
    for junk in (float("-inf"), float("nan"), 123.0):
        col = th.where(mask, column, th.full((), junk))
        kl = _kl(logits, mask, col)
        (g,) = th.autograd.grad(kl.sum(), logits)
        assert th.equal(kl, base), junk                                    # bit for bit
        assert th.equal(g, g_base), junk
    # the learner's own illegal logits: any value there is masked before the softmax
    moved = logits.detach().clone()
    moved[~mask] += 7.0
    np.testing.assert_allclose(_kl(moved, mask, column).numpy(), base.detach().numpy(), atol=1e-5)


def test_the_direction_is_the_learners_expectation_and_splits_into_entropy_plus_cross_entropy():
    logits, mask, column = _rows(2)
    logp = MC.masked_logits(logits, mask, N_ACTIONS)
    mb = MC.mask_bool(mask, logp)
    kl = M.reverse_kl_rows(logp, mb, column)
    p = MC.probs(logp)
    cross = -(th.where(mb, p * column, th.zeros(()))).sum(-1)              # −Σ π log π_mag
    np.testing.assert_allclose(kl.detach().numpy(), (cross - MC.entropy(logp, mb)).detach().numpy(), atol=1e-5)
    # NOT the forward KL(π_mag ‖ π): the two differ on these rows
    fwd = th.where(mb, column.exp() * (column - logp), th.zeros(())).sum(-1)
    assert (kl - fwd).abs().max() > 1e-3


def test_the_KL_to_the_policy_itself_is_exactly_zero_and_to_the_masked_uniform_is_log_n_minus_entropy():
    logits, mask, _ = _rows(3)
    logp = MC.masked_logits(logits, mask, N_ACTIONS)
    mb = MC.mask_bool(mask, logp)
    own = th.where(mb, logp.detach(), th.zeros(()))
    assert th.equal(M.reverse_kl_rows(logp, mb, own), th.zeros(logp.shape[0]))
    uni = M.masked_uniform_logp(mask)
    assert th.equal(uni[~mask], th.zeros_like(uni[~mask]))
    np.testing.assert_allclose(th.where(mask, uni.exp(), th.zeros(())).sum(-1).numpy(), 1.0, atol=1e-6)
    want = th.log(mask.sum(-1).float()) - MC.entropy(logp, mb)
    np.testing.assert_allclose(M.reverse_kl_rows(logp, mb, uni).detach().numpy(), want.detach().numpy(), atol=1e-5)


# ── the cadence ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_the_magnet_is_a_frozen_copy_that_refreshes_at_the_first_boundary_at_or_after_the_cadence():
    policy = _Policy()
    model = _model(policy, refresh=1000)
    mag = M.attach(model, model_path=None, same_run=False, saved_coef=0.0, say=lambda _m: None)
    assert mag is model._magnet and mag.step == 0 and mag.equals(policy)
    theta0 = policy.a.weight.detach().clone()
    svc = _Svc()
    for steps in (384, 768, 999):                                          # update boundaries BEFORE the cadence
        _move(policy)
        model.num_timesteps = steps
        assert M.refresh_if_due(svc, 7, model) is False
    assert not svc.loads and mag.step == 0 and mag.refreshes == 0
    assert th.equal(mag.state["a.weight"], theta0) and not mag.equals(policy)      # frozen: the moves did not reach it
    _move(policy)
    model.num_timesteps = 1152                                             # the first boundary at or after 1000
    assert M.refresh_if_due(svc, 7, model) is True
    assert mag.step == 1152 and mag.refreshes == 1 and mag.equals(policy)
    assert [(s, i) for s, i, _w in svc.loads] == [(7, "magnet@1152")]
    assert th.equal(svc.loads[0][2], policy.a.weight)                      # T2's slot got the weights of THIS boundary
    _move(policy)
    model.num_timesteps = 2151                                             # 999 steps of age: not yet
    assert M.refresh_if_due(svc, 7, model) is False and len(svc.loads) == 1
    model.num_timesteps = 2152                                             # exactly the cadence: due
    assert M.refresh_if_due(svc, 7, model) is True and mag.step == 2152 and len(svc.loads) == 2


def test_off_builds_no_magnet_and_a_collector_slot_without_one_is_a_typed_fatal():
    from main.exit_codes import TrainExitCode, exit_code_for
    model = _model(_Policy(), coef=0.0)
    assert M.attach(model, model_path=None, same_run=False, saved_coef=0.0, say=lambda _m: None) is None
    assert model._magnet is None
    for fn in (M.refresh_if_due, M.load_slot, M.fill):
        with pytest.raises(M.MagnetError) as e:
            fn(_Svc(), 3, model)
        assert exit_code_for(e.value) == int(TrainExitCode.FATAL_CONFIG)


def test_the_startup_load_hands_the_slot_a_RESUMED_magnet_and_leaves_the_learner_bit_exact():
    policy = _Policy()
    model = _model(policy)
    M.attach(model, model_path=None, same_run=False, saved_coef=0.0, say=lambda _m: None)
    svc = _Svc()
    assert M.load_slot(svc, 5, model) is False and not svc.loads           # the slot's template IS the magnet
    theta0 = policy.a.weight.detach().clone()
    _move(policy)
    live = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    assert M.load_slot(svc, 5, model) is True
    assert svc.loads[0][:2] == (5, "magnet@0") and th.equal(svc.loads[0][2], theta0)   # the MAGNET's weights were loaded
    assert all(th.equal(live[k], v) for k, v in policy.state_dict().items())           # and the learner restored


# ── the sidecar: save, resume, fork ─────────────────────────────────────────────────────────────────────────────
class _Saveable(SimpleNamespace):
    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(b"zip")


def _committed(tmp_path, *, steps: int = 5000) -> Tuple[Any, str]:
    """A magnet run's committed checkpoint whose magnet (set at step 1152) is NOT the checkpoint's weights."""
    from main.train.run_io import commit_checkpoint
    policy = _Policy()
    model = _Saveable(**vars(_model(policy, refresh=1000)))
    M.attach(model, model_path=None, same_run=False, saved_coef=0.0, say=lambda _m: None)
    _move(policy)
    model.num_timesteps = 1152
    assert M.refresh_if_due(_Svc(), 1, model)
    _move(policy)
    model.num_timesteps = steps
    order: List[str] = []
    zip_path = str(tmp_path / "checkpoints" / f"checkpoint_{steps}_steps.zip")
    final = commit_checkpoint(model, str(tmp_path), zip_path,
                              sidecars=lambda z: order.append(f"sidecars:{os.path.exists(M.sidecar_path(z))}"),
                              pointer=lambda d, rel: order.append(f"pointer:{os.path.exists(M.sidecar_path(zip_path))}"))
    # the magnet is on disk BEFORE the caller's sidecars and before latest.txt moves
    assert order == ["sidecars:True", "pointer:True"] and final == zip_path
    return model, final


def test_commit_checkpoint_writes_the_magnet_beside_the_zip_and_a_resume_reads_it_back(tmp_path):
    saved, zip_path = _committed(tmp_path)
    side = M.sidecar_path(zip_path)
    assert side.endswith("checkpoint_5000_steps.magnet.pt") and os.path.isfile(side)
    assert not [n for n in os.listdir(os.path.dirname(side)) if ".tmp" in n]            # atomic: no temp left
    # the resumed learner: the checkpoint's weights (which are NOT the magnet's) at the checkpoint's step
    resumed = _model(_Policy(), refresh=1000, steps=5000)
    resumed.policy.load_state_dict(saved.policy.state_dict())
    said: List[str] = []
    mag = M.attach(resumed, model_path=zip_path, same_run=True, saved_coef=0.05, say=said.append)
    assert mag.step == 1152 and mag.age(5000) == 3848 and "sidecar" in mag.origin
    assert all(th.equal(mag.state[k], saved._magnet.state[k]) for k in saved._magnet.state)   # bit for bit
    assert not mag.equals(resumed.policy)                 # NOT re-seeded at the loaded weights
    assert said and "MAGNET" in said[0] and "1,152" in said[0]
    assert mag.due(5000)                                  # its age survived: the next boundary refreshes it


def test_no_magnet_no_sidecar(tmp_path):
    from main.train.run_io import commit_checkpoint
    model = _Saveable(policy=_Policy())                    # a run without the lever: no `_magnet` at all
    final = commit_checkpoint(model, str(tmp_path), str(tmp_path / "checkpoints" / "checkpoint_1_steps.zip"),
                              pointer=lambda d, rel: None)
    assert os.listdir(os.path.dirname(final)) == ["checkpoint_1_steps.zip"]


def test_a_same_run_resume_that_LOST_its_sidecar_is_refused_and_a_fork_seeds_a_new_magnet(tmp_path):
    from main.exit_codes import TrainExitCode, exit_code_for
    saved, zip_path = _committed(tmp_path)
    os.remove(M.sidecar_path(zip_path))
    resumed = _model(_Policy(), steps=5000)
    resumed.policy.load_state_dict(saved.policy.state_dict())
    with pytest.raises(M.MagnetError, match="has no sidecar") as e:
        M.attach(resumed, model_path=zip_path, same_run=True, saved_coef=0.05, say=lambda _m: None)
    assert exit_code_for(e.value) == int(TrainExitCode.FATAL_CONFIG)       # a restart meets the same missing file
    # a FORK of the same file (another run's dir), and a run that turns the lever ON: a new magnet, the loaded weights
    for same_run, saved_coef in ((False, 0.05), (True, 0.0)):
        mag = M.attach(resumed, model_path=zip_path, same_run=same_run, saved_coef=saved_coef, say=lambda _m: None)
        assert mag.step == 5000 and mag.age(5000) == 0 and mag.equals(resumed.policy)


def test_a_sidecar_that_is_not_this_checkpoints_is_refused(tmp_path):
    saved, zip_path = _committed(tmp_path)
    early = _model(_Policy(), steps=1000)                 # the checkpoint's step is BEFORE the magnet's 1152
    with pytest.raises(M.MagnetError, match="after the checkpoint's own step"):
        M.attach(early, model_path=zip_path, same_run=True, saved_coef=0.05, say=lambda _m: None)

    class _Other(th.nn.Module):                           # another network's weights
        def __init__(self) -> None:
            super().__init__()
            self.a = th.nn.Linear(5, 3)
    with pytest.raises(M.MagnetError, match="not this policy's"):
        M.attach(_model(_Other(), steps=5000), model_path=zip_path, same_run=True, saved_coef=0.05,
                 say=lambda _m: None)
    with open(M.sidecar_path(zip_path), "wb") as fh:      # a torn / foreign file
        fh.write(b"not a magnet")
    with pytest.raises(M.MagnetError, match="cannot be read"):
        M.attach(_model(_Policy(), steps=5000), model_path=zip_path, same_run=True, saved_coef=0.05,
                 say=lambda _m: None)


def test_the_sidecar_name():
    assert M.sidecar_path("a/checkpoint_9_steps.zip") == "a/checkpoint_9_steps.magnet.pt"
    assert M.sidecar_path("a/final_model") == "a/final_model.magnet.pt"


# ── the collector's side ─────────────────────────────────────────────────────────────────────────────────────────
class _Ticket:
    def __init__(self, lp: np.ndarray) -> None:
        self._lp = lp

    def host(self) -> Tuple[np.ndarray, None, None]:
        return self._lp, None, None


class _ColumnSvc:
    """T2 as `fill_column` sees it: masked log-probs with −inf on illegal entries, at most ``cap`` rows a flush."""

    def __init__(self, cap: int) -> None:
        self.spec = SimpleNamespace(max_rows_per_flush=cap)
        self.calls: List[Tuple[int, int]] = []
        self.flushes = 0

    def submit(self, slot: int, obs: np.ndarray, mask: np.ndarray, _priority: Any) -> _Ticket:
        assert obs.shape[0] == mask.shape[0] <= self.spec.max_rows_per_flush
        self.calls.append((int(slot), int(obs.shape[0])))
        lp = np.where(mask > 0, obs[:, :1] * 0.0 - np.log(np.maximum(mask.sum(-1, keepdims=True), 1)), -np.inf)
        return _Ticket(lp.astype(np.float32))

    def flush(self) -> None:
        self.flushes += 1


def _buffer(n_steps: int = 5, n_envs: int = 3) -> Any:
    from agents.training.rust_rollout.store import KEY_OBSERVATION
    g = np.random.default_rng(0)
    mask = (g.random((n_steps, n_envs, N_ACTIONS)) < 0.5).astype(np.float32)
    mask[..., 0] = 1.0
    mask[..., 1] = 0.0
    return SimpleNamespace(buffer_size=n_steps, n_envs=n_envs, action_masks=mask,
                           observations={KEY_OBSERVATION: g.random((n_steps, n_envs, 4)).astype(np.float32)})


def test_fill_column_serves_every_row_through_the_magnet_slot_in_bounded_flushes_and_zeroes_illegal_entries():
    buf, svc = _buffer(), _ColumnSvc(cap=4)
    col = M.fill_column(svc, 9, buf)
    assert col.shape == (5, 3, N_ACTIONS) and col.dtype == np.float32
    assert svc.calls == [(9, 4), (9, 4), (9, 4), (9, 3)] and svc.flushes == 4             # 15 rows, 4 a flush
    legal = buf.action_masks > 0
    assert np.isfinite(col).all() and (col[~legal] == 0.0).all()
    np.testing.assert_allclose(col[legal], np.broadcast_to(-np.log(legal.sum(-1, keepdims=True)), legal.shape)[legal],
                               rtol=1e-6)


def test_a_non_finite_magnet_log_prob_on_a_LEGAL_action_is_the_typed_non_finite_fatal():
    from main.exit_codes import TrainExitCode, exit_code_for

    class _Bad(_ColumnSvc):
        def submit(self, slot: int, obs: np.ndarray, mask: np.ndarray, _priority: Any) -> _Ticket:
            t = super().submit(slot, obs, mask, _priority)
            t._lp[0, 0] = np.nan                           # action 0 is legal on every row
            return t
    with pytest.raises(FloatingPointError) as e:
        M.fill_column(_Bad(cap=64), 9, _buffer())
    assert exit_code_for(e.value) == int(TrainExitCode.FATAL_NONFINITE)


def test_the_collector_declaration_refuses_the_fork_arm_and_the_overlap():
    from agents.training.rust_rollout.build import RustEnvDecl
    assert RustEnvDecl(n_envs=4).magnet is False                           # OFF by default: no slot declared
    assert RustEnvDecl(n_envs=4, device="cpu", backend="eager", magnet=True).magnet is True
    with pytest.raises(ValueError, match="fork arm"):
        RustEnvDecl(n_envs=4, device="cpu", backend="eager", magnet=True, fork=object())
    with pytest.raises(ValueError, match="--magnet-kl-coef"):
        RustEnvDecl(n_envs=4, device="cpu", backend="eager", magnet=True, overlap=True)


def test_the_magnet_slot_is_declared_LAST_in_the_trainees_group_at_the_full_bucket():
    """`build_collector`'s slot arithmetic, read at its source: the magnet is appended AFTER the caller's extra slots
    (eval consumes those positionally), popped off the ids it hands eval, and capped like the trainee's slot."""
    from agents.training.rust_rollout import build as B
    src = inspect.getsource(B.build_collector)
    assert 'list(extra_slots) + ([("magnet", trainee_policy)] if decl.magnet else [])' in src
    assert "extra_ids.pop() if decl.magnet else None" in src
    assert "decl.slot_bucket_caps(n_slots, full_bucket_slots)" in src
    assert "decl.lead_slots(trainee_slots)" in src                         # never a lead slot
    plan = SimpleNamespace(routes=lambda: [])
    tpl = _Policy()
    tpl._get_constructor_parameters = lambda: {}                           # `_arch_key` reads a policy's constructor
    extra = [("evaltrainee", tpl), ("evalsentinel", tpl)]
    groups, trainee, ids = B.slot_groups_with_extra(plan, {}, tpl, 1, extra + [("magnet", tpl)])
    assert [(n, k) for n, k, _t in groups] == [("trainee0", 4)] and trainee == [0] and ids == [1, 2, 3]
    decl = B.RustEnvDecl(n_envs=256, device="cpu", backend="eager", magnet=True)
    caps = decl.slot_bucket_caps(4, trainee + [ids[-1]])
    assert caps == (256, 64, 64, 256)                                      # trainee, eval, eval, MAGNET


# ── the flags ────────────────────────────────────────────────────────────────────────────────────────────────────
def _resolved(argv: List[str]) -> Any:
    from main.train.config import resolve_config
    from main.train.parser import build_parser
    p = build_parser()
    args = p.parse_args(argv)
    resolve_config(args, p)
    return args


def test_the_flags_are_off_by_default_and_tri_state_on_the_parser():
    from main.train.parser import build_parser
    p = build_parser()
    a = p.parse_args([])
    assert a.magnet_kl_coef is None and a.magnet_refresh_steps is None     # None = untyped: a resume can inherit
    a = p.parse_args(["--magnet-kl-coef", "0.05", "--magnet_refresh_steps", "500000"])
    assert a.magnet_kl_coef == 0.05 and a.magnet_refresh_steps == 500000
    r = _resolved([])
    assert r.magnet_kl_coef == 0.0 and r.magnet_refresh_steps == M.REFRESH_STEPS_DEFAULT == 2_000_000
    r = _resolved(["--magnet-kl-coef", "0.05"])
    assert r.magnet_kl_coef == 0.05 and r.magnet_refresh_steps == 2_000_000


def test_a_negative_coefficient_or_a_non_positive_cadence_is_refused():
    for argv in (["--magnet-kl-coef", "-0.1"], ["--magnet-kl-coef", "0.05", "--magnet-refresh-steps", "0"]):
        with pytest.raises(SystemExit):
            _resolved(argv)


def test_a_flagless_resume_INHERITS_both_and_a_typed_value_overrides():
    from agents.model.model_version import ModelVersion
    from main.train.config import inherit_saved_flag
    from main.train.parser import build_parser
    import main.train.config as cfg
    saved = ModelVersion.__new__(ModelVersion)
    saved.magnet_kl_coef, saved.magnet_refresh_steps = 0.05, 500_000
    args = build_parser().parse_args([])
    assert inherit_saved_flag(args, saved, "magnet_kl_coef", 0.0) is True and args.magnet_kl_coef == 0.05
    assert inherit_saved_flag(args, saved, "magnet_refresh_steps", 2_000_000) is True
    assert args.magnet_refresh_steps == 500_000
    args = build_parser().parse_args(["--magnet-kl-coef", "0"])            # a fork turning the lever OFF
    assert inherit_saved_flag(args, saved, "magnet_kl_coef", 0.0) is False and args.magnet_kl_coef == 0.0
    src = inspect.getsource(cfg)
    assert '_resolve("magnet_kl_coef", 0.0)' in src and '_resolve("magnet_refresh_steps", 2_000_000)' in src
    # the RECORDED coefficient is captured before the resolve overwrites it (the missing-sidecar refusal reads it)
    assert src.index("args._magnet_saved_coef =") < src.index('_resolve("magnet_kl_coef", 0.0)')


def test_both_are_RECORDED_ModelVersion_fields_and_a_pre_magnet_config_migrates_to_OFF():
    import dataclasses

    from agents.model.model_version import ModelVersion
    from agents.model.model_version.constants import MODEL_CONFIG_VERSION
    from agents.model.model_version.migrations import _migrate_config as migrate_config
    fields = {f.name: f.default for f in dataclasses.fields(ModelVersion)}
    assert fields["magnet_kl_coef"] == 0.0 and fields["magnet_refresh_steps"] == 2_000_000
    # THE CE-3d BACKPORT: no version at this side pin (constants.py), so the migration is unconditional and a config
    # of the pin's OWN version (what P* and every C0 fork carry) is the pre-magnet one.
    assert MODEL_CONFIG_VERSION == 161
    old = {"config_version": 161}
    new = migrate_config(dict(old))
    assert new["magnet_kl_coef"] == 0.0 and new["magnet_refresh_steps"] == 2_000_000
    assert new["config_version"] == 161
    kept = migrate_config({"config_version": 161, "magnet_kl_coef": 0.05, "magnet_refresh_steps": 7})
    assert kept["magnet_kl_coef"] == 0.05 and kept["magnet_refresh_steps"] == 7


def test_the_two_refused_combinations_are_launch_time_checks():
    from main.train.combination_checks import COMBINATION_CHECKS
    by = {c.name: c for c in COMBINATION_CHECKS}
    fork, overlap = by["magnet_kl_vs_fork_arm"], by["magnet_kl_vs_overlap_collect"]
    on = SimpleNamespace(magnet_kl_coef=0.05, fork_fraction=0.2, overlap_collect=True)
    off = SimpleNamespace(magnet_kl_coef=0.0, fork_fraction=0.2, overlap_collect=True)
    unset = SimpleNamespace(magnet_kl_coef=None, fork_fraction=None, overlap_collect=None)
    assert fork.predicate(on) and overlap.predicate(on)
    assert not fork.predicate(off) and not overlap.predicate(off)
    assert not fork.predicate(unset) and not overlap.predicate(unset)
    assert fork.exit_style == overlap.exit_style == "fatal_config"
    assert "--fork-fraction" in fork.message(on) and "--overlap-collect" in overlap.message(on)


def test_the_compile_cache_key_moves_only_when_the_magnet_is_on():
    from agents.model import compile_cache_shared as SH
    assert "magnet_kl_coef" in SH.COMPILE_ARG_KEYS and "magnet_kl_coef" in SH.COMPILE_ARG_KEYS_WHEN_SET
    off, on = _resolved([]), _resolved(["--magnet-kl-coef", "0.05"])
    assert "magnet_kl_coef" not in SH.run_fields(off)["compile_args"]      # an OFF run keeps the key it had
    assert SH.run_fields(on)["compile_args"]["magnet_kl_coef"] == 0.05
