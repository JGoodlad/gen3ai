"""The IN-RUN PARITY CANARY (K6, `gen3_compile_canary_v1`) — the compiled learner is still the same
function at update N.

THE GAP IT CLOSES. The startup parity gate (`compile_trainer.compile_trainer_extractor`) proves the
compiled graph equals eager at t = 0, on the committed real-observation fixture. Nothing proved it
afterwards: a graph that changes behaviour mid-run (a silent recompile into a miscompiling variant, a
cache artifact served under a key that omitted a setting — the K1b class — or a kernel that goes bad
on weights it never saw at t = 0) would train on, and the K9(b) behaviour gate cannot see it when the
rollout and the learner share the wrong function (the 2026-09-28 miscompile). The owner (2026-09-28):
"I don't review the code or the logs, I trust agents — stopping GIGO is so powerful."

WHAT RUNS. Every `CANARY_EVERY` updates (after `train()` returns, between updates), on the SAME
committed real-obs fixture as the startup gate, through the DECLARED signatures only (the canary must
never be an undeclared compile itself):

  * the DECISION readout (`compile_trainer._readout`: features, masked legal log-probs, V) at the
    ROLLOUT signature — eval / no-grad / batch `n_envs` — compiled vs eager;
  * every `GRAD_EVERY`-th canary, also the TRAIN graph (`compile_trainer._train_step`: the gate's
    probe loss, forward + backward) at the UPDATE signature — train / grad / batch `batch_size` —
    the gradient's cosine and per-parameter error, compiled vs eager.

The bars are the startup gate's (`decision_verdicts`, `train_verdict`; the TF32 rule against an EAGER
fp32 reference under `--matmul-precision high` — the gate's extra "same graph at fp32" arm is NOT
run, because a compiled call at 'highest' is a separate graph, i.e. an undeclared signature after the
lock). The weights are the live, trained ones, so vacuity is REPORTED (`compile/canary_vacuous`),
never a refusal. A disagreement is `CompileCanaryError` (a `CompileTrainerError`): the trainer exits
FATAL_CONFIG and the launcher does not restart (a restart would compile the same graph).

WHAT IT DOES NOT DO: change training. It runs under `torch.random.fork_rng`, restores the policy's
training mode, and leaves every `.grad` as it found it (None between updates); the train-graph check
reads gradients through `.backward()` on the gate's probe loss and clears them. Cost (measured on the
production surface by the K6 smoke, see `designs/training/learner_lifecycle.md`): one eager + one
compiled forward at `n_envs` per canary, plus one eager + one compiled forward+backward at
`batch_size` every `GRAD_EVERY` canaries.
"""
from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Optional

import torch

from agents.model import compile_trainer as ct

#: Updates between canaries (a production update is ~50 s, so ~20 min apart; the decision readout
#: costs two batch-`n_envs` forwards).
CANARY_EVERY = 25
#: Every this-many canaries the train graph's gradient is checked too (~100 updates apart; it costs
#: an eager + a compiled forward+backward at the production micro-batch).
GRAD_EVERY = 4

CANARY_TAG = "[CompileCanary] FATAL"


class CompileCanaryError(ct.CompileTrainerError):
    """The compiled learner no longer equals eager on the fixture at the startup gate's bars."""


def _fixture_obs(model: Any, batch: int) -> Any:
    """The committed real-obs rows at ``batch`` as the policy's full obs dict (every key of its
    observation space — the declared signature's key set), plus their legal masks (numpy bool)."""
    obs = ct._prewarm_obs(model, int(batch))
    fe = model.policy.features_extractor
    obs_dim = int(obs["observation"].shape[-1])
    _, mask = ct._parity_obs(obs_dim, int(batch), ct.resolve_device(fe))
    return obs, mask


class CompileCanary:
    """Runs the canary on its cadence; see the module docstring."""

    def __init__(self, model: Any, *, n_envs: int, batch_size: int,
                 emit: Optional[Callable[[str], None]] = None,
                 every: int = CANARY_EVERY, grad_every: int = GRAD_EVERY) -> None:
        self.model = model
        self.n_envs = int(n_envs)
        self.batch_size = int(batch_size)
        self.every = max(1, int(every))
        self.grad_every = max(1, int(grad_every))
        self._emit = emit
        self.updates = 0
        self.runs = 0
        self.last: Dict[str, float] = {}

    def after_update(self) -> Dict[str, float]:
        """Count one update; run the canary when due. Returns the TB scalars it produced (``{}`` on
        a skipped update — the cadence writes nothing, never a stale value)."""
        self.updates += 1
        if self.updates % self.every:
            return {}
        return self.run(grad=(self.runs % self.grad_every) == self.grad_every - 1)

    def run(self, *, grad: bool) -> Dict[str, float]:
        model = self.model
        policy = model.policy
        precision = torch.get_float32_matmul_precision()
        was_training = bool(policy.training)
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        out: Dict[str, float] = {}
        rules = []
        try:
            with torch.random.fork_rng(devices=devices):
                from agents.model import compile_regions as cr
                if cr.installed(model):
                    rules += self._regions(model, precision, grad, out)
                else:
                    rules += self._extractor(model, precision, grad, out)
        except ct.CompileTrainerError as exc:
            raise CompileCanaryError(
                f"{CANARY_TAG} after update {self.updates}: the COMPILED learner no longer equals "
                f"eager on the committed real-obs fixture at the startup gate's bars — {exc}\n"
                f"The startup gate proved the graph at t=0; this canary proves it at t=N. Training "
                f"on a wrong function is GIGO: fatal by design, the launcher does NOT restart it "
                f"(FATAL_CONFIG). See designs/training/learner_lifecycle.md (the canary).") from exc
        finally:
            policy.set_training_mode(was_training)
            for p in policy.parameters():
                p.grad = None
        self.runs += 1
        out["compile/canary_ok"] = 1.0
        out["compile/canary_grad_checked"] = 1.0 if grad else 0.0
        self.last = out
        line = (f"🐤 [CompileCanary] update {self.updates}: compiled == eager on the fixture "
                f"({precision}) — " + " | ".join(rules))
        print(line, flush=True)
        if self._emit is not None:
            with contextlib.suppress(Exception):
                self._emit(line[:500])
        return out

    # ------------------------------------------------------------------ the two compiled shapes
    def _regions(self, model: Any, precision: str, grad: bool, out: Dict[str, float]) -> list:
        """K8's declared regions: R0 (compiled rollout core vs eager) on the fixture at n_envs; on a
        gradient canary also R1 (compiled micro-step vs eager: loss + every policy gradient) on the
        same real labelled batch the startup gate uses."""
        from agents.model import compile_regions as cr
        from agents.model.policy import _ROLLOUT_REGIONS
        from agents.training.instrumented_ppo.micro_step import micro_step
        policy = model.policy
        rules = []
        if self.n_envs not in ct.EAGER_BATCHES:
            policy.set_training_mode(False)
            obs = ct._prewarm_obs(model, self.n_envs)
            _, mask = ct._parity_obs(int(obs["observation"].shape[-1]), self.n_envs,
                                     ct.resolve_device(policy.features_extractor))
            c = cr._r0_readout(model, _ROLLOUT_REGIONS[policy], obs, mask)
            e = cr._r0_readout(model, cr._rollout_core, obs, mask)
            r = None
            if precision != "highest":
                with ct._matmul_precision("highest"):
                    r = cr._r0_readout(model, cr._rollout_core, obs, mask)
            rules += ["R0 " + x for x in ct.decision_verdicts(eager=e, compiled=c, reference=r,
                                                              precision=precision,
                                                              allow_vacuous=True)]
            for key in ("legal_logprob", "value"):
                out[f"compile/canary_max_abs_{key}"] = float((c[key] - e[key]).abs().max())
        if grad:
            policy.set_training_mode(True)
            args = cr._r1_args(model, cr.r1_batch(model, self.batch_size))
            comp = cr._r1_arm(model, model._compiled_micro_step, args)
            eager = cr._r1_arm(model, micro_step, args)
            ref = None
            if precision != "highest":
                with ct._matmul_precision("highest"):
                    ref = cr._r1_arm(model, micro_step, args)
            names = [n for n, _ in ct.grad_parameters(model, policy.features_extractor)]
            rules.append("R1 " + cr._r1_verdict(eager, comp, ref, precision, names,
                                                cr.weights_regime(model)))
            out["compile/canary_grad_cosine"] = ct._cos(comp["grad"], eager["grad"])
        return rules

    def _extractor(self, model: Any, precision: str, grad: bool, out: Dict[str, float]) -> list:
        """The legacy extractor-only compile (torch 2.5.1): the gate's own readout and train step."""
        policy = model.policy
        fe = policy.features_extractor
        rules = []
        obs, mask = _fixture_obs(model, self.n_envs)
        policy.set_training_mode(False)
        c_read = ct._readout(model, fe, obs, mask)
        with ct.eager_extractor(fe):
            e_read = ct._readout(model, fe, obs, mask)
            ref_read = None
            if precision != "highest":
                with ct._matmul_precision("highest"):
                    ref_read = ct._readout(model, fe, obs, mask)
        rules += ct.decision_verdicts(eager=e_read, compiled=c_read, reference=ref_read,
                                      precision=precision, allow_vacuous=True)
        for key in ("features", "legal_logprob", "value"):
            if key in e_read and key in c_read:
                out[f"compile/canary_max_abs_{key}"] = float((c_read[key] - e_read[key]).abs().max())
        if grad:
            tobs, tmask = _fixture_obs(model, self.batch_size)
            policy.set_training_mode(True)
            names = [n for n, _ in ct.grad_parameters(model, fe)]
            c_train = ct._train_step(model, fe, tobs, tmask)
            with ct.eager_extractor(fe):
                e_train = ct._train_step(model, fe, tobs, tmask)
                ref_train = None
                if precision != "highest":
                    with ct._matmul_precision("highest"):
                        ref_train = ct._train_step(model, fe, tobs, tmask)
            rules.append(ct.train_verdict(eager=e_train, compiled=c_train, reference=ref_train,
                                          precision=precision, allow_vacuous=True, param_names=names,
                                          param_bar=ct._MAX_PARAM_GRAD_REL_TRAINED))
            out["compile/canary_grad_cosine"] = ct._cos(c_train["grad"], e_train["grad"])
        return rules
