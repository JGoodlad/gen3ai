"""The IN-RUN PARITY CANARY (K6, `gen3_compile_canary_v1`; persistence `gen3_compile_canary_v2`) — the compiled learner is still the same
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

  * the DECISION readout (masked legal log-probs, V) at the ROLLOUT signature — eval / no-grad /
    batch `n_envs` — compiled vs eager;
  * the TRAIN graph at the UPDATE signature — train / grad / batch `batch_size` — compiled vs
    eager: region R1's loss and every policy gradient on K8 (the per-parameter bar chosen by
    `compile_regions.weights_regime`), the gate's probe loss on the legacy extractor compile.

The bars are the startup gate's (`decision_verdicts`, `train_verdict`; the TF32 rule against an EAGER
fp32 reference under `--matmul-precision high` — the gate's extra "same graph at fp32" arm is NOT
run, because a compiled call at 'highest' is a separate graph, i.e. an undeclared signature after the
lock). The weights are the live, trained ones, so vacuity is REPORTED, never a refusal.

PERSISTENCE, NOT A SINGLE SHOT (owner, 2026-10-01: "implement the consecutive check, up it to 100").
A real miscompile is deterministic and disagrees every time; a healthy graph's rare exceedance is a
property of one batch on one weight state. So a disagreement is CONFIRMED IN THE SAME UPDATE:

  1. warn, dump the verdict (`<run_dir>/canary_disagreements.jsonl`), and re-run the whole check —
     compiled AND eager, so the confirmation is not the same arithmetic repeated — on the SAME rows
     and on an INDEPENDENT fixture slice (`compile_trainer.fixture_index` slice 1: the second half
     of the rows, tiled, the same declared shape);
  2. CONFIRMED (it disagrees again on both) — checkpoint (`final_model_canary_fatal.zip`), then
     `CompileCanaryError` (FATAL_CONFIG, not restarted);
  3. not confirmed — `compile/canary_unconfirmed_disagreements` counts it, training continues, and
     the NEXT scheduled canary decides: a disagreement at TWO CONSECUTIVE scheduled canaries is
     FATAL even when neither confirmed.

Every verdict (update, step, pass / unconfirmed / fatal) is appended to
`<run_dir>/canary_verdicts.jsonl`, and the FATAL names the SAFE ROLLBACK POINT: the newest checkpoint
written at or before the last PASSING canary's step (`rollback_point`; restarts included — the file
is the run's, not the process's).

WHAT IT DOES NOT DO: change training. It runs under `torch.random.fork_rng`, restores the policy's
training mode, and leaves every `.grad` as it found it (None between updates); the train-graph check
reads gradients through `.backward()` on the gate's probe loss and clears them. Cost (measured on the
production surface by the K6 smoke, see `designs/training/learner_lifecycle.md`): one eager + one
compiled forward at `n_envs` per canary, plus one eager + one compiled forward+backward at
`batch_size` every `GRAD_EVERY` canaries.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

import torch

from agents.model import compile_trainer as ct

#: Updates between canaries (owner, 2026-10-01: 100; a production update is ~36 s on K8, so ~1 h).
CANARY_EVERY = 100
#: The FIRST canary runs at this update, then every `CANARY_EVERY` (orchestrator, 2026-10-01): the
#: earliest point where training has moved the weights off the startup gate's state, so no run of
#: any length goes unchecked (sizing arm A — 82 updates — never reached update 100).
CANARY_FIRST = 10
#: Every this-many canaries the train graph's gradient is checked too: EVERY canary since the cadence
#: became 100. MEASURED (CUDA, arm C's weights, n_envs 48, B = 2048, 2026-10-01): a canary costs
#: 0.06-0.09 s without the gradient check and 0.65-0.71 s with it, against ~3,600 s of training
#: between canaries.
GRAD_EVERY = 1

CANARY_TAG = "[CompileCanary] FATAL"
#: The run-dir files the canary appends to (one JSON object per line).
VERDICTS_FILE = "canary_verdicts.jsonl"
DISAGREEMENTS_FILE = "canary_disagreements.jsonl"
FATAL_CHECKPOINT = "final_model_canary_fatal"


class CompileCanaryError(ct.CompileTrainerError):
    """The compiled learner no longer equals eager on the fixture at the startup gate's bars."""


def _fixture_obs(model: Any, batch: int, slice_: int = 0) -> Any:
    """The committed real-obs rows at ``batch`` as the policy's full obs dict (every key of its
    observation space — the declared signature's key set), plus their legal masks (numpy bool)."""
    obs = ct._prewarm_obs(model, int(batch), slice_)
    fe = model.policy.features_extractor
    obs_dim = int(obs["observation"].shape[-1])
    _, mask = ct._parity_obs(obs_dim, int(batch), ct.resolve_device(fe), slice_)
    return obs, mask


def _run_dir(model: Any) -> Optional[str]:
    """The run directory the trainer declared (`model.behaviour_dump_dir`, set by `model_build` for
    K9(b)'s dumps), else None (a test learner: nothing is written)."""
    d = getattr(model, "behaviour_dump_dir", None)
    return str(d) if d else None


def _append(path: Optional[str], name: str, record: Dict[str, Any]) -> None:
    if not path:
        return
    with contextlib.suppress(OSError):
        with open(os.path.join(path, name), "a") as f:
            f.write(json.dumps(record) + "\n")


_CKPT_STEP = re.compile(r"(\d+)_steps\.zip$")


def rollback_point(run_dir: Optional[str]) -> str:
    """The SAFE ROLLBACK POINT for a canary FATAL: the newest checkpoint (`checkpoints/*_<N>_steps.zip`)
    written at or before the last PASSING canary's step in `<run_dir>/canary_verdicts.jsonl`. A
    one-line human answer, never an exception."""
    if not run_dir:
        return "unknown (no run directory declared)"
    last_pass: Optional[int] = None
    with contextlib.suppress(OSError, ValueError):
        with open(os.path.join(run_dir, VERDICTS_FILE)) as f:
            for line in f:
                r = json.loads(line)
                if r.get("verdict") == "pass":
                    last_pass = int(r["num_timesteps"])
    if last_pass is None:
        return "none: no canary of this run has PASSED (the startup gate's weights are the only known-good state)"
    best: Optional[int] = None
    best_path = None
    ck = os.path.join(run_dir, "checkpoints")
    with contextlib.suppress(OSError):
        for name in os.listdir(ck):
            m = _CKPT_STEP.search(name)
            if m and int(m.group(1)) <= last_pass and (best is None or int(m.group(1)) > best):
                best, best_path = int(m.group(1)), os.path.join(ck, name)
    if best_path is None:
        return f"none: no checkpoint at or before the last passing canary (step {last_pass:,})"
    return f"{best_path} (step {best:,}; the last passing canary was at step {last_pass:,})"


class CompileCanary:
    """Runs the canary on its cadence; see the module docstring. ``saver(path)`` writes the FATAL's
    checkpoint (default `model.save`; a test seam)."""

    def __init__(self, model: Any, *, n_envs: int, batch_size: int,
                 emit: Optional[Callable[[str], None]] = None,
                 every: int = CANARY_EVERY, grad_every: int = GRAD_EVERY,
                 saver: Optional[Callable[[str], None]] = None,
                 first: Optional[int] = None) -> None:
        self.model = model
        self.n_envs = int(n_envs)
        self.batch_size = int(batch_size)
        self.every = max(1, int(every))
        # the first canary: `CANARY_FIRST` at the production cadence; a custom cadence (tests) starts
        # at its own period unless told otherwise
        self.first = int(first) if first is not None else (CANARY_FIRST if self.every == CANARY_EVERY
                                                            else self.every)
        self.grad_every = max(1, int(grad_every))
        self._emit = emit
        self._saver = saver
        self.updates = 0
        self.runs = 0
        self.unconfirmed = 0            # cumulative: disagreements that did not reproduce
        self.prev_disagreed = False     # the previous SCHEDULED canary disagreed (unconfirmed)
        self.last: Dict[str, float] = {}

    def after_update(self) -> Dict[str, float]:
        """Count one update; run the canary when due. Returns the TB scalars it produced (``{}`` on
        a skipped update — the cadence writes nothing, never a stale value)."""
        self.updates += 1
        if self.updates != self.first and self.updates % self.every:
            return {}
        return self.run(grad=(self.runs % self.grad_every) == self.grad_every - 1)

    # ---------------------------------------------------------------------------- one verdict
    def _check(self, grad: bool, slice_: int, out: Dict[str, float]) -> List[str]:
        """One full comparison (compiled AND eager re-computed) on fixture slice ``slice_``. Raises
        `CompileTrainerError` on a disagreement; restores training mode and `.grad` either way."""
        model = self.model
        policy = model.policy
        precision = torch.get_float32_matmul_precision()
        was_training = bool(policy.training)
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices):
                from agents.model import compile_regions as cr
                if cr.installed(model):
                    return self._regions(model, precision, grad, out, slice_)
                return self._extractor(model, precision, grad, out, slice_)
        finally:
            policy.set_training_mode(was_training)
            for p in policy.parameters():
                p.grad = None

    def _record(self, verdict: str, detail: str = "") -> Dict[str, Any]:
        rec = {"verdict": verdict, "update": self.updates,
               "num_timesteps": int(getattr(self.model, "num_timesteps", 0) or 0),
               "time": time.time(), "detail": detail[:4000]}
        _append(_run_dir(self.model), VERDICTS_FILE, {k: v for k, v in rec.items() if k != "detail"}
                if verdict == "pass" else rec)
        return rec

    def _say(self, line: str) -> None:
        print(line, flush=True)
        if self._emit is not None:
            with contextlib.suppress(Exception):
                self._emit(line[:500])

    def _fatal(self, why: str, first: BaseException) -> None:
        run_dir = _run_dir(self.model)
        saved = "not written (no run directory declared)"
        if run_dir:
            path = os.path.join(run_dir, FATAL_CHECKPOINT)
            try:
                (self._saver or self.model.save)(path)
                saved = path + ".zip (forensics only: these weights trained on the graph in question)"
            except Exception as exc:                          # the FATAL stands either way
                saved = f"FAILED to write ({type(exc).__name__}: {exc})"
        self._record("fatal", f"{why} — {first}")
        raise CompileCanaryError(
            f"{CANARY_TAG} after update {self.updates}: {why}. The COMPILED learner no longer equals "
            f"eager on the committed real-obs fixture at the startup gate's bars — {first}\n"
            f"Checkpoint: {saved}.\nSAFE ROLLBACK POINT: {rollback_point(run_dir)}.\n"
            f"Training on a wrong function is GIGO: fatal by design, the launcher does NOT restart it "
            f"(FATAL_CONFIG). See designs/training/learner_lifecycle.md (the canary).") from first

    def run(self, *, grad: bool) -> Dict[str, float]:
        precision = torch.get_float32_matmul_precision()
        out: Dict[str, float] = {}
        t0 = time.perf_counter()
        try:
            rules = self._check(grad, 0, out)
        except ct.CompileTrainerError as first:
            # (1) warn + dump, then CONFIRM in the same update: the same rows again and an
            # independent slice, compiled and eager both re-computed.
            _append(_run_dir(self.model), DISAGREEMENTS_FILE,
                    {"update": self.updates, "num_timesteps": int(getattr(self.model, "num_timesteps", 0) or 0),
                     "precision": precision, "grad_checked": bool(grad), "verdict": str(first)[:4000]})
            self._say(f"⚠️  [CompileCanary] update {self.updates}: compiled DISAGREES with eager — "
                      f"confirming now (the same rows + an independent slice). {str(first)[:300]}")
            again = []
            for sl in (0, 1):
                try:
                    self._check(grad, sl, {})
                except ct.CompileTrainerError as exc:
                    again.append((sl, exc))
            if len(again) == 2:                              # (2) confirmed on both
                self._fatal("the disagreement is CONFIRMED (it reproduced on the same rows and on an "
                            "independent fixture slice in the same update)", first)
            if self.prev_disagreed:                          # (4) two consecutive scheduled canaries
                self._fatal("the previous scheduled canary disagreed too — two CONSECUTIVE "
                            "disagreements, each unconfirmed", first)
            # (3) not confirmed: count it, keep training, the next scheduled canary decides
            self.unconfirmed += 1
            self.prev_disagreed = True
            self.runs += 1
            self._record("unconfirmed", f"reproduced on slice(s) {[sl for sl, _ in again]} of [0, 1]; {first}")
            self._say(f"⚠️  [CompileCanary] update {self.updates}: the disagreement did NOT confirm "
                      f"(reproduced on slice(s) {[sl for sl, _ in again]} of [0, 1]) — training "
                      f"continues; {self.unconfirmed} unconfirmed so far; a disagreement at the NEXT "
                      f"scheduled canary is FATAL.")
            out = {"compile/canary_ok": 0.0, "compile/canary_grad_checked": 1.0 if grad else 0.0,
                   "compile/canary_unconfirmed_disagreements": float(self.unconfirmed),
                   "compile/canary_seconds": time.perf_counter() - t0}
            self.last = out
            return out
        self.prev_disagreed = False
        self.runs += 1
        self._record("pass")
        out["compile/canary_ok"] = 1.0
        out["compile/canary_grad_checked"] = 1.0 if grad else 0.0
        out["compile/canary_unconfirmed_disagreements"] = float(self.unconfirmed)
        out["compile/canary_seconds"] = time.perf_counter() - t0
        self.last = out
        self._say(f"🐤 [CompileCanary] update {self.updates}: compiled == eager on the fixture "
                  f"({precision}, {out['compile/canary_seconds']:.1f} s) — " + " | ".join(rules))
        return out

    # ------------------------------------------------------------------ the two compiled shapes
    def _regions(self, model: Any, precision: str, grad: bool, out: Dict[str, float],
                 slice_: int = 0) -> list:
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
            obs = ct._prewarm_obs(model, self.n_envs, slice_)
            _, mask = ct._parity_obs(int(obs["observation"].shape[-1]), self.n_envs,
                                     ct.resolve_device(policy.features_extractor), slice_)
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
            args = cr._r1_args(model, cr.r1_batch(model, self.batch_size, slice_))
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

    def _extractor(self, model: Any, precision: str, grad: bool, out: Dict[str, float],
                   slice_: int = 0) -> list:
        """The legacy extractor-only compile (torch 2.5.1): the gate's own readout and train step."""
        policy = model.policy
        fe = policy.features_extractor
        rules = []
        obs, mask = _fixture_obs(model, self.n_envs, slice_)
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
            tobs, tmask = _fixture_obs(model, self.batch_size, slice_)
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
