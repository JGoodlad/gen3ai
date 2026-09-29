"""THE DIAGNOSTICS CADENCE (`--diagnostics-every N`, gen3_diagnostics_cadence_v1, config v124).

M5 Lane K2 (`designs/endstate/program_rust_core.md`). The learner benchmark
(`designs/research_state/measurements/learner_bench_2026-09-28/README.md`) measured the production
update at 58.5 s, of which the OPTIONAL telemetry below is 12.7% (the per-term noise-scale probe
alone 11.5%). None of it changes a number the optimizer sees; all of it answers a question whose
answer moves over tens of updates, not one. So it runs on every Nth update, and a skipped update
writes NONE of its TB scalars — a gap, never a stale repeat.

THE GATED SET — the exact set the benchmark's "all optional telemetry OFF" config removed:

  | probe                                   | tags                                  | cost / call      |
  |-----------------------------------------|---------------------------------------|------------------|
  | per-term noise sampler                  | `train/noise_scale_{g}`,              | 5 extra backward |
  |   (`noise_scale_terms.PerTermNoiseSampler`) | `train/noise_scale_ratio_{g}`,    | traversals × accum|
  |                                         | `train/noise_scale_share_{g}`,        | micro-batches    |
  |                                         | `train/noise_per_term_ms`             |                  |
  | shared-trunk gradient balance           | `grad/*` (+ `train/cf_grad_share`,    | per-term         |
  |   (`grad_balance.grad_balance_metrics`) | `train/cf_evidential_grad_share`)     | `autograd.grad`  |
  | effective rank (`rank_metrics.rank_probe`) | `rank/{trunk,value_cls,policy,vf_feat}_*` | 1 no-grad fwd + SVD |
  | per-family liveness (`edge_family_metrics`) | `edge/*`                          | param norms      |
  | pointer-cell liveness (`cell_family_metrics`) | `cell/*`                        | param norms      |

EVERYTHING ELSE STAYS EVERY UPDATE: the loss terms, `train/approx_kl` (+ per-epoch), clip
fractions, `train/grad_norm`, `train/train_ms`, the TOTAL noise scale (`train/noise_scale`,
`train/noise_scale_ratio` — free: two grad-norm reads the accumulation already makes, and the
`--adaptive-batch total` controller's input), `signal/adv_*`, the head metrics. The capacity
telemetry (`--capacity-telemetry`, off in production) keeps its OWN cadences.

A PROBE A CONSUMER READS EVERY UPDATE IS LOAD-BEARING and is exempt from the cadence, declared by
the run's own flags (`apply_training_hparams` sets the two attributes below):

  * `rank_probe_every_update` — `--rank-tripwire warn|abort` (production: `warn`). The tripwire's
    constants (skip 5, baseline 20, EMA half-life 10, persistence 3) are counted in READINGS and
    were validated at one reading per update; thinning its input would silently stretch every one
    of them ×N. So the rank probe stays every update while the tripwire is registered.
  * `noise_terms_every_update` — `--adaptive-batch` steering by a PER-TERM ratio (`policy`): the
    controller moves K off `train/noise_scale_ratio_policy`'s EMA and its warm-up gate counts that
    EMA's samples, so the per-term sampler stays every update under it. (`total` mode reads the
    total EMA, which is never gated.)

THE FIRST UPDATE OF EVERY PROCESS IS ALWAYS A DIAGNOSTIC UPDATE. Two reasons. (1) The declared
lifecycle (`program_rust_core.md` §M5): `compile_control` LOCKS the compiled learner after the
first real update, so every signature the steady state will use must be seen by then — the rank
probe's no-grad train-mode forward and the probes' `autograd.grad(retain_graph=True)` calls are
such signatures, and a probe first reached on update N would be a post-lock recompile (a typed
FATAL). (2) A restart then reads every diagnostic immediately. The latch
(`_diagnostics_ran_in_process`) is excluded from the checkpoint so each process re-arms it.

THE PHASE is the ROLLOUT INDEX, `num_timesteps // (n_steps · n_envs)`, not a process-local
counter: it is restored with the checkpoint, so the cadence keeps its phase across a launcher
restart and a reader can predict which steps carry a reading.

THE GUARANTEE (pinned by `diagnostics_cadence_test.py`): learning is BIT-IDENTICAL with the
diagnostics on and off — same seed, same buffer, one update ⇒ identical parameters, optimizer
state, losses and RNG state. Every gated probe is read-only (`autograd.grad` never writes `.grad`;
the rank forward is `no_grad` with dropout 0; the liveness reads are norms) and none draws a
random number.
"""
from __future__ import annotations

from typing import NamedTuple

#: The value a FRESH CLI run resolves to. 10 recovers ~90% of the measured 12.7% (the probes run
#: on 1 update in 10) while every gated series still gets a reading per ~1M env steps at the
#: production rollout (64 envs × 2048 steps = 131k steps/update): dense enough for every reader
#: of these tags, all of which read them over windows of tens of readings or longer.
DIAGNOSTICS_EVERY_DEFAULT = 10
#: What a config recorded before v124 means: every run before the flag ran them every update.
DIAGNOSTICS_EVERY_PRE_V124 = 1


class DiagnosticsPlan(NamedTuple):
    """Which gated probes run on THIS `train()` call."""
    due: bool            # a cadence update (or the first update of the process, or N == 1)
    noise_terms: bool    # the per-term noise sampler
    grad_balance: bool   # `grad/*` (and the two `train/cf*_grad_share` derived from it)
    rank: bool           # `rank/*`
    liveness: bool       # `edge/*` + `cell/*`


def rollout_index(model: object) -> int:
    """The number of rollouts this model has collected, from the RESTORED step counter."""
    size = int(getattr(model, "n_steps", 0) or 0) * int(getattr(model, "n_envs", 0) or 0)
    return int(getattr(model, "num_timesteps", 0) or 0) // max(1, size)


def diagnostics_every(model: object) -> int:
    return max(1, int(getattr(model, "diagnostics_every", DIAGNOSTICS_EVERY_PRE_V124) or 1))


def plan_for(model: object) -> DiagnosticsPlan:
    """The plan for the `train()` call about to run. Pure — `mark_ran` is the one side effect."""
    every = diagnostics_every(model)
    first = not bool(getattr(model, "_diagnostics_ran_in_process", False))
    due = every == 1 or first or rollout_index(model) % every == 0
    return DiagnosticsPlan(
        due=due,
        noise_terms=due or bool(getattr(model, "noise_terms_every_update", False)),
        grad_balance=due,
        rank=due or bool(getattr(model, "rank_probe_every_update", False)),
        liveness=due,
    )


def mark_ran(model: object, plan: DiagnosticsPlan) -> None:
    if plan.due:
        setattr(model, "_diagnostics_ran_in_process", True)
