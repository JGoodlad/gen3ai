"""LIVE CAPACITY TELEMETRY (`gen3_capacity_telemetry_v1`) — three continuous saturation warnings.

WHY THIS EXISTS, and why it is not an offline probe. "Is the network out of capacity?" has always
been answered here by expensive one-shot instruments — a rank probe, an ablation sweep, an offline
battery — each of which reports a NUMBER at a MOMENT. Saturation is not a moment; it is a TREND,
and a trend measured twice is a line through two points. Everything in this module therefore rides
the train loop continuously and costs a few percent of one train step, so the reading that matters
(the SHAPE of the curve over tens of millions of steps) exists at all.

THE THREE SCALAR FAMILIES, and what each one answers.

  1. `capacity/canary_*` — **THE PLASTICITY CANARY**, the centerpiece and the only SUPPLY-side
     probe of the three. A tiny head regresses the trunk's `value_pooled` onto K=4 SYNTHETIC fixed
     targets that are pure functions of the observation (below), and every `--canary-reset-steps`
     env steps ONE of those targets is RE-SEEDED, round-robin. The reset is the whole instrument:
     re-fitting a *brand-new* random function of the obs, from the same representation, with the
     same head, measures how much usable structure the representation still SUPPLIES. A model whose
     trunk has collapsed onto the policy's current answers re-fits slower and plateaus higher, and
     that shows up as `capacity/canary_recovery` degrading from one reset to the next.

     ⚠️ It measures the REPRESENTATION's richness, not the policy's headroom, and those are not the
     same claim. A rising `canary_loss` says the trunk carries less recoverable obs structure than
     it did; it does NOT say the policy would be better if it carried more. Read it as an
     early-warning that something is narrowing, then go find out what.

  2. `capacity/halfbatch_cosine` — **INTERFERENCE**. Split the minibatch in half, take each half's
     gradient on the shared trunk, and read the cosine between them. Two halves of one on-policy
     batch are i.i.d. draws from the same distribution, so a healthy batch has them broadly
     agreeing (cosine > 0, falling slowly as the gradient shrinks toward a stationary point). A
     cosine trending to zero or NEGATIVE means the batch is increasingly fighting itself — the
     model is spending its capacity trading one part of the state space against another rather
     than improving on both. That is the classic signature of a saturated shared trunk.

  3. `capacity/feature_velocity` — **DO THE FUNCTIONS STILL MOVE?** One FIXED probe batch of 256
     obs rows, captured once at launch and never changed, is forwarded every N `train()` calls and
     the mean L2 displacement of `value_pooled` between consecutive measurements is logged. Read it
     BESIDE `train/grad_norm`: weights moving (grad_norm steady) while functions do not
     (feature_velocity falling) is the exact fingerprint of a network burning gradient on a
     representation that has stopped changing.

THE SYNTHETIC TARGET FAMILY — RECORD THIS EXACTLY; a deferred offline probe must use the SAME one
or the two instruments do not cross-validate:

    seed(k, e) = CANARY_SEED_BASE + k + CANARY_RESEED_STRIDE * e
               = 20260823 + k + 1_000_000 * e            (k = target index, e = its reseed count)

    P[:, k]    = torch.randn(obs_dim, generator=torch.Generator("cpu").manual_seed(seed(k, e)))

    target_k(obs) = tanh( obs @ P[:, k] / sqrt(obs_dim) )

The generator is always CPU-seeded, so the same `(k, e)` gives the same column on any device and in
any process; `e` starts at 0 for every target and increments only when THAT target is re-seeded.
`tanh` bounds the target into (-1, 1) so the regression loss has a fixed scale across resets — an
unbounded target would make `canary_recovery` a ratio of two different scales.

WHAT IS **NOT** HERE, deliberately. None of this is an `nn.Module` on the policy or the extractor.
The canary head is owned by the PPO object, trained by its OWN optimizer, and reads a
`.detach()`ed input — so it adds no `state_dict` key, cannot disturb the policy optimizer's
positional state (the ai_v6_13 "128 vs 5" class), and cannot reach the trunk at ANY value of
anything. `capacity_gradient_isolation` in the test file is that claim as a measurement.
"""
from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, Optional

import torch as th
from torch import nn

# ----------------------------------------------------------------- the target family's constants
CANARY_SEED_BASE = 20260823
CANARY_RESEED_STRIDE = 1_000_000
CANARY_K = 4

# The head. Deliberately SMALL and deliberately non-linear: a linear probe would measure only the
# representation's linear content, and the question is whether the trunk still supplies structure a
# downstream MLP (which is what the policy and critic heads are) can use.
CANARY_HIDDEN = 128
CANARY_LR = 1e-3
CANARY_EMA_DECAY = 0.99

# The velocity probe's fixed batch — an UPPER bound, since it is sliced off the first minibatch and
# a debug run's `--batch-size` can be smaller. 256 rows is enough for a stable mean displacement and
# small enough that the forward is noise against a production minibatch.
VELOCITY_PROBE_ROWS = 256


def canary_seed(k: int, reseeds: int) -> int:
    """The seed of target ``k`` after it has been re-seeded ``reseeds`` times. See module docstring."""
    return CANARY_SEED_BASE + int(k) + CANARY_RESEED_STRIDE * int(reseeds)


def canary_projection_column(k: int, reseeds: int, obs_dim: int) -> th.Tensor:
    """One column of ``P`` — ``randn(obs_dim)`` under ``canary_seed(k, reseeds)``, always CPU-drawn.

    CPU-drawn on purpose: a CUDA generator and a CPU generator do not agree at the same seed, so a
    device-dependent target would make a run's canary curve incomparable to a laptop's replay of
    the same family.
    """
    gen = th.Generator(device="cpu").manual_seed(canary_seed(k, reseeds))
    return th.randn(int(obs_dim), generator=gen, dtype=th.float32)


def canary_targets(obs: th.Tensor, projection: th.Tensor) -> th.Tensor:
    """``obs`` [B, obs_dim] and ``P`` [obs_dim, K] -> ``tanh(obs @ P / sqrt(obs_dim))`` [B, K]."""
    return th.tanh(obs @ projection / math.sqrt(float(obs.shape[-1])))


class PlasticityCanaryHead(nn.Module):
    """The regression head — LayerNorm -> Linear -> ReLU -> Linear, K outputs.

    Shaped after `CfEvidentialHead`: a readout that is never called by any forward, whose input is
    detached unconditionally. It differs in ONE way that is the point of the design — it is not a
    child of the extractor, so it contributes no `state_dict` key and no optimizer position.
    """

    def __init__(self, in_dim: int, k: int = CANARY_K, hidden: int = CANARY_HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(k)),
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        return self.net(features)


class PlasticityCanary:
    """The head, its OWN optimizer, the projection matrix, and the round-robin reset schedule.

    Not an ``nn.Module``: it OWNS one, and keeping the container a plain object is what stops it
    ever being registered as a child of something that gets saved or optimized.
    """

    def __init__(self, feature_dim: int, obs_dim: int, *, k: int = CANARY_K,
                 reset_steps: int = 1_000_000, device: Any = "cpu", lr: float = CANARY_LR) -> None:
        self.k = int(k)
        self.obs_dim = int(obs_dim)
        self.reset_steps = int(reset_steps)
        self.device = device
        self.head = PlasticityCanaryHead(int(feature_dim), self.k).to(device)
        self.opt = th.optim.Adam(self.head.parameters(), lr=float(lr))
        self.reseeds = [0] * self.k
        self.projection = th.stack(
            [canary_projection_column(i, 0, self.obs_dim) for i in range(self.k)], dim=1
        ).to(device)                                    # [obs_dim, K]
        self.ema: list[Optional[float]] = [None] * self.k
        self.last_reset_step: Optional[int] = None      # env-step the schedule last fired (or started)
        self.next_k = 0                                 # the round-robin cursor
        self.pre_reset_loss: Optional[float] = None     # EMA of the reset target, just BEFORE its reset
        self.reset_target: Optional[int] = None         # which target was re-seeded most recently
        self.n_resets = 0
        self.steps = 0                                  # canary optimizer steps taken this process

    # ------------------------------------------------------------------------------ the schedule
    def maybe_reset(self, num_timesteps: int) -> bool:
        """Re-seed ONE target if ``reset_steps`` env steps have passed. Returns whether it fired.

        The FIRST call only arms the clock — a reset at step 0 would re-seed a target that has never
        been fitted, so `canary_recovery`'s denominator would be an untrained loss.
        """
        now = int(num_timesteps)
        if self.last_reset_step is None:
            self.last_reset_step = now
            return False
        if self.reset_steps <= 0 or now - self.last_reset_step < self.reset_steps:
            return False
        k = self.next_k
        # The denominator of `canary_recovery`: what this target's loss had settled to under the
        # OLD projection. None on the very first reset of a target that never converged.
        self.pre_reset_loss = self.ema[k]
        self.reseeds[k] += 1
        self.projection[:, k] = canary_projection_column(
            k, self.reseeds[k], self.obs_dim).to(self.projection.device)
        # The head's weights are deliberately NOT reset. "Can this network still fit a new function
        # with the parameters it has now" is the question; re-initialising the head would ask a
        # different and much easier one.
        self.ema[k] = None
        self.reset_target = k
        self.next_k = (k + 1) % self.k
        self.last_reset_step = now
        self.n_resets += 1
        return True

    # ------------------------------------------------------------------------------ one training step
    def step(self, features: th.Tensor, obs: th.Tensor) -> None:
        """ONE optimizer step of the canary head on a DETACHED feature batch.

        ``features`` [B, feature_dim] — the trunk's `value_pooled` for this minibatch.
        ``obs``      [B, obs_dim]     — the SAME rows' raw observation (the target's only input).

        The detach is unconditional and is done HERE rather than trusted to the caller, so there is
        exactly one place the isolation claim has to hold.
        """
        feats = features.detach()
        with th.no_grad():
            targets = canary_targets(obs.detach().to(self.projection.dtype), self.projection)
        pred = self.head(feats.to(targets.dtype))
        per_target = (pred - targets).pow(2).mean(dim=0)        # [K]
        loss = per_target.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        vals = per_target.detach().float().cpu().tolist()
        for i, v in enumerate(vals):
            prev = self.ema[i]
            self.ema[i] = v if prev is None else CANARY_EMA_DECAY * prev + (1.0 - CANARY_EMA_DECAY) * v
        self.steps += 1

    # ------------------------------------------------------------------------------ the read
    def metrics(self, num_timesteps: int) -> Dict[str, float]:
        """The `capacity/canary_*` scalars. Empty until at least one step has run."""
        live = [v for v in self.ema if v is not None]
        if not live:
            return {}
        out: Dict[str, float] = {
            "canary_loss": float(sum(live) / len(live)),
            "canary_resets": float(self.n_resets),
        }
        if self.last_reset_step is not None:
            out["canary_age"] = float(int(num_timesteps) - self.last_reset_step)
        rt = self.reset_target
        if rt is not None and self.ema[rt] is not None:
            out["canary_loss_reset"] = float(self.ema[rt])
            # THE one-number read. >1 = the re-seeded target is still worse than the retired one
            # was; it decays toward ~1 as the head re-fits. Compare it AT A MATCHED `canary_age` —
            # the whole curve is the measurement, and a bare value is a point on it.
            if self.pre_reset_loss:
                out["canary_recovery"] = float(self.ema[rt] / self.pre_reset_loss)
        return out


# --------------------------------------------------------------------- 2. half-batch trunk cosine

def flat_grad(loss: th.Tensor, params) -> th.Tensor:
    """Flattened ``d loss / d params``, zeros for a param the loss does not reach.

    ``autograd.grad`` — NOT ``backward()`` — so ``.grad`` is never written and the real optimizer's
    accumulated gradient is untouched. That is the no-corruption property the cosine probe rests on.
    """
    grads = th.autograd.grad(loss, list(params), allow_unused=True, retain_graph=False)
    return th.cat([
        (g if g is not None else th.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def _ppo_surrogate(model, obs, actions, masks, advantages, old_log_prob, returns, clip_range):
    """The PPO objective on one slice — clipped policy loss + ``vf_coef``·MSE, nothing else.

    Deliberately the PLAIN form rather than the run's full fold: the cosine asks whether the two
    halves of a batch agree about the RL objective, and folding in a dozen auxiliaries would make
    the answer a statement about the auxiliaries' agreement instead. PopArt/tail-weighting are
    likewise skipped — both are monotone rescalings of the same per-sample residual, so they move
    the gradient's LENGTH and not the angle this probe reads.
    """
    values, log_prob, _entropy = model.policy.evaluate_actions(obs, actions, action_masks=masks)
    values = values.flatten()
    ratio = th.exp(log_prob - old_log_prob)
    policy_loss = -th.min(
        advantages * ratio,
        advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
    ).mean()
    value_loss = th.nn.functional.mse_loss(returns, values)
    return policy_loss + float(model.vf_coef) * value_loss


def halfbatch_trunk_cosine(model, rollout_data, actions, advantages, trunk_params,
                           clip_range: float) -> Dict[str, float]:
    """Cosine between the two half-batches' SHARED-TRUNK gradients. ``{}`` when not measurable.

    ``advantages`` must already be normalized over the WHOLE minibatch (the caller's live tensor):
    re-normalizing each half against its own mean/std would inject a difference the batch does not
    have and bias the cosine down.

    ⚠️ Runs its own ``evaluate_actions`` forwards, so it CLOBBERS the extractor stashes for this
    minibatch — it must be called after every term that reads them (it is: the caller places it
    after the optimizer step).
    """
    params = list(trunk_params)
    n = int(actions.shape[0])
    half = n // 2
    if not params or half < 2:
        return {}
    grads = []
    for sl in (slice(0, half), slice(half, 2 * half)):
        obs = {k: v[sl] for k, v in rollout_data.observations.items()}
        masks = rollout_data.action_masks
        surrogate = _ppo_surrogate(
            model, obs, actions[sl], None if masks is None else masks[sl],
            advantages[sl], rollout_data.old_log_prob[sl], rollout_data.returns[sl], clip_range)
        if not surrogate.requires_grad:
            # No graph at all (a frozen policy, an inference-mode context). `autograd.grad` would
            # RAISE here, and the caller turns a raise into "telemetry off for the rest of the
            # run" — a whole probe lost to a measurable, recoverable condition.
            return {}
        grads.append(flat_grad(surrogate, params))
    g0, g1 = grads
    n0, n1 = float(g0.norm()), float(g1.norm())
    if n0 <= 0.0 or n1 <= 0.0:
        return {}
    return {
        "halfbatch_cosine": float(th.dot(g0, g1)) / (n0 * n1),
        # The two halves' gradient MAGNITUDES. A cosine near zero with wildly unequal norms is a
        # different story (one half dominates) from a cosine near zero with equal norms (genuine
        # disagreement), and the scalar alone cannot tell them apart.
        "halfbatch_grad_norm_ratio": min(n0, n1) / max(n0, n1),
    }


# ------------------------------------------------------------------------------ 3. feature velocity

def probe_features(model, probe_obs: th.Tensor) -> Optional[th.Tensor]:
    """One no_grad EAGER extractor forward on the frozen probe batch -> its ``value_pooled``.

    EAGER (``type(fe).forward``) and observation-key-only, for the reasons `cf_terms` gives: the
    compile flags patch the BOUND ``fe.forward``, and routing a second, differently-shaped obs dict
    through the compiled entry point would add a graph shape for a diagnostic. The
    ObservationDebugger is suppressed — these are replayed rows, not the board this process is
    about to act on.
    """
    fe = model.policy.features_extractor
    dbg_ctx = getattr(fe, "suppress_observation_debugger", contextlib.nullcontext)()
    with dbg_ctx, th.no_grad():
        type(fe).forward(fe, {"observation": probe_obs})
    pooled = getattr(fe, "last_value_pooled", None)
    return None if pooled is None else pooled.detach().float()


def feature_velocity_metrics(current: th.Tensor, previous: Optional[th.Tensor]) -> Dict[str, float]:
    """Displacement of the frozen probe batch's features since the previous measurement."""
    if previous is None or previous.shape != current.shape:
        return {}
    displacement = (current - previous).norm(dim=-1)
    return {
        "feature_velocity": float(displacement.mean()),
        "feature_velocity_cos": float(
            th.nn.functional.cosine_similarity(current, previous, dim=-1).mean()),
        # Scale-free companion: the representation's own norm drifts over a run, so a falling raw
        # velocity can mean "the features shrank" rather than "the function stopped moving".
        "feature_velocity_rel": float(
            displacement.mean() / (previous.norm(dim=-1).mean() + 1e-8)),
    }


# ------------------------------------------------------------------------------------ the holder

class CapacityTelemetry:
    """Everything the flag owns, in one object hung off the PPO model as ``model._capacity``.

    Built LAZILY on the first minibatch that can feed it (it needs the live feature width and the
    obs width, both of which the first `value_pooled` supplies), and never built at all when the
    flag is off — so OFF costs one boolean per minibatch and holds no probe batch, no projection
    matrix and no optimizer.

    ⚠️ **NOT CHECKPOINTED, by design.** `PpoHyperparameters._excluded_save_params` names
    ``_capacity``, so the canary head, its Adam state, the projection matrix and the frozen probe
    batch are all re-created fresh on a resume. The canary's loss therefore JUMPS at every resume
    and every launcher restart, and `canary_recovery` restarts its curve. That is an accepted
    simplification, not an oversight: persisting it would mean pickling an optimizer into every
    checkpoint for a diagnostic, and the alternative reading — compare recoveries WITHIN a restart
    window — is available for free. At production throughput a 3-hour launcher window is ~16M env
    steps, so a 1M-step reset interval still fires ~16 times per window.
    """

    def __init__(self, *, reset_steps: int = 1_000_000, cosine_every: int = 50,
                 velocity_every: int = 50, canary_k: int = CANARY_K) -> None:
        self.reset_steps = int(reset_steps)
        self.cosine_every = int(cosine_every)
        self.velocity_every = int(velocity_every)
        self.canary_k = int(canary_k)
        self.canary: Optional[PlasticityCanary] = None
        self.probe_obs: Optional[th.Tensor] = None
        self.prev_features: Optional[th.Tensor] = None
        self.minibatches = 0        # minibatches seen (== optimizer steps at grad_accum 1)
        self.train_calls = 0
        self.cosine_samples: list[Dict[str, float]] = []
        self.canary_steps_this_train = 0

    # ------------------------------------------------------------------ per-minibatch
    def observe(self, model, rollout_data, actions, advantages, trunk_params,
                clip_range: float, features: Optional[th.Tensor], num_timesteps: int) -> None:
        """The whole per-minibatch half: canary step, then (on cadence) the half-batch cosine.

        ``features`` is the ``value_pooled`` this minibatch's `evaluate_actions` forward left
        behind, snapshotted by the caller BEFORE any own-forward fold clobbered it. ``None`` (a
        non-Gen3 extractor, a stale batch) skips the canary for this minibatch and is COUNTED —
        `capacity/canary_steps` reading 0 with the flag on is the tell.
        """
        self.minibatches += 1
        obs = rollout_data.observations.get("observation")
        if features is not None and obs is not None and features.shape[0] == obs.shape[0]:
            if self.canary is None:
                self.canary = PlasticityCanary(
                    int(features.shape[-1]), int(obs.shape[-1]), k=self.canary_k,
                    reset_steps=self.reset_steps, device=features.device)
            self.canary.maybe_reset(num_timesteps)
            self.canary.step(features, obs)
            self.canary_steps_this_train += 1
        if self.probe_obs is None and obs is not None:
            self.probe_obs = obs[:VELOCITY_PROBE_ROWS].detach().clone()
        if self.cosine_every > 0 and self.minibatches % self.cosine_every == 0:
            got = halfbatch_trunk_cosine(
                model, rollout_data, actions, advantages, trunk_params, clip_range)
            if got:
                self.cosine_samples.append(got)

    # ------------------------------------------------------------------ per-train()
    def finish_train(self, model, num_timesteps: int) -> Dict[str, float]:
        """Fold the per-minibatch samples, run the velocity probe on cadence, return the scalars."""
        self.train_calls += 1
        out: Dict[str, float] = {"canary_steps": float(self.canary_steps_this_train)}
        self.canary_steps_this_train = 0
        if self.canary is not None:
            out.update(self.canary.metrics(num_timesteps))
        if self.cosine_samples:
            for key in self.cosine_samples[0]:
                vals = [s[key] for s in self.cosine_samples if key in s]
                out[key] = float(sum(vals) / len(vals))
            self.cosine_samples = []
        if (self.probe_obs is not None and self.velocity_every > 0
                and self.train_calls % self.velocity_every == 0):
            current = probe_features(model, self.probe_obs)
            if current is not None:
                out.update(feature_velocity_metrics(current, self.prev_features))
                self.prev_features = current
        return out
