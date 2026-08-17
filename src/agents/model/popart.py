"""PopArt value-target normalizer (van Hasselt et al. 2016, *Learning values across
many orders of magnitude*).

The dual-head extractor shares one transformer trunk; with a large / unclipped value loss the
value gradient **swamps** the trunk (see `agents/training/grad_balance.py`). The root cause is
the value-target *scale*: with γ≈0.9999 the returns run to ±hundreds, so the value MSE — and its
gradient — dwarfs the policy gradient. PopArt fixes the scale **at the source, adaptively**:

* **ART** — *Adaptively Rescale Targets.* Maintain a running ``(mu, sigma)`` of the value targets
  (the returns) and train the value head against ``(target - mu) / sigma`` so the value loss — and
  its gradient into the shared trunk — stays O(1) regardless of the return scale.
* **POP** — *Preserve Outputs Precisely.* Whenever ``(mu, sigma)`` updates, rewrite the value head's
  output ``Linear`` so its **de-normalized** prediction is unchanged for every input
  (``W' = (sigma_old/sigma_new)·W``, ``b' = (sigma_old·b + mu_old − mu_new)/sigma_new``). This makes
  the stats update a no-op on the actual value function — so adapting the scale never corrupts what
  the critic has learned (the failure mode of naive running-std normalization).

Usage contract: the policy's value head outputs **normalized** values; callers that need a real-unit
value (GAE / advantages / bootstrapping) call :meth:`denormalize`; the PPO loss normalizes both the
prediction and the target via :meth:`normalize`; once per ``train()`` call the trainer calls
:meth:`update` with the rollout's returns (which also runs POP). Buffers register in the policy
state_dict, so ``(mu, sigma)`` save/restore across checkpoints automatically.

Pure (torch-only, no SB3) → unit-tested in ``popart_test.py``; the load-bearing test is **POP
invariance** (de-normalized outputs identical before/after a stats update).
"""
from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import torch as th
from torch import nn

# Per-``train()``-call EMA decay for the running (mu, second-moment) estimate. One update per
# rollout (~10-update window ⇒ tracks the slow return drift from policy improvement / reward
# annealing while staying smooth). A module constant, not a CLI flag — the only flag is on/off.
_DEFAULT_BETA = 0.1
# Lower bound on sigma so normalization can't divide by ~0 on a degenerate (near-constant) return
# batch. Far below any real return scale, so it never binds in normal training.
_SIGMA_FLOOR = 1e-2


class PopArtNormalizer(nn.Module):
    """Running value-target normalizer with output-preserving (POP) weight surgery.

    Holds scalar buffers ``mu`` / ``sigma`` / ``nu`` (second moment) / ``initialized``; no learnable
    parameters. ``normalize`` / ``denormalize`` are the (de)normalization maps; ``update`` advances
    the running stats from a batch of targets and rescales the value head's output layer to preserve
    its de-normalized outputs.
    """

    if TYPE_CHECKING:
        # Registered buffers exist only dynamically, so `Module.__getattr__` types them
        # `Any`. Declaring them keeps the (de)normalization maps typed. No runtime effect.
        mu: th.Tensor
        sigma: th.Tensor
        nu: th.Tensor
        initialized: th.Tensor

    def __init__(self, beta: float = _DEFAULT_BETA, sigma_floor: float = _SIGMA_FLOOR) -> None:
        super().__init__()
        self._beta = float(beta)
        self._floor = float(sigma_floor)
        self.register_buffer("mu", th.zeros(()))
        self.register_buffer("sigma", th.ones(()))
        self.register_buffer("nu", th.ones(()))  # running E[target^2]
        self.register_buffer("initialized", th.zeros((), dtype=th.bool))

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Map real-unit values/targets → normalized space (the value-loss space)."""
        return (x - self.mu) / self.sigma

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        """Map the value head's normalized output → a real-unit value (for GAE / advantages)."""
        return x * self.sigma + self.mu

    @th.no_grad()
    def update(self, targets: th.Tensor, output_layer: nn.Linear) -> Dict[str, float]:
        """Advance running ``(mu, sigma)`` from ``targets`` and POP-rescale ``output_layer``.

        Call **once per ``train()``** before the gradient epochs so the loss uses the fresh stats and
        the value head stays output-consistent across the shift. ``output_layer`` is the value head's
        final ``nn.Linear`` (e.g. ``policy.value_net``); it is rescaled in place (under ``no_grad``).
        Returns ``{popart/mu, popart/sigma}`` for logging.

        The POP rescale changes the layer's weights outside the optimizer, so Adam's momentum for
        those params is momentarily stale — negligible because the EMA keeps ``sigma_old/sigma_new ≈
        1`` each call (the standard PopArt approximation; optimizer state is intentionally not
        rescaled).
        """
        t = targets.detach().reshape(-1).to(self.mu)
        if t.numel() == 0:
            return self.stats()

        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()

        batch_mu = t.mean()
        batch_nu = (t * t).mean()
        if not bool(self.initialized):
            self.mu.copy_(batch_mu)
            self.nu.copy_(batch_nu)
            self.initialized.fill_(True)
        else:
            self.mu.mul_(1.0 - self._beta).add_(batch_mu, alpha=self._beta)
            self.nu.mul_(1.0 - self._beta).add_(batch_nu, alpha=self._beta)
        self.sigma.copy_(th.sqrt(th.clamp(self.nu - self.mu * self.mu, min=self._floor * self._floor)))

        # POP: preserve the de-normalized output of every input across the (mu, sigma) change.
        ratio = old_sigma / self.sigma
        output_layer.weight.mul_(ratio)
        output_layer.bias.copy_((old_sigma * output_layer.bias + old_mu - self.mu) / self.sigma)
        return self.stats()

    def stats(self) -> Dict[str, float]:
        """The current running stats as plain floats (for TensorBoard)."""
        return {"popart/mu": float(self.mu), "popart/sigma": float(self.sigma)}
