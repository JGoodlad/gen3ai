"""Dual-head maskable policy for the value-dedicated CLS readout (H4 / Option C).

`Gen3FeaturesExtractor.forward` returns a ``(pi_features, vf_features)`` tuple: the
transformer body is shared, but the actor and critic read it through independent CLS
pools + projection heads. Stock SB3 policies assume the features extractor returns a
single tensor, so this policy overrides the four methods that consume features and routes
each half of the tuple to its own ``mlp_extractor`` branch.

Design note — we deliberately keep ``share_features_extractor=True`` so SB3 builds exactly
ONE features-extractor instance (one transformer body). The "sharing" is real at the body
level; the split happens inside the extractor's two readouts and is preserved here by
unpacking the tuple. We do NOT use ``share_features_extractor=False`` because that would
make SB3 instantiate a second full body (Option A, ~2× compute) — not what Option C wants.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th

from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

from agents.model.popart import PopArtNormalizer


class Gen3DualHeadMaskablePolicy(MaskableMultiInputActorCriticPolicy):
    """Maskable actor-critic policy whose features extractor yields a (pi, vf) tuple.

    ``self.extract_features(obs)`` returns ``(pi_features, vf_features)`` because the
    shared-extractor path simply returns whatever the extractor returns. Each consumer
    below unpacks that tuple and feeds ``mlp_extractor.forward_actor`` / ``forward_critic``
    the appropriate half. Everything else (action distribution, value net, masking) is
    inherited unchanged.

    **PopArt (opt-in via ``use_popart=True`` in ``policy_kwargs``).** When enabled, the value head
    (``value_net``) outputs *normalized* values and a :class:`~agents.model.popart.PopArtNormalizer`
    de-normalizes every value site below, so callers (GAE / advantages / bootstrapping) always see
    real-unit values while the PPO loss trains in normalized space (see
    ``agents/training/instrumented_ppo.py``). The normalizer is built **after** ``super().__init__``
    (which builds ``value_net``); its ``(mu, sigma)`` buffers ride the policy state_dict, so they
    save/restore across checkpoints. ``use_popart`` is version-checked (``ModelVersion``) — it cannot
    be toggled on a resumed model.
    """

    def __init__(self, *args, use_popart: bool = False, value_from_dist: bool = False, **kwargs):
        # super().__init__ builds value_net (SB3 _build); the normalizer wraps it afterwards.
        super().__init__(*args, **kwargs)
        self.popart = PopArtNormalizer() if use_popart else None
        # gen3_dist_critic_v1 (Phase B): when True the GAE/bootstrap/deployed value is E[Z] from the
        # distributional head instead of the scalar value_net (which freezes as a fallback + monitor).
        # A resume-immutable training-behavior toggle (the belief_grad_mode class) — see set_ / the
        # ModelVersion gate. Requires value_dist_mode == "shaping" (the head must be a live critic).
        self._value_from_dist = bool(value_from_dist)

        # gen3_identity_init_guard_v1: SB3's `_build()` just ran
        # `features_extractor.apply(init_weights, gain=sqrt(2))`, which orthogonally re-initialises
        # EVERY nn.Linear in the extractor — silently destroying every deliberate zero-init inside it
        # (refine_proj, outgoing_proj, status_{in,out}_proj, film_pi/vf, and the belief heads whose
        # zero-init is what makes their cold-start posterior EQUAL the prior). Restore them here, now
        # that the policy is fully built. See Gen3FeaturesExtractor.restore_identity_init.
        for _fe in {id(m): m for m in (getattr(self, "features_extractor", None),
                                       getattr(self, "pi_features_extractor", None),
                                       getattr(self, "vf_features_extractor", None)) if m is not None}.values():
            if hasattr(_fe, "restore_identity_init"):
                _fe.restore_identity_init()

    def set_value_from_dist(self, on: bool) -> None:
        """Apply value_from_dist at RUNTIME (the --value-from-dist migration path). SB3's load
        reconstructs the policy from the ZIP's SAVED policy_kwargs, so a first Phase-B resume (from a
        pre-v45 checkpoint whose kwargs lack the key) would otherwise be a SILENT NO-OP — the migration
        notice prints but the loaded policy keeps _value_from_dist=False (the 2026-07-22 catch:
        grad/value_dist_share stayed ~0.05 instead of ~0.5). Call this post-load on resume; no-op when
        unchanged. Same fix as features_extractor.set_belief_grad_mode."""
        changed = bool(on) != bool(getattr(self, "_value_from_dist", False))
        self._value_from_dist = bool(on)
        if changed:
            print(f"[Gen3DualHeadMaskablePolicy] value_from_dist APPLIED at runtime -> {bool(on)} "
                  f"(critic = {'distributional E[Z]' if on else 'scalar value_net'})")

    def _denorm(self, values: th.Tensor) -> th.Tensor:
        """Map the value head's (possibly normalized) output to a real-unit value. Identity when
        PopArt is disabled, so the real-unit GAE / advantage path is unchanged."""
        return self.popart.denormalize(values) if self.popart is not None else values

    def _critic_value(self, latent_vf: th.Tensor) -> th.Tensor:
        """The real-unit critic value used by GAE / bootstrap / deployment. Phase B: E[Z] from the
        distributional head (normalized) → _denorm (same PopArt peg as the scalar) → real units, so
        the plumbing is byte-for-byte the scalar path except the source. Falls back to the scalar
        value_net when off, or if the dist logits aren't stashed (defensive — never silently wrong)."""
        if self._value_from_dist:
            fe = self.features_extractor
            head = getattr(fe, "value_dist_head", None)
            logits = getattr(fe, "last_value_dist_logits", None)
            if head is not None and logits is not None:
                return self._denorm(head.mean(logits))
        return self._denorm(self.value_net(latent_vf))

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        """gen3_pointer_head_v1: add the pointer head's per-action DELTA to the flat head's logits.

        All three logit sites (`forward`, `evaluate_actions`, `get_distribution`) funnel through this
        method, and each calls `extract_features` immediately before — so the extractor's stash is
        always fresh for THIS batch. That makes this the one correct interception point; adding the
        delta in any single caller would silently skip the other two (e.g. PPO's epoch recompute in
        `evaluate_actions` would then disagree with the rollout's `forward`, corrupting the ratio).

        The delta is a plain additive term on the logits, computed deterministically from the same
        observation, so log-prob / entropy / the PPO ratio are all unaffected in form — and because
        the head's scorers are zero-init, the delta is EXACTLY 0 until it trains, making an ON run
        byte-identical to the flat-head baseline at step 0.
        """
        dist = super()._get_action_dist_from_latent(latent_pi)
        delta = getattr(self.features_extractor, "last_pointer_delta", None)
        if delta is None:
            return dist
        # sb3's MaskableCategoricalDistribution holds the logits on `dist.distribution.logits`;
        # rebuild through the public API so masking/log_prob/entropy all see the combined logits.
        return dist.proba_distribution(action_logits=dist.distribution.logits + delta)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ):
        pi_features, vf_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self._critic_value(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ):
        pi_features, vf_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        # gen3_exploiter_distill_v1: stash the (masked) pi distribution so the exploiter-distillation KL in
        # InstrumentedMaskablePPO.train() can REUSE this forward instead of a redundant second
        # get_distribution. The masked logits give a BIT-IDENTICAL KL (over LEGAL actions the logits are
        # unchanged; illegal actions contribute exactly 0 either way). A no-op for any non-distill run.
        self._last_pi_distribution = distribution
        log_prob = distribution.log_prob(actions)
        values = self._critic_value(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(
        self, obs, action_masks: Optional[np.ndarray] = None
    ) -> MaskableDistribution:
        pi_features, _ = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs) -> th.Tensor:
        _, vf_features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return self._critic_value(latent_vf)
