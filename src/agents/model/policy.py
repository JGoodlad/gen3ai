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


class Gen3DualHeadMaskablePolicy(MaskableMultiInputActorCriticPolicy):
    """Maskable actor-critic policy whose features extractor yields a (pi, vf) tuple.

    ``self.extract_features(obs)`` returns ``(pi_features, vf_features)`` because the
    shared-extractor path simply returns whatever the extractor returns. Each consumer
    below unpacks that tuple and feeds ``mlp_extractor.forward_actor`` / ``forward_critic``
    the appropriate half. Everything else (action distribution, value net, masking) is
    inherited unchanged.
    """

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ):
        pi_features, vf_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
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
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
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
        return self.value_net(latent_vf)
