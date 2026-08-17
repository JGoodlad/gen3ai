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

from typing import Any, Optional, Tuple

import numpy as np
import torch as th

from stable_baselines3.common.type_aliases import PyTorchObs
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

from agents.model.arch_constants import D_MODEL
from agents.model.popart import PopArtNormalizer


# gen3_policy_activation_pin_v1: the nonlinearity of the SB3 `mlp_extractor` tower
# (`net_arch = [512, 512]`, both the actor and the critic branch).
#
# This value was NEVER chosen here. Until 2026-08-16 `train_rl_agent.py` passed `net_arch` but not
# `activation_fn`, so the tower ran on `MaskableActorCriticPolicy`'s SIGNATURE DEFAULT
# (`activation_fn: type[nn.Module] = nn.Tanh` — the PPO/MuJoCo default). That is a defensible
# choice and it is preserved EXACTLY here: pinning is behaviour-neutral by construction.
#
# It is pinned because the alternative is a silent dependency on someone else's default. An
# sb3-contrib upgrade that changed it would rewrite four layers of the live policy, and NOTHING in
# this repo would notice: `ModelVersion.from_layout_and_policy_kwargs` records
# `net_arch` but has no activation field at all, so the tower's SHAPE is version-checked while its
# NONLINEARITY is not — an activation swap is retrain-class yet weight-shape-neutral, so
# `check_compatible` cannot see it and an old checkpoint would load clean into a different
# function with no `[ModelVersion] FATAL`.
#
# ⚠️ The pin governs NEW models only. On resume SB3 rebuilds the policy from the ZIP's own saved
# `policy_kwargs`, so a checkpoint written before this landed carries no `activation_fn` and still
# takes whatever the installed sb3-contrib defaults to. Those checkpoints are protected by the
# version pin on the code, not by this constant.
#
# Changing this is an ARCHITECTURE change: bump `ARCH_SIGNATURE` deliberately (see the
# model-versioning playbook in `src/agents/model/CLAUDE.md`), because nothing else will catch it.
POLICY_ACTIVATION_FN = th.nn.Tanh


class _NoFlatActionNet(th.nn.Module):
    """gen3_pointer_native_v1: a RAISING stub in `action_net`'s slot.

    The flat positional head does not exist in this generation — the pointer head (which needs the
    extractor's per-entity stash, not just `latent_pi`) is built in `_build` and called explicitly in
    `_get_action_dist_from_latent`. Any residual SB3 code path that calls `self.action_net(latent)`
    would be silently running a policy we deleted; make that a loud error, never an `Identity`
    fallback (which would emit `latent_pi` AS the logits — a garbage policy, not a crash)."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - defensive
        raise RuntimeError(
            "The flat action_net was removed (gen3_pointer_native_v1) — action logits come from "
            "Gen3DualHeadMaskablePolicy.pointer_head via _get_action_dist_from_latent. A code path "
            "calling action_net directly is running the deleted flat policy."
        )


class Gen3DualHeadMaskablePolicy(MaskableMultiInputActorCriticPolicy):
    """Maskable actor-critic policy whose features extractor yields a (pi, vf) tuple.

    ``self.extract_features(obs)`` returns ``(pi_features, vf_features)`` because the
    shared-extractor path simply returns whatever the extractor returns. Each consumer
    below unpacks that tuple and feeds ``mlp_extractor.forward_actor`` / ``forward_critic``
    the appropriate half. The value net and masking are inherited unchanged.

    **Pointer-native action head (gen3_pointer_native_v1).** There is NO flat positional
    action head in this generation: ``_build`` replaces SB3's ``action_net`` Linear with a
    raising stub and builds ``self.pointer_head`` (:class:`PointerNativeActionHead`), which
    scores each action from the token of the entity it selects — move logit k from the
    REQUEST-slot-k move token ⊕ its op cells, switch logit j from our-team token j ⊕ its
    incoming/OAX cells, struggle from the context — with ``latent_pi`` as the shared decision
    context. ``_get_action_dist_from_latent`` builds the distribution from those logits
    directly. Position-equivariant by construction: one shared scorer per entity, no logit
    row ever learns "slot j" positionally, and the sorted-vs-request ordering bug class is
    unrepresentable at the logits.

    **PopArt (opt-in via ``use_popart=True`` in ``policy_kwargs``).** When enabled, the value head
    (``value_net``) outputs *normalized* values and a :class:`~agents.model.popart.PopArtNormalizer`
    de-normalizes every value site below, so callers (GAE / advantages / bootstrapping) always see
    real-unit values while the PPO loss trains in normalized space (see
    ``agents/training/instrumented_ppo.py``). The normalizer is built **after** ``super().__init__``
    (which builds ``value_net``); its ``(mu, sigma)`` buffers ride the policy state_dict, so they
    save/restore across checkpoints. ``use_popart`` is version-checked (``ModelVersion``) — it cannot
    be toggled on a resumed model.
    """

    def _build(self, lr_schedule: Any) -> None:
        """gen3_pointer_native_v1: build SB3's stack, then REPLACE the flat action head.

        `super()._build` creates `action_net = Linear(latent_dim_pi, 11)` (the flat positional head),
        ortho-inits everything, and builds the optimizer. This generation has no flat head: swap in a
        raising stub (see `_NoFlatActionNet`), build the `PointerNativeActionHead` sized from the
        extractor's cell dims + `latent_dim_pi`, and REBUILD the optimizer — the one `super()` just
        made holds the deleted Linear's params and not the pointer head's (dead params in a param
        group would ride every checkpoint; missing ones would silently never train).

        Ordering note: the head is created AFTER `super()._build`'s ortho-init `apply`, so its
        zero-init scorers survive without the M1 guard — all logits are exactly 0 at step 0, i.e.
        the cold-start policy is uniform-over-legal (the correct fresh-run init)."""
        super()._build(lr_schedule)
        from agents.model.features_extractor import PointerNativeActionHead  # local: avoid import cycle
        fe = self.features_extractor
        self.action_net = _NoFlatActionNet()
        self.pointer_head = PointerNativeActionHead(
            # gen3_entity_move_seats_v1: move tokens are the REFINED E3 trunk seats (d_model-wide),
            # not the raw 32-dim PokemonEncoder tokens — the extractor owns the width.
            move_token_dim=fe.pointer_move_token_dim, d_model=D_MODEL,
            ctx_dim=self.mlp_extractor.latent_dim_pi,
            move_cell_dim=fe.pointer_move_cell_dim,
            switch_cell_dim=fe.pointer_switch_cell_dim,
        )
        # `Optimizer.__init__` is typed without `lr`; every concrete class SB3 selects takes it.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),  # type: ignore[call-arg]
                                              **self.optimizer_kwargs)

    def __init__(self, *args: Any, use_popart: bool = False, value_from_dist: bool = False,
                 **kwargs: Any) -> None:
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

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """gen3_pointer_native_v1: the action logits ARE the pointer head's scores.

        All three logit sites (`forward`, `evaluate_actions`, `get_distribution`) funnel through this
        method, and each calls `extract_features` immediately before — so the extractor's
        `last_pointer_inputs` stash is always fresh for THIS batch. That makes this the one correct
        scoring point; computing logits in any single caller would silently skip the other two (e.g.
        PPO's epoch recompute in `evaluate_actions` would then disagree with the rollout's `forward`,
        corrupting the ratio).

        `latent_pi` is the head's decision CONTEXT — the same policy-tower output the deleted flat
        head consumed, so the op block / beliefs / FiLM all condition every pointer score. Masking is
        applied by the callers on the returned distribution, downstream of these logits, exactly as
        before."""
        inputs = getattr(self.features_extractor, "last_pointer_inputs", None)
        if inputs is None:
            raise RuntimeError(
                "last_pointer_inputs is None — _get_action_dist_from_latent was called without a "
                "preceding extract_features on this policy's extractor (the pointer head has no "
                "flat fallback)."
            )
        tok_req, valid, team_tokens, move_cells, switch_cells = inputs
        if tok_req.shape[0] != latent_pi.shape[0]:
            raise RuntimeError(
                f"stale pointer stash: batch {tok_req.shape[0]} vs latent_pi {latent_pi.shape[0]} — "
                "the extractor forward and this latent are from different batches."
            )
        logits = self.pointer_head(latent_pi, tok_req, valid, team_tokens, move_cells, switch_cells)
        # Build through the public API so masking / log_prob / entropy all see these logits.
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
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
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
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
        self, obs: PyTorchObs, action_masks: Optional[np.ndarray] = None
    ) -> MaskableDistribution:
        pi_features, _ = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        _, vf_features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return self._critic_value(latent_vf)
