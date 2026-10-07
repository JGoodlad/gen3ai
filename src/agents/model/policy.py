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

from typing import TYPE_CHECKING, Any, Optional, Tuple, cast

import numpy as np
import torch as th

from stable_baselines3.common.type_aliases import PyTorchObs
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

from agents.model import masked_categorical as _mc
from agents.model.arch_constants import D_MODEL, POINTER_HIDDEN, TRUNK_POINTER_HIDDEN
from agents.model.critic_mode import CRITIC_UNRECORDED, CRITIC_MODES, is_winprob

if TYPE_CHECKING:
    # SB3 types `features_extractor` as `BaseFeaturesExtractor` (ours is duck-typed, not a subclass);
    # the read sites cast to the real class so torch >= 2.8's `Module.__getattr__` (`Tensor | Module`)
    # does not type its attributes. Typing only.
    from agents.model.features_extractor import Gen3FeaturesExtractor


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

    **The critic** is ``critic`` in ``policy_kwargs`` (`critic_mode`): ``winprob`` reads
    ``sigmoid(win_head logit)``, ``shaped`` (what an absent record means) reads the scalar
    ``value_net``. PopArt and the distributional value head were DELETED (deletion pass L1; a
    checkpoint's pickled ``use_popart`` / ``value_from_dist`` are stripped by
    ``snapshot._DEAD_POLICY_KWARGS_JUDGED``).
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
        fe = cast("Gen3FeaturesExtractor", self.features_extractor)
        # X5 U4 (fixed_mass): retire α / β AFTER the ortho-init draws above and BEFORE the optimizer
        # below (`ExtractorApi.retire_superseded_intent_heads`); blob: a no-op.
        if hasattr(fe, "retire_superseded_intent_heads"):
            fe.retire_superseded_intent_heads()
        # gen3_policy_readout_trunk_v1 (`--policy-readout trunk`, audit F2): RETIRE the flat policy tower
        # — the extractor's `pre_proj_norm` / `projection` and SB3's `mlp_extractor.policy_net` — AFTER
        # the ortho-init draws above (they drew exactly as under `tower`, so no surviving module's initial
        # bytes move) and BEFORE the optimizer below. The actor branch becomes the EMPTY Sequential (the
        # identity), so every `forward_actor` caller — this policy, the T2 `DecisionModule`, the compiled
        # trainer — passes the state query's read straight to the pointer head unchanged. `tower`: a no-op.
        trunk = bool(getattr(fe, "retire_policy_tower", None) and fe.retire_policy_tower())
        pointer_hidden = POINTER_HIDDEN
        if trunk:
            self.mlp_extractor.policy_net = th.nn.Sequential()
            self.mlp_extractor.latent_dim_pi = fe.policy_ctx_dim
            pointer_hidden = TRUNK_POINTER_HIDDEN
        self.action_net = _NoFlatActionNet()
        self.pointer_head = PointerNativeActionHead(
            # gen3_entity_move_seats_v1: move tokens are the REFINED E3 trunk seats (d_model-wide),
            # not the raw 32-dim PokemonEncoder tokens — the extractor owns the width.
            move_token_dim=fe.pointer_move_token_dim, d_model=D_MODEL,
            ctx_dim=self.mlp_extractor.latent_dim_pi,
            move_cell_dim=fe.pointer_move_cell_dim,
            switch_cell_dim=fe.pointer_switch_cell_dim,
            hidden=pointer_hidden,
        )
        # `Optimizer.__init__` is typed without `lr`; every concrete class SB3 selects takes it.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),  # type: ignore[call-arg]
                                              **self.optimizer_kwargs)

    def __init__(self, *args: Any, critic: str = CRITIC_UNRECORDED, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # gen3_winprob_critic_mode_v1: WHICH readout is the value function. 'shaped' is every
        # generation through gen-16 — `value_net` / E[Z] in raw shaped-return units — and what an
        # ABSENT kwarg means (`CRITIC_UNRECORDED`: a pre-v109 checkpoint's saved policy_kwargs never
        # carried the key), NOT the bare-argv `CRITIC_DEFAULT`.
        # 'winprob' routes `_critic_value` to sigmoid(win_head logit) ∈ [0,1]; see critic_mode.py.
        if str(critic) not in CRITIC_MODES:
            raise ValueError(f"unknown critic {critic!r} (want one of {CRITIC_MODES})")
        self._critic_mode = str(critic)

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

        # gen3_ridealong_heads_v1 (v126): the DETACHED RIDE-ALONG heads (V ensemble, RND, A, B),
        # declared by the extractor's `ridealong_*` kwargs and built HERE — after `_build` made
        # `self.optimizer` (so they are in none of its param groups: the learner steps them with
        # their own) and after the ortho-init `apply` (so it cannot draw from the global RNG over
        # their Linears). Built from a PRIVATE seed inside `fork_rng`, so the global stream is
        # untouched; never called by `forward`. None when every flag is off (no state_dict key).
        from agents.model.ridealong_heads import build_ridealong  # local: keep import light
        _obs_space = getattr(self.observation_space, "spaces", {}).get("observation")
        _obs_dim = int(_obs_space.shape[0]) if _obs_space is not None else 0
        self.ridealong = build_ridealong(self.features_extractor, obs_dim=_obs_dim)

    def _critic_value(self, latent_vf: th.Tensor) -> th.Tensor:
        """The critic value used by GAE / bootstrap / deployment.

        gen3_winprob_critic_mode_v1: under ``critic='winprob'`` the value IS the win-prob head's
        probability — ``sigmoid(logit) ∈ [0,1]``. There is NO fallback: under this mode
        ``value_net`` is in no loss graph, so quietly returning it would be a critic the training
        loop believes in and nothing updates (the v89 orphaned-route class). Under ``shaped`` the
        value is the scalar ``value_net``."""
        # `getattr` with the default rather than `self._critic_mode`: this method is called on
        # policy-shaped STUBS and could be reached on a policy restored
        # from a pre-v109 checkpoint whose saved `policy_kwargs` never carried the key. An
        # absent field means the historical critic — the same read every other consumer does.
        if is_winprob(getattr(self, "_critic_mode", CRITIC_UNRECORDED)):
            fe = cast("Gen3FeaturesExtractor", self.features_extractor)
            logits = getattr(fe, "last_win_prob_logits", None)
            if getattr(fe, "win_head", None) is None or logits is None:
                raise RuntimeError(
                    "critic='winprob' but "
                    + ("the extractor has no win_head (--win-prob-mode is 'none')"
                       if getattr(fe, "win_head", None) is None
                       else "last_win_prob_logits was not stashed by the preceding forward")
                    + " — the scalar value_net is in NO loss graph under this critic, so falling "
                    "back to it would be a silently-wrong critic (the v89 orphaned-route class). "
                    "Check that extract_features ran on THIS policy's extractor before the critic "
                    "read, and that --win-prob-mode is read_only or shaping.")
            if logits.shape[0] != latent_vf.shape[0]:
                raise RuntimeError(
                    f"stale win-prob stash: logits batch {logits.shape[0]} vs latent_vf "
                    f"{latent_vf.shape[0]} — the extractor forward and this critic read are "
                    "from different batches.")
            # [B,1] like `value_net(latent_vf)`, so every caller's `.flatten()` / `.squeeze(-1)`
            # is unchanged. Probability units are the only currency here.
            return th.sigmoid(logits.reshape(-1, 1))
        return cast(th.Tensor, self.value_net(latent_vf))

    def _pointer_logits(self, latent_pi: th.Tensor) -> th.Tensor:
        """gen3_pointer_native_v1: the action logits ARE the pointer head's scores.

        All three logit sites (`forward`, `evaluate_actions`, `get_distribution`) funnel through this
        method, and each calls `extract_features` immediately before — so the extractor's
        `last_pointer_inputs` stash is always fresh for THIS batch. That makes this the one correct
        scoring point; computing logits in any single caller would silently skip the other two (e.g.
        PPO's epoch recompute in `evaluate_actions` would then disagree with the rollout's `forward`,
        corrupting the ratio).

        `latent_pi` is the head's decision CONTEXT — the same policy-tower output the deleted flat
        head consumed, so the op block / beliefs / FiLM all condition every pointer score. Masking is
        applied by the callers, downstream of these logits, exactly as before."""
        inputs = cast("Gen3FeaturesExtractor", self.features_extractor).last_pointer_inputs
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
        logits: th.Tensor = self.pointer_head(latent_pi, tok_req, valid, team_tokens, move_cells,
                                              switch_cells)
        return logits

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """sb3's distribution object over `_pointer_logits` — for `get_distribution`'s callers
        (`predict`, the offline readers), which use the object API. The two
        hot paths (`forward`, `evaluate_actions`) use the FUNCTIONAL masking below instead."""
        # Build through the public API so masking / log_prob / entropy all see these logits.
        return self.action_dist.proba_distribution(action_logits=self._pointer_logits(latent_pi))

    def masked_logp(self, latent_pi: th.Tensor, action_masks: Any) -> th.Tensor:
        """gen3_functional_masking_v1 (M5 Lane K8): the masked, normalised log-probabilities sb3's
        `MaskableCategorical` would end with — as plain tensor ops (`agents.model.masked_categorical`,
        bit-identical, no `__dict__.pop`, no `torch.distributions` object, no validation host read),
        so the learner's micro-step can trace as ONE region."""
        n_actions = int(self.action_dist.action_dim)  # type: ignore[attr-defined]  # the categorical's
        return _mc.masked_logits(self._pointer_logits(latent_pi), action_masks, n_actions)

    def rollout_core(self, obs: Any, action_masks: Any) -> Tuple[th.Tensor, th.Tensor]:
        """The rollout forward as TENSORS — ``(values, masked_logp)`` — everything `forward` computes
        before the action draw (functional masking, no distribution object). EAGER only: the learner
        process compiles nothing here (the compiled rollout region R0 was deleted, P10-E — the Rust
        collector's rollout forward is the T2 inference service's own `DecisionModule`); this is the
        seam `forward` and `compile_regions.weights_regime`'s freshness read share."""
        pi_features, vf_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self._critic_value(latent_vf)
        return values, self.masked_logp(latent_pi, action_masks)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # The action draw is the same `multinomial` call as sb3's distribution object (the same RNG
        # stream), pinned bit-for-bit (`masked_categorical_test`). Eager: no compiled rollout region.
        values, logp = self.rollout_core(obs, action_masks)
        actions = _mc.mode(logp) if deterministic else _mc.sample(logp)
        log_prob = _mc.log_prob(logp, actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions_functional(
        self, obs: th.Tensor, actions: th.Tensor, action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """`evaluate_actions` as TENSORS only — ``(values, log_prob, entropy, masked_logp,
        masks_bool)`` — for the learner micro-step (K8's region R1), which must not build the
        distribution stash object inside a compiled region; the caller sets
        `_last_pi_distribution` from the returned tensors."""
        pi_features, vf_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # gen3_functional_masking_v1: functional masked logits (bit-identical to sb3's object).
        logp = self.masked_logp(latent_pi, action_masks)
        masks_bool = _mc.mask_bool(action_masks, logp)
        log_prob = _mc.log_prob(logp, actions)
        values = self._critic_value(latent_vf)
        return values, log_prob, _mc.entropy(logp, masks_bool), logp, masks_bool

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        values, log_prob, entropy, logp, masks_bool = self.evaluate_actions_functional(
            obs, actions, action_masks)
        # Stash the (masked) pi so the ride-along heads REUSE this forward instead of a redundant
        # second one. `MaskedPi` answers `.distribution.logits` / `.distribution.probs` exactly as the
        # sb3 object did (over LEGAL actions the logits are unchanged; illegal actions contribute
        # exactly 0 either way).
        self._last_pi_distribution = _mc.MaskedPi(logp, masks_bool)
        return values, log_prob, entropy

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
