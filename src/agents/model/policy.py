"""Dual-head maskable policy for the value-dedicated CLS readout (H4 / Option C).

`Gen3FeaturesExtractor.forward` returns a ``(pi_features, value_pooled)`` tuple: the
transformer body is shared, but the actor and critic read it through independent CLS
pools. Stock SB3 policies assume the features extractor returns a single tensor, so this
policy overrides the methods that consume features: the policy half goes through the actor
tower (``mlp_extractor.forward_actor``) to the pointer head; the value half is the win-prob
head's input, and the critic is that head (``_critic_value``) — there is no critic tower.

Design note — we deliberately keep ``share_features_extractor=True`` so SB3 builds exactly
ONE features-extractor instance (one transformer body). The "sharing" is real at the body
level; the split happens inside the extractor's two readouts and is preserved here by
unpacking the tuple. We do NOT use ``share_features_extractor=False`` because that would
make SB3 instantiate a second full body (Option A, ~2× compute) — not what Option C wants.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Optional, Tuple, cast

import numpy as np
import torch as th

from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

from agents.model import masked_categorical as _mc
from agents.model.arch_constants import D_MODEL, POINTER_HIDDEN, TRUNK_POINTER_HIDDEN
from agents.model.critic_mode import CRITIC_UNRECORDED, CRITIC_WINPROB, is_winprob

if TYPE_CHECKING:
    # SB3 types `features_extractor` as `BaseFeaturesExtractor` (ours is duck-typed, not a subclass);
    # the read sites cast to the real class so torch >= 2.8's `Module.__getattr__` (`Tensor | Module`)
    # does not type its attributes. Typing only.
    from agents.model.features_extractor import Gen3FeaturesExtractor


# gen3_policy_activation_pin_v1: the nonlinearity of the SB3 `mlp_extractor` tower
# (`net_arch = [512, 512]`, the ACTOR branch — the critic has no tower since the version break, F1).
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


class _NoValueNet(th.nn.Module):
    """gen3_x5_version_break_v1 part 2 (architecture audit F1): a RAISING stub in `value_net`'s slot.

    The scalar value head (and the extractor projection + SB3 critic tower that fed it) is DELETED: the
    win-prob head is the only critic, read by `_critic_value`. An SB3 path that still called
    `self.value_net(latent)` would be running a critic that does not exist; make that a loud error,
    never an `Identity` (which would return the latent AS the value)."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - defensive
        raise RuntimeError(
            "The scalar value_net was deleted at the version break (config v144, architecture audit F1) — "
            "the critic is sigmoid(win_head logit), read by Gen3DualHeadMaskablePolicy._critic_value. A "
            "code path calling value_net directly is running a critic that no longer exists.")


class Gen3DualHeadMaskablePolicy(MaskableMultiInputActorCriticPolicy):
    """Maskable actor-critic policy whose features extractor yields a (pi, vf) tuple.

    ``self.extract_features(obs)`` returns ``(pi_features, value_pooled)`` because the
    shared-extractor path simply returns whatever the extractor returns. Each consumer
    below unpacks that tuple, feeds the policy half to ``mlp_extractor.forward_actor`` and
    reads the critic through ``_critic_value`` (the win-prob head; no critic tower, F1).

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

    **The critic** is the win-prob head: ``_critic_value`` reads ``sigmoid(win_head logit)``, stashed by
    the extractor forward. ``critic`` in ``policy_kwargs`` must be ``winprob``: the scalar ``value_net``
    the ``shaped`` critic read was DELETED at the version break (config v144, architecture audit F1)
    together with the extractor's ``value_pre_norm`` / ``value_projection`` and SB3's critic tower
    (``mlp_extractor.value_net``), so the extractor's value half (``value_pooled``) reaches the critic
    read untouched and ``value_net`` is a raising stub. PopArt and the distributional value head were
    DELETED (deletion pass L1; a checkpoint's pickled ``use_popart`` / ``value_from_dist`` are stripped
    by ``snapshot._DEAD_POLICY_KWARGS_JUDGED``).
    """

    def _build_mlp_extractor(self) -> None:
        """gen3_x5_version_break_v1 part 2 (architecture audit F1): the ACTOR tower only.

        ``net_arch`` names the actor's widths (a list, or a dict's ``pi``); the critic branch is the EMPTY
        Sequential (SB3's ``vf=[]``), so ``forward_critic`` is the identity on ``value_pooled`` and the deleted
        512 → 512 → 512 critic tower is never built — it held 525,312 parameters no loss read."""
        na = self.net_arch
        pi_arch = list(na.get("pi", [])) if isinstance(na, dict) else list(na)
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=dict(pi=pi_arch, vf=[]),
                                          activation_fn=self.activation_fn, device=self.device)
        # The critic branch is the identity on the extractor's value half, `value_pooled` [B, D_MODEL].
        self.mlp_extractor.latent_dim_vf = D_MODEL

    def _build(self, lr_schedule: Any) -> None:
        """Build the policy's OWN stack: the actor tower, the pointer head, the optimizer.

        SB3's ``_build`` is NOT called. It would construct ``action_net = Linear(latent_dim_pi, 11)`` (the
        flat positional head this generation does not have, gen3_pointer_native_v1) and ``value_net =
        Linear(latent_dim_vf, 1)`` (the scalar critic deleted at the version break, F1) only for this
        method to discard them. Here neither is built: both slots hold RAISING stubs (`_NoFlatActionNet`,
        `_NoValueNet`). What SB3's does that this keeps, in its order: the mlp extractor (actor only,
        `_build_mlp_extractor`), the orthogonal re-init of the extractor then the mlp extractor (gain √2,
        SB3's), then the retire hooks (after the re-init's draws, before the optimizer), the
        `PointerNativeActionHead` (sized from the extractor's cell dims + `latent_dim_pi`) and the
        optimizer over every surviving parameter.

        Ordering note: the head is created AFTER the ortho-init `apply`, so its zero-init scorers survive
        without the M1 guard — all logits are exactly 0 at step 0, i.e. the cold-start policy is
        uniform-over-legal (the correct fresh-run init)."""
        if not self.share_features_extractor:
            raise ValueError("Gen3DualHeadMaskablePolicy shares ONE features extractor between actor and "
                             "critic (share_features_extractor=True): its two readouts split inside it.")
        self._build_mlp_extractor()
        self.action_net = _NoFlatActionNet()
        self.value_net = _NoValueNet()
        if self.ortho_init:
            for module in (self.features_extractor, self.mlp_extractor):
                module.apply(partial(self.init_weights, gain=np.sqrt(2)))
        from agents.model.features_extractor import PointerNativeActionHead  # local: avoid import cycle
        fe = cast("Gen3FeaturesExtractor", self.features_extractor)
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
        # gen3_move_resolution_v1 (v141, `--move-resolution on`): retire the seven per-action blocks the
        # move-resolution family replaces, at the same point and for the same reason (`off`: a no-op).
        if hasattr(fe, "retire_superseded_action_cells"):
            fe.retire_superseded_action_cells()
        # gen3_value_threat_inject_off_v1 (v142, `--value-threat-inject off`, audit F10): retire the critic's
        # token-content threat projection at the same point and for the same reason (ON: a no-op).
        if hasattr(fe, "retire_value_threat_inject"):
            fe.retire_value_threat_inject()
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
        # gen3_winprob_critic_mode_v1: WHICH readout is the value function — since the version break
        # (config v144, architecture audit F1) only 'winprob': `_critic_value` is sigmoid(win_head logit)
        # ∈ [0,1]. 'shaped' (and an ABSENT kwarg, `CRITIC_UNRECORDED`: a pre-v109 checkpoint's saved
        # policy_kwargs never carried the key) read the scalar `value_net`, which no longer exists; such a
        # checkpoint is below MIGRATION_FLOOR anyway and runs PINNED to its own commit.
        if not is_winprob(critic):
            raise ValueError(
                f"critic={critic!r}: the only critic is {CRITIC_WINPROB!r} (sigmoid(win_head logit)). The "
                "scalar value_net a 'shaped' critic read was DELETED at the version break (config v144, "
                "architecture audit F1); an absent record means 'shaped' (pre-v109). Run such a checkpoint "
                "PINNED to its own commit.")
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

    def _critic_value(self, vf_features: th.Tensor) -> th.Tensor:
        """The critic value used by GAE / bootstrap / deployment: the win-prob head's probability,
        ``sigmoid(logit) ∈ [0,1]``, ``[B,1]``, read off the logits the preceding extractor forward stashed.

        ``vf_features`` is the extractor's value half (``value_pooled``, the head's own input) and is read
        for its BATCH only — the guard that the stash belongs to this forward. There is NO fallback and no
        second critic: the scalar ``value_net`` was deleted at the version break (config v144, audit F1)."""
        # `getattr` with the default rather than `self._critic_mode`: this method is called on
        # policy-shaped STUBS. A recorded non-winprob critic cannot be built (`__init__` refuses it).
        mode = getattr(self, "_critic_mode", CRITIC_UNRECORDED)
        if not is_winprob(mode):
            raise RuntimeError(f"critic={mode!r} has no readout: the scalar value_net was deleted at the "
                               "version break (config v144, architecture audit F1).")
        fe = cast("Gen3FeaturesExtractor", self.features_extractor)
        logits = getattr(fe, "last_win_prob_logits", None)
        if getattr(fe, "win_head", None) is None or logits is None:
            raise RuntimeError(
                "critic='winprob' but "
                + ("the extractor has no win_head (--win-prob-mode is 'none')"
                   if getattr(fe, "win_head", None) is None
                   else "last_win_prob_logits was not stashed by the preceding forward")
                + " — the win-prob head is the only critic. Check that extract_features ran on THIS "
                "policy's extractor before the critic read, and that --win-prob-mode is read_only or "
                "shaping.")
        if logits.shape[0] != vf_features.shape[0]:
            raise RuntimeError(
                f"stale win-prob stash: logits batch {logits.shape[0]} vs value features "
                f"{vf_features.shape[0]} — the extractor forward and this critic read are "
                "from different batches.")
        # [B,1], so every caller's `.flatten()` / `.squeeze(-1)` reads one value per row. Probability units
        # are the only currency here.
        return th.sigmoid(logits.reshape(-1, 1))

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
        values = self._critic_value(vf_features)
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
        # gen3_functional_masking_v1: functional masked logits (bit-identical to sb3's object).
        logp = self.masked_logp(latent_pi, action_masks)
        masks_bool = _mc.mask_bool(action_masks, logp)
        log_prob = _mc.log_prob(logp, actions)
        values = self._critic_value(vf_features)
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
        return self._critic_value(vf_features)
