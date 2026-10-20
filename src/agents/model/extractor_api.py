"""`ExtractorApi` — the extractor's non-forward SURFACE: stash reads, and the three setters.

Split out of `features_extractor.py` 2026-08-23 (one responsibility per file). Everything here
is read or called from OUTSIDE a forward pass — the `last_*` stash properties every consumer
(the policy, `instrumented_ppo`'s aux losses, the prober, inference) reads, the pointer-cell
widths the policy sizes its head from, the SB3 ortho-init repair, and the belief-grad-mode
stamping. A base class rather than free functions so every body keeps its `self.` spelling and
mypy still resolves each attribute against the constructor that assigns it.
"""
from typing import Dict, Optional, Tuple

import torch

from agents.model.arch_constants import D_MODEL, MOVE_RESOLUTION_MOVE_DIM, MOVE_RESOLUTION_SWITCH_DIM
from agents.model.belief_heads import BELIEF_GRAD_MODES, _BELIEF_SUPERVISION_KEYS
from agents.model.extractor_build import ExtractorBuild
from agents.model.extractor_ctx import PointerInputs
from agents.model.flat_intent import FlatIntentInputs
from agents.model.hypothesis_set import HypothesisSet


def drop_child(owner: torch.nn.Module, name: str) -> None:
    """RETIRE child module ``name`` of ``owner``: remove it from ``_modules`` and leave a PLAIN ``None``
    attribute (gen3_x5_version_break_v1 part 2, the strict-load hole).

    ``owner.name = None`` on a REGISTERED child keeps the key in ``_modules`` with a ``None`` value, and
    torch's ``load_state_dict(strict=True)`` then treats ``name`` as a child that is merely absent:
    an UNEXPECTED key under ``<prefix>.name.`` is SWALLOWED, never reported (``_load_from_state_dict``
    skips any key whose first component names a child). So a checkpoint carrying a retired module's
    weights loaded "cleanly" into a model that has none. A plain attribute is not a child, so the same
    key is an unexpected key and a strict load refuses it. Readers keep their spelling (``owner.name``
    reads ``None``)."""
    if name in owner._modules:
        delattr(owner, name)
    setattr(owner, name, None)


class ExtractorApi(ExtractorBuild):
    """The non-forward surface of `Gen3FeaturesExtractor` — see that class."""

    def restore_identity_init(self) -> int:
        """Re-zero every Linear this extractor deliberately zero-initialised. Returns the count.

        WHY THIS EXISTS. SB3's `ActorCriticPolicy._build()` runs
        ``self.features_extractor.apply(partial(self.init_weights, gain=sqrt(2)))``
        (stable_baselines3/common/policies.py:617-631), and `init_weights` ORTHOGONALLY
        re-initialises EVERY `nn.Linear` it finds. `ortho_init` defaults True and nothing here
        overrides it — so every deliberate zero-init inside the extractor was silently destroyed the
        moment the policy was built, in every real training run.

        That silently falsified a documented invariant for every shipped zero-init feature —
        `prefuse_proj`, the edge-bias family maps and `film_pi`/`film_vf` all claim
        "zero-init ⇒ identity-at-init ⇒ ON starts byte-identical" — and,
        more insidiously, the belief heads' cold-start contract: `MoveBelief.move_head`,
        `SpreadBelief.{stat,nature,ev}_head` and `HPTypeBelief.type_head` are zero-init precisely so
        the cold-start posterior EQUALS the Smogon prior. Clobbered, they start at prior ⊕ noise.

        It went unnoticed because every unit test builds the module or a bare extractor DIRECTLY,
        where the zero-init survives; only SB3-wrapped construction destroys it.

        The set is captured by OBSERVATION at the end of `__init__` (any Linear whose weight is
        all-zero once construction finishes was zero-init'd on purpose) rather than by a hand-kept
        list, so a future zero-init module is protected automatically — the failure mode that
        produced this bug cannot recur by omission. Biases are not tracked: SB3's `init_weights`
        already zeroes every bias.
        """
        by_name = dict(self.named_modules())
        n = 0
        for name in self._identity_init_zeroed:
            mod = by_name.get(name)
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.zeros_(mod.weight)
                if mod.bias is not None:
                    torch.nn.init.zeros_(mod.bias)
                n += 1
        return n

    def retire_policy_tower(self) -> bool:
        """gen3_policy_readout_trunk_v1 (`--policy-readout trunk`, audit F2): drop the extractor half of
        the flat policy tower (`pre_proj_norm`, `projection`) — the policy context is `policy_query`'s
        read of the trunk instead. Returns True when it dropped them.

        Called by `Gen3DualHeadMaskablePolicy._build` AFTER SB3's orthogonal re-init and BEFORE the
        optimizer is built (the retire-after-re-init precedent): the two modules were
        constructed and re-initialised exactly as under `tower`, so the global RNG stream — every later
        module's initial bytes — is unchanged, while the retired modules hold no state_dict key, no
        optimizer slot and no forward use. `tower`: a no-op."""
        if self.policy_query is None or self.projection is None:
            return False
        # Typed as the modules for every `tower` reader; under `trunk` no forward path reads them (the
        # trunk branch of `forward_internal` / `forward` returns before), so None is safe.
        drop_child(self, "pre_proj_norm")
        drop_child(self, "projection")
        return True

    @property
    def policy_ctx_dim(self) -> int:
        """The width of `pi_features` — the pointer head's decision context before the SB3 actor branch.
        `tower`: `projection_dim` (the tower then maps it to `NET_ARCH[-1]`); `trunk`: `D_MODEL` (the
        state query's read, and the actor branch is the identity)."""
        return D_MODEL if self.policy_query is not None else int(self.projection_dim)
    def set_belief_grad_mode(self, mode: str) -> None:
        """Apply a belief-grad-mode at RUNTIME (the --allow-belief-grad-mode-change migration path).

        SB3's load reconstructs the extractor from the ZIP's saved policy_kwargs, so a resume that
        passes a different --belief-grad-mode would otherwise be a SILENT NO-OP (the 2026-07-21
        incident: the migration notice printed but the loaded extractor kept 'detached' —
        grad/*_norm_shared stayed exactly 0). The mode lives in THREE places (this attr,
        `_belief_detach`, and the `detach_read` flag stamped on each belief head); this is the ONE
        setter that updates them all — call it post-load on the resume path (a no-op when unchanged).

        gen3_belief_label_only_v1 widened that to FOUR places (`_belief_label_only` and the heads'
        `publish_detach` join the list) — which is exactly why the stamping is now a single
        `_stamp_belief_grad_flags()` shared with `__init__`, rather than a loop duplicated in two
        methods that a future mode could update in only one of."""
        if mode not in BELIEF_GRAD_MODES:
            raise ValueError(f"belief_grad_mode must be one of {'|'.join(BELIEF_GRAD_MODES)}, "
                             f"got {mode!r}")
        changed = mode != getattr(self, "belief_grad_mode", None)
        self.belief_grad_mode = mode
        self._belief_detach = (mode == "detached")
        self._belief_label_only = (mode == "label_only")
        self._stamp_belief_grad_flags()
        if changed:
            print(f"[Gen3FeaturesExtractor] belief_grad_mode APPLIED at runtime -> {mode!r} "
                  f"(detach_read={'on' if self._belief_detach else 'off'}, "
                  f"publish_detach={'on' if self._belief_label_only else 'off'} across the belief heads)")

    def set_oracle_reveal(self, spec: str) -> None:
        """Apply the RUN's resolved reveal mode to the live extractor after a load (gen3_reveal_provenance_v1).

        SB3's load rebuilds the extractor from the ZIP's saved ``policy_kwargs`` -- the PARENT's, on a fork -- so a
        diagnostic knock-in fork's ``oracle_reveal`` attribute read the parent's ``off`` while its observation (the env
        core's, declared from ``RustEnvDecl``) carried the reveal. The attribute is INERT in the forward (no module is
        built from it, no weight moves: ``__init__``) and ``metadata.json`` used to copy it, so the fork recorded the
        wrong mode. This is the ONE setter that fixes it; ``main.train.model_build`` calls it on every ``--model`` load
        through ``agents.model.resume_reapply`` (the same call that applies ``belief_grad_mode``).

        It deliberately touches ONLY the attribute, never the policy's constructor kwargs
        (``features_extractor_kwargs``): those are the forward-identity of the network (``forward_fingerprint``,
        ``main.h2h``'s ``EngineArch`` toggles), and a fork's forward IS its parent's -- rewriting them would split the
        fork from its plain control into two slot groups for no behavioural reason."""
        from utils.rust_env.protocol import oracle_reveal_label

        new = oracle_reveal_label(spec)
        changed = new != getattr(self, "oracle_reveal", None)
        self.oracle_reveal = new
        if changed:
            print(f"[Gen3FeaturesExtractor] oracle_reveal APPLIED at runtime -> {new!r} "
                  "(the loaded extractor carried the zip's mode; the observation itself is the env core's)")

    def set_tau_solve(self, solve: str) -> None:
        """Apply the RUN's resolved τ solve (`--tau-solve`, gen3_tau_fp32_alt_v1) to the live extractor after a load.

        SB3's load rebuilds the extractor from the ZIP's saved ``policy_kwargs`` -- the PARENT's on a fork, and none at
        all before config v160 -- so a fork that types (or inherits from ``model_config.json``) a solve its parent's zip
        does not carry would otherwise run the zip's. The ONE holder is ``hypothesis_builder.tau_solve`` (every
        fixed-size construction reads it); a network without a builder runs no τ and only checks the value. Like
        ``set_oracle_reveal`` it never touches ``features_extractor_kwargs``: the two solves are the same real function
        (<= 2 fp32 ulp of τ apart), so a fork that switched solve is not a different forward for slot grouping. The
        consequence, harmless for the same reason: a model the run REBUILDS from that zip's kwargs (not a deep copy of
        the learner) runs the zip's solve until the run's own fresh-built kwargs carry it."""
        from agents.model.hypothesis_set import check_tau_solve

        solve = check_tau_solve(solve)
        hb = getattr(self, "hypothesis_builder", None)
        if hb is None:
            return
        changed = solve != hb.tau_solve
        hb.tau_solve = solve
        if changed:
            print(f"[Gen3FeaturesExtractor] tau_solve APPLIED at runtime -> {solve!r} (the loaded zip carried another)")

    def _stamp_belief_grad_flags(self) -> None:
        """Push `belief_grad_mode` down onto the heads — the ONE place either per-head flag is set.

        `detach_read` (cut route B, the trunk read) goes on all four state-prediction heads.
        `publish_detach` (cut route C, the head's own reinjection) goes on the three that HAVE a
        reinjection; `BeliefHead` is a pure readout with nothing to publish, and the extractor-level
        `_publish_belief` covers every consumer that reads a stash rather than being handed the tensor
        by the head.
        """
        _item = getattr(self, "item_belief_head", None)
        for _bh in (self.move_belief, self.spread_belief, self.hp_type_belief_head,
                    _item, self.belief_head, getattr(self, "hypothesis_builder", None)):
            if _bh is not None:
                _bh.detach_read = self._belief_detach
        for _bh in (self.move_belief, self.spread_belief, self.hp_type_belief_head,
                    _item):
            if _bh is not None:
                _bh.publish_detach = self._belief_label_only

    # Read-only forwarders for the shared embedding tables — they are a model-level concept
    # and several tests/inspectors reach for them by name. Properties add no state_dict keys.
    @property
    def species_embedding(self) -> torch.nn.Embedding: return self.embeddings.species_embedding
    @property
    def move_embedding(self) -> torch.nn.Embedding: return self.embeddings.move_embedding
    @property
    def item_embedding(self) -> torch.nn.Embedding: return self.embeddings.item_embedding
    @property
    def ability_embedding(self) -> torch.nn.Embedding: return self.embeddings.ability_embedding
    @property
    def type_embedding(self) -> torch.nn.Embedding: return self.embeddings.type_embedding
    @property
    def hp_type_idx_map(self) -> torch.Tensor: return self.embeddings.hp_type_idx_map

    # gen3_extractor_stashes_v1 — the READ surface over the typed stash container (see
    # ExtractorStashes). Every consumer keeps its historical `last_*` spelling — the policy's
    # pointer head + dist critic, instrumented_ppo's aux losses, the prober, inference — and a
    # stray WRITE to any of these names raises AttributeError (no setter) instead of silently
    # forking the state. Writes go through `self.stash.<field>` only.
    @property
    def last_pointer_inputs(self) -> Optional[PointerInputs]: return self.stash.pointer_inputs
    @property
    def last_flat_intent_logits(self) -> Optional[torch.Tensor]: return self.stash.flat_intent_logits
    @property
    def last_flat_intent(self) -> Optional["FlatIntentInputs"]: return self.stash.flat_intent
    @property
    def last_belief_logits(self) -> Optional[Dict[str, torch.Tensor]]: return self.stash.belief_logits
    @property
    def last_hypothesis(self) -> Optional[HypothesisSet]: return self.stash.hypothesis
    @property
    def last_opp_believed_mask(self) -> Optional[torch.Tensor]: return self.stash.opp_believed_mask
    @property
    def last_opp_active_local(self) -> Optional[torch.Tensor]: return self.stash.opp_active_local
    @property
    def last_move_belief_logits(self) -> Optional[torch.Tensor]: return self.stash.move_belief_logits
    @property
    def last_move_latent_table(self) -> Optional[torch.Tensor]: return self.stash.move_latent_table
    @property
    def last_spread_belief(self) -> Optional[torch.Tensor]: return self.stash.spread_belief
    @property
    def last_spread_nature_logits(self) -> Optional[torch.Tensor]: return self.stash.spread_nature_logits
    @property
    def last_spread_ev(self) -> Optional[torch.Tensor]: return self.stash.spread_ev
    @property
    def last_item_logits(self) -> Optional[torch.Tensor]: return self.stash.item_logits
    @property
    def last_hp_type_logits(self) -> Optional[torch.Tensor]: return self.stash.hp_type_logits
    @property
    def last_damage_block(self) -> Optional[torch.Tensor]: return self.stash.damage_block
    @property
    def last_value_pooled(self) -> Optional[torch.Tensor]: return self.stash.value_pooled

    @property
    def last_trunk_tokens(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]: return self.stash.trunk_tokens

    @property
    def last_value_cls(self) -> Optional[torch.Tensor]: return self.stash.value_cls

    @property
    def last_features_out(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]: return self.stash.features_out
    @property
    def last_win_prob_logits(self) -> Optional[torch.Tensor]: return self.stash.win_prob_logits
    # Private per-forward hand-offs with external test readers keep their names too (same
    # read-only discipline; the T0->T1/T2 contract is documented on the dataclass fields).
    @property
    def _entity_latent_table(self) -> Optional[torch.Tensor]: return self.stash.entity_latent_table
    @property
    def _belief_supervision(self) -> Dict[str, Optional[torch.Tensor]]:
        return self.stash.belief_supervision

    # gen3_pointer_native_v1: the pointer head's per-action cell widths — the policy sizes the head's
    # Linears from these at build time (0 when the source op block is off, so a missing block narrows
    # the Linear instead of silently feeding zeros at a learned weight).
    @property
    def pointer_move_token_dim(self) -> int:
        # gen3_entity_move_seats_v1: the head reads the REFINED E3 seats (d_model), no longer the raw
        # 32-dim PokemonEncoder move tokens.
        return D_MODEL

    @property
    def pointer_move_cell_dim(self) -> int:
        # The op's move cell, then the move-resolution family's move block (gen3_move_resolution_v1) when it is
        # built (the policy sizes the pointer move scorer's in_features from this at build time).
        base = self.damage_op.pointer_move_cell_dim if self.damage_op is not None else 0
        return base + (MOVE_RESOLUTION_MOVE_DIM if self.move_resolution_cell is not None else 0)
    @property
    def pointer_switch_cell_dim(self) -> int:
        base = self.damage_op.pointer_switch_cell_dim if self.damage_op is not None else 0
        # gen3_move_resolution_v1: the move-resolution family's switch block.
        base += MOVE_RESOLUTION_SWITCH_DIM if self.move_resolution_cell is not None else 0
        # gen3_static_recovery_v1 (built with the op): the LAST block, the target's entry-hazard cost.
        return base + self.pointer_switch_extra_dim

    @property
    def pointer_switch_extra_dim(self) -> int:
        """The width of the switch cell's TRAILING block that the pointer head reads through its own zero-init
        projection (`gen3_static_recovery_v1`: the entry-hazard cost, built with the op); 0 without the op."""
        from agents.model.static_facts import SWITCH_HAZARD_DIM
        return SWITCH_HAZARD_DIM if getattr(self, "switch_hazard_cost", False) else 0

    def _publish_belief(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Hand a belief output to the FORWARD (reinject / the op / the edge cells / the seats / the
        pointer stash / the prober). Under `label_only` this is a STOP-GRAD copy — gen3_belief_label_only_v1.

        The cut lives at the ONE publish boundary per head rather than at each consumer, and that is the
        whole design: `last_move_belief_logits` alone has eleven downstream readers, so a per-consumer
        rule would be one forgotten site away from silently reopening the route. Publishing instead
        isolates a consumer added TOMORROW by construction.

        Returns the identical object under shaping/detached (and for `None`), so those modes stay
        byte-identical — `detach()` never changes a value, only the graph, so even under `label_only` the
        FORWARD is bit-identical and only the backward differs.
        """
        return t.detach() if (self._belief_label_only and t is not None) else t

    def belief_supervision(self, name: str) -> Optional[torch.Tensor]:
        """The LIVE (graph-carrying) belief output `name`, for the SUPERVISED aux losses ONLY.

        ⚠️ **An aux loss MUST read its target through here, never off the `last_*` attribute.** Under
        `label_only` that attribute is a stop-grad publication (`_publish_belief`), so a loss reading it
        would train nothing at all — and would do so SILENTLY, since the loss value and every metric
        derived from it look exactly the same. The gate test that would catch it is
        `belief_label_only_gate_test.py::test_every_belief_loss_still_trains_its_head`.

        Returns the identical object the `last_*` stash holds under shaping/detached, and `None` for a
        head that is not built (the caller's existing `is None` guards are unchanged).
        """
        if name not in _BELIEF_SUPERVISION_KEYS:
            raise KeyError(
                f"unknown belief supervision key {name!r}; expected one of "
                f"{sorted(_BELIEF_SUPERVISION_KEYS)}. Add the key to _BELIEF_SUPERVISION_KEYS and "
                "register the LIVE tensor where the head's stash is published."
            )
        return self.stash.belief_supervision.get(name)
