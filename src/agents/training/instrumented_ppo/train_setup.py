"""`TrainSetup` — everything `train()` resolves BEFORE the epoch loop, and nothing else.

Three methods, each returning what the fold then reads: the opponent-intent label alignment (a
buffer edit with no result), the FOLD FLAGS (which terms are live this call), and the PROBE SETUP
(the once-per-`train()` diagnostics and the gradient sampler).

None of this is the fold. Each flag is computed once and read by the `if <x>_on:` guard of the term
it owns, so the sequence in `ppo.train()` stays straight-line source with its guards inline — the
property `instrumented_ppo_hub_contract_test` reads. The two result containers are `NamedTuple`s
carrying the SAME names the fold uses, so `train()` unpacks them back into the locals the loop was
written against and the loop body is unchanged.
"""
from typing import Any, NamedTuple

from agents.model.critic_mode import is_winprob
from agents.training.fork_arm import PG_MASK_KEY as FORK_PG_MASK_KEY
from agents.training.grad_balance import shared_trunk_parameters
from agents.training.instrumented_ppo.diagnostics_cadence import DiagnosticsPlan, mark_ran, plan_for
from agents.training.instrumented_ppo.noise_scale_terms import (
    NULL_TAGGER,
    PerTermNoiseSampler,
    per_term_enabled,
)
from agents.training.instrumented_ppo.signal_metrics import advantage_density_metrics


class FoldFlags(NamedTuple):
    """WHICH terms this `train()` call folds. Every field is read by exactly the guard of the term
    it names; the reasoning for each one is the comment above its computation below."""
    belief_aux_on: Any
    move_belief_on: Any
    move_latent_on: Any
    spread_belief_on: Any
    hp_type_belief_on: Any
    item_belief_on: Any
    critic_winprob: Any
    win_prob_on: Any
    scaffolding_on: Any
    policy_grad_coef: Any
    fork_pg_mask_on: Any


class ProbeSetup(NamedTuple):
    """The once-per-`train()` diagnostics state. `ns_terms` is `_ns_terms` in `train()`; a
    `NamedTuple` field cannot start with an underscore, and the local name is what the fold's
    `_ntg.add(...)` seam is written against."""
    shared_trunk: Any
    grad_balance: Any
    rank_metrics: Any
    edge_metrics: Any
    cell_metrics: Any
    grad_norms: Any
    capacity: Any
    capacity_metrics: Any
    signal_metrics: Any
    accum: Any
    noise_g_small_sq: Any
    noise_g_big_sq: Any
    ns_terms: Any
    diag: DiagnosticsPlan


class TrainSetup:
    """Mixin: the pre-loop half of `train()`."""

    def _align_opp_intent_labels(self) -> None:
        # +OPPONENT INTENT (gen3_opp_intent_v1): ALIGN the labels to the predictions, ONCE, BEFORE
        # `get()` flattens and shuffles. The env emits, at buffer row i, what the opponent did at
        # decision i-1 (their turn-t action is only observable while building the obs for t+1), so
        # row i's own label sits at row i+1 of the same env column. Shifting here — while the
        # [n_steps, n_envs] structure and `episode_starts` still exist — is the only place the
        # episode-boundary drop is even expressible; after the shuffle the adjacency is gone.
        # Idempotent per rollout: collect_rollouts refills these keys every time.
        if getattr(self, "opp_intent_coef", 0.0) > 0.0:
            _obs_buf = getattr(self.rollout_buffer, "observations", None)
            if isinstance(_obs_buf, dict) and "opp_action_kind" in _obs_buf:
                from agents.training.opp_intent_labels import (KIND_UNKNOWN, SWITCH_SLOT_NONE,
                                                               align_labels_to_predictions)
                _starts = self.rollout_buffer.episode_starts
                # EVERY one-ahead intent key must be shifted, including `opp_switch_species`.
                # It was omitted originally, so beta's CONTENT-ADDRESSED target read the species of
                # decision t-1 against the kind/slot of decision t. That is not merely wrong, it is
                # INVISIBLE: on most rows the stale species is 0 -> resolve_believed_slot_by_content
                # returns INTENT_IGNORE and the path silently no-ops, which reads exactly like the
                # documented "the belief is too cold to clear the floor" case below. Two consecutive
                # switch-ins is the one shape where it resolves — to the PREVIOUS switch-in's slot.
                for _k, _fill in (("opp_action_kind", KIND_UNKNOWN), ("opp_action_num", 0),
                                  ("opp_switch_slot", SWITCH_SLOT_NONE),
                                  ("opp_switch_species", 0),
                                  # `opp_class` is CONSTANT within an episode, so the shift is a
                                  # semantic no-op — included anyway so every intent label is
                                  # row-aligned by the same rule. A reader should never have to
                                  # remember which of these keys was shifted and which was not;
                                  # that asymmetry is what produced the bug documented above.
                                  ("opp_class", 0)):
                    _obs_buf[_k] = align_labels_to_predictions(_obs_buf[_k], _starts, _fill)

    def _behaviour_probe(self) -> None:
        # +K9(b) BEHAVIOUR-POLICY CONSISTENCY + the STALENESS probe (M5 Lane G): one learner forward
        # on one micro-batch, BEFORE any optimizer step and while the buffer is still [n_steps, n_envs]
        # (the rows' policy versions are aligned to that layout). Called by `train()` whenever
        # `learner_gates.behaviour_gate_mode` == "probe" (every check but off); a buffer with no version
        # record is judged as every row current. See `rust_rollout/consistency.py`.
        if str(getattr(self, "behaviour_check", "off") or "off") != "off":
            from agents.training.rust_rollout.consistency import behaviour_probe
            behaviour_probe(self)

    def _resolve_fold_flags(self) -> FoldFlags:
        """WHICH terms are live this call. Pure resolution."""
        # Compute once: the aux path is fully skipped when off → loss stays byte-identical to upstream.
        belief_aux_on = self.opp_belief_aux_coef > 0.0
        move_belief_on = self.move_belief_coef > 0.0  # +MOVE-BELIEF reinjection-head supervised loss
        move_latent_on = self.move_belief_latent_coef > 0.0  # +MOVE-LATENT grading (gen3_unified_move_system_v1)
        spread_belief_on = self.spread_belief_coef > 0.0  # +SPREAD-belief supervision (gen3_unified_spread_belief_v1)
        hp_type_belief_on = self.hp_type_belief_coef > 0.0  # +HP-TYPE belief CE (gen3_opp_hp_type_belief_v1)
        item_belief_on = self.item_belief_coef > 0.0  # +ITEM belief CE (gen3_item_belief_v1)
        # +WIN-PROB: the head's MODE (none/read_only/shaping) lives on the extractor; the loss is added
        # whenever the mode is on. read_only vs shaping differ only in whether the
        # extractor stop-grads the head's input (the trunk gradient) — the loss term itself is identical.
        # +CRITIC MODE (gen3_winprob_critic_mode_v1): under the win-prob critic the win-prob head IS
        # the value function, so the BCE below stops being an auxiliary and becomes THE value loss
        # — at `vf_coef`, tagged "value" (never "aux": §1.4 of the design records that
        # `train/noise_scale_value` spent the distributional-critic era describing a zero-weighted
        # term). The scalar `value_loss` survives as a diagnostic; its TERM is dropped.
        critic_winprob = is_winprob(getattr(self.policy, "_critic_mode", "shaped"))
        win_prob_on = getattr(self.policy.features_extractor, "win_prob_mode", "none") != "none"
        # +SCAFFOLDING GAUGE: gated on the HEAD's existence alone — the
        # gauge is an observability read of whatever the head currently says. ALWAYS ON when the
        # head exists; there is no flag, matching the `signal/` group.
        scaffolding_on = getattr(self.policy.features_extractor, "win_prob_mode", "none") != "none"
        # +PG-COEF (gen3_policy_grad_coef_v1, `--policy-grad-coef`): the PPO policy-gradient term's own weight.
        # 1.0 (default) takes the UNSCALED `policy_loss` tensor — the loss expression is then
        # byte-identical to upstream; 0.0 removes the policy-gradient contribution alone. Scales ONLY `policy_loss` — entropy and the value term
        # keep their own coefficients (`ent_coef`, `vf_coef`).
        policy_grad_coef = float(getattr(self, "policy_grad_coef", 1.0))
        # +FORK-MASK (gen3_fork_v1): is the fork step's POLICY-TERM mask live for this call? The
        # predicate is the OBS KEY's presence and not the flag's value, deliberately: the key is
        # declared only when `--fork-fraction > 0`, and reading it is what the fold actually
        # depends on. A run whose flag is on but whose env never declared the key (a config
        # mismatch) then takes the unmasked expression instead of a KeyError three frames into the
        # loss.
        fork_pg_mask_on = (
            isinstance(self.rollout_buffer.observations, dict)
            and FORK_PG_MASK_KEY in self.rollout_buffer.observations)
        return FoldFlags(
            belief_aux_on=belief_aux_on, move_belief_on=move_belief_on, move_latent_on=move_latent_on,
            spread_belief_on=spread_belief_on, hp_type_belief_on=hp_type_belief_on, item_belief_on=item_belief_on,
            critic_winprob=critic_winprob, win_prob_on=win_prob_on, scaffolding_on=scaffolding_on,
            policy_grad_coef=policy_grad_coef,
            fork_pg_mask_on=fork_pg_mask_on,
        )

    def _train_probe_setup(self) -> ProbeSetup:
        """The once-per-`train()` probes and the gradient sampler."""
        # +INSTRUMENTATION: gradient-balance + value-scale diagnostics (grad_balance.py).
        # The dual-head extractor shares one trunk; both losses' gradients compete there. We
        # sample that pull ONCE per train() call (first minibatch) so vf_coef can be tuned to a
        # number rather than inferred from KL.
        # +DIAGNOSTICS CADENCE (gen3_diagnostics_cadence_v1): which OPTIONAL probes run on this call
        # — every `--diagnostics-every`-th update, the first update of the process, and any probe a
        # consumer declared load-bearing. A skipped probe leaves its dict EMPTY, so its TB tags are
        # not written at all (a gap, never a stale repeat). Read-only either way: bit-identical
        # learning is pinned by `diagnostics_cadence_test.py`.
        diag = plan_for(self)
        mark_ran(self, diag)
        shared_trunk = shared_trunk_parameters(self.policy.features_extractor)
        grad_balance: dict[str, float] = {}
        rank_metrics: dict[str, float] = {}  # effective rank of trunk / value_cls / policy reps (once/train)
        edge_metrics: dict[str, float] = {}  # edge/<fam>_{weight,grad}_norm — per-family liveness
        cell_metrics: dict[str, float] = {}  # cell/<name>_{weight,grad}_norm — per-CELL liveness
        grad_norms: list[float] = []  # pre-clip total grad norm (shows grad-clip activity)

        # +CAPACITY TELEMETRY (gen3_capacity_telemetry_v1): the plasticity canary, the half-batch
        # trunk-gradient cosine and the fixed-probe feature velocity. `None` when the flag is off,
        # and OFF is the whole cost — no head, no optimizer, no projection matrix, no probe batch,
        # and no extra forward or backward anywhere below. See `capacity_telemetry.py`.
        capacity = self._capacity()
        capacity_metrics: dict[str, float] = {}

        # +SIGNAL (gen3_signal_rate_metrics_v1): ADVANTAGE DENSITY — how much action-attributable
        # learning signal this rollout carries. Read ONCE per train() off the buffer's RAW GAE
        # advantages, HERE, because this is the last point at which they still exist unmodified:
        # the minibatch loop below applies `normalize_advantage`, which forces std→1 per minibatch
        # and so erases the very quantity being measured. Read-only numpy over the buffer — no
        # torch, no RNG, no gradient path, and the advantages PPO fits are untouched.
        # ⚠️ UNITS: these ride the run's own return units (`adv_kurtosis` is scale-free and compares
        # across runs). Must be read WITH `signal/outcome_entropy`
        # — see signal_metrics.py's module docstring for the mirror paradox and the 2x2 reading.
        signal_metrics = advantage_density_metrics(self.rollout_buffer.advantages)

        # +GRAD-ACCUM: number of `batch_size` micro-batches whose gradients are summed before one
        # optimizer.step() (1 = OFF, stock one-step-per-minibatch). See the class attr docstring.
        accum = max(1, int(getattr(self, "grad_accum_steps", 1)))

        # +NOISE-SCALE: when accumulating (accum>=2) we get gradient norms at two batch sizes for free —
        # one micro-batch (batch_size) and the full first group (batch_size·accum) — which is exactly
        # what the McCandlish gradient-noise-scale estimator needs. Captured once per train() (group 0 of
        # epoch 0) so |G_small|² and |G_big|² come from the SAME data; folded into the EMAs after the epochs.
        noise_g_small_sq = None   # ‖single micro-batch gradient‖²  (B = batch_size)
        noise_g_big_sq = None     # ‖accumulated group gradient‖²   (B = batch_size·accum)
        # +NOISE-SCALE PER-TERM: the same two points, taken per LOSS GROUP so the total reading can
        # be told apart from the PPO policy term's own (noise_scale_terms.py's docstring is the why).
        # Built only on a DIAGNOSTICS-CADENCE update (`--diagnostics-every`, diagnostics_cadence.py:
        # the cadence divides its cost) and NULL otherwise, in which case every `_ntg.add(...)`
        # below is a passthrough and no extra gradient is ever taken.
        _ns_terms = NULL_TAGGER
        if accum >= 2 and per_term_enabled(self) and diag.noise_terms:
            _ns_terms = PerTermNoiseSampler(list(self.policy.parameters()))
        return ProbeSetup(
            shared_trunk=shared_trunk, grad_balance=grad_balance, rank_metrics=rank_metrics,
            edge_metrics=edge_metrics, cell_metrics=cell_metrics, grad_norms=grad_norms,
            capacity=capacity, capacity_metrics=capacity_metrics,
            signal_metrics=signal_metrics, accum=accum, noise_g_small_sq=noise_g_small_sq,
            noise_g_big_sq=noise_g_big_sq, ns_terms=_ns_terms, diag=diag,
        )

    # ------------------------------------------------------------------ K8 region R1's setup
    def _micro_static(self, f: FoldFlags) -> Any:
        """The learner micro-step's STATIC flags and coefficients (`micro_step.MicroStatic`) for this
        `train()` call — every value R1 branches on or multiplies by that does not change within a
        run (`gen3_learner_micro_step_v1`). Read from the same attributes and buffer keys the inline
        fold read, at the same moment (once per call)."""
        from gymnasium import spaces

        from agents.training.instrumented_ppo.micro_step import MicroStatic
        progress = self._current_progress_remaining
        clip_range = float(self.clip_range(progress))
        clip_vf = (float(self.clip_range_vf(progress)) if self.clip_range_vf is not None else None)
        value_mode = "plain" if (f.critic_winprob or self.clip_range_vf is None) else "clipped"
        return MicroStatic(
            discrete=isinstance(self.action_space, spaces.Discrete),
            normalize_advantage=bool(self.normalize_advantage),
            clip_range=clip_range, clip_range_vf=clip_vf, value_mode=value_mode,
            critic_winprob=bool(f.critic_winprob),
            vf_coef=float(self.vf_coef), ent_coef=float(self.ent_coef),
            policy_grad_coef=float(f.policy_grad_coef),
            fork_pg_mask=bool(f.fork_pg_mask_on),
            belief_aux_on=bool(f.belief_aux_on), move_belief_on=bool(f.move_belief_on),
            move_latent_on=bool(f.move_latent_on), spread_belief_on=bool(f.spread_belief_on),
            hp_type_belief_on=bool(f.hp_type_belief_on), item_belief_on=bool(f.item_belief_on),
            belief_coefs=tuple((k, float(getattr(self, k))) for k in (
                "opp_belief_aux_coef", "move_belief_coef", "move_belief_latent_coef",
                "spread_belief_coef", "hp_type_belief_coef", "item_belief_coef")),
            moves_weight=float(self.opp_belief_moves_weight),
            intent_on=float(getattr(self, "opp_intent_coef", 0.0)) > 0.0,
            intent_coef=float(getattr(self, "opp_intent_coef", 0.0)),
            setvalued_on=float(getattr(self, "beta_setvalued_coef", 0.0)) > 0.0,
            setvalued_coef=float(getattr(self, "beta_setvalued_coef", 0.0)),
            bot_label_weight=float(getattr(self, "intent_label_bot_weight", 1.0)),
            win_prob_on=bool(f.win_prob_on),
        )

    def _micro_region(self) -> Any:
        """R1 as installed: the compiled region under --compile-trainer (K8), else the function."""
        compiled = getattr(self, "_compiled_micro_step", None)
        if compiled is not None:
            return compiled
        from agents.training.instrumented_ppo.micro_step import micro_step
        return micro_step
