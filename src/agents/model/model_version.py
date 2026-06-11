from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

# Bump this whenever the ModelVersion schema changes (fields added/renamed/removed).
# Also add a migration case in _migrate_config().
#
# v3: added `vf_coef` — the PPO value-loss coefficient, recorded so a training resume
#   with a different `--vf-coef` is a hard error (changing the value head's gradient
#   scale mid-run is a silent training change). It is NOT weight-shape-relevant, so it
#   is deliberately EXCLUDED from check_compatible()'s universal load-check (which gates
#   frozen eval / self-play-pool / distill opponents too, where vf_coef is irrelevant);
#   it is enforced only on the training-resume path via check_vf_coef(). Old configs
#   migrate to the SB3 default 0.5 (= the value every pre-flag run was trained with).
#
# v4: added the reward-config hparams — `bias_additivity` (--bias-additivity, the per-run
#   BIAS additive↔telescoping knob), `mat_alive_weight` (--mat-alive-weight, the material-PBRS
#   per-mon-alive weight), and `bias_redesign` (--bias-redesign, the staged no-progress-clock +
#   reframe enable). Like vf_coef, these are resume-immutable VALUE-meaning hparams (changing them
#   mid-run silently shifts the reward) but NOT weight-shape — enforced only on the training-resume
#   path via check_reward_config(), excluded from check_compatible(). Old configs migrate to the
#   defaults (the single-variable run: 1.0 / 1.25 / False).
#
# v5: added `switch_bias_weight` (--switch-bias-weight, the belief-risk-scaled stay-into-KO BIAS lever
#   for the under-switch pathology; design_reward_switching.md §7). Same resume-immutable VALUE-meaning
#   treatment as the v4 reward hparams (folded into check_reward_config, excluded from
#   check_compatible). Old configs migrate to 0.0 (OFF = the lever absent, behavior unchanged).
#
# v6: added `use_popart` (PopArt value-target normalization toggle). Unlike the v3-v5 VALUE-meaning
#   hparams, PopArt changes the value head's STRUCTURE (normalized output + mu/sigma buffers), so it
#   is enforced in check_compatible() (gates EVERY load), not the resume-only path. Old configs
#   default False (no PopArt).
#
# v7: added `draw_penalty` (--draw-penalty, the terminal reward for a DRAW / 250-turn timeout). Same
#   resume-immutable VALUE-meaning treatment as the v4-v5 reward hparams (folded into
#   check_reward_config, excluded from check_compatible). Old configs migrate to -30.0 (== a decisive
#   loss = the prior behavior, where a tie scored -VICTORY_VALUE).
#
# v8: added `attend_unrevealed_opponents` (--attend-unrevealed-opponents). A BEHAVIORAL toggle that
#   keeps the opponent's still-hidden party attendable in the transformer instead of key-masking it.
#   Like v6/use_popart it changes the forward pass (the mask, policy AND value) rather than a reward
#   meaning, so it is enforced in check_compatible(); but unlike PopArt it leaves the state_dict
#   identical (no weight-shape / ARCH_SIGNATURE change). Old configs default False (baseline masking).
MODEL_CONFIG_VERSION = 8

# Change this when the neural architecture changes structurally in a way that makes
# weights from a different signature incompatible (e.g. adding LSTM, replacing attention).
# Same-family dim changes (role_token_size 128→256) don't need a new signature —
# check_compatible() catches those via the dim fields.
#
# v2 (gen3_unified_v2): turn-history TurnDelta slot expanded to 88 dims —
#   actor / target / switch_to species IDs (×6), boost deltas (×14), phase flag,
#   target_hp_delta, per-slot HP-level vectors, target-status onehots (×14, at
#   move-fire time, for Flash Fire-vs-frozen and sleep-talker reads). The history
#   embedding now reaches the species_embedding table for the first time, a new
#   wire that's not weight-compatible with v1 even if total_dim coincidentally
#   matched.
#
# v3 (gen3_abilities_v1): per-Pokémon ability block expanded 2 → 3 dims
#   ([ability1_id, ability2_id, known_flag]). For unrevealed opp slots the two
#   dex-possible Gen 3 abilities are written so the model has prior knowledge
#   (e.g. Snorlax = Immunity OR Thick Fat) instead of a flat zero. The role
#   encoder embeds BOTH ability IDs through the existing ability_embedding
#   table — a wire that didn't exist in v2. POKEMON_FULL_DIM 97 → 98, total
#   obs dim 2414 → 2426.
#
# v4 (gen3_abilities_v2): ability block grows to 4 dims with an inserted
#   `dominance` scalar — the Smogon-observed probability of ability1.
#   Layout becomes [ability1_id, ability2_id, dominance, known]. Priors are
#   now sourced from data/pokemon/gen3_ability_priors.json (top-2 by Smogon
#   usage), replacing the dex-slot-order approach from v3. POKEMON_FULL_DIM
#   98 → 99, total obs dim 2426 → 2438. The role encoder picks up the
#   dominance scalar as a passthrough float alongside the two ability
#   embeddings.
#
# v5 (gen3_move_outcome_v1): each turn-history TurnDelta slot gains move-outcome
#   reporting — our/opp move-outcome onehots (hit/miss/fail, ×6), our/opp crit
#   bits (×2), and the |cant| reason onehot widens 5 → 11 (recharge/taunt/
#   disable/imprison/truant/nopp added, with "move:"/"ability:" prefix
#   normalization). These are pass-through scalars routed through the existing
#   history embedding, inserted before the species-ID tail. TURN_DELTA_DIM
#   88 → 108 (+12 from the wider cant onehot, +8 from outcome/crit); total obs
#   dim shifts by N_HISTORY_TURNS × 20. Not weight-compatible with v4 — the
#   history projection input width changed.
#
# v6 (gen3_modular_v1): pure structural refactor — forward_internal decomposed
#   into phase nn.Modules (Embeddings / ObsUnpack / PokemonEncoder /
#   TeamTransformer / CLSPool / ProjectionAssembler). The math, dims, and outputs
#   are byte-identical to v5, but state_dict keys are now phase-prefixed
#   (e.g. move_network.* → pokemon_encoder.move_network.*, our_cls →
#   cls_pool.our_cls). Old checkpoints are intentionally incompatible so they
#   fail with a clean arch-family error instead of an SB3 strict-load KeyError.
#
# v7 (gen3_dual_value_v1): value-dedicated CLS readout (H4 / Option C). CLSPool
#   gains a third learned query (`value_cls`) that attends over all 12 team
#   tokens to produce a global value summary; ProjectionAssembler now emits a
#   (pi_combined, vf_combined) pair, and the root extractor has a second
#   projection head (`value_pre_norm` + `value_projection`). `forward` returns a
#   (pi_features, vf_features) tuple consumed by the new
#   `Gen3DualHeadMaskablePolicy`. The transformer body stays shared; only the
#   readout + projection + critic mlp branch are now independent. New weights and
#   a tuple-returning forward make this incompatible with v6 checkpoints.
#
# v8 (gen3_live_state_v1): the active-context + global-env blocks are re-sourced from
#   the event-sourced LiveView and substantially enriched (retrain-class). Active
#   context grows 23 → 55: the volatile block goes from a hand-picked 9 to the full
#   source-derived gen3 set (VOLATILE_DIM=41, crash-don't-drop, perish/stockpile
#   counters normalised) — recovering ~30 dropped volatiles (Disable/Encore/Taunt/
#   Destiny Bond/Curse/Yawn/Flash Fire/partial-trap/…). Global env grows 13 → 18:
#   weather is event-sourced with cause-aware permanence + turns-remaining (ability
#   weather = permanent, move weather = 5-turn countdown — read from the |-weather|
#   protocol, never guessed), the dead gen4+ weather slot is dropped, and per-side
#   Safeguard + Mist are added alongside Reflect/Light Screen. The weather feature the
#   extractor broadcasts into per-mon move context widens 6 → 7. Obs dim 2734 → 2823;
#   the global-token / active-ctx projection input widths all shift. Not weight-
#   compatible with v7.
#
# v9 (gen3_own_spread_v1): the own-team spread block (per-mon IVs/EVs/nature, 18 dims ×6
#   slots) now carries REAL data instead of constant fallbacks. gen3ou has no team preview,
#   so poke-env's apply_teambuilder_team (which matches the empty team-preview list) never
#   attached the spread, and own Pokemon.ivs/evs/nature stayed None — the spread block had
#   been emitting a constant vector (IVs all-31, EVs all-0, neutral nature) for every own mon,
#   i.e. zero signal. Fixed in the poke-env fork: Battle.parse_request now calls
#   backfill_teambuilder_spread() after building the team from the request, matching the
#   declared teambuilder team by species and filling in IVs/EVs/nature (spread only — it does
#   not re-run the full _update_from_teambuilder, so request-derived moves/PP/stats are
#   untouched). The obs spread block + LiveView read mon.ivs as before, now populated. Obs DIM
#   is unchanged (still 2823) — only the spread VALUES change — but the meaning of those dims
#   changes, so this is retrain-class: old checkpoints must not silently load.
#
# v10 (gen3_turn_delta_v2): TurnDelta is now folded from the event log (Step 4 of
#   the event-sourced battle migration). New per-decision-window fields: an 8-dim
#   faint-cause multi-hot per side (attack/hazard/weather/status/recoil/selfko/
#   leechseed/other), and our_attempted_move_id (the move we pressed, preserved even
#   when it never fired — freeze/sleep/flinch/cant/KO-before-act). attempted_switch_to
#   is NOT encoded (a pressed switch always executes, so it == switch_to); faint counts
#   live on the dataclass for reward but aren't encoded (redundant with the faint flags
#   + cause popcount). The cant one-hot switches to the authoritative gen3_effects vocab
#   (slp/frz/par/flinch/recharge/attract/disable/taunt/imprison/focuspunch/nopp/truant),
#   crash-don't-drop. Volatiles added to the active-context block: doomdesire/futuresight
#   (`-start` future-move volatiles) + the 11 gen3 ability-activation volatiles (Immunity/
#   Synchronize/Oblivious/Insomnia/Limber/OwnTempo/ShedSkin/StickyHold/SuctionCups/
#   VitalSpirit/MagmaArmor — poke-env's -activate path records them as effects; MagmaArmor
#   required adding Effect.MAGMA_ARMOR to the fork's enum); the event-log fuzz's per-decision
#   check + training smoke caught doomdesire/immunity. Ability activations now ALSO reveal
#   the opponent's ability persistently (abstract_battle -activate handler sets mon.ability
#   when None → per-mon ability block flips known=1), so the 11 ability-activation volatiles
#   COLLAPSE to one shared `ability_activated` slot (identity is in the ability block; the
#   volatile is just a hint to go look). VOLATILE_DIM 41 → 44. TurnDelta also folds STATUS
#   TRANSITIONS from the event log: our/opp status_applied + status_cured (4 × 7-dim
#   onehots) — the per-turn event (e.g. Lum Berry curing Toxic to enable a Dragon Dance),
#   distinct from the current-status snapshot; the cause-identity stays in the item/ability
#   block. Plus our/opp item-used BITS (2) marking an item was consumed/removed this window
#   (just a bit — the WHICH is in the per-mon item block, parity with ability_activated).
#   The embedded-ID positions are no longer hardcoded in the extractor: a single
#   TURN_DELTA_EMBEDDED_IDS manifest (in turn_delta_encoder) drives both the encoder
#   layout and features_extractor.embed_delta_slot (11 embedded IDs: 3 move + 2 type +
#   6 species). TURN_DELTA_DIM = 157, obs dim 2823 → 3299. Builds on v9 (own-team spread
#   backfill carries through). Not weight-compatible with v9.
#
# v11 (gen3_turn_delta_v3): turn-history window correctness fix. `prev_N_delta_vecs` was
#   folding each of the N history slots over `events_since(cursor)` — i.e. that turn's
#   cursor THROUGH NOW (no upper bound) — so every slot but the most-recent reported the
#   *latest* turn's event-derived fields (move/outcome/boosts/status/faint-cause), and the
#   per-step cost was O(N²). Now each slot folds exactly its own decision window
#   (`events_between(cursors[-1-i], cursors[-i])`; end=None for the most-recent). Obs dim is
#   unchanged (3299) — only the turn-history values change (older slots now carry their own
#   turn) — so this is retrain-class, not weight-shape-incompatible.
#
# v12 (gen3_trapping_signals_v1): route the three trapping signals into the model so it can
#   learn the hidden-information trap read (Arena Trap / Shadow Tag / Magnet Pull / Mean Look).
#   (1) + (2) two new reactive obs bits from the server-authoritative LegalActions snapshot —
#   trapped (confirmed cannot switch; redundant with the mask but explicit) and maybe_trapped
#   (the opponent MIGHT trap us; switches stay legal, so this is the only way the model can see
#   the risk before attempting a blind pivot and eating a rejection). They sit before the
#   matchups in the reactive block, so the extractor picks them up in non_matchup_rest;
#   REACTIVE_DIM 300 -> 302. (3) the rejected pivot becomes a first-class history event: a new
#   EventKind.CHOICE_REJECTED is recorded out-of-band (poke-env intercepts |error|[Unavailable
#   choice] before parse_message, so a duck-typed hook in _handle_battle_message calls
#   Gen3Battle.record_choice_rejected), TurnView folds it (attempted_rejected), TurnDelta gains
#   attempted_switch_rejected + the restored attempted_switch_to, and each TurnDelta slot gains
#   2 dims — an attempted_switch_rejected bit + the embedded attempted-switch species id
#   (manifest entry #12). TURN_DELTA_DIM 157 -> 159. Obs dim 3299 -> 3321 (+2 reactive +
#   N_HISTORY_TURNS x 2 history). Builds on v11. Not weight-compatible with v11.
# gen3_item_num_fix_v1: the per-Pokémon item id is now the true item-dex `num` (from data/, via
#   the gen3_data facade), not Showdown's `spritenum` as before. Obs dim unchanged (3321) and the
#   item embedding table size is unchanged (max_items=600 still covers the new max, 499), but the
#   item id -> item meaning is re-mapped for every item, so item embeddings learned under the old
#   ids are semantically invalid. Re-meaning an obs block is retrain-class. Builds on
#   gen3_trapping_signals_v1; not weight-compatible with it.
#
# gen3_move_effects_v1: action-aligned per-move EFFECT features in the reactive block. The only
#   per-move signals that previously reached the policy head in REQUEST (action) order were base
#   power and the type multiplier — so for status/utility moves (power 0, neutral multiplier) every
#   option looked identical at the head, and the model could not tell a setup move from a heal from
#   a wasted Toxic (it clicked immune Toxic into Poison-types for many turns). Now each of the 4
#   request-order move slots carries 9 flags — is_boost, is_heal, is_protect, is_phaze, is_hazard,
#   inflicts_status, status_will_land, pp_fraction, status_will_land_known. Static flags are derived
#   in the acquisition tool
#   from the field Showdown keys each mechanic on (flags.heal, volatileStatus, forceSwitch,
#   sideCondition, primary `status`, declarative self-positive boosts) PLUS a curated callback
#   override for Belly Drum (onHit-only boost); Curse's type-conditional setup is resolved live in
#   the encoder. status_will_land is a PRIOR-WEIGHTED probability in [0,1] (priors first, then
#   confirmation — same ability-distribution path as the matchup cells): 0 on a certain block
#   (type immunity / already statused / Substitute), else 1 − P(ability blocks the status) over the
#   opponent's Smogon ability prior, collapsing to 0/1 once the ability is revealed; the trailing
#   status_will_land_known bit flags confirmed-vs-prior with the SAME predicate the per-mon ability
#   block's `known` flag uses (revealed ability OR a type-certain hard block), so the policy can
#   tell a confirmed outcome from a prior estimate — parity with how abilities are routed. The block
#   sits before the matchups, so the extractor picks it up in non_matchup_rest → both policy and
#   value projection input widths grow (auto-discovered). REACTIVE_DIM 302 → 338; obs dim 3321 → 3357.
#   Builds on gen3_item_num_fix_v1; not weight-compatible with it.
# gen3_incoming_damage_v1: per-our-mon INCOMING-DAMAGE / OHKO BELIEF block (incoming_damage.py +
#   gen3_{move,spread,item}_priors): for the opp active vs each of our 6 mons, the phys/spec
#   expected-damage-fraction + mode-max P(KO) (gen3 damage formula + fixed-damage branch
#   [Seismic Toss/Night Shade/…] + Reflect/Screen/Sub/burn/weather modifiers + roll→P(KO), over the
#   usage-prior belief: revealed∪prior moves, offensive-tail stat) + P(outspeed) over the Speed
#   distribution, then 3 opp recovery scalars (Suicune-Rest discriminator). Sits after move-effects,
#   before the matchups → flows to both heads via non_matchup_rest (auto-discovered widths).
#   REACTIVE_DIM 338 → 371; obs dim 3357 → 3390. Builds on gen3_move_effects_v1; not weight-compatible.
# gen3_incoming_damage_v2: re-calibrates the incoming-damage / OHKO belief VALUES (same 33-dim block,
#   same obs dim 3390 — only the numbers change, so it's retrain-class, not weight-shape). Two
#   complementary belief-value fixes for the calibration tail found on run_20260606_204351 (17% of
#   direct-hit deaths read P(KO)<0.25): (1) P(KO) was too timid on near-OHKOs — the offensive-stat
#   tail percentile is raised 0.85→0.95 (the KO magnitude rides the tail; expected-damage
#   re-normalises to the mean, so the chip belief is unchanged) AND a gen3 critical-hit term
#   (_CRIT_P=1/16, ×2, screen-ignoring) is folded into P(KO), so a hit that only KOs on a strong set
#   or a crit reads a calibrated risk instead of ~0; (2) the candidate set is widened so the killing
#   move is no longer silently absent — a revealed bare Hidden Power (dex BP 0) expands into per-type
#   candidates (~70 BP, typed from the HP tracker's narrowed distribution / Smogon HP prior),
#   variable-power Return/Frustration (dex BP 0) are priced at 102 BP, and the prior floor/cap widen
#   (0.12→0.05, 4→6 per channel) so a low-usage super-effective coverage move survives into the pool
#   (the per-defender max over p_in_set·P(KO) is the real type-effectiveness gate). The HP tracker is
#   now threaded into the incoming-damage encoder. Not weight-compatible with v1 (the belief values a
#   reload would read are different → old critic readings of the block are invalid).
# gen3_markovian_progress_v1: adds the turns_since_progress reactive scalar (vec[14]) — the
#   log-saturated no-progress clock (design_markovian_reward_and_features.md §5.1), an
#   EpisodeTracker-owned cross-turn counter threaded into encode() like the HP tracker.
#   REACTIVE_SCALAR_DIM 14 → 15 → REACTIVE_DIM 371 → 372, obs dim 3390 → 3391. The scalar is
#   present in every run (the clock always tracks it for the obs); the no-progress PENALTY +
#   the obs-keyed reward reframes are gated on the reward's bias_redesign flag, so the
#   single-variable material-clutch-fix run and the bias-redesign run share one architecture.
#   The reward redesign also folds the material spine into a PBRS Φ_mat and renames the belief
#   PBRS field (pbrs_material → pbrs_belief); those are reward-VALUE changes (retrain-class) that
#   need no further arch bump. Not weight-compatible with gen3_incoming_damage_v2 (obs dim +1).
# gen3_incoming_crit_split_v1: SPLITS the incoming-damage belief's P(KO) into a modal no-crit line +
#   a per-channel crit-risk DELTA (crit-inclusive − no-crit ∈ [0, _CRIT_P]), and adds a per-mon
#   threat-PROVENANCE scalar (the dominant KO threat's p_in_set: 1.0 = a REVEALED move, <1.0 = a
#   usage-prior GUESS, 0.0 = no candidate can KO). Motivation: the model over-weighted uncontrollable
#   crit RNG (it should optimise EXPECTED value over the modal line, with crit as a priced tail) and had
#   no signal for how much of a threat is KNOWN vs guessed — both validated as gaps by the
#   representation-probe harness (the rep barely encodes damage spread). The crit risk is exposed as the
#   DELTA (not the near-redundant absolute crit-inclusive line, which is ≤6% above no-crit and gets
#   buried after standardization). INCOMING_PER_MON 5 → 8 → INCOMING_DMG_DIM 33 → 51 → REACTIVE_DIM
#   372 → 390, obs dim 3391 → 3409. Crit was ALREADY computed (folded into P(KO) since v2); this unblends
#   it as a delta + adds provenance, so the underlying numbers are unchanged — but the block layout/width
#   differ, so it is not weight-compatible with gen3_markovian_progress_v1.
ARCH_SIGNATURE = "gen3_incoming_crit_split_v1"


class ModelVersionError(Exception):
    pass


@dataclass
class ModelVersion:
    # Schema and architecture identity
    config_version: int
    arch_signature: str

    # From state_encoder.get_layout()
    species_embedding_dim: int
    max_species: int
    move_embedding_dim: int
    max_moves: int
    item_embedding_dim: int
    max_items: int
    ability_embedding_dim: int
    max_abilities: int
    type_embedding_dim: int
    max_types: int
    total_dim: int
    active_context_dim: int

    # From features_extractor.py module constants
    role_token_size: int
    projection_dim: int
    move_net_hidden: List[int]
    role_encoder_hidden: List[int]
    active_ctx_hidden: List[int]
    n_history_turns: int

    # From policy_kwargs in train_rl_agent.py
    net_arch: List[int]

    # PPO value-loss coefficient (`--vf-coef`). Recorded for resume-immutability
    # (check_vf_coef), NOT a weight-shape field — see MODEL_CONFIG_VERSION v3 note and
    # its exclusion from _WEIGHT_FIELDS in check_compatible(). Defaults to the SB3 default
    # so versions built for a weight-shape-only check (current_model_version, the roundtrip
    # test) need not supply it.
    vf_coef: float = 0.5

    # Reward-config hparams (v4) — resume-immutable VALUE-meaning, NOT weight-shape. Default = the
    # single-variable run (material clutch-fix only; BIAS additive). Enforced via check_reward_config.
    bias_additivity: float = 1.0
    mat_alive_weight: float = 1.25
    bias_redesign: bool = False
    switch_bias_weight: float = 0.0   # v5: belief-risk-scaled stay-into-KO BIAS lever (default OFF)
    # v7: terminal reward for a DRAW / 250-turn timeout. -30.0 = the prior behavior (tie == decisive
    # loss). Resume-immutable VALUE-meaning (check_reward_config), excluded from the weight-shape check.
    draw_penalty: float = -30.0

    # v6 feature toggle (value-checked, not weight-shape): PopArt value-target normalization. The
    # value head's parameterization + buffers differ when on, so it cannot be toggled on a resume.
    # Defaulted (must follow the defaulted fields above) so weight-shape-only callers need not supply it.
    use_popart: bool = False

    # v8 behavioral toggle (value-checked, not weight-shape): keep the opponent's still-hidden party
    # ATTENDABLE in the transformer instead of key-masking unrevealed slots like fainted mons. Changes
    # the forward-pass mask (policy AND value), not any weight shape or the obs layout, so it lives in
    # config_version (not ARCH_SIGNATURE) and is checked in check_compatible — resuming with a
    # different value would silently change the masking the policy trained under.
    attend_unrevealed_opponents: bool = False

    @classmethod
    def from_layout_and_policy_kwargs(
        cls,
        layout: Dict[str, Any],
        policy_kwargs: Dict[str, Any],
        vf_coef: float = 0.5,
        reward_config=None,
    ) -> ModelVersion:
        from agents.model.features_extractor import (
            ROLE_TOKEN_SIZE,
            PROJECTION_DIM,
            MOVE_NET_HIDDEN,
            ROLE_ENCODER_HIDDEN,
            ACTIVE_CTX_HIDDEN,
            NET_ARCH,
            N_HISTORY_TURNS,
        )
        return cls(
            config_version=MODEL_CONFIG_VERSION,
            arch_signature=ARCH_SIGNATURE,
            species_embedding_dim=layout["species_embedding_dim"],
            max_species=layout["max_species"],
            move_embedding_dim=layout["move_embedding_dim"],
            max_moves=layout["max_moves"],
            item_embedding_dim=layout["item_embedding_dim"],
            max_items=layout["max_items"],
            ability_embedding_dim=layout["ability_embedding_dim"],
            max_abilities=layout["max_abilities"],
            type_embedding_dim=layout["type_embedding_dim"],
            max_types=layout["max_types"],
            total_dim=layout["total_dim"],
            active_context_dim=layout["active_context_dim"],
            role_token_size=ROLE_TOKEN_SIZE,
            projection_dim=PROJECTION_DIM,
            move_net_hidden=list(MOVE_NET_HIDDEN),
            role_encoder_hidden=list(ROLE_ENCODER_HIDDEN),
            active_ctx_hidden=list(ACTIVE_CTX_HIDDEN),
            n_history_turns=N_HISTORY_TURNS,
            net_arch=list(policy_kwargs.get("net_arch", NET_ARCH)),
            vf_coef=vf_coef,
            bias_additivity=float(getattr(reward_config, "bias_additivity", 1.0)),
            mat_alive_weight=float(getattr(reward_config, "mat_alive_weight", 1.25)),
            bias_redesign=bool(getattr(reward_config, "bias_redesign", False)),
            switch_bias_weight=float(getattr(reward_config, "switch_bias_weight", 0.0)),
            draw_penalty=float(getattr(reward_config, "draw_penalty", -30.0)),
            use_popart=bool(policy_kwargs.get("use_popart", False)),
            attend_unrevealed_opponents=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "attend_unrevealed_opponents", False)
            ),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json_file(cls, path: str) -> ModelVersion:
        with open(path) as f:
            data = json.load(f)
        data = _migrate_config(data)
        return cls(**data)

    def check_compatible(self, saved: ModelVersion) -> None:
        """Raises ModelVersionError if saved is incompatible with self (current code).
        Call as: current_version.check_compatible(saved_version).
        """
        # Architecture family — hard stop if different
        if self.arch_signature != saved.arch_signature:
            raise ModelVersionError(
                f"Architecture family mismatch: saved='{saved.arch_signature}', "
                f"current='{self.arch_signature}'.\n"
                "These models use structurally different networks and cannot be loaded interchangeably.\n"
                "Start a fresh training run, or use subprocess isolation for league play."
            )

        # Weight-relevant fields — all must match exactly
        _WEIGHT_FIELDS = {
            "total_dim", "active_context_dim",
            "species_embedding_dim", "max_species",
            "move_embedding_dim", "max_moves",
            "item_embedding_dim", "max_items",
            "ability_embedding_dim", "max_abilities",
            "type_embedding_dim", "max_types",
            "role_token_size", "projection_dim",
            "move_net_hidden", "role_encoder_hidden", "active_ctx_hidden",
            "n_history_turns",
            "net_arch",
        }
        current = asdict(self)
        saved_d = asdict(saved)
        mismatches = [
            f"  {k}: saved={saved_d[k]!r}, current={current[k]!r}"
            for k in sorted(_WEIGHT_FIELDS)
            if current[k] != saved_d.get(k)
        ]
        if mismatches:
            raise ModelVersionError(
                "Model weight-shape mismatch — cannot load saved model with current architecture.\n"
                "Mismatched fields:\n" + "\n".join(mismatches) + "\n\n"
                "Fix: restore matching constants, or start a fresh training run."
            )

        # Feature toggle — value-checked (not weight-shape) but STRUCTURAL: PopArt adds value-head
        # buffers + normalized output, so loading a use_popart mismatch breaks the state_dict on
        # EVERY load. Unlike vf_coef / reward-config (value-meaning, resume-only) it lives here in
        # check_compatible (gates eval / pool / distill loads too), with a dedicated message.
        if self.use_popart != saved.use_popart:
            raise ModelVersionError(
                f"PopArt mismatch: saved={saved.use_popart}, current={self.use_popart}.\n"
                "PopArt changes the value head's parameterization (normalized output + running "
                "mu/sigma buffers), so it cannot be toggled on a resumed model.\n"
                "Resume with the matching --use-popart setting, or start a fresh training run."
            )

        # Behavioral toggle — value-checked (not weight-shape): unmasking the opponent's hidden
        # party changes the transformer's key_padding_mask (policy AND value forward). The state_dict
        # is identical either way, but a resume that flips it would feed the policy a different mask
        # than it trained under. Lives here (gates resume) with a dedicated message; same-run
        # pool/sentinel/distill snapshots carry the same value so they pass trivially.
        if self.attend_unrevealed_opponents != saved.attend_unrevealed_opponents:
            raise ModelVersionError(
                f"attend_unrevealed_opponents mismatch: saved={saved.attend_unrevealed_opponents}, "
                f"current={self.attend_unrevealed_opponents}.\n"
                "Unmasking the opponent's hidden party changes the transformer mask the policy was "
                "trained under, so it cannot be toggled on a resumed model.\n"
                "Resume with the matching --attend-unrevealed-opponents setting, or start a fresh run."
            )

    def check_opponent_compatible(self, foreign: "ModelVersion") -> None:
        """Gate for loading a frozen model from ANOTHER run as an inference-only OPPONENT
        (a "stable opponent"). Call as: ``current_version.check_opponent_compatible(foreign)``.

        A stable opponent is a pure ``observation -> action`` function: it consumes the obs the
        LIVE encoder produces and emits an action index that crosses into the shared battle. So the
        ONLY axis that must match is the OBSERVATION FAMILY — and ``arch_signature`` is the proxy
        for it: any obs-layout/meaning change bumps the signature, so equal signatures guarantee the
        same obs layout. (It ALSO bumps on pure network-structure refactors, making this stricter
        than strictly necessary — but in a safe direction, and same-arch ⟹ identical net sizes, so
        the foreign zip rebuilds its extractor at shapes matching its own weights with no further
        check needed. If an obs-identical-but-model-refactored opponent is ever wanted, split a
        dedicated ``obs_signature`` out of ``arch_signature`` and gate on that instead.)

        Deliberately DISTINCT from ``check_compatible`` (which gates the trainee's own resume + the
        self-play pool/sentinels, where every ``_WEIGHT_FIELD`` AND ``use_popart`` must match): an
        opponent never shares weights with the trainee and never reads its value head, so
        ``use_popart`` / ``vf_coef`` / the reward-config hparams are all irrelevant to its forward
        and are deliberately NOT checked here.
        """
        if self.arch_signature != foreign.arch_signature:
            raise ModelVersionError(
                f"Stable opponent architecture-family mismatch: "
                f"opponent='{foreign.arch_signature}', current='{self.arch_signature}'.\n"
                "A stable opponent must share the live run's arch_signature — i.e. the SAME "
                "observation layout (a different signature means the live encoder cannot feed it).\n"
                "Use an opponent trained at the current architecture, or start the new run at the "
                "opponent's architecture."
            )
        # Defensive: same arch_signature already implies these match, but a hand-edited config
        # could lie — and feeding the opponent a wrong-width obs would be a silent-garbage bug.
        for field in ("total_dim", "active_context_dim"):
            cur, opp = getattr(self, field), getattr(foreign, field)
            if cur != opp:
                raise ModelVersionError(
                    f"Stable opponent {field} mismatch: opponent={opp}, current={cur} "
                    "(arch_signature matched — the opponent's model_config.json looks hand-edited)."
                )

    def check_vf_coef(self, requested: float) -> None:
        """Raise ModelVersionError if `requested` (the resume `--vf-coef`) differs from this
        saved config's vf_coef.

        Call as: saved_version.check_vf_coef(args.vf_coef).

        vf_coef is a training-loss coefficient, not a weight-shape concern, so it is
        deliberately NOT part of check_compatible() — that gates EVERY checkpoint load,
        including the frozen eval / self-play-pool / distill opponents, where vf_coef is
        irrelevant (the forward pass is identical regardless of it). This check is invoked
        ONLY on the training-resume path: silently changing the value head's gradient scale
        mid-run would let a forgotten/typo'd flag drift training, so a resume with a
        different value is a hard error rather than a quiet change.
        """
        if not math.isclose(self.vf_coef, requested, rel_tol=1e-9, abs_tol=1e-12):
            raise ModelVersionError(
                f"vf_coef mismatch: saved={self.vf_coef!r}, requested={requested!r}.\n"
                "The PPO value-loss coefficient is fixed for the lifetime of a run — changing it on "
                "resume silently alters the value head's gradient scale.\n"
                f"Fix: resume with --vf-coef {self.vf_coef!r}, or start a fresh training run to use "
                f"{requested!r}."
            )

    def check_reward_config(self, reward_config) -> None:
        """Raise ModelVersionError if the resume `reward_config` differs from this saved config's
        reward hparams (bias_additivity / mat_alive_weight / bias_redesign). Like check_vf_coef:
        these are VALUE-meaning (changing them mid-run silently shifts the reward), NOT weight-shape,
        so they are enforced ONLY on the training-resume path and excluded from check_compatible().
        Call as: saved_version.check_reward_config(args_reward_config)."""
        req_ba = float(getattr(reward_config, "bias_additivity", 1.0))
        req_maw = float(getattr(reward_config, "mat_alive_weight", 1.25))
        req_br = bool(getattr(reward_config, "bias_redesign", False))
        req_sbw = float(getattr(reward_config, "switch_bias_weight", 0.0))
        req_dp = float(getattr(reward_config, "draw_penalty", -30.0))
        problems = []
        if not math.isclose(self.bias_additivity, req_ba, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"  bias_additivity: saved={self.bias_additivity!r}, requested={req_ba!r}")
        if not math.isclose(self.mat_alive_weight, req_maw, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"  mat_alive_weight: saved={self.mat_alive_weight!r}, requested={req_maw!r}")
        if self.bias_redesign != req_br:
            problems.append(f"  bias_redesign: saved={self.bias_redesign!r}, requested={req_br!r}")
        if not math.isclose(self.switch_bias_weight, req_sbw, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"  switch_bias_weight: saved={self.switch_bias_weight!r}, requested={req_sbw!r}")
        if not math.isclose(self.draw_penalty, req_dp, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"  draw_penalty: saved={self.draw_penalty!r}, requested={req_dp!r}")
        if problems:
            raise ModelVersionError(
                "Reward-config mismatch on resume — these hparams are fixed for a run's lifetime "
                "(changing them silently shifts the reward / objective):\n" + "\n".join(problems) +
                "\n\nFix: resume with the saved values, or start a fresh run."
            )


def _migrate_config(data: dict) -> dict:
    """Apply incremental forward-migrations to bring an old config up to the current schema."""
    version = data.get("config_version", 1)
    if version < 2:
        # v2: added n_history_turns. Old models used a single TurnDelta (N=1).
        data.setdefault("n_history_turns", 1)
        data["config_version"] = 2
    if version < 3:
        # v3: added vf_coef. Every pre-flag run trained with the SB3 default 0.5.
        data.setdefault("vf_coef", 0.5)
        data["config_version"] = 3
    if version < 4:
        # v4: added reward-config hparams. Pre-flag runs used the single-variable defaults.
        data.setdefault("bias_additivity", 1.0)
        data.setdefault("mat_alive_weight", 1.25)
        data.setdefault("bias_redesign", False)
        data["config_version"] = 4
    if version < 5:
        # v5: added the switch-bias lever. Pre-flag runs had it absent (OFF).
        data.setdefault("switch_bias_weight", 0.0)
        data["config_version"] = 5
    if version < 6:
        # v6: added use_popart. Old models did not use PopArt value normalization.
        data.setdefault("use_popart", False)
        data["config_version"] = 6
    if version < 7:
        # v7: added draw_penalty. Old runs scored a tie/timeout as a decisive loss (-VICTORY_VALUE).
        data.setdefault("draw_penalty", -30.0)
        data["config_version"] = 7
    if version < 8:
        # v8: added attend_unrevealed_opponents. Old models key-masked unrevealed opp slots.
        data.setdefault("attend_unrevealed_opponents", False)
        data["config_version"] = 8
    return data
