"""`Gen3RewardManager` — the per-decision reward FOLD and its orchestrator.

⚠️ **The fold SEQUENCE is not split, and that is the design.** `process_turn_reward` applies one
treatment per `RewardClass` and folds ~35 terms in an order that is a CONTRACT (the PBRS folds
before the suppressions, the suppressions before the bias refund — see its docstring). Splitting
that sequence across modules would turn straight-line source order into something a reader has to
reassemble, which is the same rule `instrumented_ppo/ppo.py` keeps for its minibatch fold
(`ccd08003`).

Everything that is NOT the sequence moved out on 2026-09-07, taking this file from 1,990 lines —
ten short of the size gate's 2,000-line hard bound — to what you see:

    reward_config.py       the DECLARATIONS: `RewardClass`, `RewardConfig`, `RewardBreakdown`
                           (and its `_REGISTRY`, the reward's source of truth) and
                           `SWITCH_BIAS_DROP_FAMILY`. Re-exported below.
    reward_bias_terms.py   `RewardBiasTerms` — every `_compute_*` that produces one additive
                           BIAS field. A mixin: it reads this manager's cross-turn state.
    reward_potentials.py   `RewardPotentials` — the Φ potentials, `_pbrs_step`, `_hand_pbrs_on`
                           and the eight `_fold_*_pbrs`. Also a mixin.
    reward_composition.py  the stateless per-class CENSUS over a config (2026-09-06).
    reward_weights.py      every tunable MAGNITUDE (2026-08-23).

What stays here: the manager's state and its lifecycle (`__init__` / `reset` / `record_action` /
`report_episode`), the CURRENT-BOARD ACCESSORS every term reads the LiveView through, the
class-level applications (`_apply_progress_clock` / `_apply_bias_drops` /
`_apply_pbrs_suppression` / `_fold_bias_refund`), and the sequence itself.

⚠️ **A patch target follows the symbol.** `_encode_incoming_block` is read in
`reward_potentials`, not here; `src/test_stub_vacuity_gate_test.py` fails a stale target rather
than letting it stub nothing.
"""
from typing import Optional
import numpy as np
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel
from agents.action.constants import SWITCH_END as _SWITCH_END
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.training.reward_bias_terms import RewardBiasTerms
from agents.training.reward_potentials import RewardPotentials
from agents.training.reward_verify import build_twin as _build_verify_twin, verify_turn
from agents.training.reward_term_stats import (
    RewardTermAccumulator as _RewardTermAccumulator,
    tracked_terms as _tracked_terms,
)
from agents.observation.incoming_damage_encoder import IncomingBeliefMemo as _IncomingBeliefMemo


# --- The reward's DECLARATIONS ----------------------------------------------------------
# Moved to `reward_config.py` (2026-09-07) and re-exported HERE so every
# `from agents.training.reward_manager import RewardConfig` (the spelling ~20 modules and tests
# use) keeps resolving, exactly like `reward_weights` and `reward_composition`. Explicit rather
# than a star import: a name dropped from the other side is then an ImportError at import time.
from agents.training.reward_config import (   # noqa: E402,F401 - declared re-export hub
    RewardClass, RewardConfig, RewardBreakdown, SWITCH_BIAS_DROP_FAMILY)

# --- THE COMPOSITION ANNOUNCER -----------------------------------------------------------
# Extracted to `reward_composition.py` (2026-09-06) — the stateless, config-duck-typed census and
# its one-line render, which had grown to 117 lines inside a file at the size gate's hard bound.
# Re-exported HERE so every `from agents.training.reward_manager import reward_class_composition`
# (and the three underscore names the tests read) still resolves, exactly like `reward_weights`.

from agents.training.reward_composition import (   # noqa: E402,F401 - declared re-export hub
    _rc, _pbrs_term_active, _bias_term_active,
    reward_class_composition, reward_config_digest, format_reward_composition,
    inert_reward_flags, reward_composition_block,
)


# --- The reward's tunable MAGNITUDES ---------------------------------------------------
# Moved to `reward_weights.py` (the size ratchet's documented remedy) and re-exported HERE so
# every `from agents.training.reward_manager import <WEIGHT>` keeps resolving. The list is
# explicit rather than a star import: a name dropped from the other side is then an
# ImportError at import time, which is the loudest a missing constant can be.
from agents.training.reward_weights import (   # noqa: F401 - declared re-export hub
    HP_VALUE, VICTORY_VALUE, FINISHING_BLOW_BONUS, _TIMEOUT_TURN_CAP, MAT_HP_WEIGHT,
    MAT_ALIVE_WEIGHT, STALL_TAX_START_TURN, STALL_TAX_PER_TURN, STALL_TAX_RAMP_TURNS,
    STALL_TAX_MAX, STRUGGLE_LOOP_TAX, STRUGGLE_LOOP_THRESHOLD, SWITCH_BASE_BONUS, STATUS_BONUS,
    STATUS_TEMPO_WEIGHT, _TEMPO_STATUSES, ROAR_BONUS, OPP_BOOST_WEIGHT, ROAR_BOOST_WEIGHT,
    SE_SWITCH_BONUS, SLEEP_SWAP_BONUS, SPIKES_LAYER_BONUS, HAZARD_WEIGHT, SPIKES_WASTE_PENALTY,
    FAILED_ROAR_PENALTY, FUTILE_ATTACK_PENALTY, FUTILE_IMMUNE_PENALTY, ESCAPE_THREAT_BONUS,
    MATCHUP_PENALTY, PBRS_RISK_WEIGHT, PBRS_GAMMA, SWITCH_RISK_THRESHOLD, SAFE_PIVOT_PKO_MAX,
    STAY_RISK_TAX_FLOOR, ESCAPE_RISK_FRACTION, PROTECT_SWITCH_BONUS, STATUS_IMMUNE_SWITCH_BONUS,
    FUTILE_SETUP_PENALTY, SETUP_LOW_HP_THRESHOLD, SETUP_LOW_HP_MAX_PENALTY,
    STATUS_WASTED_PENALTY, BOOST_UTILIZED_SCALE, EXPLOSION_BLOCK_BONUS, BOOST_WEIGHT,
    REPETITION_STEP, REPETITION_ZERO_EFFECT_STEP, REPETITION_TAX_FLOOR, BOUNCING_TAX_STEP,
    BOUNCING_TAX_FLOOR, DEAD_MATCHUP_TAX_STEP, DEAD_MATCHUP_TAX_FLOOR, BOOST_MOVES,
    STATUS_INFLICTING_MOVES)


class Gen3RewardManager(RewardBiasTerms, RewardPotentials):
    """
    Self-contained reward calculator for the Gen 3 RL trainee agent.

    The two mixins are decomposition, not layering: `RewardBiasTerms` holds the per-term BIAS
    computations and `RewardPotentials` the Φ potentials and their PBRS folds, both written
    against `self` exactly as they were when they lived here. Neither carries state; every
    attribute path is this class's.

    Owns the full reward pipeline:
      - record_action()       — tracks the action choice and computes switch subsidy
      - process_turn_reward() — computes the full turn reward from the TurnDelta + the
                                current-board LiveView (built once per turn)

    Satisfies the RewardFunction protocol. Must only be called for the trainee's
    battle — the env gates this at the call site.
    """
    def __init__(self, log_level: LogLevel = LogLevel.QUIET,
                 config: Optional[RewardConfig] = None, progress_clock=None,
                 *, _shadow: bool = False):
        self.log_level = log_level
        # Per-run reward config (bias_additivity / mat_alive_weight / no_progress_penalty / gamma).
        # Default = the single-variable run (λ=1 bias additive, material always-on).
        self.config = config or RewardConfig()
        # The shared ProgressClock (owned by EpisodeTracker, updated at embed/record time). The reward
        # READS it for the no-progress penalty; obs and reward thus key on ONE value (design §5.1).
        self.progress_clock = progress_clock
        self.switch_count = 0
        self.forced_switch_count = 0
        self.attack_count = 0
        self.total_reward = 0.0
        self.last_switch_turn = -1
        self.logger = RateLimitedLogger(interval_seconds=1.0)
        self.episode_logger = RateLimitedLogger(interval_seconds=5.0)
        self._last_switched_from = "NULL"   # species we last voluntarily switched away from
        self._last_action_idx = -1
        self._last_reward_metadata = {}
        self._consecutive_struggle = 0
        self.struggle_turns = 0
        self._prev_opp_boosts: dict = {}    # opp active boosts after last turn (for Roar check)
        # opp spikes layers after last turn (for the Spikes bonus delta). NOTE: ProgressClock keeps
        # its OWN `_prev_spikes` for the hazard-progress check — two copies of the same fact, kept
        # coherent because both read the single per-turn LiveView (the clock at embed, this at fold).
        self._prev_opp_spikes: int = 0
        self._prev_opp_se_threat: bool = False  # did opp have a revealed SE move vs us last turn
        self._prev_our_statused = 0
        self._prev_opp_statused = 0
        self._last_switch_was_roared = False
        self._consecutive_attack_repeats: int = 0
        self._consecutive_bounces: int = 0          # A→B→A→B oscillation depth
        self._consecutive_dead_matchup_stays: int = 0  # turns stuck in a 0×-only matchup
        self._last_attack_had_effect: bool = True
        self._our_active_hp_before: float = 1.0
        self._opp_active_hp_before: float = 1.0
        self._our_boosts_before: np.ndarray = np.zeros(7, dtype=np.int8)
        self._last_opp_seen_by: dict[str, str] = {}
        # maps our_species → opp_species when this mon last switched in (voluntary, not roared)
        # PBRS state: Φ(s) at the previous window's end per potential (None at episode start → the
        # first transition adds no shaping), the belief re-gate's active KO-risk snapshot, and the
        # running BIAS accumulator for the bias-additivity refund (design §1.2).
        self._prev_phi_belief: Optional[float] = None   # incoming-KO belief PBRS (was _prev_phi)
        self._prev_phi_mat: Optional[float] = None       # material PBRS Φ_mat (design §2)
        self._prev_phi_status: Optional[float] = None    # non-damaging-tempo status PBRS (design §2.7)
        self._prev_phi_progress: Optional[float] = None   # anti-stall PBRS Φ_progress
        self._prev_phi_hazard: Optional[float] = None      # hazard/spikes PBRS Φ_hazard (design §2.6)
        self._prev_phi_boost: Optional[float] = None       # stored-boost PBRS Φ_boost
        self._prev_phi_opp_boosts: Optional[float] = None  # opp-boost-disruption PBRS Φ_opp_boosts
        self._prev_phi_roar: Optional[float] = None        # dedicated phaze-out-boosts PBRS Φ_roar
        self._prev_active_ko_risk: float = 0.0
        self._prev_safe_pivot: bool = False              # did a safe bench pivot exist at decision time
        self._cur_can_switch: bool = True                # was a switch LEGAL at this decision (mask) —
                                                         # gates the stay-tax so a trapped stay isn't taxed
        self._bias_acc: float = 0.0                      # Σ BIAS-class contributions this episode
        # Normalized material margin ∈ [−1,1] (clamped Φ_mat / bound), stashed by _compute_phi_mat each
        # turn as a cheap by-product. Read by gen3_env's win_margin obs key for the win-prob head's
        # closeness-stratified training metrics (value lives in close games, |margin|≈0). Reward-neutral.
        self._last_material_margin: float = 0.0

        # --- Suppressed-term fast path — design + shadow mode: `reward_verify.py` ------------
        # Activeness derived ONCE from `_bias_term_active` (the source the startup census reads),
        # NEVER a hand-copied name list (the v79 lesson). `_bias_term_active(f) is False` implies
        # the FINAL `bd.f` is 0.0 on every path; where imprecise it errs ACTIVE (it omits the
        # progress clock's extra zeroing under --bias-redesign) — time, never correctness.
        self._active_bias: frozenset = frozenset(
            n for n in RewardBreakdown.registry_fields(RewardClass.BIAS)
            if _bias_term_active(self.config, n))
        self._skip_inactive_bias: bool = not _shadow    # the shadow twin computes EVERYTHING

        # --- TERMINAL-ONLY short circuit (`gen3_terminal_only_short_circuit_v1`) ----------------
        # True when this run's reward is the ±victory_value terminal ALONE — no PBRS, no BIAS.
        # Read from `reward_class_composition`, the SAME announcer the startup line and the
        # `reward/` export derive from, so there is ONE predicate for "what is this reward made
        # of". NEVER a second hand-written condition on `hand_shaping`: `--arm-no-progress-tax`
        # re-arms one BIAS term under `--no-hand-shaping`, so the two are not equivalent (the v79
        # lesson the `_active_bias` block above already encodes). The shadow twin is excluded — it
        # must compute everything to verify against. Detail: this method's docstring.
        _comp = reward_class_composition(self.config)
        self._terminal_only: bool = (not _shadow
                                     and _comp["pbrs"] == 0 and _comp["bias"] == 0)
        self._verify_twin = None if _shadow else _build_verify_twin(
            Gen3RewardManager, self.config, progress_clock, LogLevel.QUIET)

        # Belief-block memo (`gen3_belief_block_memo_v1`) — Φ_belief's `encode_block` was 60.0% of
        # this method. CONTENT-keyed and therefore exact, not a state carry; the key-coverage proof
        # and the "why not `_state_epoch`" argument live in `IncomingBeliefMemo`'s docstring. The
        # shadow twin gets NO memo, so `GEN3AI_REWARD_VERIFY=1` gates key COMPLETENESS by
        # construction — a twin with its own memo would warm identically under an under-key and the
        # two would agree on the same wrong number.
        self._belief_memo = None if _shadow else _IncomingBeliefMemo()

        # The `reward/` live export (`gen3_reward_term_export_v1`) — per-decision sums of the
        # ACTIVE terms, drained by `env_method` once per rollout. The tracked set is derived from
        # `reward_class_composition`, the SAME predicates the folds are gated on, so the export
        # cannot disagree with the startup census. Rationale: `reward_term_stats.py`'s docstring.
        self._term_stats = None if _shadow else _RewardTermAccumulator(
            _tracked_terms(reward_class_composition(self.config)))

    def _bias_active(self, *names: str) -> bool:
        """Is ANY of these BIAS fields reachable under this run's config? The suppressed-term
        gate — always True on the shadow twin. Callers pass the field name(s) the guarded
        computation ASSIGNS, so the gate and its term cannot drift apart."""
        if not self._skip_inactive_bias:
            return True
        return not self._active_bias.isdisjoint(names)

    def drain_reward_terms(self):
        """Drain-and-zero the `reward/` term accumulator (`env_method` PULL, once per rollout).
        ``None`` on the shadow twin, which must stay observationally identical."""
        return self._term_stats.drain() if self._term_stats is not None else None

    def _prev_phi_fields(self) -> tuple:
        """Every ``_prev_phi_*`` PBRS carry-over, DERIVED from the instance — the single source for
        `reset()` and for the test that pins it. Declaring the attribute is the whole ceremony."""
        return tuple(sorted(n for n in vars(self) if n.startswith("_prev_phi_")))

    def reset(self):
        if self._verify_twin is not None:
            self._verify_twin.reset()
        self.switch_count = 0
        self.forced_switch_count = 0
        self.attack_count = 0
        self.total_reward = 0.0
        self.last_switch_turn = -1
        self._last_switched_from = "NULL"
        self._last_action_idx = -1
        self._last_reward_metadata = {}
        self._consecutive_struggle = 0
        self.struggle_turns = 0
        self._prev_opp_boosts = {}
        self._prev_opp_spikes = 0
        self._prev_opp_se_threat = False
        self._prev_our_statused = 0
        self._prev_opp_statused = 0
        self._last_switch_was_roared = False
        self._consecutive_attack_repeats = 0
        self._consecutive_bounces = 0
        self._consecutive_dead_matchup_stays = 0
        self._last_attack_had_effect = True
        self._our_active_hp_before = 1.0
        self._opp_active_hp_before = 1.0
        self._our_boosts_before = np.zeros(7, dtype=np.int8)
        self._last_opp_seen_by = {}
        self._last_material_margin = 0.0
        # EVERY `_prev_phi_*`, from the attribute set — the hand list here dropped `_prev_phi_roar`
        # for the whole of its life (`gen3_prev_phi_reset_v1`: eight declared, seven cleared, and
        # nothing anywhere to notice). Pinned by `test_reset_clears_every_prev_phi_potential`.
        for _name in self._prev_phi_fields():
            setattr(self, _name, None)
        self._prev_active_ko_risk = 0.0
        if self._belief_memo is not None:
            self._belief_memo.clear()     # episode scope; correctness-neutral (content-keyed)
        self._prev_safe_pivot = False
        self._cur_can_switch = True
        self._bias_acc = 0.0

    def record_action(self, ctx: BattleContext, action: int) -> None:
        """
        Records the action the model chose for this turn.
        Called before the turn is processed, using the context the model saw.

        Switch detection uses the action index and ctx.phase directly so the
        subsidy is credited in the SAME turn as the switch, not the next one.
        """
        self._pending_subsidy = 0.0
        self._last_switch_was_roared = False
        # Decision-time switch legality (server-authoritative mask: switch slots are [0, SWITCH_END)).
        # Snapshotted here, at THIS decision, so the stay-tax never fires when we were TRAPPED — even
        # on the turn a trap is first applied (the mask already reflects it before any reject event).
        self._cur_can_switch = bool(np.any(ctx.mask[:_SWITCH_END]))

        # Snapshot HP and boosts at decision time for use in process_turn_reward
        our_slot = ctx.our_slot_map.get(ctx.our_active, 0)
        opp_slot = ctx.opp_slot_map.get(ctx.opp_active, 0)
        active_norm = str(ctx.our_active).upper()
        has_live_active = active_norm not in ("NONE", "NULL", "NONE_P1", "NONE_P2")
        self._our_active_hp_before = float(ctx.our_hp[our_slot]) if has_live_active else 0.0
        self._opp_active_hp_before = float(ctx.opp_hp[opp_slot])
        self._our_boosts_before = ctx.our_boosts.copy()

        repetition_tax = 0.0
        struggle_loop_tax = 0.0

        if action >= 6:
            # Attack or Struggle
            self.attack_count += 1

            # A move breaks any switch-oscillation streak.
            self._consecutive_bounces = 0

            if action == self._last_action_idx and action != -1:
                self._consecutive_attack_repeats += 1
                n = self._consecutive_attack_repeats   # 1, 2, 3, ... (uncapped)
                step = (REPETITION_ZERO_EFFECT_STEP if not self._last_attack_had_effect
                        else REPETITION_STEP)
                repetition_tax = max(-step * n, REPETITION_TAX_FLOOR)
                self._pending_subsidy += repetition_tax
            else:
                self._consecutive_attack_repeats = 0

            if action == 10:  # struggle — forced by server when all PP depleted
                self.struggle_turns += 1
                self._consecutive_struggle += 1
                if self._consecutive_struggle >= STRUGGLE_LOOP_THRESHOLD:
                    struggle_loop_tax = STRUGGLE_LOOP_TAX
                    self._pending_subsidy += struggle_loop_tax
            else:
                self._consecutive_struggle = 0

            self._last_reward_metadata = {
                "type": "ATTACK",
                "repetition_tax": repetition_tax,
                "struggle_loop_tax": struggle_loop_tax,
            }
        else:
            # Switch (action 0-5 = team slot index)
            self._consecutive_struggle = 0
            self._consecutive_attack_repeats = 0
            is_forced = ctx.phase == "forced_switch"

            if is_forced and has_live_active:
                # Roar/Whirlwind: mon is alive but phazed out — no subsidy, skip bonuses.
                # Phazing isn't voluntary oscillation, so it doesn't count as a bounce.
                self._consecutive_bounces = 0
                self._last_switch_was_roared = True
                self.forced_switch_count += 1
                self._last_switched_from = ctx.our_active
                self._last_reward_metadata = {"type": "FORCED_ROAR"}

            elif is_forced:
                # Post-faint replacement — no subsidy, not an oscillation
                self._consecutive_bounces = 0
                self.forced_switch_count += 1
                self._last_reward_metadata = {"type": "FORCED_FAINT"}

            else:
                # Voluntary switch — INTENT only. The subsidy / bounce-tax / escape
                # bonus and their cross-turn counters (switch_count, last_switch_turn,
                # _last_switched_from, _consecutive_bounces) are NOT computed here: a
                # pressed switch can silently fail to execute (poke-env "gap=0": the
                # opponent faints on hazard entry the same window and our switch never
                # realizes), and crediting/perturbing on the PRESS mis-attributes a
                # reward to a turn that shows no switch. We stash the intent and settle
                # it in _apply_switch_outcome() once delta.our_switch_to confirms the
                # switch actually happened.
                slot_to_species = {v: k for k, v in ctx.our_slot_map.items()}
                target_species = slot_to_species.get(action)
                self._last_reward_metadata = {
                    "type": "VOLUNTARY",
                    "decision_turn": ctx.turn,
                    "switch_from": ctx.our_active,   # mon we're leaving
                    "target_species": target_species,
                }

        self._last_action_idx = action

        # Shadow mode: keep the twin's cross-turn counters in lockstep with ours, or the per-turn
        # comparison would flag state drift as a skip bug.
        if self._verify_twin is not None:
            self._verify_twin.record_action(ctx, action)

    def _apply_switch_outcome(self, delta: TurnDelta, bd: "RewardBreakdown") -> None:
        """Settle the voluntary-switch subsidy at OUTCOME time.

        ``record_action`` recorded that the model PRESSED a voluntary switch (intent);
        this fires only when the event-sourced ``delta`` confirms a switch actually
        happened (``our_switch_to is not None``). On a poke-env "gap=0" no-op — the
        press didn't execute because the opponent fainted on hazard entry the same
        window — we credit nothing and leave the bounce/spam counters untouched, so a
        phantom switch neither earns +0.5 nor perturbs future oscillation magnitudes.

        For a realized switch the math + counter mutations are identical to the old
        record_action path (same inputs: the intent-captured target / decision-turn /
        switched-from mon, the persistent counters, and the pre-turn opp-SE-threat
        snapshot which `_update_opp_se_threat` hasn't refreshed yet this turn)."""
        meta = self._last_reward_metadata
        if delta.our_switch_to is None:
            return  # pressed a switch but it never executed — no credit, no counters

        target = meta.get("target_species")
        decision_turn = meta.get("decision_turn", -1)

        # Bounce: switched straight back to the mon we just left. Escalates with the
        # oscillation depth so A↔B for 10-30 turns is prohibitive, not a rounding error.
        if (target and target == self._last_switched_from
                and self._last_switched_from not in ("NULL", "NONE")):
            self._consecutive_bounces += 1
            bd.switch_bouncing_tax = max(
                BOUNCING_TAX_STEP * self._consecutive_bounces, BOUNCING_TAX_FLOOR)
        else:
            self._consecutive_bounces = 0

        # Spam-gate: a BACK-TO-BACK switch (we also switched last turn) earns no flat switch subsidy.
        # Gated in BOTH arms. (ai_v5_6 regression fix: the redesign previously DROPPED this gate on the
        # assumption the no-progress clock subsumed switch-spam — it does NOT. The clock's flat −0.15
        # is dwarfed by the per-switch reframes [se_switch+escape+pivot ≈ +0.5–0.95/switch], so a
        # bounce-farm policy collected them every turn, never attacked, timed out, and lost even to
        # random. See `progress_clock_fuzz_test` / the switch-farm guard in reward_bias_terms_test.)
        spam_mult = 1.0 if (decision_turn - self.last_switch_turn) > 1 else 0.0
        bd.switch_base = SWITCH_BASE_BONUS * spam_mult

        if self._prev_opp_se_threat or self._prev_active_ko_risk >= SWITCH_RISK_THRESHOLD:
            bd.escape_threat_switch = ESCAPE_THREAT_BONUS

        if self.config.bias_redesign and spam_mult == 0.0:
            # Under the redesign the anti-spam family is suppressed (subsumed by the clock), so the
            # switch reframes need their OWN bounce brake: a back-to-back switch zeros the WHOLE
            # per-switch reward family (se_switch / escape / pivot_* were folded earlier this turn in
            # process_turn_reward). Cycle-AGNOSTIC — it kills an A↔B 2-cycle AND an A→B→C N-cycle alike
            # (every-turn switching → no reward from ANY switch term), which the 2-cycle-only
            # switch_bouncing_tax misses. switch_count/last_switch_turn updates below still run.
            bd.escape_threat_switch = 0.0
            bd.se_switch = 0.0
            bd.pivot_protect = bd.pivot_status = bd.pivot_damage = 0.0

        # Belief-risk-scaled escape reward (the under-switch lever; OFF unless --switch-bias-weight>0):
        # reward leaving a high imminent-KO spot FOR A SAFE PIVOT, scaled by the risk escaped. Gated on
        # `_prev_safe_pivot` (symmetry with the stay-tax) so we reward escaping TO safety — NOT
        # sacrificing a fresh mon into the same threat — which also removes the no-safe-pivot rotation
        # farm. Asymmetric (< the stay-tax via ESCAPE_RISK_FRACTION) so there's no surface to bounce-farm
        # (the escalating switch_bouncing_tax + Φ_mat HP loss are the other brakes).
        if (self.config.switch_bias_weight > 0.0 and self._prev_safe_pivot
                and self._prev_active_ko_risk >= SWITCH_RISK_THRESHOLD):
            bd.escape_risk_bonus = (
                self.config.switch_bias_weight * ESCAPE_RISK_FRACTION * self._prev_active_ko_risk)

        self.switch_count += 1
        self.last_switch_turn = decision_turn
        self._last_switched_from = meta.get("switch_from", "NULL")

    def _apply_progress_clock(self, bd: "RewardBreakdown") -> None:
        """Read the shared ProgressClock's stashed penalty into ``bd.no_progress_tax`` and suppress
        the escalating anti-spam family it SUBSUMES (design §3.1) — gated on ``bias_redesign``. In the
        default single-variable run the clock only tracks the obs scalar (no penalty, taxes intact).

        ``switch_bouncing_tax`` is **deliberately NOT suppressed** (ai_v5_6 regression): the flat
        no-progress charge (−0.15) does not out-weigh the per-switch reframes, so the escalating
        2-cycle bounce tax is kept as a real negative brake alongside the spam-gate in
        `_apply_switch_outcome`. The clock still subsumes repetition / struggle / dead-matchup."""
        # Active under --bias-redesign OR --all-shaping-pbrs (the latter keeps no_progress_tax as the
        # anti-stall tilt; --stall-pbrs later zeros it and folds Φ_progress instead).
        if not ((self.config.bias_redesign or self.config.all_shaping_pbrs)
                and self.progress_clock is not None):
            return
        bd.no_progress_tax = float(getattr(self.progress_clock, "last_penalty", 0.0))
        bd.repetition_tax = bd.struggle_tax = bd.dead_matchup_tax = 0.0

    def _apply_bias_drops(self, bd: "RewardBreakdown") -> None:
        """De-bias cleanup (audit TIER-1 distorter removal): zero dropped BIAS terms BEFORE the refund
        fold (so they leave the bias accumulator too). Both flags default OFF → byte-identical no-op.
        `drop_redundant_bias` removes terms redundant with an existing PBRS/terminal term; `drop_switch_bias`
        removes the hand-coded (learnable) switch-strategy subsidy. See `RewardConfig` for the rationale."""
        if self.config.drop_redundant_bias:
            bd.stall_tax = 0.0
            bd.matchup_penalty = 0.0
        if self.config.drop_switch_bias:
            # SWITCH_BIAS_DROP_FAMILY is the single source of truth — shared with
            # `_bias_term_active`, so the census can't disagree with what is actually zeroed.
            for _name in SWITCH_BIAS_DROP_FAMILY:
                setattr(bd, _name, 0.0)

    def _apply_pbrs_suppression(self, bd: "RewardBreakdown") -> None:
        """End-state PBRS cleanup — two independent switches. Called BEFORE _apply_bias_drops /
        _fold_bias_refund so the zeroed terms leave the bias accumulator too. Both OFF → no-op (the
        no-op-equivalence test pins it). Mirrors how _apply_progress_clock suppresses the anti-spam family.

        --all-shaping-pbrs ("everything but stall"): ZERO every BIAS term EXCEPT the anti-stall tilt
          `no_progress_tax`, so ALL non-stall shaping is policy-invariant. The converts ride the new
          potentials (Φ_hazard/Φ_boost/Φ_opp_boosts/Φ_status); the futility taxes dissolve (ΔΦ=0); the
          good-outcome bonuses (finishing_blow/explosion_block/status_wasted) are redundant with Φ_mat.
          The bad turn-ramp `stall_tax` is among the zeroed terms (it taxed winning long games).
        --stall-pbrs ("stall"): ZERO `no_progress_tax` + `stall_tax` (Φ_progress, folded above, carries
          the anti-stall signal policy-invariantly). Both flags on ⇒ the WHOLE BIAS class is zero.
        --no-hand-shaping (CLEAN WORLD): zero the whole BIAS class outright — `no_progress_tax`
          included, which is what distinguishes it from --all-shaping-pbrs. The eight potentials
          never folded at all (every `_fold_*_pbrs` early-returns via `_hand_pbrs_on`), so what
          survives is TERMINAL plus whatever trainer-side shaping the run asked for."""
        if not self.config.hand_shaping:
            for name in bd.registry_fields(RewardClass.BIAS):
                setattr(bd, name, 0.0)
        if self.config.all_shaping_pbrs:
            # Everything-but-stall → PBRS: zero all BIAS except the kept anti-stall tilt.
            for name in bd.registry_fields(RewardClass.BIAS):
                if name != "no_progress_tax":
                    setattr(bd, name, 0.0)
        if self.config.stall_pbrs:
            # Stall → PBRS: the no_progress tilt is replaced by Φ_progress; the turn-ramp goes too.
            bd.no_progress_tax = 0.0
            bd.stall_tax = 0.0

    def _fold_bias_refund(self, bd: "RewardBreakdown") -> None:
        """BIAS-additivity accumulate-and-refund (design §1.2): emit −(1−λ)·Δacc into
        ``bd.bias_refund`` so the BIAS class's net episode contribution is λ·acc. At λ=1 the refund
        is identically 0 → byte-identical to the additive biases. Call LAST, after every BIAS field
        (incl. the clock read + anti-spam suppression) is finalised."""
        lam = self.config.bias_additivity
        if lam >= 1.0:
            return
        bias_sum = sum(getattr(bd, name) for name in bd.registry_fields(RewardClass.BIAS))
        new_acc = self._bias_acc + bias_sum
        bd.bias_refund = -(1.0 - lam) * (PBRS_GAMMA * new_acc - self._bias_acc)
        self._bias_acc = new_acc

    def process_turn_reward(self, battle, delta: TurnDelta) -> float:
        """Computes the full reward for a completed turn from the TurnDelta.

        Builds a RewardBreakdown with every named component; stores it on
        self._last_breakdown so callers (e.g. BattleRecorder) can inspect the
        per-signal contributions without touching the battle object.

        **TERMINAL-only short circuit** (``gen3_terminal_only_short_circuit_v1``). When
        ``reward_class_composition`` reports 0 PBRS and 0 BIAS terms — the win-prob arm's
        ``--no-hand-shaping --terminal-indicator`` — the reward IS ``victory_value·1{win}`` and
        every potential/bias computation below is dead. ``_active_bias`` already skipped the ~25
        gated BIAS computes; ``_terminal_only`` additionally skips the handful that were UNGATED
        *because* their cross-turn mutations feed BIAS terms, which under this composition have no
        reader. Each skip site below names the reader it proved dead, and the full table of what
        SURVIVES and why is in ``designs/training/reward.md`` → *And a SECOND fast path*.

        🚨 **A SHORT CIRCUIT MAY NEVER SKIP AN OBSERVATION FEATURE.** ``_fold_material_pbrs`` runs
        unconditionally below and computes Φ_mat above its own gate, because its by-product
        ``_last_material_margin`` is gen3_env's ``win_margin`` obs key — a reader the reward
        composition does not own. "No BIAS term reads this" is the right test for a cross-turn
        mutation and the WRONG one for anything the observation carries.
        """
        bd = RewardBreakdown()

        # Current-board facts are read through the vetted LiveView read-model, built
        # ONCE per turn here and threaded to every helper. The reward manager never
        # reaches into the raw poke-env battle for "what is true now".
        live = battle.live_view()

        # --- TERMINAL: the ±victory_value win/loss (out of scope; never shaped) ---
        # `victory_value` defaults to the `VICTORY_VALUE` module constant (30.0), so this is the
        # same number it always was; it is a config field so a ±1 terminal is reachable by flag
        # (gen3_clean_world_config_v1). The constant remains the DEFAULT's single source.
        victory = float(self.config.victory_value)
        indicator = bool(getattr(self.config, "terminal_indicator", False))
        won, lost, finished = self._terminal(live)
        if won:
            bd.win_loss = victory
        elif finished and indicator:
            # gen3_winprob_critic_mode_v1: the WIN INDICATOR. Every non-win terminal — decisive
            # loss, pre-cap tie, 250-turn timeout — pays exactly 0.0, so the undiscounted return
            # is `victory_value * 1{win}` and (at victory_value 1.0) V(s) == P(win|s) exactly.
            # `draw_penalty` and the draw/loss ORDERING are inapplicable here by construction,
            # which is why `--draw-penalty` is REFUSED under this mode rather than ignored.
            bd.win_loss = 0.0
        elif finished:
            # Non-win terminal. A no-progress STALL ends with the trainee FORFEITING at the turn cap
            # (gen3_env issues ForfeitBattleOrder at turn>=cap → lost=True, turn>=cap) — NOT a tie — so
            # the timeout is detected by the turn count, not by won/lost. A timeout takes draw_penalty
            # (set < -victory_value to make a stall strictly worse than a clean loss); a DECISIVE loss
            # — and the rare PRE-CAP TIE, which shares this branch — stays -victory_value.
            #
            # 🚨 THE ORDERING IS THE POINT, not the magnitudes. `draw_penalty = -35 < -30` exists so
            # that stalling to the cap is strictly worse than losing cleanly. On a ±1 terminal a
            # `draw_penalty` of 0.0 would INVERT that — a 250-turn stall would become the best
            # non-winning outcome, and a losing agent's optimal play would be to run out the clock.
            # The owner's ruling for the clean arm is draw = loss: `--victory-value 1.0
            # --draw-penalty -1.0`, with stall-rate / mean game length as a PRIMARY safety endpoint.
            timed_out = live.turn >= _TIMEOUT_TURN_CAP
            bd.win_loss = self.config.draw_penalty if timed_out else -victory
        is_terminal = won or lost or finished

        # --- Material PBRS Φ_mat (design §2): replaces the unconditional hp/faint base spine ---
        self._fold_material_pbrs(bd, live, is_terminal)

        # --- Explosion / self-destruct ---
        # The survive-the-Explosion credit is now carried by Φ_mat (opp lost a mon, we lost nothing),
        # so the old +2.0 literal is DELETED (design §2.5). The explosion_block bonus (no-sold the
        # Explosion via immunity / 0 damage) is KEPT — it lives inside the same `not we_fainted` gate.
        # ⚠️ Every `_bias_active(...)`-gated block below is a PURE value computation this run's
        # composition forces to 0.0. The gate skips the COMPUTE, never a cross-turn MUTATION:
        # `_compute_spikes_bonus`, `_compute_status_reward`, `_apply_switch_outcome`,
        # `_update_opp_se_threat` and the `_last_opp_seen_by` update all stay UNGATED, so the
        # manager's observable state is identical whether the skip fires or not (grep-verified).
        opp_event = delta.opp_damaging_event
        if opp_event is not None and self._bias_active("explosion_block") \
                and opp_event.move_id in ("explosion", "selfdestruct"):
            if not delta.we_fainted:
                if delta.our_hp_delta.sum() == 0.0 or opp_event.effectiveness == 0.0:
                    bd.explosion_block = EXPLOSION_BLOCK_BONUS
            # When we_fainted: Φ_mat already prices the mon loss; the symmetric trade undervalues it
            # (see _compute_self_ko_penalty) — the HP-scaled penalty below corrects it (default OFF).

        # --- Self-KO penalty (HP-scaled): Φ_mat prices a healthy 1-for-1 self-KO trade at ~0, so the
        # critic learns to like throwing away a healthy mon. OFF unless --self-ko-hp-penalty > 0. ---
        if self._bias_active("self_ko_penalty"):
            bd.self_ko_penalty = self._compute_self_ko_penalty(delta)

        # --- Finishing blow ---
        if self._bias_active("finishing_blow"):
            bd.finishing_blow = self._compute_finishing_blow_bonus(delta, live)

        # --- Attack signals ---
        if self._bias_active("roar"):
            bd.roar = self._compute_roar_bonus(delta, live)
        if self._bias_active("futile_attack"):
            bd.futile_attack = self._compute_futile_attack_penalty(delta, live)
        if self._bias_active("futile_setup"):
            bd.futile_setup = self._compute_futile_setup_penalty(delta)
        if self._bias_active("setup_low_hp"):
            bd.setup_low_hp = self._compute_setup_low_hp_penalty(delta)
        if self._bias_active("boost_utilized"):
            bd.boost_utilized = self._compute_boost_utilized(delta, live)

        # --- Field control ---
        # Ungated by `_bias_active` because it advances `_prev_opp_spikes` and raw `bd.spikes`
        # feeds `_last_attack_had_effect` below. Both readers are BIAS (`spikes`/`futile_spikes`,
        # and `repetition_tax` via the flag), so TERMINAL-only skips the whole call.
        if not self._terminal_only:
            bd.spikes, bd.futile_spikes = self._compute_spikes_bonus(delta, live)

        # --- Positional: penalty for staying in against a known threat ---
        if self._bias_active("matchup_penalty"):
            bd.matchup_penalty = self._compute_matchup_penalty(delta)
        # Belief-risk-scaled stay-into-KO tax (the under-switch lever; OFF unless --switch-bias-weight>0).
        if self._bias_active("stay_risk_tax"):
            bd.stay_risk_tax = self._compute_stay_risk_tax(delta)
        # Escalating penalty for refusing to pivot out of a 0×-only matchup — the most expensive
        # BIAS item (2.8% of the stage: a movedex walk + ≤4 effectiveness calls per non-switch
        # turn). Skippable WHOLE despite mutating `_consecutive_dead_matchup_stays`: that
        # counter's ONLY reader is this helper (grep-verified — no reference outside this file;
        # inside, only here / `__init__` / `reset`), so suppressed it is not observable state.
        if self._bias_active("dead_matchup_tax"):
            bd.dead_matchup_tax = self._compute_dead_matchup_tax(delta, live)

        # --- Switch rewards (see SWITCH_REWARDS.md for full breakdown) ---
        # Pivot, SE, and sleep-out are skipped when phazed — roar removes our
        # choice, so those signals don't apply.
        if delta.our_switch_to is not None and not self._terminal_only:
            # TERMINAL-only: every term in this block is BIAS, and the one ungated mutation
            # (`_last_opp_seen_by`) is read ONLY by `_compute_se_switch_bonus` — also BIAS.
            if not self._last_switch_was_roared:
                if self._bias_active("pivot_protect", "pivot_status", "pivot_damage"):
                    bd.pivot_protect, bd.pivot_status, bd.pivot_damage = self._compute_pivot_bonus(delta, live)
                if self._bias_active("se_switch"):
                    bd.se_switch = self._compute_se_switch_bonus(delta, live)
                if self._bias_active("sleep_out"):
                    bd.sleep_out = self._compute_sleep_out_bonus(delta, live)
                # Update per-mon opponent tracker for future se_switch gating. NEVER gated —
                # cross-turn state (and `se_switch` is active under other compositions).
                our_mon_in = live.ours.active
                opp_mon_in = live.opp.active
                if our_mon_in and opp_mon_in and not opp_mon_in.fainted:
                    self._last_opp_seen_by[our_mon_in.species] = opp_mon_in.species
            if self._bias_active("sleep_in"):
                bd.sleep_in = self._compute_sleep_in_penalty(delta, live)

        # --- Status signals ---
        # Ungated for the same reason as `spikes` above: `_prev_our/opp_statused` and
        # `_d_opp_statused` only ever reach BIAS terms.
        _d_opp_statused = 0
        if not self._terminal_only:
            bd.status, _d_opp_statused = self._compute_status_reward(delta, live)
        if self._bias_active("status_wasted"):
            bd.status_wasted = self._compute_status_wasted_penalty(delta, _d_opp_statused)

        # --- Subsidy / taxes ---
        # Attack taxes are action-keyed (a pressed move/struggle reliably resolves or
        # is |cant|), so record_action computes them. The switch subsidy is OUTCOME-keyed
        # and settled here against delta.our_switch_to (see _apply_switch_outcome).
        if hasattr(self, "_pending_subsidy"):
            del self._pending_subsidy
        meta = self._last_reward_metadata
        if meta.get("type") == "VOLUNTARY":
            self._apply_switch_outcome(delta, bd)
        elif meta.get("type") == "ATTACK":
            bd.repetition_tax = meta.get("repetition_tax", 0.0)
            bd.struggle_tax = meta.get("struggle_loop_tax", 0.0)

        # --- Progressive stall tax — kept GENTLE (design §4.3): the progress clock is offense-centric
        # and can't see defensive stalls, so a soft absolute-turn term covers them. ~−10 to turn 250.
        if self._bias_active("stall_tax") and battle.turn > STALL_TAX_START_TURN:
            ramp = (battle.turn - STALL_TAX_START_TURN) / STALL_TAX_RAMP_TURNS
            bd.stall_tax = -min(STALL_TAX_PER_TURN * ramp, STALL_TAX_MAX)

        # --- No-progress clock: read the stashed penalty + suppress the anti-spam family it subsumes
        # (design §4 / §3.1). Gated on bias_redesign; in the default run the clock only feeds the obs.
        self._apply_progress_clock(bd)

        # Update end-of-turn snapshots for next turn's checks. `_prev_opp_boosts` is read only by
        # `_compute_roar_bonus` and `_prev_opp_se_threat` only by `_apply_switch_outcome` /
        # `_compute_matchup_penalty` — all BIAS, so TERMINAL-only carries neither snapshot.
        if not self._terminal_only:
            self._prev_opp_boosts = self._opp_active_boosts(live)
            self._update_opp_se_threat(live)

        # --- Belief PBRS (design_reward_switching.md) → pbrs_belief + the re-gate risk snapshot ---
        # 🚨 THE EXPENSIVE ONE, and the reason this short circuit is worth having. The fold gates
        # only its EMITTED FIELD; the `_belief_potential_and_risk` call above that gate — one
        # `encode_block`, historically 60.0% of this method — runs unconditionally for two
        # snapshots that feed `_compute_stay_risk_tax` / `_apply_switch_outcome`. Both are BIAS,
        # so under TERMINAL-only nothing reads either and the encode is pure waste — a whole-call
        # skip the fold's own gate structurally cannot make.
        if not self._terminal_only:
            self._fold_belief_pbrs(bd, live, is_terminal)

            # --- Non-damaging-tempo status PBRS Φ_status (design §2.7) → pbrs_status (bias_redesign only) ---
            self._fold_status_pbrs(bd, live, is_terminal)

            # --- End-state PBRS potentials (design §2.6/§2.7 + boost/opp-boost/progress) → gated on all_shaping_pbrs ---
            self._fold_progress_pbrs(bd, is_terminal)
            self._fold_hazard_pbrs(bd, live, is_terminal)
            self._fold_boost_pbrs(bd, live, is_terminal)
            # Φ_opp_boosts and Φ_roar are the SAME potential at two weights (−w·Σmax(0, opp boost
            # stage)), so the Σ is computed once and threaded to both instead of walking twice.
            # The Σ is a 7-stage walk that ran even with both folds off; here it is provably dead.
            opp_positive_boosts = self._opp_positive_boost_stages(live)
            self._fold_opp_boosts_pbrs(bd, live, is_terminal, positive=opp_positive_boosts)
            self._fold_roar_pbrs(bd, live, is_terminal, positive=opp_positive_boosts)

            # Track whether our last move did anything PRODUCTIVE this turn, for the
            # escalating repetition tax. A move counts as effective if it dealt damage,
            # gained us a stat boost, landed a status, or added a hazard layer. Capped
            # setup (no boost change), capped hazards, redundant status, and immune/no-op
            # attacks all flip this to False, routing the next repeat through the steeper
            # ZERO_EFFECT step — that's how capped setup/hazards escalate.
            # Its ONLY reader is `record_action`'s `repetition_tax` (BIAS), and two of its four
            # inputs come from calls TERMINAL-only skipped above.
            self._last_attack_had_effect = (
                float(delta.opp_hp_delta.sum()) < 0
                or int(delta.our_boost_delta.sum()) > 0
                or _d_opp_statused > 0
                or bd.spikes > 0
            )

        # --- End-state PBRS suppression: zero the subsumed BIAS terms + redundant drops (gated on
        # all_shaping_pbrs). Runs AFTER all PBRS folds + the _last_attack_had_effect read (which reads
        # the raw bd.spikes), BEFORE the v12 drops, so the suppressed terms leave the bias accumulator. ---
        self._apply_pbrs_suppression(bd)

        # --- De-bias cleanup: zero dropped BIAS terms before the refund fold (both default OFF = no-op) ---
        self._apply_bias_drops(bd)

        # --- BIAS-additivity accumulate-and-refund (design §1.2) — LAST, after every BIAS field ---
        self._fold_bias_refund(bd)

        self._last_breakdown = bd
        reward = bd.total
        # +REWARD EXPORT: fold the ACTIVE terms. `reward` is passed rather than re-derived, so
        # `reward/untracked_abs_mean` compares against the number training actually saw.
        if self._term_stats is not None:
            self._term_stats.observe(bd, reward)
        self.total_reward += reward
        self._log_turn(bd, battle, meta)
        if self._verify_twin is not None:
            verify_turn(self._verify_twin, battle, delta, bd,
                        RewardBreakdown.field_names(), self._active_bias)
        return reward

    def _log_turn(self, bd: "RewardBreakdown", battle, meta: dict) -> None:
        """DETAILED/DEBUG per-turn trace (no effect on the reward) — split out of process_turn_reward."""
        if not (self.log_level >= LogLevel.DETAILED and self.logger.should_log()):
            return
        base_reward = bd.win_loss + bd.pbrs_material
        subsidy_val = bd.switch_base + bd.switch_bouncing_tax + bd.repetition_tax + bd.struggle_tax
        self.logger.log(
            f"  [REWARD] Turn {battle.turn} | Base: {base_reward:+.4f} | Subsidy: {subsidy_val:+.2f} | Won: {battle.won}\n",
            force=True
        )
        if self.log_level < LogLevel.DEBUG:
            return
        if meta.get("type") == "VOLUNTARY":
            realized = "realized" if bd.switch_base or bd.switch_bouncing_tax or bd.escape_threat_switch else "no-op (pressed switch didn't execute)"
            print(f"    🔍 [DEEP TRACE] Type: VOLUNTARY SWITCH ({realized})")
            print(f"       Base:{bd.switch_base:+.2f} | Bouncing:{bd.switch_bouncing_tax:+.2f} | Escape:{bd.escape_threat_switch:+.2f}")
        elif meta.get("type") == "ATTACK" and (bd.repetition_tax != 0 or bd.struggle_tax != 0):
            print(f"    🔍 [DEEP TRACE] Type: ATTACK | Repetition Tax: {bd.repetition_tax:.2f} | Struggle Loop Tax: {bd.struggle_tax:.2f}")
        elif meta.get("type") == "FORCED_FAINT":
            print("    🔍 [DEEP TRACE] Type: FORCED SWITCH (post-faint, no subsidy)")
        elif meta.get("type") == "FORCED_ROAR":
            print("    🔍 [DEEP TRACE] Type: FORCED SWITCH (roar/whirlwind, no bonuses)")

    def report_episode(self, battle):
        if self.log_level < LogLevel.PERIODIC or self.total_reward == 0:
            return

        status = "UNKNOWN"
        our_alive = opp_alive = turns = 0
        if battle:
            # alive counts via the LiveView read-model; win/lost/finished/turn are meta
            # scalars (allowed off the battle directly).
            live = battle.live_view()
            if battle.won: status = "WIN"
            elif battle.lost: status = "LOSS"
            elif battle.finished: status = "TIE/STALL"
            our_alive = sum(1 for m in live.ours.mons if not m.fainted)
            opp_alive = sum(1 for m in live.opp.mons if not m.fainted)
            turns = battle.turn

        self.episode_logger.log(
            f"\n🏁 Episode Finished | Reward: {self.total_reward:6.2f} | Status: {status:4} | "
            f"Mon: {our_alive} vs {opp_alive} | Turns: {turns:3} | "
            f"Attacks: {self.attack_count:2} | Sw(Vol): {self.switch_count:2} | Sw(For): {self.forced_switch_count:2} | "
            f"Struggle: {self.struggle_turns:2}\n",
            force=False
        )
