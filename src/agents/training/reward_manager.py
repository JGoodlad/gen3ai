from dataclasses import dataclass, field, fields
from enum import Enum
from typing import ClassVar, Optional
import numpy as np
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel
from agents.enums import Status
from agents.action.constants import SWITCH_END as _SWITCH_END
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.gen3_data import moves as _movedex
from agents.gen3_mechanics import (
    INVULNERABLE_MOVES as _INVULNERABLE_MOVES,
    effective_multiplier_by_types as _effective_multiplier_by_types_fn,
    STATUS_MOVE_IMMUNITY as _STATUS_MOVE_IMMUNITY,
)
from agents.enums import PokemonType as _PokemonType
from agents.observation.incoming_damage_encoder import encode_block as _encode_incoming_block
from agents.observation.constants import (
    TEAM_SIZE as _TEAM_SIZE,
    INCOMING_PER_MON as _INCOMING_PER_MON,
)


def _ptype(name) -> "Optional[_PokemonType]":
    """LiveView type-id string (e.g. ``'fire'``) -> ``PokemonType`` enum — the primitive
    the mechanics helpers key on. ``None`` passes through."""
    return _PokemonType[name.upper()] if name else None


def _status_enum(name) -> "Optional[Status]":
    """LiveView status-id string (e.g. ``'slp'``) -> ``Status`` enum. ``None`` passes
    through. Only ``FRZ`` actually changes an effectiveness result (Flash Fire), but we
    convert faithfully so the LiveView path is byte-identical to the raw-battle path."""
    return Status[name.upper()] if name else None

class RewardClass(Enum):
    """The three reward classes (design_markovian_reward_and_features.md §1.1). The fold
    loop applies one treatment per class — no per-term special-casing:

    * TERMINAL — the ±30 win/loss. Emitted as-is; never shaped/blended/flag-affected.
    * PBRS     — pure potential hints, ALWAYS telescoping (Φ(terminal)=0), objective-neutral
                 regardless of any flag. The field already holds γ·Φ(s′) − Φ(s).
    * BIAS     — soft shaping whose additive↔telescoping mix is set by `--bias-additivity`
                 (accumulate-and-refund). The field holds the additive per-turn value.
    """
    TERMINAL = "terminal"
    PBRS = "pbrs"
    BIAS = "bias"


@dataclass
class RewardConfig:
    """Per-run reward configuration (resume-immutable; recorded in model_config.json).

    `bias_additivity` (λ ∈ [0,1], default 1.0) sets how much the BIAS class biases the
    objective: 1.0 = fully additive (byte-identical to today), 0.0 = fully telescoping
    (pure PBRS hint), λ = blend (episode contribution λ·acc). It is a per-run CONSTANT,
    NOT annealed within a run. `gamma` MUST equal the PPO gamma (asserted in train_rl_agent).
    """
    bias_additivity: float = 1.0
    mat_alive_weight: float = 1.25
    no_progress_penalty: float = 0.15
    # Belief-risk-scaled switch BIAS lever (design_reward_switching.md §7). 0.0 = OFF (default →
    # byte-unchanged run). >0 enables the risk-scaled stay-tax (−w·active_risk, safe-pivot-gated) +
    # escape reward (+w·ESCAPE_RISK_FRACTION·active_risk). The behavioral fix for under-switching the
    # policy-invariant pbrs_belief can't provide. Per-run constant, resume-immutable.
    switch_bias_weight: float = 0.0
    gamma: float = 0.9999
    # Staged BIAS redesign (design §1.3). OFF → the default single-variable run: today's anti-spam
    # taxes (repetition/bouncing/dead-matchup/struggle) + today's roar/status/spikes, so the ONLY
    # reward-behavior change vs the live baseline is the material clutch-fix. ON → the redesign: the
    # no-progress clock replaces the anti-spam family, and the obs-keyed reframes apply. The
    # turns_since_progress OBS scalar is present either way (the clock always tracks it); only the
    # PENALTY + the reframes are gated, so both arms share one architecture (and can A/B by resume).
    bias_redesign: bool = False


@dataclass
class RewardBreakdown:
    """Per-component reward breakdown for a single turn. Stored on Gen3RewardManager
    as _last_breakdown after each process_turn_reward() call. Zero fields are omitted
    from to_dict() so the JSON stays compact."""

    # Base outcome
    win_loss: float = 0.0          # TERMINAL — the ±30 win/loss (out of scope; never shaped)
    # Material PBRS Φ_mat (design §2): HP/faint/margin folded into ONE always-on potential
    # (declared-team material), so material no longer banks the lead — every win returns +30,
    # every loss −30. = γ·Φ_mat(s′) − Φ_mat(s), Φ_mat(terminal)=0. Replaces the old unconditional
    # hp_ours/hp_opp/faint_ours/faint_opp base spine (the clutch-vs-dominant fix).
    pbrs_material: float = 0.0
    explosion: float = 0.0         # vestigial: the survive-Explosion credit is now carried by Φ_mat
                                   # (opp lost a mon); the old +2.0 literal is deleted (design §2.5)
    explosion_block: float = 0.0   # Ghost immune or Protect blocked opponent Explosion
    finishing_blow: float = 0.0    # damaging move secured the KO

    # Attack signals
    roar: float = 0.0
    futile_attack: float = 0.0
    futile_setup: float = 0.0      # setup move used at stat cap (+6 or -6)
    setup_low_hp: float = 0.0      # setup move chosen below 40% HP (penalty)
    boost_utilized: float = 0.0    # attacked while holding active stat boosts
    status_wasted: float = 0.0     # status-inflicting move had no effect

    # Field control
    spikes: float = 0.0

    # Positional
    matchup_penalty: float = 0.0
    dead_matchup_tax: float = 0.0  # escalating penalty for staying in a 0×-only matchup
    stay_risk_tax: float = 0.0     # belief-risk-scaled penalty for staying into a high P(KO) when a
                                   # safe pivot exists (the under-switch lever; --switch-bias-weight)

    # Switch: subsidy (set by record_action before the turn)
    switch_base: float = 0.0       # flat per-voluntary-switch subsidy
    switch_bouncing_tax: float = 0.0  # penalty for immediately switching back
    repetition_tax: float = 0.0    # same attack repeated consecutively
    struggle_tax: float = 0.0      # struggle loop penalty

    # Switch: pivot signals (what the opponent did on our switch turn)
    pivot_protect: float = 0.0     # opponent used Protect/Detect/Endure
    pivot_status: float = 0.0      # opponent's status move was type-immune on our switch-in
    pivot_damage: float = 0.0      # opponent's damaging move hit our switch-in less than old mon

    # Switch: offensive threat
    se_switch: float = 0.0         # our switch-in has a SE move vs opponent active
    escape_threat_switch: float = 0.0  # switched out while opp had a revealed SE threat vs us
    escape_risk_bonus: float = 0.0     # belief-risk-scaled reward for escaping a high-P(KO) spot
                                       # (the under-switch lever; --switch-bias-weight)

    # Switch: sleep rotation
    sleep_out: float = 0.0         # rotated a sleeping mon to bench
    sleep_in: float = 0.0          # sent in a sleeping mon (penalty)

    # Status signals
    status: float = 0.0
    futile_spikes: float = 0.0     # wasted Spikes at the 3-layer cap (design §2.6 — split out of `spikes`)

    # Progressive stall tax (kept GENTLE — the progress clock can't see defensive stalls, §4.3)
    stall_tax: float = 0.0
    # Anti-no-progress clock (design §4): one Markovian penalty collapsing the anti-spam family.
    # Charged per no-progress window; obs-keyed on turns_since_progress.
    no_progress_tax: float = 0.0

    # Belief-based switch shaping (design_reward_switching.md): PBRS over the incoming-KO belief
    # (RENAMED from the mis-named `pbrs_material`). Policy-invariant; NEVER touches the ±30 terminal.
    pbrs_belief: float = 0.0

    # BIAS-class accumulate-refund (design §1.2): the −(1−λ)·Δacc per-turn refund that dials the
    # BIAS class from fully additive (λ=1, refund≡0, byte-identical to today) toward fully
    # telescoping (λ=0). NOT a reward term — the fold's output; excluded from the registry.
    bias_refund: float = 0.0

    # ---- The reward registry (design §1.1): field name → class. Single source of truth that
    # drives the fold (one treatment per class, no per-term special-casing) AND the breakdown.
    # PBRS fields already hold γ·Φ(s′)−Φ(s); BIAS fields hold the additive per-turn value;
    # `bias_refund` is the fold mechanism, not a term (excluded). Coverage is exhaustive + 1:1. ----
    _REGISTRY: ClassVar[dict] = {
        "win_loss": RewardClass.TERMINAL,
        "pbrs_material": RewardClass.PBRS,
        "pbrs_belief": RewardClass.PBRS,
        # everything else is BIAS:
        "explosion": RewardClass.BIAS, "explosion_block": RewardClass.BIAS,
        "finishing_blow": RewardClass.BIAS, "roar": RewardClass.BIAS,
        "futile_attack": RewardClass.BIAS, "futile_setup": RewardClass.BIAS,
        "setup_low_hp": RewardClass.BIAS, "boost_utilized": RewardClass.BIAS,
        "status_wasted": RewardClass.BIAS, "spikes": RewardClass.BIAS,
        "futile_spikes": RewardClass.BIAS, "matchup_penalty": RewardClass.BIAS,
        "dead_matchup_tax": RewardClass.BIAS, "stay_risk_tax": RewardClass.BIAS,
        "escape_risk_bonus": RewardClass.BIAS, "switch_base": RewardClass.BIAS,
        "switch_bouncing_tax": RewardClass.BIAS, "repetition_tax": RewardClass.BIAS,
        "struggle_tax": RewardClass.BIAS, "pivot_protect": RewardClass.BIAS,
        "pivot_status": RewardClass.BIAS, "pivot_damage": RewardClass.BIAS,
        "se_switch": RewardClass.BIAS, "escape_threat_switch": RewardClass.BIAS,
        "sleep_out": RewardClass.BIAS, "sleep_in": RewardClass.BIAS,
        "status": RewardClass.BIAS, "stall_tax": RewardClass.BIAS,
        "no_progress_tax": RewardClass.BIAS,
    }

    # Groups ordered by how frequently they produce non-zero values (for the compact string only).
    _GROUPS: ClassVar[tuple] = (
        ("base",   ("win_loss", "pbrs_material", "explosion", "explosion_block", "finishing_blow")),
        ("attack", ("roar", "futile_attack", "futile_setup", "setup_low_hp",
                    "boost_utilized", "status_wasted", "repetition_tax", "struggle_tax")),
        ("switch", ("switch_base", "switch_bouncing_tax", "escape_threat_switch",
                    "escape_risk_bonus", "pivot_protect", "pivot_status",
                    "pivot_damage", "se_switch", "sleep_out", "sleep_in")),
        ("field",  ("spikes", "futile_spikes", "matchup_penalty", "dead_matchup_tax",
                    "stay_risk_tax", "status", "stall_tax", "no_progress_tax")),
        ("shaping", ("pbrs_belief", "bias_refund")),
    )

    @classmethod
    def registry_fields(cls, reward_class: "RewardClass") -> tuple:
        """The breakdown fields belonging to ``reward_class`` (the registry is the source of truth)."""
        return tuple(name for name, c in cls._REGISTRY.items() if c is reward_class)

    @property
    def total(self) -> float:
        """Sum every registry term + the bias_refund mechanism. Equivalent to summing all
        dataclass float fields (the registry covers them 1:1, `bias_refund` is summed too)."""
        return sum(getattr(self, f.name) for f in fields(self)
                   if isinstance(getattr(self, f.name), float))

    def to_dict(self) -> dict:
        """Grouped, compact JSON dict.

        Each category becomes a single string of 'key=±value' pairs for non-zero fields.
        Empty categories are omitted. 'total' is always present.

        Example:
            {'total': 0.06, 'base': 'pbrs_material=-0.44',
             'switch': 'switch_base=+0.50 se_switch=+0.20 pivot_damage=+0.10'}
        """
        result: dict = {"total": round(self.total, 4)}
        for group_name, group_fields in self._GROUPS:
            parts = []
            for fname in group_fields:
                v = getattr(self, fname)
                if v != 0.0:
                    parts.append(f"{fname}={v:+.4g}")
            if parts:
                result[group_name] = " ".join(parts)
        return result


HP_VALUE = 2.0
VICTORY_VALUE = 30.0
FINISHING_BLOW_BONUS = 0.5   # extra bonus for KO'ing with a damaging move

# --- Material PBRS Φ_mat (design §2). Φ_mat = MAT_HP_WEIGHT·(Σ our_hp − Σ opp_hp)
#     + MAT_ALIVE_WEIGHT·(n_alive_ours − n_alive_opp), over the DECLARED team size (unrevealed opp
#     mons count as full-HP-alive → Φ_mat(s_0)≈0, no opp-reveal jumps, no start-state variance).
#     Φ_mat(terminal)=0 → telescopes to −Φ_mat(s_0) → every win returns +30, every loss −30. ---
MAT_HP_WEIGHT = HP_VALUE          # 2.0 — reproduces the old hp_ours/hp_opp per-turn density exactly
MAT_ALIVE_WEIGHT = 1.25           # = old FAINT_BASE(0.5)+FAINT_MATERIAL_PENALTY(0.75): matches the old
                                  # non-HP immediate faint magnitude (stated invariant, design §2.4). The
                                  # old −0.75 preservation BIAS is REMOVED — the +30 + Φ_mat's dense
                                  # material density teach preservation without an objective bias.

# --- BIAS-additivity (design §1.2). PBRS_GAMMA already == the PPO gamma (asserted in train_rl_agent).
# The no-progress penalty magnitude lives on RewardConfig.no_progress_penalty (single source of truth);
# the turns_since_progress cap lives on progress_clock.PROGRESS_CLOCK_CAP (the obs + clock owner). ---

# Progressive stall tax — kept GENTLE (design §4.3): the progress clock is offense-centric and cannot
# see DEFENSIVE stalls (heal/Protect wars), so a soft absolute-turn term covers them. Re-tuned to a
# ~−10 ceiling (was an integral of −21.3). Starts later + ramps slower than the old version.
STALL_TAX_START_TURN = 100      # was 60 — push the soft pressure later so a normal mid-length game pays ~0
STALL_TAX_PER_TURN = 0.02       # base rate; multiplied by the ramp fraction below (was 0.05)
STALL_TAX_RAMP_TURNS = 40       # turns-past-start over which the rate ramps up by 1× (was 20)
STALL_TAX_MAX = 0.15            # per-turn clamp (was 0.5) — keeps the cumulative ~−10 to the 250 forfeit
STRUGGLE_LOOP_TAX = -0.5
STRUGGLE_LOOP_THRESHOLD = 3

SWITCH_BASE_BONUS = 0.5        # flat per-voluntary-switch bonus
STATUS_BONUS = 0.3             # reward for inflicting status; penalty for receiving
ROAR_BONUS = 0.2               # reward for Roar when spikes on opp side or opp had positive boosts
SE_SWITCH_BONUS = 0.2          # reward for switching in a mon with a SE damaging move vs opp active
SLEEP_SWAP_BONUS = 0.25        # reward for rotating a sleeping mon out; penalty for rotating one in
SPIKES_LAYER_BONUS = 0.5       # per layer added to opponent's side (credit assignment bridge)
SPIKES_WASTE_PENALTY = -0.2    # wasted turn using Spikes when 3 layers already up
FAILED_ROAR_PENALTY = -0.2     # Roar used but opponent didn't switch
FUTILE_ATTACK_PENALTY = -0.05  # attacking move used but opponent net gained HP (Leftovers > damage)
FUTILE_IMMUNE_PENALTY = -0.5   # flat per-turn penalty for attacking into a type immunity
                               # (our_effectiveness == 0.0). The ESCALATION on a repeated
                               # immune attack comes from the zero-effect repetition tax below.
ESCAPE_THREAT_BONUS = 0.25     # voluntarily switching out while opp threatens us (revealed SE OR belief)
MATCHUP_PENALTY = -0.15        # per turn we stay in while opp threatens us (revealed SE OR belief)

# --- Belief-based switch shaping (design_reward_switching.md) ---
# Fix for the confirmed under-switch pathology: the policy switches LESS as the incoming-KO belief
# rises (the obs has the signal + the critic reads it, but the reward never pushed switching for the
# damage-magnitude / unrevealed / prior-based threats — only revealed-SE). Two parts:
#  (A) re-gate ESCAPE_THREAT_BONUS / MATCHUP_PENALTY on the incoming-KO belief (OR-ed with the old
#      revealed-SE gate), so they fire on the threats they used to miss; and
#  (B) potential-based reward shaping (Ng 1999) over the belief — policy-invariant, bridges the
#      credit-assignment gap by bringing the avoided-faint benefit forward to the switch decision.
# NEVER touches the ±30 terminal: PBRS uses Phi(terminal)=0, contributing only a policy-invariant
# constant to the return.
PBRS_RISK_WEIGHT = 2.0         # potential weight; a full-HP mon at certain imminent KO ≈ -2.0 in Phi
                              # (between FAINT_MATERIAL_PENALTY 0.75 and the full faint cost ~-3.25)
PBRS_GAMMA = 0.9999           # MUST equal the PPO gamma (train_rl_agent default) for policy-invariance
SWITCH_RISK_THRESHOLD = 0.5   # belief P(KO)·(1-P(outspeed)) above which the active mon counts as
                              # "threatened" for the escape/stay re-gate

# --- Belief-risk-scaled switch BIAS (the "stay-into-KO" lever; design_reward_switching.md §7) ---
# The shipped `pbrs_belief` is policy-INVARIANT (a potential difference telescoping to −Φ(s_0)), so it
# CANNOT move a converged under-switch preference — verified on run_20260607_102632: switch-mass still
# inverts vs P(KO) and the stay-and-die rate is unchanged vs the V1 control. These two BIAS-class terms
# DO change the objective: a stay-tax that prices staying in a high imminent-KO spot when a safer pivot
# exists, and a symmetric (smaller) escape reward, both SCALED by the calibrated incoming-KO belief.
# Gated by RewardConfig.switch_bias_weight (default 0.0 = OFF → the default single-variable run is
# byte-unchanged). Being BIAS-class, they also ride --bias-additivity, so a λ=1 vs λ=0 A/B on a fixed
# weight isolates whether it is the *bias* (objective tilt), not merely the magnitude, that helps.
SAFE_PIVOT_PKO_MAX = 0.35      # a non-fainted bench mon whose incoming P(KO) ≤ this is a viable "safe
                              # pivot": the stay-tax fires ONLY when one exists, so a forced stay
                              # (no bench, or every bench-in also dies) is never penalised. Bench risk
                              # uses RAW P(KO) (a switch-in always eats the turn's hit; speed can't save
                              # it that turn), unlike the active's outspeed-discounted risk.
STAY_RISK_TAX_FLOOR = -2.0     # per-turn clamp so one stay-tax can't dwarf the faint (~-3.25) / ±30
ESCAPE_RISK_FRACTION = 0.5     # escape reward = weight·fraction·risk_escaped — deliberately ASYMMETRIC
                              # (< the stay-tax) so there is no positive-reward surface to bounce-farm
                              # (the escalating switch_bouncing_tax + Φ_mat HP loss also guard it)
PROTECT_SWITCH_BONUS = 0.10    # opponent used Protect/Detect/Endure on our switch turn
STATUS_IMMUNE_SWITCH_BONUS = 0.10  # our switch-in was immune to their status move

FUTILE_SETUP_PENALTY = -0.3
SETUP_LOW_HP_THRESHOLD = 0.40      # HP fraction below which setup is penalised
SETUP_LOW_HP_MAX_PENALTY = -0.10   # penalty at 0% HP; scales linearly to 0 at threshold
STATUS_WASTED_PENALTY = -0.3
BOOST_UTILIZED_SCALE = 0.03        # reward = boost_stage * scale * damage_dealt
EXPLOSION_BLOCK_BONUS = 1.0        # Ghost immune or Protect blocks opponent Explosion

# Repetition tax escalation — LINEAR and UNCAPPED (clamped only by the floor).
# A 12-30 turn spam must be catastrophic, not a rounding error, so the cost grows
# every consecutive turn instead of plateauing after the 4th repeat. The tax for the
# n-th consecutive repeat is max(-STEP * n, FLOOR). A "no-op" repeat (the move did
# nothing productive — no damage, no boost gained, no status landed, no hazard added)
# uses the much steeper ZERO_EFFECT step so capped setup (Calm Mind past +6), capped
# hazards (Spikes at 3), redundant status, Protect/Wish/Recover loops, and immune
# attacks all bite hard and fast. A legitimately-productive repeat (still dealing
# damage or still gaining a boost) only pays the gentle normal step.
REPETITION_STEP = 0.03              # normal productive-attack repeat, per consecutive turn
REPETITION_ZERO_EFFECT_STEP = 0.15  # no-op / immune / capped repeat — bites hard
REPETITION_TAX_FLOOR = -3.0         # per-turn clamp so one turn can't dwarf win/loss

# Switch-bouncing tax — ESCALATING (was a flat -0.15). A→B→A→B oscillation dodges the
# move-repetition tax because the action index alternates, so it needs its own
# escalating counter. The n-th consecutive bounce costs max(STEP * n, FLOOR).
BOUNCING_TAX_STEP = -0.15
BOUNCING_TAX_FLOOR = -2.0

# Dead-matchup tax — fires when the active Pokémon has NO damaging move with >0×
# effectiveness vs the opponent's active mon and we DID NOT switch out. This is the
# "trapped, must pivot" signal: the matchup re-ranks moves but can't lift switches
# above the collapsed "stay in and click" prior, so we make staying strictly worse
# than pivoting and escalate it every turn we refuse to leave.
DEAD_MATCHUP_TAX_STEP = -0.10
DEAD_MATCHUP_TAX_FLOOR = -2.0

BOOST_MOVES: frozenset[str] = frozenset({
    "calmmind", "dragondance", "swordsdance", "nastyplot",
    "agility", "rockpolish", "bulkup", "cosmicpower",
    "acidarmor", "barrier", "irondefense", "amnesia",
    "growth", "meditate", "sharpen", "doubleteam", "minimize",
    "harden", "withdraw", "defensecurl", "stockpile",
})

STATUS_INFLICTING_MOVES: frozenset[str] = frozenset({
    "toxic", "poisonpowder",
    "thunderwave",
    "willowisp",
    "sleeppowder", "hypnosis", "spore", "lovelykiss", "sing",
})


class Gen3RewardManager:
    """
    Self-contained reward calculator for the Gen 3 RL trainee agent.

    Owns the full reward pipeline:
      - record_action()       — tracks the action choice and computes switch subsidy
      - process_turn_reward() — computes the full turn reward from the TurnDelta + the
                                current-board LiveView (built once per turn)

    Satisfies the RewardFunction protocol. Must only be called for the trainee's
    battle — the env gates this at the call site.
    """
    def __init__(self, log_level: LogLevel = LogLevel.QUIET,
                 config: Optional[RewardConfig] = None, progress_clock=None):
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
        self._prev_active_ko_risk: float = 0.0
        self._prev_safe_pivot: bool = False              # did a safe bench pivot exist at decision time
        self._cur_can_switch: bool = True                # was a switch LEGAL at this decision (mask) —
                                                         # gates the stay-tax so a trapped stay isn't taxed
        self._bias_acc: float = 0.0                      # Σ BIAS-class contributions this episode

    def reset(self):
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
        self._prev_phi_belief = None
        self._prev_phi_mat = None
        self._prev_active_ko_risk = 0.0
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
        bouncing_tax = 0.0
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

        spam_mult = 1.0 if (decision_turn - self.last_switch_turn) > 1 else 0.0
        bd.switch_base = SWITCH_BASE_BONUS * spam_mult

        if self._prev_opp_se_threat or self._prev_active_ko_risk >= SWITCH_RISK_THRESHOLD:
            bd.escape_threat_switch = ESCAPE_THREAT_BONUS

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

    def _compute_roar_bonus(self, delta: TurnDelta, live) -> float:
        """Reward Roar when it forces a switch AND spikes are up or opp had positive boosts.
        Penalise Roar when it fails to force any switch at all (wasted turn)."""
        if delta.our_move_id != "roar":
            return 0.0
        if delta.opp_switch_to is None:
            return FAILED_ROAR_PENALTY
        has_spikes = self._opp_spikes(live) > 0
        had_boosts = any(v > 0 for v in self._prev_opp_boosts.values())
        return ROAR_BONUS if (has_spikes or had_boosts) else 0.0

    def _compute_se_switch_bonus(self, delta: TurnDelta, live) -> float:
        """Reward switching in a mon that threatens the opponent with a SE move.

        First checks revealed moves for confirmed SE; if none are revealed yet,
        falls back to checking whether any of our mon's own types are SE vs the
        opponent (a reliable proxy for STAB moves in Gen 3 OU). Reads current-board
        state through the LiveView ``LivePokemon`` (move power/type via ``gen3_movedex``,
        effectiveness via the mechanics primitive).
        """
        if delta.our_switch_to is None:
            return 0.0
        our_mon = live.ours.active
        opp_mon = live.opp.active
        if not our_mon or not opp_mon:
            return 0.0

        # Only award on voluntary switches; forced post-faint replacements don't count
        if self._last_reward_metadata.get("type") != "VOLUNTARY":
            return 0.0
        # Opponent must be alive at switch-in
        if opp_mon.fainted:
            return 0.0

        # Gate: only fire if the opponent has switched since this mon was last in.
        # Same matchup without opponent switching = bonus already spent for this matchup.
        our_species = our_mon.species
        opp_species = opp_mon.species
        if self._last_opp_seen_by.get(our_species) == opp_species:
            return 0.0

        # Confirmed SE via revealed move
        for mid in our_mon.move_ids:
            md = _movedex.get(mid)
            if md is None or md.base_power <= 0:
                continue
            if self._live_eff_mult(md.type, opp_mon) >= 2.0:
                return SE_SWITCH_BONUS
        # Fallback: STAB type advantage (no moves revealed yet)
        for t in our_mon.types:
            if self._live_eff_mult(_ptype(t), opp_mon) >= 2.0:
                return SE_SWITCH_BONUS
        return 0.0

    def _compute_status_reward(self, delta: TurnDelta, live) -> tuple[float, int]:
        """One-time reward when the statused-mon count changes on either side.
        Returns (reward, d_opp) where d_opp is the delta in opponent statused count."""
        our_statused, opp_statused = self._statused_counts(live)
        d_our = our_statused - self._prev_our_statused
        d_opp = opp_statused - self._prev_opp_statused
        self._prev_our_statused = our_statused
        self._prev_opp_statused = opp_statused
        return (d_opp - d_our) * STATUS_BONUS, d_opp

    # =========================================================
    # SWITCH REWARDS
    #
    # All switch-specific signals funnel through
    # _compute_all_switch_bonuses, which dispatches to one
    # focused sub-function per outcome type.  The subsidy set
    # by record_action() is applied separately at the end of
    # process_turn_reward (it runs before the turn, not after).
    # =========================================================

    def _compute_pivot_bonus(self, delta: TurnDelta, live) -> tuple[float, float, float]:
        """Return (protect_bonus, status_bonus, damage_bonus) for this switch turn.

        Uses `delta.opp_resolved_move_id` — protocol-truth attribution when a
        damaging event is set, falling back to the inferred `delta.opp_move_id`
        for non-damaging moves (status, Roar, etc.). Avoids the stale-last_move
        class of bug that bit HP attribution. The opp-move presence + power read uses
        the LiveView active mon's revealed moves + ``gen3_movedex``.
        """
        opp_move_id = delta.opp_resolved_move_id
        if delta.opp_switch_to is not None or opp_move_id is None:
            return (0.0, 0.0, 0.0)

        if opp_move_id in _INVULNERABLE_MOVES:
            return (self._pivot_protect_bonus(), 0.0, 0.0)

        opp_mon = live.opp.active
        if not opp_mon or opp_move_id not in opp_mon.move_ids:
            return (0.0, 0.0, 0.0)
        md = _movedex.get(opp_move_id)
        base_power = md.base_power if md is not None else 0

        if base_power == 0:
            return (0.0, self._pivot_status_bonus(opp_move_id, live), 0.0)
        return (0.0, 0.0, self._pivot_damage_bonus(opp_move_id, delta, live))

    def _pivot_protect_bonus(self) -> float:
        """Opponent used Protect/Detect/Endure — we repositioned for free."""
        return PROTECT_SWITCH_BONUS

    def _pivot_status_bonus(self, opp_move_id: str, live) -> float:
        """Opponent used a status move our switch-in was immune to (type or already statused)."""
        new_mon = live.ours.active
        if not new_mon:
            return 0.0
        if self._live_status_move_immune(opp_move_id, new_mon):
            return STATUS_IMMUNE_SWITCH_BONUS
        return 0.0

    def _pivot_damage_bonus(self, opp_move_id, delta: TurnDelta, live) -> float:
        """Opponent used a damaging move — bonus if it hit our new mon less than the old one.

        Signal A: comparison of actual type effectiveness vs old mon vs new mon.
        The raw HP delta is already penalised by hp_ours, so this signal focuses purely
        on whether the switch improved the matchup.

        `mult_vs_new` prefers `delta.opp_damaging_event.effectiveness` when the
        event's target matches our switch-in — it's the protocol-confirmed bucket
        (no drift if our local mechanics disagree with Showdown's). Falls back
        to a local recompute when the event is missing or hit a different
        target (e.g. opp damaged prev_active before switch-in was on field).
        `mult_vs_old` always recomputes — the protocol can't tell us what the
        multiplier *would have been* if we hadn't switched. The move type comes from
        ``gen3_movedex`` and the prev/new mons from the LiveView.
        """
        new_mon = live.ours.active
        if not new_mon:
            return 0.0
        prev_mon = live.ours.get(delta.our_prev_active)
        if prev_mon is None:
            return 0.0
        md = _movedex.get(opp_move_id)
        move_type = md.type if md is not None else None
        opp_event = delta.opp_damaging_event
        if opp_event is not None and opp_event.target_species == new_mon.species:
            mult_vs_new = opp_event.effectiveness
        else:
            mult_vs_new = self._live_eff_mult(move_type, new_mon)
        mult_vs_old = self._live_eff_mult(move_type, prev_mon)
        if mult_vs_new < mult_vs_old:
            return 0.15 if mult_vs_new == 0 else 0.10
        return 0.0

    def _compute_sleep_out_bonus(self, delta: TurnDelta, live) -> float:
        """Reward rotating a sleeping mon to the bench on a voluntary switch.
        Preserving a sleeping mon's PP/position has strategic value; post-faint
        replacements don't qualify since there's no choice involved."""
        if self._last_reward_metadata.get("type") != "VOLUNTARY":
            return 0.0
        prev = live.ours.get(delta.our_prev_active)
        if prev is None:
            return 0.0
        return SLEEP_SWAP_BONUS if prev.status == "slp" else 0.0

    def _compute_sleep_in_penalty(self, delta: TurnDelta, live) -> float:
        """Penalise sending in a sleeping mon — it can't act and wastes a slot.
        Applies to voluntary switches and post-faint replacements; skipped for roar
        since the phazer chose our slot, not us."""
        if self._last_switch_was_roared:
            return 0.0
        our_mon = live.ours.active
        if our_mon and our_mon.status == "slp":
            return -SLEEP_SWAP_BONUS
        return 0.0

    def _compute_matchup_penalty(self, delta: TurnDelta) -> float:
        """Per-turn penalty for staying in while the opp threatened us last turn — a revealed SE
        move OR the incoming-KO belief (P(KO)·(1-outspeed) ≥ threshold). Uses last-turn's snapshot
        so we only penalise for threats known at decision time. The belief OR-gate lights this up
        for the damage-magnitude / unrevealed / prior-based threats the revealed-SE gate misses."""
        if delta.our_switch_to is not None:
            return 0.0  # we switched out — no staying-in penalty
        threatened = self._prev_opp_se_threat or self._prev_active_ko_risk >= SWITCH_RISK_THRESHOLD
        return MATCHUP_PENALTY if threatened else 0.0

    def _compute_stay_risk_tax(self, delta: TurnDelta) -> float:
        """Belief-risk-scaled penalty for STAYING into a high imminent-KO spot when a safe pivot was
        available (the under-switch lever; design_reward_switching.md §7). OFF unless
        ``switch_bias_weight > 0``. Keyed entirely on DECISION-TIME snapshots
        (``_prev_active_ko_risk`` / ``_prev_safe_pivot``, set at the end of last turn from the board
        the policy then acted on), so it prices the choice the policy actually made.

        Fires only when ALL hold: a switch was LEGAL this decision (``_cur_can_switch`` — never tax a
        TRAPPED stay, the key false-positive guard); we did NOT switch (``our_switch_to is None``); the
        move did NOT fizzle to RNG (``not our_failed_to_move`` — flinch / full-para / sleep / freeze is
        not a deliberate stay, mirroring the progress-clock's FREEZE); the chosen move did NOT KO the
        opponent (``not opp_fainted`` — then staying won the exchange, not a misplay); the decision-time
        KO-risk was high (``>= SWITCH_RISK_THRESHOLD``); and a safe bench pivot existed
        (``_prev_safe_pivot``). Staying-and-fainting IS included (the exact pathology). Tax scales with
        the risk, clamped at ``STAY_RISK_TAX_FLOOR``. Unlike ``pbrs_belief`` (policy-invariant) this is
        an additive BIAS → it changes the objective the actor optimises, which is the point."""
        w = self.config.switch_bias_weight
        if w <= 0.0:
            return 0.0
        if (delta.our_switch_to is not None or delta.opp_fainted
                or delta.our_failed_to_move or not self._cur_can_switch):
            return 0.0
        if self._prev_active_ko_risk < SWITCH_RISK_THRESHOLD or not self._prev_safe_pivot:
            return 0.0
        return max(-w * self._prev_active_ko_risk, STAY_RISK_TAX_FLOOR)

    def _compute_dead_matchup_tax(self, delta: TurnDelta, live) -> float:
        """Escalating penalty for refusing to pivot out of a 0×-only matchup.

        Fires when EVERY damaging move our active Pokémon has does 0× to the
        opponent's active mon (e.g. an Electric attacker staring at a Ground type,
        a Normal attacker into a Ghost) and we chose to stay in rather than switch.
        The per-turn cost grows with how many consecutive turns we've stayed
        trapped, so a switch — which resets the counter to zero — strictly
        dominates clicking another useless move.

        Resets (and charges nothing) whenever we switch, faint, lack a live
        opponent, or have at least one >0× damaging move. Skips forced-switch
        slots entirely (we had no move choice there). Requires at least one
        revealed damaging move to judge — our own active mon's full moveset is
        populated from the request, so this is reliable for the trainee's mon.
        """
        if delta.our_switch_to is not None or delta.we_fainted:
            self._consecutive_dead_matchup_stays = 0
            return 0.0
        if delta.phase_is_forced_switch:
            return 0.0

        our_mon = live.ours.active
        opp_mon = live.opp.active
        if not our_mon or not opp_mon or opp_mon.fainted:
            self._consecutive_dead_matchup_stays = 0
            return 0.0
        damaging = [
            md for md in (_movedex.get(mid) for mid in our_mon.move_ids)
            if md is not None and md.base_power > 0
        ]
        if not damaging:
            self._consecutive_dead_matchup_stays = 0
            return 0.0
        best_mult = max(self._live_eff_mult(md.type, opp_mon) for md in damaging)
        if best_mult > 0.0:
            self._consecutive_dead_matchup_stays = 0
            return 0.0

        # Every damaging option is type-immune and we stayed in — escalate.
        self._consecutive_dead_matchup_stays += 1
        return max(DEAD_MATCHUP_TAX_STEP * self._consecutive_dead_matchup_stays,
                   DEAD_MATCHUP_TAX_FLOOR)

    # =========================================================
    # CURRENT-BOARD ACCESSORS
    #
    # These read current-board facts through the vetted LiveView
    # read-model (built once per turn in process_turn_reward) so
    # the reward manager never reaches into the raw poke-env battle
    # for "what is true now".
    # =========================================================

    def _opp_spikes(self, live) -> int:
        """Spikes layers on the opponent's side (0-3)."""
        return live.opp.side_conditions.get("spikes", 0)

    def _statused_counts(self, live) -> tuple[int, int]:
        """(our, opp) counts of non-fainted mons carrying a status condition."""
        our = sum(1 for m in live.ours.mons if m.status is not None and not m.fainted)
        opp = sum(1 for m in live.opp.mons if m.status is not None and not m.fainted)
        return our, opp

    def _opp_active_boosts(self, live) -> dict:
        """Opponent active mon's current stat-stage boosts ({} if none on field)."""
        return dict(live.opp.active.boosts) if live.opp.active else {}

    def _terminal(self, live) -> tuple:
        """(won, lost, finished) — terminal battle flags."""
        return live.won, live.lost, live.finished

    def _live_eff_mult(self, move_type, live_mon) -> float:
        """``effective_multiplier`` for a LiveView ``LivePokemon``, via the primitive
        ``effective_multiplier_by_types`` (no poke-env ``Pokemon``). Byte-identical to
        the raw-battle path: the LivePokemon carries the same types/ability/status the
        raw ``mon`` would expose, just in id-string form."""
        types = live_mon.types
        t1 = _ptype(types[0]) if types else None
        t2 = _ptype(types[1]) if len(types) > 1 else None
        return _effective_multiplier_by_types_fn(
            move_type, t1, t2, live_mon.ability, _status_enum(live_mon.status)
        )

    def _live_status_move_immune(self, move_id, live_mon) -> bool:
        """``is_status_move_immune`` for a LiveView ``LivePokemon`` — type immunity to
        the status the move inflicts, or the mon already carrying a status."""
        immune_types = _STATUS_MOVE_IMMUNITY.get(move_id, frozenset())
        mon_types = {_ptype(t) for t in live_mon.types}
        return bool(immune_types & mon_types) or live_mon.status is not None

    def _update_opp_se_threat(self, live) -> None:
        """Snapshot whether opp active has a revealed SE move vs our active, for next turn."""
        our_mon = live.ours.active
        opp_mon = live.opp.active
        if not our_mon or not opp_mon:
            self._prev_opp_se_threat = False
            return
        for mid in opp_mon.move_ids:
            md = _movedex.get(mid)
            if md is not None and md.base_power > 0:
                if self._live_eff_mult(md.type, our_mon) >= 2.0:
                    self._prev_opp_se_threat = True
                    return
        self._prev_opp_se_threat = False

    def _belief_potential_and_risk(self, live) -> tuple[float, float, float]:
        """The incoming-KO belief potential Φ(s), the active mon's imminent KO-risk, and the safest
        bench mon's incoming P(KO) — all from the LiveView read-model (one belief encode, no raw
        battle).

        Φ = PBRS_RISK_WEIGHT · ( Σ_alive hp_frac  −  hp_active · risk_active ), where
        risk_active = max(phys_pko, spec_pko)·(1 − P(outspeed)) for the on-field mon. This is
        **expected surviving material**: a benched mon counts its full HP (it isn't hit this turn);
        the active mon is discounted by its imminent KO chance. So switching a doomed active to the
        bench RAISES Φ (its HP stops being discounted) → positive shaping AT the switch decision,
        while a faint never raises Φ (it removes a positive contribution). Φ is a pure function of
        the board state → policy-invariant under PBRS.

        Returns ``(phi, active_risk, min_bench_pko)``. ``active_risk`` is the outspeed-discounted
        KO-risk snapshot for the escape/stay re-gate. ``min_bench_pko`` is the LOWEST RAW incoming
        P(KO) across our non-fainted bench mons (1.0 if none) — raw, not outspeed-discounted, because
        a switch-in always eats the turn's hit (speed can't save it that turn). It feeds the
        safe-pivot gate of the belief-risk stay-tax (a stay is only taxed when a pivot would survive)."""
        block = _encode_incoming_block(live)
        mons = live.ours.mons
        total_hp = 0.0
        active_imminent = 0.0
        active_risk = 0.0
        min_bench_pko = 1.0
        for i, lm in enumerate(mons[:_TEAM_SIZE]):
            if lm is None or lm.fainted:
                continue
            hp = float(lm.hp_fraction)
            total_hp += hp
            base = i * _INCOMING_PER_MON
            pko = max(float(block[base + 2]), float(block[base + 3]))
            if lm.active:
                outspeed = float(block[base + 4])
                active_risk = pko * (1.0 - outspeed)
                active_imminent = hp * active_risk
            elif hp > 0.0:   # a real, switchable bench mon (guard a 0-HP-not-yet-fainted false "safe")
                min_bench_pko = min(min_bench_pko, pko)
        phi = PBRS_RISK_WEIGHT * (total_hp - active_imminent)
        phi = max(0.0, min(phi, PBRS_RISK_WEIGHT * _TEAM_SIZE))
        return phi, active_risk, min_bench_pko

    def _compute_phi_mat(self, live) -> float:
        """The material potential Φ_mat(s) (design §2.2), from the LiveView read-model.

        Φ_mat = MAT_HP_WEIGHT·(Σ our_hp_frac − Σ opp_hp_frac) + MAT_ALIVE_WEIGHT·(n_alive_ours −
        n_alive_opp), summed over the **declared team size** — unrevealed opponent mons count as
        full-HP-alive. So Φ_mat(s_0)≈0 (6−6 HP, 6−6 alive), there are no opp-reveal discontinuities
        (the opp sum only ever decreases from a 6.0 baseline), and the per-episode telescoping
        constant −Φ_mat(s_0)≈0 with near-zero cross-episode variance. A pure function of the board
        (HP + alive set) → policy-invariant under PBRS.
        """
        our = live.ours
        opp = live.opp
        # No known mons → neutral (production always has all 6 from turn 1; this guards mock/standalone
        # paths where the team list is empty, so Φ_mat doesn't read a degenerate 0-vs-6 state).
        if not our.mons:
            return 0.0
        # Our team is fully known: sum HP over the (≤6) known mons, alive = non-fainted.
        our_hp = sum(float(m.hp_fraction) for m in our.mons[:_TEAM_SIZE] if not m.fainted)
        our_alive = sum(1 for m in our.mons[:_TEAM_SIZE] if not m.fainted)
        # Opp: revealed mons carry their real HP/alive; unrevealed declared slots count full-HP-alive.
        opp_team_size = getattr(opp, "team_size", None) or _TEAM_SIZE
        opp_team_size = min(int(opp_team_size), _TEAM_SIZE)
        opp_revealed = list(opp.mons[:_TEAM_SIZE])
        opp_hp = sum(float(m.hp_fraction) for m in opp_revealed if not m.fainted)
        opp_alive = sum(1 for m in opp_revealed if not m.fainted)
        n_unrevealed = max(0, opp_team_size - len(opp_revealed))
        opp_hp += n_unrevealed * 1.0     # unrevealed → full HP
        opp_alive += n_unrevealed        # unrevealed → alive
        alive_w = self.config.mat_alive_weight
        phi = MAT_HP_WEIGHT * (our_hp - opp_hp) + alive_w * (our_alive - opp_alive)
        bound = MAT_HP_WEIGHT * _TEAM_SIZE + alive_w * _TEAM_SIZE
        return max(-bound, min(phi, bound))

    def _compute_spikes_bonus(self, delta: TurnDelta, live) -> tuple[float, float]:
        """Return (layer_bonus, futile_waste). The layer-added credit is the BIAS term `spikes`
        (its telescoping form is Φ_hazard, reached at bias_additivity→0 — design §2.6); the
        wasted-Spikes-at-cap penalty is split out into the Markovian `futile_spikes` term."""
        curr = self._opp_spikes(live)
        new_layers = curr - self._prev_opp_spikes
        self._prev_opp_spikes = curr
        if new_layers > 0:
            return new_layers * SPIKES_LAYER_BONUS, 0.0
        if delta.our_move_id == "spikes" and curr == 3:
            return 0.0, SPIKES_WASTE_PENALTY
        return 0.0, 0.0

    def _compute_futile_attack_penalty(self, delta: TurnDelta, live) -> float:
        """Penalise attacking moves where the opponent's total HP went up or stayed even
        (Leftovers healed as much or more than we dealt). Skips status moves, switches,
        and cases where we failed to act or the opponent used Rest."""
        if delta.our_move_id is None:
            return 0.0  # we switched
        if delta.our_failed_to_move:
            return 0.0  # paralysis / sleep — not our fault
        if delta.opp_switch_to is not None:
            return 0.0  # they switched; HP delta is noisy (fresh mon entering)
        # Rest detection uses opp_resolved_move_id (protocol-truth when an
        # event fired). Raw delta.opp_move_id could be a stale "rest" from
        # an earlier turn after opp switched between snapshots, mis-skipping
        # the penalty on a normal turn where they didn't actually rest.
        if delta.opp_resolved_move_id == "rest":
            return 0.0  # opponent used Rest; large self-heal is expected
        # Damaging-move gate: our move must be a revealed damaging move (power via gen3_movedex).
        our_mon = live.ours.active
        md = (_movedex.get(delta.our_move_id)
              if our_mon and delta.our_move_id in our_mon.move_ids else None)
        if md is None or md.base_power == 0:
            return 0.0  # status or utility move — handled by other signals
        # Type immunity: 0 damage by definition — use the harder penalty.
        if delta.our_effectiveness == 0.0:
            return FUTILE_IMMUNE_PENALTY
        # Net HP sum across all opp slots: bench mons don't change between turns in Gen 3,
        # so the sum is dominated by the active slot. >= 0 means we made no net progress.
        if delta.opp_hp_delta.sum() >= 0:
            return FUTILE_ATTACK_PENALTY
        return 0.0

    def _compute_futile_setup_penalty(self, delta: TurnDelta) -> float:
        """Penalise using a stat-boosting move when already at the ±6 cap."""
        if delta.our_move_id not in BOOST_MOVES:
            return 0.0
        if delta.our_failed_to_move or delta.we_fainted:
            return 0.0
        # If no boost stage changed, the move had zero mechanical effect
        if delta.our_boost_delta.sum() == 0:
            return FUTILE_SETUP_PENALTY
        return 0.0

    def _compute_setup_low_hp_penalty(self, delta: TurnDelta) -> float:
        """Penalise choosing a setup move below 40% HP."""
        if delta.our_move_id not in BOOST_MOVES:
            return 0.0
        if delta.our_failed_to_move or delta.we_fainted:
            return 0.0
        hp = self._our_active_hp_before
        if hp >= SETUP_LOW_HP_THRESHOLD:
            return 0.0
        return SETUP_LOW_HP_MAX_PENALTY * (1.0 - hp / SETUP_LOW_HP_THRESHOLD)

    def _compute_status_wasted_penalty(self, delta: TurnDelta, d_opp_statused: int) -> float:
        """Penalise status-inflicting moves that produced no status event."""
        if delta.our_move_id not in STATUS_INFLICTING_MOVES:
            return 0.0
        if delta.our_failed_to_move:
            return 0.0
        if delta.opp_switch_to is not None:
            return 0.0  # opp switched; ambiguous
        if d_opp_statused > 0:
            return 0.0  # status landed — no penalty
        return STATUS_WASTED_PENALTY

    def _compute_boost_utilized(self, delta: TurnDelta, live) -> float:
        """Reward attacking moves that leverage active stat boosts."""
        if delta.our_move_id is None or delta.our_switch_to is not None:
            return 0.0
        if delta.our_failed_to_move:
            return 0.0
        mon = live.ours.active
        md = (_movedex.get(delta.our_move_id)
              if mon and delta.our_move_id in mon.move_ids else None)
        if md is None or md.base_power == 0:
            return 0.0
        # Use the higher of atk (idx 0) or spa (idx 2) boost
        effective_boost = max(int(self._our_boosts_before[0]), int(self._our_boosts_before[2]))
        if effective_boost <= 0:
            return 0.0
        damage_dealt = max(0.0, -float(delta.opp_hp_delta.sum()))
        return effective_boost * BOOST_UTILIZED_SCALE * damage_dealt

    def _compute_finishing_blow_bonus(self, delta: TurnDelta, live) -> float:
        """Extra bonus when a damaging move secures the KO.

        Suppressed when our own mon faints the same turn (Explosion / Self-Destruct,
        or any mutual KO). A kill that costs us our own mon is not a clean finishing
        blow: without this guard a 1-for-1 *healthy* Explosion trade nets POSITIVE,
        because the symmetric hp_ours/hp_opp and faint_ours/faint_opp terms cancel
        and the +0.5 tips it over — teaching the policy to use Explosion as a free
        KO button. The trade-cost side is handled by the faint_ours material penalty.
        """
        if not delta.opp_fainted:
            return 0.0
        if delta.we_fainted:
            return 0.0
        if delta.our_move_id is None or delta.our_switch_to is not None:
            return 0.0
        if delta.our_failed_to_move:
            return 0.0
        mon = live.ours.active
        md = (_movedex.get(delta.our_move_id)
              if mon and delta.our_move_id in mon.move_ids else None)
        if md is None or md.base_power == 0:
            return 0.0
        return FINISHING_BLOW_BONUS

    # =========================================================
    # PBRS / clock / bias-refund FOLDS — the per-class treatment process_turn_reward applies.
    # Kept as named one-purpose methods so the orchestrator reads as a phase sequence and the
    # telescoping math (the §2.3 dominant-win footgun) lives in ONE place.
    # =========================================================
    @staticmethod
    def _pbrs_step(prev: Optional[float], phi_next: float, is_terminal: bool) -> tuple[float, float]:
        """One PBRS shaping step: F = γ·Φ(s′) − Φ(s), with the absorbing ``Φ(terminal)=0`` convention
        and the standard ``prev is None`` first-window skip. Returns ``(shaped, new_prev)``. The
        terminal-zeroing lives HERE once, so a new potential can't forget it (design §2.3)."""
        phi = 0.0 if is_terminal else phi_next
        shaped = (PBRS_GAMMA * phi - prev) if prev is not None else 0.0
        return shaped, phi

    def _fold_material_pbrs(self, bd: "RewardBreakdown", live, is_terminal: bool) -> None:
        """Material PBRS Φ_mat (design §2) → ``bd.pbrs_material``. Replaces the old unconditional
        hp/faint base spine; telescopes to −Φ_mat(s_0) → every win +30, every loss −30."""
        bd.pbrs_material, self._prev_phi_mat = self._pbrs_step(
            self._prev_phi_mat, self._compute_phi_mat(live), is_terminal)

    def _fold_belief_pbrs(self, bd: "RewardBreakdown", live, is_terminal: bool) -> None:
        """Incoming-KO belief PBRS (design_reward_switching.md) → ``bd.pbrs_belief``; also snapshots
        the active KO-risk AND whether a safe bench pivot exists for next turn's escape/stay re-gate
        (both keyed at decision time — read by `_compute_stay_risk_tax` / `_apply_switch_outcome`)."""
        phi_belief_next, active_risk_next, min_bench_pko_next = self._belief_potential_and_risk(live)
        bd.pbrs_belief, self._prev_phi_belief = self._pbrs_step(
            self._prev_phi_belief, phi_belief_next, is_terminal)
        self._prev_active_ko_risk = 0.0 if is_terminal else active_risk_next
        self._prev_safe_pivot = (not is_terminal) and (min_bench_pko_next <= SAFE_PIVOT_PKO_MAX)

    def _apply_progress_clock(self, bd: "RewardBreakdown") -> None:
        """Read the shared ProgressClock's stashed penalty into ``bd.no_progress_tax`` and suppress
        the escalating anti-spam family it SUBSUMES (design §3.1) — gated on ``bias_redesign``. In the
        default single-variable run the clock only tracks the obs scalar (no penalty, taxes intact)."""
        if not (self.config.bias_redesign and self.progress_clock is not None):
            return
        bd.no_progress_tax = float(getattr(self.progress_clock, "last_penalty", 0.0))
        bd.repetition_tax = bd.struggle_tax = bd.switch_bouncing_tax = bd.dead_matchup_tax = 0.0

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
        """
        bd = RewardBreakdown()

        # Current-board facts are read through the vetted LiveView read-model, built
        # ONCE per turn here and threaded to every helper. The reward manager never
        # reaches into the raw poke-env battle for "what is true now".
        live = battle.live_view()

        # --- TERMINAL: the ±30 win/loss (out of scope; never shaped) ---
        won, lost, finished = self._terminal(live)
        if won:
            bd.win_loss = VICTORY_VALUE
        elif lost or finished:
            bd.win_loss = -VICTORY_VALUE
        is_terminal = won or lost or finished

        # --- Material PBRS Φ_mat (design §2): replaces the unconditional hp/faint base spine ---
        self._fold_material_pbrs(bd, live, is_terminal)

        # --- Explosion / self-destruct ---
        # The survive-the-Explosion credit is now carried by Φ_mat (opp lost a mon, we lost nothing),
        # so the old +2.0 literal is DELETED (design §2.5). The explosion_block bonus (no-sold the
        # Explosion via immunity / 0 damage) is KEPT — it lives inside the same `not we_fainted` gate.
        opp_event = delta.opp_damaging_event
        if opp_event is not None and opp_event.move_id in ("explosion", "selfdestruct"):
            if not delta.we_fainted:
                if delta.our_hp_delta.sum() == 0.0 or opp_event.effectiveness == 0.0:
                    bd.explosion_block = EXPLOSION_BLOCK_BONUS
            # When we_fainted: Φ_mat already prices the mon loss; no extra penalty.

        # --- Finishing blow ---
        bd.finishing_blow = self._compute_finishing_blow_bonus(delta, live)

        # --- Attack signals ---
        bd.roar = self._compute_roar_bonus(delta, live)
        bd.futile_attack = self._compute_futile_attack_penalty(delta, live)
        bd.futile_setup = self._compute_futile_setup_penalty(delta)
        bd.setup_low_hp = self._compute_setup_low_hp_penalty(delta)
        bd.boost_utilized = self._compute_boost_utilized(delta, live)

        # --- Field control ---
        bd.spikes, bd.futile_spikes = self._compute_spikes_bonus(delta, live)

        # --- Positional: penalty for staying in against a known threat ---
        bd.matchup_penalty = self._compute_matchup_penalty(delta)
        # Belief-risk-scaled stay-into-KO tax (the under-switch lever; OFF unless --switch-bias-weight>0).
        bd.stay_risk_tax = self._compute_stay_risk_tax(delta)
        # Escalating penalty for refusing to pivot out of a 0×-only matchup.
        bd.dead_matchup_tax = self._compute_dead_matchup_tax(delta, live)

        # --- Switch rewards (see SWITCH_REWARDS.md for full breakdown) ---
        # Pivot, SE, and sleep-out are skipped when phazed — roar removes our
        # choice, so those signals don't apply.
        if delta.our_switch_to is not None:
            if not self._last_switch_was_roared:
                bd.pivot_protect, bd.pivot_status, bd.pivot_damage = self._compute_pivot_bonus(delta, live)
                bd.se_switch = self._compute_se_switch_bonus(delta, live)
                bd.sleep_out = self._compute_sleep_out_bonus(delta, live)
                # Update per-mon opponent tracker for future se_switch gating
                our_mon_in = live.ours.active
                opp_mon_in = live.opp.active
                if our_mon_in and opp_mon_in and not opp_mon_in.fainted:
                    self._last_opp_seen_by[our_mon_in.species] = opp_mon_in.species
            bd.sleep_in = self._compute_sleep_in_penalty(delta, live)

        # --- Status signals ---
        bd.status, _d_opp_statused = self._compute_status_reward(delta, live)
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
        if battle.turn > STALL_TAX_START_TURN:
            ramp = (battle.turn - STALL_TAX_START_TURN) / STALL_TAX_RAMP_TURNS
            bd.stall_tax = -min(STALL_TAX_PER_TURN * ramp, STALL_TAX_MAX)

        # --- No-progress clock: read the stashed penalty + suppress the anti-spam family it subsumes
        # (design §4 / §3.1). Gated on bias_redesign; in the default run the clock only feeds the obs.
        self._apply_progress_clock(bd)

        # Update end-of-turn snapshots for next turn's checks
        self._prev_opp_boosts = self._opp_active_boosts(live)
        self._update_opp_se_threat(live)

        # --- Belief PBRS (design_reward_switching.md) → pbrs_belief + the re-gate risk snapshot ---
        self._fold_belief_pbrs(bd, live, is_terminal)

        # Track whether our last move did anything PRODUCTIVE this turn, for the
        # escalating repetition tax. A move counts as effective if it dealt damage,
        # gained us a stat boost, landed a status, or added a hazard layer. Capped
        # setup (no boost change), capped hazards, redundant status, and immune/no-op
        # attacks all flip this to False, routing the next repeat through the steeper
        # ZERO_EFFECT step — that's how capped setup/hazards escalate.
        self._last_attack_had_effect = (
            float(delta.opp_hp_delta.sum()) < 0
            or int(delta.our_boost_delta.sum()) > 0
            or _d_opp_statused > 0
            or bd.spikes > 0
        )

        # --- BIAS-additivity accumulate-and-refund (design §1.2) — LAST, after every BIAS field ---
        self._fold_bias_refund(bd)

        self._last_breakdown = bd
        reward = bd.total
        self.total_reward += reward
        self._log_turn(bd, battle, meta)
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
            print(f"    🔍 [DEEP TRACE] Type: FORCED SWITCH (post-faint, no subsidy)")
        elif meta.get("type") == "FORCED_ROAR":
            print(f"    🔍 [DEEP TRACE] Type: FORCED SWITCH (roar/whirlwind, no bonuses)")

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
