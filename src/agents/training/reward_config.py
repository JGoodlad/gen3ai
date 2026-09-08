"""The reward's DECLARATIONS — what a run is configured with, and what a turn's reward is made of.

Split out of `reward_manager.py` (2026-09-07) alongside `reward_bias_terms.py` and
`reward_potentials.py`, which took that file from 1,990 lines — ten short of the size gate's
2,000-line hard bound — down to a fold sequence and its orchestrator. This is the DECLARATIVE
half of the cut the backlog named ("the per-term fold functions vs `RewardConfig`): three
declarations and one tuple, none of which touch a manager, a battle or a turn.

* `RewardClass`     — the three classes the fold applies ONE treatment per (TERMINAL/PBRS/BIAS).
* `RewardConfig`    — the per-run, resume-IMMUTABLE reward configuration, recorded in
                      `model_config.json` and reconstructed from it on every eval / resume path.
* `RewardBreakdown` — the per-turn record, and `_REGISTRY`: the field→class map that IS the
                      reward's source of truth. The fold reads it; the census reads it; the
                      `reward/` export reads it.
* `SWITCH_BIAS_DROP_FAMILY` — the terms `--drop-switch-bias` removes, declared ONCE so
                      `_apply_bias_drops` and the census cannot disagree about what the flag drops.

**Every public name here is re-exported by `reward_manager`**, so `from
agents.training.reward_manager import RewardConfig` (the spelling ~20 modules and tests use)
still resolves. `reward_composition` imports them from HERE at module level: this module's only
dependency is `reward_weights`, so there is no cycle to defer around any more.
"""
from dataclasses import dataclass, fields
from enum import Enum
from typing import ClassVar

from agents.training.reward_weights import PBRS_GAMMA


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
    # Terminal reward for a DRAW / 250-turn timeout (no winner). More negative than a decisive loss
    # makes stalling to the turn cap strictly worse than losing cleanly — it removes the
    # discount-driven micro-incentive to delay an inevitable loss and discourages no-progress
    # stall-wars. A decisive loss stays -VICTORY_VALUE.
    # DEFAULT -35.0 (owner decision 2026-08-18): the value every validated ai_v8 run trained with.
    # -30.0 (a tie scored identically to a decisive loss) was the historical default and was tuned
    # under the ADDITIVE-bias regime, which the composition below no longer runs — see the
    # `all_shaping_pbrs` note. Pass `--draw-penalty -30` for the old value.
    # Per-run constant, resume-immutable (recorded in model_config.json, value-checked).
    draw_penalty: float = -35.0
    # Decision-time-HP-scaled self-KO penalty weight. 0.0 = OFF (default → byte-unchanged). >0 charges
    # −w·(our active HP fraction at decision time) when our mon self-KOs (Explosion / Self-Destruct +
    # we_fainted). The symmetric material PBRS prices a healthy 1-for-1 trade at ~0, so the critic
    # learns to value a full-HP self-KO POSITIVELY (measured dV ≈ +2.9, PPO advantage ≈ +1.5 on
    # ≥80%-HP self-KOs) → the policy throws away healthy mons (turn-1 full-HP Explosion). Scaling by HP
    # spares the legitimate low-HP "explode a dying mon to secure a KO" play. Per-run constant,
    # resume-immutable (recorded in model_config.json, value-checked via check_reward_config).
    self_ko_hp_penalty: float = 0.0
    # De-bias cleanup (audit TIER-1 distorter removal). Both default OFF = byte-identical no-op (the
    # registry-coverage / no-op tests pin it); each ZEROES its BIAS terms BEFORE the bias-refund fold,
    # so they leave the accumulator too. Per-run constants, resume-immutable (check_reward_config).
    #   `drop_redundant_bias` — terms REDUNDANT with an existing PBRS/terminal term:
    #       stall_tax       ↔ no_progress clock + --draw-penalty (it also taxed winning long games on raw turn count)
    #       matchup_penalty ↔ pbrs_belief (the same incoming-KO threat signal, but additive not telescoping)
    #   `drop_switch_bias` — the HAND-CODED switch-strategy subsidy (switch_base / switch_bouncing_tax /
    #       escape_threat_switch / se_switch / pivot_* / sleep_in / sleep_out). Switching value is LEARNABLE
    #       from Φ_mat + pbrs_belief + win/loss, so hand-rewarding it distorts the objective.
    drop_redundant_bias: bool = False
    drop_switch_bias: bool = False
    # End-state PBRS rollout, split into TWO independent switches (design §2.6/§2.7 + boost/opp-boost
    # /progress potentials). Resume-immutable, value-checked (check_reward_config). NOT weight-shape
    # (no ARCH_SIGNATURE bump).
    #
    # `all_shaping_pbrs` = "everything but stall". ON: (1) fold Φ_hazard/Φ_boost/Φ_opp_boosts + Φ_status
    #   (PBRS class), (2) ZERO every BIAS term EXCEPT the anti-stall tilt `no_progress_tax` — so all
    #   NON-stall shaping becomes policy-invariant. (The bad turn-ramp `stall_tax`, which taxed winning
    #   long games, IS zeroed; the progress-aware `no_progress_tax`, which protects winning stalls, is the
    #   one BIAS term kept — and it is itself activated here so the stall tilt works without --bias-redesign.)
    #
    # DEFAULT ON (owner decision 2026-08-18). Every validated ai_v8 run trained with it; every ai_v9 run
    # through gen-14 trained WITHOUT it, with no recorded rationale — a SILENT composition drift at the
    # generation boundary (designs/research_state/ledger.md → "the reward composition drifted SILENTLY at
    # the v8→v9 boundary"). It was invisible because it is training-only, bumps no signature, and is
    # absent from check_compatible. The default now equals the validated composition; `--no-all-shaping-pbrs`
    # is the fallback, and `reward_class_composition` below makes the choice visible at every launch.
    all_shaping_pbrs: bool = True
    # `stall_pbrs` = "stall". ON: fold Φ_progress (PBRS) and ZERO `no_progress_tax` + `stall_tax`, so the
    #   anti-stall signal is policy-invariant too. Run --all-shaping-pbrs WITH --stall-pbrs for a FULLY-PBRS
    #   reward (TERMINAL + PBRS only, zero bias); run --all-shaping-pbrs WITHOUT it to keep the no_progress
    #   stall tilt as the single acknowledged BIAS (insurance against stall-regression — watch stall-rate).
    stall_pbrs: bool = False
    # --- The no-progress clock's two intent-restoring fixes (both default OFF = byte-identical) ---
    # Motivation is measured: probe M (`bias_tax_head_alignment_2026-08-29.md`) censused what the tax
    # actually charges, probe N (`no_progress_tax_review_2026-08-29.md`) traced the intent against the
    # implementation. The mechanism lives on `ProgressClock`; these fields exist HERE because the
    # clock's behaviour is part of the reward a run trained under — resume-immutable, value-checked,
    # recorded in model_config.json, and read by BOTH the training env and the eval-side
    # RewardTracker so eval measures the same clock training used. Turning either ON changes the
    # `turns_since_progress` OBS scalar as well as the charge (they key on ONE value by design), so
    # both are retrain-class — no dim moves and no ARCH_SIGNATURE bump, but not a mid-run toggle.
    #
    # `progress_decision_tense` (F1) — point BOTH window gates at the decision being charged rather
    #   than the one after it. Fixes the sit-out landing on 13.2% of FULL-agency decisions AND the
    #   zero-agency post-faint replacement being charged 63.9% of the time (36.3% of all charges).
    # `progress_switch_freeze` (F2b) — a voluntary switch that fails the (offense-only) progress
    #   predicate FREEZES instead of charging. 42.7% of all charges; that charge was designed inside
    #   a composition that also paid `switch_base +0.5` for the same pivot, and `928a00b` zeroed
    #   every such credit while keeping the tax.
    progress_decision_tense: bool = False
    progress_switch_freeze: bool = False

    # --- gen3_clean_world_config_v1 (ai_v12 build wave A). The three switches the CLEAN-WORLD arm
    # needs and the flag surface could not express. EVERY default below is today's behaviour, so a
    # flagless run is byte-identical; the census (`reward_class_composition`) proves it. ---
    #
    # `hand_shaping` — the MASTER off-switch for every HAND-DESIGNED dense term. False ⇒ all EIGHT
    #   `_fold_*_pbrs` early-return (material and belief included) AND the whole BIAS class is
    #   zeroed, leaving TERMINAL alone. It exists because the two jobs `all_shaping_pbrs` bundles
    #   are ANTI-CORRELATED: `--no-all-shaping-pbrs` silences five potentials but REVIVES 25 BIAS
    #   terms, because `asp` is also `_bias_term_active`'s master gate — so "no hand PBRS *and* no
    #   BIAS" was unreachable by any combination of the pre-existing flags
    #   (`designs/research_state/measurements/no_progress_tax_review_2026-08-29.md` §5.1).
    #
    #   ⚠️ STATE THIS HONESTLY IN ANY WRITE-UP: every PBRS term is policy-INVARIANT by construction
    #   (Φ(terminal)=0, telescoping), so removing them CANNOT change the optimal policy. It changes
    #   learning dynamics and conceptual complexity. The clean-world claim's real content is "the
    #   hand terms cost more in interference and tuning than they buy in credit assignment" — never
    #   "the hand terms bias the objective". The only class that biases the objective is BIAS.
    hand_shaping: bool = True
    # The two potentials that had NO gate at all (`_pbrs_term_active` returned True unconditionally
    # for them and neither fold had an early return). Deliberately INDEPENDENT of `all_shaping_pbrs`
    # rather than folded into it — see the anti-correlation note above.
    pbrs_material: bool = True
    pbrs_belief: bool = True
    # The TERMINAL magnitude, promoted from the `reward_weights.VICTORY_VALUE` module constant so a
    # ±1 terminal is reachable BY FLAG. A win scores +victory_value; a decisive loss AND a rare
    # pre-cap tie score −victory_value; a 250-turn TIMEOUT scores `draw_penalty` instead.
    # Default 30.0 == the constant (the two are pinned equal by `reward_defaults_test.py`).
    # ⚠️ `MAT_HP_WEIGHT` / `MAT_ALIVE_WEIGHT` are calibrated AGAINST the 30 scale, so a ±1 terminal
    # with Φ_mat still on would make material dwarf the outcome. Moot under `--no-hand-shaping`
    # (Φ_mat is off there), which is the composition the ±1 terminal exists for.
    victory_value: float = 30.0
    # gen3_winprob_critic_mode_v1 — the TERMINAL as a WIN INDICATOR, for the win-prob critic.
    # False (the default) is every generation to date: a win pays +victory_value, a decisive loss
    # and a rare pre-cap tie pay -victory_value, a 250-turn TIMEOUT pays `draw_penalty`.
    # True pays +victory_value on a WIN and **0.0 on everything else** (loss, tie, timeout alike).
    #
    # It is not a taste. Under `--critic winprob` the critic is sigmoid(logit) ∈ [0,1] and the
    # value target is P(win); GAE mixes the REWARD with that critic, so the reward has to be the
    # win INDICATOR or the two are in different scales and every terminal TD error carries a
    # systematic, state-dependent offset (a loss would read `-V` against a truth of `0-V`). With
    # this on and `victory_value == 1.0` the undiscounted return from any state is exactly
    # 1{win}, so V(s) == E[return] == P(win|s) with no approximation term — which is the whole
    # identity the win-prob critic rests on.
    #
    # ⚠️ `draw_penalty` and the `draw_penalty <= -victory_value` ORDERING become INAPPLICABLE, not
    # merely inert: a critic bounded in [0,1] cannot represent "a timeout is worse than a loss" at
    # all. That anti-stall defence has to come from the obs deadline clock and, if the stall rate
    # rises, from `no_progress_tax_armed` below. `--draw-penalty` is REFUSED under this mode
    # rather than silently ignored (main.train.combination_checks).
    terminal_indicator: bool = False
    # gen3_winprob_critic_mode_v1 (design gap B4) — re-arm the anti-stall tilt under
    # `--no-hand-shaping`. The master switch zeroes the WHOLE BIAS class, `no_progress_tax`
    # included; this is the one exception, and it exists because the clean-world composition and
    # the win-prob terminal each drop an anti-stall defence. False (the default) is today's
    # behaviour exactly: with `hand_shaping` on it changes nothing either way, because the term is
    # already reachable.
    no_progress_tax_armed: bool = False

    # --- single source of truth: build once, flow everywhere (training + eval + version record) ---
    # Adding a reward flag = add the field above + a matching `--field-name` CLI arg. `from_args`
    # picks it up (no hand-threading), `from_dict` reconstructs it for eval/resume, and the eval
    # reward then automatically matches what the policy was trained with. This DRY-ness exists because
    # a hand-threaded field was once silently MISSED on the eval path (eval measured the wrong reward).
    @classmethod
    def from_args(cls, args) -> "RewardConfig":
        """THE construction site from parsed CLI args. Every field whose name matches a CLI dest is
        pulled from ``args``; ``gamma`` is the fixed PPO discount (0.9999, asserted == model.gamma)."""
        vals = {f.name: getattr(args, f.name)
                for f in fields(cls) if f.name != "gamma" and hasattr(args, f.name)}
        # gen3_winprob_critic_mode_v1: `--gamma` is now a flag, and PBRS is only policy-invariant
        # when PBRS_GAMMA == reward_config.gamma == model.gamma (asserted at build time whenever
        # any hand potential is folded). An UNSET --gamma resolves to the historical 0.9999 in
        # `main.train.config`, so a flagless run reads exactly as it always did.
        vals["gamma"] = float(getattr(args, "gamma", None) or PBRS_GAMMA)
        return cls(**vals)

    @classmethod
    def from_dict(cls, d: "dict | None") -> "RewardConfig":
        """Reconstruct from a ``model_config.json`` dict — the helper EVERY snapshot-loading consumer
        (eval workers, resume) uses so the reward the policy was TRAINED with is the reward used to
        MEASURE it. Unknown keys (arch fields / use_popart / …) are ignored; any reward field absent
        from an older config falls back to its dataclass default."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


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
    # (An `explosion` field sat here until 2026-08-18 as a "vestigial" placeholder — the +2.0
    # literal was deleted in design §2.5 and the survive-Explosion credit rides Φ_mat. Nothing
    # ever assigned it again, so it was a permanent 0.0 that still counted as an ACTIVE BIAS
    # term in `reward_class_composition` — a census meant to name a run's hand-coded incentives
    # was naming one that could not fire. Removed; the additive-regime BIAS count is 25, not 26.)
    explosion_block: float = 0.0   # Ghost immune or Protect blocked opponent Explosion
    finishing_blow: float = 0.0    # damaging move secured the KO
    self_ko_penalty: float = 0.0   # HP-scaled penalty for self-KOing a (healthy) mon — Explosion/
                                   # Self-Destruct throws away its future value (--self-ko-hp-penalty)

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

    # Non-damaging-tempo-status PBRS Φ_status (design §2.7 / §7.4): the standing value of an opponent
    # held in par/slp/frz that Φ_mat can't price. = γ·Φ_status(s′) − Φ_status(s), Φ_status(terminal)=0,
    # telescopes to 0 (policy-invariant). Non-zero ONLY under bias_redesign (the default count-diff
    # status BIAS already carries the standing value; folding it there would double-count).
    pbrs_status: float = 0.0

    # End-state PBRS potentials (gated on all_shaping_pbrs; 0.0 in the default run). Each holds
    # γ·Φ(s′)−Φ(s), Φ(terminal)=0 → telescopes to net-zero (policy-invariant).
    pbrs_progress: float = 0.0    # Φ_progress = −no_progress_penalty·progress_clock.value() (anti-stall)
    pbrs_hazard: float = 0.0      # Φ_hazard = HAZARD_WEIGHT·(opp_spikes − our_spikes) (design §2.6)
    pbrs_boost: float = 0.0       # Φ_boost = BOOST_WEIGHT·Σmax(0,our_active_boost)·hp_frac (stored offense)
    pbrs_opp_boosts: float = 0.0  # Φ_opp_boosts = −OPP_BOOST_WEIGHT·Σmax(0,opp_active_boost) (phaze value)
    pbrs_roar: float = 0.0        # Φ_roar = −ROAR_BOOST_WEIGHT·Σmax(0,opp_active_boost) — DEDICATED
                                  # phaze-out-boosts PBRS, folded INTO --all-shaping-pbrs; same potential
                                  # shape as pbrs_opp_boosts but its own weight (stacks there)

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
        "pbrs_status": RewardClass.PBRS,
        "pbrs_progress": RewardClass.PBRS, "pbrs_hazard": RewardClass.PBRS,
        "pbrs_boost": RewardClass.PBRS, "pbrs_opp_boosts": RewardClass.PBRS,
        "pbrs_roar": RewardClass.PBRS,
        # everything else is BIAS:
        "explosion_block": RewardClass.BIAS,
        "finishing_blow": RewardClass.BIAS, "self_ko_penalty": RewardClass.BIAS,
        "roar": RewardClass.BIAS,
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
        ("base",   ("win_loss", "pbrs_material", "explosion_block", "finishing_blow",
                    "self_ko_penalty")),
        ("attack", ("roar", "futile_attack", "futile_setup", "setup_low_hp",
                    "boost_utilized", "status_wasted", "repetition_tax", "struggle_tax")),
        ("switch", ("switch_base", "switch_bouncing_tax", "escape_threat_switch",
                    "escape_risk_bonus", "pivot_protect", "pivot_status",
                    "pivot_damage", "se_switch", "sleep_out", "sleep_in")),
        ("field",  ("spikes", "futile_spikes", "matchup_penalty", "dead_matchup_tax",
                    "stay_risk_tax", "status", "stall_tax", "no_progress_tax")),
        ("shaping", ("pbrs_belief", "pbrs_status", "pbrs_progress", "pbrs_hazard",
                     "pbrs_boost", "pbrs_opp_boosts", "pbrs_roar", "bias_refund")),
    )

    # Derived-once memos: per-PROCESS constants that were rebuilt EVERY turn (measured:
    # `registry_fields` 1.6%, `total`'s `dataclasses.fields()` 2.0% of `process_turn_reward`).
    # Lazy, not class-body, so they stay DERIVED from the registry rather than a copy of it.
    _REGISTRY_FIELDS: ClassVar[dict] = {}
    _TOTAL_FIELDS: ClassVar[tuple] = ()

    @classmethod
    def registry_fields(cls, reward_class: "RewardClass") -> tuple:
        """The breakdown fields belonging to ``reward_class`` (the registry is the source of truth)."""
        cached = cls._REGISTRY_FIELDS.get(reward_class)
        if cached is None:
            cached = tuple(name for name, c in cls._REGISTRY.items() if c is reward_class)
            cls._REGISTRY_FIELDS[reward_class] = cached
        return cached

    @classmethod
    def field_names(cls) -> tuple:
        """Every declared breakdown field, in declaration order (cached; see `_TOTAL_FIELDS`)."""
        names = cls._TOTAL_FIELDS
        if not names:
            names = cls._TOTAL_FIELDS = tuple(f.name for f in fields(cls))
        return names

    @property
    def total(self) -> float:
        """Sum every registry term + the bias_refund mechanism (= every dataclass float field).
        Sums the cached NAME tuple instead of re-deriving `dataclasses.fields()` and
        re-`isinstance`-ing 39 values per turn — equivalent only because every field is declared
        `float`, which `reward_skip_parity_test` pins."""
        return sum(getattr(self, n) for n in RewardBreakdown.field_names())

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


# --- The reward COMPOSITION announcer (gen3_reward_composition_v1) -------------------------------
# The v8→v9 reward drift (designs/research_state/ledger.md, 2026-08-18) was invisible because
# nothing ever STATED what the reward was made of: `--all-shaping-pbrs` simply stopped being passed
# at a generation boundary, and every ai_v9 run through gen-14 trained a fully-additive 26-term
# BIAS objective where the validated ai_v8 composition was near-policy-invariant. Nothing
# failed; no signature moved; `check_compatible` does not look at reward hparams. This is the
# counter-measure and the seed of the launch-diff gate: ONE pure function that turns a config into a
# per-class census, printed at startup and recorded into metadata.json. The census itself lives in
# `reward_composition.py`; what it reads — the registry and the drop family — is declared here.
#
# ACTIVE means "this CONFIG does not structurally force the term to zero". It is a statement about
# the configuration, not about a battle — an active term is still 0.0 on most turns.
#
# Duck-typed on purpose: it reads the reward field NAMES, so a `RewardConfig`, a `ModelVersion`
# (which records the same fields) or a plain argparse namespace all work.

# The hand-coded switch-strategy subsidy `--drop-switch-bias` removes. Declared once here and
# consumed by BOTH `Gen3RewardManager._apply_bias_drops` (which zeroes them) and
# `reward_composition._bias_term_active` (the census), so the two can never disagree about what
# the flag drops.
SWITCH_BIAS_DROP_FAMILY = (
    "switch_base", "switch_bouncing_tax", "escape_threat_switch", "se_switch",
    "pivot_protect", "pivot_status", "pivot_damage", "sleep_out", "sleep_in",
)
