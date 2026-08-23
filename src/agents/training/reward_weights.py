"""The reward's tunable MAGNITUDES — every weight, bonus, penalty, threshold and clamp.

Split out of ``reward_manager.py`` on 2026-08-23 because the size ratchet
(``src/file_size_gate_test.py``) fired on that file and the documented remedy is decomposition,
never an allowlist entry. This is the natural seam: the module below is a flat block of constants
with no behaviour and no dependency on the manager, and "what number does this term carry" is a
question people answer far more often than they read the fold that consumes it.

**It is a MOVE, not a change.** Every value is verbatim, and ``reward_manager`` re-exports all of
them, so `from agents.training.reward_manager import SE_SWITCH_BONUS` still resolves — the tests
that read these by that path are untouched. The one import it needs is the stall cap, kept here
rather than duplicated so the reward's TIMEOUT test cannot drift from where the env forfeits.

⚠️ **Changing a value here is a RETRAIN-class change**, not a tuning knob you flip mid-run: these
define the objective. The per-run, config-driven magnitudes live on ``RewardConfig`` instead.
"""
from agents.training.stall import StallConfig as _StallConfig

HP_VALUE = 2.0
VICTORY_VALUE = 30.0
FINISHING_BLOW_BONUS = 0.5   # extra bonus for KO'ing with a damaging move

# The turn at which gen3_env forfeits a stalled battle (ForfeitBattleOrder). A terminal at/after this
# turn is a no-progress TIMEOUT (the trainee hit the cap), scored with RewardConfig.draw_penalty —
# kept in sync with the env's stall cap so the reward's timeout test matches where the env forfeits.
_TIMEOUT_TURN_CAP = _StallConfig().threshold

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

# --- Φ_status: a non-damaging-tempo-status potential (design §2.7 / §7.4 hedge). Toxic/burn/poison's
#     value is the HP chip → already in Φ_mat; paralysis / sleep / freeze are NON-damaging tempo
#     ("the opponent loses turns") whose value Φ_mat can't see. Under bias_redesign the status BIAS is
#     the per-window TRANSITION event (fires on the status flip only), so the STANDING value of a held
#     tempo-status vanished — Φ_status restores it as a telescoping (policy-invariant) potential:
#     Φ_status(s) = STATUS_TEMPO_WEIGHT · (opp_tempo_statused − our_tempo_statused), over non-fainted
#     mons. Nobody is statused at s_0 → Φ_status(s_0)=0, and Φ_status(terminal)=0 → net episode
#     contribution telescopes to 0 (no objective bias; the small per-application event-BIAS nudge is the
#     §7.4 standing-hedge). Gated on bias_redesign (the default count-diff status BIAS already pays the
#     standing value → folding it there would double-count). Uniform over the three types — do NOT
#     weight by type (that bakes strategy; design §2.7). ---
STATUS_TEMPO_WEIGHT = 0.3      # per non-damaging-statused mon (par/slp/frz); the §7.4-guard knob
_TEMPO_STATUSES = frozenset({"par", "slp", "frz"})  # non-damaging tempo statuses Φ_mat can't price

ROAR_BONUS = 0.2               # reward for Roar when spikes on opp side or opp had positive boosts
OPP_BOOST_WEIGHT = 0.15        # Φ_opp_boosts per opp positive boost stage (phaze-disruption potential)
ROAR_BOOST_WEIGHT = 0.25       # Φ_roar per opp positive boost stage — the DEDICATED phaze-out-boosts PBRS,
                               # folded INTO --all-shaping-pbrs (no separate flag). A touch stronger than
                               # OPP_BOOST_WEIGHT since the model under-roars; it STACKS with Φ_opp_boosts
                               # under all_shaping_pbrs (both are policy-invariant PBRS, so stacking only
                               # scales the proportional roar-out-boosts shaping — see _fold_roar_pbrs).
SE_SWITCH_BONUS = 0.2          # reward for switching in a mon with a SE damaging move vs opp active
SLEEP_SWAP_BONUS = 0.25        # reward for rotating a sleeping mon out; penalty for rotating one in
SPIKES_LAYER_BONUS = 0.5       # per layer added to opponent's side (credit assignment bridge)
HAZARD_WEIGHT = SPIKES_LAYER_BONUS  # 0.5 — Φ_hazard per-layer weight == the additive spikes BIAS it converts
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
BOOST_WEIGHT = 0.03                # Φ_boost per our-active positive boost stage, scaled by hp_fraction

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
