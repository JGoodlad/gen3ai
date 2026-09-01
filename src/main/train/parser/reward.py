"""The `# --- Reward config ---` section (design_markovian_reward_and_features.md).
Resume-immutable, value-checked.

Lifted VERBATIM out of the old single-file `parser.py` (lines 260-328); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from main.train.parser.base import BoolFlag


def add_reward_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
    # --- Reward config (design_markovian_reward_and_features.md). Resume-immutable, value-checked. ---
    parser.add_argument("--bias-additivity", "--bias_additivity", dest="bias_additivity", type=float,
                        default=1.0, help="BIAS-class additive↔telescoping knob λ∈[0,1] (default 1.0 = "
                        "fully additive, byte-identical to today's biases). 0.0 = fully telescoping "
                        "(pure PBRS hint). Per-run constant (NOT annealed). Resume-immutable.")
    parser.add_argument("--mat-alive-weight", "--mat_alive_weight", dest="mat_alive_weight", type=float,
                        default=1.25, help="Material PBRS Φ_mat per-mon-alive weight (default 1.25). "
                        "Resume-immutable.")
    parser.add_argument("--no-progress-penalty", "--no_progress_penalty", dest="no_progress_penalty",
                        type=float, default=0.15, help="Flat per-no-progress-window penalty magnitude "
                        "(default 0.15; only charged when --bias-redesign).")
    parser.add_argument("--bias-redesign", "--bias_redesign", dest="bias_redesign", action=BoolFlag,
                        default=False, help="Enable the staged BIAS redesign: the no-progress clock "
                        "replaces the anti-spam taxes + the obs-keyed reframes apply. Default OFF = the "
                        "single-variable run (material clutch-fix only). Pass --no-bias-redesign (or "
                        "--bias-redesign false) to set it off explicitly. Resume-immutable.")
    parser.add_argument("--switch-bias-weight", "--switch_bias_weight", dest="switch_bias_weight",
                        type=float, default=0.0, help="Belief-risk-scaled stay-into-KO BIAS lever for "
                        "the under-switch pathology (design_reward_switching.md §7). 0.0 = OFF "
                        "(default; behavior unchanged). >0 taxes staying in a high-P(KO) spot when a "
                        "safe pivot exists (−w·risk) + rewards escaping it. BIAS-class, so it also "
                        "rides --bias-additivity (λ=1 additive vs λ=0 telescoping A/B). Resume-immutable.")
    parser.add_argument("--draw-penalty", "--draw_penalty", dest="draw_penalty", type=float,
                        default=-35.0, help="Terminal reward for a DRAW / 250-turn timeout (no "
                        "winner). DEFAULT -35.0 = the validated ai_v8 value: stalling to the turn "
                        "cap is strictly worse than losing cleanly, which cancels the discount-driven "
                        "micro-incentive to delay an inevitable loss. A decisive loss stays -30. Pass "
                        "-30 for the historical default (a tie scored as a decisive loss), which was "
                        "tuned under the additive-BIAS regime --all-shaping-pbrs replaces. "
                        "Resume-immutable (recorded + value-checked in model_config.json).")
    parser.add_argument("--self-ko-hp-penalty", "--self_ko_hp_penalty", dest="self_ko_hp_penalty",
                        type=float, default=0.0,
                        help="Decision-time-HP-scaled penalty (-w*hp) for self-KOing a mon via "
                        "Explosion/Self-Destruct. Default 0.0 = OFF (behavior unchanged). The symmetric "
                        "material PBRS prices a healthy 1-for-1 trade at ~0, so the critic learns to "
                        "value a full-HP self-KO positively and the policy throws away healthy mons "
                        "(measured: ~38%% of explosions are on >=80%%-HP mons). >0 (e.g. 2.5) charges "
                        "the squandered HP, sparing legitimate low-HP sac-for-KO. Resume-immutable "
                        "(recorded + value-checked in model_config.json).")
    parser.add_argument("--drop-redundant-bias", "--drop_redundant_bias", dest="drop_redundant_bias",
                        action=BoolFlag, default=False, help="De-bias cleanup: zero BIAS terms REDUNDANT "
                        "with an existing PBRS/terminal term — stall_tax (covered by the no-progress clock "
                        "+ --draw-penalty; it also taxed winning long games on raw turn count) and "
                        "matchup_penalty (the same incoming-KO threat signal as pbrs_belief, but additive "
                        "not telescoping). Default OFF = byte-identical. Resume-immutable, value-checked.")
    parser.add_argument("--drop-switch-bias", "--drop_switch_bias", dest="drop_switch_bias",
                        action=BoolFlag, default=False, help="De-bias cleanup: zero the HAND-CODED "
                        "switch-strategy subsidy (switch_base, switch_bouncing_tax, escape_threat_switch, "
                        "se_switch, pivot_*, sleep_in/out) — switching value is LEARNABLE from Φ_mat + "
                        "pbrs_belief + win/loss, so hand-rewarding it distorts the objective. Default OFF "
                        "= byte-identical. Resume-immutable, value-checked.")
    parser.add_argument("--all-shaping-pbrs", "--all_shaping_pbrs", dest="all_shaping_pbrs",
                        action=BoolFlag, default=True, help="END-STATE PBRS, 'everything but stall': "
                        "fold Φ_hazard/Φ_boost/Φ_opp_boosts + Φ_status (telescoping, policy-invariant) "
                        "and ZERO every BIAS term EXCEPT the anti-stall tilt no_progress_tax — so all "
                        "non-stall shaping is policy-invariant (the bad turn-ramp stall_tax is zeroed). "
                        "DEFAULT ON = the validated ai_v8 composition (1 TERMINAL + 7 PBRS + 1 BIAS); "
                        "--no-all-shaping-pbrs is the fallback and restores the fully-additive "
                        "25-term BIAS objective every ai_v9 run drifted into. Pair with --stall-pbrs for "
                        "a FULLY-PBRS reward, or use alone to keep the no_progress stall tilt as the one "
                        "acknowledged BIAS. Resume-immutable, value-checked.")
    parser.add_argument("--stall-pbrs", "--stall_pbrs", dest="stall_pbrs",
                        action=BoolFlag, default=False, help="END-STATE PBRS, 'stall': fold Φ_progress "
                        "(telescoping anti-stall over the turns_since_progress clock) and ZERO "
                        "no_progress_tax + stall_tax, so the anti-stall signal is policy-invariant too. "
                        "Default OFF. Run --all-shaping-pbrs WITH --stall-pbrs ⇒ the whole BIAS class is "
                        "zero (TERMINAL + PBRS only); WITHOUT it ⇒ keep the no_progress stall tilt as "
                        "insurance against stall-regression (watch the stall-rate canary). Resume-"
                        "immutable, value-checked.")
