"""Pure metric-formatting helpers + the dashboard display order.

Framework-agnostic (no Rich, no Textual) — consumed by the Textual dashboard
(`app.py`). Keep this the single source of truth for how a metric renders so the
numbers stay consistent everywhere.
"""


def _elapsed_str(seconds: float) -> str:
    h, rem = divmod(int(max(0, seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _secs_str(seconds: float) -> str:
    """A bare ``Ns`` second-count, switching to ``XmYs`` once past 600 s (10 min) so long
    staleness/duration spans (``took …``, ``… ago``) stay legible instead of a huge raw count."""
    s = int(max(0, seconds))
    if s > 600:
        m, rem = divmod(s, 60)
        return f"{m}m{rem}s"
    return f"{s}s"


def _fmt_val(v: float) -> str:
    """4 significant figures; comma-separated integers for large whole numbers."""
    if isinstance(v, float) and v == int(v) and abs(v) >= 1000:
        return f"{int(v):,}"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 1000 else str(v)
    return f"{v:.4g}"


def _fmt_metric(key: str, v: float) -> str:
    """Format a metric value; win rates + distilled-fraction render as percentages, the
    all-distilled flag as yes/no, ELO as a whole rating + its CI as ±points."""
    if key == "eval/elo":
        return f"{v:.0f}"
    if key == "eval/elo_ci":
        return f"±{v:.0f}"
    if "win_rate" in key or key == "distill/frac_active_opponents_distilled":
        return f"{v * 100:.1f}%"
    if key == "distill/all_distilled":
        return "yes" if v >= 0.5 else "no"
    return _fmt_val(v)


# Short, legible row labels for keys whose tail is verbose (otherwise the label is the
# part after '/'). Keeps the dashboard's distill block readable.
_METRIC_LABELS = {
    "eval/elo": "ELO",
    "eval/elo_ci": "ELO 95% CI",
    # Opponent-mix curriculum telemetry (the INTENDED per-episode opponent probabilities).
    "train/selfplay_fraction": "pool frac",
    "train/stable_fraction": "stable frac",
    "train/nonbot_fraction": "nonbot frac",
    # Gradient noise scale (B_simple, McCandlish) — only under --grad-accum-steps>=2. noise_ratio =
    # B_simple / effective-batch: ≫1 noise-limited (bigger batch helps), ≪1 diminishing returns.
    "train/noise_scale": "noise scale",
    "train/noise_scale_ratio": "noise/batch",
    "distill/all_distilled": "all distilled",
    "distill/frac_active_opponents_distilled": "distilled",
    "distill/n_ready": "ready",
    "distill/n_running": "running",
    "distill/n_exhausted": "exhausted",
    # Gradient balance (shared-trunk value-vs-policy pull) — tune vf_coef / PopArt to these.
    "grad/value_share": "value share",
    "grad/value_policy_logratio": "log val/pol grad",
    "grad/policy_value_cosine": "policy-value cos",
    "grad/policy_norm_shared": "policy grad-norm",
    "grad/value_norm_shared": "value grad-norm",
    # Belief-aux pull on the shared trunk (--opp-belief-aux-coef / --move-belief-mode /
    # --opp-belief-latent-coef): belief = the COMBINED aux pull, latent = the role-token predictor alone.
    "grad/belief_share": "belief share",
    "grad/belief_norm_shared": "belief grad-norm",
    "grad/belief_policy_cosine": "belief-policy cos",
    "grad/latent_share": "latent share",
    "grad/latent_norm_shared": "latent grad-norm",
    "grad/latent_policy_cosine": "latent-policy cos",
    # PopArt value-target normalizer (--use-popart).
    "popart/mu": "value mu",
    "popart/sigma": "value sigma",
    "popart/value_weight_norm": "value head |W|",
    # Hidden-opponent belief-aux diagnostics (--opp-belief-aux-coef / --move-belief-mode /
    # --opp-belief-latent-coef). species_acc is the headline; move_* = the move-belief head;
    # latent_* = the role-token latent predictor (std→0 while cosine→1 is the collapse NO-GO).
    "belief/species_acc": "species acc",
    "belief/species_acc_above_chance": "species acc>chance",
    "belief/moves_precision": "moves prec",
    "belief/moves_recall": "moves recall",
    "belief/coverage": "coverage",
    "belief/k_mean": "k mean",
    "belief/species_ce": "species CE",
    "belief/moves_bce": "moves BCE",
    "belief/aux_loss": "aux loss",
    "belief/move_bce": "move BCE",
    "belief/move_precision": "move prec",
    "belief/move_recall": "move recall",
    "belief/move_revealed_slots": "move revealed",
    "belief/move_unrevealed_slots": "move unrevealed",
    "belief/move_loss": "move loss",
    "belief/latent_cosine": "latent cos",
    "belief/latent_cosine_above_chance": "latent cos>chance",
    "belief/latent_cosine_baseline": "latent cos base",
    "belief/latent_std": "latent std",
    "belief/latent_vicreg": "latent vicreg",
    "belief/latent_loss": "latent loss",
    # Distributional value head (--value-dist-mode, v29): the interpretability critic's aggregate
    # health. entropy/std fall as it sharpens; PIT → 0.5 ⟺ calibrated; |E[Z]−G| in support units.
    "value_dist/ce": "dist CE",
    "value_dist/entropy": "dist entropy",
    "value_dist/std": "dist std",
    "value_dist/pit_mean": "dist PIT>.5",
    "value_dist/mean_abs_err": "dist E[Z]-G",
}


def _metric_label(key: str) -> str:
    """Display label for a metric row — a short override if known, else the part after '/'.
    Word-separator spaces are rendered as underscores so the curated short labels (``value share``)
    match the underscore house style of the raw TensorBoard keys (``ep_len_mean``); other notation
    in a label (``-``, ``/``, ``>``, ``|W|``) is intentional and left as-is."""
    return _METRIC_LABELS.get(key, key.partition("/")[2]).replace(" ", "_")


# Preferred display order; any unlisted keys are appended alphabetically.
_METRIC_ORDER = [
    # Eval — ELO headline first (anchored Bradley-Terry skill rating + its 95% CI), then
    # aggregate win rates, then per-opponent win rates, then per-opponent rewards.
    # Episode lengths fall alphabetically after (less actionable).
    "eval/elo",
    "eval/elo_ci",
    "eval/win_rate_mean",
    "eval/win_rate_vs_bots",
    "eval/win_rate_vs_pool",
    "eval/mean_reward_mean",
    "eval/mean_reward_vs_bots",
    "eval/mean_reward_vs_pool",
    "eval/win_rate_vs_random",
    "eval/win_rate_vs_heuristic",
    "eval/win_rate_vs_heuristic2",
    "eval/win_rate_vs_staller",
    "eval/win_rate_vs_staller_v2",
    "eval/win_rate_vs_aggressive",
    "eval/win_rate_vs_aggressive_v2",
    "eval/win_rate_vs_setup_sweep",
    "eval/win_rate_vs_setup_sweep_v2",
    "eval/mean_reward_vs_random",
    "eval/mean_reward_vs_heuristic",
    "eval/mean_reward_vs_heuristic2",
    "eval/mean_reward_vs_staller",
    "eval/mean_reward_vs_staller_v2",
    "eval/mean_reward_vs_aggressive",
    "eval/mean_reward_vs_aggressive_v2",
    "eval/mean_reward_vs_setup_sweep",
    "eval/mean_reward_vs_setup_sweep_v2",
    # Rollout
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    # Time
    "time/fps",
    "time/total_timesteps",
    # Train
    "train/approx_kl",
    "train/clip_fraction",
    "train/clip_range",
    "train/entropy_loss",
    "train/explained_variance",
    "train/learning_rate",
    "train/loss",
    "train/n_updates",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/grad_norm",
    # Gradient noise scale (only emitted under --grad-accum-steps>=2): B_simple = the critical batch
    # size, and its ratio to the effective batch (≫1 = noise-limited, ≪1 = diminishing returns).
    "train/noise_scale",
    "train/noise_scale_ratio",
    # Value-scale (PopArt prep): the (μ, σ) + tail an adaptive return normalizer would track,
    # and the value head's actual output spread. Watch for non-stationary scale drift.
    "train/return_mean",
    "train/return_std",
    "train/return_abs_max",
    "train/value_pred_std",
    # Opponent-mix curriculum telemetry (self-play only): the INTENDED per-episode opponent
    # probabilities — pool (self-play) share, stable cross-run share, and their sum (non-bot;
    # bot = 1 − nonbot). selfplay_fraction is POOL-only, NOT the curriculum coin pushed to envs.
    "train/nonbot_fraction",
    "train/selfplay_fraction",
    "train/stable_fraction",
    # Belief-aux diagnostics (only present when a belief aux is on) — rendered in their own
    # `belief/` section directly BELOW train/ (same column). species_acc is the headline; the
    # move_* block is the move-belief head; the latent_* block is the role-token latent predictor
    # (watch latent_std — →0 while latent_cosine→1 is the collapse NO-GO). The shared-trunk PULL of
    # this aux lives separately under grad/belief_* / grad/latent_*.
    "belief/species_acc",
    "belief/species_acc_above_chance",
    "belief/moves_precision",
    "belief/moves_recall",
    "belief/coverage",
    "belief/k_mean",
    "belief/species_ce",
    "belief/moves_bce",
    "belief/aux_loss",
    "belief/move_bce",
    "belief/move_precision",
    "belief/move_recall",
    "belief/move_revealed_slots",
    "belief/move_unrevealed_slots",
    "belief/move_loss",
    "belief/latent_cosine",
    "belief/latent_cosine_above_chance",
    "belief/latent_cosine_baseline",
    "belief/latent_std",
    "belief/latent_vicreg",
    "belief/latent_loss",
    # Gradient balance: value-vs-policy pull on the SHARED trunk. value_share ~0.5 = balanced,
    # →1 = value swamps the trunk; value_policy_logratio = log10(‖g_v‖/‖g_p‖) is the same imbalance
    # on a linear non-saturating scale (0 = balanced, >0 = value dominates) — the legible gauge for
    # watching a PopArt / vf_coef fix land; policy_value_cosine <0 = the heads conflict.
    # (See grad_balance.py.)
    "grad/value_share",
    "grad/value_policy_logratio",
    "grad/policy_value_cosine",
    "grad/policy_norm_shared",
    "grad/value_norm_shared",
    # Belief-aux pull on the shared trunk (only present when a belief aux is on): the COMBINED
    # belief_share (species CE + move BCE + latent) and the latent role-token predictor broken out on
    # its own (latent_share) — watch each sit ~5-15%; a spike with a degrading policy = the aux is
    # fighting the actor → lower its coef.
    "grad/belief_share",
    "grad/belief_norm_shared",
    "grad/belief_policy_cosine",
    "grad/latent_share",
    "grad/latent_norm_shared",
    "grad/latent_policy_cosine",
    # PopArt (only present under --use-popart): running value-target (mu, sigma) — should track
    # train/return_mean & return_std — and the POP-rescaled value-head weight norm (stays bounded).
    "popart/mu",
    "popart/sigma",
    "popart/value_weight_norm",
    # Distillation (only present under --distill-opponents) — headline first.
    "distill/all_distilled",
    "distill/frac_active_opponents_distilled",
    "distill/n_ready",
    "distill/n_running",
    "distill/n_exhausted",
]
