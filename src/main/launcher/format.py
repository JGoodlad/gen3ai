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
    # PopArt value-target normalizer (--use-popart).
    "popart/mu": "value mu",
    "popart/sigma": "value sigma",
    "popart/value_weight_norm": "value head |W|",
}


def _metric_label(key: str) -> str:
    """Display label for a metric row — a short override if known, else the part after '/'."""
    return _METRIC_LABELS.get(key, key.partition("/")[2])


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
