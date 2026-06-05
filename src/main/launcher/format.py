"""Pure metric-formatting helpers + the dashboard display order.

Framework-agnostic (no Rich, no Textual) — consumed by the Textual dashboard
(`app.py`). Keep this the single source of truth for how a metric renders so the
numbers stay consistent everywhere.
"""


def _elapsed_str(seconds: float) -> str:
    h, rem = divmod(int(max(0, seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _fmt_val(v: float) -> str:
    """4 significant figures; comma-separated integers for large whole numbers."""
    if isinstance(v, float) and v == int(v) and abs(v) >= 1000:
        return f"{int(v):,}"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 1000 else str(v)
    return f"{v:.4g}"


def _fmt_metric(key: str, v: float) -> str:
    """Format a metric value; win rates render as percentages."""
    if "win_rate" in key:
        return f"{v * 100:.1f}%"
    return _fmt_val(v)


# Preferred display order; any unlisted keys are appended alphabetically.
_METRIC_ORDER = [
    # Eval — aggregate first, then per-opponent win rates, then per-opponent rewards.
    # Episode lengths fall alphabetically after (less actionable).
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
]
