"""Core TensorBoard plotting logic.

Reads scalar events from a TensorBoard directory and writes one PNG per metric
group (eval, train, rollout, …) into an output directory.  Intended to be called
from the launcher TUI (press `p`) or from the standalone CLI `plot_run.py`.

Visual style matches TensorBoard: dark background, raw (faint) + EMA-smoothed
(solid) lines, one subplot per tag, 3-column grid per group.
"""

import glob
import os
import threading

import matplotlib
matplotlib.use("Agg")  # headless — must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils.git import get_main_repo_root


# ---------------------------------------------------------------------------
# Pinned tags — populate this list to add a "pinned" group as the first PNG,
# containing hand-picked key charts regardless of their prefix.
# Example:
#   PINNED_TAGS = ["eval/win_rate_mean", "eval/win_rate_vs_Heuristic",
#                  "rollout/ep_rew_mean"]
# ---------------------------------------------------------------------------
PINNED_TAGS: list[str] = []

# Render order for groups; any group not listed is appended alphabetically.
_GROUP_ORDER = ["pinned", "eval", "rollout", "train", "time", "hparams"]

# Visual constants
_ACCENT = "#bb44ff"          # TensorBoard purple
_FIG_BG = "#1a1a1a"
_AXES_BG = "#262626"
_GRID_ALPHA = 0.15
_NCOLS = 3
_MAX_PER_PAGE = 6            # charts per PNG (2 rows × 3 cols)
_ROW_HEIGHT = 5.0            # inches per row of subplots
_FIG_WIDTH = 24.0            # inches
_DPI = 200

# Mutex so concurrent TUI presses don't stomp each other's matplotlib state.
_plot_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def find_tb_dir_for_run(run_dir: str) -> str | None:
    """Derive the tensorboard directory from a model run directory.

    Extracts the timestamp from ``models/run_YYYYMMDD_HHMMSS`` and globs
    ``<repo>/tensorboard/*<timestamp>*/`` for a match.

    Returns the first match (most recently modified if several), or None.
    """
    run_name = os.path.basename(run_dir.rstrip("/\\"))
    if not run_name.startswith("run_"):
        return None
    timestamp = run_name[len("run_"):]  # e.g. "20260521_091605"

    try:
        repo_root = get_main_repo_root()
    except Exception:
        return None

    tb_root = os.path.join(repo_root, "tensorboard")
    if not os.path.isdir(tb_root):
        return None

    matches = [
        m for m in glob.glob(os.path.join(tb_root, f"*{timestamp}*"))
        if os.path.isdir(m)
    ]
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def generate_plots(tb_dir: str, out_dir: str, smoothing: float = 0.75) -> list[str]:
    """Read ``tb_dir``, render one PNG per metric group into ``out_dir``.

    Args:
        tb_dir:    Path to a TensorBoard run directory (contains tfevents files).
        out_dir:   Destination directory; created if it doesn't exist.
        smoothing: EMA weight (0 = no smoothing, 0.99 = very heavy).

    Returns:
        List of absolute paths to the written PNG files.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # lazy import

    os.makedirs(out_dir, exist_ok=True)

    ea = EventAccumulator(tb_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    # Group tags by prefix (everything before the first "/").
    groups: dict[str, list[str]] = {}
    for tag in tags:
        prefix = tag.partition("/")[0] if "/" in tag else "misc"
        groups.setdefault(prefix, []).append(tag)

    # Inject pinned group if configured.
    if PINNED_TAGS:
        tag_set = set(tags)
        pinned = [t for t in PINNED_TAGS if t in tag_set]
        if pinned:
            groups["pinned"] = pinned

    # Determine render order.
    order: list[str] = [g for g in _GROUP_ORDER if g in groups]
    for g in sorted(groups):
        if g not in order:
            order.append(g)

    plt.style.use("dark_background")

    written: list[str] = []

    with _plot_lock:
        for group in order:
            group_tags = sorted(groups[group])

            # Read scalar data for every tag in this group.
            data: dict[str, tuple[list, list]] = {}
            for tag in group_tags:
                try:
                    events = ea.Scalars(tag)
                except Exception:
                    continue
                if not events:
                    continue
                steps = [e.step for e in events]
                values = [e.value for e in events]
                data[tag] = (steps, values)

            if not data:
                continue

            tags_with_data = sorted(data.keys())
            pages = _chunk(tags_with_data, _MAX_PER_PAGE)
            multi = len(pages) > 1

            for page_idx, page_tags in enumerate(pages):
                n = len(page_tags)
                ncols = min(_NCOLS, n)
                nrows = (n + ncols - 1) // ncols

                fig, axes = plt.subplots(
                    nrows, ncols,
                    figsize=(_FIG_WIDTH, nrows * _ROW_HEIGHT),
                    squeeze=False,
                )
                fig.patch.set_facecolor(_FIG_BG)

                for idx, tag in enumerate(page_tags):
                    row, col = divmod(idx, ncols)
                    ax = axes[row][col]
                    steps, values = data[tag]

                    # Raw line (faint).
                    ax.plot(steps, values, color=_ACCENT, alpha=0.2, linewidth=1.0)

                    # EMA-smoothed line (solid).
                    smoothed = values
                    if len(values) > 1:
                        smoothed = _ema(values, smoothing)
                        ax.plot(steps, smoothed, color=_ACCENT, alpha=1.0, linewidth=2.0)

                    # Style the axes.
                    ax.set_facecolor(_AXES_BG)
                    ax.grid(True, color="white", alpha=_GRID_ALPHA, linewidth=0.5)
                    ax.tick_params(axis="y", colors="white", labelsize=9)
                    ax.tick_params(axis="x", colors="white", labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#444444")

                    # X-axis: abbreviated step counts.
                    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_step))

                    # Pin win-rate axes to [0, 1] so all rate charts share a scale.
                    if "win_rate" in tag:
                        ax.set_ylim(0.0, 1.0)

                    # Annotate the final smoothed value in the corner for quick reading.
                    ax.annotate(
                        f"{smoothed[-1]:.4g}",
                        xy=(0.98, 0.95), xycoords="axes fraction",
                        ha="right", va="top", color="#dddddd", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#333333", ec="none", alpha=0.85),
                    )

                    # Title: strip the group prefix.
                    short = tag.partition("/")[2] if "/" in tag else tag
                    ax.set_title(short, color="white", fontsize=11, pad=6)

                # Hide unused subplot slots.
                for idx in range(n, nrows * ncols):
                    row, col = divmod(idx, ncols)
                    axes[row][col].set_visible(False)

                title = f"{group} ({page_idx + 1}/{len(pages)})" if multi else group
                fig.suptitle(title, color="#aaaaaa", fontsize=10, y=1.01)
                fig.tight_layout(h_pad=2.0, w_pad=1.5)

                stem = f"{group}_{page_idx + 1}" if multi else group
                path = os.path.join(out_dir, f"{stem}.png")
                fig.savefig(path, dpi=_DPI, bbox_inches="tight",
                            facecolor=_FIG_BG, edgecolor="none")
                plt.close(fig)
                written.append(os.path.abspath(path))

    return written


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk(items: list, size: int) -> list[list]:
    """Split a list into consecutive chunks of at most *size* elements."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def _ema(values: list[float], weight: float) -> list[float]:
    """Exponential moving average — same formula as TensorBoard's smoothing."""
    last = values[0]
    out: list[float] = []
    for v in values:
        last = last * weight + (1.0 - weight) * v
        out.append(last)
    return out


def _fmt_step(x: float, _pos) -> str:
    """Abbreviate large step counts: 140000000 → '140M', 5000 → '5K'."""
    if x >= 1_000_000:
        return f"{x / 1_000_000:.0f}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}K"
    return str(int(x))
