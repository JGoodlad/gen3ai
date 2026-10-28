"""THE PER-ROW CLIP-MARGIN BAR, STAGE 1 (`gen3_canary_row_bar_v1`; ruled 2026-10-28,
`designs/training/canary_recalibration_2026-10-28.md`).

WHAT CHANGED. Until this pin the canary held the COMPILED arm to ONE global margin M (`compile_regions.R1_CLIP_MARGIN`
0.72, or the `--canary-shift-max` raise, or the yardstick re-run's M'): its largest |Δ log π| from the fp32 reference,
over every fixture row, had to be <= M. Now each fixture row i of the compiled arm has its OWN bar

    b_i = min(ROW_BAR_K x max(null_i, ROW_BAR_FLOOR), ROW_BAR_CAP)          k 6.0, floor 0.02, cap 0.72

where null_i is row i's fp32 dither null on the canary's own weights, computed at check time (`canary_row_null`: K =
`ROW_BAR_SEEDS` 16 seeds of relative 2^-9), and the arm refuses iff some row's shift exceeds its bar. k is 4 x the
healthy maximum (1.479) of the canary's two bf16 arms over the W0 CUDA census (20 X26-lineage checkpoints x 64 rows).

WHAT DID NOT CHANGE.
  * The relocation stays ONE scalar (`relocate_clip_boundary` at the model's margin), and every per-row bar is <= the
    cap <= that margin, so rule 8 holds by construction (no arm's bar passes the relocation margin).
  * The yardsticks (eager, the fp32 reference) keep the scalar rule, its events and the M' re-run; only the JUDGED arm
    takes per-row bars, and neither `--canary-shift-max` nor M' raises them past the cap (stage 1 is strictly tighter
    than the global rule: the cap IS M).
  * The startup gate (`gate_regions`) keeps the scalar rule: only the scheduled canary and its two confirmation halves
    compute a null.

THE FALLBACK. The null runs under its wall cap (`canary_row_null.cap_seconds`): when it is off, skipped, truncated,
in error or has fewer than `ROW_BAR_SEEDS` seeds, every row's bar is the CAP, which is the global rule at M = 0.72. The
record names the reason (``row_bars.source``); the per-row bars never fall back to anything looser than the cap.

THE VERSION. Every canary record carries ``canary_rule_version`` = `CANARY_RULE_VERSION`. A record without the field
was written at a pin before this change and judged the compiled arm at the global M (`LEGACY_RULE_VERSION`);
`rule_version_of` reads either. A run's rule is its pin's code: a run pinned before this commit never sees it.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, NamedTuple, Optional

import numpy as np
import torch

from agents.model import canary_row_null as RN

#: The rule every canary record of this pin names (``canary_rule_version``).
CANARY_RULE_VERSION = "per_row_stage1"
#: What a record WITHOUT ``canary_rule_version`` judged by: the compiled arm at the global margin M.
LEGACY_RULE_VERSION = "global_m"
#: k: 4 x 1.479, the healthy maximum of row shift / max(16-seed null, 0.02) over the census's compiled and eager bf16
#: arms at B = 2,048 on CUDA (X26 380M, row 63, eager), rounded UP to 2 significant digits (R1's convention).
ROW_BAR_K = 6.0
#: The null's floor (`canary_row_null.ROW_NULL_FLOOR`): a row whose null is pure rounding is held at k x 0.02 = 0.12.
ROW_BAR_FLOOR = RN.ROW_NULL_FLOOR
#: The null's seed count the k was derived at (`canary_row_null.ROW_NULL_SEEDS`); a null with fewer seeds falls back.
ROW_BAR_SEEDS = RN.ROW_NULL_SEEDS
#: Stage 1's cap C = M = `compile_regions.R1_CLIP_MARGIN` (pinned equal by `canary_row_bar_test`). Stage 2 (C = 1.2)
#: waits on a leased read of the gradient rule on rows relocated at 1.2.
ROW_BAR_CAP = 0.72


def rule_version_of(record: Mapping[str, Any]) -> str:
    """The canary rule a ``canary_verdicts.jsonl`` record was judged by (`LEGACY_RULE_VERSION` when it carries none)."""
    return str(record.get("canary_rule_version") or LEGACY_RULE_VERSION)


def row_bars(null: Any, *, k: float = ROW_BAR_K, floor: float = ROW_BAR_FLOOR, cap: float = ROW_BAR_CAP) -> np.ndarray:
    """``min(k x max(null_i, floor), cap)`` per row. A non-finite null reads as unbounded, so its row sits at the cap."""
    n = np.asarray(null, dtype=np.float64)
    n = np.where(np.isfinite(n), n, np.inf)
    out: np.ndarray = np.minimum(float(k) * np.maximum(n, float(floor)), float(cap))
    return out


class RowBars(NamedTuple):
    """One canary's per-FIXTURE-row bars. ``bars`` None = the fallback (every row at the cap); ``source`` is
    ``"null"`` or ``"fallback: <why>"``."""
    bars: Optional[np.ndarray]
    source: str
    null: Optional[np.ndarray] = None

    @property
    def live(self) -> bool:
        return self.bars is not None

    @property
    def label(self) -> str:
        """The bars' kind for a rule line or a refusal (the reason itself is in the record's ``canary_rule``)."""
        return "the null" if self.bars is not None else "the fallback: every row at the cap"

    def for_batch(self, fixture_rows: Any) -> "BatchRowBars":
        """The bars per BATCH row of a check whose batch row j tiles fixture row ``fixture_rows[j]``."""
        idx = np.asarray(fixture_rows, dtype=np.int64).reshape(-1)
        if self.bars is None:
            return BatchRowBars(np.full(idx.shape, ROW_BAR_CAP), idx, self)
        if idx.size and (idx.min() < 0 or idx.max() >= self.bars.size):
            raise ValueError(f"fixture rows {int(idx.min())}..{int(idx.max())} outside the {self.bars.size} rows "
                             "the null was computed on")
        return BatchRowBars(self.bars[idx], idx, self)

    def record(self) -> Dict[str, Any]:
        """The record's ``canary_rule`` block: the rule's constants and where this canary's bars came from."""
        rec: Dict[str, Any] = {"version": CANARY_RULE_VERSION, "k": ROW_BAR_K, "floor": ROW_BAR_FLOOR,
                               "seeds": ROW_BAR_SEEDS, "cap": ROW_BAR_CAP, "row_bars_source": self.source}
        if self.bars is not None:
            rec["bar_per_row"] = [float(f"{x:.4g}") for x in self.bars]
        return rec


class BatchRowBars(NamedTuple):
    """`RowBars` laid out per batch row (``bars[j]`` is fixture row ``fixture_rows[j]``'s bar)."""
    bars: np.ndarray
    fixture_rows: np.ndarray
    rows: RowBars


def fallback(why: str) -> RowBars:
    """Every row at the cap (the global rule at M = `ROW_BAR_CAP`), with the reason recorded."""
    return RowBars(None, f"fallback: {why}")


def from_null(raw: Optional[Mapping[str, Any]]) -> RowBars:
    """The bars from `canary_row_null.compute`'s result, or the FALLBACK naming why it cannot bind (off, skipped,
    truncated, an error, or fewer seeds than `ROW_BAR_SEEDS`)."""
    if raw is None:
        return fallback("the per-row null is off for this canary")
    for key in ("error", "skipped", "truncated"):
        if raw.get(key):
            return fallback(f"the null {key}: {str(raw[key])[:200]}")
    null = raw.get("null")
    if null is None:
        return fallback("the null returned no rows")
    per = np.asarray(raw.get("per_seed", np.zeros((0, 0))))
    seeds = int(per.shape[0]) if per.ndim == 2 else 0
    if seeds != ROW_BAR_SEEDS:
        return fallback(f"the null has {seeds} seeds, the rule's k was derived at {ROW_BAR_SEEDS}")
    null = np.asarray(null, dtype=np.float64)
    return RowBars(row_bars(null), "null", null)


def judge(ref_log_prob: torch.Tensor, arms: Mapping[str, Mapping[str, torch.Tensor]],
          batch: BatchRowBars) -> Dict[str, Dict[str, Any]]:
    """Per JUDGED arm: its worst row by shift / bar (batch row, fixture row, shift, bar, fraction) and whether it
    FAILS (some row's shift > its bar; a non-finite shift fails). ``arms`` must hold the judged arms only."""
    ref = ref_log_prob.detach().double().cpu().reshape(-1)
    bars = torch.as_tensor(batch.bars, dtype=torch.float64).reshape(-1)
    if bars.shape != ref.shape:
        raise ValueError(f"per-row bars for {bars.numel()} batch rows, the fixture has {ref.numel()}")
    out: Dict[str, Dict[str, Any]] = {}
    for name, arm in arms.items():
        d = torch.nan_to_num((arm["log_prob"].detach().double().cpu().reshape(-1) - ref).abs(), nan=float("inf"))
        frac = d / bars
        row = int(frac.argmax()) if frac.numel() else -1
        f = float(frac[row]) if row >= 0 else 0.0
        out[name] = {"fraction": f, "row": row,
                     "fixture_row": int(batch.fixture_rows[row]) if row >= 0 else -1,
                     "shift": float(d[row]) if row >= 0 else 0.0, "bar": float(bars[row]) if row >= 0 else 0.0,
                     "fails": not f <= 1.0}
    return out


def line(per_row: Mapping[str, Mapping[str, Any]], batch: BatchRowBars) -> str:
    """The rule-line fragment: the source of the bars and each judged arm's worst row against its bar."""
    arms = ", ".join(f"{n} {d['shift']:.3g} / {d['bar']:.3g} = {d['fraction']:.3f} (fixture row {d['fixture_row']})"
                     for n, d in per_row.items())
    return (f"per-row bars [{CANARY_RULE_VERSION}: min({ROW_BAR_K:g} x max(null, {ROW_BAR_FLOOR:g}), "
            f"{ROW_BAR_CAP:g}); {batch.rows.label}]: {arms}")


def refusal(name: str, d: Mapping[str, Any], flips: int, batch: BatchRowBars) -> str:
    """The judged arm's refusal text under the per-row rule."""
    return (f"--compile-trainer region R1: the {name} arm moved fixture row {d['fixture_row']} (batch row {d['row']}) "
            f"by {d['shift']:.3g} from the fp32 eager reference, past that row's bar {d['bar']:.3g} = min({ROW_BAR_K:g} "
            f"x max(its dither null, {ROW_BAR_FLOOR:g}), {ROW_BAR_CAP:g}) [{CANARY_RULE_VERSION}; bars from "
            f"{batch.rows.label}], and switched {flips} row(s)' PPO clip branch. The per-row bar is k x the row's own "
            f"fp32 dither null (gen3_canary_row_bar_v1): a healthy bf16 arm read at most 1.48 x it over the W0 census. "
            f"Replay the dump (python -m main.ops.canary_dump_replay --dump) before any resume.")
