"""Batch 2's REGISTERED VOID guards (STAGED_PLAN §4.1.1 / §4.1.2 LR-F, §4.2 SPRT), read at a seed's arm end. Pure
judges over constructed rows plus thin readers of a run directory (read-only); ``python -m main.x35.kit void``
exposes them. A VOID seed enters no read and is re-run at the same pin and seed after its cause is fixed.

* **SPRT** (``sprt`` / ``toy_sprt``), from the seed's own append-only ``sprt_promotion.jsonl`` (T6), deterministic
  integer counts:
  - **V-S1, the lever did not act:** no DECIDED ``verdict`` line by the final ⇒ VOID.
  - **V-S2, restarts ate the test:** ABANDONED candidates > 25 % of the candidates (strictly; ``4 · abandoned >
    candidates``) ⇒ VOID.
* **LR-F** (``lrfloor`` / ``toy_lrfloor``), from the fork's OWN TensorBoard rows (``main.x35.tbrows``): a trailing-2M
  mean of ``train/approx_kl`` above **0.08** ⇒ VOID. Rule 8: a mean within 1e-9 of 0.08 is recorded, not a fire.

Nothing reads a seed before it has finished (``final_model.zip``): an unfinished seed is INCONCLUSIVE, never a PASS.

DECLARED HERE (choices the registration leaves open; each a FINDING of the F-AS-8 build):
* A CANDIDATE is a step with any log line. A candidate is ABANDONED when it has an ``abandoned`` line, a ``verdict``
  line whose ``reason`` is ``abandoned`` (an eval-core failure inside the test), or a ``start`` with no later line.
  V-S1 counts only DECIDED verdicts (``reason`` not ``abandoned``): a test cut short is not the lever acting.
* An unparseable log line could hide a verdict: INCONCLUSIVE, never read past.
* LR-F's windows: one per logged row at step ``s >= fork + 2M``, the mean of the rows in ``(s − 2M, s]``. Every
  window is judged (the by-hand read sees only the windows at its restarts and days, so this is the stricter reading).
  A step logged twice (a crash replay) keeps the LATER segment's value. A non-finite KL row is VOID (a broken trust
  region, as K9(b) treats a non-finite value). No row past ``fork + 2M``, or an unknown fork step, is INCONCLUSIVE.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PASS, VOID, INCONCLUSIVE = "PASS", "VOID", "INCONCLUSIVE"
RULE8_EPS = 1e-9
#: T6's log (``agents.training.sprt_promotion.LOG_NAME``), named here so the reader does not import the trainer.
SPRT_LOG = "sprt_promotion.jsonl"
#: V-S2's bar: abandoned candidates over candidates, strictly above.
V_S2_MAX_FRACTION = (1, 4)
LRF_KL_TAG = "train/approx_kl"
LRF_KL_BAR = 0.08
LRF_WINDOW_STEPS = 2_000_000
SPRT_ARMS = ("sprt", "toy_sprt")
LRF_ARMS = ("lrfloor", "toy_lrfloor")


@dataclass(frozen=True)
class VoidRead:
    run: str
    guard: str
    verdict: str
    detail: Dict[str, Any] = field(default_factory=dict)
    fires: Tuple[str, ...] = ()
    near: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"schema": "x35_batch2_void_v1", "run": self.run, "guard": self.guard, "verdict": self.verdict,
                "detail": self.detail, "fires": list(self.fires), "near_bar": list(self.near),
                "reasons": list(self.reasons)}


def _verdict(fires: Sequence[str], reasons: Sequence[str]) -> str:
    return VOID if fires else INCONCLUSIVE if reasons else PASS


# ------------------------------------------------------------------------------------------------ SPRT: V-S1 / V-S2
def parse_sprt_log(text: str) -> Tuple[List[Dict[str, Any]], int]:
    """``(rows, n_unparseable)`` of a ``sprt_promotion.jsonl`` text (blank lines skipped)."""
    rows: List[Dict[str, Any]] = []
    bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(r, dict) and "step" in r and "event" in r:
            rows.append(r)
        else:
            bad += 1
    return rows, bad


def judge_sprt(run: str, rows: Sequence[Dict[str, Any]], *, finished: bool, n_unparseable: int = 0) -> VoidRead:
    """V-S1 / V-S2 over one seed's log rows (module docstring)."""
    reasons: List[str] = []
    if not finished:
        reasons.append("the seed has no final_model.zip: the guards read at arm end")
    if n_unparseable:
        reasons.append(f"{n_unparseable} unparseable log line(s): a verdict could hide in one")
    last: Dict[int, Dict[str, Any]] = {}
    decided: Dict[int, bool] = {}
    abandoned = set()
    for r in rows:
        st = int(r["step"])
        last[st] = r
        ev = r.get("event")
        if ev == "abandoned" or (ev == "verdict" and r.get("reason") == "abandoned"):
            abandoned.add(st)
        elif ev == "verdict":
            decided[st] = bool(r.get("promoted"))
    abandoned |= {st for st, r in last.items() if r.get("event") == "start"}   # started, never ended
    abandoned -= set(decided)
    n_cand, n_ab, n_dec = len(last), len(abandoned), len(decided)
    fires: List[str] = []
    if n_dec == 0:
        fires.append("V-S1: no decided verdict line: the lever did not act")
    num, den = V_S2_MAX_FRACTION
    if den * n_ab > num * n_cand:
        fires.append(f"V-S2: {n_ab} of {n_cand} candidate(s) abandoned, more than {num}/{den}")
    detail = {"candidates": n_cand, "decided": n_dec, "abandoned": n_ab,
              "promoted": sum(decided.values()), "abandoned_steps": sorted(abandoned)}
    return VoidRead(run, "SPRT V-S1/V-S2", _verdict(fires, reasons), detail, tuple(fires), (), tuple(reasons))


def read_sprt(run_dir: str) -> VoidRead:
    run = os.path.basename(os.path.normpath(run_dir))
    p = Path(run_dir) / SPRT_LOG
    text = p.read_text(errors="replace") if p.is_file() else ""
    rows, bad = parse_sprt_log(text)
    return judge_sprt(run, rows, finished=(Path(run_dir) / "final_model.zip").is_file(), n_unparseable=bad)


# ------------------------------------------------------------------------------------------ LR-F: the KL guard
def judge_lrf_kl(run: str, fork: Optional[int], segments: Sequence[Sequence[Tuple[int, float]]], *,
                 finished: bool) -> VoidRead:
    """The trailing-2M mean approx-KL guard over the fork's own rows, ``segments`` in chronological order (a later
    segment's value for a step replaces an earlier one's)."""
    reasons: List[str] = []
    if not finished:
        reasons.append("the seed has no final_model.zip: the guard reads at arm end")
    if fork is None:
        reasons.append("the fork step is unknown: the inherited rows cannot be excluded")
        return VoidRead(run, "LR-F KL", INCONCLUSIVE, {}, (), (), tuple(reasons))
    by_step: Dict[int, float] = {}
    for seg in segments:
        for s, v in seg:
            if s > fork:
                by_step[int(s)] = float(v)
    steps = sorted(by_step)
    fires: List[str] = []
    near: List[str] = []
    bad = [s for s in steps if not math.isfinite(by_step[s])]
    if bad:
        fires.append(f"non-finite approx KL at step(s) {bad[:5]}")
    ends = [s for s in steps if s >= fork + LRF_WINDOW_STEPS]
    if not ends:
        reasons.append(f"no {LRF_KL_TAG} row at or past fork + {LRF_WINDOW_STEPS:,}: no full window")
    worst: Optional[Tuple[float, int]] = None
    lo = 0
    for s in ends if not bad else []:
        while steps[lo] <= s - LRF_WINDOW_STEPS:
            lo += 1
        win = [by_step[t] for t in steps[lo:] if t <= s]
        m = math.fsum(win) / len(win)
        if worst is None or m > worst[0]:
            worst = (m, s)
        if abs(m - LRF_KL_BAR) <= RULE8_EPS:
            near.append(f"window ending {s:,}: mean {m!r} within 1e-9 of {LRF_KL_BAR} (recorded, no fire)")
        elif m > LRF_KL_BAR:
            fires.append(f"window ending {s:,}: trailing-2M mean approx KL {m:.6f} > {LRF_KL_BAR}")
    detail = {"fork_step": fork, "rows_after_fork": len(steps), "windows": len(ends),
              "max_window_mean": worst[0] if worst else None, "max_window_end": worst[1] if worst else None}
    return VoidRead(run, "LR-F KL", _verdict(fires, reasons), detail, tuple(fires[:20]),
                    tuple(near), tuple(reasons))


def read_lrf_kl(run_dir: str) -> VoidRead:
    from main.x35.tbrows import fork_step, load_segments

    fork = fork_step(run_dir)
    segs = load_segments(run_dir, [LRF_KL_TAG], after_step=fork) if fork is not None else []
    return judge_lrf_kl(os.path.basename(os.path.normpath(run_dir)), fork,
                        [seg.rows.get(LRF_KL_TAG, []) for seg in segs],
                        finished=(Path(run_dir) / "final_model.zip").is_file())


def read_void(arm: str, run_dir: str) -> VoidRead:
    """The registered VOID guard of a batch-2 ``arm`` on one seed's run directory."""
    if arm in SPRT_ARMS:
        return read_sprt(run_dir)
    if arm in LRF_ARMS:
        return read_lrf_kl(run_dir)
    raise ValueError(f"arm {arm!r} has no batch-2 VOID guard ({SPRT_ARMS + LRF_ARMS})")
