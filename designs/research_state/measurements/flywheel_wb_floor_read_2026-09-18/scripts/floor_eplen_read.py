#!/usr/bin/env python3
"""THE 75M RUN-LEVEL FLOOR — the G7 EXCURSION at ~30M, across two seeds of one config.

Registration sec 8.4b made ``rollout/ep_len_mean`` and the stall/timeout fraction PRIMARY on the
win-prob arm, by its own ``[CRITIC]`` banner's instruction (*a [0,1] critic cannot express "a
timeout is worse than a loss"*). The completion entry for W_b
[ledger 2026-09-18, ``32273e1a``] banked that **the G7 excursion REPRODUCED across seeds** —
W_b's worst ratio 1.160 at ~30M against W's 1.131, both reversing.

This script reads the SERIES underneath that claim on both arms and asks two questions the
banked scalars cannot answer: **is it the same STEP RANGE, and is it the same MAGNITUDE?**

Instrument: the TensorBoard events via ``main.ops.tb_read``.
  * ``eval/mean_ep_len_vs_bots`` — the G7 ep_bots half. AMENDMENT 6: each arm is referenced to
    its OWN first 2M inside the self-play regime (the 4M-6M window, n = 2 cycles, frozen), bar
    1.25x, and a breach is a REPORT, never a kill.
  * ``eval/mean_ep_len_vs_pool`` — DESCRIPTIVE, NO BAR. Its population is non-stationary by
    construction (the sentinels gain a stronger self every 2M).
  * ``rollout/ep_len_mean`` — ~90% self-play on both arms, never comparable to the bot-eval
    series; the arm-to-arm comparison here is rollout-vs-rollout, which is matched.

🚨 This row is a DESCRIPTOR. No bar attaches to a cross-arm ep_len difference, and rule 12
binds: a timeout is never a semantic outcome. The frozen G7 reference is itself a DRAW-LEVEL
quantity (it moved 3.0 steps between these two seeds), so a ratio is comparable only WITHIN an
arm — which is exactly why both the ratio and the raw turns are printed.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/goodlad/dev/gen3ai/src")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from main.ops import tb_read  # noqa: E402

MODELS = Path("/home/goodlad/dev/gen3ai/models")
EP_BOTS = "eval/mean_ep_len_vs_bots"
EP_POOL = "eval/mean_ep_len_vs_pool"
ROLLOUT = "rollout/ep_len_mean"
REF_WINDOW = (4_000_000, 6_000_000)   # AMENDMENT 6: the arm's OWN first 2M inside self-play
REF_N = 2
G7_BAR = 1.25


def series(run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    pts = (tb_read.load(run_dir, [tag]) or {}).get(tag) or []
    if not pts:
        return np.array([]), np.array([])
    s = np.array([p[0] for p in pts], dtype=float)
    v = np.array([p[1] for p in pts], dtype=float)
    o = np.argsort(s)
    return s[o], v[o]


def read(run: str, human: str) -> dict:
    d = MODELS / run
    rec: dict = {"run": run, "human_description": human}
    crossing = tb_read.selfplay_crossing_step(d)
    rec["selfplay_crossing"] = crossing
    cs = crossing.get("step") if isinstance(crossing, dict) else None

    sb, vb = series(d, EP_BOTS)
    if not len(sb):
        return {**rec, "error": f"NO SCALAR for {EP_BOTS} — absence is not a zero"}
    ref_pts = vb[:REF_N]
    ref = float(ref_pts.mean()) if len(ref_pts) else float("nan")
    ratio = vb / ref
    peak = int(np.argmax(ratio))
    post = sb > (cs or 0)
    rec["ep_bots"] = {
        "tag": EP_BOTS, "n_cycles": int(len(sb)),
        "frozen_reference_turns": round(ref, 3),
        "frozen_reference_cycles": int(REF_N),
        "frozen_reference_n_cycles": int(len(ref_pts)),
        "frozen_reference_cycle_steps": [int(x) for x in sb[:REF_N]],
        "bar": G7_BAR,
        "worst_ratio": round(float(ratio.max()), 3),
        "worst_ratio_step": int(sb[peak]),
        "worst_ratio_turns": round(float(vb[peak]), 2),
        "pct_of_bar": round(100.0 * float(ratio.max()) / G7_BAR, 1),
        "final_ratio": round(float(ratio[-1]), 3),
        "final_turns": round(float(vb[-1]), 2),
        "post_crossing_mean_turns": round(float(vb[post].mean()), 2) if post.any() else None,
        "series": [{"step": int(a), "turns": round(float(b), 2), "ratio": round(float(c), 3)}
                   for a, b, c in zip(sb, vb, ratio)],
    }
    # THE ~30M EXCURSION the completion entry names, read as the LOCAL maximum inside a window
    # pinned HERE, before either arm's number was looked at: 24M-34M, the span the entry quotes
    # ("1.047 -> 1.087 -> 1.111 at 30M, back to 0.989 by 54M").
    med_ratio, med_turns = float(np.median(ratio)), float(np.median(vb))
    win = (sb >= 24_000_000) & (sb <= 34_000_000)
    if win.any():
        wi = np.flatnonzero(win)
        pk = wi[int(np.argmax(ratio[win]))]
        # walk out from the local peak while the series is still rising towards it
        lo_i = pk
        while lo_i - 1 >= 0 and vb[lo_i - 1] < vb[lo_i] and lo_i - 1 >= wi[0] - 1:
            lo_i -= 1
        hi_i = pk
        while hi_i + 1 < len(vb) and vb[hi_i + 1] < vb[hi_i] and ratio[hi_i + 1] > med_ratio:
            hi_i += 1
        rec["ep_bots"]["excursion_24M_34M"] = {
            "definition": ("the LOCAL maximum of eval/mean_ep_len_vs_bots inside a 24M-34M window "
                           "pinned before either arm was read — the span the completion entry "
                           "quotes. A DESCRIPTOR on the arm's own series, never a bar."),
            "rise_from_step": int(sb[lo_i]), "rise_from_turns": round(float(vb[lo_i]), 2),
            "peak_step": int(sb[pk]), "peak_turns": round(float(vb[pk]), 2),
            "peak_ratio": round(float(ratio[pk]), 3),
            "pct_of_bar_at_peak": round(100.0 * float(ratio[pk]) / G7_BAR, 1),
            "recovered_at_step": (int(sb[hi_i + 1]) if hi_i + 1 < len(sb) else None),
            "recovered_turns": (round(float(vb[hi_i + 1]), 2) if hi_i + 1 < len(sb) else None),
            "recovered_ratio": (round(float(ratio[hi_i + 1]), 3) if hi_i + 1 < len(sb) else None),
            "arm_median_turns": round(med_turns, 2), "arm_median_ratio": round(med_ratio, 3),
            "amplitude_turns_over_arm_median": round(float(vb[pk]) - med_turns, 2),
            "amplitude_turns_over_rise_start": round(float(vb[pk] - vb[lo_i]), 2),
            "ratio_series_26_34M": [{"step": int(a), "turns": round(float(b), 2),
                                     "ratio": round(float(c), 3)}
                                    for a, b, c in zip(sb, vb, ratio)
                                    if 26_000_000 <= a <= 34_000_000],
        }
    rec["ep_bots"]["global_worst"] = {
        "step": int(sb[peak]), "turns": round(float(vb[peak]), 2),
        "ratio": round(float(ratio.max()), 3),
        "note": "the run's WORST ratio anywhere — not necessarily the 30M excursion",
    }

    sp, vp = series(d, EP_POOL)
    if len(sp):
        rec["ep_pool_DESCRIPTIVE_NO_BAR"] = {
            "n_cycles": int(len(sp)), "final_turns": round(float(vp[-1]), 2),
            "max_turns": round(float(vp.max()), 2), "max_step": int(sp[int(np.argmax(vp))]),
            "note": ("non-stationary by construction (the sentinel pool gains a stronger self "
                     "every 2M) — no bar, no verdict")}

    sr, vr = series(d, ROLLOUT)
    if len(sr):
        pm = sr > (cs or 0)
        rec["rollout_ep_len_mean"] = {
            "n_points": int(len(sr)),
            "post_crossing_mean": round(float(vr[pm].mean()), 2) if pm.any() else None,
            "post_crossing_max": round(float(vr[pm].max()), 2) if pm.any() else None,
            "post_crossing_max_step": int(sr[pm][int(np.argmax(vr[pm]))]) if pm.any() else None,
            "median20_at_end": round(float(np.median(vr[-20:])), 2),
            "last_value": round(float(vr[-1]), 2),
            "note": ("~90% self-play on both arms; never comparable to the bot-eval series. "
                     "The arm-to-arm comparison is rollout-vs-rollout, which IS matched, and no "
                     "bar attaches to it.")}
    return rec


def main() -> int:
    out = {"generated_at": _dt.datetime.now().astimezone().isoformat(),
           "instrument": "TensorBoard events via main.ops.tb_read",
           "amendment": ("READ AMENDMENT 6: the ep_bots half is referenced to the arm's OWN first "
                         "2M inside the self-play regime, bar 1.25x, and a breach is a REPORT, "
                         "never a kill. The ep_pool half is DESCRIPTIVE with NO BAR."),
           "runs": {}}
    for run, human in (("ai_v13_02_flywheel_winprob", "arm W -- the WIN-PROB arm, seed 1001"),
                       ("ai_v13_04_flywheel_winprob_b", "W_b -- arm W's TOKEN-EXACT SEED REPLICATE, seed 1002"),
                       ("ai_v13_01_flywheel_shaped", "arm S -- the SHAPED arm, seed 1001 (context only)")):
        try:
            out["runs"][run] = read(run, human)
        except Exception as exc:                       # noqa: BLE001
            out["runs"][run] = {"run": run, "error": f"{type(exc).__name__}: {exc}"}

    W = out["runs"].get("ai_v13_02_flywheel_winprob", {})
    B = out["runs"].get("ai_v13_04_flywheel_winprob_b", {})
    if "ep_bots" in W and "ep_bots" in B:
        ew = W["ep_bots"].get("excursion_24M_34M")
        eb = B["ep_bots"].get("excursion_24M_34M")
        out["excursion_across_seeds"] = {
            "question": "same STEP RANGE? same MAGNITUDE?",
            "armW": ew, "armWb": eb,
            "peak_step_gap_steps": (abs(ew["peak_step"] - eb["peak_step"]) if ew and eb else None),
            "same_peak_step": (bool(ew["peak_step"] == eb["peak_step"]) if ew and eb else None),
            "rise_windows": ({"armW": [ew["rise_from_step"], ew["peak_step"]],
                              "armWb": [eb["rise_from_step"], eb["peak_step"]]} if ew and eb else None),
            "peak_ratio_delta": (round(eb["peak_ratio"] - ew["peak_ratio"], 3) if ew and eb else None),
            "amplitude_turns_over_arm_median_delta": (
                round(eb["amplitude_turns_over_arm_median"] - ew["amplitude_turns_over_arm_median"], 2)
                if ew and eb else None),
            "amplitude_turns_over_rise_start_delta": (
                round(eb["amplitude_turns_over_rise_start"] - ew["amplitude_turns_over_rise_start"], 2)
                if ew and eb else None),
            "global_worst": {"armW": W["ep_bots"]["global_worst"], "armWb": B["ep_bots"]["global_worst"]},
            "frozen_reference_turns": {"armW": W["ep_bots"]["frozen_reference_turns"],
                                       "armWb": B["ep_bots"]["frozen_reference_turns"],
                                       "delta": round(B["ep_bots"]["frozen_reference_turns"]
                                                      - W["ep_bots"]["frozen_reference_turns"], 3)},
            "reading": ("A DESCRIPTOR. The frozen reference is itself a draw-level quantity, so a "
                        "RATIO is comparable only WITHIN an arm; the raw turns are printed beside "
                        "it for the cross-arm read. No bar attaches, and rule 12 binds: a timeout "
                        "is never a semantic outcome."),
        }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("floor_eplen_read.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out.get("excursion_across_seeds"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
