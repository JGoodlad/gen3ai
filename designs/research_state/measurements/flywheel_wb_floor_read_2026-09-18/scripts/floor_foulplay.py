#!/usr/bin/env python3
"""THE 75M RUN-LEVEL FLOOR — Foul Play at `--search-time-ms 1000`, and the RULE-23 width comparison.

Foul Play has **NO iteration budget** — `--search-time-ms` is wall clock only
(`fp/search/main.py:53`) — which makes it a **WIDTH METER** under standing rule 23: *a meter whose
value depends on the box's throughput needs a CONTEMPORANEOUS control or width matching, never a
fixed bar.* So every read carries its realized MCTS visits per decision, and a Δ between two
campaigns is quoted ONLY with the two realized widths beside it.

Inputs: W_b's merged `games.jsonl` (this campaign, 2026-09-18) and the banked arm-W
(`flywheel_pair_read_2026-09-15/out/foul_play/`) and arm-S
(`flywheel_armS_reads_2026-09-14/out/foul_play/`) ones, plus the 0.02 leg's de-risk summary.

🚨 **NO MODEL CLAIM MAY BE TAKEN FROM THIS ROW WHILE THE WIDTHS ARE UNMATCHED**, and across the
three banked campaigns the win rates already ordered EXACTLY inversely to realized width
(1.40 M -> 0.388, 1.249 M -> 0.450, 1.153 M -> 0.475). The floor computed here is therefore a
floor on a WIDTH METER read under a different box load, and it is reported as such.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

Z = 1.959963985
B_ROOT = Path("/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/foul_play")
W_ROOT = Path("/home/goodlad/dev/gen3ai/designs/research_state/measurements/"
              "flywheel_pair_read_2026-09-15/out/foul_play")
S_ROOT = Path("/home/goodlad/dev/gen3ai/designs/research_state/measurements/"
              "flywheel_armS_reads_2026-09-14/out/foul_play")
# The 0.02 leg's banked de-risk, same harness, same 8 pinned pool teams, 80 games.
DERISK = {"wins": 31, "n": 80, "visits": 1.40e6,
          "label": "ai_v12_02_winprob_critic@75M (the 0.02 leg), foul_play_derisk_2026-09-14"}


def wilson(k: int, n: int):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def load(root: Path) -> list[dict]:
    return [json.loads(ln) for ln in (root / "games.jsonl").read_text().splitlines() if ln.strip()]


def campaign(games: list[dict], label: str) -> dict:
    n = len(games)
    w = sum(1 for g in games if g["our_win"])
    p, lo, hi = wilson(w, n)
    vis = [g["fp_mcts_visits_mean"] for g in games if g.get("fp_mcts_visits_mean")]
    think = [g["our_think_ms_mean"] for g in games if g.get("our_think_ms_mean")]
    fpt = [g["fp_think_s_mean"] for g in games if g.get("fp_think_s_mean")]
    by_sess: dict = defaultdict(list)
    for g in games:
        by_sess[g.get("team_file")].append(g)      # one PINNED pool team per session
    sessions = []
    for s in sorted(by_sess, key=lambda x: (x is None, str(x))):
        rs = by_sess[s]
        sv = [g["fp_mcts_visits_mean"] for g in rs if g.get("fp_mcts_visits_mean")]
        sessions.append({
            "our_team": s,
            "games": len(rs), "our_wins": sum(1 for g in rs if g["our_win"]),
            "win_rate": round(sum(1 for g in rs if g["our_win"]) / len(rs), 3),
            "mean_turns": round(sum(g["turns"] for g in rs) / len(rs), 1),
            "fp_visits_per_decision_mean": round(sum(sv) / len(sv), 1) if sv else None})
    return {
        "label": label, "n": n, "our_wins": w, "win_rate": round(p, 4),
        "wilson95": [round(lo, 4), round(hi, 4)],
        "mean_turns": round(sum(g["turns"] for g in games) / n, 1),
        "max_turns": max(g["turns"] for g in games),
        "our_forfeit_fired_n": sum(1 for g in games if g.get("our_forfeit_fired")),
        "our_think_ms_mean": round(sum(think) / len(think), 1) if think else None,
        "fp_think_s_mean": round(sum(fpt) / len(fpt), 3) if fpt else None,
        "fp_visits_per_decision_mean": round(sum(vis) / len(vis), 1) if vis else None,
        "fp_visits_session_range": [min(s["fp_visits_per_decision_mean"] for s in sessions
                                        if s["fp_visits_per_decision_mean"]),
                                    max(s["fp_visits_per_decision_mean"] for s in sessions
                                        if s["fp_visits_per_decision_mean"])],
        "games_without_a_matched_fp_log": sum(1 for g in games if not g.get("fp_log")),
        "fp_error_lines_total": sum(g.get("fp_errors_in_session_log", 0) for g in games),
        "sessions": sessions,
    }


def main() -> int:
    out: dict = {"instrument": ("Foul Play @6c467c08 + poke-engine 0.0.48 (--features "
                                "poke-engine/gen3), --search-time-ms 1000 --search-parallelism 1 "
                                "--search-threads 1; 8 sessions x 10 games, our side pinning one "
                                "pool team per session (the same 8 files, same sorted export, same "
                                "stride on both arms), Foul Play drawing a random pool team per "
                                "battle from the same export"),
                 "rule_23": ("Foul Play has NO iteration budget — --search-time-ms is WALL CLOCK "
                             "(fp/search/main.py:53) — so it is a WIDTH meter and every read "
                             "carries its realized visits/decision. A delta is quoted ONLY with "
                             "the two realized widths beside it."),
                 "campaigns": {}}
    out["campaigns"]["armWb"] = campaign(load(B_ROOT), "W_b (ai_v13_04_flywheel_winprob_b) "
                                         "@75,005,952, 2026-09-18")
    out["campaigns"]["armW"] = campaign(load(W_ROOT), "arm W (ai_v13_02_flywheel_winprob) "
                                        "@75,005,952, 2026-09-16")
    out["campaigns"]["armS"] = campaign(load(S_ROOT), "arm S (ai_v13_01_flywheel_shaped) "
                                        "@75,005,952, 2026-09-14")

    def contrast(a, b, label):
        ca, cb = out["campaigns"][a], out["campaigns"][b]
        d, lo, hi = newcombe(ca["our_wins"], ca["n"], cb["our_wins"], cb["n"])
        ratio = (cb["fp_visits_per_decision_mean"] / ca["fp_visits_per_decision_mean"]
                 if ca["fp_visits_per_decision_mean"] else None)
        return {"label": label, "delta": round(d, 4), "abs_delta": round(abs(d), 4),
                "newcombe95": [round(lo, 4), round(hi, 4)],
                f"{a}_realized_visits": ca["fp_visits_per_decision_mean"],
                f"{b}_realized_visits": cb["fp_visits_per_decision_mean"],
                "width_ratio_b_over_a": round(ratio, 4) if ratio else None,
                "width_matched": bool(ratio and 0.95 <= ratio <= 1.05),
                "verdict": ("NOT DETECTED — the difference interval covers zero; never 'equal' "
                            "(rule 6)" if lo <= 0 <= hi else
                            "DETECTED at 95% — the interval excludes zero"),
                "rule_23": ("a difference between two campaigns at UNMATCHED width may not be "
                            "attributed to a model")}

    out["THE_FLOOR_W_minus_Wb"] = contrast("armW", "armWb",
        "THE 75M RUN-LEVEL FLOOR on this cell — arm W (seed 1001) minus W_b (seed 1002). 🚨 It is "
        "a floor on a WIDTH METER, measured at a DIFFERENT box load, so it bounds "
        "run-to-run-plus-width variation and not run-to-run variation alone.")
    out["THE_FINDING_S_minus_W"] = contrast("armS", "armW",
        "the pair's row — arm S minus arm W (the pair read reported -0.025 [-0.175, +0.127])")
    out["S_minus_Wb"] = contrast("armS", "armWb", "arm S minus W_b")
    f, g = out["THE_FLOOR_W_minus_Wb"], out["THE_FINDING_S_minus_W"]
    fl, dd = f["abs_delta"], abs(g["delta"]); lo, hi = g["newcombe95"]
    inside = (lo <= fl <= hi) or (lo <= -fl <= hi)
    out["floor_verdict"] = {
        "finding_abs": dd, "finding_ci95": [lo, hi], "floor_abs": fl,
        "clause_a_abs_delta_gt_floor": bool(dd > fl),
        "clause_b_ci_excludes_floor_point": bool(not inside),
        "verdict": ("OUTSIDE THE 75M FLOOR" if (dd > fl and not inside) else "WITHIN FLOOR at n = 2"),
        "caveats": ["🚨 rule 23 overrides the verdict either way: the widths are not matched and no "
                    "difference among these campaigns may be attributed to a model",
                    "one replicate pair BOUNDS a floor; no CI attaches to it (rules 19/22)",
                    "WITHIN FLOOR is NEVER 'equivalent' (rule 6)"]}
    b = out["campaigns"]["armWb"]
    d2, lo2, hi2 = newcombe(DERISK["wins"], DERISK["n"], b["our_wins"], b["n"])
    out["third_leg_minus_armWb"] = {
        "label": DERISK["label"], "n": DERISK["n"],
        "win_rate": round(DERISK["wins"] / DERISK["n"], 4),
        "realized_visits": DERISK["visits"],
        "delta": round(d2, 4), "newcombe95": [round(lo2, 4), round(hi2, 4)],
        "verdict": ("NOT DETECTED" if lo2 <= 0 <= hi2 else "DETECTED at 95%"),
    }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("floor_foulplay.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "campaigns"}, indent=1))
    for tag in ("armS", "armW", "armWb"):
        c = out["campaigns"][tag]
        print(tag, c["our_wins"], "/", c["n"], c["win_rate"], c["wilson95"],
              "visits", c["fp_visits_per_decision_mean"], c["fp_visits_session_range"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
