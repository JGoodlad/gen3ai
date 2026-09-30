"""THE DEPTH-3 SUCCESSOR SLICE as M5's gate reads it (program §2 M5 "Gate": "a depth-3 successor slice
(search's default depth) equals ``search_driver``'s rows") — Lane I's lockstep harness
(``utils.rust_env.successors_parity``), delegated to, never re-implemented.

THE READING (written down in ``m5_laneJ/PROGRESS.md``, because the one-line definition leaves three
things open):

* "search's default depth" = ``SearchConfig.max_depth`` (3) — asserted here, so a change of the
  default re-opens this reading instead of silently gating a different depth;
* "equals ``search_driver``'s rows" = Lane I's gate ①: the in-process tree and the ``search_driver``
  BINARY in lockstep, every root field and every arm field + ROW BYTES at plies 1–3, no allowlist,
  every depth non-vacuous (rows at 1, 2 and 3; D10 leaves among them);
* a batch BOTH roads refuse with the SAME error is an equality, not a difference — but it is not
  coverage either. Those are finding F-LI-1 (a follow-up choice fed into the next turn, in BOTH
  roads). The slice reports them as ``refused_both`` beside the verdict: PASS means "equal wherever
  either road serves an arm"; it does NOT mean depth 3 is correct past a replacement switch that
  ends the turn.

MILESTONE = Lane I's registered milestone sources (ladder milestone 12, pool 6, procedural 6
battles, 3 turns each, seed 17); COMMIT = its routine corpus (ladder commit 3 + pool 1, 2 turns).
"""
from __future__ import annotations

import time
from typing import List, Tuple

#: (source, battles, corpus seed, turns per battle) — Lane I's registered tiers.
TIERS = {
    "commit": ((("ladder_commit", 3, 5), ("pool", 1, 6)), 5, 2),
    "milestone": ((("ladder_milestone", 12, 17), ("pool", 6, 17), ("procedural", 6, 17)), 17, 3),
}


def default_depth() -> int:
    from main.search_dividend.search import SearchConfig

    return int(SearchConfig().max_depth)


def run_tier(tier: str, lib=None) -> dict:
    from utils.rust_env import successors as S
    from utils.rust_env import successors_parity as SP

    if default_depth() != 3:
        raise AssertionError(f"search's default depth is {default_depth()}, not 3 — the depth-3 slice no longer "
                             "gates the default; re-open the reading (m5_laneJ/PROGRESS.md) before trusting it")
    if lib is None:
        from main.rust_core_m5.slice_n import build

        lib, _ = build("selfcheck")
    sources, run_seed, n_turns = TIERS[tier]
    parts: List[dict] = []
    for source, n, seed in sources:
        t0 = time.time()
        core = S.SearchCore(lib=lib)
        try:
            logs = SP.make_logs(SP._teams(source), n, seed, core=core)
        finally:
            core.close()
        cen = SP.run(logs, seed=run_seed, lib=lib, n_turns=n_turns)
        depths = {int(d): int(c) for d, c in sorted(cen.by_depth.items())}
        nonvacuous = all(depths.get(d, 0) > 0 for d in (1, 2, 3)) and cen.rows > 0 and cen.mid > 0
        parts.append({"source": source, "battles": cen.battles, "roots": cen.roots, "arms": cen.arms,
                      "rows": cen.rows, "d10_leaves": cen.mid, "terminal_arms": cen.ended, "by_depth": depths,
                      "differences": cen.diffs[:20], "n_differences": len(cen.diffs),
                      "refused_both": len(cen.refused), "refused_examples": cen.refused[:3],
                      "nonvacuous": nonvacuous, "ok": not cen.diffs and nonvacuous,
                      "seconds": round(time.time() - t0, 1)})
    return {"component": "depth3", "tier": tier, "default_depth": 3, "parts": parts,
            "ok": all(p["ok"] for p in parts), "refused_both": sum(p["refused_both"] for p in parts)}


def render(res: dict) -> str:
    lines = [f"DEPTH-3 SUCCESSOR SLICE, tier {res['tier']}: {'PASS' if res['ok'] else 'FAIL'}"
             + (f" — {res['refused_both']} batches refused by BOTH roads (F-LI-1, open)" if res["refused_both"] else "")]
    for p in res["parts"]:
        lines.append(f"  {p['source']:<17} {p['battles']} battles: {p['arms']} arms, {p['rows']} rows byte-equal"
                     f" ({p['n_differences']} differences), D10 leaves {p['d10_leaves']}, by depth {p['by_depth']}, "
                     f"refused-both {p['refused_both']} ({p['seconds']} s)")
    return "\n".join(lines)


def summary(res: dict) -> Tuple[bool, str]:
    return res["ok"], f"{sum(p['rows'] for p in res['parts'])} rows, refused-both {res['refused_both']}"
