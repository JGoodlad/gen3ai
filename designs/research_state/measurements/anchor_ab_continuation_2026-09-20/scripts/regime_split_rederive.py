#!/usr/bin/env python3
"""RETROACTIVE: re-derive this campaign's 84 sub-cells under the SPLIT regime flags.

WHY. The read of 2026-09-20 recorded `composite_regime_verified` FALSE on a third of its sub-cells
(hazard W-J, `ffc3b0ef`), and the entry could only say that the flag was a composite and that every
per-decision `argmax_match_rate` it could see was 1.0000. `gen3_anchor_regime_split_v1` splits that
composite into the two facts it was hiding:

* **`regime_verified_decisions`** — did the regime take effect on every decision, in BOTH halves,
  over at least one decision? (greedy ⇒ `argmax_match_rate == 1.0`; t1 ⇒ materially below 1.0.)
  This is what the SOP's verification reads.
* **`peer_clean`** — did every peer process exit 0?

This script applies exactly those definitions to the saved evidence, so the 8,400-game read's flag
is explained rather than argued about. It RE-DERIVES NOTHING ELSE: no win rate, no CI, no verdict
is recomputed here, and none changes — the games were always played and always recorded.

Evidence used, per sub-cell, all of it already committed under `../out/<cell>/`:
  * `<half>/peer_*_report.json` — the peer driver's own `sample_kwarg_values`,
    `argmax_match_rate`, `n_decisions`, `error`, `regime_check_ok`;
  * `summary.json` — `provenance.peer_report.peer_rc` and the recorded composite;
  * `games.jsonl.gz` — the rows, read only to confirm the cell's n and the per-row
    `their_argmax_match_rate` agree with the half reports (a cross-check, not a second source).

    python regime_split_rederive.py [<out-dir>] [<dest.json>]
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

HALVES = ("ours_challenge", "peer_challenge")


def _half_report(cell_dir: Path, half: str):
    d = cell_dir / half
    if not d.is_dir():
        return None
    for path in sorted(d.glob("peer_*_report.json")):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None
    return None


def _decisions_ok(blob, regime: str) -> bool:
    """The SPLIT definition, applied to a saved report — the peer script's own rule, verbatim.

    REGIME-APPROPRIATE, and no longer VACUOUS: a missing rate or zero decisions FAILS, because a
    check that never ran reads exactly like one that passed.
    """
    if not isinstance(blob, dict):
        return False
    rate = blob.get("argmax_match_rate")
    n_dec = blob.get("n_decisions") or 0
    kwargs = blob.get("sample_kwarg_values") or blob.get("sample_kwargs")
    ok_kw = kwargs == [regime != "greedy"]
    ok_rate = (rate is not None and n_dec >= 1
               and (rate == 1.0 if regime == "greedy" else rate < 1.0))
    return bool(ok_kw and ok_rate)


def _rows(cell_dir: Path):
    path = cell_dir / "games.jsonl.gz"
    if not path.exists():
        return []
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "out"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else out_dir.parent / "regime_split_rederived.json"

    cells, flips, still_false = [], [], []
    for cell_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        summary_path = cell_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        pr = summary.get("provenance", {}).get("peer_report", {}) or {}
        regime = summary["cell"]["their_regime"]
        rows = _rows(cell_dir)

        per_half, rates = {}, []
        for half in HALVES:
            blob = _half_report(cell_dir, half)
            ok = _decisions_ok(blob, regime)
            per_half[half] = {
                "present": blob is not None,
                "argmax_match_rate": (blob or {}).get("argmax_match_rate"),
                "n_decisions": (blob or {}).get("n_decisions"),
                "peer_error": (blob or {}).get("error"),
                "regime_verified_decisions": ok,
            }
            if (blob or {}).get("argmax_match_rate") is not None:
                rates.append(blob["argmax_match_rate"])

        decisions = all(per_half[h]["regime_verified_decisions"] for h in HALVES)
        errors = [per_half[h]["peer_error"] for h in HALVES if per_half[h]["peer_error"]]
        rc = pr.get("peer_rc")
        peer_clean = (rc == 0 or rc is None) and not errors
        old = bool(pr.get("regime_verified"))

        rec = {
            "cell": cell_dir.name,
            "status": summary.get("status"),
            "n": summary.get("n"),
            "rows_in_games_jsonl": len(rows),
            "their_regime": regime,
            "peer_rc": rc,
            "old_composite_regime_verified": old,
            "regime_verified_decisions": decisions,
            "peer_clean": bool(peer_clean),
            "new_composite_would_be": bool(decisions and peer_clean),
            "per_decision_rates": sorted(set(rates)),
            "row_rates": sorted({r.get("their_argmax_match_rate") for r in rows
                                 if r.get("their_argmax_match_rate") is not None}),
            "halves": per_half,
            "peer_errors": errors,
        }
        # the cross-check: the rows' own recorded rate must agree with the half reports
        rec["rows_agree_with_half_reports"] = (set(rec["row_rates"]) <= set(rec["per_decision_rates"])
                                               or not rec["row_rates"])
        cells.append(rec)
        if not old and decisions:
            flips.append(rec["cell"])
        elif not old and not decisions:
            still_false.append(rec["cell"])

    out = {
        "what": ("gen3_anchor_regime_split_v1 applied retroactively to the 2026-09-20 A/B "
                 "continuation campaign. Nothing but the two verification flags is re-derived; no "
                 "win rate, CI or verdict changes."),
        "definitions": {
            "regime_verified_decisions": ("both halves' sample kwarg matches the regime AND the "
                                          "per-decision argmax rate is regime-appropriate "
                                          "(greedy: == 1.0; t1: < 1.0) over >= 1 decision"),
            "peer_clean": "every peer process exited 0 and raised nothing",
            "regime_verified": "DEPRECATED — the AND of the two above",
        },
        "n_sub_cells": len(cells),
        "n_old_composite_false": sum(1 for c in cells if not c["old_composite_regime_verified"]),
        "n_flips_to_verified_decisions": len(flips),
        "n_still_unverified": len(still_false),
        "flipped": sorted(flips),
        "still_unverified": sorted(still_false),
        "cells": cells,
    }
    dest.write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("n_sub_cells", "n_old_composite_false",
                       "n_flips_to_verified_decisions", "n_still_unverified")}, indent=1))
    bad = [c["cell"] for c in cells if not c["rows_agree_with_half_reports"]]
    print(f"rows-vs-half-report disagreements: {len(bad)} {bad}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
