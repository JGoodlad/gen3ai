"""Static-token screen look 3: the DEVIATION check of all SIXTEEN seeds, read-only, from the runs' own records.

Look 2's per-run check (``../static_screen_look2_2026-10-09/deviations.py``'s ``check``: TB files, attach markers,
crashes, the launcher log's Restarts summary / crash / auto-restart lines, the chain's lines, every K9(b) probe in the
full child log) over all sixteen runs, plus, for look 3, every K9(b) STOP classified against the amendments:

* AMENDMENT 1 (tie-only stop, resumed): a ``behaviour_violations.jsonl`` row whose ONLY failed condition is the
  ``excluded_frac`` ceiling (0.15) and whose max abs d log pi is < 1e-4 — read both as the row's ``max`` condition
  (the judged rows) and as the max ``abs_dlogp`` over every dumped row (judged + excluded, "max abs d log pi" in the
  amendment's words). Any other failed condition, or a max ≥ 1e-4, is NOT an amendment-1 stop.
* AMENDMENT 2 (warn switch): S3 only; look 1's ``s3_validity.py`` verdict (every post-switch probe < 1e-4).

The brief's expectation (each compared, a mismatch is a FINDING): S1 / S2 one amendment-1 resume each; S3 the
amendment-2 switch; S6 one amendment-1 tie-only resume (06:26) + the chain swap (``chain_look3b.sh`` ADOPTED S6's
launcher at 07:00 for the disk gate — the launcher process is unchanged, so no new child); every other seed none.

    python deviations.py [--models M] [--jobs ~/.claude/jobs/st_screen] [--out deviations.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "static_screen_look2_2026-10-09"))
sys.path.insert(1, str(HERE.parent / "static_screen_look1_2026-10-08"))
import deviations as D2  # noqa: E402  (look 2's per-run check)

SEEDS = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008)
BAR = 1e-4
EXPECTED = {"rb_st_static_s1001": "amendment 1 x1", "rb_st_static_s1002": "amendment 1 x1",
            "rb_st_static_s1003": "amendment 2 (warn switch)", "rb_st_static_s1006": "amendment 1 x1 + chain adopt"}


def stops(rd: Path) -> List[Dict[str, Any]]:
    f = rd / "behaviour_violations.jsonl"
    out: List[Dict[str, Any]] = []
    if not f.exists():
        return out
    for ln in f.read_text().splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        conds = d.get("conditions") or []
        failed = [c["statistic"] for c in conds if not c.get("ok")]
        cmax = next((c["value"] for c in conds if c["statistic"] == "max"), None)
        rmax = max((r.get("abs_dlogp", 0.0) for r in d.get("rows") or []), default=None)
        efrac = next((c["value"] for c in conds if c["statistic"] == "excluded_frac"), None)
        a1 = (failed == ["excluded_frac"] and cmax is not None and cmax < BAR and (rmax is None or rmax < BAR))
        out.append({"num_timesteps": d.get("num_timesteps"), "n_updates": d.get("n_updates"), "failed": failed,
                    "max_judged": cmax, "max_dumped_rows": rmax, "excluded_frac": efrac,
                    "rows_current": d.get("rows_current"), "rows_excluded": d.get("rows_excluded"),
                    "amendment_1_tie_only": a1})
    return out


def launcher_events(jobs: Path, run: str) -> List[str]:
    ll = jobs / f"{run}.launcher.log"
    if not ll.exists():
        return []
    return [x.strip() for x in ll.read_text(errors="replace").splitlines()
            if re.search(r"Child crashed|Auto-restart|Training complete|Restarts\s*:", x)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(D2.ARCHIVE_MODELS))
    ap.add_argument("--jobs", default=str(D2.JOBS))
    ap.add_argument("--out", default=str(HERE / "deviations.json"))
    a = ap.parse_args()
    runs = []
    for arm in ("legacy", "static"):
        for s in SEEDS:
            run = f"rb_st_{arm}_s{s}"
            r = D2.check(Path(a.models), Path(a.jobs), run)
            r["k9b_stops"] = stops(Path(a.models) / run)
            r["launcher_events"] = launcher_events(Path(a.jobs), run)
            n_stop = len(r["k9b_stops"])
            all_a1 = all(x["amendment_1_tie_only"] for x in r["k9b_stops"])
            if r["clean"]:
                got = "none"
            elif run == "rb_st_static_s1003":
                got = "amendment 2 (warn switch)"
            elif n_stop and all_a1 and r["launcher_crash_lines"] == n_stop:
                got = f"amendment 1 x{n_stop}" + (" + chain adopt" if any("ADOPT" in c for c in r["chain"]) else "")
            else:
                got = "UNCLASSIFIED"
            r["deviation_found"] = got
            r["deviation_expected"] = EXPECTED.get(run, "none")
            r["matches_brief"] = got == r["deviation_expected"]
            runs.append(r)
    out = {"runs": runs, "all_match_brief": all(r["matches_brief"] for r in runs)}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for r in runs:
        print(f"{r['run']}: found={r['deviation_found']!r} expected={r['deviation_expected']!r} "
              f"match={r['matches_brief']} children(tb/attach)={r['tb_files']}/{r['child_attach_markers']} "
              f"restarts={r['launcher_restarts_summary']} crashes={r['launcher_crash_lines']} "
              f"stops={[(x['n_updates'], x['failed'], x['max_judged'], x['max_dumped_rows'], x['excluded_frac']) for x in r['k9b_stops']]} "
              f"warn={r['warn_in_metadata']} pins={[p[:8] for p in r['pin_history']]} probes={r['k9b_probes']} "
              f"max|dlogp| judged/cur={r['k9b_max_abs_dlogp_judged']}/{r['k9b_max_abs_dlogp_current']} "
              f"max_excl={r['k9b_max_excluded_frac']} final={r['final_model']}")
    print("all match the brief:", out["all_match_brief"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
