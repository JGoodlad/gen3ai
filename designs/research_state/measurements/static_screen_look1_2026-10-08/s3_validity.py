"""Static-token screen look 1: VERIFY seed S3's amendment-2 validity from its FULL child log (read-only).

``design_static_tokens.md`` Decision record 2026-10-08, screen amendment 2: a seed that finishes under
``--behaviour-check warn`` is NOT INCONCLUSIVE provided EVERY K9(b) probe after the switch keeps max abs d log pi
< 1e-4. The plain ``launcher_child.log`` is a ring buffer, so this reads ``launcher_child.full.log``.

    python s3_validity.py [--run models/rb_st_static_s1003] [--out s3_validity.json]

It splits the log at its ``===== child attached`` markers, takes the LAST attach (the warn switch) and every one
after it, and parses each per-update ``behaviour/`` table block. The verdict is on ``max_abs_dlogp_current`` (every
current row probed, the amendment's "max abs d log pi"); the judged / excluded maxima are reported beside it. A probe
is a ``behaviour/`` block whose ``rows_probed`` > 0. Exit 0 = VALID, 1 = INVALID (a probe at or above the bar), 3 =
the log could not support the check (no switch attach, no probe after it)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

BAR = 1e-4
ATTACH = re.compile(r"^===== child attached (\S+ \S+) \(pid (\d+)\) =====")
ROW = re.compile(r"^\|\s+(\w+)\s+\|\s+(\S+)\s+\|")
FIELDS = ("max_abs_dlogp_current", "max_abs_dlogp_judged", "max_abs_dlogp_excluded", "excluded_frac",
          "rows_probed", "rows_excluded", "violations_total_max")
STEP = re.compile(r"^\|\s+total_timesteps\s+\|\s+(\S+)\s+\|")


def parse(lines: List[str]) -> List[Dict[str, Any]]:
    """Every ``behaviour/`` block, with the most recent ``total_timesteps`` seen before it."""
    out: List[Dict[str, Any]] = []
    cur = None
    step = None
    for ln in lines:
        m = STEP.match(ln)
        if m:
            step = float(m.group(1))
        if ln.startswith("| behaviour/"):
            cur = {"line_step": step}
            continue
        if cur is not None:
            r = ROW.match(ln)
            if r and ln.startswith("|    "):
                if r.group(1) in FIELDS:
                    cur[r.group(1)] = float(r.group(2))
                continue
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/home/goodlad/dev/gen3ai/models/rb_st_static_s1003")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "s3_validity.json"))
    a = ap.parse_args()
    log = Path(a.run) / "launcher_child.full.log"
    lines = log.read_text(errors="replace").splitlines()
    attaches = [(i, m.group(1), int(m.group(2))) for i, ln in enumerate(lines) if (m := ATTACH.match(ln))]
    argv_warn = [i for i, ln in enumerate(lines) if "behaviour-check" in ln and "warn" in ln]
    if not attaches:
        print("no child attach marker", file=sys.stderr)
        return 3
    i0, t0, pid0 = attaches[-1]
    segs = []
    for k, (i, t, pid) in enumerate(attaches):
        end = attaches[k + 1][0] if k + 1 < len(attaches) else len(lines)
        blocks = [b for b in parse(lines[i:end]) if b.get("rows_probed", 0) > 0]
        segs.append({"attached": t, "pid": pid, "line": i + 1, "probes": len(blocks),
                     "max_abs_dlogp_current": max((b.get("max_abs_dlogp_current", 0.0) for b in blocks), default=None),
                     "max_abs_dlogp_judged": max((b.get("max_abs_dlogp_judged", 0.0) for b in blocks), default=None),
                     "max_excluded_frac": max((b.get("excluded_frac", 0.0) for b in blocks), default=None)})
    post = [b for b in parse(lines[i0:]) if b.get("rows_probed", 0) > 0]
    res: Dict[str, Any] = {"run": str(a.run), "log": str(log), "log_lines": len(lines), "bar": BAR,
                           "switch_attach": {"time": t0, "pid": pid0, "line": i0 + 1},
                           "warn_mentions_lines": [i + 1 for i in argv_warn][:20],
                           "segments": segs, "post_switch_probes": len(post)}
    if not post:
        res["verdict"] = "CANNOT CHECK (no probe after the switch)"
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        return 3
    for f in ("max_abs_dlogp_current", "max_abs_dlogp_judged", "max_abs_dlogp_excluded", "excluded_frac",
              "violations_total_max"):
        vals = [b[f] for b in post if f in b]
        res[f"post_{f}_max"] = max(vals) if vals else None
        res[f"post_{f}_n"] = len(vals)
    over = [b for b in post if b.get("max_abs_dlogp_current", 0.0) >= BAR]
    res["post_probes_at_or_over_bar"] = over
    res["first_probe_step"] = post[0].get("line_step")
    res["last_probe_step"] = post[-1].get("line_step")
    res["verdict"] = "VALID" if not over else "INVALID"
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "segments"}, indent=1))
    for s in segs:
        print(s)
    return 0 if not over else 1


if __name__ == "__main__":
    raise SystemExit(main())
