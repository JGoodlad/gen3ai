"""M9b — bank the dual-scored decision rows for commit.

The recorder writes three 11-float probability vectors per row (one per arm). NO AXIS READS
THEM — they exist only for the KL / |ΔV| divergence diagnostics, which are computed once and
land in the JSON. They are also ~⅔ of the bytes. So the committed row set drops ``p`` and keeps
everything an axis or a future axis could need: the per-action class vector, the board strata,
and all three arms' argmax and value on the identical board.

That is M4's convention (`v8_fold_behavioral_fingerprint_2026-08-31_rows_*.jsonl.gz`) and it is
what makes a NEW axis definition testable offline without replaying a single battle.

Run:
  python v8_gift_decay_fingerprint_bank.py --in '/tmp/m9b/rows_untaught_*.jsonl.gz' \
      --out designs/research_state/measurements/v8_gift_decay_fingerprint_2026-09-01_rows_untaught.jsonl.gz
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys

DROP = ("p",)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    n = 0
    with gzip.open(a.out, "wt") as out:
        for pat in a.inp:
            for path in sorted(glob.glob(pat)):
                with gzip.open(path, "rt") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            d = json.loads(ln)
                        except json.JSONDecodeError:
                            continue  # a half-written final line
                        for k in DROP:
                            d.pop(k, None)
                        out.write(json.dumps(d, separators=(",", ":")) + "\n")
                        n += 1
    print(f"[bank] {n} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
