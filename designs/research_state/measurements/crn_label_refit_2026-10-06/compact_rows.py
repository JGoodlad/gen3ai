"""Commit form of the rows: the two truth files compacted (``truth.write_compact``), the capture index
gzipped. The full rows, the capture's feature ``.npz`` files and the root read's probabilities live in
``~/gen3ai_archive/crn_label_refit_2026-10-06/rows/`` (copy them back into ``rows/`` to re-run
``refit.py``; ``capture.py`` regenerates the features exactly).

    python compact_rows.py
"""

from __future__ import annotations

import gzip
from pathlib import Path

from main.policy_spectrum.truth import write_compact

R = Path(__file__).resolve().parent / "rows"
for name in ("truth_held_S32", "truth_train_S8"):
    n = write_compact(R / f"{name}.jsonl", R / f"{name}.compact.jsonl.gz")
    print(f"{name}: {n} rows")
with open(R / "cap" / "index.jsonl", "rb") as f, gzip.GzipFile(R / "cap_index.jsonl.gz", "wb", mtime=0) as g:
    g.write(f.read())
print("cap index gzipped")
