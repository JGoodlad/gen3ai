"""CPU: the per-row fp32 dither null (canary_row_null.compute, cap lifted) on one canary state; prints the rows that move most.

    PYTHONPATH=src python row_null.py <run_dir> dump|anchor <path> <out.json> [shift_max]
"""
import json, sys
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(2)
from main.ops import canary_dump_replay as R
from agents.model import canary_row_null as N
run, kind, path, out = sys.argv[1:5]
shift = float(sys.argv[5]) if len(sys.argv) > 5 else None
model, b = R.build_model(Path(run), Path(path) if kind == "dump" else None, "cpu", 64,
                         anchor=Path(path) if kind == "anchor" else None, shift_max=shift)
raw = N.compute(model, cap_s=1e9)
rec = N.record(raw)
null = np.array([float(x) for x in rec["null_per_row"]])
order = np.argsort(-null)
res = {"null_per_row": null.tolist(), "top": [(int(i), float(null[i])) for i in order[:12]],
       "per_seed_rows_over_0.05": [[int(i) for i in np.where(np.array([float(x) for x in s]) > 0.05)[0]] for s in rec["per_seed"]]}
Path(out).write_text(json.dumps(res, indent=1))
print(run.split('/')[-1], "top rows (fixture row, max |dlogp| over 16 dithers):", [(i, round(v, 3)) for i, v in res["top"]])
print("  first-half rows > 0.05:", [i for i in range(32) if null[i] > 0.05], " second-half:", [i for i in range(32, 64) if null[i] > 0.05])
