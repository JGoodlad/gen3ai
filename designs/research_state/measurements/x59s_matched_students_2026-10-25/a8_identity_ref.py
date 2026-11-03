"""Amendment 8 identity, REFERENCE side (main's tree): each follow-up student's zip loaded by main's strict loader,
its policy logits on the export's own probe rows (the first 256 of its rows' validation shard)."""
import json, sys
from pathlib import Path
import numpy as np
from main.human_data import rowset as RS
from main.human_data.student import THREADS
from main.policy_spectrum.reader import inference_globals, policy_logits
from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict

out = Path(sys.argv[1]); root = Path(sys.argv[2])
st = json.loads((root / "read/students/students.json").read_text())["students"]
for lab, s in sorted(st.items()):
    d = root / "read/students" / lab
    rows = Path((d / "rows_path.txt").read_text().strip())
    man, _ = RS.load_manifest(rows)
    prec = man["val"] if man["val"]["rows"] else man["train_shards"][0]
    probe = RS.load_shard(rows, prec, ("rows", "mask"))
    r, m = probe["rows"][:256], probe["mask"][:256].astype(np.uint8)
    with inference_globals(THREADS):
        mod = load_checkpoint_strict(s["zip_path"], device="cpu", **historical_load_kwargs(s["zip_path"]))
        mod.policy.eval()
        lg = policy_logits(mod, r, m)
    np.savez(out / f"{lab}.npz", rows=r, mask=m, ref=lg)
    print(lab, lg.shape, flush=True)
