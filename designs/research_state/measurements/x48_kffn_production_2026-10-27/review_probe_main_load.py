import sys, os
import numpy as np, torch
from agents.model import growth as G
from agents.model.snapshot import load_model_snapshot
from agents.model.model_version import ModelVersion
out = sys.argv[1]
torch.set_num_threads(2)
saved = ModelVersion.from_json_file(os.path.join(out, "model_config.json"))
print("main reads config: kffn", getattr(saved, "trunk_ffn_by_kind", None), "ffn_dim", getattr(saved, "trunk_ffn_dim", None))
m = load_model_snapshot(os.path.join(out, "final_model.zip"), env=None, current_version=saved, device="cpu")
p = G.cpu_probe(m); obs, mask = G.gate_rows(m.policy)
got = G.forward_record(p, obs, mask)["logits"]; ref = np.load(os.path.join(out, "ref_logits.npy"))
print("main-loaded kffn child vs pin forward: equal", bool(np.array_equal(got, ref)))
