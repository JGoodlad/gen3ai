import dataclasses, sys, os
import numpy as np, torch
from agents.model import growth as G
from agents.model.snapshot import load_model_snapshot, save_model_snapshot
from agents.model.model_version import ModelVersion
from main.train.checkpoint_state import _validate_or_reset_optimizer_state
zp, cfg, out = sys.argv[1:4]
torch.set_num_threads(2)
saved = ModelVersion.from_json_file(cfg)
fields = {"trunk_ffn_by_kind": {"parent": "off", "child": "on"}}
cv = dataclasses.replace(saved, trunk_ffn_by_kind="on")
model = load_model_snapshot(zp, env=None, current_version=cv, device="cpu", grow=fields)
_validate_or_reset_optimizer_state(model, zp)
rep = G.apply_growth(model, fields)
os.makedirs(out, exist_ok=True)
model.save(os.path.join(out, "final_model"))
save_model_snapshot(out, dataclasses.replace(cv, growth={"fields": fields, **rep}), git_hash="x48review")
p = G.cpu_probe(model); obs, mask = G.gate_rows(model.policy)
np.save(os.path.join(out, "ref_logits.npy"), G.forward_record(p, obs, mask)["logits"])
print("saved", out)
