"""Pre-launch CPU probe: grow P* by trunk_ffn_by_kind off->on with the PIN's code, the trainer's own sequence
(load_model_snapshot(grow=...) -> _validate_or_reset_optimizer_state -> apply_growth); then compare P* and the grown
child under the bf16 trunk on CPU (autocast) on the same 64 rows, and at batch 1."""
import dataclasses, json, sys
import numpy as np, torch
from agents.model import growth as G
from agents.model.snapshot import load_model_snapshot
from agents.model.model_version import ModelVersion
from main.train.checkpoint_state import _validate_or_reset_optimizer_state
from agents.model.trunk_precision import set_trunk_precision

zp = sys.argv[1]; cfg = sys.argv[2]
torch.set_num_threads(2)
saved = ModelVersion.from_json_file(cfg)
print("saved trunk_ffn_by_kind:", getattr(saved, "trunk_ffn_by_kind", None), "trunk_layers", saved.trunk_layers)
fields = {"trunk_ffn_by_kind": {"parent": "off", "child": "on"}}
child_v = dataclasses.replace(saved, trunk_ffn_by_kind="on")
model = load_model_snapshot(zp, env=None, current_version=child_v, device="cpu", grow=fields)
n_opt_state = len(model.policy.optimizer.state)
_validate_or_reset_optimizer_state(model, zp)
print("optimizer state entries before growth:", n_opt_state, "after realign:", len(model.policy.optimizer.state))
parent_probe = G.cpu_probe(model)
rep = G.apply_growth(model, fields)
print("REPORT", json.dumps(rep, default=str))
child_probe = G.cpu_probe(model)
obs, mask = G.gate_rows(model.policy)
# bf16 trunk on CPU (autocast), both nets
out = {}
for name, pol in (("parent", parent_probe), ("child", child_probe)):
    set_trunk_precision(pol, "bf16")
    out[name] = G.forward_record(pol, obs, mask)
    set_trunk_precision(pol, "fp32")
    out[name + "_fp32"] = G.forward_record(pol, obs, mask)
fin = np.isfinite(out["parent"]["logits"])
d = lambda a, b: float(np.abs(a[fin] - b[fin]).max())
print("fp32 64-row child-parent max|dlogit|", d(out["child_fp32"]["logits"], out["parent_fp32"]["logits"]))
print("bf16 64-row child-parent max|dlogit|", d(out["child"]["logits"], out["parent"]["logits"]),
      "value", float(np.abs(out["child"]["values"] - out["parent"]["values"]).max()))
print("bf16-vs-fp32 shift (parent) max|dlogit|", d(out["parent"]["logits"], out["parent_fp32"]["logits"]))
# batch 1 rows, fp32 and bf16
mx32 = mxbf = 0.0
for i in range(8):
    o, m = obs[i:i+1], mask[i:i+1]
    for prec in ("fp32", "bf16"):
        set_trunk_precision(parent_probe, prec); set_trunk_precision(child_probe, prec)
        a = G.forward_record(parent_probe, o, m)["logits"]; b = G.forward_record(child_probe, o, m)["logits"]
        f = np.isfinite(a)
        v = float(np.abs(a[f] - b[f]).max())
        if prec == "fp32": mx32 = max(mx32, v)
        else: mxbf = max(mxbf, v)
print("batch-1 (8 rows) child-parent max|dlogit| fp32", mx32, "bf16", mxbf)
# optimizer state: are copies' moments equal to their sources'?
opt = model.policy.optimizer
tt = model.policy.features_extractor.team_transformer
from agents.model.trunk_kffn import trunk_rounds
r0 = trunk_rounds(tt)[0]
s_src = opt.state.get(r0.linear1.weight); s_dst = opt.state.get(r0.kind_ffn["board"].linear1.weight)
print("copied moments equal:", s_src is not None and s_dst is not None and torch.equal(s_src["exp_avg_sq"], s_dst["exp_avg_sq"]),
      "step", s_src.get("step") if s_src else None)
print("param_group lr", opt.param_groups[0]["lr"], {k: v for k, v in opt.param_groups[0].items() if k not in ("params",)})
