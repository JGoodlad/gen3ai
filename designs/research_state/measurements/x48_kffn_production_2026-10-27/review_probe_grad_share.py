"""Per-kind gradient-energy share of each round's FFN on the gate rows (fp32 CPU, a proxy loss), and the inherited
AdamW second moment's scale vs eps: how strongly the COPIED exp_avg_sq throttles each expert's first steps."""
import dataclasses, sys
import numpy as np, torch
from agents.model import growth as G
from agents.model.snapshot import load_model_snapshot
from agents.model.model_version import ModelVersion
from main.train.checkpoint_state import _validate_or_reset_optimizer_state
from agents.model.trunk_kffn import trunk_rounds, EXPERT_KINDS
zp, cfg = sys.argv[1], sys.argv[2]
torch.set_num_threads(2)
saved = ModelVersion.from_json_file(cfg)
fields = {"trunk_ffn_by_kind": {"parent": "off", "child": "on"}}
model = load_model_snapshot(zp, env=None, current_version=dataclasses.replace(saved, trunk_ffn_by_kind="on"),
                            device="cpu", grow=fields)
_validate_or_reset_optimizer_state(model, zp)
G.apply_growth(model, fields)
pol = model.policy
pol.set_training_mode(False)
from agents.model.trunk_precision import set_trunk_precision
set_trunk_precision(pol, "fp32")
obs, mask = G.gate_rows(pol)
o = {"observation": torch.as_tensor(obs), "action_mask": torch.as_tensor(mask.astype(np.int8))}
dist = pol.get_distribution(o, action_masks=mask).distribution
logits = dist.logits
probs = torch.softmax(logits.masked_fill(~torch.as_tensor(mask.astype(bool)), -1e9), -1)
# proxy policy loss: entropy-weighted log-prob of a SAMPLED action (seeded), plus the value
g = torch.Generator().manual_seed(0)
a = torch.multinomial(probs.detach(), 1, generator=g).squeeze(1)
adv = torch.randn(len(a), generator=g)
loss_pi = -(adv * torch.log(probs[torch.arange(len(a)), a] + 1e-12)).mean()
v = pol.predict_values(o).squeeze(-1)
loss_v = ((v - torch.randn(len(v), generator=g)) ** 2).mean()
opt = pol.optimizer
for name, loss in (("policy", loss_pi), ("value", loss_v)):
    pol.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    tt = pol.features_extractor.team_transformer
    for i, r in enumerate(trunk_rounds(tt)):
        mods = {"mon": r, **{k: r.kind_ffn[k] for k in EXPERT_KINDS}}
        e = {k: sum(float((getattr(m, l).weight.grad ** 2).sum()) for l in ("linear1", "linear2")) for k, m in mods.items()}
        # the SHARED gradient = the sum of the per-kind gradients (the experts are copies)
        tot = sum(float(((sum(getattr(m, l).weight.grad for m in mods.values())) ** 2).sum()) for l in ("linear1", "linear2"))
        print(name, "round", i, "shared energy %.3g" % tot, {k: round(e[k] / tot, 4) for k in e})
r = trunk_rounds(pol.features_extractor.team_transformer)[0]
st = opt.state.get(r.linear1.weight)
if st:
    s = st["exp_avg_sq"].sqrt()
    print("P* round0 linear1.weight sqrt(v): median %.3g p10 %.3g p90 %.3g; eps 1e-5; step %s" % (
        float(s.median()), float(s.quantile(0.1)), float(s.quantile(0.9)), st.get("step")))
