import os, sys, torch, numpy as np
from agents.inference.service.decision import DecisionModule, policy_reference
from agents.inference.service.parity import fixture_rows
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
p = load_foreign_opponent("/home/goodlad/dev/gen3ai/models/ai_v14_06_lbat_ctrl_fix/final_model.zip", current_version=current_model_version(load_mappings()), device="cpu")[0].policy.eval()
for m in p.modules():
    if hasattr(m, "_debugger"): m._debugger = None
dm = DecisionModule(p).eval()
persist = set(dm.state_dict().keys())
names = [n for n, _ in dm.named_parameters()] + [n for n, _ in dm.named_buffers() if n in persist]
d = dict(dm.named_parameters()); d.update(dict(dm.named_buffers()))
obs, mask = fixture_rows(p.observation_space["observation"].shape[0], 8)
tens = [torch.as_tensor(obs), torch.as_tensor(mask)] + [d[n].detach() for n in names]
D = "/tmp/m5t2/cpp/in"; os.makedirs(D, exist_ok=True)
with open(f"{D}/manifest.txt", "w") as fh:
    for k, t in enumerate(tens):
        t = t.contiguous(); code = {torch.float32: "f32", torch.bool: "b8", torch.int64: "i64"}[t.dtype]
        t.numpy().tofile(f"{D}/{k}.bin"); fh.write(f"{k}.bin {code} {t.dim()} {' '.join(map(str, t.shape))}\n")
lp, v = policy_reference(p.cuda(), tens[0].cuda(), tens[1].cuda())
np.save(f"{D}/ref_logp.npy", lp.cpu().numpy()); np.save(f"{D}/ref_v.npy", v.cpu().numpy())
print("dumped", len(tens))
