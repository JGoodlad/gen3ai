import os, torch, numpy as np, traceback
from agents.inference.service.decision import DecisionModule
from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
model, _ = load_foreign_opponent("/home/goodlad/dev/gen3ai/models/ai_v14_06_lbat_ctrl_fix/final_model.zip", current_version=current_model_version(load_mappings()), device="cpu")
pol = model.policy.eval().to("cuda")
for m in pol.modules():
    if hasattr(m, "_debugger"): m._debugger = None
obs, mask = load_parity_rows(pol.observation_space["observation"].shape[0])
o = torch.as_tensor(obs[:8]).cuda(); m = torch.as_tensor(mask[:8]).cuda()
dm = DecisionModule(pol).eval()
with torch.no_grad(): dm(o, m)
torch.cuda.synchronize()
import warnings, collections
sites=collections.Counter()
def hook(message, category, filename, lineno, file=None, line=None):
    st=[l for l in traceback.format_stack()[:-1] if "/src/agents/model" in l or "/src/utils" in l]
    sites[st[-1].strip().split("\n")[0] if st else "?"]+=1
warnings.showwarning=hook
torch.cuda.set_sync_debug_mode("warn")
with torch.no_grad(): dm(o, m)
torch.cuda.set_sync_debug_mode(0)
for k,v in sites.items(): print(v, k)
