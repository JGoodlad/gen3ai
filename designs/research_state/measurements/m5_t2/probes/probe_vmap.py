import os, sys, time, traceback, torch
from torch.func import functional_call, stack_module_state, vmap
from agents.inference.service.fixtures import perturbed_fresh_policy
from agents.inference.service.decision import DecisionModule, policy_reference
from agents.inference.service.parity import fixture_rows
S = int(os.environ.get("S", "2")); B = int(os.environ.get("B", "4"))
dev = os.environ.get("DEV", "cpu")
pols = [perturbed_fresh_policy(s).to(dev) for s in range(S)]
mods = [DecisionModule(p).eval() for p in pols]
params, bufs = stack_module_state(mods)
base = mods[0]
D = pols[0].observation_space["observation"].shape[0]
obs, mask = fixture_rows(D, S * B)
o = torch.as_tensor(obs, device=dev).view(S, B, D); m = torch.as_tensor(mask, device=dev).view(S, B, -1)
def f(p, b, o, m):
    return functional_call(base, (p, b), (o, m))
try:
    with torch.no_grad():
        out = vmap(f, in_dims=(0, 0, 0, 0))(params, bufs, o, m)
    print("VMAP OK")
    for s in range(S):
        r = policy_reference(pols[s], o[s], m[s]); fin = torch.isfinite(r[0])
        print(s, "dlogp", float((out[0][s]-r[0])[fin].abs().max()), "dV", float((out[1][s]-r[1]).abs().max()))
except Exception as e:
    tb = [l for l in traceback.format_exc().splitlines() if "/src/agents" in l]
    print("VMAP FAIL", type(e).__name__, str(e)[:600]); print("\n".join(tb[-4:]))
