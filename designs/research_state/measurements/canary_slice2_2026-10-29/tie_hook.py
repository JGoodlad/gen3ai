"""CPU: hooks argsort / argmax / max during one fp32 R1 forward on the 64 golden rows and lists, per call site, the rows
whose top two values tie EXACTLY (a tie-break a compiled kernel could resolve differently).

    PYTHONPATH=src python tie_hook.py <run_dir> dump|anchor <path> [shift_max]
"""
import sys, traceback
from pathlib import Path
import torch
torch.set_num_threads(2)
from main.ops import canary_dump_replay as R
from agents.model import compile_regions as cr
from agents.model.trunk_precision import fp32_scope
run, kind, path = sys.argv[1:4]
shift = float(sys.argv[4]) if len(sys.argv) > 4 else None
model, _ = R.build_model(Path(run), Path(path) if kind == "dump" else None, "cpu", 64,
                         anchor=Path(path) if kind == "anchor" else None, shift_max=shift)
calls = {}
def site():
    return " ".join(f"{f.filename.split('/')[-1]}:{f.lineno}" for f in traceback.extract_stack()[-4:-2])
def note(name, x, dim, top1):
    if not torch.is_floating_point(x) or x.dim() == 0 or x.shape[0] != 64:
        return
    x = x.detach().float()
    s = torch.sort(x, dim=dim, descending=True).values if not top1 else torch.topk(x, 2, dim=dim).values if x.shape[dim] > 1 else None
    if s is None:
        return
    a, b = s.narrow(dim, 0, 1), s.narrow(dim, 1, 1)
    tie = ((a == b) & torch.isfinite(a)).reshape(64, -1).any(-1)
    key = name + " @ " + site()
    rows = calls.setdefault(key, set())
    rows.update(int(i) for i in torch.nonzero(tie).flatten().tolist())
def wrap(obj, attr, name, top1):
    real = getattr(obj, attr)
    def f(x, *a, **k):
        dim = k.get("dim", a[0] if a and isinstance(a[0], int) else -1)
        if name in ("max", "amax") and not (a or "dim" in k):
            return real(x, *a, **k)
        try:
            note(name, x, dim, top1)
        except Exception as e:
            pass
        return real(x, *a, **k)
    setattr(obj, attr, f)
    return real
saved = [(torch.Tensor, "argsort", wrap(torch.Tensor, "argsort", "argsort", False)),
         (torch, "argsort", wrap(torch, "argsort", "argsort", False)),
         (torch.Tensor, "argmax", wrap(torch.Tensor, "argmax", "argmax", True)),
         (torch, "argmax", wrap(torch, "argmax", "argmax", True)),
         (torch.Tensor, "max", wrap(torch.Tensor, "max", "max", True))]
b = cr.r1_batch(model, 64, 0)
args = cr._r1_args(model, b)
from agents.training.instrumented_ppo.micro_step import micro_step
with fp32_scope(model.policy), torch.no_grad():
    micro_step(*args)
for o, a, r in saved:
    setattr(o, a, r)
print(Path(run).name)
for k, v in calls.items():
    print("  ", k, "rows with an exact top tie:", sorted(v)[:40])
