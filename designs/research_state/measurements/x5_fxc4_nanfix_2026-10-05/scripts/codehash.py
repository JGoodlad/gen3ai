"""The BLOB-IDENTITY proof (gen3_fm_index_max_v1): hash what torch.compile generates for an arm's R1 (the
learner micro-step, forward + backward) and T2 (the inference service's `decide`, the `graph` backend's
callable) — the dynamo FX graph code and every Inductor output module — with the FX / AOT caches OFF so
every artifact is generated in this process. Run once at HEAD and once at the fix: blob's hashes must
be EQUAL. Usage: codehash.py <arm> <out.json> [<dump_dir>]"""
import hashlib
import json
import os
import sys

import torch as th
import torch._functorch.config as FC
import torch._inductor.config as IC
from torch._inductor.codecache import PyCodeCache
from torch._inductor.compile_fx import compile_fx

from agents.model import compile_regions as cr
from agents.model.compile_control import apply_compile_config
from agents.training import learner_golden as LG
from agents.training.instrumented_ppo.micro_step import micro_step

arm, out = sys.argv[1:3]
dump = sys.argv[3] if len(sys.argv) > 3 else None
B = 2048
th.set_float32_matmul_precision("highest")
m = LG.build_learner() if arm == "blob" else LG.build_arm_learner(arm)
m.policy.to("cuda")
m.device = th.device("cuda")
m.batch_size = B
from agents.model.compile_cache import ensure_hermetic_cache  # noqa: E402
ensure_hermetic_cache("x5 nanfix codehash")
apply_compile_config()
IC.fx_graph_cache = False
FC.enable_autograd_cache = False
DYN: dict = {"R1": [], "T2": []}
REGION = ["R1"]


def recording(gm, example_inputs):
    DYN[REGION[0]].append(gm.code)
    return compile_fx(gm, example_inputs)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def modules_now():
    return {getattr(mod, "__file__", "") for mod in PyCodeCache.modules}


row = {"arm": arm, "torch": th.__version__}
# the T2 replica is copied BEFORE any forward: a forward leaves graph tensors in the stashes, which
# `served_replica`'s deepcopy refuses
from agents.inference.service.engine import decide  # noqa: E402
from agents.inference.service.parity import fixture_rows  # noqa: E402
from agents.inference.service.slots import served_replica  # noqa: E402
rep = served_replica(m.policy)
rep.optimizer = None
from agents.inference.service.decision import DecisionModule  # noqa: E402
rep = DecisionModule(rep.to("cuda").eval()).eval()          # what a slot serves (`SlotGroup.modules`)
# ---- R1: forward + backward of the learner micro-step on the gate's rows
before = modules_now()
b = cr.r1_batch(m, B)
m.policy.set_training_mode(True)
args = cr._r1_args(m, b)
comp = th.compile(micro_step, backend=recording, fullgraph=True, dynamic=False)
c = cr._r1_arm(m, comp, args)
th.cuda.synchronize()
r1_files = sorted(modules_now() - before)
row["R1_grad_finite"] = bool(th.isfinite(c["grad"]).all())
# ---- T2: the served replica's decide() at one bucket (the `graph` backend's compiled callable)
REGION[0] = "T2"
before = modules_now()
D = int(m.policy.observation_space["observation"].shape[0])
o, mk = fixture_rows(D, 64)
t2 = th.compile(decide, backend=recording, dynamic=False)
with th.no_grad():
    t2(rep, th.as_tensor(o).cuda(), th.as_tensor(mk).cuda())
th.cuda.synchronize()
t2_files = sorted(modules_now() - before)
for region, files in (("R1", r1_files), ("T2", t2_files)):
    srcs = [open(f).read() for f in files if f]
    row[f"{region}_dynamo_graphs"] = len(DYN[region])
    row[f"{region}_dynamo_code_sha"] = [sha(x) for x in DYN[region]]
    row[f"{region}_inductor_modules"] = len(srcs)
    row[f"{region}_inductor_module_sha"] = sorted(sha(x) for x in srcs)
    row[f"{region}_inductor_bytes"] = sum(len(x) for x in srcs)
    row[f"{region}_combined_sha"] = sha("".join(sorted(sha(x) for x in srcs))
                                        + "".join(sha(x) for x in DYN[region]))
    if dump:
        os.makedirs(os.path.join(dump, region), exist_ok=True)
        for i, x in enumerate(DYN[region]):
            open(os.path.join(dump, region, f"dynamo_{i}.py"), "w").write(x)
        for x in srcs:
            open(os.path.join(dump, region, f"inductor_{sha(x)[:16]}.py"), "w").write(x)
json.dump(row, open(out, "w"), indent=1)
print(json.dumps({k: row[k] for k in row if k.endswith(("combined_sha", "modules", "graphs", "finite"))}),
      flush=True)
