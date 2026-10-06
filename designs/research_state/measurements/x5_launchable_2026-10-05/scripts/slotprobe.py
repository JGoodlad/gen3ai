"""F-XC-3 probe: what does a T2 slot load leave allocated on cuda under fixed_mass (vs blob)?

Builds an InferenceService (graph backend) of the arm's policy, loads a second policy into a slot three
times, and after each load diffs the live CUDA tensors (gc census): shape, bytes, and whether each is held by
the slot replica's extractor / op stash. Usage: slotprobe.py <arm> <out.json>"""
import copy
import gc
import weakref
import json
import sys

import torch

from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
from agents.model.parity_probe import perturb_
from agents.training import learner_golden as LG

arm, out = sys.argv[1], sys.argv[2]
buckets = tuple(int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "8,64").split(","))


def pol(seed):
    m = LG.build_learner() if arm == "blob" else LG.build_arm_learner(arm)
    return perturb_(m.policy.eval(), seed=1000 + seed, scale=0.05)


a, b = pol(0), pol(1)
spec = ServiceSpec(groups=(SlotGroupSpec("pool", 2, a),), device="cuda", backend="graph",
                   buckets=buckets, max_rows_per_flush=256, lanes=int(sys.argv[4]) if len(sys.argv) > 4 else 1)
svc = InferenceService(spec).startup()


def census():
    out = {}
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                out[id(o)] = weakref.ref(o)
        except Exception:
            pass
    return out


def stash_ids(policy):
    ids = {}
    fe = policy.features_extractor
    for owner_name, owner in (("fe", fe), ("op", getattr(fe, "damage_op", None))):
        st = getattr(owner, "stash", None) if owner is not None else None
        if st is None:
            continue
        for f, v in vars(st).items():
            def walk(x, path):
                if torch.is_tensor(x):
                    ids[id(x)] = path
                elif isinstance(x, (list, tuple)):
                    for k, y in enumerate(x):
                        walk(y, f"{path}[{k}]")
                elif isinstance(x, dict):
                    for k, y in x.items():
                        walk(y, f"{path}[{k}]")
                elif hasattr(x, "__dataclass_fields__") or hasattr(x, "_fields"):
                    for k in (getattr(x, "_fields", None) or x.__dataclass_fields__):
                        walk(getattr(x, k), f"{path}.{k}")
            walk(v, f"{owner_name}.stash.{f}")
    # other tensor attributes on any submodule (non-param, non-buffer)
    for name, mod in fe.named_modules():
        pnames = {id(p) for p in mod.parameters(recurse=False)} | {id(x) for x in mod.buffers(recurse=False)}
        for k, v in vars(mod).items():
            if torch.is_tensor(v) and id(v) not in pnames:
                ids.setdefault(id(v), f"fe.{name}.{k}")
    return ids


rows = []
slot = svc.slot("pool", 1)
torch.cuda.synchronize()
for n, p in enumerate((b, a, b)):
    gc.collect()
    before_alloc = torch.cuda.memory_allocated()
    before = census()
    held_before = {}
    for gpol in svc.groups[0].policies:
        held_before.update(stash_ids(gpol))
    del_names = None
    svc.load(slot, p, f"p{n}")
    torch.cuda.synchronize()
    gc.collect()
    after = census()
    grew = torch.cuda.memory_allocated() - before_alloc
    new = [w() for k, w in after.items() if (k not in before or before[k]() is None) and w() is not None]
    gone = [k for k, w in before.items() if w() is None]
    gone_bytes = 0
    gone_owners = {}
    for k in gone:
        o = held_before.get(k, "?")
        gone_owners[o.split(".")[0]] = gone_owners.get(o.split(".")[0], 0) + 1
    held = {}
    for gpol in svc.groups[0].policies:
        held.update(stash_ids(gpol))
    detail = sorted(([list(t.shape), str(t.dtype), t.untyped_storage().nbytes(), held.get(id(t), "?")]
                     for t in new), key=lambda r: -r[2])
    rows.append({"load": n, "grew_bytes": grew, "new_tensors": len(new),
                 "new_bytes": sum(r[2] for r in detail), "top": detail[:40], "gone_n": len(gone), "gone_bytes": gone_bytes, "gone_owners": gone_owners,
                 "by_owner": {}})
    agg = {}
    for r in detail:
        key = r[3].split(".")[0] + "." + (r[3].split(".")[2] if r[3].count(".") >= 2 else r[3])
        agg[key] = agg.get(key, 0) + r[2]
    rows[-1]["by_owner"] = dict(sorted(agg.items(), key=lambda kv: -kv[1])[:20])
    unk = [t for t in new if held.get(id(t), "?") == "?"]
    rows[-1]["unknown"] = []
    for t in unk:
        refs = []
        for r in gc.get_referrers(t):
            if r is new or r is unk:
                continue
            refs.append(type(r).__name__ + (":" + ",".join(list(r.keys())[:6]) if isinstance(r, dict) else ""))
        rows[-1]["unknown"].append([list(t.shape), str(t.dtype), t.untyped_storage().nbytes(), refs[:4]])
    del new, after, unk
    print(json.dumps({k: rows[-1][k] for k in ("load", "grew_bytes", "new_tensors", "new_bytes", "gone_n", "gone_bytes", "gone_owners", "unknown")}))
json.dump({"arm": arm, "buckets": buckets, "rows": rows}, open(out, "w"), indent=1)
