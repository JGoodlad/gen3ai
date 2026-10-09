"""Extractor cost per row, `--op-reduction max` vs `principled`, CPU: matmul FLOPs (FlopCounterMode), an
element-op count (every other aten op: max(input, output) elements, 1 op each), parameters, and the eager forward
wall (median of 7, 1 thread). Production config, and the screen bundle (static + move-resolution + speed-physics)."""
import json
import statistics
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

from agents.model.compile_parity_fixture import load_parity_rows
from agents.model import op_reduction_extractor_test as T

torch.set_num_threads(1)
_MM = {"mm", "addmm", "bmm", "baddbmm", "matmul", "linear", "_scaled_dot_product_flash_attention_for_cpu",
       "_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention", "convolution"}


class ElemCount(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.__name__.split(".")[0]
        if name not in _MM:
            sizes = [a.numel() for a in args if isinstance(a, torch.Tensor)]
            outs = out if isinstance(out, (tuple, list)) else [out]
            sizes += [o.numel() for o in outs if isinstance(o, torch.Tensor)]
            if sizes and not name.startswith(("view", "_unsafe_view", "expand", "t", "permute", "slice",
                                              "select", "unsqueeze", "squeeze", "detach", "alias", "as_strided")):
                self.n += max(sizes)
        return out


def measure(extra):
    tog = T._production_toggles()
    tog.pop("op_reduction", None)
    tog.update(extra)
    res = {}
    for mode in ("max", "principled"):
        m, enc = T._build_real_policy(op_reduction=mode, **tog)
        pol = m.policy
        fe = pol.features_extractor
        obs, _ = load_parity_rows(enc.dimension)
        x = {"observation": torch.as_tensor(obs[:64])}
        B = 64
        fc = FlopCounterMode(display=False)
        with fc:
            fe(x)
        with torch.no_grad():
            ec = ElemCount()
            with ec:
                fe(x)
            ts = []
            for _ in range(7):
                t0 = time.perf_counter()
                fe(x)
                ts.append(time.perf_counter() - t0)
        res[mode] = {"matmul_flops_per_row": fc.get_total_flops() / B, "elem_ops_per_row": ec.n / B,
                     "fe_params": sum(p.numel() for p in fe.parameters()),
                     "policy_params": sum(p.numel() for p in pol.parameters()),
                     "fwd_ms_per_64_median7": 1e3 * statistics.median(ts)}
    a, b = res["max"], res["principled"]
    res["delta"] = {k: (b[k] - a[k], (b[k] - a[k]) / a[k]) for k in a}
    return res


out = {"production": measure({}),
       "bundle": measure({"token_encoding": "static", "move_resolution": "on", "speed_physics": "on"})}
print(json.dumps(out, indent=1))
