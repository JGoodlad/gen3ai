"""Does ONE eager R1 call change anything the NEXT call reads? (gpu_checks_endstate, 2026-10-09). CPU.

    python state_diff_probe.py [--cfg E]

The R1 argument tuple's static half (`MicroStatic`, rebuilt by `_micro_static(_resolve_fold_flags())`), the
features extractor's scalar / tensor attributes and every policy buffer, before and after an eager `micro_step`
(+ backward). A difference is state that makes "the first consumer of a batch" compute something else than a
later one.
"""
import argparse
import os
import sys

import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noise_cfg as NC                                     # noqa: E402
from agents.model import compile_regions as cr            # noqa: E402
from agents.training.instrumented_ppo.micro_step import micro_step   # noqa: E402


def snap(obj):
    return {k: (v.detach().clone() if th.is_tensor(v) else v) for k, v in vars(obj).items()
            if th.is_tensor(v) or isinstance(v, (int, float, bool, str, type(None)))}


def diff(a, b):
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out.append(k)
        elif th.is_tensor(a[k]) or th.is_tensor(b[k]):
            if not (th.is_tensor(a[k]) and th.is_tensor(b[k]) and a[k].shape == b[k].shape and th.equal(a[k], b[k])):
                out.append(k)
        elif a[k] != b[k]:
            out.append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="E")
    a = ap.parse_args()
    NC.MODEL_SEED = 42
    model, cm, _ = NC.build(a.cfg, 0)
    cm.__enter__()
    b = cr.r1_batch(model, 256)
    model.policy.set_training_mode(True)
    fe = model.policy.features_extractor
    mods = {n: m for n, m in model.policy.named_modules()}
    before = {n: snap(m) for n, m in mods.items()}
    bufs = {n: x.detach().clone() for n, x in model.policy.named_buffers()}
    a1 = NC.args_for(model, model.policy, b, th.device("cpu"))
    o = micro_step(*a1)
    o.loss.backward()
    a2 = NC.args_for(model, model.policy, b, th.device("cpu"))
    print("MicroStatic diffs:", [(f, x, y) for f, x, y in zip(a1[-1]._fields, a1[-1], a2[-1]) if x != y], flush=True)
    after = {n: snap(m) for n, m in mods.items()}
    md = {n: diff(before[n], after[n]) for n in mods}
    print("module attr diffs:", {n: d for n, d in md.items() if d}, flush=True)
    print("buffer diffs:", [n for n, x in model.policy.named_buffers() if not th.equal(x, bufs[n])], flush=True)
    del fe


if __name__ == "__main__":
    main()
