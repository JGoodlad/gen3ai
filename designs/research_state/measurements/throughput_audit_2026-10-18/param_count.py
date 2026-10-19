"""Our learner's parameter count at HEAD, by top-level module (CPU, no GPU, no data written).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src python designs/research_state/measurements/throughput_audit_2026-10-18/param_count.py

Builds the production learner exactly as the K9 golden does (`learner_golden.build_learner()`, the
`--arch production` surface) and prints parameters per top-level child of the policy, plus the
embedding-table share (what a "parameter count" quoted by another project usually includes).
"""
from __future__ import annotations

import json
import sys

import torch as th


def main() -> int:
    from agents.training.learner_golden import build_learner

    model = build_learner()
    pol = model.policy
    total = sum(p.numel() for p in pol.parameters())
    by_child = {n: sum(p.numel() for p in m.parameters()) for n, m in pol.named_children()}
    emb = sum(m.weight.numel() for m in pol.modules() if isinstance(m, th.nn.Embedding))
    lin = sum(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
              for m in pol.modules() if isinstance(m, th.nn.Linear))
    out = {"total_params": total, "embedding_params": emb, "linear_params": lin, "by_child": by_child}
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
