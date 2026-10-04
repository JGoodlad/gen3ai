"""CPU cost of the production forward + backward on the learner golden's 64 rows, for an A/B of the
parent tree vs the fix (gen3_nonformula_damage_v1 adds the non-formula override to every kernel).
One thread, eager, fp32; INTERLEAVE the two trees' runs and compare medians.

    PYTHONPATH=<tree>/src python perf.py --reps 30
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    from agents.training import learner_golden as LG
    torch.set_num_threads(1)
    with LG._one_thread():
        model = LG.build_learner()
    LG.load_buffer_into(model)
    rb = model.rollout_buffer
    data = next(rb.get(batch_size=64))
    pol = model.policy
    pol.train()
    times = []
    for i in range(a.reps + 3):
        t0 = time.perf_counter()
        out = pol.evaluate_actions(data.observations, data.actions.long().flatten(), action_masks=data.action_masks)
        values, logp = out[0], out[1]
        (values.sum() + logp.sum()).backward()
        pol.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        if i >= 3:
            times.append(dt)
    print(f"median {statistics.median(times)*1000:.1f} ms  min {min(times)*1000:.1f} ms  n={len(times)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
