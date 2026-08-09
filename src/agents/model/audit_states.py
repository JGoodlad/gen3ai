"""Stratified, deterministic eval-trace state sampling for the ablation probes.

THE BUG THIS REPLACES (`gen3_audit_state_sampler_v1`): both `edge_ablation_audit.py` and
`op_block_split_audit.py` collected states with `sorted(glob.glob(...))` + break-at-cap. Step
directories sort LEXICALLY — `step_10000032` < `step_2000016` — so every committed measurement
drew its whole sample from ONE mid-run trace dir while presenting itself as a pool average
(design_op_tensors.md §2.5.1). The concat-deletion acceptance clause explicitly requires
stratified sampling, and these probes are its instruments — there is no re-run after the fact.

The sampler here:
  * buckets every matched `*_states.npz` by its two parent dirs — `(step_dir, opponent)`;
  * takes ONE FILE PER (step, opponent) BUCKET per pass until the row cap is covered — so
    every step dir AND every opponent contributes from the first pass, regardless of glob
    order or file sizes;
  * then sub-samples rows to PER-STEP QUOTAS (an even split of the cap across step dirs,
    with shortfall from row-poor steps redistributed) using a SEEDED permutation
    (`np.random.default_rng(seed)` — deterministic, no wall clock: these emit committed
    measurement artifacts). File-level balance alone is NOT enough — early-step battles
    run far longer, so one step's files can carry 5-10x the rows (measured: 43% of a
    1500-cap sample from step_2000016 under uniform row subsampling);
  * returns a `coverage` dict (per-step and per-opponent SAMPLED-row counts + file counts)
    that callers must write into their output's provenance — a reader verifies coverage
    instead of trusting it.
"""
from __future__ import annotations

import glob
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np


def collect_states(patterns: Sequence[str], max_states: int, seed: int = 0,
                   ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """→ ``(obs [N, D] float32, legal_masks [N, A] bool, coverage dict)``, N ≤ max_states.

    The legal mask is recovered from the captured post-mask logits exactly as before
    (finite threshold above the -1e9 masking floor)."""
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no states.npz matched {patterns!r}")

    buckets: Dict[Tuple[str, str], List[str]] = {}
    for f in files:
        opp = os.path.basename(os.path.dirname(f))
        step = os.path.basename(os.path.dirname(os.path.dirname(f)))
        buckets.setdefault((step, opp), []).append(f)
    for v in buckets.values():
        v.sort()
    steps = sorted({s for s, _ in buckets})
    step_opps = {s: sorted(o for s2, o in buckets if s2 == s) for s in steps}

    # One file per (step, opponent) bucket per PASS, until the pool covers a per-step share
    # of the cap (or every file is consumed) — every step AND every opponent contributes from
    # pass 1. The cap check sits BETWEEN passes: a mid-pass break would let pass order decide
    # which buckets contribute at small caps, which is the original bug's shape.
    bucket_keys = [(s, o) for s in steps for o in step_opps[s]]
    cursor = {k: 0 for k in buckets}
    obs_parts: Dict[str, List[np.ndarray]] = {s: [] for s in steps}
    mask_parts: Dict[str, List[np.ndarray]] = {s: [] for s in steps}
    opp_parts: Dict[str, List[np.ndarray]] = {s: [] for s in steps}
    step_rows = {s: 0 for s in steps}
    n_files = 0
    # Collect ~2x the per-step quota so the seeded subsample has slack even when one step's
    # files are tiny; row-rich steps stop early instead of flooding the pool.
    per_step_target = 2 * max(1, max_states // max(1, len(steps)))
    progressed = True
    while progressed and sum(min(r, per_step_target) for r in step_rows.values()) < 2 * max_states:
        progressed = False
        for s, o in bucket_keys:
            if step_rows[s] >= per_step_target:
                continue
            b = buckets[(s, o)]
            if cursor[(s, o)] >= len(b):
                continue
            f = b[cursor[(s, o)]]
            cursor[(s, o)] += 1
            z = np.load(f)
            if "obs" not in z or "logits" not in z:
                continue                                   # malformed file: skip, keep the pass
            k = int(z["obs"].shape[0])
            obs_parts[s].append(z["obs"])
            mask_parts[s].append(z["logits"] > -1e8)
            opp_parts[s].append(np.full(k, o))
            step_rows[s] += k
            n_files += 1
            progressed = True

    live_steps = [s for s in steps if step_rows[s] > 0]
    if not live_steps:
        raise FileNotFoundError(f"no usable states.npz matched {patterns!r}")

    # PER-STEP QUOTAS: an even split of the cap, with shortfall from row-poor steps
    # redistributed to the rest (waterfall over steps sorted by available rows).
    quota = {}
    remaining_cap, remaining_steps = max_states, sorted(live_steps, key=lambda s: step_rows[s])
    for i, s in enumerate(remaining_steps):
        share = remaining_cap // (len(remaining_steps) - i)
        quota[s] = min(step_rows[s], share)
        remaining_cap -= quota[s]

    rng = np.random.default_rng(seed)
    obs_keep, mask_keep, step_keep, opp_keep = [], [], [], []
    for s in live_steps:
        o_s = np.concatenate(obs_parts[s])
        m_s = np.concatenate(mask_parts[s])
        p_s = np.concatenate(opp_parts[s])
        keep = np.sort(rng.permutation(len(o_s))[:quota[s]])
        obs_keep.append(o_s[keep])
        mask_keep.append(m_s[keep])
        step_keep.append(np.full(len(keep), s))
        opp_keep.append(p_s[keep])
    obs = np.concatenate(obs_keep).astype(np.float32)
    masks = np.concatenate(mask_keep).astype(bool)
    steps_arr = np.concatenate(step_keep)
    opps_arr = np.concatenate(opp_keep)

    coverage = {
        "sampler": "stratified_round_robin_v1",
        "seed": int(seed),
        "n_states": int(len(obs)),
        "n_files_read": int(n_files),
        "n_files_matched": int(len(files)),
        "per_step": dict(sorted(Counter(steps_arr.tolist()).items())),
        "per_opponent": dict(sorted(Counter(opps_arr.tolist()).items())),
    }
    return obs, masks, coverage
