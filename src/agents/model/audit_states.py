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

THE SECOND BUG THIS FIXES (`gen3_audit_mask_recovery_v1`, 2026-08-22): the legal mask was
recovered as `logits > -1e8`, which only works if the stored logits are POST-mask. They never
were. `inference/player.py` stashes `logits[0]` — the RAW pre-mask row — while the `-1e9` mask
offset lives in a separate `masked_logits` local that is never written to disk. Measured over
the archive: **0 of 800+ sampled `states.npz` across every run back to ai_v5 carries a single
logit below -1e8**, so the recovery returned ALL-LEGAL on every row of every audit ever run,
and `edge_ablation_audit`'s "zero legal actions" guard passed vacuously by construction. On a
400-file sample **38.4% of the action space was wrongly counted legal** (min 18%, max 68%) —
not cosmetic: the audits renormalize the policy over this mask and sum the KL across it, and
`critic_route_audit` / `endofrun` read the same numbers.

The real mask ships in the sibling `*_summary.json` (`invocations[i]["actions"]` — an ordered
11-entry dict of `{"prob", "valid"}`, index-aligned with the npz rows by construction in
`battle_recorder._all_action_labels`), and from 2026-08-22 also as a first-class `action_mask`
array inside the npz itself. `recover_legal_mask` prefers the array, accepts a genuinely
post-mask logit row, falls back to the summary sibling, and otherwise REFUSES — a vacuous guard
is the bug, so an unrecoverable trace must stop the audit, never default to legal.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# The mask offset the inference path adds to illegal actions (`player.py`: `logits + (mask-1)*1e9`).
# A finite threshold well above it separates "masked" from any real logit — real pre-mask logits
# live at O(10), so nothing legitimate is ever within eight orders of magnitude of this.
MASK_FLOOR = -1e8


class TraceMaskUnavailable(ValueError):
    """No trustworthy legal mask exists for a trace — the audit must stop, not assume all-legal."""


def recover_legal_mask(path: str, z: Any) -> np.ndarray:
    """→ the per-row legal-action mask ``[T, A]`` bool for one `*_states.npz`.

    Sources, in order of preference — each EXACT, never a heuristic:

    1. ``z["action_mask"]`` — the recorder writes the live mask from 2026-08-22 onward.
    2. POST-MASK logits. **The detection rule: a trace is post-mask iff ANY stored logit is
       below ``MASK_FLOOR``.** Only the ``(mask-1)*1e9`` offset can put a logit there, and a
       pre-mask logit never comes close, so the test is exact in both directions. When it
       holds, ``logits > MASK_FLOOR`` recovers the mask that offset encoded.
    3. The sibling ``*_summary.json`` — ``invocations[i]["actions"]`` is an ordered dict whose
       ``valid`` flags are the mask the player actually used, written index-for-index with the
       npz rows. This is the branch every archived trace takes.

    Raises `TraceMaskUnavailable` when none applies. Silence is not an option here: an
    all-legal default is the exact failure this function exists to remove.
    """
    logits = np.asarray(z["logits"])
    n_rows, n_act = int(logits.shape[0]), int(logits.shape[1])

    if "action_mask" in z.files:
        m = np.asarray(z["action_mask"])
        if m.shape != (n_rows, n_act):
            raise TraceMaskUnavailable(
                f"{path}: action_mask shape {m.shape} != logits shape {(n_rows, n_act)}")
        return m.astype(bool)

    if bool((logits < MASK_FLOOR).any()):
        return logits > MASK_FLOOR

    summary_path = (path[: -len("_states.npz")] + "_summary.json"
                    if path.endswith("_states.npz") else "")
    if summary_path and os.path.exists(summary_path):
        with open(summary_path) as fh:
            invocations = json.load(fh).get("invocations", [])
        if len(invocations) != n_rows:
            raise TraceMaskUnavailable(
                f"{path}: summary has {len(invocations)} invocations but the npz has {n_rows} "
                "rows — the trace pair is not aligned, so its mask cannot be trusted")
        rows = []
        for i, inv in enumerate(invocations):
            acts = inv.get("actions")
            n_have = len(acts) if isinstance(acts, dict) else 0
            if n_have != n_act:
                raise TraceMaskUnavailable(
                    f"{path}: invocation {i} carries {n_have} action entries, expected {n_act} "
                    "(duplicate action LABELS collapse the dict and would silently shift "
                    "every index)")
            rows.append([bool(e["valid"]) for e in acts.values()])
        return np.asarray(rows, dtype=bool)

    raise TraceMaskUnavailable(
        f"{path}: no legal mask is recoverable — the npz has no 'action_mask' array, its logits "
        "are PRE-mask (nothing below -1e8, so the historical `logits > -1e8` recovery would "
        "return ALL-LEGAL), and no usable sibling summary was found at "
        f"{summary_path or '<unparseable prefix>'}. Re-record the trace or point the audit at "
        "traces that carry a mask; proceeding all-legal would make the audit's legality guard "
        "vacuous (gen3_audit_mask_recovery_v1).")


def collect_states(patterns: Sequence[str], max_states: int, seed: int = 0,
                   ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """→ ``(obs [N, D] float32, legal_masks [N, A] bool, coverage dict)``, N ≤ max_states.

    The legal mask comes from `recover_legal_mask` — the npz's own ``action_mask``, a genuinely
    post-mask logit row, or the sibling summary's ``valid`` flags. A trace carrying none of
    those raises `TraceMaskUnavailable` and stops the whole sample; it is never defaulted to
    all-legal (`gen3_audit_mask_recovery_v1`)."""
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
            # A file that is malformed SHAPE-wise is skipped above; a file whose MASK cannot be
            # recovered is not — it raises. The two are different failures: the first drops a
            # file from an otherwise-honest sample, the second would silently make every guard
            # downstream vacuous.
            mask_parts[s].append(recover_legal_mask(f, z))
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
