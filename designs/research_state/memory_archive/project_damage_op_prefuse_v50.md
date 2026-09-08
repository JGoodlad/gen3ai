---
name: project_damage_op_prefuse_v50
description: "v50 --damage-op-prefuse BUILT: one pre-attention damage op; +28.2% B=1 CPU, head-block shift 3% of the P1 ceiling; the win IS the refine-loop deletion"
metadata: 
  node_type: memory
  type: project
  originSessionId: a5393c87-26f4-459c-8019-65b19f98cdf6
  modified: 2026-08-02T04:26:37.123Z
---

> **Archived 2026-09-08** — v50 build record; prefuse is production and stated in designs/ARCHITECTURE.md. Preserved verbatim; nothing below is current.

**`--damage-op-prefuse` (v50, `gen3_damage_op_prefuse_v1`) BUILT 2026-08-01, NOT run.** The op used to
run TWICE per forward: a LEAN `discrete_*` recompute in the between-layers refine loop (×2 in
production) + the FULL 835-dim block post-transformer. ON: `_spread_hp_damage` (SpreadBelief +
HPTypeBelief + move-latent table + full op — a new shared helper with TWO call sites) runs on the
PRE-transformer role tokens, its per-our-mon incoming rows inject onto our tokens via a zero-init
`prefuse_proj`, and the SAME full block still feeds both heads. STRUCTURAL bool; OFF bit-identical;
requires `--damage-op` + `--move-belief-prefuse`; MUTUALLY EXCLUSIVE with `--damage-refine-rounds>0`.

**MEASURED CPU (tmp/pfsp_opponent_sweep.py, B=1, 1 thread, min-of-200, idle box): 6.452 → 4.617 ms =
+28.2%, −4,126 aten calls.** But `--damage-refine-rounds 0` ALONE = 4.620 ms — **the CPU win IS the
refine-loop deletion**; the prefuse itself is ~free (one Linear over 6 tokens) and only buys back a
pre-attention physics path the bare deletion would lose. Do not claim the unification as the source of
the speedup.

**RISK QUANTIFIED, not hand-waved (tmp/damage_prefuse_kl.py, 3000 real bridge states, same weights,
injection zeroed):** head block cosine **0.988**, masked KL(post‖pre) **0.0005 = 3.0% of the
re-measured zero-block ceiling** (0.0182), **2.9% argmax flips**. Two structural bounds explain the
smallness, both test-pinned: (1) prefuse is REQUIRED, so the MOVE belief — the op's dominant input — is
**bit-identical** in both shapes; only the spread + HP-type posteriors are re-sourced. (2) At cold start
the block is **bit-identical** (all belief heads zero-init ⇒ token-independent posteriors) — divergence
is created by TRAINING, not by the reordering. FLOOR not verdict: the snapshot is a 500k-step run (v48's
obs 2992→2889 means NO models/ checkpoint loads in-tree, so one had to be trained —
`tmp/prefuse_probe_train.sh`), its absolute ceiling 0.0182 ≪ P1's 0.9385, and a fresh run trains UNDER
the new shape.

**Evidence posture kept honest:** the CPU cost is the justification; the "attention reasons over
full-fidelity physics" story is secondary and this codebase's evidence is AGAINST it (K9/K10 null 3-for-3,
K10a: the lean kernel was already a 91.8%-agreement proxy). The v36/v37 outgoing/status trunk residuals
ride the deleted loop and are NOT reproduced. See [[project_damage_op_block_audit]] (P1/K10/K10a),
[[project_gpu_damage_op]].

**Incidental:** `python src/main/train_rl_agent.py --help` is BROKEN on main (pre-existing, not from this
work) — some help string has a bare `%` → argparse `ValueError: unsupported format character 't' at
index 723`. Help text is `%`-expanded, so literal percents need `%%`.
