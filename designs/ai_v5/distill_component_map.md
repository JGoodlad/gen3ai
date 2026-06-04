# Component Distillation — Decomposing the `pokemon_encoder` into Cost vs Fidelity

**Prior best:** `cheaper_encoder` — 4.67x speedup, h2h 0.443 (the bar to beat).
**This iteration:** decompose the load-bearing `pokemon_encoder` into sub-components, measure
which carry COST and which carry FIDELITY, then GPU-distill + shrink the load-bearing ones and
re-assemble the cheapest faithful student. All ms are **1-thread, batch-1, CUDA-hidden CPU**
(the worker inference regime), median of stable runs. Teacher full forward ≈ **1.945–2.143 ms**
(reference frames differ slightly across rigs; speedups are computed within a frame).

---

## 1. Component-Importance Map — COST vs FIDELITY, per sub-component

The single table that drives every decision below. **Cost** = isolated 1-thread CPU ms (and % of
the 2.143 ms teacher forward). **Fidelity** = `Δtop1` when that one sub-component is cheapened/
dropped with everything else teacher-frozen (baseline top1 = 0.855; higher Δ = more load-bearing).

| Sub-component | Cost (ms) | % of teacher | Fidelity Δtop1 (worst-case) | token_cos when cheapened | Verdict |
|---|---:|---:|---:|---:|---|
| **team_transformer** (2-layer, 23-token encoder) | **1.129** | **52.7%** | n/a (the head, not in this ablation) | — | **COST KING** — the teacher bottleneck |
| cls_pool (3 CLS cross-attn) | 0.196 | 9.1% | — | — | Cost-only, head-side |
| **unpack** (frozen ObsUnpack) | **0.117** | **5.5%** | **0** (no learnable net) | — | **IRREDUCIBLE FLOOR** — not distillable |
| **embeddings** (per-id: species/move/item/ability/type) | 0.031 | 1.4% | **+0.040** (cos 0.532, KL +60%) | 0.532 | **FIDELITY KING** — the dominant carrier |
| **move_processor** (MLP 83→96→32 + within-mon attn + LN) | 0.058 | 2.7% | **+0.019** low-rank / **−0.005** full Linear | 0.993 (linear) | Fidelity from the **projection only**; width/ReLU/attn non-load-bearing |
| role_encoder (per-mon MLP → 12×128) | 0.038 | 1.8% | +0.001 (linear: +0.000) | 0.978 | Negligible — shrinkable to pure linear, free |
| within_mon_attn (2-head MHA 32d + LN residual) | 0.043–0.070 | ~2.5% | **0** / −0.004 (drop) | 0.852 | **FREE TO DROP** — zero fidelity, cheap to begin with |
| move_mlp (shared, all 48 slots) | 0.018 | 0.8% | (subsumed in move_processor) | — | Negligible |

**Cost is concentrated in the transformer family** (team_transformer 53% + cls_pool 9% = the head
carries ~66% of teacher cost), while the whole `pokemon_encoder` that `cheaper_encoder` distills
away is only **0.308 ms = 14.4%**. **Fidelity is concentrated in exactly one place: per-id
embeddings.** Everything else inside the encoder (move MLP hidden width, within-mon attention, role
encoder capacity) is near-zero fidelity.

> **Method caveat (load-bearing):** token-cosine and Δtop1 **dissociate**. `attn_identity` drops
> cosine to 0.852 yet costs 0 top1; only `emb_meanpool`'s 0.532 cosine maps to real harm. Cosine
> **over-weights** the attention/role transforms relative to task value — the downstream matchup head
> + team transformer recover from large per-token perturbations **as long as per-id identity
> survives**. Rank importance by **Δtop1, not cosine.**

---

## 2. Surgical Shrinks — what each bought (speed) and cost (fidelity)

Each shrink isolates one sub-component, behaviorally distills a cheap replacement on GPU (MSE onto
teacher activations, then soft-KL T=0.7 ε=0.02 onto teacher logits), and measures component ms,
full forward, fidelity, and h2h@bridge. **Faithful bar:** h2h ≥ 0.42.

| Shrink | Component ms (teacher→shrunk) | Component speedup | End-to-end | top1 | kl | h2h (N=150) | Beats 4.67x? |
|---|---|---:|---:|---:|---:|---:|:--:|
| **comp_move_processor** | 0.058 → 0.0064 | **9.09x** | 3.42x | 0.863 | 0.043 | 0.453 ± 0.080 | no |
| **within_mon_attn** | 0.070 → 0.024 | 2.9x | 3.64x | 0.868 | 0.042 | 0.387 ± 0.078 | no |
| **comp_head_transformer → lite_attn** | 0.184 → 0.106 | 1.7x | **6.02x** | 0.856 | 0.049 | 0.433 ± 0.079 | **YES** |

**Reading each result:**

- **move_processor (9.09x on the component, ZERO fidelity cost).** Folding the move MLP
  (83→96→32) + 2-head within-mon attn + LayerNorm into one `Linear(83→32)` (2,688 params) recovers
  the whole path at top1 0.863 (≈ teacher ceiling, **above** cheaper_encoder's 0.858), kl 0.043, token
  cos 0.957. Clean confirmation of the fidelity map: only the input→output projection is load-bearing;
  the hidden width, ReLU, and the entire within-mon attention reconstruct from one Linear. **But
  end-to-end only 3.42x** — it keeps the expensive frozen `pokemon_encoder` (embeddings + role_encoder
  + stitch ≈ 250 µs), so shrinking a 58 µs sub-part caps the gain. **Wrong lever for end-to-end speed.**

- **within_mon_attn (most droppable, near-worthless for speed).** Dropping it entirely costs top1
  0.866→0.859 (−0.007), KL flat; a distilled cheap Linear mixer **matches** the teacher (top1 0.868,
  kl 0.042) and is h2h-identical (0.387 vs teacher-baseline 0.393). But it's only 0.04–0.07 ms (~2% of
  forward), so the faithful-frozen-encoder regime tops out at **3.64x** — far below 4.67x. **Free to
  drop in fidelity; useless for speed.** (Its h2h ~0.39 < 0.42 is the head/policy ceiling, not a cost
  of the shrink — the teacher-attn baseline is also 0.393.)

- **head_transformer → lite_attn (the only stackable winner, 6.02x).** Replacing the 1-layer 8-head
  TransformerEncoderLayer with **two cheap `Linear(128→1)` learned-attention pools** (one per side, no
  QKV/multihead/FFN) cut the head region (tf+gather+mlp) 0.184→0.106 ms (1.7x) and end-to-end
  0.403→0.318 ms → **4.76x→6.02x (+26%)** at **zero fidelity cost** (top1 0.856 vs 0.858, kl 0.049
  vs 0.046, h2h 0.433 — clears the bar). The multihead mixing was nearly redundant; one 1-D learned
  score per side recovers it. Fully attention-free MLP is faster (6.58x) but drops top1 to 0.844 / h2h
  0.413 (below 0.42) → the lite attention carries a **tiny** load-bearing role. ffn128 is most faithful
  (top1 0.865) but slowest (5.05x).

---

## 3. The Assembled Cheapest-Faithful Student

**Only `head_transformer → lite_attn` is composable with a cheap encoder.** The move_processor and
within_mon_attn shrinks keep the expensive frozen teacher encoder (3.4–3.6x) and **cannot** be combined
with the cheap MLP encoder — **but the cheap encoder already bakes in both of their lessons**: a
per-slot linear-ish projection (= the move_processor finding) and no within-mon self-attention (= the
within_mon_attn finding). So the cheapest faithful stack is:

> **[lean cheap per-slot MLP encoder] → [lite_attn token-mix head]**

Built in 3 stages: (1) MSE-distill a lean 1-hidden-layer, width-128 encoder (131K→**63K params**) onto
the teacher's frozen 12×128 role tokens; (2) attach the lite_attn head via soft-KL T=0.7 ε=0.02;
(3) light end-to-end KL finetune so the head absorbs residual encoder error.

| Metric | Assembled student | cheaper_encoder baseline | lite_attn |
|---|---:|---:|---:|
| **Speedup** | **6.40x** | 4.67x | 6.02x |
| **student_ms** | **0.304** (vs teacher 1.945) | — | 0.318 |
| **h2h (N=250)** | **0.428 ± 0.061** | 0.443 ± 0.056 | 0.433 |
| top1 | 0.851 | 0.858 | 0.856 |
| kl | 0.052 | 0.046 | 0.049 |
| ent_ratio | 1.10 | — | 1.09 |

**ASSEMBLED beats 4.67x decisively at 6.40x, and h2h holds faithful: 0.428 ± 0.061 (N=250) is
statistically indistinguishable from the 0.443 ± 0.056 baseline (CIs overlap) and inside the 0.42
bar.** Shrinking the cheap encoder itself (2-hidden-192 → 1-hidden-128) only nudged 6.31x→6.40x and
cost ~0.005 top1 (0.856→0.851) — because **unpack now dominates**, so further encoder shrinking buys
almost nothing.

Files: model `/tmp/distill/comp_assembled.pt`, build `/tmp/distill/comp_assembled.py`, eval
`/tmp/distill/comp_assembled_eval.py`, log `/tmp/distill/comp_assembled_eval.log`.

---

## 4. The Irreducible Inference Floor

Measured on the final 0.304 ms student, the cost now splits:

| Stage | ms | % of student | Distillable? |
|---|---:|---:|:--:|
| **unpack (frozen ObsUnpack)** | **0.107** | **36%** | **NO** — pure Python tensor slicing/gathers, no learnable net, identical for teacher & student |
| lean_encoder | 0.065 | 21% | marginal (already 63K params; shrinking costs top1) |
| head + lite_attn pools + matchup-gather + final MLP | 0.122 | 40% | only place with meaningful headroom left |

**`unpack` is the binding constraint.** It carries no learnable weights, so no distillation touches it
— both teacher and student pay the identical 0.107 ms. The hard ceiling for **any** student is
`teacher / unpack = 1.945 / 0.107 ≈ 18x`; realistically **~6–7x** once a minimal faithful head +
encoder sit on top of the irreducible floor. We are at 6.40x — essentially at that practical ceiling.

The remaining headroom (the 0.122 ms head/glue) is **fidelity-bound, not cost-bound**: shrinking it
further drops top1 below the ~0.855 faithful ceiling. The real limit is now the **~0.86 top1 / ~0.44
h2h policy-side ceiling**, not encoder fidelity. To go faster you must either accept a worse policy or
attack `unpack` (a non-distillation, pure-engineering problem — vectorize/fuse the Python gathers).

---

## 5. Key Insight — where to invest GPU, and is async-distillation paying off?

**For THIS architecture, the one component worth GPU investment is the `head_transformer`.** It is the
single biggest *learnable* chunk in any cheap student (~26% of forward, co-largest with the
non-distillable unpack), and it's the same transformer family that costs the teacher 53% — so it's
exactly where headroom lives. Distilling it to a `lite_attn` (two `Linear(128→1)` pools) was the **only
shrink that composed into the cheap student** and delivered the decisive jump (4.76x→6.02x, +26%, zero
h2h loss). Everything else is a trap:
- **embeddings** are the fidelity king but only 1.4% of cost → **never** shrink them; keep per-id
  identity faithful (the one ablation that hurts: top1 −0.040).
- **move_processor / within_mon_attn / role_encoder** are individually <3% of cost and ~0 fidelity →
  the cheap encoder should simply *bake in* their lessons (linear-ish per-slot projection, no within-mon
  attention, linear role encoder), not be distilled in isolation. Distilling them standalone wins on the
  component (move_proc 9x) but is **wasted GPU** for end-to-end speed because they keep the expensive
  frozen encoder (capped at 3.4–3.6x).

**Is the async-GPU / CPU-cheap trade paying off?** Yes, but with a sharp diminishing return.
Quantitatively, the productive GPU spend bought:
- **4.67x → 6.40x end-to-end = +1.73x CPU speedup** (student 0.30 ms vs ~0.42 ms), essentially all of
  it from the **single** `head_transformer → lite_attn` distillation. That one targeted distill is the
  high-ROI GPU hour.
- The other shrinks (move_processor, within_mon_attn) consumed offline GPU and returned **0 net
  end-to-end speedup** — they only *confirmed* the fidelity map (which let us safely build the lean
  encoder), so their value was diagnostic, not direct.

**Bottom line for the user:** we are now within ~10% of the practical speed ceiling for a faithful
student. The remaining 0.107 ms `unpack` floor (36% of forward) is **not** a distillation target — it's
a vectorization/fusion engineering task. Further GPU distillation of the head can recover at most the
~0.12 ms glue, but only by sacrificing the ~0.86 top1 / ~0.44 h2h policy ceiling. **Stop distilling;
the next 1.5–2x must come from attacking `unpack`, not the network.**

---

## Executive Summary

1. The `pokemon_encoder`'s cost and fidelity live in **different places**: cost is the transformer
   family (team_transformer 53%, cls_pool 9%), fidelity is **only** per-id embeddings (Δtop1 +0.040).
2. Within the encoder, move-MLP width/ReLU, within-mon attention, and role-encoder capacity are **all
   near-zero fidelity** — a single Linear or pure-linear replacement matches the teacher.
3. Three surgical shrinks: move_processor (9.09x component, 3.42x e2e), within_mon_attn (3.64x e2e),
   head_transformer→lite_attn (**6.02x e2e**). Only the last beats the 4.67x bar.
4. **Only the head shrink composes** with a cheap encoder; the other two keep the expensive frozen
   encoder, but the cheap encoder already embodies their lessons.
5. Assembled `[lean cheap encoder] + [lite_attn head]` → **6.40x**, h2h **0.428 ± 0.061 (N=250)** —
   statistically tied with the 0.443 baseline, inside the 0.42 faithful bar.
6. The irreducible floor is **`unpack` = 0.107 ms = 36%** of the student: frozen, learnable-net-free,
   not distillable; hard ceiling `teacher/unpack ≈ 18x`, practical ~6–7x.
7. We are at the **practical ceiling**; remaining headroom is fidelity-bound (the ~0.86 top1 / ~0.44
   h2h policy limit), not cost-bound. The only GPU hour that paid off was the head distillation
   (+1.73x); move/attn distills returned 0 net e2e speedup (diagnostic value only).
8. Next speed gain must come from **vectorizing `unpack`** (engineering, not distillation), not from
   shrinking the network further.

**Final best student:** `assembled` ([lean cheap per-slot encoder] + [lite_attn head]) — **6.40x**
speedup (0.304 ms vs 1.945 ms teacher), **h2h 0.428 ± 0.061 (N=250)**, top1 0.851, kl 0.052.
Files: `/tmp/distill/comp_assembled.pt`, `/tmp/distill/comp_assembled.py`.
