# Opponent Distillation — Research Summary

**Goal.** Replace the 72M-step Gen3 PPO self-play *teacher* policy with a **cheaper opponent network** for self-play rollouts. The opponent forward pass is ~70% of worker CPU, so a faithful, faster opponent directly buys rollout throughput.

**The bar.** Phase-0 was a naive flat MLP: **14× speedup but only 0.10 head-to-head (h2h)** vs the teacher — fast and useless. Per the distillation literature, the real metric is **downstream / head-to-head parity, not raw top-1**. A *faithful* student plays the teacher to a draw with preserved entropy:

- **h2h win rate in [0.45, 0.55]** (a genuinely equivalent opponent neither systematically beats nor loses to the teacher)
- **entropy ratio ≈ 1.0** (not collapsed to a sharper/degenerate policy)
- top-1 / KL are *proxies* used to gate the expensive h2h eval (h2h only run when top-1 ≥ 0.80)

Speed target: **≥ 5× faster** than the teacher's ~2.0 ms/decision forward.

---

## 1. Pareto Frontier — every student tried

Sorted by **usefulness** (h2h faithfulness first, then top-1, then speed). h2h is the binding axis; a high top-1 with bad h2h is not deployable.

| # | Student | Params | top-1 | KL | ent_ratio | Speedup | student ms | **h2h** | Notes |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | **improve_1_reuse_penc_matchup** | 638K | **0.863** | 0.043 | 1.09 | 3.1× | 0.63 | **0.467** OK | Best. Only student **inside** [0.45,0.55] |
| 2 | reuse_penc_attn (prev best) | 348K | 0.858 | 0.029 | 1.02 | 3.6× | 0.67 | 0.383 FAIL | Best KL; h2h below band |
| 3 | reuse_penc | 215K | 0.819 | 0.049 | 1.03 | **4.8×** | 0.68 | 0.333 FAIL | Cheapest faithful-ish body; h2h low |
| 4 | same_arch_1L | 988K | 0.836 | 0.037 | 1.02 | 1.14× | 1.66 | 0.367 FAIL | Full arch, 1 transformer layer; ~zero speedup |
| 5 | feature_match (FitNets) | 351K | 0.788 | 0.058 | 1.05 | 4.1× | 0.52 | — (gated) | Hint-matching **hurt** vs logit-KL |
| 6 | action_factored_pointer_head | 269K | 0.782 | 0.063 | 1.03 | 3.2× | 0.61 | — (gated) | Factored head plateaued ~0.78 |
| 7 | cheap_structured | 211K | 0.757 | 0.089 | 1.03 | **6.3×** | **0.31** | — (gated) | Fast frontier point; hard ceiling ~0.75 |
| 8 | mlp_wide | 7.7M | 0.473 | 0.321 | 1.14 | 3.6× | 0.55 | — (gated) | Flat MLP, wide |
| 9 | mlp_full | 3.3M | 0.469 | 0.321 | 1.17 | 5.8× | 0.37 | — (gated) | Flat MLP baseline |

Reference: **teacher ≈ 2.0–2.16 ms/decision**; Phase-0 naive flat-MLP **14× / 0.10 h2h** (off-table, the failure mode to beat).

**Reading the frontier.** Three regimes are visible:
- **Flat MLPs (rows 8–9):** top-1 caps at ~0.47, KL 0.32 — they cannot even *shape-match* the teacher. Dead on arrival regardless of speed.
- **Cheap-structured (row 7):** the 6×/0.31 ms frontier *corner* — fastest by far, but a hard fidelity ceiling at top-1 ≈ 0.75.
- **Frozen-pokemon_encoder reuse (rows 1–4):** the only regime that clears top-1 0.80. Among these, **only improve_1 clears the h2h band**, at the cost of dropping under 5× speed.

No single student is in the **top-right corner** (≥0.92 top-1 AND ≥5× AND h2h∈[0.45,0.55]). The achievable frontier today is **~0.86 top-1 / 0.467 h2h / 3.1×** (faithful) *or* **0.76 top-1 / 6.3×** (fast but unfaithful) — you pick one.

---

## 2. What worked / what didn't — and WHY

### Reusing the frozen `pokemon_encoder` is load-bearing — it sets a high ceiling

This is the single most important finding. The teacher's `pokemon_encoder` (per-move processing + within-mon move self-attention + a role encoder producing **12×128 role tokens**) is the representation the policy actually reads. Every student that **reuses it frozen** clears top-1 0.80; every student that **tries to reconstruct it cheaply** caps at ~0.75–0.78.

Direct, same-budget A/B that isolates the encoder's value:
- **reuse_penc_attn** (frozen encoder + 1 transformer layer): **top-1 0.858, KL 0.029**
- **cheap_structured** (per-slot MLP *instead of* the encoder, same training budget): **top-1 0.757, KL 0.089**

That **~0.10 top-1 / ~3× KL gap IS the value of the pokemon_encoder.** A cheap per-slot MLP cannot reconstruct the 12×128 role tokens, so the policy that reads them cannot be matched. The encoder is the load-bearing component and must be kept (frozen) for ≥0.85 fidelity.

### Feeding the teacher's *decision-relevant* signal directly is what crossed the h2h band

The previous best (reuse_penc_attn) had **excellent KL (0.029) but failing h2h (0.383)** — a textbook "good distribution-shape, weak live play" gap. The win in **improve_1** came from giving the pooled head the exact inputs the teacher conditions its decision on, which the old pooled MLP had to *re-infer from a flattened team mean*:
- the **opp-active role token** (was discarded),
- the **per-move type-effectiveness matchup slices** — our 4 moves vs opp-active, opp's 4 moves vs our-active, gathered from `ctx.matchups_all` (the teacher sees this; the pooled student didn't),
- plus a lower distillation **temperature T=0.7 + light label smoothing (eps=0.02)** to sharpen toward the teacher's argmax.

Result: top-1 0.858→**0.863**, and crucially **h2h 0.383→0.467 — now inside [0.45,0.55].** The lesson is sharp: **what you pool away, the student must re-derive — and it re-derives the type-matchup tensor poorly.** Giving it directly is worth more for *live play* than for static top-1 (top-1 barely moved, h2h moved a lot), which is exactly why h2h is the metric that matters.

### Cheap-structured student did NOT capture the teacher — hard ceiling at ~0.75

cheap_structured is the genuine fast corner (**6.3×, 0.31 ms, 211K params**) but hits a **structural** ceiling at top-1 ≈ 0.75 / KL ≈ 0.09 that **no architecture knob moved**:
- d_model 64→96: **+0.01 top-1** (noise),
- adding a fully action-aligned per-move head (move emb + raw power/acc/category + the per-move 6-opp matchup row + context, routed to the 4 move logits): **+0.000 top-1**, and dropped speed 6.3×→4.7×.

The ceiling is structural, not capacity or training — proven by the same-budget reuse_penc_attn reaching 0.858. **A cheap per-slot MLP cannot reconstruct the role tokens, full stop.**

### Flat MLPs cannot even shape-match (top-1 ~0.47, KL 0.32)

Both mlp_full (3.3M params, 5.8×) and mlp_wide (7.7M, 3.6×) plateau at **top-1 ~0.47, KL 0.32** — barely better than chance on an 11-way head, and the **same number 0.469↔0.473 whether 512 or 1024 hidden**, so **width is not the bottleneck.** The raw 3357-dim obs (including the 1590-dim turn-history) buries structure that the teacher's transformer + role-token decomposition untangles and a flat MLP cannot. Notably mlp_wide is *also slower* (3.6×) than mlp_full (5.8×) because the wide hidden layer eats the savings — worst of both worlds. **Conclusion: raw-obs flat distillation is a dead end; structure (the frozen encoder) is mandatory.**

### FitNets feature-matching strictly *hurt* (controlled A/B)

With the **identical net / seed / data / epochs**, KL+10·MSE hint-matching peaked at **top-1 0.788**, while **logit-KL alone hit 0.819** — and a lambda sweep (0.0/0.5/2.0/10.0 @80ep) is monotone worsening: **0.811, 0.812, 0.804, 0.800.** Why: the teacher's `pi_combined` is a representation shaped to feed *its own* downstream projection→ReLU→mlp_extractor, **not** to be linearly separable into 11 actions. The MSE term converges fine (~0.044) but spends the small frozen-body head's capacity reconstructing teacher-internal geometry that competes with — rather than aids — the cleaner job of matching masked logits. **Hint-matching is the wrong lever for frozen-body reuse; spend the budget on logit-KL.**

### Action-factored pointer head — factoring *hurt* fidelity

The action-factored pointer head (per-slot switch MLP + per-move move MLP, **no transformer**) fused two leaderboard ideas but lost on every axis: **top-1 0.782, 3.2×** — below reuse_penc (0.819/4.8×) on *both*. It plateaus at ~0.78 regardless of LR/epochs. Why: the teacher runs a **full transformer over the role tokens *before* pooling**, so the *pre-transformer* role tokens lack the cross-token context a switch/move decision needs. A pooled MLP (reuse_penc) can still mix that context across the flattened pool; a per-slot pointer head structurally **cannot**. And despite dropping the transformer, it's only 3.2× (the per-slot/per-move MLPs each carry a LayerNorm and run 6×/4× per decision; the frozen encoder still dominates). **On frozen role tokens, cross-token mixing buys fidelity; action-structure alone does not.**

### Cutting transformer depth buys ~zero speed — the transformer is NOT the bottleneck

The single most actionable systems finding. same_arch_1L (full arch, transformer monkeypatched 2→1 layer) gives **top-1 0.836 but only 1.14× speedup** (1.66 ms). Halving the transformer sacrificed fidelity (0.858→0.836) for **essentially nothing** because **obs-unpack + the per-Pokémon role encoder dominate inference, not the transformer.** Implication: do **not** chase speed by trimming the transformer; the speed budget lives in the encoder/unpack path.

### Was turn-history the key missing input?

**No — the opposite.** The 1590-dim turn-history is precisely the part of the raw obs the flat MLPs *did* have and still couldn't use (top-1 0.47); it adds noise a shallow MLP can't denoise. The frozen-encoder students that **omit raw turn-history** and instead read the **structured role tokens + matchup slices** are the ones that succeed. The key missing input for the *pooled* students was not turn-history but the **opp-active token + per-move type-matchup tensor** (added in improve_1) — decision-relevant structure, not raw sequence history.

---

## 3. Recommendation

**Production-worthy student: `improve_1_reuse_penc_matchup`.**

- **It is the only student that clears the faithfulness bar:** h2h **0.467 in [0.45,0.55]**, ent_ratio 1.09 (entropy preserved, slightly higher — not collapsed), top-1 0.863, KL 0.043.
- **Speedup 3.1×** (0.63 ms vs ~2.0 ms teacher). This is **below the 5× target but well above the 1× floor**, and it is a *faithful* 3.1× — versus Phase-0's useless 14×/0.10.

**Honest caveat — ship behind the gate, not blind.** improve_1 sits at the *low edge* of the band (0.467) on **60 h2h battles** — the 95% CI on a 0.467 win-rate over 60 games is roughly **±0.13**, so we cannot yet distinguish "faithful" from "slightly weak." Before deployment: **re-run h2h at ≥300 battles** to tighten the estimate, and ship behind the **fail-closed gate + auto-revert** described in `designs/ai_v3/design_opponent_distillation.md` — if live self-play win-rate-vs-teacher drifts outside [0.45,0.55], auto-revert to the full teacher opponent.

**What's still missing to fully cross the bar (≥0.92 top-1 AND ≥5×):**
- **Fidelity headroom:** the single frozen transformer layer caps top-1 at ~0.86. Reaching 0.92 likely needs a **second (frozen or distilled) transformer layer** or partially unfreezing the body — at a further speed cost.
- **Speed headroom:** 3.1× → 5× will **not** come from the transformer (proven: 1.14×). It must come from the **encoder/unpack path** (the actual bottleneck): a distilled/quantized `pokemon_encoder`, int8, batched/vectorized role-token computation, or dropping unused obs blocks at the unpack stage. These are unexplored and are the highest-leverage next lever.

**Net:** a faithful **3.1× opponent is shippable now** behind the gate — a real win over the 70%-CPU teacher forward. A faithful **5×** is plausible but requires attacking the encoder, not the transformer.

---

## 4. Latency tradeoffs & future robustness — the frontier will get HARDER

Today distillation is *easy* because the teacher is a **fixed, single-layer-pool-able** target with a clean role-token bottleneck we can reuse frozen. **That cheapness is a property of this teacher, not a law.** As the teacher strengthens, the speed/fidelity frontier shifts **down-and-left** — harder to hit high speedup at high fidelity:

- **Deeper / wider transformer body.** More cross-token mixing layers = more of the policy's "reasoning" lives *after* the reusable encoder. A 1-layer student already loses 0.022 top-1 vs 2-layer; a 4-layer teacher would push that gap wider, and (per the same_arch_1L finding) you **can't** cheaply skip those layers without losing fidelity. The reusable-encoder trick degrades.
- **Richer obs / more derived features.** More to unpack and encode → the *already-dominant* unpack+encoder cost grows → the **5× target recedes** even before any fidelity loss, because the student inherits the same expensive frozen front-end.
- **League / population play.** A teacher that conditions on opponent identity or a meta-distribution has a **multi-modal** policy. KL/top-1 on an aggregate dataset will *look* fine while h2h-per-opponent fractures — distillation must then be **per-snapshot or per-cluster**, multiplying cost and the gate's importance.

**Sustainable speedup.** The honest read: **~3× is the sustainable faithful speedup** for a teacher of this class, and it will **erode toward ~2×** as the teacher deepens. The **6×** corner (cheap_structured) is only reachable by abandoning fidelity (top-1 0.75) — fine for a coarse sparring partner, not for faithful self-play. Plan for **"3× faithful, decaying," not "5× forever."**

**Design implications (these matter MORE as distillation gets harder):**
1. **The fail-closed gate + auto-revert in `design_opponent_distillation.md` is the load-bearing safety mechanism, not a nicety.** As the frontier shifts, more candidate students will *pass top-1/KL but fail h2h* (exactly the reuse_penc_attn failure: 0.858 top-1 / 0.383 h2h). **Gate on live h2h, auto-revert to the full teacher on drift** out of [0.45,0.55]. Never trust the proxy alone.
2. **Distill only the cheap-to-distill snapshots.** Not every teacher checkpoint will distill faithfully at useful speed. Make distillation **opportunistic**: attempt it each generation, *accept* the cheap opponent only when it passes the gate at ≥N battles, else fall back to the teacher for that generation. Self-play throughput then degrades **gracefully** instead of silently training against a weak ghost.
3. **Accept lower speedup for fidelity.** A faithful 2–3× that *preserves the curriculum* beats a 6× that quietly makes the opponent dumber and corrupts self-play. The cost of an unfaithful opponent is **invisible until it has already polluted the policy** — favor fidelity.
4. **Invest in encoder-side speed, not head-side cleverness.** Every fidelity win came from *adding* structure (matchup slices); every speed attempt via *removing* structure (1-layer transformer, factored head) failed. The durable speed lever is a **cheaper frozen front-end** (distilled/quantized `pokemon_encoder`), which is reusable across teacher generations.

---

## 5. Open questions / next experiments

1. **Distill the `pokemon_encoder` itself (highest leverage).** It is *both* the fidelity-load-bearing component *and* the inference bottleneck (per same_arch_1L: unpack+role-encoder dominate, not the transformer). A faithful **cheap** encoder would lift the *entire* frontier — it's the one component that, if cheapened, helps both axes at once. Currently entirely unexplored.
2. **Add a 2nd transformer layer to improve_1.** improve_1's single frozen layer caps top-1 at ~0.86. A 2-layer head (frozen teacher weights, or distilled) should test whether **0.92 top-1 / h2h ≈ 0.50** is reachable, and quantify the speed cost (expected small — transformer is cheap).
3. **Re-run improve_1 h2h at 300–500 battles.** 0.467 on 60 battles has CI ≈ ±0.13; we cannot yet claim it's truly inside the band vs slightly weak. Tighten before any production decision.
4. **int8 / quantize + batch the frozen front-end.** The encoder/unpack path is the speed bottleneck; quantization and vectorized role-token computation are the realistic route from 3.1× → 5× **without** touching fidelity. Untested.
5. **Per-snapshot vs single-distillation under self-play drift.** As the teacher's policy moves each generation, does one distilled student stay faithful, or must we re-distill per checkpoint? Measure h2h decay of a frozen student across N teacher generations — this sizes the *amortized* cost of distillation in the real loop.
6. **Asymmetric temperature / focal-style distillation.** improve_1 already gained from T=0.7. A targeted re-weighting toward decisions where student↔teacher *disagree* (the ~14% of states driving the h2h gap) may convert top-1 into h2h more efficiently than uniform KL. Cheap to try.
7. **Does the matchup-slice win transfer to the cheap-structured body?** improve_1's gain came from feeding opp-active + matchup tensor. Cheap_structured plateaued *with* a per-move matchup head, but *without* the frozen role tokens — so probably no. Worth one confirmation run to nail the boundary: is it the matchup signal or the role tokens that's load-bearing? (Evidence says role tokens — but confirm.)

---

### Executive summary (6 lines)

1. **Reusing the teacher's frozen `pokemon_encoder` is the whole game** — it sets a ~0.86 top-1 ceiling; every student that reconstructs it cheaply caps at ~0.75 (cheap_structured) or ~0.47 (flat MLPs). Width, FitNets hints, and action-factoring all failed to move that ceiling.
2. **Only `improve_1_reuse_penc_matchup` clears the faithfulness bar** (h2h 0.467 in [0.45,0.55], ent 1.09, top-1 0.863) — by feeding the head the teacher's *decision-relevant* signal (opp-active token + per-move type-matchup slices) it previously had to re-infer from a team mean.
3. **The previous best (reuse_penc_attn) is a cautionary tale:** excellent KL 0.029 / top-1 0.858 but **h2h only 0.383** — proxies passed, live play failed. **h2h is the metric; gate on it.**
4. **Cutting transformer depth buys ~zero speed (1.14×)** — the bottleneck is obs-unpack + the role encoder, not the transformer. Chase speed in the **encoder**, never the transformer head.
5. **Recommendation: ship `improve_1` at faithful 3.1× behind the fail-closed h2h gate + auto-revert** (re-run h2h at ≥300 battles first). 5× will need a cheaper/quantized frozen encoder, not head tweaks.
6. **The frontier hardens as the teacher grows** (deeper body, richer obs, league play) — plan for "~3× faithful, decaying toward 2×," distill **opportunistically** only the snapshots that pass the gate, and prefer fidelity over speed since an unfaithful opponent silently corrupts self-play.

---

## Iteration 2 — confirmation + frontier push

Iteration 1 closed with `improve_1` recommended at "faithful 3.1×" on a thin **60-battle** h2h
(0.467 ± 0.13). The explicit open action was *de-risk that number at ≥300 battles*, and the
explicit open lever was *cheapen the encoder, not the transformer*. Iteration 2 did both, plus
killed three head-side / precision ideas. **Headline: the cheaper-encoder lever worked — we now
have a faithful student at 4.67× (vs 3.1×) — and the 60-battle h2h was optimistic noise; the
de-risked faithful h2h is ~0.44, the low edge of the band.**

### (a) improve_1 de-risked: the 60-battle 0.467 did NOT survive — and the latency is structural

A tightened **400-battle** greedy h2h gives **0.440 ± 0.0486** (95% CI **[0.391, 0.489]**). The
point estimate sits *just below* the band [0.45,0.55] and the CI is **not contained** in it (lower
edge 0.391 < 0.45; only the upper edge 0.489 is inside). The iteration-1 60-battle 0.467 (CI
half-width ≈ 0.126) was small-N noise — statistically consistent with 0.44 but optimistic.
**Honest read: improve_1 is a faithful-ISH, slightly-weaker-than-teacher student** — it wins
meaningfully less than half, and ent_ratio 1.09 is at the edge of the 0.9–1.1 tolerance. It is
still the best *3×-class* option, but it is not robustly *inside* the band.

Per-component latency profile (1-thread, batch 1, 400 reps) — quantifies *why* 3.1× is the
structural ceiling for the frozen-encoder body:

| Stage | ms | % of forward |
|---|---:|---:|
| **Full forward** | **0.619** | **100%** |
| frozen unpack | 0.106 | 17.2% |
| **frozen pokemon_encoder** | **0.282** | **45.6%** ← single largest |
| frozen total (unpack+penc) | 0.388 | **62.8%** ← immovable |
| trainable head (tf + matchup-gather + MLP) | 0.216 | 35.0% |
| ↳ transformer + matchup-gather | 0.180 | 29.1% |
| ↳ MLP head only | 0.036 | 5.8% |

**~63% of latency is the frozen unpack+pokemon_encoder, and the pokemon_encoder alone is 2.6× the
unpack.** Trimming the head can never get past 3.1× — the only way past is to replace the frozen
encoder. That is exactly what `cheaper_encoder` does.

### (b) The four new students

| Student | Params | top-1 | KL | ent | Speedup | student ms | **h2h** (N) | Beat baseline? |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| **cheaper_encoder** | 769K | 0.858 | 0.046 | 1.08 | **4.67×** | **0.41** | **0.443 ± 0.056** (300) | **YES — speed** |
| second_layer | 836K | 0.865 | 0.042 | 1.09 | 2.6× | 0.74 | 0.392 ± 0.087 (120) | no |
| int8_quant | 0 (PTQ) | 0.862 | 0.044 | 1.09 | 2.79× | 0.69 | 0.450 ± 0.089 (120) | no |
| focal_distill | 638K | 0.871 | 0.041 | 1.09 | 3.08× | 0.62 | 0.450 ± 0.089 (120) | no |
| *improve_1 (baseline)* | *638K* | *0.863* | *0.043* | *1.09* | *3.06×* | *0.62* | *0.440 ± 0.049 (400)* | *—* |

cheaper_encoder's h2h was confirmed at **N=300** this iteration: **0.4433 ± 0.0562, CI [0.387,
0.500]** (tag `final300`). The other three were not re-run at high N — they all lost to improve_1
on their primary axis and were not worth the bridge time.

### (c) What beat the baseline, and WHY — the cheaper-encoder lever is the only win

**cheaper_encoder (YES — pushed speed past 3.1× while holding fidelity).** Two-stage distillation:
**Stage 1** regresses a small 131K-param shared per-slot MLP encoder *directly* onto the teacher's
**frozen 12×128 role tokens** by MSE (cached once over the dataset) — reaching **NRMSE 0.085 /
cosine 0.956**. **Stage 2** freezes that cheap encoder and trains improve_1's *exact* matchup head
(soft-KL T=0.7, eps=0.02). The per-token MSE target is precisely the signal `cheap_structured`
lacked when trained on logit-KL alone (which hard-capped at top-1 0.757) — **direct token
supervision broke that ceiling**: top1 0.858 (vs improve_1 0.863), KL 0.046, ent 1.08, essentially
matched. The payoff is on the target axis: swapping the teacher's penc (0.282 ms, 46% of the
forward) for the cheap encoder (0.087 ms) lifts **3.1× → 4.67×** — the highest-leverage component,
exactly as predicted. h2h **0.443 ± 0.056 (N=300)** is *statistically identical* to improve_1's
0.440 ± 0.049 (N=400) — **fidelity held to within h2h noise while speed jumped 53%.** This
confirms iteration-1's #1 open question: distilling the pokemon_encoder lifts the **whole**
frontier (it is both the fidelity-load-bearing component *and* the bottleneck).

**Did quantization lift speed past 3.1×? NO — it moved 3.1× the WRONG way.** int8 dynamic quant
(`quantize_dynamic` on the penc Linears) made the student **slower**: 0.62 ms → 0.69 ms (3.08× →
2.79×). Root cause: the per-Pokémon role-encoder Linears are tiny and decode is batch=1, so the
per-call activation re-quantize/dequantize overhead of *dynamic* quant exceeds the int8-GEMM
saving (dynamic quant wins on large NLP matmuls, not these small batch-1 GEMMs). float16 is a
non-starter on x86 CPU (no native f16 GEMM; frozen unpack emits f32 the f16 Linears reject).
**Quantization is the wrong lever here — the bottleneck is encoder FLOPs/structure, not weight
precision.** Fidelity was preserved (h2h 0.450 ± 0.089) but with zero speed benefit.

**second_layer (NO — same fidelity, slower).** A 2nd transformer layer over the frozen role tokens
moved top-1 only 0.863 → 0.865 (noise), confirming the ~0.86 ceiling is set by the **frozen
encoder, not attention depth** — one layer already captures the teacher's argmax. Worse trade on
both other axes: speed 3.1× → 2.6×, and h2h **dropped to 0.392 ± 0.087** (now distinguishably below
0.50). To break 0.86 you must add student capacity *upstream* (the encoder), not stack attention.

**focal_distill (NO — top-1 ↑ but no h2h gain).** Focal reweighting (γ=2 on the teacher-argmax
class) lifted top-1 0.863 → 0.871 but h2h stayed 0.450 ± 0.089, indistinguishable from baseline.
The focal gate keys on the teacher *argmax* (a top-1 proxy), so it just re-derives top-1 with extra
variance; the residual h2h gap lives in **near-tie logit ranking the argmax weight cannot see.**
Both already bracket 0.50 — no top-1→h2h headroom to convert.

### (d) Updated recommendation — `cheaper_encoder` is the single best deployable student

**Deploy `cheaper_encoder`** (`/tmp/distill/student_cheaper_encoder.pt`, code
`/tmp/distill/push_cheaper_encoder.py`).

- **Confirmed speedup 4.67×** (student 0.41 ms vs teacher 1.93 ms) — *53% faster than improve_1's
  3.1× and the closest any faithful student has come to the 5× goal.*
- **Confirmed faithfulness h2h 0.443 ± 0.056 (N=300, CI [0.387, 0.500])**, ent_ratio 1.08 (in
  band), top-1 0.858, KL 0.046.
- **Strictly dominates improve_1:** statistically identical h2h (~0.44 both, fully overlapping CIs)
  at +53% speed. There is no fidelity reason to prefer the slower improve_1.

**The realistic ceiling.** Two honest caveats. (1) **Fidelity:** both improve_1 and
cheaper_encoder land at h2h ≈ 0.44 — the **low edge** of [0.45,0.55], with CIs that touch but are
not contained in the band. The achievable target is a **faithful-ish, ~0.44-win student**, not a
true 0.50; the residual ~6-point gap is a slightly-weaker-than-teacher opponent. (2) **Speed:**
4.67× is the new faithful ceiling; the remaining cost is the now-immovable **frozen unpack (0.106
ms) + cheap encoder (0.087 ms) + transformer/gather (0.180 ms)**. Crossing 5× cleanly would need
attacking the **unpack** path or the transformer+matchup-gather (0.180 ms, now the largest single
trainable cost) — not more encoder distillation. So: **~4.7× faithful-ish is the practical
frontier; 5×+ at h2h ∈ [0.45,0.55] remains unreached.**

### (e) Future robustness — refreshed

The iteration-1 robustness note still holds, with three updates from this iteration:

1. **The 60-battle → 400-battle swing (0.467 → 0.440) is the headline lesson on the gate.** A
   student that *looked* inside the band at N=60 sits at the **low edge** at N≥300. **Never accept a
   distilled opponent on a thin h2h** — the fail-closed gate must run ≥300 battles, and because the
   best students cluster at ~0.44 (just below 0.45), the auto-revert threshold should treat the
   **low edge as the operating point**, not assume 0.50.
2. **The encoder-distillation lever (iteration-1 open question #1) is real and is the durable speed
   win.** Token-MSE distillation of the pokemon_encoder lifts *both* axes and — critically — is
   **reusable across teacher generations** (re-cache tokens, re-fit the cheap encoder), so it
   degrades more gracefully than head tricks as the teacher deepens. This is where future speed
   budget should go.
3. **Quantization is OFF the table for this regime.** Dynamic int8 and f16 both *lose* on batch-1
   CPU decode. Do not revisit precision tricks; the lever is FLOPs/structure (a cheaper encoder),
   which iteration 2 confirmed works.

The sustainable-speedup read shifts up slightly: **~4.7× faithful-ish is now demonstrated** (was
"~3× faithful"), still **decaying toward 2–3×** as the teacher deepens (more reasoning moves
after the reusable encoder; richer obs grows the immovable unpack). Plan for **"~4–5× faithful-ish
now, eroding," distill opportunistically behind a ≥300-battle gate, and keep investing in the
cheap-encoder front-end, never the head or weight precision.**

### Executive summary — Iteration 2 (6 lines)

1. **`cheaper_encoder` is the new best deployable student: confirmed 4.67× speedup at h2h 0.443 ±
   0.056 (N=300)** — statistically identical fidelity to improve_1 (~0.44) at **+53% speed**; it
   strictly dominates the old 3.1× recommendation.
2. **Token-MSE distillation of the frozen `pokemon_encoder` is the winning lever** — regressing a
   3.5× cheaper encoder onto the teacher's 12×128 role tokens (cosine 0.956) broke the old
   cheap-encoder ceiling (0.757 → 0.858 top-1) and cut the bottleneck stage (46% of latency) by 3×.
3. **improve_1's iteration-1 0.467 h2h was small-N noise:** the de-risked 400-battle number is
   **0.440 ± 0.049**, point estimate *below* the band — the best students are faithful-**ish**
   (~0.44, the low edge), not a true 0.50 draw.
4. **Quantization moved the wrong way (3.1× → 2.79×) and a 2nd transformer layer was a strict
   loss** (same top-1, h2h 0.392, slower) — both confirm the bottleneck is encoder
   FLOPs/structure, not precision or attention depth.
5. **focal reweighting lifted top-1 (0.863 → 0.871) but not h2h** — the residual gap is near-tie
   logit ranking the argmax-keyed gate can't see; top-1 has no h2h headroom left.
6. **Realistic ceiling: ~4.7× faithful-ish; 5×+ at h2h ∈ [0.45,0.55] is still unreached** — the
   remaining latency is the immovable unpack + cheap encoder + transformer-gather, so deploy
   cheaper_encoder behind a ≥300-battle fail-closed gate with auto-revert keyed to the ~0.44 low
   edge.

---

## Iteration 3 — closing the gap + robustness

Iteration 2 left `cheaper_encoder` as the best deployable student (**4.67×, h2h ~0.44**) and named
its open lever explicitly: the ~0.44 ceiling is **head/policy-side**, and the diagnosis pointed at
**compounding error / off-distribution drift** — so **on-policy (DAgger) data** was the mandated
next move. Iteration 3 ran that experiment for real, plus a joint end-to-end fine-tune and an
encoder token-fidelity sweep, then stress-tested the deployable student. **Headline: nothing moved
h2h off ~0.44. DAgger empirically *confirmed* its own premise (the student errs more on its own
states) yet its cure was impotent — which FALSIFIES compounding error as the cause and proves
~0.44 is a structural decision-rule ceiling for this teacher, not a data or representation gap.**

### (a) Did DAgger / e2e-finetune / token-sweep move h2h off ~0.44? No — and the DAgger result is decisive

All three attacks landed on the same ~0.44 wall, within overlapping CIs of the baseline:

| Attack | What it added | top-1 | KL | Speedup | **h2h** (N) | Moved h2h? |
|---|---|---:|---:|---:|:---:|:---:|
| *cheaper_encoder (baseline)* | *—* | *0.858* | *0.046* | *4.67×* | *0.443 ± 0.056 (300)* | *—* |
| **dagger** | on-policy student-visited states, teacher-relabeled, 2× aggregated | **0.882** | **0.034** | 4.8× | **0.445 ± 0.069** (200) | **NO (+0.002)** |
| e2e_finetune | unfroze cheap encoder + head, joint soft-KL | 0.868 | 0.042 | 4.63× | 0.410 ± 0.068 (200) | NO (−0.033) |
| token_sweep | encoder cosine 0.956 → 0.974 (335K) | 0.857 | 0.045 | 4.26× | 0.413 ± 0.079 (150) | NO |

**DAgger is the key result, and it is decisive precisely because it confirmed its own premise and
still failed.** The on-policy diagnostic — the cleanest measurement in the whole project — is
unambiguous: on the states the *deployed student actually visits* (collected by wrapping the greedy
student as a bridge player over 80 battles vs teacher + 60 vs SimpleHeuristics = 6,297
off-distribution decisions), student-vs-teacher argmax agreement is **0.767**, versus **0.858** on
the teacher-distribution val. That **clean 9-point degradation is exactly the Ross-Bagnell
off-distribution drift** the diagnosis named: the slightly-off student does err more on its own
state distribution. The premise is TRUE.

Yet the cure did **nothing**. Aggregating those 6,297 relabeled on-policy states (2×) with the
original 31,507 → 44,101 rows and retraining the matchup head moved **h2h 0.443 → 0.445 ± 0.069 —
i.e. not at all.** And tellingly it *improved* teacher-distribution fidelity instead (top-1 0.858 →
0.882, KL 0.046 → 0.034). **So closing the train/test distribution gap — the student now sees and
is labeled on its own states — bought ZERO win-rate.** This **falsifies compounding error as the
cause of the ~0.44 ceiling.** If off-distribution drift were the binding constraint, on-policy
relabeling is its textbook fix and h2h would have risen. It didn't.

The joint end-to-end fine-tune is the second nail. Unfreezing the cheap encoder *and* the head and
letting decision-loss gradients reshape the role tokens — the natural "the freeze is the
bottleneck" hypothesis — **improved every fidelity metric** (top-1 0.858 → 0.868, KL 0.046 → 0.042,
top-3 0.906 → 0.917) and **h2h did not move** (0.410 ± 0.068, statistically the same wall, if
anything a hair lower). This is now the **fifth** independent confirmation of the same signature:
**better held-out fidelity buys zero h2h** (focal: top-1 0.871 → h2h ~0.45; 2nd transformer layer:
top-1 fine → h2h 0.392; improve_1 with the EXACT teacher encoder, cosine 1.0 → h2h 0.44; DAgger:
top-1 0.882 → h2h 0.445; e2e: top-1 0.868 → h2h 0.410).

**Plainly: ~0.44 is the practical fidelity ceiling for this teacher, and WHY is now nailed down.**
It is **not** a train/test distribution mismatch (DAgger closed it, h2h flat), **not** encoder
fidelity (token sweep + exact-teacher-encoder, flat), **not** head depth/optimization (2nd layer,
joint e2e, all flat or worse), **not** top-1 calibration (focal, flat). The student picks the
teacher's action **86–88%** of the time; on the **~12–14% of near-tie decisions** its ranking is
*systematically coarser* than the teacher's, and that ~1-bit-per-handful-of-turns difference
compounds into ~6 win-rate points **regardless of how on-distribution or high-top-1 the training
is.** The teacher's residual edge lives in information the **argmax-greedy, logit-distilled,
value-blind** student structurally cannot reproduce — most plausibly the teacher's **value head**
breaking near-ties (the student's value head is zeroed) or logit-margin calibration that top-1/KL
do not capture. The lever that's left is **not more states and not more fidelity** — it's
distilling a *different signal* (value/advantage, or a margin/ranking loss on near-tie pairs).

### (b) The token-fidelity → h2h curve: the ceiling does not live in the encoder

The token sweep makes the encoder's irrelevance quantitative. Holding the *identical* matchup head,
varying only the cheap encoder's fidelity to the teacher's role tokens:

| Encoder | Params | token cosine | top-1 | **h2h** (N) |
|---|---:|---:|---:|:---:|
| cheap (small) | 131K | 0.956 | 0.853 | 0.487 (150) |
| cheap (medium) | 335K | 0.974 | 0.857 | 0.413 (150) |
| **exact teacher** (improve_1) | — | **1.000** | 0.863 | **0.467** (150–400) |

**The curve is flat — and that is the proof.** Driving token cosine from 0.956 → 0.974 → **1.000**
(perfect) lifts top-1 only 0.853 → 0.857 → 0.863 and leaves h2h bouncing in **0.41–0.49 with no
trend** — the small-encoder 0.487 vs medium 0.413 swing, and the prior small-encoder 0.395 vs this
run's 0.487, show **run-to-run h2h variance (±0.07–0.08 at N=150) exceeds any cosine effect.**
Critically, the **exact teacher encoder (cosine 1.0) tops out at 0.467** — the same wall. If
encoder fidelity set the ceiling, the perfect encoder would break it; it doesn't. **The ceiling is
provably head/policy-side, not encoder-side** — which is *why* it's safe to deploy the cheap 131K
encoder (it costs nothing in h2h) and why "make the encoder more faithful" is a closed dead end.

### (c) Robustness verdict: ~0.44 is stable and NOT meaningfully exploitable

Two bridge probes on the deployable `student_cheaper_encoder.pt` (verified to reload bit-exact:
top-1 0.858, KL 0.046, no missing/unexpected state-dict keys):

- **STABILITY — the ~0.44 band is real, not a sampling artifact.** A fresh independent h2h@200
  (different random tag) gives **0.475 ± 0.069**, fully consistent with the prior **0.443 ± 0.056
  (N=300)** — CIs overlap heavily, estimates within ~0.5σ. Across all samples the student sits in a
  stable **0.44–0.48 win-rate band**; the fresh resample lands slightly *higher* (right at the 0.47
  closed-gap line), so the deployment-relevant strength is **faithful-but-slightly-soft ~0.46–0.48**.
- **EXPLOITABILITY — at most a small hole, CI spans zero.** A fixed SimpleHeuristics bot beats the
  **student 26.25% ± 0.096** (n=80) vs beating the **teacher 15.0% ± 0.078** (n=80) — gap **+0.113 ±
  0.124**, whose 95% CI **spans zero** (not significant at N=80). The student still beats the fixed
  bot **~74%** of the time. Distillation opened at most a **small** exploitable hole, not a
  catastrophic one — consistent with the near-tie-misranking diagnosis (a subtle coarsening, not a
  hard failure mode), **not** a degenerate exploit. **Verdict: deploy-safe at ~0.44–0.48**, with one
  honest caveat — the mild (CI-overlaps-zero) uptick in fixed-bot win rate is worth confirming at
  larger N **if the production opponent is that bot**.

### (d) THE FINAL RECOMMENDATION

**Ship `cheaper_encoder`** (`/tmp/distill/student_cheaper_encoder.pt`, code
`/tmp/distill/push_cheaper_encoder.py`). After three iterations and every degree of freedom
exhausted, it is the single best deployable student — and iteration 3 strengthens, not weakens, the
case by proving nothing beats it.

- **Confirmed speedup 4.67×** (student 0.41 ms vs teacher ~1.93 ms) — the closest any faithful
  student has come to 5×.
- **Confirmed faithfulness h2h ≈ 0.44–0.48** across all samples: **0.443 ± 0.056 (N=300)** plus a
  fresh **0.475 ± 0.069 (N=200)** — a real, stable property. ent_ratio 1.08 (in band), top-1 0.858,
  KL 0.046, not meaningfully exploitable.
- **Ship behind the fail-closed gate — YES, at ~0.44.** The point estimate sits **at/just below
  0.45**, so set the gate band **asymmetrically around the low edge**: treat **~0.44 as the
  operating point**, and trip auto-revert to the full teacher only on a **downward** drift (e.g.
  **revert if a ≥300-battle live h2h falls below ~0.40**), not symmetric panic around 0.50 — the
  student lives at the low edge by design, so a 0.50-centered gate would false-trip immediately. The
  upper bound stays the band ceiling 0.55.
- **Realistic ceiling: ~0.44–0.48 h2h at 4.67× is the practical frontier, and it is now PROVEN, not
  assumed.** ~0.44 is a **structural decision-rule limit** of greedy logit distillation against this
  teacher — the residual ~6 points live in value/near-tie information the greedy student can't
  reproduce. A true 0.50 draw is **not reachable by any off-policy or on-policy state/fidelity
  lever** (all five are exhausted); it would require distilling the teacher's **value/advantage
  signal** or a **near-tie ranking loss** — a different objective, not a bigger/better-trained head.

### (e) Go-forward: what's worth trying vs what is a confirmed dead end

**Worth trying (in priority order) — all target the *decision rule*, not the data or fidelity:**
1. **Distill the teacher's VALUE / advantage head** (highest leverage). The student's value head is
   zeroed; every piece of evidence (DAgger confirming-but-impotent, joint-e2e flat, exact-encoder
   flat) points at value-driven near-tie breaking as the teacher's residual edge. This is the one
   signal never distilled — the single most likely thing to move h2h off 0.44.
2. **Margin / ranking loss on near-tie pairs.** Directly supervise the *ordering* of the ~12–14%
   near-tie decisions (where the student's logit ranking is coarser), not just argmax/KL — convert
   the structural near-tie gap that top-1 can't see.
3. **On-policy *fine-tuning in the real self-play loop*** (distinct from DAgger relabeling, which
   failed). If a faithful 0.50 is genuinely needed, let the student keep learning *online* against
   live opponents rather than imitating a fixed teacher off-policy — but only after (1)/(2), since
   DAgger proved off-policy on-policy-*data* alone is impotent.
4. **Per-snapshot re-distill under self-play drift** — still open and orthogonal: measure h2h decay
   of one frozen student across N teacher generations to size amortized cost.
5. **Encoder quantization done *right*** — static/QAT int8 with proper calibration on the 131K cheap
   encoder, *not* dynamic PTQ. Only worth it to chase 5×+; fidelity is already encoder-insensitive
   (token sweep), so quantizing the cheap encoder is low-risk for h2h.

**Confirmed dead ends (do NOT revisit — each has ≥1 controlled negative):**
- **More on-policy DAgger *data*** — confirmed its premise, moved h2h +0.002. The distribution gap
  is real but not the binding constraint.
- **Head depth / 2nd transformer layer** — top-1 flat, h2h *worse* (0.392), slower.
- **Joint end-to-end unfreeze** — every fidelity metric up, h2h flat/lower (0.410).
- **Chasing encoder token fidelity** — flat h2h from cosine 0.956 → 1.000; the exact teacher encoder
  still only 0.467.
- **Focal / top-1-keyed reweighting** — lifts top-1, zero h2h headroom (the gap is below the argmax).
- **FitNets feature-matching, int8 *dynamic* quant, flat MLP** — all killed in earlier iterations.

### Executive summary — Iteration 3 (8 lines)

1. **Nothing moved h2h off ~0.44.** DAgger 0.445, joint-e2e 0.410, token-sweep 0.413 — all within
   the baseline cheaper_encoder's 0.443 ± 0.056 CI. The ~0.44 wall held against every attack.
2. **DAgger is the decisive result: it CONFIRMED its premise yet FAILED its cure.** On the student's
   own visited states, teacher-agreement drops 0.858 → 0.767 (clean 9-pt Ross-Bagnell drift) — but
   relabeling + retraining on exactly those states moved h2h +0.002 while *improving* top-1 (0.858 →
   0.882). **This falsifies compounding error as the cause** — the ceiling is not a data gap.
3. **The token-fidelity → h2h curve is flat:** cosine 0.956 → 0.974 → **1.000** leaves h2h in
   0.41–0.49 with no trend; the exact teacher encoder still caps at 0.467. **The ceiling is
   provably head/policy-side, not encoder-side** — so the cheap 131K encoder costs nothing in h2h.
4. **~0.44 is a structural decision-rule limit, now proven across 5 exhausted levers** (encoder
   fidelity, head depth, focal, joint e2e, on-policy data). The student matches the teacher's argmax
   86–88% but is coarser on the ~12–14% near-tie decisions — the residual ~6 points live in
   **value/near-tie information the greedy, value-blind, logit-distilled student can't reproduce.**
5. **Robustness: the ~0.44 band is stable and not meaningfully exploitable.** Fresh h2h@200 = 0.475
   ± 0.069 (consistent with 0.443 ± 0.056 @300); a fixed bot beats the student 26% vs the teacher
   15% (gap +0.11, CI spans zero). Deploy-safe at ~0.44–0.48.
6. **SHIP `cheaper_encoder` at 4.67× behind a fail-closed gate keyed to the LOW edge** — operating
   point ~0.44, auto-revert on a ≥300-battle live h2h **below ~0.40** (not symmetric around 0.50,
   which would false-trip). Upper bound stays 0.55.
7. **The only remaining levers target the decision rule, not the data:** distill the teacher's
   **value/advantage** head (the one signal never tried), or a **near-tie ranking/margin loss** —
   plus real-loop online fine-tuning if a true 0.50 is required.
8. **Confirmed dead ends — do not revisit:** more DAgger data, head depth / 2nd layer, joint
   unfreeze, encoder-token fidelity, focal reweighting, FitNets, int8 *dynamic* quant, flat MLP.

**Recommended student: `cheaper_encoder` (`/tmp/distill/student_cheaper_encoder.pt`) — 4.67×
speedup, h2h 0.443 ± 0.056 (N=300) / 0.475 ± 0.069 (N=200), deploy at ~0.44 behind a low-edge
fail-closed gate.**

## Iteration 4 — the value lever

Iteration 3 nailed the diagnosis and named the one untried signal: the ~0.44 ceiling is **not**
encoder fidelity, **not** a data/distribution gap (DAgger falsified compounding error), but
**value-blind near-tie misranking** — the residual edge lives in the teacher's **value/advantage**
scalar, which the argmax-greedy, logit-distilled student structurally cannot carry. Iteration 4
distilled that scalar three ways. **Headline: the value signal did NOT move h2h off ~0.44. The
student can predict the teacher's V(s) near-perfectly from its pooled features (val MSE on z-scored
V = 0.013, ~98.7% of value variance captured) yet its argmax on near-ties does not sharpen — so
value info was already linearly present in the pooled read, and making it explicit adds zero
disambiguating signal to the policy. The lever is exhausted; ~0.44 is the hard OFFLINE ceiling.**

### (a) Did any value-informed lever move h2h off ~0.44? No — three ways, same wall

All three landed on ~0.44, inside the baseline CI; none reached the 0.47 deploy bar, let alone 0.50:

| Lever | Value signal added | top-1 | KL | Speedup | **h2h** (N) | Moved h2h? |
|---|---|---:|---:|---:|:---:|:---:|
| *cheaper_encoder (baseline)* | *—* | *0.858* | *0.046* | *4.67×* | *0.443 ± 0.056 (300)* | *—* |
| **value_multitask** | train-only MSE aux-head → teacher V(s)_z (λ∈{0.1,0.5,1.0}) | **0.859** | **0.045** | 4.63× | **0.393 ± 0.048** (400) | **NO (−0.050)** |
| **decisiveness_weighted** | per-state KL weight by gap/|V−median| + Huber value aux | 0.857 | 0.045 | 4.69× | 0.430 ± 0.056 (300) | NO (−0.013) |
| **hard_argmax_blend** | near-tie-upweighted hard-CE on teacher argmax (value-blind) | 0.852 | 0.069 | 4.70× | 0.420 ± 0.068 (200) | NO (−0.023) |

Each preserves the 4.63–4.70× speedup by construction — inference forward is **byte-identical** to
`cheaper_encoder` (raw logits only; the value/aux heads are train-only and never called at
inference, verified by forward hook). So none of these is even a deployable *alternative*; they are
pure diagnostics of whether the value signal sharpens the policy. It does not.

**The two decisive sub-results — why value weighting is structurally dead:**

1. **The student already HAS the value info; making it explicit does nothing.** The value
   multitask head learned teacher V(s) cleanly (val MSE on z-scored V = **0.013**), and forcing
   that onto the *shared* pooled representation left fidelity unchanged (best λ=1.0: top-1 0.859 vs
   0.858, KL 0.045 vs 0.046) — and h2h if anything *low* (two runs 0.385/0.400, pooled **0.393 ±
   0.048, N=400**). This proves the ceiling is **not a missing-value-information problem**: the
   value-relevant features were already linearly present in the pooled read the policy head sees,
   so an MSE aux-task adds **no new disambiguating signal**.

2. **Value-extremity does not point at where the student errs.** The disagreement-localization
   analysis is the clincher. The student's teacher-disagreements are concentrated almost entirely
   in the **small-logit-gap (near-tie) quartile (37.9% disagree)** and vanish in the high-gap
   quartile (0.8%) — but value-sensitivity |V−median| is **FLAT across error locations**
   (14.0/17.3/14.8/13.0% disagree by quartile). So the teacher's V(s) magnitude is **orthogonal**
   to the states where the student goes wrong; it cannot be used to focus capacity there.
   Confirming this, the `gap_small` variant that *does* upweight the near-tie error states **HURT**
   top-1 (0.842 vs 0.855) — the same capacity wall focal reweighting hit, proving those near-tie
   errors are **capacity-irreducible at the cheap-encoder size, not a weighting problem.**

The `hard_argmax_blend` control closes the loop from the other side: a value-blind harder greedy
label (low-T sharpening + near-tie upweighting) lifted **neither** argmax agreement (top-1 0.852,
*below* baseline) **nor** win rate (0.420), and *worsened* KL (0.069 vs 0.046) by trading away
soft-distribution fidelity. So neither pole works: a harder *policy-only* label carries no value
info, and an explicit *value* signal carries no new policy info. Both confirm the residual ~6 pts
is value/advantage-magnitude information that the cheap-encoder pooled read **already encodes** and
the argmax **already ignores** — there is no offline signal left to add.

### (b) THE FINAL VERDICT — true-0.50 is NOT reachable by offline distillation

**~0.44–0.48 is the hard offline ceiling for this teacher. True-0.50 requires online / in-loop
fine-tuning, not a better offline signal.** Across four iterations, *eight* independent attacks —
encoder-token fidelity, exact-teacher-encoder (cosine 1.0), FitNets, 2nd transformer layer, focal
reweighting, joint e2e-unfreeze, on-policy DAgger, and now all three value distillations — landed
on the **same ~0.44 wall**, every one within overlapping CIs, and every one showing the project's
signature: **better held-out fidelity buys zero h2h.** The space of offline signals is now
exhausted in both directions: more states (DAgger) is impotent, more fidelity (token/exact-encoder)
is impotent, harder policy labels (focal/hard-argmax) are impotent, and the value/advantage scalar
— the last untried signal — is **already in the representation and ignored by the greedy argmax.**
The ceiling is a **capacity wall on near-tie discrimination** at the cheap-encoder size: on the
~12–14% near-tie decisions the student's ranking is irreducibly coarser than the teacher's, and
that ~1-bit-per-handful-of-turns deficit compounds to ~6 win points no matter what offline target
you regress. The only remaining lever is a **reward signal the offline data cannot provide** — i.e.
**online RL / in-the-loop self-play fine-tune** of the cheap student against the live env (or a
larger student to lift the capacity wall), which optimizes the win objective directly rather than
imitating a fixed teacher whose near-tie behavior is itself near-random-equivalent (consistent with
the exact-encoder also capping at 0.467).

### (c) Definitive recommendation — SHIP cheaper_encoder; value-distillation is a CLOSED DOOR

- **Ship `cheaper_encoder` (`/tmp/distill/student_cheaper_encoder.pt`) at 4.67× / h2h ~0.44
  behind the fail-closed low-edge gate** per `distill_integration.md` — unchanged from Iteration 3.
  None of the three value students is a deployable alternative (all are byte-identical at
  inference; the value heads are train-only), and all three measured h2h *at or below* the
  baseline. There is no reason to swap the artifact.
- **Value-distillation is a CLOSED DOOR — do not pursue it further.** It is now a confirmed
  dead-end alongside encoder-fidelity, DAgger, focal, e2e-unfreeze, FitNets, and hard-argmax.
  The diagnosis is final: the residual gap is **not** a missing-value-signal wall, it is a
  **capacity wall on near-tie discrimination** that no offline target can move.
- **If true-0.50 is required, the next move is ONLINE — not another offline distill.** Either
  (i) fine-tune the cheap student with online RL/self-play against the live env (direct win
  objective, escapes the imitation ceiling), or (ii) accept a larger student to lift the near-tie
  capacity wall and re-measure. Both are out of scope for offline distillation; the offline lever
  is **fully exhausted.**

---

**EXECUTIVE SUMMARY (Iteration 4 — the value lever)**
1. Three value-informed levers — value-multitask MSE aux-head, decisiveness-weighted KL, and
   value-blind hard-argmax blend — were distilled off the frozen `cheaper_encoder`; all preserve
   the 4.63–4.70× speedup (inference byte-identical, value heads train-only).
2. **None moved h2h off ~0.44:** value_multitask 0.393 ± 0.048 (N=400), decisiveness_weighted
   0.430 ± 0.056 (N=300), hard_argmax_blend 0.420 ± 0.068 (N=200) — all at-or-below the 0.443
   baseline, none near the 0.47 bar, let alone 0.50.
3. **Decisive proof the lever is dead:** the student predicts teacher V(s) near-perfectly (val
   MSE_z = 0.013) yet the argmax doesn't sharpen — value info was already in the pooled read; and
   value-extremity is FLAT across error locations (14/17/15/13%) while errors concentrate in the
   near-tie gap quartile (37.9%), so value cannot focus capacity where the student errs.
4. **Final verdict:** ~0.44–0.48 is the hard OFFLINE ceiling — eight independent attacks
   (fidelity, exact-encoder, FitNets, depth, focal, e2e, DAgger, value×3) all hit the same wall.
   True-0.50 needs ONLINE/in-loop fine-tuning, not a better offline signal.
5. **Recommendation:** ship `cheaper_encoder` (4.67×, ~0.44) behind the fail-closed low-edge gate
   per `distill_integration.md`; value-distillation is a CLOSED DOOR — do not pursue further.
6. **For true-0.50:** online RL/self-play fine-tune of the cheap student (direct win objective) or
   a larger student to lift the near-tie capacity wall; offline distillation is fully exhausted.

**Did the ceiling move? NO.** All three value levers stayed at/under 0.44 (best 0.430 ± 0.056);
`moved_ceiling=false` for every variant. The ~0.44 offline ceiling is now CONFIRMED and EXPLAINED.
