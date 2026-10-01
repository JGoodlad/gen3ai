# K8.3 — the fold rewrite is a PURE REFACTOR: the equivalence evidence (2026-09-30)

**Verdict.** Running fold steps 1–3a as the static micro-step (`instrumented_ppo/micro_step.py`, region
R1 — the static belief bank, the static opponent-intent fold, the PPO core and win-prob BCE as
`(value, weight)` diagnostics) changes what one PPO update computes by float rounding ONLY: the
post-update parameters differ from the legacy inline fold by at most **1.19e-7** (both torches),
inside the legacy fold's OWN reduction-order noise (the same tree at 2 / 4 / 8 CPU threads instead of 1:
**5.4e-6 – 3.3e-5**), every pinned loss within 1.2e-7 relative, and the TB scalar key set is identical.
Tag: MEASURED (the K9 learner golden's production-surface learner on its committed real buffer, CPU,
fp32, eager, one update = 2 epochs × a full + a ragged accumulation group).

**The declared bar** (the brief: "a re-record is justified as a pure-refactor equivalence within
declared bars, never silently"): post-update parameters within the matched-noise envelope — max |Δ|
BELOW the smallest legacy-vs-legacy thread-count control — and every pinned loss within 1e-6 relative;
the same TB tags present. Met on both torches; the golden was re-recorded under both with this reason.

| torch | new vs legacy (1 thread) | legacy 1 vs 2 threads | 1 vs 4 | 1 vs 8 |
|---|---|---|---|---|
| 2.5.1+cu121 | **1.19e-7** | 1.33e-5 | 5.43e-6 | 6.41e-6 |
| 2.8.0+cu126 | **1.19e-7** | 3.26e-5 | 5.43e-6 | 6.41e-6 |

Component-level, float64 (`belief_bank_static_test`, `instrumented_ppo_intent_fold_test`): every static
belief term, metric and gradient equals the legacy function to 1e-12 on REAL stashes and on synthetic
edges (nothing / everything scored, every believed-slot count k = 0..6, PAD labels, NaN in a scored vs
an unscored slot); the intent fold's metrics to 8.8e-16 and its gradients bit-equal (the set-valued
loss, float32 by the legacy code's own `.float()`, to 1.2e-7).

Why fp32 differs at all: masked reductions sum over every slot with `where(mask, x, 0)` instead of over
the selected rows, so the float addition order changes; Adam's normalised step then amplifies rounding
in near-zero gradients — which is why the comparison is against a matched-noise control and not
against zero.

Rerun: `one_update.py <out.npz> [--threads N]` once per tree (PYTHONPATH = that tree's `src/`), then
`compare.py a.npz b.npz`; `results/` holds the comparison JSONs.
