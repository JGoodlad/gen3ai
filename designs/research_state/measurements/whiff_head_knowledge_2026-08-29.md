# PROBE L — does the WIN-PROB head already know about the whiffs?

*Measured 2026-08-29 · `models/ai_v9_29_rev1_0823`, all 11 trace steps that carry a snapshot,
`sentinel_*` · CPU-only, 2 shards, `nice 15` · data `whiff_head_knowledge_2026-08-29.json`
(+ `_decisions.json.gz`, one row per swept decision).*

## The question

The bait hunt convicted CREDIT and acquitted perception: at the moment we fire an immune move into
a voluntary pivot, α already calls SWITCH and β already names the slot. The owner's follow-up is
the next link in that chain — **at those decisions, did the model's own WIN-PROB head prefer a
different action, at decision time?** If it did, the head is a ready-made teacher and the lever is
distillation-shaped. If it did not, the gap is obs/coverage and no policy-side lever can help until
the head is fixed.

The prior read on this (`bait_programme_habit_verdict.json`, instrument 4) was **n = 23** loop
decisions scored on the **scalar V**. This is the same question at **617 immune-whiff decisions**,
on the **win-prob head** (probe G: the head that actually beats the played action; V does not clear
zero), **with matched controls and a measured dice floor** — both of which the n=23 read lacked and
both of which turn out to matter.

## What was measured

- **Subject** `models/ai_v9_29_rev1_0823`. Every eval-trace step is scored by **its own**
  `snapshot.zip` — the exact weights that wrote those traces — so "at decision time" is literal.
- **Census (model-free)**: `main.prober.loops.analyze_battle` on **834** `sentinel_*` battles across
  11 steps, reading the raw Showdown protocol from each `*_replay.html`, never the rendered
  timeline. 4,567 moved-into voluntary pivots → **677 whiffs** (622 immune, 54 fail, 1 near-zero),
  151 re-clicks, 84 loop battles.
- **The model read**: for each sampled decision, `lookahead` sweeps EVERY legal action one ply —
  our action varies, the opponent plays its **RECORDED** move, CRN on the realized dice — then
  materializes each successor obs through the real encoder and reads the **win-prob head at s′**
  (and the scalar V beside it). An action whose turn ends the battle is scored 1.0 / 0.0, as a
  one-ply search would see it.
- **Controls, drawn from the same battles**: `hit_pivot` (they pivoted, we moved in, it CONNECTED)
  and `no_pivot`. **These are not optional.** Probe G measured that the one-ply win-prob head beats
  the played action *generally* (+0.0219, 35% agreement), so a raw "the head prefers an alternative
  on X% of whiffs" is not evidence of anything whiff-specific.
- **2,013 decisions over 382 battles** after dropping decisions with <3 legal actions
  (672 whiff / 676 hit_pivot / 665 no_pivot). 17 errors: 9 `ArchDriftError` (the `step_4000032`
  trace dir ships no `snapshot.zip`) and 8 materializer prefix desyncs — 0.8%, all reported.
- Bootstrap CIs are **over battles**, never over decisions.

### The load path is verified, not assumed

Reloading each step's own snapshot reproduces the recorded decision **exactly**: over 6 sampled
whiff decisions, max |Δ logit| **1.3e-05**, max |Δp| **4.8e-04** (the trace rounds probs to 0.1%,
so that is rounding), |ΔV| ≤ **6e-06**, |Δ P(win)| **exactly 0.0**. The census's move label also
matches `lookahead`'s echoed `chosen` on every row — the `--inv`-is-a-list-position trap that made
the prior probe's first run audit the wrong decisions cannot recur here.

---

## 1. THE CROSS-TAB — the head knows, and it knows *whiff-specifically*

| arm | n | battles | head ranks an alternative above the played action | median margin | mean margin | scalar V agrees | median chosen-prob |
|---|---|---|---|---|---|---|---|
| **immune whiff** | **617** | 373 | **0.964** [0.948, 0.978] | **0.0492** | 0.0696 | 0.984 | 0.907 |
| all whiffs | 672 | 381 | 0.964 [0.948, 0.979] | 0.0443 | 0.0661 | 0.976 | 0.876 |
| `hit_pivot` control | 676 | 284 | 0.751 [0.720, 0.783] | 0.0317 | 0.0380 | 0.813 | 0.792 |
| `no_pivot` control | 665 | 295 | 0.623 [0.585, 0.661] | 0.0079 | 0.0197 | 0.672 | 0.620 |

**The registered bar was ≥60%. It reads 96.4%.**

But the number that carries the claim is the **difference**, one CI on the contrast:

| whiff − control | Δ head-prefers-alt | Δ mean margin (win-prob units) |
|---|---|---|
| vs `hit_pivot` | **+0.213** [+0.177, +0.248] | **+0.0316** [+0.0233, +0.0408] |
| vs `no_pivot` | **+0.342** [+0.300, +0.384] | **+0.0499** [+0.0403, +0.0601] |

Both clear zero comfortably. The head's disagreement with the policy is **elevated specifically at
whiffs**, over and above its ordinary background disagreement — including over the tightest control
available, a turn where the opponent pivoted exactly the same way and our move connected.

**The alternative the head prefers is never itself a known whiff.** `head_prefers_NONWHIFF_alt` is
bit-identical to `head_prefers_alt` in every arm: in 0 of 617 immune-whiff decisions was the
top-ranked alternative a move already seen to whiff against that same arrival in that same battle.

Split by outcome (the hunt's registered winning-position confound): losses **0.976** / median margin
0.062, wins **0.950** / 0.029. The effect is present in both arms and larger in losses. By step it
is flat from 6M to 24M (0.88–1.00, no trend) — this is not a late-training artifact.

## 2. THE MARGIN IS REAL — measured against the dice, not asserted

The headline is one CRN line. 149 decisions were additionally re-rolled on **6 independent dice
streams per action**, and the paired margin `max_alt wp − chosen wp` recomputed on each. The
spread of that margin across streams **is** the leaf noise floor for this comparison.

| arm | n | median within-decision sd of the margin | CRN margin > 2 sd | preference holds on EVERY dice stream |
|---|---|---|---|---|
| **immune whiff** | 45 | **0.00062** | **0.822** [0.705, 0.929] | **0.867** [0.761, 0.956] |
| `hit_pivot` | 57 | 0.01860 | 0.491 [0.375, 0.597] | 0.596 [0.480, 0.706] |
| `no_pivot` | 45 | 0.00175 | 0.489 [0.341, 0.625] | 0.533 [0.370, 0.706] |

At a whiff the head's preference is **dice-invariant**: median floor 6.2e-04 against a median
margin of 4.9e-02 — **two orders of magnitude of headroom** — and it survives every independent
dice stream in 87% of cases. That is exactly what an immunity should look like: the whiff's
badness is deterministic, so the dice have nothing to say about it. In the controls the same
preference is a coin-flip across dice (0.53–0.60), which is what a genuinely marginal ranking
looks like and is the reason the floor had to be measured rather than assumed.

## 3. α FLAGS THE PIVOT — but it flags *the pivot*, not *the whiff*

| arm | mean P(SWITCH) | median | α top-1 == SWITCH |
|---|---|---|---|
| immune whiff | 0.503 [0.482, 0.523] | 0.532 | **0.724** [0.685, 0.762] |
| loop steps | — | 0.576 | 0.737 |
| `hit_pivot` | 0.492 [0.466, 0.518] | 0.486 | 0.723 |
| `no_pivot` | 0.294 [0.274, 0.313] | 0.237 | 0.356 |

- **whiff − `no_pivot`: +0.209 [+0.181, +0.237]** — strongly elevated.
- **whiff − `hit_pivot`: +0.010 [−0.023, +0.043]** — a clean NULL.

The registered prediction ("α flags the pivot on a majority") **passes**: α's top-1 is SWITCH on
72.4% of immune-whiff decisions, median P(SWITCH) 0.53, rising to 0.576 on loop steps. But the
matched control makes the honest reading sharper than the prediction was: **α is elevated at
PIVOTS, not at WHIFFS.** It is identical at a pivot we punish and a pivot we whiff into. That is
correct behaviour, not a defect — α predicts the *opponent's* action and has no business knowing
whether our move will connect — and it locates the whiff-specific knowledge squarely in the
win-prob head, which is where §1 found it.

## 4. REPEAT OFFENDERS — the disagreement does NOT grow, and it never had room to

| click | n | head prefers alt | mean margin | median chosen-prob | median α P(SWITCH) |
|---|---|---|---|---|---|
| not in a loop | 412 | 0.961 | 0.0677 | 0.858 | 0.490 |
| **1st click** | 81 | **1.000** | 0.0715 | 0.958 | 0.626 |
| **2nd click** | 81 | 0.951 | 0.0687 | 0.946 | 0.562 |
| **3rd+ click** | 43 | 0.953 | 0.0864 | 0.953 | 0.534 |

- later − first, mean margin: **+0.0033 [−0.0214, +0.0269]** — NULL.
- later − first, prefers-alt: **−0.048 [−0.096, −0.014]** — a marginal *decrease*, from a ceiling of 1.000.

**The registered expectation (disagreement grows as α evidence accumulates) is REFUTED, and the
reason is a ceiling, not an absence.** The head is already at 100% disagreement on the *first*
click of a loop. There is nowhere for the 2nd and 3rd to go. The evidence never needed to
accumulate — the head had the answer before the loop began. Meanwhile the policy's confidence is
*flat* across clicks at ~0.95, which is the habit verdict's B3 restated per-ordinal.

## 5. THE STARVATION READING — the size of the gap, in policy probability

For the decisions where the head prefers an alternative, what did the policy actually sample that
alternative at?

| arm | median policy p on the head's preferred action | fraction below p = 0.05 |
|---|---|---|
| **immune whiff** | **0.002** | **0.773** |
| `hit_pivot` | 0.010 | 0.677 |
| `no_pivot` | 0.067 | 0.432 |

At an immune whiff the action the head ranks first is sampled at **two parts in a thousand**, and
sits under 5% three-quarters of the time. This is the habit verdict's exploration-starvation
mechanism measured on the other side of the same coin: the advantage is there, and on-policy
sampling will essentially never realize it.

## 6. THE SHAPING ACCOUNTING — the registered lever is MIS-SPECIFIED

**`win_prob_mode="shaping"` is not reward shaping and is not PBRS.** In this codebase `"shaping"`
is a **stop-gradient toggle on an auxiliary head's INPUT** — `extractor_forward.py:726-728`:

```python
wp_in = value_pooled if self.win_prob_mode == "shaping" else value_pooled.detach()
```

and the term folded into the loss (`instrumented_ppo/ppo.py:728-740`, `value_terms.py:45-46`) is

> `win_prob_coef · BCEwithLogits(win_logit(s), MC episode outcome)`

— **one** state, no successor, no potential difference. `reward_manager.py:265-292` is the declared
exhaustive reward registry; its 8 PBRS members are material / belief / status / progress / hazard /
boost / opp_boosts / roar and its one TERMINAL member is `win_loss`. There is no win-prob entry in
any family, and `grep win_prob` over `src/agents/training/reward*.py` returns nothing. The reward,
the return and the advantage never see this head.

**What it can and cannot reach.** `ProjectionAssembler.forward` returns
`(pi_combined, value_pooled)`, where `pi_combined` is the team pools + the active token + the
non-matchup tail. `value_pooled` is the **value half only**. The win-prob head reads `value_pooled`
and feeds nothing forward, so its sole route to action selection is a shared-trunk gradient.
Measured at step 25,067,760 from the run's own `grad_balance` scalars:

| quantity | value |
|---|---|
| `grad/win_prob_share` | **0.0102** (1.02% of the total trunk gradient) |
| `grad/win_prob_norm_shared` | 0.0398 — **1.73%** of the policy head's own pull (2.298) |
| `grad/win_prob_policy_cosine` | **−0.133** — it pulls *against* the policy gradient |

(`grad_balance.py:107-111` documents `grad/*_share` as an L1-of-norms proxy, i.e. an **upper
bound**; the true share is ≤ 1.02%.)

**And the hypothetical, clearly labelled as one.** *If* a PBRS potential φ(s)=P(win|s) at 0.05
existed, the measured whiff would generate `0.05 × 0.0326 = 1.6e-03` per step (median realized
ΔP(win) on a whiff turn is **−0.0326**; mean −0.0386, p10 −0.133). Against `VICTORY_VALUE = 30`
that is **5.4e-05**; against the *smallest* registered dense term (`BOOST_WEIGHT = 0.03`) it is
5.4e-02; against one 25% HP chunk of material PBRS (0.5) it is 3.3e-03 — and the per-minibatch
advantage normalization would sweep it out. **Homeopathic.** But it is a hypothetical: no such term
runs.

> **Verdict: "a shaping-dose ladder above the ACTIVE 0.05" is not a lever, because 0.05 is not a
> dose of what the ledger thought it was.** Raising `win_prob_coef` raises a representational
> pressure on `value_pooled` that already points 0.133 cosine *against* the policy gradient. It
> cannot express a per-ACTION preference to the policy at **any** coefficient, because the head is
> a state-level scalar: the ranking measured in §1 does not exist inside the network at all. It
> exists only once a **simulator** has been composed with the head — one re-roll per candidate
> action — and PPO never performs that composition. That is the structural reason the knowledge
> cannot leak into behaviour on its own, and the structural argument for the distillation shape.

## 7. THE EVIDENTIAL HEAD — live, and confident exactly where it is wrong

`CfEvidentialHead` (`--cf-evidential`, coef 0.05) is built and **trained** in this run:
checkpoint LayerNorm γ has std 0.029 against an init std of exactly 0.0 (weight decay cannot
manufacture per-element spread), and `cf/evid_*` is logged through step 25M. It is never called
from the extractor forward and has no trace key, so it was read by a fresh forward on the recorded
obs → `stash.value_pooled` → `cf_evid_head`, the same composition the training term uses.

Liveness on 500 decisions at 24M: mean spans 0.213–0.938 (sd 0.165), evidence α+β spans 4.4–16.3,
`corr(mean, win-prob head) = 0.819`, `corr(mean, realized outcome) = 0.488`, and **0%** of rows sit
at `Beta(1,1)`. It is live.

| arm | n | median evidence α+β | median Beta sd | median mean |
|---|---|---|---|---|
| whiff | 90 | **10.07** | 0.132 | 0.746 |
| `hit_pivot` | 79 | 10.04 | 0.124 | 0.770 |
| `no_pivot` | 331 | 9.24 | 0.131 | 0.784 |

**The head does not confess uncertainty at whiffs.** Its evidence is equal to the hit-pivot control
and slightly *higher* than at ordinary decisions; the epistemic width is ~0.13 everywhere. So the
one-ply disagreement of §1 is a *confident* disagreement, not a coin-flip the confessor is hedging.

⚠️ **Weight this by dosage.** The head sees `cf/evid_n ≈ 12` rows per fold over 129 folds
(~1.5k row-visits) and `cf/evid_nll` is flat at ≈ ln 2. A NULL here would be weak evidence; the
readable part is the non-null direction — equal-or-higher confidence at whiffs.

---

## The selected reading, and the decision rule it triggers

**HEAD-KNOWS, at 96.4% with a whiff-specific excess of +0.21 over the tightest control, a margin
two orders of magnitude above its measured dice floor, and a confident evidential head behind it.
POLICY-IGNORES, at a median sampling probability of 0.002 on the action the head prefers.**

The registered decision rule fires the **distillation-shaped** branch: self-distil from the head's
own one-ply ranking on high-confidence disagreements — the defensive-search overrule mechanism
recast as a training signal. Probe L strengthens that branch with a structural argument the
registration did not have: the head's ranking **is not a quantity the network computes**. It is the
head composed with a simulator, one re-roll per candidate action. Nothing in PPO performs that
composition, so no coefficient, no dose and no gradient route can deliver it — only an explicit
teacher that materializes the ranking and writes it back as a policy target.

**The second half of the rule is REFUTED, not merely unselected.** "A shaping-dose ladder above
0.05" does not name a real mechanism: `win_prob_mode="shaping"` is a stop-grad toggle on an
auxiliary BCE, not a PBRS potential (§6), and even the hypothetical PBRS version is homeopathic.
The ledger's gen-12 note that win-prob shaping was "exonerated @0.05, never yet shipped" is
consistent with this — what shipped was the *auxiliary head*, not the reward term, and the two got
conflated.

**This also removes the last excuse for the α/β line.** α is elevated at pivots and identical
between a pivot we punish and one we whiff into (§3) — perception was never the deficit, and this
probe adds that the *value* signal is not the deficit either. Every link in the chain from
perception → belief → physics coordinate → value ranking now measures correct. The failure is
entirely in actuation, and the only levers left are the two the habit verdict named: a
**deliberate-bait exploiter** (raise the cost of the habit) and **search-as-teacher on bait
decisions** (raise the sampling of the starved alternative). Probe L is the quantitative case that
the teacher already exists inside the model and needs only to be materialized.

## Caveats

1. **The opponent is frozen at its RECORDED move**, so the one-ply read measures "ranks correctly
   GIVEN the switch", not "given its own uncertainty". Defensible because α calls SWITCH top-1 on
   72.4% of exactly these decisions — but it is an assumption, and it inherits the same caveat the
   n=23 read carried.
2. **One ply.** A deeper search could reverse a shallow preference. Probe G's finding that the
   per-decision offset cancels between siblings but not across depth says a 1-ply paired read is
   the *favourable* case for the critic; this is not a claim about depth ≥ 2.
3. **Δwp is the head's CLAIM, not realized gain.** Nothing here was confirmed by rollouts to a
   terminal. The relative ranking is the measurement.
4. **`sentinel_*` only** — self-play opponents, matched to the bait hunt's scope. The bot arms and
   the ladder distribution are not covered.
5. **The dice floor rests on 149 decisions**, 45 of them immune whiffs. The floor's *size* is a
   point estimate; the whiff-vs-control contrast in dice stability (0.87 vs 0.53–0.60) is the
   robust part.
6. `step_4000032` ships no `snapshot.zip`, so its 9 sampled decisions could not be scored.
7. The evidential dosage caveat in §7.

## Reproduce

The five producer scripts ship beside this file.

```bash
export PYTHONPATH=$PYTHONPATH:src
cd designs/research_state/measurements
nice -n 15 python whiff_head_knowledge_loadcheck.py                       # the load-path acid test
for s in $(ls <run>/eval_traces); do                                      # the model-free census
  nice -n 15 python whiff_head_knowledge_census.py --step $s --opponent 'sentinel_*' \
      --out /tmp/_c_$s.jsonl && cat /tmp/_c_$s.jsonl >> /tmp/whiff_census_all.jsonl
done
nice -n 15 python whiff_head_knowledge_sweep.py --census /tmp/whiff_census_all.jsonl \
      --seeds 6 --shard 0 --shards 2                                      # the one-ply sweep
nice -n 15 python whiff_head_knowledge_evidential.py --n 500
nice -n 15 python whiff_head_knowledge_analyze.py --out whiff_head_knowledge_2026-08-29.json
```

`--seeds 6` produces the dice floor; `--seeds 0` is the fast CRN-only pass used for the bulk of the
population (the two mix freely — the analyzer takes the floor from whichever rows carry seeds).
`--dry-run` on the sweep sizes the population without touching a model.
