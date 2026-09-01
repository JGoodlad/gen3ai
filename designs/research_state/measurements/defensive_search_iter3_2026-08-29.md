# DEFENSIVE PAIRED SEARCH — iteration 3: CONFIRM BEFORE YOU OVERRULE

*Measured 2026-08-29, 13:54–21:24 UTC · 1275 orientation-games / 637 swap-pairs / 55,958
decisions · CPU-only, 3 shards at `nice 15` (~2 cores realized), BLAS pinned · 7.50 h real elapsed
(22.45 h summed battle wall) · `models/` read-only · zero errors, zero timeouts, zero unfinished.*

Registered in the ledger entry landed at `35dbc3c` ("DEFENSIVE SEARCH ITER 2 … TWO DISPATCHES",
dispatch **J**). Data: [`defensive_search_iter3_2026-08-29.json`](defensive_search_iter3_2026-08-29.json);
rows archived beside it (`defensive_search_iter3_2026-08-29_rows.jsonl.gz`); scoring script
`defensive_search_iter3_report.py`; the code landed as `3c8eb97`.

---

## 1. Verdict

**The rollouts reject 97.5% of what the leaf certifies, and among the 5% of confirms they can
RESOLVE, the leaf's certified overrule is right 51.4% of the time — a coin flip. The win rate is
the null again, now at ±0.0105.** Iteration 2 convicted the leaf by showing that 13× more
certified overrules moved nothing; this iteration convicts it directly, by asking a
sim-scored instrument what those certifications were worth.

| registered prediction (ledger `35dbc3c`, scored, never adjusted) | outcome |
|---|---|
| overrules FALL from 5.82% to **1.5–3.5%** of decisions | **MISSED LOW, by more than an order of magnitude: 0.11%** (55 acted overrules of 49,351 handled decisions). The direction is right and the magnitude is not; the band assumed a 40–75% rejection rate and the measured one is **97.5%**. Probe H had already measured the same filter from the other side (the `playoff` arm's action-change rate collapsed to 0.074 of searched decisions) and that number, not the band, is what this reproduces. |
| report the confirmation REJECTION RATE directly — the leaf-bias meter in vivo | **0.9753** (2,171 of 2,226 attempts). On the 1,956 attempts that carried evidence — excluding 88 where the rollout family raised and 182 the clock cut below `MIN_PAIRS` — **0.9719**. |
| win rate ≥ 0.50, no regression vs iteration 2 on shared seeds (PRIMARY) | **HELD.** Paired **0.4996 [0.4891, 0.5101]**; against iteration 2 on the **637 shared game indices** the paired difference is **−0.0055 [−0.0296, +0.0186]** — zero, with regression excluded at a width that would have caught 3 pp. |
| STRETCH: paired CI above 0.50 (resolves only if the true rate ≥ ~0.525) | **REFUTED**, at the tightest width this programme has produced: ±0.0105, where a true rate ≥ 0.5105 would have cleared. |
| FREE DIAGNOSTIC: what distinguishes a confirmed overrule from a rejected one | **Nothing the leaf knows.** The race's own leaf margin does not separate them (+0.0054 [−0.0095, +0.0201]), nor does root P(win) distance, nor the legal-action count. Only the TURN separates (upheld 12.3 vs rejected 20.2, CI [−10.2, −5.6]) — post-hoc, one of five comparisons, and reported as a lead rather than a finding. |

**The one-line finding: of 2,226 overrules the race certified under 13 rounds of CRN-paired
evidence and the `seq` rule's anytime guarantee, paired terminal rollouts upheld 55. On 60.5% of
them the substituted action changed NOTHING about the outcome in every paired line it was tried
in, and the sign of the paired difference split 429 for the overrule against 450 for the policy.**
That is not a weak signal being lost in noise; it is the absence of a signal, measured with the
opponent's response in the loop — the one ingredient probe G's +0.0219 was measured without.

---

## 2. The cell

**Checkpoint** `models/ai_v9_29_rev1_0823/checkpoints/checkpoint_9995088_steps.zip` — the same
checkpoint as iterations 1–2 and every historical mirror arm, at the same `--games-seed 7`, so
game *g* is the same pinned dice and the same team draw in all three cells. Everything except the
one flag pair is iteration 2's invocation verbatim:

```
python -m main.search_dividend <ckpt> --arm honest --budget 1 --root-strategy defensive \
  --defensive-leaf winprob --defensive-wp-margin 0.15 \
  --defensive-confirm 6 --defensive-confirm-deadline-s 30 \
  --defensive-contested-deadline-s 3.0 \
  --games-start <lo> --games <n> --games-seed 7 --opponents self \
  --battle-timeout-s 1800 --battle-idle-s 120
```

**THE ONE CHANGE:** before acting on an overrule the race has certified, the race's winner and the
policy's own action are settled by up to **6 paired rollouts to a terminal**, under the SAME
post-divergence dice and the same policy-sampling RNG, and the policy's action stands unless the
paired difference clears `2·SE` over `≥ 4` pairs. Gate 0.15, `seq` floor 5, win-prob leaf, depth 1,
contested deadline 3 s, the racer's 64-round supply: all unchanged.

**Sharding.** Three processes over disjoint half-open windows ([0,267), [267,534), [534,800)) via
`--games-start`. Seed and team draw are functions of the index alone (the scoring script refuses
overlapping windows and refuses a mixed cell), so the rows concatenate into exactly the file one
process would have written.

### Why N = 6, and the bound that no affordable N clears

`MIN_PAIRS = 4` is the playoff's floor, so N must exceed it or a single failed pair auto-declines.
N=6 keeps that margin and requires **3 of 6 paired rollouts to flip the game outcome** in the same
direction before an overrule may act — computed by driving the real rule, not asserted. N=8 would
allow 3 of 8 but priced at 9.2–9.8 h against a ≤8 h budget.

**The mission's sizing target — "N such that the confirm's own decision noise is small relative to
the ~5 pp gaps at stake" — is unreachable at any budget, and saying so is part of the result.** The
paired difference's spread is ~0.5 in units of a game outcome, so a 5 pp bar needs several hundred
pairs per decision; at 3.77 s per pair that is ~30 minutes per decision. **The confirm is a COARSE
filter by construction**, and its measured `min_detectable_paired_difference` (0.50) is published
beside every rate it produced. What it can answer is not "is the leaf's 2 pp edge real" but "does
this particular overrule survive contact with the opponent's reply" — and that turns out to be the
question worth asking.

### The cut, pre-registered before any outcome was computed

The N=8 pilot (12 orientation-games, same flags otherwise) measured 2.25 confirm attempts per game
and 2.57 s per pair, projecting **≈8.1 h** for 1600 games against the ≤8 h budget. **A time-based
stopping rule was therefore fixed at 13:59 UTC — five minutes after launch, with no outcome
statistic of the cell computed by anyone — and enforced by a watcher: stop all three shards at
21:24:00 UTC.** Progress monitoring from launch to cut read game counts, error lines and pace only.

A wall-clock instant is independent of every outcome in the cell; a rule that stopped when an
interval "looked resolved" would be optional stopping and would invalidate the interval it stopped
on. The cut fired at 21:24:25 UTC and bought **1275 of the registered 1600 orientation-games (637
of 800 pairs, 79.7%)**. Shard A reached index 216, so **iteration 1's whole cell (indices 0–199) is
covered and the three-way shared-seed row is complete.**

**Timeout hygiene.** Zero timeouts, zero errors, 1275/1275 finished against the 1800 s backstop and
the 120 s idle bound — the latter raised at startup precisely because a nested rollout silences the
live stream for a whole decision.

---

## 3. The headline, and the three paired comparisons

| arm | n (decisive) | win rate | 95% CI | paired | paired CI | pairs |
|---|---|---|---|---|---|---|
| `mirror_honest_1s` — the historical bar | 239 | 0.2929 | [0.239, 0.354] | 0.2938 | [0.235, 0.352] | 120 |
| `playoff_10s` — the best prior arm, 20 s | 80 | 0.4500 | [0.346, 0.559] | 0.4500 | [0.373, 0.527] | 40 |
| `defensive_1s` — iteration 1 | 397 | 0.4937 | [0.445, 0.543] | 0.4938 | [0.459, 0.529] | 200 |
| `defensive_1s_contested3s` — iteration 2 | 1591 | 0.5003 | [0.476, 0.525] | 0.5003 | [0.480, 0.520] | 800 |
| **…`_confirm6` — THIS CELL** | **1270** | **0.4992** | **[0.4718, 0.5267]** | **0.4996** | **[0.4891, 0.5101]** | **637** |

(Historical rows quoted, never re-run. 5 ties, excluded from the denominator.)

| comparison | Δ | 95% CI | excludes 0 |
|---|---|---|---|
| vs **iteration 2**, **paired on the 637 shared indices** — THE REGISTERED ROW | **−0.0055** | **[−0.0296, +0.0186]** | **no** |
| vs iteration 2, unpaired (Newcombe) | −0.0011 | [−0.0379, +0.0357] | no |
| vs iteration 1, paired on the 200 shared indices | +0.0063 | [−0.0316, +0.0441] | no |
| vs `honest_1s`, paired on the 120 shared indices | **+0.2021** | [+0.1398, +0.2643] | yes |

**Why this interval is tighter than iteration 2's on 20% FEWER pairs — a structural fact, not a
lucky draw.** A side-swapped pair scores 0.5 whenever its two orientations agree, i.e. whenever the
game turned on the team draw rather than on which side searched. Here **92.6% of pairs split**
(590 of 637; 23 won-both, 23 lost-both, 1 at 0.25), giving a pair-score sd of **0.1348** against
iteration 2's implied ~0.29. The arm plays the policy on all but 0.11% of decisions, so the mirror
resolves by team draw and the paired estimator becomes very sharp. *An estimator that gets more
precise as the treatment gets smaller is telling you about the treatment, not about your luck* —
and it is why the stretch refutation here is stronger than iteration 2's despite the smaller n.

---

## 4. The confirm stage — the leaf-bias meter

Over the **2,226** attempted confirms (an attempt occurs exactly when the race separated on a
non-policy action — iteration 2's "overrule", by the same definition):

| stratum | n | share |
|---|---|---|
| the rollout family RAISED (no pairs at all) | 88 | 4.0% |
| the clock cut it below `MIN_PAIRS = 4` | 182 | 8.2% |
| **evidence-backed** (≥ 4 paired rollouts) | **1,956** | **87.9%** |
| …of which CONCLUSIVE (cleared 2·SE, either direction) | 107 | 5.5% of backed |
| …**…UPHELD the overrule** | **55** | **2.8% of backed** |
| …**…REVERSED it** (the rollouts preferred the policy) | **52** | 2.7% of backed |
| …INCONCLUSIVE | 1,849 | 94.5% of backed |

Three readings, in ascending order of what they cost the leaf:

1. **The rejection rate is 0.9753** overall and **0.9719** on the evidence-backed stratum — the two
   agree, so the meter is not an artifact of the clock or the driver. The strata are kept apart
   deliberately: `MIN_PAIRS` is a floor on the SAMPLE, so an attempt cut to three pairs is declined
   before any rollout outcome is consulted, and pooling it would hand the clock's refusals to the
   leaf's account.
2. **Among the 107 confirms the rollouts could RESOLVE, the leaf's certification won 55 and lost 52
   — an uphold share of 0.514.** This is the sharpest statement the instrument can make. On the
   decisions where "was the overrule right" is a measurable question at all, the answer is a coin
   flip. A CRN-paired, `seq`-certified separation on the one-ply win-prob leaf carries **no
   directional information** about which action actually wins the game.
3. **60.5% of attempts (1,347) produced a paired mean of EXACTLY ZERO** — under shared dice and
   shared sampling, the substituted action changed nothing about the outcome in every line it was
   tried in. Over all attempts the sign split **429 for the overrule vs 450 for the policy** (mean
   of paired means −0.0055). The middle of the distribution is not "small effects we cannot
   resolve"; it is *no effect*.

**Realized cost:** 11,670 rollout pairs, **5.24 pairs per attempt** (633 attempts truncated below
the 6-pair cap by the 30 s deadline), **3.77 s per pair**, **19.78 s per attempt**, **34.5 s per
game**. 529 individual pairs failed and 27 rollouts hit the 250-turn cap (scored 0.5 by the shared
`gen3_cf_draw_at_cap_v1` rule).

---

## 5. The rate table — and one confound this cell introduced

Over the 49,351 decisions the strategy handled (of 55,958; the rest are forced switches and counted
search failures — §7):

| branch | iteration 2 | **iteration 3** |
|---|---|---|
| FORCED by the gate | 0.7515 | **0.7504** |
| RACED | 0.2485 | **0.2496** |
| …separated, of raced | 0.4542 | **0.3554** |
| …**PROPOSED** overrule, of all decisions | 0.0582 | **0.0451** |
| …**ACTED** overrule, of all decisions | 0.0582 | **0.0011** |
| rounds per race | 13.17 | **8.89** |
| mean search s per raced decision | 2.278 (of 3) | **2.518 (of 3)** |

**The gate is identical (75.0% vs 75.2% forced) and the acted overrule rate is the intended
collapse. But the RACE got weaker, and that is this cell's own doing.** The same 3 s deadline
bought **8.89 rounds instead of 13.17** while spending MORE of it (2.518 s vs 2.278 s) — each round
cost more wall-clock, because the confirm's terminal rollouts raise the box's load and the race's
rounds are wall-clock-bounded. Separation fell with it (0.454 → 0.355), and the PROPOSED overrule
rate — the row that should have been invariant, since the confirm is a filter after the race, not a
change to it — fell 0.0582 → 0.0451 with it.

**What this does and does not threaten.** It does not touch the headline: the rejection rate is
conditional on a certification having occurred, and 2,226 of them is ample. It does mean this
cell's race is somewhat evidence-poorer than iteration 2's, so the certifications being tested come
from slightly shorter races — if anything a charitable population for the leaf, since iteration 2
showed longer races certify *more* and pay *nothing*. It also means **the confirm partially undid
iteration 2's "spend the bank" gain**, which belongs in the cost column rather than the results
column.

### The realized envelope

| | iteration 2 | **iteration 3** | projected pre-launch |
|---|---|---|---|
| wall s per game | 24.44 | **63.38** | ~52–55 |
| …search | 21.46 | **24.33** | — |
| …confirm | — | **34.54** | ~27–34 |
| uniform-1s notional per game | 37.91 | 38.71 | — |
| SEARCH still banked, of notional | 43.4% | **37.1%** | — |

The projection was made from a 12-game pilot at N=8 and came in 15–20% under, entirely because
`s_per_pair` rose from 2.57 s (pilot) to 3.77 s (cell) — three shards' rollouts contend where one
pilot's did not. **The confirm does not fit inside the bank and is not claimed to:** one pair of
terminal rollouts costs ~3.8 s against the 1 s notional of the whole decision it settles. This
configuration is an INSTRUMENT for reading the leaf, not a deployable ladder setting — iteration
1's uniform 1 s remains that.

---

## 6. Confirmed vs rejected — the free diagnostic

If some cheap feature separated the overrules that survive from those that do not, the expensive
stage could be triaged by it. Every feature is recorded at the decision, BEFORE any rollout was
spent.

| feature | upheld (n=55) | rejected (n=2171) | Welch CI on the difference | separates |
|---|---|---|---|---|
| the race's own **leaf margin** | 0.0893 | 0.0839 | [−0.0095, +0.0201] | **no** |
| root \|P(win) − 0.5\| | 0.0734 | 0.0724 | [−0.0110, +0.0130] | **no** |
| legal actions | 7.87 | 7.55 | [−0.048, +0.689] | **no** |
| game **turn** | **12.3** | **20.2** | **[−10.20, −5.58]** | **yes** |
| \|paired mean\| | 0.585 | 0.105 | [+0.447, +0.513] | mechanical — it IS the decision variable |

**The headline is the first row, and it is a null with teeth.** How strongly the race preferred the
action it certified says *nothing* about whether that action survives contact with the opponent's
reply. The leaf's confidence is not calibrated to the leaf's correctness — which is the winner's
curse stated as a measurement rather than as a mechanism: what a race certifies is the residual
differential bias, and bias does not carry a magnitude that predicts truth.

**The turn effect is a lead, not a finding.** It is one of five comparisons, chosen after the fact,
and the plausible mechanism (early positions have more room for one action to change an outcome;
by turn 20 many lines are decided in ways the gate's 0.15 margin does not catch) is a hypothesis a
later cell would have to register in advance. A pre-registered turn-stratified confirm is the cheap
follow-up if anyone wants one.

---

## 7. Accounting

Of the 55,958 decisions: 49,351 reached the defensive rule (88.2% — 88.4% in both earlier cells),
5,500 `not_move_selection` (9.8%), and **1,107 search failures (2.0%)** — `root_failed` 537,
`prefix_gate_failed` 481, `search_error` 89 — the same prefix-replay-mismatch family iterations 1–2
and the playoff read diagnosed (1.5% and 1.6% there), all falling back to the policy's action and
so biased toward the null. The confirm's own 88 raising attempts and 529 failed pairs are almost
certainly the same family reached through `replay_counterfactual`; the error TEXT is not carried on
the per-decision confirm event, which is a small observability gap this cell noticed and did not
close.

**What was cut: 325 of 1600 orientation-games, by the pre-registered time rule in §2, decided
before any outcome was computed.** Nothing else was cut. Leaf verified in the artifact rather than
assumed: the scoring script REFUSES a cell whose `score_mode` is not `win_prob` on every row or
whose `n_defensive_no_win_prob` is non-zero — both clean here.

**Tests** (all in `3c8eb97`): the shared-clock defect is pinned as a measurement (one pair, then a
refusal) and the fresh clock is pinned to buy N; a conclusive REVERSAL is asserted NOT to read as a
confirmation; the confirm's seconds are asserted out of `elapsed_s` and out of the bank; a
non-overruling verdict never reaches the rollouts; OFF is still a no-op with a runner attached; the
four rejection reasons fold apart; the event list carries the diagnostic fields; and a new CLI gate
reads the REAL construction site in `__main__.main` and fails if any `DefensiveConfig` field is fed
by no flag. Suite: **352 passed** in `src/main/search_dividend` (was 341), **2,413** across
`src/main`, three static gates (mypy, ruff, file-size) green.

---

## 8. Caveats

1. **The registered overrule band is scored as MISSED, and the miss is large.** 0.11% against
   1.5–3.5%. The band's arithmetic (5.82% × a 40–75% rejection rate) was the wrong model; probe H's
   0.074 was the right one and was already in the record. Registered numbers are registered numbers.
2. **The race weakened (§5).** Rounds per race 13.17 → 8.89 under the confirm's own load. The
   comparison to iteration 2 is therefore not perfectly *ceteris paribus* on the race, only on the
   flags. A cell that wanted the race held exactly fixed would have to run the confirm offline
   against iteration 2's recorded overrules — which is dispatch **K**, still open.
3. **The confirm is a COARSE filter and its bound is published** (3 of 6 rollouts must flip the
   outcome). It cannot see a 5 pp edge, and no affordable N can. So "97.5% rejected" means *these
   overrules do not clear a large bar*, not *these overrules are worth exactly zero*. The
   coin-flip uphold share (§4.2) and the 60.5% exact-zero mass (§4.3) are what rule out the
   small-but-real reading, and they do it without depending on the bar.
4. **A rejected confirm lands in `n_defensive_kept`**, because the verdict follows the ACTION
   played. So iteration 2's `kept` column and this cell's are not the same quantity; the record and
   the scoring script both carry `proposed_overrule_*` and `kept_on_own_action` for that reason.
5. **One checkpoint, one budget pair, mirror only, depth 1.** Every caveat from iterations 1–2
   carries over unchanged, including that the mirror carries no ELO anchor, that the gate's
   coverage does not transfer off the mirror, and that depth ≥ 2 breaks the pairing argument the
   leaf depends on.
6. **The rollouts are mirror-exact and only mirror-exact.** A self-rollout IS the estimand here
   because the opponent is the same network; against a different opponent distribution the same
   confirm would carry opponent-model bias, and its rejection rate would be measuring something
   else.

---

## 9. What this changes

**The leaf is convicted twice over, and the second conviction is direct.** Iteration 2 was an
inference from a null: 13× more certified overrules, zero movement. This cell asks the question
outright and gets an answer that does not depend on a win rate — on the decisions where terminal
rollouts can resolve the comparison at all, a race-certified overrule is right **51.4%** of the
time. The one-ply win-prob leaf, CRN-paired and separation-certified, ranks sibling actions no
better than the policy already does. That is the same +0.0219 probe G measured under a FROZEN
opponent, now measured with the opponent replying, and it does not survive.

**The refusal architecture survives everything.** Three cells, three configurations, one result:
gate + futility stop + (now) rollout confirmation holds the arm at the mirror null (+0.202 against
plain search at the same budget) while every non-refusing arm sits 5–21 points below it. Refusing
is still the whole of the value.

**What is retired.** "Better evidence for the same leaf" is now measured at zero three ways — more
rounds (iteration 2), more clock (iteration 2), and an unbiased sim-scored second opinion (this
cell). Adding a fourth filter to a depth-1 win-prob read is not a live lever.

**What the numbers point at instead.** The leaf itself has to change, and the two candidates now
carry a price and a target: (a) a **contrastive** critic objective trained to rank siblings rather
than to score states — probe G sized this at ≤5.7 pp and this cell says the current head has
approximately none of it; and (b) **depth**, which needs the per-decision offset solved first
because the pairing argument that makes depth 1 work collapses at depth ≥ 2. Dispatch **K** (re-run
iteration 2's 3,531 recorded overrules under opponent-marginalized paired rollouts) is now partly
pre-empted by §4 and should be re-scoped: the interesting residue is not "what fraction were
artifacts" — it is 97.5% — but whether the 55 survivors share anything the diagnostic in §6 did not
test.
