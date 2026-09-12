# PREDICTION — the search dividend of the current WIN-PROB value heads

*Written 2026-09-11 ~11:40 UTC, **before the first battery cell was launched**. Scored, never
adjusted. The only search_dividend process that ran before this file was a **4-battle plumbing
smoke** (`tmp/search_dividend/smoke.jsonl`, ctrl10M, 2 swap-pairs, wr 0.50) whose sole purpose was
to time a game and confirm the winprob leaf resolves; its rows are in no cell and are not read as
evidence.*

---

## 1. The question

The owner's week goal is *"validate the value net as win-prob → search + distillation"*. The
critic ladder (ledger 2026-09-10/11; `UNDERSTANDING.md` §4.2b, §7) says the win-prob head
**resolves and conditions weakly**. This measurement asks the behavioural half of the same
question, which no ladder read can answer:

> **Does search that uses the current win-prob head as its LEAF EVALUATOR beat that same
> network's bare policy, at matched teams and matched dice?**

The mirror cell (`--opponents self`, side-swapped) is the instrument: the searched side plays the
*same network with search structurally off*, so the two sides differ in exactly one thing and the
**no-effect point is 0.50 by construction**.

## 2. The protocol (registered, reproduced — not invented)

Reproduced from the registered defensive-search cells:
`designs/research_state/measurements/defensive_search_first_cell_2026-08-29.md` (iteration 1) and
`defensive_search_iter2_2026-08-29.md` (iteration 2), registered in the ledger entries
"THE SYNTHESIS: DEFENSIVE PAIRED SEARCH" and "ITERATION 2 … SPEND THE BANK" (2026-08-29), with the
battery's own reproducibility rules (`battery.py`: per-game seed and team draw are functions of
`(opponent, game_index, --games-seed)` alone; games play **serially** inside a process; shards take
disjoint `--games-start` windows so they concatenate into the file one process would have written).

```
python -m main.search_dividend <ckpt> --arm honest --budget 1 --root-strategy defensive \
  --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0 \
  [--defensive-contested-deadline-s 3.0] \
  --games-start <lo> --games <n> --games-seed 7 --opponents self \
  --battle-timeout-s 1800 --battle-idle-s 120
```

* **`--games-seed 7`** — the historical salt, so the battles (teams + dice) are **identical** to
  the v9 cells this replaces and identical **across the three heads**: every comparison here is
  paired at the battle level.
* **The ladder rungs are the two registered operating points of the defensive strategy**, not an
  invented sweep: **rung A** = contested deadline unset (a contested decision gets the uniform
  `--budget 1 s`; iteration 1's cell exactly) and **rung B** = `--defensive-contested-deadline-s
  3.0` (iteration 2's cell exactly, the best registered operating point).
* **Reference cell:** `--root-strategy grid --arm honest --budget 1` — the *naive* search with no
  defensive scaffolding, whose historical value on a shaped-critic v9 checkpoint was **0.2929**
  (search losing catastrophically). This is the purest form of the question above.
* **CPU, `nice 15`, BLAS pinned, `--impl node`** as in the registered cells. The GPU is left idle
  **on purpose**: the budget is a WALL CLOCK, so moving the forward to cuda would change what a
  second buys and break comparability with every historical cell. Recorded as a decision, not an
  oversight.
* Heads: `ai_v12_11_ladder_ctrl10M` @10000032 (ladder control), `ai_v12_19_ladder_lambda09`
  @10000032 (λ 0.9, the best-calibrated head), `ai_v12_02_winprob_critic` @73121280 (the 73M head).
  All three are `--critic winprob`, so `--score auto` **resolves** to `win_prob` and `winprob` is
  the only legal leaf (`defensive.resolve_for_critic`).

## 3. THE BAR

Every claim is on the **paired** statistic (side-swap pairs, team draw differenced out) and on the
**delta's own CI**, never on a point estimate against a point estimate.

| verdict | rule (defensive rung B, per head) |
|---|---|
| **SEARCH PAYS** | paired CI95 **lower bound > 0.50** |
| **real but unresolved** | point ≥ 0.51 with the CI straddling 0.50 |
| **NO DIVIDEND DETECTED** | CI straddles 0.50 with point in [0.49, 0.51] — the null, at resolving width |
| **SEARCH HARMS** | paired CI95 **upper bound < 0.50** |

Planned width: **400 swap-pairs per head at rung B** (paired CI ≈ ±0.025 at the observed pair sd),
**150 pairs at rung A** (≈ ±0.04), **100 pairs** for the grid reference (≈ ±0.05). A result inside
±0.025 of the null is a *result*, not an n problem — stated in advance so a 0.505 is not read as a
near-miss.

**INCONCLUSIVE rule:** a timeout is never a semantic outcome. Errored/unfinished games are excluded
from the denominator and reported separately; **if more than 25% of attempted battles time out or
error, the cell is INCONCLUSIVE and reported as such**, not as a win rate.

## 4. Registered predictions (scored in the README, never adjusted)

1. **Rung B, all three heads: NO DIVIDEND DETECTED** — paired point in [0.48, 0.52], CI straddling
   0.50, replicating iteration 2's exact null (0.5003 [0.4803, 0.5203]) on a *different critic*.
   Confidence ~65%. The competing hypothesis is that the shaped-scalar leaf was the problem and a
   trained win-prob head fixes it, which would show as a point ≥ 0.52 on at least one head.
2. **Ordering (weak):** λ 0.9 ≥ ctrl10M (better calibration ⇒ a better leaf), and 73M ≥ the two 10M
   heads (more training ⇒ better resolution). Any ordering whose delta CI straddles zero is
   reported as NOT DETECTED, not as a trend.
3. **The grid reference beats its historical 0.2929 but still loses:** predicted 0.35–0.48. If it
   lands at or above 0.50 the win-prob leaf has *removed* the harm that convicted the shaped leaf,
   which would be the largest finding available here.
4. **Overrule rate rises vs iteration 2's 5.82%:** predicted 8–15% at rung B (the smoke's 4 battles
   showed 12.1%). A null win rate at *twice* iteration 2's certified-overrule volume strengthens
   the "the leaf, not the allocator" verdict rather than weakening it.
5. **Forced fraction falls below iteration 2's ~74–82%:** predicted 55–70% (the smoke showed 61.2%),
   because a win-prob head's |P(win)−0.5| distribution is not the shaped critic's.

## 5. What each outcome means for the search-and-distill path

* **PAYS (CI > 0.50 on any head)** — the path is open at that budget. The measured dividend is the
  **ceiling** on what distillation can recover (a student recovers a fraction of its teacher's
  edge), so the number is directly the input to the search-as-teacher (AWR) plan; the next move is
  a teacher cell at that head and budget.
* **NO DIVIDEND DETECTED** — the head is **not good enough to be a leaf**, independently of its
  ladder resolution: there is no behavioural signal to distill, and a search teacher would be
  distilling the policy back into itself at 25× the cost. The binding constraint stays the LEAF,
  which is what §4.2b's weak conditioning predicts, and the next lever is the critic objective
  (contrastive / better resolution), not more search.
* **HARMS (CI < 0.50)** — worse than neutral: distilling from this search would inject the leaf's
  **differential** bias (the winner's-curse mechanism banked at iteration 2 — CRN pairing cancels
  dice noise and the shared offset, so what a race certifies is the residual differential bias as
  much as signal). The search-and-distill path is **blocked** until the head changes.
* The **grid vs defensive** contrast separates "the leaf is bad" from "the leaf is bad *and* needs
  a gate to be survivable": if grid still loses badly while defensive sits on the null, the gate is
  doing all the work and the leaf carries no exploitable signal at all.

---

## 6. AMENDMENT — the extension rule, registered 2026-09-11 19:59 UTC *mid-cell*

**Declared while rung B was 58% played (230 of 400 pairs per head) and after that partial read was
seen.** Stating it here rather than acting on it silently is the whole point: an extension decided
*after* a completed result is optional stopping, and an extension decided *on* a partial result is
optional stopping too unless its RULE and its FINAL n are fixed in advance and both reads are
published.

What was seen at 230 pairs: ctrl10M 0.5250 [0.4952, 0.5548], lambda09 0.5216 [0.4911, 0.5522],
wp73M 0.4947 [0.4634, 0.5260] — the two 10M heads sit above the null with intervals that just
contain it.

**THE RULE, fixed now:** when rung B completes at its registered 400 pairs, if a head's paired CI
still straddles 0.50 with a point estimate ≥ 0.51, that head's cell is **extended to exactly 800
pairs** (fresh game indices 400–799; no index is replayed) and the README reports **both**:

* the **pre-registered 400-pair read** — the primary, the one the bar in §3 is scored against; and
* the **800-pair extended read**, labelled a SECOND LOOK, with the multiplicity stated plainly:
  two looks at the same hypothesis means the nominal 95% interval on the extended read is
  optimistic, so a lower bound that clears 0.50 by a hair on the second look is reported as
  *suggestive*, and only a lower bound clearing it by more than the first look's own half-width is
  read as **SEARCH PAYS**.

No head is extended on any other trigger, and no cell is stopped early for any reason.

---

## 7. ADDITION — two more heads, and the AXES question (registered 2026-09-12 ~01:35 UTC, before the first cell)

Ordered by the coordinator after the three-head read landed. **The bar is unchanged**: paired CI
lower bound > 0.50 in the mirror, on the delta's own CI, at 800 pairs of the registered rung-B
cell; `grid` @1 s at 100 pairs beside it as the unguarded reference. Same protocol, same
`--games-seed 7`, same game indices, so **the two new heads play the same battles as the three
already read** and every contrast is paired.

### The two heads, and why these two

| head | lever | what the CRITIC LADDER said about it |
|---|---|---|
| `ai_v12_17_ladder_strata` @10000032 | class-balanced BCE (`win_prob_strata_weight` 1.0) | **the one lever that moved the ladder's conditioning row** |
| `ai_v12_20_ladder_denseaux` @10000032 | dense within-game auxiliary targets (`win_prob_dense_aux` 1.0) | **null on the conditioning row** — but dense within-game supervision is exactly what the SHAPED critic had, and the shaped critic is the better leaf (the 2026-09-11 finding) |

**The question they jointly answer: are LEAF QUALITY and CONDITIONING separate axes?** The ladder
measures conditioning (does `V` move with the opponent / the team); the mirror measures leaf quality
(does ranking actions on `V` beat the policy). Nothing so far has tested whether a lever that moves
one moves the other.

### What each outcome means — named in advance

* **denseaux pays, strata does not** ⇒ **the axes are SEPARATE, and dense within-game supervision is
  the leaf lever.** The conditioning row would then be the wrong meter to steer the leaf by, and the
  search-and-distill path reopens on a denseaux-style target (next step: a rung-B cell at a larger
  dense-aux coefficient, then a teacher cell).
* **strata pays, denseaux does not** ⇒ **conditioning IS leaf quality**, the ladder's conditioning
  row is a valid behavioural proxy, and the cheap offline meter can steer the search programme.
* **neither pays** ⇒ neither lever touches the leaf; the axes question is not *settled* by this pair
  (a null on both is consistent with "separate but neither moved" and with "same axis, neither
  moved"), and what the cells then decide is narrower but still useful: whether either lever moves
  the MECHANISM rates (separation-of-raced, overrule rate) even with the win rate on the null.
  **That secondary read is pre-registered here as the fallback**, because it is the quantity that
  separated the shaped critic from the win-prob head on 2026-09-11 (45.4% vs 11.9–15.0%).
* **both pay** ⇒ the leaf is improvable by more than one route, and the programme's constraint is
  neither lever but their absence to date.

### Registered predictions

1. **Neither head clears the bar.** Both paired CIs straddle 0.50 at 800 pairs; confidence ~70%,
   for the reason the three-head read gives: `grid` is catastrophic on every win-prob head measured
   so far, and the defensive gate's job is to buy that back to the null, not past it.
2. **denseaux is the better LEAF of the two on the mechanism rates** — higher separation-of-raced
   and a higher overrule rate than strata — because dense within-game supervision is what the leaf
   needs to rank ACTIONS at one ply. Registered directionally; scored on the delta's CI.
3. **strata's `grid` cell is no better than the three already measured (0.2025–0.2600), and may be
   worse.** A class-balanced target sharpens P(win) toward the extremes, and a sharper leaf that is
   still mis-ranked is a worse ranker, not a better one.
4. **denseaux's `grid` cell is the best of the five heads** — the one place a dense-supervision head
   could show a leaf advantage without the gate hiding it. Predicted 0.28–0.40; still below 0.50.
5. The **triage-forced fraction** differs sharply between the two heads (see the disclosure below).

**DISCLOSURE — what was seen before this was written.** Two 4-battle plumbing smokes (timing +
leaf resolution) ran before this section: strata forced **95.8%** of decisions and raced 4.2%;
denseaux forced **67.4%** and raced 32.6% — the extremes of the five heads measured (the other three
sit at 75–80%). Prediction 5 is therefore *informed*, not blind, and is recorded as such;
predictions 1–4 concern win rates and separation, which those 8 battles do not resolve. Smoke rows
are in no cell.

### Two protocol notes, registered because they are deviations

* **THE BOX IS NO LONGER QUIET.** The three-head battery ran at load ~14; a GPU training run and a
  stepcurve eval campaign are now live and the load is ~33. **A budget is a WALL CLOCK**, so a
  second buys less width now: the denseaux smoke realized K = 4.0–4.25 worlds against the earlier
  heads' 4.85–5.48. Therefore a **CONTEMPORANEOUS ANCHOR** is added — `ai_v12_11_ladder_ctrl10M`
  @10000032, the same rung-B cell, 150 pairs, running *alongside* the two new heads. Its job is the
  MECHANISM calibration (at ~11k decisions the separation and overrule rates resolve to ~1% even
  though its win-rate CI is only ±0.035), so that "strata separates less than ctrl10M" is a
  statement about the head rather than about the box. **Cross-contention comparisons without the
  anchor are reported as caveated; comparisons through the anchor are not.**
* **Shard counts are unequal** (denseaux 5, strata 2, anchor 1) because the heads' per-battle cost
  differs ~3× — strata forces 96% of decisions and therefore searches almost nothing. Every shard
  is `nice 15`, below the live training run's `nice 10`, which outranks this measurement by design.

### 7b. STOPPING RULE — registered 2026-09-12 01:57 UTC, blind to every outcome

The box is carrying a live training run and a stepcurve campaign; each of this battery's shards is
getting **~50–70% of one core**, so the registered 800 pairs/head may not complete in a reasonable
session. Registered now, **before any win rate from these cells has been looked at**:

* **The cells run until 07:30 UTC.** Whatever swap-pairs are complete at that instant constitute the
  cell, and the achieved pair count is published beside every number.
* **The stop is a WALL CLOCK and nothing else.** No cell is stopped early, extended, or resized on
  the basis of any outcome; the rule is outcome-blind by construction, which is what keeps it from
  being optional stopping.
* **The extension rule of §6 does NOT apply to these cells** — it was a rule for a completed
  pre-registered n, and these cells may not reach one. A cell that stops short is reported at its
  own width with its own CI, and "unresolved" is a legitimate result.
* Shard counts were rebalanced at 01:56 UTC (strata 2 → 4 shards over disjoint 200-game windows,
  resuming the rows already written) purely for throughput. No flag, seed, index or head changed.

---

## 7c. ADDITION — a third head: `cflabels`, and the L1/L2 bar this is now read against (registered 2026-09-12 ~04:25 UTC, before its first cell)

Ordered by the coordinator while the strata/denseaux cells were mid-flight. **Registered before any
win rate from ANY phase-2 cell has been looked at** — up to this moment only row counts and realized
widths have been read, deliberately.

### The head

`ai_v12_12_ladder_cflabels` @10000032 — `--cf-records --cf-winprob-coef 0.5`, pinned `f3502568`
(the SAME pin as the `ctrl10M` anchor, so the anchor is a same-commit control). Null on the ladder's
conditioning row, **but it is the closest thing already trained to a SUCCESSOR-DISCRIMINATION
objective**: its auxiliary labels are the outcomes of the move taken versus an alternative from the
same state — which is, structurally, the quantity a one-ply leaf is asked to rank.

### The bar it is read against (ledger 2026-09-11, "THE LEAF-QUALITY METER")

* **L1 = separation-of-raced** — the mechanism row. Bar: **> 18.1 %** (the win-prob family's maximum
  15.0 % plus the 3.1 pp within-family floor).
* **L2 = the paired mirror win rate** — the outcome row, null 0.50 STRUCTURAL. Bar: **paired CI
  lower bound > 0.50**.
* A lever clearing L1 but not L2 reads *"mechanism moved, outcome not yet"* — reported, not a pass.
* **The ≤ ~20 % provisional rule (ledger addendum, same day):** the L1 floor is the RANGE of three
  points, so it is biased small and the 18.1 % bar is probably permissive. **An L1 clear at ≤ ~20 %
  is PROVISIONAL**, pending a fourth win-prob head — which this battery itself supplies; **a clear
  at ≳ 25 % is not provisional.** The floor demotes, never promotes.

### What each outcome means — named in advance

* **cflabels pays where strata and denseaux did not** ⇒ **a within-game LABEL objective moves leaf
  quality, and the axes separate**: a lever that is null on the conditioning row moves the leaf, so
  the ladder's conditioning row is not the meter the search path should be steered by. It would also
  name the *kind* of target that works — successor discrimination, not calibration and not class
  balance — which is the most actionable result available from the five-head set.
* **cflabels null too** ⇒ **counterfactual auxiliaries AT THIS COEFFICIENT (0.5, `cf_head_only`,
  a 150k-step label lag) do not move the leaf either.** Combined with the other two, no lever in the
  trained ladder moves L1/L2, and the constraint is the objective's SHARE/FORM rather than the
  presence of a counterfactual signal. That is a dose-limited null, not a refutation of successor
  discrimination — stated now so it cannot be over-read later.

### Registered predictions

1. **cflabels has the highest L1 of the three new heads** — it is the only one whose auxiliary is a
   successor contrast. Confidence ~45% that it is highest of the three; **~30% that it clears
   18.1%**; ~12% that it clears it decisively (≳ 25%).
2. **cflabels does not clear L2.** Confidence ~85%, for the same reason as the other heads: `grid`
   is catastrophic on every win-prob head measured, and the defensive gate buys that back to the
   null rather than past it.
3. **Its `grid` cell lands in the 0.20–0.32 band** with the other four — i.e. no head yet measured
   makes unguarded search survivable.

### Dose caveat, registered UNREAD

A sibling run of this arm was retired as `…DEAD_fatal_config_cf_duty_cycle_6pct`, i.e. a cf duty
cycle of ~6% was once a fatal config. **The live arm's realized cf duty cycle has NOT been read**
and is not asserted here; if this head reads null on L1, the null is reported as *dose-unread*
rather than as a verdict on counterfactual labels.

### 7d. DEADLINE AMENDMENT (registered here, still outcome-blind)

§7b set an outcome-blind 07:30 UTC stop. A third head arrived after that rule, so **the deadline for
every phase-2/3 cell moves to 09:00 UTC**, fixed now and still blind to every win rate. The stop
remains a wall clock and nothing else; a cell that stops short is reported at its achieved pair count
with its own CI, and "unresolved" stays a legitimate result. `cflabels` gets its OWN contemporaneous
`ctrl10M` anchor window (indices 150–299), because its cell runs in a later contention regime than
the strata/denseaux cells and the anchor is what makes the cross-head L1 comparison a statement about
heads rather than about the box.
