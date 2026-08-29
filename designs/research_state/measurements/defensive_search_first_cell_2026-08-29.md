# DEFENSIVE PAIRED SEARCH — the first mirror cell

*Measured 2026-08-29, 01:00–02:15 PDT · 400 orientation-games / 200 swap-pairs / 16,942 decisions ·
CPU-only, one core, `nice 15`, BLAS pinned · 1.19 h wall · `models/` read-only.*

Registered in the ledger's synthesis entry (`5f98d26`). Data:
[`defensive_search_first_cell_2026-08-29.json`](defensive_search_first_cell_2026-08-29.json);
rendering script beside it; code `src/main/search_dividend/defensive.py`
(`--root-strategy defensive`).

---

## 1. Verdict

**PRIMARY MET, decisively. STRETCH NOT MET. The overrule-rate prediction is REFUTED — by a factor
of five, and the reason is a mechanism the registration did not anticipate.**

| registered bar | outcome |
|---|---|
| "DEFENSIVE decisively above the historical `honest_1s` mirror arm (0.292) with its CI reaching 0.50 — *search stops losing*" | **MET.** 0.4937 [0.4448, 0.5427] unpaired, **0.4938 [0.4586, 0.5289] paired**. Against `honest_1s` the difference is **+0.2008 [+0.1229, +0.2738]** unpaired and **+0.2167 [+0.1393, +0.2940]** paired on the 120 game indices both arms played. The intervals reach 0.50 and exclude it from below. |
| "CI above 0.50 — *search finally pays*" | **NOT MET, and not close.** The point estimate is 0.4937 — a hair *below* the null, not above it. This is not an n problem: the paired interval is ±0.035 wide at 200 pairs, so a real dividend of even +3 pp would have shown. **The honest reading is that search has stopped costing anything, not that it has started paying.** |
| prediction: overrule rate ≈ 8–17% of decisions | **REFUTED. 1.80%** of the decisions the strategy handled (270 of 14,974), 6.90% of the RACED ones. §4 diagnoses why, and the cause is budget, not design. |

**The one-line finding: the whole 20-point deficit that plain search carried was overrule damage,
and refusing to overrule recovers all of it and nothing more.** The mirror's no-effect point is
0.50 by construction, so an arm that lands on it is an arm that has stopped doing harm. Every
component of this strategy is a refusal, and the sum of the refusals is exactly the null.

---

## 2. The cell

**Checkpoint** `models/ai_v9_29_rev1_0823/checkpoints/checkpoint_9995088_steps.zip` — the **same**
checkpoint the historical mirror arms played, at the **same** `--games-seed 7`. Game *g* is
therefore the same pinned dice and the same team draw in both cells, verified on the seed field
(120 of 200 game indices overlap `honest_1s`'s 120). That is what licenses the paired comparison
in §3 rather than a matched-in-spirit one.

```
python -m main.search_dividend <ckpt> --arm honest --budget 1 --root-strategy defensive \
  --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0 \
  --games 200 --games-seed 7 --opponents self --battle-timeout-s 1800 --battle-idle-s 120
```

`--opponents self` is the MIRROR: the searched side against the SAME network with search
structurally off, side-swapped by default so the team draw differences out. **The null is 0.50 by
construction, not by estimate.** The confirm stage was OFF, as registered — one new mechanism at a
time.

**Timeout hygiene.** **Zero timeouts, zero errors, zero unfinished games** (400/400 finished). The
backstops were raised pre-emptively per the playoff lesson (a timed-out game does not cost one row,
it poisons the rest of the cell by leaving an uncancellable search issuing `open_root` for a dead
battle); the longest game was **84 s** against the 1800 s backstop and the longest decision 1 s
against the 120 s idle bound, so the known poison defect had no opportunity to fire.

---

## 3. The headline, and what it is measured against

| arm | budget | n (decisive) | win rate | 95% CI | paired | paired CI | pairs |
|---|---|---|---|---|---|---|---|
| `mirror_base` — the policy alone | — | 60 | 0.5000 | [0.377, 0.623] | 0.500 | [0.500, 0.500] | 30 |
| `mirror_honest_1s` — **THE BAR** | 1 s | 239 | **0.2929** | [0.239, 0.354] | 0.2938 | [0.235, 0.352] | 120 |
| `mirror_honest_3s` | 3 s | 78 | 0.2692 | [0.183, 0.377] | 0.2756 | [0.183, 0.369] | 39 |
| `mirror_oracle_1s` | 1 s | 118 | 0.3220 | [0.245, 0.411] | 0.3250 | [0.235, 0.415] | 60 |
| `mirror_oracle_3s` | 3 s | 78 | 0.4359 | [0.331, 0.546] | 0.4342 | [0.322, 0.546] | 38 |
| `playoff_10s` — the best prior arm | 20 s | 80 | 0.4500 | [0.346, 0.559] | 0.4500 | [0.373, 0.527] | 40 |
| **`defensive_1s` — THIS CELL** | **1 s** | **397** | **0.4937** | **[0.4448, 0.5427]** | **0.4938** | **[0.4586, 0.5289]** | **200** |

Every historical row is **quoted, not re-run**, from the files the battery played them into:
`tmp/search_dividend/{mirror_base,mirror_honest_1s,mirror_honest_3s,mirror_oracle_1s,mirror_oracle_3s,playoff_10s}.jsonl`
in the main checkout (the same files probe H aggregated). Ties are excluded from every denominator
and reported separately; this cell had 3.

**Against the bar, two ways:**

| comparison | Δ | 95% CI | excludes 0 |
|---|---|---|---|
| vs `honest_1s`, unpaired (Newcombe) | **+0.2008** | [+0.1229, +0.2738] | yes |
| vs `honest_1s`, **paired on the 120 shared game indices** | **+0.2167** | [+0.1393, +0.2940] | yes |

The paired row is the one to quote. A mirror game's team draw is asymmetric and at these n it is
most of the variance — the exploiter work already measured an apparent "edge" that was entirely
team draw — so differencing on the game index is what makes +0.217 a statement about the decision
rule rather than about which side drew the better six.

**And against the best prior arm at 1/20th the budget.** `playoff_10s` reached 0.450 [0.346,
0.559] by spending **20 s per decision** and nesting whole terminal rollouts inside a live turn;
this reaches 0.494 [0.445, 0.543] at **1 s**, with a mean of 0.90 s on the 23% of decisions it
searched at all. That is not a claim that defensive beats playoff — the intervals overlap heavily
and the two were never played head to head — but the deployability difference is real: 20 s/turn
out-accrues the 150 s ladder timer and 1 s does not.

---

## 4. The rate table — and the mechanism the prediction missed

Over the **14,974** decisions the strategy handled (`forced + raced` accounts for all of them;
the other 1,968 of the game's 16,942 never reached it — see §5):

| branch | n | rate | of raced |
|---|---|---|---|
| **FORCED** by the gate | 11,059 | **0.7385** | — |
| …because `\|P(win)−0.5\| ≥ 0.15` | 11,059 | 0.7385 | — |
| …because `n_legal ≤ 1` | 0 | 0.0000 | — |
| **RACED** | 3,915 | **0.2615** | — |
| …**FUTILITY** (never separated → keep the policy action) | 3,301 | 0.2204 | **0.8432** |
| …**KEPT** (separated *on* the policy's own action) | 344 | 0.0230 | 0.0879 |
| …**OVERRULED** (separated on a different action → played it) | **270** | **0.0180** | **0.0690** |
| no win-prob head | 0 | 0.0000 | — |

**The gate reproduced probe H's rule but not its coverage: 73.9% forced here against H's 82.5%.**
That is expected and it is a population difference, not a disagreement — H measured on eval traces
against a bot roster, where `|P(win)−0.5|` is large far more often than it is in a mirror against
an equally strong copy of yourself. H flagged exactly this ("the *coverage* of any threshold is
optimistic; the *ordering* of features should transfer"), and the ordering is what was inherited.
The gate's `n_legal` clause fired zero times, because the live player already declines a
single-token root as `not_move_selection` before the engine sees it — the clause is a correctness
guard, not a live contributor.

### Why the overrule rate is 1.8% and not 8–17%

The registered 8–17% was built as *H's contested fraction × I's separable fraction* (≈0.26 × 0.48).
The contested fraction came in at 0.26, dead on. **The separable fraction did not: 15.7% here
against probe I's 47.8%.** The race counters say why, and the number is unambiguous:

| | this cell | probe I's bank |
|---|---|---|
| rounds per race | **4.61** | 32 (fixed supply) |
| `seq` rule's elimination FLOOR | **5** | 5 |
| decisions whose race hit the deadline | 3,301 = **100% of the futility mass** | n/a (offline) |
| separated | 0.157 | 0.478 |

**The mean race is 4.6 rounds long and no elimination is legal before round 5.** Every one of the
3,301 futility stops is also a `deadline_truncated` — an exact identity, not a correlation. So the
majority of this cell's futility mass is not the game's U-shape refusing to separate; it is a race
that **was never allowed to try**. Probe I's 52.2% never-separate rate was measured with 32 paired
samples available; at a 1 s budget the clock buys about a seventh of that.

This does not weaken the verdict — a race that cannot separate correctly declines to overrule, and
declining is what produced the 20-point recovery. But it does relocate the next question. **The
strategy is currently budget-limited at the floor, not evidence-limited**, and the futility rate
is therefore an upper bound on the true one rather than a measurement of it.

Two secondary readings from the same counters, both consistent with the racer working as designed:
**2.53 actions eliminated per race** (so elimination *is* firing on the races that reach the floor)
and **6,973 arm evaluations saved** against what a uniform grid over the same rounds would have
spent.

### Banked time

| | |
|---|---|
| total clock handed back | **11,536.9 s** |
| per decision | **0.77 s** of the 1 s notional budget — **77.1% unspent** |
| per game | **28.8 s** |
| mean search wall on a RACED decision | 0.897 s (of 1 s) |

Read this against probe H's budget model: at 150 s/game and ~42 decisions, a uniform search affords
2.84 s/decision. This strategy spends its second on 26% of decisions and hands back 77% of the
notional total — i.e. the same games at **5.4 s per contested decision** if the banked clock were
redistributed, or the current cell run at roughly a quarter of the wall. §4 says the marginal
second would go to raising the race past its floor.

---

## 5. Accounting — every decision, and what was cut

Of the **16,942** decisions in the 400 games:

| | n | share |
|---|---|---|
| reached the defensive rule | 14,974 | 88.4% |
| `not_move_selection` (forced switches / no branchable root) | 1,711 | 10.1% |
| **search failures** (counted fallbacks, all → the policy's action) | **257** | **1.5%** |
| — `root_failed` | 148 | 0.9% |
| — `prefix_gate_failed` | 84 | 0.5% |
| — `search_error` | 25 | 0.1% |

**The 257 search failures are CONCENTRATED, not scattered: 23 of 400 games carry all of them**
(worst game 30 decisions). The captured `fallback_details` name one cause — `prefix replay
produced N decisions for a branch at index N — prefix_chunks and prefix_actions disagree` — which
is the same family the playoff formal read diagnosed (its "prefix replay decision-count mismatch",
4 of 17 captured texts there). Every one falls back to the policy's action, so the failure mode
biases this cell **toward the 0.50 null**; at 1.5% of decisions it cannot move the reading
materially in either direction, and it is not a new defect.

**Nothing was cut for time.** The cell played its full registered allocation (200 game indices ×
2 orientations = 400 orientation-games) in **1.19 h** against a 5 h budget, on one core at
`nice 15` beside the box's live GPU work. `--defensive-confirm` was 0 throughout, as registered.

**Pipeline validation.** The leaf is verified in the artifact, not assumed: every row records
`score_mode: "win_prob"`, and `n_defensive_no_win_prob` is **0** across all 400 games — the search
ranked on the head probe G measured, on every decision, with `check_leaf` standing by to raise if
it ever had not.

---

## 6. Caveats

1. **The stretch bar's failure is a result, not a power problem.** At 200 swap-pairs the paired
   interval is ±0.035. A dividend of +5 pp would have cleared 0.50 comfortably; the point estimate
   is 0.4937. Read this as "search stopped losing", never as "search is about to pay".
2. **Budget-limited at the floor (§4).** The mean race is 4.6 rounds against an elimination floor
   of 5, so the separation rate here is not the game's separability — it is this budget's. Every
   number in the futility column would move under a larger budget or a lower floor, and the
   overrule rate would move with them.
3. **The gate's coverage does not transfer off the mirror.** 73.9% forced here vs H's 82.5% on
   bot-roster traces. Against a weaker opponent `|P(win)−0.5|` is large more often, so a ladder
   cell would force more and race less.
4. **One checkpoint, one budget, one opponent surface.** `ai_v9_29_rev1_0823` at 9.995 M steps, 1 s
   per decision, mirror only. Nothing here claims the composite transfers to another generation, and
   probe I's standing caveat applies unchanged: the grid at this same 1 s budget agrees with its own
   large-budget argmax on only 86.1% of decisions.
5. **A mirror carries no ELO anchor.** These are direct win rates against a constructed 0.50 null
   and are never read through the anchored-ELO fit.
6. **Depth 1 only, and that is load-bearing rather than incidental.** Probe G's pairing argument
   cancels a per-DECISION offset between SIBLINGS; at depth ≥2 the tree compares nodes at different
   decisions where that offset (RMS 0.200, 72.8% of the critic's leaf error) does not cancel at all
   and becomes the dominant term. A deeper defensive search is a different experiment, not a bigger
   one.
7. **`playoff_10s` and this cell were never played head to head**, and their intervals overlap.
   The comparison in §3 is between two separately-measured arms against a common construction, not
   a paired contest.

---

## 7. What this changes

**The composite works as a damage-control instrument and it is now the battery's best-read arm at
a deployable budget.** Search has gone from −20.7 pp (honest_1s) to −0.6 pp against the mirror null
at the same checkpoint, the same battles and the same 1 s clock, by doing nothing except refusing:
refusing to search a decided position (73.9% of decisions), refusing to overrule without separation
(84.3% of the races it did run), and refusing to score on the leaf that does not beat the played
action.

**The next question is no longer "does overruling hurt" — it is "can the race be given enough
evidence to overrule at all".** The mechanism in §4 is specific and cheap to act on: the race dies
0.4 rounds short of the floor at which its first elimination becomes legal. Two obvious arms, both
one flag:

* **spend the banked clock** — 77% of the notional budget is unspent, and the whole of it is
  currently unreachable because the gate hands it back before a `Deadline` exists. A time manager
  that redistributed it would put ~5.4 s on each contested decision, i.e. ~5× the rounds.
* **or lower the floor and pay the measured false-drop rate** — probe I priced this exactly
  (floor 3 = 8.0% false drops, floor 4 = 3.0%, floor 5 = 0.83%) and found the power to separate a
  real gap is 1.000 at every one of them.

Which of those is right is a measurement, and it is the one this cell has earned the right to make:
the strategy's refusals are no longer the thing under suspicion.
