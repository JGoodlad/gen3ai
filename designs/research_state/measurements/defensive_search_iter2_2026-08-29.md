# DEFENSIVE PAIRED SEARCH — iteration 2: SPEND THE BANK

*Measured 2026-08-29, 09:48–13:28 UTC · 1600 orientation-games / 800 swap-pairs / 68,585
decisions · CPU-only, 3 shards at `nice 15` (~3 cores), BLAS pinned · ~3.7 h real elapsed
(10.86 h summed battle wall) · `models/` read-only · zero errors, zero timeouts, zero unfinished.*

Registered in the ledger entry landed at `4cf81fd` ("ITERATION 2 DISPATCHED … SPEND THE BANK").
Data: [`defensive_search_iter2_2026-08-29.json`](defensive_search_iter2_2026-08-29.json); rows
archived beside it (`defensive_search_iter2_2026-08-29_rows.jsonl.gz`); scoring script
`defensive_search_iter2_report.py`; the flag landed as `934fb20`
(`--defensive-contested-deadline-s`, plus the futility-split counters and `--games-start`
sharding).

---

## 1. Verdict

**The MECHANISM did exactly what the diagnosis said it would, and the DIVIDEND did not appear.
No-regression (the primary bar) HELD. The stretch is REFUTED at resolving width.** The race,
freed from its floor, separates at probe I's ceiling and overrules three times as often — and
the win rate does not move by a millimetre. The missing dividend was never budget-limited after
all: it is the *leaf*.

| registered prediction (ledger `4cf81fd`, scored, never adjusted) | outcome |
|---|---|
| separated-of-raced rises from 0.157 toward probe I's ~0.48 ceiling | **HELD, emphatically: 0.4542** — at 95% of the ceiling (0.478). The first cell's "budget-limited at the floor" diagnosis was correct and is now closed: rounds per race went **4.61 → 13.17** against the unchanged floor of 5. |
| overrules rise 1.8% → 6–12% of all decisions | **MARGINALLY MISSED LOW: 5.82%** (3,531 of 60,662) — 0.2 pp under the registered band's lower edge. Directionally exactly as predicted (a 3.2× rise); the honest score is *not in range*. Of raced decisions: 6.9% → **23.4%**. |
| win rate ≥ iteration 1 (NO REGRESSION is the primary bar) | **HELD.** 0.5003 vs iteration 1's 0.4938 at full n; on the 200 shared game indices the paired difference is **−0.0037 [−0.0505, +0.0430]** — zero, at a width that would have caught a ±4 pp change. Tripling the overrule volume cost nothing. |
| STRETCH: paired CI above 0.50 (resolves only if true rate ≥ ~0.525; a 0.51-ish point is "real but unresolved", pre-stated) | **REFUTED, and not in the pre-stated grey zone.** Paired 0.5003 [0.4803, 0.5203] at ±0.020 — a true dividend of even +2 pp would have cleared. The point estimate is the null itself. |

**The one-line finding: 3,531 overrules — thirteen times iteration 1's 270 — moved the win rate
from 0.4938 to 0.5003, i.e. onto the null exactly. The separated overrules are win-rate
NEUTRAL.** When the race, with 13 rounds of CRN-paired evidence and the `seq` rule's anytime
guarantee, certifies that a different action scores higher on the one-ply win-prob leaf than the
policy's own choice, playing that action neither wins nor loses games. The evidence problem this
iteration was designed to remove is removed; what remains is that the thing being measured
precisely does not predict game outcomes better than the policy already does. That is a verdict
about the **depth-1 win-prob leaf as an overrule criterion**, not about the racer, the gate, or
the clock — all three did their jobs to specification.

---

## 2. The cell

**Checkpoint** `models/ai_v9_29_rev1_0823/checkpoints/checkpoint_9995088_steps.zip` — the same
checkpoint as iteration 1 and every historical mirror arm, at the same `--games-seed 7`.
**The first 400 orientation-games (indices 0–199) are seed-identical to iteration 1's entire
cell**, which is what licenses the paired no-regression row in §3; indices 200–799 extend the
range rather than re-using indices. Everything except the one flag is iteration 1's invocation
verbatim:

```
python -m main.search_dividend <ckpt> --arm honest --budget 1 --root-strategy defensive \
  --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0 \
  --defensive-contested-deadline-s 3.0 \
  --games-start <lo> --games <n> --games-seed 7 --opponents self \
  --battle-timeout-s 1800 --battle-idle-s 120
```

**THE ONE CHANGE:** a decision that PASSES the triage gate is granted **3 s** instead of the
uniform `--budget` of 1 s. Gate threshold (0.15), `seq` rule, elimination floor (5), win-prob
leaf, depth 1, confirm off, the racer's 64-round supply: all unchanged. No max-rounds bump was
needed — the supply was 64 all along and the mean race reaches only ~13 rounds at 3 s, so the
clock still governs.

**Sharding.** Three processes over disjoint half-open game-index windows ([0,267), [267,534),
[534,800)) via the new `--games-start`. The seed and the team draw are functions of the index
alone (asserted by the scoring script, which refuses overlapping windows), so the shards' rows
concatenate into exactly the file one process would have written. ~3 cores at `nice 15` beside
the box's live GPU work; load stayed ~3 on 16 cores throughout, so the contention factor was 1.0
and the 3 s wall-clock deadline was undistorted.

**Timeout hygiene.** Zero timeouts, zero errors, 1600/1600 finished. Longest game **474 s**
against the 1800 s backstop (one 250-turn cap game); the poison defect had no opportunity to
fire.

---

## 3. The headline, and the two paired comparisons

| arm | budget | n (decisive) | win rate | 95% CI | paired | paired CI | pairs |
|---|---|---|---|---|---|---|---|
| `mirror_honest_1s` — the historical bar | 1 s | 239 | 0.2929 | [0.239, 0.354] | 0.2938 | [0.235, 0.352] | 120 |
| `mirror_honest_3s` — a UNIFORM 3 s | 3 s | 78 | 0.2692 | [0.183, 0.377] | 0.2756 | [0.183, 0.369] | 39 |
| `defensive_1s` — iteration 1 | 1 s | 397 | 0.4937 | [0.445, 0.543] | 0.4938 | [0.459, 0.529] | 200 |
| **`defensive_1s_contested3s` — THIS CELL** | 1 s + 3 s contested | **1591** | **0.5003** | **[0.4758, 0.5249]** | **0.5003** | **[0.4803, 0.5203]** | **800** |

(Historical rows quoted, never re-run. 9 ties, excluded from the denominator. Note
`mirror_honest_3s`: a *uniform* 3 s makes plain search WORSE, so the contested deadline is not
"more budget" generically — it is more budget behind the refusals.)

| comparison | Δ | 95% CI | excludes 0 |
|---|---|---|---|
| vs `honest_1s`, unpaired (Newcombe) | +0.2074 | [+0.1420, +0.2668] | yes |
| vs `honest_1s`, paired on the 120 shared game indices | +0.1938 | [+0.1063, +0.2812] | yes |
| vs **iteration 1**, unpaired | +0.0066 | [−0.0482, +0.0613] | no |
| vs **iteration 1**, **paired on the 200 shared game indices** — THE REGISTERED ROW | **−0.0037** | **[−0.0505, +0.0430]** | **no** |

The iteration-1 paired row is the scored one: the two cells differ in exactly one flag, and on
the shared indices they are the same pinned dice and the same team draw. It reads **zero**, at a
width that would have detected ±4 pp.

---

## 4. The rate table — the mechanism moved, the outcome did not

Over the **60,662** decisions the strategy handled (of the game's 68,585; the rest are forced
switches and counted search failures — §5):

| branch | iteration 1 | **iteration 2** | of raced (it1 → it2) |
|---|---|---|---|
| FORCED by the gate | 0.7385 | **0.7515** | — |
| RACED | 0.2615 | **0.2485** | — |
| …SEPARATED | 0.0410 | **0.1129** | 0.157 → **0.4542** |
| …KEPT (separated on the policy's own action) | 0.0230 | **0.0547** | 0.088 → **0.2200** |
| …OVERRULED | 0.0180 | **0.0582** | 0.069 → **0.2342** |
| …FUTILITY | 0.2204 | **0.1357** | 0.843 → **0.5458** |
| rounds per race | 4.61 | **13.17** | floor 5, supply 64 |
| eliminated per race | 2.53 | **5.35** | |
| mean search s per raced decision | 0.897 (of 1) | **2.278 (of 3)** | |

Three readings:

1. **The floor diagnosis is closed.** At 13.2 rounds the race is well past the floor at which
   elimination becomes legal, and separation lands at **0.454 — 95% of probe I's 0.478
   offline ceiling**. The first cell's futility rate was indeed this budget's, not the game's.
2. **The KEPT column is the quiet null replica.** Among separated races, 48.4% separate ON the
   policy's own action (3,316 kept vs 3,531 overruled) — the race, given evidence, certifies the
   policy's choice about as often as it contradicts it. Combined with the overruled half being
   win-rate neutral (§1), the leaf's separations carry approximately zero information the policy
   does not already act on.
3. **The futility split the new counter was built for reads 99.93% deadline-truncated** (8,223
   of 8,229; 6 genuine). This is NOT iteration 1's exact identity returning: the mean race now
   runs 13 rounds before the clock ends it, so these are races that had real evidence and did
   not separate on it. But the "genuine non-separation" counter can only certify a race that
   exhausts the 64-round supply, which 3 s does not reach — so the honest statement is that the
   futility mass is *still clock-ended*, now after ~13 rounds rather than ~4.6, and probe I's
   U-shape (52% never separate at 32 samples) says most of it would not separate at any clock.

### The realized envelope — the registered claim, checked

| | registered estimate | measured |
|---|---|---|
| contested decisions per game | ~11 | **9.42** |
| search s per game | ~33 | **21.46** |
| uniform-1s notional per game | ~42 | **37.91** |
| inside the envelope | claimed | **yes — 43.4% of the uniform notional still banked** (iteration 1: 76.6%) |

Mean battle wall 24.44 s/game (iteration 1: 10.73). The raw `defensive_banked_s` counter now
mixes scales (a forced decision banks the 1 s notional, a raced one banks its own 3 s deadline's
residual), so the quotable bank figure is the spend-derived `banked_frac_of_uniform` above — the
scoring script carries the caveat.

---

## 5. Accounting

Of the 68,585 decisions: 60,662 reached the defensive rule (88.4% — identical to iteration 1),
6,812 `not_move_selection` (9.9%), and **1,110 search failures (1.6%)** — `root_failed` 473,
`prefix_gate_failed` 556, `search_error` 81, `no_scored_arm` 1 — concentrated in 81 of 1600
games, the same prefix-replay-mismatch family iteration 1 and the playoff read diagnosed, all
falling back to the policy's action (bias toward the null, immaterial at 1.6%).

**Nothing was cut.** The full registered n = 1600 orientation-games (800 pairs, paired CI
±0.020 vs the targeted ±0.025) played in ~3.7 h real time against the ≤7 h budget, on 3 niced
cores against the ≤3-core budget. Leaf verified in the artifact: every row `score_mode:
"win_prob"`, `n_defensive_no_win_prob` = 0 across all 1600 games.

**Tests.** The flag's default preserves iteration-1 behaviour byte-identically
(`test_an_unset_contested_deadline_is_the_first_cells_behaviour_exactly`, parametrized over
every battery budget); the 3 s deadline provably reaches the racer's clock AND the width
allocator through one shared `contested_budget_s()`
(`test_the_contested_deadline_actually_reaches_the_racer`,
`test_the_width_plan_is_sized_to_the_SAME_clock_the_deadline_enforces`); a forced decision still
banks the uniform notional; the futility split folds and reports; `--games-start` sharding is
pinned to the seed-is-a-function-of-the-index property. Suite: **359 passed** in
`src/main/search_dividend` (was 339), **2,402 passed** across `src/main`, three static gates
(mypy, ruff, file-size) green.

---

## 6. Caveats

1. **The stretch refutation is now at resolving width and should be treated as a result.** ±0.020
   at 800 pairs; the point estimate is 0.5003. Two independent cells (n=200 and n=800 pairs) have
   now put this composite exactly on the null. "Search stops losing" is established; "search
   pays" is, at this leaf and depth, falsified rather than under-powered.
2. **The overrule-rate prediction is scored as missed (5.82% vs 6–12%)** even though the miss is
   0.2 pp — registered numbers are registered numbers. The structural reason it runs under the
   registration's arithmetic (H's contested × I's separable ≈ 0.25 × 0.45 ≈ 11%) is the KEPT
   column: nearly half of separations land on the policy's own action, which the registration's
   product did not model.
3. **The futility-genuine counter is effectively unpopulated at this clock** (6 events). To
   measure the game's true never-separate mass live, the supply (64 rounds), not the deadline,
   must be the binding constraint — that is a different, longer cell and probe I already priced
   the answer offline.
4. **One checkpoint, one budget pair, mirror only, depth 1.** All of iteration 1's §6 caveats
   carry over unchanged, including that the mirror carries no ELO anchor and that depth ≥ 2
   breaks the pairing argument the leaf depends on.
5. **The wall cost tripled** (10.73 → 24.44 s/game summed wall) for zero dividend. As a *ladder*
   configuration the contested deadline buys nothing and costs clock; iteration 1's 1 s uniform
   configuration remains the deployable operating point.

---

## 7. What this changes

**The "spend the bank" arm has done its job by failing cleanly.** Iteration 1 ended with two
candidate next moves — spend the banked clock, or lower the floor — both premised on the race
being evidence-starved. This cell removed the starvation (separation at the offline ceiling,
13 rounds per race) and the dividend did not appear, which retires *both* moves: more rounds and
a lower floor are the same lever, and the lever is now measured at zero. The refusal
architecture (gate + futility stop) remains the whole of the value — it is what holds the arm at
the null while every non-refusing arm in the battery sits 5–21 points below it.

**The next question is the LEAF, not the allocator.** The depth-1 win-prob read, even
CRN-paired and separation-certified, does not rank sibling actions better than the policy's own
choice where it matters for outcomes. The instruments that could move that are different in kind:
a better one-ply value (probe G's own headline was only +0.022 win-prob against the played
action — consistent with what this cell just measured at game granularity), terminal rollouts as
the overrule criterion (`--defensive-confirm`, built and off — but the playoff arm already
showed rollouts at deployable budgets are inconclusive-dominated), or depth — which the pairing
argument says requires solving the per-decision offset first. None of these is a flag on the
current racer; the racer is finished business.
