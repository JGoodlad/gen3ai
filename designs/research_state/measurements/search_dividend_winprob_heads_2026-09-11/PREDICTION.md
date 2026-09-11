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
