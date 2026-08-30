# Defensive paired search, iteration 3 — rollout-confirmed overrules (`--defensive-confirm 6`)

**Date** 2026-08-29 · **1,275 games / 637 complete side-swap pairs / 55,958 decisions** · mirror
(`--opponents self`, `--games-seed 7`) · 3 shards over disjoint `--games-start` windows
([0,267), [267,534), [534,800)), CPU-only at `nice 15`, pre-registered wall-clock cut enforced by
explicit-PID `cut.sh` at 21:24Z (all three shards exited on SIGTERM, zero errors, zero unfinished
rows). Rows: `defensive_search_iter3_2026-08-29_rows.jsonl.gz` · summary: sibling `.json`.

**Scored by the parent session** after the shard-completion watcher failed to fire (the agent's
last wake was 14:08 PDT; the cut landed 14:24 PDT as scheduled and the data sat finished for ~7 h).
The analysis spec below is the agent's own pre-cut draft, verbatim in intent.

## 1. The one change vs iteration 2

Iteration 2's invocation verbatim (same checkpoint `ai_v9_29_rev1_0823 @ 9,995,088`, same seed,
same gate 0.15 / floor 5 / win-prob leaf / depth 1 / contested deadline 3 s), plus:

```
--defensive-confirm 6 --defensive-confirm-deadline-s 30
```

Before acting on an overrule the race has certified, the race's winner and the policy's own action
are settled by up to 6 paired rollouts to a terminal; the policy stands unless the paired
difference clears 2·SE over ≥ 4 pairs. Three latent defects in the built-but-never-exercised
confirm were found and fixed before the cell ran (no reachable clock → 100% spurious rejection;
rollout wall billed to the search by subtraction; a rejection countable as a confirmation) — each
would have produced a clean-looking null. The verdict-keyed counters that fix #3 installed are
what the table below is read from.

## 2. Results against the registered predictions

| registered | outcome |
|---|---|
| PROPOSED overrule rate continues iteration 2's 5.82% (the race-unchanged check) | **MISSED: 3.98% [3.82, 4.14]** — but see §4: realized width collapsed under box contention (k_worlds 9.28 vs 14.0, separated-of-raced 0.355 vs 0.454), which degrades separation without any code change. Not scoreable as a race change. |
| ACTED overrule rate lands in 1.5–3.5% | **MISSED LOW: 0.098% [0.08, 0.13]** (55 of 55,958). The band assumed the confirm would mostly *resolve*; instead 91.2% of confirms are inconclusive at r ≤ 6. |
| NO REGRESSION vs iteration 2 (primary bar) | **HELD.** Paired 0.4976 [0.4702, 0.5251] vs iteration 2's 0.5003 [0.4803, 0.5203]. The mirror sits on its null. |

## 3. The finding — terminal adjudication says certified overrules are coin flips

Of **2,226** confirm attempts (race separated on a non-policy action):

| verdict | n | % of attempts |
|---|---|---|
| inconclusive (policy kept) | 2,031 | **91.2%** |
| confirmed (overrule acted) | 55 | 2.5% |
| reversed (rollouts preferred the policy) | 52 | 2.3% |
| error (0 pairs completed; driver, not leaf) | 88 | 4.0% |
| no_budget | 0 | 0% |

Two numbers carry the verdict:

1. **The confirm is underpowered by an order of magnitude, structurally.** Median SE at stop is
   0.167 → a 2·SE bar of 0.333 on a paired-outcome mean, against a typical |paired mean| of
   **0.097** among inconclusives. Resolving the typical effect needs ~12× the pairs (~70, ≈180 s
   per confirm at the measured 2.57 s/pair) — infeasible live, at any budget this program would
   call live. "Inconclusive" is a power statement, not evidence of equality.
2. **When the rollouts DO separate (107 events), the verdict splits 55 confirmed vs 52
   reversed** — the depth-1 win-prob leaf's certified overrule is upheld at terminal exactly as
   often as it is refuted. Confirmed events show mean paired diff +0.585, reversed −0.587 —
   symmetric, large, and sign-balanced. The leaf-margin gradient is weakly directional
   (margin ≥ 0.10: 21–11; margin < 0.02: 3–9) but at n too small to act on.

Combined with iteration 2 (13× overrule volume moved the win rate onto the null exactly) and the
playoff screen's leaf-BIAS verdict, this closes the confirm rung of the R-ladder: **at depth 1,
overrules certified by the shaped v9 win-prob leaf carry no terminal-adjudicable content.** The
gate + futility architecture (holding the arm at 0.50 for ~34 s of banked clock per game) remains
the whole of the demonstrated value of the defensive apparatus.

**Bearing on ai_v12 route 3 (search-overrule filtering):** direct adverse evidence, with the
scope caveat that every cell so far used the v9 *shaped* head in the shaped-reward world. Route
3's experiment survives only on the hypothesis that a clean-world ±1-terminal Q head produces
leaf differences that are real where these were noise — the prior is now low, and the ai_v12
cell is exactly the test of that hypothesis.

## 4. Caveats

- **The iteration-2 width comparison is load-confounded.** The cell ran 06:57–14:24 PDT beside
  probe P's 352-shard eval fleet; the wall-clock budget bought k_worlds 9.28 (vs 14.0) and
  m_opp 5.23 (vs 6.0). Everything cross-iteration (proposed rate, separated-of-raced) inherits
  that confound; everything within-cell (the verdict split, the mirror, the power numbers) does
  not. Per the project's contention rules, the cross-iteration rows are reported, not scored.
- **88 confirm errors, all at r = 0** (no pair completed, mean 3.26 s wall) — a rollout-driver
  or clock finding at 4.0% of attempts. The verdict-keyed counters kept these out of the leaf
  evidence, which is precisely what defect-fix #3 was for.
- One checkpoint, one budget pair, mirror only, depth 1; no ELO anchor. Iteration 1's §6 caveats
  carry unchanged.
- 5 tied games (0.4%); max turns 250 reached at least once (cap games score as configured).
