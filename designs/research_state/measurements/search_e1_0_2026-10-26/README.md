# E1-0: the bot referee on banked bot-stratum picks (2026-10-25)

**Status: READ — UNDECIDED** (registered rule; every VOID guard passed). No relabel of the ledger's bot-stratum IMPs
is required, and a gross self-referee misreading is NOT ruled out. Registered in [`PREREGISTRATION.md`](PREREGISTRATION.md)
(sha256 `9cf95f9d…41163`; file hashes in [`e10_registration.json`](e10_registration.json)), committed at local commit
`c928814f2` (2026-10-25 17:45 PDT) before any bot-referee playout of a registered state; the result is
[`result_e10.json`](result_e10.json).

| file | what |
|---|---|
| `PREREGISTRATION.md` | the question, states, referees, statistic, rules, VOID guards, power, predictions |
| `play_e10.py` | the play side at X26's pin: every legal action × the 24 referee seeds, bot or self referee (`main.value_search.playout.referee`, loaded by file path from this checkout) |
| `run_e10.sh`, `loop_e10.sh` | the CPU-rule wrapper (one thread, SCHED_IDLE, nice 19, agent cores, `mem_cap`, short-lane chunks) and the chunk loop |
| `score_e10.py` | the read: each experiment's own scorer with the referee's T substituted; `--selftest` on planted data |
| `posthoc_e10.py` | POST HOC, descriptive: the split-half noise ceiling of the rank agreement |

Archive: `~/gen3ai_archive/search_e1_0/` (`rows/bot/`, `rows/self/`, `smoke/`, `logs/`).

## 1. How it was read

Exactly as registered: the selftest passed, the play ran in 14 short-lane chunks (bot mode, then self mode), and
`score_e10.py` ran once. Nothing was re-run or changed after the outcome.

| guard | result |
|---|---|
| G-REPRO | the self-referee through the shipped `referee` path replays **160 / 160** states, **0 of 28,392** referee characters differ from the banked truth rows |
| G-IDENT | **53 / 53** published bot-stratum points reproduced (maturity 225M G_F; stage 1's primary X − F_legal; stage 1c's piKL and argmax arms) |
| G-PICKS | the 28,839 picks are identical under both referees |
| G-COVER / TOKENS / BOT-SIDE / BOT-NAME | 160 / 160 bot rows, 0 refused; tokens equal; the bot's side never forwarded; the registered bots and seeds |

## 2. The primary family (bot-stratum states, X26 225M, 98.75 % battle-clustered intervals)

D = IMP under the bot referee minus IMP under the self-referee, on the same state and pick.

| cell | n | IMP, self referee | IMP, bot referee | D | DISAGREES |
|---|---|---|---|---|---|
| `MAT:O:LEAF-K8` (oracle) | 160 | **+0.038** [+0.009, +0.070] | +0.014 [−0.012, +0.046] | −0.024 [−0.054, +0.009] | no |
| `MAT:F:LEAF-K8` (honest search) | 143 | −0.010 [−0.033, +0.012] | +0.010 [−0.012, +0.040] | +0.020 [−0.010, +0.054] | no |
| `S1:C_pool-F_legal:LEAF-K8` (team identification) | 142 | +0.027 [−0.002, +0.054] | −0.006 [−0.034, +0.019] | −0.033 [−0.068, +0.002] | no |
| `1C:PIKL:C128L:LEAF-K8` | 143 | +0.009 [−0.011, +0.027] | +0.006 [−0.007, +0.025] | −0.004 [−0.028, +0.030] | no |

- No D interval excludes 0, and no sign class flips to the OPPOSITE sign. The oracle's self interval is POSITIVE and
  its bot interval NULL; under the rule that is not a disagreement.
- Every D interval is wider than ±0.015, so REFEREE-AGREES is not reached: **UNDECIDED**, as §6 forecast (the measured
  per-state correlations of the two referees' IMPs are 0.14–0.32 on three cells and −0.35 on C128L, below the ≈ 0.6
  the design guessed, so the intervals are near the ρ = 0 column of §6).
- The direction is consistent: the cells that gain under the self-referee (O, C_pool − F_legal) gain less against the
  bot, and the cell that loses (F) turns slightly positive. Against the real bot the IMPs move toward 0, or past it
  (DESCRIPTIVE).

## 3. Secondary and descriptive (95 %; not verdicts)

- **The oracle gap on bots** G_F = O − F (maturity 225M, LEAF-K8): self +0.050 [+0.025, +0.077], bot +0.010
  [−0.015, +0.035], D **−0.040 [−0.068, −0.015]**. Against the real bot, knowing the hidden facts is worth about a
  fifth as much per decision as the self-referee said. This is a SECONDARY cell of 108, without a multiplicity
  correction.
- 23 of the 108 secondary cells meet the disagreement rule at 95 % (no multiplicity correction). One is G_F above.
  The other 22 are all honest or open-population arms: stage 1c's arms under the argmax and piKL, and stage 1's B,
  C_forum_strict, C_ladder and NS, mostly at CRITIC or TRUNC-4-K8. Each reads NEGATIVE-leaning under the self-referee
  (−0.004 to −0.023) and positive-leaning under the bot (+0.007 to +0.031), D +0.021 to +0.047. So "honest search
  plays at or below the policy" is, on the bot states, partly a self-referee reading (DESCRIPTIVE). The full table is
  `result_e10.json` `secondary_cells`.
- **The two referees rank actions differently.**
  - Per state, Spearman over the legal actions between T_self and T_bot: mean 0.18, median 0.19 (97 states with a
    defined rank). The top-T action is the same in 48 % of states.
  - POST HOC (`posthoc_e10.py`, split halves of 12 seeds): self vs self 0.42 (mean), bot vs bot 0.43, self vs bot
    0.19. So the cross-referee agreement is about HALF of each referee's own noise ceiling: a real difference in
    which action is best, not just noise.
  - Across the 112 cells, Spearman of the mean IMPs is 0.16.
- **The level:** mean T over actions 0.61 under the self-referee, 0.78 under the bot (the bots are weaker than the
  network); the review's estimate was ≈ 0.1.
- Predictions (§7): the level and the oracle's sign held; the rank agreement was lower than guessed (0.19 vs 0.4;
  48 % vs 50 % top-action).

## 4. What it says

The banked bot-stratum IMPs are self-referee numbers, and E1-0 cannot certify them against the real bot. It also does
not detect a gross misreading in any primary cell at this size. The descriptive pattern points one way: against the
real bot, the self-referee's gains (the oracle, team identification) shrink toward 0, honest search's slightly
negative picks turn slightly positive, and the two referees' action rankings agree only about half as well as each
agrees with itself. The decisive E1 instrument read needs more states: the primary D intervals are ±0.026 to ±0.036
here, so resolving ±0.015 at 98.75 % needs ≈ 4–5× this bank (≈ 650–750 bot states). E1's two-stage sizing (review
F7) should start from this unit's variance.

## 5. FINDINGS

1. **The plateau step's banked rows are unread.** The maturity battery's plateau-step queue finished 2026-10-23
   01:07 (`QUEUE_DONE`, nothing parked), but no `result_mat_plateau.json` exists and the README still says RUNNING.
   E1-0 did not touch those rows, so as not to read them before their registered read.
2. **Only one anchor.** E1-0 re-scores X26 225M; the 25M / 100M / 175M LEAN reads and the P\* plateau are not
   re-scored.
3. **The bot's dice are fresh**, not the game's: the real dice state at the decision cannot be recovered. The in-core
   bot is otherwise the bot that played (memoryless apart from its two streams; eval plays the in-core bots).
4. **Short-lane chunks hit their 560 s bound** 7 times (exit 124). Each loses only the state in flight, which the next
   chunk re-plays; G-COVER and G-REPRO confirm complete, exact rows.
5. **Cost:** 1.37 CPU-h of play (bot 0.35, self 1.02); the wall time was about 3.5 h, almost all of it the heavy-slot queue.
