# E1-0: do the banked bot-stratum IMPs survive a referee that plays the REAL bot? PREREGISTRATION (2026-10-25)

**Written and committed BEFORE any bot-referee playout of a registered state exists.** Spec:
[`design_search_next.md`](../../../endstate/design_search_next.md) (DRAFT, `1d792221`) D-1 and E1's instrument term;
its review [`search_next_review_2026-10-26/REVIEW.md`](../search_next_review_2026-10-26/REVIEW.md) (`9738296f`), F8,
which recommends this cell first. The read is [`score_e10.py`](score_e10.py), the play side [`play_e10.py`](play_e10.py);
both are part of this registration. Nothing below changes after a registered state is played; a deviation is reported
as a FINDING in the README, never edited here.

Abbreviations: P\* = the plateau policy; IMP = the win probability gained per decision by playing a scorer's pick
instead of the policy's top choice, judged by the referee T on held-out seeds (`search_ledger.md` §6.2); O / F / FN /
POOL as in the ledger; CI = a battle-clustered percentile bootstrap interval; D = IMP under the bot referee minus IMP
under the self-referee, on the same state and the same pick.

**What was touched before registration** (no bot-referee outcome of a registered state was played, printed or read):
- The code: `main.value_search.playout.referee` (the referee path with the in-core bot toggle, default OFF) and its
  tests (`referee_integration_test.py`, on the committed `BANK_V1`; the bot test fails when the passthrough is
  removed, checked by hand).
- An ENGINEERING SMOKE of `play_e10.py` on SNAPSHOT-stratum LEAN states (not registered here; their opponent was a
  snapshot, so their bot rows mean nothing), playing `heuristic` as the bot (§8). It read cost, and the self-mode
  replay's character agreement with the banked truth rows.
- The read's planted selftest (`score_e10.py --selftest`).
- The self-referee side only (already published): the extraction's G-IDENT check against the published bot-stratum
  points, and the per-state spread of the self IMPs, for §6's power.

## 0. The question

Every IMP in the ledger was scored by a referee whose playouts put the searched network on BOTH sides (D-1). A "bot-stratum
IMP" is therefore a gain against our own network's continuation from a state a bot game reached. **On the bot-stratum
states, do the picks the maturity battery, world-sampler stage 1 and stage 1c already banked gain the same against
the bot that actually played the state?** If not, the ledger's bot-stratum IMPs are relabelled as self-referee reads
(SELF-REFEREE-MISREADS, the design's E1 rule), and E1's full build is worth its day. E1-0 resolves a GROSS misreading
(about ±0.015, F8), not the game bar.

## 1. States, anchor, banked picks

- **States:** the maturity battery's INTERIM bank, LEAN, **bot stratum: 160 states**
  (`../hidden_info_maturity_x26_2026-10-18/states_mat_x26.json`, `lean` and `stratum == "bot"`; every one of the nine
  in-core bots). Each experiment reads its OWN valid set inside them (≈ 143: the states F could build and its own
  exclusions), exactly as its registered read did.
- **The anchor:** X26 **225M** (the maturity archive's registered copy, sha256 re-checked at load), the one anchor and
  bank that all three experiments share. The pin `9c033489` plays (its release core, which already carries the `bot`
  key); one torch thread.
- **The banked picks:** every source and scorer each experiment scored, re-derived by its own code from its banked
  rows (deterministic):
  - maturity 225M: O, F, FN at LEAF-K8, LEAF-K4, TRUNC-4-K8, CRITIC (argmax; `score_mat.score_anchor`), and G_F = O − F;
  - stage 1: every arm (`F_legal`, `F_off`, `B`, `C_ladder`, `C_forum_strict`, `NS`, `C_pool`, `F_rolls`,
    `F_rolls_w`, and the banked O / F / FN) at the same scorers (`score_s1.score`), and each primary X − F_legal;
  - stage 1c: every played arm (`F8`, `F128L`, `C128L`, `M256L`, `F128d`, `C128d`, `C8`) under piKL (L1's fitted λ)
    and the argmax, at LEAF-K8, TRUNC-4-K8, CRITIC (`read_1c.read_b`'s per-state rule).
- **Not in scope:** the 25M / 100M / 175M LEAN anchors (cost; the same question at a weaker policy), and the plateau
  step's P\* rows on both banks: its queue finished 2026-10-23 but its registered read has not been made, and
  re-scoring them would read its outcomes first (a FINDING in the README).

## 2. The two referees

Both play every legal action × the banked referee's own 24 seeds (`truth.turn_seeds(id, 48)[16:40]`) to the END in the
TRUE world, from the banked root command, X26 225M greedy on our side, production's stall forfeit on our side,
`max_turns` 999 (the banked truth rows' settings).
- **(a) SELF**, as banked: the truth rows' seeds 16..39, unchanged. Its replay through the shipped referee path is the
  determinism guard G-REPRO (§5).
- **(b) BOT:** the opponent is the state's own scripted bot (`cycle_opp`'s name) INSIDE the core, never forwarded; its
  route seed is `bot_seed(id)` = the first 52 bits of sha256(`search_e1_0_v1:bot:` + id). One playout open per action,
  so branch b = the seed's index and the bot's dice streams are COMMON across actions, as the battle's dice are.
  - The bot's real dice state at that decision is not recoverable, so its streams are fresh; an in-core bot is
    memoryless apart from its dice (`bots::Bot` holds two RNG streams), so the referee's bot IS the bot that played.
  - The opponent's open root decision (a simultaneous move) is the bot's own choice, as in the real game.

**The substitution.** Each experiment's scoring code runs unchanged, with one change: in mode (b), Tier 1's
`t1_data.truth_arrays` sees a bot state's truth row with seeds 16..39 replaced by the bot referee's. No scorer reads
seeds 16..39, so every pick is the banked pick (G-PICKS checks it) and only T changes.

## 3. The statistic

Per cell (an experiment × source × scorer × rule) and state: IMP = T(pick) − T(top-1) under each referee, on the states
where both referees hold a valid row.
- **The per-decision gain** under each referee: the mean IMP_self and IMP_bot.
- **The difference** D = mean(IMP_bot − IMP_self), paired by state.
- **The rank agreement:**
  - over actions: per state, Spearman between T_self and T_bot over the legal actions, and the share of states whose
    top-T action is the same (ties toward the policy's order);
  - over states: per cell, the correlation of the per-state IMPs (states where either is non-zero);
  - over cells: Spearman between the cells' mean IMP_self and mean IMP_bot.
- **Intervals:** ONE battle-clustered bootstrap over the bot states' battles, 2,000 draws, `default_rng(20261026)`,
  states weighted equally, every cell on the same draws.
- **PRIMARY FAMILY (4 cells; Bonferroni, 98.75 % intervals):**
  1. `MAT:F:LEAF-K8`: honest search today (maturity 225M F, argmax);
  2. `MAT:O:LEAF-K8`: the oracle;
  3. `S1:C_pool-F_legal:LEAF-K8`: stage 1's only PAYS (team identification over the honest base);
  4. `1C:PIKL:C128L:LEAF-K8`: stage 1c's corpus-likelihood arm under its anchored rule.
- Every other cell is SECONDARY (95 %), DESCRIPTIVE, and listed when it would meet the rule below.

## 4. The decision rule

Rule 8: a bound within ±0.0005 of its boundary is not clear. A cell **DISAGREES** iff
- D's interval excludes 0 (lower > +0.0005 or upper < −0.0005) **and** |D| > 0.0018 (the design's
  SELF-REFEREE-MISREADS magnitude), **or**
- the two referees' sign classes are OPPOSITE (POSITIVE: lower > +0.0005; NEGATIVE: upper < −0.0005).

The verdict (`score_e10.verdict`):
- **RELABEL-NEEDED** (= the design's SELF-REFEREE-MISREADS) iff any primary cell DISAGREES. The ledger's bot-stratum
  IMPs are relabelled as self-referee reads, with the bot referee's numbers beside them.
- **REFEREE-AGREES** iff no primary cell disagrees and every primary D interval lies inside ±0.015 (lower > −0.0145,
  upper < +0.0145). Meaning: no gross misreading on these states. It does NOT certify agreement at the game bar
  (0.0018), and it is no read of S, V or W against the bot.
- **UNDECIDED** otherwise (an interval too wide to rule out a gross misreading, with no disagreement).

## 5. VOID guards (any failure VOIDs the unit; nothing is read)

- **G-REPRO (determinism):** the self-referee replayed through the shipped `referee` path at the pin (`play_e10.py
  --mode self`) reproduces the banked truth rows' seeds 16..39 CHARACTER FOR CHARACTER on every action of every bot
  state with a valid truth row.
- **G-IDENT:** the self-mode extraction reproduces every PUBLISHED bot-stratum point (maturity 225M G_F; stage 1's
  five primary X − F_legal at LEAF-K8 and TRUNC-4-K8; stage 1c's piKL and argmax arms at its three scorers) within
  1.5e-5 (the jsons round to 5 decimals) and at the published n.
- **G-PICKS:** the sequence of picks in mode (b) equals mode (a)'s (the substitution changes T only).
- **G-COVER:** every bot state with a valid truth row holds a bot row; at most 2 % refused (a bot refusal or a core
  error), and those states are dropped from every cell.
- **G-TOKENS:** every bot row's root tokens equal the bank's.
- **G-BOT-SIDE:** in every bot row, the forward answered no decision of the bot's side.
- **G-BOT-NAME:** every bot row's bot and seed are the registered ones.

## 6. Power at the actual bank size

From the SELF side only (published numbers; no bot row existed). The bank holds 160 bot states in 141 battles; each
cell's valid set is below. SE_self is the bootstrap SE of the mean IMP_self. With SE_bot ≈ SE_self and a correlation
ρ between the two referees' per-state IMPs, SE_D = SE_self · sqrt(2(1 − ρ)), and the 98.75 % half-width is ≈ 2.50 SE_D.

| primary cell | states (battles) | mean IMP_self | SE_self | picks ≠ top-1 | D half-width, ρ = 0 | ρ = 0.6 |
|---|---|---|---|---|---|---|
| `MAT:F:LEAF-K8` | 143 (125) | −0.0102 | 0.0095 | 35 % | ±0.034 | ±0.021 |
| `MAT:O:LEAF-K8` | 160 (141) | +0.0378 | 0.0122 | 38 % | ±0.043 | ±0.027 |
| `S1:C_pool-F_legal:LEAF-K8` | 142 (124) | +0.0270 | 0.0111 | 48 % | ±0.039 | ±0.025 |
| `1C:PIKL:C128L:LEAF-K8` | 143 (125) | +0.0092 | 0.0074 | 20 % | ±0.026 | ±0.017 |

**What this can and cannot see.**
- RELABEL-NEEDED through D needs |D| ≳ 0.02–0.04 (a shift of the same size as the cells' own self IMPs). A sign flip
  can be seen only on `MAT:O` (the one primary cell whose self interval is clear of 0 at 98.75 %).
- **REFEREE-AGREES needs every half-width < 0.0145, so it is reachable only if ρ ≳ 0.8 on every cell.** With no
  instrument effect and the design's ρ ≈ 0.6, the expected verdict is UNDECIDED. That is E1-0's honest power at 160
  states: it detects a GROSS misreading, and cannot certify its absence. Raising the bot referee's seeds would not
  help much (the self side's 24 banked seeds are fixed) and was not done.
- G-IDENT already passed on the self side: 53 / 53 published bot-stratum points reproduced (§0).

## 7. Predictions (GUESSES, written before any bot row)

- The bot referee's T LEVEL is higher than the self-referee's (the bots are weaker than X26 225M): mean T over actions
  +0.05 to +0.15 (the review's state-level estimate, ≈ 0.1).
- `MAT:O`: D ≈ −0.015 [−0.04, +0.01] (part of the oracle's edge is fitted to the network's own continuation; ceiling effects
  against a weak opponent).
- `MAT:F`: D ≈ +0.005 (a weak bot punishes honest search's bad picks less).
- `S1:C_pool-F_legal`: D ≈ −0.01. `1C:PIKL:C128L`: D ≈ 0.
- Rank agreement: state-level Spearman of T over actions ≈ 0.4 (median); top-T action the same in ≈ 50 % of states;
  cell-level Spearman of the means ≈ 0.6 (the design's guess for the IMP correlation).
- Verdict: UNDECIDED ≈ 55 %, RELABEL-NEEDED ≈ 35 %, REFEREE-AGREES ≈ 10 %.

## 8. Cost, CPU discipline, archive

- **Smoke (engineering, §0):** 2 snapshot-stratum states, self mode: 0 of 312 referee characters differ from the banked
  truth rows; 12–79 CPU-s a state (0.10–0.41 CPU-s a playout), peak 1.3 GB. Bot mode (the same 2 states against
  `heuristic`, engineering only): 5–18 CPU-s a state (0.04–0.09 CPU-s a playout), 0 rows of the bot's side forwarded,
  tokens equal, peak 1.0 GB.
- **Estimate:** 160 states × ≈ 7 actions × 24 seeds ≈ 27,000 playouts per mode; self ≈ 1.5–2 CPU-h, bot ≈ 1 CPU-h
  (the forward answers one side; the smoke says less). ≈ 2–3 CPU-h in all, inside the review's 1–3 CPU-h order.
- **CPU (root rule 5b):** `run_e10.sh`: one torch thread, `chrt --idle`, `nice -n 19`, `taskset -c 5-7,13-15`, each
  chunk in its own `mem_cap` scope (6 GB), bounded by `timeout` ≤ 45 min (the long lane) or in the short lane
  (≤ 600 s); `CUDA_VISIBLE_DEVICES=""`. No GPU.
- **Order:** the bot mode first, then the self mode (G-REPRO); both in the registered order sha256(`search_e1_0_v1:` +
  id); a chunk caps NEW work, so the done states are always a prefix.
- **Archive:** `~/gen3ai_archive/search_e1_0/` (`rows/bot/`, `rows/self/`, `smoke/`). Read-only on `models/`, the
  maturity, stage-1 and stage-1c archives.
- **The read** runs once both modes are complete: `score_e10.py` -> `result_e10.json`, then the README.

## 9. What each verdict changes

- **RELABEL-NEEDED (SELF-REFEREE-MISREADS):** `search_ledger.md` relabels every bot-stratum IMP it quotes (stage 1's
  per-stratum, stage 1c's bot stratum, the maturity G_F on bots, and "on bots and own snapshots alike") as a
  SELF-REFEREE read, with the bot referee's number beside it; E1's full build (the instrument term on every stratum)
  becomes worth its day.
- **REFEREE-AGREES:** the ledger notes that the banked bot-stratum IMPs survive the real bot at ±0.015 (no gross
  misreading); no relabel; E1's instrument term drops in rank (S, V and W against the bot stay open).
- **UNDECIDED:** no relabel; the ledger quotes the bot referee's numbers as DESCRIPTIVE beside the banked ones, with
  "not resolved at E1-0's size", and E1 sizes its instrument term from this unit's measured variance (the review's F7
  two-stage rule).
- **VOID:** nothing is read; the failing guard is a FINDING and is fixed first.
