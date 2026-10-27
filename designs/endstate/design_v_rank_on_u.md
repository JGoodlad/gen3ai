# V-RANK — does the search's value rank the moves of ONE position as well on unfamiliar teams as on home72, against Kakuna?

**Status: DRAFT (2026-10-27), explicit-only; NOT REGISTERED.** The PREREG is NOT READY: it needs the builds B-V1–B-V4
(§6) and a pin. §3–§4 are the text it will register unchanged, with the pin, the engine and stage 0's measured cost
added. This unit wrote this doc and ran one CPU probe on banked rows (§5.1; < 1 CPU-min, read-only, no game). It
answers FINDING F5 of the literature survey (`5d554c23`,
[`context_generalisation_2026-10-27.md`](../research_state/literature/context_generalisation_2026-10-27.md)) and
gates S-U ([`design_search_u_headroom.md`](design_search_u_headroom.md)).

**Abbreviations.**
- P\* = `rb_x26_s2601` @280,000,170, the plateau policy, greedy. Kakuna = Metamon's Kakuna ckpt 34, greedy.
- home72 = 72 pool teams. U = H (`heldout72_hpfix`), F (`forum72`) and P72 at equal panel weight (X51's definition).
- S = R0 §4's one-ply search, preset `r0s4` (8 honest worlds, `dice="seed"`, the TRUNC-4-K8 leaf: every legal action
  played 8 times 4 turns deep with P\* on both sides, then the critic V; piKL λ 0.015).
- **Q̂(a)** = S's value of legal action a at a decision: what the search ranks. It is V as one-ply search uses it.
- **y(a)** = the result (win 1, tie ½, loss 0) of the real game in which we play a at that decision and P\* greedy
  everywhere else, against Kakuna, on the deal's own seed. Its expectation, Q_K(a), is the ground truth.
- **β** = the within-node calibration slope of y on Q̂ (§4.1): the share of Q̂'s action differences that are real
  against Kakuna. β = 1: Q̂'s differences are realised one for one; β = 0: they are noise.
- pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; GUESS = an unmeasured effect;
  ESTIMATE = a cost from measured rates; MY JUDGEMENT = an inference no read states; PDL = the performance-difference
  lemma.

---

## 0. One screen

1. **The question (F5).** H-MID's "V's ordering is intact on U" is an AUC over GAMES. One-ply search needs V to rank
   the candidate ACTIONS of one position. No read has measured that on U, and no read has measured it against a
   non-self opponent at all.
2. **The ground truth: live branching against Kakuna.** Our play and Kakuna's are deterministic on one engine (0 of
   2,100 replayed games differ, `kakuna_determinism_2026-10-27`). So a deal replayed from turn 1 reaches the same
   position byte for byte. At a chosen position we force each legal action in turn, then let P\* and Kakuna play the
   game out. Every branch is a real game against the real opponent: **no opponent model anywhere in the truth**, so
   H-MID's M1 flaw (a self-play referee) does not arise.
3. **The read.** 2 positions a deal (turns 10–19 and 20–30, a hash draw), every legal action branched, Q̂ from S at
   the position. 1,200 positions a set (home72; U at 400 a panel). The primary is the slope β on U, and the
   difference Δβ = β_U − β_home72.
4. **What it can decide, and what it cannot.** On P\*'s own banked states (the self-play proxy) Q̂'s slope is 0.54, yet
   search gains nothing there (L1 at P\*: +0.0001 a decision). So **a good slope is necessary for search to pay, not
   sufficient**. The read can turn S-U OFF; it can only make S-U ELIGIBLE, never predict that it pays.
5. **The gate on S-U** (§4.3). RANK-NULL-ON-U (UB(β_U) < 0.25, below the slope at which the typical piKL deviation
   breaks even) or RANK-WORSE-ON-U (UB(Δβ) < 0) turns S-U OFF. RANK-HOLDS-ON-U (LB(Δβ) > −0.25 and LB(β_U) > 0)
   makes it ELIGIBLE. Anything else leaves S-U held, as now.
6. **Power** (at the probe's spread, §5.2): at 1,200 a set, a null slope on U is caught 0.95 of the time; a U slope
   half the home one (Δβ −0.27) reads WORSE 0.78; an intact one (Δβ 0) reads HOLDS 0.71. A false kill is ≤ 2.5 % a
   look. **The two banked Kakuna games show a Q̂ spread ≈ 0.2–0.4× the probe's**; if that holds, power falls by about
   √2.5, and the registered sizing rule (outcome-free) doubles n at most.
7. **Cost** (ESTIMATE, §5.3). ≈ 2.4 CPU-min a position, **≈ 96 CPU-h (80–115) at 1,200 a set**, ≈ 48 CPU-h at the
   look that can kill; ≈ 190 at the size cap. S-U costs ≈ 270 (phase 1, Amendment 1's turn window) to ≈ 1,000 CPU-h.
8. **Expected class** (GUESS): HOLDS 0.45, WORSE 0.12, NULL 0.10, UNDECIDED 0.33. P(S-U OFF) ≈ 0.2.

---

## 1. The question

### 1.1 What is measured, and what search needs

| read | what it ranks | referee | on U? |
|---|---|---|---|
| H-MID (`f200258f`) | GAMES by V at turn 10 (AUC 0.667 U, 0.663 home72) | the game against Kakuna | yes, between games |
| L1 at P\* (`search_l1_plateau_2026-10-25`) | the piKL pick against π's top, IMP +0.0001 | P\* on both sides (self) | no: pool teams |
| E1-0 (`search_e1_0_2026-10-26`) | the self-referee's misreading against an in-core bot | the bot | no |
| **V-RANK (this doc)** | **the ACTIONS of one position, by Q̂** | **Kakuna, live** | **yes, within a node** |

S-U's case for headroom on U is that a level error in V is common to all successors of a node and cancels, while V's
ORDERING survives (S-U §1.1). That ordering is the within-node one, which H-MID never measured (S-U §8 A1;
survey F5). S-U §10 rejected a decision-level read as a GATE because its referee would be P\* (E1-0's open problem).
Live branching removes that objection: the referee is Kakuna.

### 1.2 Why the self-play proxy cannot answer it

The self-referee plays the continuation with P\* on both sides. On U it models Kakuna as P\* piloting a team P\*
pilots ≈ 11 pp worse than its own. The premium is opponent-relative and cancels in self-play (H-MID REVIEW M1). A
self-referee ranking read on U would grade Q̂ against a second copy of the same blind spot. It is kept here only as a
REPORTED decomposition (§4.4), never as the truth.

### 1.3 Relation to the search plan

This is E1's V-term of [`design_search_next.md`](design_search_next.md) (V's within-state ranking against the real
referee), for the one opponent E1 cannot run in the core (Kakuna, its D-7). E1 buys 24 in-core playouts an action;
V-RANK buys ONE live game an action and makes up the noise with positions (§5).

---

## 2. The ground truth: options

| option | what it is | does the truth assume the opponent is P\*? | verdict |
|---|---|---|---|
| **Live branching against Kakuna** | replay the deal to the position (deterministic), force action a, play out P\* vs Kakuna; one game an action | **no** | **CHOSEN.** Exact referee; noisy (one outcome an action), so it needs positions |
| Rollouts with a Kakuna proxy in the core | the in-core bots or P\* as the opponent | yes, or a proxy unrelated to Kakuna | rejected: H-MID's M1 flaw (P\*) or the wrong opponent (bots: that is E1) |
| Natural experiments: P\*'s greedy choice differs between near-identical states | P\* is deterministic, so a different choice means a different input | n/a | rejected: the differing input is the confound, the continuations differ, and the events are rare. The cleanest instance, the engine pair E1 / E3 (`kakuna_determinism_2026-10-27` §2), has 15 events in 1,260 games |
| The X51 twins (P\* and X26 @161M on the same deals) | their first divergence is a same-state branch | no | rejected: each branch continues with a different policy, so it is not one ply |
| Kakuna's own choice where Kakuna pilots our teams (the X37a cells K(X, Y)) | does Q̂ rank Kakuna's move first? | no | rejected as a truth: agreement with a different player measures style, not value, and Kakuna pilots U only ≈ 11 pp better on AVERAGE. It also needs P\*'s encoder on Kakuna's side (REVIEW T3). A possible free descriptive later |
| Outcome-matched pairs (H-MID's matched V) | between-game | no | rejected: between games, not within a node |
| A short-horizon referee (Kakuna to turn t + h, then V) | cheaper | no | rejected: V in the referee shares Q̂'s errors (circular) |

Two properties make live branching affordable.
- **Determinism.** Same engine, same deal: identical games (0 of 2,100, `kakuna_determinism_2026-10-27`; X51's
  argmax rate 1.0000; H-MID 1,600 / 1,600). So no state cloning is needed, and no Metamon internals.
- **Common dice.** Every branch of a position runs on the battle's one seed, so the branches share their random
  numbers as far as their play allows. That cuts the noise of within-node contrasts (§5.1 models it: one referee seed
  per state, shared by its actions).

---

## 3. The read

**Deals.** X51's shape: a deal is (panel, unit seed, game) of a 20-game role-balanced series, played alone
(`main.anchors --deal-game G`). Blocks b = 0, 1, …: each block holds one deal of each U panel (unit seed b // 20,
game b % 20) and three home72 deals (unit seeds 3·(b // 20) + j, j = 0, 1, 2, game b % 20). So each block adds 6
positions a set, and a size change never moves an earlier block's deals.

**The raw game.** Each deal is first played raw (P\* vs Kakuna, greedy both sides). It is π's branch at every position
of the deal, and it defines the positions.

**Positions** (outcome-blind, fixed now). Eligible: our decisions with ≥ 2 legal actions. Stratum A: turns 10–19;
stratum B: turns 20–30. In each stratum the eligible decision with the smallest
`sha256("vrank_v1|<deal key>|<turn>|<decision index>")` is drawn (the deal key is S-U's: a digest of the battle's
`START`). A stratum with no eligible decision adds nothing. Banked X51 rows: 93–97 % of games reach turn 20 (home72
92.8 %, H 97.0 %, F 96.3 %, P72 93.3 %), ≈ 11 decisions a game in turns 10–19 and ≈ 10 in turns 20–30.

**Branches.** At each position, one game per legal action other than π's argmax, in ascending action index: the
deal replayed from turn 1 with P\* greedy, the forced action at the position, P\* greedy after it. π's own branch is
the raw game.

**Q̂ and the shadow arms.** In a position's first branch, at the position and before the forced action is sent, the
player runs:
- S (`r0s4`) on the battle's true log, exactly as S-U's live searcher does (its root guards included), and records
  Q̂(a) for every legal action, the per-world leaf values [A × 8], log π and the exclusion if no world is consistent;
- the self-referee `oracle_leaf8` (B-V4): the TRUE world (`reveal="oracle"`), every legal action played 8 times to
  the end with P\* on both sides. It gives Q_self(a), the in-core self-referee, for §4.4 only.

A position S excludes (`WorldError`) gets no Q̂, so its remaining branches are not played; the exclusion rate is
reported by set.

**G-BRANCH (a GIGO tooth).** Every branch's protocol (both sides, `|t:|` removed) and `START` must be byte-identical
to the raw game's through the request at the position, and the position's live tokens must equal the raw game's. A
mismatch HOLDS the lane for a root cause; an unresolved one VOIDs the read (it is a nondeterminism that corrupts the
pairing). The branch must then play the forced action (recorded).

**Counting.** A game counts by `main.anchors.results.unit_counts` with the forfeit-win guard
(`runner.forced_end_verdict`): a game a peer's drop decided is no outcome. A position enters the primaries iff Q̂
exists and every branch (the raw game included) counted. Positions dropped are reported by set and cause.

**One engine, one pin.** Every game of the read on the engine registered with the PREREG. Nothing is imported: X51's
banked rows (E0) size the read and nothing else. The raw games are compared with X51's game for game as a REPORTED
cross-engine guard.

**Order.** Blocks in order; within a deal, the raw game, then position A's branches, then position B's. One deal is
one unit (one process: the model and the peer start once for ≈ 16 games).

---

## 4. Estimands and the decision rule

### 4.1 The statistic

For a position p with legal actions a: q_c = Q̂(a) − mean_p Q̂, y_c = y(a) − mean_p y. Per panel,

β_panel = Σ_p Σ_a q_c · y_c / Σ_p Σ_a q_c².

β_U = the mean of β_H, β_F, β_P72 (equal panel weight); β_home = home72's; **Δβ = β_U − β_home**.

- **Why a slope.** Outcome noise in y does not bias it (y is the regressand). A within-node rank correlation of single
  outcomes is attenuated by that noise, by an amount that differs with each set's outcome variance.
- **Why every legal action.** S takes its argmax over every legal action, and branching all of them buys the most
  precision per CPU: the probe's SE × √cost is 0.271 for all actions against 0.296 for three (π's top, Q̂'s top, one
  other; `power_probe.json`).
- **Intervals.** A percentile bootstrap over DEALS (both positions of a deal together), resampled within each panel,
  B = 2,000, one `numpy.random.default_rng(20261027)`, the U panels drawn first (`heldout72_hpfix`, `forum72`, `p72`),
  then home72; 2.5 / 97.5 percentiles. Bars compared exactly; a bound within 0.0005 of a bar reads UNDECIDED (root
  rule 8).

### 4.2 The classes

| class | rule | meaning |
|---|---|---|
| **RANK-NULL-ON-U** | UB(β_U) < 0.25 | on U, under a quarter of Q̂'s action differences are real against Kakuna: below the slope at which the median piKL deviation breaks even (§5.1: π's top beats an equally rated alternative by 1.6 pp, and the median deviation's Q̂ gap is 0.066, so break-even ≈ 0.24) |
| **RANK-WORSE-ON-U** | UB(Δβ) < 0, and not RANK-HOLDS-ON-U | Q̂ ranks a node's moves worse on U than on home72, beyond the margin |
| **RANK-HOLDS-ON-U** | LB(Δβ) > −0.25 and LB(β_U) > 0 | Q̂'s ranking on U is real, and not worse than home72's by more than 0.25 (about half the probe's 0.54) |
| UNDECIDED | otherwise | |

RANK-NULL-ON-U is read on its own and dominates: with NULL, the Δβ class is recorded but the gate reads NULL. A
result both worse and within the margin (UB(Δβ) < 0 < LB(Δβ) + 0.25) reads HOLDS, with "worse, within the margin"
recorded.

**Two looks.** Look 1 at half the final blocks reads RANK-NULL-ON-U only, and NULL STOPS the read. Look 2 (final)
reads every class. A NULL at either look has a false rate ≤ 2 × 2.5 % where the true β_U is at the bar (MY JUDGEMENT,
the Bonferroni bound); where β_U is the probe's 0.54, it is ≈ 0.

### 4.3 The gate on S-U (registered with this read; the orchestrator applies it)

| V-RANK | S-U |
|---|---|
| **RANK-NULL-ON-U** (either look) | **OFF.** S-U closes unrun: the search's preferences on U are not real enough against Kakuna to beat π's choice at a typical deviation. Amendment 1 is not registered |
| **RANK-WORSE-ON-U** | **OFF as registered.** Its primary A (I) loses its only mechanism, V's intact ordering. A G_U-only read needs a new case and a new registration |
| **RANK-HOLDS-ON-U** | **ELIGIBLE.** F5 closes in S-U's favour; S-U proceeds through Amendment 1 on the orchestrator's go (and the owner's, for a rented box). **Not a prediction that search pays** (§0.4) |
| UNDECIDED | **Unchanged.** S-U stays held; the read is reported |

### 4.4 REPORTED, never decided

- β per panel, β_home alone, and β in the level-at-10 subgroup (from the raw game's material at turn 10, H-MID's
  population) and by stratum (turns 10–19, 20–30).
- **The search-facing contrasts** (noisy, §5.1):
  - β_C, the slope inside the contenders (π's top 3 ∪ Q̂'s top 2);
  - the partial slope of y on Q̂ given π's logits (Q̂'s information beyond π);
  - α_top, π's top advantage over an alternative that Q̂ rates equal, by set: a decision-level read of the piloting
    deficit (smaller on U if π pilots U worse);
  - H, the true advantage over π's top of the contenders Q̂ prefers;
  - the piKL pick's decision-level IMP against Kakuna, y(pick) − y(π's top).
- **The self-referee decomposition (the M1 / A2 read).** Same positions, by set:
  - β_self, the slope of Q_self on Q̂ (the self-referee's view of Q̂);
  - the slope of y on Q_self, corrected for Q_self's 8-seed noise (its reliability from the seed variance).
  It answers "does the self-play continuation rank actions like the real one?". On U it isolates the leaf's opponent
  model (S-U A2) from V.
- Q̂'s Monte Carlo reliability by set (from the per-world values) and β corrected for it: separates "V ranks worse"
  from "S's estimate is noisier" on U.
- The world quality (S-U's species hit) and the exclusion rate by set; the share of deals with a stratum-B position;
  the raw games against X51; the cost per position and per deal.

### 4.5 VOID

Counted games on two engines or an unknown one; an unresolved G-BRANCH mismatch; a raw game on another checkpoint
(sha256); any S record whose preset differs from `r0s4`, or any self-referee record whose preset differs from
`oracle_leaf8`.

---

## 5. Power and cost

### 5.1 The probe (banked rows, SELF-REFEREE, sizing only)

`../research_state/measurements/v_rank_on_u_2026-10-27/scripts/power_probe.py` → `power_probe.json`. It reads the
maturity battery's plateau bank through L1's registered loader: 719 states of P\*'s own pool-team games (464 battles),
every legal action played to the end in the TRUE world by P\* on both sides (24 referee seeds), and the honest F
worlds' TRUNC-4-K8 values (Q̂). One referee seed per state, shared by its actions, has the shape of one live branch
per action. Pure numpy, < 1 CPU-min, 1.1 GB, read-only.

| quantity | value |
|---|---|
| legal actions a state | 7.5 |
| Q̂'s within-node SD | 0.052 (median 0.046); Σ q_c² a state 0.030 (turns 10–30: 0.033) |
| the truth's within-node SD (24 seeds, noise-corrected) | 0.119 |
| **β, Q̂ against the 24-seed truth** | **0.54 [0.43, 0.64]** (turns 10–30: 0.54 [0.40, 0.68]) |
| β with ONE seed an action | 0.57 [0.39, 0.74]; SE 0.089 (0.091 over battles: clustering × 1.02) |
| β inside the contenders | 0.30 [0.11, 0.49] |
| partial β given π's logits | 0.37 [0.25, 0.49] |
| α_top (π's top over an equally rated alternative) | 1.6 pp |
| H (Q̂-preferred contenders over π's top) | +0.0006 [−0.015, +0.016]; one seed: SD 0.47 a state |
| piKL deviations | 13.8 % of states; their Q̂ gap over π's top: median 0.066 (IQR 0.039–0.110) |

Three readings shape the design.
1. **The slope is real and search still gains nothing.** β = 0.54 and β_C = 0.30, yet H ≈ 0 and L1's IMP at P\* is
   +0.0001. The linear slope over-predicts at the top of the ranking. Hence §0.4: the read gates OFF, not ON.
2. **The gain-facing estimands cannot reach a game bar.** H's one-seed SD of 0.47 gives SE ≈ 0.018 at 1,000 positions
   a set. Through the PDL, S-U's U-HEADROOM size (I ≈ +5–7 pp over ≈ 30 searched decisions) is ≈ 0.002 a decision,
   or ≈ 0.007 a deviation. That would need ≈ 10⁵ positions a set, the review's F12 again. They stay REPORTED.
3. **The slope has power.** Its one-seed SE is 0.089 at 719 states, so it scales to a useful precision at 10³ a set.

**The Q̂ spread on Kakuna games may be smaller (FINDING V-2).** The only searched Kakuna decisions banked, S-U stage
0's first game and the build smoke (31 decisions at turns 10–30; 3 games on 2 deals), give Σ q_c² ≈ 0.007–0.013 a state:
0.2–0.4× the probe's. SE scales as 1 / √(Σ q_c²), so power could be ≈ √2.5 lower. Stage 0 measures it (§5.4).

### 5.2 Power (normal approximation; SE from the probe, scaled by √(719 / n))

| n a set | SE β_U | SE Δβ | P(NULL) at β_U = 0 / 0.10 | false NULL at β_U = 0.54 | P(WORSE) at Δβ = −0.27 | P(HOLDS) at Δβ = 0 |
|---|---|---|---|---|---|---|
| 600 (look 1) | 0.099 | 0.141 | 0.71 / 0.33 | ≈ 0 | 0.49 | 0.43 |
| **1,200** | **0.070** | **0.099** | **0.95 / 0.57** | **≈ 0** | **0.78** | **0.71** |
| 2,000 | 0.054 | 0.077 | 0.996 / 0.79 | ≈ 0 | 0.94 | 0.90 |
| 1,200 at the banked Kakuna spread (Σ q_c² 0.012) | 0.111 | 0.157 | 0.61 / 0.27 | ≈ 0 | 0.41 | 0.36 |
| 2,400 at that spread (the cap) | 0.079 | 0.111 | 0.89 / 0.48 | ≈ 0 | 0.68 | 0.61 |

The first three rows are `power_probe.json` (`power`); the last two are the same model at SE × √(0.030 / 0.012).
False WORSE at Δβ = 0 is 2.5 %; false HOLDS at Δβ = −0.27 is ≈ 1.5 %. The model assumes Kakuna's outcome noise and
the misfit of the truth around Q̂ are the probe's (UNVERIFIED: E1-0's direction says a non-self referee shrinks the
slope, which lowers the power of HOLDS and raises that of NULL).

### 5.3 Cost (ESTIMATE from measured rates)

| item | rate | source |
|---|---|---|
| a game in a unit, both sides | ≈ 12 s of wall (CPU ≈ 12–18 s, ESTIMATE) | X51: 11.8 s a game in 20-game units |
| a one-game unit's start-up | ≈ 20 s of wall | S-U stage 0's raw units: 24–29 s for 27–76 turns |
| S (`r0s4`) at turns 10–30 | median ≈ 6 s (turns 10–19), ≈ 13 s (20–29), max 44 s; ≈ 12 s mean | S-U `AMENDMENT_1_DRAFT.md` §1; the smoke 6.4 s |
| `oracle_leaf8` | ≈ 8 × 8 × 0.3 s ≈ 20 s (UNVERIFIED) | `design_search_next.md` E2's full playout ≈ 0.3 CPU-s |
| legal actions at a Kakuna position, turns 10–30 | 8–9 (31 banked decisions) | S-U stage 0 and smoke |

**A position:** ≈ 7 forced branches × 12–18 s + S and the self-referee ≈ 32 s + its share of the raw game and the
start-up ≈ 8 s ≈ **120–170 CPU-s, central ≈ 2.4 CPU-min**. A deal (2 positions, ≈ 16 games) ≈ 5 CPU-min, so a deal
unit may fit the short lane (≤ 600 s) at nice 19; stage 0 says.

| size | positions | CPU-h (central, range) |
|---|---|---|
| stage 0 (5 blocks) | 60 | ≈ 2.4 |
| look 1 (100 blocks) | 1,200 | ≈ 48 (40–58) |
| **final, 1,200 a set (200 blocks)** | **2,400** | **≈ 96 (80–115)** |
| the cap, 2,400 a set (400 blocks) | 4,800 | ≈ 190 (160–230) |

### 5.4 Stage 0 and the sizing rule (fixed now; cost is never an outcome)

- **Stage 0** = blocks 0–4 (30 deals, 60 positions). It reads, without any y: the cost per position and per deal,
  G-BRANCH on every branch, the exclusion rate, the legal-action count, and **s̄, the mean Σ q_c² a position** (Q̂
  only). Its games count toward the read if the play path is unchanged afterwards.
- **The sizing rule:** final blocks = 200 × min(2, max(1, 0.030 / s̄)), rounded up to a multiple of 20. Q̂'s spread is
  outcome-free, so sizing on it reads nothing of the result. Look 1 stays at half the final blocks.
- **If s̄ < 0.015** the cap binds, and the read at 400 blocks has the power of the table's "cap" row (≈ 190 CPU-h).
  The orchestrator then decides, before any y is read, between that and dropping V-RANK (S-U's status is then
  unchanged). That is a cost decision, made on stage 0's outcome-free numbers only.

**Schedule.** One Kakuna lane in the heavy slots (rule 5b), behind the reads ranked above it (N1-P, H-CTX, D3 look 2,
X59-S). At ≈ 1 effective lane, ≈ 96 CPU-h is ≈ 4–8 days of wall: X51's 8.0 h of play took 18.5 h behind the queue.

---

## 6. Builds (PROPOSED, each default OFF)

| # | build | size (MY JUDGEMENT) |
|---|---|---|
| B-V1 | **The branch player** in `main.anchors`' core slot, beside S-U's `LiveSearcher` (`src/main/anchors/search_side.py`) and sharing its honest-log cut and root guards: at one named decision (turn, decision index) of the deal, check the live tokens against the raw game's, run the shadow arms (S, the self-referee) if asked, then play the forced action; greedy everywhere else. One record per branch decision (the forced action, the tokens, Q̂ and the per-world values, log π, Q_self, an exclusion, CPU). A regime stamp of its own. Tests: the forced action is played; the prefix is byte-identical to the raw game; the shadow S's Q̂ equals a stand-alone `search_decision` on the same log and key | ≈ 0.5 agent-day |
| B-V2 | **The deal unit**: the raw game, the hash draw of §3 from its decision log, and the branches, in ONE process (the model and the peer start once), with G-BRANCH. As a `main.reads` group kind, for the engine stamp, the pin, the queue and resume at deal granularity (`../tools/reads.md`) | ≈ 1 agent-day |
| B-V3 | **The rule** in `main.reads` (§4): the slopes, the deal bootstrap, the classes, the looks, VOID; a test that recovers planted slopes and reads each class at its bar, rule 8 included | ≈ 0.5 agent-day |
| B-V4 | **The preset `oracle_leaf8`** beside `r0s4` in `src/main/anchors/search_presets.py`: `reveal="oracle"`, depth to the end, 8 worlds × 8, values only | ≈ 1 h |

Then the PREREG (§3–§4 verbatim, the pin, the engine, the peer pack) and stage 0.

---

## 7. Predictions (GUESSES, before any branch)

- β_home ≈ 0.45 (0.2, 0.7): E1-0's direction, a non-self referee shrinks what the self-referee sees (0.54).
- β_U ≈ 0.40 (0.1, 0.7); Δβ ≈ −0.05 (−0.3, +0.15). V's ordering of games was intact on U (H-MID); within a node the
  leaf's P\*-as-Kakuna model (S-U A2) pulls U down a little.
- α_top smaller on U by ≈ 1 pp (π's top choice less good where it pilots worse); H ≈ 0 on both.
- The self-referee's slope against Kakuna below 1 on both sets and lower on U.
- Classes: HOLDS 0.45, WORSE 0.12, NULL 0.10 (at look 1: 0.06), UNDECIDED 0.33. S-U OFF ≈ 0.2.
- Stage 0: s̄ ≈ 0.015–0.025 (between the bank and the two Kakuna games); ≈ 2–3 CPU-min a position.

---

## 8. Adversarial self-check

**A1. A slope is not a gain.** The probe shows it: β 0.54 on P\*'s own states, with no search gain. So HOLDS cannot
license S-U, and the doc says so (§0.4, §4.3). What HOLDS buys is that S-U's premise survives F5. That is the
question F5 asked. The read is a filter: it can stop S-U cheaply, and never start it.

**A2. Is the read worth ≈ 96 CPU-h?** As a pure S-U filter it is marginal: P(OFF) ≈ 0.2 × S-U's 270–1,000 CPU-h saves
≈ 55–200 CPU-h in expectation, about its cost. It earns its place on two other counts:
- it is the only within-node ranking read against a non-self opponent;
- its self-referee decomposition is the first measurement of how far the self-play continuation misranks moves
  against Kakuna. Every search on V inherits that question (the search ledger's referee problem; E1).
If S-U will not run in the next month anyway, V-RANK is not urgent (MY JUDGEMENT; the orchestrator rules). At the size
cap (≈ 190 CPU-h) the filter's arithmetic no longer pays on its own; that is why §5.4 hands the cap to the
orchestrator.

**A3. The truth's continuation is P\*, not P\* + S.** One-ply search's improvement is exactly "a better action here,
then π" (the PDL's advantage under π). S-U's searched arm continues with search. To first order the PDL sums π's
advantages along the searched trajectory, and those are what is measured here. The positions are on π's trajectory,
not the searched arm's (first order again).

**A4. One outcome an action.** It is noisy but unbiased for the slope. The common dice cut the within-node noise. The
deal bootstrap carries the correlation of a deal's two positions (they share the raw game). The probe models one
seed per state shared by its actions, which is the live structure.

**A5. Q̂'s own noise attenuates β.** Eight worlds × eight playouts make Q̂ a noisy estimate of its own expectation, and
a noisy regressor shrinks the slope. That is deliberate: the search acts on the noisy Q̂, so the attenuated slope is
the one that decides a deviation. But if Q̂ is noisier on U (worse-fitting worlds, S-U A3), Δβ mixes "V ranks worse"
with "S estimates worse". The per-world values give Q̂'s reliability by set, and the reliability-corrected β is
REPORTED (§4.4).

**A6. The leaks.** S's worlds keep the truth's hidden counters and the seed-reuse speed-tie information (the search
ledger §6.1). These make Q̂ better informed than an honest search, and the truth branches play out that same hidden
state. Both push β up, on both sets, to first order equally (S-U A4). They are S-U's leaks too, so V-RANK measures S
as S-U would play it. The world quality covariate is REPORTED.

**A7. The bars are judgement.** The NULL bar of 0.25 is the break-even slope at the median piKL deviation (α 0.016 /
gap 0.066), taken from the self-play proxy: against Kakuna, α_top may differ. The margin of 0.25 is about half the
proxy's slope. Both are registered before any branch. β's scale is absolute (a share of Q̂'s differences that is
real), so a referee that changes the slope's level changes which class is reached, not what the classes mean.

**A8. The power rests on the proxy's spread and noise** (FINDING V-2). Two banked Kakuna games already show a smaller
Q̂ spread. The sizing rule uses only that outcome-free quantity, and the cap stops it from growing the read past 2×
without the orchestrator.

**A9. Turns 10–30 only.** The read says nothing about the endgame, where S-U's cost explodes (Amendment 1 §1). It
matches Amendment 1's variant A (search turns 10–30) exactly. If S-U registers variant E (search to the end),
V-RANK's scope is narrower than S-U's, and the gate is still a necessary condition for the turns it covers.

**A10. Selection.** A position needs ≥ 2 legal actions (S's own rule) and a game reaching the stratum: 93–97 % of games
reach turn 20 (home72 lowest). S's exclusions select positions whose worlds are hard to fit, and that differs by set
(S-U A3). All three are reported by set. None depends on the outcome.

**A11. Determinism is the read's foundation.** If a branch's prefix differs from the raw game's, the pairing is lost.
The evidence is strong (0 of 2,100 same-engine replays differ), and G-BRANCH checks every branch. Forcing an action
cannot change the prefix: it acts at the position, after the guard.

**A12. Kakuna's answer at the forced turn** is simultaneous: it does not see our forced action, exactly as in a real
game. After it, Kakuna reacts to the deviation as it would to any move. Nothing here models it.

**A13. P72 in pooled U.** P72 is weak and procedural (X51 FINDING 2). β per panel is REPORTED. An equal-weight β_U
pulled down by P72 alone would still read as U, as S-U's G_U would.

**A14. Is the slope the right summary of "ranks correctly"?** A rank correlation is the literal reading, but single
outcomes attenuate it, by amounts that differ with each set's outcome variance. The slope is unbiased under that noise
and has a decision meaning (§4.2's break-even). The contender slope, H and α_top cover the top of the ranking, where
search acts. They are REPORTED, because they cannot reach a game bar (§5.1, reading 2).

**A15. Could V-RANK's answer be read off S-U's own games instead?** No. S-U observes one action a decision (the
searched or the raw one), never the siblings. Its stage 0 has the siblings' Q̂ but no truth for them.

---

## 9. FINDINGS (standing rule 7)

- **V-1.** The probe is a SELF-REFEREE proxy on pool-team states. It sizes the read and motivates §0.4. It is not a
  result about Kakuna or U.
- **V-2.** Q̂'s within-node spread on the 31 banked Kakuna decisions (3 games, 2 deals) is 0.2–0.4× the probe's. Power may be ≈
  1.6× worse than §5.2's main rows. Stage 0 measures it, and the sizing rule acts on it.
- **V-3.** The cost per branch game and the self-referee's cost are UNVERIFIED (X51's in-unit wall per game, and the
  0.3 CPU-s a full playout from another read). Stage 0 measures both.
- **V-4.** No live branch was played and nothing was built. The deterministic-replay premise rests on
  `kakuna_determinism_2026-10-27` and S-U's G-PREFIX design, not on a forced deviation. Whether the front end accepts
  a forced action at a named decision is B-V1's first test.
- **V-5.** The probe's own π logits rank the referee's values NEGATIVELY within a node on average (Spearman −0.09),
  while π's top choice is good (α 1.6 pp). That is descriptive and pool-only. Reading the logits' ordering below the
  top as a ranking would be wrong.
- **V-6.** The banked reconstructed rows (`recon_kakuna.jsonl.gz`) were used only to size the strata (§3). The read
  does not import them: they are on E0.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-27 | **This doc CREATED (DRAFT, explicit-only); the PREREG not ready (builds B-V1–B-V4 first)** (research designer, orchestrator brief; one < 1-CPU-min probe on banked rows, no game) | V-RANK: at 2 hashed positions a deal (turns 10–19, 20–30), every legal action branched as a real P\* vs Kakuna game by deterministic replay; Q̂ from S (`r0s4`) at the position; the within-node slope β on U and Δβ against home72; RANK-NULL-ON-U / RANK-WORSE-ON-U turn S-U OFF, RANK-HOLDS-ON-U makes it ELIGIBLE; 1,200 positions a set, a NULL-only look at half, an outcome-free sizing rule capped at 2×; ≈ 96 CPU-h central; a self-referee arm at the same positions as a REPORTED decomposition | rollouts with a Kakuna proxy (the M1 flaw); natural experiments in banked games (confounded, rare); the X51 twins (not one ply); Kakuna's own choices as the truth (style, not value); a short-horizon V referee (circular); a gain-facing primary (H or IMP: ≈ 10⁵ positions for a game bar); three actions a position (less precision per CPU) | `../research_state/measurements/v_rank_on_u_2026-10-27/power_probe.json`; survey `5d554c23` F5; H-MID `f200258f` and REVIEW `90708bff` (M1, M4); S-U §1.1, §8 A1, §10 and `AMENDMENT_1_DRAFT.md` §1; L1 at P\*; `kakuna_determinism_2026-10-27`; X51 README (cost) |
