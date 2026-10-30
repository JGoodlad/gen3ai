# THE DEVIATION-ONLY LIVE READ: do S′'s deviations from π win more games against REAL Kakuna? — PREREG (2026-10-28)

Written and committed BEFORE any game of this read is played. Source: the S-U gate review's M3 step 3
([`../su_headroom_2026-10-27/REVIEW_gate.md`](../su_headroom_2026-10-27/REVIEW_gate.md), `52f470c21`), licensed by the
leak-reduced re-gate's GO ([`../su_regate_rolls_2026-10-28/README.md`](../su_regate_rolls_2026-10-28/README.md),
`a61375fa7`). Spec: [`read_spec.json`](read_spec.json) (`main.reads`, rule `deviation_live`). Driver:
`main.value_search.live_branch`. Play path of the branches: `main.anchors --our-force` (`main.anchors.forced_branch`).
Pin: `c01ae8ee5` (the commit that lands the forced branch, the driver and the rule).

**Abbreviations.**
- P\* = `rb_x26_s2601` @280,000,170 (sha256 `814823c5…91ed`), greedy. Kakuna = Metamon's Kakuna, checkpoint 34, greedy.
- home72 = 72 pool teams. U = H (`heldout72_hpfix`), F (`forum72`) and P72 at equal weight.
- π = P\*'s greedy policy; π's top = the action it plays.
- S′ = `r0s4_rolls@10`: 8 honest worlds with `dice="rolls"`, TRUNC-4-K8, piKL λ 0.015, run offline through the live
  searcher's `_search` (the re-gate's search).
- A DEVIATION = a state where S′'s pick differs from π's top.
- T = the gate's self-referee (24 truth seeds, P\* on both sides, common dice). It is REPORTED here, never decided on.
- y = a game's result for us: win 1, tie ½, loss 0.
- LB / UB = a 95 % interval's lower / upper bound.
- GUESS = an unmeasured effect. ESTIMATE = a cost from measured rates.

---

## 1. The question

The gate and the re-gate scored S′'s deviations under T, which plays P\* for the opponent: S′'s own opponent model. So a
GO there could mean "S′ fixes π's mistakes on U" or "S′ out-guesses a P\*-shaped opponent that T also plays" (REVIEW_gate
M2). The banked rows cannot separate the two.

**This read asks the real opponent.** At a raw P\* vs Kakuna state where S′ deviates, does playing S′'s pick there,
then P\* greedy thereafter, win more often against Kakuna than the raw game did, where P\* played π's top from the same
state?

## 2. The ground truth: two real games from one state

- **π's arm is the raw game, already played.** Fresh raw P\* vs Kakuna games keep the honest input log of each of our
  decisions in turns 10–30 (`--keep-input-logs`, the gate's recorder).
- **S′'s arm is a FORCED BRANCH** (`main.anchors --our-force`, a `--deal-game G` unit). The deal is replayed from turn 1
  with P\* greedy. At the state, S′'s pick is played; after it, P\* greedy plays on until the game ends.
- **Determinism makes the prefix shared.** On one engine, our play and Kakuna's replay exactly
  (`kakuna_determinism_2026-10-27`: 0 of 2,100 replayed games differ). So the branch reaches the state byte for byte,
  and the play after it is the only difference.
- **G-BRANCH checks this live for every branch**, at the state and before the forced action is sent. These must all
  equal the raw decision's:
  - the battle's index in its series;
  - the sha256 of the honest input log (the START document, which holds both teams, both names and the seed, plus
    every command of both sides so far);
  - the choice tokens;
  - the row's sha256;
  - π's greedy action.
  A mismatch fails the unit by name. Afterwards the branch's capture must also equal the raw game's protocol through
  the branch turn (`main.reads.prefix.compare`).
- **No opponent model anywhere in the truth.** Both arms are real games against Kakuna.

**The continuation's randomness (decided: one continuation an arm, the deal's own battle seed shared).**
- Both arms run on the deal's battle seed. They share every random number until their play diverges: common random
  numbers, as far as the play allows.
- **Several continuations a branch were considered and rejected:**
  1. π's arm is free only as the raw game. A second continuation needs a mid-battle reseed, which the front end does
     not have, so it would be a core change under the read. It would also need π's arm replayed under the same reseed.
  2. At the measured rates it is less efficient. From the re-gate's 469 U deviations under T: the per-deviation spread
     of the true difference is σ_s² ≈ 0.025 and one seed's noise is σ_n² ≈ 0.265. A deviation's fixed cost is
     F ≈ 36 CPU-s (raw share, S′, T) and a branch game c ≈ 28 CPU-s.
     - K continuations an arm cost F + (2K − 1)c and leave σ_s² + σ_n²/K.
     - cost × variance is 18.5 at K = 1, 18.9 at K = 2 and 19.9 at K = 3.
     - So the games buy more as more deviations than as more continuations.
- **The noise this leaves is handled by n** (§6), never by a re-roll.

## 3. The states, the search, the branches

**Raw games.**
- P\* vs Kakuna, greedy both sides, whole 20-game role-balanced units with `load.input_log = "10-30"`.
- Group `u` plays H, F and P72 on unit seeds **200–244**. Group `home` plays home72 on unit seeds **200–224**. These
  seeds are fresh: they are outside S-U's 0–59 and the gate's 100–109.
- The peer's team pack is `gen3_peer_showdown_pack_v1`, stamped per cell.

**ONE engine:** `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`.
- It is the gate's and the re-gate's engine, re-stamped at this registration's tree (`main.anchors.server.engine_stamp()`).
- It is not S-U's registered `bdc5d2be…`. An S-U amendment that cites this read registers this engine.
- Every counted game of this read, raw and branch, is on it, or the read is VOID. The offline scorer's stamp is also
  recorded (`engine.txt`) before its first chunk.

**The draw** (`live_branch select`: the gate's `sign_gate.select`, outcome-blind, written once after the raw lane
ends).
- Per raw game: the 2 kept decisions (turns 10–30, ≥ 2 legal actions) with the smallest
  `sha256("sugate_v1|<deal key>|<turn>|<decision index>")`.
- Counted raw units only.

**S′ and the quota** (`live_branch score`).
- **The registered order** is, per panel: unit seed, then game index, then draw hash.
- S′ runs on each state in that order, exactly as the re-gate ran it (`sign_gate.GateScorer.score`):
  - the root guard (the kept log must replay to the decision's tokens and the live row's sha256);
  - S′'s `_search`;
  - T on a deviation;
  - the Speed ±1 marker.
- **A panel stops at its QUOTA of S′ deviations: H 600, F 600, P72 600, home72 300.** That is ≈ 1,800 U deviations, the
  review's size.
  - The quota is a count of S′'s deviations, and it reads no outcome.
  - A state where S′ finds no consistent world is an exclusion: π plays, nothing is branched.
  - A panel whose raw seeds run out before its quota reads what it has. The shortfall is reported; nothing re-sizes.

**The branches** (`live_branch plan`, written once).
- One forced-branch unit per scored deviation, forcing S′'s pick.
- **One NULL control per (panel, raw unit):** π's own top forced at that unit's scored state with the smallest draw
  hash. A null must replay its raw game WHOLE, byte for byte, with the same result and length.
- The units play with `python -m main.x35.kakuna run --state <branch state>`: one-game units on the heavy slots' short
  lane, 95xx ports with claims, `kakuna.MAX_ATTEMPTS` (2) attempts each.

## 4. The rule (`deviation_live`; the spec's `decision` block)

- **D_i = y(S′'s branch_i) − y(the raw game of i)**, per deviation whose branch counted (`results.unit_counts`, with
  the forfeit-win guard).
- Per panel: the mean of D. **Pooled U = the mean of H's, F's and P72's** (equal weight).
- **Interval:** a game-clustered percentile bootstrap (`sign_gate.cluster_ratio_boot`).
  - One `numpy.random.default_rng(20261028)`, B = 2,000, panels resampled in the order H, F, P72; 2.5 / 97.5
    percentiles.
  - The cluster is the raw game: its two states share y(raw).

| outcome | rule | what follows |
|---|---|---|
| **GO** | LB(pooled U) > 0 + 0.0005 | **S′'s deviations beat π against the real opponent on U.** It licenses the **amended S-U**: a live searched-play read, S′ playing every decision from turn 10 against Kakuna, paired by deal with the raw game, on this engine (`design_search_u_headroom.md`; S-U REVIEW B1's items, the M3 / M4 amendments, S′ in place of S). It also retires V-RANK's all-sibling slope as S-U's filter. It licenses no gain of any size: a decision-level contrast says nothing about how deviations compose over a game. |
| **STOP** | otherwise. An LB within 0.0005 of 0 reads STOP, flagged `boundary` (rule 8) | No live S-U read on S′. Search stays paused for goal (b). Recorded as: "S′ at P\* is not shown to improve on π against Kakuna on U at ≈ 1,800 deviations." DOES-NOT-CLOSE-B is NOT claimed, and "S′ is no better than π" is NOT claimed (a null is a power statement, §6). |

**VOID** (named, before any estimate; each is one of the rule's or `live_branch.void_reasons`' reasons).

The raw arm:
- the counted raw units span two engines, an unknown one, or one other than registered;
- a counted raw unit on other bytes than P\*'s sha256, with a search stamp, or with a regime other than greedy.

The draw and the scoring:
- a selected state from an uncounted cell, with a wrong draw hash, or more than 2 from one game;
- a scored state never selected, or scored twice;
- the scored rows are not each panel's registered-order prefix up to its quota;
- a searched row not on `r0s4_rolls`.

The branches:
- **G-BRANCH** failed in any attempt of any unit;
- a counted branch's capture differs from its raw game's protocol before the branch turn;
- **a null control** whose whole protocol, result or length differs from its raw game's;
- a branch that played another action than S′'s scored pick, or not at its raw game's series index;
- a counted game on another engine;
- a deviation without its branch, or a branch without its deviation;
- **more than 10 % of a panel's branches DROPPED** (never counted after 2 attempts; Kakuna's turn-250 fault, SOP H5,
  is the known cause).

Below 10 %, drops are excluded and reported by panel and cause. A drop is outcome-correlated only through game length.

## 5. The pre-play tooth (run at the pin before any game of the read)

`scripts/branch_tooth.py` forces branches of the S-U **gate's** banked raw games: unit seeds 100–109, outside this
read, on this engine, through this read's own code (`plan_branches`, then `main.x35.kakuna run`).
- **Branches:** three real S′ deviations (the re-gate's scored picks) at games 0, 2 and 10 of two units, so both halves
  of a series are covered.
- **Nulls:** two NULL controls.
- **PASS iff** every unit counts, G-BRANCH passed live, each branch's protocol equals its raw game's through the branch
  turn, and each null replays its raw game whole with the same result and length.
- A PASS also verifies the SOP's UNVERIFIED note on this engine: a lone `--deal-game G` unit reproduces game G of a
  whole 20-game unit.
- **Its development run on this tree passed** (§9). A FAIL stops the read before play.

## 6. Power (normal approximation; inputs measured on the re-gate's S′ rows)

- **The effect if Kakuna behaved like T:** S′'s mean T(pick) − T(top) per U deviation is **+0.034** (H +0.048, F
  +0.025, P72 +0.030; 469 deviations).
- **The noise:** one seed separates the two actions on 28–31 % of U deviations, so Var(D) ≈ 0.29 per deviation.
- **The clustering:** two states of a game share y(raw). The within-game correlation of D is ≈ 0.3 (MY JUDGEMENT:
  flips share their sign), and ≈ 40 % of deviations share a game. That gives a design effect ≈ 1.12.
- **SE(pooled U) ≈ √(0.29 / 1,800) × 1.06 ≈ 0.0135.**

| true pooled-U mean D | z | P(GO) |
|---|---|---|
| +0.034 (Kakuna = T) | 2.5 | ≈ 0.71 |
| +0.017 (half of T's) | 1.3 | ≈ 0.24 |
| 0 | 0 | 0.025 (the false GO) |

- **A STOP is therefore weak evidence of no effect** below about half T's. This is the review's z ≈ 2.8, less the
  clustering.
- The read cannot size an S-U gain: S-U's game-level bar (≈ +5–7 pp over ≈ 30 searched decisions) is ≈ 0.002 a decision
  through the performance-difference lemma, far below this read's resolution. That is why a GO licenses a live
  searched-play read rather than predicting its result.

## 7. Cost and schedule (ESTIMATE from measured rates; cost never re-sizes the read)

| item | rate (source) | count | CPU-h |
|---|---|---|---|
| raw games | 9 CPU-s a game (the gate's 20-game units, median 181 CPU-s) | U 2,700 + home72 500 | ≈ 8.0 |
| S′ + the Speed marker | 4.2–6.1 + 0.5 CPU-s a state (re-gate) | ≈ 5,240 states (quota ÷ deviation rate 0.43 / 0.39 / 0.40 / 0.38) | ≈ 8.0 |
| T on deviations (REPORTED) | ≈ 11.5 CPU-s (re-gate) | ≈ 2,100 | ≈ 6.7 |
| branch units | ≈ 33 CPU-s a one-game unit (§9's smoke: 29–66, median 33.7; S-U stage 0's raw per-game units: 24–31) | ≈ 2,100 + ≈ 130 nulls | ≈ 20 (18–25) |
| **total** | | | **≈ 43 (37–52)** |

- **The branches' start-up dominates their cost.** A one-game unit loads P\* and starts a Kakuna process for one game
  (FINDING DL-2).
- T costs ≈ 6.7 CPU-h. It is kept as the M2 bridge: on the same deviations, does T's verdict agree with Kakuna's?
- **Wall time:**
  - the raw lane: 160 twenty-game units at ≈ 4–12 min of wall each on one heavy slot, ≈ 11–30 h;
  - scoring: ≈ 15 h in 30-min chunks;
  - the branches: ≈ 2,230 short-lane units, serial. A unit plays in ≈ 1–2 min, but each waits FIFO for a heavy slot
    (the smoke's waited 4–20 min behind the queue), so the lane's wall is set by the queue: ≈ 40–80 h at today's load.
- **How it runs** (rule 5b): `nice -n 19` on the agent cores (`taskset -c 5-7,13-15`), `OMP_NUM_THREADS=1` (the gate's
  G-4), under `mem_cap.sh`.
- **The order: it queues behind X59-S, X48 and D3 look 2**, the orchestrator's queue.

## 8. Reported, never decided

- **home72's mean D**, and **U − home72** (one bootstrap stream, U panels then home72).
- **The live sign accuracy** P(D > 0 | D ≠ 0), pooled U and home72: the gate's statistic, read live.
- **T against Kakuna on the same deviations (REVIEW_gate M2):**
  - T's mean IMP;
  - live D − T's IMP, pooled U and home72;
  - the agreement of T's sign with the live sign where both split.
  - M2's "S′ exploits a P\*-shaped opponent" predicts the gap live − T to be more negative on U than on home72.
- **The strata** (re-gate §5): the Speed-pinned stratum against the rest, and leak-exposed against leak-clean, pooled U.
- **Per panel:** branches planned, counted, dropped and split; nulls counted; games; branch CPU; quota met / short;
  the deviation and exclusion rates.

No outcome is printed while the read runs: the scorer prints progress counts only, and the branch lane writes the
cells.

## 9. The build's smoke (before this PREREG; outside the read)

All on this build's tree, on the registered engine, CPU only (nice 19, the agent cores, `OMP_NUM_THREADS=1`).
- **The forced branch on real in-process battles** (`src/main/anchors/forced_branch_integration_test.py`, slow). A branch
  shares the raw game's protocol through its turn and plays its forced action. A null replays the raw game whole, byte
  for byte. A tampered log digest fails by name before any action.
- **The pre-play tooth on the gate's banked Kakuna games** (`scripts/branch_tooth.py`, state
  `~/gen3ai_archive/dlive_smoke_2026-10-28/`): **PASS**, 5 of 5 units.
  - Three real S′ deviations: H unit 100 games 0 and 2, P72 unit 101 game 10. Each passed G-BRANCH and has the raw
    protocol through its branch turn.
  - Two nulls: H game 0, and P72 game 10, in the series' second half. Each replays its game of the raw 20-game unit
    whole: the same protocol, result and length (57 and 23 turns).
  - So a lone `--deal-game G` unit reproduces game G of a whole unit on this engine. Every unit's battle tag was
    `battle-gen3ou-1` (DL-1).
  - Unit CPU: 66.4, 35.4, 32.0 and 29.0 CPU-s (the fifth unit's end event was lost when the smoke's own 55-min
    supervisor bound expired while it queued; it then played and counted).
  - Wall: 6–22 min a unit, almost all of it the heavy-slot queue.
- **The scorer** (`live_branch score`, real P\*, 6 gate states of home72 unit 101, quota 1). It scored in the registered
  order and stopped at the first deviation (the 6th state). It reproduced the re-gate's rows exactly: picks, Q̂ and,
  on the deviation, T's 24 outcome strings. S′ took 3.5–6.9 CPU-s a state.
- **The capture tooth's boundary** (read-only, the gate's 1,560 kept decisions). In our side's protocol the line
  `|turn|<the decision's turn>` comes before the decision's request in 1,560 of 1,560. So "the protocol through
  `|turn|T`" never includes the branch's own action, and a branch's capture check cannot fail on the deviation itself.

## 10. Predictions (GUESSES, before any game)

- Pooled-U mean D ≈ +0.012 (−0.01 to +0.035). E1-0's direction is that gains shrink against a real opponent, and M2's
  opponent-model share of T's effect is unknown.
- home72 ≈ +0.008.
- Live − T on U ≈ −0.02, more negative than on home72.
- **P(GO) ≈ 0.3.**
- G-BRANCH: 0 failures. Nulls: all identical. Drops < 2 % a panel.

## Registration

- Pin `c01ae8ee5` (`check-spec`: `play_path_differs_at_head` 0; the spec's sha256 `9821b331…6851`); engine `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`; preset `r0s4_rolls@10`.
- State `~/gen3ai_archive/dlive_read_2026-10-28/`; the artefacts are under `…/live/` (`selected.jsonl`,
  `scored.jsonl`, `branches/`).
- The order:
  1. a clean detached worktree at the pin, bootstrapped `--with-rust` (DL-4); the pre-play tooth (§5) there;
  2. `main.reads plan` and `run --lane anchor`, from a clean detached worktree at the pin;
  3. `live_branch select` (once), then `score` (chunks, `--budget-s 1800`), then `plan` (once);
  4. `main.x35.kakuna run --state …/live/branches`;
  5. `main.reads record`.
- Committed with [`read_spec.json`](read_spec.json) before any game. Written by the deviation-only live read's designer
  (orchestrator brief, 2026-10-28).

## FINDINGs

- **DL-1.** A one-game unit's battle TAG is a per-process counter (`ws_frontend._BATTLE_SEQ`): game G of a lone
  `--deal-game G` unit is `battle-gen3ou-1`, while game G of a 20-game unit is `battle-gen3ou-<G + 1>`. G-BRANCH compares
  the series index, never the tag. The build's first integration run caught this.
- **DL-2.** The cost runs above the review's ≈ 20–25 CPU-h. One-game branch units pay a start-up each, and T is kept as
  a REPORTED arm. A series-batched branch unit (one process forcing one deviation in each game of a raw unit, the
  others replaying as nulls) would save an ESTIMATED ≈ 10 CPU-h. It is not built.
- **DL-4.** `engine_stamp()`'s env scope hashes the gitignored `src/rust_env/Cargo.lock`. A fresh worktree that was not
  built `--with-rust` therefore stamps `env=f2918a06…/128` for the same tracked sources (this build's worktree did,
  until the lock was present). The play and scoring checkouts must be bootstrapped `--with-rust`. `live_branch score`
  takes the registered `--engine` and refuses another before any state, and the rule VOIDs on a counted game's engine.
- **DL-3.** T shares S′'s opponent model, so T's agreement with Kakuna is descriptive. It does not validate T as a gate
  for other reads unless it holds on both U and home72.

## Amendment A1 (2026-10-30, before any game of the read): how the read is operated

Written by the read's operator before the tooth or any raw game. It changes no rule, no quota, no seed, no estimand and
no VOID reason; it registers the operating choices the PREREG left open.

- **One checkout** serves play, scoring and the branch lane: the detached worktree
  `/home/goodlad/dev/gen3ai-reads/dlive_play_c01ae8ee` at the pin, bootstrapped `--with-rust` (DL-4). Its
  `engine_stamp()` is the registered `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129` (checked 2026-10-30
  14:33). The operator's pipeline (`<state>/ops/pipeline.sh`) re-checks the pin, a clean tracked tree and the engine
  before the tooth, before the raw lane, before every score chunk and before the branch lane, and HOLDS on a mismatch.
- **The order** is the PREREG's (Registration), after X59-S's Δ_restriction lane exits (the orchestrator's queue). The
  tooth's state is `<state>/tooth/`; its FAIL stops the read before any raw game.
- **The raw lane** runs at `--concurrency 1` (X63 Stage A holds the heavy slots' priority), `--hours 72`. A lane that
  stops with units neither counted nor exhausted is resumed (`main.reads resume`): a new supervisor life, with
  `MAX_ATTEMPTS` per life as in every read. `select` runs once, after the lane ends.
- **The scorer** reads the plan's verified copy `<state>/ckpts/pstar/checkpoint_280000170_steps.zip`, re-hashed to the
  pinned sha256 (`814823c5…91ed`) before every chunk. Chunks run as the re-gate's did (G-4): one torch / BLAS thread,
  `chrt -i 0 nice -n 19 taskset -c 5-7,13-15 mem_cap.sh 8 timeout 2700`, `--budget-s 1800`, the registered `--quota`
  and `--engine`. A chunk killed by its bound (rc 124) is retried; three in a row HOLD. A state interrupted mid-score
  is never written, so it is scored whole by the next chunk.
- **The branch lane** is ONE `main.x35.kakuna run` supervisor life with no stop epoch. A dropped unit is `collect`'s
  (no counted cell once it has `MAX_ATTEMPTS` `end` events), code at the pin. Should the lane need a restart (a crash,
  or a pause the orchestrator or X63's operator asks for), the supervisor is stopped by PID and restarted only after its
  in-flight unit's child has exited (a restart moves a live cell aside). A restart gives each uncounted unit
  `MAX_ATTEMPTS` more (the queue counts per life); any unit counted on a third or later attempt is listed in the record
  by panel as a FINDING.
- **The record** is written by `main.reads record` from the same checkout.
