# S′-LIVE: does S′ playing every decision beat P\* against the REAL Kakuna on unfamiliar teams? — PREREG (2026-11-01)

Written and committed BEFORE any game of this read. Design: [`design_sprime_live_play.md`](../../../endstate/design_sprime_live_play.md).
Spec: [`read_spec.json`](read_spec.json) (`main.reads`, rule `searched_play`). Pin: `0c76b521e` (the commit that lands
the builds of §6). State: `~/gen3ai_archive/sprime_live_play_2026-11-01/`.

**Abbreviations.**
- P\* = `rb_x26_s2601` @280,000,170 (sha256 `814823c5…91ed`), greedy. Kakuna = Metamon's Kakuna, checkpoint 34, greedy.
- S′ = `r0s4_rolls@10`: 8 honest worlds with `dice="rolls"`, TRUNC-4-K8, piKL λ 0.015 (`main.anchors.search_presets`).
- U = H (`heldout72_hpfix`), F (`forum72`) and P72 at equal weight.
- E4 = the engine `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`.
- dlive = the deviation-only live read ([`../dlive_read_2026-10-28/`](../dlive_read_2026-10-28/README.md), record `02fb0f2fc`).
- y = a game's result for us: win 1, tie ½, loss 0. D = y(searched) − y(raw) for one deal. G_U = §5's estimand.
- LB / UB = a 95 % interval's lower / upper bound. pp = percentage points.
- GUESS = an unmeasured effect. ESTIMATE = a cost from measured rates.

---

## 1. The question

**Does P\* + S′, searching every eligible decision from turn 10, win more games against Kakuna on U than P\* alone,
on the same deals?** The truth is two real games per deal against the real opponent. No referee and no opponent
model enter it.

| class | rule | what it licenses | what it does NOT license |
|---|---|---|---|
| **GO-LEAKY** | LB(G_U) > 0, with every post-treatment unpaired deal imputed at its worst case for GO (§5) | Only this: S′ WITH the builder's true-seed information (§7) beats P\* on U against Kakuna. It licenses building and registering a leak-honest replication on fresh deals, and that work alone is un-paused. If the searched arm's pooled-U score clears 0.50, it is quoted at most as "GOAL-B-REACHED (leaky)". The flag `go_below_margin` (UB < +3 pp) is effectively unreachable at the expected d (§9) | anything about deployable search; "goal (b) met"; any training, league or public-server change; a size claim beyond the interval |
| **NULL** | not GO-LEAKY, and UB(G_U) < +3 pp, with the opposite imputation | S′ as built adds less than 3 pp on U (95 %). It cannot carry goal (b) alone at any practical n (§5's margin). The search ledger records S′ at P\* as **closed as a stand-alone route** to goal (b), with [LB, UB] entered beside the other levers. Creditable under A-LEAK (§7) | excluding a gain of ≤ 3 pp that would stack with other levers; any claim about other presets, a Kakuna-modelled leaf, deeper search or home72 |
| **STOP** | otherwise (or a bound near its bar) | Nothing: unresolved at the registered n. Search stays paused for goal (b) | a re-run at a larger n without a new registration |

A bound within 0.0005 of its bar is never a crossing (root rule 8): that class does not hold, and the record is
flagged `boundary`.

## 2. Why this is a new registered question, not a re-read of dlive

**What dlive registered.** dlive's §4 says a STOP means: "No live S-U read on S′. Search stays paused for goal (b)."
It recorded exactly that on 2026-11-01. This read IS a live searched-play read on S′, so it is held to that line
honestly:

- **The consequence was a LICENCE decision, and its licence is not used here.** dlive's GO would have licensed the
  *amended S-U*: S-U's design (home72 and U, the interaction I, S-U's classes) with S′ in place of S. That licence was
  not granted, and this read does not claim it. S-U stays unplayed on S′. This read is not S-U: no home72, no I, its
  own classes, its own rule.
- **This read is licensed by a new ruling**, the orchestrator's brief of 2026-11-01, on dlive's own FINDING DL-10:
  - dlive assumed the re-gate's +0.034 per departure, so it had ≈ 0.71 power;
  - the self-referee re-measured S′'s effect on dlive's fresh states at +0.0099 [+0.0014, +0.0193];
  - at that size dlive's power was ≈ 0.11.
  A filter with 0.11 power cannot carry the decision it was meant to gate. The game-level question it gated is
  goal (b)'s own, and it is still open.
- **It is a different estimand, and dlive could not measure it.**
  - dlive measured ONE forced departure followed by P\* greedy, at turns 10–30 only.
  - This read measures the TOTAL effect of S′ deciding every eligible decision of a game. That includes how
    departures compose, S′'s own state distribution, and every turn after 30, which holds ≈ 40 % of a mean game's
    searched decisions (§8).
  - The bridge between the two (the performance-difference identity) is UNVERIFIED, and this read does not test it.
- **It plays new outcomes on NEW deals.**
  - Every game is new. The deals are fresh: U unit seeds **300–344**, which no prior Kakuna read has played.
  - dlive's branch and raw outcomes enter nothing here. dlive's raw arm supplies only the cost model's game lengths
    and the E4 raw score behind §5's margin context.
- **The forking-paths cost, stated** (review M1). This read exists partly because dlive's point estimate leaned
  positive (+1.2 pp a departure).
  - **Why not dlive's deals.** The brief allowed them. On those deals a searched game follows the raw game until S′'s
    first departure, so on ≈ 200 of them it would replay a dlive branch's first move and its seed luck. Every deal
    would also carry dlive's deal-level luck: H's +4.2 pp there is partly that. That correlation would have shifted
    this read's estimate toward dlive's lean, by ≈ 0.1–0.2 pp, and raised the false-GO rate from 2.5 % to ≈ 3 %.
  - **Fresh seeds remove it, at no cost**, because the raw arm is re-played anyway. This read's nominal error rates
    (§9: 2.5 % false GO-LEAKY, 2.5 % false NULL at +3 pp) are its own.
  - **What the selection still means:**
    1. **No pooling.** A GO-LEAKY here is never pooled with, or "confirmed by", dlive's estimate.
    2. **dlive's record stands as written:** "S′ at P\* is not shown to improve on π against Kakuna on U at ≈ 1,800
       deviations". It is not re-read or re-classed.
    3. **The filter-then-read pipeline was overruled for power.** The search ledger's decision record says so, so a
       later reader does not mistake this read for one dlive licensed.

## 3. The arms, the deals, the pairing

- **P\* + S′** (`pstar_s`, `load.search: r0s4_rolls@10`). Which decisions S′ makes:
  - S′ decides every one of our decisions at turn ≥ 10 with ≥ 2 legal actions (`main.anchors.search_side`).
  - Every other decision is P\*'s greedy choice.
  - A decision whose honest builder finds no consistent world in its tries is an EXCLUSION: P\* plays it, it is
    recorded, and it is part of S′ as defined.
  - Any other search error fails the game: a named defect, never a fallback.
- **P\* raw** (`pstar`), greedy, the same pinned bytes (`spec.search_twins`).
- **The opponent and server.** Kakuna, greedy, peer pack `gen3_peer_showdown_pack_v1`. Our `ws_frontend` serves on
  a 9XXX port, through the `main.reads` anchor lane.
- **The deals:** U unit seeds **300–344** on H, F and P72, 20 deals a seed. That is **2,700 deals, 900 a panel**,
  all fresh (§2). home72 is not played (design §7).
- **One `per_game` group plays both arms**, deal by deal: game g's raw unit, then its searched unit, on every panel,
  before game g + 1. Unit seeds are the outer loop, so a stopped queue stays balanced by seed and panel.
- **Every unit is ONE game** (`main.anchors --games 20 --deal-game g`): game g's role, both team draws and its battle
  seed, alone in a fresh process.
- **Why both arms are lone one-game processes.** In dlive's FINDING DL-7, Kakuna sometimes chose differently in a
  lone replay than in game g of a 20-game process, on a byte-identical view: 9 of 2,100 branches. Every lone replay
  agreed with the other lone attempt. With both arms lone, that asymmetry is not built into the pairing. (dlive's
  raw arm could not have been imported anyway: a searched group imports nothing, and its deals are excluded by §2.)
- **The shared prefix.** Before turn 10 both arms are the same deterministic play on the same seed and teams.
  G-PREFIX requires each pair's protocol (`|t:|` removed, both sides, and the `START` document) to be
  byte-identical through `|turn|10` (§6).
- **After turn 10** the arms differ by design. GUESS: any residual Kakuna nondeterminism there adds variance, not
  bias. Its mechanism is unknown (DL-7), and the searched arm's Kakuna process sees long idle waits. Any bias is
  bounded by near-tie flips and is ≪ 1 pp.

## 4. The engine, the bytes, the checkout

- **One engine: E4.** It is the registered engine of the gate, the re-gate and dlive, re-stamped at this
  registration's tree. Every counted unit of both arms is on it, or the read is VOID.
- **What differs from dlive's pin `c01ae8ee5`.**
  - Unchanged: `main.value_search` (the search itself) and the Rust sources, so the engine stamp is the same E4.
  - Changed on the play path between `c01ae8ee5` and `02fb0f2fc`, before this build (`main.reads.playpath`), ten
    files:
    - seven in `src/agents/model/`: `canary_param_null.py` (new) and its test data, `canary_row_bar.py`,
      `compile_canary.py`, `compile_regions.py`, `growth.py`, `trunk_depth.py`, `trunk_ffn_width.py`;
    - `src/main/reads/__init__.py`;
    - `src/utils/team_corpora/__init__.py`.
    They are compile-canary, growth and corpus code, meant for training and new architectures.
  - This build adds L1's constant, L2's record fields, and `src/main/reads/__init__.py`'s docstring.
- **That none of them moves P\*'s forward or S′ is MEASURED, not argued: the BANK TOOTH** (`live_identity
  bank-tooth`, run at the pin before registration, §13). S′ was re-run at the pin on a hashed sample of dlive's
  selected states, and its `search` block had to equal dlive's offline `scored.jsonl` row exactly (searched or
  excluded, the pick, Q̂, log π, root V). So the pin's S′ is dlive's S′. The read's identity guard (§6) then shows
  that the live S′ is the pin's S′.
- **The checkout** is a clean detached worktree at the pin, bootstrapped `--with-rust`. That is DL-4: the env stamp
  hashes the gitignored `Cargo.lock`. Its `engine_stamp()` is checked against E4 before the first unit and before every
  `resume` (§12). The runner itself checks only the commit, so this is an operator step. It also fails closed: every
  unit stamps its engine, and the rule VOIDs on any other.
- **One host.** Every unit of both arms plays on this box.
  - A rented box is a new host: torch's numerics differ across CPU types, and that can split a near-tie in our side
    or Kakuna's.
  - It needs an amendment registered before its first game. Under it, every remaining unit of both arms plays on
    that host, and the identity guard's offline re-score (§6) runs there too. A host's units are never pooled with
    this box's without a REPORTED cross-host tooth.

## 5. The rule (`searched_play`; the spec's `decision` block)

- **Pairs.** A deal is a PAIR iff both arms' one-game units COUNTED (`results.unit_counts`, the forfeit-win guard
  included).
- **Unpaired deals** are counted apart, by kind:
  - PRE-TREATMENT: G-PREFIX peer drops, which happen before turn 10;
  - POST-TREATMENT: every other deal that finished without a pair, i.e. only one side counted, or neither did,
    after its attempts. Timer kills, a recurring search defect and peer faults after turn 10 are all this kind.
- **Pre-treatment drops are excluded.** Post-treatment drops enter the classes IMPUTED at their worst case for each
  class, inside their unit's statistic like a pair (review M2):

  | missing side | for GO-LEAKY's LB | for NULL's UB |
  |---|---|---|
  | searched | D = −y_raw | D = 1 − y_raw |
  | raw | D = y_s − 1 | D = y_s |
  | both | D = −1 | D = +1 |

- **The record reports three intervals:** paired-only (descriptive), GO-imputed and NULL-imputed. Each bootstrap
  starts from a fresh `default_rng(20261101)`.
- **The unit statistic** is the mean D over a (panel, unit seed)'s pairs, in exact fractions.
- **The panel estimate** G_panel is the mean of its unit statistics.
- **G_U** is the mean of G_H, G_F and G_P72.
- **The interval** is a percentile bootstrap over units within each panel:
  - `rng = numpy.random.default_rng(20261101)`, B = 2,000;
  - panels in the order H, F, P72;
  - for each, `idx = rng.integers(0, n, size=(B, n))` over its unit statistics in unit-seed order, and replicate b
    is the mean of `x[idx[b]]`;
  - G_U's replicate b is the mean of the panels' replicate b;
  - the bounds are `numpy.percentile(reps, [2.5, 97.5])`.
  It is S-U's scheme without home72.
- **The classes** are §1's: GO-LEAKY on the GO-imputed LB, NULL on the NULL-imputed UB. They are compared as exact
  fractions with `rule8_eps` 0.0005.
- **The margin, +3 pp, is fixed by goal (b):**
  - On E4, raw P\* scored **0.486** on pooled U against Kakuna (dlive's 2,700 raw games: H 0.472, F 0.491, P72 0.494).
  - Goal (b) needs a pooled-U score with LB > 0.50. At this n that is a point estimate of ≈ 0.519, i.e. a gain of
    **≈ +3.3 pp**.
  - So a gain whose UB is below +3 pp cannot by itself take P\* to goal (b) at a practical n. At +2 pp, reaching
    LB > 0.50 would take ≈ 54,000 games.
  - S-U's 4.3 pp was the same gap at goal (b)'s usual n of 1,200, from X51's 0.487.

**VOID** (named reasons, before any estimate; `rules_searched_play.void_reasons`).
- **G-PREFIX:**
  - it was not run on the group;
  - a mismatch that the registered classifier (§6) does not class as a PEER DROP;
  - G-PREFIX checked a different number of deals than the pairs plus the peer-dropped deals.
- **Engines:** counted units on two engines, an unknown one, or one other than E4.
- **Bytes:** any counted unit or resolved model on bytes other than P\*'s sha256.
- **Regime:**
  - a counted searched unit not stamped `greedy+search=r0s4_rolls@10`, or not played with that preset's config;
  - a counted raw unit carrying a search stamp, or a regime other than greedy.
- **Results:** a counted unit whose `games.jsonl` result disagrees with its summary.
- **Units:** a panel with fewer than 2 units holding a pair.
- **Unfinished deals:** any deal whose units are neither counted nor out of their attempts. The read is incomplete;
  this is never a drop.
- **Fail-closed, and not a rule reason:** a unit that exhausted its attempts with no summary at all (a SIGKILL, or
  an OOM at the 12 GB cap) leaves the group `unplayed`. `record` then refuses to write, because the spec registers
  no partial read. The operator re-runs that deal's units under a new supervisor life. If the deal still cannot
  play, the read is stopped and reported, never decided.
- **Drops,** in any U panel:
  - pre-treatment drops above **2 %** of its 900 deals;
  - post-treatment drops above **0.5 %**.
- **The identity guard (§6):**
  - it FAILs;
  - it re-scored fewer than 300 decisions;
  - its output does not cover exactly the registered-order first 600 sampled decisions of the cells (or all of
    them, if fewer).

Below the caps, unpaired deals are handled as above and reported by panel and cause.

## 6. The guards and the teeth

**G-PREFIX and the pre-registered peer drop.**
- **The built behaviour.** G-PREFIX (`main.reads.prefix`) compares each pair as soon as both units count. A mismatch
  HOLDS the lane (`by: G-PREFIX`).
- **The classification.** The rule classes each mismatch with AM-DL-PEER's classifier, moved into
  `main.reads.peer_divergence` unchanged. A test asserts its output equals the dlive script's on every synthetic case.
  The branch turn is 10:
  - **`peer_first`** (Kakuna's first different command answers a request on a view identical in both games, every
    one of our commands on an identical view is equal, and the divergence turn is < 10): a **PEER DROP**. The deal
    is excluded and counted per panel.
  - **anything else** (`ours_first`, `unattributable`, an unreadable capture): **VOID**.
- **The operator's step.** On a hold, the operator runs the read-only, outcome-blind
  `python -m main.reads.peer_divergence prefix --spec SPEC --state STATE --group u`. It prints each mismatch's deal,
  class and divergence turn, never a result.
  - For a `peer_first` mismatch, the operator releases the lane with
    `python -m main.reads resume --state STATE --lane anchor --by <who> --why "peer_first <deal>"`.
  - Any other class stops the read as VOID.
  - `release-guard-hold` is a different hold (`decide`'s re-derived guards) and is never used for G-PREFIX.
- **The GUESS on the rate.** It is ≈ 0: lone-vs-lone, every DL-7 unit's two lone attempts agreed. It is reported
  per panel.

**The choice-level identity guard** (`main.value_search.live_identity rescore`, L3). The S′ that PLAYS must be,
decision for decision, the S′ that the offline scorer runs: the scorer path of the re-gate and dlive, with the
search code unchanged since dlive's pin.
- **The sample.** A live searched or excluded decision is in the sample iff
  `int(sha256(key)[:8], 16) % 16 == 0`, with key = `su_live_v1|<deal>|t<turn>|n<n>`. That choice is outcome-blind.
  A sampled record also keeps its honest log and root (L2).
- **The re-score.** After the lane ends, the first **600** sampled decisions, in the registered order
  (sha256(key) first, across all cells, so they spread evenly over panels and over the weeks of play; then panel,
  unit seed, game and attempt dir as tie-breaks), are re-run OFFLINE at the pin with one torch thread (G-4), through
  `sign_gate.GateScorer`'s root guards and the live searcher's `_search`. Every attempt dir counts, partial ones
  included.
- **What must be equal, exactly as recorded:** searched-or-excluded, the pick, Q̂, log π and root V.
- **PASS iff** ≥ 300 decisions are re-scored and none differs. The expected count is ≈ 6,800 sampled (40 decisions a
  game ÷ 16 × 2,700), so 600 are re-scored, at ≈ 2 CPU-h.
- **What it covers.** Every turn, the late game included.
- **It reads no result.** It prints counts only.
- **What a FAIL means.** A FAIL is a VOID: the live player is not the measured S′. Possible causes are thread count,
  the engine, or the search code.
- **The cross-check against dlive itself** is the BANK TOOTH (§4, §13): the pin's offline S′ equals dlive's offline
  rows. The smoke's `live_identity check` matched none, because its game departed before the bank's states (§13).
  So it adds nothing.

**The cost checkpoint** (operational, never a stop on outcome).
- **When:** the queue's first 120 units, i.e. unit seed 300's 60 deals on all three panels, both arms
  (`run --max-units 120`).
- **What the operator reads:**
  - `main.reads search-cost` (CPU, decisions, exclusions, deviation rate, G-PREFIX);
  - the identity guard on the cells so far;
  - nothing else.
- **What follows:**
  - The lane HOLDs if:
    - the identity guard's re-score of the checkpoint's sampled decisions FAILs;
    - G-PREFIX holds;
    - any searched unit used up its attempts uncounted. That is a recurring defect: it is root-caused from its
      error class, never its result;
    - the projected worst wall time of a game (its slowest wall per turn × 250 turns) exceeds the 10-h watchdog.
  - If the mean searched game costs > 2× the model (> 19 CPU-min), the orchestrator decides between continuing
    locally and the rental (§8).
  - The size never changes.

**The watchdog (L1).**
- A searched unit is bounded at **10 h** of wall (it was 3 h).
- At the cost model's p99–max (59–204 CPU-min) and 1–2.5× wall under contention, 3 h would have killed the longest
  ≈ 0.3–1 % of searched games.
- That is a drop driven by the TREATED game's length (post-treatment), so it is correlated with the outcome.
- A timer firing is a harness defect, never an outcome. Such a deal is a post-treatment drop: imputed (§5) and capped
  at 0.5 %.

**Built, unchanged:**
- **The forfeit-win guard:** a game ended by a peer drop or a forfeit other than our own turn cap never counts.
- **The off-loop search:** the peer's keepalive is held (S-U A9).
- **The 250-turn cap:** it is symmetric in both arms. 0.5 % of dlive's raw U games reached it.

## 7. The leaks

- **What leaks.** S′'s worlds replay the prefix on the TRUE battle seed, so speed ties and hidden counters leak
  (`search_ledger.md` §6.1). `dice="rolls"` removed the damage-roll leak (the re-gate).
- **What that does to the classes.** A GO-LEAKY is LEAK-INCLUSIVE by name. NULL is creditable under **A-LEAK**, an
  assumption: the leaks do not lower G. S′ is not a Bayes-optimal planner, and true-seed screening changes which
  worlds are drawn, so A-LEAK can fail in principle. dlive's strata support it but do not prove it.
- **A warning sign, REPORTED only.** In dlive's strata the leak-exposed departures read +2.6 pp and the leak-clean
  ones −0.06 pp. Both intervals included 0. A GO-LEAKY's leak-honest replication is therefore not a formality.

## 8. Cost and wall time (ESTIMATE; cost never re-sizes the read)

**The model.**
- **Per decision.** S′ costs 4.1 CPU-s at turns 10–14, 4.9 at 15–19, 5.4 at 20–24 and 6.3 at 25–30 (dlive's 5,328
  offline rows). Exclusions are 3 % and cost 13.7 CPU-s each.
- **Past turn 30** the one lean-screen searched game's per-turn cost
  ([`../su_headroom_2026-10-27/AMENDMENT_1_DRAFT.md`](../su_headroom_2026-10-27/AMENDMENT_1_DRAFT.md) §1b) is scaled
  by S′'s ratio to it at turns 10–29 (5.0 / 7.8). Past turn 63 it is flat (low) or rising with the turn (high).
- **Decisions per game** come from dlive's 3,200 raw games, parsed from their captures. The parse agrees with all
  67,720 input-log rows on turn and legal count.

| per U game | median | mean | p90 | p99 |
|---|---|---|---|---|
| turns | 36 | 46.5 | 81 | 193 |
| our decisions at turn ≥ 10 with ≥ 2 legal | 31 | 40.2 | 76 | — |
| searched-game CPU-min (low / high) | 3.9 | 7.8 / 9.6 | 19.5 / 20.7 | 59 / 104 |

**The read:**

| item | CPU-h |
|---|---|
| 2,700 searched games | 352–433 |
| 2,700 raw one-game units (≈ 23 CPU-s each, dlive's branch units) | ≈ 17 |
| the identity guard's offline re-score (600 decisions, ≈ 10 CPU-s each) | ≈ 2 |
| **total** | **≈ 370–450** |

- **What sets the mean.** 16 % of U games pass turn 63, and they set the mean.
- **Where a mean game's CPU goes:** 0.5 min start-up, 1.8 for turns 10–30, 2.9 for turns 31–63, and 2.6–4.4 after
  that.
- **What the model does not know** (FINDING S′L-1):
  - it rests on ONE searched game's late profile;
  - it uses raw-game lengths;
  - it leaves out the Speed marker's 0.3 s.

**Wall, locally.** `--concurrency 2` on the 2 heavy slots, `nice -n 19` on the agent cores, under `mem_cap.sh`.
- A searched game runs about single-core: stage 0's wall / CPU was 1.06.
- With both slots this read's, that is **≈ 8–10 days**. Shared with the queue it is **≈ 2–4 weeks**.

**Wall, rented** (an OPTION; it needs the owner; UNVERIFIED: no rented CPU has played a Kakuna game or a search).
- **The box:** CPU only, no GPU, no human data. A 64-vCPU / 128–192 GB box at ≈ 3–4 GB a game (the searched process,
  2.4 GB in the build's smoke, plus its Kakuna peer) holds ≈ 30–45 games at a time.
- **The time and price:** ≈ 10–25 box-hours (the upper end at the smoke's cost ratio), **≈ $20–90** at ≈ $1.5–3.5 a box-hour. One amendment first (§4).

## 9. Power (normal approximation; SE = √(d / 2,700))

| true G_U | d = 0.40: P(GO) / P(NULL) | d = 0.45: P(GO) / P(NULL) |
|---|---|---|
| 0 | 0.025 / 0.69 | 0.025 / 0.64 |
| +1 pp | 0.13 / 0.38 | 0.12 / 0.34 |
| +2 pp | 0.38 / 0.13 | 0.34 / 0.12 |
| **+3 pp** | **0.69** / 0.02 | **0.64** / 0.02 |
| **+3.5 pp** | **0.82** / 0.01 | **0.77** / 0.01 |
| **+4 pp** | **0.91** / 0.00 | **0.87** / 0.00 |

- **d, the discordance, is a GUESS.** dlive's single forced departure split its pair 28 % of the time. X51's two
  different policies split 36–41 %. A searched game departs ≈ 12–16 times, so d ≈ 0.40–0.45.
- **The design effect.** The unit bootstrap of 45 units a panel costs little against a deal-level SE. Deals are
  independent: each has its own teams and seed.
- **So 2,700 deals reach 80 % power at ≈ +3.4–3.6 pp,** inside the brief's +3 to +4 pp.
- **What the record reports about power.** It reports the achieved d (b + c only) and the power it implies. Neither
  changes n.

## 10. Reported, never decided

- G per panel with its interval, from the same replicates.
- Each arm's pooled-U score over the paired games (`strat_score`). The searched arm's LB against 0.50 is goal (b)'s
  meter, quoted with its leak caveat.
- The discordance share (b + c; never b − c).
- **The three intervals** (§5): paired-only, GO-imputed and NULL-imputed. Pre- and post-treatment drops by panel and
  cause.
- **Per panel:**
  - pairs and peer drops;
  - one-sided deals and timer kills;
  - the deviation rate, the exclusions and the searched CPU per game.
- **The identity guard:** decisions sampled and re-scored, by panel and turn bucket.

No outcome is printed while the read runs. The operator reads statuses, G-PREFIX, the identity guard and
`search-cost` only.

## 11. Predictions (GUESSES, before any game)

- **G_U ≈ +4 pp (−2, +12).**
  - The performance-difference identity, applied to dlive's point estimate (+0.012 a departure × ≈ 14 departures a
    game), gives ≈ +15 pp.
  - It is shrunk hard, for four reasons:
    1. dlive's interval includes 0;
    2. the gate-to-fresh shrinkage (DL-10) may not be over;
    3. late-game decisions were never sampled;
    4. the leaf's opponent model is P\*, not Kakuna (S-U A2).
- **By panel:** no ordering predicted. dlive's H > F > P72 rested on single departures.
- **The classes:** P(GO-LEAKY) ≈ 0.5 (`go_below_margin` ≈ 0); P(NULL) ≈ 0.2; P(STOP) ≈ 0.3.
- **d ≈ 0.42.**
- **Guards:**
  - G-PREFIX: 0 mismatches.
  - The identity guard: PASS, 600 re-scored.
  - Drops < 0.5 % a panel.
- **Cost:** the mean searched game ≈ 9 CPU-min (7–13).

## 12. Operation (the play command)

From the MAIN checkout, after the registration commit:

```bash
P=0c76b521e
git worktree add --detach /home/goodlad/dev/gen3ai-reads/sprime_live_$P $P
cd /home/goodlad/dev/gen3ai-reads/sprime_live_$P
nice -n 19 taskset -c 5-7,13-15 ./scripts/bootstrap.sh --skip-env --with-rust --no-check
export PYTHONPATH=$PWD/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES=""
SPEC=/home/goodlad/dev/gen3ai/designs/research_state/measurements/sprime_live_play_2026-11-01/read_spec.json
STATE=/home/goodlad/gen3ai_archive/sprime_live_play_2026-11-01
E4='port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129'
engine_ok() { "$PY" -c "from main.anchors.server import engine_stamp as e; s=e(); print(s); assert s=='$E4'"; }
engine_ok
"$PY" -m main.reads check-spec --spec "$SPEC"          # pin $P; play_path_differs_at_head must be 0
"$PY" -m main.reads plan --spec "$SPEC" --state "$STATE"
# the cost checkpoint (§6): unit seed 300, both arms, 120 units; detached, watched by its PID
setsid nohup "$PY" -m main.reads run --state "$STATE" --lane anchor --hours 72 --concurrency 2 --max-units 120 \
    > "$STATE/run_checkpoint.log" 2>&1 < /dev/null &
"$PY" -m main.reads pid --state "$STATE" --lane anchor --wait 180      # watch it with [ -d /proc/<PID> ]
# when it exits: cost, G-PREFIX, and the identity re-score of the checkpoint's sample (no result is read)
"$PY" -m main.reads search-cost --state "$STATE"
RESCORE=(-m main.value_search.live_identity rescore --cells "$STATE/anchor/u/pstar_s/cells"
         --panels heldout72_hpfix,forum72,p72 --model "$STATE/ckpts/pstar_s/checkpoint_280000170_steps.zip"
         --model-sha256 814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed --engine "$E4")
nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 "$PY" "${RESCORE[@]}" --max 600 --min 1 \
    --out "$STATE/identity/checkpoint.json"
# then the rest, one supervisor life at a time; the engine before every life
engine_ok && setsid nohup "$PY" -m main.reads resume --state "$STATE" --lane anchor --hours 72 --concurrency 2 \
    --by <operator> > "$STATE/run_resume_<n>.log" 2>&1 < /dev/null &
# when every unit is counted or exhausted: the registered re-score, then the record
nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 "$PY" "${RESCORE[@]}" --max 600 --min 300 \
    --out "$STATE/identity/rescore.json"
"$PY" -m main.reads record --state "$STATE"
```

- **The checkpoint's re-score** uses `--min 1`, because seed 300 alone holds ≈ 150 sampled decisions. The registered
  300 applies at the record, and the record's re-score is a fresh output file over the whole state.
- **The flags are the CLI's.** Check them with `--help` at the pin before the first unit; a refusal there is never
  worked around. The `ckpts/pstar_s/` path is `plan`'s verified copy; use whatever path `plan` reports.
- The supervisor is watched by its PID (`main.reads pid`), never by a name match.
- Stop it only by its PID, and only after its in-flight units' children have exited (dlive A1).
- **The order:** the orchestrator's queue. The read takes both heavy slots for days (§8), so it runs when the
  queue's higher reads allow, or on the rental under its amendment.

## 13. The build's smoke (before this PREREG; outside the read)

Run on this build's tree (a detached snapshot of it, bootstrapped `--with-rust`, stamping E4), CPU only, through
`main.reads` (`plan`, `run --max-units 2`). It played **home72 unit seed 200 game 0**: a dlive deal outside this
read's panels and seeds. State: `~/gen3ai_archive/sprime_live_smoke_2026-11-01{,_rep}/`.

| unit | wall | CPU | peak memory |
|---|---|---|---|
| raw g0 (lone) | 20 s | 18.5 s | 2.23 GB |
| searched g0 | 590 s | 592 s | 2.43 GB |
| searched g0 again (the replay, under the ship gate's load) | 689 s | 687 s | 2.42 GB |

- **The searched game.** 76 of our decisions; 66 reached the search: 56 searched, 10 excluded, 28 departures (0.50).
  - Search CPU was 9.5 min: 7.5 building worlds, 1.7 in playouts.
  - A searched decision took 8.1 CPU-s at the median (p90 12.9, max 18.3).
  - The game ran to turn 71; the raw game ended at turn 49.
- **G-PREFIX:** `ok`, in both runs.
- **Determinism.** The replay's captures (`|t:|` removed) and its `search_decisions.jsonl` (CPU and wall fields
  removed) are identical to the first run's, for both units.
- **The identity fields.**
  - All 66 search-path records carry `log_sha256` and `frame_sha256`.
  - The first four (t10n9 to t11n12, before S′'s first departure) equal dlive's kept raw decisions of that game, on
    both digests and π's greedy action.
- **`live_identity check` against dlive's offline rows:** 0 matched, 2 key-only. That is the expected outcome: S′
  departed at t11n12, before the bank's two states of this game (t19 and t26), so neither was reached by the
  identical history.
- **`live_identity rescore`:** PASS, 3 of 3 sampled decisions re-scored offline equal to the live record
  (t11n12 searched, t35n37 searched, t39n42 excluded), 0 mismatches, 21 CPU-s.
- **The lone raw game against dlive's in-unit game 0** (`main.reads.peer_divergence`): `identical`.
- **What it says about cost (one game, FINDING S′L-4).**
  - The whole unit took 9.9 CPU-min against the model's ≈ 6 for a 71-turn game: about 1.6×.
  - Exclusions were 15 % (dlive's home72 rate: 6 %).
  - If that ratio held for every game, the read would cost ≈ 600–700 CPU-h. The cost checkpoint measures it.

**The BANK TOOTH** (`live_identity bank-tooth`, at the build's tree, which is the pin's code). It took dlive's
selected states with sha256(key) % 64 == 0: 94 states, 16 never scored by dlive, so **78 compared**. By panel: F 22,
H 22, home72 14, P72 20; they include 2 exclusions and 30 departures. Every field of dlive's `search` block except
the five timing fields was compared: `config`, `deviation`, `leak`, `log_pi`, `played`, `q`, `root_v`, `searched`,
`species_hit`, `spent`, `world_tries`. **0 mismatches.** It ran on one thread at 375 CPU-s, peak 1.0 GB. So the
pin's S′ is dlive's S′, decision for decision.

The build's tests (each fails on revert) are listed in the registration commit. The routine gate passed every test.
It exited 1 on its 8-s per-test time bar alone: 11 tests outside this change, under a live run's contention.

## Registration

- Pin `0c76b521e`. `check-spec`: `play_path_differs_at_head` 0. The spec's sha256 is `57f0ee87…fb7a`.
- Engine E4. Preset `r0s4_rolls@10`.
- State `~/gen3ai_archive/sprime_live_play_2026-11-01/`.
- Committed with [`read_spec.json`](read_spec.json) before any game.
- Written by the S′-LIVE designer (orchestrator brief, 2026-11-01). Two independent adversarial reviews (§ Reviews)
  were folded in before registration.

## Reviews (before registration)

Two fresh opus-high reviewers, independent of each other and of the designer, reviewed the drafts. They reported
blockers and majors only. Neither found a blocker, and both read REGISTER AFTER FIXES. Every major is folded in
above.

**Review 1: statistics and design.**

| id | finding | how it was handled |
|---|---|---|
| M1 | The drafts played dlive's own deals (seeds 200–244), the deals whose positive lean prompted this read. A searched game replays the raw game up to S′'s first departure, so ≈ 200 deals would replay dlive branches, and every deal shares dlive's deal-level luck. The claim that the error rates were unchanged was false (≈ +0.1–0.2 pp of bias, false GO ≈ 3 %) | **Fresh seeds 300–344** (§2, §3). The dlive-matched identity guard was replaced by the sampled offline re-score (§6) |
| M2 | Post-treatment, outcome-correlated drops (timer kills, recurring search defects, peer faults after turn 10) were lumped with prefix peer drops under one 2 % cap. They could move G by ≈ ±1 pp, and the worst-case bounds were too loose (±1 rather than [−y, 1 − y]) | Pre and post drops are counted apart, with caps of 2 % and **0.5 %**. **The classes impute post-treatment drops at their worst case for each class** (§5). A searched unit that uses up its attempts HOLDs the lane. The checkpoint projects worst-case wall time against the watchdog (§6) |
| M3 | The NULL licence over-claimed: a ≤ 3 pp gain would still stack with other levers; "leaks only raise G" is an assumption; 3 pp against S-U's 4.3 was unexplained | NULL means "closed as a STAND-ALONE route", with [LB, UB] beside the other levers (§1). A-LEAK is named as an assumption (§7). The 3.3 vs 4.3 reconciliation is in §5 |
| M4 | The GO licence could be over-read as deployable search | The class is renamed **GO-LEAKY** in the rule, the record and the ledger. It licenses only a leak-honest replication; the score is quoted at most as "GOAL-B-REACHED (leaky)" (§1) |
| notes | `go_below_margin` is unreachable at d ≈ 0.42; "never a bias" should read GUESS | Both done (§1, §3, §11) |

**Review 2: harness, guards, engine identity.**

| id | finding | how it was handled |
|---|---|---|
| M1 | The re-score's registered order was panel-first, so the first 600 would all be H seeds ≈ 300–313: no check of F, P72 or later supervisor lives | The order is now **sha256(key) first across all cells**, spreading evenly over panels and time (§6). A test fails on revert |
| M2 | Nothing measured that the pin's S′ equals dlive's: the smoke's dlive check matched 0. §4's "exactly two places" was false: ten play-path files differ from `c01ae8ee5` before this build | The **BANK TOOTH** (`live_identity bank-tooth`) re-runs S′ at the pin on a hashed sample of dlive's states and requires dlive's offline rows exactly (§4, §13). §4 lists the ten files |
| M3 | A deal left unfinished (a failed unit whose retry never ran) was silently dropped | Unfinished deals are now a VOID reason (§5), with a test that fails on revert. The fail-closed `unplayed` path is stated |
| M4 | §12's commands would not run: rescore lacked its required flags; resume lacked `--concurrency 2`; runs were in the foreground; the seed was wrong; the spec had no pin; the rescore was unwrapped; the engine check per life was not in the runner | §12 rewritten: the full rescore flags, detached runs watched by PID, `--concurrency 2`, `mem_cap.sh` wrapping, the engine checked before every `resume`, and the pin added at registration |
| M5 | The G-PREFIX release named the wrong verb (`release-guard-hold`), and there was no outcome-blind classification step | A read-only verb, `main.reads.peer_divergence prefix`, prints the class only. The release is `resume --by --why` (§6) |

## FINDINGs

- **S′L-1 (the cost rests on one game's late profile).** Past turn 30, every cost figure is one lean-screen game,
  scaled. The cost checkpoint measures it on 60 searched games before the rest is played.
- **S′L-2 (dlive's raw outcomes are known).** Their U score of 0.486 sets the margin's context (§5). The read plays
  fresh deals, so no deal of this read has a known outcome.
- **S′L-4 (the smoke's searched game cost ≈ 1.6× its model).** It is one game, but the cost could reach ≈ 600–700
  CPU-h. The checkpoint's 60 games decide between local and rented play. They never change n.
- **S′L-5 (the build widened assemble).** `assemble` used to mark a group inconclusive on any G-PREFIX mismatch, so a
  pre-registered peer drop could never reach the rule. `Rule.owns_g_prefix` (true only for `searched_play`) hands
  the mismatch to the rule. `search_headroom` is unchanged. The lane still HOLDs on a mismatch, and the operator
  releases it.
- **S′L-3 (the decision-to-game bridge is not tested here).** A GO or NULL says nothing about whether
  per-departure figures predict game gains.
