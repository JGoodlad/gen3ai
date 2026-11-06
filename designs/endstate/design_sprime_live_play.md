# S′-LIVE — does S′ playing every decision beat P\* against the real Kakuna on unfamiliar teams?

**Status: PLAYED and RECORDED (2026-11-05): GO-LEAKY**, G_U +4.60 pp [+2.66, +6.48] (GO-imputed LB +2.57) on 2,698
fresh paired U deals, searched-arm U score 0.545 [0.526, 0.564], leak-inclusive; it licenses a leak-honest replication
only ([`README.md`](../research_state/measurements/sprime_live_play_2026-11-01/README.md)). Registered as
[`PREREG.md`](../research_state/measurements/sprime_live_play_2026-11-01/PREREG.md); operated under AM-K6 (wall-clock
only). It is the direct game-level read that the deviation-only live read (dlive) could not be. The dlive read
STOPPED on a power statement (§1). Every piece it plays on was already BUILT for S-U
([`design_search_u_headroom.md`](design_search_u_headroom.md) §6). This unit adds four builds: a 10-h searched
watchdog, two identity fields on each searched decision, the choice-level identity guard, and the rule
`searched_play` with its peer-drop classifier (§6).

**Abbreviations.**
- P\* = `rb_x26_s2601` @280,000,170 (sha256 `814823c5…91ed`), greedy. Kakuna = Metamon's Kakuna, checkpoint 34, greedy.
- S′ = `r0s4_rolls@10`: 8 honest worlds (`dice="rolls"`), the TRUNC-4-K8 leaf, piKL at λ 0.015.
- U = H (`heldout72_hpfix`), F (`forum72`) and P72 at equal panel weight.
- E4 = the engine `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`, the one the gate, the re-gate and dlive played.
- G_U = the U win-rate gain of P\* + S′ over P\*, paired by deal. A tie is ½.
- d = the share of paired deals whose two results differ. pp = percentage points.
- LB / UB = the lower / upper bound of a 95 % interval. GUESS = unmeasured. ESTIMATE = a cost from measured rates.

---

## 1. Why this read, and why now

- **What dlive found** (`../research_state/measurements/dlive_read_2026-10-28/README.md`). Forcing S′'s pick at one
  departure from P\*, then letting P\* play on, won **+1.20 pp** more per departure. The interval was
  [−1.55, +3.88] (n = 1,793 departures on U), so the read was a STOP.
- **Why that STOP is weak.** On those fresh states the self-referee measured S′'s effect at only +0.0099 a departure
  (DL-10). At that size dlive had about **0.11 power**. Its STOP says the sample was too small, not that S′ does
  nothing.
- **What the point estimate would mean in games.** S′ departs from P\* on about 0.39 of its decisions. A U game has
  about 31 searched decisions at the median (40 at the mean; §4). The performance-difference identity sums the
  per-decision advantage along a game, so the point estimate maps to **+10 pp or more** in games (UNVERIFIED).
  That bridge needs an advantage measured on S′'s own states. It also needs turns after 30, which dlive never
  sampled.
- **Only a game-level read answers goal (b)'s question:** does S′ playing every decision beat P\* against Kakuna on U?

## 2. The arms

- **P\* + S′.** S′ decides every one of our decisions from turn 10 on that has at least 2 legal actions. That is the
  search's own eligibility rule (`main.anchors.search_presets`). Every other decision is P\*'s greedy choice. A
  decision with no consistent world is an EXCLUSION: P\* plays it, as in every bank.
  - Built: `main.anchors --our-search r0s4_rolls@10` (`search_side.py`).
  - The search runs off the event loop, keyed by (deal, turn, decision), on the CPU with one thread, so a searched
    game replays exactly.
- **P\* raw.** Greedy, no search: the dlive raw arm's regime and bytes.
- **The opponent** is Kakuna ckpt 34, greedy, with the peer pack `gen3_peer_showdown_pack_v1`, on E4. Our own
  `ws_frontend` serves on a 9XXX port. There is no public server and no training.

## 3. Deals, pairing, and Kakuna's nondeterminism

- **The deals are FRESH:** U unit seeds **300–344** on H, F and P72. That is 2,700 deals, 900 a panel. No prior
  Kakuna read has played these seeds.
  - The brief allowed reusing dlive's raw deals (seeds 200–244). The stats review (M1) showed why that is a mistake.
    This read exists partly because dlive's estimate on those deals leaned positive. A searched game replays the raw
    game up to S′'s first departure, so on ≈ 200 of them it would replay a dlive branch's first move and seed luck.
    Every one of those deals would also carry dlive's deal-level luck: H's +4.2 pp there is partly that.
  - Fresh seeds remove that correlation, and they cost nothing more, because the raw arm is re-played anyway.
- **dlive's raw games still serve as data:** the cost model's game lengths and decision counts (§4), the raw arm's
  E4 score behind the margin, and the build smoke's cross-check against dlive's offline S′ rows.
- **One `per_game` group plays both arms**, deal by deal: the raw game first, then the searched game. Each is a
  ONE-GAME unit (`--deal-game G`). This is S-U's lane (B3), unchanged.
- **Both arms are lone processes.** In dlive, Kakuna sometimes chose differently in a lone replay than in game G of a
  20-game process: 9 of 2,100 branches (0.43 %), DL-7. Every lone replay agreed with the other lone replay. With
  both arms lone, that asymmetry is not built into the pairing.
- **The shared prefix.** Before turn 10 both arms are the same deterministic play on the deal's own battle seed.
  - **G-PREFIX** (built) requires each pair's protocol to be byte-identical through `|turn|10`.
  - **A mismatch is classified, with the rule fixed before play** (dlive's lesson, AM-DL-PEER). Kakuna choosing
    differently first, on an identical view, before turn 10, is a PRE-TREATMENT PEER DROP: the deal is excluded and
    counted. Any other mismatch VOIDs the read.
- **After turn 10** the arms diverge by design. GUESS: any residual Kakuna noise there adds variance, not bias. Its
  mechanism is unknown (DL-7), but it is bounded by near-tie flips and is ≪ 1 pp.

## 4. Meter, rule, power, cost

**The primary meter.**
- D = y(searched) − y(raw) per deal.
- G_U = the mean over the three U panels of each panel's mean D, equally weighted.
- The interval is a percentile bootstrap over 20-deal units within each panel: B = 2,000 and one fixed seed. This
  is S-U's scheme without home72.

**The decision** (rule `searched_play`; exact fractions; a bound within 0.0005 of its bar is never a crossing):

| class | rule | in plain words |
|---|---|---|
| **GO-LEAKY** | LB(G_U) > 0, with every post-treatment unpaired deal imputed at its worst case for GO | P\* + S′, WITH the builder's true-seed information, beats P\* on U against Kakuna. It licenses only building and registering a leak-honest replication on fresh deals. It says nothing about deployable search |
| **NULL** | not GO-LEAKY, and UB(G_U) < **+3 pp**, with the opposite imputation | S′ as built adds less than 3 pp (95 %). It cannot carry goal (b) alone at any practical n. A gain of ≤ 3 pp that would stack with other levers is NOT excluded. Creditable under A-LEAK: the leaks do not lower G |
| **STOP** | otherwise | Unresolved. No licence |

**Why the margin is +3 pp.**
- On E4 the raw arm scored 0.486 on pooled U (dlive's 2,700 raw games).
- Goal (b) needs a pooled-U score whose LB is above 0.50. At this n that means a point estimate of 0.519, so
  **+3.3 pp**.
- S-U's 4.3 pp was the same gap at goal (b)'s usual 1,200 games, from X51's 0.487. At 2,700 games the gap is 3.3.
- A gain below 3 pp cannot carry goal (b) by itself at this n. Reaching LB > 0.50 at +2 pp would need ≈ 54,000
  games.

**Power** (normal approximation; SE = √(d / 2,700); P(GO) = Φ(δ / SE − 1.96)):

| true G_U | d = 0.40: P(GO) / P(NULL) | d = 0.45: P(GO) / P(NULL) |
|---|---|---|
| 0 | 0.025 / 0.69 | 0.025 / 0.64 |
| +2 pp | 0.38 / 0.13 | 0.34 / 0.12 |
| **+3 pp** | **0.69** / 0.02 | **0.64** / 0.02 |
| **+3.5 pp** | **0.82** / 0.01 | **0.77** / 0.01 |
| **+4 pp** | **0.91** / 0.00 | **0.87** / 0.00 |

- **d is a GUESS.** A single forced departure split its pair 28 % of the time (dlive). X51's twin policies split
  36–41 %. A game with 12–16 departures sits between that and independence.
- So 2,700 deals give 80 % power at about +3.4 to +3.6 pp, inside the +3 to +4 pp the brief sized for.
- The false-GO rate is 2.5 %. The false-NULL rate at a true +3 pp is 2.5 %.

**Cost** (ESTIMATE from measured rates):
- **The per-decision cost.** S′ costs 4.1 CPU-s a decision at turns 10–14, rising to 6.3 at turns 25–30 (dlive's
  5,328 offline rows). After turn 30, the one lean-screen game's late profile is scaled to S′ (×0.64). Late
  decisions cost 10–35 CPU-s.
- **How many decisions.** The dlive raw games give the counts: a U game is 36 turns at the median, 46.5 at the
  mean, p99 193, and 0.5 % hit the 250-turn cap. It has 31 searched decisions at the median (mean 40); 2.6 % of
  decisions have only one legal action.
- **Per searched game:** median ≈ 3.9 CPU-min, mean ≈ 7.8–9.6, p99 ≈ 59–104 and max ≈ 87–204. The long games set
  the mean.
- **The read:**

  | item | CPU-h |
  |---|---|
  | 2,700 searched games | 352–433 |
  | 2,700 raw one-game units (≈ 23 CPU-s each, dlive's branch units) | ≈ 17 |
  | the identity guard's offline re-score (600 decisions) | ≈ 2 |
  | **total** | **≈ 370–450** |

  - The mean rests on one game's late profile. The build's one smoke game (71 turns) cost ≈ 1.6× its model. At
    that ratio the read is ≈ 600–700 CPU-h. The PREREG's cost checkpoint (§5) measures it on the first 60 searched
    games. Cost never changes n.

**Wall time.**
- **Locally**, on the 2 heavy slots with `--concurrency 2`: a searched game runs single-core (stage 0's wall / CPU
  was 1.06). With both slots ours that is **≈ 8–10 days** (≈ 13–15 at the smoke's ratio). Shared with the queue's
  other reads it is **≈ 2–5 weeks**.
- **On a rented CPU box** (UNVERIFIED; needs the owner; CPU only, no human data, no GPU): a 64-vCPU / 128–192 GB
  box holds ≈ 30–45 games at a time at ≈ 3–4 GB a game (searched process plus its Kakuna peer). That is ≈ 10–25
  box-hours (the upper end at the smoke's cost ratio), **≈ $20–90** at ≈ $1.5–3.5 a box-hour. It would be one host for every unit of both arms, under an
  amendment registered before its first game (§5).

## 5. Guards

| guard | what it checks | on failure |
|---|---|---|
| **The engine** | every counted unit stamps E4 (`main.anchors.server.engine_stamp`); the playing checkout is a clean detached worktree at the pin, bootstrapped `--with-rust` (DL-4) | VOID |
| **Bytes and regime** | both arms on P\*'s sha256; the raw arm `greedy` with no search stamp; the searched arm stamped and played as `r0s4_rolls@10` with its preset config | VOID |
| **G-PREFIX** (built) | every pair byte-identical through `\|turn\|10` | the lane HOLDs; a `peer_first` mismatch (classifier, §3) is a PEER DROP; anything else VOIDs |
| **The choice-level identity guard** (new) | a hashed 1-in-16 sample of live searched decisions keeps its honest log. The first 600 of them in a registered order are re-scored OFFLINE at the pin by dlive's own scorer path (one thread). Excluded or not, the pick, Q̂, log π and root V must equal the live record exactly. It covers every turn, not only early ones | VOID if any differs or fewer than 300 are re-scored. Checked first at the cost checkpoint. The BANK TOOTH, run at the pin before registration, shows that the pin's offline S′ equals dlive's offline rows |
| **The forfeit-win guard** (built) | a game ended by a peer drop or a forfeit other than our own turn cap is never an outcome | the unit does not count |
| **Drops** | PRE-treatment (a G-PREFIX peer drop) and POST-treatment (one side, or neither, counted after its attempts: a timer kill, a recurring search defect, a peer fault after turn 10) are counted apart | VOID above 2 % pre or 0.5 % post of a panel's deals. Post-treatment drops enter the CLASSES imputed at their worst case for each class: for GO-LEAKY, D = −y_raw, y_s − 1, or −1 if neither side counted; for NULL, the opposite. A searched unit that fails its attempts HOLDs the lane for a root cause |
| **The watchdog** (raised) | a searched game is bounded at 10 h of wall (was 3 h). The longest ≈ 0.3–1 % would otherwise be killed, and a kill driven by the treated game's length is an outcome-correlated drop | — |

- **The cost checkpoint.** After the first 60 searched games (unit seed 200, all three panels, games 0–19), the
  operator reads only CPU, G-PREFIX and the identity guard, never a result.
  - The lane HOLDs if the identity guard fails.
  - If the mean searched game costs more than 2× the model (> 19 CPU-min), the orchestrator decides between the
    rental and continuing locally. The size is unchanged either way.
  - The checkpoint also projects each game's worst wall time (its slowest wall per turn × 250 turns). The lane HOLDs
    if that exceeds the 10-h watchdog.
- **The leaks.** S′'s worlds replay the prefix on the TRUE battle seed (speed ties, hidden counters).
  - `dice="rolls"` removed the damage-roll leak.
  - So a GO is leak-inclusive, and only NULL is credited outright (`search_ledger.md` §6.1). That credit rests on
    A-LEAK, an assumption: the leaks do not lower G. dlive's strata (leak-exposed +2.6 pp, leak-clean −0.06) support
    it but do not prove it.
- **One host.** Torch's numerics differ across CPU types, and that alone could split a near-tie. A rented box is a
  new host, so it needs an amendment before its first game. The amendment must:
  - run the identity guard's offline re-score on that box;
  - play every unit of both arms there.

## 6. Builds (this unit; tests fail on revert)

| # | build | why |
|---|---|---|
| L1 | `main.x35.kakuna.UNIT_SEARCH_S` 3 h → 10 h | no outcome-correlated watchdog kills (§5) |
| L2 | every searched or excluded record carries `log_sha256` and `frame_sha256`; a hashed 1-in-16 sample also keeps its honest log and root (`rescore_sample`) | the identity guard's inputs; no choice changes |
| L3 | `main.value_search.live_identity`: `rescore` (the read's guard) and `check` (live against dlive's offline rows, the smoke's) | the choice-level identity guard |
| L4 | rule `searched_play` (`main.reads.rules_searched_play`), with AM-DL-PEER's classifier moved into `main.reads.peer_divergence` | GO-LEAKY / NULL / STOP; worst-case imputation of post-treatment drops; G-PREFIX peer drops; VOIDs; REPORTED rows |

## 7. What this read does not do, and the alternatives

**What it does not do:**
- **No home72.** Goal (b) is U, and the interaction (S-U's I) is a separate question. A home72 arm would add ≈ 25 %
  to the cost. A GO can be followed by one.
- **No interim look.** A searched group cannot import, so a look needs a partial-record path, which is not built.
  The saving (≤ ≈ 200 CPU-h) comes only when the effect is large.
- **Not S-U.** It is not S-U's registration (`r0s4`, engine `bdc5d2be…`, home72 and I). S-U stays unplayed.

**The alternatives:**

| alternative | why not |
|---|---|
| import dlive's raw arm (saves ≈ 17 CPU-h) | it builds DL-7's process-shape asymmetry into the pairing, and needs a cross-state import into a searched group plus a peer-drop rule on imported captures |
| dlive's deals (seeds 200–244), with the guard matched to dlive's rows | it correlates this read with the dlive estimate that prompted it (review M1). Fresh seeds cost nothing more |
| search turns 10–30 only (≈ ⅓ of the cost) | it is not S′ playing every decision. The late game holds ≈ 40 % of a mean game's searched decisions |
| 3,600 deals (80 % power at +3 pp) | +33 % CPU. 2,700 already reach 80 % at about +3.5 pp |
| a decision-level re-read at more departures | it answers dlive's question, not the game-level one, and composition and turns > 30 stay unmeasured |

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-11-01 | **This doc CREATED** (S′-LIVE designer, orchestrator brief) | the direct game-level read: S′ every eligible decision vs P\* greedy against Kakuna on U, 2,700 fresh deals (seeds 300–344), both arms as lone one-game units, G_U, GO-LEAKY / NULL (UB < +3 pp) / STOP with post-treatment drops imputed at their worst case, the sampled offline re-score as the identity guard, the 10-h watchdog, G-PREFIX peer drops pre-registered | importing dlive's raw arm; dlive's deals (review M1); home72; an interim look; a turn-capped search; 3,600 deals | dlive README (`02fb0f2fc`) DL-7, DL-10; the cost model over dlive's 3,200 raw games and 5,328 S′ rows; S-U AMENDMENT_1_DRAFT §1b; the two reviews (PREREG § Reviews) |
| 2026-11-05 | **PLAYED: GO-LEAKY** (its operator, orchestrator brief) | the status line: played and recorded; G_U +4.60 pp [+2.66, +6.48], GO-imputed LB +2.57, searched U 0.545 [0.526, 0.564]; AM-K6 ran life 4 at `--concurrency 6` (wall-clock only) | K = 8 on cores 1-7,9-15 (a play-path change: `CORES` is pinned in `main.x35.kakuna`); a retry life for exhausted units (orchestrator: recorded as exhausted, imputed) | `measurements/sprime_live_play_2026-11-01/README.md`; ledger 2026-11-05 |
