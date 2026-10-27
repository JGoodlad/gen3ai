# S-U: does one-ply search on P\* gain MORE on unfamiliar teams than on home72? — PREREG (2026-10-27)

Written and committed BEFORE any searched game counts. Design (the argument, the alternatives, the self-check):
[`design_search_u_headroom.md`](../../../endstate/design_search_u_headroom.md). Spec: [`read_spec.json`](read_spec.json)
(`main.reads`, rule `search_headroom`). This PREREG fixes what the design left open (the pin, the engine, the rule's
parameters, the cost model) and changes none of its classes, bars or size.

Abbreviations: pp = percentage points; P\* = `rb_x26_s2601` @280,000,170 (sha256 `814823c5…91ed`), the plateau policy;
Kakuna = Metamon's Kakuna, checkpoint 34, greedy; home72 = 72 pool teams; U = `heldout72_hpfix` (H), `forum72` (F) and
`p72` at equal panel weight (X51's definition); S = R0 §4's search, preset `r0s4` (8 honest worlds `dice="seed"`,
TRUNC-4-K8, piKL λ 0.015), from turn 10, at our decisions with ≥ 2 legal actions; G_set = WR(P\* + S) − WR(P\*) on a
set, paired by deal; I = G_U − G_home72; d = the share of paired deals whose two outcomes differ; LB / UB = the lower
/ upper bound of a 95 % interval.

## 1. The question

H-MID (`f200258f`) found P\* converts level turn-10 positions ≈ 11 pp worse on U than on home72 against Kakuna, with
its critic's game ordering intact on both. One-ply search compares the actions of ONE node, so a level error common to
all successors cancels. **Does search recover part of the U conversion gap (I), and does it close goal (b) (G_U)?**
(design §0–§1).

## 2. The read

- **Arms**: `pstar` (P\* raw, greedy) and `pstar_s` (the same file with `load.search = "r0s4@10"`, stamped
  `our_regime = greedy+search=r0s4@10`), against Kakuna greedy, on our in-process front end (core slot) on a 95xx
  port, the peer offline on its pinned weights.
- **One `per_game` anchor group `su`**: team sets home72, `heldout72_hpfix`, `forum72`, `p72`; unit seeds 0–19 (phase
  1). Each seed's 20-game series is 20 one-game units (`main.anchors --deal-game G`); the queue is unit seeds
  outermost, then the game, then the set, the raw twin before the searched game, so every deal is played raw then
  searched on one engine in one queue.
- **G-PREFIX**: each pair's protocol (both sides, `|t:|` removed, and the `START` document) is byte-identical through
  `|turn|10`; a mismatch HOLDS the lane, and an unresolved one makes the read VOID.
- **The engine**: ONE engine, the pin's (`main.reads.engine`; recorded with stage 0 below). Nothing is imported: X51's
  banked raw rows are on E0, and the raw arm is re-played here.
- **The peer's team pack**: `gen3_peer_showdown_pack_v1` (SOP H22, `b0441623`); forum72 plays under the new pack, a
  regime of its own (`peer_pack_changed`), so no banked forum72 Kakuna row pools with this read.

## 3. The decision rule (`search_headroom`; the exact `params` are the spec's `decision` block)

Exactly the design's §3 (`main.reads.rules_search_headroom`, B4):

- **Pairs.** A deal (set, unit seed, game) is a pair iff BOTH its raw and its searched unit counted
  (`main.anchors.results.unit_counts`, the forfeit-win guard included). A result is win 1, tie ½, loss 0. Nothing is
  imputed; deals with one side counted are reported per set.
- **Statistics.** Per (set, unit seed): the mean paired difference (searched − raw) over its pairs. G_set = the mean of
  its units' statistics (units weigh equally); G_U = the equal-weight mean of the three U sets' G; I = G_U − G_home72.
- **Intervals.** A percentile bootstrap over units, resampled within each set: ONE `numpy.random.default_rng(20261027)`,
  B = 2,000, the U sets drawn first in the order `heldout72_hpfix`, `forum72`, `p72`, then home72 (so phase 2's added
  home72 units leave G_U's interval exactly as phase 1 read it); 2.5 / 97.5 percentiles.
- **Bars** as exact fractions; a bound within 0.0005 of a bar takes UNDECIDED (root rule 8).

| primary | class | rule |
|---|---|---|
| A (I) | **U-HEADROOM** | LB(I) > 0 and LB(G_U) > 0 (phase 2 only) |
| A (I) | **U-LESS** | UB(I) < 0 |
| A (I) | **NO-U-PREMIUM** | UB(I) < +5.5 pp (may co-occur with U-LESS, and at phase 2 with U-HEADROOM; both are then recorded) |
| B (G_U) | **DOES-NOT-CLOSE-B** (creditable) | UB(G_U) < +4.3 pp |
| B (G_U) | **U-GAIN (leaky)** | LB(G_U) > 0 |
| B (G_U) | **GOAL-B-REACHED (leaky)** | the searched arm's stratified U score over its paired games (`strat_score`, a tie a non-win, X51's rule) has LB > 0.50; reported, never claimed as goal (b) met |

Otherwise a primary reads UNDECIDED. **At phase 1, A is futility-only**: only U-LESS or NO-U-PREMIUM can be read, and
either one STOPS the read; a would-be U-HEADROOM is recorded under `not_read_at_phase_1` and reads UNDECIDED.

The spec's `decision` block (the rule's `check` refuses any other key, a non-`per_game` group, a searched label
without its raw twin, or an `expected_search` other than the searched label's `load.search`):

```json
{"rule": "search_headroom", "params": {"group": "su", "searched": "pstar_s", "raw": "pstar",
  "home_panel": "home72", "u_panels": ["heldout72_hpfix", "forum72", "p72"], "phase": 1,
  "bars": {"no_u_premium_pp": 5.5, "dnc_b_pp": 4.3, "goal_b": 0.5},
  "bootstrap": {"B": 2000, "seed": 20261027}, "rule8_eps": 0.0005, "expected_search": "r0s4@10",
  "engine": "port=bdc5d2be3af5e1fb/94;env=95b45f104119bdb2/129"}}
```

## 4. Size, phases, stop

Unchanged from the design §4.4.
- **Stage 0** = the queue's first 40 units: the first 5 deals of unit seed 0 on each of the four sets, raw and
  searched (20 pairs). It reads COST ONLY (§5), plus G-PREFIX and the peer's stability; no outcome is read or shown.
- **Phase 1** = unit seeds 0–19 on all four sets: 1,200 U deals and 400 home72 deals, 1,600 searched games. **G_U's
  classes are read here, and are final.** I is read futility-only: the read STOPS if NO-U-PREMIUM or U-LESS is
  reached.
- **Phase 2** = home72 unit seeds 20–59 (+800 searched games), only if I is UNDECIDED at phase 1, registered by an
  amendment BEFORE any phase-2 game (its group and how the rule pools it with phase 1's home72; each phase-2 deal is
  still raw-then-searched in one queue on the same engine). I's classes are then read on 1,200 / 1,200.
- **Two looks on I**, the negative class at either look (false rate ≤ 2 × 2.5 %, Bonferroni); the positive class once.
- **The re-size rule** (fixed, cost is never an outcome): if stage 0's median searched game costs > 50 CPU-min
  (`main.reads search-cost`'s `resize` input), phase 2 is dropped and I becomes REPORTED. Nothing else re-sizes.
- **The playing commit**: stage 0 plays at the spec's `pin`. The full read does not start without the orchestrator's
  go (and the owner's, for a rented box). If it plays at another commit (a lane change for searched units, §5; another
  host), an amendment registers it before its first game; stage 0's 20 pairs count toward the read only if that
  commit's play path is byte-identical to the pin's (`main.reads.playpath`) on the same engine, else they are
  cost-only and their deals are re-played.

## 5. Cost (ESTIMATE from the build's smoke; stage 0 MEASURES it)

The design's ≈ 25 CPU-min a searched game (15–37) came from other reads' per-decision costs. The build's smoke
(`5fea300b`, `~/gen3ai_archive/su_build_smoke_2026-10-27/`, 12 searched decisions on 2 deals, turns 10–19) measured
**6.4 CPU-s a searched decision** (median, 5.1–8.2; the worker thread plus its replay children) and ≈ 10.3 CPU-s of
WHOLE-process CPU more per searched decision than the identical raw game.

| quantity | model | value |
|---|---|---|
| searched decisions a game | turns ≥ 10 hold ≈ 74 % of a 35-turn game's ≈ 50 choices; ≥ 2 legal 12 / 12 in the smoke | 30–37 |
| search CPU a searched game | 30–37 × 6.4 s | 3.2–4.0 CPU-min |
| WHOLE-unit CPU a searched game | 11.8 s (the raw game, X51) + 30–37 × 10.3 s | **5.3–6.6 CPU-min, central ≈ 6** |
| a raw per-game unit | 11.8 s + the unit's start-up (model load, the peer) | ≈ 0.5–1 CPU-min (stage 0 measures) |
| phase 1 (1,600 searched + 1,600 raw units) | | **≈ 160 + 20 ≈ 180 CPU-h** |
| the full read (2,400 + 2,400) | | **≈ 240 + 30 ≈ 270 CPU-h** (the design's ≈ 1,000) |

Unmeasured until stage 0: later turns (deeper games cost more per playout), H and P72 (none was searched), the
start-up per unit, and the slot contention's wall factor (1–1.5× at nice 19, 2.5–3× under SCHED_IDLE).

**The power is unchanged** (design §4.3; it does not depend on cost). At 1,200 / 1,200 and d = 0.4:
DOES-NOT-CLOSE-B 0.65 at G_U = 0; U-GAIN 0.65 at G_U = +4.3 pp; U-HEADROOM 0.49 / 0.77 at I = +5 / +7 pp; NO-U-PREMIUM
0.57 at I = 0. Re-quoted at the smoke's cost (6 CPU-min a searched game):

| n_U / n_home72 | searched games | CPU-h at 5.3 / 6 / 6.6 min |
|---|---|---|
| 400 / 400 | 800 | 71 / 80 / 88 |
| 1,200 / 400 (phase 1) | 1,600 | 141 / 160 / 176 |
| 1,200 / 1,200 (phase 2) | 2,400 | 212 / 240 / 264 |
| 1,800 / 1,800 | 3,600 | 318 / 360 / 396 |

(The raw twins add ≈ 11.8 s each plus start-up.) An interaction a third of the gap (≈ 3.5 pp) still needs ≈ 5,100
deals a set at 80 % power, ≈ 1,000 CPU-h at this cost: still out of scope, a matter for its own registration.

## 6. Predictions (GUESSES, before play)

The design's §7, unchanged except the cost: the deviation rate 8–15 % of searched decisions, a few points higher on
U; G_home72 = +0.5 pp (−3, +4); G_U = +1.0 pp (−3, +5); I = +0.5 pp (−4, +5); DOES-NOT-CLOSE-B 0.55, U-GAIN (leaky)
0.15, U-HEADROOM 0.08, NO-U-PREMIUM 0.45, U-LESS 0.07, a phase-1 stop on I ≈ 0.3. **Cost: a searched game at 4–8
CPU-min of whole-unit CPU** (was 20–30 in the design, before the smoke).

## 7. VOID, guards and integrity

- **VOID** (the rule): counted units on two engines or an unknown one; an unresolved G-PREFIX mismatch; a raw arm on
  another checkpoint (sha256); any searched record whose search preset differs from `r0s4@10`; a raw unit carrying a
  search stamp.
- **The forfeit-win guard** (`main.anchors.runner.forced_end_verdict`, landed with this PREREG's pin): a game a
  FORCELOSE decided (a peer's disconnect or keepalive timeout, a peer's `/forfeit`, our own side dropping) other than
  our own turn-limit cap is no outcome: its row is dropped, the unit FAILS (`non_battle_termination`) and never counts.
  A searched decision runs off the event loop (S-3), so a drop is a harness defect, never a result; any VOID game is
  reported with stage 0 and the read.
- **Not imputed**: a deal whose raw or searched unit did not count (after `MAX_ATTEMPTS`) is no pair, and is reported.
- **Leaks**: every positive class of G_U is LEAK-INCLUSIVE (the worlds replay the true seed; `search_ledger.md` §6.1),
  so only DOES-NOT-CLOSE-B is creditable to honest search. I's leak-robustness is first order (design A4, S-4).

## 8. Reported, never decided

G_home72 alone and G per panel; I on F alone; the deviation rate by set; searched CPU-s per decision and per game; the
pooled discordance b + c after stage 0 and before phase 2's decision, with the achieved power (never the sign); the
raw arm's score per panel on this engine beside X51's (a cross-engine guard, never pooled). The level-at-10 subgroup,
the world-quality covariate and H-MID's Δbias re-read are REPORTED by separate scripts over the captures and
`search_decisions.jsonl` after the read (the rule does not compute them).

## Registration

- **Pin** `bf5f1a8e7` (B1–B4 and the forfeit-win guard; on top of `b0441623`'s peer pack, H22). The spec registers
  it; stage 0 plays from a clean detached worktree at it (`../../../tools/reads.md` §1).
- **Engine** `port=bdc5d2be3af5e1fb/94;env=95b45f104119bdb2/129` (`main.anchors.server.engine_stamp()` at the pin; the
  spec's `engine` param, so a unit on any other engine VOIDs the read). Stage 0's README records each unit's stamp.
- **Peer pack** `gen3_peer_showdown_pack_v1`: forum72 plays with `peer_pack_changed` true, the other three sets false;
  stage 0's README records it per set.
- **State** `~/gen3ai_archive/su_headroom_2026-10-27/`, the anchor lane at `--concurrency 1`.
- Committed with [`read_spec.json`](read_spec.json) before any searched game counts. Written by the S-U operator
  (orchestrator brief, 2026-10-27).
