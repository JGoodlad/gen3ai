# S-U GATE TEST: do S's deviations from π beat chance on unfamiliar teams? — PREREG (2026-10-27)

Written and committed BEFORE any state of this read is played, selected or scored. Source: the S-U adversarial review's
"gate test" ([`../su_headroom_2026-10-27/REVIEW.md`](../su_headroom_2026-10-27/REVIEW.md), `7aafa527`, § The gate
test, B1, B2, M6). Spec: [`read_spec.json`](read_spec.json) (`main.reads`, rule `deviation_sign`). Scorer:
`main.value_search.sign_gate`. Pin: `835c6b4b2` (the commit that lands the recorder, the scorer and the rule).

Abbreviations: P\* = `rb_x26_s2601` @280,000,170 (sha256 `814823c5…91ed`); Kakuna = Metamon's Kakuna, checkpoint 34,
greedy; home72 = 72 pool teams; U = `heldout72_hpfix` (H), `forum72` (F), `p72` at equal weight; S = S-U's registered
search `r0s4@10` (8 honest worlds `dice="seed"`, TRUNC-4-K8, piKL λ 0.015) on the LEAN world screen; π = P\*'s greedy
policy; T = the self-referee (below); LB / UB = the lower / upper bound of a 95 % interval.

## 1. The question

On fresh U states of the read's kind (P\* piloting an unfamiliar team against Kakuna), are S's DEVIATIONS from π right
more often than wrong under S's own model of the opponent? The review's B2 answered it for P\*'s own pool-team
self-play states: sign accuracy **0.513 [0.447, 0.580]** at P\* against **0.580 [0.518, 0.645]** at 25M. S-U (≈ 370–560
CPU-h) and V-RANK (≈ 96) are worth playing only if S's deviations beat chance on U.

**Why a self-referee is valid for a STOP** (the review's argument, MY JUDGEMENT as it was the reviewer's): T shares
S's model of the opponent (P\* on both sides), so it is the best case for S's choices; E1-0's direction is that gains
shrink against a real opponent. Chance-level deviations under T therefore predict chance or worse against Kakuna. A GO
licenses only the next stage, never a gain.

## 2. The states

- **Games.** Raw P\* against Kakuna greedy, unit seeds **100–109** (outside S-U's 0–59) × 20 games × 4 panels (home72,
  H, F, P72) = **800 games**: 40 whole 20-game units, one anchor group `gate`, label `pstar` with
  `load.input_log = "10-30"` (`main.anchors --keep-input-logs 10-30`). The raw play is unchanged by the flag; every
  decision of ours at turns 10–30 with ≥ 2 legal actions keeps its honest input log (the live searched player's own
  cut), its live search key, tokens, greedy choice and the live row's sha256.
- **ONE engine:** `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129` (`main.anchors.server.engine_stamp()` at the pin;
  main's current engine, the lean screen included). It is the spec's `engine` param: a cell on any other engine VOIDs
  the read. **It is not S-U's registered engine** (`bdc5d2be…/95b45f10…`, before `995c04a8a`'s Struggle fix): an S-U
  amendment that wants this gate as "the read's engine" registers this engine (FINDING G-1).
- **The peer's team pack:** `gen3_peer_showdown_pack_v1` (H22); forum72 plays with `peer_pack_changed` true, the other
  three sets false (stamped per cell, recorded with the read).
- **The draw** (`sign_gate select`, outcome-blind, frozen once written): per game, the 2 kept decisions with the
  smallest `sha256("sugate_v1|<deal key>|<turn>|<decision index>")` (V-RANK's draw), from COUNTED cells only. A game
  with one eligible decision gives one; none, none. Expected ≈ 1,550–1,600 states (≈ 400 a panel).

## 3. Per state (`sign_gate score`)

1. **The root guard.** The kept log is replayed: its tokens must equal the kept decision's and its row must hash to the
   live row's sha256 (G12's guard, offline). A failure is a named defect, never a skip; it holds the scoring.
2. **S exactly as registered:** `LiveSearcher._search` (its keying `su_live_v1|deal|t|n`, its root guards, the forward's
   greedy choice equal to the one we played, the lean screen). An exclusion (no consistent world) is recorded; π plays.
3. **The self-referee on a DEVIATION** (S's pick ≠ π's top-1): `main.value_search.playout.referee` on the TWO actions,
   24 truth seeds (`truth.turn_seeds(key, 40)[16:40]`, the maturity bank's referee layout, disjoint from the search's
   dice), P\* greedy on both sides in the TRUE world, the seeds common across the pair. T(a) = the mean outcome (win 1,
   tie ½, loss 0).
4. **Recorded:** the deviation, the exclusion, the search CPU (worker thread + replay children, and its world /
   playout split), the referee CPU, the leak block (`rewritten_moves`, `rewritten_switches`, `attribution_miss`) and the
   §6.1 Speed ±1 counterfactual flips of the seen mons (`main.info_audit.checks.seed_reuse` on the true root; its
   unflipped control must pass).

**Not done:** the review's optional second keying of S (reproducibility) is not run (FINDING G-2; S's determinism per
key is the live player's tested property, `search_side_integration_test`).

## 4. The GIGO tooth (before any U state is read)

`scripts/bank_check.py` re-scores the review's bank rows (L1's registered loader, 719 states per anchor) with the gate's
own acting rule (`sign_gate.pikl_pick` = `main.value_search.rules`' piKL pick), sign value and estimator
(`cluster_ratio_boot`) at the review's B = 1,000, seed 20261027, and must reproduce every printed digit of
`../su_headroom_2026-10-27/review_results.json`: sign accuracy 0.5132 [0.4468, 0.5801] n 228 (P\*) and 0.5798
[0.5184, 0.6447] n 238 (25M); IMP +0.0001 [−0.0100, +0.0111] and +0.0193 [+0.0083, +0.0305] (L1's registered cell);
the deviation rate 0.4019 / 0.4145. Any difference stops the read before play. (A development run of the same script
on the pin's tree, before this PREREG, already matched every digit; the registered run is at the pin, after it.)

## 5. The rule (`deviation_sign`; the spec's `decision` block)

- **Sign accuracy** = P(T(pick) > T(π top) | S deviates, the two T differ). A state counts in it iff it was searched,
  deviated and its two T differ; per panel the ratio over its countable deviations; **pooled U = the mean of H, F and
  P72's** (equal weight).
- **Interval:** a game-clustered percentile bootstrap, ONE `numpy.random.default_rng(20261027)`, B = 2,000; each
  replicate resamples each panel's games with replacement in the order H, F, P72 (every selected game takes part,
  with or without a countable deviation, as in the review's `boot`); 2.5 / 97.5 percentiles.

| outcome | rule | what follows (the review's table) |
|---|---|---|
| **GO** | LB(pooled U) > 0.50 + 0.0005 | V-RANK's live branches, then the amended S-U (B1's four items, M3, M4) if V-RANK does not turn it OFF |
| **STOP** | otherwise (an LB within 0.0005 of 0.50 is STOP, flagged `boundary`: the review's "rule 8's ε") | No live S-U read, and no V-RANK as its filter. Recorded: "S at P\* is a chance-level perturbation of π on U (and on home72, B2) under its own model". Search stays paused for goal (b). DOES-NOT-CLOSE-B is NOT claimed |

This is the review's rule verbatim, as the precommit memo's ruling 6 requires
([`../../../endstate/precommit_reads_2026-10-27.md`](../../../endstate/precommit_reads_2026-10-27.md) §2.7, §4 item 6);
its routes and traps (§2.7) apply unchanged.

**VOID** (the rule, named): cells on two engines, an unknown one or another than registered; a cell on other bytes than
P\*'s sha256, with a search stamp or a regime other than greedy; a selected state from an uncounted cell, with a wrong
draw hash, or > 2 from one game; a selected state not scored, scored twice, or scored but never selected.

## 6. Power and cost (the review's ESTIMATE)

- **Power.** ≈ 1,200 U states give ≈ 380 countable deviations (40 % deviating, 79 % of those untied, as on the bank):
  SE ≈ 0.026. P(GO) ≈ 0.88 at a true 0.58 (25M-like), ≈ 0.06 at P\*'s home 0.51.
- **Cost.** The review's ESTIMATE was ≈ 20–35 CPU-s a state, **≈ 10–15 CPU-h**. The build's smoke (`835c6b4b2`'s
  message: real P\* on stage 0's 4 banked home72 decisions at turns 10, 29, 32, 43, before this PREREG and outside this
  read) measured S at 4–31 CPU-s, the referee 23 CPU-s on its one deviation, the flips 0.2–1.2 s, so 21–60 s of wall a
  state: **≈ 15–25 CPU-h** for ≈ 1,600 states, plus ≈ 2–3 CPU-h of raw play. Cost is never an outcome: nothing
  re-sizes. Run at concurrency 1, nice 19 on the agent cores, behind N1-P look 1, X59-S and D3 look 2, starting when
  the H-CTX read ends; the scoring in bounded long-job chunks (`--budget-s`, resumable), one at a time.

## 7. Reported, never decided

Home72's sign accuracy (should reproduce B2's ≈ 0.51); the U − home72 difference (sign only, its mechanism); per set:
the deviation rate (the noise dose), the exclusion rate, the countable deviations, IMP (T(pick) − T(top), 0 where S
agrees with π), search and referee CPU, the leak block and the Speed ±1 flip rate. No outcome is printed while scoring
(progress counts only).

## 8. Predictions (GUESSES, before any state)

Pooled-U sign accuracy ≈ 0.52 (0.45–0.60); home72 ≈ 0.51; the deviation rate ≈ 0.40 on home72 and a few points higher
on U (π is less sure there); exclusions ≈ 5–15 % of states, higher on P72 and H; **P(GO) ≈ 0.25**.

## Registration

- Pin `835c6b4b2`; engine `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`; state `~/gen3ai_archive/su_gate_2026-10-27/`;
  the anchor lane at `--concurrency 1`; the artefacts `~/gen3ai_archive/su_gate_2026-10-27/gate/selected.jsonl` and
  `…/scored.jsonl`.
- Order: the bank tooth (§4) → plan and play from a clean detached worktree at the pin → `select` (once) → `score`
  (chunks) → `main.reads record`.
- Committed with [`read_spec.json`](read_spec.json) before any game. Written by the S-U gate operator (orchestrator
  brief, 2026-10-27).
