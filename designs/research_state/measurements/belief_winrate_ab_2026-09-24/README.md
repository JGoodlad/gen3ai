# Belief win-rate A/B: does the belief heads' off-pool overconfidence cost GAMES? (2026-09-24)

**Verdict.** Decision-rule **branch 1 fires**, and only branch 1: **DiD = +4.8 pp [+1.9, +7.8]**.
The mechanism is narrower than the branch label ("memorisation COSTS GAMES OFF-POOL") suggests.
On POOL opponents, zeroing the learned belief deltas (PRIOR-ONLY) **loses 5.2 pp of win rate
[−7.4, −3.0]**. On LADDER opponents it changes the win rate by **−0.4 pp [−2.5, +1.7], which is
NOT DETECTED**. The learned deltas are worth about five points of win rate against the pool they
memorised, and nothing detectable against teams from outside it. They do not detectably LOSE games
to the Smogon prior off-pool: E_LADDER is not positive, so the "direct ladder input" clause does not
fire, and neither does branch 3. The calibration read found the heads worse than the prior on ladder
teams. That belief-quality loss does not show up as lost games at this n; what goes missing
off-pool is the heads' pool-side benefit. Zeroing the move-posterior delta by itself reproduces
essentially the whole effect (move-only DiD +4.4 pp [+1.3, +7.4]; its downstream reach into the item
and spread heads is included). At the owner's named strata of k (opponent species revealed): the
greedy action changes on 17.8% (k = 3) and 15.7% (k = 5) of LADDER decisions under ALL, and 18.7% and
16.5% on POOL. The win-rate DiD stratified by k at the first divergent decision is +5.1 pp
[−1.6, +11.9] at k = 3 and +9.8 pp [−6.5, +26.2] at k = 5, both NOT DETECTED (§5 A, E).
**Tag: MEASUREMENT · branch 1 (DiD POSITIVE +4.8 pp [+1.9, +7.8]) · E_pool NEGATIVE −5.2
[−7.4, −3.0] · E_ladder NOT DETECTED −0.4 [−2.5, +1.7] · bit-reproducible (360/360) · proofs
5,992/5,992.**

Pre-registration: [`PREDICTION.md`](PREDICTION.md), commit `1a971e88` (2026-09-24 07:46 PT), which is
on `origin/main`. It was committed before any registered row existed. Every registered endpoint is
reported below. The additions made after the read are labelled POST-REGISTRATION (§5), and the
deviations are listed in §6.

## 1. What ran

* **Model:** `ai_v13_22_popr1_loop/final_model.zip` (sha256 prefix `88fc961a819abe5f`, re-verified at
  analysis time). It was loaded by its own pin `6eb9c776` from a detached worktree
  (`gen3ai-wt/pin-6eb9c776-bwr`, now removed), with its own Rust `sim_bridge` and cwd set to the pin
  tree, through `main.capacity.load_policy` → `check_compatible`.
* **The switch** ([`scripts/prior_switch.py`](scripts/prior_switch.py)) uses forward hooks that zero
  each prior⊕delta Linear. The ALL scope covers 6 heads and MOVE covers `move_belief.move_head` only.
  **The hooks fired in every switched battle:** exactly 6 zeroed calls per decision in all 5,997
  finished PRIOR-ONLY battles, exactly 1 per decision in all 5,999 finished MOVE battles, and 0 in
  every LEARNED battle.
* **Battles:** in-process Rust bridge, CPU, nice 15, 3 workers, concurrency 1. Our side (p1) is the
  model GREEDY with the switch set to OFF, ALL or MOVE. The opponent (p2) is an unmodified copy,
  also GREEDY. Battle index `i` fixes the seed, our team (pool), the POOL opponent and the LADDER
  opponent (Metamon, out-of-pool, Heal Bell and Sleep Talk removed; see
  [`out/manifest.json`](out/manifest.json)).
* **n:** 3,000 indices × 6 cells = **18,000 battles, every registered (cell, index) present, 0
  duplicate lines**. Rows were analysed with `analyze.py`'s rule (the latest line wins); with no
  duplicates, that rule was a no-op.

**Raw rows are NOT committed.** They live in the durable archive `~/gen3ai_archive/belief_winrate_ab_2026-09-24/`:

| file | md5 |
|---|---|
| `rows/rows_w0.jsonl` | `8646a2d60cd8a5e23b6b48e6c7e14cdb` |
| `rows/rows_w1.jsonl` | `d6945bd3df9286737072a3e0f9e53165` |
| `rows/rows_w2.jsonl` | `9b16c16e93a5b7690818ea24c1d47091` |
| `repeat/rows_w0.jsonl` · `w1` · `w2` (registered §5 repeat) | `0855c869eba8bf2ce1f617ea377d9f31` · `21d864114faf19050570a5fe9adc1ac3` · `9e1b2051a30409ca9c9763576d8b8280` |
| `no_side_read/rows_w0.jsonl` · `w1` · `w2` (registered §5 no-perturbation) | `44260fe454972a7ab3f489b818050c70` · `d144139f835f22e271793b634da71848` · `69dc3cd762bec829ef722d76d1a7abb5` |
| `manifest_full.json` (the packed teams) | `7e2f640e839490847d1812e5f179314b` |

The main rows were md5-verified against the run's scratch copy in `/tmp/belief_wr/rows`.

Reproduce the analysis: `python scripts/analyze.py report <archive>/rows out/`, then
`python scripts/post_registration.py <archive>/rows out/` and
`python scripts/first_divergence.py <archive>/rows out/`. Each run takes a few seconds on one core.
The generated tables are in [`out/tables.md`](out/tables.md), and every number is in
[`out/results.json`](out/results.json), [`out/post_registration.json`](out/post_registration.json) and
[`out/first_divergence.json`](out/first_divergence.json). A third session re-ran `analyze.py` and
`post_registration.py` from the archive rows, and the outputs were **byte-identical** to the committed
`out/` files (the bootstraps are seeded).

## 2. Completeness and failures

Fewer than 1% of battles failed in every cell, far below the 25% INCONCLUSIVE bar. **Four indices
are dropped from every cell, which keeps the pairing, so n = 2,996 paired indices** for both the
primary and the secondary analysis.

| cell | attempted | failed | share |
|---|---|---|---|
| pool:learned / pool:prior / pool:move | 3,000 each | 0 | 0.00% |
| ladder:learned | 3,000 | 1 | 0.03% |
| ladder:prior | 3,000 | 3 (2 timeouts) | 0.10% |
| ladder:move | 3,000 | 1 | 0.03% |

| index | cell(s) | failure |
|---|---|---|
| 2542 | ladder:prior, ladder:move | `UnknownVolatileError: 'typechange'` (the opponent's Kecleon, Color Change). **Already fixed on main**: `40330daa` classifies `typechange` as NOT_A_VOLATILE. The pin predates the fix |
| 1541 | ladder:learned | poke-env `available_moves_from_request` assertion: a Solar Beam in the request of a mon whose tracked moves are Counter / Seismic Toss / Explosion / Rollout (Clear Body). **Cause not diagnosed** (§7) |
| 1492, 2345 | ladder:prior | TIMEOUT after **4,101 and 4,691 decisions**, with no finished turn count. Both LADDER opponents carry a Dusclops with **Imprison**. The hypothesis is a rejected-choice loop, **UNVERIFIED** (§7) |

The drops cannot move the verdict. Filling each failed cell with its extreme 0/1 outcome in each
direction (POST-REGISTRATION C) leaves the DiD between **+4.75 [+1.77, +7.77]** and **+4.88 [+1.90,
+7.90]**.

## 3. The registered read

**Win rate** (a draw counts ½; 95% percentile bootstrap, 10,000 resamples of battle indices, drawn
jointly across cells; n = 2,996):

| belief \ opponent teams | POOL | LADDER |
|---|---|---|
| LEARNED | **0.4838** [0.4658, 0.5017] | **0.6649** [0.6479, 0.6819] |
| PRIOR-ONLY (ALL deltas zeroed) | **0.4321** [0.4142, 0.4496] | **0.6614** [0.6442, 0.6786] |
| *MOVE-only (secondary)* | *0.4282* [0.4109, 0.4461] | *0.6532* [0.6362, 0.6701] |

| contrast | point | 95% CI | read |
|---|---|---|---|
| E_POOL = WR_prior − WR_learned | **−5.2 pp** | [−7.4, −3.0] | NEGATIVE |
| E_LADDER = WR_prior − WR_learned | **−0.4 pp** | [−2.5, +1.7] | NOT DETECTED |
| **DiD = E_LADDER − E_POOL (HEADLINE)** | **+4.8 pp** | **[+1.9, +7.8]** | **POSITIVE** |
| *(descriptive) WR_learned, ladder − pool* | +18.1 pp | [+15.6, +20.5] | POSITIVE |

**Secondary: MOVE-only (attribution; no decision rule):**

| contrast | point | 95% CI | read |
|---|---|---|---|
| E^move_POOL | −5.6 pp | [−7.7, −3.3] | NEGATIVE |
| E^move_LADDER | −1.2 pp | [−3.2, +0.9] | NOT DETECTED |
| DiD^move | +4.4 pp | [+1.3, +7.4] | POSITIVE |

In the MOVE arm, the item and spread outputs still move, because they read the tokens after the
move reinjection (PREDICTION §1). So "the move delta" here means the move delta plus its downstream
effect on those two heads.

**The switch changes the outcome of about a third of battles.** The share of indices whose outcome
differs from LEARNED's is 37.3% (pool:prior), 34.2% (ladder:prior), 38.6% (pool:move) and 33.9%
(ladder:move). Greedy play diverges chaotically once one action changes. The pairing still narrows
the CIs relative to the registered worst case (±2.2 pp against the ±2.5 pp planned for a simple
effect).

**Descriptive:**

| cell | mean turns | mean decisions | draws (winner = tie) | capped at turn 250 |
|---|---|---|---|---|
| pool:learned | 44.3 | 48.5 | 5 | 7 |
| pool:prior | 41.6 | 46.0 | 9 | 3 |
| ladder:learned | 46.7 | 50.4 | 8 | 6 |
| ladder:prior | 46.5 | 50.2 | 7 | 10 |
| pool:move | 42.5 | 46.9 | 6 | 2 |
| ladder:move | 46.4 | 50.1 | 10 | 11 |

**The action-change SIDE READ.** At every committed LEARNED decision, the same obs was re-forwarded
with the switch ON. The rate is the share of decisions whose greedy action changes, as a ratio of
sums with a battle bootstrap. k = the number of opponent species revealed (k = 0 never occurs,
because the lead is always revealed). **Sanity: 0 OFF-re-forward mismatches (must be 0). PASS.**

| column:scope | k=1 | k=2 | **k=3** | k=4 | **k=5** | k=6 | all |
|---|---|---|---|---|---|---|---|
| pool:ALL | 18.9 [17.9, 19.9] | 18.8 [17.8, 19.9] | **18.7 [18.0, 19.4]** | 17.5 [16.8, 18.2] | **16.5 [16.0, 17.1]** | 14.5 [13.9, 15.1] | 16.7 [16.4, 17.0] |
| ladder:ALL | 18.5 [17.6, 19.5] | 17.7 [16.9, 18.5] | **17.8 [17.1, 18.6]** | 17.1 [16.4, 17.8] | **15.7 [15.1, 16.3]** | 14.3 [13.7, 14.9] | **16.2 [15.9, 16.5]** |
| pool:MOVE | 16.4 [15.5, 17.4] | 17.3 [16.4, 18.4] | **16.8 [16.1, 17.5]** | 15.8 [15.2, 16.5] | **14.4 [13.8, 14.9]** | 11.9 [11.4, 12.5] | 14.6 [14.3, 14.9] |
| ladder:MOVE | 17.5 [16.6, 18.5] | 16.4 [15.6, 17.2] | **16.4 [15.7, 17.2]** | 15.7 [15.0, 16.5] | **14.4 [13.9, 15.0]** | 12.6 [12.1, 13.1] | 14.7 [14.4, 15.0] |

(% of decisions; each cell's n of decisions is in `out/tables.md`. LADDER has 151,073 decisions in
total and POOL has 145,394.)

**Decision-rule branches:**
1. **FIRES.** The DiD CI [+1.9, +7.8] lies entirely above 0.
2. Does not fire. The DiD is detected, and the ALL / LADDER action-change rate is 16.2%, above the
   10% threshold at every k.
3. Does not fire. E_POOL is below 0, but E_LADDER's CI straddles 0.
* REVERSED and UNRESOLVED do not fire. The "positive E_LADDER" clause does not fire either.

## 4. What it means, and what it does not

* **The learned deltas buy win rate only where they were learned.** On pool teams the deltas are
  worth +5.2 pp. On ladder teams no worth is detected: the delta's worth (−E_LADDER) has a CI of
  [−1.7, +2.5] pp. The heads therefore have their calibration read's memorisation in PLAY as well
  as in the belief metrics. **The DiD does NOT show that the learned heads lose games to the prior
  off-pool.** That would need E_LADDER > 0, and it reads −0.4 pp (NOT DETECTED).
* **The action changes do not explain the asymmetry.** The switch flips about as many greedy actions
  on ladder teams as on pool teams: 16.2% vs 16.7% of decisions. POOL is +0.55 pp [+0.13, +0.97]
  higher (POST-REGISTRATION A), in the opposite direction to the prediction. The count is almost the
  same; what differs is the value of the changed decisions, which cost games on POOL and are neutral
  on LADDER. That reading fits pool-specific knowledge: on POOL, the learned delta is correct about
  the opponent's hidden sets.
* **Assumption behind attributing the DiD to pool-specific knowledge:** zeroing the delta also feeds
  the co-adapted policy an off-distribution posterior. The DiD removes that shock only if it is the
  same size in both columns. This read cannot separate a column-dependent co-adaptation shock from
  pool knowledge. A retrain on wider teams is what tests it.
* **Ladder-campaign input** (branch 1's registered consequence): wider opponent teams in training
  are a concrete candidate, because the belief heads' in-play value does not transfer beyond the
  pool. Branch 3 did not fire, so this read does not argue for keeping the heads as they are; it
  says they currently earn nothing detectable off-pool. The belief-quality result says they are
  also mis-calibrated there.
* **For the NEW lineage's opponent mix:** the read supports putting realistic ladder teams into the
  opponent mix. What they can buy is the belief channel's pool-side value, about +5 pp on the teams it
  was trained against, carried over to ladder teams, and only if heads trained on the wider mix
  transfer. That is an upper bound on the channel's worth, not a measured gain. The read also shows
  no cost to guard against, because the current heads do not detectably lose games on ladder teams.
  **Hazard for that mix:** these teams are Metamon's as filled, about 92% HP Dark. A head trained
  against them would memorise that artifact as it memorised the pool. Repair or re-sample the hidden
  fields (HP type, and EVs and nature where they are Metamon defaults) before they become training
  opponents. Opponents are the implicit channel the Smogon-priors rule allows, so this does not
  touch the priors themselves.
* **Scope:** one checkpoint, greedy vs greedy self-play, against Metamon teams as filled (≈92% HP
  Dark, PREDICTION §7). The opponent's beliefs are unmodified in every cell.

**Predictions scored** (PREDICTION §6):

| prediction | observed | right? |
|---|---|---|
| WR_learned POOL ≈ 0.50 | 0.484 [0.466, 0.502] | ≈ yes |
| WR_learned LADDER 0.62–0.68 | 0.665 | yes |
| action change ALL 10–25% | 16.2–16.7% | yes |
| higher on LADDER than POOL | POOL higher by 0.55 pp | **no** |
| MOVE ≈ half of ALL | MOVE ≈ 87–91% of ALL | **no** |
| E_POOL ≈ −3 pp | −5.2 pp | direction yes, size larger |
| E_LADDER ≈ −1 pp | −0.4 pp | yes |
| DiD ≈ +2 pp, NOT DETECTED | +4.8 pp, DETECTED | **no** (branch 1 was given about 20%) |
| about 30% on branch 3 | did not fire | — |

## 5. POST-REGISTRATION additions (declared; they cannot change the verdict)

These were added after the registered numbers were read. A–D were added by the analysing session
and are implemented in [`scripts/post_registration.py`](scripts/post_registration.py). E was added by
the finishing session and is implemented in [`scripts/first_divergence.py`](scripts/first_divergence.py).

* **A. Owner addition: side read at k = 3 and k = 5 as named strata.**

  | stratum | ALL: POOL | ALL: LADDER | POOL − LADDER | MOVE: POOL | MOVE: LADDER | POOL − LADDER |
  |---|---|---|---|---|---|---|
  | k = 3 | 18.7% | 17.8% | +0.90 pp [−0.18, +1.92] NOT DETECTED | 16.8% | 16.4% | +0.36 [−0.60, +1.35] NOT DETECTED |
  | k = 5 | 16.5% | 15.7% | +0.82 pp [−0.02, +1.66] NOT DETECTED | 14.4% | 14.4% | −0.06 [−0.82, +0.73] NOT DETECTED |
  | all k | 16.7% | 16.2% | +0.55 pp [+0.13, +0.97] POSITIVE | 14.6% | 14.7% | −0.15 [−0.55, +0.27] NOT DETECTED |

  The change rate falls monotonically with k, from about 19% at k = 1 to about 14% at k = 6, in both
  columns. The heads matter more while more of the opponent's team is hidden.
* **B. Owner addition: the WIN-RATE effect at k = 3 and k = 5, stratified by K_final. INVALID;
  the valid version is E below.** k is a per-decision quantity. K_final, the LEARNED trajectory's final
  reveal count, is a selection on the control arm's own outcome. **Every LEARNED battle with K_final ≤ 4 is a LEARNED loss** (WR_learned = 0.000), and
  K_final = 5 has WR_learned 0.000 on POOL and 0.008 on LADDER. The battle only ends before the last
  reveal when WE lose. In those strata, E = WR_prior − WR_learned is ≥ 0 by construction.

  | K_final | POOL n | POOL WR_l / WR_p | LADDER n | LADDER WR_l / WR_p |
  |---|---|---|---|---|
  | 3 | 49 | 0.000 / 0.163 | 38 | 0.000 / 0.395 |
  | 5 | 418 | 0.000 / 0.251 | 256 | 0.008 / 0.504 |
  | 6 | 2,329 | 0.622 / 0.482 | 2,582 | 0.771 / 0.694 |

  These numbers are shown only to show why the stratifier is invalid; none of them is an effect.
* **E. Owner addition, the valid version: the WIN-RATE effect stratified by k at the FIRST DIVERGENT
  decision** ([`scripts/first_divergence.py`](scripts/first_divergence.py),
  [`out/first_divergence.json`](out/first_divergence.json)). Greedy vs greedy with a fixed seed means
  the LEARNED and switched trajectories of an index are identical up to the first decision at which
  the switch changes the greedy action. The LEARNED row's side read names that decision, and its k
  (`k_div`) is a function of that SHARED prefix. So membership is the same under both arms and is
  fixed before the treatment touches the trajectory, which makes it a valid stratifier. **The premise
  was checked, not assumed:** every index with no divergence replays identically under the switched
  arm (winner, turns, n_dec), with 0 violations in 20 / 19 (ALL) and 32 / 21 (MOVE) such indices, and
  no switched battle ends before its divergence decision. **Meaning:** the effect of the switch left
  ON for the whole battle, in battles whose first switch-changed decision happens with k opponent
  species revealed. It is not "the switch at k only". The first divergence is early: the median is
  decision 0 at k = 1, decision 5–6 at k = 3 and decision 13 at k = 5. Only about 3% of indices first
  diverge at k = 5, so that stratum is small. 95% percentile bootstrap, 10,000 resamples within each
  stratum.

  | arm | k_div | POOL n | E_POOL (pp) | LADDER n | E_LADDER (pp) | DiD (pp) |
  |---|---|---|---|---|---|---|
  | ALL | **3** | 570 | −4.5 [−9.5, +0.4] ND | 543 | +0.6 [−3.9, +5.1] ND | **+5.1 [−1.6, +11.9] NOT DETECTED** |
  | ALL | **5** | 92 | −4.3 [−16.3, +7.6] ND | 91 | +5.5 [−5.5, +16.5] ND | **+9.8 [−6.5, +26.2] NOT DETECTED** |
  | MOVE | **3** | 600 | −5.1 [−10.1, −0.1] NEG | 572 | +1.0 [−3.7, +5.6] ND | +6.0 [−0.8, +12.9] NOT DETECTED |
  | MOVE | **5** | 113 | −5.3 [−15.9, +5.3] ND | 109 | −1.8 [−11.9, +8.3] ND | +3.5 [−11.0, +18.0] NOT DETECTED |
  | ALL | 1 | 1,112 | −5.9 [−9.4, −2.3] NEG | 1,114 | −2.1 [−5.7, +1.6] ND | — |
  | ALL | 2 | 904 | −5.2 [−9.2, −1.3] NEG | 912 | −0.2 [−4.0, +3.7] ND | — |

  **Read:** at k = 3 and k = 5 every stratum effect and both DiDs are NOT DETECTED, except MOVE's
  E_POOL at k = 3, whose upper bound is −0.1. The point estimates have the same sign pattern as the
  headline (POOL negative, LADDER near 0, DiD positive, about +5 pp at k = 3), but each stratum holds
  about 1/5 (k = 3) or 1/30 (k = 5) of the battles, and it cannot resolve a 5 pp DiD alone. Nothing
  here says the effect depends on k. The headline DiD is carried by the pooled read, not by any one
  stratum.
* **C. Drop sensitivity:** see §2. The DiD's worst case is +4.75 [+1.77, +7.77].
* **D. Turn-cap sensitivity.** The 39 battles that reached turn 250 ended with both sides
  stall-forfeiting. The sim always processes p1's FORCELOSE first (`gen3_cf_draw_at_cap_v1`), so
  every one of them is recorded as OUR loss (recorded winner = 0 in all 39). They are draws by
  construction. Re-scoring them as ½ gives: **DiD +4.96 [+2.00, +7.94] POSITIVE**; E_POOL −5.24
  [−7.41, −3.10]; E_LADDER −0.28 [−2.34, +1.84] NOT DETECTED; DiD^move +4.56 [+1.54, +7.58]. The
  verdict is unchanged. §3 keeps the registered scoring (the recorded winner).

## 6. Deviations from the registration

1. **Analysed by a successor session.** The registering session ran every battle, then ended in a
   session restart before any analysis. `scripts/analyze.py` was written after the registration
   commit: the file is timestamped 07:48, and `1a971e88` is 07:46. It was exercised on synthetic
   rows (`/tmp/belief_wr/synth`) and left uncommitted. It is committed here unchanged. I checked it
   against §3–§4 and it matches: 10,000 index-bootstrap resamples drawn jointly across cells,
   failed indices dropped from every cell, the side read as a ratio of sums over resampled battles.
   **UNVERIFIED:** I cannot prove that the registering session read no registered row between 07:46
   and its end. The script does not look tuned (it is the generic §3 recipe), and its output is
   the only thing this record reports.
2. **The registered §5 checks (the repeat and the no-perturbation run) were run by the analysing
   session, AFTER the main read.** They used the same code, manifest and 3-worker `i mod 3` sharding,
   on a much busier box (load ≈ 32 on 16 cores). **Repeat:** indices 0–59, all 6 cells, with
   `--proof-decisions 1000`. **360 / 360 rows identical** on (status, winner, turns, n_dec, side).
   **Proofs (a) OFF == unmodified, (b) ALL ON == prior, and (c) MOVE ON: 5,992 / 5,992 decisions**,
   with pi logits differing under ALL on 5,992 / 5,992. **No-perturbation:** LEARNED cells with
   `--no-side-read`, **120 / 120 identical** on (winner, turns, n_dec). The read is bit-reproducible.
3. **The timeout label is wrong.** The rows' `err` says `wall>1800s`, but both timeouts came from
   the bridge runner's own per-battle backstop: `_PER_BATTLE_TIMEOUT` 180 s × the contention scale,
   wall 180 s and 329 s, raised as a `TimeoutError` inside `run_local_battles`. The registered
   1,800 s bound never bound anything. Either way the battles are INCONCLUSIVE, as registered.
4. **POST-REGISTRATION A–E** (§5) were added after the read. The owner asked for A (the side read at
   k = 3 and k = 5) and for the win-rate effect at those k, which is B (invalid) and E (valid). C and D
   are sensitivity checks.
5. **Finished by a third session.** The analysing session was paused by the owner after it had
   written everything above except §5 E, and before it committed. The finishing session re-verified
   the archive md5s, confirmed that every (cell, index) was present with 0 duplicate lines,
   reproduced `out/` byte-identically, re-checked `analyze.py` against PREDICTION §3–§4, added §5 E,
   and committed the record.

## 7. Findings and hazards

* **Imprison: a possible decision livelock (UNVERIFIED).** Two PRIOR-ONLY battles against LADDER
  teams with Dusclops + Imprison ran 4,101 and 4,691 decisions without finishing. That is far more
  than the turn cap allows at one decision per turn, which suggests a rejected-choice re-request
  loop. Supporting evidence: `designs/ops/TECH_DEBT_BACKLOG.md`'s P2 `ab_replay` row records the sim
  REJECTING a choice (Will-O-Wisp) that Dusclops's Imprison hides. The request does not mark the move
  disabled, and a greedy policy re-offered the same obs would choose it again. Not diagnosed. If it is
  real, it would hang a live ladder game on the timer. It is not in the backlog as an agent-side hazard.
* **A poke-env tracker assertion on a ladder battle** (index 1541): `available_moves_from_request`
  sees Solar Beam on a mon tracked as Counter / Seismic Toss / Explosion / Rollout. Not diagnosed,
  and it is not in the backlog. On a live game it would kill the parse task.
* **Capped battles are scored as p1 losses** (§5 D). Any reader of these rows or of
  `local_battle_runner` outcomes who counts `winner` at turn 250 inherits the artifact.
* `typechange` (Kecleon) crashed the pinned encoder; it is already fixed on main (`40330daa`).

## Files

* [`PREDICTION.md`](PREDICTION.md): the registration (`1a971e88`).
* `scripts/`: [`prep_manifest.py`](scripts/prep_manifest.py), [`prior_switch.py`](scripts/prior_switch.py),
  [`run_ab.py`](scripts/run_ab.py), [`analyze.py`](scripts/analyze.py) (registered analysis, as used),
  [`post_registration.py`](scripts/post_registration.py) (§5 A–D),
  [`first_divergence.py`](scripts/first_divergence.py) (§5 E).
* `out/`: `manifest.json` (team shas and seeds), `results.json` and `tables.md` (registered),
  `post_registration.json`, `first_divergence.json`, `repeat_check.json`, `no_side_read_check.json`.
