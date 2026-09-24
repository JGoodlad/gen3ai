# PRE-REGISTRATION: does the belief heads' off-pool overconfidence cost WIN RATE? (2026-09-24)

**Committed BEFORE any registered number exists.** Two plumbing smokes ran first (indices 0–3,
written to scratch, NEVER analysed). They printed only status, wall time, decision counts and the
switch-proof counters. No winner, win rate or action-change rate was read from them.

## 0. The question

UNDERSTANDING.md §6: *does the belief heads' off-pool overconfidence cost WIN RATE?* The calibration
read ([`../belief_calibration_2026-09-24/`](../belief_calibration_2026-09-24/README.md)) found the
learned belief heads WORSE than the Smogon prior they fuse with on Metamon ladder teams (species
−1.28 nats; the reinjected revealed-slot move posterior −7.8 pp hidden-move recall). That is a
belief-QUALITY finding. This read asks whether it reaches PLAY: switch the learned deltas off at
inference and measure the win rate on pool vs ladder opponent teams.

## 1. The knob: the PRIOR-ONLY switch (`scripts/prior_switch.py`)

Every prior-fused belief head is `posterior logits = Smogon prior buffer ⊕ Linear(read)`, and the
Linear (the learned delta) is zero-init, so a zero delta IS the prior. The switch registers a forward
hook on each delta Linear that, while ON, replaces its output with exact zeros. Nothing else changes:
revealed moves are still pinned certain (applied after the delta), the typed-HP composition still
runs (on the prior HP-type posterior), and the trained `reinject` projections still soft-embed the
now-prior posterior. It is a measurement-script hook; no production code changes.

| Scope | Delta heads zeroed | Reaches pi/vf? |
|---|---|---|
| **ALL** (the PRIOR-ONLY arm) | `move_belief.move_head` (reinjected move posterior), `hp_type_belief_head.type_head` (typed-HP composition + reinject), `spread_belief.nature_head` + `ev_head` (op + reinject), `item_belief_head.item_head` (P(Choice Band) in the op), `belief_head.species_head` (the species side readout, covered for completeness) | yes, except the species readout |
| **MOVE** (secondary arm) | `move_belief.move_head` only | yes |

NOT covered, because they are not prior⊕delta heads: `BeliefHead.moves_head` (a from-scratch side
readout), `T0SpeciesPrior` (parameter-free Smogon naive Bayes, already prior-only), the intent heads,
and the move-latent table. Under the MOVE scope, the item and spread heads' OUTPUTS still change,
because they read the opponent tokens after the move reinjection (`extractor_forward`
`_spread_hp_damage(role_tokens[:, TEAM_SIZE:])`). Their own deltas stay learned.

**Proof** (`run_ab.py --proof-decisions`, counted per decision, never raised). The smokes already
ran it: it passed on 305 of 305 decisions for (a) and (b), and on 145 of 145 for (c). The first smoke
failed (c) on all 160 decisions, because (c) then also demanded that the item logits be unchanged,
which was wrong for the reason above. After that clause was removed, (c) passed on 145 of 145. It is
re-run on the registered repeat slice (§5).
* (a) **OFF == unmodified:** with the hooks registered but OFF, the trainee's pi logits, value, greedy
  action and every belief stash are `torch.equal` to an independently loaded, hook-free copy of the
  checkpoint.
* (b) **ALL ON == prior exactly:** every covered stash (`move_belief_logits`, `hp_type_logits`,
  `item_logits`, `spread_nature_logits`, `spread_ev`, `spread_belief`, species logits) is
  `torch.equal` to its prior-buffer formula with delta := 0, and pi logits differ from OFF.
* (c) **MOVE ON:** the move posterior equals the prior move row (reveals pinned, composed with the
  LEARNED HP-type posterior), and the HP-type logits equal OFF's.

## 2. Design

* **Model:** `models/ai_v13_22_popr1_loop/final_model.zip` (sha256 prefix `88fc961a819abe5f`;
  belief config: move/species prior fusion, `move_belief_mode both`, `hp_belief_mode composed`,
  item belief, nature/EV spread belief, `belief_grad_mode shaping`). It is loaded by ITS OWN pin
  `6eb9c776`: a detached worktree at that commit (`gen3ai-wt/pin-6eb9c776-bwr`, NOT the live run's
  `/tmp/launcher-6eb9c776-*` tree), cwd = that tree (`git diff 6eb9c776 main -- data/` is empty), its
  own Rust build via `POKESIM_SIM_BRIDGE_BIN`, loaded through `main.capacity.load_policy` →
  `check_compatible`.
* **Battles:** in-process Rust bridge, CPU, nice 15, 3 worker processes, concurrency 1,
  `torch` 1 thread. **Our side (p1)** is the model GREEDY with the switch OFF (LEARNED), ALL
  (PRIOR-ONLY) or MOVE (secondary). **The opponent (p2)** is a separate, UNMODIFIED copy of the model,
  GREEDY.
* **2×2 (+ secondary):** belief {LEARNED, PRIOR-ONLY} × opponent teams {POOL, LADDER}, plus
  {MOVE} × {POOL, LADDER} as the secondary attribution arm.
* **Pairing:** battle index `i` fixes the sim seed (`20260925·1000 + i`), our team, the POOL
  opponent team and the LADDER opponent team. All six cells of index `i` share them. Only the switch
  (rows) and the opponent's team (columns) vary.
* **Teams** (`scripts/prep_manifest.py`, `out/manifest.json`):
  * **Our team and the POOL opponent:** uniform with replacement from the 719-team pool minus the
    29 Sleep Talk teams (690 kept; the pool has 0 Heal Bell teams).
  * **LADDER opponent:** the calibration read's registered Metamon `hl_05_26/gen3ou` filter (15,154
    of 22,862: Showdown-legal, port coverage predicates, out-of-pool species set, unique species set)
    minus 363 Heal Bell teams (the obs encoder crashes on the `healbell` volatile; backlog P1). That
    leaves **14,791 (64.7% of the download)**, with 0 Sleep Talk (the coverage predicates already
    drop it). Distinct teams, drawn by seeded permutation, all accepted by the pinned
    `Gen3Teambuilder`.
* **n: 3,000 paired indices per cell** (12,000 primary battles plus 6,000 secondary). Why 3,000
  rather than the suggested 2,000: the smoke measured about 1–7 s per battle, so 3,000 fits in about
  4–5 h on 3 workers, and it moves the worst-case minimum detectable effect as follows.
  Worst case means independent paired outcomes, p ≈ 0.5: the SE of a simple effect is
  √(2·0.25/n), and the DiD's SE is √2 times that. The MDE is 2.8·SE (80% power, α 0.05).
  | n | simple-effect 95% CI half-width | DiD 95% CI half-width | DiD MDE (80%) |
  |---|---|---|---|
  | 2,000 | ±3.1 pp | ±4.4 pp | 6.3 pp |
  | 3,000 | ±2.5 pp | ±3.6 pp | **5.1 pp** |

  The real CIs will be narrower: a battle in which the switch never changes an action replays
  identically (greedy vs greedy, fixed seed), so the pair is concordant.
* **No optional stopping:** the run goes to n = 3,000 and is analysed once.

## 3. Endpoints

**WR** = our wins / battles (a draw counts ½), per cell. **Simple effects:**
`E_col = WR_prior − WR_learned` for col ∈ {POOL, LADDER}. **HEADLINE:**
**DiD = E_LADDER − E_POOL.** CIs are 95% percentile bootstrap, 10,000 resamples of battle INDICES,
drawn jointly across all cells (paired). No correlation is ever computed across battles.

**SIDE READ** (LEARNED cells, along the LEARNED trajectories): at every committed trainee decision,
the same obs is also forwarded with the switch ON (ALL, and separately MOVE). The **action-change
rate** is the share of decisions whose greedy action differs from LEARNED's. It is reported per
column and scope, stratified by k = the number of opponent species revealed in the obs (0–6), with
battle-bootstrap CIs (a ratio of sums over resampled battles). A sanity column (the OFF re-forward's
action must equal the committed action) must be exactly 0.

**Secondary (attribution; no decision rule):** `E^move_col = WR_move − WR_learned`, and its DiD.

**Descriptive:** turns per battle and decisions per battle, per cell.

## 4. Decision rule

On the primary cells:
1. **DiD > 0 with its CI entirely above 0:** memorisation COSTS GAMES OFF-POOL. Wider opponent teams
   in training become a concrete ladder-campaign input.
2. **DiD NOT DETECTED (the CI straddles 0) AND a LOW action-change rate** (ALL scope, LADDER column,
   all strata, < 10% of decisions): the overconfidence is mostly COSMETIC for play.
3. **PRIOR-ONLY hurts in BOTH columns** (E_POOL's and E_LADDER's CIs both entirely below 0): the
   policy DEPENDS on the learned delta. That argues for retraining the heads on wider data, not
   removing them.

Branches are not exclusive: every branch that fires is reported (e.g. 1 + 3 means the delta is
load-bearing, and it is worth less off-pool than on). Two further outcomes are named in advance:
* **DiD < 0 with its CI entirely below 0: REVERSED.** The learned deltas are worth MORE off-pool
  than on-pool, the opposite of the memorisation cost.
* **DiD NOT DETECTED with an action-change rate ≥ 10%: UNRESOLVED.** Actions change, but the win
  rate does not detectably follow at this n. The CI is stated, and it is never read as
  "equivalent".

A positive E_LADDER alone (CI above 0: PRIOR-ONLY wins more on ladder teams) is reported as a direct
ladder input whatever the DiD.

**Rules of evidence.** A CI that straddles 0 reads "NOT DETECTED", never "equivalent". No pooled
correlation across battles. A battle that errors or exceeds its wall bound (1,800 s) is
INCONCLUSIVE and never a result. Its index is dropped from every primary cell (and from the
secondary analysis), so pairing survives. The drop count is reported per cell. **If more than 25% of
the attempted battles in any cell fail, the read is INCONCLUSIVE.** A timeout is never a semantic
outcome.

## 5. Reproducibility and non-perturbation checks (registered)

* **Repeat:** indices 0–59, all six cells, re-run into a separate directory. Every row's (status,
  winner, turns, n_dec, side read) must be identical. This slice runs with `--proof-decisions 1000`,
  so it also re-runs proofs (a)–(c) on every decision in it.
* **The side read does not perturb play:** indices 0–59, the LEARNED cells, re-run with
  `--no-side-read`. (winner, turns, n_dec) must equal the side-read run.
* Any mismatch is reported as a finding, and the read is flagged NOT bit-reproducible.

## 6. Predictions (mine, before any number)

* WR_learned: POOL ≈ 0.50, LADDER ≈ 0.62–0.68. The calibration read saw 0.51 / 0.66 with the same
  pilot, on different draws.
* Action-change rate (ALL scope): 10–25% of decisions, higher on LADDER than POOL. MOVE scope:
  about half of ALL's.
* E_POOL ≈ −3 pp: the policy is co-adapted to its own delta, so the prior is off-distribution input.
  E_LADDER ≈ −1 pp. **DiD ≈ +2 pp, NOT DETECTED** at n = 3,000. That is branch UNRESOLVED if the
  action-change rate is ≥ 10%. I put about 30% on branch 3 firing (PRIOR-ONLY hurts both), and
  about 20% on branch 1.

## 7. Known caveats (stated in advance)

* **HP type on LADDER is contaminated:** about 92% of Metamon Hidden Powers are HP Dark (the
  calibration read, §3). The LADDER column therefore tests play against Metamon's teams as filled,
  not a realistic HP-type distribution.
* **The opponent's own beliefs are unmodified** and see OUR (pool) team in both columns, so its
  belief quality is constant across columns and rows.
* **One checkpoint, one pilot (greedy self).** Nothing here speaks to stochastic play or to
  other checkpoints.
