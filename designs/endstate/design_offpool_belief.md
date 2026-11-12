# The opponent belief off the pool: what is wrong, what fixes it, and what BOUNDARY-1 needs

**Status: DECIDED 2026-11-09 — no head reshape; the fix is the LABELS.** The meter is BUILT
(`python -m main.offpool_belief`, [`../tools/offpool_belief.md`](../tools/offpool_belief.md)) and its baseline is
REPORTED (§3; ledger 2026-11-09). The post-hoc repair of P\* (MF-3 step b) is SCREENED OUT: the light screen X68
([`../research_state/measurements/x68_belief_repair_screen_2026-11-09/SCREEN.md`](../research_state/measurements/x68_belief_repair_screen_2026-11-09/SCREEN.md)
§9) read DROP on 2026-11-09 (§6; screen evidence, never a verdict). Breadth in the loop stays the only shown repair,
and BOUNDARY-1's 2 × 2 tests it. Nothing here is a strength claim: no game read has yet DETECTED a win-rate effect of belief quality
(ledger 2026-10-26, the belief-prior-mix reads at λ = 0.25 and λ = 1).

**The MOVE leg (2026-11-10, §8): DIAGNOSED, and a training lever BUILT off by default.** The deficit is a
species × move table that is over-confident against its own labels, held up by gradient that is not the label's.
The lever is `--belief-grad-mode move_label_only`; its GPU read, the light screen **X69, read DROP on 2026-11-11**
(`../research_state/measurements/x69_move_label_only_screen_2026-11-10/SCREEN.md` §9; §8; screen evidence, never a
verdict): the lever stays off and is closed at the P\* + 20M dose.

This doc answers MF-3 of [`../research_state/EXPERIMENT_BACKLOG.md`](../research_state/EXPERIMENT_BACKLOG.md) and the
review's lever R4 ([`../model/generalisation_review_2026-10-26.md`](../model/generalisation_review_2026-10-26.md)).

## 1. The problem

The network carries a BELIEF about what the opponent has not shown: which species are still unseen, which moves a
revealed mon has not used, which item it holds. P\* (the plateau baseline `pstar_plateau`) trained by self-play, where
every opponent pilots a team from our 719-team pool. On teams outside the pool its belief about the unseen species is
confidently wrong: it gives the species that are really there LESS probability than the Smogon usage prior the
belief starts from (ledger 2026-10-26, H3). A belief that is worse than its own starting point is the cheapest
first-order defect the review found, and the belief is also what a search samples opponent worlds from.

## 2. What the belief IS today (read from the code)

**Every leg is already "Smogon prior + a learned residual that starts at zero".** The request this doc was opened
with (re-parameterise the belief as prior + residual so an untrained head falls back to the prior) is the shape the
code already has:

| leg | where | form | the residual |
|---|---|---|---|
| unseen species | `HypothesisBuilder.species_set` (`src/agents/model/hypothesis_set.py`) | scores = log prior + δ(opponent tokens, revealed slots, global input), then `fixed_mass_presence` (the presences sum to the number of hidden mons) | `delta_global`, `delta_hidden`, `delta_out` (+ `delta_norm`); `delta_out` zero-initialised |
| unshown moves | `MoveBelief` (`src/agents/model/belief_heads.py`) | logits = `move_head(norm(token))` + the prior's logits for that species | `move_head` (128 → 400, linear), zero-initialised |
| unstated item | `ItemBelief` (same file) | logits = `item_head(norm(token))` + log prior | `item_head` (128 → 600, linear), zero-initialised |

Together the three residuals are 212,184 of the network's 5,624,179 parameters. An untrained residual scores
exactly the prior (the meter's λ = 1 column is bit-identical to the three output layers zeroed). **So the failure is
not the parameterisation. It is what the residual was trained on:** the labels of self-play are pool teams, the
residual learns "the opponent's unseen mons are the rest of a pool team", and nothing in training ever tells it
that a team can come from anywhere else.

## 3. The baseline (REPORTED, 2026-11-09)

The meter banks the decisions of P\* playing greedily against itself with one seat on a panel team, and scores each
checkpoint's belief and its prior on the rows of the seat FACING the panel team (two CPU forwards a row). Numbers
are the mean over the panel's 72 teams of each team's mean; intervals are a paired bootstrap over teams (10,000
draws, seed 20261109); a difference is BETTER / WORSE only when its whole interval is on one side of zero. Panels:
`home72` (in the pool, the reference), and the three held-out panels H (`heldout72_hpfix`, ladder teams), F
(`forum72`, forum teams) and P72 (`p72`, procedural teams). Arms: P\*; C0 = five pool-only 20M-step continuations of
P\*; N1-P = five continuations with half of BOTH sides' teams drawn from 12,000 procedural teams (`--team-mix
n1_proc_v1@ALL:0.5 --team-mix-sides both`); X60-T = six continuations with human-built ladder teams in the pool
(`--human-data teams:hr_spec_v1.r3@0.5`).

**Unseen species: mean log presence of the species really there, belief − prior (nats; higher is better).**

| arm (checkpoints) | home72 (in the pool) | H | F | P72 |
|---|---|---|---|---|
| the prior itself (level) | -1.84 | -2.08 | -2.25 | -3.03 |
| P\* (1) | +1.22 [+1.07, +1.36] BETTER | -3.00 [-4.23, -1.86] WORSE | -3.97 [-5.34, -2.79] WORSE | -7.66 [-8.84, -6.51] WORSE |
| C0 (5) | +1.23 [+1.08, +1.38] BETTER | -3.01 [-4.24, -1.87] WORSE | -4.02 [-5.40, -2.85] WORSE | -7.80 [-9.01, -6.63] WORSE |
| N1-P (5) | +0.59 [+0.47, +0.72] BETTER | +0.04 [-0.06, +0.14] not detected | -0.07 [-0.17, +0.02] not detected | -0.13 [-0.21, -0.07] WORSE |
| X60-T (6) | +0.89 [+0.77, +1.01] BETTER | -0.05 [-0.21, +0.10] not detected | -0.29 [-0.48, -0.12] WORSE | -0.84 [-1.01, -0.67] WORSE |
| P\* floored at λ = 0.25 | +1.09 [+0.96, +1.22] BETTER | -0.17 [-0.31, -0.03] WORSE | -0.30 [-0.44, -0.17] WORSE | -0.79 [-0.88, -0.70] WORSE |

**Unshown moves and unstated items, belief − prior (mean log probability of the truth).**

| leg | arm | home72 (in the pool) | H | F | P72 |
|---|---|---|---|---|---|
| unshown moves | P\* | +0.14 [+0.07, +0.22] BETTER | -0.33 [-0.42, -0.26] WORSE | -0.05 [-0.11, +0.02] not detected | -0.34 [-0.40, -0.28] WORSE |
| unshown moves | C0 | +0.14 [+0.07, +0.21] BETTER | -0.33 [-0.42, -0.25] WORSE | -0.06 [-0.12, +0.01] not detected | -0.34 [-0.40, -0.28] WORSE |
| unshown moves | N1-P | +0.08 [+0.02, +0.15] BETTER | -0.30 [-0.39, -0.23] WORSE | -0.13 [-0.19, -0.07] WORSE | -0.37 [-0.42, -0.31] WORSE |
| unshown moves | X60-T | +0.16 [+0.10, +0.23] BETTER | -0.20 [-0.27, -0.13] WORSE | -0.00 [-0.06, +0.05] not detected | -0.25 [-0.30, -0.20] WORSE |
| unstated item | P\* | +0.18 [+0.11, +0.26] BETTER | +0.00 [-0.09, +0.08] not detected | -0.01 [-0.22, +0.18] not detected | -0.39 [-0.63, -0.17] WORSE |
| unstated item | C0 | +0.19 [+0.11, +0.26] BETTER | -0.02 [-0.10, +0.06] not detected | -0.01 [-0.23, +0.18] not detected | -0.40 [-0.64, -0.17] WORSE |
| unstated item | N1-P | +0.14 [+0.07, +0.21] BETTER | +0.03 [-0.03, +0.10] not detected | +0.15 [+0.06, +0.24] BETTER | +0.09 [+0.03, +0.16] BETTER |
| unstated item | X60-T | +0.16 [+0.10, +0.22] BETTER | +0.06 [-0.00, +0.12] not detected | +0.15 [+0.06, +0.24] BETTER | +0.09 [+0.03, +0.16] BETTER |

Units scored a panel (home72 / H / F / P72): species 3,950 / 4,316 / 3,954 / 3,918; moves 28,057 / 25,984 / 26,443 /
26,966; items 7,887 / 7,198 / 8,013 / 9,437. The table of record, with the rank metrics and every arm against C0, is
`~/gen3ai_archive/offpool_belief_2026-11-09/table.md` (`table.json` sha256 `2ddb266d5b2693a8…`).

What it says:

1. **P\* and the pool-only control are WORSE than the prior on all three held-out panels**, by 3 to 8 nats on the
   species leg, and the meter reproduces the one-off probe's numbers (ledger H3: −4.6 / −6.2 / −10.0 against the
   prior's −2.0 / −2.4 / −3.2). In the pool the same belief is 1.2 nats BETTER than the prior. The belief has
   memorised the pool.
2. **Team breadth in the training loop repairs the species leg with the head unchanged.** N1-P's species belief
   is at the prior on H and F and 0.13 nats below it on P72, an improvement of 3 to 8 nats over C0 (+3.05 / +3.95 /
   +7.67, each BETTER), after 20M steps. It still does not BEAT the prior off the pool, and it keeps about half of
   the in-pool advantage (+0.59 against C0's +1.23). **Human ladder teams in the loop (X60-T) do the same on the
   panel of their own family and less elsewhere:** at the prior on H (−0.05), WORSE on F (−0.29) and P72 (−0.84),
   still 3 to 7 nats better than C0 and with more of the in-pool advantage kept (+0.89). No arm beats the prior off
   the pool.
3. **A prior floor does less than breadth.** Mixing P\*'s belief with its own prior at λ = 0.25 (the built
   inference-only `belief_prior_mix`) leaves the species leg WORSE than the prior on all three panels
   (−0.17 / −0.30 / −0.79 on H / F / P72); N1-P is at −0.13 or better.
4. **The move leg is NOT repaired by breadth.** N1-P's unshown-move belief is as far below its prior as C0's on H
   and P72, including on P72, the panel built by the same generator family N1-P trained on; X60-T's is 0.09 to 0.13
   nats better than C0's on H and P72 and still WORSE than the prior there. The gap is real on a
   proper score too (the full Bernoulli log-likelihood over a mon's candidate moves on P72: prior −5.7, P\* −7.8,
   N1-P −7.5 a unit; a 768-row check), so it is not an artefact of scoring only the true moves. **Why: §8**
   (diagnosed 2026-11-10; the proper score puts the deficit at 0.6 to 2.2 nats a unit, 5 to 6 times the column
   above). X68's fine-tune did not repair this leg either (§6).
5. The item leg is at its prior off the pool for P\* and C0 (WORSE only on P72), and breadth moves it to BETTER on F
   and P72.

**What a belief can be at best.** A belief is an estimate of the opponent population it was TRAINED against. Our
deployment opponents (Kakuna, Metamon, Foul Play, people) never bring a pool team except by coincidence, so the
in-pool sharpness is worth nothing against them, and "beat the prior off the pool" needs labels from a population
that resembles the deployment one. Procedural teams are drawn from the Smogon tables the prior is built from, so
they can remove the harm and cannot be expected to add information about human teams; only human-built teams can
(X60-T's row).

## 4. The options

| # | lever | what it can do | cost | state |
|---|---|---|---|---|
| 1 | **Team breadth in the loop** (`--team-mix`, or `--human-data teams:`) | Takes the species leg from 3–8 nats below the prior to the prior (§3) with no model change. Does not repair the move leg. | A recipe line. In-house strength on home72 −3.28 pp at 20M (X61P, ledger 2026-10-30). | BUILT; in BOUNDARY-1's option B |
| 2 | **Post-hoc belief-only fine-tune** (MF-3 step b): retrain only the three residuals of a finished checkpoint on broad-team labels, trunk frozen | Repairs a pool-only checkpoint without retraining it; 100 % broad labels, so no pool half to average with. Its policy then reads a belief it never trained on. | ≈ 2 CPU lane-h, no GPU | BUILT (`main.offpool_belief trainbank` / `finetune`); SCREENED OUT (X68 DROP, §6): better than P\* off the pool, still below the prior, and the in-pool advantage is lost |
| 3 | Inference prior mix λ (`agents.model.belief_prior_mix`) | Caps the harm at −log λ (1.39 nats at 0.25). Measured: still WORSE than the prior at 0.25; no win-rate effect DETECTED at 0.25 or 1. | none | BUILT; not adopted |
| 4 | A trained-in floor: the same mix inside the training forward | The same cap, with the residual trained through it. Cannot beat option 1: the floor only bounds how wrong a wrong residual is. | a model flag, a config version, golden checks | SPECIFIED (§7), not built |
| 5 | A bounded residual: δ ← c · tanh(δ / c) | Caps how far one species' score can move from the prior. The same limit as 4, and it also caps the in-pool gain. | as 4 | SPECIFIED (§7), not built |
| 6 | A penalty that pulls the residual to the prior (KL to prior, residual dropout) | Shrinks the residual everywhere, in the pool as much as off it: the network has no input that says "this team is unfamiliar", so it cannot shrink selectively. | a training flag | REJECTED |
| 7 | An "unfamiliar team" input for the head to gate on | Would let the head choose the prior off the pool. It is a fact about OUR pool, not about the battle: it cannot be computed honestly at deployment, and the pool must not enter the model as a prior. | — | REJECTED |
| 8 | **The move head trained by its labels alone** (`--belief-grad-mode move_label_only`): the losses that read the head's published posterior no longer send gradient into `move_head` | Removes a gradient two orders larger than the label's and unrelated to it (§8). The forward is unchanged, so a finished checkpoint can be forked with it. Cannot add information the pool's labels lack: its ceiling is the pool's own per-species table, at or above the prior on the proper score on H and F, below it on P72. | a value of an existing flag; no config version | BUILT 2026-11-10, off by default; SCREENED OUT at the P\* + 20M dose: light screen X69 read DROP 2026-11-11 (its arms A and AB, §8) |
| 9 | **The move latent loss off** (`--move-belief-latent-coef 0`) | Removes the one loss term whose descent grows the residual: its variance hinge rewards a sharper move belief whatever the truth is (§8). Loses the "a near move grades as near" signal. | an existing training flag | BUILT (pre-existing); SCREENED OUT at the P\* + 20M dose: light screen X69 read DROP 2026-11-11 (its arms B and AB, §8) |

## 5. The decision

**No head reshape, for BOUNDARY-1's base or otherwise.** Reasons, in order of weight:

1. The shape asked for is the shape we have (§2). There is nothing to re-parameterise.
2. The measured cause is the labels, and the measured fix is breadth: N1-P's unchanged head sits at the prior off
   the pool (§3 point 2). Options 4 and 5 can only bound the damage of a wrong residual (to −log λ, or to c); the
   0.25 floor read on P\* shows that bound is worse than what breadth already delivers (§3 point 3).
3. No structure can make the residual shrink only where it is wrong, because nothing the network reads marks a
   team as unfamiliar (options 6, 7).
4. BOUNDARY-1's new arm already changes the width. A second architecture change in the same base would need its
   own config version and golden checks before Friday 2026-11-13, for a lever with no shown game effect (the two λ
   reads against Kakuna are NOT DETECTED) and would break "one lever per A/B".

**What BOUNDARY-1 gets instead.**

- **A breadth arm repairs its own species belief in the loop.** Option B's new arm and the 2 × 2's breadth cells
  carry `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both`; nothing more is needed.
- **A pool-only arm will have P\*'s defect** (option S has no breadth in either arm; every option's control arm is
  pool-only). That is a known property of those arms, not a reason to change their head.
- **The meter is a REPORTED guard on BOUNDARY-1's finals** (≈ 8 CPU-minutes a checkpoint against the banked rows,
  no GPU): each arm's `species_logp` − prior on H, F and P72. It decides nothing in BOUNDARY-1's rule. For an arm
  trained on `n1_proc_v1`, P72 is generator-adjacent, so F is the panel to read (the same hazard as BOUNDARY-1's
  breadth guard).
- **A pool-only final has no shown repair after the fact.** X68 read DROP (§6): retraining the residuals on
  procedural labels with the trunk frozen does not reach the prior, so no checkpoint is repaired that way.

**Can it make Friday's base?** Yes, because it asks nothing of the base: no model change, no new flag, no version.
The breadth recipe line and the meter both exist today.

## 6. The screen: X68, the post-hoc repair of P\*

Registered as a LIGHT SCREEN (`SCREEN.md`, linked in the status block). One lever: the labels P\*'s three residuals
are trained on. The candidate is P\* with only those 212,184 parameters retrained on the labels of 1,200 games
against procedural teams (`n1_proc_v1`, four disjoint 300-team shards, a fifth for validation; the epoch is picked
on the validation shard, no panel row is read). The control is P\* itself and the prior, on the banked rows of §3.
The primary meter is `species_logp` on F and H. It runs on CPU (≈ 2 lane-hours) and needs no GPU lease.

The rule: PROMOTE needed the candidate better than P\* on both panels and no more than 0.25 nats below the prior at
the interval's lower end on both; anything else is DROP. A screen adopts nothing.

**Result: DROP, 2026-11-09** (`SCREEN.md` §9; ledger 2026-11-09 SCREEN; 1.85 CPU lane-h, epoch 4 of 8 picked). Screen
evidence, never a verdict and never quoted as a claim.

| panel | fine-tuned − P\* | fine-tuned − prior | the −0.25 bar |
|---|---|---|---|
| F | +3.23 [+2.23, +4.39] | −0.74 [−1.05, −0.47] | NOT met |
| H | +2.32 [+1.30, +3.44] | −0.68 [−0.94, −0.43] | NOT met |

- **Off the pool it is much better than P\* and still below the prior.** P72 (reported) reads the same way:
  +6.64 against P\*, −1.02 against the prior.
- **The in-pool advantage vanished:** on `home72` the species leg went from +1.22 above the prior (P\*) to −0.18
  below it.
- **The validation loss never reached the prior's:** 10.98 at the picked epoch against 10.25 for the prior alone
  (P\* itself: 23.55), on a shard of the training corpus's own family; later epochs rose while the training loss
  kept falling.
- **The move leg is not repaired:** fine-tuned − prior is WORSE on all three held-out panels (−0.33 H, −0.14 F,
  −0.37 P72). On the full log-likelihood it is 9 to 12 nats a unit below the prior: the fine-tune's species and
  item losses trained `move_head` through its published posterior (§8), so this line is not evidence about
  training the head on its labels.
- **The lesson.** Residual retraining on procedural labels does not reach the prior. Breadth in the loop is the
  measured route (§3), and BOUNDARY-1's 2 × 2 (its breadth cells) tests it; option 4 is the candidate if a later
  fresh pool-only run needs a bound (§7). Step (b) reopens only with a new label source. The fine-tuned
  checkpoint plays nothing and seeds nothing.

**What the screen cannot say.** Procedural labels carry no information about human teams beyond the prior, so
"beats the prior on F and H" is not expected and is not the bar. The human corpus (`hr_spec_v1`) is the only label
source that could, and it enters training only through `--human-data` (`designs/endstate/human_data_architecture.md`);
X60-T's finals are that read, in the loop (§3).

## 7. The fallback, specified and not built

If a later fresh run needs a structural bound (X68 read DROP, so a pool-only run that must still seed a search has
no after-the-fact repair), build
option 4, a trained-in floor, behind one model flag that is off in production:

- **Form.** Species: π ← (1 − λ) · π(prior ⊕ δ) + λ · π(prior), applied in the TRAINING forward as it is at
  inference today, so the residual is fitted through the floor. Moves and items: the same mix on σ and on the
  softmax. λ is a constant in `src/agents/model/arch_constants.py`, the flag a `ModelVersion` field with a config
  version bump, a migration default of off, an `ARCH_SIGNATURE` entry and the golden checks of
  `src/agents/model/CLAUDE.md`.
- **Why 4 over 5.** The floor bounds the LOSS a wrong residual can take (−log λ a unit) and is the form two game
  reads already used; the bounded residual bounds a logit and its loss bound depends on the prior's value.
  (Written for the species leg. On the move leg the bounded residual is the better post-hoc transform on every
  panel, §8; either changes the forward, so neither can be forked from a finished checkpoint.)
- **Its registered test** is the meter of §3 on a matched fresh pair, never a fork: a floor changes what the
  residual learns from step 0.
- **What would make us build it:** X68 DROP together with a game read that DETECTS a cost of the pool-only belief.
  The DROP exists (screen evidence, §6); the game read does not.

## 8. The move leg: diagnosed, and a lever built (2026-11-10)

The record is
[`../research_state/measurements/move_leg_offpool_2026-11-10/README.md`](../research_state/measurements/move_leg_offpool_2026-11-10/README.md)
(ledger 2026-11-10; CPU only, descriptive, no verdict). Numbers are belief − prior on home72 (in the pool) / H / F /
P72, in nats a unit (a revealed opponent mon with a true move not yet shown).

- **The deficit is larger than the meter shows.** `moves_logp` scores only the true moves. On the full Bernoulli
  log-likelihood over a mon's candidate moves (a proper score: it also charges a raised move that is not there)
  P\* reads +0.39 / −1.83 / −0.61 / −2.21, 5 to 6 times the meter's column off the pool, and F is below the
  prior on it. The meter prints that score as `moves_ll` from this change on: a claim about the move leg reads it.
- **What it is: the conditioning on species.** 83 % of `move_head`'s output is a fixed (species, move) offset that
  does not depend on the battle, and that table alone carries half to two thirds of the deficit. 94 to 96 % of the
  true unshown moves are (species, move) pairs the pool attests: the failure is familiar moves in unfamiliar
  combinations, not unknown moves.
- **It is not what memorising the pool gives.** The pool's own per-species move frequencies used as the belief (a
  yardstick, never a prior) score +1.22 / +0.06 / +0.19 / −0.40: at or above the prior on H and F. P\*'s table
  correlates ≈ 0.5 with it and is over-confident against its own labels: the residual scale that maximises the
  log-likelihood is 0.5 in the pool and 0.125 to 0.375 off it, the move label loss's own descent shrinks the
  residual on 8 of 8 in-pool batches, and the harm is in the raises (keeping only the lowerings beats the prior on
  every panel).
- **What holds it there: gradient that is not the label's.** Two measured candidates. (a) *The consumers.* The
  policy, critic, species and item losses reach `move_head` through its published posterior with a mean gradient
  ≈ 350 times the label's, at cosine −0.15 with it. X68's fine-tune shows what this route does when it is the only
  strong signal: its species and item losses (244 and 30 times the label's gradient) raised moves that have 0
  labels (Dugtrio's Explosion from a prior of 0.02 to 0.99), 9 to 12 nats below the prior. (b) *The move latent
  loss's variance hinge* rewards predictions that differ more from slot to slot, whatever the truth is; along the
  residual's scale it outweighs the label loss two to one, at 128 and at 1,024 rows a batch. Which of the two
  holds PRODUCTION's over-confidence is not shown.
- **What breadth can and cannot do.** N1-P recovers a fifth of the deficit on H and P72 on the proper score
  (−1.40 / −1.86 against C0's −1.81 / −2.20). Even a perfect pool table is 0.15 to 0.27 nats below the prior on the
  true moves off the pool: that part needs labels, as the species leg did.
- **The levers (options 8, 9).** `--belief-grad-mode move_label_only` cuts (a) for the move head alone
  ([`../model/phase_pipeline.md`](../model/phase_pipeline.md)); `--move-belief-latent-coef 0` removes (b). Both
  leave the forward bit-identical, so a fork of P\* can carry them and the first read needs no fresh run.
- **Not shown.** No training contrast exists: the routes are gradients at one checkpoint on in-pool rows, and the
  policy term is a proxy. Nothing here is a strength claim, and option 8 removes a signal the policy sends to a
  head it reads: production ran `label_only` on every belief head at gen-11 and is `shaping` today
  ([`../model/op_contracts.md`](../model/op_contracts.md); the reads behind that return are pre-era and are not
  re-derived here: a caution, not a comparator). The read of 2026-10-07 (ledger, MOTIVATED COGNITION) found a move
  head below its prior on the pool in the blob arm and mapped `label_only` onto the per-mon move head.
- **The screen X69 (2026-11-11): DROP.** Three forks of P\* to +20M steps, one run each, against the five banked C0
  finals on `moves_ll` (arm − C0, 72 teams a panel, 95 %; `SCREEN.md` §9, ledger 2026-11-11 SCREEN). Bar: lower
  bound ≥ +0.6 on H, > 0 on F, ≥ −0.2 on `home72`. `move_label_only` (A): H +0.552 [+0.480, +0.624], F +0.571
  [+0.497, +0.649], `home72` +0.531 [+0.463, +0.599]. The move latent loss off (B): +0.046 [+0.001, +0.088] /
  +0.093 [+0.047, +0.140] / +0.046 [+0.015, +0.077]. Both (AB): +0.494 [+0.427, +0.559] / +0.546 [+0.476, +0.621] /
  +0.544 [+0.475, +0.616]. Every arm misses the H bar and meets the other two. A and AB stay 1.27 and 1.33 nats
  below the prior on H. The gradient lever closes at this dose and reopens only with a longer dose argued from
  these numbers; the bounded residual and the trained-in floor (§7, the last bullet here) stay open for a fresh
  run. Screen evidence: one run an arm, no game played, never a comparator of a later claim.
- **Post-hoc transforms of P\*'s residual, for the record** (full log-likelihood): × 0.25, +0.63 / +0.06 / +0.37 /
  −0.04; bounded at one logit by tanh, +0.86 / +0.03 / +0.45 / −0.08; floored on the prior at 0.25, +0.72 / −0.47 /
  +0.12 / −0.61. A bound beats a floor on this leg. Either is a structural change a fork cannot carry, and each
  caps the symptom and leaves its cause.

## Decision record

| date | decision | chosen | rejected, and why | evidence |
|---|---|---|---|---|
| 2026-11-09 | **This doc CREATED; the off-pool belief meter built** (MF-3 builder, orchestrator brief: an offline meter first, then the design) | `main.offpool_belief`: a banked, team-clustered read of each belief leg against its own prior on the registered held-out panels, scoring-only banks outside the checkout | re-running the one-off P5 probe script: a `measurements/` script is a frozen record, never a tool; scoring each checkpoint on its own games: the beliefs would be compared on different decisions | `../tools/offpool_belief.md`; the package's tests |
| 2026-11-09 | **No head reshape; the fix is the labels** | team breadth in the loop for a fresh run (already in BOUNDARY-1's breadth arms); the meter as a REPORTED guard on BOUNDARY-1's finals | a prior + residual re-parameterisation: all three legs already have it (§2); a trained-in floor or a bounded residual in BOUNDARY-1's base: they bound a wrong residual and breadth removes it, the 0.25 floor on P\* is still WORSE than the prior, and no game effect of the belief is DETECTED; a pull-to-prior penalty or a familiarity input: §4 rows 6, 7 | §3's baseline; ledger 2026-11-09 (the baseline), 2026-10-26 (H3 and the two λ reads) |
| 2026-11-09 | **MF-3 step (b) REGISTERED as the light screen X68** | P\*'s three residuals retrained on procedural-team labels, trunk frozen, read on the banked panels; CPU only | the ladder corpus as the label source: a human corpus enters training only through `--human-data`, and X60-T's finals already read human teams in the loop; a full pre-registration: the screen sets no direction and adopts nothing | `SCREEN.md`; `main.offpool_belief trainbank --check` / `finetune --check` |
| 2026-11-09 | **MF-3 step (b) SCREENED OUT: X68 read DROP** | close the post-hoc repair for a frozen trunk at this size; breadth in the loop is the measured route and BOUNDARY-1's 2 × 2 tests it | repairing any pool-only checkpoint by this fine-tune: it stays 0.68 to 0.74 nats below the prior off the pool and loses the in-pool advantage; building option 4 now: no game read DETECTS a cost of the pool-only belief | `SCREEN.md` §9; ledger 2026-11-09 SCREEN (screen evidence, never a verdict) |
| 2026-11-10 | **The move leg DIAGNOSED** (orchestrator brief: is it memorisation of pool movesets, the conditioning on species, or something else) | the conditioning on species is the carrier (a static species × move table, 83 % of the residual); the table is over-confident against its own labels, held up by the consumers' gradient through the publication and by the move latent loss's variance hinge; memorisation of the pool's sets is real and small | "the pool's movesets, so breadth fixes it": an ideal memoriser of the pool is at or above the prior on H and F, and N1-P recovers a fifth; "a mass effect of scoring only true moves": the belief puts less mass on a mon's moves than the prior, and the proper score makes the deficit larger | §8; `../research_state/measurements/move_leg_offpool_2026-11-10/README.md`; ledger 2026-11-10 |
| 2026-11-10 | **A training lever BUILT, off by default: `--belief-grad-mode move_label_only`** (`gen3_move_label_only_v1`, a fourth value of an existing field; no config version) | cut the consumers' gradient into the MOVE head alone, forward bit-identical, so a fork of P\* can read it; the existing `--move-belief-latent-coef 0` is the second lever | `label_only` on every head: it also cuts the species and item heads, whose off-pool repair is breadth, and production left it after gen-11; a bounded residual or a trained-in floor on the move head: structural, needs a fresh pair and a config version, and caps the symptom; a breadth source for moves: N1-P already is one and recovers a fifth | §8; `src/agents/model/belief_label_only_gate_test.py`, `belief_grad_mode_test.py`; `designs/CHANGELOG.md` 2026-11-10 |
| 2026-11-10 | **The levers' GPU read REGISTERED as the light screen X69; NOT LAUNCHED** | three forks of P\* to +20M steps, one run each: `move_label_only` (A), the move latent loss off (B), both (AB), against the five banked C0 finals on the meter's `moves_ll` on H, F and `home72`; ≈ 12.3 GPU-h; it can only PROMOTE an arm to a full registration or DROP | one arm: `move_label_only` does not cut the variance hinge, so A alone cannot tell the two routes apart; `moves_logp` as the primary: blind to an over-confident raise; a game read inside the screen: past the tier's budget, so a promoted arm plays nothing until a full registration reads strength | `../research_state/measurements/x69_move_label_only_screen_2026-11-10/SCREEN.md` |
| 2026-11-11 | **The move-leg gradient lever SCREENED OUT at the P\* + 20M dose: X69 read DROP** | both flags stay at their recipe values (`shaping`, the move latent loss at 0.05); the three forks play nothing and seed nothing; a bounded residual or a trained-in floor stays open for a fresh run | promoting `move_label_only` on its +0.55-nat gain on H: its lower bound (+0.480) is under the registered +0.6 and a screen's rule is not re-read after the look; a longer dose now: it reopens only with an argument from the DROP's own numbers, as a new registration | `SCREEN.md` §9; ledger 2026-11-11 SCREEN (screen evidence, never a verdict) |
