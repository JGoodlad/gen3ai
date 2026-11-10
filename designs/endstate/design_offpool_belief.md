# The opponent belief off the pool: what is wrong, what fixes it, and what BOUNDARY-1 needs

**Status: DECIDED 2026-11-09 — no head reshape; the fix is the LABELS.** The meter is BUILT
(`python -m main.offpool_belief`, [`../tools/offpool_belief.md`](../tools/offpool_belief.md)) and its baseline is
REPORTED (§3; ledger 2026-11-09). The post-hoc repair of P\* (MF-3 step b) is REGISTERED as the light screen X68
([`../research_state/measurements/x68_belief_repair_screen_2026-11-09/SCREEN.md`](../research_state/measurements/x68_belief_repair_screen_2026-11-09/SCREEN.md)),
not run. Nothing here is a strength claim: no game read has yet DETECTED a win-rate effect of belief quality
(ledger 2026-10-26, the belief-prior-mix reads at λ = 0.25 and λ = 1).

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
   N1-P −7.5 a unit; a 768-row check), so it is not an artefact of scoring only the true moves. **UNVERIFIED:**
   why. Candidates: a linear head on a token that the pool shaped; 20M steps too few to re-fit it; a difference
   between `p72`'s and `n1_proc_v1`'s set generators. X68 reads this leg directly (§6).
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
| 2 | **Post-hoc belief-only fine-tune** (MF-3 step b): retrain only the three residuals of a finished checkpoint on broad-team labels, trunk frozen | Repairs a pool-only checkpoint without retraining it; 100 % broad labels, so no pool half to average with. Its policy then reads a belief it never trained on. | ≈ 2 CPU lane-h, no GPU | BUILT (`main.offpool_belief trainbank` / `finetune`); X68 REGISTERED |
| 3 | Inference prior mix λ (`agents.model.belief_prior_mix`) | Caps the harm at −log λ (1.39 nats at 0.25). Measured: still WORSE than the prior at 0.25; no win-rate effect DETECTED at 0.25 or 1. | none | BUILT; not adopted |
| 4 | A trained-in floor: the same mix inside the training forward | The same cap, with the residual trained through it. Cannot beat option 1: the floor only bounds how wrong a wrong residual is. | a model flag, a config version, golden checks | SPECIFIED (§7), not built |
| 5 | A bounded residual: δ ← c · tanh(δ / c) | Caps how far one species' score can move from the prior. The same limit as 4, and it also caps the in-pool gain. | as 4 | SPECIFIED (§7), not built |
| 6 | A penalty that pulls the residual to the prior (KL to prior, residual dropout) | Shrinks the residual everywhere, in the pool as much as off it: the network has no input that says "this team is unfamiliar", so it cannot shrink selectively. | a training flag | REJECTED |
| 7 | An "unfamiliar team" input for the head to gate on | Would let the head choose the prior off the pool. It is a fact about OUR pool, not about the battle: it cannot be computed honestly at deployment, and the pool must not enter the model as a prior. | — | REJECTED |

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
- **If X68 PROMOTES**, a pool-only final can be repaired after the fact in ≈ 2 CPU lane-hours; whether to do that
  for any checkpoint that plays or seeds a search is the full registration's question (§6), not this doc's.

**Can it make Friday's base?** Yes, because it asks nothing of the base: no model change, no new flag, no version.
The breadth recipe line and the meter both exist today.

## 6. The screen: X68, the post-hoc repair of P\*

Registered as a LIGHT SCREEN (`SCREEN.md`, linked in the status block). One lever: the labels P\*'s three residuals
are trained on. The candidate is P\* with only those 212,184 parameters retrained on the labels of 1,200 games
against procedural teams (`n1_proc_v1`, four disjoint 300-team shards, a fifth for validation; the epoch is picked
on the validation shard, no panel row is read). The control is P\* itself and the prior, on the banked rows of §3.
The primary meter is `species_logp` on F and H. It runs on CPU (≈ 2 lane-hours) and needs no GPU lease.

- **PROMOTE** (better than P\* on both panels, and no more than 0.25 nats below the prior at the interval's lower
  end on both): the repair earns a full registration, which must put a number on what a repaired belief is WORTH
  (a game read against Kakuna on unfamiliar teams, and the belief as the searched player's world sampler). A
  screen adopts nothing.
- **DROP**: P\*'s belief cannot be repaired after the fact at this size with a frozen trunk. Breadth in the loop
  stays the only shown repair, and option 4 becomes the candidate for the next fresh run.
- It also reads, as reported columns, the leg breadth did not repair (moves) and what the repair costs in the pool.

**What the screen cannot say.** Procedural labels carry no information about human teams beyond the prior, so
"beats the prior on F and H" is not expected and is not the bar. The human corpus (`hr_spec_v1`) is the only label
source that could, and it enters training only through `--human-data` (`designs/endstate/human_data_architecture.md`);
X60-T's finals are that read, in the loop (§3).

## 7. The fallback, specified and not built

If a later fresh run needs a structural bound (X68 DROPS and a pool-only run must still seed a search), build
option 4, a trained-in floor, behind one model flag that is off in production:

- **Form.** Species: π ← (1 − λ) · π(prior ⊕ δ) + λ · π(prior), applied in the TRAINING forward as it is at
  inference today, so the residual is fitted through the floor. Moves and items: the same mix on σ and on the
  softmax. λ is a constant in `src/agents/model/arch_constants.py`, the flag a `ModelVersion` field with a config
  version bump, a migration default of off, an `ARCH_SIGNATURE` entry and the golden checks of
  `src/agents/model/CLAUDE.md`.
- **Why 4 over 5.** The floor bounds the LOSS a wrong residual can take (−log λ a unit) and is the form two game
  reads already used; the bounded residual bounds a logit and its loss bound depends on the prior's value.
- **Its registered test** is the meter of §3 on a matched fresh pair, never a fork: a floor changes what the
  residual learns from step 0.
- **What would make us build it:** X68 DROP together with a game read that DETECTS a cost of the pool-only belief.
  Neither exists today.

## Decision record

| date | decision | chosen | rejected, and why | evidence |
|---|---|---|---|---|
| 2026-11-09 | **This doc CREATED; the off-pool belief meter built** (MF-3 builder, orchestrator brief: an offline meter first, then the design) | `main.offpool_belief`: a banked, team-clustered read of each belief leg against its own prior on the registered held-out panels, scoring-only banks outside the checkout | re-running the one-off P5 probe script: a `measurements/` script is a frozen record, never a tool; scoring each checkpoint on its own games: the beliefs would be compared on different decisions | `../tools/offpool_belief.md`; the package's tests |
| 2026-11-09 | **No head reshape; the fix is the labels** | team breadth in the loop for a fresh run (already in BOUNDARY-1's breadth arms); the meter as a REPORTED guard on BOUNDARY-1's finals | a prior + residual re-parameterisation: all three legs already have it (§2); a trained-in floor or a bounded residual in BOUNDARY-1's base: they bound a wrong residual and breadth removes it, the 0.25 floor on P\* is still WORSE than the prior, and no game effect of the belief is DETECTED; a pull-to-prior penalty or a familiarity input: §4 rows 6, 7 | §3's baseline; ledger 2026-11-09 (the baseline), 2026-10-26 (H3 and the two λ reads) |
| 2026-11-09 | **MF-3 step (b) REGISTERED as the light screen X68** | P\*'s three residuals retrained on procedural-team labels, trunk frozen, read on the banked panels; CPU only | the ladder corpus as the label source: a human corpus enters training only through `--human-data`, and X60-T's finals already read human teams in the loop; a full pre-registration: the screen sets no direction and adopts nothing | `SCREEN.md`; `main.offpool_belief trainbank --check` / `finetune --check` |
