# RUNBOOK — gen-13 end-of-run audit battery → the gen-14 config

**Pre-registered 2026-08-17, BEFORE gen-13 launches** — decision rules written before the numbers
exist (the concat-deletion precedent). Run from the RUN'S OWN pinned worktree; write reports into
`designs/research_state/measurements/` with provenance.

Gen-13 = `ai_v9_15_gen13_hb_events_stack_0817`, fresh-init at **v89** (`1fa4733`), 25M steps.
Delta from gen-12: `--history-events` (v81) · `h` graduates + `r` added to the family string ·
`--item-belief` (v83) · `--intent-threshold` (v84) · `--intent-conditional` (v85) ·
`--damage-matrices both` · `--op-drop-renders` + `--op-believed-lean` (v86) ·
`--value-entity-pool-full` (v82) · `--value-clock` + `--value-intent` (v87).

---

## 0. ⚠️ THE ARTIFACT RULE — decided in advance, because it was ambiguous last time

`python -m main.endofrun` §1 reads **`main.elo`'s fit over `eval_results.jsonl`** (the SPARSE
in-run fit, se ≈ 14.4/node). The frozen-vs-frozen `snapshot_ladder/ladder.json` is a DIFFERENT and
tighter artifact (se ≈ 10/node). On gen-12 the two disagreed materially — sparse Δ −24.8
CI [−65.4, +15.8] vs dense Δ −12.4 CI [−40.3, +15.5] — and the runbook prose ("dense offline
anchored ladder") pointed at one while `feedback_elo_reading_rules` pointed at the other. Choosing
after seeing the numbers is exactly the renegotiation pre-registration exists to prevent.

**RULE, fixed now:** the headline verdict is the **`snapshot_ladder/ladder.json` tail-4**, at
matched snapshot COUNT, at run END. The sparse `main.elo` number is reported alongside as
ORIENTATION and is never the verdict. If the two disagree in VERDICT, that disagreement is itself
reported as a finding and the dense one stands.

## 1. Non-inferiority vs gen-12 (the generation gate)

- Tail-4 mean of the dense ladder, both runs, matched snapshot count.
- **NON_INFERIOR** iff `Δ ≥ −15.0` **AND** `CI95-low > −40.0` (unchanged margins).
- **INFERIOR** iff the whole CI sits below −15. Otherwise **INCONCLUSIVE**.
- ⚠️ **An INCONCLUSIVE result is not a pass.** Gen-12 was INCONCLUSIVE and proceeded anyway on an
  owner call; that is a decision, not a verdict, and it must be recorded as one again if repeated.
- **Tie-break for INCONCLUSIVE, pre-authorized:** add games to the frozen ladder rather than
  re-slice the window. `load_games` SUMS duplicate lines by design ("independent samples of the
  SAME frozen matchup pool"), so 100 → 300 games/pair cuts per-node SE ~√3 and is pure variance
  reduction on a stationary Bernoulli. `play_pairs` currently skips measured pairs, so this needs
  a small force/extra-games flag. **Never** widen the tail-K to change a verdict.

## 2. The FIVE value routes — did they finally do anything? (THE headline of this generation)

Gen-13 is the **first run in which any of them can affect the critic** (v89 `1fa4733`; before it
`--value-from-dist` bypassed `vf_combined` entirely and gen-11/gen-12 trained them at exactly zero
gradient). So this is a genuine first measurement, not a re-read.

- **Liveness (necessary, not sufficient):** every route's zero-init projection must be off zero —
  `value_entity_pool.out_proj`, `intent_value_reduce.proj`, `value_clock_route`, `value_intent_route`,
  the v84 p_KO route. A 6k-step smoke already moved all five, so a 25M run reading zero on any of
  them means a REGRESSION IN THE WIRING, not a weak feature. `value_route_gradient_test.py` is the
  standing guard.
- **Effect:** `critic_route_audit` per-route |dV| — and note these arms are now MEANINGFUL for the
  first time. Compare against `threat` (the one route that was live all along) as the reference
  scale, not against zero.
- **Decision:** a route at ≥ half `threat`'s |dV| KEEPS. All five null ⇒ the v74/v80/v82/v84/v87
  critic-route program is a measured dead end and gen-14 deletes the lot (the honest outcome the
  two inert generations could never deliver).

## 3. `h` re-read + the `r` verdict

- `h` graduated on gen-12's §2 (|dV| 0.1618 vs median live family 0.0392 = 4.1×). Re-read it
  ALONGSIDE `r` and `--history-events`: `h` is compiled pair-history, the event seats are the same
  content in event form, so a large `h` drop when the seats are on is EVIDENCE THE SEATS CARRIED IT,
  not a regression.
- `r` (H-C reference edges) uses the same rule as `h` did: ALIVE at ≥ 0.5 × median live family.
  Zero-init ⇒ any nonzero is learned use.

## 4. H-B event seats — the gate on gen-14's frame deletion

The `event_seats` ablation arm (key-mask ALL H-B seats) + the seat usage audit.

- **Bar:** the seats must carry at least as much as the 7×159 TurnDelta frames they are meant to
  replace. Read the seat arm's |dV| + masked-KL against the frame content's own dependence.
- **KEEP + proceed:** seats at or above the bar ⇒ gen-14 deletes the 7×159 lag frames + the
  prev-turn action mask (−1124 dims), ALONE in its generation (the one non-zero-init deletion;
  `design_history_entity.md` row-2).
- **HOLD:** seats below the bar ⇒ the frames stay and H-B is re-examined before any deletion.

## 5. Mechanic usage (G2 — did the v84/v85 conditional cells move behavior?)

`python -m agents.model.mechanic_usage_baseline` on gen-13's traces vs
`measurements/gen12_mechanic_usage_baseline.json`. The cells exist to close the gap between a
mechanic's PICK rate and its own predicted probability. Gen-12's end-of-run reference:

| move | picked | mean prob |
|---|---|---|
| counter | 6.0% | 9.3% |
| pursuit | 3.9% | 7.3% |
| substitute | 1.9% | 4.7% |
| endure | 0.0% | 0.5% |
| protect | 24.5% | 21.0% |
| destinybond | 15.8% | 13.2% |

**Read the GAP, not the level** — a pick rate moving toward its own probability is the cells
working; both moving together is a policy shift, not conditional execution.

## 6. Wave-1 critic deletions — now HYGIENE, and re-derive the license here

`MultiSeedValueReadout` + `seed_diagnostics`, the `hidden_opp_belief` VF half, and the
`non_matchup_rest` VF concat sit in `vf_parts` beyond `value_pooled`, which `--value-from-dist`
still does not read. They are provably zero-gradient for the critic, so deleting them is code
removal, not a critic change — **do not report it as one**, and do not bundle it with a behavioral
arm. `--value-threat-inject` is NOT in this set (it writes into `value_pooled` and trained).

## 7. THE CRITIC-CALIBRATION GATE — did the blindness on losses move? *(added 2026-08-17, mid-run)*

*Added while gen-13 is mid-run (~16M steps): no awareness / PIT / calibration measurement of
gen-13 exists and none runs before the run completes, so the rules below are still fixed before
any number. §2 measures whether the five routes are LIVE and USED (mechanism); this section
measures the thing they were built for (outcome): the critic's established failure mode is
over-confidence specifically on losing/stall trajectories — win_prob 0.7–0.98 on decisions whose
resampled-dice win-rate is 0.0–0.4 (ledger C2 scope note + S1's probes, gen-12 @24M) — while
LEVEL calibration vs the pool is fine (C2: gap −0.011, Spearman +0.66).*

Method rules binding this section (the lessons that killed three clean findings in one session):
**no max-over-candidates statistic anywhere** (S1: a null sim reads +0.24 from selection alone);
**selection-free sampling only** (C2: the capture quota manufactured +14.5); **like-for-like**
(same script version, same command, same trace tier and opponent mix; gen-12's number read
before gen-13's).

Three metrics, directions fixed now, each run on BOTH gen-12 and gen-13 final checkpoints:

- **7a. Awareness re-read** — `python -m main.prober.query awareness <run>` (model-free):
  blind-loss fraction ↓, median lead-time ↑, cap-aware@5 ↑, loss-side coverage80 toward the
  nominal 0.80. Anchors: gen-10 baselines 7.2% blind / lead 7 / cap-aware@5 0.50 /
  coverage80 0.44.
- **7b. Stall-conditional rollout-PIT** — the `rollout_pit_probe` method restricted to the stall
  class: battles ≥50 turns, decisions at game turn ≥30, LOSSES only. Report (i) mean(win_prob −
  rollout win-rate) with battle-clustered SE, and (ii) the **confident-blind fraction** — the
  share of sampled loss-decisions with win_prob ≥ 0.7 whose rollout win-rate is ≤ 0.4. n ≥ 25
  decisions per run, ≥ 8 rollouts each, pivots evenly spaced through the eligible window (never
  selected by badness). From each checkpoint's own pinned worktree (weight drift).
- **7c. Mechanism cross-check** — §2's liveness + per-route |dV| (must be established BEFORE
  interpreting a FAIL here), plus the C1 rank re-measure (participation ratio of `value_pooled`
  on a ≥v89 checkpoint).

**Decision rule:**

- **PASS (delivery line vindicated)** = 7a blind-loss fraction falls AND 7b confident-blind
  fraction falls, each by more than its battle-bootstrap uncertainty. → Keep investing
  critic-side; §8's feature (if convicted) is an additive candidate, not a rescue.
- **FAIL (delivery line EXHAUSTED)** = both flat or worse WITH §2 confirming the routes trained.
  → The bottleneck is not delivery: the next lever is **input coverage** (§8's verdict) or the
  **training distribution** of stall games — explicitly NOT more critic routes, NOT search (S1:
  one-ply gaps statistically zero), NOT tail-weighted value loss (K1: strong-opp residuals
  sub-Gaussian; the new evidence is *conditional bias*, and no loss re-weighting adds signal the
  input lacks).
- **Mixed** = report both halves, no partial credit; this gate answers exactly one question —
  did the blindness move.

**Confound, stated in advance:** gen-13 is not a v89 A/B (fresh init + the whole enable stack).
An improvement is LINE-level evidence; only flatness-with-§2-confirmed is sharply interpretable —
with every route provably live and provably trained, nothing is left in the delivery story to fix.

## 8. The opponent-PP observability probe (runs NOW, on gen-12 — pre-registered before computing)

**The fact:** the obs encodes opponent `current_pp` as ALWAYS FULL
(`src/agents/observation/moves.py:129-130` — "Showdown doesn't track opponent PP for Gen 3");
no tracker counts opponent move uses anywhere in the tree; the 7-turn history window cannot span
a stall war. Our own PP is real (it rides the request). A Gen-3 stall war is decided by PP
accounting — recovery-move PP, Pressure, who Struggles first — so the single most predictive
quantity for exactly the game class the critic is blind on is structurally invisible to it.
Opponent move usage is PUBLIC information (every `|move|` protocol line), so a tracker is
"provide raw known facts", not a prior — no Smogon-rule issue. And the win-prob head is
MC-supervised (`win_prob_callback.py`), which rules out bootstrapped self-confirmation as the
mechanism for ITS blindness: with ground-truth labels, a persistent class-conditional miss means
off-distribution states or missing input. This probe tests the missing-input branch.

**Hypothesis:** cumulative usage/PP features carry outcome signal BEYOND everything the critic
already reads. A **mechanism check, not an A/B** — conviction licenses building the obs feature;
the feature's payoff is then measured the normal way (its own generation arm).

**Population (power-checked 2026-08-17; feature–outcome relationships unexamined):** gen-12
(`ai_v9_14_gen12_h_entitypool_shaping_0816`) eval traces, all step tiers and opponents; battles
with `meta.turns ≥ 50` — **608 battles (253 LOSS / 355 WIN)**; decision points at game turn
≥ 30. **Unit of inference = battle** (decisions within a battle share their outcome): grouped CV
and battle-level resampling everywhere; anything else is leakage.

**Features** — computed from `summary.json` invocations only (`outcome.our.action` /
`outcome.opp.action`, cumulated up to each decision; no model loading, no obs decoding — the v80
layout drifted and the summaries carry the whole signal): our/opp total recorded move uses;
our/opp RECOVERY uses (fixed list: recover, softboiled, rest, wish, moonlight, morningsun,
synthesis, milkdrink, slackoff); per-side max single-move use count (the Struggle-horizon
proxy); the our−opp differential of each.

**Test:** logistic regression, GroupKFold(5) grouped by battle, metric AUC over pooled held-out
decisions:

- **Baseline** = recorded `win_prob` (the critic's own read at that decision) + game turn — so
  PP must add beyond everything-the-critic-knows AND beyond "it's late".
- **Augmented** = baseline + the PP features.
- **Primary (conviction):** ΔAUC = augmented − baseline > 0 with the 95% battle-bootstrap CI
  excluding 0, AND a battle-level permutation null (PP feature blocks shuffled across battles,
  ≥1000 permutations) at p < 0.05.
- **Secondary (the pointed slice):** among decisions with win_prob ≥ 0.7, a PP-features-only
  logistic separates eventual losses from wins at AUC ≥ 0.65 with the battle-bootstrap CI
  excluding 0.5 — "when the critic says winning, the PP ledger knows better".

**Verdicts:** **CONVICTED** → build the opponent-PP tracker obs feature (per revealed move: use
count + estimated remaining-PP fraction; per-side recovery aggregate) as a gen-14 rider
candidate; `levers/opp_pp_observability.md` gets the GO. **NULL** → PP-observability falls for
the stall class; the next suspect is the training DISTRIBUTION of stall games, investigated
before any objective redesign.

**Caveats recorded in advance:** usage counts from decision-point outcomes UNDERCOUNT (turns
without a recorded invocation; Pressure and PP-Ups unmodeled) — noise that biases AGAINST
conviction, so a positive is conservative and a null is weakened if the measured count-coverage
is poor (report it: fraction of game turns contributing an opp action). Recovery-use counts
correlate with team archetype; grouped CV handles the within-battle part, and archetype signal is
acceptable for an OBS feature (unlike a prior) — the production feature would carry it too. The
secondary slice conditions on win_prob ≥ 0.7, which is a selection ON THE CRITIC'S OWN READ —
legitimate here because the claim being tested is precisely about that slice, and the outcome
labels are not selected.

**RESULT (2026-08-17, run AS REGISTERED — `measurements/gen12_opp_pp_probe.json`): NOT
CONVICTED.** 608 battles / 39,656 decisions, zero skips, count-coverage 0.738 (above the ~0.7
floor, so the null is meaningful). **Primary NULL**: base AUC 0.8874 → augmented 0.8848,
ΔAUC = **−0.0026**, battle-bootstrap CI95 [−0.0178, +0.0102], permutation p = 0.12 (the null's
mean is −0.008 — nine extra features cost AUC under this CV, and the real PP features merely
lose *less* than shuffled ones). **Secondary BELOW ITS BAR**: PP-only AUC on the win_prob ≥ 0.7
slice (23,326 decisions, 4,453 losses) = **0.595**, CI95 [0.512, 0.668] — a real-but-weak signal
(CI excludes 0.5) far under the registered 0.65. Since the archetype confound biases TOWARD
conviction, the null is conservative. Reading: the critic's own win_prob + turn already
separates long-game outcomes at 0.887, and the PP ledger adds nothing detectable at the margin —
the strong form of the observability story (PP is THE missing stall determinant) is dead on
gen-12 traces. Scope note, not a bar move: the population is ALL ≥50-turn games (the broad
long-game class); a maximally-narrow certified-PP-war-only test was never registered, and the
secondary's weakness makes it unpromising. **The registered verdict applies: the next suspect
for the stall blindness is the training DISTRIBUTION of stall games, investigated before any
objective redesign — and §7's FAIL branch now points there directly.**

## The gen-14 draft (edited by the verdicts above)

```
gen-13's config
  + the 7x159 TurnDelta frame deletion        # §4 KEEP only — alone in its generation
  - the value routes that nulled in §2        # deletion, per §2's all-null branch
  + unconditionalize the riders gen-13 adopted (drop-renders / believed-lean / item-belief),
    deleting their legacy branches            # only if no regression is attributable to them
  - c1/c3/c5/x from the family string (+ c2/c4 if the G3 verdict says the CELLS carry it)
```
Re-read **d3** after believed-lean: a channel carrying DISTORTED content also reads low, so the
lean fix may REVIVE it — that is a KEEP signal, not a delete signal
(`design_opponent_intent` §7a(3)).

---
*Verdicts are decision-support against these rules; this file is the registration of record.*
