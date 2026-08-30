# Search as a TEACHER — per-decision gains are real; they do not compose in play

**Status:** 🟡 **ACTIVE — built, dormant** · **Ledger:** `5f98d26` (probe G) · `79e8b11` (probe H) ·
`d2a0212` (probe I) · `4cf81fd` (defensive search iter 1) · `35dbc3c` (iter 2) · `2af60c2` (probe K)
· `deb0bc9` (the transfer cell) · `1984dc7` (the three-route taxonomy) · `f5c8a77` (ai_v12 build)

One-line claim: *a CRN-paired, win-prob-leaf, triage-gated search makes measurably better decisions
than the policy (+4.7pp per overrule, opponent-marginalized) and wins no more games (τ = 0.17,
excludes 1.0) — so its value is the TARGETS it manufactures, not the moves it plays.*

**This lever is compatible with the owner's hard constraint by construction**: search stays an
offline teacher / diagnostic, and what ships into the model is a distilled target or an amortized
readout, never runtime deliberation.

## Known (cleared the honesty gates)

- **Pair first; the offset is free money (`5f98d26`).** Critic error decomposes into a **per-DECISION
  shared offset = 0.728 of true MSE [0.674, 0.780]** (RMS 0.200) and a **DIFFERENTIAL = 0.272**
  (RMS 0.122). The offset is per-decision, not global (global 0.26%), so it **cancels between
  siblings at the same decision and NOT across depth** ⇒ shallow paired search is the favored
  regime and depth ≥2 re-admits the offset. Contrastive critic training is therefore **SIZED at
  ≤5.7pp of per-decision regret — a later lever, not the constraint**; the critic already captures
  71% of achievable ranking gain.
- **The leaf must be the WIN-PROB head, not V (`5f98d26`).** Ranking by the one-ply win-prob head
  beats the action the policy actually played by **+0.0219 [+0.0089, +0.0364]**; the scalar value
  head does **not** clear zero (+0.0135 [−0.0007, +0.0280]).
- **"Forced decisions" is REFUTED (`79e8b11`).** Search flips **0.694 [0.681, 0.707]** of decisions
  (corroborated at 60–67% over ~40k real battery searches), and **no cheap policy-confidence feature
  separates the flips** — gap/entropy/top-1 sit at or below the random null, and drop-one CV says
  removing the logit gap *improves* the triage. **What IS separable is flip COST: 83% of the
  dividend sits in 22.7% of decisions worth ≥5pp, found only by |P(win) − 0.5|.** The line of
  record: *the policy does not know when search will overrule it; the critic knows when being
  overruled would not matter.*
- **Racing works, and it audited the instrument (`d2a0212`).** Successive elimination with
  CRN-paired difference CIs buys 1.47× on the deadline axis, 1.87–2.40× on spend. Separation is
  **U-SHAPED with an empty middle** — 52.2% never separate within 32 samples; the rest separate at
  the floor — so non-separation is itself a mid-search triage signal (a futility stop). 🚨 **And the
  retroactive caveat: the battery's own 1 s cell agrees with its own large-budget argmax on only
  86.1% of decisions — ~1 in 7 historical "searched" decisions was allocator noise.**
- **Defensive paired search got search to STOP LOSING (`4cf81fd`).** Mirror **0.4937 [0.4448,
  0.5427]** (paired 0.4938 ± 0.035) vs honest_1s's 0.2929 — Δ **+0.2008 [+0.1229, +0.2738]** — and
  it beats playoff_10s (0.450) at **one twentieth the budget**, on 120 literally-identical battles.
  Stretch (CI above 0.50) honestly not met: a hair under at resolving width, *a result, not an n
  problem*.
- **🔴 The dividend is ZERO, and the mechanism is exonerated (`35dbc3c`).** Iteration 2 moved every
  counter exactly to spec — separated-of-raced 0.157 → **0.454** (95% of I's ceiling), overrules
  1.8% → **5.82%** (13×), rounds/race 4.61 → 13.17, envelope verified — and the win rate landed
  **0.5003 [0.4803, 0.5203]: the point estimate IS the null.** 13× more evidence-certified
  overrules bought nothing.
- **The WINNER'S CURSE of a biased instrument (`35dbc3c`), the durable mechanism lesson.** CRN
  pairing removes dice noise *and* the shared offset, so what racing CERTIFIES is the leaf's
  residual **differential** bias (RMS 0.122 — larger than most true gaps) as much as signal.
  *Statistical separation of a biased reader is not correctness.*
- **The leaf is PARTIALLY EXONERATED (`2af60c2`).** Re-judged under opponent-MARGINALIZED paired
  rollouts, iteration 2's own 3,531 overrules are worth **+0.0474 [+0.0216, +0.0730] per decision —
  REAL** — and probe G's +2.2pp was **not** a frozen-opponent artifact (paired diff +0.0062 n.s.).
  The game-level zero stands as fact; its attribution to the leaf does not.
- **COMPOUNDING is convicted (`deb0bc9`).** 8,100 games / 4,050 paired units / 200k decisions:
  **A − B = +0.0020 [−0.0039, +0.0079]** against a naive expectation of +1.16pp ⇒ **τ = 0.17
  [−0.34, +0.68], EXCLUDES full transfer.** Checkpoint and the bot half of population are removed
  with the dividend still absent. Signature (suggestive only, post-treatment conditioning): the
  overrule-count gradient runs **+3.9 → −2.1 → −8.3pp** for 1 / 2 / 3+ overrules.
- **Good behavior worth keeping:** the triage gate **auto-scales dose to headroom** — against
  saturated bots it forced 92.6% and overruled 0.245/game, 9× below the mirror. A safe search backs
  off when it is already winning.
- **All three routes are BUILT and OFF (`f5c8a77`).** Route 1 `--win-prob-pbrs-coef`; routes 2+3
  `--search-teacher-mode winprob_oneply` with confirmation through the existing
  `--teacher-confirm-rollouts`; `defensive.gate`/`DefensiveConfig` is IMPORTED by the teacher so the
  searcher's "contested" and the teacher's cannot drift.

## Not-known

- **Does rollout CONFIRMATION rescue the play dividend?** Iteration 3 (`--defensive-confirm`,
  top-2 paired rollouts before any overrule — the playoff mechanism, the only historically
  non-losing arm) was dispatched; registered: overrules fall to ~1.5–3.5%, win ≥ 0.50 no-regression.
  **If confirmed overrules also net zero, search-as-PLAYER is closed at these checkpoints.**
- Whether compounding is *selection* (searched states are systematically the ones where a
  substitution changes the continuation distribution adversely) or *interaction* (the one-substitution
  Q^π assumes the POLICY plays on; live, the SEARCHER plays on). The transfer cell isolates the
  coefficient, not the cause.
- Whether route 2's transplanted targets carry the same differential bias into the WEIGHTS, where no
  confirmation step can catch it later. This is the reason route 2 inherits the search programme's
  discipline as a requirement rather than a recommendation.
- The harder half of population: eval sentinels were not constructible as battery opponents.

## Pros

- Every component carries its own measurement — gate (H), evaluate (G × I), futility (I), confirm
  (playoff) — so the composition is assembled from evidence rather than intuition.
- The teacher route sidesteps the failure that killed the player route: compounding destroys gains
  *in play*, but a distilled target is graded per decision and never has to survive its own
  continuation.
- Route 3 → route 2 is the AlphaZero loop in miniature: **search manufactures the curriculum**, and
  confirmed overrules are by construction the highest-quality targets available.
- E5's Q-head makes the same knowledge FREE at inference (one forward, eleven win probs) — and its
  **amortization residual is the value of one-ply search as a number**, so the programme carries its
  own retirement criterion.

## Cons

- **Route 2 has NO shield.** PBRS is protected by the telescoping-invariance theorem (a
  miscalibrated potential costs speed, not correctness); a distillation target is simply believed.
  It imports the head's differential bias directly — the iteration-2 curse, moved from inference to
  the weights.
- The whole programme rests on a leaf that is *better than the policy* and *not correct* — G's own
  numbers say the critic's excess over a 32-rollout MC oracle does not clear zero.
- Every historical battery cell it cites carries the 14% allocator-noise caveat.
- Confirmation is expensive (rollouts), which is exactly the cost the amortized Q head exists to
  remove — so the cheap version depends on a head that has not been trained yet.

## Next test

1. **Iteration 3's verdict** (running) — the door for the whole play branch.
2. **Close wave C's named gap: the `q_labels` producer.** The Q head exists dormant at v107 and no
   training pilot is possible until the counterfactual producer emits per-action labels (schema v1,
   additive-optional, **list-of-objects never parallel arrays** — three same-length lists can be
   written in the wrong order and read as valid).
3. **Then route 2's first arm**, with the confirmed-overrule filter as the target-quality gate, and
   `--teacher-confirm-rollouts 0` reserved *only* as the control arm that measures what the
   confirmation is worth.
