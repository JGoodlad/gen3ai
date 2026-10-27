# Adversarial review: the piloting-gap forensics and H-MID (2026-10-27)

**Scope.** This review attacks two reads and the designs that lean on them:
- the piloting-gap forensics ([`../piloting_gap_forensics_2026-10-26/README.md`](../piloting_gap_forensics_2026-10-26/README.md),
  `c758fcb82`);
- H-MID ([`README.md`](README.md), [`PREREG.md`](PREREG.md), `f200258f`);
- what X62 ([`design_x62_team_recombination.md`](../../../endstate/design_x62_team_recombination.md), `fca58b24`) and S-U
  ([`design_search_u_headroom.md`](../../../endstate/design_search_u_headroom.md), `f4496fcf` / `b1c88594`) take from them.

**Method.** The checks are CPU only and read the archived rows (`recon_{kakuna,pm}.jsonl.gz`, the forensics'
`games.jsonl.gz`). They run through H-MID's registered loader and the forensics' own `features()`, and are written in
[`scripts/review_checks.py`](scripts/review_checks.py). Every number below is in [`review_results.json`](review_results.json).
The run took about 2 min at nice 19 on the agent cores, under 1 GB. Every check is post hoc and descriptive. Intervals
are 95 % percentile bootstraps over 20-game units, resampled within each panel (B = 1,000, or 500 for the
sensitivity grid). U = H, F and P72 at equal weight. "Level" means |D(10)| < 1 unless a band is named.

## Verdict

- **Both reads' measurements hold.** H-MID's numbers reproduce, and they are robust in direction to the band, the turn
  and the binning. The forensics' phase picture is corroborated by V, a richer proxy for the state than material.
- **Two INTERPRETATIONS do not hold.** Both are written into STATUS and UNDERSTANDING, and both feed X62's rationale:
  1. "The premium goes with the TEAM, not the set." The measure behind it cannot tell team from set (B1).
  2. "The critic shares P\*'s home-team view, and sees about a third of the gap." A self-play critic cannot learn an
     opponent-relative premium (M1). The third it "sees" is a turn-1 team prior, carried mostly by P72 (M2).
- **X62 stands only as gated.** Its gate, rc72b (H-CTX), is the ONE discriminating test of team against set, and
  nothing in these two reads moves its prior.
- **S-U stands as an independent question.** H-MID neither motivates it nor rules it out.

## BLOCKER

**B1. "The edge travels with the TEAM, not the set" is not evidenced. The per-mon KO measure is the game result
restated.** The forensics (§3–4) read that, restricted to sets in our pool on both sides, P\*'s KOs per entry minus
Kakuna's fall from home72 to U by +0.093 [+0.004, +0.176]. This review's own estimate is +0.100 [+0.015, +0.196].

KOs are close to an accounting identity of the result: the winner KOs all six opposing mons. An 11 pp win-rate gap
therefore moves KOs per entry by about 0.1 on ANY subset of mons, familiar or not. The check confirms this:
- **Within won games**, the seen-set drop is −0.011 [−0.092, +0.062].
- **Within lost games**, it is −0.027 [−0.117, +0.070].
- Once the result is fixed, nothing is left for the measure to localise. Fixing the result conditions on an outcome
  of the premium, so these two strata are a diagnostic, not an estimate.

The comparison that DOES hold the result fixed puts P\*'s seen-set mons against its unseen-set mons ON THE SAME U
team, minus the same contrast for Kakuna's mons in the same games:
- U overall: **−0.007 [−0.093, +0.070]**;
- per panel: H −0.020, F −0.101 [−0.205, 0.000], P72 +0.101 [−0.089, +0.288].

The two accounts predict different values here:
- **H-CTX** predicts ≈ 0.
- **The set-level account** predicts that the familiar sets keep part of their home72 edge, so a positive value, at
  most ≈ +0.08.

An interval of ±0.08 cannot separate them. The forensics' "set novelty costs about a third as much again" is inside
the same noise.

**Consequence.** Team against set is OPEN. The only discriminating read is rc72b: P\* against Kakuna on new sixes of
verbatim pool sets (`../hctx_rc72_2026-10-26/PREREG.md`, running). This does not block X62, which is gated on rc72b.
It does block quoting "goes with the team" as a finding. STATUS and UNDERSTANDING are corrected in this commit.

## MAJOR

**M1. H-MID's verdict was close to predetermined by V's training. It cannot place the memorisation in the critic.**
- V is trained on self-play and league games in which both sides pilot pool teams. A U team never appears.
- The premium is opponent-relative: P\* pilots its own sixes better than Kakuna does. Against a P\*-like opponent,
  in a cell where both sides draw from the same panel, the premium cancels by symmetry. No training signal could teach
  V a home72-versus-U difference.
- PREREG §6's reason for reading only Δbias ("Δbias holds the opponent fixed") is therefore wrong. The opponent
  mismatch is not an additive offset. It is the opponent × set interaction, which is the premium itself.
- So POLICY-CANNOT-ACT needed V to know something it had no signal for (G_V ≥ +5 pp). The registered branches were
  reachable on the synthetic stand-in, not on a self-play critic.
- What the verdict does show is real and useful: V's level does not carry the premium.
- What it does NOT show:
  - that the critic "shares the memorisation" or "shares the policy's home-team view";
  - that value-training breadth is a lever. Under self-play, more teams would shrink the policy's premium, not teach
    V a premium.
- Test T2 below discriminates.

**M2. The "third of the gap" V sees is its turn-1 team prior, and P72 carries most of it.**

At the level-10 games, V's set gap is all there before play:

| quantity | value |
|---|---|
| G_V at turn 1 | +0.039 [+0.029, +0.050] |
| G_V at turn 10 | +0.033 |
| V's own revision from turn 1 to turn 10 (set gap) | **−0.006 [−0.028, +0.018]** |
| outcome gap matched on V(1) | +0.081 [+0.015, +0.147] (matched on V(10): +0.086) |

**V's in-game reading adds nothing to explaining the set gap.**

Per panel, level at 10:

| | home72 | H | F | P72 |
|---|---|---|---|---|
| win rate W | 0.609 | 0.486 | 0.545 | 0.472 |
| mean V | 0.557 | 0.541 | 0.539 | 0.494 |
| V(1), all games | 0.592 | 0.576 | 0.565 | 0.528 |
| pool-set share | 1.00 | 0.38 | 0.70 | 0.31 |

- **G_V** (home72 − panel): H +0.016 [−0.014, +0.048], F +0.018 [−0.015, +0.050], **P72 +0.063 [+0.034, +0.097]**.
- **Δbias:** **H +0.107 [+0.024, +0.181]**, F +0.046 [−0.031, +0.124], P72 +0.074 [−0.015, +0.164].
- **V(1) does not track set familiarity.** It rates H above F, although F's sets are 70 % pool and H's are 38 %. It
  singles out the procedural teams.
- **Against PastaMittens, V "sees" all of P72's gap:** Δbias +0.007, G_V +0.081 against G_y +0.088.

**Reading.**
- V's set gap is a judgement of team quality relative to the pool, made from our own six at turn 1. In a symmetric
  cell that judgement should cancel, because the opponent's team comes from the same panel. It is not a view of the
  familiarity deficit.
- The verdict's direction survives, and is if anything stronger: on H and F, V's in-game view sees none of the gap.
- "V sees ≈ 30 %" and "a level shift V partly sees" overstate it.
- **No Simpson reversal.** Every panel points the same way on G_y and Δbias. But only H's Δbias is detected alone.

**M3. Neither read discriminates H-CTX from the set-level account, so neither supports X62 beyond its gate.**
- The cross-panel gradient (X62 §1.3: home72 0.598, F 0.530, H 0.497, P72 0.433, falling with the pool-set share)
  fits both accounts.
- So does the within-U regression: +3.0 pp per seen set of ours and −1.9 per seen set of theirs. In the symmetric
  design both sides' counts move together, so the set-count model predicts ≈ 3.2 × 1.0 ≈ +3 pp of the 11 pp gap.
- That leaves about 8 pp to team context, to non-additivity (a fully familiar six), or to panel traits other than
  sets. P72's procedural teams are the obvious case (M2).
- B1 removes the per-mon evidence for "team".
- X62's §1.1 bullets 1 and 2 are therefore context, not support, for H-CTX. Its own guess (+2.0 pp) should not lean
  on them.
- X62's G2 line "the critic shares the memorisation" falls with M1.
- **X62 stands as gated on rc72b.** That gate was always the right one.

**M4. S-U: H-MID does not motivate it, and it does not count against it.**
- S-U's leaf is P\* playing both sides for 4 turns, then V. That is the self-play regime, where the premium cancels
  (M1). The search therefore evaluates actions as if the opponent were P\*.
- Any U headroom must come from local errors visible under that self-model, not from the premium.
- V's set offset is a turn-1 constant (M2), so it does cancel inside a node, as S-U argues. But equal game-level AUC
  says nothing about ranking actions within a node (S-U already concedes this in §1.1).
- **S-U stands as an independent, expensive question** (≈ 600–1,000 CPU-h after `b1c88594`). "H-MID motivates it"
  (`search_ledger.md` Q17) should read "H-MID does not rule it out".

## MINOR

- **m1. Sensitivity (the brief's binning and turn question).** Over turns 5–20 × bands ±0.4 / ±1 / ±2 / all states,
  and over decile, pooled-quintile and three-bin V bins:
  - Δbias runs +0.050 to +0.142 and the matched-V gap +0.067 to +0.146.
  - A linear y ~ V + U fit gives U −0.080 [−0.148, −0.019].
  - The direction is robust. The registered CALL is fragile at the narrow band: ±0.4 at turn 10 gives
    +0.050 [−0.030, +0.135], which would read INCONCLUSIVE.
  - Δbias needs no "level" filter: all turn-10 states give +0.080 [+0.032, +0.132], and turn 1 gives +0.077. H-MID is
    a whole-game statement, not a mid-game one.
- **m2. Selection at turn 10: none.** All 1,600 games reach turn 10. 74.5 % of home72 and 77.5 % of U games are level.
  By turn 20, 7 % of games have ended.
- **m3. "Not the opening" is corroborated but not sharp.**
  - The gap in E[V(t)] (V = the result once a game is over) is +0.035 at turn 1, +0.032 at turn 10, +0.058 at 20,
    +0.086 at 30, +0.107 at 50 and +0.112 at the end. V, a richer state proxy than material, sees nothing open by
    turn 10.
  - But a steady per-turn deficit, spread over a ≈ 43-turn game, predicts only ≈ +1.8 pp by turn 10. That sits at the
    upper bound of V's revision interval (+1.8). "Lost after turn 10" is not separated from "lost at a steady rate
    that shows later".
  - The finding that holds: nothing measurable is lost in the opening.
- **m4. How much weight the PastaMittens replication carries.** It shares teams, seeds and our deterministic policy;
  V(1) is identical to the 4th decimal. It replicates the opponent only.
  - Pooled it is INCONCLUSIVE (Δbias +0.033). Per panel, V tracks P72's gap fully against it.
  - H-MID deserves the weight of ONE read on ONE team sample. Its direction is robust against Kakuna, and its size
    depends on the opponent.
- **m5. The forensics' many measures (≈ 20 DiDs, uncorrected).** FINDING 6 there already treats the two targeting
  measures as leads. The null "decision mix moves ≤ ±1 pp" is a fair absence claim at that width.
- **m6. Forum72's packing defect** (`b0441623e`). Kakuna's sft431 Gyarados hit Frustration at base power 1 in every
  banked forum72 Kakuna read. This is one team of 72, on Kakuna's side, so it slightly flatters P\*'s F score, and
  with it U, in both reads. Not re-measured.
- **m7. The mon tier ignores item and spread.** A forensics "seen set" on a U team may differ in item or EVs, which
  biases its read toward H-CTX. rc72b uses verbatim blocks.

## Discriminating tests (CPU-cheap)

| # | test | cost | separates |
|---|---|---|---|
| T1 | **rc72b (running).** Add one mechanism read to it: the within-team DiD is moot there (every set is seen), but V's G_V on rc72b against home72 says whether V's turn-1 prior tracks pool-TEAM membership or team quality | already paid; V read ≈ 6 min CPU with H-MID's `reconstruct.py` | team against set (B1, M3); the nature of V's prior (M2) |
| T2 | **V in a self-like cell.** Run H-MID's reconstruction on a symmetric P\*-vs-P\*-family cell on home72 and U: X37b's in-house games if captures exist, else 400 games a set (≈ 1 lane-h) | ≈ 1–2 lane-h + 15 CPU-min | M1. If V's Δbias there ≈ the in-house premium (≈ 4 pp, X37b), V is a faithful self-play critic and H-MID's miss is the opponent interaction, not memorisation in the critic |
| T3 | **Two-sided V coherence.** Re-encode KAKUNA's side of the same 1,600 captures with P\*'s encoder (both packed teams are in START) and read V. A coherent critic gives V_ours + V_theirs ≈ 1 on every panel | ≈ 10 CPU-min after a side switch in `reconstruct.py` | M2. If the sum drops on P72 / U, V's set offset is an out-of-pool artefact of V, not information |
| T4 | **V(1) per team against team strength.** Correlate each team's mean V(1) with its Kakuna-vs-Kakuna score (`kcal`-style rows, or the X37 TEAM STRENGTH read `82858f433`) | minutes, banked rows | M2: a team-quality prior predicts a positive correlation within each panel |

## Docs changed by this review

- STATUS row 3 and the forensics / H-MID result rows: "goes with the TEAM" → open, pending rc72b; "V sees ≈ 30 %" →
  a turn-1 prior that sees none of the in-game gap.
- UNDERSTANDING item 4 and §"Where it is lost": the same two corrections, citing this review.
- UNDERSTANDING §6's search-on-U row: "not built" → S-U registered (`ffd9a443a`), stage 0 HELD.
- A review pointer at the top of both READMEs.

Not edited (outside this unit's brief; reported as findings):
- X62 §1.1 bullet 1 and G2 ("the critic shares the memorisation");
- `design_beat_metamon_path.md` l. 228 ("per mon it goes with the TEAM context");
- `search_ledger.md` Q17 ("H-MID motivates it").
