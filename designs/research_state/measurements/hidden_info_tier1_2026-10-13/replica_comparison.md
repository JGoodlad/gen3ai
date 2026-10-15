# The hidden-info study, Tier 1 against the BLIND REPLICA: the third-party comparison (2026-10-15)

**Status: a THIRD-PARTY comparison (`design_hidden_info_binding.md` §12.2 R-REPLICATE, §13.1).** The comparing agent ran
neither implementation and changed neither's verdicts. The numbers come from `replica_comparison.json`, written by
`replica_compare.py` (this folder). That script reads the original's frozen blinded table (sha `fc3aced3…`, checked on
load), its arm key, its world files and the pilot's Phase 1 rows and bank, and reads the replica's `imp_per_state.json`,
`rows/*.jsonl.gz` and `marginals.json` (`../hidden_info_replica_2026-10-11/`). It is not blind: both READMEs' headline
numbers had been read before the criteria below were written.

## In plain language

Two agents built the "fair world" instrument separately, and neither saw the other's code. The question is whether they
measure the same thing. **On the 978 states where both built every arm, they agree on almost everything they both
measured.**
- **The value of hidden information agrees:** G is +0.070 for the original and +0.072 for the replica.
- **The moves' small sufficiency share agrees:** s_M is 0.11 vs 0.12.
- **The moves' necessity share is close:** n_M is 0.39 vs 0.47. The replica's higher figure is in the direction its
  learnset handicap predicts.
- **The tie-breaking artifact reappears in the replica's own worlds:** the honest 8-world leaf ties in 60 % of states,
  and the dose curve takes the same step at p = 0.001.

**The headline verdict, "the never-seen Pokémon dominate the moves" (rule R3), is NOT REPLICATED.** The replica never
built the never-seen-class arms. The same is true of the stage, recovery, evaluator and dose/placebo rules. The two
implementations also differ on one thing they both measured: how the gap falls with the number of opponent Pokémon seen.
The original's slope is clear and the replica's is half as steep and not clear.

## 1. What each implementation ran: compare only like with like

| dimension | ORIGINAL (Tier 1, `67bee8d6` / `f01b9e51` / `c6f71d9a` / `6a7ee72c`) | BLIND REPLICA (`34ee1d6c`) | consequence |
|---|---|---|---|
| checkpoints / pins | `rb_es_end2_s2001` @ `62085750`, `rb_es_prod_s2001` @ `95d014fa`; worlds built at HEAD, played at PAD pins (pin + the gender pad's engine code) | the same two checkpoints, built and played at the unmodified pins | same networks; the engine differs only by the gender pad (identity-checked by both) |
| states | registered 600 per checkpoint + the k-stratified supplement (+200 / +220) | registered 600 per checkpoint | compared on the registered states only |
| arms | O, F, F+M, O−M, F+U, O−U, FN, D50, D90, W90, MAP100, POOLSETS, POOL; P = the pilot's v2 base (`det2`); HE@O / P / F | O, P (its own re-build of the pilot's base), F, F+M, O−M | only O, P, F, F+M, O−M can be compared; everything else is NOT REPLICATED |
| worlds | 8 per state; the information-lever builder; never-seen genders via the engine's draw PAD | 8 per state, ≤ 60 tries; explicit genders + the gen-5 LCG advanced past the truth's gender draws (its A1) | both fix the gender-dice leak, by different means; both keep the dice-silent leak (A2: inherent to reusing the true seed) |
| move prior | Smogon usage, NO learnset filter (`world_draws.Prior`; `team_legality` checks no learnsets) | Smogon usage filtered by `learnset.is_legal` at the pin, i.e. the table before `64a695dc` (1,500 legal pairs missing) | **the replica's handicap (§5)** |
| F+M | the seen mons' true moves; Hidden Power's TYPE drawn with the spread (class S) | the true moves INCLUDING Hidden Power's true type (its A9) | in the original's F+M worlds, 16.5 % of seen-mon sets carry a non-true HP type (100 % equal up to the type); the replica's F+M is a slightly stronger reveal |
| opponent's root-turn choice (≈ 1 % of states) | kept by slot in worlds (`root_choice_in_log=False`) | dropped in worlds (its A3) | 8 states; the original's sensitivity read changes nothing |
| scorers | LEAF-K1/K4/K8, TRUNC-d-K, CRITIC, the hand evaluator | LEAF-K1/K4/K8 only | the critic, TRUNC and HE comparisons are NOT REPLICATED |
| contrast pairing | each contrast on the states its sources hold | the "common set": all five arms ok | **compared here on C = the states where BOTH implementations hold all five arms** (978 = 491 end2 + 487 prod, 711 battles) |
| bootstrap | battle-clustered, 2,000 draws, seed 20261011, pooled within checkpoint | the same specification | here ONE bootstrap (`t1_stats.Boot`) for both, so every cross-implementation difference is paired on the same draws |

## 2. The agreement criteria (stated before this script first ran)

- **Brief criterion:** a quantity AGREES iff the two 95 % intervals overlap AND every R-rule condition computed from it
  reads the same (rule 8 applied). It DISAGREES otherwise. It is NOT REPLICATED when the replica has no such quantity.
- **Registered §12.2:** on the same states, the point estimates of every registered IMP-scale contrast differ by < 0.005
  (rule 8: within ±0.0005 of 0.005 is BOUNDARY). The per-class marginals' TV distance is < 0.02. Every verdict is the same.
- **Supplementary (descriptive):** the PAIRED difference (original − replica, same states, same draws) has a 95 %
  interval containing 0.

## 3. The instrument checks (no outcome involved): PASS

- **The truth and the referee are identical.** IMP_O is EQUAL state by state on all 1,200 registered states at LEAF-K1,
  K4 and K8 (max |Δ| 0). The replica's O rows equal the Phase 1 outcome strings for seeds 0–7 on 1,200 / 1,200 states.
- **One scorer reproduces both implementations exactly.** Every IMP of both, re-derived from the per-world outcomes by the
  original's `imp_of`, matches what the implementation reported: 0 differences over 3,398 (original: F, F+M, O−M) and 4,650
  (replica: P, F, F+M, O−M) state-arm pairs. The two scorers (tie toward the policy's order, 1e-6) are therefore the same function.
- **The two implementations see the same thing.** For every state where both built a world, the searched side's seen
  opponent mons and their revealed moves are identical: 1,164 / 1,164 states. Only moves were compared; Hidden Power was
  normalised to its base.

## 4. The agreement table (pooled end2 + prod, LEAF-K8, on C: 978 states / 711 battles)

| quantity | ORIGINAL | REPLICA | paired orig − rep | brief criterion | §12.2 point |
|---|---|---|---|---|---|
| IMP_O | +0.0752 [+0.0645, +0.0867] | +0.0752 [+0.0645, +0.0867] | 0 (identical) | AGREE | AGREE |
| IMP_P | +0.0066 [−0.0035, +0.0179] | +0.0097 [−0.0012, +0.0203] | −0.0031 [−0.0138, +0.0081] | AGREE | AGREE |
| IMP_F | +0.0056 [−0.0048, +0.0163] | +0.0037 [−0.0067, +0.0138] | +0.0019 [−0.0093, +0.0128] | AGREE | AGREE |
| IMP_F+M | +0.0135 [+0.0034, +0.0242] | +0.0123 [+0.0018, +0.0238] | +0.0012 [−0.0102, +0.0132] | AGREE | AGREE |
| IMP_O−M | +0.0479 [+0.0372, +0.0587] | +0.0412 [+0.0301, +0.0520] | +0.0067 [−0.0025, +0.0160] | AGREE | **DISAGREE** (0.0067) |
| **G = O − F** | **+0.0696** [+0.0591, +0.0806] | **+0.0715** [+0.0602, +0.0834] | −0.0019 [−0.0128, +0.0093] | **AGREE** (B2 both) | AGREE |
| G_P = O − P | +0.0686 [+0.0581, +0.0789] | +0.0655 [+0.0549, +0.0765] | +0.0031 [−0.0081, +0.0138] | AGREE | AGREE |
| P − F (items / abilities / spreads told) | +0.0010 [−0.0093, +0.0112] | +0.0060 [−0.0059, +0.0170] | −0.0050 [−0.0207, +0.0110] | AGREE | BOUNDARY (0.0050) |
| σ_M = F+M − F | +0.0079 [−0.0017, +0.0179] | +0.0086 [−0.0029, +0.0203] | −0.0007 [−0.0155, +0.0144] | AGREE (B8's σ half fails on both checkpoints, both) | AGREE |
| ν_M = O − O−M | +0.0273 [+0.0189, +0.0358] | +0.0340 [+0.0256, +0.0432] | −0.0067 [−0.0160, +0.0025] | AGREE (B8's ν half holds on both checkpoints, both) | **DISAGREE** (0.0067) |
| **s_M** | **+0.114** [−0.025, +0.240] | **+0.120** [−0.044, +0.261] | −0.007 [−0.206, +0.199] | **DISAGREE** on one condition: s upper < 0.25 holds for the original (0.240), fails for the replica (0.261); B3 fails on both, s lower > 0 fails on both | (a share, not an IMP contrast) |
| **n_M** | **+0.392** [+0.280, +0.510] | **+0.475** [+0.366, +0.601] | −0.083 [−0.234, +0.065] | **AGREE** (B4 fails on both; lower ≥ 0.25 holds on both; upper < 0.25 fails on both) | (a share) |

Per checkpoint on C, all as orig / rep:

| checkpoint | G | s_M | n_M |
|---|---|---|---|
| end2 | +0.059 / +0.063 | 0.12 [−0.14, 0.33] / 0.19 [−0.07, 0.40] | 0.41 [0.21, 0.62] / 0.52 [0.34, 0.72] |
| prod | +0.081 / +0.080 | 0.11 [−0.07, 0.27] / 0.07 [−0.16, 0.26] | 0.38 [0.25, 0.52] / 0.44 [0.30, 0.59] |

The other leaf scorers on C (reported, not registered):
- **LEAF-K4:** G +0.057 / +0.067; s_M 0.01 [−0.18, 0.18] / **0.23 [0.08, 0.36]**; n_M 0.50 / 0.54.
- **LEAF-K1:** G +0.029 / +0.031; s_M −0.18 / −0.10; n_M 0.55 / 0.59.

The LEAF-K4 sufficiency shares do not overlap well, and the replica is the higher one: it is the stronger F+M reveal and
carries the handicap (§5).

**The M-only cascade of §12.3** uses R1, then NECESSARY-NOT-SUFFICIENT, then SHARED; R2, R3 and B5–B7 need arms the replica
did not run.
- **On C:** the original reads MOVES-NECESSARY-NOT-SUFFICIENT and the replica reads MOVES-SHARED.
- **On their own published sets:** the reverse. The original's pooled s_M upper is 0.252, so it reads SHARED; the replica's
  is 0.227, so it reads NECESSARY-NOT-SUFFICIENT.
- **Why it flips:** in all four reads, s_M's upper bound sits within 0.025 of the 0.25 line. That is outside rule 8's
  ±0.0005 band but far inside the sampling noise.
- **Consequence:** the cascade's choice between NECESSARY-NOT-SUFFICIENT and SHARED is undetermined at this power. The
  registered verdict is not affected, because R3 is checked first (§12.3).

**Why the published B4 reads differ (the replica holds B4 at n_M 0.533; the original fails it at 0.416).** The cause is
SELECTION, not disagreement.
- The replica built all five arms on 111 states where the original's honest builder could not fit F, F+M or P.
- On those 111 states, the replica's n_M is **1.17 [0.76, 1.99]** (ν_M +0.067 on G +0.057). The original's own ν_M there is
  +0.040 [+0.020, +0.063].
- Adding these states raises the replica's n_M from 0.475 (on C) to 0.533.
- **Implication:** the states an honest builder cannot fit are ones where hiding the moves costs more than average. So every
  common-set share is conditional on fittable states, and n_M is probably understated in BOTH implementations. The
  exclusions are not missing at random (FINDING 3).

## 5. The learnset handicap: scope, direction and size

**Scope** (from the true opponent teams of the 978 states on C, against the 1,500 pairs the pre-`64a695dc` table lacked).
- **How often a missing pair matters.** The true team holds at least one missing pair in **910 / 978** states (93 %). A
  seen mon's HIDDEN true move is a missing pair in **616 / 978** (63 %). A never-seen mon's true set holds one in **530 / 978**
  (54 %). One of the two holds in 877 / 978 (90 %).
- **The top pairs** (states; Smogon inclusion): Swampert Hydro Pump (420; 0.53), Tyranitar Dragon Dance (334; 0.42),
  Swampert Curse (261; 0.07), Tyranitar Pursuit (172; 0.23), Gengar Will-O-Wisp (171; 0.51), Jolteon Baton Pass
  (153; 0.79), Charizard Beat Up (129; 0.34).
- **What the honest F worlds hold.** On the seen mons' hidden true moves:
  - The original holds a missing-pair move in **0.327** of its world-slots; the replica in **0.000** (7,936 slots, 10.5 % of
    all hidden-move slots).
  - On the other hidden moves, the original holds 0.532 and the replica 0.543. The replica moves the prior mass it could not
    give to the missing pairs onto the other moves.
  - Overall, the replica's worlds hold **0.486** of the true hidden moves against the original's **0.510**: 2.4 points fewer.
- **Marginals** (seen-mon moves, M.move). The two implementations' world move distributions differ by a TV of 0.140 in F and
  0.155 in O−M. With the missing pairs removed, that drops to **0.103 / 0.111**. The full-sample noise floor is ≈ 0.08–0.09
  (§6). So the handicap explains most of the move-marginal difference, and what remains is near noise.

**Direction.** Each arm that re-draws moves from the filtered prior puts its worlds further from the truth.
- IMP_F goes down, so G goes up.
- F+M − F goes up, so s_M goes up.
- IMP_O−M goes down, so ν_M and n_M go up.
- IMP_P goes down.

**Every bias points TOWARD the moves.** The replica's agreement that the moves are a minority share is therefore
conservative with respect to the handicap.

**Size.**
- **Mechanistic estimate.** The 2.4-point hit-rate loss is ≈ 5 % of the move information between F and F+M (0.49 of hit
  rate). On Amendment 3's LINEAR M curve that is worth ≈ +0.0004 on σ_M and ≈ +0.0013 on ν_M, so n_M ≈ +0.02.
- **Empirical bound.** The measured ν_M difference is +0.0067 [−0.0025, +0.0160] (replica higher), so n_M is at most
  ≈ +0.10 at the point and ≈ +0.23 at the upper bound.
- **The missing pairs are iconic moves** (Dragon Dance, Baton Pass, Hydro Pump). Each may matter more than an average move,
  so the mechanistic figure may understate the effect. The empirical bound caps it.
- **The affected-vs-unaffected split does not locate it.**
  - IMP_O−M on the 616 affected vs 362 unaffected states gives a difference-in-differences of −0.012 [−0.030, +0.007]. That
    is the OPPOSITE sign to the mechanism and its interval contains 0.
  - The F / F+M / P difference-in-differences use only 101 unaffected states. Two of them exclude 0 with opposite signs
    (F+M −0.040, P +0.045), and one would expect that from noise and team-archetype confounding across four tests.
  - These are uninformative; the doc does not lean on them.
- **By reveal count**, the original − replica G difference trends with k: slope −0.012 [−0.022, −0.002] per SD of k. At
  k = 6 it is −0.020 [−0.042, +0.001]. At k = 6 no Pokémon is unseen, so the seen mons' hidden facts are all F re-draws,
  and that is where the handicap acts. This is consistent with the handicap but not specific to it (the replica's item,
  ability and spread draws also differ).

## 6. §12.2 R-REPLICATE as registered: NOT MET (and partly unmeetable)

- **TV < 0.02 per class: fails on most classes. But the threshold sits BELOW the sampling noise floor.**
  - Split-half TV inside ONE implementation, by battle, on C (the full-sample floor is ≈ split-half / √2):
    - M.move 0.11–0.13;
    - U.move 0.07–0.08;
    - U.species 0.04–0.05;
    - I.item 0.18;
    - A.ability 0.09.
  - Two halves of the SAME implementation fail the registered test. The rule cannot pass at ~10⁴–10⁵ draws spread over
    hundreds of categories (FINDING 2).
  - Measured against the noise:
    - **Exceed the floor:** M.move (F, O−M: the handicap, §5) and U.move (0.111 in F; 0.082 with the missing pairs removed,
      floor ≈ 0.05: the never-seen move sets differ beyond the handicap).
    - **At or below the floor:** U.species 0.036, U.item 0.013, U.ability 0.027, I.item 0.039, A.ability 0.002.
    - **Sanity tooth:** every class O−M keeps TRUE (I, A, S, U) reads TV 0 exactly.
- **Point estimates < 0.005:** met by G, G_P, σ_M, IMP_O / P / F / F+M. **Fails** on ν_M and IMP_O−M (0.0067, in the
  handicap's direction). BOUNDARY on P − F (0.0050).
- **Every verdict the same:** the replica issues no verdict. The M-only cascade differs on C (§4), and every other rule is
  NOT REPLICATED.

## 7. The four points the brief asked about

**(a) The per-class split: does the replica also put the never-seen species first? NOT REPLICATED.**
- The replica ran no F+U or O−U, so neither s_U nor n_U, nor R3's paired U − M contrasts, exists for it. No combination of
  its arms identifies U.
- What it does replicate is the M side of R3:
  - s_M ≈ 0.11–0.12, well below the 0.5 that B3 needs;
  - n_M 0.39 / 0.47, below B4's 0.5 on C;
  - R1 fails in both. The moves are neither refuted outright nor a majority.
- For R3 to hold on the replica's numbers, n_U would have to exceed n_M by a clear margin. The original's n_U is 0.82
  [0.70, 0.96] against a replica n_M of 0.47, which is suggestive, but a cross-implementation mix is not a replication.

**(b) The honest gap's size: REPLICATED.**
- G is +0.0696 vs +0.0715, paired difference −0.0019 [−0.0128, +0.0093], both B2. Per checkpoint: end2 +0.059 / +0.063,
  prod +0.081 / +0.080.
- Telling the seen mons' items, abilities and spreads (P − F) adds +0.001 / +0.006 (BOUNDARY on §12.2's point rule; both
  small). This agrees with the original's finding that they add ≈ 0.003.

**(c) The dose curve and the tie-break step: the ARTIFACT REPLICATES on the replica's independent worlds** (post hoc for
both; the replica registered no dose arm). M8 is the TRUE world at weight p and the implementation's own F worlds at
1 − p, on C.

| | tied honest LEAF-K8 argmax | p = 0 | 0.001 | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| ORIGINAL | 0.608 [0.579, 0.639] | +0.006 | +0.024 | +0.025 | +0.035 | +0.055 | +0.065 | +0.074 | +0.075 |
| REPLICA | 0.603 [0.571, 0.634] | +0.004 | +0.028 | +0.029 | +0.040 | +0.058 | +0.066 | +0.076 | +0.075 |

- **The step:** the jump from p = 0 to 0.001 is +0.019 [+0.013, +0.025] in the original and +0.024 [+0.017, +0.031] in the
  replica.
- **The flat stretch:** from 0.001 to 0.1 the curve rises +0.0006 / +0.0017.
- **The real rise:** from 0.1 to 1 it rises +0.050 / +0.046.
- **Agreement:** every paired difference contains 0.
- **Same scorer:** the replica's scorer has the same tie rule (an argmax within 1e-6, toward the policy's order;
  `replica/score.py`), and §3 shows it is the same function.

So the registered P-STAR = 0.1, TODAY = 0.009 and WITHIN-REACH sit on a step BOTH builders produce. The step is a property
of the scorer (a mean of 8 binary outcomes), not of either builder. The registered dose and placebo arms (D50, D90, W90,
MAP100) and B6 / B7 are NOT REPLICATED.

**(d) The evaluator comparison: NOT REPLICATED.** The replica ran LEAF scorers only (its A13: §13.1 names only LEAF-K8).
The critic search, TRUNC-4 and Foul Play's hand evaluator (R-EVAL) have no second implementation.

**Also: the stage gradient on k (a SUB-read; the registered R-STAGE-VOI uses all Tier-1S states, which the replica does
not hold). It DIFFERS.**
- On C, the original's slope of G on z-scored k is **−0.023 [−0.033, −0.013]** (TREND-DOWN). The replica's is **−0.011
  [−0.022, +0.0003]**, which is not clear by rule 8.
- G by k (1–2 / 3 / 4 / 5 / 6), orig / rep: +0.087 / +0.073; +0.089 / +0.088; +0.087 / +0.099; +0.047 / +0.054;
  +0.023 / +0.043.
- The two implementations disagree on where along the game the gap sits. The replica has a larger G at k = 6 (the handicap's
  locus) and a smaller one at k = 1–2 (its never-seen draw: species ∝ the MEAN teammate prior over the revealed species).

## 8. Rule 1's replication condition, verdict by verdict

The condition is B1's "R-REPLICATE agree" and §13.2's "a conclusion may steer research only when R-REPLICATE agrees".

| verdict (original, §16.8) | replication status | why |
|---|---|---|
| R-GIGO for the Tier-1 worlds PASS | **PARTLY MET** (instrument level) | the truth, the referee and the scorer are identical; the views agree on 1,164 / 1,164 states; the replica's own audits are its own (identity 1,199 / 1,200; tooth T4 missed on 29 states) |
| G = +0.070 (B2) | **MET** | §7 (b) |
| R-MOVES = MOVES-REFUTED (R3) | **PARTLY MET** | the M half (s_M, n_M; B3, B4, R1 fail in both; B8's halves alike) replicates by the brief's criterion, except s_M's upper-0.25 condition (the NECESSARY-NOT-SUFFICIENT / SHARED knife edge) and §12.2's point rule on ν_M. R3 itself (U dominates) is NOT replicated: no U arms |
| R-MOVES B6 (dose), B7 (placebo: a wrong move set hurts) | **NOT MET** | no D50 / D90 / W90 in the replica |
| R-RIVALS U = U-SHARED | **NOT MET** | no U arms |
| R-RECOVER (POOL 0.80, POOLSETS, MAP100) | **NOT MET** | no such arms |
| R-EVAL (not evaluator-bound; information-bound support) | **NOT MET** | LEAF scorers only |
| R-STAGE-VOI / -AXIS / -MECH / -CLASS / -SEARCH | **NOT MET** | no supplement, no U, no critic. The k-gradient sub-read DIFFERS (§7) |
| R-DOSE (P-STAR 0.1, TODAY 0.009, CONCAVE) and the post-hoc tie-break finding | **PARTLY MET** (descriptive) | the M8 curve and its tie-break step reproduce on the replica's own worlds, post hoc for both; FN (TODAY) and M5 / M5W / M5X are not replicated |
| Amendment 3 (per-class curves) | **NOT MET** | needs F+U |
| FN buys nothing | **NOT MET** | no FN |
| "items / abilities / spreads add ≈ 0.003" (descriptive) | **PARTLY MET** | P − F +0.001 / +0.006, AGREE by the brief, BOUNDARY on §12.2 |

**R-REPLICATE as a whole (§12.2): NOT MET.** The point rule fails on ν_M. TV < 0.02 fails, though it is below the noise floor
(FINDING 2). No verdict exists on the replica's side. By §13.2, **nothing that rests on the U class, the stage rules,
R-RECOVER or R-EVAL may steer research yet.** The honest gap G and the moves' minority sufficiency share now have an
independent second measurement.

## 9. What the red team should attack hardest (in order)

1. **The U class: R3 rests entirely on arms that only the original built** (F+U, O−U). The original's never-seen machinery is
   unreplicated: the species draw, the gender pad's draw for a never-seen mon on a true draw slot, O−U's construction, and
   the k = 6 identities. The two builders' never-seen worlds already differ beyond noise (U.move TV), and their G-by-k
   gradients differ (§7). An independent F+U / O−U on C is the single most valuable next read.
2. **The not-missing-at-random exclusions.** Both builders drop the states they cannot fit (the original's F 8.8 % of all
   Tier-1S states, the replica's 6.4 % of the registered states), and those are the states where hiding the moves costs most (§4: n_M 1.17 on the replica's 111). Bound
   R3 under the extremes: what n_U − n_M becomes if the excluded states carried n_M ≈ 1.
3. **The NECESSARY-NOT-SUFFICIENT / SHARED knife edge.** s_M's upper bound is within 0.025 of 0.25 in all four reads. Should
   R3 fall, the fallback verdict is a coin flip at this power.
4. **Common-mode failures the replica cannot see.** Both use the same Phase 1 truth rows, referee seeds, bank, registered
   states, pins, Smogon prior, the true dice seed (the dice-silent leak of Focus Band / King's Rock / speed ties is in BOTH),
   and the same written spec. A shared spec ambiguity (is Hidden Power's type M or S?) shows up as an implementation
   difference only where they chose differently. Attack with a second referee and a different-seed world set.
5. **Tie-dominated contrasts.** 60 % of states tie under the honest 8-world leaf, so IMP_F ≈ 0 is mostly "the tie went to the
   policy's top choice". Every share is a ratio over a G whose honest half sits on the tie rule. Re-read the shares with a
   tie-free scorer (LEAF-K16, or the critic as the tie-break), as the post-hoc finding suggests.
6. **The handicap's size.** It is bounded empirically at n_M +0.10 (point), but it is concentrated on iconic moves (Dragon
   Dance, Baton Pass, Hydro Pump). A replica re-run on the fixed table (its O−M and F arms; cost not measured here)
   would remove it.

## 10. FINDINGS (standing rule 7)

1. **R3's never-seen dominance has NO independent replication.** The replica did not build F+U / O−U because §13.1's
   hand-over spec named only P, F and F+M ("so that G_P, G and s_M can be compared"). The spec, not the replica, scoped U
   out. Every other Tier-1 verdict except G and the M shares is likewise unreplicated.
2. **§12.2's "TV < 0.02" cannot pass at this sample size.** Two halves of one implementation fail it on every class
   (split-half floors 0.04–0.19). The rule needs a noise-calibrated threshold (e.g. TV − its permutation null) before any
   future replication is judged by it.
3. **The honest builders' exclusions are not missing at random.** On the 111 states only the replica could fit, its n_M is
   1.17. Common-set shares understate the moves' necessity share in both implementations.
4. **The NECESSARY-NOT-SUFFICIENT / SHARED call of §12.3's M-only cascade flips** between the implementations and between
   state sets. s_M's upper bound sits at 0.23–0.26 in every read.
5. **A definitional difference: the original's F+M re-draws Hidden Power's TYPE** (with the spread; 16.5 % of seen-mon sets
   differ from the truth only in it). The replica tells the true type. The registered partition puts IVs in S, so the
   original is the literal reading, but F+M is then a slightly weaker reveal in the original. σ_M agrees anyway (0.0079 vs
   0.0086).
6. **The G-by-k gradient differs** between implementations on C: the original −0.023 [−0.033, −0.013] per SD, the replica
   −0.011 [−0.022, +0.0003]. The difference is itself clear: −0.012 [−0.022, −0.002].
7. **The learnset handicap is real in the replica's worlds:** 0 % hits on 7,936 missing-pair slots against the original's
   33 %. It is NOT detectable as a per-state effect on IMP (the difference-in-differences contain 0 or flip sign). Its
   bound is ν_M ≤ +0.016 at the upper end.
8. **Not verified here:** the replica's P is its own re-build of the pilot's base, not the original's `det2` rows, so G_P's
   agreement compares two P builds. The view comparison covers moves only (not items or abilities). The replica's
   two-revision rows (its FINDING 4) were taken as reported. No GPU and no new playouts were used. Everything was read from
   committed or archived rows (rule 5b: nice 19, cores 5–7 / 13–15, `mem_cap` 8 GB, peak 1.3 GB).

## Reproduce

```bash
PYTHONPATH=<worktree>/src nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 \
  /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 \
  designs/research_state/measurements/hidden_info_tier1_2026-10-13/replica_compare.py   # → replica_comparison.json (~1 min)
```
