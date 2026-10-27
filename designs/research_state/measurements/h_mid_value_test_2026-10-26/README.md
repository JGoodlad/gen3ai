# H-MID: P\*'s critic does not see its unfamiliar-team conversion deficit (2026-10-26)

**Verdict (registered rule, [`PREREG.md`](PREREG.md), `2dd52278`): CRITIC-MISCALIBRATED-ON-U.** From materially
level positions at turn 10, P\* wins 10.8 pp more often on home72 than on U, and its own critic sees about a third of
that. Δbias = **+0.076 [+0.016, +0.139]**, against the bar of ≥ +0.05 with LB > 0. The PastaMittens replication
points the same way but is INCONCLUSIVE.

Abbreviations: pp = percentage points; LB = lower bound of a 95 % interval; V = P\*'s critic, its win probability
(`sigmoid(win_head(value_pooled))`); U = H, F and P72 at equal weight; level = −1 < D(10) < 1 (within one mon of
material, the forensics' population); bias = mean V − win rate; Δbias = bias_U − bias_home72 = G_y − G_V, the part of
the outcome gap V does not see. Every interval is a 95 % percentile bootstrap over 20-game units resampled within each
cell, B = 2,000.

## Reconstruction (PREREG §3): exact

All 3,200 captures were replayed through the sim_bridge built at each read's own pin (`63f97a575` for Kakuna,
`d32ffcee4` for PastaMittens), and P\* re-read every `__OBS__` row with that pin's code.

| read | games | protocol identical | frames = choices | greedy argmax = recorded choice | excluded |
|---|---:|---:|---:|---:|---:|
| P\* vs Kakuna (X51) | 1,600 | 1,600 | 1,600 | **79,799 / 79,799** | 0 |
| P\* vs PastaMittens | 1,600 | 1,600 | 1,600 | **81,183 / 81,183** | 0 |

Every replayed result equals the forensics row's result. These are the first V traces of anchor games (forensics
FINDING 1); the rows are archived (§ Files).

## Results: level states at turn 10, P\* vs Kakuna (n = 298 home72, 930 U)

| | home72 | U | home72 − U |
|---|---|---|---|
| win rate W | 0.609 [0.562, 0.662] | 0.501 [0.462, 0.540] | G_y **+0.108 [+0.046, +0.173]** |
| mean V | 0.557 [0.538, 0.580] | 0.525 [0.510, 0.538] | G_V **+0.033 [+0.010, +0.059]** |
| bias (V − W) | −0.052 [−0.103, −0.004] | +0.024 [−0.017, +0.063] | Δbias (U − home72) **+0.076 [+0.016, +0.139]** |
| Brier | 0.2245 | 0.2284 | ΔBrier +0.004 [−0.018, +0.026] |
| log loss | 0.643 | 0.647 | +0.005 [−0.044, +0.052] |
| AUC of V for the outcome | 0.663 [0.586, 0.735] | 0.667 [0.633, 0.702] | −0.004 [−0.090, +0.078] |

**(a) Calibration.** V rates level U positions 3.3 pp below level home72 ones, while P\* converts them 10.8 pp worse.
In absolute terms, V is pessimistic on home72 (−5.2 pp) and about right on U (+2.4, interval covers 0). The rule reads
only the difference (PREREG §6: V was never trained against Kakuna, so its absolute level mixes in that mismatch).

**(b) The policy given the value.** At the SAME V, P\* converts less on U. The matched-V outcome gap, U reweighted to
home72's V distribution, is **+0.086 [+0.023, +0.151]**: about 80 % of the raw gap survives matching on V. Where V says
"winning" (V ≥ 0.6, ≈ 40 % of level states on both sets), P\* converts **0.756 on home72 against 0.650 on U** (+0.106
[+0.026, +0.186]). Reliability by V bin (win rate / mean V):

| V bin | home72 | U |
|---|---|---|
| 0.2–0.4 | 0.458 / 0.334 (16 %) | 0.307 / 0.322 (21 %) |
| 0.4–0.6 | 0.508 / 0.512 (40 %) | 0.490 / 0.498 (38 %) |
| 0.6–0.8 | 0.741 / 0.681 (37 %) | 0.631 / 0.678 (34 %) |
| ≥ 0.8 | 0.867 / 0.847 (5 %) | 0.822 / 0.838 (4 %) |

**(c) Discrimination.** V ranks positions on U as well as on home72 (AUC 0.667 vs 0.663 at level 10, 0.711 vs 0.712
over all turn-10 states). What fails on U is V's LEVEL, not its ordering of games.

**When the critic's view forms (secondary, descriptive).** V's set gap is there at turn 1 and does not grow through
the mid game, while the outcome gap stays at ≈ 11 pp:

| turn (level states) | G_y | G_V | Δbias |
|---:|---|---|---|
| 1 (all games) | +0.112 | +0.035 [+0.026, +0.044] | +0.077 [+0.019, +0.136] |
| 5 | +0.115 | +0.035 | +0.080 [+0.021, +0.135] |
| 10 | +0.108 | +0.033 | +0.076 [+0.016, +0.139] |
| 15 | +0.114 | +0.029 | +0.084 [+0.017, +0.149] |
| 20 | +0.141 | +0.060 | +0.081 [+0.002, +0.159] |
| 30 | +0.039 | +0.034 | +0.005 [−0.060, +0.070] (n = 131 / 466) |

The V drift from turn 10 to 20 (V at the end if the game ends first) is +0.017 on home72 and +0.001 on U (difference
−0.017 [−0.054, +0.021]). The critic does not revise its view of U positions downward over the mid game. The deficit
reaches it only at the outcome.

**PastaMittens (replication, outside the verdict): INCONCLUSIVE.** Level at 10, n = 314 / 928: G_y +0.074 [+0.006,
+0.150], G_V +0.040 [+0.016, +0.065], Δbias +0.033 [−0.027, +0.098], matched-V gap +0.038 [−0.023, +0.101], AUC 0.662
vs 0.658. The direction is the same with a smaller outcome gap against this opponent (≈ half of it unseen against ≈
70 % against Kakuna). The twins share teams, seeds and our deterministic policy, so this replicates the opponent, not
the team sample.

## Predictions (PREREG §7)

| | prediction | result |
|---|---|---|
| P1 (forensics README) | ΔBrier ≥ +0.02 | **MISS**: +0.004 [−0.018, +0.026]. Brier is dominated by resolution: a uniform ≈ 5 pp level shift moves it by ≈ 0.003, so Brier was the wrong instrument for this question |
| P2 (this author) | CRITIC-MISCALIBRATED-ON-U with G_V ≤ +0.04 | **HIT**: G_V +0.033 |
| P3 (this author) | AUC on U lower by ≥ 0.03 at turn 10 | **MISS**: equal (+0.001) |

## Reading

1. **The critic shares P\*'s home-team view.** It rates level positions on unfamiliar teams almost as highly as on
   home teams (about 3 pp lower, already at turn 1), but P\* converts them about 11 pp worse. At equal V it wins 8.6 pp
   less on U. So the critic's value carries little of the familiarity premium. Put the other way, against Kakuna P\*
   converts its familiar teams better than its own critic expects, and its unfamiliar ones about as V expects. These
   are one fact read from two absolute baselines.
2. **The miss is a level shift, not lost discrimination.** V orders U games as well as home72 games (equal AUC). It
   knows which U positions are better. It does not know that P\* converts the whole set worse. This fits the
   forensics' team-context account (H-CTX): a per-team skill that applies across the game, not specific positions
   the critic misreads.
3. **What it means for the levers.** Under the brief's map the lever is value-training breadth, or search on the
   value. Two qualifications (FINDING 1). (a) A one-ply search compares successors WITHIN a game, so V's
   between-set level error does not by itself rule out search headroom on U. V's within-set ordering is intact. A
   direct search-headroom read on U (one ply on V against the policy's choice, banked positions) is the test. (b) A
   breadth lever that trains on more teams moves the policy and the critic together. This read cannot separate
   "value breadth" from "policy breadth" as a TRAINING lever. It says only that the critic does not already hold the
   answer the policy lacks.
4. **H-MID's second branch is not supported.** POLICY-CANNOT-ACT needed V to see most of the gap (G_V ≥ 0.05 with
   Δbias covering 0); V sees ≈ 30 %.

## FINDINGS (standing rule 7)

1. **The lever map is contested and this read does not settle it.** The brief maps CRITIC-MISCALIBRATED-ON-U to
   "value training breadth or search on the value". The forensics README maps the same branch to "search on V cannot
   help". The data (§ Reading 3) argue that neither follows directly: V's level is wrong on U but its ordering is not.
2. **The absolute bias is ambiguous by design.** V is under-confident on home72 (−5.2 pp, LB < 0) and about calibrated
   on U. Whether to call that "V optimistic on U" or "V blind to P\*'s home-team skill" depends on an absolute baseline
   that V's training opponents (self-play and league, never Kakuna) set. The registered rule reads only the
   difference.
3. **The pinned loader returns the policy in `training=True` mode** (`load_checkpoint_strict` at `63f97a575` /
   `d32ffcee4`; HEAD's loader returns `False`). Production dropout is 0, and the X51 read itself loaded it this way.
   The 100 % argmax agreement shows it changes nothing here. It is a latent hazard for any network that gains dropout.
4. **The two-game pre-registration smoke ran MAIN's code, not the pin's** (`python -I` drops `PYTHONPATH`, so the
   editable install served HEAD `c758fcb8`). The full runs dropped `-I`, and each printed the pin's `agents/__init__.py`
   path. The 64 / 64 smoke agreement therefore holds at HEAD too. That is incidental evidence that HEAD's encoder
   reads these games identically, not a check of it.
5. **`reconstruct.py` changed after registration by one diagnostic print** (the imported code path), before any V was
   read. Nothing else in the registered scripts changed.
6. **The first Kakuna reconstruction was killed by my outer `timeout` while it waited for a heavy slot** (exit 124,
   partial file discarded). The re-run completed (1,600 / 1,600).
7. **Ties:** one Kakuna game and two PastaMittens games end in a tie and count 0.5 (PREREG §4). The Kakuna tie
   (`pstar_home72_13`, 142 turns) is in the level-10 home72 population. Scoring it 0 or 1 instead moves W_home72, and
   so Δbias, by ±0.0017, which does not change the verdict.
8. **Turn 30 is thin and selected** (n = 131 / 466, games still live): its Δbias of ≈ 0 is not evidence that the
   critic catches up late. The turn-1 V gap of 3.5 pp is unexplained. Both panels draw both sides from the same set, so
   it is not a team-strength edge. It may be a familiarity cue the critic reads from our own six. Not tested.

## Files

- `PREREG.md`: the registration (`2dd52278`).
- `scripts/reconstruct.py`: capture → pinned sim_bridge replay → P\*'s V and logits per decision, with the fidelity
  checks. Run with `PYTHONPATH=<pin>/src`, `POKESIM_SIM_BRIDGE_BIN=<pin>/src/rust_sim/target/release/sim_bridge`,
  `CUDA_VISIBLE_DEVICES=""`. About 6 min per 1,600 games on the agent cores, 1.1 GB peak.
- `scripts/analyze.py` → `results.json`: every number above, the verdict and the prediction scores (≈ 2 min, no
  torch).
- Reconstructed rows (one per game, every decision's turn, V, argmax, recorded choice and margin; 0.9 MB each):
  `~/gen3ai_archive/h_mid_value_test_2026-10-26/recon_{kakuna,pm}.jsonl.gz`, sha256 in `recon_sha256.txt`. The
  analysis also reads the forensics rows (`~/gen3ai_archive/piloting_gap_forensics_2026-10-26/games.jsonl.gz`).
