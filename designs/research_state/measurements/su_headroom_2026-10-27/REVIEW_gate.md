# Adversarial review: the S-U gate's GO (2026-10-28)

**Scope.** This review attacks the S-U gate's verdict, **GO**: S's deviations from π are right **0.635 [0.578, 0.687]**
of the time on pooled U (n 346 countable deviations; H 0.640, F 0.660, P72 0.605), against home72's 0.537 [0.458,
0.617]. It reads:
- the gate's [`PREREG.md`](../su_gate_2026-10-27/PREREG.md) and [`read_spec.json`](../su_gate_2026-10-27/read_spec.json)
  (`ef05d165c`, pin `835c6b4b2`);
- its scorer `main.value_search.sign_gate`, the referee `main.value_search.playout.referee` and the world builder
  `main.value_search.worlds`, at the pin;
- its record, [`README.md`](../su_gate_2026-10-27/README.md) (`fea88cee`), and its state
  `~/gen3ai_archive/su_gate_2026-10-27/` (the frozen draw, the scored rows, `decision.json`), read-only;
- this folder's [`REVIEW.md`](REVIEW.md) (`7aafa527`), which proposed the gate;
- [`design_v_rank_on_u.md`](../../../endstate/design_v_rank_on_u.md), the licensed next step;
- H-CTX ([`../hctx_rc72_2026-10-26/README.md`](../hctx_rc72_2026-10-26/README.md), `79ef3e16`) and its A-K check
  ([`../hctx_kk_2026-10-27/README.md`](../hctx_kk_2026-10-27/README.md), `79ffe8a4`).

The PREREG and the gate's record are not edited.

**Method.** Checks run on the banked rows only: no game, search or playout. They are in
[`scripts/gate_review_checks.py`](scripts/gate_review_checks.py), and every number below is in
[`gate_review_results.json`](gate_review_results.json). The script runs in under a minute on one core at nice 19.
- The verdict was re-derived twice: the pin's own `sign_gate read`, and the script's clone of `cluster_ratio_boot`.
  Both reproduce `decision.json` to the last digit.
- The selection was re-derived from the 39 counted cells' kept logs with the pin's `select`. It gives the same 1,560
  keys in the same order.
- Everything else is **POST HOC**: this review chose its splits after seeing the scored rows.
  - It looked at seven splits: species hit, the Q̂ gap, the deviation's kind, T(π top), past rewrites, the Speed pin,
    and the last two combined.
  - Read every split's interval with that in mind.

Abbreviations follow the PREREG.
- T is the 24-seed self-referee.
- "Countable" means S deviated and its two T differ.
- **Leak-exposed** means the state carries one of the two leak markers the gate recorded per state. Both are
  properties of the TRUE root, fixed before S runs:
  - a rewritten past opponent choice (`leak.rewritten_moves` or `rewritten_switches` > 0);
  - a seen opponent mon whose Speed the true seed pins (`speed_flip.speed_pinned` > 0).

## Verdict

- **The GO is computed correctly under its registered rule, and it stands.** No BLOCKER. These all hold:
  - the selection and the arithmetic reproduce;
  - a team-clustered interval is no wider than the game-clustered one;
  - ties and exclusions cannot flip the verdict.
- **The GO does not show that an HONEST S beats chance on U** (M1). Its whole excess over home72 sits in the
  leak-exposed half of the U states, at 0.716 [0.643, 0.782].
  - In the leak-clean half, U reads 0.554 [0.469, 0.629] and home72 0.563. That LB would read STOP.
  - The leak's value, not its exposure, differs by panel: about half the states are exposed in every panel.
  - This is the review's M1 with a measured sign: the seed leak pays more where the Smogon prior is wrong.
- **The self-referee's opponent model differs by panel** (M2). Kakuna plays teams from the same panel as ours, so on
  U both S's leaves and T model the opponent as P\* piloting an UNFAMILIAR team. On home72 that opponent is P\* at
  home.
  - "S fixes π's mistakes on U" and "S and T share a weaker, P\*-shaped opponent on U" predict the same +0.098.
  - The banked rows cannot separate them.
- **V-RANK as drafted is the wrong next ≈ 96 CPU-h** (M3). The two cheaper steps below come first.
  - Its primary (the all-siblings slope) cannot see the deviation contrast, as the S-U review's M6 already said.
  - At the gate's own measured effect, a live check of the deviation contrast has z ≈ 1.4 at V-RANK's U size, even
    if Kakuna behaved exactly like the self-referee.
  - It would carry the same seed leak.
- **The order this review recommends:**
  1. 0 CPU: record this review's leak split beside the GO.
  2. ≈ 5 CPU-h: re-gate with a leak-reduced S.
  3. Then a deviation-only live branching read sized for the contrast (≈ 20–25 CPU-h), in place of V-RANK's
     all-sibling slope.

## BLOCKER

None.

- **The verdict reproduces:** 0.6348775 [0.5778559, 0.6865951], n 346. The record's refusal ("the decision carries no
  verdict", `pipeline.log`) is a rule-shape defect (the operator's G-5). It does not touch the numbers.
- **The draw reproduces:** 1,560 keys, two per game, all from counted cells. Every root guard passed, since the score
  raised no `GateError`.
- **Thread count does not move S.** The 15 states first scored under the default thread pool and set aside have the
  same picks as their final rows, |ΔQ̂| ≤ 1e-6, and identical referee strings on all 6 of their deviations. This
  partly covers the skipped second keying (G-2), across thread counts only.

## MAJOR

**M1. The GO lives in the leak-exposed states, and the U excess over home72 lives there entirely.**

| pooled sign accuracy (game-clustered) | U | home72 |
|---|---|---|
| all countable deviations (the registered primary) | **0.635 [0.578, 0.687]**, n 346 | 0.537 [0.458, 0.617], n 134 |
| leak-exposed states | **0.716 [0.643, 0.782]**, n 172 | 0.500 [0.375, 0.638], n 54 |
| leak-clean states | **0.554 [0.469, 0.629]**, n 174 | 0.563 [0.458, 0.667], n 80 |
| a rewrite, Speed not pinned | 0.720 [0.620, 0.810], n 103 | 0.500, n 36 |
| Speed pinned, no rewrite | 0.718 [0.590, 0.851], n 44 | 0.417, n 12 |

- **The interaction.** The exposed-minus-clean gap is +0.161 [+0.057, +0.268] on U and −0.062 [−0.243, +0.106] on
  home72. The interaction is **+0.224 [+0.021, +0.427]**. It is post hoc, one of seven splits, and its LB is near 0.
- **Two markers, one direction.** Each marker alone, without the other, gives ≈ 0.72 on U and ≤ 0.52 on home72.
- **Exposure is not what differs.** The exposed share of searched states is 0.47 on home72 and 0.49–0.55 on U.
  - Exposed states are later (turn ≈ 20 against ≈ 18) and better (root V 0.57–0.64 against 0.41–0.53) on every
    panel, home72 included.
  - Home72 shows no gap, so game phase alone does not explain U's.
- **The mechanism** (MY JUDGEMENT, UNVERIFIED by any read). With `dice="seed"`, the world builder makes every world
  reproduce every printed damage number on the TRUTH's own roll, and replay the prefix on the true seed (`worlds.py`,
  THE DICE).
  - That screen pins hidden spreads, Speeds and silent choices toward the truth. T plays in the truth.
  - The screen's information gain grows with the distance between the Smogon prior and the true sets: P72's
    procedural spreads, H's ladder sets and F's forum sets, against the pool's standard spreads. This review's M1
    predicted exactly that and left the sign open; here it reads positive.
- **Why it matters beyond this gate.** The live searcher builds its worlds the same way. So V-RANK's branches and the
  S-U read would both measure S **with** the leak, on the in-process core.
  - A search on a real server has no true seed (MY JUDGEMENT).
  - A U gain that rides on the leak would not transfer to play against Metamon.
- **What it does not show.** That the honest S is at chance. 0.554 has half the n, and its interval holds the full
  read's point estimate.
- **Fix.**
  - Report the leak split beside the GO. The operator's ledger line "leak exposure is higher on U" should read "the
    leak's value is higher on U; its exposure is about equal".
  - Before any live spend, re-gate with a leak-reduced S (M3, step 2).

**M2. The self-referee's opponent is P\* on the panel's teams, so the shared-model bias is panel-dependent by
construction.**
- **The teams.** The anchor's 20-game series are role-balanced. In every panel, 44–50 of the ≈ 50 teams Kakuna played
  are teams we played in the same panel (`teams` in the results).
  - In the true log, and so in T, the opponent's team is a U team on U and a pool team on home72.
  - T and S's leaves both pilot it with P\* greedy.
- **The two opponents differ in strength.** P\* pilots unfamiliar teams worse: against Kakuna it scores 0.615 on
  home72 and 0.540 on F (H-CTX, one engine).
  - On U, S and T therefore share a weaker, P\*-specific opponent: deterministic, greedy, and blind where P\* is blind
    on an unfamiliar six.
  - On home72 they share P\* at its best.
  - Kakuna is not P\*. V-RANK §2 puts it ≈ 11 pp better than P\* at piloting U teams.
- **Two accounts, one prediction.** Both predict U > home72 under T:
  - "π is worse on U, and S fixes it", which transfers;
  - "S exploits the P\*-on-U opponent that T also plays", which does not.
  The banked rows hold no non-P\* opponent, so they cannot separate the two.
- **A descriptive hint, not evidence.** Where S switches and π attacks, U reads 0.60 / 0.78 / 0.69 on H / F / P72,
  against 0.52 on home72. Out-guessing a deterministic greedy opponent pays there.
- **Consequence.** This review's own argument still stands: "chance under T predicts chance or worse against Kakuna"
  is valid for a STOP. It says nothing for a GO, and the PREREG says so ("a GO licenses only the next stage").
  - The U − home72 difference (+0.098 [−0.002, +0.194]) has no mechanism reading at all. It is not only
    "leak-confounded" but opponent-model-confounded.

**M3. What the GO licenses: V-RANK as drafted is under-powered for the contrast that matters, and it inherits M1.**
- **The contrast.** S acts on π's top against S's pick. V-RANK's primary is the all-siblings slope β, whose ON class
  is uninformative (`REVIEW.md` M6: β 0.54 on P\*'s states with H ≈ 0).
- **Its deviation statistic.** It is reported only, at ≈ 480 U deviations (1,200 U positions × 0.4).
- **Its power, at the gate's own measurement.** On U the mean T(pick) − T(top) per deviation is +0.035, and a seed
  separates the two actions 29 % of the time.
  - So one live game an action gives ≈ 0.56 accuracy among split pairs.
  - At 448 deviations that is z ≈ 1.4, even if Kakuna behaved exactly like T. At home72's n it is z ≈ 0.5.
  - Its 96 CPU-h buys a slope this decision does not need. Its deviation row cannot confirm the GO, and it can refute
    it only if Kakuna reverses it.
- **A Kakuna referee in the core does not exist.** `referee(opp_bot=…)` takes only in-core scripted bots, and Kakuna
  is Metamon, outside the core (V-RANK §2 rejects proxies).
  - A bot re-referee of the 480 countable deviations would cost ≈ 2 CPU-h. It only swaps one wrong opponent for
    another, and P\* beats bots, so T saturates and ties rise. Low value.
- **The cheaper path.**
  1. **0 CPU.** M1's split, recorded beside the GO now.
  2. **≈ 5 CPU-h (ESTIMATE at the gate's seed-dice rates): a leak-reduced re-gate on the SAME 1,560 states.**
     - It is S with `dice="rolls"` (F1 tier A, built). That needs a new preset beside `r0s4`
       (`main.anchors.search_presets`: one line) and a short registration.
     - S runs at ≈ 5 CPU-s a state, and the referee at ≈ 12 CPU-s on each new deviation.
     - It removes the damage-roll truth filter. It does not remove the true seed's partial and residual leaks (speed
       ties, hidden durations).
     - So keep the Speed-pin marker as a registered stratifier, and read the leak-clean stratum as a co-primary.
     - If its pooled-U LB is ≤ 0.50, the GO was the leak's, and search stays paused.
  3. **If it holds: a deviation-only live branching read** in place of V-RANK's all-sibling slope.
     - At each position only S's pick is branched against Kakuna. π's branch is the raw game, already played, so a
       deviation costs one game.
     - Sized for the contrast: ≈ 1,800 U deviations (≈ 4,500 positions) give z ≈ 2.8 at T's effect.
     - It costs ≈ 20–25 CPU-h (ESTIMATE): raw games ≈ 7.5, S ≈ 6, branches ≈ 7.5. It needs V-RANK's forced-branch builds
       (B-V1–B-V4).
     - The gate's 1,560 states can be its first block, since their deals replay deterministically.

## MINOR

- **m1. The bank tooth validates arithmetic, not the U scorer.**
  - `bank_check.py` re-scores BANKED Q̂ and T with the gate's acting rule, sign and estimator. It never runs
    `_search`, the world builder, the root replay or the referee, and it holds no U state.
  - What validates the U path is the root guard (1,560 of 1,560), this review's re-derived draw, and the thread
    invariance above.
  - Nothing validates T itself. It is P\*'s playouts by definition.
- **m2. Exclusions differ by panel, in the direction opposite to M1's prediction.**
  - The rates are home72 9.8 %, F 10.8 %, H 1.4 % and P72 1.5 %. Excluded states sit later (turns 20–24, against
    18.5–19.5).
  - **Bound:** count each panel's expected countable deviations among its exclusions as all WRONG. Pooled U is then
    0.605 (F 0.589), so they cannot flip the GO.
  - The cause is UNVERIFIED.
- **m3. Ties.** F ties on 29 % of its deviations and home72 on 11 %. Scoring a tie as ½ gives pooled U 0.604 and
  home72 0.533. Robust.
- **m4. Clustering.** Clustered by our team (50–54 a panel), U reads [0.585, 0.681]. Clustered by the opponent's
  team, [0.583, 0.685].
  - Both are no wider than the game-clustered [0.578, 0.687], so team-level correlation is not hiding.
  - The percentile bootstrap's slight under-coverage (`REVIEW.md` m6) is immaterial at an LB of 0.578.
- **m5. The engine.** The gate ran on `e00a212a…/3d0d5c9c…` (post-Struggle fix), not S-U's registered
  `bdc5d2be…/95b45f10…` (FINDING G-1).
  - S and T share it, so the gate is internally consistent.
  - Any V-RANK or S-U amendment that cites the gate must register this engine. So must the re-gate of M3.
- **m6. `attribution_miss` is Hidden Power.** All 832 opponent move commands absent from the true set are an untyped
  `hiddenpower` choice (home72 165, H 254, F 207, P72 206).
  - The template keeps each one verbatim in every world.
  - Where that choice was protocol-silent, every world must give that mon Hidden Power: a small leak, unquantified.
  - Untyped HP should resolve to the true set's typed slot, as the revealed-HP branch already does.
- **m7. One H unit dropped whole** (`pstar__heldout72_hpfix__106`, Kakuna's turn-250 fault, G-3). That is 20 games,
  and H is left with 180. The drop is rule-based and outcome-blind. Immaterial.
- **m8. The prediction missed high.** The PREREG guessed 0.52 and P(GO) 0.25; the read gave 0.635. M1 is the
  likeliest account of the miss: the review's B2 bank (pool teams, self-play) carries home-like leak value only.

## Not edited (FINDINGS for the orchestrator)

- **The operator's ledger entry, README and status lines.** "Leak exposure is higher on U" should read "the leak's VALUE
  is higher on U; its exposure is ≈ equal (0.47 vs 0.49–0.55)". The GO should carry M1's leak-clean row (0.554
  [0.469, 0.629]) and M2's opponent-model confound.
- **The next step.** V-RANK's licence (`design_v_rank_on_u.md`, the S-U design's decision row) should follow M3's
  order: the leak-reduced re-gate first, then a deviation-only live read. Re-sizing or replacing V-RANK is the
  orchestrator's call.
- **The world builder (m6).** The untyped Hidden Power choice is kept verbatim in the template.
