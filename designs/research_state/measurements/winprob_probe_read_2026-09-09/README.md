# THE PROBE READ — where the win-prob critic loses "who am I playing" and "whose team am I holding"

Follow-up to [`winprob_mixture_diagnostic_2026-09-09`](../winprob_mixture_diagnostic_2026-09-09/README.md),
which established that arm A's win-prob critic emits close to one marginal win probability per state
(turn-1–3 between-opponent spread of `V` = **0.334×** the outcome's) and that the trainee's OWN team
explains **~9×** more of the critic's residual than the opponent does. That measurement said the head
fails to condition. It could not say *why*, and the three possible whys have three different
treatments. This one decides between them.

Run 2026-09-09, offline, CPU only, nothing written under `models/`. Reproduce with `./run.sh`.

**VERDICT: (iii), on every target and on BOTH substrates** — with one cell (CTRL's opponent Elo, a
three-sentinel axis) reading otherwise under the linear rule for a reason this measurement identifies
and closes in §5. The information is in the value path —
`value_pooled`, the tensor the win head literally reads, decodes the opponent's class at **AUC 0.846**
and the trainee's team identity at **macro AUC 0.974** on turn-1 states, both far above their
permutation nulls — while `V` itself, one MLP away from that tensor, sits at **AUC 0.532** on opponent
class (INSIDE its null of 0.511) and at **AUC 0.631** on team identity (above its own null of 0.413,
but **0.343** below the tensor it reads, CI [−0.361, −0.315]), and explains **R² 0.010** of the team's
win rate against `value_pooled`'s **0.671**. The network has not discarded the signal and the signal
is not unobservable. **The head is handed the answer and does not use it.** The treatment is therefore
on the TARGET, not on the input — and §12 of the mixture diagnostic's sketch, a FiLM conditioner that
injects an opponent code into `value_pooled`, is aimed at a gap that is not the binding one.

---

## 1. The three outcomes and their decisions — stated BEFORE the reading

| | what the numbers would look like | what it would mean | the treatment |
|---|---|---|---|
| **(i)** | a linear decoder recovers the target from the RAW observation early but NOT from the trunk / value-path features | the network **DISCARDS** the signal on the way in | an **auxiliary decoding loss** on the value path |
| **(ii)** | neither raw nor features decode it early, and both improve over turns | the signal is **not observable early** | a conditioning **INPUT** — Elo scalar + class bit via FiLM into `value_pooled` |
| **(iii)** | the value-path features DO decode it, and `V` still does not separate | the head **has the information and does not use it** | **target / capacity** levers |

Pre-registered expectation, written down before the run: **(ii) for the opponent** — Gen 3 has no team
preview, so at turn 1 the only opponent-visible fact is the lead species, and identity should be close
to unreadable until the opponent has played — and **(i) or (iii) for the own team**, which is in the
observation verbatim from turn 1. *Both halves of that expectation were wrong.* The opponent's class
and rating are strongly readable at turn 1 (the lead species and the team it implies carry it), and
the value path keeps them.

---

## 2. The frame

| | |
|---|---|
| substrate A | `ai_v12_02_winprob_critic`, last four trace cycles (50 / 60 / 70 / 74M) — **29,495 states · 951 battles · 216 trainee teams · 14 opponents** |
| substrate CTRL | `ai_v12_11_ladder_ctrl10M`, all five trace cycles (2 / 4 / 6 / 8 / 10M) — **25,564 states · 873 battles · 180 teams · 12 opponents**, 40 draws excluded (no binary outcome) |
| checkpoints | the traces' OWN snapshot in each case — `eval_traces/step_74000016/snapshot.zip` and `eval_traces/step_10000032/snapshot.zip`. **Both LOAD at HEAD** (`gen3_critic_route_wave_v1`, no dropped kwargs); neither needed a compatibility hack |
| one frozen forward | every state of every cycle is forwarded through that ONE checkpoint, so the features, `V` and the trunk all come from the same weights. **QC:** on the cycle the snapshot belongs to, the re-forwarded `V` reproduces the recorded `win_probs` to **8.9e-07** (A) and **3.8e-06** (CTRL) |
| selection | every cycle carries an `eval_manifest.json` at a known schema (1 for A, 2 for CTRL) — **no cycle is SELECTION UNKNOWN**, no cell refused. Every fit and every score is Horvitz–Thompson weighted by `capture_rate_win` / `capture_rate_loss` |
| strength axis | bot Elo from `data/gen3_bot_elo_anchors.json`; sentinel Elo from an ALL-STEPS bot-anchored refit of each run's own snapshot ladder (`refit_ladder.py`), with the positional `sentinel_k` → snapshot map VERIFIED against the manifest's own win counts in every cell |

### Feature sets — and why these four

From `projection.py` and `extractor_forward.py`, not from prose:

| set | dims | what it is |
|---|---|---|
| **RAW** | 2501 | the observation vector as recorded |
| **TRUNK (pi)** | 512 | `extract_features()[0]` — the policy head's input: the two team pools + the refined active token + the scalar tail + the hidden-opp belief |
| **VALUE (vf)** | 512 | `extract_features()[1]` — the value head's input |
| **POOLED** | 128 | `stash.value_pooled`. `vf_combined IS value_pooled` before the projection, and the win head reads this tensor directly: **`V = sigmoid(win_prob_head(value_pooled))`.** One MLP separates this column from the next one |
| **V** | 1 | the critic's own output. "Does `V` rank opponents?" is a decode from a single scalar, which is the mixture diagnostic's question restated as a probe |

### Targets

`opp_elo` (the opponent's rating) · `opp_class` (self-play snapshot vs scripted bot) ·
`own_team_wr` (the trainee team's **leave-one-battle-out** IPW win rate, teams with ≥4 battles — the
LOO is what stops the label carrying the outcome of the battle it labels) · `own_team_id` (own-team
identity as a one-vs-rest macro AUC over the 23 (A) / 33 (CTRL) teams with ≥8 battles).

### Decoder

Weighted ridge, standardised, penalty chosen by an **inner** grouped CV inside each **outer** grouped
CV — **both grouped by battle**, so no battle is ever in train and test at once. Binary targets are
scored by the weighted AUC of the ridge prediction. Turn buckets: **1** · **1–3** · **4–10** ·
**11–24** · **≥25**, at most 2 states per battle per bucket. Every cell carries a **permutation null**
run through the identical pipeline (penalty selection included, so selection optimism is priced into
the null too), and every interval is a **battle-clustered bootstrap**. A cell is **DETECTED** only
when the score's CI lower bound clears its own null's 95th percentile, and every RAW-vs-TRUNK-vs-VALUE
comparison is on the **delta's own CI**, paired on the same battle draw.

The full grid — every cell, every feature set, every null and every delta CI — is
[`tables.md`](tables.md) and [`probe_stats.json`](probe_stats.json). What follows is the part that
carries the reading.

---

## 3. Arm A — the headline rows

**Turn 1 only** (the board has said nothing; every battle contributes exactly one state, n = 1,010
states / 951 battles). **Bold = DETECTED** against that cell's own permutation null; the nulls sit at
R² ≈ 0.00 and AUC ≈ 0.50 throughout and are listed per cell in [`tables.md`](tables.md):

| target | RAW | TRUNK (pi) | VALUE (vf) | POOLED | V | Δ(V − pooled), 95 % CI |
|---|---|---|---|---|---|---|
| opponent Elo (R²) | **0.281** [0.221, 0.338] | **0.247** | **0.205** | **0.177** [0.112, 0.234] | 0.006 [−0.014, 0.021] · *not detected* | **[−0.231, −0.109]** |
| opponent class (AUC) | **0.881** [0.855, 0.907] | **0.883** | **0.858** | **0.846** [0.811, 0.877] | 0.532 [0.486, 0.574] · *not detected* (null 0.511) | **[−0.372, −0.258]** |
| own team LOO win rate (R²) | **0.826** [0.746, 0.902] | **0.794** | **0.681** | **0.671** [0.527, 0.790] | 0.010 [−0.011, 0.027] · *not detected* | **[−0.779, −0.521]** |
| own team identity (macro AUC) | **0.959** [0.935, 0.983] | **0.973** | **0.988** | **0.974** [0.966, 0.981] | **0.631** [0.611, 0.660] (null 0.413) | **[−0.361, −0.315]** |

Three of the four `V` cells cannot be told from their own null. The fourth can, and is still 0.343
below the tensor it reads.

**Turns 1–3** (n = 1,902 / 951): opponent Elo raw 0.320 · pooled **0.229** · V **0.057**; opponent class
raw 0.932 · pooled **0.903** · V **0.671**; own-team WR raw 0.811 · pooled **0.585** · V **0.006**;
own-team identity raw 0.983 · pooled **0.955** · V **0.566**.

**Over the whole game**, `V`'s decode of the opponent rises with the turn (0.006 → 0.057 → 0.130 →
0.160 → 0.109 on Elo; AUC 0.532 → 0.671 → 0.767 → 0.801 → 0.772 on class) — the critic learns who it
is playing only as the board tells it — while `value_pooled` already carries the answer at turn 1 and
never carries much more (0.177 → 0.229 → 0.288 → 0.323 → 0.340). And `V`'s decode of the OWN team
*never* rises: 0.010 → 0.006 → 0.015 → 0.067 → 0.078 on the win rate, AUC 0.631 → 0.566 → 0.540 →
0.504 → 0.415 on identity, i.e. **the critic's opinion becomes LESS informative about its own team as
the game goes on**, while `value_pooled` still identifies the team at AUC 0.904 by turn 11–24.

### The two gaps, sized against each other

| turn 1 | representation gap RAW → POOLED | head gap POOLED → V | ratio |
|---|---|---|---|
| own team LOO win rate | −0.155, CI **[−0.269, −0.059]** | −0.661, CI **[−0.779, −0.521]** | **4.3×** |
| opponent Elo | −0.105, CI **[−0.167, −0.050]** | −0.170, CI **[−0.231, −0.109]** | 1.6× |
| opponent class | −0.036, CI **[−0.057, −0.019]** | −0.314, CI **[−0.372, −0.258]** | **8.7×** |
| own team identity | +0.015, CI [−0.010, 0.037] (not detected) | −0.343, CI **[−0.361, −0.315]** | — |

Both gaps are real; they are not the same size. The representation loses a little on the way in — and
that loss GROWS with the turn on the own-team axis (Δ(pooled − raw) on the win rate runs −0.155 at
turn 1, −0.230 at turns 1–3, **−0.443** at turns 4–10, **−0.465** at 11–24, **−0.655** at ≥25) — but at
the decision point where the mixture defect is sharpest, the head throws away four to nine times more
than the trunk ever lost.

---

## 4. CTRL — the substrate the ladder arms actually run on

The same shape at 10M steps, with the head's failure if anything **more** complete: `V` is inside its
own permutation null on the opponent at **every** bucket, turn 1 included (opponent class AUC 0.606 /
0.531 / 0.573 / 0.464 / 0.527 against nulls of 0.555 / 0.548 / 0.549 / 0.511 / 0.501; opponent Elo R²
−0.007 / −0.007 / −0.008 / 0.023 / 0.042), while `value_pooled` decodes class at AUC 0.835–0.895
throughout and Elo at R² 0.185–0.224 from turn 4 on.

| turn 1 (n = 985 / 873) | RAW | TRUNK | VALUE | POOLED | V |
|---|---|---|---|---|---|
| opponent Elo (R²) | 0.077 | 0.091 | 0.091 | 0.049 | −0.007 |
| opponent class (AUC) | 0.889 | 0.870 | 0.871 | **0.861** | **0.606** |
| own team LOO WR (R²) | 0.738 | 0.707 | 0.617 | **0.628** | **0.028** |
| own team identity (AUC) | 0.995 | 0.982 | 0.985 | **0.971** | **0.639** |

⚠️ **The opponent-Elo cell is the one place the two substrates differ, and the reason is power, not
mechanism.** CTRL's sentinel ladder spans 1742–2005 against arm A's 1907–2031, but CTRL has only THREE
sentinels and they appear in only three of its five cycles, so the Elo axis there is nearly the
bot/sentinel split re-expressed as a number. Read CTRL's `opp_class` row, which is the same question
without the axis, and it agrees with A exactly. **Cross-run Elo is not comparable** in any case — each
refit is anchored to the bots within its own run.

⚠️ CTRL ran the **NEW eval regime** (greedy sentinels drawing the trainee's own teams) and arm A the
OLD one. That boundary moves win rates by ~8.9 pp and therefore moves `own_team_wr`'s label scale, but
it cannot manufacture a decodability gap between `value_pooled` and `V` — both read the same states.

---

## 5. The reading, per target

`summarize.py` applies the decision rule of §1 **in code**, so the verdict is a function of the
numbers rather than of the author. It is applied to BOTH early buckets, because they answer slightly
different questions: **turn 1** is the sharpest window (nothing has happened, every battle contributes
exactly one state, and the two columns condition on the same event); **turns 1–3** is the mixture
diagnostic's own window, by which the opponent has moved twice and `V` has had a chance to react to
the board.

| substrate | target | at turn 1 | at turns 1–3 |
|---|---|---|---|
| **A** | opponent Elo | **(iii)** | (iii-partial) — `V` has acquired a weak signal (R² 0.057) |
| **A** | opponent class | **(iii)** | (iii-partial) — `V` 0.671 vs pooled 0.903 |
| **A** | own team win rate | **(iii)** | **(iii)** |
| **A** | own team identity | (iii-partial) — `V` 0.631 vs pooled 0.974 | (iii-partial) |
| **CTRL** | opponent Elo | (ii) — see below | (i) — see below |
| **CTRL** | opponent class | **(iii)** | **(iii)** |
| **CTRL** | own team win rate | **(iii)** | (iii-partial) — `V` 0.058 vs pooled 0.390 |
| **CTRL** | own team identity | (iii-partial) | (iii-partial) |

**"(iii-partial)" means `V` clears its own null but is still far below the tensor it reads** — every
one of those cells carries a V−pooled delta CI entirely below zero (−0.36 to −0.32 on team identity;
−0.27 to −0.19 on class at turns 1–3). It is not a different mechanism; it is (iii) with the head
having recovered a fraction of what it was handed, and on arm A the fraction only appears once the
board has spoken.

**The two CTRL opponent-Elo cells are the only ones that read otherwise, and the counter-evidence is
in this measurement.** CTRL has THREE sentinels appearing in three of its five cycles, so its Elo axis
is close to the bot/sentinel split re-expressed as a number and is underpowered by construction —
`raw` itself is NOT detected there at turn 1 (R² 0.077, CI [−0.022, 0.162]). Two checks contradict the
literal verdict: (a) the same question WITHOUT the strength axis, `opp_class`, reads **(iii)** on that
substrate at every bucket, `V` sitting inside its own null at turn 1 (0.606 vs null 0.555) while
`value_pooled` reads 0.861; and (b) the MLP probe of §6(a) recovers opponent Elo from `value_pooled`
at **R² 0.130, CI [0.014, 0.238]**, clear of its null, where the linear probe found 0.054 and could
not clear it — so CTRL's `(i)`/`(ii)` on Elo is a **linear-probe artefact**, not a representational
gap. Reported as the rule emitted it, and corrected here rather than quietly overridden.

**The two axes did NOT come out differently in kind, which is itself the finding.** The mixture
diagnostic's two conditioning failures — opponent (small) and own team (~9× larger) — have the SAME
mechanism on both substrates. Neither is an input problem.

## 6. Counter-hypotheses

**(a) A linear probe can miss a non-linearly coded quantity.** Checked with a 1-hidden-layer MLP
(64 tanh units, same grouped folds, same IPW weights, same permutation null) on the turns-1–3 cells.
It moves the numbers the WRONG way for the counter-hypothesis — on **both** substrates the MLP reads
MORE out of `value_pooled` than the ridge did:

| turns 1–3, `value_pooled` | linear | MLP |
|---|---|---|
| A · opponent Elo (R²) | 0.229 | **0.273** [0.170, 0.358] |
| A · opponent class (AUC) | 0.903 | **0.925** [0.903, 0.943] |
| A · own team win rate (R²) | 0.585 | **0.746** [0.665, 0.830] |
| CTRL · opponent Elo (R²) | 0.054 *(not detected)* | **0.130** [0.014, 0.238] *(detected)* |
| CTRL · opponent class (AUC) | 0.835 | **0.901** [0.834, 0.953] |
| CTRL · own team win rate (R²) | 0.390 | **0.626** [0.532, 0.718] |

The value path carries MORE than the linear probe found, so the head-side gap is if anything larger
than §3 says, and the one cell where the linear rule read `(i)`/`(ii)` flips to detected.
**ELIMINATED — it strengthens the reading.**

**(b) Leakage through the battle grouping.** Every fold is split on battles, so no state is scored by a
model that saw another state of the same battle. Two residual routes were closed explicitly: the
own-team win-rate label is **leave-one-battle-out**, so it cannot contain the outcome of the battle it
labels; and the own-team nulls permute the **team assignment across battles** (rebuilding the
one-vs-rest labels and recomputing the LOO win rates from the shuffled assignment) rather than
permuting the labels, so the null has the same team-size structure as the real problem.

**(c) The loss-preferring capture rate.** Every fit and every score is HT-reweighted by the manifest's
per-opponent capture rates, so all four decodes are statements about the eval population rather than
the loss-enriched captured slice. The skew is large — arm A's traced sample wins 0.50 of its battles
against a true 0.83 — and it is exactly the confound that reweighting exists for.

**(d) "`V` is a scalar, of course it decodes less."** True and not exculpatory. A scalar can rank
opponents perfectly; a calibrated critic's `E[V | opponent]` must EQUAL `E[y | opponent]` (the mixture
diagnostic's identity test), so a critic that separates opponents would show a high `V`-decode of
opponent strength by construction. `V`'s decode is the identity test in probe form, and it fails the
same way.

**(e) Is the turn-1 opponent decode an artefact of who plays whom?** No: the trainee's team is drawn
independently of the queued opponent, and the decodable opponent signal at turn 1 is the lead species
and the team it implies. What matters for the reading is not WHY the opponent is readable at turn 1 but
that `value_pooled` reads it and `V` does not — a within-state comparison in which every confound is
shared by both columns.

**(f) A calibrated head SHOULD shrink toward the prior under a noisy target.** This is the strongest
remaining counter-hypothesis and it is not eliminated — it is the mechanism §7 names. Under a proper
scoring rule with one 0/1 label per episode, the variance-minimising response to a weak conditional
signal IS to emit the marginal. That does not rescue the critic (a between-opponent win-rate spread of
0.365 is not a weak signal), but it does say the failure is in what the head is PAID to do, not in what
it can see.

---

## 7. What this means for the treatment

**The FiLM conditioning arm sketched in the mixture diagnostic's §12 is aimed at the wrong gap.**
Injecting an opponent code (Elo scalar + class embedding) into `value_pooled` adds information that
`value_pooled` demonstrably already carries — opponent class at AUC 0.846 and rating at R² 0.177 on
turn-1 states, before anything has happened. An arm that hands the head a cleaner copy of a signal it
already ignores can only help if the failure is representational, and it is not. Keep the arm as a
CONTROL if it is cheap (it would falsify this reading if it moved the spread ratio), but it is not the
first thing to build.

**The first thing to build is a head-side test, and it is nearly free.** `V = sigmoid(head(value_pooled))`
and `value_pooled` is already extracted for both substrates by `extract.py`. So:

1. **Refit ONLY the win head on a frozen `value_pooled`**, offline, on the traced states, against
   (a) the terminal outcome it actually trains on and (b) a conditional target — the per-(opponent,
   team) empirical win rate, or an MC continuation. If (b) separates opponents and (a) does not, the
   trunk is sufficient and the entire failure is target-side. This is a CPU afternoon, not a GPU arm.
2. If that confirms it, the live levers are **target-side**: lower-variance value targets (the
   cf-labelled twin heads and the search leaves already in the tree), and/or opponent-stratified
   weighting of the value loss so the between-opponent component is not swamped by the within-battle
   variance the head is currently minimising.
3. **Head capacity** is the cheap second lever: the win head is a small MLP over a 128-dim tensor that
   provably contains the answer, and the MLP probe above recovers noticeably more from that same
   tensor than a linear map does.

**And one thing this read does NOT license.** It says the information is present and unused; it does
not say a head that used it would be BETTER. The identity test (V against its own MC continuation) is
what would tell those apart, and it should be re-run on anything that moves the spread ratio — exactly
as the mixture diagnostic's §12 already required.

---

## 8. Hazards hit (a hazard is a finding)

1. 🚨 **A NEAR-CONSTANT COLUMN, STANDARDISED, IS AMPLIFIED NOISE — and 2,501 of them defeat any
   penalty.** The first revision read out-of-fold R² of **−1.05** on `own_team_wr` from the RAW
   observation and **−0.82** on `opp_elo` from `vf` — worse than predicting the mean, from feature sets
   that visibly contain the answer. Columns are now dropped, not rescaled, when the training sd is
   below `1e-4 × (1 + |mu|)`.
2. 🚨 **A RARE-VALUE COLUMN IS AN OUTLIER FACTORY, and the inner CV cannot see it.** A channel that is
   all-zero in the training fold and non-zero in the test fold, or one that fires on two rows, survives
   a variance floor; standardised it puts those rows at ±30, and in the `d > n` dual regime the ridge
   extrapolates off them until ONE test row destroys the score (measured: R² **−1221** on `own_team_wr`
   at turns 4–10). Fixed by requiring **≥5 training rows to differ from the column's median**, with a
   ±8 clip on the standardised values as the belt behind it. The variance floor alone does not catch it.
3. 🚨 **A ONE-VS-REST PERMUTATION THAT PERMUTES THE LABEL LIST IS NOT A NULL.** The first revision's
   `own_team_id` null re-ordered the same set of one-vs-rest label vectors, so the macro AUC was
   identical to the observed score by construction — the "null" equalled the measurement to four
   decimal places in every cell, which is what gave it away. The null now permutes the **team
   assignment across battles**.
4. 🚨 **A GROUP-LEVEL PERMUTATION IS NOT A CHANCE LEVEL FOR A LABEL THAT IS A FUNCTION OF THE GROUP.**
   Shuffling the TEAM → win-rate map while keeping the label constant within the real team gives a
   "null" that any team-identifying decoder beats: it read **0.98** against a real score of 0.83 and
   convicted a genuine decode of being chance. It is kept, relabelled, as the **identity-mediation
   reference** — it answers "does the representation carry the LABEL beyond bare GROUP IDENTITY", which
   for `own_team_wr` it does not, on any feature set. Different question; never the chance level.
5. ⚠️ **A checkpoint that loads is not a checkpoint at the traces' step.** Both substrates' features
   come from ONE snapshot forwarded over several cycles, so `V_fwd` reproduces the recorded `V` only on
   the exact cycle (8.9e-07 / 3.8e-06) and differs elsewhere (mean |Δ| 0.047 on A, 0.110 on CTRL). That
   is by design — one frozen model is what makes the feature sets comparable — but the off-cycle rows
   are a probe of the FINAL model on earlier states, not a replay.
6. ⚠️ `fit_ladder` still silently returns an UNANCHORED ladder when called from the wrong cwd (the
   hazard the mixture diagnostic recorded). `refit_ladder.py` here inherits its `chdir(repo_root())`
   and its refusal, unchanged.
7. ⚠️ CTRL's `eval_traces/step_2000016` and `step_4000032` carry no sentinels at all (9 bot opponents
   only) and its sentinel ladder rests on four rated nodes. Its opponent-Elo axis is therefore weak by
   construction; its `opp_class` axis is not.

---

## 9. Ledger paragraph (for the orchestrator to append — this file does NOT edit the ledger)

> **2026-09-09 · PROBE READ on the win-prob critic — the mechanism is (iii): THE HEAD HAS THE
> INFORMATION AND DOES NOT USE IT.** Following the mixture diagnostic, linear probes (weighted ridge,
> grouped-by-battle nested CV, battle-clustered bootstrap, permutation null through the identical
> pipeline, every fit and score HT-reweighted by the manifest capture rates) were fit on four feature
> sets from ONE frozen forward — the raw 2501-dim obs, the trunk `pi`, the value `vf`, and
> `stash.value_pooled`, the tensor the win head literally reads (`V = sigmoid(head(value_pooled))`) —
> plus `V` itself as a fifth decoder. Two substrates, both loading at HEAD unmodified:
> `ai_v12_02_winprob_critic` @74M (29,495 states / 951 battles / 216 teams / 14 opponents, 4 cycles,
> re-forward reproduces the recorded win prob to 8.9e-07 on the exact cycle) and the ladder pin
> `ai_v12_11_ladder_ctrl10M` @10M (25,564 / 873 / 180 / 12, 5 cycles, 3.8e-06). On TURN-1 states of
> arm A, `value_pooled` decodes the opponent's CLASS at **AUC 0.846 [0.811, 0.877]** and the trainee's
> TEAM IDENTITY at **macro AUC 0.974 [0.966, 0.981]**, and explains **R² 0.671 [0.527, 0.790]** of the
> team's leave-one-out win rate and **R² 0.177 [0.112, 0.234]** of the opponent's Elo — every one clear
> of its permutation null. `V`, one MLP downstream, reads **AUC 0.532 [0.486, 0.574]** on class (inside
> its null), **AUC 0.631** on team identity, **R² 0.010 [−0.011, 0.027]** on team win rate and **R²
> 0.006** on Elo. The paired deltas V−pooled are **[−0.372, −0.258]** (class), **[−0.361, −0.315]**
> (team identity), **[−0.779, −0.521]** (team win rate) and **[−0.231, −0.109]** (Elo), all clear of
> zero. The RAW→POOLED representation loss is real but SMALLER at the decision point (−0.036 class,
> −0.155 team WR, −0.105 Elo) — the head gap is 4.3–8.7× it — though it WIDENS late on the own-team
> axis (−0.155 at turn 1 → −0.655 at turn ≥25). CTRL agrees: `V` is inside its own null on opponent
> class at every bucket past turn 1 while `value_pooled` reads 0.835–0.895. A 64-unit MLP probe moves
> the numbers AGAINST the "non-linear coding was missed" counter-hypothesis (pooled → own-team WR R²
> 0.746 vs the linear 0.585). **Reading (i) — the network discards it — is REFUTED; reading (ii) — not
> observable early — is REFUTED (the opponent is readable at turn 1 from the lead and the team it
> implies, contrary to the pre-registered expectation).** Therefore the mixture diagnostic's §12 FiLM
> conditioning arm addresses a gap that is not binding, and the first move is head-side and offline:
> refit ONLY the win head on a frozen `value_pooled` against a conditional target, then target-variance
> levers (cf-labelled twins / search leaves, opponent-stratified value-loss weighting) and head
> capacity. One cell dissents under the linear rule — CTRL's opponent ELO reads (ii) at turn 1 and (i)
> at turns 1–3 — and it is an underpowered-axis plus linear-probe artefact rather than a
> counter-example: CTRL has three sentinels in three of five cycles so even `raw` is not detected there
> at turn 1 (R² 0.077 [−0.022, 0.162]), the axis-free `opp_class` question reads (iii) on that substrate
> at every bucket, and the MLP probe recovers opponent Elo from `value_pooled` at R² 0.130 [0.014,
> 0.238], clear of its null. Where `V` does clear its own null it is still far below what it is handed:
> every "(iii-partial)" cell carries a V−pooled delta CI entirely below zero. Four tooling hazards
> recorded, each of which produced a wrong number before it was caught:
> standardising near-constant obs columns (OOF R² −1.05), rare-value columns exploding the d>n dual
> ridge (R² −1221, invisible to the inner CV), a one-vs-rest null that permuted the label LIST and so
> equalled the measurement exactly, and a group-level permutation used as a chance level when it is an
> identity-mediation reference (it read 0.98 against a true 0.83).
> Measurement: `designs/research_state/measurements/winprob_probe_read_2026-09-09/`.

---

## 10. Proposed amendment to UNDERSTANDING.md §4.2b (text only — not applied here)

> Append to §4.2b, after the mixture-diagnostic paragraph:
>
> **The conditioning failure is a HEAD failure, not a representation failure.**
> [MEASURED, `winprob_probe_read_2026-09-09`] On turn-1 states of `ai_v12_02_winprob_critic` @74M,
> `stash.value_pooled` — the tensor the win head reads, with `V = sigmoid(head(value_pooled))` —
> linearly decodes the opponent's class at AUC 0.846 [0.811, 0.877], the trainee's team identity at
> macro AUC 0.974 [0.966, 0.981], the team's leave-one-out win rate at R² 0.671 [0.527, 0.790] and the
> opponent's Elo at R² 0.177 [0.112, 0.234], each clear of a permutation null run through the identical
> pipeline. `V` itself reads 0.532 (inside its null), 0.631, 0.010 and 0.006 on the same four; every
> paired V−pooled delta CI excludes zero. The raw-obs → value-path loss exists but is 4.3–8.7× smaller
> than the value-path → V loss at that decision point. The ladder's own control substrate
> (`ai_v12_11_ladder_ctrl10M` @10M) agrees, with `V` inside its null on opponent class at every bucket
> past turn 1. A 64-unit MLP probe recovers MORE from `value_pooled` than the linear probe (own-team
> win rate R² 0.746 vs 0.585), so non-linear coding is not the escape. **Consequence:** the
> value-side opponent-CONDITIONING arm sketched in the mixture diagnostic's §12 targets a gap that is
> not binding; the binding one is between `value_pooled` and the head's output, i.e. the TARGET the head
> is trained on. Where `V` does clear its own null (opponent class and Elo from turn 3 on, team identity
> throughout) it still sits far below the tensor it reads, every V−pooled delta CI excluding zero. **Open:** whether refitting the win head alone on a frozen `value_pooled` against a
> conditional target (per-(opponent, team) empirical win rate, or an MC continuation) recovers the
> between-opponent spread — an offline CPU test, not a GPU arm. **Caveat:** a proper scoring rule with
> one 0/1 label per episode makes shrinking toward the marginal the variance-minimising response, so
> "has it and does not use it" is a statement about what the head is PAID to do, not about what it can
> represent.

---

## 11. Files

`refit_ladder.py` (bot-anchored all-steps ladder refit, per run) · `extract.py` (the per-state table +
the frozen forward's four feature sets) · `decode.py` (the grouped-CV ridge probe, the nulls, the
bootstrap, the deltas) · `mlp_probe.py` (the non-linearity check) · `summarize.py` (the reading rule,
applied by code rather than by eye) · `run.sh` · `probe_stats.json` + `tables.md` (the committed
summary; the feature arrays are ~300 MB per substrate and are NOT committed — `run.sh` regenerates
them from `models/` in ~40 min).
