# THE REPRESENTATION-LEVEL OPPONENT-CLASS DECODE — does `strata` raise and `vf15` lower the class information in `value_pooled`, or only in V?

Run 2026-09-11, offline, CPU only, nothing written under `models/`. Reproduce with
[`run.sh`](run.sh). The two branches and the "moves with" bar were registered in
[`PREDICTION.md`](PREDICTION.md) **before any `pooled → class` number existed for any of the five
checkpoints**, and are applied **in code** by [`tabulate.py`](tabulate.py).

---

## VERDICT

**BRANCH A.** Both arms move at the representation, on both draws, with the right sign, past the
run-level floor. **The two rows are REDUNDANT and the cheaper V-level row in `critic_read` REMAINS
the decision row.** My registered prior (branch B at ~65 %) was **WRONG** — on both arms, in
opposite directions.

| arm | Δ pooled→class @ t4–10 (hp800 · hp800b) | Δ V→class @ t4–10 | ×floor (pooled) | ×floor (V) | MOVES WITH |
|---|---|---|---|---|---|
| **`strata`** (`--win-prob-strata-weight 1.0`) | **+0.048 · +0.064** | +0.050 · +0.051 | 2.09 · 2.07 | 2.53 · 2.16 | **YES, both draws** |
| **`vf15`** (`--vf-coef 1.5`) | **−0.052 · −0.045** | −0.088 · −0.089 | 2.23 · 1.45 | 4.46 · 3.82 | **YES, both draws** |

**Three findings, in order of consequence.**

**1 — The two DOWNGRADED sentences are RESTORED, with one of them corrected in the direction of
being an understatement and the other in the direction of being half right.**

- `strata`: *"the representation carries more class information, not only the head"* — **CONFIRMED**.
  Δpooled / ΔV = **0.97 and 1.27** at t4–10: the representation gains as much as V does, one for
  one. At **t1–3 the representation gains nearly twice what V shows** (+0.076 / +0.087 pooled
  against +0.043 / +0.039 at V), so early in a battle the head is cashing in about half of what
  the trunk it is handed has gained. The sentence was, if anything, too weak.
- `vf15`: *"raising the value-loss coefficient REMOVED opponent information from the shared
  representation"* — **CONFIRMED AT t4–10 AND t11–24, AT ABOUT HALF THE SIZE OF THE V EFFECT, AND
  NOT AT ALL EARLY.** Δpooled / ΔV = **0.59 and 0.50** at t4–10; at **t1–3 the representation is
  UNMOVED** (−0.012 / −0.004, inside every floor) while V falls (−0.020 / −0.028). So roughly half
  of `vf15`'s V-level damage is a genuine loss of class information in the tensor the head reads,
  and roughly half is the head's own use of an only-partly-degraded representation — a distinction
  the V row alone could not make and this row does.

**2 — The V-level row is not merely cheaper, it is the TIGHTER of the two.** The control-pair floor
on the representation row is **0.023 (hp800) / 0.031 (hp800b)**; on the V row it is **0.020 /
0.023**. The representation row's run-level noise is ~15–35 % larger, its between-draw spread on
`strata` is 5–10× the V row's (+0.0101 vs −0.0010), and it costs a ~25-minute forward pass and a
300 MB tensor per frame against a `critic_read` invocation. **A row that is noisier, slower and
tells the same story is not a replacement for the decision row.** The consequence registered in
branch A therefore lands with an extra reason behind it.

**3 — An unlooked-for cross-instrument validation.** `frame_check.py`'s `V_to_opp_class_AUC`,
computed by a different fitter on an unmatched frame with a different fold split, reproduces
`critic_read`'s tool-v6 `cond.opp_class_auc.t4_10` **to the third decimal on four of five
checkpoints**:

| | this file (hp800) | `critic_read` v6 (hp800) |
|---|---|---|
| `strata` | **0.7613** | 0.760 |
| `ctrl10M` | **0.7114** | 0.710 |
| `ctrl10M_b` | **0.7052** | 0.707 |
| `ctrl10M_c` | **0.6917** | 0.688 |
| `vf15` | **0.6235** | 0.628 |

Two independently built instruments agreeing to ±0.005 on five checkpoints is a stronger statement
about the row's validity than either instrument's own CI, and it is free.

---

## 1. Why this measurement exists

The 2026-09-11 CORRECTION established that the ladder's conditioning decision row
`cond.opp_class_auc.t4_10` and its reference `cond.spread_ratio_optimal.*` are both fitted from
**V — the win head's scalar OUTPUT** — and not from `value_pooled`, the 128-dim tensor the head
reads. Two campaign claims were downgraded on that basis: `strata`'s "the representation carries
more class information, not only the head" (V-level Δ **+0.050 [+0.036, +0.064]** against a
**0.0219** floor, held at +0.033 and +0.044 on two further draws) and `vf15`'s "raising the
value-loss coefficient REMOVED opponent information from the shared representation" (V-level Δ
**−0.082 [−0.096, −0.066]**, repeated at −0.080). Neither sentence had ever been measured at the
representation, and the research direction — a win-prob value function good enough to be a search
leaf — rests on the representation, not on the scalar.

This runs the row that measures it, on five checkpoints and two independent eval draws, with the
branch registered in advance.

---

## 2. The frame

Five checkpoints, all at **`step_10000032`**, each read on the offline `eval_trace_gen` trees the
ladder already reads — **two independent eval draws** where both exist:

| draw | tree | seed | games |
|---|---|---|---|
| `hp800` | `.../hp_eval/hp800/<run>/eval_traces/step_10000032` | 20260910 | 800 × 12 opponents |
| `hp800b` | `.../hp_eval/hp800b/<run>/eval_traces/step_10000032` | 20260911 | 800 × 12 opponents |

`ai_v12_16_ladder_ctrl10M_c` has **no `hp800b` tree** (checked before the prediction was written),
so draw B carries one control pair instead of two. That is stated in the floor column, never
absorbed.

**Three controls, not one.** `ctrl10M`, `ctrl10M_b` and `ctrl10M_c` are three runs of the same
configuration, so the |control − control| differences ARE the run-level component of any delta on
this row. Per the campaign's rule 19 a two- or three-point spread **BOUNDS** a floor and carries
no CI; it demotes and never promotes.

## 3. The instrument — and what changed relative to `critic_read`

`extract.py` forwards each checkpoint over its own eval tree and writes `value_pooled` — **the win
head's literal input**, 128-dim — beside the recorded win probability. `frame_check.py` then fits,
**out of fold and grouped by battle** (`GroupedRidgeCV`), two decoders of the same binary label
("is this opponent a sentinel?") on the **same rows, the same folds and the same subsample**:

- `pooled_to_opp_class_AUC` — 128-dim `value_pooled` → class. **The representation.**
- `V_to_opp_class_AUC` — the single column `V` → class. **The head's scalar output.**

That side-by-side construction is the whole point: the two numbers differ only in the tensor, so
their difference is not contaminated by a different frame, a different fold split, a different
quota or a different fitter. Both scripts were run **unmodified** from
`../winprob_refit_ncurve_2026-09-10/`.

🚨 **This is NOT `cond.opp_class_auc.t4_10`.** That row comes from `main.ops.conditioning_meters`
on a quota-matched frame with its own bootstrap CI. The `V_to_opp_class_AUC` column here is the
same *quantity* computed by a different instrument on an unmatched frame, and it is here as the
**internal comparator for the pooled row**, not as a re-run of the ladder's row. Where the two
agree that is a cross-instrument check worth having; where they differ, `critic_read` is the row of
record for V and this file is the row of record for `value_pooled`.

**Own-team leak.** Every frame carries `own_team_LOO_to_opp_class_AUC` — the trainee's own team's
leave-one-battle-out ability to predict the opponent's class. The N-curve's hazard 1 is that a
turn-1 "opponent decode" was an own-team decode at AUC 0.86 on a frame where only 37 of 180 teams
ever faced a sentinel. On these full-capture 800-game trees every team plays every opponent and the
leak must sit at ~0.51; a departure REFUSES the frame, as registered in PREDICTION §5.
---

## 4. THE TABLE

Full machine-readable form in [`frame_check.json`](frame_check.json) and
[`branch_test.json`](branch_test.json); the generated markdown is [`TABLE.md`](TABLE.md).

### 4a. Points — opponent-class decode AUC, out of fold, grouped by battle

| run | draw | pooled t1–3 | **pooled t4–10** | pooled t11–24 | V t1–3 | **V t4–10** | V t11–24 |
|---|---|---|---|---|---|---|---|
| `ctrl10M` | hp800 | 0.684 | **0.830** | 0.803 | 0.580 | **0.711** | 0.715 |
| `ctrl10M` | hp800b | 0.667 | **0.824** | 0.798 | 0.583 | **0.710** | 0.716 |
| `ctrl10M_b` | hp800 | 0.707 | **0.853** | 0.822 | 0.595 | **0.705** | 0.714 |
| `ctrl10M_b` | hp800b | 0.712 | **0.855** | 0.822 | 0.579 | **0.686** | 0.708 |
| `ctrl10M_c` | hp800 | 0.686 | **0.822** | 0.801 | 0.567 | **0.692** | 0.705 |
| **`strata`** | hp800 | 0.760 | **0.879** | 0.820 | 0.623 | **0.761** | 0.735 |
| **`strata`** | hp800b | 0.754 | **0.889** | 0.813 | 0.621 | **0.760** | 0.728 |
| **`vf15`** | hp800 | 0.672 | **0.778** | 0.734 | 0.560 | **0.624** | 0.611 |
| **`vf15`** | hp800b | 0.663 | **0.780** | 0.738 | 0.554 | **0.620** | 0.606 |

### 4b. Deltas against `ctrl10M`, with the floors

`FLOOR` = the widest |control − `ctrl10M`| on that draw (rule 19: a two- or three-point spread
BOUNDS the run-level component and carries no CI). On **hp800** two pairs are available
(`ctrl10M_b` +0.023 / `ctrl10M_c` −0.009 on pooled; −0.006 / −0.020 on V); on **hp800b** only
`ctrl10M_b` exists, so that floor is a SINGLE difference and is weaker still.

| row | draw | FLOOR | `strata` Δ | ×floor | `vf15` Δ | ×floor |
|---|---|---|---|---|---|---|
| **pooled → class t4–10** | hp800 | **0.0232** | **+0.0484** | 2.09 | **−0.0518** | 2.23 |
| **pooled → class t4–10** | hp800b | **0.0310** | **+0.0643** | 2.07 | **−0.0449** | 1.45 |
| **V → class t4–10** | hp800 | **0.0197** | **+0.0499** | 2.53 | **−0.0879** | 4.46 |
| **V → class t4–10** | hp800b | **0.0234** | **+0.0505** | 2.16 | **−0.0894** | 3.82 |
| pooled → class t1–3 | hp800 | 0.0227 | +0.0760 | 3.35 | −0.0121 | 0.53 |
| pooled → class t1–3 | hp800b | 0.0447 | +0.0869 | 1.94 | −0.0038 | 0.09 |
| V → class t1–3 | hp800 | 0.0151 | +0.0429 | 2.84 | −0.0198 | 1.31 |
| V → class t1–3 | hp800b | 0.0033 | +0.0385 | 11.7 | −0.0283 | 8.6 |
| pooled → class t11–24 | hp800 | 0.0186 | +0.0173 | 0.93 | −0.0694 | 3.73 |
| pooled → class t11–24 | hp800b | 0.0234 | +0.0148 | 0.63 | −0.0605 | 2.59 |
| V → class t11–24 | hp800 | 0.0095 | +0.0207 | 2.18 | −0.1032 | 10.9 |
| V → class t11–24 | hp800b | 0.0075 | +0.0119 | 1.59 | −0.1101 | 14.7 |

⚠️ The four t1–3/t11–24 V rows on **hp800b** carry ×floor values of 8–15 because that draw's single
control pair happens to land at 0.003–0.008 on those buckets. **A one-point floor that comes out
small is luck, not precision** — those ratios are printed because the generator prints them and
must not be read as effect sizes. The decision bucket is t4–10, fixed in PREDICTION §3, and it is
the one with a stable floor on both rows.

### 4c. The decomposition — how much of V's move is at the representation

| arm | draw | Δpooled | ΔV | **Δpooled / ΔV** |
|---|---|---|---|---|
| `strata` | hp800 | +0.0484 | +0.0499 | **0.97** |
| `strata` | hp800b | +0.0643 | +0.0505 | **1.27** |
| `vf15` | hp800 | −0.0518 | −0.0879 | **0.59** |
| `vf15` | hp800b | −0.0449 | −0.0894 | **0.50** |

This is the row the measurement was commissioned to produce and the one the V-level row cannot
emit: **`strata`'s effect is entirely a representation effect; `vf15`'s is about half
representation, half head.** The two draws agree on both arms to within the eval-draw component.

### 4d. Between-draw spread — the eval-draw component

| run | pooled t4–10 | V t4–10 | pooled t1–3 | V t1–3 |
|---|---|---|---|---|
| `ctrl10M` | −0.0058 | −0.0016 | −0.0167 | +0.0024 |
| `ctrl10M_b` | +0.0020 | −0.0188 | +0.0053 | −0.0160 |
| `strata` | **+0.0101** | −0.0010 | −0.0058 | −0.0020 |
| `vf15` | **+0.0011** | −0.0031 | −0.0084 | −0.0061 |

At the decision bucket the eval-draw component is **≤ 0.010 on pooled and ≤ 0.019 on V** across all
four runs — smaller than either row's run-level floor, which is the campaign's expected ordering
(a fresh eval draw is cheaper noise than a fresh run) and a small validation of the floor
machinery.

### 4e. Frame QC — every frame passed every refusal in PREDICTION §5

| run | draw | battles | teams | **own-team → class leak** | **max \|V_fwd − V_rec\|** | true WR |
|---|---|---|---|---|---|---|
| `ctrl10M` | hp800 / hp800b | 9,557 / 9,557 | 602 | **0.495 / 0.496** | 1.8e-06 / 1.8e-06 | 0.860 / 0.863 |
| `ctrl10M_b` | hp800 / hp800b | 9,563 / 9,542 | 602 | **0.493 / 0.497** | 1.3e-06 / 1.5e-06 | 0.859 / 0.852 |
| `ctrl10M_c` | hp800 | 9,546 | 602 | **0.494** | 1.3e-06 | 0.850 |
| `strata` | hp800 / hp800b | 9,540 / 9,536 | 602 | **0.492 / 0.498** | 1.4e-06 / 1.0e-06 | 0.853 / 0.845 |
| `vf15` | hp800 / hp800b | 9,537 / 9,559 | 602 | **0.496 / 0.496** | 2.3e-06 / 2.5e-06 | 0.874 / 0.869 |

- **Own-team leak 0.492–0.498 on all nine frames** — at the null, exactly as the N-curve's §3
  predicts for a full-capture matched-team frame, and nowhere near the 0.86 that made the probe
  read's turn-1 decode an own-team decode. 574–585 of 602 teams face a sentinel on a single
  800-game tree (the N-curve's pooled three-tree frame reached 602/602); the leak is at null
  regardless.
- **Frozen-forward QC 1.0e-06 to 2.5e-06**, four orders below the 1e-3 refusal, on all nine.
- **Battle counts 9,536–9,563** — a 0.3 % spread, so both sides of every delta have matched
  precision without any post-hoc trimming.
- Every checkpoint stashes `win_prob_logits`; no arm was excluded.

---

## 5. The reading

**The question was: does `strata` raise and `vf15` lower the REPRESENTATION's class information at
t4–10, or only V's? The answer is BOTH — and the sizes differ between the two levers in a way that
is itself the finding.**

`strata` re-weights the win-prob BCE so the bot and pool strata contribute equally. The registered
prior in §4 of `PREDICTION.md` said this was the arm most likely to be a head-side reallocation,
because the most direct thing a loss reweighting can do is change how the head spends its one
dimension. **That was wrong.** `strata`'s representation moves one-for-one with its head at t4–10
and nearly two-for-one at t1–3: the reweighted value gradient reshapes the tensor the head is
handed, and the head then under-uses the early-window part of what it gains. The mechanism story
the ledger recorded — class-balanced weighting stops the majority stratum from drowning the
between-class difference — survives at the representation, which is where it was claimed.

`vf15` multiplies the whole value gradient by 1.5. The registered prior said this was the arm with
the better claim to a trunk-level effect. **That was half right.** At t4–10 and t11–24 the
representation genuinely loses class information (−0.05 / −0.07, both past the floor, both draws),
but only about half of what V loses; and at t1–3 the representation loses nothing measurable while
V still falls. So the sentence "more value gradient strips opponent information from the shared
trunk" is TRUE at mid and late turns and is NOT the whole account: a comparable share of `vf15`'s
V-level damage is the head making worse use of a representation that is only partly degraded.

**What this settles for the ladder.** The branch was registered before the data: `pooled → class`
moves with the V-derived class AUC across both arms and both draws, at the bar fixed in
PREDICTION §3, so **the two rows are redundant and `cond.opp_class_auc.t4_10` stays the decision
row** — reinforced, not merely survived, by §2's finding that the representation row is the noisier
and far more expensive of the two. `strata_b`'s verdict, already worded at V-level, needs no
amendment; and a PASS there may now legitimately carry the representation reading with it, because
on `strata` the two rows are the same statement. That licence is specific to `strata` and does not
generalise: `vf15` shows the two rows can differ by 2× and, at t1–3, can differ in whether there is
an effect at all. **The general rule stays: V-level is the decision row; a representation claim
about a NEW lever needs this measurement run on that lever, and it costs ~25 CPU-minutes per
frame.**

**What it does not settle.** Nothing here says the row predicts leaf quality or strength; the L1/L2
battery is the test of that and is registered separately. And none of this promotes
`pooled → class` into a bar for anything — it has no within-draw CI (§6.3).

---

## 6. Confounds and limits

1. **A two-or-three-point floor has no CI and can be small by luck.** `FLOOR_pooled` is the wider of
   two control-pair differences on draw A and a single difference on draw B. `lambda09_b`'s
   registration addendum already made this point about L1: for n = 3 the expected range is ≈ 1.69 σ,
   so a spread underestimates the run-level σ. **The floor demotes and never promotes** — an arm
   inside it is not detected; an arm outside it is detected *at the standard the campaign uses*,
   which is weaker than a CI.
2. **One run per arm.** `strata` and `vf15` are each ONE training run. A representation delta that
   clears the floor is a run-level statement at the campaign's standard, not a replicated one.
   `strata_b` is the registered run-level replicate at V-level; it has no representation read yet.
3. **No bootstrap CI on the pooled row.** `frame_check.py` emits a point AUC per bucket. The
   precision here is bounded by the between-draw spread (reported) and the control-pair spread
   (the floor), not by a within-draw interval. A within-draw CI would be a cheap addition and is
   named in §8 as the next increment.
4. **The decoder is a ridge.** `pooled → class` is a LINEAR read of a 128-dim tensor. A lever could
   change the class information in the representation in a way a linear probe cannot see, which
   would bias the pooled row toward "flat". The direction of that bias matters for branch B and is
   stated as such: **a flat pooled row is weaker evidence of "the representation did not change"
   than a moved pooled row is of "it did".**
5. **`value_pooled` is the win head's input, not the shared trunk.** It sits at the end of the value
   path. A lever that changes the shared trunk but is undone by the value path would read flat
   here; a lever that changes only the value path's last layers would read moved. The sentence
   "removed opponent information from the SHARED representation" is therefore not exactly what this
   row tests either — it tests the tensor the head is handed, which is the quantity the research
   question ("is the head handed the answer?") actually names.
6. **Unmatched quota.** `critic_read` quota-matches the arm and control before differencing;
   `frame_check.py` does not. Both sides of every delta here are full-capture 800-game trees with
   602 teams and ~9,500 battles, so the exposure is matched by construction rather than by
   post-hoc trimming — but the battle counts are not identical to the digit and are printed.
7. **t1–3 is reported and decides nothing**, per PREDICTION §3. The N-curve established the
   opponent is unobservable at turn 1 on a matched-team frame, and the campaign's decision bucket
   is t4–10.
8. 🚨 **THE DECODE IS ON EACH RUN'S OWN TRAJECTORIES.** Every checkpoint is read on the battles IT
   played, so a delta in `pooled → class` conflates "this network encodes the class better" with
   "this network's states are more class-separable". `vf15`'s overall true win rate is 0.87 against
   `strata`'s 0.85 and `ctrl10M`'s 0.86, so the state distributions are not identical. **This
   applies equally to the V row**, which is why it cannot change the redundancy verdict — the two
   rows are contaminated by the same thing in the same direction — but it does bound what
   "representation" means here: it is the representation *on the states this policy reaches*, not
   on a common state set. A common-state version (forward every checkpoint over ONE run's tree) is
   the clean form and is named in §8.

---

## 7. Hazards hit (a hazard is a finding)

1. ⚠️ **A ONE-POINT FLOOR CAN COME OUT ARBITRARILY SMALL AND PRINT AN ENORMOUS ×floor.**
   `ctrl10M_c` has no `hp800b` tree, so draw B's floor is a single control difference; on the V
   t1–3 and t11–24 buckets it landed at 0.003–0.008 and the generated ×floor column reads 8.6–14.7.
   Those are not effect sizes. The decision bucket (t4–10) has a two-pair floor on draw A and a
   stable one on draw B, which is why it was fixed in advance. **A floor built from n control pairs
   must print n**, and `branch_test.json` does.
2. ⚠️ **`frame_check.py`'s PROGRESS PRINT LABELS EACH ROW WITH `basename(dir)`, WHICH HERE IS THE
   DRAW, NOT THE RUN.** Nine frames printed as `[hp800]` / `[hp800b]` and the stdout log alone
   cannot say which checkpoint a row belongs to; only the ordering and the `== <name>` header
   disambiguate. The JSON keys are correct and the analysis reads the JSON, so nothing is wrong in
   the result — but a reader of [`frame_check.log`](frame_check.log) must count headers. The script
   was left **unmodified** deliberately (another agent is running it concurrently); the one-line
   fix is named in §8.
3. ⚠️ **THE REPRESENTATION ROW IS NOISIER THAN THE OUTPUT ROW AT RUN LEVEL** (floor 0.023–0.031 vs
   0.020–0.023). The intuition that "closer to the weights" means "less noisy" is wrong here: a
   128-dim decoder has more freedom to fit a run's idiosyncrasies than a one-column decoder does.
   Anyone tempted to promote a representation-level row to a bar on that intuition should read
   §4b first.
4. ⚠️ **MY REGISTERED PRIOR WAS WRONG ON BOTH ARMS AND IN OPPOSITE DIRECTIONS** — `strata` was
   predicted flat at the representation and moved one-for-one; `vf15` was predicted to move and
   moved at only half the V size, with nothing at all early. The prediction is kept in
   [`PREDICTION.md`](PREDICTION.md) unedited. A stated prior that is wrong in both directions is a
   sign the reasoning ("a loss reweighting is head-side, a coefficient is trunk-side") was a
   just-so story, and it should not be reused on the next lever without this measurement.
5. ⚠️ **THE BOX WAS CPU-SATURATED THROUGHOUT** (load 34–50 on 16 cores, 0 % idle, a live training
   arm plus a concurrent offline pipeline). Extraction ran at `nice 15`, two threads, three lanes,
   `CUDA_VISIBLE_DEVICES=""`; wall time was ~16–27 min per frame against the ~15 min the N-curve's
   three-tree extraction took per substrate on a quieter box. **No number here is a timing
   measurement** and none is used as one.

6. 🚨 **HAZARD 2 BIT ME WHILE I WAS WRITING HAZARD 2.** The first draft of §4a's `V t11–24` column
   was hand-transcribed from [`frame_check.log`](frame_check.log) and every one of its nine cells
   was wrong — I had read `ctrl10M_c`'s rows as `ctrl10M`'s, because the log labels rows by draw
   and not by run. It was caught by a programmatic re-check of the committed README against
   `frame_check.json` before the commit, and the corrected column is above. **No number in a
   measurement README should be typed by hand when a JSON holds it** — §4b/4c/4d and
   [`TABLE.md`](TABLE.md) are generated by `tabulate.py` and were correct; the one hand-built table
   was the one that broke. The verdict and every delta were unaffected (they are computed, not
   typed), which is the only reason this is a hazard and not a retraction.

---

## 8. The next increments, cheapest first

- **One line in `frame_check.py`**: label the progress print with the `--dir` NAME rather than
  `basename(dir)` (hazard 2). Trivial, and it removes a real misreading risk from a log that gets
  pasted into entries.
- **A within-draw battle-clustered bootstrap on `pooled → class`** (§6.3). `GroupedRidgeCV` already
  returns out-of-fold predictions, so a percentile bootstrap over battles is arithmetic on arrays
  already in memory — minutes, no re-forward. It would turn this row from "clears a floor" into
  "clears a floor with a CI", the standard the V row already meets.
- **The COMMON-STATE version** (§6.8, the confound that actually bounds the claim): forward all
  five checkpoints over ONE run's eval tree, so every decode reads the same states. ~25 CPU-min per
  checkpoint and it separates "encodes the class better" from "visits more separable states".
- **`vf15_b` and `strata_b` at the representation**, if and when those runs are read — the
  decomposition in §4c is one run per arm and the Δpooled/ΔV ratio (0.97/1.27 vs 0.59/0.50) is the
  quantity a replicate would confirm.
- 🚫 **NOT worth building: a representation-level bar.** §2 and hazard 3 say the row is noisier and
  ~200× more expensive per read than the row it would replace. It is a mechanism instrument, not a
  gate.

---

## 9. Files

| file | holds |
|---|---|
| [`PREDICTION.md`](PREDICTION.md) | the two branches, the "moves with" bar, the mixed-outcome and draw-disagreement rules, my (wrong) prior — committed at `be6a19a0`, before any extraction ran |
| [`run.sh`](run.sh) | the whole pipeline, idempotent; reuses `../winprob_refit_ncurve_2026-09-10/{extract,frame_check}.py` UNMODIFIED |
| [`tabulate.py`](tabulate.py) | PREDICTION §3 applied in code: floors, deltas, between-draw spread, the branch verdict |
| [`frame_check.json`](frame_check.json) | every point, all five buckets, all nine frames |
| [`branch_test.json`](branch_test.json) | floors, deltas, ×floor, per-draw and aggregate verdicts |
| [`TABLE.md`](TABLE.md) | the generated table |
| [`frame_check.log`](frame_check.log) | the run log (see hazard 2 on its labels) |
| `extract_meta_<run>_<draw>.json` | per-frame provenance: snapshot md5, manifest seed and `checkpoint_sha`, per-opponent capture rates, draws excluded, the frozen-forward QC |

The 128-dim tensors (2.8 GB) are NOT committed; `run.sh` regenerates them in ~25 CPU-min per frame.

---

## 10. Ledger paragraph — ready to append (this file does NOT edit `ledger.md`)

> ### 2026-09-11 · READ · the REPRESENTATION-level opponent-class decode lands: **BRANCH A** — `strata` (+0.048 / +0.064) and `vf15` (−0.052 / −0.045) BOTH move `pooled → class` at t4–10 past the run-level floor on both eval draws, with the same signs as their V-level deltas, so the two rows are REDUNDANT and `cond.opp_class_auc.t4_10` REMAINS the decision row; the two DOWNGRADED sentences are RESTORED, `strata`'s as an understatement (Δpooled/ΔV = 0.97 and 1.27; at t1–3 the representation gains nearly TWICE what V shows) and `vf15`'s as half an account (Δpooled/ΔV = 0.59 and 0.50 at t4–10, and NOTHING at t1–3 while V still falls)
>
> `measurements/repr_class_decode_2026-09-11/`; `winprob_refit_ncurve_2026-09-10/{extract,frame_check}.py` run UNMODIFIED over `value_pooled` (the win head's literal input) for `strata`, `vf15`, `ctrl10M`, `ctrl10M_b` and `ctrl10M_c` at `step_10000032`, on both offline 800-game draws (seed 20260910 for all five; seed 20260911 for all but `ctrl10M_c`, which has no such tree). Registered in `PREDICTION.md` before any number existed, with the branch, the bar (`sign(Δpooled) == sign(ΔV)` AND `|Δpooled| >` the widest control-pair difference on that draw), the mixed-outcome rule (mixed ⇒ branch B) and the draw-disagreement rule all fixed; applied in code by `tabulate.py`. **Floors** (rule 19, no CI): pooled 0.0232 (hp800, two pairs) / 0.0310 (hp800b, ONE pair); V 0.0197 / 0.0234. **Both arms MOVE WITH on both draws** — `strata` at 2.09× / 2.07× the pooled floor against 2.53× / 2.16× the V floor, `vf15` at 2.23× / 1.45× against 4.46× / 3.82×. **(1) The restored sentences.** `strata`'s "the representation carries more class information, not only the head" is CONFIRMED and was too weak: Δpooled/ΔV is 0.97 and 1.27 at t4–10, and at t1–3 the representation moves +0.076 / +0.087 against V's +0.043 / +0.039 — the head cashes in about half of what the trunk gains early. `vf15`'s "raising the value coefficient REMOVED opponent information from the shared representation" is CONFIRMED at t4–10 and t11–24 but at HALF the V effect (0.59, 0.50), and at t1–3 the representation is unmoved (−0.012 / −0.004, inside every floor) while V falls (−0.020 / −0.028) — so roughly half of `vf15`'s V-level damage is the head making worse use of an only-partly-degraded tensor, a split the V row cannot make. **(2) The decision row is reinforced, not merely survived.** The representation row's run-level floor is 15–35 % WIDER than the V row's, its between-draw spread on `strata` is 5–10× larger, and it costs a ~25-CPU-minute forward pass and a 300 MB tensor per frame against a `critic_read` invocation. Noisier, slower, same story. `strata_b`'s V-level verdict needs no amendment and a PASS there may carry the representation reading **for `strata` specifically**, since on that lever the two rows are one statement; `vf15` shows the licence does not generalise. **(3) A free cross-instrument validation:** `frame_check.py`'s V column, a different fitter on an unmatched frame, reproduces tool-v6 `cond.opp_class_auc.t4_10` to ±0.005 on all five checkpoints (0.7613/0.760, 0.7114/0.710, 0.7052/0.707, 0.6917/0.688, 0.6235/0.628) — stronger evidence for the row's validity than either instrument's own CI. **Frame QC, all nine frames:** own-team → class leak **0.492–0.498** (the N-curve's hazard-1 confound is absent), frozen-forward max |V_fwd − V_rec| **1.0e-06 to 2.5e-06**, battles 9,536–9,563, 602 teams each. **Hazards:** a ONE-pair floor (hp800b) printed ×floor values of 8.6–14.7 on the non-decision buckets and those are luck, not precision; `frame_check.py` labels its progress rows with `basename(dir)` = the DRAW, so nine frames all print as `[hp800]`/`[hp800b]` (JSON keys are correct; the script was left unmodified because another agent was running it concurrently); every decode is on that run's OWN trajectories, so the claim is "the representation on the states this policy reaches", and the clean common-state version is named as the next increment; **the pre-registered prior was WRONG on both arms in opposite directions** and is kept unedited. **Standing:** the representation row is a MECHANISM instrument, not a gate — no bar is promoted to it. Tag: **MEASURED · BRANCH A · claims RESTORED · decision row UNCHANGED**.
