# `strata_b` AT THE REPRESENTATION — do the two `strata` seeds differ in the TRUNK or at the HEAD?

Run 2026-09-12, offline, CPU only, nothing written under `models/`. Reproduce with [`run.sh`](run.sh).
The two branches, the bar, the mixed rules and the declared power limit were registered in
[`PREDICTION.md`](PREDICTION.md) **before any `pooled → class` number existed for
`ai_v12_24_ladder_strata_b`** (committed separately, at `c6cb3178`), and are applied **in code** by
[`tabulate.py`](tabulate.py).

---

## VERDICT

**BRANCH TRUNK, on both eval draws.** The two seeds of the class-balanced-BCE lever differ in the
tensor the win head reads, not only in the scalar it emits.

| draw | `strata_b` − `strata`, **pooled** | ×floor | `strata_b` − `strata`, **V** | ×floor | branch |
|---|---|---|---|---|---|
| `hp800` | **−0.0419** | **1.81** | −0.0820 | 4.16 | **TRUNK** |
| `hp800b` | **−0.0484** | **1.56** | −0.0843 | 3.60 | **TRUNK** |

My registered prior (TRUNK at ≈ 70 %) was **right**, and the margin is the interesting part.

**Three findings, in order of consequence.**

**1 — `strata`'s REPRESENTATION-level result does not replicate either, and the lever is now NOT
CONFIRMED at both levels.** `07d42a51` recorded `strata`'s `pooled → class` at t4–10 as **+0.048 /
+0.064** above `ctrl10M`, past the run-level floor on both draws, and restored the campaign sentence
*"the representation carries more class information, not only the head"* as an understatement. Its
seed replicate reads **+0.0065 / +0.0159** — **0.28× and 0.51× the floor, NOT DETECTED**. The
replicate gives back **87 % / 75 %** of `strata`'s pooled gain. The within-lever spread at the
representation is **0.042 / 0.048**, i.e. **1.8× / 1.6× the control-replicate floor**, exactly the
shape the V-level verdict of 2026-09-12 found at 3.6×. **One run of the lever moved the
representation; its seed replicate did not.**

**2 — `strata_b`'s representation sits AT the controls, not below them; only its HEAD is below.**
This is the question the increment was commissioned to answer and the answer is not the symmetric
one.

| | `strata` | `strata_b` | reading |
|---|---|---|---|
| pooled → class t4–10, Δ vs `ctrl10M` | +0.0484 / +0.0643 (2.09× / 2.07× floor) | **+0.0065 / +0.0159 (0.28× / 0.51×)** | `strata` ABOVE the control band; `strata_b` **INSIDE** it |
| V → class t4–10, Δ vs `ctrl10M` | +0.0499 / +0.0505 (2.53× / 2.16×) | **−0.0321 / −0.0338 (1.63× / 1.44×)** | `strata` above; `strata_b` **BELOW** the control band |

So the seed reversal is a **REVERSION of the representation to the control level plus a head-side
deficit on top of it**. `strata_b` is handed a control-grade `value_pooled` and reads it worse than
a control does. That decomposition is precisely what the V row cannot emit, and it puts a level on
the dissociation the 2026-09-12 VERDICT found inside this run (`gate.resolution.all` at or above all
three controls while `cond.opp_class_auc.t4_10` sits below all three): **the class-decode deficit
against the controls is a HEAD deficit — the trunk is not below the controls at all.**

**3 — The within-lever seed difference is about HALF trunk, half head.** `Δ_B pooled / Δ_B V` is
**0.51 / 0.57** — the two seeds differ twice as much at V as at the representation. That ratio is
`vf15`'s (0.59 / 0.50 — `07d42a51` §4c), not `strata`'s own arm-vs-control ratio (0.97 / 1.27,
same source). The lever's
*arm-vs-control* effect on run `ai_v12_17` was one-for-one; the *seed-to-seed* difference between
two runs of that lever is only half carried by the trunk. **Those are different quantities and the
campaign should not have expected them to match** — the second is a statement about training-run
variance, the first about a lever.

**What this does NOT change.** The redundancy verdict of `07d42a51` (branch A — the two rows move
together in sign, the V row is cheaper and tighter, `cond.opp_class_auc.t4_10` stays the decision
row) **survives on a third checkpoint of this lever**, and nothing here promotes the representation
row to a bar. And the V-level verdict of 2026-09-12 (`strata` NOT CONFIRMED) is untouched: this
increment neither rescues it nor worsens it — it says where in the network the failure to replicate
lives.

---

## 1. Why this measurement exists

The 2026-09-12 VERDICT entry names it, in consequence (1):

> `07d42a51`'s licence — "a `strata_b` PASS may carry the representation reading for `strata`
> specifically" — is not exercised; the representation finding (Δpooled/ΔV 0.97–1.27,
> understatement) stands for run `ai_v12_17` alone. **The cheap next increment is named: the same
> `pooled → class` decode on `strata_b`'s two frames (~50 CPU-min), which says whether the two
> `strata` runs differ in the TRUNK or only at the head.**

`07d42a51` §8 names the same thing (*"`vf15_b` and `strata_b` at the representation, if and when
those runs are read"*). This is that increment, with the branch registered in advance.

---

## 2. The frame

**`ai_v12_24_ladder_strata_b@step_10000032`**, read on two independent offline `eval_trace_gen`
draws, against the NINE frames `07d42a51` already computed, **reused verbatim**.

| draw | tree | seed | provenance |
|---|---|---|---|
| `hp800` | `.../hp_eval/hp800/ai_v12_24_ladder_strata_b/eval_traces/step_10000032` | 20260910 | ALREADY EXISTED — generated by `hp_eval/hp_strata_b.sh` for the V-level verdict; the same draw the controls' `hp800` frames used |
| `hp800b` | `.../hp_eval/hp800b/ai_v12_24_ladder_strata_b/eval_traces/step_10000032` | 20260911 | **GENERATED FOR THIS MEASUREMENT**, with `hp800b.sh`'s invocation token for token (`--games 800 --sentinels 3 --workers 4 --concurrency 1 --nice 15 --seed 20260911 --shard-games 25`) |

**The pair is genuine.** `strata` and `strata_b` both ran pinned to **`f871e79f`**
(`metadata.json`'s `pin_history`), both at `--win-prob-strata-weight 1.0` and `--vf-coef 0.5`, argv
identical but for `--seed 1002` and the run name. Both record `eval_sentinel_greedy: true`, as do
all three controls and `vf15` — the 2026-09-07 opponent-regime boundary does not cut through this
comparison. Each run's three sentinels are its own pool snapshots at steps 4000032 / 6000000 /
8000016, the same three steps for every run in the set.

**Three controls, not one.** `ctrl10M`, `ctrl10M_b` and `ctrl10M_c` are three runs of the same
configuration, so |control − control| IS the run-level component. Per rule 19 a two- or three-point
spread **BOUNDS** a floor, carries no CI, demotes and never promotes. `ctrl10M_c` has **no `hp800b`
tree**, so draw B's floor is a SINGLE difference — stated in every floor column, never absorbed.

## 3. The instrument

`extract.py` forwards the checkpoint over its eval tree and writes `value_pooled` — **the win head's
literal 128-dim input** — beside the recorded win probability. `frame_check.py` then fits, **out of
fold and grouped by battle** (`GroupedRidgeCV`), two decoders of the same binary label ("is this
opponent a sentinel?") on the **same rows, the same folds and the same subsample**:
`pooled_to_opp_class_AUC` (the representation) and `V_to_opp_class_AUC` (the head's scalar output).
Both scripts were run **UNMODIFIED** from `../winprob_refit_ncurve_2026-09-10/`, exactly as
`../repr_class_decode_2026-09-11/run.sh` ran them, with the same `--cap 20000 --seed 20260910`.

**The nine reused frames are not recomputed.** `frame_check.py` treats each extraction dir
independently, so [`run.sh`](run.sh) runs it on the two new dirs only and merges the result into
`07d42a51`'s committed `frame_check.json`; the merge step asserts no key collision. Every
`strata` / `vf15` / control number below is byte-identical to the one in that file.

🚨 **This is NOT `cond.opp_class_auc.t4_10`.** That row comes from `main.ops.conditioning_meters`
via `critic_read` on a quota-matched frame with its own bootstrap CI. The `V_to_opp_class_AUC`
column here is the same *quantity* from a different instrument on an unmatched frame, and it is the
**internal comparator for the pooled row**. See §4f for how closely the two agree on this
checkpoint.

---

## 4. THE TABLE

Machine-readable in [`frame_check.json`](frame_check.json) (all 11 frames) and
[`branch_test.json`](branch_test.json); the generated markdown is [`TABLE.md`](TABLE.md). **Every
number in this README was copied from those files and then re-checked against them programmatically
by [`check_readme.py`](check_readme.py)** — `07d42a51`'s hazard 6 was a hand-typed column.

### 4a. Points — opponent-class decode AUC, out of fold, grouped by battle

| run | draw | pooled t1–3 | **pooled t4–10** | pooled t11–24 | V t1–3 | **V t4–10** | V t11–24 |
|---|---|---|---|---|---|---|---|
| `ctrl10M` | hp800 | 0.6840 | **0.8302** | 0.8029 | 0.5803 | **0.7114** | 0.7145 |
| `ctrl10M` | hp800b | 0.6673 | **0.8244** | 0.7982 | 0.5827 | **0.7098** | 0.7159 |
| `ctrl10M_b` | hp800 | 0.7067 | **0.8534** | 0.8215 | 0.5954 | **0.7052** | 0.7139 |
| `ctrl10M_b` | hp800b | 0.7120 | **0.8554** | 0.8216 | 0.5794 | **0.6864** | 0.7084 |
| `ctrl10M_c` | hp800 | 0.6857 | **0.8217** | 0.8009 | 0.5672 | **0.6917** | 0.7050 |
| `strata` | hp800 | 0.7600 | **0.8786** | 0.8202 | 0.6232 | **0.7613** | 0.7352 |
| `strata` | hp800b | 0.7542 | **0.8887** | 0.8130 | 0.6212 | **0.7603** | 0.7278 |
| **`strata_b`** | hp800 | 0.7175 | **0.8367** | 0.8047 | 0.5768 | **0.6793** | 0.7021 |
| **`strata_b`** | hp800b | 0.7173 | **0.8403** | 0.8073 | 0.5740 | **0.6760** | 0.6959 |
| `vf15` | hp800 | 0.6719 | **0.7784** | 0.7335 | 0.5605 | **0.6235** | 0.6113 |
| `vf15` | hp800b | 0.6635 | **0.7795** | 0.7377 | 0.5544 | **0.6204** | 0.6058 |

### 4b. THE BRANCH TEST — axis B, the within-lever contrast (PREDICTION §3, §5)

`FLOOR` = the widest |control − `ctrl10M`| on that row and draw. **hp800** has two pairs
(`ctrl10M_b` +0.0232 / `ctrl10M_c` −0.0085 on pooled; −0.0062 / −0.0197 on V); **hp800b** has ONE
(`ctrl10M_b` +0.0310 on pooled, −0.0234 on V) and is correspondingly weaker.

| draw | Δ pooled (`strata_b`−`strata`) | FLOOR | ×floor | Δ V | FLOOR | ×floor | Δpooled/ΔV | gives back | **BRANCH** |
|---|---|---|---|---|---|---|---|---|---|
| `hp800` | **−0.0419** | 0.0232 (2 pairs) | **1.81** | −0.0820 | 0.0197 (2) | 4.16 | **0.51** | **0.866** | **TRUNK** |
| `hp800b` | **−0.0484** | 0.0310 (1 pair) | **1.56** | −0.0843 | 0.0234 (1) | 3.60 | **0.57** | **0.753** | **TRUNK** |

"gives back" = `[q_pooled(strata) − q_pooled(strata_b)] / [q_pooled(strata) − q_pooled(ctrl10M)]` —
the fraction of `strata`'s pooled gain over the control that the replicate does not have.

### 4c. Axis A — every run against `ctrl10M` at the decision bucket (REPORTED, not the branch test)

Per PREDICTION §5 this axis is **under-powered against a move the size of `strata_b`'s V deficit**
and is reported with that stated, not used to decide the branch.

| run | draw | Δ pooled | ×floor | Δ V | ×floor |
|---|---|---|---|---|---|
| `ctrl10M_b` | hp800 | +0.0232 | 1.00 | −0.0062 | 0.31 |
| `ctrl10M_b` | hp800b | +0.0310 | 1.00 | −0.0234 | 1.00 |
| `ctrl10M_c` | hp800 | −0.0085 | 0.37 | −0.0197 | 1.00 |
| `strata` | hp800 | +0.0484 | 2.09 | +0.0499 | 2.53 |
| `strata` | hp800b | +0.0643 | 2.07 | +0.0505 | 2.16 |
| **`strata_b`** | hp800 | **+0.0065** | **0.28** | **−0.0321** | **1.63** |
| **`strata_b`** | hp800b | **+0.0159** | **0.51** | **−0.0338** | **1.44** |
| `vf15` | hp800 | −0.0518 | 2.23 | −0.0879 | 4.46 |
| `vf15` | hp800b | −0.0449 | 1.45 | −0.0894 | 3.82 |

### 4d. The non-decision buckets (§3 rule 4 — reported, decide nothing)

| bucket | draw | Δ pooled (`strata_b`−`strata`) | FLOOR | Δ pooled (`strata_b`−`ctrl10M`) | FLOOR |
|---|---|---|---|---|---|
| t1–3 | hp800 | −0.0425 | 0.0227 | **+0.0335** | 0.0227 |
| t1–3 | hp800b | −0.0369 | 0.0447 | **+0.0500** | 0.0447 |
| t11–24 | hp800 | −0.0155 | 0.0186 | +0.0018 | 0.0186 |
| t11–24 | hp800b | −0.0057 | 0.0234 | +0.0091 | 0.0234 |

⚠️ **The reversion is not uniform across the battle, and at t1–3 it is incomplete.** `strata_b`'s
representation sits **ABOVE** `ctrl10M` at t1–3 on both draws (+0.034 at 1.48× the floor, +0.050 at
1.12×) while its V sits at or just below the control there (−0.0035 / −0.0087). So `strata_b` keeps
part of the lever's *early*-window representation enrichment and none of its mid-window one. At
t11–24 both contrasts are inside the floor. **This decides nothing** — t4–10 was fixed in advance
and t1–3's hp800b floor is a single 0.0447 control difference — but it is the one place the two
seeds' trunks are not simply "lever" and "control".

### 4e. Frame QC — every frame passed every refusal in PREDICTION §6

| run | draw | battles | teams | **own-team → class leak** | **max \|V_fwd − V_rec\|** | true WR |
|---|---|---|---|---|---|---|
| `ctrl10M` | hp800 / hp800b | 9,557 / 9,557 | 602 | **0.4949 / 0.4958** | 1.76e-06 / 1.85e-06 | 0.8596 / 0.8625 |
| `ctrl10M_b` | hp800 / hp800b | 9,563 / 9,542 | 602 | **0.4933 / 0.4966** | 1.25e-06 / 1.49e-06 | 0.8585 / 0.8517 |
| `ctrl10M_c` | hp800 | 9,546 | 602 | **0.4938** | 1.34e-06 | 0.8504 |
| `strata` | hp800 / hp800b | 9,540 / 9,536 | 602 | **0.4923 / 0.4984** | 1.43e-06 / 1.01e-06 | 0.8531 / 0.8452 |
| **`strata_b`** | hp800 / hp800b | **9,566 / 9,567** | 602 | **0.4950 / 0.4964** | **2.32e-06 / 5.01e-06** | **0.8349 / 0.8343** |
| `vf15` | hp800 / hp800b | 9,537 / 9,559 | 602 | **0.4962 / 0.4955** | 2.32e-06 / 2.50e-06 | 0.8735 / 0.8688 |

- **Own-team leak 0.4950 / 0.4964** on the two new frames — at the null, inside the 0.492–0.498 band
  the nine reused frames occupy. The N-curve's hazard-1 confound (a turn-1 "opponent decode" that is
  really an own-team decode at 0.86) is absent. 584 / 585 of 602 teams face a sentinel.
- **Frozen-forward QC 2.32e-06 / 5.01e-06.** The hp800b value is the **largest in the eleven-frame
  set**, about twice the previous maximum, and still **200× below the 1e-3 refusal**. Recorded, not
  a refusal; see hazard 3.
- **Battles 9,566 / 9,567 against the set's 9,536–9,567** — a 0.3 % spread, so both sides of every
  contrast have matched precision with no post-hoc trimming. Capture rate 1.0 on every opponent,
  both frames (full capture, no quota); 34 / 33 draws excluded.
- Same snapshot on both draws (`snapshot_md5 29f3363e…`, `checkpoint_sha dfdd624d49bb78ad`).
- **Between-draw spread on `strata_b` is the TIGHTEST in the set**: +0.0036 on pooled t4–10 and
  −0.0033 on V t4–10, against `strata`'s +0.0101 / −0.0010 and `ctrl10M_b`'s +0.0020 / −0.0188. The
  eval-draw component is far smaller than the run-level floor, the campaign's expected ordering.

### 4f. Cross-instrument check — a sixth checkpoint

`frame_check.py`'s `V_to_opp_class_AUC` at t4–10 on the `hp800` tree reads **0.6793**. `critic_read`
v6's `cond.opp_class_auc.t4_10` for the arm, on the SAME tree, reads **0.6792** in the `ctrl10M`
pair — agreement to **1e-4**. Two honest caveats: v6 quota-matches per pair, so the same arm prints
**0.6731 / 0.6792 / 0.6804** across its three control pairs, and the 1e-4 agreement is with one of
those three; the fair statement is that this instrument lands **inside the 0.673–0.680 band the
quota-matched pairs span**. That extends `07d42a51`'s ±0.005 five-checkpoint agreement to a sixth
and is free.

---

## 5. The reading

**The question was whether the two `strata` seeds differ in the trunk or at the head. They differ in
the trunk — and also at the head, and the two halves point in the same direction and are about
equal.**

Read the two contrasts together. Against its seed twin, `strata_b`'s representation is down
**−0.042 / −0.048**, past the floor on both draws: the trunk moved. Against the controls, that same
representation is **+0.007 / +0.016**, inside the floor: the trunk moved *back to where a control
sits*, not past it. At V the picture is one notch worse — `strata_b` is **−0.032 / −0.034** below
`ctrl10M`, outside the floor on both draws. So the two seeds' V-level gap (**−0.082 / −0.084**) is
about half mirrored at the representation (**−0.042 / −0.048**, `Δpooled/ΔV` = **0.51 / 0.57**) and
about half not — and the half that is not is what carries `strata_b` past the control band at V
while its representation stays inside it. (The two AUCs come from different decoders, so the halves
are a ratio, not an additive decomposition.)

**What that costs the lever.** `strata`'s representation claim was the strongest surviving sentence
in the campaign about `--win-prob-strata-weight` — `07d42a51` called it an understatement. It is now
a **one-run** statement with a seed replicate inside the floor, the same fate as its V-level twin,
and the honest family-level summary is: *one run of the class-balanced-BCE lever read +0.048 / +0.064
at the representation and +0.050 / +0.051 at V; its seed replicate read +0.007 / +0.016 and
−0.032 / −0.034. The lever is NOT CONFIRMED at either level, and its within-lever run-to-run spread
is 1.6–1.8× the control floor on the representation row and 3.6–4.2× on the V row.* Note that the
spread exceeds the control floor **more** at V than at the representation on both draws — whatever
is making these two runs differ shows up larger in the scalar.

**What it buys the ladder.** Two things the V row could not say.

1. **A level for the within-run dissociation.** The 2026-09-12 VERDICT found `strata_b` resolving
   outcomes at least as well as every control (`gate.resolution.all` +0.009–0.010, each CI clear of
   zero) while decoding opponent class worse than every control. The class half of that dissociation
   is now localised: **it is not in `value_pooled`.** `strata_b`'s trunk carries control-grade class
   information; its win head does not express it. A "the run is broken at the trunk" account of
   `strata_b` is excluded by this row.
2. **A calibration of what the representation row adds.** On an arm-vs-control contrast for this
   lever the two rows were one statement (Δpooled/ΔV = 0.97 / 1.27). On a seed-vs-seed contrast for
   the same lever they are not (0.51 / 0.57). **Run-to-run variance is more head-concentrated than
   the lever's effect was** — which is a reason to keep reading the cheaper V row as the decision
   row (it is where the variance lives, so it is the row a floor must bound) and a reason not to
   generalise any Δpooled/ΔV ratio from one contrast type to another.

**What it does not settle.** Nothing here says which of the two `strata` runs is "the lever". n = 2
cannot separate the 2026-09-12 VERDICT's two readings — (a) class-balanced BCE inflates run-to-run
variance on this row, or (b) one of the two runs is atypical — and a representation read does not
help, because it reproduces the same ambiguity one layer down. Nothing here touches leaf quality or
strength; the L1/L2 battery is that test. And nothing here promotes `pooled → class` to a bar: it
has no within-draw CI, it is the noisier row, and `07d42a51`'s "🚫 NOT worth building: a
representation-level bar" stands.

---

## 6. Confounds and limits

1. **A one- or two-point floor has no CI and can be small or large by luck.** The decisive contrast
   clears a TWO-pair floor at 1.81× on `hp800` and a ONE-pair floor at 1.56× on `hp800b`. The second
   rests on a single control difference. Both are the campaign's standard, which is weaker than a CI.
2. 🚨 **THE DECODE IS ON EACH RUN'S OWN TRAJECTORIES**, and here that bites harder than it did in
   `07d42a51`. `strata_b`'s overall true win rate is **0.8349 / 0.8343** — the **lowest of the six
   checkpoints**, 2.5 pp below `ctrl10M` (0.8596 / 0.8625) and 1.1–1.8 pp below `strata`. So
   "`strata_b`'s representation reverted to control level" is entangled with "`strata_b` visits a
   different, slightly weaker-policy state distribution". The direction of that bias is **not
   known** — a weaker policy could plausibly reach more or less class-separable states — so it is
   stated as an open confound, not signed. **This applies to the V row identically**, which is why
   it cannot flip the TRUNK-vs-HEAD comparison (both sides are contaminated the same way), but it
   does bound what "representation" means: the representation *on the states this policy reaches*.
   The common-state version (forward all six checkpoints over ONE run's tree) remains the clean
   form and is still unbuilt — §7's first increment.
3. **The decoder is a ridge — a LINEAR read of a 128-dim tensor.** A flat pooled row is weaker
   evidence of "the representation did not change" than a moved pooled row is of "it did". The
   **branch** here rests on a MOVED row (axis B) and is not exposed to that bias; the **axis-A**
   statement "`strata_b`'s representation is at the controls" rests on a flat row and is
   correspondingly weaker. Read finding 2 with that asymmetry in mind.
4. **`value_pooled` is the win head's input, not the shared trunk.** It sits at the end of the value
   path. A difference in the shared trunk that the value path undoes would read flat here; one that
   lives only in the value path's last layers reads moved. "TRUNK" in this file means *the tensor
   the head is handed*, which is the quantity the research question ("is the head handed the
   answer?") names — not literally the shared body.
5. **No within-draw CI on the pooled row.** `frame_check.py` emits a point AUC per bucket. Precision
   is bounded by the between-draw spread (§4e) and the control-pair floor, not by an interval. The
   battle-clustered bootstrap `07d42a51` §8 named is still unbuilt.
6. **Unmatched quota.** `critic_read` quota-matches before differencing; `frame_check.py` does not.
   Every frame here is a full-capture 800-game tree with 602 teams and ~9,550 battles, so exposure
   is matched by construction rather than by trimming — and the counts are printed (§4e).
7. **One seed replicate.** This is n = 2 for the lever at the representation, the same n the V-level
   verdict had. A third `strata` run would say which of the two is atypical; nothing cheaper will.
8. **t1–3 and t11–24 decide nothing** (§3 rule 4, fixed in advance). §4d reports them because the
   t1–3 picture is genuinely non-uniform, not because it is being read.

---

## 7. Hazards hit (a hazard is a finding)

1. ⚠️ **THE QUESTION AS ASKED HAD A FALSE DICHOTOMY IN IT, AND THE PRE-REGISTRATION CAUGHT IT.** The
   commissioned question was "does `strata_b`'s representation ALSO sit below the controls, or is it
   like `strata`'s?" — two options. The answer is **neither**: it sits *between* them, at the
   controls. PREDICTION §3 had anticipated exactly one intermediate (`PARTIAL-TRUNK`, for a
   representation that moved partway but stayed above the control band) and defined `TRUNK` as
   "separated from `strata` downward AND not above the controls", which is what fired. **Had the
   branch been "below the controls = trunk, at `strata` = head", this result would have had no
   branch to land in.** A registration that only enumerates the two hypotheses in the commissioning
   sentence is a registration with a hole in it.
2. ⚠️ **A PRE-DECLARED POWER LIMIT CHANGED WHICH CONTRAST DECIDED.** PREDICTION §5 noted before the
   data that `FLOOR_pooled` (0.023 / 0.031) is comparable to `strata_b`'s V-level deficit against
   the controls (`critic_read` v6's −0.0294 / −0.0338, the figures PREDICTION §5 quoted from the
   2026-09-12 VERDICT entry), so an arm-vs-control representation move of that size could read
   "within floor" while being real — and fixed the branch on the within-lever contrast instead,
   where the expected separation is 2.6–3.4× the floor. That was the right call: axis A came in at
   **0.28× and 0.51× the floor** and would have been uninformative, while axis B came in at 1.81×
   and 1.56× and decided cleanly. **Choosing the powered contrast AFTER seeing 0.28× would have been
   unreportable.**
3. ⚠️ **THE hp800b FROZEN-FORWARD QC IS THE LARGEST IN THE ELEVEN-FRAME SET** — `5.01e-06` against a
   previous maximum of `2.50e-06`. It is 200× below the 1e-3 refusal and nothing is being refused,
   but it is recorded here rather than buried in the QC table, because a QC statistic that doubles
   is the kind of thing a later frame will want a baseline for.
4. ⚠️ **`frame_check.py` STILL LABELS ITS PROGRESS ROWS WITH `basename(dir)` = THE DRAW, NOT THE
   RUN** (`07d42a51` hazard 2, unfixed). Both new frames print as `[hp800]` / `[hp800b]` and the
   stdout alone cannot say which checkpoint a row belongs to. The script was left **UNMODIFIED**
   again, deliberately: it is the shared instrument `../repr_class_decode_2026-09-11/` and the
   N-curve both run, another agent may be running it concurrently, and modifying it would break the
   "run unmodified" property that makes the nine reused rows comparable. The one-line fix stays on
   the increment list.
5. ⚠️ **THE BOX CARRIED THE LIVE `ai_v12_25_ladder_vf15_b` TRAINING ARM THROUGHOUT** (launched 02:09
   PT; load average 12–26 on 16 cores). Everything here ran CPU-only at `nice 15` with
   `CUDA_VISIBLE_DEVICES=""`; the `hp800b` trace generation took **64 min** at 4 workers (02:32:25 -> 03:36:48 PT) and the
   two extractions **223 s** and **159 s**. **No number in this file is a timing measurement** and none is used
   as one.
6. ✅ **The hand-typed-number hazard did not recur.** `07d42a51`'s hazard 6 was a README column
   transcribed by hand, every cell wrong. [`check_readme.py`](check_readme.py) re-checks every
   2-to-4-decimal number in this file against `frame_check.json` and `branch_test.json` before the
   commit; its output is recorded in the commit message. The generated [`TABLE.md`](TABLE.md) and
   [`branch_test.json`](branch_test.json) remain the sources.

---

## 8. The next increments, cheapest first

- **THE COMMON-STATE VERSION** (§6.2), now the binding limit rather than a nicety. `strata_b`'s true
  WR is 2.5 pp below `ctrl10M`'s, the widest state-distribution gap in the set, and the headline
  finding ("its representation is at control level") is exactly the kind of statement that gap can
  manufacture. Forward all six checkpoints over ONE run's eval tree: ~25 CPU-min per checkpoint,
  ~2.5 CPU-h total, no new eval generation.
- **`vf15_b` at the representation**, when `ai_v12_25_ladder_vf15_b` finishes. It is the other
  registered replicate and the only remaining test of whether `vf15`'s representation-level result
  (−0.052 / −0.045, Δpooled/ΔV 0.59 / 0.50) replicates under a seed. Its `hp800` and `hp800b` draws
  do not exist yet; ~70 min of generation each plus ~4 CPU-min of extraction.
- **A within-draw battle-clustered bootstrap on `pooled → class`** (§6.5, `07d42a51` §8). Arithmetic
  on arrays `GroupedRidgeCV` already returns; minutes, no re-forward. It would give the decisive
  axis-B contrast a CI instead of a floor.
- **One line in `frame_check.py`** (hazard 4), once no one is running it concurrently.
- 🚫 **STILL NOT worth building: a representation-level bar.** Unchanged from `07d42a51`. This
  measurement adds a reason — on the seed-vs-seed contrast the variance is *more* concentrated in V
  (Δpooled/ΔV ≈ 0.5), so V is where a floor needs to bound it.

---

## 9. Files

| file | holds |
|---|---|
| [`PREDICTION.md`](PREDICTION.md) | the TRUNK / HEAD branches, the bar, the PARTIAL-TRUNK / draw-disagreement / VOID rules, the declared power limit, my prior — committed at `c6cb3178`, before any extraction ran |
| [`run.sh`](run.sh) | the whole pipeline, idempotent: the `hp800b` generation, both extractions, `frame_check.py` on the two NEW dirs only, the merge onto the nine reused frames, the branch test. Reuses `../winprob_refit_ncurve_2026-09-10/{extract,frame_check}.py` **UNMODIFIED** |
| [`tabulate.py`](tabulate.py) | PREDICTION §3 applied in code: floors, both axes, between-draw spread, the branch verdict |
| [`frame_check.json`](frame_check.json) | all **11** frames — the 2 new ones plus the 9 reused verbatim from `../repr_class_decode_2026-09-11/frame_check.json` |
| [`branch_test.json`](branch_test.json) | floors, axis A, axis B, ×floor, per-draw and aggregate branch |
| [`TABLE.md`](TABLE.md) | the generated table |
| [`check_readme.py`](check_readme.py) | the anti-hazard-6 check: every number in this README must exist in the JSONs |
| `extract_meta_ai_v12_24_ladder_strata_b_<draw>.json` | per-frame provenance: snapshot md5, manifest seed and `checkpoint_sha`, the three sentinel steps, per-opponent capture rates and true win rates, draws excluded, the frozen-forward QC, wall seconds |

No stdout log is kept. `extract.py`'s and `frame_check.py`'s console output is fully reproduced by
the two `extract_meta_*.json` files and [`frame_check.json`](frame_check.json), and the wrapper that
ran the pipeline in the background truncated its own stdout to the last 60 lines — a partial log is
worse than none, and `07d42a51` hazard 2 is a warning about reading these logs at all.

The 128-dim tensors (~600 MB for the two frames) live under
`/home/goodlad/.claude/jobs/9ab51de6/tmp/repdecode_strata_b/` and are **NOT committed**;
`run.sh` regenerates them in ~4 CPU-min per frame from the eval trees.

---

## 10. Ledger paragraph — ready to append (this file does NOT edit `ledger.md`)

> ### 2026-09-12 · READ · `strata_b` at the REPRESENTATION: **BRANCH TRUNK on both draws** — the two `strata` seeds differ in the tensor the head reads (`pooled → class` t4–10 **−0.042 / −0.048**, 1.81× / 1.56× the control floor), so `strata`'s RESTORED representation claim is now ALSO a one-run statement that does NOT replicate; but `strata_b`'s representation sits **AT** the controls (**+0.007 / +0.016**, 0.28× / 0.51× floor, NOT DETECTED) while only its **V** sits BELOW them (−0.032 / −0.034, 1.63× / 1.44×) — **the class-decode deficit against the controls is a HEAD deficit, not a trunk one**, and the seed-to-seed difference is ~half trunk, ~half head (Δpooled/ΔV **0.51 / 0.57**) against the 0.97 / 1.27 the same lever's arm-vs-control contrast showed
>
> `measurements/repr_class_decode_strata_b_2026-09-12/`; `winprob_refit_ncurve_2026-09-10/{extract,frame_check}.py` run **UNMODIFIED** over `value_pooled` for `ai_v12_24_ladder_strata_b@step_10000032` on both offline 800-game draws — `hp800` (seed 20260910, the tree the V-level verdict already used) and `hp800b` (seed 20260911, **generated for this read** with `hp800b.sh`'s invocation token for token). The nine frames of `07d42a51` (`strata`, `vf15`, `ctrl10M`, `ctrl10M_b`, `ctrl10M_c`) are **REUSED VERBATIM** and not recomputed; `frame_check.py` treats each extraction dir independently and the merge asserts no key collision. Registered in `PREDICTION.md` at `c6cb3178` before any number existed, with the TRUNK / HEAD branches, the PARTIAL-TRUNK and draw-disagreement rules, and — decisively — **a declared power limit fixing the branch on the WITHIN-LEVER contrast** (`strata_b` − `strata`, expected separation 2.6–3.4× the floor) rather than on `strata_b` − `ctrl10M`, whose floor (0.023 / 0.031) is comparable to the effect it would have to resolve; applied in code by `tabulate.py`. **The read.** `strata_b` pooled t4–10 **0.8367 / 0.8403** against `strata`'s 0.8786 / 0.8887 and `ctrl10M`'s 0.8302 / 0.8244; V t4–10 **0.6793 / 0.6760** against `strata`'s 0.7613 / 0.7603 and `ctrl10M`'s 0.7114 / 0.7098. Floors (rule 19, no CI): pooled 0.0232 (hp800, two pairs) / 0.0310 (hp800b, ONE pair), V 0.0197 / 0.0234. The replicate **gives back 87 % / 75 %** of `strata`'s pooled gain over the control. **(1) The lever is now NOT CONFIRMED at BOTH levels.** `07d42a51` restored "the representation carries more class information, not only the head" as an understatement on the strength of `strata`'s +0.048 / +0.064; the seed replicate reads +0.007 / +0.016, inside the floor. The within-lever spread is 0.042 / 0.048 on the representation row (1.8× / 1.6× the control floor) against 0.082 / 0.084 on the V row (4.2× / 3.6×) — **the spread exceeds the floor by more at V than at the representation on both draws**. **(2) The 2026-09-12 within-run dissociation now has a LEVEL.** `strata_b` resolves outcomes at least as well as every control while decoding opponent class worse than every control; this read localises the class half: **`value_pooled` is at control grade and the win head is what is below it.** A "broken at the trunk" account of `strata_b` is excluded. **(3) A Δpooled/ΔV ratio does not transfer between contrast types.** For this lever, arm-vs-control was 0.97 / 1.27 (one-for-one, `07d42a51`) and seed-vs-seed is 0.51 / 0.57 — run-to-run variance is more head-concentrated than the lever's effect was, which is a fresh reason the cheaper **V row stays the decision row** (it is where the variance lives) and a reason not to reuse a ratio across contrast types. Branch A (the two rows are redundant, no representation bar) **survives on a third checkpoint of this lever**; nothing is promoted. **Frame QC, both new frames:** own-team → class leak **0.4950 / 0.4964** (inside the nine reused frames' 0.492–0.498 band; the N-curve's hazard-1 confound absent), frozen-forward max |V_fwd − V_rec| **2.32e-06 / 5.01e-06** (the hp800b value is the largest in the 11-frame set, still 200× below the 1e-3 refusal), battles **9,566 / 9,567** against the set's 9,536–9,567, 602 teams, capture 1.0 on every opponent, same snapshot md5 on both draws. Cross-instrument: this instrument's V t4–10 on `hp800` reads **0.6793** against `critic_read` v6's **0.6792** in the `ctrl10M` pair — inside the 0.673–0.680 band v6's three quota-matched pairs span; a sixth checkpoint for `07d42a51`'s ±0.005 agreement. **Hazards:** the commissioned question's two options (**below the controls** vs **like `strata`'s**) did not contain the answer — the registration's third case is what it landed in, and a registration that enumerates only the commissioning hypotheses has a hole in it; the pre-declared power limit is what made the read decidable, since the axis it displaced came in at 0.28× / 0.51× the floor and choosing the powered contrast afterwards would have been unreportable; **the decode is on each run's OWN trajectories and `strata_b`'s true WR (0.8349 / 0.8343) is the lowest in the set, 2.5 pp below `ctrl10M`** — the widest state-distribution gap yet differenced on this row, so the common-state version is promoted to the binding next increment; `frame_check.py` still labels progress rows by DRAW not RUN and was left unmodified on purpose; the box carried the live `vf15_b` arm throughout and nothing here is a timing measurement. Tag: **MEASURED · BRANCH TRUNK · `strata` NOT CONFIRMED at the representation either · the control-relative deficit is HEAD-side · decision row UNCHANGED**.
