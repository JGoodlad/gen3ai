# PREDICTION — the REPRESENTATION-level opponent-class decode, registered before any number exists

Written 2026-09-11, **before `extract.py` or `frame_check.py` had been run on any of the five
checkpoints in this measurement**. No `pooled_to_opp_class_AUC` for `strata`, `vf15`,
`ctrl10M_b` or `ctrl10M_c` existed anywhere when this file was written; the only prior
representation-level numbers in the campaign are the N-curve's `ctrl10M` 0.819 and `ctrl10M_b`
0.851 at t4–10, on a DIFFERENT (three-tree pooled, ~24k-battle) frame.

---

## 1. Why the measurement exists

The ladder's conditioning decision row is `cond.opp_class_auc.t4_10` and its companion reference
`cond.spread_ratio_optimal.*`. The 2026-09-11 CORRECTION established that **both are fitted from
`V` — the win head's scalar OUTPUT — not from `value_pooled`**, the 128-dim tensor the head
actually reads. Two campaign claims were downgraded on that basis:

- `strata` (`--win-prob-strata-weight 1.0`): "the decodable-part term rises with it — the
  **representation** carries more class information, not only the head" — V-level Δ
  **+0.050 [+0.036, +0.064]** (hp800 / seed 20260910) against a **0.0219** two-draw run-level floor;
  held at +0.033 and +0.044 on two further draws.
- `vf15` (`--vf-coef 1.5`): "raising the value-loss coefficient **REMOVED opponent information from
  the shared representation**" — V-level Δ **−0.082 [−0.096, −0.066]**, repeated at −0.080.

Neither sentence was ever measured at the representation. This measurement runs the row that would
measure it: `frame_check.py`'s `pooled_to_opp_class_AUC`, out-of-fold, battle-grouped, from
`extract.py`'s `value_pooled` — with `V_to_opp_class_AUC` computed **on the same rows, the same
folds and the same subsample** so the two are directly differenced rather than compared across
tools.

---

## 2. THE QUESTION

**Does `strata` raise, and `vf15` lower, the REPRESENTATION's opponent-class information at
turns 4–10 — or only V's?**

Both, one, or neither are all informative outcomes. The answer is given as a delta against
`ctrl10M` read against a measured floor, never as a significance verdict alone.

---

## 3. THE TWO BRANCHES, named before the data (per the orchestrator's addendum)

### Branch A — REDUNDANT
`pooled → class` at t4–10 **moves WITH** the V-derived class AUC across `strata`, `vf15` and the
three controls: same sign, and deltas of comparable size relative to their own floors.
**Consequence:** the two rows are redundant, and the cheaper V-level row already in `critic_read`
**remains the decision row**. The two downgraded sentences are RESTORED as written.

### Branch B — RE-SPECIFY  ← the consequential branch
`pooled → class` **does NOT move with** the V-derived class AUC (e.g. `strata`'s V-AUC +0.05 with
`pooled → class` flat; or `vf15`'s V-AUC −0.08 with `pooled → class` flat).
**Consequence:** **the ladder has been deciding on the head's OUTPUT while the research question is
about the REPRESENTATION, and the decision row must be re-specified.** The two downgraded sentences
do not come back; the mechanism story ("more value gradient strips opponent information from the
shared trunk") is REFUTED at the representation for `vf15`, and `strata`'s lever is a head-side
reallocation of an unchanged representation. Every arm judged on `cond.opp_class_auc.t4_10` —
including `strata_b`, whose verdict is already worded at V-level — is judged on a row that does not
answer the question the research direction rests on.

### THE BAR FOR "MOVES WITH", stated so it cannot be softened afterwards

For an arm X ∈ {`strata`, `vf15`}, on a given eval draw, X **MOVES WITH** iff **both**:

1. **SIGN.** `sign(Δ_pooled(X)) == sign(Δ_V(X))`, where `Δ_q(X) = q(X) − q(ctrl10M)` on the SAME
   draw, SAME bucket (t4–10), SAME script.
2. **MAGNITUDE.** `|Δ_pooled(X)| > FLOOR_pooled(draw)`, where
   `FLOOR_pooled(draw) = max over available control pairs of |q_pooled(ctrl_j) − q_pooled(ctrl10M)|`
   on that draw — i.e. `max(|ctrl10M_b − ctrl10M|, |ctrl10M_c − ctrl10M|)` on draw A (hp800, seed
   20260910) and `|ctrl10M_b − ctrl10M|` on draw B (hp800b, seed 20260911; `ctrl10M_c` has no
   hp800b tree — checked before this file was written).

An arm that satisfies (1) and fails (2) **does NOT move with**: that is branch B for that arm.
An arm that fails (1) does not move with either way. The comparison of "size relative to their own
floors" is reported as the ratio `|Δ_pooled| / FLOOR_pooled` beside `|Δ_V| / FLOOR_V`, with
`FLOOR_V` built the identical way from the same control pairs on the same rows.

**Pre-committed resolution of the MIXED outcome** (one arm moves with, the other does not): this is
**branch B**, not a draw. Redundancy is a claim about the row, and a row that tracks V for one
lever and not for another is not a substitute for the representation measurement; the decision row
must be re-specified. The report says which arm did which.

**Pre-committed resolution of a DRAW DISAGREEMENT** (an arm moves with on one draw and not the
other): reported as NOT ESTABLISHED on that arm, with both draws printed; the campaign's rule 18
(same sign on both draws) and rule 19 (a two-draw spread BOUNDS a floor and carries no CI) apply
unchanged. A single-draw clear is never upgraded by the absence of a second draw.

**No post-hoc re-bucketing.** The decision bucket is **t4–10**, fixed here, because §3 of the
N-curve README established the opponent is unobservable at turn 1 and the campaign's decision row
is already t4–10. t1–3 is REPORTED beside it and decides nothing.

---

## 4. MY PRIOR (a prediction, not the bar)

I expect **branch B, partially: `pooled → class` moves LESS than V does, relative to each row's own
floor, and I expect at least one of the two arms to be flat at the representation.**

Reasoning, stated so it can be wrong:

- `value_pooled` at t4–10 already decodes the class at ~0.82–0.85 on this frame, while V manages
  ~0.69–0.76. The scalar is the bottleneck, not the tensor. A lever that changes what the head
  *does with* an unchanged representation moves the scalar and leaves the tensor alone; a lever
  that changes the representation must move both.
- `strata` re-weights the win-prob BCE across bot/sentinel strata. Its most direct effect is on
  how the head allocates its one dimension between strata — a head-side reallocation. For the
  representation to move, the reweighted value gradient must reshape the trunk, which is a weaker,
  second-order path. **Prediction: `strata` FLAT at the representation** (|Δ_pooled| inside the
  floor), with V up. That is branch B.
- `vf15` multiplies the whole value gradient by 1.5 against an unchanged policy gradient, which is
  a trunk-level intervention, so it has the better claim to a representation effect. **Prediction:
  `vf15` moves at the representation in the same (negative) direction but at a smaller multiple of
  its floor than at V.** Whether it clears the floor I genuinely do not know; I put it near even.
- Confidence: branch B ≈ 65 %, branch A ≈ 35 %. If both arms clear their floors with the right
  signs at comparable relative magnitude, branch A is the answer and I was wrong.

---

## 5. What would make the measurement INCONCLUSIVE (declared now)

- `extract.py`'s QC `max |V_fwd − V_rec|` above its 1e-3 refusal on any checkpoint — the frozen
  forward is not reproducing the recorded head, and every row from that tree is void.
- The own-team → opp_class leak above ~0.55 on any frame — the frame is not matched-team and a
  class decode there is partly an own-team decode (the N-curve's hazard 1). The eval trees are
  full-capture 800-game matched-team draws, so ~0.51 is expected; a departure REFUSES the frame.
- A checkpoint that stashes no `win_prob_logits` (`extract.py` raises) — that arm is not a win-prob
  head and is out of the comparison, reported as such.
- Battle counts differing by more than ~10 % between an arm and `ctrl10M` on the same draw: the
  decode's precision then differs between the two sides of a delta, and the delta is reported with
  that noted rather than dropped.

## 6. Fixed analysis parameters (registered, so they cannot be tuned toward a branch)

- Buckets from `decode.BUCKETS`, unchanged: `t1`, `t1_3`, **`t4_10`**, `t11_24`, `t25+`.
- `frame_check.py` run **unmodified**, `--seed 20260910`, `--cap 20000` (above every draw's battle
  count, so every draw uses ALL its battles and the cap subsamples nothing — chosen for
  determinism, not tuned).
- `extract.py` run **unmodified**, one tree per extraction dir, `--threads 2`, `--batch 256`,
  `nice -n 15`, `CUDA_VISIBLE_DEVICES=""`.
- Runs: `ai_v12_17_ladder_strata`, `ai_v12_10_ladder_vf15`, `ai_v12_11_ladder_ctrl10M`,
  `ai_v12_15_ladder_ctrl10M_b`, `ai_v12_16_ladder_ctrl10M_c`, all `@step_10000032`.
  Draws: `hp800` (seed 20260910) for all five; `hp800b` (seed 20260911) for all but `ctrl10M_c`.
