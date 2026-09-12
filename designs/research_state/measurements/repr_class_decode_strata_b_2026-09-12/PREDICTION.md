# PREDICTION — `strata_b` at the REPRESENTATION: do the two `strata` seeds differ in the TRUNK or at the HEAD?

Registered 2026-09-12, **before `extract.py` or `frame_check.py` had been run on
`ai_v12_24_ladder_strata_b` on either draw**. No `pooled_to_opp_class_AUC` for `strata_b` exists
anywhere at the time of writing — the only representation-level numbers in the campaign are
[`../repr_class_decode_2026-09-11/frame_check.json`](../repr_class_decode_2026-09-11/frame_check.json)'s
nine frames (`strata`, `vf15`, `ctrl10M`, `ctrl10M_b`, `ctrl10M_c`), none of which is this run.
This file is committed on its own, before the read.

---

## 1. Why the measurement exists

Two entries set it up.

- **2026-09-11 · READ (`07d42a51`)** — the representation-level decode landed BRANCH A: `strata`'s
  V-level opponent-class gain is matched one-for-one at the representation
  (Δpooled/ΔV = **0.97** on hp800, **1.27** on hp800b at t4–10), so on that lever "the
  representation carries more class information, not only the head" is CONFIRMED. Its §8 names, as
  a next increment, *"`vf15_b` and `strata_b` at the representation, if and when those runs are
  read"*.
- **2026-09-12 · VERDICT** — `strata_b`, the registered seed replicate of `strata`
  (same commit `f871e79f`, argv identical but `--seed 1002` and the run name), **FAILS** the
  confirmatory read: `cond.opp_class_auc.t4_10` = **0.6792 / 0.6798** on the two offline draws
  against `strata`'s **0.7501 / 0.7590 / 0.7599**, i.e. **BELOW all three controls** (0.688–0.716).
  The within-lever spread is **0.08, 3.6× the 0.022 control-replicate floor**. Its consequence (1)
  names this measurement by name: *"the same `pooled → class` decode on `strata_b`'s two frames
  (~50 CPU-min), which says whether the two `strata` runs differ in the TRUNK or only at the head."*

That is the question, and nothing about it has been measured.

---

## 2. THE QUESTION

**`strata` and `strata_b` are two seeds of one lever whose V-level opponent-class decode differs by
0.08. Does their REPRESENTATION differ by a comparable amount (the two runs differ in the TRUNK), or
is `strata_b`'s `value_pooled` like `strata`'s while only the scalar the head emits is low (they
differ at the HEAD)?**

---

## 3. THE TWO BRANCHES, named before the data

Notation, per eval draw D ∈ {`hp800` (seed 20260910), `hp800b` (seed 20260911)}, all at bucket
**`t4_10`**, all from `frame_check.py` run unmodified:

- `q_p(X, D)` = `pooled_to_opp_class_AUC` — the 128-dim `value_pooled` → "is this opponent a
  sentinel?" decode, out of fold, grouped by battle. **The representation.**
- `q_v(X, D)` = `V_to_opp_class_AUC` — the same decode from the single column `V`, on the SAME rows,
  folds and subsample. **The head's scalar output.**
- `FLOOR_p(D)` = the widest |control − `ctrl10M`| on `q_p` on that draw, from the parent
  measurement's already-computed frames: **0.0232** on `hp800` (two pairs: `ctrl10M_b` +0.0232,
  `ctrl10M_c` −0.0088) and **0.0310** on `hp800b` (ONE pair: `ctrl10M_b`; `ctrl10M_c` has no
  `hp800b` tree). `FLOOR_v(D)` built the identical way: **0.0197 / 0.0234**.
  Rule 19: a two- or three-point spread **BOUNDS** a floor, carries no CI, demotes and never
  promotes.

### Branch TRUNK — the two runs differ in the representation
`q_p(strata_b, D) < q_p(strata, D) − FLOOR_p(D)` on **both** draws — the replicate's representation
is separated DOWNWARD from `strata`'s by more than the run-level floor — **and** `strata_b`'s own
delta against the controls is not upward past the floor, i.e. `q_p(strata_b,D) − q_p(ctrl10M,D) ≤
+FLOOR_p(D)`.
**Consequence:** the lever's seed-to-seed reversal is a TRUNK-level phenomenon. The class-balanced
BCE reshapes (or fails to reshape) the tensor the head is handed, and which way it goes is not
determined by the lever. `07d42a51`'s Δpooled/ΔV ≈ 1 for `strata` generalises across seeds as a
statement about the LEVEL at which the lever acts, while its SIGN does not replicate. The
representation row adds nothing to the V-level verdict here either — the two rows again tell one
story — and the standing "V-level is the decision row" is reinforced a second time.

### Branch HEAD — the two runs differ only at the readout  ← the consequential branch
`|q_p(strata_b, D) − q_p(strata, D)| ≤ FLOOR_p(D)` on **both** draws — the replicate's
representation is indistinguishable from `strata`'s at the campaign's run-level standard — **while**
`|q_v(strata_b, D) − q_v(strata, D)| > FLOOR_v(D)` on both draws (the V-level difference the ledger
already recorded, re-measured here by this instrument).
**Consequence:** **the two rows DISSOCIATE across seeds of one lever.** `strata_b` would be carrying
`strata`'s enriched representation while its one-dimensional readout fails to express it, and the
V-level MISS of 2026-09-12 would be a statement about the head, not about the lever's effect on the
representation. That would (a) partially rehabilitate the lever's mechanism claim — class-balanced
BCE does enrich the trunk, on both seeds — while leaving the V-level verdict (NOT CONFIRMED on the
decision row) untouched, and (b) be the first case in the campaign where the representation row says
something the V row cannot, which is precisely the condition `07d42a51` §5 said would break the
"the two rows are redundant" licence. It would make a representation read mandatory before any
future lever is judged on `cond.opp_class_auc.t4_10`.

### THE MIXED / INTERMEDIATE RULES, fixed here so they cannot be softened

1. **PARTIAL-TRUNK.** If `q_p(strata_b)` is separated downward from `q_p(strata)` past `FLOOR_p`
   **but still sits ABOVE `ctrl10M` by more than `FLOOR_p`** — the representation moved partway and
   remains enriched relative to the controls — that is **neither branch**. It is reported as
   **PARTIAL-TRUNK**, with the fraction `[q_p(strata) − q_p(strata_b)] / [q_p(strata) − q_p(ctrl10M)]`
   printed, and the conclusion is "both levels moved, the head moved further".
2. **DRAW DISAGREEMENT.** If the branch test resolves differently on `hp800` and `hp800b`, the
   verdict is **NOT ESTABLISHED**, both draws printed. Campaign rule 18 (same sign on both draws)
   and rule 19 apply unchanged. A single-draw clear is never upgraded by the other draw being
   ambiguous.
3. **BOTH LEVELS FLAT.** If `|q_v(strata_b) − q_v(strata)| ≤ FLOOR_v` on a draw — this instrument's
   V column failing to reproduce the V-level gap `critic_read` v6 measured — the branch test on that
   draw is **VOID**, because the HEAD branch's precondition is unmet and the frames disagree with
   the row of record. That would itself be the finding and is reported as a cross-instrument
   conflict, not absorbed. (`07d42a51` §VERDICT-3 found the two instruments agree to ±0.005 on five
   checkpoints, so this is not expected.)
4. **NO POST-HOC RE-BUCKETING.** The decision bucket is **`t4_10`**, fixed here. `t1_3` and `t11_24`
   are reported beside it and decide nothing — the N-curve established the opponent is unobservable
   at turn 1 on a matched-team frame, and the campaign's decision row is t4–10.
5. **THE FLOORS ARE NOT RE-DERIVED.** `FLOOR_p` and `FLOOR_v` are the parent measurement's, from its
   committed `frame_check.json`, unchanged. The control, `strata` and `vf15` rows are **REUSED
   VERBATIM** and are not recomputed; `frame_check.py` treats each extraction dir independently, so
   merging two new frames into that JSON cannot change an existing row.

---

## 4. MY PRIOR (a prediction, not the bar)

**I expect branch TRUNK, at ≈ 70 %.** Branch HEAD ≈ 20 %, PARTIAL-TRUNK ≈ 10 %.

Reasoning, stated so it can be wrong:

- The parent measurement found `strata`'s effect **entirely representational** — Δpooled/ΔV = 0.97
  and 1.27 at t4–10, and nearly 2:1 at t1–3. On this lever the two levels are one statement. The
  most parsimonious account of a seed that reverses the lever's V-level sign is that it reverses
  both, because on the other seed they moved together.
- A HEAD-only difference between two seeds of the SAME configuration is the more specific
  hypothesis: it requires two trunks that carry the same class information while the single
  direction the win head reads differs by 0.08 AUC. That is representable — the head is one linear
  functional of a 128-dim tensor, and a rotation of the readout is invisible to a 128-dim ridge —
  but it needs the rotation to happen without the tensor moving, and nothing in the campaign has
  yet exhibited that.
- The ledger's within-run dissociation (`strata_b`'s `gate.resolution.all` at or above all three
  controls while its class decode sits below all three) is a dissociation between two DIFFERENT
  quantities at the same level. It is not evidence about the level at which the class effect lives,
  and I decline to read it as such.
- If TRUNK lands, the interesting number is not the branch but the magnitude: whether
  `q_p(strata) − q_p(strata_b)` is ≈ 0.08 (matching V one-for-one, as `strata` did against
  `ctrl10M`) or materially smaller.

---

## 5. DECLARED POWER LIMIT — read this before reading a "within floor" result

🚨 **This design is well powered for the WITHIN-LEVER contrast and poorly powered for
`strata_b`-vs-control.** The two `strata` seeds differ by 0.08 at V; if the representation tracked
that one-for-one the expected `q_p` separation is ~0.06–0.08, i.e. **2.6–3.4× `FLOOR_p`**, which the
floor can resolve. But `strata_b`'s V-level deficit against `ctrl10M` is only **−0.029 / −0.034**,
and `FLOOR_p` is **0.023 / 0.031** — so a representation move of exactly that size against the
CONTROLS would land at 1.0–1.5× the floor and could read "within floor" while being real.

**Therefore, pre-committed:** the branch is decided on the **`strata` vs `strata_b`** contrast
(axis B), which has the separation to resolve it. The `strata_b` vs `ctrl10M` contrast (axis A) is
**REPORTED, with its ×floor, and is not the branch test**; a within-floor axis-A result is recorded
as NOT DETECTED and explicitly does NOT establish "the representation is like the controls".

---

## 6. What would make the measurement INCONCLUSIVE (declared now)

Carried unchanged from the parent PREDICTION §5, because the instrument is unchanged:

- `extract.py`'s QC `max |V_fwd − V_rec|` above its **1e-3** refusal on either `strata_b` frame —
  the frozen forward is not reproducing the recorded head and every row from that tree is void.
- The own-team → opp_class leak above **~0.55** on either frame — the frame is not matched-team and
  the class decode is partly an own-team decode (the N-curve's hazard 1). The nine parent frames sat
  at 0.492–0.498; a departure REFUSES the frame.
- `strata_b` stashing no `win_prob_logits` (`extract.py` raises) — it is a `--critic winprob` run so
  this is not expected; if it happened the arm is out of the comparison and it is reported as such.
- Battle counts differing by more than ~10 % from `strata` on the same draw: the two sides of the
  decisive contrast would then differ in precision, and the delta is reported with that noted rather
  than dropped. (The parent's nine frames spanned 9,536–9,563 battles, a 0.3 % spread.)

## 7. Fixed analysis parameters (registered, so they cannot be tuned toward a branch)

- Checkpoint: **`ai_v12_24_ladder_strata_b@step_10000032`**, the same step as all five parent
  checkpoints.
- Draws: **`hp800`** — the seed-20260910 tree that already exists at
  `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_24_ladder_strata_b/`, the same draw
  the controls' `hp800` frames used; and **`hp800b`**, generated for this measurement with
  `main.ops.eval_trace_gen … --games 800 --sentinels 3 --workers 4 --concurrency 1 --nice 15
  --seed 20260911 --shard-games 25`, matching `hp800b.sh`'s invocation for every other run on that
  draw token for token.
- `extract.py` run **unmodified** from `../winprob_refit_ncurve_2026-09-10/`, one tree per
  extraction dir, `--threads 2 --batch 256`, `nice -n 15`, `CUDA_VISIBLE_DEVICES=""`.
- `frame_check.py` run **unmodified** from the same directory, `--cap 20000 --seed 20260910`
  (the cap is above every draw's battle count, so it subsamples nothing; chosen for determinism,
  not tuned). Buckets from `decode.BUCKETS`, unchanged.
- Tensors under `/home/goodlad/.claude/jobs/9ab51de6/tmp/repdecode_strata_b/`. **Nothing is written
  under `models/`** and the 128-dim tensors are not committed.
- The box carries the live `ai_v12_25_ladder_vf15_b` training arm (launched 02:09 PT 2026-09-12).
  Everything here is CPU-only and niced. **No number in this measurement is a timing measurement.**
