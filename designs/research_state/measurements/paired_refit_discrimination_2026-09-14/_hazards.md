## 8. Hazards and findings about the instruments

Every one of these is a finding, not an aside.

1. **🚨 `ai_v12_22_ladder_rollout` DOES NOT EXIST.** The task named it; `models/ai_v12_22_*` is
   `ai_v12_22_ladder_lambda095` and the rollout-target arm is **`ai_v12_23_ladder_rollout`**. 23 is
   what ran, and the substitution is recorded in `PREDICTION.md` rather than made silently.
2. **`main.ops.critic_read` cannot read an EXTERNAL head.** It addresses an arm by run NAME or run
   DIRECTORY and recomputes from that run's recorded npz — there is no seam for a head fitted
   elsewhere. The conditioning guard here is therefore the N-curve's `frame_check` decode applied
   to the refit head's own outputs, which is the fallback the task named. It is a RAW monotone AUC,
   not the tool's out-of-fold decoder fit; they agree in magnitude (the original head reads 0.711
   here against the published 0.723 / 0.700) because a 1-D monotone decoder's AUC is nearly the AUC
   of V itself — `conditioning_meters` says so itself.
3. **The AUC's ORIENTATION is load-bearing and is not stated in most quotations of the row.** The
   published meter labels `pool` (sentinel) as 1 and is an out-of-fold FIT, which picks its own
   direction; a RAW AUC on the same labels reads **0.289** for the head that "reads 0.711". Any
   hand-rolled decode of this row must say which class is 1, or it will look like an
   anti-detector.
4. **A GREEDY-vs-GREEDY continuation stalls, and the stall is concentrated in recorded DRAWS.** The
   first fork smoke put two forks at turns 178 and 183 of a recorded draw and **five of their six
   branches hit the 250-turn cap**. Excluding recorded-draw battles and bounding the divergence
   turn at 40 took the capped rate to **13 branches in 15,228 (0.09 %)**. A capped branch is decided
   by SEAT and is not an outcome; a fork design that did not exclude draws would have built a fifth
   of its dataset out of forfeit ordering.
5. **A fork dataset's shard must not be stopped ALPHABETICALLY.** This tree's filenames sort
   `draw_* < loss_* < win_*`, so a shard that emits its picked forks in file order and is stopped
   by the clock returns a sample of LOSSES. Caught in the 20-battle smoke — all 12 forks came from
   loss battles. `forks.py` shuffles the BATTLE order and keeps each battle's forks together, which
   buys record-load locality and an unbiased prefix at once.
6. **`--leaf-head` is exact, and that was measured on the real checkpoint, not asserted.** Swapping
   the checkpoint's own head back in leaves the policy logits and V **byte-identical**
   (max \|ΔV\| = 0.0); swapping the refit head in moves V by mean 0.028 and leaves the policy logits
   byte-identical. Both Part C cells run THROUGH the flag (the control loads `head_original.pt`),
   so a difference between them cannot be the hook.
7. **A results ROW still cannot name the head that produced it** — the 2026-09-11 battery's hazard 1
   is unfixed, and `--leaf-head` adds one more thing a row cannot say about itself. The swap is
   announced at startup with the file's sha1, and cells are kept apart by FILE NAME only.
8. **Every wall-clock number here is contention-coupled** (rule 23). The box carried a live training
   arm throughout; load average ran 22–55 on 16 cores. That is why every L1 in this directory is
   printed beside its realized K and read only against a control from the SAME window, and why
   `report.py` has had the 18.1 % bar removed rather than inherited.
9. **The fork target was missed by 4×** — 5,076 forks against the ~20,000 asked for, because six
   shards and a battery had to share a box with a training run. The consequence is the width of
   §5's intervals (±0.035 on a pairwise accuracy), not their validity; the two detected deltas are
   detected at this width and the one null is a null at ±0.014 on the delta.
10. **The refits early-stop fast.** Every arm found its best validation loss within ~50 optimisation
    steps and then ran 1,000 more without improving. These are 4-tensor fits on a 128-dim input
    with ~9.6k rows; a longer budget is not obviously what the ranking arm lacks, but "the ranking
    term needs more optimisation" is not excluded by this measurement.
11. **One checkpoint, one seed, one opponent family.** Part B is a single-arm study (rules 2 and 22):
    the +0.086 is a CANDIDATE until it replicates on a second checkpoint, and the continuation
    ecology is three pool sentinels played greedy, not the training mixture.
