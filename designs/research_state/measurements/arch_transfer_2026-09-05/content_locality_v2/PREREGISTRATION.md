# Pre-registration — content locality v2

**Provenance, stated exactly.** The two predictions and the null below were fixed by the
dispatching brief **before this session loaded a model or generated a state**; they are
transcribed here verbatim, not composed after the fact. The transcription itself was written after
the gen-era n=3 arm had finished running — so this file is a *record* of a prior registration, not
a claim to have been the first thing typed. Nothing in the predictions has been edited, and the
n=3 arm is reported below in full whether or not it agrees with them. Correction 3 (the bootstrap
sizing) arrived mid-flight as a course correction from the coordinator and is a **method** fix, not
a prediction; it is registered here for completeness with its expected direction (point estimates
unchanged; only the v8 pooled-L CI moves).

## What is being re-run, and what is NOT

`content_locality/` measured the right statistic on the wrong networks. This re-runs the identical
measurement with **three** changes and nothing else:

1. **Checkpoint resolution.** Teachers are resolved by
   `agents.training.fixed_opponent_pool._resolve_zip_and_config(run_dir, None)` — IMPORTED, the
   exact call `main/train/model_build.py` makes for a `--distill-teacher`. Every fold in both eras
   named a run DIRECTORY, so the resolver's directory rung applies:
   `best_model/best_model.zip` → `final_model.zip` → `best_model.zip`. Parents are unchanged (the
   folds named them as explicit `.zip` paths).
2. **Two references on the gen side.** REF-A = the fold parent `ai_v9_59_R2ACTION_0827/final_model.zip`;
   REF-B = the exploiters' true fork origin `ai_v9_29_rev1_0823/final_model.zip` @25,067,760. On
   the v8 side the two coincide, because its three teachers fork FROM the fold parent.
3. **The cluster bootstrap is sized from its own array** (`boot.py`), with an assertion that the
   drawn index range equals the cluster count.

Unchanged: teams, seeds, battles per team, the parent-piloted state generation, the pilot, the
reference opponent, `concurrency=1`, the bridge impl per era, the KL function, and the two
adjacent-checkpoint floor pairs per era (REF-B additionally gets its own floor from rev-1's two
nearest checkpoints — declared as an addition, not a substitution).

## Registered predictions

**(i) Under the resolved checkpoints, both of H1's readings survive.**
The cross-era ordering (`R_v8 > R_gen`) and the within-gen null (funded vs unfunded R NOT
DETECTED) both hold. Rationale: `teacher_distance` already re-measured the underlying levels on
the resolved checkpoints and saw them move in the strengthening direction (the unfunded half's
untaught KL fell 0.5990 → 0.5536; v8's set mean fell 0.2740 → 0.2329, i.e. 5.2× → 4.5× its floor).

**(ii) Under REF-B, the gen teachers' R RISES toward v8's 1.45.** Two readings are registered in
advance so neither can be chosen afterwards:

* **R ≥ 1.3** ⇒ the "gen teachers are global" conclusion was largely the **fork-origin offset**:
  every gen teacher's KL against the fold parent carries a shared constant (the parent is a
  *sibling* 3.05M steps away from the common origin), and that constant dilutes the own/sibling
  ratio. The two eras' exploiters would then be **alike**, and the era difference in H1 an artifact
  of what each teacher was measured against.
* **R ≈ 1.1** ⇒ gen exploiters are **genuinely global from their own origin**, consistent with
  `exploiter_drift`'s flat ρ (1.22 at 150k steps, 1.26 at 5.0M, three framings all NOT DETECTED),
  and the fork origin is **not** the explanation.

**The null, registered:** wherever two R values' CIs overlap, the verdict is **NOT DETECTED** — never
a direction. A cross-era or within-era difference whose bootstrap CI contains zero is reported as
NOT DETECTED.

**(iii) Method (course correction, not a prediction).** The bootstrap fix changes no point
estimate anywhere; it changes only the v8 pooled-L CI (`primary_A_era`), whose 23-cell array was
resampled with indices in [0, 22).

## Cross-checks that must pass before anything is reported

* The gen n=3 arm must reproduce the canonical `offline_collateral_kl` batch: **1100** untaught
  states on indices 0–7. Asserted in code.
* The gen n=9 arm must reproduce `teacher_distance`'s untaught per-team counts
  `[280, 399, 333, 458, 714, 592, 391, 301]`. Asserted in code.
* The v8 arms must reproduce `content_locality`'s own untaught per-team counts
  (n=3 `[109, 104, 98, 96, 88, 80, 92, 78]`, n=9 `[266, 255, 260, 312, 270, 265, 303, 259]`).
  Asserted in code.
* **ACID**: no two teachers may produce an identical per-team KL vector — a mis-resolved path
  masquerading as a null.
* Every teacher's resolved sha256 is recorded beside the file v1 scored.

## External cross-reference held in advance

The coordinator's z-swap probe measured v8's sibling-control **R = 1.8316 [1.5349, 2.1782]** on the
resolved (`best_model`) checkpoints — `semistall3`'s off-slice KL roughly halves under the right
file. This arm's v8 R should land near that. **If it does not, the difference is a finding to
report, not noise to smooth.**
