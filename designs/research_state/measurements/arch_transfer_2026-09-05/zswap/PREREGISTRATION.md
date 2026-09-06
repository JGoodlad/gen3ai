# PRE-REGISTRATION — H7 "z-swap": does v8's FiLM team-code carry the exploiters' specialisation?

**Frozen 2026-09-05, before a single state was generated or a single forward run.**
Nothing below is a result. The only thing already established when this was written is the
STRUCTURE of the z path, read off the era source at `b13b30b2` (recorded in §1, and itself a
deliverable) — no checkpoint had been loaded and no KL computed.

## 0. Why

[`../content_locality/`](../content_locality/README.md) measured that **v8-era teachers are LOCAL**
(sibling-control `R = 1.4498 [1.2728, 1.6722]` — each teacher diverges from the fold parent ~45%
more on its own taught teams than its siblings do on those same states) while **gen-era teachers
are GLOBAL** (`R ≈ 1.07–1.10`, CI touching or including 1). v8's fold GIFTED; gen-era folds ROB.

The owner's hypothesis: v8's entire line ran with **zarch** conditioning ON — a 32-dim per-team
code `z` with reconstruction + VICReg losses, FiLM-modulating the heads. If FiLM lets an exploiter
specialise **through z** rather than through the shared weights, its change stays local, and the
fold inherits general skill rather than team-specific tactics. The gen era has no such code.

This probe tests **"z localises CHANGE"**. It does NOT test "z improves performance" — an earlier
v8-era measurement already found FiLM gradients orthogonal across teams, ~2/3 of z's energy in one
shared direction ("lazy mode"), and a free per-team LUT code buying nothing. Those are a different
claim and are not re-litigated here.

## 1. What zarch IS (read from the era source BEFORE measuring — no checkpoint loaded)

At `b13b30b2`, all of it in `src/agents/model/features_extractor.py`:

* **`ZArchEncoder` (L3450–3505)** — z is **OBS-DERIVED**, not a stored per-team table. A
  permutation-invariant DeepSets code over OUR 6 mons' *invariant* facts: species ⊕ item ⊕ ability
  ⊕ mean-of-4-move embeddings ⊕ the 18-dim spread block (IVs/EVs/nature). Per mon → a shared
  `atom_mlp` (`Linear(atom_in,64) → ReLU → Linear(64,32)`) → **mean over the 6** → `LayerNorm` →
  `z [B,32]`. Every embedding-table read is `.detach()`ed, so z's gradients cannot reshape the
  trunk. Team-STATIC by construction (no HP/status/boosts/PP), so within a battle z is constant.
* **`recon_head`** — species multi-hot logits, aux loss only, never fed forward.
* **Construction (L3999–4020)** — `zarch_film ∈ {off, heads}`; when on, `film_pi` and `film_vf`
  are `Linear(32, 2*PROJECTION_DIM)`, **both weight and bias zero-init** (exact identity at init).
* **Computation (L4160–4167)** — inside `forward_internal`, stashed as `self.last_zarch`.
  `recon_logits` is computed only when `self.training`.
* **Consumption (L4425–4430)** — the ONLY consumption site, in `forward`, **post-projection,
  pre-ReLU**:
  ```
  dg_pi, db_pi = film_pi(last_zarch).chunk(2, -1)
  dg_vf, db_vf = film_vf(last_zarch).chunk(2, -1)
  pi_pre = pi_pre * (1 + dg_pi) + db_pi ;  vf_pre = vf_pre * (1 + dg_vf) + db_vf
  ```

**Therefore the swap is STRUCTURALLY WELL-DEFINED and is a true swap of the code, not of an
encoder we hope is equivalent.** z is never fed into the trunk; it enters the network at one line,
at the very end, on both heads. Substituting a different `[B,32]` tensor at `last_zarch` changes
*exactly* the FiLM modulation and nothing else. Because z is obs-derived, "P's z" means **the
tensor P's own `zarch_encoder` produces on the same observation** (captured from P's own forward),
i.e. swapping the z-*encoder*'s output — stated here because the brief requires it be said which.

All five models involved carry `zarch_dim: 32, zarch_film: "heads", zarch_recon_coef: 1.0,
zarch_vicreg_coef: 0.1` (verified from each `model_config.json`), so no arm is a no-op for want of
the feature being on.

## 2. Conditions

`P` = fold parent `ai_v8_04`. `T` = each of v8_14's three teachers (`ai_v8_09_pool10`,
`ai_v8_06_semistall_3team`, `ai_v8_13_defensive10`). Same states for every arm.
`M[z]` = model M's trunk/heads evaluated with the FiLM code forced to `z`.

| id | condition | statistic |
|---|---|---|
| **a** | baseline, each model its own z | `KL(T[z_T] ‖ P[z_P])` |
| **b** | teacher's weights, **parent's z** | `KL(T[z_P] ‖ P[z_P])` |
| **c1** | parent's weights, **teacher's z** — "P diverging from itself" | `KL(P[z_T] ‖ P[z_P])` |
| **c2** | both on the teacher's z | `KL(T[z_T] ‖ P[z_T])` |
| **d0** | both with z **zeroed** | `KL(T[0] ‖ P[0])` |
| **dμ** | both at each model's own **state-mean** z | `KL(T[z̄_T] ‖ P[z̄_P])` |

`d0` is *not* an identity ablation: the FiLM biases are zero at init but free thereafter, so
`film(0)` is the learned bias. `dμ` removes z's team-to-team VARIATION while keeping its mean
level, which is the ablation matched to "2/3 of z's energy is one shared direction".

**Mechanism precheck (registered, and it can pre-empt everything else):** the **z-sensitivity**
`KL(T[z_P] ‖ T[z_T])` and `KL(P[z_T] ‖ P[z_P])` — how much swapping z moves a *single* network.
If these are themselves below the matched-noise floor, the swap is **a near-no-op by construction**
and conditions b/c/d cannot speak to "does z carry specialisation"; they instead answer
"z barely does anything in this era", which is a real answer and will be reported as one.

## 3. Predictions and RAILS (frozen)

Primary readout: **the fraction of on-slice KL removed by the z-swap**,
`f_b = 1 − KL_b / KL_a`, on each teacher's OWN taught teams, with a cluster bootstrap CI.

* **`f_b > 0.50`** (CI excluding 0.50 from below) ⇒ **z CARRIES THE SPECIALISATION** — FiLM
  localised the exploiter's change.
* **`f_b < 0.20`** (CI excluding 0.20 from above) ⇒ **it lives in the SHARED WEIGHTS** — FiLM did
  not localise it.
* between the two rails ⇒ **PARTIAL**, reported as such, no verdict claimed.
* **Off-slice** (`untaught`) `f_b` should change LITTLE if the z story is right; a z-swap that
  removes on-slice and off-slice KL equally is not localisation, it is a global rescale, and will
  be reported as such. The registered contrast is `f_b(taught-own) − f_b(untaught)`, CI over teams.
* **c1** should RAISE P's divergence from itself on taught states "by a comparable amount" to
  what b removes. Registered comparison: `KL_c1(taught-own)` vs `KL_a − KL_b` on the same teams.
* **Locality under the swap**: recompute the sibling-control `R` (content_locality's PRIMARY B,
  same 21 singly-taught teams, same formula) under condition **b**. **If `R_b` falls toward 1
  (CI including 1) while `R_a` excludes it, z is what made the v8 teachers local.** If `R_b ≈ R_a`,
  it is not.

**NULL / floor.** The matched-noise floor is content_locality's own: two adjacent `ai_v8_04`
checkpoints scored against the parent final on the same states — KL `0.0263` / `0.0535` on taught,
`0.0383` / `0.0664` on untaught. **Any KL CHANGE smaller in absolute value than the larger floor on
that slice (0.0535 taught, 0.0664 untaught) is WITHIN FLOOR** and reported as such, never as an
effect. Both floor checkpoints are re-scored in this run under conditions a and b as well, so the
floor is measured on *this* run's states rather than inherited.

Vocabulary: **SIGNIFICANT** (CI excludes the rail) / **WITHIN FLOOR** / **NOT DETECTED**.

## 4. My own competing prediction (registered so it can be wrong)

I expect **`f_b` to land LOW — below the 0.20 rail** — and the z-sensitivity precheck to be small.
Reasons, all structural and available before measuring: the FiLM site is a rank-≤32 affine
modulation of the *final* head features only, applied after every trunk computation is finished;
the exploiter's PPO gradient flows through the entire trunk with nothing stopping it, so the path
of least resistance for "get better on these 10 teams" is the trunk, not a 32-dim code whose
encoder reads only team-static facts; and the prior "lazy mode" finding says z's own variation is
mostly one shared direction, which is exactly the direction a swap does *not* change much between
two models trained from the same parent. If that is right, the answer to the owner's hypothesis is
"FiLM was available as a localisation mechanism and the exploiters did not use it" — which leaves
v8's measured locality (R 1.45) needing a different explanation, and I register in advance that
this probe would then NOT have explained it.

## 5. Secondary, computed unconditionally

**Parameter-mass split.** Per teacher, the fraction of `‖θ_T − θ_P‖²` (and of `‖θ_T − θ_P‖₁`) that
lives in the **z path** (`zarch_encoder.*`, `film_pi.*`, `film_vf.*`) versus the shared trunk/heads,
alongside each group's share of the parameter COUNT (a group that is 0.1% of parameters holding
0.1% of the displacement has done nothing special). This is the brief's fallback deliverable; it is
cheap and it addresses the hypothesis from the weight side, so it is run regardless of §3.

## 6. Recipe (fixed here, not after seeing numbers)

States: **content_locality's v8 recipe, unchanged** — the fold parent pilots each team against the
fixed reference `ai_v8_03_zarch_control_0718` final, `stochastic=False` both sides, node bridge,
`concurrency=1`, sim seed `[team_index+1,2,3,4]`, pool sequence `random.Random(61000+i)`. Teams:
the deduped union of the three teachers' recorded `--trainee-teams` (22, one doubly-taught and
excluded from the sibling control) + probe P's first 8 untaught. `per_team = 3` and `9`.

**ACID, registered:** condition **a** must reproduce content_locality's committed
`per_team_kl_fwd` for the same teacher and `per_team`. It is the same states, the same models and
the same formula, so a mismatch means the state batch did not reproduce and the run is void. The
comparison will be printed as executed output.

KL: forward `KL(T ‖ P)` over legal actions, `era_kl.masked_kl_rows_era` — the same verbatim copy
content_locality gated bit-identical against the gen-era import.

Analysis: **team is the cluster**; cluster bootstrap over teams (20000 resamples, seeded);
per-teacher and pooled; the six conditions × {taught-own, taught-sibling, untaught}.

---

## AMENDMENT 1 (2026-09-05, after the first pass, before any rail was read as final)

**Which FILE is "the teacher" changed. No rail, prediction or statistic changed.**

The recipe above inherited content_locality's teacher paths,
`<run>/final_model_interrupted.zip`. That is **not the checkpoint the fold distilled.** v8_14's
recorded argv names each teacher as a **run DIRECTORY**
(`--distill-teacher /…/ai_v8_09_pool10_exploiter_0723:*;…`), and a directory is resolved by
`fixed_opponent_pool._resolve_zip_and_config` through the rung order
`best_model/best_model.zip` → `final_model.zip` → `best_model.zip`. All three teacher runs have
`best_model/best_model.zip`; none has `final_model.zip`; and `final_model_interrupted.zip` is not
a rung at all. So the fold's teachers were the **best_model** checkpoints.

The **PARENT is unaffected** — the fold names it as a direct
`--model models/ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip`, which the resolver
returns verbatim. The floors are parent-run checkpoints and are likewise unaffected, and so is the
STATE batch, which is a function of the parent and the reference opponent alone.

**Consequence for this probe:** `best_model` is the HEADLINE and `final_interrupted` is retained
as a labelled SECONDARY (`$ZSWAP_TEACHER_FILE`, recorded in every artifact's `_meta`). Both are
run at n=3 and n=9 from the same script.

**Every ratio must be WITHIN-FILE.** `f_b`, `R`, and every `zsens*` compare conditions computed on
the SAME checkpoint, so no number in this probe mixes files. The one cross-file comparison that
exists is the ACID gate, which now asserts the two directions separately: the floors and the state
batch must reproduce content_locality bit-for-bit in BOTH modes, while the teacher rows must be
bit-identical in `final_interrupted` and must **DIFFER** in `best_model` — agreement there would
mean the two files are the same model and would void the reason for the re-run, so it is reported
as a failure rather than accepted.
