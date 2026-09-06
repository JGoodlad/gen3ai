# H7 z-swap — did v8's FiLM team-code carry the exploiters' specialisation?

**2026-09-05. Offline. No training, no launcher, no server. CPU only, `nice -n 10`.**
Four cells (2 teacher-file modes × n∈{3,9} battles/team), ~29 minutes of battles (~42 minutes wall) in total.

**Answer: NO — and the measurement is not a no-op, which is what makes it worth having.**
The pre-registered swap removes **3.8%** of the on-slice divergence (`f_b = +0.0383
[+0.0212, +0.0554]`, n=9), far below the registered 0.20 rail ⇒ **SHARED WEIGHTS**. The
sibling-control locality that motivated the whole probe is **completely untouched** by the swap
(`R_a − R_b = −0.0011 [−0.0379, +0.0318]`, **NOT DETECTED**). Meanwhile FiLM is *not* a small
decoration — it modulates the head features by **175–247% of their own norm** — so the null is
"the mechanism was engaged and did not localise", not "the feature was off".

---

## 1. What zarch IS

**Nobody had written this down since the fresh generation, so it is the first deliverable.** Read
off the era source at `b13b30b2`, all of it in `src/agents/model/features_extractor.py`:

> **`z` is OBS-DERIVED, not a stored per-team table.** `ZArchEncoder` (**L3450–3505**) is a
> permutation-invariant DeepSets code over OUR six mons' *invariant* facts — species ⊕ item ⊕
> ability ⊕ the mean of the four move embeddings ⊕ the 18-dim spread block (IVs/EVs/nature). Each
> mon goes through a shared `atom_mlp` (`Linear(atom_in, 64) → ReLU → Linear(64, 32)`), the six are
> **averaged**, and the result is `LayerNorm`ed to `z ∈ ℝ³²`. Every embedding-table read is
> `.detach()`ed, so nothing in the z path can reshape the trunk's embeddings. Every input is
> constant within a battle, so z is **team-static** by construction. It is built in
> `forward_internal` (**L4160–4167**) and stashed as `self.last_zarch`; a `recon_head` side readout
> produces species multi-hot logits for the aux loss **only when `self.training`**, and is never fed
> forward. **z then enters the network at EXACTLY ONE site** — `forward`, **L4425–4430**,
> post-projection and pre-ReLU on both heads: `[Δγ‖Δβ] = film_{pi,vf}(z)`, then
> `pi_pre ← pi_pre·(1+Δγ_pi) + Δβ_pi` and likewise for `vf_pre`. The two generators are
> `Linear(32, 2·PROJECTION_DIM)` with **weight and bias both zero-init**, so FiLM is the exact
> identity at initialisation (**L3999–4020**). **z never touches the trunk.**

Because z enters at one line at the very end, **the swap is structurally well-defined**:
substituting a different `[B,32]` tensor for `last_zarch` changes *exactly* the FiLM modulation and
nothing else. And because z is obs-derived, **"the parent's z" means the tensor the parent's own
`zarch_encoder` produces on the same observation** — captured from the parent's own forward. So this
probe swaps the **z-encoder's output**, not a lookup-table row; that distinction is stated here
because it is the one the brief asked to have made explicit.

All five models carry `zarch_dim: 32, zarch_film: "heads", zarch_recon_coef: 1.0,
zarch_vicreg_coef: 0.1` (verified from each `model_config.json`), so no arm is a no-op for want of
the feature being on.

## 2. Pre-registration

[`PREREGISTRATION.md`](PREREGISTRATION.md), frozen before a single state was generated — including
§1's structural reading (no checkpoint had been loaded), the six conditions, the rails, and **my own
competing prediction that `f_b` would land below the 0.20 rail**, with reasons. It carries one
**AMENDMENT** (teacher-file resolution, §5 below); no rail, prediction or statistic changed.

## 3. Conditions

`P` = fold parent `ai_v8_04`; `T` = each of v8_14's three teachers. `M[z]` = model `M` evaluated
with the FiLM code forced to `z`. Same states for every arm; forward `KL` over legal actions.

| id | condition | what it isolates |
|---|---|---|
| **a** | `KL(T[z_T] ‖ P[z_P])` | baseline (**= content_locality's statistic**) |
| **b** | `KL(T[z_P] ‖ P[z_P])` | **the swap** — teacher's weights, parent's code |
| **c1** | `KL(P[z_T] ‖ P[z_P])` | the parent diverging from *itself* on the teacher's code |
| **c2** | `KL(T[z_T] ‖ P[z_T])` | both on the teacher's code |
| **d0** | `KL(T[0] ‖ P[0])` | both with z zeroed (**off-manifold**; `film(0)` = the learned bias) |
| **dμ** | `KL(T[z̄_T] ‖ P[z̄_P])` | both at their own state-mean code — kills team-to-team **variation** only, on-manifold |
| `zsens*_T` | `KL(T[z_X] ‖ T[z_T])` | how far each intervention moves **one** network from itself |

## 4. Recipe

State batch is **content_locality's v8 recipe, unchanged**: the fold parent pilots each team against
the fixed reference `ai_v8_03_zarch_control_0718` final, `stochastic=False` both sides, **node**
bridge, `concurrency=1`, sim seed `[team_index+1,2,3,4]`, pool sequence `random.Random(61000+i)`.
Teams: the deduped union of the three teachers' recorded `--trainee-teams` (**22**, one doubly-taught
and excluded from the sibling control) + probe P's first 8 untaught. Code: `/tmp/v8rep_era`,
commit `b13b30b2`, READ-ONLY, `PYTHONDONTWRITEBYTECODE=1`.

KL: `era_kl.masked_kl_rows_era` — content_locality's verbatim copy, already gated bit-identical
against the gen-era `masked_kl_rows` import by its `kl_unit_test.py`.

**Cluster = TEAM.** Cluster bootstrap over teams, 20 000 resamples, seed `20260905`.

| cell | states | battles wall | total wall |
|---|---|---|---|
| `best_model` n=3 | 4 180 | 254 s | 368 s |
| `best_model` n=9 | 11 650 | 618 s | 913 s |
| `final_interrupted` n=3 | 4 180 | 229 s | 337 s |
| `final_interrupted` n=9 | 11 650 | 628 s | 920 s |

### Three ACID gates, all executed

**(i) The swap mechanism is faithful.** Before any measurement, each of the six models is checked
twice: the shim **disarmed** must reproduce the unpatched logits, and the shim **armed with the
model's own captured z** must too. All 24 checks returned exactly `0.000e+00`.

**(ii) The state batch reproduces content_locality exactly.** Condition `a` *is* content_locality's
statistic, so it must reproduce it. It does — bit-for-bit, in both modes, on the models that share a
file:

```
=== ACID: …/zswap_n9.json
         vs …/../content_locality/v8_era_n9.json   (teacher-file mode = best_model) ===
  recorded teacher rungs: {'pool10': 'best_model/best_model.zip', 'semistall3': 'best_model/best_model.zip', 'defensive10': 'best_model/best_model.zip'}
  states   zswap 11650   content_locality 11650   MATCH
  per-team state counts identical: True
  team order identical: True

  model            role          max|zswap.a - CL|    mean KL  verdict
  FLOOR_c277178    floor                 0.000e+00   0.029495  OK (must match)
  FLOOR_c275758    floor                 0.000e+00   0.056926  OK (must match)
  pool10           teacher               1.831e-02   0.384081  OK (must DIFFER)
  semistall3       teacher               3.086e-01   0.123012  OK (must DIFFER)
  defensive10      teacher               1.824e-02   0.310779  OK (must DIFFER)

  PASS
```

The gate asserts **both directions**: the floors (parent-run checkpoints, same file) must match, and
the teachers — loaded from a *different* checkpoint in this mode — must **differ**. Agreement there
would mean the two files are the same model and would void the reason for the re-run, so it is
reported as a failure rather than quietly accepted. In `final_interrupted` mode every row must match,
and does.

**(iii) No two models produced an identical KL vector** (`acid_all_distinct: true`), so a
mis-resolved path cannot masquerade as a null.

## 5. The teacher-file correction — and it moves one teacher by 2×

**This probe's headline uses `best_model/best_model.zip` for each teacher, not
`final_model_interrupted.zip`.** v8_14's recorded argv names each teacher as a **run DIRECTORY**:

```
--distill-teacher /home/goodlad/dev/gen3ai/models/ai_v8_09_pool10_exploiter_0723:*;…
--model models/ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip
```

`fixed_opponent_pool._resolve_zip_and_config` resolves a directory through
`best_model/best_model.zip` → `final_model.zip` → `best_model.zip`. All three teacher runs **have**
`best_model/best_model.zip`; none has `final_model.zip`; **`final_model_interrupted.zip` is not a
rung at all.** The PARENT is a direct `--model …/final_model_interrupted.zip` and is returned
verbatim, so it — and the floors, and therefore the whole state batch — is unaffected.

It matters, and only for one teacher:

| teacher | slice | `best_model` KL (headline) | `final_interrupted` KL | ratio |
|---|---|---|---|---|
| pool10 | taught-own | 0.4722 | 0.4719 | 1.00 |
| pool10 | untaught | 0.3176 | 0.3223 | 0.99 |
| **semistall3** | **taught-own** | **0.2156** | **0.4478** | **0.48** |
| **semistall3** | **untaught** | **0.1036** | **0.2190** | **0.47** |
| defensive10 | taught-own | 0.3586 | 0.3529 | 1.02 |
| defensive10 | untaught | 0.2775 | 0.2807 | 0.99 |

*(n=9; n=3 agrees to ±0.01 on every row.)*

**Every ratio in this probe is WITHIN-FILE.** `f_b`, `R`, and every `zsens*` compare conditions
computed on the *same* checkpoint, so no reported number mixes files. The only cross-file comparison
is ACID gate (ii), which is explicitly two-directional.

🚩 **This also corrects [`../content_locality/`](../content_locality/README.md), which used the
wrong file.** Its headline `R = 1.4498 [1.2728, 1.6722]` is, on the checkpoints the fold actually
distilled, **`R_a = 1.8316 [1.5349, 2.1782]`** (n=9; n=3 `1.7940 [1.4814, 2.1584]`). The mechanism is
that semistall3 is a *sibling* on 19 of the 21 singly-taught teams, so halving its KL lowers the
sibling mean (0.2959 → 0.2303) and raises the ratio. **The direction of that probe's verdict is
unchanged and strengthened — the v8 teachers are even more LOCAL than it reported** — but the number
in its README, and any cross-era delta computed from it, is understated.

## 6. Mechanism first — is there anything to swap?

Registered as a precheck that could pre-empt everything else. It does not: **FiLM is a dominant term,
and the exploiters grew it.**

| model | FiLM modulation ‖h·Δγ + Δβ‖ / ‖h‖, **pi** | **vf** |
|---|---|---|
| parent `ai_v8_04` | **174.5%** | 190.7% |
| FLOOR c277178 (−405k steps) | 178.8% | 194.9% |
| FLOOR c275758 (−1.82M steps) | 180.4% | 196.1% |
| teacher pool10 | **231.8%** | 233.0% |
| teacher semistall3 | **223.7%** | 238.6% |
| teacher defensive10 | **243.5%** | 246.7% |

Two things to read here. FiLM is **larger than the features it modulates** — this is not a small
correction to the head, it is a co-equal term. And the **exploiters raised it by ~30–40%** while
ordinary continued training was drifting it slightly *down* (180.4 → 174.5 across 1.82M parent
steps). The z path was actively engaged by exploiter training.

The codes moved too, and far more than ordinary training moves them:

| model | ‖z‖ | RMS distance to `z_P` | as % of ‖z‖ |
|---|---|---|---|
| FLOOR c277178 | 16.29 | 1.33 | **8.2%** |
| FLOOR c275758 | 16.28 | 1.91 | **11.7%** |
| pool10 | 17.05 | 7.93 | **46.5%** |
| semistall3 | 17.02 | 6.34 | **37.2%** |
| defensive10 | 17.33 | 7.81 | **45.0%** |

**And yet the behaviour barely notices.** (n=9, taught-own slice, against the 0.0535 taught floor):

| intervention on ONE network | KL from itself | vs floor |
|---|---|---|
| teacher gets the parent's code `T[z_P] ‖ T[z_T]` | 0.0237 [0.0189, 0.0286] | **WITHIN FLOOR** |
| teacher gets its own mean code `T[z̄] ‖ T[z_T]` | 0.0449 [0.0261, 0.0717] | **WITHIN FLOOR** |
| teacher gets z = 0 `T[0] ‖ T[z_T]` | 0.1112 [0.0940, 0.1342] | above floor |
| parent gets the teacher's code `P[z_T] ‖ P[z_P]` | 0.0149 [0.0125, 0.0175] | **WITHIN FLOOR** |
| parent gets its own mean code | 0.0214 | **WITHIN FLOOR** |
| parent gets z = 0 | 0.0530 | **WITHIN FLOOR** |

**So FiLM's magnitude is carried almost entirely by a component that does not vary with the team.**
The parent's per-state code has norm ≈16.23 while its deviation from the state-mean has RMS 10.66,
so the constant part has norm **≥ 12.24** (exactly: ‖z̄‖² = E‖z‖² − 10.66², and E‖z‖² ≥ 16.23² by
Jensen — a lower bound, not an estimate). Removing all team-to-team variation from the code moves the
network by 0.045; removing the code entirely moves it by 0.111. **A large, mostly-constant affine
reparameterisation of the head, with a small team-conditional rider** — which is the earlier
"lazy mode / ~2/3 of z's energy in one shared direction" finding, restated at the behavioural level
and reproduced here from this probe's own artifacts.

## 7. Results

n=9 is the headline; **n=3 replicates every row and every verdict** (`analysis_n3.log`).

### 7.1 The pre-registered primary

`f_b = 1 − KL_b/KL_a` — the fraction of divergence the swap removes. Rails: `>0.50` ⇒ z carries it;
`<0.20` ⇒ shared weights.

| slice | n=9 | n=3 | verdict |
|---|---|---|---|
| **taught-own** | **+0.0383 [+0.0212, +0.0554]** | +0.0356 [+0.0108, +0.0577] | **SHARED WEIGHTS** |
| untaught | +0.0069 [−0.0032, +0.0188] | +0.0083 [−0.0031, +0.0179] | **SHARED WEIGHTS** |

The absolute KL the swap removes on-slice is **+0.0149 [+0.0081, +0.0218]** against a taught floor of
0.0535 — **WITHIN FLOOR**. The swap removes less divergence than two adjacent parent checkpoints
differ by.

`f_b(taught-own) − f_b(untaught) = +0.0314 [+0.0109, +0.0514]` — **SIGNIFICANT**, and honestly
reported as such, but it is a 3-percentage-point preference inside an effect that is itself 4% of
the quantity in question, and both slices sit an order of magnitude below the rail. It says the swap
is very slightly more on-slice than off-slice; it does not say z carries the specialisation.

### 7.2 The pooled condition table (n=9, taught-own / untaught)

| condition | taught-own | untaught |
|---|---|---|
| **a** baseline | 0.3893 [0.3427, 0.4405] | 0.2329 [0.2202, 0.2453] |
| **b** the swap | 0.3744 [0.3284, 0.4237] | 0.2313 [0.2172, 0.2454] |
| **c1** parent on teacher's z | 0.0149 [0.0125, 0.0175] *(floor)* | 0.0031 *(floor)* |
| **c2** both on teacher's z | 0.3922 [0.3480, 0.4389] | 0.2339 |
| **dμ** both at own mean z | 0.3566 [0.3221, 0.3890] | 0.2281 |
| **d0** both at z=0 | 0.2728 [0.2440, 0.3001] | 0.1694 |

`f_dμ = +0.0840 [+0.0423, +0.1393]` — even deleting **all** team-conditional modulation from both
models, on-manifold, removes only 8.4%, and the absolute removal (+0.0327) is still **within floor**.

`f_d0 = +0.2994 [+0.2623, +0.3412]` is the one number that could be misread. It is **not** the swap,
it is **off-manifold** (z=0 is never a real code; it moves the teacher 0.111 from itself), and it is
a **joint** intervention that removes the difference in the FiLM *generator weights* as well as in
the codes. It says the FiLM path as a whole carries ~30% of the T-vs-P divergence — through weights
that apply to *every* team's code, which is a shared-weights fact, not a localisation one. The
on-manifold version of the same question is `f_dμ`, and it is 8.4%.

### 7.3 c1 — the registered "comparable amount" check

Pre-registered: giving the parent the teacher's z should raise its self-divergence by about what the
swap removes. **It does, exactly**: `c1 = 0.0149` vs `KL_a − KL_b = 0.0149`, difference
`+0.0000 [−0.0064, +0.0063]`. The prediction holds numerically — but at a magnitude **within the
noise floor**, so this is a consistency check confirming the arithmetic of a null, not evidence for
the hypothesis.

### 7.4 Locality under the swap — the decisive row

Sibling-control `R` = (a teacher's KL on its own taught team) / (its siblings' KL on the same states),
over the 21 singly-taught teams. `1.00` = perfectly global. Recomputed under each condition:

| condition | R (n=9) | R (n=3) |
|---|---|---|
| **a** baseline | **1.8316 [1.5349, 2.1782]** | 1.7940 [1.4814, 2.1584] |
| **b** the swap | **1.8327 [1.5351, 2.1735]** | 1.8136 [1.4953, 2.1728] |
| c2 | 1.8431 [1.5462, 2.1816] | 1.8165 [1.5094, 2.1716] |
| dμ | 1.6637 [1.4405, 1.8804] | 1.6184 [1.3917, 1.8383] |
| d0 | 1.7246 [1.4979, 1.9424] | 1.6690 [1.4342, 1.8908] |

| paired contrast | n=9 | n=3 | verdict |
|---|---|---|---|
| **R_a − R_b** (the swap) | **−0.0011 [−0.0379, +0.0318]** | −0.0197 [−0.0725, +0.0308] | **NOT DETECTED** |
| R_a − R_dμ | +0.1679 [+0.0532, +0.3451] | +0.1756 [+0.0268, +0.3961] | SIGNIFICANT |
| R_a − R_d0 | +0.1070 [−0.0256, +0.2981] | +0.1250 [−0.0635, +0.3769] | NOT DETECTED |

**The z-swap does nothing to locality at all** — the registered "if `R_b` falls toward 1, z is what
made the teachers local" is answered flatly in the negative at both n. Deleting the whole
team-conditional FiLM (dμ) *does* reduce locality significantly, from 1.83 to 1.66 — but that is
**~0.17 of the 0.83 excess, roughly 20%**, and `R_dμ` still excludes 1 comfortably. **The other ~80%
of v8's locality is in the shared trunk and heads, and this probe does not explain it.**

### 7.5 Parameter mass — the same answer from the weight side

Fraction of `‖θ_T − θ_P‖²` living in each group. *Enrichment* = displacement share ÷ parameter share;
`1.0` is a group carrying exactly its fair share. `recon_head` is broken out because it is
**aux-loss-only and never fed forward**, so displacement there cannot change behaviour — folding it
into "the z path" would bias the z path's enrichment downward.

| model | z_encoder | film generators | recon_head *(inert)* | shared trunk + heads | ‖Δθ‖ |
|---|---|---|---|---|---|
| | 0.24% of params | 1.92% | 0.38% | 97.46% | |
| FLOOR c277178 | 0.02% (0.08×) | 0.01% (0.00×) | 0.05% | 99.93% (1.03×) | 6.97 |
| FLOOR c275758 | 0.06% (0.25×) | 0.04% (0.02×) | 0.63% | 99.26% (1.02×) | 8.33 |
| pool10 | 0.13% (0.55×) | 0.96% (0.50×) | 0.74% | 98.16% (1.01×) | 16.49 |
| semistall3 | 0.06% (0.26×) | 0.29% (0.15×) | 0.06% | 99.59% (1.02×) | 11.05 |
| defensive10 | 0.09% (0.38×) | 0.58% (0.30×) | 0.52% | 98.80% (1.01×) | 14.74 |

**The behavioural z path (encoder + generators) is 2.166% of the parameters and holds 0.35–1.10% of
the exploiters' squared displacement — enrichment 0.16–0.51×, i.e. consistently *below* its fair
share.** The exploiters put proportionally *less* of their change into the z path than a uniform
allocation would. They did put more there than plain continued training does (the floors sit at
0.01–0.05×), which agrees with §6's finding that the path was engaged — but engagement never became
localisation. Note the floors have much smaller total displacement, so the enrichment comparison
across rows is scale-sensitive and is offered as a direction, not a coefficient.

## 8. Verdict

| pre-registered question | answer |
|---|---|
| `f_b > 0.50` ⇒ z carries the specialisation | **NO.** `+0.0383 [+0.0212, +0.0554]` |
| `f_b < 0.20` ⇒ it lives in the shared weights | **YES — SIGNIFICANT**, at both n, on both slices |
| off-slice `f_b` changes little | yes; taught−untaught `+0.0314 [+0.0109, +0.0514]`, real but tiny |
| c1 raises P's self-divergence by a comparable amount | **yes, exactly** — and both sides are within floor |
| `R_b` falls toward 1 ⇒ z made the teachers local | **NO. `R_a − R_b` NOT DETECTED at both n** |
| mechanism precheck: is the swap a no-op by construction? | **the swap's *effect* is at the floor, but FiLM is NOT inert** — it is 175–247% of ‖h‖ and the exploiters grew it |

**The owner's hypothesis is refuted for the code and only weakly supported for the path.** FiLM was
available as a localisation mechanism, v8's exploiters demonstrably *used* the FiLM path — they
raised its magnitude ~30–40% and moved their codes ~3–6× further than ordinary training does — and the
result was a team-conditional behavioural effect **at the noise floor**. Substituting the parent's
code into a teacher changes the teacher less than two adjacent parent checkpoints differ, and leaves
the sibling-control locality exactly where it was.

**My own registered competing prediction was correct** (`f_b` below the 0.20 rail, z-sensitivity
small), including its stated consequence: **this probe does NOT explain v8's locality.** At most ~20%
of the excess `R` is attributable to team-conditional FiLM, via the *generator weights* rather than
the code, and the remaining ~80% sits in the shared trunk — exactly where the gen era also has its
weights, and therefore not an explanation of the era difference.

**Consequence for the arch→transfer line.** "Add a per-team FiLM code to the gen era" should **not**
be funded on the strength of the content_locality result. Three independent readings agree: the code
is swappable with no behavioural consequence; the parameter mass avoids the z path; and the earlier
LUT arm already found a free per-team code buys no performance (+0.024 CI [−0.016, +0.064]). What
survives is a smaller, sharper question this probe cannot answer — **why is the v8 trunk's
displacement team-local when the gen trunk's is not**, given both are shared weights under PPO.

## 9. Findings and hazards hit

1. **The fold's teacher is `best_model/best_model.zip`, not `final_model_interrupted.zip`** — §5.
   `content_locality`'s v8 arm used the wrong file, which halves semistall3's KL and understates its
   headline `R` as `1.4498` where the correct figure is **`1.8316 [1.5349, 2.1782]`**. Its verdict
   direction is unchanged and strengthened; its number and any delta derived from it are not. Found
   by a sibling probe, verified here from v8_14's own recorded argv plus the resolver's rung order.
2. **`content_locality`'s pooled `primary_A_era` bootstrap under-samples by one cell.** `own_all` has
   **23** entries (10+3+10, one team taught twice) but `bsT = rng.integers(0, 22, …)` draws indices
   in `[0,22)`, so cell 22 — `defensive10`'s last team, `65bfb2e8b4` — **never enters that CI**. The
   point estimate uses all 23. It affects only `primary_A_era`'s `kl_taught_ci95` / `L_ci95`;
   **`primary_B` (the headline `R`) is correctly sized** (`bsB` is built from `len(own)`). Verified by
   re-deriving the array lengths from that probe's own committed artifact.
3. **An aux-only readout inside a "path" biases a parameter-mass split.** `recon_head` is 0.38% of
   parameters and up to 0.74% of a teacher's displacement, but it is never fed forward, so including
   it in "the z path" (as the first pass did) lowers the enrichment of the part that can actually
   change behaviour. Split out here; both figures are in the JSON.
4. **A shim that installs itself and a caller that also wraps it produce a silent double wrap.** The
   first pass had `acid_shim` install `_ZShim` and `score_model` wrap it again, giving
   `zarch_encoder.real.real.*` state-dict keys. It was **behaviourally harmless** — the outer shim
   short-circuits and the inner stays disarmed, and condition `a` reproduced content_locality
   bit-for-bit through it — but it renames every z-path key and would silently reshape the parameter
   split if the two sides of that comparison ever wrapped differently. Fixed by making `acid_shim`
   the sole installer and asserting it is not already shimmed.
5. **Not a defect, but the reason this probe is worth more than its null:** the pre-registered
   precheck ("if z barely does anything, the swap is a no-op by construction") came back the *other*
   way. FiLM is a co-equal term in the head features, so "z does not carry the specialisation" is a
   claim about how an engaged mechanism was used, not an artifact of an unused one.

## 10. Limits, stated plainly

* **State distribution is parent-piloted.** Every teacher is scored on states the *parent* reaches.
  That is right for a divergence-from-parent statistic and it is what makes teachers comparable, but
  it is not the distribution the fold's rollouts see, and these levels must not be merged with the
  live `distill/collateral_kl_vs_parent` column.
* **Three teachers, 21 singly-taught teams, 2 siblings per team.** A 2-sibling mean is noisy; that
  widens `R` without biasing it. The `semistall3` row rests on **3** teams and its per-teacher CIs
  are wide — only the pooled figures are quoted in the verdict.
* **`d0` is off-manifold and is labelled so everywhere.** `z = 0` is never a real code. `dμ` is its
  on-manifold counterpart and is the number to quote for "how much is team-conditional".
* **This is a within-era probe.** It says where v8's exploiters put their change. It does **not**
  compare eras, and it cannot attribute the v8-vs-gen locality difference to architecture — v8 and
  the gen era differ in head type, refine loop, exploiter recipe, teacher count, pool and parent
  maturity as well as in zarch.
* **`R_a − R_dμ` is a ratio moving because a component was deleted from both numerator and
  denominator.** Reading it as "team-conditional FiLM supplies ~20% of the locality" is the intended
  reading; it is not a causal decomposition and the three ablations do not sum.
* **Enrichment across rows is scale-sensitive.** The floor checkpoints have half the total
  displacement of the teachers, so their much lower enrichment is a direction, not a coefficient.
* **Both n are reported; neither was chosen after seeing the other.** n=3 reproduces
  content_locality's canonical state batch exactly (that is the ACID gate); n=9 is a declared power
  extension. No conclusion changes between them.

## 11. Files

| file | what |
|---|---|
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | frozen before any state was generated; + AMENDMENT 1 (teacher file) |
| [`era_zswap.py`](era_zswap.py) | the measurement (run from the era-pinned checkout) |
| [`analyze.py`](analyze.py) | the readout — pure post-processing, runs in either tree |
| [`acid_vs_content_locality.py`](acid_vs_content_locality.py) | the two-directional ACID gate |
| [`verify_readme.py`](verify_readme.py) (+ `.log`) | **re-derives all 147 numbers this README quotes from the artifacts; exits non-zero on any drift** |
| [`run_era.sh`](run_era.sh) · [`run_both.sh`](run_both.sh) | the runners (mode + n) |
| `zswap_n3.json` · `zswap_n9.json` (+ `.log`) | **HEADLINE** — teachers from `best_model/best_model.zip` |
| `zswap_finalint_n3.json` · `zswap_finalint_n9.json` (+ `.log`) | **SECONDARY** — teachers from `final_model_interrupted.zip` |
| `analysis_n3.json` · `analysis_n9.json` (+ `.log`) | headline readouts |
| `analysis_finalint_n3.json` · `analysis_finalint_n9.json` (+ `.log`) | secondary readouts |

### Reproduce

```bash
cd designs/research_state/measurements/arch_transfer_2026-09-05/zswap
./run_both.sh best_model        # HEADLINE: n=3 then n=9, each ACID-checked and analysed
./run_both.sh final_interrupted # SECONDARY (the wrong-file arm, kept as a labelled control)
```

`run_era.sh` cds into `/tmp/v8rep_era` (READ-ONLY, commit `b13b30b2`) because `TeamLoader` resolves
`data/teams` relative to the CWD, exports `PYTHONPATH=<era>/src`, `PYTHONDONTWRITEBYTECODE=1`,
`GEN3AI_TIMEOUT_SCALE=12`, pins the four BLAS thread counts to 1, and runs at `nice -n 10`. Every
artifact is written back into this directory; nothing is written into the era checkout. Measured
beside a live training run at load ~16–22 on 16 cores — which is why the recipe carries an explicit
sim seed and `concurrency=1`, and why the ACID gate reproduces bit-for-bit across runs taken under
very different load.
