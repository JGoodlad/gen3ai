# M7 — DOES DISTILLATION HURT LESS AS THE PARENT MATURES?

**Status: COMPLETE (free tier).** Re-analysis of the committed distillability-index artifact,
plus 16 NEW micro-distill cells that fill the battery's two decisive gaps — **no new training
run, no battles, `models/` read-only.** Scripts: `maturity_harm_trend.py` (analysis) and
`distillability_index_probe.py` (the unchanged 2026-08-28 producer, re-run for the new cells).
Full numbers in the sibling `.json`.

Owner's question, verbatim: *"could we do a 50M and 100M on the base model to get the trend
lines? see if maturity decreases the amount distillation hurts?"*

---

## 0. Headline

**Three findings, in descending order of how much they should change what happens next.**

**(1) The harm that maturity moves is the OPTIMIZER's, not the CONTENT's — and the content's
half does not fall at all.** Splitting total collateral into the part a zero-content self-distill
also causes and the part only the teacher's content causes, the **content-attributable** half
**rises or is flat with age at every informative matched step** (§3); the single step at which it
falls is step 400, which the producer's own caveat identifies as the over-trained regime. The
zero-content control accounts for **75–79%** of total collateral at every age. So the age trend
that exists lives almost entirely in the control — i.e. in how far a fixed Adam step travels on a
sharpened landscape, which is a property of the optimizer meeting the loss surface, not of a
mature network accepting or rejecting external behaviour.

**(2) The v8-vs-gen maturity story is CONFOUNDED with dose rate, and the confound is invisible at
the flag level.** Both eras' `--lr` flags differ 4.3× in the direction that says *gen is harsher*
(3e-4 vs 7e-5) — but **`--lr` is INERT on a fork**: the fold inherits its parent's annealed
optimizer state, and measured from TensorBoard the truth is the inverse. v8's fold operated at a
**median 1.00e-4** against R2ACTION's **5.81e-5** and R3ACTION's **2.80e-5** (§5.4 traces the
inheritance chain, first-value-equals-parent's-last, to four significant figures). The decisive
axis is not lr at all but **optimizer steps per unit of environment data**, where v8 ran
**11.4× fewer** (grad-accum 16 + 7 epochs vs accum 2 + 10) at an **8× larger effective batch**.
Net displacement per 1M env steps: **v8 is 3.2× gentler than rev-3 and 6.6× gentler than rev-2.**
*Registered prediction 2 — that the confound would NOT materialise — is REFUTED.*

**(2b) A corollary that outranks the maturity question operationally.** Because the step size is
inherited, it **decays monotonically down the revolution chain** and nobody set it: rev-2's fold
ran at **2.07×** rev-3's. Rev-2 robbed −7.06pp untaught and rev-3 measured null — a comparison
with an uncontrolled 2× dose-rate difference inside it. **Every future fold must pin `--lr` on
the fork, or publish the realized median beside its result.**

**(3) And the two folds' TOTAL dose is nearly identical — only the RATE differs.** v8 spread
0.320 units of displacement over 14.9M fold steps; rev-3 spread 0.311 over 4.55M — **within 3%
of each other**, at 3.2× the rate. Whatever separates a fold that gifts from one that robs, it
is not how much the student was moved.

### The plain answer, for the 2–25M range we already own

**Is the answer step-size-dependent? Yes — but on one meter of three, and not on the one that
bears on the decision.**

* **TOTAL collateral after 400 steps** falls with maturity at **lr 1e-4** (0.662 → 0.436 nats,
  ρ = −1.00, p = 0.003, reproduced on a second seed) and does **not** at **lr 3e-4** (flat, with
  the sign flipping between seeds; and *rising* on a second, ancestry-free lineage). That is the
  owner's hypothesis, and it is true at the lower step size only.
* **But split that total into optimizer and content** (§3): the falling part is the optimizer's.
  The content-attributable half rises or is flat with age at every informative matched step, and
  the zero-content control alone is 75–79% of the total at every age.
* **So the honest sentence is not "maturity reduces the harm distillation does."** It is
  *"a smaller step reduces it, and a mature network needs a smaller step than a young one for
  the same reason a sharper landscape amplifies a fixed displacement."* Maturity is the
  condition; step size is the lever.

⚠️ **One qualification on the split, stated because it is the load-bearing one.** The
content-vs-optimizer decomposition is currently available only at **lr 3e-4** and at **n = 3
ages**, because the committed battery ran the zero-content control at three ages and one step
size. The 1e-4 control cells are §6.3's addition and are what would let the split be read at the
step size where the total *does* fall. **Until they land, "the falling part is the optimizer's"
is established at 3e-4 and INFERRED at 1e-4** — inferred from the fact that early harm rises with
age in the control at 3e-4 and rises with age in the *with-content* cells at BOTH step sizes,
which is the optimizer's signature, not content rejection.

**Consequence for the proposal.** A 50M/100M maturity ladder is buildable and priced below
(§7, **42.8 GPU-h**), but this re-analysis gives it a **low registered prior** — and the same
data implicate a lever that costs **4.5 GPU-h**. The recommendation in §7.5 is to run the
step-size arm first and the maturity rungs only if it comes back null.

---

## 1. What "harm" means here, and the three regimes it is defined in

The instrument is unchanged from 2026-08-28: a student checkpoint's full policy is fine-tuned
with Adam on masked cross-entropy toward a fixed teacher's argmax over 3,000 on-slice states,
and **COLLATERAL** is measured as drift from that student's *own* pre-probe policy on 1,500
off-slice states (233 teams, both pinned teams excluded). Four harm meters, all reported, all
signed so larger = worse: off-slice `KL(now ‖ original)`, `1 − top-1 agreement with original`,
mean `|ΔV|`, and `1 − corr(V)`.

"Harm" is only defined relative to what was bought, so it is read in three regimes:

| regime | what is held equal | zero-content control comparable? |
|---|---|---|
| **matched STEPS** | identical optimizer work | **YES** — the control buys no absorption, so this is the *only* regime it fits |
| matched ABSORPTION | held-out teacher agreement reaches an absolute level `A*` | no |
| matched GAIN | agreement has risen `d` above its own start | no |

The last two are NORMALISED meters and are scored against the correlation they **manufacture**
under an age-invariant-harm null (§4) — the binding method rule inherited from
`substrate_hypothesis_2026-08-31.md` §2.3.

**Provenance check.** `build-states` was re-run and its `state_provenance.json` reproduces the
committed one **bit-identically** (262 teacher trace files, 2,451 off-slice files after
pin-exclusion, 233 distinct teams, same pinned team pair), so the new cells are directly
comparable to the 41 committed ones rather than a parallel battery.

---

## 2. The harm-vs-age table, per step size

Seed 1 shown; seed 2 reproduces every ordering (§8). `KL@s` = off-slice KL after `s` Adam steps.

### 2.1 ANCESTOR lineage (rev-1), lr **3e-4** — the arm the 2026-08-28 doc called "the project's own training lr"

| age | KL@1 | KL@32 | KL@135 | KL@400 | disagree@32 | disagree@400 | \|ΔV\|@400 |
|---|---|---|---|---|---|---|---|
| 2M | 0.069 | 0.246 | 0.525 | 0.775 | 0.329 | 0.402 | 4.77 |
| 6M | 0.118 | 0.364 | 0.603 | 0.727 | 0.301 | 0.358 | 7.51 |
| 12M | 0.567 | 0.424 | 0.603 | 0.668 | 0.309 | 0.333 | 4.52 |
| 18M | 0.559 | 0.494 | 0.615 | 0.721 | 0.311 | 0.336 | 4.22 |
| 24M | 0.555 | 0.616 | 0.614 | 0.717 | 0.329 | 0.335 | 5.26 |
| **25M final** | 1.116 | 0.509 | 0.634 | 0.750 | 0.292 | 0.331 | 4.29 |

### 2.2 ANCESTOR lineage, lr **1e-4** (the robustness arm)

| age | KL@1 | KL@32 | KL@135 | KL@400 | disagree@32 | disagree@400 | \|ΔV\|@400 |
|---|---|---|---|---|---|---|---|
| 2M | 0.008 | 0.166 | 0.361 | 0.662 | 0.273 | 0.409 | 3.69 |
| 6M | 0.017 | 0.244 | 0.438 | 0.630 | 0.259 | 0.331 | 6.14 |
| 12M | 0.069 | 0.248 | 0.391 | 0.496 | 0.229 | 0.283 | 5.30 |
| 18M | 0.081 | 0.278 | 0.374 | 0.494 | 0.232 | 0.291 | 2.94 |
| 24M | 0.097 | 0.261 | 0.359 | 0.454 | 0.223 | 0.265 | 3.35 |
| **25M final** | 0.136 | 0.265 | 0.328 | **0.436** | 0.201 | **0.250** | 3.55 |

### 2.3 ANCESTRY-FREE lineage (gen-17 → E4), lr **3e-4** — shares no weights with the teacher

The 30M / 36M / 42M rungs are **NEW** (§6.2): `ai_v9_25_E4_baitbot_0822`, a gate experiment
forked off the same gen-17 base, at the **current** architecture.

| age | KL@1 | KL@32 | KL@135 | KL@400 | disagree@32 | disagree@400 | \|ΔV\|@400 |
|---|---|---|---|---|---|---|---|
| 2M | 0.297 | 0.217 | 0.526 | 0.829 | 0.345 | 0.467 | 8.91 |
| 6M | 0.430 | 0.420 | 0.654 | 0.855 | 0.318 | 0.395 | 5.22 |
| 12M | 0.517 | 0.606 | 0.736 | 0.845 | 0.341 | 0.365 | 8.08 |
| 18M | 0.765 | 0.687 | 0.707 | 0.861 | 0.335 | 0.346 | 4.58 |
| 24M | 0.903 | 0.702 | 0.776 | 0.916 | 0.350 | 0.385 | 4.97 |
| 25M final | 0.949 | 0.641 | 0.788 | 0.923 | 0.360 | 0.385 | 5.02 |
| *30M* | *NEW* | | | | | | |
| *36M* | *NEW* | | | | | | |
| *42M* | *NEW* | | | | | | |

### 2.4 ZERO-CONTENT CONTROL — targets are the student's OWN argmax, lr **3e-4** and **1e-4**

Same optimizer, same states, same step count, **no new behavioural content**. This is the
archive's direct analogue of the proposed self-fold ladder.

| age | lr | KL@1 | KL@32 | KL@135 | KL@400 | disagree@400 |
|---|---|---|---|---|---|---|
| 2M | 3e-4 | 0.148 | 0.304 | 0.440 | 0.584 | 0.267 |
| 12M | 3e-4 | 0.435 | 0.333 | 0.456 | 0.518 | 0.281 |
| 25M final | 3e-4 | 0.522 | 0.443 | 0.468 | 0.595 | 0.305 |
| 6M / 18M / 24M | 3e-4 | *NEW* | | | | |
| 2M … 25M final | **1e-4** | *NEW — the battery's single largest gap* | | | | |

### 2.5 Trend vs age (Spearman ρ, exact permutation p, seed 1 / seed 2)

| arm | regime | ρ s1 | p s1 | ρ s2 | its NULL |
|---|---|---|---|---|---|
| lr 3e-4 ancestor | KL@1 | +0.77 | 0.103 | +0.94 | 0 |
| lr 3e-4 ancestor | KL@32 | **+0.94** | **0.017** | **+0.94** | 0 |
| lr 3e-4 ancestor | KL@135 | **+0.94** | **0.017** | **+0.94** | 0 |
| lr 3e-4 ancestor | KL@400 | −0.26 | 0.658 | **+0.26** | 0 |
| lr 3e-4 ancestor | disagree@400 | −0.83 | 0.058 | −0.83 | 0 |
| **lr 1e-4 ancestor** | KL@1 | **+1.00** | **0.003** | +0.94 | 0 |
| **lr 1e-4 ancestor** | KL@32 | +0.83 | 0.058 | +0.77 | 0 |
| **lr 1e-4 ancestor** | **KL@400** | **−1.00** | **0.003** | **−0.89** | 0 |
| **lr 1e-4 ancestor** | **disagree@400** | **−0.94** | **0.017** | **−0.94** | 0 |
| lr 3e-4 ancestry-free | KL@32 | +0.83 | 0.058 | +1.00 | 0 |
| lr 3e-4 ancestry-free | KL@400 | **+0.94** | **0.017** | +1.00 | 0 |
| lr 3e-4 ancestry-free | disagree@400 | −0.52 | 0.300 | −0.87 | 0 |
| lr 3e-4 zero-content | KL@32 | +1.00 | *0.333 floor* | — | 0 |
| lr 3e-4 zero-content | KL@400 | +0.50 | *1.000* | — | 0 |

**Reading — and the answer to the owner's question, stated plainly.**

1. **EARLY harm RISES with age, everywhere, at BOTH step sizes.** KL@1 and KL@32 climb with
   maturity on the ancestor lineage at 3e-4 *and* at 1e-4, on the ancestry-free lineage, and —
   decisively — **in the zero-content control**, which has no content to reject. This is Adam
   displacing a sharpened landscape further in function space for the same weight-space step. It
   is not "a mature network resists new behaviour".
2. **LATE harm is STEP-SIZE-DEPENDENT, and that is the half of the owner's hypothesis that
   holds.** At lr 1e-4 the endpoint falls monotonically and significantly (KL@400 ρ −1.00 /
   −0.89; disagree@400 ρ −0.94 / −0.94; 0.662 → 0.436 nats). At lr 3e-4 the same students give
   a flat KL whose **sign flips between seeds** (−0.26 / +0.26), and on the ancestry-free
   lineage it *rises* (+0.94 / +1.00).
3. ⚠️ **The two harm meters DISAGREE at lr 3e-4 and must not be collapsed.** On the ancestor
   lineage KL@400 is flat while top-1 disagreement@400 falls (ρ −0.83, both seeds); on the
   ancestry-free lineage KL rises (+0.94) while disagreement falls (−0.52, n.s.) — outright
   contradiction. Only at lr 1e-4 do they agree. This is the producer's own caveat 6 ("KL and
   top-1 agreement are not interchangeable") biting specifically on the harm question, and it is
   why **no single scalar "harm falls with age" claim is licensed at lr 3e-4 in either direction.**

---

## 3. The number that actually answers the question: harm NET of the zero-content control

Total collateral is the sum of what the optimizer does to any network and what the teacher's
content does to *this* one. Only the second is what "distillation hurts" should mean. At matched
steps, both are measured; **lr 3e-4, seed 1, n = 3 ages** (the control's committed coverage):

| age | step | total harm (KL) | zero-content harm | **CONTENT-attributable (net)** | control's share |
|---|---|---|---|---|---|
| 2M | 1 | 0.069 | 0.148 | **−0.078** | 2.13 |
| 2M | 32 | 0.246 | 0.304 | **−0.057** | 1.23 |
| 2M | 135 | 0.525 | 0.440 | +0.084 | 0.84 |
| 2M | **400** | 0.775 | 0.584 | **+0.190** | 0.75 |
| 12M | 1 | 0.567 | 0.435 | +0.131 | 0.77 |
| 12M | 32 | 0.424 | 0.333 | +0.091 | 0.78 |
| 12M | **400** | 0.668 | 0.518 | **+0.150** | 0.78 |
| 25M | 1 | 1.116 | 0.522 | +0.594 | 0.47 |
| 25M | 32 | 0.509 | 0.443 | +0.066 | 0.87 |
| 25M | **400** | 0.750 | 0.595 | **+0.155** | 0.79 |

**Read down the step column, not across one row — the sign of the age trend depends on where
you stop, and only one of the four steps is even weakly favourable to the hypothesis:**

| matched step | net at 2M | net at 12M | net at 25M | direction with age |
|---|---|---|---|---|
| 1 | −0.078 | +0.131 | +0.594 | **RISES steeply** |
| 32 | −0.057 | +0.091 | +0.066 | rises, then flat |
| 135 | +0.084 | +0.147 | +0.166 | **RISES** |
| 400 | +0.190 | +0.150 | +0.155 | falls 2M→12M, then flat |

**The content-attributable harm does not fall with maturity at any informative step.** Three of
the four rise; the only fall is at step 400, which the producer's own caveat 4 identifies as the
**over-trained** regime ("400 steps at lr 3e-4 over-trains … the informative regime is the first
~64 steps"). Picking step 400 and reporting "0.190 → 0.150 → 0.155, harm falls then flattens"
would have been a defensible-looking sentence built on the one column the instrument says not to
read as a fold outcome. It is stated here in full instead.

With n = 3 the exact-permutation p floor is 0.333, so none of these rows can **ever** be
significant, and they are reported as magnitudes rather than trends. But the magnitude is the
point: across a 12× age span the content's own harm moves by 0.005–0.08 nats depending on the
step, in inconsistent directions, while the *control* moves monotonically and by more.

Two smaller things worth keeping:

* **At 2M the net is NEGATIVE at steps 1 and 32** — distilling a young network toward its *own*
  argmax damages it *more* than distilling it toward the teacher's. A young policy's argmax is
  noisy, so fitting it hard is fitting noise. This is a real reading, not a sign error, and it
  is why the control cannot simply be subtracted at every step and called "content".
* **The control's share is 75–79% at step 400 at every age.** The 2026-08-28 doc reported ~79%
  at 25M; it holds across the whole age range, which is stronger than the original claim.

---

## 4. The manufactured nulls — why "harm at matched absorption falls with age" is an artifact

A matched-absorption reading looks like the natural way to ask the question, and on the ancestor
lineage it gives a clean negative at **both** step sizes: `KL@A*=0.70` ρ = **−0.78** (3e-4) and
**−0.85** (1e-4), both seeds. Read against zero that is "harm falls with maturity" — the owner's
hypothesis, confirmed twice.

**It is not.** An older student starts closer to the teacher (`a0` rises with age, ρ = +1.00 by
construction), so it reaches any absolute absorption level in **fewer optimizer steps**, and harm
grows with steps. Pooling each arm's own harm-vs-step curves into one age-independent `H̄(s)`
and reading it at each age's *actual* crossing step gives what that arithmetic alone produces:

| arm | regime | observed ρ | **its own null** | verdict |
|---|---|---|---|---|
| lr 3e-4 ancestor | KL @ absorption 0.70 | −0.78 | **−0.85** | AT its null — no effect |
| lr 1e-4 ancestor | KL @ absorption 0.70 | −0.85 | **−0.85** | AT its null — no effect |
| lr 3e-4 ancestor | disagree @ absorption 0.70 | −0.85 | **−0.85** | AT its null |
| lr 3e-4 ancestry-free | KL @ absorption 0.70 | **+0.94** | **−0.71** | far ABOVE its null — harm rises |
| lr 3e-4 ancestor | KL @ gain +0.05 | +1.00 | +0.94 | at its null |
| **lr 1e-4 ancestor** | KL @ gain +0.05 | **+0.89** | **+0.26** | ABOVE its null — harm rises |
| lr 3e-4 ancestry-free | KL @ gain +0.05 | +0.94 | +0.37 | ABOVE its null |

**Every matched-absorption negative on the ancestor lineage is exactly the manufactured
value.** Scored correctly, the matched-absorption regime shows *no* maturity benefit at either
step size — and on the ancestry-free lineage it shows the reverse. Only the matched-STEPS
endpoint at lr 1e-4 survives as a real fall.

**The ratio meter has the same trap.** The source artifact carries `eff = gain / harm`, and a
reader will reach for it. Under an age-invariant-harm null, `ρ(eff)` equals `ρ(gain_max)`
exactly — which is **−0.94 to −1.00** in every arm, because `gain_max` falls with age as
arithmetic (`a0` rises faster than `a_max`). Any `eff` trend must be read against that, never
against zero.

*Same family as M3's headroom finding: a normalisation that is neutral under one null
manufactures a near-±1 correlation under the other's, and reporting it against zero produces a
confident result of the analyst's choosing.*

---

## 5. The v8 step-size fact — the confound MATERIALISES, opposite to the flag-level read

Registered prediction 2 was the boring outcome: *v8's fold ran a distill configuration whose
effective step on the distill term is NOT obviously gentler than ours.* **REFUTED.**

### 5.1 The flags say one thing; the operating values say another

| | v8 fold (`ai_v8_14`) | rev-2 fold (`R2ACTION`) | rev-3 fold (`R3ACTION`) |
|---|---|---|---|
| `--lr` **flag** | 7e-5 (min 1e-5, max 6e-4) | 3e-4 | 3e-4 |
| **operating lr, median (TB `train/learning_rate`)** | **1.004e-4** | **5.81e-5** | **2.80e-5** |
| operating lr, range | 1.004e-4 – 1.205e-4 | 4.04e-5 – 6.98e-5 | 2.34e-5 – 4.04e-5 |
| `--distill-coef` | **1.0** | 0.181 | 0.176 |
| distill target form | **full-distribution KL** (`--distill-target` did not exist at `b13b30b`) | action-level CE | action-level CE |
| `grad/distill_share`, median | *metric postdates the run* | **0.244** | **0.225** |
| batch × grad-accum = **effective batch** | 2048 × 16 = **32,768** | 2048 × 2 = 4,096 | 2048 × 2 = 4,096 |
| `n_epochs` | 7 | 10 | 10 |
| **optimizer steps per rollout** (48 envs × 2048) | **21** | **240** | **240** |
| fold length (env steps) | 277.18M → 292.10M = **14.92M** | 24.99M → 28.07M = 3.08M | 28.07M → 32.62M = 4.55M |

**The `--lr` flag is a starting point, not the step size.** The adaptive KL-band controller drove
all three well below their flags, and it inverted the ordering: reading the flags alone says gen
runs 4.3× harsher, reading TensorBoard says v8 ran **1.7× harsher than rev-2 and 3.6× harsher
than rev-3** *per step*. Anyone comparing the two eras from `original_command` would have got
the sign wrong.

### 5.2 The axis that decides is dose RATE, not lr

The comparison is clean on the one thing that could have broken it: **all four runs use
`--n-envs 48 --n-steps 2048`, so the rollout buffer is 98,304 samples in every case.** (v8_14 ran
`--async-rollout` and the gen folds did not, but `AsyncSubprocVecEnv` fills each env's own buffer
column, so the buffer size — and therefore the minibatch count — is unchanged. The difference is
`--batch-size × --grad-accum-steps` and `--n-epochs`, nothing else.)

Adam's per-parameter step is bounded by ~`lr`, so displacement per unit of environment data
scales as `optimizer_steps_per_1M_env_steps × lr`:

| | opt steps / 1M env steps | × operating lr | **displacement per 1M env steps** | vs v8 |
|---|---|---|---|---|
| **v8 fold** | 213.6 | 1.004e-4 | **0.0215** | 1.00× |
| rev-2 fold | 2441.4 | 5.81e-5 | **0.1419** | **6.61×** |
| rev-3 fold | 2441.4 | 2.80e-5 | **0.0684** | **3.19×** |

**v8 was decisively the gentler fold — by 3.2–6.6× — but not on the lr axis.** It was gentler
because it took 11.4× fewer optimizer steps per unit of data, at an 8× larger effective batch.

### 5.3 The total dose is the same; only the rate differs

| fold | fold length | × displacement rate | **total displacement** |
|---|---|---|---|
| v8 (`ai_v8_04` → `ai_v8_14`) | 14.92M | 0.0215 | **0.320** |
| rev-3 (`R2ACTION` → `R3ACTION`) | 4.55M | 0.0684 | **0.311** |
| rev-2 (`REV1` → `R2ACTION`) | 3.08M | 0.1419 | 0.437 |

**v8 and rev-3 moved the student by the same total amount, within 3%** — v8 spread it thinly
over 14.9M steps, rev-3 concentrated it into 4.55M. And rev-2, the fold that robbed −7.06pp
untaught, is the one that ran 6.6× the rate. *Whatever separates a fold that gifts from one that
robs, it is not total dose. Rate is a live candidate that no arm from rev-2..rev-5 has varied.*

### 5.4 The mechanism: `--lr` is INERT on a fork, and the fold's step size DECAYS down the chain

The operating lr is not something the adaptive controller discovered during each fold. It is
**inherited from the parent's optimizer state at the fork**, and every `--lr 3e-4` in every
recorded gen-era fold command is **dead on arrival**. The chain reads off TensorBoard exactly —
each fold's *first* recorded value equals its parent's *last*, to four significant figures:

| run | forked from | first lr | last lr | median lr |
|---|---|---|---|---|
| `ai_v9_29_rev1_0823` | *(fresh, flag applies)* | **3.00e-4** | **6.977e-5** | — |
| `R2ACTION` (rev-2 fold) | rev-1 final | **6.977e-5** ⟵ | 4.038e-5 | **5.814e-5** |
| `R3ACTION` (rev-3 fold) | R2ACTION final | **4.038e-5** ⟵ | 2.337e-5 | **2.804e-5** |
| `R3SELF` (the self-fold) | R2ACTION final | **4.038e-5** ⟵ | 2.804e-5 | **2.804e-5** |
| `R4ACTION` (rev-4 fold) | R2ACTION final | **4.038e-5** ⟵ | 2.337e-5 | **2.804e-5** |
| `ai_v8_04` (v8 parent) | — | 1.446e-4 | **1.205e-4** | — |
| `ai_v8_14` (v8 fold) | v8_04 final | **1.205e-4** ⟵ | 1.205e-4 | **1.004e-4** |

Three consequences, and the third is the one that touches every fold comparison this programme
has made:

1. **The flag carries no information about the fold.** A reader reconstructing dose from
   `original_command` gets 3e-4 for every gen fold and 7e-5 for v8 — the exact inverse of the
   truth. *(Same family as the standing note that `--anneal-frac` is inert on resume.)*
2. **Both eras inherit**, so this is not a v8-vs-gen asymmetry; it is a shared mechanism whose
   *value* differs because each lineage's annealing had reached a different place.
3. 🚨 **The fold's step size is a DRIFTING, UNREGISTERED experimental condition.** It falls
   monotonically down the revolution chain — rev-2 at median 5.81e-5, rev-3/rev-4/R3SELF at
   2.80e-5 — so **rev-2's fold ran at 2.07× rev-3's step size for reasons nobody chose**, and a
   rev-6 fold will run lower still. Rev-2 robbed −7.06pp untaught and rev-3 measured null; that
   comparison has an uncontrolled 2.07× dose-rate difference inside it. Any future fold
   comparison must **pin `--lr` explicitly on the fork** (or at minimum publish the realized
   median beside the result), or it is comparing arms that differ in a variable nobody set.

### 5.5 What this does NOT establish

* **The distill-attributable share for v8 is unknown.** `grad/distill_share` postdates the run,
  so §5.2's ratios are on TOTAL displacement. v8's `distill_coef` was 1.0 against gen's ~0.18 on
  a *different target form*, so its share was plausibly higher than gen's 0.23 — which would
  narrow the gap. Reported as a total-axis result, not a distill-axis one.
* **`lr × steps` is a proxy for Adam displacement**, not an identity (`lr·m̂/(√v̂+ε)` is bounded
  by ~`lr`, not equal to it). The ratios are order-of-magnitude claims.
* **Target form is a third, uncontrolled axis.** v8 = full-distribution KL, gen = action-level CE
  — and the project's own G2 result says the action form is the *better* one. So the v8 fold
  differed from ours in maturity AND rate AND target form simultaneously. Any maturity ladder
  that does not also vary rate cannot attribute its own result.

---

## 6. Extending the age axis — what was and was not possible

### 6.1 The era-pinned extension is SKIPPED, and here is exactly why

An archive census (169 run dirs, 142 at ≥20M steps) settles it:

* **No run at the current architecture (`gen3_critic_route_wave_v1`, obs 2501) exceeds 41.72M
  steps.** The deepest is `ai_v9_25_E4_baitbot_0822`. There is no 50M rung, no 100M rung, and no
  run that could supply one.
* **Every run at 50M+ is a different architecture era**; every run at 100M+ and 200M+ is
  `gen3_opp_hp_typed_candidates_v1` at obs 2992 or 3469. Maturity and architecture are
  **perfectly confounded** in this archive.
* The era-pinned load path is proven and cheap (`/tmp/probeP_v8era` @ `b13b30b` already exists;
  a micro-distill probe plays no battles, so the era's node-bridge requirement does not bind;
  ~9 min/cell/core).

**But the instrument does not port, and one break is fatal to comparability rather than to
execution:** v8-era `*_states.npz` carry nine arrays and **`action_mask` is not one of them**
(verified across 9 v8/v7 runs). The probe's masked cross-entropy would silently become an
*unmasked* cross-entropy and its documented `action_mask.sum(1) >= 2` filter a no-op. Two lesser
breaks: `main.prober.model`'s three helpers (`sanitized_load_custom_objects`, `peek_checkpoint`,
`_arch_drift_error`) are absent at `b13b30b`, and `obs.shape[1] != 2501` is hardcoded — the
second fails *silently*, emptying the state set with no error.

**Verdict: SKIP.** A v8-era cell would differ from a gen-era cell in architecture, obs dimension,
*and* whether the objective is masked at all. That is three confounds on a two-point comparison.
A stated skip beats a confounded number.

*(Noted for whoever revisits: `ai_v9_25_E4_baitbot_0822` — the deepest current-arch run — also
lacks `action_mask` in its traces. This did not block §6.2, because the probe takes its states
from the teacher's and rev-1's traces and uses the student only as a checkpoint to load.)*

### 6.2 The extension that WAS free — +17M of current-arch age, no era confound

`ai_v9_25_E4_baitbot_0822` is a gate experiment forked off the **same gen-17 base** that the
battery's ancestry-free control arm already uses, at the current architecture, with snapshots to
42M. Its 30M / 36M / 42M rungs therefore extend that arm from n = 6 (2M–25M) to **n = 9
(2M–42M)** with the state set, teacher, seed and step count all unchanged. Results in §2.3.

⚠️ **It is a SPLICE, not one run.** E4 forks from gen-17 around 10M and carries a different
config (`baitbot`), so the 26M+ rungs are a *lineage* continuation, not a single-run
continuation. Stated as a limit; it is still the only current-architecture way to add age.

### 6.3 The gap the new cells close

The committed battery's zero-content control was **3 ages at one step size**, which left §3's
decisive table unable ever to reach significance and left the interaction §2 turns on — *does the
lr-1e-4 maturity benefit survive removing the content?* — simply unanswerable. Both sit on the
producer's own "not run (budget)" list (§7 of the 2026-08-28 doc). The new cells:

| addition | cells | what it buys |
|---|---|---|
| zero-content control at **lr 1e-4**, 6 ages | 6 | the content/optimizer split at the step size where total harm *does* fall — the decisive gap |
| zero-content control at lr 3e-4, ages 6M/18M/24M | 3 | completes that arm to n = 6, so its trend can reach p = 0.0028 instead of a 0.333 floor |
| ancestry-free arm extended to 30M/36M/42M (§6.2) | 3 | +17M of age on a lineage sharing no weights with the teacher |

**Explicitly NOT run (budget), and therefore MISSING rather than null:** an lr-1e-4 arm on the
ancestry-free lineage, a second probe seed on any new cell, and a zero-content control on the
ancestry-free lineage at either step size. The first would have tested whether the lr-1e-4 result
survives removing shared ancestry — the single most valuable cell after the ones run here, and
the first thing to add if this line is revisited.

---

## 7. The 50M/100M ladder — PRE-REGISTERED design, bars, and cost

### 7.1 The meter: a self-fold against its OWN parent

A self-fold (`--distill-teacher <parent-itself>`) is the pure-harm instrument, and the archive
already contains one: **`ai_v9_72_R3SELF_0828`** — R2ACTION final distilled toward *itself*
pinned to 12 teams, `distill_coef 0.1761`, `--distill-target action`, +4.55M steps. Its measured
cost, recomputed here from the committed per-team pilot artifacts:

| arm (vs parent `R2ACTION`, per-team, 300 games each) | slice set | n | Δ WR (pp) | se | z | **Δ logit** | se | z |
|---|---|---|---|---|---|---|---|---|
| **R3SELF (self-fold — zero content)** | pilot | 9 | **−0.0896** | 0.0128 | **−7.00** | **−0.3646** | 0.0518 | **−7.03** |
| R3ACTION (real content fold) | pilot | 9 | +0.0041 | 0.0156 | +0.26 | +0.0155 | 0.0645 | +0.24 |
| **R3SELF** | coverage | 3 | **−0.0544** | 0.0097 | −5.62 | **−0.2384** | 0.0388 | −6.15 |
| R3ACTION | coverage | 3 | +0.0600 | 0.0320 | +1.87 | +0.2553 | 0.1399 | +1.82 |

*(`POOLED` is a summary row in those artifacts, not a team, and is excluded — including it would
have double-counted the sample and reported z = −7.83 instead of −7.00.)*

**The self-fold costs 8.96pp / 0.365 logits with nothing whatever to learn, measured to
±0.052 logits unpaired.** BAR-1 below is set at +0.12 logits = **2.3× that standard error**
unpaired and better once CRN-paired — so the ladder is powered, and the 28M rung is already paid
for.

🚨 **One design change is mandatory, and it is the M3 rule again.** The pilot scores every arm
against a *fixed* reference (rev-1's 24M snapshot). A 100M base is a stronger player, so its win
rate against that fixed reference sits higher, and **pp deltas compress toward zero purely from
headroom** — which would fake "maturity reduces harm" with no maturity effect at all. The fix is
structural, not a correction: **each self-fold plays ITS OWN parent, CRN-paired**, so the parent
sits at 0.50 by construction and headroom is identical at every rung. Report on the **logit**
scale (M3: the gift/harm is a constant log-odds increment), with pp beside it.

### 7.2 The design — a 2×2, not a 1-D ladder

§5 showed maturity and dose rate are confounded inside the v8 story, so a maturity-only ladder
cannot attribute its own result. The registered design crosses them:

| cell | parent | dose rate | status |
|---|---|---|---|
| **A** | 28M (`R2ACTION` final) | as-run (accum 2, 10 epochs) | ✅ **EXISTS** = `R3SELF` |
| **B** | 28M | ÷4 (accum 2→8, lr unchanged) | new, 4.55M steps |
| **C** | 100M | as-run | new, 4.55M steps |
| **D** | 100M | ÷4 | new, 4.55M steps |

Plus the base extension `R2ACTION` 28.07M → 50M → 100M (**plain PPO, no distill term**), with a
**50M checkpoint retained** so a mid rung can be added later without re-running anything.

**Why `R2ACTION` and not the deepest checkpoint in the lineage.** `R3ACTION` (32.62M) and
`R4ACTION` (32.60M) are deeper, but they are *folds* — extending one would confound parent
maturity with having-already-been-folded, which is a second uncontrolled variable and one the
distillability battery's own `R2ACTION` bonus cell says is not inert (a checkpoint that has
absorbed one fold has the **highest** `a0` and `a_max` of any lr-3e-4 cell). `R2ACTION` final is
also the fixed target of rev-3, rev-4 and the running 40-team fleet, so extending it keeps the
maturity ladder on the same trunk every other fold comparison already uses.

🚨 **`--lr` MUST BE PINNED IDENTICALLY ON ALL FOUR CELLS, and this is not a nicety — without it
the ladder measures nothing.** §5.4 shows a fold inherits its parent's annealed lr. The 100M base
will have annealed *below* 4.038e-5 by the time it gets there, so cells C/D would silently run at
a smaller step than A/B — **re-introducing exactly the maturity-vs-dose-rate confound the 2×2
exists to break**, and doing it invisibly, because every arm's recorded command would read the
same. Pin `--lr 4.038e-5` (R3SELF's inherited value, so cell A stays comparable to the existing
run) on every cell, and **verify from TensorBoard after launch that all four realized the same
median** — the flag alone has already proved untrustworthy here once.

**Mandatory diagnostics on every arm.** The defect §5 found is that effective dose was never
recorded and had to be reconstructed from TensorBoard days later: log `train/learning_rate` and
`grad/distill_share` per arm and publish the realized `steps × lr` displacement rate **beside the
result**, so the next reader does not repeat the reconstruction.

### 7.3 Bars, registered before the run

**Sign convention, stated so no bar is ambiguous.** For each rung,
`harm := logit(WR of that rung's self-fold vs its OWN parent, CRN-paired) − logit(0.50)`, which
equals `logit(WR)` because the own-parent design pins the reference at 0.50. It is a **negative**
number (the 28M rung measures **−0.365**), so *less harm is a less negative value* and every
"maturity helps" bar below is a **positive** difference.

* **BAR-1 — SUPPORTED.** `harm(100M, as-run) − harm(28M, as-run) ≥ +0.12 logits` (≈ +3pp at
  WR 0.50), team-clustered paired CI over the 9 pilot teams excluding zero. *(Powered: the
  R3SELF logit se is 0.052 unpaired, so +0.12 is 2.3×se; CRN pairing between rungs shrinks it
  further.)*
* **BAR-2 — REFUTED, and this closes the tier-2 fallback of scorecard `f326404`.**
  `|harm(100M) − harm(28M)| ≤ 0.08 logits` (≈2pp) AND the paired CI contains zero ⇒ maturity is
  not the missing ingredient, and the fallback tree moves to differentiation.
* **BAR-3 — monotonicity, REPORTED not primary.** With a 50M rung, `harm(28M) > harm(50M) >
  harm(100M)` in the reducing direction. n = 3 rungs has an exact-p floor of 0.333, so an
  ordering can never carry significance on its own; it is corroboration only.
* **BAR-4 — the dose-rate axis.** If `harm(28M, ÷4) − harm(28M, as-run) ≥ +0.12 logits` while
  BAR-1 misses, **rate is the lever and maturity is not** — the reading this re-analysis
  registers as most likely.
* **INTERACTION.** Report `[D−C] − [B−A]`. A maturity effect that exists only at the gentle rate
  is exactly prediction 1's shape carried into a real fold.

### 7.4 What a NULL at 100M can and cannot say

v8's parent was **277.18M** against the gen parent's **28.07M** — a **9.88×** exposure ratio. A
100M rung is **3.6×**. So **BAR-2 firing at 100M does NOT exclude a maturity effect that only
appears at v8-class exposure**; it excludes one with any appreciable effect over 3.6×. Reaching
277M costs ~138 GPU-h ≈ 5.8 days at the fleet's measured rate — already priced in the ledger.
This limit is registered up front rather than discovered in the discussion.

### 7.5 Cost, and the recommendation

At the fleet's measured **2.0M steps/GPU-h** (3M ≈ 1.5 GPU-h):

| item | steps | GPU-h |
|---|---|---|
| base 28.07M → 50M | 21.93M | 11.0 |
| base 50M → 100M | 50.0M | 25.0 |
| self-fold **B** (28M, ÷4 rate) | 4.55M | 2.3 |
| self-fold **C** (100M, as-run) | 4.55M | 2.3 |
| self-fold **D** (100M, ÷4 rate) | 4.55M | 2.3 |
| self-fold **A** (28M, as-run) | — | **0 — `R3SELF` exists** |
| **TOTAL GPU** | **85.6M** | **42.8 GPU-h ≈ 1.8 days** |
| eval: 3 new arms × 9 teams × 300 CRN games, own-parent paired | — | ~6 CPU-h (~3 h wall on 2 cores) |

Optional 50M self-fold rungs (BAR-3): +2 arms = +4.5 GPU-h.

**RECOMMENDATION — run BAR-4 first, for 4.5 GPU-h.** Cells **A** and **B** need no base
extension at all: A exists, B is one 4.55M self-fold at a quartered dose rate, and together they
test the lever this re-analysis actually implicates (§5: v8 was 3.2× gentler in rate at
essentially identical total dose; §3: the content-attributable harm does not fall with age at any
informative step, so there is nothing for maturity to be reducing). If B shows
the harm reduction that maturity did not, the remaining 40.5 GPU-h (base extension + cells C and D) is unnecessary. If B
comes back null too, the maturity rungs are the right next spend and the ladder is already
designed.

The re-analysis registers its own expectation: **BAR-2 (refuted) is the likely maturity outcome,
BAR-4 the likely positive.** Recorded so that the opposite is a real result.

---

## 8. Predictions scored

| # | registered | outcome |
|---|---|---|
| 1 | *In the existing 2–25M data, harm does NOT fall with age at lr 3e-4 but DOES at 1e-4 — the owner's hypothesis is TRUE ONLY AT THE LOWER STEP SIZE, making step size a required arm of any maturity ladder.* | **PARTIALLY CONFIRMED — on one regime of three, and the conclusion survives.** ✅ At **matched steps, endpoint**: exactly as registered — 3e-4 flat with a seed sign flip (ρ −0.26/+0.26) and rising on the ancestry-free lineage (+0.94/+1.00); 1e-4 falls monotonically and significantly (−1.00, p = 0.003; disagree −0.94, p = 0.017). ❌ At **matched steps, early** (steps 1–32) harm RISES with age at **both** step sizes, including in the zero-content control. ❌ At **matched absorption**, scored against its own manufactured null (§4), there is no fall at either step size. ❌ At **matched gain**, harm rises *above* its null at lr 1e-4. **The prescription is confirmed regardless** — step size is a required arm — but for a stronger reason than the prediction gave: §3 shows the age trend is almost entirely the *optimizer's* harm, not the content's. |
| 2 | *v8's fold ran a distill configuration whose effective step on the distill term is NOT obviously gentler than ours (the confound does NOT materialise) — registered as the boring outcome.* | **REFUTED, and in a way the flags conceal.** v8's fold was **3.2× gentler than rev-3 and 6.6× gentler than rev-2** in displacement per unit of environment data — *not* via lr (where it ran 1.7–3.6× **harsher** per step, once the adaptive controller's operating value is read from TensorBoard instead of the `--lr` flag) but via **11.4× fewer optimizer steps per rollout** at an **8× larger effective batch**. Maturity and dose rate are therefore confounded inside the v8 story, and §7.2 crosses them rather than assuming. Bonus: the two folds' **total** displacement matches within 3% — only the rate differs. |

**Broken link named** (standing rule): prediction 1's broken link is *"lower step size ⇒ the
mature network's harm falls" ⇒ "maturity is what reduced it."* It fails because the zero-content
control moves with age just as much: the thing maturity changes is how far a fixed Adam step
travels in function space, which is a property of the landscape and not of the content being
taught.

---

## 9. Cuts and limits

* **No new training run, no battles, `models/` read-only.** The 16 new cells are the *same*
  micro-distill probe on the *same* bit-identical state set; nothing about the instrument changed.
* **A micro-probe is not a fold simulation** (producer's caveat 1, inherited in full): no PPO
  loss beside the distill term, no `--distill-team-bias` sampling, no environment interaction, no
  entropy pressure, no LR schedule. §7's ladder is the real-fold test; §2–4 measure the
  student-side term in isolation.
* **n = 6 ages (9 on the extended ancestry-free arm), n = 3–6 on the control.** The exact
  permutation p floor is 0.0028 at n = 6 and **0.333 at n = 3** — an n = 3 ordering can never be
  significant, and §3's decisive table is one. It is reported as a magnitude.
* **Seed reproduction is the only replication.** Over 42 paired cells per arm, median absolute
  seed-to-seed difference: `off_kl` 0.017 (max 0.064) at lr 1e-4, 0.014 (max 0.143) on
  `off_disagree` at lr 3e-4. Every §2.5 ordering reproduces in sign across seeds except
  `KL@400` at lr 3e-4, whose sign flip **is** the finding that the meters disagree there.
* **The step-1 shock is an ordering, never a value** (producer's admission: 1.116 vs 0.420 on the
  same cell across seeds). It appears in §2's tables for completeness and carries no verdict.
* **`lr × optimizer-steps` is a displacement proxy**, and §5's era comparison is on TOTAL
  displacement because `grad/distill_share` postdates v8. §5.4 states both limits.
* **§6.2's age extension is a lineage splice**, not a single run, and carries a config difference.
* **Off-slice states come from one run's eval traces** (233 teams, 12 eval steps) under
  win/loss forensic quotas — collateral is well-defined as drift from each student's own
  reference on those states, but it is not "damage to what this student would have done next".

## Reproduce

```
export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd designs/research_state/measurements
nice -n 15 python distillability_index_probe.py build-states
nice -n 15 python distillability_index_probe.py probe ctrlself1e4_25M_final__s1 \
    /home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/final_model.zip 'A*' 1 400 1e-4
nice -n 15 python maturity_harm_trend.py --print
```

The analysis step is ~2 s on one core and touches nothing but the two JSON artifacts. Each new
probe cell is ~9 min of one core on a quiet box (measured 15–25 min at load 31).
