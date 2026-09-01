# M9 — THE MISSING CONTROL: does PLAIN TRAINING rob untaught teams?

**2026-08-31 · owner-ordered.** The control nobody had run.

**The dispatch's premise, stated as it was believed going in:** every gen-era fold robs untaught
teams (rev-4 −6.50pp here; rev-2 −7.06pp on probe Q's separately-drawn set), while M5 found —
unregistered — that rev-3's fold moves the policy *less* than what it called a matched no-fold
control. Those two cannot both stand as written: a fold below its own noise floor has nothing to
radiate. The missing cell is what ORDINARY TRAINING does to the same teams over the same span, and
nobody had ever run it.

**Both halves of that premise turn out to be wrong, in different ways.** M5's control was never
fold-free (§2.1), and rev-2's fold does not rob on *this* instrument at all (§3). What the probe
actually delivers is a plain-training null, the campaign's first replicate-derived noise floor, and
a narrowing of "the era robs" down to one generation's fold.

Artifacts beside this file: `plain_training_robbery_2026-08-31.json` (every number, every audit) ·
`plain_training_robbery_2026-08-31_tables.md` (machine-rendered; every table below is copied from
it verbatim) · `plain_training_robbery.py` (assembly + statistics, no battles, ~40 s) ·
`plain_training_robbery_inputs/` (every input banked, so this reproduces without the
session-scoped job directory).

---

## HEADLINE — a run with NO distillation robs untaught teams "significantly"

**One sentence carries this document, and everything else is commentary on it:**

> **`R2CTRL` — no teacher, no distillation loss, no team bias, byte-identical source and effective
> configuration to `R2PLAIN` — robs untaught teams by −4.56pp [−6.56, −2.31], z = −2.60.**
> Had that arm carried a `--distill-coef`, this campaign would have recorded it as a fold that
> robbed, and nothing in its published statistics would have said otherwise.

1. **PLAIN TRAINING'S EFFECT IS UNRESOLVABLE AT ONE RUN PER ARM.** The two identically-configured
   plain runs read **−0.37pp** and **−4.56pp** on the same 8 untaught teams — mean **−2.47pp**, 38%
   of rev-4's fold. The registered predictions score **NOT ESTABLISHED** and **FAILED** (§6): the
   quantity P1 asked about is *smaller than the spread between two draws of it*.
2. **THE NOISE FLOOR, MEASURED FOR THE FIRST TIME: −4.19pp [−6.94, −1.37] untaught (8 teams) and
   −3.70pp [−6.07, −1.26] taught (9 teams).** Two numbers from different cuts, team counts and game
   counts, agreeing at ~4pp. **No fold effect in this campaign has ever been required to clear it**,
   and several do not — rev-2 taught +1.33, rev-4 taught −3.67, rev-3's celebrated null.
3. **Rev-2's fold did NOT rob on the very set rev-4's REPRO-1 was scored on: +0.88pp
   [−1.62, +3.56], floor +2.33pp.** The ledger's *"rev-2 robbed −7.1pp untaught"* is probe Q's
   **different** team set. **"EVERY gen-era fold robs" does not survive within a single instrument.**
4. **Only rev-4's fold clears the floor** (−6.50pp z = −3.73; floor stratum −8.67pp z = −4.32) —
   1.6× the untaught floor pooled, 2.1× on the floor stratum. It is the one robbery in the table a
   control cannot account for.

**What this does to the mission's proposed reframe.** *"Training on a narrow team distribution robs,
and distillation is incidental"* is **NOT SUPPORTED as stated**: the plain-training draws (−0.37,
−4.56) bracket rather than reproduce the fold's −6.50, and the rev-2 fold is itself null-positive.
But the weaker and more damaging version **IS** supported — **an arm with no distillation at all can
produce a "significant" robbery indistinguishable from the ones the campaign has attributed to
folding.** The correct target of suspicion is not the distillation term; it is the **one-run-per-arm
design**.

⚠️ **Two campaign-wide corrections fall out of the audits, independent of every number above:**
neither of M5's "matched no-fold controls" is fold-free (§2.1), and the ledger's published fold
intervals are BINOMIAL rather than team-clustered (§1.2).

---

## 1. The arms, and why they are a matched contrast

**Suitable step-matched fold-free controls DO exist, and the mission's fallback (price a cell that
would create one) is not needed.** The campaign built them and never measured them on the untaught
cut. All four rev-2-era arms resume from the SAME checkpoint
(`models/ai_v9_29_rev1_0823/final_model.zip`), run the SAME budget (`--steps 28067760`, i.e.
25.22M → 28.07M ≈ 2.85M steps), carry the SAME `--seed 42`, and — verified below — ran on
**byte-identical `src/`**.

| arm | run | what it is | distillation flags |
|---|---|---|---|
| **REV1FIN** | `ai_v9_29_rev1_0823` `final_model.zip` | the common PARENT (24.99M) | — |
| **R2PLAIN** | `ai_v9_62_R2PLAIN_0827` | **PLAIN training**, no teacher of any kind | `--distill-coef 0.0`, no `--distill-teacher` |
| **R2CTRL** | `ai_v9_58_R2CTRL_0827` | designed as an ECOLOGY control; **as run, a REPLICATE of R2PLAIN** (§2.2) | `--distill-coef 0.0`, `--distill-teacher` (5 teachers) — INERT at coef 0 |
| **R2ACTION** | `ai_v9_59_R2ACTION_0827` | the rev-2 **FOLD** | `--distill-coef 0.1810`, `--distill-target action`, `--distill-topk 1`, `--distill-value-feat-coef 0.5` |

The argv diff is asserted at run time by the script, not assumed: `R2ACTION vs R2PLAIN` differs on
seven keys and **all seven are in the distillation family**
(`--distill-coef`, `--distill-teacher`, `--distill-target`, `--distill-topk`, `--distill-gate`,
`--distill-value-feat-coef`, `--rank-tripwire`); `R2CTRL vs R2PLAIN` differs on **exactly one**
(`--distill-teacher`).

Two further arms enter from banked data: **R4ACTION** (`ai_v9_76_R4ACTION_0830`, the rev-4 fold off
R2ACTION — the cell the scorecard's REPRO-1 failed on) and **R3SELF** (`ai_v9_72_R3SELF_0828`, the
zero-teacher-content self-fold off R2ACTION).

### 1.1 The instrument, unchanged

Battles come from `axis_split_untaught_arm.py`, **run unmodified**. The arm pilots one pinned team
against the FIXED reference `ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` drawing from
the validated 719-team pool; both sides stochastic; in-process **rust** bridge; CPU; **8 untaught
teams × 200 games**, `n` inherited from the instrument and not re-chosen. The two banked arms
(R2ACTION, R4ACTION) were collected by the byte-identical script, so every number in §3 sits on one
scale.

**One arm was adopted rather than duplicated.** The M1 axis-split probe was concurrently collecting
`REV1FIN` on this box with a **byte-identical** copy of the collector (verified by `diff` before
adoption; the duplicate this probe had launched was killed to return the core). Its artifact is
banked here under `plain_training_robbery_inputs/`.

### 1.2 `net`, `ordered`, and the common-team restriction

The composition discipline from `exploitability_taught_untaught_2026-08-31.md` is binding and is
satisfied **by construction rather than by correction**: every arm is measured on the *identical* 8
teams, so the common-team restriction is exact and no team-set reweighting is possible. On this
instrument `net` and `ordered` **coincide**, and saying so is more honest than printing one number
twice: the reference is ONE frozen model shared by every arm, and there is no `seniority` term
because no arm gets a head start against it. The rule exists to stop a *moving* reference from
hiding an effect; here the reference is pinned.

**Both CI conventions are reported, and the reason is a correction.** The mission required
team-clustered intervals on the grounds that this corpus's residual dispersion measures **2.52×**
binomial. **That figure does not transfer to this instrument, and the script measures rather than
assumes it:** across all 15 cells the observed per-team spread runs **0.62–1.31× the binomial-implied
spread, mean ≈ 0.98** (`dispersion_ratio` in the JSON). At n = 200 per team a win-rate contrast
carries ±7pp of binomial noise per team, which **swamps** the team-to-team heterogeneity the 2.52×
was measuring in the extraction corpus. Cluster and binomial intervals are therefore similar here,
and neither is a fig leaf for the other.

Separately, it emerged while reproducing the banked rev-4 cell that **the ledger's published
intervals are BINOMIAL** — `−0.0650 [−0.0994, −0.0306]` and `−0.0867 [−0.1263, −0.0471]` reproduce
to the third decimal on the binomial convention and not on the cluster one. Both columns are printed
throughout: binomial for exact comparability with what is banked, cluster as the interval that
carries the claim.

## 2. Two audits that changed what the campaign's other numbers MEAN

Both are computed by the script and land in the JSON; neither is a reading of prose.

### 2.1 NEITHER of M5's "matched no-fold controls" is fold-free

M5 §1.1 introduces a control pair — *parent ← an earlier checkpoint of the parent's own run* —
and describes it as "**ordinary training, no fold**". It is the denominator of every ratio in M5
§3 and therefore of its headline unregistered finding. **In both eras it is a fold span.**

| era | M5's control checkpoint | what it actually is |
|---|---|---|
| **gen** | `ai_v9_59_R2ACTION_0827/snapshots/snapshot_000024000000.zip` | **byte-identical** (md5 `df3d5620…`) to `ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip`. A resume RE-PUBLISHES the parent's self-play pool, so the file in the fold run's pool *is* rev-1's snapshot. The "4.07M control span" is rev-1's last ~0.99M of plain training **plus the ENTIRE 2.85M rev-2 fold**. |
| **v8** | `ai_v8_04_distill_4teacher_0722/checkpoints/checkpoint_269716291_steps.zip` | a checkpoint of a run whose own `cli_args` carry **`distill_coef = 1.0`** and a `--distill-teacher`. Its 7.46M "ordinary training" span is **distillation throughout**. |

**What survives and what does not.** M5's §3 ratios remain valid as *fold ÷ fold* comparisons —
"rev-3's fold moved the policy 0.91× as much as rev-2's fold did" is a real, useful fact, and the
v8 side is likewise fold-vs-fold. What does **not** survive is the sentence those ratios licensed,
and it is the sentence this whole probe was dispatched on:

> *"A fold that is below its own no-fold noise floor has nothing to radiate."* — M5 §6

**No no-fold floor was measured**, in either era, so nothing was below it. That floor is what §3
supplies for the first time. (This is the *recorded-versus-effective* class the campaign has now hit
four times — a plan is not a record, and a path is not a provenance.)

### 2.2 R2CTRL is not an ecology control — it is a REPLICATE of R2PLAIN, and that is a gift

R2CTRL was launched to hold the team distribution constant while folding no loss. **As run it did
not**: at its own commit `77f922e7`, `src/main/train/config.py` sets `args._distill_pairs = []`
and only fills it `if args.distill_coef and args.distill_coef > 0`, and **every** consumer of
`_distill_pairs` (the team-bias teambuilder, the teacher attach in `model_build.py`, the eval pin in
`matchup_spec.py`) is gated on that same emptiness or on the coefficient directly. So
`--distill-teacher` at coef 0 loaded no teacher, applied no team bias, and emitted no
`distill_mask`. The live tree's own `apply_distill_team_bias` docstring records the defect and names
this run: *"`ai_v9_58_R2CTRL_0827` asked for exactly that, got an effective bias of 0.0."*

The audit adds the fact the docstring does not: **`git diff 77f922e7 4714e0f8 -- src/` is EMPTY**,
so R2CTRL and R2PLAIN ran byte-identical source. Same parent, same budget, same seed, same code,
same effective configuration, two processes.

**That makes them a REPLICATE PAIR, and their difference is the campaign's missing noise floor** —
what two identically-specified plain-training runs do to each other on these meters. Nothing in the
fold table has ever been read against it. They demonstrably diverged (73,469 vs 67,667 recorded
episodes; 717 vs 685 distinct teams drawn), as 48 async workers, seedless bridge dice and an
evolving self-play pool guarantee.

**Measured, on two independent cuts: −4.19pp [−6.94, −1.37] untaught (8 teams × 200) and −3.70pp
[−6.07, −1.26] taught (9 × 300).** Different team sets, different counts, different game budgets,
agreeing at ~4pp. That is the yardstick §3–§5 use.

**⚠️ It is ONE replicate pair, n = 2 runs.** It gives a point estimate of run-to-run spread with no
interval on the spread itself. It is not a variance estimate; it is an existence proof that the
spread is not small — and, via `R2CTRL`'s z = −2.60 against the parent, that a single draw of a
distillation-free arm can present as a significant robbery.

## 3. THE HEADLINE CELL — untaught-8, control vs fold

8 untaught teams × 200 games × 4 complete arms = **6,400 battles** collected for this section
(REV1FIN and R2PLAIN new; R2ACTION and R4ACTION banked on the identical instrument).

### 3.1 The pooled cell

| contrast | teams | pooled WR base → arm | Δ | 95% CI (cluster over teams) | 95% CI (binomial) | z |
|---|---|---|---|---|---|---|
| **PLAIN TRAINING, draw 1** (R2PLAIN − REV1FIN) | 8 | 0.5737 → 0.5700 | **−0.37pp** | [−3.31, +2.31] | [−3.80, +3.05] | −0.21 |
| **PLAIN TRAINING, draw 2** (R2CTRL − REV1FIN) | 8 | 0.5737 → 0.5281 | **−4.56pp** | [−6.56, −2.31] | [−8.00, −1.12] | **−2.60** |
| rev-2 FOLD (R2ACTION − REV1FIN) | 8 | 0.5737 → 0.5825 | **+0.88pp** | [−1.62, +3.56] | [−2.54, +4.29] | +0.50 |
| distillation-specific (R2ACTION − R2PLAIN) | 8 | 0.5700 → 0.5825 | **+1.25pp** | [−1.87, +4.81] | [−2.16, +4.66] | +0.72 |
| **rev-4 FOLD** (R4ACTION − R2ACTION) | 8 | 0.5825 → 0.5175 | **−6.50pp** | [−9.75, −3.06] | [−9.92, −3.08] | **−3.73** |
| **THE NOISE FLOOR** (R2CTRL − R2PLAIN) | 8 | 0.5700 → 0.5281 | **−4.19pp** | [−6.94, −1.37] | [−7.62, −0.75] | −2.39 |

**Read rows 1 and 2 together — they are the same experiment, run twice.** `R2PLAIN` and `R2CTRL`
share a parent, a step budget, a seed, byte-identical source and (§2.2) an identical effective
configuration. One reads a clean null; the other reads a robbery whose interval excludes zero on
both conventions. **A single run of this arm can land anywhere in a 4.2pp band, and the campaign
has only ever run one.**

**The banked rev-4 row reproduces the ledger to the third decimal** (`−0.0650 [−0.0994, −0.0306]`,
`z = −3.70` published; `−6.50pp [−9.92, −3.08]`, `z = −3.73` here on the binomial convention). That
is the instrument check: the same script that produced the new rows regenerates the old one.

### 3.2 THE FLOOR STRATIFICATION — mandatory, and it is where the two effects separate

Registered stratum: `headroom_screen.json` (R2ACTION at n = 150) `> 0.55` ⇒ 6 floor teams
{ce35b736, 9909f2e9, 9d5f8458, f7ba5702, 90b94599, dbf81d8e}, 2 sub-floor {61590463, 92832108}.

| contrast | FLOOR (n=6) | sub-floor (n=2) | concentration? |
|---|---|---|---|
| **PLAIN, draw 1** (R2PLAIN) | **−0.08pp** [−4.03, +3.86] | −1.25pp [−8.12, +5.62]† | **NO** — ordering reversed |
| **PLAIN, draw 2** (R2CTRL) | **−3.42pp** [−7.38, +0.55] | **−8.00pp** [−14.89, −1.11]† | **NO** — ordering reversed |
| rev-2 FOLD | +2.33pp [−1.60, +6.26] | −3.50pp [−10.38, +3.38]† | no |
| **rev-4 FOLD** | **−8.67pp** [−12.60, −4.73] **z=−4.32** | **+0.00pp** [−6.91, +6.91]† | **YES** — the registered pattern |

† the sub-floor stratum is **2 teams**; a cluster bootstrap over 2 clusters has only three distinct
resample means, so the binomial interval is the honest one there and is what is quoted.

**The two effects have OPPOSITE stratum profiles, and that is the cleanest discriminator in the
document.** Rev-4's fold does what REPRO-2 was registered to detect — damage where competence
existed (−8.67pp floor), exactly zero where it did not. **Both** plain-training draws do the
reverse: their damage, such as it is, sits on the **sub-floor** pair (−1.25 and −8.00) and is
smaller on the floor (−0.08 and −3.42). Whatever ordinary training does to these teams, it is not
the fold's pattern.

The per-team floor deltas show why the plain rows are noise rather than a mechanism:

| floor team | REV1FIN | R2PLAIN | Δ plain | R2ACTION | Δ rev-2 fold |
|---|---|---|---|---|---|
| U_ce35b736 | 0.5650 | 0.4800 | **−8.50** | 0.5650 | 0.00 |
| U_9909f2e9 | 0.5750 | 0.6250 | **+5.00** | 0.6400 | +6.50 |
| U_9d5f8458 | 0.5650 | 0.5850 | +2.00 | 0.5450 | −2.00 |
| U_f7ba5702 | 0.5800 | 0.6150 | +3.50 | 0.6000 | +2.00 |
| U_90b94599 | 0.5750 | 0.5850 | +1.00 | 0.5850 | +1.00 |
| U_dbf81d8e | 0.5950 | 0.5600 | −3.50 | 0.6600 | +6.50 |
| **mean** | | | **−0.08** | | **+2.33** |

A 13.5pp spread scattered around zero is the signature of run-to-run noise, and §3.1's replicate row
says that noise is ~3.6pp — the same order as the per-team scatter here once binomial error
(±7pp/team at n = 200) is folded in.

## 4. The TAUGHT-9 companion (already banked — no battles)

Every arm had **already** been measured on the standing fold-quality meter — the 9 teams rev-2's
fleet actually taught, 9 × 300 games, same fixed reference, same collector family — so this half of
the table cost nothing to produce and had simply never been differenced this way.

| contrast | teams | pooled WR base → arm | Δ | 95% CI (cluster over teams) | 95% CI (binomial) | z |
|---|---|---|---|---|---|---|
| **PLAIN, draw 1** (R2PLAIN − REV1FIN) | 9 | 0.5619 → 0.5381 | **−2.37pp** | [−5.48, +0.48] | [−5.02, +0.28] | −1.76 |
| **PLAIN, draw 2** (R2CTRL − REV1FIN) | 9 | 0.5619 → 0.5011 | **−6.07pp** | [−8.93, −3.26] | [−8.72, −3.43] | −4.49 |
| **fold_rev2** (R2ACTION − REV1FIN) | 9 | 0.5619 → 0.5752 | **+1.33pp** | [−1.85, +4.67] | [−1.30, +3.97] | +0.99 |
| **distillation_specific** (R2ACTION − R2PLAIN) | 9 | 0.5381 → 0.5752 | **+3.70pp** | [+0.56, +6.70] | [+1.06, +6.34] | +2.75 |
| **THE NOISE FLOOR** (R2CTRL − R2PLAIN, two replicates) | 9 | 0.5381 → 0.5011 | **−3.70pp** | [−6.07, −1.26] | [−6.36, −1.05] | −2.73 |
| **fold_rev4** (R4ACTION − R2ACTION) | 9 | 0.5752 → 0.5385 | **−3.67pp** | [−5.93, −1.26] | [−6.30, −1.03] | −2.72 |
| **self_fold_zero_content** (R3SELF − R2ACTION) | 9 | 0.5752 → 0.4856 | **−8.96pp** | [−11.15, −6.41] | [−11.60, −6.33] | −6.67 |

Three readings, in order of how much they cost the campaign:

1. **The noise floor is 3.70pp here and it is "significant" on the campaign's own conventions.** Two
   runs that differ in nothing an optimizer can see separate by 3.70pp with a CI excluding zero at
   z = −2.73. **Every taught-side fold effect in the table is at or inside that magnitude**:
   rev-2's fold is +1.33pp (smaller than the floor), rev-4's is −3.67pp — **equal to it, to 0.03pp,
   with an almost identical z**. A single-run-per-arm design cannot separate those from a re-roll of
   the same recipe. It **replicates on the untaught cut at −4.19pp** (§3.1), which is what makes it
   a property of the training process rather than of one meter.
2. **Plain training moves this meter downward on both draws** (−2.37pp and −6.07pp), so the
   plain-training hop spans −2.4 to −6.1pp taught and −0.4 to −4.6pp untaught. Consistently
   negative, never resolvable.
3. **The two effects that clear the floor comfortably are the zero-content self-fold (−8.96pp) and
   distillation-beyond-replicate (+7.41pp).** Those are the only taught-side statements this corpus
   supports at one run per arm.

⚠️ **Do not read the `R2CTRL − R2PLAIN` row as an "ecology effect."** §2.2 shows the two arms are
configurationally identical as run, so the row measures run-to-run noise and is labelled as such.
The `ecology_*` contrast names surviving in the JSON are the pre-audit names of these cells and are
kept only so the artifact's keys stay stable; their meaning is the replicate one.

## 5. The three-way decomposition

The mission asked which of three legs carries the robbery: plain-training drift, self-distillation
harm, or the real fold. On the untaught-8 the rev-2-era decomposition is **exactly additive by
construction** (all three arms share one parent and one meter):

```
rev-2 fold outcome   =  plain-training drift  +  distillation-specific
      +0.88pp        =       -0.37pp          +       +1.25pp          (all 8)
      +2.33pp        =       -0.08pp          +       +2.42pp          (floor, n=6)
```

| leg | untaught-8 | floor (n=6) | taught-9 | verdict |
|---|---|---|---|---|
| **plain-training drift**, 2 draws | −0.37 / **−4.56** | −0.08 / −3.42 | −2.37 / **−6.07** | **UNRESOLVED** — the two draws straddle the question |
| **distillation-specific** (R2ACTION − R2PLAIN) | +1.25pp n.s. | +2.42pp n.s. | **+3.70pp SIG** | positive everywhere; its only significant reading HELPS |
| **the rev-2 fold, net** (R2ACTION − REV1FIN) | +0.88pp n.s. | +2.33pp n.s. | +1.33pp n.s. | **no robbery in the rev-2 era at all** |
| **the rev-4 fold** (R4ACTION − R2ACTION) | **−6.50pp z=−3.73** | **−8.67pp z=−4.32** | −3.67pp z=−2.72 | **the only robbery that clears the floor** |
| self-distillation, zero content (R3SELF − R2ACTION) | *not run* | *not run* | **−8.96pp z=−6.67** | destructive, taught-side only (§8.7) |
| **replicate noise floor** (R2CTRL − R2PLAIN) | **−4.19pp** | −3.33pp | **−3.70pp** | the yardstick every row above must clear |

**Which leg carries the robbery.** Not plain training *as a demonstrated cause* — its two draws
(−0.37, −4.56) bracket zero-to-substantial, so the honest verdict is **unresolved**, and its stratum
profile is the fold's inverse (§3.2). Not the rev-2 fold — null-positive on every cut. **What
remains is rev-4's fold**, the one contrast that clears the ~4pp replicate floor on the untaught cut
(1.6× pooled, 2.1× on the floor stratum) and the only one whose stratum profile matches the
registered damage signature.

**But the same table is what forbids reading that conclusion too hard.** A ~4pp floor means rev-4's
−6.50pp is a ~1.6× effect, not a ~6× one, and it has been measured **once**.

**The honest gap in the decomposition.** Rev-4's hop has **no matched plain control** — there is no
`R4PLAIN` forked from R2ACTION for 4.3M steps without distillation. So the −6.50pp is *fold-minus-
parent*, not *fold-minus-plain*, and this document establishes only that the analogous quantity one
generation earlier was zero. **Pricing that cell: one 4.3M-step run off `R2ACTION/final_model.zip`
with `--distill-coef 0` and no `--distill-teacher` (≈2 GPU-h at the fleet's measured 3M ≈ 1.5 GPU-h),
plus one 8×200 untaught arm (~75–150 min on one niced core).** It is the single measurement that
would convert "rev-4's fold robbed" from a parent-relative statement into a causal one, and at ~3%
of a fleet's budget it is the cheapest decisive purchase available.

## 6. Predictions, scored

**P1 — "Plain training ALSO robs untaught teams, at more than half the fold's magnitude — i.e. most
of the measured 'fold robbery' is ordinary narrow-distribution training." → NOT ESTABLISHED, and the
instrument cannot establish it at one run per arm.**

Plain training was measured **twice**, on the same 8 teams, with identical configuration:

| draw | Δ vs REV1FIN | as a fraction of rev-4's −6.50pp | verdict against P1's ≤ −3.25pp bar |
|---|---|---|---|
| `R2PLAIN` | −0.37pp [−3.31, +2.31] | 6% | **misses** |
| `R2CTRL` | −4.56pp [−6.56, −2.31] | 70% | **clears** |
| mean | **−2.47pp** | **38%** | misses |

**One draw clears the bar and the other misses it.** Scoring this FAIL on `R2PLAIN` alone — which is
exactly what a one-run design would have done, and what I would have written had `R2CTRL` not
finished — would have been an artifact of which run happened to be labelled the control. The
defensible statement is: *the mean is 38% of the fold, below the predicted >50%, but the spread
between two draws (4.19pp) is larger than the distance from either draw to the bar.* **P1 is not
refuted; it is unanswerable on this evidence**, and that is itself the finding.

**P2 — "The floor stratum shows the same concentration pattern for plain training as for the fold."
→ FAILS, on both draws, in the same direction.**
The fold's signature is concentration: rev-4 reads **−8.67pp floor / +0.00pp sub-floor**. Both plain
draws inverted it — **−0.08 / −1.25** and **−3.42 / −8.00**, floor damage *smaller* than sub-floor
damage each time. Two independent runs agreeing on the reversal is stronger evidence than either
alone. The per-team floor deltas show why: `R2PLAIN`'s are **−8.50, +5.00, +2.00, +3.50, +1.00,
−3.50** (§3.2) — a 13.5pp spread scattered about zero, the signature of noise rather than of a
mechanism that prefers competent teams.

**The unregistered result is the one worth carrying:** the replicate pair puts a number on that noise
for the first time — **−4.19pp untaught, −3.70pp taught** — and it is the yardstick every fold
verdict in this campaign has been quoted without.

## 7. What this does to the campaign's framing

The mission asked for an explicit statement under each outcome. Two of the three inputs to that
statement do not depend on the untaught cell at all, so they are stated first and stand whatever §3
turned out to be.

### 7.1 The fold table has never been read against a noise floor

The campaign's central claim — *gen-era folds rob, v8's gifted, the sign flip is total* — rests on
**one run per arm**. §2.2 produces the first replicate pair the campaign has ever had, and on the
taught meter two runs that differ in nothing an optimizer can see separate by **−3.70pp with a CI
excluding zero (z = −2.73)**. Rev-4's taught-side fold effect is **−3.67pp (z = −2.72)**. Those are
the same number to 0.03pp.

This does **not** say the fold effects are noise. It says the corpus **cannot currently tell**,
because no fold effect was ever required to clear a replicate. That is a different and far more
repairable problem than "the fold does nothing", and it has a price: one `R2PLAIN`-style run per
claim worth keeping. Against a 60M-step fleet, a 2.85M control is ~5% of the budget.

### 7.2 M5's floor argument is withdrawn

§2.1 shows both of M5's controls are fold spans. The reading *"rev-3's fold is below its own no-fold
noise floor, so it has nothing to radiate"* therefore had no floor under it — it was fold-to-fold
throughout. M5's **measurements** stand and remain useful as fold-vs-fold ratios; the **mechanism
sentence** does not. The floor it believed it had is what §3 and §4 supply for the first time, and
it is not small.

### 7.3 The headline cell, and the reframe it does not license

**The outcome that landed is not one of the two the mission anticipated.** It asked what follows if
plain training robs at comparable magnitude (fold exonerated) or if it does not (fold implicated).
The answer is *neither*: **plain training robs by −4.56pp in one draw and −0.37pp in another**, so
the instrument the whole campaign is built on cannot resolve the question it was asked. The precise
post-M9 statement:

> On the untaught-8, over the rev-2 hop, **the fold does not rob (+0.88pp) and plain training's
> effect is unresolved (−0.37 / −4.56)**. One generation later rev-4's fold robs −6.50pp against a
> **measured ~4.2pp replicate floor**. The robbery is therefore **not a property of folding** — it
> is a property of *that fold*, at ~1.6× the noise, measured once.

That is materially different from *"every gen-era fold robs, v8 gifted, the sign flip is total."*
The sign flip is not between eras; it is between **two folds inside the gen era**, one of which this
document shows to be null, against a floor nobody had measured.

**Three consequences, in order of how much they should change behaviour.**

1. **NO FOLD VERDICT SHOULD BE QUOTED WITHOUT A REPLICATE — this is the actionable one.** The floor
   is ~4pp, and three headline fold effects sit at or inside it (rev-2 taught +1.33, rev-4 taught
   −3.67, rev-3's celebrated ±null). A no-distillation arm produced a z = −2.60 "robbery". The fix
   is a second control run per claim: **2.85M steps ≈ 1.5 GPU-h, ~5% of a fleet's budget** — far
   cheaper than the fleets whose conclusions currently rest on single draws.
2. **The v8-vs-gen framing loses one of its two poles.** If rev-2's fold is null-positive on untaught
   teams, "gen-era folds rob" was carrying rev-4 plus probe Q's differently-drawn rev-2 cell, not a
   property of the era. Before more effort goes into *why v8 gifted and gen robs*, the gen side needs
   re-establishing — on this instrument half of it is not a fact.
3. **The maturity hypothesis is untouched, and saying so is the honest position.** Nothing here
   speaks to v8's 277M-step parent. M9 removes *narrow-distribution training* as a demonstrated cause
   and damages *"the era robs"* as a premise; it does not adjudicate what made v8 different. The
   fallback tree's tier-2 maturity cell remains the live experiment — but it should be run with two
   control arms, not one.

**What M9 does NOT show.** It does not show folds are harmless (rev-4's is not, and R3SELF's
zero-content fold is actively destructive at −8.96pp taught). It does not show the untaught-8 is the
right team set — it is balance/stall-heavy by construction (§8.8). And it does not explain the −6.50:
identifying *what* rev-4's fold did that rev-2's did not is the successor question, and §5 prices the
one run that would make it answerable.

## 8. Limits — read before quoting

1. **ONE replicate pair.** The noise floor rests on n = 2 runs. It is an existence proof that
   run-to-run spread is not small, not an estimate of its distribution — there is no interval on
   the spread itself. Anyone who wants the campaign's fold verdicts to survive should fund a
   second replicate pair before funding another fleet; that is the cheapest decisive purchase in
   this document.
2. **One lineage, one hop length.** Every arm here is rev-1 → 2.85M steps, or R2ACTION → ~4.3M.
   Nothing establishes that the plain-training effect scales with span, and the mission's own
   maturity hypothesis (v8's parent was 277M steps) is untouched: a 2.85M hop off a 25M parent may
   behave nothing like a 15M hop off a 277M one.
3. **The reference is an ANCESTOR of every arm, and REV1FIN is only ~0.99M past it.** The meter is
   "win rate against a frozen ancestor while piloting a pinned team". A drop from REV1FIN to
   R2PLAIN is a real loss of competence *on that meter*; it does not distinguish "the policy got
   worse" from "the policy drifted away from what beats this particular frozen opponent". Both are
   what the campaign has always meant by robbery — the whole taught/untaught table is built on this
   meter — but the distinction is not resolved here and no arm was scored against a second
   reference.
4. **The floor stratum is registered but its variable is the rev-4 fold's PARENT.** Membership comes
   from `headroom_screen.json` (R2ACTION at n = 150, `> 0.55`), fixed by scorecard REPRO-2 before
   any of these arms was measured. That makes it independent of every arm this document differences
   — which is exactly why it was preferred to each arm's own measured WR, whose use would induce
   regression to the mean. It does mean the stratum is defined by a model that is the CHILD in the
   rev-2 contrast and the PARENT in the rev-4 one. Using one membership for both keeps the two cells
   on one scale; it is not a claim that the cut is optimal for either.
5. **n = 200 per team, 8 teams.** Per-team deltas carry roughly ±9pp of binomial noise, so no
   per-team row is evidence; the pooled and stratified rows are the powered reads. The sub-floor
   stratum is **2 teams** — a cluster bootstrap over 2 clusters has three distinct resample means
   and its interval should not be read as an interval. The binomial column is the honest one there.
6. **Arms are not battle-paired.** Both sides act stochastically and the sim dice are free; the arms
   share only the per-team opponent-team draw *sequence*. This is inherited from the instrument
   (changing it would put the new arms on a second scale) and it is why every contrast is an
   unpaired two-proportion difference rather than a paired one.
7. **`R3SELF` was NOT run on the untaught cut.** Its self-distillation leg is carried from the
   banked TAUGHT-9 meter (−8.96pp) and from the campaign's own coverage read (−11.4pp, ledger
   2026-08-30). Pricing the missing cell: one arm on this instrument is 8 × 200 battles ≈ 75–150 min
   on one niced core (the wide range is contention — this collection ran beside a 20-arm GPU fleet at
   load ~50). It is the single cheapest addition to this table.
8. **The team set is structurally balance/stall-heavy.** Probe Q's selection note records that
   hyper-offense cannot be represented in an untaught set because all four hyper-offense sample teams
   are in the taught union. That caveat rides every number on this instrument, this document
   included.
9. **Not measured here: whether the plain-training drop is the SAME drop the fold produces.** A
   common magnitude is not a common mechanism. Establishing that would need the behavioural
   fingerprint (M4/M5's meters) run on `R2PLAIN` — which, given §2.1, is also the measurement that
   would give M5 the no-fold denominator it believed it had.

## 9. Reproducing

```bash
export PYTHONPATH=$PYTHONPATH:src
M=designs/research_state/measurements

# the battle arms — ONE core each, ~75-150 min depending on box load. models/ is READ-ONLY.
# GEN3AI_TIMEOUT_SCALE is not cosmetic here: at load ~50 the unscaled poke-env stall watchdog
# killed an arm mid-collection.
GEN3AI_TIMEOUT_SCALE=8 nice -n 15 python $M/axis_split_untaught_arm.py \
    models/ai_v9_29_rev1_0823/final_model.zip  REV1FIN  untaught_REV1FIN.json 200 3
GEN3AI_TIMEOUT_SCALE=8 nice -n 15 python $M/axis_split_untaught_arm.py \
    models/ai_v9_62_R2PLAIN_0827/final_model.zip R2PLAIN untaught_R2PLAIN.json 200 3
GEN3AI_TIMEOUT_SCALE=8 nice -n 15 python $M/axis_split_untaught_arm.py \
    models/ai_v9_58_R2CTRL_0827/final_model.zip  R2CTRL  untaught_R2CTRL.json  200 3

# the assembly + every audit (no battles, ~40 s)
nice -n 15 python $M/plain_training_robbery.py --out $M/plain_training_robbery_2026-08-31
```

`untaught_R2ACTION.json` / `untaught_R4ACTION.json` and the six `taught9_*.json` meters are banked
under `plain_training_robbery_inputs/` together with `headroom_screen.json` (the stratification
source) and the four `*.argv` files, so nothing here depends on a session-scoped job directory. In a
linked worktree the run dirs must be reachable as `models/<run>` — symlinking the four runs from the
main checkout is enough, and `models/` is never written.
