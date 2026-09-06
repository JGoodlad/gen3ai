# Content locality **v2** — the same measurement, on the checkpoints the folds actually loaded, and against the teachers' own origin

**2026-09-06. Offline. No training, no launcher, no server. CPU only, `nice -n 10`, BLAS pinned to
one thread, beside a live training run (load 18–34 on 16 cores).** ~40 CPU-minutes of battles.

This re-runs [`../content_locality/`](../content_locality/README.md) with **three corrections and
nothing else**. That probe is left exactly as it stands — it is the record of what was measured;
the ledger already carries the correction. Read this file for the numbers.

| | correction | why |
|---|---|---|
| **1** | teachers resolved by the **training path's own resolver** | v1 scored a network no fold ever distilled from — for all **19/19** teachers |
| **2** | the gen side reported under **two references** | the gen exploiters do not fork from the fold parent; v1's reference carried a shared constant offset |
| **3** | the cluster bootstrap is **sized from its own array** | v1's v8 pooled-L resampled a 23-cell array with indices in `[0, 22)` |

[`PREREGISTRATION.md`](PREREGISTRATION.md) holds the predictions, with an honest note on when it
was transcribed.

---

## The three corrections, in detail

### 1 — Checkpoint resolution: **19 of 19 teachers were the wrong file**

`main/train/model_build.py` loads a `--distill-teacher` like this:

```python
from agents.training.fixed_opponent_pool import _resolve_zip_and_config as _rzc_d
...
_zip_d, _cfg_d, _ = _rzc_d(_tp, None)   # run-dir → (zip, config)
```

and `_resolve_zip_and_config`'s directory rung is
`best_model/best_model.zip` → `final_model.zip` → `best_model.zip`. Every fold in this batch named
a run **directory** — read from each fold's own `metadata.json`, not assumed:

```
ai_v9_162_TCUNFA_0903  cli_args distill_teacher:
   models/ai_v9_92_R5F00_0831:*;models/ai_v9_94_R5F02_0831:*; … ;models/ai_v9_106_R5F14_0831:*
ai_v8_14_distill3_0725 cli_args distill_teacher:
   …/ai_v8_09_pool10_exploiter_0723:*;…/ai_v8_06_semistall_3team_exploiter_0722:*;…/ai_v8_13_defensive10_exploiter_0725:*
```

So the directory rung applies, and **that resolver is imported here rather than re-implemented**
— in both eras (the era checkout's copy of the function is **byte-identical** to this tree's,
diffed, so the v8 arm imports its own tree's). [`resolve_teachers.py`](resolve_teachers.py) records
the resolution with sha256 on both sides; executed output in
[`resolved_teachers.log`](resolved_teachers.log):

```
  ai_v9_92_R5F00_0831        -> ai_v9_92_R5F00_0831/best_model/best_model.zip  sha 01a20e99822a
                                was final_model.zip                            sha f5f57be964ac  DIFFERENT
  …
  ai_v8_06_semistall_3team_exploiter_0722
                             -> …/best_model/best_model.zip                    sha b7b5da9bd48b
                                was final_model_interrupted.zip                sha 3a2036d165f0  DIFFERENT

  19/19 teachers resolve to a DIFFERENT file than content_locality scored
```

**The parents are unchanged**, because the folds named them as explicit `.zip` paths and the
resolver has nothing to do there — verified from the same metadata
(`--model models/ai_v9_59_R2ACTION_0827/final_model.zip`,
`--model models/ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip`).

The resolver returns a config beside the zip (`<run>/best_model/model_config.json` for all 19), and
that config is what is passed. It **cannot change any number**: `load_foreign_opponent` uses
`config_path` only to build a `ModelVersion` for `check_opponent_compatible`; the network is
rebuilt from the zip's own pickled `policy_kwargs`.

### 2 — Reference: the gen exploiters do not fork from the fold parent

Verified per run with `python -m main.lineage`, never assumed:

```
ai_v9_92_R5F00_0831   role=exploiter
    exploiter target: ai_v9_59_R2ACTION_0827
    └─ ai_v9_29_rev1_0823  …  via models/ai_v9_29_rev1_0823/final_model.zip  @25,067,760 steps  sha=2f17fbab0b5f

ai_v9_120_R5FUND00_0901   role=exploiter
    exploiter target: ai_v9_59_R2ACTION_0827
    └─ ai_v9_92_R5F00_0831  …  @28,115,184 steps
       └─ ai_v9_29_rev1_0823  …  @25,067,760

ai_v9_59_R2ACTION_0827   role=fold
    └─ ai_v9_29_rev1_0823  …  @25,067,760 steps  sha=2f17fbab0b5f     ← the SAME checkpoint

ai_v8_09_pool10_exploiter_0723 / ai_v8_06_semistall_3team… / ai_v8_13_defensive10…   role=exploiter
    exploiter target: ai_v8_04_distill_4teacher_0722
    └─ ai_v8_04_distill_4teacher_0722  …  via …/final_model_interrupted.zip @277,583,267  sha=3b7792c87347
                                                                            ← the fold PARENT
```

R2ACTION is the R5F runs' `--exploiter` **target** and a **sibling** fork of the same rev-1
checkpoint; the R5FUND runs continue from their own R5F final. So rev-1's final is the true origin
of both gen halves, while in the v8 era the parent **is** the origin. Two references on the gen
side, one on the v8 side:

| tag | model | question it answers |
|---|---|---|
| **REF-A** | `ai_v9_59_R2ACTION_0827/final_model.zip` | how far is this teacher from the model that will absorb it? (what the fold sees) |
| **REF-B** | `ai_v9_29_rev1_0823/final_model.zip` @25,067,760 | how far has this teacher travelled from where it started? (what the exploiter did) |
| v8 | `ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip` | both at once |

Only the reference distribution changes between the two gen columns — identical states, identical
teams, identical teacher networks. On this state batch the two references are
**`KL(parent ‖ origin) = 0.3342`** apart (n=9; 0.3361 at n=3), i.e. the constant that every REF-A
number carries.

> 🚩 **NAMING COLLISION with `../exploiter_drift/`.** That probe calls the ORIGIN **REF-A** and the
> fold parent **REF-P**. This probe calls the fold parent **REF-A** and the origin **REF-B**. The
> letters are swapped. When comparing the two documents, match on the *model*, never on the letter.

Each reference carries its **own** matched-noise floor. REF-A keeps v1's two adjacent R2ACTION
checkpoints unchanged (the brief's "same two pairs per era"); REF-B needed a floor of its own, so
rev-1's two nearest retained checkpoints (−78,768 and −228,768 steps) were added. **That addition
is declared** — it is beyond "keep the same two pairs", which is honoured for REF-A.

### 3 — The bootstrap is sized from its own array

v1 drew one index matrix per *expected* cluster count and reused it by name. In
`v8_era_locality.py` that went wrong: `own_all` holds one cell per (teacher, taught team) —
10 + 3 + 10 = **23** — while `bsT` was drawn as `rng.integers(0, nT, …)` with
`nT = len(taught_union) = 22`, the **deduped** union (one team is taught by two teachers). Cell 22,
`defensive10`'s last taught team, was unreachable in every pooled-L CI.

[`boot.py`](boot.py) derives the matrix from `len(vals)`, caches it per size, and **asserts the
drawn index range equals the cluster count**. A caller cannot pass a mismatched matrix because it
never passes one. [`boot_bug_demo.py`](boot_bug_demo.py) measures the defect on v1's own committed
data — the under-sampled draw is **reproduced bit-identically**, which is what makes the diagnosis
a measurement:

```
  own_all cells                23   (10 + 3 + 10 teacher-team pairs)
  len(taught_union) used by v1 22   (deduped — one team taught twice)
  => v1 drew indices in [0, 22) over a 23-cell array; cell 22 was unreachable

  RECORDED (v1)              L 1.5220  CI [1.3551, 1.7205]   taught CI [0.3756, 0.4661]
  REPRODUCED under-sampled   L 1.5220  CI [1.3551, 1.7205]   taught CI [0.3756, 0.4661]
  CORRECTLY SIZED            L 1.5220  CI [1.3572, 1.7136]   taught CI [0.3767, 0.4637]
```

Point estimate identical; only the interval moves, and only slightly. **The headline
sibling-control R was correctly sized in v1 and is unaffected by this correction** — its arrays are
16 (gen) and 21 (v8) cells with matching matrices. The gen arm had no size mismatch anywhere, so
its point estimates are untouched and its CIs move only by the different (correctly sized) draws.

---

## The states are the SAME states — asserted, not hoped for

Everything about state generation is copied verbatim from v1: the same 24 (gen) / 30 (v8) teams,
the same parent pilot, the same reference opponent, `concurrency=1`, the same bridge impl per era
(rust / node), and the same seeds — sim `[team_index+1, 2, 3, 4]`, pool sequence
`random.Random(61000 + i)`, gen-era pilot policy `71000 + i`, opponent policy `72000 + i`. Three
independent cross-checks are **assertions in the scripts**, so a drifted batch refuses to report:

| check | expected | got |
|---|---|---|
| gen n=3 untaught states (the canonical `offline_collateral_kl` batch) | 1100 | **1100 — PASS** |
| gen n=9 untaught per-team (`teacher_distance`'s gen arm) | `[280, 399, 333, 458, 714, 592, 391, 301]` | **identical — PASS** |
| v8 n=3 / n=9 untaught per-team (`content_locality`'s own) | `[109,104,98,96,88,80,92,78]` / `[266,255,260,312,270,265,303,259]` | **identical — PASS** |

And the strongest one, which needed no assertion: **every matched-noise floor reproduces v1 to
four decimal places** (gen REF-A n=9 `0.0374 / 0.0401 / L 1.0725` and `0.0654 / 0.0760 / L 1.1620`;
v8 n=9 `0.0383 / 0.0263` and `0.0664 / 0.0535`; the n=3 floors likewise). The floor networks are
unchanged by the corrections, so identical floors on identical states is exactly what a byte-clean
reproduction looks like.

**ACID** (no two teachers may share a per-team KL vector — a mis-resolved path masquerading as a
null): `acid_all_distinct: true` in all four artifacts.

**KL gate.** The v8 era predates `instrumented_ppo.distill_anchor`, so it uses
`../content_locality/era_kl.py`; its bit-identity with the imported `masked_kl_rows` was re-run in
this session — 7 synthetic cases, worst `|Δ| = 0.000e+00`, PASS.

---

## Results (n=9; n=3 agrees throughout and is reported beside)

The batches are **10025** gen states (24 teams) and **11650** v8 states (30 teams) at n=9; 3339 and
4180 at n=3.

`R` = per taught team, `KL(own teacher ‖ reference) / mean KL(same-era sibling teachers ‖ reference)`
on the **same states**. `R = 1.00` is perfectly GLOBAL. R is primary because it holds the team, the
state distribution and the recipe fixed — the floor shows the team-set effect is real and points in
**opposite directions** in the two eras, so a cross-era comparison of the raw `L` would read the
team sets rather than the teachers.

### Headline — sibling-control R, v2 beside v1

| arm | reference | v1 R (n=9) | **v2 R (n=9)** | v2 R (n=3) |
|---|---|---|---|---|
| v8 (3 teachers) | parent = origin | 1.4498 [1.2728, 1.6722] | **1.8316 [1.5334, 2.1744]** | 1.7940 [1.4803, 2.1551] |
| gen unfunded R5F (8) | REF-A fold parent | 1.0723 [0.9803, 1.1634] | **1.0722 [0.9432, 1.1977]** | 1.0956 [0.9208, 1.2971] |
| gen funded R5FUND (8) | REF-A fold parent | 1.1016 [1.0008, 1.1987] | **1.1067 [1.0026, 1.2071]** | 1.1013 [0.9604, 1.2441] |
| gen unfunded R5F (8) | REF-B true origin | — | **1.2542 [1.0318, 1.4663]** | 1.2961 [1.0439, 1.5443] |
| gen funded R5FUND (8) | REF-B true origin | — | **1.1953 [1.0972, 1.2983]** | 1.1859 [1.0794, 1.2859] |

### Absolute levels (n=9)

| half | reference | KL on own taught | KL on untaught 8 | raw L |
|---|---|---|---|---|
| gen unfunded | REF-A | 0.5789 [0.4908, 0.6741] | 0.5536 [0.5030, 0.6034] | 1.0502 [0.8998, 1.2137] |
| gen funded | REF-A | 0.7613 [0.6756, 0.8555] | 0.6957 [0.6400, 0.7546] | 1.0971 [0.9900, 1.1912] |
| gen unfunded | REF-B | 0.3433 [0.2639, 0.4186] | 0.2576 [0.2031, 0.3045] | 1.3306 [1.1904, 1.4944] |
| gen funded | REF-B | 0.5317 [0.4800, 0.5959] | 0.4160 [0.3795, 0.4576] | 1.2808 [1.1950, 1.3697] |
| v8 (all 3) | parent = origin | 0.3969 (own, in the sibling control) | 0.2303 (siblings, same teams) | 1.6202 [1.2922, 2.0816] |

### Contrasts (n=9)

| contrast | reference | delta | CI95 | verdict |
|---|---|---|---|---|
| v8 − gen unfunded (R, unpaired) | REF-A | +0.7594 | [+0.4312, +1.1236] | **SIGNIFICANT** |
| v8 − gen funded (R, unpaired) | REF-A | +0.7249 | [+0.4111, +1.0829] | **SIGNIFICANT** |
| v8 − gen unfunded (R, unpaired) | REF-B | +0.5774 | [+0.2053, +0.9832] | **SIGNIFICANT** |
| v8 − gen funded (R, unpaired) | REF-B | +0.6363 | [+0.3196, +0.9919] | **SIGNIFICANT** |
| gen funded − unfunded (R, paired on 16 teams) | REF-A | +0.0345 | [−0.0795, +0.1636] | **NOT DETECTED** |
| gen funded − unfunded (R, paired on 16 teams) | REF-B | −0.0589 | [−0.2508, +0.1472] | **NOT DETECTED** |

At n=3 the same six read `+0.6983 / +0.6926 / +0.4978 / +0.6080` (all SIGNIFICANT) and
`+0.0057 / −0.1102` (both NOT DETECTED).

### Matched-noise floor (n=9) — nothing here is WITHIN FLOOR

| era | pair | reference | KL untaught | KL taught | floor L |
|---|---|---|---|---|---|
| gen | `FLOORA_ckptA` (−47k) | REF-A | 0.0374 | 0.0401 | 1.0725 |
| gen | `FLOORA_ckptB` (−197k) | REF-A | 0.0654 | 0.0760 | 1.1620 |
| gen | `FLOORB_ckptA` (−79k) | REF-B | 0.0112 | 0.0133 | 1.1894 |
| gen | `FLOORB_ckptB` (−229k) | REF-B | 0.0415 | 0.0453 | 1.0923 |
| v8 | `FLOOR_c277178` (−405k) | parent = origin | 0.0383 | 0.0263 | 0.6878 |
| v8 | `FLOOR_c275758` (−1.82M) | parent = origin | 0.0664 | 0.0535 | 0.8053 |

Every teacher aggregate sits above its floor on **both** sides of every ratio, so **no teacher is
reported WITHIN FLOOR** — but the margins are not uniform and the thinnest one is load-bearing:

| era / reference | teacher aggregates (per-teacher taught & untaught) | that reference's floors | multiples |
|---|---|---|---|
| gen REF-A | 0.4044 – 1.0278 | 0.0374 – 0.0760 | 5.3× – 27.5× |
| gen REF-B | 0.1073 – 0.7228 | 0.0112 – 0.0453 | 2.4× – 64.5× |
| v8 | 0.1036 – 0.4722 | 0.0263 – 0.0664 | **1.56×** – 17.9× |

⚠️ **The 1.56× is `semistall3`'s untaught KL (0.1036) against the larger v8 floor (0.0664)** — and
that is precisely the cell the whole v8 R jump rests on. It clears the floor (2.0× against the
mean of the two untaught floors, 1.56× against the worse one) but it is the closest call in this
document, it comes from a 3-team teacher, and it is a *denominator*, so its noise is amplified in
the ratio. Read v8's `R 1.8316` with that in mind: the direction is robust across n=3/n=9 and two
independently-written scripts, the exact magnitude leans on one thin cell.

The raw `L` remains confounded exactly as v1 said — the gen floors sit at `L` 1.07–1.19, the v8
floors at 0.69–0.81, in *opposite* directions — which is why R is the headline.

### What the resolved checkpoint changed, teacher by teacher

The v8 side moved most, and asymmetrically:

| teacher | n taught | v1 KL untaught | **v2 KL untaught** | v1 L | **v2 L** |
|---|---|---|---|---|---|
| `pool10` | 10 | 0.3223 | **0.3176** | 1.4643 | **1.4869** |
| `semistall3` | 3 | 0.2190 | **0.1036** | 2.0450 | **2.0816** |
| `defensive10` | 10 | 0.2807 | **0.2775** | 1.2571 | **1.2922** |
| **pooled** | 23 cells | 0.2740 | **0.2329** | 1.5220 [1.3551, 1.7205] | **1.6718 [1.4567, 1.9161]** |

`semistall3`'s off-slice divergence **more than halves** on the file the fold actually loaded
(0.2190 → 0.1036), which is the whole of the v8 R jump: its own-team KL barely moves while the
denominator it contributes to its siblings collapses. On the gen side the shift is smaller and
lands almost entirely on the **unfunded** half:

| teacher | v1 | **v2** | Δ | | teacher | v1 | **v2** | Δ |
|---|---|---|---|---|---|---|---|---|
| `UNF00` | 0.5759 | **0.5669** | −0.0089 | | `FUND00` | 0.8239 | **0.8324** | +0.0084 |
| `UNF02` | 0.5379 | **0.4242** | **−0.1136** | | `FUND02` | 0.5944 | **0.5961** | +0.0017 |
| `UNF04` | 0.6921 | **0.6481** | −0.0439 | | `FUND04` | 0.7063 | **0.7180** | +0.0117 |
| `UNF06` | 0.6474 | **0.5148** | **−0.1326** | | `FUND06` | 0.7810 | **0.7876** | +0.0067 |
| `UNF08` | 0.5063 | **0.4956** | −0.0107 | | `FUND08` | 0.6374 | **0.6280** | −0.0094 |
| `UNF10` | 0.5776 | **0.5690** | −0.0086 | | `FUND10` | 0.6157 | **0.6096** | −0.0061 |
| `UNF12` | 0.5678 | **0.5504** | −0.0173 | | `FUND12` | 0.6636 | **0.6417** | −0.0219 |
| `UNF14` | 0.6868 | **0.6595** | −0.0273 | | `FUND14` | 0.7527 | **0.7523** | −0.0004 |

so the paired **FUNDED − UNFUNDED untaught** gap grows by 45%:
**+0.1421 [+0.0889, +0.2006]** on the resolved checkpoints against v1's +0.0979 [+0.0547, +0.1493]
— both SIGNIFICANT. Under REF-B it is **+0.1584 [+0.1031, +0.2202]**, and the taught-side gap is
**+0.1824 [+0.1127, +0.2643]** (REF-A) / **+0.1883 [+0.1244, +0.2609]** (REF-B).

---

## Two independent reproductions, unplanned and exact

1. **`teacher_distance`'s re-measurement.** That probe computed the gen halves' untaught KL on the
   resolved checkpoints with a separately-written script: UNF **0.5536**, FUND **0.6957**, and
   FUNDED−UNFUNDED **+0.1421 [+0.0889, +0.2015]**. This probe measures **0.5536 / 0.6957 /
   +0.1421 [+0.0889, +0.2006]** — identical to four decimals, with the CI upper bound differing in
   the fourth place because the bootstrap draws differ (correction 3 changed the matrices).
2. **The coordinator's z-swap probe.** It measured v8's sibling-control R at **1.8316
   [1.5349, 2.1782]** on the resolved checkpoints. This probe measures **1.8316 [1.5334, 2.1744]**
   — the point estimate agrees to four decimals and the CI to the third, again from a different
   bootstrap draw. Two independently-written scripts landing on the same number is the strongest
   evidence in this document that the resolution correction is right.

---

## Verdict

### H1, cross-era ordering (`R_v8 > R_gen`) — **SIGNIFICANT, and STRENGTHENED**

Prediction (i)'s first half is **confirmed**. On the checkpoints the folds actually used, v8-era
teachers are *more* local than v1 reported — `R 1.4498 → **1.8316** [1.5334, 2.1744]` — while the
gen side is essentially unmoved under REF-A (`1.0723 → 1.0722`, `1.1016 → 1.1067`). *(The gen
point estimates barely move but their CIs widen — v1's unfunded [0.9803, 1.1634] against v2's
[0.9432, 1.1977]. That is the DATA, not the bootstrap: the resolution moved individual teachers
unevenly (`UNF02` −0.114, `UNF06` −0.133, the rest ≤0.044), which raises the between-team variance
while leaving the mean.)* The era gap
therefore roughly **doubles**: +0.35/+0.38 in v1 becomes **+0.72/+0.76**, CIs excluding zero at
both n and under both gen references. v8's teachers concentrate their divergence on their own
taught teams; the gen teachers do not.

### H2, within-gen (funded vs unfunded R) — **NOT DETECTED**, under both references

Prediction (i)'s second half is **confirmed**. `+0.0345 [−0.0795, +0.1636]` (REF-A) and
`−0.0589 [−0.2508, +0.1472]` (REF-B) at n=9; `+0.0057` and `−0.1102` at n=3, all spanning zero.
Locality still does not distinguish the teacher half that robs from the half that does not — what
distinguishes them is plain **magnitude**, and the correction makes that *larger*: the funded half
sits +0.142 further from the parent off-slice, not +0.098.

### Prediction (ii), the fork-origin question — **PARTIAL: neither registered branch fired**

Measuring the gen teachers from their **own origin** does raise R, and the rise is real:
unfunded `1.0722 → 1.2542`, funded `1.1067 → 1.1953`, and both REF-B CIs now exclude 1 where the
unfunded REF-A CI did not. But it lands **between** the two registered readings — not `≥ 1.3`
(which would have said the era difference was the fork-origin offset) and not `≈ 1.1` (which would
have said the origin explains nothing).

The registered dichotomy also missed a third possibility that actually occurred: **v8's R rose
too**, by more than the gen side's did, because correction 1 hit v8 hardest. So the two effects do
not cancel and the gap does not close — under REF-B it is still **+0.58 / +0.64 with CIs excluding
zero**. In distance-from-global terms (`R − 1`), the gen teachers reach **31%** (unfunded) and
**23%** (funded) of v8's locality measured from their own origin, against 9% and 13% measured from
the fold parent.

**The honest statement: the fork-origin offset accounts for 24% of the apparent gap on the
unfunded half (+0.7594 → +0.5774) and 12% on the funded half (+0.7249 → +0.6363), and does NOT
rescue the "gen teachers are global" reading.** Gen-era exploiters remain markedly more globally
displaced than v8-era ones even when each is measured from where it actually started.

**And the origin column corroborates `exploiter_drift` almost exactly.** That probe measured
`ρ = KL_on / KL_off` against the origin at **1.22 at 150k steps and 1.26 at 5.0M**, flat, on a
frozen 152-state batch with a different statistic form. This probe's REF-B unfunded R is
**1.2542** on 10,025 parent-piloted states. Two probes, two state distributions, two statistics,
the same number — *"an exploiter is a globally drifting policy with a small local bonus, from the
beginning."* The bonus is real (the CI excludes 1) and it is small.

### What this does and does not change for the ledger

* v1's two headline conclusions **stand, both stronger**. Nothing is retracted.
* Every **level** v1 published is superseded by the tables above.
* The "gen teachers are global" reading survives as a **comparative** claim (against v8), not as an
  absolute one: from their own origin they carry a small but detectable +25% own-team bonus.
* Locality remains **useless as a predictor of which fold robs**. Magnitude remains the axis that
  separates the halves, and the corrected magnitudes are larger.

---

## Limits, stated plainly

* **State distribution is parent-piloted.** Every teacher is scored on states the *parent* reaches,
  not on states it would itself reach. Correct for a divergence-from-reference statistic and what
  makes teachers comparable — but not the distribution the fold's own rollouts see. These levels
  must never be merged with the live `distill/collateral_kl_vs_parent` column.
* **REF-B keeps the same states as REF-A**, which are parent-piloted. So REF-B answers "how far has
  this teacher moved from its origin, *as seen from the parent's states*" — it changes the
  reference, not the measure. A fully origin-native version would pilot with rev-1's final and is
  **not** what was run.
* **The cross-era comparison cannot attribute anything to architecture.** Architecture, exploiter
  budget, teacher count (3 vs 8), teams per teacher (10/3/10 vs 2), pool, parent maturity and the
  play recipe (greedy + node vs seeded-stochastic + rust) all differ. "Gen-era teachers are less
  local" is measured; "because of the pointer head" is not.
* **Sibling count differs** (7 gen siblings vs 2 v8). A 2-sibling mean is noisier, which widens the
  v8 per-team R without biasing it.
* **The v8 taught sets overlap.** `ai_v8_06`'s `9d5f845869e899ee.txt` hashes to `564b9be3ae`, also
  `ai_v8_09`'s `t00`; that one team has no clean "did not teach it" sibling and is excluded from
  the sibling control (21 singly-taught teams, not 22). Unchanged from v1; the script prints it.
* **`L` for the gen halves rests on 2 taught teams per teacher.** Only the 8-teacher group
  bootstrap is quoted; R (16 teams) is what the verdict rests on.
* **Two battle counts, not a pre-registered n.** n=3 reproduces the canonical batch exactly; n=9 is
  the declared power extension. Both are reported and no conclusion changes between them.
* **The box was busy** (load 18–34 on 16 cores). Wall clocks below are therefore not comparable to
  v1's idle-box timings; the *numbers* are unaffected — every source of randomness is seeded and
  `concurrency=1`, and the floors reproduce to four decimals, which is the proof.

---

## Files

| file | what |
|---|---|
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | the predictions and the null, with an honest provenance note |
| [`resolve_teachers.py`](resolve_teachers.py) → `resolved_teachers.json` (+ `.log`) | correction 1 made auditable — the imported resolver, per teacher, with sha256 both ways |
| [`boot.py`](boot.py) | correction 3 — the size-derived cluster bootstrap with its assertion |
| [`boot_bug_demo.py`](boot_bug_demo.py) (+ `.log`) | the under-sampling reproduced bit-identically on v1's own data, then fixed |
| [`gen_era_locality_v2.py`](gen_era_locality_v2.py) | the gen-era measurement, both references |
| [`v8_era_locality_v2.py`](v8_era_locality_v2.py) | the v8-era measurement (run from the era tree) |
| [`combine_v2.py`](combine_v2.py) | the cross-era readout; recomputes every ratio from the per-team vectors |
| [`emit_tables.py`](emit_tables.py) → `tables.md` | every numeric table above, emitted from the artifacts (nothing transcribed) |
| [`verify_readme.py`](verify_readme.py) (+ `.log`) | 19 load-bearing numbers in this README recomputed from the JSON and required to appear verbatim — **PASS** |
| `gen_era_v2_n3.json` · `gen_era_v2_n9.json` (+ `.log`) | gen artifacts |
| `v8_era_v2_n3.json` · `v8_era_v2_n9.json` (+ `.log`) | v8 artifacts |
| `combined_v2_n3.json` · `combined_v2_n9.json` (+ `.log`) | the joined readout |
| `run_gen.sh` · `run_v8.sh` | the exact invocations |

The KL function and its bit-identity gate are **not duplicated here** — they are
[`../content_locality/era_kl.py`](../content_locality/era_kl.py) and
[`../content_locality/kl_unit_test.py`](../content_locality/kl_unit_test.py), imported and re-run.

### Reproduce

```bash
# gen era (this tree) — resolver table, then the two arms
python designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolve_teachers.py resolved_teachers.json
./designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/run_gen.sh 3 gen_era_v2_n3.json gen_era_v2_n3.log
./designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/run_gen.sh 9 gen_era_v2_n9.json gen_era_v2_n9.log

# v8 era (from the era-pinned READ-ONLY checkout at b13b30b2)
./designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/run_v8.sh 3 v8_era_v2_n3.json v8_era_v2_n3.log
./designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/run_v8.sh 9 v8_era_v2_n9.json v8_era_v2_n9.log

# join + tables
python .../content_locality_v2/combine_v2.py gen_era_v2_n9.json v8_era_v2_n9.json combined_v2_n9.json
python .../content_locality_v2/emit_tables.py > tables.md
```

Wall clock, box under a live training run (load 18–34 on 16 cores), CPU only, `nice -n 10`:
gen n=3 **336 s**, gen n=9 **1100 s**, v8 n=3 **295 s**, v8 n=9 **629 s**.
