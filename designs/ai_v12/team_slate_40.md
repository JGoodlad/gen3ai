# ai_v12 — the 40-team candidate slate

**Status: a SELECTION INPUT, not a launch order.** It is built to be read the day the rev-4
shape discriminator lands, so the next fleet can be launched the same day. Nothing here has been
trained; no battles were played to produce it.

Sources are existing artifacts only. Every number carries the scale it was measured on and the
n behind it; an unmeasured team is listed as UNMEASURED with n=0 and is **never imputed**.

### The four things a reader needs before anything else

1. **The curated-32 gate BINDS, and 40 > 32.** After rev-4 only **8** curated teams are
   untaught. A 40-team fleet needs the sample set widened, the research override taken, or the
   ambition cut to ≤32 — §1.
2. **Coverage is thin: 44 of 719 teams have a fixed-reference estimate**
   (§2). Half the slate is nominated rather than measured and says so.
3. **The wide training-time table does not substitute for one** — it was tested as a predictor
   and came back ρ ≈ 0.4–0.5 (§3). It nominates; it never ranks.
4. **The ranking is a ranking on `−baseline_WR`**, because the ceiling is treated as constant.
   That is licensed by the ceiling reframe, and its limit is stated in §4.

## 1. The constraint that shapes everything: the curated-32 cap

`--exploiter` refuses a trainee team that is not one of the curated `data/teams/sample/` teams.

| | |
|---|---|
| enforced in | `src/main/train/matchup_setup.py:127-140` — FATAL, `TrainExitCode.FATAL_CONFIG` |
| predicate | `agents.training.matchup_spec.validate_exploiter_trainee_is_sample` (`matchup_spec.py:265`) |
| scope | `mix_kind == 'exploiter'` with a PINNED trainee (`pinned` / `pin_biased` / `pin_multi`); a generalist exploiter is out of scope |
| curated set size | **32** `.txt` files in `data/teams/sample/` (`teams.json` is the manifest, not a team) |
| documented escape hatch | `--allow-nonsample-trainee` — prints a warning and skips the gate; its own comment says *"Use for capacity studies … NOT for a teacher you intend to distil as-is"* |

**It binds, and it has already bitten once.** The rev-3 coverage picks were drawn from the full
719-team pool, REJECTED by this gate, and re-picked from the curated set — and the admission
harness's team dict was never updated, which is the 16/36-cell mid-flight save recorded at
61608ac. `probes/coverage_sweep.json`'s three `picks` (`a05a190b50`, `27b7b27e8a`, `f9f8d0608a`)
are still pool teams; the teams actually trained are the three curated `COV_*`.

### The arithmetic that caps the ambition

| set | n | note |
|---|---|---|
| curated teams | 32 | the whole legal exploiter-trainee universe |
| taught by F5a–e + F6a–f (incl. the 9 meter) | 12 | rev-2 and rev-3 fleets |
| newly taught by rev-4 R4S3a/b/c (frozen argvs) | 12 | 24 distinct once rev-4 lands |
| **curated and still untaught after rev-4** | **8** | `ce35b736 · b89e1e37 · 9909f2e9 · e11829f0 · f7ba5702 · dbf81d8e · a04c29cf · 9f27f5d3` |
| held out (`probes/offpin_{0,1}.txt`) | 2 | pool teams, not curated — the off-slice transfer instrument |

> **A 40-team revolution cannot be built from the curated set.** Even ignoring the exclusions
> entirely, 32 < 40; honoring them leaves **8**. Three ways forward, none of them free, and this
> is a NAMED GAP for the design rather than something S1 fixes:
>
> 1. **Promote vetted teams into `data/teams/sample/`** — what the refusal message itself
>    advises (*"promote this one into the sample set first if it is proven"*). ~32 promotions
>    are needed. The gate's stated purpose is that a teacher pilots a *tournament-proven* team, so
>    a promotion needs a vetting criterion that does not yet exist in the tree.
> 2. **`--allow-nonsample-trainee`** — one flag, zero code. But its own comment excludes exactly
>    this use (a teacher you intend to distil), so taking it means overriding a documented
>    intent, deliberately and in writing.
> 3. **Shrink the ambition to ≤ 32** — e.g. 4 teachers × 8, which still tests breadth against
>    rev-3's 6×2 and fits the count-dominates N≤10 bound.

## 2. Coverage — what is actually measured

Pool: **719** teams. Fixed-reference piloting estimates exist for
**44 of 719** (6.1%).

| evidence tier | teams | n / team | what it is |
|---|---|---|---|
| **A — gen-15 direct** | 18 | 150–400 | R2-ACTION final pilots the pinned team; opponent draws the pool |
| **B — rev-1 shifted** | 26 | 200–400 | rev-1 final pilots; shifted by the measured generation offset |
| **C — n=40 screen** | 67 | 40 | nomination only; quantized to ±0.025 and selection-prone |
| **UNMEASURED** | 608 | 0 | no fixed-reference measurement exists |
| *(training-time WR, any tier)* | 665 | ≥30 | a **different scale** — see §3 |

## 3. Three scales, and why only two of them may be pooled

| scale | pilot | opponent | comparable to the ~0.69 ceiling? |
|---|---|---|---|
| `gen15` | R2-ACTION final | rev-1@24M *or* R2-ACTION final, drawing the 719-pool | **yes** — the ceiling is the set-mean of the rev-3 teacher cells on this harness |
| `rev1f` | rev-1 final | rev-1@24M | after the offset below |
| `train` | the run itself, mid-training | a MOVING self-play pool | **no** |

**Two calibrations were measured rather than assumed:**

* **Opponent equivalence.** On the 3 coverage teams that carry both, R2-ACTION-final and
  rev-1@24M as the *opponent* differ by **-0.0025**
  (n=3 teams, 300–400 games each) — indistinguishable from zero, so the two
  gen-15 sub-harnesses are pooled.
* **Generation offset.** Over **18** curated teams carrying both pilots, gen-15 minus
  rev-1-final is **-0.0100**. Tier-B rows are shifted by exactly this.
  *This is itself a finding:* on these teams the current generation pilots **no better than**
  rev-1 did — the same 'meter teams are mined out / the fold redistributes' reading the rev-3
  recap (ade78c1) arrived at from the other direction.

**The training-time table was tested as a predictor and largely FAILED.** `team_win_rates`
(the task-#18 machinery, `TeamWinRateCallback` → `metadata.json`) is by far the widest source —
717 teams in `ai_v9_58_R2CTRL_0827`, 716 in `ai_v9_29_rev1_0823` at a median 582 pool games —
but rank-correlating it against the fixed-reference estimates on the overlap gives
**ρ = 0.526 (n=16, SE ≈ 0.258)**
on curated teams and
**ρ = 0.364 (n=12, SE ≈ 0.302)**
on pool teams — i.e. **z ≈ 2.1 and z ≈ 1.2**. The curated arm is borderline; the pool arm is
nothing. Even taking the larger at face value, ρ ≈ 0.5 accounts for a quarter of the rank
variance, and the *levels* are not even close (the same teams read 0.55–0.80 on the training
scale and 0.37–0.56 on the fixed-reference one). The artifact's own `notes` field predicted this
— *"a low win rate with team T may mean we pilot T badly OR that T is weak"* — and a moving
self-play opponent plus a bot-heavy early curriculum compound it. **So it nominates; it never
ranks, and it is never pooled with a fixed-reference number.** (The code-rank lesson applied:
gate a signal on *does it PREDICT?*, not on *is it available?*)

## 4. Headroom

`headroom(T) = 0.6896 − baseline_WR(T)`, both on the gen-15 scale.

The ceiling is the set-mean of the **12 rev-3 teacher ABSOLUTE cells** in `r3_admission.json`
(observed per-team range **0.5925–0.775**,
matching the admission record). The ceiling-reframe result (61608ac) is what makes this
estimable with no teacher trained: teachers land at ~0.69 *regardless* of where the target
starts (budget +67% ⇒ +0.0019 z=0.16; target start 0.46–0.61), so headroom is a property of the
**baseline**, not of the teacher.

⚠️ **The per-team ceiling spread is ±9pp and is NOT modeled here.** A team whose true ceiling is
0.59 and whose baseline is 0.37 has 22pp of headroom, not 32pp. Ranking on a constant ceiling is
therefore a ranking on `−baseline_WR`; it is defensible only because the baseline spread
(0.35–0.65) is ~1.7× the ceiling spread. Treat the ordering as robust and the magnitudes as
upper-ish bounds.

## 5. The slate

Split by evidence, deliberately — a 40-row table where half the rows are n=40 would read as one
measurement.

### 5a. CORE 20 — measured, launchable ordering

| # | sha | file (under `data/teams/`) | set | archetype | baseline WR | n | 95% CI | headroom | tier |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `a05a190b50` | `others/schmuck_nick/836c2cc75f027b43.txt` | POOL | offense | 0.355 | 200 | [0.291, 0.424] | **0.335** | B |
| 2 | `27b7b27e8a` | `others/mcmegan/5d4bc12fda99fc04.txt` | POOL | balance | 0.360 | 200 | [0.296, 0.429] | **0.330** | B |
| 3 | `c84f2b64a2` | `sample/9f27f5d3e34021a7.txt` | curated | stall | 0.370 | 200 | [0.306, 0.439] | **0.320** | B |
| 4 | `f9f8d0608a` | `others/mcmegan/7fda48d98ca8efdc.txt` | POOL | stall | 0.370 | 200 | [0.306, 0.439] | **0.320** | B |
| 5 | `4b1a1c2e58` | `others/johnnyg2/be96c6d78622e609.txt` | POOL | offense | 0.370 | 200 | [0.306, 0.439] | **0.320** | B |
| 6 | `b4545b1317` | `others/giraffe/5578a6f415bdef2f.txt` | POOL | stall | 0.390 | 200 | [0.325, 0.459] | **0.300** | B |
| 7 | `b42e937375` | `others/giraffe/5f4e66573157e347.txt` | POOL | offense | 0.410 | 200 | [0.344, 0.479] | **0.280** | B |
| 8 | `b456a6cdbf` | `others/giraffe/b65ac0cbc96406c0.txt` | POOL | semi_stall | 0.430 | 200 | [0.363, 0.499] | **0.260** | B |
| 9 | `acff984b98` | `others/giraffe/995cc8a375500826.txt` | POOL | hyper_offense | 0.440 | 200 | [0.373, 0.509] | **0.250** | B |
| 10 | `4ad12a36ea` | `others/giraffe/50bffebdd7cbc754.txt` | POOL | hyper_offense | 0.450 | 200 | [0.382, 0.519] | **0.240** | B |
| 11 | `715845ed34` | `others/giraffe/a916e380aef589cf.txt` | POOL | hyper_offense | 0.465 | 200 | [0.397, 0.534] | **0.225** | B |
| 12 | `8a8543d41d` | `others/giraffe/37b7d8c99216ce6a.txt` | POOL | semi_stall | 0.465 | 200 | [0.397, 0.534] | **0.225** | B |
| 13 | `3495ef83ef` | `sample/e11829f0561ef5a9.txt` | curated | balance | 0.545 | 200 | [0.476, 0.612] | **0.145** | B |
| 14 | `f5d46ca0fc` | `others/mcmegan/c16c4b77942e1a00.txt` | POOL | semi_stall | 0.545 | 200 | [0.476, 0.612] | **0.145** | B |
| 15 | `a7406f6c97` | `sample/a04c29cf769e9a11.txt` | curated | balance | 0.560 | 200 | [0.491, 0.627] | **0.130** | B |
| 16 | `324235812b` | `sample/9909f2e98e981ccc.txt` | curated | stall | 0.570 | 200 | [0.501, 0.636] | **0.120** | B |
| 17 | `01cd428c76` | `sample/ce35b7368c3d692e.txt` | curated | balance | 0.570 | 200 | [0.501, 0.636] | **0.120** | B |
| 18 | `4771662cf7` | `sample/f7ba5702fe856292.txt` | curated | stall | 0.615 | 200 | [0.546, 0.679] | **0.075** | B |
| 19 | `21022d30fb` | `sample/b89e1e37caa40e6a.txt` | curated | stall | 0.650 | 200 | [0.582, 0.712] | **0.040** | B |
| 20 | `569ebae46d` | `sample/dbf81d8ecae51c39.txt` | curated | balance | 0.655 | 200 | [0.587, 0.717] | **0.035** | B |

Of these, **8** are curated (legal today); **12** are pool teams requiring §1's widening.

### 5b. PROVISIONAL 20 — nominated, **must be re-screened**

Every row below rests on an n=40 screen (±0.08 at 1 SE) plus the weak training-time signal.
The listed order is a nomination order, **not a measurement**; adjacent rows are tied within noise.

| # | sha | file (under `data/teams/`) | archetype | screen WR (n=40) | headroom | train WR | train n |
|---|---|---|---|---|---|---|---|
| 21 | `c43b16ff77` | `others/yak_attack/70c06b0328863535.txt` | balance | 0.465 | 0.225 | 0.423 | 137 |
| 22 | `7ad65bac76` | `others/giraffe/6b0e85bcbb852991.txt` | balance | 0.465 | 0.225 | 0.442 | 154 |
| 23 | `c5bf93d942` | `others/yak_attack/00b19e5d8762fcfd.txt` | offense | 0.440 | 0.250 | 0.583 | 982 |
| 24 | `3ab6020379` | `others/giraffe/ec21f99b72e5767d.txt` | semi_stall | 0.440 | 0.250 | 0.633 | 638 |
| 25 | `356d929368` | `others/yak_attack/6c99ed9a5fb079b9.txt` | stall | 0.440 | 0.250 | 0.677 | 542 |
| 26 | `3ec9787dfd` | `others/yak_attack/48bdf3938e72aa6b.txt` | stall | 0.440 | 0.250 | 0.679 | 1280 |
| 27 | `1ca483a3fd` | `others/giraffe/6f0228da330b90e5.txt` | balance | 0.490 | 0.200 | 0.488 | 43 |
| 28 | `4bb434f8ba` | `others/giraffe/e216c746abc4f927.txt` | hyper_offense | 0.465 | 0.225 | 0.629 | 636 |
| 29 | `58cd9624fc` | `others/giraffe/d7f8595af2ef442a.txt` | hyper_offense | 0.465 | 0.225 | 0.659 | 642 |
| 30 | `b022e1bc92` | `others/johnnyg2/21d69ab6da7736d5.txt` | hyper_offense | 0.490 | 0.200 | 0.577 | 104 |
| 31 | `507fe7fd7b` | `others/yak_attack/a8f59ed79cd79215.txt` | balance | 0.440 | 0.250 | 0.789 | 341 |
| 32 | `c349b569ca` | `others/johnnyg2/bcd3290ff6824219.txt` | offense | 0.465 | 0.225 | 0.701 | 532 |
| 33 | `e3d458dc03` | `others/yak_attack/48c20534cc938c3e.txt` | semi_stall | 0.465 | 0.225 | 0.707 | 92 |
| 34 | `b9dac42529` | `others/giraffe/2778004a133b19d4.txt` | hyper_offense | 0.465 | 0.225 | 0.720 | 75 |
| 35 | `f21abb7e74` | `others/johnnyg2/440260e5240a694a.txt` | stall | 0.465 | 0.225 | 0.744 | 1149 |
| 36 | `6d8f5c1c26` | `others/yak_attack/312a711e196e3c48.txt` | balance | 0.490 | 0.200 | 0.652 | 227 |
| 37 | `8b6b8c8f52` | `others/yak_attack/5874c12610c3f415.txt` | semi_stall | 0.490 | 0.200 | 0.667 | 498 |
| 38 | `9726bf37fc` | `others/giraffe/d63250ac4a5b20ff.txt` | offense | 0.490 | 0.200 | 0.672 | 250 |
| 39 | `93d4167530` | `others/johnnyg2/5f8ac094d96bff26.txt` | hyper_offense | 0.490 | 0.200 | 0.679 | 448 |
| 40 | `dc94160296` | `others/giraffe/55f9b006e460f138.txt` | hyper_offense | 0.490 | 0.200 | 0.700 | 373 |

> **Rev-4's 12 teams are excluded CONDITIONALLY and the exclusion is reversible.** Its argvs are
> frozen but its teachers are not folded; if rev-4 is abandoned or its fleet is not admitted,
> those 12 return to the candidate pool. They are tagged `rev4_pending:<arm>` in the JSON's
> per-team `excluded` list, separately from `taught:<teacher>`, precisely so the two can be
> reversed independently.

### 5c. UNMEASURED — candidates too, ranked separately, n=0

**608** eligible pool teams have no fixed-reference measurement. They are
candidates on equal footing once screened — a team is not weak because nobody has measured it.
The 20 with the lowest training-time pool win rate (the weak nomination signal, ρ≈0.4) are:

| sha | file (under `data/teams/`) | archetype | train WR (pool) | train n |
|---|---|---|---|---|
| `f29ed69af3` | `others/yak_attack/88a79879fa283313.txt` | stall | 0.332 | 683 |
| `67eea52a8e` | `others/yak_attack/83ebc5f8b3130dd9.txt` | semi_stall | 0.333 | 132 |
| `f9de26727a` | `others/yak_attack/a7ef9c9e926f0c5c.txt` | offense | 0.333 | 93 |
| `fdd21c18e2` | `others/giraffe/053f4563a57792a7.txt` | hyper_offense | 0.383 | 47 |
| `ceab5d099e` | `others/yak_attack/d088ac1235f6bf46.txt` | stall | 0.385 | 545 |
| `d44f2afea1` | `others/yak_attack/fc3bd4921a1ab434.txt` | semi_stall | 0.396 | 53 |
| `015e78dda7` | `others/johnnyg2/2c5a6ac7bd26b756.txt` | balance | 0.400 | 40 |
| `177d825984` | `others/mcmegan/81f65cfe57575890.txt` | semi_stall | 0.423 | 895 |
| `e7f8cd4094` | `others/yak_attack/62aca6492e928174.txt` | offense | 0.425 | 47 |
| `048182d1e9` | `others/mcmegan/c2455eb24cc123a8.txt` | stall | 0.426 | 881 |
| `6fabb5ff99` | `others/yak_attack/caf3b5e3921121d2.txt` | offense | 0.427 | 396 |
| `382e285965` | `others/yak_attack/de6092df9621942b.txt` | balance | 0.429 | 70 |
| `2d58357f00` | `others/yak_attack/bbe4f97e7a4e18c8.txt` | offense | 0.438 | 48 |
| `8cd386c07d` | `others/johnnyg2/39ca8ea07e3d8c99.txt` | balance | 0.439 | 57 |
| `47dc388b25` | `others/mcmegan/b9430a4bcb9177f3.txt` | semi_stall | 0.440 | 50 |
| `8e796cacc7` | `others/giraffe/c872b774d3ace5fb.txt` | offense | 0.441 | 93 |
| `3e9bdcee48` | `others/yak_attack/f14364dcfac7e015.txt` | semi_stall | 0.446 | 148 |
| `2ad1da4e32` | `others/giraffe/c9aebfe958515937.txt` | stall | 0.450 | 229 |
| `a9f6bcf79c` | `others/johnnyg2/c93248f100dbd588.txt` | semi_stall | 0.450 | 80 |
| `a9edcd1df5` | `others/mcmegan/dacb57de93f956a6.txt` | semi_stall | 0.451 | 82 |

### Source-folder spread — checked, because archetype is not the only diversity axis

`data/teams/others/` is five AUTHOR folders. A fleet drawn deep into one author inherits that
author's building habits as surely as it would inherit a single archetype, and this tree has the
precedent: `yak_attack` was 66% of team draws before the 1601→719 dedupe. So it was measured
rather than assumed:

| folder | in slate | slate share | pool share |
|---|---|---|---|
| `giraffe` | 15 | 38% | 51% |
| `sample` | 8 | 20% | 4% |
| `yak_attack` | 8 | 20% | 24% |
| `johnnyg2` | 5 | 12% | 13% |
| `mcmegan` | 3 | 8% | 5% |
| `schmuck_nick` | 1 | 2% | 2% |

**No cap was needed.** The largest folder (`giraffe`) is 38% of the slate against 51% of the
pool — the headroom ranking already draws it BELOW proportion, so an author cap would only
displace measured teams for nominated ones. Re-check this table if the slate is re-cut; the
`path` field in the JSON carries the folder for exactly that purpose.

### Archetype spread

| class | in slate | share |
|---|---|---|
| balance | 10 | 25% |
| stall | 9 | 22% |
| hyper_offense | 9 | 22% |
| offense | 6 | 15% |
| semi_stall | 6 | 15% |

Max class share **25%** ≤ the 40% cap. (Pool-wide the classes run 12.5–28.7%, so the cap binds on `balance` alone.)

## 6. Draft assignments

Both drafts deal the same 40 teams; only the shape differs. Teams are ordered by archetype then
headroom and dealt round-robin, so each teacher gets a spread rather than a monoculture — the
v8 10-team structure logic (a teacher that only ever sees stall learns stall, and the fold
inherits that narrowness).

### 6a. WIDE — 5 teachers × 8 teams (the endorsed shape if rev-4's 3×8 wins)

**T1** (balance×2, hyper_offense×2, offense×1, semi_stall×2, stall×1)  
```
--trainee-teams data/teams/others/mcmegan/5d4bc12fda99fc04.txt,data/teams/others/yak_attack/312a711e196e3c48.txt,data/teams/others/giraffe/995cc8a375500826.txt,data/teams/others/giraffe/2778004a133b19d4.txt,data/teams/others/johnnyg2/be96c6d78622e609.txt,data/teams/others/giraffe/b65ac0cbc96406c0.txt,data/teams/others/mcmegan/c16c4b77942e1a00.txt,data/teams/others/yak_attack/48bdf3938e72aa6b.txt
```

**T2** (balance×2, hyper_offense×2, offense×1, semi_stall×1, stall×2)  
```
--trainee-teams data/teams/others/yak_attack/a8f59ed79cd79215.txt,data/teams/sample/e11829f0561ef5a9.txt,data/teams/others/giraffe/50bffebdd7cbc754.txt,data/teams/others/johnnyg2/21d69ab6da7736d5.txt,data/teams/others/giraffe/5f4e66573157e347.txt,data/teams/others/giraffe/ec21f99b72e5767d.txt,data/teams/sample/9f27f5d3e34021a7.txt,data/teams/others/johnnyg2/440260e5240a694a.txt
```

**T3** (balance×2, hyper_offense×2, offense×1, semi_stall×1, stall×2)  
```
--trainee-teams data/teams/others/yak_attack/70c06b0328863535.txt,data/teams/sample/a04c29cf769e9a11.txt,data/teams/others/giraffe/a916e380aef589cf.txt,data/teams/others/johnnyg2/5f8ac094d96bff26.txt,data/teams/others/yak_attack/00b19e5d8762fcfd.txt,data/teams/others/giraffe/37b7d8c99216ce6a.txt,data/teams/others/mcmegan/7fda48d98ca8efdc.txt,data/teams/sample/9909f2e98e981ccc.txt
```

**T4** (balance×2, hyper_offense×2, offense×1, semi_stall×1, stall×2)  
```
--trainee-teams data/teams/others/giraffe/6b0e85bcbb852991.txt,data/teams/sample/ce35b7368c3d692e.txt,data/teams/others/giraffe/e216c746abc4f927.txt,data/teams/others/giraffe/55f9b006e460f138.txt,data/teams/others/johnnyg2/bcd3290ff6824219.txt,data/teams/others/yak_attack/48c20534cc938c3e.txt,data/teams/others/giraffe/5578a6f415bdef2f.txt,data/teams/sample/f7ba5702fe856292.txt
```

**T5** (balance×2, hyper_offense×1, offense×2, semi_stall×1, stall×2)  
```
--trainee-teams data/teams/others/giraffe/6f0228da330b90e5.txt,data/teams/sample/dbf81d8ecae51c39.txt,data/teams/others/giraffe/d7f8595af2ef442a.txt,data/teams/others/schmuck_nick/836c2cc75f027b43.txt,data/teams/others/giraffe/d63250ac4a5b20ff.txt,data/teams/others/yak_attack/5874c12610c3f415.txt,data/teams/others/yak_attack/6c99ed9a5fb079b9.txt,data/teams/sample/b89e1e37caa40e6a.txt
```

### 6b. NARROW — 20 teachers × 2 teams (if the discriminator picks narrow)

| teacher | teams | archetypes |
|---|---|---|
| N1 | `others/mcmegan/5d4bc12fda99fc04.txt`, `others/johnnyg2/be96c6d78622e609.txt` | balance / offense |
| N2 | `others/yak_attack/a8f59ed79cd79215.txt`, `others/giraffe/5f4e66573157e347.txt` | balance / offense |
| N3 | `others/yak_attack/70c06b0328863535.txt`, `others/yak_attack/00b19e5d8762fcfd.txt` | balance / offense |
| N4 | `others/giraffe/6b0e85bcbb852991.txt`, `others/johnnyg2/bcd3290ff6824219.txt` | balance / offense |
| N5 | `others/giraffe/6f0228da330b90e5.txt`, `others/giraffe/d63250ac4a5b20ff.txt` | balance / offense |
| N6 | `others/yak_attack/312a711e196e3c48.txt`, `others/giraffe/b65ac0cbc96406c0.txt` | balance / semi_stall |
| N7 | `sample/e11829f0561ef5a9.txt`, `others/giraffe/ec21f99b72e5767d.txt` | balance / semi_stall |
| N8 | `sample/a04c29cf769e9a11.txt`, `others/giraffe/37b7d8c99216ce6a.txt` | balance / semi_stall |
| N9 | `sample/ce35b7368c3d692e.txt`, `others/yak_attack/48c20534cc938c3e.txt` | balance / semi_stall |
| N10 | `sample/dbf81d8ecae51c39.txt`, `others/yak_attack/5874c12610c3f415.txt` | balance / semi_stall |
| N11 | `others/giraffe/995cc8a375500826.txt`, `others/mcmegan/c16c4b77942e1a00.txt` | hyper_offense / semi_stall |
| N12 | `others/giraffe/50bffebdd7cbc754.txt`, `sample/9f27f5d3e34021a7.txt` | hyper_offense / stall |
| N13 | `others/giraffe/a916e380aef589cf.txt`, `others/mcmegan/7fda48d98ca8efdc.txt` | hyper_offense / stall |
| N14 | `others/giraffe/e216c746abc4f927.txt`, `others/giraffe/5578a6f415bdef2f.txt` | hyper_offense / stall |
| N15 | `others/giraffe/d7f8595af2ef442a.txt`, `others/yak_attack/6c99ed9a5fb079b9.txt` | hyper_offense / stall |
| N16 | `others/giraffe/2778004a133b19d4.txt`, `others/yak_attack/48bdf3938e72aa6b.txt` | hyper_offense / stall |
| N17 | `others/johnnyg2/21d69ab6da7736d5.txt`, `others/johnnyg2/440260e5240a694a.txt` | hyper_offense / stall |
| N18 | `others/johnnyg2/5f8ac094d96bff26.txt`, `sample/9909f2e98e981ccc.txt` | hyper_offense / stall |
| N19 | `others/giraffe/55f9b006e460f138.txt`, `sample/f7ba5702fe856292.txt` | hyper_offense / stall |
| N20 | `others/schmuck_nick/836c2cc75f027b43.txt`, `sample/b89e1e37caa40e6a.txt` | offense / stall |

### 6c. ⚠️ The budget arithmetic, which is the same for BOTH shapes

Read from the frozen argvs, not assumed (`--steps` minus rev-1 final's 25,067,760):

| fleet | teachers × teams | per teacher | **per team** | total |
|---|---|---|---|---|
| rev-3 (F6a–f) | 6 × 2 | 5.0M | **2.5M** | 30.0M |
| rev-4 (R4S3a–c) | 3 × 8 | 10.0M | **1.25M** | 30.0M |
| a 40-team fleet at the SAME total | 5 × 8 *or* 20 × 2 | 6.0M / 1.5M | **0.75M** | 30.0M |
| a 40-team fleet at rev-4's per-team budget | 5 × 8 *or* 20 × 2 | 10.0M / 2.5M | **1.25M** | 50.0M |

**The shape choice does not change the per-team budget** — 5×8 and 20×2 deal the same 40 teams,
so at a fixed total they both land on 0.75M/team. What the discriminator decides is *breadth per
teacher*, and that is orthogonal to this row. The real hazard is shared:

> **0.75M/team is BELOW every budget ever measured here.** The budget-invariance result
> (61608ac §2: +67% ⇒ +0.0019, z=0.16) was measured between **1.5M and 2.5M**, and rev-4 already
> steps outside it at 1.25M. Invariance over a tested band is not licence to extrapolate below
> it — the ceiling account predicts a floor somewhere, and nothing has looked for it. **Either
> budget the 40-team fleet at 50M total (1.25M/team, matching rev-4), or measure the 0.75M point
> first.** Rev-4's own absolute rows give that check for free: if its 1.25M teachers land at the
> same ~0.69 as rev-3's 2.5M ones, the band widens downward by one point of evidence — read it
> before committing 40 forks to 0.75M.

## 7. What needs a FRESH screen before final selection

**The rule, from the withdrawn §7 seniority split (61608ac):** selecting a team on an estimate
and then reporting that same estimate is selection-on-the-minimum. The measured regression-to-mean
there was **+0.061** — larger than most of the differences this slate ranks on. So:

* **The 20 PROVISIONAL rows (§5b) and any UNMEASURED promotion (§5c) need a screen at n ≥ 200,**
  played on **fresh games** with a seed family disjoint from `coverage_sweep`'s
  (`52000 + idx`), `headroom_screen`'s (`1000 + 9 + idx`) and the admission's (`41000 + idx`).
  Reusing any of those makes the confirm a re-report of the selection.
* **`coverage_sample.json` is a live instance of the failure it warns about**: its `screen` and
  `confirm` blocks are byte-identical for all 23 rows (`screen_wr == wr` exactly), so its
  "two-stage" structure collapsed to one measurement. Its numbers are usable as a single n=200
  estimate — which is how they are used here — but **not** as a confirmation of anything.
* **`coverage_sweep.json` did it right**: n=40 screen → an INDEPENDENT n=200 confirm on the
  bottom-12, and the confirm values move (0.275→0.380, 0.425→0.370). Those 12 rows are the
  strongest pool-team evidence in the tree and are used at face value.
* **The 8 untaught curated teams are being measured RIGHT NOW.** `probes/run_headroom.sh` is
  running `headroom_screen.py` over the 20 non-taught curated teams at n=150 against R2-ACTION
  final; all 8 sit at screen indices 8–19. **Re-run this slate when it finishes.** Note what
  that actually buys: n=150 DIRECT is not a larger sample than the n=200 SHIFTED number they
  carry here — it is a sample that needs no generation-offset assumption. Prefer the direct
  one for that reason, not for precision, and treat agreement between the two as the real
  signal (the offset was fitted on these same teams, so a large disagreement would indict it).
* **The ceiling itself deserves one manipulation before 40 teachers are spent on it.** Every
  headroom number is `0.69 − baseline`; if F6-CURR's absolute row (requested at 61608ac §9)
  shows a curriculum that lifts 0.69, the whole ranking is against a moving bar.

## 8. Provenance

| artifact | what it contributed |
|---|---|
| `/home/goodlad/.claude/jobs/1046b1d6/tmp/probes/r3_admission.json` | the rev-3 admission battery — the `target` baseline (12 teams, n=400) AND the 12 teacher ABSOLUTE cells the ~0.69 ceiling is the mean of |
| `/home/goodlad/.claude/jobs/1046b1d6/tmp/probes/fleet_admission.json` | the rev-2 battery — the `rev1final` pilot row on the 9 meter teams (n=400) |
| `/home/goodlad/.claude/jobs/1046b1d6/tmp/probes/pilot_R2ACTION_n300.json` | current-generation piloting of the 9 meter teams (n=300) |
| `/home/goodlad/.claude/jobs/1046b1d6/tmp/probes/cov_R2ACTION.json` | current-generation piloting of the 3 coverage teams (n=300) |
| `/home/goodlad/.claude/jobs/1046b1d6/tmp/probes/cov_rev1fin.json` | the rev-1-final companion of the row above — the generation-offset anchor |
| `/home/goodlad/.claude/jobs/1046b1d6/tmp/probes/headroom_screen.json` | the LIVE n=150 screen over the 20 non-taught curated teams |
| `/home/goodlad/.claude/jobs/1046b1d6/tmp/probes/coverage_sample.json` | the 23 non-meter curated teams at n=200 under the rev-1 pilot |
| `/home/goodlad/.claude/jobs/1046b1d6/tmp/probes/coverage_sweep.json` | 80 POOL teams at n=40 + an independent n=200 confirm on the bottom 12 |
| `models/<run>/metadata.json:team_win_rates` | the task-#18 per-team tracking — 658 teams, nomination signal only |
| `data/teams/gen3_team_archetypes.json` | the pace-class label and tags for all 719 teams |

**Freshness of the live source:** 6 of 20 `headroom_screen` rows had landed when this was built.

The two admission artifacts and the piloting/coverage probes live in a **session-scoped job
directory** (`~/.claude/jobs/1046b1d6/tmp/probes`), not in the repo. Every number that this slate
depends on is copied into `team_slate_40.json` so the slate survives that directory's deletion.

**Re-running this slate.** `designs/ai_v12/team_slate_build.py` regenerates both files from the
artifacts above in ~5 s (read-only; no models, no battles). Run it after the in-flight headroom
screen finishes — it promotes the 8 untaught curated teams from tier B to tier A and re-fits the
generation offset.

```bash
export PYTHONPATH=$PYTHONPATH:src
python designs/ai_v12/team_slate_build.py
```

**Key-convention warning, recorded because it cost a rejoin.** `coverage_sample.py` fingerprints
a team with `sha1(text)` on the **UNSTRIPPED** text, while `team_archetypes.team_sha`,
`MatchupSpec` pins, `TeamWinRateCallback` and this slate all use `sha1(text.strip())[:10]`. The
tell is that all 23 of its rows carry `"class": "?"` — its own archetype lookup silently missed.
The 23 identities were recovered by re-hashing the curated files under both conventions (23/23
matched on `raw`). Sixth specimen of the recorded-vs-effective derived-key genre.

