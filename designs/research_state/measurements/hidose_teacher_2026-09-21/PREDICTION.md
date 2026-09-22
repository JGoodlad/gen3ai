# PRE-REGISTRATION — `ai_v13_18_teach5_offense_hidose`: is the era-2 teacher failure a DOSE effect or a TEAM-LEVEL CEILING?

**Committed BEFORE the arm exists.** Nothing in this directory has been measured on
`ai_v13_18_teach5_offense_hidose`, because the arm has not been launched. This session **PREPARED
and VALIDATED** the argv (§2) and **did not run it** — the launch is the Training Run session's,
verbatim from [`launch.sh`](launch.sh).

The GO is the 2026-09-21 admission entry (`be03bd74`, *NONE of the three 5-team teachers is
admitted*) plus the stage-A read of §1.3, whose two surviving accounts this arm is built to
separate.

---

## 1. The question

### 1.1 What is already measured

Four teachers have now been manufactured **on the plateau parent**
(`models/ai_v13_12_plateau/final_model.zip` @95,158,272) and read through the **same admission
gate** — each on its own five teams, 800 games/team, against the fixed third party
`untaught_meter_opponent` (= `ai_v9_29_rev1_0823@24,000,000`), both refs in ONE
`main.untaught_meter` invocation (CRN), paired bootstrap over the five TEAMS (20,000 draws, one
shared index set, seed 20260915).

| teacher | recipe | realized dose | vs-target (head-to-head) | **gate Δ vs the parent** | 95 % CI | ADMITTED |
|---|---|---:|---:|---:|---|---|
| `ai_v13_13_exploit5_offense` | `--exploiter` plateau, bots 50 % | 8.392e-9 = **0.39×** | **0.657** [0.610, 0.702] ✅ | **−1.00 pp** | [−4.80, +2.13] | **NO** |
| `ai_v13_14_exploit5_balance` | same | 8.392e-9 = **0.39×** | 0.525 [0.476, 0.573] ✗ | **+0.58 pp** | [−0.32, +1.28] | **NO** |
| `ai_v13_15_exploit5_stall` | same | 8.392e-9 = **0.39×** | 0.455 [0.407, 0.504] ✗ | **−1.12 pp** | [−3.37, +0.88] | **NO** |
| `ai_v13_16_teach5_offense_dist` (**stage A**) | `--self-play`, `--fork-lr 5.5e-5 --fork-lr-freeze` | 8.392e-9 = **0.39×** | — (no target) | **−1.37 pp** | [−2.78, +0.27] | **NO** |

Every Δ sits inside ±1.4 pp against a 4.75 pp floor at the loosest registered bar. The gate is
**decisive, not marginal**, and it has now refused a teacher built with an explicit opponent
(`--exploiter`) and a teacher built without one (`--self-play`), on the same parent, the same five
teams and the same dose.

> The stage-A row is derived here from `admission_offense_dist.json`
> (`~/.claude/jobs/1046b1d6/tmp/admission/out/`) with the paired bootstrap copied VERBATIM from
> `admission_delta.py` — same 20,000 draws, same seed 20260915, the TEAM as the unit. Levels:
> teacher **58.55 pp** (2342/4000), plateau parent **59.92 pp** (2397/4000), 0 timeouts. Per team:
> −1.87 / −3.12 / −0.63 / **+1.62** / −2.88 pp.

### 1.2 🚨 THE TWO ACCOUNTS THE STAGE-A READ COULD NOT SEPARATE

Stage A was the arm that was supposed to settle this and did not, because it moved the RECIPE
(`--exploiter` → `--self-play`) while holding the DOSE at 0.39×. Two accounts survive it, and they
make opposite predictions about the arm registered here:

* **(A) DOSE.** Every era-2 teacher ran at **0.39× the v8 reference** — `8.392e-9`, against the
  three era-1 exploiters' **1.78× / `3.815e-8`**, a **4.5× gap** caused by `--fork-lr` being unset
  in the era-2 template (`7afa2b34`). The era-2 offense teacher's own vs-target curve was **still
  CLIMBING at the endpoint** (0.590 → 0.670 → 0.670 → 0.700) where all three era-1 exploiters had
  already flattened at 0.740 by +3M. On this account the teachers are simply **under-trained**: the
  optimiser never moved them far enough from the parent for a piloting difference to exist, and a
  4.5× larger step within the same +8M budget would produce one.
* **(B) TEAM-LEVEL CEILING.** The plateau parent is a full-pool generalist that has **provably
  stopped gaining** on its off-slice meter (`3459ecce`, Δ −1.50 pp [−3.75, +0.62] within a 3.69 pp
  floor). On this account it is already at the attainable level on these five teams against this
  opponent, the residual is variance the policy cannot remove, and **no dose** converts +8M of
  specialisation into a measurable piloting gain. The supporting observation is that the one
  teacher that DID beat the parent head-to-head is the **most negative** on the gate (−1.00 pp):
  what a bigger step buys is **target-specific counterplay**, which the gate is built not to
  reward.

**The two accounts are separated by exactly one lever: the dose.** This arm moves that lever and
nothing else.

### 1.2b 🚨 AMENDMENT — the stage-A ledger entry names a THIRD account

Landed while this registration was being written: `cc02cff5`, *the REDESIGN fails the same way*.
It banks the stage-A number this document re-derives (**−1.37 pp [−2.78, +0.27]**, with the
plateau-parent column reproducing **59.92 pp** to the decimal on the same five teams at the same
seed — so the teacher column is the only thing that moved), and it names **three** live accounts
rather than two: dose/budget, the team-level ceiling, **and a third-party meter that cannot resolve
differences this small**.

That third account is real and this arm does **not** separate it. Registered consequence, stated
here rather than discovered later:

* If the arm **is** admitted, the meter demonstrably CAN resolve a difference on this slice, and the
  third account dies with the result — branch (a) is unaffected.
* If the arm is **not** admitted, branch (b) must be written as **"the team-level ceiling OR meter
  resolution"**, not as the ceiling alone. The tie-break is a **positive control** the campaign does
  not yet have: a pair of refs on these five teams whose gap on this meter is known to be large.
  The cheapest candidate is the plateau parent against a much weaker banked ref (`armW`, which is
  **14.00 pp** below it on the untaught 8 — 46.19 vs 60.19); if the meter reads that gap at 800
  games/team on THIS slice, resolution is not the constraint and the ceiling account stands alone.
  **That control is NOT run here and is not part of this registration** — it is named so that branch
  (b) cannot be over-read.

`cc02cff5` also takes branch (b) *at the era-2 dose* and launches `ai_v13_17_fold_k1` as a COST
read. This arm does not reverse that: it asks whether branch (b) also holds at **1.78×**, which is
the one dose at which a teacher on this lineage has ever been observed to flatten.

### 1.3 What this arm is

`ai_v13_18_teach5_offense_hidose` — **`ai_v13_13_exploit5_offense`'s own resolved argv, with the
dose put back to era-1's.** Fork of the plateau parent, the SAME five offense teams, the SAME
+8,000,000 steps, the SAME seed 1001, `--team-block-episodes 1`, trained against the frozen
plateau parent exactly as `ai_v13_13` was (`--exploiter models/ai_v13_12_plateau/final_model.zip
--exploiter-keep-bots`, **no `--self-play`**). The offense recipe is chosen because it is the only
one of the three that demonstrably exploited its target — if the dose account is right anywhere, it
is right there.

**THE ONE CHANGE: `--fork-lr 2.5e-4 --fork-lr-freeze`.** (Plus `--run-name`, which is not a lever.)

---

## 2. The argv, verbatim, and its dose arithmetic

### 2.1 🚨 WHY `2.5e-4`, stated as arithmetic rather than asserted

`main.dose`'s formula, read from `agents.training.dose` and printed by the tool:

```
updates_per_env_step = n_epochs / (batch_size * grad_accum_steps)
dose_rate            = lr_median * updates_per_env_step
```

Every run in this lineage shares one shape — `--batch-size 2048 --grad-accum-steps 32
--n-epochs 10` — so:

```
effective batch      = 2048 * 32                = 65,536
updates_per_env_step = 10 / 65,536              = 1.52588e-4      (identical in EVERY row of out/dose.txt)
```

The era-1 exploiters' measured dose is **3.815e-8 = 1.78×** the v8 reference
(`ai_v8_14_distill3_0725`, `dose_rate = 2.145e-8`). Inverting the formula at this fixed shape:

```
lr = 3.815e-8 / 1.52588e-4 = 2.500e-4
```

and forward, as a check:

```
2.5e-4 * 1.52588e-4 = 3.8147e-8        ⇒ 3.8147e-8 / 2.145e-8 = 1.778 ≈ 1.78x   ✅
5.5e-5 * 1.52588e-4 = 8.392e-9         ⇒ 0.39x   (the era-2 dose, reproduced)   ✅
2.8e-5 * 1.52588e-4 = 4.272e-9         ⇒ 0.20x   (the plateau parent's own)      ✅
```

**`--fork-lr 2.5e-4` is therefore the value that reproduces the era-1 exploiters' dose at this
batch and these epochs — not an approximation of it, the same number.** `--fork-lr-freeze` is
required as well, and is not cosmetic: without it the KL controller runs live and the lr is **not
stationary within the arm** (that is precisely how era-2 ended at a median of 5.5e-5 after starting
at 2.8e-5, `7afa2b34`). A dose claim about a run whose lr wandered is not a dose claim.

The full table is [`out/dose.txt`](out/dose.txt).

### 2.2 The argv — VERBATIM ([`argv_hidose.txt`](argv_hidose.txt))

Everything below the launcher entry point. **230 tokens; the template
(`models/ai_v13_13_exploit5_offense/metadata.json` → `original_command`) is 227.**

```
--device cuda --log-level periodic --belief-grad-mode shaping --beta-setvalued-coef 0.05 --bias-additivity 1.0 --clip-range 0.15 --clip-range-vf none --compile-opponents --compile-opponents-strict --compile-trainer --consequence-topk 6 --damage-candidate-k 0 --damage-matrices both --damage-op --damage-outgoing --damage-topk 6 --defensive-entropy-anneal-frac 0.0 --defensive-entropy-boost 1.0 --device cuda --distill-coef 0.0 --distill-value-coef 0.0 --distill-value-feat-coef 0.0 --edge-bias-families d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5,h,r --ent-coef 0.05 --entity-tail-seats --entity-topk-seats 6 --eval-battles 100 --eval-concurrency-per-worker 1 --eval-device cpu --eval-shard-games 25 --eval-workers 5 --exploiter-bot-fraction 0.5 --exploiter-temp-anneal-frac 0.2 --exploiter-temp-end 1.0 --exploiter-temp-mode fixed --exploiter-temp-ratchet-factor 0.9 --exploiter-temp-ratchet-games 500 --exploiter-temp-ratchet-wr 0.55 --hp-belief-mode composed --hp-type-belief-coef 0.05 --intent-move-cell --keep-crashes 10 --keep-eval-snapshots 10 --keep-eval-trace-steps 20 --keep-stalls 50 --log-level periodic --lr 0.0003 --mat-alive-weight 1.25 --min-lr 1e-05 --move-belief-coef 0.05 --move-belief-latent-coef 0.05 --move-belief-mode both --move-candidate-floor 0.02 --move-latent --move-prior-fusion --n-envs 48 --n-epochs 10 --n-sentinels 5 --n-steps 2048 --no-progress-penalty 0.15 --opd-beta 1.0 --opd-coef 0.0 --opp-belief-aux-coef 0.05 --opp-belief-cls-k 6 --opp-belief-moves-weight 1.0 --opp-intent-coef 0.05 --search-teacher-batch-size 256 --search-teacher-beta 1.0 --search-teacher-buffer-size 20000 --search-teacher-coef 0.0 --search-teacher-value-coef 0.0 --self-ko-hp-penalty 0.0 --snapshot-ladder-games 100 --species-prior-fusion --spread-belief --spread-belief-coef 0.05 --spread-belief-nature --switch-bias-weight 0.0 --t0-species-prior --teacher-confirm-rollouts 8 --teacher-gen-battles 12 --teacher-refresh-steps 500000 --teacher-search-budget 200 --teacher-search-freq 0 --teacher-search-workers 3 --team-block-episodes 1 --team-pfsp off --team-pfsp-cap 3.0 --team-pfsp-floor 0.05 --unified-damage both --unified-moves both --use-bridge rust --value-entity-pool --value-threat-inject --warmstart-battles 200 --warmstart-bc-steps 4000 --weight-decay 1e-05 --win-prob-mode shaping --history-events --item-belief --intent-threshold --intent-conditional --op-drop-renders --op-believed-lean --value-entity-pool-full --pair-outcome-cell --pair-outcome-switch --switch-branch-cell --conditional-threat-cell --critic winprob --no-hand-shaping --terminal-indicator --victory-value 1.0 --draw-penalty 0 --vf-coef 0.5 --intent-label-bot-weight 0.25 --batch-size 2048 --grad-accum-steps 32 --restart-interval-hours 3 --steps 103158272 --seed 1001 --eval-sentinel-greedy --no-value-true-team --pin-commit 6eb9c776940ed6040ddb90acfbc28b038457fc42 --run-name ai_v13_18_teach5_offense_hidose --model models/ai_v13_12_plateau/final_model.zip --exploiter models/ai_v13_12_plateau/final_model.zip --trainee-teams data/teams/sample/9eb3abdc52876a63.txt,data/teams/sample/ac17a9dde5.txt,data/teams/sample/a185b2d193.txt,data/teams/sample/9ba039ba8a.txt,data/teams/sample/b904dbe059.txt --exploiter-keep-bots --fork-lr 2.5e-4 --fork-lr-freeze
```

### 2.3 The TOKEN DIFF against the template — the whole of it

```
- ai_v13_13_exploit5_offense          (--run-name)
+ ai_v13_18_teach5_offense_hidose     (--run-name)
+ --fork-lr 2.5e-4
+ --fork-lr-freeze
```

Nothing else moves. `--steps 103158272` is a **TOTAL** and the parent is the same checkpoint, so
this is the same **+8,000,000** the template ran. The five `--trainee-teams` paths are byte-identical
to the offense set in `teamsets_5x3.json` (anchor `9eb3abdc52876a63`), none of which is a member of
the untaught 8 — and the `gen3_untaught_teacher_guard_v1` startup guard (`1393d192`) now enforces
that by content sha rather than by luck.

### 2.4 🚨 VALIDATED BY EXECUTING, not by clause-checking

Both commands were run from the MAIN checkout on 2026-09-21, CPU-only, `nice 15`; full transcripts
in [`out/checkargs.txt`](out/checkargs.txt) and [`out/dry_run.txt`](out/dry_run.txt).

**`python -m main.checkargs --argv "$(cat argv_hidose.txt)"`**

```
checked 131 flags from --argv
  accepted by the PINNED parser  : 129      (template: 127)
  launcher-owned (not forwarded) : 2        (template: 2)
  unrecognized                   : 0        (template: 0)
  ARCH SURFACE ... ✓ every ARCH-surface key matches the production mirror
  ✓ this command still launches
```

The template `models/ai_v13_13_exploit5_offense` was re-checked in the same transcript and returns
the same verdict line. **THE ARCH-SURFACE DIFF IS EMPTY**: both argvs resolve every ARCH-surface
key to the production mirror (`production_config@360f8378dd90`; 39 of 51 registry toggles are the
ARCH surface), so there is no key on which they can differ. The only difference between the two
`checkargs` reports is the **+2 accepted flags** — `--fork-lr` and `--fork-lr-freeze`, the lever —
and the `--run-name`.

**`python -m main.launcher --dry-run $(cat argv_hidose.txt)`**

```
role        : FORK of /home/goodlad/dev/gen3ai/models/ai_v13_12_plateau
run dir     : models/ai_v13_18_teach5_offense_hidose   [would be created]
pin         : 6eb9c776940ed6040ddb90acfbc28b038457fc42  (source: pin_commit)
steps       : --steps 103,158,272 vs checkpoint at 95,158,272 (sidecar) → +8,000,000 steps
transport   : in-process bridge [rust] (no Showdown server)
effective   : 56 unset flag(s) INHERITED from the parent's model_config.json
      --fork-lr              0.00025      (from the argv)
      --fork-lr-freeze       True         (from the argv)
      --distill-coef         0.0          (from the argv)
✓ DRY RUN — this command would launch. Nothing was created or modified.
```

A `--dry-run` was used rather than "launch and kill" deliberately: the latter is harmless on a fork
and **destructive on a restart**, and the habit is what matters.

**What the dry run CANNOT check** (it says so itself, and they are the launch-time gates): the
architecture compatibility check against the checkpoint, the `ModelVersion` round-trip smoke, the
resolved `--compile-*` decisions, the self-play pool seeding, the obs dimension and arch signature.
Those are the first two minutes of the real launch and nothing offline substitutes for them.

---

## 3. THE READ — registered in full, before the arm exists

### 3.1 The ADMISSION cell (the primary endpoint)

**Copied VERBATIM from the 2026-09-21 gate** (`~/.claude/jobs/1046b1d6/tmp/admission/run_admission.sh`
and `admission_delta.py`), which itself reused `split_lossoff_read_2026-09-20/scripts/harness/run_slice.sh`.
The ONLY change is the teacher ref.

* refs: `teacher=models/ai_v13_18_teach5_offense_hidose/final_model.zip` **and**
  `plateau_parent=models/ai_v13_12_plateau/final_model.zip`, **both in ONE invocation** so they see
  identical teams and identical dice (CRN);
* teams: **its own five offense teams, in REGISTERED order** — the order IS the seed offset, so the
  manifest `teams_offense.json` is reused byte-identically;
* opponent: the registry name **`untaught_meter_opponent`** (= `ai_v9_29_rev1_0823@24,000,000`) — a
  fixed THIRD PARTY, never a sentinel (a sentinel is the trainee's own snapshot and so differs
  between refs);
* `--games-per-team 800` ⇒ **8,000 battles**; `--seed 0`, concurrency 1, `--workers 5`, `nice 15`,
  CPU-only, from the MAIN checkout;
* statistic: **paired bootstrap over the five TEAMS** (rule 10 — the team is the unit), 20,000
  draws, ONE shared index set, **seed 20260915**, verbatim from `admission_delta.py`.

The harness is committed here and is runnable as-is:
[`scripts/harness/run_admission_hidose.sh`](scripts/harness/run_admission_hidose.sh) with the frozen
manifest [`scripts/harness/teams_offense.json`](scripts/harness/teams_offense.json), adjudicated by
[`scripts/hidose_delta.py`](scripts/hidose_delta.py).

### 3.2 The COLLATERAL cell (the untaught 8) — read whichever way the primary goes

`main.untaught_meter` with the **untaught-8 manifest in its canonical order** (the harness default,
as in `split_lossoff_read_2026-09-20/scripts/harness/run_untaught.sh`), registry opponent
`untaught_meter_opponent`, `--seed 0`, concurrency 1, both refs in ONE invocation. The plateau
parent's banked level is **60.19 pp** (`3459ecce`); the registered off-slice floor is **3.69 pp**.
This cell exists because a 4.5× dose increase applied to a five-team slice is exactly the shape that
should cost off-slice if anything does, and because the era-1 fold's whole indictment was an
off-slice shortfall. Harness:
[`scripts/harness/run_untaught_hidose.sh`](scripts/harness/run_untaught_hidose.sh) — it passes no
`--teams`, deliberately, so the tool's canonical untaught-8 manifest is the one that is used.

### 3.3 THE BAR — one rule, stated before the first battle

Identical in wording to the gate it reuses:

> **OUTSIDE the floor iff `|Δ| > floor` AND the 95 % CI excludes the floor POINT.**
> **ADMITTED iff OUTSIDE *and* ABOVE the parent.** Not admitted ⇒ **report, do not fold.**

* admission floors: **4.75 pp (low) and 8.50 pp (high)**, the two slice cells of the 2026-09-20
  read. `ADMITTED` requires the verdict at **both** floors, exactly as `admission_delta.py` computes
  it — the high floor is the binding one.
* collateral floor: **3.69 pp**, the off-slice floor every read in this series has used.
* ⚠️ WITHIN FLOOR is **never** "equivalent" (rule 6), and the floor's replicate level is the RUN
  (rule 19): this is one run against one run, so a within-floor verdict is **NOT DETECTED**, not
  "proven flat".

### 3.4 PRIOR — registered, and it is against the arm

**P(ADMITTED) ≈ 0.25.** Subjective, and here is the reasoning on both sides so a reader can price it.

*Against admission (why 0.25 and not 0.5):* the era-1 exploiters ran at **exactly this dose** and
the control read (`58389caa`) still found them "level with or below the continuation **on their own
team**" — the one prior datum at 1.78× points the same way the era-2 gate does. The gate is
decisive rather than marginal on all four existing teachers (max |Δ| 1.37 pp against a 4.75 pp bar),
the per-team signs are mixed in every cell, and the head-to-head winner is the most negative — the
established mechanism (`be03bd74`) is that this recipe manufactures **target-specific counterplay**,
which more dose should produce MORE of, not less.

*For admission (why 0.25 and not 0.05):* the era-2 offense curve was genuinely **unconverged** —
still climbing at +6.8M where 1.78× arms flattened by +3M — so the era-2 teachers are not a fair
test of what this recipe produces at convergence; and no era-1 teacher was ever measured under THIS
protocol against a plateaued parent, so account (A) has never actually been refuted, only
never supported.

**A registered secondary prediction, independent of the gate:** the vs-target curve reaches **≥0.72
by +3M and flattens**, reproducing the era-1 shape, because that is what the dose bought there.
If the arm is not admitted but its vs-target curve DOES reproduce era-1's, that is the strongest
possible form of branch (b) — same exploit, same dose, still nothing to teach.

### 3.5 BRANCHES — registered in advance, including the uncovered case

**(a) ADMITTED** (Δ OUTSIDE 4.75 **and** 8.50, ABOVE the parent) ⇒ **DOSE was the account.**
Teachers exist on this parent at **1.78×** and the era-2 null was an artefact of an unset
`--fork-lr`. Consequences, registered now so they are not negotiated afterwards: the balance and
stall archetypes get **the same treatment** — re-run at `--fork-lr 2.5e-4 --fork-lr-freeze`, matched
to each other and to this arm, before any K ladder, because a one-teacher ladder is a weaker
contrast than a three-teacher one and the archetype spread (offense − stall = +0.245 head-to-head)
is the only clean within-era finding the campaign holds. Stage 3 is then re-registered on the
ADMITTED set.

**(b) NOT ADMITTED** (within floor, either sign) ⇒ **the TEAM-LEVEL CEILING *or* METER
RESOLUTION (§1.2b) is the surviving account for this parent, and the teacher line is CLOSED for it
at this dose too.** Five teachers — three exploiters,
one self-play specialist, one at 4.5× the dose — would then have failed the same gate on the same
parent across two recipes and two doses. The honest statement is scoped: it closes *manufacturing a
teacher by specialising THIS parent on a pinned team set*, at ≤ +8M, on the offense archetype; it
does **not** close distillation, the K ladder, or teachers built any other way (a differently-
architected teacher, a search teacher, a teacher from a different parent). Nothing is extended to
+12M — the deficit is a near-exact null with mixed per-team signs, not a budget shortfall.

**(c) ADMITTED, WITH COLLATERAL** (gate OUTSIDE and ABOVE, **and** the untaught-8 Δ OUTSIDE the
3.69 pp floor and BELOW the parent) ⇒ **report it as such and do NOT treat the admission as clean.**
A teacher that is better on its five teams and measurably worse on the untaught eight is the exact
shape that convicted the era-1 fold (−9.75 pp off-slice, `58389caa`), and folding from it re-creates
the confound the plateau block was spent removing. The registered consequence is that branch (a)'s
"the same treatment for balance and stall" is **suspended**, and the decision goes back to the
orchestrator with both numbers side by side, because trading off-slice level for on-slice level is a
research choice and not mine to make.

**(d) UNCOVERED — not admitted AND collateral outside the floor below** ⇒ a 1.78× dose on a
five-team slice DAMAGED this parent off-slice while buying nothing on-slice. Registered as its own
outcome rather than folded into (b): it would make the dose actively harmful on this parent and is
an argument against re-running balance and stall at any dose. Report; launch nothing.

**(e) UNCOVERED — admitted BELOW the parent** is impossible by construction (`ADMITTED` requires
`delta_pp > 0`); a Δ OUTSIDE the floor and BELOW is a refusal, and is reported as branch (d)'s
on-slice analogue.

### 3.6 🚨 Facts LOOKED AT while scoping, disclosed rather than hidden

Pre-registration is worthless if the registrant has already seen the answer. Everything read while
scoping this job is declared: the four teachers' gate numbers and per-team rows (§1.1, all banked
before this arm was conceived); the stage-A `admission_offense_dist.json` (re-derived here, §1.1);
`main.dose` on eight runs (§2.1, `out/dose.txt`); the three era-2 and the stage-A `original_command`
strings; `teamsets_5x3.json`; and the live arm `ai_v13_17_fold_k1`'s argv and step count
(99,778,560 at the time of writing). **Nothing about `ai_v13_18_teach5_offense_hidose` has been
measured, because it does not exist.**

---

## 4. What this session did and did not do

* **DID:** built the argv from `ai_v13_13`'s recorded `original_command`, validated it by
  EXECUTING `checkargs` and `launcher --dry-run`, derived the `2.5e-4` from `main.dose`'s own
  formula and verified it three ways, and wrote [`launch.sh`](launch.sh).
* **DID NOT:** launch anything, touch the GPU, write under `models/`, or edit `ledger.md` /
  `UNDERSTANDING.md`. The GPU carried `ai_v13_17_fold_k1` throughout; everything here ran
  `CUDA_VISIBLE_DEVICES="" nice -n 15` on CPU.
* **The launch is the Training Run session's.** `launch.sh` is runnable verbatim and re-validates
  before it launches.

---

## 5. Ready-to-append ledger paragraph

> *(Not appended — `ledger.md` is untouched by this session. This is the paragraph to append when
> the arm is launched or the decision is taken.)*

### 2026-09-21 · OPS · **THE HIGH-DOSE TEACHER IS PREPARED, NOT LAUNCHED — `ai_v13_18_teach5_offense_hidose` puts the dose back to era-1's 1.78× and changes nothing else, so the ONE lever the stage-A read could not move is the only thing that differs.**

`designs/research_state/measurements/hidose_teacher_2026-09-21/` registers, before the arm exists,
the read that separates the two accounts left standing by the 2026-09-21 admission gate: **(A) the
era-2 teachers were under-trained at 0.39×**, or **(B) the plateau parent is at a team-level ceiling
on these five teams and no dose buys a piloting gain.** Four teachers have now failed the same gate
on the same parent — offense **−1.00 pp** [−4.80, +2.13], balance **+0.58** [−0.32, +1.28], stall
**−1.12** [−3.37, +0.88], and stage A `ai_v13_16_teach5_offense_dist` (self-play, frozen 5.5e-5)
**−1.37 pp** [−2.78, +0.27] (58.55 vs 59.92 pp, 8,000 battles, 0 timeouts, re-derived here with
`admission_delta.py`'s own bootstrap, seed 20260915) — every one inside ±1.4 pp against a 4.75 pp
floor, and all four at **8.392e-9 = 0.39×**.

**THE ARM, PREPARED AND VALIDATED BUT DELIBERATELY NOT LAUNCHED.**
`ai_v13_18_teach5_offense_hidose` is `ai_v13_13_exploit5_offense`'s own resolved `original_command`
with a **four-token diff**: `--run-name`, plus **`--fork-lr 2.5e-4 --fork-lr-freeze`** (227 → 230
tokens). Same plateau-parent fork, same five offense teams, same +8,000,000 (a `--steps` TOTAL of
103,158,272), same seed 1001, same `--team-block-episodes 1`, same `--exploiter
models/ai_v13_12_plateau/final_model.zip --exploiter-keep-bots`, no `--self-play`. **The `2.5e-4` is
arithmetic, not a guess:** every run in the lineage shares `--batch-size 2048 --grad-accum-steps 32
--n-epochs 10` ⇒ `updates/step = 10/65,536 = 1.52588e-4`, so era-1's measured **3.815e-8** inverts
to `3.815e-8 / 1.52588e-4 = 2.500e-4`, and forward `2.5e-4 × 1.52588e-4 = 3.8147e-8 = 1.78×` the v8
reference `2.145e-8` — the same number, at the same shape. `--fork-lr-freeze` is load-bearing: era-2
inherited 2.8e-5 WITHOUT the freeze and its live KL controller annealed it up to 8.36e-5, which is
how "same recipe" produced a non-stationary lr. **Validated by EXECUTING**: `checkargs` **131 / 129
accepted / 2 launcher-owned / 0 unrecognized**, the **ARCH-surface diff vs `ai_v13_13` EMPTY** (both
match `production_config@360f8378dd90` on every ARCH key; the only report difference is the +2
fork-lr flags), and `launcher --dry-run` resolves **FORK of `ai_v13_12_plateau` → +8,000,000 steps,
`--fork-lr 0.00025`, `--fork-lr-freeze True`, pin `6eb9c776`, nothing created**.

**THE READ IS REGISTERED IN FULL:** the admission gate verbatim — its own five teams in registered
order, both refs in ONE `main.untaught_meter` invocation (CRN), `untaught_meter_opponent` as the
fixed third party, 800 games/team = 8,000 battles, `--seed 0`, concurrency 1, paired bootstrap over
the five TEAMS at seed 20260915 — **plus a collateral untaught-8 cell read whichever way the primary
goes** (floor 3.69 pp, parent's banked level 60.19 pp). Rule: OUTSIDE iff `|Δ| > floor` AND the CI
excludes the floor point; ADMITTED iff OUTSIDE **and** ABOVE, at **both** floors 4.75 / 8.50.
Prior **P(admitted) ≈ 0.25**, registered against the arm, with a secondary prediction that its
vs-target curve reaches ≥0.72 by +3M and flattens. Branches: **(a) admitted ⇒ dose was the account,
teachers exist at 1.78×, and balance + stall get the same treatment before any K ladder; (b) not
admitted ⇒ the team-level ceiling is the surviving account and the teacher line is CLOSED for this
parent** (five teachers, two recipes, two doses, one gate) — scoped to *specialising THIS parent on
a pinned team set at ≤ +8M*, closing neither distillation nor the K ladder; **(c) admitted WITH an
untaught-8 cost outside the floor ⇒ reported as such, (a)'s consequences SUSPENDED and the trade
goes back to the orchestrator**, because on-slice-for-off-slice is the exact shape that convicted
the era-1 fold; **(d) uncovered — not admitted AND collateral below ⇒ the dose is actively harmful
on this parent; launch nothing.** Tag: **OPS · `ai_v13_18_teach5_offense_hidose` PREPARED, NOT
LAUNCHED · 4-token diff from `ai_v13_13`, ARCH diff EMPTY, checkargs 131/129/2/0, dry-run +8,000,000
clean · dose `2.5e-4 × 1.52588e-4 = 3.815e-8 = 1.78×`, the era-1 value re-derived from the formula ·
stage A `ai_v13_16` re-read at **−1.37 pp** [−2.78, +0.27], the FOURTH gate refusal at 0.39× ·
prior 0.25 AGAINST · four branches + the uncovered case registered · `launch.sh` ready for the
Training Run session; this session launched nothing and touched no GPU**.
