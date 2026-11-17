# X70 follow-up (iii), the entropy coefficient itself: a from-scratch `ent_coef 0` arm, or a schedule: DRAFT pre-registration (2026-11-17)

**Status: DRAFT. NOT REGISTERED. Nothing here is a rule, and nothing launches from it.** Written before X70 has
trained a step (blind by construction). It becomes a registration only if its trigger (§1) fires and the
orchestrator names it; it is then copied into a new folder as `PREREG.md` with every **UNVERIFIED** and "at
registration" item filled, and lands on main before the first step. Tier when registered: **FULL** (a fresh-run
comparison whose WIN would put a coefficient to the owner). Siblings:
[`DRAFT_PREREG_i_matched_entropy.md`](DRAFT_PREREG_i_matched_entropy.md),
[`DRAFT_PREREG_ii_fresh_anchor.md`](DRAFT_PREREG_ii_fresh_anchor.md). Queue item P4 of
`~/.cache/gen3ai/handoffs/post_b1_queue_proposal_2026-11-16.md` (§2 row Q2(b)). Abbreviations: draft (ii)'s.

## 1. When this draft applies

- **X70 reads `NOT-SEPARATED`** (X70 §10: "CE-3d's +3.97 pp is then read as an entropy-coefficient result, and the
  open question becomes the coefficient itself"); or **`ANCHOR-COSTS`** (X70 §10: "the entropy coefficient goes to
  the owner as above"; listing it here is this draft's reading of that clause, for the orchestrator to confirm); or
- **draft (i) reads `NOT-SEPARATED` or `DIRECTION-COSTS`, `MATCHED`** (then c\* is a named candidate beside 0, §3.3).

## 2. The question and the lever, in plain language

At the plateau, lowering the uniform entropy bonus helped (0.05 → 0.0125: +1.97 pp; the swap, which removed it,
+3.97 pp over C0), and X70 will have said whether removing it with no anchor at all still helps. B1 trains every
run at 0.0125 from scratch, a value chosen from the plateau evidence (B1 Amendment 2). **Is B1's 0.0125 the right
bonus from step 0, or does a run with less (none) end stronger?** Two forms, one chosen at registration (§3.1):

- **(iii-a), the main arm: no entropy bonus from step 0.** B1 C's argv with `--ent-coef 0.0125` → `--ent-coef 0`.
  ONE flag value differs from the comparator: the cleanest single-lever arm in this set.
- **(iii-b), the schedule: B1's bonus to 50M, none after.** Forks of B1 C's three runs at their 50M checkpoints with
  `--ent-coef 0`, to 100M. There is no entropy-schedule flag on main (no `--ent-coef` schedule or anneal in
  `src/main/train/parser/`; only the learning rate anneals), so a step schedule is the one form that needs no
  build: a typed `--ent-coef` on a fork is applied (`src/main/train/model_build.py`: `model.ent_coef =
  args.ent_coef` on the load path), as X67's and X70's forks rely on.

## 3. The arms

### 3.1 Which form, fixed now (a draft choice, for the orchestrator to confirm)

From X70's two REPORTED readings (X70 §5.4) and its labels, at registration:

| X70's `x67` reading (ent 0 vs 0.0125 at the plateau) | `ENT-COLLAPSED` on X70 | form |
|---|---|---|
| "ent_coef 0 beats 0.0125" or "not separated" (X70 §5.4's wording, abbreviated) | no | **(iii-a)** |
| "ent_coef 0 beats 0.0125" or "not separated" | yes | **(iii-b)**: a policy that collapses with no bonus from a trained start is a poor bet from a random one; the schedule keeps B1's bonus through the exploring phase |
| "ent_coef 0 LOSES to 0.0125" | either | **neither**: the dose curve turned over above 0 at the plateau; B1's 0.0125 stands and the dose curve goes to the owner (X70 §10). With draft (i)'s c\* in hand, a (iii-a) arm at c\* is the one exception the orchestrator may name |

### 3.2 (iii-a): `ent0`, fresh, at B1's launch pin

Seeds **5131, 5132, 5133**; pin `15136b61` (B1 C's own). A grep of `src/`, `designs/`, `scripts/`, `tools/` and
every `models/*/metadata.json` on 2026-11-17 found none of 5131–5133 or 5141–5143 as a seed or `_s<seed>` name. The
first seed ([`argv_iii_ent0_s5131.txt`](argv_iii_ent0_s5131.txt)):

```
--restart-interval-hours 6 --pin-commit 15136b61634b667bb9da5ccc6647ff9265238829 --arch production --ent-coef 0
--behaviour-check warn --checkpoint-every-steps 1000000 --steps 50000000 --seed 5131 --device cuda
--log-level periodic --run-name rb_b1ent0_s5131
```

`python -m main.checkargs --argv`, 2026-11-17, CPU, `CUDA_VISIBLE_DEVICES=""`, nice 19 on the agent cores: against
main `81594609b` ([`checkargs_iii_main.txt`](checkargs_iii_main.txt)) and the pin `15136b61`
([`checkargs_iii_pin15136b61.txt`](checkargs_iii_pin15136b61.txt), "validated against PINNED parser @15136b61"):
**exit 0 both**, 0 unrecognised, every ARCH-surface key the production mirror's, ONE recipe knob off
`recipe.fresh`, TYPED: `ent_coef 0.0`.

### 3.3 (iii-b): `csched`, forks of B1 C at 50M

Three forks, one of each B1 C run at its 50M checkpoint (`rb_b1c_s5101` → `checkpoint_50000412_steps.zip`, `s5102`
→ `checkpoint_50000041`, `s5110` → `checkpoint_50000220`; the files B1's 50M look reads), seeds **5141–5143**, to
100M. The first ([`argv_iiib_sched_s5141.txt`](argv_iiib_sched_s5141.txt)):

```
--restart-interval-hours 6 --pin-commit 15136b61634b667bb9da5ccc6647ff9265238829
--model <main>/models/rb_b1c_s5101/checkpoints/checkpoint_50000412_steps.zip --ent-coef 0 --behaviour-check warn
--checkpoint-every-steps 1000000 --steps 100000000 --seed 5141 --device cuda --log-level periodic
--run-name rb_b1csched_s5141
```

`checkargs`, the same conditions, against main ([`checkargs_iiib_main.txt`](checkargs_iiib_main.txt)) and the pin
([`checkargs_iiib_pin15136b61.txt`](checkargs_iiib_pin15136b61.txt)): **exit 0 both**, resolved against the fork
parent's record (75 flags inherited), `✓ fork arch`, the lever row `ent_coef 0.0 (TYPED)`, the parent's
`n_epochs` 10 restored. **Not run for either form:** the launcher's `--dry-run`, any kit.

### 3.4 The control

**(iii-a): B1 C's three runs** at 50M and 100M by B1 §5.1's file rule; the arm differs from them in `--ent-coef`'s
value and the seed alone. **(iii-b): the same three runs at 100M** (their phase-2 continuation, which keeps
0.0125): each fork shares its first 50M with one of its columns. If the orchestrator places the arm in another B1
cell's frame (W, Br, WB), the same substitution in that cell's argv, at that cell's cost.

## 4. Dose, GPU-h

| item | GPU-h | source |
|---|---:|---|
| (iii-a) three runs to 50M | **≈ 28.5–29** + ≈ 0.5–2 of starts | B1 C's realised span, 9.43–9.59 h a run ([`b1_entropy_read.log`](b1_entropy_read.log), TensorBoard wall times); starts B1 §12 |
| (iii-a) 50M → 100M | ≈ 30.2 | B1 §12's 10.07 a narrow run; **UNVERIFIED** (B1's phase 2 has not run) |
| (iii-b) three forks, 50M → 100M | **≈ 30.2** + starts | the same 10.07; **UNVERIFIED** |
| a look, 45,000 games | ≈ 0.33 | as draft (ii) §4; **UNVERIFIED** rate |
| **total** | **(iii-a) ≈ 29–31 to 50M, ≈ 60–62 to 100M; (iii-b) ≈ 31–32** | |

## 5. The statistic and the bar

**Both forms: draft (ii) §5 exactly** (B1 §5.2 with one arm a side, the N view, the 1.76 pp floor, V̂ = s²_on / 3 +
s²_off / 3 on 4 df, B1 §5.3's classes and the 50M harm stop through `main.b1.cross.classify`; the same FINDING on
the missing registered function). (iii-b) has no 50M look (it starts at 50M) and reads at 100M only.

**Why (iii-b) does not get a paired rule here.** Each fork shares 50M of history with one comparator run, so the
lineage effect cancels in Δ̂ but not in the unpaired variance: the rule above is VALID and CONSERVATIVE for it. A
paired rule (each fork against its own parent's continuation) would be sharper but needs new code and its own
power model, whose key input, how far two runs from one 50M checkpoint drift apart by 100M, is **UNMEASURED** (the
nearest figure, P\* + 20M forks' 0.3–0.5 pp seed SD, X70 §6.1, is another horizon and another start).

**REPORTED, never deciding:** end entropy by 5M window (B1 C at 0.0125: ≈ 1.13–1.24 nats in the first 5M,
0.56–0.61 in every 5M window from 15M, ending at 0.558–0.587 at 50M; [`b1_entropy_read.log`](b1_entropy_read.log)); `lr_median`; and
**`ENT-COLLAPSED`**: any arm run ending below half the lowest B1 C end entropy at the same step (at 50M: below
0.279 nats, half of `rb_b1c_s5110`'s 0.5575). A label, not a VOID (X70 §8: voiding a collapse would condition the
sample on the lever's own effect). Kakuna as in draft (ii) §5.

## 6. Power ([`power_fresh.log`](power_fresh.log); the model and seed SD sources are draft (ii) §6's)

The same 3 × 3 design: **median half-width 3.28 pp at a seed SD of 1.76; P(WIN) at +2 / +3 / +4 pp = 0.16 / 0.36 /
0.61**; at 2.5 pp, 0.15 / 0.28 / 0.43. At a true zero: NO-USEFUL-GAIN 0.60, UNDECIDED 0.38, LOSS 0.011. The
plateau's dose curve moved by ≈ +2 pp per step (0.05 → 0.0125: +1.97 [+1.35, +2.59]); if the fresh effect of
0.0125 → 0 is that size, **this read most likely ends UNDECIDED**, and the report says so with the table. For
(iii-b) the figures are an upper bound on the half-width (§5). **At registration:** re-run with B1's measured 50M
seed SD.

## 7. What each outcome licenses (registering only; nothing moves a recipe without the owner)

- **WIN** (with a confirming Kakuna read that is not NEGATIVE) ⇒ "B1's 0.0125 was not the best fresh coefficient":
  (iii-a) puts `ent_coef 0` from step 0, (iii-b) the step schedule, to the owner as a CANDIDATE for the fresh recipe,
  with X71's exploitability read if it exists (the queue's Q3 veto). It licenses registering the same arm at the
  plateau (the queue's Q1 continuation is where that would be read).
- **LOSS** ⇒ B1's 0.0125 stands against that form from scratch; the plateau gain from removing the bonus does not
  transfer to it. After an (iii-a) LOSS, (iii-b) may be registered (it keeps the bonus where exploration matters).
- **NO-USEFUL-GAIN / UNDECIDED** ⇒ recorded with the interval; 0.0125 stands by default, never quoted as "equal".
- **STOP-HARM at 50M** ((iii-a) only) ⇒ no phase 2; as LOSS.

## 8. What must be true at registration

1. The trigger (§1) by X70's committed `verdict.json` and its two reports (`report_c0.json`, `report_x67.json`),
   and §3.1's form read from them; the form is written in the PREREG.
2. P5's one-fresh-cell frame on main (statistic function with a test, the resolved-config diff, the continue rule);
   where it differs from draft (ii) §4–§5, P5's governs.
3. B1's 50M look read and its seed SD quoted; §6 re-run.
4. The comparator files exist and are unthinned (B1 C's 50M and 100M checkpoints; for (iii-b) the three 50M parents
   above, read by sha256). **Nothing protects them after B1's §9 reading closes** (FINDING, as in draft (ii)).
5. A kit builds the argv (`main.b1.arms` or its successor); `checkargs --pin 15136b61` and `--dry-run` re-run; the
   seeds re-checked; each run's torch is 2.8.0+cu126 on every resume (B1 Amendment 6 item 8). (iii-b) forks a B1
   run: the dry run must read `role : FORK of models/rb_b1c_s…`, never a RESTART of it (models/ rule).
6. The GPU slot is the orchestrator's; no B1 run or leg is paused for it.
