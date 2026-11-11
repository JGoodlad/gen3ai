# X67, the entropy screen: the READ (2026-11-11)

**Verdict: PASS.** Three forks of P\* trained on for 20M steps with a quarter of the entropy bonus (`ent_coef`
0.0125) beat the eight banked no-lever forks (C0, `ent_coef` 0.05) head to head, both sides greedy:
**D = +1.97 pp, 90 % interval [+1.35, +2.59]** (Welch df 6.2). Every fork carries the label **LR-MOVED**: the
learning-rate controller cut the rate one step further than C0's (8.372e-05 against 1.0047e-04). **ENT-UNMOVED does
not apply**: the lever moved the policy's entropy from 1.01 to 0.68 nats.

Registration: [`PREREG.md`](PREREG.md) (`1bab9e87a`, unamended). The verdict is
[`cross/read_cross.py`](cross/read_cross.py)'s [`record/verdict.json`](record/verdict.json), written by
[`screen_rule.py`](screen_rule.py) `decide`. Nothing in `PREREG.md`, `screen_rule.py` or `cross/` was edited.
Abbreviations: pp = percentage points; P\* = `pstar_plateau` (`rb_x26_s2601` @280,000,170); C0 = P\* + 20M with no
lever at the production side pin; D = mean cell score − 50, the X67 arm's lead over C0; df = degrees of freedom;
SE = standard error; K9(b) = the behaviour-policy check (does the learner's log-probability of each played action
match the one stored at play time).

## 1. The cross (PREREG §5)

Played 2026-11-11 04:51 to 12:32 from a clean detached checkout at the pin `f4b25238`
(`drive_cross.sh <pin> <work> cpu`, nice 19 on the agent cores 5-7,13-15, under `mem_cap.sh` 24 GB; peak 1.49 GB).
24 cells × 1,000 mirrored team pairs = **48,000 games**; every cell `OK`, 1,000 pairs, **0 aborted, 0 timeouts**,
210 draws (0.44 %). Family `x67_entropy_screen_h2h`, request `x67_entropy_screen`, protocol
`gen3_eval_protocol_v1_h2h`, eval ledger `models/_ledger` (family-register seq 74400).

**Columns: 8**, chosen by [`cross/plan_cross.py`](cross/plan_cross.py), not by hand: the five banked C0 finals
(s3501–s3505) and X60-T's three top-up finals (s3506–s3508); none left out ([`record/shas.json`](record/shas.json),
[`record/cells_plan.json`](record/cells_plan.json)).

h, the X67 fork's score against each C0 final, pp ([`record/h.json`](record/h.json)):

| X67 fork \ C0 | 3501 | 3502 | 3503 | 3504 | 3505 | 3506 | 3507 | 3508 | row mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s6701 | 52.33 | 52.33 | 51.10 | 53.73 | 51.18 | 53.35 | 53.15 | 51.55 | **52.34** |
| s6702 | 52.33 | 51.95 | 52.10 | 53.20 | 49.93 | 51.10 | 52.40 | 51.35 | **51.79** |
| s6703 | 52.02 | 52.62 | 52.12 | 50.80 | 51.52 | 52.68 | 51.30 | 51.12 | **51.78** |
| col mean | 52.23 | 52.30 | 51.78 | 52.58 | 50.88 | 52.38 | 52.28 | 51.34 | |

23 of 24 cells are above 50; the one below is s6702 against s3505, at 49.93.

## 2. The verdict (PREREG §5.3)

| quantity | value |
|---|---|
| **class** | **PASS** (the interval's lower end > 0; no end within 1e-9 pp of 0) |
| D | **+1.969 pp** |
| 90 % interval | **[+1.346, +2.592]**, SE 0.323 pp, Welch df 6.22, t(0.95) 1.931 |
| row variance (3 row means) | sample 0.102 pp², **floored** to 1.25 / 8 = 0.156 |
| column variance (8 column means) | sample 0.348 pp², **floored** to 1.25 / 3 = 0.417 |
| **pessimistic interval** (the fresh-run seed SD of 1.76 pp as the floor) | **[−0.646, +4.584]**, SE 1.19 pp, df 3.64: it spans 0 |
| day split | D over the banked columns (trained 10-21 to 10-24) **+1.95 pp**; over the top-up columns (trained 11-09/10, within a day of X67) **+2.00 pp** |

Both floors bound: the forks' seeds spread no wider than one cell's binomial noise. The screen read the
"fork evidence" regime of PREREG §6 (seed SD ≈ 0.5 pp, half-width ≈ 0.8–1.0 pp), not the fresh-run one. The
pessimistic interval is the reading under the assumption that the X67 seeds really spread like fresh runs, which the
cross itself does not show.

**The prediction** (PREREG §1, a GUESS): D in [−1, +2] pp, most likely within ± 1. The read sits at the top edge of
that range and outside the ± 1 the registration thought most likely.

## 3. VOID guards (PREREG §8): none fired; three valid forks

Checked 2026-11-11 by the read operator from each run's own record. The Training session's arm-end lines for s6701
and s6702 agree.

| guard | s6701 | s6702 | s6703 |
|---|---|---|---|
| launcher rc 0 | 0 (resume #1's rc file) | 0 (kit END line) | 0 (kit END line) |
| `Last step` ≥ 300,000,170 | 300,021,355 | 300,061,997 | 300,061,533 |
| `final_model.zip` | yes | yes | yes |
| newest checkpoint within 98,304 steps | 300,000,009 (Δ 21,346) | 300,000,209 (Δ 61,788) | 300,000,212 (Δ 61,321) |
| K9(b) at arm end (`main.x35.batch.k9b_read`) | PASS, max 4.31e-4 | PASS, max 2.31e-4 | PASS, max 2.09e-4 |
| launcher log: pin `f4b25238`, FORK of `rb_x26_s2601` | yes (resume: RESTART of its own run, same pin) | yes | yes |
| post-spawn pool-seed check | PASS | PASS | PASS |
| hyperparameter table prints `ent_coef 0.0125` (never 0.05, no `[ModelVersion] FATAL`) | yes, at launch and at the resume | yes | yes |
| `metadata.ent_coef` and `cli_args.ent_coef` | 0.0125 / 0.0125 | 0.0125 / 0.0125 | 0.0125 / 0.0125 |
| `main.dose` epochs / effective batch | 10 / 65,536 | 10 / 65,536 | 10 / 65,536 |
| `main.sidecar_audit` | 1 span, 0 PIN-SPLIT, 21 sidecars | same | same |

s6701 stopped once at 19:03 (≈ 295.5M) on the known T2 near-tie compile-parity failure (`launcher_rc=3`, "compiled
extractor DISAGREES with eager on legal_logprob", max 2.99e-3) and was auto-resumed once from checkpoint 295,000,204.
This was the orchestrator's pre-approved §13-style resume, not a VOID: its `pin_history` is one span at the pin from
280,000,170. Every fork's child log carries the same `--behaviour-check warn` K9(b) single-shot notices C0's do
(2 to 7 a run, C0 s3501: 7); the arm-end read is the guard. Records: [`record/k9b_x67.json`](record/k9b_x67.json),
[`record/dose_x67.txt`](record/dose_x67.txt), [`record/sidecar_audit_x67.txt`](record/sidecar_audit_x67.txt).

## 4. Labels (PREREG §8; they never change the class)

| fork | `lr_median` (`main.dose`) | LR-MOVED (outside 9.05e-05 to 1.105e-04) | dose rate |
|---|---|---|---|
| s6701 | 8.372e-05 | **yes** (−16.7 % against C0's 1.0047e-04) | 1.703e-08 |
| s6702 | 8.372e-05 | **yes** | 1.703e-08 |
| s6703 | 8.372e-05 | **yes** | 1.703e-08 |
| C0 (all eight) | 1.0047e-04 | | 2.044e-08 |

**ENT-UNMOVED: no.** X67's mean end entropy is 0.676 nats against the lowest C0 seed's 1.004 (§5).

## 5. Reported diagnostics (PREREG §9; they decide nothing)

`main.ops.run_ref.scalar_events` (the reader behind `main.ops.tb_read`) on `train/entropy_loss`,
`train/learning_rate` and `train/approx_kl`, the arm's own points after the fork step. Entropy in nats is
−`entropy_loss`, and the end value is the mean of the last 20 points. In the run that resumed, points the abandoned
attempt logged past the resume step are dropped (s6701: 5). Script and output:
[`record/diag9_entropy.py`](record/diag9_entropy.py), [`record/diag9_entropy.json`](record/diag9_entropy.json),
[`record/diag9_entropy.txt`](record/diag9_entropy.txt).

| run | end entropy (nats) | mean entropy by 5M window after the fork | learning rate | approx KL, median |
|---|---|---|---|---|
| X67 s6701 | 0.659 | 0.781 → 0.699 → 0.678 → 0.663 | 1.206e-4 → 1.005e-4 (281.0M) → 8.372e-5 (286.2M) | 0.0185 |
| X67 s6702 | 0.683 | 0.785 → 0.704 → 0.685 → 0.688 | → 1.005e-4 (281.3M) → 8.372e-5 (287.1M) | 0.0188 |
| X67 s6703 | 0.687 | 0.781 → 0.702 → 0.692 → 0.689 | → 1.005e-4 (281.0M) → 8.372e-5 (286.0M) | 0.0188 |
| **X67 mean** | **0.676** | | two cuts each | |
| C0 s3501–s3508 | 1.004–1.018, **mean 1.010** | flat at ≈ 1.00–1.02 | one cut each, to 1.005e-4 (280.99M–286.3M) | 0.0179–0.0183 |

- **The lever is live.** Entropy falls from P\*'s ≈ 1.0 nats to ≈ 0.68 within ≈ 10M steps and settles there; C0's
  stays flat at ≈ 1.01.
- **The rate.** At the smaller bonus the per-update policy change (approx KL) runs slightly higher (0.0185–0.0188
  against 0.0179–0.0183), so the KL controller cut the rate a second time in every fork, at 286–287M. Over the
  20M, X67 learned at 0.83× C0's dose rate.
- **Greedy minus sampled** and **Kakuna on U** were NOT run: both are conditional in §9 ("only if Kakuna lanes are
  free" / "only if CPU allows"), and X60-T's look-2 play lanes held the agent cores for the whole read.

## 6. What the PASS licenses (PREREG §10)

> **PASS ⇒ `ent_coef` 0.0125 is named for the base recipe of BOTH arms of BOUNDARY-1** in that PREREG's §10 fill;
> it is never carried by one arm. **A PASS with `LR-MOVED` goes to the orchestrator with both numbers before the
> recipe moves.**

The two numbers: **D +1.97 pp [+1.35, +2.59]**, and **`lr_median` 8.372e-05 against C0's 1.0047e-04** (dose rate
0.83× C0's). The rate cut is the KL controller responding to the smaller bonus, so it is part of the lever's own
effect, but the forks did not get the same optimizer dose as C0.

**This read moves no recipe.** `production_config.json`'s recipe block and `design_learner_recipe.md` §3.22 move
together, in BOUNDARY-1's registration. The owner's 2026-09-29 ruling (§3.9: "no direct adoption of a lower bonus")
stands until the owner lifts it there.

**The limits, as registered before the number:**
1. **A late fork is not a fresh run.** The bonus changed at step 280M of a policy shaped by 0.05; the licence puts
   0.0125 in from step 0, where early exploration is a different regime.
2. **Head to head against siblings is not strength against others.** A sharper greedy policy may beat its own
   lineage and gain nothing against Kakuna; §9's Kakuna read, the only outside evidence, was not run.
3. **One dose.** +20M at one coefficient; nothing about 0.025, 0.005 or a schedule.

## 7. FINDINGs

- **X67-R1 (the pessimistic interval spans 0).** Under the fresh-run seed-SD floor (1.76 pp), [−0.65, +4.58]. The
  PASS rests on the cross's own seed spread, which the registration chose deliberately (§5.3); both floors bound,
  so the interval is already the widest the binomial floor allows at this shape.
- **X67-R2 (the dose differs).** LR-MOVED on all three forks; see §6.
- **X67-R3 (no outside evidence).** §9's Kakuna and greedy-minus-sampled reads were not run. Whether the gain
  survives against Kakuna is unmeasured.
- **X67-R4 (the slim child log has no hyperparameter table).** Each run's `launcher_child.log` no longer holds the
  `ent_coef` table line; `launcher_child.full.log` does (s6701 twice, at launch and at the resume; the others once;
  never 0.05). The banner guard read the slim log live and logged "banner OK" for every launch. The VOID check above
  used the full log.
- **X67-R5 (the recipe doc is now stale on one line).** `designs/endstate/design_learner_recipe.md` §3.9 still says
  the screen is "REGISTERED, not run". The read operator's brief forbids editing that doc. It moves with BOUNDARY-1's
  registration (the orchestrator's step).
