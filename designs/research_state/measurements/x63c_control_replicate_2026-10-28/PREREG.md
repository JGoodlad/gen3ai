# X63-C s6302 — the X63 control replicate: PRE-REGISTRATION (2026-10-28)

**Status: REGISTERED before any launch, any game and any read. NOT LAUNCHED** (no `rb_x63c_*` run exists). The launch
is the orchestrator's GO through the Training session's arm chain (kit arm `x63c`). Nothing below changes after the
arm launches except by a dated amendment, written before the read, that says why.

**This is a CONTROL, not a lever.** X63-C trains no new idea. It is X26c run again with another seed, so that the
distance between the two runs measures one thing: how far two seeds of the SAME +100M continuation drift apart.

Abbreviations: pp = percentage points; X26@380M = `rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip`
(sha256 prefix `7e0ebe942eaa`); X26c = `rb_x26c_s2601`, X26@380M + 100M at the pin `f4b25238`, seed 2601, no lever;
F = `forum72` against Kakuna (X63's primary panel); σ = the true between-seed SD of a +100M continuation's F score.

## 1. Why now, and why first

The X63 review's BLOCKER B1 ([`../../../endstate/reviews/x63_review_2026-10-27.md`](../../../endstate/reviews/x63_review_2026-10-27.md),
`ca51bf99`) found that every error rate in X63's decision rule
([`../../../endstate/design_x63_long_breadth_continuation.md`](../../../endstate/design_x63_long_breadth_continuation.md),
`49fbf75a`) rests on σ, and no read has measured σ. The only measured seed spreads come from 20M forks (C0, D1).
Two seeds given five times as long on a plateau may drift further apart. Fix 3 of B1 is to run the Stage-B control
replicate FIRST, as the only direct +100M control-to-control spread. This file registers that run.

X63-C carries no lever, so its value does not depend on any of X63's gates (H-CTX, N1-P look 1, D3 look 2). Its σ
also serves any later +100M continuation read against X26c, whichever lever it tests.

## 2. The arm (kit arm `x63c`: `python -m main.x35.kit describe --arm x63c --seed 6302 --call reference_population_production.json`)

| field | value |
|---|---|
| run | `rb_x63c_s6302` |
| start | X26@380M (step 380,000,195), not P\*. The kit holds the start's sha256 and refuses a file that differs |
| pin | `f4b25238def57900c0305bb2a3a5a8f9edaef285` (X26c's) |
| argv | X26c's recorded `original_command`, token for token, with only `--seed 6302` and `--run-name rb_x63c_s6302` changed: `--model <X26@380M> --behaviour-check warn --steps 480000195 --checkpoint-every-steps 1000000 --device cuda --seed 6302 --run-name rb_x63c_s6302`, plus the launcher's `--restart-interval-hours 24 --pin-commit <pin>` (X26c's own) |
| dose | +100,000,000 steps |
| GPU | ≈ 19.9 GPU-h (X26c's measured 100M) |
| thin | the planner's (its 10M nodes and the final are kept, so the 400 / 430 / 480M read points survive) |

Only the seed differs from X26c, and so does the seeded snapshot pool's RNG. **The seeded pool set itself must equal
X26c's** (`pool_seed.json`, the same parent checkpoint at the same pin), and the kit checks that after the spawn.

## 3. What it measures

**The registered quantity.** d = F(X63-C) − F(X26c) at +100M (480M), each a plain proportion over its model's F games,
on the read's ONE engine and pack regime (X63 design §6.1; X26c's rows are re-played there). Then
**σ̂_CC² = max(0, d² − (p̂₁q̂₁/G₁ + p̂₂q̂₂/G₂)) / 2**, the control-to-control seed variance net of its binomial part.
The games are X63 Stage B's for the control side: 4,000 F games and 1,200 home72 games at +100M, at the cell and engine
X63's read registers.

**Its weakness, stated before the number.** One pair gives σ̂ one degree of freedom. Its 95 % interval runs from
≈ 0.45 σ̂ to ≈ 32 σ̂ (χ²₁), so a single σ̂ bounds σ from BELOW better than from above. A Stage-C pair (`rb_x63c_s6303`,
not registered here) would give 2 df.

**How it is used.** σ̂_CC enters X63's rule only through the amendment that answers B1 fix 2 (a σ floor or a
t quantile). This file decides nothing on its own, and X63-C has no GAIN / NULL class.

**REPORTED, never deciding:**
- d at +20 / +50M (400M, 430M; 1,200 F games per model), the trajectory of the spread;
- S on home72 for both finals (Welch, as D1), the in-house spread at +100M;
- the learning-rate path beside X26c's (8.37e-5 to 471M, 6.98e-5 from 472M; the review's M1). Two seeds of one
  argv may step the KL controller at different times; that difference is part of what σ measures, not a defect.

**Prediction** (GUESS, the review's prior; it decides nothing): σ ∈ [0.5, 3] pp, most likely 1–1.5 pp.

## 4. VOID guards (any one ⇒ the arm is VOID; no σ is read from it)

- not finished: the arm chain's completeness check (rc 0, `Last step` ≥ 480,000,195, `final_model.zip`, the newest
  checkpoint within 98,304 steps of it);
- K9(b) not PASS at the arm's end (AM-6);
- the launcher log names another pin, or the dry run is not `FORK of …/rb_x26_s2601`;
- the post-spawn pool-seed check fails: a `fork_step` other than 380,000,195, or a seeded set other than X26c's.

**Throughput < 1,000 steps/s in the first hour ⇒ PAGE**: HOLD and keep training; the orchestrator decides. A slow
control is still a valid control. **Canary family: C0** (the production architecture and argv; `canary_decide`).

## 5. What this file does NOT do

It does not register X63-U, X63's statistic or its classes, any game schedule beyond the control side's Stage-B
games, or a Stage-C pair. X63's design still carries B1 and five MAJORs unanswered, and X63-U launches only under its
own registration and gates.
