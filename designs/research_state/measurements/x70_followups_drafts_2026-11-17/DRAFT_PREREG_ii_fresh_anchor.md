# X70 follow-up (ii), the from-scratch anchor arm in BOUNDARY-1's frame: DRAFT pre-registration (2026-11-17)

**Status: DRAFT. NOT REGISTERED. Nothing here is a rule, and nothing launches from it.** Written before X70 has
trained a step (blind by construction). It becomes a registration only if its trigger (§1) fires and the
orchestrator names it; it is then copied into a new folder as `PREREG.md` with every **UNVERIFIED** and "at
registration" item filled, and it lands on main before the first step. Tier when registered: **FULL** (a fresh-run
comparison, `../../README.md` § Registration tiers). Siblings:
[`DRAFT_PREREG_i_matched_entropy.md`](DRAFT_PREREG_i_matched_entropy.md),
[`DRAFT_PREREG_iii_fresh_ent0.md`](DRAFT_PREREG_iii_fresh_ent0.md). Queue item P4 of
`~/.cache/gen3ai/handoffs/post_b1_queue_proposal_2026-11-16.md` (§2 row Q2(b)).

Abbreviations: X70's ([`../x70_ent0_control_2026-11-13/PREREG.md`](../x70_ent0_control_2026-11-13/PREREG.md)) and
BOUNDARY-1's (B1, [`../boundary1_scratch_2026-11-09/PREREG.md`](../boundary1_scratch_2026-11-09/PREREG.md): the N
view, home72, H, F, P72, U, lo / hi, seed SD). B1 C = B1's control cell, `rb_b1c_s5101`, `s5102`, `s5110`.

## 1. When this draft applies

- **X70 reads `ANCHOR-ADDS` with overlapping end entropies or with `ENT-ABOVE-SWAP`** (X70 §10: "a longer-horizon or
  fresh-run magnet arm, which the owner weighs against B1's queue"); or
- **draft (i) reads `DIRECTION-ADDS`, `MATCHED`**; or
- **X70 reads `ANCHOR-ADDS` and `X70-IS-MATCHED`** (draft (i) §1: X70's own cross was already at a matched entropy).

`NOT-SEPARATED`, `ANCHOR-COSTS`, `INCONCLUSIVE`, or (i)'s `NOT-SEPARATED` / `DIRECTION-COSTS`: this draft does not
apply.

## 2. The question and the lever, in plain language

At the plateau, a pull toward the policy's own 2M-step-old copy beat having no regulariser, with no sign that it was
only the entropy level. **Does the same regulariser, used from step 0 in place of B1's small entropy bonus, make a
stronger agent from scratch?** The lever is the swap's spelling exactly: no uniform entropy bonus
(`--ent-coef 0`) and the trailing anchor at β 0.05, refreshed every 2,000,000 steps (`--magnet-kl-coef 0.05`; the
refresh is the flag's default, `designs/training/magnet_kl.md`).

**What the comparison also changes, stated before any number.** In loss terms β · KL(π ‖ π_mag) = −β · H(π) −
β · Σ π log π_mag (X70 §1), so the arm carries an entropy weight of **0.05 inside the KL** against B1 C's **0.0125**,
plus the pull. At the plateau that larger weight did not hold entropy up (the swap ended at 0.49 nats, below X67's
0.68 at 0.0125), but from scratch it is unmeasured. The arm therefore reads "the swap's regulariser against B1's
base", the form a recipe adoption would take, not "the pull alone". The weight-matched spelling (`--magnet-kl-coef
0.0125`, entropy weight equal to B1's, a pull a quarter as strong) is the alternative in §3; it is an UNTESTED
spelling at any horizon, which is why it is not the default.

## 3. The arm and its control

**Arm `mag`** (a draft name: `src/main/b1/arms.py` holds only C / W / Br / WB): **B1 C's argv with
`--ent-coef 0.0125` replaced by `--ent-coef 0 --magnet-kl-coef 0.05`**, seeds **5121, 5122, 5123**, at **B1's launch
pin `15136b61`** (the comparator's own pin: the fresh-cell frame compares runs trained on one code). A grep of
`src/`, `designs/`, `scripts/`, `tools/` and every `models/*/metadata.json` on 2026-11-17 found none of 5121–5126 as
a seed or `_s<seed>` name; the registration re-runs `python -m main.b1.arms check --tree`. The first seed
([`argv_ii_anchor_s5121.txt`](argv_ii_anchor_s5121.txt)):

```
--restart-interval-hours 6 --pin-commit 15136b61634b667bb9da5ccc6647ff9265238829 --arch production --ent-coef 0
--magnet-kl-coef 0.05 --behaviour-check warn --checkpoint-every-steps 1000000 --steps 50000000 --seed 5121
--device cuda --log-level periodic --run-name rb_b1mag_s5121
```

**`python -m main.checkargs --argv`, 2026-11-17, CPU, `CUDA_VISIBLE_DEVICES=""`, nice 19 on the agent cores:**
against main `81594609b` ([`checkargs_ii_main.txt`](checkargs_ii_main.txt)) and against the pin `15136b61`
([`checkargs_ii_pin15136b61.txt`](checkargs_ii_pin15136b61.txt), "validated against PINNED parser @15136b61"):
**exit 0 both**, 0 unrecognised, every ARCH-surface key the production mirror's, and exactly two recipe knobs off
`recipe.fresh`, both TYPED: `ent_coef 0.0` and `magnet_kl_coef 0.05`. The resolved recipe keeps
`overlap_collect` false and `fork_fraction` 0 (`designs/production_config.json`, the same at the pin), the two
settings the magnet refuses. The weight-matched alternative (seeds 5124–5126,
[`argv_iib_anchor_weightmatched_s5124.txt`](argv_iib_anchor_weightmatched_s5124.txt),
[`checkargs_iib_pin15136b61.txt`](checkargs_iib_pin15136b61.txt)): exit 0, `magnet_kl_coef 0.0125`. **Not run:** the
launcher's `--dry-run`, any kit.

**The control: B1 C's three banked runs**, each read through B1 §5.1's file rule (the first
`checkpoints/checkpoint_<N>_steps.zip` at or after the look's step and below it + 1M, never `final_model.zip`):
the arm differs from them in the lever's two tokens and the seed alone. **If the orchestrator puts the arm in another
B1 cell's frame** (the continuing cell is W, Br or WB), the arm is THAT cell's argv with the same substitution, and
§5's cost is B1 §12's for that cell (wide × 1.392).

## 4. Dose, looks, GPU-h

Fresh to 50M (phase 1, the comparator's own phase-1 step), the 50M look, then a resume to 100M under the same rules
as B1 §4 (a same-run resume, `--model models/<run>/final_model.zip --steps 100000000`, nothing else typed), the 100M
look. The continue-at-50M rule is the one P5's frame registers (queue §4 P5); this draft's default is B1's own
harm stop (§5).

| item | GPU-h | source |
|---|---:|---|
| three runs to 50M | **≈ 30–31** | B1 C's realised span, first to last logged update, 9.43–9.59 h a run ([`b1_entropy_read.log`](b1_entropy_read.log), from their TensorBoard wall times), × 1.049 for the magnet's pass (CE-3d §16.7: ≈ 3.5 s of a ≈ 71 s iteration). **UNVERIFIED:** that overhead on a fresh run |
| six starts (two a run) | ≈ 0.5–2 | B1 §12: 0.33 h cold, 251 s warm (B1 slot S8) |
| three runs, 50M → 100M | ≈ 31.7 | B1 §12's 10.07 a narrow run × 1.049; **UNVERIFIED** (B1's phase 2 has not run) |
| two looks, 45,000 games each | ≈ 0.33 each | 9 cells × (1,000 home72 + 3 × 500) pairs; B1 §12's ≈ 2.0 GPU-h a 270,000-game look, scaled; **UNVERIFIED** rate. CPU: ≈ 4.3 h a look on three agent cores at 2.9 games/s |
| **total** | **≈ 31–33 to 50M; ≈ 63–66 to 100M** | |

## 5. The statistic and the bar: BOUNDARY-1's, one arm a side

Cells: every arm run against every B1 C run, 9 cells, B1 §5.1's protocol (mirrored pairs, 1,000 on home72 and 500
on each of H, F, P72, both sides greedy, `--seed 0`, oracle reveal off, one engine, the pin's). **Deciding view N**
(½ home72 + ⅙ H + ⅙ F + ⅙ P72): the lever is not a team-breadth lever, so B1's width view applies.

B1 §5.2's statistic with one arm a side: Δ̂ = mean of the 9 cells − 50; s²_on, s²_off = each side's sample variance
of its three run means, **each floored at 1.76²**; **V̂ = s²_on / 3 + s²_off / 3 on 4 df** (t 2.132); two-sided 90 %.
**Classes: B1 §5.3's, unchanged** (`main.b1.cross.classify`): at 50M **STOP-HARM** iff hi < −2 pp, else CONTINUE; at
100M **WIN** lo > 0, **LOSS** hi < 0, **NO-USEFUL-GAIN** hi < +4 pp, else **UNDECIDED**; a bound within 0.005 pp of
its threshold fails (`at_boundary`). **The code:** `main.b1.cross.factor_stat` is the 2 × 2's and does not take one
arm a side; [`power_fresh.py`](power_fresh.py) builds the statistic from B1's own `_arm_var`, `_stat` and
`classify`. A registered function with a test is P5's build or this registration's (FINDING).

**REPORTED, never deciding:** each run's end entropy (B1 C's three end at 0.558–0.587 nats at 50M, within 0.56–0.61 in every 5M window from 15M, [`b1_entropy_read.log`](b1_entropy_read.log)); the
`train/magnet_kl` saw-tooth; each run's `lr_median` (`main.dose`; B1 Amendment 6 item 7: within C it already spans
1.736e-4 to 2.083e-4); `ENT-COLLAPSED` (any arm run ending below half the lowest B1 C end entropy, a label, not a
VOID, X70 §8's reasoning). **A confirming Kakuna read is not part of this draft**: a WIN goes to the owner only with
one, registered by a dated NOTE before any game (rule `b1_kakuna`, contrast arm − C on U); ≈ 45 lane-h for the arm's
three runs, ≈ 90 if C must be replayed (**UNVERIFIED** that `b1_kakuna` can import banked rows: no import is
listed in `designs/tools/reads.md`).

## 6. Power ([`power_fresh.py`](power_fresh.py) → [`power_fresh.log`](power_fresh.log), 20,000 draws a row)

The model: h = 50 + D + a_i − b_j + e, run effects at the fresh-run seed SD, cell noise 0.68 pp (B1 §11's N view).
The seed SD sources: **1.76 pp**, pooled over 14 groups of fresh 15M runs at `ent_coef` 0.05
([`../boundary1_scratch_2026-11-09/seed_spread.log`](../boundary1_scratch_2026-11-09/seed_spread.log)); 2.5 and 3.1
are B1 §11's pessimistic columns.

| arm runs × C runs, seed SD | median half-width | P(WIN) at +2 / +3 / +4 / +5 / +6 pp | at 0: NO-USEFUL-GAIN / UNDECIDED / LOSS | P(STOP-HARM) at 0 / −2 / −4 / −6 |
|---|---:|---|---|---|
| **3 × 3, 1.76** | **3.28** | 0.16 / 0.36 / **0.61** / 0.81 / 0.93 | 0.60 / 0.38 / 0.011 | 0.000 / 0.010 / 0.16 / 0.61 |
| 3 × 3, 2.5 | 4.21 | 0.15 / 0.28 / 0.43 / 0.60 / 0.75 | 0.40 / 0.54 / 0.026 | 0.003 / 0.027 / 0.15 / 0.43 |
| 3 × 3, 3.1 | 5.08 | 0.13 / 0.23 / 0.34 / 0.47 / 0.60 | 0.30 / 0.62 / 0.036 | 0.007 / 0.037 / 0.13 / 0.34 |
| 6 × 3, 1.76 (for scale: +3 runs, ≈ +31 GPU-h to 50M) | 2.50 | 0.31 / 0.60 / 0.84 / 0.95 / 0.99 | 0.81 / 0.15 / 0.021 | 0.000 / 0.021 / 0.30 / 0.84 |

**In words.** Three fresh runs against B1 C's three find a +4 pp regulariser effect about three times in five at
the floor, and a +2 pp one about one time in six; at a true zero the read says NO-USEFUL-GAIN 60 % of the time and
UNDECIDED 38 %. **The fork evidence is +1.97 pp (X67 over C0) and +3.97 pp (the swap over C0) at the plateau**, so
an effect of the size we have seen is likely UNDECIDED here. The comparator side is fixed at three runs, so more arm
runs buy less than they would in B1 (6 × 3 at +4: 0.84). **At registration:** B1's 50M look (≈ 2026-11-18) gives
the first measured seed SD at 50M in this recipe; the table is re-run with it (the floor stays 1.76 unless the
measurement is above it).

## 7. What each outcome licenses (registering only; nothing moves a recipe without the owner)

- **WIN** (with a confirming Kakuna read that is not NEGATIVE) ⇒ the swap's regulariser is a CANDIDATE for the fresh
  recipe, put to the owner with the interval, the entropy and rate labels, and X71's exploitability read if it
  exists (the queue's Q3 veto). It moves no `production_config.json` key and no B1 arm by itself.
- **LOSS** ⇒ the magnet row closes for fresh runs at this spelling.
- **NO-USEFUL-GAIN** ⇒ closes as "any from-scratch gain is below +4 pp at 100M", quoted with hi.
- **UNDECIDED** ⇒ recorded with its interval; nothing is retired or adopted; no fourth run (B1 Amendment 6 item 2's
  reasoning).
- **STOP-HARM at 50M** ⇒ no phase 2; the row closes for fresh runs at this spelling.
- `ENT-COLLAPSED` or a `lr_median` far from C's is quoted beside any class and changes none.

## 8. What must be true at registration

1. The trigger in §1 has fired, by X70's committed `verdict.json` (and draft (i)'s, if it ran).
2. P5's one-fresh-cell frame is on main: the resolved-config-diff check (arm vs C: the lever's keys alone), the
   statistic as a function with a test, and its continue-at-50M rule; where it differs from §4–§5, P5's governs.
3. B1's 50M look is read, its seed SD quoted; §6 re-run.
4. The comparator's files exist and are unthinned: each B1 C run's 50M and 100M checkpoints by §5.1's rule (B1 §4:
   none is thinned before B1's §9 reading closes; **after that, nothing protects them**: FINDING).
5. A kit arm (`main.b1.arms` or its successor) builds the argv; `checkargs --pin 15136b61` and the launcher's
   `--dry-run` re-run from it; the seeds re-checked disjoint.
6. **UNVERIFIED and owed before GO:** the magnet from step 0 at the B1 pin on the card (CE-3d's GPU evidence is
   forks at `9e97cad5`; the fit-check fix `13c50a5e6` is in `15136b61`, checked with `git merge-base`): a ≤ 10
   GPU-min start at a run boundary, by the lease holder. If the orchestrator places the arm in a WIDE cell, the
   startup fit at width 256 with the magnet's slot is a second such check.
7. Each run's torch is the comparator's (2.8.0+cu126, B1 Amendment 6 item 8) on every resume.
8. The GPU slot is the orchestrator's; no B1 run or leg is paused for it.
