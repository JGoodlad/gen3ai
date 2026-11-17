# X70 follow-up (i), the matched-realised-entropy arm: DRAFT pre-registration (2026-11-17)

**Status: DRAFT. NOT REGISTERED. Nothing here is a rule, and nothing launches from it.** It is written before X70
has trained a step, so it is blind by construction. It becomes a registration only if X70 reads class (i) below; it
is then copied into a new folder as `PREREG.md`, every **UNVERIFIED** and every "at registration" item is filled,
and it lands on main before the first step. Tier when registered: **FULL** (`../../README.md` § Registration
tiers): it costs over 15 GPU-h, and its classes decide whether the magnet goes on to a fresh-run arm. Its two
siblings are [`DRAFT_PREREG_ii_fresh_anchor.md`](DRAFT_PREREG_ii_fresh_anchor.md) and
[`DRAFT_PREREG_iii_fresh_ent0.md`](DRAFT_PREREG_iii_fresh_ent0.md). Written for the orchestrator's queue item P4
(`~/.cache/gen3ai/handoffs/post_b1_queue_proposal_2026-11-16.md`, §2 row Q2(b), §4 P4).

Abbreviations are X70's ([`../x70_ent0_control_2026-11-13/PREREG.md`](../x70_ent0_control_2026-11-13/PREREG.md),
its header): pp, P\*, C0, X67, the swap (CE-3d's arm), π_mag, β, K9(b). End entropy = the mean of a run's last 20
points of −`train/entropy_loss` (X67, CE-3d and X70 §8 share this definition). T = the swap's mean end entropy.

## 1. When this draft applies (X70's classes, read from its §5.3, §8 and §10)

| X70's read | which draft it licenses |
|---|---|
| `ANCHOR-ADDS` **and** `ENT-BELOW-SWAP` (every X70 fork ended below 0.4838 nats, the lowest swap fork) | **this draft** (X70 §10: "first the matched-realised-entropy arm … because the result is not yet 'the anchor'") |
| `ANCHOR-ADDS` with overlapping entropies, or with `ENT-ABOVE-SWAP` | draft (ii) |
| `NOT-SEPARATED`, or `ANCHOR-COSTS` | draft (iii) (X70 §10: "the open question becomes the coefficient itself"; for `ANCHOR-COSTS`, "the entropy coefficient goes to the owner as above") |
| `INCONCLUSIVE` | none: never interpreted |

**One exit inside class (i), `X70-IS-MATCHED`.** If X70's mean end entropy is within 0.030 nats of T (§4.3's band),
X70's own deciding cross already compared the anchor with no regulariser at a matched entropy: no matched arm is
trained, and X70's `ANCHOR-ADDS` is read as draft (ii)'s licence. `coef_rule.py pick` refuses with that name.

## 2. The question and the lever, in plain language

X70's swap-against-ent-0 cross says the trailing anchor beat having no regulariser, but the ent-0 forks also ended
sharper than the swap's. So the win may be "the anchor's direction" (stay near your recent self) or just "the swap
stopped at a better entropy level". **The lever: a small, plain entropy bonus, sized so that P\* + 20M forks END at
the swap's entropy (T ≈ 0.4907 nats) with no anchor.** The deciding cross is then the swap against these forks: two
arms at one realised entropy, one with the anchor's direction and one without.

## 3. The arm

`x70me` (a name for the draft; the kit holds no such arm): **X70's argv with `--ent-coef 0` replaced by `--ent-coef
<c*>`**, c\* from §4. Everything else as X70 §2: a fork of P\* by sha256, `--behaviour-check warn`, +20M
(`--steps 300000170`), `n_epochs` 10, effective batch 65,536, the adaptive learning-rate controller, no magnet,
**training pin `9e97cad5`** (the swap's and X70's: the deciding cross keeps X70 §4.1's one-pin match), play pin
`333a73dd`. Seeds **7341–7344** (a fifth, 7345, only after a `REPICK`, §4.3). A grep of `src/`, `designs/`,
`scripts/`, `tools/` and every `models/*/metadata.json` on 2026-11-17 found none of 7341–7345 as a
seed or `_s<seed>` name; the registration re-runs the kit's own check.

**The argv**, with a PLACEHOLDER coefficient (0.0025 is NOT c\*; it is a value in §4.2's range, typed to prove the
form) — [`argv_i_matched_s7341.txt`](argv_i_matched_s7341.txt):

```
--model <main>/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip --behaviour-check warn --steps 300000170
--checkpoint-every-steps 1000000 --device cuda --ent-coef 0.0025 --seed 7341 --run-name rb_x35_x70me_s7341
```

**`python -m main.checkargs --argv`, 2026-11-17, CPU, `CUDA_VISIBLE_DEVICES=""`, nice 19 on the agent cores:**
- against main `81594609b` ([`checkargs_i_main.txt`](checkargs_i_main.txt)): **exit 0**, 0 unrecognised, resolved
  against the fork parent's record, `✓ fork arch`, the lever row `ent_coef 0.0025 (TYPED) production: 0.05`;
- against the training pin `9e97cad5` ([`checkargs_i_pin9e97cad5.txt`](checkargs_i_pin9e97cad5.txt)): **exit 0**,
  "validated against PINNED parser @9e97cad5", the same lever row, `✓ fork arch`, no magnet row.
- **Not run:** the launcher's `--dry-run`, the kit (`x70me` does not exist), the arm chain's plan.

## 4. The coefficient rule (concrete, from banked trajectories; no pilot run before the arm)

### 4.1 The inputs, all banked except X70's four

| point | ent_coef | end entropy, nats | source |
|---|---|---|---|
| C0 (8 finals) | 0.05 | mean 1.0095 (SD 0.0043) | `../x67_entropy_screen_2026-11-09/record/diag9_entropy.json` |
| X67 (3) | 0.0125 | mean 0.6762 (SD 0.0148) | the same file |
| the swap (3) = the TARGET T | (0, β 0.05) | mean **0.4907** (SD 0.0063; 0.4838–0.4962) | `../ce3d_magnet_screen_2026-11-11/record/diag9_ce3d.json` |
| X70 (4) | 0 | H0 = their mean | X70's §9 record, when it exists; never retyped |

All are P\* + 20M forks, so they share the horizon the arm will train to ([`coef_rule_selfcheck.log`](coef_rule_selfcheck.log)
prints the first three from the files).

### 4.2 The rule ([`coef_rule.py`](coef_rule.py) `pick`)

Fit the one monotone curve **H(c) = H0 + a · c^p** through the three points (0, H0), (0.0125, H_X67),
(0.05, H_C0): p = ln((H_X67 − H0) / (H_C0 − H0)) / ln(1/4). Invert it at T:

```
c* = 0.0125 · ((T − H0) / (H_X67 − H0))^(1/p),   rounded to 1e-5
```

It refuses (exit 3) when X70 is not `ENT-BELOW-SWAP` and when T − H0 ≤ 0.030 (`X70-IS-MATCHED`, §1). What it gives
across the H0 that class (i) allows ([`coef_rule_selfcheck.log`](coef_rule_selfcheck.log)):

| X70's mean end entropy H0 | p | **c\*** | c if H were linear in c | c if H were linear in √c |
|---:|---:|---:|---:|---:|
| 0.10 | 0.33 | 0.00384 | 0.00848 | 0.00575 |
| 0.20 | 0.38 | 0.00344 | 0.00763 | 0.00466 |
| 0.30 | 0.46 | 0.00283 | 0.00634 | 0.00321 |
| 0.40 | 0.57 | 0.00178 | 0.00411 | 0.00135 |
| 0.45 | 0.65 | 0.00091 | 0.00225 | 0.00040 |

**Why this form.** Two banked points cannot fix a curve; X70 supplies the third, at c = 0, so the curve is pinned at
both ends of the range c\* lives in (0 to 0.0125). The power law passes through all three, is monotone, and holds the
two simpler forms as special cases (p = 1 linear, p = ½ square-root). The two banked points alone predict
X70's H0: **0.565 nats if H is linear in c, 0.343 if linear in √c** (`coef_rule_selfcheck.log`); class (i) requires
H0 < 0.4838, which the linear form already fails, so the data at registration will say which way the curve bends.

### 4.3 The first fork is the check: KEEP or REPICK once ([`coef_rule.py`](coef_rule.py) `check`)

The arm's forks run one at a time (X70 §3). After fork 1 (seed 7341) ends and passes its guards, its end entropy
h1 is read:
- **KEEP** if |h1 − T| ≤ **0.030 nats** (twice the largest within-arm SD of a banked end entropy, X67's 0.0148,
  rounded up). Fork 1 is the arm's first fork; seeds 7342–7344 follow at c\*.
- **REPICK** otherwise: c\*\* = linear interpolation between the two measured points that bracket T among
  (0, H0), (c\*, h1), (0.0125, H_X67). Fork 1 is RETIRED from the cross (REPORTED only, its seed spent); seeds
  7342–7345 train at c\*\*. **Once only.** A non-monotone set of points, or T outside them, is a HOLD for the
  orchestrator ([`coef_rule_demo.log`](coef_rule_demo.log) shows each verdict on HYPOTHETICAL values).

**Why a first-fork check and not a 2–4 GPU-h pilot.** A short pilot cannot read the quantity: the swap's entropy was
still falling in its last 5M window (0.536 → 0.495) and X70's will likely be too, so an entropy at 5–10M does not
predict the one at 20M. A full fork is ≈ 4.1 GPU-h, and on KEEP it is not a cost at all: it is the arm's first
fork. The REPICK costs one fork.

### 4.4 What the rule is uncertain about (stated before X70 reads)

1. **The form.** Three points, three parameters: the fit has no residual and nothing checks it. Between the forms
   above c\* differs by up to ≈ 2.3×. The first-fork check bounds the damage; it does not remove it.
2. **The horizon is not a steady state.** X67's entropy settled within ≈ 10M (its last two 5M windows ≈ flat); the
   swap's was still falling. "Ends at T at +20M" does not mean "spent the 20M at the swap's entropies": the matched
   arm's trajectory shape is unknown, and the cross compares arms matched at their END only.
3. **The points mix pins.** C0 and X67 trained at `f4b25238`; X70, the swap and the arm at `9e97cad5`. CE-3d §4.3
   has only CPU evidence that a β = 0 learner is unchanged between them.
4. **The learning-rate controller answers the coefficient.** X67's forks were all `LR-MOVED` (8.372e-5 against
   1.005e-4); a small bonus may move the rate as X67's did. Reported as X70's `LR-MOVED` label, never corrected for.
5. **End entropy depends on history, not only on the coefficient.** BOUNDARY-1's fresh runs at 0.0125 sit at
   ≈ 0.56–0.61 nats in every 5M window from 15M to 50M and end at 0.558–0.587 (B1 C's three runs,
   [`b1_entropy_read.log`](b1_entropy_read.log)), while P\* + 20M at 0.0125 ended at 0.68. The curve here is a P\*-fork curve and is used only for
   P\* forks.
6. **UNVERIFIED:** how the hyperparameter table prints a five-decimal coefficient, which the chain banner must
   match (X70 §7); the registration takes the banner from a dry run at c\*.

## 5. The control, the cross, the statistic and the bar (X70's, reused)

**Deciding cross `direction`: rows = the three banked swap forks (`rb_x35_ce3dswap_s7321`–`s7323`), columns = the
four matched forks, 12 cells, 4,000 mirrored pairs a cell (96,000 games), one look,** X70 §5.1's protocol unchanged
(greedy both sides, schedule seed 0, `--oracle-reveal-mode off`, purpose `ab`, the play pin, one venue a cross,
X70 §14.2). **The statistic is X70's `screen_rule.py` `decide`, copied unchanged** (CE-3d's estimator: D = mean(h) −
50 = the swap's lead; floored row and column variances; Welch df; 90 % interval; rule 8 at 1e-9 pp). Its classes,
renamed for this question:

| class | condition | reading |
|---|---|---|
| `DIRECTION-ADDS` | lower end > 0 | at a matched end entropy, the trailing anchor still beats a plain bonus: the anchor's direction adds |
| `DIRECTION-COSTS` | upper end < 0 | at a matched end entropy, the plain bonus beats the anchor |
| `NOT-SEPARATED` | neither | quoted with its upper end: "the anchor's direction adds less than `<hi>` pp at a matched entropy" |
| `INCONCLUSIVE` | an input missing or invalid | never interpreted |

**REPORTED, decides nothing:** the matched forks against X70's four (4 × 4 at 1,000 pairs, 32,000 games): what a
small bonus does beside none. **Labels:** `MATCHED` iff the arm's mean end entropy is within 0.030 nats of T, else
`UNMATCHED-HIGH` / `-LOW`; X70's `ENT-BELOW-SWAP` / `ENT-ABOVE-SWAP` / `ENT-COLLAPSED` / `LR-MOVED`, by X70's
`screen_rule.py labels`. **VOID guards:** X70 §8's, with the arm's own `ent_coef` (c\* or c\*\*) as the lever value.

## 6. Dose, GPU-h, CPU

| item | GPU-h | note |
|---|---:|---|
| four forks, +20M each | **≈ 16.4** | 4.10 a fork, X70 §11's figure (the mean of the 20 X35 production forks' records) |
| after a `REPICK`: one more fork | + 4.1 | seed 7345 |
| the deciding cross (96,000 games) | ≈ 0.67 | **UNVERIFIED**: X70 §11's carried estimate; X70's own on-card crosses will have measured it by registration |
| the reported cross (32,000 games) | ≈ 0.22 | the same rate |
| **total** | **≈ 17.3 (KEEP) to ≈ 21.4 (REPICK)** | or the crosses on the CPU: ≈ 128,000 games at 2.9 games/s ≈ 12.3 h on three agent cores (CE-3d's measured rate) |

## 7. Power (X70's design, X70's simulation)

The deciding cross IS X70's design (3 swap rows × 4 columns at 4,000 pairs, the same rule), so X70's
[`power.log`](../x70_ent0_control_2026-11-13/power.log) applies unchanged: **P(`DIRECTION-ADDS`) at D = +1 pp is
0.83 / 0.70 / 0.58 at a fork seed SD of 0.3 / 0.4 / 0.5 pp; at +1.5 pp, 0.99 / 0.94 / 0.87; median half-width
0.63–0.87 pp; false positive ≤ 0.046 a side.** The seed SDs are banked: the swap's rows varied by less than their
binomial floor in both CE-3d crosses, C0's eight columns by ≈ 0.39 pp above it (X70 §6.1).

**What size matters.** Under class (i), X70's own D (swap − ent-0) has a lower end above 0, so it is at least ≈ 0.6
to 0.9 pp. If the whole of it is the entropy level, D here is ≈ 0; if the whole of it is direction, D here is about
X70's D. **At registration:** re-run X70's `screen_rule.py simulate` with the X70 forks' MEASURED column spread
(the first measured spread of a near-regulariser-free arm) and quote P(`DIRECTION-ADDS`) at X70's own point
estimate.

## 8. What each outcome licenses (registering only; nothing moves a recipe without the owner)

- **`DIRECTION-ADDS`, `MATCHED`** ⇒ registering draft (ii), the fresh-run anchor arm in BOUNDARY-1's frame.
- **`NOT-SEPARATED`, `MATCHED`** ⇒ the magnet row closes at this spelling as "its gain over no regulariser is not
  shown to exceed its entropy level"; the open question is the coefficient: draft (iii), with c\* named as a
  candidate value beside 0.
- **`DIRECTION-COSTS`, `MATCHED`** ⇒ the same closure, stronger: a plain bonus reaches the anchor's entropy and
  beats it, and costs no magnet pass (≈ 4.9 % of an iteration, CE-3d §16.7). Draft (iii) with c\* as the candidate.
- **Any class with `UNMATCHED-*`** ⇒ the class is REPORTED and licenses nothing: the arm missed its one job. A
  second attempt is a new registration.
- **`INCONCLUSIVE`** ⇒ never interpreted.

Limits carried from X70 §10 unchanged: three swap forks (2 df); a late fork and a short horizon; one spelling of
each side; head to head against siblings is not strength against others; exploitability unmeasured (X71's).

## 9. What must be true at registration

1. X70's verdict is `ANCHOR-ADDS` with `ENT-BELOW-SWAP`, and `coef_rule.py pick` on X70's four recorded end
   entropies returns a c\* (no refusal). The pick's output is committed beside the PREREG.
2. The training pin `9e97cad5` (tag `ce3d-pin-9e97cad5`) and the play pin `333a73dd` still resolve; the kit gains
   arm `x70me` (lever `--ent-coef <c*>`, its four seeds plus 7345, `batch.ARM_PINS` at `CE3D_PIN`), with a test like
   `src/main/x35/x70_test.py`; `checkargs --pin 9e97cad5` and the launcher's `--dry-run` re-run at c\*.
3. The chain items name the off-main pin (`off_main_pins`, X70 §14.1 G9); the banner is the table's own row at c\*.
4. X70's crosses' measured card rate replaces §6's estimate; X70's measured column spread replaces §7's.
5. The swap forks' three `final_model.zip` and `final_model.magnet.pt` are unchanged (sha256 as in CE-3d's record).
6. The GPU slot is the orchestrator's; nothing here pauses a BOUNDARY-1 run.
