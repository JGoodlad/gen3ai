# REVIEW: X48, per-token-kind FFN experts at the production plateau (adversarial, pre-launch, 2026-10-27)

**Status: REVIEW of the registration `fc888c18f` and the chain file `~/.claude/jobs/x35/chains/x35-x48-oct27.json`,
written before any X48 launch. Verdict: LAUNCHABLE, with no BLOCKER, 3 MAJORs and 8 MINORs.**

Two MAJORs (M1, M2) should land before the GO at ≈ 20:00. Each is a one-line chain edit or a dated PREREG NOTE, and
the PREREG allows NOTEs "before the first GO". The third (M3) can go in the same NOTE. This review does not edit the
PREREG or the chain file.

**What was read:** [`PREREG.md`](PREREG.md), [`power.py`](power.py) / [`power.log`](power.log), the transcripts
([`argv_x48_s3501.txt`](argv_x48_s3501.txt), [`checkargs_x48_s3501.txt`](checkargs_x48_s3501.txt),
[`dryrun_x48_s3501.txt`](dryrun_x48_s3501.txt), [`arm_chain_plan_x48.txt`](arm_chain_plan_x48.txt)), the kit arm
(`src/main/x35/kit.py` `x48`), the chain file and its draft (they differ only in `go_by`), the N4 review `7660d717`
([`../n4_capacity_screen_2026-10-27/REVIEW.md`](../n4_capacity_screen_2026-10-27/REVIEW.md)), and `designs/model/versioning.md`.
The code was read at the PIN `f4b25238`, because that is the code that runs: `agents/model/growth.py`,
`trunk_kffn.py`, `main/train/model_build.py`, `grow_fork.py` and `train_rl_agent.py`. Also read: `main.ops.canary_decide`
(the family ruling), `main/ops/arm_chain/{chain,postspawn}.py`, `designs/endstate/design_growth_builds_kffn_int.md`
(F-GB-5, F-GB-11), and the rule `n4_screen` as built on branch `n4-screen-rule` (`f3d5a3e8e`, not yet on main).

**What was run.** All of it was CPU only, with `CUDA_VISIBLE_DEVICES=""`, nice 19 on cores 5-7,13-15 and `ulimit -v`
12 GB. Nothing was launched, and `models/` was only read. Outputs: [`review_probe.log`](review_probe.log).
1. [`review_probe_growth.py`](review_probe_growth.py) ran from a detached worktree AT THE PIN. It grows **P\* itself**
   by the trainer's own sequence: `load_model_snapshot(grow=…)`, then `_validate_or_reset_optimizer_state`, then
   `apply_growth`. It then compares parent and child under the fp32 and the bf16 trunk, at 64 rows and at batch 1.
2. [`review_probe_grad_share.py`](review_probe_grad_share.py) measures each token kind's share of the shared FFN's
   gradient energy on P\*'s 64 gate rows. It uses a PROXY policy loss and value loss, not PPO's.
3. [`review_probe_pin_save.py`](review_probe_pin_save.py) saves P\* grown at the pin.
   [`review_probe_main_load.py`](review_probe_main_load.py) then loads it with MAIN's `load_model_snapshot` (the read
   plays on a main commit).
4. Re-ran `power.py`'s functions at my own σ. Read C0's trailing approx KL with `main.ops.tb_read`, and C0's LR from
   `snapshot_history`. Took sha256 of C0's finals and of P\*. Checked `git log` drift of `data/`,
   `deps/pokemon-showdown` and the env since C0 trained.

Abbreviations are the PREREG's: P\*, C0, S, K, pp, LB / UB, GPU-h, lane-h. Other terms:
- **f_k**: kind k's share of the shared FFN's gradient energy.
- **v**: AdamW's second moment (`exp_avg_sq`).
- **β2** = 0.999.

---

## 0. One screen

| rank | id | the finding | the fix (before the GO unless said) |
|---|---|---|---|
| BLOCKER | — | none. The growth is P\*'s function on production's own checkpoint at the pin (§1), the control frame is clean (no `data/`, Showdown or env drift; C0's sha256s match), the decision rule's error rates hold at my σ (§4), and a missing growth banner STOPs the run (§5) | — |
| MAJOR | M1 | **The COPIED second moments throttle the new experts for most of the 20M.** The run takes ≈ 4,080 AdamW steps. Each expert inherits the shared FFN's v, which was built from ALL 64 tokens' gradient, but now sees only its own kind's (f_k 0.01–0.27). Averaged over the run, board / threat / event experts move at ≈ 0.35–0.65 of their own-calibrated rate (≈ 0.2–0.5 over the first 10M). The last round's board / threat / event experts get no policy or value gradient at all. VOID 6's "each move at about η" is the opposite of this. A NO-GO would partly read "the lever was throttled" | a dated NOTE: an **"experts not engaged"** label on a NO-GO from §8.7's D_k / null; correct VOID 6's sentence; put "copied AdamW state (Q-K3)" in the NO-GO text |
| MAJOR | M2 | **VOID 4 cannot catch a growth that did not copy the moments.** "`optimizer_state` has no `copied` count" passes `copied: 0`, and the chain's banner `expect` passes any growth line. The values are now MEASURED on P\* at the pin: carried 280, fresh 0, copied 48, 48 tensors, 791,040 parameters, max \|Δ\| 0.0 | chain: `banner.expect` → `"fresh for 0, COPIED for 48"` (all three items). PREREG NOTE: VOID 4 = those exact counts |
| MAJOR | M3 | **The LR confound is checked in one direction only.** M4 labels a NO-GO when X48's LR is LOW. But M1's throttle predicts LOWER FFN KL, so the KL controller may keep X48 one rung HIGHER (×1.2). That is LR-F's own lever, and LR-F is in flight. A GO could then be partly LR | a symmetric label: GO with X48's mean LR > 1.1× C0's reads **"LR-elevated"** (a weak GO) |
| MINOR | m1–m8 | "bit for bit" is a CPU claim; checkargs' "fresh" note; the canary ruling to verify at update 10; the first CUDA compile of a kffn learner; a throughput page holds the chain; the read's cost and device; K's anti-conservatism; MIXED's reach | §6 |

---

## 1. Is the growth function-identical at the fork on production?

**On CPU, yes, and now measured on P\* itself** (F-GB-11 had measured it only on a one-AdamW-step parent). This is
the pin's code on `checkpoint_280000170_steps.zip`, by the trainer's sequence:

| check | result |
|---|---|
| fork-time gate (fp32 eager CPU, 64 rows, one batch) | **bit-identical**, max \|Δ logits\| 0.0, max \|Δ value\| 0.0 |
| the same 64 rows under the **bf16 trunk** (CPU autocast) | **0.0** (logits and values) |
| batch 1, 8 rows: fp32 / bf16 | **1.1e-5** / 0.0 (F-GB-5's small-M kernel reassociation, reproduced) |
| bf16 trunk vs fp32 on the parent (the bf16 noise floor) | 0.25 max \|Δ logit\| |
| optimizer | 280 parent tensors carried, 0 fresh, **48 copied**; each copy's `exp_avg_sq` equals its source's; `step` 56,880 |
| new parameters | 48 tensors, 791,040 (3 rounds × 4 experts × 65,920) |
| MAIN loads the grown checkpoint the pin wrote | yes, its forward **equal** to the pin's (the read plays on main) |

**On the CUDA learner (bf16 trunk, compiled R1), it is identical only up to kernel rounding, and that cannot be
checked without a lease.** The split runs six GEMMs on row slices plus a `cat`, where the shared FFN ran one GEMM.
cuBLAS / Inductor may pick other tilings for the slices, so the child's logits can differ from P\*'s at the
last-ulp level of a bf16 GEMM. That difference is orders of magnitude below the bf16 trunk's own 0.25 shift, which
both arms already carry. It is not a confound. The PREREG's "P\*'s function bit for bit" should say "on the CPU gate".
(This is m1.)

---

## 2. M1: the copied moments throttle the lever (MAJOR)

**The mechanism.** Before the fork, one shared FFN parameter's v is an EMA of the square of the WHOLE gradient, which
sums over all 64 tokens. Q-K3 gives every expert a clone of that v. After the fork, expert k's gradient is only its
own kind's part. So its Adam step m/√v starts too small by about √(v₀ / E[g_k²]) ≈ 1/√f_k. That factor decays as v
re-forms, with β2 = 0.999 per AdamW step. Q-K3 avoided a step-1 jolt, as intended, but this was its price. The PREREG
does not state that price, and VOID 6 states its opposite ("each move at about η").

**The step count.** P\*'s AdamW `step` is 56,880 at 280M. Its `_n_updates` is 28,440 (one per epoch), so the run
takes 2 AdamW steps per epoch. C0's `_n_updates` rose by 2,040 over its 20M. **X48 therefore takes ≈ 4,080 AdamW
steps.** By the final, β2^4080 ≈ 0.017, so the inherited v is mostly gone. At +10M it is still 13 %.

**The shares** (proxy loss, 64 gate rows; an ESTIMATE of PPO's f_k, but the direction is robust):

| round | loss | mon | board | move | threat | event |
|---|---|---|---|---|---|---|
| 0 | policy / value | 0.70 / 0.57 | 0.035 / 0.088 | 0.17 / 0.055 | 0.029 / 0.23 | 0.015 / 0.032 |
| 1 | policy / value | 0.74 / 0.58 | 0.012 / 0.077 | 0.20 / 0.082 | 0.033 / 0.27 | 0.018 / 0.017 |
| 2 | policy / value | 0.87 / 0.86 | **0.0** / 0.074 | 0.12 / **0.0** | **0.0 / 0.0** | **0.0 / 0.0** |

**The throttle.** Damping is D(t) = √(β2^t / f + 1 − β2^t), measured against an expert whose v is its own:

| f_k | at the fork | +5M | +10M | +20M | mean rate over 20M | over the first 10M |
|---|---|---|---|---|---|---|
| 0.015 | 8.2× | 5.0× | 3.1× | 1.45× | **0.35** | 0.21 |
| 0.03 | 5.8× | 3.6× | 2.3× | 1.24× | **0.46** | 0.29 |
| 0.09 | 3.3× | 2.2× | 1.5× | 1.08× | **0.64** | 0.47 |
| 0.2 | 2.2× | 1.6× | 1.2× | 1.03× | 0.77 | 0.64 |
| 0.7 (mon, the shared FFN itself) | 1.2× | 1.07× | 1.03× | 1.0× | 0.96 | 0.92 |

Two further facts:
- **The last round's board, threat and event experts get no policy or value gradient.** The heads do not read those
  positions after the last round, so the gradient is exactly 0. They are 3 × 65,920 = 197,760 of the 791,040 new
  parameters (25 %). Their copied momentum and weight decay move them slightly; nothing teaches them. An auxiliary
  head that reads those positions would change this, and the proxy did not include one (UNVERIFIED).
- C0's mon FFN is unthrottled, and X48's barely is (0.96). So X48 is "C0 plus experts that learn slowly", never "C0
  minus something". That keeps a GO interpretable; it is a NO-GO this weakens.

**Why it matters.** A NO-GO already reads narrowly ("within 20M"; M5). It now also covers a lever delivered at about
a third to two thirds of its rate on 3 of its 4 new experts. This is the N4 review's M1 lesson again: a capacity
lever whose engagement is not checked yields NO-GOs that are stronger in text than in fact. The PREREG has the
instrument already, §8.7's D_k / null. It is reported only, so it never qualifies the class.

**The fix: one dated NOTE before the GO** (the read's text and labels, not §5's classes):
1. **"experts not engaged" (a weak NO-GO).** It applies to a NO-GO where, at the final, §8.7's D_k / null ≤ 1.0 for
   every non-mon kind in rounds 0–1, averaged over linear1 and linear2 and over the three seeds. Round 2 is excluded
   because its zero-gradient experts cannot engage by construction. The read then states the experts did not diverge
   more than two copies of C0's own FFN drift would by chance. Because this is a label, not a class, it needs no
   change to `n4_screen`'s classes. It needs the §8.7 script (FINDING 8), run before the class is published.
2. **VOID 6's reason, corrected:** "copied experts add 30 % parameters, which move SLOWER than η at first (inherited
   v)". The KL bar itself stays. C0's trailing approx KL is 0.018 (`tb_read`, last 25 rollouts), far from 0.08.
3. **The NO-GO text** gains "with the experts' AdamW state copied from the shared FFN (Q-K3)".

**Not recommended tonight:** resetting the experts' v (fresh `exp_avg_sq`, copied `exp_avg`). That is a code change
at a new pin, with a new C0-comparability argument, so it is a new registration. If X48 reads NO-GO with "experts not
engaged", that variant is the natural follow-up.

---

## 3. M2: VOID 4 and the banner do not pin the copy (MAJOR)

**The problem.** VOID 4 says a seed is VOID if "`optimizer_state` has no `copied` count". `regrow_optimizer` writes
`copied` whenever the row declares copies, and **counts 0** when the source parameter has no state. That happens
when `_validate_or_reset_optimizer_state` resets a checkpoint's state. Such a run trains the REJECTED Q-K3 alternative
(fresh moments) under X48's name, and VOID 4 as written passes it. The chain's `banner.expect` (`"[GrowFork] grown:"`)
passes any growth line. Rule `n4_screen` also leaves VOID 4 to the operator (its NOTE 1 on `f3d5a3e8e`), so nothing
scripted checks the counts.

**The likelihood is low:** on P\* the counts are exactly right (§1). The fix is still free, and the run is the first
time this growth line appears in a production log (§11).

**The fix (before the GO):**
- **Chain file, all three items:** `"banner": {"expect": "fresh for 0, COPIED for 48", …}`. The trainee's line at
  the pin prints `optimizer state carried for 280, fresh for 0, COPIED for 48; the parent's function BIT FOR BIT on
  64 real rows …` (`model_build.py:290`). No other line contains `COPIED for`; the pool's growth lines read
  `pool snapshot grown (…)` and `seeded pool grown:`. A missing or different line is then VOID → `stop` within 20 min,
  automatically. (The `carried` count is left out of the substring on purpose: it depends on the checkpoint's
  optimizer, not the lever.)
- **PREREG NOTE, VOID 4 made exact:** the run's `model_config.json` `growth` has `optimizer_state` == {carried 280,
  no_state 0, fresh 0, copied 48}, `new_parameters` == {tensors 48, elements 791,040}, and `function_check`
  `bit_identical` true with both max \|Δ\| 0.0. Any other value is VOID. Run it as a script at arm end, beside
  `kit k9b`, not "read by hand".

---

## 4. The decision rule at my own σ

**σ_S.** G1 measured six production finals (C0 and A1, s3501–s3503) at 500 pairs. Their pooled within-arm SD is
0.41 pp on 4 df, against a per-final measurement SE of 0.55 pp. So the seed SD's point estimate is 0. Its 90 % upper
bound comes from the χ² bound on the total SD (0.79 at 500 pairs) minus measurement: **0.57 pp**. At 2,000 pairs the
measurement SE is 0.275, giving **σ_S ≈ 0.28 (point) to 0.63 (90 % upper)**. The PREREG's 0.4–0.6 sits inside this
range, and its pessimistic 1.07 is outside it.

| σ_S | S alone: GO-S / NO-GO-S at true 0 · +1 · +2 pp | joint (σ_k 1 pp), true (0,0): GO / NO-GO / UNDECIDED |
|---|---|---|
| 0.28 | 0.039 / 0.961 · 0.946 / 0.050 · 1.000 / 0.000 | 0.089 / 0.906 / 0.005 |
| 0.45 | 0.039 / 0.952 · 0.654 / 0.242 · 0.991 / 0.001 | (between the rows) |
| 0.63 | 0.039 / 0.866 · 0.422 / 0.277 · 0.896 / 0.008 | 0.089 / 0.817 / 0.094 |

**What this means for the screen:**
- It is **well powered on S**: a true +2 pp reads GO 0.85–0.94, and a true 0 reads NO-GO 0.82–0.91.
- **The family false GO is 0.089 at every σ_S.** S's Welch test runs at 0.039, and K adds the rest.
- A true K-only effect of +3 pp reads NO-GO 0.34–0.38. §5.3 item 4's text already says that ("smaller ones are not
  excluded").
- The 13 GPU-h are unlikely to read UNDECIDED (≤ 0.10 at null) UNLESS X48's seed spread is much larger than C0's.
  Welch absorbs that case, and the UNDECIDED line prints power at the observed σ.

**The engine and the control** hold:
- C0's three finals and P\* match the PREREG's sha256s.
- The pin carries `pinned_data_test.py`, so it reads its own `data/`. No `data/` commit, no Showdown submodule move
  and no conda change has landed since C0 trained (10-21 to 10-24).
- X48's finals and C0's are played on one main commit, and main loads a pin-written kffn checkpoint bit for bit
  (§1). The replay design is right: G1's banked S is at 500 pairs, and the banked K rows predate the regime split.

---

## 5. VOID completeness and the chain

| question | answer |
|---|---|
| The growth banner is missing but the run proceeds? | **Covered.** `postspawn.banner_guard` raises VOID at `within_min` (20) after the child log exists, and `on_void: stop` SIGTERMs the launcher. The trainee's banner prints at model load (`launcher_child.log` line ≈ 21 on LR-F s3503), after the pool growth: 20 snapshots, each a strict load, a growth and two 64-row CPU probes, ≈ 1–2 min of CPU. The 20 min window is ample. M2 makes the expected line exact |
| A growth that FATALs? | `[GrowFork] FATAL` / `[ModelVersion] FATAL` are VOID lines → stop |
| A restart (crash, canary resume, the 24 h interval)? | A same-run restart of a grown fork is a no-op that keeps the recorded stamp (`grow_fork.decide`; `test_kffn_a_grow_fork_resolves_on_and_a_restart_keeps_it` at the pin) |
| The pool? | The post-spawn T54 check compares snapshot NAMES (growth rewrites the files, not the names); VOID 5 reads the `grown` block |
| The +10M nodes for §8.4 / §8.7? | `thin: planner` kept C0's `checkpoint_290000009_steps.zip`; X48 uses the same planner |
| Not covered | the copied counts (M2); engagement (M1); the canary family's healthy ratio (m3) |

**Throughput: page, not stop, is right.** Slowness is not invalidity. But a page is a chain HOLD ("nothing after a
HOLD runs"), so s3502 / s3503 wait for the orchestrator (m5). The expected rate is ≈ 1,330–1,470 steps/s (LR-F
s3502 / s3503 read 1,459–1,470 today), against a bar of 1,000. A false page is unlikely.

---

## 6. MINOR

- **m1, "bit for bit":** it holds on the CPU gate. On the CUDA learner, identity holds up to bf16 GEMM rounding (§1).
  Fix the PREREG §2 and kit comment wording.
- **m2, checkargs and the dry run print "new parameters start fresh" for this growth** (FINDING 6). The text is
  generic and wrong for B-KFFN. The fix is on main, not the pin: print the row's own `Growable.optimizer_rule`
  (it already says "COPY … clone of that parameter's AdamW state").
- **m3, the canary family ruling is sound in mechanism but unverified in number.** The step adds no new op class: the
  same bf16 GEMM with fp32 accumulation, ReLU, a slice and a `cat`. At micro-batch 2048, the smallest segment GEMM
  has M = 6,144 rows, not the small-M regime. Verify it at no cost: compare X48's update-10 / 100 / 200 PASS ratios
  (descriptive, in `canary_verdicts.jsonl`) against C0's own (s3501: 1.29, 1.62, 0.79). If X48's healthy maximum
  exceeds C0's family maximum of 2.108, re-rule before 2.64 judges an X48 FATAL.
- **m4, this is the first CUDA compiled kffn learner.** K-11 compiled the kffn extractor with CPU Inductor only. If
  R1's startup gate or the inventory refuses on CUDA, the child exits FATAL_CONFIG after its banner. The chain then
  sees an END before throughput, which is a page and a HOLD, so the GPU idles until a human. §11's "the first GO is
  watched by hand" covers it if someone is awake at ≈ 20:00–20:30. Otherwise give the chain a non-X48 fallback item.
- **m5, a throughput page HOLDs the whole chain** (§5). Tell the Training session that s3502 / s3503 need a clear-hold
  after any page.
- **m6, the read's cost and device:**
  - C0 replayed at 2,000 pairs is ≈ 15 of the ≈ 31 S-lane-h, and half the Kakuna lane-h. If LR-F's or N1-P's reads
    use the same playing commit and regime, bank ONE C0 replay for all of them (the engine rule pools same-regime
    cells).
  - Pin S's device to CPU, G1's regime, so the §8.6 pin-drift line differs by engine only, not also by device.
- **m7, K's false GO rises with seed overdispersion:** 0.056 at σ_k 1 pp, 0.075 at 2 pp (`power.log` (2)). That makes
  the family 0.09–0.11. State the range, not 0.089.
- **m8, MIXED fires only on SIGNIFICANT harm.** A true (S +2, K −3) still reads GO 0.36. That is acceptable for a
  screen whose GO buys a re-read, not an adoption (§6 says so). The N4-P re-read must carry K at the PR rule's power
  before anything joins the recipe.

---

## 7. Exactly what to change before ≈ 20:00

1. **Chain file** `~/.claude/jobs/x35/chains/x35-x48-oct27.json`, items 0–2: `"banner"."expect"` from
   `"[GrowFork] grown:"` to `"fresh for 0, COPIED for 48"` (M2). Nothing else in the chain needs to change.
2. **PREREG, one dated NOTE (before the first GO):**
   - (a) VOID 4's exact counts, scripted at arm end (M2).
   - (b) the "experts not engaged" NO-GO label from §8.7 D_k / null, rounds 0–1, ≤ 1.0 (M1).
   - (c) VOID 6's corrected reason, and "copied AdamW state (Q-K3)" in the NO-GO text (M1).
   - (d) the "LR-elevated" GO label at > 1.1× C0's mean LR (M3). C0's mean LR is 1.02–1.05e-4 (`snapshot_history`;
     the rungs are ×1.2), so X48 at 1.206e-4 for most nodes would trip it.

   If `n4_screen` lands after the NOTE, (b) and (d) are rule params or post-rule labels, as LR-confounded is.
3. Nothing in §5's classes changes. No look 2, no seed change, no pin change.
