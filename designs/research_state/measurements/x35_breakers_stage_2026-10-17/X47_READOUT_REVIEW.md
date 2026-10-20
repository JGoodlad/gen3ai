# X47's readout: ADVERSARIAL REVIEW before its first read (2026-10-20)

**Verdict: SOUND-WITH-AMENDMENTS.** The equal-time comparison point is computed correctly. Its biases are measured,
and together they move s_eq by about 3M steps at most. None of the findings is CRITICAL: none corrupts a read
that is already registered. Three are HIGH, because each one decides what the read can license:
- **F-XR-1:** the G-C guard has never been run on a toy checkpoint.
- **F-XR-2:** what a final UNDECIDED does to X47b has not been ruled.
- **F-XR-3:** the `seq` tool reads a run that is still training as if it were finished.

The amendments (§3) can be registered now, because nothing has been read yet. Nothing in the kit, the readout or
the launch sheet was edited here.

**What was reviewed:**
- `src/main/x35/x47.py` (B-X47-READ), and the toy readout path in `readout.py`, `refpop.py`, `batch.py` and
  `tbrows.py`;
- [`BATCH1_LAUNCH_SHEET.md`](BATCH1_LAUNCH_SHEET.md) AM-1 to AM-9;
- [`X35_BATCH1_REVIEW.md`](X35_BATCH1_REVIEW.md), mainly F-RV-4 and F-RV-9;
- [`design_bot_ideas_screen.md`](../../../endstate/design_bot_ideas_screen.md) §1 and §2.1.

**What was measured.** The two live arms' own TensorBoard rows were read with `main.x35.tbrows`:
- toy C0 s3601, finished at 360,098,229;
- X47 s3601, running, read at 291.97M and again at 296.10M.

The kit's own `python -m main.x35.x47 seq` was then run on both. Inputs were the X26 drift pre-read
(`~/.cache/gen3ai/x26_drift_preread/results.json`) and the toy R.

**How it ran:** on CPU only (`CUDA_VISIBLE_DEVICES=""`, `nice -n 19 taskset -c 5-7,13-15`), reading `models/`
without writing to it. The scripts are in [`review_x47_2026-10-20/`](review_x47_2026-10-20/); they dump the rows
to `/tmp/x47rev/`. The R1 simulations were checked against the kit's own `decide_r1` and `x47.decide`: 0
mismatches in 2,000 draws.

---

## 0. The short version

- **ρ is 1.13, not the 1.3–1.4 the design estimated (MEASURED).**
  - The tool reads ρ = 1.125: collect 1.065, update 1.202. That is X47 s3601's first 6.1M steps against all 70M of
    toy C0 s3601.
  - So s_eq comes out near **352M**, about 62M of the 70M steps (88 %). The design expected 50–55M.
  - The PRIMARY (equal time) and the SECONDARY (matched steps) are therefore nearly the same read. Most of the
    "equal GPU-time" attack does not matter here: every bias we can measure moves s_eq by about 3M steps or less.
    On the X26 drift pre-read, C0's score changes by about 0.05 pp over 3M steps.
- **What limits the read is POWER, not the comparison point.**
  - At σ ≈ 1.07 pp (X26's node-to-node spread), a true +3 pp effect ends GAIN 16 % of the time after look 2;
    +4 pp, 58 %; +5 pp, 91 %.
  - With no effect, NULL is reached only 17 % of the time. The usual result is UNDECIDED, and today UNDECIDED
    licenses nothing (F-XR-2).
- **Two of the guards a GAIN needs have never run on the toy.**
  - G-C reads X26's banks, and whether it loads a toy (let alone a depth-4 toy) is UNVERIFIED (F-XR-1).
  - If G-C cannot be read, `decide` turns a GAIN into INCONCLUSIVE.
  - The S read itself spans three architectures, and one `main.h2h` engine serves at most two (F-XR-6).
- **One stated design fact is false: in-learn eval is INSIDE `collect_ms`.**
  - The code wraps the whole collector, callbacks included, in the collect timer
    (`instrumented_ppo/rollout_probes.py`).
  - So Σ counts the eval cycles: about 3.8 % of C0's Σ.
  - X47's eval costs only about 1.6 s more per cycle, a bias of about 0.25M steps. So the fix is to correct the
    claim and register it, not to change the statistic (F-XR-4).
- **Toy C0 is a valid control for X47 (checked in §1.4).** The two runs share the pin, the T\* zip, the argv
  except for `--grow`, torch, the bf16 trunk, the 20-snapshot pool names, K9(b) `warn` and the restart interval.

---

## 1. What the live arms measure

### 1.1 Time per iteration (each row is about 98.3k steps)

| quantity | toy C0 s3601 (all 713 rows) | X47 s3601 (rows to 296.1M) | ratio |
|---|---|---|---|
| `collect_ms` per row, median, eval rows excluded | 11.15 s (p5 10.40, p95 11.82, max 14.41) | 11.69 s | 1.05 |
| collect on an eval row (mean) | 27.53 s (35 rows) | 29.13 s (3 rows) | — |
| `eval/wall_sec` per cycle (mean) | 16.28 s | 16.60 s | 1.02 |
| `update_wall_s` per row, median | 9.26 s (min 9.18, max 11.05) | 11.30 s | 1.22 |
| first update (compile, cache warm) | 10.48 s | 12.82 s | +1.2 s / +1.5 s, once |
| Σ = collect + update, whole run | 15,209.8 s = B (one seed) | — | — |
| Σ ÷ wall time from the first row to the last | 0.999 | 0.974 (first 20 rows) | — |
| **ρ (the tool, `main.x35.x47 seq`)** | — | **1.125** (collect 1.065, update 1.202) | — |

- **Projected s_eq:** 290.0M + 70.1M ÷ 1.125 ≈ **352.3M**, so node 352M. The update-only basis gives about
  348M, and the final is 360M. All three nodes sit within 12M steps of each other.
- **Contention.** C0 s3601 ran from 02:32 to 06:46 PDT under the agent load of the night (rule 5b pins agents to 6
  logical CPUs). Its hourly median collect varied from 11.05 to 11.40 s, about 3 %.
  - Collect is about 55 % of Σ, so a 3 % shift sustained over a whole arm moves Σ by about 1.7 % and s_eq by
    about 1.2M steps.
  - Per-row spikes average out over 700 rows.
- **The X47 GPU-h per seed:** about 4.3 × 1.13 ≈ **4.9 GPU-h**, not the sheet's 5.6–6.0. T1 is then about
  3 × 4.3 + 3 × 4.9 ≈ 27.6 GPU-h, and T1b about 18.4.

### 1.2 The equal-time point under attack (each question in the brief)

| attack | how the code handles it | measured size | outcome |
|---|---|---|---|
| `update_wall_s` is logged one row late | each tag is summed over its own rows; Σ(N) leaves out at most the newest update (≈ 11 s) | ≈ 0.07 % of B, in X47's favour | **holds** (it is declared in `x47.py`) |
| collect is inflated by agent CPU load | not handled; collect is wall time | arm-level ≤ ≈ 3 % of collect ⇒ ≤ ≈ 1.2M steps of s_eq | **minor**; C2's interleave balances its sign over seeds |
| startup and compile | before the first row, so excluded on both arms; the compile cache is warm, so the first update costs only +1.2 s / +1.5 s | negligible | **holds** |
| in-learn eval | the docstring says it is excluded by construction, but it is **not** (F-XR-4) | 574 s of C0's Σ (3.8 %); X47 pays ≈ 56 s more | **the claim is false**; the bias against X47 is ≈ 0.25M steps |
| X47's slower steps | that is the point of the primary | ρ 1.125 | **by design** |
| the GROWN self-play pool (F-RV-9a) | X47 pays for depth-4 forwards on depth-2 functions | the whole collect excess is 6.5 %, and only part of it is the opponents ⇒ ≤ ≈ 2.3M steps of s_eq | **minor**. It is also the true steady-state cost of a depth-4 recipe: X47 promoted its own depth-4 self at 292M |
| contention that hits the two arms differently | B is the C0 seeds' mean; each X47 seed's node moves on its own | ≤ ≈ 1.2M steps per seed | **minor** |

**What all this adds up to.** The worst case, every bias stacked against X47, moves s_eq about 3–4M steps earlier.
The drift pre-read's slope for X26 is 1.66 ± 1.69 pp per 100M, so C0's score moves about 0.05 pp over that span.
For X47, the shift costs about 5 % of whatever X47 gains along its run. Both are far below one S measurement's
standard error of 0.59 pp. **The comparison point is not where this read can go wrong.**

### 1.3 The yardstick (toy R)

- R has six members, weighted equally:
  - **five from T\*'s own lineage:** 80, 150, 220, 280 and 290M (290M is T\* itself);
  - **one from outside it:** `pstar_x26`, the production P\*. T\* scores 41.6 % against it.
- The arms' TRAINING pool is the same lineage: 20 snapshots from 92M to 270M. Several sit right next to members of
  R: snapshot 224M beside member 220M, 270M beside 280M and 290M, and 136–142M beside 150M.
- So S rewards beating near-twins of one's own training opponents. Both arms share that, so Δ is fair. But a
  CAPACITY lever is exactly the kind that can fit specific opponents better. That is a reason to report the
  out-of-lineage member on its own (F-XR-7), not a reason to change R.
- **Measurement noise:** one cell of 500 pairs has an SE of about 1.5 pp. S over the six cells has an SE of about
  0.59 pp (the pre-read's 290M node: S 56.45, SE 0.59). Of the X26 node-to-node SD of 1.07 pp, about 0.59 is
  measurement and about 0.89 is something else.

### 1.4 Fairness (toy C0 as X47's control)

| item | toy C0 s3601 | X47 s3601 | |
|---|---|---|---|
| pin (`metadata.json` `git_hash`, `pin_history`) | `4609c155` | `4609c155` | = `x35_pin_batch1_toy.txt` |
| `--model` (T\*) | `…s4002_deep/…/checkpoint_290000087_steps.zip` | same | |
| argv | the kit's toy C0 line | the same line + `--grow trunk_layers=4` (`original_command`) | one lever |
| torch / trunk precision | 2.8.0+cu126 / bf16 | 2.8.0+cu126 / bf16 | |
| seeded pool (`pool_seed.json`) | 20 snapshots, 92–270M, none dropped | the same 20 names, plus a `grown` block (2 → 4) | digest record `2e86c4f9…` |
| K9(b) | `warn` | `warn` | C0 arm-end read: PASS (max 1.1e-4, p99 2.3e-5) |
| restart interval | 24 h | 24 h | no restart expected inside an arm |

The seeds are not paired in effect. X47's new rounds draw fresh random numbers at growth, so its dice stop matching
C0's from the first update. Welch does not pair the seeds, so this costs nothing; it only means "seed i" carries no
matching.

---

## 2. FINDINGS, ranked by severity (standing rule 7)

### HIGH

**F-XR-1. The guards that let a GAIN count have never run on a toy, nor on a depth-4 toy.**
- `x47.decide` REQUIRES G-F and G-C, and any required guard that is INCONCLUSIVE turns a primary GAIN into
  INCONCLUSIVE.
- G-C is read by `main.policy_spectrum`:
  - B1 is entropy on X26's 7,028-turn bank;
  - B2 is the starved share on X26's 306-turn truth.
- Both banks were built from production states. Nothing shows that the reader loads a toy checkpoint, let alone a
  grown depth-4 one, on those banks. The bank records an `arch_signature`, and the reader verifies it.
- G-F (`cross_stat` on X47 s_eq × C0 finals) needs a two-architecture engine. That is within `MAX_GROUPS`, but it
  has never been run with a grown toy.
- **Risk:** the one class that licenses X47b (ADOPT-CANDIDATE) cannot be reached for reasons of plumbing.

**F-XR-2. The most likely final class licenses nothing, and nobody has ruled what it does to X47b.**
- The design makes X47b conditional on GAIN or PER-STEP ONLY. At the realistic σ, a true effect of 0–3 pp ends
  UNDECIDED 83–84 % of the time (§4).
- So "UNDECIDED" will often be read as "depth does not help". The read cannot say that: after look 2, the smallest
  effect it detects 80 % of the time is about 4.6 pp at σ = 1.07.
- Two further effects make the primary slightly harder to pass:
  - the primary reads at 88 % of the steps;
  - if X47's gain grows along the run, the primary sees about 88 % of the final effect. A +5 pp final effect then
    ends GAIN about 75 % of the time, not 91 %.
- Look 1 sends a null X47 to look 2 48 % of the time, costing about 18 GPU-h.

**F-XR-3. `python -m main.x35.x47 seq` reads a run that is still training as if it had finished.**
- Run today on X47 s3601 at 296.1M, it returned `"s_eq": 296102866, "is_final": true`.
- `load_run` never checks that the run has finished: no check for `final_model.zip`, for `metadata.json`
  `num_timesteps` equal to the last row, or for `--steps` being reached.
- A read made before all six arms of a look have ended (the read lands in 12–24 h, while X47 s3603 may still be
  running) would silently mis-state:
  - B, if a C0 is unfinished;
  - an X47 seed's "final", and so its s_eq and ρ.

### MEDIUM

**F-XR-4. In-learn eval, checkpoint writes and snapshot saves are INSIDE `collect_ms`. The docstring says they are
"excluded by construction".**
- `RolloutProbes.collect_rollouts` starts the timer before `rc.collect(self, callback, …)`, so every callback is
  inside it.
- Measured on C0, each eval row's collect is 27.5 s against an 11.15 s median. The extra 16.4 s matches
  `eval/wall_sec`, 16.28 s.
- Σ is therefore about the wall time of the whole training loop (0.999 of the span from the first row to the last).
- The bias is small: X47's eval cycle costs about 1.6 s more, about 56 s over 35 cycles, about 0.25M steps of s_eq.
  But the statistic differs from its own description, and the description is what was registered.

**F-XR-5. At ρ ≈ 1.13, PER-STEP ONLY is mostly measurement noise, yet it is reported as "throughput-dependent".**
- Simulation: X47 gains nothing at all between s_eq and its final, so the two reads of a seed differ only by
  measurement noise (SE 0.59 each).
- PER-STEP ONLY still fires 5.2 % of the time at Δ = 3 pp and 9.1 % at Δ = 4 pp; with a real gain of +0.5 pp
  over the last 8M steps, 19–22 %.
- It licenses X47b either way, as the design says. But the label sent to the owner ("depends on X44 / X45 / X14")
  would be wrong in most cases.

**F-XR-6. One S read of an X47 node spans THREE architectures, and the plan for that read is not written.**
- An X47 node is a depth-4 toy. Its R cells are against five depth-2 toy members and one production P\*.
- `main.h2h.arch.MAX_GROUPS = 2`, so a single `play-many` call over all of R is REFUSED with a typed
  `CellArchMismatch`. Toy C0 needs only two architectures, so the toy refpop's band-cell recipe does not transfer.
- No driver or runbook exists for X35's S reads of the toy pair.
- The look-1 read needs:
  - 3 C0 finals × 6 cells;
  - 3 X47 s_eq nodes × 6 cells, plus 3 X47 finals × 6 cells for the secondary;
  - G-F's 3 × 3 cells.

  That is 63 cells of 500 pairs, about 63,000 games: about 15 h on one CPU lane at 1.18 games/s, or one GPU
  window.
- **UNVERIFIED:** whether `main.h2h` loads a grown depth-4 toy at all. The smoke for it ran into the box's
  heavy-job limit: one attempt timed out after 600 s with no output and left no process behind. It was not retried,
  because two heavy jobs were already running.

**F-XR-7. The yardstick is five-sixths in-lineage.**
- §1.3 explains why. A GAIN that comes only from the lineage members would show that X47 beats its own family
  better; it would not show general strength.
- The only check from outside the lineage is the P\* member, at 1/6 weight. G-K is reported only, and whether it
  can run on the toy is UNVERIFIED (X37c).

**F-XR-8. At look 2, the budget B and every s_eq are recomputed, and nothing registers that.**
- Look 2's B is the mean over five C0 seeds, so s3601–s3603's s_eq can move by a node.
- `compare` handles this correctly if it is given all five C0 runs. But the plan does not say that the look-1 S
  reads at the old nodes are dropped and the new nodes are read. That is up to 3 × 6 extra cells.

### LOW

**F-XR-9. F-RV-9 can now be closed: its two biases are measured and both are small.**
- (a) The grown pool costs at most about 2.3M steps; (b) load costs at most about 1.2M steps per seed (§1.2).
- F-RV-9 also proposed a "flip rule" (a class that differs between the Σ and update-only bases ⇒ UNDECIDED). That
  rule should NOT be adopted:
  - the two nodes (about 352M and 348M) are about 4M steps apart on a plateau;
  - any flip between them is the measurement noise of two separate S reads.

  `x47.py` already reports the update-only node without reading S there; keep it that way.

**F-XR-10. The registered DESCRIPTOR has no reader.**
- The bot screen §2.1 registers the weight norms of the extra rounds (`out_proj`, `linear2`) over training, but
  `x47.py` does not compute them.
- Without them, a NULL or a LOSS cannot tell two things apart: "the rounds never moved off identity" (the
  optimisation never engaged them) and "the rounds learned and did not help".

**F-XR-11. A LOSS or NULL is a verdict on THIS recipe, not on depth.**
- In this recipe:
  - the new parameters start with zero AdamW moments at step 0, with no LR warm-up of their own;
  - AdamW's decay pulls the zero-initialised projections back toward zero;
  - the trunk is bf16.
- The dose is also different from X47b's: the toy goes from 2 to 4 rounds (×2), production from 3 to 4 (×1.33).
  A toy effect does not scale to X47b's dose in any way we can state.
- K9(b) VOID is also likelier for the deeper bf16 trunk. That is a cost (a re-run), not a bias.

**F-XR-12. Which file is "the final" is not registered.**
- C0's final is `final_model.zip` (`num_timesteps` 360,098,229, the last row's step). Its last checkpoint is
  `checkpoint_360000067_steps.zip`, a different file.
- An X47 s_eq node is a `checkpoint_<N>_steps.zip`, or `final_model.zip` when `is_final`.

### INFO

- **I-1.** The sheet's ρ and GPU-h for X47 (1.3–1.4; 5.6–6.0 GPU-h per seed) and the bot screen's "s_eq ≈ 50–55M"
  are superseded by §1.1: ρ 1.13, about 4.9 GPU-h, s_eq about 352M. The bot screen is updated in the same commit
  as this review, because it is always-current.
- **I-2.** C0 s3601's in-trainer K9(b) `warn` bar (1e-4) fired once: max 1.1e-4 at update 34,070
  (`behaviour_violations.jsonl`). X35's arm-end ceilings (1e-3 / 1e-4 p99) PASS it. This is the expected behaviour
  of `warn`, and it is recorded so that a later reader is not surprised.
- **I-3.** X47's look 1 costs about 3 × 4.9 ≈ 14.7 GPU-h of X47 training; the design estimated about 17. T1b
  (about 18.4 GPU-h) follows with a probability of 0.48 to 0.98, depending on the true effect (§4).

---

## 3. AMENDMENTS (pre-read; each can be registered as written)

| id | severity | amendment | where |
|---|---|---|---|
| **AM-X47-1** | HIGH (F-XR-1) | **Prove the guards can be read before the first read.** On CPU at rule 5b, run G-C's B1 reader on T\* and on X47 s3601's newest 1M checkpoint (minutes each), and run one `cross_stat` cell between a grown and a plain toy (10 pairs). If either cannot be read, the orchestrator rules BEFORE the read whether G-C is required on the toy or REPORTED only (as G-K is). Without that ruling, a GAIN ends INCONCLUSIVE by code | a CPU pre-read; `x47.REQUIRED_GUARDS` if the ruling changes it |
| **AM-X47-2** | HIGH (F-XR-2) | **Register what each final class does to X47b, before the read.** Proposed: GAIN / ADOPT-CANDIDATE ⇒ X47b runs; PER-STEP ONLY ⇒ X47b runs (as written); **UNDECIDED with Δ̂ > 0 ⇒ X47b keeps its rank-9 conditional slot, unlicensed but not dropped, and the report states the detectable effect ("no effect ≥ 4.6 pp at 80 %")**; NULL ⇒ X47b drops in rank, not out (the toy only ranks before calibration); LOSS ⇒ X47b is not run on this evidence. The report never says "depth does not help" on anything but a NULL | `design_bot_ideas_screen.md` §1.4 / §2.1 |
| **AM-X47-3** | HIGH (F-XR-3) | **`seq` refuses a run that has not finished**: it needs `final_model.zip` and `metadata.json` `num_timesteps` equal to the last TB row's step, and that step ≥ the arm's `--steps`. It also needs the run's K9(b) arm-end record to read PASS (AM-6), so a VOID arm cannot enter a read. A refusal is INCONCLUSIVE (exit 2). Until this is built, the read is made only after `kit k9b-batch` has passed for all six arms | `x47.load_run` + a test that fails on revert |
| **AM-X47-4** | MEDIUM (F-XR-4) | **Register Σ as the loop's wall time INCLUDING in-learn eval, checkpoint writes and snapshot saves**, which is what the code measures, and correct the docstring. Report the eval share per arm (Σ `eval/wall_sec`) beside ρ. Do NOT subtract it: the measured bias is about 0.25M steps, and subtracting would couple Σ to a third tag's row alignment | `x47.py` docstring |
| **AM-X47-5** | MEDIUM (F-XR-5) | **A PER-STEP ONLY report carries D = mean over seeds of (S_final − S_s_eq)** with its two-sided 95 % t interval (df n − 1). It is called "throughput-dependent" only when the lower bound is > 0 (rule 8 at 0); otherwise it is called "BOUNDARY: the secondary crossed the bar within measurement noise". The class, and what it licenses (AM-X47-2), do not change | `x47.read` (reported) |
| **AM-X47-6** | MEDIUM (F-XR-6) | **Write X47's read plan before the read**: the cells per engine (engine A: X47 nodes × the five toy members, plus G-F's X47 × C0 cells; engine B: X47 nodes × P\*; engine C: C0 finals × all of R), the family and request names, the 500-pair cells, about 63k games for look 1, and where it runs (a GPU window or about 15 CPU lane-hours). The first item of the plan is a 2-pair smoke of a grown toy against T\* and against P\* | the launch sheet §4.2 or a read sheet |
| **AM-X47-7** | MEDIUM (F-XR-7) | **REPORT, registered now, never deciding:** each member's Δ with its Welch interval, and the out-of-lineage member (`pstar_x26`) on its own. A GAIN whose `pstar_x26` Δ̂ ≤ 0 goes to the owner labelled **"in-lineage"** | `x47.read` (reported) |
| **AM-X47-8** | MEDIUM (F-XR-8) | **At look 2, B and every X47 seed's s_eq are recomputed over all five C0 seeds.** S is read at each new node, and a look-1 S is reused only when its node did not move. The look-1 nodes and the look-2 nodes are both recorded | `x47.py` docstring; the read plan |
| **AM-X47-9** | LOW (F-XR-9) | **Close F-RV-9.** Its biases are measured and small (§1.2). The update-only s_eq stays a REPORTED node, never read for S. No flip rule | `X35_BATCH1_REVIEW.md` F-RV-9 (by the orchestrator) |
| **AM-X47-10** | LOW (F-XR-10) | **Build the descriptor reader**: Frobenius norms of each new round's `out_proj` and `linear2` on every 1M checkpoint of each X47 seed, on CPU (minutes). Reported beside the verdict, with one line of reading: "engaged" or "stayed near identity" | a small CPU script; `x47` reported |
| **AM-X47-11** | LOW (F-XR-12) | **Register the files:** "final" = `final_model.zip` (its `num_timesteps` = the last row's step); an s_eq node = `checkpoint_<N>_steps.zip`, or `final_model.zip` when `is_final` | `x47.py` docstring |

---

## 4. Power (the registered R1 at 97.5 % two-sided per look, bar ±2 pp; 200k draws per row)

σ is the per-seed SD of S, the same on both arms. Its anchor is X26's node-to-node SD of **1.07 pp** (the drift
pre-read's 290 / 330 / 380M nodes: one seed, three nodes, measurement SE 0.59). The pre-read's bracket is 0.8–1.6.
The look-2 seeds are drawn only on CONTINUE, as the kit does.

| σ (pp) | true Δ (pp) | look 1: GAIN | look 1: CONTINUE | after look 2: GAIN | after look 2: NULL | after look 2: UNDECIDED |
|---|---|---|---|---|---|---|
| 1.07 | 0 | 0.000 | 0.475 | 0.000 | 0.165 | 0.835 |
| 1.07 | 2 | 0.008 | 0.975 | 0.016 | 0.014 | 0.970 |
| 1.07 | 3 | 0.052 | 0.947 | 0.164 | 0.001 | 0.835 |
| 1.07 | 4 | 0.184 | 0.816 | 0.576 | 0 | 0.424 |
| 1.07 | 5 | 0.407 | 0.593 | 0.911 | 0 | 0.089 |
| 1.07 | 6 | 0.645 | 0.355 | 0.994 | 0 | 0.006 |
| 1.2 | 0 | 0.000 | 0.486 | 0.000 | 0.098 | 0.902 |
| 1.2 | 4 | 0.146 | 0.854 | 0.477 | 0 | 0.523 |
| 1.2 | 5 | 0.329 | 0.671 | 0.837 | 0 | 0.163 |
| 1.6 | 5 | 0.185 | 0.815 | 0.580 | 0 | 0.420 |
| 0.8 | 0 | 0.000 | 0.423 | 0.000 | 0.416 | 0.584 |
| 0.8 | 4 | 0.329 | 0.671 | 0.837 | 0 | 0.163 |

- **The false GAIN rate is about 0**: 0.016 even at Δ exactly on the bar.
- **The half-width of the interval at σ 1.07:** about 3.05 pp at look 1 (a GAIN needs Δ̂ > 5.05) and about 1.86
  pp at look 2 (Δ̂ > 3.86).
- **NULL** needs a half-width under 2 pp, which only look 2 reaches, and only barely.
- **Spurious PER-STEP ONLY** (X47 gains nothing between s_eq and the final; σ_train 0.89, σ_meas 0.59): 5.2 % at
  Δ = 3, 9.1 % at Δ = 4, 3.6 % at Δ = 5, 0.4 % at Δ = 6 (`review_x47_2026-10-20/perstep.py`).

---

## 5. What each class would and would not license

| final class | licenses | does NOT license |
|---|---|---|
| **ADOPT-CANDIDATE** (primary GAIN, G-F and G-C pass) | X47b (production 3 → 4, about 16 GPU-h); the claim "T\*'s plateau is broken by two identity-init rounds at about equal wall time, on the toy's yardstick"; R-T2's hint that the toy ceiling is bound by capacity | adopting depth 4 anywhere (the toy only ranks until X35's transfer calibration passes); the mechanism "composition", which needs the descriptor (AM-X47-10) and an equal-parameter width control; any production effect size (the dose differs, F-XR-11); general strength if the P\* member does not move (AM-X47-7) |
| **TRADE** (GAIN, a guard FAILs) | a report to the owner naming the guard; X47b only on the orchestrator's ruling | adoption |
| **PER-STEP ONLY** | X47b (as written); "BOUNDARY" or "throughput-dependent" per AM-X47-5 | the throughput story unless D's interval excludes 0 |
| **UNDECIDED** (final) | under AM-X47-2: X47b keeps its conditional slot; the report says "no effect ≥ ≈ 4.6 pp" (σ 1.07, 80 %) | "depth does not help"; dropping X47b |
| **NULL** | "\|Δ\| < 2 pp at about equal wall time on the toy, under this growth recipe"; X47b lower in rank | dropping X47b before calibration (§1.5 there); a verdict on depth itself (F-XR-11) |
| **LOSS** | "two identity-init rounds grown at T\* with fresh AdamW state, bf16, this LR, hurt the toy"; no X47b on this evidence | "depth hurts"; a fresh deeper network or a warmed-up growth may differ |
| **INCONCLUSIVE** | nothing; fix the named input and re-read | a reading of Δ̂ |

---

## 6. What was not verified (standing rule 7)

- whether `main.h2h` plays a grown depth-4 toy checkpoint (F-XR-6): the smoke was not completed;
- whether `main.policy_spectrum` reads a toy checkpoint on X26's banks (F-XR-1);
- whether `main.anchors` loads a toy (X37c, G-K; already UNVERIFIED);
- σ for the TOY: every power figure uses production's X26 drift spread as a proxy. The first toy S reads (three C0
  finals) give σ̂ with 2 df;
- the per-arm ρ of X47 s3602 and s3603 (§1.1 is s3601's first 6.1M steps only; the tool recomputes ρ at the read).
