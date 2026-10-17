# X26's K9(b) stop at update 13,900 (2026-10-17): NUMERICAL, not a fault

**Verdict: NUMERICAL (benign).** The one row over the 1e-4 bar is the most fp32-ill-conditioned row we
have. Its gap between the stored behaviour log-prob (μ) and the learner's recomputed log-prob (π) is two
equally valid fp32 evaluations of one function, and the gap is no larger than that row's own
rounding spread. No structural fault class fits the evidence. **No checkpoint is void.** The last clean
update is 13,899, and `final_model_exception.zip` holds its weights.

| | |
|---|---|
| run | `rb_x26_s2601` (production, bf16 trunk, pin `9c033489`, cap 500M) |
| event | update 13,900, step 136,974,910, 06:49:40. The K9(b) probe ran before any optimizer step: max \|log π − log μ\| over the 982 judged rows was 1.0109e-4 against the 1e-4 bar. This was the run's first K9(b) record |
| inputs (read only) | the orchestrator's copies at `~/.cache/gen3ai/x26_k9b_hold/` (SHA256SUMS there): the dump (`behaviour_violation_u13900_policy.pt` / `_rows.npz`, `behaviour_violations.jsonl`), `final_model_exception.zip`, `checkpoint_136000153_steps.zip`; the run's `tb/` events |
| code | the pinned tree (`git archive 9c033489 src`, plus links to `data/` and `deps/`; `data/` is identical between `9c033489` and `9ab23c93`). CPU only, `CUDA_VISIBLE_DEVICES=""` |
| scripts | `replay.py` → `replay.json`; `analyse.py` → `analysis.json`; `tb_history.py` → `tb_history.json` |

## 1. The replay

The dump holds the observations of the 32 WORST rows of the fatal update's full-buffer scan. The scan covered
90,615 current rows, and the violator is the worst of them. It does **not** hold the other 1,023 probe rows'
observations, so every per-row number below is over those 32 rows (FINDING F1).

For each row, `replay.py` computes:
- **ref64**: the learner's forward in **float64** (eager, CPU). A `TorchFunctionMode` upcasts every float32
  operand and dtype argument, so every intermediate is float64. Two float64 paths (the learner's
  `evaluate_actions` and T2's `DecisionModule`) agree to 5.8e-14. This is the exact function to well below
  any fp32 effect.
- the fp32 learner path on CPU, at batch 32 and at batch 1;
- the fp32 T2 decision forward (`DecisionModule`, eval mode, no_grad) on CPU, at buckets 8, 64 and 256;
- a **sensitivity** reading: 16 draws in which every weight is moved to a random float32 neighbour (one ulp)
  or left alone. The spread of a row's log π over these draws measures how far two equally valid fp32
  evaluations of THAT row sit apart;
- the same rows under the checkpoint ten updates earlier (`136,000,153`), showing what a stale-weight fault
  looks like.

**Reproduction.** The dumped weights are bit-identical to `final_model_exception.zip`'s policy (1,113 keys).
The dump's rows align with the JSON record: the stored log μ, action and mask match, and the stored
distribution's legal set equals the mask. The stored and GPU-learner values in the dump give
|Δ| = 1.0109e-4, which reproduces the record.

**The violator, row 903** (buffer t 3, env 213): 2 legal actions (mask `00010100000`), the taken action 5 at
p = 0.167, raw logits −3.474 / −5.083, entropy 0.45.

| path (fp32 unless named) | log π − ref64 (row 903) | median \|·\| over the other 31 rows |
|---|---|---|
| stored μ (T2: inductor-compiled + CUDA graph, eval, GPU) | **+6.4e-5** | 1.8e-5 |
| learner, GPU (the probe's recomputation) | **+1.65e-4** | 1.6e-5 |
| learner, CPU, batch 32 / batch 1 | +1.48e-4 / +1.45e-4 | 1.7e-5 / 1.4e-5 |
| T2 decision forward, CPU, buckets 8 / 64 / 256 | +1.52e-4 (all three) | 1.7e-5 |

- **Every fp32 evaluation of row 903 misses the exact value by 0.6–1.65e-4**, which is 8–9× the other rows'
  typical error. The stored μ is the CLOSEST of them to the exact value. The learner is further away, in
  the same direction as every CPU fp32 path.
- **The sensitivity reading ranks row 903 first of the 32.** Its log-π spread under one-ulp weight rounding is
  1.13e-4 (std 3.7e-5), against a median of 4.0e-5 (std 1.1e-5) for the other rows. Weight rounding ALONE
  moves this row across a range wider than the gate's 1.01e-4 gap.
- **Normalised by its own sensitivity, the violator is ordinary**: gate |Δ| / sensitivity std is 2.7 for
  row 903, against a median of 3.1 (max 9.9) for the other 31 rows. Row 903 crossed the bar because its
  conditioning is the worst, not because its gap is unusual for its conditioning.
- **Where the error grows.** The fp32 error sits in ONE raw logit (−1.85e-4 on one legal logit, −2.7e-6 on
  the other). Its relative error grows through the extractor, which ends at 8× the other rows' median, and
  peaks in the damage operator's worst-case projection `op_worst_proj` (53×) and the pointer head's
  `switch_score` (17×). This is a smooth, steep region of the function. A hard cutoff would show as a jump:
  the 16 rounding draws spread continuously over 1.13e-4, and the fp64 and every fp32 path agree in sign.

## 2. The decomposition

**(i) A different numerical path: YES, and it is the whole gap.** T2 serves μ from a different program:
the decision module compiled by inductor and captured in a CUDA graph, in eval mode, at bucket shapes. The
learner computes π with the eager `evaluate_actions` in train mode. Both are fp32 at `highest` precision.
On this row, ANY fp32 program lands 0.6–1.7e-4 from the exact value, and the two programs landed 1.01e-4
apart. On CPU, batch composition and bucket shape do not move the row: the decision forward gives the same
value at buckets 8, 64 and 256, and the learner gives 1.45e-4 vs 1.48e-4 at batch 1 vs 32. So the spread
comes from the KERNELS (inductor's fused reductions vs eager's), not from the batch.

(ii) **Structural: excluded on every class we could test.**
- **Stale weights / policy version.** The row was served at policy version 160 by T2 slot 24, the current
  version and slot. The dumped weights equal the exception save. Weights ten updates old move these rows by a
  median of 0.84, and by at least 0.0094 on every one of the 32 rows. A stale-weight fault would move every
  row of the version, but the full scan's other 90,614 current rows all sit below 5.7e-5 (p99 9.3e-6).
- **Row misalignment / wrong action.** The dump aligns with the record. The stored distribution's legal set
  equals the mask, and the stored μ of BOTH legal actions agrees with the recomputed π to 2e-5 / 1e-4. That
  is one shared logit-gap shift of 1.2e-4, the same for both actions, which a wrong row or a wrong action
  index cannot produce.
- **Mask / legal-action mismatch.** The legal sets are identical, and an illegal action carries −∞ / −1e8,
  not a 1e-4 offset.
- **Observation mismatch.** This cannot be replayed directly, because the collector's input for the row is
  not stored separately. Two facts argue against it. First, the violator is the worst-conditioned of the 32
  rows on all three measures (the fp32 error, the sensitivity, the GPU learner's error). Second, its gap is
  typical for its conditioning (§1). A corrupted observation has no reason to land on the
  worst-conditioned row; under that hypothesis the chance is 1 in 32.

(iii) **A near-tie the margin missed: NO.** The row's closest declared selection site (the
`hypothesis_set.py:214` argsort) is at a relative margin of 0.0785, nearly 400× the 2e-4 exclusion margin.
No undeclared hard cutoff flips either: the rounding draws are continuous and every path agrees in sign. The
"different kind of cutoff" is a STEEP SMOOTH region (§1, "Where the error grows"). The margin system does not
and should not declare it, because no fp32 rounding flips it.

## 3. The forecast (`tb_history.json`, the gate's own statistic `behaviour/max_abs_dlogp_judged`, 1,388 updates)

| window | median per-update judged max | p90 | max | p99 median |
|---|---|---|---|---|
| 0–10M | 6.9e-6 | 1.2e-5 | 3.2e-5 | 3.1e-6 |
| 60–70M | 1.6e-5 | 2.2e-5 | 5.4e-5 | 6.7e-6 |
| 100–110M | 1.9e-5 | 3.5e-5 | 4.8e-5 | 7.9e-6 |
| 130–137M | 2.3e-5 | 3.7e-5 | 9.7e-5 | 9.3e-6 |

- The judged max's scale grows about linearly, by ≈ +1e-5 per 100M steps from 10M on. The near misses were
  9.52e-5 (116.5M) and 9.73e-5 (130.1M), and then 1.01e-4.
- **The all-current max, ties included,** is a different statistic: it reached 0.041 (57.4M), 0.039 (79.9M),
  1.4e-3 and 9.4e-4. All of those were tie-excluded rows, so they are not violations.
- **The rate now.** Over the last 30M steps (306 updates) the per-update P(judged max ≥ 1e-4) is 0.33 %
  empirically, and a log-linear tail fit over 3e-5 to 9e-5 gives 0.37 %. That is one recurrence per ≈ 270
  updates ≈ 26M steps at today's scale.
- **Forward** (ESTIMATE; it assumes the linear scale growth continues, and evaluates the tail fit at
  1e-4 / scale):

  | step | scale | P(judged max ≥ 1e-4) per update | steps between recurrences |
  |---|---|---|---|
  | 200M | ×1.28 | 1.0 % | ≈ 9.6M |
  | 250M | ×1.49 | 1.7 % | ≈ 5.7M |
  | 370M | ×2.0 | 3.9 % | ≈ 2.5M |
  | 500M (cap) | ×2.56 | 6.4 % | ≈ 1.5M |

  Summed to the cap, that is **≈ 110 expected K9(b) stops** if the run goes the distance.
- At the cap the fit puts P(judged max ≥ 1e-3) at ≈ 5e-9 per update. The p99, scaled the same way, would
  sit near 2.4e-5.

## 4. Options for X26 at its pin, and the recommendation

- **(A) Resume `fatal` as is.** This is viable only for the next ≈ 25M steps. Over the remaining 363M the
  forecast gives ≈ 110 FATALs (≈ 10 before 200M). Each costs the steps since the last save, up to 1M (≈ 0.5M expected) plus
  the restart. Under registered item 3, each one is also a GIGO stop that must be DIAGNOSED before resuming.
  That is the same CPU replay every time, for an event this measurement shows to be the expected tail of
  fp32 rounding on a sharpening policy. (A) treats a predicted rounding event as a fault, so it fails rule 8
  from the other side.
- **(B) Resume with `--behaviour-check warn` under a new registered Amendment 3 with monitored ceilings.
  RECOMMENDED.** K9(b) keeps running, recording and dumping every update (under `warn` a violation's scan
  is a seeded 4,096-row sample). The training session reads `python -m main.ops.tb_read <run> --tag
  behaviour/max_abs_dlogp_judged --tag behaviour/p99_abs_dlogp_current` at every read point, at every planned
  restart and at least daily. It STOPS the run (a GIGO stop under item 3) on any ONE update with:
  1. judged max ≥ **1e-3**: 10× the bar, and 16× the forecast median judged max at the cap (≈ 6e-5). The
     fitted rounding tail puts it at ≈ 5e-9 per update even there. A fault that moves one row by 1e-3 is
     caught;
  2. p99 over the current rows ≥ **1e-4**: 10× today's 9.3e-6, and 4× the cap forecast (≈ 2.4e-5). Stale weights or a
     transport fault move EVERY row by ≥ 1e-2 (§2 ii), so the p99 is the many-row detector;
  3. any non-finite |Δ|.

  The ceilings are DETERMINISTIC (rule 8): they are fixed thresholds on the logged per-update statistic,
  each ≥ 4× from its forecast at the cap, with no averaging and no tolerance. The cost is that a LOCALIZED
  fault between 1e-4 and 1e-3 on one row now passes. That is exactly the band this replay shows rounding
  reaches. Item 2's existing `warn` clause (judged max < 1e-4) is superseded for X26 by these ceilings.
- **(C) Other options, not recommended.**
  - `--behaviour-check off` removes the detector entirely.
  - An fp32-trunk switch changes training, not the probe: the rounding is in the fp32 forward on BOTH sides
    of the comparison.
  - A watcher that enforces (B)'s ceilings automatically is worth building, but it is new ops code, not a
    blocker. Until it exists the ceilings are read by hand, as item 2 already does.

**Resume from `final_model_exception.zip`, not `checkpoint_136000153`.** Its weights are bit-identical to the
weights that played the fatal rollout and passed every earlier update's probe: the state after update
13,899, before any optimizer step of 13,900. It carries the optimizer state, `latest.txt` already names it,
and the launcher's own crash restart resumes it. With a NUMERICAL verdict nothing after 136.0M is void, so
falling back would throw away ≈ 975k clean steps for no gain.

The resume argv (same run, same pin, every other flag inherited; the training session verifies it with
`--dry-run` first, as item 2 requires; FINDING F4):

```
python -m main.launcher --model models/rb_x26_s2601/final_model_exception.zip --behaviour-check warn --device cuda
```

## 5. A design note for FUTURE runs: should K9(b)'s bar scale with sharpness?

Yes in kind, but not with "sharpness" as such. What sets a row's fp32 gap is its CONDITIONING: the rounding
spread of its log π, which here was 9× the median on a row with only moderate logits (|raw| ≤ 5.1) and
moderate p (0.17). Entropy and logit magnitude do not predict it. The principled bar is per row: judge |Δ|
against k × the row's own fp32 sensitivity. The sensitivity can be read with a second forward under a
one-ulp weight dither, on the violating rows only, so it costs nothing on a clean update. A row fails when
|Δ| ≥ max(1e-4, k·σ_row) with k fixed (today's 32 rows give ratios ≤ 9.9, so k ≈ 20 would be a
measured choice). Only a fixed k keeps the gate deterministic. The cheaper alternative is a fixed bar that
tracks a logged scale, for example a bar of 10 × the run's trailing p99. It is easier to build but weaker:
it moves with the population, not the row. Either one is a design for the next pin's `consistency.py`, not
for X26.

## FINDINGS (standing rule 7)

- **F1.** The K9(b) dump stores observations for the 32 worst SCAN rows and the violator only, not for the
  1,024 probe rows. "Recompute all 1,024 rows" was therefore done for the 32 rows held. The scan record
  covers all 90,615 current rows (1 over the bar, p99 9.3e-6).
- **F2.** The inductor-compiled decision path could not be rebuilt on this box's CPU: a g++ `CppCompileError`
  in the generated kernel, recorded in `replay.json` `compiled_error`. T2's exact GPU program (inductor CUDA +
  a CUDA graph) was NOT re-run: there was no GPU lease. The stored μ is that program's output, and §1
  places it within the row's fp32 envelope.
- **F3.** The per-row float buffers are lookup tables. A one-ulp dither of them is a different table, not
  rounding: the first draft dithered them and read a 2-nat "sensitivity". The sensitivity reading therefore
  dithers PARAMETERS only.
- **F4.** I did not run the launcher's `--dry-run` for the resume argv: this agent's sandbox refused the
  module run. The training session must run it before the resume, together with `python -m main.checkargs`.
- **F5.** The forecast assumes the judged max keeps growing linearly. If it grows faster, (B)'s p99 ceiling is
  the first to tighten. Re-read the forecast with `tb_history.py` at each 25M read point.
- **F6.** The TensorBoard record has no row for update 13,900: the fatal raised before the record was
  flushed. `tb_history.py` adds the 1.0109e-4 by hand.
