# X34 canary u1700 UNCONFIRMED — a bf16 conditioning tail, not a compile bug (2026-10-19)

**Run.** X34 = `rb_toy_ent_s4002_deep`, pin `ae2d0901`, bf16 trunk, `--compile-trainer`, live.

**Event** (launcher 21:29:35, update 1700, 267,205,580 steps). The log said: "compiled DISAGREES with eager — the EAGER
arm moved a fixture row's log-prob by up to 0.856 from the fp32 eager reference … switched 0 row(s)' PPO clip branch".

| read | max abs dlogp, compiled | max abs dlogp, eager |
|---|---|---|
| scheduled read (all 64 rows) | 0.521 | 0.856 |
| confirmation, slice 2 (rows 0-31) | 0.521 | 0.856 |
| confirmation, slice 1 (rows 32-63) | 0.191 | 0.151 |

The verdict was UNCONFIRMED. The next disagreement is FATAL.

## Verdict

**This is a bf16 conditioning tail that the canary's design misreads as a compile disagreement. It is not a compile bug.**

1. **The arm that broke the margin was EAGER bf16, the canary's own yardstick.** At the pin, `compile_regions._r1_judge`
   hands `_clip_margin_rule` the arms in the order `compiled`, `eager`, `fp32 reference`.
   - Compiled read 0.5209, inside the margin M = 0.72 (4 × `R1_LOGPROB_SHIFT_MAX` 0.18), so it passed.
   - Eager read 0.8563 and raised.
   - The scheduled and slice-2 records carry no `fp32 reference` key; they stop at the raising arm. Slice 1, which passed,
     carries it.
   - No clip branch switched.
   - The bf16 gradient rule's worst ratio was 3.58 against K = 12 (`pointer_head.move_score.bias`).

   So nothing compiled disagreed with anything. The canary words a yardstick out of margin as "compiled DISAGREES with
   eager", and it counts the event on the confirm / FATAL ladder.
2. **Both bf16 arms move together across X34's whole history** (`result.json` → `canary_history_x34`). The compiled and
   eager shifts track each other:
   - 0.06–0.31 through u1600;
   - both jump on the same state at u1700 (0.52 / 0.86);
   - both fall back on the other half (0.19 / 0.15).

   A forward miscompile would move the compiled arm alone. X26's u600 archive (`~/.cache/gen3ai/x26_canary_u600/`)
   shows the same pattern: eager 0.38–0.43 and compiled 0.28–0.48 on passing canaries, before its 0.76 unconfirmed.
3. **The fp32 function itself is this ill-conditioned on the fixture rows.** These reads are CPU, the pin's code, the
   checkpoints strict-loaded, the canary's own rows (`r1_batch`, B = 2048).
   - In fp32 alone (no bf16, no compiler), dithering the WEIGHTS by half a bf16 ulp (relative 2^-9,
     `diag_bf16_canary.dithered`) moves a fixture row's log-prob by:
     - at 267.0M (12 seeds): up to **0.41** (row 0) and **0.54** (row 43);
     - at 268.0M (24 seeds): up to 0.30 and 0.34.
   - Per-row dither MEDIANS reach 0.19 at 267.0M (rows 1, 20, 23, 28, 33, 43, 44, 47 and 50 are all above 0.1).
   - Eager bf16 at the same weights reads 0.095–0.126.
   - The canary's margin of 0.72 is therefore only 1.3–2.4× the function's own sensitivity to a half-ulp weight change.
     A 0.856 tail on a state two updates later, the u1700 weights, is in family.
   - It is a per-STATE lottery, as on X26 (`designs/training/compile_flags.md` "bf16 drift on a sharpening policy":
     adjacent checkpoints differ 10×).
4. **Compiled equals eager at the SAME precision.** On CPU Inductor at 268.0M (the production `compile_regions.install`
   with its tile2d workaround; C++ codegen, not the run's Triton kernels):

   | comparison | slice 2 | slice 1 |
   |---|---|---|
   | compiled bf16 vs fp32 | 0.126 | 0.103 |
   | compiled bf16 vs eager bf16 | 0.028 | 0.017 |

   The compiled-vs-eager gap is 4–6× tighter than either arm's distance from fp32. The canary's yardstick, both arms
   against the fp32 reference, measures mostly bf16's conditioning, not the compiler.
5. **The op is a smooth accumulation, not a discrete one.** The module bisect is on 267.0M's slice-2 worst row (row 4),
   eager bf16 against eager fp32, relative L2 of each module's output for that row (`result_267M.json` →
   `bisect_slice2_worst_row`).
   - Everything before the first `team_transformer` layer runs fp32 (exactly 0).
   - The error enters at transformer layer 0 (2.5e-3), grows through layer 1 (6.4e-3), `cls_pool` (8.6e-3) and
     `projection` (6.6e-3), and reaches `pointer_head.move_score` at 1.3e-2 relative (0.12 absolute on logits up to 8.3).
   - No layer jumps. There is no near-tie selection and no outsized attention bias.
   - The policy MLP's first layer has pre-activations up to 52, so its tanh saturates. That is the robust direction.

   The taken action is a LOW-probability one (log π = −5.3 against a 0.88-probability favourite, action 7). Then
   d log π(a) ≈ dz_a − dz_favourite: a row whose taken action is unlikely carries the full logit error into the canary's
   statistic, with no softmax damping. The fixture's old log-probs come from a near-uniform policy, so on trained weights
   many taken actions are unlikely.

**The row at u1700 cannot be named.** The u1700 weights were not saved, and the canary records neither the row index
nor the weights. At 267.0M and 268.0M the worst rows differ by checkpoint (4, 22, 33, 40; dither: 0, 43, 20). The
first-half rows {0, 1, 20, 23, 28} are the ill-conditioned candidates.

## The canary fix (filed as a TECH_DEBT row, P1 for X35; not built in this unit)

1. **A yardstick arm out of margin is never a disagreement.** If the eager arm or the fp32 reference arm breaks the
   margin, the canary records it as a `yardstick_out_of_margin` event with the row index and the shifts. The event does
   not count toward the confirm / FATAL ladder.
2. **One re-relocation, deterministic (rule 8).**
   - Re-run the rung once at M′ = max(M, 1.25 × the largest YARDSTICK shift) and judge there.
   - The compiled arm's shift never sets the margin, so a miscompile cannot widen its own bar.
   - The forwards do not depend on the old log-probs, so the measured shifts carry over and the precondition is
     categorical at M′.
3. **Record the same-precision forward**: per row, |log π_compiled − log π_eager|. Its max and row go in every canary
   record. It becomes a judged bar only after a CUDA census; CPU reads 0.017–0.028 here.
4. **The instrument gap.** Record the worst row index per arm on every read, and dump the policy `state_dict` on any
   unconfirmed or FATAL verdict. Without them this read could not name the u1700 row.

**What it would have done at u1700:**
- Eager's 0.856 is a yardstick event.
- The rung re-runs at M′ = 1.07. Both arms are inside (0.52, 0.86), and 0 switches were already recorded.
- The gradient rule judges the pair. Its worst ratio read 3.58 < 12 on the original relocation; on the re-relocated
  rows this is **UNVERIFIED**, because the weights are gone.
- Expected verdict: **PASS**. Nothing carries to u1800.

**For a pin that cannot take code**: a `--canary-shift-max` raise on a resume, at the orchestrator's approval (TRAINING_RUN_SOP rule 5).
- 0.22 gives M = 0.88 and covers u1700.
- 0.25–0.30 also covers the CPU dither tail.
- The raise relocates more rows. The loss is read on the golden's own rows at this pin, so the loss statistic is unaffected.

## Files

- `locate_row.py`: the read. Run it with `PYTHONPATH=<pin worktree>/src` and `CUDA_VISIBLE_DEVICES=""`.
- `result_267M.json`: checkpoint_267000182, E32 / Ebf per row, the 12-seed fp32 dither null, and the module bisect.
- `result_268M.json`: checkpoint_268000034, the same reads with a 24-seed null plus Cbf on CPU Inductor.
- `result.json`: the summary, X34's canary history, and what the fix does at u1700.
- Inputs, with sha256: `~/.cache/gen3ai/x34_canary_u1700/` (the checkpoints, the canary records and the golden buffer).

Command, on the agent cores at nice 19:
`mem_cap.sh --blocking … 16 python locate_row.py --zip <ckpt> --out result_<lab>.json --dither N [--compiled]`.
It takes 148 s, or 501 s with `--compiled`.

## Caveats (each a FINDING)

- **CPU vs CUDA.** No CPU read reproduces 0.856. The u1700 weights are gone, and CPU bf16 GEMMs are not CUDA's.
  - The CPU eager shifts track CUDA's family: 0.10–0.13 here, against CUDA's 0.15–0.31 on X34's passing canaries.
  - The magnitude of the u1700 tail is read from the conditioning gauge, not reproduced.
- **CPU Inductor reads the graph-level path only**, not the run's Triton kernels.
- **The 1.25 headroom and the same-precision bar** are proposals with no CUDA census behind them (**UNVERIFIED**).
