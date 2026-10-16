# Compute efficiency: per-component cost of the end state and the toy (2026-10-16)

The cost ledger behind [`../../../endstate/design_compute_efficiency.md`](../../../endstate/design_compute_efficiency.md)
§1. CPU only, read-only on `models/`, run under rule 5b (`taskset -c 5-7,13-15`, `nice -n 19`,
`CUDA_VISIBLE_DEVICES=""`) beside a live X26, at `38d76746`.

| file | what |
|---|---|
| `component_cost.py` | params, counted FLOPs and CPU time per row, forward and backward, attributed per component. The forward is attributed by innermost `record_function` label, the backward by autograd `sequence_nr`. It runs the real micro-step (production), a surrogate step (both nets) and the T2 inference forward |
| `read_cost.py` | rolls the labels up into the doc's components (`RULES`) |
| `iteration_split.py` | the medians of each run's per-iteration rows (`collect_ms`, `update_wall_s`, GPU wait, fps) |
| `result/cost_production.json` | production at 2,048 rows (micro-step + surrogate), inference at 64 and 256 rows |
| `result/cost_toy.json` | `rb_toy_ent_s4002/final_model.zip`, the same reads (surrogate only) |
| `result/iteration_split.json` | X26, EB s2005, toy s4002 and toy s4003 |

Reproduce from the worktree root, with `PYTHONPATH=$PWD/src`:
```
python designs/research_state/measurements/compute_efficiency_2026-10-16/component_cost.py --rows 2048 --reps 3 --infer-rows 64 256 --out <dir>/cost_production.json
python designs/research_state/measurements/compute_efficiency_2026-10-16/component_cost.py --rows 2048 --reps 3 --infer-rows 64 256 --toy models/rb_toy_ent_s4002/final_model.zip --out <dir>/cost_toy.json
python designs/research_state/measurements/compute_efficiency_2026-10-16/read_cost.py <dir>/cost_production.json --section micro_step
```

**Caveats.**
- These are CPU-eager fp32 numbers at ONE thread. The FLOPs and the shares carry over; the absolute seconds carry
  the box's contention.
- The profiler inflates the many-small-op code: the op's kernels issue ≈ 51,000 ops per forward.
- The FLOP counts are the matmul family only (mm / addmm / bmm / SDPA's decomposition).
- The GPU numbers quoted in the doc come from earlier measurements, cited there.
