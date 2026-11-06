# X65-F1: a production trainer beside the Kakuna-lane search workload (2026-11-05)

The question (FINDING X65-F1, `designs/endstate/design_x65_exit0_search_distillation.md`): can CPU search work on
the agent cores (5-7, 13-15) overlap a live production training run, and at what cost to each?

## Verdict

**Overlap is viable.**
- The search workload loses about **13 % of its throughput** while the trainer runs.
- The pinned trainer loses nothing measurable.
- The read is **not confounded**: no foreign process ran on the agent cores during the two windows that are compared.

| meter | with trainer ÷ without | how it is measured |
|---|---|---|
| **per-decision CPU time** (primary) | **1.153** [1.148, 1.158] | the same 78 states, paired by state; 95 % bootstrap over states |
| ↳ no-trainer null (A′ ÷ A) | 1.013 [1.007, 1.018] | the two clean baselines against each other |
| decisions per minute | 0.891 | B2 ÷ the pooled clean baselines (A clean, A′) |
| ↳ against each baseline | 0.951 (A clean) / 0.856 (A′) | the two baselines differ by 10 % in rate (A ÷ A′ = 0.90) |
| trainer steps/s | **1,405** pinned and under load, against X26's 1,360 | checkpoint 281,000,152 → 282,000,049 in 711.85 s; X26 is the median of 1,152 rollout rows in its launcher log (p10 1,344, p90 1,378) |

**The primary meter.** It is the per-decision CPU time, paired by state. The rate meter is noisier: each worker
restarts its process every 78 decisions, and the first window counts its start-up. That is why the two no-trainer
baselines agree within 1.3 % on CPU time but differ by 10 % on rate.

**Throughput factor.** A 1.153× CPU cost per decision means about 1/1.153 ≈ **0.87× the throughput**. The pooled
rate ratio, 0.89, agrees.

**Why the search slows.** The trainer sits on other physical cores. CPUs n and n+8 are SMT siblings, so 0-4,8-12
and 5-7,13-15 are disjoint physical cores. The slowdown therefore comes through shared resources such as the L3
cache, memory bandwidth and the package clock, not through SMT.

**Search output was unaffected.** All 2,951 decisions in the three compared spans reproduced the banked search
exactly (`mismatches` = 0).

## Design

**The workload** (`group_v1.sh`, under a 24 GB `scripts/ops/mem_cap.sh` scope):
- 6 single-threaded loops at nice 19, one per agent CPU.
- Each loop is `main.value_search.live_identity bank-tooth`, which re-runs S′ (`r0s4_rolls@10`, P* =
  `rb_x26_s2601` @ 280,000,170) on the 78 scored states of the dlive hashed sample, pass after pass.
- Engine: `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`.
- Its output went to `~/gen3ai_archive/contention_2026-11-05/`. It feeds no read.

**The trainer:**
- A production-architecture fork of P*: the X65 control argv with the scratch name `rb_x65f1_contention_s6599`,
  seed 6599, `--steps 285000170`, `--checkpoint-every-steps 1000000`.
- `GEN3AI_MODELS_DIR` pointed at a scratch directory outside `models/`.
- Launched under `taskset -c 0-4,8-12` at nice 0.
- **The pin held.** At 90 s, every one of the trainer's 35 processes and 118 threads had `Cpus_allowed_list`
  = 0-4,8-12. The only process off the pin was the `nohup` bash wrapper that started it. No child escaped.
- The fork was stopped by its PID at 19:13:36. Its scratch run directory was then deleted. It wrote nothing to
  `models/`.

**Foreign load** (`foreign_sampler.py` → `foreign.tsv`, `foreign.top.jsonl`):
- Every 60 s it records the CPU-seconds that processes outside the workload's `x65f1` scopes spent on the agent
  CPUs, plus each agent CPU's busy fraction from `/proc/stat`.
- A process counts as being on an agent CPU by the CPU it last ran on. That is an approximation at a 60 s grain.

## Windows

From `summary.json`. All times are PST on 11-05.

| window | span | trainer | rows | rows/min | foreign (mean / max share of the 6 CPUs) | load avg |
|---|---|---|---|---|---|---|
| A clean | 17:56:24 – 18:16:24 | none | 756 | 37.8 | 0.1 % / 0.8 % (sampler from 17:59; 18 of 20 min covered) | 7.3 |
| A tail, **CONFOUNDED, unused** | 18:16:24 – 18:26:24 | none | 189 | 18.9 | 16.9 % / 24.9 % | 13.3 |
| B1, **CONFOUNDED, aborted** | 18:33 – 18:38 | warm-up | — | — | 15–28 % | — |
| B2 warm-up, unused | 18:43:21 – 18:47:21 | pre-`learn()` | 186 | 46.5 | 0.0 % | 9.0 |
| **B2 training** | 18:47:21 – 19:13:22 | in `learn()` from 18:47:12 | 935 | 35.9 | 0.0 % / 0.0 % (26 of 26 min covered) | 8.1 |
| **A′ clean** | 19:14:13 – 19:44:13 | none (stopped 19:14:09) | 1,260 | 42.0 | 0.0 % / 0.0 % (24 of 30 min covered; the sampler's 6,000 s ended 19:38) | 6.3 |

**Two confounds**, both from builder agents doing CPU work on the agent cores:
- **A's tail:** a slow `pytest` plus 4 multiprocessing spawn workers.
- **B1:** a routine-gate `pytest -n 3` with 3 xdist workers.

The orchestrator held the builders off the agent cores from 18:30 until 19:45. Every window used here started only
after 5 consecutive sampler minutes under 5 % foreign load.

## Caveats

**The trainer has no in-session control.**
- It never ran without the workload in this session, so its steps/s is compared with X26's recorded rate.
- X26 was unpinned and ran at another point in training, and its agent-core load is unknown.
- The fork matches it at the same pool size (20 snapshots, the cap) and on the same card.
- The fork's logged fps has median 1,466 (n = 21, min 1,388). Its checkpoint-to-checkpoint rate, 1,405, is the
  figure quoted.

**The time spans are approximate.** A row's place in time comes from its worker's cumulative `cpu_s`. A span is
selected by the row counts at its wall-clock ends, as described in `analyze.py`.

**This was one trainer, 26 minutes, one workload.** The shared-resource cost may differ for a search with a
different memory footprint.

**A run-health note.** The fork's `--behaviour-check warn` logged 2 behaviour violations
(`fork_behaviour_violations.jsonl`): max 1.22e-4 and 1.54e-4 against a bar of 1e-4, with p99 about 2e-5. They are
the same class its parent `rb_x26_s2601` shows. This is a property of the run, not of the contention.

## Files

- `analyze.py`: produces `summary.json` from the files here (`python analyze.py`).
- `spans.json`: the windows.
- `counts_<phase>.tsv`: the workload's per-minute counts.
- `rows_<phase>.jsonl.gz`: every decision's `cpu_s` and state key, with its pass file.
- `foreign.tsv`, `foreign.top.jsonl`: the foreign-load series.
- `fork_fps_series.txt`, `fork_ckpts.txt`: the trainer's throughput evidence.
- `group_v1.sh`, `foreign_sampler.py`: the workload and the sampler, as run.
