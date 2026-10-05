# F-XC-4: the fixed_mass compile gate's `0.00e+00` and `cosine 0.000000` (2026-10-05)

**Verdict: (a) and (b) REJECTED; (c) holds.** The gate's two arms are independent, and R1 under
`--belief-tokens fixed_mass` is one compiled graph. The "cosine 0.000000" FATAL was a **NON-FINITE
compiled backward**, which `compile_trainer._cos` read as an orthogonal gradient. The cause is a torch
2.8 Inductor defect on the fixed_mass graph, inside the joint-graph pass `remove_noop_ops`: removing its
first 1,973 no-ops leaves the compiled gradient finite, and the 1,974th removal (an `aten.alias` of
`DamageOperator._p_outspeed`'s sigmoid, in the D2 edge family's kernel) makes it NaN on 41 parameters.
That alias is not the whole trigger: keeping EVERY alias (5,140 other removals made) is still NaN, so the
defect is fusion-dependent. The bit-equal loss is the forward's own arithmetic. Tags: **MEASURED**
unless marked.

- **Box / commit.** RTX 3080 Ti, torch 2.8.0+cu126, fp32 'highest', desktop stopped. HEAD `e8008c2d`;
  the gate fix in this unit changes no arithmetic, so the readings hold for the fix commit too.
- **Learner.** The K9 fixed_mass golden learner (`learner_golden.build_arm_learner("fixed_mass")`:
  production + the lever), CUDA. "fresh" = the golden builder's perturbation set to 0 (`PERTURB_SCALE`),
  i.e. the zero-init pointer head (`weights_regime` reads `fresh`); the gate's own perturbed rung is the
  trained-like state. The rows are R1's gate rows (`compile_regions.r1_batch`, the K9 golden buffer), B = 2,048.
  The X26 ride-along heads are not built here (R1 judges the policy's own parameters; the heads are
  outside PPO's graph).

## Evidence

| question | measurement | reading | file |
|---|---|---|---|
| (a) are the two sides one object? | the installed dispatcher's inner callable is dynamo's wrapper of `micro_step` (`_torchdynamo_orig_callable is micro_step`), not `micro_step`; the two losses live in different storage; the gradients are NOT bit-equal | **no** | `results/gate_fm_fresh.txt` |
| (b) does R1 silently not compile? | dynamo `unique_graphs` 1, `graph_break` {}, `calls_captured` 21,318, AOT `ok` 1; `assert_inventory`: one cache entry on `micro_step`; the dispatcher's eager-body guard silent | **no — one compiled graph** | `results/gate_fm_fresh.txt`, `results/terms_fm_cuda.json` |
| what is the cosine 0? | compiled gradient NaN on **41 parameters** (the five embeddings, every `pokemon_encoder` parameter, `move_belief.{move_head, reinject, norm}`, `spread_belief.{nature_head, ev_head}`, `hp_type_belief_head.*`); eager finite. `_cos`: a NaN norm fails `na > 0` and returned 0.0 | **a non-finite compiled backward** | `results/nan_inductor.json` |
| reproducible? | every rep in a process (3 live + 3 perturbed-rung reps), every unmodified compile at HEAD (8 of 8 processes), `split_reductions=False` too | deterministic at HEAD | `results/gate_fm_fresh.txt`, `results/nan_nosplit.json` |
| which compiler layer? | `eager`, `aot_eager`, `aot_eager_decomp_partition`, `…_crossref` finite; `inductor` NaN. `CompilerBisector` → `inductor` / `joint_graph_passes` / `remove_noop_ops` | **Inductor, not the decompositions** | `results/compiler_bisector.txt`, `results/nan_aot_eager.json`, `results/nan_decomp.json` |
| which node? | `remove_noop_ops` re-implemented with a removal budget (`scripts/noop_bisect.py`; FX/AOT caches off): 5,809 removals in all; 1,973 allowed → finite, 1,974 → NaN (a binary search over 14 compiles, every reading consistent with one threshold). Removal #1,974 = `alias_175 = aten.alias(sigmoid_277)`, the sigmoid of `DamageOperator._p_outspeed` (`damage_op.py:652`), called from `_outgoing_attacker_matrix` (`damage_op_blocks.py:586`) ← `pairwise_bench_outgoing` (`damage_op_pairwise.py:813`, the D2 edge family) | **the first tripping removal** (not the only one: `results/alias_skip_gate.txt` — the real gate with every `aten.alias` removal skipped is still NaN on the same 41) | `results/noop_bisect.txt`, `results/noop_culprit.json` |
| where does the NaN surface? | a scalar probe on each PokemonEncoder pass's output: the REAL pass's output gradient is NaN, the hypothesis pass's finite, δ_θ's input finite. (Probes INSIDE the damage op move the graph enough that the NaN vanishes — a Heisenbug, so the damage-op-level probe reads are not banked as evidence.) | the gradient reaching the real encoder output | `results/encoder_output_probes.txt` |
| CPU? | CPU Inductor: finite; loss 1 ulp apart (6.0e-8) | CUDA-only | `results/terms_fm_cpu.json` |
| the `0.00e+00` | the compiled and eager LOSS are bit-equal in some runs (every perturbed-rung rep; the `nan_probe` runs) and 1–2 ulp apart in others (live rep: 0.82493532 vs 0.82493544). R1's loss is ONE fp32 scalar; its terms differ by 2e-10 to 6e-8 each (`terms_fm_cuda.json`) and can round to the same sum. The gate line's "train features" for R1 is that same loss (`_r1_verdict` passes `eager["loss"]` as `features`), so "loss and features both 0" was one observation, not two | the forward's own arithmetic, not shared state | `results/terms_fm_cuda.json` |
| can the gate see a planted difference under fixed_mass? | CPU (dynamo `eager` backend), the fixed_mass learner: a 1e-3 loss plant → `LOSS disagrees`; a 10 % gradient plant on `pokemon_encoder.move_network.0.weight` → per-parameter FATAL naming it; a NaN plant → `NonFiniteGateArmError` naming it. On CUDA the NaN itself was the planted-in-the-wild difference the gate caught | **yes** | `src/agents/model/compile_regions_independence_test.py` |

## Consequences

- **A fixed_mass `--compile-trainer` launch at HEAD is REFUSED at its startup gate** (now
  `NonFiniteGateArmError`, naming the 41 parameters). The cost ablation at `889add9d` saw 1 of 4 such
  launches FATAL and 3 pass (per-parameter 1.1e-5): either the NaN was compile-dependent then, or U4
  (`350fb83b`, after `889add9d`) changed the graph so it now trips every time. **UNVERIFIED** (not
  re-run at `889add9d`).
- **fixed_mass's +16.7 % `train_ms` is a COMPILED cost, not an eager fallback** (the fmB / fmA launches
  that passed ran one compiled graph; nothing ran eager). (b) explains none of it.
- **What compiling it would take** (not done here: each changes compiled arithmetic). Keeping the aliases
  is NOT enough (measured: still NaN). Candidates, in order of evidence: (1) `remove_noop_ops` off for
  R1 (finite at budget 0, `results/noop_bisect.txt`; a global Inductor pass change: every arm's
  generated kernels and speed move, so it needs every arm's gate + the learner benchmark); (2) find the
  fused kernel that produces the first NaN (`TORCH_COMPILE_DEBUG` output code + a per-buffer NaN check —
  `nan_asserts` cannot be used directly: the forward holds legitimate `-inf` masks) and respell the
  model op that feeds it; (3) a newer torch, with an upstream report. The mechanism (a saved buffer
  reused in place is the usual one) is **UNVERIFIED**; adding scalar probes inside the damage op moved
  the graph enough that the NaN vanished, so it is fusion-dependent.
- **The fix verified on CUDA:** the alias-skip run above went through the REAL gate with this unit's code
  and FATAL'd `NonFiniteGateArmError … 41 of 257 parameter(s) (e.g. …species_embedding.weight …)`.

## The fix in this unit (gate class, no arithmetic)

- `gen3_gate_nonfinite_named_v1`: `compile_trainer.train_verdict` refuses a non-finite gradient on
  either arm FIRST (`NonFiniteGateArmError`, named, compiled arm first); `_cos` returns NaN, not 0.0.
- `gen3_gate_independent_arms_v1`: `compile_regions._r1_pair` (every R1 rung of the gate and the
  canary) reads `region_calls`' counters:
  the compiled arm dispatches the compiled route once and runs R1's Python body zero times; the eager
  arm the reverse. Else a typed refusal.
- Tests that fail on revert: `compile_regions_independence_test.py` (6 of its 9 fail at HEAD's code:
  the NaN naming, the canary's confirmed naming, `_cos`, and the three independence refusals — one of
  them a gate that compared eager with eager and PASSED).

## F-XC-5 on the GPU: K9(b)'s excluded share, fixed_mass, regime A, at HEAD (`fxc5/`)

**Under the 0.15 ceiling: 0.090–0.113 on updates 2–5, 0 on update 1; max |Δ log π| exactly 0 on every
judged and excluded row.** At `889add9d` regime A read 0.54–0.58 (`x5_cost_ablation_2026-10-04` §1 item 4).

| update (step) | excluded_frac | rows excluded / judged (current-version rows) |
|---|---|---|
| 1 (207,257) | 0.000 | 0 / 2,048 (2,048) — selection-free: the scorers are still zero-init (`gen3_behaviour_tie_identity_v1`) |
| 2 (306,143) | 0.113 | 116 / 908 (1,024) |
| 3 (402,925) | 0.094 | 96 / 928 (1,024) |
| 4 (502,573) | 0.090 | 92 / 932 (1,024) |
| 5 (516,719) | 0.107 | 110 / 914 (1,024) |

- **Setup.** The cost ablation's scripts and argv (`--arch production`, the X26 heads, `--belief-tokens
  fixed_mass`, eval off), regime A = a seeded 20-snapshot pool (`build_pool.py`), self-play from the first
  rollout (`rust_env/p2_policy_decisions` 91.7k–95.3k of ~98k per rollout), `--behaviour-check warn` (no
  violation fired). Measurement-only overrides: `PROF_FIT_HEADROOM_MIB=256`, `PROF_SLOT_LOAD_TOL_MIB=8`
  (F-XC-2 / F-XC-3 stand) and a NEW one, `PROF_LOG_KEY_MAX=80` (below).
- **DEVIATIONS (each limits the read).** (1) `--no-compile-trainer`: a fixed_mass compile gate FATALs at
  HEAD (above), so the learner's side of K9(b) is EAGER here (production compares T2 with the compiled
  learner). (2) The trainee is FRESH and the pool is fmB5's checkpoint after ONE update, perturbed — not a
  1.2M-step regime-B checkpoint; four informative updates, not a steady state. (3) 5 updates, one launch.
- **NEW FINDING (a launch blocker):** the first regime-A launch (fmA5) CRASHED at its first logger dump:
  `ValueError: Key '   flat_switch_target_recall_top1_pool' truncated to '   flat_switch_target_recall_top1...'
  that already exists` (`fxc5/fmA5_logger_crash.txt`). The stdout table (`train_logger.HumanOutputFormat`,
  36 characters) truncates U4's per-class `flat_switch_target_recall_top1_{bot,pool}`
  (`instrumented_ppo/flat_intent_fold.py:75`) to one key once both classes hold rows — i.e. **every
  fixed_mass self-play run dies at its first dump after pool rows exist**. Regime B (bots only) does not
  trip it. fmA5b ran with the table widened in the wrapper only (TB keys untouched). Not fixed here (out of
  scope).
- GPU time: 51 minutes (two launches plus the retry), over the 45-minute box by the retry.

## Reproduce

`scripts/gpu_run.sh <GB> <timeout_s> <log> <script> [args]` (the lease token, `scripts/ops/gpu_lock.sh`,
`scripts/ops/mem_cap.sh`; written for this unit's worktree path — edit `W`).

```bash
gpu_run.sh 24 1500 gate.log   scripts/gate.py fixed_mass fresh 3          # identities, counters, NaN per rep, the real gate
gpu_run.sh 24 1500 nan.log    scripts/nan_probe.py fixed_mass fresh inductor out.json   # + aot_eager / aot_eager_decomp_partition
gpu_run.sh 28 3300 bis.log    scripts/cbisect.py                          # CompilerBisector
gpu_run.sh 28 3300 noop.log   scripts/noop_bisect.py culprit.json         # the node
```

`scripts/terms.py` (per-term compiled vs eager, `cpu` or `cuda`) and `scripts/probe_modules.py` (scalar
probes on module / method inputs and outputs) are the localisation tools.
