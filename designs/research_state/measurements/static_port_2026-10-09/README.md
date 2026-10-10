# The static port at HEAD: cost, production identity, the CPU smoke (2026-10-09)

The build: `gen3_static_port_v1`, config v147 (`designs/endstate/design_static_tokens.md` §12). The identity proof
against the screen's pin is its own folder, [`../static_port_identity_2026-10-09/`](../static_port_identity_2026-10-09/README.md).
CPU only (the screen holds the GPU lease).

## 1. Production is byte-identical (the F6b unit's `graph_sha.py` method)

The production extractor (`extractor_compiles_test._build_production_extractor`, seed 0, perturbed), 8 seeded rows,
one thread: the dynamo FX graph text, the state_dict bytes and the outputs, at the parent `2e357971` and at the port.

| tree | graph lines | graph sha256[:16] | state_dict sha256[:16] | outputs sha256[:16] | params |
|---|---|---|---|---|---|
| parent `2e357971` | 20,162 | `8b3767853c780fb0` | `749c56159ab028f4` | `50f4780aae896b43` | 1,937,942 |
| the port | 20,162 | `8b3767853c780fb0` | `749c56159ab028f4` | `50f4780aae896b43` | 1,937,942 |

The graph equals the F6b unit's recorded `8b376785…`. The one production-path code change, `DamageOperator.spikes_entry`
(the Spikes entry rule factored out of `pairwise_entry`), is the same ops under the same local names. The K9 learner
golden passes unchanged (`learner_golden_test.py`, in the routine gate).

## 2. Cost (`cost.py` → `cost.json`)

The production toggles through a REAL SB3 build (`identity_init_test._build_real_policy`, whose policy MLP is the
test harness's, so the extractor column is the comparable one), the compile parity fixture's 64 real rows, matmul FLOPs
by `FlopCounterMode` (grad enabled), the eager forward wall the median of 7 at one thread (beside other load: wall is
within noise, not a measurement).

| config | extractor params | matmul FLOP / row |
|---|---|---|
| legacy (production), both trees | 1,937,942 | 60,304,432 |
| `static`, parent | 1,888,068 | 60,214,704 |
| `static`, port | 1,885,060 (−3,008) | 60,088,752 (−0.21 %) |
| + `--mon-hazard-cost on` | 1,885,316 (+256) | 60,094,896 (+6,144) |
| + `--move-actor-state on` | 1,886,084 (+1,024) | 60,090,800 (+2,048) |
| + both | 1,886,340 (+1,280) | 60,096,944 (+8,192) |
| the bundle (static, move resolution, speed physics, value-threat-inject off, principled op, obs facts v1), parent | 1,893,788 | 75,521,896 |
| the bundle, port, no facts | 1,890,780 | 75,395,944 |
| the bundle, port, both facts | 1,892,060 | 75,404,136 |

The −3,008 is the type SUM (S's first Linear loses one 16-wide block: −16 × 256 = −4,096) and the outgoing set function
(`Linear(6, 32)` + `Linear(32, 128)`, both bias-free, 4,288, against `Linear(24, 128)` + bias, 3,200: +1,088). On the
production surface's K9 learner (`learner_golden.build_learner`, `--arch production`), the static policy has 2,466,055
parameters, 2,467,335 with both facts, 2,472,287 for the bundle with both facts.

```bash
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
$PY -E designs/research_state/measurements/static_port_2026-10-09/cost.py --src <checkout>/src [--parent]
```

## 3. The CPU `--debug` smoke (static + both facts + the full bundle)

```
--arch production --debug --steps 10000 --token-encoding static --mon-hazard-cost on --move-actor-state on
--move-resolution on --speed-physics on --no-value-threat-inject --op-reduction principled --obs-facts v1
```

launched through a `runpy` wrapper (the trainer's file name in no argv; `PYTHONPATH` set to this checkout's `src` inside
the wrapper so every child imports it), `GEN3AI_MODELS_DIR` a scratch directory, CUDA hidden, beside a 3-worker test run.

**EXIT 0** (2026-10-09, ~13 min wall on CPU beside a 3-worker test run): `🧪 [DEBUG SHAPE]` auto-shaped the update, `🎯 [CRITIC] winprob`, `🦀 [ENV CORE] rust`, `[ModelVersion] Round-trip smoke test PASSED`, five per-update tables with a `behaviour/` block, `🧊 [LEARNER FREEZE] released — learn() returned; 10 checks passed.`, `Training complete`. The run's `model_config.json` records config 147, `token_encoding` static, `mon_hazard_cost` / `move_actor_state` on, `move_resolution` / `speed_physics` on, `op_reduction` principled, `obs_facts` v1, `value_threat_inject` false. `--debug` is CPU, eager T2, no compile: the compiled / CUDA layer is NOT exercised here. It was run on 2026-10-09 inside R / E (`../gpu_checks_endstate_2026-10-09/`): R (static + both facts + the recovery levers) PASSES; E (+ the bundle) FAILED the R1 startup gate at `43a59bbd`, and the same offline probe reads clean at `c0f528b4`.

## 4. The observation

Unchanged: both facts read existing columns (`global_env`'s Spikes layers, the per-mon type / ability / ability-known /
species columns, HP and the status one-hot), so the Rust encoder benchmark (`python -m
agents.observation.rust_encoder_benchmark`) does not apply.
