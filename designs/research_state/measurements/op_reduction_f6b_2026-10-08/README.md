# F6b PRINCIPLED OPERATOR REDUCTIONS — the build's identity, cost and tie reads (2026-10-08)

`--op-reduction {max,principled}` (`gen3_op_reduction_principled_v1`, config v146; architecture audit F6b,
`designs/endstate/design_arch_audit.md` F6). CPU only (no GPU lease: the static-token screen holds it). Every number
here is MEASURED on this box at the build commit's tree (the replay JSONs' `commit` field reads the base `957d4dbd`
because the runs were made from the uncommitted worktree on top of it).

## 1. `max` is byte-identical (`graph_sha.py`)

The production extractor (`designs/production_config.json`, `extractor_compiles_test._build_production_extractor`,
seed 0, perturbed), 8 seeded rows, one thread: the dynamo FX graph text (`gm.code`, one graph), the state_dict bytes
and the eager outputs, at the base tree (`git archive 957d4dbd src`) and at this build.

| config | tree | graph lines | graph sha256[:16] | state_dict | outputs |
|---|---|---|---|---|---|
| production | base `957d4dbd` | 20,162 | `8b3767853c780fb0` | `749c56159ab028f4` | `50f4780aae896b43` |
| production | this build, `max` | 20,162 | `8b3767853c780fb0` | `749c56159ab028f4` | `50f4780aae896b43` |
| production | this build, `principled` | 20,141 | `37650fcd5f1e20c4` | `ca6ec4930722c9c6` | `b314f3a21da9d3b5` |
| bundle (static + move-resolution + speed-physics) | base | 25,079 | `75d9482ce3f1dd93` | `6bad4a0fa841cdd8` | `8d3b3918df087b27` |
| bundle | this build, `max` | 25,079 | `75d9482ce3f1dd93` | `6bad4a0fa841cdd8` | `8d3b3918df087b27` |

The K9 learner golden (`learner_golden_test.py`, the one recorded entry `2.8.0+cu126`) passes unchanged.

**A hazard found on the way:** dynamo names graph nodes after LOCAL VARIABLES. A first draft of `other_move_cells`
bound the same `torch.where` to a new local `tw` and multiplied it — identical ops — and the graph hash moved
(`where_952` → `tw_2`). The `max` expression is kept verbatim (`designs/model/op_contracts.md` "ONE switch").

## 2. Cost (`cost.py` → `cost.json`)

`_build_real_policy` (the SB3 construction path) on the production toggles and on the screen bundle, 64 real rows
(the compile parity fixture), one thread. Matmul FLOPs from `FlopCounterMode`; "element ops" = every other aten op
counted as max(input, output) elements (a rough, declared proxy); wall = eager forward, median of 7.

| | matmul FLOPs / row | element ops / row | extractor params | fwd ms / 64 rows |
|---|---|---|---|---|
| production `max` | 60,304,432 | 2,787,044 | 1,937,942 | 118.87 |
| production `principled` | 60,307,504 (+3,072, +0.005 %) | 2,787,911 (+867, +0.03 %) | 1,938,198 (+256) | 118.82 (noise) |
| bundle `max` | 75,571,048 | 4,536,473 | 1,889,052 | 148.81 |
| bundle `principled` | 75,574,120 (+3,072, +0.004 %) | 4,537,340 (+867, +0.02 %) | 1,889,308 (+256) | 147.95 (noise) |

The +256 parameters are exactly `op_worst_proj` (2 → 128, no bias); the state_dict delta test pins it. The production
learner (2,518,937, ARCHITECTURE §2) becomes 2,519,193 under `principled` — DERIVED from that delta.

## 3. Ties: K9(b)'s excluded share at the same weights (`replay_mode.py` → `replay_{max,principled}.json`)

The `k9_flip_judge_2026-10-08` harness (a seeded 2,048-row complete-game Rust-collector rollout, 16 envs × 128
steps, T2 eager CPU, a seeded random p2, then the REAL probe on every row) at the HEAD reconstruction of the u1480
static-screen dump (`convert_u1480.py`, read-only from `models/rb_st_static_s1001/`), arm `static_fm`, once per mode.
`principled` adds `op_worst_proj` at its init (zeros). Each mode PLAYS its own rollout from the same weights and
seeds, so the row sets differ; the site census is the like-for-like read.

| | excluded before the flip-judge | flip-judged | excluded after | judged max \|Δlog π\| | violations |
|---|---|---|---|---|---|
| `max` (reproduces the harness's HEAD row) | 212 (10.35 %) | 144 | 68 (3.32 %) | 4.3e-6 | 0 |
| `principled` | 162 (7.91 %) | 120 | 42 (2.05 %) | 3.3e-6 | 0 |

What stays excluded:

| site | `max` | `principled` |
|---|---|---|
| `hypothesis_set.py` argsort, not expressed (the typed-HP floor ties) | 45 | 38 |
| `damage_kinds.py` `fixed >= hp`, multi | 15 | 3 |
| `damage_op.py` dominant-move `argmax` (acc / provenance) | 7 | **0 — the site is gone** |
| `hypothesis_set.py` argsort, multi | 1 | 1 |

The structural change is the last-but-one row: the principled forward has no dominant-move argmax, so no row can be
excluded there. The other moves are the two rollouts sampling different states.

## 4. Smokes

`--debug --steps 4608 --arch production --allow-nonproduction-arch --op-reduction principled` (CPU, 223 s) and the
same with `--token-encoding static --move-resolution on --speed-physics on` (265 s): exit 0, the round-trip smoke
PASSED, `[LEARNER FREEZE] released — 4 checks passed`, `Training complete`; `model_config.json` records
`op_reduction: "principled"`, config 146.

## Deferred (GPU; no lease)

Compile parity of the `principled` forward and backward (the noisy-OR's `prod` backward under Inductor is UNVERIFIED);
the T2 inference service's graph build on CUDA; a real launch's first two minutes; the GPU cost read (update wall,
`train_ms`).
