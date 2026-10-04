# The two-architecture head-to-head engine: the identity and determinism proof (F-U6-1, 2026-10-04)

**What this proves.** `main.h2h` (`play` and `play-many`) now serves cells whose player and opponent are TWO
architectures — the X5 A/B's cross, an X5 `fixed_mass` seed against a `blob` seed (`designs/endstate/design_x5_belief_tokens.md`
§7.4). One engine declares one T2 slot group per architecture (`src/main/h2h/arch.py`, `declare_engine`). Each group holds
only the slots its cells need (a player's eval slot, an opponent's sentinel slot), and the engine builds one eval core per
(player group, opponent group) pair the cells use. Two claims are checked here on REAL production (`blob`) checkpoints, the
X5 P0 sizing finals (`models/sizing_A2_n48_e10_s1001`, `sizing_Ap_n48_e10_s1002`, `sizing_B_n256_e10_s1001`, final
snapshots, read-only), each next to a `fixed_mass` checkpoint:

1. **A same-architecture cell is unchanged on a two-group engine.** U6's four blob cells play the games they play on
   single-cell `play`'s single-group engine.
2. **A cross cell is deterministic.** Re-run on the same seeds, it plays the same games, even on an engine whose groups
   are declared in the opposite order.

The `fixed_mass` checkpoint is **seeded PERTURBED-fresh**, because no trained `fixed_mass` run exists yet. It uses the test
fixture's recipe: production surface plus `--belief-tokens fixed_mass`, seed 7, perturb seed 2700, built at one thread.
Everything ran CPU only (T2 eager, 64 envs, 4 core threads, 4 torch threads) under `scripts/ops/mem_cap.sh` (peak 1.13 GB).
Nothing under `models/` was written, and the ledger roots were temporary.

**Plays.**
- (1) ONE `play_cells` call over U6's four blob cells, then `fixed_mass` vs A2 and `fixed_mass` vs B. This engine has two
  groups: group 0 is `blob` (a player and an opponent slot) and group 1 is `fixed_mass` (a player slot). It runs two eval
  cores, combos `(0,0)` and `(1,0)`.
- (2) The two cross cells alone, on a fresh ledger. Here group 0 is `fixed_mass` (a player slot) and group 1 is `blob`
  (an opponent slot), with one core, combo `(0,1)`.
- Both plays use 100 mirrored pairs per cell, in 2 batches of 50, schedule seed 0. These are U6's seeds, so U6's committed
  single-cell rows are the reference.

## Result (MEASURED; `compare.json`, exit 0)

| check | result |
|---|---|
| (a) every blob row of (1) vs U6's single-cell `play_edge` rows (`../h2h_multicell_2026-10-04/rows.json.gz`): counts, pentanomial, team counters, seed block, outcome digest | **equal, 8 / 8 rows** |
| (b) each blob row's new `compute.outcome_digest_all` (EVERY game, near-ties included) vs U6's engine-level `digest_all` of the same cell and batch on a FRESH single-group engine (`games.json.gz`) | **equal, 8 / 8** (800 games) |
| (c) the cross rows of (1) vs (2): counts, pentanomial, team counters, seed block, regime, outcome digest, `outcome_digest_all`, near-tie list, both decision counts, both near-tie decision counts | **equal, 4 / 4 rows** (400 games) |

The two kinds of game differ in how many near-ties they hold:
- **Blob games:** 126 of 800 have a decision inside the GPU near-tie bar (2e-3).
- **Cross games:** 319 of 400 do, because a perturbed-fresh policy's log-probs are nearly flat.

The row's `outcome_digest` leaves these games out by design (standing rule 8), but `outcome_digest_all` includes them.
`outcome_digest_all` matches anyway, because the forward is deterministic on one device, one backend and one batch shape.

**Limits, stated.**
- **The cross outcomes are lopsided.** The perturbed-fresh `fixed_mass` policy loses 399 of 400 games to the trained finals,
  so check (c)'s W/L/D vector carries little information. The end turns that `outcome_digest_all` hashes, and the
  decision counts, do vary. A balanced cross (two perturbed-fresh policies) is compared game for game in pytest
  (`src/main/h2h/play_cross_integration_test.py`).
- **(c) is a same-process, same-device replay.** A replay in another process, or on the GPU, can flip a near-tie decision
  (F-U6). That is why the row digest lists those games by index instead of hashing them.

## Cost (CPU, this box, beside other agents)

| quantity | value |
|---|---|
| pre-flight (every side sorted into the two groups, 4 distinct checkpoints loaded once) | 3.1 s |
| engine start, two groups (3 slots) + two eval cores | 6.8 s (one group, U6: 4.7 – 5.5 s) |
| engine start, the cross alone (two one-slot groups, one core) | 5.3 s |
| swap to a cross cell (`set_cell`) | 0.22 s |
| slot loads per cross cell (2 batches × 2 loads, copy-checked and parity-gated) | 1.4 s |
| play, 200 games per cross cell | 29 – 34 s |

## GPU memory, ESTIMATED (not measured: CPU-only agent)

- **Weights per slot** (`SlotGroup.storage_bytes`, unique tensors) are **11.7 MiB for `blob`** and **12.0 MiB for
  `fixed_mass`**. On CUDA, each slot is its own lane (`lanes = min(n_slots, 8)`), and a lane's private graph pool is sized
  by its largest capture. At 64 envs the buckets are (8, 64). Measured 2026-10-01 for production `blob`, a 64-row pool is
  about 60 MiB. The `fixed_mass` pool is unmeasured.
- **The X5 cross declares TWO slots:** a `fixed_mass` player slot and a `blob` opponent slot. That is as many as today's
  single-architecture engine, so it costs about 24 MiB of weights, 2 lanes and about 120–160 MiB of graph pools.
  This is roughly today's engine, plus one more architecture's compiled kernels.
- **The worst case**, where both architectures play both roles, is 4 slots: about 50 MiB of weights and about
  250–330 MiB of pools.
- Either case fits beside the CUDA context with large headroom. The engine runs only when no training run holds the GPU
  lease (design_evaluation.md §7, the offline engine).
- **Engine start on the GPU** compiles and captures per architecture. Expect up to ~2× F-P0-4's 114–167 s, paid once per
  look, not per cell.

## DEFERRED to the GPU owner (exact commands, from the repo root, `PYTHONPATH=$PWD/src`, `gen3ai_torch28`)

```bash
M=/home/goodlad/dev/gen3ai/models
# (0) a fixed_mass checkpoint to play (until X5 arms exist): the proof's recipe, into a scratch dir outside models/
python - <<'P'
import sys; sys.path.insert(0, "designs/research_state/measurements/h2h_cross_2026-10-04")
from pathlib import Path; import cross_proof as C
print(C.build_fixed_mass(Path("/tmp/h2h_gpu_x5/run_x5_fixed_mass_fresh")))
P
FM=/tmp/h2h_gpu_x5/run_x5_fixed_mass_fresh/snapshot_000000007000.zip
# (a) GPU memory + startup of the CROSS engine (two one-slot groups): read engine.startup_s and the peak
scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24 python -m main.h2h play-many --players $FM \
  --opponents $M/sizing_A2_n48_e10_s1001/final_model.zip $M/sizing_B_n256_e10_s1001/final_model.zip \
  --pairs 1000 --batch-pairs 500 --device cuda --out /tmp/h2h_gpu_cross --label fu61_gpu_check
#   while it runs: nvidia-smi --query-compute-apps=pid,used_memory --format=csv -l 5
# (b) GPU identity: a blob cell on a two-group engine vs single-cell play, then compare the rows' outcome digests
cat > /tmp/h2h_cells_x.json <<J
[["$M/sizing_B_n256_e10_s1001/final_model.zip", "$M/sizing_A2_n48_e10_s1001/final_model.zip"],
 ["$FM", "$M/sizing_A2_n48_e10_s1001/final_model.zip"]]
J
scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24 python -m main.h2h play-many --cells /tmp/h2h_cells_x.json \
  --pairs 1000 --batch-pairs 500 --device cuda --out /tmp/h2h_gpu_twogroup --label fu61_gpu_check
scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24 python -m main.h2h play \
  --player $M/sizing_B_n256_e10_s1001/final_model.zip --opponent $M/sizing_A2_n48_e10_s1001/final_model.zip \
  --pairs 1000 --batch-pairs 500 --device cuda --out /tmp/h2h_gpu_single1 --label fu61_gpu_check
python -m main.h2h read /tmp/h2h_gpu_twogroup --json > /tmp/a.json; python -m main.h2h read /tmp/h2h_gpu_single1 --json > /tmp/b.json
```

## Files

| file | what |
|---|---|
| `cross_proof.py` | `play` (the two `play_cells` calls) and `compare` (checks a–c against U6's committed dumps) |
| `run.sh` | the driver; runs from the repo root, the only cwd `main.h2h` accepts |
| `rows.json.gz` | both plays' rows and engine blocks |
| `compare.json` | the verdict and timings; `run.log` holds the run's lines |

Reproduce with `PYTHONPATH=<tree>/src ./run.sh` (about 6 min on the CPU).
