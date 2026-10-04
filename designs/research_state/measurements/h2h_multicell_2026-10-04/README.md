# The multi-cell head-to-head engine: the digest proof and the CPU per-cell cost (eval U6 / X5 U0, 2026-10-04)

**What this proves.** `python -m main.h2h play-many` (`src/main/h2h/many.py`) plays many cells (player vs opponent) on ONE
engine, swapping the weights in the two declared T2 slots per cell. The claim is **storage and engine reuse only**: every
game of every cell is the game single-cell `main.h2h play` plays on the same seeds. Checked here on REAL production-architecture
checkpoints: the X5 P0 sizing finals (`models/sizing_A2_n48_e10_s1001`, `sizing_Ap_n48_e10_s1002`, `sizing_B_n256_e10_s1001`,
final snapshots, read-only). CPU only (T2 eager, 64 envs, 4 core threads, 4 torch threads), under `scripts/ops/mem_cap.sh`
(peak 1.0 GB). Nothing under `models/` written; the ledger roots were temporary.

**Cells.** A2 vs A′, A′ vs B, B vs A2, A′ vs A2 (the last is the reverse of the first: the same schedule key, so the same team
pairs). 100 mirrored pairs per cell in 2 batches of 50, schedule seed 0.

## Result (MEASURED; `compare.json`, exit 0)

| check | result |
|---|---|
| (1) game logs, FRESH engine per cell vs ONE engine swapped cell to cell (`H2HEngine.set_cell`), every field the executor's trimmed log keeps (result, teams, swapped, end turn, winner, forfeit, near-tie counts) | **identical, 800 / 800 games**, every cell and batch |
| (2) outcome digest (games clear of the 2e-3 near-tie bar) and `digest_all` (EVERY game, near-ties included) | **equal**, all 8 batches |
| (3) single-cell `play_edge` rows vs `play_cells` rows: counts, pentanomial, team counters, seed block, outcome digest | **equal**, 8 / 8 rows |
| (4) each row's outcome digest = the engine-level games' digest of that cell and batch | **equal**, 8 / 8 |

126 of the 800 games hold a decision inside the GPU near-tie bar (2e-3); they are excluded from the row digest by
design (standing rule 8), and INCLUDED in checks (1) and `digest_all` — which match anyway, because on one device, one
backend and one batch shape the forward is deterministic.

**A script defect, fixed after the run (stated, not hidden).** The first `compare` keyed the engine-level digests by the
cycle seed; A2-vs-A′ and A′-vs-A2 share it (the schedule key is order-independent), so check (4) compared the A′-vs-A2
row against the A2-vs-A′ games and reported 2 "problems". The key is now (player, opponent, batch) and the same dumped
data compares clean; checks (1)–(3) were clean in both versions.

## The CPU per-cell cost (MEASURED, this box, load average ~4–5 beside other agents)

| quantity | value |
|---|---|
| engine start (fresh, per cell in the single-cell path) | 4.7 – 5.5 s (T2 eager on the CPU: no compile) |
| pre-flight (every side's architecture checked, 3 distinct checkpoints loaded once) | 2.4 s, once per call |
| swap to a new cell (`set_cell`: host loads + the checks) | **0.27 s** with both checkpoints cached; **1.1 s** when one is loaded for the first time |
| slot loads per cell (2 batches × 2 `InferenceService.load`, each copy-checked and parity-gated at every bucket) | 0.95 s — per BATCH in both paths, unchanged |
| play, 200 games per cell | 34 – 41 s (4.9 – 5.8 games/s) |
| wall, 4 cells: single-cell × 4 vs `play-many` | 170.7 s vs 159.6 s |

So on the CPU the per-cell overhead after the first cell is **~0.3–1.1 s instead of ~5 s**; the saving that matters is on
the GPU, where an engine start is 114–167 s of graph compile and capture (F-P0-4). **The GPU per-cell time is NOT measured
here (CPU-only agent): DEFERRED, see below.**

## DEFERRED to the GPU owner (exact commands, from the repo root, `PYTHONPATH=$PWD/src`, `gen3ai_torch28`)

```bash
# (a) the GPU per-cell cost and the GPU digest identity: the same 4 cells on T2 graph
M=/home/goodlad/dev/gen3ai/models
cat > /tmp/h2h_cells.json <<J
[["$M/sizing_A2_n48_e10_s1001/final_model.zip", "$M/sizing_Ap_n48_e10_s1002/final_model.zip"],
 ["$M/sizing_Ap_n48_e10_s1002/final_model.zip", "$M/sizing_B_n256_e10_s1001/final_model.zip"],
 ["$M/sizing_B_n256_e10_s1001/final_model.zip", "$M/sizing_A2_n48_e10_s1001/final_model.zip"],
 ["$M/sizing_Ap_n48_e10_s1002/final_model.zip", "$M/sizing_A2_n48_e10_s1001/final_model.zip"]]
J
scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24 python -m main.h2h play-many --cells /tmp/h2h_cells.json \
  --pairs 1000 --batch-pairs 500 --device cuda --out /tmp/h2h_gpu_many --label u6_gpu_check
#   read: the JSON's engine.startup_s (one compile) vs engine.cells[i].set_cell_s (expected: about a second)
# (b) single-cell on the GPU for ONE of those cells, then compare its rows' outcome digests with (a)'s
scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24 python -m main.h2h play \
  --player $M/sizing_B_n256_e10_s1001/final_model.zip --opponent $M/sizing_A2_n48_e10_s1001/final_model.zip \
  --pairs 1000 --batch-pairs 500 --device cuda --out /tmp/h2h_gpu_single --label u6_gpu_check
```

`/tmp/h2h_gpu_many` and `/tmp/h2h_gpu_single` are separate roots, so (b)'s seed blocks do not collide with (a)'s. The host
copies of the checkpoints are loaded on the CPU in BOTH paths now (the single-cell engine used to load the player on the
device; T2 copies the weights into its slot either way, and the slot check is byte-exact), so (b) vs (a) is the GPU
identity check.

## Files

| file | what |
|---|---|
| `multicell_proof.py` | `games` (engine level: fresh vs swapped), `rows` (the CLI paths: `play_edge` per cell vs `play_cells`), `compare` |
| `run.sh` | the driver (runs from the repo root: the team pool is read cwd-relative) |
| `games.json.gz`, `rows.json.gz` | the dumps (`compare` reads them gzipped) |
| `compare.json` | the verdict + timings; `games.log`, `rows.log`, `run.log` the runs' lines |

Reproduce: `PYTHONPATH=<tree>/src ./run.sh` (≈ 12 min on the CPU).
