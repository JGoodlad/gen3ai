# X35's TOY-side reference population R, frozen at T\* = X34 @ 290M (2026-10-19)

**Status: R (toy side) is FROZEN.** The file is
[`../x35_breakers_stage_2026-10-17/reference_population_toy.json`](../x35_breakers_stage_2026-10-17/reference_population_toy.json)
(content sha256 `856c04cb87e9415eda9289845d66a770582c30e773bb7c23df7fdb6221a55a4d`, frozen 2026-10-20T06:21:50Z =
2026-10-19 23:21:50 PDT, `tool_commit` `03b983be6`), and `python -m main.x35.refpop verify` on it prints OK. It replaces
the 200M STAND-IN (`reference_population_toy_STANDIN_200m.json`), which is history. No X35 arm exists. Nothing was
launched.

- **The rule:** [`design_compute_efficiency.md`](../../../endstate/design_compute_efficiency.md) §5.6.4 (the node rules and
  the band are `src/main/x35/refpop.py`'s docstring).
- **T\*:** X34's plateau checkpoint (X34 PLATEAUED and is stopped, the orchestrator's news of 2026-10-19). The file is
  `models/rb_toy_ent_s4002_deep/checkpoints/checkpoint_290000087_steps.zip`, step 290,000,087, sha256
  `cfa7f7bee7b7a43959f496ce6d5a34556bae914de9c80f90f9342a05a22965e7`. That hash was checked on the archive file and on
  the verified copy in `~/.cache/gen3ai/x34_plateau/` (its `SHA256SUMS` lists it; `sha256sum` of the two files is equal).
- **P\*** (the toy's X candidate): X26 @ 280,000,170, sha256 `814823c5…91ed`, the production R's own call.

## 1. The nodes (`refpop plan`)

T\* is the 290M node. A1's archive is N_k, the 10M node nearest k · 270M / 8, ties DOWN (exact rational arithmetic;
hand-computed first: 33.75, 67.5, 101.25, 135, 168.75, 202.5, 236.25, 270M; the planner agrees). The archive spans two
run directories (`rb_toy_ent_s4002`, X34's 100M parent, and `rb_toy_ent_s4002_deep`); no node is claimed by both.

| | nominal | file step | run dir | sha256 (12) | note |
|---|---|---|---|---|---|
| N_1 | 30M | 30,000,227 | `rb_toy_ent_s4002` | `14b63b715725` | 33.75 → 30 |
| N_2 | 70M | 70,000,183 | `rb_toy_ent_s4002` | `bf5a3ab5749f` | 67.5 → 70 |
| N_3 | 100M | 100,000,118 | `rb_toy_ent_s4002` | `2863623227c5` | 101.25 → 100 |
| N_4 | 130M | 130,000,186 | `…_deep` | `d23e89eaaad4` | 135 is a tie → 130 |
| N_5 | 170M | 170,000,152 | `…_deep` | `b6906250958e` | 168.75 → 170 |
| N_6 | 200M | 200,000,244 | `…_deep` | `bb6271a33aa6` | 202.5 → 200 (the former stand-in) |
| N_7 | 240M | 240,000,091 | `…_deep` | `a0334813f602` | 236.25 → 240 |
| N_8 | 270M | 270,000,057 | `…_deep` | `9e94dc5455e6` | exact |

The archive belongs to A1's training. It is **not** part of R; it is frozen in the same file so that the kit builds
toy A1's argv from it.

## 2. R's members

| block | label | role | step | sha256 (12) | why it is in |
|---|---|---|---|---|---|
| A | `toy_pstar` | T\* | 290,000,087 | `cfa7f7bee7b7` | the rule |
| A | `toy_280m` | T\* − 10M | 280,000,155 | `2d83f2c15c5c` | the rule |
| A | `toy_080m` | mid(N_2, N_3) = 85M, a tie → 80M | 80,000,019 | `b0c1ce9679b8` | the rule (F25's declared rounding) |
| A | `toy_150m` | mid(N_4, N_5) = 150M | 150,000,190 | `717ddc06ec51` | the rule |
| A | `toy_220m` | mid(N_6, N_7) = 220M | 220,000,245 | `74411c497035` | the rule |
| X | `pstar_x26` | P\* (another lineage AND another architecture) | 280,000,170 | `814823c514fb` | band score 0.4160, inside [0.20, 0.80] |

**Six members: 5 in the A-block, 1 in the X-block.** The A-block is disjoint by file from the archive (the tool checks).
Full hashes are in the frozen file.

## 3. The band cell and the exclusions

| X candidate | cell | score | 95 % (pair-clustered) | decision |
|---|---|---|---|---|
| `pstar_x26` | **T\*** vs P\*, 500 mirrored pairs / 1,000 games, greedy vs greedy, reveal off, CPU eager | **0.4160** (W/L/D 413 / 581 / 6; pentanomial [163, 3, 253, 1, 80]) | [0.3863, 0.4457] | **KEPT**: 0.216 above the lower edge, nowhere near rule 8's 1e-9 |

- **Orientation:** T\* is the PLAYER (the toy freeze matches `player == T*`, `opponent == the X candidate`): T\* scored
  41.6 %, i.e. P\* is the stronger player by this cell.
- **Rule-8 near-band exclusions:** none (the one played score is 0.216 from the nearest edge). **Band exclusions:** none.
- **A cross-architecture cell.** T\* is the toy network, P\* the production one; the h2h engine served two slot groups
  (fingerprints `fa633628241e` for the toy, `18ba1ba0b8e5` for production) in ONE engine, and the regime id is
  `adfdbee824c9eb0a`, the SAME regime as the production side's band cell.
- **Evidence:** [`band_cell_toy.json`](band_cell_toy.json) (the engine's summary line), [`play_band_toy.log`](play_band_toy.log)
  (all 10 batches with digests) and [`play_band_toy.sh`](play_band_toy.sh) (the resumable script).
- The rows are in the SCRATCH eval ledger `~/.cache/gen3ai/x35_refpop/ledger` (the production band cell's), under family
  `x35_band_toy` and request `x35_band_toy`. `refpop freeze --band-ledger` read the cell from there by sha256. Nothing
  was written under `models/`.

## 4. Cost

| what | amount |
|---|---|
| games | 1,000 (one cell) |
| wall time | 14.1 min (23:07 → 23:21 PDT) |
| throughput | 1.18 games/s on ONE lane (`--threads 1 --torch-threads 1`; the box load was 6–13 with another agent's pytest on the same cores) |
| how it ran | `nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 16`, `CUDA_VISIBLE_DEVICES=""`, no GPU lease |
| peak memory | 1.06 GB |
| CPU lane-hours | ≈ 0.24 |

## 5. Findings

- **F-TR1, R's paths are in X34's run directories.** Every toy member except `pstar_x26` points into
  `models/rb_toy_ent_s4002/checkpoints/` or `models/rb_toy_ent_s4002_deep/checkpoints/`, because the kit passes R's T\*
  path as the fork's `--model` (its lineage must be `rb_toy_ent_s4002_deep`). `verify` and every population read depend
  on those 13 checkpoint files (the file's `archive` and `a_block`, T\* included) staying in place, byte for byte. Any
  thinning of X34 or of `rb_toy_ent_s4002` must keep them, and the toy fork's seeded pool is X34's `snapshots/` (POOL,
  `x35_pool_batch1_toy.json`), which must not move either. `pstar_x26` points into X26's run dir (R's F-R3 list).
- **F-TR2, the band copies carry arch records copied as they are NOW.** `main.h2h` refuses a `.zip` without a sibling
  `model_config.json`. The cell played read-only copies laid out as `~/.cache/gen3ai/x35_refpop/tstar/rb_toy_ent_s4002_deep/
  {checkpoints/<zip>, model_config.json}` (T\*'s arch record copied from X34's run dir at 23:06, config_version 158,
  sha256 `83e8dd72…`) and the production band cell's `pstar/` copy (F-R4). The engine pre-flight resolved two
  architectures and raised no arch error.
- **F-TR3, the production R cannot read this cell.** `refpop freeze --band-ledger` matches `player == the freeze's P*`
  and `opponent == the X candidate`; the toy cell has T\* as the PLAYER, so for the production side (ruling C5: add T\*
  as a second X member) it finds NO cell (checked: `_band_from_ledger` returns `{'tstar': None}`). C5 needs a second
  cell with P\* as the player (≈ 14 min of one CPU lane), or a `--band-json` cell that declares score = 1 − 0.4160 =
  0.5840 (`main.h2h` draws the same team pairs in either orientation by its default schedule key; whether the battle
  draws coincide, so that the two scores are exact complements, was not checked: the symmetry is an assumption). **Since done (ruling C5, 2026-10-19):** the P\*-player cell was played and measured (0.5905 [0.5605, 0.6205]) and the production R re-frozen with T\* as a second X member: [`../x35_refpop_freeze_2026-10-19/README.md`](../x35_refpop_freeze_2026-10-19/README.md) §6.
- **F-TR4, the toy side's R is now frozen, so pre-check 7 binds the toy side too.** R must predate every `rb_x35*` run
  directory; this file's `frozen_at_epoch` is 1792477310. A toy arm launched later is fine; nothing in this unit
  creates an `rb_x35*` directory.
- **F-TR5, the toy pool PRE-READ and F31's analogue.** [`toy_pool_preread.json`](toy_pool_preread.json) holds the read-only
  `kit pool-check` set (no `--write`): X34's `snapshots/` at the fork step 290,000,087 is **20 snapshots, 92M … 270M, digest
  `2e86c4f9d7a10da2ef5e0516516fc31c219dc966842ccf74567bafdc0c41bbd5`**; none is above the fork step, so T54's filter drops
  none. It is NOT `x35_pool_batch1_toy.json` (the first toy GO writes that and must recompute the same digest, X34 being
  stopped). The pool's descriptive scalars (`snapshots/summary.json`: `win_rate_vs_bots` 0.91, `last_eval_step`
  292,000,216, `pool_generation` 19) are X34's LATEST, 2M steps past T\*: the same known bias as F31 (it matters for toy A2's
  PFSP weights only, and b-ON and the toy C0 share it).
