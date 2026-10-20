# X35's REFERENCE POPULATION R, frozen at P\* = X26 @ 280M (2026-10-19)

**Status: R (production side) is FROZEN.** The file is
[`../x35_breakers_stage_2026-10-17/reference_population_production.json`](../x35_breakers_stage_2026-10-17/reference_population_production.json)
(content sha256 `39bae18f9c0473372e0d3061f89201b6da0374c9788c2620b864924aa615bec0`, frozen 2026-10-19T08:22:55Z), and
`python -m main.x35.refpop verify` on it prints OK. No X35 arm exists. Nothing was launched. The **toy side's R was NOT
frozen at this freeze**, because T\* did not exist yet (F-R1); **it has since been frozen on the real T\* (X34 @ 290M,
2026-10-19, later the same day): [`../x35_toy_refpop_freeze_2026-10-19/README.md`](../x35_toy_refpop_freeze_2026-10-19/README.md)**.
**UPDATE, ruling C5 (same night, §6): this production R was RE-FROZEN with T\* as a second X member** (7 members, content
sha256 `cef0960e…e4519`); the file described in §1-§5 is kept as `reference_population_production_PRE_C5.json` (history,
content `39bae18f…5bec0`). §1-§5 below describe that first freeze and stay true of its A-block, archive and `production`
member.

- **The rule:** [`design_compute_efficiency.md`](../../../endstate/design_compute_efficiency.md) §5.6.4.
- **The tool:** `src/main/x35/refpop.py` (T55, `b18684a3`), run from a worktree at `82858f43` (that is the file's
  `tool_commit`).
- **P\*:** X26's `TIER1_PLATEAU` call at 280M (PREREGISTRATION Amendment 5, `fed3c416`). The file is
  `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`, sha256 `814823c5…91ed`. That hash was checked
  on the archive file and on the verified copy in `~/.cache/gen3ai/x26_plateau/` (its `SHA256SUMS` is OK).

## 1. The nodes (`refpop plan`)

P\* is the 280M node. A1's archive is N_k, the 10M node nearest k · 260M / 8, with ties going DOWN.

| | nominal | file step | sha256 (12) | note |
|---|---|---|---|---|
| N_1 | 30M | 30,000,045 | `3bd8b64bc90f` | 32.5 → 30 |
| N_2 | 60M | 60,000,135 | `413dd63fae57` | 65 is a tie → 60 |
| N_3 | 100M | 100,000,037 | `e1858ffe8a7d` | 97.5 → 100 |
| N_4 | 130M | 130,000,065 | `93180dd78112` | exact |
| N_5 | 160M | 160,000,085 | `fde1749ebdaa` | 162.5 → 160 |
| N_6 | 190M | 190,000,228 | `03e43c7e63af` | 195 is a tie → 190 |
| N_7 | 230M | 230,000,074 | `d84714fca1d0` | 227.5 → 230 |
| N_8 | 260M | 260,000,235 | `c539bcefce49` | exact |

The archive belongs to A1's training. It is **not** part of R. It is frozen in the same file so that the launch kit
builds A1's argv from it.

## 2. R's members

| block | label | role | step | sha256 (12) | why it is in |
|---|---|---|---|---|---|
| A | `x26_pstar` | P\* | 280,000,170 | `814823c514fb` | the rule |
| A | `x26_270m` | P\* − 10M | 270,000,128 | `091979cd61bb` | the rule |
| A | `x26_080m` | mid(N_2, N_3) = 80M | 80,000,111 | `52cebba08e38` | the rule |
| A | `x26_140m` | mid(N_4, N_5) = 145M, a tie → 140M | 140,000,062 | `70d515b8c65c` | the rule (F25's declared rounding) |
| A | `x26_210m` | mid(N_6, N_7) = 210M | 210,000,149 | `b379a47b3d84` | the rule |
| X | `production` | registry baseline `rb_es_end2_s2005/final_model.zip` (15,047,521) | — | `4047f4380661` | band score 0.6935, inside [0.20, 0.80] |

**Six members: 5 in the A-block, 1 in the X-block.** The A-block is disjoint by file from the archive, which the
tool checks. Full hashes are in the frozen file.

## 3. The band cell and the exclusions

| X candidate | cell | score | 95 % (pair-clustered) | decision |
|---|---|---|---|---|
| `production` | P\* vs `rb_es_end2_s2005@15047521`, 500 mirrored pairs / 1,000 games, greedy vs greedy, reveal off, CPU eager | **0.6935** (W/L/D 690 / 303 / 7; pentanomial [36, 2, 229, 5, 228]) | [0.6664, 0.7206] | **KEPT**: 0.1065 inside the upper edge, nowhere near rule 8's 1e-9 |
| T\* (X34's plateau node) | **not played: X34 has not launched, so no T\* file exists** | — | — | **ABSENT**: not a rule-8 exclusion and not a band exclusion (F-R1) |

- **Rule-8 near-band exclusions:** none. The one played score is 0.094 from the nearest edge.
- **Band exclusions:** none.
- **Evidence:** [`band_cell_production.json`](band_cell_production.json) (the engine's summary line),
  [`play_band.log`](play_band.log) (all 10 batches with digests), and [`play_band.sh`](play_band.sh) (the resumable
  queue).
- The rows are in a SCRATCH eval ledger, `~/.cache/gen3ai/x35_refpop/ledger`, under family `x35_band` and request
  `x35_band_production`. `refpop freeze --band-ledger` read the cell from there by sha256. It was not written under
  `models/`.

## 4. Cost

| what | amount |
|---|---|
| games | 1,000 (one cell) |
| wall time | 13.5 min (01:09 → 01:22 PDT) |
| throughput | 1.23 games/s on ONE lane (`--threads 1 --torch-threads 1`) |
| how it ran | `nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 16`, `CUDA_VISIBLE_DEVICES=""` |
| peak memory | 1.06 GB |
| CPU lane-hours | ≈ 0.23, far under the 12 lane-hour bound |

The run went beside the live X26 and the hidden-info maturity battery's 2 slots, and touched neither.

## 5. Findings

- **F-R1, T\* IS ABSENT and R IS WRITE-ONCE.** §5.6.4 names two X-block candidates: T\* and `production`. X34
  (`rb_toy_ent_s4002_deep`) has not launched, so T\* does not exist. This production-side R therefore has ONE
  other-lineage member. The file is write-once ("a new call is a new file"), so T\* can enter the production R only
  through a NEW freeze, which must happen before any X35 arm is read against it.
  - **The orchestrator's call:** either accept a 1-member X-block for batch 1, or re-freeze after X34's plateau, which
    would be AFTER the production arms launch, while R1 is read against R.
  - No stand-in was substituted for T\*: `rb_toy_ent_s4002`'s 100M final is X34's start, not its plateau.
  - The toy side's R cannot be frozen until T\* exists.
- **F-R2, the F20 lean is heavier.** Five of six members are X26's own nodes, adjacent to A1's training archive. With
  one X member, the equal-weight S is 5/6 own-lineage. A1's R1 leans toward A1 by construction more than the rule's
  7-member sizing assumed. The X-block, G-F and G-K remain the unbiased checks.
- **F-R3, R's paths are in X26's LIVE run directory.** Every entry except `production` points at
  `models/rb_x26_s2601/checkpoints/…`. That is deliberate: `main.x35.kit` passes R's P\* path as the fork's
  `--model`, and the fork's lineage must be `rb_x26_s2601`. A copy elsewhere would mis-attribute it.
  - So `verify` and every population read depend on those 14 checkpoint files staying in place, byte for byte.
  - Any later thinning of X26 must keep them. The list is the frozen file's `archive` and `a_block`.
- **F-R4, P\*'s band copy carries a copied arch record.** `main.h2h` refuses a `.zip` without a sibling
  `model_config.json`. The verified copy in `~/.cache/gen3ai/x26_plateau/` has none.
  - So the cell played a read-only copy laid out as `~/.cache/gen3ai/x35_refpop/pstar/rb_x26_s2601/{checkpoints/<zip>,
    model_config.json}`. Its `model_config.json` was copied from X26's run directory as it is NOW (config_version 158,
    `gen3_mon_tied_gain_v1`, sha256 `df71eea6…`). That is after Amendment 6's resume, not the record as of 280M.
  - The engine pre-flight resolved one architecture for both checkpoints and raised no arch error.
  - The first launch attempt failed on exactly this refusal before playing anything. Its only side effect was the
    family registration in the scratch ledger.
- **F-R5, `production` is weak against P\*.** At 0.69 it is inside the band, but it is a 15M end-state run against a
  280M one. It shares X26's architecture, so it is not a "cross-architecture" member. It is another lineage only.
- **F-R6, the archive's tie rule moved two nodes down.** N_2 at 65M goes to 60M and N_6 at 195M goes to 190M, under
  the tool's exact-rational "ties down". So does mid(N_4, N_5) at 145M, which goes to 140M. All three are as F25
  declared. They are reported because they differ from naive rounding.

## 6. C5: the re-freeze with T\* as a second X member (2026-10-19, orchestrator ruling)

- **The cell (measured, not assumed symmetric):** P\* (X26 @ 280,000,170, the PLAYER) vs T\* (X34 @ 290,000,087, sha256
  `cfa7f7be…a22965e7`), 500 mirrored pairs / 1,000 games, greedy vs greedy, reveal off, CPU eager, one engine serving the
  two architectures (fingerprints for production and toy; regime `adfdbee824c9eb0a`, the same as the other two band
  cells): **0.5905, 95 % [0.5605, 0.6205]** (pair-clustered), SE 1.532 pp, W/L/D 588 / 407 / 5, pentanomial
  [80, 1, 247, 2, 170]. The toy freeze's T\*-player cell read 0.4160 (so the exact complement is 0.5840): the two
  orientations differ by 0.0065 = 0.3 SE of the difference. **KEPT** (0.2095 below the upper edge, nowhere near rule 8).
  Evidence: [`band_cell_tstar_c5.json`](band_cell_tstar_c5.json), [`play_band_c5.log`](play_band_c5.log) (all 10 batches
  with digests), [`play_band_c5.sh`](play_band_c5.sh) (`play_band.sh` without its `family-register`).
- **The ledger:** the cell is in the SCRATCH ledger `~/.cache/gen3ai/x35_refpop/ledger`, family `x35_band`, request
  `x35_band_production`: the SAME request as the `production` cell, so one `--band-ledger` read decides both X members.
  Nothing was written under `models/`.
- **The freeze:** the old file moved aside (`git mv`) to `reference_population_production_PRE_C5.json`, then
  `refpop freeze --side production --pstar <P*> --archive-dir models/rb_x26_s2601/checkpoints --x production=baseline:production
  --x tstar=<T* zip in rb_toy_ent_s4002_deep/checkpoints> --band-ledger … --band-family x35_band --band-request
  x35_band_production`, run at `tool_commit` `0fee8fd15`. **7 members: A-block `x26_pstar` / `x26_270m` / `x26_080m` /
  `x26_140m` / `x26_210m`, X `production` (0.6935) and X `tstar` (0.5905); content sha256
  `cef0960eac605f1dc026442d1089dc23654e23df91010feaaeea7890901e4519`**, frozen 2026-10-20T06:51:59Z. `refpop verify` is OK
  on it, on the PRE_C5 file and on the toy file. The archive, A-block and P\* are byte-identical to the first freeze.
- **Cost:** 1,000 games, 17.2 min wall (23:34 -> 23:51 PDT, 1.09 games/s on one lane, `--threads 1 --torch-threads 1`),
  `nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 16`, `CUDA_VISIBLE_DEVICES=""`, peak 0.82 GB; about 0.29 CPU
  lane-hours.
- **F-C5-1, F-R2 is lighter:** 5 of the 7 members are X26's own nodes (was 5 of 6).
- **F-C5-2, the production R now depends on one X34 file:** T\*'s `checkpoint_290000087_steps.zip` in
  `models/rb_toy_ent_s4002_deep/checkpoints/` (it is already the toy R's `pstar`, F-TR1); `verify` re-hashes it at every
  launch.
- **F-C5-3, the family-register step was dropped:** family `x35_band` was registered at the first freeze's commit
  (`82858f43`), and `family-register` refuses a re-register at another commit ("other terms"). The second cell is the same
  family and request, so it only plays.
- **F-C5-4, the cell's T\* arch record is the toy cell's copy** (`~/.cache/gen3ai/x35_refpop/tstar/…`, F-TR2).
