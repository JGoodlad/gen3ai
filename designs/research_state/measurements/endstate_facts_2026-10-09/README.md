# The fact-completion levers and the END-STATE arm: identity, mechanics at source, cost, K9(b), the CPU smoke (2026-10-09)

The build: `gen3_endstate_facts_v1`, config v152 (`designs/endstate/design_hand_computed_features.md` §4 ranks 3-5, §5
rank 1, finding 7; `design_static_tokens.md` §14). Five levers, each behind its own flag, OFF in production —
`--move-resolution-facts {off,full}`, `--status-facts {off,exact}`, `--ko-ramp {ramp,exact}`, `--drop-progress-clock
{off,on}`, `--g-ledger {coarse,eot}` — and the named arm **`--arch endstate`** (`src/main/train/arch_arms.py`) =
`static_recovery` + `--move-resolution on --speed-physics on --value-threat-inject off --op-reduction principled
--obs-facts v1` + the five, on the production recipe. CPU only (a GPU-check agent held the lease).

## 1. Flags off is byte-identical (`shas.py`, over `../static_recovery_2026-10-09/graph_sha.py`)

The extractor (`extractor_compiles_test._build_production_extractor`, seed 0), 8 seeded rows, one thread: the dynamo FX
graph text, the state_dict bytes and the outputs, at the parent `58f8149a` (`git archive` of `src/`, `data/`,
`production_config.json`) and at this build (rebased onto `2ec395c0`, which moved no file under `src/agents/model`,
`src/agents/observation` or `production_config.json`), every new flag at its default.

| config | tree | graph lines | graph sha256[:16] | state_dict sha256[:16] | outputs sha256[:16] | params |
|---|---|---|---|---|---|---|
| production (legacy) | parent / this build | 20,148 | `421c6b98ce7937f4` | `749c56159ab028f4` | `51c02c6c6342a045` | 1,937,942 |
| `static` | parent / this build | 20,291 | `6fb7bfd8364205d6` | `1274a2311555fd98` | `92cd6c3b61c07f1c` | 1,885,060 |
| `static_recovery` | parent / this build | 20,656 | `859fe82912a8583a` | `b7e2bf2f9807b7cd` | `edef79de32277a26` | 2,020,228 |
| **E** = `static_recovery` + the bundle (what the GPU check validates) | parent / this build | 25,436 | `b3361475ef3677b7` | `4451ffc5108942db` | `23f3ff11dd020732` | 2,028,580 |
| **`endstate`** (the arm) | this build | 34,643 | `55e1a5649289d923` | `3345d4eb83d9a6ce` | `d70ac227d47dbd82` | 2,029,540 |

Every config traces to ONE dynamo graph on CPU (no graph break), the arm included. (The first E read differed in the
graph text only — a renamed local in `MoveResolutionCell.forward`; the off path keeps its original spelling verbatim
now, and E reads the parent's hash.) The arm's extra +9,207 graph lines are the exact P(KO)'s 32 clamped terms per
site, the restored / status / cure blocks and the one end-of-turn rule in the `g` cell.

```bash
python designs/research_state/measurements/endstate_facts_2026-10-09/shas.py <parent src> <this src>   # -> shas.out
```

**Addendum (`gen3_probe_facts_v1`, v153, the second commit):** with `--effective-stats on --move-target-state on` in
the arm, production (`421c6b98…`) and E (`b3361475…`) are still the parent's hash for hash; the extended arm is ONE
graph, 34,898 lines, `55c7f5c009d4595b`, state `df71da47741a8a0f`, outputs `c78ca89766000c71`, 2,031,460 params (+1,920:
two zero-init projections, 7 → 128 and 8 → 128); the K9(b) recorder still sees no undeclared op; the CPU `--debug` smoke of the extended arm PASSED (1,039 s, peak 3.36 GB; a run beside the routine gate timed out at the test's 1,200 s bound, now 1,800 s). Stat tables VERIFIED:
`sim/pokemon.ts` `boostTable = [1, 1.5, 2, 2.5, 3, 3.5, 4]`, `data/mods/gen3/scripts.ts` accuracy
`[1, 4/3, 5/3, 2, 7/3, 8/3, 3]`, paralysis ×0.25 (`data/mods/gen4/conditions.ts`).

## 2. The mechanics, VERIFIED in `deps/pokemon-showdown` (pinned `e0551883`; `[Gen 3] OU` = gen3 → gen4 → … → base)

| rule | gen-3 fact | source |
|---|---|---|
| damage roll | `floor(floor(base · (100 − random(16))) / 100)`: r uniform on 85 … 100, applied LAST (after STAB and the chart) | `sim/battle.ts` `randomizer`; `data/mods/gen3/scripts.ts` `modifyDamage` |
| crit chance | gen ≤ 5 `critMult = [0, 16, 8, 4, 3, 2]` by `critRatio` (default 1): 1/16; `critRatio: 2` (Slash, Razor Leaf, Crabhammer, Karate Chop, Cross Chop, Aeroblast, Air Cutter, Blaze Kick, Leaf Blade, Poison Tail, Razor Wind, Sky Attack) 1/8 | `sim/battle-actions.ts` `getDamage`; `sim/dex-moves.ts` (`critRatio` default 1); `data/pokemon/gen3_moves.json` `critRatio` |
| crit damage | ×2 after the `+2`, before the roll; Reflect / Light Screen skip a crit | `data/mods/gen3/scripts.ts` `modifyDamage`; the screens' `ModifyDamagePhase1` |
| KO | damage ≥ current HP (integers) | the HP model |
| their HP precision | `Standard` ⊃ `Standard AG` ⊃ `HP Percentage Mod`: the shared HP is `ceil(100 · hp / maxhp)` (99 while not full), so p < 100 % ⇒ (p − 1 %, p]; the Rust bridge folds the non-owner's HP the same way | `data/rulesets.ts`; `sim/pokemon.ts` `getHealth`; `src/rust_sim/src/bridge.rs` `fold_hp_line` |
| burn | Physical move's base damage ×0.5 BEFORE the `+2`, skipped with Guts; a non-formula move never reaches `modifyDamage` | `data/mods/gen3/scripts.ts` `modifyDamage` |
| paralysis | speed ×0.25 (`chainModify(0.25)` unless Quick Feet, gen 4+); full paralysis 1/4 before the move | `data/mods/gen4/conditions.ts` `par` (inherited by gen 3) |
| sleep / freeze | sleep `time = random(2, 6)` (1–4 turns asleep); freeze thaws 1/5 per move attempt — rule constants; the per-mon sleep-wake belief is the observation's O1 | `data/mods/gen3/conditions.ts` `slp`; `data/mods/gen4/conditions.ts` `frz` |
| cure paths | Heal Bell / Aromatherapy cure the party; Natural Cure cures on switch-out (gen 3 = gen 4's); Lum cures any major status, Chesto sleep, both `gen: 3` | `data/moves.ts`; `data/mods/gen4/abilities.ts` `naturalcure`; `data/items.ts` `lumberry` / `chestoberry` |
| Substitute HP | `Math.floor(target.maxhp / 4)` | `data/moves.ts` `substitute` |
| Focus Punch | `lostFocus` is set by damage to the USER (a hit on its Substitute does not) | `data/mods/gen4/moves.ts` `focuspunch` |

## 3. Cost (`cost.py` → `cost.json`; CPU, 64 real rows, eager no-grad, 1 thread, median of 21 interleaved, load ~4-6)

| config | extractor params | policy params | matmul FLOP / row | fwd ms / 64 rows (IQR) |
|---|---|---|---|---|
| production | 1,937,942 | 1,997,785 | 60.30 M | 121.4 (116.0–127.8) |
| E (`static_recovery` + the bundle) | 2,025,948 | 2,085,151 | 94.31 M | 161.3 (156.0–173.0) |
| E + `--move-resolution-facts full` | 2,026,156 (+208) | 2,085,359 | 94.31 M | 161.3 |
| E + `--status-facts exact` | 2,026,700 (+752) | 2,085,903 | 94.30 M (the `neutralization` einsum leaves) | 162.9 |
| E + `--ko-ramp exact` | 2,025,948 | 2,085,151 | 94.31 M | **177.3 (+9.9 %)** |
| E + `--drop-progress-clock on` | 2,025,948 | 2,085,151 | 94.31 M | 161.6 |
| E + `--g-ledger eot` | 2,025,948 | 2,085,151 | 94.31 M | 162.1 |
| **`endstate`** (the arm) | **2,026,908 (+960)** | **2,086,111** | **94.30 M** | **181.3 (+12.4 % vs E)** |

The exact P(KO) is the whole cost: 32 clamped element-wise terms per (defender, candidate) cell at every KO site
(~2,300 incoming cells per row), which the matmul FLOP counter does not see; Inductor fuses them into one kernel, so
the eager CPU wall is an upper bound on the compiled cost (the GPU cost is a DEFERRED read). The other levers are
< 1 % of the wall; the parameters they add (+960) are zero-init projections only.

## 4. K9(b) (`k9_tie_margins.py`: the tie-margin recorder over the forward, every zero-init projection planted, 64 fixture rows)

No undeclared discrete op in any config (the recorder's `check()` passes for production, E and `endstate`). The
excluded share at `FP32_TIE_EPS` 2e-4: production 0.062, E 0.062, `endstate` 0.078 (one more row of 64; DESCRIPTIVE,
64 rows). The exact P(KO) adds NO discrete site: a staircase over the 16 rolls would be a step function of a
weight-driven damage estimate, i.e. a threshold within K9(b)'s epsilon of some step on most rows (~2,300 cells per
row) — the reason each roll's KO is resolved over the observed HP interval instead (continuous, piecewise linear). The
one new comparison (`hp_frac >= 1.0`, their reported HP against full) is an observation read (EXACT).

## 5. The CPU `--debug` smoke of `--arch endstate`

`src/main/train/debug_shape_smoke_integration_test.py::test_arch_debug_smoke_reaches_an_update_and_exits_0[endstate]`
(`slow`): `--arch endstate --debug --steps 10000`, CUDA hidden, a sealed run archive, `--debug`'s auto-shaped update, under
`scripts/ops/mem_cap.sh 24`. **PASSED** (2026-10-09, this build): exit 0, at least one update with its metrics table, `Training complete.`, the run's `model_config.json` records every overlay key and `arch_source` starts `endstate@`. Wall 884 s beside other load, peak 3.34 GB; recorded in `designs/ops/slow_tier_status.json`. `--debug` is CPU, eager T2, no compile: the compiled / CUDA layer is NOT exercised (DEFERRED, §7).

## 6. The observation

Unchanged: every lever reads existing columns; `--drop-progress-clock on` zeroes `turns_since_progress` at the model's
input (`ObsUnpack`), never in the Rust layout. The Rust encoder benchmark does not apply.

## 7. DEFERRED to a GPU lease (the new flags only)

CUDA compile parity forward + backward of `--arch endstate` (and of E + `--ko-ramp exact` alone: the 32-term fused
kernel, the straight clamp backward); T2's CUDA-graph build; the K9(b) behaviour gate on a real first update (the
excluded share on real rows at the learner's micro-batch); a real two-minute `--compile-trainer` launch; the cost read
(`train_ms`, the T2 flush) against E at the same commit.
