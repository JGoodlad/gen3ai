# The static-recovery levers: production identity, the mechanics at source, cost, the CPU smoke (2026-10-09)

The build: `gen3_static_recovery_v1`, config v150 (`designs/endstate/design_static_tokens.md` §13). Three levers, each
behind its own flag, OFF in production — `--trunk-layers N`, `--switch-hazard-cost`, `--eot-residual` — and the named
arm `--arch static_recovery` (`src/main/train/arch_arms.py`) = `static` + `--mon-hazard-cost on` + `--move-actor-state
on` + `--trunk-layers 3` + `--switch-hazard-cost on` + `--eot-residual on`, on the production recipe. CPU only (a screen
holds the GPU lease).

## 1. Flags off is byte-identical (`graph_sha.py`, the F6b unit's method)

The extractor (`extractor_compiles_test._build_production_extractor`, seed 0, perturbed), 8 seeded rows, one thread: the
dynamo FX graph text, the state_dict bytes and the outputs, at the parent `f0869884` and at this build, with every new
flag at its default.

| config | tree | graph lines | graph sha256[:16] | state_dict sha256[:16] | outputs sha256[:16] | params |
|---|---|---|---|---|---|---|
| production (legacy) | parent | 20,144 | `11338474e1547f11` | `749c56159ab028f4` | `50f4780aae896b43` | 1,937,942 |
| production (legacy) | this build | 20,144 | `11338474e1547f11` | `749c56159ab028f4` | `50f4780aae896b43` | 1,937,942 |
| `static` | parent | 20,287 | `9c1ce2c193208a89` | `1274a2311555fd98` | `ce0412ea6fdb968a` | 1,885,060 |
| `static` | this build | 20,287 | `9c1ce2c193208a89` | `1274a2311555fd98` | `ce0412ea6fdb968a` | 1,885,060 |
| the bundle + both v147 facts | parent | 25,148 | `319e5561972e78d2` | `ee40af54b4808c09` | `7448eda7cb87df2f` | 1,894,692 |
| the bundle + both v147 facts | this build | 25,148 | `319e5561972e78d2` | `ee40af54b4808c09` | `7448eda7cb87df2f` | 1,894,692 |
| **the combined arm** (`static_recovery`'s overlay) | this build | 20,651 | `622ec33f37b7fa2f` | `b7e2bf2f9807b7cd` | `2b21c72c5176b01b` | 2,020,228 |

Every config traces to ONE dynamo graph on CPU (no graph break), the combined arm included. (The bundle: `static`, `--move-resolution on`, `--speed-physics on`, `--op-reduction principled`, `--obs-facts v1`,
`--mon-hazard-cost on`, `--move-actor-state on`.) The K9 learner golden (`learner_golden_test.py`, in the routine
gate) passes unchanged. The parent's production graph is not the port's `8b376785…`: v148 / v149 changed the `x`
cell's ops.

```bash
cd <checkout>/src && python -c "import runpy,sys; sys.argv=['g', '{}']; \
  runpy.run_path('../designs/research_state/measurements/static_recovery_2026-10-09/graph_sha.py', run_name='__main__')"
```

## 2. The end-of-turn residual's mechanics, VERIFIED in `deps/pokemon-showdown`

At the pinned commit `e0551883`. `[Gen 3] OU` is `mod: 'gen3'` (`config/formats.ts`), inheriting gen3 → gen4 → … →
base; a mod entry with `inherit: true` overrides only what it states. Showdown floors each amount (minimum 1) and caps
damage at the current HP and a heal at the max HP; the rule prices the NOMINAL fraction and clamps the NET.

| component | gen-3 rule | source |
|---|---|---|
| Leftovers | +1/16 own max HP; no heal at full HP; residual order 10, sub 4 | `data/mods/gen4/items.ts` leftovers over `data/items.ts`; `sim/battle.ts` heal |
| sand chip | −1/16; Rock / Ground / Steel immune; **Sand Veil grants immunity in gen 3** (the base `onImmunity` is inherited; gen 4 only changes its evasion); Cloud Nine / Air Lock suppress; Dig / Dive semi-invulnerable immune; field residual order 8 | `data/conditions.ts` sandstorm; `data/typechart.ts`; `data/abilities.ts` sandveil, cloudnine, airlock; `data/mods/gen4/conditions.ts` |
| hail chip | −1/16; only Ice immune (Ice Body, Snow Cloak, Overcoat, Magic Guard are gen 4+) | `data/conditions.ts` hail; `data/typechart.ts` |
| a timed weather's LAST turn | the field residual decrements the duration FIRST and, at 0, ends the weather and `continue`s past `onFieldResidual`: no chip that turn. A move-set weather lasts 5 (`weather_turns` 5 → chip at 4, 3, 2, 1 left → ends at 1); an ability weather is permanent in gen 3. The observation's turns-remaining is `5 − turns_active` (`encoder/mod.rs`), the engine's `weather_turns` at decision time | `sim/battle.ts` `fieldEvent`'s duration branch; Rust `turn/residuals.rs` (`expiring = weather_turns == 1`) |
| Rain Dish | +1/16 in (effective) rain; order 10, sub 3 | `data/mods/gen3/abilities.ts` raindish |
| burn / poison | −1/8 each in gen 3 (base 1/16 burn is gen 7+); order 10, sub 6 | `data/mods/gen6/conditions.ts` brn; `data/conditions.ts` psn |
| Toxic | stage += 1 (cap 15) BEFORE the damage, `floor(maxhp/16) · stage`: the next tick is (n+1)/16 with n the ticks taken; the stage resets to 0 on switch-in (a toxic mon switching back in ticks 1/16) | `data/conditions.ts` tox (`onStart`, `onSwitchIn`, `onResidual`) |
| Shed Skin | 33 % status cure at order 10, sub 3 — BEFORE the status tick (sub 6) and Nightmare (sub 7) | `data/mods/gen4/abilities.ts` shedskin; `data/abilities.ts` |
| Leech Seed | −floor(seeded max HP / 8); the heal goes to whoever occupies the seeder's slot and equals the damage actually dealt (so ≤ the seeded mon's HP); **Liquid Ooze (gen 3): the seeder TAKES that amount**; cleared on switch-out; order 10, sub 5 | `data/moves.ts` leechseed; `data/mods/gen4/moves.ts`; `data/mods/gen4/abilities.ts` liquidooze |
| Wish | **heals HALF THE RECIPIENT's max HP in gen 3** (the gen-4 mod replaces the condition: `this.heal(target.baseMaxhp / 2)`; base gen 5+ stores the wisher's); duration 2: used on turn N, heals at the end of turn N+1 whoever is in the slot; order 7. The observation's board flag (0.5) is set iff a Wish was used LAST turn (`trackers/history.rs`), i.e. it heals at the end of THIS turn | `data/mods/gen4/moves.ts` wish; `sim/battle.ts` slot conditions |
| Ingrain | +1/16; order 10, sub 1 | `data/moves.ts` ingrain; `data/mods/gen4/moves.ts` |
| Curse (Ghost) | −1/4 of the cursed mon's max HP; a volatile, cleared on switch-out; order 10, sub 8 | `data/moves.ts` curse; `data/mods/gen4/moves.ts` |
| Nightmare | −1/4 while asleep; removed on waking; order 10, sub 7 | `data/moves.ts` nightmare; `sim/pokemon.ts` cureStatus |
| NOT included | partial trap −1/16 (its last turn ends without damage; the timer is OBS-FACTS-only), Future Sight / Doom Desire (priced at cast, not observed), Dig / Dive's turn (no observation column) | `data/mods/gen5/conditions.ts` partiallytrapped; `data/mods/gen3/moves.ts` futuresight |

**Gen-3 surprise:** a faint cancels the rest of the turn and the replacement switches in BEFORE the residuals, so it
receives that turn's Wish, weather chip and Leech Seed heal — which is what the rule's "the mon on the field at the end
of the turn" means for a forced switch.

**The observation's side:** the four volatiles (`curse`, `ingrain`, `leechseed`, `nightmare`) exist only in each side's
ACTIVE context (`encoder/mod.rs`); the Toxic counter is `min(ticks, 8) / 8`, reset on switch-out and switch-in
(`present/mon.rs`, `present/board_reading.rs`); the item block is `[num, known, consumed]` (`encoder/slot.rs`).

## 3. Cost (`cost.py` → `cost.json`)

The production toggles through a REAL SB3 build (`identity_init_test._build_real_policy`; its policy MLP is the test
harness's, so the extractor column is the comparable one; the switch-cell projection lives on the POLICY's pointer
head, so it shows in the policy column only), the compile parity fixture's 64 real rows, matmul FLOPs by
`FlopCounterMode` (grad enabled; SDPA counted), the eager forward wall per 64 rows (no_grad, 1 thread, median of 41
calls interleaved round-robin across the configs, IQR beside it). The box carried other load (loadavg 6.9 on 16
threads), so the wall is DESCRIPTIVE; the parameter and FLOP columns are exact.

| config | extractor params | policy params | matmul FLOP / row | fwd ms / 64 rows (IQR) |
|---|---|---|---|---|
| legacy (production) | 1,937,942 | 1,997,785 | 60.30 M | 164.1 (159.9–169.6) |
| legacy + `--trunk-layers 3` | 2,070,422 (+132,480) | 2,130,265 | 78.53 M (+30.2 %) | 181.0 (177.7–185.0) |
| `static` | 1,885,060 | 1,944,903 | 60.09 M | 161.4 (158.6–166.3) |
| `static` + `--trunk-layers 3` | 2,017,540 (+132,480) | 2,077,383 | 78.96 M (+31.4 %) | 177.4 (174.2–181.5) |
| `static` + `--switch-hazard-cost on` | 1,885,060 | 1,945,031 (+128) | 60.09 M | 161.3 (158.9–168.4) |
| `static` + `--eot-residual on` | 1,886,468 (+1,408) | 1,946,311 | 60.12 M (+33.8 k) | 162.5 (160.1–170.1) |
| `static` + the two v147 facts | 1,886,340 (+1,280) | 1,946,183 | 60.10 M | 161.5 (159.3–166.3) |
| **`static_recovery`** (the combined arm) | **2,020,228** | **2,080,199** | **79.01 M** | **177.4 (173.0–182.2)** |

**The trunk round is the whole cost.** One extra round is 132,480 parameters (in_proj 49,536 + out_proj 16,512 + FFN
65,920 + two LayerNorms 512) and ≈ 18.9 M matmul FLOP / row (the 64-key trunk: four 128-wide projections per token +
the attention itself), +31 % of the extractor's matmul FLOPs, but only ≈ +10 % of the eager CPU forward (the op and
the encoders dominate the wall). Every hand fact together costs < 0.1 % in FLOPs. The combined arm against `static`:
+135,168 extractor parameters, +31.5 % FLOPs, +9.9 % CPU forward; against legacy (production): +4.2 % parameters,
+31.0 % FLOPs, +8.1 % CPU forward. The GPU cost (`train_ms`, the T2 flush) is a DEFERRED read.

```bash
cd <checkout>/src && python -c "import runpy,sys; sys.argv=['cost.py','--reps','41']; \
  runpy.run_path('../designs/research_state/measurements/static_recovery_2026-10-09/cost.py', run_name='__main__')"
```

## 4. The CPU `--debug` smoke of the combined arm

`src/main/train/debug_shape_smoke_integration_test.py::test_arch_debug_smoke_reaches_an_update_and_exits_0`
(`slow`), parametrized `[production, static_recovery]`: `--arch <arm> --debug --steps 10000`, the trainer through a
`runpy` wrapper (no literal trainer name in any argv), CUDA hidden, a sealed run archive, `--debug`'s auto-shaped
update. **Both PASSED** (2026-10-09, `f0869884` + this build, under `scripts/ops/mem_cap.sh 24`, peak 2.88 GB): exit 0,
at least one update with its metrics table, `Training complete.`; for `static_recovery` the run's `model_config.json`
records every overlay key (static, the four `on` levers, `trunk_layers` 3) and `arch_source` starts
`static_recovery@`. Wall 712 s (production) and 893 s (the arm) beside other load. Recorded in
`designs/ops/slow_tier_status.json`. `--debug` is CPU, eager T2, no compile: the compiled / CUDA layer is NOT exercised
(DEFERRED).

## 5. The observation

Unchanged: every lever reads existing columns (the per-mon item / ability / type / condition / counter / sleep-belief /
spread / HP / active columns, the active contexts' volatiles, the weather block, the Wish board scalars, the Spikes
layers). The Rust encoder benchmark (`python -m agents.observation.rust_encoder_benchmark`) does not apply.
