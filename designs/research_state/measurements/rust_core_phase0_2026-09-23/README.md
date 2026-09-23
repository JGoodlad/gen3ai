# RUST CORE PROGRAM — PHASE 0: the baseline, the event-attribution spike, the tech-debt inventory

<!-- A MEASUREMENT record (2026-09-23, worktree of `da0ab0b0`). The plan it licenses is
`designs/endstate/program_rust_core.md`; the end state is
`designs/endstate/design_three_tier_environment.md`. Do not rewrite the tables — a later
milestone appends its own after-table and reports against these. -->

**Box.** 16 cpus, a live training arm (`ai_v13_21_wcont_b`, `--n-envs 48`) owning the GPU and most
of the CPU; load average **14–37** across this session. Everything here ran on CPU at `nice 10`.
**Every timing is reported as an interleaved ratio with its load beside it; an absolute ms is a
reading of that minute's box, not a constant.**

## 0. Headlines

| | result | tag |
|---|---|---|
| **(a) searched decision** | view road **143 ms/decision wide B** (684 arms / 10 banked decisions) and **23.5 ms at B = 1**, medians of 6 interleaved one-road-per-process rounds at loads 14–31; paired view ÷ protocol **0.711** (range 0.653–0.804) wide, **0.834** at B = 1. **Python glue + trackers + folds = 59.4 % of the view road's wall at wide B (66.3 % counting the protocol/event folds; 76.1 % with transport/JSON); the rust engine ≈ 4 %** (+ an unsplit 8.5 % `open_root`) | MEASURED |
| **(b) per-env opponent inference** | compiled B = 1 CPU forward **2.44–4.64 ms** (6.5–7.3× over eager; 1 torch thread; loads 30–35); a B = 48 CPU batch **0.92–1.26 ms/row**. At the live arm's **445 steps/s** — and assuming every opponent is a network — **0.4–2.1 cores of 16** | MEASURED; the core count is an UPPER BOUND |
| **(c) obs / live-view build** | cold obs build **0.52–0.66 ms**, production shape (assembler warm + view warm) **0.15–0.16 ms**; `LiveView.from_battle` **0.109–0.112 ms**, 1.26–1.27× the pre-micros reference, **723 Python calls / build (load-free)** | MEASURED |
| **(d) inventory** | the Python/JS re-derivation chain is **28,910 LOC in 11 links** (§1d); the cutover's deletion manifest names ≈ 7.7k LOC of it; the named oracles (poke-env, the event layer, read-models, trackers, the encoder) keep ≈ 17k | COUNTED |
| **SPIKE** | source-typed events for 15 of 35 event kinds (82 % of event lines by volume) equal `Gen3Battle`'s reading on **297 real bridge battles × 2 viewers — 259,782 per-viewer comparisons, 0 residual, 0 unmatched** once **8 named reading rules** are applied; **129,298 / 129,298** typed records re-derived from their own line text; the sim's truth for the outcome OWNER re-derived from line order on **12,850 / 12,850** outcome lines. **§3.4's hazard is tractable**: 6 of 36 attribution rules need a poke-env disambiguation rule, every one text-derivable | MEASURED, two seeds |

---

## 1. THE BASELINE

### (a) One searched decision — the view road, one road per process, interleaved

```bash
export PYTHONPATH=$PYTHONPATH:src
export POKESIM_SEARCH_DRIVER_BIN=$PWD/src/rust_sim/target/release/search_driver   # THIS worktree's build
R=designs/research_state/measurements/rust_core_phase0_2026-09-23
T=models/ai_v12_02_winprob_critic/eval_traces           # read-only (the main checkout's models/)
B=src/main/search_dividend/search_decision_benchmark.py
# per round r = 1..6, in this order, each its own process:
python3 $R/run_bench.py $B --traces $T --decisions 10 --m-opp 3 --arm honest --k-worlds 4 --roads view
python3 $R/run_bench.py $B --traces $T --decisions 10 --m-opp 3 --arm honest --k-worlds 4 --roads protocol
python3 $R/run_bench.py $B --traces $T --decisions 10 --m-opp 1 --n-actions 1 --arm honest --k-worlds 1 --roads view
python3 $R/run_bench.py $B --traces $T --decisions 10 --m-opp 1 --n-actions 1 --arm honest --k-worlds 1 --roads protocol
# the expand_many rust / transport / python split (driver-side timer), twice:
POKESIM_SEARCH_TIMING=1 python3 $R/run_bench.py \
  designs/research_state/measurements/expand_many_2026-09-22/profile_expand_many.py \
  --traces $T --decisions 10 --m-opp 3 --k-worlds 4 --road view
python3 $R/aggregate_baseline.py                          # the tables below, from logs/
```

`run_bench.py` exists only because the benchmark borrows the honest arm's pool from
`main.search_dividend.__main__`, and this brief forbids importing a `__main__` module: it registers
a stub carrying a verbatim copy of that one function and runs the script unmodified.

| B | round | load (view) | view ms/dec | load (protocol) | protocol ms/dec | view ÷ protocol |
|---|---|---|---|---|---|---|
| wide (684 arms) | 1 | 29.6 | 213.5 | 28.5 | 300.0 | 0.712 |
| | 2 | 27.8 | 210.3 | 31.2 | 300.1 | 0.701 |
| | 3 | 31.3 | 143.3 | 29.3 | 178.2 | 0.804 |
| | 4 | 24.1 | 129.2 | 24.1 | 197.9 | 0.653 |
| | 5 | 19.9 | 143.2 | 18.7 | 194.4 | 0.737 |
| | 6 | 16.9 | 137.6 | 15.9 | 193.6 | 0.711 |
| | **median** | | **143.2** | | **196.2** | **0.711** |
| 1 (10 arms) | 1–6 | 14.3–30.9 | 35.0 · 39.3 · 22.1 · 23.2 · 23.4 · 23.7 | | 44.4 · 45.2 · 26.3 · 28.0 · 27.2 · 47.1 | 0.79 · 0.87 · 0.84 · 0.83 · 0.86 · 0.50 |
| | **median** | | **23.5** | | **36.2** | **0.834** |

**The view road's EXCLUSIVE wall, grouped into the program's buckets** (pooled over the 6 view
runs; `expand_many` is split by the driver timer's own shares — **29.9 % rust / 70.1 % pipe +
`json.loads` + the reply's GC**; `open_root` has no such split and is its own row):

| bucket | wide B | B = 1 |
|---|---|---|
| rust engine (the `expand_many` child share) | 4.2 % | 1.0 % |
| rust `open_root` (engine + its JSON, unsplit) | 8.5 % | 29.0 % |
| transport / JSON (pipe, dumps, `json.loads`, the reply's GC) | 9.9 % | 2.4 % |
| Python parse (prefix replay through poke-env, event folds) | 6.8 % | 24.7 % |
| trackers / fork (thaw, prefix decision track, successor context) | 18.3 % | 9.6 % |
| read-model folds (`view_adapter`, mask, choices) | 26.7 % | 9.2 % |
| encode (obs) | 11.2 % | 1.5 % |
| Python glue (uninstrumented) | 14.5 % | 22.7 % |
| **§7's "Python glue + trackers + folds"** | **59.4 %** (66.3 % with the Python-parse row) | **41.4 %** (66.1 %) |

**For the record only** — the owner licensed the program on unification, so this is not a go/no-go:
the design's §7 decision point reads **licensed (> 50 %) at wide B on its own definition**, and on
the broad definition at both shapes. The engine the rewrite ports FROM is ≈ 4–13 % of the wall.

⚠️ **The `view_adapter read-models` row is 0.65 ms/arm here against 0.29 on 2026-09-22** while the
decision got cheaper. `expand_many_2026-09-22/` §2 showed an exclusive row can be someone else's
GC; this record does not split it. **UNVERIFIED** which it is.

### (b) Per-env opponent inference on CPU, at the training shape

`opponent_inference_bench.py` times exactly the opponent's per-decision network call
(`policy.get_distribution` on a Dict obs, `no_grad`, ONE torch thread as `env_factory` sets it) on
the live arm's own snapshot (`ai_v13_21_wcont_b/snapshots/snapshot_000072000000.zip`) over **48
real banked decisions** (`eval_traces/step_76000032/staller_v2/loss_s1_003_states.npz`), compiled
with the production `maybe_compile_extractor`; arms alternated round by round, each a median of 7
rounds of a min over 20 calls.

| run | threads | load | eager B=1 | **compiled B=1** | eager B=48 per row | compiled B=48 per row | compile gate's own reading |
|---|---|---|---|---|---|---|---|
| r1 | 1 | 34.8 | 30.04 ms | **4.64 ms** (6.48×) | 1.82 ms | 1.26 ms | 18.33 → 2.62 ms (7.0×) |
| r2 | 1 | 30.6 | 17.85 ms | **2.44 ms** (7.32×) | 1.29 ms | 0.92 ms | 21.73 → 3.05 ms (7.1×) |
| t4 | 4 | 25.2 | 16.77 ms | 2.08 ms (8.07×) | 0.75 ms | 0.46 ms | 17.03 → 1.89 ms (9.0×) |

max|Δlogits| eager vs compiled at B = 1: **1.91e-06** (1 thread), 1.67e-06 (4). The runbook's
5.07e-07 does not say which tensor it was measured on; **UNVERIFIED** that the two are the same
quantity — T2's gate must name its tensor.

**Pricing Tier 2.** The live arm advanced 500,016 steps between its last two checkpoints in 1,124 s
(mtimes) = **445 steps/s** wall-average. If EVERY opponent decision were a network forward (an upper
bound — the pool also fields bots), opponent inference costs 445 × 2.44–4.64 ms = **1.1–2.1 cores**
at this load, or **0.43 cores** at the runbook's quiet-box 0.976 ms. **The design's "sixty-four
opponents stop costing sixty-four cores" is not what this box does**: a compiled B = 1 forward is a
small fraction of an env worker's time. Tier 2's case is unification, the batched search leaf, and
that N envs per process cannot exist without batched opponents (the program doc restates it).

### (c) The obs-build and live-view benchmarks

```bash
python3 src/agents/training/obs_build_benchmark.py --seed 0          # twice
python3 src/agents/training/live_view_build_benchmark.py --seed 0    # twice, interleaved with ^
```

| benchmark | r1 (load 23.6) | r2 (load 25.9) | load-free |
|---|---|---|---|
| full per-decision obs build, COLD | 0.656 ms | 0.520 ms | cProfile COLD block **1,578,401** calls (identical both runs) |
| encode, view memo warm | 0.422 | 0.373 | |
| assembler WARM + view WARM (**the production shape**) | **0.160** | **0.152** | WARM block **520,001** calls |
| `live_view()` alone | 0.111 | 0.120 | |
| `LiveView.from_battle` (frozen turn-100 board), new vs reference | 0.1089 vs 0.1398 (**1.273×**) | 0.1121 vs 0.1388 (**1.258×**) | **723 vs 1,139 Python calls / build** |

The call counts are the numbers a later milestone should compare against on a busy box.

### (d) The INVENTORY — the Python-side re-derivation chain, and what retires it

`python3 designs/research_state/measurements/rust_core_phase0_2026-09-23/inventory_loc.py` (whole
files, non-test, at `da0ab0b0`).

| # | link | LOC | owner (leaf doc) | absorbed by | the OLD PATH it lets the cutover DELETE |
|---|---|---|---|---|---|
| 1 | poke-env battle-state parse (`poke_env/battle/*`, 10 files) | 6,528 | `src/agents/battle/CLAUDE.md` (the fork) | Tier 1a `parse` + `present()` | none in production after M7 — **survives as the parity ORACLE** |
| 2 | `Gen3Battle` event layer + `BattleEvent` schema | 1,468 | `src/agents/battle/CLAUDE.md` | M1 (`CoreEvent` + its reading projection) | the production event build; **the schema and `Gen3Battle` survive as the oracle** |
| 3 | read-models: `LiveView` / `TurnView` / `LegalActions` / `StrictBattleView` | 1,124 | same | M2 (`OneSidedView`, `legal_actions`), M3 (`TurnView` → version methods) | production use; survive as the `present()` oracle; the strict-API lock's idea becomes the Rust type boundary |
| 4 | second constructors: `view_adapter` + `ViewEventFolder` | 1,111 | same + `designs/rust_sim/one_sided_view.md` | M1 / M2 | **deleted** — the view road, `view_pN`/`view_pN_at` JSON, D10's chunk cut, the fold's five-fact light board |
| 5 | per-decision folds + trackers (`TurnDelta`, snapshot, `EpisodeTracker`, choice-band, HP, reward tracker, clone pins, wish/sleep beliefs) | 2,985 | `src/agents/training/CLAUDE.md` | M3 | the pinned-pickle tracker thaw (`clone_pins.py`), `turn_delta_legacy.py`; the trackers survive as the oracle until the cutover |
| 6 | encoder (20 sub-encoders + assembler) | 5,481 | `src/agents/observation/CLAUDE.md` | M4 | the assembler cache + `live_view` memo (Python perf layers); **the encoder survives as the oracle until two full seeds are byte-clean post-cutover** |
| 7 | legality / action (mask, mapper, serialize) | 741 | `src/agents/action/` | M2 (`legal_actions`) | production use of poke-env's derived legality flags (the "second poke-env-interpreted seam") |
| 8 | transport: bridge client + node drivers + search/replay sessions | 3,707 | `src/rust_sim/CLAUDE.md`, `utils/bridge/README.md` | M4 / M5 | the per-env bridge child, `--use-bridge node\|off` for training, `search_session` JSON, node `search_driver.js` / `replay_driver.js` / `replay_kernels.js`; `local_sim_bridge.js` **survives** (it gates the sim) |
| 9 | search materializers (protocol road + view road) | 1,520 | `designs/rust_sim/one_sided_view.md` | M2 (search adopts early) | the protocol road (`materialize_branches*`, `_PlayerSnapshot`), the D10 / depth fallbacks, `--materializer protocol\|view` |
| 10 | env process layer (env, wrappers, async vec-env, forkserver / compile preload, pool cache) | 3,475 | `designs/ops/training_runbook.md` | M5 + T2 | forkserver preload, `SubprocVecEnv`, `--async-rollout`, `--compile-opponents{,-preload,-strict}` |
| 11 | player-side inference (per-env opponent forward) | 770 | `src/agents/training/CLAUDE.md` | T2 | per-consumer model load + forward |
| | **TOTAL** | **28,910** | | | |

For scale: the Rust port is 40,647 LOC (`src/rust_sim/src`, including this spike's ~300).

---

## 2. THE SPIKE — attribution emitted at the source, and read back from protocol text

### (a) Every attribution rule, classified

**SOURCE-FACT** = the simulator knows it at the transition (true of every row — it is the reason the
end state emits at the source). The column that matters is whether the PROTOCOL carries it:
**DERIVABLE** (unambiguous from the line, plus the five light-board facts `event_fold.py` names) or
**AMBIGUOUS** (needs poke-env's disambiguation rule). `R*` = the reading rules of §2b.

| # | rule | where | protocol | notes |
|---|---|---|---|---|
| 1 | classify every keyword; raise on unknown / non-gen3 | `battle_event.py:634-650` | DERIVABLE | 120 keywords: 51 EVENT · 24 CONTROL · 38 COSMETIC · 7 UNSUPPORTED |
| 2 | the Player intercepts `request`/`showteam`/`win`/`tie`/`error`/`bigerror`/`t:`/`expire`/`uhtmlchange` before `parse_message` | `poke_env/player/player.py:159,430-471` | DERIVABLE | none is an event |
| 3 | `\|error\|[Unavailable choice]` → an OUT-OF-BAND `CHOICE_REJECTED` on the REJECTED side's log only | `player.py:461-471`, `gen3_battle.py:350-379` | DERIVABLE, **per-viewer** | the source knows the refusal; only one viewer's log carries it |
| 4 | `turn` = the last `\|turn\|N` | `gen3_battle.py:564` | DERIVABLE | |
| 5 | `seq` dense per log — synthetics take a seq | `gen3_battle.py:381-388` | DERIVABLE | forces R5 into the projection |
| 6 | side = ident prefix vs `player_role` | `gen3_battle.py:455-458` | DERIVABLE | |
| 7 | species of an ident = the mon keyed by NICKNAME; one never introduced by `\|switch\|` details is CREATED with `species = nickname` | `poke_env/battle/abstract_battle.py:383-428` | **AMBIGUOUS** (nickname ≠ species) | reached only by a line naming an unrevealed bench mon (Heal Bell `-curestatus\|p2: Nick`); 0 in the corpus |
| 8 | active species of a side | `gen3_battle.py:466-470` | DERIVABLE (state) | |
| 9 | HP fraction = the viewer's rendering: own exact, opponent `ceil%` | `gen3_battle.py:472-474`; `bridge.rs::hp_percent` | DERIVABLE — **R6** (presentation) | fires on every opponent HP event |
| 10 | MOVE `move_id` from the line, never `last_move` | `gen3_battle.py:618` | DERIVABLE | |
| 11 | MOVE target = explicit ident, else the OTHER side's active | `gen3_battle.py:532-539` | **AMBIGUOUS** for an empty / `[still]` target — **R4** | the sim's truth is the real target (the user for Refresh / Rest) |
| 12 | MOVE `target_status` sampled PRE-line | `gen3_battle.py:541` | DERIVABLE (state) | not compared by the spike (NOT CAPTURED) |
| 13 | MOVE `from_move`: `[from]` minus `item:`/`ability:`, `move:` stripped, the SAME move (Pursuit) and `lockedmove` excluded | `battle_event.py:40-76`, `gen3_battle.py:505-519` | **AMBIGUOUS** — gen3's bare `[from] Sleep Talk` / `[from] Pursuit` / `[from] lockedmove` share one shape | exercised: Sleep Talk 6, Snatch 2, Pursuit 93; lockedmove 0 |
| 14 | `\|move\|…\|[miss]` / `[notarget]` also emits a SYNTHETIC MISS / FAIL (`from="move-suffix"`) | `gen3_battle.py:272-274,488-502` | DERIVABLE — **R5** | one miss is TWO MISS events |
| 15 | SWITCH / DRAG: side, species from DETAILS | `gen3_battle.py:631-635` | DERIVABLE | |
| 16 | FAINT | `:638-639` | DERIVABLE | |
| 17 | DAMAGE / HEAL: `amount = after − before` (viewer rendering), `reason = [from]`; **`[of]` dropped** | `:642-652` | DERIVABLE; the reading LOSES `[of]` | 2,608 `[of]` lines in the corpus (Recoil 2,099, Leech Seed 391, drain 113, Volt/Water Absorb 5) |
| 18 | SETHP `hp`, `amount`, `reason` | `:655-665` | DERIVABLE | |
| 19 | BOOST / UNBOOST / SETBOOST: stat, signed amount | `:668-674` | DERIVABLE | outside the spike's subset |
| 20 | CLEARBOOST `op` = keyword (seven collapse into one kind) | `:675-679` | DERIVABLE | outside the subset |
| 21 | STATUS `status`, `reason` | `:682-686` | DERIVABLE | |
| 22 | CURESTATUS `status = sm[3]`; for `-cureteam` sm[3] is `[from] move: Aromatherapy` | `:687-691` | DERIVABLE — **R8** (a quirk) | 174 events |
| 23 | CANT: an ability-sourced cant is filed on the HOLDER; `[of]` names the blocked mon | `:694-712` | **AMBIGUOUS** without the rule (`gen3_damp_cant_v1`) | 0 Damp cants in the corpus |
| 24 | CRIT / MISS / FAIL owner = the last `\|move\|` line's side, reset only at `\|turn\|` | `abstract_battle.py:711,1634`; `gen3_battle.py:716` | **AMBIGUOUS** — **R1** | the sim's truth differs on a mid-turn switch-in's Intimidate → Clear Body `-fail` |
| 25 | MISS / FAIL target = the ident at index 2 — **for `-miss` that is the USER** | `gen3_battle.py:717,727` | DERIVABLE — **R3** | |
| 26 | IMMUNE / RESISTED / SUPEREFFECTIVE owner = mover, else the side OPPOSITE the defender | `:733-738` | **AMBIGUOUS** — **R2** | fired 0 times in the corpus |
| 27 | IMMUNE / RESISTED / SUPEREFFECTIVE carry only the multiplier; `[from] ability:` DROPPED | `:744-749` | DERIVABLE — **R7** | 30 events |
| 28 | FAIL keeps a real `[from]` so Intimidate-blocked is distinguishable | `:718-729` | DERIVABLE | |
| 29 | ITEM / ENDITEM: `item`, `[from]`/`[of]` merged | `:752-754` | DERIVABLE | outside the subset |
| 30 | ABILITY: `ability`, `op` reveal/end, `[from]`/`[of]` | `:757-763` | DERIVABLE | outside the subset |
| 31 | WEATHER: `weather`; permanent iff `[from] ability`; `[upkeep]` continues; `none` clears | `:766-768,392-409` | DERIVABLE | outside the subset |
| 32 | FIELD / SIDE (side from the side ref) | `:771-784` | DERIVABLE | outside the subset |
| 33 | VOLATILE_START / END / ACTIVATE: `effect = sm[3]`, `op`, `[from]`/`[of]` | `:787-792` | DERIVABLE | outside the subset |
| 34 | PREPARE / MUSTRECHARGE / TRANSFORM / FORMECHANGE / SWAP | `:795-815` | DERIVABLE | outside the subset |
| 35 | `_parse_from`: the LAST `[from]` / `[of]` token wins | `:477-486` | DERIVABLE | |
| 36 | the view-side `ViewEventFolder` light board (the five facts) | `event_fold.py` | DERIVABLE | its move-owner reset is MISSING — §3 F2 |

**Count: 6 of 36 rules are AMBIGUOUS** (#7, #11, #13, #23, #24, #26), **17 %** (#3 is derivable but per-viewer);
of those, **three fired in the corpus** (#11 4,816×, #13 101×, #24 24×, and #7/#23/#26 never). Every
AMBIGUOUS rule is itself a TEXT rule (poke-env is a text parser), so a parser that must reproduce
the READING can always do it. What text alone cannot always give is the SIM'S TRUTH — and on the
one field the spike measured for that (#24) it could (§2b parse-back).

### (b) The prototype and the diff

**Mechanism (Rust, off the production path).** A cargo feature `event_spike` (OFF by default —
no production binary is built with it) adds a `Sink` to `ProtocolBuilder`, the single funnel every
omniscient line passes through: each TYPED emit method stages a record from its own typed
arguments (`MonRef`, `Cause`, `HpStatus` integers), and `push_raw` — the one place a line is
appended — commits exactly one record per line, typed or not. The one fact the builder cannot see,
which ACTION the engine is running, comes from the turn loop's `QAction` dispatch (+ Pursuit's
nested strike). The `event_spike` binary replays a recorded battle through the SAME
`BridgeSession` the production `sim_bridge` drives and **refuses** unless its record count equals the
line count. Typed: switch, drag, move (+ its `[from]`, announce `[miss]`, retro `[miss]`/`[still]`),
faint, damage, heal (incl. the two raw `[silent]` heals), crit, supereffective, resisted, immune,
miss, fail, status, curestatus / cureteam, cant — **16 of 51 EVENT keywords, 82 % of event lines**.

**The diff** (`event_spike_diff.py`). Each battle is PLAYED for real over the production rust
`sim_bridge` (two seeded random players with `Gen3Battle`, pinned teams, fixed sim seed,
concurrency 1), then REPLAYED through `event_spike` from its `__RECON__` record; **the replayed
per-side chunks were byte-identical to the live ones on all 297 battles** (the harness refuses
otherwise). Per viewer, source events in `BattleEvent` shape are diffed field by field against that
viewer's live log: **raw** (source facts) and **ruled** (source + the named reading rules).

```bash
cargo build --release --bin sim_bridge --manifest-path src/rust_sim/Cargo.toml
cargo build --release --features event_spike --bin event_spike \
    --manifest-path src/rust_sim/Cargo.toml --target-dir src/rust_sim/target/spike
POKESIM_SIM_BRIDGE_BIN=$PWD/src/rust_sim/target/release/sim_bridge \
python3 $R/event_spike_diff.py --battles 150 --key0 0    --tag S0    --out /tmp/spike_key0
POKESIM_SIM_BRIDGE_BIN=$PWD/src/rust_sim/target/release/sim_bridge \
python3 $R/event_spike_diff.py --battles 150 --key0 5000 --tag S5000 --out /tmp/spike_key5000
python3 $R/spike_coverage.py /tmp/spike_key0 /tmp/spike_key5000
```

| seed (keys) | battles ok / refused | per-viewer events compared | **raw** divergences (source vs reading) | **ruled** divergences | unmatched |
|---|---|---|---|---|---|
| 0–149 | 148 / 2 | 142,496 | MOVE.target 2,904 · MISS.target 544 · CURESTATUS.status 108 · IMMUNE.from 26 · FAIL.side 8 · FAIL.actor 8 · (+692 synthetic MISS) | **0** | **0** |
| 5000–5149 | 149 / 1 | 117,286 | MOVE.target 1,912 · MISS.target 432 · CURESTATUS.status 66 · FAIL.side 16 · FAIL.actor 16 · IMMUNE.from 4 · (+494 synthetic MISS) | **0** | **0** |

Every other compared field — `turn`, `side`, `actor`, `move_id`, `from_move`, `amount`, `hp_after`,
`reason`, `status`, CANT's `of_side`/`of_actor` — agreed on every event of both seeds. The three
refused battles are the port's 1,000-turn panic (§3 F1), not divergences.

**The eight READING rules** (each firing = a place poke-env's reading is not the sim's truth):

| rule | what the reading does | fired (both seeds, per viewer) |
|---|---|---|
| R1 | an outcome line belongs to the last `\|move\|` line's side until `\|turn\|` — not to the action the engine runs | 24 (Intimidate → Clear Body `-fail` on a mid-turn switch-in) |
| R2 | an effectiveness line with no open move is owned by the side opposite the defender | 0 |
| R3 | a MISS event's target is the named ident — the USER | 976 visible (+ same-name mirrors, 148 of 692 on seed 0) |
| R4 | a `[still]` / empty-target move targets the FOE's active | 4,816 |
| R5 | `\|move\|…\|[miss]` adds a second, synthetic MISS | 1,186 synthetic events |
| R6 | HP in the viewer's rendering | every opponent HP event (always on) |
| R7 | effectiveness events drop `[from] ability:` | 30 |
| R8 | `-cureteam`'s `status` is the `[from]` clause | 174 |

**Parse-back** (the §3.2 gate, sketched on the subset — prototyped in the PYTHON harness
(`event_spike_diff.parse_back`), not in Rust; M1 ports it). Every typed record's facts (ident → side +
name, HP integers, cause, `[of]`, the move name) re-derived from its own OMNISCIENT line text:
**129,298 / 129,298**. The outcome OWNER (the engine's move scope, or none) re-derived from line
order by a 4-token rule — `|move|X` opens X's move; `|switch|` (any, Baton Pass entries included),
`|turn|`, `|upkeep` and the bare `|` separator close it; `|drag|` does not: **12,850 / 12,850**.
**So the one attribution the protocol does not print is still recoverable from it** — poke-env's R1
is simply a coarser reset than the text supports.

**Coverage — what the corpus did NOT exercise** (`logs/spike_coverage.log`): Damp's `[of]` cant (0),
a slot-less future-move `-miss` (0), a Heal Bell bench `-curestatus` (0), `[from] lockedmove` (0),
R2 (0). A rule that never fired was not tested, only not contradicted — the program's M1 gate adds
the 22-scenario protocol corpus for exactly these.

### (c) Verdict on §3.4's hazard

**TRACTABLE, and the hard part is not where the design put it.** On the 82 % of event lines the
spike typed: attribution at the source is a ~40-line hook on one funnel (conservation held on every
battle); reading it back from text needs no information the text lacks; 6 of 36 attribution rules
(17 %) are poke-env disambiguation rules, all text-derivable. **The work is the READING**: 8 rules
by which `Gen3Battle`'s log differs from the sim's truth, the reward was trained on that reading,
and the core must emit BOTH (truth fields + the reading projection) until a retrain chooses.

**What surprised** (the one-sided view found four contract changes; this found five):
1. **The event schema is poorer than the source.** It drops `[of]` on 2,608 damage/heal lines, drops
   the ability behind an immunity, names the ATTACKER as a miss's target and the FOE as a `[still]`
   move's target, and writes one miss twice. None of this is visible from inside the Python layer.
2. **The sim's truth for the outcome owner is text-recoverable** — the protocol's `|` separators and
   `|switch|` lines already mark action boundaries; poke-env just does not use them.
3. **`ViewEventFolder` has a latent divergence** (§3 F2) that no gate covers.
4. **The port panics where Showdown ties** (§3 F1) — a random-player corpus reaches turn 1,000 on 1 %
   of battles.
5. **The funnel is nearly complete already**: 17 `push_raw` call sites bypass the typed methods in
   40k lines of engine, and the byte gates mean a typed rewrite of each is mechanically checkable.

**Should the spike code land?** Recommend **yes, as-is, behind its feature**, as M1's starting point:
it is compile-time absent from every production binary, `cargo test` is green with and without it,
and its harness is the first slice of the program's parity gate. The alternative (delete, rebuild
at M1) costs a re-derivation of the 8 rules.

---

## 3. FINDINGS — hazards, skips and what could not be verified

| id | finding | status |
|---|---|---|
| **F1** | The port PANICS at 1,000 committed turns (`src/rust_sim/src/turn/driver.rs:808`, `BATTLE_TURN_CAP`); the pinned Showdown **TIES** at turn > 1000 and emits `\|bigerror\|` auto-tie warnings from turn 500 (`deps/pokemon-showdown/sim/battle.ts:1836-1848`). **3 of 300** random-player battles reached it (`key51`, `key138`, `key5049`). Training forfeits far earlier, so no production path reaches it; a fuzz corpus does | MEASURED; a port parity gap |
| **F2** | **`ViewEventFolder._apply("turn")` does not reset `_current_move_user_side`**, which poke-env's `end_turn` does (`abstract_battle.py:1634`). Reproduced on a constructed 7-line protocol (a start-of-turn switch-in whose Intimidate is blocked by Clear Body): the live log reads the `-fail` as `side=None`, the fold — seeded at depth 1 across a `\|turn\|`, and after `branch()` at depth 2 — reads `side=ours, actor=gyarados`. Reachable wherever one fold spans a `\|turn\|` and a start-of-turn outcome line follows: depth ≥ 2, and `SearchConfig.max_depth` defaults to **3**. `event_fold_parity_fuzz_test` seeds at depth 1 only, so no gate sees it | REPRODUCED (constructed); its production RATE is UNMEASURED |
| F3 | `view_adapter read-models` measured 0.65 ms/arm vs 0.29 on 2026-09-22 (§1a); may be GC billed to the row | UNVERIFIED |
| F4 | Opponent-inference cores are an UPPER BOUND (every opponent a network); the live arm's opponent mix was not read | UNVERIFIED mix |
| F5 | 445 steps/s is a wall average from two checkpoint mtimes (one interval) | single interval |
| F6 | The design's §1 "~89 % gradient, ~11 % rollout" is inconsistent with the runbook's +33 % end-to-end FPS from the opponent compile alone; not resolved here | UNRESOLVED |
| F7 | `target_status` (MOVE) and every kind outside the 16 typed keywords were NOT compared | SKIPPED — scope |
| F8 | The spike corpus is seeded-RANDOM players; a trained policy visits different states (Damp, Future Sight, Heal Bell never occurred) | SKIPPED — M1's gate adds scenario + policy battles |
| F9 | The spike hooks are feature-gated edits to production-crate files (`protocol.rs`, `turn/driver.rs`, `turn/switch.rs`, `turn/residuals.rs`, `turn/status_moves.rs`, `lib.rs`, `Cargo.toml`). Default-feature `cargo test`: **766 passed, 0 failed, 4 ignored**; the production `sim_bridge` was rebuilt after the edits and the spike binary's per-side chunks matched it byte-for-byte on 297 battles (and matched the PRE-edit build on the first run) | VERIFIED; the orchestrator decides whether engine-crate edits may land at all |
| F10 | The 2026-09-22 profile's expand_many split (30.2 % rust) re-measures at **29.9 %** after side elision — the elision did not move the rust share | MEASURED |

---

## 4. Files

| file | what |
|---|---|
| `run_bench.py` | runs a benchmark script with a stub for the one function it borrows from a `__main__` module |
| `aggregate_baseline.py` | §1a's tables from `logs/decision_*` + `logs/expand_many_split_*` |
| `opponent_inference_bench.py` | §1b |
| `inventory_loc.py` | §1d |
| `event_spike_diff.py` | §2b — play, replay through `event_spike`, diff raw + ruled, parse-back |
| `spike_coverage.py` | §2b's coverage census |
| `event_spike_report_key{0,5000}.json` | the two seeds' full reports (tables, examples, the rule texts) |
| `logs/` | every run's raw output with its load line; `cargo_test_*.log` |
| `src/rust_sim/src/event_spike.rs`, `src/rust_sim/src/bin/event_spike.rs` | the spike (feature `event_spike`) |

---

## 5. Ready-to-append ledger paragraph

> **2026-09-23 — RUST CORE PROGRAM, PHASE 0: the baseline is banked, and §3.4's "hardest piece" is
> TRACTABLE — the work is the READING, not the parser.** Record:
> `designs/research_state/measurements/rust_core_phase0_2026-09-23/`; plan:
> `designs/endstate/program_rust_core.md`. **BASELINE** (CPU, load 14–37 on 16 cpus, every number an
> interleaved ratio): a searched decision on the view road is **143 ms wide B / 23.5 ms at B = 1**
> (medians of 6 one-road-per-process rounds; view ÷ protocol 0.711 / 0.834); **Python glue +
> trackers + folds are 59.4 % of its wall (66.3 % with the protocol/event folds), the rust engine ≈
> 4 %** — the design's §7 decision point reads licensed on its own terms (a record only; the owner
> licensed the program on unification). A compiled B = 1 opponent forward is **2.4–4.6 ms** on CPU
> (6.5–7.3× over eager) and a B = 48 batch 0.9–1.3 ms/row, so at the live arm's 445 steps/s opponent
> inference is **0.4–2.1 cores of 16** (upper bound) — Tier 2's case is unification and the
> prerequisite for N envs per process, not freed CPU. Obs build 0.15–0.16 ms production shape;
> `LiveView` 0.11 ms, 723 calls/build. The Python chain the program retires is 28,910 LOC in 11
> links. **SPIKE**: a feature-gated sink on `ProtocolBuilder` (OFF in every production binary)
> emits typed events at the source for 16 of 51 EVENT keywords (82 % of event lines); on **297 real
> bridge battles × 2 viewers, two seeds**, replayed byte-identically from `__RECON__`, the source
> events equal `Gen3Battle`'s reading on **259,782 comparisons with 0 residual and 0 unmatched once 8
> named reading rules are applied** (R1 outcome owner = last `|move|` until `|turn|`, 24 firings; R3
> a MISS's target is the user, 976; R4 a `[still]` move targets the foe, 4,816; R5 one miss = two
> MISS events, 1,186; R7 immunity drops its ability, 30; R8 `-cureteam`'s status is its `[from]`,
> 174; R2 0; R6 HP presentation always). **Parse-back**: 129,298/129,298 typed facts re-derived from
> line text, and the sim's truth for the outcome OWNER re-derived from line order on 12,850/12,850
> lines with a 4-token reset rule. 6 of 36 attribution rules (17 %) are poke-env disambiguation
> rules, all text-derivable. **FINDINGS**: the port PANICS at 1,000 committed turns where Showdown
> ties (3/300 random battles); `ViewEventFolder` never resets the move owner at `|turn|` (reproduced
> on a constructed protocol; reachable at depth ≥ 2, default `max_depth` 3; no gate covers it); the
> event schema drops `[of]` on 2,608 damage/heal lines and the ability behind an immunity. `cargo
> test` 766/0/4 with the spike present.
