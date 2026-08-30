# CLAUDE.md — Acquisition tools (`tools/`)

`tools/` is the **acquisition layer**: the *only* place that knows the three upstreams (poke-env
static data, the Showdown `.ts` source tree, Smogon usage stats). Each tool *derives* and
normalizes its upstream into committed files under `data/`, so the derivation is reproducible
rather than a one-off hand edit. The runtime never imports `tools/`; it reads `data/` through the
`agents.gen3_data` facade (see `src/agents/gen3_data/CLAUDE.md`). This is the acquisition-vs-access
split.

## The tools

| Tool | Upstream | Output |
|---|---|---|
| `pokemon_data_extractor/sync.py` | poke-env pokedex + static moves/natures/`learnset.json` + `GenData` type chart; Showdown `abilities.ts` / `items.ts` / `aliases.ts` | `data/pokemon/gen3_{species,moves,abilities,items,type_chart,natures,learnset,move_aliases}.json` |
| `smogon_stats_downloader/sync.py` | Smogon monthly chaos JSON (12-month window) | `data/pokemon/gen3_smogon_stats.json` |
| `smogon_stats_downloader/compute_priors.py` | the aggregated stats + pokedex | `data/pokemon/gen3_{ability,hidden_power,move,item,spread,teammate}_priors.json` |
| `sample_team_downloader/sync.py` | Smogon forum sample-team thread | `data/teams/sample/` |
| `others_team_downloader/sync.py` | PokePaste dumps | `data/teams/others/` |

## `pokemon_data_extractor` — the reference-data extractor

One `--datasets` entry per file (`all` rebuilds every file); each builder is registered in
`_BUILDERS`:

```bash
python tools/pokemon_data_extractor/sync.py                          # all, gen 3, write files
python tools/pokemon_data_extractor/sync.py --datasets moves species # subset
python tools/pokemon_data_extractor/sync.py --gen 3 --stdout         # print, don't write
```

Notes that bite:
- **`build_type_chart`** dumps `GenData.from_gen(gen).type_chart` directly, so the committed JSON
  is byte-identical to what `gen3_mechanics` used to read live — effectiveness is unchanged.
- **`build_items`** stores the **item-dex `num`** under our schema's `num` field. The regex is
  `\bnum:` so it matches the item number and NOT `spritenum:` (the sprite index, which appears
  earlier in the block — e.g. Leftovers spritenum=242 vs num=234). Items with no positive item-dex
  num (e.g. Berserk Gene, a removed Gen-2 item) are dropped — not real gen-3 items. Cross-gen
  aliases legitimately share a num (Sitrus/GoldBerry, Silk Scarf/Pink Bow). (Historical note: the
  data once used spritenum; switching to the true num was the `gen3_item_num_fix_v1` retrain.)
- Gen filtering uses per-gen `num` ceilings (`_GEN_MAX_*`) so post-gen-3 abilities/species/items
  are excluded.
- **`build_species`** emits every BASE form **plus the gen-N-legal ALTERNATE FORMES**
  (`gen3_species_formes_v1`): for gen 3 the curated `_GEN_ALT_FORMES` entries — **Deoxys-Attack /
  -Defense / -Speed** (team-legal, each with its OWN base stats) and **Castform-Sunny / -Rainy /
  -Snowy** (`battleOnly`, same stats, retyped) — plus every **cosmetic** forme a kept base declares
  (`cosmeticFormes`: the **27 Unown letters**, which have NO pokedex entry and are synthesized by
  Showdown's `dex-species.ts` as a CLONE of the base with only name/forme/baseSpecies changed, so
  the extractor clones them identically). 386 → **419 rows**. The builder previously dropped EVERY
  non-base forme, which **cost the `src/rust_sim` port 6.6% of gen3 random-battle TEAMS / ~14% of
  battles** (`MonState::from_set`: `unknown species "Deoxys-Speed"`) — the single largest
  team-construction failure cause, and a DATA gap, not an engine one.
  **Why the blanket filter existed, and why a `num` rule can't replace it:** poke-env's static
  `gen{N}pokedex.json` is NOT gen-filtered by forme — it carries **135 formes with a gen-3 `num`**
  (Megas / Gmax / Alolan / Galarian / Hisuian / Paldean / Pikachu cosmetics / Totems), all
  post-gen-3, and its own `gen` field is present on only 31 of 140 forme entries. The ORACLE is
  therefore the MOD-CHAIN-resolved `Dex.mod('gen3')` (`exists && !isNonstandard && gen <= 3`; the
  gen3 mod marks every later forme `isNonstandard: 'Future'`), which returns exactly the six above —
  hence a curated table, the `_GEN3_ITEM_MECHANICS` precedent, **drift-gated by
  `node src/rust_sim/harness/dump_gen3_mechanics.js --check`** (its SPECIES section asserts the
  committed file equals the resolved gen-3 universe field-for-field).
  Every forme row carries **`baseSpecies`** (+ `battleOnly` when applicable), and that field is the
  **load-bearing safety contract**: a forme SHARES its base's national-dex `num`, and `num` is what
  the obs species channel and every `table[species.num]` model buffer are keyed by — so a
  num-indexed consumer MUST iterate `gen3_data.species.base_form_ids()`, or the last forme row
  written silently redefines the base (Deoxys-Speed's stats landing on num 386, Castform-Sunny's
  FIRE on num 351). Adding the rows was proven value-neutral: 33 pure insertions, all 386
  pre-existing rows byte-identical, all 49 species-num-indexed tensors bit-identical.
  `build_learnset` filters to base forms for the same reason (a forme has no learnset of its own —
  poke-env ships `deoxysattack` with an EMPTY movepool, and an empty legality set would make the
  move-belief gate prune every candidate), so `gen3_learnset.json` is unchanged.
- **`build_items`** ALSO carries the **`gen3_item_mechanics_v1` structured mechanics fields**
  (only-when-present, obs-neutral like `critRatio` — the obs encodes items by per-id `num`
  lookup, so both new FIELDS and the 4 new ENTRIES change no existing value): `typeBoost
  {type, mod:[num,den], fold: stat|basePower|basePowerDirect}`, `statMods {stat:[num,den]}` +
  `onlySpecies` + `untransformedOnly`, `choice`, the declaratively-parsed `isBerry`, and the
  **`gen3_accuracy_pipeline_v1` `accMod`** (Bright Powder / Lax Incense — `{op:multiply, mod:0.9,
  side:defender}`; the exact float is stored verbatim so Rust + JS parse identical f64 bits). The
  mechanics come from the curated `_GEN3_ITEM_MECHANICS` table (item handlers are JS callbacks —
  invisible declaratively, the `_CURES_SELF_STATUS` precedent) **derived from the RESOLVED
  `Dex.mod('gen3')`** by `src/rust_sim/harness/dump_gen3_mechanics.js` — NEVER from a single
  `.ts` file (the mod-chain law: gen4 REWRITES Light Ball to an onBasePower double, gen3
  rewrites it again to SpA-ONLY ×2). Every regeneration is drift-gated by
  `node src/rust_sim/harness/dump_gen3_mechanics.js --check` (committed JSON vs the resolved
  dist). `_GEN4_ITEMS_APPLIED_IN_GEN3` (odd/rock/rose/wave incense) is a documented exception
  to the gen filter — gen4-named items the sim still applies under gen3 formats, in the
  e2e/fuzzer `MODELED_ITEMS`, priced by the `src/rust_sim` port (×4915/4096 at base power —
  probe-settled, NOT ×1.1). Item blocks are bounded at the NEXT item header (not a fixed
  5000-char window) so per-block flags can't be read from the following item. Consumed by
  `src/rust_sim/src/dex/items.rs` → the engine's generic per-class damage folds.
  **`build_items` ALSO carries the `gen3_ability_batch4_v1` PROC_ITEM rows** (only-when-present,
  obs-neutral): `flinchSecondary {chance: 10, moves: [...]}` on King's Rock — the EXECUTION-derived
  130-id gen3 move list (the dump derives it by CALLING the resolved `onModifyMove` against every
  gen≤3 move; the 17 typed Hidden Powers dedupe to the one sim id `hiddenpower`) — and
  `surviveLethal {chance: [1, 10]}` on Focus Band. Both drift-gated by `dump_gen3_mechanics.js
  --check` and consumed by the `src/rust_sim` port (`ItemData::{flinch_secondary, survive_lethal}`).
- **`build_abilities`** ALSO carries **`dmgMod`** + **`accMod`** (only-when-present, obs-neutral)
  from the curated `_GEN3_ABILITY_MECHANICS`. `dmgMod` — the pinch family (Torrent/Blaze/Overgrow/
  Swarm: BP ×1.5 at hp≤⅓), Huge/Pure Power (Atk ×2), Guts (Atk ×1.5 statused), Marvel Scale (Def
  ×1.5 statused), Hustle (Atk ×1.5 DIRECT), Thick Fat (sourceBasePower ×0.5 Ice/Fire) — **CONSUMED
  by the `src/rust_sim` ability DMG_MOD class** (Phase 2 — `dex/abilities.rs` → `AbilityData.dmg_mod`,
  the engine's `resolve_atk_stat_mods` / `resolve_def_stat_mods` / `resolve_bp_mods` fold it). `accMod`
  (`gen3_accuracy_pipeline_v1`) — the ACCURACY class: Compound Eyes (`{op:chain, mod:[13,10],
  side:attacker}` ×1.3), Sand Veil (`chain [8,10] defender weather:sandstorm` ×0.8-in-sand), Hustle
  (`chain [3277,4096] attacker physicalTypesOnly` ×0.8 physical-TYPE) — **CONSUMED by the port's to-hit
  fold** (`dex/accmod.rs::AccMod` → `turn.rs::effective_accuracy`); Hustle now ships FULLY (its Atk ×1.5
  pairs with its acc ×0.8), off the DATA-ONLY list. `build_abilities` ALSO carries **`statusImmune`**
  (`gen3_status_immune_v1`, only-when-present, obs-neutral) — `{statuses: [par|slp|psn|tox|brn|frz, …],
  phase: "setStatus"|"immunity"}` for the STATUS_IMMUNE class: Limber (par) / Insomnia + Vital Spirit (slp)
  / Immunity (psn,tox) / Water Veil (brn) block via `onSetStatus` (phase=setStatus); Magma Armor (frz)
  blocks via `onImmunity` (phase=immunity, BEFORE the SetStatus event). **CONSUMED by the port's
  `try_set_status`** (`dex/abilities.rs` → `AbilityData.status_immune`), the `phase` selecting whether the
  block gates before/after the gen3ou SetStatus clause shuffle (probe-settled — Own Tempo/Oblivious block a
  VOLATILE not a status, so they carry no `statusImmune`). `build_abilities` ALSO carries **`contactAttract`** (`gen3_ability_batch4_v1`, Cute Charm —
  `{chance: [1, 3]}`: the same DamagingHit-position contact roll as `contactProc`, but on a pass it
  adds the ATTRACT volatile to the attacker; the gender gate lives inside the volatile's onStart).
  Same dump-derivation + `--check` drift gate as the
  items (the drift gate now pins `dmgMod`/`accMod`/`statusImmune`/`contactProc`-family/
  `contactAttract` on abilities and the full mechanics-field set — incl. `flinchSecondary`/
  `surviveLethal` — on items).
- **`build_moves`** (gen3_unified_move_system_v1) extracts the structured SECONDARY effects — reversing
  the old "secondary status is incidental" decision. `_secondary_effects` normalizes a move's
  `secondary`/`secondaries` into `secondaryEffects = {col: percent}` over the 10 `_SECONDARY_COLS`
  (`secondary.status`→its column, `volatileStatus`→flinch/confusion, foe `boosts`→`foe_statdrop`,
  `self.boosts`→`self_boost`; Tri Attack's `onHit` is a curated `_SECONDARY_ONHIT` split like Belly Drum),
  plus `priority`, `drainFraction`, `recoilFraction`. These are **GPU-side only** (the DamageOperator +
  MoveLatentEncoder read them) — they do NOT enter the obs vector, so the obs golden is unchanged.
  `build_moves` ALSO **overrides the Hidden Power `num`** (`gen3_typed_hidden_power_ids_v1`): Showdown
  ships all 17 HP ids at `num=237`, but the bare `hiddenpower` keeps 237 while the 16 typed variants are
  re-numbered to distinct **355-370** from the deterministic module-level `_HP_TYPE_NUMS` map (alphabetical
  by type, so the extractor-parity test reproduces the file). This lets OUR known-HP be represented by the
  move embedding itself (the opponent's unrevealed HP stays the typeless 237); it IS an obs-value change
  (our HP move-id channel) → retrain-class, golden regenerated. See `src/agents/model/CLAUDE.md` →
  `gen3_typed_hidden_power_ids_v1` and `designs/ai_v6/design_typed_hidden_power_ids.md`.
- **`build_moves`** also carries **`critRatio`** (only when present — the ~dozen gen-3 high-crit moves
  Slash/Crabhammer/Aircutter/Blaze Kick/Leaf Blade/…; absent ⇒ ratio 1). It is **NOT in the obs vector**
  (the facade ignores it, like the secondary fields), so it is obs-neutral; it exists for the `src/rust_sim`
  Rust port's gen-3 crit ratio (1/16 vs 1/8), data-driven rather than a hardcoded move list.
- **`build_moves`** also carries **`pp`** + **`noPPBoosts`** (always-present, `gen3_pp_tracking_v1`) — the
  move's BASE PP + the no-PP-ups flag. **NOT in the obs vector** (the facade ignores them, like `critRatio`),
  so obs-neutral; they exist for the `src/rust_sim` Rust port's PP tracking + forced-Struggle layer, which
  computes a moveslot's in-battle MAX PP as `calculatePP(move, 3) = pp * 8/5` (the `Pokemon` ctor's default
  3 PP-ups) for a normal move, or the raw `pp` for a `noPPBoosts` move (Struggle = 1).
- **`build_moves`** ALSO carries **`secondaryBoosts`** (only-when-present, like `critRatio`) — the
  STRUCTURED stat-boost spec the flat `secondaryEffects` `{col:percent}` discards. `_secondary_boosts`
  walks `secondary` + every `secondaries[i]`, emitting `{chance, target:"foe", boosts}` for a foe
  stat-DROP (`secondary.boosts` — Crunch `{spd:-1}`, Rock Tomb `{spe:-1}`) and `{chance, target:"self",
  boosts}` for a self stat-RAISE (`secondary.self.boosts` — Meteor Mash `{atk:1}`, Ancient Power all 5).
  It carries the `(stat, stages, foe-vs-self)` the flatten loses so the **`src/rust_sim` Rust port** can
  apply the real boost (the engine reads `secondaryBoosts` → `apply_secondary_boost`); the file diff is
  just the ~24 boost-secondary moves and it is **obs-neutral** (the facade ignores it, like `critRatio`).
  Tri Attack is NOT encoded here — its `onHit` `sample(['brn','par','frz'])` is a curated
  `_SECONDARY_ONHIT` split for the obs `secondaryEffects`, and the Rust engine special-cases / fail-loud
  guards it directly.
- **`build_moves`** ALSO carries **`selfBoosts`** (only-when-present, like `critRatio`/`secondaryBoosts`)
  — the PRIMARY self-boost spec for a PURE SETUP move (`gen3_setup_moves_v1`). `_self_boosts` emits the
  `{stat:stages}` map for a `target:self` Status move (bp 0) whose ENTIRE effect is its declarative
  top-level `boosts` — Swords Dance `{atk:2}`, Dragon Dance `{atk:1,spe:1}`, Calm Mind `{spa:1,spd:1}`,
  Agility `{spe:2}`, Bulk Up / Amnesia / Tail Glow / the +Def & +Atk one-stat moves (17 in all). It is
  gated to the PURE setup moves: every boosted stat must be a POSITIVE battle stat in
  `_SELF_BOOST_STATS` (atk/def/spa/spd/spe — accuracy/evasion EXCLUDED here; note the parenthetical
  that used to justify this by "the `src/rust_sim` accuracy roll ignores the evasion table" is
  STALE, see `statDropBoosts` below — the live reason is that the +evasion moves are not pure
  declarative self-boosts), and the move must
  carry NO other effect (NO `volatileStatus` → excludes Defense Curl/Minimize, NO `self`/`secondary`,
  NO `onHit`/`onTryHit`/`heal` → excludes Belly Drum's HP cost; Curse is `target:normal` → excluded).
  It exists for the **`src/rust_sim` Rust port** (the engine reads `selfBoosts` → `self_boost_spec` →
  the draw-free `boost()` apply); the file diff is just the ~17 pure setup moves and it is **obs-neutral**
  (the facade ignores it). The e2e fuzz's `MODELED_SETUP_MOVES` is DERIVED from this field so the
  allow-list stays GIGO-proof in lockstep with the engine.
- **`build_moves`** ALSO carries **`selfDrops`** (only-when-present, like `selfBoosts`,
  `gen3_move_coverage_batch1_v1`) — the top-level `move.self.boosts` SELF STAT-DROP on a DAMAGING
  move: Overheat `{spa:-2}`, Superpower `{atk:-1,def:-1}`, Psycho Boost `{spa:-2}`. `_self_drops`
  emits the `{stat:stages}` (all NEGATIVE) map for a damaging move whose `self` block is EXACTLY a
  `boosts` map of self stat-DROPS in the battle stats (no accuracy/evasion, no other `self` key).
  It exists for the **`src/rust_sim` Rust port** (`MoveData::self_drops` → `turn.rs::apply_self_drops`,
  which draws the gen3 `selfDrops` `random(100)` then applies the drop draw-free); the file diff is
  just the handful of self-drop moves and it is **obs-neutral** (the facade ignores it). Rapid Spin's
  `self` block is `{}` (its clear is an `onAfterHit`, not a self-boost) → correctly excluded.
- **`build_moves`** ALSO carries **`statDropBoosts`** (only-when-present, like `selfDrops`,
  `gen3_move_coverage_batch2_v1`) — the declarative FOE STAT-DROP for a standalone stat-drop STATUS
  move: Screech `{def:-2}`, Charm `{atk:-2}`, Metal Sound `{spd:-2}`, Feather Dance `{atk:-2}`, Tickle
  `{atk:-1,def:-1}`, Fake Tears `{spd:-2}`, Cotton Spore / Scary Face `{spe:-2}`. `_stat_drop_boosts`
  emits the `{stat:stages}` (all NEGATIVE) map for a `category:Status`, foe-targeting (`target:normal`)
  move (bp 0) whose ENTIRE effect is its declarative top-level `boosts` of foe drops in
  `_STAT_DROP_STATS` — the battle stats **PLUS `accuracy`/`evasion`** (`gen3_sand_attack_v1`) — with no
  `status`/`volatileStatus`/`self`/`secondary`/`onHit`/`heal`, and NO `selfdestruct` (so Memento is
  excluded). ⚠️ **accuracy/evasion were EXCLUDED until 2026-08-18, and the stated reason — "the
  `src/rust_sim` accuracy roll ignores the evasion table" — was FALSE by then**:
  `gen3_accuracy_pipeline_v1` folds both boosts[5] and boosts[6] through the gen-3 `boostTable` in
  `speed.rs::effective_accuracy`, a path already load-bearing for Mud-Slap. The stale exclusion kept
  four moves fail-loud for nothing — `sandattack` (0.72 of the gen3ou move-slot prior mass, the
  largest single gap), `smokescreen`, `kinesis`, `flash`. Probe: `harness/probe_sandattack.js`.
  The narrower `_SELF_BOOST_STATS` still gates `_self_boosts`/`_self_drops`, correctly but for a
  DIFFERENT reason than the one written there: Double Team is self-evasion and Minimize also carries
  a `volatileStatus`, so neither is a pure declarative self-boost. Relaxing THAT guard is a separate,
  unprobed question — do not assume this change covers it. It exists for the **`src/rust_sim` Rust port**
  (`MoveData::stat_drop_boosts` → the stat-drop arm in `run_status_move`, which draws the accuracy roll
  then applies the drop draw-free via `apply_secondary_boost`); obs-neutral (the facade ignores it).
  Refresh / Heal Bell / Aromatherapy reuse the pre-existing `curesSelfStatus` / `curesTeamStatus`
  flags; the weather/screen ids are pinned in `turn.rs`.
- **`build_moves`** ALSO passes through **`multihit`** + **`multiaccuracy`** (only-when-present,
  `gen3_move_coverage_batch7_v1`) from the poke-env static `gen3moves.json` — the MULTI-STRIKE spec:
  `multihit` is a plain integer (Double Kick / Twineedle / Bonemerang **2**, Triple Kick **3**) or a
  `[2, 5]` array (the variable family Pin Missile / Bullet Seed / Icicle Spear / Rock Blast / Barrage /
  Comet Punch / Double Slap / Spike Cannon / Arm Thrust / Fury Attack / Fury Swipes / Bone Rush);
  `multiaccuracy: true` marks Triple Kick's per-strike accuracy re-roll. Both are **obs-neutral** (the
  `agents.gen3_data` facade ignores them, like `critRatio`/`selfDrops`) and exist for the
  **`src/rust_sim` Rust port** (`MoveData::{multihit, multiaccuracy}` → `turn.rs::run_multihit`: a
  fixed count draws NO count roll, `[2,5]` draws ONE `sample([2,2,2,3,3,3,4,5])`, then each strike runs
  the normal damage path + per-strike secondary; a `multiaccuracy` move FAIL-LOUDS — unmodeled).

## Team downloaders — one manifest entry per team

`sample_team_downloader/sync.py` and `others_team_downloader/sync.py` each write a per-folder
`teams.json` manifest (`{id, name, format, valid, errors, file, source}` rows) next to the `.txt`
team files. **The invariant every manifest must hold: one entry per distinct team (== per `id` ==
per `.txt` file).** The runtime `TeamLoader` (`src/utils/team_loader/`) appends a team's text once
per **entry**, then draws uniformly — so a duplicated entry silently multiplies that team's
training/eval draw weight.

This bit us once: the Yak Attack PokePaste dump names a team once **per Pokémon**
(`"<Mon>/<Team Name>"`, six headers sharing one 6-mon text → one id/file), and the generator didn't
dedupe, so `others/yak_attack/teams.json` had 1122 rows over 185 files (174 valid). That made
yak_attack ~66% of **every** training and eval team draw (pool 1601 over 719 unique). Fix
(`gen3_team_pool_dedupe`): `others_team_downloader/sync.py::collapse_duplicate_teams` collapses rows
by `id` before writing (name strips the `"<Mon>/"` prefix, `valid` = AND over the group, `errors` =
union, idempotent), so a re-run can't reproduce it; the deduped pool is **719 unique teams (32
sample + 687 others)**. `TeamLoader._load_teams` also dedupes by resolved file path as
defense-in-depth (loud warning if a manifest references a file twice). Guards:
`src/utils/team_loader/team_manifest_test.py` (data-contract: one-entry-per-file over all manifests
+ collapse-fn units) and `loader_test.py` (synthetic per-mon dedupe + the 32/687/719 count pin). A
changed team pool is a **data-distribution change** (training *and* eval) — land it at a clean
retrain boundary, never mid-A/B.

### 🚨 Both downloaders NAME their encodings, and the committed data already carries a mojibake

`requests` falls back to **ISO-8859-1** for a `text/*` response with no `charset` (RFC 2616), so
`res.text` on a UTF-8 PokePaste turns `é` into `Ã©` — and the file is then written back as UTF-8,
baking it in. `data/teams/others/mcmegan/*.txt` hold the bytes of `PtÃ©ra` where `Ptéra` was meant,
and that nickname surfaced years later as a `KeyError: 'ptãra'` inside a depth-2 search replay,
where it was filed as a *chunk-transport double-encode* and chased in the wrong subsystem entirely
(the real defect was a missing ply — `gen3_search_depth2_chunk_gap_v1` in `designs/CHANGELOG.md`).
Both syncs now override an ISO-8859-1 guess with `apparent_encoding` and write with an explicit
`encoding="utf-8"` (a bare `open(..., 'w')` used the LOCALE encoding, so the same download was not
even reproducible across machines).

**The already-committed bytes are deliberately NOT rewritten.** A team file is hashed into
`pin_sha` (`MatchupSpec`) and keys `data/teams/gen3_team_archetypes.json`, and the downloaders key
`teams.json` by `sha256(team_text)` — so "fixing" a nickname re-ids the team, orphans its archetype
label and breaks every provenance record that named it. A nickname is cosmetic to the model (the
obs never reads one), so the correction belongs at the next deliberate pool boundary, not as a
tidy-up. `src/main/search_dividend/depth2_replay_integration_test.py` pins the current bytes so a
re-sync that changes them is a visible event rather than a silent re-id.

## Reproducibility is tested

`src/agents/gen3_data/extractor_parity_test.py` re-runs the builders and asserts they reproduce
the committed `data/` files (and that type-chart/natures still equal their poke-env source). A
hand-edit that drifts a committed file from what the extractor produces fails there. After editing
a builder, regenerate (`sync.py`) and run the obs golden
(`training/gen3_data_obs_parity_integration_test.py`) — a value change there is retrain-class.

- **`build_move_aliases`** (`gen3_move_alias_resolution_v1`) emits `data/pokemon/gen3_move_aliases.json`
  — a flat `{alias_id: canonical_move_id}` map parsed from Showdown's `deps/pokemon-showdown/data/aliases.ts`
  (`wisp`→`willowisp`, `sd`→`swordsdance`, `twave`→`thunderwave`, the typed `hpice`→`hiddenpowerice`, …),
  kept only for aliases whose CANONICAL id is a gen-3 move and EXCLUDING the bare `hp`→`hiddenpower`
  (the `src/rust_sim` port represents Hidden Power by its distinct typed move name). It is consumed
  **ONLY by the `src/rust_sim` port** — its dex's `moves()` resolves a packed-team alias exactly like
  Showdown's `dex.moves.get()`, so a team carrying a shorthand token (the sample pool's Gengar writes
  `wisp`) runs the SAME move the sim runs (the e2e_86 draw-count cascade fix). **Obs-neutral**: the
  `agents.gen3_data` facade names its files explicitly and never loads it, so the RL obs is unchanged.

Building items/abilities/aliases needs the Showdown submodule initialized
(`git submodule update --init`); type-chart/natures/species read only poke-env static data.
