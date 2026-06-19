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
| `pokemon_data_extractor/sync.py` | poke-env pokedex + static moves/natures/`learnset.json` + `GenData` type chart; Showdown `abilities.ts` / `items.ts` | `data/pokemon/gen3_{species,moves,abilities,items,type_chart,natures,learnset}.json` |
| `smogon_stats_downloader/sync.py` | Smogon monthly chaos JSON (12-month window) | `data/pokemon/gen3_smogon_stats.json` |
| `smogon_stats_downloader/compute_priors.py` | the aggregated stats + pokedex | `data/pokemon/gen3_{ability,hidden_power}_priors.json` |
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

## Reproducibility is tested

`src/agents/gen3_data/extractor_parity_test.py` re-runs the builders and asserts they reproduce
the committed `data/` files (and that type-chart/natures still equal their poke-env source). A
hand-edit that drifts a committed file from what the extractor produces fails there. After editing
a builder, regenerate (`sync.py`) and run the obs golden
(`training/gen3_data_obs_parity_integration_test.py`) — a value change there is retrain-class.

Building items/abilities needs the Showdown submodule initialized
(`git submodule update --init`); type-chart/natures/species read only poke-env static data.
