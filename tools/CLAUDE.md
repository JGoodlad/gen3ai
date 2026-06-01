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
| `pokemon_data_extractor/sync.py` | poke-env pokedex + static moves/natures + `GenData` type chart; Showdown `abilities.ts` / `items.ts` | `data/pokemon/gen3_{species,moves,abilities,items,type_chart,natures}.json` |
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

## Reproducibility is tested

`src/agents/gen3_data/extractor_parity_test.py` re-runs the builders and asserts they reproduce
the committed `data/` files (and that type-chart/natures still equal their poke-env source). A
hand-edit that drifts a committed file from what the extractor produces fails there. After editing
a builder, regenerate (`sync.py`) and run the obs golden
(`training/gen3_data_obs_parity_integration_test.py`) — a value change there is retrain-class.

Building items/abilities needs the Showdown submodule initialized
(`git submodule update --init`); type-chart/natures/species read only poke-env static data.
