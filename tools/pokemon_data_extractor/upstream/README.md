# `upstream/poke_env_static/` — a FROZEN acquisition input (P6 of the poke-env retirement, 2026-10-08)

The static JSON poke-env ships (`pokedex/gen{N}pokedex.json`, `moves/gen{N}moves.json`, `typechart/gen{N}typechart.json`,
`learnset.json`, `natures.json`) — itself a dump of Pokémon Showdown's data — copied byte-for-byte from the vendored
fork's `src/poke_env/data/static/` (last changed at `cbe6148e`) before the fork was deleted from this tree. It is DATA,
not code: `sync.py` reads it with plain `json.load`, and the type chart that `GenData.type_chart` used to compute is
now computed by `sync.build_type_chart` from `typechart/gen{N}typechart.json` with the same rule (damageTaken 0/1/2/3 →
1 / 2 / 0.5 / 0).

**Proof it changes nothing:** before and after the move, `python tools/pokemon_data_extractor/sync.py --out-dir <TEMP>`
regenerated every `data/pokemon/gen3_*.json` file byte-identical to the committed one (abilities, moves, species,
items, type chart, natures, learnset, move aliases).

It is one of the acquisition layer's upstreams (`tools/CLAUDE.md`). A change to it is a DATA change: regenerate
into a temp dir, diff against `data/`, and land the `data/` change deliberately — never while a pinned run is live.
