# `upstream/poke_env_static/`: a FROZEN acquisition input

The static JSON that poke-env shipped (`pokedex/gen{N}pokedex.json`, `moves/gen{N}moves.json`,
`typechart/gen{N}typechart.json`, `learnset.json`, `natures.json`), itself a dump of Pokémon Showdown's data,
copied byte-for-byte before the vendored fork was deleted. It is DATA, not code: `sync.py` reads it with plain
`json.load`, and the type chart is computed by `sync.build_type_chart` from `typechart/gen{N}typechart.json`
(damageTaken 0/1/2/3 means 1 / 2 / 0.5 / 0).

It is one of the acquisition layer's upstreams (`tools/CLAUDE.md`). A change to it is a DATA change: regenerate with
`python tools/pokemon_data_extractor/sync.py --out-dir <TEMP>`, diff against `data/`, and land the `data/` change
deliberately, never while a run pinned before `gen3_pinned_data_root_v1` is live (root rule 6).
