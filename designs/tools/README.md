# `designs/tools/` — the acquisition layer's detail

The detail lifted out of `tools/CLAUDE.md` (2026-10-10). Always-current: each doc OWNS its topic and is
updated in the same pass as the tool it describes. The leaf keeps the map, the commands and the rules;
`src/agents/gen3_data/CLAUDE.md` owns the per-file schema of what lands in `data/pokemon/` and the facade
that reads it. The original leaf is frozen at `designs/research_state/claude_md_archive/tools_CLAUDE_2026-10-10.md`.

| doc | holds |
|---|---|
| [`pokemon_data_extractor.md`](pokemon_data_extractor.md) | per-builder notes for `pokemon_data_extractor/sync.py`: the type chart's rule, item `num` vs `spritenum`, the species FORMES and why a `num` rule can't select them, the item / ability / move MECHANICS fields the Rust port reads and their drift gate, the typed Hidden Power `num`, the move aliases |
| [`smogon_priors.md`](smogon_priors.md) | the chaos-record denominator W, the 12-month merge, the format-spec filter, and the throwing checks before a write |
| [`team_downloaders.md`](team_downloaders.md) | the one-entry-per-team manifest invariant and its history, the `main.promote_teams` second writer, the named encodings and the committed mojibake |
