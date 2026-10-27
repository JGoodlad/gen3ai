# `designs/tools/` — the acquisition layer's and the read tooling's detail

The detail behind `tools/CLAUDE.md` and `src/agents/gen3_data/CLAUDE.md`, and the design of record of `src/main/reads/`.
Always-current: each doc OWNS its topic
and is updated in the same pass as the tool or facade module it describes. The leaves keep the map, the commands
and the rules. The pre-split tools leaf is history at
`designs/research_state/claude_md_archive/tools_CLAUDE_2026-10-10.md`.

| doc | holds |
|---|---|
| [`data_files.md`](data_files.md) | the CONSUMER-side schema of every file in `data/pokemon/` and the archetypes file, each with the facade accessor that reads it and the load-time checks |
| [`pokemon_data_extractor.md`](pokemon_data_extractor.md) | per-builder notes for `pokemon_data_extractor/sync.py`: the type chart's rule, item `num` vs `spritenum`, the species FORMES and why a `num` rule can't select them, the item / ability / move MECHANICS fields the Rust port reads and their drift gate, the typed Hidden Power `num`, the move aliases |
| [`smogon_priors.md`](smogon_priors.md) | the chaos-record denominator W, the 12-month merge, the format-spec filter, and the throwing checks before a write |
| [`team_downloaders.md`](team_downloaders.md) | the one-entry-per-team manifest invariant and its history, the `main.promote_teams` second writer, the named encodings and the committed mojibake |
| [`replay_corpus.md`](replay_corpus.md) | the public gen3ou replay corpus under `<main>/replays/` (NOT `data/`): how it grows, why its collector stopped, the supervision and the freshness alarm |
| [`reads.md`](reads.md) | `main.reads`, the ONE module for registered reads: the spec, the settled-file rule, the anchor and S lanes (one supervisor per state path that records its own pid and honors a hold, cross-state import, the Metamon anchors), the regime id always recorded (the engine in it, and the per-read engine table), the VOID gate, the registered rules by name, the record schema, the golden reproductions of three shipped reads, and the per-read scripts and bug classes it replaced |
| [`convergence_2026-10-26.md`](convergence_2026-10-26.md) | the ops and read tooling's duplicated helpers (ports, flocks, liveness, process identity, hashing, records, ship): the overlap map, the canonical copy of each, the target modules, what landed and the staged plan |
