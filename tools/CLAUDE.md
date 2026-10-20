# CLAUDE.md — Acquisition tools (`tools/`)

`tools/` is the acquisition layer, the only code that knows the three upstreams: the poke-env static
JSON (frozen as data under `pokemon_data_extractor/upstream/poke_env_static/`; no poke-env code runs),
the Showdown `.ts` sources (`deps/pokemon-showdown/`) and Smogon usage stats. Each tool derives its
upstream into committed files under `data/`; the runtime never imports `tools/` and reads `data/` through
`agents.gen3_data`. Detail: `designs/tools/README.md` (the index); what each `data/` file holds:
`designs/tools/data_files.md`.

| Tool | Upstream | Output |
|---|---|---|
| `pokemon_data_extractor/sync.py` | the frozen static JSON (pokedex, moves, natures, learnset, typechart); Showdown `abilities.ts` / `items.ts` / `aliases.ts` | `data/pokemon/gen3_{species,moves,abilities,items,type_chart,natures,learnset,move_aliases}.json` |
| `smogon_stats_downloader/sync.py` | Smogon monthly chaos JSON (12-month window) | `data/pokemon/gen3_smogon_stats.json` |
| `smogon_stats_downloader/compute_priors.py` | the aggregated stats + pokedex | `data/pokemon/gen3_{ability,hidden_power,move,item,spread,teammate}_priors.json` |
| `sample_team_downloader/sync.py` | Smogon forum sample-team thread | `data/teams/sample/` |
| `others_team_downloader/sync.py` | PokePaste dumps | `data/teams/others/` |
| `replay_corpus_downloader/sync.py` | the public replay archive | `<main>/replays/showdown/…`, not `data/`; never run against the live server (`replay_corpus_downloader/README.md`); STOPPED since 2026-09-23, status and plan in `designs/tools/replay_corpus.md`; `replay_corpus_downloader.freshness` is the read-only staleness alarm |

`cloud/vast/` is not acquisition: it rents a vast.ai box for the cloud-GPU benchmark
(`designs/ops/cloud_gpu_benchmark.md`, `cloud/vast/README.md`). Its API key and ssh key never leave this
machine; one instance at a time; every create arms the deadman.

## Rules

- **A `data/` write is a data change, under root standing rule 6.** Regenerate into a temp dir
  (`sync.py --out-dir <TEMP>`), diff against `data/`, and land the change deliberately. A changed team
  pool changes the training and eval distribution: land it at a clean retrain boundary, never mid-A/B.
- **Run `compute_priors.py` with `PYTHONPATH=src`**: it imports `agents.gen3_data.format_spec` to drop
  banned mass. Its W denominator and throwing checks: `designs/tools/smogon_priors.md`.
- **`data/teams/sample/` has a second writer, `python -m main.promote_teams`**, and
  `sample_team_downloader/sync.py` rebuilds `teams.json` from the forum thread alone, so a re-sync drops
  every promoted team. Check `data/teams/sample/PROMOTION_MANIFEST.json` first and re-promote from its
  recorded seed afterwards (`designs/tools/team_downloaders.md`).
- **Never "fix" the committed mojibake** in `data/teams/others/mcmegan/*.txt`: a team file's bytes key
  `pin_sha` and the archetypes file, so a rewrite re-ids the team
  (`src/utils/team_loader/committed_team_bytes_test.py`).

## `pokemon_data_extractor`

`sync.py --datasets <name …>` rebuilds a subset (`all` by default); `--out-dir <TEMP>` writes elsewhere;
`--stdout` prints. Items, abilities and aliases need the Showdown submodule; species, moves, learnset,
type chart and natures read only the frozen static JSON. Per-builder notes:
`designs/tools/pokemon_data_extractor.md`.

- **Mechanics come from the resolved `Dex.mod('gen3')`, never a single `.ts` file** (gen 4 rewrites Light
  Ball and gen 3 rewrites it again). The curated item / ability mechanics tables and the forme list are
  derived by `src/rust_sim/harness/dump_gen3_mechanics.js`; run it with `--check` after every
  regeneration.
- **Mechanics fields are obs-neutral; some builder outputs are obs VALUES.** `critRatio`, `pp`, the boost /
  multihit fields, the item / ability mechanics and `gen3_move_aliases.json` carry no observation value (the Rust port and the packed-team decoders read them). The
  typed Hidden Power `num` (355–370) and the item `num` (the item-dex `num`, not `spritenum`) are read by
  the observation: changing them is retrain-class.
- After editing a builder: regenerate to a temp dir, diff, run the drift check above and
  `python -m agents.training.golden_obs_core --check`. `src/agents/gen3_data/extractor_parity_test.py`
  fails when a committed file drifts from the builders' output.
