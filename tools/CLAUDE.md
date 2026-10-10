# CLAUDE.md — Acquisition tools (`tools/`)

`tools/` is the **acquisition layer**: the *only* place that knows the three upstreams — the poke-env
static JSON (FROZEN as data under `pokemon_data_extractor/upstream/poke_env_static/` since the vendored
fork was deleted, P6 of the poke-env retirement; read with plain `json.load`, no poke-env code runs — its
`README.md`), the Showdown `.ts` source tree (`deps/pokemon-showdown/`), and Smogon usage stats. Each tool
*derives* and normalizes its upstream into committed files under `data/`, so the derivation is
reproducible rather than a hand edit. The runtime never imports `tools/`; it reads `data/` through the
`agents.gen3_data` facade (`src/agents/gen3_data/CLAUDE.md` owns every per-file schema). Detail lives in
`designs/tools/` (`README.md` is the index; the pre-cleanup leaf is
`designs/research_state/claude_md_archive/tools_CLAUDE_2026-10-10.md`).

## The tools

| Tool | Upstream | Output |
|---|---|---|
| `pokemon_data_extractor/sync.py` | the frozen poke-env static JSON (pokedex, moves, natures, `learnset.json`, typechart); Showdown `abilities.ts` / `items.ts` / `aliases.ts` | `data/pokemon/gen3_{species,moves,abilities,items,type_chart,natures,learnset,move_aliases}.json` |
| `smogon_stats_downloader/sync.py` | Smogon monthly chaos JSON (12-month window) | `data/pokemon/gen3_smogon_stats.json` |
| `smogon_stats_downloader/compute_priors.py` | the aggregated stats + pokedex | `data/pokemon/gen3_{ability,hidden_power,move,item,spread,teammate}_priors.json` |
| `sample_team_downloader/sync.py` | Smogon forum sample-team thread | `data/teams/sample/` |
| `others_team_downloader/sync.py` | PokePaste dumps | `data/teams/others/` |
| `replay_corpus_downloader/sync.py` | the PUBLIC replay archive (`replay.pokemonshowdown.com`) | `<main>/replays/showdown/<format>/<date>/battle-<id>.log` — NOT `data/`; a RE-WRITE of a lost script, never run against the live server (`replay_corpus_downloader/README.md`) |

## Rules

- 🚨 **A `data/` write is a DATA change.** No `data/` change while a pinned run is live or queued (a pin
  isolates code, not data — pinned children read `data/` from main). Regenerate into a temp dir first
  (`sync.py --out-dir <TEMP>`), diff against `data/`, and land the change deliberately; a changed TEAM
  pool is a training *and* eval distribution change — land it at a clean retrain boundary, never mid-A/B.
- 🚨 **Smogon chaos: never divide a WEIGHTED field by `Raw count`** (`gen3_smogon_prior_denominator_v1`,
  F-X5-41). `Raw count` is unweighted; `Abilities` / `Items` / `Spreads` / `Happiness` share the
  rating-weighted total W and `Σ Moves = 4 W`. The move prior is `Moves[m] / W`
  (`compute_priors.weighted_count`); the `Raw count` form was deflated ×0.10–0.94. `check_priors` /
  `weighted_count` THROW before any write; the 12-month merge sums every weighted field; `usage` is
  latest-month only, never a count. Detail: `designs/tools/smogon_priors.md`.
- 🚨 **The FORMAT SPEC filters every prior at acquisition** (`gen3_format_spec_priors_v1`): run
  `compute_priors.py` with `PYTHONPATH=src` (it imports `agents.gen3_data.format_spec`);
  `check_format_legal` THROWS if banned mass survives. Detail: `designs/tools/smogon_priors.md`.
- 🚨 **`data/teams/sample/` has a SECOND writer, `python -m main.promote_teams`** — and
  `sample_team_downloader/sync.py` rebuilds `teams.json` from the forum thread ALONE, so a re-sync silently
  drops every promoted team. Check `data/teams/sample/PROMOTION_MANIFEST.json` before running it and
  re-promote from its recorded seed afterwards (`designs/tools/team_downloaders.md`).
- 🚨 **Every team manifest holds ONE entry per distinct team** (== per `id` == per `.txt`):
  `TeamLoader` draws once per ENTRY, so a duplicate multiplies a team's draw weight (Yak Attack once
  reached ~66% of every draw; `others_team_downloader/sync.py::collapse_duplicate_teams` now collapses by
  `id`). Guards: `src/utils/team_loader/team_manifest_test.py`, `loader_test.py` (the pool-count pin).
- 🚨 **The downloaders name their encodings, and the committed mojibake is deliberately NOT rewritten.**
  `requests` guesses ISO-8859-1 on a `text/*` response with no charset; both syncs override it with
  `apparent_encoding` and write `encoding="utf-8"`. The committed `data/teams/others/mcmegan/*.txt` still
  hold `PtÃ©ra`: a team file's bytes key `pin_sha` and `data/teams/gen3_team_archetypes.json`, so a
  "fix" re-ids the team. `src/utils/team_loader/committed_team_bytes_test.py` pins the bytes
  (`designs/tools/team_downloaders.md`).

## `pokemon_data_extractor` — commands and the hazards that bite

One `--datasets` entry per file (`all` rebuilds every file); each builder is registered in `_BUILDERS`:

```bash
python tools/pokemon_data_extractor/sync.py                          # all, gen 3, WRITES data/pokemon/
python tools/pokemon_data_extractor/sync.py --datasets moves species # subset
python tools/pokemon_data_extractor/sync.py --gen 3 --stdout         # print, don't write
python tools/pokemon_data_extractor/sync.py --out-dir <TEMP>         # regenerate elsewhere, then diff
```

Building items / abilities / aliases needs the Showdown submodule (`git submodule update --init`);
type chart / natures / species / learnset read only the frozen static JSON. Per-builder detail:
`designs/tools/pokemon_data_extractor.md`.

- 🚨 **A forme SHARES its base species' `num`.** `build_species` emits the 386 base forms plus the 33
  gen-3 formes (Deoxys ×3, Castform ×3, the 27 Unown letters; 419 rows), each with `baseSpecies`. A
  num-indexed consumer MUST iterate `gen3_data.species.base_form_ids()`, or the last forme written
  redefines the base. The forme list is a CURATED table (`_GEN_ALT_FORMES`), because the static pokedex
  carries 135 post-gen-3 formes with gen-3 nums — a `num` rule cannot select them.
- 🚨 **Mechanics come from the RESOLVED `Dex.mod('gen3')`, never from a single `.ts` file** (the
  mod-chain law: gen4 rewrites Light Ball, gen3 rewrites it again). The curated `_GEN3_ITEM_MECHANICS` /
  `_GEN3_ABILITY_MECHANICS` tables and the species formes are derived by
  `src/rust_sim/harness/dump_gen3_mechanics.js`, and **every regeneration is drift-gated by
  `node src/rust_sim/harness/dump_gen3_mechanics.js --check`**.
- **Mechanics fields are obs-neutral** (`critRatio`, `pp`, `secondaryBoosts`, `selfBoosts`, `selfDrops`,
  `statDropBoosts`, `multihit`, the item/ability mechanics, `gen3_move_aliases.json`): the facade ignores
  them; the `src/rust_sim` port reads them. **Not every builder change is**: the typed Hidden Power `num`
  (355–370, `gen3_typed_hidden_power_ids_v1`) and the item `num` (the item-dex `num`, NOT `spritenum` —
  the regex is `\bnum:`) are obs VALUES — retrain-class.
- `_SELF_BOOST_STATS` (gating `selfBoosts` / `selfDrops`) still excludes accuracy/evasion for a reason
  OTHER than the stale one once written beside it; relaxing it is an unprobed question
  (`designs/tools/pokemon_data_extractor.md`, `statDropBoosts`).

## Reproducibility is tested

`src/agents/gen3_data/extractor_parity_test.py` re-runs the builders and asserts they reproduce the
committed `data/` files (and that type chart / natures still equal their frozen upstream). A hand-edit
that drifts a committed file from the extractor's output fails there. After editing a builder,
regenerate into a temp dir, diff, then run the drift gate above and the obs golden
(`python -m agents.training.golden_obs_core --check`) — a value change there is retrain-class.
`tools/pokemon_data_extractor/pokemon_data_extractor_test.py` and `smogon_stats_downloader/sync_test.py`
are the tools' own unit tests.
