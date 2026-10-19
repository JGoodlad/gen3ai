# The files in `data/` — what each one holds

Always-current: update this in the same pass as a builder or a facade module that changes a file's
shape. This is the CONSUMER-side schema: what each committed file holds and the facade accessor
(`agents.gen3_data`) that reads it. Why each builder emits what it emits is
[`pokemon_data_extractor.md`](pokemon_data_extractor.md); the priors' producer contract is
[`smogon_priors.md`](smogon_priors.md); the facade's own rules and hazards are
`src/agents/gen3_data/CLAUDE.md`.

Every file is loaded once (a lazy singleton in `gen3_data._base`) and raises `FileNotFoundError` /
`ValueError` when missing or empty.

## Reference data (deterministic) — `tools/pokemon_data_extractor/sync.py`

- **`gen3_species.json`** — species id → `{num, baseStats, name, types, gender, baseSpecies?, battleOnly?}`.
  `types` are UPPERCASED to the TypeEncoder axis. `gender`: a fixed `"M"` / `"F"`, `"N"` genderless, absent = a
  gender-ratio species the engine draws at construction (`SpeciesData.gender`). **419 rows**: the 386 base
  forms + 33 gen-3 formes (Deoxys-Attack / -Defense / -Speed with their own stats, the 27 Unown letters,
  Castform's 3 weather formes), each carrying `baseSpecies` (`gen3_species_formes_v1`); the num-indexing rule is the leaf's.
- **`gen3_moves.json`** — move id → `{num, basePower, type, category, accuracy, never_miss, hasSecondary,
  hasRecoil, priority, secondaryEffects {col: percent}, drainFraction, recoilFraction, …}` plus the
  obs-neutral mechanics fields the Rust port reads. The structured secondary / priority / drain fields
  (`gen3_unified_move_system_v1`) are GPU-side only, not in the obs vector. `category` is the dex category
  verbatim (`MoveData.dex_category`): read it only for Status vs damaging (the leaf's rule). The damage op's
  table build refuses a damaging BP-0 move it has no declared model for (`damage_tables.DAMAGE_MODELS`).
  Typed Hidden Power has distinct nums (`gen3_typed_hidden_power_ids_v1`): bare `hiddenpower` = 237, the 16
  typed variants = 355–370.
- **`gen3_items.json`** — item id → `{num, name}`; `num` is the item-dex number (cross-gen aliases share one).
- **`gen3_abilities.json`** — ability id → `{num, name}`.
- **`gen3_type_chart.json`** — `{DEF: {ATT: multiplier}}` (`type_chart.chart()` / `.multiplier(def, att)`).
- **`gen3_natures.json`** — nature → `{num, stat multipliers}` (`natures.get(name)`; `natures.multipliers()` for the dict form).
- **`gen3_learnset.json`** — species id → `[move_id, …]`, the gen-3 LEGAL movepool: the union over the
  pre-evolution chain, plus Sketch's pool for Smeargle; base forms only; bare `hiddenpower` only
  (`learnset.is_legal`, `learnset.get_legal_moves` → `frozenset | None`). Legality, not frequency (that is
  `priors.moves`). Guards: `learnset_test.py`, `learnset_showdown_integration_test.py` (the table equals
  Showdown's validator).
- **`gen3_move_aliases.json`** — `{alias_id: canonical_move_id}` from Showdown's `aliases.ts` (`wisp` →
  `willowisp`), MOVES only (`gen3_move_alias_resolution_v1`). Read by the Rust port's dex and by
  `gen3_data.moves.move_aliases()` / `canonical_id()` (packed-team decodes, `team_legality`,
  `info_levers.packed`); the observation path never reads it. The pool's only variant spelling is that one
  `wisp` (`info_levers/packed_test.py` censuses the 719 teams).

## Smogon-derived priors — `tools/smogon_stats_downloader/`

`sync.py` merges 12 months of chaos JSON into **`gen3_smogon_stats.json`** (per-species `Moves` / `Items` /
`Spreads` / `Teammates` are summed counts); `compute_priors.py` derives six files, banned entities removed
at acquisition (the Rust encoder reads the same files, so the filter cannot be load-time only):

| file | accessor | holds |
|---|---|---|
| `gen3_ability_priors.json` | `priors.ability(species)` | `{ability_id: p}` |
| `gen3_hidden_power_priors.json` | `priors.hidden_power(species)` | `{hp_type: p}` |
| `gen3_move_priors.json` | `priors.moves(species)` | `P(m in set) = Moves[m] / W`; per species `Σ P + Moves[""] / W == 4` exactly |
| `gen3_item_priors.json` | `priors.items(species)` | `{item_id: p}` |
| `gen3_spread_priors.json` | `priors.spreads(species)` | the top-25 chaos `Spreads` per species |
| `gen3_teammate_priors.json` | `priors.teammates(species)` | chaos `Teammates` normalised per species: the one species × species JOINT Smogon publishes (the hidden-team belief's coupling prior) |

W is the rating-weighted set total (`Σ Abilities`), never the unweighted `Raw count`
(`gen3_smogon_prior_denominator_v1`). Chaos `Moves` are per-species marginals: within-species move-pair
couplings have no Smogon source. Two accessors read `gen3_smogon_stats.json` at call time and have no file:

- `priors.all_spreads(species)` — every chaos `Spreads` entry (weights sum 1), without the top-25 cut,
  which drops a lumpy tail; its consumer is the speed physics' mixture
  (`agents.model.belief_tables.build_species_speed_mix`).
- `priors.species_usage()` — each species' W (`_weighted_count`), never the `Raw count` share
  (`gen3_smogon_species_usage_weighted_v1`).

Every prior load throws `PriorInvariantError` on a file that is not a distribution, on a move table off
the `== 4` identity (`_checked_moves`), on a usage table that is not W (`_checked_species_usage`), and on a
banned ability / item / move / teammate (`_checked_format_legal`). `species_usage()` excludes banned species.

## Pool-derived — `data/teams/gen3_team_archetypes.json`

Every pool team labelled by PACE class (hyper_offense / offense / balance / semi_stall / stall, a
transparent composition rubric) plus style TAGS (sand, spikes, spin, spinblock, phaze, trap, trap_core,
wish, boom, choice, …), keyed by `team_archetypes.team_sha` (`sha1(team_str)[:10]`, the MatchupSpec
`pin_sha` convention). Derived by `python -m agents.training.team_archetypes`; loaded by
`team_archetypes.load_team_archetypes`. Readers: `team_winrate_callback.py` (each team's archetype in the
per-team win-rate table) and `main/promote_teams.py` (checks the artifact covers the pool). It is a
MEASUREMENT of the pool, never a prior the network reads.
