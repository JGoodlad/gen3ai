# CLAUDE.md — Data facade (`src/agents/gen3_data/`)

`gen3_data` is the **single, domain-facing interface over the project's Pokémon data**. The
runtime reads **only** `data/` through it and is blind to where the data originally came from.

## Acquisition vs. access (the governing split)

| Layer | Knows about | Job |
|---|---|---|
| **Acquisition** (`tools/`) | poke-env static data, the Showdown `.ts` source tree, Smogon usage stats | derive + normalize → write files to `data/pokemon/` |
| **`data/pokemon/`** | nothing | normalized, committed JSON — the contract between the layers |
| **Access** (`gen3_data`) | only `data/` | typed, domain-organized lookups for the runtime |

The three upstreams collapse to **one place that knows them** (`tools/`, see `tools/CLAUDE.md`).
Everything downstream asks by *concept*, never by *source*.

## The facade

```python
from agents import gen3_data
gen3_data.moves.get(move_id)            # MoveData(num, base_power, type, category, accuracy,
                                        #   priority, drain_fraction, recoil_fraction,
                                        #   secondary_effects + .secondary_chance(col),      # v24
                                        #   self_boosts — the pure-setup (stat, stages) map  # C1
                                        #   [gen3_setup_moves_v1; empty for Belly Drum/Curse/
                                        #    Defense Curl by the pure-setup gates], …)
gen3_data.species.get(species_id)       # SpeciesData(num, base_stats, types, base_species, battle_only)
gen3_data.species.base_form_ids()       # BASE forms only — one id per national-dex num
gen3_data.items.get(item_id)            # ItemData(num, name)
gen3_data.abilities.get(ability_id)     # AbilityData(num, name)
gen3_data.natures.get(nature_name)      # NatureData(multipliers); .multipliers() for the dict form
gen3_data.type_chart.chart()            # {DEF: {ATT: multiplier}}; .multiplier(def, att)
gen3_data.priors.ability(species)       # {ability_id: probability}     (Smogon)
gen3_data.priors.hidden_power(species)  # {hp_type: probability}         (Smogon)
gen3_data.priors.teammates(species)     # {teammate_id: P(teammate|species)} (Smogon chaos
                                        #  Teammates — the one species×species JOINT published;
                                        #  the hidden-team belief's coupling prior)
gen3_data.learnset.is_legal(species, move_id)      # gen3 legal-movepool gate (hard legality)
gen3_data.learnset.get_legal_moves(species)        # frozenset|None (None = unknown → no constraint)
```

`learnset` is the **legality** primitive (which moves a species can LEGALLY learn in gen3) — distinct
from `priors.moves` (how OFTEN a legal move is run). The move-belief prior uses it to PRUNE impossible
candidate moves (`damage_tables.build_move_prior_logits`) — **unconditionally**, since a move a species
cannot learn carrying belief mass is a correctness bug, not a tunable. Its tolerance contract is that an
unknown species yields `None`/`True` ("no constraint", never "no moves"), so the gate can never wrongly
prune; the builder honours that by flattening any row it never wrote (national-dex num 0, dex gaps) to
the legal-unobserved floor rather than the impossible default.

## Species FORMES — `raw()` vs `base_form_ids()` (`gen3_species_formes_v1`)

`gen3_species.json` carries **419** rows: the 386 base forms **plus 33 gen-3 alternate/cosmetic
FORMES** — Deoxys-Attack/-Defense/-Speed (their own base stats), the 27 Unown letters (cosmetic
clones of Unown), and Castform-Sunny/-Rainy/-Snowy (`battle_only`, retyped). They are there because
a battle genuinely fields them: without the rows the `src/rust_sim` port could not construct **6.6%
of gen3 random-battle teams** (`unknown species "Deoxys-Speed"`), and the obs `SpeciesEncoder` would
raise on the same species.

**A forme SHARES its base's national-dex `num`** — and `num` is what the obs species channel and
every model buffer (`table[species.num] = …` in `agents/model/damage_tables.py`) are keyed by. So a
forme is *observationally* its base, and:

| use | iterate |
|---|---|
| id → facts (`get`, encoding a mon, the port's stat calc) | `raw()` / `get()` — formes included |
| anything indexed by `species.num` (GPU buffers, num→id decode) | **`base_form_ids()`** |

Iterating `raw()` into a num-indexed table is last-write-wins: it would put Deoxys-Speed's
95/90/95/90/180 on num 386 and Castform-Sunny's FIRE on num 351 — a plausible-but-false value no
shape check catches. `SpeciesData.base_species` (the base id, `None` for a base form) is the flag;
`base_form_ids()` is the bijection onto the nums. Guards: `species_test.py`
(coverage + the num bijection), `damage_tables_test.py` (the tables hold the BASE forme),
`src/rust_sim/tests/species_formes_test.rs` (a packed forme team constructs), and the producer-side
oracle gate `node src/rust_sim/harness/dump_gen3_mechanics.js --check`.

## Concept-module discipline (per submodule)

Each submodule mirrors `moves.py` (the original `gen3_movedex`, the template):
- an immutable `@dataclass(frozen=True)` keyed by id;
- parsed **once** via a lazy singleton from `_base` (`load_json` + `singleton`) — no file is
  re-read or re-parsed;
- `get(id)` is tolerant (returns `None` for an unknown/unrevealed id), `*_data(id)` is
  crash-don't-drop (raises `KeyError`);
- poke-env value-enums (`PokemonType`, `MoveCategory`) are borrowed as **keys/names only** —
  never called, never made to carry data. The data is ours; the enums are just the keys.
- `.raw()` returns the parsed JSON dict; `state_encoder.load_mappings` assembles the encoder
  mappings from `.raw()` so each file is parsed once and shared.

`_base.py` owns the only path resolution (`data/pokemon/`, repo-root-relative so CWD doesn't
matter), validation (missing/empty → raise), and the singleton idiom.

## Who reads through it

`state_encoder.load_mappings()` (assembles the encoder's species/moves/items/abilities/priors/
natures + reverse maps), `gen3_mechanics` (`type_chart`), `hidden_power_tracker` (`priors`). All
poke-env *static-data* reads have been removed from the runtime; the data layer is poke-env-free.

## Performance note (why we own the data)

Owning the data as plain dataclasses lets us control lookup cost instead of inheriting poke-env's
property machinery (`move.entry`, `GenData.from_gen`, …) on the hot obs path. **Caveats that must
stay live, not become dex lookups:** current/max PP is battle *state*, not reference data; and
`move.category` has a fixed-power disagreement vs a movedex re-derivation — the obs encoder keeps
the live-`move.category` memoization (`moves._category_val`). See `observation/CLAUDE.md`. Any
swap of a live poke-env property for a facade lookup is gated by the obs-build benchmark and must
be proven value-neutral (or it's a retrain-class change).

## Value-neutrality (changing the data layer)

Reorganizing *where/through-what* data loads must not change any observed value. The guards:
- **`training/gen3_data_obs_parity_integration_test.py`** — the linchpin: replays a fixed,
  deterministic battle set and asserts every per-decision obs vector is byte-identical to the
  golden fixture (`golden_obs_fixture.json`, captured by `golden_obs_capture.py`).
- **`extractor_parity_test.py`** — committed files == upstream; builders reproduce committed.
- **per-dex + facade tests** here; **`gen3_mechanics_test.py`** pins the effectiveness chart.

A *value* change to the data is retrain-class: bump `ARCH_SIGNATURE` and regenerate the golden
fixture. (Example: `gen3_item_num_fix_v1` switched the item id from Showdown's spritenum to the
true item-dex `num` — same obs dim, but every item id re-meaned, so old item embeddings are
invalid.)

## The files in `data/pokemon/` — what each one holds

*(Moved here from the root `CLAUDE.md` on 2026-09-07: the root states the governing rules, this leaf owns the per-file schema. Regenerate every reference file with `tools/pokemon_data_extractor/sync.py`; the priors with `tools/smogon_stats_downloader/`.)*

Reference data (deterministic) under `data/pokemon/`, all regenerable via
`tools/pokemon_data_extractor/sync.py`:
- `gen3_species.json` — species id → `{num, baseStats, name, types}` (`types` UPPERCASED to the TypeEncoder
  axis — `gen3_bidir_threat_trunk_v1`, for the op's expected-latent read; the obs still reads revealed types live).
  **419 rows** = the 386 base forms + the 33 gen-3 ALTERNATE/COSMETIC FORMES (`gen3_species_formes_v1`:
  Deoxys-Attack/-Defense/-Speed with their own base stats, the 27 Unown letters, Castform's 3 weather
  formes), each carrying `baseSpecies`. Formes were missing before and cost the `src/rust_sim` port
  **6.6% of gen3 random-battle teams / ~14% of battles** at construction. A forme SHARES its base's
  `num`, and the obs species channel + every `table[species.num]` model buffer are num-keyed — so
  num-indexed consumers MUST iterate `gen3_data.species.base_form_ids()` (see
  `src/agents/gen3_data/CLAUDE.md`)
- `gen3_moves.json` — move id → `{num, basePower, type, accuracy, never_miss, hasSecondary, hasRecoil,
  priority, secondaryEffects {col: percent}, drainFraction, recoilFraction, …}` (the structured
  secondary/priority/drain fields are `gen3_unified_move_system_v1` — GPU-side only, NOT in the obs vector).
  **Typed Hidden Power has distinct nums** (`gen3_typed_hidden_power_ids_v1`): bare `hiddenpower`=237,
  the 16 typed variants=355-370 (Showdown ships them all at 237; the extractor tool overrides — see
  `tools/CLAUDE.md`). OUR known HP uses the distinct num; the opponent's unrevealed HP is the bare 237.
- `gen3_items.json` — item id → `{num, name}` (`num` is the item-dex number; cross-gen aliases share one num)
- `gen3_abilities.json` — ability id → `{num, name}`
- `gen3_type_chart.json` — `{DEF: {ATT: multiplier}}` effectiveness chart (was live `GenData`)
- `gen3_natures.json` — nature → `{num, stat multipliers}` (was live `poke_env/.../natures.json`)
- `gen3_learnset.json` — species id → `[move_id, ...]` gen3 legal movepool (the hard legality gate the
  move-belief prior uses to prune impossible candidate moves; via `gen3_data.learnset`)
- `gen3_move_aliases.json` — `{alias_id: canonical_move_id}` from Showdown's `aliases.ts`
  (`wisp`→`willowisp`, `sd`→`swordsdance`, …). **Consumed ONLY by the `src/rust_sim` port** (its dex
  resolves a packed-team move alias like the RL runtime never touches); the `agents.gen3_data` facade
  does NOT load it, so it is obs-neutral. `gen3_move_alias_resolution_v1`.

Smogon-derived priors (probabilistic), via `tools/smogon_stats_downloader/` (`sync.py` merges 12
months of chaos JSONs → `compute_priors.py` derives six committed artifacts). **ALL priors must be
Smogon-derived; only the MODEL gets bias against the pool** (owner rule 2026-08-15): anything the
network READS must trace to Smogon (or ground-truth labels / ladder replays) — pool structure may
enter only implicitly, through training against pool opponents (team sampling / league targeting
are the sanctioned pool consumers). The 719-team pool may MEASURE structure (it is the only
set-level joint we own) but never ships as a prior:
- `gen3_smogon_stats.json` (raw aggregated chaos stats; per-species `Moves`/`Items`/`Spreads`/
  `Teammates` are 12-month summed counts) → `gen3_ability_priors.json`,
  `gen3_hidden_power_priors.json`, `gen3_move_priors.json`, `gen3_item_priors.json`,
  `gen3_spread_priors.json`, and `gen3_teammate_priors.json` — the chaos `Teammates` field
  normalized per species: the ONE species×species JOINT Smogon publishes (the hidden-team
  belief's coupling prior; `gen3_data.priors.teammates`). Note chaos `Moves` are per-species
  MARGINALS — within-species move-pair couplings exist in the data we can measure (pool) but
  have no Smogon source, so they stay with in-battle evidence + learning.

Pool-derived (a committed calibration artifact, same pattern):
- `data/teams/gen3_team_archetypes.json` — every pool team labeled by PACE class
  (hyper_offense/offense/balance/semi_stall/stall via a transparent composition rubric) + style
  TAGS (sand/spikes/spin/spinblock/phaze/**trap**/**trap_core**/wish/boom/choice/…), keyed by
  `sha1(team_str)[:10]` (the MatchupSpec `pin_sha` convention, so labels join every provenance
  record). Derived by `python -m agents.training.team_archetypes` (a k-means cross-tab prints as
  the unsupervised sanity check); consumed by league targeting (the `trap_core` exploiter
  shortlist) and future archetype-aware team sampling. Loader:
  `agents.training.team_archetypes.load_team_archetypes`.

All are loaded once (lazy singletons) and raise `FileNotFoundError` / `ValueError` if missing or
empty. The data layer is poke-env-free; the only poke-env touches left in the battle layer are a
parser sentinel (`GenData.UNKNOWN_ITEM`) and the `to_id_str` string util — neither is static data.
