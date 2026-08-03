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
                                        #   secondary_effects + .secondary_chance(col), …)  # v24
gen3_data.species.get(species_id)       # SpeciesData(num, base_stats, types, base_species, battle_only)
gen3_data.species.base_form_ids()       # BASE forms only — one id per national-dex num
gen3_data.items.get(item_id)            # ItemData(num, name)
gen3_data.abilities.get(ability_id)     # AbilityData(num, name)
gen3_data.natures.get(nature_name)      # NatureData(multipliers); .multipliers() for the dict form
gen3_data.type_chart.chart()            # {DEF: {ATT: multiplier}}; .multiplier(def, att)
gen3_data.priors.ability(species)       # {ability_id: probability}     (Smogon)
gen3_data.priors.hidden_power(species)  # {hp_type: probability}         (Smogon)
gen3_data.learnset.is_legal(species, move_id)      # gen3 legal-movepool gate (hard legality)
gen3_data.learnset.get_legal_moves(species)        # frozenset|None (None = unknown → no constraint)
```

`learnset` is the **legality** primitive (which moves a species can LEGALLY learn in gen3) — distinct
from `priors.moves` (how OFTEN a legal move is run). The move-belief prior uses it to PRUNE impossible
candidate moves (`damage_tables.build_move_prior_logits(..., learnset_gate=True)`); its tolerance
contract is that an unknown species yields `None`/`True` ("no constraint", never "no moves"), so the
gate can never wrongly prune.

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
