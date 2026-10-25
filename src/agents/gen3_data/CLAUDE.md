# CLAUDE.md — the data facade (`src/agents/gen3_data/`)

`gen3_data` is the runtime's only reader of `data/`: typed lookups by concept (`moves`, `species`,
`items`, `abilities`, `natures`, `type_chart`, `priors`, `learnset`, `format_spec`, `hidden_power`,
`hp_repair`). It knows no upstream; `tools/` derives `data/` (`tools/CLAUDE.md`). What each file holds,
field by field, is `designs/tools/data_files.md`.

## Where `data/` is

Every runtime reader of `data/` resolves it through `utils.paths.repo_path` (the checkout the code was IMPORTED
from), never the CWD: `_base` here, `utils.team_loader.default_base_dir()` and the ELO anchors outside. A
launcher-pinned child stands in main's checkout, so a CWD-relative read sees main's `data/` and reopens root
rule 6's freeze (`designs/launcher/pinning_and_worktrees.md`; gate `src/main/launcher/pinned_data_test.py`).

## Module conventions

Each concept module follows `moves.py`: a frozen dataclass keyed by id, parsed once through a `_base`
singleton (`_base` owns path resolution; a missing or empty file raises). `get(id)` returns `None` for an
unknown id; `*_data(id)` raises `KeyError`. `.raw()` is the parsed dict. The value enums (`PokemonType`,
`MoveCategory`, defined in `agents/enums.py`) are keys only and never carry data. Every dex file loads
through `_base.load_dex_json`, which refuses a row without a numeric `num` (it would alias index 0).

## Hazards

- **A forme shares its base species' `num`.** `raw()` / `get()` include the 33 gen-3 formes; anything
  indexed by `species.num` (model buffers, num → id decodes) must iterate `species.base_form_ids()`, or a
  forme's stats and types overwrite its base's. Guards: `species_test.py`, `damage_tables_test.py`.
- **Read a move's dex `category` only for Status vs damaging** (`MoveData.dex_category` /
  `is_dex_damaging`). Gen 3 splits physical / special by TYPE: `MoveData.category` derives that, and calls
  every BP-0 move Status (Seismic Toss, Super Fang, Return).
- **Hidden Power ids:** our known typed HP is its distinct num, 355–370; the opponent's unrevealed HP is
  the bare `hiddenpower`, 237.
- **A packed team's move token is not an id until resolved** through `moves.canonical_id` (`wisp` →
  `willowisp`). A packed species / item / ability the facade does not know is a throwing defect
  (`info_levers.packed.require_known`), never a silent miss.
- **`learnset` answers `None` / `True` for an unknown species: no constraint, never "no moves"**, so the
  move prior's legality prune (`belief_tables.build_move_prior_logits`) cannot wrongly prune.
- **Priors divide by W, the rating-weighted set total, never `Raw count`.** Banned entities are removed
  at acquisition, not at load, because the Rust encoder reads the same files; every prior load throws
  `PriorInvariantError` on a broken distribution or a banned entity (`priors._checked_*`).
- **gen3ou's rules come from `format_spec`, never a constant** (`format_spec.active()`, `is_banned(…)`, the
  clause accessors; a test plants a spec by swapping `format_spec._ACTIVE`). A new ban or clause is a
  spec edit with its source: `python -m main.format_drift check` fails on any difference from master.
  The doc of record is `designs/endstate/design_format_spec.md`.

## Hidden Power has one implementation (`hidden_power.py`)

Every HP type → IVs and IVs → type / power in Python reads this module: `HP_IVS` (the validator's fill,
power 70), `ivs_of` / `ivs_packed`, `type_of` / `power_of` (the gen-3 formula), `iv_patterns` and
`FORMULA_TYPES`. `utils.gen3_utils.GEN3_HP_IVS` and `hp_repair.HP_IVS` are this object.
`hidden_power_single_impl_test.py` fails on a second table, formula or formula-order type list in any
`.py` or `.rs` under `src/`, so import the module. Two copies stay outside it: the Rust engine's own
`getHiddenPower` (`src/rust_sim/src/state.rs`, pinned by `hidden_power_test.py`) and the JS harness
`ou_random_teams.js` (seeded output is a frozen panel's provenance). `hp_repair.py`'s docstring is the spec for
re-sampling the typeless all-31 Hidden Power of replay-derived teams.

## Changing the data

A VALUE change in `data/` (a re-numbered id, a changed stat) is retrain-class: version it per
`designs/model/versioning.md` and re-record the obs golden (`python -m agents.training.golden_obs_core`).
A change to how data loads must change no value: `golden_obs_core_test.py` and
`extractor_parity_test.py` (the committed files equal the builders' output) are the gates.
A team added to `data/teams` must be LEVEL 100: `TeamLoader` refuses a pool holding a `Level:` line of any other value
(`utils.team_corpora.check_label_encodable`, the corpora's guard; the env's spread labels fault off level 100), so a
non-100 team makes every trainer and eval launch fail at pool load.
