"""Render the Rust core's OBSERVATION LAYOUT from the Python encoder's own constants
(``gen3_core_obs_layout_v1``, the Rust Core Program's M4).

``src/rust_sim/src/encoder/layout.rs`` is GENERATED from the single sources the Python encoder
reads — ``agents/observation/constants.py`` (every offset and dim), ``gen3_effects`` (the volatile
slot table, the cant vocabulary), ``turn_view.FAINT_CAUSE_VOCAB``, the per-encoder index maps
(``TypeEncoder.TYPE_TO_IDX``, the status / weather / screen maps), the saturation LUT, the sleep
wake tables, the protect floor, poke-env's pre-split category map — and committed. A dim change
is a ONE-PLACE edit (``constants.py``): ``rust_core_obs_layout_test.py`` (routine, ~free) fails the
day the committed file is stale, before any byte comparison could.

Every float is written with ``repr`` (the shortest string that round-trips), so the Rust literal
parses to the SAME f64 the Python expression produced.

    python -m agents.observation.rust_core_obs_layout --write
    python -m agents.observation.rust_core_obs_layout --check
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

_W = Callable[[str], None]

from utils.paths import repo_path

OUT = repo_path("src", "rust_sim", "src", "encoder", "layout.rs")

#: Every ``constants.py`` integer the encoder addresses, by name (a renamed constant fails here).
_USIZE_CONSTANTS = (
    "POKEMON_SPECIES_OFFSET", "POKEMON_ITEMS_OFFSET", "POKEMON_TYPES_OFFSET",
    "POKEMON_ABILITIES_OFFSET", "POKEMON_CONDITION_OFFSET", "POKEMON_MOVES_OFFSET",
    "POKEMON_HP_OFFSET", "POKEMON_SPECIES_KNOWN_OFFSET", "POKEMON_COUNTER_OFFSET",
    "POKEMON_SPREAD_OFFSET", "POKEMON_SPREAD_DIM", "POKEMON_HP_REVEALED_OFFSET",
    "POKEMON_HP_PROBS_OFFSET", "POKEMON_HP_BLOCK_DIM", "POKEMON_SLEEP_BELIEF_OFFSET",
    "POKEMON_SLEEP_BELIEF_DIM", "POKEMON_RECENCY_OFFSET", "POKEMON_RECENCY_DIM",
    "POKEMON_PROTECT_OFFSET", "POKEMON_LAST_ACTION_OFFSET", "POKEMON_LAST_ACTION_DIM",
    "POKEMON_VECTOR_DIM", "POKEMON_TRAPPED_OFFSET", "POKEMON_MAYBE_TRAPPED_OFFSET",
    "POKEMON_ACTIVE_OFFSET", "POKEMON_FULL_DIM",
    "SPECIES_ID_DIM", "ITEM_ID_DIM", "ITEM_KNOWN_DIM", "ITEM_CONSUMED_DIM", "COMBINED_TYPES_DIM",
    "ABILITY_SLOT_DIM", "ABILITY_DOMINANCE_DIM", "ABILITY_KNOWN_DIM", "MOVE_SLOT_DIM",
    "CONDITION_DIM", "STATS_DIM", "POKEMON_COUNTER_DIM",
    "BOOSTS_DIM", "VOLATILES_DIM", "ACTIVE_CONTEXT_DIM",
    "WEATHER_ONEHOT_DIM", "CLOCK_DIM", "CLOCK_OFFSET_IN_GLOBAL", "GLOBAL_ENV_DIM",
    "REACTIVE_SCALAR_DIM", "ACTIVE_REQ_MOVES_PER", "ACTIVE_REQ_MOVES_DIM",
    "ACTIVE_REQ_MOVES_OFFSET", "REACTIVE_DIM",
    "PAIR_HISTORY_CELL_DIM", "PAIR_HISTORY_DIM",
    "EVENT_WINDOW_N", "EVENT_TOKEN_DIM", "EVENT_WINDOW_DIM",
    "TEAM_SIZE", "NUM_POKEMON",
    "OFFSET_OUR_TEAM", "OFFSET_OPP_TEAM", "OFFSET_CONTEXT", "OFFSET_GLOBAL", "OFFSET_REACTIVE",
    "OFFSET_PAIR_HISTORY", "OFFSET_EVENT_WINDOW",
    "MAX_TURNS", "MAX_SPIKES", "MAX_PP",
    "EVENT_T_PAD", "EVENT_T_MOVE", "EVENT_T_SWITCH_IN", "EVENT_T_FAINT", "EVENT_T_STATUS_APPLIED",
    "EVENT_T_STATUS_CURED", "EVENT_T_BOOST", "EVENT_T_ITEM_REVEAL", "EVENT_T_HAZARD",
    "EVENT_T_SWITCH_REJECTED", "EVENT_T_CANT", "N_EVENT_TYPES",
    "ITEM_TR_NONE", "ITEM_TR_REVEALED", "ITEM_TR_CONSUMED", "ITEM_TR_REMOVED", "ITEM_TR_SWAPPED",
)


def _rs(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _f64(x: float) -> str:
    """A Rust f64 literal that parses to exactly ``x`` (``repr`` round-trips)."""
    r = repr(float(x))
    return r if ("." in r or "e" in r or "inf" in r or "nan" in r) else r + ".0"


def _str_table(w: _W, name: str, items: Iterable[str]) -> None:
    items = list(items)
    w(f"pub static {name}: [&str; {len(items)}] = [{', '.join(_rs(s) for s in items)}];")


def _pairs_usize(w: _W, name: str, items: Iterable[Tuple[str, int]]) -> None:
    items = list(items)
    body = ", ".join(f"({_rs(k)}, {int(v)})" for k, v in items)
    w(f"pub static {name}: [(&str, usize); {len(items)}] = [{body}];")


def _f64_table(w: _W, name: str, xs: Iterable[float]) -> None:
    xs = list(xs)
    w(f"pub static {name}: [f64; {len(xs)}] = [{', '.join(_f64(x) for x in xs)}];")


def render() -> str:
    from agents.battle.turn_view import FAINT_CAUSE_VOCAB
    from agents.gen3_mechanics import _PROTECT_COUNTER_MAX
    from agents.observation import constants as C
    from agents.observation import gen3_effects as E
    from agents.observation import global_env as G
    from agents.observation import pokemon as PK
    from agents.observation import sleep_belief as SB
    from agents.observation.assembler import SAT_LUT
    from agents.observation.types import TypeEncoder
    from agents.observation.wish_belief import WISH_HEAL_FRACTION
    from agents.training.episode_tracker import PairHistoryTracker, RecencyTracker
    from poke_env.battle.move import Move

    # `Gen3ObservationEncoder.base_dimension` is exactly this expression; the test pins the two.
    obs_dim = C.OFFSET_EVENT_WINDOW + C.EVENT_WINDOW_DIM
    out: List[str] = []
    w = out.append
    w("// @generated by `python -m agents.observation.rust_core_obs_layout --write` — DO NOT EDIT.")
    w("// Source of truth: `agents/observation/constants.py` (every offset / dim), `gen3_effects`,")
    w("// `turn_view.FAINT_CAUSE_VOCAB`, the Python sub-encoders' index maps, `assembler.SAT_LUT`,")
    w("// `sleep_belief`'s wake tables, `gen3_mechanics`' protect floor and poke-env's pre-split")
    w("// category map. Pinned by `src/agents/observation/rust_core_obs_layout_test.py`.")
    w("#![allow(dead_code)]")
    w("")
    w("/// `Gen3ObservationEncoder.dimension` — the flat row's length.")
    w(f"pub const OBS_DIM: usize = {obs_dim};")
    for name in _USIZE_CONSTANTS:
        w(f"pub const {name}: usize = {int(getattr(C, name))};")
    w("")
    w("/// `constants.EventCol` — the 22 columns of one event row, by name.")
    for c in C.EventCol:
        w(f"pub const EV_{c.name}: usize = {int(c)};")
    w("")
    w("/// `constants.EVENT_STATUS_IDS` (the event row's STATUS column) — also `pokemon._STATUS_STR_IDX`.")
    _pairs_usize(w, "EVENT_STATUS_IDS", sorted(C.EVENT_STATUS_IDS.items(), key=lambda kv: kv[1]))
    _pairs_usize(w, "CONDITION_STATUS_IDX", sorted(PK._STATUS_STR_IDX.items(), key=lambda kv: kv[1]))
    w("")
    w("/// `gen3_effects.VOLATILE_SLOTS` — the active context's volatile columns, in order.")
    _str_table(w, "VOLATILE_SLOTS", E.VOLATILE_SLOTS)
    w("/// `gen3_effects.GEN3_VOLATILE_TO_SLOT` — (id, slot index, value), sorted by id.")
    rows = sorted((vid, E.VOLATILE_SLOTS.index(slot), val)
                  for vid, (slot, val) in E.GEN3_VOLATILE_TO_SLOT.items())
    w(f"pub static VOLATILE_TO_SLOT: [(&str, usize, f64); {len(rows)}] = [")
    for vid, idx, val in rows:
        w(f"    ({_rs(vid)}, {idx}, {_f64(val)}),")
    w("];")
    w("/// `gen3_effects.NOT_A_VOLATILE` — ids that reach the view but encode to nothing.")
    _str_table(w, "NOT_A_VOLATILE", sorted(E.NOT_A_VOLATILE))
    w("/// `gen3_effects.CANT_REASONS_LIVE` — the event row's CANT column is 1 + the index.")
    _str_table(w, "CANT_REASONS_LIVE", E.CANT_REASONS_LIVE)
    w("/// `turn_view.FAINT_CAUSE_VOCAB` — the event row's FAINT_CAUSE column is 1 + the index.")
    _str_table(w, "FAINT_CAUSE_VOCAB", FAINT_CAUSE_VOCAB)
    w("")
    w("/// `TypeEncoder.TYPE_TO_IDX` — keyed by the `PokemonType` NAME the per-mon block reads")
    w("/// (`THREE_QUESTION_MARKS` is NOT a key: a `???`-typed mon reads 0, as in Python), and by the")
    w("/// `\"???\"` spelling the move encoder translates to.")
    _pairs_usize(w, "TYPE_TO_IDX", sorted(TypeEncoder.TYPE_TO_IDX.items(), key=lambda kv: kv[1]))
    w("/// poke-env's `Move._MOVE_CATEGORY_PER_TYPE_PRE_SPLIT`: the `PokemonType` NAMES whose damaging")
    w("/// moves are SPECIAL (the rest are PHYSICAL) — `moves._category_val`'s gen-3 source.")
    special = sorted(t.name for t, cat in Move._MOVE_CATEGORY_PER_TYPE_PRE_SPLIT.items()
                     if cat.name == "SPECIAL")
    _str_table(w, "SPECIAL_TYPES_PRE_SPLIT", special)
    w("")
    w("/// `global_env._WEATHER_IDX` (the `None` key is index 0, the default).")
    _pairs_usize(w, "WEATHER_IDX", [(k, v) for k, v in G._WEATHER_IDX.items() if k is not None])
    w(f"pub const WEATHER_MAX_TURNS: f64 = {_f64(G._WEATHER_MAX_TURNS)};")
    w("/// `global_env._LOG_MAX_TURNS` = `math.log(1 + MAX_TURNS)`.")
    w(f"pub const LOG_MAX_TURNS: f64 = {_f64(G._LOG_MAX_TURNS)};")
    w("/// `global_env._SCREEN_CONDITIONS`, as the lower-cased `SideCondition` names the view keys by.")
    _str_table(w, "SCREEN_CONDITIONS", [c.name.lower() for c in G._SCREEN_CONDITIONS])
    w("")
    w("/// `active_context`'s boost order.")
    _str_table(w, "BOOST_ORDER", ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"))
    w("/// `PokemonEncoder._NATURE_STAT_ORDER`.")
    _str_table(w, "NATURE_STAT_ORDER", PK.PokemonEncoder._NATURE_STAT_ORDER)
    w("")
    w("/// `assembler.SAT_LUT` — `log1p(n) / log(11)` for n = 0..=10, in float64.")
    _f64_table(w, "SAT_LUT", SAT_LUT)
    w(f"pub const RECENCY_SAT: usize = {int(RecencyTracker._SAT)};")
    w(f"pub const PAIR_SAT: usize = {int(PairHistoryTracker._SAT)};")
    w("")
    w("/// `sleep_belief`'s verified gen-3 wake tables (index = the observed counter, clamped).")
    _f64_table(w, "SLEEP_OPP_NOEB", SB._OPP_NOEB)
    _f64_table(w, "SLEEP_OPP_EB", SB._OPP_EB)
    _f64_table(w, "SLEEP_REST_NOEB", SB._REST_NOEB)
    _f64_table(w, "SLEEP_REST_EB", SB._REST_EB)
    w(f"pub const SLEEP_MAX_K: usize = {int(SB._MAX_K)};")
    _str_table(w, "SLEEP_USABLE_MOVES", sorted(SB._SLEEP_USABLE_MOVES))
    w(f"pub const EARLY_BIRD: &str = {_rs(SB._EARLY_BIRD)};")
    w(f"pub const UNKNOWN_ABILITY: &str = {_rs(SB._UNKNOWN_ABILITY)};")
    w("")
    w("/// `gen3_mechanics._PROTECT_COUNTER_MAX` — the 1/8 floor of the protect-success odds.")
    w(f"pub const PROTECT_COUNTER_MAX: u64 = {int(_PROTECT_COUNTER_MAX)};")
    w("/// `wish_belief.WISH_HEAL_FRACTION`.")
    w(f"pub const WISH_HEAL_FRACTION: f64 = {_f64(WISH_HEAL_FRACTION)};")
    w("")
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = render()
    if a.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text)
        print(f"wrote {OUT}")
        return 0
    if not OUT.exists() or OUT.read_text() != text:
        print(f"{OUT} is STALE — run `python -m agents.observation.rust_core_obs_layout --write`")
        return 1
    print("current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
