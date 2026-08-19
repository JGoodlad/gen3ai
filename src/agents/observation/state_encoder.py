import math

import numpy as np
from .base import ObservationEncoder
from .pokemon import PokemonEncoder
from .active_context import ActiveContextEncoder
from .global_env import GlobalEnvEncoder
from .species import SpeciesEncoder
from .items import ItemsEncoder
from .types import TypeEncoder
from .abilities import AbilitiesEncoder
from .moves import MovesEncoder
from .reactive import ReactiveEncoder
from .sleep_belief import build_sleep_sources
from agents.gen3_data import moves as gen3_movedex
from .constants import (
    POKEMON_VECTOR_DIM,
    POKEMON_FULL_DIM,
    POKEMON_ACTIVE_OFFSET,
    POKEMON_TRAPPED_OFFSET,
    POKEMON_MAYBE_TRAPPED_OFFSET,
    TEAM_SIZE,
    OFFSET_OUR_TEAM,
    OFFSET_OPP_TEAM,
    OFFSET_CONTEXT,
    OFFSET_GLOBAL,
    OFFSET_REACTIVE,
    REACTIVE_DIM,
    OFFSET_PAIR_HISTORY,
    PAIR_HISTORY_DIM,
    PAIR_HISTORY_CELL_DIM,
    OFFSET_EVENT_WINDOW,
    EVENT_WINDOW_N,
    EVENT_TOKEN_DIM,
    EVENT_WINDOW_DIM,
    EVENT_T_MOVE,
    EVENT_COL,
    ACTIVE_CONTEXT_DIM,
    GLOBAL_ENV_DIM
)
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Dict, Any, List, Optional, Tuple
from agents.observation.gen3_effects import cant_reason_id
from agents.battle.turn_view import faint_cause_id
from agents.action.mask_generator import Gen3ActionMasker
from agents.observation.reactive import ReactiveEncoder as _ReactiveEncoder

def load_mappings() -> Dict[str, Any]:
    """Assemble the observation encoder's reference mappings from the ``gen3_data`` facade.

    Each ``data/pokemon/`` file is parsed once by its concept module (``gen3_data.species``,
    ``.moves``, ``.items``, ``.abilities``, ``.priors``, ``.natures``); this borrows their raw
    dicts and inverts id→name reverse maps. The three upstreams (poke-env / Showdown / Smogon)
    stay hidden behind the facade — this loader speaks only domain concepts, no file paths."""
    from agents import gen3_data
    mappings: Dict[str, Any] = {
        # Reference dexes — the raw {id: {num, …}} dicts the sub-encoders read directly. A fresh
        # outer dict per call (matching the previous loader's semantics); inner records are the
        # shared, immutable-by-convention singletons.
        "species": dict(gen3_data.species.raw()),
        "moves": dict(gen3_data.moves.raw()),
        "abilities": dict(gen3_data.abilities.raw()),
        "items": dict(gen3_data.items.raw()),
        # Smogon usage priors for opp-unrevealed ability encoding ({species: {ability_id: prob}}).
        "ability_priors": gen3_data.priors.ability_raw(),
        # Nature stat multipliers ({nature: {atk/def/spa/spd/spe: mult}}) for spread encoding.
        "natures": gen3_data.natures.multipliers(),
    }

    # Pre-compute reverse mappings for IDs to names
    mappings["reverse"] = {}
    for category in ["species", "moves", "abilities", "items"]:
        rev: Dict[int, str] = {}
        for name, data in mappings[category].items():
            if isinstance(data, dict) and "num" in data:
                # gen3_species_formes_v1: an alternate/cosmetic FORME (Deoxys-Speed,
                # Unown-B, Castform-Sunny) shares its BASE's national-dex num, and the obs
                # species channel IS that num — so a forme can never be the decode of a
                # num. Skip forme rows outright rather than relying on "base sorts first".
                if data.get("baseSpecies"):
                    continue
                num = data["num"]
                # Hidden Power (gen3_typed_hidden_power_ids_v1): the OPPONENT's bare HP keeps num 237 →
                # decode it as the bare "hiddenpower" (its type is unknowable); OUR typed HP have
                # DISTINCT nums (355-370), each mapping uniquely to its typed name. The `or name ==
                # "hiddenpower"` only matters for 237 (it has no other claimant now), kept harmless.
                if num not in rev or name == "hiddenpower":
                    rev[num] = name
            elif isinstance(data, (int, float)):
                rev[int(data)] = name
        mappings["reverse"][category] = rev

    return mappings

def get_observation_encoder(mappings: Dict[str, Any]) -> "Gen3ObservationEncoder":
    return Gen3ObservationEncoder(mappings)

class Gen3ObservationEncoder(ObservationEncoder):
    """
    Top-level encoder that orchestrates the entire Gen 3 observation vector.
    Total dimensions: base_dimension. gen3_frame_deletion_v1 deleted the two tail blocks
    (the 11-dim prev-turn action mask and the N x TURN_DELTA_DIM lag frames), so the vector
    is now exactly what the block encoders write — there is no appended tail.
    """
    
    def __init__(self, mappings: Optional[Dict[str, Any]] = None) -> None:
        self.mappings = mappings or {}
        mappings = self.mappings
        
        # Sub-encoders
        rev = self.mappings.get("reverse", {})
        self.species_encoder = SpeciesEncoder(mappings.get("species"), rev.get("species"))
        self.items_encoder = ItemsEncoder(mappings.get("items"), rev.get("items"))
        self.type_encoder = TypeEncoder()
        # Smogon-derived per-species ability priors, loaded by load_mappings()
        # from data/pokemon/gen3_ability_priors.json. The encoder picks the
        # top-2 abilities by Smogon usage for opp-unrevealed slots and writes
        # the dominance probability of ability1 alongside.
        self.abilities_encoder = AbilitiesEncoder(
            mappings.get("abilities"),
            rev.get("abilities"),
            species_to_ability_priors=mappings.get("ability_priors", {}),
        )
        self.moves_encoder = MovesEncoder(rev.get("moves"))
        
        self.pokemon_encoder = PokemonEncoder(
            self.species_encoder,
            self.items_encoder,
            self.type_encoder,
            self.abilities_encoder,
            self.moves_encoder,
            natures=mappings.get("natures", {}),
        )
        
        self.active_context_encoder = ActiveContextEncoder(mappings.get("moves"))
        self.global_env_encoder = GlobalEnvEncoder()
        # ability_priors threads into reactive so matchup cells against
        # unrevealed opp abilities show expected effectiveness instead of
        # the live (None → 1.0×) fallback. Mirrors the AbilitiesEncoder wiring.
        self.reactive_encoder = ReactiveEncoder(
            ability_priors=mappings.get("ability_priors", {}),
        )
        self.current_battle_id = None

    @property
    def base_dimension(self) -> int:
        """Raw encoder output dimension, before the previous-turn mask is appended."""
        # gen3_event_window_v1: the H-B event window sits after the H-A2 block, closing base.
        return OFFSET_EVENT_WINDOW + EVENT_WINDOW_DIM

    @property
    def dimension(self) -> int:
        """Full observation dimension. gen3_frame_deletion_v1: identical to `base_dimension` —
        the prev-mask and lag-frame tail blocks are gone. Kept as a distinct property because
        every consumer reads `dimension`, and a future tail block would land here again."""
        return self.base_dimension

    # Why the `type: ignore[override]` below — LiveView-subject encoder; the ABC still
    # declares the pre-ai_v4 `encode(item, battle)`. See ActiveContextEncoder.encode.
    def encode(self, battle: AbstractBattle, hp_tracker: Any = None,  # type: ignore[override]
               legal: Any = None, progress_clock: Any = None, recency: Any = None,
               pair_history: Any = None, event_window: Any = None) -> np.ndarray:
        """Encode the full base observation vector.

        hp_tracker: optional HiddenPowerTracker whose per-species probability
        vectors are written into each opponent mon's 17-dim HP block. None
        leaves the blocks at zero (e.g. when called outside the training env).

        progress_clock: optional EpisodeTracker-owned ProgressClock (design §5.1) whose
        log-saturated turns_since_progress scalar is written into the reactive block
        (vec[14]). None (inference / unit-test path) leaves it 0.

        legal: optional :class:`~agents.battle.live_view.LegalActions` snapshot for this
        decision (the server-authoritative legality the mask is built from). It feeds the
        reactive block's trapped / maybe_trapped obs bits. The env / inference players build
        it once per decision and thread it here to avoid a second `from_battle`; when ``None``
        it is derived from the strict view (so eval / battle2 / standalone callers still get
        the bits), and on the plain-Battle / unit-test path (no strict_view) it stays None →
        the two bits encode as 0.
        """
        vec = np.zeros(self.base_dimension, dtype=np.float32)

        # Build the current-board read-model ONCE per decision through the strict boundary
        # (`battle.strict_view().live`) and thread it to every sub-encoder. Requires a
        # Gen3Battle; a plain poke-env Battle / unit-test mock (no strict_view/live_view)
        # falls back to None, and each sub-encoder then reads the raw object — byte-identical.
        strict = battle.strict_view() if hasattr(battle, "strict_view") else None
        if strict is not None:
            live = strict.live
            if legal is None:
                legal = strict.legal
        else:
            live = battle.live_view() if hasattr(battle, "live_view") else None

        # Sleep WAKE belief sources (gen3_sleep_wake_belief_v1): fold the event log ONCE per encode
        # for the Rest-source + Sleep-Talk-reliability of each asleep mon — but only when someone is
        # actually asleep, so the common no-sleep decision pays nothing.
        any_asleep = live is not None and any(
            m.status == "slp" for m in (*live.ours.mons, *live.opp.mons)
        )
        sleep_sources = build_sleep_sources(battle) if any_asleep else None

        # Tier H-A1 (gen3_pair_history_v1): each side's LAST ACTION as the 6-tuple the active
        # mon's slot carries — [move_num (embedding id; 0 = none/switch), was_switch,
        # hit, miss, fail, crit] (outcome order = turn-delta's _OUTCOME_ORDER). The move id
        # string→dex-num conversion uses the same gen3_movedex single source MovesEncoder uses.
        def _last_action_tuple(side: str) -> Optional[Tuple[Any, ...]]:
            if pair_history is None:
                return None
            mid, was_switch, outcome, crit = pair_history.last_action(side)
            num = 0.0
            if mid:
                md = gen3_movedex.get(mid)
                num = float(md.num) if md is not None else 0.0
            return (num, was_switch,
                    1.0 if outcome == "hit" else 0.0,
                    1.0 if outcome == "miss" else 0.0,
                    1.0 if outcome == "fail" else 0.0,
                    crit)
        _la_ours = _last_action_tuple("ours")
        _la_opp = _last_action_tuple("opp")

        # 1. Our Team — current-board per-mon facts read through the LiveView slot.
        our_team_list = self.get_team_list(battle, is_opponent=False)
        for i in range(TEAM_SIZE):
            mon = our_team_list[i] if i < len(our_team_list) else None
            live_mon = live.ours.get(mon.species) if (live is not None and mon is not None) else None
            rec = (recency.values("ours", mon.species)
                   if (recency is not None and mon is not None) else None)
            is_active = 1.0 if (mon and mon.active) else 0.0
            mon_vec = self.pokemon_encoder.encode(
                mon, battle, is_own=True, live_mon=live_mon, sleep_sources=sleep_sources,
                recency_vals=rec,
                last_action_vals=(_la_ours if is_active > 0.5 else None),
            )

            start = OFFSET_OUR_TEAM + (i * POKEMON_FULL_DIM)
            vec[start : start + POKEMON_VECTOR_DIM] = mon_vec
            vec[start + POKEMON_ACTIVE_OFFSET] = is_active
            # gen3_entity_rehome_v1: the OUR-side trapping bits ride the trapped ENTITY — the
            # active mon's slot (server-authoritative LegalActions facts; bench slots stay 0).
            # trapped: confirmed cannot switch (redundant with the mask but explicit);
            # maybe_trapped: the opponent MIGHT be trapping us — the one signal the model has
            # no other way to see.
            if is_active > 0.5 and legal is not None:
                vec[start + POKEMON_TRAPPED_OFFSET] = 1.0 if legal.trapped else 0.0
                vec[start + POKEMON_MAYBE_TRAPPED_OFFSET] = 1.0 if legal.maybe_trapped else 0.0

        # 2. Opponent Team — HP block populated from the tracker when supplied.
        opponents = self.get_team_list(battle, is_opponent=True)

        for i in range(TEAM_SIZE):
            mon = opponents[i] if i < len(opponents) else None
            live_mon = live.opp.get(mon.species) if (live is not None and mon is not None) else None
            if hp_tracker is not None and mon is not None:
                hp_probs = hp_tracker.get_probs(mon.species)
                hp_known = hp_tracker.is_known(mon.species)
            else:
                hp_probs = None
                hp_known = False
            rec = (recency.values("opp", mon.species)
                   if (recency is not None and mon is not None) else None)
            # Active-ness hoisted above the encode call so the H-A1 last-action tuple can ride
            # the ACTIVE slot's vector (same LiveView-first logic as the flag write below).
            if live_mon is not None:
                _opp_is_active = 1.0 if live_mon.active else 0.0
            else:
                _opp_is_active = 1.0 if (mon and mon is battle.opponent_active_pokemon) else 0.0
            mon_vec = self.pokemon_encoder.encode(
                mon, battle, is_own=False, hp_probs=hp_probs, hp_known=hp_known,
                live_mon=live_mon, sleep_sources=sleep_sources, recency_vals=rec,
                last_action_vals=(_la_opp if _opp_is_active > 0.5 else None),
            )
            # Active flag through the LiveView slot (LivePokemon.active is set at fold time
            # from poke-env's opponent_active_pokemon accessor, so this is byte-identical to
            # the old `mon is battle.opponent_active_pokemon` identity check). Fall back to the
            # raw check only on the plain-Battle / unit-test path where there is no LiveView.
            if live_mon is not None:
                is_active = 1.0 if live_mon.active else 0.0
            else:
                is_active = 1.0 if (mon and mon is battle.opponent_active_pokemon) else 0.0

            start = OFFSET_OPP_TEAM + (i * POKEMON_FULL_DIM)
            vec[start : start + POKEMON_VECTOR_DIM] = mon_vec
            vec[start + POKEMON_ACTIVE_OFFSET] = is_active

        # 3. Active Context + 4. Global Environment — sourced from the same LiveView
        # (current-board read-model folded from the event log).
        our_active = live.ours.active if live else None
        opp_active = live.opp.active if live else None
        vec[OFFSET_CONTEXT : OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM] = \
            self.active_context_encoder.encode(our_active)
        vec[OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM : OFFSET_CONTEXT + (2 * ACTIVE_CONTEXT_DIM)] = \
            self.active_context_encoder.encode(opp_active)

        # 4. Global Environment
        if live is not None:
            vec[OFFSET_GLOBAL : OFFSET_GLOBAL + GLOBAL_ENV_DIM] = \
                self.global_env_encoder.encode(live)
        
        # 5. Reactive Features — fainted counts + active-status read through the LiveView;
        # trapped / maybe_trapped from the LegalActions snapshot; the move-effectiveness
        # matrices stay on the raw battle (see reactive.py).
        vec[OFFSET_REACTIVE : OFFSET_REACTIVE + REACTIVE_DIM] = \
            self.reactive_encoder.encode(battle, hp_tracker=hp_tracker, live=live, legal=legal,
                                         progress_clock=progress_clock)

        # 6. Tier H-A2 (gen3_pair_history_v1): the pair-history block — h[i, j] per (their mon
        # i, our mon j), row-major (opp_slot, our_slot, cell), joined by the SAME team-list
        # order the per-mon blocks use so cell (i, j) names the same two entities all battle.
        # None (standalone/test path) leaves the block zero, like the other optional trackers.
        if pair_history is not None:
            for i in range(TEAM_SIZE):
                _opp_mon = opponents[i] if i < len(opponents) else None
                _opp_sp = _opp_mon.species if _opp_mon is not None else None
                for j in range(TEAM_SIZE):
                    _our_mon = our_team_list[j] if j < len(our_team_list) else None
                    _our_sp = _our_mon.species if _our_mon is not None else None
                    _b = OFFSET_PAIR_HISTORY + (i * TEAM_SIZE + j) * PAIR_HISTORY_CELL_DIM
                    vec[_b : _b + PAIR_HISTORY_CELL_DIM] = \
                        pair_history.pair_values(_opp_sp, _our_sp)

        # 7. Tier H-B (gen3_event_window_v1): the last-N event records, oldest-first, padded
        # with zero rows at the FRONT (most-recent event is always the last valid row — a
        # stable read for the flag-gated event-seat encoder). Ids are embedding ids; no Linear
        # reads this block raw. None (standalone/test path) leaves the block zero like the
        # other optional trackers.
        #
        # gen3_event_col_names_v1: every column is addressed through `EventCol`, the ONE
        # declaration `team_transformer.EventSeats` also reads. Bare integer literals here (and
        # matching ones there) were a producer/consumer pair bound by position with nothing
        # relating them — the class the positional-binding sweep convicted five times. The
        # members are ints, so this compiles to the same arithmetic and emits the same bytes.
        if event_window is not None:
            from agents import gen3_data
            _c = EVENT_COL       # the PLAIN-INT mirror (see constants.py) + a LOAD_FAST alias:
                                 # an EventCol member here costs ~36% of this write loop
            _rows = event_window.window()[-EVENT_WINDOW_N:]
            _cur = event_window.turn
            _base = OFFSET_EVENT_WINDOW + (EVENT_WINDOW_N - len(_rows)) * EVENT_TOKEN_DIM
            for _ri, _r in enumerate(_rows):
                _o = _base + _ri * EVENT_TOKEN_DIM
                _actor = gen3_data.species.get(_r["actor"]) if _r["actor"] else None
                _target = gen3_data.species.get(_r["target"]) if _r["target"] else None
                _mv = gen3_movedex.get(_r["move_id"]) if _r["move_id"] else None
                vec[_o + _c.TYPE] = float(_r["t"])
                vec[_o + _c.ACTOR_SPECIES] = float(_actor.num) if _actor is not None else 0.0
                vec[_o + _c.ACTOR_SIDE] = (1.0 if _r["side"] == "ours"
                                           else (-1.0 if _r["side"] == "opp" else 0.0))
                vec[_o + _c.TARGET_SPECIES] = float(_target.num) if _target is not None else 0.0
                vec[_o + _c.MOVE] = float(_mv.num) if _mv is not None else 0.0
                _mag = _r["hp_delta"]
                vec[_o + _c.MAGNITUDE] = (max(-1.0, min(1.0, _mag)) if _r["t"] == EVENT_T_MOVE
                                          else max(-1.0, min(1.0, _mag / 6.0)))
                if _r["t"] == EVENT_T_MOVE:
                    vec[_o + _c.OUT_HIT] = 0.0 if (_r["missed"] or _r["failed"]) else 1.0
                    vec[_o + _c.OUT_MISS] = 1.0 if _r["missed"] else 0.0
                    vec[_o + _c.OUT_FAIL] = 1.0 if _r["failed"] else 0.0
                    vec[_o + _c.CRIT] = 1.0 if _r["crit"] else 0.0
                    # the eff one-hot is INDEXED, not written per column — `EVENT_EFF_GROUP`'s
                    # order is the contract and its contiguity is asserted by event_window_test.
                    vec[_o + _c.EFF_NEUTRAL + int(_r["eff"])] = 1.0
                vec[_o + _c.WE_FIRST] = 1.0 if _r["we_first"] else 0.0
                vec[_o + _c.STATUS] = float(_r["status"])
                # gen3_frame_deletion_v1: `.get` because only the CANT branch sets the key —
                # every other record type leaves it absent, which must read as a clean 0.
                vec[_o + _c.CANT] = float(cant_reason_id(_r.get("cant")))
                # gen3_event_semantics_v1 — same `.get` reasoning: only the FAINT / ITEM branches
                # set their key, and every other row must read a clean 0.
                vec[_o + _c.FAINT_CAUSE] = float(faint_cause_id(_r.get("faint_cause")))
                vec[_o + _c.ITEM_TRANSITION] = float(_r.get("item_tr", 0))
                _ago = max(0, _cur - int(_r["turn"]))
                vec[_o + _c.TURNS_AGO] = math.log1p(min(_ago, 10)) / math.log(11.0)
                vec[_o + _c.FORCED_WINDOW] = float(_r["forced_window"])
                vec[_o + _c.VALID] = 1.0

        return vec

    def get_observation(self, battle: AbstractBattle) -> Dict[str, Any]:
        """
        Standardized entry point for getting the full observation dictionary.
        Includes both the encoded state vector and the action mask.
        """
        if battle.wait:
            error_msg = f"⚠️ [OBSERVATION] CRITICAL: Observation requested while battle.wait is True for {battle.battle_tag}"
            print(error_msg)
            raise RuntimeError(error_msg)
            
        obs = self.encode(battle)
        mask = Gen3ActionMasker.get_mask(battle)
        return {
            "observation": obs,
            "action_mask": mask
        }

    def get_layout(self) -> Dict[str, Any]:
        pokemon_layout = self.pokemon_encoder.get_layout()
        return {
            "parts": {
                "our_team": {
                    "start": OFFSET_OUR_TEAM, 
                    "end": OFFSET_OPP_TEAM, 
                    "reshape": (TEAM_SIZE, POKEMON_FULL_DIM)
                },
                "opp_team": {
                    "start": OFFSET_OPP_TEAM, 
                    "end": OFFSET_CONTEXT, 
                    "reshape": (TEAM_SIZE, POKEMON_FULL_DIM)
                },
                "context": {
                    "start": OFFSET_CONTEXT, 
                    "end": OFFSET_GLOBAL, 
                    "reshape": (2, self.active_context_encoder.dimension)
                },
                "global": {
                    "start": OFFSET_GLOBAL, 
                    "end": OFFSET_REACTIVE, 
                    "dim": self.global_env_encoder.dimension
                },
                "reactive": {
                    "start": OFFSET_REACTIVE, 
                    "end": self.dimension, 
                    "dim": self.reactive_encoder.dimension
                },
                "pair_history": {
                    "start": OFFSET_PAIR_HISTORY,
                    "end": OFFSET_PAIR_HISTORY + PAIR_HISTORY_DIM,
                    "reshape": (TEAM_SIZE, TEAM_SIZE, PAIR_HISTORY_CELL_DIM)
                }
            },
            "pokemon": pokemon_layout,
            # gen3_frame_deletion_v1: total_dim == base_dim; the prev_mask_dim / turn_delta_dim /
            # n_history_turns / turn_history_offset / turn_history_dim keys are DELETED with their
            # blocks. Consumers that sliced by them (ObsUnpack, the prober's offsets, the schema)
            # are updated in the same pass — a key left behind reading 0 would be sliced silently.
            "total_dim": self.dimension,
            "base_dim": self.base_dimension,
            "active_context_dim": ACTIVE_CONTEXT_DIM,
            "pair_history_offset": OFFSET_PAIR_HISTORY,
            "pair_history_dim": PAIR_HISTORY_DIM,
            "pair_history_cell_dim": PAIR_HISTORY_CELL_DIM,
            "event_window_offset": OFFSET_EVENT_WINDOW,
            "event_window_dim": EVENT_WINDOW_DIM,
            "event_window_n": EVENT_WINDOW_N,
            "event_token_dim": EVENT_TOKEN_DIM,
            "reactive_layout": _ReactiveEncoder().get_layout(),
            "global_layout": self.global_env_encoder.get_layout(),
            "max_species": 400,
            "species_embedding_dim": 32,
            "max_moves": 400,
            "move_embedding_dim": 16,
            "max_items": 600,
            "item_embedding_dim": 16,
            "max_abilities": 100,
            "ability_embedding_dim": 16,
            "max_types": 20, # 18 types + placeholders
            "type_embedding_dim": 16
        }

    def get_features_extractor_kwargs(self) -> Dict[str, Any]:
        return {
            "layout": self.get_layout(),
            "mappings": self.mappings
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        desc: Dict[str, Any] = {"our_team": [], "opp_team": []}
        
        # 1. Teams
        for i in range(TEAM_SIZE):
            start = OFFSET_OUR_TEAM + (i * POKEMON_FULL_DIM)
            mon_vec = vector[start : start + POKEMON_VECTOR_DIM]
            is_active = vector[start + POKEMON_ACTIVE_OFFSET] > 0.5
            if np.any(mon_vec):
                mon_desc = self.pokemon_encoder.describe_vector(mon_vec)
                mon_desc["active"] = is_active
                desc["our_team"].append(mon_desc)
                
            start_opp = OFFSET_OPP_TEAM + (i * POKEMON_FULL_DIM)
            opp_vec = vector[start_opp : start_opp + POKEMON_VECTOR_DIM]
            is_active_opp = vector[start_opp + POKEMON_ACTIVE_OFFSET] > 0.5
            if np.any(opp_vec):
                opp_desc = self.pokemon_encoder.describe_vector(opp_vec)
                opp_desc["active"] = is_active_opp
                desc["opp_team"].append(opp_desc)
        
        # 2. Context
        our_active_ctx = vector[OFFSET_CONTEXT : OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM]
        opp_active_ctx = vector[OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM : OFFSET_CONTEXT + (2 * ACTIVE_CONTEXT_DIM)]
        desc["our_active"] = self.active_context_encoder.describe_vector(our_active_ctx)
        desc["opp_active"] = self.active_context_encoder.describe_vector(opp_active_ctx)
        
        # 3. Global
        global_vec = vector[OFFSET_GLOBAL : OFFSET_GLOBAL + GLOBAL_ENV_DIM]
        desc["world"] = self.global_env_encoder.describe_vector(global_vec)
        
        # 4. Reactive
        reactive_vec = vector[OFFSET_REACTIVE : OFFSET_REACTIVE + REACTIVE_DIM]
        desc["momentum"] = self.reactive_encoder.describe_vector(reactive_vec)

        # gen3_frame_deletion_v1: there is no TurnDelta tail to describe — the obs ends at base.
        # What HAPPENED last turn is read from the H-B event window instead (`event_window`),
        # which is the block that replaced these frames.

        return desc

    def integrity_check(self, vector: np.ndarray) -> Tuple[List[str], bool]:
        warnings = []
        is_critical = False
        desc = self.describe_vector(vector)
        
        # 1. Active Pokémon Check
        our_active = [mon for mon in desc['our_team'] if mon.get('active')]
        if len(our_active) > 1:
            warnings.append(f"CRITICAL: Multiple active Pokémon on our team: {[m['species'] for m in our_active]}")
            is_critical = True
        elif len(our_active) == 0:
            warnings.append("Note: No active Pokémon found on our team.")
            
        opp_active = [mon for mon in desc['opp_team'] if mon.get('active')]
        if len(opp_active) > 1:
            warnings.append(f"CRITICAL: Multiple active Pokémon on opponent team: {[m['species'] for m in opp_active]}")
            is_critical = True
            
        # 2. HP/Fainted Consistency
        fainted_our_list = len([mon for mon in desc['our_team'] if float(mon['hp'].strip('%')) == 0])
        fainted_our_momentum = desc['momentum']['fainted_our']
        if fainted_our_list != fainted_our_momentum:
             warnings.append(f"CRITICAL: Our fainted count mismatch! Team list ({fainted_our_list}) != momentum ({fainted_our_momentum})")
             is_critical = True
             
        fainted_opp_list = [mon['species'] for mon in desc['opp_team'] if float(mon['hp'].strip('%')) == 0]
        fainted_opp_count = len(fainted_opp_list)
        fainted_opp_momentum = desc['momentum']['fainted_opp']
        if fainted_opp_momentum > fainted_opp_count:
             warnings.append(f"CRITICAL: Opponent fainted count (momentum={fainted_opp_momentum}) > seen in team list ({fainted_opp_count}). Team fainted: {fainted_opp_list}")
             is_critical = True
        elif fainted_opp_momentum < fainted_opp_count:
             warnings.append(f"Mismatch: Opponent fainted count (momentum={fainted_opp_momentum}) < seen in team list ({fainted_opp_count}). Team fainted: {fainted_opp_list}")
             is_critical = True

        return warnings, is_critical
