import numpy as np
from .base import ObservationEncoder
from . import assembler as _assembler
from .assembler import describe_offset, write_event_row
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
    POKEMON_RECENCY_OFFSET,
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
    ACTIVE_CONTEXT_DIM,
    GLOBAL_ENV_DIM
)
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Dict, Any, List, Optional, Tuple
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
               pair_history: Any = None, event_window: Any = None,
               assembler: Any = None) -> np.ndarray:
        """Encode the full base observation vector.

        assembler: optional :class:`~agents.observation.assembler.ObsAssembler`
        (`gen3_obs_assembler_v1`) — the per-episode incremental cache, owned by the
        ``EpisodeTracker``. This method is then the **scheduler**: the per-block writers below
        are unchanged and run either way, and the assembler only decides which of them can be
        skipped because nothing they read has moved. ``None`` (every standalone / offline /
        unit-test caller) is the full rebuild, which is also the oracle the byte-identity fuzz
        compares against. The emitted vector is IDENTICAL either way — that is the contract, not
        an aspiration, and it is what licenses this to be an internal swap with no flag.

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

        # gen3_obs_assembler_v1 — the scheduler decision. `warm` means every block below may
        # consult the cache; otherwise this is a full rebuild into (and seeding) the cache.
        # ⚠️ `asm` is the assembler ONLY when it is actually usable. Rebinding here rather
        # than testing `assembler is not None` at each use site is deliberate: an assembler
        # threaded by a caller that has no strict view (a plain poke-env Battle, a unit-test
        # mock) must be left completely untouched — writing the vector elsewhere and then
        # `commit()`-ing it would mark an UNWRITTEN buffer as ready, and the next real decision
        # would serve zeros from the warm path. Its fold cursor is monotone, so skipping a
        # decision costs a wider dirty set next time and nothing else.
        asm: Any = assembler if (strict is not None and live is not None) else None
        warm: bool = False
        vec: np.ndarray
        if asm is not None:
            warm = bool(asm.prepare(
                strict, live,
                (hp_tracker is not None, legal is not None, progress_clock is not None,
                 recency is not None, pair_history is not None, event_window is not None),
                hp_tracker.revision if hp_tracker is not None else None,
            ))
            vec = asm.buf if warm else asm.begin_full()
        else:
            vec = np.zeros(self.base_dimension, dtype=np.float32)

        # Sleep WAKE belief sources (gen3_sleep_wake_belief_v1): the Rest-source +
        # Sleep-Talk-reliability of each asleep mon. `build_sleep_sources` folds the WHOLE event
        # log (twice) per encode and is gated on "is anyone actually asleep" so the common
        # no-sleep decision pays nothing; the assembler folds the same two event families
        # incrementally, which removes the O(turns²) shape on the sleep-heavy decisions where
        # the gate does not save you.
        any_asleep = live is not None and any(
            m.status == "slp" for m in (*live.ours.mons, *live.opp.mons)
        )
        if not any_asleep:
            sleep_sources = None
        elif asm is not None:
            sleep_sources = asm.sleep_sources()
        else:
            sleep_sources = build_sleep_sources(battle)

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
        #
        # gen3_obs_assembler_v1: a CLEAN slot keeps the 119 dims already in the buffer and pays
        # only its 3 recency dims, which are a deterministic function of the turn number and
        # therefore move under every mon on every turn (census §1.2 DENSE). The three appended
        # tail dims are rewritten unconditionally for the same reason the actives are always
        # dirty — they are request-sourced, and a cached request bit that survives one decision
        # too long is exactly the misalignment class gen3_op_move_align_v1 exists to prevent.
        our_team_list = self.get_team_list(battle, is_opponent=False)
        for i in range(TEAM_SIZE):
            mon = our_team_list[i] if i < len(our_team_list) else None
            species = mon.species if mon is not None else None
            live_mon = live.ours.get(species) if (live is not None and mon is not None) else None
            rec = (recency.values("ours", species)
                   if (recency is not None and mon is not None) else None)
            is_active = 1.0 if (mon and mon.active) else 0.0
            start = OFFSET_OUR_TEAM + (i * POKEMON_FULL_DIM)

            if warm and asm.slot_is_clean("ours", i, species):
                if rec is not None:
                    vec[start + POKEMON_RECENCY_OFFSET] = rec[0]
                    vec[start + POKEMON_RECENCY_OFFSET + 1] = rec[1]
                    vec[start + POKEMON_RECENCY_OFFSET + 2] = rec[2]
            else:
                mon_vec = self.pokemon_encoder.encode(
                    mon, battle, is_own=True, live_mon=live_mon, sleep_sources=sleep_sources,
                    recency_vals=rec,
                    last_action_vals=(_la_ours if is_active > 0.5 else None),
                )
                vec[start : start + POKEMON_VECTOR_DIM] = mon_vec
                if asm is not None:
                    asm.note_slot("ours", i, species)

            vec[start + POKEMON_ACTIVE_OFFSET] = is_active
            # gen3_entity_rehome_v1: the OUR-side trapping bits ride the trapped ENTITY — the
            # active mon's slot (server-authoritative LegalActions facts; bench slots stay 0).
            # trapped: confirmed cannot switch (redundant with the mask but explicit);
            # maybe_trapped: the opponent MIGHT be trapping us — the one signal the model has
            # no other way to see.
            if is_active > 0.5 and legal is not None:
                vec[start + POKEMON_TRAPPED_OFFSET] = 1.0 if legal.trapped else 0.0
                vec[start + POKEMON_MAYBE_TRAPPED_OFFSET] = 1.0 if legal.maybe_trapped else 0.0
            else:
                vec[start + POKEMON_TRAPPED_OFFSET] = 0.0
                vec[start + POKEMON_MAYBE_TRAPPED_OFFSET] = 0.0

        # 2. Opponent Team — HP block populated from the tracker when supplied.
        opponents = self.get_team_list(battle, is_opponent=True)

        for i in range(TEAM_SIZE):
            mon = opponents[i] if i < len(opponents) else None
            species = mon.species if mon is not None else None
            live_mon = live.opp.get(species) if (live is not None and mon is not None) else None
            rec = (recency.values("opp", species)
                   if (recency is not None and mon is not None) else None)
            # Active-ness hoisted above the encode call so the H-A1 last-action tuple can ride
            # the ACTIVE slot's vector (same LiveView-first logic as the flag write below).
            if live_mon is not None:
                _opp_is_active = 1.0 if live_mon.active else 0.0
            else:
                _opp_is_active = 1.0 if (mon and mon is battle.opponent_active_pokemon) else 0.0
            start = OFFSET_OPP_TEAM + (i * POKEMON_FULL_DIM)

            if warm and asm.slot_is_clean("opp", i, species):
                if rec is not None:
                    vec[start + POKEMON_RECENCY_OFFSET] = rec[0]
                    vec[start + POKEMON_RECENCY_OFFSET + 1] = rec[1]
                    vec[start + POKEMON_RECENCY_OFFSET + 2] = rec[2]
            else:
                if hp_tracker is not None and mon is not None:
                    hp_probs = hp_tracker.get_probs(species)
                    hp_known = hp_tracker.is_known(species)
                else:
                    hp_probs = None
                    hp_known = False
                mon_vec = self.pokemon_encoder.encode(
                    mon, battle, is_own=False, hp_probs=hp_probs, hp_known=hp_known,
                    live_mon=live_mon, sleep_sources=sleep_sources, recency_vals=rec,
                    last_action_vals=(_la_opp if _opp_is_active > 0.5 else None),
                )
                vec[start : start + POKEMON_VECTOR_DIM] = mon_vec
                if asm is not None:
                    asm.note_slot("opp", i, species)

            # Active flag through the LiveView slot (LivePokemon.active is set at fold time
            # from poke-env's opponent_active_pokemon accessor, so this is byte-identical to
            # the old `mon is battle.opponent_active_pokemon` identity check). Fall back to the
            # raw check only on the plain-Battle / unit-test path where there is no LiveView.
            vec[start + POKEMON_ACTIVE_OFFSET] = _opp_is_active

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
                                         progress_clock=progress_clock,
                                         wish_pending=(asm.wish_pending(live.turn)
                                                       if asm is not None else None))

        # 6. Tier H-A2 (gen3_pair_history_v1): the pair-history block — h[i, j] per (their mon
        # i, our mon j), row-major (opp_slot, our_slot, cell), joined by the SAME team-list
        # order the per-mon blocks use so cell (i, j) names the same two entities all battle.
        # None (standalone/test path) leaves the block zero, like the other optional trackers.
        #
        # Assembled as ONE list and written with ONE slice assignment. Every cell's
        # `recency_of_last_pairing` ticks on every turn, so there is nothing here to cache —
        # which made 36 five-float numpy slice assignments the largest single cost left on the
        # warm path once the per-mon slots stopped being rebuilt. The species are hoisted out
        # of the inner loop for the same reason: they were re-read 36 times.
        if pair_history is not None:
            _pv = pair_history.pair_values
            _our_sps = [(our_team_list[j].species if j < len(our_team_list) else None)
                        for j in range(TEAM_SIZE)]
            _blk: List[float] = []
            for i in range(TEAM_SIZE):
                _opp_mon = opponents[i] if i < len(opponents) else None
                _opp_sp = _opp_mon.species if _opp_mon is not None else None
                for _our_sp in _our_sps:
                    _blk.extend(_pv(_opp_sp, _our_sp))
            vec[OFFSET_PAIR_HISTORY : OFFSET_PAIR_HISTORY + PAIR_HISTORY_DIM] = _blk

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
        #
        # gen3_obs_assembler_v1: the flat layout FRONT-pads, so an append shifts every retained
        # row — a representation artifact, not churn in the values (21 of 22 columns of a row
        # never change after the append). The warm path therefore shifts the block once and
        # writes only the new rows; the cold path below writes them all. Both go through the
        # SAME `write_event_row`, so the two schedulers cannot drift in content.
        if event_window is not None:
            _rows = event_window.window()[-EVENT_WINDOW_N:]
            _cur = event_window.turn
            if warm:
                asm.update_window(_rows, _cur, event_window.open_records())
            else:
                _base = OFFSET_EVENT_WINDOW + (EVENT_WINDOW_N - len(_rows)) * EVENT_TOKEN_DIM
                for _ri, _r in enumerate(_rows):
                    write_event_row(vec, _base + _ri * EVENT_TOKEN_DIM, _r, _cur)
                if asm is not None:
                    asm.seed_window(_rows, _cur)

        if asm is not None:
            asm.commit()
            # The buffer is persistent and mutated in place next decision; every consumer
            # (SB3's rollout buffer, the recorder, the materializer) holds the obs past this
            # call, so it gets its own copy. ~2.5k float32 — single-digit microseconds.
            out: np.ndarray = vec.copy()
            if _assembler.OBS_VERIFY:
                self._verify_against_full_rebuild(
                    out, battle, hp_tracker, legal, progress_clock, recency,
                    pair_history, event_window)
            return out
        return vec

    def _verify_against_full_rebuild(
        self, got: np.ndarray, battle: AbstractBattle, hp_tracker: Any, legal: Any,
        progress_clock: Any, recency: Any, pair_history: Any, event_window: Any,
    ) -> None:
        """``GEN3AI_OBS_VERIFY=1`` shadow mode — encode the SAME decision the cold way and
        raise on the first differing index.

        The oracle is the full rebuild run fresh, not a second read of cache state: it re-folds
        the Wish and sleep-source maps from the whole log and re-encodes all 12 slots and all 32
        event rows. Cheap enough to leave on in a smoke test and in the fuzz tier permanently;
        far too expensive for training, which is why it is opt-in rather than defaulted.
        """
        want = self.encode(battle, hp_tracker=hp_tracker, legal=legal,
                           progress_clock=progress_clock, recency=recency,
                           pair_history=pair_history, event_window=event_window,
                           assembler=None)
        if np.array_equal(got, want):
            return
        bad = np.flatnonzero(got != want)
        first = int(bad[0])
        raise AssertionError(
            f"[GEN3AI_OBS_VERIFY] incremental obs != full rebuild at turn "
            f"{getattr(battle, 'turn', '?')}: {bad.size} differing dims, first at index "
            f"{first} ({describe_offset(first)}) — incremental {got[first]!r} vs full "
            f"{want[first]!r}. All differing indices: {bad[:32].tolist()}"
            f"{' …' if bad.size > 32 else ''}"
        )

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
