"""``TurnDelta`` — the per-decision HISTORY record's FIELD LAYOUT (the fold is deleted).

The Python fold that built it from the event log (``build_from_events`` over two ``BattleContext``
snapshots) is DELETED with the Python battle layer (T27 P6 slice 6d-2): the Rust core folds the same
window (``src/rust_sim/src/trackers/delta.rs``) and the prober reads its frozen projection. What stays is
the FROZEN field layout and its constants, for the kept readers: the feature-coverage probes
(``agents.model.feature_coverage._support`` builds ``TurnDelta``s by hand and encodes them with
``TurnDeltaEncoder``, the archive decoder), and ``SELF_KO_MOVES`` (``main.prober.forensics``).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, TYPE_CHECKING
import numpy as np

from agents.gen3_mechanics import BOOST_DIM

if TYPE_CHECKING:
    from agents.enums import Status


class DamagingMoveEvent(NamedTuple):
    """Per-side snapshot of the last damaging move resolved in a window — OWNED here since P6 of the poke-env
    retirement (it was poke-env's ``abstract_battle.DamagingMoveEvent``, the same five fields in the same order, so
    a fold's values and equality are unchanged; nothing ``isinstance``-checks it). ``target_status`` is the
    defender's status AT MOVE-FIRE TIME (Flash Fire vs a frozen holder thaws in the same hit)."""
    user_species: str
    target_species: str
    target_status: "Optional[Status]"
    move_id: str
    effectiveness: float


# Moves whose user always faints and which always connect when used (a neutral
# hit emits no effectiveness event, so the damaging-event "connected" signal would
# miss them; an immune target DOES emit, so it's covered by the event either way).
SELF_KO_MOVES = frozenset({"explosion", "selfdestruct"})


@dataclass
class TurnDelta:
    """
    Diff between two consecutive BattleContexts.

    Built after each turn completes, capturing what actions were taken and what
    changed. Passed to the reward function and written into the info dict for
    callbacks to consume without touching the battle object.
    """
    # What we did this turn
    our_move_id: str | None       # move ID (e.g. "rockslide"), None if we switched
    our_switch_to: str | None     # species we switched to, None if we moved
    our_prev_active: str          # species that was active at turn start

    # What they did this turn
    opp_move_id: str | None       # move ID from poke-env's last_move tracking; None if switched
    opp_switch_to: str | None     # species they switched to, None if they moved
    opp_prev_active: str
    opp_move_known: bool          # False only when we know they attacked but have no move ID
                                  # (e.g. Explosion aftermath where the attacker is no longer active)

    # HP outcomes per slot (indexed by slot_map from BattleContext)
    our_hp_delta: np.ndarray      # (6,) float32 — negative means damage taken
    opp_hp_delta: np.ndarray      # (6,) float32

    # Faint events this turn
    we_fainted: bool
    opp_fainted: bool

    # Did each side fail to act this turn?
    # Derived from curr_ctx cant_reason — True whenever |cant| fired for that side.
    our_failed_to_move: bool
    our_cant_reason: str | None
    opp_failed_to_move: bool
    opp_cant_reason: str | None

    # Stat-stage deltas for each side's active Pokémon (BOOST_STATS order).
    # Positive = gained a stage, negative = lost a stage this turn.
    # Zero when the active mon switched (new mon starts from its own current stages).
    our_boost_delta: np.ndarray   # (7,) int8
    opp_boost_delta: np.ndarray   # (7,) int8

    # Type-effectiveness of each side's last damaging move (snapshotted from curr_ctx).
    # 0.0=immune, 0.5=resisted, 1.0=neutral, 2.0=super-effective.
    # None when the side switched, used a non-damaging move, or the battle just started.
    our_effectiveness: float | None
    opp_effectiveness: float | None

    # True = we executed our action before the opponent this turn.
    # None when one or both sides performed a normal switch.
    we_moved_first: bool | None

    # Full per-side damaging-move record (user / target / target_status at
    # fire time / move_id / effectiveness), pass-through from the curr_ctx
    # snapshot. None when the side didn't use a damaging move whose
    # effectiveness was confirmed by an explicit emission — preserves the
    # "skip on uncertainty" semantics of the underlying property. Use this
    # instead of (move_id, effectiveness) tuples for attribution-sensitive
    # consumers (reward shaping, replay recording) where the protocol-truth
    # user/target identity matters; the bare opp_move_id / opp_effectiveness
    # fields stay for callers that just need any signal.
    our_damaging_event: Optional[DamagingMoveEvent] = None
    opp_damaging_event: Optional[DamagingMoveEvent] = None

    # True when this delta closes on a forced_switch input request (mid-turn
    # replacement after a faint, end-of-turn replacement). False for normal
    # move-selection turns. Lets the model distinguish half-turn replacement
    # slots from full action-pair slots — without this, the absence of an opp
    # move in a forced-switch slot looks like "opp voluntarily passed."
    phase_is_forced_switch: bool = False

    # Per-slot HP levels AT THE END OF THE TURN (i.e. from curr_ctx). Carried
    # alongside the deltas so the encoder can expose the full HP trajectory
    # to the model across the history window without forcing the transformer
    # to inverse-cumsum delta scalars across attention positions. In slot_map
    # order; zeros for unrevealed slots.
    our_hp_after: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    opp_hp_after: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))

    # HP delta on the named target species of each side's damaging move this
    # turn. our_target_hp_delta = opp's damaging move's target (i.e. our mon
    # that got hit); opp_target_hp_delta = our damaging move's target. None
    # when the side didn't use a damaging move or the target slot can't be
    # resolved. Pairs with the actor/target species IDs in the encoder's
    # slot — gives the model "how hard the named target got hit."
    our_target_hp_delta: Optional[float] = None
    opp_target_hp_delta: Optional[float] = None

    # Per-side move outcome for the turn, one of "hit" / "miss" / "fail", or
    # None when the side switched, was prevented from moving (|cant| — covered
    # by the separate cant one-hot), or used no identifiable move. "hit" means
    # the move connected / did its thing (damage or status applied); "miss" =
    # accuracy miss (|-miss|); "fail" = the move executed but did nothing
    # (Protect on repeat, Substitute on existing sub, |-fail|/|-notarget|/
    # |-nothing|). crit is orthogonal — a "hit" may also crit.
    our_move_outcome: Optional[str] = None
    opp_move_outcome: Optional[str] = None
    our_move_crit: bool = False
    opp_move_crit: bool = False

    # --- multi-KO + cause ------------------------------------------------
    # Count of mons that fainted on each side in this decision window.
    # (we_fainted / opp_fainted are kept as quick bool checks; these carry
    # the exact count for multi-KO turns.)
    our_faint_count: int = 0
    opp_faint_count: int = 0

    # Multi-hot over FAINT_CAUSE_VOCAB (8 dims). A turn with two faints of
    # different causes will have two bits set. Zeros when no mon fainted.
    # Shape: (FAINT_CAUSE_DIM,) float32.
    our_faint_causes: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32)
    )
    opp_faint_causes: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32)
    )

    # --- attempted action ------------------------------------------------
    # The move WE pressed, preserved even when it never fired (cant / frozen /
    # KO-before-acting). Decoded from the raw action index against prev_ctx at
    # build time, so it always reflects genuine intent. Only the MOVE is kept:
    # a pressed switch always executes (switches aren't subject to freeze/sleep/
    # flinch/cant and the mask gates legality), so an attempted_switch_to would
    # always equal our_switch_to. Opp attempted action is not observable.
    our_attempted_move_id: Optional[str] = None

    # --- Status transitions THIS window (folded from the event log) ------
    # The status each side's active GAINED (status_applied) or LOST
    # (status_cured) this decision window — the *event*, distinct from the
    # current-status snapshot in the per-mon block. Lets the history window
    # carry temporal patterns the snapshot can't ("Tyranitar's Toxic was
    # cured, then it Dragon Danced" — Lum-Berry-enabled setup). None when no
    # status change on that side. Stored as poke-env Status enums (the encoder
    # one-hots them). The CAUSE (item vs ability vs natural) is NOT stored
    # here — it lives in the per-mon item/ability block (single source of
    # truth); this is purely the transition.
    our_status_applied: Optional["Status"] = None
    our_status_cured: Optional["Status"] = None
    opp_status_applied: Optional["Status"] = None
    opp_status_cured: Optional["Status"] = None

    # --- Item consumed/removed THIS window (folded from |-enditem|) -------
    # The item id each side's active LOST this window (Berry eaten, Knock Off,
    # Trick). Stored as the id string (reward/replay can use the identity); the
    # ENCODER emits only a BIT ("an item was used") — the WHICH lives in the
    # per-mon item block ([item_id, known, consumed=1]), so putting the id here
    # too would duplicate it. The history just marks the resource event +
    # timing; the per-mon block names it. Parity with ability_activated. None
    # when no item was lost this window.
    our_item_lost: Optional[str] = None
    opp_item_lost: Optional[str] = None

    # --- Trapping: rejected switch (gen3_trapping_signals_v1) ------------
    # ``attempted_switch_rejected``: the server REFUSED a switch we chose this window
    # (|error|[Unavailable choice]) — we tried to pivot and were trapped (Arena Trap /
    # Shadow Tag / Magnet Pull / Mean Look). Folded from the out-of-band CHOICE_REJECTED
    # event (TurnView.ours.attempted_rejected). The "switches always execute" assumption that
    # dropped attempted_switch_to is exactly false here. ``attempted_switch_to``: the species
    # we PRESSED a switch to (intent), decoded from the action index — parity with
    # ``our_attempted_move_id``; set whenever a switch was attempted, so on a rejected pivot it
    # names the mon we tried to bring in (our_switch_to is None — the switch never happened).
    # Only OUR side: the opponent's attempted action is not observable.
    attempted_switch_rejected: bool = False
    attempted_switch_to: Optional[str] = None

    # --- What the α/β label and the progress clock need beyond the frozen layout ---------------
    # gen3_intent_label_semantics_fixes_v1 / gen3_progress_clock_attribution_fix_v1. NONE of these
    # is encoded into the observation; each is folded from the event log by `TurnView`.
    #
    # ``opp_dragged`` — the opponent's active was DRAGGED in this window (our Roar / Whirlwind), so
    # its ``opp_switch_to`` is a mon it never chose (L1, and L2: refused, then dragged).
    # ``opp_switch_is_replacement`` — the opponent's switch-in filled a slot EMPTIED BY A FAINT
    # that lay in the PREVIOUS window (the decision that opened this one saw its active fainted):
    # the free replacement, not a chosen switch (L3). The same-window case is ``opp_fainted``.
    # ``opp_called_via`` — the move that CALLED the opponent's executed move (Sleep Talk → Rest,
    # Mirror Move, Metronome, Assist, Nature Power, Magic Coat, Snatch): the CALLER is the choice (L4).
    # ``opp_choice_overridden`` — an Encore landed on the opponent's mon before it moved this turn,
    # so the move it executed is the encored one, not the one it chose (L5).
    # ``our_move_hit_delta`` — HP fraction our moves' OWN hits took off the opponent (≤ 0); the
    # progress clock's clause (i) requires it (T1).
    opp_dragged: bool = False
    opp_switch_is_replacement: bool = False
    opp_called_via: Optional[str] = None
    opp_choice_overridden: bool = False
    our_move_hit_delta: float = 0.0

    @property
    def opp_resolved_move_id(self) -> Optional[str]:
        """Opp's move id with protocol-truth preference.

        Returns `opp_damaging_event.move_id` when the event is set (the
        |move| line of the just-resolved turn, captured before any |faint|
        or active-slot reshuffle). Falls back to `opp_move_id` for
        non-damaging moves (status, Roar, Calm Mind, BP, switches) where no
        event ever promotes.

        Attribution-sensitive callers (reward shaping, replay recording)
        should use this instead of raw `opp_move_id` — the latter is
        vulnerable to stale `last_move` reads when opp's mon switches
        between snapshots.
        """
        if self.opp_damaging_event is not None:
            return self.opp_damaging_event.move_id
        return self.opp_move_id

    @classmethod
    def empty(cls) -> "TurnDelta":
        return cls(
            our_move_id=None, our_switch_to=None, our_prev_active="NULL",
            opp_move_id=None, opp_switch_to=None, opp_prev_active="NULL",
            opp_move_known=False,
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=np.zeros(6, dtype=np.float32),
            we_fainted=False, opp_fainted=False,
            our_failed_to_move=False, our_cant_reason=None,
            opp_failed_to_move=False, opp_cant_reason=None,
            our_boost_delta=np.zeros(BOOST_DIM, dtype=np.int8),
            opp_boost_delta=np.zeros(BOOST_DIM, dtype=np.int8),
            our_effectiveness=None,
            opp_effectiveness=None,
            we_moved_first=None,
            our_damaging_event=None,
            opp_damaging_event=None,
            phase_is_forced_switch=False,
            our_hp_after=np.zeros(6, dtype=np.float32),
            opp_hp_after=np.zeros(6, dtype=np.float32),
            our_target_hp_delta=None,
            opp_target_hp_delta=None,
            our_move_outcome=None,
            opp_move_outcome=None,
            our_move_crit=False,
            opp_move_crit=False,
        )
