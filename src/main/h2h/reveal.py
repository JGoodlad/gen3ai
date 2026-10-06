"""THE PER-SIDE ORACLE REVEAL of the head-to-head engine (X5 A/B, ``design_x5_belief_tokens.md`` §7.6–§7.7(a)).

An ORACLE checkpoint (one whose ``model_config.json`` records ``oracle_reveal`` ``species`` or ``full``,
``agents.model.oracle_reveal``) was trained on observations whose opponent block is told the other side's team, by the
Rust encoder (``encoder::oracle``). The engine plays it exactly so: the eval core's startup declaration carries a
PER-SIDE level (``Spec.oracle_reveal`` as a ``[p1, p2]`` pair, ``core::spec::Reveal``), and each side's parse chain is
given the OTHER side's team at its own side's level — the same ``Oracle`` path, the same declared field sets, as the
training core (``rust_env/tests/oracle_reveal_test.rs::a_per_side_reveal_writes_each_side_exactly_as_the_symmetric_core_of_its_level``).

A plan declares ONE mode (:data:`MODES`), and every cell's two levels follow from it and from the two checkpoints'
RECORDED levels (:func:`side_levels`), never typed:

* ``off`` — no reveal on either side: today's engine, byte-identical (protocol ``gen3_eval_protocol_v1_h2h``). An
  oracle checkpoint is REFUSED here (:class:`RevealModeError`): it would be fed observations it never trained on.
* ``one_sided`` — ONE-SIDED CLAIRVOYANCE (§7.7(a)'s PRIMARY read, the ceiling C): the oracle side at its recorded
  level, the other side ``off`` — it plays exactly as trained.
* ``both_sided`` — the other side is ALSO told the oracle's team, at the same level (SECONDARY, DESCRIPTIVE: that
  side never saw a revealed team in training, so its play may be degraded by the unfamiliar input; never read as
  "information parity").

A reveal mode needs EXACTLY ONE oracle side per cell: a cell with none, or an oracle against an oracle, is a typed
:class:`RevealModeError` (§7.7(a) pairs an oracle seed with a blob or fixed_mass seed). An oracle side at a level
other than the one it recorded is :class:`RevealLevelMismatch` (:func:`check_side_levels`, which the engine runs on
every cell against the core it is about to play on). Each mode is its own eval PROTOCOL (:data:`PROTOCOL_OF_MODE`),
so its rows are their own regime and never pool with ``off`` rows; every row also carries the two levels in
``compute.oracle_reveal``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

from main.h2h.errors import H2HError

if TYPE_CHECKING:
    from main.h2h.play import PlayerRef

#: The plan's reveal modes, ``off`` first (the default: today's engine).
MODES = ("off", "one_sided", "both_sided")
#: The eval protocol each mode plays (``eval_ledger.schema.PROTOCOLS``).
PROTOCOL_OF_MODE: Dict[str, str] = {
    "off": "gen3_eval_protocol_v1_h2h",
    "one_sided": "gen3_eval_protocol_v1_h2h_oracle_one_sided",
    "both_sided": "gen3_eval_protocol_v1_h2h_oracle_both_sided",
}
OFF_OFF: Tuple[str, str] = ("off", "off")

Levels = Tuple[str, str]


class RevealModeError(H2HError):
    """A plan's reveal MODE that the cell's checkpoints do not fit: an oracle checkpoint under ``off``, a reveal mode
    over a cell with no oracle side, or an oracle against an oracle."""


class RevealLevelMismatch(H2HError):
    """A side about to play at a reveal level other than the one its checkpoint and the mode require — above all an
    ORACLE checkpoint at a level it did not train under (a ``species`` checkpoint told the whole set, or nothing)."""


def recorded_level(ref: "PlayerRef") -> str:
    """The ``oracle_reveal`` level ``ref``'s run recorded (``"off"`` for a run that predates the field)."""
    from agents.model.oracle_reveal import recorded_oracle_reveal

    return recorded_oracle_reveal(ref.zip_path)


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise RevealModeError(f"oracle reveal mode {mode!r} is not one of {MODES}")
    return mode


def side_levels(mode: str, player_level: str, opponent_level: str, cell: str = "the cell") -> Levels:
    """The cell's ``(player = p1, opponent = p2)`` reveal levels under ``mode``, from the two checkpoints' RECORDED
    levels (module docs). Typed refusals: :class:`RevealModeError`."""
    check_mode(mode)
    oracle = [lv for lv in (player_level, opponent_level) if lv != "off"]
    if mode == "off":
        if oracle:
            who = "player" if player_level != "off" else "opponent"
            raise RevealModeError(
                f"{cell}: the {who} was trained under --oracle-reveal {oracle[0]} (a DIAGNOSTIC observation mode) and "
                "the plan's reveal mode is `off`, so it would be fed observations it never trained on — declare "
                "--oracle-reveal-mode one_sided (the oracle side told the other team, X5 A/B §7.7(a)'s primary read) or "
                "both_sided")
        return OFF_OFF
    if not oracle:
        raise RevealModeError(f"{cell}: reveal mode {mode} needs an ORACLE checkpoint on one side, and neither side "
                              "recorded an --oracle-reveal level — play it with --oracle-reveal-mode off")
    if len(oracle) == 2:
        raise RevealModeError(f"{cell}: both sides are oracle checkpoints ({player_level} vs {opponent_level}); reveal "
                              f"mode {mode} pairs ONE oracle side with a non-oracle one (X5 A/B §7.7(a)) — an oracle vs "
                              "oracle cell is not a declared mode")
    level = oracle[0]
    if mode == "both_sided":
        return (level, level)
    return (player_level, opponent_level)


def check_side_levels(levels: Levels, mode: str, player_level: str, opponent_level: str,
                      cell: str = "the cell") -> None:
    """REFUSE (:class:`RevealLevelMismatch`) unless ``levels`` — the ``(p1, p2)`` reveal of the core a cell is about
    to play on — are exactly what ``mode`` and the two RECORDED levels require (:func:`side_levels`). An oracle side
    at any level but its own is named first."""
    want = side_levels(mode, player_level, opponent_level, cell)
    if tuple(levels) == want:
        return
    for side, rec, got in (("player (p1)", player_level, levels[0]), ("opponent (p2)", opponent_level, levels[1])):
        if rec != "off" and got != rec:
            raise RevealLevelMismatch(f"{cell}: the {side} was trained under --oracle-reveal {rec} and would play at "
                                      f"{got!r} — an oracle checkpoint plays at its OWN recorded level only")
    raise RevealLevelMismatch(f"{cell}: reveal levels {tuple(levels)} (p1, p2) are not what mode {mode} requires for "
                              f"recorded levels ({player_level}, {opponent_level}): {want}")


def row_block(mode: str, levels: Levels) -> Optional[Dict[str, str]]:
    """The ``compute.oracle_reveal`` stamp of a row (``None`` = absent: a ``v1_h2h`` row is byte-identical to before
    the reveal existed)."""
    if mode == "off":
        return None
    return {"mode": mode, "player": levels[0], "opponent": levels[1]}
