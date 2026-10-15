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
* ``withheld`` — the HONEST-PLAY read (the hidden-information study's knock-in forks, ``design_hidden_info_binding.md``
  §8.4): the oracle-trained checkpoint plays at level ``off`` BY DECLARATION, BELOW the level it recorded, against a
  plain checkpoint — did learning WITH the information change its play WITHOUT it (a dependence cost, or a transfer)?
  Both sides are ``off``, so the games are the ``off`` games' observations exactly; what differs is the DECLARATION,
  which is why it is its own protocol and its rows say whose reveal was withheld (``withheld_level``). It is the one
  explicit exception to ``off``'s refusal of an oracle checkpoint: the refusal stays the default, because an oracle
  checkpoint fed ``off`` observations it never trained on is a dependence read only when it is declared one.

A reveal mode needs EXACTLY ONE oracle side per cell: a cell with none, or an oracle against an oracle, is a typed
:class:`RevealModeError` (§7.7(a) pairs an oracle seed with a blob or fixed_mass seed). ``withheld`` takes the same
rule for its own reasons (:func:`side_levels`): with no oracle side there is nothing to withhold, and with two the
row's stamp could name only ONE withheld level and two dependence costs could not be told apart. An oracle side at a
level other than the one the mode requires is :class:`RevealLevelMismatch` (:func:`check_side_levels`, which the
engine runs on every cell against the core it is about to play on). Each mode is its own eval PROTOCOL
(:data:`PROTOCOL_OF_MODE`), so its rows are their own regime and never pool with the other modes' rows; every row
also carries the two levels in ``compute.oracle_reveal``, and (:func:`oracle_side`) the oracle checkpoint's PROVENANCE:
its recorded level and canonical reveal hash, and whether it is a DIAGNOSTIC fork (the consent record in its lineage).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from main.h2h.errors import H2HError

if TYPE_CHECKING:
    from main.h2h.play import PlayerRef

#: The plan's reveal modes, ``off`` first (the default: today's engine).
MODES = ("off", "one_sided", "both_sided", "withheld")
#: The eval protocol each mode plays (``eval_ledger.schema.PROTOCOLS``).
PROTOCOL_OF_MODE: Dict[str, str] = {
    "off": "gen3_eval_protocol_v1_h2h",
    "one_sided": "gen3_eval_protocol_v1_h2h_oracle_one_sided",
    "both_sided": "gen3_eval_protocol_v1_h2h_oracle_both_sided",
    "withheld": "gen3_eval_protocol_v1_h2h_oracle_withheld",
}
OFF_OFF: Tuple[str, str] = ("off", "off")

Levels = Tuple[str, str]


class RevealModeError(H2HError):
    """A plan's reveal MODE that the cell's checkpoints do not fit: an oracle checkpoint under ``off``, a reveal mode
    over a cell with no oracle side, or an oracle against an oracle."""


class RevealProvenanceMismatch(H2HError):
    """An oracle checkpoint whose recorded reveal level and its lineage's diagnostic-fork consent disagree (the nearest
    consent's child spec does not mean the level the run's ``model_config.json`` records): a row stamped from either
    would name a reveal the checkpoint may not have trained under."""


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
                "--oracle-reveal-mode one_sided (the oracle side told the other team, X5 A/B §7.7(a)'s primary read), "
                "both_sided, or withheld (the oracle side deliberately played BELOW its recorded level: the "
                "honest-play read)")
        return OFF_OFF
    if not oracle:
        raise RevealModeError(f"{cell}: reveal mode {mode} needs an ORACLE checkpoint on one side, and neither side "
                              "recorded an --oracle-reveal level — play it with --oracle-reveal-mode off")
    if len(oracle) == 2:
        why = ("; the row's stamp names ONE withheld level, and two dependence costs in one cell could not be told "
               "apart" if mode == "withheld" else "")
        raise RevealModeError(f"{cell}: both sides are oracle checkpoints ({player_level} vs {opponent_level}); reveal "
                              f"mode {mode} pairs ONE oracle side with a non-oracle one (X5 A/B §7.7(a)) — an oracle vs "
                              f"oracle cell is not a declared mode{why}")
    level = oracle[0]
    if mode == "withheld":
        return OFF_OFF          # the oracle side plays BELOW its recorded level, by declaration
    if mode == "both_sided":
        return (level, level)
    return (player_level, opponent_level)


def check_side_levels(levels: Levels, mode: str, player_level: str, opponent_level: str,
                      cell: str = "the cell") -> None:
    """REFUSE (:class:`RevealLevelMismatch`) unless ``levels`` — the ``(p1, p2)`` reveal of the core a cell is about
    to play on — are exactly what ``mode`` and the two RECORDED levels require (:func:`side_levels`). An oracle side
    at any level but its own is named first; under ``withheld`` the requirement is ``(off, off)``, and ONLY there is
    an oracle side accepted at ``off`` (``off`` refuses it by :class:`RevealModeError` in :func:`side_levels`, the
    reveal modes by this check)."""
    want = side_levels(mode, player_level, opponent_level, cell)
    if tuple(levels) == want:
        return
    if mode == "withheld":
        told = [side for side, lv in (("player (p1)", levels[0]), ("opponent (p2)", levels[1])) if lv != "off"]
        raise RevealLevelMismatch(f"{cell}: reveal mode withheld plays EVERY side at off by declaration, but the "
                                  f"{' and '.join(told)} would be told {tuple(levels)} (p1, p2)")
    for side, rec, got in (("player (p1)", player_level, levels[0]), ("opponent (p2)", opponent_level, levels[1])):
        if rec != "off" and got != rec:
            raise RevealLevelMismatch(f"{cell}: the {side} was trained under --oracle-reveal {rec} and would play at "
                                      f"{got!r} — an oracle checkpoint plays at its OWN recorded level only (below "
                                      "it only under the declared mode withheld)")
    raise RevealLevelMismatch(f"{cell}: reveal levels {tuple(levels)} (p1, p2) are not what mode {mode} requires for "
                              f"recorded levels ({player_level}, {opponent_level}): {want}")


def oracle_side(ref: "PlayerRef", side: str, level: str, cell: str = "the cell") -> Dict[str, Any]:
    """The PROVENANCE of the oracle checkpoint ``ref`` (the cell's ``side``, ``"player"`` / ``"opponent"``, recorded
    at ``level``): what a row's reader needs to tell a DIAGNOSTIC fork's play from a production checkpoint's.

    * ``level`` / ``canonical`` / ``reveal_hash`` — the RECORDED ``--oracle-reveal`` label (``model_config.json``, via
      :func:`recorded_level`), its canonical clause text and the ``InfoReveal`` hash. Derived from the recorded label,
      never read off ``metadata.json``'s per-checkpoint ``oracle_reveal_record``: a diagnostic fork's carries its
      PARENT's extractor mode (``off``), not the fork's.
    * ``diagnostic_fork`` / ``fork_consent`` — whether the checkpoint's lineage carries a ``--diagnostic-fork-consent``
      (``main.train.diagnostic_fork``; the NEAREST stamp, the run's own or an ancestor's: a plain fork of a consented
      run inherits its observation mode) and that stamp's run, schema, reason, date and the two spec hashes. A
      checkpoint trained fresh under ``--oracle-reveal`` carries no consent: ``diagnostic_fork`` False, ``fork_consent``
      null (it is still a diagnostic OBSERVATION mode; it is just not a fork).

    :class:`RevealProvenanceMismatch` if the nearest consent's child spec does not MEAN ``level``."""
    from agents.model.oracle_reveal import record, same
    from agents.training.lineage import diagnostic_fork_stamps, run_dir_of

    rec = record(level)
    run_dir = run_dir_of(ref.zip_path)
    stamps = diagnostic_fork_stamps(run_dir) if run_dir else []
    consent: Optional[Dict[str, Any]] = None
    if stamps:
        near = stamps[0]
        child, parent = near.get("child_spec") or {}, near.get("parent_spec") or {}
        if not same(str(child.get("label") or "off"), level):
            raise RevealProvenanceMismatch(
                f"{cell}: the {side} {ref.id} records --oracle-reveal {level} but the nearest diagnostic-fork consent "
                f"in its lineage ({near.get('run')}, {str(near.get('date') or '?')[:10]}) names the child spec "
                f"{child.get('label')!r} — a row stamped from either would name a reveal it may not have trained under")
        consent = {"run": near.get("run"), "self": bool(near.get("self")), "schema": near.get("schema"),
                   "reason": near.get("reason"), "date": near.get("date"),
                   "parent_spec_hash": parent.get("hash"), "child_spec_hash": child.get("hash")}
    return {"side": side, "run": os.path.basename(os.path.normpath(run_dir)) if run_dir else None,
            "level": level, "canonical": rec["canonical"], "reveal_hash": rec["hash"],
            "diagnostic_fork": consent is not None, "fork_consent": consent}


def cell_oracle(mode: str, player: "PlayerRef", opponent: "PlayerRef", player_level: str, opponent_level: str,
                cell: str = "the cell") -> Optional[Dict[str, Any]]:
    """The cell's oracle checkpoint's provenance (:func:`oracle_side`) under a reveal ``mode``; ``None`` under ``off``
    (the row of before the reveal). The cell's levels have been resolved first (:func:`side_levels`), so exactly one
    side is an oracle."""
    if mode == "off":
        return None
    if player_level != "off":
        return oracle_side(player, "player", player_level, cell)
    return oracle_side(opponent, "opponent", opponent_level, cell)


def row_block(mode: str, levels: Levels, oracle: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The ``compute.oracle_reveal`` stamp of a row (``None`` = absent: a ``v1_h2h`` row is byte-identical to before
    the reveal existed). ``oracle`` is the cell's oracle checkpoint's provenance (:func:`cell_oracle`), carried as
    ``oracle`` on a row that has one (the engine always passes it). A ``withheld`` row also carries ``withheld_level``
    — the oracle side's RECORDED level, which its ``player`` / ``opponent`` ``off`` no longer names — and so REQUIRES
    ``oracle``."""
    if mode == "off":
        return None
    block: Dict[str, Any] = {"mode": mode, "player": levels[0], "opponent": levels[1]}
    if mode == "withheld":
        if oracle is None:
            raise RevealModeError("a withheld row's stamp names the oracle side's RECORDED level (withheld_level): "
                                  "no oracle provenance was given")
        block["withheld_level"] = oracle["level"]
    if oracle is not None:
        block["oracle"] = dict(oracle)
    return block
