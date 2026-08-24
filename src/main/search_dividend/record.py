"""Synthesizing a :class:`ReconstructionRecord` for a battle that is STILL BEING PLAYED.

The whole search tier (``SearchSession.open_root`` / ``expand_many``, and every offline probe
built on it) is anchored on a reconstruction record — and until now a record only existed AFTER
a battle, arriving as the bridge child's ``__RECON__`` frame just before ``__END__``. A depth-1
search at decision time needs the same object mid-battle.

Nothing new has to be captured to get one. ``replay_kernels.js`` reads a record in exactly two
places: ``writeStart`` takes the ``>start`` / ``>player`` lines out of ``input_log``, and
``buildToTurn`` replays ``commands``. Everything else on the dataclass is metadata. So a live
record is three facts we already own:

* the ``>start`` options — WE issue the bridge ``START``, so the format and (when the driver pins
  one) the seed are ours to write down. The seed must be EXPLICIT: a seedless START makes the
  child MINT one and report it only at ``__RECON__`` time, i.e. after the battle we are trying to
  search inside;
* the two ``>player`` payloads — ``run_local_battles`` calls ``get_next_team()`` on each player
  and stashes the result on ``player._current_packed_team``;
* the committed choices, in bridge-arrival order — every one of them passes through
  ``BattleStreamClient.send_message`` as ``/choose <payload>``, which is byte-for-byte the
  ``[side, payload]`` pair a real record stores (verified against a live record: ``['p1', 'switch
  Magneton'], ['p2', 'move earthquake']``).

⚠️ **A synthesized record is a HYPOTHESIS until it is checked.** Command interleaving, a re-sent
choice, a fabricated seed — any of these silently produces a record that replays a DIFFERENT
battle, and the search would then be answering about a position we were never in. So every
consumer of this record runs :func:`~main.search_dividend.determinize.prefix_matches` against the
protocol we actually observed, and a mismatch is a COUNTED fallback rather than a result. That
gate is not defensive decoration: it is the same byte-identity check the hidden-info floor probe
used to validate 535 of 535 determinized worlds, and it is what makes this construction a
measurement instead of an assumption.
"""

from __future__ import annotations

import json
import random
from typing import List, Optional, Sequence, Tuple

from utils.bridge.reconstruction import ReconstructionRecord


def mint_seed(rng: Optional[random.Random] = None) -> str:
    """A fresh explicit sim seed, in the ``sodium,<hex>`` form the bridge and the replay drivers
    both parse (``gen3_bridge_seed_forms_v1`` made every form work on rust as well as node).

    The driver pins a seed per battle because the record needs the RESOLVED one at decision time,
    not at ``__RECON__`` time — see the module header."""
    r = rng or random.SystemRandom()
    return "sodium," + "".join(f"{r.randrange(256):02x}" for _ in range(16))


class LiveRecordBuilder:
    """Accumulates one battle's reconstruction facts while it is being played.

    Thread-affinity: every mutation happens on ``POKE_LOOP`` (the bridge feed, the choice
    send-back and ``choose_move`` all share that one loop), so no locking is needed and the
    command order is exactly the order the child received.
    """

    def __init__(self, *, battle_format: str, seed: str, battle_tag: Optional[str] = None,
                 trainee_username: Optional[str] = None,
                 chunk_sink: Optional[Sequence[tuple]] = None, our_side: str = "p1"):
        self.battle_format = battle_format
        self.seed = seed
        self.battle_tag = battle_tag
        self.trainee_username = trainee_username
        # The runner APPENDS to this list as the battle plays, so reading it is always current —
        # no second copy to keep in sync, and no "capture the chunk" hook to forget.
        self.chunk_sink: Sequence[tuple] = chunk_sink if chunk_sink is not None else []
        self.our_side = our_side
        self._players: dict = {}
        self._commands: List[Tuple[str, str]] = []

    # -- capture ------------------------------------------------------------

    def set_player(self, side: str, name: str, packed_team: str) -> None:
        self._players[side] = {"name": name, "team": packed_team}

    def add_command(self, side: str, payload: str) -> None:
        self._commands.append((side, payload))

    @property
    def our_lines(self) -> List[str]:
        """The our-side protocol we have actually observed — the left-hand side of the gate."""
        return observed_our_lines(self.chunk_sink, self.our_side)

    @property
    def n_commands(self) -> int:
        return len(self._commands)

    def ready(self) -> bool:
        return "p1" in self._players and "p2" in self._players

    # -- emit ---------------------------------------------------------------

    def build(self) -> ReconstructionRecord:
        """The record as of RIGHT NOW. Cheap (a few string joins) — call it per decision.

        The ``>start`` line is written with the same key order the bridge writes, though nothing
        reads it positionally: ``start_options()`` parses it as JSON.
        """
        if not self.ready():
            raise ValueError("both >player payloads must be set before building a record")
        start = {"formatid": self.battle_format, "seed": self.seed}
        input_log = [">start " + json.dumps(start)]
        for side in ("p1", "p2"):
            input_log.append(f">player {side} " + json.dumps(self._players[side]))
        return ReconstructionRecord(
            format_id=self.battle_format,
            prng_seed=self.seed,
            input_log=tuple(input_log),
            commands=tuple(self._commands),
            battle_tag=self.battle_tag,
            trainee_username=self.trainee_username,
        )


# ---------------------------------------------------------------------------
# the choice tap
# ---------------------------------------------------------------------------
#
# The commands are tapped at ``BattleStreamClient._write_choice`` — the LAST point before the
# bytes reach the bridge child. Two reasons it is there and not on ``choose_move``:
#
#   * it sees the choice that was actually SENT, including poke-env's own fallbacks
#     (``/choose default`` from a ``None`` predict, a redecide exhaustion, ``DEFAULT_CHOICE_CHANCE``)
#     — precisely the tokens that killed two production launches when the rust bridge did not
#     model them, and precisely the ones a ``choose_move`` return value never shows;
#   * it needs no per-player wiring, which matters because ``run_local_battles`` REPLACES both
#     players' ``ps_client`` when it attaches the transport. Anything wrapped before the runner
#     starts is thrown away, and anything wrapped from inside the first ``choose_move`` has
#     already missed whichever side moved first.
#
# The tap is a CLASS patch guarded by a module-level active builder. That is deliberate global
# state and it is safe for exactly one reason: the driver plays ONE battle at a time per process
# (``chunk_sink`` is not side-deduped across concurrent battles either — see
# ``local_battle_runner``'s own docstring). :func:`set_active_builder` asserts that rather than
# trusting it.

_ACTIVE: dict = {"builder": None}
_ORIGINAL_WRITE_CHOICE = None


def install_choice_tap() -> None:
    """Patch ``BattleStreamClient._write_choice`` to record into the active builder. Idempotent."""
    global _ORIGINAL_WRITE_CHOICE
    if _ORIGINAL_WRITE_CHOICE is not None:
        return
    from utils.bridge.battle_stream_client import BattleStreamClient

    inner = BattleStreamClient._write_choice

    async def _tapped(self, room: str, choice: str):
        builder = _ACTIVE["builder"]
        if builder is not None:
            builder.add_command(self._side, choice)
        return await inner(self, room, choice)

    BattleStreamClient._write_choice = _tapped        # type: ignore[method-assign]
    _ORIGINAL_WRITE_CHOICE = inner


def uninstall_choice_tap() -> None:
    """Put the original method back. Exists for the TESTS, and it is not fussiness: a class patch
    installed by one test file persists for the whole pytest process, so it would ride into every
    bridge test collected after it. A probe that leaves the tree patched is a probe that gets
    blamed for someone else's failure."""
    global _ORIGINAL_WRITE_CHOICE
    if _ORIGINAL_WRITE_CHOICE is None:
        return
    from utils.bridge.battle_stream_client import BattleStreamClient

    BattleStreamClient._write_choice = _ORIGINAL_WRITE_CHOICE   # type: ignore[method-assign]
    _ORIGINAL_WRITE_CHOICE = None


def set_active_builder(builder: Optional[LiveRecordBuilder]) -> None:
    """Route the tap at ``builder``. Refuses to displace a live one — two overlapping battles
    would interleave their commands into one record, and the resulting record would replay a
    battle that never happened while looking perfectly well-formed."""
    if builder is not None and _ACTIVE["builder"] is not None:
        raise RuntimeError(
            "a LiveRecordBuilder is already active — the search-dividend driver plays one "
            "battle at a time per process; overlapping battles would interleave their commands")
    _ACTIVE["builder"] = builder


def active_builder() -> Optional[LiveRecordBuilder]:
    return _ACTIVE["builder"]


def observed_our_lines(chunk_sink: Sequence[tuple], side: str) -> List[str]:
    """Our-side protocol lines from a ``run_local_battles(chunk_sink=…)`` list.

    The sink is a flat ``[(side, chunk), …]`` for a SINGLE battle — the runner does not
    side-dedupe it across concurrent battles, which is why the driver runs one battle per call.
    """
    return [ln for (s, chunk) in chunk_sink if s == side for ln in chunk.split("\n")]
