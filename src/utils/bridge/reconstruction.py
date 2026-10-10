"""Battle reconstruction records — the record type and the offline replay / re-roll primitives.

A battle's **reconstruction record** is the full-information (referee-view)
data needed to rebuild it bit-for-bit offline: the resolved PRNG seed, both packed
teams, the sim's own ``inputLog``, and the raw command sequence (a Rust core trace
writes it as the ``<prefix>_reconstruction.json`` artifact beside the trace;
``input_log`` is state-faithful, ``commands`` is protocol-faithful).

**The one-sided / omniscient wall (hard requirement).** This record contains the
opponent's full team and the dice. Nothing in the observation/training pipeline
reads it: a consumer that wants "what the agent knew" reads the *per-side protocol
chunks* that :func:`replay_battle` / :func:`reroll_turn` regenerate — exactly the
bytes the live agent saw — through the Rust core's parse chain
(``main.prober.core_walk``), never the omniscient state.

(The live-capture registry that joined a bridge battle's ``__RECON__`` frame to its
forensic trace — ``offer_record`` / ``register_trace_prefix`` — is deleted with the
Python bridge road, T27 P6 slice 6d-2.)

Offline replay / re-roll (the probe-agnostic primitive)
-------------------------------------------------------
:func:`replay_battle` re-runs the full command log through a fresh battle in the
Rust ``search_driver`` binary and returns the regenerated per-side
chunks + the final omniscient outcome. :func:`reroll_turn` reconstructs to the
start of turn ``t``, then resolves that one turn under N fresh PRNG seeds
(the battle's PRNG is swapped in place — all sim randomness routes through
it), with each side's start-of-turn action independently sourced from the
recorded choice, a uniform-random legal choice, or an explicit choice string.
What to fix vs re-sample, how many seeds, and what to measure are deliberately
NOT decided here — Phase-1 consumers compose those.

All three run the Rust ``search_driver`` binary (one binary serves these verbs AND the
clone-and-branch search), resolved by the one seam ``sim_bridge_bin.search_driver_spawn_argv``.
A missing/unbuildable binary raises. There is no engine choice: the Node drivers
(``replay_driver.js`` / ``search_driver.js``) and the ``impl=`` selector were DELETED on
2026-10-10 (owner: "delete node"; ``designs/deleted_flags.md``).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union


RECORD_VERSION = 1
RECON_SUFFIX = "_reconstruction.json"

_STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")

# The MOVE alias table ({alias_id: canonical_move_id}, e.g. wisp → willowisp), loaded lazily once
# per process from data/pokemon/gen3_move_aliases.json (derived from Showdown's `aliases.ts` by
# tools/pokemon_data_extractor — the SAME table the Rust port's dex resolves a packed-team move alias
# with). It used to be dumped LIVE from the Showdown dist with a `node -e` subprocess, and fell back
# SILENTLY to {} when node was missing or slow; deleted 2026-10-10 (owner: "delete node").
_ALIAS_CACHE: Optional[Dict[str, str]] = None


def _sim_aliases() -> Dict[str, str]:
    """The move-id alias table, cached per process. Pool packed strings can carry alias ids (a
    team export wrote "Wisp"); ``Teams.unpack`` resolves them, so a review decode must too or its
    ids won't join with anything else in the system (protocol/obs/summary all speak canonical
    ids). Read from ``data/`` through the ``agents.gen3_data`` facade — no subprocess, no
    fallback: a missing file RAISES (crash-don't-drop). ``packed_team_decode_integration_test.py``
    validates the resolved decode against the sim's own ``Teams.unpack`` over the whole pool."""
    global _ALIAS_CACHE
    if _ALIAS_CACHE is None:
        from agents.gen3_data import moves as gen3_moves

        _ALIAS_CACHE = dict(gen3_moves.move_aliases())
    return _ALIAS_CACHE


def decode_packed_team(packed: str) -> List[dict]:
    """THE one place a packed team string is decoded for review tooling.

    Returns one dict per mon: ``{species, item, ability, moves, nature,
    evs: {hp..spe}, ivs: {hp..spe}, gender, level}`` with the packed format's
    omission-defaults applied (empty EV field = all 0, empty IV field = all 31,
    level omitted = 100) and ids resolved through the move alias table (a pool
    string can say ``wisp``; everything downstream speaks ``willowisp``).
    Delegates to ``utils.team_packing``'s ``parse_packed_team`` — never a
    reimplementation; validated field-by-field
    against the sim's own ``Teams.unpack`` over the ENTIRE team pool by
    ``packed_team_decode_integration_test.py``. Replay/re-roll deliberately do
    NOT use this: they feed the packed string back to the sim verbatim.
    """
    from utils.showdown_id import to_id_str
    from utils.team_packing import Teambuilder

    aliases = _sim_aliases()

    def rid(raw: str) -> str:
        i = to_id_str(raw)
        return to_id_str(aliases[i]) if i in aliases else i

    out = []
    for mon in Teambuilder.parse_packed_team(packed):
        out.append({
            "species": rid(mon.species or mon.nickname),
            "item": rid(mon.item) if mon.item else "",
            "ability": rid(mon.ability) if mon.ability else "",
            "moves": [rid(m) for m in mon.moves],
            "nature": to_id_str(mon.nature) if mon.nature else "",
            "evs": dict(zip(_STAT_ORDER, (int(v) for v in mon.evs))),
            "ivs": dict(zip(_STAT_ORDER, (int(v) for v in mon.ivs))),
            "gender": mon.gender or "",
            "level": int(mon.level or 100),
        })
    return out


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconstructionRecord:
    """One battle's full-information reconstruction record (see module header)."""

    format_id: str
    prng_seed: str                 # the RESOLVED seed (e.g. "sodium,<hex>" or "1,2,3,4")
    input_log: tuple               # battle.inputLog: >start / >player / committed choices
    commands: tuple                # raw [side, choice] pairs the bridge processed, in order
    battle_tag: Optional[str] = None     # agent-side join key (filled at write time)
    trainee_username: Optional[str] = None  # which player was the recorded agent (label only)
    v: int = RECORD_VERSION

    # -- derived views ------------------------------------------------------
    def start_options(self) -> dict:
        """The parsed ``>start {...}`` options — includes the resolved ``seed``."""
        for line in self.input_log:
            if line.startswith(">start "):
                return json.loads(line[len(">start "):])
        raise ValueError("reconstruction record has no >start line")

    def players(self) -> Dict[str, dict]:
        """``{"p1": {...name, team...}, "p2": {...}}`` parsed from the ``>player`` lines."""
        out: Dict[str, dict] = {}
        for line in self.input_log:
            if line.startswith(">player "):
                side, _, payload = line[len(">player "):].partition(" ")
                out[side] = json.loads(payload)
        return out

    def username(self, side: str) -> str:
        return self.players()[side]["name"]

    def packed_team(self, side: str) -> str:
        return self.players()[side]["team"]

    def team_details(self, side: str) -> List[dict]:
        """``side``'s FULL team decoded for review (moves, EVs, IVs, nature,
        item, ability, level — see :func:`decode_packed_team`). Referee-view
        data: fine for analysis/reports, must never feed the obs pipeline."""
        return decode_packed_team(self.packed_team(side))

    def side_of(self, username: str) -> str:
        """Which sim side (``p1``/``p2``) the player named ``username`` held."""
        for side, info in self.players().items():
            if info.get("name") == username:
                return side
        raise KeyError(f"no player named {username!r} in record")

    # -- (de)serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "v": self.v,
            "battle_tag": self.battle_tag,
            "trainee_username": self.trainee_username,
            "format_id": self.format_id,
            "prng_seed": self.prng_seed,
            "input_log": list(self.input_log),
            "commands": [list(c) for c in self.commands],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReconstructionRecord":
        return cls(
            format_id=d["format_id"],
            prng_seed=d["prng_seed"],
            input_log=tuple(d["input_log"]),
            commands=tuple(tuple(c) for c in d["commands"]),
            battle_tag=d.get("battle_tag"),
            trainee_username=d.get("trainee_username"),
            v=d.get("v", RECORD_VERSION),
        )

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReconstructionRecord":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Offline replay / re-roll (drives the Rust search_driver binary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BattleReplay:
    """A full verbatim replay: the regenerated per-side protocol + final outcome.

    ``p1_chunks``/``p2_chunks`` are the one-sided streams — byte-identical to what
    the live bridge fed each player. ``outcome`` is omniscient (referee view):
    keep it out of anything the encoder touches.
    """

    p1_chunks: List[str]
    p2_chunks: List[str]
    outcome: dict   # {turn, ended, winner, p1: {alive, team_hp}, p2: {...}}


@dataclass(frozen=True)
class TurnReroll:
    """One re-rolled resolution of the target turn under a fresh seed."""

    seed: str
    choices_used: dict        # {"p1": [...], "p2": [...]} — incl. any follow-ups
    outcome: dict             # omniscient post-turn summary (same shape as BattleReplay)
    turn_log: List[str]       # omniscient |...| lines of the re-rolled turn (ground truth)
    p1_chunks: List[str]      # one-sided SUFFIX protocol for the re-rolled turn
    p2_chunks: List[str]


@dataclass(frozen=True)
class RerollResult:
    """Reconstruction point + N re-rolls of one turn (see :func:`reroll_turn`)."""

    turn: int
    pre_state: dict           # omniscient board snapshot at the decision point
    requests: dict            # {"p1": request-json, "p2": ...} — the choice surfaces
    recorded_choices: dict    # {"p1": str|None, "p2": str|None} — the original turn-t picks
    prefix_p1_chunks: List[str]   # one-sided protocol from battle start → decision point
    prefix_p2_chunks: List[str]
    rerolls: List[TurnReroll] = field(default_factory=list)


def _run_driver(request: dict, timeout: float) -> dict:
    """Spawn ONE offline replay/re-roll driver, feed it ``request``, return its JSON reply.

    The child is the Rust ``search_driver`` binary, which serves the replay / reroll /
    reroll_many verbs AND the clone-and-branch search ``search_session.py`` drives; it is
    resolved (built if needed) by ``sim_bridge_bin.search_driver_spawn_argv``.
    """
    from utils.bridge.sim_bridge_bin import search_driver_spawn_argv
    argv = search_driver_spawn_argv()
    proc = subprocess.run(
        argv,
        input=json.dumps(request).encode(),
        capture_output=True,
        timeout=timeout,
    )
    who = f"replay driver [{argv[-1]}]"
    stdout = proc.stdout.decode("utf-8", "replace")
    stderr = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        # ⚠️ A driver reports its diagnosis as a JSON ``{"error": …}`` on STDOUT and then exits
        # non-zero with an EMPTY stderr — so a stderr-only message reads as a silent crash and
        # sends the reader hunting for something that never happened. Measured: the rust
        # `search_driver` refusing turn 1 produced literally `failed (rc=1): ` and nothing else.
        # Surface the stdout reason FIRST, and say so explicitly when both streams are empty.
        try:
            reason = str(json.loads(stdout).get("error") or "")
        except (ValueError, AttributeError):
            reason = stdout.strip()[-2000:]
        raise RuntimeError(
            f"{who} failed (rc={proc.returncode})"
            + (f": {reason}" if reason else "")
            + (f" [stderr] {stderr[-2000:]}" if stderr.strip()
               else ("" if reason else " — EMPTY stdout AND stderr"))
        )
    out = json.loads(stdout)
    if "error" in out:
        raise RuntimeError(f"{who}: {out['error']}")
    return out


def replay_battle(record: ReconstructionRecord, *, timeout: float = 120.0) -> BattleReplay:
    """Re-run the battle verbatim (original seed, original commands) and return the
    regenerated per-side protocol + the final omniscient outcome.

    This is the reproducibility primitive: the per-side chunk sequences are
    byte-identical to what the live battle fed each player — modulo ``|t:|``
    wall-clock timestamp lines, which the sim stamps at emission time and which
    are state- and obs-invisible. Feed them to the Rust core's parse chain
    (``main.prober.core_walk.read_streams``) to rebuild the agent's exact rows.
    """
    out = _run_driver({"mode": "replay", "record": record.to_dict()}, timeout)
    return BattleReplay(p1_chunks=out["p1_chunks"], p2_chunks=out["p2_chunks"],
                        outcome=out["outcome"])


def reroll_turn(
    record: ReconstructionRecord,
    turn: int,
    *,
    seeds: Sequence[str] = (),
    p1_action: str = "recorded",
    p2_action: str = "recorded",
    followup: str = "random",
    timeout: float = 300.0,
) -> RerollResult:
    """Reconstruct to the start of turn ``turn`` and re-roll it under each seed.

    ``p1_action``/``p2_action`` choose that side's START-OF-TURN action source:
    ``"recorded"`` (the original pick), ``"random"`` (uniform over the request's
    legal moves+switches, drawn from an aux PRNG derived from the re-roll seed),
    or any explicit sim choice string (e.g. ``"move 2"``, ``"switch 3"``). A
    Phase-1 policy source composes as explicit strings: materialize the obs at
    the decision point, run the policy, pass its choice here.

    Mid-turn follow-up requests in a re-rolled timeline (a forced switch after a
    faint that the original timeline may not even contain) are answered by the
    ``followup`` policy: ``"random"`` (uniform legal, aux-PRNG) or ``"default"``
    (the sim's deterministic default order). They are *new* decisions with no
    recorded answer — which distribution to use is a consumer choice.

    ``seeds`` entries are sim PRNG seeds: ``"a,b,c,d"`` (gen-5 int quadruple) or
    ``"sodium,<hex>"`` — plus the special value ``"original"``, which keeps the
    battle's own mid-game PRNG state (no swap). With both actions ``recorded``,
    ``"original"`` reproduces the REALIZED turn exactly (recorded follow-ups
    included), scored through the same outcome pipeline as the re-rolls so its
    margin is directly comparable; with a different action it answers "same
    dice stream, different action". ``seeds=()`` just reconstructs (no
    re-rolls) — useful for reading the decision point's pre-state/requests/
    prefix protocol.

    Everything omniscient in the result (``pre_state``, ``outcome``,
    ``turn_log``) is for ground-truth analysis only; obs materialization must use
    only the per-side chunks.
    """
    out = _run_driver(
        {
            "mode": "reroll",
            "record": record.to_dict(),
            "turn": turn,
            "seeds": list(seeds),
            "p1_action": p1_action,
            "p2_action": p2_action,
            "followup": followup,
        },
        timeout,
    )
    rerolls = [
        TurnReroll(
            seed=r["seed"], choices_used=r["choices_used"], outcome=r["outcome"],
            turn_log=r["turn_log"], p1_chunks=r["p1_chunks"], p2_chunks=r["p2_chunks"],
        )
        for r in out["rerolls"]
    ]
    return RerollResult(
        turn=out["turn"],
        pre_state=out["pre_state"],
        requests=out["requests"],
        recorded_choices=out["recorded_choices"],
        prefix_p1_chunks=out["prefix_p1_chunks"],
        prefix_p2_chunks=out["prefix_p2_chunks"],
        rerolls=rerolls,
    )


@dataclass(frozen=True)
class ArmReroll:
    """One arm of a batched re-roll — its own action source(s) + seed. Same shape as
    :class:`TurnReroll`, plus the arm's ``label`` (the caller's tag, e.g. an action index)."""

    label: object
    seed: str
    choices_used: dict
    outcome: dict
    turn_log: List[str]
    p1_chunks: List[str]
    p2_chunks: List[str]


@dataclass(frozen=True)
class RerollManyResult:
    """Reconstruction point + N independent ARM re-rolls of one turn, resolved in ONE process
    (see :func:`reroll_many`)."""

    turn: int
    pre_state: dict
    requests: dict
    recorded_choices: dict
    prefix_p1_chunks: List[str]
    prefix_p2_chunks: List[str]
    arms: List[ArmReroll] = field(default_factory=list)


def reroll_many(
    record: ReconstructionRecord,
    turn: int,
    arms: Sequence[dict],
    *,
    followup: str = "random",
    timeout: float = 300.0,
) -> RerollManyResult:
    """Resolve N independent ARMS of turn ``turn`` in ONE driver process — one ``buildToTurn`` + module
    load instead of once per arm. Each ``arm`` is a dict ``{p1_action, p2_action, seed, label}`` with
    EXACTLY the per-side action-source semantics of :func:`reroll_turn` (``"recorded"`` / ``"random"`` /
    an explicit sim choice string; the special seed ``"original"`` keeps the battle's own mid-game PRNG).

    This is purely a PERFORMANCE batching of :func:`reroll_turn`: a candidate-action sweep (the prober's
    one-ply lookahead) pays the driver spawn + battle rebuild ONCE rather than per candidate. Each arm runs in its OWN fresh session, so an
    arm's suffix chunks are BYTE-IDENTICAL to the same single :func:`reroll_turn` — guarded by
    ``reroll_many_parity_fuzz_test.py``. The shared prefix chunks + decision-point views are returned
    once. ``arms=()`` just reconstructs (prefix/requests only)."""
    out = _run_driver(
        {
            "mode": "reroll_many",
            "record": record.to_dict(),
            "turn": turn,
            "arms": [dict(a) for a in arms],
            "followup": followup,
        },
        timeout,
    )
    arm_results = [
        ArmReroll(
            label=a.get("label"), seed=a["seed"], choices_used=a["choices_used"],
            outcome=a["outcome"], turn_log=a["turn_log"],
            p1_chunks=a["p1_chunks"], p2_chunks=a["p2_chunks"],
        )
        for a in out["arms"]
    ]
    return RerollManyResult(
        turn=out["turn"], pre_state=out["pre_state"], requests=out["requests"],
        recorded_choices=out["recorded_choices"],
        prefix_p1_chunks=out["prefix_p1_chunks"], prefix_p2_chunks=out["prefix_p2_chunks"],
        arms=arm_results,
    )
