"""Battle reconstruction records — capture registry, artifact writer, and the
offline replay / re-roll primitives.

A bridge battle's **reconstruction record** is the full-information (referee-view)
data needed to rebuild it bit-for-bit offline: the resolved PRNG seed, both packed
teams, the sim's own ``inputLog``, and the raw command sequence the bridge child
processed (``local_sim_bridge.js`` emits it as a ``__RECON__`` frame just before
``__END__``; see that file's header for the exact fields and why *both* logs are
kept — ``input_log`` is state-faithful, ``commands`` is protocol-faithful).

**The one-sided / omniscient wall (hard requirement).** This record contains the
opponent's full team and the dice. It exists ONLY at the bridge/sim layer and in
the separate ``<prefix>_reconstruction.json`` artifact written here. Nothing in
the observation/training pipeline reads it: the obs encoder consumes poke-env's
one-sided battle view exclusively, and the offline materializer
(``agents.training.obs_materializer``) is fed only the *per-side protocol chunks*
that :func:`replay_battle` / :func:`reroll_turn` regenerate — exactly the bytes
the live agent saw — never the omniscient state. Keep it that way: any new
consumer that wants "what the agent knew" must go through the per-side chunks.

Capture → artifact join
-----------------------
The ``__RECON__`` frame arrives on the bridge's stdout *after* the ``|win|``
chunks (the streams close first), but the eval forensic trace for the same battle
is written *during* the ``|win|`` chunk's handling — so neither side can simply
hand the other the data. The two sides meet in the bounded registry below, keyed
by battle tag; whichever arrives second completes the pair and writes the
artifact:

- the demux/dispatch loop calls :func:`offer_record` when ``__RECON__`` arrives;
- the forensic writer calls :func:`register_trace_prefix` right after it persists
  a trace (``<prefix>_summary.json`` etc.).

Battles whose trace is never persisted (quota-dropped, training episodes) leave
an orphan entry that FIFO-eviction reclaims — capture is always-on and cheap, so
no flag is threaded through the bridge.

Offline replay / re-roll (the probe-agnostic primitive)
-------------------------------------------------------
:func:`replay_battle` re-runs the full command log through a fresh in-process
``BattleStream`` (``replay_driver.js``) and returns the regenerated per-side
chunks + the final omniscient outcome. :func:`reroll_turn` reconstructs to the
start of turn ``t``, then resolves that one turn under N fresh PRNG seeds
(``stream.battle.prng`` is swapped in place — all sim randomness routes through
it), with each side's start-of-turn action independently sourced from the
recorded choice, a uniform-random legal choice, or an explicit choice string.
What to fix vs re-sample, how many seeds, and what to measure are deliberately
NOT decided here — Phase-1 consumers compose those.
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

_DRIVER_JS = str(Path(__file__).parent / "replay_driver.js")

RECORD_VERSION = 1
RECON_SUFFIX = "_reconstruction.json"

_STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")

# Sim alias table ({alias_id: canonical display name}, e.g. wisp → Will-O-Wisp),
# dumped lazily once per process. None = not loaded yet; {} = bridge unavailable.
_ALIAS_CACHE: Optional[Dict[str, str]] = None
_ALIAS_DUMP_JS = (
    "const a=require(process.argv[1]+'/dist/data/aliases');"
    "process.stdout.write(JSON.stringify(a.Aliases));"
)
_PS_PATH = str(Path(__file__).parents[3] / "deps" / "pokemon-showdown")


def _sim_aliases() -> Dict[str, str]:
    """The sim's own id-alias table, cached per process. Pool packed strings can
    carry alias ids (a team export wrote "Wisp"); ``Teams.unpack`` resolves them,
    so a review decode must too or its ids won't join with anything else in the
    system (protocol/obs/summary all speak canonical ids). Falls back to ``{}``
    when node / the sim dist is unavailable (pure-unit environments) — the
    integration sweep validates the resolved decode against the sim itself."""
    global _ALIAS_CACHE
    if _ALIAS_CACHE is None:
        try:
            proc = subprocess.run(["node", "-e", _ALIAS_DUMP_JS, _PS_PATH],
                                  capture_output=True, timeout=30)
            _ALIAS_CACHE = json.loads(proc.stdout.decode()) if proc.returncode == 0 else {}
        except (OSError, ValueError, subprocess.SubprocessError):
            _ALIAS_CACHE = {}
    return _ALIAS_CACHE


def decode_packed_team(packed: str) -> List[dict]:
    """THE one place a packed team string is decoded for review tooling.

    Returns one dict per mon: ``{species, item, ability, moves, nature,
    evs: {hp..spe}, ivs: {hp..spe}, gender, level}`` with the packed format's
    omission-defaults applied (empty EV field = all 0, empty IV field = all 31,
    level omitted = 100) and ids resolved through the sim's alias table (a pool
    string can say ``wisp``; everything downstream speaks ``willowisp``).
    Delegates to poke-env's ``parse_packed_team`` — the same parser the live
    battle path uses — never a reimplementation; validated field-by-field
    against the sim's own ``Teams.unpack`` over the ENTIRE team pool by
    ``packed_team_decode_integration_test.py``. Replay/re-roll deliberately do
    NOT use this: they feed the packed string back to the sim verbatim.
    """
    from poke_env.data.normalize import to_id_str
    from poke_env.teambuilder.teambuilder import Teambuilder

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
# Capture registry — the __RECON__ ↔ forensic-trace join (see module header)
# ---------------------------------------------------------------------------

_REGISTRY_CAP = 64  # plenty: traces are written within the same battle's teardown

_lock = threading.Lock()
_records: "OrderedDict[str, dict]" = OrderedDict()      # tag -> raw record dict
_pending: "OrderedDict[str, tuple]" = OrderedDict()     # tag -> (out_prefix, extra)


def _evict(d: OrderedDict) -> None:
    while len(d) > _REGISTRY_CAP:
        d.popitem(last=False)


def offer_record(battle_tag: str, raw: dict) -> Optional[str]:
    """Bridge demux entry point: a battle's ``__RECON__`` record arrived.

    If a forensic trace prefix is already registered for the tag, writes the
    artifact now and returns its path; otherwise stashes the record (bounded,
    FIFO-evicted) for a later :func:`register_trace_prefix` and returns None.
    """
    with _lock:
        pending = _pending.pop(battle_tag, None)
        if pending is None:
            _records[battle_tag] = raw
            _records.move_to_end(battle_tag)
            _evict(_records)
            return None
    out_prefix, extra = pending
    return _write_artifact(out_prefix, battle_tag, raw, extra)


def register_trace_prefix(battle_tag: str, out_prefix: str,
                          extra: Optional[dict] = None) -> Optional[str]:
    """Forensic-writer entry point: a trace for ``battle_tag`` was persisted at
    ``<out_prefix>_*``. If the battle's record already arrived, writes the
    artifact now and returns its path; otherwise remembers the prefix (bounded)
    for the imminent :func:`offer_record`. ``extra`` keys (e.g.
    ``trainee_username``) are merged into the artifact."""
    with _lock:
        raw = _records.pop(battle_tag, None)
        if raw is None:
            _pending[battle_tag] = (out_prefix, extra)
            _pending.move_to_end(battle_tag)
            _evict(_pending)
            return None
    return _write_artifact(out_prefix, battle_tag, raw, extra)


def pop_record(battle_tag: str) -> Optional[ReconstructionRecord]:
    """Pop a stashed record (tools/tests); None if absent."""
    with _lock:
        raw = _records.pop(battle_tag, None)
    if raw is None:
        return None
    return ReconstructionRecord.from_dict({**raw, "battle_tag": battle_tag})


def _write_artifact(out_prefix: str, battle_tag: str, raw: dict,
                    extra: Optional[dict]) -> str:
    rec = {**raw, "battle_tag": battle_tag, **(extra or {})}
    path = f"{out_prefix}{RECON_SUFFIX}"
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    return path


# ---------------------------------------------------------------------------
# Offline replay / re-roll (drives replay_driver.js)
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
    proc = subprocess.run(
        ["node", _DRIVER_JS],
        input=json.dumps(request).encode(),
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"replay_driver.js failed (rc={proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace')[-2000:]}"
        )
    out = json.loads(proc.stdout.decode())
    if "error" in out:
        raise RuntimeError(f"replay_driver.js: {out['error']}")
    return out


def replay_battle(record: ReconstructionRecord, *, timeout: float = 120.0) -> BattleReplay:
    """Re-run the battle verbatim (original seed, original commands) and return the
    regenerated per-side protocol + the final omniscient outcome.

    This is the reproducibility primitive: the per-side chunk sequences are
    byte-identical to what the live battle fed each player — modulo ``|t:|``
    wall-clock timestamp lines, which the sim stamps at emission time and which
    sit in poke-env's ``MESSAGES_TO_IGNORE`` (state- and obs-invisible). Feed
    them to ``agents.training.obs_materializer`` to rebuild the agent's exact
    observations.
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
