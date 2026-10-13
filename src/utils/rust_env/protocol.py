"""The Rust env's PROTOCOL — opcodes, status codes, pool counters and the typed error classes.

Half of the ONE contract table (``columns.py`` holds the other half, the columns) that
``python -m utils.rust_env.columns --write`` renders into ``src/rust_env/src/core/columns.rs``
(M5 Lane 0, ``designs/endstate/program_rust_core.md`` §2 M5). Both front ends (Lane A's FFI and
Lane B's process) forward an OPCODE and a column set to ONE ``core::dispatch`` and turn its STATUS
into one of the exception classes below — neither front end names an op's semantics.

A new op, status or counter is a row here plus a regeneration; ``columns_test.py`` (routine) fails
the day the committed Rust is stale.

This module is dependency-free (no numpy, no poke-env, no ``agents``): every consumer may import it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union


@dataclass(frozen=True)
class Op:
    name: str
    code: int  # one byte on the process front end's pipe; an argument on the FFI
    owner: str  # the lane that implements it
    doc: str


#: Every opcode ``core::dispatch`` accepts. A code is one ASCII byte so a pipe trace reads.
OPS: Tuple[Op, ...] = (
    Op("RESET", ord("R"), "0",
       "start every env's episode from the staged `ep_team` / `ep_seed`; write every output column"),
    Op("STEP", ord("S"), "0",
       "feed `action` for every (env, side) with `need` = 1, run to the next caller decision; an env whose "
       "episode ended (or was quarantined) writes `done` and auto-resets from the staged inputs"),
)


@dataclass(frozen=True)
class Status:
    name: str
    code: int
    exc: Optional[str]  # the Python class raised for it (None = success)
    doc: str


#: What ``core::dispatch`` returns. 0 is success; every other code is a typed failure of the WHOLE
#: batch (a quarantined battle is NOT a batch failure — see ``refused`` in the columns).
STATUSES: Tuple[Status, ...] = (
    Status("OK", 0, None, "the op completed; every output column is written"),
    Status("FAULT", 1, "CoreFault",
           "the core broke one of its own invariants (a `CoreError::Fault`): a BUG; the env's input log is banked"),
    Status("CALLER", 2, "CallerError",
           "the caller sent input no battle can take (an illegal action, a team index out of range, a seed "
           "word >= 65536): `CoreError::Malformed`"),
    Status("PANIC", 3, "CorePanic", "a panic inside the core, caught at the env; the env's input log is banked"),
    Status("LIFECYCLE", 4, "LifecycleViolation",
           "the op broke the DECLARED LIFECYCLE (dispatch before freeze, a column set that differs from the "
           "frozen binding, an unknown opcode); counted in an `*_after_freeze` counter where it is one"),
    Status("BUDGET", 5, "RefusalBudgetExceeded",
           "a quarantine would exceed the refusal budget declared at startup: a refusal STORM is systemic, "
           "never absorbed one battle at a time"),
)


@dataclass(frozen=True)
class Counter:
    name: str
    doc: str
    after_freeze: bool = False  # a lifecycle counter: must stay 0 for the process's life


#: The pool-level ``counters`` column (u64 each), in index order. Written by every dispatch.
COUNTERS: Tuple[Counter, ...] = (
    Counter("DISPATCHES", "ops dispatched (every status)"),
    Counter("DECISIONS", "rows written (one per (env, side) with need = 1 after an op)"),
    Counter("EPISODES_STARTED", "episodes started (RESET + every auto-reset)"),
    Counter("EPISODES_ENDED", "episodes that ended by the battle's own end (not quarantine)"),
    Counter("REFUSALS", "battles QUARANTINED (a `CoreError::Refusal`: banked, done = 1, refused = 1)"),
    Counter("CORE_NS_LAST", "wall ns of the last dispatch INSIDE the core (the caller's wall minus it = transport)"),
    Counter("CORE_NS_TOTAL", "wall ns of every dispatch inside the core"),
    Counter("THREADS_SPAWNED_AFTER_FREEZE", "worker threads created after freeze", after_freeze=True),
    Counter("ENVS_ADDED_AFTER_FREEZE", "envs added to the pool after freeze", after_freeze=True),
    Counter("COLUMN_REBINDS_AFTER_FREEZE",
            "dispatches that named a column set other than the frozen binding (each one refused)", after_freeze=True),
    Counter("BANK_GROWTH_AFTER_FREEZE",
            "refusal-bank reallocations after freeze (the bank's capacity = the refusal budget, reserved at startup)",
            after_freeze=True),
)


#: The STARTUP DECLARATION's keys (``core::spec::Spec::from_json``): EVERY one is required (null
#: where the doc allows), an unknown one is refused. Rendered into ``columns.rs`` as ``SPEC_KEYS``,
#: which ``spec.rs`` reads, so the two languages cannot disagree about the spec's shape.
SPEC_KEYS: Tuple[str, ...] = (
    "n", "threads", "format_id", "names", "teams", "turn_limit",
    "terminal", "refusal_budget", "bank_dir", "labels", "opponents", "oracle_reveal",
)

#: The ORACLE REVEAL's X32 LEVEL names (``--oracle-reveal``; ``encoder::oracle::OracleSpec::LEVELS``): ``off`` is the
#: production mode (the row is byte-identical to the build that had no reveal); ``species`` / ``full`` are the two
#: DIAGNOSTIC levels X32 built (``designs/endstate/design_x5_belief_tokens.md`` §7.6). Since the information-lever
#: spec (``designs/endstate/design_information_levers.md`` §6) the flag takes ANY spec the training observation honours
#: (:func:`oracle_reveal_label`); these three stay the LABELS of their specs, so a run that recorded one reads exactly
#: as before. A test pins the list to Rust's.
ORACLE_REVEAL_LEVELS: Tuple[str, ...] = ("off", "species", "full")


def oracle_reveal_label(spec: str) -> str:
    """The LABEL of an ``--oracle-reveal`` spec text (a preset such as ``ki_moves``, a level, or a clause list),
    validated against the TRAINING OBSERVATION's support (``agents.info_levers.consumers.OBS``): the level name for
    ``off`` / ``species`` / ``full`` (whichever text it was given as), else the canonical text — exactly what the Rust
    core parses (``OracleSpec::parse``). A lever the observation cannot honour (``hp``, ``counters``, a ``~placebo`` /
    ``~map`` source, an unseen set-class part without ``species:unseen``) is refused BY NAME (``ValueError``)."""
    from agents.info_levers.consumers import OBS
    from agents.info_levers.spec import InfoRevealError

    if not isinstance(spec, str):
        raise ValueError(f"oracle_reveal {spec!r} is not a spec text")
    try:
        return OBS.validate(spec).label
    except InfoRevealError as e:
        raise ValueError(f"oracle_reveal {spec!r}: {e}") from None


def spec_json(*, n: int, threads: int, teams: "list[str]", names: "tuple[str, str]",
              turn_limit: Optional[int],
              refusal_budget: int, bank_dir: Optional[str], format_id: str = "gen3ou",
              labels: "tuple[str, ...]" = (), terminal: "Optional[dict]" = None,
              opponents: "Optional[list[dict]]" = None,
              oracle_reveal: "Union[str, Tuple[str, str]]" = "off") -> str:
    """The startup declaration as the core parses it (every key explicit in the JSON — nothing
    defaulted except ``format_id``, the one format the core runs, and ``labels``: the Lane-C label
    FAMILIES the core writes, ``label_inventory`` names; none by default), and ``terminal`` (M5 Lane
    D): the terminal-reward declaration, defaulting to the PRODUCTION terminal
    (``episode.PRODUCTION_TERMINAL``, pinned against ``designs/production_config.json`` by
    ``episode_test.py``). ``turn_limit`` is the STALL FORFEIT threshold — ``StallConfig().threshold``
    in production (``episode.stall_threshold()``); ``None`` = no forfeit (harnesses only).
    ``opponents`` (M5 Lane E) is the OPPONENT ROUTE TABLE the ``ep_opp`` column indexes
    (``{"kind": "external"}`` / ``{"kind": "policy", "slot": k}`` / ``{"kind": "bot", "bot": name}``);
    default one EXTERNAL route, so a zeroed ``ep_opp`` is Lane 0's shape (the caller answers p2).
    ``oracle_reveal`` is the ORACLE REVEAL spec (any spec the training observation honours, written as its LABEL:
    :func:`oracle_reveal_label`): ``off`` is the production observation; a diagnostic spec tells every side's opponent
    block graded facts of the other side's true team — the run's recorded mode, so the training pool and its eval core
    must be given the SAME one. A ``(p1, p2)`` PAIR is a PER-SIDE spec (each side's chain told the other side's team at
    its own spec): only the head-to-head engine declares one (``main.h2h``, X5 A/B §7.7(a)). A pair whose two labels
    agree is written as the string, so a symmetric spec's text is exactly what it was before the per-side form existed
    (and an ``off`` / ``species`` / ``full`` spec's text exactly what it was before the information-lever spec)."""
    import json

    if isinstance(oracle_reveal, str):
        levels = (oracle_reveal, oracle_reveal)
    else:
        levels = tuple(oracle_reveal)
        if len(levels) != 2:
            raise ValueError(f"oracle_reveal {oracle_reveal!r} is not a spec text or a (p1, p2) pair of them")
    reveal_labels = tuple(oracle_reveal_label(lv) for lv in levels)
    reveal: "Union[str, list[str]]" = reveal_labels[0] if reveal_labels[0] == reveal_labels[1] else list(reveal_labels)

    from utils.rust_env import episode as EP

    term = EP.terminal_json(EP.PRODUCTION_TERMINAL if terminal is None else terminal)

    spec = {
        "n": n, "threads": threads, "format_id": format_id, "names": list(names), "teams": list(teams),
        "turn_limit": turn_limit,
        "terminal": term, "refusal_budget": refusal_budget, "bank_dir": bank_dir, "labels": list(labels),
        "opponents": [dict(r) for r in (opponents if opponents is not None else ({"kind": "external"},))],
        "oracle_reveal": reveal,
    }
    assert tuple(spec) == SPEC_KEYS, "spec_json and SPEC_KEYS drifted"
    return json.dumps(spec)


def error_from_json(text: str) -> RustEnvError:
    """The typed exception for ``DispatchError::json`` (``{"status", "env", "kind", "class",
    "message", "script"}``)."""
    import json

    d = json.loads(text)
    return error_for(int(d["status"]), d["message"], kind=d.get("kind"), py_class=d.get("class"),
                     env=d.get("env"), script=d.get("script"))


def counter_index() -> Dict[str, int]:
    return {c.name: i for i, c in enumerate(COUNTERS)}


# ------------------------------------------------------------------ the typed error classes


class RustEnvError(RuntimeError):
    """Base of every error the Rust env raises through a front end. ``kind`` / ``py_class`` /
    ``message`` are ``CoreError::json``'s fields when the core supplied one."""

    status = -1

    def __init__(self, message: str, *, kind: Optional[str] = None, py_class: Optional[str] = None,
                 env: Optional[int] = None, script: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.py_class = py_class
        self.env = env
        self.script = script  # the banked input log (a `sim_bridge` / `core_events` script), if any


class CoreFault(RustEnvError):
    status = 1


class CallerError(RustEnvError):
    status = 2


class CorePanic(RustEnvError):
    status = 3


class LifecycleViolation(RustEnvError):
    status = 4


class RefusalBudgetExceeded(RustEnvError):
    status = 5


_EXC = {cls.__name__: cls for cls in (CoreFault, CallerError, CorePanic, LifecycleViolation,
                                      RefusalBudgetExceeded)}


def error_for(status: int, message: str, **kw) -> RustEnvError:
    """The typed exception for a non-OK status (an unknown status is itself a fault)."""
    for s in STATUSES:
        if s.code == status and s.exc is not None:
            return _EXC[s.exc](message, **kw)
    return CoreFault(f"unknown status {status}: {message}", **kw)


def _check_table() -> None:
    names = [s.exc for s in STATUSES if s.exc is not None]
    missing = set(names) - set(_EXC)
    if missing:
        raise AssertionError(f"STATUSES names classes this module does not define: {sorted(missing)}")
    for s in STATUSES:
        if s.exc is not None and _EXC[s.exc].status != s.code:
            raise AssertionError(f"{s.exc}.status = {_EXC[s.exc].status}, the table says {s.code}")
    for seq, what in ((OPS, "op"), (STATUSES, "status")):
        codes = [x.code for x in seq]
        if len(set(codes)) != len(codes) or not all(0 <= c < 256 for c in codes):
            raise AssertionError(f"{what} codes must be distinct bytes: {codes}")


_check_table()
