"""THE WORLD LEGALITY CHECK (F2 of ``designs/endstate/design_world_sampler.md``): every set a world builder draws, and
every world team it returns, passes Showdown's own ``TeamValidator('gen3ou')`` from the PINNED ``deps/pokemon-showdown``.

Why (MEASURED 2026-10-19, ``measurements/world_sampler_screen_2026-10-19/validate_worlds.py``): the honest builder drew
each mon's unrevealed moves slot by slot from Smogon's per-species marginals, so 581 of 11,160 stored F worlds (5.2 %),
in 445 of 1,395 states (32 %), held a set no player can hold: egg moves no single father passes on together (Tyranitar
Dragon Dance + Pursuit), event moves from different events (Blissey Wish + Aromatherapy). The true teams: 0 of 1,395.
``agents.gen3_data.team_legality`` checks the format's clauses and bans only; learnset COMBINATIONS are the validator's.

THE METHOD: REJECTION through the validator itself, never a hand-written table (re-implementing breeding and event
rules is a GIGO class: one wrong rule silently removes the truth). A drawn set is redrawn until the validator accepts
it, so the draw is the prior CONDITIONED on legality: P(set | species, the facts already fixed, legal). Every verdict is
memoised per set text, and the validator is ONE persistent Node child (``utils/bridge/validate_stream.js``), so a world
costs a few milliseconds of validation (Node's start-up and the dex load paid once per process).

THE TRUE-TEAM GUARD (the design's own requirement, §3.3 point 4): the TRUTH must pass every rule the builder applies.
A true set the check rejects raises :class:`LegalityRuleError` — a DEFECT of the rule (or of the truth's data), never an
exclusion, so a mis-ported rule can never silently remove the truth from the support.

Infrastructure failures (no ``node``, an unbuilt submodule, a child that dies, a team that does not import) raise
:class:`ValidatorUnavailable`; they are never a verdict.
"""
from __future__ import annotations

import atexit
import json
import os
import select
import shutil
import subprocess
import tempfile
import threading
from typing import IO, Dict, List, Optional, Sequence, Tuple

from utils.paths import src_path

FORMAT = "gen3ou"
#: one request's bound (contention-scaled): a validator that answers nothing in this long is a dead child, not a verdict
REQUEST_TIMEOUT_S = 120.0


class LegalityRuleError(RuntimeError):
    """The TRUE team fails the legality rule the builder applies: a defect, never a ``WorldError`` exclusion."""


class IllegalWorldError(RuntimeError):
    """A RETURNED world fails the legality rule (``world_guard.check_legal``): a builder path that skipped the
    rejection, a defect — never a ``WorldError`` exclusion."""


class ValidatorUnavailable(RuntimeError):
    """The validator could not answer (no node, an unbuilt submodule, a dead child, an unimportable team)."""


def strip_pad(entry: str, pad: Optional[str]) -> str:
    """A packed entry without the engine's gender-draw PAD marker (ours, not Showdown's: ``N*`` / ``M*`` → ``N`` / ``M``)."""
    if not pad:
        return entry
    f = entry.split("|")
    if len(f) > 7 and f[7].endswith(pad):
        f[7] = f[7][: -len(pad)]
    return "|".join(f)


class Validator:
    """Showdown's ``TeamValidator(FORMAT)`` behind one persistent Node child, memoised per team text.

    ``problems(teams)`` → per team, the validator's error list (empty = legal). A one-set team validates that set alone
    (every rule a set can break on its own: learnset, egg / event combinations, EVs, IVs, Hidden Power, bans)."""

    def __init__(self, fmt: str = FORMAT, script: Optional[str] = None):
        self.fmt = fmt
        self.script = script or str(src_path("utils", "bridge", "validate_stream.js"))
        self._memo: Dict[str, Tuple[str, ...]] = {}
        self._proc: Optional[subprocess.Popen] = None
        self._pid_owner: Optional[int] = None
        self._err: Optional[IO[str]] = None
        self._lock = threading.Lock()
        self.requests = 0                  # round trips to the child (a cost meter)

    # ---- the child ---------------------------------------------------------------------------------------------
    def _child(self) -> subprocess.Popen:
        if self._proc is not None and self._pid_owner == os.getpid() and self._proc.poll() is None:
            return self._proc
        node = shutil.which("node")
        if node is None:
            raise ValidatorUnavailable("no `node` on PATH: the world builder's legality check needs Showdown's "
                                       "validator (Node's one runtime job, root CLAUDE.md)")
        if not os.path.exists(self.script):
            raise ValidatorUnavailable(f"the validator bridge is missing: {self.script}")
        self._err = tempfile.TemporaryFile(mode="w+")      # a file, never a pipe nobody drains (it could fill)
        self._proc = subprocess.Popen([node, self.script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self._err, text=True, bufsize=1)
        self._pid_owner = os.getpid()
        atexit.register(self.close)
        return self._proc

    def close(self) -> None:
        """Stop OUR child (by its own handle; only in the process that started it)."""
        p, self._proc = self._proc, None
        if p is not None and self._pid_owner == os.getpid() and p.poll() is None:
            try:
                p.stdin.close()                                  # type: ignore[union-attr]
                p.wait(timeout=5)
            except Exception:
                p.kill()
                p.wait(timeout=5)

    def _ask(self, teams: List[str]) -> List[Tuple[str, ...]]:
        from utils.contention import scale_timeout

        p = self._child()
        try:
            p.stdin.write(json.dumps({"format": self.fmt, "teams": teams}) + "\n")   # type: ignore[union-attr]
            p.stdin.flush()                                                           # type: ignore[union-attr]
        except BrokenPipeError as e:
            raise ValidatorUnavailable(f"the validator child died: {self._stderr(p)}") from e
        ready, _, _ = select.select([p.stdout], [], [], scale_timeout(REQUEST_TIMEOUT_S))
        if not ready:
            self.close_hard()
            raise ValidatorUnavailable(f"the validator answered nothing in {REQUEST_TIMEOUT_S:.0f} s (scaled)")
        line = p.stdout.readline()                                                    # type: ignore[union-attr]
        if not line:
            raise ValidatorUnavailable(f"the validator child exited: {self._stderr(p)}")
        got = json.loads(line)
        self.requests += 1
        if isinstance(got, dict):
            raise ValidatorUnavailable(f"the validator refused the request: {got.get('fatal', got)}")
        if len(got) != len(teams):
            raise ValidatorUnavailable(f"the validator answered {len(got)} verdicts for {len(teams)} teams")
        out = []
        for t, v in zip(teams, got):
            if v.get("valid") is None:
                raise ValidatorUnavailable(f"the validator raised on {t!r}: {v.get('raised')}")
            out.append(tuple(e for e in v.get("errors", []) if "Bug Reports" not in e) if not v["valid"] else ())
        return out

    def close_hard(self) -> None:
        p, self._proc = self._proc, None
        if p is not None and self._pid_owner == os.getpid():
            p.kill()
            p.wait(timeout=5)

    def _stderr(self, p: subprocess.Popen) -> str:
        try:
            p.wait(timeout=2)
            if self._err is None:
                return "(no stderr)"
            self._err.seek(0)
            return (self._err.read() or "")[-2000:]
        except Exception:
            return "(no stderr)"

    # ---- the verdicts ------------------------------------------------------------------------------------------
    def problems(self, teams: Sequence[str]) -> List[Tuple[str, ...]]:
        """Per team text, the validator's errors (``()`` = legal); one round trip for every text not yet memoised."""
        with self._lock:
            todo = sorted({t for t in teams if t not in self._memo})
            if todo:
                for t, errs in zip(todo, self._ask(todo)):
                    self._memo[t] = errs
            return [self._memo[t] for t in teams]


_SHARED: Dict[str, Validator] = {}


def shared(fmt: str = FORMAT) -> Validator:
    """The process's one validator for ``fmt`` (its memo shared by every builder in the process)."""
    if fmt not in _SHARED:
        _SHARED[fmt] = Validator(fmt)
    return _SHARED[fmt]


class Legality:
    """The builder's legality rule: a set is legal iff Showdown's validator accepts it as a one-set team, and a world
    iff it accepts the whole team. ``pad``: the engine's gender-draw PAD marker, stripped before validation.

    A test or a measurement may inject another ``validator`` (anything with ``problems(teams)``); the builder's
    default is the pinned Showdown one."""

    def __init__(self, validator=None, pad: Optional[str] = None):
        self.v = validator if validator is not None else shared()
        self.pad = pad
        self.checked = 0             # sets / teams asked about (memo hits included)

    def set_problems(self, entries: Sequence[str]) -> List[Tuple[str, ...]]:
        self.checked += len(entries)
        return self.v.problems([strip_pad(e, self.pad) for e in entries])

    def set_ok(self, entry: str) -> bool:
        return not self.set_problems([entry])[0]

    def team_problems(self, teams: Sequence[str]) -> List[Tuple[str, ...]]:
        self.checked += len(teams)
        return self.v.problems(["]".join(strip_pad(e, self.pad) for e in t.split("]")) for t in teams])

    def require_true_team(self, team: str, key: str = "") -> None:
        """THE TRUE-TEAM GUARD: every true set, and the true team, must pass the rule; else :class:`LegalityRuleError`.
        (A rule that rejects the truth would exclude it from every world's support: a defect, never an exclusion.)"""
        ents = team.split("]")
        bad = [(i, e, p) for i, (e, p) in enumerate(zip(ents, self.set_problems(ents))) if p]
        whole = self.team_problems([team])[0]
        if bad or whole:
            first = "; ".join(f"slot {i} {e.split('|')[0] or e.split('|')[1]}: {' / '.join(p[:2])}"
                              for i, e, p in bad[:3])
            raise LegalityRuleError(f"{key}: the TRUE team fails the builder's legality rule (a defect of the rule or "
                                    f"of the truth's data, never an exclusion): {first or ' / '.join(whole[:3])}")
