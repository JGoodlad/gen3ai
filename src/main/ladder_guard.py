"""The LADDER policy, in code (owner ruling 2026-10-09).

    "Yes, allow mode ladder, just never for our agents without my explicit approval, and we would only
    ever do it at concurrency 1."

``python src/main/play.py --mode ladder`` is allowed for USERS, behind four hard guards, and this module is the one
place they live (``main.play`` calls :func:`check_ladder_policy` before it connects; ``main.live.client``'s
``LiveClient.ladder`` calls it again, silently, before EVERY ``/search`` — the seam that actually queues, so a caller
that skips ``main.play`` is guarded too, and a guard that lapses mid-session stops the session at the next game):

1. **CONCURRENCY 1, forced** (:data:`LADDER_CONCURRENCY`). One battle at a time, no option to raise it: the ladder
   client sends the next ``/search`` only after the previous battle ended and refuses a second live battle room, and
   ``main.play`` refuses a typed ``--concurrency`` with this reason.
2. **No parse panic outstanding.** The T28 halt marker (``main.live.halt``) refuses every live entry point.
3. **A FRESH drift-gate record.** The public server runs Showdown master and the live reader refuses an unknown
   keyword BY DESIGN, so a session needs ``python src/main/ladder_drift_scan.py`` to have passed in the last
   :data:`DRIFT_MAX_AGE_DAYS` days. The scan writes the record (:func:`record_drift_result`) — the LATEST result, so
   a later drift finding revokes an earlier green. Only a FULL scan (fresh download, fresh Showdown master clone,
   every check, at least :data:`DRIFT_MIN_REPLAYS` replays) can record a green; a partial one can only record drift.
4. **NEVER for OUR agents without the owner's explicit approval.** A session Claude Code runs in (``CLAUDECODE`` /
   ``CLAUDE_CODE_ENTRYPOINT`` / ``CLAUDE_CODE_SESSION_ID`` in the environment, or in any ANCESTOR process's, so
   ``env -u CLAUDECODE python …`` does not escape it) REFUSES unless the OWNER has written the approval token
   (:func:`approval_path`) — a JSON file with a ``purpose`` and an ``expires`` at most :data:`APPROVAL_MAX_HOURS`
   ahead. Agents must never create it. A revoked or expired token stops the session at the next game boundary.

THE HONEST LIMIT. Guard 4 is a TRIPWIRE against an agent doing this by accident or on its own initiative — not a
security boundary. Nothing here stops a process that deliberately forges the markers' absence in a way the ancestor
walk cannot see (a re-parented daemon on a box with no ``/proc``), nor an agent that writes the token file itself;
the second is why the refusal says the owner creates it and the project's CLAUDE.md says an agent never does.

WHERE THE STATE LIVES. Per USER, like the halt marker: ``~/.local/state/gen3ai/`` (``$GEN3AI_LADDER_APPROVAL_FILE`` /
``$GEN3AI_LADDER_DRIFT_FILE`` override and are authoritative). 🚨 Under pytest the defaults are SEALED (a READ sees no
file, a WRITE raises :class:`LadderStateUnsealed`): a test must name its own files, never touch the owner's real
approval or drift record.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: One battle at a time. A constant of the policy, not a flag: there is no option to raise it.
LADDER_CONCURRENCY = 1
#: The drift-gate record is good for this long (the public server's master moves; so does a Showdown deploy).
DRIFT_MAX_AGE_DAYS = 2
#: A scan of fewer replays than this is a smoke, not a gate: it can record DRIFT but never a green.
DRIFT_MIN_REPLAYS = 60
#: The owner's approval token may carry an expiry at most this far ahead of NOW (a token written for "next
#: month" is refused until it is within a day of its expiry).
APPROVAL_MAX_HOURS = 24
#: Environment variables Claude Code sets in the commands it runs (verified in a harness shell, 2026-10-09:
#: ``CLAUDECODE=1``, ``CLAUDE_CODE_ENTRYPOINT=cli``, ``CLAUDE_CODE_SESSION_ID``). ANY of them marks an agent session.
AGENT_ENV_MARKERS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID")

APPROVAL_ENV = "GEN3AI_LADDER_APPROVAL_FILE"
DRIFT_ENV = "GEN3AI_LADDER_DRIFT_FILE"
DRIFT_SCHEMA = "gen3_ladder_drift_gate_v1"

#: How many ancestor processes the agent-session walk inspects.
_MAX_ANCESTORS = 64
_PYTEST_UNSEALED_APPROVAL = Path("/nonexistent/gen3ai-ladder-approval-unsealed-under-pytest.json")
_PYTEST_UNSEALED_DRIFT = Path("/nonexistent/gen3ai-ladder-drift-unsealed-under-pytest.json")


class LadderRefused(RuntimeError):
    """A ladder guard refused; the message names which one and the way out."""


class LadderStateUnsealed(RuntimeError):
    """Under pytest with no override: the real approval / drift file must never be written by a test."""


def _state_dir() -> Path:
    return Path.home() / ".local" / "state" / "gen3ai"


def approval_path() -> Path:
    """The owner's approval token: ``$GEN3AI_LADDER_APPROVAL_FILE``, else the per-user default. Sealed under pytest."""
    env = os.environ.get(APPROVAL_ENV)
    if env:
        return Path(env)
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return _PYTEST_UNSEALED_APPROVAL
    return _state_dir() / "ladder_owner_approval.json"


def drift_path(*, write: bool = False) -> Path:
    """The drift-gate record: ``$GEN3AI_LADDER_DRIFT_FILE``, else the per-user default. Sealed under pytest."""
    env = os.environ.get(DRIFT_ENV)
    if env:
        return Path(env)
    if os.environ.get("PYTEST_CURRENT_TEST"):
        if write:
            raise LadderStateUnsealed(
                f"a test tried to WRITE the REAL ladder drift record: set ${DRIFT_ENV} to a tmp path")
        return _PYTEST_UNSEALED_DRIFT
    return _state_dir() / "ladder_drift_green.json"


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _parse_time(text: Any) -> _dt.datetime:
    """An ISO-8601 instant WITH a UTC offset (``Z`` accepted). A naive stamp is refused: it has no meaning across
    time zones, and 'expires at 18:00' must not depend on where the box thinks it is."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing")
    s = text.strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    t = _dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        raise ValueError("no UTC offset (write e.g. 2026-10-09T18:00:00+00:00)")
    return t


# -- guard 4: agent sessions need the owner's token -----------------------------------------------------------------

def _ancestor_markers() -> List[Tuple[int, str]]:
    """``(pid, marker)`` for every ancestor process whose INITIAL environment carries an agent marker.

    ``/proc/<pid>/environ`` is the environment the process was started with — exactly what a child inherits — so the
    harness's shell still shows ``CLAUDECODE`` even when the final ``python`` was started as ``env -u CLAUDECODE …``.
    Linux-only and best effort: an unreadable ancestor (or no ``/proc``) is skipped, never an error."""
    out: List[Tuple[int, str]] = []
    pid = os.getppid()
    for _ in range(_MAX_ANCESTORS):
        if pid <= 1:
            break
        try:
            blob = Path(f"/proc/{pid}/environ").read_bytes()
            entries = [kv.decode("utf-8", "replace") for kv in blob.split(b"\0") if kv]
            out.extend((pid, k) for k in AGENT_ENV_MARKERS
                       if any(e.startswith(k + "=") and e != k + "=" for e in entries))
        except OSError:
            pass
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])  # field 4 = ppid, after the "(comm)" field
        except (OSError, ValueError, IndexError):
            break
    return out


def agent_session_reasons(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """Why this process counts as an AGENT session (empty = a human's). ``environ=None`` is the real process: its own
    environment AND its ancestors'. An explicit mapping is checked as given, with no ancestor walk."""
    if environ is not None:
        return [f"${k}" for k in AGENT_ENV_MARKERS if environ.get(k)]
    reasons = [f"${k}" for k in AGENT_ENV_MARKERS if os.environ.get(k)]
    if not reasons:  # only a process that cleared its own markers needs the ancestors to give it away
        reasons = [f"${k} on ancestor process {pid}" for pid, k in _ancestor_markers()]
    return reasons


@dataclass(frozen=True)
class OwnerApproval:
    purpose: str
    expires: _dt.datetime
    path: Path


def read_owner_approval(now: Optional[_dt.datetime] = None, path: Optional[Path] = None, *,
                        why: Tuple[str, ...] = ()) -> OwnerApproval:
    """The owner's token, or :class:`LadderRefused` saying exactly what is wrong with it (``why`` = the markers that
    made this an agent session, named in the refusal)."""
    now = now or _utcnow()
    p = path or approval_path()
    who = f" ({', '.join(why)})" if why else ""
    example = ('{"purpose": "<what this ladder session is for>", '
               f'"expires": "{(now + _dt.timedelta(hours=4)).isoformat(timespec="seconds")}"}}')
    if not p.exists():
        raise LadderRefused(
            f"ladder REFUSED in an agent session{who}: agent sessions need the owner's explicit approval; the owner creates "
            "the token; agents must never create it. "
            f"No approval token at {p}. (The owner writes it by hand, e.g. {example} — an expiry at most "
            f"{APPROVAL_MAX_HOURS} h ahead and a purpose.)")
    try:
        obj = json.loads(p.read_text())
        if not isinstance(obj, dict):
            raise ValueError("not a JSON object")
        purpose = obj.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("'purpose' must be a non-empty string")
        expires = _parse_time(obj.get("expires"))
    except (OSError, ValueError) as exc:
        raise LadderRefused(
            f"ladder REFUSED in an agent session{who}: the owner's approval token at {p} is unreadable ({exc}); "
            f"the owner rewrites it, e.g. {example}. Agents must never create it.") from exc
    if expires <= now:
        raise LadderRefused(
            f"ladder REFUSED in an agent session{who}: the owner's approval token at {p} EXPIRED at "
            f"{expires.isoformat(timespec='seconds')}. Agent sessions need the owner's explicit approval; the owner "
            "creates a new token; agents must never create it.")
    if expires > now + _dt.timedelta(hours=APPROVAL_MAX_HOURS):
        raise LadderRefused(
            f"ladder REFUSED in an agent session{who}: the owner's approval token at {p} expires at "
            f"{expires.isoformat(timespec='seconds')}, more than {APPROVAL_MAX_HOURS} h ahead; an approval is for a "
            f"day at most. The owner rewrites it, e.g. {example}.")
    return OwnerApproval(purpose.strip(), expires, p)


# -- guard 3: the drift gate ----------------------------------------------------------------------------------------

def record_drift_result(rc: int, *, battle_format: str, n_replays: int, full: bool, showdown_commit: Optional[str],
                        checks: List[str], now: Optional[_dt.datetime] = None,
                        path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Record a ``ladder_drift_scan`` outcome (the LATEST one wins). Returns the record, or ``None`` when nothing was
    written: ``rc`` 0 from a PARTIAL scan (offline / skipped checks / a user-named Showdown checkout / too few replays)
    is not evidence of a fresh pass; ``rc`` 2 (no replays) is not a verdict at all; ``rc`` 1 (drift found) is always
    recorded, so a drift finding revokes an earlier green even from a partial scan."""
    if rc not in (0, 1) or (rc == 0 and not (full and n_replays >= DRIFT_MIN_REPLAYS)):
        return None
    from utils.git import get_git_hash
    rec: Dict[str, Any] = {
        "schema": DRIFT_SCHEMA,
        "time": (now or _utcnow()).isoformat(timespec="seconds"),
        "status": "green" if rc == 0 else "drift",
        "format": battle_format,
        "n_replays": int(n_replays),
        "commit": get_git_hash(),
        "showdown_master": showdown_commit,
        "checks": list(checks),
    }
    p = path or drift_path(write=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)
    return rec


def check_drift_fresh(battle_format: str, now: Optional[_dt.datetime] = None,
                      path: Optional[Path] = None) -> Dict[str, Any]:
    """The drift record if it is a GREEN, for this format, at most :data:`DRIFT_MAX_AGE_DAYS` old; else
    :class:`LadderRefused` naming the command to run."""
    now = now or _utcnow()
    p = path or drift_path()
    fix = ("run `python src/main/ladder_drift_scan.py --n 200` (a full scan: it downloads fresh replays and a fresh "
           f"Showdown master clone; it records the result in {p}) and fix any drift it finds")
    if not p.exists():
        raise LadderRefused(f"ladder REFUSED: no drift-gate record at {p}: {fix}.")
    try:
        rec = json.loads(p.read_text())
        t = _parse_time(rec.get("time"))
        status, fmt = rec.get("status"), rec.get("format")
    except (OSError, ValueError, AttributeError) as exc:
        raise LadderRefused(f"ladder REFUSED: the drift-gate record at {p} is unreadable ({exc}): {fix}.") from exc
    if status != "green":
        raise LadderRefused(f"ladder REFUSED: the latest drift scan ({t.isoformat(timespec='seconds')}) FOUND DRIFT "
                            f"(status {status!r}): the live reader would halt on a real game. Fix it, then {fix}.")
    if fmt != battle_format:
        raise LadderRefused(f"ladder REFUSED: the drift record is for format {fmt!r}, this session is "
                            f"{battle_format!r}: {fix}.")
    age = now - t
    if age < _dt.timedelta(minutes=-5) or age > _dt.timedelta(days=DRIFT_MAX_AGE_DAYS):
        raise LadderRefused(
            f"ladder REFUSED: the drift-gate record is {age.total_seconds() / 3600.0:.1f} h old (limit "
            f"{DRIFT_MAX_AGE_DAYS * 24} h, and not from the future): {fix}.")
    return rec


# -- the entry ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderPermit:
    """What :func:`check_ladder_policy` found: how this session was cleared."""
    concurrency: int
    battle_format: str
    agent_reasons: Tuple[str, ...]
    approval: Optional[OwnerApproval]
    drift: Dict[str, Any]


def check_ladder_policy(*, battle_format: str, now: Optional[_dt.datetime] = None) -> LadderPermit:
    """All the ladder guards, in order: the T28 halt (:class:`main.live.halt.HaltActive`), the agent-session approval,
    the drift record (both :class:`LadderRefused`). Raises, never prints — callable before every ``/search``."""
    from main.live.halt import refuse_if_halted
    now = now or _utcnow()
    refuse_if_halted("ladder")
    reasons = agent_session_reasons()
    approval = read_owner_approval(now, why=tuple(reasons)) if reasons else None
    drift = check_drift_fresh(battle_format, now)
    return LadderPermit(LADDER_CONCURRENCY, battle_format, tuple(reasons), approval, drift)


def respect_notice(permit: LadderPermit, *, forfeit_turn_limit: int) -> str:
    """The short respect-for-players notice a ladder session prints at startup."""
    lines = [
        "[ladder] RESPECT FOR PLAYERS: the other side is a real person. This session plays ONE battle at a time "
        f"(concurrency {permit.concurrency}, fixed), never chats, never sends a PM and never challenges anyone; a "
        f"stalled game is forfeited at turn {forfeit_turn_limit}, as a training episode is; a parse panic HALTS all "
        "play until root-caused (python -m main.live.halt status). Follow the server's rules and its policy on bots.",
        f"[ladder] drift gate: green at {permit.drift.get('time')} (commit {str(permit.drift.get('commit'))[:12]}, "
        f"Showdown master {str(permit.drift.get('showdown_master'))[:12]}).",
    ]
    if permit.approval is not None:
        lines.append(f"[ladder] AGENT SESSION ({', '.join(permit.agent_reasons)}) running under the OWNER'S approval: "
                     f"{permit.approval.purpose!r}, expires {permit.approval.expires.isoformat(timespec='seconds')}.")
    return "\n".join(lines)
