"""A FORCED BRANCH of a raw anchor game (``main.anchors --our-force SPEC``): the deviation-only live read's play path
(``designs/research_state/measurements/dlive_read_2026-10-28/``; the S-U gate review's M3, V-RANK's build B-V1).

A raw P* vs Kakuna game is deterministic on one engine (``kakuna_determinism_2026-10-27``: 0 of 2,100 replayed games
differ), so replaying its deal (``--deal-game G`` with the raw unit's ``--seed-base`` / ``--team-seed``) reaches every
decision of the raw game byte for byte. A forced branch replays the deal with our checkpoint greedy everywhere EXCEPT
at ONE named decision of ours, where it plays a named action (the search's pick, or π's own top as a NULL control),
then lets P* (greedy) and the peer finish the game. The branch's outcome against the raw game's is a real game's
answer to "what does playing this action here, then P* thereafter, win against this opponent?": no opponent model
anywhere in the truth.

**The spec** (one JSON file, schema :data:`SCHEMA`, built from the raw game's KEPT decision by :func:`spec_from_kept`):
the decision's identity (the raw battle's tag and its index in the series, ``game``; ``turn``; the core frame index
``n``; the live search ``key``), the action to
force, π's greedy action there (``pi_argmax``), and the raw decision's fingerprints: its choice ``tokens``, the live
core row's ``frame_sha256`` and :func:`log_sha256` of the battle's honest input log at that decision
(:func:`main.anchors.search_side.honest_log`: the START document — both packed teams, both names, the seed — and every
command of both sides before the decision).

**G-BRANCH (the GIGO tooth), checked live at the named decision BEFORE the forced action is sent.** The branch game's
index in its series (the front end's ``battle_index_base`` count: ``--deal-game G`` gives game G the index G a whole
series gives it; the TAG is a per-process counter, so a one-game unit's tag is not the raw game's and is never
compared), its honest input log's sha256, its choice tokens, its core row's sha256 and π's greedy action must all
equal the raw decision's. Equal logs mean the same seed, teams and every command of both sides through the decision, so the
two games share their whole prefix; the row and tokens pin what our side saw. Any difference is a named failure
(:class:`BranchError`, the unit's ``core_slot_error``): the unit never counts, and the read holds for a root cause. A
game that ends without reaching the named decision, or reaches it twice, fails the half's verification
(``regime_verified_decisions = false``, ``force_mismatch``), so it never counts either.

**The record**: one JSON line at the named decision in ``<out>/branch.jsonl`` (schema :data:`RECORD_SCHEMA`): every
fingerprint compared, the action played, and the guard's verdict, written before the action is sent (a failing guard
is recorded, then raised).

Pure and light at import (no torch, no engine): ``main.anchors.cli`` validates a spec with it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

SCHEMA = "gen3_forced_branch_v1"
RECORD_SCHEMA = "gen3_forced_branch_record_v1"
KINDS = ("branch", "null")
_FIELDS = ("schema", "kind", "key", "battle", "game", "turn", "n", "action", "pi_argmax", "tokens", "frame_sha256",
           "log_sha256")


class BranchError(RuntimeError):
    """A forced branch that cannot be played honestly: a spec that does not hold, or G-BRANCH's mismatch at the named
    decision (the branch game is not the raw game's prefix). Named, never a fallback to the greedy choice."""


def log_sha256(log: Mapping[str, Any]) -> str:
    """The sha256 of an honest input log (``search_side.honest_log``'s dict): canonical JSON, sorted keys."""
    return hashlib.sha256(json.dumps(dict(log), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stamp(spec: Mapping[str, Any]) -> str:
    """The suffix a forced branch adds to ``our_regime``: ``+force=<kind>:a<action>@t<turn>n<n>``, so a branch row
    never pools with a raw one."""
    return f"+force={spec['kind']}:a{int(spec['action'])}@t{int(spec['turn'])}n{int(spec['n'])}"


def validate(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """The spec as a dict, or a :class:`BranchError` naming what does not hold."""
    if not isinstance(spec, Mapping):
        raise BranchError(f"a forced-branch spec is a JSON object, not {type(spec).__name__}")
    missing = [k for k in _FIELDS if k not in spec]
    if missing:
        raise BranchError(f"a forced-branch spec lacks {missing}")
    if spec["schema"] != SCHEMA:
        raise BranchError(f"spec schema {spec['schema']!r}, expected {SCHEMA!r}")
    if spec["kind"] not in KINDS:
        raise BranchError(f"spec kind {spec['kind']!r} is none of {KINDS}")
    for k in ("game", "turn", "n", "action", "pi_argmax"):
        if isinstance(spec[k], bool) or not isinstance(spec[k], int) or spec[k] < 0:
            raise BranchError(f"spec {k}={spec[k]!r}: an int >= 0")
    if not 0 <= spec["action"] <= 10 or not 0 <= spec["pi_argmax"] <= 10:
        raise BranchError(f"spec action {spec['action']} / pi_argmax {spec['pi_argmax']}: an action index 0..10")
    tokens = spec["tokens"]
    if not isinstance(tokens, Mapping) or len(tokens) < 2:
        raise BranchError("spec tokens: the raw decision's choice tokens, >= 2 legal actions")
    if str(spec["action"]) not in tokens or str(spec["pi_argmax"]) not in tokens:
        raise BranchError(f"spec action {spec['action']} / pi_argmax {spec['pi_argmax']} is not a legal action of "
                          f"the raw decision {sorted(tokens)}")
    if (spec["kind"] == "null") != (spec["action"] == spec["pi_argmax"]):
        raise BranchError(f"a {spec['kind']!r} spec forces action {spec['action']} with pi_argmax "
                          f"{spec['pi_argmax']}: a null branch forces π's own top, a branch any other action")
    for k in ("frame_sha256", "log_sha256"):
        v = spec[k]
        if not (isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)):
            raise BranchError(f"spec {k}: a sha256 hex digest")
    return dict(spec)


def load_spec(path: Any) -> Dict[str, Any]:
    p = Path(path)
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise BranchError(f"{p}: unreadable forced-branch spec ({exc})") from None
    return validate(doc)


def spec_from_kept(rec: Mapping[str, Any], action: int, *, game: int) -> Dict[str, Any]:
    """The spec forcing ``action`` at a KEPT raw decision (``main.anchors.input_log``'s record) of game ``game`` of its
    series: kind ``null`` when ``action`` is π's greedy action there, ``branch`` otherwise."""
    if int(rec.get("index", game)) != int(game):
        raise BranchError(f"{rec.get('key')}: the kept decision is battle index {rec.get('index')}, not game {game}")
    spec = {"schema": SCHEMA, "kind": "null" if int(action) == int(rec["pi_argmax"]) else "branch",
            "key": str(rec["key"]), "battle": str(rec["battle"]), "game": int(game), "turn": int(rec["turn"]),
            "n": int(rec["n"]), "action": int(action), "pi_argmax": int(rec["pi_argmax"]),
            "tokens": {str(k): str(v) for k, v in rec["tokens"].items()}, "frame_sha256": str(rec["frame_sha256"]),
            "log_sha256": log_sha256(rec["log"])}
    return validate(spec)


class ForcedBranch:
    """Plays one forced branch (module docstring) for one half of a ``--deal-game`` unit: ``choose`` is called at
    every decision of ours after the greedy policy decided, and returns the action to play."""

    def __init__(self, spec: Mapping[str, Any], out: Path) -> None:
        self.spec = validate(spec)
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.stats: Dict[str, Any] = {"key": self.spec["key"], "kind": self.spec["kind"],
                                      "action": self.spec["action"], "turn": self.spec["turn"], "n": self.spec["n"],
                                      "decisions": 0, "forced": 0, "g_branch": None}

    def choose(self, front: Any, tag: str, d: Mapping[str, Any], mask: Any, conn: Any, argmax: int) -> int:
        """The action to play at this decision of ours: the forced one at the named decision (after G-BRANCH),
        ``argmax`` (the greedy choice) everywhere else."""
        from main.anchors.input_log import frame_sha256
        from main.anchors.search_side import honest_log

        self.stats["decisions"] += 1
        turn, n = int(d.get("turn") or 0), int(d["n"])
        if (turn, n) != (self.spec["turn"], self.spec["n"]):
            return int(argmax)
        if self.stats["forced"]:
            raise BranchError(f"{self.spec['key']}: the named decision t{turn} n{n} was reached twice")
        battle = front.battles.get(tag)
        if battle is None:
            raise BranchError(f"{tag}: the front end holds no such battle")
        side = next((s for s, c in battle.conns.items() if c is conn), None)
        if side is None:
            raise BranchError(f"{tag}: our connection is not a side of the battle")
        log, at, update = honest_log(battle, side)
        index = getattr(battle, "index", None)
        live = {"game": None if index is None else int(index), "log_sha256": log_sha256(log),
                "tokens": {str(k): str(v) for k, v in (d.get("tokens") or {}).items()},
                "frame_sha256": frame_sha256(d["frame"]["b64"]), "pi_argmax": int(argmax)}
        diff = sorted(k for k, v in live.items() if v != self.spec[k])
        legal = [int(i) for i in np.flatnonzero(np.asarray(mask))]
        if self.spec["action"] not in legal:
            diff.append("action_illegal")
        rec = {"schema": RECORD_SCHEMA, "key": self.spec["key"], "kind": self.spec["kind"], "battle": str(tag),
               "index": live["game"],
               "side": side, "turn": turn, "n": n, "at": at, "update": update, "legal": legal,
               "forced": int(self.spec["action"]), "pi_argmax": int(argmax), "live": live,
               "g_branch": "pass" if not diff else "fail", "mismatch": diff}
        with open(self.out, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
        self.stats["g_branch"] = rec["g_branch"]
        if diff:
            raise BranchError(f"G-BRANCH: {self.spec['key']} t{turn} n{n}: the branch game's {diff} differ from the "
                              "raw decision's (the replay is not the raw game's prefix)")
        self.stats["forced"] += 1
        return int(self.spec["action"])

    def summary(self) -> Dict[str, Any]:
        return dict(self.stats, path=str(self.out))


def check_half(stats: Optional[Mapping[str, Any]], spec: Mapping[str, Any], games: int) -> Optional[str]:
    """Why a half that played ``games`` games did NOT play the spec's branch exactly as named (None = it did): a half
    with a game must have forced once with G-BRANCH passing; a half with none forces nothing."""
    if games <= 0:
        return None
    if stats is None:
        return "the half had no forced-branch player"
    if stats.get("key") != spec["key"]:
        return f"the half forced {stats.get('key')!r}, the spec names {spec['key']!r}"
    if stats.get("forced") != 1 or stats.get("g_branch") != "pass":
        return (f"the named decision t{spec['turn']} n{spec['n']} was forced {stats.get('forced')} time(s), "
                f"G-BRANCH {stats.get('g_branch')!r} (needs exactly 1, 'pass')")
    return None
