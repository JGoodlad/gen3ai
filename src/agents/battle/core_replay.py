"""Replay recorded battles through ``core_events`` — the CORE half of the old parity harness, owned on the Rust path.

``RecordedBattle`` is one battle's INPUT log (format, seed, both players, every command); :func:`run_core` feeds a
batch of them to ONE ``core_events`` process (``--trackers`` / ``--obs`` / ``--views`` / ``--record-dir`` as asked)
and returns one result dict per battle. Moved out of ``agents.battle.rust_core_parity`` in P6 of the poke-env
retirement (2026-10-08): that module compared the core to the Python ``Gen3Battle`` (poke-env) and is deleted with
it, while ``main.policy_spectrum`` (Lane S's bank and replay) reads battles through this half alone. Nothing here
imports poke-env.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

@dataclass
class RecordedBattle:
    """One battle's INPUT log (what both paths re-derive from) + the bytes its players got."""

    label: str
    format_id: str
    seed: str
    p1: Dict[str, str]
    p2: Dict[str, str]
    commands: List[List[str]]
    init_seed: bool = False
    quick_claw: bool = False
    #: sha256 of the per-side chunks as RECORDED (``None`` for a corpus with no live capture).
    chunks_sha: Optional[str] = None

    def script(self) -> List[str]:
        start = {"label": self.label, "formatid": self.format_id, "seed": self.seed,
                 "p1": self.p1, "p2": self.p2, "init_seed": self.init_seed,
                 "quick_claw": self.quick_claw}
        out = ["START " + json.dumps(start)]
        for cmd in self.commands:
            side, choice = cmd[0], cmd[1]
            if side == "forcelose":
                out.append(f"FORCELOSE {choice}")
            else:
                # A capture golden's per-decision choice is fed only while the side's choice is
                # open (`CHOOSEIF`); a live recording's command is fed as the child received it.
                verb = "CHOOSEIF" if len(cmd) > 2 and cmd[2] == "if_open" else "CHOOSE"
                out.append(f"{verb} {side} {choice}")
        out.append("END")
        return out


def chunks_sha(chunks: Sequence[Tuple[str, str]]) -> str:
    """The digest of a battle's per-side chunk stream, ``[(side, text), …]`` in flush order."""
    h = hashlib.sha256()
    for side, text in chunks:
        h.update(f"{side}\x00{text}\x01".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the core
# ---------------------------------------------------------------------------

#: The EMISSION SELF-CHECK's stderr summary (``emission_check::summary``), printed by a
#: ``core_events`` built with the check on — a production build prints none.
_SELFCHECK_RE = re.compile(r"^emission_selfcheck omniscient=(\d+) per_viewer=(\d+) split=(\d+) frame=(\d+)$")
SELFCHECK_KINDS = ("omniscient", "per_viewer", "split", "frame")


def selfcheck_counts(stderr: str) -> Optional[Dict[str, int]]:
    """The self-check counts a ``core_events`` run reported on stderr, or ``None`` when the binary
    ran without the check (a production build)."""
    for line in stderr.splitlines():
        m = _SELFCHECK_RE.match(line.strip())
        if m:
            return dict(zip(SELFCHECK_KINDS, map(int, m.groups())))
    return None


def run_core(battles: Sequence[RecordedBattle], record_dir: Optional[str] = None,
             commit: str = "unknown", views: bool = False,
             selfcheck: Optional[collections.Counter] = None, trackers: bool = False,
             obs: bool = False) -> List[dict]:
    """Replay ``battles`` through the core in ONE process; one result dict per battle.
    ``views`` also captures slice V's decision boards (``core_events --views``). ``selfcheck``
    accumulates the EMISSION SELF-CHECK's counts (``selfcheck["runs_without"]`` counts a process
    that ran without the check)."""
    from utils.bridge.sim_bridge_bin import resolve_core_events_bin

    argv = [resolve_core_events_bin()]
    if views:
        argv.append("--views")
    if trackers:
        argv.append("--trackers")
    if obs:
        argv.append("--obs")
    if record_dir:
        argv += ["--record-dir", record_dir, "--commit", commit]
    stdin = "\n".join(line for b in battles for line in b.script()) + "\n"
    p = subprocess.run(argv, input=stdin, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"core_events failed (exit {p.returncode}): {p.stderr.strip()[-2000:]}")
    if selfcheck is not None:
        counts = selfcheck_counts(p.stderr)
        if counts is None:
            selfcheck["runs_without"] += 1
        else:
            selfcheck.update(counts)
    out = [json.loads(line) for line in p.stdout.splitlines() if line.strip()]
    if len(out) != len(battles):
        raise RuntimeError(f"core_events answered {len(out)} battles for {len(battles)}")
    return out


def core_chunks(res: dict) -> List[Tuple[str, str]]:
    return [("p1" if side == 0 else "p2", "\n".join(lines)) for side, lines in res["chunks"]]
