#!/usr/bin/env python3
"""NO-TRAINING repro for the ``--use-bridge=rust`` worker death (~8 min, twice out of two).

Runs in ~5 seconds. No model, no PPO, no 25M-step run, no Showdown server, no
``SubprocVecEnv``: it drives one ``sim_bridge`` child over a raw pipe and speaks the
documented stdin/stdout protocol by hand. Node is the control arm and must pass every case.

WHY AN ``__ERR__`` IS THE WHOLE BUG
-----------------------------------
``__ERR__`` is NOT an in-band error. ``BridgeSession._dispatch`` raises on it,
``_persistent_read_loop`` retires the reader and calls ``_signal_transport_dead()``, and every
in-flight ``step()`` parked in ``race_get`` raises
``ShowdownException: Showdown websocket dropped while waiting for battle state`` — about a
websocket the bridge does not use. So **anything node tolerates on a CHOOSE, rust must tolerate
too**; a stricter parser there is a whole-run crash, not a better error.

That also explains why the production evidence looked like nothing: the ``__ERR__`` text is
latched into ``_child_error`` and printed only by the NEXT ``reset()`` — which never runs,
because the worker dies in ``step()`` first and the parent then cascades on dead
``SubprocVecEnv`` pipes. And the appended ``race_trace.dump_recent()`` came back empty simply
because that ring buffer is a no-op unless ``GEN3_RACE_TRACE=1``.

THE TWO DIVERGENCES
-------------------
1. ``CHOOSE <side> default`` / ``pass`` — the Node bridge writes every token to the sim
   verbatim, so Showdown's ``Side.choose`` handled ``default``/``auto``/``pass``/``skip``;
   ``bridge.rs::parse_choice`` accepted only ``move ``/``switch `` and answered ``__ERR__``.
   ``/choose default`` is ROUTINE, not a tail event: ``singles_env.py``'s ``action == -2``, an
   inference player whose predict returns ``None``, its redecide-budget exhaustion, and
   ``Player.DEFAULT_CHOICE_CHANCE``. This fires as a RATE per decision, which is why the crash
   landed at the same ~8-minute mark at load 31 and at load 5 alike.
2. A stray CHOOSE after ``__END__`` — in PERSISTENT mode (the training default) the child
   resets itself at ``__END__`` (``Session::reset`` → ``bridge = None``), and
   ``BridgeSession._dispatch`` fires poke-env's feeds as UN-AWAITED tasks, so a late answer to
   the ending battle's last ``|request|`` can arrive with no battle live. ``handle_choose``
   already dropped it, but the CHOOSE arm then fell through to ``flush_new_chunks``, which has
   no bridge and returned ``Err("no battle in progress (missing START)")``.

Note what the child does NOT do in either case: it does not exit and it does not hang. It stays
alive and healthy (``child alive after: True``) and will happily accept the next ``START``. The
casualty is the parent's reader.

USAGE
-----
    export PYTHONPATH=$PWD/src
    python src/rust_sim/harness/rust_bridge_stray_choose_repro.py            # both impls
    python src/rust_sim/harness/rust_bridge_stray_choose_repro.py --impl rust

Exit 0 = every case behaved. Exit 1 = reproduced.
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import subprocess
import sys
import threading
from typing import List, Optional

from utils.bridge.sim_bridge_bin import bridge_spawn_argv
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

_READ_TIMEOUT_S = 20.0


class Child:
    """A raw pipe to one sim_bridge child. Line protocol, nothing else.

    ONE background reader thread feeds a queue. (A per-call ``readline`` thread that is
    abandoned on timeout stays blocked in ``read`` and swallows the *next* line into a box
    nobody reads — which silently turns a real ``__ERR__`` into an apparent success.)
    """

    def __init__(self, impl: str):
        self.impl = impl
        self.proc = subprocess.Popen(
            bridge_spawn_argv(impl),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.stderr: List[bytes] = []
        self._lines: "queue.Queue[str]" = queue.Queue()
        for target in (self._pump_stdout, self._drain_stderr):
            threading.Thread(target=target, daemon=True).start()

    def _pump_stdout(self) -> None:
        for raw in self.proc.stdout:
            self._lines.put(raw.decode().rstrip("\n"))
        self._lines.put("")  # EOF sentinel

    def _drain_stderr(self) -> None:
        for line in self.proc.stderr:
            self.stderr.append(line)

    def send(self, line: str) -> None:
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()

    def readline(self, timeout: float = _READ_TIMEOUT_S) -> Optional[str]:
        """One stdout line, ``""`` on EOF, or ``None`` if silent for ``timeout``."""
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        try:
            self.send("END")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def _decode(frame: str) -> str:
    return base64.b64decode(frame.split(" ", 1)[1]).decode()


def _choice_for(chunk: str) -> Optional[str]:
    """Turn a ``|request|`` line into a legal choice token, or ``None`` if none is owed.

    Deliberately minimal — enough to walk a gen3ou battle to its end (moves + the
    force-switch path). ``wait`` requests owe nothing.
    """
    for line in chunk.split("\n"):
        if not line.startswith("|request|") or len(line) <= len("|request|"):
            continue
        req = json.loads(line[len("|request|"):])
        if req.get("wait"):
            return None
        side = req.get("side") or {}
        mons = side.get("pokemon") or []
        if req.get("forceSwitch"):
            for i, mon in enumerate(mons, start=1):
                if not mon.get("active") and "0 fnt" not in (mon.get("condition") or ""):
                    return f"switch {i}"
            return "default"
        active = (req.get("active") or [None])[0]
        if active:
            for i, mv in enumerate(active.get("moves", []), start=1):
                if not mv.get("disabled") and mv.get("pp", 1) > 0:
                    return f"move {i}"
            return "move 1"
    return None


def _teams():
    loader = TeamLoader()
    return loader.get_sample_teams() or loader.get_all_teams()


def run_case(impl: str, *, end_via: str, verbose: bool = True) -> dict:
    """Drive one battle to ``__END__`` on a PERSISTENT child, then send ONE stray CHOOSE.

    ``end_via='forfeit'`` ends it with FORCELOSE (the training reset-mid-battle path);
    ``end_via='natural'`` plays it out. Returns what the child did with the stray CHOOSE.
    """
    tb = Gen3Teambuilder(_teams())
    child = Child(impl)
    result = {"impl": impl, "end_via": end_via}
    try:
        start = {
            "formatid": "gen3ou",
            "persistent": True,
            "p1": {"name": "P1", "team": tb.yield_team()},
            "p2": {"name": "P2", "team": tb.yield_team()},
        }
        child.send("START " + json.dumps(start))

        # --- drive to __END__ -------------------------------------------------------
        decisions = 0
        forfeited = False
        while True:
            line = child.readline()
            if line is None:
                result["outcome"] = "CHILD WENT SILENT before __END__"
                return result
            if line == "__END__":
                break
            if line.startswith("__ERR__"):
                result["outcome"] = f"unexpected __ERR__ during battle: {_decode(line)}"
                return result
            if line.startswith("__RECON__"):
                continue
            side = line.split(" ", 1)[0]
            choice = _choice_for(_decode(line))
            if choice is None:
                continue
            decisions += 1
            if decisions > 4000:
                result["outcome"] = "battle never ended (driver bug)"
                return result
            if end_via == "forfeit" and not forfeited and decisions >= 6 and side == "p2":
                # The training seam's path: reset() lands mid-battle → /forfeit → FORCELOSE,
                # while the OTHER side still has an unanswered |request| in flight.
                forfeited = True
                child.send("FORCELOSE p1")
                continue
            child.send(f"CHOOSE {side} {choice}")

        # --- the stray late CHOOSE --------------------------------------------------
        # In production this is the fire-and-forget ``client.feed()`` task in
        # BridgeSession._dispatch resolving into a poke-env choice AFTER the child already
        # emitted __END__ and reset itself. Here we just send it.
        child.send("CHOOSE p2 move 1")
        reply = child.readline(timeout=3.0)
        if reply is None or reply == "":
            result["outcome"] = "IGNORED (child stayed silent — correct)"
            result["failed"] = False
        elif reply.startswith("__ERR__"):
            result["outcome"] = f"__ERR__ {_decode(reply)}"
            result["failed"] = True
        else:
            result["outcome"] = f"unexpected frame: {reply[:80]}"
            result["failed"] = True

        # A child that emitted __ERR__ is still ALIVE and still accepts a new START — proof
        # that this is a protocol-semantics bug, not a crash. The Python side kills the run
        # anyway, because __ERR__ retires the reader.
        result["child_alive_after"] = child.proc.poll() is None
    finally:
        child.close()
    if verbose:
        flag = "❌ FAIL" if result.get("failed") else "✅ ok   "
        print(f"  {flag}  impl={impl:<5} end_via={end_via:<8} → {result['outcome']}"
              f"   (child alive after: {result.get('child_alive_after')})")
    return result


def run_token_case(impl: str, token: str, *, verbose: bool = True) -> dict:
    """Send ONE mid-battle ``CHOOSE p1 <token>`` and report what the child does with it.

    poke-env emits ``/choose default`` (``DefaultBattleOrder``) from
    ``Player._handle_battle_request`` whenever a request was rejected, with probability
    ``DEFAULT_CHOICE_CHANCE = 1/1000`` (``player.py:54,349,384``). ``BattleStreamClient``
    turns that into ``CHOOSE p1 default``. Showdown's ``Side.choose`` accepts it; the port's
    ``parse_choice`` (``bridge.rs:89``) accepts only ``move ``/``switch ``.
    """
    tb = Gen3Teambuilder(_teams())
    child = Child(impl)
    result = {"impl": impl, "token": token}
    try:
        child.send("START " + json.dumps({
            "formatid": "gen3ou", "persistent": True,
            "p1": {"name": "P1", "team": tb.yield_team()},
            "p2": {"name": "P2", "team": tb.yield_team()},
        }))
        # Drain the opening frames to SILENCE, so the only thing that can follow is the
        # child's answer to our token. (Stopping at the first p1 request leaves p2's frame
        # in the pipe and it gets misread as the reply.)
        saw_request = False
        while True:
            line = child.readline(timeout=3.0)
            if line is None:      # child went quiet: every opening frame is drained
                break
            if line == "":
                result["outcome"] = "child exited during opening"
                return result
            if not line.startswith("__") and _choice_for(_decode(line)) is not None:
                saw_request = True
        if not saw_request:
            result["outcome"] = "no opening request"
            return result

        child.send(f"CHOOSE p1 {token}")
        reply = child.readline(timeout=5.0)
        if reply is None or reply == "":
            result["outcome"] = "accepted (no error frame)"
            result["failed"] = False
        elif reply.startswith("__ERR__"):
            result["outcome"] = f"__ERR__ {_decode(reply)}"
            result["failed"] = True
        else:
            result["outcome"] = f"accepted → {reply.split(' ', 1)[0]} frame"
            result["failed"] = False
    finally:
        child.close()
    if verbose:
        flag = "❌ FAIL" if result.get("failed") else "✅ ok   "
        print(f"  {flag}  impl={impl:<5} token={token!r:<10} → {result['outcome']}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["node", "rust", "both"], default="both")
    args = ap.parse_args()

    impls = ["node", "rust"] if args.impl == "both" else [args.impl]
    results = []

    print("CASE 1 — the bare wire verbs poke-env sends: `/choose default` (singles_env action")
    print("         == -2, an inference player's None predict, its redecide exhaustion, and")
    print("         Player.DEFAULT_CHOICE_CHANCE) and `/choose pass`, on a live request:\n")
    for impl in impls:
        for token in ("default", "pass"):
            results.append(run_token_case(impl, token))

    print("\nCASE 2 — stray CHOOSE after __END__ on a PERSISTENT child:\n")
    for impl in impls:
        for end_via in ("natural", "forfeit"):
            results.append(run_case(impl, end_via=end_via))

    bad = [r for r in results if r.get("failed")]
    print()
    if bad:
        impls_bad = sorted({r["impl"] for r in bad})
        print(f"REPRODUCED: {len(bad)}/{len(results)} cases produced a fatal __ERR__ ({impls_bad}).")
        print("An __ERR__ retires BridgeSession's reader → _signal_transport_dead() →")
        print("ShowdownException('Showdown websocket dropped …') in every in-flight step().")
        return 1
    print(f"All {len(results)} cases behaved. No repro.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
