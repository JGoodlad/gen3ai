"""SearchSession — a WARM, persistent clone-and-branch search-server (the Python half of the
search driver: ``search_driver.js`` under ``impl="node"``, the ``search_driver`` Rust binary
under ``impl="rust"``).

Where :func:`reconstruction.reroll_turn` / :func:`reconstruction.reroll_many` spawn a
fresh driver process and rebuild the battle from turn 1 per call, ``SearchSession`` keeps
ONE process alive holding a node-snapshot cache, so a multi-ply BEAM over candidate
lines (the prober's ``better_line`` probe) can branch a search TREE from any explored
node by cloning its mid-battle state (``State.serializeBattle`` / ``deserializeBattle``,
~1.7 ms/clone, constant in depth) instead of re-replaying the whole prefix per node.

**Transport selection.** ``impl`` picks WHICH driver child is spawned, exactly the way
``--use-bridge={node,rust}`` picks the live battle transport; the seam is
``sim_bridge_bin.search_driver_spawn_argv``. ``"node"`` (the default) is the historical
behavior byte-for-byte. ``"rust"`` execs the std-only ``src/rust_sim`` ``search_driver``
binary, which speaks the identical request→one-line-response protocol — so nothing below
changes but the executable. A missing/unbuildable rust binary raises a clear
``SimBridgeBinaryError``; it NEVER falls back to node (a "rust" search that silently ran on
node would answer a different question than the one asked).

Lifecycle (one process per ``better_line`` call; context-managed):

    with SearchSession(record) as ss:
        root = ss.open_root(turn)
        nodes = ss.expand_many([
            {"node_id": root.node_id, "p1_action": "move 1",
             "p2_action": "move 2", "seed": "original", "label": 0},
            ...
        ])

The per-side ``p1_chunks`` / ``p2_chunks`` an expand returns are the COMPLETE one-sided
views (root → that node) — feed them straight to :mod:`agents.training.obs_materializer`,
no cross-ply threading. ``pre_state`` / ``outcome`` are OMNISCIENT (referee view): they
drive only the opponent model and the dice, never the obs encoder (the one-sided /
omniscient wall, identical to the re-roll path).

The protocol is synchronous request → one-line response; calls are strictly sequential
(a beam expands one batch at a time), so a background reader thread feeds a queue that
:meth:`_call` drains with a timeout — a wedged child fails ONE call (and the session),
never hangs the prober.
"""

from __future__ import annotations

import collections
import json
import queue
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence

from utils.bridge.reconstruction import ReconstructionRecord
from utils.bridge.sim_bridge_bin import search_driver_spawn_argv


@dataclass(frozen=True)
class RootView:
    """The decision-point root of a search (turn-T board, both requests open)."""

    node_id: str
    requests: dict            # {"p1": request-json, "p2": ...} — the choice surfaces
    recorded_choices: dict    # {"p1": str|None, "p2": str|None} — the original turn-T picks
    pre_state: dict           # omniscient board snapshot (referee view)
    prefix_p1_chunks: List[str]
    prefix_p2_chunks: List[str]


@dataclass(frozen=True)
class ExpandedNode:
    """One expanded arm: the child node + its FULL one-sided chunks + outcome."""

    label: object
    node_id: Optional[str]    # None when the turn ENDED the battle (terminal — no child)
    ended: bool
    stuck: bool
    outcome: dict             # omniscient (referee view)
    requests: Optional[dict]  # the child's open requests (None when ended)
    choices_used: dict
    p1_chunks: List[str]      # COMPLETE one-sided view, root → this node
    p2_chunks: List[str]


class SearchError(RuntimeError):
    pass


class SearchSession:
    """A live search-driver process. One per ``better_line`` call, OR — for the search-teacher's
    background workers — ONE WARM process REUSED across many battles (pass ``record=None`` here and the
    per-battle record to :meth:`open_root`), so the ~0.6 s spawn is amortized over hundreds of
    searches instead of paid per search. Each ``open_root`` starts a fresh tree (the driver clears its
    node cache), so reuse is memory-bounded and battle-independent.

    ``impl`` selects the driver child (``"node"`` — the default and the historical behavior — or
    ``"rust"``); see the module header."""

    def __init__(self, record: "Optional[ReconstructionRecord]" = None, *, timeout: float = 120.0,
                 impl: str = "node"):
        self._record = record
        self._timeout = timeout
        self._seq = 0
        self._closed = False
        # Which driver child to spawn. Resolved BEFORE Popen so an unbuildable rust binary raises
        # the actionable SimBridgeBinaryError here rather than surfacing as a dead child later.
        self.impl = impl
        self._argv = search_driver_spawn_argv(impl)
        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._q: "queue.Queue[str]" = queue.Queue()
        # Bounded: a crashing child can dump a huge stderr (a serialized battle / heap); keep only the
        # tail (more than enough for _err_tail's last 2000 chars) so a crash can't spike RAM.
        self._stderr: "collections.deque[str]" = collections.deque(maxlen=4000)
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._errreader = threading.Thread(target=self._pump_stderr, daemon=True)
        self._reader.start()
        self._errreader.start()

    # -- process plumbing ---------------------------------------------------

    def _pump_stdout(self) -> None:
        try:
            for line in self._proc.stdout:        # type: ignore[union-attr]
                self._q.put(line)
        except Exception:
            pass
        finally:
            self._q.put("")                       # sentinel: stream closed

    def _pump_stderr(self) -> None:
        try:
            for line in self._proc.stderr:        # type: ignore[union-attr]
                self._stderr.append(line)
        except Exception:
            pass

    def _err_tail(self) -> str:
        return "".join(self._stderr)[-2000:]

    def _who(self) -> str:
        """How the child is named in an error — the impl plus the actual argv[0], so a rust
        failure self-diagnoses ('which binary was that?') instead of reading like a node crash."""
        return f"search driver [{self.impl}: {self._argv[-1]}]"

    def _call(self, payload: dict) -> dict:
        if self._closed:
            raise SearchError("SearchSession is closed")
        self._seq += 1
        payload = {**payload, "id": self._seq}
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")   # type: ignore[union-attr]
            self._proc.stdin.flush()                             # type: ignore[union-attr]
        except (BrokenPipeError, ValueError) as e:
            raise SearchError(f"{self._who()} died: {e}\n{self._err_tail()}")
        try:
            line = self._q.get(timeout=self._timeout)
        except queue.Empty:
            self.close(kill=True)
            raise SearchError(
                f"{self._who()} timed out after {self._timeout}s on cmd "
                f"{payload.get('cmd')!r}\n{self._err_tail()}")
        if line == "":
            raise SearchError(
                f"{self._who()} closed its stream (rc={self._proc.poll()})\n{self._err_tail()}")
        out = json.loads(line)
        if out.get("id") != self._seq:
            raise SearchError(
                f"{self._who()}: response id {out.get('id')} != request id {self._seq} (desync)")
        if not out.get("ok"):
            raise SearchError(f"{self._who()}: {out.get('error')}")
        return out

    # -- API ----------------------------------------------------------------

    def open_root(self, turn: int, *, record: "Optional[ReconstructionRecord]" = None) -> RootView:
        """Reconstruct to the start of turn ``turn`` and snapshot it as the search root. ``record``
        targets a SPECIFIC battle on a reused session (else the one passed to ``__init__``); it also
        clears the driver's node cache, so a warm process serves many battles' searches in turn."""
        rec = record if record is not None else self._record
        if rec is None:
            raise SearchError("open_root needs a record (pass record= or construct with one)")
        out = self._call({"cmd": "open_root", "record": rec.to_dict(), "turn": int(turn)})
        return RootView(
            node_id=out["node_id"], requests=out["requests"],
            recorded_choices=out["recorded_choices"], pre_state=out["pre_state"],
            prefix_p1_chunks=out["prefix_p1_chunks"], prefix_p2_chunks=out["prefix_p2_chunks"])

    def expand_many(self, arms: Sequence[dict]) -> List[ExpandedNode]:
        """Expand N arms from their parent nodes in one round-trip. Each ``arm`` is a dict
        ``{node_id, p1_action, p2_action, seed, label, recorded_exact?, followup?}`` with the
        per-side action semantics of :func:`reconstruction.reroll_turn` (``"recorded"`` only
        reproduces the realized turn on the ROOT via ``recorded_exact``; off-root a
        ``"recorded"`` source falls through to the follow-up policy, so callers pass explicit
        choice strings for the opponent at interior plies)."""
        out = self._call({"cmd": "expand_many", "arms": [dict(a) for a in arms]})
        return [
            ExpandedNode(
                label=a.get("label"), node_id=a.get("node_id"), ended=bool(a.get("ended")),
                stuck=bool(a.get("stuck")), outcome=a.get("outcome") or {},
                requests=a.get("requests"), choices_used=a.get("choices_used") or {},
                p1_chunks=a.get("p1_chunks") or [], p2_chunks=a.get("p2_chunks") or [])
            for a in out["arms"]
        ]

    # -- teardown -----------------------------------------------------------

    def close(self, *, kill: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if not kill and self._proc.poll() is None:
                self._proc.stdin.write(json.dumps({"cmd": "close", "id": -1}) + "\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()                                                # type: ignore[union-attr]
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        for s in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                s.close()        # type: ignore[union-attr]
            except Exception:
                pass

    def __enter__(self) -> "SearchSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
