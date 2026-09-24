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

🚨 **The per-side ``p1_chunks`` / ``p2_chunks`` an expand returns are THAT ARM'S OWN PLY —
its one-sided SUFFIX, not the view from the root.** The materializer wants
``prefix + every ply on the path``, so a caller that deepens MUST accumulate:
``parent_chunks + e.p1_chunks``, the way :mod:`main.prober.better_line` threads
``our_suffixes``. This docstring used to promise "the COMPLETE one-sided view (root → that
node)" and the search-dividend deepener believed it, which is how a depth-2 replay came to be
fed a protocol with plies 1..d-1 missing (``gen3_search_depth2_chunk_gap_v1`` — see
:class:`main.search_dividend.deepen.TreeNode`'s ``chunks``). At depth 1 the two readings
coincide, so nothing caught it for as long as depth 1 was all that ran.

``pre_state`` / ``outcome`` are OMNISCIENT (referee view): they drive only the opponent model
and the dice, never the obs encoder (the one-sided / omniscient wall, identical to the re-roll
path).

``view_p1`` / ``view_p2`` are the OTHER side of that wall (`gen3_one_sided_view_v1`, rust only):
the same board PROJECTED onto what each side has observed, in the shape
:class:`~agents.battle.live_view.LiveView` holds, so a successor's read-models can be built
without replaying its protocol. :mod:`agents.battle.view_adapter` is the constructor and
``designs/rust_sim/one_sided_view.md`` is the contract. They are ``{}`` under ``impl="node"``.

``view_p1_at`` / ``view_p2_at`` are the ordered boards at the decisions an arm resolved INSIDE
itself (`gen3_view_at_intermediate_v1`, deferral D10) — a faint's replacement round is a second
request in the same arm, and ``view_pN`` describes the board after it rather than at it.

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
import sys
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from utils.bridge.reconstruction import ReconstructionRecord
from utils.bridge.sim_bridge_bin import search_driver_spawn_argv
from utils.contention import describe_contention, scale_timeout

# gen3_contention_robust_timeouts_v1 — how long to wait for the search-driver child to REAP after
# it has been sent ``close``. A cooperative exit is milliseconds of work, so 5 s is enormous
# headroom on an idle box; it is still a wall-clock bound on a subprocess, so on a loaded box it
# measures the box rather than the child. Read at CALL time (a session opened idle can easily be
# closed beside a trainer), and ``scale_timeout`` rather than a ``ProgressDeadline`` because a
# ``wait()`` exposes no incremental progress to bound.
_CLOSE_REAP_TIMEOUT = 5.0


def _close_reap_timeout() -> float:
    """The post-``close`` child-reap bound, stretched to the CPU share actually available."""
    return scale_timeout(_CLOSE_REAP_TIMEOUT)


@dataclass(frozen=True)
class RootView:
    """The decision-point root of a search (turn-T board, both requests open)."""

    node_id: str
    requests: dict            # {"p1": request-json, "p2": ...} — the choice surfaces
    recorded_choices: dict    # {"p1": str|None, "p2": str|None} — the original turn-T picks
    pre_state: dict           # omniscient board snapshot (referee view)
    prefix_p1_chunks: List[str]
    prefix_p2_chunks: List[str]
    # The ONE-SIDED VIEW of this board per side (`gen3_one_sided_view_v1`) — the projection of
    # the same state onto what that side has OBSERVED, in the shape `LiveView` holds. Unlike
    # `pre_state` (omniscient, referee-only) these ARE obs-legal: they are the other side of the
    # wall, and `agents.battle.view_adapter` builds the read-models straight from one. `{}` from
    # a driver that predates the field (node's `search_driver.js` emits none).
    view_p1: dict = field(default_factory=dict)
    view_p2: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExpandedNode:
    """One expanded arm: the child node + THIS PLY's one-sided chunks + outcome.

    ``p1_chunks`` / ``p2_chunks`` are the arm's own turn — a SUFFIX. Composing a materializable
    protocol is the caller's job and is ``prefix + <every ply from the root>``; see the module
    header."""

    label: object
    node_id: Optional[str]    # None when the turn ENDED the battle (terminal — no child)
    ended: bool
    stuck: bool
    outcome: dict             # omniscient (referee view)
    requests: Optional[dict]  # the child's open requests (None when ended)
    choices_used: dict
    p1_chunks: List[str]      # THIS PLY's one-sided suffix — NOT root → this node
    p2_chunks: List[str]
    # The arm's RESULTING board, projected per side (`gen3_one_sided_view_v1`). Obs-legal, unlike
    # `outcome`; `{}` under `impl="node"`, which emits no such field.
    view_p1: dict = field(default_factory=dict)
    view_p2: dict = field(default_factory=dict)
    # The boards at the decisions this ply resolved INSIDE itself, in order, per side
    # (`gen3_view_at_intermediate_v1`, deferral D10). A ply that KOs one of our mons opens a
    # SECOND request inside the same arm and the port answers it from its own follow-up policy,
    # so `view_pN` above is one decision PAST the row a per-request consumer wants; entry `k`
    # here is the board at that side's `k`-th non-final request of the ply. EMPTY on the
    # ordinary arm, and empty under `impl="node"` and on a `recorded_exact` arm — a consumer
    # that finds no entry falls back exactly as it did before the field existed.
    view_p1_at: List[dict] = field(default_factory=list)
    view_p2_at: List[dict] = field(default_factory=list)
    # `materializer=core` (`gen3_core_search_v1`, rust only): the arm's LEAF as a Rust-core
    # version — `{view, legal, request, events, mid, text_view?}` per asked-for side, the view being
    # the core's `present()` of the side's own stream (every poke-env reading rule applied in
    # Rust; `agents.battle.core_view` builds the read-models from it with no rule), `events` the
    # ply's readings (the tracker input), `mid` whether the leaf is an intermediate (D10)
    # decision, and `text_view` the same leaf folded the OTHER way when the arm was integrity-
    # checked. `None` on a non-core node.
    core_p1: Optional[dict] = None
    core_p2: Optional[dict] = None


class SearchError(RuntimeError):
    pass


class ElidedSide(dict):
    """The payload of a side the request DECLINED to ask for — falsy, and LOUD on any real read.

    ``gen3_expand_many_side_elision_v1``. When :meth:`SearchSession.expand_many` is given a
    ``side``, the rust driver omits the other side's ``view_pN`` / ``pN_chunks`` entirely (43.0%
    of the reply bytes on the banked search decisions). What lands in the ``ExpandedNode`` in
    their place is one of these rather than ``{}`` / ``[]``, because the two failure modes are not
    the same failure: an empty board ENCODES — into a well-formed observation of a battle nobody
    played — while this raises.

    It is a ``dict`` subclass with ``__len__`` == 0 on purpose, so the existing
    ``payload or {}`` guards in :mod:`main.search_dividend.search` still read it as "no payload"
    and take their COUNTED fallback (``view_fallback_no_payload``). Falsy is the recoverable
    answer; every way of actually reading a value out of it is the unrecoverable one."""

    __slots__ = ("_field",)

    def __init__(self, field_name: str) -> None:
        super().__init__()
        object.__setattr__(self, "_field", field_name)

    def _refuse(self, *_a, **_k):
        raise SearchError(
            f"{self._field} was ELIDED: this expand_many asked for one side only. Re-issue the "
            f"call without side= (or with the other side) if you need it — reading an elided "
            f"payload as empty would encode a board nobody played.")

    __getitem__ = _refuse
    get = _refuse
    keys = _refuse
    items = _refuse
    values = _refuse
    __iter__ = _refuse
    __contains__ = _refuse

    def __repr__(self) -> str:
        return f"<ElidedSide {self._field}>"


def _side_payload_list(arm: dict, key: str, elided: bool):
    """:func:`_side_payload` for a field the caller wants as a fresh ``list``.

    The copy is taken ONLY on a present value: calling ``list()`` on an :class:`ElidedSide` would
    raise here, at construction, rather than at the read that actually wanted the side — and an
    error that fires where nobody asked for anything is a worse diagnostic than the one the
    sentinel exists to give."""
    if key in arm:
        return list(arm[key] or [])
    return ElidedSide(key) if elided else []


def _side_payload(arm: dict, key: str, empty, elided: bool):
    """The arm's value for ``key``: the driver's when PRESENT, else empty-or-refusing.

    The distinction is on the KEY, never on the request: ``search_driver.js`` emits no ``view_pN``
    at all, so under ``impl="node"`` the absence means "this driver has none" and the historical
    ``{}`` is right. Only a ``side``-bearing request can turn an absence into a refusal, and only
    for the side it declined."""
    if key in arm:
        return arm[key] or empty
    return ElidedSide(key) if elided else empty


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

    def open_root(self, turn: int, *, record: "Optional[ReconstructionRecord]" = None,
                  core: Optional[str] = None, side: Optional[str] = None,
                  trackers: bool = False) -> RootView:
        """Reconstruct to the start of turn ``turn`` and snapshot it as the search root. ``record``
        targets a SPECIFIC battle on a reused session (else the one passed to ``__init__``); it also
        clears the driver's node cache, so a warm process serves many battles' searches in turn.

        ``core`` (``"typed"`` / ``"text"``, rust only — ``gen3_core_search_v1``) builds the tree of
        Rust-core BattleVersions instead of bare sessions, each successor folded TYPED at the
        source or from the side's TEXT; every arm expanded from it then carries ``core_pN``.
        ``side`` (core only) folds that side's stream alone, at every version of the tree.
        ``trackers`` (typed core only, ``gen3_core_trackers_v1``) folds the per-decision TRACKERS on
        every version too — read by nothing yet but the fork-cost measurement."""
        rec = record if record is not None else self._record
        if rec is None:
            raise SearchError("open_root needs a record (pass record= or construct with one)")
        req: dict = {"cmd": "open_root", "record": rec.to_dict(), "turn": int(turn)}
        if core is not None:
            if core not in ("typed", "text"):
                raise SearchError(f"open_root: core must be 'typed' or 'text', got {core!r}")
            req["core"] = core
            if side is not None:
                if side not in ("p1", "p2"):
                    raise SearchError(f"open_root: side must be 'p1' or 'p2', got {side!r}")
                req["side"] = side
            if trackers:
                req["trackers"] = True
        out = self._call(req)
        return RootView(
            node_id=out["node_id"], requests=out["requests"],
            recorded_choices=out["recorded_choices"], pre_state=out["pre_state"],
            prefix_p1_chunks=out["prefix_p1_chunks"], prefix_p2_chunks=out["prefix_p2_chunks"],
            view_p1=out.get("view_p1") or {}, view_p2=out.get("view_p2") or {})

    def expand_many(self, arms: Sequence[dict], *,
                    side: Optional[str] = None, integrity: int = 0) -> List[ExpandedNode]:
        """Expand N arms from their parent nodes in one round-trip. Each ``arm`` is a dict
        ``{node_id, p1_action, p2_action, seed, label, recorded_exact?, followup?}`` with the
        per-side action semantics of :func:`reconstruction.reroll_turn` (``"recorded"`` only
        reproduces the realized turn on the ROOT via ``recorded_exact``; off-root a
        ``"recorded"`` source falls through to the follow-up policy, so callers pass explicit
        choice strings for the opponent at interior plies).

        🚨 ``seed="original"`` IS NOT A DICE SAMPLE — it is the REALIZED stream. The driver swaps
        the PRNG only for a non-``"original"`` seed (``if (!isOriginal) b.prng = new PRNG(seed)``)
        and ``open_root`` replays the record to the start of the turn, so the arm resolves from the
        battle's own mid-game PRNG state: measured 2026-08-24 over 12 consecutive live decisions,
        expanding the realized ``(p1_action, p2_action)`` pair that way reproduced the real turn's
        our-side protocol BYTE-FOR-BYTE 11 of 12 times, against 14 of 36 for fresh seeds. That is
        exactly what an OFFLINE counterfactual wants (a CRN anchor against what really happened),
        and exactly what a LIVE search must not have — reading it is a ply of clairvoyance no
        player has, and it silently made the search-dividend probe's dice-width ladder measure its
        own dilution. In a live decision, mint every seed.

        ``side="p1"``/``"p2"`` (``gen3_expand_many_side_elision_v1``, **rust only**) asks the
        driver to SKIP the other side's ``view_pN`` / ``pN_chunks``. A search runs for ONE side
        and reads exactly those two fields for it; the other side's copy measured **43.0% of the
        reply bytes** — rendered, piped and ``json.loads``-ed only to be dropped
        (``designs/research_state/measurements/expand_many_2026-09-22/README.md``). The requested
        side's payload is BYTE-IDENTICAL either way. An elided side comes back as an
        :class:`ElidedSide`: falsy, so the existing ``payload or {}`` guards take their COUNTED
        fallback, but RAISING on any read, so a caller that wanted it fails loudly instead of
        encoding an empty board. ``search_driver.js`` ignores the key and still returns both
        sides — the sentinel keys on the field being ABSENT, never on the request, so nothing is
        elided under ``impl="node"``."""
        req: dict = {"cmd": "expand_many", "arms": [dict(a) for a in arms]}
        if side is not None:
            if side not in ("p1", "p2"):
                raise SearchError(f"expand_many: side must be 'p1' or 'p2', got {side!r}")
            req["side"] = side
        # INTEGRITY (core nodes, `gen3_core_search_v1`): every Nth core arm is folded both ways
        # (typed + text) and the driver FAILS the call on any disagreement; 0 = off.
        if integrity:
            req["integrity"] = int(integrity)
        out = self._call(req)
        return [
            ExpandedNode(
                label=a.get("label"), node_id=a.get("node_id"), ended=bool(a.get("ended")),
                stuck=bool(a.get("stuck")), outcome=a.get("outcome") or {},
                requests=a.get("requests"), choices_used=a.get("choices_used") or {},
                p1_chunks=_side_payload(a, "p1_chunks", [], side == "p2"),
                p2_chunks=_side_payload(a, "p2_chunks", [], side == "p1"),
                view_p1=_side_payload(a, "view_p1", {}, side == "p2"),
                view_p2=_side_payload(a, "view_p2", {}, side == "p1"),
                # D10's per-intermediate-decision boards are ONE-SIDED payloads on the same
                # terms, so they elide with their own side. `list(...)` only where the driver
                # actually sent one — an `ElidedSide` must reach the dataclass INTACT, since
                # `list()` on it would raise here instead of where a consumer reads it.
                view_p1_at=_side_payload_list(a, "view_p1_at", side == "p2"),
                view_p2_at=_side_payload_list(a, "view_p2_at", side == "p1"),
                core_p1=a.get("core_p1"), core_p2=a.get("core_p2"))
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
        reap_budget = _close_reap_timeout()
        try:
            self._proc.wait(timeout=reap_budget)
        except subprocess.TimeoutExpired:
            # Self-diagnosing, per the project timeout rule: this kill used to be silent, so a
            # STARVED reap was indistinguishable from a WEDGED child.
            print(
                f"⚠️  [{self._who()}] did not exit within {reap_budget:.1f}s of close "
                f"(base {_CLOSE_REAP_TIMEOUT:.1f}s x contention scale) — killing it. "
                f"{describe_contention()}",
                file=sys.stderr,
                flush=True,
            )
            self._proc.kill()
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
