"""The byte-differential gate for :mod:`utils.bridge.ws_frontend`.

THE QUESTION THIS ANSWERS. The front end puts a websocket, a room header and an `rqid` between the
sim and a third-party client. Every one of those is a place where the game a client sees could
quietly stop being the game the sim played — and the metamon/foul-play de-risks both recorded that
the expensive failure is not a crash but a SILENTLY DIFFERENT battle. So the gate is the port's own
shape, one layer out: play a SEEDED battle through the front end, then replay the exact command
stream that battle produced into the Node `local_sim_bridge.js`, and assert the per-side protocol
TEXT is byte-identical.

WHY A REPLAY AND NOT TWO LIVE RUNS. A live A/B would have to assume both engines segment decisions
identically — the artifact `gen_sim_bridge_diff.js`'s docstring calls out. Here the decisions are
already DISCOVERED (the front end recorded what its clients actually chose, in order), so replaying
them removes that degree of freedom entirely: one command stream, two engines, one diff.

WHAT IS NORMALIZED, AND WHY EACH IS LEGITIMATE — there are exactly two, and nothing else:

* ``|t:|<anything>`` → ``|t:|``. Wall-clock lines; the ONE documented protocol exception poke-env
  ignores, normalized by the crate's own protocol gate for the same reason. The whole payload is
  dropped, not just an epoch: Node emits `|t:|1789430503` while the Rust port emits the literal
  `|t:|<NORMALIZED>`, and a regex that only matched digits reported that as the gate's first
  divergence on every battle.
* ``,"rqid":N`` is stripped from a ``|request|`` line. That is the front end's deliberate
  injection, and the REAL server injects it too (`server/room-battle.ts:796-801`) — so a Node
  bridge that never saw a server cannot have it. The splice is appended at the end of the object,
  so removing it restores the sim's own bytes and any OTHER byte difference inside the request
  still fails the gate.

A `|request|` count and the rqid MONOTONICITY are asserted separately, because stripping a field
must never be able to hide the field being wrong.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.bridge.sim_bridge_bin import bridge_spawn_argv

# The same 16 MiB stdout line budget every bridge reader uses — see ws_frontend.BRIDGE_STREAM_LIMIT.
from utils.bridge.ws_frontend import BRIDGE_STREAM_LIMIT

_T_RE = re.compile(r"^\|t:\|.*$", re.M)
_RQID_RE = re.compile(r',"rqid":\d+\}$')

# How long a replay may go without producing a stdout line before we call it wedged. A replayed
# battle emits continuously; this is a wedge detector, not a duration cap (the project rule: a
# timeout is never a semantic outcome, so a stall is reported as INCONCLUSIVE by the caller).
REPLAY_IDLE_BUDGET_S = 60.0

# How long the child must be SILENT before the next recorded command is written. See the
# pacing hazard in `replay_through`: without it the child exits having emitted nothing.
REPLAY_SETTLE_S = 0.02


@dataclass
class BattleCapture:
    """One battle's complete repro: what was sent to the child, and what each client received."""

    tag: str
    seed: Optional[List[int]]
    #: every stdin line the front end wrote to its bridge child, in order (`START …`, `CHOOSE …`).
    commands: List[str] = field(default_factory=list)
    #: every per-side chunk the front end relayed, as ``(slot, text)``, in emission order.
    chunks: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"tag": self.tag, "seed": self.seed, "commands": self.commands,
                "chunks": [list(c) for c in self.chunks]}

    @classmethod
    def from_dict(cls, raw: dict) -> "BattleCapture":
        return cls(tag=raw["tag"], seed=raw.get("seed"), commands=list(raw["commands"]),
                   chunks=[(s, t) for s, t in raw["chunks"]])

    @property
    def is_seeded(self) -> bool:
        """A capture is only REPLAYABLE if its ``START`` pinned the dice.

        A seedless ``START`` makes the child mint its own seed, so a replay runs a different
        battle — and the gate would then report a divergence that is not one. Refuse instead.
        """
        for cmd in self.commands:
            if cmd.startswith("START "):
                return "seed" in json.loads(cmd[len("START "):])
        return False


def side_text(chunks: List[Tuple[str, str]]) -> Dict[str, str]:
    """Fold a chunk list into ONE normalized protocol string per side.

    Chunk boundaries are deliberately NOT part of the comparison: they are a scheduler artifact
    of whichever child produced them (the crate harness makes the same call), while "the protocol
    text this client received" is the thing a client's parser actually consumes.
    """
    out: Dict[str, List[str]] = {"p1": [], "p2": []}
    for slot, text in chunks:
        out.setdefault(slot, []).append(text)
    return {slot: normalize("\n".join(parts)) for slot, parts in out.items()}


def normalize(text: str) -> str:
    """Apply the two legitimate normalizations (see the module docstring)."""
    text = _T_RE.sub("|t:|", text)
    return "\n".join(
        (f"|request|{_RQID_RE.sub('}', ln[len('|request|'):])}"
         if ln.startswith("|request|") else ln)
        for ln in text.split("\n"))


def request_rqids(chunks: List[Tuple[str, str]]) -> Dict[str, List[int]]:
    """Every ``rqid`` the front end handed each side, in order — the injection's own check."""
    out: Dict[str, List[int]] = {"p1": [], "p2": []}
    for slot, text in chunks:
        for ln in text.split("\n"):
            if ln.startswith("|request|"):
                match = _RQID_RE.search(ln)
                out.setdefault(slot, []).append(int(match.group(0)[len(',"rqid":'):-1])
                                               if match else -1)
    return out


async def replay_through(commands: List[str], impl: str = "node") -> List[Tuple[str, str]]:
    """Feed a recorded command stream to a fresh bridge child and collect its per-side chunks.

    ``impl="node"`` is the REFERENCE side of the gate — `local_sim_bridge.js`, the transport the
    whole project was validated against before the port existed.

    🚨 **THE WRITES MUST BE PACED, AND A BLAST PRODUCES SILENCE RATHER THAN AN ERROR.** Node's
    stdin `data` handler runs `handleLine` SYNCHRONOUSLY for every line in the chunk it received,
    while the per-side pumps that produce output are `for await` loops that have not run yet. Send
    the whole stream at once and the child reaches the recorded ``END`` — `exitWhenDrained(0)` —
    before a single protocol chunk has been written, and exits having emitted NOTHING. Measured
    here on 2026-09-14: 0 chunks for a 106-command battle, no error, no stderr. So each command is
    followed by a QUIESCENCE settle (the same lever `gen_sim_bridge_diff.js` uses), and a recorded
    ``END`` is dropped — it is a teardown verb, not part of the battle.
    """
    battle_cmds = [c for c in commands if c.strip() != "END"]
    proc = await asyncio.create_subprocess_exec(
        *bridge_spawn_argv(impl),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, limit=BRIDGE_STREAM_LIMIT)
    chunks: List[Tuple[str, str]] = []
    ended = asyncio.Event()
    errors: List[str] = []
    stderr_lines: List[str] = []
    last_line_at = [asyncio.get_running_loop().time()]

    async def reader() -> None:
        while True:
            line = await proc.stdout.readline()
            last_line_at[0] = asyncio.get_running_loop().time()
            if not line:
                break
            text = line.decode().rstrip("\n")
            if text == "__END__":
                ended.set()
                break
            if text.startswith("__ERR__"):
                errors.append(base64.b64decode(text[len("__ERR__ "):]).decode("utf-8"))
                ended.set()
                break
            if text.startswith("__RECON__"):
                continue
            slot, b64 = text.split(" ", 1)
            chunks.append((slot, base64.b64decode(b64).decode("utf-8")))

    async def drain_stderr() -> None:
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            stderr_lines.append(line.decode(errors="replace").rstrip())

    async def settle() -> None:
        """Block until the child has been silent for ``REPLAY_SETTLE_S``."""
        loop = asyncio.get_running_loop()
        while True:
            quiet = loop.time() - last_line_at[0]
            if quiet >= REPLAY_SETTLE_S:
                return
            await asyncio.sleep(REPLAY_SETTLE_S - quiet)

    task = asyncio.ensure_future(reader())
    err_task = asyncio.ensure_future(drain_stderr())
    try:
        for cmd in battle_cmds:
            if ended.is_set() or task.done():
                break
            proc.stdin.write((cmd + "\n").encode())
            await proc.stdin.drain()
            await settle()
        # The battle ends on its own once the last recorded choice lands; `END` before that would
        # cut the child mid-flush. Wait for `__END__` under an IDLE bound rather than a duration
        # cap, then tear down.
        await asyncio.wait_for(ended.wait(), timeout=REPLAY_IDLE_BUDGET_S)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"bridge replay ({impl}) produced no __END__ within {REPLAY_IDLE_BUDGET_S}s after "
            f"{len(battle_cmds)} commands and {len(chunks)} chunks — the replay is INCONCLUSIVE, "
            f"not a divergence. child stderr: {stderr_lines[:3]}") from exc
    finally:
        if proc.returncode is None:
            try:
                proc.stdin.write(b"END\n")
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:  # pragma: no cover
                proc.kill()
                await proc.wait()
        # Close the pipes here rather than leaving them to `__del__`, which can run after the
        # loop is closed and then raises `RuntimeError: Event loop is closed` out of a destructor.
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            transport.close()
        task.cancel()
        err_task.cancel()
    if errors:
        raise RuntimeError(f"bridge replay ({impl}) failed: {errors[0]}")
    return chunks


def first_divergence(front: Dict[str, str], reference: Dict[str, str]) -> Optional[str]:
    """``None`` when every side matches, else a message naming the FIRST differing line."""
    for slot in sorted(set(front) | set(reference)):
        a = front.get(slot, "").split("\n")
        b = reference.get(slot, "").split("\n")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return (f"{slot} line {i} differs:\n  front end: {x!r}\n  node      : {y!r}")
        if len(a) != len(b):
            longer, which = (a, "front end") if len(a) > len(b) else (b, "node")
            return (f"{slot}: {which} emitted {abs(len(a) - len(b))} extra line(s); "
                    f"first extra: {longer[min(len(a), len(b))]!r}")
    return None


async def check_capture(capture: BattleCapture, *, reference_impl: str = "node") -> Optional[str]:
    """Run the whole gate for one battle. ``None`` = byte-identical.

    Raises rather than returning a divergence when the capture cannot be judged at all (an
    unseeded battle, a wedged replay) — an inconclusive run must never read as a pass OR a fail.
    """
    if not capture.is_seeded:
        raise ValueError(
            f"{capture.tag} was played WITHOUT a pinned seed, so a replay is a different battle "
            "and the comparison is meaningless. Start the front end with --seed-base.")
    reference = await replay_through(capture.commands, impl=reference_impl)
    rqids = request_rqids(capture.chunks)
    flat = [r for slot in ("p1", "p2") for r in rqids.get(slot, [])]
    if not flat:
        raise ValueError(f"{capture.tag} relayed no |request| at all — nothing could have moved.")
    if any(r < 1 for r in flat):
        return f"{capture.tag}: a |request| reached a client with no rqid injected"
    merged = sorted(flat)
    if merged != list(range(1, len(merged) + 1)):
        return (f"{capture.tag}: rqids are not the server's single 1..N sequence across both "
                f"slots — got {merged[:12]}…")
    return first_divergence(side_text(capture.chunks), side_text(reference))
