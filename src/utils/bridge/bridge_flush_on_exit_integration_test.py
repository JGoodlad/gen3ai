"""Regression gate for `gen3_bridge_flush_on_exit_v1` — the truncated-`__RECON__` bug.

`process.stdout.write()` to a PIPE is asynchronous in Node, and a bare `process.exit()`
discards whatever is still queued in Node userspace. The bridge used to emit the (large)
`__RECON__` record + `__END__` and then `process.exit(0)` — so whenever the Python reader was
slow to drain (run_20260807_135637's final eval at `--eval-concurrency 100`), the recon line
was cut mid-base64 and `_offer_recon` logged `Incorrect padding` / "cannot be 1 more than a
multiple of 4" — 224 times, every one in the final-eval window, zero during periodic evals
(whose one-battle-at-a-time workers drain promptly). The fix routes every exit through
`exitWhenDrained()` (a zero-length write's callback fires only after all queued writes reach
the kernel pipe, whose contents survive child exit).

This test reproduces the race DETERMINISTICALLY, no battle needed, via the `__ERR__` path —
the same `out()` + exit-immediately-after window with a payload whose size WE control:

1. spawn the bridge, send one giant (1 MB) unknown command — `fail()` emits a ~1.3 MB
   `__ERR__ <b64>` line and the process heads for exit(1);
2. deliberately DO NOT read stdout for a beat — the kernel pipe holds only ~64 KiB, so a
   pre-fix child (bare `process.exit`) tears down with most of the line undrained;
3. then drain to EOF and assert the base64 payload arrived COMPLETE.

On the pre-fix bridge this fails exactly like production did (truncated b64). On the fixed
bridge the child blocks in `exitWhenDrained` until we read, and the payload is intact.
"""
from __future__ import annotations

import base64
import os
import subprocess
import time

import pytest

_BRIDGE_JS = os.path.join(os.path.dirname(__file__), "local_sim_bridge.js")

pytestmark = pytest.mark.integration


def test_large_final_line_survives_child_exit():
    payload = "X" * 1_000_000  # one token, no spaces → echoed whole into the __ERR__ message
    proc = subprocess.Popen(
        ["node", _BRIDGE_JS],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        proc.stdin.write((payload + "\n").encode())
        proc.stdin.flush()
        # The race window: give the child every chance to exit WITHOUT us reading. A pre-fix
        # child is gone (poll() == 1) with ~64 KiB in the pipe; a fixed child is still alive,
        # blocked handing the rest of the line to the kernel.
        time.sleep(1.0)
        out, _ = proc.communicate(timeout=30)  # closes stdin, then drains stdout to EOF
    finally:
        if proc.poll() is None:
            proc.kill()

    lines = [ln for ln in out.decode().splitlines() if ln.startswith("__ERR__")]
    assert lines, f"no __ERR__ line arrived at all (got {out[:200]!r})"
    b64 = lines[0][len("__ERR__ "):]
    msg = base64.b64decode(b64).decode("utf-8")  # pre-fix: raises binascii.Error (padding)
    assert payload in msg, (
        f"__ERR__ arrived but the payload is incomplete ({len(msg)} chars) — "
        f"the drain-aware exit regressed"
    )


def test_reader_limit_admits_long_battle_recon_lines():
    """Part 2 of the same bug: once the drain-aware exit delivers long lines COMPLETE, the
    asyncio reader must accept them — the 64 KiB default readline limit turned a delivered
    1000-turn-battle `__RECON__` line into `LimitOverrunError` → a crashed battle (the
    intermittent `local_sim_bridge_integration_test` failures; training's 250-turn stall cap
    kept its lines under the old limit, which is why only long-battle dice tripped it).

    Deterministic: emit a 512 KiB single line through the SAME spawn shape the runner uses and
    read it with `readline()`. Fails with the stock 64 KiB limit; passes under
    `BRIDGE_STREAM_LIMIT`."""
    import asyncio

    from utils.bridge.local_battle_runner import BRIDGE_STREAM_LIMIT

    payload_chars = 512 * 1024
    script = f'process.stdout.write("__RECON__ " + "A".repeat({payload_chars}) + "\\n__END__\\n")'

    async def read_one_line(limit):
        proc = await asyncio.create_subprocess_exec(
            "node", "-e", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=limit,
        )
        try:
            line = await proc.stdout.readline()
        finally:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        return line

    # The fixed limit admits the whole line.
    line = asyncio.run(read_one_line(BRIDGE_STREAM_LIMIT))
    assert line.decode().rstrip("\n") == "__RECON__ " + "A" * payload_chars

    # And the stock default provably does NOT — the regression this test exists to pin.
    with pytest.raises(ValueError):
        asyncio.run(read_one_line(2 ** 16))
