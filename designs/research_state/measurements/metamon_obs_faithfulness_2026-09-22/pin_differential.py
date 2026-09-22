#!/usr/bin/env python3
"""CHECK 4 — would Metamon have seen a DIFFERENT battle on its OWN Showdown pin?

The question the decision-level check really asks is whether Metamon's observation of a battle
changes when the SERVER changes. Running its policy on two different battles would answer nothing
(different battles produce different actions for reasons that have nothing to do with parsing), so
the comparison has to be the SAME battle on two pins. A ``ws_frontend`` capture makes that
possible: ``commands`` is every stdin line the front end wrote to its bridge child — a seeded
``START`` plus every committed ``CHOOSE`` — and replaying it regenerates the battle exactly.

So: replay each captured battle's commands through ``local_sim_bridge.js`` twice, once against
**our** pinned submodule (``e0551883f``) and once against **Metamon's bundled** one
(``d62d3a398``, 13 commits ahead), and compare the per-side protocol text.

🚨 **WHY A BYTE COMPARISON SETTLES THE ARGMAX QUESTION.** Metamon's policy is a deterministic
function of the protocol stream it receives (greedy — ``Agent.get_actions(sample=False)``, verified
per decision at ``argmax_match_rate = 1.0000``). Identical input therefore gives identical argmax
at every decision, and the disagreement count is 0 *by construction* rather than by sampling 200
decisions and hoping. Where the streams DIFFER, the differing battles are named so the policy can
be run on both — a byte difference is where the check starts, not where it stops.

The two legitimate normalizations (``ws_frontend_replay.normalize``) are applied: the ``|t:|``
wall-clock timestamp, and the ``rqid`` counter, neither of which is battle state.
"""

import argparse
import asyncio
import glob
import json
import os
import sys

REPO = "/home/goodlad/dev/gen3ai"
sys.path.insert(0, os.path.join(REPO, "src"))

from utils.bridge.ws_frontend_replay import normalize, side_text  # noqa: E402

BRIDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pin_differential_bridge.js")
SETTLE_S = 0.02
IDLE_BUDGET_S = 120.0


async def replay(commands, ps_path):
    """The captured command stream through one Showdown checkout; per-side chunks back.

    The pacing is not optional — see ``ws_frontend_replay.replay_through``: a blasted stream makes
    the child reach ``END`` before a single protocol chunk has been written, and it exits having
    emitted NOTHING, with no error and no stderr.
    """
    import base64

    env = dict(os.environ, GEN3AI_PS_PATH=ps_path)
    proc = await asyncio.create_subprocess_exec(
        "node", BRIDGE, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=env, limit=1 << 24)
    chunks, errors = [], []
    ended = asyncio.Event()
    loop = asyncio.get_running_loop()
    last = [loop.time()]

    async def reader():
        while True:
            line = await proc.stdout.readline()
            last[0] = loop.time()
            if not line:
                break
            text = line.decode().rstrip("\n")
            if text == "__END__":
                ended.set()
                break
            if text.startswith("__ERR__"):
                errors.append(base64.b64decode(text[len("__ERR__ "):]).decode())
                ended.set()
                break
            if text.startswith("__RECON__"):
                continue
            slot, b64 = text.split(" ", 1)
            chunks.append((slot, base64.b64decode(b64).decode()))

    async def settle():
        while True:
            quiet = loop.time() - last[0]
            if quiet >= SETTLE_S:
                return
            await asyncio.sleep(SETTLE_S - quiet)

    task = asyncio.ensure_future(reader())
    drain = asyncio.ensure_future(proc.stderr.read())
    try:
        for cmd in [c for c in commands if c.strip() != "END"]:
            if ended.is_set() or task.done():
                break
            proc.stdin.write((cmd + "\n").encode())
            await proc.stdin.drain()
            await settle()
        try:
            await asyncio.wait_for(ended.wait(), timeout=IDLE_BUDGET_S)
        except asyncio.TimeoutError:
            errors.append("no __END__ within the idle budget")
    finally:
        if proc.returncode is None:
            try:
                proc.stdin.write(b"END\n")
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        task.cancel()
        await drain
    return chunks, errors


async def main_async(args):
    captures = sorted(glob.glob(os.path.join(args.captures, "*.json")))[: args.limit]
    if not captures:
        raise SystemExit(f"no captures under {args.captures} — refusing a vacuous run")
    rows = []
    for path in captures:
        capture = json.load(open(path))
        ours, e1 = await replay(capture["commands"], args.ours_ps)
        theirs, e2 = await replay(capture["commands"], args.theirs_ps)
        a, b = side_text(ours), side_text(theirs)
        slots = sorted(set(a) | set(b))
        identical = all(a.get(s) == b.get(s) for s in slots)
        row = {"tag": capture["tag"], "identical": identical,
               "errors_ours": e1, "errors_theirs": e2,
               "n_chunks_ours": len(ours), "n_chunks_theirs": len(theirs),
               "live_stream_matches_ours": normalize(
                   "\n".join(t for s, t in capture["chunks"] if s == "p1")) == a.get("p1")}
        if not identical:
            for slot in slots:
                x, y = a.get(slot, ""), b.get(slot, "")
                if x != y:
                    for i, (lx, ly) in enumerate(zip(x.split("\n"), y.split("\n"))):
                        if lx != ly:
                            row["first_divergence"] = {"slot": slot, "line": i,
                                                       "ours": lx, "theirs": ly}
                            break
                    break
        rows.append(row)
        print(json.dumps(row), flush=True)

    n_err = sum(1 for r in rows if r["errors_ours"] or r["errors_theirs"])
    summary = {
        "n_battles": len(rows),
        "n_identical": sum(1 for r in rows if r["identical"]),
        "n_divergent": sum(1 for r in rows if not r["identical"]),
        "n_replay_errors": n_err,
        # 🚨 The control: our OWN replay must reproduce the live capture, or a "no divergence"
        # verdict would only mean the replay harness produced nothing on either pin.
        "n_reproducing_the_live_stream": sum(1 for r in rows if r["live_stream_matches_ours"]),
        "divergences": [r for r in rows if not r["identical"]],
    }
    if summary["n_reproducing_the_live_stream"] == 0:
        raise SystemExit("the replay reproduced the live stream 0 times — this run proves "
                         "NOTHING about the two pins and must not be read as agreement")
    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=1)
    print(json.dumps(summary, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--ours-ps", default=os.path.join(REPO, "deps/pokemon-showdown"))
    ap.add_argument("--theirs-ps", default="/home/goodlad/dev/metamon/server/pokemon-showdown")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", required=True)
    asyncio.run(main_async(ap.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
