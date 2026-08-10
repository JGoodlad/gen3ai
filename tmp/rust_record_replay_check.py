"""Can the RUST bridge replay a NODE-recorded command stream byte-for-byte?

That is precisely what a Rust `buildToTurn` would do (drive a fresh session from the
record's `>start` + `>player` lines, then feed `record.commands`), so this is the
load-bearing precondition for a Rust search driver. Scratch tool (tmp/).

    python tmp/rust_record_replay_check.py [tmp/search_golden_node.json]

Replays each golden case's record through BOTH `local_sim_bridge.js` and the rust
`sim_bridge` binary and diffs the per-side chunk streams (|t:| normalized — the port's
one documented emission exception).
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from pathlib import Path

from utils.bridge.sim_bridge_bin import bridge_spawn_argv

T_LINE = re.compile(r"^\|t:\|.*$", re.M)


def norm(chunk: str) -> str:
    return T_LINE.sub("|t:|<N>", chunk)


def start_payload(record: dict) -> dict:
    start = next(l for l in record["input_log"] if l.startswith(">start "))
    opts = json.loads(start[len(">start "):])
    players = [l for l in record["input_log"] if l.startswith(">player ")]
    out = {"formatid": opts["formatid"], "seed": opts.get("seed")}
    for line in players:
        rest = line[len(">player "):]
        side, blob = rest.split(" ", 1)
        out[side] = json.loads(blob)
    return out


def run(argv, record: dict, max_cmds: int | None = None):
    """Drive one bridge child through the record and collect its per-side chunks.

    stdin is kept OPEN until `__END__` arrives: the node child is async and an early
    EOF makes it exit before the battle resolves (that mistake reported 0 node chunks).
    """
    import queue as _q
    import threading

    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    chunks = {"p1": [], "p2": []}
    errs = []
    done = threading.Event()
    q: _q.Queue = _q.Queue()

    def reader():
        for line in p.stdout:
            line = line.rstrip("\n")
            if line.startswith("p1 ") or line.startswith("p2 "):
                tag, b64 = line.split(" ", 1)
                chunks[tag].append(base64.b64decode(b64).decode("utf-8", "replace"))
            elif line.startswith("__ERR__ "):
                errs.append(base64.b64decode(line.split(" ", 1)[1]).decode("utf-8", "replace"))
            elif line == "__END__":
                done.set()
        q.put(None)

    errbuf = []
    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=lambda: errbuf.append(p.stderr.read()), daemon=True).start()

    p.stdin.write("START " + json.dumps(start_payload(record)) + "\n")
    p.stdin.flush()
    cmds = record["commands"]
    if max_cmds is not None:
        cmds = cmds[:max_cmds]
    for side, choice in cmds:
        if done.is_set():
            break
        line = (f"FORCELOSE {choice}" if side == "forcelose"
                else f"CHOOSE {side} {choice}")
        p.stdin.write(line + "\n")
        p.stdin.flush()
    done.wait(timeout=120)
    try:
        p.stdin.write("END\n")
        p.stdin.flush()
        p.stdin.close()
    except Exception:
        pass
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()
    return chunks, errs, "".join(x for x in errbuf if x)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tmp/search_golden_node.json"
    cases = json.load(open(path))["cases"]
    seen = set()
    records = []
    for c in cases:
        key = json.dumps(c["record"]["input_log"][:1]) + str(len(c["record"]["commands"]))
        if key not in seen:
            seen.add(key)
            records.append(c["record"])
    print(f"{len(records)} distinct records")

    node_argv = ["node", str(Path("src/utils/bridge/local_sim_bridge.js").resolve())]
    rust_argv = bridge_spawn_argv("rust")
    ok = True
    for i, rec in enumerate(records):
        nc, ne, nerr = run(node_argv, rec)
        rc, re_, rerr = run(rust_argv, rec)
        for side in ("p1", "p2"):
            a = [norm(x) for x in nc[side]]
            b = [norm(x) for x in rc[side]]
            if a != b:
                ok = False
                print(f"record {i} {side}: NODE {len(a)} chunks vs RUST {len(b)} chunks")
                for k in range(max(len(a), len(b))):
                    x = a[k] if k < len(a) else "<missing>"
                    y = b[k] if k < len(b) else "<missing>"
                    if x != y:
                        print(f"  first diff at chunk {k}:")
                        print(f"   node: {x[:400]!r}")
                        print(f"   rust: {y[:400]!r}")
                        break
                break
        else:
            print(f"record {i}: MATCH ({len(nc['p1'])}/{len(nc['p2'])} chunks)")
        if ne or re_:
            print(f"  node errs={ne} rust errs={re_}")
            ok = False
        if rerr.strip():
            print(f"  rust stderr: {rerr.strip()[:300]}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
