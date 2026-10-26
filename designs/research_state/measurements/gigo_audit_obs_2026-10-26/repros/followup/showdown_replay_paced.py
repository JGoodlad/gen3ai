"""showdown_replay.py with PACED command feeding (each CHOOSE waits for the bridge to go idle), because a bulk feed
makes the Node bridge's no-progress reject-streak guard trip on scripts that carry refused hidden-trap switches (a
replay-tool artifact, not an engine divergence). Compares BOTH sides' |request| JSON and |error| lines.
Usage: showdown_replay_paced.py <games json> [first] [last] [profile]"""
import base64, json, select, subprocess, sys, time
from collections import Counter
from utils.paths import repo_path
from utils.rust_env import ffi

games = json.load(open(sys.argv[1]))["games"]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(games)
profile = sys.argv[4] if len(sys.argv) > 4 else "selfcheck"
lib = ffi.load(ffi.default_path(profile))
bridge = str(repo_path("src/utils/bridge/local_sim_bridge.js"))
IDLE = float(__import__("os").environ.get("REPLAY_IDLE_S", "0.12"))


def run_node(script):
    p = subprocess.Popen(["node", bridge], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    sides = {"p1": [], "p2": []}
    err = []

    def drain(wait_first):
        end = time.time() + wait_first
        while time.time() < end:
            r, _, _ = select.select([p.stdout], [], [], 0.02)
            if not r:
                continue
            ln = p.stdout.readline()
            if not ln:
                return False
            if ln[:3] in ("p1 ", "p2 "):
                sides[ln[:2]] += base64.b64decode(ln[3:]).decode().split("\n")
            elif ln.startswith("__ERR__"):
                err.append(base64.b64decode(ln[8:]).decode()[:200])
            end = time.time() + IDLE
        return True
    for i, c in enumerate(script.splitlines()):
        try:
            p.stdin.write(c + "\n"); p.stdin.flush()
        except BrokenPipeError:
            break
        drain(1.5 if i == 0 else IDLE)
    drain(0.5)
    p.kill()
    return sides, err


def canon(l):
    if l.startswith("|request|"):
        v = json.loads(l[9:]); v.pop("rqid", None)
        return ("req", json.dumps(v, sort_keys=True))
    return ("err", l)


res = Counter(); first = []
for gi in range(lo, min(hi, len(games))):
    script = games[gi]["fin"]["script"]
    t = ffi.trace(lib, script, commit="a", label="x")
    node, err = run_node(script)
    bad = None
    for s in ("p1", "p2"):
        rust = [canon(json.loads(l)["text"]) for l in t["records"][s].splitlines()[1:]
                if json.loads(l)["text"].startswith(("|request|", "|error|"))]
        nd = [canon(x) for x in node[s] if x.startswith(("|request|", "|error|"))]
        if rust != nd:
            k = next((i for i in range(min(len(rust), len(nd))) if rust[i] != nd[i]), min(len(rust), len(nd)))
            bad = (s, len(rust), len(nd), k, rust[k][1][:500] if k < len(rust) else None, nd[k][1][:500] if k < len(nd) else None)
            break
    if bad is None:
        res["identical"] += 1
    else:
        res["DIFFER"] += 1; first.append((gi, bad, err))
    print(f"game {gi}: {'ok' if bad is None else 'DIFFER'}", flush=True)
print(dict(res))
for gi, (s, la, lb, k, a, b), err in first:
    print(f"game {gi} side {s}: rust {la} node {lb}; first diff at #{k}\n  RUST: {a}\n  NODE: {b}\n  node err: {err}")
