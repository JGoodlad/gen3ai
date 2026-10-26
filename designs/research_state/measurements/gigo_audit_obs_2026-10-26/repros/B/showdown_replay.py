"""GIGO audit B: replay each finished game's INPUT LOG through the REAL Showdown (the Node bridge,
deps/pokemon-showdown dist) and compare p1's |request| JSON and |error| lines, in order, to the Rust
engine's (ffi.trace's p1 record). A request the engine gets wrong vs Showdown is the only way the
(request-pure) mask can forbid a legal action or allow an illegal one.

Usage: showdown_replay.py <games json> [max games]
"""
import base64
import json
import subprocess
import sys
from collections import Counter

from utils.paths import repo_path
from utils.rust_env import ffi

lib = ffi.load(ffi.default_path("release"))
games = json.load(open(sys.argv[1]))["games"]
lim = int(sys.argv[2]) if len(sys.argv) > 2 else len(games)
bridge = str(repo_path("src/utils/bridge/local_sim_bridge.js"))


def norm(line):
    return line


res = Counter()
first_div = []
for gi, g in enumerate(games[:lim]):
    script = g["fin"]["script"]
    t = ffi.trace(lib, script, commit="a", label="x")
    rust = [json.loads(l)["text"] for l in t["records"]["p1"].splitlines()[1:]]
    rust = [l for l in rust if l.startswith("|request|") or l.startswith("|error|")]
    p = subprocess.Popen(["node", bridge], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p.stdin.write(script); p.stdin.flush()
    out_lines = []
    import select, time
    t_end = time.time() + 120
    while time.time() < t_end:
        r, _, _ = select.select([p.stdout], [], [], 5)
        if not r:
            break
        ln = p.stdout.readline()
        if not ln or ln.startswith("__END__"):
            break
        out_lines.append(ln.rstrip("\n"))
    try:
        p.stdin.write("END\n"); p.stdin.close()
    except BrokenPipeError:
        pass
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
    p.stderr_text = p.stderr.read()
    node = []
    for ln in out_lines:
        if ln.startswith("p1 "):
            for x in base64.b64decode(ln[3:]).decode().split("\n"):
                if x.startswith("|request|") or x.startswith("|error|"):
                    node.append(x)
    # compare request JSON as parsed objects (key order is byte-gated elsewhere)
    def canon(l):
        if l.startswith("|request|"):
            v = json.loads(l[9:])
            v.pop("rqid", None)
            return ("req", json.dumps(v, sort_keys=True))
        return ("err", l)
    a, b = [canon(x) for x in rust], [canon(x) for x in node]
    if a == b:
        res["identical"] += 1
        continue
    res["DIFFER"] += 1
    k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    first_div.append((gi, len(a), len(b), k, a[k][1][:700] if k < len(a) else None, b[k][1][:700] if k < len(b) else None,
                      p.stderr_text[-300:]))
print(dict(res))
for d in first_div[:6]:
    gi, la, lb, k, ra, nb, err = d
    print(f"game {gi}: rust {la} lines, node {lb}; first difference at #{k}")
    print("  RUST:", ra)
    print("  NODE:", nb)
    if err.strip():
        print("  node stderr:", err)
