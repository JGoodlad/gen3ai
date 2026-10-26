"""Full p1 (or p2) stream diff, Rust engine (ffi.trace record) vs real Showdown (Node bridge), for one game.
Usage: diff_game.py <games json> <game index> [p1|p2] [context]"""
import base64, json, select, subprocess, sys, time
from utils.paths import repo_path
from utils.rust_env import ffi

lib = ffi.load(ffi.default_path("release"))
g = json.load(open(sys.argv[1]))["games"][int(sys.argv[2])]
side = sys.argv[3] if len(sys.argv) > 3 else "p1"
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 25
script = g["fin"]["script"]
t = ffi.trace(lib, script, commit="a", label="x")
rust = [json.loads(l)["text"] for l in t["records"][side].splitlines()[1:]]
open("/tmp/gigo_audit/B/_cur.script", "w").write(script)
bridge = str(repo_path("src/utils/bridge/local_sim_bridge.js"))
p = subprocess.run(["bash", "-c", f"(cat /tmp/gigo_audit/B/_cur.script; sleep 25) | timeout 40 node {bridge}"],
                   capture_output=True, text=True)
out = p.stdout.splitlines()
node = []
for ln in out:
    if ln.startswith(side + " "):
        node += base64.b64decode(ln[3:]).decode().split("\n")
    elif ln.startswith("__ERR__"):
        print("NODE ERR", base64.b64decode(ln[8:]).decode())


def keep(l):
    return l and not l.startswith("|t:|") and l != "|"


def canon(l):
    if l.startswith("|request|"):
        v = json.loads(l[9:]); v.pop("rqid", None)
        return "|request|" + json.dumps(v, sort_keys=True)
    return l


rust = [canon(l) for l in rust if keep(l)]
node = [canon(l) for l in node if keep(l)]
k = next((i for i in range(min(len(rust), len(node))) if rust[i] != node[i]), min(len(rust), len(node)))
print(f"rust {len(rust)} lines, node {len(node)}; first difference at {k}")
for i in range(max(0, k - ctx), k):
    print("   ", rust[i][:220])
print("RUST>", "\n      ".join(x[:400] for x in rust[k:k + 6]))
print("NODE>", "\n      ".join(x[:400] for x in node[k:k + 6]))
print("script commands up to here:", sum(1 for _ in script.splitlines()))
