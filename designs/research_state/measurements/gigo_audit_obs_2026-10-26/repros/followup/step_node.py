"""Feed a script to the Node bridge ONE command at a time; print each command with the lines it produced (errors, turns, moves)."""
import base64, select, subprocess, sys, time
script = open(sys.argv[1]).read().splitlines()
lo, hi = int(sys.argv[2]), int(sys.argv[3])
p = subprocess.Popen(["node", "/home/goodlad/dev/gen3ai/.claude/worktrees/gigo-open-items/src/utils/bridge/local_sim_bridge.js"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
def drain(t=0.6):
    out = []
    end = time.time() + t
    while time.time() < end:
        r, _, _ = select.select([p.stdout], [], [], 0.05)
        if r:
            ln = p.stdout.readline()
            if not ln: break
            if ln[:3] in ("p1 ", "p2 "):
                out.append((ln[:2], base64.b64decode(ln[3:]).decode().split("\n")))
            else:
                out.append(("-", [ln.strip()[:100]]))
            end = time.time() + 0.15
    return out
for i, c in enumerate(script):
    p.stdin.write(c + "\n"); p.stdin.flush()
    o = drain(1.5 if i == 0 else 0.5)
    if lo <= i <= hi:
        print(f"[{i}] {c[:70]}")
        for side, lines in o:
            for l in lines:
                if l.startswith(("|error|", "|turn|", "|move|", "|cant|", "|-start|", "|-end|", "|switch|", "|-mustrecharge", "|faint|", "|request|")) or side == "-":
                    print("    %s %s" % (side, l[:110] if not l.startswith("|request") else l[:50]))
p.kill()
