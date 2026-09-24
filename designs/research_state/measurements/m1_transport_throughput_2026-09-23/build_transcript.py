"""Fold the per-child stdin captures of `tee_sim_bridge.sh` into ONE persistent-mode transcript.

Each capture is one battle's `START {json}` + its `CHOOSE ...` lines, in the order the Python
bridge sent them. The START json gets `"persistent": true` (one `sim_bridge` process then plays
every battle in sequence, resetting at each `__END__`), and a final `END` closes the process.
A late CHOOSE after a battle's `__END__` is a documented no-op in `sim_bridge`, so the stream
replays deterministically: the battles carry their pinned seeds.

    python build_transcript.py <capture_dir> <out.txt>
"""

from __future__ import annotations

import json
import os
import sys


def main(cap_dir, out_path):
    files = sorted(f for f in os.listdir(cap_dir) if f.endswith(".in"))
    n_start = n_choose = 0
    with open(out_path, "w") as out:
        for name in files:
            for line in open(os.path.join(cap_dir, name)):
                line = line.rstrip("\n")
                if line.startswith("START "):
                    spec = json.loads(line[len("START "):])
                    if "seed" not in spec:
                        raise SystemExit(f"{name}: unseeded START — the replay would not be pinned")
                    spec["persistent"] = True
                    out.write("START " + json.dumps(spec, separators=(",", ":")) + "\n")
                    n_start += 1
                elif line.startswith("CHOOSE "):
                    out.write(line + "\n")
                    n_choose += 1
                elif line.strip() in ("", "END"):
                    continue
                else:
                    out.write(line + "\n")
        out.write("END\n")
    print(f"{len(files)} captures -> {n_start} STARTs, {n_choose} CHOOSEs -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
