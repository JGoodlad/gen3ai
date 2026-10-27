"""Identity-unit queue at a pin (scratch, read-only on models): the pin's OWN main.x35.kakuna plan / run, with every
unit captured (home72 units too: --capture-dir changes no stamp and no play; the kakuna queue captures panel units only).
usage: identity_queue.py <state> <finals.json> <label:teamset:unit_seed> ...   (PYTHONPATH = the pin's src, cwd = pin)"""
import json
import sys
from pathlib import Path

import main.x35.kakuna as K

_orig = K.unit_argv


def _argv(u, out, port, py=sys.executable, *, short=True):
    a = _orig(u, out, port, py, short=short)
    return a if "--capture-dir" in a else a + ["--capture-dir", str(out / "capture")]


K.unit_argv = _argv
state, finals = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
state.mkdir(parents=True, exist_ok=True)
if not (state / "plan.json").exists():
    units = []
    for sel in sys.argv[3:]:
        lab, ts, s = sel.split(":")
        f = [x for x in finals if x["label"] == lab]
        units += K.plan_units(f, teamsets=(ts,), unit_seeds=(int(s),))
    (state / "plan.json").write_text(json.dumps({"units": [u.__dict__ for u in units]}, indent=1) + "\n")
    print("planned", [u.name for u in units], flush=True)
sys.exit(K.cmd_run(state, None))
