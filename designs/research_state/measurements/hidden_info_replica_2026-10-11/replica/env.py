"""Put the PIN checkout first on the import path and name its binaries. Import this before anything that imports
the project (``agents``, ``main``, ``utils``): the replication runs every project call at the checkpoint's pin."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PINS = {"end2": "62085750baab1ca275d810fbe0bb5a960904e183", "prod": "95d014faeeb4d699703264a19094afee6f4eafc2"}
RUNS = {"end2": "rb_es_end2_s2001", "prod": "rb_es_prod_s2001"}
PIN_ROOT = Path(os.path.expanduser("~/.cache/gen3ai/vdp"))
ARCHIVE = Path(os.path.expanduser("~/gen3ai_archive/value_disc_pilot"))
MODELS = Path("/home/goodlad/dev/gen3ai/models")
HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[2]
STATES_DIR = REPO / "research_state" / "measurements" / "value_disc_pilot_2026-10-11"
if not STATES_DIR.exists():
    STATES_DIR = HERE.parent / "value_disc_pilot_2026-10-11"


def setup(cp: str) -> Path:
    src = PIN_ROOT / f"pin_{cp}" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.environ["POKESIM_CORE_EVENTS_BIN"] = str(src / "rust_sim" / "target" / "release" / "core_events")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return src


def lib_path(cp: str) -> Path:
    return PIN_ROOT / f"pin_{cp}" / "src" / "rust_env" / "target" / "release" / "libpokesim_env.so"


def checkpoint(cp: str) -> Path:
    return MODELS / RUNS[cp] / "final_model.zip"


def registered_states(cp: str) -> dict:
    d = json.loads((STATES_DIR / f"states_{cp}.json").read_text())
    if d["pin"] != PINS[cp]:
        raise RuntimeError(f"states_{cp}.json pin {d['pin']} != {PINS[cp]}")
    return d["states"]
