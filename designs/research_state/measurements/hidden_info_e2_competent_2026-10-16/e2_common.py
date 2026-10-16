"""Shared constants of E2, the COMPETENT-OPPONENT state bank (`PREREGISTRATION.md`, this folder). Pure Python: importable
by the pin-side drivers, the HEAD-side world builder and the numpy-only read alike (no model or engine at module level).

Where things run (PREREGISTRATION §1-§2; `run_e2.sh` selects the checkout):
  * GAMES   at ``~/.cache/gen3ai/hib_e2/pin_<cp>`` (the pin commit + the Showdown submodule the team validator needs):
            the pin's own ``main.ops.eval_trace_gen`` (`gen_games.sh`);
  * BANK and PHASE 1 (the truth) at the UNMODIFIED vdp pin checkouts Tier 1 used (``~/.cache/gen3ai/vdp/pin_<cp>``);
  * WORLDS  built at the E2 HEAD (a detached checkout of the commit that ships this harness, ``~/.cache/gen3ai/hib_e2/head``);
  * every WORLD PLAYOUT at Tier 1's PAD pins (``~/.cache/gen3ai/hib_t1/pin_<cp>_pad``).

Archive: ``~/gen3ai_archive/hidden_info_e2/<cp>/`` (games, bank, rows, worlds, logs). This folder: the scripts, the committed
ids (`states_e2_<cp>.json`) and the read.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
T1 = MEAS / "hidden_info_tier1_2026-10-13"
PILOT = MEAS / "value_disc_pilot_2026-10-11"
BRC = MEAS / "hidden_info_by_reveal_count_2026-10-12"
MODELS = Path("/home/goodlad/dev/gen3ai/models")
ARCH = Path(os.environ.get("HIB_E2_ARCHIVE") or (Path.home() / "gen3ai_archive" / "hidden_info_e2"))
PILOT_ARCH = Path.home() / "gen3ai_archive" / "value_disc_pilot"
T1_ARCH = Path.home() / "gen3ai_archive" / "hidden_info_tier1"
HEAD_DIR = Path.home() / ".cache" / "gen3ai" / "hib_e2" / "head"

#: the two checkpoints (Tier 1's), each with its plain pin, its pad pin and the E2 game-generation checkout
CPS: Dict[str, dict] = {
    "end2": {"run": "rb_es_end2_s2001", "pin": "62085750baab1ca275d810fbe0bb5a960904e183",
             "pad": "67f3f7fa", "pad_dir": "/home/goodlad/.cache/gen3ai/hib_t1/pin_end2_pad",
             "pin_dir": "/home/goodlad/.cache/gen3ai/vdp/pin_end2",
             "gen_dir": "/home/goodlad/.cache/gen3ai/hib_e2/pin_end2",
             "steps": (10000009, 12000157, 14000347), "seeds": (202610161, 202610162, 202610163)},
    "prod": {"run": "rb_es_prod_s2001", "pin": "95d014faeeb4d699703264a19094afee6f4eafc2",
             "pad": "9a698801", "pad_dir": "/home/goodlad/.cache/gen3ai/hib_t1/pin_prod_pad",
             "pin_dir": "/home/goodlad/.cache/gen3ai/vdp/pin_prod",
             "gen_dir": "/home/goodlad/.cache/gen3ai/hib_e2/pin_prod",
             "steps": (10000002, 12000178, 14000168), "seeds": (202610164, 202610165, 202610166)},
}
ORDER = ("end2", "prod")                      # end2 first (today's production lineage)

# ---- §1.1 games ----
GAMES = 160
GEN_FLAGS = ("--games", str(GAMES), "--sentinels", "5", "--opponents", "random", "--n-envs", "16", "--threads", "2",
             "--torch-threads", "1")
EXTENSION_SEED_OFFSET = 1000                  # §1.3: the one pre-registered extension batch (not used unless a quintile
                                              # cannot be filled)

# ---- §1.2 capture ----
CAPTURE_SALT = "hib_e2_capture_v1:"
CAPTURE_CAPS = {"loss": 24, "draw": 12, "win": 12}

# ---- §1.3 draw ----
STATES_SALT = "hib_e2_states_v1:"
PER_QUINTILE = 120
PER_BATTLE = 2
EDGE_EPS = 1e-6

# ---- §2 arms (labels are directory names: E2 is not blinded; the read is fixed as code before any arm output exists)
ARMS: Dict[str, str] = {"F": "off", "F+M": "F+M", "F+U": "F+U", "O-M": "O-M", "O-U": "O-U"}
BUILD_ORDER = ("F", "F+U", "F+M", "O-M", "O-U")   # F first: F+U shares F's seen-mon screen (common random numbers)
J = 8
N_LEAF, N_TRUTH = 16, 24


def capture_key(battle_id: str) -> str:
    return hashlib.sha256((CAPTURE_SALT + battle_id).encode()).hexdigest()


def states_key(did: str) -> str:
    return hashlib.sha256((STATES_SALT + did).encode()).hexdigest()


def gen_run(cp: str, step: int) -> str:
    """The shadow run dir's NAME (a bank battle id begins with it)."""
    return f"e2_{cp}_{step}"


def gen_root(cp: str) -> Path:
    """The models-root the bank loader reads the generated shadow runs from."""
    return ARCH / cp / "gen"


def arm_dir(label: str) -> str:
    return label.replace("+", "p").replace("-", "m")


def cp_dir(cp: str) -> Path:
    d = ARCH / cp
    d.mkdir(parents=True, exist_ok=True)
    return d


def bank_path(cp: str) -> Path:
    return ARCH / cp / "bank"


def states_path(cp: str) -> Path:
    return HERE / f"states_e2_{cp}.json"


def worlds_dir(cp: str, label: str) -> Path:
    d = cp_dir(cp) / "worlds" / arm_dir(label)
    d.mkdir(parents=True, exist_ok=True)
    return d


def rows_dir(cp: str, label: str) -> Path:
    """``label`` an arm, or ``O`` (Phase 1, the truth)."""
    d = cp_dir(cp) / "rows" / ("O" if label == "O" else arm_dir(label))
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_row(f, row: dict) -> None:
    f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    f.flush()
    os.fsync(f.fileno())


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:          # a torn last line from a kill: ignored, re-done
                pass
    return out


def read_dir(d: Path, pattern: str = "*.jsonl") -> Dict[str, dict]:
    """{id: the LAST row with that id} over every matching file of ``d`` (a re-run row supersedes)."""
    out: Dict[str, dict] = {}
    if not d.exists():
        return out
    for p in sorted(d.glob(pattern)):
        for r in read_rows(p):
            out[r["id"]] = r
    return out


def e2_states(cp: str) -> Dict[str, dict]:
    p = states_path(cp)
    return json.loads(p.read_text())["states"] if p.exists() else {}


def shard(ids: Sequence[str], i: int, of: int) -> List[str]:
    return [x for k, x in enumerate(sorted(ids)) if k % of == i]


def add_paths() -> None:
    """The pilot's folder (its `common`, `make_states`, `run_playouts`, `run_det_playouts`), the by-reveal-count folder
    (`stage_facts`) and Tier 1's folder (`t1_stats`, `build_worlds`' helpers) on the path, after this folder."""
    for p in (T1, BRC, PILOT):
        if str(p) not in sys.path:
            sys.path.insert(1, str(p))
