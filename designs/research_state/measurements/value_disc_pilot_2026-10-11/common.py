"""Shared constants and helpers of the value-discrimination pilot (design_value_discrimination.md §7).

Runs ONLY under ``run_at_pin.sh`` (PYTHONPATH = the checkpoint's own pin checkout, CUDA hidden).
Every path under ``models/`` is opened READ-ONLY; every write goes to the archive or this folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = Path("/home/goodlad/dev/gen3ai/models")
ARCHIVE = Path.home() / "gen3ai_archive" / "value_disc_pilot"

#: the two checkpoints (each at its OWN pin; run_at_pin.sh checks the pin checkout's HEAD)
CPS: Dict[str, dict] = {
    "end2": {"run": "rb_es_end2_s2001", "pin": "62085750baab1ca275d810fbe0bb5a960904e183"},
    "prod": {"run": "rb_es_prod_s2001", "pin": "95d014faeeb4d699703264a19094afee6f4eafc2"},
}

#: the seed plan (registered in README §3 before any outcome was read): ONE nested list per decision,
#: ``truth.turn_seeds(id, 48)``. Leaf seeds 0..15, truth seeds 16..39 (24), critic one-ply seeds 40..47.
N_LEAF = 16
N_TRUTH = 24
N_PLAY = N_LEAF + N_TRUTH          # 40 seeds played to the end per legal action
CRIT_SEEDS = (40, 48)              # the critic's one-ply successors (never played to the end)
SAMPLE_SALT = "value_disc_pilot_2026:"
PER_QUINTILE = 120
PER_BATTLE = 2
EDGE_EPS = 1e-6                    # rule 8: a state within this many logits of a quintile edge is excluded


def cp() -> str:
    c = os.environ.get("VDP_CP")
    if c not in CPS:
        raise SystemExit("run through run_at_pin.sh (VDP_CP unset)")
    return c


def ckpt_path(c: str) -> Path:
    return MODELS / CPS[c]["run"] / "final_model.zip"


def out_dir(c: str) -> Path:
    d = ARCHIVE / c
    d.mkdir(parents=True, exist_ok=True)
    return d


def check_pin(c: str) -> None:
    sha = os.environ.get("VDP_PIN_SHA", "")
    if sha != CPS[c]["pin"]:
        raise SystemExit(f"pin checkout HEAD {sha!r} != {CPS[c]['pin']} for {c}")
    import main  # noqa: F401  (the pin's src must be first on the path)

    if not str(Path(main.__file__).resolve()).startswith(os.environ["VDP_PIN"]):
        raise SystemExit(f"main imported from {main.__file__}, not the pin checkout")


def threads() -> int:
    return int(os.environ.get("VDP_THREADS", "4"))


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def sample_key(did: str) -> str:
    return hashlib.sha256((SAMPLE_SALT + did).encode()).hexdigest()


def load_model(c: str):
    """CP on CPU through the strict loader; refuses a non-winprob critic or extra obs keys."""
    from main.policy_spectrum.qhat import check_winprob
    from main.policy_spectrum.reader import load_checkpoint

    m = load_checkpoint(ckpt_path(c), device="cpu")
    check_winprob(m)
    return m


def clock_offset() -> int:
    """The obs offset of the CLOCK group's ``remaining_linear`` scalar (turn = 250·(1 − x)): the
    global-env block's start (``OFFSET_GLOBAL``) + the clock's offset in it + 1."""
    from agents.observation.constants import CLOCK_OFFSET_IN_GLOBAL, OFFSET_GLOBAL

    return int(OFFSET_GLOBAL) + int(CLOCK_OFFSET_IN_GLOBAL) + 1


def decode_turn(rows: np.ndarray, off: int) -> np.ndarray:
    from agents.observation.constants import MAX_TURNS

    return np.rint(MAX_TURNS * (1.0 - rows[:, off].astype(np.float64))).astype(np.int64)


def write_row(f, row: dict) -> None:
    f.write(json.dumps(row, sort_keys=True) + "\n")
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
            except json.JSONDecodeError:      # a torn last line from a kill: ignored, re-done
                pass
    return out
