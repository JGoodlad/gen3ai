"""Shared constants of the OFF-POOL BANK (``PREREGISTRATION.md``). Pure Python at import (no torch, no ``main``), so
the battle generator, the bank draw, the scoring driver and the numpy-only reader all import the same numbers.

Where things run: every driver that touches the engine, the world builder or the network runs from a DETACHED
checkout of L3's code commit (:data:`CODE_SHA`, ``--code``), the commit L3 and its decoy counter-test ran on. This
folder's scripts are loaded by path; the ``src`` they import is the checkout's (``-I``, :func:`use_code`).

Archive (``$OFFPOOL_ARCHIVE``, default ``~/gen3ai_archive/search_offpool_bank``)::

    ckpt/                          read-only copies of P* and the five sentinels (sha256-checked)
    traces/cyc<k>/eval_traces/step_280000170/<opponent>/   the generated cycles (shadow run dirs)
    bank/, census.npz              the frozen bank (states_offpool.json in this folder holds its ids)
    decoy_corpus.json              the off-pool decoy corpus (packed, HP-repaired hl_05_26 teams)
    rows.jsonl, recheck.jsonl      the scoring rows
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
ARCH = Path(os.environ.get("OFFPOOL_ARCHIVE") or (Path.home() / "gen3ai_archive" / "search_offpool_bank"))
MAIN = Path("/home/goodlad/dev/gen3ai")

#: L3's code commit (L3 README; the decoy counter-test ran on it too)
CODE_SHA = "8fe4fb899"
#: X26's pin: the commit X26 trained at, start to end (``metadata.json`` ``pin_history``)
X26_PIN = "9c0334890d05785bbfadcb2ecbe3fe2d548efe9b"
RUN = "rb_x26_s2601"

# ---- the acting model and the anchor (PREREGISTRATION §1) ----
PSTAR_STEP = 280000170
MODEL_CONFIG_SHA = "df71eea63809d630629e59455cf599ebb202b50b3cbcd3e795a65d06b30e801c"   # the run's, = maturity's
PSTAR_SHA = "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"
#: the plateau cycle's five sentinels, in the live cycle's order (eval_results.jsonl at 280000170): sentinel_<i>
SENTINELS = [
    (268000171, "558edce3644a8a6cc5aa2f45b07df88eef8780631e0bf623c4b6e6e00c662400"),
    (236000155, "34f5f483c084d0558915837b71a7b6b10d6728b4dafca3889468113093fe763f"),
    (194000195, "587e07fdc830cd2567e215b8560c4591189d351ac8ecc46a83459c09002d30ec"),
    (160000085, "fde1749ebdaa354b96ae3d9492daef77c843855382b24c1f079650dbe145bd7d"),
    (134000128, "958c6fa306371287b56180cbf63bfc8b7a3011440e73ced3a86f71447a00879d"),
]

# ---- the opponents' teams (PREREGISTRATION §2) ----
HELDOUT_DIR = MEAS / "heldout72_hpfix_kakuna_2026-10-18" / "teams_heldout72_hpfix"
HELDOUT_MANIFEST = MEAS / "heldout72_hpfix_kakuna_2026-10-18" / "heldout72_hpfix_manifest.json"
HL_SOURCE = Path.home() / "gen3ai_archive" / "metamon_cache_2026-09-24" / "teams" / "hl_05_26" / "gen3ou"
COMPETITIVE_DIR = Path.home() / "dev" / "metamon" / "cache" / "teams" / "competitive" / "gen3ou"
POOL_PIN = Path.home() / "gen3ai_archive" / "world_sampler_stage1" / "pool" / "pool_pin.json"

# ---- the generated cycles (PREREGISTRATION §3) ----
#: a cycle is played in three PARTS, each one heavy-slot chunk, so every forward is batched: the nine bots, the first
#: three sentinels, the last two. Every game is a pure function of (the part's seed, the opponent's label, the game index).
PARTS = {"bots": None, "sentA": (0, 3), "sentB": (3, 5)}          # sentinel index ranges into SENTINELS
GAMES = {"bots": 18, "sentA": 30, "sentB": 30}                    # per opponent: 162 bot and 150 sentinel battles
CYCLES = ("cyc1",)                          # the cycle the bank is drawn from
EXTENSION = "cyc2"                          # the ONE pre-registered extension (a short cell): the same parts, new seeds
SEEDS = {("cyc1", "bots"): 2026102011, ("cyc1", "sentA"): 2026102012, ("cyc1", "sentB"): 2026102013,
         ("cyc2", "bots"): 2026102021, ("cyc2", "sentA"): 2026102022, ("cyc2", "sentB"): 2026102023}
SMOKE_SEED = 2026102099                     # the engineering smokes' cycles: never enter the bank
BOTS = ("random", "heuristic", "heuristic2", "staller", "staller_v2", "aggressive", "aggressive_v2", "setup_sweep",
        "setup_sweep_v2")

# ---- the draw (the maturity battery's rule; PREREGISTRATION §4) ----
STRATA = ("snapshot", "bot")
PER_CELL = 32                               # LEAN: 64 per k bin, 320 in all (L3's bank size before exclusions)
PER_BATTLE_PER_BIN = 1
PER_BATTLE = 3
STATES_SALT = "search_offpool_bank_v1:"

# ---- the scoring (PREREGISTRATION §5–§6) ----
J = 8
MAX_TRIES = 60
D = 4
MAX_GATED = 1000
NEAR = 4
DECOY_SALT = "search_offpool_decoy_v1:"
RECHECK_SALT = "search_offpool_recheck_v1:"
N_RECHECK = 10


def d(*parts) -> Path:
    p = ARCH.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def states_path(smoke: bool = False) -> Path:
    """The bank's ids (this folder), or the engineering smoke's (the archive: never a registered bank)."""
    return (ARCH / "smoke" / "states_smoke.json") if smoke else (HERE / "states_offpool.json")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def ids_sha(ids) -> str:
    """sha256 of the sorted ids concatenated (``maturity_bank.ids_sha``; L3's G1)."""
    return hashlib.sha256("".join(sorted(ids)).encode()).hexdigest()


def heldout_texts() -> Dict[str, str]:
    """The 72 heldout72_hpfix team texts by id, each sha256-checked against its manifest."""
    man = json.loads(HELDOUT_MANIFEST.read_text())
    out = {}
    for tid, rec in sorted(man["teams"].items()):
        b = (HELDOUT_DIR / f"{tid}.txt").read_bytes()
        if hashlib.sha256(b).hexdigest() != rec["sha256"]:
            raise SystemExit(f"{tid}: sha256 differs from heldout72_hpfix_manifest.json")
        out[tid] = b.decode()
    if len(out) != 72:
        raise SystemExit(f"heldout72_hpfix holds {len(out)} teams, not 72")
    return out


def use_code(code: Path) -> Path:
    """Put the detached checkout's ``src`` (and the measurement folders this unit borrows) first on ``sys.path`` and
    refuse a wrong commit or a ``main`` package imported from elsewhere."""
    code = Path(code).resolve()
    head = os.popen(f"git -C {code} rev-parse HEAD").read().strip()
    if not head.startswith(CODE_SHA):
        raise SystemExit(f"{code} is at {head[:9]}, not L3's code commit {CODE_SHA}")
    meas = code / "designs" / "research_state" / "measurements"
    for p in (meas / "hidden_info_tier1_2026-10-13", meas / "world_sampler_stage1_2026-10-19",
              meas / "search_l3_2026-10-20", code / "src"):
        sys.path.insert(0, str(p))
    import main as _m

    if not str(Path(_m.__file__).resolve()).startswith(str(code / "src")):
        raise SystemExit(f"main imported from {_m.__file__}, not {code}: run with python -I")
    return code


def stratum(opp: str) -> str:
    if opp.startswith("sentinel_"):
        return "snapshot"
    if opp in BOTS:
        return "bot"
    raise SystemExit(f"{opp}: neither stratum")
