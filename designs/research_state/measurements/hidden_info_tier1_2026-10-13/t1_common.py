"""Shared constants of the hidden-information binding study's TIER 1 (`designs/endstate/design_hidden_info_binding.md`
§6, §12, §12.13 Amendment 2, §15). Pure Python: importable by the HEAD-side world builder, the pin-side drivers and the
numpy-only readers alike (no model, no engine import at module level).

Where things run (README §2):
  * the WORLD BUILDER runs at THIS STUDY'S HEAD (`main.value_search.worlds`, the information-lever builder, honest by
    default, the engine's gender draw pad) — `run_head.sh`;
  * every PLAYOUT runs at the checkpoint's OWN pin (`run_pin.sh`): the truth (Phase 1) and the network-belief dump at
    the unmodified pin checkouts; every WORLD at a PAD pin = the pin + the code of `98146ddb` (the engine's gender draw
    pad, nothing else; a team with no `*` marker takes the pin's path exactly). The pad pins' engine source equals
    HEAD's except comments and the oracle-reveal encoder (byte-inert at `off`).

Archive: ``~/gen3ai_archive/hidden_info_tier1/<cp>/`` (bulk); this folder holds the scripts, the committed ids and the
summaries. The value-discrimination pilot's folder and archive are read-only inputs (its registered banks, Phase 1 rows,
its v2 Phase 2 rows `det2`).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
PILOT = MEAS / "value_disc_pilot_2026-10-11"
HIB = MEAS / "hidden_info_binding_2026-10-11"
BRC = MEAS / "hidden_info_by_reveal_count_2026-10-12"
MODELS = Path("/home/goodlad/dev/gen3ai/models")
ARCH = Path(os.environ.get("HIB_T1_ARCHIVE") or (Path.home() / "gen3ai_archive" / "hidden_info_tier1"))
PILOT_ARCH = Path.home() / "gen3ai_archive" / "value_disc_pilot"
HEAD_WT = Path("/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a70fa77a612561867")

#: the two closing-test checkpoints, each played at its OWN pin (registration §1.1, §12.13)
CPS: Dict[str, dict] = {
    "end2": {"run": "rb_es_end2_s2001", "pin": "62085750baab1ca275d810fbe0bb5a960904e183",
             "pad": "67f3f7fa", "pad_dir": "/home/goodlad/.cache/gen3ai/hib_t1/pin_end2_pad",
             "pin_dir": "/home/goodlad/.cache/gen3ai/vdp/pin_end2"},
    "prod": {"run": "rb_es_prod_s2001", "pin": "95d014faeeb4d699703264a19094afee6f4eafc2",
             "pad": "9a698801", "pad_dir": "/home/goodlad/.cache/gen3ai/hib_t1/pin_prod_pad",
             "pin_dir": "/home/goodlad/.cache/gen3ai/vdp/pin_prod"},
}

# ------------------------------------------------------------------------------------------------
# the seed plan (the pilot's, unchanged): truth.turn_seeds(id, 48): leaf 0..15, truth 16..39, critic 40..47
# ------------------------------------------------------------------------------------------------
N_LEAF, N_TRUTH = 16, 24
J = 8                              # worlds played per state and arm (world j, dice seed j)

# ------------------------------------------------------------------------------------------------
# STAGE axes (§12.13.1, frozen)
# ------------------------------------------------------------------------------------------------
K_BINS = ("1-2", "3", "4", "5", "6")
TURN_BINS = ("<=7", "8-14", "15-21", ">=22")
MAT_BINS = ("<=7", "8-9", "10-11", "12")


def k_bin(k: int) -> str:
    if not 1 <= int(k) <= 6:
        raise ValueError(f"reveal count {k} outside 1..6")
    return "1-2" if k <= 2 else str(int(k))


def turn_bin(t: int) -> str:
    t = int(t)
    return "<=7" if t <= 7 else "8-14" if t <= 14 else "15-21" if t <= 21 else ">=22"


def mat_bin(m: int) -> str:
    m = int(m)
    return "<=7" if m <= 7 else "8-9" if m <= 9 else "10-11" if m <= 11 else "12"


# ------------------------------------------------------------------------------------------------
# the k-stratified SUPPLEMENT (§12.13.2, frozen): states added per checkpoint and k bin
# ------------------------------------------------------------------------------------------------
SUPP_TARGET: Dict[str, Dict[str, int]] = {
    "end2": {"1-2": 7, "3": 61, "4": 52, "5": 49, "6": 31},     # +200
    "prod": {"1-2": 0, "3": 65, "4": 55, "5": 43, "6": 57},     # +220
}
SUPP_SALT = "hib_stage_supp_v1:"
SUPP_PER_BATTLE_PER_BIN = 1
SUPP_PER_BATTLE = 3


def supp_key(did: str) -> str:
    return hashlib.sha256((SUPP_SALT + did).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------
# the TIER-1 ARMS (§15 Tier 1 + §12.13.3). "states": "all" = registered + supplement; "registered" = the 600 only.
# kind: "spec" = `WorldSampler(reveal=spec)` (an information-lever preset); "fn" / "poolsets" / "pool" = this unit's
# prior builds (README §4). O (the truth) is Phase 1 and is not coded (it is the referee's own world).
# ------------------------------------------------------------------------------------------------
ARMS: Dict[str, dict] = {
    "F": {"kind": "spec", "spec": "off", "states": "all"},
    "F+M": {"kind": "spec", "spec": "F+M", "states": "all"},
    "O-M": {"kind": "spec", "spec": "O-M", "states": "all"},
    "F+U": {"kind": "spec", "spec": "F+U", "states": "all"},
    "O-U": {"kind": "spec", "spec": "O-U", "states": "all"},
    "FN": {"kind": "fn", "spec": "off", "states": "all"},
    "D50": {"kind": "spec", "spec": "D50", "states": "registered"},
    "D90": {"kind": "spec", "spec": "D90", "states": "registered"},
    "W90": {"kind": "spec", "spec": "W90", "states": "registered"},
    "MAP100": {"kind": "spec", "spec": "MAP100", "states": "registered"},
    "POOLSETS": {"kind": "poolsets", "spec": "off", "states": "registered"},
    "POOL": {"kind": "pool", "spec": "off", "states": "registered"},
}
#: the build order inside one state: F first (its seen-mon screen is shared by every arm whose seen-mon constraint is
#: F's: D50, D90, W90, MAP100, F+U, FN — common random numbers for their paired contrasts), then the rest
BUILD_ORDER = ("F", "D50", "D90", "W90", "MAP100", "F+U", "FN", "F+M", "O-M", "O-U", "POOLSETS", "POOL")
ARM_SALT = "hib_arm_v1:"
CODE_SCHEME = "code = 'T' + the 1-based rank (2 digits) of sha256('hib_arm_v1:' + label) among the Tier-1 labels"


def arm_codes() -> Dict[str, str]:
    """{code: label} (§12.10: code = the rank of sha256("hib_arm_v1:" + label))."""
    labs = sorted((hashlib.sha256((ARM_SALT + lab).encode()).hexdigest(), lab) for lab in ARMS)
    return {f"T{i + 1:02d}": lab for i, (_, lab) in enumerate(labs)}


def key_path() -> Path:
    """The SEALED key lives in the archive until every arm output exists (its sha256 is committed beside the
    scripts, `arm_key.sha256`); it is then committed as `arm_key.json`."""
    return ARCH / "arm_key.json"


def key_bytes() -> bytes:
    codes = arm_codes()
    doc = {"salt": ARM_SALT, "scheme": CODE_SCHEME,
           "codes": {c: {"label": lab, **ARMS[lab]} for c, lab in codes.items()}}
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()


def load_key() -> Dict[str, str]:
    """{label: code} from the sealed key file (refuses a key whose bytes differ from the scheme's)."""
    p = key_path()
    b = p.read_bytes()
    if b != key_bytes():
        raise SystemExit(f"{p}: the sealed key does not match the registered scheme")
    doc = json.loads(b)
    return {v["label"]: c for c, v in doc["codes"].items()}


def arm_states(label: str) -> str:
    return ARMS[label]["states"]


# ------------------------------------------------------------------------------------------------
# paths and durable rows
# ------------------------------------------------------------------------------------------------

def cp_dir(cp: str) -> Path:
    d = ARCH / cp
    d.mkdir(parents=True, exist_ok=True)
    return d


def worlds_dir(cp: str, code: str) -> Path:
    d = cp_dir(cp) / "worlds" / code
    d.mkdir(parents=True, exist_ok=True)
    return d


def rows_dir(cp: str, code: str) -> Path:
    d = cp_dir(cp) / "rows" / code
    d.mkdir(parents=True, exist_ok=True)
    return d


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


# ------------------------------------------------------------------------------------------------
# the state sets
# ------------------------------------------------------------------------------------------------

def registered_states(cp: str) -> dict:
    """The pilot's registered 600 (`states_<cp>.json`): {id: extra}."""
    return json.loads((PILOT / f"states_{cp}.json").read_text())["states"]


def supplement_states(cp: str) -> dict:
    """The supplement's per-state extras (`states_supp_<cp>.json`, written by `make_supp_bank.py`): {id: extra}."""
    p = HERE / f"states_supp_{cp}.json"
    return json.loads(p.read_text())["states"] if p.exists() else {}


def bank_path(cp: str, which: str) -> Path:
    """``registered``: the pilot's bank (read-only); ``supp``: this unit's supplement bank."""
    return (PILOT_ARCH / cp / "bank") if which == "registered" else (cp_dir(cp) / "bank_supp")


def shard(ids: Sequence[str], i: int, of: int) -> List[str]:
    return [x for k, x in enumerate(sorted(ids)) if k % of == i]


def pick_first(xs, pred) -> Optional[object]:
    return next((x for x in xs if pred(x)), None)
