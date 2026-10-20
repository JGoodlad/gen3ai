"""Shared constants of the WORLD SAMPLER's STAGE 1 (``designs/endstate/design_world_sampler.md`` §6.2;
``PREREGISTRATION.md`` in this folder). Pure Python at import: importable by the builder side (HEAD + this unit's code),
the play side (X26's pin + the damage-roll control) and the numpy-only reader alike.

WHERE THINGS RUN (``run_s1.sh`` selects the checkout and refuses a wrong HEAD):
  * builder  ``$WS1_BUILDER`` (default ``~/.cache/gen3ai/ws_stage1/builder``: a DETACHED checkout of the commit that
             ships the launch): every arm's WORLDS (``build_s1.py``), with the builder at that commit (F2's legality
             rule, F1 tier A's ``dice="rolls"``, the Hidden Power fill of 2026-10-19);
  * pin      ``~/.cache/gen3ai/ws_stage1/pin_rolls`` = X26's pin ``9c033489`` + ONE cherry-pick, ``fc010d7d3`` (the
             Rust core's damage-roll control, ``gen3_roll_control_v1``; its Rust files are byte-identical to HEAD's and
             it changes nothing unarmed): every PLAYOUT (``play_s1.py``), so a ``dice="rolls"`` world replays the rolls
             its gate accepted (a world rebuilt from its team alone loses them) and every other arm plays exactly as at
             the pin (checked: ``identity_s1.py`` re-plays banked F worlds and must reproduce the banked F rows).

READ-ONLY INPUTS (the maturity battery's archive, ``~/gen3ai_archive/hidden_info_maturity``; its plateau step is
running and appends there: torn lines are skipped and nothing is written there): the interim bank, the 225M anchor's
checkpoint copy, roots, truth rows (O), the banked F / FN world rows and their 225M playouts, the 225M belief rows.

ARCHIVE (``$WS1_ARCHIVE``, default ``~/gen3ai_archive/world_sampler_stage1``)::

    worlds/<arm>/worlds.jsonl     one row per (state, arm): the builder's worlds (+ rolls / log_w for F_rolls)
    rows/<arm>/rows.jsonl         one row per (state, arm): the pilot's det-format playout row at X26 225M
    identity/rows.jsonl           the identity tooth's re-plays of banked F worlds
    logs/                         the queue's and each unit's logs
    STOP_AT, QUEUE_DONE, queue.pid
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
REPO = HERE.parents[3]
MAT_DIR = MEAS / "hidden_info_maturity_x26_2026-10-18"
T1 = MEAS / "hidden_info_tier1_2026-10-13"
PILOT = MEAS / "value_disc_pilot_2026-10-11"
BRC = MEAS / "hidden_info_by_reveal_count_2026-10-12"
STAGE0 = MEAS / "world_sampler_screen_2026-10-19"

MAT_ARCH = Path(os.environ.get("HIB_MAT_ARCHIVE") or (Path.home() / "gen3ai_archive" / "hidden_info_maturity"))
ARCH = Path(os.environ.get("WS1_ARCHIVE") or (Path.home() / "gen3ai_archive" / "world_sampler_stage1"))
CACHE = Path.home() / ".cache" / "gen3ai" / "ws_stage1"

#: the anchor (design §6.2): X26 at 225M, the last LEAN anchor of the maturity battery
ANCHOR = "x26_225M"
#: X26's pin and the play checkout = that pin + the damage-roll control (fc010d7d3) cherry-picked
X26_PIN = "9c0334890d05785bbfadcb2ecbe3fe2d548efe9b"
ROLL_CONTROL_COMMIT = "fc010d7d387c683c08d7a2ec0fb2ecba52cce217"
#: the sha256 of "".join(sorted ids) of the 287 common states (the maturity LEAN read's common set, its README §5.1)
COMMON_IDS_SHA256 = "cf8d8c6fc09c9a85ce976a85a6195eb52977900659ac9be218b47562d9a09dfb"
N_STATES = 287

J = 8
MAX_TRIES = 60

#: THE ARMS, in BUILD order per state (the screen cache is shared: F_legal first, every sibling takes F_legal's
#: seen-mon screen; F_rolls keys its own by the dice mode). PREREGISTRATION §3.
ARMS = ("F_legal", "F_off", "B", "C_ladder", "C_forum_strict", "NS", "C_pool", "F_rolls")
PRIMARY = ("B", "C_ladder", "C_forum_strict", "C_pool", "NS")
#: the banked arms of the maturity battery at 225M (read, never re-run)
BANKED = ("O", "F", "FN")

ORDER_SALT = "ws_stage1_order_v1:"


def arm_dir(arm: str) -> str:
    return arm.replace("+", "p").replace("-", "m")


def d(*parts) -> Path:
    p = ARCH.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def worlds_path(arm: str) -> Path:
    return d("worlds", arm_dir(arm)) / "worlds.jsonl"


def rows_path(arm: str) -> Path:
    return d("rows", arm_dir(arm)) / "rows.jsonl"


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:          # a torn last line (a kill, or a file still being written)
                    pass
    return out


def read_by_id(path: Path, first: bool = False) -> Dict[str, dict]:
    """{id: the LAST (or FIRST) row with that id}."""
    out: Dict[str, dict] = {}
    for r in read_rows(path):
        if first and r["id"] in out:
            continue
        out[r["id"]] = r
    return out


def write_row(f, row: dict) -> None:
    """One fsync'd JSONL row, stamped with the launch's driver commit when the queue set one."""
    drv = os.environ.get("WS1_DRIVER_SHA")
    if drv:
        row = {**row, "driver": drv}
    f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    f.flush()
    os.fsync(f.fileno())


def mat_states() -> Dict[str, dict]:
    return json.loads((MAT_DIR / "states_mat_x26.json").read_text())["states"]


@functools.lru_cache(maxsize=1)
def _common_ids() -> tuple:
    return tuple(_compute_common_ids())


def common_ids() -> List[str]:
    return list(_common_ids())


def _compute_common_ids() -> List[str]:
    """The 287 states (design §6.2): the maturity interim bank's LEAN ids whose banked F worlds are OK (the LEAN read's
    common set: truth ok, no root tie, F ok at every anchor; all 33 drops were F exclusions). Recomputed from the banked
    F WORLD rows only (no outcome) and checked against the registered sha256."""
    st = mat_states()
    lean = sorted(k for k, v in st.items() if v["lean"])
    fw: Dict[str, dict] = {}
    for r in read_rows(MAT_ARCH / "worlds" / "shared" / "F" / "worlds_0_of_1.jsonl"):
        if r["id"] in st and r["id"] not in fw:
            fw[r["id"]] = r
    ids = [i for i in lean if fw.get(i, {}).get("ok")]
    sha = hashlib.sha256("".join(ids).encode()).hexdigest()
    if sha != COMMON_IDS_SHA256 or len(ids) != N_STATES:
        raise SystemExit(f"the common states do not reproduce the registered set: {len(ids)} ids, sha256 {sha}")
    return ids


def ordered_ids() -> List[str]:
    """The 287 states in the REGISTERED processing order: sha256(ORDER_SALT + id). Every state is built and played in
    every arm before the next starts, so any done prefix is a uniform random sample of the 287 (PREREGISTRATION §5)."""
    return sorted(common_ids(), key=lambda i: hashlib.sha256((ORDER_SALT + i).encode()).hexdigest())


def anchor_ckpt() -> dict:
    """The 225M anchor's registered copy {file, sha256, step} (the maturity archive's ckpts.json)."""
    e = json.loads((MAT_ARCH / "ckpts.json").read_text())["anchors"][ANCHOR]
    return {"file": e["file"], "sha256": e["sha256"], "step": int(e["step"])}


def stop_at() -> Optional[float]:
    """The registered STOP time (epoch seconds) written at launch, or None."""
    p = ARCH / "STOP_AT"
    return float(p.read_text().split()[0]) if p.exists() else None
