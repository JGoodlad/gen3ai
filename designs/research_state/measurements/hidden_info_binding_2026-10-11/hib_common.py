"""Shared constants and helpers of the hidden-information binding study, MODEL-INPUT axis.

Runs ONLY under ``run_at_pin.sh`` (PYTHONPATH = the checkpoint's own pin checkout, CUDA hidden), except the
pure-numpy parts (`ARMS`, `arm_codes`) which the scorer / unblinder / tests import with ``python -I``.

Checkpoints (``hib_cp()``):
  end2        models/rb_es_end2_s2001/final_model.zip   pin 62085750   states: end2's registered 600
  prod        models/rb_es_prod_s2001/final_model.zip   pin 95d014fa   states: prod's registered 600
  end2_s2002  models/rb_es_end2_s2002/final_model.zip   pin 62085750   states: END2's registered 600 (replication)
``end2_s2002`` runs as ``HIB_CKPT=end2_s2002 run_at_pin.sh end2 ...`` (the pin is end2's; this module refuses any
other combination). Every path under ``models/`` is opened read-only; outputs go to
``~/gen3ai_archive/hidden_info_binding/<cp>/``. The pilot folder (value_disc_pilot_2026-10-11) is imported read-only.

Named ``hib_common`` (not ``common``) so the pilot's own ``common`` stays importable by the pilot modules we reuse.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "value_disc_pilot_2026-10-11"
MODELS = Path("/home/goodlad/dev/gen3ai/models")
ARCHIVE = Path(os.environ.get("HIB_ARCHIVE") or (Path.home() / "gen3ai_archive" / "hidden_info_binding"))
PILOT_ARCHIVE = Path.home() / "gen3ai_archive" / "value_disc_pilot"

CPS: Dict[str, dict] = {
    "end2": {"run": "rb_es_end2_s2001", "pin_cp": "end2", "states": "end2"},
    "prod": {"run": "rb_es_prod_s2001", "pin_cp": "prod", "states": "prod"},
    "end2_s2002": {"run": "rb_es_end2_s2002", "pin_cp": "end2", "states": "end2"},
}

# ------------------------------------------------------------------------------------------------
# the arms (registered by the brief): family, dose p. TRUE@0.00 is the hook-on null (bit-identical to BASE).
# ------------------------------------------------------------------------------------------------
FAMILIES = ("TRUE", "PLACEBO", "SELFMAP", "TRUEHYP")
DOSES: Dict[str, Tuple[float, ...]] = {
    "TRUE": (0.0, 0.6, 0.8, 0.95, 1.0),
    "PLACEBO": (0.8, 0.95, 1.0),
    "SELFMAP": (0.8, 0.95, 1.0),
    "TRUEHYP": (0.95, 1.0),
}
ARM_SALT = "hib_arm_v1:"
PLACEBO_SALT = "hib_placebo_v1:"


def arm_label(family: str, p: float) -> str:
    return f"{family}@{p:.2f}"


ARMS: List[Tuple[str, float]] = [(f, p) for f in FAMILIES for p in DOSES[f]]


def arm_codes() -> Dict[str, dict]:
    """{code: {"family", "p", "label"}}: codes A01.. assigned in sha256(ARM_SALT + label) order (opaque)."""
    labs = sorted(((hashlib.sha256((ARM_SALT + arm_label(f, p)).encode()).hexdigest(), f, p) for f, p in ARMS))
    return {f"A{i + 1:02d}": {"family": f, "p": p, "label": arm_label(f, p)} for i, (_, f, p) in enumerate(labs)}


def write_arm_key(path: Path = HERE / "arm_key.json") -> Dict[str, dict]:
    key = arm_codes()
    path.write_text(json.dumps({"salt": ARM_SALT, "codes": key}, indent=1, sort_keys=True) + "\n")
    return key


def placebo_seed(did: str, slot: int) -> int:
    return int(hashlib.sha256((PLACEBO_SALT + did + ":" + str(slot)).encode()).hexdigest()[:16], 16)


# ------------------------------------------------------------------------------------------------
# run-time (pin) helpers
# ------------------------------------------------------------------------------------------------

def hib_cp() -> str:
    vdp = os.environ.get("VDP_CP")
    want = os.environ.get("HIB_CKPT") or vdp
    if want not in CPS:
        raise SystemExit(f"run through run_at_pin.sh (VDP_CP={vdp!r}, HIB_CKPT={os.environ.get('HIB_CKPT')!r})")
    if CPS[want]["pin_cp"] != vdp:
        raise SystemExit(f"{want} runs at the {CPS[want]['pin_cp']} pin, not {vdp}")
    return want


def pin_paths() -> None:
    """Put this folder first and the pilot folder second on sys.path (pilot modules import their own ``common``)."""
    for p in (str(PILOT), str(HERE)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def check_pin(cp: str) -> None:
    pin_paths()
    import common as PC  # the pilot's

    PC.check_pin(CPS[cp]["pin_cp"])


def ckpt_path(cp: str) -> Path:
    return MODELS / CPS[cp]["run"] / "final_model.zip"


def out_dir(cp: str) -> Path:
    d = ARCHIVE / cp
    d.mkdir(parents=True, exist_ok=True)
    return d


def states_file(cp: str) -> Path:
    return PILOT / f"states_{CPS[cp]['states']}.json"


def bank_dir(cp: str) -> Path:
    return PILOT_ARCHIVE / CPS[cp]["states"] / "bank"


def phase1_rows_dir(cp: str) -> Path:
    return PILOT_ARCHIVE / CPS[cp]["states"] / "rows"


def load_model(cp: str):
    from main.policy_spectrum.qhat import check_winprob
    from main.policy_spectrum.reader import load_checkpoint

    m = load_checkpoint(ckpt_path(cp), device="cpu")
    check_winprob(m)
    return m


def data_dir() -> Path:
    return Path(os.environ["VDP_PIN"]) / "data" / "pokemon"


def safe_name(did: str) -> str:
    return hashlib.sha256(did.encode()).hexdigest()[:20]


def roots(bank, ids) -> Dict[str, dict]:
    """{decision id: {"row" float32 [obs], "mask" uint8 [11], "side", "battle" (BankBattle), "d" (bank row),
    "opp_team" (the packed team of the side OPPOSITE the decision's)}}: the root rows re-derived by replaying the
    banked battles at the pin (`main.policy_spectrum.replay.replay`, as the pilot's `det_worlds.root_belief`)."""
    import numpy as np
    from main.policy_spectrum.replay import replay

    want = set(ids)
    decs = [d for d in sorted(bank.decisions, key=lambda x: x["id"]) if d["id"] in want]
    missing = want - {d["id"] for d in decs}
    if missing:
        raise RuntimeError(f"{len(missing)} ids are not in the bank")
    bids = sorted({d["battle"] for d in decs})
    reps = dict(zip(bids, replay([bank.battles[bank.battle_index[b]].recorded() for b in bids], workers=1)))
    out = {}
    for d in decs:
        rb = reps[d["battle"]]
        if not rb.ok:
            raise RuntimeError(f"{d['battle']}: replay failed: {rb.error}")
        x = rb.decisions[0 if d["side"] == "p1" else 1][d["n"]]
        b = bank.battles[bank.battle_index[d["battle"]]]
        opp = b.p2 if d["side"] == "p1" else b.p1
        out[d["id"]] = {"row": np.array(x.row, dtype=np.float32), "mask": np.asarray(x.mask, dtype=np.uint8),
                        "side": d["side"], "battle": b, "d": d, "opp_team": opp["team"]}
    return out
