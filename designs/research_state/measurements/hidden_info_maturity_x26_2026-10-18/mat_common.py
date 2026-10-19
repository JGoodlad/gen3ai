"""Shared constants of the MATURITY BATTERY on X26 (`design_hidden_info_binding.md` §12.13.7; `PREREGISTRATION.md`).
Pure Python at import: importable by the pin-side drivers, the HEAD-side world builder and the numpy-only reader alike.

The battery's GENERIC parts live in ``src/main/hidden_info_study/`` (``maturity_series`` = the series registry, the
25M trigger, the read-only checkpoint copy, the cap-new-work rule; ``maturity_bank`` = the frozen-bank draw;
``maturity_rules`` = R-MATURITY). They are loaded here BY FILE PATH from THIS checkout (:func:`src_module`), so a
pin-side driver whose ``PYTHONPATH`` names the pin's ``src`` runs this exact source (the ``hand_eval_pin.py`` pattern).

Where things run (`run_mat.sh` selects the checkout and refuses a wrong HEAD):
  * pin   ``~/.cache/gen3ai/hib_mat/pin_<series>`` (the series' pin commit; X26 = ``9c033489``, which already holds
          the gender draw pad, so the world playouts run here too): the BANK (replay only), the ROOTS, the TRUTH
          (Phase 1), the network BELIEF read and every WORLD PLAYOUT;
  * head  ``$HIB_MAT_HEAD`` (default ``~/.cache/gen3ai/hib_mat/head``: a detached checkout of the commit that launched
          the queue): the WORLD builder (§12.13.7 (f): the builder, prior tables and presets of that commit).

Archive (``$HIB_MAT_ARCHIVE``, default ``~/gen3ai_archive/hidden_info_maturity``)::

    traces/<run>/eval_traces/step_<n>/   the copied eval cycles the bank is drawn from (+ traces_manifest.json)
    ckpt/model_config.json, ckpt/<label>/checkpoint_<step>_steps.zip   the read-only checkpoint copies
    ckpts.json                           the series' registered anchors (the queue's work list; the trigger appends)
    bank/, census.npz                    the frozen bank (states_mat_<series>.json in this folder holds its ids)
    roots/<label>.jsonl                  the checkpoint's own root logits / V / top1 per bank state
    fn_belief/<label>.jsonl              the checkpoint's network belief per bank state (FN)
    worlds/shared/<arm>/                 network-free worlds (F, F+M, F+U, V_oracle): built once, played per checkpoint
    worlds/<label>/FN/                   the checkpoint's FN worlds
    rows/<label>/<O|arm>/                every playout row
    logs/                                the queue's and each unit's logs
    plateau/                             THE PLATEAU BANK's namespace (amendment A1): its own bank/, census.npz, roots/,
                                         fn_belief/, worlds/, rows/, score/ (ckpt/, ckpts.json, traces/, logs/ are shared)

BANKS (amendment A1, 2026-10-19): ``interim`` (PREREGISTRATION §2, the archive root) and ``plateau`` (§2.5, drawn from
P*'s three newest cycles, salt ``hib_maturity_x26_plateau_v1:``, the interim battles excluded; ``<archive>/plateau/``).
A driver selects its bank with ``--bank`` (:func:`set_bank`); the queue counts with :func:`using_bank`.

PER-(ANCHOR, BANK) MODES (A1): ``ckpts.json`` anchors carry ``banks: {bank: {mode, priority}}`` (an anchor without it is
the original registration: ``{interim: {mode, priority}}`` from its top-level fields). Modes: ``lean`` (320 ids; O, F,
FN), ``full`` (800; + F+M, F+U), ``full_hide`` (800; + the hide-only O−M, O−U). A mode is only ever RAISED.

THE BUILDER stays the registration commit ``826bee89`` (§12.13.7 (f): a changed builder starts a new series): the queue
runs the drivers of the checkout it is launched from (``HIB_MAT_DRIVER_SHA``, stamped into every row as ``driver``) and
builds worlds in ``$HIB_MAT_HEAD`` at :data:`BUILDER_SHA`.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
REPO = HERE.parents[3]
T1 = MEAS / "hidden_info_tier1_2026-10-13"
PILOT = MEAS / "value_disc_pilot_2026-10-11"
BRC = MEAS / "hidden_info_by_reveal_count_2026-10-12"
MODELS = Path("/home/goodlad/dev/gen3ai/models")
ARCH = Path(os.environ.get("HIB_MAT_ARCHIVE") or (Path.home() / "gen3ai_archive" / "hidden_info_maturity"))
CACHE = Path.home() / ".cache" / "gen3ai" / "hib_mat"
#: ``HIB_MAT_PILOT=1``: the ENGINEERING PILOT (states OUTSIDE the bank, `make_mat_bank.py --pilot`): every output under
#: ``<archive>/pilot/`` and the bank = ``<archive>/pilot_bank``; never trajectory output
PILOT_MODE = os.environ.get("HIB_MAT_PILOT") == "1"

SERIES_NAME = "x26"

# ---- the bank (PREREGISTRATION §2) ----
DRAW_CYCLES = ("step_230000074", "step_232000091", "step_234000025")    # X26's newest three cycles at registration
EXTENSION_CYCLE = "step_228000030"                                       # §2.4: the one pre-registered extension
STRATA = ("snapshot", "bot")                                             # opponent = an own snapshot / a roster bot
FULL_PER_CELL = 80                                                       # 160 per k bin (§12.13.7: 800 states)
LEAN_PER_CELL = 32                                                       # 64 per k bin (§12.13.7: 320 states)
STATES_SALT = "hib_maturity_x26_v1:"
PER_BATTLE_PER_BIN = 1
PER_BATTLE = 3
PILOT_SALT = "hib_maturity_pilot_v1:"

#: the registered world-builder commit (PREREGISTRATION §1; §12.13.7 (f)): every world of the series is built here
BUILDER_SHA = "826bee8919c0986fd3733d3d06f03803822a929a"


@dataclass(frozen=True)
class BankSpec:
    """One frozen bank of the series. ``subdir`` = its archive namespace (None = the archive root)."""

    name: str
    cycles: Tuple[str, ...]
    extension: str
    salt: str
    states_file: str
    subdir: Optional[str]
    exclude: Tuple[str, ...] = ()


BANKS: Dict[str, BankSpec] = {
    "interim": BankSpec("interim", DRAW_CYCLES, EXTENSION_CYCLE, STATES_SALT, "states_mat_x26.json", None),
    # §2.5 / A1: P* = checkpoint_280000170's three newest eval cycles (all recorded at the pin), the one extension the
    # cycle before them, a new salt, and every battle of the interim bank excluded
    "plateau": BankSpec("plateau", ("step_276000077", "step_278000056", "step_280000170"), "step_274000392",
                        "hib_maturity_x26_plateau_v1:", "states_mat_x26_plateau.json", "plateau", ("interim",)),
}
_BANK = ["interim"]


def set_bank(name: str) -> BankSpec:
    if name not in BANKS:
        raise SystemExit(f"unknown bank {name!r} (one of {sorted(BANKS)})")
    _BANK[0] = name
    return BANKS[name]


def bank() -> BankSpec:
    return BANKS[_BANK[0]]


@contextlib.contextmanager
def using_bank(name: str) -> Iterator[BankSpec]:
    prev = _BANK[0]
    set_bank(name)
    try:
        yield BANKS[name]
    finally:
        _BANK[0] = prev

# ---- the arms (§12.13.7 (e)) ----
#: label -> (information-lever spec or "fn", shared across checkpoints?)
ARMS: Dict[str, tuple] = {"F": ("off", True), "F+M": ("F+M", True), "F+U": ("F+U", True), "FN": ("fn", False),
                          # A1: R3's hide-only half (E2's arms, `agents.info_levers.spec.PRESETS`), network-free
                          "O-M": ("O-M", True), "O-U": ("O-U", True)}
MODE_ARMS = {"lean": ("F", "FN"), "full": ("F", "FN", "F+M", "F+U"),
             "full_hide": ("F", "FN", "F+M", "F+U", "O-M", "O-U")}
MODE_RANK = {"lean": 0, "full": 1, "full_hide": 2}
#: the shared world arms in BUILD order (E2's: F first, so every sibling re-samples F into its screen cache)
SHARED_BUILD_ORDER = ("F", "F+U", "F+M", "O-M", "O-U")
HIDE_ARMS = ("O-M", "O-U")
J = 8
N_IDENTITY = 12


def src_module(name: str):
    """``src/main/hidden_info_study/<name>.py`` of THIS checkout, loaded by file path (cached in sys.modules under a
    private name, so the pin's ``main`` package is never shadowed)."""
    key = f"_hib_mat_{name}"
    if key in sys.modules:
        return sys.modules[key]
    p = REPO / "src" / "main" / "hidden_info_study" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(key, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def series():
    return src_module("maturity_series").SERIES[SERIES_NAME]


def arm_dir(label: str) -> str:
    return label.replace("+", "p").replace("-", "m")


#: archive entries shared by every bank (never under a bank's namespace)
SHARED_ROOTS = ("ckpt", "logs", "traces")


def bank_root() -> Path:
    """The current bank's archive namespace (the pilot's ``pilot/`` for the interim bank in pilot mode)."""
    b = bank()
    if PILOT_MODE:
        if b.subdir:
            raise SystemExit("the pilot runs on the interim bank only")
        return ARCH / "pilot"
    return ARCH / b.subdir if b.subdir else ARCH


def d(*parts) -> Path:
    p = (ARCH if parts[0] in SHARED_ROOTS else bank_root()).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def traces_root() -> Path:
    """The models-root the bank loader reads the COPIED eval cycles from (``<root>/<run>/eval_traces/...``)."""
    return ARCH / "traces"


def bank_path() -> Path:
    return ARCH / "pilot_bank" if PILOT_MODE else bank_root() / "bank"


def census_path() -> Path:
    return bank_root() / "census.npz"


def states_path(name: Optional[str] = None) -> Path:
    return HERE / BANKS[name or bank().name].states_file


def ckpts_path() -> Path:
    return ARCH / "ckpts.json"


def registered_ckpts() -> Dict[str, dict]:
    """{label: {step, file, sha256, mode, ...}} from the archive's ckpts.json (the queue's work list)."""
    p = ckpts_path()
    return json.loads(p.read_text())["anchors"] if p.exists() else {}


def anchor_banks(label: str) -> Dict[str, dict]:
    """{bank: {mode, priority}} of one registered anchor (A1); the original registration's top-level mode / priority
    is the interim bank's when the anchor has no ``banks`` entry."""
    e = registered_ckpts()[label]
    return e.get("banks") or {"interim": {"mode": e["mode"], "priority": e["priority"]}}


def anchor_mode(label: str, bank_name: Optional[str] = None) -> Optional[str]:
    """The anchor's mode on a bank (default: the current one); None = the anchor is not scored on that bank."""
    e = anchor_banks(label).get(bank_name or bank().name)
    return None if e is None else e["mode"]


def ckpt_file(label: str) -> Path:
    e = registered_ckpts()[label]
    return Path(e["file"])


def worlds_dir(label: Optional[str], arm: str) -> Path:
    """Shared arms live under worlds/shared/; FN under worlds/<label>/FN/; validation builds under worlds/shared/V_<spec>."""
    if arm.startswith("V:"):
        return d("worlds", "shared", "V_" + arm_dir(arm[2:]))
    if ARMS[arm][1]:
        return d("worlds", "shared", arm_dir(arm))
    if label is None:
        raise ValueError(f"{arm} is per checkpoint: name the checkpoint")
    return d("worlds", label, arm_dir(arm))


def rows_dir(label: str, arm: str) -> Path:
    """``arm`` an arm, ``O`` (the truth) or ``V:<spec>``."""
    sub = "O" if arm == "O" else ("V_" + arm_dir(arm[2:])) if arm.startswith("V:") else arm_dir(arm)
    return d("rows", label, sub)


def write_row(f, row: dict) -> None:
    """One fsync'd JSONL row. A1: the queue's driver commit (``driver``) and the bank (``bank``) are stamped on every row
    written under a launch that sets ``HIB_MAT_DRIVER_SHA`` (the interim LEAN rows predate them)."""
    drv = os.environ.get("HIB_MAT_DRIVER_SHA")
    if drv:
        row = {**row, "driver": drv, "bank": bank().name}
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


def read_dir(dd: Path, pattern: str = "*.jsonl") -> Dict[str, dict]:
    """{id: the LAST row with that id} over every matching file of ``dd`` (a re-run row supersedes)."""
    out: Dict[str, dict] = {}
    if not dd.exists():
        return out
    for p in sorted(dd.glob(pattern)):
        for r in read_rows(p):
            out[r["id"]] = r
    return out


def bank_states() -> Dict[str, dict]:
    if PILOT_MODE:
        ids = (ARCH / "pilot_ids.txt").read_text().split()
        return {x: {"lean": True} for x in ids}
    p = states_path()
    if not p.exists():
        return {}
    key = (str(p), p.stat().st_mtime_ns)
    if key not in _STATES_CACHE:                    # the queue counts every poll: parse each states file once
        _STATES_CACHE[key] = json.loads(p.read_text())["states"]
    return _STATES_CACHE[key]


_STATES_CACHE: Dict[tuple, Dict[str, dict]] = {}


def mode_ids(mode: str) -> List[str]:
    """The current bank's ids a mode reads: LEAN = the nested 320, FULL (and FULL + hide-only) = all 800."""
    st = bank_states()
    if mode == "lean":
        return sorted(x for x, e in st.items() if e["lean"])
    if mode in ("full", "full_hide"):
        return sorted(st)
    raise ValueError(mode)


def identity_ids() -> List[str]:
    """The 12 identity states: the first 12 LEAN ids of the INTERIM bank in sha256('hib_mat_identity_v1:' + id) order
    (fixed, no outcome). The identity is per ANCHOR (the builder at the HEAD = Phase 1 at the pin), so it is read on the
    interim bank only, whichever bank is current (A1)."""
    import hashlib

    with using_bank("interim"):
        ids = mode_ids("lean")
    return sorted(ids, key=lambda x: hashlib.sha256(("hib_mat_identity_v1:" + x).encode()).hexdigest())[:N_IDENTITY]


def add_paths() -> None:
    """The pilot's folder (its `common`, `make_states`, `run_playouts`, `run_det_playouts`, `det_worlds`), the
    by-reveal-count folder (`stage_facts`) and Tier 1's folder (`t1_data`, `build_worlds`' helpers, `fn_worlds`) on the
    path, after this folder."""
    for p in (T1, BRC, PILOT):
        if str(p) not in sys.path:
            sys.path.insert(1, str(p))


# ---- pin side ----

def check_pin() -> None:
    """Refuse anything but the series' pin checkout, and a ``main`` imported from anywhere else."""
    kind, head, pdir = (os.environ.get(k, "") for k in ("HIB_MAT_KIND", "HIB_MAT_CHECKOUT_HEAD", "HIB_MAT_CHECKOUT_DIR"))
    if kind != "pin" or head != series().pin:
        raise SystemExit(f"not the {SERIES_NAME} pin: kind={kind!r} head={head!r} (run_mat.sh pin)")
    import main

    if not str(Path(main.__file__).resolve()).startswith(pdir):
        raise SystemExit(f"main imported from {main.__file__}, not {pdir}")


def check_head() -> None:
    if os.environ.get("HIB_MAT_KIND") != "head":
        raise SystemExit("worlds are built at the HEAD checkout only (run_mat.sh head)")
    import main

    if not str(Path(main.__file__).resolve()).startswith(os.environ["HIB_MAT_CHECKOUT_DIR"]):
        raise SystemExit(f"main imported from {main.__file__}, not the HEAD checkout")


def threads() -> int:
    return int(os.environ.get("VDP_THREADS", "1"))


def load_model(label: str):
    """The registered checkpoint copy on CPU through the strict loader; refuses a non-winprob critic. The copy's sha256
    is re-checked against ckpts.json first."""
    from main.policy_spectrum.qhat import check_winprob
    from main.policy_spectrum.reader import load_checkpoint

    e = registered_ckpts()[label]
    p = Path(e["file"])
    if src_module("maturity_series").sha256_file(p) != e["sha256"]:
        raise SystemExit(f"{p}: sha256 differs from ckpts.json")
    m = load_checkpoint(p, device="cpu")
    check_winprob(m)
    return m


def install_pilot_loader(label: str) -> None:
    """The pilot's modules call ``common.load_model(cp)``; point it at this checkpoint's copy (the generic
    checkpoint-at-its-pin entry replacing the pilot's hardcoded {end2, prod})."""
    import common as PC

    PC.load_model = lambda _c: load_model(label)


def shard(ids: Sequence[str], i: int, of: int) -> List[str]:
    return [x for k, x in enumerate(sorted(ids)) if k % of == i]
