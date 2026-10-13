"""Shared bits of the STAGE CENSUS and the k-stratified SUPPLEMENT (`census.py`, `select_supplement.py`,
`make_supp_bank.py`, `phase1_supp.py`; registration `design_hidden_info_binding.md` §12.13.1-§12.13.2).

The pin-side scripts run through ``run_pin.sh <end2|prod> ...`` (the UNMODIFIED pin checkouts). They re-derive the
eligible pool EXACTLY as the pilot's ``make_states.py`` does: the run's last three eval-trace cycles, every trace of
every non-``ext_`` roster opponent, the ``core_events --trackers --obs`` replay, gate ① per battle, the clock-decode turn
gate, eligibility = a FREE move round, turn 2-40, >= 3 legal actions. The pilot's own functions are imported (never
copied): ``make_states.revealed_counts`` (k) and ``stage_facts.parse_stage`` (material).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_common as T

# the pilot's folder first (its `common`), then the by-reveal-count folder (its `stage_facts`; it has no `common`)
for _p in (T.BRC, T.PILOT):
    if str(_p) not in sys.path:
        sys.path.insert(1, str(_p))


def cp() -> str:
    c = os.environ.get("VDP_CP")
    if c not in T.CPS:
        raise SystemExit("run through run_pin.sh (VDP_CP unset)")
    return c


def check_plain_pin(c: str) -> None:
    """Refuse anything but the UNMODIFIED pin checkout of ``c`` (run_pin.sh end2 / prod), and refuse a ``main``
    imported from anywhere else."""
    kind, head, pdir = (os.environ.get(k, "") for k in ("HIB_T1_PIN_KIND", "HIB_T1_PIN_HEAD", "HIB_T1_PIN_DIR"))
    if kind != "pin" or head != T.CPS[c]["pin"] or pdir != T.CPS[c]["pin_dir"]:
        raise SystemExit(f"not the plain pin of {c}: kind={kind!r} head={head!r} dir={pdir!r}")
    import main

    if not str(Path(main.__file__).resolve()).startswith(pdir):
        raise SystemExit(f"main imported from {main.__file__}, not {pdir}")


def patch_categories() -> None:
    """make_states' patch, verbatim in effect: the core's choice tokens are SLOT-based at these pins ("move 1"), so
    the category stamp is the token's verb."""
    import main.policy_spectrum.categories as CAT

    CAT.token_category = lambda t: t.split()[0]
    CAT.token_subtype = lambda t: None


def eval_picks(c: str):
    """make_states' trace list: (cycles, [(Source, opponent, prefix)], capture block)."""
    import make_states as MS

    from main.policy_spectrum import bank as B

    run = T.CPS[c]["run"]
    tr = T.MODELS / run / "eval_traces"
    cycles = sorted((p for p in tr.glob("step_*") if p.is_dir()), key=lambda p: int(p.name[5:]))[-MS.N_CYCLES:]
    picks, capture = [], {}
    for cyc in cycles:
        man = json.loads((cyc / "eval_manifest.json").read_text())
        if man.get("git_hash") != T.CPS[c]["pin"]:
            raise SystemExit(f"{cyc}: recorded at {man.get('git_hash')}, not the pin")
        sel = man.get("selection", {}).get("opponents", {})
        src = B.Source(run, cyc.name, f"{c}@{cyc.name}")
        for opp in man["opponents"]:
            if opp.startswith("ext_"):
                continue
            B.opponent_class(opp)
            s = sel.get(opp, {})
            capture[f"{cyc.name}/{opp}"] = {k: s.get(k) for k in (
                "battles_played", "battles_won", "battles_drawn", "capture_rate_win", "capture_rate_loss",
                "capture_rate_draw", "traces_written", "traces_won", "traces_drawn")}
            for p in B.trace_prefixes(cyc / opp):
                picks.append((src, opp, p))
    return cycles, picks, capture


def eligible(row: dict) -> bool:
    import make_states as MS

    return row["kind"] == "free" and MS.TURN_MIN <= row["turn"] <= MS.TURN_MAX and row["n_legal"] >= MS.MIN_LEGAL


def n_team(b) -> dict:
    """stage_facts' team sizes (entries of the packed team)."""
    return {who: len([e for e in (b.p1 if who == "p1" else b.p2)["team"].split("]") if e.strip()])
            for who in ("p1", "p2")}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def census_path(c: str) -> Path:
    return T.cp_dir(c) / "census.npz"


def supplement_path(c: str) -> Path:
    return HERE / f"supplement_{c}.json"
