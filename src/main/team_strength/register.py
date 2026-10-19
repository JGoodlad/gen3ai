"""THE REGISTRATION of a team-strength read (``gen3_tsr_registration_v1``): every input frozen BEFORE any game.

``build(spec)`` reads every named team set (byte for byte, with its digest and per-file team ids), draws the
REFERENCE panel (``panel.draw_reference``) and each pilot-B SUBSET (``panel.subset``), proves every rated and reference
team packs under the gen3ou validator (a team that cannot be packed cannot be rated), and summarises the pilot-A plan.
The registration's ``digest`` is sha256 over its canonical JSON without that field; it keys the plan's schedule, and
every unit records it. A set directory inside this checkout is recorded RELATIVE to the repository root (so a pinned
checkout reads its own copy); one outside is absolute.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from main.team_strength import panel as PN
from main.team_strength import plan as PLN

SCHEMA = "gen3_tsr_registration_v1"


def resolve_dir(d: str) -> Path:
    from utils.paths import repo_root

    p = Path(d)
    return p if p.is_absolute() else repo_root() / p


def _record_dir(d: str) -> str:
    from utils.paths import repo_root

    p = Path(d)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.resolve().relative_to(repo_root().resolve()))
    except ValueError:
        return str(p)


def build(spec: Mapping[str, Any], *, check_packing: bool = True) -> Dict[str, Any]:
    sets = {name: PN.read_set(name, resolve_dir(d)) for name, d in spec["sets"].items()}
    rq = spec["reference"]
    ref = PN.draw_reference(sets, rq["quota"], int(rq["seed"]))
    reg: Dict[str, Any] = {"schema": SCHEMA, "name": spec["name"], "notes": spec.get("notes", []),
                           "sets": {}, "reference": {}, "pilots": json.loads(json.dumps(spec["pilots"])),
                           "fit": dict(spec["fit"])}
    for name, ts in sets.items():
        blk = ts.block()
        blk["dir"] = _record_dir(spec["sets"][name])
        reg["sets"][name] = blk
    reg["reference"] = {"quota": dict(rq["quota"]), "seed": int(rq["seed"]), "members": [m.block() for m in ref],
                        "digest": PN.panel_digest(ref)}
    k = reg["pilots"].get("kakuna")
    if k is not None:
        k["subsets"] = {s: PN.subset(sets[s], int(k["subset_size"]), int(k["subset_seed"]), s)
                        for s in k["subset_sets"]}
    if check_packing:
        from main.team_strength.scheduled import ScheduledTeams

        ScheduledTeams(PN.text_of(sets))          # raises if any team fails the validator / packs ambiguously
    reg["membership"] = PN.membership(sets)
    reg["digest"] = PLN.digest({k2: v for k2, v in reg.items() if k2 != "digest"})
    return reg


def check(reg: Mapping[str, Any]) -> None:
    """REFUSE a registration whose digest does not match its content."""
    want = PLN.digest({k: v for k, v in reg.items() if k != "digest"})
    if want != reg.get("digest"):
        raise PN.PanelError(f"registration digest {str(reg.get('digest'))[:12]} != its content's {want[:12]}")


def load(path: Path) -> Dict[str, Any]:
    reg = json.loads(Path(path).read_text())
    if reg.get("schema") != SCHEMA:
        raise PN.PanelError(f"{path}: schema {reg.get('schema')!r} is not {SCHEMA}")
    check(reg)
    for blk in reg["sets"].values():
        blk["dir"] = str(resolve_dir(blk["dir"]))
    return reg
