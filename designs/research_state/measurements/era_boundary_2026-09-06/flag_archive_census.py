"""Per-registry-flag archive census over models/*/model_config.json.

Reports, for every one of the 49 registry flags:
  * how many gen-9+ runs (config_version >= 69) recorded it at an ENABLED value
  * the newest such run (by directory mtime) and its config_version
  * how many gen-9+ runs recorded it at its DISABLED value
  * the set of distinct enabled values seen

"Enabled" = not the flag's disabled sentinel: False / 0 / 0.0 / None / "off" / "none".
That is deliberately value-based rather than default-based, because several registry
flags default to an ENABLED value (attend_unrevealed_opponents=True).

Run:  PYTHONPATH=src python designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.py
"""
import json
import pathlib
import os
import sys

from agents.model.flag_registry import REGISTRY
from utils.paths import main_models_dir

GEN9_FLOOR = 69  # the first config_version whose arch_signature is in the current lineage

DISABLED = (False, 0, 0.0, None, "off", "none", "")


def is_enabled(v):
    if isinstance(v, bool):
        return v
    return v not in DISABLED


def main():
    models = main_models_dir()
    if models is None:
        print("no models/ archive on this box", file=sys.stderr)
        return 1

    rows = []
    for d in sorted(os.listdir(models)):
        cfg = os.path.join(models, d, "model_config.json")
        if not os.path.exists(cfg):
            continue
        try:
            with open(cfg) as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print(f"UNREADABLE {d}: {exc}", file=sys.stderr)
            continue
        rows.append((d, os.path.getmtime(cfg), data))

    gen9 = [r for r in rows if int(r[2].get("config_version", 0)) >= GEN9_FLOOR]
    print(f"runs with a model_config.json : {len(rows)}")
    print(f"gen-9+ (config_version >= {GEN9_FLOOR}) : {len(gen9)}")
    vers = sorted({int(r[2].get('config_version', 0)) for r in gen9})
    print(f"gen-9+ config_version span    : {vers[0]}..{vers[-1]}")
    print()

    hdr = f"{'flag':38s} {'default':>10s} {'tier':13s} {'klass':16s} {'ON':>4s} {'OFF':>4s} {'absent':>6s}  newest ON run"
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for flag in REGISTRY:
        on, off, absent = [], 0, 0
        for name, mtime, data in gen9:
            if flag.name not in data:
                absent += 1
                continue
            v = data[flag.name]
            if is_enabled(v):
                on.append((mtime, name, v, int(data.get("config_version", 0))))
            else:
                off += 1
        on.sort(reverse=True)
        newest = f"{on[0][1]} (v{on[0][3]}, {on[0][2]!r})" if on else "-"
        print(
            f"{flag.name:38s} {str(flag.default):>10s} {flag.tier.value:13s} "
            f"{flag.klass.value:16s} {len(on):4d} {off:4d} {absent:6d}  {newest}"
        )
        out[flag.name] = {
            "default": flag.default,
            "tier": flag.tier.value,
            "klass": flag.klass.value,
            "cli_name": flag.cli_name,
            "since": flag.since,
            "on_count": len(on),
            "off_count": off,
            "absent_count": absent,
            "on_runs": [{"run": n, "value": v, "config_version": cv} for _, n, v, cv in on],
            "distinct_on_values": sorted({repr(v) for _, _, v, _ in on}),
        }

    with open("designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json", "w") as fh:
        json.dump({"n_runs": len(rows), "n_gen9": len(gen9), "flags": out}, fh, indent=2)
    print("\nwrote designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
