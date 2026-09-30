"""``python -m main.rust_core_m5 gates|slice-n|depth3|throughput|verdict`` — see the package docstring.

Every subcommand writes its JSON to ``--results`` (default: this lane's measurement directory) as
``<component>_<tier>.json``; ``verdict`` composes the M5 gate from what is there. A component that
was never run reads NOT RUN — never a pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def default_results() -> Path:
    from utils.paths import repo_path

    return repo_path("designs", "research_state", "measurements", "m5_laneJ", "results")


def _head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write(results: Path, name: str, payload: dict) -> Path:
    from utils.paths import main_models_dir

    results.mkdir(parents=True, exist_ok=True)
    md = main_models_dir()
    if md is not None and str(results.resolve()).startswith(str(md.resolve())):
        raise SystemExit("refusing to write under models/ (read-only for this harness)")
    payload = {**payload, "commit": _head()}
    p = results / f"{name}.json"
    p.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    return p


def _gates(a) -> int:
    from main.rust_core_m5 import gates as G

    if a.from_status:
        vs = G.recorded(head=_head())
        name, tier = "gates_recorded", "recorded"
    else:
        lanes = [x for x in a.lanes.split(",") if x] if a.lanes else None
        vs = G.run(a.tier, gpu=a.gpu, workers=a.workers, lanes=lanes)
        # a SUBSET run never overwrites the full table: its own file, overlaid by `verdict`
        name, tier = (f"gates_{a.tier}" + (f"_lanes_{'_'.join(lanes)}" if lanes else "")), a.tier
    print(G.render(vs))
    p = _write(a.results, name, {"component": "gates", "tier": tier, "gpu": bool(a.gpu), "rows": G.to_json(vs)})
    print(f"\n→ {p}")
    return 0 if all(v.verdict in (G.PASS, G.NOT_BUILT) for v in vs) else 1


def _slice_n(a) -> int:
    from main.rust_core_m5 import slice_n as S

    res = S.run_tier(a.tier)
    print(S.render(res))
    print(f"\n→ {_write(a.results, f'slice_n_{a.tier}', res)}")
    return 0 if res["ok"] else 1


def _depth3(a) -> int:
    from main.rust_core_m5 import depth3 as D

    res = D.run_tier(a.tier)
    print(D.render(res))
    print(f"\n→ {_write(a.results, f'depth3_{a.tier}', res)}")
    return 0 if res["ok"] else 1


def compose(results: Path) -> dict:
    """The M5 GATE (program §2 M5 order constraint 4) from the component JSONs in ``results``."""
    from main.rust_core_m5 import gates as G
    from main.rust_core_m5 import lanes as L

    def load(name: str) -> Optional[dict]:
        p = results / f"{name}.json"
        return json.loads(p.read_text()) if p.is_file() else None

    lanes = load("gates_milestone") or load("gates_recorded") or load("gates_commit")
    overlays = []
    if lanes is not None:
        # a later SUBSET run of the same tier replaces its lanes' rows (e.g. `--lanes E,T2 --gpu`)
        for f in sorted(results.glob(f"gates_{lanes['tier']}_lanes_*.json"), key=lambda q: q.stat().st_mtime):
            sub = json.loads(f.read_text())
            lanes = {**lanes, "rows": [next((r for r in sub["rows"] if r["lane"] == x["lane"]), x) for x in lanes["rows"]]}
            overlays.append(f.name)
    items = []
    if lanes is None:
        items.append(("lane gates", G.NOT_RUN, "run `gates --tier milestone` (or `--from-status`)"))
    else:
        by = {r["lane"]: r for r in lanes["rows"]}
        for row in L.LANES:
            if not row.m5_gate:
                continue
            r = by.get(row.lane)
            v = r["verdict"] if r else G.NOT_RUN
            gpu = f", GPU part {r['gpu']}" if r and r.get("gpu") else ""
            src = next((o for o in overlays if f"_{row.lane}_" in o.replace(".json", "_")), "")
            items.append((f"lane {row.lane} — {row.title}", v, f"{lanes['tier']}{gpu}" + (f" ({src})" if src else "")))
    for comp, label in (("slice_n", "slice N at the env level"), ("depth3", "the depth-3 successor slice")):
        m, c = load(f"{comp}_milestone"), load(f"{comp}_commit")
        if m is not None:
            extra = f", refused by both roads: {m.get('refused_both')} (F-LI-1)" if comp == "depth3" else ""
            items.append((label, G.PASS if m["ok"] else G.FAIL, f"MILESTONE at {m['commit'][:8]}{extra}"))
        elif c is not None:
            items.append((label, G.INCONCLUSIVE if c["ok"] else G.FAIL,
                          f"COMMIT tier only at {c['commit'][:8]} — the gate is the MILESTONE tier"))
        else:
            items.append((label, G.NOT_RUN, f"run `{comp.replace('_', '-')} --tier milestone`"))
    tp = sorted(results.glob("throughput*.json"))
    tp48 = [json.loads(p.read_text()) for p in tp]
    tp48 = [t for t in tp48 if int(t.get("regime", {}).get("n_envs", t.get("n_envs", 0)) or 0) == 48]
    if tp48:
        prod = [t for t in tp48 if (t.get("regime", {}).get("opponent") or {}).get("name") == "production_mix"]
        shape = (f"{len(prod)} at the PRODUCTION shape (Lane G: learner sampling + production mix + the "
                 "complete-game collector on the rust arm)" if prod else "env step only")
        items.append(("throughput A/B at --n-envs 48", "MEASURED",
                      f"{len(tp48)} read(s), {shape}; a DESCRIPTOR (no bar registered)"))
    else:
        items.append(("throughput A/B at --n-envs 48", G.NOT_RUN, "run `throughput --n-envs 48 …`"))
    met = all(v in (G.PASS, "MEASURED") for _, v, _ in items)
    missing = [k for k, v, _ in items if v not in (G.PASS, "MEASURED")]
    return {"component": "verdict", "m5_gate": "MET" if met else "NOT MET", "items": items, "missing": missing}


def _verdict(a) -> int:
    res = compose(a.results)
    print(f"M5 GATE: {res['m5_gate']}\n")
    print("| component | verdict | evidence |\n|---|---|---|")
    for k, v, e in res["items"]:
        print(f"| {k} | **{v}** | {e} |")
    if res["missing"]:
        print("\nNot met by: " + "; ".join(res["missing"]))
    print(f"\n→ {_write(a.results, 'verdict', res)}")
    return 0 if res["m5_gate"] == "MET" else 1


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "throughput":
        from main.rust_core_m5 import throughput as T

        return int(T.main(argv[1:]))
    ap = argparse.ArgumentParser(prog="python -m main.rust_core_m5", description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=None, help="where component JSONs are written / read")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gates", help="every lane's own gate, run or read, as one verdict table")
    g.add_argument("--tier", choices=("commit", "milestone"), default="commit")
    g.add_argument("--from-status", action="store_true", help="read the banked MILESTONE verdicts; run nothing")
    g.add_argument("--gpu", action="store_true", help="also run the GPU tests (under the GPU lock)")
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--lanes", default="", help="comma-separated lane ids (default: all)")
    for name, helptext in (("slice-n", "slice N at the env level"), ("depth3", "the depth-3 successor slice")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--tier", choices=("commit", "milestone"), default="commit")
    sub.add_parser("throughput", help="the throughput A/B (its own flags: `throughput --help`)")
    sub.add_parser("verdict", help="compose the M5 gate from the component JSONs")
    a = ap.parse_args(argv)
    a.results = a.results or default_results()
    return {"gates": _gates, "slice-n": _slice_n, "depth3": _depth3, "verdict": _verdict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
