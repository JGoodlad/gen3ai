"""X26c read, SECONDARY: S of the continuation against the FROZEN R (PREREG §4). ``../x35_breakers_stage_2026-10-17/
d1_read/d1p_read.py`` (itself the X26 drift pre-read's machinery) with only the nodes, the family / request names, the
per-node panel and the readiness rule changed.

NODES, played in this order as they become ready:
  * ``x26_380m_home72``  X26 @380,000,195 (the fork point) on the home72 panel (``--panel home72``)    S2, reported
  * ``x26c_480m_pool``   the 480M final on the eval team builder (no panel): THE DRIFT READ'S REGIME   S1, registered
  * ``x26c_480m_home72`` the 480M final on the home72 panel                                            S2, reported
Each (node, member of R) cell is 500 completed mirrored pairs, greedy vs greedy, CPU, ``torch_threads`` 1; a UNIT is
two engines time-boxed to <= 40 min in ONE heavy slot (``mem_cap.sh --slot``), on the agent cores at nice 19.

    s_read.py run  --cache ~/gen3ai_archive/x26c_read/s --hours 20
    s_read.py read --cache ~/gen3ai_archive/x26c_read/s --out results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent / "x35_breakers_stage_2026-10-17"
sys.path.insert(0, str(STAGE / "x26_drift_preread"))
sys.path.insert(0, str(HERE))
import finals as FN
import preread as PR

REFPOP = STAGE / "reference_population_production.json"
REFPOP_SHA = "cef0960eac605f1d"
FAMILIES = {None: "x26c_s_pool_2026-10-25", "home72": "x26c_s_home72_2026-10-25"}
PROTOCOL = PR.PROTOCOL
RULE = "X26c PREREG §4: S of the continuation on the frozen R (population_score)"
PAIRS = PR.PAIRS
BATCH_PAIRS = PR.BATCH_PAIRS
#: (label, panel, source) in play order; source "fork" = X26 @380M, "final" = the 480M final (finals.final_480m)
NODES = (("x26_380m_home72", "home72", "fork"), ("x26c_480m_pool", None, "final"),
         ("x26c_480m_home72", "home72", "final"))
POLL_S = 300
UNIT_TIME_UP = PR.UNIT_TIME_UP


def req_id(label: str) -> str:
    return f"x26c_{label}_b{BATCH_PAIRS}"


def panel_of(label: str) -> str | None:
    return {lab: p for lab, p, _s in NODES}[label]


def _log(msg: str) -> None:
    print(f"[x26c-s {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def refpop() -> dict[str, Any]:
    return json.loads(REFPOP.read_text())


def source(kind: str) -> Path | None:
    return FN.FORK if kind == "fork" else FN.final_480m()


def cell_pairs(cache: Path, label: str) -> dict[str, int]:
    from main.h2h import cross as CX

    st = PR._state(cache)
    label_of = {v["sha256"]: k for k, v in st["members"].items()}
    try:
        cells = CX.look_cells(cache / "ledger", FAMILIES[panel_of(label)], req_id(label), PROTOCOL)
    except Exception as exc:  # noqa: BLE001 - no row yet reads as nothing played
        _log(f"{label}: no cells read ({type(exc).__name__}: {exc})")
        return {}
    sha = st["nodes"][label]["sha256"]
    return {label_of[c.opponent]: c.n_pairs for c in cells if c.player == sha and c.opponent in label_of}


def cmd_run(a: argparse.Namespace) -> int:
    cache = Path(a.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    st = PR._state(cache)
    if "stop_epoch" not in st:
        st["started_epoch"] = time.time()
        st["stop_epoch"] = time.time() + 3600.0 * a.hours
        st["stop_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st["stop_epoch"]))
        st["refpop_content_sha256"] = refpop()["content_sha256"]
        PR._save_state(cache, st)
    if not st["refpop_content_sha256"].startswith(REFPOP_SHA):
        _log(f"STOP: R is {st['refpop_content_sha256'][:16]}, not the drift read's {REFPOP_SHA}")
        return 3
    _log(f"registered stop {st['stop_iso']}; R content {st['refpop_content_sha256'][:12]}")
    py = sys.executable
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for fam in FAMILIES.values():
        subprocess.run([py, "-m", "main.eval_ledger", "family-register", "--root", str(cache / "ledger"), "--id",
                        fam, "--decision-kind", "ab_verdict", "--rule", RULE, "--protocol", PROTOCOL], check=True,
                       env=env)
    st.setdefault("members", {})
    for m in refpop()["members"]:
        dst, sha = PR.copy_checked(Path(m["path"]), cache / "ckpts" / "members" / m["label"], m["sha256"])
        st["members"][m["label"]] = {"copy": str(dst), "sha256": sha, "source": m["path"]}
    PR._save_state(cache, st)
    st.setdefault("nodes", {})
    while True:
        rp = refpop()
        if rp["content_sha256"] != st["refpop_content_sha256"]:
            _log("STOP: R changed under the read; nothing is re-scored silently")
            return 3
        labels = [m["label"] for m in rp["members"]]
        pending: list[tuple[str, list[str]]] = []
        waiting: list[str] = []
        for label, _panel, kind in NODES:
            if label not in st["nodes"]:
                z = source(kind)
                if z is None:
                    waiting.append(label)
                    continue
                dst, sha = PR.copy_checked(z, cache / "ckpts" / "nodes" / label)
                from main.x35.kit import _final_step
                st["nodes"][label] = {"copy": str(dst), "sha256": sha, "source": str(z),
                                      "step": _final_step(str(dst))}
                PR._save_state(cache, st)
                _log(f"{label}: {z} step {st['nodes'][label]['step']:,} sha {sha[:12]} copied")
            have = cell_pairs(cache, label)
            left = [lab for lab in labels if have.get(lab, 0) < PAIRS]
            if left:
                pending.append((label, left))
        if not pending and not waiting:
            break
        if time.time() + 60.0 >= st["stop_epoch"]:
            _log(f"STOP at the registered stop time: pending {pending}, waiting for {waiting}")
            break
        if not pending:
            _log(f"all ready nodes complete; waiting for {waiting} (poll {POLL_S} s)")
            time.sleep(POLL_S)
            continue
        # the REGISTERED node (S1) first whenever it is ready
        pending.sort(key=lambda x: 0 if x[0] == "x26c_480m_pool" else 1)
        engines: list[str] = []
        for label, left in pending:
            k = min(len(left), a.workers - len(engines))
            engines += [f"{label}:" + ",".join(left[w::k]) for w in range(k)]
            if len(engines) >= a.workers:
                break
        box = int(min(a.unit_s, st["stop_epoch"] - time.time()))
        argv = [str(Path(a.mem_cap)), "--slot", "--name", "x26cs", str(a.cap_gb), "nice", "-n", "19",
                "taskset", "-c", a.cpus, py, str(Path(__file__).resolve()), "unit", "--cache", str(cache),
                "--box-s", str(box), "--threads", str(a.threads), *[x for e in engines for x in ("--engine", e)]]
        _log(f"unit {engines}, box {box} s: queued for a heavy slot")
        t0 = time.time()
        rc = subprocess.run(argv, env=env, check=False).returncode
        _log(f"unit {engines}: exit {rc} after {time.time() - t0:.0f} s (slot wait included)")
        if rc not in (0, UNIT_TIME_UP):
            return rc
        st = PR._state(cache)
        cmd_read(argparse.Namespace(cache=str(cache), out=str(cache / "results.json")))
    cmd_read(argparse.Namespace(cache=str(cache), out=str(cache / "results.json")))
    return 0


def cmd_unit(a: argparse.Namespace) -> int:
    cache = Path(a.cache).expanduser()
    st = PR._state(cache)
    py = sys.executable
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    deadline = time.time() + a.box_s
    procs = []
    for spec in a.engine:
        label, labs = spec.split(":", 1)
        panel = panel_of(label)
        argv = [py, "-m", "main.h2h", "play-many", "--players", st["nodes"][label]["copy"], "--opponents",
                *[st["members"][lab]["copy"] for lab in labs.split(",")], "--pairs", str(PAIRS), "--batch-pairs",
                str(BATCH_PAIRS), "--purpose", "ab", "--family", FAMILIES[panel], "--request", req_id(label),
                "--out", str(cache / "ledger"), "--label", "x26c_s", *(["--panel", panel] if panel else []),
                "--device", "cpu", "--torch-threads", "1", "--threads", str(a.threads)]
        logf = open(cache / f"play_{label}.log", "a")  # noqa: SIM115 - closed when the child is reaped
        procs.append((spec, subprocess.Popen(argv, env=env, stdout=logf, stderr=subprocess.STDOUT), logf))
    while any(p.poll() is None for _s, p, _f in procs) and time.time() < deadline:
        time.sleep(5)
    timed_out = False
    for _s, p, _f in procs:
        if p.poll() is None:
            timed_out = True
            p.terminate()
    rc = 0
    for spec, p, logf in procs:
        try:
            r = p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            r = p.wait()
        logf.close()
        _log(f"unit engine {spec}: exit {r}{' (time box)' if timed_out else ''}")
        if not timed_out:
            rc = rc or r
    return UNIT_TIME_UP if timed_out and rc == 0 else rc


def cmd_read(a: argparse.Namespace) -> int:
    from main.h2h import cross as CX
    from main.x35 import readout as R

    cache = Path(a.cache).expanduser()
    st = PR._state(cache)
    rp = refpop()
    mem = {m["label"]: m["sha256"] for m in rp["members"]}
    label_of = {v: k for k, v in mem.items()}
    rows = []
    for label, panel, kind in NODES:
        node = st.get("nodes", {}).get(label)
        if node is None:
            rows.append({"label": label, "panel": panel, "s_pp": None, "status": "not started"})
            continue
        try:
            cells = CX.look_cells(cache / "ledger", FAMILIES[panel], req_id(label), PROTOCOL)
        except Exception:  # noqa: BLE001 - a node with no row yet
            cells = []
        ps = R.population_score(cells, node["sha256"], mem)
        mine = [c for c in cells if c.player == node["sha256"] and c.opponent in label_of]
        ses = {label_of[c.opponent]: 100.0 * PR._cell_se(c.pairs) for c in mine if c.pairs and sum(c.pairs)}
        row: dict[str, Any] = {"label": label, "panel": panel, "source": node["source"], "step": node["step"],
                               "sha256": node["sha256"], "s_pp": ps.s_pp, "per_member_pp": dict(ps.per_member_pp),
                               "per_member_se_pp": ses, "inconclusive": list(ps.inconclusive),
                               "n_pairs": {label_of[c.opponent]: c.n_pairs for c in mine},
                               "regime_ids": sorted({str(getattr(c, "regime_id", "")) for c in mine})}
        if ps.s_pp is not None and len(ses) == len(mem):
            se = math.sqrt(math.fsum(x * x for x in ses.values())) / len(mem)
            row.update(se_pp=se, ci95_pp=[ps.s_pp - PR.Z95 * se, ps.s_pp + PR.Z95 * se])
        rows.append(row)
    out: dict[str, Any] = {"families": {str(k): v for k, v in FAMILIES.items()}, "pairs_per_cell": PAIRS,
                           "batch_pairs": BATCH_PAIRS, "refpop": REFPOP.name,
                           "refpop_content_sha256": rp["content_sha256"],
                           "regime": "main.h2h play-many, greedy vs greedy, mirrored pairs, cpu, torch_threads 1; "
                                     "pool = the eval team builder (no panel), home72 = --panel home72",
                           "stop_iso": st.get("stop_iso"), "nodes": rows}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({r["label"]: [r.get("s_pp"), r.get("se_pp"), r.get("n_pairs")] for r in rows}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cache", required=True)
    r.add_argument("--hours", type=float, default=20.0)
    r.add_argument("--threads", type=int, default=2, help="the Rust core's worker threads per engine")
    r.add_argument("--workers", type=int, default=2, help="engines per unit (one cell each at a time)")
    r.add_argument("--mem-cap", default=str(HERE.parents[3] / "scripts" / "ops" / "mem_cap.sh"))
    r.add_argument("--cap-gb", type=int, default=12)
    r.add_argument("--cpus", default="5-7,13-15")
    r.add_argument("--unit-s", type=int, default=2400, help="a unit's time box (one heavy-slot hold), s")
    u = sub.add_parser("unit")
    u.add_argument("--cache", required=True)
    u.add_argument("--box-s", type=int, required=True)
    u.add_argument("--threads", type=int, default=2)
    u.add_argument("--engine", action="append", required=True, help="<node label>:<member>[,<member>...]")
    d = sub.add_parser("read")
    d.add_argument("--cache", required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "unit": cmd_unit, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
