"""X35 SPRT (STAGED_PLAN §4.2) look 1: S of the three SPRT finals (s3501-s3503) on the FROZEN reference population R
(``reference_population_production.json``, 7 members), in G1's regime (``adfdbee824c9eb0a``): 500 mirrored pairs per
(final, member) cell, ``main.h2h play-many``, greedy vs greedy, the eval team builder (no panel), CPU. The matched
control's (C0 s3501-s3503) S are G1's own (``../g1/results.json``), reused, not replayed.

    s_read.py run  --cache ~/gen3ai_archive/x35_sprt_read --hours 14    # detached, resumable, a registered stop
    s_read.py read --cache ~/gen3ai_archive/x35_sprt_read --out s_results.json --gate-input r1_input.json

This is ``../g1/g1_read.py`` with only the finals, the family / request names and the output changed: ``read`` writes
``{"sprt": {seed: S}, "c0": {seed: S}}`` (C0 from G1) once all three SPRT S are conclusive, and no gate.
The per-final interval is the pre-read's DECLARED one: se(S) = sqrt(sum se_m^2) / 7.
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
STAGE = HERE.parent
sys.path.insert(0, str(STAGE / "x26_drift_preread"))
import preread as PR  # noqa: E402  - the pre-read's copy / se / ledger helpers, unchanged

REFPOP = STAGE / "reference_population_production.json"
MODELS = Path("/home/goodlad/dev/gen3ai/models")
STATUS = Path.home() / ".claude" / "jobs" / "x35" / "status.txt"
FAMILY = "x35_sprt_read_2026-10-24"
PROTOCOL = PR.PROTOCOL
RULE = "X35 STAGED_PLAN 4.2 SPRT R1 look 1 (population_score of design_compute_efficiency §5.6.4, G1 regime)"
PAIRS = PR.PAIRS
BATCH_PAIRS = PR.BATCH_PAIRS
PSTAR_STEP = 280_000_170
LENGTH = 20_000_000
#: (label, arm, seed, run), in the order they are played (the two finals ready at the start first)
FINALS = (("sprt_s3501", "sprt", "3501", "rb_x35_sprt_s3501"), ("sprt_s3502", "sprt", "3502", "rb_x35_sprt_s3502"),
          ("sprt_s3503", "sprt", "3503", "rb_x35_sprt_s3503"))
G1_RESULTS = STAGE / "g1" / "results.json"
POLL_S = 300
UNIT_TIME_UP = PR.UNIT_TIME_UP


def req_id(label: str) -> str:
    return f"sprt_{label}_b{BATCH_PAIRS}"


def _log(msg: str) -> None:
    print(f"[sprt {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def refpop() -> dict[str, Any]:
    return json.loads(REFPOP.read_text())


def done_runs() -> set[str]:
    try:
        lines = STATUS.read_text().splitlines()
    except FileNotFoundError:
        return set()
    out = set()
    for ln in lines:
        if "X35END: DONE " in ln:
            out.add(ln.split("X35END: DONE ", 1)[1].split(":", 1)[0].strip())
    return out


def final_ready(run: str) -> Path | None:
    z = MODELS / run / "final_model.zip"
    return z if (run in done_runs() and z.is_file()) else None


def copy_final(z: Path, dst_dir: Path) -> tuple[Path, str, int]:
    from main.x35.kit import _final_step

    step = _final_step(str(z))
    if step < PSTAR_STEP + LENGTH:
        raise SystemExit(f"{z}: final at {step:,} < P* + 20M = {PSTAR_STEP + LENGTH:,}")
    dst, sha = PR.copy_checked(z, dst_dir)
    return dst, sha, step


def cell_pairs(cache: Path, label: str) -> dict[str, int]:
    from main.h2h import cross as CX

    st = PR._state(cache)
    label_of = {v["sha256"]: k for k, v in st["members"].items()}
    try:
        cells = CX.look_cells(cache / "ledger", FAMILY, req_id(label), PROTOCOL)
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
    _log(f"registered stop {st['stop_iso']}; R content {st['refpop_content_sha256'][:12]}")
    py = sys.executable
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run([py, "-m", "main.eval_ledger", "family-register", "--root", str(cache / "ledger"), "--id", FAMILY,
                    "--decision-kind", "ab_verdict", "--rule", RULE, "--protocol", PROTOCOL], check=True, env=env)
    st.setdefault("members", {})
    for m in refpop()["members"]:
        dst, sha = PR.copy_checked(Path(m["path"]), cache / "ckpts" / "members" / m["label"], m["sha256"])
        st["members"][m["label"]] = {"copy": str(dst), "sha256": sha, "source": m["path"]}
    PR._save_state(cache, st)
    st.setdefault("nodes", {})
    while True:
        rp = refpop()
        if rp["content_sha256"] != st["refpop_content_sha256"]:
            _log(f"STOP: R changed under the read ({rp['content_sha256'][:12]} != "
                 f"{st['refpop_content_sha256'][:12]}); nothing is re-scored silently")
            return 3
        labels = [m["label"] for m in rp["members"]]
        pending: list[tuple[str, list[str]]] = []
        waiting: list[str] = []
        for label, _arm, _seed, run in FINALS:
            if label not in st["nodes"]:
                z = final_ready(run)
                if z is None:
                    waiting.append(run)
                    continue
                dst, sha, step = copy_final(z, cache / "ckpts" / "finals" / label)
                st["nodes"][label] = {"copy": str(dst), "sha256": sha, "source": str(z), "step": step, "run": run}
                PR._save_state(cache, st)
                _log(f"{label}: final {z} step {step:,} sha {sha[:12]} copied")
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
            _log(f"all ready finals complete; waiting for {waiting} (poll {POLL_S} s)")
            time.sleep(POLL_S)
            continue
        engines: list[str] = []
        for label, left in pending:
            k = min(len(left), a.workers - len(engines))
            engines += [f"{label}:" + ",".join(left[w::k]) for w in range(k)]
            if len(engines) >= a.workers:
                break
        box = int(min(a.unit_s, st["stop_epoch"] - time.time()))
        argv = [str(Path(a.mem_cap)), "--slot", "--name", "x35sprt", str(a.cap_gb), "nice", "-n", "19",
                "taskset", "-c", a.cpus, py, str(Path(__file__).resolve()), "unit", "--cache", str(cache),
                "--box-s", str(box), "--threads", str(a.threads), *[x for e in engines for x in ("--engine", e)]]
        _log(f"unit {engines}, box {box} s: queued for a heavy slot")
        t0 = time.time()
        rc = subprocess.run(argv, env=env, check=False).returncode
        _log(f"unit {engines}: exit {rc} after {time.time() - t0:.0f} s (slot wait included)")
        if rc not in (0, UNIT_TIME_UP):
            return rc
        st = PR._state(cache)
        cmd_read(argparse.Namespace(cache=str(cache), out=str(cache / "results.json"),
                                    gate_input=str(cache / "r1_input.json")))
    cmd_read(argparse.Namespace(cache=str(cache), out=str(cache / "results.json"),
                                gate_input=str(cache / "r1_input.json")))
    return 0


def cmd_unit(a: argparse.Namespace) -> int:
    """ONE heavy-slot unit (the pre-read's, with this read's family and request names)."""
    cache = Path(a.cache).expanduser()
    st = PR._state(cache)
    py = sys.executable
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    deadline = time.time() + a.box_s
    procs = []
    for spec in a.engine:
        label, labs = spec.split(":", 1)
        argv = [py, "-m", "main.h2h", "play-many", "--players", st["nodes"][label]["copy"], "--opponents",
                *[st["members"][lab]["copy"] for lab in labs.split(",")], "--pairs", str(PAIRS), "--batch-pairs",
                str(BATCH_PAIRS), "--purpose", "ab", "--family", FAMILY, "--request", req_id(label),
                "--out", str(cache / "ledger"), "--label", "x35_sprt", "--device", "cpu",
                "--torch-threads", "1", "--threads", str(a.threads)]
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
    for label, arm, seed, run in FINALS:
        node = st.get("nodes", {}).get(label)
        if node is None:
            rows.append({"label": label, "arm": arm, "seed": seed, "run": run, "s_pp": None, "status": "not started"})
            continue
        try:
            cells = CX.look_cells(cache / "ledger", FAMILY, req_id(label), PROTOCOL)
        except Exception:  # noqa: BLE001 - a final with no row yet
            cells = []
        ps = R.population_score(cells, node["sha256"], mem)
        mine = [c for c in cells if c.player == node["sha256"] and c.opponent in label_of]
        ses = {label_of[c.opponent]: 100.0 * PR._cell_se(c.pairs) for c in mine if c.pairs and sum(c.pairs)}
        row: dict[str, Any] = {"label": label, "arm": arm, "seed": seed, "run": run, "source": node["source"],
                               "step": node["step"], "sha256": node["sha256"], "s_pp": ps.s_pp,
                               "per_member_pp": dict(ps.per_member_pp), "per_member_se_pp": ses,
                               "inconclusive": list(ps.inconclusive),
                               "n_pairs": {label_of[c.opponent]: c.n_pairs for c in mine},
                               "regime_ids": sorted({str(getattr(c, "regime_id", "")) for c in mine})}
        if ps.s_pp is not None and len(ses) == len(mem):
            se = math.sqrt(math.fsum(x * x for x in ses.values())) / len(mem)
            row.update(se_pp=se, ci95_pp=[ps.s_pp - PR.Z95 * se, ps.s_pp + PR.Z95 * se])
        rows.append(row)
    out: dict[str, Any] = {"family": FAMILY, "pairs_per_cell": PAIRS, "batch_pairs": BATCH_PAIRS,
                           "refpop": REFPOP.name, "refpop_content_sha256": rp["content_sha256"],
                           "regime": "main.h2h play-many, greedy vs greedy, mirrored pairs, eval team builder "
                                     "(no panel), cpu, torch_threads 1",
                           "stop_iso": st.get("stop_iso"), "finals": rows}
    done = [r for r in rows if r["s_pp"] is not None and "se_pp" in r and not r["inconclusive"]]
    if len(done) == len(FINALS):
        g1 = json.loads(G1_RESULTS.read_text())
        c0 = {r["seed"]: r["s_pp"] for r in g1["finals"] if r["arm"] == "c0"}
        doc = {"sprt": {r["seed"]: r["s_pp"] for r in done}, "c0": c0}
        out["c0_from"] = {"file": "g1/results.json", "family": g1["family"],
                          "refpop_content_sha256": g1["refpop_content_sha256"]}
        if g1["refpop_content_sha256"] != rp["content_sha256"]:
            raise SystemExit("G1's R is not this read's R: C0's S cannot be reused")
        Path(a.gate_input).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({r["label"]: [r.get("s_pp"), r.get("se_pp"), r.get("n_pairs")] for r in rows}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cache", required=True)
    r.add_argument("--hours", type=float, default=14.0)
    r.add_argument("--threads", type=int, default=2, help="the Rust core's worker threads per engine")
    r.add_argument("--workers", type=int, default=2, help="engines per unit (one cell each at a time)")
    r.add_argument("--mem-cap", default=str(HERE.parents[4] / "scripts" / "ops" / "mem_cap.sh"))
    r.add_argument("--cap-gb", type=int, default=12)
    r.add_argument("--cpus", default="5-7,13-15")
    r.add_argument("--unit-s", type=int, default=2400, help="a unit's time box (one heavy-slot hold), s")
    u = sub.add_parser("unit")
    u.add_argument("--cache", required=True)
    u.add_argument("--box-s", type=int, required=True)
    u.add_argument("--threads", type=int, default=2)
    u.add_argument("--engine", action="append", required=True, help="<final label>:<member>[,<member>...]")
    d = sub.add_parser("read")
    d.add_argument("--cache", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--gate-input", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "unit": cmd_unit, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
