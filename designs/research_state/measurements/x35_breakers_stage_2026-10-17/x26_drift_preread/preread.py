"""X35 AM-2's ZERO-TRAINING VARIANCE PRE-READ (``X35_BATCH1_REVIEW.md`` F-RV-2): X26's own plain continuation past
P* (280M), scored on X35's REGISTERED population score S (``main.x35.readout.population_score``) against the FROZEN
reference population R (``reference_population_production.json``), at the readout's regime: 500 mirrored pairs per
(node, member) cell, ``main.h2h play-many``, greedy vs greedy, the eval team builder (no panel), CPU.

DESCRIPTIVE ONLY: it never enters R1 (X26's pin ``9c033489`` has no T62, its pool is not reseeded, one seed).

    preread.py run  --cache ~/.cache/gen3ai/x26_drift_preread --hours 10     # detached, resumable, a registered stop
    preread.py read --cache ~/.cache/gen3ai/x26_drift_preread --out results.json

``run`` copies each node (and each member of R) into ``<cache>/ckpts`` with its sha256 checked against the source
(and, for members, against the frozen R), registers ONE family in ``<cache>/ledger`` (never ``models/``), and plays
the nodes in a SPREAD order (so any prefix covers the range). The driver itself holds NO heavy slot: each UNIT -- one
node's unfinished cells on two engines, TIME-BOXED to 40 min -- takes a heavy slot through ``mem_cap.sh --slot`` and
releases it at the box's end (orchestrator 2026-10-20: a slot is held per bounded chunk, never per run, so jobs
interleave). A cell's 500 pairs are recorded in 100-pair batches (the resume grain; a box's in-flight batch is lost
and its claim voided). A re-run resumes from the ledger. No unit starts past the stop time written to
``<cache>/state.json`` at the first start, and a unit's box never runs past it.

``read`` builds S per node with the kit's own ``population_score`` over the family's cells (``main.h2h.cross.
look_cells``). The per-node interval is DECLARED HERE (the readout defines no per-final interval): the six cells are
independent, each with its pair-level standard error (``mirrored_pairs.pair_score_ci``'s variance), so
se(S) = sqrt(sum se_m^2) / 6 and the interval is S +- 1.96 se(S).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
REFPOP = STAGE / "reference_population_production.json"
#: R before ruling C5 (2026-10-19 23:51 PDT) added T*: the pre-read began on it, and reports S over it too
REFPOP_PRE_C5 = STAGE / "reference_population_production_PRE_C5.json"
RUN_CKPTS = Path("/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints")
FAMILY = "x26_drift_preread_2026-10-19"
PROTOCOL = "gen3_eval_protocol_v1_h2h"
RULE = "X35_BATCH1_REVIEW.md F-RV-2 zero-training pre-read (descriptive; population_score of design_compute_efficiency §5.6.4)"
PAIRS = 500
#: pairs per ledger row (resume granularity); the request spec carries it, so the requests are named for it
BATCH_PAIRS = 100
#: the spread order: every prefix covers 290-380M as evenly as it can
ORDER_M = (290, 380, 330, 310, 350, 300, 370, 320, 360, 340)
STRIDE = 10_000_000
CKPT_INTERVAL = 1_000_000
Z95 = 1.959963985


def req_id(nominal_m: int) -> str:
    return f"x26pre_{nominal_m}m_b{BATCH_PAIRS}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def node_path(nominal_m: int) -> Path:
    """The ONE checkpoint of the nominal node (refpop's node rule: M <= step < M + 1M)."""
    lo = nominal_m * 1_000_000
    hits = [p for p in RUN_CKPTS.glob("checkpoint_*_steps.zip")
            if lo <= int(p.name.split("_")[1]) < lo + CKPT_INTERVAL]
    if len(hits) != 1:
        raise SystemExit(f"node {nominal_m}M: {len(hits)} checkpoints ({hits})")
    return hits[0]


def copy_checked(src: Path, dst_dir: Path, want_sha: str | None = None) -> tuple[Path, str]:
    """Copy ``src`` (read-only) into ``dst_dir``; the copy's sha256 must equal the source's (and ``want_sha``)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    src_sha = sha256_file(src)
    if want_sha is not None and src_sha != want_sha:
        raise SystemExit(f"{src}: sha256 {src_sha} differs from the frozen {want_sha}")
    if not dst.exists() or sha256_file(dst) != src_sha:
        tmp = dst.with_suffix(".part")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    got = sha256_file(dst)
    if got != src_sha:
        raise SystemExit(f"{dst}: the copy's sha256 {got} differs from the source's {src_sha}")
    # the arch provenance the resolver needs beside the zip: the checkpoint's own sidecar and its run's
    # model_config.json (looked up as main.h2h does: the zip's dir, then that dir's parent)
    side = src.with_suffix(".json")
    if side.is_file():
        shutil.copyfile(side, dst_dir / side.name)
    for d in (src.parent, src.parent.parent):
        if (d / "model_config.json").is_file():
            shutil.copyfile(d / "model_config.json", dst_dir / "model_config.json")
            break
    else:
        raise SystemExit(f"{src}: no model_config.json beside it or in its run dir")
    return dst, got


def members(path: Path = REFPOP) -> list[dict[str, str]]:
    return [dict(m) for m in json.loads(path.read_text())["members"]]


def _state(cache: Path) -> dict[str, Any]:
    p = cache / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_state(cache: Path, st: dict[str, Any]) -> None:
    tmp = cache / "state.json.part"
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
    os.replace(tmp, cache / "state.json")


def _log(msg: str) -> None:
    print(f"[preread {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def cmd_run(a: argparse.Namespace) -> int:
    cache = Path(a.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    st = _state(cache)
    if "stop_epoch" not in st:
        st["started_epoch"] = time.time()
        st["stop_epoch"] = time.time() + 3600.0 * a.hours
        st["stop_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st["stop_epoch"]))
        _save_state(cache, st)
    _log(f"registered stop {st['stop_iso']} (epoch {st['stop_epoch']:.0f})")
    ledger = cache / "ledger"
    py = sys.executable
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run([py, "-m", "main.eval_ledger", "family-register", "--root", str(ledger), "--id", FAMILY,
                    "--decision-kind", "ab_verdict", "--rule", RULE, "--protocol", PROTOCOL], check=True, env=env)
    mem_paths = []
    st.setdefault("members", {})
    for m in members():
        dst, sha = copy_checked(Path(m["path"]), cache / "ckpts" / "members" / m["label"], m["sha256"])
        st["members"][m["label"]] = {"copy": str(dst), "sha256": sha, "source": m["path"]}
        mem_paths.append(str(dst))
    _save_state(cache, st)
    st.setdefault("nodes", {})
    labels = [m["label"] for m in members()]
    while True:
        # every node's unfinished cells against the CURRENT R, in the spread order; a node is copied when first
        # reached (a C5-style re-freeze that adds a member reopens the nodes already played, which is intended)
        pending: list[tuple[int, list[str]]] = []
        for nm in ORDER_M:
            if str(nm) not in st["nodes"]:
                if pending:
                    break
                src = node_path(nm)
                dst, sha = copy_checked(src, cache / "ckpts" / "nodes")
                st["nodes"][str(nm)] = {"copy": str(dst), "sha256": sha, "source": str(src)}
                _save_state(cache, st)
            have = cell_pairs(cache, nm)
            left = [lab for lab in labels if have.get(lab, 0) < PAIRS]
            if left:
                pending.append((nm, left))
        if not pending:
            break
        if time.time() + 60.0 >= st["stop_epoch"]:
            _log(f"STOP at the registered stop time, before {pending}")
            break
        engines: list[str] = []
        for nm, left in pending:
            k = min(len(left), a.workers - len(engines))
            engines += [f"{nm}:" + ",".join(left[w::k]) for w in range(k)]
            if len(engines) >= a.workers:
                break
        box = int(min(a.unit_s, st["stop_epoch"] - time.time()))
        argv = [str(Path(a.mem_cap)), "--slot", "--name", "x26pre", str(a.cap_gb), "nice", "-n", "19",
                "taskset", "-c", a.cpus, py, str(Path(__file__).resolve()), "unit", "--cache", str(cache),
                "--box-s", str(box), "--threads", str(a.threads), *[x for e in engines for x in ("--engine", e)]]
        _log(f"unit {engines}, box {box} s: queued for a heavy slot")
        t0 = time.time()
        rc = subprocess.run(argv, env=env, check=False).returncode
        _log(f"unit {engines}: exit {rc} after {time.time() - t0:.0f} s (slot wait included)")
        if rc not in (0, UNIT_TIME_UP):
            return rc
        st = _state(cache)
        cmd_read(argparse.Namespace(cache=str(cache), out=str(cache / "results.json")))
    cmd_read(argparse.Namespace(cache=str(cache), out=str(cache / "results.json")))
    return 0


#: a unit's exit when its time box ran out (its engines were stopped; their recorded batches stay)
UNIT_TIME_UP = 124


def cell_pairs(cache: Path, nm: int) -> dict[str, int]:
    """Completed pairs per member label in node ``nm``'s request (from the ledger)."""
    from main.h2h import cross as CX

    st = _state(cache)
    label_of = {v["sha256"]: k for k, v in st["members"].items()}
    try:
        cells = CX.look_cells(cache / "ledger", FAMILY, req_id(nm), PROTOCOL)
    except Exception as exc:  # noqa: BLE001 - no row yet reads as nothing played
        _log(f"node {nm}M: no cells read ({type(exc).__name__}: {exc})")
        return {}
    sha = st["nodes"][str(nm)]["sha256"]
    return {label_of[c.opponent]: c.n_pairs for c in cells if c.player == sha and c.opponent in label_of}


def cmd_unit(a: argparse.Namespace) -> int:
    """ONE heavy-slot unit, time-boxed: one engine (torch_threads 1) per ``--engine <node>:<label>,<label>...``,
    each playing its cells in turn; at the box's end the engines are stopped (SIGTERM; their recorded 100-pair
    batches stay, an in-flight batch is lost and its claim voided on the next claim)."""
    cache = Path(a.cache).expanduser()
    st = _state(cache)
    py = sys.executable
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    deadline = time.time() + a.box_s
    procs = []
    for spec in a.engine:
        nm_s, labs = spec.split(":", 1)
        share = labs.split(",")
        argv = [py, "-m", "main.h2h", "play-many", "--players", st["nodes"][nm_s]["copy"], "--opponents",
                *[st["members"][lab]["copy"] for lab in share], "--pairs", str(PAIRS), "--batch-pairs",
                str(BATCH_PAIRS), "--purpose", "ab", "--family", FAMILY, "--request", req_id(int(nm_s)),
                "--out", str(cache / "ledger"), "--label", "x26_drift_preread", "--device", "cpu",
                "--torch-threads", "1", "--threads", str(a.threads)]
        logf = open(cache / f"play_{nm_s}m.log", "a")  # noqa: SIM115 - closed when the child is reaped
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


def _cell_se(pairs: Sequence[int]) -> float:
    n = sum(pairs)
    mu = sum(c * i for i, c in enumerate(pairs)) / (4.0 * n)
    var = sum(c * (i / 4.0 - mu) ** 2 for i, c in enumerate(pairs)) / n
    return math.sqrt(var / n)


def _sd(xs: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = math.fsum(xs) / len(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _chi2_sf(x: float, df: int) -> float:
    from scipy.stats import chi2

    return float(chi2.sf(x, df))


def read_over(cache: Path, path: Path) -> dict[str, Any]:
    """S per node over the members of the R in ``path`` (the kit's ``population_score``), and the spread summary."""
    from main.h2h import cross as CX
    from main.x35 import readout as R

    st = _state(cache)
    mem = {m["label"]: m["sha256"] for m in members(path)}
    label_of = {v: k for k, v in mem.items()}
    rows = []
    for nm in sorted(int(k) for k in st.get("nodes", {})):
        node = st["nodes"][str(nm)]
        try:
            cells = CX.look_cells(cache / "ledger", FAMILY, req_id(nm), PROTOCOL)
        except Exception:  # noqa: BLE001 - a node with no row yet
            cells = []
        ps = R.population_score(cells, node["sha256"], mem)
        mine = [c for c in cells if c.player == node["sha256"] and c.opponent in label_of]
        ses = {label_of[c.opponent]: 100.0 * _cell_se(c.pairs) for c in mine if c.pairs and sum(c.pairs)}
        row: dict[str, Any] = {"nominal_m": nm, "source": node["source"], "sha256": node["sha256"],
                               "s_pp": ps.s_pp, "per_member_pp": dict(ps.per_member_pp),
                               "per_member_se_pp": ses, "inconclusive": list(ps.inconclusive),
                               "n_pairs": {label_of[c.opponent]: c.n_pairs for c in mine}}
        if ps.s_pp is not None and len(ses) == len(mem):
            se = math.sqrt(math.fsum(x * x for x in ses.values())) / len(mem)
            row.update(se_pp=se, ci95_pp=[ps.s_pp - Z95 * se, ps.s_pp + Z95 * se])
        rows.append(row)
    done = [r for r in rows if r["s_pp"] is not None and "se_pp" in r]
    xs = [float(r["nominal_m"]) for r in done]
    ys = [float(r["s_pp"]) for r in done]
    summ: dict[str, Any] = {"n_nodes": len(done), "nodes_m": [r["nominal_m"] for r in done]}
    sd = _sd(ys)
    if sd is not None:
        mse = math.fsum(r["se_pp"] ** 2 for r in done) / len(done)
        my = math.fsum(ys) / len(ys)
        q = math.fsum((r["s_pp"] - my) ** 2 / r["se_pp"] ** 2 for r in done)
        summ.update(mean_s_pp=my, sd_s_pp=sd, mean_meas_se_pp=math.sqrt(mse),
                    sd_beyond_meas_pp=math.sqrt(max(0.0, sd * sd - mse)), sd_ratio_to_meas=sd / math.sqrt(mse),
                    meas_only_chi2=q, meas_only_df=len(done) - 1, meas_only_p=_chi2_sf(q, len(done) - 1))
    if len(done) >= 3:
        mx, my = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
        sxx = math.fsum((x - mx) ** 2 for x in xs)
        b = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        res = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
        rsd = math.sqrt(math.fsum(r * r for r in res) / (len(res) - 2))
        summ.update(slope_pp_per_100m=100.0 * b, slope_se_pp_per_100m=100.0 * rsd / math.sqrt(sxx),
                    residual_sd_pp=rsd)
    return {"refpop": path.name, "content_sha256": json.loads(path.read_text())["content_sha256"],
            "members": sorted(mem), "nodes": rows, "summary": summ}


def cmd_read(a: argparse.Namespace) -> int:
    cache = Path(a.cache).expanduser()
    st = _state(cache)
    out = {"family": FAMILY, "pairs_per_cell": PAIRS, "batch_pairs": BATCH_PAIRS,
           "regime": "main.h2h play-many, greedy vs greedy, mirrored pairs, eval team builder (no panel), cpu, "
                     "torch_threads 1",
           "stop_iso": st.get("stop_iso"), "stop_history": st.get("stop_history", []),
           "registered": read_over(cache, REFPOP), "pre_c5": read_over(cache, REFPOP_PRE_C5),
           "assumed_sigma_bracket_pp": [0.8, 1.6], "am2_gate_pp": 1.2}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k]["summary"] for k in ("registered", "pre_c5")}, indent=1, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cache", required=True)
    r.add_argument("--hours", type=float, default=10.0)
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
    u.add_argument("--engine", action="append", required=True, help="<node M>:<label>[,<label>...]")
    d = sub.add_parser("read")
    d.add_argument("--cache", required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "unit": cmd_unit, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
