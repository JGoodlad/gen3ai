"""THE X14 A/B READ (staged 2026-10-20): the throughput of `--overlap-collect` ON vs OFF, from the two forks' own
TensorBoard rows, plus the K9(b) section-7 ceiling read of each arm. Shared by `x14_cell.py` (writes the numbers into
its cell json right after the run) and `read_battery.py` (prints the verdict). Read-only on the run directories; CPU.

THE UNIT OF MEASUREMENT IS ONE DUMP. The trainer logs once per rollout, at the end of the iteration (``rollout/collect_ms``
is present at every dump; ``train/train_ms`` is the update that closed the cycle). A restart (a crash auto-restart, a
periodic restart) writes a NEW event file: ONE SEGMENT PER PROCESS (`main.x35.tbrows`). Inside a process:

* **The first 3 dumps are dropped** (`BURN_IN`, X14 README 3.1: "the updates after the 3rd per process"; the ON arm's
  first update runs alone by design, and the first rollout of a process carries its warm-up). The kept dumps are k >= 4.
* **fps is the WALL fps of the cycle**: ``(step_k - step_{k-1}) / (wall_k - wall_{k-1})`` from the event wall times, so it
  needs no cumulative state. ``time/fps`` is NOT used to decide: SB3 logs it as ``(num_timesteps - at_start) / elapsed``
  since the process's `learn()` began (the archived 2026-10-20 OFF arm reads 1820, 923, 803, ... 675 over 11 dumps), so
  at 15 dumps it is dominated by the first rollouts and the warm-up. Its median over the kept dumps is printed beside the
  wall fps (the README's number), never decided on.
* **The decision**: ``ratio = median wall fps (ON) / median wall fps (OFF)``; bar >= x1.10 for "worth a learning A/B"
  (`BAR`). A ratio within 1e-9 relative of the bar is AT_BAR, never a pass or a fail (standing rule 8). Fewer than
  `MIN_CYCLES` kept dumps in either arm is INCOMPLETE. The 25-75 % band of each arm's per-dump fps is printed; a band
  that straddles the bar is said so.

THE K9(b) CEILING READ (X35 AM-6 = X26 Amendment 3, `main.x35.batch.judge_k9b`, the SAME function and ceilings): over the
fork's OWN rows (steps > the fork step, ALL of them, burn-in included -- it is a safety read), a judged max >= 1e-3, a
p99 >= 1e-4, any non-finite value or the child log's NON-FINITE K9(b) line FIRES (verdict VOID = FLAG); a value within 1e-9
relative of a ceiling is recorded and does not fire; no row is INCONCLUSIVE, never a pass. The OFF arm reads the learner's
``behaviour/*`` tags; the ON arm reads ``behaviour_snapshot/*`` (the rows were played by the previous version; the learner's
own probe has almost no current rows) and ``behaviour/*`` too where present. ``--behaviour-check warn`` means REPORT, never
abort: a flag does not HOLD the battery.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

#: X14 README 3.1.
BURN_IN = 3
MIN_CYCLES = 6
BAR = 1.10
RULE8_EPS = 1e-9
PARENT_STEP = 380_000_195
ROWS_PER_UPDATE = 98_304
UPDATES = 15
TARGET_STEPS = PARENT_STEP + UPDATES * ROWS_PER_UPDATE            # 381,474,755: ABSOLUTE, 15 updates past the parent

DUMP_TAG = "rollout/collect_ms"
CYCLE_TAGS = ("rollout/collect_ms", "train/train_ms", "time/fps", "lifecycle/cuda_update_peak_reserved_mib")
OVERLAP_TAGS = ("overlap/play_ms", "overlap/join_wait_ms", "overlap/prox_forward_ms", "overlap/is_ess",
                "overlap/is_weight_truncated_frac", "overlap/behaviour_kl",
                "behaviour_snapshot/excluded_frac", "behaviour_snapshot/max_abs_dlogp_judged")
K9B_FAMILIES = {
    "behaviour": {"judged_max": "behaviour/max_abs_dlogp_judged", "p99": "behaviour/p99_abs_dlogp_current"},
    "behaviour_snapshot": {"judged_max": "behaviour_snapshot/max_abs_dlogp_judged",
                           "p99": "behaviour_snapshot/p99_abs_dlogp_current"},
}
K9B_REQUIRED = {"off": "behaviour", "on": "behaviour_snapshot"}
T2_GROWTH_NEEDLE = "grew the CUDA reservation"
CHILD_LOG = "launcher_child.full.log"

Row = tuple[int, float, float]                  # (step, value, wall_time)


def _ensure_src() -> None:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


# ------------------------------------------------------------------------------------------------- TB rows

def own_segments(run_dir: str, tags: Sequence[str]) -> tuple[int | None, list[dict[str, Any]]]:
    """``(fork_step, segments)``: the run's OWN event files (the inherited parent prefix excluded), one per process,
    chronological. A segment is ``{"path", "first_wall", "rows": {tag: [(step, value, wall)]}}`` with every row at a
    step > the fork step; a tag a file lacks is absent. An unknown fork step is returned as None (the caller refuses)."""
    _ensure_src()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from main.x35.tbrows import fork_step, own_event_files

    fork = fork_step(run_dir)
    segs: list[dict[str, Any]] = []
    for f in own_event_files(run_dir):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        avail = set(ea.Tags().get("scalars", []))
        rows: dict[str, list[Row]] = {}
        first = math.inf
        for t in tags:
            if t not in avail:
                continue
            evs = ea.Scalars(t)
            first = min([first] + [e.wall_time for e in evs])
            rows[t] = sorted((int(e.step), float(e.value), float(e.wall_time)) for e in evs
                             if fork is None or int(e.step) > fork)
        if rows:
            segs.append({"path": f, "first_wall": first, "rows": rows})
    segs.sort(key=lambda s: (s["first_wall"], s["path"]))
    return fork, segs


def _quantile(xs: Sequence[float], p: float) -> float:
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = p * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _stats(xs: Sequence[float]) -> dict[str, float] | None:
    xs = [float(x) for x in xs if math.isfinite(x)]
    if not xs:
        return None
    return {"n": len(xs), "median": statistics.median(xs), "q25": _quantile(xs, 0.25), "q75": _quantile(xs, 0.75),
            "min": min(xs), "max": max(xs)}


def kept_dumps(segments: Sequence[Mapping[str, Any]], burn_in: int = BURN_IN) -> list[dict[str, Any]]:
    """The measured dumps (k >= burn_in + 1 within each process), each with its cycle wall fps and every cycle /
    overlap tag's value at that step. Pure over the segments."""
    out: list[dict[str, Any]] = []
    for si, seg in enumerate(segments):
        rows = seg["rows"]
        dumps = rows.get(DUMP_TAG) or []
        by_tag = {t: {s: v for s, v, _ in rs} for t, rs in rows.items()}
        for k, (step, _v, wall) in enumerate(dumps, 1):
            if k <= burn_in:
                continue
            pstep, _pv, pwall = dumps[k - 2]
            dwall = wall - pwall
            if dwall <= 0 or step <= pstep:
                continue
            rec: dict[str, Any] = {"segment": si, "k": k, "step": step, "cycle_s": dwall,
                                   "fps_wall": (step - pstep) / dwall}
            for t in (*CYCLE_TAGS, *OVERLAP_TAGS):
                if step in by_tag.get(t, {}):
                    rec[t] = by_tag[t][step]
            out.append(rec)
    return out


# ------------------------------------------------------------------------------------------------- K9(b)

def k9b_read(run_dir: str, arm: str, fork: int | None, segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """AM-6's ceilings over the fork's own rows, one family per tag prefix (module doc). ``verdict`` is VOID (a FLAG)
    iff ANY family fires; otherwise the REQUIRED family's verdict (PASS | INCONCLUSIVE: no row). Errors in the
    import or the read are recorded as ``UNREAD``, never raised."""
    try:
        _ensure_src()
        from main.x35 import batch as XB

        log = os.path.join(run_dir, CHILD_LOG)
        text = Path(log).read_text(errors="replace") if os.path.isfile(log) else ""
        required = K9B_REQUIRED[arm]
        fams: dict[str, Any] = {}
        for fam, tags in K9B_FAMILIES.items():
            series = {key: [(s, v) for seg in segments for (s, v, _w) in seg["rows"].get(tag, [])]
                      for key, tag in tags.items()}
            k = XB.judge_k9b(os.path.basename(os.path.normpath(run_dir)), fork, series, text if fam == required else "")
            fams[fam] = k.as_dict()
        fired = [f for f, d in fams.items() if d["verdict"] == XB.VOID]
        verdict = XB.VOID if fired else fams[required]["verdict"]
        over = {fam: sum(1 for seg in segments for (_s, v, _w) in seg["rows"].get(tags["judged_max"], [])
                         if math.isfinite(v) and v >= 1e-4) for fam, tags in K9B_FAMILIES.items()}
        return {"verdict": verdict, "required_family": required, "flag": verdict != XB.PASS,
                "fired_families": fired, "ceilings": dict(XB.K9B_CEILINGS), "families": fams,
                "n_judged_at_or_over_1e-4": over}
    except Exception as exc:  # noqa: BLE001 -- a read error is a FLAG, never a failed cell
        return {"verdict": "UNREAD", "flag": True, "error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------------------------------------- one arm

def summarise(run_dir: str, arm: str) -> dict[str, Any]:
    """Everything the A/B reads off ONE arm's run directory (read-only)."""
    tags = list(dict.fromkeys([*CYCLE_TAGS, *OVERLAP_TAGS,
                               *[t for f in K9B_FAMILIES.values() for t in f.values()]]))
    fork, segs = own_segments(run_dir, tags)
    kept = kept_dumps(segs)
    all_dumps = [d for s in segs for d in (s["rows"].get(DUMP_TAG) or [])]
    per_tag = {t: _stats([r[t] for r in kept if t in r]) for t in (*CYCLE_TAGS, *OVERLAP_TAGS)}
    overlap_rows = sum(len(s["rows"].get("overlap/join_wait_ms", [])) for s in segs)
    growth = 0
    log = os.path.join(run_dir, CHILD_LOG)
    if os.path.isfile(log):
        growth = Path(log).read_text(errors="replace").count(T2_GROWTH_NEEDLE)
    pins: list[str] = []
    try:
        meta = json.loads((Path(run_dir) / "metadata.json").read_text())
        pins = [str(h.get("git_hash")) for h in meta.get("pin_history") or []]
    except (OSError, ValueError):
        pass
    return {"arm": arm, "run_dir": run_dir, "fork_step": fork, "n_segments": len(segs), "n_dumps": len(all_dumps),
            "max_step": max([d[0] for d in all_dumps] or [0]), "burn_in": BURN_IN, "n_cycles": len(kept),
            "fps_wall": _stats([r["fps_wall"] for r in kept]), "cycle_s": _stats([r["cycle_s"] for r in kept]),
            "tags": per_tag, "overlap_rows": overlap_rows, "t2_growth_warnings": growth, "pin_history": pins,
            "k9b": k9b_read(run_dir, arm, fork, segs)}


def arm_problems(s: Mapping[str, Any], arm: str, pin: str | None) -> list[str]:
    """Why a finished arm is NOT the arm that was specified (empty = it is): the target steps, the overlap rows
    (ON must have them, OFF none), the pin the run recorded."""
    p: list[str] = []
    if s["fork_step"] != PARENT_STEP:
        p.append(f"fork step {s['fork_step']} is not the parent checkpoint's {PARENT_STEP}")
    if s["max_step"] < TARGET_STEPS:
        p.append(f"the run stopped at step {s['max_step']:,} < the target {TARGET_STEPS:,}")
    if arm == "on" and s["overlap_rows"] <= 0:
        p.append("the ON arm logged no overlap/join_wait_ms row: it never overlapped (a vacuous arm)")
    if arm == "off" and s["overlap_rows"] > 0:
        p.append("the OFF arm logged overlap/* rows: it overlapped (the flag leaked in)")
    if pin and (not s["pin_history"] or any(h != pin for h in s["pin_history"])):
        p.append(f"the run recorded pin_history {s['pin_history']}, not the battery's pin {pin}")
    return p


# ------------------------------------------------------------------------------------------------- the decision

def decide(off: Mapping[str, Any] | None, on: Mapping[str, Any] | None) -> dict[str, Any]:
    """The X14 throughput verdict (module doc). ``off`` / ``on`` are `summarise` results (or None = not run)."""
    miss = []
    for name, s in (("x14_off", off), ("x14_on", on)):
        if not s:
            miss.append(f"{name}: no result")
        elif not s.get("fps_wall") or s["n_cycles"] < MIN_CYCLES:
            miss.append(f"{name}: {s.get('n_cycles', 0)} kept dumps < {MIN_CYCLES}")
    if miss or not off or not on:
        return {"verdict": "INCOMPLETE", "missing": miss}
    ratio = on["fps_wall"]["median"] / off["fps_wall"]["median"]
    lo = on["fps_wall"]["q25"] / off["fps_wall"]["q75"]
    hi = on["fps_wall"]["q75"] / off["fps_wall"]["q25"]
    at_bar = abs(ratio - BAR) <= RULE8_EPS * BAR
    verdict = "AT_BAR" if at_bar else "CLEARS" if ratio >= BAR else "BELOW"
    logged = None
    a, b = (on["tags"].get("time/fps") or {}).get("median"), (off["tags"].get("time/fps") or {}).get("median")
    if a and b:
        logged = a / b
    t = {k: (off["tags"].get(k) or {}).get("median") for k in ("train/train_ms", "rollout/collect_ms")}
    hide = None
    if t["train/train_ms"] and t["rollout/collect_ms"]:
        hide = min(t.values()) / (t["train/train_ms"] + t["rollout/collect_ms"])    # type: ignore[type-var]
    flags = [f"{n}: K9(b) {s['k9b']['verdict']}" for n, s in (("x14_off", off), ("x14_on", on)) if s["k9b"].get("flag")]
    return {"verdict": verdict, "ratio": ratio, "ratio_band": [lo, hi], "band_straddles_bar": lo < BAR <= hi,
            "bar": BAR, "ratio_time_fps_logged": logged, "off_max_hideable_share": hide,
            "k9b_flags": flags, "cycles": {"off": off["n_cycles"], "on": on["n_cycles"]}}


# ------------------------------------------------------------------------------------------------- selftest

def _write_run(root: Path, arm: str, procs: Sequence[Sequence[tuple[int, float, dict[str, float]]]],
               fork: int = PARENT_STEP, pin: str = "p" * 40, name: str | None = None) -> str:
    """A synthetic fork run dir: ``procs`` = per process a list of ``(step, wall, {tag: value})`` dumps."""
    from torch.utils.tensorboard import SummaryWriter

    rd = root / (name or f"rb_synth_{arm}")
    (rd / "tb").mkdir(parents=True)
    (rd / "tb" / "INHERITED_FROM.json").write_text(json.dumps({"fork_step": fork}))
    (rd / "metadata.json").write_text(json.dumps({"pin_history": [{"git_hash": pin}]}))
    for pi, dumps in enumerate(procs):
        w = SummaryWriter(log_dir=str(rd / "tb"), filename_suffix=f".proc{pi}")
        for step, wall, tags in dumps:
            for t, v in tags.items():
                w.add_scalar(t, v, global_step=step, walltime=wall)
        w.close()
    return str(rd)


def selftest() -> str:
    """CPU check of the reader on SYNTHETIC runs (no GPU, no real run): burn-in is per PROCESS, the ratio is the wall
    fps' median ratio and not the cumulative ``time/fps``, a ceiling fires, a non-finite value fires, the vacuity
    checks work, and a ratio at the bar is AT_BAR. Raises on any miss."""
    nan = float("nan")

    def proc(t0: float, n: int, cyc: float, slow: float, step0: int, extra: dict[str, float], k9: float,
             snap: float | None) -> list[tuple[int, float, dict[str, float]]]:
        out, t, step = [], t0, step0
        for k in range(1, n + 1):
            dt = slow if k <= 3 else cyc                      # the first 3 dumps of a process are slow (warm-up)
            t += dt
            step += ROWS_PER_UPDATE
            tags = {"rollout/collect_ms": 30000.0, "train/train_ms": 40000.0, "time/fps": 500.0 + k,
                    "lifecycle/cuda_update_peak_reserved_mib": 8300.0, **extra,
                    "behaviour/max_abs_dlogp_judged": k9, "behaviour/p99_abs_dlogp_current": 1e-6}
            if snap is not None:
                tags["behaviour_snapshot/max_abs_dlogp_judged"] = snap
                tags["behaviour_snapshot/p99_abs_dlogp_current"] = 1e-6
            out.append((step, t, tags))
        return out

    with tempfile.TemporaryDirectory(prefix="x14_selftest_") as td:
        root = Path(td)
        off_dir = _write_run(root, "off", [proc(1000.0, 8, 100.0, 300.0, PARENT_STEP, {}, 5e-5, None),
                                           proc(9000.0, 7, 100.0, 300.0, PARENT_STEP + 8 * ROWS_PER_UPDATE, {}, 5e-5,
                                                None)])
        on_dir = _write_run(root, "on", [proc(1000.0, 15, 80.0, 300.0, PARENT_STEP,
                                              {"overlap/join_wait_ms": 1.0, "overlap/play_ms": 50000.0}, 5e-5, 5e-5)])
        off, on = summarise(off_dir, "off"), summarise(on_dir, "on")
        assert off["n_segments"] == 2 and on["n_segments"] == 1, (off["n_segments"], on["n_segments"])
        assert off["n_cycles"] == (8 - 3) + (7 - 3) == 9 and on["n_cycles"] == 12, (off["n_cycles"], on["n_cycles"])
        # steps per cycle = ROWS_PER_UPDATE, so the wall fps is exactly rows / seconds, the burn-in dumps excluded
        assert abs(off["fps_wall"]["median"] - ROWS_PER_UPDATE / 100.0) < 1e-6, off["fps_wall"]
        assert abs(on["fps_wall"]["median"] - ROWS_PER_UPDATE / 80.0) < 1e-6, on["fps_wall"]
        d = decide(off, on)
        assert d["verdict"] == "CLEARS" and abs(d["ratio"] - 1.25) < 1e-9, d
        assert d["ratio_time_fps_logged"] is not None and abs(d["ratio_time_fps_logged"] - 1.0) < 0.1, d
        assert off["k9b"]["verdict"] == "PASS" and on["k9b"]["verdict"] == "PASS", (off["k9b"], on["k9b"])
        assert not arm_problems(off, "off", "p" * 40) and not arm_problems(on, "on", "p" * 40), (
            arm_problems(off, "off", "p" * 40), arm_problems(on, "on", "p" * 40))
        assert any("stopped at step" in w for w in arm_problems({**on, "max_step": TARGET_STEPS - 1}, "on", None))
        # vacuity: the ON arm without overlap rows, the OFF arm with them, a wrong pin
        assert any("never overlapped" in w for w in arm_problems({**on, "overlap_rows": 0}, "on", None))
        assert any("leaked" in w for w in arm_problems({**off, "overlap_rows": 3}, "off", None))
        assert any("pin_history" in w for w in arm_problems(on, "on", "q" * 40))
        # a ceiling: one judged max of 2e-3 in the ON arm's SNAPSHOT family fires; a NaN p99 fires
        bad_dir = _write_run(root, "bad", [proc(1000.0, 15, 80.0, 300.0, PARENT_STEP, {"overlap/join_wait_ms": 1.0},
                                                5e-5, 2e-3)])
        bad = summarise(bad_dir, "on")
        assert bad["k9b"]["verdict"] == "VOID" and "behaviour_snapshot" in bad["k9b"]["fired_families"], bad["k9b"]
        nan_dir = _write_run(root, "nan", [proc(1000.0, 15, 80.0, 300.0, PARENT_STEP, {"overlap/join_wait_ms": 1.0},
                                                5e-5, nan)])
        assert summarise(nan_dir, "on")["k9b"]["verdict"] == "VOID", "a non-finite value must fire"
        # the OFF arm's required family is `behaviour`; the absent snapshot family is not a flag, and 9.9e-4 is under 1e-3
        under_dir = _write_run(root, "under", [proc(1000.0, 15, 100.0, 300.0, PARENT_STEP, {}, 9.9e-4, None)])
        under = summarise(under_dir, "off")["k9b"]
        assert under["verdict"] == "PASS" and not under["fired_families"], under
        assert under["n_judged_at_or_over_1e-4"]["behaviour"] == 15, under
        # no post-fork row: absence is INCONCLUSIVE, never a pass
        empty_dir = _write_run(root, "empty", [[(PARENT_STEP, 5000.0, {"rollout/collect_ms": 1.0})]])
        assert summarise(empty_dir, "off")["k9b"]["verdict"] == "INCONCLUSIVE"
    base = {"fps_wall": {"median": 1.0, "q25": 1.0, "q75": 1.0}, "n_cycles": 12, "k9b": {"flag": False},
            "tags": {}}
    at = decide({**base}, {**base, "fps_wall": {"median": BAR, "q25": BAR, "q75": BAR}})
    below = decide({**base}, {**base, "fps_wall": {"median": BAR - 1e-6, "q25": 1.0, "q75": 1.2}})
    assert at["verdict"] == "AT_BAR" and below["verdict"] == "BELOW" and below["band_straddles_bar"], (at, below)
    assert decide({**base, "n_cycles": 5}, base)["verdict"] == "INCOMPLETE"
    return ("x14_read ok: burn-in per process (9 + 12 kept dumps), wall-fps ratio 1.25 (not the cumulative time/fps), "
            "ceiling / non-finite / empty / under-ceiling K9(b), the vacuity checks and AT_BAR")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        print(selftest())
    elif len(sys.argv) > 2:
        print(json.dumps(summarise(sys.argv[1], sys.argv[2]), indent=1, default=str))
    else:
        print("usage: x14_read.py --selftest | <run_dir> <off|on>")
