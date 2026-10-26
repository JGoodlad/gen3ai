"""The canary logger: every canary record of every LIVE run, one line each, plus the watch alerts.

Replaces ``canary_log_v*.sh`` + ``live_runs.py`` + ``aw_files.sh``. Its class bugs:

* **Name globs missed three run families** (v1-v11): the runs come from :func:`live.discover` (launcher argv /
  trainer children) plus every run already in the seen file — never a pattern.
* **The seen file was keyed by run NAME and a row count**: a run dir re-created under the same name with fewer
  rows was skipped until it overtook the old count. The key here is the canary file's ``(path, inode)``.
* **Bars by run-name prefix** (``rb_x35t_x47`` or else C0's) silently applied C0's bars to every new family: this
  logger applies NO family bar. It logs each record's raw readings (compiled / eager, their ratio, the
  same-precision gap) and alerts only on a NON-PASS record whose worst reading exceeds ``alert_above`` (one rule, no
  family); the bars belong to the reader that decides.
* **A bounded loop lapsed silently** (v9, after its 24 h): :func:`watch` writes an ``expiring`` line on its last
  pass and a heartbeat file every pass.
* Alerts: a live launcher whose run is UNRESOLVED; a run with NO canary record ``first_record_min`` after its
  launcher started (default 60: a production cold start reached its first record at ~34 min, 40 was too close); a
  newest record older than ``stale_h``. Each alert fires once per case.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from main.ops.arm_chain import live


def _now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _get(d: Any, *ks: str) -> Any:
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def format_row(run: str, r: Dict[str, Any], alert_above: float) -> Tuple[str, bool]:
    """(the log line, whether it alerts)."""
    m = _get(r, "watch", "scheduled", "clip_margin", "max_abs_dlogp") or {}
    sp = _get(r, "watch", "scheduled", "clip_margin", "same_precision", "max_abs_dlogp")
    c, e = m.get("compiled"), m.get("eager")
    vals = [v for v in (c, e) if isinstance(v, (int, float))]
    ratio = round(c / e, 3) if isinstance(c, (int, float)) and isinstance(e, (int, float)) and e else None
    verdict = str(r.get("verdict"))
    line = (f"{run} u{r.get('update')} @{r.get('num_timesteps')} {verdict} compiled {c} eager {e} ratio {ratio} "
            f"same-precision {sp}")
    hi = max(vals) if vals else None
    alerts = verdict.lower() != "pass" and (hi is None or hi > alert_above)
    return line, alerts


@dataclass
class Seen:
    path: str

    def load(self) -> Dict[str, int]:
        try:
            return {k: int(v) for k, v in json.loads(open(self.path).read()).items()}
        except (OSError, ValueError):
            return {}

    def save(self, d: Dict[str, int]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)


def _key(f: str) -> str:
    return f"{os.path.realpath(f)}#{os.stat(f).st_ino}"


def scan(run_dirs: Iterable[str], seen: Seen, write: Callable[[str], None], alert: Callable[[str, str], None],
         alert_above: float = 0.54) -> int:
    """Log every new canary row of ``run_dirs``; the number of rows logged."""
    s = seen.load()
    n = 0
    for rd in sorted(set(run_dirs)):
        f = os.path.join(rd, "canary_verdicts.jsonl")
        if not os.path.isfile(f):
            continue
        k = _key(f)
        text = open(f, errors="replace").read()
        rows = text.splitlines()
        if rows and not text.endswith("\n"):
            rows = rows[:-1]   # a row still being written: read it whole on the next pass, never half
        for i in range(s.get(k, 0), len(rows)):
            run = os.path.basename(rd)
            try:
                line, hot = format_row(run, json.loads(rows[i]), alert_above)
            except ValueError:
                line, hot = f"{run} row {i + 1}: UNPARSABLE canary record", True
            write(line)
            if hot:
                alert(f"canary {k} {i}", f"CANARY non-PASS above {alert_above}: {line}")
            n += 1
        s[k] = len(rows)
    seen.save(s)
    return n


def seen_dirs(seen: Seen) -> List[str]:
    return sorted({os.path.dirname(k.split("#", 1)[0]) for k in seen.load()})


def liveness_alerts(runs: List[live.LiveRun], alert: Callable[[str, str], None], now: float,
                    first_record_min: float = 60.0, stale_h: float = 3.5) -> None:
    for r in runs:
        if r.run_dir is None:
            alert(f"unresolved {r.pid}:{r.start_ticks}", f"live launcher {r.pid}: run UNRESOLVED ({r.reason}) — its "
                                                         "canaries are NOT being logged")
            continue
        f = os.path.join(r.run_dir, "canary_verdicts.jsonl")
        started = now - r.elapsed_s
        if r.elapsed_s <= first_record_min * 60:
            continue
        if not os.path.isfile(f) or os.path.getmtime(f) < started:
            alert(f"nocanary {r.pid}:{r.start_ticks}", f"live run {os.path.basename(r.run_dir)} (launcher {r.pid}, up "
                                                       f"{r.elapsed_s // 60} min) has NO canary record since it started")
        elif now - os.path.getmtime(f) > stale_h * 3600:
            alert(f"stale {r.pid}:{r.start_ticks}:{int(os.path.getmtime(f))}",
                  f"live run {os.path.basename(r.run_dir)}: newest canary record "
                  f"{(now - os.path.getmtime(f)) / 60:.0f} min old (> {stale_h:g} h)")


def watch_files(runs: List[live.LiveRun], alert: Callable[[str, str], None]) -> List[str]:
    """Each live launcher's stdout log (what an alert watch should follow); a non-file stdout alerts once."""
    out = []
    for r in runs:
        if r.source == "orphan-trainer":
            continue
        if r.stdout and os.path.isfile(r.stdout):
            out.append(r.stdout)
        else:
            alert(f"stdout {r.pid}:{r.start_ticks}", f"live launcher {r.pid} logs to {r.stdout or '?'!r}, not a "
                                                     "regular file — the alert watch cannot follow it")
    return out


class Alerter:
    """Fires each case once (keys persisted), through ``emit``."""

    def __init__(self, path: str, emit: Callable[[str], None]):
        self.path, self.emit = path, emit

    def __call__(self, key: str, msg: str) -> None:
        try:
            done = set(open(self.path).read().splitlines())
        except OSError:
            done = set()
        if key in done:
            return
        self.emit(msg)
        with open(self.path, "a") as f:
            f.write(key + "\n")


def watch(models_dir: str, out_dir: str, emit: Callable[[str], None], interval_s: float = 300.0,
          max_hours: float = 168.0, alert_above: float = 0.54, first_record_min: float = 60.0,
          proc: str = "/proc", sleep: Callable[[float], None] = time.sleep,
          passes: Optional[int] = None) -> int:
    os.makedirs(out_dir, exist_ok=True)
    seen = Seen(os.path.join(out_dir, "canary_seen.json"))
    alert = Alerter(os.path.join(out_dir, "canary_alerted.txt"), emit)
    log_path = os.path.join(out_dir, "canary_log.txt")

    def write(line: str) -> None:
        with open(log_path, "a") as f:
            f.write(f"[{_now_iso()}] {line}\n")

    total = passes if passes is not None else max(1, int(max_hours * 3600 / interval_s))
    for i in range(total):
        runs = live.discover(models_dir, proc)
        liveness_alerts(runs, alert, time.time(), first_record_min=first_record_min)
        dirs = [r.run_dir for r in runs if r.run_dir] + seen_dirs(seen)
        scan(dirs, seen, write, alert, alert_above=alert_above)
        with open(os.path.join(out_dir, "canary_heartbeat"), "w") as f:
            f.write(f"{_now_iso()} pass {i + 1}/{total}\n")
        if i + 1 == total:
            emit(f"canary watch EXPIRING after {total} passes ({total * interval_s / 3600:.1f} h): re-arm it")
            break
        sleep(interval_s)
    return 0
