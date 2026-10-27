"""The canary logger: every canary record of every LIVE run, one line each, plus the watch alerts.

Replaces ``canary_log_v*.sh`` + ``live_runs.py`` + ``aw_files.sh``. Its class bugs:

* **Name globs missed three run families** (v1-v11): the runs come from :func:`live.discover` (launcher argv /
  trainer children) plus every run already in the seen file — never a pattern.
* **The seen file was keyed by run NAME and a row count**: a run dir re-created under the same name with fewer
  rows was skipped until it overtook the old count. The key here is the canary file's ``(path, inode)``.
* **A non-PASS below a value bar paged nobody** (10-27: N1-P s3501's u200 UNCONFIRMED read compiled 0.1675, under
  the old 0.54 bar, and no alert fired): EVERY non-PASS record (UNCONFIRMED, FATAL, anything not ``pass``) alerts,
  whatever its readings. The family ratio bar (``main.ops.canary_decide.FAMILY_BARS``) is only a DESCRIPTIVE tag on
  the log line (``[descriptive: ratio>2.64 (c0 bar)]``); it never alerts (the resume decision is canary_decide's).
* **A dump the near-margin rotation deletes** (``canary_u*_near_policy.pt`` keeps the newest 3 per run) or a run dir
  thinned / moved later loses the weights a verdict was read on: :func:`archive_dumps` copies every
  ``canary_u<u>_s<steps>[_r<n>]_<unconfirmed|fatal|near>_policy.pt`` to ``<archive>/<run>_canary_u<u>/`` with a
  ``.sha256`` beside it, never over an existing copy.
* **A bounded loop lapsed silently** (v9, after its 24 h): :func:`watch` writes an ``expiring`` line on its last
  pass and a heartbeat file every pass.
* Alerts: a live launcher whose run is UNRESOLVED; a run with NO canary record ``first_record_min`` after its
  launcher started (default 60: a production cold start reached its first record at ~34 min, 40 was too close); a
  newest record older than ``stale_h``. Each alert fires once per case.

The launcher-log alerts (:data:`LAUNCHER_ALERT_ERE`) are not this loop's: the alert watch follows
:func:`watch_files` (the status file, each live launcher's stdout log and its run's ``launcher_child.full.log``)
through :func:`monitor_filter`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from main.ops.arm_chain import live
from main.ops.arm_chain.fsguard import UnsafeRemove, remove_under
from main.ops.canary_decide import FAMILY_BARS, family_of

#: The launcher-log lines that page (operator review 10-27): a crash, a wedge, an update that cannot fit, a
#: behaviour-policy mismatch, a Rust stamp mismatch, and a compiled-vs-eager disagreement.
LAUNCHER_ALERT_ERE = r"Traceback|Child crashed|WEDGE|UpdateWontFit|BEHAVIOUR-POLICY MISMATCH|StampMismatch|DISAGREES"

#: An unconfirmed / FATAL / near-margin canary weight dump (``agents.model.compile_canary._save_new``).
DUMP_RE = re.compile(r"canary_u(\d+)_s(\d+)(?:_r\d+)?_(unconfirmed|fatal|near)_policy\.pt")


def monitor_filter(chain: str) -> str:
    """The ONE ERE the alert Monitor greps over :func:`watch_files` (``arm_chain monitor`` applies it itself): the
    chain's HOLD / SHIP-HOLD / COMPLETE lines, its ``MONITOR-REARM`` cue at each GO (a ``tail -F`` Monitor must re-arm
    on the new run's logs), a dead or stale supervisor (``SUPERVISOR DEAD|STALE``, written by ``monitor``), the canary
    logger's HOLD lines, and the launcher-log alerts. ``chain`` is a slug (``spec``), so it needs no escaping (GNU
    grep >= 3.8 warns on a stray ``\\-``); ``<name>`` is the doc's placeholder."""
    if chain != "<name>" and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", chain):
        raise ValueError(f"chain name {chain!r} is not a slug")
    return (r"ARMCHAIN\[" + chain + r"\]: (HOLD|SHIP-HOLD|chain COMPLETE|MONITOR-REARM|SUPERVISOR (DEAD|STALE))|"
            r"CANARYLOG: HOLD|AWFILES: HOLD|" + LAUNCHER_ALERT_ERE)


def _now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _get(d: Any, *ks: str) -> Any:
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def format_row(run: str, r: Dict[str, Any]) -> Tuple[str, bool]:
    """(the log line, whether it alerts). EVERY non-PASS alerts, whatever its readings; a ratio above the run
    family's registered bar is a DESCRIPTIVE tag only."""
    m = _get(r, "watch", "scheduled", "clip_margin", "max_abs_dlogp") or {}
    sp = _get(r, "watch", "scheduled", "clip_margin", "same_precision", "max_abs_dlogp")
    c, e = m.get("compiled"), m.get("eager")
    ratio = round(c / e, 3) if isinstance(c, (int, float)) and isinstance(e, (int, float)) and e else None
    verdict = str(r.get("verdict"))
    line = (f"{run} u{r.get('update')} @{r.get('num_timesteps')} {verdict} compiled {c} eager {e} ratio {ratio} "
            f"same-precision {sp}")
    fam = family_of(run)
    if fam is not None and ratio is not None and ratio > FAMILY_BARS[fam]:
        line += f" [descriptive: ratio>{FAMILY_BARS[fam]:g} ({fam} bar)]"
    return line, verdict.lower() != "pass"


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


def scan(run_dirs: Iterable[str], seen: Seen, write: Callable[[str], None], alert: Callable[[str, str], None]) -> int:
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
                line, hot = format_row(run, json.loads(rows[i]))
            except ValueError:
                line, hot = f"{run} row {i + 1}: UNPARSABLE canary record", True
            write(line)
            if hot:
                alert(f"canary {k} {i}", f"CANARY non-PASS: {line}")
            n += 1
        s[k] = len(rows)
    seen.save(s)
    return n


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def archive_dumps(run_dirs: Iterable[str], archive_root: str, write: Callable[[str], None],
                  alert: Callable[[str, str], None]) -> int:
    """Copy each canary weight dump of ``run_dirs`` to ``<archive_root>/<run>_canary_u<u>/<name>`` + ``<name>.sha256``
    (``sha256sum`` format), verified after the copy. Never over an existing copy (one with other bytes alerts); a
    copy already there with its sha file is skipped unhashed. The number copied."""
    n = 0
    for rd in sorted(set(run_dirs)):
        try:
            names = sorted(os.listdir(rd))
        except OSError:
            continue
        run = os.path.basename(rd.rstrip("/"))
        for name in names:
            m = DUMP_RE.fullmatch(name)
            if not m:
                continue
            src = os.path.join(rd, name)
            dest_dir = os.path.join(archive_root, f"{run}_canary_u{m.group(1)}")
            dest, sha_file = os.path.join(dest_dir, name), os.path.join(dest_dir, name + ".sha256")
            if os.path.exists(dest) and os.path.exists(sha_file):
                continue
            try:
                digest = _sha256(src)
                os.makedirs(dest_dir, exist_ok=True)
                if os.path.exists(dest):
                    if _sha256(dest) != digest:
                        alert(f"dumpclash {dest}", f"canary dump {src} differs from the archived {dest}: NOT copied")
                        continue
                else:
                    tmp = os.path.join(dest_dir, f".{name}.{os.getpid()}.tmp")
                    shutil.copyfile(src, tmp)
                    if _sha256(tmp) != digest:
                        raise OSError(f"the copy of {src} does not hash to its source")
                    os.link(tmp, dest)            # never over an existing file
                    remove_under(tmp, dest_dir)
                with open(sha_file + ".tmp", "w") as f:
                    f.write(f"{digest}  {name}\n")
                os.replace(sha_file + ".tmp", sha_file)
            except (OSError, UnsafeRemove) as e:
                alert(f"dumpfail {src}", f"canary dump {src} NOT archived to {dest_dir}: {e}")
                continue
            write(f"{run} canary dump {name} archived to {dest_dir} (sha256 {digest[:16]}…)")
            n += 1
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
    """What the alert watch follows (with :func:`monitor_filter`): each live launcher's stdout log, and its run's
    ``launcher_child.full.log`` (the trainer's own lines — a child Traceback, BEHAVIOUR-POLICY MISMATCH — land there;
    it ROTATES by rename, which ``tail -F`` follows, where the ring ``launcher_child.log`` is rewritten in place and
    would re-page old lines). A non-file stdout alerts once."""
    out = []
    for r in runs:
        if r.source == "orphan-trainer":
            continue
        if r.stdout and os.path.isfile(r.stdout):
            out.append(r.stdout)
        else:
            alert(f"stdout {r.pid}:{r.start_ticks}", f"live launcher {r.pid} logs to {r.stdout or '?'!r}, not a "
                                                     "regular file — the alert watch cannot follow it")
        if r.run_dir:
            out.append(os.path.join(r.run_dir, "launcher_child.full.log"))
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


def watch(models_dir: str, out_dir: str, emit: Callable[[str], None], archive_root: str, interval_s: float = 300.0,
          max_hours: float = 168.0, first_record_min: float = 60.0,
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
        scan(dirs, seen, write, alert)
        archive_dumps(dirs, archive_root, write, alert)
        with open(os.path.join(out_dir, "canary_heartbeat"), "w") as f:
            f.write(f"{_now_iso()} pass {i + 1}/{total}\n")
        if i + 1 == total:
            emit(f"canary watch EXPIRING after {total} passes ({total * interval_s / 3600:.1f} h): re-arm it")
            break
        sleep(interval_s)
    return 0
