"""Capture representations for many checkpoints over one bank, one worker process per checkpoint, at the
checkpoints' own checkout (`gen3_probe_battery_v1`). Resumable: a label whose capture exists is skipped.

A spec is ``LABEL=path.zip`` or ``LABEL=path.zip@rand<SEED>`` — the latter is a FRESH build of that checkpoint's
architecture under ``torch.manual_seed(SEED)`` (the battery's "architecture alone" baseline), never its weights.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from main.probe_battery.bank import BatteryError, refuse_models_output, run_worker

_RAND = re.compile(r"^(?P<zip>.+\.zip)@rand(?P<seed>\d+)$")


def parse_spec(spec: str) -> Tuple[str, str, Optional[int]]:
    if "=" not in spec:
        raise BatteryError(f"name each checkpoint as LABEL=PATH.zip[@randSEED] (got {spec!r})")
    label, rest = spec.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise BatteryError(f"label {label!r}: letters, digits, '_', '.', '-' only")
    m = _RAND.match(rest)
    if m:
        return label, m.group("zip"), int(m.group("seed"))
    if not rest.endswith(".zip"):
        raise BatteryError(f"{spec!r}: name the .zip (a run directory resolves to its LAST snapshot and moves)")
    return label, rest, None


def capture_all(*, checkout: Path, expect_commit: str, bank: Path, specs: Sequence[str], out: Path,
                threads: int = 4) -> List[str]:
    refuse_models_output(out)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for spec in specs:
        label, zp, rnd = parse_spec(spec)
        dst = out / f"{label}.npz"
        if dst.exists():
            done.append(label)
            continue
        args = ["capture", "--bank", str(bank), "--ckpt", zp, "--label", label, "--out", str(dst),
                "--threads", str(threads)]
        if rnd is not None:
            args += ["--random-init", str(rnd)]
        run_worker(checkout, expect_commit, args, name="probebat-cap", mem_gb=24)
        done.append(label)
    return done
