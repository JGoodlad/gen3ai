"""X26c read: the models it plays, resolved READ-ONLY from ``models/`` and copied (sha-checked, with their sidecars)
into the read's archive. One rule for "the 480M final" (PREREG §2), used by both runners.

    python finals.py status             # is the 480M final ready? (exit 0 yes, 1 not yet)
    python finals.py copy --dst DIR     # copy P*, X26 @380M and (once ready) the 480M final; print finals JSON
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "x35_breakers_stage_2026-10-17" / "x26_drift_preread"))
import preread as PR

MODELS = Path("/home/goodlad/dev/gen3ai/models")
RUN = MODELS / "rb_x26c_s2601"
PARENT = MODELS / "rb_x26_s2601"
PSTAR = PARENT / "checkpoints" / "checkpoint_280000170_steps.zip"
FORK = PARENT / "checkpoints" / "checkpoint_380000195_steps.zip"
FINAL_MIN_STEP = 480_000_000
#: a final is read only once nothing has written it for this long (the end chain has finished)
QUIET_S = 300
_CKPT = re.compile(r"checkpoint_(\d+)_steps\.zip$")


def _step(z: Path) -> int:
    from main.x35.kit import _final_step

    return _final_step(str(z))


def final_480m() -> Path | None:
    """PREREG §2: ``final_model.zip`` if its recorded step is >= 480,000,000, else the newest checkpoint whose step
    is >= 480,000,000; in either case untouched for QUIET_S seconds. None = not ready."""
    now = time.time()
    z = RUN / "final_model.zip"
    if z.is_file() and now - z.stat().st_mtime >= QUIET_S and _step(z) >= FINAL_MIN_STEP:
        return z
    best = None
    for c in (RUN / "checkpoints").glob("checkpoint_*_steps.zip"):
        m = _CKPT.search(c.name)
        if (m and int(m.group(1)) >= FINAL_MIN_STEP and now - c.stat().st_mtime >= QUIET_S
                and (best is None or int(m.group(1)) > int(_CKPT.search(best.name).group(1)))):
            best = c
    if best is not None and _step(best) >= FINAL_MIN_STEP:
        return best
    return None


def copy(label: str, src: Path, dst_root: Path) -> dict:
    dst, sha = PR.copy_checked(src, dst_root / label)
    return {"label": label, "arm": label, "seed": 2601, "path": str(dst), "sha256": sha, "source": str(src),
            "step": _step(dst)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    c = sub.add_parser("copy")
    c.add_argument("--dst", required=True)
    c.add_argument("--which", default="pstar,x26_380m,x26c_480m")
    a = ap.parse_args(argv)
    f = final_480m()
    if a.cmd == "status":
        print(json.dumps({"final_480m": str(f) if f else None}))
        return 0 if f else 1
    out = []
    srcs = {"pstar": PSTAR, "x26_380m": FORK, "x26c_480m": f}
    for lab in a.which.split(","):
        if srcs[lab] is None:
            print(f"{lab}: not ready", file=sys.stderr)
            return 1
        out.append(copy(lab, srcs[lab], Path(a.dst)))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
