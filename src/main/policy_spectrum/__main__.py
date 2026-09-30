"""CLI for the policy-spectrum instrument (M5 Lane S). See ``main/policy_spectrum/__init__.py``.

  python -m main.policy_spectrum build  --out <bank dir>            # once; gate ① runs inside
  python -m main.policy_spectrum gate   --bank <bank dir>           # re-check gate ① here
  python -m main.policy_spectrum read   --bank <bank dir> --out <reads dir> \\
        --ckpt models/<run>/checkpoints/<ckpt>.zip=<label> [...]   # forward passes only, CPU
  python -m main.policy_spectrum report --bank <bank dir> --reads <reads dir> \\
        --labels L1,L2,... [--pair A:B ...] --out <report.md>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _commit() -> str:
    from utils.git import get_git_hash

    try:
        return get_git_hash()
    except Exception:                                    # noqa: BLE001 - a read still runs
        return "unknown"


def _models() -> Path:
    from utils.paths import main_models_dir

    md = main_models_dir()
    if md is None:
        sys.exit("[policy_spectrum] no models/ archive (main_models_dir() is None)")
    return md


def cmd_build(a) -> int:
    from main.policy_spectrum.bank import build
    from main.policy_spectrum.reader import refuse_under_models

    out = Path(a.out)
    refuse_under_models(out)
    if (out / "manifest.json").exists() and not a.force:
        sys.exit(f"[policy_spectrum] {out} already holds a bank — a bank is written ONCE (--force to rebuild)")
    t = time.time()
    man = build(_models(), out, workers=a.workers, commit=_commit())
    c = man["counts"]
    print(f"[policy_spectrum] bank {man['content_sha256'][:12]}: {c['decisions']} decisions, "
          f"{c['battles']} battles; gate ① PASS on {man['gate1']['decisions_checked']} recorded "
          f"decisions ({time.time() - t:.1f} s)")
    print(json.dumps(c, indent=1))
    return 0


def cmd_gate(a) -> int:
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import reencode

    bank = load_bank(Path(a.bank))
    _, _, gate = reencode(bank, workers=a.workers)
    print(json.dumps(gate, indent=1))
    return 0 if gate["obs_as_recorded"] else 1


def cmd_read(a) -> int:
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import read_checkpoint, reencode, refuse_under_models

    out = Path(a.out)
    refuse_under_models(out)
    bank = load_bank(Path(a.bank))
    t = time.time()
    rows, masks, gate = reencode(bank, workers=a.workers)
    print(f"[policy_spectrum] re-encoded {len(rows)} decisions in {time.time() - t:.1f} s; "
          f"obs as recorded: {gate['obs_as_recorded']} ({gate['byte_equal']}/{gate['recorded_rows_checked']})",
          flush=True)
    commit = _commit()
    for spec in a.ckpt:
        path, _, label = spec.partition("=")
        label = label or Path(path).stem
        t = time.time()
        r = read_checkpoint(bank, rows, masks, gate, Path(path), label, out, threads=a.threads,
                            commit=commit, models_root=_models(), note=a.note)
        sp = r["strata"]["all"]["all"]["spectrum"]
        print(f"[policy_spectrum] {label}: rank1/2/3 {sp[0]:.3f}/{sp[1]:.3f}/{sp[2]:.3f} "
              f"H {r['strata']['all']['all']['entropy']:.3f} ({time.time() - t:.1f} s)"
              + (f"; recording agreement {r['recording_agreement']}" if r["recording_agreement"] else ""),
              flush=True)
    return 0


def cmd_report(a) -> int:
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.report import load_reads, paired_rows, render

    bank = load_bank(Path(a.bank))
    reads_dir = Path(a.reads)
    labels = [s for s in a.labels.split(",") if s]
    reads = load_reads(reads_dir, labels)
    pairs = [tuple(p.split(":", 1)) for p in (a.pair or [])]
    paired = paired_rows(bank, reads_dir, pairs)
    pre = Path(a.preamble).read_text() if a.preamble else ""
    md = render(bank, reads, paired, a.title, pre)
    Path(a.out).write_text(md + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps({"labels": labels, "paired": paired}, indent=1) + "\n")
    print(f"[policy_spectrum] wrote {a.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.policy_spectrum")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--force", action="store_true")
    g = sub.add_parser("gate")
    g.add_argument("--bank", required=True)
    g.add_argument("--workers", type=int, default=2)
    r = sub.add_parser("read")
    r.add_argument("--bank", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--ckpt", action="append", required=True, help="<path.zip>[=<label>]")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--note", default=None)
    p = sub.add_parser("report")
    p.add_argument("--bank", required=True)
    p.add_argument("--reads", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--pair", action="append", help="<label a>:<label b>")
    p.add_argument("--out", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--title", default="Policy spectrum")
    p.add_argument("--preamble", default=None)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "gate": cmd_gate, "read": cmd_read, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
