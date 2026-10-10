"""``python -m main.probe_battery {bank,capture,probe,report,behaviour,depth,depth-report}`` — see ``designs/prober/probe_battery.md``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _labelled(specs: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"[probe_battery] name each checkpoint as LABEL=PATH.zip (got {s!r})")
        lab, path = s.split("=", 1)
        if not path.endswith(".zip"):
            raise SystemExit(f"[probe_battery] {s!r}: name the .zip (a run directory moves with the run)")
        out.append((lab, path))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.probe_battery", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bank", help="play + replay + select a decision bank (CPU, at the checkpoints' checkout)")
    p.add_argument("--checkout", required=True, help="the checkout at the checkpoints' commit (read-only use)")
    p.add_argument("--expect-commit", required=True, help="refuse unless the checkout's HEAD starts with this")
    p.add_argument("--legacy", nargs="+", default=[], help="LABEL=ckpt.zip (one arm)")
    p.add_argument("--static", nargs="+", default=[], help="LABEL=ckpt.zip (the other arm)")
    p.add_argument("--pairs", type=int, default=16, help="mirrored pairs per cell")
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--target", type=int, default=25000, help="decisions to keep")
    p.add_argument("--out", required=True)
    p.add_argument("--n-envs", type=int, default=32)

    p = sub.add_parser("capture", help="representations of checkpoints over a bank (CPU, at their checkout)")
    p.add_argument("--checkout", required=True)
    p.add_argument("--expect-commit", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--ckpt", nargs="+", required=True, help="LABEL=ckpt.zip; LABEL=ckpt.zip@rand<SEED> = a fresh "
                   "random-init build of that checkpoint's architecture")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)

    p = sub.add_parser("probe", help="fit the linear probes on captured representations (CPU, no model)")
    p.add_argument("--bank", required=True)
    p.add_argument("--caps", required=True, help="the capture directory")
    p.add_argument("--labels", nargs="*", default=None, help="capture labels (default: every capture)")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)

    p = sub.add_parser("report", help="aggregate the probes per arm and rank the catalogue")
    p.add_argument("--probes", required=True)
    p.add_argument("--arm", action="append", required=True, help="ARM=label,label,... (e.g. legacy=L1,L2)")
    p.add_argument("--random", action="append", default=[], help="ARM=label,... random-init captures of that arm")
    p.add_argument("--out", required=True)

    p = sub.add_parser("behaviour", help="phazing behavioural probes on constructed obs edits (CPU)")
    p.add_argument("--checkout", required=True)
    p.add_argument("--expect-commit", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--ckpt", nargs="+", required=True, help="LABEL=ckpt.zip")
    p.add_argument("--arm", action="append", required=True, help="ARM=label,label,...")
    p.add_argument("--out", required=True)

    p = sub.add_parser("depth", help="depth-use + capacity-use diagnostics per checkpoint (CPU, at their checkout)")
    p.add_argument("--checkout", required=True)
    p.add_argument("--expect-commit", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--ckpt", nargs="+", required=True, help="LABEL=ckpt.zip[@rand<SEED>]")
    p.add_argument("--n-rows", type=int, default=4000, help="the bank subsample (every k-th row)")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)

    p = sub.add_parser("depth-report", help="summarise the depth / capacity diagnostics per arm + the verdict")
    p.add_argument("--depth", required=True, help="the depth results directory")
    p.add_argument("--arm", action="append", required=True, help="ARM=label,label,...")
    p.add_argument("--random", action="append", default=[], help="ARM=label,... random-init builds")
    p.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    from main.probe_battery import bank as B

    if a.cmd == "bank":
        man = B.build(checkout=Path(a.checkout), expect_commit=a.expect_commit, legacy=_labelled(a.legacy),
                      static=_labelled(a.static), pairs=a.pairs, seed=a.seed, target=a.target, out=Path(a.out),
                      n_envs=a.n_envs)
        print(json.dumps({k: man[k] for k in ("decisions", "battles", "by_archetype", "by_phase")}, indent=1))
        return 0
    if a.cmd == "capture":
        from main.probe_battery import capture as C

        C.capture_all(checkout=Path(a.checkout), expect_commit=a.expect_commit, bank=Path(a.bank),
                      specs=a.ckpt, out=Path(a.out), threads=a.threads)
        return 0
    if a.cmd == "probe":
        from main.probe_battery import probes as P

        P.probe_all(bank=Path(a.bank), caps=Path(a.caps), labels=a.labels, out=Path(a.out), threads=a.threads)
        return 0
    if a.cmd == "report":
        from main.probe_battery import report as R

        R.write_report(probes=Path(a.probes), arms=R.parse_arms(a.arm), randoms=R.parse_arms(a.random),
                       out=Path(a.out))
        return 0
    if a.cmd == "depth":
        from main.probe_battery import depth as D

        D.run_all(checkout=Path(a.checkout), expect_commit=a.expect_commit, bank=Path(a.bank), specs=a.ckpt,
                  out=Path(a.out), n_rows=a.n_rows, threads=a.threads)
        return 0
    if a.cmd == "depth-report":
        from main.probe_battery import depth as D
        from main.probe_battery import report as R

        v = D.write_report(res_dir=Path(a.depth), arms=R.parse_arms(a.arm), randoms=R.parse_arms(a.random),
                           out=Path(a.out))
        print(json.dumps(v, indent=1))
        return 0
    if a.cmd == "behaviour":
        from main.probe_battery import behaviour as BH

        BH.run(checkout=Path(a.checkout), expect_commit=a.expect_commit, bank=Path(a.bank), specs=a.ckpt,
               arms=a.arm, out=Path(a.out))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
