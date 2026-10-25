"""ONE checkpoint of the per-row null CENSUS (gpu_census.sh drives it, one process per checkpoint; README here).

Picks the run's first checkpoint at or after ``--at`` (e.g. 330M), then runs the x26c margin read
(`../x26c_canary_margin_2026-10-25/margin_read.py`, verbatim) on it with EVERY fixture row as a row of interest, so the
JSON carries each row's per-seed fp32 null (the K-sensitivity) beside the arms' per-row shifts (`--compiled`: the
production compiled arm at the canary's B = 2048). Two shims, both stated in the JSON's log line:
  * ``--vf-coef`` is typed through the resume resolution (the x26c README's F7: a re-read of an X26-lineage
    checkpoint is refused for it; it weights the critic loss only, so no log-prob read here depends on it);
  * fp32 matmul precision 'highest', the canary's one precision.

    python census_read.py --run-dir <models/run> --at 330M --out <dir> [--device cuda] [--dither 32] [--vf-coef 0.5]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "x26c_canary_margin_2026-10-25"))


def pick(run_dir: Path, at: str) -> Path:
    target = int(float(at.rstrip("M")) * 1_000_000)
    ck = sorted(((int(m.group(1)), p) for p in (run_dir / "checkpoints").glob("checkpoint_*_steps.zip")
                 if (m := re.search(r"checkpoint_(\d+)_steps\.zip$", p.name))), key=lambda t: t[0])
    for step, p in ck:
        if step >= target:
            return p
    raise SystemExit(f"no checkpoint of {run_dir} at or after {at}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--at", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dither", type=int, default=32)
    ap.add_argument("--vf-coef", default=None)
    ap.add_argument("--no-compiled", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_float32_matmul_precision("highest")
    if a.vf_coef is not None:
        import main.ops.cuda_build_validate as V
        orig = V.resolved_args
        V.resolved_args = lambda argv, device: orig(list(argv) + ["--vf-coef", str(a.vf_coef)], device)
    zp = pick(Path(a.run_dir), a.at)
    label = f"{Path(a.run_dir).name}_{a.at}"
    print(f"[census] {label}: {zp} (device {a.device}, dither {a.dither}, vf-coef typed {a.vf_coef})", flush=True)
    import margin_read
    sys.argv = ["margin_read.py", "--ckpt", f"{label}={zp}", "--out", a.out, "--dither", str(a.dither),
                "--rows", ",".join(str(r) for r in range(64)), "--device", a.device]
    if not a.no_compiled:
        sys.argv.append("--compiled")
    return int(margin_read.main())


if __name__ == "__main__":
    raise SystemExit(main())
