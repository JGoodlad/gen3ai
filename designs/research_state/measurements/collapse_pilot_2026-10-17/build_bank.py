"""The collapse pilot's TURN BANK: X26's (`rb_x26_s2601`) own eval traces at 30M / 90M / 150M, through the
policy-spectrum bank builder unchanged (`main.policy_spectrum.bank.build`: the per-(cycle, opponent) 2 win + 2 loss +
1 draw draw by hash, gate ① = every recorded decision re-encodes byte-equal at this checkout, stamped strata).

X26's pin is `9c033489`; between it and the build commit the encoder (`src/rust_sim/src`, `src/rust_env/src`,
`src/agents/observation`) is unchanged, and gate ① checks that on every recorded decision anyway.

    PYTHONPATH=src python designs/research_state/measurements/collapse_pilot_2026-10-17/build_bank.py \
        --out ~/gen3ai_archive/collapse_pilot_2026-10-17/bank_x26
"""
import argparse
import json
import subprocess
from pathlib import Path

from main.policy_spectrum.bank import Source, build
from utils.paths import main_models_dir

RUN = "rb_x26_s2601"
SOURCES = (
    Source(RUN, "step_30000045", "X26@30M"),
    Source(RUN, "step_90000067", "X26@90M"),
    Source(RUN, "step_150000100", "X26@150M"),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: the bank is written once")
    models = main_models_dir()
    if models is None:
        raise SystemExit("no models/ archive")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    man = build(models, out, sources=SOURCES, workers=a.workers, commit=commit)
    print(json.dumps({k: man[k] for k in ("content_sha256", "counts", "dropped_single_legal")}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
