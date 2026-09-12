"""Every `--ent-coef` the archive has ever run, crossed with `arch_signature`.

    export PYTHONPATH=$PYTHONPATH:src
    python3 ent_coef_archive_survey.py [--json OUT.json]

WHY THIS IS THE DECISIVE TABLE. The question "has the entropy coefficient ever been isolated?"
is answerable without running anything: if no two runs share an architecture and differ only in
`--ent-coef`, the coefficient is perfectly COLLINEAR with the era and no observational read of the
archive can separate it. This prints that cross-tab.

The coefficient is read from `metadata.json`'s IMMUTABLE `original_command` (the argv as typed),
and the run's `arch_signature` from `model_config.json`. A run whose `original_command` is a
SCRIPT path (a surgery fork — `ai_v8_01_zarch_film_0717` is one) carries no argv and is reported
as `unstated`, never as the parser default: the default would be a fabricated 0.02 for a run whose
TensorBoard logged `hparams/ent_coef = 0.05`. Read the value from TB for those.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

from utils.paths import main_models_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also write the cross-tab here")
    a = ap.parse_args()

    md_dir = main_models_dir()
    if md_dir is None:
        print("SKIP: no models/ archive reachable (main_models_dir() is None).")
        return 0

    tab: dict = collections.defaultdict(collections.Counter)
    examples: dict = collections.defaultdict(list)
    for md in sorted(glob.glob(os.path.join(str(md_dir), "*", "metadata.json"))):
        run_dir = os.path.dirname(md)
        run = os.path.basename(run_dir)
        try:
            d = json.loads(open(md).read())
        except Exception:
            continue
        oc = d.get("original_command") or {}
        argv = oc.get("argv") if isinstance(oc, dict) else oc
        if not isinstance(argv, str):
            continue
        m = re.search(r"--ent-coef\s+(\S+)", argv)
        val = m.group(1) if m else "unstated (no argv / flag absent — read TB hparams/ent_coef)"
        try:
            arch = json.loads(open(os.path.join(run_dir, "model_config.json")).read()) \
                .get("arch_signature", "?")
        except Exception:
            arch = "?"
        tab[arch][val] += 1
        if len(examples[(arch, val)]) < 4:
            examples[(arch, val)].append(run)

    out = {arch: dict(c) for arch, c in sorted(tab.items())}
    width = max((len(a_) for a_ in out), default=10)
    for arch, c in out.items():
        print(f"{arch:{width}s}  {c}")
    print()
    vals: collections.Counter = collections.Counter()
    for c in tab.values():
        vals.update(c)
    print("ent-coef totals:", dict(vals.most_common()))
    crossing = [arch for arch, c in tab.items()
                if len({v for v in c if v.startswith(('0.02', '0.05'))}) > 1]
    print("architectures carrying BOTH 0.02 and 0.05:", crossing or "NONE — "
          "the coefficient is perfectly collinear with the era")
    if a.json:
        json.dump(dict(cross_tab=out, totals=dict(vals),
                       archs_with_both_0_02_and_0_05=crossing,
                       examples={f"{k[0]}|{k[1]}": v for k, v in examples.items()}),
                  open(a.json, "w"), indent=2)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
