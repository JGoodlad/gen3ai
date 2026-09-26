"""Build the NEW LINEAGE base-run argv from the old lineage's recorded fresh root, and print the diff.

    python designs/research_state/measurements/new_lineage_2026-09-26/scripts/build_argv.py [--pin <sha>]

The template is `ai_v13_02_flywheel_winprob`'s recorded `original_command` (the fresh root of the
last production lineage, 75M steps), copied verbatim to `template_ai_v13_02_flywheel_winprob.txt`
with the launcher path dropped. The registration (§2) allows exactly these moves:

  --run-name     -> ai_v14_01_base
  --pin-commit   -> the pin (origin/main HEAD when the registration was finished)
  + --arch production   (the launch guard: the production surface applied as if typed)
  + --obs-source core   (already the default on the rust bridge; named so the argv says it)

Anything else differing is a registration change. The script asserts that and writes `argv_base.txt`.
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
TEMPLATE = HERE / "template_ai_v13_02_flywheel_winprob.txt"
RUN_NAME = "ai_v14_01_base"
DEFAULT_PIN = "8d07051aa767331cfa1ca2292029d5217cbf6d40"


def flags(tokens):
    out, key = {}, None
    for t in tokens:
        if t.startswith("--"):
            key = t
            out.setdefault(key, [])
        else:
            out[key].append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default=DEFAULT_PIN)
    a = ap.parse_args()
    tpl = shlex.split(TEMPLATE.read_text())
    toks = list(tpl)
    for flag, val in (("--run-name", RUN_NAME), ("--pin-commit", a.pin)):
        i = toks.index(flag)
        toks[i + 1] = val
    toks += ["--arch", "production", "--obs-source", "core"]

    before, after = flags(tpl), flags(toks)
    moved = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    allowed = {"--run-name", "--pin-commit", "--arch", "--obs-source"}
    assert set(moved) <= allowed, f"unregistered move(s): {set(moved) - allowed}"
    assert "--model" not in after and "--stable-opponents" not in after, "the base run is FRESH"
    assert after["--steps"] == ["75000000"], after["--steps"]
    print(f"== base  {RUN_NAME}  ({len(tpl)} -> {len(toks)} tokens vs ai_v13_02_flywheel_winprob)")
    for k in moved:
        print(f"     {k}: {before.get(k)} -> {after.get(k)}")
    (KIT / "argv_base.txt").write_text(" ".join(toks) + "\n")
    print(f"wrote {KIT / 'argv_base.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
