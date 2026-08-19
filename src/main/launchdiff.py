"""Launch-diff gate — refuse to launch a run whose command differs from a reference
run's in ways nobody acknowledged.

WHY. A training flag can go missing between generations and nothing notices: it is
training-only, it bumps no `ARCH_SIGNATURE`, and `check_compatible` never sees it. That
is exactly how `--all-shaping-pbrs` was present in all 20 `ai_v8_*` runs and absent from
every `ai_v9_*` one — the reward silently went from 8 PBRS + 1 BIAS to 3 PBRS + 28
fully-additive BIAS, and the lineage trained that way for its whole life. The state was
even documented as current; what was missing was any moment where someone had to SAY the
difference out loud.

So this gate does not judge. It ENUMERATES every difference against a reference run's
recorded `launcher_command` and requires each one to be acknowledged by name. An
intended change costs one `--ack` entry; an unintended one is impossible to miss,
because the launch does not proceed until it is listed.

Usage:
    python -m main.launchdiff --ref models/<reference_run> --argv "<the new command>"
    python -m main.launchdiff --ref models/<ref> --argv "..." --ack all-shaping-pbrs,draw-penalty

Exit 0 = every difference acknowledged (or none). Exit 1 = unacknowledged differences.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from typing import Dict, List, Tuple

# Launcher-owned flags are not forwarded to the trainer; differing here is not a
# behavioural difference in the RUN, so they are reported separately rather than gating.
LAUNCHER_OWNED = {
    "--restart-interval-hours", "--restart-grace-minutes", "--max-crash-restarts",
    "--nice", "--no-pin", "--sync-to-main", "--showdown-port", "--run-name",
}
# Flags whose value is expected to differ per run and never needs acknowledging.
PER_RUN = {"--run-name", "--model", "--steps"}


def parse_flags(argv: List[str]) -> Dict[str, str]:
    """argv -> {flag: value}. A bare flag (next token is another flag, or absent) is "ON"."""
    out: Dict[str, str] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[tok] = argv[i + 1]
            i += 2
        else:
            out[tok] = "ON"
            i += 1
    return out


def load_reference(run_dir: str) -> Dict[str, str]:
    meta = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta):
        raise SystemExit(f"[launchdiff] no metadata.json in {run_dir}")
    cmd = json.load(open(meta)).get("launcher_command")
    if not cmd:
        raise SystemExit(
            f"[launchdiff] {run_dir} records no launcher_command — it cannot be a reference. "
            "(Runs before the provenance change have none; pick a newer reference.)")
    return parse_flags(shlex.split(cmd)[1:])


def diff(ref: Dict[str, str], new: Dict[str, str]) -> Tuple[list, list, list]:
    """(dropped, added, changed) — each entry (flag, ref_value, new_value)."""
    dropped = [(f, ref[f], None) for f in sorted(ref) if f not in new]
    added = [(f, None, new[f]) for f in sorted(new) if f not in ref]
    changed = [(f, ref[f], new[f]) for f in sorted(ref)
               if f in new and ref[f] != new[f]]
    return dropped, added, changed


def report(ref: Dict[str, str], new: Dict[str, str], acked: set) -> int:
    dropped, added, changed = diff(ref, new)
    rows = ([("DROPPED", f, a, b) for f, a, b in dropped]
            + [("ADDED", f, a, b) for f, a, b in added]
            + [("CHANGED", f, a, b) for f, a, b in changed])
    gating = [r for r in rows if r[1] not in LAUNCHER_OWNED and r[1] not in PER_RUN]
    informational = [r for r in rows if r not in gating]

    print(f"[launchdiff] {len(rows)} difference(s) vs the reference "
          f"({len(gating)} gating, {len(informational)} launcher-owned/per-run)")
    if informational:
        print("\n  not gating (launcher-owned or expected-per-run):")
        for kind, f, a, b in informational:
            print(f"    {kind:8} {f:32} {a!r} -> {b!r}")

    unacked = []
    if gating:
        print("\n  BEHAVIOURAL differences — each must be acknowledged:")
        for kind, f, a, b in gating:
            key = f.lstrip("-")
            ok = key in acked
            if not ok:
                unacked.append(key)
            print(f"    [{'ACK' if ok else '   '}] {kind:8} {f:32} {a!r} -> {b!r}")

    if unacked:
        print(f"\n[launchdiff] ✗ {len(unacked)} UNACKNOWLEDGED difference(s): "
              f"{','.join(unacked)}")
        print("[launchdiff]   If they are intended, re-run with:")
        print(f"[launchdiff]   --ack {','.join(sorted(set(unacked)))}")
        print("[launchdiff]   If any is NOT intended, you just caught a silent drift.")
        return 1
    print("\n[launchdiff] ✓ every behavioural difference is acknowledged")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Refuse a launch whose diff vs a reference is unacknowledged.")
    ap.add_argument("--ref", required=True, help="reference run dir (reads its metadata.json launcher_command)")
    ap.add_argument("--argv", required=True, help="the proposed command, quoted")
    ap.add_argument("--ack", default="", help="comma-separated flag names (no leading --) to acknowledge")
    a = ap.parse_args()
    acked = {x.strip().lstrip("-") for x in a.ack.split(",") if x.strip()}
    return report(load_reference(a.ref), parse_flags(shlex.split(a.argv)), acked)


if __name__ == "__main__":
    sys.exit(main())
