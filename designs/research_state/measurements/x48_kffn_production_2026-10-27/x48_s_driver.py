"""X48 look 1, PREREG NOTE 5: the S lane's units with ARCHITECTURE-PURE engines, so the X48 cells can play at the pin.

WHY. At the pin `86cfdf7ef` an h2h engine serves at most `main.h2h.arch.MAX_GROUPS` (2) architectures. A C0 final and
six of R's seven members (production, x26_*) share one architecture and `tstar` is the second, so every C0 engine
holds two. An X48 final is a THIRD (per-kind FFN experts): an engine that pairs it with `tstar` AND any other member
is refused at preflight (`CellArchMismatch`, before any game; 2026-10-29 02:23, the lane's first X48 unit, rc 1).
`runner.s_engines` splits a label's members over engines by stride (`left[w::k]`), which mixes them, and a pinned
spec cannot move to a patched runner (`repin` refuses a registered pin).

WHAT. Exactly the S lane's loop (`main.reads.runner.run`, lane `s`) with ONE change: a label's pending members are
grouped by architecture before they are put on engines, so every engine is the label plus members of ONE
architecture. Everything else is the runner's own code at the pin: R re-verified before every unit (`_s_members`),
the pending cells (`_s_pending`), the unit argv (`s_unit_argv`: `mem_cap.sh --slot`, nice 19, cores 5-7,13-15, the
`main.reads s-unit` verb, which re-hashes every model and runs `play_many_argv`, the same `--pairs`, `--batch-pairs`,
family, request, ledger, `--device cpu --torch-threads 1`), the box (2,400 s), the rc handling, the registered stop.
Pairs, seeds, regime and ledger rows are those the runner would write; only the grouping of cells into processes
differs. Run it from the clean pin worktree with its `src` on PYTHONPATH. It stops at `--stop-epoch`, when nothing
is pending, or when `<state>/s/x48_driver.stop` exists (checked between units). Log: `<state>/s/x48_driver.jsonl`.

    cd /home/goodlad/dev/gen3ai-reads/x48_play_86cfdf7e && PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES="" \
      python .../x48_s_driver.py --state ~/gen3ai_archive/x35_x48_read/look1 --stop-epoch <the registered stop>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from main.reads import layout as LY
from main.reads import runner as R

#: The fields that make two members' architectures differ for the engine (read from each member copy's
#: model_config.json); members equal on all of them share a slot group.
ARCH_KEYS = ("arch_signature", "trunk_layers", "trunk_ffn_by_kind", "damage_op", "edge_bias_families",
             "entity_tail_seats", "entity_topk_seats", "opp_intent", "token_encoding", "history_events")


def arch_key(copy: str) -> Tuple[Any, ...]:
    cfg = json.loads((Path(copy).parent / "model_config.json").read_text())
    return tuple(json.dumps(cfg.get(k), sort_keys=True) for k in ARCH_KEYS)


def pure_engines(pending: Sequence[Tuple[str, Sequence[str]]], members: Dict[str, Dict[str, Any]],
                 workers: int) -> List[str]:
    """The pending ``(label, [members])`` cells as at most ``workers`` engines, each the label plus members of ONE
    architecture (in the members' own order)."""
    engines: List[str] = []
    for label, left in pending:
        groups: Dict[Tuple[Any, ...], List[str]] = {}
        for m in left:
            groups.setdefault(arch_key(members[m]["copy"]), []).append(m)
        for ms in groups.values():
            if len(engines) < workers:
                engines.append(f"{label}:" + ",".join(ms))
        if len(engines) >= workers:
            break
    return engines


def _jl(path: Path, ev: Dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"t": time.time(), **ev}, sort_keys=True) + "\n")


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--stop-epoch", type=float, required=True)
    ap.add_argument("--workers", type=int, default=R.S_WORKERS)
    a = ap.parse_args(argv)
    state = a.state.expanduser()
    spec = R.load_state(state)
    models = R._load_models(state)
    jl = state / "s" / "x48_driver.jsonl"
    stop_file = state / "s" / "x48_driver.stop"
    _jl(jl, {"event": "start", "stop_epoch": a.stop_epoch, "workers": a.workers})
    while True:
        if stop_file.exists():
            _jl(jl, {"event": "stopped", "why": "stop file"})
            return 0
        for g in spec.s:
            R._s_members(spec, g)                                   # R re-verified before every unit
        pending = R._s_pending(spec, state, models)
        if not pending:
            _jl(jl, {"event": "done"})
            return 0
        if time.time() + 60.0 >= a.stop_epoch:
            _jl(jl, {"event": "stop_at", "left": [(g, lab) for g, lab, _ in pending]})
            return 0
        g = pending[0][0]
        members = json.loads((LY.s_dir(state, g) / "state.json").read_text())["members"]
        engines = pure_engines([(lab, left) for gg, lab, left in pending if gg == g], members, a.workers)
        box = int(min(R.S_UNIT_S, a.stop_epoch - time.time()))
        cmd = R.s_unit_argv(state, g, engines, box, deadline_epoch=a.stop_epoch)
        before = R._progress(spec, state, g, engines, models)
        _jl(jl, {"event": "unit", "group": g, "engines": engines, "box_s": box})
        rc = subprocess.run(cmd, check=False, env=R.child_env()).returncode
        _jl(jl, {"event": "unit_end", "group": g, "engines": engines, "rc": rc})
        if rc not in (0, R.UNIT_TIME_UP):
            return rc
        if rc == 0 and R._progress(spec, state, g, engines, models) == before:
            _jl(jl, {"event": "refused", "why": "a unit ended OK with no new pair"})
            return 2


if __name__ == "__main__":
    sys.exit(main())
