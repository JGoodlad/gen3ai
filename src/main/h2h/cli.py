"""CLI of ``main.h2h`` — see ``main/h2h/__init__.py``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from agents.training import eval_ledger as L
from main.h2h import play as PL
from main.h2h import stats as ST


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.h2h", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play", help="play N mirrored pairs of PLAYER vs OPPONENT and append §0b COUNT rows")
    p.add_argument("--player", required=True, help="the MEASURED side: a .zip, a run dir (its last snapshot) or <run>@<step>")
    p.add_argument("--opponent", required=True, help="the other side (same spec forms)")
    p.add_argument("--pairs", type=int, required=True, help="mirrored team pairs (2 games each)")
    p.add_argument("--out", required=True, help="directory for the ledger shard (REFUSED under models/)")
    p.add_argument("--label", default="h2h", help="the row's `run` (the study that produced it)")
    p.add_argument("--purpose", default=PL.DEFAULT_PURPOSE, choices=L.PURPOSES)
    p.add_argument("--seed", type=int, default=0, help="the schedule seed (team pairs + battle seeds)")
    p.add_argument("--schedule-key", default=None,
                   help="default: a digest of the two checkpoints' hashes, ORDER-INDEPENDENT (A-vs-B and B-vs-A "
                        "draw the same team pairs)")
    p.add_argument("--batch-pairs", type=int, default=PL.DEFAULT_BATCH_PAIRS, help="pairs per row / per cycle")
    p.add_argument("--device", default="cpu")
    p.add_argument("--backend", default="", choices=("", "eager", "graph", "aot"),
                   help="T2 backend (default: graph on CUDA, eager on CPU)")
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=4, help="the Rust core's worker threads")
    p.add_argument("--torch-threads", type=int, default=4, help="intra-op threads of a CPU forward")
    p.add_argument("--front", default="proc", choices=("proc", "ffi"))
    p.add_argument("--profile", default="release", choices=("release", "selfcheck"))
    r = sub.add_parser("read", help="pool a directory's rows per edge (validates every row; never mixes regimes)")
    r.add_argument("dir")
    r.add_argument("--json", action="store_true")
    r.add_argument("--regime", default=None, help="pool only this regime_id")
    return ap


def cmd_play(a: argparse.Namespace) -> int:
    L.refuse_under_models(a.out)
    player, opponent = PL.resolve_player(a.player), PL.resolve_player(a.opponent)
    compute = PL.Compute(device=a.device, backend=a.backend, n_envs=a.n_envs, threads=a.threads,
                         torch_threads=a.torch_threads, front=a.front, profile=a.profile)
    summ = PL.play_edge(a.out, player, opponent, pairs=a.pairs, batch_pairs=a.batch_pairs, schedule_seed=a.seed,
                        schedule_key=a.schedule_key, purpose=a.purpose, run_label=a.label, compute=compute)
    print(ST.format_edge(summ))
    print(json.dumps(summ, sort_keys=True))
    return 0


def cmd_read(a: argparse.Namespace) -> int:
    if not Path(a.dir).exists():
        print(f"[h2h] no such directory: {a.dir}", file=sys.stderr)
        return 2
    rows = L.read_rows(a.dir)
    edges = ST.group_edges(rows, regime_id=a.regime)
    out = [ST.edge_summary(rs) for _k, rs in sorted(edges.items(), key=lambda kv: (kv[1][0]["player"]["id"], kv[1][0]["opponent"]["id"]))]
    if a.json:
        print(json.dumps(out, indent=1, sort_keys=True))
    else:
        for s in out:
            print(ST.format_edge(s))
        print(f"[h2h] {len(rows)} rows, {len(out)} edge(s), all valid under {L.SCHEMA}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    a = _parser().parse_args(argv)
    return {"play": cmd_play, "read": cmd_read}[a.cmd](a)
