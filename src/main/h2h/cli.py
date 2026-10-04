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


def _common(p: argparse.ArgumentParser, request_help: Optional[str] = None) -> None:
    """The arguments ``play`` and ``play-many`` share (the same plan, the same compute)."""
    p.add_argument("--pairs", type=int, required=True, help="mirrored team pairs (2 games each) per cell")
    p.add_argument("--out", default=None,
                   help="the ledger ROOT (default: the run archive's <archive>/_ledger; any other root under models/ "
                        "is REFUSED)")
    p.add_argument("--label", default="h2h", help="the row's `run` (the study that produced it)")
    p.add_argument("--purpose", default=PL.DEFAULT_PURPOSE, choices=L.PURPOSES)
    p.add_argument("--request", default=None,
                   help=request_help or ("the request id (default: derived from the two players, the regime and the "
                                         "schedule, so a re-run resumes it)"))
    p.add_argument("--family", default=None,
                   help="the request family (a registered group-sequential read, e.g. an X5 A/B: "
                        "`python -m main.eval_ledger family-register` first)")
    p.add_argument("--request-kind", default=None, choices=L.REQUEST_KINDS,
                   help="default: ab_cell under --purpose ab, else adhoc")
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


def _compute(a: argparse.Namespace) -> PL.Compute:
    return PL.Compute(device=a.device, backend=a.backend, n_envs=a.n_envs, threads=a.threads,
                      torch_threads=a.torch_threads, front=a.front, profile=a.profile)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.h2h", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play", help="play N mirrored pairs of PLAYER vs OPPONENT and append §0b COUNT rows")
    p.add_argument("--player", required=True, help="the MEASURED side: a .zip, a run dir (its last snapshot) or <run>@<step>")
    p.add_argument("--opponent", required=True, help="the other side (same spec forms)")
    _common(p)
    m = sub.add_parser("play-many", help="play MANY cells (player vs opponent) on ONE engine, each exactly as `play` "
                                         "plays it (module main.h2h.many)")
    src = m.add_mutually_exclusive_group(required=True)
    src.add_argument("--cells", default=None,
                     help="a JSON list of cells: [player, opponent] or {\"player\": ..., \"opponent\": ...}")
    src.add_argument("--players", nargs="+", default=None,
                     help="with --opponents: the CROSS, every player against every opponent (players outer)")
    m.add_argument("--opponents", nargs="+", default=None, help="the cross's other side (with --players)")
    _common(m, request_help="ONE request for EVERY cell (an X5 look); default: each cell its own default request, "
                            "so a re-run resumes, and single-cell `play` on the same arguments shares its rows")
    r = sub.add_parser("read", help="pool a ledger's h2h rows per edge (validates every row; never mixes regimes)")
    r.add_argument("dir", nargs="?", default=None,
                   help="a ledger root, or a legacy flat directory of v1 shards (default: the run archive's ledger)")
    r.add_argument("--json", action="store_true")
    r.add_argument("--regime", default=None, help="pool only this regime_id")
    return ap


#: ``read``: an ESTIMATE-style listing over every h2h row under the root (any request), one regime at a time. A v1
#: row (P0's) lacks only its outcome digest, which a count read does not need.
H2H_READ = L.ReaderDecl(
    name="main.h2h.read", purposes=L.ALL_PURPOSES,
    regime=L.RegimeFilter(protocol=PL.PROTOCOL, play="greedy", opponent_play="greedy", mirrored=True),
    requests="any", selection="include", flags_ok=frozenset({"digest_unrecorded"}), inference="conditional")


def cmd_play(a: argparse.Namespace) -> int:
    if a.out is not None:
        L.check_write_root(a.out)
    player, opponent = PL.resolve_player(a.player), PL.resolve_player(a.opponent)
    compute = _compute(a)
    summ = PL.play_edge(a.out, player, opponent, pairs=a.pairs, batch_pairs=a.batch_pairs, schedule_seed=a.seed,
                        schedule_key=a.schedule_key, purpose=a.purpose, run_label=a.label, compute=compute,
                        request_id=a.request, family=a.family, request_kind=a.request_kind)
    print(ST.format_edge(summ))
    print(json.dumps(summ, sort_keys=True))
    return 0


def cmd_play_many(a: argparse.Namespace) -> int:
    from main.h2h import many as MANY

    if a.out is not None:
        L.check_write_root(a.out)
    if a.players is not None and not a.opponents:
        raise SystemExit("[h2h] --players needs --opponents (the cross)")
    if a.cells is not None and a.opponents:
        raise SystemExit("[h2h] --opponents goes with --players, not --cells")
    specs = MANY.cells_from_file(a.cells) if a.cells is not None else MANY.cross(a.players, a.opponents)
    out = MANY.play_cells(a.out, MANY.resolve_cells(specs), pairs=a.pairs, batch_pairs=a.batch_pairs,
                          schedule_seed=a.seed, schedule_key=a.schedule_key, purpose=a.purpose, run_label=a.label,
                          compute=_compute(a), request_id=a.request, family=a.family, request_kind=a.request_kind)
    for summ in out["cells"]:
        print(ST.format_edge(summ))
    print(json.dumps(out, sort_keys=True))
    return 0


def cmd_read(a: argparse.Namespace) -> int:
    root = Path(a.dir) if a.dir is not None else L.archive_ledger_root()
    if not root.exists():
        print(f"[h2h] no such directory: {root}", file=sys.stderr)
        return 2
    reads = L.read_by_regime(H2H_READ, root=root)
    if a.regime is not None:
        reads = {k: v for k, v in reads.items() if k == a.regime}
    rows = [r for got in reads.values() for r in got.rows]
    out = []
    for got in reads.values():
        edges = ST.group_edges(list(got.rows))
        out.extend(ST.edge_summary(rs) for _k, rs in edges.items())
    out.sort(key=lambda s: (s["player"], s["opponent"], s["regime_id"]))
    if a.json:
        print(json.dumps(out, indent=1, sort_keys=True))
    else:
        for s in out:
            print(ST.format_edge(s))
        print(f"[h2h] {len(rows)} rows, {len(out)} edge(s) in {len(reads)} regime(s), all valid under {L.SCHEMA} "
              f"(v1 rows upgraded on read)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    a = _parser().parse_args(argv)
    return {"play": cmd_play, "play-many": cmd_play_many, "read": cmd_read}[a.cmd](a)
