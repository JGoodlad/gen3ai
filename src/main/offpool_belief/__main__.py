"""``python -m main.offpool_belief`` — the off-pool belief meter's three steps, and the belief-only fine-tune.

    bank       play the pilot's greedy self-play against ONE registered panel and freeze the facing seat's rows
    score      one checkpoint on one or more banks (CPU; two forwards a row: its belief, and the prior alone)
    table      score files -> the team-clustered table (belief vs the Smogon prior; arms vs a reference arm)
    trainbank  TRAINING rows: the same game loop against a slice of a registered ``--team-mix`` corpus
    finetune   train one checkpoint's three belief deltas on training banks, everything else frozen (``--check``
               resolves everything and stops before the first step)

The meter is CPU. Every bank, score and fine-tune is a file OUTSIDE every checkout and outside ``models/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from main.offpool_belief import DEFAULT_PILOT, OFFPOOL_PANELS, HOME_PANEL, OffPoolBeliefError


def _pilot(ref: str) -> "tuple[str, Optional[str]]":
    """``ref`` is a checkpoint .zip, or a registered baseline NAME (resolved through the baseline registry)."""
    from agents.training import baselines

    if ref.endswith(".zip"):
        return ref, None
    try:
        return baselines.resolve(ref).zip_path, ref
    except baselines.BaselineError as e:
        raise OffPoolBeliefError(f"--pilot {ref!r} is neither a .zip nor a resolvable baseline: {e}") from e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.offpool_belief", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank", help="build one panel's row bank")
    b.add_argument("--panel", required=True, help=f"a registered panel: {HOME_PANEL} (the in-pool reference) or "
                                                  f"one of {', '.join(OFFPOOL_PANELS)}")
    b.add_argument("--pilot", default=DEFAULT_PILOT, help="the checkpoint that plays both seats: a .zip or a "
                                                          "baseline name (default: %(default)s)")
    b.add_argument("--out", required=True, help="the bank directory (new; outside every checkout and models/)")
    b.add_argument("--games-per-team", type=int, default=None)
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--n-envs", type=int, default=None, help="games in flight (does not change a game)")
    b.add_argument("--threads", type=int, default=1, help="torch intra-op threads")
    s = sub.add_parser("score", help="score one checkpoint on banks")
    s.add_argument("--bank", action="append", required=True, help="a bank directory (repeat: one per panel)")
    s.add_argument("--ckpt", required=True, help="the checkpoint .zip to score")
    s.add_argument("--label", required=True, help="what the table calls this checkpoint")
    s.add_argument("--out", required=True, help="the score .json (outside every checkout and models/)")
    s.add_argument("--mix", type=float, action="append", default=[],
                   help="also report the belief blended with the prior at this λ (0 < λ < 1; repeat)")
    s.add_argument("--threads", type=int, default=1)
    t = sub.add_parser("table", help="score files -> the clustered table")
    t.add_argument("--arm", action="append", required=True, metavar="NAME=a.json[,b.json…]",
                   help="one arm: its name and its checkpoints' score files (repeat)")
    t.add_argument("--versus", default=None, help="the reference arm every other arm is paired against")
    t.add_argument("--panel", action="append", default=None, help="restrict to these panels (repeat)")
    t.add_argument("--md", default=None, help="write the markdown table here (default: stdout)")
    t.add_argument("--json", dest="json_out", default=None, help="write the table document here")
    r = sub.add_parser("trainbank", help="build one shard of TRAINING rows from a --team-mix corpus")
    r.add_argument("--corpus", required=True, help="a registered --team-mix corpus (utils.team_corpora); a "
                                                   "human-data corpus is refused")
    r.add_argument("--tier", default="", help="the corpus tier (default: the corpus's default tier)")
    r.add_argument("--teams", type=int, required=True, help="teams in this shard (one game each)")
    r.add_argument("--offset", type=int, required=True, help="where the shard starts in the seeded permutation")
    r.add_argument("--pilot", default=DEFAULT_PILOT)
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=None, help="the permutation's seed: ONE per fine-tune")
    r.add_argument("--n-envs", type=int, default=None)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--check", action="store_true", help="resolve the shard (corpus, draw, held-out contamination) "
                                                        "and print it; play nothing, write nothing")
    f = sub.add_parser("finetune", help="the belief-only fine-tune (designs/tools/offpool_belief.md)")
    f.add_argument("--init", default=DEFAULT_PILOT, help="the checkpoint to fine-tune: a .zip or a baseline name")
    f.add_argument("--train", action="append", required=True, help="a training bank directory (repeat)")
    f.add_argument("--val", required=True, help="the validation bank: a disjoint shard of the same corpus")
    f.add_argument("--out", required=True, help="the fine-tune directory (new)")
    f.add_argument("--device", default="cpu", help="cpu, or cuda under a GPU lease")
    f.add_argument("--threads", type=int, default=1)
    f.add_argument("--check", action="store_true", help="resolve and print the plan; train nothing, write nothing")
    return p


def _trainbank(a: argparse.Namespace) -> int:
    from main.offpool_belief import finetune as FT

    zip_path, name = _pilot(a.pilot)
    if a.check:
        kw = {} if a.seed is None else {"seed": a.seed}
        doc = FT.shard_plan(a.corpus, a.tier, zip_path, Path(a.out), teams=a.teams, offset=a.offset, **kw)
        print(json.dumps(doc, indent=1))
        print(f"CHECK OK: {doc['n_games']} teams of {a.corpus}, 0 held-out hits; nothing played, nothing written")
        return 0
    kw = {k: v for k, v in (("seed", a.seed), ("n_envs", a.n_envs)) if v is not None}
    FT.build_trainbank(a.corpus, a.tier, zip_path, Path(a.out), teams=a.teams, offset=a.offset, pilot_name=name,
                       threads=a.threads, **kw)
    return 0


def _finetune(a: argparse.Namespace) -> int:
    from main.offpool_belief import finetune as FT

    zip_path, _name = _pilot(a.init)
    train, val, out = [Path(t) for t in a.train], Path(a.val), Path(a.out)
    if a.check:
        from utils.torch_state_guard import torch_globals

        with torch_globals(num_threads=int(a.threads)):
            doc = FT.plan(zip_path, train, val, out, dict(FT.RECIPE), banks_optional=True)
        print(json.dumps(doc, indent=1))
        rows = ("banks not built yet" if doc["train_rows"] is None else
                f"{doc['train_rows']} training rows, {doc['val_rows']} validation rows, {doc['total_steps']} steps")
        print(f"CHECK OK: {rows}; {doc['trained_parameters']} trained / {doc['frozen_parameters']} frozen "
              "parameters; nothing trained, nothing written")
        return 0
    FT.run(zip_path, train, val, out, device=a.device, threads=a.threads)
    return 0


def _bank(a: argparse.Namespace) -> int:
    from main.offpool_belief import bank as B

    zip_path, name = _pilot(a.pilot)
    kw = {k: v for k, v in (("games_per_team", a.games_per_team), ("seed", a.seed), ("n_envs", a.n_envs))
          if v is not None}
    B.build(a.panel, zip_path, Path(a.out), pilot_name=name, threads=a.threads, **kw)
    return 0


def _score(a: argparse.Namespace) -> int:
    from main.offpool_belief import bank as B
    from main.offpool_belief import panels as PN
    from main.offpool_belief import score as SC

    out = Path(a.out)
    PN.refuse_under_models(out, "the score file")
    PN.refuse_inside_repo(out, "the score file")
    if out.exists():
        raise OffPoolBeliefError(f"{out} exists: a score is never overwritten")
    doc = SC.score_checkpoint(a.ckpt, a.label, [B.load_bank(Path(b)) for b in a.bank], mixes=a.mix,
                              threads=a.threads)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc) + "\n")
    os.replace(tmp, out)
    print(f"[score {a.label}] -> {out}")
    return 0


def _table(a: argparse.Namespace) -> int:
    from main.offpool_belief import score as SC
    from main.offpool_belief import table as T

    arms: Dict[str, List[dict]] = {}
    for spec in a.arm:
        name, sep, files = spec.partition("=")
        if not sep or not name or not files:
            raise OffPoolBeliefError(f"--arm {spec!r}: write NAME=a.json[,b.json]")
        if name in arms:
            raise OffPoolBeliefError(f"--arm {name!r} named twice")
        arms[name] = [SC.load_score(Path(f)) for f in files.split(",") if f]
    doc = T.build(arms, versus=a.versus, panels=a.panel)
    md = T.markdown(doc)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(doc, indent=1) + "\n")
    if a.md:
        Path(a.md).write_text(md)
    else:
        sys.stdout.write(md)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return {"bank": _bank, "score": _score, "table": _table, "trainbank": _trainbank,
                "finetune": _finetune}[a.cmd](a)
    except OffPoolBeliefError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
