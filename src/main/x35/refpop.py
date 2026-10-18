"""X35's NODE RULES and the REFERENCE-POPULATION FREEZE (``design_compute_efficiency.md`` §5.6.3-§5.6.4).

THE 10M NODE. A run checkpoints every 1M steps, and a checkpoint's step overshoots its nominal million by the rollout
remainder (``checkpoint_180000055_steps.zip`` is the 180M node). The node of nominal M (a multiple of 10M) is the ONE
``checkpoints/checkpoint_<s>_steps.zip`` with M <= s < M + 1,000,000; none, or two (across the archive's run
directories), is a typed refusal. P* (or T*) must itself be a node.

A1'S ARCHIVE. N_k for k = 1..8 is the node nearest k·(P* − 20M)/8, ties DOWN (exact rational arithmetic, so a tie is
a tie). For P* = 180M that is 20, 40, …, 160M.

THE A-BLOCK of R: P*, P* − 10M, and the midpoints of A1's adjacent pairs (N_2 | N_3, N_4 | N_5, N_6 | N_7). For
P* = 180M: 180, 170, 130, 90, 50M. DECLARED HERE (§5.6.4 is silent): a midpoint that is not a 10M multiple rounds to
the nearest one, ties DOWN; an A-block node that coincides with an archive node or with another A-block node is a
typed refusal (§5.6.4: the A-block is "disjoint by FILE from A1's training nodes").

THE X-BLOCK: candidates from another lineage no arm trains against (production: T* and the registry baseline
``production``; toy: P*). Each is KEPT iff P*'s mirrored-pair score against it -- one OK cell of >= 500 completed
pairs -- lies in [0.20, 0.80]; a score within 1e-9 of either edge is EXCLUDED (standing rule 8: an input within a
rounding error of a decision boundary is excluded, never decided by chance). A candidate without such a cell makes
the freeze REFUSE: R is not frozen with an undecided member.

THE FREEZE writes ONE JSON, write-once, before the first arm: P*, the archive, the A-block, every X candidate with
its band score and decision, the members, and every file's sha256, plus ``content_sha256`` over the canonical JSON.
``verify`` re-hashes every file and the content; ``launch_x35.sh`` runs it before every launch.

    python -m main.x35.refpop plan   --side production --pstar <zip> --archive-dir <run>/checkpoints [...] \
                                     --x tstar=<zip> --x production=baseline:production
    python -m main.x35.refpop freeze <the same> --band-json <cells.json> | --band-ledger <root> --band-family F \
                                     --band-request Q   --out <stage>/reference_population_<side>.json
    python -m main.x35.refpop verify <stage>/reference_population_<side>.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

STRIDE = 10_000_000
CKPT_INTERVAL = 1_000_000
ARCHIVE_K = 8
BAND = (0.20, 0.80)
RULE8_EPS = 1e-9
MIN_PAIRS = 500
SCHEMA = "x35_reference_population_v1"
SIDES = ("production", "toy")
CKPT_RE = re.compile(r"^checkpoint_(\d+)_steps\.zip$")


class RefPopError(ValueError):
    """A node rule or a freeze precondition does not hold (named, never guessed past)."""


def sha256_file(path: "str | os.PathLike[str]") -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def node_of(step: int) -> Optional[int]:
    """The nominal 10M node a checkpoint step belongs to, or None if it is not a node."""
    m = step // STRIDE * STRIDE
    return m if step - m < CKPT_INTERVAL else None


@dataclass(frozen=True)
class Node:
    nominal: int
    step: int
    path: str


def scan_nodes(dirs: Sequence["str | os.PathLike[str]"]) -> Dict[int, Node]:
    """``{nominal: Node}`` over every checkpoint directory (a toy archive spans two run dirs)."""
    out: Dict[int, Node] = {}
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            raise RefPopError(f"{p}: not a checkpoint directory")
        for f in sorted(p.iterdir()):
            m = CKPT_RE.match(f.name)
            if not m:
                continue
            step = int(m.group(1))
            nom = node_of(step)
            if nom is None:
                continue
            if nom in out:
                raise RefPopError(f"two checkpoints claim the {nom:,} node: {out[nom].path} and {f}")
            out[nom] = Node(nom, step, str(f.resolve()))
    return out


def nearest_multiple(x: Fraction) -> int:
    """The 10M multiple nearest ``x``, ties DOWN."""
    q = x / STRIDE
    f = math.floor(q)
    return int((f + 1 if q - f > Fraction(1, 2) else f) * STRIDE)


def archive_targets(p_nominal: int) -> List[int]:
    """A1's N_1..N_8 (nominal steps) for a P* node."""
    if p_nominal % STRIDE:
        raise RefPopError(f"P* {p_nominal:,} is not a 10M node")
    out = [nearest_multiple(Fraction(k * (p_nominal - 2 * STRIDE), ARCHIVE_K)) for k in range(1, ARCHIVE_K + 1)]
    if out[0] < STRIDE or len(set(out)) != ARCHIVE_K:
        raise RefPopError(f"P* {p_nominal:,} is too early for 8 distinct archive nodes ≥ 10M: {out}")
    return out


def a_block_targets(p_nominal: int) -> List[Tuple[str, int]]:
    """The A-block's (role, nominal) pairs (module docstring)."""
    n = archive_targets(p_nominal)
    out = [("pstar", p_nominal), ("pstar-10M", p_nominal - STRIDE)]
    for i, j in ((2, 3), (4, 5), (6, 7)):
        out.append((f"mid(N{i},N{j})", nearest_multiple(Fraction(n[i - 1] + n[j - 1], 2))))
    noms = [x for _, x in out]
    clash = sorted(set(noms) & set(n))
    if clash or len(set(noms)) != len(noms):
        raise RefPopError(f"the A-block {noms} is not disjoint from the archive {n} (or repeats): {clash}")
    return out


def label(prefix: str, nominal: int) -> str:
    return f"{prefix}_{nominal // 1_000_000:03d}m"


@dataclass(frozen=True)
class BandCell:
    score: Optional[float]
    n_pairs: int
    verdict: str
    reasons: Tuple[str, ...] = ()


def band_decision(cell: Optional[BandCell], *, min_pairs: int = MIN_PAIRS) -> Tuple[bool, Optional[str]]:
    """``(kept, why_not)``. Raises :class:`RefPopError` when the cell cannot decide (the freeze refuses)."""
    if cell is None:
        raise RefPopError("no P* cell against this candidate")
    if cell.verdict != "OK" or cell.score is None:
        raise RefPopError(f"the cell is not OK ({cell.verdict}: {list(cell.reasons)})")
    if cell.n_pairs < min_pairs:
        raise RefPopError(f"{cell.n_pairs} completed pairs < {min_pairs}")
    s = cell.score
    for edge in BAND:
        if abs(s - edge) <= RULE8_EPS:
            return False, f"score {s!r} within {RULE8_EPS} of the band edge {edge} (rule 8: excluded)"
    if s < BAND[0] or s > BAND[1]:
        return False, f"score {s:.4f} outside [{BAND[0]}, {BAND[1]}]"
    return True, None


def _canon(doc: Mapping[str, Any]) -> bytes:
    return json.dumps({k: v for k, v in doc.items() if k != "content_sha256"}, sort_keys=True,
                      separators=(",", ":")).encode()


def _node_entry(n: Node, role: str, lab: str) -> Dict[str, Any]:
    return {"role": role, "label": lab, "nominal": n.nominal, "step": n.step, "path": n.path,
            "sha256": sha256_file(n.path)}


def _need(nodes: Mapping[int, Node], nominal: int, what: str) -> Node:
    if nominal not in nodes:
        raise RefPopError(f"{what}: no checkpoint for the {nominal:,} node in the archive directories")
    return nodes[nominal]


def plan(*, side: str, pstar: str, archive_dirs: Sequence[str], label_prefix: str) -> Dict[str, Any]:
    """The archive and the A-block resolved to files (no hashing, no band): what ``freeze`` would freeze."""
    if side not in SIDES:
        raise RefPopError(f"side {side!r} is not one of {SIDES}")
    m = CKPT_RE.match(Path(pstar).name)
    if not m:
        raise RefPopError(f"P* {pstar} is not a checkpoints/checkpoint_<N>_steps.zip")
    pn = node_of(int(m.group(1)))
    if pn is None:
        raise RefPopError(f"P* {pstar} is not a 10M node")
    nodes = scan_nodes(archive_dirs)
    p_node = _need(nodes, pn, "P*")
    if Path(p_node.path) != Path(pstar).resolve():
        raise RefPopError(f"P* {pstar} is not the archive's own {pn:,} node ({p_node.path})")
    arch = [(k, _need(nodes, t, f"N_{k}")) for k, t in enumerate(archive_targets(pn), start=1)]
    ablock = [(role, _need(nodes, t, f"A-block {role}")) for role, t in a_block_targets(pn)]
    return {"side": side, "pstar": p_node, "archive": arch, "a_block": ablock, "label_prefix": label_prefix}


def freeze(*, side: str, pstar: str, archive_dirs: Sequence[str], label_prefix: str,
           x_candidates: Sequence[Tuple[str, str]], band: Mapping[str, Optional[BandCell]], out: str,
           tool_commit: str = "", now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Freeze R and write it (write-once). ``x_candidates`` = [(label, zip path)]; ``band`` = {label: P*'s cell}."""
    if os.path.exists(out):
        raise RefPopError(f"{out} exists: R is frozen ONCE (a new call is a new file)")
    pl = plan(side=side, pstar=pstar, archive_dirs=archive_dirs, label_prefix=label_prefix)
    p_node: Node = pl["pstar"]
    xs: List[Dict[str, Any]] = []
    for lab, path in x_candidates:
        try:
            kept, why = band_decision(band.get(lab))
        except RefPopError as exc:
            raise RefPopError(f"X candidate {lab}: {exc} -- R cannot be frozen with an undecided member") from exc
        c = band[lab]
        assert c is not None
        sha = sha256_file(path)
        if sha == sha256_file(p_node.path):
            raise RefPopError(f"X candidate {lab} is P* itself")
        xs.append({"label": lab, "path": str(Path(path).resolve()), "sha256": sha, "band_score": c.score,
                   "band_n_pairs": c.n_pairs, "kept": kept, "excluded_because": why})
    archive = [dict(_node_entry(n, f"N_{k}", label(label_prefix, n.nominal)), k=k) for k, n in pl["archive"]]
    ablock = [_node_entry(n, role, label(label_prefix, n.nominal) if role != "pstar" else f"{label_prefix}_pstar")
              for role, n in pl["a_block"]]
    members = ([{"block": "A", "label": e["label"], "path": e["path"], "sha256": e["sha256"]} for e in ablock]
               + [{"block": "X", "label": x["label"], "path": x["path"], "sha256": x["sha256"]} for x in xs
                  if x["kept"]])
    t = now or dt.datetime.now(dt.timezone.utc)
    doc: Dict[str, Any] = {
        "schema": SCHEMA, "side": side, "frozen_at": t.isoformat(), "frozen_at_epoch": int(t.timestamp()),
        "tool_commit": tool_commit, "rule": {"stride": STRIDE, "archive_k": ARCHIVE_K, "band": list(BAND),
                                             "rule8_eps": RULE8_EPS, "min_pairs": MIN_PAIRS,
                                             "midpoint_rounding": "nearest 10M, ties down"},
        "pstar": ablock[0] | {"role": "pstar"}, "archive": archive, "a_block": ablock, "x_block": xs,
        "members": members,
    }
    doc["content_sha256"] = hashlib.sha256(_canon(doc)).hexdigest()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{out}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, out)
    return doc


def verify(path: str) -> List[str]:
    """Every problem with a frozen R file: its content hash, and each named file's sha256 now."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return [f"{path}: unreadable ({exc})"]
    probs: List[str] = []
    if doc.get("schema") != SCHEMA:
        probs.append(f"schema {doc.get('schema')!r} != {SCHEMA}")
    if hashlib.sha256(_canon(doc)).hexdigest() != doc.get("content_sha256"):
        probs.append("content_sha256 does not match the file's content (edited after the freeze?)")
    seen: Dict[str, str] = {}
    for e in [doc.get("pstar") or {}, *doc.get("archive", []), *doc.get("a_block", []), *doc.get("x_block", [])]:
        p, want = e.get("path"), e.get("sha256")
        if not p or not want:
            probs.append(f"an entry without path / sha256: {e.get('label')}")
            continue
        if p in seen:
            if seen[p] != want:
                probs.append(f"{p}: two recorded sha256s")
            continue
        seen[p] = want
        if not os.path.exists(p):
            probs.append(f"{p}: missing")
        elif sha256_file(p) != want:
            probs.append(f"{p}: sha256 changed since the freeze")
    return probs


# ------------------------------------------------------------------------------------------------------- CLI
def _band_from_json(path: str) -> Dict[str, Optional[BandCell]]:
    raw = json.loads(Path(path).read_text())
    return {k: BandCell(score=v.get("score"), n_pairs=int(v.get("n_pairs", 0)), verdict=str(v.get("verdict", "")),
                        reasons=tuple(v.get("reasons") or ())) for k, v in raw.items()}


def _band_from_ledger(root: str, family: str, request: str, pstar_sha: str,
                      xs: Sequence[Tuple[str, str]]) -> Dict[str, Optional[BandCell]]:
    from main.h2h.cross import look_cells

    cells = look_cells(root, family, request, "gen3_eval_protocol_v1_h2h")
    out: Dict[str, Optional[BandCell]] = {}
    for lab, path in xs:
        sha = sha256_file(path)
        hit = [c for c in cells if c.player == pstar_sha and c.opponent == sha]
        if len(hit) > 1:
            raise RefPopError(f"X candidate {lab}: {len(hit)} cells in request {request}")
        out[lab] = (BandCell(score=hit[0].score, n_pairs=hit[0].n_pairs, verdict=hit[0].verdict,
                             reasons=tuple(hit[0].reasons)) if hit else None)
    return out


def _resolve_x(spec: str) -> Tuple[str, str]:
    lab, _, ref = spec.partition("=")
    if not lab or not ref:
        raise RefPopError(f"--x wants <label>=<zip | baseline:NAME>, got {spec!r}")
    if ref.startswith("baseline:"):
        from agents.training.baselines import resolve

        return lab, str(resolve(ref.split(":", 1)[1]).zip_path)
    return lab, ref


def _tool_commit() -> str:
    from utils.git import get_git_hash

    return get_git_hash()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x35.refpop", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "freeze"):
        p = sub.add_parser(name)
        p.add_argument("--side", required=True, choices=SIDES)
        p.add_argument("--pstar", required=True, help="the start node's checkpoints/checkpoint_<N>_steps.zip")
        p.add_argument("--archive-dir", action="append", required=True, help="a checkpoints/ dir (repeatable)")
        p.add_argument("--label-prefix", default=None, help="node labels (default: x26 / toy by side)")
        p.add_argument("--x", action="append", default=[], help="an X candidate <label>=<zip | baseline:NAME>")
        if name == "freeze":
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument("--band-json", help="{label: {score, n_pairs, verdict, reasons}} (tests / hand reads)")
            g.add_argument("--band-ledger", help="an eval-ledger root holding P*'s band cells")
            p.add_argument("--band-family")
            p.add_argument("--band-request")
            p.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("file")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "verify":
            probs = verify(a.file)
            for x in probs:
                print(f"[refpop verify] {x}", file=sys.stderr)
            print(f"[refpop verify] {a.file}: {'OK' if not probs else f'{len(probs)} problem(s)'}")
            return 0 if not probs else 1
        prefix = a.label_prefix or ("x26" if a.side == "production" else "toy")
        xs = [_resolve_x(s) for s in a.x]
        if a.cmd == "plan":
            pl = plan(side=a.side, pstar=a.pstar, archive_dirs=a.archive_dir, label_prefix=prefix)
            print(f"P*      {pl['pstar'].nominal:>13,}  {pl['pstar'].path}")
            for k, n in pl["archive"]:
                print(f"N_{k}     {n.nominal:>13,}  {n.path}")
            for role, n in pl["a_block"]:
                print(f"A {role:12s} {n.nominal:>13,}  {n.path}")
            for lab, path in xs:
                print(f"X {lab:12s} {path}")
            print("band cells to play (NOT run by this tool; 500 mirrored pairs, greedy vs greedy):\n"
                  f"  python -m main.h2h play-many --players {pl['pstar'].path} --opponents "
                  + " ".join(p for _, p in xs) + " --pairs 500 --purpose ab --family x35_band "
                  f"--request x35_band_{a.side} --device cpu --torch-threads 1 --threads 2")
            return 0
        if a.band_json:
            band = _band_from_json(a.band_json)
        else:
            if not (a.band_family and a.band_request):
                raise RefPopError("--band-ledger needs --band-family and --band-request")
            band = _band_from_ledger(a.band_ledger, a.band_family, a.band_request, sha256_file(a.pstar), xs)
        doc = freeze(side=a.side, pstar=a.pstar, archive_dirs=a.archive_dir, label_prefix=prefix,
                     x_candidates=xs, band=band, out=a.out, tool_commit=_tool_commit())
        print(f"[refpop] FROZE {a.out}: {len(doc['members'])} members "
              f"({sum(m['block'] == 'X' for m in doc['members'])} X), content {doc['content_sha256'][:12]}")
        return 0
    except RefPopError as exc:
        print(f"[refpop] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
