"""THE LIVE IDENTITY GUARD: the S′ that PLAYS live is, decision for decision, the S′ a banked OFFLINE read scored.

A live searched game (``main.anchors --our-search``, :mod:`main.anchors.search_side`) writes one record per decision
of ours to ``<cell>/search_decisions.jsonl``; each record that reaches the search (searched or excluded) carries its
``key`` (``su_live_v1|<deal>|t<turn>|n<n>``) and the decision's IDENTITY, ``log_sha256`` (the honest input log handed
to the search, :func:`main.anchors.forced_branch.log_sha256`) and ``frame_sha256`` (the live core row,
:func:`main.anchors.input_log.frame_sha256`). An offline read (``main.value_search.sign_gate``: the dlive read's
``selected.jsonl`` and ``scored.jsonl``) ran the SAME searcher (``LiveSearcher._search``) on kept raw decisions, each
selected row carrying its ``key``, its ``log`` and its ``frame_sha256``.

A live record is MATCHED to a banked state iff the scored file holds its ``key`` AND that key's selected row has the
live record's ``frame_sha256`` and ``log_sha256(selected log)`` equal to the live ``log_sha256``: the same battle,
the same history of both sides, the same view. A key whose digests differ is ``key_only`` (the searched game reached
that turn and decision index by another history: not the same state, never compared); a key the scored file does not
hold is ``unbanked``. For every matched state the guard compares, exactly as recorded (both rounded to 6 dp by the
same code): ``searched``, ``why`` (the exclusion), ``played`` (the pick), ``pi_argmax``, ``q``, ``log_pi``,
``root_v``. Any difference is a MISMATCH.

VERDICT: PASS iff ``n_matched >= min_matched``, zero mismatches and every searched-path record carries its identity;
else FAIL, with the reason. It reads no game result (never ``games.jsonl``, a summary or a capture) and prints counts.

THE RE-SCORE (``rescore``, the read's own guard on FRESH deals that no bank holds). A live searched-path record whose
key is in the hashed sample (``main.anchors.search_side.rescore_sampled``, 1 in 16) carries ``rescore_sample: true``,
the honest ``log``, the ``mask`` and the ``tokens``. ``rescore`` takes those records of every attempt dir in the
REGISTERED ORDER (:func:`registered_order`: sha256(key) FIRST, across every cell, then the panel in the order given,
the unit seed, the game and the attempt dir as tie-breaks, so any first N is a uniform draw over panels and seeds),
the first ``--max``, and re-runs S′ offline at THIS checkout exactly as the dlive
read's scorer did (``sign_gate.GateScorer``: its root guards, then ``LiveSearcher._search``; S′ only, no referee, no
Speed marker; one torch / BLAS thread, as the live unit plays), on the registered engine and the model's checked
bytes. Each state's ``searched``, ``why``, ``played``, ``q``, ``log_pi`` and ``root_v`` must equal the live record's
exactly; a root guard that refuses is a mismatch too. PASS iff >= ``--min`` states were re-scored and none differs.
Resumable: each re-scored state is appended to ``<out>.rows.jsonl`` and skipped on a restart.

THE BANK TOOTH (``bank-tooth``): is THIS checkout's S′ the one a banked offline read scored? On a hashed sample of the
bank's ``selected.jsonl`` (:func:`bank_sampled`: ``int(sha256(key)[:8], 16) % 64 == 0``), in sha256(key) order,
skipping rows the bank never scored, the first ``--max`` (80) are re-run through the same S′ path as ``rescore`` (the
scorer's root guards, then ``_search``; no referee, no Speed marker; one thread), and the produced ``search`` block
must equal the bank's ``scored.jsonl`` row's exactly on every field but the CPU / wall timings
(:data:`VOLATILE`).

    python -m main.value_search.live_identity check --cells DIR --selected F --scored F --min-matched N [--out F]
    python -m main.value_search.live_identity rescore --cells DIR --panels P[,P..] --model ZIP --model-sha256 H
        --engine STAMP --out F [--max 600] [--min 300]
    python -m main.value_search.live_identity bank-tooth --selected F --scored F --search PRESET@T --model ZIP
        --model-sha256 H --engine STAMP --out F [--max 80]

``--cells``: the searched label's cells dir (``<state>/anchor/<group>/<label>/cells``, its kakuna state's
``plan.json`` one level up); every attempt dir under it is read (``<unit>`` and ``<unit>.partial.*``) and reported per
dir. Exit 0 PASS, 1 FAIL, 2 input error (an unreadable or missing file, a key twice in ``selected`` or ``scored``, a
sampled record whose flag is not its key's hash).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "gen3_live_identity_v1"
RESCORE_SCHEMA = "gen3_live_rescore_v1"
#: the fields a re-score compares (``pi_argmax`` is its input, guarded by the search's own forward check)
RESCORE_FIELDS: Tuple[str, ...] = ("searched", "why", "played", "q", "log_pi", "root_v")
RESCORE_MAX, RESCORE_MIN = 600, 300
BANK_SCHEMA = "gen3_live_bank_tooth_v1"
BANK_MOD, BANK_MAX = 64, 80
#: a search block's timing fields: never compared (everything else a scored row's ``search`` holds is)
VOLATILE = frozenset({"cpu_s", "wall_s", "process_cpu_s", "world_cpu_s", "playout_cpu_s"})
LOG = "search_decisions.jsonl"
#: the fields compared on a matched state: (live record field, scored row ``search`` field)
FIELDS: Tuple[str, ...] = ("searched", "why", "played", "pi_argmax", "q", "log_pi", "root_v")
PASS, FAIL = "PASS", "FAIL"


class IdentityInputError(ValueError):
    """An input the guard cannot read honestly (named; exit 2)."""


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    try:
        return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    except (OSError, ValueError) as exc:
        raise IdentityInputError(f"{path}: unreadable ({exc})") from None


def _by_key(rows: Sequence[Mapping[str, Any]], what: str) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for r in rows:
        k = r.get("key")
        if not isinstance(k, str):
            raise IdentityInputError(f"{what}: a row without a key")
        if k in out:
            raise IdentityInputError(f"{what}: key {k} appears twice (a state must be one row)")
        out[k] = r
    return out


def bank(selected: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{key: {frame_sha256, log_sha256, panel, offline}}`` over the SCORED states (``offline``: the scored row's
    ``search`` block plus its ``pi_argmax``)."""
    from main.anchors.forced_branch import log_sha256

    sel = _by_key(selected, "selected")
    out: Dict[str, Dict[str, Any]] = {}
    for k, r in _by_key(scored, "scored").items():
        s = sel.get(k)
        if s is None:
            raise IdentityInputError(f"scored: key {k} was never selected")
        if not isinstance(s.get("log"), Mapping) or not isinstance(s.get("frame_sha256"), str):
            raise IdentityInputError(f"selected: key {k} carries no log / frame_sha256")
        off = dict(r.get("search") or {})
        off["pi_argmax"] = r.get("pi_argmax")
        out[k] = {"frame_sha256": s["frame_sha256"], "log_sha256": log_sha256(s["log"]), "panel": r.get("panel"),
                  "offline": off}
    return out


def attempt_dirs(cells: Path) -> List[Path]:
    """Every attempt dir under ``cells`` holding a decision log (``<unit>`` and ``<unit>.partial.*``), sorted."""
    return sorted(p for p in Path(cells).iterdir() if p.is_dir() and (p / LOG).is_file())


def check(cells: Path, selected: Path, scored: Path, *, min_matched: int) -> Dict[str, Any]:
    """The guard's document (module docstring)."""
    from utils.hashing import sha256_file

    if not isinstance(min_matched, int) or isinstance(min_matched, bool) or min_matched < 1:
        raise IdentityInputError(f"min_matched {min_matched!r}: an int >= 1")
    cells = Path(cells)
    if not cells.is_dir():
        raise IdentityInputError(f"{cells}: no cells dir")
    for p in (selected, scored):
        if not Path(p).is_file():
            raise IdentityInputError(f"{p}: missing")
    banked = bank(_jsonl(Path(selected)), _jsonl(Path(scored)))
    tot = {"n_records": 0, "n_matched": 0, "n_key_only": 0, "n_unbanked": 0, "n_no_identity": 0}
    by_dir: Dict[str, Dict[str, int]] = {}
    by_panel: Dict[str, Dict[str, int]] = {}
    mism: List[Dict[str, Any]] = []
    for d in attempt_dirs(cells):
        row = by_dir.setdefault(d.name, {k: 0 for k in tot} | {"mismatches": 0})
        for rec in _jsonl(d / LOG):
            if "key" not in rec:                      # before_turn / one_legal: never reached the search
                continue
            row["n_records"] += 1
            if not (isinstance(rec.get("log_sha256"), str) and isinstance(rec.get("frame_sha256"), str)):
                row["n_no_identity"] += 1
                continue
            b = banked.get(rec["key"])
            if b is None:
                row["n_unbanked"] += 1
                continue
            if b["frame_sha256"] != rec["frame_sha256"] or b["log_sha256"] != rec["log_sha256"]:
                row["n_key_only"] += 1
                by_panel.setdefault(str(b["panel"]), {"n_matched": 0, "n_key_only": 0})["n_key_only"] += 1
                continue
            row["n_matched"] += 1
            by_panel.setdefault(str(b["panel"]), {"n_matched": 0, "n_key_only": 0})["n_matched"] += 1
            for f in FIELDS:
                live, off = rec.get(f), b["offline"].get(f)
                if live != off:
                    row["mismatches"] += 1
                    mism.append({"key": rec["key"], "cell": d.name, "field": f, "live": live, "offline": off})
    for row in by_dir.values():
        for k in tot:
            tot[k] += row[k]
    why: List[str] = []
    if tot["n_matched"] < min_matched:
        why.append(f"{tot['n_matched']} matched state(s), fewer than min_matched {min_matched}")
    if mism:
        why.append(f"{len(mism)} mismatch(es) on matched states (first {mism[0]['key']} {mism[0]['field']})")
    if tot["n_no_identity"]:
        why.append(f"{tot['n_no_identity']} searched-path record(s) carry no log_sha256 / frame_sha256")
    return {"schema": SCHEMA,
            "inputs": {"cells": str(cells), "selected": {"path": str(selected), "sha256": sha256_file(Path(selected))},
                       "scored": {"path": str(scored), "sha256": sha256_file(Path(scored))},
                       "min_matched": min_matched, "fields": list(FIELDS)},
            **tot, "by_panel": dict(sorted(by_panel.items())), "by_dir": dict(sorted(by_dir.items())),
            "mismatches": mism, "verdict": FAIL if why else PASS, "reason": "; ".join(why) or None}


# ------------------------------------------------------------------------------------------------ the re-score
def _unit_of(dir_name: str) -> str:
    return dir_name.split(".partial.", 1)[0]


def sampled(cells: Path, panels: Sequence[str]) -> List[Dict[str, Any]]:
    """Every sampled searched-path record under ``cells`` (every attempt dir), in the REGISTERED ORDER, each as
    ``{cell, panel, unit_seed, game, key, rec}``. Refuses a dir whose unit is not in the plan or not on ``panels``, and
    a record whose ``rescore_sample`` flag is not its key's hash."""
    import hashlib

    from main.anchors.search_side import rescore_sampled
    from main.x35 import kakuna as KA

    cells = Path(cells)
    try:
        plan = {u.name: u for u in KA.load_plan(cells.parent)}
    except (OSError, ValueError, TypeError) as exc:
        raise IdentityInputError(f"{cells.parent}: no readable plan.json ({exc})") from None
    out = []
    for d in attempt_dirs(cells):
        u = plan.get(_unit_of(d.name))
        if u is None or u.game is None:
            raise IdentityInputError(f"{d.name}: not a per-game unit of {cells.parent}'s plan")
        if u.teamset not in panels:
            raise IdentityInputError(f"{d.name}: panel {u.teamset!r} is not one of {list(panels)}")
        for rec in _jsonl(d / LOG):
            if "key" not in rec:
                continue
            flag = rec.get("rescore_sample") is True
            if flag != rescore_sampled(rec["key"]):
                raise IdentityInputError(f"{d.name} {rec['key']}: rescore_sample {flag} is not the key's hash")
            if flag:
                out.append({"order": (hashlib.sha256(rec["key"].encode()).hexdigest(), list(panels).index(u.teamset),
                                      int(u.unit_seed), int(u.game), d.name),
                            "cell": d.name, "panel": u.teamset, "unit_seed": int(u.unit_seed), "game": int(u.game),
                            "key": rec["key"], "rec": rec})
    return [{k: v for k, v in e.items() if k != "order"} for e in sorted(out, key=lambda e: e["order"])]


def registered_order(cells: Path, panels: Sequence[str], max_n: int) -> List[Dict[str, Any]]:
    """The first ``max_n`` sampled records of :func:`sampled`: the states a re-score must cover, ``[{cell, key}]``."""
    return [{"cell": e["cell"], "key": e["key"]} for e in sampled(cells, panels)[:max_n]]


def one_thread() -> None:
    """One torch / BLAS thread and no GPU, as the live unit plays (``kakuna.unit_env``); set before torch loads, and
    torch's own count again in case it already has."""
    import os

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)


def _scorer_for(model: str, model_sha256: str, engine: str, search: str) -> Any:
    """``sign_gate.GateScorer`` on ``model`` (its sha256 checked), at this checkout's engine (checked), one thread."""
    from main.anchors.server import engine_stamp
    from utils.hashing import sha256_file

    here = engine_stamp()
    if here != engine:
        raise IdentityInputError(f"this checkout's engine is {here}, the registered one {engine}")
    got = sha256_file(Path(model)) if Path(model).is_file() else None
    if got != model_sha256:
        raise IdentityInputError(f"{model}: sha256 {str(got)[:12]} is not {model_sha256[:12]}")
    one_thread()
    from main.value_search.sign_gate import GateScorer

    return GateScorer(model, search)


def rescore_one(scorer: Any, rec: Mapping[str, Any]) -> Dict[str, Any]:
    """S′ offline on one sampled live record (the dlive scorer's root guards, then the searcher's ``_search``):
    ``{"search": the offline record}`` or ``{"error": the root guard's refusal}``."""
    import numpy as np

    from agents.battle.core_obs import wrap_row

    try:
        _truth, d = scorer.root(rec)
        row = np.array(wrap_row(d["frame"]), dtype=np.float32)
        s = scorer.s._search(rec["log"], int(rec["at"]), rec["side"], row, np.asarray(rec["mask"], dtype=np.int8), d,
                             rec["key"], int(rec["pi_argmax"]))
    except Exception as exc:          # a refused root or forward is a MISMATCH, named, never a skip
        return {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    return {"search": s}


def compare(live: Mapping[str, Any], got: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The fields of a re-scored state that differ from the live record (``root_guard`` when it was refused)."""
    if "error" in got:
        return [{"field": "root_guard", "live": None, "offline": got["error"]}]
    off = got["search"]
    return [{"field": f, "live": live.get(f), "offline": off.get(f)} for f in RESCORE_FIELDS
            if live.get(f) != off.get(f)]


def rescore(cells: Path, panels: Sequence[str], *, out: Path, model: str, model_sha256: str, engine: str,
            max_n: int = RESCORE_MAX, min_n: int = RESCORE_MIN, scorer: Any = None) -> Dict[str, Any]:
    """The re-score's document (module docstring); ``scorer``: a test's stand-in for ``sign_gate.GateScorer``."""
    for k, v in (("max", max_n), ("min", min_n)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise IdentityInputError(f"--{k} {v!r}: an int >= 1")
    order = sampled(Path(cells), panels)[:max_n]
    searches = sorted({str(e["rec"].get("search")) for e in order})
    if len(searches) > 1:
        raise IdentityInputError(f"the sampled records name {len(searches)} searches {searches}")
    if scorer is None and order:
        scorer = _scorer_for(model, model_sha256, engine, searches[0])
    rows_p = Path(str(out) + ".rows.jsonl")
    rows_p.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["cell"], r["key"]): r for r in (_jsonl(rows_p) if rows_p.is_file() else [])}
    try:
        for e in order:
            if (e["cell"], e["key"]) in done:
                continue
            got = rescore_one(scorer, e["rec"])
            row = {"cell": e["cell"], "key": e["key"], "panel": e["panel"], "mismatches": compare(e["rec"], got),
                   "cpu_s": (got.get("search") or {}).get("cpu_s")}
            with open(rows_p, "a") as f:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            done[(e["cell"], e["key"])] = row
            print(f"[live_identity] rescored {len(done)} / {len(order)}", flush=True)     # counts only
    finally:
        if scorer is not None and hasattr(scorer, "close"):
            scorer.close()
    rows = [done[(e["cell"], e["key"])] for e in order]
    mism = [dict(m, key=r["key"], cell=r["cell"]) for r in rows for m in r["mismatches"]]
    why = []
    if len(rows) < min_n:
        why.append(f"{len(rows)} state(s) re-scored, fewer than --min {min_n}")
    if mism:
        why.append(f"{len(mism)} mismatch(es) (first {mism[0]['key']} {mism[0]['field']})")
    by_panel: Dict[str, int] = {}
    for r in rows:
        by_panel[r["panel"]] = by_panel.get(r["panel"], 0) + 1
    doc = {"schema": RESCORE_SCHEMA,
           "inputs": {"cells": str(cells), "panels": list(panels), "model": str(model), "model_sha256": model_sha256,
                      "engine": engine, "max": max_n, "min": min_n, "search": searches[0] if searches else None,
                      "fields": list(RESCORE_FIELDS), "rows": str(rows_p)},
           "n_sampled": len(sampled(Path(cells), panels)), "n_rescored": len(rows), "by_panel": by_panel,
           "rescored": [{"cell": r["cell"], "key": r["key"]} for r in rows],
           "cpu_s": round(sum(float(r["cpu_s"] or 0.0) for r in rows), 3), "mismatches": mism,
           "verdict": FAIL if why else PASS, "reason": "; ".join(why) or None}
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return doc


# ------------------------------------------------------------------------------------------------ the bank tooth
def bank_sampled(key: str) -> bool:
    """The bank tooth's hashed sample of a bank's selected states: ``int(sha256(key)[:8], 16) % 64 == 0``."""
    import hashlib

    return int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16) % BANK_MOD == 0


def bank_compare(bank: Mapping[str, Any], got: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every non-timing field of the bank's ``search`` block the re-run does not reproduce exactly."""
    if "error" in got:
        return [{"field": "root_guard", "bank": None, "here": got["error"]}]
    here = got["search"]
    return [{"field": f, "bank": bank.get(f), "here": here.get(f)} for f in sorted((set(bank) | set(here)) - VOLATILE)
            if bank.get(f) != here.get(f)]


def bank_tooth(selected: Path, scored: Path, *, out: Path, search: str, model: str, model_sha256: str, engine: str,
               max_n: int = BANK_MAX, scorer: Any = None) -> Dict[str, Any]:
    """The bank tooth's document (module docstring); ``scorer``: a test's stand-in for ``sign_gate.GateScorer``."""
    import hashlib

    from utils.hashing import sha256_file

    if not isinstance(max_n, int) or isinstance(max_n, bool) or max_n < 1:
        raise IdentityInputError(f"--max {max_n!r}: an int >= 1")
    for p in (selected, scored):
        if not Path(p).is_file():
            raise IdentityInputError(f"{p}: missing")
    sel = _by_key(_jsonl(Path(selected)), "selected")
    sc = _by_key(_jsonl(Path(scored)), "scored")
    sample = sorted((k for k in sel if bank_sampled(k)), key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    take = [k for k in sample if k in sc][:max_n]
    rows_p = Path(str(out) + ".rows.jsonl")
    rows_p.parent.mkdir(parents=True, exist_ok=True)
    done = {r["key"]: r for r in (_jsonl(rows_p) if rows_p.is_file() else [])}
    try:
        for k in take:
            if k in done:
                continue
            if scorer is None:
                scorer = _scorer_for(model, model_sha256, engine, search)
            got = rescore_one(scorer, sel[k])
            row = {"key": k, "panel": sc[k].get("panel"), "mismatches": bank_compare(sc[k].get("search") or {}, got),
                   "cpu_s": (got.get("search") or {}).get("cpu_s")}
            with open(rows_p, "a") as f:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            done[k] = row
            print(f"[live_identity] bank tooth {len(done)} / {len(take)}", flush=True)       # counts only
    finally:
        if scorer is not None and hasattr(scorer, "close"):
            scorer.close()
    rows = [done[k] for k in take]
    mism = [dict(m, key=r["key"]) for r in rows for m in r["mismatches"]]
    by_panel: Dict[str, int] = {}
    for r in rows:
        by_panel[str(r["panel"])] = by_panel.get(str(r["panel"]), 0) + 1
    why = []
    if not rows:
        why.append("no sampled state the bank scored")
    if mism:
        why.append(f"{len(mism)} mismatch(es) (first {mism[0]['key']} {mism[0]['field']})")
    doc = {"schema": BANK_SCHEMA,
           "inputs": {"selected": {"path": str(selected), "sha256": sha256_file(Path(selected))},
                      "scored": {"path": str(scored), "sha256": sha256_file(Path(scored))}, "search": search,
                      "model": str(model), "model_sha256": model_sha256, "engine": engine, "max": max_n,
                      "mod": BANK_MOD, "volatile": sorted(VOLATILE), "rows": str(rows_p)},
           "n_sampled": len(sample), "n_unscored_skipped": sum(1 for k in sample if k not in sc),
           "n_compared": len(rows), "by_panel": by_panel, "compared": [r["key"] for r in rows],
           "cpu_s": round(sum(float(r["cpu_s"] or 0.0) for r in rows), 3), "mismatches": mism,
           "verdict": FAIL if why else PASS, "reason": "; ".join(why) or None}
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return doc


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.value_search.live_identity", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    c = sub.add_parser("check", help="match the live searched records to the banked offline rows and compare them")
    c.add_argument("--cells", required=True, help="the searched label's cells dir")
    c.add_argument("--selected", required=True, help="the offline read's selected.jsonl")
    c.add_argument("--scored", required=True, help="the offline read's scored.jsonl")
    c.add_argument("--min-matched", type=int, required=True, help="PASS needs at least this many matched states")
    c.add_argument("--out", default=None, help="write the document here (JSON)")
    r = sub.add_parser("rescore", help="re-run S′ offline on the hashed sample of live records and compare")
    r.add_argument("--cells", required=True, help="the searched label's cells dir")
    r.add_argument("--panels", required=True, help="the panels, comma-separated, in the registered order")
    r.add_argument("--model", required=True, help="the searched label's checkpoint zip")
    r.add_argument("--model-sha256", required=True, help="its registered sha256")
    r.add_argument("--engine", required=True, help="the registered engine stamp (this checkout must build it)")
    r.add_argument("--out", required=True, help="the document (JSON); <out>.rows.jsonl holds the re-scored states")
    r.add_argument("--max", type=int, default=RESCORE_MAX, help="re-score the first N of the registered order")
    r.add_argument("--min", type=int, default=RESCORE_MIN, help="PASS needs at least this many re-scored")
    t = sub.add_parser("bank-tooth", help="re-run S′ on a hashed sample of a bank's states and compare its rows")
    t.add_argument("--selected", required=True, help="the bank's selected.jsonl")
    t.add_argument("--scored", required=True, help="the bank's scored.jsonl")
    t.add_argument("--search", required=True, help="the bank's registered search (e.g. r0s4_rolls@10)")
    t.add_argument("--model", required=True, help="the checkpoint zip the bank scored with")
    t.add_argument("--model-sha256", required=True, help="its registered sha256")
    t.add_argument("--engine", required=True, help="the registered engine stamp (this checkout must build it)")
    t.add_argument("--out", required=True, help="the document (JSON); <out>.rows.jsonl holds the compared states")
    t.add_argument("--max", type=int, default=BANK_MAX, help="compare the first N sampled scored states")
    a = ap.parse_args(argv)
    if a.verb == "bank-tooth":
        try:
            doc = bank_tooth(Path(a.selected), Path(a.scored), out=Path(a.out), search=a.search, model=a.model,
                             model_sha256=a.model_sha256, engine=a.engine, max_n=a.max)
        except IdentityInputError as exc:
            print(json.dumps({"schema": BANK_SCHEMA, "error": str(exc)}))
            return 2
        print(json.dumps({k: doc[k] for k in ("verdict", "reason", "n_sampled", "n_unscored_skipped", "n_compared",
                                              "by_panel", "cpu_s")} | {"mismatches": len(doc["mismatches"])},
                         sort_keys=True))
        return 0 if doc["verdict"] == PASS else 1
    if a.verb == "rescore":
        try:
            doc = rescore(Path(a.cells), [x for x in a.panels.split(",") if x], out=Path(a.out), model=a.model,
                          model_sha256=a.model_sha256, engine=a.engine, max_n=a.max, min_n=a.min)
        except IdentityInputError as exc:
            print(json.dumps({"schema": RESCORE_SCHEMA, "error": str(exc)}))
            return 2
        print(json.dumps({k: doc[k] for k in ("verdict", "reason", "n_sampled", "n_rescored", "by_panel", "cpu_s")}
                         | {"mismatches": len(doc["mismatches"])}, sort_keys=True))
        return 0 if doc["verdict"] == PASS else 1
    try:
        doc = check(Path(a.cells), Path(a.selected), Path(a.scored), min_matched=a.min_matched)
    except IdentityInputError as exc:
        print(json.dumps({"schema": SCHEMA, "error": str(exc)}))
        return 2
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    counts = {k: doc[k] for k in ("n_records", "n_matched", "n_key_only", "n_unbanked", "n_no_identity")}
    print(json.dumps({"verdict": doc["verdict"], "reason": doc["reason"], **counts, "by_panel": doc["by_panel"],
                      "mismatches": len(doc["mismatches"])}, sort_keys=True))
    return 0 if doc["verdict"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
