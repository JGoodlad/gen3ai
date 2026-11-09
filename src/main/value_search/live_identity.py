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
exactly; for a LEAK-HONEST preset (``main.anchors.search_side.HONEST_DICE``: ``r0s4_honest``, ``r0s4_honest_inc``)
also ``search_key`` (I's digest) and the ``honest`` build record (the chain's counts, the fails, each world's attempt /
re-rolls / feeds, the root command, the world teams; for ``r0s4_honest_inc`` its ``inc`` block too: the step, the
digests of the carry it started from and returned, the worlds carried, extended, new and retired), so the re-score
checks that the live builder drew the same worlds from the same information set; a root guard that refuses is a
mismatch too. A decision the SAFETY NET cut (an envelope preset's ``why: "deadline"``: π played, the search ran whole)
is compared as its search returned it (:func:`search_view`), so it re-scores like any searched decision. PASS iff >= ``--min`` states were re-scored and none differs. Resumable: each re-scored state is appended
to ``<out>.rows.jsonl`` and skipped on a restart; ``--jobs N`` re-scores N states at once, each worker process its own
scorer (the rows are the same, in any order).

THE INCREMENTAL BUILDER'S CARRY (``--carry``, ``r0s4_honest_inc`` only; design_sprime_honest.md §9.2 C19, C26, FINDING
S′H-14). Its decision is a function of I through the CARRY of its battle's earlier searched decisions.
``rebuild`` (the default): every re-scored state runs on a searcher whose carry store is EMPTY, so the builder rebuilds
the battle's earlier checkpoints from I alone (``honest_inc``'s cache-miss path) before the decision; the record's
``honest.inc.carry_in`` then checks that the live carry was that same function of I, which the live process reached by
its cache. It costs the battle's world building up to the decision. ``seeded``: the state runs from the carry the live
record kept (``inc_carry_in``), refused as a mismatch (``inc_carry_in``) unless that carry hashes to the record's
``honest.inc.carry_in``: the decision's last step only, at a decision's cost, and it does NOT check the carry itself.
The mode is in the document's ``inputs.carry`` (None for any other preset).

THE BANK TOOTH (``bank-tooth``): is THIS checkout's S′ the one a banked offline read scored? On a hashed sample of the
bank's ``selected.jsonl`` (:func:`bank_sampled`: ``int(sha256(key)[:8], 16) % 64 == 0``), in sha256(key) order,
skipping rows the bank never scored, the first ``--max`` (80) are re-run through the same S′ path as ``rescore`` (the
scorer's root guards, then ``_search``; no referee, no Speed marker; one thread), and the produced ``search`` block
must equal the bank's ``scored.jsonl`` row's exactly on every field but the CPU / wall timings
(:data:`VOLATILE`).

    python -m main.value_search.live_identity check --cells DIR --selected F --scored F --min-matched N [--out F]
    python -m main.value_search.live_identity rescore --cells DIR --panels P[,P..] --model ZIP --model-sha256 H
        --engine STAMP --out F [--max 600] [--min 300] [--carry rebuild|seeded] [--jobs N]
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
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "gen3_live_identity_v1"
RESCORE_SCHEMA = "gen3_live_rescore_v1"
#: the fields a re-score compares (``pi_argmax`` is its input, guarded by the search's own forward check)
RESCORE_FIELDS: Tuple[str, ...] = ("searched", "why", "played", "q", "log_pi", "root_v")
#: what a LEAK-HONEST preset's re-score compares on top (every field is a count or a digest: no timing)
HONEST_FIELDS: Tuple[str, ...] = ("search_key", "honest")
RESCORE_MAX, RESCORE_MIN = 600, 300
#: how an incremental leak-honest record's re-score gets its carry (module docstring); the first is the default
CARRY_MODES: Tuple[str, ...] = ("rebuild", "seeded")
#: the dice of the incremental builder (``main.anchors.search_presets``: ``r0s4_honest_inc``)
INC_DICE = "fresh_inc"
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


def _jsonl(path: Path, *, torn_tail_ok: bool = False) -> List[Dict[str, Any]]:
    """The JSON lines of ``path``; ``torn_tail_ok``: a last line that does not parse AND has no newline (a write the
    process died inside: a killed attempt, :func:`sampled`) is dropped (:func:`torn_tail`); any other bad line refuses."""
    try:
        text = Path(path).read_text()
        lines = [x for x in text.splitlines() if x.strip()]
        if torn_tail_ok and torn_tail(path):
            lines = lines[:-1]
        return [json.loads(x) for x in lines]
    except (OSError, ValueError) as exc:
        raise IdentityInputError(f"{path}: unreadable ({exc})") from None


def torn_tail(path: Path) -> bool:
    """Whether ``path``'s last line is a TORN write: not newline-terminated and not valid JSON (the live searcher writes
    each record whole and newline-terminated, so only a process killed mid-write leaves one)."""
    text = Path(path).read_text()
    if not text or text.endswith("\n"):
        return False
    try:
        json.loads(text.rsplit("\n", 1)[-1])
        return False
    except ValueError:
        return True


def search_view(rec: Mapping[str, Any]) -> Mapping[str, Any]:
    """A live record as its search returned it (``main.anchors.search_side.search_view``; imported lazily)."""
    from main.anchors.search_side import search_view as view

    return view(dict(rec))


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
            view = search_view(rec)                    # a safety-net cut compared as its search returned it
            for f in FIELDS:
                live, off = view.get(f), b["offline"].get(f)
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
        # a killed attempt may end mid-line: a .partial dir, or the last attempt of a unit that never counted
        killed = ".partial." in d.name or not KA.is_done(cells.parent, u)
        if u.teamset not in panels:
            raise IdentityInputError(f"{d.name}: panel {u.teamset!r} is not one of {list(panels)}")
        for rec in _jsonl(d / LOG, torn_tail_ok=killed):       # its torn last line is dropped (registered)
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


def checkout_code() -> Dict[str, Any]:
    """The re-scoring checkout: its commit and whether its tracked tree is dirty (the rule holds a pinned leak-honest
    read's re-score to its pin, ``rules_searched_play.read_identity``)."""
    from utils.paths import repo_root

    r = str(repo_root())
    commit = subprocess.run(["git", "-C", r, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "-C", r, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True, check=True)
    return {"commit": commit.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def _check_inputs(model: str, model_sha256: str, engine: str) -> None:
    """This checkout builds the registered ``engine`` and ``model`` has the registered sha256 (named refusals)."""
    from main.anchors.server import engine_stamp
    from utils.hashing import sha256_file

    here = engine_stamp()
    if here != engine:
        raise IdentityInputError(f"this checkout's engine is {here}, the registered one {engine}")
    got = sha256_file(Path(model)) if Path(model).is_file() else None
    if got != model_sha256:
        raise IdentityInputError(f"{model}: sha256 {str(got)[:12]} is not {model_sha256[:12]}")


def _scorer_for(model: str, model_sha256: str, engine: str, search: str) -> Any:
    """``sign_gate.GateScorer`` on ``model`` (its sha256 checked), at this checkout's engine (checked), one thread."""
    _check_inputs(model, model_sha256, engine)
    one_thread()
    from main.value_search.sign_gate import GateScorer

    return GateScorer(model, search)


class _CarryRefused(RuntimeError):
    """A seeded re-score's kept carry is missing or does not hash to the record's ``honest.inc.carry_in``."""


def _prepare_carry(scorer: Any, rec: Mapping[str, Any], carry: Optional[str]) -> None:
    """Before an incremental record's re-score (``carry`` set and the searcher's sampler the incremental builder's):
    EMPTY the sampler's carry store, so no state borrows a carry an earlier one built (each re-score is a function of
    its record alone, in any order and across a resume); ``seeded``: then put the record's kept carry in it, checked
    against its ``honest.inc.carry_in`` digest. Any other searcher is left as it is."""
    sampler = getattr(getattr(scorer, "s", None), "sampler", None)
    if carry is None or getattr(sampler, "dice", None) != INC_DICE:
        return
    sampler.store.clear()
    if carry != "seeded":
        return
    from main.value_search.honest_inc import digest, unpack

    want = ((rec.get("honest") or {}).get("inc") or {}).get("carry_in")
    kept = rec.get("inc_carry_in")
    if want is None and kept is None:
        return                                     # the battle's first searched decision: built cold, nothing to seed
    if want is None or kept is None:
        raise _CarryRefused(f"the record keeps inc_carry_in {kept is not None} but honest.inc.carry_in {want!r}")
    text = unpack(str(kept))
    if digest(text) != want:
        raise _CarryRefused(f"inc_carry_in hashes to {str(digest(text))[:12]}, honest.inc.carry_in is {str(want)[:12]}")
    sampler.seed(text)


def rescore_one(scorer: Any, rec: Mapping[str, Any], carry: Optional[str] = None) -> Dict[str, Any]:
    """S′ offline on one sampled live record (the dlive scorer's root guards, then the searcher's ``_search``):
    ``{"search": the offline record}`` or ``{"error": the root guard's refusal}`` (``{"error", "carry": True}`` when a
    seeded carry is refused). ``carry``: an incremental record's carry mode (:data:`CARRY_MODES`; None for any other
    preset); ``rebuilt`` reports how many earlier checkpoints the builder rebuilt (process history, never compared)."""
    import numpy as np

    from agents.battle.core_obs import wrap_row

    try:
        _prepare_carry(scorer, rec, carry)
    except _CarryRefused as exc:
        return {"error": str(exc), "carry": True}
    try:
        _truth, d = scorer.root(rec)
        row = np.array(wrap_row(d["frame"]), dtype=np.float32)
        s = scorer.s._search(rec["log"], int(rec["at"]), rec["side"], row, np.asarray(rec["mask"], dtype=np.int8), d,
                             rec["key"], int(rec["pi_argmax"]))
    except Exception as exc:          # a refused root or forward is a MISMATCH, named, never a skip
        return {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    out: Dict[str, Any] = {"search": s}
    sampler = getattr(scorer.s, "sampler", None)
    if carry is not None and getattr(sampler, "dice", None) == INC_DICE:
        out["rebuilt"] = int(getattr(sampler, "last_rebuilt", 0) or 0)
    return out


def _dice(search: Any) -> str:
    from main.anchors.search_presets import SearchSpecError, parse

    try:
        return parse(str(search))[0].dice
    except SearchSpecError as exc:
        raise IdentityInputError(f"the sampled records' search {search!r}: {exc}") from None


def is_honest(search: Any) -> bool:
    """Whether a registered ``<preset>@<turn>`` is a leak-honest preset (its dice one of
    ``main.anchors.search_side.HONEST_DICE``: ``r0s4_honest``, ``r0s4_honest_inc``)."""
    from main.anchors.search_side import HONEST_DICE

    return _dice(search) in HONEST_DICE


def is_incremental(search: Any) -> bool:
    """Whether a registered ``<preset>@<turn>`` builds its worlds incrementally (``r0s4_honest_inc``: a carry)."""
    return _dice(search) == INC_DICE


def compare_fields(search: Any) -> Tuple[str, ...]:
    """The fields a re-score of records of ``search`` compares."""
    return RESCORE_FIELDS + (HONEST_FIELDS if is_honest(search) else ())


def compare(live: Mapping[str, Any], got: Mapping[str, Any], fields: Sequence[str] = RESCORE_FIELDS
            ) -> List[Dict[str, Any]]:
    """The ``fields`` of a re-scored state that differ from the live record (``root_guard`` when it was refused). The
    offline record is compared as the live one was written (a JSON round trip: tuples become lists); the live one as
    its SEARCH returned it (``search_side.search_view``: a safety-net cut's pick is the search's, which re-scores)."""
    if "error" in got:
        return [{"field": "inc_carry_in" if got.get("carry") else "root_guard", "live": None, "offline": got["error"]}]
    live = search_view(live)
    off = json.loads(json.dumps(got["search"], sort_keys=True, default=str))
    return [{"field": f, "live": live.get(f), "offline": off.get(f)} for f in fields if live.get(f) != off.get(f)]


#: a re-score worker process's scorer (``--jobs``; :func:`_worker_init`)
_WORKER: Dict[str, Any] = {}


def _worker_init(model: str, model_sha256: str, engine: str, search: str, factory: Any = None) -> None:
    _WORKER["scorer"] = factory() if factory is not None else _scorer_for(model, model_sha256, engine, search)


def _worker_one(rec: Mapping[str, Any], carry: Optional[str]) -> Dict[str, Any]:
    """One state on this worker's own scorer, as plain JSON (what crosses the process boundary)."""
    return json.loads(json.dumps(rescore_one(_WORKER["scorer"], rec, carry), sort_keys=True, default=str))


def rescore(cells: Path, panels: Sequence[str], *, out: Path, model: str, model_sha256: str, engine: str,
            max_n: int = RESCORE_MAX, min_n: int = RESCORE_MIN, scorer: Any = None, carry: str = CARRY_MODES[0],
            jobs: int = 1, scorer_factory: Any = None, mp_context: str = "spawn") -> Dict[str, Any]:
    """The re-score's document (module docstring); ``scorer``: a test's stand-in for ``sign_gate.GateScorer`` (one
    process); ``jobs`` > 1: that many worker processes, each with its own scorer (``scorer_factory()``, a test's, else
    :func:`_scorer_for`), started by ``mp_context``."""
    for k, v in (("max", max_n), ("min", min_n), ("jobs", jobs)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise IdentityInputError(f"--{k} {v!r}: an int >= 1")
    if carry not in CARRY_MODES:
        raise IdentityInputError(f"--carry {carry!r}: one of {list(CARRY_MODES)}")
    if jobs > 1 and scorer is not None:
        raise IdentityInputError("a scorer stand-in runs in one process (--jobs 1)")
    order = sampled(Path(cells), panels)[:max_n]
    searches = sorted({str(e["rec"].get("search")) for e in order})
    if len(searches) > 1:
        raise IdentityInputError(f"the sampled records name {len(searches)} searches {searches}")
    fields = compare_fields(searches[0]) if searches else RESCORE_FIELDS
    mode = carry if searches and is_incremental(searches[0]) else None
    rows_p = Path(str(out) + ".rows.jsonl")
    rows_p.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["cell"], r["key"]): r for r in (_jsonl(rows_p) if rows_p.is_file() else [])}
    todo = [e for e in order if (e["cell"], e["key"]) not in done]

    def put(e: Mapping[str, Any], got: Mapping[str, Any]) -> None:
        row = {"cell": e["cell"], "key": e["key"], "panel": e["panel"], "mismatches": compare(e["rec"], got, fields),
               "cpu_s": (got.get("search") or {}).get("cpu_s")}
        if "rebuilt" in got:
            row["rebuilt"] = got["rebuilt"]
        with open(rows_p, "a") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        done[(e["cell"], e["key"])] = row
        print(f"[live_identity] rescored {len(done)} / {len(order)}", flush=True)     # counts only

    if todo and jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if scorer_factory is None:
            _check_inputs(model, model_sha256, engine)          # refuse here, named, before any worker starts
        with ProcessPoolExecutor(max_workers=min(jobs, len(todo)), mp_context=mp.get_context(mp_context),
                                 initializer=_worker_init,
                                 initargs=(model, model_sha256, engine, searches[0], scorer_factory)) as ex:
            futs = {ex.submit(_worker_one, e["rec"], mode): e for e in todo}
            for f in as_completed(futs):
                put(futs[f], f.result())
    elif todo:
        if scorer is None:
            scorer = _scorer_for(model, model_sha256, engine, searches[0])
        try:
            for e in todo:
                put(e, rescore_one(scorer, e["rec"], mode))
        finally:
            if hasattr(scorer, "close"):
                scorer.close()
    rows = [done[(e["cell"], e["key"])] for e in order]
    try:
        code: Optional[Dict[str, Any]] = checkout_code()
    except (OSError, subprocess.CalledProcessError):
        code = None                                    # no git checkout: the rule refuses a pinned read's re-score
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
                      "fields": list(fields), "carry": mode, "rows": str(rows_p), "code": code},
           "n_sampled": len(sampled(Path(cells), panels)), "n_rescored": len(rows), "by_panel": by_panel,
           "rescored": [{"cell": r["cell"], "key": r["key"]} for r in rows],
           "cpu_s": round(sum(float(r["cpu_s"] or 0.0) for r in rows), 3), "mismatches": mism,
           "verdict": FAIL if why else PASS, "reason": "; ".join(why) or None}
    if mode is not None:
        doc["rebuilt"] = sum(int(r.get("rebuilt") or 0) for r in rows)
    # a killed attempt's decision log may end inside a record: its torn last line is dropped (registered), and named
    doc["torn_tails"] = [d.name for d in attempt_dirs(Path(cells)) if torn_tail(d / LOG)]
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
    r.add_argument("--carry", choices=CARRY_MODES, default=CARRY_MODES[0],
                   help="r0s4_honest_inc only: rebuild each state's carry from I (default) or seed the kept one")
    r.add_argument("--jobs", type=int, default=1, help="re-score this many states at once (one process each)")
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
                          model_sha256=a.model_sha256, engine=a.engine, max_n=a.max, min_n=a.min, carry=a.carry,
                          jobs=a.jobs)
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
