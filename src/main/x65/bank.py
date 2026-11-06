r"""B1, THE LABEL BANK (PREREG §2): every decision of our side in S′-LIVE's COUNTED games, its observation row rebuilt
from the game's capture, and on each SEARCHED decision S′'s piKL target π′, outcome-blind and guarded bit for bit.

    python -m main.x65.bank build --state ~/gen3ai_archive/sprime_live_play_2026-11-01 --out <dir> [--workers 3]
    python -m main.x65.bank check --out <dir>

**Source.** The counted units of S′-LIVE's ``record.json`` (``inputs.deals``: a deal's ``raw`` / ``searched`` unit
whose status is ``counted``; only those structural fields are kept, never a unit's games), each read from its FINAL
attempt dir ``<state>/anchor/u/<label>/cells/<unit>`` (never a ``.partial.<epoch>`` dir). From a cell the builder
reads ``search_decisions.jsonl`` (searched arm) and the capture's ``START`` document, ``seed`` and ``commands``; the
capture's ``chunks`` (they carry the result) are dropped unread, and :func:`read_json` REFUSES ``games.jsonl``,
``summary.json`` and every other result-bearing file by name (:data:`FORBIDDEN`). The core replays below stop BEFORE
our last command, so no replay reaches the battle's end (a ``|win|`` / ``|tie`` line in one is a refusal).

**The ordered rebuild** (FINDING of the frame probe: the one miss of 976 / 977 was a naive prefix). The live searcher
handed the search ``search_side.honest_log``: every command of the request updates BEFORE ours, ordered (update, p1
before p2, arrival), plus OUR re-choices of our own update (after an ``[Unavailable choice]``), never the opponent's
choice of our update. The capture keeps ARRIVAL order without the update index, so the builder re-derives it the way
the front end counted it (``ws_frontend._note_side_line``): it replays the arrival-order prefix through the core once
(:func:`replay_prefix`), counts each side's ``|request|`` lines (a re-request after ``[Unavailable choice]`` asks the
SAME update), and maps each side's j-th command to its j-th decision (the tracker's ``choice`` must be the command's
token, its ``after`` chunk a request of that side). Our k-th decision's log is then :func:`honest_cut`, exactly
``honest_log``'s filter and sort, and its row is ``worlds.prepare_root(log, at, side, key)``'s ``ref[-1]``.

**The identity guard (a GIGO check).** A searched decision is a LABEL ROW only if its rebuilt row hashes to the
recorded ``frame_sha256``; a mismatch is DROPPED and counted, never kept. More than :data:`MAX_DROP_FRAC` of a
panel's label rows dropped VOIDs the bank (``manifest.void``; the trainer refuses a void bank). An excluded decision
(``searched: false`` past turn 10) also carries ``frame_sha256``: it is the PATH CHECK of the anchor rows (a mismatch
drops that row). The rebuilt ``at``, ``update`` and ``log_sha256`` (and the whole ``log`` on the 1-in-16 re-score
sample) are compared with the record and reported.

**Anchor rows** (every decision of our side, both arms; their teacher is P\* online, in the trainer). A raw game keeps
no ``frame_sha256``, so a row with no recorded digest must pass two other checks or it is dropped and counted: the
honest-cut row equals the arrival-order replay's row (two orderings, one state), and P\*'s greedy action on the row
(``qhat.logits_and_values``, the live searcher's forward) is the action the game recorded (raw: the command played,
the raw arm being greedy; searched: ``pi_argmax``). A row whose two best legal logits lie within :data:`TIE_EPS` is
excluded from the argmax check (root rule 8) and counted, never passed by chance.

**Keys.** ``deal`` = ``search_side.deal_key`` of the START (the raw and searched game of a deal share it; the
searched records' ``deal`` must equal it); INNER VALIDATION (PREREG §3) = ``sha256(deal)[0] % 10 == 0``, a function of
the deal alone, so a deal's rows of both arms fall on one side. The bank is fold-agnostic: the trainer (B2) selects
its panels. ``turn`` is the last ``|turn|`` our side saw before the decision's request (checked against every searched
record's ``turn``), ``n`` our decision index.

**The manifest** (``bank_manifest.json``; its sha256 in ``bank_manifest.sha256``) lists every shard's sha256 (one
``.npz`` per arm, panel and unit seed), the counts and drops per panel and reason, the engine (must be E4), the
commit and the source's ``record.json`` sha256. ``build`` is resumable: a shard whose npz and json exist is kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SHARD_SCHEMA = "gen3_x65_bank_shard_v1"
MANIFEST_SCHEMA = "gen3_x65_bank_manifest_v1"
LAMBDA = 0.015
FROM_TURN = 10
U_PANELS = ("heldout72_hpfix", "forum72", "p72")
LABELS = {"raw": "pstar", "searched": "pstar_s"}
ARM_CODE = {"pstar": 0, "pstar_s": 1}
GROUP = "u"
MAX_DROP_FRAC = 0.005
INNER_VAL_MOD = 10
E4 = "port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129"
N_ACT = 11
TIE_EPS = 1e-4
KEY_SCHEMA = "su_live_v1"
#: result-bearing files the builder never opens (outcome-blind, PREREG §2)
FORBIDDEN = frozenset({"games.jsonl", "summary.json", "forced_ends.jsonl", "peer_acceptor_report.json",
                       "peer_acceptor.log", "ws_frontend.log", "decision.json", "cells.json", "inputs.json"})
_REQ = "|request|"
_INVALID = "|error|[Invalid choice]"
_UNAVAILABLE = "|error|[Unavailable choice]"
#: the per-row arrays of a shard (``obs`` the row, ``mask`` the legal actions, ``pi_prime`` S′'s target on a label row)
FIELDS = ("obs", "mask", "label", "pi_prime", "q", "log_pi", "played", "pi_argmax", "deviation", "arm", "game",
          "unit_seed", "deal", "turn", "n", "update", "at", "inner_val", "leak_attr", "leak_rewr", "leak_resid",
          "leak_json", "key", "frame_checked", "pstar_check")


class BankError(RuntimeError):
    """A game, row or bank that does not match what it claims (a defect for that game: it is dropped and counted)."""


class OutcomeBlindError(BankError):
    """An attempt to read a result-bearing file (PREREG §2: the builder is outcome-blind)."""


# ------------------------------------------------------------------------------------------------ reading
def guard_path(path: Path) -> Path:
    p = Path(path)
    if p.name in FORBIDDEN or p.name.startswith("peer_acceptor"):
        raise OutcomeBlindError(f"{p}: the label bank is outcome-blind and never reads {p.name}")
    return p


def read_json(path: Path) -> Any:
    with open(guard_path(path)) as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(guard_path(path)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def counted_units(record: Path) -> list[dict[str, Any]]:
    """The counted units of S′-LIVE's record: ``{arm label, unit, panel, unit_seed, game}`` (only these fields)."""
    doc = read_json(record)
    if doc.get("rule") != "searched_play" or doc.get("void") is not None:
        raise BankError(f"{record}: not a valid searched_play record")
    ins = doc["inputs"]
    if ins.get("raw") != LABELS["raw"] or ins.get("searched") != LABELS["searched"]:
        raise BankError(f"{record}: arms {ins.get('raw')!r} / {ins.get('searched')!r}")
    out = []
    for d in ins["deals"]:
        for arm in ("raw", "searched"):
            if d[arm]["status"] == "counted":
                out.append({"label": LABELS[arm], "unit": str(d[arm]["unit"]), "panel": str(d["panel"]),
                            "unit_seed": int(d["unit_seed"]), "game": int(d["game"])})
    del doc
    if any(u["panel"] not in U_PANELS for u in out):
        raise BankError("a counted unit outside U")
    return out


def read_capture(cell: Path) -> tuple[dict[str, Any], list[str]]:
    """``(START document, CHOOSE commands in arrival order)`` of a cell's ONE capture; the chunks are never kept."""
    caps = sorted((cell / "capture").glob("*.json"))
    if len(caps) != 1:
        raise BankError(f"{cell}: {len(caps)} captures")
    cap = read_json(caps[0])
    commands = list(cap["commands"])
    seed = list(cap["seed"])
    del cap
    if not commands or not commands[0].startswith("START "):
        raise BankError(f"{cell}: the capture opens with no START")
    start = json.loads(commands[0][len("START "):])
    if list(start.get("seed") or []) != seed:
        raise BankError(f"{cell}: the START seed is not the capture's")
    cmds: list[str] = []
    for c in commands[1:]:
        if not c.startswith("CHOOSE "):
            break                           # FORCELOSE / END: the battle is over, no decision follows
        cmds.append(c)
    return start, cmds


def our_side(start: dict[str, Any]) -> str:
    sides = list((start.get("core_obs") or {}).get("sides") or [])
    if len(sides) != 1 or sides[0] not in ("p1", "p2"):
        raise BankError(f"the START names core sides {sides!r}: our side is the one in-process core slot")
    return str(sides[0])


def inner_val(deal: str) -> bool:
    """PREREG §3: the deal's sha256 has a first byte ≡ 0 (mod 10)."""
    return hashlib.sha256(deal.encode()).digest()[0] % INNER_VAL_MOD == 0


def pi_prime(q: dict[str, float], log_pi: dict[str, float], lam: float = LAMBDA) -> np.ndarray:
    """``[11]`` float64: softmax over the legal actions (the keys of ``q``) of log π + Q̂ / λ; 0 elsewhere."""
    if sorted(q) != sorted(log_pi) or not q:
        raise BankError("q and log_pi name different legal actions")
    a = np.array(sorted(int(k) for k in q))
    s = np.array([float(log_pi[str(i)]) + float(q[str(i)]) / lam for i in a], np.float64)
    s = np.exp(s - s.max())
    out = np.zeros(N_ACT, np.float64)
    out[a] = s / s.sum()
    return out


# ------------------------------------------------------------------------------------------------ the replay
def _token(cmd: str) -> tuple[str, str]:
    _, side, tok = cmd.split(" ", 2)
    return side, tok


def _log(start: dict[str, Any], cmds: Sequence[str]) -> dict[str, Any]:
    from main.anchors.search_side import _seed_text

    return {"format_id": str(start["formatid"]), "seed": _seed_text(start.get("seed")),
            "names": [str(start["p1"]["name"]), str(start["p2"]["name"])],
            "teams": [str(start["p1"]["team"]), str(start["p2"]["team"])], "cmds": list(cmds)}


def replay_prefix(start: dict[str, Any], cmds: Sequence[str]) -> dict[str, Any]:
    """The arrival-order prefix through the core (trackers and rows on). Never reaches the battle's end."""
    from agents.battle.core_replay import RecordedBattle, run_core
    from main.value_search.worlds import log_commands

    log = _log(start, cmds)
    rb = RecordedBattle(label="x65_bank", format_id=log["format_id"], seed=log["seed"],
                        p1={"name": log["names"][0], "team": log["teams"][0]},
                        p2={"name": log["names"][1], "team": log["teams"][1]}, commands=log_commands(log))
    res = run_core([rb], trackers=True, obs=True)[0]
    if not res.get("ok"):
        raise BankError(f"the arrival-order prefix does not replay: {res.get('error')}")
    if res.get("ended"):
        raise BankError("the prefix replay reached the battle's end (outcome-blind: refused)")
    res.pop("winner", None)
    for _s, lines in res["chunks"]:
        for ln in lines:
            if ln == "|tie" or ln.startswith(("|win|", "|tie|")):
                raise BankError("a result line in a prefix replay (outcome-blind: refused)")
    return res


def request_updates(res: dict[str, Any]) -> list[dict[int, int]]:
    """Per side (0 = p1, 1 = p2): ``{chunk index: update}`` of every chunk carrying a non-wait ``|request|``, the
    update counted as the front end counts ``req_n`` (every request, a wait included; a re-request after an
    ``[Unavailable choice]`` asks the same update). An ``[Invalid choice]`` (the front end drops the command from its
    log) is refused: the bank cannot place it."""
    out: list[dict[int, int]] = [{}, {}]
    count = [0, 0]
    err = [False, False]
    for i, (s, lines) in enumerate(res["chunks"]):
        for ln in lines:
            if ln.startswith(_INVALID):
                raise BankError(f"an [Invalid choice] for side {s + 1} at chunk {i}: not placeable")
            if ln.startswith(_UNAVAILABLE):
                err[s] = True
            if ln.startswith(_REQ) and len(ln) > len(_REQ):
                if err[s]:
                    err[s] = False
                else:
                    count[s] += 1
                if not json.loads(ln[len(_REQ):]).get("wait"):
                    out[s][i] = count[s]
    return out


def decision_updates(res: dict[str, Any], cmds: Sequence[str]) -> tuple[list[int], list[list[int]]]:
    """``(update of every command in cmds, per side the update of each decision of that side in the replay)``. A
    side's j-th command is its j-th decision: the tracker's ``choice`` must be the command's token and its ``after``
    chunk a request of that side."""
    req = request_updates(res)
    per_side: list[list[int]] = [[], []]
    for s in (0, 1):
        for e in res["trackers"][s]:
            if "obs" not in e:
                continue
            a = int(e["after"])
            if a not in req[s]:
                raise BankError(f"side {s + 1}'s decision after chunk {a} answers no request of that side")
            per_side[s].append(req[s][a])
    seen = [0, 0]
    upd: list[int] = []
    for c in cmds:
        side, tok = _token(c)
        s = 0 if side == "p1" else 1
        j = seen[s]
        ents = [e for e in res["trackers"][s] if "obs" in e]
        if j >= len(ents) or str(ents[j].get("choice")) != tok:
            raise BankError(f"{side}'s command {j} {tok!r} is not its replayed decision "
                            f"{ents[j].get('choice') if j < len(ents) else None!r}")
        upd.append(per_side[s][j])
        seen[s] += 1
    return upd, per_side


def honest_cut(start: dict[str, Any], cmds: Sequence[str], upd: Sequence[int], side: str, p: int,
               u: int) -> dict[str, Any]:
    """``search_side.honest_log`` for ``side``'s decision whose command is ``cmds[p]`` and whose request update is
    ``u``: of the commands written before it, every one of an earlier update and ``side``'s own of update ``u``,
    ordered (update, p1 before p2, arrival); a peer's switch re-spelled the core's way."""
    from main.anchors.search_side import core_command

    rows = [(i, int(upd[i]), _token(cmds[i])[0], cmds[i]) for i in range(p)
            if upd[i] < u or (upd[i] == u and _token(cmds[i])[0] == side)]
    rows.sort(key=lambda r: (r[1], 0 if r[2] == "p1" else 1, r[0]))
    teams = {"p1": str(start["p1"]["team"]), "p2": str(start["p2"]["team"])}
    return _log(start, [core_command(c, teams) for _i, _u, _s, c in rows])


def naive_cut(start: dict[str, Any], cmds: Sequence[str], at: int) -> dict[str, Any]:
    """The frame probe's NAIVE prefix ``cmds[:at]`` (what B1 replaces; kept for the regression test)."""
    return _log(start, cmds[:at])


def _turn_before(res: dict[str, Any], viewer: int, after: int) -> int:
    t = 0
    for i, (s, lines) in enumerate(res["chunks"][: after + 1]):
        if s != viewer:
            continue
        for ln in lines:
            if ln.startswith("|turn|"):
                t = int(ln[len("|turn|"):])
    return t


def rebuild_row(log: dict[str, Any], side: str, k: int, key: str) -> dict[str, Any]:
    """Our decision ``k``'s row, mask, tokens and turn from ``prepare_root`` on its honest log."""
    from main.value_search import worlds as W

    at = len(log["cmds"])
    truth = W.prepare_root(log, at, side, key, root_choice_in_log=False)
    if truth.n != k or len(truth.ref) != k + 1:
        raise BankError(f"{key}: the honest log opens decision {truth.n}, not {k}")
    v = 0 if side == "p1" else 1
    ent = [e for e in truth.true["trackers"][v] if "obs" in e][k]
    return {"b64": truth.ref[-1], "mask": [int(x) for x in ent["mask"]],
            "tokens": {int(a): str(t) for a, t in (ent.get("tokens") or {}).items()},
            "turn": _turn_before(truth.true, v, int(ent["after"])), "at": at}


# ------------------------------------------------------------------------------------------------ one game
def game_rows(cell: Path, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every decision of our side in one counted game, rebuilt and guarded (module docstring). Returns
    ``(rows, stats)``; a row is kept iff its guards pass. Raises :class:`BankError` for a game that cannot be
    placed at all."""
    from main.anchors.forced_branch import log_sha256
    from main.anchors.input_log import frame_sha256
    from main.anchors.search_side import deal_key

    start, cmds = read_capture(cell)
    side = our_side(start)
    ours = [i for i, c in enumerate(cmds) if _token(c)[0] == side]
    if not ours:
        raise BankError(f"{cell}: no command of ours")
    deal = deal_key(start, side)
    res = replay_prefix(start, cmds[: ours[-1]])
    upd, per_side = decision_updates(res, cmds[: ours[-1]])
    v = 0 if side == "p1" else 1
    arrival = [e for e in res["trackers"][v] if "obs" in e]
    if len(arrival) != len(ours) or len(per_side[v]) != len(ours):
        raise BankError(f"{cell}: {len(arrival)} replayed decisions of ours for {len(ours)} commands")
    recs: list[dict[str, Any]] | None = None
    if label == LABELS["searched"]:
        recs = read_jsonl(cell / "search_decisions.jsonl")
        if [int(r["n"]) for r in recs] != list(range(len(ours))) or any(r["side"] != side for r in recs):
            raise BankError(f"{cell}: the decision records are not our decisions 0..{len(ours) - 1}")
    st: dict[str, Any] = {"decisions": len(ours), "labels_recorded": 0, "labels_kept": 0, "drop": {},
                          "frame_checked": 0, "frame_ok": 0, "xcheck_bad": 0, "log_sha_bad": 0, "at_bad": 0,
                          "rescore_log_bad": 0, "mode_bad": 0, "turn_bad": 0, "rows": 0}

    def drop(reason: str, is_label: bool) -> None:
        st["drop"][reason] = st["drop"].get(reason, 0) + 1
        if is_label:
            st["drop"]["label:" + reason] = st["drop"].get("label:" + reason, 0) + 1

    rows: list[dict[str, Any]] = []
    for k, p in enumerate(ours):
        u = per_side[v][k]                      # the update our k-th command answers (the last: our last request)
        log = honest_cut(start, cmds, upd, side, p, u)
        rec = recs[k] if recs is not None else None
        is_label = bool(rec and rec.get("searched"))
        st["labels_recorded"] += int(is_label)
        try:
            key = f"{KEY_SCHEMA}|{deal}|t?|n{k}"
            r = rebuild_row(log, side, k, key)
        except Exception as exc:                      # noqa: BLE001 - a row the core cannot rebuild is dropped
            drop(f"rebuild:{type(exc).__name__}", is_label)
            continue
        key = f"{KEY_SCHEMA}|{deal}|t{r['turn']}|n{k}"
        _side, tok = _token(cmds[p])
        hits = [a for a, t in r["tokens"].items() if t == tok]
        if len(hits) != 1:
            drop("played_token", is_label)
            continue
        played = hits[0]
        row: dict[str, Any] = {"b64": r["b64"], "mask": r["mask"], "label": False, "played": played,
                               "pi_argmax": -1, "deviation": False, "turn": r["turn"], "n": k, "update": u,
                               "at": r["at"], "key": key, "deal": deal, "leak": None, "q": None, "log_pi": None,
                               "frame_checked": False}
        xok = r["b64"] == arrival[k]["obs"]["b64"]
        st["xcheck_bad"] += int(not xok)
        if rec is not None:
            if int(rec["turn"]) != r["turn"]:
                st["turn_bad"] += 1
                drop("turn", is_label)
                continue
            if int(rec["played"]) != played:
                drop("played", is_label)
                continue
            row["pi_argmax"] = int(rec["pi_argmax"])
            if rec.get("frame_sha256"):
                st["frame_checked"] += 1
                row["frame_checked"] = True
                if frame_sha256(r["b64"]) != rec["frame_sha256"]:
                    drop("frame_sha256", is_label)                # THE identity guard: never kept
                    continue
                st["frame_ok"] += 1
                if rec.get("key") != key or rec.get("deal") != deal:
                    drop("key", is_label)
                    continue
                st["at_bad"] += int(int(rec["at"]) != r["at"] or int(rec["update"]) != u)
                st["log_sha_bad"] += int(rec.get("log_sha256") != log_sha256(log))
                if rec.get("rescore_sample"):
                    st["rescore_log_bad"] += int(rec.get("log") != log)
            elif not xok:
                drop("xcheck", False)
                continue
            if is_label:
                if not row["frame_checked"]:
                    drop("no_frame_sha256", True)               # a label row is never unverified
                    continue
                legal = sorted(int(a) for a in rec["q"])
                if legal != [i for i, m in enumerate(r["mask"]) if m]:
                    drop("legal_mask", True)
                    continue
                pp = pi_prime(rec["q"], rec["log_pi"])
                st["mode_bad"] += int(int(np.argmax(pp)) != int(rec["played"]))
                leak = rec.get("leak") or {}
                row.update(label=True, pi_prime=pp, q=rec["q"], log_pi=rec["log_pi"], leak=leak,
                           deviation=bool(rec.get("deviation")))
                st["labels_kept"] += 1
        elif not xok:
            drop("xcheck", False)
            continue
        rows.append(row)
    st["rows"] = len(rows)
    return rows, st


def pstar_check(rows: list[dict[str, Any]], logits: np.ndarray) -> list[int]:
    """Per row: 1 = P\\*'s greedy action is the recorded one (raw: played; searched: ``pi_argmax``), 2 = the two best
    legal logits lie within :data:`TIE_EPS` (excluded, root rule 8), 0 = a mismatch (the row is dropped)."""
    out = []
    for r, lg in zip(rows, logits):
        legal = [i for i, m in enumerate(r["mask"]) if m]
        want = r["played"] if r["arm"] == 0 else r["pi_argmax"]
        if len(legal) < 2:
            out.append(1 if want == legal[0] else 0)
            continue
        v = np.array([lg[i] for i in legal], np.float64)
        top = np.sort(v)[::-1]
        if top[0] - top[1] <= TIE_EPS:
            out.append(2)
            continue
        out.append(1 if legal[int(np.argmax(v))] == want else 0)
    return out


# ------------------------------------------------------------------------------------------------ shards
def _arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    import base64

    n = len(rows)
    a: dict[str, Any] = {
        "obs": np.stack([np.frombuffer(base64.b64decode(r["b64"]), dtype="<f4") for r in rows]).astype(np.float32)
        if n else np.zeros((0, 0), np.float32),
        "mask": np.array([r["mask"] for r in rows], np.int8).reshape(n, N_ACT),
        "label": np.array([r["label"] for r in rows], bool),
        "pi_prime": np.array([r.get("pi_prime") if r["label"] else np.zeros(N_ACT) for r in rows],
                             np.float64).reshape(n, N_ACT),
        "q": np.full((n, N_ACT), np.nan), "log_pi": np.full((n, N_ACT), np.nan),
        "played": np.array([r["played"] for r in rows], np.int8),
        "pi_argmax": np.array([r["pi_argmax"] for r in rows], np.int8),
        "deviation": np.array([r["deviation"] for r in rows], bool),
        "arm": np.array([r["arm"] for r in rows], np.int8), "game": np.array([r["game"] for r in rows], np.int16),
        "unit_seed": np.array([r["unit_seed"] for r in rows], np.int16),
        "deal": np.array([r["deal"] for r in rows], "U20"),
        "turn": np.array([r["turn"] for r in rows], np.int16), "n": np.array([r["n"] for r in rows], np.int16),
        "update": np.array([r["update"] for r in rows], np.int16), "at": np.array([r["at"] for r in rows], np.int16),
        "inner_val": np.array([inner_val(r["deal"]) for r in rows], bool),
        "leak_attr": np.array([int((r["leak"] or {}).get("attribution_miss", 0)) for r in rows], np.int16),
        "leak_rewr": np.array([int((r["leak"] or {}).get("rewritten_moves", 0)) for r in rows], np.int16),
        "leak_resid": np.array([any("counters" in str(x) for x in ((r["leak"] or {}).get("residual") or []))
                                for r in rows], bool),
        "leak_json": np.array([json.dumps(r["leak"], sort_keys=True) if r["leak"] is not None else "" for r in rows],
                              "U400"),
        "key": np.array([r["key"] for r in rows], "U64"),
        "frame_checked": np.array([r["frame_checked"] for r in rows], bool),
        "pstar_check": np.array([r["pstar_check"] for r in rows], np.int8),
    }
    for i, r in enumerate(rows):
        if r["label"]:
            for k, x in r["q"].items():
                a["q"][i, int(k)] = float(x)
            for k, x in r["log_pi"].items():
                a["log_pi"][i, int(k)] = float(x)
    return a


def shard_name(label: str, panel: str, seed: int) -> str:
    return f"{label}__{panel}__{seed}"


def build_shard(state: Path, out: Path, label: str, panel: str, seed: int, units: list[dict[str, Any]],
                model_zip: Path | None, threads: int = 1) -> dict[str, Any]:
    """One shard (an arm, a panel, a unit seed): every counted game's rows, the P\\* check, written atomically."""
    name = shard_name(label, panel, seed)
    js, npz = out / f"{name}.json", out / f"{name}.npz"
    if js.exists() and npz.exists():
        rec = json.loads(js.read_text())
        if rec.get("sha256") == sha256_file(npz):
            return rec
    rows: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    for u in sorted(units, key=lambda x: x["game"]):
        cell = state / "anchor" / GROUP / label / "cells" / u["unit"]
        try:
            g_rows, g_st = game_rows(cell, label)
        except BankError as exc:
            n_lab = 0
            if label == LABELS["searched"] and (cell / "search_decisions.jsonl").exists():
                n_lab = sum(1 for r in read_jsonl(cell / "search_decisions.jsonl") if r.get("searched"))
            games.append({"unit": u["unit"], "game": u["game"], "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                          "labels_recorded": n_lab, "labels_kept": 0, "rows": 0,
                          "drop": {"game": 1, "label:game": n_lab}})
            continue
        for r in g_rows:
            r.update(arm=ARM_CODE[label], game=u["game"], unit_seed=seed)
        rows.extend(g_rows)
        games.append({"unit": u["unit"], "game": u["game"], **g_st})
    if rows and model_zip is not None:
        logits = pstar_logits(model_zip, rows, threads)
        checks = pstar_check(rows, logits)
    else:
        checks = [1] * len(rows)
    kept = []
    n_mis = {"label": 0, "anchor": 0}
    for r, c in zip(rows, checks):
        r["pstar_check"] = c
        if c == 0:
            n_mis["label" if r["label"] else "anchor"] += 1
            continue
        kept.append(r)
    arr = _arrays(kept)
    tmp = out / f"{name}.tmp.npz"
    np.savez_compressed(tmp, **arr)  # type: ignore[arg-type]
    os.replace(tmp, npz)
    rec = {"schema": SHARD_SCHEMA, "name": name, "label": label, "panel": panel, "unit_seed": seed,
           "rows": len(kept), "labels": int(arr["label"].sum()), "sha256": sha256_file(npz), "games": games,
           "pstar_mismatch": n_mis, "pstar_ties": int(sum(1 for c in checks if c == 2)),
           "pstar_checked": model_zip is not None}
    tj = out / f"{name}.tmp.json"
    tj.write_text(json.dumps(rec, indent=1))
    os.replace(tj, js)
    return rec


def pstar_logits(model_zip: Path, rows: list[dict[str, Any]], threads: int) -> np.ndarray:
    import base64

    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    x = np.stack([np.frombuffer(base64.b64decode(r["b64"]), dtype="<f4") for r in rows]).astype(np.float32)
    m = np.array([r["mask"] for r in rows], np.uint8)
    with inference_globals(threads):
        model = _MODEL_CACHE.get(str(model_zip))
        if model is None:
            model = load_checkpoint(model_zip)
            _MODEL_CACHE[str(model_zip)] = model
        out = logits_and_values(model, x, m, batch=512)
    return out[:, :N_ACT]


_MODEL_CACHE: dict[str, Any] = {}


def _work(args: tuple[str, str, str, str, int, list[dict[str, Any]], str | None]) -> dict[str, Any]:
    state, out, label, panel, seed, units, model = args
    return build_shard(Path(state), Path(out), label, panel, seed, units, Path(model) if model else None)


def summarize(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Per panel: label rows recorded / kept / dropped (by reason), the drop share and the VOID test; anchor rows."""
    per: dict[str, dict[str, Any]] = {}
    for s in shards:
        p = per.setdefault(s["panel"], {"labels_recorded": 0, "labels_kept": 0, "label_drops": {}, "rows": {},
                                        "games": {}, "game_errors": 0, "frame_checked": 0, "frame_ok": 0,
                                        "xcheck_bad": 0, "log_sha_bad": 0, "at_bad": 0, "rescore_log_bad": 0,
                                        "mode_bad": 0, "turn_bad": 0, "drops": {}, "pstar_mismatch": 0,
                                        "pstar_ties": 0})
        p["rows"][s["label"]] = p["rows"].get(s["label"], 0) + s["rows"]
        p["games"][s["label"]] = p["games"].get(s["label"], 0) + len(s["games"])
        p["pstar_mismatch"] += sum(s["pstar_mismatch"].values())
        p["pstar_ties"] += s["pstar_ties"]
        for g in s["games"]:
            p["game_errors"] += int("error" in g)
            p["labels_recorded"] += g.get("labels_recorded", 0)
            for k in ("frame_checked", "frame_ok", "xcheck_bad", "log_sha_bad", "at_bad", "rescore_log_bad",
                      "mode_bad", "turn_bad"):
                p[k] += g.get(k, 0)
            for k, v in g.get("drop", {}).items():
                p["drops"][k] = p["drops"].get(k, 0) + v
                if k.startswith("label:"):
                    p["label_drops"][k[6:]] = p["label_drops"].get(k[6:], 0) + v
        p["labels_kept"] += s["labels"]
    void = []
    for name, p in per.items():
        p["label_drops"]["pstar_check"] = sum(sh["pstar_mismatch"]["label"] for sh in shards if sh["panel"] == name)
        lost = p["labels_recorded"] - p["labels_kept"]
        p["label_drop_frac"] = lost / max(p["labels_recorded"], 1)
        if p["label_drop_frac"] > MAX_DROP_FRAC:
            void.append(f"{name}: {lost} of {p['labels_recorded']} label rows dropped "
                        f"({p['label_drop_frac']:.4%} > {MAX_DROP_FRAC:.1%})")
    return {"per_panel": per, "void": void or None}


def build(state: Path, out: Path, workers: int = 3, model: Path | None = None,
          allow_dirty: bool = False) -> dict[str, Any]:
    """The whole bank (module docstring); returns the manifest."""
    from main.anchors.server import engine_stamp
    from utils.git import get_git_hash

    eng = engine_stamp()
    if eng != E4:
        raise BankError(f"this checkout's engine {eng!r} is not E4 {E4!r}: the rows would not be S′-LIVE's")
    commit = get_git_hash()
    dirty = _dirty()
    if dirty and not allow_dirty:
        raise BankError("the checkout is dirty: a bank names the commit that built it")
    out.mkdir(parents=True, exist_ok=True)
    record = state / "record.json"
    units = counted_units(record)
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for u in units:
        groups.setdefault((u["label"], u["panel"], u["unit_seed"]), []).append(u)
    jobs = [(str(state), str(out), lab, pan, seed, us, str(model) if model else None)
            for (lab, pan, seed), us in sorted(groups.items())]
    t0 = time.time()
    if workers <= 1:
        shards = [_work(j) for j in jobs]
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            shards = []
            for i, rec in enumerate(pool.imap(_work, jobs)):
                shards.append(rec)
                if i % 10 == 0:
                    print(f"[x65 bank] {i + 1}/{len(jobs)} shards, {time.time() - t0:.0f} s", flush=True)
    summ = summarize(shards)
    man = {"schema": MANIFEST_SCHEMA, "what": "X65 ExIt-0's label bank (PREREG §2)", "commit": commit,
           "dirty": dirty, "engine": eng, "source_state": str(state), "record_sha256": sha256_file(record),
           "lambda": LAMBDA, "from_turn": FROM_TURN, "inner_val": f"sha256(deal)[0] % {INNER_VAL_MOD} == 0",
           "max_drop_frac": MAX_DROP_FRAC, "tie_eps": TIE_EPS, "pstar": str(model) if model else None,
           "pstar_sha256": sha256_file(model) if model else None, "counted_units": len(units),
           "shards": [{k: s[k] for k in ("name", "label", "panel", "unit_seed", "rows", "labels", "sha256")}
                      for s in shards],
           "per_panel": summ["per_panel"], "void": summ["void"],
           "label_rows": int(sum(s["labels"] for s in shards)), "rows": int(sum(s["rows"] for s in shards)),
           "build_s": round(time.time() - t0, 1)}
    mp_ = out / "bank_manifest.json"
    mp_.write_text(json.dumps(man, indent=1, sort_keys=True))
    (out / "bank_manifest.sha256").write_text(sha256_file(mp_) + "\n")
    return man


def _dirty() -> bool:
    import subprocess

    from utils.paths import repo_path

    r = subprocess.run(["git", "-C", str(repo_path()), "status", "--porcelain", "--untracked-files=no"],
                       capture_output=True, text=True, check=False)
    return bool(r.stdout.strip())


# ------------------------------------------------------------------------------------------------ reading it back
def load_manifest(out: Path) -> tuple[dict[str, Any], str]:
    p = out / "bank_manifest.json"
    return json.loads(p.read_text()), sha256_file(p)


def check(out: Path, registered_sha256: str | None = None) -> dict[str, Any]:
    """The bank as the trainer will read it: the manifest's sha256 (``registered_sha256`` if given), not VOID, every
    shard's sha256. Raises :class:`BankError`."""
    man, sha = load_manifest(out)
    if man.get("schema") != MANIFEST_SCHEMA:
        raise BankError(f"{out}: schema {man.get('schema')!r}")
    if registered_sha256 is not None and sha != registered_sha256:
        raise BankError(f"VOID: the bank manifest {sha[:12]} is not the registered {registered_sha256[:12]}")
    if man.get("void"):
        raise BankError(f"VOID bank: {man['void']}")
    if man.get("engine") != E4 or man.get("dirty"):
        raise BankError("the bank was not built on E4 from a clean commit")
    for s in man["shards"]:
        if sha256_file(out / f"{s['name']}.npz") != s["sha256"]:
            raise BankError(f"{s['name']}: sha256 differs from the manifest")
    return {"manifest_sha256": sha, "rows": man["rows"], "label_rows": man["label_rows"]}


def load_rows(out: Path, panels: Iterable[str], fields: Sequence[str] = FIELDS,
              labels: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """The rows of ``panels`` (and of the arm ``labels``, default both), concatenated in manifest order, with a
    ``panel`` column (index into :data:`U_PANELS`)."""
    man, _ = load_manifest(out)
    want = set(panels)
    parts: dict[str, list[np.ndarray]] = {f: [] for f in fields}
    pan: list[np.ndarray] = []
    for s in man["shards"]:
        if s["panel"] not in want or (labels is not None and s["label"] not in labels) or not s["rows"]:
            continue
        with np.load(out / f"{s['name']}.npz") as z:
            for f in fields:
                parts[f].append(z[f])
        pan.append(np.full(s["rows"], U_PANELS.index(s["panel"]), np.int8))
    if not pan:
        raise BankError(f"no rows for {sorted(want)}")
    res = {f: np.concatenate(v) for f, v in parts.items()}
    res["panel"] = np.concatenate(pan)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x65.bank", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--state", required=True, help="S′-LIVE's read state")
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--pstar", default=None, help="P*'s zip: the anchor rows' greedy-action check")
    b.add_argument("--allow-dirty", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    c.add_argument("--manifest-sha256", default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            man = build(Path(a.state).expanduser(), Path(a.out).expanduser(), a.workers,
                        Path(a.pstar).expanduser() if a.pstar else None, a.allow_dirty)
            print(json.dumps({"rows": man["rows"], "label_rows": man["label_rows"], "void": man["void"],
                              "per_panel": {p: {k: v[k] for k in ("labels_recorded", "labels_kept",
                                                                  "label_drop_frac")}
                                            for p, v in man["per_panel"].items()}}, indent=1), flush=True)
            return 2 if man["void"] else 0
        print(json.dumps(check(Path(a.out).expanduser(), a.manifest_sha256)), flush=True)
        return 0
    except BankError as exc:
        print(f"[x65 bank] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
