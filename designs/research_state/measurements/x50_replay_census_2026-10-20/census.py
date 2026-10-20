"""X50 S0 -- the ZERO-GPU human-replay census on the Rust reader (2026-10-20).

What it measures, per public gen3ou replay of the project's own corpus (`<main>/replays/showdown/gen3ou/<date>/*.log`):

* READ (the Rust reader): each replay from BOTH seats through `live_reader` (`main.live.replay_scan.scan_one`, the
  same line filter and per-turn encoder PROBE as gate (b)) -> refusals, probe failures;
* REVEALED BY GAME END (the Rust reader): each seat's `present()` view after the whole log
  (`core_events --present-stream`) -> per side: mons revealed, moves / item / ability known per mon;
* THE TIMELINE (a text pass over the same filtered lines; the reader exposes no per-turn action record for a
  spectator stream, which has no `|request|`): per side per turn, the human's ACTION (move / switch / cant / ...),
  the forced decisions (faint replacements, Baton Pass targets), a per-decision mask-hazard proxy, and the
  OPPONENT-INTENT label each viewer's turn decision would get (two semantics: CHOICE = the opponent's observable
  turn choice; PROD = the production label rules of `agents/training/opp_intent_labels.py` /
  `rust_sim/src/trackers/mod.rs::IntentLabel::build`, approximated on the text -- see README "Method").

Every count is accumulated into string-keyed counters per rating band (`<1300`, `1300-1499`, `1500+`, `1700+` (a
subset of `1500+`), `unrated`, `all`) and written per DATE FOLDER as `shards/<date>.json.gz` (resumable: a date with
a shard is skipped). `aggregate` sums the shards into `results/census.json` + `results/census.md`.

    python census.py run --dates 2026-05-18,2026-05-19 --shards <dir> [--workers 6] [--limit N]
    python census.py aggregate --shards <dir> --out <dir>
    python census.py dates            # every date folder of the corpus, one per line

Read-only on the corpus and on `data/`. CPU only. Binaries: `$POKESIM_LIVE_READER_BIN` (live_reader) and
`$X50_CORE_EVENTS_BIN` (core_events), both from this checkout's `src/rust_sim/target/release/`.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from main.live.reader import LiveReader, ReaderRefusal
from main.live.replay_scan import battle_lines, default_corpus, scan_one, seat_names

HERE = Path(__file__).resolve().parent
SIDES = ("p1", "p2")
OTHER = {"p1": "p2", "p2": "p1"}

# --------------------------------------------------------------------------------------------- static tables
ARENA_TRAP = {"dugtrio", "diglett"}
SHADOW_TAG = {"wobbuffet", "wynaut"}
MAGNET_PULL = {"magneton", "magnemite"}
PARTIAL_TRAP = {"wrap", "bind", "firespin", "whirlpool", "sandtomb", "clamp"}
LOCKING_MOVES = {"outrage", "thrash", "petaldance", "rollout", "uproar", "bide", "icewall"}


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _load_tables() -> Tuple[Dict[str, List[str]], Set[str]]:
    """Species types and the species that CAN have Levitate (Arena Trap immunity), read-only from `data/`."""
    from utils.paths import repo_path
    sp = json.loads(repo_path("data", "pokemon", "gen3_species.json").read_text())
    ab = json.loads(repo_path("data", "pokemon", "gen3_ability_priors.json").read_text())
    types = {k: [t.lower() for t in v.get("types", [])] for k, v in sp.items() if isinstance(v, dict)}
    lev = {k for k, v in ab.items() if isinstance(v, dict) and "levitate" in v}
    _SINGLE_ABILITY.update(k for k, v in ab.items() if isinstance(v, dict) and len(v) == 1)
    return types, lev


_TYPES: Dict[str, List[str]] = {}
_LEVITATE: Set[str] = set()
_SINGLE_ABILITY: Set[str] = set()   # species with ONE ability in the Smogon priors: known from the species alone


# --------------------------------------------------------------------------------------------- the text timeline
def _ident(field: str) -> Tuple[Optional[str], str]:
    """`p1a: Nick` -> ('p1', 'Nick')."""
    m = re.match(r"^(p[12])[a-z]?: ?(.*)$", field)
    return (m.group(1), m.group(2)) if m else (None, "")


def _species_of(details: str) -> str:
    return to_id(details.split(",")[0])


class Side:
    def __init__(self) -> None:
        self.nick2sp: Dict[str, str] = {}
        self.active: Optional[str] = None
        self.seen_mons: Set[str] = set()
        self.seen_moves: Dict[str, Set[str]] = collections.defaultdict(set)
        self.vol: Set[str] = set()
        self.moved_since_in = False
        self.lock_next = False          # recharge / charge / locked-move continuation expected next turn


def _new_turn_rec(turn: int, active: Optional[str]) -> Dict[str, Any]:
    return {"turn": turn, "active": active, "pending": True, "outcome": None, "move": None, "move_rev": None,
            "hp": False, "target": None, "target_rev": None, "reason": None, "moved": False, "switched": False,
            "fainted": False, "dragged": False, "replacement": False, "encore_override": False, "haz": [],
            "moved_since_in": False}


def parse_timeline(lines: List[str]) -> Dict[str, Any]:
    """The per-side per-turn action record of one replay (see the module doc)."""
    names = seat_names(lines)
    ratings: Dict[str, Optional[int]] = {"p1": None, "p2": None}
    teamsize: Dict[str, int] = {"p1": 6, "p2": 6}
    st = {s: Side() for s in SIDES}
    turns: Dict[str, List[Dict[str, Any]]] = {"p1": [], "p2": []}
    fds: Dict[str, List[Dict[str, Any]]] = {"p1": [], "p2": []}
    anomalies: collections.Counter = collections.Counter()
    cur: Dict[str, Optional[Dict[str, Any]]] = {"p1": None, "p2": None}
    winner: Optional[str] = None
    ended = False
    bp_pending = {"p1": False, "p2": False}
    faint_pending = {"p1": False, "p2": False}

    def close_turn() -> None:
        for s in SIDES:
            r = cur[s]
            if r is not None and r["pending"]:
                r["pending"] = False
                r["outcome"] = "CANT_HIDDEN" if r["reason"] == "slp" else "NO_ACTION"

    def consume(s: str, outcome: str) -> Optional[Dict[str, Any]]:
        r = cur[s]
        if r is None or not r["pending"]:
            return None
        r["pending"] = False
        if r["reason"] == "slp" and outcome in ("FAINTED_BEFORE_ACTING", "SWITCH"):
            outcome = "CANT_HIDDEN"  # asleep and no sleep-usable move followed: the choice stays hidden
        r["outcome"] = "ENCORE_OVERRIDE" if r["encore_override"] and outcome in ("MOVE", "CANT_NAMED") else outcome
        return r

    def switch_in(s: str, nick: str, details: str, kind: str) -> None:
        sd = st[s]
        sp = _species_of(details)
        sd.nick2sp[nick] = sp
        rev = sp in sd.seen_mons
        r = cur[s]
        if kind == "drag":
            if r is not None:
                r["dragged"] = True
                r["switched"] = True
        elif r is not None and r["pending"] and not bp_pending[s] and not faint_pending[s]:
            rr = consume(s, "SWITCH")
            if rr is not None:
                rr["target"], rr["target_rev"], rr["switched"] = sp, rev, True
        elif bp_pending[s] or faint_pending[s]:
            k = "batonpass" if bp_pending[s] else "replacement"
            fds[s].append({"turn": r["turn"] if r else 0, "kind": k, "target": sp, "target_rev": rev,
                           "from": sd.active})
            if r is not None:
                r["switched"] = True
                r["replacement"] = r["replacement"] or k == "replacement"
            bp_pending[s] = False
            faint_pending[s] = False
        elif r is not None:
            anomalies["switch_unclassified"] += 1
            r["switched"] = True
        sd.active = sp
        sd.seen_mons.add(sp)
        sd.vol = set()
        sd.moved_since_in = False

    for ln in lines:
        parts = ln.split("|")
        if len(parts) < 2:
            continue
        kw = parts[1]
        if kw == "player" and len(parts) > 2 and parts[2] in SIDES:
            if len(parts) > 5 and parts[5].strip().isdigit() and ratings[parts[2]] is None:
                ratings[parts[2]] = int(parts[5].strip())
        elif kw == "teamsize" and len(parts) > 3 and parts[2] in SIDES and parts[3].isdigit():
            teamsize[parts[2]] = int(parts[3])
        elif kw == "turn":
            close_turn()
            bp_pending.update({"p1": False, "p2": False})
            faint_pending.update({"p1": False, "p2": False})
            n = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            for s in SIDES:
                sd, od = st[s], st[OTHER[s]]
                r = _new_turn_rec(n, sd.active)
                r["moved_since_in"] = sd.moved_since_in
                haz = []
                me, foe = sd.active or "", od.active or ""
                mt = _TYPES.get(me, [])
                if foe in SHADOW_TAG or (foe in MAGNET_PULL and "steel" in mt) or (
                        foe in ARENA_TRAP and "flying" not in mt and me not in _LEVITATE):
                    haz.append("trap_ability")
                for v in ("trapped", "partial", "taunt", "encore", "disable", "torment", "ingrain"):
                    if v in sd.vol:
                        haz.append(v)
                if sd.lock_next:
                    haz.append("locked")
                r["haz"] = haz
                sd.lock_next = False
                turns[s].append(r)
                cur[s] = r
        elif kw in ("switch", "drag") and len(parts) > 3:
            s, nick = _ident(parts[2])
            if s:
                switch_in(s, nick, parts[3], kw)
        elif kw == "replace" and len(parts) > 3:
            s, nick = _ident(parts[2])
            if s:
                st[s].nick2sp[nick] = _species_of(parts[3])
                anomalies["replace_line"] += 1
        elif kw == "move" and len(parts) > 3:
            s, nick = _ident(parts[2])
            if not s:
                continue
            sd = st[s]
            mv = to_id(parts[3])
            tags = [p for p in parts[4:] if p.startswith("[from]")]
            frm = to_id(tags[0][len("[from]"):]) if tags else ""
            sp = sd.nick2sp.get(nick, sd.active or "")
            r = cur[s]
            if r is not None:
                r["moved"] = True
            if frm == "lockedmove":
                consume(s, "LOCKED")
                sd.lock_next = mv in LOCKING_MOVES
            elif frm and frm != "pursuit":
                if frm == "sleeptalk":
                    sd.seen_moves[sp].add(mv)
                rr = consume(s, "CALLED_FIRST")
                if rr is not None:
                    anomalies["called_move_first"] += 1
            else:
                rev = mv in sd.seen_moves[sp]
                rr = consume(s, "MOVE")
                if rr is not None:
                    rr["move"], rr["move_rev"], rr["hp"] = mv, rev, mv.startswith("hiddenpower")
                elif r is not None and "[still]" not in parts[4:] and frm != "pursuit":
                    anomalies["move_after_consumed"] += 1
                sd.seen_moves[sp].add(mv)
                if mv == "batonpass":
                    bp_pending[s] = True
                if mv in LOCKING_MOVES:
                    sd.lock_next = True
            sd.moved_since_in = True
        elif kw == "cant" and len(parts) > 3:
            s, _nick = _ident(parts[2])
            if not s:
                continue
            reason = parts[3]
            named = to_id(parts[4]) if len(parts) > 4 and parts[4] else ""
            if to_id(reason) == "slp" and cur[s] is not None and cur[s]["pending"]:  # type: ignore[index]
                # Showdown prints `cant|slp` BEFORE a sleep-usable move (Sleep Talk / Snore) runs: the choice is
                # that move if one follows this turn, else hidden (close_turn / the next consume decide)
                cur[s]["reason"] = "slp"  # type: ignore[index]
                continue
            if to_id(reason) == "recharge":
                rr = consume(s, "LOCKED")
            elif named:
                rr = consume(s, "CANT_NAMED")
                if rr is not None:
                    rr["move"] = named
                    sp = st[s].active or ""
                    rr["move_rev"] = named in st[s].seen_moves[sp]
            else:
                rr = consume(s, "CANT_HIDDEN")
            if rr is not None:
                rr["reason"] = to_id(reason.replace("ability:", "").replace("move:", ""))
        elif kw == "faint" and len(parts) > 2:
            s, _nick = _ident(parts[2])
            if s:
                consume(s, "FAINTED_BEFORE_ACTING")
                if cur[s] is not None:
                    cur[s]["fainted"] = True  # type: ignore[index]
                faint_pending[s] = True
                bp_pending[s] = False
        elif kw == "-mustrecharge" and len(parts) > 2:
            s, _nick = _ident(parts[2])
            if s:
                st[s].lock_next = True
        elif kw == "-prepare" and len(parts) > 2:
            s, _nick = _ident(parts[2])
            if s:
                st[s].lock_next = True
        elif kw in ("-start", "-end", "-activate") and len(parts) > 3:
            s, _nick = _ident(parts[2])
            if not s:
                continue
            eff = to_id(parts[3].replace("move:", "").replace("ability:", "").replace("item:", ""))
            sd = st[s]
            key = {"taunt": "taunt", "encore": "encore", "disable": "disable", "torment": "torment",
                   "ingrain": "ingrain", "trapped": "trapped"}.get(eff)
            if eff in PARTIAL_TRAP:
                key = "partial"
            if key is None:
                continue
            if kw == "-end":
                sd.vol.discard(key)
            elif kw == "-start" or key in ("trapped", "partial"):
                sd.vol.add(key)
                if key == "encore" and cur[s] is not None and cur[s]["pending"]:  # type: ignore[index]
                    cur[s]["encore_override"] = True  # type: ignore[index]
        elif kw == "win":
            close_turn()
            ended = True
            wname = parts[2] if len(parts) > 2 else ""
            for s in SIDES:
                if wname and wname in names.get(s, []):
                    winner = s
            break
        elif kw == "tie":
            close_turn()
            ended = True
            break
    if not ended:
        close_turn()
    return {"names": {s: (names.get(s) or [s])[0] for s in SIDES}, "ratings": ratings, "teamsize": teamsize,
            "turns": turns, "fds": fds, "winner": winner, "ended": ended, "anomalies": anomalies}


# --------------------------------------------------------------------------------------------- the Rust end view
def present_view(core_events: str, seat_idx: int, name: str, lines: List[str]) -> Tuple[Optional[dict], str]:
    head = json.dumps({"viewer": seat_idx, "username": name, "team": None}, ensure_ascii=False)
    try:
        p = subprocess.run([core_events, "--present-stream"], input=head + "\n" + "\n".join(lines) + "\n",
                           capture_output=True, text=True, encoding="utf-8", timeout=120)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    try:
        obj = json.loads(p.stdout)
    except json.JSONDecodeError:
        return None, f"bad output rc={p.returncode} {p.stderr[-200:]!r}"
    if not obj.get("ok"):
        return None, str(obj.get("error"))[:200]
    return obj["view"], ""


def side_reveals(view: dict) -> Dict[str, Dict[str, Any]]:
    """species -> {moves:set, item, abil, abil_public} for the VIEWER'S OWN side at game end."""
    out: Dict[str, Dict[str, Any]] = {}
    for m in view["ours"]["mons"]:
        if not m.get("revealed", True):
            continue
        item = m.get("item") or m.get("consumed_item")
        out[m["species"]] = {"moves": {to_id(x["id"]) for x in m.get("moves", [])},
                             "item": to_id(item) if item else None,
                             "abil": m.get("ability"), "abil_public": bool(m.get("ability_public"))}
    return out


# --------------------------------------------------------------------------------------------- counting
def bands(r: Optional[int]) -> List[str]:
    if r is None:
        return ["all", "unrated"]
    b = ["all", "<1300" if r < 1300 else ("1300-1499" if r < 1500 else "1500+")]
    if r >= 1700:
        b.append("1700+")
    return b


def _nmoves(rev: Dict[str, Dict[str, Any]], sp: Optional[str]) -> int:
    if not sp or sp not in rev:
        return 0
    return len({m[:11] if m.startswith("hiddenpower") else m for m in rev[sp]["moves"]})


def count_replay(tl: Dict[str, Any], views: Dict[str, Optional[dict]], C: collections.Counter) -> None:
    rev = {s: (side_reveals(views[s]) if views[s] is not None else None) for s in SIDES}
    for s in SIDES:
        o = OTHER[s]
        bs = bands(tl["ratings"][s])
        won = "won" if tl["winner"] == s else ("lost" if tl["winner"] == o else "nores")

        def inc(key: str, n: int = 1, _bs: List[str] = bs) -> None:
            for b in _bs:
                C[f"{b}|{key}"] += n
        if tl["ratings"][s] is not None:
            C[f"hist|{min(tl['ratings'][s] // 50 * 50, 2200)}"] += 1
        inc("sides")
        R = rev[s]
        if R is None:
            inc("sides_no_view")
            continue
        ts = tl["teamsize"][s]
        nrev = len(R)
        full = nrev >= ts
        inc(f"mons_revealed|{min(nrev, 6)}")
        all4 = full and all(_nmoves(R, sp) >= 4 for sp in R)
        all_items = all(R[sp]["item"] for sp in R)
        all_abil = all(R[sp]["abil"] for sp in R)
        for sp in R:
            inc("mons")
            inc(f"mon_moves|{min(_nmoves(R, sp), 4)}")
            inc("mon_item_known", int(bool(R[sp]["item"])))
            inc("mon_abil_known", int(bool(R[sp]["abil"])))
            inc("mon_abil_public", int(R[sp]["abil_public"]))
            inc("mon_abil_single", int(sp in _SINGLE_ABILITY))
            inc("mon_abil_known_multi", int(bool(R[sp]["abil"]) and sp not in _SINGLE_ABILITY))
        if full:
            inc("sides_full")
            inc(f"sides_full|{won}")
        inc(f"sides|{won}")
        # ---- the BC side: every decision of s, its set revealed by game end
        decs = [("td", t["active"], t) for t in tl["turns"][s]] + [("fd_" + f["kind"], f["from"], f) for f in tl["fds"][s]]
        for kind, act, rec in decs:
            if act and act not in R:
                C["all|check|acting_species_not_in_view"] += 1
            a4 = _nmoves(R, act) >= 4
            ai = bool(act in R and R[act]["item"])
            aa = bool(act in R and R[act]["abil"])
            keys = ["dec", f"dec|{kind}", f"dec|{won}"]
            if a4:
                keys.append("act4")
            if ai:
                keys.append("act_item")
            if aa:
                keys.append("act_abil")
            if full:
                keys.append("full")
                if a4:
                    keys += ["full_act4", f"full_act4|{won}", f"full_act4|{kind}"]
                if all4:
                    keys.append("full_all4")
                    if all_items:
                        keys.append("full_all4_items")
                        if all_abil:
                            keys.append("full_all4_items_abil")
            for k in keys:
                inc(k)
            if kind == "td":
                inc(f"bc|{rec['outcome']}")
                haz = list(rec["haz"])
                if act in R and R[act]["item"] == "choiceband" and rec["moved_since_in"]:
                    haz.append("choice")
                for h in haz:
                    inc(f"haz|{h}")
                if haz:
                    inc("haz|any")
                elif full and a4:
                    inc("full_act4_hazclean")
                # consistency text vs reader: a move / switch the timeline saw must be in the end view
                if rec["outcome"] == "MOVE" and act:
                    mv = rec["move"]
                    inview = act in R and (mv in R[act]["moves"] or (mv.startswith("hiddenpower")
                                           and any(x.startswith("hiddenpower") for x in R[act]["moves"])))
                    C["all|check|move_in_view" if inview else "all|check|move_not_in_view"] += 1
                if rec["outcome"] == "SWITCH":
                    C["all|check|switch_in_view" if rec["target"] in R else "all|check|switch_not_in_view"] += 1
            else:
                inc(f"bc_fd|{'target_rev' if rec['target_rev'] else 'target_new'}")
        # ---- the INTENT side: viewer o, label = s's action at o's turn decision (band by s = the labelled human)
        ofd_turns = {f["turn"] for f in tl["fds"][o]}
        oturns = {t["turn"]: t for t in tl["turns"][o]}
        for t in tl["turns"][s]:
            if t["turn"] not in oturns:
                continue
            inc("int|td")
            oc = t["outcome"]
            if oc == "MOVE":
                c = "MOVE_hp" if t["hp"] else ("MOVE_rev" if t["move_rev"] else "MOVE_new")
            elif oc == "SWITCH":
                c = "SWITCH_rev" if t["target_rev"] else "SWITCH_new"
            elif oc == "CANT_NAMED":
                c = "CANT_NAMED"
            else:
                c = "X_" + oc
            inc(f"int|choice|{c}")
            # PROD (approximated): the production window rules
            if t["turn"] in ofd_turns:
                p = "X_viewer_forced_window"
            elif t["encore_override"]:
                p = "X_encore_override"
            elif t["switched"]:
                if t["moved"] or t["fainted"] or t["dragged"] or t["replacement"]:
                    p = "X_switch_cooccur"
                else:
                    p = "SWITCH_rev" if t["target_rev"] else "SWITCH_new"
            elif t["moved"] and oc in ("MOVE", "LOCKED", "ENCORE_OVERRIDE", "CALLED_FIRST"):
                if oc == "LOCKED":
                    p = "MOVE_locked"
                elif oc != "MOVE":
                    p = "X_" + oc
                else:
                    p = "MOVE_hp" if t["hp"] else ("MOVE_rev" if t["move_rev"] else "MOVE_new")
            elif oc == "CANT_NAMED":
                p = "CANT_NAMED"
            else:
                p = "X_" + str(oc)
            inc(f"int|prod|{p}")


# --------------------------------------------------------------------------------------------- the worker
_READER: Optional[LiveReader] = None


def _init() -> None:
    global _TYPES, _LEVITATE
    _TYPES, _LEVITATE = _load_tables()


def work(path: str) -> Tuple[collections.Counter, Dict[str, Any]]:
    global _READER
    C: collections.Counter = collections.Counter()
    ex: Dict[str, Any] = {}
    core_events = os.environ["X50_CORE_EVENTS_BIN"]
    C["all|replays"] += 1
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = battle_lines(text)
    # 1. the Rust reader, both seats, per-turn encoder probe (gate (b)'s scan)
    if _READER is None:
        _READER = LiveReader()
    try:
        sc = scan_one(_READER, path, SIDES, True)
    except ReaderRefusal as exc:
        _READER.close()
        _READER = None
        sc = {"refusals": [{"seat": "?", "kind": exc.kind, "message": str(exc)}], "probe_failures": [], "probes": 0}
    C["all|reader|seats_read"] += 2
    C["all|reader|probes"] += sc["probes"]
    for x in sc["refusals"]:
        C["all|reader|refusal_classified" if x.get("class") else "all|reader|refusal"] += 1
        ex.setdefault("refusal", {"path": path, **x})
    for x in sc["probe_failures"]:
        C["all|reader|probe_failure"] += 1
        ex.setdefault("probe_failure", {"path": path, **x})
    # 2. the timeline
    tl = parse_timeline(lines)
    for k, v in tl["anomalies"].items():
        C[f"all|anomaly|{k}"] += v
    if not tl["ended"]:
        C["all|replays_unfinished"] += 1
    # 3. the Rust end view per seat
    views: Dict[str, Optional[dict]] = {}
    for i, s in enumerate(SIDES):
        v, err = present_view(core_events, i, tl["names"][s], lines)
        views[s] = v
        if v is None:
            C["all|reader|present_fail"] += 1
            ex.setdefault("present_fail", {"path": path, "seat": s, "error": err})
    count_replay(tl, views, C)
    return C, ex


def cmd_dates(a: argparse.Namespace) -> int:
    corpus = Path(a.corpus or default_corpus() or "")
    for d in sorted(p.name for p in corpus.iterdir() if p.is_dir()):
        print(d)
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    corpus = Path(a.corpus or default_corpus() or "")
    shards = Path(a.shards)
    shards.mkdir(parents=True, exist_ok=True)
    for date in [d for d in a.dates.split(",") if d]:
        out = shards / f"{date}.json.gz"
        if out.exists():
            print(f"[x50] {date}: shard exists, skipped", flush=True)
            continue
        files = sorted(str(p) for p in (corpus / date).glob("*.log"))
        if a.limit:
            files = files[: a.limit]
        t0 = time.time()
        C: collections.Counter = collections.Counter()
        ex: Dict[str, Any] = {}
        with Pool(a.workers, initializer=_init) as pool:
            for c, e in pool.imap_unordered(work, files, chunksize=16):
                C.update(c)
                for k, v in e.items():
                    ex.setdefault(k, v)
        rec = {"date": date, "files": len(files), "seconds": round(time.time() - t0, 1), "counters": dict(C),
               "examples": ex, "live_reader": os.environ.get("POKESIM_LIVE_READER_BIN"),
               "core_events": os.environ.get("X50_CORE_EVENTS_BIN"), "limit": a.limit}
        tmp = out.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(rec, fh, sort_keys=True)
        tmp.rename(out)
        print(f"[x50] {date}: {len(files)} replays in {rec['seconds']}s "
              f"refusals={C['all|reader|refusal']} present_fail={C['all|reader|present_fail']}", flush=True)
    return 0


def cmd_aggregate(a: argparse.Namespace) -> int:
    from census_report import report  # the readout, beside this script
    C: collections.Counter = collections.Counter()
    meta = {"dates": [], "files": 0, "seconds": 0.0, "examples": {}, "bins": set()}
    for p in sorted(Path(a.shards).glob("*.json.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            rec = json.load(fh)
        C.update(rec["counters"])
        meta["dates"].append(rec["date"])
        meta["files"] += rec["files"]
        meta["seconds"] += rec["seconds"]
        meta["bins"].add((rec["live_reader"], rec["core_events"]))
        for k, v in rec["examples"].items():
            meta["examples"].setdefault(k, v)
    meta["bins"] = sorted(meta["bins"])
    return report(C, meta, Path(a.out))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dates", required=True)
    r.add_argument("--shards", required=True)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--limit", type=int, default=None)
    g = sub.add_parser("aggregate")
    g.add_argument("--shards", required=True)
    g.add_argument("--out", required=True)
    sub.add_parser("dates")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "aggregate": cmd_aggregate, "dates": cmd_dates}[a.cmd](a)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    sys.exit(main())
