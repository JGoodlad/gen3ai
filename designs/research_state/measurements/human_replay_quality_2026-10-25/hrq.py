"""Human-replay DATA CARD (2026-10-25): is the public gen3ou replay corpus good training data, by stratum?

    python hrq.py extract --cache <dir> [--workers 3]          # one jsonl.gz per date folder (resumable); text + reveals
    python hrq.py analyze --cache <dir> --out <dir>            # the card: results.json + results.md

CPU only, read-only on the corpus (``$HRQ_CORPUS``, default the main checkout's ``replays/showdown/gen3ou``), on
``data/teams`` (the 719-team pool), on the registered panels (never the SEALED ``forum216b``) and on the X50 rung-1b
archive (``$HRQ_X50``: the extraction's part JSONs / small npz keys and R-b1's cached per-row forwards). Design and
registration: ``designs/endstate/design_human_replay_lever.md`` §2 (the thresholds below are restated there and were
fixed before the analysis ran).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

CORPUS = Path(os.environ.get("HRQ_CORPUS", "/home/goodlad/dev/gen3ai/replays/showdown/gen3ou"))
X50 = Path(os.environ.get("HRQ_X50", str(Path.home() / "gen3ai_archive" / "x50_r1b")))
MANIFEST = "SPECTATOR_MANIFEST_2026-09-23.txt"
REPO = Path(os.environ.get("HRQ_REPO", str(Path(__file__).resolve().parents[4])))

# ---- the registered thresholds (design §2.1; fixed before `analyze` first ran) ----
STRATA = ((0, 1300, "<1300"), (1300, 1500, "1300-1499"), (1500, 1600, "1500-1599"), (1600, 1700, "1600-1699"),
          (1700, 10_000, "1700+"))
#: a name a bot or an agent harness uses (Metamon's PAC-* / h2h-* accounts and its public ensembles; ours).
BOT_NAME = re.compile(r"(^pac[- ]|^h2h[- ]|metamon|pastamittens|exeggcute|kakuna|foul ?play|gen3ai|\bbot\b|bot$|^bot)",
                      re.I)
BOT_GAMES_PER_DAY = 100      # an account with >= this many spectator games on one Los Angeles date
BOT_HOURS_PER_DAY = 20       # or games started in >= this many distinct clock hours of one date
FAST_S_PER_TURN = 2.0        # a game played faster than this per turn (>= 10 turns) is machine speed on BOTH sides
EARLY_FORFEIT_TURNS = 3      # a forfeit / timer loss at or before this turn is a walk-over
TRAIN_LAST = "2026-09-20"    # X50's train / held-out split (held out 09-21..23)

_PLAYER = re.compile(r"^\|player\|(p[12])\|([^|]*)\|([^|]*)\|(\d*)")
_SWITCH = re.compile(r"^\|(?:switch|drag|replace)\|(p[12])[ab]?: [^|]*\|([^|,]+)")
_FORFEIT = re.compile(r"^\|-message\|(.*) forfeited\.$")
_TIMER = re.compile(r"^\|-message\|(.*) lost due to inactivity\.$")
_SKIP_BODY = ("|init|", "|title|", "|j|", "|J|", "|l|", "|L|", "|c|", "|raw|", "|inactive|", "|n|", "|N|", "|t:|")


def toid(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse(text: str) -> Optional[dict]:
    """One battle -> a record (None if no |player| line). Text pass + the reveals (``agents.info_levers.revealed``)."""
    from agents.info_levers import revealed

    lines = text.split("\n")
    rec: Dict[str, Any] = {"rated": False, "gen": None, "tier": None, "gametype": None, "turns": 0, "end": "none",
                           "win": None, "loser_quit": None}
    names: Dict[str, str] = {}
    rating: Dict[str, Optional[int]] = {"p1": None, "p2": None}
    rules: List[str] = []
    order: Dict[str, List[str]] = {"p1": [], "p2": []}
    ts: List[int] = []
    chat = {"p1": 0, "p2": 0}
    body = hashlib.sha1()
    winner = None
    for ln in lines:
        if not ln.startswith("|"):
            continue
        if ln.startswith("|t:|"):
            try:
                ts.append(int(ln[4:]))
            except ValueError:
                pass
            continue
        if ln.startswith("|c|"):
            u = ln.split("|")[2:3]
            if u:
                who = toid(u[0])
                for s, n in names.items():
                    if toid(n) == who:
                        chat[s] += 1
            continue
        if not ln.startswith(_SKIP_BODY):
            body.update(ln.encode())
        if ln.startswith(("|switch|", "|drag|", "|replace|")):
            m = _SWITCH.match(ln)
            if m:
                sid = toid(m.group(2))
                if sid not in order[m.group(1)]:
                    order[m.group(1)].append(sid)
        elif ln.startswith("|turn|"):
            try:
                rec["turns"] = int(ln[6:])
            except ValueError:
                pass
        elif ln.startswith("|player|"):
            m = _PLAYER.match(ln)
            if m and m.group(2) and m.group(1) not in names:
                names[m.group(1)] = m.group(2)
                rating[m.group(1)] = int(m.group(4)) if m.group(4) else None
        elif ln.startswith("|rated|"):
            rec["rated"] = True
        elif ln.startswith("|gen|"):
            rec["gen"] = ln[5:]
        elif ln.startswith("|tier|"):
            rec["tier"] = ln[6:]
        elif ln.startswith("|gametype|"):
            rec["gametype"] = ln[10:]
        elif ln.startswith("|rule|"):
            rules.append(ln[6:].split(":")[0])
        elif ln.startswith("|-message|"):
            m = _FORFEIT.match(ln) or _TIMER.match(ln)
            if m:
                rec["end"] = "forfeit" if "forfeited" in ln else "timer"
                q = toid(m.group(1))
                rec["loser_quit"] = next((s for s, n in names.items() if toid(n) == q), None)
        elif ln.startswith("|win|"):
            winner = ln[5:]
            if rec["end"] == "none":
                rec["end"] = "ko"
            break
        elif ln == "|tie" or ln.startswith("|tie|"):
            rec["end"] = "tie"
            break
    if not names:
        return None
    if rec["end"] in ("forfeit", "timer") and winner is None:
        rec["end"] = "none"
    rec["win"] = next((s for s, n in names.items() if n == winner), None) if winner else None
    rec.update(n1=names.get("p1"), n2=names.get("p2"), r1=rating["p1"], r2=rating["p2"],
               rules=hashlib.sha1("|".join(sorted(rules)).encode()).hexdigest()[:10],
               t0=ts[0] if ts else None, t1=ts[-1] if ts else None, chat1=chat["p1"], chat2=chat["p2"],
               body=body.hexdigest()[:16])
    for seat, viewer in (("p1", 1), ("p2", 0)):                    # the OTHER seat's view of this one (memory.py)
        rv = revealed.from_lines(lines, viewer)
        mons = {revealed.toid(k): [sorted(revealed.toid(x) for x in v.moves), v.item, v.ability]
                for k, v in rv.mons.items()}
        rec["m" + seat[1]] = mons
        rec["o" + seat[1]] = order[seat]
    return rec


def fit(moves: List[str], item: Optional[str], ability: Optional[str], ms: Any) -> bool:
    """``main.x50_r1.memory._fits`` with the species already matched (canonical): every revealed move in the set
    (a bare Hidden Power fits any typed one; Struggle is no reveal), the revealed item and ability equal."""
    for mv in moves:
        if mv == "struggle":
            continue
        if mv == "hiddenpower":
            if not any(x.startswith("hiddenpower") for x in ms.moves):
                return False
        elif mv not in ms.moves:
            return False
    if item is not None and item != ms.item:
        return False
    if ability is not None and ability != ms.ability:
        return False
    return True


def manifest_set() -> set:
    return set((CORPUS / MANIFEST).read_text().split())


def extract_day(args) -> str:
    date, cache, spec = args
    out = Path(cache) / f"{date}.jsonl.gz"
    if out.exists():
        return f"{date} cached"
    rows, errs = [], 0
    for p in sorted((CORPUS / date).iterdir()):
        if p.suffix != ".log":
            continue
        try:
            rec = parse(p.read_text(errors="replace"))
        except Exception as e:                                     # noqa: BLE001 - counted, never silent
            errs += 1
            rows.append({"id": p.stem, "date": date, "err": f"{type(e).__name__}: {e}"[:200]})
            continue
        if rec is None:
            rows.append({"id": p.stem, "date": date, "err": "no player line"})
            continue
        rec["id"], rec["date"] = p.stem, date
        rec["pop"] = "spec" if f"{date}/{p.name}" in spec else "arch"
        rows.append(rec)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    tmp.rename(out)
    return f"{date} {len(rows)} ({errs} errors)"


def cmd_extract(a) -> None:
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    spec = manifest_set()
    dates = sorted(d.name for d in CORPUS.iterdir() if d.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name))
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(extract_day, [(d, a.cache, spec) for d in dates]):
            print(msg, flush=True)


def _sha(path: Path) -> Tuple[str, int]:
    b = path.read_bytes()
    return hashlib.sha256(b).hexdigest(), len(b)


def cmd_snapshot(a) -> None:
    """Freeze a corpus SNAPSHOT: a manifest (relative path, sha256, bytes) and its digest = the snapshot id's hash.
    ``spec``: the spectator manifest's paths; ``arch``: every other ``.log`` present now (dates <= ``--until``)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = manifest_set()
    if a.population == "spec":
        rels = sorted(spec)
    else:
        rels = sorted(f"{d.name}/{p.name}" for d in CORPUS.iterdir()
                      if d.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name) and d.name <= a.until
                      for p in d.iterdir() if p.suffix == ".log" and f"{d.name}/{p.name}" not in spec)
    with Pool(a.workers) as pool:
        shas = pool.map(_sha, [CORPUS / r for r in rels], chunksize=512)
    lines = [f"{r}\t{h}\t{n}" for r, (h, n) in zip(rels, shas)]
    body = ("\n".join(lines) + "\n").encode()
    digest = hashlib.sha256(body).hexdigest()
    with gzip.open(out / f"{a.id}.tsv.gz", "wb") as f:
        f.write(body)
    rec = {"id": a.id, "population": a.population, "files": len(rels), "bytes": sum(n for _, n in shas),
           "dates": [rels[0][:10], rels[-1][:10]] if rels else None, "manifest_sha256": digest,
           "manifest": f"{a.id}.tsv.gz", "corpus": str(CORPUS)}
    (out / f"{a.id}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


# ------------------------------------------------------------------------------------------------- analysis
def load(cache: Path) -> List[dict]:
    out: List[dict] = []
    for f in sorted(cache.glob("*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            out += [json.loads(x) for x in fh]
    return out


def stratum(r: Optional[int]) -> str:
    if r is None:
        return "unrated"
    for lo, hi, name in STRATA:
        if lo <= r < hi:
            return name
    return "unrated"


def pct(xs: Iterable[float], qs=(10, 50, 90)) -> Dict[str, float]:
    a = np.asarray(list(xs), dtype=float)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} if a.size else {}


def la_date_hour(t: Optional[int]) -> Tuple[Optional[str], Optional[int]]:
    if t is None:
        return None, None
    d = dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=-7)))    # PDT across the window (05-18..09-23)
    return d.date().isoformat(), d.hour


class Teams:
    """The pool and the registered panels as species / set references (``team_overlap``'s parser and forme rule)."""

    def __init__(self):
        from main import heldout_panels as HP
        from main.anchors.team_overlap import _species_id, parse_team
        from utils.team_loader import TeamLoader

        self.sid = _species_id
        pool = TeamLoader(str(REPO / "data" / "teams")).get_all_teams()
        self.ref: Dict[str, List[Any]] = {"pool": [parse_team(t, f"pool{i}") for i, t in enumerate(pool)]}
        self.strata: Dict[str, str] = {}
        for name in ("home72", "heldout72_hpfix", "forum72", "p72"):
            items = HP.panel_team_texts(name)                     # refuses a SEALED panel (forum216b is never read)
            self.ref[name] = [parse_team(txt, lab) for lab, txt in items]
        man = json.loads((REPO / "designs/research_state/measurements/x37a_eval_matrix_2026-10-18/"
                          "forum72_manifest.json").read_text())
        for s, ids in man.get("strata", {}).items():
            for i in ids:
                self.strata[str(i)] = s
        self._canon: Dict[str, str] = {}
        self.byteam: Dict[str, List[Dict[str, Any]]] = {}
        self.index: Dict[str, Dict[str, List[Tuple[int, Any]]]] = {}
        self.vec: Dict[str, Dict[str, np.ndarray]] = {}
        for name, teams in self.ref.items():
            idx: Dict[str, List[Tuple[int, Any]]] = collections.defaultdict(list)
            vec: Dict[str, np.ndarray] = {}
            for i, t in enumerate(teams):
                for ms in t.mons:
                    idx[ms.species].append((i, ms))
                    vec.setdefault(ms.species, np.zeros(len(teams), np.int8))[i] = 1
            self.index[name], self.vec[name] = dict(idx), vec
            self.byteam[name] = [{ms.species: ms for ms in t.mons} for t in teams]

    def canon(self, k: str) -> str:
        c = self._canon.get(k)
        if c is None:
            c = self._canon[k] = self.sid(k)
        return c

    def overlaps(self, mons: Dict[str, list], name: str, min_sp: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Per team of reference ``name``: shared species, and mons whose revealed facts FIT one of its sets
        (``memory._fits``' rule on canonical species; fits counted only where >= ``min_sp`` species are shared)."""
        vec, n = self.vec[name], len(self.ref[name])
        sp = np.zeros(n, np.int16)
        canon = [(self.canon(k), v) for k, v in mons.items()]
        for c, _ in canon:
            v = vec.get(c)
            if v is not None:
                sp += v
        fits = np.zeros(n, np.int16)
        for i in np.nonzero(sp >= min_sp)[0]:
            by_sp = self.byteam[name][i]
            fits[i] = sum(1 for c, (mv, it, ab) in canon if c in by_sp and fit(mv, it, ab, by_sp[c]))
        return sp, fits

    def side(self, mons: Dict[str, list], name: str) -> Tuple[int, int, int]:
        """(max species overlap, max fitting mon sets, argmax team by fits then species) against reference ``name``."""
        n = len(self.ref[name])
        if not n:
            return 0, 0, -1
        sp, fits = self.overlaps(mons, name)
        return int(sp.max()), int(fits.max()), int(np.lexsort((sp, fits))[-1])


def accounts(recs: List[dict]) -> Dict[str, Any]:
    per: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        d, h = la_date_hour(r.get("t0"))
        for s in ("1", "2"):
            n = r.get("n" + s)
            if not n:
                continue
            a = per.setdefault(toid(n), {"name": n, "games": 0, "days": collections.Counter(), "hours":
                                         collections.defaultdict(set), "rat": [], "chat": 0, "teams":
                                         collections.Counter()})
            a["games"] += 1
            if d:
                a["days"][d] += 1
                a["hours"][d].add(h)
            if r.get("r" + s):
                a["rat"].append(r["r" + s])
            a["chat"] += r.get("chat" + s, 0)
            o = r.get("o" + s) or []
            if len(o) == 6:
                a["teams"][tuple(sorted(o))] += 1
    return per


def bot_flags(per: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for k, a in per.items():
        why = []
        if BOT_NAME.search(a["name"]):
            why.append("name")
        if a["days"] and max(a["days"].values()) >= BOT_GAMES_PER_DAY:
            why.append("volume")
        if a["hours"] and max(len(v) for v in a["hours"].values()) >= BOT_HOURS_PER_DAY:
            why.append("hours")
        if why:
            out[k] = why
    return out


def x50_rows() -> Dict[str, Any]:
    """Label quality from the X50 rung-1b TRAINING extraction (738 parts): counts by the acting human's stratum."""
    parts = sorted((X50 / "train").glob("humans_*.json"))
    tot: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    counts = collections.Counter()
    for js in parts:
        meta = json.loads(js.read_text())
        counts.update({k: v for k, v in meta.get("counts", {}).items() if isinstance(v, int)})
        counts["parts"] += 1
        counts["team_memory_excluded"] += len(meta.get("team_memory_excluded", []))
        z = np.load(js.with_suffix(".npz"))
        mode, act, nl, rv, fo = (z[k] for k in ("mode", "own_action", "n_legal", "rating_viewer", "forced"))
        for lo, hi, name in STRATA:
            s = (rv >= lo) & (rv < hi)
            c = tot[name]
            c["rows"] += int(s.sum())
            c["completed"] += int((s & (mode == 2)).sum())
            c["own_label"] += int((s & (mode == 2) & (act >= 0)).sum())
            c["informative"] += int((s & (mode == 2) & (act >= 0) & (nl >= 2)).sum())
            c["switch_label"] += int((s & (mode == 2) & (act >= 0) & (act <= 5) & (nl >= 2)).sum())
            c["forced"] += int((s & (mode == 2) & fo.astype(bool) & (act >= 0) & (nl >= 2)).sum())
        s = rv <= 0
        tot["unrated"]["rows"] += int(s.sum())
        tot["unrated"]["informative"] += int((s & (mode == 2) & (act >= 0) & (nl >= 2)).sum())
    return {"parts": len(parts), "counts": dict(counts), "by_stratum": {k: dict(v) for k, v in tot.items()}}


def hom_join(by_path: Dict[str, dict], teams: Teams) -> Dict[str, Any]:
    """R-b1's cached per-row NLLs (HOM, P*) joined to the acting human's battle and team (descriptive, post hoc)."""
    from main.x50_r1.read_r0 import _ci, joint_boot

    rows = collections.defaultdict(list)
    memo: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
    cache = X50 / "rb1_cache"
    for js in sorted((X50 / "heldout").glob("humans_*.json")):
        meta = json.loads(js.read_text())
        z = np.load(js.with_suffix(".npz"))
        sel = (z["mode"] == 2) & (z["own_action"] >= 0) & (z["n_legal"] >= 2)
        hz = np.load(next(cache.glob(f"hom_*_{js.stem}.npz")))
        pz = np.load(next(cache.glob(f"pstar_*_{js.stem}.npz")))
        if int(sel.sum()) != len(hz["nll"]):
            raise SystemExit(f"{js.stem}: rb1 cache rows {len(hz['nll'])} != selection {int(sel.sum())}")
        bat, seat, rv, turn = z["battle"][sel], z["seat"][sel], z["rating_viewer"][sel], z["turn"][sel]
        base = int(js.stem.split("_")[-1]) * int(meta["part_size"])
        for j in range(len(bat)):
            path = meta["battles"][int(bat[j]) - base]          # the part's battle index is global (part x size)
            key = "/".join(Path(path).parts[-2:])
            rec = by_path.get(key)
            s = "1" if int(seat[j]) == 0 else "2"
            rows["seat_ok"].append(rec is not None and rec.get("r" + s) == int(rv[j]))
            mk = (key, s)
            if mk not in memo:
                memo[mk] = teams.side(rec["m" + s], "pool") if rec else (-1, -1, -1)
            fam = memo[mk]
            rows["fam_sp"].append(fam[0])
            rows["fam_fit"].append(fam[1])
            rows["end"].append(rec["end"] if rec else "?")
            rows["turns"].append(rec["turns"] if rec else -1)
            rows["cluster"].append(int(re.sub(r"\D", "", Path(key).stem)))
        rows["rv"].append(rv)
        rows["turn"].append(turn)
        rows["hom"].append(hz["nll"])
        rows["pstar"].append(pz["nll"])
    A = {k: (np.concatenate(v) if isinstance(v[0], np.ndarray) else np.asarray(v)) for k, v in rows.items()}
    out: Dict[str, Any] = {"rows": int(len(A["hom"])), "seat_check_share": float(np.mean(A["seat_ok"]))}
    prim = A["rv"] >= 1500

    def cell(sel):
        b = A["cluster"][sel]
        pops = {"d": (A["hom"][sel] - A["pstar"][sel], b), "hom": (A["hom"][sel], b), "pstar": (A["pstar"][sel], b)}
        bt = joint_boot(pops, np.random.default_rng(50))
        return {"rows": int(sel.sum()), "battles": int(len(np.unique(b))),
                **{k: {"mean": float(np.mean(v[0])), "ci95": _ci(bt[k])} for k, v in pops.items()}}

    strata = {
        "pool species overlap <= 3 (unfamiliar team)": prim & (A["fam_sp"] >= 0) & (A["fam_sp"] <= 3),
        "pool species overlap 4": prim & (A["fam_sp"] == 4),
        "pool species overlap 5-6 (familiar team)": prim & (A["fam_sp"] >= 5),
        "pool set fits <= 2": prim & (A["fam_fit"] >= 0) & (A["fam_fit"] <= 2),
        "pool set fits >= 4 (pool-like sets)": prim & (A["fam_fit"] >= 4),
        "turn <= 5": prim & (A["turn"] <= 5),
        "turn 6-20": prim & (A["turn"] >= 6) & (A["turn"] <= 20),
        "turn > 20": prim & (A["turn"] > 20),
        "game ends in forfeit / timer": prim & np.isin(A["end"], ["forfeit", "timer"]),
        "game played out (KO)": prim & (A["end"] == "ko"),
    }
    out["strata"] = {k: cell(s) for k, s in strata.items() if s.sum() >= 200}
    return out


def cmd_analyze(a) -> None:
    recs = load(Path(a.cache))
    res: Dict[str, Any] = {"schema": "hrq_v1", "corpus": str(CORPUS)}
    errs = [r for r in recs if "err" in r]
    ok = [r for r in recs if "err" not in r]
    spec = [r for r in ok if r["pop"] == "spec"]
    arch = [r for r in ok if r["pop"] == "arch"]
    res["files"] = {"total": len(recs), "errors": len(errs), "error_examples": errs[:5],
                    "spectator": len(spec), "archive": len(arch),
                    "spectator_dates": [min(r["date"] for r in spec), max(r["date"] for r in spec)],
                    "archive_dates": [min(r["date"] for r in arch), max(r["date"] for r in arch)] if arch else None}
    # ---- format / version
    fmt = collections.Counter((r["gen"], r["tier"], r["gametype"]) for r in spec)
    rules = collections.Counter(r["rules"] for r in spec)
    res["format"] = {"gen_tier_gametype": {" / ".join(map(str, k)): v for k, v in fmt.most_common(6)},
                     "rule_sets": dict(rules.most_common(5)), "n_rule_sets": len(rules),
                     "no_result": sum(r["end"] == "none" for r in spec)}
    # ---- duplicates
    ids = collections.Counter(r["id"] for r in ok)
    bodies = collections.Counter(r["body"] for r in spec)
    res["duplicates"] = {"id_in_two_files": sum(1 for v in ids.values() if v > 1),
                         "spectator_identical_bodies_extra_copies": sum(v - 1 for v in bodies.values() if v > 1)}
    # ---- accounts and bots (spectator)
    per = accounts(spec)
    flags = bot_flags(per)
    games = sorted((a_["games"] for a_ in per.values()), reverse=True)
    sides = 2 * len(spec)
    res["accounts"] = {"n": len(per), "sides": sides, "top1_share": games[0] / sides,
                       "top10_share": sum(games[:10]) / sides, "top100_share": sum(games[:100]) / sides,
                       "games_per_account": pct(games, (50, 90, 99)),
                       "flagged": {k: {"name": per[k]["name"], "why": v, "games": per[k]["games"],
                                       "max_per_day": max(per[k]["days"].values()) if per[k]["days"] else 0,
                                       "max_hours": max((len(x) for x in per[k]["hours"].values()), default=0),
                                       "mean_rating": float(np.mean(per[k]["rat"])) if per[k]["rat"] else None,
                                       "top_team_share": (per[k]["teams"].most_common(1)[0][1] / per[k]["games"])
                                       if per[k]["teams"] else None}
                                   for k, v in sorted(flags.items(), key=lambda kv: -per[kv[0]]["games"])},
                       "top25": [{"name": a_["name"], "games": a_["games"],
                                  "max_per_day": max(a_["days"].values()) if a_["days"] else 0,
                                  "mean_rating": float(np.mean(a_["rat"])) if a_["rat"] else None,
                                  "top_team_share": (a_["teams"].most_common(1)[0][1] / a_["games"])
                                  if a_["teams"] else None}
                                 for a_ in sorted(per.values(), key=lambda x: -x["games"])[:25]]}
    flagged = set(flags)
    # ---- side-level table by stratum (spectator, rated)
    teams = Teams()
    side_rows = []
    for r in spec:
        fast = (r["t0"] is not None and r["turns"] >= 10 and (r["t1"] - r["t0"]) / r["turns"] < FAST_S_PER_TURN)
        for s, o in (("1", "2"), ("2", "1")):
            rr = r.get("r" + s) if r["rated"] else None
            side_rows.append({"st": stratum(rr), "date": r["date"], "end": r["end"], "turns": r["turns"],
                              "won": r["win"] == "p" + s, "quit": r["loser_quit"] == "p" + s,
                              "bot": toid(r.get("n" + s) or "") in flagged,
                              "opp_bot": toid(r.get("n" + o) or "") in flagged,
                              "fast": fast, "nseen": len(r["o" + s]), "m": r["m" + s], "o": r["o" + s]})
    by = collections.defaultdict(list)
    for x in side_rows:
        by[x["st"]].append(x)
    card = {}
    for st, xs in sorted(by.items()):
        n = len(xs)
        card[st] = {
            "sides": n, "share_of_sides": n / len(side_rows),
            "turns": pct([x["turns"] for x in xs]),
            "end_forfeit": float(np.mean([x["end"] == "forfeit" for x in xs])),
            "end_timer": float(np.mean([x["end"] == "timer" for x in xs])),
            "end_ko": float(np.mean([x["end"] == "ko" for x in xs])),
            "early_walkover": float(np.mean([x["end"] in ("forfeit", "timer") and x["turns"] <= EARLY_FORFEIT_TURNS
                                             for x in xs])),
            "this_side_quit": float(np.mean([x["quit"] for x in xs])),
            "bot_side": float(np.mean([x["bot"] for x in xs])),
            "machine_speed_game": float(np.mean([x["fast"] for x in xs])),
            "full_team_seen": float(np.mean([x["nseen"] == 6 for x in xs])),
            "moves_seen_per_mon": float(np.mean([len(v[0]) for x in xs for v in x["m"].values()]))
            if any(x["m"] for x in xs) else None,
            "item_seen_per_mon": float(np.mean([v[1] is not None for x in xs for v in x["m"].values()]))
            if any(x["m"] for x in xs) else None,
        }
    res["by_stratum"] = card
    # ---- team distance: full-6 sides at >= 1500 (train dates), vs the pool and each panel
    hi = [x for x in side_rows if x["st"] in ("1500-1599", "1600-1699", "1700+") and x["nseen"] == 6
          and x["date"] <= TRAIN_LAST and not x["bot"]]
    allr = [x for x in side_rows if x["st"] != "unrated" and x["nseen"] == 6 and x["date"] <= TRAIN_LAST
            and not x["bot"]]
    res["team_distance"] = team_distance(teams, hi, allr)
    res["drift"] = drift(teams, side_rows, ok)
    # ---- labels (X50 extraction), the HOM by stratum
    res["labels_x50_train"] = x50_rows()
    by_path = {f"{r['date']}/battle-gen3ou-{r['id'].split('-')[-1]}.log": r for r in spec if r["date"] > TRAIN_LAST}
    res["hom_join"] = hom_join(by_path, teams)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("files", "format", "duplicates")}, indent=1, default=float))


def drift(teams: Teams, side_rows: List[dict], ok: List[dict]) -> Dict[str, Any]:
    """Meta / population / rating drift by calendar month (spectator), the archive beside it as its own population."""
    def marg(ts: List[Iterable[str]]) -> Dict[str, float]:
        c = collections.Counter(teams.canon(x) for t in ts for x in t)
        tot = sum(c.values())
        return {k: v / tot for k, v in c.items()}

    def tvd(p: Dict[str, float], q: Dict[str, float]) -> float:
        return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))
    refs = {n: marg([t.species6 for t in ts]) for n, ts in teams.ref.items()}
    months: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, List[dict]] = collections.defaultdict(list)
    for x in side_rows:
        if x["st"] != "unrated" and x["nseen"] == 6:
            groups["spec " + x["date"][:7]].append(x)
    for r in ok:
        if r.get("pop") == "arch" and r["rated"]:
            for s in ("1", "2"):
                if len(r["o" + s]) == 6 and r.get("r" + s):
                    groups["arch " + r["date"][:7]].append({"o": r["o" + s], "r": r["r" + s], "end": r["end"]})
    ratings: Dict[str, List[int]] = collections.defaultdict(list)
    for r in ok:
        if r["rated"]:
            for s in ("1", "2"):
                if r.get("r" + s):
                    ratings[r["pop"] + " " + r["date"][:7]].append(r["r" + s])
    prev = None
    for g in sorted(groups, key=lambda k: (k[:4] != "spec", k[5:])):
        xs = groups[g]
        if len(xs) < 500:
            continue
        m = marg([x["o"] for x in xs])
        rec = {"full6_sides": len(xs), "rating_mean_all_sides": float(np.mean(ratings[g])) if ratings[g] else None,
               "rating_share_ge1500": float(np.mean(np.asarray(ratings[g]) >= 1500)) if ratings[g] else None,
               "tvd_vs": {n: tvd(m, q) for n, q in refs.items()}}
        if prev is not None and g[:4] == prev[0][:4]:
            rec["tvd_vs_previous_month"] = tvd(m, prev[1])
        months[g] = rec
        prev = (g, m)
    # the noise floor of a month-vs-month TVD at this size: two random halves of one month
    rng = np.random.default_rng(7)
    big = max((g for g in groups if g.startswith("spec")), key=lambda g: len(groups[g]))
    ix = rng.permutation(len(groups[big]))
    h = len(ix) // 2
    floor = tvd(marg([groups[big][i]["o"] for i in ix[:h]]), marg([groups[big][i]["o"] for i in ix[h:]]))
    return {"by_month": months, "split_half_floor": {"month": big, "tvd": floor, "half_sides": h}}


def team_distance(teams: Teams, hi: List[dict], allr: List[dict]) -> Dict[str, Any]:
    """How far human teams sit from our pool, and whether they cover the unfamiliar panels (design §2.6)."""
    out: Dict[str, Any] = {"hi_sides_full6": len(hi), "rated_sides_full6": len(allr),
                           "hi_distinct_species6": len({frozenset(teams.canon(s) for s in x["o"]) for x in hi}),
                           "rated_distinct_species6": len({frozenset(teams.canon(s) for s in x["o"]) for x in allr})}
    canon = lambda o: frozenset(teams.canon(s) for s in o)                   # noqa: E731
    # species marginals (share of teams holding the species)
    def marg(team_species: List[frozenset]) -> Dict[str, float]:
        c = collections.Counter(s for t in team_species for s in t)
        return {k: v / len(team_species) for k, v in c.items()}
    H = marg([canon(x["o"]) for x in hi])
    refs = {name: marg([t.species6 for t in ts]) for name, ts in teams.ref.items()}
    keys = set(H) | set().union(*refs.values())

    def tvd(p, q):
        sp, sq = sum(p.values()), sum(q.values())
        return 0.5 * sum(abs(p.get(k, 0) / sp - q.get(k, 0) / sq) for k in keys)
    out["species_tvd_vs_humans_ge1500"] = {name: tvd(H, q) for name, q in refs.items()}
    out["species_tvd_pool_vs"] = {name: tvd(refs["pool"], q) for name, q in refs.items() if name != "pool"}
    out["human_species_absent_from_pool_mass"] = sum(v for k, v in H.items() if k not in refs["pool"]) / sum(H.values())
    rare = {k for k, v in refs["pool"].items() if v < 0.01}
    out["human_species_rare_in_pool_mass(<1% of pool teams)"] = sum(v for k, v in H.items()
                                                                    if k in rare or k not in refs["pool"]) / sum(H.values())
    # familiarity of human teams vs the pool, and of the panels' teams vs the pool (the same yardstick)
    def fam_hist(sides_m: List[dict]) -> Dict[str, Any]:
        sp, ft = collections.Counter(), collections.Counter()
        for m in sides_m:
            a, b, _ = teams.side(m, "pool")
            sp[a] += 1
            ft[b] += 1
        n = len(sides_m)
        return {"species_overlap": {k: sp[k] / n for k in range(7)}, "set_fits": {k: ft[k] / n for k in range(7)}}
    out["humans_ge1500_vs_pool"] = fam_hist([x["m"] for x in hi])
    for name in ("home72", "heldout72_hpfix", "forum72", "p72"):
        ms = [{m.species: [list(m.moves), None, None] for m in t.mons} for t in teams.ref[name]]   # the registry's rule
        out[f"{name}_vs_pool"] = fam_hist(ms)
    # coverage: per panel team, human >= 1500 sides (train dates) with >= 4 fitting sets / >= 5 shared species
    cov: Dict[str, Any] = {}
    for name in ("home72", "heldout72_hpfix", "forum72", "p72"):
        n_t = len(teams.ref[name])
        fit4 = np.zeros(n_t, np.int64)
        sp5 = np.zeros(n_t, np.int64)
        for x in hi:
            sp, ft = teams.overlaps(x["m"], name, min_sp=4)
            sp5 += sp >= 5
            fit4 += ft >= 4
        rec = {"teams": n_t, "fit4_sides": pct(fit4, (10, 25, 50, 75, 90)),
               "share_teams_fit4_ge10": float(np.mean(fit4 >= 10)), "share_teams_fit4_zero": float(np.mean(fit4 == 0)),
               "sp5_sides": pct(sp5, (10, 25, 50, 75, 90)), "share_teams_sp5_zero": float(np.mean(sp5 == 0))}
        if name == "forum72":
            labs = [t.label for t in teams.ref[name]]
            st = {}
            for s in ("CORE", "NEAR", "FAR"):
                ix = [i for i, l in enumerate(labs) if teams.strata.get(Path(l).stem) == s]
                if ix:
                    st[s] = {"teams": len(ix), "fit4_median": float(np.median(fit4[ix])),
                             "sp5_median": float(np.median(sp5[ix])), "fit4_zero": float(np.mean(fit4[ix] == 0))}
            rec["by_stratum"] = st
        cov[name] = rec
    out["panel_coverage_by_humans_ge1500"] = cov
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--cache", required=True)
    e.add_argument("--workers", type=int, default=3)
    n = sub.add_parser("snapshot")
    n.add_argument("--population", choices=("spec", "arch"), required=True)
    n.add_argument("--id", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--until", default="9999-12-31")
    n.add_argument("--workers", type=int, default=3)
    z = sub.add_parser("analyze")
    z.add_argument("--cache", required=True)
    z.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"extract": cmd_extract, "snapshot": cmd_snapshot, "analyze": cmd_analyze}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
