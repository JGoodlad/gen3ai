r"""The Kakuna-family AUDIT of a human rowset (X59-S REVIEW E B2; PREREG Amendment 2; plan §3.4). CPU only.

    python -m main.human_data.audit universe              # accounts with >= 10 seat-games, the seeded sample
    python -m main.human_data.audit export                # the sample's logs (verified) + chat counts + the reference
    python -m main.human_data.audit kakuna  [--deadline]  # Kakuna's LL per seat-game (the Metamon interpreter)
    python -m main.human_data.audit ours    [--deadline]  # P*'s and the HOM's LL over the same seat-games' rows
    python -m main.human_data.audit controls-export       # Kakuna's own seats (X51) + Metamon-family seats, read-only
    python -m main.human_data.audit kakuna --controls     # the same Metamon pipeline over the controls
    python -m main.human_data.audit read    [--flag]      # the result JSON; --flag only once Amendment 2 is on main

Every step is resumable and writes under ``~/gen3ai_archive/x59s/kakuna_audit_v1/`` (never under ``data/`` or
``models/``); account names exist only in memory (the export's log text is the registered snapshot's own bytes, in
the archive). The registered rule (PREREG Amendment 2):

1. **Universe**: every account (``acct`` = :func:`rowset.key64` of the salted id) with >= :data:`MIN_GAMES` kept
   seat-games (distinct ``bkey``) in the rowset's TRAIN and VAL shards; the rest are UNAUDITED.
2. **Games**: per account, :data:`N_SAMPLE` seat-games drawn by :func:`sample_games`
   (``default_rng([SEED, acct % 2**32])`` over its sorted bkeys; pure).
3. **Scores** per seat-game: Kakuna's mean log-probability of the human's action over the decisions Metamon's own
   parser reconstructs (``kakuna_audit_metamon_side.py``), P\*'s and the HOM's over the rowset's LABELLED rows
   (softmax over the legal mask).
4. **Account score** ``S = LL_Kakuna - LL_P*`` (decision-weighted means over its scored seat-games; secondary
   ``S_HOM``); an account needs >= :data:`MIN_SCORED` scored seat-games, else UNSCORED.
5. **Reference** (:func:`reference_mode`): the audited accounts with >= 1 own chat line per sampled game on average
   (:func:`chat_lines`); fewer than :data:`MIN_REFERENCE` of them ⇒ every scored account. Decided at ``export``,
   before any score exists.
6. **Flag** (:func:`flag_accounts`): ``S > p99`` of the reference's ``S`` (``np.percentile``, linear).
7. **Drop** (:func:`drop_fires`): flagged TRAIN rows > 1.0 % of all TRAIN rows ⇒ every row of the flagged accounts
   leaves the next rowset. ``drop_list.json`` is written either way (``drop_accounts``: the flagged accounts when
   the rule fires, else empty; ``rule_fired``; ``flagged_accounts``), its sha256 in the result.
8. **Controls**: Kakuna's own seats in X51's P\* vs Kakuna captures, 10 games per pseudo-account; POWER iff their
   median mean-LL_Kakuna exceeds the reference's p99 of account mean-LL_Kakuna (:func:`power`).
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from main.human_data import rowset as RS
from main.human_data import textpass as T

SEED = 20261026
MIN_GAMES = 10
N_SAMPLE = 10
MIN_SCORED = 5
MIN_REFERENCE = 100
REF_CHAT_PER_GAME = 1.0
FLAG_PCT = 99
DROP_PCT = 1            # flagged TRAIN rows > DROP_PCT % of TRAIN rows (integer arithmetic: no rounding at the edge)
BOUNDARY_EPS = 1e-9     # an S within this of p99 is a BOUNDARY account: not flagged, reported (rule 8)
PSEUDO_GAMES = 10
THREADS = 2
SCHEMA = "gen3_kakuna_audit_v1"
RESULT_REL = "designs/research_state/measurements/x59s_matched_students_2026-10-25/results/kakuna_audit.json"
METAMON_SIDE_REL = "designs/research_state/measurements/x59s_matched_students_2026-10-25/kakuna_audit_metamon_side.py"
PREREG_REL = "designs/research_state/measurements/x59s_matched_students_2026-10-25/PREREG.md"
AMENDMENT = "## Amendment 2"
HOM_ZIP = Path.home() / "gen3ai_archive" / "x50_r1b" / "hom" / "model" / "hom_x50r1b.zip"
R1 = Path.home() / "gen3ai_archive" / "x59s" / "hr_spec_v1.r1"
OUT = Path.home() / "gen3ai_archive" / "x59s" / "kakuna_audit_v1"


def reads_root() -> Path:
    """The read trees beside the main checkout (``<main checkout>/../gen3ai-reads``), where the anchors' captures live."""
    from utils.git import get_main_repo_root

    return Path(get_main_repo_root()).parent / "gen3ai-reads"


#: control (a): Kakuna's own seat; (b): the Metamon family (descriptive). (read root, cell prefix, family)
CONTROL_CELLS = (("x51_2026-10-20", "pstar_", "Kakuna"),
                 ("hr_reads_2026-10-25", "sr_", "SmallRL"), ("hr_reads_2026-10-25", "lil_", "LargeIL"),
                 ("hr_reads_2026-10-25", "lrl_", "LargeRL"), ("mm_strongest_2026-10-25", "pm_", "PastaMittens"))
OUR_NAME = re.compile(r"^gen3ai", re.I)
#: the controls' parser regime, matched to the bulk of the human rows (Metamon's team predictor reads the rating and
#: the date): a captured game has no rating (``Unrated`` resolves to usage rank 1500), the rowset's 1300-1499 rows
#: (78 %) resolve to rank 0; the date sits in the Jul-Dec bin, as the TRAIN window's larger part does
CONTROL_RATING = 1400
CONTROL_DATE = "2026-09-20"
_UNRATED_PLAYER = re.compile(r"^(\|player\|p[12]\|[^|\n]+\|[^|\n]*\|)$", re.M)


class AuditError(RuntimeError):
    """An input that does not match what it claims, or a step run out of order: stop."""


# ------------------------------------------------------------------------------------------------ pure rules

def sample_games(acct: int, bkeys: Iterable[int], n: int = N_SAMPLE, seed: int = SEED) -> List[int]:
    """The account's seeded sample: ``n`` of its distinct bkeys (sorted ascending first), drawn without replacement
    by ``default_rng([seed, acct % 2**32])``; every bkey when it has <= n. A pure function of (acct, the set)."""
    bk = np.unique(np.asarray(list(bkeys), np.int64))
    if len(bk) <= n:
        return [int(x) for x in bk]
    rng = np.random.default_rng([seed, int(acct) % 2 ** 32])
    return sorted(int(x) for x in bk[rng.choice(len(bk), size=n, replace=False)])


def chat_lines(text: str, name: str) -> int:
    """The ``|c|`` (or ``|c:|<ts>|``) lines whose speaker is ``name``: the speaker with one leading rank character
    (any non-alphanumeric) stripped, then :func:`textpass.toid`, equals ``toid(name)``."""
    want = T.toid(name)
    if not want:
        return 0
    n = 0
    for ln in text.split("\n"):
        if ln.startswith("|c|"):
            parts = ln.split("|", 3)
            sp = parts[2] if len(parts) > 2 else ""
        elif ln.startswith("|c:|"):
            parts = ln.split("|", 4)
            sp = parts[3] if len(parts) > 3 else ""
        else:
            continue
        if sp and not sp[0].isalnum():
            sp = sp[1:]
        n += int(T.toid(sp) == want)
    return n


def reference_mode(chat_mean: Dict[int, float]) -> Tuple[str, List[int]]:
    """``("chat", members)`` when >= :data:`MIN_REFERENCE` audited accounts average >= 1 own chat line per sampled
    game, else ``("all_scored", [])`` (every scored account is the reference). From the counts alone."""
    q = sorted(a for a, m in chat_mean.items() if m >= REF_CHAT_PER_GAME)
    return ("chat", q) if len(q) >= MIN_REFERENCE else ("all_scored", [])


def account_score(games: Sequence[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Decision-weighted means over an account's SCORED seat-games (each a dict of sums and counts); None when it has
    fewer than :data:`MIN_SCORED`."""
    if len(games) < MIN_SCORED:
        return None
    nk = sum(g["k_n"] for g in games)
    npr = sum(g["p_n"] for g in games)
    k = sum(g["k_ll"] for g in games) / nk
    p = sum(g["p_ll"] for g in games) / npr
    h = sum(g["h_ll"] for g in games) / npr
    return {"ll_kakuna": k, "ll_pstar": p, "ll_hom": h, "S": k - p, "S_HOM": k - h,
            "kakuna_argmax": sum(g["k_hits"] for g in games) / nk, "kakuna_decisions": int(nk),
            "pstar_rows": int(npr), "scored_games": len(games)}


def flag_accounts(scores: Dict[int, float], reference: Sequence[int]) -> Tuple[float, List[int], List[int]]:
    """(p99 of the reference's S, the FLAGGED accounts (S > p99, strictly), the BOUNDARY accounts (|S - p99| <=
    :data:`BOUNDARY_EPS`; never flagged, reported))."""
    ref = np.asarray([scores[a] for a in reference], np.float64)
    if len(ref) == 0:
        raise AuditError("an empty reference")
    p99 = float(np.percentile(ref, FLAG_PCT))
    boundary = sorted(a for a, s in scores.items() if abs(s - p99) <= BOUNDARY_EPS)
    flagged = sorted(a for a, s in scores.items() if s > p99 and a not in set(boundary))
    return p99, flagged, boundary


def drop_fires(flagged_train_rows: int, train_rows: int) -> bool:
    """The drop rule: flagged TRAIN rows > :data:`DROP_PCT` % of TRAIN rows, in integers (exactly 1.0 % does not
    fire)."""
    return int(flagged_train_rows) * 100 > DROP_PCT * int(train_rows)


def power(pseudo_means: Sequence[float], ref_means: Sequence[float]) -> Dict[str, Any]:
    """Control (a): POWER iff the Kakuna pseudo-accounts' median mean-LL_Kakuna exceeds the reference's p99 of
    account mean-LL_Kakuna."""
    if not len(pseudo_means) or not len(ref_means):
        return {"verdict": "NO-POWER", "why": "no pseudo-accounts or no reference"}
    p99 = float(np.percentile(np.asarray(ref_means, np.float64), FLAG_PCT))
    med = float(np.median(np.asarray(pseudo_means, np.float64)))
    return {"verdict": "POWER" if med > p99 else "NO-POWER", "pseudo_median_ll_kakuna": med,
            "reference_p99_ll_kakuna": p99,
            "pseudo_above_p99": int(sum(m > p99 for m in pseudo_means)), "pseudo_accounts": len(pseudo_means)}


def has_amendment(prereg_text: str) -> bool:
    return any(ln.startswith(AMENDMENT) for ln in prereg_text.splitlines())


def amendment_on_main() -> bool:
    """Brief §9: the registered rule must be on origin/main before any flag is read."""
    from utils.git import get_main_repo_root

    root = str(get_main_repo_root())
    subprocess.run(["git", "-C", root, "fetch", "-q", "origin"], check=True, timeout=300)
    r = subprocess.run(["git", "-C", root, "show", f"origin/main:{PREREG_REL}"], capture_output=True, text=True,
                       timeout=120)
    return r.returncode == 0 and has_amendment(r.stdout)


# ------------------------------------------------------------------------------------------------ io helpers

def _write_json(p: Path, obj: Any) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, p)


def _sha(p: Path) -> str:
    return RS.S_sha(p)


def _shard_recs(man: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return [("train", r) for r in man["train_shards"]] + [("val", man["val"])]


def _guard(out: Path) -> None:
    RS._guard_out(out)


# ------------------------------------------------------------------------------------------------ 1-2 universe

def universe(r1: Path, out: Path) -> Dict[str, Any]:
    man, man_sha = RS.load_manifest(r1 / "rows")
    rows_per: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    games: Dict[int, Dict[int, int]] = collections.defaultdict(dict)
    split_of: Dict[int, str] = {}
    totals: collections.Counter = collections.Counter()
    for split, rec in _shard_recs(man):
        z = RS.load_shard(r1 / "rows", rec, ("acct", "bkey", "seat"))
        totals[f"{split}_rows"] += len(z["acct"])
        for a, b, s, n in zip(*[c.tolist() for c in _group(z["acct"], z["bkey"], z["seat"])]):
            rows_per[a][split] += n
            if split_of.setdefault(a, split) != split:
                raise AuditError(f"account {a} has rows in both splits")
            prev = games[a].setdefault(b, s)
            if prev != s:
                raise AuditError(f"account {a} sits in both seats of battle {b}")
    if totals["train_rows"] != man["train_rows"] or totals["val_rows"] != man["val_rows"]:
        raise AuditError("row totals differ from the rows manifest")
    accts = []
    for a in sorted(games):
        g = games[a]
        if len(g) < MIN_GAMES:
            continue
        pick = sample_games(a, g.keys())
        accts.append({"acct": a, "split": split_of[a], "games": len(g), "rows": sum(rows_per[a].values()),
                      "sample": [[b, g[b]] for b in pick]})
    aud = {x["acct"] for x in accts}
    unaud = {sp: sum(c[sp] for a, c in rows_per.items() if a not in aud) for sp in ("train", "val")}
    res = {"schema": SCHEMA + ":universe", "rows_manifest_sha256": man_sha, "rowset": man["rowset"], "seed": SEED,
           "min_games": MIN_GAMES, "n_sample": N_SAMPLE, "train_rows": man["train_rows"], "val_rows": man["val_rows"],
           "accounts_total": len(games), "accounts_audited": len(accts),
           "audited_by_split": dict(collections.Counter(x["split"] for x in accts)),
           "unaudited_rows": unaud,
           "unaudited_train_row_share": unaud["train"] / man["train_rows"],
           "sampled_seat_games": sum(len(x["sample"]) for x in accts), "accounts": accts}
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "universe.json", res)
    return res


def _group(acct: np.ndarray, bkey: np.ndarray, seat: np.ndarray):
    """(acct, bkey, seat, rows) per distinct (acct, bkey, seat)."""
    k = np.stack([acct.astype(np.int64), bkey.astype(np.int64), seat.astype(np.int64)], 1)
    u, n = np.unique(k, axis=0, return_counts=True)
    return u[:, 0], u[:, 1], u[:, 2], n


def load_universe(out: Path) -> Dict[str, Any]:
    p = out / "universe.json"
    if not p.exists():
        raise AuditError("run `universe` first")
    return json.loads(p.read_text())


# ------------------------------------------------------------------------------------------------ export

def _keep_index(r1: Path) -> Dict[int, Dict[str, Any]]:
    scan = json.loads((r1 / "scan.json").read_text())
    if _sha(r1 / "keep.jsonl.gz") != scan["keep_sha256"]:
        raise AuditError("keep.jsonl.gz differs from the scan's record")
    out = {}
    with gzip.open(r1 / "keep.jsonl.gz", "rt") as fh:
        for ln in fh:
            k = json.loads(ln)
            out[RS.key64(k["rel"])] = {f: k[f] for f in ("rel", "sha", "n", "seats", "acct")}
    return out


def export(r1: Path, out: Path, store: Path, registry: Path, corpus: Path) -> Dict[str, Any]:
    """The sample's logs, re-verified against the snapshot manifest row (``snapshots.read_verified``), as the Metamon
    side's input; per seat-game the account's own chat lines (names in memory only); the reference decision."""
    from main.human_data import snapshots as S

    uni = load_universe(out)
    scan = json.loads((r1 / "scan.json").read_text())
    snap = scan["snapshot"]
    population = S.registry_row(snap, registry)["population"]
    keep = _keep_index(r1)
    salt = T.load_salt()
    by_battle: Dict[int, Dict[str, int]] = collections.defaultdict(dict)       # bkey -> {seat name: acct}
    for x in uni["accounts"]:
        for b, s in x["sample"]:
            by_battle[b]["p1" if s == 0 else "p2"] = x["acct"]
    chat: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    n_out = 0
    tmp = out / "export.jsonl.tmp"
    with open(tmp, "w") as fh:
        for b in sorted(by_battle):
            k = keep.get(b)
            if k is None:
                raise AuditError(f"battle {b} is not in keep.jsonl.gz")
            text = S.read_verified(snap, population, corpus, (k["rel"], k["sha"], k["n"])).decode(
                "utf-8", errors="replace")
            rec = T.record(text)
            if rec is None:
                raise AuditError(f"{k['rel']}: no player line")
            for seat, a in sorted(by_battle[b].items()):
                if seat not in k["seats"] or RS.key64(T.acct(rec["names"][seat], salt)) != a:
                    raise AuditError(f"{k['rel']} {seat}: the log's player is not account {a}")
                chat[str(a)][str(b)] = chat_lines(text, rec["names"][seat])
            gameid = k["rel"].rsplit("/", 1)[-1][len("battle-"):-len(".log")]
            fh.write(json.dumps({"key": str(b), "gameid": gameid, "date": k["rel"][:10], "text": text,
                                 "seats": sorted(by_battle[b])}, separators=(",", ":")) + "\n")
            n_out += 1
    os.replace(tmp, out / "export.jsonl")
    mean = {int(a): sum(v.values()) / N_SAMPLE for a, v in chat.items()}
    mode, members = reference_mode(mean)
    ref = {"schema": SCHEMA + ":reference", "decided_at": "export (before any score)", "mode": mode,
           "qualifying_accounts": sum(m >= REF_CHAT_PER_GAME for m in mean.values()),
           "rule": f">= {REF_CHAT_PER_GAME} own chat line per sampled game; < {MIN_REFERENCE} qualifying => every "
                   "scored account", "members": members}
    _write_json(out / "chat.json", {"per_seat_game": chat, "mean_per_game": {str(a): m for a, m in mean.items()}})
    _write_json(out / "reference.json", ref)
    return {"battles": n_out, "export_sha256": _sha(out / "export.jsonl"), "reference_mode": mode,
            "qualifying_accounts": ref["qualifying_accounts"]}


# ------------------------------------------------------------------------------------------------ controls

_HP = re.compile(r"\b\d+/\d+\b")


def _subject(ln: str) -> str:
    parts = ln.split("|")
    return parts[2][:2] if len(parts) > 2 and parts[2][:2] in ("p1", "p2") else ""


def capture_spectator_log(cap: Dict[str, Any]) -> str:
    """A spectator-equivalent log of a captured battle from the two players' streams (``|request|`` / ``|error|``
    chunks and ``|t:|`` lines dropped), aligned line by line on their HP-normalised text (``difflib``):

    * a line both views show: the view of the player who is NOT its subject (the public, percentage HP a spectator
      sees; a side's own view carries its exact HP);
    * a line only ONE view shows whose subject is that view's own side (a private reveal, e.g. a silent ability):
      dropped, as a spectator never sees it;
    * anything else (a one-sided line about the other side, or with no side subject): the capture is refused.
    """
    import difflib

    drop = ("|request|", "|error|")

    def stream(seat: str) -> List[str]:
        return [ln for c in cap["chunks"] if c[0] == seat and not c[1].startswith(drop)
                for ln in c[1].split("\n") if not ln.startswith("|t:|")]

    a, b = stream("p1"), stream("p2")
    na, nb = [_HP.sub("#", x) for x in a], [_HP.sub("#", x) for x in b]
    lines = ["|init|battle"]
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, na, nb, autojunk=False).get_opcodes():
        if op == "equal":
            for p, q in zip(a[i1:i2], b[j1:j2]):
                lines.append(q if _subject(p) == "p1" else p)
            continue
        for ln in a[i1:i2]:
            if _subject(ln) != "p1":
                raise AuditError(f"{cap.get('tag')}: a line only p1 sees is not about p1: {ln[:80]!r}")
        for ln in b[j1:j2]:
            if _subject(ln) != "p2":
                raise AuditError(f"{cap.get('tag')}: a line only p2 sees is not about p2: {ln[:80]!r}")
    return "\n".join(lines) + "\n"


def controls_export(out: Path, reads: Optional[Path] = None) -> Dict[str, Any]:
    """Every completed (``summary.json`` status OK) control cell's captures → the Metamon side's input; read-only."""
    reads = reads if reads is not None else reads_root()
    meta: Dict[str, Dict[str, Any]] = {}
    counts: collections.Counter = collections.Counter()
    tmp = out / "controls_export.jsonl.tmp"
    with open(tmp, "w") as fh:
        for root, prefix, family in CONTROL_CELLS:
            cdir = reads / root / "cells"
            if not cdir.is_dir():
                counts[f"missing_root:{root}"] += 1
                continue
            for cell in sorted(p for p in cdir.iterdir() if p.name.startswith(prefix)
                               and re.fullmatch(re.escape(prefix) + r"[a-z0-9]+_\d+", p.name)):
                sm = cell / "summary.json"
                if not sm.exists() or json.loads(sm.read_text()).get("status") != "OK":
                    counts[f"cell_skipped_incomplete:{family}"] += 1
                    continue
                for cp in sorted((cell / "capture").glob("battle-*.json"),
                                 key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
                    cap = json.loads(cp.read_text())
                    try:
                        text = capture_spectator_log(cap)
                    except AuditError:
                        counts[f"capture_unpaired:{family}"] += 1
                        continue
                    names = {m.group(1): m.group(2) for m in re.finditer(r"^\|player\|(p[12])\|([^|]+)", text, re.M)}
                    peer = [s for s, n in names.items() if not OUR_NAME.match(n)]
                    if len(peer) != 1 or "|win|" not in text and "|tie" not in text:
                        counts[f"capture_no_single_peer_or_result:{family}"] += 1
                        continue
                    text = _UNRATED_PLAYER.sub(lambda m: m.group(1) + str(CONTROL_RATING), text)
                    key = f"ctl:{root}:{cell.name}:{cp.stem}"
                    meta[key] = {"family": family, "cell": cell.name, "root": root, "seat": peer[0]}
                    fh.write(json.dumps({"key": key, "gameid": "gen3ou-" + re.sub(r"\D", "", cp.stem),
                                         "date": CONTROL_DATE, "text": text, "seats": peer},
                                        separators=(",", ":")) + "\n")
                    counts[f"captures:{family}"] += 1
    os.replace(tmp, out / "controls_export.jsonl")
    _write_json(out / "controls_meta.json", {"counts": dict(counts), "games": meta})
    return dict(counts)


# ------------------------------------------------------------------------------------------------ kakuna (subprocess)

def metamon_env(cfg: Dict[str, Any]) -> Dict[str, str]:
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "", "PYTHONHASHSEED": "0",
                "METAMON_CACHE_DIR": str(cfg["cache_dir"]), "KAUDIT_THREADS": str(THREADS),
                "TORCH_NUM_THREADS": str(THREADS)})
    return env


def kakuna(out: Path, controls: bool, deadline: Optional[float], limit: Optional[int], check_cache: int) -> int:
    from utils.paths import repo_path

    cfg = json.loads(repo_path("designs", "ops", "anchors.json").read_text())["opponents"]["metamon"]
    inp = out / ("controls_export.jsonl" if controls else "export.jsonl")
    if not inp.exists():
        raise AuditError(f"run {'controls-export' if controls else 'export'} first")
    dst = out / ("controls_kakuna.jsonl" if controls else "kakuna.jsonl")
    argv = [cfg["python"], "-u", str(repo_path(*METAMON_SIDE_REL.split("/"))), "--in", str(inp), "--out", str(dst),
            "--ident-out", str(out / ("controls_kakuna_ident.json" if controls else "kakuna_ident.json"))]
    if deadline is not None:
        argv += ["--deadline", repr(deadline)]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if check_cache:
        argv += ["--check-cache", str(check_cache)]
    return subprocess.call(argv, cwd=cfg["checkout"], env=metamon_env(cfg))


# ------------------------------------------------------------------------------------------------ ours (P*, HOM)

def ll_rows(logits: np.ndarray, mask: np.ndarray, act: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per row: log-probability of ``act`` under the softmax over the legal ``mask``, and top-1 == act."""
    from main.human_data.teacher import masked_probs

    p = masked_probs(logits, mask)
    idx = np.arange(len(act))
    return np.log(np.maximum(p[idx, act], 1e-300)), (np.where(mask.astype(bool), logits, -np.inf).argmax(1) == act)


def ours(r1: Path, out: Path, hom: Path, deadline: Optional[float]) -> int:
    """Per shard (resumable): for every sampled seat-game's LABELLED rows, P*'s (from the S-P label column's stored
    logits) and the HOM's (re-run, 2 threads) log-likelihood sums."""
    from main.human_data.teacher import _logits, _load
    from main.policy_spectrum.reader import inference_globals

    uni = load_universe(out)
    man, man_sha = RS.load_manifest(r1 / "rows")
    if man_sha != uni["rows_manifest_sha256"]:
        raise AuditError("the rows manifest moved since `universe`")
    lman = json.loads((r1 / "labels_pstar" / "labels_manifest.json").read_text())
    if lman["rows_manifest_sha256"] != man_sha:
        raise AuditError("the P* labels were made from another rowset")
    lrec = {r["name"]: r for r in lman["shards"]}
    want = {(b, s) for x in uni["accounts"] for b, s in x["sample"]}
    want_b = np.asarray(sorted({b for b, _ in want}), np.int64)
    odir = out / "ours"
    odir.mkdir(exist_ok=True)
    hom_sha = _sha(hom)
    with inference_globals(THREADS):
        model = None
        for _, rec in _shard_recs(man):
            dst = odir / f"{rec['name']}.json"
            if dst.exists():
                continue
            if deadline is not None and time.time() >= deadline:
                print(f"[audit] deadline reached before {rec['name']}", flush=True)
                return 3
            t0 = time.time()
            z = RS.load_shard(r1 / "rows", rec, ("bkey", "seat", "act", "mask", "rid"))
            lp = r1 / "labels_pstar" / f"{rec['name']}.npz"
            if lrec[rec["name"]]["rows_sha256"] != rec["sha256"] or _sha(lp) != lrec[rec["name"]]["sha256"]:
                raise AuditError(f"{lp}: not the label shard of this rows shard")
            lz = np.load(lp)
            if not np.array_equal(lz["rid"], z["rid"]):
                raise AuditError(f"{lp}: row ids differ from the rows shard")
            sel = np.isin(z["bkey"], want_b) & (z["act"] >= 0)
            sel &= np.asarray([(int(b), int(s)) in want for b, s in zip(z["bkey"], z["seat"])]) if sel.any() else sel
            idx = np.nonzero(sel)[0]
            res: Dict[str, Dict[str, float]] = {}
            if len(idx):
                if model is None:
                    model = _load(hom, THREADS)
                rows = np.load(r1 / "rows" / f"{rec['name']}.npz")["rows"][idx]
                act = z["act"][idx].astype(np.int64)
                mask = z["mask"][idx]
                llp, t1p = ll_rows(lz["logits"][idx].astype(np.float64), mask, act)
                llh, t1h = ll_rows(_logits(model, rows, mask).astype(np.float64), mask, act)
                for j, i in enumerate(idx.tolist()):
                    k = f"{int(z['bkey'][i])}:{int(z['seat'][i])}"
                    g = res.setdefault(k, {"p_n": 0, "p_ll": 0.0, "h_ll": 0.0, "p_top1": 0, "h_top1": 0})
                    g["p_n"] += 1
                    g["p_ll"] += float(llp[j])
                    g["h_ll"] += float(llh[j])
                    g["p_top1"] += int(t1p[j])
                    g["h_top1"] += int(t1h[j])
            _write_json(dst, {"shard": rec["name"], "rows_sha256": rec["sha256"], "hom_sha256": hom_sha,
                              "rows_scored": int(len(idx)), "seat_games": res})
            print(f"[audit] ours {rec['name']}: {len(idx)} rows, {len(res)} seat-games ({time.time() - t0:.0f} s)",
                  flush=True)
    return 0


def load_ours(out: Path, man: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    res: Dict[str, Dict[str, float]] = {}
    for _, rec in _shard_recs(man):
        p = out / "ours" / f"{rec['name']}.json"
        if not p.exists():
            raise AuditError(f"`ours` has not finished: {p.name} is missing")
        d = json.loads(p.read_text())
        if d["rows_sha256"] != rec["sha256"]:
            raise AuditError(f"{p}: made from another rows shard")
        for k, g in d["seat_games"].items():               # a seat-game's rows may straddle a shard boundary
            if k in res:
                for f, v in g.items():
                    res[k][f] += v
            else:
                res[k] = dict(g)
    return res


def load_kakuna(p: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if p.exists():
        with open(p) as fh:
            for ln in fh:
                if ln.strip():
                    r = json.loads(ln)
                    out[(r["key"], r["seat"])] = r
    return out


# ------------------------------------------------------------------------------------------------ read

def _seat_name(s: int) -> str:
    return "p1" if s == 0 else "p2"


def _q(xs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(xs, np.float64)
    if not len(a):
        return {}
    return {"n": int(len(a)), "mean": float(a.mean()), "p01": float(np.percentile(a, 1)),
            "p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)), "min": float(a.min()),
            "max": float(a.max())}


def alignment(k: Dict[str, Any], rows_turns: Optional[List[int]] = None) -> Dict[str, int]:
    """Kakuna-side decision counts of one seat-game (the diagnostic: what Metamon's parser left out)."""
    d = k.get("decisions") or []
    return {"metamon_decisions": len(d), "scored": sum(x[3] for x in d), "missing": sum(x[2] == "x" for x in d)}


def controls_read(out: Path, ref_ll: Sequence[float]) -> Dict[str, Any]:
    p = out / "controls_meta.json"
    if not p.exists():
        return {"status": "NOT RUN"}
    meta = json.loads(p.read_text())
    ks = load_kakuna(out / "controls_kakuna.jsonl")
    fam: Dict[str, List[str]] = collections.defaultdict(list)
    for key in sorted(meta["games"], key=lambda k: (meta["games"][k]["root"], meta["games"][k]["cell"],
                                                    int(re.sub(r"\D", "", k.rsplit(":", 1)[-1]) or 0))):
        fam[meta["games"][key]["family"]].append(key)
    res: Dict[str, Any] = {"export_counts": meta["counts"]}
    for f, keys in fam.items():
        fails: collections.Counter = collections.Counter()
        ok = []
        for key in keys:
            r = ks.get((key, meta["games"][key]["seat"]))
            if r is None:
                fails["not_scored_yet"] += 1
            elif not r["ok"]:
                fails[f"{r['stage']}:{r['error']}"] += 1
            elif r["n_dec"] == 0:
                fails["no_decisions"] += 1
            else:
                ok.append(r)
        pseudo = []
        for i in range(0, len(ok) - PSEUDO_GAMES + 1, PSEUDO_GAMES):
            g = ok[i:i + PSEUDO_GAMES]
            n = sum(x["n_dec"] for x in g)
            pseudo.append({"ll_kakuna": sum(x["ll_sum"] for x in g) / n,
                           "ll_kakuna_raw": sum(x["ll_raw_sum"] for x in g) / n,
                           "argmax": sum(x["argmax_hits"] for x in g) / n, "decisions": n})
        res[f] = {"games": len(keys), "scored_games": len(ok), "failures": dict(fails),
                  "pseudo_accounts": pseudo, "pseudo_ll_kakuna": _q([x["ll_kakuna"] for x in pseudo]),
                  "pseudo_argmax": _q([x["argmax"] for x in pseudo]),
                  "power": power([x["ll_kakuna"] for x in pseudo], ref_ll)}
    return res


def read(r1: Path, out: Path, do_flag: bool) -> Dict[str, Any]:
    uni = load_universe(out)
    man, man_sha = RS.load_manifest(r1 / "rows")
    if man_sha != uni["rows_manifest_sha256"]:
        raise AuditError("the rows manifest moved since `universe`")
    ref = json.loads((out / "reference.json").read_text())
    chat = json.loads((out / "chat.json").read_text())["mean_per_game"]
    ks = load_kakuna(out / "kakuna.jsonl")
    ou = load_ours(out, man)
    fails: collections.Counter = collections.Counter()
    acc: Dict[int, Dict[str, Any]] = {}
    games_all = cut = scored_games = 0
    for x in uni["accounts"]:
        a = x["acct"]
        g_ok = []
        for b, s in x["sample"]:
            games_all += 1
            k = ks.get((str(b), _seat_name(s)))
            o = ou.get(f"{b}:{s}")
            if k is None:
                fails["kakuna_not_run"] += 1
                continue
            if not k["ok"]:
                fails[f"kakuna_{k['stage']}:{k['error']}"] += 1
                continue
            cut += k["n_cut"]
            if k["n_dec"] == 0:
                fails["kakuna_no_decisions"] += 1
                continue
            if o is None or o["p_n"] == 0:
                fails["no_labelled_rows"] += 1
                continue
            g_ok.append({"k_n": k["n_dec"], "k_ll": k["ll_sum"], "k_hits": k["argmax_hits"], "p_n": o["p_n"],
                         "p_ll": o["p_ll"], "h_ll": o["h_ll"]})
        scored_games += len(g_ok)
        sc = account_score(g_ok)
        acc[a] = {"split": x["split"], "rows": x["rows"], "games": x["games"], "sampled": len(x["sample"]),
                  "chat_per_game": chat.get(str(a), 0.0), **(sc or {"scored_games": len(g_ok)}),
                  "scored": sc is not None}
    scored = {a: v for a, v in acc.items() if v["scored"]}
    if ref["mode"] == "chat":
        reference = [a for a in ref["members"] if a in scored]
    else:
        reference = sorted(scored)
    train_rows = man["train_rows"]
    unscored_train = sum(v["rows"] for v in acc.values() if not v["scored"] and v["split"] == "train")
    res: Dict[str, Any] = {
        "schema": SCHEMA, "rowset": man["rowset"], "rows_manifest_sha256": man_sha,
        "rule": "PREREG Amendment 2 (X59-S REVIEW E B2)",
        "universe": {k: uni[k] for k in ("accounts_total", "accounts_audited", "audited_by_split", "unaudited_rows",
                                         "unaudited_train_row_share", "sampled_seat_games", "train_rows",
                                         "val_rows")},
        "scoring": {"seat_games": games_all, "scored_seat_games": scored_games, "unscored_by_reason": dict(fails),
                    "kakuna_decisions_cut_past_max_steps": cut, "accounts_scored": len(scored),
                    "accounts_unscored": len(acc) - len(scored),
                    "unscored_train_row_share": unscored_train / train_rows},
        "reference": {"mode": ref["mode"], "qualifying_accounts": ref["qualifying_accounts"],
                      "size": len(reference)},
        "distributions": {f: _q([v[f] for v in scored.values()]) for f in
                          ("S", "S_HOM", "ll_kakuna", "ll_pstar", "ll_hom", "kakuna_argmax")},
        "reference_distributions": {f: _q([scored[a][f] for a in reference]) for f in ("S", "ll_kakuna")},
        "controls": controls_read(out, [scored[a]["ll_kakuna"] for a in reference]),
        "idents": {n: json.loads((out / n).read_text()) for n in ("kakuna_ident.json", "controls_kakuna_ident.json")
                   if (out / n).exists()},
    }
    flagged: List[int] = []
    if do_flag:
        if not amendment_on_main():
            raise AuditError("Amendment 2 is not on origin/main: the flag step refuses (brief §9)")
        p99, flagged, boundary = flag_accounts({a: v["S"] for a, v in scored.items()}, reference)
        ftrain = sum(acc[a]["rows"] for a in flagged if acc[a]["split"] == "train")
        fval = sum(acc[a]["rows"] for a in flagged if acc[a]["split"] == "val")
        fires = drop_fires(ftrain, train_rows)
        drop = {"schema": SCHEMA + ":drop_list", "rowset": man["rowset"], "rows_manifest_sha256": man_sha,
                "rule": f"flagged TRAIN rows > {DROP_PCT} % of r1 TRAIN rows", "rule_fired": fires,
                "flagged_train_rows": ftrain, "train_rows": train_rows,
                "drop_accounts": sorted(flagged) if fires else [], "flagged_accounts": sorted(flagged)}
        _write_json(out / "drop_list.json", drop)
        res["flag"] = {"reference_p99_S": p99, "flagged_accounts": len(flagged), "boundary_accounts": len(boundary),
                       "flagged_train_rows": ftrain, "flagged_val_rows": fval,
                       "flagged_train_row_share": ftrain / train_rows,
                       "drop_rule": f"flagged TRAIN rows > {DROP_PCT} % of r1 TRAIN rows",
                       "drop_fires": fires,
                       "drop_list": str(out / "drop_list.json"), "drop_list_sha256": _sha(out / "drop_list.json")}
        for a in flagged:
            acc[a]["flagged"] = True
    else:
        res["flag"] = {"status": "NOT READ (run `read --flag` once Amendment 2 is on origin/main)"}
    res["accounts"] = [{"acct": str(a), **{k: v for k, v in acc[a].items()}} for a in sorted(acc)]
    return res


# ------------------------------------------------------------------------------------------------ cli

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.audit", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("universe")
    e = sub.add_parser("export")
    for p in (e,):
        p.add_argument("--store", default=None)
        p.add_argument("--registry", default=None)
        p.add_argument("--corpus", default=None)
    sub.add_parser("controls-export")
    k = sub.add_parser("kakuna")
    k.add_argument("--controls", action="store_true")
    k.add_argument("--deadline", type=float, default=None)
    k.add_argument("--budget-s", type=float, default=None, help="deadline = process start + this (after a slot wait)")
    k.add_argument("--limit", type=int, default=None)
    k.add_argument("--check-cache", type=int, default=0)
    o = sub.add_parser("ours")
    o.add_argument("--deadline", type=float, default=None)
    o.add_argument("--budget-s", type=float, default=None, help="deadline = process start + this (after a slot wait)")
    o.add_argument("--hom", default=str(HOM_ZIP))
    r = sub.add_parser("read")
    r.add_argument("--flag", action="store_true", help="run the flag + drop step (refuses before Amendment 2)")
    r.add_argument("--result", default=None, help="default: the committed results/kakuna_audit.json")
    for p in sub.choices.values():
        p.add_argument("--r1", default=str(R1))
        p.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    r1, out = Path(a.r1), Path(a.out)
    if getattr(a, "budget_s", None) is not None:
        a.deadline = time.time() + a.budget_s if a.deadline is None else min(a.deadline, time.time() + a.budget_s)
    try:
        _guard(out)
        out.mkdir(parents=True, exist_ok=True)
        if a.cmd == "universe":
            res = universe(r1, out)
            print(json.dumps({k: v for k, v in res.items() if k != "accounts"}, indent=1), flush=True)
            return 0
        if a.cmd == "export":
            from main.human_data import snapshots as S

            res = export(r1, out, Path(a.store) if a.store else S.default_store(),
                         Path(a.registry) if a.registry else S.default_registry(),
                         Path(a.corpus) if a.corpus else S.default_corpus())
            print(json.dumps(res, indent=1), flush=True)
            return 0
        if a.cmd == "controls-export":
            print(json.dumps(controls_export(out), indent=1), flush=True)
            return 0
        if a.cmd == "kakuna":
            return kakuna(out, a.controls, a.deadline, a.limit, a.check_cache)
        if a.cmd == "ours":
            return ours(r1, out, Path(a.hom), a.deadline)
        from utils.paths import repo_path

        res = read(r1, out, a.flag)
        dst = Path(a.result) if a.result else repo_path(*RESULT_REL.split("/"))
        _write_json(dst, committed_view(res, out / "kakuna_audit_full.json"))
        print(json.dumps({k: v for k, v in res.items() if k not in ("accounts", "controls")}, indent=1), flush=True)
        return 0
    except (AuditError, RS.RowsetError) as exc:
        print(f"[audit] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 1


def committed_view(res: Dict[str, Any], full: Path) -> Dict[str, Any]:
    """The committed result: the full per-account table goes to the archive (``full``, its sha256 recorded); the
    commit keeps the flagged accounts' rows and each control's summary, not every pseudo-account."""
    _write_json(full, res)
    out = {k: v for k, v in res.items() if k not in ("accounts", "controls")}
    p99 = (res.get("flag") or {}).get("reference_p99_S")
    flagged = [x for x in res.get("accounts", []) if p99 is not None and x.get("S") is not None and x["S"] > p99]
    out["accounts"] = {"full_table": str(full).replace(str(Path.home()), "~"), "full_table_sha256": RS.S_sha(full),
                       "n": len(res.get("accounts", [])), "flagged": sorted(flagged, key=lambda x: -x["S"])}
    ctl = {}
    for k, v in (res.get("controls") or {}).items():
        if isinstance(v, dict) and "pseudo_accounts" in v:
            pa = v["pseudo_accounts"]
            ctl[k] = {**{kk: vv for kk, vv in v.items() if kk != "pseudo_accounts"},
                      "pseudo_median_argmax": float(np.median([x["argmax"] for x in pa])) if pa else None}
        else:
            ctl[k] = v
    out["controls"] = ctl
    return out


if __name__ == "__main__":
    sys.exit(main())
