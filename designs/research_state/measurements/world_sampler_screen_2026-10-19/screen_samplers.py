"""WORLD-SAMPLER SCREEN, stage 0 (ZERO playouts): how much weight does each candidate sampler put on the TRUE hidden team?

Design: ``designs/endstate/design_world_sampler.md`` (§2 the target metrics, §3 the candidates, §5.1 this stage).
POST HOC and DESCRIPTIVE: written and run on 2026-10-19 after every bank it reads had been read by its own study; it
reads only INPUTS of those studies (each state's revealed opponent info, the true team, the stored world rows' fact
records, the network's belief rows), never a playout outcome or an IMP.

Banks (read-only):
  * the maturity battery's INTERIM bank, its LEAN 320 ids only (the FULL remainder is being built by the running plateau
    step: not read), opponents = roster bots and X26's own snapshots, X26 at 230M-234M;
  * E2's two banks (end2, prod at 15M), opponents = the networks' own snapshots only.
Opponent teams in every bank are training-POOL teams (719), and 711 of the 719 pool teams appear verbatim in the forum
collection: the forum corpus used here is ``w_forum_v1`` (pool-exact copies already removed) and a STRICT variant that
also drops every corpus team sharing >= 4 mon sets (species + move set) with the state's TRUE team.

Samplers of the never-seen species set U* (h = 6 - k slots, k = the opponent mons seen):
  a    F's own rule (``world_draws.never_seen_species``): sequential, without replacement, each slot ∝ the MEAN over
       the seen species r of Smogon's teammate share t_r(s) (usage when that is all zero);
  b    Smogon only, the team hangs together: sequential, ∝ u(s) · Π_{r in seen ∪ drawn so far} lift_r(s), lift_r(s) =
       (t_r(s) + ALPHA·ū(s)) / ū(s) (naive Bayes over the one species×species joint Smogon publishes);
  c    a TEAM-LEVEL corpus model: the corpus teams CONSISTENT with what play revealed (every seen species, each with its
       revealed moves) give an empirical distribution over U*; mixed with b by λ = n / (n + 1) (n consistent teams);
       corpora: forum (``w_forum_v1`` ABC), forum_strict, ladder (``utils.ladder_corpus`` full, Metamon's hl_05_26);
  d    the network's own species belief (FN's rule: sequential ∝ the X5 species presence π; F's weights if π is empty),
       maturity LEAN only, at each anchor's own belief row.
Metrics per state (pooled means, battle-clustered bootstrap): hit = E[#true never-seen species drawn] / h (set-based,
the study's ``match``); p_exact = P(drawn set == U*), exact (sum over the h! orders); recall@h (the h species of
largest inclusion probability); for the team: q_SM = P(U* exact AND every seen mon's unrevealed moves exact), the
exact-world mass on the two classes the study found binding (U species, M), the unit of the dose curve's p.
The seen mons' moves under F are the STORED F worlds' own match rate (``facts[..].moves.match``), so F's numbers are
what F actually played; under c they are mixed by the same λ.

Run (light CPU, one process; root rule 5b):
  CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 python -I screen_samplers.py --out result_screen.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from agents.gen3_data import priors, species as SPECIES  # noqa: E402
from agents.info_levers import packed as PK  # noqa: E402

ARCH = Path.home() / "gen3ai_archive"
MAT = ARCH / "hidden_info_maturity"
E2 = ARCH / "hidden_info_e2"
MEAS = REPO / "designs" / "research_state" / "measurements"
MAT_STATES = MEAS / "hidden_info_maturity_x26_2026-10-18" / "states_mat_x26.json"
ANCHORS = ("x26_025M", "x26_100M", "x26_175M", "x26_225M")

ALPHA = 0.05            # b's smoothing toward usage (fixed before any run)
N_MC = 4000             # Monte Carlo draws per (state, sampler) for inclusion / hit / recall
NEAR = 4                # forum_strict: drop corpus teams sharing >= NEAR mon sets with the TRUE team
BOOT = 2000
BOOT_SEED = 20261019
P_STAR = (0.1, 0.2, 0.3)


def canon(m: str) -> str:
    mid = PK.move_id(m)
    return "hiddenpower" if PK.is_hp(mid) else mid


Team = Tuple[Tuple[str, FrozenSet[str]], ...]       # ((species, canon move set), ...) x 6


def parse_packed(team: str) -> Team:
    out = []
    for e in team.split("]"):
        if not e.strip():
            continue
        f = PK.parse_set(e)
        out.append((PK.species_id(f), frozenset(canon(m) for m in f[4].split(",") if m)))
    return tuple(out)


def parse_export(text: str) -> Team:
    """A Showdown export paste -> Team (species from '(Species)' or the name; '- Move' lines)."""
    out = []
    for block in [b for b in text.replace("\r", "").split("\n\n") if b.strip()]:
        lines = [x.strip() for x in block.split("\n") if x.strip()]
        head = lines[0].split("@")[0].strip()
        parts = [p.strip(" )") for p in head.split("(")]
        names = [p for p in parts[1:] if p not in ("M", "F", "")]
        sp = PK.toid(names[-1] if names else parts[0])
        mv = []
        for x in lines[1:]:
            if x.startswith("-"):
                m = x[1:].strip()
                mv.append("hiddenpower" if m.lower().startswith("hidden power") else canon(m))
        out.append((sp, frozenset(mv)))
    return tuple(out)


# ------------------------------------------------------------------------------------------------
# the Smogon prior (F's candidate pool and weights)
# ------------------------------------------------------------------------------------------------

class Smogon:
    def __init__(self) -> None:
        usage = dict(priors.species_usage())
        self.pool: List[str] = sorted(s for s in usage if priors.moves(s) and SPECIES.get(s) is not None)
        self.ix = {s: i for i, s in enumerate(self.pool)}
        u = np.array([usage[s] for s in self.pool], dtype=np.float64)
        self.ubar = u / u.sum()
        C = len(self.pool)
        self.T = np.zeros((C, C))               # T[r, s] = Smogon teammate share t_r(s)
        self.has_t = np.zeros(C, dtype=bool)
        for r in self.pool:
            t = priors.teammates(r)
            if t:
                self.has_t[self.ix[r]] = True
                for s, v in t.items():
                    if s in self.ix:
                        self.T[self.ix[r], self.ix[s]] = float(v)
        self.loglift = np.log((self.T + ALPHA * self.ubar[None, :]) / self.ubar[None, :])
        self.loglift[~self.has_t, :] = 0.0

    def f_weights(self, seen: Sequence[str]) -> np.ndarray:
        """F's ``weights()``: the mean teammate share over the seen species (usage when all zero)."""
        rs = sorted(seen)
        w = np.mean([self.T[self.ix[r]] if r in self.ix else np.zeros(len(self.pool)) for r in rs], axis=0) \
            if rs else np.zeros(len(self.pool))
        return w if w.sum() > 0 else self.ubar.copy()


# ------------------------------------------------------------------------------------------------
# sequential samplers: exact P(set) and Monte Carlo inclusion
# ------------------------------------------------------------------------------------------------

def p_set_fixed(w: np.ndarray, present: Sequence[int], target: Sequence[int]) -> float:
    """P(a sequential no-replacement draw ∝ fixed ``w`` returns exactly ``target``)."""
    base = w.copy()
    base[list(present)] = 0.0
    tot0 = base.sum()
    if tot0 <= 0:
        return 0.0
    p = 0.0
    for order in itertools.permutations(target):
        tot, q = tot0, 1.0
        for s in order:
            if base[s] <= 0 or tot <= 0:
                q = 0.0
                break
            q *= base[s] / tot
            tot -= base[s]
        p += q
    return p


def p_set_poe(S: Smogon, seen_ix: Sequence[int], target: Sequence[int]) -> float:
    lw0 = np.log(S.ubar) + S.loglift[list(seen_ix)].sum(axis=0)
    p = 0.0
    for order in itertools.permutations(target):
        lw = lw0.copy()
        mask = np.zeros(len(lw), dtype=bool)
        mask[list(seen_ix)] = True
        q = 1.0
        for s in order:
            ww = np.where(mask, 0.0, np.exp(lw - lw[~mask].max()))
            q *= ww[s] / ww.sum()
            mask[s] = True
            lw = lw + S.loglift[s]
        p += q
    return p


def mc_fixed(rng: np.random.Generator, w: np.ndarray, present: Sequence[int], h: int) -> np.ndarray:
    """Inclusion frequency of each candidate over N_MC Gumbel-top-h draws (= sequential ∝ w, no replacement)."""
    lw = np.log(np.where(w > 0, w, 0.0) + 1e-300)
    lw[list(present)] = -np.inf
    lw[w <= 0] = -np.inf
    g = lw[None, :] + rng.gumbel(size=(N_MC, len(w)))
    top = np.argpartition(-g, h - 1, axis=1)[:, :h]
    return np.bincount(top.ravel(), minlength=len(w)) / N_MC


def mc_poe(rng: np.random.Generator, S: Smogon, seen_ix: Sequence[int], h: int) -> np.ndarray:
    C = len(S.pool)
    lw = np.tile(np.log(S.ubar) + S.loglift[list(seen_ix)].sum(axis=0), (N_MC, 1))
    lw[:, list(seen_ix)] = -np.inf
    cnt = np.zeros(C)
    rows = np.arange(N_MC)
    for _ in range(h):
        pick = np.argmax(lw + rng.gumbel(size=lw.shape), axis=1)
        np.add.at(cnt, pick, 1.0)
        lw = lw + S.loglift[pick]
        lw[rows, pick] = -np.inf
    return cnt / N_MC


# ------------------------------------------------------------------------------------------------
# corpora
# ------------------------------------------------------------------------------------------------

class Corpus:
    def __init__(self, name: str, teams: List[Team]):
        self.name = name
        self.teams = [t for t in teams if len(t) == 6 and len({s for s, _ in t}) == 6]
        self.by_species: Dict[str, Set[int]] = defaultdict(set)
        self.moves: List[Dict[str, FrozenSet[str]]] = []
        #: per-species SET counts over every corpus team (a per-species set prior, team context ignored)
        self.sets: Dict[str, Counter] = defaultdict(Counter)
        for i, t in enumerate(self.teams):
            self.moves.append({s: m for s, m in t})
            for s, m in t:
                self.by_species[s].add(i)
                self.sets[s][m] += 1

    def consistent(self, revealed: Dict[str, Set[str]]) -> List[int]:
        ids: Optional[Set[int]] = None
        for r in revealed:
            got = self.by_species.get(r, set())
            ids = set(got) if ids is None else ids & got
            if not ids:
                return []
        return sorted(i for i in (ids or set()) if all(revealed[r] <= self.moves[i][r] for r in revealed))


def load_corpora() -> Dict[str, Corpus]:
    from utils.ladder_corpus import teams as ladder_teams
    from utils.team_corpora import load_corpus
    from utils.team_packing import ConstantTeambuilder

    forum = []
    for text in load_corpus("w_forum_v1", "ABC").texts:
        try:
            forum.append(parse_packed(ConstantTeambuilder(text).packed_team))
        except Exception:
            forum.append(parse_export(text))
    ladder = [parse_packed(t) for t in ladder_teams("full")]
    return {"forum": Corpus("forum", forum), "ladder": Corpus("ladder", ladder)}


# ------------------------------------------------------------------------------------------------
# states
# ------------------------------------------------------------------------------------------------

def _jsonl(path: Path) -> List[dict]:
    out = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                         # a torn last line of a file still being written
    return out


def load_states() -> List[dict]:
    """Every state with an OK F world row: id, population, battle, side, revealed, the true opponent team, F's facts."""
    lean = {k for k, v in json.loads(MAT_STATES.read_text())["states"].items() if v["lean"]}
    mat_meta = json.loads(MAT_STATES.read_text())["states"]
    pops = [("mat_lean", MAT / "bank", MAT / "worlds" / "shared" / "F" / "worlds_0_of_1.jsonl", lean),
            ("e2_end2", E2 / "end2" / "bank", E2 / "end2" / "worlds" / "F" / "worlds_0_of_1.jsonl", None),
            ("e2_prod", E2 / "prod" / "bank", E2 / "prod" / "worlds" / "F" / "worlds_0_of_1.jsonl", None)]
    out = []
    for pop, bank, fpath, keep in pops:
        battles = {b["battle_id"]: b for b in _jsonl(bank / "battles.jsonl.gz")}
        decs = {d["id"]: d for d in _jsonl(bank / "decisions.jsonl.gz")}
        seen_ids: Set[str] = set()
        for r in _jsonl(fpath):
            did = r["id"]
            if (keep is not None and did not in keep) or did in seen_ids or not r.get("ok"):
                continue
            seen_ids.add(did)
            d = decs[did]
            opp = "p2" if d["side"] == "p1" else "p1"
            stratum = (mat_meta[did]["stratum"] if pop == "mat_lean" else "snapshot")
            out.append({"id": did, "pop": pop, "battle": f"{pop}:{d['battle']}", "side": d["side"],
                        "stratum": stratum, "true": parse_packed(battles[d["battle"]][opp]["team"]),
                        "revealed": {s: {canon(m) for m in ms} for s, ms in r["revealed"].items()},
                        "facts": r["facts"]})
    return out


def f_empirical(st: dict) -> dict:
    """What F's 8 stored worlds actually held: the set-based never-seen species hit, and per seen mon with hidden
    slots the fraction of worlds whose unrevealed moves are ALL right."""
    hits, mv = [], defaultdict(list)
    for world in st["facts"]:
        ns = [x for x in world if not x.get("seen")]
        if ns:
            hits.append(np.mean([bool(x.get("species_fact", {}).get("match")) for x in ns]))
        for x in world:
            if x.get("seen") and x["moves"].get("slots", 0) > 0:
                mv[x["species"]].append(bool(x["moves"]["match"]))
    return {"hit": float(np.mean(hits)) if hits else None, "moves": {s: float(np.mean(v)) for s, v in mv.items()}}


def fn_rows(anchor: str) -> Dict[str, dict]:
    return {r["id"]: r for r in _jsonl(MAT / "fn_belief" / f"{anchor}.jsonl")}


def fn_empirical(anchor: str) -> Dict[str, dict]:
    """What the anchor's stored FN worlds held, per state: the set-based species hit and, per seen mon with hidden
    slots, the fraction of worlds whose unrevealed moves are ALL right (the network's move belief, CP-drawn)."""
    return {r["id"]: f_empirical(r) for r in _jsonl(MAT / "worlds" / anchor / "FN" / "worlds_0_of_1.jsonl")
            if r.get("ok")}


# ------------------------------------------------------------------------------------------------
# per-state evaluation
# ------------------------------------------------------------------------------------------------

def species_metrics(inc: np.ndarray, p_exact: float, target: Sequence[int], h: int) -> dict:
    top = np.argsort(-inc, kind="stable")[:h]
    return {"hit": float(inc[list(target)].sum() / h), "p_exact": float(p_exact),
            "logp": float(math.log(max(p_exact, 1e-12))), "recall": float(len(set(top) & set(target)) / h)}


def mon_key(t: Team) -> Set[Tuple[str, FrozenSet[str]]]:
    return set(t)


def evaluate(st: dict, S: Smogon, corpora: Dict[str, Corpus], fn: Dict[str, Dict[str, dict]],
             fn_emp: Dict[str, Dict[str, dict]]) -> dict:
    rng = np.random.default_rng(int(hashlib.sha256(f"wss_v1:{st['id']}".encode()).hexdigest()[:16], 16))
    true_sp = [s for s, _ in st["true"]]
    true_mv = dict(st["true"])
    seen = sorted(st["revealed"])
    bad = [s for s in seen if s not in true_mv]
    unseen = [s for s in true_sp if s not in st["revealed"]]
    h = len(unseen)
    out: dict = {"id": st["id"], "pop": st["pop"], "battle": st["battle"], "stratum": st["stratum"], "k": len(seen),
                 "h": h, "gigo_seen_not_true": bad, "unseen_out_of_pool": [s for s in unseen if s not in S.ix]}
    emp = f_empirical(st)
    out["F_emp_hit"] = emp["hit"]
    # the seen mons' unrevealed moves: which mons have hidden slots, and F's stored exact-completion rate
    hidden = {s: true_mv[s] for s in seen if len(true_mv[s] - st["revealed"][s]) > 0}
    pF_moves = {s: emp["moves"].get(s, 0.0) for s in hidden}
    out["n_seen_hidden"] = len(hidden)
    out["pM_F"] = float(np.prod(list(pF_moves.values()))) if hidden else 1.0
    out["pM_mon_F"] = float(np.mean(list(pF_moves.values()))) if hidden else float("nan")
    # a PER-SPECIES SET prior from each corpus (the sets of that species holding every revealed move), mixed with F's
    # own per-mon rate by λ = n / (n + 1): can set priors alone make the seen mons' moves exact?
    for cname in ("forum", "ladder"):
        C = corpora[cname]
        per = []
        for s in hidden:
            cnt = C.sets.get(s, Counter())
            n = sum(c for m, c in cnt.items() if st["revealed"][s] <= m)
            hit = cnt.get(true_mv[s], 0) if st["revealed"][s] <= true_mv[s] else 0
            lam = n / (n + 1.0)
            per.append(lam * (hit / n if n else 0.0) + (1 - lam) * pF_moves[s])
        out[f"pM_sets_{cname}"] = float(np.prod(per)) if hidden else 1.0
        out[f"pM_mon_sets_{cname}"] = float(np.mean(per)) if hidden else float("nan")
    seen_ix = [S.ix[s] for s in seen if s in S.ix]
    sam: Dict[str, dict] = {}
    if h > 0 and not out["unseen_out_of_pool"]:
        tgt = [S.ix[s] for s in unseen]
        wa = S.f_weights([s for s in seen if s in S.ix])
        sam["a"] = species_metrics(mc_fixed(rng, wa, seen_ix, h), p_set_fixed(wa, seen_ix, tgt), tgt, h)
        inc_b = mc_poe(rng, S, seen_ix, h)
        pb = p_set_poe(S, seen_ix, tgt)
        sam["b"] = species_metrics(inc_b, pb, tgt, h)
        for anchor, rows in fn.items():
            row = rows.get(st["id"])
            if row is None:
                continue
            pi = np.array([float(row["species"].get(s, 0.0)) for s in S.pool])
            w = pi if (np.delete(pi, seen_ix).sum() > 0) else wa
            sam[f"d@{anchor}"] = species_metrics(mc_fixed(rng, w, seen_ix, h), p_set_fixed(w, seen_ix, tgt), tgt, h)
            femp = fn_emp.get(anchor, {}).get(st["id"])
            if femp is not None:
                pM_fn = float(np.prod([femp["moves"].get(s, 0.0) for s in hidden])) if hidden else 1.0
                sam[f"d@{anchor}"]["pM"] = pM_fn
                sam[f"d@{anchor}"]["q_SM"] = sam[f"d@{anchor}"]["p_exact"] * pM_fn
    for cname, strict in (("forum", False), ("forum_strict", True), ("ladder", False)):
        C = corpora["ladder" if cname == "ladder" else "forum"]
        ids = C.consistent(st["revealed"])
        if strict:
            tk = mon_key(st["true"])
            ids = [i for i in ids if len(mon_key(C.teams[i]) & tk) < NEAR]
        n = len(ids)
        lam = n / (n + 1.0)
        unseen_set = frozenset(unseen)
        rec = {"n": n, "lambda": lam}
        # team-level fractions among the consistent corpus teams
        f_u = f_m = f_um = f_full = 0.0
        inc_c = np.zeros(len(S.pool))
        for i in ids:
            sp_i = frozenset(C.moves[i]) - set(seen)
            u_ok = sp_i == unseen_set
            m_ok = all(C.moves[i][s] == true_mv[s] for s in hidden)
            f_u += u_ok
            f_m += m_ok
            f_um += u_ok and m_ok
            f_full += u_ok and all(C.moves[i][s] == true_mv[s] for s in true_sp)
            for s in sp_i:
                if s in S.ix:
                    inc_c[S.ix[s]] += 1.0
        if n:
            f_u, f_m, f_um, f_full, inc_c = f_u / n, f_m / n, f_um / n, f_full / n, inc_c / n
        rec.update({"covers_U": bool(f_u > 0), "frac_U": f_u, "frac_M": f_m, "frac_UM": f_um, "frac_full": f_full})
        if "b" in sam:
            tgt = [S.ix[s] for s in unseen]
            inc = lam * inc_c + (1 - lam) * mc_poe(rng, S, seen_ix, h)
            pe = lam * f_u + (1 - lam) * sam["b"]["p_exact"]
            rec.update(species_metrics(inc, pe, tgt, h))
            rec["q_SM"] = float(lam * f_um + (1 - lam) * sam["b"]["p_exact"] * out["pM_F"])
        elif h == 0:
            rec["q_SM"] = float(lam * f_m + (1 - lam) * out["pM_F"])
        rec["pM"] = float(lam * f_m + (1 - lam) * out["pM_F"])
        sam[f"c_{cname}"] = rec
    if h == 0:
        out["q_SM_F"] = out["pM_F"]
    elif "a" in sam:
        out["q_SM_F"] = sam["a"]["p_exact"] * out["pM_F"]
        sam["b"]["q_SM"] = sam["b"]["p_exact"] * out["pM_F"]
    out["samplers"] = sam
    return out


# ------------------------------------------------------------------------------------------------
# aggregation: battle-clustered bootstrap
# ------------------------------------------------------------------------------------------------

def boot_mean(vals: np.ndarray, clusters: np.ndarray) -> dict:
    ok = ~np.isnan(vals)
    vals, clusters = vals[ok], clusters[ok]
    if len(vals) == 0:
        return {"n": 0}
    uc, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=vals)
    cnts = np.bincount(inv)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(uc), size=(BOOT, len(uc)))
    est = sums[draws].sum(axis=1) / cnts[draws].sum(axis=1)
    return {"n": int(len(vals)), "mean": float(vals.mean()), "lo": float(np.percentile(est, 2.5)),
            "hi": float(np.percentile(est, 97.5))}


def summarise(rows: List[dict], subset_name: str) -> dict:
    cl = np.array([r["battle"] for r in rows])
    names = sorted({k for r in rows for k in r["samplers"]})
    out: dict = {"subset": subset_name, "states": len(rows), "battles": int(len(set(cl))),
                 "k_hist": dict(sorted(Counter(r["k"] for r in rows).items())),
                 "F_emp_hit": boot_mean(np.array([r["F_emp_hit"] if r["F_emp_hit"] is not None else np.nan
                                                  for r in rows], float), cl),
                 "pM_F": boot_mean(np.array([r["pM_F"] if r["n_seen_hidden"] else np.nan for r in rows], float), cl),
                 "pM_mon_F": boot_mean(np.array([r["pM_mon_F"] for r in rows], float), cl),
                 **{f"{m}_{c}": boot_mean(np.array([r[f"{m}_{c}"] if r["n_seen_hidden"] else np.nan for r in rows],
                                                   float), cl)
                    for m in ("pM_sets", "pM_mon_sets") for c in ("forum", "ladder")},
                 # descriptive products (independence ASSUMED): the network's species set x a per-species set prior
                 **{f"q_hybrid_{anc}_{c}": boot_mean(np.array(
                     [r["samplers"][f"d@{anc}"]["p_exact"] * r[f"pM_sets_{c}"] if f"d@{anc}" in r["samplers"]
                      else np.nan for r in rows], float), cl) for anc in ("x26_225M",) for c in ("forum", "ladder")},
                 "q_SM_F": boot_mean(np.array([r.get("q_SM_F", np.nan) for r in rows], float), cl),
                 "samplers": {}}
    for nm in names:
        blk = {}
        for m in ("hit", "p_exact", "logp", "recall", "q_SM", "pM", "covers_U", "frac_U", "n"):
            v = np.array([float(r["samplers"].get(nm, {}).get(m, np.nan)) for r in rows], float)
            if np.all(np.isnan(v)):
                continue
            blk[m] = boot_mean(v, cl)
        for thr in P_STAR:
            v = np.array([float(r["samplers"][nm]["q_SM"] >= thr) if "q_SM" in r["samplers"].get(nm, {})
                          else np.nan for r in rows], float)
            if not np.all(np.isnan(v)):
                blk[f"share_q_ge_{thr}"] = boot_mean(v, cl)
        # paired difference vs F's own rule (a): the same states, the same bootstrap
        if nm != "a":
            for m in ("hit", "p_exact"):
                v = np.array([r["samplers"][nm][m] - r["samplers"]["a"][m]
                              if (m in r["samplers"].get(nm, {}) and "a" in r["samplers"]) else np.nan
                              for r in rows], float)
                if not np.all(np.isnan(v)):
                    blk[f"{m}_minus_a"] = boot_mean(v, cl)
        out["samplers"][nm] = blk
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "result_screen.json"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    S = Smogon()
    corpora = load_corpora()
    states = load_states()
    if a.limit:
        states = states[: a.limit]
    fn = {anc: fn_rows(anc) for anc in ANCHORS}
    fn_emp = {anc: fn_empirical(anc) for anc in ANCHORS}
    rows = [evaluate(st, S, corpora, fn, fn_emp) for st in states]
    # GIGO: the analytic F rule against what F's stored worlds actually held (same states)
    val = [(r["samplers"]["a"]["hit"], r["F_emp_hit"]) for r in rows if "a" in r["samplers"] and r["F_emp_hit"] is not None]
    val_fn = {anc: [(r["samplers"][f"d@{anc}"]["hit"], fn_emp[anc][r["id"]]["hit"]) for r in rows
                    if f"d@{anc}" in r["samplers"] and r["id"] in fn_emp[anc] and fn_emp[anc][r["id"]]["hit"] is not None]
              for anc in fn_emp}
    subsets = {"pooled": rows, "mat_lean": [r for r in rows if r["pop"] == "mat_lean"],
               "mat_lean_bot": [r for r in rows if r["pop"] == "mat_lean" and r["stratum"] == "bot"],
               "mat_lean_snapshot": [r for r in rows if r["pop"] == "mat_lean" and r["stratum"] == "snapshot"],
               "e2": [r for r in rows if r["pop"].startswith("e2")]}
    for kb in ("1-2", "3", "4", "5", "6"):
        ks = {"1-2": (1, 2), "3": (3,), "4": (4,), "5": (5,), "6": (6,)}[kb]
        subsets[f"pooled_k{kb}"] = [r for r in rows if r["k"] in ks]
    res = {
        "schema": "world_sampler_screen_v1",
        "what": "stage 0 of design_world_sampler.md §5: zero-playout truth-weight meters of candidate samplers",
        "params": {"ALPHA": ALPHA, "N_MC": N_MC, "NEAR": NEAR, "BOOT": BOOT, "BOOT_SEED": BOOT_SEED,
                   "corpora": {k: len(v.teams) for k, v in corpora.items()}, "smogon_pool": len(S.pool)},
        "gigo": {
            "states": len(rows),
            "seen_not_in_true_team": sum(bool(r["gigo_seen_not_true"]) for r in rows),
            "unseen_out_of_smogon_pool": sum(bool(r["unseen_out_of_pool"]) for r in rows),
            "F_rule_vs_stored_F_worlds": {"n": len(val), "analytic_hit": float(np.mean([x for x, _ in val])),
                                          "stored_worlds_hit": float(np.mean([y for _, y in val])),
                                          "corr": float(np.corrcoef(np.array(val).T)[0, 1]) if len(val) > 2 else None},
            "FN_rule_vs_stored_FN_worlds": {anc: {"n": len(v), "analytic_hit": float(np.mean([x for x, _ in v])),
                                                  "stored_worlds_hit": float(np.mean([y for _, y in v]))}
                                            for anc, v in val_fn.items() if v},
        },
        "summary": {k: summarise(v, k) for k, v in subsets.items() if v},
        "cpu_s": round(time.process_time(), 1), "wall_s": round(time.time() - t0, 1),
    }
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    with gzip.open(Path(a.out).with_suffix(".rows.jsonl.gz"), "wt") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps(res["gigo"], indent=1), f"\nwall {res['wall_s']} s")


if __name__ == "__main__":
    main()
