"""POOL-vs-PRIOR confound (hidden-information binding study, 2026-10-11). DESCRIPTIVE ONLY: the pool, the
Metamon corpus and the human replays MEASURE structure here; nothing below ships as a prior.

Question: the value-discrimination pilot's honest worlds (opponent sets drawn from the SMOGON usage prior) put
the true move in a re-drawn slot 16 % of the time and hold ~45 % of the true unseen moves. The opponents there
play OUR TRAINING POOL's teams. How well does the same Smogon prior predict (a) the pool's sets, (b) real human
sets, (c) the curated sample teams, (d) the opponent teams the pilot actually met?

Sources
  pool        utils.team_loader.TeamLoader over data/teams (719 teams; per-collection breakdown recorded)
  sample      data/teams/sample (a SUBSET of the pool: 72 teams)
  metamon     src/utils/ladder_corpus (Metamon hl_05_26 gen3ou, 22,813 teams): real ladder teams RECONSTRUCTED
              from replays, the unrevealed fields FILLED FROM USAGE STATS (Metamon README) -> biased TOWARD the
              prior; an upper bound for human sets, not an honest one
  replays     <main>/replays/showdown/gen3ou raw public replay logs (REVEALED information only, honest but
              CENSORED); a seeded sample of N_REPLAYS logs
  prior_self  REFERENCE ceiling: for each pool mon, N_SELF synthetic TRUE move sets drawn from the Smogon prior by
              the pilot's own sampler (only its move statistics are meaningful) = the hit rates if the prior
              were exactly the population the sets come from
  bank_<cp>   the pilot's 600 states per checkpoint: the opponent team (the side opposite the decision's side)
              from ~/gen3ai_archive/value_disc_pilot/<cp>/bank, and the moves revealed at the state (the world
              records' ``revealed``)

Statistics (P = agents.gen3_data.priors, Smogon; per-move inclusion probabilities sum to 4 per species):
  moves  i   exp_recall_r0  = sum_{m in T} P(m) / |T|
         ii  top4_recall_r0 = |top4(P) & T| / |T|
         iii r = 2, averaged over all C(|T|,2) revealed pairs R, U = T \\ R, candidates C = keys(P) \\ R (a
             revealed Hidden Power removes every other HP key): exp_recall_r2 = sum_{m in U} min(1, P(m) *
             |U| / P(C)) / |U|; top2_recall_r2 = |top_|U|(P on C) & U| / |U|
         iv  the pilot's slot draw (determinize.sample_moves: candidates = prior keys \\ (kept | BANNED_MOVES),
             Hidden Power only of the set's IV type, drawn without replacement prop. to P, filling the empty slots
             in index order), MC_DRAWS Gumbel-top-k draws: slot_hit (positional, the pilot's "16 %") and
             set_hit (share of the true unrevealed moves drawn, the pilot's "45 %"), at r = 0 and r = 2
         HP-merged variants (suffix _hpm): every typed Hidden Power collapsed to "hiddenpower" (P summed), the
         only form the replay protocol shows
  item   p_true, top1;  ability (species with >= 2 Smogon abilities): p_true, top1
  spread p_exact (all chaos spreads), p_exact_top25 (the prior file the pilot draws from), top1, coarse
         (nature's boosted/hindered stat + the two most-invested EV stats): p_coarse, top1_coarse
  teammates for every ordered pair (A, B) on a team: P(B | A), rank of B among A's teammates (descending; an
         unlisted B gets the mid-rank of the unlisted), top5 recall of A's 5 true teammates
  CENSORED (replays vs the pool like-for-like): per REVEALED move: P_hpm(m) and m in top4_hpm. The pool's
         version weights each of a mon's moves by E[revealed-move count | species] / |T| from the replays (=
         random censoring at the replays' per-species reveal count); the bank's version uses the pilot's ACTUAL
         reveals.
Weightings: per_mon (the source's own species mix), per_species (each species equal), smogon_std (each species'
mean weighted by its Smogon usage). CIs: bootstrap over TEAMS (replays: a team = one side of one replay;
bank: a team = one battle's opponent side), 2,000 draws, numpy default_rng(20261011), restarted per source.

    python -I poolprior_read.py            ->  poolprior_read.json (beside this file)
"""
import collections
import glob
import gzip
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from agents.gen3_data import priors  # noqa: E402
from agents.gen3_data import natures as natures_mod  # noqa: E402
from utils.team_packing import Teambuilder  # noqa: E402

SEED = 20261011
N_BOOT = 2000
MC_DRAWS = 200
N_SELF = 5
N_REPLAYS = 40000
MAIN = Path("/home/goodlad/dev/gen3ai")
REPLAY_DIR = MAIN / "replays" / "showdown" / "gen3ou"
ARCH = Path.home() / "gen3ai_archive" / "value_disc_pilot"
PILOT = REPO / "designs" / "research_state" / "measurements" / "value_disc_pilot_2026-10-11"
HIGH_RATING = 1500

HP_TYPES = ["fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water",
            "grass", "electric", "psychic", "ice", "dragon", "dark"]
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
# determinize.BANNED_MOVES, verbatim (the pilot's slot-draw candidate filter)
BANNED_MOVES = frozenset("""attract bide blastburn bounce camouflage dig dive flatter fly followme foresight
frenzyplant grudge helpinghand hydrocannon ingrain lockon magiccoat magnitude metronome mindreader mirrormove
mist mudsport naturepower nightmare odorsleuth present psywave razorwind roleplay sketch skullbash skyattack
snore spite spitup stockpile swallow teleport triplekick watersport""".split())


def toid(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def hp_type_of(ivs):
    iv = dict(zip(STATS, ivs))
    v = sum((iv[k] & 1) << i for i, k in enumerate(("hp", "atk", "def", "spe", "spa", "spd")))
    return HP_TYPES[v * 15 // 63]


def hpm(m):
    return "hiddenpower" if m.startswith("hiddenpower") else m


# ---------------------------------------------------------------- priors (cached per species)
_MOVE_CACHE, _MOVE_HPM_CACHE = {}, {}


def move_prior(sid):
    if sid not in _MOVE_CACHE:
        _MOVE_CACHE[sid] = dict(priors.moves(sid))
    return _MOVE_CACHE[sid]


def move_prior_hpm(sid):
    if sid not in _MOVE_HPM_CACHE:
        out = collections.defaultdict(float)
        for m, p in move_prior(sid).items():
            out[hpm(m)] += p
        _MOVE_HPM_CACHE[sid] = dict(out)
    return _MOVE_HPM_CACHE[sid]


def topk(P, k, exclude=()):
    keys = sorted((m for m in P if m not in exclude), key=lambda m: (-P[m], m))
    return keys[:k]


NATURE_PM = {}
for _n, _mult in natures_mod.multipliers().items():
    _plus = [k for k, v in _mult.items() if v > 1.0]
    _minus = [k for k, v in _mult.items() if v < 1.0]
    NATURE_PM[toid(_n)] = (_plus[0] if _plus else None, _minus[0] if _minus else None)


def coarse_key(nature, evs):
    pm = NATURE_PM.get(toid(nature), (None, None))
    order = sorted(range(6), key=lambda i: (-evs[i], i))
    return (pm, tuple(sorted(order[:2])))


_SPREAD_CACHE = {}


def spread_tables(sid):
    if sid not in _SPREAD_CACHE:
        full = [(toid(n), tuple(int(x) for x in e), float(w)) for n, e, w in priors.all_spreads(sid)]
        top = [(toid(n), tuple(int(x) for x in e), float(w)) for n, e, w in priors.spreads(sid)]
        exact = collections.defaultdict(float)
        for n, e, w in full:
            exact[(n, e)] += w
        exact_top = collections.defaultdict(float)
        for n, e, w in top:
            exact_top[(n, e)] += w
        coarse = collections.defaultdict(float)
        for n, e, w in full:
            coarse[coarse_key(n, e)] += w
        _SPREAD_CACHE[sid] = (dict(exact), dict(exact_top), dict(coarse))
    return _SPREAD_CACHE[sid]


_TOP1 = {}


def _argmax(d):
    return max(d, key=lambda k: (d[k], str(k))) if d else None


def top1_item(sid):
    if ("item", sid) not in _TOP1:
        _TOP1[("item", sid)] = _argmax(priors.items(sid))
    return _TOP1[("item", sid)]


def top1_ability(sid):
    if ("ab", sid) not in _TOP1:
        _TOP1[("ab", sid)] = _argmax(priors.ability(sid))
    return _TOP1[("ab", sid)]


def spread_top1(sid):
    if ("sp", sid) not in _TOP1:
        exact, _t, coarse = spread_tables(sid)
        _TOP1[("sp", sid)] = (_argmax(exact), _argmax(coarse))
    return _TOP1[("sp", sid)]


_TOPK = {}


def topk_cached(tag, sid, P, k):
    if (tag, sid, k) not in _TOPK:
        _TOPK[(tag, sid, k)] = topk(P, k)
    return _TOPK[(tag, sid, k)]


USAGE = priors.species_usage()
ALL_SPECIES = sorted(USAGE)


# ---------------------------------------------------------------- set parsing (packed format, every source)
def parse_packed_mon(entry):
    f = entry.split("|")
    f += [""] * (12 - len(f))
    sid = toid(f[1] or f[0])
    ivs = [int(x) if x else 31 for x in f[8].split(",")] if f[8] else [31] * 6
    if len(ivs) != 6:
        ivs = (ivs + [31] * 6)[:6]
    endf = f[11].split(",")
    hptype = toid(endf[1]) if len(endf) > 1 and endf[1] else None
    ivtype = hp_type_of(ivs)
    moves = []
    for m in f[4].split(","):
        m = toid(m)
        if not m:
            continue
        if m == "hiddenpower":
            m = "hiddenpower" + (hptype or ivtype)
        moves.append(m)
    evs = [int(x) if x else 0 for x in f[6].split(",")] if f[6] else [0] * 6
    evs = (evs + [0] * 6)[:6]
    return {"sid": sid, "item": toid(f[2]) or "nothing", "ability": toid(f[3]), "moves": moves,
            "nature": toid(f[5]) or "serious", "evs": tuple(evs), "hp_iv_type": ivtype}


def parse_packed_team(packed):
    return [parse_packed_mon(e) for e in packed.split("]") if e]


# ---------------------------------------------------------------- the accumulator + bootstrap
class Source:
    def __init__(self, name):
        self.name = name
        self.team_index = {}
        self.stats = collections.defaultdict(lambda: ([], [], [], []))   # t, s, v, w
        self.notes = collections.Counter()
        self.species_count = collections.Counter()

    def team(self, key):
        if key not in self.team_index:
            self.team_index[key] = len(self.team_index)
        return self.team_index[key]

    def add(self, stat, t, sid, v, w=1.0):
        if sid not in SP_INDEX:
            self.notes[f"{stat}: species without a usage entry"] += 1
            return
        a = self.stats[stat]
        a[0].append(t)
        a[1].append(SP_INDEX[sid])
        a[2].append(float(v))
        a[3].append(float(w))


SP_INDEX = {s: i for i, s in enumerate(ALL_SPECIES)}
U_VEC = np.array([USAGE[s] for s in ALL_SPECIES], dtype=np.float64)


def _weighted(Swv, Sw):
    """per_species and smogon_std from per-species sums (rows: draws)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        m = Swv / Sw
    present = Sw > 0
    per_species = np.nanmean(np.where(present, m, np.nan), axis=-1)
    uw = np.where(present, U_VEC, 0.0)
    smog = np.nansum(np.where(present, m, 0.0) * uw, axis=-1) / uw.sum(axis=-1)
    return per_species, smog


def summarize(src, keep_draws):
    n = len(src.team_index)
    out, draws = {}, {}
    mats = {}
    for stat, (t, s, v, w) in src.stats.items():
        t, s, v, w = (np.asarray(x) for x in (t, s, v, w))
        Awv = sparse.csr_matrix((w * v, (t, s)), shape=(n, len(ALL_SPECIES)))
        Aw = sparse.csr_matrix((w, (t, s)), shape=(n, len(ALL_SPECIES)))
        Swv = np.asarray(Awv.sum(axis=0)).ravel()
        Sw = np.asarray(Aw.sum(axis=0)).ravel()
        ps, sm = _weighted(Swv[None], Sw[None])
        out[stat] = {"n_rows": int(len(v)), "n_weight": round(float(w.sum()), 3),
                     "n_teams": int(len(np.unique(t))),
                     "per_mon": {"est": float((w * v).sum() / w.sum())},
                     "per_species": {"est": float(ps[0])}, "smogon_std": {"est": float(sm[0])},
                     "n_species": int((Sw > 0).sum())}
        mats[stat] = (Awv.T.tocsr(), Aw.T.tocsr())
    rng = np.random.default_rng(SEED)
    boot = {stat: {"per_mon": [], "per_species": [], "smogon_std": []} for stat in mats}
    chunk = 100
    for c0 in range(0, N_BOOT, chunk):
        C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(min(chunk, N_BOOT - c0))])
        C = C.astype(np.float64).T          # n x B
        for stat, (AwvT, AwT) in mats.items():
            Swv = (AwvT @ C).T               # B x S
            Sw = (AwT @ C).T
            ps, sm = _weighted(Swv, Sw)
            boot[stat]["per_mon"].append(Swv.sum(1) / Sw.sum(1))
            boot[stat]["per_species"].append(ps)
            boot[stat]["smogon_std"].append(sm)
    for stat in mats:
        for k in ("per_mon", "per_species", "smogon_std"):
            d = np.concatenate(boot[stat][k])
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            out[stat][k]["ci95"] = [round(float(lo), 4), round(float(hi), 4)]
            out[stat][k]["est"] = round(out[stat][k]["est"], 4)
            if keep_draws:
                draws[(src.name, stat, k)] = d
    return out, draws


# ---------------------------------------------------------------- per-mon statistics on FULL sets
def gumbel_draws(rng, keys, w, k, n):
    """n sequential draws of k keys without replacement prop. to w (Plackett-Luce = Gumbel top-k)."""
    g = np.log(np.asarray(w, dtype=np.float64))[None, :] + rng.gumbel(size=(n, len(keys)))
    return np.argsort(-g, axis=1)[:, :k]


_MC_CACHE = {}


def slot_draw(rng, sid, true_moves, kept_idx, hp_iv_type):
    """The pilot's sample_moves on a set whose slots ``kept_idx`` are kept: (positional slot-hit rate, set-hit
    rate) over MC_DRAWS draws, or None if the prior cannot fill the slots."""
    key = (sid, tuple(true_moves), tuple(kept_idx), hp_iv_type)
    if key in _MC_CACHE:
        return _MC_CACHE[key]
    P = move_prior(sid)
    have = {true_moves[i] for i in kept_idx}
    has_hp = any(m.startswith("hiddenpower") for m in have)
    keys = [m for m in P if m not in have and m not in BANNED_MOVES
            and not (m.startswith("hiddenpower") and (has_hp or m != "hiddenpower" + hp_iv_type))]
    empty = [i for i in range(len(true_moves)) if i not in kept_idx]
    if len(keys) < len(empty) or not empty:
        _MC_CACHE[key] = None
        return None
    idx = gumbel_draws(rng, keys, [P[m] for m in keys], len(empty), MC_DRAWS)
    karr = np.array(keys)
    drawn = karr[idx]                                    # n x len(empty), in empty-slot order
    truth_slot = np.array([true_moves[i] for i in empty])
    slot_hit = float((drawn == truth_slot[None, :]).mean())
    unrev = set(truth_slot.tolist())
    set_hit = float(np.isin(drawn, list(unrev)).sum(axis=1).mean() / len(unrev))
    _MC_CACHE[key] = (slot_hit, set_hit)
    return _MC_CACHE[key]


def r_stats(P, T, R):
    """(exp_recall, top_|U|_recall) for true set T with revealed subset R under inclusion prior P."""
    U = [m for m in T if m not in R]
    if not U:
        return None
    hp_rev = any(m.startswith("hiddenpower") for m in R) and any(k != "hiddenpower" for k in P if
                                                                k.startswith("hiddenpower"))
    excl = set(R)
    if hp_rev:
        excl |= {k for k in P if k.startswith("hiddenpower")}
    PC = sum(p for m, p in P.items() if m not in excl)
    if PC <= 0:
        return None
    exp = sum(min(1.0, P.get(m, 0.0) * len(U) / PC) for m in U) / len(U)
    top = topk(P, len(U), excl)
    return exp, len(set(top) & set(U)) / len(U)


def mon_set_stats(src, t, mon, rng, do_mc=True, w=1.0):
    sid = mon["sid"]
    src.species_count[sid] += 1
    P = move_prior(sid)
    T = list(dict.fromkeys(mon["moves"]))
    if not P:
        src.notes["mons without a Smogon move prior"] += 1
    elif T:
        for tag, PP, TT in (("", P, T), ("_hpm", move_prior_hpm(sid), list(dict.fromkeys(hpm(m) for m in T)))):
            src.add("moves.exp_recall_r0" + tag, t, sid, sum(PP.get(m, 0.0) for m in TT) / len(TT), w)
            src.add("moves.top4_recall_r0" + tag, t, sid, len(set(topk_cached(tag, sid, PP, 4)) & set(TT)) / len(TT), w)
            if len(TT) >= 3:
                ex, tp = [], []
                for R in itertools.combinations(TT, 2):
                    r = r_stats(PP, TT, R)
                    if r:
                        ex.append(r[0])
                        tp.append(r[1])
                if ex:
                    src.add("moves.exp_recall_r2" + tag, t, sid, float(np.mean(ex)), w)
                    src.add("moves.top2_recall_r2" + tag, t, sid, float(np.mean(tp)), w)
        if do_mc and len(T) == len(mon["moves"]):
            res0 = slot_draw(rng, sid, T, (), mon["hp_iv_type"])
            if res0:
                src.add("moves.mc_slot_hit_r0", t, sid, res0[0], w)
                src.add("moves.mc_set_hit_r0", t, sid, res0[1], w)
            if len(T) >= 3:
                rs = [slot_draw(rng, sid, T, kept, mon["hp_iv_type"]) for kept in itertools.combinations(range(len(T)), 2)]
                rs = [r for r in rs if r]
                if rs:
                    src.add("moves.mc_slot_hit_r2", t, sid, float(np.mean([r[0] for r in rs])), w)
                    src.add("moves.mc_set_hit_r2", t, sid, float(np.mean([r[1] for r in rs])), w)
    # item
    IP = priors.items(sid)
    if IP:
        src.add("item.p_true", t, sid, IP.get(mon["item"], 0.0), w)
        src.add("item.top1", t, sid, float(top1_item(sid) == mon["item"]), w)
    # ability (two-ability species)
    AP = priors.ability(sid)
    if len(AP) >= 2 and mon["ability"]:
        src.add("ability.p_true", t, sid, AP.get(mon["ability"], 0.0), w)
        src.add("ability.top1", t, sid, float(top1_ability(sid) == mon["ability"]), w)
    # spread
    exact, exact_top, coarse = spread_tables(sid)
    if exact:
        key = (mon["nature"], mon["evs"])
        src.add("spread.p_exact", t, sid, exact.get(key, 0.0), w)
        src.add("spread.p_exact_top25", t, sid, exact_top.get(key, 0.0), w)
        src.add("spread.top1", t, sid, float(spread_top1(sid)[0] == key), w)
        ck = coarse_key(mon["nature"], mon["evs"])
        src.add("spread.p_coarse", t, sid, coarse.get(ck, 0.0), w)
        src.add("spread.top1_coarse", t, sid, float(spread_top1(sid)[1] == ck), w)


def teammate_stats(src, t, sids, w=1.0):
    sids = [s for s in sids if s]
    for a in sids:
        TP = priors.teammates(a)
        if not TP:
            src.notes["teammate: species without a teammate prior"] += 1
            continue
        mates = [b for b in sids if b != a]
        if not mates:
            continue
        order = sorted(TP, key=lambda k: (-TP[k], k))
        rank = {k: i + 1 for i, k in enumerate(order)}
        n_unl = len(ALL_SPECIES) - 1 - len(order)
        mid_unl = len(order) + (n_unl + 1) / 2.0
        for b in mates:
            src.add("teammates.p_true", t, a, TP.get(b, 0.0), w)
            src.add("teammates.rank", t, a, rank.get(b, mid_unl), w)
        src.add("teammates.top5_recall", t, a, len(set(order[:5]) & set(mates)) / min(5, len(mates)), w)


# ---------------------------------------------------------------- (a) pool, (c) sample
def load_pool():
    """TeamLoader's own walk, with the collection label kept; asserted equal to TeamLoader's output."""
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        from utils.team_loader import TeamLoader
        ref = TeamLoader().get_all_teams()
        sample, other = [], []
        seen = set()
        for root, _dirs, files in os.walk("data/teams"):
            if "teams.json" not in files:
                continue
            for entry in json.load(open(os.path.join(root, "teams.json"))):
                if entry.get("valid") is False or not entry.get("file"):
                    continue
                p = os.path.join("data", entry["file"])
                if not os.path.exists(p):
                    raise SystemExit(f"pool file missing: {p}")
                rp = os.path.realpath(p)
                if rp in seen:
                    continue
                seen.add(rp)
                txt = open(p).read().strip()
                coll = os.path.basename(root)
                (sample if "sample" in root else other).append((coll, entry["file"], txt))
        teams = sample + other
    finally:
        os.chdir(cwd)
    if [x[2] for x in teams] != ref:
        raise SystemExit("pool walk does not reproduce TeamLoader().get_all_teams()")
    return teams


def packed_from_paste(txt):
    return "]".join(m.packed for m in Teambuilder.parse_showdown_team(txt))


# ---------------------------------------------------------------- (b) replays
OF_HOLDER = {"sandstream", "drizzle", "drought", "roughskin", "static", "poisonpoint", "effectspore", "flamebody",
             "cutecharm", "synchronize", "stench"}


def ident(x):
    if len(x) >= 4 and x[0] == "p" and x[1] in "12" and ":" in x:
        return (x[:2], x.split(":", 1)[1].strip())
    return None


def parse_replay(path):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    rating, species, conflict = {}, {}, set()
    moves = collections.defaultdict(set)
    item, abil = {}, collections.defaultdict(set)
    taint_item, taint_move, taint_abil, pending = set(), set(), set(), set()
    trick = None
    for ln in text.split("\n"):
        if not ln.startswith("|"):
            continue
        f = ln.split("|")
        tag = f[1] if len(f) > 1 else ""
        if tag in ("move", "switch", "drag", "turn", "upkeep", "") and pending:
            taint_item |= pending
            pending = set()
        if tag == "player" and len(f) >= 6 and f[5].strip().isdigit():
            rating[f[2]] = int(f[5])
        elif tag in ("switch", "drag", "replace") and len(f) >= 4:
            k = ident(f[2])
            if k:
                sp = toid(f[3].split(",")[0])
                if k in species and species[k] != sp:
                    conflict.add(k)
                species[k] = sp
        elif tag == "move" and len(f) >= 4:
            k = ident(f[2])
            rest = f[5:]
            if not k or any(x.startswith("[from]") and "Pursuit" not in x for x in rest):
                continue
            m = toid(f[3])
            if m in ("struggle", "recharge") or k in taint_move:
                continue
            moves[k].add(hpm(m))
            if m == "mimic":
                taint_move.add(k)
        elif tag == "-transform":
            k = ident(f[2])
            if k:
                taint_move.add(k)
        elif tag == "-activate" and len(f) >= 5 and "move: Trick" in f[3]:
            src_k = ident(f[2])
            of = [ident(x[5:].strip()) for x in f[4:] if x.startswith("[of]")]
            trick = (src_k, of[0] if of else None)
        if tag == "-item" and len(f) >= 4:
            k = ident(f[2])
            rest = " ".join(f[4:])
            if "move: Trick" in rest:
                if trick and k in trick:
                    other = trick[1] if k == trick[0] else trick[0]
                    if other and other not in taint_item and other not in item:
                        item[other] = toid(f[3])
                pending |= {x for x in (trick or ()) if x}
                pending.add(k)
            elif "move: Thief" in rest or "move: Covet" in rest:
                pending.add(k)
            elif k and k not in taint_item and k not in item:
                item[k] = toid(f[3])
        elif tag == "-enditem" and len(f) >= 4:
            k = ident(f[2])
            if k and k not in taint_item and k not in item:
                item[k] = toid(f[3])
        if "[from] item:" in ln and tag not in ("-item", "-enditem"):
            k = ident(f[2]) if len(f) > 2 else None
            it = [toid(x.split("item:", 1)[1]) for x in f if x.startswith("[from] item:")]
            if k and it and k not in taint_item and k not in item:
                item[k] = it[0]
        # abilities
        if tag == "-ability" and len(f) >= 4:
            k = ident(f[2])
            of = [ident(x[5:].strip()) for x in f[4:] if x.startswith("[of]")]
            if len(f) >= 5 and (f[4] == "Trace" or "[from] ability: Trace" in f[4:]):
                if k and k not in taint_abil:
                    abil[k].add("trace")
                if of and of[0] and of[0] not in taint_abil:
                    abil[of[0]].add(toid(f[3]))
                if k:
                    taint_abil.add(k)
            elif k and k not in taint_abil:
                abil[k].add(toid(f[3]))
        elif "[from] ability:" in ln:
            a = [toid(x.split("ability:", 1)[1]) for x in f if x.startswith("[from] ability:")][0]
            of = [ident(x[5:].strip()) for x in f if x.startswith("[of]")]
            line_mon = ident(f[2]) if len(f) > 2 else None
            holder = of[0] if (of and (a in OF_HOLDER or line_mon is None)) else line_mon
            if holder and holder not in taint_abil:
                abil[holder].add(a)
    return {"rating": rating, "species": species, "conflict": conflict, "moves": moves, "item": item,
            "abil": abil}


# ---------------------------------------------------------------- main
def main():
    res = {"meta": {}, "sources": {}, "comparisons": {}, "findings_counts": {}}
    all_draws = {}
    keep = True

    # ---- (a) pool + (c) sample
    pool = load_pool()
    pool_src, sample_src = Source("pool"), Source("sample")
    coll_srcs = {}
    pool_packed = []
    rng = np.random.default_rng(SEED)
    for i, (coll, fpath, txt) in enumerate(pool):
        packed = packed_from_paste(txt)
        pool_packed.append(packed)
        mons = parse_packed_team(packed)
        srcs = [pool_src] + ([sample_src] if coll == "sample" else [])
        cs = coll_srcs.setdefault(coll, Source("pool:" + coll))
        srcs.append(cs)
        for s in srcs:
            t = s.team(i)
            for mon in mons:
                mon_set_stats(s, t, mon, rng)
            teammate_stats(s, t, [m["sid"] for m in mons])
    # REFERENCE (the "prior is exactly right" ceiling): for every pool mon, N_SELF synthetic TRUE sets drawn by
    # the pilot's own sampler from the Smogon prior (same candidates, HP of the mon's IV type), scored as above
    selfsrc = Source("prior_self_poolmix")
    rng_s = np.random.default_rng(SEED)
    for i, (coll, fpath, txt) in enumerate(pool):
        t = selfsrc.team(i)
        for mon in parse_packed_team(pool_packed[i]):
            P = move_prior(mon["sid"])
            if not P:
                continue
            keys = [m for m in P if m not in BANNED_MOVES
                    and not (m.startswith("hiddenpower") and m != "hiddenpower" + mon["hp_iv_type"])]
            if len(keys) < 4:
                continue
            for idx in gumbel_draws(rng_s, keys, [P[m] for m in keys], 4, N_SELF):
                fake = dict(mon, moves=[keys[j] for j in idx])
                mon_set_stats(selfsrc, t, fake, rng_s, w=1.0 / N_SELF)
    print("stage: pool done", flush=True)
    res["meta"]["pool"] = {
        "loader": "utils.team_loader.TeamLoader(base_dir='data/teams').get_all_teams() (reproduced here with labels)",
        "n_teams": len(pool), "by_collection": dict(collections.Counter(c for c, _, _ in pool)),
        "n_mons": sum(len(parse_packed_team(p)) for p in pool_packed)}

    # ---- the replays (b): a seeded sample
    files = sorted(glob.glob(str(REPLAY_DIR / "*" / "*.log")))
    rr = np.random.default_rng(SEED)
    pick = sorted(rr.choice(len(files), size=min(N_REPLAYS, len(files)), replace=False).tolist())
    rep = {"all": Source("replays"), "hi": Source(f"replays_ge{HIGH_RATING}")}
    rep_full4 = {"all": Source("replays_full4"), "hi": Source(f"replays_full4_ge{HIGH_RATING}")}
    rep_t6 = {"all": Source("replays_team6"), "hi": Source(f"replays_team6_ge{HIGH_RATING}")}
    r_by_species = collections.defaultdict(list)
    rn = collections.Counter()
    legal_ab = collections.defaultdict(set)
    for sid in ALL_SPECIES:
        legal_ab[sid] |= set(priors.ability(sid))
    replay_rows = []
    dates = set()
    for j in pick:
        p = files[j]
        dates.add(Path(p).parent.name)
        r = parse_replay(p)
        if r is None:
            rn["unreadable"] += 1
            continue
        rn["replays"] += 1
        replay_rows.append((p, r))
    # legal abilities: Smogon keys + everything the (legal) pool / Metamon sets carry
    lad_rows = []
    with gzip.open(REPO / "src" / "utils" / "ladder_corpus" / "teams.jsonl.gz", "rt") as fh:
        for line in fh:
            lad_rows.append(json.loads(line))
    for packed in pool_packed + [x["packed"] for x in lad_rows]:
        for mon in parse_packed_team(packed):
            if mon["ability"]:
                legal_ab[mon["sid"]].add(mon["ability"])
    # pass 1: per-species revealed-move counts (for the pool's like-for-like censoring)
    for p, r in replay_rows:
        for k, sp in r["species"].items():
            if k in r["conflict"]:
                continue
            nm = len(r["moves"].get(k, ()))
            if nm <= 4:
                r_by_species[sp].append(nm)
    for p, r in replay_rows:
        for side in ("p1", "p2"):
            mons = {k: sp for k, sp in r["species"].items() if k[0] == side and k not in r["conflict"]}
            if not mons:
                continue
            rating = r["rating"].get(side)
            tiers = ["all"] + (["hi"] if rating is not None and rating >= HIGH_RATING else [])
            rn["team_sides"] += 1
            rn["team_sides_rated"] += rating is not None
            rn["team_sides_hi"] += "hi" in tiers
            for tier in tiers:
                s = rep[tier]
                t = s.team((p, side))
                for k, sp in mons.items():
                    s.species_count[sp] += 1
                    mv = r["moves"].get(k, set())
                    if len(mv) > 4:
                        s.notes["mons with > 4 distinct revealed moves (dropped)"] += 1
                        continue
                    PH = move_prior_hpm(sp)
                    if PH and mv:
                        top4 = set(topk_cached("_hpm", sp, PH, 4))
                        for m in mv:
                            s.add("censored.p_revealed_move_hpm", t, sp, PH.get(m, 0.0))
                            s.add("censored.revealed_in_top4_hpm", t, sp, float(m in top4))
                    if len(mv) == 4 and PH:
                        f4 = rep_full4[tier]
                        ft = f4.team((p, side))
                        TT = sorted(mv)
                        f4.add("moves.exp_recall_r0_hpm", ft, sp, sum(PH.get(m, 0.0) for m in TT) / 4)
                        f4.add("moves.top4_recall_r0_hpm", ft, sp, len(set(topk_cached("_hpm", sp, PH, 4)) & set(TT)) / 4)
                        ex, tp = zip(*[r_stats(PH, TT, R) for R in itertools.combinations(TT, 2)])
                        f4.add("moves.exp_recall_r2_hpm", ft, sp, float(np.mean(ex)))
                        f4.add("moves.top2_recall_r2_hpm", ft, sp, float(np.mean(tp)))
                    it = r["item"].get(k)
                    IP = priors.items(sp)
                    if it and IP:
                        s.add("censored.item.p_true", t, sp, IP.get(it, 0.0))
                        s.add("censored.item.top1", t, sp, float(top1_item(sp) == it))
                        s.notes["item reveals: " + it] += 1
                    ab = r["abil"].get(k, set())
                    AP = priors.ability(sp)
                    if len(AP) >= 2 and ab:
                        if len(ab) > 1 or not ab <= legal_ab[sp]:
                            s.notes["ability: conflicting / not-legal attribution (dropped)"] += 1
                        else:
                            a = next(iter(ab))
                            s.add("censored.ability.p_true", t, sp, AP.get(a, 0.0))
                            s.add("censored.ability.top1", t, sp, float(top1_ability(sp) == a))
                if len(mons) == 6:
                    t6 = rep_t6[tier]
                    teammate_stats(t6, t6.team((p, side)), list(mons.values()))
    print("stage: replays done", flush=True)
    e_r = {sp: float(np.mean(v)) for sp, v in r_by_species.items()}
    glob_r = float(np.mean([x for v in r_by_species.values() for x in v]))
    r_hist = collections.Counter(x for v in r_by_species.values() for x in v)
    res["meta"]["replays"] = {
        "dir": str(REPLAY_DIR), "files_total": len(files), "sampled": len(pick), "sample_seed": SEED,
        "date_folders_in_sample": [min(dates), max(dates)], "n_date_folders": len(dates), **dict(rn),
        "revealed_move_count_hist_per_appeared_mon": {str(k): v for k, v in sorted(r_hist.items())},
        "mean_revealed_moves_per_appeared_mon": round(glob_r, 4),
        "high_rating_cut": HIGH_RATING}

    # pool like-for-like censoring: each move of a pool mon weighted by E[r | species] / |T|
    for i, packed in enumerate(pool_packed):
        for s in ([pool_src] + ([sample_src] if pool[i][0] == "sample" else [])):
            t = s.team(i)
            for mon in parse_packed_team(packed):
                PH = move_prior_hpm(mon["sid"])
                TT = list(dict.fromkeys(hpm(m) for m in mon["moves"]))
                if not PH or not TT:
                    continue
                er = e_r.get(mon["sid"], glob_r)
                top4 = set(topk_cached("_hpm", mon["sid"], PH, 4))
                for m in TT:
                    s.add("censored.p_revealed_move_hpm", t, mon["sid"], PH.get(m, 0.0), er / len(TT))
                    s.add("censored.revealed_in_top4_hpm", t, mon["sid"], float(m in top4), er / len(TT))
    # the pool's items restricted to the items the replays reveal at all (like-for-like for censored items)
    rev_items = {k.split(": ", 1)[1]: v for k, v in rep["all"].notes.items() if k.startswith("item reveals: ")}
    revealable = {it for it, c in rev_items.items() if c >= 20}
    for i, packed in enumerate(pool_packed):
        for s in ([pool_src] + ([sample_src] if pool[i][0] == "sample" else [])):
            t = s.team(i)
            for mon in parse_packed_team(packed):
                IP = priors.items(mon["sid"])
                if IP and mon["item"] in revealable:
                    s.add("censored.item.p_true", t, mon["sid"], IP.get(mon["item"], 0.0))
                    s.add("censored.item.top1", t, mon["sid"],
                          float(top1_item(mon["sid"]) == mon["item"]))
    res["meta"]["replays"]["revealable_items_ge20"] = sorted(revealable)
    res["meta"]["replays"]["item_reveal_counts_top"] = dict(sorted(rev_items.items(), key=lambda x: -x[1])[:15])

    # ---- (b') Metamon hl_05_26 (usage-FILLED)
    met = Source("metamon_hl0526")
    rng_m = np.random.default_rng(SEED)
    for row in lad_rows:
        mons = parse_packed_team(row["packed"])
        t = met.team(row["sha"] + str(row["i"]))
        for mon in mons:
            mon_set_stats(met, t, mon, rng_m)
        teammate_stats(met, t, [m["sid"] for m in mons])
    print("stage: metamon done", flush=True)
    lm = json.loads((REPO / "src" / "utils" / "ladder_corpus" / "manifest.json").read_text())
    res["meta"]["metamon"] = {"file": "src/utils/ladder_corpus/teams.jsonl.gz", "n_teams": len(lad_rows),
                              "source": lm["source"], "data_sha256": lm["data_sha256"]}

    # ---- (d) the pilot's opponent teams
    pool_keys = {}
    for i, packed in enumerate(pool_packed):
        mons = parse_packed_team(packed)
        pool_keys[tuple(sorted((m["sid"], tuple(sorted(m["moves"])), m["item"]) for m in mons))] = i
    bank_meta = {}
    bank_srcs = {}
    for cp in ("prod", "end2"):
        with gzip.open(ARCH / cp / "bank" / "battles.jsonl.gz", "rt") as fh:
            battles = {b["battle_id"]: b for b in (json.loads(x) for x in fh)}
        with gzip.open(ARCH / cp / "bank" / "decisions.jsonl.gz", "rt") as fh:
            decs = {d["id"]: d for d in (json.loads(x) for x in fh)}
        states = json.loads((PILOT / f"states_{cp}.json").read_text())["states"]
        revealed = {}
        for dn in ("det", "det_b", "det_c"):
            for wp in sorted((ARCH / cp / dn).glob("worlds_*.jsonl")):
                for line in wp.read_text().splitlines():
                    w = json.loads(line)
                    if "revealed" in w and w["id"] in states:
                        if w["id"] in revealed and revealed[w["id"]] != w["revealed"]:
                            raise SystemExit(f"{cp}: inconsistent revealed for {w['id']}")
                        revealed[w["id"]] = w["revealed"]
        s = Source(f"bank_{cp}")
        rng_b = np.random.default_rng(SEED)
        in_pool = collections.Counter()
        opp_names = collections.Counter()
        for sid_state in sorted(states):
            d = decs[sid_state]
            b = battles[d["battle"]]
            opp = b["p2"] if d["side"] == "p1" else b["p1"]
            mons = parse_packed_team(opp["team"])
            key = tuple(sorted((m["sid"], tuple(sorted(m["moves"])), m["item"]) for m in mons))
            in_pool[key in pool_keys] += 1
            opp_names[d.get("opp_class", "?")] += 1
            t = s.team(d["battle"])
            for mon in mons:
                mon_set_stats(s, t, mon, rng_b)
            teammate_stats(s, t, [m["sid"] for m in mons])
            rv = revealed.get(sid_state)
            if rv is None:
                s.notes["states without a world record (no revealed set)"] += 1
                continue
            for mon in mons:
                sid = mon["sid"]
                if sid not in rv:
                    continue
                seen = set(rv[sid])
                T = mon["moves"]
                kept = tuple(i for i, m in enumerate(T) if m in seen or (m.startswith("hiddenpower") and "hiddenpower" in seen))
                if len(kept) == len(T):
                    continue
                P = move_prior(sid)
                if not P:
                    continue
                R = [T[i] for i in kept]
                rs = r_stats(P, T, R)
                if rs:
                    s.add("at_reveal.exp_recall", t, sid, rs[0])
                    s.add("at_reveal.topU_recall", t, sid, rs[1])
                mc = slot_draw(rng_b, sid, T, kept, mon["hp_iv_type"])
                if mc:
                    s.add("at_reveal.mc_slot_hit", t, sid, mc[0], len(T) - len(kept))
                    s.add("at_reveal.mc_set_hit", t, sid, mc[1], len(T) - len(kept))
                PH = move_prior_hpm(sid)
                top4 = set(topk_cached("_hpm", sid, PH, 4))
                for m in {hpm(x) for x in R}:
                    s.add("censored.p_revealed_move_hpm", t, sid, PH.get(m, 0.0))
                    s.add("censored.revealed_in_top4_hpm", t, sid, float(m in top4))
        bank_srcs[cp] = s
        bank_meta[cp] = {"states": len(states), "battles": len(s.team_index),
                         "states_with_revealed": len(revealed),
                         "opp_team_is_a_pool_team(states)": {str(k): v for k, v in in_pool.items()},
                         "opp_class(states)": dict(opp_names)}
    print("stage: bank done", flush=True)
    res["meta"]["bank"] = bank_meta

    # ---- summarize
    order = ([pool_src, sample_src, selfsrc] + [coll_srcs[c] for c in sorted(coll_srcs)] + [met]
             + [rep["all"], rep["hi"], rep_full4["all"], rep_full4["hi"], rep_t6["all"], rep_t6["hi"]]
             + [bank_srcs["prod"], bank_srcs["end2"]])
    for s in order:
        st, dr = summarize(s, keep)
        all_draws.update(dr)
        res["sources"][s.name] = {"n_teams": len(s.team_index), "stats": st,
                                  "notes": dict(sorted(s.notes.items(), key=lambda x: -x[1])[:25]),
                                  "top_species": dict(s.species_count.most_common(12))}
        print(s.name, len(s.team_index), {k: v["per_mon"]["est"] for k, v in sorted(st.items())}, flush=True)

    # ---- comparisons: A - B on independent bootstrap draws
    def comp(a, b, stat, w="per_mon"):
        da, db = all_draws.get((a, stat, w)), all_draws.get((b, stat, w))
        if da is None or db is None:
            return None
        ea = res["sources"][a]["stats"][stat][w]["est"]
        eb = res["sources"][b]["stats"][stat][w]["est"]
        d = da - db
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return {"a": ea, "b": eb, "delta": round(ea - eb, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}

    pairs = [
        ("replays", "pool", ["censored.p_revealed_move_hpm", "censored.revealed_in_top4_hpm", "censored.item.p_true",
                             "censored.item.top1", "censored.ability.p_true", "censored.ability.top1"]),
        (f"replays_ge{HIGH_RATING}", "pool", ["censored.p_revealed_move_hpm", "censored.revealed_in_top4_hpm",
                                             "censored.item.p_true", "censored.item.top1"]),
        ("replays_full4", "pool", ["moves.exp_recall_r0_hpm", "moves.top4_recall_r0_hpm", "moves.exp_recall_r2_hpm",
                                   "moves.top2_recall_r2_hpm"]),
        ("replays_team6", "pool", ["teammates.p_true", "teammates.rank", "teammates.top5_recall"]),
        ("metamon_hl0526", "pool", ["moves.exp_recall_r0", "moves.top4_recall_r0", "moves.exp_recall_r2",
                                    "moves.top2_recall_r2", "moves.mc_slot_hit_r2", "moves.mc_set_hit_r2",
                                    "item.p_true", "item.top1", "ability.p_true", "spread.p_exact", "spread.top1",
                                    "spread.p_coarse", "spread.top1_coarse", "teammates.p_true", "teammates.rank"]),
        ("prior_self_poolmix", "pool", ["moves.exp_recall_r0", "moves.top4_recall_r0", "moves.exp_recall_r2",
                                        "moves.top2_recall_r2", "moves.mc_slot_hit_r0", "moves.mc_set_hit_r0",
                                        "moves.mc_slot_hit_r2", "moves.mc_set_hit_r2"]),
        ("sample", "pool", ["moves.exp_recall_r0", "moves.top4_recall_r0", "moves.mc_set_hit_r2", "item.p_true",
                            "spread.p_exact", "spread.p_coarse"]),
        ("bank_prod", "pool", ["moves.exp_recall_r0", "moves.top4_recall_r0", "moves.mc_set_hit_r2"]),
    ]
    for a, b, stats in pairs:
        for stat in stats:
            for w in ("per_mon", "smogon_std"):
                c = comp(a, b, stat, w)
                if c:
                    res["comparisons"][f"{a} - {b} | {stat} | {w}"] = c
    res["meta"].update({
        "seed": SEED, "n_boot": N_BOOT, "mc_draws": MC_DRAWS,
        "priors": "agents.gen3_data.priors (moves/items/ability/spreads/all_spreads/teammates/species_usage)",
        "python": sys.version.split()[0], "numpy": np.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (HERE / "poolprior_read.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
