"""STAGE 1's ARMS as world samplers (design §6.2; ``PREREGISTRATION.md`` §3). BUILDER SIDE (HEAD + this unit).

Every arm but C_pool is F's own builder (``main.value_search.worlds.WorldSampler(reveal="off")``: the screen, the
template, the gate, the legality rule, the guard) with at most THREE departures, each swapped in for the length of one
``sample`` call by :func:`hooks` (Tier 1's FN precedent, ``fn_worlds._fn_species``):

  * the NEVER-SEEN SPECIES draw (``worlds.never_seen_species``);
  * a never-seen mon's WHOLE SET (``worlds.new_set``), or only its MOVES inside F's own set draw;
  * a SEEN mon's UNREVEALED MOVES (``world_draws.draw_moves`` called by the builder's seen-mon loop, after the screen).

A SEEN mon's item, ability and spread always come from F's SCREEN on the SHARED cache (common random numbers with
F_legal; the corpus never has to fit the log's damage numbers). A Hidden Power is ONE move whatever its type, and its
INCLUSION and TYPE on a seen mon are spread facts of that screen (FN's rule): a corpus or set source supplies the other
moves (``corpus_seen_moves``).

  F_legal         F at the builder commit (F2's validator rule on): the baseline of record
  F_off           F at the builder commit with the legality rule OFF (an always-legal validator: the pre-F2 draw)
  B               (b) never-seen species by joint Smogon teammates, sequential, ∝ u(s)·Π_{r ∈ seen ∪ drawn} lift_r(s),
                  lift_r(s) = (t_r(s) + 0.05·ū(s)) / ū(s) (stage 0's rule); everything else F's
  C_ladder        (c) per world, with probability λ = n / (n + ALPHA) a corpus team drawn uniformly among the n LEGAL
                  ladder teams CONSISTENT with the reveals (every seen species, each holding its revealed moves): its
                  never-seen mons whole (species and set), its seen mons' moves; else a B world
  C_forum_strict  (c) on ``w_forum_v1`` minus every team sharing >= 4 mon sets with the state's TRUE team
  NS              never-seen species by the NETWORK's 225M belief (FN's rule: ∝ π over the candidates, F's weights when
                  π has no mass there); every mon's moves, with probability n_s / (n_s + 1), from a uniformly drawn
                  forum set of its species holding its revealed moves (n_s of them), else F's draw; everything else F's
  C_pool          Tier 1's POOL (``pool_worlds.PoolSampler``): the training pool's teams consistent with the reveals,
                  each gated whole: the closed-population calibration ceiling. The pool is X26's PIN's packing
                  (``pool_pin.py``), the population the bank's opponents came from
  F_rolls         F at the builder commit with ``dice="rolls"`` (F1 tier A): a world carries its rolls and log-weight

Per-world state lives on the world's own rng (``id(rng)``): ``build_base`` makes one rng per (world, try) and passes it
to the species draw first, then to every set and move draw of that world. A decision (corpus or backoff, which team)
is made from a PEEK of that rng's state (it is not advanced), so a backoff world is EXACTLY the B world of the same
(world, try).
"""
from __future__ import annotations

import contextlib
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s1_common as E  # noqa: E402
import corpus_s1 as CS  # noqa: E402

for _p in (E.T1,):
    if str(_p) not in sys.path:
        sys.path.insert(1, str(_p))

from agents.info_levers import mixture as MX  # noqa: E402
from agents.info_levers.packed import is_hp, parse_set, typed_moves  # noqa: E402
from main.value_search import world_draws as WD  # noqa: E402
from main.value_search import world_guard as WG  # noqa: E402
from main.value_search import world_legality as WL  # noqa: E402
from main.value_search import worlds as W  # noqa: E402

ALPHA_B = 0.05                       # (b)'s smoothing toward usage (stage 0's ALPHA, fixed before stage 0 ran)


class HookError(RuntimeError):
    """A hook found the builder in a state its contract excludes: a DEFECT (never an exclusion)."""


def canon(m: str) -> str:
    return "hiddenpower" if is_hp(m) else m


def peek(rng: np.random.Generator, salt: str) -> np.random.Generator:
    """A generator seeded by ``rng``'s CURRENT state and ``salt``, leaving ``rng`` untouched."""
    st = rng.bit_generator.state["state"]
    h = hashlib.sha256(f"{salt}:{st['state']}:{st['inc']}".encode()).hexdigest()[:16]
    return np.random.default_rng(int(h, 16))


# ------------------------------------------------------------------------------------------------
# (b): joint Smogon teammates
# ------------------------------------------------------------------------------------------------

class JointSmogon:
    """Stage 0's (b) over the builder's own candidates (``Prior.pool``: the same 216 species F draws from)."""

    def __init__(self, P: WD.Prior):
        self.pool = list(P.pool)
        self.ix = {s: i for i, s in enumerate(self.pool)}
        u = np.array([float(P.usage[s]) for s in self.pool])
        self.ubar = u / u.sum()
        n = len(self.pool)
        T = np.zeros((n, n))
        has = np.zeros(n, dtype=bool)
        for r in self.pool:
            t = P.mates(r)
            if t:
                has[self.ix[r]] = True
                for s, v in t.items():
                    if s in self.ix:
                        T[self.ix[r], self.ix[s]] = float(v)
        self.loglift = np.log((T + ALPHA_B * self.ubar[None, :]) / self.ubar[None, :])
        self.loglift[~has, :] = 0.0

    def species(self, rng, slots: Sequence[int], true_sid: Dict[int, str], revealed, source: str = "b"):
        present = set(revealed)
        lw = np.log(self.ubar) + sum((self.loglift[self.ix[r]] for r in present if r in self.ix),
                                     np.zeros(len(self.pool)))
        out: Dict[int, str] = {}
        prov: Dict[int, dict] = {}
        for s in slots:
            cand = [i for i, x in enumerate(self.pool) if x not in present]
            if not cand:
                return None
            m = max(lw[i] for i in cand)
            pick = MX.draw(rng, {self.pool[i]: float(np.exp(lw[i] - m)) for i in cand})
            if pick is None:
                return None
            present.add(pick)
            lw = lw + self.loglift[self.ix[pick]]
            out[s] = pick
            prov[s] = {"how": "drawn", "source": source, "match": pick == true_sid[s]}
        return out, prov


# ------------------------------------------------------------------------------------------------
# a source's moves on a SEEN mon (the Hidden Power is the screen's)
# ------------------------------------------------------------------------------------------------

def corpus_seen_moves(rng, cmoves: Sequence[str], prior, keep: Dict[int, str], n_slots: int, truth: Sequence[str],
                      hp_type: Optional[str], hp_rule: str, orig, counts: Counter, source: str):
    """A seen mon's free move slots from a source set ``cmoves`` (typed ids). The source's NON-Hidden-Power moves not
    already revealed fill the free slots; a Hidden Power is a spread fact: present iff the screen's IVs carry one
    (``hp_rule == "force"``, of ``hp_type``) or play revealed one. Too many source moves for the room: a uniform subset
    (``hp_added``: the screen added a Hidden Power the source set lacks); too few (``hp_dropped``: the source's Hidden
    Power the screen does not carry; or a short set): the rest by F's own draw (``orig``) around them."""
    free = [i for i in range(n_slots) if i not in keep]
    k = len(free)
    if k == 0:
        return orig(rng, prior, keep, n_slots, None, truth, hp_type, hp_rule=hp_rule)
    rev = set(keep.values())
    rev_c = {canon(m) for m in rev}
    has_hp_rev = any(is_hp(m) for m in rev)
    src_hp = any(is_hp(m) for m in cmoves)
    pool = [m for m in cmoves if not is_hp(m) and m not in rev_c and m not in WD.BANNED_MOVES]
    picks: List[str] = []
    if hp_rule == "force" and not has_hp_rev:
        if not hp_type:
            raise HookError("hp_rule force without a Hidden Power type")
        picks.append(f"hiddenpower{hp_type}")
        if not src_hp:
            counts["hp_added"] += 1
    elif src_hp and not has_hp_rev:
        counts["hp_dropped"] += 1
    room = k - len(picks)
    if len(pool) > room:
        sel = sorted(int(x) for x in rng.choice(len(pool), size=room, replace=False))
        pool = [pool[i] for i in sel]
        counts["moves_cut"] += 1
    picks += pool
    if len(picks) < k:
        counts["moves_filled"] += 1
        keep2 = dict(keep)
        for i, m in zip(free, picks):
            keep2[i] = m
        got = orig(rng, prior, keep2, n_slots, None, truth, hp_type, hp_rule=hp_rule)
        if got is None:
            return None
        slots = list(got[0])
    else:
        slots = [keep.get(i) for i in range(n_slots)]
        for i, m in zip(free, picks):
            slots[i] = m
    if len({canon(m) for m in slots}) != n_slots:
        raise HookError(f"a source move twice: {slots}")
    want = {canon(m) for m in truth if m not in rev}
    drawn = [slots[i] for i in free]
    hits = sum(canon(m) in want for m in drawn)
    match = sorted(canon(m) for m in slots) == sorted(canon(m) for m in truth)
    counts["seen_mons"] += 1
    return [str(m) for m in slots], {"how": "corpus", "source": source, "match": bool(match), "true_hits": hits,
                                     "expected_true_hits": None, "slots": k, "target_hits": 0}


def set_facts(entry: str, f_true: Optional[List[str]], source: str) -> dict:
    """A never-seen corpus set's fact records, every ``match`` flag recomputed by the guard's own function."""
    rec = WG._recomputed(parse_set(entry), f_true)
    out = {cls: {"how": "corpus", "source": source, "match": bool(rec[cls])} for cls in ("item", "ability", "spread")}
    out["moves"] = {"how": "corpus", "source": source, "match": bool(rec["moves"])}
    return out


# ------------------------------------------------------------------------------------------------
# the proposals (one arm's departures from F)
# ------------------------------------------------------------------------------------------------

class Proposal:
    """F's draws everywhere (``NotImplemented``); an arm overrides what it changes."""

    name = "F"

    def __init__(self):
        self.root: Optional[W.Root] = None
        self.key: Optional[str] = None
        self.world: Dict[int, tuple] = {}
        self.counts: Counter = Counter()
        self._true_sid: Dict[tuple, str] = {}

    def begin(self, root: "W.Root", key: str) -> None:
        self.root, self.key = root, key
        self.world = {}
        self.counts = Counter()
        self._true_sid = {}
        for f, sid in zip(root.sets, root.sids):
            t = tuple(typed_moves(f))
            self._true_sid[t] = None if t in self._true_sid else sid      # None: ambiguous (raises only if asked)
        self.on_root()

    def on_root(self) -> None:
        pass

    def seen_sid(self, truth: Sequence[str]) -> str:
        """The seen mon whose TRUE typed moves are ``truth`` (the builder's seen-mon loop passes ``moves_of[i]``)."""
        sid = self._true_sid.get(tuple(truth))
        if sid is None or self.root is None or sid not in self.root.kept:
            raise HookError(f"{self.key}: a seen-mon move draw whose true moves {list(truth)} name no single seen mon")
        return sid

    def species(self, rng, P, slots, true_sid, revealed, spec):
        return NotImplemented

    def never_seen_set(self, rng, P, sid, f_true, spec, legal, tally):
        return NotImplemented

    def never_seen_moves(self, rng, sid, f_true, prior, n_slots, truth, hp_type, orig):
        return NotImplemented

    def seen_moves(self, rng, prior, keep, n_slots, truth, hp_type, hp_rule, orig):
        return NotImplemented

    def report(self) -> dict:
        return dict(sorted(self.counts.items()))


class BProposal(Proposal):
    name = "B"

    def __init__(self, joint: JointSmogon):
        super().__init__()
        self.joint = joint

    def species(self, rng, P, slots, true_sid, revealed, spec):
        if spec.lever("species", "unseen") is not None:
            raise HookError("stage 1's arms are honest: no species lever")
        self.counts["worlds_b"] += 1
        return self.joint.species(rng, slots, true_sid, revealed)


class CorpusProposal(BProposal):
    """(c): a consistent corpus team with probability λ = n / (n + alpha), else a B world."""

    def __init__(self, joint: JointSmogon, corpus: CS.Corpus, alpha: float, strict: bool, name: str):
        super().__init__(joint)
        self.corpus, self.alpha, self.strict, self.name = corpus, float(alpha), strict, name
        self.cands: List[int] = []
        self.lam = 0.0
        self.state_meta: dict = {}

    def on_root(self) -> None:
        root = self.root
        assert root is not None
        kept = {s: set(m) for s, m in root.kept.items()}
        cands = self.corpus.consistent(kept)
        n_all = len(cands)
        n_near = 0
        if self.strict:
            near = self.corpus.near_copies({CS.mon_set(e) for e in root.entries})
            n_near = sum(1 for i in cands if i in near)
            cands = [i for i in cands if i not in near]
        self.cands = cands
        n = len(cands)
        self.lam = n / (n + self.alpha) if n else 0.0
        never = frozenset(s for s in root.sids if s not in root.kept)
        covers = sum(1 for i in cands if frozenset(self.corpus.sets[i]) - set(root.kept) == never)
        self.state_meta = {"n_consistent": n_all, "n_near_removed": n_near, "n": n, "lambda": round(self.lam, 6),
                           "covers_U": covers}

    def species(self, rng, P, slots, true_sid, revealed, spec):
        if spec.lever("species", "unseen") is not None:
            raise HookError("stage 1's arms are honest: no species lever")
        n = len(self.cands)
        if n:
            pk = peek(rng, f"ws1_corpus:{self.name}")
            if pk.random() < self.lam:
                idx = self.cands[int(pk.integers(n))]
                others = [s for s in self.corpus.sets[idx] if s not in self.root.kept]
                if len(others) != len(slots):
                    raise HookError(f"{self.key}: corpus team {idx} has {len(others)} unseen mons for {len(slots)} "
                                    "never-seen slots")
                perm = pk.permutation(len(others))
                out = {s: others[int(perm[k])] for k, s in enumerate(slots)}
                prov = {s: {"how": "drawn", "source": f"corpus:{self.name}", "match": out[s] == true_sid[s]}
                        for s in slots}
                self.world[id(rng)] = ("corpus", idx)
                self.counts["worlds_corpus"] += 1
                return out, prov
        self.world[id(rng)] = ("backoff",)
        self.counts["worlds_b"] += 1
        return self.joint.species(rng, slots, true_sid, revealed)

    def _idx(self, rng) -> Optional[int]:
        w = self.world.get(id(rng))
        if w is None:
            raise HookError(f"{self.key}: a draw on a world the species hook never saw")
        return w[1] if w[0] == "corpus" else None

    def never_seen_set(self, rng, P, sid, f_true, spec, legal, tally):
        idx = self._idx(rng)
        if idx is None:
            return NotImplemented
        e = self.corpus.entry(idx, sid)
        if legal is not None and not legal(e):
            raise HookError(f"{self.key}: corpus set {e!r} fails the legality rule (the corpus was pre-validated)")
        self.counts["never_seen_sets"] += 1
        return e, set_facts(e, f_true, f"corpus:{self.name}")

    def seen_moves(self, rng, prior, keep, n_slots, truth, hp_type, hp_rule, orig):
        idx = self._idx(rng)
        if idx is None:
            return NotImplemented
        cm = typed_moves(parse_set(self.corpus.entry(idx, self.seen_sid(truth))))
        return corpus_seen_moves(rng, cm, prior, keep, n_slots, truth, hp_type, hp_rule, orig, self.counts,
                                 f"corpus:{self.name}")

    def report(self) -> dict:
        return {**self.state_meta, **super().report()}


class NSProposal(Proposal):
    """The network's species (FN's rule) × the forum's per-species sets (λ_s = n_s / (n_s + 1)) for every mon's moves."""

    name = "NS"

    def __init__(self, beliefs: Dict[str, dict], forum: CS.Corpus):
        super().__init__()
        self.beliefs, self.forum = beliefs, forum

    def species(self, rng, P, slots, true_sid, revealed, spec):
        if spec.lever("species", "unseen") is not None:
            raise HookError("stage 1's arms are honest: no species lever")
        import fn_worlds as FW

        row = self.beliefs.get(self.key)
        if row is None:
            raise W.WorldError(f"{self.key}: no 225M network-belief row")
        pi = {s: float(v) for s, v in row["species"].items()}
        present = set(revealed)
        out: Dict[int, str] = {}
        prov: Dict[int, dict] = {}
        for s in slots:
            cand = [x for x in P.pool if x not in present]
            w = {x: pi[x] for x in cand if pi.get(x, 0.0) > 0}
            src = "net"
            if not w:
                w = FW._smogon_weights(P, cand, revealed)
                src = "smogon_fallback"
            self.counts[f"species_{src}"] += 1
            pick = MX.draw(rng, w)
            if pick is None:
                return None
            present.add(pick)
            out[s] = pick
            prov[s] = {"how": "drawn", "source": f"NS:{src}", "match": pick == true_sid[s]}
        return out, prov

    def _sets(self, sid: str, rev_c: set) -> list:
        return [(mv, e) for mv, e in self.forum.species_sets.get(sid, ()) if rev_c <= mv]

    def never_seen_moves(self, rng, sid, f_true, prior, n_slots, truth, hp_type, orig):
        sets = self._sets(sid, set())
        n = len(sets)
        pk = peek(rng, f"ws1_ns_never:{sid}")
        if not n or pk.random() >= n / (n + 1.0):
            self.counts["never_moves_F"] += 1
            return NotImplemented
        cm = typed_moves(parse_set(sets[int(pk.integers(n))][1]))
        if hp_type is not None:
            raise HookError("an honest never-seen draw with a fixed Hidden Power type")
        if len(cm) > n_slots:
            cm = [cm[i] for i in sorted(int(x) for x in rng.choice(len(cm), size=n_slots, replace=False))]
        self.counts["never_moves_forum"] += 1
        if len(cm) < n_slots:
            got = orig(rng, prior, {i: m for i, m in enumerate(cm)}, n_slots, None, truth, None)
            if got is None:
                return None
            cm = list(got[0])
        want = {canon(m) for m in truth}
        hits = sum(canon(m) in want for m in cm)
        return [str(m) for m in cm], {"how": "forum_set", "source": "forum", "match": hits == n_slots and bool(truth),
                                      "true_hits": hits, "expected_true_hits": None, "slots": n_slots,
                                      "target_hits": 0}

    def seen_moves(self, rng, prior, keep, n_slots, truth, hp_type, hp_rule, orig):
        assert self.root is not None
        sid = self.seen_sid(truth)
        rev_c = {canon(m) for m in self.root.kept.get(sid, ())}
        sets = self._sets(sid, rev_c)
        n = len(sets)
        pk = peek(rng, f"ws1_ns_seen:{sid}")
        if not n or pk.random() >= n / (n + 1.0):
            self.counts["seen_moves_F"] += 1
            return NotImplemented
        cm = typed_moves(parse_set(sets[int(pk.integers(n))][1]))
        self.counts["seen_moves_forum"] += 1
        return corpus_seen_moves(rng, cm, prior, keep, n_slots, truth, hp_type, hp_rule, orig, self.counts, "forum")


# ------------------------------------------------------------------------------------------------
# the hooks
# ------------------------------------------------------------------------------------------------

_ORIG = {"nss": W.never_seen_species, "new_set": W.new_set, "draw": WD.draw_moves, "root": W.prepare_root}
_CTX: dict = {"prop": None, "never": None}


def _h_root(log, at, side, key, root_choice_in_log=False):
    r = _ORIG["root"](log, at, side, key, root_choice_in_log)
    p = _CTX["prop"]
    if p is not None:
        p.begin(r, key)
    return r


def _h_nss(rng, P, slots, true_sid, revealed, spec):
    p = _CTX["prop"]
    got = p.species(rng, P, slots, true_sid, revealed, spec) if p is not None else NotImplemented
    return _ORIG["nss"](rng, P, slots, true_sid, revealed, spec) if got is NotImplemented else got


def _h_new_set(rng, P, sid, f_true, spec, legal=None, tally=None):
    p = _CTX["prop"]
    if p is not None:
        got = p.never_seen_set(rng, P, sid, f_true, spec, legal, tally)
        if got is not NotImplemented:
            return got
    prev = _CTX["never"]
    _CTX["never"] = (sid, f_true)
    try:
        return _ORIG["new_set"](rng, P, sid, f_true, spec, legal=legal, tally=tally)
    finally:
        _CTX["never"] = prev


def _h_draw(rng, prior, keep, n_slots, lever, truth, hp_type, hp_rule="any"):
    p = _CTX["prop"]
    orig = _ORIG["draw"]
    if p is not None and lever is None:
        if hp_rule == "any" and _CTX["never"] is not None and not keep:
            sid, f_true = _CTX["never"]
            got = p.never_seen_moves(rng, sid, f_true, prior, n_slots, truth, hp_type, orig)
        elif hp_rule in ("force", "none"):
            got = p.seen_moves(rng, prior, keep, n_slots, truth, hp_type, hp_rule, orig)
        else:
            got = NotImplemented
        if got is not NotImplemented:
            return got
    return orig(rng, prior, keep, n_slots, lever, truth, hp_type, hp_rule=hp_rule)


@contextlib.contextmanager
def hooks(prop: Optional[Proposal]):
    """Install ``prop``'s departures for the length of ONE ``WorldSampler.sample`` call (None: F, nothing swapped)."""
    if prop is None:
        yield
        return
    if _CTX["prop"] is not None:
        raise HookError("hooks are not re-entrant")
    _CTX["prop"], _CTX["never"] = prop, None
    W.prepare_root, W.never_seen_species, W.new_set, WD.draw_moves = _h_root, _h_nss, _h_new_set, _h_draw
    try:
        yield
    finally:
        W.prepare_root, W.never_seen_species, W.new_set = _ORIG["root"], _ORIG["nss"], _ORIG["new_set"]
        WD.draw_moves = _ORIG["draw"]
        _CTX["prop"], _CTX["never"] = None, None


class _AlwaysLegal:
    """The pre-F2 rule (no validator): every set and team legal (F2's ``rebuild_worlds.py --mode off``)."""

    def problems(self, teams):
        return [() for _ in teams]


class Arms:
    """Every arm's sampler for one builder process (one Prior, one playout library, the corpora loaded once)."""

    def __init__(self, alpha: float, beliefs: Dict[str, dict]):
        from utils.rust_env import ffi as F
        from utils.rust_env.build import ensure_built

        self.P = WD.Prior()
        self.lib = F.load(ensure_built("release")[1])
        pad = W._gender_pad()
        legal = WL.Legality(pad=pad)
        mk = dict(prior=self.P, reveal="off", max_tries=E.MAX_TRIES, playout_lib=self.lib)
        self.samplers = {
            "F_legal": W.WorldSampler(**mk),
            "F_off": W.WorldSampler(**mk, legality=WL.Legality(validator=_AlwaysLegal(), pad=pad)),
            "F_rolls": W.WorldSampler(**mk, dice="rolls"),
        }
        base = self.samplers["F_legal"]
        for arm in ("B", "C_ladder", "C_forum_strict", "NS"):
            self.samplers[arm] = base
        self.joint = JointSmogon(self.P)
        self.ladder = CS.load("ladder", legal, self.P)
        self.forum = CS.load("forum", legal, self.P)
        self.alpha = float(alpha)
        self.beliefs = beliefs
        import json

        import pool_teams as PT
        import pool_worlds as PW

        # C_pool's pool AS X26'S PIN PACKED IT (``pool_pin.py``): the builder commit re-packs 34 of the 719 teams with
        # another Hidden Power IV fill, and POOL gates a candidate whole
        blob = json.loads((E.ARCH / "pool" / "pool_pin.json").read_text())
        if hashlib.sha256("\n".join(blob["teams"]).encode()).hexdigest() != blob["sha256"]:
            raise HookError("pool_pin.json does not match its sha256")
        idx: Dict[str, List[int]] = {}
        for i, t in enumerate(blob["teams"]):
            idx.setdefault(t, []).append(i)
        self.pool_pin = PT.Pool(teams=list(blob["teams"]), sha256=blob["sha256"], index_of=idx)
        self.pool = PW.PoolSampler(max_tries=E.MAX_TRIES, pool=self.pool_pin, ws=base)

    def proposal(self, arm: str) -> Optional[Proposal]:
        if arm == "B":
            return BProposal(self.joint)
        if arm == "C_ladder":
            return CorpusProposal(self.joint, self.ladder, self.alpha, strict=False, name="ladder")
        if arm == "C_forum_strict":
            return CorpusProposal(self.joint, self.forum, self.alpha, strict=True, name="forum_strict")
        if arm == "NS":
            return NSProposal(self.beliefs, self.forum)
        return None

    def sample(self, arm: str, log: dict, at: int, side: str, key: str, cache: dict):
        """(``worlds.Worlds``, the proposal's report or None)."""
        if arm == "C_pool":
            return self.pool.sample(log, at, side, E.J, key=key, screen_cache=cache), None
        prop = self.proposal(arm)
        with hooks(prop):
            w = self.samplers[arm].sample(log, at, side, E.J, key=key, screen_cache=cache)
        return w, (prop.report() if prop is not None else None)
