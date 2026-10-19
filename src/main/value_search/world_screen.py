"""THE SEEN-MON SCREEN: honest posterior draws of a SEEN opponent mon's item, ability and spread, conditioned on the
log the searched side saw (``designs/endstate/design_information_levers.md`` §7.2).

A seen mon's hidden facts shaped that log, so a draw counts only if the world it builds replays to the searched
side's exact protocol and rows (the sampler's GATE). One :class:`MonSampler` per (seen mon, constraint) of a decision
serves every world that needs it (a constraint: the facts a lever told, the Hidden Power type the world's moves fix,
an exact-HP requirement). The screening world holds every OTHER opponent mon at its true facts (the factorisation of
§7.2; the sampler's joint gate re-checks the combined world).

THE PRIOR (the honest belief before the log): item and ability ∝ the Smogon usage; the spread (nature, EVs) ∝
  π(s) = (1 − ε)·J(s) + ε·N(nature)·Π_stat M_stat(EV)
J the species' Smogon spread table (every chaos spread), N its nature marginal, M each stat's EV marginal smoothed
by a uniform floor (``FLOOR``) over 0, 4, …, 252, an EV total above 510 having no mass. J alone is a coarse histogram:
a mon whose log pins its bulk outside J's support (a pool spread Smogon never lists) has NO consistent J candidate,
and an honest player then believes the EVs that fit, so the smooth term gives every legal spread mass.

THE SAMPLER: (1) REJECTION — independent prior draws, each screened; an accepted one is an exact posterior draw.
(2) when fewer than needed are accepted, a Metropolis-Hastings CHAIN on the prior × the gate (local moves: EVs
shifted between / added to stats, a nature, an item, an ability; and independent prior draws), started at an accepted
draw (stationary from the start) or, when none was accepted, at the draw whose replay matched the log the longest,
hill-climbed on that matched length until it fits the log and then burnt in (``BURN``) before it is sampled every
``THIN`` steps. No step reads the truth. A sampler that cannot fit the log in ``MAX_STEPS`` fails (the world's try
fails, counted ``screen``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.info_levers import mixture as MX
from agents.info_levers.packed import (HP_TYPES, Spread, ability_id, hp_iv_masks, hp_type_of, item_id, set_ivs, toid,
                                       typed_moves)

EPS = 0.25
FLOOR = 0.15
EV_STEPS = tuple(range(0, 256, 4))
NATURES = ("hardy", "lonely", "brave", "adamant", "naughty", "bold", "docile", "relaxed", "impish", "lax", "timid",
           "hasty", "serious", "jolly", "naive", "modest", "mild", "quiet", "bashful", "rash", "calm", "gentle",
           "sassy", "careful", "quirky")
#: nature → (the stat it raises, the stat it lowers); the five neutral natures are absent
NATURE_EFFECT = {"lonely": ("atk", "def"), "brave": ("atk", "spe"), "adamant": ("atk", "spa"), "naughty": ("atk", "spd"),
                 "bold": ("def", "atk"), "relaxed": ("def", "spe"), "impish": ("def", "spa"), "lax": ("def", "spd"),
                 "timid": ("spe", "atk"), "hasty": ("spe", "def"), "jolly": ("spe", "spa"), "naive": ("spe", "spd"),
                 "modest": ("spa", "atk"), "mild": ("spa", "def"), "quiet": ("spa", "spe"), "rash": ("spa", "spd"),
                 "calm": ("spd", "atk"), "gentle": ("spd", "def"), "sassy": ("spd", "spe"), "careful": ("spd", "spa")}
#: the prior on a Hidden Power's IV pattern: the canonical one (the most 31s) has this mass, the type's other 30/31
#: patterns share the rest (pool sets use non-canonical ones: the blind replica's note)
HP_CANON = 0.8
REJECT_ROUNDS = 2          # independent draws per needed sample, before a chain
MAX_STEPS = 200
SEARCH_WIDTH = 8
STALL = 15                 # search steps without progress before a restart from a fresh prior draw
BURN = 6
THIN = 2

Key = Tuple[object, ...]
State = Tuple[str, str, str, Tuple[int, ...], Optional[str]]     # item, ability, nature id, EVs, Hidden Power type


@dataclass(frozen=True)
class ScreenKey:
    """What a sampler is conditioned on besides the log: the seen mon's slot; its Hidden Power (``hp``: ``must`` — play
    revealed one, or a told move set holds one: its TYPE is drawn —, ``none`` — a told move set holds none —, or
    ``free`` — the prior decides whether it carries one and of which type; the IVs follow, and they shape the exact
    stats the log pins); the facts a lever FIXED (cls → value); the values a placebo EXCLUDES; the classes a map lever
    takes the mode of (``map``); and whether the exact HP must equal the truth's."""

    slot: int
    hp: str
    moves_told: bool = False          # the world's moves ARE the truth's (a mass-1 truth lever): legality is screened
    fixed: Tuple[Tuple[str, object], ...] = ()
    exclude: Tuple[Tuple[str, object], ...] = ()
    map: Tuple[str, ...] = ()
    need_hp: bool = False


class SpreadPrior:
    """π(nature, EVs) for one species (module docstring)."""

    def __init__(self, joint: Dict[Tuple[str, Tuple[int, ...]], float]):
        tot = sum(joint.values()) or 1.0
        self.joint = {k: v / tot for k, v in joint.items()}
        nat: Dict[str, float] = {}
        marg = [dict.fromkeys(EV_STEPS, 0.0) for _ in range(6)]
        for (n, evs), w in self.joint.items():
            nat[n] = nat.get(n, 0.0) + w
            for s, e in enumerate(evs):
                e4 = min(252, 4 * (int(e) // 4))
                marg[s][e4] = marg[s].get(e4, 0.0) + w
        self.nat = {n: (1 - FLOOR) * nat.get(n, 0.0) + FLOOR / len(NATURES) for n in NATURES}
        self.marg = [{e: (1 - FLOOR) * m[e] + FLOOR / len(EV_STEPS) for e in EV_STEPS} for m in marg]
        # the smooth term's draw is the product TRUNCATED to EV totals ≤ 510: its mass there (a fixed-seed estimate,
        # deterministic) normalises it, so `logp` is the density `draw` samples and an independence move's ratio is 1
        rng = np.random.default_rng(20261012)
        keys = [np.array(list(m.keys())) for m in self.marg]
        probs = [np.array(list(m.values())) / sum(m.values()) for m in self.marg]
        tot = sum(rng.choice(k, size=4000, p=pr) for k, pr in zip(keys, probs))
        self.z_smooth = max(float(np.mean(tot <= 510)), 1e-6)

    def logp(self, nature: str, evs: Tuple[int, ...]) -> float:
        if sum(evs) > 510 or any(e < 0 or e > 255 for e in evs) or (any(e > 252 for e in evs)
                                                                   and (nature, evs) not in self.joint):
            return -math.inf
        smooth = self.nat.get(nature, 0.0) * math.prod(self.marg[s].get(min(252, 4 * (e // 4)), 0.0)
                                                         for s, e in enumerate(evs))
        p = (1 - EPS) * self.joint.get((nature, evs), 0.0) + EPS * smooth / self.z_smooth
        return math.log(p) if p > 0 else -math.inf

    def draw(self, rng: np.random.Generator) -> Tuple[str, Tuple[int, ...]]:
        if rng.random() >= EPS:
            got = MX.draw(rng, self.joint)
            if got is not None:
                return got
        for _ in range(100):
            n = MX.draw(rng, self.nat) or "hardy"
            evs = tuple(int(MX.draw(rng, m) or 0) for m in self.marg)
            if sum(evs) <= 510:
                return n, evs
        return "hardy", (84,) * 6


@dataclass
class MonSampler:
    """Posterior draws of one seen mon's (item, ability, spread) under one :class:`ScreenKey`."""

    key: ScreenKey
    sid: str
    f_true: List[str]
    items: Dict[str, float]
    abilities: Dict[str, float]
    spreads: SpreadPrior
    hp_prior: Dict[str, float]           # P(type | the set holds a Hidden Power)
    hp_any: float                        # P(the set holds a Hidden Power)
    rng: np.random.Generator
    needed: int
    samples: List[State] = field(default_factory=list)
    state: Optional[State] = None
    state_logp: float = -math.inf
    consistent: bool = False
    best_len: float = -1.0               # the current search state's score (``divergence_score``)
    stall: int = 0
    restarts: int = 0
    best_ever: Optional[Tuple[float, State]] = None    # the search's best (score, state) across restarts
    hint: Optional[Tuple[str, int]] = None         # the stat the current search state's first difference implicates
    #: entry → is this mon's set gen3ou-legal (set by the sampler for a key whose moves are TOLD: a drawn item or
    #: ability that breaks a clause with the told moves is rejected without a replay; never for drawn moves, where it
    #: would filter by the truth's unrevealed moves)
    legal: Optional[Callable[[str], bool]] = None
    steps: int = 0
    burn_left: int = 0
    since: int = 0
    failed: bool = False
    proposals: int = 0
    best_map: Optional[Tuple[float, State]] = None
    #: the PROPOSAL stream's honesty, per drawn class: [proposals, equal to the truth, Σ the prior's mass on the truth]
    prop_stats: Dict[str, List[float]] = field(default_factory=dict)

    def note_proposal(self, s: State) -> None:
        """Count an independent prior draw against the truth (the leak audit: a proposal stream that peeks at the
        truth matches it more often than the prior's own mass on it)."""
        t = true_state(self.f_true)
        for cls, i, w in (("item", 0, self.items), ("ability", 1, self.abilities)):
            if self._fixed(cls) is not None:
                continue
            ex = self._excl(cls)
            tot = sum(v for k, v in w.items() if k != ex)
            st = self.prop_stats.setdefault(cls, [0, 0, 0.0])
            st[0] += 1
            st[1] += int(s[i] == t[i])
            st[2] += (w.get(t[i], 0.0) / tot) if tot > 0 and t[i] != ex else 0.0

    # ---------------------------------------------------------------- the prior
    def _fixed(self, cls: str):
        return dict(self.key.fixed).get(cls)

    def _excl(self, cls: str):
        return dict(self.key.exclude).get(cls)

    def logp(self, s: State) -> float:
        item, ab, nat, evs, hpt = s
        lp = 0.0
        if self._fixed("item") is None:
            w = self.items.get(item, 0.0)
            if w <= 0 or item == self._excl("item"):
                return -math.inf
            lp += math.log(w)
        if self._fixed("ability") is None:
            w = self.abilities.get(ab, 0.0)
            if w <= 0 or ab == self._excl("ability"):
                return -math.inf
            lp += math.log(w)
        if self._fixed("spread") is None:
            if self._excl("spread") == (nat, evs):
                return -math.inf
            lp += self.spreads.logp(nat, evs)
        if self._fixed("spread") is None:
            lp += self._hp_logp(hpt)
        return lp

    def _hp_logp(self, hpt: Optional[str]) -> float:
        if self.key.hp == "none":
            return 0.0 if hpt is None else -math.inf
        if hpt is None:
            return -math.inf if self.key.hp == "must" else math.log(max(1.0 - self.hp_any, 1e-12))
        ty, k = hp_split(hpt)
        assert ty is not None
        p = self.hp_prior.get(ty, 0.0) * (1.0 if self.key.hp == "must" else self.hp_any)
        return (math.log(p) + _mask_logp(ty, k)) if p > 0 else -math.inf

    def _hp_draw(self) -> Optional[str]:
        if self.key.hp == "none" or (self.key.hp == "free" and self.rng.random() >= self.hp_any):
            return None
        ty = MX.draw(self.rng, self.hp_prior) or HP_TYPES[-1]
        n = len(hp_iv_masks(ty))
        k = 0 if (n <= 1 or self.rng.random() < HP_CANON) else 1 + int(self.rng.integers(n - 1))
        return _hp_name(ty, k)

    def draw(self) -> State:
        r = self.rng
        item = self._fixed("item")
        if item is None:
            item = MX.draw(r, self.items, [self._excl("item")]) or ""
        ab = self._fixed("ability")
        if ab is None:
            ab = MX.draw(r, self.abilities, [self._excl("ability")]) or ""
        if self._fixed("spread") == "TRUE":
            _i, _a, nat, evs, hpt = true_state(self.f_true)
        else:
            for _ in range(20):
                nat, evs = self.spreads.draw(r)
                if (nat, evs) != self._excl("spread"):
                    break
            hpt = self._hp_draw()
        return str(item), str(ab), nat, tuple(evs), hpt

    def local(self, s: State) -> Tuple[State, bool]:
        """A local move and whether it is SYMMETRIC: EVs shifted between two stats or added / removed on one, a
        uniform nature or Hidden Power type (symmetric: Metropolis ratio = the prior ratio); an item or an ability
        redrawn from its own prior (an independence move on that coordinate: ratio 1)."""
        r = self.rng
        item, ab, nat, evs, hpt = s
        kinds = []
        if self._fixed("spread") is None:
            kinds += ["shift", "shift", "add", "nature"]
            if self.key.hp != "none":
                kinds.append("hp")
        if self._fixed("item") is None:
            kinds.append("item")
        if self._fixed("ability") is None and len(self.abilities) > 1:
            kinds.append("ability")
        if not kinds:
            return s, True
        k = kinds[int(r.integers(len(kinds)))]
        e = list(evs)
        if k == "shift":
            a, b = (int(x) for x in r.choice(6, 2, replace=False))
            d = 4 * int(r.choice([1, 2, 4, 8, 16, 32]))
            e[a] -= d
            e[b] += d
        elif k == "add":
            a = int(r.integers(6))
            e[a] += 4 * int(r.choice([1, 2, 4, 8, 16])) * (1 if r.random() < 0.5 else -1)
        elif k == "nature":
            nat = NATURES[int(r.integers(len(NATURES)))]
        elif k == "hp":                          # uniform over the allowed (type, IV pattern) values: symmetric
            opts: List[Optional[str]] = [_hp_name(ty, i) for ty in HP_TYPES for i in range(len(hp_iv_masks(ty)))]
            if self.key.hp == "free":
                opts.append(None)
            hpt = opts[int(r.integers(len(opts)))]
        elif k == "item":
            item = MX.draw(r, self.items, [self._excl("item")]) or item
        elif k == "ability":
            ab = MX.draw(r, self.abilities, [self._excl("ability")]) or ab
        return (item, ab, nat, tuple(e), hpt), k not in ("item", "ability")

    def guided(self, s: State, hint: Tuple[str, int]) -> State:
        """A move DIRECTED by the first difference (the search phase only, never a sampling step): ``power`` — the
        opponent's attacking stats (Atk / SpA EVs, a nature raising or lowering one, a Choice Band); ``bulk`` — its HP /
        Def / SpD EVs or a nature on Def / SpD; ``speed`` — its Spe EVs or a nature on Spe. The sign is the hint's (``divergence_hint``)."""
        r = self.rng
        item, ab, nat, evs, hpt = s
        kind, sign = hint
        # +1 more of the stat, -1 less, 0 unknown: a too-soft hit (power +1) wants MORE, a too-bulky mon (bulk +1) LESS
        more = (sign if kind == "power" else -sign) if sign != 0 else 0
        if self._fixed("spread") is not None:
            return self.local(s)[0]
        stats = {"power": (1, 3), "bulk": (0, 2, 4), "speed": (5,)}[kind]
        e = list(evs)
        d = 4 * int(r.choice([1, 2, 4, 8, 16]))
        roll = r.random()
        if roll < 0.6:
            a = int(r.choice(stats))
            step = d if more > 0 else -d if more < 0 else (d if r.random() < 0.5 else -d)
            e[a] = min(252, max(0, e[a] + step))
            if sum(e) > 508:                       # keep the budget: take it from a stat the hint does not name
                others = [i for i in range(6) if i not in stats and e[i] > 0]
                if others:
                    o = int(r.choice(others))
                    e[o] = max(0, e[o] - (sum(e) - 508))
        elif roll < 0.85 or kind != "power" or self._fixed("item") is not None:
            names = {1: "atk", 2: "def", 3: "spa", 4: "spd", 5: "spe"}
            want = names.get(int(r.choice([x for x in stats if x != 0] or [5])), "spe")
            up = (more > 0) if more != 0 else bool(r.random() < 0.5)
            opts = [nm for nm, (u, dn) in NATURE_EFFECT.items() if (u == want if up else dn == want)]
            if opts:
                nat = opts[int(r.integers(len(opts)))]
        else:
            if more > 0 and "choiceband" in self.items and self._excl("item") != "choiceband":
                item = "choiceband"
            elif more < 0 and item == "choiceband":
                item = MX.draw(r, self.items, ["choiceband", self._excl("item")]) or item
        return item, ab, nat, tuple(e), hpt

    # ---------------------------------------------------------------- the entry a state writes
    def entry(self, s: State, rev_hp_slot: Optional[int], f_base: Optional[Sequence[str]] = None) -> str:
        """The packed entry of state ``s`` written onto ``f_base`` (default: the TRUE entry; the world passes its own,
        which holds its drawn moves). A fact equal to the truth's keeps the true text; a TOLD-true spread is the true
        spread verbatim (its IVs included)."""
        item, ab, nat, evs, hpt = s
        f = list(self.f_true if f_base is None else f_base)
        t_spread = Spread.of(self.f_true)
        if item != item_id(self.f_true):
            f[2] = item
        if ab != ability_id(self.f_true):
            f[3] = ab
        if self._fixed("spread") == "TRUE":
            ivs = tuple(t_spread.ivs)
        else:
            ivs = ivs_of(hpt)
            sp = Spread(nat[:1].upper() + nat[1:], tuple(evs), ivs)
            if not sp.same(t_spread):
                sp.write(f)
        if rev_hp_slot is not None:
            mv = f[4].split(",")
            want = "hiddenpower" + hp_type_of(ivs)
            if typed_moves(self.f_true)[rev_hp_slot] != want:
                mv[rev_hp_slot] = want
            f[4] = ",".join(mv)
        if self._fixed("spread") != "TRUE":
            clear_stale_hp_type(f)
        return "|".join(f)

    def done(self) -> bool:
        return self.failed or len(self.samples) >= self.needed


def clear_stale_hp_type(f: List[str]) -> None:
    """Empty the packed set's explicit Hidden Power TYPE field (``misc``: happiness, hpType, …) when it disagrees with the
    set's IVs. Three pool sets carry one (Salamence ``Flying``, Salamence ``Grass``, Tyranitar ``Bug``); a world that
    copies the true set and draws another spread kept the TRUTH's field, so its set named two Hidden Power types:
    Showdown's validator rejects it (28 of F2's 581 illegal worlds) and Showdown's battle would play the FIELD's type
    (``pokemon.ts``: ``hpType = set.hpType || ivs``), where our Rust core plays the IVs'. The type is a SPREAD fact
    (§7.2 of the levers spec): the IVs decide it, so the stale field goes."""
    while len(f) < 12:
        f.append("")
    misc = f[11].split(",")
    if len(misc) > 1 and misc[1] and toid(misc[1]) != hp_type_of(set_ivs(f)):
        misc[1] = ""
        f[11] = ",".join(misc).rstrip(",")


def hp_split(hpt: Optional[str]) -> Tuple[Optional[str], int]:
    """``"fire/2"`` → (``fire``, 2); ``"fire"`` → (``fire``, 0: the canonical pattern); None → (None, 0)."""
    if hpt is None:
        return None, 0
    ty, _, k = hpt.partition("/")
    return ty, int(k) if k else 0


def ivs_of(hpt: Optional[str]) -> Tuple[int, ...]:
    """The IVs a state's Hidden Power fixes (all 31 when the set holds none). IVs below 30 are never drawn: a
    minimised Attack (0 Atk, or 2 / 3 under a Hidden Power) is taken as full Attack — the owner's ruling of
    2026-10-12 ("0 attack adds so little value … toss those in as full attack"); a decision whose log pins the Attack
    (a confusion self-hit, a Struggle endgame) stays an EXCLUSION, counted (design §7.5)."""
    ty, k = hp_split(hpt)
    if ty is None:
        return (31,) * 6
    masks = hp_iv_masks(ty)
    return tuple(masks[min(k, len(masks) - 1)])


def _hp_name(ty: str, k: int) -> str:
    return ty if k == 0 else f"{ty}/{k}"


def _mask_logp(ty: str, k: int) -> float:
    n = len(hp_iv_masks(ty))
    p = HP_CANON if k == 0 else (1.0 - HP_CANON) / max(1, n - 1)
    return math.log(p) if n > 1 or k == 0 else 0.0


def true_state(f: Sequence[str]) -> State:
    """The TRUE (item, ability, nature, EVs, Hidden Power) — None when the set holds no Hidden Power; its IV pattern's
    index when the true IVs are one of the type's 30/31 patterns (else the canonical one stands for them)."""
    sp = Spread.of(f)
    if not any(m.startswith("hiddenpower") for m in typed_moves(f)):
        return item_id(f), ability_id(f), toid(sp.nature), tuple(sp.evs), None
    ty = hp_type_of(sp.ivs)
    masks = hp_iv_masks(ty)
    k = masks.index(list(sp.ivs)) if list(sp.ivs) in masks else 0
    return item_id(f), ability_id(f), toid(sp.nature), tuple(sp.evs), _hp_name(ty, k)


#: ``gate(jobs)`` → per job (passed, matched length): a job is (opponent team text, slot whose exact HP must match).
Gate = Callable[[List[Tuple[str, int, bool]]], List[Tuple[bool, float, Optional[Tuple[str, int]]]]]


def run(samplers: List[MonSampler], rev_hp: Dict[int, Optional[int]], entries: List[str], gate: Gate) -> int:
    """Fill every sampler's ``samples`` (module docstring). ``rev_hp``: slot → its revealed bare Hidden Power's move
    slot. Returns the number of screening replays. Every replay of a step is ONE batched gate call."""
    replays = 0

    def jobs_of(cands: List[Tuple[MonSampler, State]]):
        out = []
        for m, s in cands:
            tt = list(entries)
            tt[m.key.slot] = m.entry(s, rev_hp.get(m.key.slot))
            out.append(("]".join(tt), m.key.slot, m.key.need_hp))
        return out

    def screened(cands: List[Tuple[MonSampler, State]]):
        """The gate's verdicts, an illegal set refused without a replay."""
        jobs = jobs_of(cands)
        legal = [m.legal is None or m.legal(j[0].split("]")[m.key.slot]) for (m, _), j in zip(cands, jobs)]
        got = iter(gate([j for j, ok in zip(jobs, legal) if ok]) if any(legal) else [])
        return [next(got) if ok else (False, -1.0, None) for ok in legal]

    # (1) REJECTION: independent prior draws (a map lever takes the most probable accepted state at the end)
    for _round in range(REJECT_ROUNDS):
        cands: List[Tuple[MonSampler, State]] = []
        for m in samplers:
            if m.done():
                continue
            for _ in range(m.needed - len(m.samples)):
                d = m.draw()
                m.note_proposal(d)
                cands.append((m, d))
        if not cands:
            break
        res = screened(cands)
        replays += len(cands)
        for (m, s), (ok, ln, hint) in zip(cands, res):
            m.proposals += 1
            if ok:
                if len(m.samples) < m.needed:
                    m.samples.append(s)
                if not m.consistent:
                    m.state, m.state_logp, m.consistent = s, m.logp(s), True
            elif not m.consistent and ln > m.best_len:
                m.state, m.best_len, m.hint = s, ln, hint
    for m in samplers:
        if m.done():
            continue
        if m.state is None:
            m.state = m.draw()
        m.state_logp = m.logp(m.state)
        m.burn_left = 0 if m.consistent else BURN
    # (2) the CHAIN: hill-climb on the matched length until the log fits, then Metropolis-Hastings, thinned
    while any(not m.done() for m in samplers):
        live = [m for m in samplers if not m.done()]
        cands = []
        sym: List[bool] = []
        for m in live:
            assert m.state is not None
            # searching (the log not yet fitted): SEARCH_WIDTH proposals a step, the best kept; sampling: one MH step
            for _ in range(1 if m.consistent else SEARCH_WIDTH):
                if not m.consistent and m.hint is not None and m.rng.random() < 0.5:
                    s, sy = m.guided(m.state, m.hint), False      # directed by the first difference (search only)
                elif m.rng.random() < 0.25:
                    s, sy = m.draw(), False            # an independence draw from the prior: MH ratio 1
                else:
                    s, sy = m.local(m.state)
                cands.append((m, s))
                sym.append(sy)
        res = screened(cands)
        replays += len(cands)
        best: Dict[int, Tuple[float, bool, State, Optional[Tuple[str, int]]]] = {}
        for (m, s), (ok, ln, hint), sy in zip(cands, res, sym):
            m.proposals += 1
            lp = m.logp(s)
            if not m.consistent:
                if lp > -math.inf and (ok or ln >= m.best_len):
                    cur = best.get(id(m))
                    if cur is None or (ok, ln) > (cur[1], cur[0]):
                        best[id(m)] = (ln, ok, s, hint)
            elif ok and lp > -math.inf and (not sy or math.log(max(m.rng.random(), 1e-300)) < lp - m.state_logp):
                m.state, m.state_logp = s, lp
        for m in live:
            m.steps += 1
            if not m.consistent:
                if id(m) in best:
                    ln, ok, s, hint = best[id(m)]
                    m.stall = 0 if (ok or ln > m.best_len) else m.stall + 1
                    m.state, m.best_len, m.state_logp, m.consistent, m.hint = s, ln, m.logp(s), ok, hint
                else:
                    m.stall += 1
                if not m.consistent and m.stall >= STALL:     # a local optimum: restart
                    m.restarts += 1
                    if m.best_ever is not None and m.restarts % 2 == 1:   # from the best state, shaken
                        st_ = m.best_ever[1]
                        for _ in range(3):
                            st_ = m.local(st_)[0]
                        m.state, m.best_len, m.stall, m.hint = st_, -1.0, 0, None
                    else:                                                  # from a fresh prior draw
                        m.state, m.best_len, m.stall, m.hint = m.draw(), -1.0, 0, None
                if not m.consistent and m.state is not None and (m.best_ever is None or m.best_len > m.best_ever[0]):
                    m.best_ever = (m.best_len, m.state)
            if m.consistent and (m.best_map is None or m.state_logp > m.best_map[0]):
                assert m.state is not None
                m.best_map = (m.state_logp, m.state)
            if m.consistent:
                if m.burn_left > 0:
                    m.burn_left -= 1
                else:
                    m.since += 1
                    if m.since >= THIN:
                        m.since = 0
                        assert m.state is not None
                        m.samples.append(m.state)
            if m.steps >= MAX_STEPS and not m.consistent:
                m.failed = True                    # never fitted the log: no honest draw found (terminal)
    for m in samplers:                      # a MAP lever: the most probable consistent state the sampler visited
        if m.key.map and m.samples:
            pool = [(m.logp(s), s) for s in m.samples] + ([m.best_map] if m.best_map is not None else [])
            top = max(pool, key=lambda x: x[0])[1]
            m.samples = [top] * len(m.samples)
    return replays


def facts_of(s: State, f_true: Sequence[str], how: Dict[str, Tuple[str, str]]) -> Dict[str, dict]:
    """Each screened fact's provenance (``how``: cls → (revealed / told / drawn, source)) and whether the world's
    value equals the truth's."""
    t_item, t_ab, t_nat, t_evs, _ = true_state(f_true)
    item, ab, nat, evs, hpt = s
    t_ivs = tuple(Spread.of(f_true).ivs)
    ivs = ivs_of(hpt)
    told_true = how["spread"] == ("told", "truth")
    match = {"item": item == t_item, "ability": ab == t_ab,
             "spread": told_true or (nat == t_nat and tuple(evs) == tuple(t_evs) and ivs == t_ivs)}
    return {c: {"how": how[c][0], "source": how[c][1], "match": bool(match[c])} for c in ("item", "ability", "spread")}
