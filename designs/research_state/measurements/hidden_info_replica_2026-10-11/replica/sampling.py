"""The Smogon prior, filtered to what the pinned engine can run and the gen3ou clauses allow, and the draws.

Every draw here reads ``agents.gen3_data.priors`` at the PIN (the caller puts the pin's ``src/`` first on the
path). Nothing here reads a true team: the builder hands in only what the searched side has seen.
"""
from __future__ import annotations

import functools
import itertools
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.gen3_data import format_spec as fs
from agents.gen3_data import learnset, priors
from agents.gen3_data import species as species_dex
from agents.gen3_data import team_legality

HERE = Path(__file__).resolve().parent.parent

#: Showdown's canonical Hidden Power IVs (``data/typechart.ts`` ``HPivs``; unlisted stats are 31).
HP_IVS = {
    "bug": {"atk": 30, "def": 30, "spd": 30}, "dark": {}, "dragon": {"atk": 30}, "electric": {"spa": 30},
    "fighting": {"def": 30, "spa": 30, "spd": 30, "spe": 30}, "fire": {"atk": 30, "spa": 30, "spe": 30},
    "flying": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30}, "ghost": {"def": 30, "spd": 30},
    "grass": {"atk": 30, "spa": 30}, "ground": {"spa": 30, "spd": 30}, "ice": {"atk": 30, "def": 30},
    "poison": {"def": 30, "spa": 30, "spd": 30}, "psychic": {"atk": 30, "spe": 30},
    "rock": {"def": 30, "spd": 30, "spe": 30}, "steel": {"spd": 30}, "water": {"atk": 30, "def": 30, "spa": 30},
}
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def _load_probe(kind: str) -> set:
    bad = set()
    for line in (HERE / f"engine_probe_{kind}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["verdict"] != "ran":
                bad.add(r.get("id") or r.get("move"))
    return bad


ENGINE_BAD = {k: _load_probe(k) for k in ("move", "item", "ability", "species")}
BANNED: Dict[str, set] = {"species": set(), "item": set(), "move": set(), "ability": set()}
BANNED_SPECIES_MOVE: set = set()
for _b in fs.iter_bans():
    if not _b.on_pinned:
        continue
    if _b.kind in BANNED:
        BANNED[_b.kind].add(_b.ids[0])
    elif _b.kind == "species_move":
        BANNED_SPECIES_MOVE.add(tuple(_b.ids))


def is_hp(move_id: str) -> bool:
    return move_id.startswith("hiddenpower")


def hp_type(move_id: str) -> Optional[str]:
    return move_id[len("hiddenpower"):] or None if is_hp(move_id) else None


def move_ok(species: str, m: str) -> bool:
    """Engine-supported, not banned (alone or for this species), learnable (a typed Hidden Power is learnable
    iff Hidden Power is)."""
    base = "hiddenpower" if is_hp(m) else m
    return (m not in ENGINE_BAD["move"] and base not in ENGINE_BAD["move"] and base not in BANNED["move"]
            and (species, base) not in BANNED_SPECIES_MOVE and learnset.is_legal(species, base))


def base_of(species: str) -> str:
    sd = species_dex.get(species)
    return (sd.base_species or species) if sd is not None else species


# ------------------------------------------------------------------------------------------ conditional Poisson

def _esp(w: np.ndarray, k: int) -> np.ndarray:
    """Elementary symmetric polynomials e_0..e_k of ``w``."""
    e = np.zeros(k + 1)
    e[0] = 1.0
    for x in w:
        e[1:] = e[1:] + x * e[:-1]
    return e


def cp_inclusion(w: np.ndarray, k: int) -> np.ndarray:
    """Inclusion probabilities of the size-``k`` conditional-Poisson design with weights ``w``:
    pi_i = w_i e_{k-1}(w without i) / e_k(w), with e_j(w without i) = e_j(w) - w_i e_{j-1}(w without i)."""
    e = _esp(w, k)
    em = np.ones_like(w)            # e_0(w without i)
    for j in range(1, k):
        em = e[j] - w * em
    return w * em / e[k]


def cp_weights(pi: np.ndarray, k: int, iters: int = 500) -> np.ndarray:
    """Working weights w of a conditional-Poisson design of size ``k`` whose inclusion probabilities are ``pi``
    (0 < pi < 1, sum = k), by the fixed-point iteration w <- w * pi / pi(w) (converged to 1e-10)."""
    w = pi / (1.0 - pi)
    for _ in range(iters):
        incl = cp_inclusion(w, k)
        if np.max(np.abs(incl - pi)) < 1e-10:
            break
        w = w * pi / incl
        w = w / np.exp(np.mean(np.log(w)))
    return w


def cap_mass(p: Dict[str, float], k: int) -> Tuple[List[str], Dict[str, float]]:
    """Renormalise ``p`` to mass ``k``; entries that would exceed 1 become CERTAIN (returned first) and the rest is
    renormalised to the remaining mass, repeatedly."""
    sure: List[str] = []
    rest = {m: v for m, v in p.items() if v > 0}
    while True:
        kk = k - len(sure)
        if kk <= 0 or not rest:
            return sure, {}
        tot = sum(rest.values())
        if len(rest) <= kk:
            return sure + sorted(rest, key=lambda m: -rest[m]), {}
        scaled = {m: v * kk / tot for m, v in rest.items()}
        over = [m for m, v in scaled.items() if v >= 1.0 - 1e-12]
        if not over:
            return sure, scaled
        for m in sorted(over, key=lambda m: -scaled[m]):
            sure.append(m)
            rest.pop(m)


def cp_sample(p: Dict[str, float], k: int, rng: np.random.Generator) -> List[str]:
    """A size-``k`` subset with inclusion probabilities = ``p`` renormalised to mass ``k`` (conditional Poisson).
    Fewer than ``k`` candidates: all of them."""
    sure, rest = cap_mass(p, k)
    kk = k - len(sure)
    if kk <= 0 or not rest:
        return sure[:k]
    names = sorted(rest)
    key = (tuple(names), tuple(round(rest[m], 15) for m in names), kk)
    w = _CP_CACHE.get(key)
    if w is None:
        w = _CP_CACHE[key] = cp_weights(np.array([rest[m] for m in names]), kk)
    n = len(names)
    # suffix ESPs: B[i, j] = e_j(w[i:])
    B = np.zeros((n + 1, kk + 1))
    B[n, 0] = 1.0
    for i in range(n - 1, -1, -1):
        B[i] = B[i + 1]
        B[i, 1:] += w[i] * B[i + 1, :-1]
    out: List[str] = []
    r = kk
    for i, m in enumerate(names):
        if r == 0:
            break
        if rng.random() < w[i] * B[i + 1, r - 1] / B[i, r]:
            out.append(m)
            r -= 1
    return sure + out


_CP_CACHE: Dict[tuple, np.ndarray] = {}


def weighted_pick(p: Dict[str, float], rng: np.random.Generator) -> str:
    names = sorted(p)
    w = np.array([p[n] for n in names], dtype=float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


# ------------------------------------------------------------------------------------------ per-class priors

def move_prior(species: str, exclude: Sequence[str], allow_hp: bool) -> Dict[str, float]:
    ex = set(exclude)
    out = {}
    for m, v in priors.moves(species).items():
        if m in ex or not move_ok(species, m) or (is_hp(m) and not allow_hp):
            continue
        out[m] = float(v)
    return out


def draw_moves(species: str, revealed: Sequence[str], k: int, rng: np.random.Generator,
               allow_hp: bool) -> List[str]:
    """``k`` unrevealed moves by conditional Poisson over the prior renormalised to mass ``k`` on the unrevealed
    candidates, at most one Hidden Power in the whole set (``allow_hp`` False when the set already holds one)."""
    cand = move_prior(species, revealed, allow_hp)
    for _ in range(500):
        got = cp_sample(cand, k, rng)
        if sum(1 for m in got if is_hp(m)) <= 1:
            return got
    got = [m for m in cp_sample(cand, k, rng) if not is_hp(m)]
    return got


def hp_type_prior(species: str) -> Dict[str, float]:
    p = {t: float(v) for t, v in priors.hidden_power(species).items() if t in HP_IVS
         and move_ok(species, "hiddenpower" + t)}
    if not p:
        p = {t: 1.0 for t in HP_IVS if move_ok(species, "hiddenpower" + t)}
    return p


def item_prior(species: str) -> Dict[str, float]:
    out = {}
    for it, v in priors.items(species).items():
        iid = "" if it == "nothing" else it
        if iid and (iid in ENGINE_BAD["item"] or iid in BANNED["item"]):
            continue
        out[iid] = out.get(iid, 0.0) + float(v)
    return out


def ability_prior(species: str) -> Dict[str, float]:
    return {a: float(v) for a, v in priors.ability(species).items()
            if a not in ENGINE_BAD["ability"] and a not in BANNED["ability"]}


def spread_prior(species: str) -> List[Tuple[str, Tuple[int, ...], float]]:
    """EVERY Smogon chaos spread (``priors.all_spreads``: the full usage distribution; the top-25 file
    ``priors.spreads`` covers only ≈ 49 % of the bank's opponent mons, so its support would exclude half the
    truths)."""
    return [(n, tuple(int(x) for x in e), float(w)) for n, e, w in priors.all_spreads(species)]


#: Smogon publishes no IV distribution. The one IV convention that varies in gen-3 OU sets is the ATTACK IV of a
#: set that does not need it (0, or the lowest value with the same two low bits when Hidden Power needs them).
#: Our choice (spec ambiguity, reported): the attack IV is "max" or "min" with equal prior mass, screened by the
#: gate like every other S fact.
ATK_IV_VARIANTS = ("max", "min")


@functools.lru_cache(maxsize=None)
def iv_patterns(hp_t: Optional[str]) -> Tuple[Tuple[int, ...], ...]:
    """The max-power IV patterns of a Hidden Power type: every IV 30 or 31 (both keep the power bit, so the power is
    70) whose type bits give ``hp_t`` -- Showdown's canonical pattern first. Smogon publishes no IV usage; the pool
    teams use non-canonical patterns too (e.g. Hidden Power Ice as Spe 30 alone), and the gate is exact to one HP
    point, so the support is every pattern, uniformly (spec ambiguity, reported). No Hidden Power: all 31."""
    if hp_t is None:
        return ((31,) * 6,)
    canon = tuple(31 if s not in HP_IVS[hp_t] else HP_IVS[hp_t][s] for s in STATS)
    pats = [p for p in itertools.product((30, 31), repeat=6) if hp_type_of_ivs(p) == hp_t]
    return (canon,) + tuple(p for p in pats if p != canon)


def iv_keys(hp_t: Optional[str]) -> List[str]:
    """Every IV choice of the S class for a set with this Hidden Power type: attack variant x pattern."""
    pats = iv_patterns(hp_t)
    return [f"{v}|{','.join(map(str, p))}" for v in ATK_IV_VARIANTS for p in pats]


def iv_key_weight(hp_t: Optional[str]) -> float:
    return 1.0 / len(iv_keys(hp_t))


def ivs_for(hp_t: Optional[str], atk_variant: str) -> List[int]:
    """IVs for a type and an IV key (``"max"`` / ``"min"``, optionally ``"|<pattern>"``; a pattern of another
    type falls back to the canonical one)."""
    var, _, pat = atk_variant.partition("|")
    base = list(iv_patterns(hp_t)[0])
    if pat:
        p = [int(x) for x in pat.split(",")]
        if hp_t is None or hp_type_of_ivs(p) == hp_t:
            base = p
    iv = dict(zip(STATS, base))
    if var == "min":
        iv["atk"] = (iv["atk"] & 3) if hp_t is not None else 0
    return [iv[s] for s in STATS]


# ------------------------------------------------------------------------------------------ never-seen species

def teammate_prior(revealed_species: Sequence[str], exclude_bases: Sequence[str]) -> Dict[str, float]:
    """P(species) for a never-seen slot ∝ the MEAN over the revealed species of the Smogon teammate conditional
    (spec: "species ∝ the teammate prior given the revealed species"), restricted to species the engine runs, the
    format allows, the Species Clause allows (base species not already on the team) and the prior can build a set
    for (a move prior exists)."""
    ex = set(exclude_bases)
    acc: Dict[str, float] = {}
    rs = [s for s in revealed_species if priors.teammates(s)]
    for r in rs:
        for s, v in priors.teammates(r).items():
            acc[s] = acc.get(s, 0.0) + float(v) / len(rs)
    out = {}
    for s, v in acc.items():
        if (s in ENGINE_BAD["species"] or s in BANNED["species"] or base_of(s) in ex or not priors.moves(s)
                or not priors.ability(s) or not priors.items(s)):
            continue
        out[s] = v
    return out


# ------------------------------------------------------------------------------------------ a set

@dataclass
class MonSet:
    name: str
    species_name: str
    species: str
    item: str
    ability: str
    moves: List[str]
    nature: str
    evs: Tuple[int, ...]
    ivs: List[int]
    gender: str = ""
    tail: str = "||"            # shiny|level|happiness (blank = default)
    origin: Dict[str, str] = field(default_factory=dict)   # per class: 'revealed' / 'true' / 'drawn'

    def pack(self) -> str:
        sp = self.species_name if to_id_safe(self.species_name) != to_id_safe(self.name) else ""
        evs = ",".join("" if v == 0 else str(v) for v in self.evs)
        ivs = ",".join("" if v == 31 else str(v) for v in self.ivs)
        if all(v == 31 for v in self.ivs):
            ivs = ""
        if all(v == 0 for v in self.evs):
            evs = ""
        return f"{self.name}|{sp}|{self.item}|{self.ability}|{','.join(self.moves)}|{self.nature}|{evs}|" \
               f"{self.gender}|{ivs}|{self.tail}"

    def copy(self) -> "MonSet":
        return replace(self, moves=list(self.moves), ivs=list(self.ivs), origin=dict(self.origin))


def to_id_safe(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def unpack(packed: str) -> List[MonSet]:
    out = []
    for chunk in packed.split("]"):
        f = chunk.split("|")
        while len(f) < 12:
            f.append("")
        name = f[0]
        spn = f[1] or f[0]
        evs = tuple(int(x) if x else 0 for x in (f[6].split(",") if f[6] else [""] * 6))
        ivs = [int(x) if x else 31 for x in (f[8].split(",") if f[8] else [""] * 6)]
        out.append(MonSet(name=name, species_name=spn, species=to_id_safe(spn), item=f[2], ability=f[3],
                          moves=[m for m in f[4].split(",") if m], nature=f[5], evs=evs, ivs=ivs, gender=f[7],
                          tail="|".join(f[9:12])))
    return out


def pack_team(sets: Sequence[MonSet]) -> str:
    return "]".join(s.pack() for s in sets)


def legality(sets: Sequence[MonSet], truth: Optional[Sequence[MonSet]] = None) -> List[str]:
    """Every gen3ou problem the PINNED format enforces (team_legality, ladder-only rules excluded), plus ≤ 1
    Hidden Power per set, learnset legality, Hidden Power type = the IVs' type, and engine support. With
    ``truth``, a problem the TRUE team has too (same rule, slot and fact, e.g. a revealed move our learnset table
    calls illegal) is the battle's own and is not held against the world."""
    if truth is not None:
        own = set(legality(truth))
        return [p for p in legality(sets) if p not in own]
    probs = [f"{p.rule}:{p.slot}" for p in team_legality.validate_team(team_legality.parse_packed(pack_team(sets)))
             if not p.ladder_only]
    for i, s in enumerate(sets):
        hps = [m for m in s.moves if is_hp(m)]
        if len(hps) > 1:
            probs.append(f"two_hp:{i}")
        for m in s.moves:
            if m in ENGINE_BAD["move"]:
                probs.append(f"engine_move:{i}:{m}")
            if not learnset.is_legal(s.species, "hiddenpower" if is_hp(m) else m):
                probs.append(f"learnset:{i}:{m}")
        if s.item in ENGINE_BAD["item"]:
            probs.append(f"engine_item:{i}:{s.item}")
        if hps and hp_type(hps[0]) is not None:
            if hp_type_of_ivs(s.ivs) != hp_type(hps[0]):
                probs.append(f"hp_ivs:{i}")
        if len(set(s.moves)) != len(s.moves):
            probs.append(f"dup_move:{i}")
    return probs


_HP_ORDER = ["fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water", "grass",
             "electric", "psychic", "ice", "dragon", "dark"]


def hp_type_of_ivs(ivs: Sequence[int]) -> str:
    """Gen-3 Hidden Power type from the IVs (bits of hp, atk, def, spe, spa, spd)."""
    hp, atk, de, spa, spd, spe = ivs
    bits = [hp & 1, atk & 1, de & 1, spe & 1, spa & 1, spd & 1]
    v = sum(b << i for i, b in enumerate(bits))
    return _HP_ORDER[v * 15 // 63]
