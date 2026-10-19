"""THE WORLD'S FACTS under an information-lever spec: every hidden fact of the opponent's team, for one world.

The world sampler (``worlds.WorldSampler``) builds a world from the TRUE packed team; this module decides each hidden
fact of it, per the spec (``agents.info_levers``; ``designs/endstate/design_information_levers.md`` §7):

* a fact play REVEALED (``agents.info_levers.revealed``) is the truth's, always;
* a fact whose lever FIRES (probability = its mass, decided once per world and mon) is the lever's TARGET (the truth,
  a wrong-but-plausible placebo, or the honest prior's mode: ``agents.info_levers.mixture``);
* every other fact is drawn from the Smogon prior (``agents.gen3_data.priors``) — HONEST.

A SEEN mon's item, ability and spread (the spread = nature, EVs and IVs; the IVs carry the Hidden Power type) shaped
the history the searched side saw, so they are not drawn once: :class:`SeenProposer` PROPOSES them and the sampler
SCREENS each proposal by replay (the world is kept only if the searched side's protocol and rows stay byte-equal).
Its moves never executed (a used move is revealed), so they are drawn directly (the CP presence mixture). A NEVER-SEEN
mon never acted: its species and whole set are drawn directly.

Every fact's PROVENANCE is recorded (``revealed`` / ``told`` (+ source) / ``drawn``) with whether the world's value
equals the truth, for the leak audit.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from agents.info_levers import mixture as MX
from agents.info_levers.spec import InfoReveal, Lever

from agents.info_levers.packed import (  # noqa: F401  (re-exported for the sampler and its tests)
    HP_TYPES, STATS, PackedError, Spread, ability_id, hp_ivs, hp_type_of, is_hp, item_id, move_id, parse_set,
    require_known, set_evs, set_ivs, species_id, toid, typed_moves)

#: the engine's FAIL-LOUD moves and items (``scan_move_probe``, PROBE_KIND=move|item, 2026-10-11, at
#: 62085750 and 95d014fa): drawing one would panic the core.
BANNED_MOVES = frozenset("""attract bide blastburn bounce camouflage dig dive flatter fly followme foresight
frenzyplant grudge helpinghand hydrocannon ingrain lockon magiccoat magnitude metronome mindreader mirrormove
mist mudsport naturepower nightmare odorsleuth present psywave razorwind roleplay sketch skullbash skyattack
snore spite spitup stockpile swallow teleport triplekick watersport""".split())
BANNED_ITEMS = frozenset({"machobrace", "mail", "mentalherb", "shellbell"})


class WorldError(RuntimeError):
    pass


class Prior:
    """The Smogon prior through the ``agents.gen3_data`` facade (every draw an honest world makes)."""

    def __init__(self):
        from agents.gen3_data import priors, species

        self._p, self._s = priors, species
        self.usage = dict(priors.species_usage())
        self.pool = sorted(s for s in self.usage if priors.moves(s) and species.get(s) is not None)
        self._cache: Dict[Tuple[str, str], dict] = {}

    def _memo(self, what: str, sid: str, build) -> dict:
        k = (what, sid)
        if k not in self._cache:
            self._cache[k] = build()
        return self._cache[k]                    # read-only by convention: every caller copies before editing

    def name(self, sid: str) -> str:
        return self._s.get(sid).name

    def moves(self, sid: str) -> Dict[str, float]:
        return self._memo("moves", sid, lambda: {k: v for k, v in sorted(self._p.moves(sid).items())
                                                 if k not in BANNED_MOVES})

    def items(self, sid: str) -> Dict[str, float]:
        return self._memo("items", sid, lambda: {k: v for k, v in sorted(self._p.items(sid).items())
                                                 if k not in BANNED_ITEMS and k != "nothing"})

    def ability(self, sid: str) -> Dict[str, float]:
        return self._memo("ability", sid, lambda: dict(sorted(self._p.ability(sid).items())))

    def spreads(self, sid: str) -> Dict[Tuple[str, Tuple[int, ...]], float]:
        """{(nature id, EVs): weight} over EVERY chaos spread (not the top-25 cut: a lumpy read needs the tail)."""
        def build() -> dict:
            out: Dict[Tuple[str, Tuple[int, ...]], float] = {}
            for nat, evs, w in self._p.all_spreads(sid) or self._p.spreads(sid):
                k = (toid(nat), tuple(int(x) for x in evs))
                out[k] = out.get(k, 0.0) + float(w)
            return dict(sorted(out.items())) or {("hardy", (84,) * 6): 1.0}
        return self._memo("spreads", sid, build)

    def hidden_power(self, sid: str) -> Dict[str, float]:
        return self._memo("hp", sid, lambda: dict(sorted(self._p.hidden_power(sid).items()))
                          or {t: 1.0 for t in HP_TYPES})

    def mates(self, sid: str) -> Dict[str, float]:
        return self._p.teammates(sid)


def _nature_name(nid: str) -> str:
    return nid[:1].upper() + nid[1:] if nid else ""


# ------------------------------------------------------------------------------------------------
# provenance
# ------------------------------------------------------------------------------------------------

def _fact(how: str, match: bool, source: str = "") -> dict:
    return {"how": how, "source": source, "match": bool(match)}


def fires(rng: np.random.Generator, lever: Optional[Lever]) -> bool:
    """Does ``lever`` fire for this (world, mon)? Mass 1: always, no random number spent."""
    if lever is None:
        return False
    return lever.mass_bp >= 10_000 or rng.random() < lever.mass


# ------------------------------------------------------------------------------------------------
# the MOVES of a mon (seen: its unrevealed slots; never-seen: all four)
# ------------------------------------------------------------------------------------------------

def draw_moves(rng: np.random.Generator, prior: Dict[str, float], keep: Dict[int, str], n_slots: int,
               lever: Optional[Lever], truth: Sequence[str], hp_type: Optional[str],
               hp_rule: str = "any") -> Optional[Tuple[List[str], dict]]:
    """Fill the move slots not in ``keep`` (``n_slots`` in all): the CP presence mixture of the lever's target with
    the prior (``mixture.mix_presence``; p = 0 is the honest draw). ``truth`` = the true set's typed ids. The Hidden
    Power (at most one): ``hp_rule`` ``any`` — of ``hp_type`` when given, else any type; ``none`` — none (the
    world's IVs say the set holds none); ``force`` — exactly one, of ``hp_type`` (the world's IVs say it holds one).
    Returns the slots and the provenance (how many target / true moves the world holds, and the honest
    expectation of the true ones: their exact CP inclusion)."""
    spelled = {m: c for m in prior if (c := move_id(m)) != m}
    if spelled:      # a table naming a move by a name variant draws it TWICE beside its canonical spelling (2026-10-15)
        raise PackedError(f"the move table names moves by a name variant, not the canonical id: {spelled}")
    have = set(keep.values())
    free = [i for i in range(n_slots) if i not in keep]
    k = len(free)
    if k == 0:
        return [keep[i] for i in range(n_slots)], _fact("revealed", True)
    has_hp = any(is_hp(m) for m in have)

    def ok(m: str) -> bool:
        if m in have or m in BANNED_MOVES:
            return False
        if is_hp(m):
            if has_hp or m == "hiddenpower" or hp_rule == "none":
                return False
            return hp_type is None or m == f"hiddenpower{hp_type}"
        return True

    cand = {m: w for m, w in prior.items() if ok(m)}
    forced = f"hiddenpower{hp_type}" if (hp_rule == "force" and not has_hp and hp_type) else None
    unrev_true = [m for m in truth if m not in have]
    target: List[str] = []
    p = 0.0
    if lever is not None:
        target = [m for m in MX.set_target(lever.source, unrev_true, cand, k, rng, exclude=have, at_most_one=is_hp)
                  if ok(m)]
        p = lever.mass
    pres = MX.mix_presence(cand, target, p, k)
    if forced is not None:
        pres[forced] = 1.0                      # the IVs carry this Hidden Power: the set holds it surely
    keys = sorted(pres)
    for _ in range(50):
        got = MX.cp_sample(rng, keys, [pres[m] for m in keys], k)
        if got is None:
            return None
        if sum(is_hp(m) for m in got) <= 1:
            break
    else:
        return None
    slots = [keep.get(i) for i in range(n_slots)]
    for i, m in zip(free, got):
        slots[i] = m
    # a Hidden Power is ONE move whatever its type (the type is a SPREAD fact: the IVs): moves match modulo it
    def canon(m: str) -> str:
        return "hiddenpower" if is_hp(m) else m

    want = {canon(m) for m in unrev_true}
    hits = sum(canon(m) in want for m in got)
    incl = MX.cp_inclusion([pres[m] for m in keys], k)
    expect = float(sum(p_ for m, p_ in zip(keys, incl) if canon(m) in want))
    how = "told" if lever is not None else "drawn"
    return [str(m) for m in slots], {"how": how, "source": lever.source if lever else "", "match": hits == k,
                                     "true_hits": hits, "expected_true_hits": expect, "slots": k,
                                     "target_hits": sum(m in set(target) for m in got)}


def align_to_truth(slots: List[str], truth: Sequence[str], keep: Set[int]) -> List[str]:
    """The drawn moves re-ordered so every drawn move that is also a true move sits in its TRUE slot (the free slots
    only; ``keep`` slots stay): a world holding the true set is the true entry, slot for slot (the move order is the
    opponent's own action order)."""
    free = [i for i in range(len(slots)) if i not in keep]
    drawn = [slots[i] for i in free]
    out = list(slots)
    placed: Set[int] = set()
    left = list(drawn)
    for i in free:
        if i < len(truth) and truth[i] in left:
            out[i] = truth[i]
            left.remove(truth[i])
            placed.add(i)
    for i in free:
        if i not in placed:
            out[i] = left.pop(0)
    return out


# ------------------------------------------------------------------------------------------------
# NEVER-SEEN mons: the species, then the whole set
# ------------------------------------------------------------------------------------------------

def never_seen_species(rng: np.random.Generator, P: Prior, slots: Sequence[int], true_sid: Dict[int, str],
                       revealed: Set[str], spec: InfoReveal) -> Optional[Tuple[Dict[int, str], Dict[int, dict]]]:
    """Each never-seen slot's species: where the ``species:unseen`` lever fires, its target (this slot's TRUE
    species; a placebo drawn ∝ the teammate prior excluding every true never-seen species; the prior's mode); else ∝
    the Smogon teammate co-occurrence with the revealed species (species usage when none). Never a species already on
    the team. Fired slots are placed first, so a draw cannot take a told slot's truth."""
    lv = spec.lever("species", "unseen")
    fired = [s for s in slots if fires(rng, lv)]
    present = set(revealed)
    out: Dict[int, str] = {}
    prov: Dict[int, dict] = {}
    truths = {true_sid[s] for s in slots}

    def weights() -> Dict[str, float]:
        cand = [s for s in P.pool if s not in present]
        if revealed:
            w = {s: float(np.mean([P.mates(r).get(s, 0.0) for r in sorted(revealed)])) for s in cand}
            if sum(w.values()) > 0:
                return w
        return {s: float(P.usage.get(s, 0.0)) for s in cand}

    for s in fired + [s for s in slots if s not in fired]:
        if s in fired:
            assert lv is not None
            if lv.source == "truth" and true_sid[s] not in present:
                pick: Optional[str] = true_sid[s]
            else:
                pick = MX.categorical_target(lv.source, true_sid[s], weights(), rng,
                                             exclude=sorted(truths - {true_sid[s]}) if lv.source == "placebo" else ())
            how = "told"
        else:
            pick = MX.draw(rng, weights())
            how = "drawn"
        if pick is None:
            return None
        present.add(pick)
        out[s] = pick
        prov[s] = _fact(how, pick == true_sid[s], lv.source if (how == "told" and lv is not None) else "")
    return out, prov


#: draws of a never-seen mon's whole set before its try fails on legality (F2: the prior conditioned on legality)
SET_REDRAWS = 25


def new_set(rng: np.random.Generator, P: Prior, sid: str, f_true: Optional[List[str]],
            spec: InfoReveal, legal: Optional[Callable[[str], bool]] = None,
            tally: Optional[Dict[str, int]] = None) -> Optional[Tuple[str, dict]]:
    """A never-seen slot's whole set for species ``sid``. ``f_true``: the TRUE set when ``sid`` is one of the true
    team's never-seen mons (an ``unseen`` lever applies to it then), else None (every fact drawn).

    ``legal`` (F2, ``world_legality``): the set is REDRAWN until ``legal(entry)`` holds, up to ``SET_REDRAWS`` draws
    (then None, ``tally["illegal_exhausted"]``): the draw is P(set | species, legal), rejection on Showdown's own
    validator. Each lever's coin is flipped ONCE per mon, before the first draw (in the order the draw reads it), so a
    redraw never re-flips a lever and a lever's mass stays exact; a set legal at its first draw consumes the rng
    exactly as without the check. ``tally["illegal_set"]`` counts the rejected draws."""
    coins: Dict[str, Optional[Lever]] = {}

    def lever(cls: str) -> Optional[Lever]:
        if cls not in coins:
            lv = spec.lever(cls, "unseen") if f_true is not None else None
            coins[cls] = lv if fires(rng, lv) else None
        return coins[cls]

    for _ in range(SET_REDRAWS if legal is not None else 1):
        got = _one_set(rng, P, sid, f_true, spec, lever)
        if got is None or legal is None or legal(got[0]):
            return got
        if tally is not None:
            tally["illegal_set"] = tally.get("illegal_set", 0) + 1
    if tally is not None:
        tally["illegal_exhausted"] = tally.get("illegal_exhausted", 0) + 1
    return None


def _one_set(rng: np.random.Generator, P: Prior, sid: str, f_true: Optional[List[str]], spec: InfoReveal,
             lever: Callable[[str], Optional[Lever]]) -> Optional[Tuple[str, dict]]:
    """One draw of :func:`new_set` (``lever``: the mon's once-flipped lever coins)."""
    facts: Dict[str, dict] = {}

    lv_s = lever("spread")
    true_sp = Spread.of(f_true) if f_true is not None else None
    hp_fixed: Optional[str] = None
    if lv_s is not None and lv_s.source == "truth" and true_sp is not None:
        hp_fixed = hp_type_of(true_sp.ivs)
    truth_moves = typed_moves(f_true) if f_true is not None else []
    lv_m = spec.lever("moves", "unseen") if f_true is not None else None   # its mass is the presence mixture
    n_slots = len(truth_moves) if (lv_m is not None and lv_m.mass_bp >= 10_000 and lv_m.source == "truth") else 4
    got = draw_moves(rng, P.moves(sid), {}, n_slots, lv_m, truth_moves, hp_fixed)
    if got is None:
        return None
    moves, facts["moves"] = got
    if f_true is None:
        facts["moves"]["match"] = False
    items, abil = P.items(sid), P.ability(sid)
    t_item = item_id(f_true) if f_true is not None else None
    t_ab = ability_id(f_true) if f_true is not None else None
    vals = {}
    for cls, truth, w in (("item", t_item, items), ("ability", t_ab, abil)):
        lv = lever(cls)
        tgt = MX.categorical_target(lv.source, truth, w, rng) if lv is not None else None
        v = MX.mix_draw(rng, 1.0, tgt, w) if tgt is not None else MX.draw(rng, w)
        vals[cls] = v or ""
        facts[cls] = _fact("told" if lv is not None else "drawn", truth is not None and vals[cls] == truth,
                           lv.source if lv is not None else "")
    hp = [m for m in moves if is_hp(m)]
    if lv_s is not None and true_sp is not None:
        if lv_s.source == "truth":
            spread = true_sp
        else:
            key = MX.categorical_target(lv_s.source, true_sp.key(), P.spreads(sid), rng) or true_sp.key()
            spread = Spread(_nature_name(key[0]), key[1], tuple(hp_ivs(hp[0][len("hiddenpower"):])) if hp else (31,) * 6)
    else:
        key = MX.draw(rng, P.spreads(sid)) or ("hardy", (84,) * 6)
        spread = Spread(_nature_name(key[0]), key[1], tuple(hp_ivs(hp[0][len("hiddenpower"):])) if hp else (31,) * 6)
    facts["spread"] = _fact("told" if lv_s is not None else "drawn",
                            true_sp is not None and spread.key() == true_sp.key() and spread.ivs == true_sp.ivs,
                            lv_s.source if lv_s is not None else "")
    f = ["", "", "", "", "", "", "", "", "", "", "", ""]
    f[0] = P.name(sid)
    f[2] = "" if vals["item"] in ("", "nothing") else vals["item"]
    f[3] = vals["ability"]
    f[4] = ",".join(moves)
    spread.write(f)
    return "|".join(f), facts
