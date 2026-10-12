"""The FAIR-WORLD builder of the independent replication (design_hidden_info_binding.md §13.1, Amendment 1).

A WORLD is the true battle with the opponent's team replaced: every fact the searched side has NOT seen is
re-drawn (which facts, per arm, below), past opponent choices are re-rendered so the world can execute them, and the
world is ACCEPTED only if replaying its command log to the root gives the searched side byte-equal protocol lines
(everything it was sent) AND byte-equal observation rows (with mask and legal tokens) at every one of its decisions,
and the world team breaks no gen3ou rule the pinned format enforces.

Arms (the classes of §6.1: M = unseen move slots of seen mons + the type of a revealed bare Hidden Power,
U = never-seen slots, I / A / S = item / ability / nature-EVs-IVs of seen mons):

====== =========================== =================================================================
arm    sampled                     true
====== =========================== =================================================================
O      nothing (the true battle)   everything (the truth tool's own root, incl. a recorded pending
                                   opponent choice)
P      M, U                        I, A, S of seen mons
F      M, U, I, A, S               nothing hidden
F+M    U, I, A, S                  M
O-M    M                           U, I, A, S
ID     nothing (identity audit)    the true team pushed through THIS builder's rendering and gate
====== =========================== =================================================================

Choices made where the spec is silent (each reported as a spec ambiguity):
- the opponent's PENDING root-turn choice (recorded before the root in the log, not yet executed, so not seen) is
  DROPPED in every constructed world: the opponent's root decision is answered by the checkpoint's greedy policy
  in the world, identically across the searched side's actions. In O it stays (the referee's convention).
- team ORDER and move-slot ORDER are kept from the truth (a seen mon sits in its true slot, a revealed move in its
  true slot); slot positions are not a hidden-fact class of §6.1 and changing them would add a non-information
  difference between O and every world.
- a past opponent ``move X`` whose X is not in the world's set (an unexecuted choice of an unseen move) is
  rendered as the world's move in X's TRUE slot; a ``hiddenpower<t>`` as the world's Hidden Power; a
  ``switch S`` to a slot whose species was re-drawn as the world's mon in S's slot.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.battle.core_replay import RecordedBattle, run_core
from agents.gen3_data import moves as moves_dex
from agents.gen3_data import priors
from agents.gen3_data import species as species_dex
from main.policy_spectrum.replay import _viewer_decisions
from main.policy_spectrum.truth import cmd_index

from . import sampling as S
from . import view as V

ARMS = ("O", "P", "F", "F+M", "O-M", "ID")
SCREENED = ("F", "F+M")
MAX_TRIES = 60
N_WORLDS = 8


def rng_for(*key: str) -> np.random.Generator:
    h = hashlib.sha256(("hir_replica_v1:" + ":".join(key)).encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def gender_field(species: str, shown: Optional[str]) -> str:
    """A species with a fixed gender (or genderless) is packed blank; otherwise the shown gender (blank = none)."""
    return "" if species_dex.raw().get(species, {}).get("gender") else (shown or "")


_MOVE_IDS: Optional[set] = None


def move_ids() -> set:
    global _MOVE_IDS
    if _MOVE_IDS is None:
        _MOVE_IDS = set(moves_dex.raw().keys())
    return _MOVE_IDS


# ===================================================================================== the state

@dataclass
class Prep:
    sid: str
    side: str
    v: int
    opp: str
    at: int
    battle: object
    decision: dict
    prefix: List[List[str]]          # commands[:at] as the truth logged them
    pending: Optional[int]           # index in ``prefix`` of the opponent's pending root-turn choice, or None
    truth_lines: List[str]
    truth_rows: List[tuple]
    opp_active: Dict[int, str]       # prefix index of an opponent command -> the opponent's active mon's name
    seen: V.Seen
    true_opp: List[S.MonSet]
    own: List[S.MonSet]
    seen_slot: Dict[str, int]        # protocol name -> truth slot
    error: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    #: SEED ALIGNMENT (the construction-draw fix, see ``lcg_advance``): every world plays with EXPLICIT genders on
    #: both teams and the raw seed advanced past the truth's turn-0 gender draws, so the world's own
    #: composition never changes the dice. ``own_team``: the searched side's team with its realized genders.
    own_team: str = ""
    opp_real_gender: Dict[int, str] = field(default_factory=dict)   # truth slot -> realized gender ('' = none)
    world_seed: str = ""
    gender_draws: int = 0


def _recorded(b, opp: str, opp_team: str, commands: Sequence[Sequence[str]], own_team: Optional[str] = None,
              seed: Optional[str] = None) -> RecordedBattle:
    p1 = dict(b.p1)
    p2 = dict(b.p2)
    own = "p1" if opp == "p2" else "p2"
    for who, team in ((opp, opp_team), (own, own_team)):
        if team is not None:
            (p1 if who == "p1" else p2)["team"] = team
    return RecordedBattle(label=b.battle_id, format_id=b.format_id, seed=seed or b.seed, p1=p1, p2=p2,
                          commands=[list(c) for c in commands])


_LCG_A = 0x5D588B656C078965
_LCG_C = 0x269EC3
_M64 = (1 << 64) - 1


def lcg_advance(seed: str, k: int) -> str:
    """The gen-5 LCG seed ``"a,b,c,d"`` (16-bit words, most significant first) advanced by ``k`` draws (one
    ``next`` per ``sample``; ``src/rust_sim/src/prng/gen5.rs``)."""
    w = [int(x) for x in seed.split(",")]
    if len(w) != 4:
        raise ValueError(f"not a gen5 seed: {seed!r}")
    st = (w[0] << 48) | (w[1] << 32) | (w[2] << 16) | w[3]
    for _ in range(k):
        st = (st * _LCG_A + _LCG_C) & _M64
    return ",".join(str((st >> sh) & 0xFFFF) for sh in (48, 32, 16, 0))


def draws_gender(m: S.MonSet) -> bool:
    """Whether the turn-0 construction draws this mon's gender (blank packed gender, ratio'd species)."""
    return not m.gender and not species_dex.raw().get(m.species, {}).get("gender")


def realized_genders(res: dict, viewer: int) -> Dict[str, str]:
    """A viewer's own mons' realized genders, by name, from its FIRST request's details."""
    import json as _json
    for side, lines in res["chunks"]:
        if side != viewer:
            continue
        for ln in lines:
            if ln.startswith("|request|"):
                req = _json.loads(ln[len("|request|"):])
                out = {}
                for p in req.get("side", {}).get("pokemon", []):
                    det = p["details"].split(", ")
                    out[p["ident"].split(": ", 1)[1]] = next((x for x in det[1:] if x in ("M", "F")), "")
                if out:
                    return out
    return {}


def lines_of(res: dict, v: int) -> List[str]:
    return [ln for side, lines in res["chunks"] if side == v for ln in lines]


def rows_of(res: dict, v: int) -> List[tuple]:
    return [(e["obs"]["b64"], tuple(e["mask"]), tuple(sorted(e["tokens"].items())))
            for e in res["trackers"][v] if "obs" in e]


def prepare(bank, dec: dict, sid: str) -> Prep:
    d = dec[sid]
    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    v = 0 if side == "p1" else 1
    opp = "p2" if side == "p1" else "p1"
    at = cmd_index(b, side, d["n"])
    prefix = [list(c) for c in b.commands[:at]]
    opp_team = b.p2["team"] if opp == "p2" else b.p1["team"]
    res = run_core([_recorded(b, opp, opp_team, prefix)], trackers=True, obs=True)[0]
    if not res["ok"]:
        raise RuntimeError(f"{sid}: the truth prefix does not replay: {res.get('error')}")
    truth_lines = lines_of(res, v)
    truth_rows = rows_of(res, v)
    # the opponent's answered decisions <-> its CHOOSE commands, in order: who was active at each
    ov = 1 - v
    ods = [x for x in _viewer_decisions(res, ov) if x.choice is not None]
    opp_cmd_idx = [j for j, c in enumerate(prefix) if c[0] == opp]
    if len(ods) != len(opp_cmd_idx):
        raise RuntimeError(f"{sid}: {len(ods)} answered opponent decisions for {len(opp_cmd_idx)} commands")
    opp_active: Dict[int, str] = {}
    for j, od in zip(opp_cmd_idx, ods):
        act = [p for p in od.request.get("side", {}).get("pokemon", []) if p.get("active")]
        if act:
            opp_active[j] = act[0]["ident"].split(": ", 1)[1]
    # the pending root-turn choice: the LAST opponent command after the searched side's previous command whose
    # removal leaves the searched side's stream unchanged (it has not executed)
    pending = None
    own_idx = [j for j, c in enumerate(prefix) if c[0] == side]
    if opp_cmd_idx and (not own_idx or opp_cmd_idx[-1] > own_idx[-1]):
        j = opp_cmd_idx[-1]
        alt = prefix[:j] + prefix[j + 1:]
        r2 = run_core([_recorded(b, opp, opp_team, alt)], trackers=True, obs=True)[0]
        if r2["ok"] and lines_of(r2, v) == truth_lines and rows_of(r2, v) == truth_rows:
            pending = j
    own_team = b.p1["team"] if side == "p1" else b.p2["team"]
    own = S.unpack(own_team)
    own_ab = {m.name: S.to_id_safe(m.ability) for m in own}
    seen = V.parse(truth_lines, side, own_ab, move_ids(), lambda sp: set(priors.ability(sp)))
    true_opp = S.unpack(opp_team)
    # seed alignment: count the truth's turn-0 gender draws (both teams), read every realized gender
    g_own = realized_genders(res, v)
    g_opp = realized_genders(res, ov)
    n_draws = sum(1 for m in own + true_opp if draws_gender(m))
    own_x = [m.copy() for m in own]
    for m in own_x:
        m.gender = gender_field(m.species, g_own.get(m.name, ""))
    prep = Prep(sid=sid, side=side, v=v, opp=opp, at=at, battle=b, decision=d, prefix=prefix, pending=pending,
                truth_lines=truth_lines, truth_rows=truth_rows, opp_active=opp_active, seen=seen,
                true_opp=true_opp, own=own, seen_slot={}, own_team=S.pack_team(own_x),
                opp_real_gender={i: gender_field(t.species, g_opp.get(t.name, "")) for i, t in enumerate(true_opp)},
                world_seed=lcg_advance(b.seed, n_draws), gender_draws=n_draws)
    for name, m in seen.mons.items():
        slots = [i for i, t in enumerate(true_opp) if t.name == name]
        if len(slots) != 1 or true_opp[slots[0]].species != m.species:
            prep.error = f"seen mon {name!r} ({m.species}) matches truth slots {slots}"
            return prep
        prep.seen_slot[name] = slots[0]
    return prep


# ===================================================================================== rendering and the gate

def render(prep: Prep, world: List[S.MonSet], keep_pending: bool = False) -> Tuple[List[List[str]], int]:
    """The world's command log to the root: the opponent's past choices re-rendered for the world's team."""
    out: List[List[str]] = []
    changed = 0
    for j, c in enumerate(prep.prefix):
        if j == prep.pending and not keep_pending:
            continue
        if c[0] != prep.opp:
            out.append(list(c))
            continue
        verb, _, arg = c[1].partition(" ")
        if verb == "move":
            name = prep.opp_active.get(j)
            slot = next((i for i, t in enumerate(prep.true_opp) if t.name == name), None)
            if slot is None:
                out.append(list(c))
                continue
            wset = world[slot]
            mid = arg
            if mid not in wset.moves:
                hp = [m for m in wset.moves if S.is_hp(m)]
                if S.is_hp(mid) and hp:
                    mid = hp[0]
                else:
                    tmoves = prep.true_opp[slot].moves
                    k = next((i for i, m in enumerate(tmoves) if m == arg or (S.is_hp(m) and S.is_hp(arg))), 0)
                    mid = wset.moves[k] if k < len(wset.moves) else wset.moves[0]
            if mid != arg:
                changed += 1
            out.append([c[0], f"move {mid}"])
        elif verb == "switch":
            slot = next((i for i, t in enumerate(prep.true_opp)
                         if t.name == arg or t.species_name == arg or S.to_id_safe(t.name) == S.to_id_safe(arg)),
                        None)
            if slot is None or world[slot].name == prep.true_opp[slot].name:
                out.append(list(c))
            else:
                out.append([c[0], f"switch {world[slot].name}"])
                changed += 1
        else:
            out.append(list(c))
    return out, changed


@dataclass
class GateResult:
    ok: bool
    first_bad_line: int
    lines_equal: bool
    rows_equal: Optional[bool]
    error: Optional[str] = None


def gate_many(prep: Prep, worlds: Sequence[List[S.MonSet]], rows: bool, keep_pending: bool = False
              ) -> List[GateResult]:
    """Replay each world's rendered log to the root in ONE core process; compare the searched side's lines (and,
    with ``rows``, its observation rows, masks and legal tokens) with the truth's."""
    recs = []
    for w in worlds:
        cmds, _ = render(prep, w, keep_pending)
        recs.append(_recorded(prep.battle, prep.opp, S.pack_team(w), cmds, prep.own_team, prep.world_seed))
    out: List[GateResult] = []
    for chunk0 in range(0, len(recs), 128):
        part = recs[chunk0:chunk0 + 128]
        try:
            results = run_core(part, trackers=rows, obs=rows)
        except RuntimeError as exc:          # a world the core refuses (a panic): judge one by one
            results = []
            for r in part:
                try:
                    results.append(run_core([r], trackers=rows, obs=rows)[0])
                except RuntimeError as e2:
                    results.append({"ok": False, "error": str(e2)[:300]})
            _ = exc
        for res in results:
            if not res.get("ok"):
                out.append(GateResult(False, -1, False, None, str(res.get("error"))[:300]))
                continue
            wl = lines_of(res, prep.v)
            tl = prep.truth_lines
            n = min(len(wl), len(tl))
            first = next((i for i in range(n) if wl[i] != tl[i]), n)
            leq = first == n and len(wl) == len(tl)
            if leq:
                first = len(tl) + 1
            req = None
            if rows:
                req = rows_of(res, prep.v) == prep.truth_rows
            out.append(GateResult(bool(leq and (req is None or req)), first, leq, req))
    return out


# ===================================================================================== world construction

@dataclass
class SeenDraw:
    """The S / I / A facts of one seen mon in a world (the screened classes)."""
    item: str
    ability: str
    nature: str
    evs: Tuple[int, ...]
    atk_variant: str
    hp_t: Optional[str]           # the type of a REVEALED BARE Hidden Power (None when there is none)
    ivs: Optional[Tuple[int, ...]] = None   # explicit IVs (the screen's synthetic spreads); else from the variant


def species_name(sp: str) -> str:
    sd = species_dex.get(sp)
    return sd.name if sd is not None else sp


def _seen_base(prep: Prep, m: V.SeenMon) -> S.MonSet:
    gender = gender_field(m.species, m.gender)
    # the details' public flags: shiny and level (happiness is never shown; blank = the default 255)
    tail = f"{'S' if m.shiny else ''}|{m.level if m.level is not None else ''}|"
    return S.MonSet(name=m.name, species_name=m.species_name, species=m.species, item="", ability="", moves=[],
                    nature="", evs=(0,) * 6, ivs=[31] * 6, gender=gender, tail=tail)


def seen_moves(prep: Prep, m: V.SeenMon, slot: int, arm: str, rng: np.random.Generator,
               hp_t: Optional[str], hp_screened: bool = False) -> List[str]:
    """Slot-aligned moves of a seen mon: revealed moves in their true slots, the rest drawn (M sampled) or true.

    ``hp_screened`` (F): a mon with NO revealed Hidden Power has its unrevealed Hidden Power decided by the screen
    (``hp_t``: the type, or None for none), because a Hidden Power sets IVs and the gate sees the stats: the draw
    then places that Hidden Power among the unrevealed moves (or excludes one) and draws the rest."""
    truth = prep.true_opp[slot]
    if arm in ("F+M", "O", "ID"):
        return list(truth.moves)
    nslots = 4
    out: List[Optional[str]] = [None] * nslots
    rev = list(m.moves)
    for mv in rev:
        k = next((i for i, t in enumerate(truth.moves) if t == mv or (mv == "hiddenpower" and S.is_hp(t))), None)
        real = ("hiddenpower" + hp_t) if (mv == "hiddenpower" and hp_t) else mv
        if k is not None and k < nslots and out[k] is None:
            out[k] = real
        else:
            free = [i for i in range(nslots) if out[i] is None]
            prep.notes.append(f"revealed_not_in_true_set:{m.name}:{mv}")
            if free:
                out[free[0]] = real
    have_hp = any(x is not None and S.is_hp(x) for x in out)
    if hp_screened and not have_hp and hp_t is not None:
        free = [i for i in range(nslots) if out[i] is None]
        if free:
            out[free[0]] = "hiddenpower" + hp_t
            have_hp = True
    k_draw = sum(1 for x in out if x is None)
    revealed_ids = [x for x in out if x is not None] + (["hiddenpower"] if have_hp else [])
    drawn = S.draw_moves(m.species, revealed_ids, k_draw, rng, allow_hp=not have_hp and not hp_screened)
    it = iter(drawn)
    filled = [x if x is not None else next(it, None) for x in out]
    return [x for x in filled if x is not None]


def draw_never_seen(prep: Prep, world: List[Optional[S.MonSet]], rng: np.random.Generator) -> None:
    revealed = [m.species for m in prep.seen.order()]
    for i in range(6):
        if world[i] is not None:
            continue
        bases = [S.base_of(w.species) for w in world if w is not None]
        tp = S.teammate_prior(revealed, bases)
        sp = S.weighted_pick(tp, rng)
        mv = S.draw_moves(sp, [], 4, rng, allow_hp=True)
        hp = next((S.hp_type(m) for m in mv if S.is_hp(m)), None)
        nature, evs, _ = _pick_spread(sp, rng)
        variant = S.ATK_IV_VARIANTS[int(rng.integers(2))]
        gender = gender_field(sp, "M" if rng.random() < 0.5 else "F")
        world[i] = S.MonSet(name=species_name(sp), species_name=species_name(sp), species=sp,
                            item=S.weighted_pick(S.item_prior(sp), rng), ability=S.weighted_pick(S.ability_prior(sp), rng),
                            moves=mv, nature=nature, evs=evs, ivs=S.ivs_for(hp, variant), gender=gender, tail="||",
                            origin={"U": "drawn"})


def _pick_spread(sp: str, rng: np.random.Generator):
    rows = S.spread_prior(sp)
    w = np.array([r[2] for r in rows])
    k = int(rng.choice(len(rows), p=w / w.sum()))
    return rows[k]


def seen_prior_draw(prep: Prep, m: V.SeenMon, arm: str, rng: np.random.Generator) -> SeenDraw:
    """One joint prior draw of a seen mon's unrevealed I / A / S (and the bare-HP type in F)."""
    sp = m.species
    item = m.item if m.item is not None else S.weighted_pick(S.item_prior(sp), rng)
    ap = S.ability_prior(sp)
    ability = m.ability if m.ability is not None else S.weighted_pick(ap, rng)
    nature, evs, _ = _pick_spread(sp, rng)
    variant = S.ATK_IV_VARIANTS[int(rng.integers(2))]
    hp_t = None
    if "hiddenpower" in m.moves:
        if arm == "F":
            hp_t = S.weighted_pick(S.hp_type_prior(sp), rng)
        else:                                            # F+M: the true type
            slot = prep.seen_slot[m.name]
            hp_t = next(S.hp_type(t) for t in prep.true_opp[slot].moves if S.is_hp(t))
    return SeenDraw(item, ability, nature, tuple(evs), variant, hp_t)


def seen_mode(prep: Prep, m: V.SeenMon, arm: str) -> SeenDraw:
    sp = m.species
    ip, ap = S.item_prior(sp), S.ability_prior(sp)
    rows = S.spread_prior(sp)
    best = max(rows, key=lambda r: r[2])
    hp_t = None
    if "hiddenpower" in m.moves:
        if arm == "F":
            hp = S.hp_type_prior(sp)
            hp_t = max(hp, key=hp.get)
        else:
            slot = prep.seen_slot[m.name]
            hp_t = next(S.hp_type(t) for t in prep.true_opp[slot].moves if S.is_hp(t))
    return SeenDraw(m.item if m.item is not None else max(ip, key=ip.get),
                    m.ability if m.ability is not None else max(ap, key=ap.get),
                    best[0], tuple(best[1]), "max", hp_t)


def assemble(prep: Prep, arm: str, rng: np.random.Generator, seen_draws: Optional[Dict[str, SeenDraw]] = None,
             fixed_moves: Optional[Dict[str, List[str]]] = None, fixed_u: Optional[List[Optional[S.MonSet]]] = None,
             hp_allowed: Optional[Dict[str, Dict[str, float]]] = None) -> List[S.MonSet]:
    """One world team for ``arm``. ``seen_draws``: the screened S / I / A per seen mon (F, F+M)."""
    world: List[Optional[S.MonSet]] = [None] * 6
    for name, m in prep.seen.mons.items():
        slot = prep.seen_slot[name]
        t = prep.true_opp[slot]
        if arm in ("ID", "O"):
            world[slot] = t.copy()
            world[slot].gender = prep.opp_real_gender[slot]
            continue
        base = _seen_base(prep, m)
        if arm in ("P", "O-M"):
            # S true; the bare-HP type follows the true IVs in P (S is true), is drawn in O-M (class M)
            true_hp = next((S.hp_type(x) for x in t.moves if S.is_hp(x)), None)
            hp_t = None
            if "hiddenpower" in m.moves:
                hp_t = true_hp if arm == "P" else S.weighted_pick(
                    (hp_allowed or {}).get(name) or S.hp_type_prior(m.species), rng)
            mv = fixed_moves[name] if fixed_moves else seen_moves(prep, m, slot, arm, rng, hp_t)
            base.moves = mv
            base.item = t.item
            base.ability = t.ability
            base.nature, base.evs, base.ivs = t.nature, t.evs, list(t.ivs)
            whp = next((S.hp_type(x) for x in mv if S.is_hp(x)), None)
            if whp is not None and whp != S.hp_type_of_ivs(t.ivs):
                variant = "min" if t.ivs[1] <= 3 else "max"
                base.ivs = S.ivs_for(whp, variant)
            base.origin = {"M": "drawn", "I": "true", "A": "true", "S": "true"}
        else:                                            # F, F+M
            sd = seen_draws[name]
            mv = fixed_moves[name] if fixed_moves else seen_moves(prep, m, slot, arm, rng, sd.hp_t,
                                                                  hp_screened=(arm == "F"))
            base.moves = mv
            base.item, base.ability, base.nature, base.evs = sd.item, sd.ability, sd.nature, sd.evs
            whp = next((S.hp_type(x) for x in mv if S.is_hp(x)), None)
            if sd.ivs is not None and whp == sd.hp_t:
                base.ivs = list(sd.ivs)
            else:   # a Hidden Power drawn in M (no bare one revealed) sets its own IVs
                base.ivs = S.ivs_for(whp, sd.atk_variant if whp == sd.hp_t else sd.atk_variant.split("|")[0])
            base.origin = {"M": "true" if arm == "F+M" else "drawn", "I": "drawn", "A": "drawn", "S": "drawn"}
        world[slot] = base
    if arm in ("O-M", "ID", "O"):
        for i in range(6):
            if world[i] is None:
                world[i] = prep.true_opp[i].copy()
                world[i].gender = prep.opp_real_gender[i]
    elif fixed_u is not None:
        for i in range(6):
            if world[i] is None:
                world[i] = fixed_u[i].copy()
    else:
        draw_never_seen(prep, world, rng)
    return [w for w in world]          # type: ignore[misc]
