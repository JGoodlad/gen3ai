"""PHASE 2's DETERMINIZER (README §8): honest-information worlds for a sampled state, by sampling the
opponent's HIDDEN team from the Smogon prior and REJECTING every world in which the searched side
would have observed anything different.

What is hidden from the searched side at the root, and what this samples:
- an opponent slot it has never seen (no switch / drag in its view): the WHOLE set is sampled —
  species (∝ the Smogon teammate co-occurrence with the revealed species, or species usage when none
  is revealed; never a species already on the team), four distinct moves (∝ the species' move
  usage), item, ability, nature + EVs (the spread table); a Hidden Power sets its IVs;
- a slot it HAS seen: every move slot whose move it never saw used is re-sampled from that species'
  move usage; species, item, ability, nature, EVs, IVs and the used moves are KEPT (they shaped what
  it observed: damage, speed order, item activations). That keeps a residual leak, named in §8.

THE GATE (the posterior's rejection step): the world's input log, truncated at the root, is replayed
through the pin's ``core_events``; the searched side's observation rows for EVERY one of its decisions
up to and including the root must be BYTE-EQUAL to the true log's. A world that changes anything the
searched side saw (a different damage roll from a different stat, a shifted dice stream from a
different construction draw, a different Sleep Talk pick) is resampled, up to ``MAX_TRIES`` times.
Rejection makes the sampled parts a draw from the prior CONDITIONED on the observations.

Only moves / items the pin's engine runs are sampled (its ``scan_move_probe`` FAIL-LOUD list is
excluded; the prior is renormalized over the rest).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

HP_TYPES = ["fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water",
            "grass", "electric", "psychic", "ice", "dragon", "dark"]
STATS = ("hp", "atk", "def", "spa", "spd", "spe")       # the packed order
MAX_TRIES = 30
SALT = "vdp_det_v1:"

#: the pin engine's FAIL-LOUD moves / items (scan_move_probe at 62085750 and 95d014fa, 2026-10-11)
BANNED_MOVES = frozenset("""attract bide blastburn bounce camouflage dig dive flatter fly followme foresight
frenzyplant grudge helpinghand hydrocannon ingrain lockon magiccoat magnitude metronome mindreader mirrormove
mist mudsport naturepower nightmare odorsleuth present psywave razorwind roleplay sketch skullbash skyattack
snore spite spitup stockpile swallow teleport triplekick watersport""".split())
BANNED_ITEMS = frozenset({"machobrace", "mail", "mentalherb", "shellbell"})
#: Phase 2c: draws where the network's candidates could not fill the slots and the Smogon prior did (per state)
NET_FALLBACK = {"seen": 0, "new": 0}


def toid(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def hp_type_of(ivs: Sequence[int]) -> str:
    iv = dict(zip(STATS, ivs))
    lsb = [iv[k] & 1 for k in ("hp", "atk", "def", "spe", "spa", "spd")]
    v = sum(b << i for i, b in enumerate(lsb))
    return HP_TYPES[v * 15 // 63]


def hp_ivs(t: str) -> List[int]:
    """30/31 IVs (max power) giving Hidden Power type ``t``, with the most 31s."""
    best = None
    for mask in range(64):
        b = [(mask >> i) & 1 for i in range(6)]          # hp, atk, def, spe, spa, spd
        v = sum(x << i for i, x in enumerate(b))
        if HP_TYPES[v * 15 // 63] != t:
            continue
        if best is None or sum(b) > sum(best):
            best = b
    iv = dict(zip(("hp", "atk", "def", "spe", "spa", "spd"), (30 + x for x in best)))
    return [iv[k] for k in STATS]


def parse_set(entry: str) -> List[str]:
    f = entry.split("|")
    if len(f) < 12:
        f += [""] * (12 - len(f))
    return f


def set_ivs(f: List[str]) -> List[int]:
    raw = f[8].split(",") if f[8] else [""] * 6
    return [int(x) if x else 31 for x in raw]


class Priors:
    def __init__(self, data_dir: Path):
        d = Path(data_dir)
        J = lambda n: json.loads((d / n).read_text())  # noqa: E731
        self.moves = J("gen3_move_priors.json")
        self.items = J("gen3_item_priors.json")
        self.abilities = J("gen3_ability_priors.json")
        self.spreads = J("gen3_spread_priors.json")
        self.mates = J("gen3_teammate_priors.json")
        self.species = J("gen3_species.json")
        st = J("gen3_smogon_stats.json")["data"]
        self.usage = {toid(k): float(v.get("usage", 0) or 0) for k, v in st.items()}
        self.pool = sorted(s for s in self.moves if s in self.species)

    def name(self, sid: str) -> str:
        return self.species[sid]["name"]


def _draw(rng, keys: List[str], w: np.ndarray) -> Optional[str]:
    w = np.asarray(w, dtype=np.float64)
    if len(keys) == 0 or w.sum() <= 0:
        return None
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


def sample_moves(rng, P: Priors, sid: str, keep: Dict[int, str], hp_type: Optional[str]) -> Optional[List[str]]:
    """Fill the 4 move slots: ``keep`` = {slot: move id} stays; every other slot is drawn from the
    species' move usage without replacement. ``hp_type``: the only Hidden Power type allowed (None =
    any, and then the IVs follow the drawn type)."""
    pri = P.moves.get(sid)
    if not pri:
        return None
    slots: List[Optional[str]] = [keep.get(i) for i in range(4)]
    have = {m for m in slots if m}
    for i in range(4):
        if slots[i]:
            continue
        has_hp = any(m.startswith("hiddenpower") for m in have)
        keys = [m for m in pri if m not in have and m not in BANNED_MOVES]
        keys = [m for m in keys if not (m.startswith("hiddenpower")
                                        and (has_hp or (hp_type is not None and m != f"hiddenpower{hp_type}")))]
        m = _draw(rng, keys, np.array([pri[k] for k in keys]))
        if m is None:
            return None
        slots[i] = m
        have.add(m)
    return [m for m in slots]   # type: ignore[misc]


def cp_sample(rng, keys: List[str], pis: Sequence[float], m: int) -> Optional[List[str]]:
    """Phase 2c: a FIXED-SIZE subset of ``m`` keys drawn as INDEPENDENT presences π conditioned on the count = m
    (the conditional-Poisson design: P(S) ∝ Π_{i∈S} π_i/(1−π_i); the natural sampler for the network's
    fixed-mass marginals, which are the I-projection of independent presences onto that count). Exact sequential
    draw from the elementary symmetric sums; returns the keys in a random order, or None if fewer than m."""
    n = len(keys)
    if m <= 0:
        return []
    if n < m:
        return None
    p = np.clip(np.asarray(pis, dtype=np.float64), 1e-9, 1 - 1e-9)
    w = p / (1 - p)
    w = w / w.max()
    # E[i, j] = e_j(w[i:]) (the elementary symmetric sum of order j over the suffix)
    E = np.zeros((n + 1, m + 1))
    E[n, 0] = 1.0
    for i in range(n - 1, -1, -1):
        E[i, 0] = 1.0
        for j in range(1, m + 1):
            E[i, j] = E[i + 1, j] + w[i] * E[i + 1, j - 1]
    out, need = [], m
    for i in range(n):
        if need == 0:
            break
        if E[i, need] <= 0:
            return None
        if rng.random() < w[i] * E[i + 1, need - 1] / E[i, need]:
            out.append(keys[i])
            need -= 1
    if need:
        return None
    return [out[k] for k in rng.permutation(len(out))]


def net_moves(rng, mw: Dict[str, float], keep: Dict[int, str], hp_type: Optional[str],
              n_slots: int = 4) -> Optional[List[str]]:
    """Phase 2c: fill the move slots not in ``keep`` from the NETWORK's move presence ``mw`` ({move id: π} over
    the mon's unrevealed candidates; `move_belief.root_move_beliefs`). Only moves the pin's engine runs; at most
    one Hidden Power, and only of type ``hp_type`` when the IVs are fixed (a seen mon). Returns the slots."""
    have = set(keep.values())
    has_hp = any(m.startswith("hiddenpower") for m in have)
    keys, pis = [], []
    for k in sorted(mw):
        if k in have or k in BANNED_MOVES:
            continue
        if k.startswith("hiddenpower"):
            if has_hp or k == "hiddenpower" or (hp_type is not None and k != f"hiddenpower{hp_type}"):
                continue
        keys.append(k)
        pis.append(mw[k])
    free = [i for i in range(n_slots) if i not in keep]
    for _ in range(50):
        got = cp_sample(rng, keys, pis, len(free))
        if got is None:
            return None
        if sum(m.startswith("hiddenpower") for m in got) <= 1:
            break
    else:
        return None
    slots = [keep.get(i) for i in range(n_slots)]
    for i, m in zip(free, got):
        slots[i] = m
    return slots   # type: ignore[return-value]


def sample_new_set(rng, P: Priors, sid: str, mw: Optional[Dict[str, float]] = None) -> Optional[str]:
    moves = sample_moves(rng, P, sid, {}, None) if mw is None else net_moves(rng, mw, {}, None)
    if moves is None:
        return None
    it = {k: v for k, v in P.items.get(sid, {}).items() if k not in BANNED_ITEMS}
    item = _draw(rng, list(it), np.array(list(it.values()))) if it else "nothing"
    ab = P.abilities.get(sid, {})
    ability = _draw(rng, list(ab), np.array(list(ab.values()))) if ab else ""
    sp = P.spreads.get(sid) or [["Hardy", [84] * 6, 1.0]]
    k = int(rng.choice(len(sp), p=np.array([x[2] for x in sp]) / sum(x[2] for x in sp)))
    nature, evs = sp[k][0], sp[k][1]
    hp = [m for m in moves if m.startswith("hiddenpower")]
    ivs = ",".join(str(x) for x in hp_ivs(hp[0][len("hiddenpower"):])) if hp else ""
    evs_s = ",".join(str(int(x)) if int(x) else "" for x in evs)
    return "|".join([P.name(sid), "", "" if item in (None, "nothing") else item, ability or "", ",".join(moves),
                     nature, evs_s, "", ivs, "", "", ""])


def resample_revealed(rng, P: Priors, entry: str, seen_moves: Set[str],
                      mw: Optional[Dict[str, float]] = None) -> Tuple[str, int]:
    """A revealed slot: its unseen move slots re-drawn (from the Smogon prior, or, Phase 2c, from the network's
    move presence ``mw``). Returns (entry, slots re-drawn)."""
    f = parse_set(entry)
    sid = toid(f[1] or f[0])
    mv = f[4].split(",")
    # the protocol names "Hidden Power" without its type: a seen "hiddenpower" keeps the set's HP move
    keep = {i: m for i, m in enumerate(mv)
            if m in seen_moves or (m.startswith("hiddenpower") and "hiddenpower" in seen_moves)}
    if len(keep) == len(mv) or sid not in P.moves:
        return entry, 0
    ivs = set_ivs(f)
    if mw is not None:
        out = net_moves(rng, mw, keep, hp_type_of(ivs), len(mv))
        if out is None:          # the network's candidates cannot fill the slots: the Smogon prior (counted)
            out = sample_moves(rng, P, sid, keep, hp_type_of(ivs))
            NET_FALLBACK["seen"] += int(out is not None)
    else:
        out = sample_moves(rng, P, sid, keep, hp_type_of(ivs))
    if out is None:
        return entry, 0
    f[4] = ",".join(out)
    return "|".join(f), len(mv) - len(keep)


def sample_world(rng, P: Priors, team: str, revealed: Dict[str, Set[str]],
                 species_w: Optional[Dict[str, float]] = None,
                 move_w: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None,
                 draw_slots: Optional[Set[int]] = None) -> Tuple[Optional[str], dict]:
    """The opponent's packed team with its hidden parts drawn. ``draw_slots`` (v2, Amendment 8; None = v1): the
    never-seen slots whose true mon drew a turn-0 gender (`place_genders`); a v2 world that cannot be built returns
    (None, {"fail": ...}). ``revealed``: {species id: seen move ids}.
    ``species_w`` (Phase 2b, Amendment 5): never-seen slots draw their species ∝ these weights (the network's
    own X5 species presence π at the root) instead of the Smogon teammate prior; sets stay Smogon.
    ``move_w`` (Phase 2c, Amendment 6): the network's per-mon move presence keyed by species NUM, ``{"seen": {num:
    {move: π}}, "hyp": {num: {move: π}}}``: a seen mon's unseen move slots, and a never-seen slot whose drawn species
    is one of the root's HYPOTHESIS species, draw their moves from it (items, abilities, spreads stay Smogon)."""
    entries = team.split("]")
    ids = [toid(parse_set(e)[1] or parse_set(e)[0]) for e in entries]
    present = {s for s in ids if s in revealed}
    out, new, redrawn, net_seen, net_new = [], 0, 0, 0, 0
    num = lambda s_: int(P.species.get(s_, {}).get("num") or 0)  # noqa: E731
    for e, sid in zip(entries, ids):
        if sid in revealed:
            if move_w is not None and move_w.get("true_seen"):    # Phase 2c-T: a seen mon keeps its TRUE set
                out.append(e)
                continue
            mw = move_w["seen"].get(num(sid)) if move_w is not None else None
            e2, k = resample_revealed(rng, P, e, revealed[sid], mw)
            out.append(e2)
            redrawn += k
            net_seen += int(mw is not None and k > 0)
            continue
        cand = [s for s in P.pool if s not in present]
        if species_w is not None:
            w = np.array([species_w.get(s, 0.0) for s in cand])
        elif revealed:
            w = np.array([np.mean([P.mates.get(r, {}).get(s, 0.0) for r in revealed]) for s in cand])
        else:
            w = np.array([P.usage.get(s, 0.0) for s in cand])
        if w.sum() <= 0:
            w = np.array([P.usage.get(s, 0.0) for s in cand])
        s = _draw(rng, cand, w)
        mw = move_w["hyp"].get(num(s)) if (move_w is not None and s) else None
        ent = sample_new_set(rng, P, s, mw) if s else None
        if ent is None and mw is not None:         # the network's candidates cannot fill 4 slots: Smogon (counted)
            ent = sample_new_set(rng, P, s)
            NET_FALLBACK["new"] += int(ent is not None)
            mw = None
        net_new += int(ent is not None and mw is not None)
        if ent is None:
            if draw_slots is not None:     # v2: never fall back to the TRUE set of a never-seen slot (a leak)
                return None, {"fail": "sample"}
            out.append(e)
            continue
        present.add(s)
        out.append(ent)
        new += 1
    st = {"new_slots": new, "redrawn_move_slots": redrawn}
    if move_w is not None:
        st.update({"net_moves_seen_mons": net_seen, "net_moves_new_slots": net_new})
    if draw_slots is not None:
        out2 = place_genders(rng, P, out, [i for i, sid in enumerate(ids) if sid not in revealed], draw_slots)
        if out2 is None:
            return None, {"fail": "gender_count"}
        out = out2
    return "]".join(out), st


def species_draws_gender(P: Priors, sid: str) -> bool:
    """The engine draws a gender for a mon iff its packed set leaves the gender empty AND its species has no fixed
    gender (`rust_sim` `draw_turn0_genders`: ONE uniform M / F sample; a fixed 'M' / 'F' / genderless 'N' is draw-free)."""
    return P.species.get(sid, {}).get("gender") is None


def gender_draw_slots(P: Priors, team: str) -> Set[int]:
    """The team slots whose mon consumes a turn-0 gender draw."""
    out = set()
    for i, e in enumerate(team.split("]")):
        f = parse_set(e)
        if not f[7] and species_draws_gender(P, toid(f[1] or f[0])):
            out.add(i)
    return out


def place_genders(rng, P: Priors, entries: List[str], never_seen: List[int], draw_slots: Set[int]) -> Optional[List[str]]:
    """v2 (Amendment 8): the world's NEVER-SEEN mons placed so that the turn-0 gender draws happen at EXACTLY the true
    team's draw slots, whatever species were drawn, so the pre-root dice (and every seen mon's drawn gender) are the
    truth's and the gate never conditions on the never-seen species' gender classes beyond what the count requires.

    ``draw_slots`` = the never-seen slots whose TRUE mon drew a gender. The world's never-seen mons are an exchangeable
    sample, so they may be permuted among the never-seen slots: r = |draw_slots| of the world's gender-RATIO'd mons
    (chosen uniformly) go to those slots with the gender left empty (they draw; the drawn value is the true slot's coin,
    a fair M / F the searched side never saw); every other never-seen mon gets an EXPLICIT gender from the engine's
    prior (uniform M / F for a ratio'd species; its fixed gender otherwise), so it draws nothing. Returns None when the
    world holds fewer ratio'd never-seen mons than r (the RESIDUAL conditioning: such a world cannot reproduce the true
    dice without an engine change; counted as ``gender_count``)."""
    ns = sorted(never_seen)
    need = sorted(s for s in draw_slots if s in ns)
    sets = {i: parse_set(entries[i]) for i in ns}
    ratio = [i for i in ns if species_draws_gender(P, toid(sets[i][1] or sets[i][0]))]
    if len(ratio) < len(need):
        return None
    pick = [ratio[k] for k in rng.permutation(len(ratio))[: len(need)]]
    rest = [i for i in ns if i not in pick]
    rest = [rest[k] for k in rng.permutation(len(rest))]
    others = [s for s in ns if s not in need]
    out = list(entries)
    for slot, src in zip(need, pick):
        f = list(sets[src])
        f[7] = ""
        out[slot] = "|".join(f)
    for slot, src in zip(others, rest):
        f = list(sets[src])
        sid = toid(f[1] or f[0])
        g = P.species.get(sid, {}).get("gender")
        f[7] = ("M" if rng.random() < 0.5 else "F") if g is None else ("" if g == "N" else g)
        out[slot] = "|".join(f)
    return out


def team_problems(team: str) -> Set[Tuple[str, int]]:
    """(rule, slot) of every gen3ou format-spec violation of a packed team (`agents.gen3_data.team_legality` at the
    pin: the bans and the clauses, a ladder-only rule included)."""
    from agents.gen3_data import team_legality as tl

    return {(p.rule, int(p.slot)) for p in tl.validate_team(tl.parse_packed(team))}


def opp_actives(res: dict, side_idx: int, n_cmds: int) -> List[Optional[str]]:
    """The species id of side ``side_idx``'s ACTIVE mon at each of its first ``n_cmds`` commands: the k-th command
    answers the side's k-th request that asks for a choice (``wait`` requests ask for none), read from that side's
    OWN request lines (the truth's; used only to attribute a past move choice to the mon that made it)."""
    import json as _json

    out: List[Optional[str]] = []
    for side, lines in res["chunks"]:
        if side != side_idx:
            continue
        for ln in lines:
            if not ln.startswith("|request|"):
                continue
            body = ln[len("|request|"):]
            if not body.strip():
                continue
            rq = _json.loads(body)
            if rq.get("wait") or rq.get("teamPreview"):
                continue
            act = [p for p in rq.get("side", {}).get("pokemon", []) if p.get("active")]
            out.append(toid(act[0]["details"].split(",")[0]) if act else None)
    if len(out) < n_cmds:
        raise RuntimeError(f"side {side_idx}: {len(out)} choice requests for {n_cmds} commands")
    return out[:n_cmds]


def protocol_lines(res: dict, viewer: int, n: int) -> List[str]:
    """Every protocol line the viewer was sent up to its n-th obs-bearing entry (inclusive), in order."""
    ent = [e for e in res["trackers"][viewer] if "obs" in e]
    after = int(ent[n]["after"])
    return [ln for side, lines in res["chunks"][: after + 1] if side == viewer for ln in lines]


def reveals(res: dict, viewer: int, n: int, extra: bool = False) -> Tuple[Dict[str, Set[str]], List[str]]:
    """The opponent species the viewer has seen, with the move ids seen used, up to its n-th
    obs-bearing entry (inclusive), and the obs frames (b64) of entries 0..n. ``extra`` (v2, Amendment 8): also the
    public lines that name the mon's OWN move outside ``|move|``: ``|-singleturn|<mon>|move: M`` (Focus Punch's
    tightening; gen 3's other single-turn effects are the user's own moves too) and ``|cant|<mon>|<reason>|M`` (the move
    it chose and could not use: Taunt, Disable, Imprison, a lost focus). Lines naming a move used ON the mon (Leech
    Seed's ``-start``, a ``[from] move:`` tag of the other side's move) are not reveals and stay unparsed."""
    opp = f"p{2 - viewer}a: "
    chunks = res["chunks"]
    ent = [e for e in res["trackers"][viewer] if "obs" in e]
    if len(ent) <= n:
        raise RuntimeError(f"viewer has {len(ent)} decisions, the root is {n}")
    after = int(ent[n]["after"])
    nick: Dict[str, str] = {}
    seen: Dict[str, Set[str]] = {}
    for side, lines in chunks[: after + 1]:
        if side != viewer:
            continue
        for line in lines:
            p = line.split("|")
            if len(p) < 4 or not p[2].startswith(opp):
                continue
            if p[1] in ("switch", "drag"):
                sid = toid(p[3].split(",")[0])
                nick[p[2]] = sid
                seen.setdefault(sid, set())
            elif p[1] == "move" and p[2] in nick:
                seen[nick[p[2]]].add(toid(p[3]))
            elif extra and p[2] in nick and p[1] == "-singleturn" and p[3].startswith("move: "):
                seen[nick[p[2]]].add(toid(p[3][len("move: "):]))
            elif extra and p[2] in nick and p[1] == "cant" and len(p) > 4 and p[4].strip():
                seen[nick[p[2]]].add(toid(p[4].replace("move: ", "")))
    return seen, [e["obs"]["b64"] for e in ent[: n + 1]]


def world_rng(did: str, j: int):
    return np.random.default_rng(int(hashlib.sha256(f"{SALT}{did}:{j}".encode()).hexdigest()[:16], 16))


def build_worlds(battle, side: str, at: int, n: int, did: str, P: Priors, J: int,
                 max_tries: int = MAX_TRIES, root_choice_in_log: bool = False,
                 species_w: Optional[Dict[str, float]] = None,
                 move_w: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None,
                 builder: str = "v2") -> dict:
    """J gate-passed worlds for one root. Returns {"teams": [packed or None] * J, "tries": [...],
    "stats": [...], "revealed": ..., "true_rows": int}.

    ``builder`` "v2" (Amendment 8, the default; the fix for the hidden-info study's R-GIGO FAIL): (1) the gate also
    requires the searched side's PROTOCOL lines up to the root byte-equal to the truth's (the rows alone missed gender
    tokens, a Focus Punch tightening line, a Beat Up hit); (2) ``reveals`` also parses the ``-singleturn`` / ``cant``
    lines naming the mon's own move; (3) the never-seen mons are placed so the turn-0 gender draws happen at the true
    draw slots (`place_genders`: no conditioning on their gender classes beyond the draw count); (4) a world whose team
    breaks a gen3ou rule the true team does not break is rejected; a never-seen slot that cannot be sampled fails the
    try instead of keeping its TRUE set. "v1" = the pilot's Phase 2 / 2b builder, byte-for-byte."""
    v2 = builder == "v2"
    if builder not in ("v1", "v2"):
        raise ValueError(f"builder {builder!r}")
    from agents.battle.core_replay import RecordedBattle, run_core

    viewer = 0 if side == "p1" else 1
    opp_key = "p2" if side == "p1" else "p1"
    cmds = [list(c) for c in battle.commands[:at]]

    def rec(team_opp: Optional[str], label: str, commands=None):
        p = {"p1": dict(battle.p1), "p2": dict(battle.p2)}
        if team_opp is not None:
            p[opp_key] = dict(p[opp_key], team=team_opp)
        return RecordedBattle(label=label, format_id=battle.format_id, seed=battle.seed, p1=p["p1"], p2=p["p2"],
                              commands=cmds if commands is None else commands)

    true = run_core([rec(None, f"{did}#true")], trackers=True, obs=True)[0]
    if not true["ok"]:
        raise RuntimeError(f"{did}: the true prefix does not replay: {true.get('error')}")
    revealed, ref = reveals(true, viewer, n, extra=v2)
    ref_proto = protocol_lines(true, viewer, n) if v2 else None
    team = (battle.p2 if side == "p1" else battle.p1)["team"]
    entries = team.split("]")
    sets = [parse_set(e) for e in entries]
    sids = [toid(f[1] or f[0]) for f in sets]
    moves_of = [f[4].split(",") for f in sets]
    # The log names the opponent's past choices by MOVE / SPECIES name. A choice the searched side
    # never saw executed (full paralysis, a gen-3 faint cancelling the rest of the turn, ...) must
    # still resolve in a world where that slot holds another move: such a MOVE choice is rewritten
    # to its SLOT ("move k", the true set's slot of that move in the only kept mon that has it), and
    # such a SWITCH (to a never-seen mon) to the world's species in that team slot. Only an ambiguous
    # case (the move in several kept mons' sets, none seen), and the opponent's ROOT-turn choice when
    # the log already holds it (``root_choice_in_log``, ~1 % of roots), stay by name: a counted LEAK.
    kept = {k: set(v) for k, v in revealed.items()}
    leak = {"moves": [], "root_choice": None, "rewritten_moves": 0, "rewritten_switches": 0}
    last_opp = max((i for i, c in enumerate(cmds) if c[0] == opp_key), default=None)
    n_opp_cmd = 0
    actives = opp_actives(true, 1 - viewer, sum(1 for c in cmds if c[0] == opp_key)) if v2 else []
    tmpl: List[list] = []
    for i, (s_, t) in enumerate(cmds):
        if s_ != opp_key:
            tmpl.append([s_, t])
            continue
        if v2 and t.startswith("move ") and not t[5:].strip().isdigit():
            # v2: the choice belongs to the opponent's ACTIVE mon at that command (its own request), never to
            # whichever mon happens to hold the same move (v1's rule kept a choice by name when ANOTHER mon had the
            # move seen, so a world whose active mon lacked it was refused: a rejection on an unseen fact)
            m = toid(t[5:])
            hp = m.startswith("hiddenpower")
            a_ = actives[n_opp_cmd]
            k = sids.index(a_) if a_ in sids else None
            if k is None or m not in moves_of[k]:
                tmpl.append([s_, t])             # not in the active mon's true set: an attribution miss (counted)
                leak["attribution_miss"] = leak.get("attribution_miss", 0) + 1
            elif i == last_opp and root_choice_in_log:
                kept.setdefault(a_, set()).add(m)
                leak["root_choice"] = m
                tmpl.append([s_, t])
            elif m in kept.get(a_, set()) or (hp and "hiddenpower" in kept.get(a_, set())):
                tmpl.append([s_, t])
            else:
                tmpl.append([s_, ("MOVE", k, moves_of[k].index(m))])
                leak["rewritten_moves"] += 1
            n_opp_cmd += 1
            continue
        if v2:
            n_opp_cmd += 1
        if t.startswith("move ") and not t[5:].strip().isdigit():
            m = toid(t[5:])
            hp = m.startswith("hiddenpower")
            cand = [k for k in range(len(sets)) if sids[k] in kept and m in moves_of[k]]
            seen_for = [k for k in cand if m in kept[sids[k]] or (hp and "hiddenpower" in kept[sids[k]])]
            if i == last_opp and root_choice_in_log:
                for k in cand:
                    kept[sids[k]].add(m)
                leak["root_choice"] = m
                tmpl.append([s_, t])
            elif seen_for:
                tmpl.append([s_, t])
            elif len(cand) == 1:
                tmpl.append([s_, ("MOVE", cand[0], moves_of[cand[0]].index(m))])
                leak["rewritten_moves"] += 1
            else:
                for k in cand:
                    kept[sids[k]].add(m)
                    leak["moves"].append(f"{sids[k]}:{m}")
                tmpl.append([s_, t])
            continue
        if t.startswith("switch ") and not t[7:].strip().isdigit():
            sp = toid(t[7:])
            if sp in kept or sp not in sids:
                tmpl.append([s_, t])
            else:
                tmpl.append([s_, ("SLOT", sids.index(sp))])
                leak["rewritten_switches"] += 1
            continue
        tmpl.append([s_, t])
    revealed = kept

    def world_cmds(team_w: str) -> List[list]:
        sets_w = [parse_set(e) for e in team_w.split("]")]

        def render(t):
            if not isinstance(t, tuple):
                return t
            if t[0] == "SLOT":
                return f"switch {sets_w[t[1]][0]}"
            return f"move {sets_w[t[1]][4].split(',')[t[2]]}"      # ("MOVE", mon, slot)
        return [[s_, render(t)] for s_, t in tmpl]

    rngs = [world_rng(did, j) for j in range(J)]
    teams: List[Optional[str]] = [None] * J
    tries = [0] * J
    stats: List[Optional[dict]] = [None] * J
    fails: Dict[str, int] = {}
    if v2:
        never_seen = {i for i, s in enumerate(sids) if s not in revealed}
        draw_slots = gender_draw_slots(P, team) & never_seen
        true_problems = team_problems(team)
    for _ in range(max_tries):
        todo = [j for j in range(J) if teams[j] is None]
        if not todo:
            break
        cand = {}
        for j in todo:
            if v2:
                cand[j] = sample_world(rngs[j], P, team, revealed, species_w, move_w, draw_slots=draw_slots)
            else:
                cand[j] = sample_world(rngs[j], P, team, revealed, species_w, move_w)
            tries[j] += 1
        if v2:      # rejected before any replay: a world that cannot be built, or one breaking a gen3ou rule
            for j in list(todo):
                if cand[j][0] is None:
                    k_ = cand[j][1]["fail"]
                    fails[k_] = fails.get(k_, 0) + 1
                    todo.remove(j)
                elif team_problems(cand[j][0]) - true_problems:
                    fails["clause"] = fails.get("clause", 0) + 1
                    todo.remove(j)
            if not todo:
                continue
        out = run_core([rec(cand[j][0], f"{did}#w{j}", world_cmds(cand[j][0])) for j in todo],
                       trackers=True, obs=True)
        for j, r in zip(todo, out):
            if not r["ok"]:
                fails["refused"] = fails.get("refused", 0) + 1
                msgs = fails.setdefault("refused_msgs", [])
                if len(msgs) < 3:
                    msgs.append(str(r.get("error"))[:300])
                continue
            try:
                _, rows = reveals(r, viewer, n)
            except RuntimeError:
                fails["short"] = fails.get("short", 0) + 1
                continue
            if rows != ref:
                k = next(i for i, (a, b) in enumerate(zip(rows, ref)) if a != b)
                fails["rows_differ"] = fails.get("rows_differ", 0) + 1
                fails.setdefault("first_diff_at", []).append(k)
                continue
            if v2 and protocol_lines(r, viewer, n) != ref_proto:
                fails["protocol_differ"] = fails.get("protocol_differ", 0) + 1
                continue
            teams[j] = cand[j][0]
            stats[j] = cand[j][1]
    if move_w is not None:
        fails["net_fallback"] = dict(NET_FALLBACK)
        NET_FALLBACK.update(seen=0, new=0)
    return {"teams": teams, "cmds": [world_cmds(t) if t else None for t in teams], "tries": tries,
            "stats": stats, "fails": fails, "leak": leak,
            "revealed": {k: sorted(v) for k, v in revealed.items()}, "n_rows": len(ref)}
