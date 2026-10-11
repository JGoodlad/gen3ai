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


def sample_new_set(rng, P: Priors, sid: str) -> Optional[str]:
    moves = sample_moves(rng, P, sid, {}, None)
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


def resample_revealed(rng, P: Priors, entry: str, seen_moves: Set[str]) -> Tuple[str, int]:
    """A revealed slot: its unseen move slots re-drawn. Returns (entry, slots re-drawn)."""
    f = parse_set(entry)
    sid = toid(f[1] or f[0])
    mv = f[4].split(",")
    # the protocol names "Hidden Power" without its type: a seen "hiddenpower" keeps the set's HP move
    keep = {i: m for i, m in enumerate(mv)
            if m in seen_moves or (m.startswith("hiddenpower") and "hiddenpower" in seen_moves)}
    if len(keep) == len(mv) or sid not in P.moves:
        return entry, 0
    ivs = set_ivs(f)
    out = sample_moves(rng, P, sid, keep, hp_type_of(ivs))
    if out is None:
        return entry, 0
    f[4] = ",".join(out)
    return "|".join(f), len(mv) - len(keep)


def sample_world(rng, P: Priors, team: str, revealed: Dict[str, Set[str]]) -> Tuple[str, dict]:
    """The opponent's packed team with its hidden parts drawn. ``revealed``: {species id: seen move ids}."""
    entries = team.split("]")
    ids = [toid(parse_set(e)[1] or parse_set(e)[0]) for e in entries]
    present = {s for s in ids if s in revealed}
    out, new, redrawn = [], 0, 0
    for e, sid in zip(entries, ids):
        if sid in revealed:
            e2, k = resample_revealed(rng, P, e, revealed[sid])
            out.append(e2)
            redrawn += k
            continue
        cand = [s for s in P.pool if s not in present]
        if revealed:
            w = np.array([np.mean([P.mates.get(r, {}).get(s, 0.0) for r in revealed]) for s in cand])
        else:
            w = np.array([P.usage.get(s, 0.0) for s in cand])
        if w.sum() <= 0:
            w = np.array([P.usage.get(s, 0.0) for s in cand])
        s = _draw(rng, cand, w)
        ent = sample_new_set(rng, P, s) if s else None
        if ent is None:
            out.append(e)
            continue
        present.add(s)
        out.append(ent)
        new += 1
    return "]".join(out), {"new_slots": new, "redrawn_move_slots": redrawn}


def reveals(res: dict, viewer: int, n: int) -> Tuple[Dict[str, Set[str]], List[str]]:
    """The opponent species the viewer has seen, with the move ids seen used, up to its n-th
    obs-bearing entry (inclusive), and the obs frames (b64) of entries 0..n."""
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
    return seen, [e["obs"]["b64"] for e in ent[: n + 1]]


def world_rng(did: str, j: int):
    return np.random.default_rng(int(hashlib.sha256(f"{SALT}{did}:{j}".encode()).hexdigest()[:16], 16))


def build_worlds(battle, side: str, at: int, n: int, did: str, P: Priors, J: int,
                 max_tries: int = MAX_TRIES, root_choice_in_log: bool = False) -> dict:
    """J gate-passed worlds for one root. Returns {"teams": [packed or None] * J, "tries": [...],
    "stats": [...], "revealed": ..., "true_rows": int}."""
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
    revealed, ref = reveals(true, viewer, n)
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
    tmpl: List[list] = []
    for i, (s_, t) in enumerate(cmds):
        if s_ != opp_key:
            tmpl.append([s_, t])
            continue
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
    for _ in range(max_tries):
        todo = [j for j in range(J) if teams[j] is None]
        if not todo:
            break
        cand = {}
        for j in todo:
            cand[j] = sample_world(rngs[j], P, team, revealed)
            tries[j] += 1
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
            teams[j] = cand[j][0]
            stats[j] = cand[j][1]
    return {"teams": teams, "cmds": [world_cmds(t) if t else None for t in teams], "tries": tries,
            "stats": stats, "fails": fails, "leak": leak,
            "revealed": {k: sorted(v) for k, v in revealed.items()}, "n_rows": len(ref)}
