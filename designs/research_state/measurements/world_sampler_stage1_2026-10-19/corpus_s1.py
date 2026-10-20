"""STAGE 1's CORPORA, as the team-level samplers draw them (design §4 (c), §6.2): every corpus team as six NORMALISED
packed sets, kept only when every set passes Showdown's ``TeamValidator('gen3ou')`` (the builder's own rule,
``world_legality``) and names only moves and items the engine runs. Measurement-only (owner question Q1 is OPEN: a
corpus sampler is an OFFLINE measurement; nothing here enters ``data/``, a network input or a belief prior).

* ``ladder`` = ``utils.ladder_corpus`` ``full`` (Metamon's ``hl_05_26``; its never-seen parts are Metamon's
  reconstruction, F7);
* ``forum``  = ``utils.team_corpora`` ``w_forum_v1`` ``ABC`` (pool-exact copies already removed; F3: near copies of
  pool teams remain, so a forum arm drops near copies of the state's TRUE team: ``strict``).

NORMALISATION (one entry rule, ``normalize_entry``): the species name as the nickname (no species field), every move
token resolved to its canonical id; a Hidden Power is TYPED by the set's IVs after Showdown's own fill: when the set's
explicit Hidden Power type (a typed move, else the packed ``hpType`` field) disagrees with the IVs and every IV is 31,
the IVs become ``packed.hp_ivs(type)`` (the validator's fill, the trainer's ``fix_gen3_hp_ivs``); otherwise the IVs
decide. The ``hpType`` field is then cleared (``world_screen.clear_stale_hp_type``'s reason: our Rust core plays the
IVs' type). Item, ability, nature, EVs, IVs, gender, level and happiness stay the corpus set's.

A mon set (the STRICT filter's and the near-copy meter's unit, stage 0's ``NEAR``) = (species, its canonical move SET,
a Hidden Power one move whatever its type).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s1_common as E  # noqa: E402

NEAR = 4                     # a corpus team sharing >= NEAR mon sets with the TRUE team is a near copy (stage 0)

MonSet = Tuple[str, FrozenSet[str]]


def canon(m: str) -> str:
    from agents.info_levers.packed import is_hp, move_id

    mid = move_id(m)
    return "hiddenpower" if is_hp(mid) else mid


def normalize_entry(entry: str, name_of) -> str:
    """One corpus set in the form every stage-1 world uses (module docstring). ``name_of(sid)``: the species' name."""
    from agents.info_levers.packed import hp_ivs, hp_type_of, move_id, parse_set, set_ivs, species_id, toid

    f = list(parse_set(entry))
    sid = species_id(f)
    raw = [m for m in f[4].split(",") if m]
    mids = [move_id(m) for m in raw]
    misc = f[11].split(",")
    field_t = toid(misc[1]) if len(misc) > 1 and misc[1] else ""
    typed = [m[len("hiddenpower"):] for m in mids if m.startswith("hiddenpower") and m != "hiddenpower"]
    want = typed[0] if typed else field_t
    ivs = set_ivs(f)
    if any(m.startswith("hiddenpower") for m in mids) and want and hp_type_of(ivs) != want and all(v == 31 for v in ivs):
        ivs = list(hp_ivs(want))
    f[8] = ",".join("" if v == 31 else str(v) for v in ivs) if any(v != 31 for v in ivs) else ""
    t_iv = hp_type_of(set_ivs(f))
    f[4] = ",".join(("hiddenpower" + t_iv) if m.startswith("hiddenpower") else m for m in mids)
    if len(misc) > 1:
        misc[1] = ""
        f[11] = ",".join(misc).rstrip(",")
    f[0], f[1] = name_of(sid), ""
    return "|".join(f)


def mon_set(entry: str) -> MonSet:
    from agents.info_levers.packed import parse_set, species_id, typed_moves

    f = parse_set(entry)
    return species_id(f), frozenset(canon(m) for m in typed_moves(f))


class Corpus:
    """Legal normalised corpus teams, indexed for the consistency query (stage 0's rule: every seen species on the
    team, its set holding every revealed move; a revealed bare Hidden Power matches any typed one)."""

    def __init__(self, name: str, teams: List[List[str]], dropped: Dict[str, int], source_sha: str):
        self.name = name
        self.teams = teams                                    # [team][slot] -> normalised packed entry
        self.dropped = dropped
        self.source_sha = source_sha
        self.sets: List[Dict[str, MonSet]] = []
        self.by_species: Dict[str, Set[int]] = defaultdict(set)
        #: per species, every corpus set of it (multiplicity kept): (its canonical move set, its normalised entry)
        self.species_sets: Dict[str, List[Tuple[FrozenSet[str], str]]] = defaultdict(list)
        for i, t in enumerate(teams):
            ms = {}
            for e in t:
                sid, mv = mon_set(e)
                ms[sid] = (sid, mv)
                self.by_species[sid].add(i)
                self.species_sets[sid].append((mv, e))
            self.sets.append(ms)

    def consistent(self, kept: Dict[str, Set[str]]) -> List[int]:
        """Indices of the teams holding every seen species with every revealed move (canonical; bare Hidden Power)."""
        ids: Optional[Set[int]] = None
        for s in kept:
            got = self.by_species.get(s, set())
            ids = set(got) if ids is None else ids & got
            if not ids:
                return []
        want = {s: {canon(m) for m in ms} for s, ms in kept.items()}
        return sorted(i for i in (ids or set()) if all(want[s] <= self.sets[i][s][1] for s in want))

    def near_copies(self, true_sets: Set[MonSet]) -> Set[int]:
        """Teams sharing >= NEAR mon sets with ``true_sets`` (the strict filter)."""
        cnt: Dict[int, int] = defaultdict(int)
        for sid, mv in true_sets:
            for i in self.by_species.get(sid, ()):
                if self.sets[i][sid][1] == mv:
                    cnt[i] += 1
        return {i for i, c in cnt.items() if c >= NEAR}

    def entry(self, i: int, sid: str) -> str:
        for e in self.teams[i]:
            if mon_set(e)[0] == sid:
                return e
        raise KeyError(f"{self.name} team {i} holds no {sid}")


def _raw(name: str) -> Tuple[List[str], str]:
    """The corpus' packed teams, in corpus order, and the source's sha256 stamp."""
    if name == "ladder":
        from utils.ladder_corpus import manifest, teams

        return list(teams("full")), manifest()["tiers"]["full"]["sha256"]
    if name == "forum":
        from utils.team_corpora import load_corpus
        from utils.team_packing import ConstantTeambuilder

        c = load_corpus("w_forum_v1", "ABC")
        out = []
        for text in c.texts:
            try:
                out.append(ConstantTeambuilder(text).packed_team)
            except Exception:                      # a paste that does not pack: dropped, counted by the caller
                out.append("")
        return out, c.tier_sha256
    raise ValueError(name)


def build(name: str, legality, prior) -> Corpus:
    """Normalise, check and index one corpus. ``legality``: a ``world_legality.Legality`` (its validator memoised);
    ``prior``: the builder's ``world_draws.Prior`` (the species' names)."""
    from main.value_search.world_draws import BANNED_ITEMS, BANNED_MOVES
    from agents.info_levers.packed import item_id, parse_set, typed_moves

    raw, sha = _raw(name)
    dropped: Dict[str, int] = defaultdict(int)
    cand: List[List[str]] = []
    for t in raw:
        if not t:
            dropped["unpackable"] += 1
            continue
        ents = [e for e in t.split("]") if e.strip()]
        if len(ents) != 6:
            dropped["not_six"] += 1
            continue
        try:
            norm = [normalize_entry(e, prior.name) for e in ents]
        except Exception:
            dropped["unknown_species_or_move"] += 1
            continue
        sids = [mon_set(e)[0] for e in norm]
        if len(set(sids)) != 6:
            dropped["species_twice"] += 1
            continue
        if any(set(typed_moves(parse_set(e))) & BANNED_MOVES or item_id(parse_set(e)) in BANNED_ITEMS for e in norm):
            dropped["engine_banned"] += 1
            continue
        cand.append(norm)
    # every SET by the validator (one round trip for every set not yet memoised), then the teams whose sets all pass
    sets = sorted({e for t in cand for e in t})
    ok = dict(zip(sets, (not p for p in legality.set_problems(sets))))
    teams = []
    for t in cand:
        if all(ok[e] for e in t):
            teams.append(t)
        else:
            dropped["illegal_set"] += 1
    return Corpus(name, teams, dict(dropped), sha)


def load(name: str, legality, prior) -> Corpus:
    """``build`` memoised on disk (``<archive>/corpus/<name>.json``, keyed by the source stamp and this file's sha)."""
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    p = E.d("corpus") / f"{name}.json"
    _, sha = _raw(name) if name == "ladder" else (None, None)
    if p.exists():
        blob = json.loads(p.read_text())
        if blob.get("code") == me and (sha is None or blob.get("source_sha") == sha):
            return Corpus(name, blob["teams"], blob["dropped"], blob["source_sha"])
    c = build(name, legality, prior)
    p.write_text(json.dumps({"code": me, "source_sha": c.source_sha, "dropped": c.dropped, "teams": c.teams}))
    return c
