"""POOLSETS, HEAD SIDE: F's worlds with each species' SET facts drawn from the TRAINING POOL's own per-species
distribution instead of Smogon's (`design_hidden_info_binding.md` §6.2 / §6.4: "F's structure, the pool's per-species
set frequencies instead of Smogon's"). A MEASUREMENT of the eval distribution's mismatch to the Smogon prior — the pool
never ships as a prior (owner, 2026-08-15).

Per species the pool holds (``pool_teams.load_pool()``, the trainer's packed pool, 719 teams, duplicates counted):

* ``moves``: inclusion frequency (sets of the species holding the move / sets of the species), Hidden Power TYPED as
  the pool runs it (``typed_moves``: the IVs type a bare one); engine-banned moves dropped (as F's ``Prior``);
* ``hidden_power``: P(type | the set holds one) over the species' pool sets with one;
* ``items`` / ``ability``: frequencies (an empty item = no item: dropped, as F drops ``nothing``; engine-banned items
  dropped; an empty ability field is COUNTED and dropped);
* ``spreads``: (nature id, EVs) frequencies — fed to the SAME ``world_screen.SpreadPrior`` F uses (its smooth term
  included: the (1 − ε)·J + ε·N·ΠM structure, with J the pool's table).

A species the pool never holds (or whose table for a class is empty after the drops) takes Smogon's for that class:
COUNTED (``PoolSetsPrior.fallback``). The NEVER-SEEN SPECIES draw is F's (Smogon teammate co-occurrence: ``mates``,
``usage`` and ``pool`` are Smogon's), so a never-seen species outside the pool takes Smogon sets (counted per world).

SCREEN-CACHE NAMESPACE: POOLSETS' seen-mon samplers carry the pool's item / ability / spread / Hidden Power tables, so
they must never be F's. :class:`PoolSetsSampler` keeps its samplers in its OWN sub-dict of the harness's cache,
``cache[("POOLSETS", key)]``; only the gate-verdict dict ``cache[("verdict", key)]`` (opponent team text → the strict
or lenient replay verdict, a pure function of the text) is shared with whatever dict is passed. Either F's
per-decision cache or a dict of POOLSETS' own (``caches["POOLSETS"]``) is safe; the latter forgoes only the shared
verdicts::

    from poolsets_worlds import poolsets_sampler
    PS = poolsets_sampler(max_tries=60)                       # once per process (builds the pool tables)
    w = PS.sample(log, at, side, 8, key=did, root_choice_in_log=..., screen_cache=cache)   # a worlds.Worlds
    w.poolsets   # {"per_world": [[per-slot table source]], "counts": {...}}
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pool_teams as PT  # noqa: E402

from main.value_search import worlds as W  # noqa: E402
from main.value_search.world_draws import BANNED_ITEMS, BANNED_MOVES, Prior  # noqa: E402

CLASSES = ("moves", "items", "ability", "spreads", "hidden_power")


def pool_tables(teams: Sequence[str]) -> Dict[str, Dict[str, dict]]:
    """{species id: {class: {value: frequency}}} over every set of ``teams`` (each pool team counted as often as it
    appears). Moves / items / abilities: share of the species' sets; spreads: share of sets; hidden_power: share of
    the species' Hidden Power sets. Values are FREQUENCIES (not renormalised after drops: the draw renormalises)."""
    n: Counter = Counter()
    acc: Dict[str, Dict[str, Counter]] = {}
    blank_ability = 0
    for t in teams:
        for e in t.split("]"):
            f = W.parse_set(e)
            sid = W.toid(f[1] or f[0])
            n[sid] += 1
            a = acc.setdefault(sid, {c: Counter() for c in CLASSES})
            mv = W.typed_moves(f)
            for m in set(mv):
                a["moves"][m] += 1
            hp = [m for m in mv if W.is_hp(m)]
            if hp:
                a["hidden_power"][hp[0][len("hiddenpower"):]] += 1
            it = W.toid(f[2])
            if it:
                a["items"][it] += 1
            ab = W.toid(f[3])
            if ab:
                a["ability"][ab] += 1
            else:
                blank_ability += 1
            sp = W.Spread.of(f)
            a["spreads"][sp.key()] += 1
    out: Dict[str, Dict[str, dict]] = {}
    for sid, a in acc.items():
        nh = sum(a["hidden_power"].values())
        out[sid] = {
            "moves": {m: c / n[sid] for m, c in sorted(a["moves"].items()) if m not in BANNED_MOVES},
            "items": {i: c / n[sid] for i, c in sorted(a["items"].items()) if i not in BANNED_ITEMS and i != "nothing"},
            "ability": {x: c / n[sid] for x, c in sorted(a["ability"].items())},
            "spreads": {k: c / n[sid] for k, c in sorted(a["spreads"].items())},
            "hidden_power": {t: c / nh for t, c in sorted(a["hidden_power"].items())} if nh else {},
        }
    out["__meta__"] = {"sets": dict(n), "blank_ability": {"count": blank_ability}}   # type: ignore[dict-item]
    return out


class PoolSetsPrior(Prior):
    """Smogon (``base``) for the species draw; the pool's per-species tables for every SET fact the pool holds."""

    def __init__(self, tables: Dict[str, Dict[str, dict]], base: Optional[Prior] = None):   # noqa: D107
        self.base = base or Prior()
        b = self.base
        self._p, self._s, self.usage, self.pool, self._cache = b._p, b._s, b.usage, b.pool, b._cache
        self.tables = {k: v for k, v in tables.items() if k != "__meta__"}
        self.meta = tables.get("__meta__", {})
        #: (class, species) → times a draw / sampler read Smogon's table because the pool has none for it
        self.fallback: Counter = Counter()
        self.reads: Counter = Counter()

    def source(self, sid: str, cls: str) -> str:
        t = self.tables.get(sid, {}).get(cls)
        if cls == "hidden_power":
            return "pool" if sid in self.tables else "smogon_fallback"
        return "pool" if t else "smogon_fallback"

    def _get(self, cls: str, sid: str, smogon):
        self.reads[cls] += 1
        if self.source(sid, cls) == "pool":
            t = self.tables[sid][cls]
            if cls == "hidden_power" and not t:
                return {x: 1.0 for x in W.HP_TYPES}       # the species never carries one in the pool: P(any) = 0
            return t
        self.fallback[(cls, sid)] += 1
        return smogon(sid)

    def moves(self, sid: str) -> Dict[str, float]:
        return self._get("moves", sid, self.base.moves)

    def items(self, sid: str) -> Dict[str, float]:
        return self._get("items", sid, self.base.items)

    def ability(self, sid: str) -> Dict[str, float]:
        return self._get("ability", sid, self.base.ability)

    def spreads(self, sid: str):
        return self._get("spreads", sid, self.base.spreads)

    def hidden_power(self, sid: str) -> Dict[str, float]:
        return self._get("hidden_power", sid, self.base.hidden_power)

    def mates(self, sid: str) -> Dict[str, float]:
        return self.base.mates(sid)

    def name(self, sid: str) -> str:
        return self.base.name(sid)


class PoolSetsSampler:
    def __init__(self, max_tries: int = 60, base: Optional[Prior] = None, pool: Optional[PT.Pool] = None,
                 playout_lib=None):
        self.pool = pool or PT.load_pool()
        self.P = PoolSetsPrior(pool_tables(self.pool.teams), base)
        self.ws = W.WorldSampler(prior=self.P, max_tries=max_tries, reveal="off", playout_lib=playout_lib)

    def namespace(self, screen_cache: Optional[dict], key: str) -> dict:
        """POOLSETS' own sub-cache: its seen-mon samplers apart, the gate verdicts shared (pure in the team text)."""
        if screen_cache is None:
            return {}
        sub = screen_cache.setdefault(("POOLSETS", key), {})
        sub[("verdict", key)] = screen_cache.setdefault(("verdict", key), {})
        return sub

    def sample(self, log: dict, at: int, side: str, J: int, key: str, root_choice_in_log: bool = False,
               screen_cache: Optional[dict] = None) -> "W.Worlds":
        f0 = Counter(self.P.fallback)
        w = self.ws.sample(log, at, side, J, key, root_choice_in_log=root_choice_in_log,
                           screen_cache=self.namespace(screen_cache, key))
        w.poolsets = provenance(w, self.P, Counter(self.P.fallback) - f0)   # type: ignore[attr-defined]
        w.extra = {"poolsets": w.poolsets}                                   # type: ignore[attr-defined]
        return w


def poolsets_sampler(max_tries: int = 60, base: Optional[Prior] = None, pool: Optional[PT.Pool] = None
                     ) -> PoolSetsSampler:
    return PoolSetsSampler(max_tries=max_tries, base=base, pool=pool)


def provenance(w: "W.Worlds", P: PoolSetsPrior, fb: Counter) -> dict:
    """Per world, per slot: whether the species' set tables came from the pool or fell back to Smogon."""
    per: List[List[dict]] = []
    agg: Counter = Counter()
    for team in w.teams:
        rows = []
        for i, e in enumerate(team.split("]")):
            f = W.parse_set(e)
            sid = W.toid(f[1] or f[0])
            r = {"slot": i, "species": sid, "tables": "pool" if sid in P.tables else "smogon_fallback"}
            agg[r["tables"]] += 1
            rows.append(r)
        per.append(rows)
    return {"per_world": per, "counts": dict(agg),
            "fallback_reads": {f"{c}:{s}": n for (c, s), n in sorted(fb.items())}}


def table_summary(tables: Dict[str, Dict[str, dict]]) -> Dict[str, object]:
    sp = {k: v for k, v in tables.items() if k != "__meta__"}
    return {"species": len(sp), "sets": sum(tables["__meta__"]["sets"].values()),   # type: ignore[index]
            "blank_ability": tables["__meta__"]["blank_ability"]}   # type: ignore[index]
