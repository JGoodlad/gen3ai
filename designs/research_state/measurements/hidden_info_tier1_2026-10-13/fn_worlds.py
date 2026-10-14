"""FN, HEAD SIDE: F's worlds with the never-seen SPECIES and the hidden MOVES drawn from the NETWORK's own belief at the
root (`design_hidden_info_binding.md` §12.13.3: the value-discrimination pilot's C′ sampler re-based onto the
information-lever builder). The belief rows come from `fn_belief.py` (one CPU forward at the checkpoint's own pin).

What differs from F (``WorldSampler(reveal="off")``), and nothing else:

* a NEVER-SEEN slot's species ∝ the network's X5 species presence π over the candidates not already on the team (the
  pilot's B′ rule; candidates = the builder's ``Prior.pool``, as F's). Only when π puts NO mass on the candidates does
  the slot take F's own weights (the Smogon teammate co-occurrence with the revealed species, else usage) — COUNTED;
* a SEEN mon's unrevealed moves: the CP presence draw (``world_draws.draw_moves``, F's own) over the network's per-mon
  move presence for that mon instead of the Smogon move prior;
* a never-seen mon whose drawn species is one of the root's HYPOTHESIS species: its moves from that hypothesis'
  presence (the rest of its set — item, ability, spread — Smogon, as F's);
* a mon the network's candidates cannot fill (fewer positive, engine-runnable candidates than move slots), or a seen
  species the belief row has no presence for: Smogon for that mon, COUNTED.

Everything else is F's: items, abilities, spreads, every never-seen non-hypothesis set (Smogon), the genders, the
template, the gate, the playout check. A SEEN mon's item / ability / spread (its IVs: whether it carries a Hidden Power
and of which type) come from F's SCREEN: FN runs on the SAME ``screen_cache`` F was built on, its screen keys equal F's
(the spec is ``off`` for both and no key field reads the prior). When FN runs on a FRESH cache (F built in another
process), it builds the seen-mon samplers itself ON F'S PARAMETERS: the builder's ``sampler_for`` would read
``hp_any`` (P(the set holds a Hidden Power)) from ``Prior.moves`` — the NETWORK's presence for a seen mon — so the
samplers dict is swapped for a guarded copy (:class:`_FScreen`) whose ``__setitem__`` resets ``hp_any`` to Smogon's
and checks every other parameter equals F's construction (a difference raises :class:`FNScreenMissing`); the
sampler's rng is seeded by (key, screen key) alone, so its draws then equal F's by determinism
(``fn_worlds_screen_test.py`` checks both the shared-cache and the fresh-cache case against F). Consequence (a
named deviation from the pilot's C′): a seen mon's Hidden Power INCLUSION and TYPE follow F's screen, not the network;
the network's typed Hidden Power channels are used only where the screen's IVs carry that type (forced) — and dropped
otherwise (the remaining presences renormalised by ``mix_presence``, as for F).

Interface (the harness, per decision, AFTER F on the same cache)::

    from fn_worlds import fn_sampler, load_beliefs
    beliefs = load_beliefs(cp, "registered")                  # {id: row}
    S = fn_sampler(beliefs[did], max_tries=60)                # or fn_sampler(row, base=<shared Prior>, ws=<shared>)
    w = S.sample(log, at, side, 8, key=did, root_choice_in_log=..., screen_cache=cache)   # a worlds.Worlds
    w.fn   # {"per_world": [[per-slot provenance]], "counts": {...}, "row_check": {...}}

``screen_cache``: F's per-decision cache (shared seen-mon samplers and gate verdicts), or a fresh dict / None (FN
rebuilds the samplers on F's parameters). Raises ``worlds.WorldError`` (F's exclusions) or :class:`FNScreenMissing`
(a seen-mon sampler whose construction differs from F's) — both an EXCLUSION for the harness.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Set

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import t1_common as T  # noqa: E402

from main.value_search import world_draws as WD  # noqa: E402
from main.value_search import worlds as W  # noqa: E402
from main.value_search.world_draws import BANNED_MOVES, Prior, is_hp  # noqa: E402


_SHARED: dict = {}


class FNScreenMissing(W.WorldError):
    """FN asked for a seen-mon sampler F did not build on this screen cache."""


def load_beliefs(cp: str, states: str = "registered") -> Dict[str, dict]:
    p = T.cp_dir(cp) / f"fn_belief_{states}.jsonl"
    rows = T.read_dir(p.parent, p.name)
    if not rows:
        raise FileNotFoundError(f"{p}: no belief rows (run fn_belief.py at the pin first)")
    return rows


def _num(sid: str) -> int:
    from agents.gen3_data import species

    sd = species.get(sid)
    return int(getattr(sd, "num", 0) or 0) if sd is not None else 0


def fillable(w: Dict[str, float], revealed: Set[str], slots: int) -> bool:
    """Can a CP draw over presence ``w`` fill ``slots`` free move slots whatever the screen says of a Hidden Power?
    (the positive, engine-runnable, non-Hidden-Power candidates outside ``revealed`` must number ≥ ``slots``: with
    the screen's ``none`` rule every Hidden Power channel is dropped)."""
    n = sum(1 for m, p in w.items() if p > 0 and m not in BANNED_MOVES and not is_hp(m) and m not in revealed)
    return n >= slots


class FNPrior(Prior):
    """The Smogon prior (delegated to ``base``) with the network's species presence and per-mon move presence.

    ``seen``: {species id of a mon SEEN at the root: its revealed move ids} (the builder's ``root.kept``) — decides
    which presence (seen / hypothesis) a species' moves read."""

    def __init__(self, row: dict, seen: Dict[str, Set[str]], base: Optional[Prior] = None):   # noqa: D107
        self.base = base or Prior()
        b = self.base
        self._p, self._s, self.usage, self.pool, self._cache = b._p, b._s, b.usage, b.pool, b._cache
        self.row_id = row.get("id")
        self.species_pi: Dict[str, float] = {s: float(v) for s, v in row["species"].items()}
        seen_w = {int(n): w for n, w in row["moves"]["seen"].items()}
        hyp_w = {int(n): w for n, w in row["moves"]["hyp"].items()}
        self.counts: Dict[str, int] = {"species_draws": 0, "species_fallback": 0, "species_dropped_mass_states": 0}
        #: sid → (source, presence or None). source: net_seen | net_hyp | smogon (not a network species) |
        #: smogon_fallback (the network's presence cannot fill the mon, or a seen mon has none)
        self.move_src: Dict[str, tuple] = {}
        self.check = {"seen_mons": len(seen), "seen_no_presence": [], "seen_unfillable": [], "num_collisions": []}
        nums: Dict[int, list] = {}
        for sid in seen:
            nums.setdefault(_num(sid), []).append(sid)
        for n, sids in nums.items():
            if len(sids) > 1:
                self.check["num_collisions"].append(sids)
        for sid, rev in sorted(seen.items()):
            n = _num(sid)
            w = seen_w.get(n)
            if w is None or len(nums.get(n, [])) > 1:
                self.move_src[sid] = ("smogon_fallback", None)
                self.check["seen_no_presence"].append(sid)
                continue
            # the builder's free slots: 4 − |revealed| (a revealed bare Hidden Power counts once)
            free = max(0, 4 - len(rev))
            if not fillable(w, set(rev), free):
                self.move_src[sid] = ("smogon_fallback", None)
                self.check["seen_unfillable"].append(sid)
                continue
            self.move_src[sid] = ("net_seen", dict(sorted(w.items())))
        self._seen = set(seen)
        self._seen_nums = {_num(s) for s in seen}
        self._hyp = hyp_w
        mass = sum(v for s, v in self.species_pi.items() if s not in self.pool)
        self.check["species_mass_outside_pool"] = round(mass, 6)

    # ---- moves: the network's presence where it has one, Smogon otherwise ----
    def moves(self, sid: str) -> Dict[str, float]:
        if sid not in self.move_src:
            n = _num(sid)
            w = self._hyp.get(n) if (sid not in self._seen and n not in self._seen_nums) else None
            if w is None:
                self.move_src[sid] = ("smogon", None)
            elif not fillable(w, set(), 4) and not (fillable(w, set(), 3) and any(is_hp(m) and p > 0
                                                                                  for m, p in w.items())):
                self.move_src[sid] = ("smogon_fallback", None)
            else:
                self.move_src[sid] = ("net_hyp", dict(sorted(w.items())))
        src, w = self.move_src[sid]
        return w if w is not None else self.base.moves(sid)

    def items(self, sid: str) -> Dict[str, float]:
        return self.base.items(sid)

    def ability(self, sid: str) -> Dict[str, float]:
        return self.base.ability(sid)

    def spreads(self, sid: str):
        return self.base.spreads(sid)

    def hidden_power(self, sid: str) -> Dict[str, float]:
        return self.base.hidden_power(sid)

    def mates(self, sid: str) -> Dict[str, float]:
        return self.base.mates(sid)

    def name(self, sid: str) -> str:
        return self.base.name(sid)


# ------------------------------------------------------------------------------------------------
# the never-seen species draw: the network's presence (worlds.never_seen_species, swapped for FN only)
# ------------------------------------------------------------------------------------------------
_ORIG_NSS = W.never_seen_species


def _smogon_weights(P: Prior, cand, revealed) -> Dict[str, float]:
    """F's own weights (`world_draws.never_seen_species.weights`)."""
    import numpy as np

    if revealed:
        w = {s: float(np.mean([P.mates(r).get(s, 0.0) for r in sorted(revealed)])) for s in cand}
        if sum(w.values()) > 0:
            return w
    return {s: float(P.usage.get(s, 0.0)) for s in cand}


def fn_never_seen_species(rng, P, slots, true_sid, revealed, spec):
    if not isinstance(P, FNPrior):
        return _ORIG_NSS(rng, P, slots, true_sid, revealed, spec)
    if spec.lever("species", "unseen") is not None:
        raise W.WorldError("FN is honest: a species:unseen lever has no meaning here")
    from agents.info_levers import mixture as MX

    present = set(revealed)
    out: Dict[int, str] = {}
    prov: Dict[int, dict] = {}
    for s in slots:
        cand = [x for x in P.pool if x not in present]
        w = {x: P.species_pi[x] for x in cand if P.species_pi.get(x, 0.0) > 0}
        src = "net"
        P.counts["species_draws"] += 1
        if not w:
            w = _smogon_weights(P, cand, revealed)
            src = "smogon_fallback"
            P.counts["species_fallback"] += 1
        pick = MX.draw(rng, w)
        if pick is None:
            return None
        present.add(pick)
        out[s] = pick
        prov[s] = {"how": "drawn", "source": "", "match": pick == true_sid[s], "src": src}
    return out, prov


@contextmanager
def _fn_species():
    W.never_seen_species = fn_never_seen_species
    try:
        yield
    finally:
        W.never_seen_species = _ORIG_NSS


def smogon_hp_any(base: Prior, sid: str) -> float:
    """F's ``MonSampler.hp_any`` for ``sid`` (``WorldSampler.sample.sampler_for``, with F's Smogon prior)."""
    return min(1.0, sum(v for mv, v in base.moves(sid).items() if is_hp(mv)))


class _FScreen(dict):
    """The seen-mon samplers FN sees: F's own where they exist. A sampler FN must build itself (F ran in another
    process, or on another cache) is put on F's parameters before it is stored: the builder's ``sampler_for`` reads
    every parameter but one from FN's prior, which delegates them to Smogon (items, abilities, spreads, the Hidden
    Power type prior); the one it reads from ``Prior.moves`` — ``hp_any``, P(the set holds a Hidden Power) — would be
    the NETWORK's, so it is reset to Smogon's (the sampler's rng is seeded by (key, screen key) alone and is untouched
    at construction). Every other parameter is checked against F's construction; a difference raises."""

    def __init__(self, own: dict, base: Prior, log: list):
        super().__init__(own)
        self.base, self.log = base, log

    def __setitem__(self, k, v):
        from main.value_search import world_screen as WS

        sid = v.sid
        want = {"items": self.base.items(sid), "abilities": self.base.ability(sid),
                "hp_prior": self.base.hidden_power(sid)}
        for f, x in want.items():
            if getattr(v, f) != x:
                raise FNScreenMissing(f"FN's seen-mon sampler {k}: {f} differs from F's construction")
        if v.spreads.joint != WS.SpreadPrior(self.base.spreads(sid)).joint:
            raise FNScreenMissing(f"FN's seen-mon sampler {k}: the spread prior differs from F's")
        if v.samples or v.steps or v.proposals:
            raise FNScreenMissing(f"FN's seen-mon sampler {k} was used before it was stored")
        net = v.hp_any
        v.hp_any = smogon_hp_any(self.base, sid)
        self.log.append({"key": repr(k), "species": sid, "hp_any_network": net, "hp_any_smogon": v.hp_any})
        super().__setitem__(k, v)


class FNSampler:
    def __init__(self, row: dict, max_tries: int = 60, base: Optional[Prior] = None,
                 ws: Optional[W.WorldSampler] = None):
        self.row = row
        # one Smogon Prior and one WorldSampler per process by default (the harness builds an FNSampler per decision;
        # the playout core is loaded once)
        self.base = base or _SHARED.setdefault("base", Prior())
        self.ws = ws or _SHARED.setdefault("ws", W.WorldSampler(prior=self.base, reveal="off"))
        if self.ws.spec.record() != W.SEARCH.validate("off").record():
            raise ValueError("FN's builder must be the honest spec (off)")
        self.max_tries = max_tries
        self.P: Optional[FNPrior] = None

    def sample(self, log: dict, at: int, side: str, J: int, key: str, root_choice_in_log: bool = False,
               screen_cache: Optional[dict] = None) -> "W.Worlds":
        if self.row.get("id") not in (None, key):
            raise ValueError(f"belief row {self.row.get('id')} used for decision {key}")
        if screen_cache is None:
            screen_cache = {}
        root = W.prepare_root(log, at, side, key, root_choice_in_log)
        P = FNPrior(self.row, {s: set(m) for s, m in root.kept.items()}, self.base)
        self.P = P
        own = screen_cache.setdefault(("samplers", key), {})
        n_own = len(own)
        built: list = []
        guard = _FScreen(own, self.base, built)
        screen_cache[("samplers", key)] = guard
        prev_P, prev_sp, prev_mt = self.ws.P, self.ws._spreads, self.ws.max_tries
        self.ws.P, self.ws._spreads, self.ws.max_tries = P, {}, self.max_tries
        try:
            with _fn_species():
                w = self.ws.sample(log, at, side, J, key, root_choice_in_log=root_choice_in_log,
                                   screen_cache=screen_cache)
        finally:
            for k_, v_ in guard.items():           # samplers FN put on F's parameters join the cache (F-identical)
                own.setdefault(k_, v_)
            screen_cache[("samplers", key)] = own
            self.ws.P, self.ws._spreads, self.ws.max_tries = prev_P, prev_sp, prev_mt
        w.fn = provenance(w, P, root)
        w.fn["screen"] = {"samplers_in_cache_before": n_own, "rebuilt_on_F_parameters": built}
        w.extra = {"fn": w.fn}
        return w


def fn_sampler(belief: dict, max_tries: int = 60, base: Optional[Prior] = None,
               ws: Optional[W.WorldSampler] = None) -> FNSampler:
    return FNSampler(belief, max_tries=max_tries, base=base, ws=ws)


def provenance(w: "W.Worlds", P: FNPrior, root: "W.Root") -> dict:
    """Per world, per opponent slot of the RETURNED team: which facts came from the network. The builder's
    ``facts`` are indexed by the pre-gender-placement slot, so a never-seen fact is matched to its team slot by species
    (unique on a team)."""
    per = []
    agg: Dict[str, int] = {}
    for team, facts in zip(w.teams, w.facts):
        ents = team.split("]")
        by_sp = {f.get("species"): f for f in facts}
        rows = []
        for i, e in enumerate(ents):
            f = W.parse_set(e)
            sid = W.toid(f[1] or f[0])
            fa = by_sp.get(sid, {})
            seen = sid in root.kept
            if seen:
                rev = root.kept[sid]
                full = len(rev) >= 4
                msrc = "revealed" if full else P.move_src.get(sid, ("smogon", None))[0]
                r = {"slot": i, "species": sid, "seen": True, "species_src": "revealed", "moves_src": msrc,
                     "item_ability_spread_src": "F_screen"}
            else:
                ssrc = (fa.get("species_fact") or {}).get("src", "?")
                msrc = P.move_src.get(sid, ("smogon", None))[0]
                r = {"slot": i, "species": sid, "seen": False, "species_src": ssrc, "moves_src": msrc,
                     "item_ability_spread_src": "smogon"}
            for k in ("species_src", "moves_src"):
                agg[f"{k}:{r[k]}"] = agg.get(f"{k}:{r[k]}", 0) + 1
            rows.append(r)
        per.append(rows)
    return {"per_world": per, "counts": dict(sorted(agg.items())), "draw_counts": dict(P.counts),
            "row_check": P.check}


__all__ = ["FNPrior", "FNSampler", "FNScreenMissing", "fn_sampler", "load_beliefs", "fillable", "provenance",
           "fn_never_seen_species", "WD"]
