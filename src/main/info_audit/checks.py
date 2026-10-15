"""The checks of the information-lever audit suite (``main.info_audit``; design §10). Each returns a verdict dict
``{"check", "pass", …counts, "examples": [≤ 10]}``.

SEARCH (the world builder, ``main.value_search.worlds``):

* ``template_identity`` — the choice template rendered against the TRUE team reproduces the true prefix (a rewritten
  slot rendered one off is invisible to a replay when the choice never executed: the hidden-info study's A1 teeth);
* ``oracle_identity`` — under the ``oracle`` preset every world IS the true team and the true prefix;
* ``regate`` — every returned world replays to the searched side's byte-equal rows and protocol (A3);
* ``consistency`` — each slot's fact record names the mon the ENGINE built at that slot, and every match flag is that
  mon's (``main.value_search.world_guard``; the 2026-10-15 never-seen record mis-pairing);
* ``legality`` — no world team breaks a gen3ou rule the true team does not (A4);
* ``genders`` — the never-seen slots that draw a gender at construction are exactly the true team's (A6's leak);
* ``provenance`` — every fact is revealed / told / drawn exactly as the spec says (a fact told without a lever, or a
  mass-1 truth lever whose world value is not the truth, fails);
* :class:`Leakage` — the per-class LEAK read: drawn facts match the truth no more often than the honest belief
  expects (the seen mons' drawn moves against their exact conditional-Poisson inclusion; the screen's independent
  proposals against the prior's mass on the truth), plus the descriptive truth-match rate of every (part, class, how).

THE LOG SOURCE: ``log_consistency`` — the Python source of what play revealed (``agents.info_levers.revealed``)
against the Rust reading the observation encodes, decision by decision of a real battle.

The observation and injection consumers' checks live beside their code (``rust_env/tests/oracle_reveal_test.rs``,
``utils/rust_env/oracle_reveal_integration_test.py``, ``agents/info_levers/inject_integration_test.py``); the suite's
runner (``__main__``) collects every verdict into one report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from agents.info_levers import revealed as RV
from agents.info_levers.spec import InfoReveal
from main.value_search import worlds as W


def _verdict(name: str, ok: bool, **kw) -> dict:
    return {"check": name, "pass": bool(ok), **kw}


def _tok(t: str) -> Tuple[str, str]:
    head, _, rest = t.partition(" ")
    return head, W.toid(rest)


# ------------------------------------------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------------------------------------------

def template_identity(root: "W.Root") -> dict:
    got = W.render(root.tmpl, root.team)
    bad = [{"i": i, "true": a, "rendered": b} for i, (a, b) in enumerate(zip(root.cmds, got))
           if a[0] != b[0] or _tok(a[1]) != _tok(b[1])]
    return _verdict("template_identity", not bad and len(got) == len(root.cmds), commands=len(got),
                    mismatches=len(bad), examples=bad[:10])


def oracle_identity(root: "W.Root", worlds: "W.Worlds") -> dict:
    bad = []
    for j, (t, lg) in enumerate(zip(worlds.teams, worlds.logs)):
        pre = lg["cmds"][: len(root.cmds)]
        want = [f"FORCELOSE {tok}" if s == "forcelose" else f"CHOOSE {s} {tok}" for s, tok in root.cmds]
        if t != root.team or [_tok(x.split(" ", 2)[-1]) for x in pre] != [_tok(x.split(" ", 2)[-1]) for x in want]:
            bad.append({"world": j, "team_equal": t == root.team})
    return _verdict("oracle_identity", not bad, worlds=len(worlds.teams), mismatches=len(bad), examples=bad[:10])


def regate(root: "W.Root", worlds: "W.Worlds", label: str = "regate") -> dict:
    from agents.battle.core_replay import run_core

    out = run_core([root.rec(t, f"{label}#{j}", W.render(root.tmpl, t)) for j, t in enumerate(worlds.teams)],
                   trackers=True, obs=True)
    bad = []
    hp = [0, 0]                      # (seen mons whose EXACT HP at the root equals the truth's, seen mons) — descriptive
    seen = [s for s in root.sids if s in root.kept]
    for j, r in enumerate(out):
        ok = bool(r["ok"])
        if ok:
            _, rows = W.reveals(r, root.viewer, root.n)
            ok = rows == root.ref and W.protocol_lines(r, root.viewer, root.n) == root.ref_proto
        if ok:
            ent = [e for e in r["trackers"][root.viewer] if "obs" in e]
            cond = W.opp_conditions(r, root.oi, int(ent[root.n]["after"]))
            hp[0] += sum(cond.get(s) == root.ref_cond.get(s) for s in seen)
            hp[1] += len(seen)
        if not ok:
            bad.append({"world": j, "core_ok": bool(r["ok"])})
    return _verdict("regate", not bad, worlds=len(out), failures=len(bad), seen_exact_hp_equal=hp[0],
                    seen_mons=hp[1], examples=bad[:10])


def consistency(root: "W.Root", worlds: "W.Worlds") -> dict:
    """SLOT ↔ SPECIES ↔ FACTS ↔ ENGINE (``main.value_search.world_guard``, the builder's own throwing guard, here as a
    verdict so a world from ANY builder, or a tampered one, is read): each slot's fact record names the mon the engine
    built there, and every match flag is that mon's (the 2026-10-15 GIGO: never-seen records stayed at the slot each
    mon was drawn for)."""
    from main.value_search import world_guard as WG

    bad = WG.problems(root, worlds.teams, worlds.facts)
    return _verdict("consistency", not bad, worlds=len(worlds.teams), inconsistent=len(bad),
                    examples=[{"world": j, "problems": p[:5]} for j, p in list(bad.items())[:10]])


def legality(root: "W.Root", worlds: "W.Worlds") -> dict:
    base = W.team_problems(root.team)
    bad = [{"world": j, "new": sorted(W.team_problems(t) - base)} for j, t in enumerate(worlds.teams)
           if W.team_problems(t) - base]
    return _verdict("legality", not bad, worlds=len(worlds.teams), violations=len(bad), examples=bad[:10])


def _draws_or_pads(team: str) -> set:
    pad = W._gender_pad()
    out = set()
    for i, e in enumerate(team.split("]")):
        f = W.parse_set(e)
        if W._draws(e) or (pad and f[7].endswith(pad)):
            out.add(i)
    return out


def genders(root: "W.Root", worlds: "W.Worlds") -> dict:
    never = {i for i, s in enumerate(root.sids) if s not in root.kept}
    want = W.gender_draw_slots(root.team) & never
    bad = [{"world": j, "draws": sorted(_draws_or_pads(t) & never), "true": sorted(want)}
           for j, t in enumerate(worlds.teams) if (_draws_or_pads(t) & never) != want]
    return _verdict("genders", not bad, worlds=len(worlds.teams), mismatches=len(bad), examples=bad[:10])


def gender_independence(worlds: "W.Worlds", pad: Optional[str]) -> dict:
    """SEED REUSE through the turn-0 GENDER dice (the blind replica's finding 1): a world is replayed on the true seed,
    so the never-seen species would be conditioned if any world were REJECTED for spending different gender draws.
    With the engine's pad every world spends exactly the truth's draws whatever its species: no world may have been
    rejected on its gender count."""
    n = int(worlds.fails.get("gender_count", 0) or 0)
    return _verdict("gender_independence", pad is not None and n == 0, gender_count_rejections=n,
                    pad=pad is not None, examples=[])


def playout_open(root: "W.Root", worlds: "W.Worlds", sampler: "W.WorldSampler") -> dict:
    """Every returned world OPENS in the playout tool (``rust_env_playout_open``), the search's own replay — the
    core_events gate accepts some choice logs the playout tool refuses (the blind replica's finding 3)."""
    bad = []
    for j, lg in enumerate(worlds.logs):
        why = sampler.playout_refusal(lg, root.at, root.side)
        if why is not None:
            bad.append({"world": j, "refusal": why[:200]})
    return _verdict("playout_open", not bad, worlds=len(worlds.logs), refused=len(bad), examples=bad[:10])


#: Hidden facts that SPEND A DIE WITHOUT PRINTING A LINE (the blind replica's finding 2; verified in
#: deps/pokemon-showdown 2026-10-12). A world is replayed on the TRUE seed, so it must spend the truth's dice: the gate
#: then sees these although an honest player cannot. ITEMS: King's Rock (a 10 % flinch secondary added to most damaging
#: moves — one roll per hit, data/mods/gen3/items.ts), Focus Band (a 1-in-10 roll on EVERY move damage its holder takes,
#: evaluated before the lethal check, data/items.ts). ABILITIES: Cute Charm, Effect Spore, Flame Body, Poison Point,
#: Static (a roll per contact hit taken, data/mods/gen3/abilities.ts), Shed Skin (a roll per turn while statused).
#: SPEED TIES: speedSort shuffles a tied group (sim/battle.ts), a die only when a tie exists. NOT one: Quick Claw (gen 3
#: rolls it battle-wide every turn, sim/battle.ts, so holding it spends nothing; its effect is the printed move order).
SILENT_DICE_ITEMS = ("King's Rock", "Focus Band")
SILENT_DICE_ABILITIES = frozenset({"cutecharm", "effectspore", "flamebody", "poisonpoint", "static", "shedskin"})
#: an item with no battle effect for the mon holding it (Ditto's only), the honest baseline of an item swap
NEUTRAL_ITEMS = ("Metal Powder", "Lucky Punch")


def _speed_evs(f: List[str], delta: int) -> Optional[List[str]]:
    """``f`` with its Speed EVs moved so the L100 Speed changes by EXACTLY ``delta`` (±1), else None."""
    from agents.gen3_data import natures, priors, species
    from agents.info_levers.packed import Spread

    sd = species.get(W.species_id(f))
    if sd is None or "spe" not in sd.base_stats:
        return None
    sp = Spread.of(f)
    nm = sp.nature[:1].upper() + sp.nature[1:].lower() if sp.nature else ""
    mult = float(natures.multipliers().get(nm, {}).get("spe", 1.0))
    base, ev, iv = int(sd.base_stats["spe"]), int(sp.evs[5]), int(sp.ivs[5])
    now = priors.gen3_stat(base, ev, mult, iv)
    for step in range(4, 256, 4):
        e2 = ev + step * (1 if delta > 0 else -1)
        if not 0 <= e2 <= 252:
            return None
        got = priors.gen3_stat(base, e2, mult, iv) - now
        if got == delta:
            evs = list(sp.evs)
            evs[5] = e2
            g = list(f)
            Spread(sp.nature, tuple(evs), sp.ivs).write(g)
            return g
        if abs(got) > abs(delta):
            return None
    return None


def seed_reuse(root: "W.Root", prior: Optional["W.Prior"] = None) -> dict:
    """The MEASURED size of the seed-reuse residual at one decision (design §7.5), by COUNTERFACTUAL
    FLIPS of the TRUE team: one hidden fact of one SEEN opponent mon changed, everything else true, gated exactly as a
    world is (rows and protocol byte-equal on the true seed). A rejected flip is a value the gate removes.

    ITEM (the mon's item play has not revealed it): swapped to a NEUTRAL item and to each dice item. Truth not a dice
    item: a dice item the gate rejects while it ACCEPTS the neutral one is a dice-only rejection (``item_dice_rejected``;
    weighted by the prior's mass on that item: ``item_mass_removed``). Truth a dice item: the neutral swap rejected =
    the gate tells the dice item (``item_dice_pinned``). A rejected neutral swap is honest inference (``item_honest``).
    ABILITY (not revealed; the species has another ability in the prior and one of the two rolls dice): the swap
    rejected (``ability_rejected``; an UPPER bound — some may be honest inference). SPEED: ±1 Speed point (Spe EVs
    only) can change no printed order except through a tie, so a rejected ±1 is a tie's die (``speed_pinned``).
    Descriptive, except its CONTROL: the unflipped true team must pass the same gate (else the check fails)."""
    from agents.battle.core_replay import run_core

    P = prior or W.Prior()
    ents = list(root.entries)
    # the CONTROL: the unflipped true team must pass the same gate, or every "rejection" below is the instrument's
    jobs: List[Tuple[str, int, str, str]] = [("truth", -1, "", root.team)]     # (kind, mon, value, team text)

    def team_with(i: int, f: List[str]) -> str:
        e = list(ents)
        e[i] = "|".join(f)
        return "]".join(e)

    for i, sid in enumerate(root.sids):
        if sid not in root.kept:
            continue
        mr = root.rv.mons.get(sid)
        f = W.parse_set(ents[i])
        if mr is not None and mr.item is None:
            neutral = NEUTRAL_ITEMS[1] if sid == "ditto" else NEUTRAL_ITEMS[0]
            for it in (neutral,) + SILENT_DICE_ITEMS:
                if W.toid(it) != W.item_id(f):
                    g = list(f)
                    g[2] = it
                    jobs.append(("item", i, it, team_with(i, g)))
        if mr is not None and mr.ability is None:
            alts = [a for a in P.ability(sid) if a != W.ability_id(f)]
            if alts and ({W.ability_id(f), *alts} & SILENT_DICE_ABILITIES):
                for a in alts:
                    g = list(f)
                    g[3] = a
                    jobs.append(("ability", i, a, team_with(i, g)))
        for d in (1, -1):
            g2 = _speed_evs(f, d)
            if g2 is not None:
                jobs.append(("speed", i, str(d), team_with(i, g2)))
    res = run_core([root.rec(t, f"seed_reuse#{k}", W.render(root.tmpl, t)) for k, (_, _, _, t) in enumerate(jobs)],
                   trackers=True, obs=True)
    ok: Dict[Tuple[str, int, str], bool] = {}
    for (kind, i, val, _t), r in zip(jobs, res):
        good = bool(r["ok"])
        if good:
            try:
                _, rows = W.reveals(r, root.viewer, root.n)
                good = rows == root.ref and W.protocol_lines(r, root.viewer, root.n) == root.ref_proto
            except W.WorldError:                    # the flipped battle ended before the root: diverged
                good = False
        ok[(kind, i, val)] = good
    out = {"item_mons": 0, "item_honest": 0, "item_dice_rejected": 0, "item_dice_tested": 0, "item_mass_removed": 0.0,
           "item_truth_dice": 0, "item_dice_pinned": 0, "ability_mons": 0, "ability_rejected": 0,
           "speed_mons": 0, "speed_pinned": 0}
    ex = []
    for i, sid in enumerate(root.sids):
        f = W.parse_set(ents[i])
        neutral = NEUTRAL_ITEMS[1] if sid == "ditto" else NEUTRAL_ITEMS[0]
        if ("item", i, neutral) in ok:
            out["item_mons"] += 1
            truth_dice = W.item_id(f) in {W.toid(x) for x in SILENT_DICE_ITEMS}
            if truth_dice:
                out["item_truth_dice"] += 1
                if not ok[("item", i, neutral)]:
                    out["item_dice_pinned"] += 1
                    ex.append({"mon": sid, "pinned": W.item_id(f)})
            elif not ok[("item", i, neutral)]:
                out["item_honest"] += 1
            else:
                items = P.items(sid)
                tot = sum(items.values()) or 1.0
                for it in SILENT_DICE_ITEMS:
                    out["item_dice_tested"] += 1
                    if not ok[("item", i, it)]:
                        out["item_dice_rejected"] += 1
                        out["item_mass_removed"] += items.get(W.toid(it), 0.0) / tot
                        ex.append({"mon": sid, "rejected": W.toid(it)})
        abil = [k for k in ok if k[0] == "ability" and k[1] == i]
        if abil:
            out["ability_mons"] += 1
            if any(not ok[k] for k in abil):
                out["ability_rejected"] += 1
                ex.append({"mon": sid, "ability_told": W.ability_id(f)})
        sp = [k for k in ok if k[0] == "speed" and k[1] == i]
        if sp:
            out["speed_mons"] += 1
            if any(not ok[k] for k in sp):
                out["speed_pinned"] += 1
                ex.append({"mon": sid, "speed_pinned": [k[2] for k in sp if not ok[k]]})
    out["item_mass_removed"] = round(out["item_mass_removed"], 6)
    return _verdict("seed_reuse", ok[("truth", -1, "")], replays=len(jobs) - 1, examples=ex[:10], **out)


def provenance(spec: InfoReveal, worlds: "W.Worlds") -> dict:
    bad = []

    def one(j: int, slot: int, part: str, cls: str, fact: dict, true_mon: bool = True) -> None:
        lv = spec.lever(cls, part)
        how = fact.get("how")
        if how == "told" and (lv is None or not true_mon) and cls != "species":
            bad.append({"world": j, "slot": slot, "fact": f"{cls}:{part}", "why": "told without a lever"})
        elif how == "told" and cls == "species" and lv is None:
            bad.append({"world": j, "slot": slot, "fact": f"{cls}:{part}", "why": "told without a lever"})
        elif how == "drawn" and lv is not None and lv.mass_bp >= 10_000 and true_mon:
            bad.append({"world": j, "slot": slot, "fact": f"{cls}:{part}", "why": "a mass-1 lever not applied"})
        elif how == "told" and lv is not None and lv.source == "truth" and lv.mass_bp >= 10_000 \
                and not fact.get("match"):
            bad.append({"world": j, "slot": slot, "fact": f"{cls}:{part}", "why": "a mass-1 truth lever not the truth"})
        elif how == "revealed" and not fact.get("match"):
            bad.append({"world": j, "slot": slot, "fact": f"{cls}:{part}", "why": "a revealed fact not the truth"})
        elif how == "told" and lv is not None and lv.source == "placebo" and fact.get("match") \
                and cls not in ("moves",):
            bad.append({"world": j, "slot": slot, "fact": f"{cls}:{part}", "why": "a placebo equal to the truth"})

    n = 0
    for j, fl in enumerate(worlds.facts):
        for f in fl:
            part = "seen" if f["seen"] else "unseen"
            if not f["seen"]:
                one(j, f["slot"], part, "species", f["species_fact"])
            for cls in ("moves", "item", "ability", "spread"):
                if cls in f:
                    n += 1
                    one(j, f["slot"], part, cls, f[cls], f.get("true_mon", True))
    return _verdict("provenance", not bad, facts=n, violations=len(bad), examples=bad[:10])


@dataclass
class Leakage:
    """The per-class LEAK read over many worlds (module docstring). ``z_max``: how many standard deviations above
    the honest expectation a drawn class may match the truth before the check fails."""

    z_max: float = 4.0
    moves: Dict[str, List[float]] = field(default_factory=dict)        # part → [hits, expected, var, slots]
    props: Dict[str, List[float]] = field(default_factory=dict)        # class → [proposals, hits, expected]
    match: Dict[str, List[int]] = field(default_factory=dict)          # "part:class:how" → [n, hits]
    samplers: Dict[str, int] = field(default_factory=lambda: {"n": 0, "chain": 0, "failed": 0})

    def add(self, worlds: "W.Worlds") -> None:
        for fl in worlds.facts:
            for f in fl:
                part = "seen" if f["seen"] else "unseen"
                if not f["seen"]:
                    self._m(f"unseen:species:{f['species_fact']['how']}", f["species_fact"]["match"])
                for cls in ("moves", "item", "ability", "spread"):
                    if cls not in f:
                        continue
                    self._m(f"{part}:{cls}:{f[cls]['how']}", f[cls]["match"])
                if f["seen"] and f["moves"]["how"] in ("drawn", "told") and "expected_true_hits" in f["moves"]:
                    m, k = f["moves"], max(1, f["moves"]["slots"])
                    st = self.moves.setdefault(f"{part}:{f['moves']['how']}", [0.0, 0.0, 0.0, 0.0])
                    st[0] += m["true_hits"]
                    st[1] += m["expected_true_hits"]
                    st[2] += m["expected_true_hits"] * max(0.0, 1.0 - m["expected_true_hits"] / k)
                    st[3] += m["slots"]
        for sm in worlds.samplers:
            self.samplers["n"] += 1
            self.samplers["chain"] += int(sm["steps"] > 0)
            self.samplers["failed"] += int(sm["failed"])
            for cls, (n, hits, exp) in sm["prop_stats"].items():
                st = self.props.setdefault(cls, [0.0, 0.0, 0.0])
                st[0] += n
                st[1] += hits
                st[2] += exp

    def _m(self, key: str, hit: bool) -> None:
        st = self.match.setdefault(key, [0, 0])
        st[0] += 1
        st[1] += int(bool(hit))

    def verdicts(self) -> List[dict]:
        out = []
        for key, (hits, exp, var, slots) in sorted(self.moves.items()):
            part, how = key.split(":")
            z = (hits - exp) / math.sqrt(max(var, 1e-9)) if var > 1e-9 else (0.0 if abs(hits - exp) < 1e-6 else 99.0)
            if how == "drawn":          # a drawn move matches the truth no more often than its honest inclusion
                out.append(_verdict(f"leak:moves:{part}:drawn", z <= self.z_max, slots=int(slots), true_hits=int(hits),
                                    expected=round(exp, 2), z=round(z, 2)))
            else:                       # a told move hits the truth exactly as often as the lever's mixture says
                out.append(_verdict(f"dose:moves:{part}:told", abs(z) <= self.z_max, slots=int(slots),
                                    true_hits=int(hits), expected=round(exp, 2), z=round(z, 2)))
        for cls, (n, hits, exp) in sorted(self.props.items()):
            var = exp * max(0.0, 1.0 - exp / max(n, 1.0))
            z = (hits - exp) / math.sqrt(max(var, 1e-9))
            out.append(_verdict(f"leak:{cls}:seen:proposals", z <= self.z_max, proposals=int(n), true_hits=int(hits),
                                expected=round(exp, 2), z=round(z, 2)))
        return out

    def table(self) -> Dict[str, dict]:
        return {k: {"n": n, "truth_match": round(h / n, 4) if n else None} for k, (n, h) in sorted(self.match.items())}


# ------------------------------------------------------------------------------------------------
# THE LOG SOURCE against the Rust reading
# ------------------------------------------------------------------------------------------------

def _species_num(sid: str) -> Optional[int]:
    from agents.gen3_data import species

    sd = species.get(sid)
    return int(sd.num) if sd is not None else None


def _move_num(mid: str) -> Optional[int]:
    from agents.gen3_data import moves

    md = moves.get(mid)
    return int(md.num) if md is not None else None


def _item_num(iid: str) -> Optional[int]:
    from agents.gen3_data import items

    it = items.get(iid)
    return int(it.num) if it is not None else None


def log_consistency(res: dict, viewer: int, label: str = "") -> dict:
    """At every decision of ``viewer`` in a ``core_events --trackers --obs`` result: each opponent mon's moves and
    original item by the Python log source vs the opponent block of the observation row the Rust reading encoded.
    ``python_only`` (the source claims a fact the reading does not show) is the leak direction for the builder and
    fails; ``obs_only`` (the reading shows a fact the conservative source does not claim) is counted."""
    import numpy as np

    from agents.battle.core_obs import wrap_row
    from agents.observation import constants as C

    hp_nums = {n for n in (_move_num("hiddenpower"),) if n is not None}
    hp_nums |= {n for t in W.HP_TYPES for n in (_move_num("hiddenpower" + t),) if n is not None}
    ent = [e for e in res["trackers"][viewer] if "obs" in e]
    counts = {"decisions": 0, "mons": 0, "moves_agree": 0, "moves_python_only": 0, "moves_obs_only": 0,
              "items_agree": 0, "items_python_only": 0, "items_obs_only": 0}
    ex: List[dict] = []
    for d, e in enumerate(ent):
        row = np.asarray(wrap_row(e["obs"]), dtype=np.float64)
        rv = RV.from_lines(RV.viewer_lines(res, viewer, int(e["after"])), viewer)
        counts["decisions"] += 1
        obs: Dict[int, Tuple[set, Optional[int]]] = {}
        for j in range(C.TEAM_SIZE):
            b = C.OFFSET_OPP_TEAM + j * C.POKEMON_FULL_DIM
            if row[b + C.POKEMON_SPECIES_KNOWN_OFFSET] <= 0:
                continue
            num = int(round(row[b + C.POKEMON_SPECIES_OFFSET]))
            mv = set()
            for m in range(4):
                mb = b + C.POKEMON_MOVES_OFFSET + m * C.MOVE_SLOT_DIM
                if row[mb + 6] > 0:
                    mv.add(int(round(row[mb])))
            it = int(round(row[b + C.POKEMON_ITEMS_OFFSET])) if row[b + C.POKEMON_ITEMS_OFFSET + 1] > 0 else None
            obs[num] = (mv, it)
        for sid, m in rv.mons.items():
            num = _species_num(sid)
            if num is None or num not in obs:
                continue
            counts["mons"] += 1
            o_mv, o_it = obs[num]
            p_mv = {n for n in (_move_num(x) for x in m.moves) if n is not None}
            # a Hidden Power is one fact whatever typing either side writes
            norm = (lambda s: {("HP" if n in hp_nums else n) for n in s})
            pm, om = norm(p_mv), norm(o_mv)
            counts["moves_agree"] += len(pm & om)
            counts["moves_python_only"] += len(pm - om)
            counts["moves_obs_only"] += len(om - pm)
            if pm - om and len(ex) < 10:
                ex.append({"battle": label, "decision": d, "species": sid, "python_only": sorted(map(str, pm - om))})
            p_it = _item_num(m.item) if m.item else None
            if p_it is not None and o_it is not None and p_it == o_it:
                counts["items_agree"] += 1
            elif p_it is not None and o_it != p_it:
                counts["items_python_only"] += 1
                if len(ex) < 10:
                    ex.append({"battle": label, "decision": d, "species": sid, "item": m.item, "obs_item": o_it})
            elif p_it is None and o_it is not None:
                counts["items_obs_only"] += 1
    ok = counts["moves_python_only"] == 0 and counts["items_python_only"] == 0
    return _verdict("log_consistency", ok, **counts, examples=ex)


def merge(verdicts: Sequence[dict]) -> dict:
    """Sum the counts of several verdicts of ONE check (pass = every one passed)."""
    if not verdicts:
        return {}
    out: dict = {"check": verdicts[0]["check"], "pass": all(v["pass"] for v in verdicts), "examples": []}
    for v in verdicts:
        for k, x in v.items():
            if k in ("check", "pass"):
                continue
            if k == "examples":
                out["examples"] = (out["examples"] + list(x))[:10]
            elif isinstance(x, (int, float)) and not isinstance(x, bool):
                out[k] = out.get(k, 0) + x
    return out
