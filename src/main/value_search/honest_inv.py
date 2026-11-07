"""THE LEAK TEST (INV): perturb only what the searched side cannot see, and its decision record must not move
(``designs/endstate/design_sprime_honest.md`` §3).

For a decision D and an ALTERNATIVE TRUTH D′ that ships the searched side exactly the same information set I (checked
by replay: :func:`same_info`), the live searcher's decision record must be byte-identical on :data:`INV_FIELDS` —
searched or excluded, the pick, Q̂, log π, root V, the playouts spent, the worlds' tries, their leak block, the key
the search drew under, the honest build record and the world teams. Pointwise, so it passes or fails
deterministically (root rule 8).

THE ALTERNATIVE TRUTHS (each built from D's own log, kept only if it ships I):

* **P1** a never-seen mon replaced by another prior draw (same gender-draw status, so D's dice stay aligned);
* **P2** a seen mon's unrevealed move replaced by another prior move (Hidden Power never touched);
* **P3** a seen mon's spread moved 4 EVs between two stats where I does not pin it;
* **P4** D's teams and commands on FRESH dice (``pokesim::fresh_dice`` in its fixed-commands mode): every segment's
  dice another's (``p4``), or only the last update's (``p4_last``: a counter drawn there persists to the root);
* **P5** an opponent choice I never saw executed replaced by another offered one.

A CASE names the channel of the leak inventory (§1, L1-L6) its pair demonstrates on ``r0s4_rolls``: with the key
PINNED to I's digest (a call-site choice, no patch) for every channel but L5, which runs today's live key (the deal).
``r0s4_honest`` must give identical records on every case; ``r0s4_rolls`` must give different ones on every case
(the test's teeth: :func:`check` fails on it). THE CONTROL: the honest record at a DIFFERENT I (the same battle one
decision earlier) must differ, so the test is not vacuous.

SEQUENCE MODE (:func:`check_sequence`, the incremental builder ``r0s4_honest_inc``, whose worlds at a decision are
carried from the battle's earlier ones): the WHOLE decision sequence of D and of D′ up to the case's decision
(:func:`sequence`: every decision of the side the live searcher searches, cut from the case's own log), each truth on
its OWN searcher (its own carry cache, so D′ never borrows a carry D built), and the records must be byte-identical
at EVERY decision. D and D′ share every I: each earlier I is a prefix of the case's.

    python -m main.value_search.honest_inv build --selected F --scored F --out FIXTURE.json.gz
    python -m main.value_search.honest_inv check --fixture FIXTURE.json.gz --preset r0s4_honest
    python -m main.value_search.honest_inv check --fixture FIXTURE.json.gz --preset r0s4_honest_inc --sequence
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from main.value_search import info_set as IS

SCHEMA = "gen3_honest_inv_v1"
#: the decision record's fields the leak test compares (never a CPU / wall time, never ``species_hit``: a covariate
#: measured AGAINST the truth after the decision, not part of it)
INV_FIELDS = ("searched", "why", "error", "played", "deviation", "q", "log_pi", "root_v", "spent", "world_tries",
              "leak", "search_key", "honest", "world_teams")
CHANNELS = ("L1", "L2", "L3", "L4", "L5", "L6")


# --------------------------------------------------------------------------------------------- the searcher
def stub_forward(seed: int = 20261105) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """A fixed deterministic network: each row's 11 logits and its V in (0, 1) are read off a keyed digest of the row's
    bytes. INV does not depend on the network, only on what the search reads, so the test needs no checkpoint; a
    digest is bounded (a linear stub on the raw rows saturates V, every Q̂ 0) and moves with ANY difference in a row,
    so a hidden difference the search's playouts reach shows in Q̂."""
    tag = str(seed).encode()

    def f(rows: np.ndarray, masks: np.ndarray) -> np.ndarray:
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        out = np.empty((len(rows), 12), dtype=np.float32)
        for i in range(len(rows)):
            h = hashlib.sha256(tag + rows[i].tobytes()).digest()
            u = np.frombuffer(h[:24], dtype=np.uint16).astype(np.float64) / 65535.0
            out[i, :11] = 4.0 * (u[:11] - 0.5)
            out[i, 11] = 0.05 + 0.9 * u[11]
        return out

    return f


class _Greedy:
    stochastic = False


class _Recorder:
    """A sampler proxy that keeps the last build's world teams (the record carries none for ``r0s4_rolls``)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.teams: List[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def sample(self, *a: Any, **k: Any) -> Any:
        self.teams = []
        w = self.inner.sample(*a, **k)
        self.teams = list(w.teams)
        return w

    def sample_info(self, *a: Any, **k: Any) -> Any:
        self.teams = []
        w = self.inner.sample_info(*a, **k)
        self.teams = list(w.teams)
        return w


@dataclass
class Decision:
    """A decision as the live searcher meets it: the log, its cut, our side, our live row (frame), mask, tokens."""

    log: dict
    at: int
    side: str
    turn: int
    n: int
    mask: List[int]
    tokens: Dict[str, str]
    key: str


def searcher(preset: str, forward=None):
    from main.anchors.search_presets import parse
    from main.anchors.search_side import LiveSearcher
    from main.value_search.honest_inc import HonestIncSampler
    from main.value_search.honest_worlds import HonestSampler
    from main.value_search.worlds import WorldSampler

    p, t = parse(preset)
    if p.dice == "fresh":
        inner: Any = HonestSampler()
    elif p.dice == "fresh_inc":
        inner = HonestIncSampler(from_turn=t)
    else:
        inner = WorldSampler(reveal=p.reveal, max_tries=p.max_tries, dice=p.dice)
    return LiveSearcher(None, _Greedy(), preset, Path("/dev/null"), sampler=_Recorder(inner),
                        forward=forward or stub_forward())


def record(s: Any, dec: Decision, key: Optional[str] = None) -> Dict[str, Any]:
    """The live searcher's record of ``dec`` projected on :data:`INV_FIELDS` (``key``: the record key the search is
    handed; default the decision's own). The live frame is I's root row (identical across truths that ship I)."""
    from agents.battle.core_obs import DTYPE, obs_dim, wrap_row
    from main.value_search import rules

    _info, frame = IS.extract(dec.log, dec.at, dec.side)
    d = {"turn": dec.turn, "n": dec.n, "tokens": dict(dec.tokens),
         "frame": {"dtype": DTYPE, "shape": [obs_dim()], "b64": frame}}
    row = np.array(wrap_row(d["frame"]), dtype=np.float32)
    mask = np.asarray(dec.mask, dtype=np.int8)
    lg = np.where(mask.astype(bool), s.forward(row[None], mask[None].astype(np.uint8))[0][:11].astype(np.float64),
                  -np.inf)
    argmax = rules.policy_order(lg, [int(a) for a in np.flatnonzero(mask)])[0]
    out = s._search(dec.log, dec.at, dec.side, row, mask, d, key or dec.key, int(argmax))
    proj = {k: out.get(k) for k in INV_FIELDS if k in out}
    proj["world_teams"] = list(getattr(s.sampler, "teams", []) or [])
    return json.loads(json.dumps(proj, sort_keys=True, default=str))


def same_info(a: Decision, b: Decision) -> bool:
    """Whether two truths ship the searched side the same I (the replay's check)."""
    try:
        ia, ra = IS.extract(a.log, a.at, a.side)
        ib, rb = IS.extract(b.log, b.at, b.side)
    except ValueError:
        return False
    return ia == ib and ra == rb


# --------------------------------------------------------------------------------------------- alternative truths
def _team_entries(log: dict, i: int) -> List[str]:
    return log["teams"][i].split("]")


def _with_team(dec: Decision, i: int, entries: Sequence[str]) -> Decision:
    lg = copy.deepcopy(dec.log)
    lg["teams"][i] = "]".join(entries)
    return Decision(**{**dec.__dict__, "log": lg})


def _scaffold(dec: Decision):
    from main.value_search.honest_worlds import Scaffold

    info, _ = IS.extract(dec.log, dec.at, dec.side)
    return info, Scaffold.of(info)


def p2_unrevealed_move(dec: Decision, rng: np.random.Generator) -> Iterator[Decision]:
    """P2: each seen mon's unrevealed (non-Hidden-Power) move replaced by another prior move it does not hold."""
    from main.value_search import world_draws as WD
    from main.value_search import world_legality as WL

    info, sc = _scaffold(dec)
    oi = 1 - info.viewer
    ents = _team_entries(dec.log, oi)
    P, legal = WD.Prior(), WL.Legality(pad=None)
    for m in sc.seen:
        k = next((j for j, e in enumerate(ents) if WD.parse_set(e)[0] == m.nick), None)
        if k is None:
            continue
        f = WD.parse_set(ents[k])
        mv = WD.typed_moves(f)
        for slot, x in enumerate(mv):
            if WD.is_hp(x) or x in m.moves:
                continue
            opts = sorted(c for c in P.moves(m.sid) if c not in mv and not WD.is_hp(c))
            for c in [opts[int(j)] for j in rng.permutation(len(opts))][:6]:
                raw = f[4].split(",")
                raw[slot] = c
                g = list(f)
                g[4] = ",".join(raw)
                e = "|".join(g)
                if legal.set_ok(e):
                    yield _with_team(dec, oi, ents[:k] + [e] + ents[k + 1:])
                    break


def p1_never_seen(dec: Decision, rng: np.random.Generator) -> Iterator[Decision]:
    """P1: a never-seen mon replaced by another species' prior set, its gender field kept in kind (named / drawn)."""
    from main.value_search import world_draws as WD
    from main.value_search import world_legality as WL
    from agents.info_levers.consumers import SEARCH

    info, sc = _scaffold(dec)
    oi = 1 - info.viewer
    ents = _team_entries(dec.log, oi)
    seen = {m.nick for m in sc.seen}
    P, legal, spec = WD.Prior(), WL.Legality(pad=None), SEARCH.validate("off")
    on_team = {WD.species_id(WD.parse_set(e)) for e in ents}
    for k, e in enumerate(ents):
        f = WD.parse_set(e)
        if f[0] in seen:
            continue
        for sid in [P.pool[int(j)] for j in rng.permutation(len(P.pool))][:30]:
            if sid in on_team:
                continue
            ns = WD.new_set(rng, P, sid, None, spec, legal=legal.set_ok)
            if ns is None:
                continue
            g = WD.parse_set(ns[0])
            g[7] = f[7] if f[7] else ""
            from main.value_search.worlds import species_draws_gender

            if bool(f[7]) != bool(g[7]) or (not g[7] and not species_draws_gender(sid)):
                continue
            if g[7] and g[7] not in ("M", "F", "N"):
                continue
            cand = "|".join(g)
            if legal.set_ok(cand):
                yield _with_team(dec, oi, ents[:k] + [cand] + ents[k + 1:])
                break


def p3_spread(dec: Decision, rng: np.random.Generator) -> Iterator[Decision]:
    """P3: a seen mon's spread with 4 EVs moved between two stats (kept only where I does not pin it)."""
    from main.value_search import world_draws as WD

    info, sc = _scaffold(dec)
    oi = 1 - info.viewer
    ents = _team_entries(dec.log, oi)
    for m in sc.seen:
        k = next((j for j, e in enumerate(ents) if WD.parse_set(e)[0] == m.nick), None)
        if k is None:
            continue
        f = WD.parse_set(ents[k])
        evs = WD.set_evs(f)
        for a, b in [tuple(int(x) for x in rng.choice(6, 2, replace=False)) for _ in range(6)]:
            if evs[a] < 4 or evs[b] > 248:
                continue
            e2 = list(evs)
            e2[a] -= 4
            e2[b] += 4
            g = list(f)
            g[6] = ",".join(str(x) if x else "" for x in e2)
            yield _with_team(dec, oi, ents[:k] + ["|".join(g)] + ents[k + 1:])


def p4_fresh(dec: Decision, tag: str, from_last: bool = False, tries: int = 1024) -> Optional[Decision]:
    """P4: D's teams and commands on fresh dice (``from_last``: only the last update's), or None when no schedule
    within ``tries`` reproduces I. The teams must name every gender (a fresh START would redraw one)."""
    from main.value_search.honest_worlds import match
    from main.value_search.worlds import log_commands

    info, _sc = _scaffold(dec)
    if any(not WD_gender(e) for t in dec.log["teams"] for e in t.split("]")):
        return None
    segs = info.segments()
    hdr = json.loads(_scaffold(dec)[1].header)
    theirs = [t for s_, t in log_commands(dec.log, dec.at) if s_ == info.opp]
    job = {"label": tag, "formatid": dec.log["format_id"],
           "p1": {"name": dec.log["names"][0], "team": dec.log["teams"][0]},
           "p2": {"name": dec.log["names"][1], "team": dec.log["teams"][1]},
           "dice": f"honest_inv:{tag}:{dec.key}", "tries": tries, "rerolls": 64, "rolls": False, "theirs": theirs}
    if from_last:
        job.update(from_segment=len(segs) - 1, seed0=dec.log["seed"])
    r = match(json.dumps(hdr), [job])[0]
    if not r["ok"]:
        return None
    lg = {k: v for k, v in dec.log.items() if k not in ("rolls", "reseed")}
    lg.update(seed=r["seed"], cmds=list(r["cmds"]), reseed=[list(x) for x in r["reseed"]])
    return Decision(**{**dec.__dict__, "log": lg, "at": len(r["cmds"])})


def WD_gender(entry: str) -> bool:
    f = entry.split("|")
    return len(f) > 7 and bool(f[7])


def p5_unexecuted(dec: Decision) -> Iterator[Decision]:
    """P5: an opponent command at an update whose segment shows no action of it, replaced by each other move it was
    offered there."""
    from agents.battle.core_replay import RecordedBattle, run_core
    from main.value_search.worlds import log_commands

    info, _sc = _scaffold(dec)
    cmds = log_commands(dec.log, dec.at)
    oi = 1 - info.viewer
    res = run_core([RecordedBattle(label="p5", format_id=dec.log["format_id"], seed=dec.log["seed"],
                                   p1={"name": dec.log["names"][0], "team": dec.log["teams"][0]},
                                   p2={"name": dec.log["names"][1], "team": dec.log["teams"][1]}, commands=cmds,
                                   reseed=dec.log.get("reseed"))], trackers=True, obs=True)[0]
    if not res["ok"]:
        return
    opp_dec = [e for e in res["trackers"][oi] if "tokens" in e]
    opp_idx = [i for i, (s_, _t) in enumerate(cmds) if s_ == info.opp]
    if len(opp_dec) < len(opp_idx):
        return
    for j, ci in enumerate(opp_idx):
        tok = cmds[ci][1]
        if not tok.startswith("move "):
            continue
        for alt in sorted(set(opp_dec[j]["tokens"].values())):
            if alt == tok or not alt.startswith("move "):
                continue
            lg = copy.deepcopy(dec.log)
            lg["cmds"][ci] = f"CHOOSE {info.opp} {alt}"
            yield Decision(**{**dec.__dict__, "log": lg})


def p5_untyped_hp(dec: Decision, max_hp_mass: float = 1.0) -> Iterator[Decision]:
    """P5 (the L6 case): an opponent MOVE command whose mon's true set holds a Hidden Power, replaced by poke-env's
    UNTYPED ``move hiddenpower`` (the sim plays it as that mon's Hidden Power, so a choice never executed ships the
    same I). Today's template keeps a token it cannot attribute verbatim, so every world without a Hidden Power on
    that mon is refused (``attribution_miss``); I is unchanged, so the honest builder never sees it."""
    from main.value_search import world_draws as WD
    from main.value_search.worlds import log_commands

    info, _sc = _scaffold(dec)
    oi = 1 - info.viewer
    P = WD.Prior()
    # only where today's worlds rarely carry one: a mon whose prior gives a Hidden Power little mass (an exclusion
    # flip needs most of today's 60 tries to fail)
    hp_mons = {WD.parse_set(e)[0] for e in _team_entries(dec.log, oi)
               if any(WD.is_hp(m) for m in WD.typed_moves(WD.parse_set(e)))
               and sum(v for mv, v in P.moves(WD.species_id(WD.parse_set(e))).items() if WD.is_hp(mv)) < max_hp_mass}
    if not hp_mons:
        return
    cmds = log_commands(dec.log, dec.at)
    for ci, (s_, tok) in enumerate(cmds):
        if s_ != info.opp or not tok.startswith("move ") or tok.startswith("move hiddenpower"):
            continue
        lg = copy.deepcopy(dec.log)
        lg["cmds"][ci] = f"CHOOSE {info.opp} move hiddenpower"
        yield Decision(**{**dec.__dict__, "log": lg})


# --------------------------------------------------------------------------------------------- the fixture
def decision_of(row: Mapping[str, Any]) -> Decision:
    return Decision(log=row["log"], at=int(row["at"]), side=row["side"], turn=int(row["turn"]), n=int(row["n"]),
                    mask=[int(x) for x in row["mask"]], tokens={str(k): str(v) for k, v in row["tokens"].items()},
                    key=row["key"])


def as_json(d: Decision) -> dict:
    return dict(d.__dict__)


def check_case(case: Mapping[str, Any], preset: str, forward=None) -> Dict[str, Any]:
    """One case under ``preset``: the two records (D, D′) and whether they are identical. ``r0s4_honest`` runs on
    its own key always; another preset runs on the PINNED key (I's digest) unless the case's channel is L5."""
    from main.anchors.search_presets import parse

    a, b = Decision(**case["d"]), Decision(**case["d_alt"])
    if not same_info(a, b):
        raise ValueError(f"case {case['name']}: D′ does not ship I")
    from main.anchors.search_side import HONEST_DICE

    honest = parse(preset)[0].dice in HONEST_DICE
    # each truth on its OWN searcher: the incremental builder's carry cache is per searcher, and D′ must never borrow
    # a carry D built
    sa, sb = searcher(preset, forward), searcher(preset, forward)
    try:
        ka, kb = _keys(case["channel"], honest, a, b)
        ra, rb = record(sa, a, ka), record(sb, b, kb)
    finally:
        sa.close()
        sb.close()
    return {"name": case["name"], "channel": case["channel"], "preset": preset, "identical": ra == rb,
            "d": ra, "d_alt": rb}


def _keys(channel: str, honest: bool, a: Decision, b: Decision):
    """The record keys the two truths' searches are handed (module docstring): their own for an honest preset,
    today's live key (the deal) for L5, else the key PINNED to I's digest."""
    if honest:
        return None, None
    if channel == "L5":
        from main.anchors.search_side import KEY_SCHEMA

        return (f"{KEY_SCHEMA}|{deal_key(a)}|t{a.turn}|n{a.n}", f"{KEY_SCHEMA}|{deal_key(b)}|t{b.turn}|n{b.n}")
    pin = f"pinned|{IS.extract(a.log, a.at, a.side)[0].key()}"
    return pin, pin


def sequence(dec: Decision, from_turn: int = 10, engine: Any = None) -> List[Decision]:
    """The DECISION SEQUENCE of ``dec``'s battle up to and including ``dec``: every decision of its side the live
    searcher searches (turn >= ``from_turn``, >= 2 legal actions), each cut from ``dec``'s own log at its own command,
    its turn, index, mask and tokens read off ONE replay. ``engine`` (a ``LeafEngine``): also check each cut's root
    against the playout tool's (its tokens, turn and decision index), as the live searcher's guards will."""
    from agents.battle.core_replay import run_core
    from agents.info_levers import revealed as RV

    rec, cmds = IS._recorded(dec.log, dec.at, dec.side)
    res = run_core([rec], trackers=True, obs=True)[0]
    if not res["ok"]:
        raise ValueError(f"{dec.key}: the log does not replay: {res.get('error')}")
    viewer = 0 if dec.side == "p1" else 1
    ent = [e for e in res["trackers"][viewer] if "obs" in e]
    ours = [i for i, (s_, _t) in enumerate(cmds) if s_ == dec.side]
    if len(ent) <= len(ours):
        raise ValueError(f"{dec.key}: {len(ent)} decisions of the viewer, the root is decision {len(ours)}")
    out: List[Decision] = []
    for j in range(len(ours) + 1):
        e = ent[j]
        turn = IS.turn_of(RV.viewer_lines(res, viewer, int(e["after"])))
        mask = [int(x) for x in e["mask"]]
        if turn < from_turn or sum(mask) < 2:
            continue
        at = ours[j] if j < len(ours) else dec.at
        d = Decision(log=dec.log, at=at, side=dec.side, turn=turn, n=j, mask=mask,
                     tokens={str(k): str(v) for k, v in e["tokens"].items()}, key=f"{dec.key}|seq{j}")
        if engine is not None:
            root = engine.root(dec.log, at, dec.side)
            got = {str(k): str(v) for k, v in (root.get("tokens") or {}).items()}
            if got != d.tokens or int(root["turn"]) != turn or int(root["n"]) != j:
                raise ValueError(f"{dec.key}: decision {j}'s cut is at turn {root['turn']} decision {root['n']} "
                                 f"offering {got}, the replay's turn {turn} decision {j} offering {d.tokens}")
        out.append(d)
    if not out or out[-1].at != dec.at or (out[-1].turn, out[-1].n) != (dec.turn, dec.n):
        raise ValueError(f"{dec.key}: the sequence does not end at the case's own decision (turn {dec.turn}, n {dec.n})")
    return out


def check_sequence(case: Mapping[str, Any], preset: str, forward=None, from_turn: int = 10) -> Dict[str, Any]:
    """SEQUENCE MODE (module docstring): the records of D's and D′'s whole decision sequences, each truth on its own
    searcher, compared decision by decision. ``identical`` iff every decision's record is."""
    from main.anchors.search_presets import parse
    from main.anchors.search_side import HONEST_DICE

    honest = parse(preset)[0].dice in HONEST_DICE
    a, b = Decision(**case["d"]), Decision(**case["d_alt"])
    sa, sb = searcher(preset, forward), searcher(preset, forward)
    try:
        qa, qb = sequence(a, from_turn, sa.engine), sequence(b, from_turn, sb.engine)
        if [(x.turn, x.n) for x in qa] != [(x.turn, x.n) for x in qb]:
            raise ValueError(f"case {case['name']}: D and D′ have other decision sequences")
        rows = []
        for x, y in zip(qa, qb):
            if not same_info(x, y):
                raise ValueError(f"case {case['name']}: D′ does not ship I at turn {x.turn} decision {x.n}")
            kx, ky = _keys(case["channel"], honest, x, y)
            rx, ry = record(sa, x, kx), record(sb, y, ky)
            rows.append({"turn": x.turn, "n": x.n, "identical": rx == ry, "d": rx, "d_alt": ry})
    finally:
        sa.close()
        sb.close()
    return {"name": case["name"], "channel": case["channel"], "preset": preset, "decisions": len(rows),
            "identical": all(r["identical"] for r in rows),
            "moved": [(r["turn"], r["n"]) for r in rows if not r["identical"]], "rows": rows}


def deal_key(d: Decision) -> str:
    """Today's live deal key (``search_side.deal_key``) of a truth: the seed (with its schedule) and both teams."""
    doc = {"format": d.log["format_id"], "seed": d.log["seed"], "reseed": d.log.get("reseed"), "side": d.side,
           "p1": [d.log["names"][0], d.log["teams"][0]], "p2": [d.log["names"][1], d.log["teams"][1]]}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:20]


def _order(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(rows, key=lambda r: hashlib.sha256(str(r["key"]).encode()).hexdigest())


def _last_onset(dec: Decision) -> bool:
    """Whether the update that ends at the root draws a hidden counter (a non-Rest sleep or a confusion onset)."""
    info, _ = IS.extract(dec.log, dec.at, dec.side)
    last = info.segments()[-1]
    return any((ln.startswith("|-status|") and ln.split("|")[3:4] == ["slp"] and "Rest" not in ln)
               or (ln.startswith("|-start|") and ln.split("|")[3:4] == ["confusion"]) for ln in last)


def build(selected: Path, scored: Optional[Path], out: Path, budget: int = 40, log=lambda x: print(x, flush=True),
          channels: Sequence[str] = CHANNELS, merge: Optional[Path] = None) -> dict:
    """Search the bank for one demonstrating pair per channel (module docstring), in a registered order (sha256 of
    the key): the first pair, valid (D′ ships I), whose ``r0s4_rolls`` records differ (L1: their world teams differ;
    L2: a counter drawn in the root's update, the world teams EQUAL and the records not; L6: exactly one of the two
    excluded). Writes the fixture."""
    rows = [json.loads(x) for x in Path(selected).read_text().splitlines() if x.strip()]
    excluded = set()
    if scored is not None:
        for x in Path(scored).read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if not (r.get("search") or {}).get("searched", True):
                    excluded.add(r["key"])
    rows = _order(rows)
    rng = np.random.default_rng(20261105)
    cases: List[dict] = []
    want = {
        "L1": lambda d: [p4_fresh(d, "p4")],
        "L2": lambda d: [p4_fresh(d, "p4_last", from_last=True)] if _last_onset(d) else [],
        "L3": lambda d: list(p2_unrevealed_move(d, rng))[:3],
        "L4": lambda d: list(p5_unexecuted(d))[:6],
        "L5": lambda d: list(p1_never_seen(d, rng))[:1],
        "L6": lambda d: [x for x in p5_untyped_hp(d, max_hp_mass=0.15) if same_info(d, x)][:2],
    }

    def good(ch: str, res: dict) -> bool:
        if res["identical"]:
            return False
        a, b = res["d"], res["d_alt"]
        if ch == "L1":
            return a["world_teams"] != b["world_teams"]
        if ch == "L2":
            return a["world_teams"] == b["world_teams"] and bool(a["world_teams"])
        if ch == "L6":
            return bool(a.get("searched")) != bool(b.get("searched"))
        return True

    if merge is not None and Path(merge).exists():
        cases = [c for c in load_fixture(merge)["cases"] if c["channel"] not in channels]
    for ch in channels:
        # L6 looks first among today's exclusions (a perturbation may let today's builder fit), then everywhere (or make
        # it exclude)
        pool = ([r for r in rows if r["key"] in excluded] + [r for r in rows if r["key"] not in excluded]
                if ch == "L6" else [r for r in rows if r["key"] not in excluded])
        tried = 0
        for r in pool:
            if tried >= (2 * budget if ch == "L6" else budget):
                break
            d = decision_of(r)
            try:
                alts = want[ch](d)
            except Exception as exc:                 # a state the perturbation cannot handle: the next state
                log(json.dumps({"channel": ch, "key": r["key"], "skip": f"{type(exc).__name__}: {str(exc)[:200]}"}))
                continue
            for alt in alts:
                if alt is None or not same_info(d, alt):
                    continue
                tried += 1
                case = {"name": f"{ch}:{r['key']}", "channel": ch, "d": as_json(d), "d_alt": as_json(alt)}
                try:
                    res = check_case(case, "r0s4_rolls@10")
                except Exception as exc:             # today's builder refusing an alternative truth: not a case
                    log(json.dumps({"channel": ch, "key": r["key"], "skip": f"{type(exc).__name__}: {str(exc)[:200]}"}))
                    continue
                log(json.dumps({"channel": ch, "key": r["key"], "identical": res["identical"],
                                "good": good(ch, res)}))
                if good(ch, res):
                    cases.append(case)
                    break
            if cases and cases[-1]["channel"] == ch:
                break
        else:
            log(json.dumps({"channel": ch, "found": False, "tried": tried}))
    # THE CONTROL: the same battle one decision earlier (another I)
    ctrl = None
    by_deal: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        by_deal.setdefault(str(r["deal"]), []).append(r)
    for r in rows:
        prev = [x for x in by_deal[str(r["deal"])] if int(x["at"]) < int(r["at"]) and x["side"] == r["side"]]
        if prev:
            p = max(prev, key=lambda x: int(x["at"]))
            ctrl = {"name": f"control:{r['key']}", "channel": "control", "d": as_json(decision_of(p)),
                    "d_alt": as_json(decision_of(r))}
            break
    fx = {"schema": SCHEMA, "cases": cases, "control": ctrl}
    with gzip.open(out, "wt") as f:
        json.dump(fx, f, sort_keys=True)
    return fx


def load_fixture(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.value_search.honest_inv")
    sub = ap.add_subparsers(dest="cmd", required=True)
    bb = sub.add_parser("build")
    bb.add_argument("--selected", required=True, type=Path)
    bb.add_argument("--scored", type=Path)
    bb.add_argument("--out", required=True, type=Path)
    bb.add_argument("--budget", type=int, default=40)
    bb.add_argument("--channels", default=",".join(CHANNELS))
    bb.add_argument("--merge", type=Path, help="keep this fixture's cases of the other channels")
    b = sub.add_parser("check")
    b.add_argument("--fixture", required=True, type=Path)
    b.add_argument("--preset", default="r0s4_honest@10")
    b.add_argument("--sequence", action="store_true", help="SEQUENCE mode: every searched decision up to the case's")
    b.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        fx = build(a.selected, a.scored, a.out, a.budget, channels=a.channels.split(","), merge=a.merge)
        print(json.dumps({"cases": [c["name"] for c in fx["cases"]], "control": bool(fx["control"])}))
        return 0
    fx = load_fixture(a.fixture)
    rows = [(check_sequence if a.sequence else check_case)(c, a.preset) for c in fx["cases"]]
    for r in rows:
        print(json.dumps({k: r[k] for k in ("name", "channel", "preset", "identical", "decisions", "moved") if k in r}),
              flush=True)
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
