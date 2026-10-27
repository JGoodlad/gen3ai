"""S-U's LIVE SEARCHED PLAYER: our checkpoint plus a registered one-ply search (:mod:`main.anchors.search_presets`,
R0 §4's ``r0s4``), deciding as the IN-PROCESS core slot of an anchor read (:mod:`main.anchors.core_side`) against a
local Metamon peer (``designs/endstate/design_search_u_headroom.md`` §2, §6 B1; ``EXTERNAL_ANCHORS_SOP.md``).

**What it does at each of our decisions.** The checkpoint's greedy choice is computed first, by the raw arm's own
:class:`~main.anchors.core_side.CorePolicy`. A decision is SEARCHED iff its turn (the core frame's) is >= the
registered first turn and it has >= 2 legal actions; every other decision plays that greedy choice, so before the
first searched decision the searched arm IS the raw arm, byte for byte (the paired prefix, G-PREFIX:
:mod:`main.reads.prefix`). A searched decision runs ``main.value_search.search_decision`` on the TRUE input log of
the battle, read in process from the front end (:func:`honest_log`), with the honest world builder
(``WorldSampler(reveal="off")``: the opponent's hidden team drawn from the Smogon prior, each world gated on the
searched side's byte-equal view), and plays its pick.

**The TRUE log, honestly cut** (:func:`honest_log`). The front end records every command it writes to the battle's
child with the REQUEST UPDATE it answers (``ws_frontend._Battle.choice_log``). The log handed to the search holds
every command of the updates before ours, ordered (update, p1 before p2), and OUR re-choices of our own update (after
an ``[Unavailable choice]``) — never the opponent's choice of our update, which the sim holds until both sides are in.
A peer's switch is re-spelled the way the core names it (:func:`core_command`: poke-env sends the species id).
So the opponent's root decision is open in every world and the forward answers it (σ̂ = π), and
``root_choice_in_log=False`` keeps the sampler from treating a past opponent command as the root's. Two guards
before any world is drawn (each a NAMED failure, never a skip): the log's replay must open OUR decision with the live
frame's tokens, turn and decision index, and its replayed observation row must be the live core frame's, byte for
byte (:meth:`LiveSearcher._check_root`).

**The leaks the honest builder keeps** (``search_ledger.md`` §6.1): every world replays the prefix on the TRUE battle
seed (speed ties, hidden counters). A searched arm's gain is therefore LEAK-INCLUSIVE.

**Deterministic per deal.** Each searched decision is keyed by the DEAL (a digest of the battle's ``START``: both
packed teams, the names, the seed — so the raw and searched games of one deal share it) and the decision (turn, our
decision index): the worlds, their screens and the playouts' dice are functions of the key, the forward is the
checkpoint on CPU, so a searched game replays exactly.

**Off the event loop.** A searched decision costs ≈ 4-55 CPU-s (S-U stage 0 on the lean world screen). It runs in a
worker thread (``asyncio.to_thread``) so this process's websocket server keeps answering the peer's keepalive pings
(poke-env's client pings every 20 s and drops a connection whose pong is 20 s late: a search ON the loop would
disconnect the peer, and the front end would book the dropped peer's forfeit as OUR win). Our own watchdog
(``--first-game-timeout`` / ``--progress-timeout``) must be sized for a searched game (``main.x35.kakuna`` sets it);
the front end has no battle timer.

**The record.** One JSON line per decision of ours (schema :data:`SCHEMA`) in ``<out>/search_decisions.jsonl``:
whether it was searched (and why not), π's greedy action, the pick, Q̂ and log π per legal action, the critic at the
root, the playouts spent, its CPU and wall seconds, the worlds' tries and leak block and their species hit rate
against the truth (S-U's REPORTED world-quality covariate, at species level). ``main.reads search-cost`` reads them.
``cpu_s`` (this worker thread plus the replay children) is the figure every banked record carries; ``world_cpu_s`` /
``playout_cpu_s`` SPLIT a decision (``search.cpu_all``: every thread plus the children, so they need not sum to
``cpu_s``): building the worlds (an exclusion is all world building) and playing the leaves.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from main.anchors.search_presets import SearchPreset, canonical, parse

SCHEMA = "gen3_live_search_decision_v1"
KEY_SCHEMA = "su_live_v1"


class LiveSearchError(RuntimeError):
    """A searched decision cannot be made honestly (a log that does not replay to the live decision, a prior that is
    not the acting policy's, a battle with no seed): a DEFECT, named, never a silent fallback to the raw choice."""


# --------------------------------------------------------------------------------------------- the TRUE log
def _seed_text(seed: Any) -> str:
    if isinstance(seed, (list, tuple)):
        return ",".join(str(int(x)) for x in seed)
    if isinstance(seed, str) and seed.strip():
        return seed.strip()
    raise LiveSearchError(f"the battle carries no seed ({seed!r}): a searched read needs --seed-base (the honest "
                          "builder replays the battle's own seed)")


def deal_key(start_doc: Dict[str, Any], side: str) -> str:
    """The DEAL's identity: a digest of what makes the battle (format, seed, both names and packed teams) and our side,
    the same for the raw and the searched game of one deal."""
    doc = {"format": start_doc.get("formatid"), "seed": start_doc.get("seed"), "side": side,
           "p1": [start_doc["p1"].get("name"), start_doc["p1"].get("team")],
           "p2": [start_doc["p2"].get("name"), start_doc["p2"].get("team")]}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:20]


def _toid(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def core_command(cmd: str, teams: Dict[str, str]) -> str:
    """``cmd`` with a SWITCH named the way the core names it. The replay feeds a logged choice by its exact core token
    (``rust_env`` ``logged_token_index``), and the core names a switch by the mon's NAME (``switch Charizard``, the
    packed team's name field). A peer names it its own way (poke-env: ``switch charizard``, the species id); the sim
    takes either, so the choice is the same: it is re-spelled by matching the side's packed team on Showdown's
    ``toID`` of the name, then of the species. Moves are already ids on both sides. A switch that names no mon of the
    team, or two, is a :class:`LiveSearchError`."""
    parts = cmd.split(" ", 3)
    if len(parts) != 4 or parts[0] != "CHOOSE" or parts[2] != "switch" or parts[3].strip().isdigit():
        return cmd
    side, want = parts[1], parts[3].strip()
    mons = [e.split("|") for e in teams[side].split("]") if e]
    names = [f[0] for f in mons]
    if want in names:
        return cmd
    for key in (lambda f: f[0], lambda f: f[1] or f[0]):
        hits = [f[0] for f in mons if _toid(key(f)) == _toid(want)]
        if len(hits) == 1:
            return f"CHOOSE {side} switch {hits[0]}"
        if len(hits) > 1:
            break
    raise LiveSearchError(f"{cmd!r}: the switch names no single mon of {side}'s team {names}")


def honest_log(battle: Any, side: str) -> Tuple[Dict[str, Any], int, int]:
    """``(log, at, update)``: the battle's core INPUT LOG for ``side``'s decision at its CURRENT request (module
    docstring), ``at`` = its length (the decision is the next command), ``update`` = the request update it answers."""
    start = getattr(battle, "start_doc", None)
    if not start:
        raise LiveSearchError(f"{getattr(battle, 'tag', '?')}: the front end recorded no START document")
    if side not in ("p1", "p2"):
        raise LiveSearchError(f"side {side!r}")
    u = int(battle.req_n[side])
    rows = [(i, upd, s, c) for i, (upd, s, c) in enumerate(battle.choice_log)
            if upd < u or (upd == u and s == side)]
    rows.sort(key=lambda r: (r[1], 0 if r[2] == "p1" else 1, r[0]))
    teams = {"p1": str(start["p1"]["team"]), "p2": str(start["p2"]["team"])}
    cmds = [core_command(c, teams) for _i, _u, _s, c in rows]
    if any(c.startswith("FORCELOSE") for c in cmds):
        raise LiveSearchError(f"{battle.tag}: a forfeit is in the log before a decision")
    log = {"format_id": str(start["formatid"]), "seed": _seed_text(start.get("seed")),
           "names": [str(start["p1"]["name"]), str(start["p2"]["name"])],
           "teams": [str(start["p1"]["team"]), str(start["p2"]["team"])], "cmds": cmds}
    return log, len(cmds), u


def species_hit(world_team: str, true_team: str) -> float:
    """The share of the TRUE team's species a world's team holds (the packed species field, or the name)."""
    from main.value_search.world_draws import parse_set, species_id

    def sp(team: str) -> List[str]:
        return [species_id(parse_set(e)) for e in team.split("]") if e]

    truth = sp(true_team)
    got = set(sp(world_team))
    return sum(1 for s in truth if s in got) / max(1, len(truth))


def _children_cpu() -> float:
    import resource

    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


# ---------------------------------------------------------------------------------------------- the searcher
class LiveSearcher:
    """One searched player for one half: ``policy`` (the raw arm's :class:`CorePolicy`, greedy) and the checkpoint
    ``model`` it wraps; ``search`` a registered spelling (``r0s4@10``); ``out`` the decision log. ``engine_lib`` /
    ``sampler`` / ``forward`` / ``preset`` are injection points for tests."""

    def __init__(self, model: Any, policy: Any, search: str, out: Path, *, engine_lib: Any = None,
                 sampler: Any = None, forward: Any = None, preset: Optional[SearchPreset] = None) -> None:
        from main.value_search.playout import LeafEngine
        from main.value_search.search import SearchConfig
        from main.value_search.worlds import WorldSampler

        if getattr(policy, "stochastic", False):
            raise LiveSearchError("a searched player decides greedily off the search; our side must be greedy")
        p, self.from_turn = parse(search)
        self.preset = preset or p
        self.search = canonical(search)
        self.policy = policy
        if forward is None:
            from main.policy_spectrum.qhat import check_winprob, logits_and_values

            check_winprob(model)            # V must be P(win): the leaf and the piKL score live on the win scale
            forward = lambda r, m: logits_and_values(model, r, m, batch=1024)   # noqa: E731
        self.forward = forward
        self.engine = LeafEngine(forward, n_cores=self.preset.n_worlds, lib=engine_lib)
        self.sampler = sampler if sampler is not None else WorldSampler(
            reveal=self.preset.reveal, max_tries=self.preset.max_tries, dice=self.preset.dice)
        self.cfg = SearchConfig(per_action=self.preset.per_action, n_worlds=self.preset.n_worlds,
                                depth=self.preset.depth, allocation="uniform", rule=self.preset.rule,
                                pikl_lambda=self.preset.pikl_lambda, weights="uniform")
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.stats: Dict[str, Any] = {"search": self.search, "preset": self.preset.as_dict(),
                                      "from_turn": self.from_turn, "decisions": 0, "from_turn_decisions": 0,
                                      "searched": 0, "excluded": 0, "deviations": 0, "search_cpu_s": 0.0,
                                      "search_wall_s": 0.0}

    def close(self) -> None:
        self.engine.close()

    # -- the decision ---------------------------------------------------------------------------------------
    def key(self, deal: str, turn: int, n: int) -> str:
        return f"{KEY_SCHEMA}|{deal}|t{turn}|n{n}"

    async def decide(self, front: Any, tag: str, d: Dict[str, Any], row: np.ndarray, mask: np.ndarray,
                     conn: Any) -> Tuple[int, int]:
        """``(chosen, π's greedy action)`` for one decision of ours (connection ``conn``, the core slot) in battle
        ``tag`` (``d`` its core frame)."""
        idx_pi, argmax = self.policy.decide(row, mask)
        turn, n = int(d.get("turn") or 0), int(d["n"])
        legal = [int(i) for i in np.flatnonzero(np.asarray(mask))]
        self.stats["decisions"] += 1
        battle = front.battles.get(tag)
        if battle is None:
            raise LiveSearchError(f"{tag}: the front end holds no such battle")
        side = next((s for s, c in battle.conns.items() if c is conn), None)
        if side is None:
            raise LiveSearchError(f"{tag}: our connection is not a side of the battle")
        rec: Dict[str, Any] = {"schema": SCHEMA, "battle": tag, "index": getattr(battle, "index", None), "side": side,
                               "turn": turn, "n": n, "n_legal": len(legal), "pi_argmax": int(argmax),
                               "search": self.search}
        if turn >= self.from_turn:
            self.stats["from_turn_decisions"] += 1
        if turn < self.from_turn or len(legal) < 2:
            rec.update(searched=False, why="before_turn" if turn < self.from_turn else "one_legal", played=int(argmax))
            self._write(rec)
            return int(argmax), int(argmax)
        log, at, update = honest_log(battle, side)
        deal = deal_key(battle.start_doc, side)
        key = self.key(deal, turn, n)
        rec.update(deal=deal, update=update, at=at, key=key)
        out = await asyncio.to_thread(self._search, log, at, side, row, mask, d, key, int(argmax))
        rec.update(out)
        self._write(rec)
        return int(rec["played"]), int(argmax)

    def _check_root(self, log: Dict[str, Any], at: int, side: str, d: Dict[str, Any], key: str) -> None:
        from main.value_search import worlds as W

        root = self.engine.root(log, at, side)
        live_tokens = {int(k): str(v) for k, v in (d.get("tokens") or {}).items()}
        got_tokens = {int(k): str(v) for k, v in (root.get("tokens") or {}).items()}
        if got_tokens != live_tokens:
            raise LiveSearchError(f"{key}: the log's replay offers {got_tokens}, the live decision {live_tokens}")
        if int(root.get("turn", -1)) != int(d.get("turn") or 0) or int(root.get("n", -1)) != int(d["n"]):
            raise LiveSearchError(f"{key}: the log's replay is at turn {root.get('turn')} decision {root.get('n')}, "
                                  f"the live decision at turn {d.get('turn')} decision {d['n']}")
        truth = W.prepare_root(log, at, side, key, root_choice_in_log=False)
        if not truth.ref or truth.ref[-1] != d["frame"]["b64"]:
            raise LiveSearchError(f"{key}: the log's replayed observation row is not the live core frame's")

    def _search(self, log: Dict[str, Any], at: int, side: str, row: np.ndarray, mask: np.ndarray,
                d: Dict[str, Any], key: str, argmax: int) -> Dict[str, Any]:
        from main.value_search import rules
        from main.value_search.search import cpu_all, search_decision
        from main.value_search.worlds import WorldError

        c0, k0, w0 = time.thread_time(), _children_cpu(), time.monotonic()
        self._check_root(log, at, side, d, key)
        out = np.asarray(self.forward(np.asarray(row, dtype=np.float32)[None],
                                      np.asarray(mask, dtype=np.uint8)[None]))[0]
        m = np.asarray(mask).astype(bool)
        lg = np.where(m, out[:11].astype(np.float64), -np.inf)
        legal = [int(a) for a in np.flatnonzero(m)]
        order = rules.policy_order(lg, legal)
        if order[0] != argmax and lg[order[0]] - lg[argmax] > 1e-5:
            raise LiveSearchError(f"{key}: the search's forward puts action {order[0]} first (logit {lg[order[0]]:.6f}), "
                                  f"the acting policy's greedy choice is {argmax} ({lg[argmax]:.6f}): piKL would anchor "
                                  "to another policy")
        lsm = rules.log_softmax_ordered(lg, order)
        a0 = cpu_all()
        try:
            res = search_decision(self.engine, log, at, side, np.asarray(row, dtype=np.float32),
                                  np.asarray(mask, dtype=np.uint8), self.cfg, key, sampler=self.sampler,
                                  root_choice_in_log=False)
        except WorldError as exc:
            # the honest builder found no consistent world in its tries: an EXCLUSION (the banks' own rule), played
            # as the raw arm's choice and recorded; any other exception is a defect and fails the game
            self.stats["excluded"] += 1
            return {"searched": False, "why": "excluded", "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                    "played": argmax, "cpu_s": round(time.thread_time() - c0 + _children_cpu() - k0, 3),
                    "world_cpu_s": round(cpu_all() - a0, 3), "playout_cpu_s": 0.0,
                    "wall_s": round(time.monotonic() - w0, 3)}
        if sorted(res.q) != legal:
            raise LiveSearchError(f"{key}: the search valued {sorted(res.q)}, the legal actions are {legal}")
        true_team = log["teams"][1 if side == "p1" else 0]
        hits = [round(species_hit(t, true_team), 4) for t in res.world_teams]
        # CPU = this worker thread (the forward, the playout and world FFI) + the replay children it reaped (each
        # world gate's ``core_events``); the validator's persistent Node child is not counted (memoised, milliseconds)
        cpu, wall = time.thread_time() - c0 + _children_cpu() - k0, time.monotonic() - w0
        self.stats["searched"] += 1
        self.stats["deviations"] += int(res.action != argmax)
        self.stats["search_cpu_s"] += cpu
        self.stats["search_wall_s"] += wall
        return {"searched": True, "played": int(res.action), "deviation": bool(res.action != argmax),
                "q": {str(a): round(float(res.q[a]), 6) for a in legal},
                "log_pi": {str(a): round(float(lsm[a]), 6) for a in legal},
                "root_v": round(float(res.root_value), 6), "spent": int(res.spent),
                "cpu_s": round(cpu, 3), "wall_s": round(wall, 3), "process_cpu_s": round(float(res.cpu_s), 3),
                "world_cpu_s": round(float(res.world_cpu_s), 3), "playout_cpu_s": round(float(res.playout_cpu_s), 3),
                "world_tries": list(res.world_tries), "leak": res.leak, "species_hit": hits,
                "config": self.preset.as_dict()}

    def _write(self, rec: Dict[str, Any]) -> None:
        with open(self.out, "a") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            f.flush()

    def summary(self) -> Dict[str, Any]:
        s = dict(self.stats)
        s["search_cpu_s"] = round(s["search_cpu_s"], 3)
        s["search_wall_s"] = round(s["search_wall_s"], 3)
        s["decisions_log"] = str(self.out)
        return s
