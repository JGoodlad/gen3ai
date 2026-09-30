"""THE DEPTH-3 SUCCESSOR SLICE (M5 Lane I's gate, ``designs/endstate/program_rust_core.md`` §2 M5):
the in-process search tree (:class:`utils.rust_env.successors.Successors`) against the
``search_driver`` BINARY's JSON road (:class:`utils.bridge.search_session.SearchSession`,
``impl="rust"``), driven in LOCKSTEP with the identical request sequence.

Per battle, per sampled turn, per searched side: ``open_root`` on both (every root field equal —
node id, requests, recorded choices, ``pre_state``, both prefixes), then THREE plies: every legal
action of ours (the core's tokens below the root) x up to ``m_opp`` opponent choices (read off the
request, :func:`main.search_dividend.alpha.legal_choices_from_request`) x the ply's dice seeds
(freshly minted, SHARED across the ply — search's CRN shape — plus the battle's own ``"original"``
stream on one arm per root). Every arm is compared field by field: label, node id, ``ended``,
``stuck``, ``outcome``, ``requests``, ``choices_used``, both ``pN_chunks`` (or the elided
sentinel), and ``core_pN``'s ``mid``, mask, tokens and ROW BYTES (the wire frame decoded vs the
in-process buffer). A ply's frontier is up to ``beam`` non-terminal, branchable children (a seeded
sample, the same on both sides). No allowlist.

The records are core INPUT LOGS played by :func:`successors.play_out` (the battle's own dice, a
seeded uniform policy) and converted with :func:`successors.log_to_record` — the oracle is the
binary, so the corpus's producer does not matter beyond being a real battle.

    export PYTHONPATH=$PYTHONPATH:src
    python -m utils.rust_env.successors_parity --battles 12 --source ladder_milestone
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from main.search_dividend.alpha import legal_choices_from_request
from utils.rust_env import successors as S


@dataclass
class Census:
    battles: int = 0
    roots: int = 0
    arms: int = 0
    rows: int = 0
    mid: int = 0
    ended: int = 0
    stuck: int = 0
    by_depth: Counter = field(default_factory=Counter)
    diffs: List[str] = field(default_factory=list)
    #: Batches BOTH roads refused with the same error (reported, not a difference).
    refused: List[str] = field(default_factory=list)

    def diff(self, where: str, what: str, a, b) -> None:
        if len(self.diffs) < 50:
            self.diffs.append(f"{where}: {what}: json={a!r:.300} inproc={b!r:.300}")
        else:
            self.diffs.append("…")

    def render(self) -> str:
        head = "✅ byte-equal" if not self.diffs else f"❌ {len(self.diffs)} differences"
        return (f"{head}: {self.battles} battles, {self.roots} roots, {self.arms} arms "
                f"(depth {dict(sorted(self.by_depth.items()))}), {self.rows} rows, {self.mid} D10 leaves, "
                f"{self.ended} terminal, {self.stuck} stuck, {len(self.refused)} batches refused alike")


def make_logs(teams: Sequence[str], n: int, seed: int, *, core: Optional[S.SearchCore] = None) -> List[dict]:
    """``n`` finished battles as core input logs: two teams drawn from ``teams``, a seeded Showdown
    seed, a seeded uniform policy, the battle's OWN dice (so the log replays)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a, b = rng.choice(len(teams), size=2, replace=False)
        log = {"format_id": "gen3ou", "seed": ",".join(str(int(x)) for x in rng.integers(0, 65536, 4)),
               "names": [f"lanei{i}a", f"lanei{i}b"], "teams": [teams[a], teams[b]], "cmds": []}
        r = S.play_out(log, 0, "p1", policy=S.uniform_random(rng), seeds=[None], stall=None, max_turns=400,
                       keep_cmds=True, actions=None, core=core)
        br = r.branches[int(rng.integers(len(r.branches)))]
        if br["end"]["truncated"]:
            continue
        out.append(dict(log, cmds=br["cmds"]))
    return out


def _last_turn(record) -> int:
    """The last turn a record reaches (its ``|turn|`` count is not in the record; count p1's
    move-kind commands as an upper bound and let ``open_root`` refuse past the end)."""
    return max(1, sum(1 for s, p in record.commands if s == "p1"))


def _row(c: Optional[dict], inproc: bool):
    if c is None or c.get("row") is None:
        return None
    if inproc or isinstance(c["row"], np.ndarray):
        return np.asarray(c["row"]).tobytes()
    from agents.battle.core_obs import wrap_row

    return wrap_row(c["row"]).tobytes()


def _cmp_arm(cen: Census, where: str, a, b) -> None:
    for f in ("label", "node_id", "ended", "stuck", "outcome", "requests", "choices_used"):
        if getattr(a, f) != getattr(b, f):
            cen.diff(where, f, getattr(a, f), getattr(b, f))
    for f in ("p1_chunks", "p2_chunks"):
        x, y = getattr(a, f), getattr(b, f)
        if type(x).__name__ == "ElidedSide" or type(y).__name__ == "ElidedSide":
            if type(x) is not type(y):
                cen.diff(where, f + " (elision)", type(x).__name__, type(y).__name__)
        elif list(x) != list(y):
            cen.diff(where, f, x, y)
    for f in ("core_p1", "core_p2"):
        x, y = getattr(a, f), getattr(b, f)
        if (x is None) != (y is None):
            cen.diff(where, f, x, y)
            continue
        if x is None:
            continue
        for k in ("mid", "mask", "tokens"):
            if x.get(k) != y.get(k):
                cen.diff(where, f"{f}.{k}", x.get(k), y.get(k))
        rx, ry = _row(x, False), _row(y, True)
        if rx != ry:
            cen.diff(where, f"{f}.row", None if rx is None else f"{len(rx)} B", None if ry is None else "differs")
        elif rx is not None:
            cen.rows += 1
            cen.mid += bool(x.get("mid"))


def compare_battle(js, ip: S.Successors, record, cen: Census, rng: np.random.Generator, *,
                   n_turns: int = 3, m_opp: int = 3, n_seeds: int = 2, beam: int = 3, depth: int = 3,
                   both_sides_first: bool = True) -> None:
    cen.battles += 1
    last = _last_turn(record)
    turns = sorted(set(int(t) for t in rng.integers(1, max(2, last), n_turns)) | {1})
    for ti, turn in enumerate(turns):
        for side in ("p1", "p2"):
            other = "p2" if side == "p1" else "p1"
            want: Optional[str] = None if (both_sides_first and ti == 0 and side == "p1") else side
            where0 = f"{record.battle_tag} T{turn} {side}{'/both' if want is None else ''}"
            try:
                ra = js.open_root(turn, record=record, core="text", side=want, trackers=True)
                ea = None
            except Exception as e:  # noqa: BLE001 — both roads must refuse alike
                ra, ea = None, str(e)
            try:
                rb = ip.open_root(turn, record=record, core="text", side=want, trackers=True)
                eb = None
            except Exception as e:  # noqa: BLE001
                rb, eb = None, str(e)
            if (ea is None) != (eb is None):
                cen.diff(where0, "open_root refusal", ea, eb)
                continue
            if ea is not None:
                continue  # a turn past the end: both refused
            cen.roots += 1
            for f in ("node_id", "requests", "recorded_choices", "pre_state", "prefix_p1_chunks", "prefix_p2_chunks"):
                if getattr(ra, f) != getattr(rb, f):
                    cen.diff(where0, f"root.{f}", getattr(ra, f), getattr(rb, f))
            ours0 = [c["token"] for c in legal_choices_from_request((ra.requests or {}).get(side))]
            frontier = [(ra.node_id, ours0, ra.requests)]
            for d in range(1, depth + 1):
                seeds = [f"sodium,{int(x):032x}" for x in rng.integers(0, 2**63, n_seeds)]
                arms, label = [], 0
                for node_id, ours, reqs in frontier:
                    opp = [c["token"] for c in legal_choices_from_request((reqs or {}).get(other))]
                    if len(opp) > m_opp:
                        opp = [opp[i] for i in sorted(rng.choice(len(opp), m_opp, replace=False))]
                    for a in ours:
                        for o in opp or ["random"]:
                            for sd in seeds:
                                arms.append({"node_id": node_id, f"{side}_action": a, f"{other}_action": o,
                                             "seed": sd, "label": label})
                                label += 1
                if d == 1 and arms:
                    arms[0] = dict(arms[0], seed="original")
                if not arms:
                    break
                xa = xb = ea = eb = None
                try:
                    xa = js.expand_many(arms, side=want, rows=True)
                except Exception as e:  # noqa: BLE001 — both roads must refuse alike
                    ea = str(e).split("]: ", 1)[-1]
                try:
                    xb = ip.expand_many(arms, side=want, rows=True)
                except Exception as e:  # noqa: BLE001
                    eb = str(e)
                if ea is not None or eb is not None:
                    if ea is None or eb is None or ea not in eb:
                        cen.diff(where0, f"depth {d} expand refusal", ea, eb)
                    else:
                        cen.refused.append(f"{where0} d{d}: {eb[:240]}")
                    break
                if len(xa) != len(xb):
                    cen.diff(where0, f"depth {d} arm count", len(xa), len(xb))
                    break
                nxt = []
                for a, b in zip(xa, xb):
                    cen.arms += 1
                    cen.by_depth[d] += 1
                    cen.ended += a.ended
                    cen.stuck += a.stuck
                    _cmp_arm(cen, f"{where0} d{d} arm {a.label}", a, b)
                    leaf = a.core_p1 if side == "p1" else a.core_p2
                    if (not a.ended and leaf and leaf.get("row") is not None
                            and legal_choices_from_request((a.requests or {}).get(side))):
                        nxt.append((a.node_id, list(leaf["tokens"].values()), a.requests))
                if len(nxt) > beam:
                    nxt = [nxt[i] for i in sorted(rng.choice(len(nxt), beam, replace=False))]
                frontier = nxt
                if not frontier:
                    break


def run(logs: Sequence[dict], *, seed: int = 0, lib=None, **kw) -> Census:
    from utils.bridge.search_session import SearchSession

    cen = Census()
    rng = np.random.default_rng(seed)
    with SearchSession(impl="rust", timeout=600) as js, S.Successors(lib=lib) as ip:
        for i, log in enumerate(logs):
            rec = S.log_to_record(log, battle_tag=f"b{i}")
            compare_battle(js, ip, rec, cen, rng, **kw)
    return cen


def _teams(source: str) -> List[str]:
    if source.startswith("ladder_"):
        from utils.ladder_corpus import teams

        return list(teams(source.split("_", 1)[1]))
    from utils.team_sources import team_list

    if source == "pool":
        from utils.teambuilder import Gen3Teambuilder

        return list(Gen3Teambuilder(team_list("pool")).packed_teams)
    return list(team_list(source, n=200))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--battles", type=int, default=12)
    ap.add_argument("--source", default="ladder_milestone", help="ladder_commit | ladder_milestone | pool | procedural")
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    ap.add_argument("--turns", type=int, default=3)
    a = ap.parse_args(argv)
    t0 = time.time()
    logs = make_logs(_teams(a.source), a.battles, a.seed)
    cen = run(logs, seed=a.seed, n_turns=a.turns)
    print(f"[successors_parity] source={a.source} seed={a.seed} {time.time() - t0:.1f}s")
    print(cen.render())
    for d in cen.diffs[:20]:
        print("   ", d)
    for d in cen.refused[:10]:
        print("    [refused alike]", d)
    return 0 if not cen.diffs else 1


if __name__ == "__main__":
    sys.exit(main())
