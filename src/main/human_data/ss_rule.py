r"""S-S's team draw under the human rowset's panel-closeness rule (X59-S REVIEW E B1; PREREG Amendment 2).

    python -m main.human_data.ss_rule profile --snapshot hr_spec_v1 --out <dir>/reveal_profile.json.gz [--n 40000]
    python -m main.human_data.ss_rule census  --profile <dir>/reveal_profile.json.gz --out <json>

The human rowset drops a battle when either side's REVEALED facts fit >= :data:`~.textpass.NEAR` mons of one
held-out panel team (``textpass.PanelMatcher``, every held-out panel, ``forum216b`` included). Self-play shows the
core every set whole, so the rule cannot be read off a self-play game; it is SIMULATED instead, from how much human
sides actually reveal:

* :func:`reveal_profile` — the REVEAL SHAPE of every side of a seeded sample of the snapshot's CANDIDATE battles (the
  battles the human rule ran on: TRAIN dates, rated, a result, >= ``rowset.MIN_TURNS`` turns, a >= 1300 seat), one
  shape per side: per seen mon, (moves revealed, item revealed, ability revealed). CANDIDATE sides, not kept ones:
  the rule acts on battles before it selects them, and the kept sides are selected for revealing little (5.14 mons
  seen against 5.66 excluded), so their shapes would under-exclude.
* :class:`PairRule` — a team's simulated reveal draws ONE whole human side shape, puts it on a uniform subset of the
  team's mons, reveals each mon's that-many moves uniformly from its set (a Hidden Power as bare ``hiddenpower``, as
  a gen-3 log names it), its item and ability where the shape says; the side is NEAR when the matcher says so.
  :meth:`PairRule.draw_pair` draws a team pair for one self-play game and REDRAWS while either side is near, before
  the game is played (no game is wasted). Each kept game's pair is recorded by ``teacher.play_part``.

Seeded and pure: the same profile, teams, panels and generator states give the same pairs. CPU only.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.human_data import textpass as T

Shape = Tuple[Tuple[int, int, int], ...]      # per seen mon: (moves revealed, item revealed, ability revealed)
SEED_PROFILE = 20261026
RATING_MIN = 1300
#: a team pair is redrawn at most this many times (a pool where every pair is near is a defect, not a loop)
MAX_REDRAWS = 10_000


class RuleError(RuntimeError):
    """A reveal profile or a team table that does not match what it claims (a defect: stop)."""


def side_shape(mons: Dict[str, Tuple[Sequence[str], Optional[str], Optional[str]]]) -> Shape:
    """One side's reveal shape (module docstring), canonical (sorted, most-revealed first); Struggle is no reveal."""
    out = [(len([m for m in moves if m != "struggle"]), int(item is not None), int(ability is not None))
           for moves, item, ability in mons.values()]
    return tuple(sorted(out, reverse=True))


def is_candidate(rec: Optional[Dict[str, Any]], rel: str) -> bool:
    """The battles the human panel rule ran on (``rowset.scan``'s filters before the exclusion; the bot and duplicate
    rules are omitted here: they move no reveal shape)."""
    from main.human_data import rowset as RS

    if rec is None or rel[:10] > RS.TRAIN_LAST or not rec["rated"]:
        return False
    if rec["end"] == "none" or (rec["win"] is None and rec["end"] != "tie"):
        return False
    if rec["turns"] < RS.MIN_TURNS:
        return False
    return any((rec["rating"].get(s) or 0) >= RATING_MIN for s in ("p1", "p2"))


def _profile_job(args) -> Optional[List[Shape]]:
    from main.human_data import rowset as RS

    r = RS._scan_job(args)
    if not is_candidate(r["rec"], r["rel"]):
        return None
    return [side_shape(r["rec"]["reveals"][s]) for s in ("p1", "p2") if s in r["rec"]["reveals"]]


def reveal_profile(snap: str, n: int, workers: int, store: Path, registry: Path, corpus: Path,
                   seed: int = SEED_PROFILE) -> Dict[str, Any]:
    """The candidate sides' reveal shapes of a seeded sample of ``n`` logs of the snapshot (module docstring)."""
    from main.human_data import snapshots as S

    row = S.registry_row(snap, registry)
    items = S.read_manifest(snap, store, registry)
    pick = sorted(np.random.default_rng(seed).choice(len(items), size=min(n, len(items)), replace=False).tolist())
    shapes: collections.Counter = collections.Counter()
    cand = 0
    with Pool(workers) as pool:
        for got in pool.imap(_profile_job, [(snap, row["population"], str(corpus), list(items[i])) for i in pick],
                             chunksize=128):
            if got is None:
                continue
            cand += 1
            shapes.update(got)
    sides = sum(shapes.values())
    return {"schema": "gen3_x59s_reveal_profile_v1", "snapshot": snap, "manifest_sha256": row["manifest_sha256"],
            "seed": seed, "logs_sampled": len(pick), "candidate_battles": cand, "sides": sides,
            "rule": "candidate battles (TRAIN dates, rated, a result, >= 11 turns, a >= 1300 seat); one shape per "
                    "side: per seen mon (moves revealed excl. Struggle, item revealed, ability revealed)",
            "mons_seen_per_side": round(sum(len(s) * c for s, c in shapes.items()) / max(sides, 1), 4),
            "shapes": sorted([[list(map(list, s)), c] for s, c in shapes.items()], key=lambda x: (-x[1], x[0]))}


def load_profile(path: Path) -> Tuple[List[Shape], np.ndarray, str]:
    """(shapes, their probabilities, the file's sha256)."""
    from main.human_data import rowset as RS

    with gzip.open(path, "rt") as fh:
        prof = json.load(fh)
    if prof.get("schema") != "gen3_x59s_reveal_profile_v1" or not prof["shapes"]:
        raise RuleError(f"{path}: not a reveal profile")
    shapes = [tuple(tuple(int(x) for x in m) for m in s) for s, _ in prof["shapes"]]
    w = np.asarray([c for _, c in prof["shapes"]], np.float64)
    return shapes, w / w.sum(), RS.S_sha(path)


def _reveal_move(mid: str) -> str:
    return "hiddenpower" if mid.startswith("hiddenpower") else mid


class PairRule:
    """The human rule on a team table (module docstring). ``teams`` are :class:`main.anchors.team_overlap.Team`
    objects in the core's team-table order; ``matcher`` defaults to every held-out panel (``forum216b`` included)."""

    def __init__(self, shapes: Sequence[Shape], probs: np.ndarray, teams: Sequence[Any], matcher: Any = None):
        if len(shapes) != len(probs) or not len(shapes):
            raise RuleError("a reveal profile needs one probability per shape")
        self.shapes = list(shapes)
        self.cdf = np.cumsum(np.asarray(probs, np.float64))
        self.cdf[-1] = 1.0
        self.teams = list(teams)
        self.matcher = matcher if matcher is not None else T.PanelMatcher()
        self.draws = 0
        self.redraws = 0

    def simulated_side(self, team: int, rng: np.random.Generator) -> Dict[str, Tuple[List[str], Optional[str],
                                                                                        Optional[str]]]:
        shape = self.shapes[int(np.searchsorted(self.cdf, rng.random(), side="right"))]
        mons = self.teams[team].mons
        k = min(len(shape), len(mons))
        who = rng.permutation(len(mons))[:k]
        order = rng.permutation(len(shape))[:k]
        side = {}
        for mi, si in zip(who.tolist(), order.tolist()):
            ms = mons[mi]
            n_mv, it, ab = shape[si]
            mv = list(ms.moves)
            pick = rng.permutation(len(mv))[:min(n_mv, len(mv))].tolist()
            side[ms.species] = (sorted(_reveal_move(mv[j]) for j in pick), (ms.item or None) if it else None,
                                (ms.ability or None) if ab else None)
        return side

    def near(self, team: int, rng: np.random.Generator) -> bool:
        fits = self.matcher.side_max_fits(self.simulated_side(team, rng))
        return any(v >= T.NEAR for v in fits.values())

    def draw_pair(self, rng_team: np.random.Generator, rng_rev: np.random.Generator) -> Tuple[int, int]:
        """A team pair for one game: two distinct teams from ``rng_team``, redrawn while either simulated side is
        near (each side's reveal from ``rng_rev``)."""
        for _ in range(MAX_REDRAWS):
            i, j = (int(x) for x in rng_team.choice(len(self.teams), 2, replace=False))
            self.draws += 1
            if not self.near(i, rng_rev) and not self.near(j, rng_rev):
                return i, j
            self.redraws += 1
        raise RuleError(f"no team pair passed the panel rule in {MAX_REDRAWS} draws")


def pool_teams() -> List[Any]:
    """The 719-team pool as parsed teams, in ``utils.team_sources.packed_teams('pool')``'s order (the core's table)."""
    from main.anchors.team_overlap import parse_team
    from utils.team_sources import team_list

    return [parse_team(x, f"pool{i}") for i, x in enumerate(team_list("pool"))]


def census(profile: Path, draws: int = 20_000, seed: int = SEED_PROFILE) -> Dict[str, Any]:
    """What the rule does to the pool (descriptive): per-team near share, the pass rate of a pair, the surviving
    teams' effective number."""
    shapes, probs, sha = load_profile(profile)
    teams = pool_teams()
    rule = PairRule(shapes, probs, teams)
    rng = np.random.default_rng([seed, 1])
    per = np.zeros(len(teams))
    reps = max(1, draws // len(teams))
    for t in range(len(teams)):
        per[t] = sum(rule.near(t, rng) for _ in range(reps)) / reps
    use = np.bincount([x for _ in range(draws // 2) for x in rule.draw_pair(rng, rng)], minlength=len(teams))
    p = use / use.sum()
    return {"profile_sha256": sha, "teams": len(teams), "reps_per_team": reps,
            "side_near_share": round(float(per.mean()), 4),
            "game_pass_share_independent": round(float(((1 - per) ** 1).mean() ** 2), 4),
            "pair_draws": rule.draws, "pair_redraws": rule.redraws,
            "pair_pass_share": round(1 - rule.redraws / max(rule.draws, 1), 4),
            "teams_never_near": int((per == 0).sum()), "teams_always_near": int((per == 1).sum()),
            "surviving_effective_teams": round(float(1 / (p ** 2).sum()), 1),
            "unrestricted_effective_teams": len(teams)}


def main(argv: Optional[List[str]] = None) -> int:
    from main.human_data import rowset as RS
    from main.human_data import snapshots as S

    ap = argparse.ArgumentParser(prog="python -m main.human_data.ss_rule", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--n", type=int, default=40000)
    p.add_argument("--workers", type=int, default=3)
    c = sub.add_parser("census")
    c.add_argument("--profile", required=True)
    c.add_argument("--draws", type=int, default=20000)
    for q in (p, c):
        q.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    RS._guard_out(out)
    if a.cmd == "profile":
        res = reveal_profile(a.snapshot, a.n, a.workers, S.default_store(), S.default_registry(), S.default_corpus())
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out, "wt") as fh:
            json.dump(res, fh)
        print(json.dumps({k: v for k, v in res.items() if k != "shapes"} | {"distinct_shapes": len(res["shapes"]),
                                                                              "sha256": RS.S_sha(out)}), flush=True)
        return 0
    res = census(Path(a.profile), a.draws)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
