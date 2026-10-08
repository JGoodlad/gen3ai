"""REPRODUCIBLE battles on the Rust env core — the training row path, seeded end to end (P6 of the poke-env retirement).

The Rust twin of ``obs_roundtrip_fuzz_test.record_fixture_battle`` (the poke-env bridge battle a collected test used
to take its battle from): the SAME battles every run, on the observation path training reads. Reproducibility is
declared, never assumed — fixed teams (the pool's first ``n_teams`` packed teams, drawn by a seeded generator), a
fixed battle seed per episode (four 16-bit words from the same generator), a seeded uniform-over-the-mask policy for
p1, a Rust BOT on p2 with ``"streams": "episode"`` (its draws a function of the game alone), ONE thread and the
envs stepped in index order. Two calls with the same arguments return byte-identical rows (pinned by
``fixture_battles_test.py``).

:func:`play_rows` returns p1's decision rows and masks (what ``agents.model.compile_parity_fixture`` takes its rows
from). The core is the cdylib :func:`utils.rust_env.build.ensure_built` builds for ``profile``.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

#: The p2 bot every fixture battle plays (a deterministic-per-game stream: ``"streams": "episode"``).
FIXTURE_BOT = {"kind": "bot", "bot": "heuristic", "seed": 3, "streams": "episode"}
NAMES = ("fixone", "fixtwo")


def _lib(profile: str, lib=None):
    if lib is not None:
        return lib
    from utils.rust_env import ffi
    from utils.rust_env.build import ensure_built

    ensure_built(profile)
    return ffi.load(ffi.default_path(profile))


#: ``policy(obs (k, OBS_DIM) f32, mask (k, 11) bool) -> k action indices`` — a batched decision rule.
Policy = Callable[[np.ndarray, np.ndarray], Sequence[int]]


def play_rows(n_battles: int, *, seed: int, n_teams: Optional[int] = 12, turn_limit: int = 250, n_envs: int = 2,
              policy: Optional[Policy] = None, self_play: bool = False, profile: str = "release",
              lib=None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Play ``n_battles`` seeded games; per finished game (in finish order) p1's ``(obs (k, OBS_DIM) f32, mask (k,
    11) bool)``. A p1 decision at ``turn >= turn_limit`` is the stall forfeit (no row), as in training.

    p1 plays ``policy`` (default: a seeded uniform draw over the mask). ``self_play``: p2 is an EXTERNAL route
    answered by the same ``policy`` (a checkpoint against itself) instead of the episode-seeded Rust bot.
    ``n_teams``: the pool's first ``n_teams`` packed teams (``None`` = the whole pool)."""
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P
    from utils.team_sources import packed_teams

    if self_play and policy is None:
        raise ValueError("self_play needs a policy for p2")
    teams = list(packed_teams("pool"))
    if n_teams is not None:
        teams = teams[:n_teams]
    opp = {"kind": "external"} if self_play else dict(FIXTURE_BOT)
    spec = P.spec_json(n=n_envs, threads=1, teams=teams, names=NAMES, turn_limit=turn_limit, refusal_budget=4,
                       bank_dir=None, opponents=[opp])
    core = ffi.FfiCore(spec, lib=_lib(profile, lib))
    c = core.cols
    rng = np.random.default_rng(seed)
    sides = (0, 1) if self_play else (0,)

    def stage(envs) -> None:
        for e in envs:
            c["ep_team"][e] = rng.choice(len(teams), 2, replace=False)
            c["ep_seed"][e] = rng.integers(0, 65536, 4)
            c["ep_opp"][e] = 0

    def decide(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if policy is None:
            return np.array([int(rng.choice(np.flatnonzero(m))) for m in mask], dtype=np.int64)
        return np.asarray(policy(obs, mask), dtype=np.int64)

    stage(range(n_envs))
    core.reset()
    stage(range(n_envs))
    buf: dict = {e: [] for e in range(n_envs)}
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    try:
        while len(out) < n_battles:
            # every pending decision, in (env, side) order; p1's at the stall threshold is the forfeit (no row)
            pend = [(e, s) for e in range(n_envs) for s in sides
                    if c["need"][e, s] and not (s == 0 and int(c["turn"][e]) >= turn_limit)]
            if pend:
                obs = np.stack([c["obs"][e, s].copy() for e, s in pend]).astype(np.float32)
                mask = np.stack([c["mask"][e, s].astype(bool) for e, s in pend])
                acts = decide(obs, mask)
                for k, (e, s) in enumerate(pend):
                    if not mask[k][acts[k]]:
                        raise ValueError(f"the policy chose an illegal action {acts[k]} (env {e}, p{s + 1})")
                    if s == 0:
                        buf[e].append((obs[k], mask[k]))
                    c["action"][e, s] = int(acts[k])
            core.step()
            for f in core.finished():
                e = f["env"]
                rows, buf[e] = buf[e], []
                if len(out) < n_battles and rows:
                    out.append((np.stack([r[0] for r in rows]).astype(np.float32), np.stack([r[1] for r in rows])))
    finally:
        close = getattr(core, "close", None)
        if close is not None:
            close()
    return out


def spread_rows(n_rows: int, *, n_battles: int, seed: int, lib=None, profile: str = "release"
                ) -> Tuple[np.ndarray, np.ndarray]:
    """``n_rows`` rows taken EVENLY across each of ``n_battles`` seeded games (early, mid and late states)."""
    games = play_rows(n_battles, seed=seed, lib=lib, profile=profile)
    per = int(np.ceil(n_rows / max(len(games), 1)))
    obs_rows, mask_rows = [], []
    for obs, mask in games:
        pick = np.unique(np.linspace(0, len(obs) - 1, num=min(per, len(obs))).round().astype(int))
        obs_rows.append(obs[pick])
        mask_rows.append(mask[pick])
    obs = np.concatenate(obs_rows)[:n_rows]
    mask = np.concatenate(mask_rows)[:n_rows]
    return obs, mask

