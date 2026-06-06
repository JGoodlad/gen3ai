"""Bradley-Terry skill rating ("ELO") for the trained model — pure engine.

WHY THIS EXISTS
---------------
Once training moves to self-play **pool play**, win-rate stops being a legible
progress signal. The promotion gate only promotes a snapshot when
``win_rate_vs_pool > promote_threshold`` and the pool is a *sliding window of recent
selves*, so ``win_rate_vs_pool`` is a treadmill pinned near 50-65% **by construction**
— it cannot trend up no matter how much the model improves. ``win_rate_vs_bots``
saturates near 100% once the fixed bots are solved.

The fix: the 9 eval bots (``random`` + 8 archetypes) are a *fixed yardstick* that never
changes within or across runs. Fit a rating model anchored to them and every frozen
snapshot lands on one absolute scale — a single number that genuinely rises with skill.

No new battles are needed: every eval cycle already plays the trainee (greedy) vs all 9
bots and vs up to 5 pool sentinels, ``EVAL_GAMES`` each. Those win-records ARE a
tournament-matrix row; we accumulate them (``eval_results.jsonl``) and fit ratings.

THE MODEL
---------
**Bradley-Terry** (a.k.a. logistic Elo / BayesElo), fit in **batch** by penalized MLE:

    P(i beats j) = sigmoid( (R_i - R_j) * ln(10)/400 )          [400 Elo pts / decade]

over the accumulated win/games matrix, with a weak Gaussian prior to keep perfect (100-0)
records finite. Batch-BT (not online K-factor Elo) is drift-free and re-runnable; the fit
is a few Newton steps over ~tens of players. Per-player uncertainty (the valuable part of
Glicko) comes from the inverse-Hessian standard error → "ELO 1532 ± 40".

**Anchor.** Bots are *pinned* to high-confidence ratings precomputed by a bot-vs-bot
round-robin (``bot_elo_calibration.py`` → ``data/gen3_bot_elo_anchors.json``); only
snapshots are fit, against that rigid frame. Because the anchor is identical across runs,
snapshot ELOs are directly comparable run-to-run. Fallback when the anchor file is absent:
pin ``random`` at ``base`` (the gauge) and let everything else float.

**Players & edges.** Each bot is keyed ``"bot:<name>"``, each snapshot ``"snap:<step>"``.
A snapshot is the SAME player whether it appears as a cycle's trainee (vs bots) or later
as a sentinel — unified by training step, which is what links the whole ladder together.

KNOWN CAVEAT (acceptable for v1): the trainee acts greedy, sentinels act stochastic@temp,
so a snapshot's rating slightly blends its greedy and stochastic strength — a small
*uniform* systematic effect that shifts the ladder but preserves the progress trend.

This module is intentionally dependency-light (numpy only — no torch/SB3) so it imports
cleanly into the eval callbacks for the live ``eval/elo`` number AND into the offline CLI
(``python -m main.elo``); the fit is the single source of truth for both.
"""
from __future__ import annotations

import os
import glob
import json
import math
from dataclasses import dataclass, field

import numpy as np

# Elo convention: 400 rating points per factor-of-10 odds. ``_S`` is the logistic slope
# that turns an Elo difference into a win probability: P = sigmoid(_S * (R_i - R_j)).
ELO_PER_DECADE = 400.0
_S = math.log(10.0) / ELO_PER_DECADE

# Default anchor base (familiar Elo floor: random ≈ 1000) and prior strength. The prior is
# a weak Gaussian centered at ``base``; ~400 Elo SD ≈ 2.3 nats. With hundreds of games per
# edge the likelihood curvature (~s²·n·p(1-p)) dwarfs the prior's 1/sd² for any player with
# real games — the prior only bites on a separated 100-0 record, exactly where it should.
DEFAULT_BASE = 1000.0
DEFAULT_PRIOR_SD = 400.0

# 95% confidence half-width = Z * SE (one place, used by the live badge AND the offline CLI).
Z95 = 1.96


def ci95(se: float) -> float:
    """95% confidence half-width for a rating standard error."""
    return Z95 * se


def win_prob(r_a: float, r_b: float) -> float:
    """Bradley-Terry win probability of player a over b on the Elo scale."""
    return _sigmoid(_S * (r_a - r_b))

BOT_ANCHORS_PATH = os.path.join("data", "gen3_bot_elo_anchors.json")

BOT = "bot:"
SNAP = "snap:"


def bot_key(name: str) -> str:
    return f"{BOT}{name}"


def snap_key(step: int) -> str:
    return f"{SNAP}{int(step)}"


def is_snapshot(player_id: str) -> bool:
    return player_id.startswith(SNAP)


def snapshot_step(player_id: str) -> int:
    return int(player_id[len(SNAP):])


# ── Data ──────────────────────────────────────────────────────────────────────


@dataclass
class EvalRow:
    """One eval cycle: the trainee frozen at ``step`` played each bot and each sentinel
    ``n_games`` times. ``bots`` maps bot name → trainee win rate; ``sentinels`` is a list
    of (sentinel_step, trainee win rate)."""
    step: int
    n_games: int
    bots: dict[str, float] = field(default_factory=dict)
    sentinels: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class EloFit:
    """Result of :func:`fit_elo`. Ratings are in Elo points on the anchored scale."""
    ratings: dict[str, float]
    se: dict[str, float]
    games: dict[str, int]
    pinned: set[str]
    base: float
    n_iter: int
    converged: bool

    def snapshot_curve(self) -> list[tuple[int, float, float]]:
        """[(step, elo, se), …] for every snapshot player, sorted by training step."""
        out = [(snapshot_step(p), self.ratings[p], self.se.get(p, 0.0))
               for p in self.ratings if is_snapshot(p)]
        return sorted(out)

    def bot_ratings(self) -> dict[str, tuple[float, float]]:
        """{bot_name: (elo, se)} for every bot player."""
        return {p[len(BOT):]: (self.ratings[p], self.se.get(p, 0.0))
                for p in self.ratings if p.startswith(BOT)}

    def latest_snapshot(self) -> tuple[int, float, float] | None:
        """The (step, elo, se) of the highest-step snapshot, or None."""
        curve = self.snapshot_curve()
        return curve[-1] if curve else None

    def rating_for_step(self, step: int) -> tuple[float, float] | None:
        """(elo, se) for the snapshot frozen at ``step``, or None if it isn't rated."""
        key = snap_key(step)
        if key not in self.ratings:
            return None
        return self.ratings[key], self.se.get(key, 0.0)


# ── Loaders ─────────────────────────────────────────────────────────────────


def load_bot_anchors(path: str = BOT_ANCHORS_PATH) -> dict | None:
    """Read the precomputed bot anchor ratings, or None if the file is absent/invalid.

    Returns the full anchor dict (``ratings``/``se``/``base``/…) from
    ``bot_elo_calibration.py``. ``fit_elo`` consumes ``ratings`` (and ``base``) to pin the
    bots; a missing file is the graceful fallback (``fit_elo`` then pins ``random``)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("ratings") else None


def _rows_from_log(run_dir: str) -> list[EvalRow]:
    """Parse the append-only ``eval_results.jsonl`` (the canonical source). Dedups by
    step (last write wins — a resume can re-eval the same step)."""
    path = os.path.join(run_dir, "eval_results.jsonl")
    by_step: dict[int, EvalRow] = {}
    if not os.path.exists(path):
        return []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            step = int(r["step"])
            sentinels = [(int(s["step"]), float(s["win_rate"]))
                         for s in r.get("sentinels", [])]
            by_step[step] = EvalRow(
                step=step,
                n_games=int(r.get("n_games", 100)),
                bots={k: float(v) for k, v in (r.get("bots") or {}).items()},
                sentinels=sentinels,
            )
    return [by_step[s] for s in sorted(by_step)]


def _row_from_block(block: dict, n_games: int = 100) -> "EvalRow | None":
    """Build one EvalRow from a metadata ``latest_eval`` block — its ``opponents`` (bot win
    rates) + optional ``pool.sentinels``. Shared by the metadata loader and the resume-republish
    ELO compute. Returns None if the block has no ``step``."""
    step = block.get("step")
    if step is None:
        return None
    bots = {name: float(d["win_rate"])
            for name, d in (block.get("opponents") or {}).items()
            if isinstance(d, dict) and "win_rate" in d}
    sentinels = [(int(s["step"]), float(s["win_rate"]))
                 for s in ((block.get("pool") or {}).get("sentinels") or [])
                 if "step" in s and "win_rate" in s]
    return EvalRow(int(step), n_games, bots, sentinels)


def _rows_from_metadata(run_dir: str, n_games: int = 100) -> list[EvalRow]:
    """Reconstruct rows from ``metadata.json`` — the top-level ``latest_eval`` plus every
    per-checkpoint ``snapshot_history[*].latest_eval`` (sparse, but survives even when the
    jsonl was never written). Each block carries ``opponents`` (bot win rates) and an
    optional ``pool.sentinels``."""
    path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return []
    blocks = []
    if isinstance(meta.get("latest_eval"), dict):
        blocks.append(meta["latest_eval"])
    for entry in (meta.get("snapshot_history") or {}).values():
        if isinstance(entry, dict) and isinstance(entry.get("latest_eval"), dict):
            blocks.append(entry["latest_eval"])
    by_step: dict[int, EvalRow] = {}
    for blk in blocks:
        row = _row_from_block(blk, n_games)
        if row is not None:
            by_step[row.step] = row
    return [by_step[s] for s in sorted(by_step)]


def _rows_from_tensorboard(run_dir: str, n_games: int = 100) -> list[EvalRow]:
    """Scrape rows from the run's TensorBoard event files — the per-cycle
    ``eval/win_rate_vs_<bot>`` scalars plus the positional ``eval/win_rate_vs_sentinel_<i>``
    + ``eval/sentinel_step_<i>`` pairs. This backfills a run that predates
    ``eval_results.jsonl`` (i.e. an already-running training run) with ZERO training change."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []
    # SB3 writes the TB logs to a SEPARATE `<repo>/tensorboard/*<timestamp>*/` dir, NOT under the
    # run dir — resolve it via the project's helper (the same one the launcher's plots use).
    event_files: list[str] = []
    try:
        from utils.plot_tb import find_tb_dir_for_run
        tb_dir = find_tb_dir_for_run(run_dir)
        if tb_dir:
            event_files = glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents.*"),
                                    recursive=True)
    except Exception:  # noqa: BLE001 — resolver is best-effort; fall back to the run dir
        pass
    if not event_files:
        # Fallback: globbing under run_dir itself (covers run_dir already BEING a tb dir).
        event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"),
                                recursive=True)
    if not event_files:
        return []
    # Bot roster names (so we can tell bot edges apart from the aggregate keys).
    try:
        from agents.training.eval_callback import eval_opponent_names
        bot_names = set(eval_opponent_names())
    except Exception:  # noqa: BLE001 — fall back to a static roster if the import chain fails
        # Last-resort mirror of _EVAL_OPPONENT_SPECS — keep in sync if the roster changes (the
        # tb backfill is best-effort; a stale name here just drops that bot's edges).
        bot_names = {"random", "heuristic", "heuristic2", "staller", "staller_v2",
                     "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2"}

    # step -> {bot_name: wr}, step -> {sentinel_idx: wr}, step -> {sentinel_idx: sentinel_step}
    bot_wr: dict[int, dict[str, float]] = {}
    sent_wr: dict[int, dict[int, float]] = {}
    sent_step: dict[int, dict[int, int]] = {}

    for ef in event_files:
        try:
            acc = EventAccumulator(ef, size_guidance={"scalars": 0})
            acc.Reload()
            tags = set(acc.Tags().get("scalars", []))
        except Exception:  # noqa: BLE001 — a half-written events file is non-fatal
            continue
        for tag in tags:
            if not tag.startswith("eval/"):
                continue
            sub = tag[len("eval/"):]
            try:
                events = acc.Scalars(tag)
            except Exception:  # noqa: BLE001
                continue
            if sub.startswith("win_rate_vs_sentinel_"):
                idx = int(sub[len("win_rate_vs_sentinel_"):])
                for e in events:
                    sent_wr.setdefault(int(e.step), {})[idx] = float(e.value)
            elif sub.startswith("sentinel_step_"):
                idx = int(sub[len("sentinel_step_"):])
                for e in events:
                    sent_step.setdefault(int(e.step), {})[idx] = int(round(e.value))
            elif sub.startswith("win_rate_vs_"):
                name = sub[len("win_rate_vs_"):]
                if name in bot_names:
                    for e in events:
                        bot_wr.setdefault(int(e.step), {})[name] = float(e.value)

    steps = sorted(set(bot_wr) | set(sent_wr))
    rows = []
    for step in steps:
        sentinels = []
        for idx, wr in sorted((sent_wr.get(step) or {}).items()):
            s_step = (sent_step.get(step) or {}).get(idx)
            if s_step is not None:
                sentinels.append((s_step, wr))
        rows.append(EvalRow(step, n_games, bot_wr.get(step, {}), sentinels))
    return rows


def load_rows(run_dir: str, source: str = "auto", n_games: int = 100) -> list[EvalRow]:
    """Load the eval results for a run. ``source``: ``log`` (eval_results.jsonl),
    ``tb`` (TensorBoard backfill), ``meta`` (metadata.json), or ``auto`` (first non-empty
    of log → tb → meta). ``n_games`` is the assumed per-pairing game count for the tb/meta
    backfills (the jsonl stores its own)."""
    if source == "log":
        return _rows_from_log(run_dir)
    if source == "tb":
        return _rows_from_tensorboard(run_dir, n_games)
    if source == "meta":
        return _rows_from_metadata(run_dir, n_games)
    if source != "auto":
        raise ValueError(f"unknown source {source!r}")
    for loader in (_rows_from_log,
                   lambda d: _rows_from_tensorboard(d, n_games),
                   lambda d: _rows_from_metadata(d, n_games)):
        rows = loader(run_dir)
        if rows:
            return rows
    return []


# ── Fit ────────────────────────────────────────────────────────────────────


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _rows_to_results(rows: list[EvalRow]):
    """Lower eval rows into the generic ``(name_a, name_b, wins_a, games)`` pairwise form the
    BT fit consumes — the trainee (``snap:<step>``) vs each bot (``bot:<name>``) and each
    sentinel (``snap:<step>``). Win counts are recovered from the stored rate × that cycle's
    ``n_games``. Shared with the round-robin via ``_aggregate``."""
    for row in rows:
        n = max(1, int(row.n_games))
        subj = snap_key(row.step)
        for name, wr in row.bots.items():
            yield subj, bot_key(name), int(round(_clamp01(wr) * n)), n
        for m_step, wr in row.sentinels:
            yield subj, snap_key(m_step), int(round(_clamp01(wr) * n)), n


def _aggregate(results) -> tuple[list[str], dict, dict]:
    """Intern players and fold ``(name_a, name_b, wins_a, games)`` results into an undirected
    edge set keyed by ``(lo, hi)`` player indices, storing the lexicographically-lower player's
    wins + total games. The ONE place the pair-ordering / win-flipping convention lives — both
    the snapshot ladder (`fit_elo`) and the round-robin (`fit_pairwise`) aggregate through here."""
    names: list[str] = []
    idx: dict[str, int] = {}

    def pid(name: str) -> int:
        if name not in idx:
            idx[name] = len(names)
            names.append(name)
        return idx[name]

    edges: dict[tuple[int, int], list[int]] = {}
    for na, nb, wa, g in results:
        a, b = pid(na), pid(nb)
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        wins_lo = wa if a < b else (g - wa)
        e = edges.setdefault((lo, hi), [0, 0])
        e[0] += int(wins_lo)
        e[1] += int(g)
    return names, idx, edges


def _fit_aggregated(names, idx, edges, pinned, base, prior_sd, max_iter, tol):
    """Pin → damped-Newton → unpack, over an already-aggregated edge set. The shared fit core
    for both public entry points. ``pinned`` maps player-id → fixed Elo (already in ``names``'
    keyspace); with no pins the best-connected player is pinned at ``base`` to fix the gauge.
    Returns ``(ratings, se, pinned_keys, n_iter, converged, games)``."""
    n = len(names)
    if n == 0:
        return {}, {}, set(), 0, True, {}
    R = np.full(n, base, dtype=np.float64)
    fixed = np.zeros(n, dtype=bool)
    pinned_keys: set[str] = set()
    for name, elo in (pinned or {}).items():
        if name in idx:
            R[idx[name]] = float(elo)
            fixed[idx[name]] = True
            pinned_keys.add(name)
    if not pinned_keys:
        anchor = _best_connected(names, edges)
        if anchor is not None:
            R[idx[anchor]] = base
            fixed[idx[anchor]] = True
            pinned_keys.add(anchor)

    edge_list = [(a, b, w, g) for (a, b), (w, g) in edges.items()]
    R, se_arr, it, converged = _newton(R, fixed, edge_list, base, prior_sd, max_iter, tol)

    games = {name: 0 for name in names}
    for (a, b), (_w, g) in edges.items():
        games[names[a]] += g
        games[names[b]] += g
    ratings = {names[i]: float(R[i]) for i in range(n)}
    se = {names[i]: float(se_arr[i]) for i in range(n)}
    return ratings, se, pinned_keys, it, converged, games


def fit_elo(
    rows: list[EvalRow],
    pin_ratings: dict[str, float] | None = None,
    *,
    base: float = DEFAULT_BASE,
    prior_sd: float = DEFAULT_PRIOR_SD,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> EloFit:
    """Fit anchored Bradley-Terry ratings (Elo scale) over the accumulated eval results.

    ``pin_ratings`` maps a bot name → fixed Elo (the precomputed anchors); those bots are held
    rigid and only snapshots are optimized. When ``None``, ``random`` is pinned at ``base`` to
    fix the gauge (the other bots float, determined indirectly via the snapshots). Damped
    Newton on the penalized log-likelihood; SE = sqrt(diag(H⁻¹))."""
    names, idx, edges = _aggregate(_rows_to_results(rows))
    if pin_ratings:
        pinned = {bot_key(name): float(elo) for name, elo in pin_ratings.items()}
    else:
        # Fallback gauge: pin `random` at base; if it isn't present `_fit_aggregated` falls
        # back to the best-connected player.
        rk = bot_key("random")
        pinned = {rk: base} if rk in idx else None
    ratings, se, pinned_keys, it, converged, games = _fit_aggregated(
        names, idx, edges, pinned, base, prior_sd, max_iter, tol)
    return EloFit(ratings, se, games, pinned_keys, base, it, converged)


def fit_pairwise(
    results: list[tuple[str, str, int, int]],
    pinned: dict[str, float] | None = None,
    *,
    base: float = DEFAULT_BASE,
    prior_sd: float = DEFAULT_PRIOR_SD,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> tuple[dict[str, float], dict[str, float], bool]:
    """Fit Bradley-Terry ratings over an arbitrary set of pairwise results — the round-robin
    entry point (the snapshot ladder uses ``fit_elo``). ``results`` is
    ``[(name_a, name_b, wins_a, games), …]``; ``pinned`` maps name → fixed Elo (else the
    best-connected player is pinned at ``base``). Returns ``(ratings, se, converged)``."""
    names, idx, edges = _aggregate(results)
    ratings, se, _keys, _it, converged, _games = _fit_aggregated(
        names, idx, edges, pinned, base, prior_sd, max_iter, tol)
    return ratings, se, converged


def _softplus(x: float) -> float:
    """log(1 + e^x), numerically stable."""
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _nll(R, fixed, edge_list, base, inv_var) -> float:
    """Penalized negative log-likelihood — the objective the Newton step descends. Used by the
    line search so a step is only accepted when it actually decreases the objective."""
    total = 0.0
    for a, b, w, g in edge_list:
        z = _S * (R[a] - R[b])
        total += w * _softplus(-z) + (g - w) * _softplus(z)
    for i in range(len(R)):
        if not fixed[i]:
            total += 0.5 * inv_var * (R[i] - base) ** 2
    return total


def _newton(R, fixed, edge_list, base, prior_sd, max_iter, tol):
    """Penalized-MLE **damped** Newton-Raphson core, shared by ``fit_elo`` / ``fit_pairwise``.

    ``R`` holds initial ratings (fixed entries are the pinned values); ``fixed`` is a bool
    mask; ``edge_list`` is ``[(a, b, wins_a, games), …]`` over player INDICES. Optimizes the
    free players against the penalized BT log-likelihood (Gaussian prior centered at
    ``base``). A backtracking line search (step-halving) guarantees descent on this concave
    objective — a full Newton step can overshoot and diverge when the data and a pinned
    anchor are far apart (e.g. near-perfect win rates). Returns ``(R, se, n_iter, converged)``
    with ``se`` an array (0 for fixed)."""
    n = len(R)
    free = np.where(~fixed)[0]
    free_pos = {int(p): k for k, p in enumerate(free)}
    F = len(free)
    inv_var = 1.0 / (prior_sd * prior_sd)

    converged = F == 0
    it = 0
    if F:
        for it in range(1, max_iter + 1):
            grad, H = _grad_and_hessian(R, fixed, edge_list, free, free_pos, base, inv_var)
            if np.max(np.abs(grad)) < tol:
                converged = True
                break
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            # Backtracking line search: accept the largest t∈{1,½,¼,…} that decreases the NLL.
            nll0 = _nll(R, fixed, edge_list, base, inv_var)
            t = 1.0
            accepted = False
            for _ in range(40):
                trial = R.copy()
                trial[free] -= t * step
                if _nll(trial, fixed, edge_list, base, inv_var) <= nll0:
                    R = trial
                    accepted = True
                    break
                t *= 0.5
            if not accepted or t * np.max(np.abs(step)) < tol:
                converged = True
                break

    # SE from the inverse Hessian at the optimum — same curvature the step descended, so the
    # assembly MUST match: both go through `_grad_and_hessian` (the grad is ignored here).
    se = np.zeros(n)
    if F:
        _g, Hf = _grad_and_hessian(R, fixed, edge_list, free, free_pos, base, inv_var)
        try:
            cov = np.linalg.inv(Hf)
            for p_idx, k in free_pos.items():
                v = cov[k, k]
                se[p_idx] = math.sqrt(v) if v > 0 else 0.0
        except np.linalg.LinAlgError:
            pass
    return R, se, it, converged


def _grad_and_hessian(R, fixed, edge_list, free, free_pos, base, inv_var):
    """Gradient + Hessian of the penalized NLL over the FREE players at ``R``. The single
    source for both the Newton step and the SE covariance, so they can never disagree."""
    F = len(free)
    grad = np.zeros(F)
    H = np.zeros((F, F))
    for a, b, w, g in edge_list:
        p = _sigmoid(_S * (R[a] - R[b]))
        d = _S * (w - g * p)
        curv = _S * _S * g * p * (1.0 - p)
        fa, fb = fixed[a], fixed[b]
        if not fa:
            ia = free_pos[a]
            grad[ia] -= d
            H[ia, ia] += curv
        if not fb:
            ib = free_pos[b]
            grad[ib] += d
            H[ib, ib] += curv
        if not fa and not fb:
            ia, ib = free_pos[a], free_pos[b]
            H[ia, ib] -= curv
            H[ib, ia] -= curv
    for k, p_idx in enumerate(free):
        grad[k] += (R[p_idx] - base) * inv_var
        H[k, k] += inv_var
    return grad, H


def _best_connected(players: list[str], edges: dict) -> str | None:
    deg: dict[int, int] = {}
    for (a, b), (_w, g) in edges.items():
        deg[a] = deg.get(a, 0) + g
        deg[b] = deg.get(b, 0) + g
    if not deg:
        return None
    return players[max(deg, key=deg.get)]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def elo_from_eval_block(block: dict, anchors_path: str = BOT_ANCHORS_PATH,
                        n_games: int = 100) -> tuple[float, float] | None:
    """Compute ``(elo, se)`` for the snapshot a single ``latest_eval`` block describes — fit from
    its own win rates (vs bots + sentinels) anchored to the bots. Lets the resume-republish show
    ELO immediately even when the saved block predates the ``elo`` field (e.g. resuming a
    checkpoint saved before this feature). Returns None without usable bot win rates."""
    row = _row_from_block(block, n_games)
    if row is None or not row.bots:
        return None
    anchors = load_bot_anchors(anchors_path)
    pin = anchors["ratings"] if anchors else None
    base = float(anchors["base"]) if anchors and "base" in anchors else DEFAULT_BASE
    return fit_elo([row], pin_ratings=pin, base=base).rating_for_step(row.step)


def fit_from_run(run_dir: str, source: str = "auto",
                 anchors_path: str = BOT_ANCHORS_PATH) -> EloFit:
    """Convenience: load a run's rows + bot anchors and fit. Used by the live callbacks
    and the offline CLI so they share one path."""
    anchors = load_bot_anchors(anchors_path)
    pin = anchors["ratings"] if anchors else None
    base = float(anchors["base"]) if anchors and "base" in anchors else DEFAULT_BASE
    fit = fit_elo(load_rows(run_dir, source), pin_ratings=pin, base=base)
    # Stamp anchor SEs onto the pinned bots for display, when known.
    if anchors and isinstance(anchors.get("se"), dict):
        for name, s in anchors["se"].items():
            k = bot_key(name)
            if k in fit.ratings:
                fit.se[k] = float(s)
    return fit
