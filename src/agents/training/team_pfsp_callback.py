"""Team-side PFSP — variance-weighted TEAM sampling by self-play win-rate.

Biases the trainee's TEAM sampling toward pool teams it wins ~50% with (max variance = most to
learn), away from teams it already crushes or always loses. Off by default (byte-identical).

Centralized aggregation (NOT per-worker EMA — the pool is ~700 teams so per-worker counts are too
sparse; NOT info-dict threading — that breaks under --async-rollout): each cycle this callback
PULLS every worker's windowed per-team win/loss counts via ``env_method``, sums them, EMA-smooths a
global per-team win-rate ``p_i``, computes the variance weight ``floor + p_i*(1-p_i)`` (capped at
``cap × mean``), and PUSHES the weights back to every worker's teambuilder. The teambuilder samples
its pool with ``random.choices(weights=...)`` and only accumulates outcomes — the math lives here.

The per-team signal is measured ONLY on self-play POOL battles (the wrapper excludes bots — we win
~0.99 vs bots, which washes out the variance signal) over the POOL teams (bias/distill-pinned teams
are excluded — they get fixed exposure via the existing bias fraction).
"""
import json
import os
import time

from stable_baselines3.common.callbacks import BaseCallback


def compute_team_pfsp_weights(emas, floor, cap):
    """Pure weight math (unit-tested): ``raw_i = floor + p_i*(1-p_i)``; then cap each at
    ``cap × mean(raw)`` (no team may exceed ``cap`` × the uniform sampling share). Returns the
    per-team weight list to hand to ``random.choices(weights=...)``."""
    raw = [floor + p * (1.0 - p) for p in emas]
    n = len(raw)
    if n == 0:
        return []
    mean = sum(raw) / n
    cap_val = cap * mean
    return [min(r, cap_val) for r in raw]


class TeamPFSPCallback(BaseCallback):
    """Aggregate per-team self-play win-rates across env workers and push variance weights.

    Constructor:
      cap / floor    — the weight cap (× uniform share) and floor (raw = floor + p*(1-p)).
      ema_beta       — EMA smoothing of each team's win-rate (higher = slower, damps eval noise).
      update_every   — act only every N rollouts (the counts accumulate across the skipped ones).
      mode           — "var" (bias sampling: push the variance weights) | "measure" (track + persist
                       the per-team win-rate ONLY, never push → sampling stays uniform). Both persist.
      persist_dir    — if set, write a ``team_winrates.json`` snapshot there each update (the offline
                       "which team is the generalist weakest on → next exploiter target" artifact).
    """

    def __init__(self, cap: float, floor: float, ema_beta: float = 0.7,
                 update_every: int = 3, mode: str = "var", persist_dir=None, verbose: int = 0):
        super().__init__(verbose)
        self._cap = float(cap)
        self._floor = float(floor)
        self._ema_beta = float(ema_beta)
        self._update_every = max(1, int(update_every))
        self._mode = str(mode)
        self._persist_dir = persist_dir
        self._ema = None            # lazily sized to n_pool on first pull, seeded 0.5
        self._cum_games = None      # cumulative self-play games per pool team (persisted for weight/confidence)
        self._rollout_count = 0
        self._update_count = 0      # times we actually pulled/pushed (for the sparse audit log)
        self._measured = set()      # pool indices that ever received a game (for n_measured)
        self._keys = None           # canonical per-index team fingerprints (verified once)
        self._archetypes = None     # {team_sha: archetype label}, best-effort (annotates the snapshot)

    def _on_step(self) -> bool:
        return True

    def _verify_keys_once(self) -> None:
        """ONE-TIME strong GIGO guard: the per-INDEX team fingerprint must be identical across every
        worker. Same pool SIZE ≠ same pool ORDER — this catches an actual identity divergence (which
        would silently mis-attribute win-rates) that the per-cycle size check can't. Static data →
        pulled once, then cached (also keys the weakest-team audit)."""
        if self._keys is not None:
            return
        key_results = [k for k in self.training_env.env_method("get_team_pfsp_keys") if k is not None]
        if not key_results:
            return
        canon = list(key_results[0])
        for kr in key_results[1:]:
            if list(kr) != canon:
                raise RuntimeError(
                    "team-PFSP per-index team IDENTITY mismatch across env workers — the pool ORDER "
                    "diverged (same size, different team per index); the per-index win-rate contract "
                    "is broken (would silently mis-attribute win-rates to the wrong teams)")
        self._keys = canon

    def _on_rollout_end(self) -> bool:
        self._rollout_count += 1
        if self._rollout_count % self._update_every != 0:
            return True

        self._verify_keys_once()

        # 1) PULL every worker's windowed counts; each is (wins, games, n_pool) or None.
        results = [r for r in self.training_env.env_method("drain_team_pfsp_counts")
                   if r is not None]
        if not results:
            return True

        # 2) GIGO-guard: every worker must report the SAME pool size (a cheap per-cycle belt; the
        #    stronger per-index identity check runs once in _verify_keys_once).
        n_pools = {int(r[2]) for r in results}
        if len(n_pools) != 1:
            raise RuntimeError(
                f"team-PFSP pool-size mismatch across env workers: {sorted(n_pools)} — the "
                "per-index win-rate contract requires an identical pool order in every worker")
        n_pool = next(iter(n_pools))

        # 3) Sum wins/games element-wise across workers → this window's global counts.
        W = [0.0] * n_pool
        G = [0.0] * n_pool
        for wins, games, _ in results:
            for i in range(n_pool):
                W[i] += wins[i]
                G[i] += games[i]

        # 4) EMA-update each measured team's win-rate; unmeasured teams keep their EMA. Accumulate
        #    cumulative games per team (persisted as the confidence/weight of each win-rate estimate).
        if self._ema is None:
            self._ema = [0.5] * n_pool
            self._cum_games = [0.0] * n_pool
        for i in range(n_pool):
            self._cum_games[i] += G[i]
            if G[i] > 0.0:
                p = W[i] / G[i]
                self._ema[i] = self._ema_beta * self._ema[i] + (1.0 - self._ema_beta) * p
                self._measured.add(i)

        # 5) Compute the variance weights (capped) — 6) PUSH them ONLY in "var" mode ("measure" tracks
        #    + persists the win-rate but never biases sampling, so the team distribution stays uniform).
        w = compute_team_pfsp_weights(self._ema, self._floor, self._cap)
        if self._mode == "var":
            self.training_env.env_method("set_team_pfsp_weights", w)

        # 7) Diagnostics + auditability.
        self._update_count += 1
        mean_w = (sum(w) / len(w)) if w else 0.0
        self.logger.record("team_pfsp/min_wr", min(self._ema))
        self.logger.record("team_pfsp/max_wr", max(self._ema))
        self.logger.record("team_pfsp/n_measured", len(self._measured))
        self.logger.record("team_pfsp/weight_spread", (max(w) / mean_w) if mean_w > 0 else 0.0)
        # Audit which teams the budget concentrates on: a sparse (every 10th update) line naming the
        # weakest MEASURED pool teams by their fingerprint@win-rate (joins the archetype/provenance
        # records via team_sha), so the weighting is inspectable — not an anonymous min/max scalar.
        if self._keys is not None and len(self._keys) == n_pool and self._measured and \
                self._update_count % 10 == 1:
            weakest = sorted(self._measured, key=lambda i: self._ema[i])[:5]
            pairs = ", ".join(f"{self._keys[i]}@{self._ema[i]:.2f}" for i in weakest)
            print(f"[team_pfsp] weakest measured pool teams (sha@win-rate): {pairs}", flush=True)

        # 8) Persist the per-team self-play win-rate snapshot (offline "which exploiter next" artifact).
        self._persist_snapshot(n_pool)
        return True

    def _load_archetypes(self):
        """Best-effort {team_sha: archetype-label} to annotate the snapshot (so the offline view reads
        'weakest = stall-class', not just anonymous shas). Never fatal — missing artifact → no labels."""
        if self._archetypes is not None:
            return self._archetypes
        self._archetypes = {}
        try:
            from agents.training.team_archetypes import load_team_archetypes
            data = load_team_archetypes() or {}
            teams = data.get("teams", data)   # {sha: {archetype, pace_score, tags, ...}}
            for sha, rec in teams.items():
                if isinstance(rec, dict) and "archetype" in rec:
                    self._archetypes[sha] = rec.get("archetype")
        except Exception:
            pass
        return self._archetypes

    def _persist_snapshot(self, n_pool) -> None:
        """Overwrite ``<persist_dir>/team_winrates.json`` with the current per-team win-rate estimates
        (weakest-first) — a bounded snapshot (not history) for offline exploiter-target selection."""
        if not self._persist_dir or self._keys is None or len(self._keys) != n_pool:
            return
        arch = self._load_archetypes()
        rows = [
            {"sha": self._keys[i], "win_rate": round(self._ema[i], 4),
             "games": int(self._cum_games[i]), "archetype": arch.get(self._keys[i])}
            for i in sorted(self._measured, key=lambda i: self._ema[i])
        ]
        step = int(self.num_timesteps)
        snap = {"step": step, "updated_at": time.time(), "mode": self._mode,
                "n_teams_measured": len(rows), "teams": rows}
        try:
            os.makedirs(self._persist_dir, exist_ok=True)
            tmp = os.path.join(self._persist_dir, "team_winrates.json.tmp")
            with open(tmp, "w") as f:
                json.dump(snap, f, indent=0)
            os.replace(tmp, os.path.join(self._persist_dir, "team_winrates.json"))  # atomic latest snapshot
            # Append a compact STEP-TAGGED row to a history JSONL so the win-rates can be tracked OVER
            # TIME offline (per-team trends + noise, not just the latest snapshot). One line per update.
            hist = {"step": step, "wr": {self._keys[i]: round(self._ema[i], 4)
                                         for i in self._measured}}
            with open(os.path.join(self._persist_dir, "team_winrates_history.jsonl"), "a") as f:
                f.write(json.dumps(hist) + "\n")
        except Exception as e:
            print(f"[team_pfsp] snapshot write failed (non-fatal): {e}", flush=True)
