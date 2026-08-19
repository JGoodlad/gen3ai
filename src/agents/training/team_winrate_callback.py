"""Per-team win-rate TRACKING (``--team-wr-tracking``, default ON) — instrumentation only.

The training loop already knows which team each episode pilots and how it ended; nothing kept the
record. This does, keyed by ``team_sha`` (the STRIP-normalized fingerprint from
``agents.training.team_archetypes.team_sha`` — the convention the archetype artifact, MatchupSpec
pins and every provenance record join on), stratified by opponent class. The outcome is the same
one the win-prob label plumbing reads (``info["win_outcome"]``): a win is 1, a **loss OR a tie** is
0, so ``wins/n`` is a win rate and not a score.

**It is an instrument, and this change ships NO prioritization consumer.** The reason is written
into the artifact itself (``notes``) so it travels with the data: a raw per-team win rate conflates
PILOT COMPETENCE with TEAM STRENGTH. Measured, not hypothetical — the ai_v8 team-PFSP work found a
team-PFSP win rate confounded by team strength, so "our win rate with team T is low" does not mean
"we pilot T badly", and anything that spends budget on that signal must first normalize against a
team-strength baseline (the team's pool-average win rate under a reference pilot).

**Seam: an ``env_method`` PULL, not an info-dict thread.** Each worker's teambuilder accumulates a
windowed per-team, per-opponent-class count; this callback drains every worker at a rollout
boundary. That is what makes it correct under BOTH ``SubprocVecEnv`` and ``--async-rollout``:
``env_method`` is drain-safe on ``AsyncSubprocVecEnv`` (it stashes in-flight step results before the
barrier RPC), whereas an info-dict route would have to reconstruct which buffer row a terminal
landed on, which the async collector alone knows. It is also the shape the team-PFSP precedent
already uses for the same reason.

**Distinct from ``--team-pfsp`` and deliberately NOT coupled to it.** That one measures only
self-play POOL battles (bots wash out its weighting signal), is off by default, keys per pool INDEX
and EMA-smooths a rate to drive sampling weights. This one counts every episode, is on by default,
keys per ``team_sha`` and keeps RAW counts (a rate with no denominator cannot serve as a headroom
denominator). The two share the builder's "which team did I just yield" draw index and nothing else
— separate counter tables, separate artifacts.
"""
import json
import os
import time

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from agents.model.opp_intent import OPP_CLASS_NAMES
from agents.model.snapshot import record_team_win_rates

#: The table rides metadata.json as the top-level ``team_win_rates`` block (the owner's
#: recording rule: beside ``latest_eval``'s per-opponent records, NO TensorBoard emission —
#: design_flywheel_tick_tock.md §6b). Deliberately NOT a standalone ``team_winrates.json`` —
#: that name belongs to ``--team-pfsp``'s own differently-keyed snapshot.

_CONFOUND_NOTE = (
    "RAW per-team win rate conflates PILOT COMPETENCE with TEAM STRENGTH (the ai_v8 team-PFSP "
    "lesson): a low win rate with team T may mean we pilot T badly OR that T is weak. Do NOT "
    "prioritize on these numbers directly — normalize against a team-strength baseline (e.g. the "
    "team's pool-average win rate under a reference pilot) first. Also read `by_class`: a "
    "pre-self-play curriculum phase is ~all `bot` episodes, where every team reads ~0.99."
)


class TeamWinRateCallback(BaseCallback):
    """Aggregate per-``team_sha`` win/loss counts across env workers; record into metadata.json (no TB, by owner rule).

    Constructor:
      run_dir       — whose metadata.json carries the table (written + reloaded on restart).
      update_every  — act only every N rollouts (counts accumulate across the skipped ones).
      top_k         — how many teams the ``teams/wr_top_k`` / ``wr_bottom_k`` summaries average over.
      min_games     — a team needs this many games before it enters the top/bottom-k summaries
                      (a 1-game team is 0.0 or 1.0 and would own both ends of the ranking).
    """

    def __init__(self, run_dir: "str | Path | None" = None, update_every: int = 3,
                 top_k: int = 5, min_games: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self._run_dir = str(run_dir) if run_dir else None
        self._update_every = max(1, int(update_every))
        self._top_k = max(1, int(top_k))
        self._min_games = max(1, int(min_games))
        self._rollout_count = 0
        # sha → [games_per_class], sha → [wins_per_class]. CUMULATIVE over the run (and across
        # restarts, via _load_existing) — raw counts, never a smoothed rate.
        self._games: dict[str, list[float]] = {}
        self._wins: dict[str, list[float]] = {}
        self._archetypes: "dict[str, str] | None" = None
        self._loaded = False

    # ── restart safety ────────────────────────────────────────────────────────

    def _load_existing(self) -> None:
        """Load-and-continue from a previous process's metadata block, if present.

        Keyed by ``team_sha``, so unlike a per-INDEX table this survives a pool that was reordered
        (or resized) between runs — an unknown sha is simply a team this run has not seen yet. A
        missing/corrupt file starts fresh; never fatal."""
        if self._loaded:
            return
        self._loaded = True
        if not self._run_dir:
            return
        path = os.path.join(self._run_dir, "metadata.json")
        try:
            with open(path) as f:
                prev = json.load(f).get("team_win_rates") or {}
        except (OSError, ValueError):
            return
        n_classes = len(OPP_CLASS_NAMES)
        for sha, rec in (prev.get("teams") or {}).items():
            by_class = rec.get("by_class") or {}
            games = [0.0] * n_classes
            wins = [0.0] * n_classes
            for code, name in OPP_CLASS_NAMES.items():
                cell = by_class.get(name) or {}
                games[code] = float(cell.get("n", 0.0))
                wins[code] = float(cell.get("wins", 0.0))
            if sum(games) > 0.0:
                self._games[sha] = games
                self._wins[sha] = wins

    # ── the periodic pull ─────────────────────────────────────────────────────

    def _on_step(self) -> bool:      # pragma: no cover - the work is at rollout end
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self._update_every != 0:
            return
        self._load_existing()

        # PULL every worker's window: each is (counts, keys) or None. `counts` is SPARSE — only the
        # pool indices that actually played — because a ~719-team pool touches a handful per window.
        results = [r for r in self.training_env.env_method("drain_team_wr_counts") if r is not None]
        if not results:
            return

        # GIGO guard: the per-INDEX counts are only joinable to a sha if every worker's pool ORDER
        # agrees. Same pool SIZE is not the same pool ORDER, and a diverged order would silently
        # attribute every number to the wrong team — so this THROWS rather than averaging garbage.
        canon = list(results[0][1])
        for _, keys in results[1:]:
            if list(keys) != canon:
                raise RuntimeError(
                    "team-WR per-index team IDENTITY mismatch across env workers — the pool ORDER "
                    "diverged (a per-index count cannot be keyed to a team_sha); every per-team "
                    "win rate this run would be attributed to the wrong team")

        n_classes = len(OPP_CLASS_NAMES)
        for counts, _keys in results:
            for idx, (wins, games) in counts.items():
                i = int(idx)
                if not 0 <= i < len(canon):
                    continue
                sha = canon[i]
                g = self._games.setdefault(sha, [0.0] * n_classes)
                w = self._wins.setdefault(sha, [0.0] * n_classes)
                for c in range(min(n_classes, len(games))):
                    g[c] += float(games[c])
                    w[c] += float(wins[c])

        self._persist()

    # ── emission ──────────────────────────────────────────────────────────────

    def rates(self, min_games: int = 1,
              opp_classes: "list[int] | None" = None) -> dict:
        """``{team_sha: win_rate}`` over teams with at least ``min_games`` games.

        ``opp_classes`` restricts the numerator/denominator to those class codes (e.g.
        ``[OPP_CLASS_POOL]`` for the self-play-only view) — the stratification that makes a raw
        rate readable during a bot-heavy curriculum phase."""
        out = {}
        for sha, games in self._games.items():
            wins = self._wins[sha]
            if opp_classes is None:
                n, w = sum(games), sum(wins)
            else:
                n = sum(games[c] for c in opp_classes if c < len(games))
                w = sum(wins[c] for c in opp_classes if c < len(wins))
            if n >= min_games:
                out[sha] = w / n
        return out

    def table(self) -> dict:
        """The full ``{team_sha: {n, wins, wr, by_class, archetype}}`` table (the artifact body)."""
        arch = self._load_archetypes()
        out = {}
        for sha, games in sorted(self._games.items(), key=lambda kv: sum(kv[1]), reverse=True):
            wins = self._wins[sha]
            n, w = sum(games), sum(wins)
            out[sha] = {
                "n": int(n),
                "wins": int(w),
                "wr": round(w / n, 4) if n else None,
                "archetype": arch.get(sha),
                "by_class": {name: {"n": int(games[code]), "wins": int(wins[code])}
                             for code, name in OPP_CLASS_NAMES.items() if games[code] > 0.0},
            }
        return out

    def _load_archetypes(self) -> "dict[str, str]":
        """Best-effort ``{team_sha: archetype}`` so the table reads 'weakest = stall-class' rather
        than anonymous shas. Joined by the SAME ``team_sha`` convention the keys are built on.
        Never fatal — a missing artifact just means no labels."""
        if self._archetypes is not None:
            return self._archetypes
        out: "dict[str, str]" = {}
        try:
            from agents.training.team_archetypes import load_team_archetypes
            data = load_team_archetypes() or {}
            for sha, rec in (data.get("teams", data) or {}).items():
                label = rec.get("archetype") if isinstance(rec, dict) else None
                if label is not None:
                    out[sha] = str(label)
        except Exception:
            pass
        self._archetypes = out
        return out

    def snapshot(self) -> dict:
        """The artifact payload — the full table plus the caveat that must travel with it."""
        return {
            "step": int(self.num_timesteps),
            "updated_at": time.time(),
            "n_teams_seen": len(self._games),
            "n_games": int(sum(sum(g) for g in self._games.values())),
            "opp_classes": dict(sorted(OPP_CLASS_NAMES.items())),
            "notes": _CONFOUND_NOTE,
            "teams": self.table(),
        }

    def _persist(self) -> None:
        """Write the full table into metadata.json's top-level ``team_win_rates`` block (a full
        snapshot, not history — the counts are cumulative, so the latest block IS the whole
        record and is what a restart reloads). No TensorBoard emission by owner rule."""
        if not self._run_dir:
            return
        try:
            os.makedirs(self._run_dir, exist_ok=True)
            record_team_win_rates(self._run_dir, self.snapshot())
        except Exception as e:      # never take a run down for an instrument
            print(f"[team_wr] metadata write failed (non-fatal): {e}", flush=True)
