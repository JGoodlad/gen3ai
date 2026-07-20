from poke_env.teambuilder import Teambuilder
from utils.gen3_utils import fix_gen3_hp_ivs
import hashlib
import random

class Gen3Teambuilder(Teambuilder):
    """
    A specialized Teambuilder for Gen 3 (ADV) that automatically handles
    validation nuances like Hidden Power IV mappings.
    """

    def __init__(self, teams, bias_teams=None, bias_prob=0.0,
                 team_pfsp="off", team_pfsp_cap=3.0, team_pfsp_floor=0.05):
        """
        Initialize with a single team string or a list of team strings.

        bias_teams: optional secondary pool; when provided, yield_team picks from
                    it with probability bias_prob (and from the main pool otherwise).
        bias_prob:  float in [0, 1].  0.0 = always use main pool (default).

        team_pfsp:  "off" (default) → yield_team is the byte-identical uniform
                    ``random.choice(self.packed_teams)`` (no new RNG draws, no tracking).
                    "var" → variance-weighted sampling over the pool teams by their
                    self-play win-rate (weights pushed from TeamPFSPCallback). The
                    teambuilder only ACCUMULATES per-team win/loss counts (drained by the
                    callback) and STORES the pushed weights; the callback computes them.
        team_pfsp_cap / team_pfsp_floor: the callback's cap / floor (stored here so the
                    construction site is the single knob source; unused by the sampler itself).
        """
        if isinstance(teams, str):
            teams = [teams]

        # STRICT VALIDATION: Ensure all teams are legal for Gen 3 OU in one batch
        from utils.bridge.team_validator import validate_teams_locally

        validations = validate_teams_locally("gen3ou", teams)

        valid_indices = []
        for i, res in enumerate(validations):
            if res.get("valid"):
                valid_indices.append(i)
            else:
                errors = ", ".join(res.get("errors", ["Unknown error"]))
                # We skip illegal teams but don't crash unless the whole pool is bad.

        if not valid_indices:
            raise ValueError(f"No valid teams found in the provided list! (First error: {validations[0].get('errors')})")

        self.packed_teams = []
        # Per-pool-team fingerprint, PARALLEL to packed_teams (index i ↔ packed_teams[i]).
        # sha1(team_str.strip())[:10] == team_archetypes.team_sha (inlined to avoid a utils→agents
        # import, matching matchup_spec's practice) so a key JOINS the archetype labels + every
        # provenance record — and lets the team-PFSP callback verify the per-INDEX team IDENTITY is
        # the same across all workers (a strong GIGO guard: same pool SIZE ≠ same pool ORDER).
        self._pool_keys = []
        for idx in valid_indices:
            team_str = teams[idx]
            # Parse and Pack
            parsed_team = self.parse_showdown_team(team_str)
            fixed_team = fix_gen3_hp_ivs(parsed_team)
            self.packed_teams.append(self.join_team(fixed_team))
            self._pool_keys.append(hashlib.sha1(team_str.strip().encode()).hexdigest()[:10])

        self.bias_prob = bias_prob
        self.bias_packed_teams = []
        if bias_teams and bias_prob > 0.0:
            if isinstance(bias_teams, str):
                bias_teams = [bias_teams]
            bias_validations = validate_teams_locally("gen3ou", bias_teams)
            for i, res in enumerate(bias_validations):
                if res.get("valid"):
                    parsed = self.parse_showdown_team(bias_teams[i])
                    fixed = fix_gen3_hp_ivs(parsed)
                    self.bias_packed_teams.append(self.join_team(fixed))

        # ── Team-side PFSP state (all inert while team_pfsp == "off") ──
        self._team_pfsp = str(team_pfsp)
        self._team_pfsp_cap = float(team_pfsp_cap)
        self._team_pfsp_floor = float(team_pfsp_floor)
        # LOCAL windowed accumulators (per pool team) — drained (and zeroed) each pull.
        self._tp_wins = [0.0] * len(self.packed_teams)
        self._tp_games = [0.0] * len(self.packed_teams)
        # Weights pushed from the callback (None → uniform sampling).
        self._tp_weights = None
        # The pool index yielded for the CURRENT battle (None → not a tracked pool battle).
        self._last_pool_idx = None
        # ── Team-blocked episodes (--team-block-episodes; 1 = off, byte-identical) ──
        # Hold each drawn team for K consecutive yields before redrawing. Rationale (the FiLM
        # sample-starvation counter): per-episode redraw sprays ~700 teams at ~4 episodes each per
        # rollout, so the per-team residual gradient in any update is estimated from ~140 decisions;
        # at K≈64 (≈ one rollout of episodes) each env carries ONE team per rollout at ~2k decisions
        # (~15× density) AND the block spans an update boundary — the env replays the same team right
        # after that team's gradient landed (the mini-exploiter learn-and-retest loop). The WHOLE
        # draw procedure is held (bias branch, PFSP weights, tracking index), so every sampling
        # feature composes: weights apply at redraw, outcomes attribute to the blocked team all K
        # episodes, a bias-drawn block stays untracked. Each env worker unpickles its OWN builder
        # copy, so blocks are per-env by construction.
        self._block_episodes = 1
        self._block_cached = None       # packed team held for the current block
        self._block_cached_idx = None   # its pool index (None = untracked bias draw)
        self._block_left = 0            # remaining yields before redraw

    def set_block_episodes(self, k: int) -> None:
        """Set the team-block length (K consecutive episodes per draw; 1 = off). Resets any
        in-flight block so a mid-run push takes effect at the next yield."""
        self._block_episodes = max(1, int(k))
        self._block_cached = None
        self._block_left = 0

    def yield_team(self):
        """Returns a team for the next battle. With ``_block_episodes > 1`` the last drawn team is
        held for K consecutive yields (see __init__); otherwise (default) every yield is a fresh
        draw — the exact legacy behavior, no extra RNG."""
        if self._block_episodes > 1:
            if self._block_left > 0 and self._block_cached is not None:
                self._block_left -= 1
                self._last_pool_idx = self._block_cached_idx   # outcomes keep attributing to the block's team
                return self._block_cached
            team = self._draw_team()
            self._block_cached = team
            self._block_cached_idx = self._last_pool_idx
            self._block_left = self._block_episodes - 1
            return team
        return self._draw_team()

    def _draw_team(self):
        """One fresh draw from the pool, biased toward bias_packed_teams if configured.

        With team_pfsp in a biasing mode the pool draw is weighted (per-team win-rate) and the
        yielded index is tracked so the battle outcome can be recorded. A bias-team battle is
        never tracked (it uses a pinned distill/bias team, not a pool team). "off" is the exact
        legacy uniform ``random.choice`` — no extra RNG draws, no tracking (byte-identical)."""
        if self.bias_packed_teams and random.random() < self.bias_prob:
            self._last_pool_idx = None
            return random.choice(self.bias_packed_teams)
        if self._team_pfsp == "off":
            return random.choice(self.packed_teams)
        n = len(self.packed_teams)
        w = self._tp_weights if self._tp_weights is not None else [1.0] * n
        idx = random.choices(range(n), weights=w, k=1)[0]
        self._last_pool_idx = idx
        return self.packed_teams[idx]

    def record_team_pfsp_outcome(self, won: float) -> None:
        """Record a battle outcome (win=1.0 / loss-or-tie=0.0) against the LAST yielded pool team.
        No-op when off or when the last yield was a bias team / untracked (``_last_pool_idx`` None)."""
        if self._team_pfsp == "off" or self._last_pool_idx is None:
            return
        idx = self._last_pool_idx
        self._tp_games[idx] += 1.0
        self._tp_wins[idx] += float(won)

    def drain_team_pfsp_counts(self):
        """Return ``(wins, games, n_pool)`` for this window and ZERO the local accumulators (each
        pull is one window). ``n_pool`` lets the callback GIGO-guard pool-size agreement across
        workers (a mismatch would silently corrupt the per-index win-rate contract)."""
        wins = list(self._tp_wins)
        games = list(self._tp_games)
        n = len(self.packed_teams)
        for i in range(n):
            self._tp_wins[i] = 0.0
            self._tp_games[i] = 0.0
        return (wins, games, n)

    def set_team_pfsp_weights(self, weights) -> None:
        """Store the sampling weights pushed by the callback. Guarded: a None or wrong-length
        vector is ignored (leaves the current weights / uniform), never partially applied."""
        if weights is None or len(weights) != len(self.packed_teams):
            return
        self._tp_weights = [max(0.0, float(x)) for x in weights]

    def get_team_pfsp_keys(self):
        """The per-pool-team fingerprints (index i ↔ packed_teams[i]). The callback pulls these
        ONCE to verify the per-index team identity is identical across all workers (the strong GIGO
        guard) and to key the audit log of which teams are up/down-weighted."""
        return list(self._pool_keys)
