import random

from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

# Stable cross-run opponents are a CAPPED minority of the self-play (challenge) bucket — the pool
# keeps the bulk, so no single fixed opponent can dominate training. Multiple un-mastered stable
# opponents SHARE this slice (so the total stable share stays ≤ this regardless of count). Mastered
# ones leave the challenge bucket entirely for the floor.
STABLE_CHALLENGE_SHARE = 0.20


class MaskableAgentWrapper(SingleAgentWrapper):
    """
    Bridges poke-env's SingleAgentWrapper to the MaskablePPO interface, and selects the
    episode's opponent at reset() from a LIVE self-play fraction.

    Five responsibilities:

    1. action_masks() — MaskablePPO calls this each step to retrieve the valid action mask.
       SingleAgentWrapper doesn't implement it, so we delegate to the inner Gen3Env.

    2. Wait-turn absorption — poke-env returns a step whenever a server round-trip completes,
       even when agent1 had no decision (e.g. the opponent is choosing a forced switch). Those
       ghost steps would pollute the rollout buffer, so we absorb them (loop until agent1 has a
       pending decision, accumulating reward).

    3. Per-episode opponent selection (the self-play curriculum). The wrapper holds the heuristic
       bot roster + one reusable pool ``RLPlayer`` + a ``SnapshotPool`` handle. At reset(), with
       probability ``self_play_fraction`` it plays the pool, else a random heuristic — the coin
       flip is per-episode so the live fraction is honored exactly. The pool *snapshot* itself is
       (re)sampled+loaded only once per generation (see ``_select_episode_opponent`` — loading is
       a ~27MB deserialize, too expensive to do per-episode), held in the reusable pool player.
       The opponent is a pure decision function over ``env.battle2`` (env.agent1/agent2 do all the
       networking), so swapping it between episodes is free and safe — the in-episode
       stale-decision path (SingleAgentWrapper.step) is untouched.

    4. Live curriculum updates — ``set_self_play_target(fraction, generation)`` is pushed by the
       eval callback via ``VecEnv.env_method`` after every eval, so the heuristic-vs-pool ratio
       tracks the model's measured strength *during* a run (no restart needed). A new generation
       triggers a pool re-scan to pick up freshly seeded/promoted snapshots.

    5. Opponent default-stat telemetry — read from the persistent pool player (not the rotating
       ``self.opponent``), so the counts aggregate across all self-play episodes.
    """

    def __init__(self, env, opponent=None, *, heuristic_opponents=None, pool=None,
                 pool_player=None, self_play_fraction=0.0, rng_seed=0,
                 heuristic_weights=None, stable_players=None, stable_labels=None,
                 stable_challenge_share=STABLE_CHALLENGE_SHARE, exploiter_player=None,
                 exploiter_keep_bots=False, exploiter_bot_fraction=0.5,
                 stable_teams=None, exploiter_team=None, opponent_pool_team=None,
                 stable_pfsp=False, team_wr_tracking=True, exploiter_rung_loader=None):
        # Back-compat: a single positional `opponent` (legacy / tests) becomes a 1-bot roster.
        roster = list(heuristic_opponents) if heuristic_opponents else (
            [opponent] if opponent is not None else [])
        if not roster:
            raise ValueError(
                "MaskableAgentWrapper needs an `opponent` or a `heuristic_opponents` roster")
        super().__init__(env, roster[0])
        self._heuristic_opponents = roster
        # Per-bot sampling weights aligned index-for-index to `roster`, biasing WHICH heuristic
        # an episode draws (e.g. the coverage-punishing aggressive_v2/heuristic2). None → uniform
        # (the original behavior, byte-for-byte). Validated at construction so a misaligned vector
        # fails loudly here, never silently per-episode. Applies only to the heuristic branch of
        # _select_episode_opponent; the pool-vs-heuristic fraction and pool sampling are untouched.
        if heuristic_weights is None:
            self._heuristic_weights = None
        else:
            w = [float(x) for x in heuristic_weights]
            if len(w) != len(roster):
                raise ValueError(
                    f"heuristic_weights len {len(w)} != heuristic roster len {len(roster)}")
            if any(x < 0 for x in w) or sum(w) <= 0:
                raise ValueError("heuristic_weights must be non-negative with a positive sum")
            self._heuristic_weights = w
        self._pool = pool
        self._pool_player = pool_player
        # Stable cross-run opponents (already-loaded RLPlayers + their labels): UN-mastered ones are
        # peers of the pool in the CHALLENGE bucket ("tossed in like another sentinel"); once the
        # trainee MASTERS one (pushed via set_stable_mastered) it joins the FLOOR bucket alongside
        # the bots ("becomes another bot"). Built once per worker, so no per-episode reload.
        self._stable_players = list(stable_players) if stable_players else []
        self._stable_labels = list(stable_labels) if stable_labels else []
        self._stable_mastered = {lab: False for lab in self._stable_labels}
        self._stable_challenge_share = float(stable_challenge_share)
        # DYNAMIC stable-opponent selection (--stable-opponent-pfsp): when the trainee faces the
        # un-mastered stable opponents (the capped challenge slice), pick WEIGHTED by how much it is
        # LOSING to each (weight = 1 − win_rate) instead of uniformly — so a stalled generalist
        # spends its exploiter budget on the axis it's failing WORST, and each opponent naturally
        # fades from the slice as it's mastered (win_rate→1 ⇒ weight→0), then the mastery flip moves
        # it to the floor. The TOTAL pool-vs-stable share is unchanged (still `stable_challenge_share`),
        # so the opponent-mix reporting/telemetry is unaffected — only WHICH un-mastered stable
        # opponent gets picked shifts. Win-rates arrive via `set_stable_win_rates` each eval. OFF
        # (default) → uniform `_rng.choice`, byte-identical to the flat-share behavior.
        self._stable_pfsp = bool(stable_pfsp)
        # --team-wr-tracking (default ON): pure instrumentation, so the gate is here at the caller
        # rather than threaded into every Gen3Teambuilder construction site.
        self._team_wr_tracking = bool(team_wr_tracking)
        self._stable_win_rates: dict = {}   # {label: p} EMA-smoothed, pushed each eval when pfsp on
        # EXPLOITER mode (--exploiter): a single fixed foreign model that is the SOLE training
        # opponent every episode — it short-circuits the whole challenge/floor/pool/stable selection
        # below. Used to train a dedicated agent that just learns to beat one target (the league
        # "exploiter" role). None → normal self-play/bot/stable selection (byte-identical).
        self._exploiter_player = exploiter_player
        # EXPLOITER + keep-bots: instead of the target being the SOLE opponent, mix the heuristic
        # bots (the always-on floor roster) back in — per episode, face the target with prob
        # (1 - exploiter_bot_fraction), else a floor/heuristic bot via _pick_floor_opponent. Lets a
        # from-scratch specialist keep a bot floor while learning to beat one strong target. Only
        # consulted when _exploiter_player is set; off → the sole-target path (byte-identical).
        self._exploiter_keep_bots = bool(exploiter_keep_bots)
        self._exploiter_bot_fraction = float(exploiter_bot_fraction)
        # PER-OPPONENT PINNED TEAMS (the league fold-back contract): a specialist stable/exploiter
        # opponent pilots ITS OWN pinned team, everyone else the shared pool. The opponent Players
        # are pure decision-functions — env.agent2 does the networking, so agent2._team decides the
        # opponent's REAL team (the training-mirror lesson) and must be switched PER EPISODE to
        # match the selected opponent. `stable_teams` is a per-stable-opponent parallel list of
        # Gen3Teambuilder|None; `exploiter_team` the exploiter target's pin (or None);
        # `opponent_pool_team` the default/pool builder to restore on unpinned episodes (the SAME
        # instance Gen3Env(opponent_team=) got, so team-draw RNG sequences are unchanged). With NO
        # pinned team anywhere the whole feature is INERT — reset() never touches agent2._team
        # (byte-identical to the pre-feature wrapper).
        if stable_teams is not None and len(stable_teams) != len(self._stable_players):
            raise ValueError(
                f"stable_teams len {len(stable_teams)} != stable_players len "
                f"{len(self._stable_players)}")
        self._team_by_opponent = {}
        for p, tb in zip(self._stable_players, stable_teams or []):
            if tb is not None:
                self._team_by_opponent[id(p)] = tb
        if exploiter_player is not None and exploiter_team is not None:
            self._team_by_opponent[id(exploiter_player)] = exploiter_team
        self._opponent_pool_team = opponent_pool_team
        self._per_opponent_teams = bool(self._team_by_opponent)
        if self._per_opponent_teams:
            if opponent_pool_team is None:
                raise ValueError(
                    "per-opponent pinned teams need opponent_pool_team (the default builder to "
                    "restore on unpinned episodes)")
            if getattr(env, "agent2", None) is None:
                raise ValueError(
                    "per-opponent pinned teams need an env with .agent2 (the opponent-side "
                    "networking player whose _team decides the opponent's real team)")
        # gen3_exploiter_temp_anneal_v1 (ratchet mode): cumulative episode outcomes vs the EXPLOITER
        # target (bot episodes excluded), read via env_method("exploiter_winrate_totals") by
        # ExploiterTempRatchetCallback to measure the trainee's live WR at the CURRENT opponent
        # temperature — the signal the fixed schedule lacks. Inert unless an exploiter is set.
        self._exploiter_games = 0
        self._exploiter_wins = 0.0
        # gen3_exploiter_pool_ladder_v1 (--exploiter-ladder): the target's WEIGHTS are swapped up a
        # ladder of frozen opponents as the trainee clears a win-rate gate. `exploiter_rung_loader`
        # is the worker-side `(zip_path, config_path) -> model` closure (it owns the arch gate,
        # device and opponent-compile settings; the wrapper stays free of model-loading policy) and
        # is None off the ladder path, which makes every branch below inert. The swap is DEFERRED to
        # the next `reset()` so an episode is never scored against two different opponents, and the
        # per-rung counters below reset with it — that is what makes the gate's win-rate a
        # per-rung quantity rather than a pooled one.
        self._exploiter_rung_loader = exploiter_rung_loader
        self._exploiter_rung_index = None    # None until the first push (those episodes count for no rung)
        self._exploiter_rung_games = 0
        self._exploiter_rung_wins = 0.0
        self._pending_exploiter_rung = None  # (index, zip_path, config_path)
        self._self_play_fraction = float(self_play_fraction)
        self._target_generation = 0
        self._scanned_generation = -1   # -1 forces a pool re-scan on the first pool selection
        self._has_pool_model = False    # a snapshot is loaded into the pool player (this gen)
        self._rng = random.Random(rng_seed)  # per-env seed → envs don't pick in lockstep

    def set_self_play_target(self, fraction: float, generation: int) -> None:
        """Live curriculum update (called via ``VecEnv.env_method`` after each eval): the
        per-episode probability of facing a pool opponent, and a generation counter whose change
        triggers a pool re-scan (to pick up newly seeded/promoted snapshots)."""
        self._self_play_fraction = float(fraction)
        self._target_generation = int(generation)

    def set_exploiter_temperature(self, temperature: float) -> None:
        """Live update of the EXPLOITER target opponent's sampling temperature (pushed via
        ``VecEnv.env_method`` each rollout by ``ExploiterTempAnnealCallback``, the
        ``set_self_play_target`` idiom). Sets it on the persistent exploiter ``RLPlayer`` so the
        next decision samples at the annealed temp (``RLPlayer.choose_move`` reads
        ``self._temperature`` fresh each call). No-op when this env has no exploiter player — a
        non-exploiter run never pushes it (byte-identical)."""
        if self._exploiter_player is not None:
            self._exploiter_player._temperature = float(temperature)

    def set_exploiter_rung(self, index: int, zip_path: str, config_path: str) -> None:
        """Live update of WHICH frozen opponent the exploiter target is (pushed via
        ``VecEnv.env_method`` by ``ExploiterLadderCallback`` on each promotion — the
        ``set_self_play_target`` idiom).

        The load is DEFERRED to the next ``reset()`` rather than done here, for two reasons: the
        episode in flight must not have its opponent's brain replaced mid-battle, and its outcome
        must be attributed to the rung it was actually played against. No-op when this env has no
        exploiter player or no rung loader (i.e. every non-ladder run — byte-identical)."""
        if self._exploiter_player is None or self._exploiter_rung_loader is None:
            return
        self._pending_exploiter_rung = (int(index), str(zip_path), str(config_path))

    def _apply_pending_exploiter_rung(self) -> None:
        """Swap a promoted rung's weights into the persistent exploiter player, between episodes.

        Mirrors ``_ensure_pool_model``'s "load once, assign to the persistent player" mechanism (a
        promotion happens a handful of times per run, so the ~27 MB load is far rarer than the
        once-per-generation pool reload). The team pin, temperature and bot fraction are deliberately
        NOT touched: the ladder varies opponent STRENGTH and nothing else."""
        if self._pending_exploiter_rung is None:
            return
        idx, zip_path, config_path = self._pending_exploiter_rung
        self._pending_exploiter_rung = None
        if idx == self._exploiter_rung_index:
            return
        self._exploiter_player.model = self._exploiter_rung_loader(zip_path, config_path)
        self._exploiter_rung_index = idx
        self._exploiter_rung_games = 0     # a new rung is a new measurement — never pool across rungs
        self._exploiter_rung_wins = 0.0

    def exploiter_rung_totals(self):
        """``(rung_index, games, wins)`` vs the CURRENT ladder rung — read via ``VecEnv.env_method``
        by ``ExploiterLadderCallback``, which drops rows whose index isn't the live one (a worker
        that hasn't reset since the push). ``(None, 0, 0.0)`` off the ladder path."""
        return (self._exploiter_rung_index, self._exploiter_rung_games, self._exploiter_rung_wins)

    def _record_exploiter_outcome(self, won: float) -> None:
        """Count an episode outcome vs the EXPLOITER target (ratchet-mode WR signal). Only counts
        when THIS episode's opponent was the exploiter target (bot episodes under
        ``--exploiter-keep-bots`` are excluded), so the WR measures difficulty vs the target itself."""
        if self._exploiter_player is not None and self.opponent is self._exploiter_player:
            self._exploiter_games += 1
            self._exploiter_wins += float(won)
            # …and against the LIVE LADDER RUNG, when a ladder is running. `_exploiter_rung_index`
            # is None before the first push, so the handful of episodes in flight at training start
            # are counted for no rung at all rather than mis-attributed to rung 0.
            if self._exploiter_rung_index is not None:
                self._exploiter_rung_games += 1
                self._exploiter_rung_wins += float(won)

    def _maybe_record_team_pfsp(self, won: float) -> None:
        """Record the trainee's team outcome for ``--team-pfsp``, but ONLY on SELF-PLAY POOL battles
        (``opponent is _pool_player``) OR EXPLOITER-TARGET battles (``opponent is _exploiter_player``)
        — bots are excluded either way (we win ~0.99 vs bots, which washes out the per-team variance
        signal). In EXPLOITER mode this is what lets ``--team-pfsp`` weight a multi-team
        (``--trainee-teams``/pin_multi) exploiter toward the teams it is LOSING with vs the target
        (concentrate the per-team budget on the laggards). No-op off / vs bots / on a bias-team yield
        (the builder's own guards); ``_exploiter_player`` is None off the exploiter path, so a self-play
        run is byte-identical. Aggregated centrally by ``TeamPFSPCallback``."""
        if self.opponent is self._pool_player or (
                self._exploiter_player is not None and self.opponent is self._exploiter_player):
            _tb = self._trainee_teambuilder()
            if _tb is not None and hasattr(_tb, "record_team_pfsp_outcome"):
                _tb.record_team_pfsp_outcome(won)

    def _maybe_record_team_wr(self, won: float) -> None:
        """Record the trainee's team outcome for ``--team-wr-tracking`` (pure instrumentation).

        Deliberately UNLIKE ``_maybe_record_team_pfsp`` above in both of its restrictions: every
        opponent class counts (stratified by ``_opponent_class``, so a bot-heavy curriculum phase
        never reads as a per-team win rate), and it is on by default. The two share only the
        builder's "which team did I just yield" index — never a counter table. No-op on a
        bias/distill yield (the builder's own guard) or a non-Gen3 builder."""
        if not self._team_wr_tracking:
            return
        _tb = self._trainee_teambuilder()
        if _tb is not None and hasattr(_tb, "record_team_wr_outcome"):
            _tb.record_team_wr_outcome(won, getattr(self, "_opponent_class", self.OPP_CLASS_BOT),
                                       self.N_OPP_CLASSES)

    def exploiter_winrate_totals(self):
        """(cumulative target-games, cumulative target-wins) this worker has played vs the EXPLOITER
        target — read via ``VecEnv.env_method`` by ``ExploiterTempRatchetCallback``, which diffs to a
        per-window WR and ratchets the target temperature. ``(0, 0.0)`` when there's no exploiter."""
        return (self._exploiter_games, self._exploiter_wins)

    def set_opponent_win_rates(self, rates) -> None:
        """Live PFSP update (pushed via ``VecEnv.env_method`` each eval): the trainee's per-snapshot
        win-rates (``{step: P(win)}``, EMA-smoothed in the callback) that weight pool sampling toward
        the selves we're losing to. Forwarded to the worker's ``SnapshotPool``; a no-op on sampling
        unless the pool was built with ``pfsp_scale > 0``. Pushed only when PFSP is on, so an off run
        never even makes this IPC call (byte-identical)."""
        if self._pool is not None:
            self._pool.set_win_rates(rates)

    def _trainee_teambuilder(self):
        """The trainee's teambuilder handle (poke-env stores ``team=`` as ``_team``; agent1 is the
        trainee). Guarded — returns None if the env/agent1/_team chain is absent."""
        return getattr(getattr(self.env, "agent1", None), "_team", None)

    def drain_team_wr_counts(self) -> "tuple | None":
        """Per-team win-rate pull (read via ``VecEnv.env_method`` by ``TeamWinRateCallback`` each
        window): drain-and-zero this worker's per-team, per-opponent-class win/game accumulators.

        ``None`` when tracking is off or the trainee builder isn't tracking-capable. **This is an
        `env_method` PULL, not an info-dict thread** — that is what makes it correct under
        ``--async-rollout`` as well as the sync barrier (see ``TeamWinRateCallback``)."""
        if not self._team_wr_tracking:
            return None
        _tb = self._trainee_teambuilder()
        if _tb is None or not hasattr(_tb, "drain_team_wr_counts"):
            return None
        drained: "tuple | None" = _tb.drain_team_wr_counts()
        return drained

    def drain_reward_terms(self):
        """`reward/` per-term pull (``VecEnv.env_method`` from ``RewardTermMetricsCallback`` once
        per rollout): drain-and-zero this worker's per-term reward sums.

        An `env_method` PULL rather than an info-dict thread, for the reason the team-WR tracker
        gives: the wave batching of ``--async-rollout`` cannot recover which BUFFER ROW a step
        landed on, and `AsyncSubprocVecEnv.env_method` is drain-safe, so one seam covers both
        collectors. ``None`` when the env has no Gen3 reward manager (a non-Gen3 env, or the
        `RewardTracker`'s shadow twin)."""
        rm = getattr(self.env, "reward_manager", None)
        drain = getattr(rm, "drain_reward_terms", None)
        return drain() if drain is not None else None

    def drain_team_pfsp_counts(self):
        """Team-side PFSP pull (read via ``VecEnv.env_method`` by ``TeamPFSPCallback`` each window):
        drain-and-zero this worker's per-team self-play win/loss accumulators. ``None`` when the
        trainee builder isn't PFSP-capable (off runs / a non-Gen3 builder)."""
        _tb = self._trainee_teambuilder()
        return _tb.drain_team_pfsp_counts() if (
            _tb is not None and hasattr(_tb, "drain_team_pfsp_counts")) else None

    def set_team_pfsp_weights(self, weights) -> None:
        """Team-side PFSP push (``VecEnv.env_method`` from ``TeamPFSPCallback``): set the trainee
        builder's per-team sampling weights. No-op when the builder isn't PFSP-capable."""
        _tb = self._trainee_teambuilder()
        if _tb is not None and hasattr(_tb, "set_team_pfsp_weights"):
            _tb.set_team_pfsp_weights(weights)

    def get_team_pfsp_keys(self):
        """Team-side PFSP identity pull (``VecEnv.env_method``, ONCE): this worker's per-pool-team
        fingerprints, so the callback can verify the per-index team identity matches across workers.
        ``None`` when the trainee builder isn't PFSP-capable."""
        _tb = self._trainee_teambuilder()
        return _tb.get_team_pfsp_keys() if (
            _tb is not None and hasattr(_tb, "get_team_pfsp_keys")) else None

    def set_stable_mastered(self, mastered_labels) -> None:
        """Pushed by the eval callback each cycle (via ``env_method``): the labels of stable
        cross-run opponents the trainee has MASTERED (win_rate ≥ threshold). A mastered opponent
        moves from the CHALLENGE bucket (peer of the pool) to the FLOOR bucket (peer of the bots) —
        it "becomes another bot", kept for coverage but no longer a thing to master. The callback's
        set only grows, so this is monotonic (no flapping on eval noise) and resume-safe (recomputed
        from eval each cycle — no stored env state)."""
        s = set(mastered_labels or ())
        self._stable_mastered = {lab: (lab in s) for lab in self._stable_labels}

    def set_stable_win_rates(self, rates) -> None:
        """Pushed by the eval callback each cycle (via ``env_method``) when ``--stable-opponent-pfsp``
        is on: the trainee's per-stable-opponent win-rate ``{label: p}`` (the same ``win_rate_vs_ext_
        <label>`` the eval already computes, EMA-smoothed). Weights the un-mastered-stable pick toward
        the opponents the trainee is LOSING to most (``_pick_stable``). Mirrors ``set_opponent_win_
        rates`` (the pool PFSP push); a no-op on selection unless ``stable_pfsp`` is set."""
        self._stable_win_rates = dict(rates or {})

    def _ensure_pool_model(self) -> bool:
        """Load/refresh the per-generation pool snapshot into the reusable pool player; return True
        if a pool model is ready.

        The snapshot is (re)sampled+loaded ONLY when the trainer signals a new generation (a seed or
        promotion — i.e. every eval), or the first time a model is needed — NOT per episode.
        ``load_model`` deserializes a ~27MB MaskablePPO; doing it per-episode against an N-deep pool
        with a small LRU thrashed the workers (they block in ``reset()`` on disk I/O → CPU ~40%, FPS
        ~1400→~500). Loading once per generation restores the "load once, reuse" throughput while
        keeping diversity: the envs sample independently and every env rotates to a fresh sample each
        generation."""
        if self._pool is None or self._pool_player is None:
            return False
        if self._target_generation != self._scanned_generation or not self._has_pool_model:
            self._pool._scan()
            if not self._pool.is_empty():
                entry = self._pool.sample()
                self._pool_player.model = self._pool.load_model(entry)  # LRU-cached
                self._has_pool_model = True
                self._scanned_generation = self._target_generation
        return self._has_pool_model

    def _stable_in(self, mastered: bool) -> list:
        """The stable-opponent players whose mastery state matches ``mastered``."""
        return [p for p, lab in zip(self._stable_players, self._stable_labels)
                if self._stable_mastered.get(lab, False) == mastered]

    def _pick_floor_opponent(self):
        """The always-on coverage bucket: the heuristic bot roster + any MASTERED stable opponents
        (each weighted like an unlisted bot, 1.0). Honors ``--bot-weights`` for the bots."""
        mastered = self._stable_in(mastered=True)
        candidates = self._heuristic_opponents + mastered
        # This floor weighting (Σ bot weights, +1.0 per mastered stable) — like the whole selection
        # split here — is mirrored for REPORTING by SelfPlayCallback._opponent_mix_fractions; keep
        # the two in sync (pinned by wrappers_test.py::test_mix_fractions_match_actual_sampling).
        if self._heuristic_weights is None:
            return self._rng.choice(candidates)
        weights = self._heuristic_weights + [1.0] * len(mastered)
        return self._rng.choices(candidates, weights=weights, k=1)[0]

    def _pick_stable(self, stable):
        """Pick one un-mastered stable opponent. With ``--stable-opponent-pfsp`` (and win-rates
        pushed), weight by how much the trainee is LOSING to each (``1 − win_rate``, floored so an
        unmeasured / just-mastered opponent still gets a small share) — spend the exploiter budget on
        the axis we're failing worst. OFF → uniform ``_rng.choice`` (byte-identical). Aligning by the
        opponent's LABEL (via ``_stable_labels``) so the win-rate map keys match the player."""
        if not self._stable_pfsp or not self._stable_win_rates or len(stable) == 1:
            return self._rng.choice(stable)
        label_of = {id(p): lab for p, lab in zip(self._stable_players, self._stable_labels)}
        weights = [max(0.05, 1.0 - float(self._stable_win_rates.get(label_of.get(id(p)), 0.5)))
                   for p in stable]
        return self._rng.choices(stable, weights=weights, k=1)[0]

    def _pick_challenge_opponent(self):
        """The CHALLENGE bucket — what the model is actively trying to master. The self-play pool
        gets the BULK; any UN-mastered stable cross-run opponents share a CAPPED minority slice
        (``_stable_challenge_share``, default 20%), so a single fixed opponent can never dominate
        training. WHICH un-mastered stable opponent is picked is loss-weighted under
        ``--stable-opponent-pfsp`` (``_pick_stable``); the total pool-vs-stable share is unchanged.
        Returns ``None`` if neither has a ready opponent (→ fall through to the floor)."""
        stable = self._stable_in(mastered=False)
        pool_ready = self._ensure_pool_model()
        if stable and pool_ready:
            if self._rng.random() < self._stable_challenge_share:
                return self._pick_stable(stable)      # the capped stable slice (loss-weighted)
            return self._pool_player                  # the bulk → the pool
        if pool_ready:
            return self._pool_player
        if stable:
            return self._pick_stable(stable)          # no pool seeded yet → stable IS the challenge
        return None

    # gen3_opp_class_v1 — WHICH KIND of opponent this episode faces, tagged at the one place that
    # decides it. The opponent-intent metrics (`opp_intent/*`) pool over an opponent mix spanning a
    # RANDOM bot, several heuristics and frozen selves, and "predict their move" means something
    # different in each: against random it is impossible BY CONSTRUCTION (and worthless if achieved),
    # against a heuristic it is easy but models a decision tree rather than a player, and only
    # against the pool does it measure the thing we care about. One pooled accuracy averages over
    # all three, so it cannot be read. Tagging the row is what lets the metric be split.
    OPP_CLASS_BOT = 0          # heuristic / random floor bots — near-deterministic or unpredictable
    OPP_CLASS_POOL = 1         # frozen selves — the distribution that actually matters
    OPP_CLASS_STABLE = 2       # cross-run stable opponents
    OPP_CLASS_EXPLOITER = 3    # the exploiter's target
    N_OPP_CLASSES = 4          # width of any per-class array keyed on the four above

    def _select_episode_opponent(self) -> None:
        """Pick this episode's opponent.

        CHALLENGE bucket (competence-gated by ``self_play_fraction``, ``_pick_challenge_opponent``) =
        the self-play pool (the bulk) + a capped minority of un-mastered stable opponents. FLOOR
        bucket (always-on coverage) = the heuristic bots + any MASTERED stable opponents. The
        per-episode coin flip honors the live ``self_play_fraction`` exactly; the pool snapshot is
        loaded once per generation (``_ensure_pool_model``), not per episode."""
        if self._exploiter_player is not None:
            if self._exploiter_keep_bots and self._rng.random() < self._exploiter_bot_fraction:
                # keep-bots: a per-episode slice faces a floor/heuristic bot instead of the target,
                # so a from-scratch specialist keeps a bot floor while learning to beat the target.
                self.opponent = self._pick_floor_opponent()
                self._opponent_class = self.OPP_CLASS_BOT
            else:
                self.opponent = self._exploiter_player   # exploiter target (sole-opponent default)
                self._opponent_class = self.OPP_CLASS_EXPLOITER
            return
        if self._rng.random() < self._self_play_fraction:
            opp = self._pick_challenge_opponent()
            if opp is not None:
                self.opponent = opp
                # The challenge bucket is pool + un-mastered STABLE; distinguish them so a stable
                # opponent's rows are not silently counted as self-play.
                self._opponent_class = (self.OPP_CLASS_STABLE
                                        if opp in self._stable_players
                                        else self.OPP_CLASS_POOL)
                return
        # Floor bucket — also the fallthrough when the challenge bucket has no ready opponent.
        self.opponent = self._pick_floor_opponent()
        self._opponent_class = self.OPP_CLASS_BOT

    def _apply_opponent_team(self) -> None:
        """The opponent's REAL team follows the selected opponent (fold-back): its own pin when it
        has one, else the shared pool builder. env.agent2 does the opponent-side networking, so its
        ``_team`` decides what the opponent actually brings (the training-mirror lesson) — switched
        per episode, BEFORE the next battle starts (the team is submitted at challenge time). Inert
        (agent2 never touched) when no opponent carries a pin."""
        if self._per_opponent_teams:
            self.env.agent2._team = self._team_by_opponent.get(
                id(self.opponent), self._opponent_pool_team)

    def reset(self, *, seed=None, options=None):
        # A promoted ladder rung lands BETWEEN episodes (see `set_exploiter_rung`). Inert — one
        # `is None` test — on every run that isn't using --exploiter-ladder.
        self._apply_pending_exploiter_rung()
        self._select_episode_opponent()
        self._apply_opponent_team()
        # Push the class down for the intent metrics. Same direction as `_apply_opponent_team`
        # (the wrapper reaches into the env, which owns the obs) and set at the same point in the
        # episode, so the label can never describe the PREVIOUS opponent.
        setattr(self.env, "_opponent_class", getattr(self, "_opponent_class", 0))
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        while not self.env.agent1_to_move and not term and not trunc:
            obs, r, term, trunc, info = super().step(0)
            reward += r
        if term or trunc:
            # Expose the battle OUTCOME for the win-probability label plumbing (win=1 / loss-or-tie=0).
            # The trainee's battle is finished here (before the VecEnv auto-resets), so battle1.won is
            # set. Consumed ONLY when the win-prob head is on (WinProbLabelCallback / the async
            # collector); harmless otherwise. A tie (won is None) counts as not-a-win → 0.0.
            b = getattr(self.env, "battle1", None)
            won = 1.0 if (b is not None and b.won is True) else 0.0
            info["win_outcome"] = won
            # +SIGNAL (gen3_signal_rate_metrics_v1): tag the outcome with the opponent CLASS so the
            # `signal/outcome_entropy` meters can be split by kind (target / pool / stable / bots).
            # A pooled outcome entropy averages over a mix whose members mean different things — the
            # same reason `_opponent_class` already exists for the intent metrics. Purely additive to
            # the info dict (SB3 reads only `episode`/`terminal_observation`/`TimeLimit.truncated`),
            # so nothing downstream changes; the class is the one selected for THIS episode at reset.
            info["opponent_class"] = int(getattr(self, "_opponent_class", self.OPP_CLASS_BOT))
            self._record_exploiter_outcome(won)   # ratchet-mode WR signal (no-op off / vs bots)
            self._maybe_record_team_pfsp(won)     # team-side PFSP per-team WR (pool + exploiter-target)
            self._maybe_record_team_wr(won)       # per-team win-rate tracking (all classes, default ON)
        return obs, reward, term, trunc, info

    def action_masks(self):
        return self.env.action_masks()

    def opponent_default_stats(self):
        """(#decisions, #defaults, #re-decides) for the self-play POOL opponent — read from the
        persistent pool player (NOT ``self.opponent``, which rotates per episode), so the counts
        aggregate across all self-play episodes. Heuristic-only envs report zeros. #re-decides
        counts stale-decision races RESOLVED by re-deciding on the now-current request."""
        op = self._pool_player
        if op is None:
            return (0, 0, 0)
        return (
            getattr(op, "_n_decisions", 0),
            getattr(op, "_n_defaults", 0),
            getattr(op, "_n_redecides", 0),
        )
