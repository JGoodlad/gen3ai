"""The COUNTERFACTUAL tier — `falsify`, `lookahead`, `better_line`, `replay_counterfactual`.

All four need the trace's `*_reconstruction.json` sibling (bridge-eval traces only); the last
three additionally load a model. They are the minutes-long probes, which is why the web front end
runs them as password-gated background jobs.
"""

from __future__ import annotations

import os

from agents.model.compile_opponents import maybe_compile_extractor


class _CounterfactualMixin:
    def falsify(self, battle_id: str, *, invs=None, worst: int = 3,
                n_seeds: int = 40, n_alts: int = 3, followup: str = "random") -> dict:
        """Dice attribution (luck vs reducible mistake) for a battle's worst — or
        explicitly chosen — decisions, by RE-ROLLING the real turns through the
        reconstruction layer (fix-both luck percentile + paired alternative-action
        sweep on a material margin). Model-free (no checkpoint); requires the
        trace's ``*_reconstruction.json`` sibling, which only bridge-eval traces
        written by the reconstruction layer carry."""
        from main.prober.falsifier import falsify_battle
        from utils.bridge.reconstruction import ReconstructionRecord

        b = self._battle(battle_id)
        recon_path = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon_path):
            raise FileNotFoundError(
                f"no reconstruction record next to this trace ({recon_path}) — "
                "the battle predates the reconstruction layer or ran websocket "
                "eval; only bridge-eval traces carry the falsifier's replay data")
        record = ReconstructionRecord.load(recon_path)
        return falsify_battle(record, self._summary(b), self._npz(b),
                              invs=invs, worst=worst, gamma=self._gamma,
                              n_seeds=n_seeds, n_alts=n_alts, followup=followup,
                              impl=self._impl)

    def lookahead(self, battle_id: str, *, invs=None, inv: "int | None" = None, worst: int = 3,
                  n_seeds: int = 0, followup: str = "random") -> dict:
        """One-ply LOOKAHEAD (Feature 1): for an anchored ``move_selection`` decision, RE-ROLL the turn
        under each legal action (the opponent plays its RECORDED move), materialize the resulting
        one-sided successor state through the real encoder, and read the loaded model's **V(s′)** — so
        the result is per-action ΔV, "what would the critic have valued each alternative at" (the value
        readout the model-free :meth:`falsify` deliberately defers; + the distributional / win-prob heads
        when the run trained them). Loads the exact→nearest→recent model. **Requires the trace's
        ``*_reconstruction.json`` sibling** (bridge-eval traces only). ``inv`` looks ahead from ONE
        decision (returns the single-decision dict); otherwise ``invs`` (or the ``worst`` δ-craters)
        returns a battle dict. ``n_seeds`` > 0 dice-averages V(s′) ± std alongside the CRN headline.
        The CHOSEN action's CRN successor reproduces the real next state, so its ``value_crn`` ≈ the
        trace's ``recorded_next_value`` — a built-in consistency anchor."""
        from main.prober.lookahead import lookahead_battle, lookahead_decision
        from utils.bridge.reconstruction import ReconstructionRecord

        b = self._battle(battle_id)
        recon_path = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon_path):
            raise FileNotFoundError(
                f"no reconstruction record next to this trace ({recon_path}) — lookahead needs the "
                "re-roll layer's replay data, which only bridge-eval traces carry")
        record = ReconstructionRecord.load(recon_path)
        model, _ = self._model_for(b)
        summary, npz = self._summary(b), self._npz(b)
        if inv is not None:
            out = lookahead_decision(model, record, summary, npz, int(inv),
                                     n_seeds=n_seeds, followup=followup, impl=self._impl)
        else:
            out = lookahead_battle(model, record, summary, npz, invs=invs, worst=worst,
                                   gamma=self._gamma, n_seeds=n_seeds, followup=followup,
                                   impl=self._impl)
        # The currency `value_crn` / `delta_v` are in. Under `--critic winprob` `win_prob_crn` is
        # the SAME number as `value_crn` (verified on the live arm: equal to 4 dp on every
        # candidate), so a surface must be able to say "one readout" rather than presenting a
        # column that can never disagree as a second opinion.
        out["critic_currency"] = self.critic_currency()
        return out

    def better_line(self, battle_id: str, inv: int, *, depth: int = 2, beam: int = 3, top_k: int = 4,
                    followup: str = "random", opponent_ckpt: "str | None" = None,
                    interior_opponent: str = "self", confirm_rollouts: int = 0,
                    search_session=None) -> dict:
        """SEARCH for a better line than the model played (``better_line.py``): a shallow CRN-anchored
        beam over the critic from an anchored ``move_selection`` decision, returning ONE contrastive
        trajectory ("at turn T, do X instead — here is the line, the per-ply ΔV/ΔP(win), and where the
        recorded play went wrong"). ``depth`` = OUR plies looked ahead (1 == :meth:`lookahead`); ``beam``
        / ``top_k`` bound the interior branching.

        The faithful-conditional opponent: the RECORDED move at the divergence ply (the value_crn
        anchor), and at INTERIOR plies the reloaded opponent reacts greedily — ``interior_opponent``:
        ``"self"`` (the trainee's own policy, a flagged self-play approximation; default), ``"ckpt"``
        (load ``opponent_ckpt`` as the opponent), or ``"none"`` (the sim's default move). When
        ``confirm_rollouts`` > 0 the recommended first action is CONFIRMED by an actual Monte-Carlo
        replay-to-end vs the RELOADED REAL opponent (:meth:`replay_counterfactual`) → win-% ± Wilson CI,
        the ground-truth check on the critic's claim. Loads the exact→nearest→recent model. **Requires
        the trace's ``*_reconstruction.json`` sibling** (bridge-eval traces only)."""
        from main.prober.better_line import better_line_decision
        from utils.bridge.reconstruction import ReconstructionRecord

        b = self._battle(battle_id)
        recon_path = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon_path):
            raise FileNotFoundError(
                f"no reconstruction record next to this trace ({recon_path}) — better-line needs the "
                "re-roll layer's replay data, which only bridge-eval traces carry")
        # An injected WARM SearchSession carries its OWN impl (it was spawned before this call), so a
        # caller that built it on one engine and the ProbeSession on another would produce a
        # correction whose search half and replay half came from different sims. That is exactly the
        # silent cross-engine mix the seam exists to prevent — refuse it rather than pick a winner.
        if search_session is not None:
            ss_impl = getattr(search_session, "impl", self._impl)
            if ss_impl != self._impl:
                raise ValueError(
                    f"injected SearchSession impl {ss_impl!r} != this ProbeSession's impl "
                    f"{self._impl!r} — build both on the same engine")

        record = ReconstructionRecord.load(recon_path)
        model, _ = self._model_for(b)
        summary, npz = self._summary(b), self._npz(b)

        opp_model = None
        opp_used = "recorded@divergence"
        if depth >= 2 and interior_opponent != "none":
            if interior_opponent == "ckpt":
                if not opponent_ckpt:
                    raise ValueError("interior_opponent='ckpt' requires opponent_ckpt=")
                from main.prober.model import ProbeModel
                opp_model = self._models.get(opponent_ckpt) or ProbeModel.load(opponent_ckpt)
                self._models[opponent_ckpt] = opp_model
                opp_used = f"reloaded:{os.path.basename(opponent_ckpt)}"
            else:                                     # "self" — the trainee as a flagged proxy
                opp_model = model
                opp_used = "reloaded:self_model_approx"

        out = better_line_decision(
            model, record, summary, npz, int(inv), depth=depth, beam=beam, top_k=top_k,
            followup=followup, opp_model=opp_model, session=search_session, impl=self._impl)
        out["interior_opponent"] = opp_used
        # The currency the beam's ΔV and every ply's V(s′) are in — see `lookahead` above.
        out["critic_currency"] = self.critic_currency()

        # Ground-truth CONFIRM of the recommended first action vs the RELOADED REAL opponent.
        if confirm_rollouts > 0 and out.get("best_alternative"):
            act = int(out["best_alternative"]["action"])
            try:
                conf = self.replay_counterfactual(
                    battle_id, int(inv), act, n_rollouts=confirm_rollouts,
                    opponent_ckpt=opponent_ckpt, narrate=False)
                out["confirm"] = {
                    "action": act, "label": out["best_alternative"]["label"],
                    "win_rate": conf.get("win_rate"), "ci": conf.get("win_rate_ci"),
                    "n_rollouts": confirm_rollouts, "opponent_source": conf.get("opponent_source"),
                    "caveats": conf.get("caveats"),
                }
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                out["confirm"] = {"error": str(e)}
        return out

    def replay_counterfactual(self, battle_id: str, inv: int, action: int, *, n_rollouts: int = 1,
                              opponent_ckpt: "str | None" = None, opponent_source: str = "auto",
                              opponent_stochastic: "bool | None" = None,
                              narrate: bool = False) -> dict:
        """COUNTERFACTUAL replay-to-end (Feature 2) — "could the model have won if it hadn't choked this
        turn?". Pick up the recorded battle at ``inv``'s turn, substitute ``action`` (a legal action
        index) for OUR side, then play the rest LIVE — the trainee's GREEDY policy vs the RELOADED real
        opponent — to a win / loss. ``n_rollouts`` > 1 resamples the post-divergence dice (Monte-Carlo)
        for a win-PROBABILITY ± Wilson CI; ``n_rollouts`` == 1 is the single realized-dice line. The
        opponent is RELOADED: a reproducible bot is rebuilt exactly, ``opponent_ckpt`` loads any
        checkpoint (e.g. a self-play sentinel) as the opponent, else the trainee's own model stands in
        (a flagged self-play approximation). A checkpoint opponent plays in the regime the record
        says it played — **stochastic** at temp 1.0, matching ``eval_worker``'s sentinels — unless
        ``opponent_stochastic`` overrides it. **Requires the trace's ``*_reconstruction.json``
        sibling.**"""
        from sb3_contrib import MaskablePPO
        from poke_env.ps_client import LocalhostServerConfiguration
        from agents.observation.state_encoder import load_mappings
        from main.prober.replay import replay_counterfactual_battle
        from utils.bridge.reconstruction import ReconstructionRecord

        b = self._battle(battle_id)
        recon_path = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon_path):
            raise FileNotFoundError(
                f"no reconstruction record next to this trace ({recon_path}) — counterfactual replay "
                "needs the re-roll layer's replay data, which only bridge-eval traces carry")
        record = ReconstructionRecord.load(recon_path)

        choice = self._resolve(b)
        if choice.path is None:
            raise FileNotFoundError(
                f"no checkpoint resolved for the counterfactual trainee: {choice.detail}")
        if self._cf_mappings is None:
            self._cf_mappings = load_mappings()

        def _load(path):
            m = self._play_models.get(path)
            if m is None:
                # Drop any saved extractor kwarg the CURRENT constructor rejects (a flag deleted or
                # demoted since the checkpoint was written — v78 value_active_readout /
                # damage_matrices_outgoing_all, v88 pubval_mode). Without this a bare load TypeErrors
                # on any current-gen checkpoint, silently breaking every rollout path (this method,
                # better-line, lookahead). Same sanitizer ProbeModel.load uses.
                from main.prober.model import sanitized_load_custom_objects
                custom_objects, _dropped = sanitized_load_custom_objects(path, "cpu")
                m = MaskablePPO.load(path, env=None, device="cpu", custom_objects=custom_objects)
                m.policy.set_training_mode(False)
                # A `--log-level periodic` checkpoint carries an ObservationDebugger that print()s a
                # "DEEP TRACE" banner on every forward — it would corrupt the CLI's JSON stdout and the
                # TUI screen. Silence it on the replay players, exactly as ProbeModel.load does.
                for mod in m.policy.modules():
                    if hasattr(mod, "_debugger"):
                        mod._debugger = None
                # These models are used ONLY for no-grad rollouts (better-line's beam,
                # replay-counterfactual's Monte-Carlo re-rolls, falsify's paired sweeps) — thousands
                # of B=1 CPU forwards, the exact shape --compile-opponents targets. Gated on
                # `compile_extractor` because a one-off `summary`/`list` query should not pay a
                # ~10-20s compile it will never amortize. Grad-enabled calls (saliency) are routed
                # to eager inside the helper, so this cannot break the gradient paths.
                maybe_compile_extractor(m, self._compile_extractor,
                                        label=f"prober:{os.path.basename(path)}", hide_cuda=True)
                self._play_models[path] = m
            return m

        play_model = _load(choice.path)
        opp_model = _load(opponent_ckpt) if opponent_ckpt else None

        return replay_counterfactual_battle(
            record, self._summary(b), self._npz(b), int(inv), int(action),
            play_model=play_model, opp_name=b.opponent, mappings=self._cf_mappings,
            server_config=LocalhostServerConfiguration, opponent_ckpt=opponent_ckpt,
            opp_model=opp_model, opponent_source=opponent_source,
            opponent_stochastic=opponent_stochastic, n_rollouts=n_rollouts,
            narrate=narrate, impl=self._impl)
