"""The DELEGATES — terms whose bodies live in their own modules, bound onto the class here.

Three families already moved out under the same pressure this split answers, and each left a thin
binding behind so every call site and every `model._<name>` test resolves unchanged:

* `belief_bank` — the five supervised belief losses, as one declarative fold (one ROW per head).
* `td_aux` — the Bellman-residual consistency term (`_td_aux_term` keeps the coefficient here).
"""
import numpy as np
import torch as th

from agents.training import belief_bank as _belief_bank


class AuxTerms:
    """The belief-bank and TD-aux delegates."""

    # ALL FIVE supervised belief losses now live in `belief_bank` (the declarative fold);
    # these aliases keep every existing call site and test resolving unchanged.
    _move_belief_loss = staticmethod(_belief_bank.move_belief_loss)
    # The three revealed-slot supervised losses MOVED to `belief_bank` (the declarative fold);
    # these aliases keep every existing call site and test resolving unchanged.
    _spread_belief_loss = staticmethod(_belief_bank.spread_belief_loss)

    _nature_ev_belief_loss = staticmethod(_belief_bank.nature_ev_belief_loss)

    _hp_type_belief_loss = staticmethod(_belief_bank.hp_type_belief_loss)

    _move_belief_latent_loss = staticmethod(_belief_bank.move_belief_latent_loss)

    _belief_aux_loss = staticmethod(_belief_bank.belief_aux_loss)

    def _td_aux_term(self):
        """The TD-consistency auxiliary for ONE minibatch: sample contiguous pairs, forward them,
        return `(weighted_term, metrics)` — or `(None, {})` when nothing is pairable.

        WHY ITS OWN SAMPLE AND ITS OWN FORWARD, per minibatch. Three constraints pin this shape:

        * `rollout_data` cannot serve it. `RolloutBuffer.get()` yields a RANDOM PERMUTATION, so a
          minibatch holds no adjacent pairs whatsoever — the pairs must be drawn from the buffer's
          surviving `[n_steps, n_envs]` structure, which means a second forward either way.
        * It runs PER MINIBATCH, not once per `train()` like the grad-balance / rank probes. Those
          are read-only diagnostics; this one carries gradient, and a once-per-`train()` fold would
          give it ONE gradient contribution against the value loss's `n_epochs x n_minibatches`
          (~240 in production). λ would then have to be ~240x rung-1's calibrated 1.0-3.0 band to
          mean the same thing, which throws away the one number the pre-registration fixed. This
          follows the shape the counterfactual folds establish here (own sample, own forward,
          every minibatch).
        * The cost is bounded by `TD_AUX_STATES` (512) rather than by `batch_size`, so the term is
          ~10% of the train step at production shapes instead of doubling it.

        It goes through `policy.predict_values`, never a hand-rolled value path: that method is what
        routes to the run's critic (`critic_mode`). Reading `value_net` directly would train a
        critic the run does not use.
        """
        from agents.training import td_aux as _td

        buf = self.rollout_buffer
        if not getattr(buf, "generator_ready", False):
            # `state_rows` are in the post-`swap_and_flatten` (env-major) convention; indexing an
            # un-flattened observation array with them would silently mis-pair states with rewards.
            raise RuntimeError(
                "_td_aux_term ran before rollout_buffer.get() flattened the observations — the "
                "contiguous-pair rows are in the post-swap_and_flatten convention and would index "
                "the wrong rows. Call it from INSIDE the minibatch loop.")

        if self._td_aux_rng is None:
            self._td_aux_rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))

        n_rows = int(buf.buffer_size) * int(buf.n_envs)
        rows, pa_np, pb_np, n_cand = _td.sample_contiguous_pairs(
            buf.episode_starts, min(_td.TD_AUX_STATES, n_rows), _td.TD_AUX_SEG_LEN, self._td_aux_rng)
        if pa_np.size == 0:
            return None, {}

        obs = {k: buf.to_torch(v[rows]) for k, v in buf.observations.items()}
        values = self.policy.predict_values(obs).flatten()          # [S] (see td_aux docs)
        # `rewards` is NOT in get()'s flatten list, so it is still [n_steps, n_envs]; swap to the same
        # env-major flat order the rows index.
        rew_flat = np.asarray(buf.rewards).swapaxes(0, 1).reshape(-1)
        rewards = buf.to_torch(rew_flat[rows]).flatten()            # [S] REAL-unit
        pair_a = th.as_tensor(pa_np, dtype=th.long, device=values.device)
        pair_b = th.as_tensor(pb_np, dtype=th.long, device=values.device)
        # UNITS: the residual is real-unit (V, raw reward) — the value loss trains in the same space.
        out = _td.td_aux_loss(values, rewards, pair_a, pair_b,
                              float(self.gamma), n_candidate=n_cand)
        if out is None:
            return None, {}
        td_loss, metrics = out
        return self.td_aux_coef * td_loss, metrics
