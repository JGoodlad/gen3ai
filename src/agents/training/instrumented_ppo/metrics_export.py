"""`TrainMetricsExport` — the ~400 lines of `self.logger.record` that close every `train()`.

Diagnostics ONLY. Nothing here carries a gradient, nothing here is part of the fold sequence, and
every method takes the accumulators `train()` filled during the epoch loop rather than reading
state back off `self` — so which numbers each block publishes is answerable from its signature.

`train()` calls them in the order they are defined below, and that order is the one the keys were
published in before the split. `train/train_ms` deliberately stays in `train()`: it is the LAST
line of the call and its comment says so, which only stays true where it is.
"""
import numpy as np

from agents.training.grad_balance import value_scale_metrics
from agents.training.instrumented_ppo.calibration import (
    announce_vf_coef_scale,
    critic_reliability,
)
from agents.training.instrumented_ppo.constants import _NOISE_SCALE_EMA_DECAY
from agents.training.instrumented_ppo.noise_scale import debiased_ema
from agents.training.scaffolding import live_gauge_metrics


class TrainMetricsExport:
    """Mixin: the metrics tail of `train()`, one method per TB prefix group."""

    def _record_grad_balance_metrics(self, grad_balance: dict, rank_metrics: dict,
                                     edge_metrics: dict, cell_metrics: dict,
                                     grad_norms: list) -> None:
        """The shared-trunk gradient-balance / rank / per-family liveness probes."""
        # +INSTRUMENTATION: gradient-balance + value-scale diagnostics. These prepare for
        # reducing vf_coef — see grad_balance.py and
        # src/agents/training/CLAUDE.md. All ride the standard logger → TensorBoard + launcher TUI.
        for _key, _val in grad_balance.items():
            self.logger.record(_key, _val)
        for _key, _val in rank_metrics.items():   # rank/{trunk,value_cls,policy}_* effective-rank probe
            self.logger.record(_key, _val)
        for _key, _val in edge_metrics.items():   # edge/<fam>_{weight,grad}_norm — is each family ALIVE?
            self.logger.record(_key, _val)
        for _key, _val in cell_metrics.items():   # cell/<name>_{weight,grad}_norm — is each cell ALIVE?
            self.logger.record(_key, _val)
        for _key, _val in value_scale_metrics(
            self.rollout_buffer.returns, self.rollout_buffer.values
        ).items():
            self.logger.record(_key, _val)
        if grad_norms:
            self.logger.record("train/grad_norm", float(np.mean(grad_norms)))

    def _record_signal_metrics(self, signal_metrics: dict, scaffold_v: list, scaffold_z: list) -> None:
        """`signal/adv_*` (read off the RAW advantages in the setup) and `train/scaffolding_gauge`."""
        # +SIGNAL (gen3_signal_rate_metrics_v1): the ADVANTAGE-DENSITY half of the `signal/` group,
        # measured above off the RAW pre-normalization advantages. `adv_raw_std` = how much the critic
        # thinks this rollout's actions mattered; `adv_raw_abs_mean` = its outlier-robust companion
        # (std rising alone ⇒ a few runaway points, not a broader density); `adv_kurtosis` = EXCESS
        # kurtosis, POSITIVE when the signal is concentrated in a few decisive turns (which is what
        # exploit signal looks like) and ≈0 when advantage mass is smeared evenly across decisions.
        # Read WITH `signal/outcome_entropy` (SignalMetricsCallback) — high outcome entropy with LOW
        # density is the mirror paradox, not health. NaN on a degenerate (constant) rollout.
        for _sk, _sv in signal_metrics.items():
            self.logger.record(f"signal/{_sk}", float(_sv))

        # +SCAFFOLDING GAUGE: `train/scaffolding_gauge` = (1 − Spearman ρ(V, P(win))) / 2 over
        # epoch 0's paired reads. 0 = the shaped critic and the win-prob head order states
        # identically (no scaffolding divergence visible in the ordering); 0.5 = independent.
        # It should SHRINK as a generation matures, and that trajectory is the registered signal
        # for annealing the shaping coefficients toward the pure game.
        # ⚠️ ORDERING ONLY — it claims nothing about magnitude, and it goes AMBIGUOUS exactly
        # where PBRS drives V_shaped toward a constant (the critic then has no variance left to
        # rank with). Read it beside `train/value_std`; the magnitude question is the offline
        # `python -m main.scaffolding_gauge`, which fits a per-checkpoint affine V→outcome map on
        # realized outcomes. NaN on a degenerate rollout, and NO key at all when the run carries
        # no win-prob head — a run without the head must leave a GAP, not a flat zero.
        if scaffold_v:
            for _gk, _gv in live_gauge_metrics(np.concatenate(scaffold_v),
                                               np.concatenate(scaffold_z)).items():
                self.logger.record(f"train/{_gk}", float(_gv))

    def _record_noise_scale_metrics(self, accum: int, noise_g_small_sq, noise_g_big_sq,
                                    _ns_terms) -> None:
        """The McCandlish fold: the total, the per-term split, and the out-of-band advisor."""
        # +NOISE-SCALE: fold this call's two-batch-size sample into the EMAs and log the smoothed
        # McCandlish 'simple' gradient noise scale B_simple = tr(Σ)/|G|² — the critical batch size.
        # Read it against your EFFECTIVE batch (batch_size·accum): `train/noise_scale_ratio` = B_simple /
        # effective; ≫1 ⇒ noise-limited (a bigger batch buys ~linear per-step progress), ≪1 ⇒
        # diminishing returns (could shrink for more update steps). Only when accumulating (needs two
        # batch sizes) AND both norms were captured (a full first group formed).
        _nsr_global = None   # +NSR-ADVISOR: this call's smoothed ratios (None until EMAs positive)
        if accum >= 2 and noise_g_small_sq is not None and noise_g_big_sq is not None:
            b_small = float(self.batch_size)
            b_big = b_small * accum
            tr_sigma, g2 = self._noise_scale_estimate(noise_g_small_sq, noise_g_big_sq, b_small, b_big)
            # DEBIASED WARM-UP (gen3_noise_scale_warmup_v1): the SAME `debiased_ema` the per-term
            # readings take, so the total and the per-term halves warm up identically. The old fold
            # anchored on its first sample at a fixed decay 0.99, so `train/noise_scale` reported
            # that sample for its first few hundred calls — and one negative first `tr(Σ)` (this
            # estimator's single-call solve can sign-flip under noise) suppressed the scalar
            # entirely, which is exactly what the R5F15 "provisional, n=2" reading was.
            self._noise_ema_s = debiased_ema(self._noise_ema_s, self._noise_ema_n,
                                             tr_sigma, _NOISE_SCALE_EMA_DECAY)
            self._noise_ema_g2 = debiased_ema(self._noise_ema_g2, self._noise_ema_n,
                                              g2, _NOISE_SCALE_EMA_DECAY)
            self._noise_ema_n += 1
            if self._noise_ema_g2 > 1e-12 and self._noise_ema_s > 0.0:
                b_simple = self._noise_ema_s / self._noise_ema_g2
                self.logger.record("train/noise_scale", float(b_simple))
                self.logger.record("train/noise_scale_ratio", float(b_simple / b_big))
                _nsr_global = float(b_simple / b_big)
        # +NOISE-SCALE PER-TERM: the SAME solve, per loss group, on the gradients the sampler
        # accumulated over that same first group. Emitted beside the total so the two are read
        # together — the finding this exists for is a DISAGREEMENT between them, and a reader who
        # has to fetch the halves from different places will not notice one. The probe self-reports
        # its own cost (`train/noise_per_term_ms`) so the overhead is a live number, not a claim.
        if _ns_terms.collecting:
            _pt = _ns_terms.result(accum)
            if _pt:
                b_small = float(self.batch_size)
                for _tag, _val in self._fold_per_term_noise(
                        _pt, b_small, b_small * accum, self._noise_ema_g2).items():
                    self.logger.record(_tag, _val)
            self.logger.record("train/noise_per_term_ms", 1000.0 * _ns_terms.probe_seconds)
            _ns_terms.release()
        # +NSR-ADVISOR: the smoothed PPO-policy-term ratio, read off the EMA state so it survives a
        # call the cadence did not sample (see `_per_term_ratio`).
        _nsr_policy = self._per_term_ratio("policy", float(self.batch_size) * accum)
        # +NSR-ADVISOR: rate-limited TUI Events warnings when a smoothed noise-scale ratio is out
        # of band, with the concrete fix in the message (see _noise_scale_advice). Only on the
        # accumulating path (the estimator needs two batch sizes). The policy-term ratio rides
        # along: it is quoted inside the band warnings and, when the two disagree, produces its own.
        if accum >= 2 and _nsr_global is not None:
            self._emit_noise_scale_warnings(_nsr_global, float(self.batch_size) * accum, _nsr_policy)

    def _record_head_metrics(self, belief_metrics: dict, win_prob_metrics: dict,
                             calib_all, calib_contested, critic_winprob: bool,
                             scaffolding_on: bool, grad_balance: dict) -> None:
        """The supervised heads' own prefixes: `belief/`, `win_prob/`, and the critic's Murphy split."""
        # +BELIEF: hidden-opponent belief-aux diagnostics under their OWN `belief/` TB prefix (NOT
        # `train/`, which is crowded — matches the dedicated `grad/`/`win_prob/`/`eval/`
        # groups). Only when the aux is on AND some minibatch had believed slots. `species_acc` is the
        # headline: top-1 accuracy of predicting a hidden mon's species — rises as the model learns to
        # anticipate the un-revealed party.
        if belief_metrics:
            for _bk, _bvals in belief_metrics.items():
                self.logger.record(f"belief/{_bk}", float(np.mean(_bvals)))

        # +WIN-PROB: auxiliary win-probability diagnostics under their OWN `win_prob/` TB prefix (NOT
        # `train/`, which is crowded — matches the dedicated `grad/`/`eval/` groups). Only when
        # the head is on AND some minibatch had a known label. Calibration: `acc` (top-1 win/loss) +
        # `brier` (lower = P(win) tracks the win rate); `pred_mean` vs `label_mean` watches a base-rate
        # collapse; `coverage` = fraction with a known label. INFORMATION VALUE (the aggregate hides it —
        # blowouts are trivial): `brier_contested`/`acc_contested` on CLOSE games (|margin|<τ; judge vs the
        # ~0.25 no-skill floor of a 50/50 game), `contested_frac`/`contested_label_mean`, and
        # `skill_vs_material` (Brier skill vs a material-only baseline — >0 ⇒ beats counting mons). The
        # shared-trunk pull rides `grad/win_prob_share` (≈0 under read_only; real under shaping).
        if win_prob_metrics:
            for _wk, _wvals in win_prob_metrics.items():
                self.logger.record(f"win_prob/{_wk}", float(np.mean(_wvals)))

        # +WIN-PROB CALIBRATION (gen3_winprob_calibration_export_v1): ECE / MCE / the 10-bin
        # reliability histogram, pooled and CONTESTED-restricted. These measure the RELIABILITY
        # term Brier only carries in a decomposition — the quantity that has to be right when the
        # head becomes the critic's only signal. Gated on the head's EXISTENCE (like the
        # scaffolding gauge): a `read_only` head is still
        # making claims worth checking. An under-populated bin publishes NaN, so a thin tail bin
        # renders as a HOLE rather than as a confident calibration error.
        for _ck2, _cv2 in calib_all.metrics().items():
            self.logger.record(f"win_prob/{_ck2}", _cv2)
        for _ck2, _cv2 in calib_contested.metrics(prefix="contested_").items():
            self.logger.record(f"win_prob/{_ck2}", _cv2)

        # +WIN-PROB CRITIC RELIABILITY (gen3_winprob_critic_mode_v1) — the DEPLOYED value's own
        # Murphy split, once per rollout, under the win-prob critic only. Beside the head's
        # calibration keys above rather than in a parallel prefix; the `critic_` infix says which
        # of the two this is. `resolution` is the meter, not `reliability` — see
        # `calibration.critic_reliability`, which owns the read and the reasoning.
        if critic_winprob:
            for _rk, _rv in critic_reliability(self.rollout_buffer).items():
                self.logger.record(f"win_prob/critic_{_rk}", _rv)
            # ONCE, first NON-DEGENERATE update: what --vf-coef does to the shared trunk now it
            # weights a BCE. Handed the EXISTING `grad_balance` probe, so the printed ratio IS
            # 10 ** grad/value_policy_logratio and no second backward runs for a banner.
            announce_vf_coef_scale(self, win_prob_metrics.get("loss"), grad_balance)

        # +WIN-PROB EPISODE-START READ: what the head says at the LEAST-informed state, against
        # what those very episodes went on to do. One extra EAGER forward over the episode-start
        # rows only (≤ a few hundred), once per `train()`. Eager `type(fe).forward` rather than the
        # bound `fe.forward` for the capacity-probe's reason: both compile flags patch the bound
        # attribute, and a second obs shape through the compiled entry point would add a dynamo
        # graph for a diagnostic (`cache_size_limit` is 8).
        for _sk2, _sv2 in self._winprob_start_metrics(scaffolding_on).items():
            self.logger.record(f"win_prob/{_sk2}", _sv2)

    def _record_capacity_metrics(self, capacity_metrics: dict, aux_metrics: dict) -> None:
        """The capacity battery and the pre-keyed `aux_metrics` sink."""
        # +CAPACITY TELEMETRY (gen3_capacity_telemetry_v1). Read them as TRENDS, never as levels —
        # every one of these is a saturation EARLY WARNING and none has a meaningful absolute value:
        #   canary_loss / canary_recovery / canary_age  the plasticity canary. `canary_recovery` is
        #       the one-number read (post-reset loss ÷ pre-reset loss for the target that was last
        #       re-seeded); compare it at a MATCHED `canary_age`, since it decays with age by design.
        #   canary_steps  how many canary updates this train() actually took. 0 with the flag ON
        #       means the `value_pooled` snapshot never arrived (a non-Gen3 extractor, or a stash
        #       that stopped being populated) — the tell that would otherwise be a silent gap.
        #   halfbatch_cosine  the two half-batches' agreement on the shared trunk. Falling toward
        #       0 / negative = the batch is fighting itself. Read with halfbatch_grad_norm_ratio.
        #   feature_velocity{,_cos,_rel}  how far the FROZEN probe batch's features moved since the
        #       last measurement. Falling velocity at constant `train/grad_norm` = weights move but
        #       functions do not.
        for _capk, _capv in capacity_metrics.items():
            self.logger.record(f"capacity/{_capk}", float(_capv))

        # (v61's `value_seeds/*` seed-collapse contract was logged here. The multi-seed critic
        # readout it monitored is DELETED — dV 0.0000 bit-exact on two consecutive end-of-run
        # audits — so the monitor went with it. Its finding survives in designs/CHANGELOG.md.)
        for _sk, _svals in aux_metrics.items():
            self.logger.record(_sk, float(np.mean(_svals)))
