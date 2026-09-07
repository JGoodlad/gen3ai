"""`TrainMetricsExport` — the ~400 lines of `self.logger.record` that close every `train()`.

Diagnostics ONLY. Nothing here carries a gradient, nothing here is part of the fold sequence, and
every method takes the accumulators `train()` filled during the epoch loop rather than reading
state back off `self` — so which numbers each block publishes is answerable from its signature.

`train()` calls them in the order they are defined below, and that order is the one the keys were
published in before the split. `train/train_ms` deliberately stays in `train()`: it is the LAST
line of the call and its comment says so, which only stays true where it is.
"""
import numpy as np

from agents.training import frozen_phi          # gen3_frozen_phi_actor_only_v1 (both seams live there)
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
        # reducing vf_coef and adding return normalization (PopArt) — see grad_balance.py and
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
        # `train/`, which is crowded — matches the dedicated `grad/`/`popart/`/`win_prob/`/`eval/`
        # groups). Only when the aux is on AND some minibatch had believed slots. `species_acc` is the
        # headline: top-1 accuracy of predicting a hidden mon's species — rises as the model learns to
        # anticipate the un-revealed party.
        if belief_metrics:
            for _bk, _bvals in belief_metrics.items():
                self.logger.record(f"belief/{_bk}", float(np.mean(_bvals)))

        # +WIN-PROB: auxiliary win-probability diagnostics under their OWN `win_prob/` TB prefix (NOT
        # `train/`, which is crowded — matches the dedicated `grad/`/`popart/`/`eval/` groups). Only when
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
        # scaffolding gauge), not on `win_prob_coef`: a `read_only` head at coefficient 0 is still
        # making claims worth checking. An under-populated bin publishes NaN, so a thin tail bin
        # renders as a HOLE rather than as a confident calibration error.
        for _ck2, _cv2 in calib_all.metrics().items():
            self.logger.record(f"win_prob/{_ck2}", _cv2)
        for _ck2, _cv2 in calib_contested.metrics(prefix="contested_").items():
            self.logger.record(f"win_prob/{_ck2}", _cv2)

        # +WIN-PROB CRITIC RELIABILITY (gen3_winprob_critic_mode_v1) — the DEPLOYED value's own
        # Murphy split, once per rollout, under `--critic winprob` only. Beside the head's
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

    def _record_term_metrics(self, value_dist_metrics: dict, teacher_metrics: dict,
                             opd_metrics: dict, distill_metrics: dict,
                             td_aux_metrics: dict) -> None:
        """The PBRS shaping magnitude and the five per-term prefixes that carry no counterfactual."""
        # +WIN-PROB PBRS (gen3_winprob_pbrs_v1, ai_v12 route 1): the shaping term's magnitude for THIS
        # rollout, computed in `collect_rollouts` (not here — the term edits rewards, not the loss, so
        # it has no per-minibatch existence). Under `train/` deliberately: it is a property of the
        # reward stream PPO is fitting, not of the win-prob head, and it belongs beside the other
        # train-loop quantities a reader checks when the loss moves. `pbrs_reward_share` is the one to
        # watch — the shaping's mean |magnitude| as a fraction of the UNSHAPED reward's, i.e. how much
        # of the return signal this coefficient has replaced.
        if self._pbrs_metrics:
            for _pk, _pv in self._pbrs_metrics.items():
                self.logger.record(f"train/pbrs_{_pk}", float(_pv))

        frozen_phi.record_metrics(self, self.logger)  # pbrs/frozen_phi_*, signal/adv_shaped_*
        # +VALUE-DIST: distributional value head diagnostics under their OWN `value_dist/` TB prefix (the
        # interpretability head's aggregate health, complementing the prober's per-decision histogram).
        # `entropy`/`std` fall as the critic sharpens; `pit_mean` ≈ 0.5 ⟺ calibrated; `mean_abs_err` =
        # |E[Z] − return| in support units. Ride the generic logger → TensorBoard + launcher TUI.
        if value_dist_metrics:
            for _vk, _vvals in value_dist_metrics.items():
                self.logger.record(f"value_dist/{_vk}", float(np.mean(_vvals)))

        # +SEARCH-TEACHER: AWR diagnostics under their OWN `teacher/` TB prefix. `agree_rate` (policy ↔
        # A* — should RISE as the distillation lands), `mean_adv` (the confirmed win-rate improvement of
        # the corrections), `mean_w` (AWR weight), `ce`, `loss`, `n`; `buffer_size` = the standalone ring
        # depth. The shared-trunk pull rides `grad/searchteacher_share` (+ `_policy_cosine` — the live
        # "is the teacher fighting the actor" signal). `teacher/yield` + `/corrections_per_cycle` are
        # emitted by SearchTeacherCallback (cross-process facts). Empty (off / empty buffer) → not logged.
        if teacher_metrics:
            for _tk, _tvals in teacher_metrics.items():
                self.logger.record(f"teacher/{_tk}", float(np.mean(_tvals)))
            cb = getattr(self, "_correction_buffer", None)
            if cb is not None:
                self.logger.record("teacher/buffer_size", float(len(cb)))

        # +OPD: on-policy self-distillation KL diagnostics under their OWN `opd/` TB prefix. `kl` = the
        # forward KL(π' ‖ π_student) being minimized (should FALL as the student matches π'),
        # `pi_target_entropy` = π' sharpness (low = decisive target), `agree_rate` = student ↔ π' mode
        # agreement (should RISE), `n` = the sampled correction count. The shared-trunk pull rides
        # `grad/opd_share`. Empty (off / empty buffer / an AWR-only π'-less sample) → not logged.
        if opd_metrics:
            for _ok, _ovals in opd_metrics.items():
                self.logger.record(f"opd/{_ok}", float(np.mean(_ovals)))

        # +DISTILL: exploiter-distillation KL diagnostics under their OWN `distill/` TB prefix. `kl` = the
        # masked forward KL(π_teacher ‖ π_student) being minimized (should FALL as the student matches the
        # specialist), `agree_rate` = student ↔ teacher mode agreement on teacher-team states (should RISE),
        # `coverage` = fraction of the minibatch on the teacher's team, `n` = teacher-team state count.
        # Under `--distill-target action` (gen3_distill_target_gate_v1) the §4.3 liveness row rides the
        # same prefix: `gated_frac` / `n_gated` (0 is a reading: the gate found nothing) /
        # `gate_agree_rate` (student argmax == teacher argmax ON GATED ROWS) / `mean_gate_adv` — the
        # dose meters G2's share-matching is read against (with grad/distill_share).
        # Empty (off / no teacher-team states in any minibatch) → not logged.
        if distill_metrics:
            for _dk, _dvals in distill_metrics.items():
                self.logger.record(f"distill/{_dk}", float(np.mean(_dvals)))

        # +TD-AUX: Bellman-residual diagnostics under their OWN `td_aux/` TB prefix. `resid_rms` is
        # the headline — the quantity the term minimises, and the live counterpart of the offline
        # ΔV-dispersion instrument the rung-1 gate used; it should FALL. `resid_mean` (SIGNED) is the
        # no-harm watch: rung 1's decomposition says this is dispersion suppression, so a bias that
        # drifts away from ~0 means the residual-gradient (Baird) term is shifting the level rather
        # than tightening it — read it beside `train/explained_variance`. `scale` is the unit the
        # residual is expressed in (PopArt's sigma; 1.0 with PopArt off), and `pair_drop_frac` is the
        # fraction of candidate pairs lost to episode boundaries. Empty (off) → not logged.
        if td_aux_metrics:
            for _tdk, _tdvals in td_aux_metrics.items():
                self.logger.record(f"td_aux/{_tdk}", float(np.mean(_tdvals)))

    def _record_cf_metrics(self, cf_buffer, cf_any_on: bool, cf_rows_sampled: int,
                           cf_metrics: dict, cf_winprob_on: bool, cf_evid_metrics: dict,
                           cf_evid_on: bool, cf_twin_metrics: dict, cf_twin_on: bool,
                           cf_shadow_metrics: dict, cf_shadow_on: bool, q_metrics: dict,
                           q_winprob_on: bool, q_onpolicy_on: bool,
                           grad_balance: dict) -> None:
        """The counterfactual family — producer liveness first, then each head's own fold."""
        # +CF-WINPROB (gen3_cf_label_plumbing_v1). Two blocks, and they answer DIFFERENT questions:
        #
        #  * `cf/*` is PRODUCER LIVENESS — published on every train() the moment a buffer exists,
        #    whether or not a single label ever arrived. An empty buffer that does not announce
        #    itself is this tree's oldest failure mode (the search-teacher's silent starvation), so
        #    `cf/buffer_fill` == 0 with a flat `cf/labels_ingested_total` is a first-class reading,
        #    not an absence of readings.
        #  * `train/cf_loss` + `train/cf_grad_share` are the TERM — only when it actually folded.
        #    `cf_grad_share` is lifted from the grad-balance probe's shared denominator so it is
        #    directly comparable with grad/policy_share et al; it reads 0.0 under `cf_head_only`
        #    (the default) because the head's input is stop-grad'd, which is the head-only stage's
        #    verification, not a defect.
        if cf_buffer is not None:
            for _ck, _cv in cf_buffer.stats(int(self.num_timesteps)).items():
                self.logger.record(_ck, _cv)
        if cf_any_on:
            # Rows CONSUMED per train() — the throughput half of liveness, which residency alone
            # cannot report. Recorded whenever the fold is enabled (0 is the reading that matters:
            # the term is on and the buffer gave it nothing).
            self.logger.record("cf/rows_sampled", float(cf_rows_sampled))
        if cf_metrics:
            for _ck2, _cvals in cf_metrics.items():
                self.logger.record(f"cf/{_ck2}", float(np.mean(_cvals)))
            self.logger.record("train/cf_loss", float(np.mean(cf_metrics.get("loss", [0.0]))))
        if cf_winprob_on:
            self.logger.record("train/cf_grad_share",
                               float(grad_balance.get("grad/cf_winprob_share", 0.0)))
        # +CF-EVIDENTIAL (gen3_cf_evidential_head_v1) — `cf/evid_*`, its own sub-prefix so a reader
        # can tell the Beta readout's numbers from the scalar head's at a glance.
        #
        #  * `nll` is the term being minimised (Beta-Binomial marginal, per rollout) and `reg` the
        #    KL pull toward Beta(1,1).
        #  * `precision_mean` (α+β) is the EVIDENCE the head claims. An unbounded climb is the
        #    evidential-overconfidence failure `--cf-evidential-reg` exists to bound; read the two
        #    together, because a falling `nll` with a runaway precision is the head buying its loss
        #    with certainty it has not earned.
        #  * `epistemic_std_mean` is THE HEADLINE and the pre-registered read: the confessed width
        #    should CORRELATE, per stratum, with `cf_audit`'s measured `sd_true_excess` for that
        #    stratum. Wide everywhere and wide nowhere are the same null.
        if cf_evid_metrics:
            for _ek, _evals in cf_evid_metrics.items():
                self.logger.record(f"cf/evid_{_ek}", float(np.mean(_evals)))
            self.logger.record("train/cf_evidential_loss",
                               float(np.mean(cf_evid_metrics.get("nll", [0.0]))))
        if cf_evid_on:
            # Reads 0.0 by construction (the head's input is always detached). Published so the
            # always-detached contract is a LIVE measurement rather than a claim in a docstring.
            self.logger.record("train/cf_evidential_grad_share",
                               float(grad_balance.get("grad/cf_evidential_share", 0.0)))
        # +CF-TWIN (gen3_cf_twin_heads_v1) — `cf/twin_*`. The PAIRED read, live.
        #
        #  * `c_loss` / `b_loss` are the two arms' cf folds; `b_coverage` is the fraction of the
        #    sampled rows that carried a single-outcome label AT ALL. **Read `b_coverage` first.**
        #    A twin-heads run whose producer ships no `outcome_label` trains B on nothing, B then
        #    equals A, and the C−B contrast silently becomes C−A while every other scalar reads
        #    healthy. That is the one way this arm can produce a confident wrong answer.
        #  * `b_vs_c_abs` / `b_minus_c` are the two heads' predictions differing on the SAME states.
        #    A `b_vs_c_abs` pinned near 0 means the label streams have not separated the heads and
        #    there is nothing to decompose yet — a coverage/dosage reading, not a result.
        #  * `*_onpolicy_*` are the mirrored control objective. They should track head A's
        #    `win_prob/*` closely; a persistent gap means the twins are NOT carrying a bit-identical
        #    copy of A's loss and the factorial's base is not shared.
        #  THE RESULT IS NONE OF THESE. It is `cf_audit`'s held-out, battle-clustered paired
        #  differences (runbook §2 as amended); these are the launch-window instrument.
        if cf_twin_metrics:
            for _tk, _tvals in cf_twin_metrics.items():
                self.logger.record(f"cf/twin_{_tk}", float(np.mean(_tvals)))
            # ABSENT, never zero — the shadow head's rule, for the shadow head's reason. The
            # COMBINED (C + B) unweighted fold, summed per minibatch inside `cf_twin_terms` and
            # only meaned here: the two arms' lists differ in length when B starves, so
            # mean(c)+mean(b) would be the mean of no minibatch that ever folded. One scalar,
            # because this block contributes ONE term to the loss; the per-arm split is already
            # live at `cf/twin_c_loss` / `cf/twin_b_loss`, which is where you read the arms.
            # The key is missing (not 0.0) when the CF fold never ran — `cf_twin_metrics` can be
            # non-empty from the ON-POLICY mirror alone, and a defaulted 0.0 would then publish
            # a perfect score for an arm that saw no counterfactual label at all.
            if "loss" in cf_twin_metrics:
                self.logger.record("train/cf_twin_loss",
                                   float(np.mean(cf_twin_metrics["loss"])))
        if cf_twin_on:
            self.logger.record("train/cf_twin_grad_share",
                               float(grad_balance.get("grad/cf_twin_share", 0.0)))
        # +CF-SHADOW (gen3_cf_twin_heads_v1) — `cf/shadow_*`.
        #
        #  * `loss` is the MSE against `mc_return` in the PopArt-normalized frame; `abs_err`,
        #    `pred_mean` and `label_mean` are the same quantities de-normalized to real shaped-return
        #    units, which is the only frame a reader can interpret.
        #  * **`shadow_vs_live_v` is THE METER** — the SIGNED mean of (shadow − live V) in real
        #    units on the same states. It is the staged-promotion evidence: a shadow sitting
        #    systematically BELOW the live critic is a live critic that is optimistic about the
        #    states the factory sampled, measured against ground truth rather than argued from a
        #    calibration curve. `live_v_vs_label` is its direct half (live V minus the MC label);
        #    read them together, since the shadow is itself a fitted head and can be wrong too.
        #  * `coverage` is `mc_return`'s label coverage — the same first-read rule as `b_coverage`.
        if cf_shadow_metrics:
            for _sk, _svals in cf_shadow_metrics.items():
                self.logger.record(f"cf/shadow_{_sk}", float(np.mean(_svals)))
            # ABSENT, never zero. A starved shadow (no `mc_return` arrived, or every row's reward
            # digest was refused) folds NO term, so there is no `loss` key — and defaulting it to
            # 0.0 would publish a PERFECT SCORE for a head that trained on nothing, which is
            # indistinguishable on a TB chart from a perfectly-fit one. `cf/shadow_coverage` still
            # publishes, so the starvation is visible rather than silent.
            if "loss" in cf_shadow_metrics:
                self.logger.record("train/cf_shadow_loss",
                                   float(np.mean(cf_shadow_metrics["loss"])))
        if cf_shadow_on:
            # 0.0 by construction (always-detached), published for the same reason as the
            # evidential head's — this head is a promotion PATH, so its passivity is the contract.
            self.logger.record("train/cf_shadow_grad_share",
                               float(grad_balance.get("grad/cf_shadow_share", 0.0)))
        # +Q-WINPROB (gen3_q_winprob_head_v1) — `q_winprob/*`, its own prefix so the PER-ACTION
        # head's numbers can never be read as the per-state win head's.
        #
        #  * **`label_coverage` and `labels_per_row` are the FIRST read, before any score.** They
        #    are the starvation tell this head is most exposed to: a producer shipping no
        #    `q_labels` trains it on nothing, and a producer shipping ONE action per state trains
        #    it into exactly the on-policy failure it exists to avoid (ledger 229e9f1).
        #  * `abs_err` / `bias` are the fit on labelled cells. Read them WITH `pred_spread` vs
        #    `label_spread`: a head that has learned nothing per-ACTION can still score well on
        #    `abs_err` by predicting each state's mean, and the spread pair is what tells the two
        #    apart — a `pred_spread` far below `label_spread` is a head that has amortized the
        #    VALUE and not the SEARCH.
        #  * `onpolicy_*` are the WEAK fallback's, prefixed apart on purpose (see the flag's
        #    caveat). They are not evidence about the counterfactual stream.
        if q_metrics:
            for _qk2, _qvals in q_metrics.items():
                self.logger.record(f"q_winprob/{_qk2}", float(np.mean(_qvals)))
            # ABSENT, never zero — the shadow head's rule for the shadow head's reason: a starved
            # fold publishes its coverage columns but no `loss`, and a defaulted 0.0 would be a
            # perfect score for a head that trained on nothing.
            if "loss" in q_metrics:
                self.logger.record("train/q_winprob_loss", float(np.mean(q_metrics["loss"])))
        if q_winprob_on or q_onpolicy_on:
            # 0.0 by construction (every input detached inside the extractor forward), published
            # so the "cannot perturb the policy" contract is a measurement, not a claim.
            self.logger.record("train/q_winprob_grad_share",
                               float(grad_balance.get("grad/q_winprob_share", 0.0)))

    def _record_capacity_and_popart_metrics(self, capacity_metrics: dict, popart,
                                            aux_metrics: dict) -> None:
        """The capacity battery, PopArt's currency readout, and the pre-keyed `aux_metrics` sink."""
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

        # +PopArt diagnostics: mu/sigma should TRACK train/return_mean/return_std (the running
        # normalizer estimate); value_weight_norm watches the POP rescale stay bounded (an explosion
        # signals a degenerate sigma / broken preservation). With PopArt on, train/value_loss is the
        # NORMALIZED loss (≈O(1)) and grad/value_policy_logratio should fall toward ~0 (the
        # aux-independent value/policy balance; grad/value_share also drops but moves with the aux count).
        if popart is not None:
            self.logger.record("popart/mu", float(self.policy.popart.mu))
            self.logger.record("popart/sigma", float(self.policy.popart.sigma))
            self.logger.record("popart/value_weight_norm", float(self.policy.value_net.weight.norm()))
            # +THE CURRENCY CONVERSION, MADE READABLE (gen3_popart_currency_readout_v1). μ and σ
            # alone say what the normalizer BELIEVES; these two say whether that belief is CURRENT.
            # `train/return_*` is RAW shaped-return currency and the value loss trains in
            # NORMALIZED currency, so the conversion in force this rollout is
            # `normalized = (raw − μ)/σ` — and applying it to THIS rollout's own returns is the
            # one-line audit of it: a tracking normalizer reads ≈0 and ≈1. A `norm_return_std`
            # drifting from 1 is PopArt LAGGING the return scale (the value gradient is then
            # mis-scaled against the trunk by exactly that factor), and a `norm_return_mean` far
            # from 0 is an offset the value head has to carry itself. Free — a mean and a std over
            # an array `value_scale_metrics` has already read.
            _pa_r = np.asarray(self.rollout_buffer.returns, dtype=np.float64).reshape(-1)
            if _pa_r.size:
                _pa_sigma = float(self.policy.popart.sigma)
                if _pa_sigma > 0.0:
                    _pa_z = (_pa_r - float(self.policy.popart.mu)) / _pa_sigma
                    self.logger.record("popart/norm_return_mean", float(_pa_z.mean()))
                    self.logger.record("popart/norm_return_std", float(_pa_z.std()))
        # (v61's `value_seeds/*` seed-collapse contract was logged here. The multi-seed critic
        # readout it monitored is DELETED — dV 0.0000 bit-exact on two consecutive end-of-run
        # audits — so the monitor went with it. Its finding survives in designs/CHANGELOG.md.)
        for _sk, _svals in aux_metrics.items():
            self.logger.record(_sk, float(np.mean(_svals)))
