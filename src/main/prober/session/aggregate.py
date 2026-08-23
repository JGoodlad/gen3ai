"""The two RUN-LEVEL aggregations over the counterfactual tier: `falsify_scan` + `calibration`.

`falsify_scan` brackets a run's losses into aleatoric / unattributed / proven-reducible;
`calibration` splits that unattributed bucket into critic-overvalued vs lost-position. Read the
`caveats` each returns — the crater bracket is an UPPER BOUND, not a measurement.
"""

from __future__ import annotations

import os

from main.prober.discovery import BattleTrace
from main.prober.session.serialize import _r, _short_id
from main.prober.session.stats import (_calibration_stats, _discounted_returns,
    _reliability_curve, _reliability_gap_at)


class _AggregateMixin:
    def falsify_scan(self, *, outcome: "str | None" = "loss",
                     opponent: "str | None" = None, step: "int | None" = None,
                     limit: "int | None" = 20, worst: int = 2, n_seeds: int = 32,
                     n_alts: int = 2, followup: str = "random",
                     concurrency: int = 1, include_decisions: bool = False) -> dict:
        """RUN-LEVEL luck-vs-mistake attribution — input to the distributional-critic
        decision (read the **caveats**: this brackets the headroom, it does not measure
        it). Falsify the worst δ-craters of every matching battle (default: losses) that
        carries a ``*_reconstruction.json`` sibling, then aggregate the per-decision
        verdicts, **weighted by crater magnitude** (|anchor δ|), into four levers — a
        MEASUREMENT-TIME attribution at one frozen checkpoint, NOT a partition of
        independent root causes:

        - ``LUCK`` → **aleatoric**: holding both actions fixed, the realized outcome sat
          in the bad tail of the dice distribution. Provably unlucky *for the chosen
          line*. Reducible only by a risk-SENSITIVE policy (CVaR/quantile action
          selection — not built) avoiding a lower-variance line; under today's
          risk-neutral PPO objective it is irreducible. (A distributional critic is a
          *prerequisite* for that, not a fix on its own.)
        - ``MISTAKE`` → **policy_reducible**: a top-k alt provably beat the chosen action
          (paired, z-tested, on MEAN material margin) ⇒ a better action existed at this
          checkpoint. The one *proven* leg.
        - ``NEUTRAL`` → **unattributed**: a real crater the sweep could pin on NEITHER
          luck NOR a better action. NOT proven critic error — "not bad-tail luck" means
          the outcome was TYPICAL, equally consistent with a genuinely-lost position the
          critic was RIGHT to crater on. Splitting it needs the model-based
          V(s)-vs-return calibration probe (a deliberate follow-up, not run here).
        - ``MIXED`` → both a proven better line AND bad dice.

        ``gate.critic_headroom_upper_bound`` = LUCK + NEUTRAL share is an **upper bound**
        on the crater fraction a (distributional) critic could address — it can only
        inflate as the alt-sweep weakens (an unproven mistake falls into LUCK/NEUTRAL),
        and it folds in the unproven ``unattributed`` leg. The only thing this scan
        *proves* is ``policy_reducible`` (MISTAKE). ``weighted_shares`` (|δ|) and
        ``count_shares`` are both reported — a large gap means a few big ambiguous
        craters dominate (anchors are pre-selected by worst δ). Model-free (no
        checkpoint). Matched battles without a reconstruction record are reported under
        ``coverage.n_skipped_no_record`` (never silently dropped). ``concurrency`` > 1
        falsifies battles in parallel — each re-roll spawns Node, so raise it only on an
        IDLE box (it contends with a live training run).
        """
        from collections import Counter

        from main.prober.falsifier import falsify_battle
        from utils.bridge.reconstruction import ReconstructionRecord

        verdicts = ("LUCK", "MISTAKE", "MIXED", "NEUTRAL")
        levers = {"LUCK": "aleatoric", "NEUTRAL": "unattributed",
                  "MISTAKE": "policy_reducible", "MIXED": "mixed"}
        _TIE_EPS = 0.05   # top-two shares within this → dominant_lever is "ambiguous"

        # 1. matching battles, split by whether they carry a reconstruction record.
        matched, jobs, skipped = [], [], []
        for b in self.tree.all_battles():
            if outcome and b.outcome != outcome:
                continue
            if opponent and b.opponent != opponent:
                continue
            if step is not None and b.step != step:
                continue
            matched.append(b)
            recon = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
            if os.path.exists(recon):
                jobs.append((b, recon))
            else:
                skipped.append(_short_id(b))
        n_with_record = len(jobs)
        if limit:
            jobs = jobs[:limit]

        # Pre-warm the summary cache on THIS thread so the worker threads only ever
        # READ the (now-populated) `self._summaries` dict — the parallel work is the
        # Node re-roll, not the JSON load, and this removes the only shared write.
        for b, _ in jobs:
            self._summary(b)

        # 2. falsify each (its time is spent in the Node re-roll subprocess, so
        #    threads overlap real work — bounded by `concurrency`).
        def _run_job(job):
            b, recon = job
            try:
                fb = falsify_battle(
                    ReconstructionRecord.load(recon), self._summary(b), self._npz(b),
                    worst=worst, gamma=self._gamma, n_seeds=n_seeds,
                    n_alts=n_alts, followup=followup, impl=self._impl)
                return (b, fb, None)
            except Exception as e:  # noqa: BLE001 — one battle's failure must not sink the scan
                return (b, None, f"{type(e).__name__}: {e}")

        if concurrency and int(concurrency) > 1 and len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(int(concurrency), len(jobs))) as ex:
                done = list(ex.map(_run_job, jobs))
        else:
            done = [_run_job(j) for j in jobs]

        # 3. aggregate, magnitude-weighted by |anchor δ|. Counters tolerate an
        #    unexpected verdict string (counted + surfaced, never a KeyError crash),
        #    mirroring the per-battle path's `.get` defensiveness.
        counts: "Counter[str]" = Counter()
        weighted: "Counter[str]" = Counter()
        n_decisions = decision_errors = 0
        battle_rows, errors = [], []
        for b, fb, err in done:
            if err is not None:
                errors.append({"battle": _short_id(b), "error": err})
                continue
            worst_dec = None
            for d in fb["decisions"]:
                # Default scan ⇒ all anchors come from select_anchors ⇒ each carries a
                # scored δ, so `w` is never None here; a future caller wiring explicit
                # `invs=` could pass an un-scored anchor (w→0, counted but zero-weight).
                w = abs(d.get("anchor_delta") or 0.0)
                counts[d["verdict"]] += 1
                weighted[d["verdict"]] += w
                n_decisions += 1
                if worst_dec is None or w > abs(worst_dec.get("anchor_delta") or 0.0):
                    worst_dec = d
            decision_errors += len(fb.get("errors", []))
            row = {
                "id": b.summary_path, "short_id": _short_id(b),
                "opponent": b.opponent, "step": b.step,
                "turns": (self._summary(b).get("meta") or {}).get("turns"),
                "verdict_counts": dict(fb["verdict_counts"]),
                "n_decisions": len(fb["decisions"]),
                "worst_decision": None if worst_dec is None else {
                    "inv": worst_dec["inv"], "turn": worst_dec["turn"],
                    "verdict": worst_dec["verdict"],
                    "anchor_delta": worst_dec.get("anchor_delta"),
                    "luck_percentile": worst_dec["luck_percentile"],
                    "best_alternative": worst_dec["best_alternative"],
                },
            }
            if include_decisions:   # full per-decision rows (the calibration probe reads these)
                row["decisions"] = fb["decisions"]
            battle_rows.append(row)

        # 4. shares — |δ|-weighted AND count-based (both reported); fall back to count
        #    weighting (announced) when every δ is ~0 (e.g. placeholder values).
        all_verdicts = list(verdicts) + [v for v in counts if v not in verdicts]
        total_w, total_c = sum(weighted.values()), sum(counts.values())
        wshare = {v: (weighted[v] / total_w if total_w > 1e-9 else 0.0) for v in all_verdicts}
        cshare = {v: (counts[v] / total_c if total_c else 0.0) for v in all_verdicts}
        if total_w > 1e-9:
            weighting, share = "delta", wshare
        elif total_c:
            weighting, share = "uniform_fallback", cshare
        else:
            weighting, share = "none", {v: 0.0 for v in all_verdicts}

        # dominant lever — None on an empty scan OR a near-tie (don't over-claim a
        # coinflip headline; the full shares are exposed for the real read).
        dominant_lever = None
        if total_c:
            ranked = sorted(all_verdicts, key=lambda v: share[v], reverse=True)
            runner = share[ranked[1]] if len(ranked) > 1 else 0.0
            if share[ranked[0]] - runner >= _TIE_EPS:
                dominant_lever = {"verdict": ranked[0], "lever": levers.get(ranked[0], "unknown")}

        gate = {
            "aleatoric": round(share["LUCK"], 4),
            "policy_reducible": round(share["MISTAKE"], 4),
            "unattributed": round(share["NEUTRAL"], 4),
            "mixed": round(share["MIXED"], 4),
            # UPPER BOUND (see caveats): folds the proven-aleatoric LUCK with the
            # unproven `unattributed` leg, and inflates as the alt-sweep weakens.
            "critic_headroom_upper_bound": round(share["LUCK"] + share["NEUTRAL"], 4),
        }
        caveats = [
            f"critic_headroom_upper_bound is an UPPER BOUND, not a measurement. MISTAKE "
            f"(policy_reducible) is assigned ONLY when the shallow sweep — top {n_alts} legal "
            f"alts by logit, a SINGLE re-rolled turn, '{followup}' mid-turn follow-up, ranked by "
            f"MEAN material margin — provably beats the chosen action. Any real-but-unproven "
            f"mistake (lower-ranked alt, multi-turn line, equal-mean lower-variance line) falls "
            f"into aleatoric/unattributed, so the bound only inflates as the search weakens; a "
            f"deeper sweep moves mass into policy_reducible.",
            "unattributed (NEUTRAL) is a RESIDUAL — not proven critic error. 'Not bad-tail luck' "
            "means the bad outcome was TYPICAL for the line, equally consistent with a "
            "genuinely-lost position the critic was RIGHT to crater on. Splitting it into "
            "critic-mean-error vs lost-position needs the model-based V(s)-vs-return calibration "
            "probe, which this model-free scan does NOT run.",
            "aleatoric (LUCK) is reducible by a distributional critic ONLY with a risk-SENSITIVE "
            "policy objective (not built) AND a lower-variance alternative — untested here, since "
            "the alt-sweep ranks by MEAN margin (an equal-mean lower-variance line scores ~0 "
            "advantage). Under today's risk-neutral PPO objective a LUCK crater is irreducible.",
            "The four verdicts are a measurement-time attribution at one frozen checkpoint, not "
            "independent root causes. In actor-critic (PPO) the critic feeds the policy gradient, "
            "so a better critic also reduces MISTAKEs over training — 'policy vs critic' is "
            "first-order, not clean.",
            "Anchors are pre-selected as the worst-δ craters, so |δ|-weighting CONCENTRATES the "
            "gate on the largest craters (a big unattributed crater gets full weight). Compare "
            "weighted_shares to count_shares: a large gap means a few big ambiguous craters dominate.",
        ]
        interp = (
            f"At MOST {gate['critic_headroom_upper_bound'] * 100:.0f}% of crater magnitude is "
            f"POTENTIALLY critic-addressable (aleatoric {gate['aleatoric'] * 100:.0f}% + "
            f"unattributed {gate['unattributed'] * 100:.0f}%) — an UPPER BOUND under a shallow "
            f"mean-only sweep (worst={worst}, {n_alts} alts, {n_seeds} seeds, '{followup}' "
            f"follow-up). {gate['policy_reducible'] * 100:.0f}% is a PROVEN better action "
            f"(policy); MIXED {gate['mixed'] * 100:.0f}%. The unattributed share needs the "
            f"V-vs-return calibration probe to split critic-error from lost-positions; aleatoric "
            f"needs a risk-sensitive policy to be reducible at all. [{weighting} weighting, "
            f"{n_decisions} decisions in {len(battle_rows)} battles]"
        )

        return {
            "run_dir": self.run_dir,
            "filters": {"outcome": outcome, "opponent": opponent, "step": step},
            "params": {"limit": limit, "worst": worst, "n_seeds": n_seeds,
                       "n_alts": n_alts, "followup": followup, "concurrency": concurrency},
            "coverage": {
                "n_matched": len(matched),
                "n_with_record": n_with_record,
                "n_falsified": len(battle_rows),
                "n_capped_by_limit": n_with_record - len(jobs),
                "n_skipped_no_record": len(skipped),
                "n_battle_errors": len(errors),
                "n_decisions": n_decisions,
                "n_decision_errors": decision_errors,
                "skipped_no_record_sample": skipped[:10],
            },
            "weighting": weighting,
            "verdict_counts": {v: counts.get(v, 0) for v in all_verdicts},
            "weighted_shares": {v: round(wshare[v], 4) for v in all_verdicts},
            "count_shares": {v: round(cshare[v], 4) for v in all_verdicts},
            "gate": gate,
            "dominant_lever": dominant_lever,
            "interpretation": interp,
            "caveats": caveats,
            "battles": battle_rows,
            "errors": errors,
        }

    def _value_return_pairs(self, b: BattleTrace) -> "list[tuple[float, float]]":
        """(V(s_i), G_i) per captured decision in battle ``b``: the recorded critic
        value vs the realized discounted return. Empty when the trace has no values."""
        summary = self._summary(b)
        npz = self._npz(b)
        values = npz.get("values")
        invs = summary.get("invocations", [])
        if values is None or not len(invs):
            return []
        rewards = []
        for inv in invs:
            r = (inv.get("outcome") or {}).get("reward")
            rewards.append(r.get("total") if isinstance(r, dict) else r)
        g = _discounted_returns(rewards, self._gamma)
        return [(float(values[i]), float(g[i])) for i in range(min(len(invs), len(values)))]

    def calibration(self, *, outcome: "str | None" = "loss",
                    opponent: "str | None" = None, step: "int | None" = None,
                    limit: "int | None" = 20, worst: int = 2, n_seeds: int = 32,
                    n_alts: int = 2, followup: str = "random", concurrency: int = 8,
                    n_bins: int = 10, overvalue_tau: float = 5.0) -> dict:
        """Critic CALIBRATION probe — resolve ``falsify_scan``'s **unattributed**
        (NEUTRAL) bucket into **`critic_overvalued`** (epistemic — a better /
        distributional critic helps) vs **`lost_position`** (the critic was right;
        only risk-sensitive play helps), by comparing the RECORDED value V(s) to the
        REALIZED discounted return G(s).

        **Selection-aware (the crux).** A loss-conditioned ``V − G`` is biased
        positive *by construction* — losses are the below-V tail of any critic, so
        even a perfectly-calibrated one looks "over-valued" on losses. The baseline is
        therefore a **reliability curve over BOTH wins and losses, binned by V** (not
        by outcome). A NEUTRAL crater is ``critic_overvalued`` only if the critic
        SYSTEMATICALLY over-values at its V-level (reliability ``gap`` > ``overvalue_tau``);
        a crater at a well-calibrated level is ``lost_position`` — a fair draw from a
        correctly-valued spot, where only a risk-sensitive policy (not a better mean)
        would help. **Model-free** (uses the recorded V — no checkpoint); the falsify
        pass that finds the unattributed craters runs at ``concurrency`` (default 8).

        Reads the **caveats**: the reliability curve is over the eval's CAPTURED sample
        (quota over-captures losses → E[G|V] biased low → the over-valuation gap, and
        thus ``critic_overvalued``, is an **UPPER BOUND**); per-crater it uses the
        AGGREGATE curve (not the crater's own one-sample G, which is selection-biased);
        the gold-standard per-crater resolution is a re-roll → policy-rollout → return
        PIT (the true distributional-critic validator), deferred.
        """
        # 1. reliability backbone over wins AND losses at the step (selection-free
        #    in the loss/win sense — binned by V). Uses ALL captured battles, not the
        #    falsify limit, for a dense curve (model-free: recorded V + rewards).
        cal_battles = [b for b in self.tree.all_battles()
                       if (step is None or b.step == step)
                       and (opponent is None or b.opponent == opponent)
                       and b.outcome in ("win", "loss")]
        V, G, Vw, Gw, Vl, Gl = [], [], [], [], [], []
        n_win_b = 0
        for b in cal_battles:
            pr = self._value_return_pairs(b)
            if b.outcome == "win":
                n_win_b += 1
            for v_i, g_i in pr:
                V.append(v_i)
                G.append(g_i)
                (Vw if b.outcome == "win" else Vl).append(v_i)
                (Gw if b.outcome == "win" else Gl).append(g_i)
        bins = _reliability_curve(V, G, n_bins)
        overall = _calibration_stats(V, G)
        # Per-outcome bias makes the SELECTION CONFOUND visible: a calibrated critic
        # shows E[V−G]<0 on wins and >0 on losses (the residual is biased toward the
        # realized outcome). The captured sample's win-fraction ≠ the true win rate
        # (eval quota), so the UNCONDITIONAL bias/curve are selection-skewed — read it
        # against these splits, not as raw miscalibration.
        overall["bias_on_wins"] = _calibration_stats(Vw, Gw)["bias"]
        overall["bias_on_losses"] = _calibration_stats(Vl, Gl)["bias"]
        overall["captured_win_fraction"] = round(n_win_b / max(1, len(cal_battles)), 4)

        # 2. the unattributed (NEUTRAL) craters via falsify (concurrency on the re-rolls).
        fs = self.falsify_scan(outcome=outcome, opponent=opponent, step=step,
                               limit=limit, worst=worst, n_seeds=n_seeds, n_alts=n_alts,
                               followup=followup, concurrency=concurrency,
                               include_decisions=True)
        by_path = {b.summary_path: b for b in self.tree.all_battles()}

        crater_w = {"critic_overvalued": 0.0, "lost_position": 0.0}
        gaps_w, n_unattr, n_no_gap, examples = [], 0, 0, []
        for row in fs["battles"]:
            vals = self._npz(by_path[row["id"]]).get("values") if row["id"] in by_path else None
            for d in row.get("decisions", []):
                if d["verdict"] != "NEUTRAL":
                    continue
                n_unattr += 1
                i = d["inv"]
                v_i = float(vals[i]) if vals is not None and i < len(vals) else None
                gap = _reliability_gap_at(bins, v_i) if v_i is not None else None
                if gap is None:
                    n_no_gap += 1
                    continue
                w = abs(d.get("anchor_delta") or 0.0)
                bucket = "critic_overvalued" if gap > overvalue_tau else "lost_position"
                crater_w[bucket] += w
                gaps_w.append((gap, w))
                if len(examples) < 10:
                    examples.append({
                        "id": row["short_id"], "inv": i, "turn": d.get("turn"),
                        "v": round(v_i, 2), "reliability_gap": round(gap, 2),
                        "anchor_delta": d.get("anchor_delta"), "bucket": bucket})

        tot_w = crater_w["critic_overvalued"] + crater_w["lost_position"]
        overvalued_share = crater_w["critic_overvalued"] / tot_w if tot_w > 1e-9 else 0.0
        mean_gap = (sum(g * w for g, w in gaps_w) / sum(w for _, w in gaps_w)) if gaps_w else 0.0
        unattr = fs["gate"]["unattributed"]   # the |δ|-weighted NEUTRAL share from falsify
        # refine the falsify gate: split the unattributed bucket into epistemic vs lost.
        gate = {
            "policy_reducible": fs["gate"]["policy_reducible"],   # PROVEN (falsify)
            "aleatoric": fs["gate"]["aleatoric"],                 # PROVEN dice (falsify)
            "unattributed": round(unattr, 4),
            "unattributed_critic_overvalued": round(unattr * overvalued_share, 4),
            "unattributed_lost_position": round(unattr * (1.0 - overvalued_share), 4),
            # the critic-MEAN-reducible estimate (epistemic) — an UPPER BOUND (see caveats).
            "critic_mean_reducible_upper_bound": round(unattr * overvalued_share, 4),
        }
        interp = (
            f"SELECTION-CONFOUNDED — read the splits, not the headline. Captured win-fraction "
            f"{overall['captured_win_fraction']} (≠ true win rate; eval quota). Bias E[V−G] splits "
            f"{_r(overall['bias_on_wins'])} on WINS vs {_r(overall['bias_on_losses'])} on LOSSES "
            f"(unconditional {_r(overall['bias'])}) — the <0-on-wins / >0-on-losses pattern is the "
            f"signature of a CALIBRATED critic, so the unconditional over-valuation is largely the "
            f"capture quota over-sampling losses, NOT miscalibration. explained-variance(G|V)="
            f"{_r(overall['ev'])}, slope={_r(overall['slope'])}. Against this skewed baseline ~"
            f"{overvalued_share * 100:.0f}% of the unattributed {unattr * 100:.0f}% sits at "
            f"'over-valued' V-levels ⇒ critic_mean_reducible ~{gate['critic_mean_reducible_upper_bound'] * 100:.0f}% "
            f"of total crater mass — a LOOSE UPPER BOUND inflated by the selection skew; the real "
            f"epistemic share needs true-win-rate reweighting or the rollout-PIT. [{n_unattr} craters, "
            f"reliability over {overall['n']} decisions]"
        )
        caveats = [
            "SELECTION CONFOUND (dominant): the reliability curve is over the eval's CAPTURED sample, "
            "whose win-fraction (captured_win_fraction) is set by the capture quota, NOT the true win "
            "rate. Since the model wins most games, losses are over-captured → E[G|V] biased LOW → the "
            "unconditional bias and every bin's gap are inflated. The bias_on_wins (<0) / bias_on_losses "
            "(>0) split is the CALIBRATED-critic signature; the unconditional E[V−G] ≈ P(win)·bias_win + "
            "P(loss)·bias_loss, which is ~0 for a calibrated critic at the true win rate. So "
            "critic_mean_reducible is a LOOSE upper bound until reweighted to the true win rate (or "
            "replaced by the selection-free rollout-PIT).",
            "critic_mean_reducible is an UPPER BOUND: the reliability curve is built over the eval's "
            "CAPTURED sample, whose win/loss mix is set by the capture quota (losses over-captured), so "
            "E[G|V] is biased LOW and the over-valuation gap is overstated. Reweighting to the true "
            "per-opponent win rate would tighten it.",
            "Per crater the split uses the AGGREGATE reliability gap at its V-level, NOT the crater's own "
            "realized G (one draw, selection-biased on losses). So a single crater's label is a "
            "population inference, not a per-state measurement.",
            "lost_position means the critic's MEAN was right; such a crater is reducible only by a "
            "risk-SENSITIVE policy (a distributional critic's other use), not by a better mean estimate.",
            "The gold-standard per-crater resolution is re-roll → POLICY ROLLOUT to terminal → the return "
            "distribution → PIT (where V sits in it) — the true distributional-critic validator. This "
            "model-free reliability version is its cheap aggregate proxy; the rollout primitive is deferred.",
            f"overvalue_tau={overvalue_tau} (return units) sets the over-valued cutoff; mean_gap and the "
            "reliability curve are reported so the threshold isn't the whole story.",
        ]
        return {
            "run_dir": self.run_dir,
            "filters": {"outcome": outcome, "opponent": opponent, "step": step},
            "params": {"limit": limit, "worst": worst, "n_seeds": n_seeds, "n_alts": n_alts,
                       "followup": followup, "concurrency": concurrency, "n_bins": n_bins,
                       "overvalue_tau": overvalue_tau},
            "overall_calibration": overall,
            "reliability_curve": bins,
            "unattributed_resolution": {
                "n_unattributed_craters": n_unattr,
                "n_without_gap": n_no_gap,
                "overvalued_share_of_unattributed": round(overvalued_share, 4),
                "mean_reliability_gap": round(mean_gap, 4),
                "examples": examples,
            },
            "gate": gate,
            "falsify_gate": fs["gate"],
            "falsify_coverage": fs["coverage"],
            "interpretation": interp,
            "caveats": caveats,
        }
