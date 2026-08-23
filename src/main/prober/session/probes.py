"""Representation + behavioural probes: `probe`, `switch_vs_info`, `history_saliency`."""

from __future__ import annotations

import numpy as np

from main.prober.engine import fit_probe, history_slot_saliency, parse_pct
from main.prober.session.probe_targets import _PROBE_TARGETS
from main.prober.session.serialize import _choice_dict


class _ProbesMixin:
    # -- representation probing ----------------------------------------------

    def _probe_ctx(self, inv: dict, npz: dict, i: int, teams: dict) -> dict:
        our = inv.get("our") or {}
        opp = inv.get("opp") or {}
        ocour = (inv.get("outcome") or {}).get("our") or {}

        def pp(s):
            try:
                return parse_pct(s)
            except Exception:  # noqa: BLE001
                return None

        return {"phase": inv.get("phase"), "our_species": our.get("species"),
                "opp_species": opp.get("species"), "our_hp": pp(our.get("hp")),
                "our_dhp": pp(ocour.get("hp_delta")), "belief": self._belief_at(npz, i),
                "opp_action": ((inv.get("outcome") or {}).get("opp") or {}).get("action"),
                "teams": teams}

    def probe(self, target: str, *, step: "int | None" = None, opponent: "str | None" = None,
              which: str = "vf", max_decisions: int = 1500, seed: int = 0) -> dict:
        """Fit a linear probe on the model's INTERNAL activations to test whether a derived
        quantity (``is_faster`` / ``damage_taken`` / ``faint_soon``) is already in the
        representation — the decisive "do we already have this info or should we hand it over"
        test. Loads the model ONCE (step → one checkpoint). Compares the representation probe to
        a baseline probe on the raw obs/belief feature we ALREADY provide, and breaks both down by
        easy-vs-contested group. Returns JSON-serializable; ``error`` key on too-few-labels."""
        spec = _PROBE_TARGETS.get(target)
        if spec is None:
            raise ValueError(f"unknown probe target {target!r}; choices: {sorted(_PROBE_TARGETS)}")
        if which not in ("vf", "pi"):
            raise ValueError("which must be 'vf' (value head) or 'pi' (policy head)")
        battles = self.tree.all_battles()
        if step is None:
            steps = sorted({b.step for b in battles})
            step = steps[-1] if steps else None
        battles = [b for b in battles
                   if (step is None or b.step == step) and (not opponent or b.opponent == opponent)]

        X, y, groups, provided = [], [], [], []
        model = choice = None
        for b in battles:
            if len(y) >= max_decisions:
                break
            try:
                if model is None:
                    model, choice = self._model_for(b)
            except FileNotFoundError as e:
                return {"target": target, "step": step, "error": f"no model: {e}"}
            summary, npz = self._summary(b), self._npz(b)
            obsmat = npz.get("obs")
            if obsmat is None:
                continue
            teams = summary.get("teams") or {}
            adim = int(npz["logits"].shape[1]) if npz.get("logits") is not None else 11
            for i, inv in enumerate(summary.get("invocations", [])):
                if len(y) >= max_decisions:
                    break
                if i >= len(obsmat):
                    continue
                ctx = self._probe_ctx(inv, npz, i, teams)
                lab = spec["label"](ctx)
                if lab is None:
                    continue
                feats = model.features(obsmat[i], np.ones(adim, dtype=np.int8))
                X.append(feats[which])
                y.append(float(lab))
                groups.append(spec["group"](ctx))
                provided.append(spec["provided"](ctx))

        if len(y) < 30:
            return {"target": target, "step": step, "n_decisions": len(y),
                    "error": "too few labeled decisions (<30) — widen step/opponent or max_decisions"}

        rep = fit_probe(X, y, spec["task"], groups=groups, seed=seed)
        # Baseline: how well does the RAW obs/belief feature we ALREADY provide predict the label?
        # The decisive comparison — the representation is "more than the feature" only if it beats this.
        prov_idx = [j for j, p in enumerate(provided) if isinstance(p, (int, float))]
        prov_report = None
        if len(prov_idx) >= 30:
            prov_report = fit_probe([[provided[j]] for j in prov_idx], [y[j] for j in prov_idx],
                                    spec["task"], groups=[groups[j] for j in prov_idx], seed=seed)
        return {
            "run_dir": self.run_dir, "step": step, "opponent": opponent, "target": target,
            "task": spec["task"], "which_features": which, "n_decisions": len(y),
            "model_resolution": _choice_dict(choice) if choice else None,
            "tests": spec["tests"], "how_to_read": spec["how_to_read"], "caveat": spec["caveat"],
            "representation_probe": rep,
            "provided_feature": spec["provided_name"], "provided_feature_baseline": prov_report,
        }


    @staticmethod
    def _revealed_opp_count(inv: dict) -> "int | None":
        """How many opponent mons we'd SEEN by this decision = active + revealed bench
        (the trace's opp.bench string, e.g. 'tyranitar(100%), salamence(50%)'). This is the
        information-level signal for the switch-vs-uncertainty hypothesis."""
        opp = inv.get("opp") or {}
        if not opp.get("species"):
            return None
        # Each bench mon is rendered 'species(hp%[,STATUS])' — one '(' apiece. Counting '('
        # is robust to the comma INSIDE the HP/status field (e.g. 'salamence(64%,TOX)'),
        # which a naive comma-split over-counts.
        bench = opp.get("bench") or ""
        n_bench = bench.count("(")
        return min(6, 1 + n_bench)

    def switch_vs_info(self, *, step: "int | None" = None, opponent: "str | None" = None,
                       outcome: "str | None" = None, max_battles: int = 400) -> dict:
        """MODEL-FREE behavioural probe: does OUR policy switch more when it knows LESS about the
        opponent? For every VOLUNTARY decision (phase != forced_switch) it records whether we
        switched and how many opp mons we'd revealed (1–6), then reports the voluntary switch-rate
        by revealed-count bucket + a correlation, plus the double-switch rate (we switch right after
        the opp switched, and we switch on consecutive own decisions). Reads traces only — no model.

        Hypothesis (user): low information ⇒ higher switch-rate (scout/pivot under uncertainty). A
        rising switch-rate as revealed-count FALLS confirms the policy's switching is
        information-sensitive; a flat curve says it is information-blind."""
        battles = self.tree.all_battles()
        if step is None:
            steps = sorted({b.step for b in battles})
            step = steps[-1] if steps else None
        battles = [b for b in battles
                   if (step is None or b.step == step)
                   and (not opponent or b.opponent == opponent)
                   and (not outcome or b.outcome == outcome)]

        by_count: dict = {}          # revealed_count -> [n_decisions, n_switch]
        n_vol = n_switch = 0
        dbl_after_opp_switch = opp_switch_opportunities = 0
        consec_switch = consec_pairs = 0
        xs, ys = [], []              # (revealed_count, switched) for correlation
        used = 0
        for b in battles:
            if used >= max_battles:
                break
            summary = self._summary(b)
            invs = summary.get("invocations", [])
            prev_switch = None
            prev_opp_switched = None
            saw = False
            for inv in invs:
                if inv.get("phase") == "forced_switch":
                    prev_switch = None  # a forced replacement breaks the voluntary chain
                    prev_opp_switched = None
                    continue
                chosen = inv.get("chosen") or ""
                switched = 1 if str(chosen).startswith("switch") else 0
                rc = self._revealed_opp_count(inv)
                if rc is None:
                    continue
                saw = True
                n_vol += 1
                n_switch += switched
                by_count.setdefault(rc, [0, 0])
                by_count[rc][0] += 1
                by_count[rc][1] += switched
                xs.append(float(rc))
                ys.append(float(switched))
                # double-switch signals
                if prev_opp_switched is not None:
                    opp_switch_opportunities += 1 if prev_opp_switched else 0
                    if prev_opp_switched and switched:
                        dbl_after_opp_switch += 1
                if prev_switch is not None:
                    consec_pairs += 1
                    if prev_switch and switched:
                        consec_switch += 1
                prev_switch = switched
                a = ((inv.get("outcome") or {}).get("opp") or {}).get("action") or ""
                prev_opp_switched = a.startswith("switched_to")
            if saw:
                used += 1

        def _rate(pair):
            return round(pair[1] / pair[0], 4) if pair[0] else None
        corr = None
        if len(xs) >= 30:
            xv, yv = np.asarray(xs), np.asarray(ys)
            if xv.std() > 0 and yv.std() > 0:
                corr = round(float(np.corrcoef(xv, yv)[0, 1]), 4)
        return {
            "run_dir": self.run_dir, "step": step, "opponent": opponent, "outcome": outcome,
            "n_voluntary_decisions": n_vol,
            "overall_voluntary_switch_rate": round(n_switch / n_vol, 4) if n_vol else None,
            "switch_rate_by_revealed_opp_count": {
                str(k): {"n": by_count[k][0], "switch_rate": _rate(by_count[k])}
                for k in sorted(by_count)
            },
            "corr_revealed_count_vs_switch": corr,
            "hypothesis": ("user: switch MORE when we know LESS → expect switch_rate to FALL as "
                           "revealed_opp_count RISES, i.e. a NEGATIVE correlation. ⚠️ The raw "
                           "correlation here is a CONFOUNDED NULL: revealed_opp_count is monotone "
                           "with game progress, so opportunity (mons-alive/trapped) and threat-info "
                           "push switch-rate in OPPOSITE directions vs the count (a Simpson erasure). "
                           "A ~0 corr does NOT mean information-blind — see double_switch below, where "
                           "switch-rate jumps 2.5× the turn after the opp switches (the policy clearly "
                           "DOES condition on opponent action). To test information-sensitivity "
                           "properly, regress switch ~ revealed_count + turn + mons_alive (forced/"
                           "trapped excluded) and read the PARTIAL coefficient, or condition on the "
                           "incoming-P(KO) threat belief within fixed (turn, mons-alive) cells."),
            "double_switch": {
                "rate_after_opp_switch": _rate([opp_switch_opportunities, dbl_after_opp_switch]),
                "n_after_opp_switch_opportunities": opp_switch_opportunities,
                "consecutive_own_switch_rate": _rate([consec_pairs, consec_switch]),
                "n_consecutive_decision_pairs": consec_pairs,
            },
        }

    def history_saliency(self, *, step: "int | None" = None, opponent: "str | None" = None,
                         max_decisions: int = 400) -> dict:
        """Per-turn-slot saliency of the turn-history block for BOTH heads — to decide whether the
        OLDER history turns carry enough signal to keep, or whether N_HISTORY_TURNS can be shortened
        to reclaim obs-build + attention compute. Loads the model once; reports each slot's policy and
        value mean|grad|, normalized by the overall obs mean|grad| (1.0 = an average obs dim)."""
        battles = self.tree.all_battles()
        if step is None:
            steps = sorted({b.step for b in battles})
            step = steps[-1] if steps else None
        battles = [b for b in battles
                   if (step is None or b.step == step) and (not opponent or b.opponent == opponent)]
        model = choice = None
        pol_acc = val_acc = None
        pol_overall = val_overall = 0.0
        npts = 0
        for b in battles:
            if npts >= max_decisions:
                break
            try:
                if model is None:
                    model, choice = self._model_for(b)
            except FileNotFoundError as e:
                return {"error": f"no model: {e}", "step": step}
            npz = self._npz(b)
            obsmat = npz.get("obs")
            if obsmat is None:
                continue
            adim = int(npz["logits"].shape[1]) if npz.get("logits") is not None else 11
            for i in range(len(self._summary(b).get("invocations", []))):
                if npts >= max_decisions:
                    break
                if i >= len(obsmat):
                    continue
                obs, mask = obsmat[i], np.ones(adim, dtype=np.int8)
                probs, _ = model.action_dist(obs, mask)
                pg = model.logit_grad(obs, mask, int(np.argmax(probs)))   # top action's logit saliency
                vg = model.value_grad(obs, mask)                          # critic saliency
                ps, vs = history_slot_saliency(pg, model.offsets), history_slot_saliency(vg, model.offsets)
                if not ps or not vs:
                    continue
                if pol_acc is None:
                    pol_acc, val_acc = np.zeros(len(ps)), np.zeros(len(vs))
                pol_acc += np.asarray(ps)
                val_acc += np.asarray(vs)
                pol_overall += float(np.abs(pg).mean())
                val_overall += float(np.abs(vg).mean())
                npts += 1
        if not npts or pol_acc is None:
            return {"error": "no decisions with a turn-history block", "step": step, "n_decisions": npts}
        pol, val = pol_acc / npts, val_acc / npts
        pol_o, val_o = (pol_overall / npts) or 1.0, (val_overall / npts) or 1.0
        slots = [{"slot": i,
                  "policy_saliency_norm": round(float(pol[i] / pol_o), 3),
                  "value_saliency_norm": round(float(val[i] / val_o), 3)} for i in range(len(pol))]
        return {
            "run_dir": self.run_dir, "step": step, "opponent": opponent, "n_decisions": npts,
            "n_history_turns": len(pol), "model_resolution": _choice_dict(choice) if choice else None,
            "note": ("per-turn-slot mean|grad|, normalized by the overall obs mean|grad| (1.0 = an "
                     "average obs dim). Slot index is obs order — the transformer's positional "
                     "embedding learns recency. A contiguous run of LOW slots (≪1.0) at one end = "
                     "those turns are ~ignored → a candidate to shorten N_HISTORY_TURNS and reclaim compute."),
            "slots": slots,
        }
