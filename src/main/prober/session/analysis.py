"""The per-decision deep read: `analyze` (loads the resolved model) and `find`."""

from __future__ import annotations

from dataclasses import asdict

from main.prober.engine import (analyze_invocation, build_opp_intent, opp_intent_text,
    summary_flags, surprise_phrase, timeline_entry_text)
from main.prober.session.serialize import _choice_dict


class _AnalysisMixin:
    def analyze(self, battle_id: str, inv_index: int) -> dict:
        """Full forensic analysis of one decision as a JSON-serializable dict
        (faithfulness, matchups, intervention, saliency, value+TD, outcome, model
        disagreement, belief/spread truth). Loads the exact→nearest→recent model."""
        b = self._battle(battle_id)
        model, choice = self._model_for(b)
        opp = self._opp_team_details(b)
        # Extractor kwargs the current code no longer accepts, dropped so this checkpoint could load
        # at all (`ProbeModel.load`). Non-empty ⇒ the rebuilt extractor is NOT bit-identical to the
        # one that played, so faithfulness is approximate — it rides `model_resolution` because that
        # is already the block a surface reads to say WHICH model produced these numbers.
        dropped = tuple(getattr(model, "dropped_kwargs", ()) or ())
        a = analyze_invocation(model, self._summary(b), self._npz(b), inv_index,
                               summary_path=b.summary_path, npz_path=b.npz_path,
                               our_hp_types=self._our_hp_types(b),
                               opp_team=(opp[0] if opp else None),
                               opp_team_details=(opp[1] if opp else None),
                               our_team_details=self._our_team_details(b))
        d = asdict(a)
        d["model_resolution"] = dict(_choice_dict(choice), dropped_kwargs=list(dropped))
        # WHICH READOUT IS THE CRITIC. Under `--critic winprob` `value` and `win_prob` below are
        # the SAME number, so a surface must be able to say so instead of presenting them as two
        # estimators that happen to agree on every decision ever rendered.
        d["critic_currency"] = self.critic_currency()
        d["protocol"] = self._protocol_for(b, d.get("turn", 0))   # raw Showdown log for this turn
        if d.get("value"):  # add the TD residual the engine (γ-agnostic) can't
            reward = (d.get("outcome") or {}).get("reward")
            rtotal = reward.get("total") if isinstance(reward, dict) else reward
            d["value"]["td_residual"] = self._td(
                rtotal, d["value"]["recorded"], d["value"]["next_recorded"])
            # ...and its plain-language gloss. The rule is that the ML term never appears without
            # it, so the pairing has to live where the number is produced, not in each renderer.
            if d["value"]["td_residual"] is not None:
                d["value"]["td_phrase"] = surprise_phrase(d["value"]["td_residual"])
        # The engine's RENDERED sentences, attached exactly as `battle_turns` attaches them.
        # `asdict` alone hands a consumer only the structured fields, so a second renderer would
        # re-derive the battle-log line and the α/β sentence — the drift the engine/renderer split
        # exists to prevent, and what `timeline_entry_text`'s own docstring says it exists to stop.
        # (The TUI got away without this by rendering its own Rich version from the same vocabulary;
        # a plain-text surface has no such excuse.)
        tl = (d.get("outcome") or {}).get("timeline")
        if tl:
            d["outcome"]["timeline"] = [dict(e, text=timeline_entry_text(e)) for e in tl]
        if d.get("opp_intent"):
            d["opp_intent"]["text"] = opp_intent_text(build_opp_intent(self._summary(b)
                                                                      ["invocations"][inv_index]))
        return d

    def find(self, battle_id: str, criterion: str, limit: "int | None" = None) -> "list[int]":
        """Invocation indices matching a criterion (most-relevant first for ranked
        ones), optionally capped to `limit`:

        - flags (model-free): ``switch`` / ``uncertain`` / ``faint``
        - value (model-free): ``value_drop`` (most negative ΔV) / ``low_value`` /
          ``high_value``
        - ``disagree`` (loads the model): chosen ≠ the model's argmax
        """
        b = self._battle(battle_id)
        summary = self._summary(b)
        n = len(summary["invocations"])

        if criterion == "disagree":
            model, _ = self._model_for(b)
            npz = self._npz(b)
            hits = [i for i in range(n)
                    if (a := analyze_invocation(model, summary, npz, i)).has_state and not a.agrees]
        elif criterion in ("value_drop", "low_value", "high_value"):
            values = self._values(b)
            scored = []
            for i in range(n):
                v, v_next = self._v(values, i), self._v(values, i + 1)
                if criterion == "value_drop" and v is not None and v_next is not None:
                    scored.append((v_next - v, i))            # ascending → biggest drops first
                elif criterion == "low_value" and v is not None:
                    scored.append((v, i))                     # ascending → lowest first
                elif criterion == "high_value" and v is not None:
                    scored.append((-v, i))                    # descending → highest first
            hits = [i for _, i in sorted(scored)]
        else:  # a model-free flag
            hits = [i for i, inv in enumerate(summary["invocations"]) if criterion in summary_flags(inv)]

        return hits[:limit] if limit else hits
