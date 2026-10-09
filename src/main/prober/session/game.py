"""`/game`'s three session methods (`designs/prober/battle_view_v2.md`):

* `battle_story` — MODEL-FREE: every turn's protocol events + the board after it, and the decisions
  of that turn with our choice against the legal set. Works on every run with traces.
* `battle_readout` — loads the battle's checkpoint (the one exact → nearest → recent ladder) and
  runs ONE batched eager forward over every recorded decision (`ProbeModel.capture_battle`): opponent
  intent + its calibration, the hypothesis tokens and their evolution, the pointer head's scores, the
  operator facts, the attention summary, the re-run win probability. CURRENT ARCHITECTURE ONLY — an
  older checkpoint raises the existing `ArchDriftError`, which a surface renders as a sentence.
* `decision_attention` — one decision's full [T,T] attention map (a layer/head, or their average)
  with its seat labels, read from the same cached capture.

The capture is cached in memory per (checkpoint path, battle) — bounded, dropped by ``close()``.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np

from main.prober.engine.protocol import parse_protocol_log
from main.prober.engine.readout import (action_display, attention_matrix, attention_summary, belief_evolution,
    hypotheses_view, intent_calibration, intent_view, operator_view, opp_actual_action, token_labels)
from main.prober.engine.turn_events import fold_turns
from main.prober.session.serialize import _choice_dict, _short_id

#: Battle captures one session keeps (each ≈ 15 MB on a 250-decision battle, mostly attention).
_READOUT_CACHE_CAP = 8

#: Plain-word explanations of every abbreviation `/game` prints, served WITH the data so the page and
#: the JSON cannot disagree about what a term means.
GLOSSARY = {
    "α": "the model's prediction of the opponent's next action (the flat opponent pointer)",
    "π": "presence — the model's probability that this species is on the opponent's team",
    "P(win)": "the critic's probability that we win from here (the win-prob head)",
    "P(KO)": "the probability the move knocks the target out",
    "P(KO first)": "the probability we knock them out before they act (move-resolution family)",
    "P(resolve)": "the probability the move does what it says — lands, not blocked or immune",
    "P(lands)": "for a status move: the probability the status lands",
    "OTHER_move": "any move of theirs the model did not give a seat of its own",
    "OTHER_species": "a switch to a mon the model has not guessed",
    "E3": "a trunk token for one of OUR active's moves",
    "E4": "a trunk token for one of THEIR active's likely moves",
    "E5": "a trunk token summarising the rest of a mon's moves (beyond its seats)",
    "pointer score": "the raw score the action head gives each action before the softmax",
    "log loss": "how surprised the prediction was by what happened — lower is better",
}


def _trainee_side(summary: dict, lines) -> str:
    src = (summary.get("meta") or {}).get("trace_source") or {}
    if src.get("trainee_side") in ("p1", "p2"):
        return str(src["trainee_side"])
    # An older (python-era) trace: our side's HP lines carry exact points, theirs hundredths.
    for ln in lines or ():
        if ln.startswith("|switch|"):
            p = ln.split("|")
            if len(p) > 4 and "/" in p[4] and not p[4].split("/")[1].startswith("100"):
                return p[2][:2]
    return "p1"


class _GameMixin:
    # ------------------------------------------------------------------ model-free
    def _battle_protocol(self, b) -> tuple:
        replay = b.summary_path[: -len("_summary.json")] + "_replay.html"
        if os.path.exists(replay):
            with open(replay, encoding="utf-8") as f:
                return parse_protocol_log(f.read())
        return tuple(self._core_protocol(b))

    def battle_story(self, battle_id: str) -> dict:
        """The turn story (model-free): ``turns`` = per game turn its typed protocol events, the board at
        the END of the turn, and the decision indices made at it; ``decisions`` = per recorded decision
        the turn, phase, our choice and the legal set with the recorded probabilities. Works at any
        architecture."""
        b = self._battle(battle_id)
        bt = self.battle_turns(battle_id)
        summary = self._summary(b)
        lines = self._battle_protocol(b)
        side = _trainee_side(summary, lines)
        our_team = [m.get("species") for m in ((summary.get("teams") or {}).get("our") or [])
                    if isinstance(m, dict)]
        folded = fold_turns(lines, trainee_side=side, our_team=our_team)
        decisions = []
        for t in bt.get("turns") or []:
            for d in t.get("decisions") or []:
                decisions.append({
                    "inv": d["inv"], "turn": d["turn"], "phase": d.get("phase"), "chosen": d.get("chosen"),
                    "chosen_index": next((i for i, a in enumerate(d.get("actions") or []) if a.get("chosen")), None),
                    "actions": [dict(a, display=action_display(a.get("label") or ""))
                                for a in d.get("actions") or []],
                    "chosen_display": action_display(d.get("chosen") or ""), "value": d.get("value"), "win_prob": d.get("win_prob"),
                    "delta_v": d.get("delta_v"), "flags": d.get("flags") or [],
                    "our": (d.get("our") or {}).get("species"), "opp": (d.get("opp") or {}).get("species"),
                    "summary": [e.get("text") for e in d.get("timeline") or []]})
        by_turn: dict = {}
        for d in decisions:
            by_turn.setdefault(int(d["turn"] or 0), []).append(d["inv"])
        # A decision whose turn the protocol never reached (no log at all on an old trace without a
        # `_replay.html`) still gets its row — an empty one that says so — so no decision is orphaned.
        have = {int(t["turn"]) for t in folded}
        folded = folded + [{"turn": n, "events": [], "board": None, "no_protocol": True}
                           for n in sorted(set(by_turn) - have)]
        folded.sort(key=lambda t: int(t["turn"]))
        turns = [dict(t, decisions=by_turn.get(int(t["turn"]), [])) for t in folded]
        return {"id": bt["id"], "short_id": bt["short_id"], "step": bt["step"], "opponent": bt["opponent"],
                "outcome": bt["outcome"], "critic_currency": bt.get("critic_currency"),
                "trainee_side": side, "n_turns": bt.get("n_turns"), "n_decisions": len(decisions),
                "turns": turns, "decisions": decisions, "glossary": GLOSSARY}

    # ------------------------------------------------------------------ model
    def _capture(self, b):
        model, choice = self._model_for(b)
        cache = self.__dict__.setdefault("_readout_cache", OrderedDict())
        key = (str(choice.path), b.summary_path)
        hit = cache.get(key)
        if hit is None:
            npz = self._npz(b)
            if "obs" not in npz or "action_mask" not in npz:
                raise FileNotFoundError(f"{_short_id(b)}: no stored observations (states.npz) — the model "
                                        "views re-run the policy on them")
            hit = (model.capture_battle(npz["obs"], npz["action_mask"]), choice)
            cache[key] = hit
            while len(cache) > _READOUT_CACHE_CAP:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return hit

    def battle_readout(self, battle_id: str) -> dict:
        """Every model panel of `/game` for one battle (LOADS THE CHECKPOINT; current architecture
        only — an older one raises `ArchDriftError`). Per decision: the policy (probabilities + the
        pointer head's raw scores), the opponent intent + the label, the hypothesis tokens, the
        attention summary, the operator facts. Battle level: the re-run P(win) series, α's calibration,
        the belief evolution, the token layout."""
        b = self._battle(battle_id)
        cap, choice = self._capture(b)
        summary = self._summary(b)
        invs = summary.get("invocations") or []
        opp = self._opp_team_details(b)
        opp_ids = [str(s) for s in (opp[0] if opp else ())]
        n = int(cap["probs"].shape[0])
        rows, intents = [], []
        for i in range(n):
            inv = invs[i] if i < len(invs) else {}
            labels = list((inv.get("actions") or {}).keys())
            legal = [bool(v.get("valid")) for v in (inv.get("actions") or {}).values()]
            chosen = next((a for a, lab in enumerate(labels) if lab == inv.get("chosen")), None)
            teams = cap["teams"][i]
            our_species = str((inv.get("our") or {}).get("species") or "")
            active = next((j for j, s in enumerate(teams.get("our", [])) if s and s == our_species), None)
            iv = intent_view(cap, i, opp_actual_action(inv), opp_ids)
            intents.append(iv)
            tl = token_labels(cap["layout"], teams, cap, i, labels) if "layout" in cap else None
            rows.append({
                "inv": i, "turn": inv.get("turn"), "chosen": inv.get("chosen"), "chosen_index": chosen,
                "policy": [{"action": a, "label": labels[a] if a < len(labels) else f"action {a}",
                            "display": action_display(labels[a]) if a < len(labels) else "",
                            "legal": legal[a] if a < len(legal) else False, "p": float(cap["probs"][i][a]),
                            "score": (float(cap["scores"][i][a]) if "scores" in cap else None),
                            "chosen": a == chosen} for a in range(int(cap["probs"].shape[1]))],
                "win_prob": float(cap["win_prob"][i]) if "win_prob" in cap else None,
                "intent": iv,
                "hypotheses": hypotheses_view(cap, i, opp_ids),
                "attention": attention_summary(cap, i, tl, chosen, active) if tl else None,
                "operator": operator_view(cap, i, labels, legal)})
        return {"id": b.summary_path, "short_id": _short_id(b), "model_resolution": _choice_dict(choice),
                "n_decisions": n, "decisions": rows,
                "win_prob_series": ([{"inv": i, "turn": rows[i]["turn"], "p": rows[i]["win_prob"]}
                                     for i in range(n)] if "win_prob" in cap else None),
                "intent_calibration": intent_calibration(intents),
                "belief_evolution": belief_evolution(cap),
                "layout": cap.get("layout"), "n_layers": cap.get("n_layers"),
                "has": {k: (k in cap) for k in ("attention", "intent_p", "slot_species", "op", "mr_move",
                                                "query_attention", "scores", "win_prob")},
                "glossary": GLOSSARY}

    def decision_attention(self, battle_id: str, inv: int, layer: "int | None" = None,
                           head: "int | None" = None) -> dict:
        """One decision's attention map: ``matrix`` [T][T] (query row × key column) for one layer/head,
        or averaged over whichever is None, with each seat's plain-word ``labels`` and which keys were
        padded (``masked``). Read from the cached capture (`battle_readout`'s forward)."""
        b = self._battle(battle_id)
        cap, _choice = self._capture(b)
        if "attention" not in cap:
            raise ValueError("this model exposes no trunk attention (no BiasedEncoderLayer was captured)")
        n = int(cap["attention"].shape[0])
        if not 0 <= int(inv) < n:
            raise IndexError(f"decision {inv} out of range (the battle has {n})")
        L, H = int(cap["attention"].shape[1]), int(cap["attention"].shape[2])
        if layer is not None and not 0 <= int(layer) < L:
            raise IndexError(f"layer {layer} out of range ({L} layers)")
        if head is not None and not 0 <= int(head) < H:
            raise IndexError(f"head {head} out of range ({H} heads)")
        summary = self._summary(b)
        invs = summary.get("invocations") or []
        labels = list(((invs[int(inv)] if int(inv) < len(invs) else {}).get("actions") or {}).keys())
        tl = token_labels(cap["layout"], cap["teams"][int(inv)], cap, int(inv), labels)
        m = attention_matrix(cap, int(inv), layer, head)
        km = cap.get("key_masked")
        query = None
        if "query_attention" in cap:
            q = cap["query_attention"][int(inv)]
            q = q[int(head)] if head is not None else q.mean(axis=0)
            query = [float(x) for x in q]
        return {"inv": int(inv), "layer": layer, "head": head, "n_layers": L, "n_heads": H,
                "labels": tl, "matrix": np.round(m, 5).tolist(),
                "masked": [bool(x) for x in km[int(inv)]] if km is not None else [False] * len(tl),
                "query": query,
                "note": ("attention is a reading aid, not an explanation: a large weight says that token's "
                         "content was mixed in, not that it caused the choice")}
