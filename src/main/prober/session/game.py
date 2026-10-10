"""`/game`'s MODEL session methods (`designs/prober/battle_view_v2.md`; the model-free story and board
are `session/story.py`):

* `battle_readout` — loads the battle's checkpoint (the one exact → nearest → recent ladder) and
  runs ONE batched eager forward over every recorded decision (`ProbeModel.capture_battle`): opponent
  intent + its calibration, the hypothesis tokens and their evolution, the pointer head's scores, the
  operator facts, the attention summary, the re-run win probability, the scouting notes (the model's
  per-opponent-mon move / item / spread / Hidden-Power-type beliefs vs the truth, `engine.scouting`). CURRENT ARCHITECTURE ONLY — an
  older checkpoint raises the existing `ArchDriftError`, which a surface renders as a sentence.
* `decision_attention` — one decision's full [T,T] attention map (a layer/head, or their average)
  with its seat labels, read from the same cached capture.

The capture is cached in memory per (checkpoint path, battle) — bounded, dropped by ``close()``.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from main.prober.engine.readout import (action_display, attention_matrix, attention_summary, belief_evolution,
    hypotheses_view, intent_calibration, intent_view, operator_view, opp_actual_action, token_labels)
from main.prober.engine.scouting import reveal_notes, scouting_view
from main.prober.session.serialize import _choice_dict, _short_id
from main.prober.session.story import GLOSSARY  # noqa: F401 — the one glossary, re-exported

#: Battle captures one session keeps (each ≈ 15 MB on a 250-decision battle, mostly attention).
_READOUT_CACHE_CAP = 8


def _join_reveal_notes(rows: list, story_turns: list) -> None:
    """The EVIDENCE beside a belief that moved: on every scouting card whose belief changed since the previous
    decision (`delta`), ``delta["after"]`` = the public events touching that mon between the previous
    decision's turn and this one's (`engine.scouting.reveal_notes`, from the model-free story)."""
    for k in range(1, len(rows)):
        t_prev, t_cur = int(rows[k - 1]["turn"] or 0), int(rows[k]["turn"] or 0)
        for mon in (rows[k].get("scouting") or {}).get("mons") or ():
            d = mon.get("delta") or {}
            if d.get("moves") or d.get("item"):
                d["after"] = reveal_notes(story_turns, t_prev, max(t_cur, t_prev + 1), mon["species"])


class _GameMixin:
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
        attention summary, the operator facts, the SCOUTING notes (`engine.scouting.scouting_view`, its
        ``delta`` against the previous decision's notes). Battle level: the re-run P(win) series, α's calibration,
        the belief evolution, the token layout."""
        b = self._battle(battle_id)
        cap, choice = self._capture(b)
        summary = self._summary(b)
        invs = summary.get("invocations") or []
        opp = self._opp_team_details(b)
        opp_ids = [str(s) for s in (opp[0] if opp else ())]
        opp_details = opp[1] if opp else None
        n = int(cap["probs"].shape[0])
        rows, intents = [], []
        scout = None
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
            scout = scouting_view(cap, i, opp_details, prev=scout)
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
                "operator": operator_view(cap, i, labels, legal),
                "scouting": scout})
        _join_reveal_notes(rows, self.battle_story(battle_id)["turns"])
        return {"id": b.summary_path, "short_id": _short_id(b), "model_resolution": _choice_dict(choice),
                "n_decisions": n, "decisions": rows,
                "win_prob_series": ([{"inv": i, "turn": rows[i]["turn"], "p": rows[i]["win_prob"]}
                                     for i in range(n)] if "win_prob" in cap else None),
                "intent_calibration": intent_calibration(intents),
                "belief_evolution": belief_evolution(cap),
                "layout": cap.get("layout"), "n_layers": cap.get("n_layers"),
                "has": {k: (k in cap) for k in ("attention", "intent_p", "slot_species", "op", "mr_move",
                                                "query_attention", "scores", "win_prob",
                                                "belief_move_nums", "belief_item_nums", "belief_spread",
                                                "belief_nature_nums", "belief_hp_type")},
                "glossary": GLOSSARY}

    def decision_attention(self, battle_id: str, inv: int, layer: "int | None" = None,
                           head: "int | None" = None) -> dict:
        """One decision's attention map: ``matrix`` [T][T] (query row × key column) for one layer/head,
        or averaged over whichever is None, with each seat's plain-word ``labels`` and which keys were
        padded (``masked``). Read from the cached capture (`battle_readout`'s forward)."""
        b = self._battle(battle_id)
        cap, _choice = self._capture(b)
        if "attention" not in cap:
            raise ValueError("this model exposes no trunk attention (no trunk round (BiasedEncoderLayer / IdentityInitRound) was captured)")
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
