"""`/game`'s MODEL-FREE half (`designs/prober/battle_viewer_ux_2026-10-09.md` §3–§5, §9):

* `battle_story` — every turn's protocol events grouped into ordered BEATS (phases, who moved first,
  forced replacements, faint causes, damage as before → after), the turn rail's per-side summary, the
  recorded P(win) at each decision, and the decisions with our choice against the legal set. Works
  on every run with traces, at any architecture.
* `battle_board` — the board ONE decision was made on, as an information-PERSPECTIVE board
  (`engine/perspective.py`): every fact tagged public / ours / hidden, with both teams' sheets from
  the battle's reconstruction record. A move-selection decision sees the start-of-turn board; a
  forced replacement sees the mid-turn board after the faint.

The protocol fold is cached in memory per battle (bounded, dropped by ``close()``).
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict

from main.prober.engine.perspective import perspective_board
from main.prober.engine.protocol import parse_protocol_log
from main.prober.engine.readout import _display_species, action_display
from main.prober.engine.turn_events import fold_turns

#: An action label the recorder writes for an EMPTY request slot.
_PLACEHOLDER = re.compile(r"^move\d$")

#: Battles whose protocol fold one session keeps (each a few hundred KB on a 250-turn battle).
_STORY_CACHE_CAP = 16

#: Plain-word explanations of every abbreviation `/game` prints, served WITH the data so the page and
#: the JSON cannot disagree about what a term means.
GLOSSARY = {
    "α": "the model's prediction of the opponent's next action (the flat opponent pointer)",
    "π": "presence — the model's probability that this species is on the opponent's team",
    "P(win)": "the critic's probability that we win from here — recorded at play time (the win-prob critic)",
    "pts": "percentage POINTS — the difference of two probabilities (62% → 55% is −7 pts)",
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
    "▸": "a move",
    "⇄": "a switch by choice",
    "↳": "sent in after a faint (a forced replacement)",
    "✕": "fainted",
    "⏸": "couldn't move",
    "◇": "ground truth — hidden from the model at that point (shown only under + Truth)",
    "spread band": "where a believed stat sits in the species' possible range at level 100: the bottom "
                   "third reads low, the top third high (a display rule, not a model output)",
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


def _assign_decisions(turn: dict, decisions: list) -> None:
    """Each decision of the turn → the beat it produced: the move-selection one → our first action,
    each forced replacement → our replace beats in order. A move-selection decision that produced NO
    action (our mon fainted before it moved) is recorded as ``unexecuted`` on the turn."""
    beats = turn.get("beats") or []
    ours = [b for b in beats if b.get("side") == "we"]
    replaces = [b for b in ours if b["kind"] == "replace"]
    action = next((b for b in ours if b["kind"] in ("move", "cant", "switch")), None)
    for d in decisions:
        if d.get("phase") == "forced_switch":
            if replaces:
                b = replaces.pop(0)
                b["decision"], d["beat"] = d["inv"], b["index"]
        elif action is not None and "decision" not in action:
            action["decision"], d["beat"] = d["inv"], action["index"]
        elif action is None and d.get("phase") == "move_selection":
            fainted = any(e.get("kind") == "faint" and e.get("side") == "we" for b in beats for e in b["effects"])
            turn["unexecuted"] = {"inv": d["inv"], "chosen": d.get("chosen_display") or d.get("chosen"),
                                  "species": d.get("our_display") or d.get("our"), "fainted_first": fainted}


class _StoryMixin:
    # ------------------------------------------------------------------ the fold (cached)
    def _battle_protocol(self, b) -> tuple:
        replay = b.summary_path[: -len("_summary.json")] + "_replay.html"
        if os.path.exists(replay):
            with open(replay, encoding="utf-8") as f:
                return parse_protocol_log(f.read())
        return tuple(self._core_protocol(b))

    def _story_fold(self, b) -> dict:
        cache = self.__dict__.setdefault("_story_cache", OrderedDict())
        hit = cache.get(b.summary_path)
        if hit is not None:
            cache.move_to_end(b.summary_path)
            return hit
        summary = self._summary(b)
        lines = self._battle_protocol(b)
        side = _trainee_side(summary, lines)
        our_team = [m.get("species") for m in ((summary.get("teams") or {}).get("our") or [])
                    if isinstance(m, dict)]
        opp = self._opp_team_details(b)
        hit = {"side": side, "turns": fold_turns(lines, trainee_side=side, our_team=our_team),
               "our_details": self._our_team_details(b), "opp_details": opp[1] if opp else None}
        cache[b.summary_path] = hit
        while len(cache) > _STORY_CACHE_CAP:
            cache.popitem(last=False)
        return hit

    # ------------------------------------------------------------------ the story
    def battle_story(self, battle_id: str) -> dict:
        """The turn story (model-free): ``turns`` = per game turn its typed protocol events, the
        ordered ``beats`` they form, the rail's per-side ``summary``, the board at the END of the turn
        and the decision indices made at it; ``decisions`` = per recorded decision the turn, phase,
        our choice and the legal set with the recorded probabilities, the recorded P(win) and the beat
        it produced. Works at any architecture."""
        b = self._battle(battle_id)
        bt = self.battle_turns(battle_id)
        fold = self._story_fold(b)
        currency = bt.get("critic_currency") or {}
        is_prob = bool(currency.get("is_probability"))
        decisions = []
        for t in bt.get("turns") or []:
            for d in t.get("decisions") or []:
                decisions.append({
                    "inv": d["inv"], "turn": d["turn"], "phase": d.get("phase"), "chosen": d.get("chosen"),
                    "chosen_index": next((i for i, a in enumerate(d.get("actions") or []) if a.get("chosen")), None),
                    # A slot the request left EMPTY (a fainted mon's moves on a forced switch) is
                    # recorded as `move0`…`move3`; it is no option at all, so it is marked rather
                    # than listed as "not available".
                    "actions": [dict(a, display=action_display(a.get("label") or ""),
                                     placeholder=bool(_PLACEHOLDER.match(a.get("label") or "")))
                                for a in d.get("actions") or []],
                    "chosen_display": action_display(d.get("chosen") or ""), "value": d.get("value"),
                    "win_prob": d.get("win_prob"),
                    "p_win": d.get("value") if is_prob else d.get("win_prob"),
                    "delta_v": d.get("delta_v"), "flags": d.get("flags") or [],
                    "our": (d.get("our") or {}).get("species"), "opp": (d.get("opp") or {}).get("species"),
                    "our_display": _display_species((d.get("our") or {}).get("species") or ""),
                    "summary": [e.get("text") for e in d.get("timeline") or []]})
        by_turn: dict = {}
        for d in decisions:
            by_turn.setdefault(int(d["turn"] or 0), []).append(d)
        # A decision whose turn the protocol never reached (no log at all on an old trace without a
        # `_replay.html`) still gets its row — an empty one that says so — so no decision is orphaned.
        folded = [{k: v for k, v in t.items() if k != "replace_boards"} for t in fold["turns"]]
        have = {int(t["turn"]) for t in folded}
        folded += [{"turn": n, "events": [], "beats": [], "summary": None, "board": None, "no_protocol": True}
                   for n in sorted(set(by_turn) - have)]
        folded.sort(key=lambda t: int(t["turn"]))
        drops = {r["inv"]: k + 1 for k, r in enumerate((bt.get("notable") or {}).get("biggest_value_drops") or [])
                 if r.get("delta_v") is not None and r["delta_v"] < 0}
        turns = []
        for t in folded:
            ds = by_turn.get(int(t["turn"]), [])
            t = dict(t, beats=[dict(bb, effects=list(bb["effects"])) for bb in t.get("beats") or []],
                     decisions=[d["inv"] for d in ds])
            _assign_decisions(t, ds)
            t["p_win"] = ds[0]["p_win"] if ds else None
            t["drop"] = min((drops[d["inv"]] for d in ds if d["inv"] in drops), default=None)
            t["forced"] = any(d.get("phase") == "forced_switch" for d in ds)
            turns.append(t)
        return {"id": bt["id"], "short_id": bt["short_id"], "step": bt["step"], "opponent": bt["opponent"],
                "outcome": bt["outcome"], "critic_currency": bt.get("critic_currency"),
                "trainee_side": fold["side"], "n_turns": bt.get("n_turns"), "n_decisions": len(decisions),
                "has_truth": fold["opp_details"] is not None,
                "turns": turns, "decisions": decisions, "glossary": GLOSSARY}

    # ------------------------------------------------------------------ one decision's board
    def battle_board(self, battle_id: str, inv: int) -> dict:
        """The board decision ``inv`` was made on, as an information-perspective board: the
        start-of-turn board for a move selection, the mid-turn board after the faint for a forced
        replacement (`engine/perspective.py`; every fact tagged public / ours / hidden)."""
        b = self._battle(battle_id)
        story = self.battle_story(battle_id)
        fold = self._story_fold(b)
        n = len(story["decisions"])
        if not 0 <= int(inv) < n:
            raise IndexError(f"decision {inv} out of range (the battle has {n})")
        d = story["decisions"][int(inv)]
        turn = int(d["turn"] or 0)
        turns = fold["turns"]
        kind, board = "start", None
        if d.get("phase") == "forced_switch":
            same = [x for x in story["decisions"] if int(x["turn"] or 0) == turn and x.get("phase") == "forced_switch"]
            k = [x["inv"] for x in same].index(d["inv"])
            cur = next((t for t in turns if int(t["turn"]) == turn), None)
            ours = [r for r in (cur or {}).get("replace_boards") or [] if r["side"] == "we"]
            if k < len(ours):
                kind, board = "replacement", ours[k]["board"]
        if board is None:
            prev = [t for t in turns if int(t["turn"]) < turn]
            board = prev[-1].get("board") if prev else None
        return {"inv": int(inv), "turn": turn, "kind": kind, "has_truth": fold["opp_details"] is not None,
                "board": perspective_board(board, our_details=fold["our_details"],
                                           opp_details=fold["opp_details"])}
