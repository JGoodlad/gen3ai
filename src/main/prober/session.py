"""Programmatic access to the probing infrastructure — for agents and scripts.

``ProbeSession`` is a thin, framework-agnostic facade over discovery + engine that
returns **JSON-serializable** dicts, so an agent can investigate a model's
behaviour without the TUI. A typical investigation:

    sess = ProbeSession("models/run_.../")
    sess.run_summary()                       # orient: steps, opponents, win/loss, identity
    sess.battles(outcome="loss", step=8_000_000)   # pick battles to look at
    sess.scan(outcome="loss", opponent="aggressive_v2")  # MODEL-FREE: worst turn PER battle, ranked
    sess.battle_overview(battle_id)          # MODEL-FREE digest: per-decision rows + `notable`
    sess.find(battle_id, "value_drop", limit=5)    # rank decisions by where V(s) cratered
    sess.find(battle_id, "disagree")         # decisions the loaded model disagrees with
    sess.analyze(battle_id, inv)             # full forensic analysis of one decision

A ``battle_id`` is either the trace's ``*_summary.json`` path (as returned by
``battles()``/``run_summary``) or a short ``step_<N>/<Opponent>/<outcome>_<idx>``
id. Model loading uses the same exact→nearest→recent ladder as the TUI, cached.
The matching CLI is ``python -m main.prober.query``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from main.prober.discovery import (
    BattleTrace,
    ModelChoice,
    build_trace_tree,
    list_checkpoints,
    resolve_model_for_step,
)
from main.prober.engine import analyze_invocation, build_board, parse_pct, summary_flags


def _active_str(side) -> str:
    s = f"{side.active_species} {side.active_hp}"
    return f"{s} {side.status}" if side.status else s

_DEFAULT_GAMMA = 0.99


def _choice_dict(c: ModelChoice) -> dict:
    return {"path": c.path, "tier": c.tier, "detail": c.detail, "manifest": c.manifest}


def _short_id(b: BattleTrace) -> str:
    return f"step_{b.step}/{b.opponent}/{b.outcome}_{b.index:03d}"


class ProbeSession:
    def __init__(self, root: str, ckpt_override: "str | None" = None, tier: str = "auto",
                 model_loader=None) -> None:
        self.tree = build_trace_tree(root)
        self.run_dir = self.tree.run_dir
        self._override = ckpt_override
        self._tier = tier
        self._model_loader = model_loader      # (path)->model; default ProbeModel.load (tests inject)
        self._models: dict = {}                 # checkpoint path → ProbeModel
        self._summaries: "dict[str, dict]" = {}
        self._by_path = {b.summary_path: b for b in self.tree.all_battles()}
        self._by_short = {_short_id(b): b for b in self.tree.all_battles()}
        self._gamma = self._read_gamma()

    # -- run orientation -----------------------------------------------------

    def run_summary(self) -> dict:
        """Orient on a run: steps, per-step model identity, opponents with win/loss
        tallies, persisted checkpoints, and γ. The natural first call. Model-free."""
        steps = []
        for sg in self.tree.steps:
            man = self.tree.manifest_for(sg.step)
            opps, wl = [], {"win": 0, "loss": 0}
            for og in sg.opponents:
                w = sum(1 for b in og.battles if b.outcome == "win")
                l = sum(1 for b in og.battles if b.outcome == "loss")
                wl["win"] += w
                wl["loss"] += l
                opps.append({"name": og.name, "win": w, "loss": l, "battles": len(og.battles)})
            steps.append({
                "step": sg.step,
                "identity": None if not man else {
                    "git_hash": man.get("git_hash"),
                    "arch_signature": man.get("arch_signature"),
                    "snapshot_available": self._snapshot_available(sg.step, man),
                },
                "opponents": opps,
                "totals": wl,
            })
        totals = {
            "win": sum(s["totals"]["win"] for s in steps),
            "loss": sum(s["totals"]["loss"] for s in steps),
            "battles": len(self.tree.all_battles()),
        }
        return {
            "run_dir": self.run_dir, "gamma": self._gamma, "n_steps": len(steps),
            "checkpoints": [{"step": s, "path": p} for s, p in list_checkpoints(self.run_dir)],
            "steps": steps, "totals": totals,
        }

    # -- discovery -----------------------------------------------------------

    def battles(self, *, outcome: "str | None" = None, opponent: "str | None" = None,
                step: "int | None" = None) -> "list[dict]":
        """List battles, optionally filtered by outcome / opponent / step."""
        out = []
        for b in self.tree.all_battles():
            if outcome and b.outcome != outcome:
                continue
            if opponent and b.opponent != opponent:
                continue
            if step is not None and b.step != step:
                continue
            out.append({
                "id": b.summary_path, "short_id": _short_id(b), "step": b.step,
                "opponent": b.opponent, "outcome": b.outcome, "index": b.index,
                "has_npz": b.npz_path is not None,
            })
        return out

    # -- model-free battle digest -------------------------------------------

    def battle_overview(self, battle_id: str) -> dict:
        """A MODEL-FREE per-decision digest: chosen, top prob, recorded V(s), ΔV to
        the next decision, TD residual (critic surprise), per-step reward + events,
        and flags — plus a `notable` summary and how a deep analyze would resolve the
        model. No checkpoint loaded."""
        b = self._battle(battle_id)
        summary = self._summary(b)
        values = self._values(b)
        invs = summary["invocations"]
        rows = []
        for i, inv in enumerate(invs):
            acts = inv.get("actions", {})
            chosen = inv.get("chosen", "")
            reward = (inv.get("outcome") or {}).get("reward")
            rtotal = reward.get("total") if isinstance(reward, dict) else reward
            v = self._v(values, i)
            v_next = self._v(values, i + 1)
            board = build_board(inv)
            rows.append({
                "inv": i, "turn": inv.get("turn"), "phase": inv.get("phase"),
                "chosen": chosen,
                "our_active": _active_str(board.ours), "opp_active": _active_str(board.opp),
                "top_prob": parse_pct(acts[chosen]["prob"]) if chosen in acts else None,
                "value": v,
                "delta_v": (v_next - v) if (v is not None and v_next is not None) else None,
                "td_residual": self._td(rtotal, v, v_next),
                "reward_total": rtotal,
                "events": (inv.get("outcome") or {}).get("events") or [],
                "flags": list(summary_flags(inv)),
            })
        return {
            "id": b.summary_path, "short_id": _short_id(b), "meta": summary.get("meta", {}),
            "gamma": self._gamma,
            "model_resolution": _choice_dict(self._resolve(b)),
            "notable": self._notable(rows),
            "invocations": rows,
        }

    # -- deep analysis (loads the resolved model) ---------------------------

    def analyze(self, battle_id: str, inv_index: int) -> dict:
        """Full forensic analysis of one decision as a JSON-serializable dict
        (faithfulness, matchups, intervention, saliency, value+TD, outcome, model
        disagreement). Loads the exact→nearest→recent model."""
        b = self._battle(battle_id)
        model, choice = self._model_for(b)
        a = analyze_invocation(model, self._summary(b), self._npz(b), inv_index,
                               summary_path=b.summary_path, npz_path=b.npz_path)
        d = asdict(a)
        d["model_resolution"] = _choice_dict(choice)
        if d.get("value"):  # add the TD residual the engine (γ-agnostic) can't
            reward = (d.get("outcome") or {}).get("reward")
            rtotal = reward.get("total") if isinstance(reward, dict) else reward
            d["value"]["td_residual"] = self._td(
                rtotal, d["value"]["recorded"], d["value"]["next_recorded"])
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

    # -- cross-battle turning-point scan (model-free) -----------------------

    def scan(self, *, outcome: "str | None" = None, opponent: "str | None" = None,
             step: "int | None" = None, limit: "int | None" = None,
             metric: str = "value_drop") -> "list[dict]":
        """Cross-battle, MODEL-FREE turning-point scan. For every matching battle,
        find its single worst decision and return them **ranked globally** — the
        one-call version of "list losses → overview each → rank by the biggest
        value drop", which is the recurring first move of any loss investigation.

        ``metric`` ranks by ``value_drop`` (most negative ΔV(s→s'), the default) or
        ``td_residual`` (most negative critic surprise δ = r + γV(s') − V(s)).
        Filters mirror ``battles()``. No checkpoint is loaded, so this is fast even
        across a whole run. Each row is ``{id, short_id, opponent, step, outcome,
        turns, worst:{inv, turn, phase, chosen, our_active, opp_active, delta_v,
        td_residual, reward_total, events, flags}}``.
        """
        if metric not in ("value_drop", "td_residual"):
            raise ValueError(f"metric must be 'value_drop' or 'td_residual', got {metric!r}")
        rows = []
        for b in self.tree.all_battles():
            if outcome and b.outcome != outcome:
                continue
            if opponent and b.opponent != opponent:
                continue
            if step is not None and b.step != step:
                continue
            tp = self._worst_turning_point(b, metric)
            if tp is not None:
                rows.append(tp)
        key = "delta_v" if metric == "value_drop" else "td_residual"
        rows.sort(key=lambda r: (r["worst"][key] if r["worst"][key] is not None else float("inf")))
        return rows[:limit] if limit else rows

    def _worst_turning_point(self, battle: BattleTrace, metric: str) -> "dict | None":
        """The single worst decision in one battle by `metric` (model-free).
        Boards are built only for the chosen invocation, so a scan stays cheap."""
        summary = self._summary(battle)
        invs = summary.get("invocations", [])
        if not invs:
            return None
        values = self._values(battle)
        best_i = best_score = best_dv = best_td = None
        for i, inv in enumerate(invs):
            v, v_next = self._v(values, i), self._v(values, i + 1)
            dv = (v_next - v) if (v is not None and v_next is not None) else None
            reward = (inv.get("outcome") or {}).get("reward")
            rtotal = reward.get("total") if isinstance(reward, dict) else reward
            td = self._td(rtotal, v, v_next)
            score = dv if metric == "value_drop" else td
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_i, best_score, best_dv, best_td = i, score, dv, td
        if best_i is None:
            return None
        inv = invs[best_i]
        board = build_board(inv)
        reward = (inv.get("outcome") or {}).get("reward")
        rtotal = reward.get("total") if isinstance(reward, dict) else reward
        return {
            "id": battle.summary_path, "short_id": _short_id(battle),
            "opponent": battle.opponent, "step": battle.step, "outcome": battle.outcome,
            "turns": (summary.get("meta") or {}).get("turns"),
            "worst": {
                "inv": best_i, "turn": inv.get("turn"), "phase": inv.get("phase"),
                "chosen": inv.get("chosen", ""),
                "our_active": _active_str(board.ours), "opp_active": _active_str(board.opp),
                "delta_v": best_dv, "td_residual": best_td, "reward_total": rtotal,
                "events": (inv.get("outcome") or {}).get("events") or [],
                "flags": list(summary_flags(inv)),
            },
        }

    # -- internals -----------------------------------------------------------

    def _notable(self, rows: "list[dict]") -> dict:
        drops = sorted(((r["delta_v"], r["inv"]) for r in rows if r["delta_v"] is not None))
        return {
            "faints": [r["inv"] for r in rows if "faint" in r["flags"]],
            "switches": [r["inv"] for r in rows if "switch" in r["flags"]],
            "uncertain_count": sum("uncertain" in r["flags"] for r in rows),
            "biggest_value_drops": [{"inv": i, "delta_v": d} for d, i in drops[:3]],
            "disagreements_hint": "call find(battle, 'disagree') to load the model and list them",
        }

    def _td(self, reward_total, v, v_next) -> "float | None":
        """TD residual δ = r + γV(s') − V(s): how surprised the critic was."""
        if reward_total is None or v is None or v_next is None:
            return None
        return float(reward_total) + self._gamma * v_next - v

    def _read_gamma(self) -> float:
        if not self.run_dir:
            return _DEFAULT_GAMMA
        try:
            with open(os.path.join(self.run_dir, "metadata.json")) as f:
                return float(json.load(f).get("gamma", _DEFAULT_GAMMA))
        except (OSError, ValueError, TypeError):
            return _DEFAULT_GAMMA

    def _snapshot_available(self, step: int, manifest: dict) -> bool:
        if not (manifest.get("snapshot") and self.run_dir):
            return False
        return os.path.exists(os.path.join(
            self.run_dir, "eval_traces", f"step_{step}", manifest["snapshot"]))

    def _values(self, battle: BattleTrace):
        return self._npz(battle).get("values")

    @staticmethod
    def _v(values, i: int) -> "float | None":
        return float(values[i]) if values is not None and 0 <= i < len(values) else None

    def _battle(self, battle_id: str) -> BattleTrace:
        b = self._by_path.get(battle_id) or self._by_short.get(battle_id)
        if b is None:  # allow a raw summary path not in the original tree
            extra = build_trace_tree(battle_id).all_battles()
            if not extra:
                raise FileNotFoundError(f"no trace found for {battle_id!r}")
            b = extra[0]
        return b

    def _summary(self, battle: BattleTrace) -> dict:
        s = self._summaries.get(battle.summary_path)
        if s is None:
            with open(battle.summary_path) as f:
                s = json.load(f)
            self._summaries[battle.summary_path] = s
        return s

    def _npz(self, battle: BattleTrace) -> dict:
        if battle.npz_path is None:
            return {}
        with np.load(battle.npz_path) as z:
            return {k: z[k] for k in z.files}

    def _resolve(self, battle: BattleTrace) -> ModelChoice:
        return resolve_model_for_step(self.tree, battle.step, self._override, self._tier)

    def _model_for(self, battle: BattleTrace):
        choice = self._resolve(battle)
        if choice.path is None:
            raise FileNotFoundError(choice.detail)
        model = self._models.get(choice.path)
        if model is None:
            if self._model_loader is not None:
                model = self._model_loader(choice.path)
            else:
                from main.prober.model import ProbeModel
                model = ProbeModel.load(choice.path)
            self._models[choice.path] = model
        return model, choice
