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
from main.prober.engine import (
    analyze_invocation, attribute_turning_point, build_board, decode_incoming_belief,
    fit_probe, history_slot_saliency, parse_pct, summary_flags, SETUP_MOVES,
)


def _active_str(side) -> str:
    s = f"{side.active_species} {side.active_hp}"
    return f"{s} {side.status}" if side.status else s


def _r(x, n=3):
    """round-or-None — compact numbers in JSON output."""
    return round(float(x), n) if isinstance(x, (int, float)) else None

_DEFAULT_GAMMA = 0.99


def _choice_dict(c: ModelChoice) -> dict:
    return {"path": c.path, "tier": c.tier, "detail": c.detail, "manifest": c.manifest}


def _short_id(b: BattleTrace) -> str:
    return f"step_{b.step}/{b.opponent}/{b.outcome}_{b.index:03d}"


# -- representation-probe targets -------------------------------------------
# Each target maps a decision (its `_probe_ctx`) to a LABEL (the derived quantity
# to recover from the model's activations; None = skip), a GROUP (easy vs
# contested — the real signal is whether the rep knows X on the HARD cases), and
# the PROVIDED obs/belief feature we already hand the model (the baseline the
# representation probe is compared against — "is the rep more than the feature?").

def _base_spe(species_id) -> "int | None":
    """Gen3 BASE speed for a species id (the realized-truth proxy for who's faster —
    EVs/nature/paralysis are the residual the 'contested' band isolates)."""
    if not isinstance(species_id, str) or not species_id or species_id == "NONE":
        return None
    from agents.gen3_data.species import get as _get
    sd = _get(species_id.lower())
    return sd.base_stats.get("spe") if sd else None


def _faster_label(ctx):
    a, b = _base_spe(ctx["our_species"]), _base_spe(ctx["opp_species"])
    if a is None or b is None or a == b:
        return None                                  # missing / a true speed tie — ambiguous
    return 1.0 if a > b else 0.0


def _faster_group(ctx):
    a, b = _base_spe(ctx["our_species"]), _base_spe(ctx["opp_species"])
    return "easy" if (a is not None and b is not None and abs(a - b) > 25) else "contested"


def _dmg_label(ctx):
    d = ctx["our_dhp"]
    if d is None or ctx["phase"] == "forced_switch":
        return None                                  # no resolved HP delta / not a combat decision
    return float(max(0.0, -d))                        # fraction of HP our active LOST this turn


def _belief_pko_group(ctx, hi, lo, names):
    bel = ctx["belief"]
    pko = bel.active_pko if bel else None
    if pko is None:
        return "unknown"
    return names[0] if pko < lo else names[1] if pko > hi else names[2]


def _faint_label(ctx):
    h, d = ctx["our_hp"], ctx["our_dhp"]
    if h is None or d is None:
        return None
    return 1.0 if (h + d) <= 0.02 else 0.0


def _faint_healthy_label(ctx):
    """Faint THIS turn, but only for a HEALTHY active (HP>=0.6) — isolates the genuine surprise-OHKO
    from the trivial low-HP→faint the plain faint_soon target conflates."""
    h, d = ctx["our_hp"], ctx["our_dhp"]
    if h is None or d is None or h < 0.6:
        return None
    return 1.0 if (h + d) <= 0.02 else 0.0


def _big_hit_label(ctx):
    """Will our active LOSE >=40% HP this turn — a less-RNG-sensitive damage-anticipation target
    than the exact magnitude (a near-OHKO is a near-OHKO regardless of the roll)."""
    d = ctx["our_dhp"]
    if d is None or ctx["phase"] == "forced_switch":
        return None
    return 1.0 if d <= -0.4 else 0.0


def _opp_switch_label(ctx):
    """Did the opponent VOLUNTARILY switch out this turn? Tests whether the representation does
    implicit opponent modeling (anticipates the opp's play) — the core of the world-model idea.
    None on ambiguous (none/unknown/forced post-faint or move-induced) actions."""
    a = ctx.get("opp_action")
    if not a or a in ("none", "unknown") or "→" in a or "_sent_in" in a:
        return None
    return 1.0 if a.startswith("switched_to") else 0.0


def _prov(ctx, attr):
    bel = ctx["belief"]
    return getattr(bel, attr, None) if bel is not None else None


_PROBE_TARGETS = {
    "is_faster": {
        "task": "classification", "label": _faster_label, "group": _faster_group,
        "provided": lambda c: _prov(c, "active_outspeed"), "provided_name": "active_outspeed",
        "tests": ("does the representation encode the true (base-)speed order? 'contested' = close "
                  "base speeds where EVs/nature decide and Leftovers/Sandstorm-residual timing must "
                  "be inferred across turns."),
        "how_to_read": ("rep accuracy >> the provided active_outspeed baseline (especially on "
                        "'contested') = the model infers speed BEYOND the feature → no new feature "
                        "needed. rep ≈ provided AND both weak on 'contested' = a real speed-inference "
                        "gap → an explicit residual-timing speed feature is a lever."),
        "caveat": ("label is BASE-speed order; the obs carries species base stats, so some recovery "
                   "is expected — the 'contested' (close base speeds) split is the informative one. "
                   "Does NOT directly test inferring speed from Leftovers/Sandstorm residual timing "
                   "(the base-speed label treats base speed as truth, so it can't isolate EV/tie cases)."),
    },
    "damage_taken": {
        "task": "regression", "label": _dmg_label,
        "group": lambda c: _belief_pko_group(c, 0.9, 0.1, ("low", "high", "contested")),
        "provided": lambda c: _prov(c, "active_exp"), "provided_name": "incoming active_exp",
        "tests": ("does the representation predict the HP fraction our active LOSES this turn? "
                  "'contested' = belief active_pko in (0.1,0.9), the high-variance / coinflip band."),
        "how_to_read": ("high r2 overall but POOR on 'contested' = the rep has the MEAN but not the "
                        "SPREAD → a p50/p90 damage feature targets exactly that band. rep r2 ≈ the "
                        "provided active_exp baseline = the scalar belief is all the rep has."),
        "caveat": ("realized HP loss has IRREDUCIBLE roll/crit variance, so even a perfect model "
                   "caps below r2=1 — a low r2 partly reflects RNG, not only a representation gap. "
                   "The rep-vs-provided DELTA (not the absolute r2) is the signal; switches/no-hit "
                   "decisions contribute 0 and make the target zero-inflated."),
    },
    "faint_soon": {
        "task": "classification", "label": _faint_label,
        "group": lambda c: _belief_pko_group(c, 0.5, 0.5, ("belief_quiet", "belief_flagged", "belief_flagged")),
        "provided": lambda c: _prov(c, "active_pko"), "provided_name": "active_pko",
        "tests": ("does the representation anticipate our active FAINTING this turn? grouped by "
                  "whether the belief flagged it (active_pko>=0.5)."),
        "how_to_read": ("high accuracy in 'belief_quiet' (the belief did NOT warn, yet the rep "
                        "predicts the faint) = the model knows more than the P(KO) feature → enrich "
                        "the belief. LOW in 'belief_quiet' = genuine surprise (unrevealed-attacker "
                        "coverage gap or irreducible RNG)."),
        "caveat": ("faint-this-turn correlates with CURRENT HP (which is in the obs), so high "
                   "accuracy partly reflects a trivial low-HP→faint read, not anticipation. To "
                   "isolate true surprise-OHKOs, re-run conditioned on healthy HP (a future "
                   "group); the 'belief_quiet' AUC is the most informative cell here."),
    },
    "faint_healthy": {
        "task": "classification", "label": _faint_healthy_label,
        "group": lambda c: _belief_pko_group(c, 0.5, 0.5, ("belief_quiet", "belief_flagged", "belief_flagged")),
        "provided": lambda c: _prov(c, "active_pko"), "provided_name": "active_pko",
        "tests": ("does the representation anticipate a HEALTHY (HP>=60%) active being OHKO'd this "
                  "turn — the genuine SURPRISE-OHKO, with the trivial low-HP→faint cases removed."),
        "how_to_read": ("rep AUC >> the active_pko baseline in 'belief_quiet' = the rep sees the "
                        "surprise OHKO the belief misses → enrich the incoming belief (unrevealed/just-"
                        "switched coverage). BOTH near chance in 'belief_quiet' = the OHKO is genuinely "
                        "not inferable from the obs → an obs-COVERAGE gap (the surprise_ohko plateau lever)."),
        "caveat": ("healthy-only positive rate is low (most healthy mons survive), so read AUC/lift, "
                   "not raw accuracy. This is the clean version of faint_soon for the surprise-OHKO question."),
    },
    "big_hit_incoming": {
        "task": "classification", "label": _big_hit_label,
        "group": lambda c: _belief_pko_group(c, 0.9, 0.1, ("low", "high", "contested")),
        "provided": lambda c: _prov(c, "active_exp"), "provided_name": "incoming active_exp",
        "tests": ("does the representation anticipate LOSING >=40% HP this turn (a big hit / near-OHKO), "
                  "a less-RNG-sensitive damage signal than the exact magnitude."),
        "how_to_read": ("rep AUC >> the active_exp baseline = the rep anticipates big hits beyond the "
                        "scalar belief. rep ≈ provided AND weak on 'contested' = the magnitude signal is "
                        "missing → a richer (p50/p90 or crit-split) damage feature is a lever."),
        "caveat": ("realized — a hit that 'should' be big can roll low (and vice-versa), so a perfect "
                   "model can't reach AUC 1; the rep-vs-provided delta is the signal."),
    },
    "opp_switches": {
        "task": "classification", "label": _opp_switch_label, "group": lambda c: "all",
        "provided": lambda c: None, "provided_name": None,
        "tests": ("does the representation anticipate the OPPONENT voluntarily switching out this turn "
                  "— a direct test of implicit opponent modeling (the world-model / lookahead idea)."),
        "how_to_read": ("rep AUC well above 0.5 = the model already does implicit opponent modeling "
                        "(a switch-prediction head would be redundant). rep AUC ≈ 0.5 = the rep does NOT "
                        "anticipate the opponent → an opponent-action prediction head (auxiliary world-model) "
                        "is a real, untapped lever — the strongest 'make the model sharper' candidate."),
        "caveat": ("no provided baseline (we give the model no opp-switch feature); voluntary switches "
                   "are ~10% of decisions, so read AUC, not accuracy. Pokémon is simultaneous-move, so "
                   "perfect prediction is impossible — but well-above-chance is the bar for 'it models the opp'."),
    },
}


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

    def falsify(self, battle_id: str, *, invs=None, worst: int = 3,
                n_seeds: int = 40, n_alts: int = 3, followup: str = "random") -> dict:
        """Dice attribution (luck vs reducible mistake) for a battle's worst — or
        explicitly chosen — decisions, by RE-ROLLING the real turns through the
        reconstruction layer (fix-both luck percentile + paired alternative-action
        sweep on a material margin). Model-free (no checkpoint); requires the
        trace's ``*_reconstruction.json`` sibling, which only bridge-eval traces
        written by the reconstruction layer carry."""
        from main.prober.falsifier import falsify_battle
        from utils.bridge.reconstruction import ReconstructionRecord

        b = self._battle(battle_id)
        recon_path = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon_path):
            raise FileNotFoundError(
                f"no reconstruction record next to this trace ({recon_path}) — "
                "the battle predates the reconstruction layer or ran websocket "
                "eval; only bridge-eval traces carry the falsifier's replay data")
        record = ReconstructionRecord.load(recon_path)
        return falsify_battle(record, self._summary(b), self._npz(b),
                              invs=invs, worst=worst, gamma=self._gamma,
                              n_seeds=n_seeds, n_alts=n_alts, followup=followup)

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
        npz = self._npz(battle)
        values = npz.get("values")
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
        worst = {
            "inv": best_i, "turn": inv.get("turn"), "phase": inv.get("phase"),
            "chosen": inv.get("chosen", ""),
            "our_active": _active_str(board.ours), "opp_active": _active_str(board.opp),
            "delta_v": best_dv, "td_residual": best_td, "reward_total": rtotal,
            "events": (inv.get("outcome") or {}).get("events") or [],
            "flags": list(summary_flags(inv)),
        }
        # Decode the incoming-damage / OHKO belief the obs HELD at this cliff (model-free, from the
        # saved obs). The decisive A/B for "did the feature fill the obs gap": a high active_pko at
        # a value cliff means the OHKO WAS in the obs (remaining error is downstream usage); a low
        # one where our active then faints means the belief is mis-calibrated.
        belief = self._belief_at(npz, best_i)
        if belief is not None:
            worst["incoming_active_pko"] = belief.active_pko
            worst["incoming_max_pko"] = belief.max_pko
            worst["incoming_active_outspeed"] = belief.active_outspeed
        return {
            "id": battle.summary_path, "short_id": _short_id(battle),
            "opponent": battle.opponent, "step": battle.step, "outcome": battle.outcome,
            "turns": (summary.get("meta") or {}).get("turns"),
            "worst": worst,
        }

    def _obs_offsets(self):
        """Lazily resolve the obs-block offsets once (builds the encoder; model-free). Cached on the
        session. None if resolution fails, so the belief decode degrades gracefully."""
        off = getattr(self, "_offsets_cache", "unset")
        if off == "unset":
            try:
                from main.prober.model import ObsOffsets
                off = ObsOffsets.resolve()
            except Exception:  # noqa: BLE001 — belief decode is best-effort
                off = None
            self._offsets_cache = off
        return off

    def _belief_at(self, npz: dict, i: int):
        """The decoded incoming-damage belief at decision ``i`` (or None — no obs / no captured
        state for that decision / offsets unresolvable)."""
        obs_arr = npz.get("obs")
        if obs_arr is None or i >= len(obs_arr):
            return None
        hs = npz.get("has_state")
        if hs is not None and i < len(hs) and not bool(hs[i]):
            return None
        off = self._obs_offsets()
        if off is None:
            return None
        return decode_incoming_belief(obs_arr[i].astype(np.float32), off)

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

    # -- loss attribution: rank failure categories by recoverable rating -----

    def _turning_point_features(self, battle: BattleTrace) -> "dict | None":
        """The DECISIVE turning point of one loss (worst ΔV) as a feature dict the engine taxonomy
        categorizes. Model-free (summary + npz). None on an empty trace."""
        summary = self._summary(battle)
        invs = summary.get("invocations", [])
        if not invs:
            return None
        values = self._npz(battle).get("values")
        best_i = best_dv = None
        for i in range(len(invs)):
            v, vn = self._v(values, i), self._v(values, i + 1)
            dv = (vn - v) if (v is not None and vn is not None) else None
            if dv is None:
                continue
            if best_dv is None or dv < best_dv:
                best_i, best_dv = i, dv
        if best_i is None:                      # no value series — fall back to the last decision
            best_i = len(invs) - 1
        inv = invs[best_i]
        # Critic surprise at the cliff: δ = r + γV(s') − V(s). A large |δ| means the value crater
        # CAUGHT THE CRITIC OFF GUARD (capacity / missing-obs lever); a small δ means it declined as
        # the critic already expected (a genuinely-lost position, not a critic fix). Splits value_cliff.
        _reward = (inv.get("outcome") or {}).get("reward")
        _rtotal = _reward.get("total") if isinstance(_reward, dict) else _reward
        v_at = self._v(values, best_i)
        td = self._td(_rtotal, v_at, self._v(values, best_i + 1))
        board = build_board(inv)
        our = inv.get("our") or {}
        ocour = (inv.get("outcome") or {}).get("our") or {}

        def pp(s):
            try:
                return parse_pct(s)
            except Exception:  # noqa: BLE001
                return None

        our_hp, our_dhp = pp(our.get("hp")), pp(ocour.get("hp_delta"))
        chosen = str(inv.get("chosen", ""))
        is_switch = chosen.startswith("switch")
        move_id = "" if is_switch else chosen.lower().replace(" ", "")
        belief = self._belief_at(self._npz(battle), best_i)
        psp = list(belief.per_slot_pko) if (belief is not None and belief.per_slot_pko) else None
        return {
            "short_id": _short_id(battle), "opponent": battle.opponent, "step": battle.step,
            "turns": (summary.get("meta") or {}).get("turns"),
            "inv": best_i, "turn": inv.get("turn"), "phase": inv.get("phase"), "chosen": chosen,
            "is_switch": is_switch, "is_setup": move_id in SETUP_MOVES,
            "our_species": our.get("species"), "our_hp": our_hp, "our_hp_delta": our_dhp,
            "faint": bool(our_hp is not None and our_dhp is not None and our_hp + our_dhp <= 0.02),
            "active_pko": (belief.active_pko if belief else None),
            "active_outspeed": (belief.active_outspeed if belief else None),
            "max_pko": (belief.max_pko if belief else None),
            "n_healthy_bench": sum(1 for m in board.ours.bench if not m.fainted),
            "min_other_pko": (min(psp) if psp else None),
            "delta_v": best_dv, "td": td, "v_at": v_at,
        }

    def _win_rates(self, step: "int | None") -> dict:
        """Per-opponent TRUE win-rate from eval_results.jsonl (row nearest `step`, else latest) — the
        loss-rate weight for the attribution. {} if absent. (NOT the trace win/loss tally, which is
        loss-WEIGHTED and would over-state the loss volume.)"""
        try:
            with open(os.path.join(self.run_dir or "", "eval_results.jsonl")) as f:
                rows = [json.loads(ln) for ln in f if ln.strip()]
        except (OSError, ValueError, TypeError):
            return {}
        if not rows:
            return {}
        row = (min(rows, key=lambda r: abs(r.get("step", 0) - step)) if step is not None else rows[-1])
        return dict(row.get("bots") or {})

    def triage(self, *, step: "int | None" = None, opponent: "str | None" = None) -> dict:
        """Loss attribution: categorize every loss's decisive turning point into the engine taxonomy,
        aggregate, and RANK the failure categories by estimated recoverable win-rate (the lever
        prioritization). Model-free. Defaults to the latest step that has loss traces."""
        from collections import defaultdict
        losses = [b for b in self.tree.all_battles() if b.outcome == "loss"]
        if step is None:
            steps = sorted({b.step for b in losses})
            step = steps[-1] if steps else None
        losses = [b for b in losses
                  if (step is None or b.step == step) and (not opponent or b.opponent == opponent)]
        wr = self._win_rates(step)

        per_opp = defaultdict(lambda: {"n": 0, "cats": defaultdict(list)})
        cat_meta: dict = {}
        for b in losses:
            feat = self._turning_point_features(b)
            if feat is None:
                continue
            a = attribute_turning_point(feat)
            cat_meta[a["category"]] = a
            po = per_opp[b.opponent]
            po["n"] += 1
            po["cats"][a["category"]].append(feat)

        bot_opps = [o for o in per_opp if o in wr]                    # only the fixed-bot anchor
        cat_total: dict = defaultdict(int)
        cat_recover: dict = defaultdict(float)
        cat_by_opp: dict = defaultdict(lambda: defaultdict(int))
        cat_examples: dict = defaultdict(list)
        for opp, po in per_opp.items():
            for cat, feats in po["cats"].items():
                cat_total[cat] += len(feats)
                cat_by_opp[cat][opp] = len(feats)
                for f in feats[:3]:
                    cat_examples[cat].append(
                        f"{f['short_id']} t{f.get('turn')}: {f.get('our_species')} chose "
                        f"{f.get('chosen')} (pko={_r(f.get('active_pko'),2)}, "
                        f"outspeed={_r(f.get('active_outspeed'),2)}, dHP={_r(f.get('our_hp_delta'),2)}, "
                        f"healthy_bench={f.get('n_healthy_bench')})")
        for opp in bot_opps:
            n = per_opp[opp]["n"] or 1
            loss_rate = max(0.0, 1.0 - float(wr.get(opp, 0.0)))
            for cat, feats in per_opp[opp]["cats"].items():
                cat_recover[cat] += loss_rate * (len(feats) / n) / max(1, len(bot_opps))

        total = sum(cat_total.values())
        # Rank by recoverable win-rate; break ties by raw volume so the order is stable AND meaningful
        # even when no bot win-rates exist (recover all 0 → fall back to "most losses first").
        cats = [{
            "category": c, "lever": cat_meta[c]["lever"], "blurb": cat_meta[c]["blurb"],
            "n": cat_total[c], "pct_of_sampled_losses": round(100 * cat_total[c] / max(1, total), 1),
            "est_recoverable_winrate_pct": round(100 * cat_recover.get(c, 0.0), 2),
            "by_opponent": dict(cat_by_opp[c]),
            "examples": cat_examples[c][:4],
        } for c in sorted(cat_total, key=lambda c: (-cat_recover.get(c, 0.0), -cat_total[c]))]

        ranking_metric = ("est_recoverable_winrate_pct = mean over BOT opponents of "
                          "loss_rate(opp) × category_share(opp); an UPPER BOUND (assumes fixing "
                          "the lever flips that loss). Ranked descending — this is the lever order.")
        caveats = [
            "Eval traces are LOSS-WEIGHTED (~10 loss / 5 win per opponent), so category SHARES are "
            "per-opponent representative but raw counts are NOT the true loss volume — the recoverable "
            "estimate corrects for that via the true per-opponent win-rate.",
            "Each loss is attributed to its single worst-ΔV turning point; a loss can have several causes.",
            "Recoverable-winrate is over BOT opponents only (the ELO anchor); sentinel/ext losses are "
            "counted (pct_of_sampled_losses) but not rating-weighted (their win-rate is gate-pinned).",
        ]
        if not bot_opps:
            ranking_metric = ("no eval_results.jsonl bot win-rates found — est_recoverable_winrate_pct "
                              "is 0 for every category; ranked by RAW loss volume instead.")
            caveats.append("No bot win-rates available: ranking fell back to loss count, NOT "
                           "rating-weighted recoverability. Run an eval cycle (writes eval_results.jsonl) "
                           "for the lever-prioritized order.")

        return {
            "run_dir": self.run_dir, "step": step,
            "n_losses_analyzed": total, "n_bot_opponents": len(bot_opps),
            "bot_win_rates": {o: wr[o] for o in bot_opps},
            "ranking_metric": ranking_metric,
            "caveats": caveats,
            "categories": cats,
        }

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
