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
    analyze_invocation, attribute_turning_point, build_board, build_our_hp_types,
    decode_incoming_belief, fit_probe, history_slot_saliency, parse_pct, parse_protocol_log,
    protocol_for_turn, summary_flags, SETUP_MOVES,
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


# Fixed-damage moves carry basePower 0 in the data but DO deal damage — they are attacks,
# not status moves (Seismic Toss / Night Shade / etc.). Exclude them from the status proxy.
_FIXED_DAMAGE_MOVES = frozenset({
    "seismictoss", "nightshade", "psywave", "sonicboom", "dragonrage",
    "superfang", "counter", "mirrorcoat", "endeavor", "bide",
})


def _opp_move_id(ctx):
    """The opponent's MOVE id this turn (normalized), or None if the action was a switch /
    forced replacement / none / unknown. Strips the '→ …' resolution suffix
    (e.g. 'seismictoss → phazed')."""
    a = ctx.get("opp_action")
    if not a or a in ("none", "unknown") or a.startswith("switched_to") or "_sent_in" in a:
        return None
    mid = a.split("→")[0].strip()
    return mid or None


def _opp_status_move_label(ctx):
    """Among turns the opponent used a MOVE: was it a STATUS (non-damaging) move vs an attack?
    A finer opponent-anticipation target than 'will they switch' — tests whether the rep
    predicts the opp's intent (set-up/utility vs damage). basePower-0 proxy, minus the
    fixed-damage attacks. None on switch/forced/no-move turns or unknown move ids."""
    from agents import gen3_data
    mid = _opp_move_id(ctx)
    if mid is None or mid in _FIXED_DAMAGE_MOVES:
        return None if mid is None else 0.0
    rec = gen3_data.moves.raw().get(mid)
    if rec is None:
        return None
    return 1.0 if int(rec.get("basePower", 0)) == 0 else 0.0


def _prov(ctx, attr):
    bel = ctx["belief"]
    return getattr(bel, attr, None) if bel is not None else None


# --- critic-calibration pure helpers (model-free; the calibration probe) --------

def _discounted_returns(rewards: "list", gamma: float) -> "list[float]":
    """Realized discounted return from each decision to terminal:
    ``G_i = Σ_{j≥i} γ^{j−i} r_j`` (a None reward counts as 0). This is the
    Monte-Carlo value TARGET the critic is trained to predict, so V(s_i) vs G_i is
    a calibration residual."""
    n = len(rewards)
    out = [0.0] * n
    acc = 0.0
    for i in range(n - 1, -1, -1):
        acc = (rewards[i] if rewards[i] is not None else 0.0) + gamma * acc
        out[i] = acc
    return out


def _reliability_curve(values, returns, n_bins: int) -> "list[dict]":
    """Equal-count reliability bins over (V, G) pairs, ascending by V. Per bin:
    ``v_lo/v_hi/v_mean/g_mean/n`` and ``gap = v_mean − g_mean`` (the critic's
    SYSTEMATIC over-valuation at that value level; >0 ⇒ V runs above realized G).
    Binning by V (not by outcome) is what makes it selection-free — a loss-only
    V−G is biased positive by construction (losses are the below-V tail)."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(returns, dtype=float)
    if v.size == 0:
        return []
    order = np.argsort(v, kind="stable")
    v, g = v[order], g[order]
    bins = []
    for idx in np.array_split(np.arange(v.size), min(max(1, n_bins), v.size)):
        if idx.size == 0:
            continue
        vb, gb = v[idx], g[idx]
        bins.append({
            "v_lo": float(vb.min()), "v_hi": float(vb.max()),
            "v_mean": float(vb.mean()), "g_mean": float(gb.mean()),
            "n": int(idx.size), "gap": float(vb.mean() - gb.mean()),
        })
    return bins


def _reliability_gap_at(bins: "list[dict]", v: float) -> "float | None":
    """The critic's systematic over-valuation (bin ``gap``) at value level ``v``:
    among the bins whose [v_lo, v_hi] contain ``v`` (equal-count bins can OVERLAP
    when V has ties at a boundary) the one whose center ``v_mean`` is nearest;
    if none contains ``v``, the nearest bin overall (clamps to the ends). None
    when there are no bins."""
    if not bins:
        return None
    containing = [b for b in bins if b["v_lo"] <= v <= b["v_hi"]]
    return min(containing or bins, key=lambda b: abs(b["v_mean"] - v))["gap"]


def _calibration_stats(values, returns) -> dict:
    """Aggregate critic calibration of recorded V against realized G:
    ``bias`` = E[V−G] (systematic over/under-valuation; ~0 if unbiased over BOTH
    outcomes), ``mae`` = E[|V−G|], ``ev`` = 1−Var(G−V)/Var(G) (how much of the
    return variance V explains), ``slope`` = OLS G~V (1 = calibrated spread, <1 =
    over-confident)."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(returns, dtype=float)
    if v.size == 0:
        return {"n": 0, "bias": None, "mae": None, "ev": None, "slope": None,
                "v_mean": None, "g_mean": None}
    resid = v - g
    var_g = float(np.var(g))
    var_v = float(np.var(v))
    return {
        "n": int(v.size),
        "bias": float(resid.mean()),
        "mae": float(np.abs(resid).mean()),
        "ev": (1.0 - float(np.var(resid)) / var_g) if var_g > 1e-9 else None,
        "slope": float(np.cov(v, g, bias=True)[0, 1] / var_v) if var_v > 1e-9 else None,
        "v_mean": float(v.mean()), "g_mean": float(g.mean()),
    }


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
    "opp_status_move": {
        "task": "classification", "label": _opp_status_move_label, "group": lambda c: "all",
        "provided": lambda c: None, "provided_name": None,
        "tests": ("among turns the opponent uses a MOVE, does the representation anticipate a STATUS "
                  "(set-up / utility) move vs an attack — a finer opponent-intent prediction than "
                  "will-they-switch (the 'what will they do if they attack' dimension)."),
        "how_to_read": ("rep AUC well above 0.5 = the model already anticipates the opp's move INTENT "
                        "(status-vs-attack) → that dimension of an opponent-action head is redundant. "
                        "rep AUC ≈ 0.5 = the rep does NOT anticipate move intent → an un-falsified "
                        "'what will they do' sub-lever worth a targeted feature/head."),
        "caveat": ("conditioned on move turns only (switches/forced excluded); status moves are a "
                   "minority, so read AUC not accuracy. Simultaneous-move ⇒ perfect prediction "
                   "impossible. ⚠️ LEAK: the obs already encodes each REVEALED opp move's category "
                   "flag (moves.py status/phys/spec, in the opp-team block that flows into the trunk), "
                   "so when a mon's only revealed moves are status this decodes a feature we ALREADY "
                   "hand the model — the AUC measures input re-presentation, NOT anticipation. Use as "
                   "a diagnostic, NOT as a falsifier (unlike opp_switches, which has no provided feature "
                   "and predicts a genuinely-hidden simultaneous choice)."),
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
        self._play_models: dict = {}            # checkpoint path → MaskablePPO (counterfactual replay players)
        self._cf_mappings = None                # lazily-loaded encoder mappings for the replay players
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

    # -- consolidated forensic exports (forensics.py) -----------------------

    def decision_table(self, *, steps=None, opponents=None, outcomes=None,
                       categories=None, max_battles=None) -> "list[dict]":
        """MODEL-FREE per-decision forensic table over the matching battles (one row each:
        move-category, our/opp species+HP, policy ``conf``, ``reward``, critic ``dV``,
        incoming-KO ``pko`` belief, faint flags, outcome). The single source for the
        softmax/dV/belief-decode plumbing every behavioural-hypothesis check reuses."""
        from main.prober import forensics
        return forensics.build_decision_table(
            self, steps=steps, opponents=opponents, outcomes=outcomes,
            categories=categories, max_battles=max_battles)

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

    def _protocol_for(self, b, turn) -> list:
        """Raw Showdown protocol lines for a decision's turn, from the trace's `*_replay.html` sibling
        (the browser-watchable log) — so the JSON `analyze` carries the exact events the summary
        collapses. Empty when the replay file is absent/unreadable."""
        replay = b.summary_path[: -len("_summary.json")] + "_replay.html"
        if not os.path.exists(replay):
            return []
        try:
            with open(replay, encoding="utf-8") as f:
                return list(protocol_for_turn(parse_protocol_log(f.read()), int(turn or 0)))
        except Exception:  # noqa: BLE001 — best-effort
            return []

    def _our_hp_types(self, b) -> "dict | None":
        """OUR team's typed Hidden Power per species from the trace's `reconstruction.json` sibling
        (`{norm_species: 'hiddenpower(bug)'}`), so a bare own HP types in the board — None for
        websocket/older traces. Best-effort; mirrors the TUI's `_load_our_hp_types`."""
        recon = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon):
            return None
        try:
            from utils.bridge.reconstruction import ReconstructionRecord
            rec = ReconstructionRecord.load(recon)
            side = rec.side_of(rec.trainee_username) if rec.trainee_username else None
            return build_our_hp_types(rec.team_details(side)) if side else None
        except Exception:  # noqa: BLE001 — privileged team is best-effort; degrade to bare HP
            return None

    def _opp_team_details(self, b) -> "tuple | None":
        """The opponent's PRIVILEGED `team_details()` (species + evs/ivs/nature/…) from the trace's
        `reconstruction.json` sibling — feeds both the slot-matched belief truth (species) and the
        spread-belief truth (derived stats). Returns `(species_tuple, details_list)` or `None` for
        websocket/older traces. Mirrors the TUI's `_load_opp_team` / `_load_opp_team_details`."""
        recon = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon):
            return None
        try:
            from utils.bridge.reconstruction import ReconstructionRecord
            rec = ReconstructionRecord.load(recon)
            side = rec.side_of(rec.trainee_username) if rec.trainee_username else None
            if not side:
                return None
            opp = "p2" if side == "p1" else "p1"
            details = rec.team_details(opp)
            return tuple(m["species"] for m in details), details
        except Exception:  # noqa: BLE001 — privileged truth is best-effort; degrade to no truth
            return None

    def _our_team_details(self, b) -> "list | None":
        """OUR (trainee's) PRIVILEGED `team_details()` from the trace's `reconstruction.json` sibling
        — feeds the forced-switch switch-in-outgoing panel (true spreads + movesets). `None` for
        websocket/older traces. Mirrors `_opp_team_details` on our own side."""
        recon = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon):
            return None
        try:
            from utils.bridge.reconstruction import ReconstructionRecord
            rec = ReconstructionRecord.load(recon)
            side = rec.side_of(rec.trainee_username) if rec.trainee_username else None
            return rec.team_details(side) if side else None
        except Exception:  # noqa: BLE001 — privileged team is best-effort
            return None

    def analyze(self, battle_id: str, inv_index: int) -> dict:
        """Full forensic analysis of one decision as a JSON-serializable dict
        (faithfulness, matchups, intervention, saliency, value+TD, outcome, model
        disagreement, belief/spread truth). Loads the exact→nearest→recent model."""
        b = self._battle(battle_id)
        model, choice = self._model_for(b)
        opp = self._opp_team_details(b)
        a = analyze_invocation(model, self._summary(b), self._npz(b), inv_index,
                               summary_path=b.summary_path, npz_path=b.npz_path,
                               our_hp_types=self._our_hp_types(b),
                               opp_team=(opp[0] if opp else None),
                               opp_team_details=(opp[1] if opp else None),
                               our_team_details=self._our_team_details(b))
        d = asdict(a)
        d["model_resolution"] = _choice_dict(choice)
        d["protocol"] = self._protocol_for(b, d.get("turn", 0))   # raw Showdown log for this turn
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

    def lookahead(self, battle_id: str, *, invs=None, inv: "int | None" = None, worst: int = 3,
                  n_seeds: int = 0, followup: str = "random") -> dict:
        """One-ply LOOKAHEAD (Feature 1): for an anchored ``move_selection`` decision, RE-ROLL the turn
        under each legal action (the opponent plays its RECORDED move), materialize the resulting
        one-sided successor state through the real encoder, and read the loaded model's **V(s′)** — so
        the result is per-action ΔV, "what would the critic have valued each alternative at" (the value
        readout the model-free :meth:`falsify` deliberately defers; + the distributional / win-prob heads
        when the run trained them). Loads the exact→nearest→recent model. **Requires the trace's
        ``*_reconstruction.json`` sibling** (bridge-eval traces only). ``inv`` looks ahead from ONE
        decision (returns the single-decision dict); otherwise ``invs`` (or the ``worst`` δ-craters)
        returns a battle dict. ``n_seeds`` > 0 dice-averages V(s′) ± std alongside the CRN headline.
        The CHOSEN action's CRN successor reproduces the real next state, so its ``value_crn`` ≈ the
        trace's ``recorded_next_value`` — a built-in consistency anchor."""
        from main.prober.lookahead import lookahead_battle, lookahead_decision
        from utils.bridge.reconstruction import ReconstructionRecord

        b = self._battle(battle_id)
        recon_path = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon_path):
            raise FileNotFoundError(
                f"no reconstruction record next to this trace ({recon_path}) — lookahead needs the "
                "re-roll layer's replay data, which only bridge-eval traces carry")
        record = ReconstructionRecord.load(recon_path)
        model, _ = self._model_for(b)
        summary, npz = self._summary(b), self._npz(b)
        if inv is not None:
            return lookahead_decision(model, record, summary, npz, int(inv),
                                      n_seeds=n_seeds, followup=followup)
        return lookahead_battle(model, record, summary, npz, invs=invs, worst=worst,
                                gamma=self._gamma, n_seeds=n_seeds, followup=followup)

    def replay_counterfactual(self, battle_id: str, inv: int, action: int, *, n_rollouts: int = 1,
                              opponent_ckpt: "str | None" = None, opponent_source: str = "auto",
                              narrate: bool = False) -> dict:
        """COUNTERFACTUAL replay-to-end (Feature 2) — "could the model have won if it hadn't choked this
        turn?". Pick up the recorded battle at ``inv``'s turn, substitute ``action`` (a legal action
        index) for OUR side, then play the rest LIVE — the trainee's GREEDY policy vs the RELOADED real
        opponent — to a win / loss. ``n_rollouts`` > 1 resamples the post-divergence dice (Monte-Carlo)
        for a win-PROBABILITY ± Wilson CI; ``n_rollouts`` == 1 is the single realized-dice line. The
        opponent is RELOADED: a reproducible bot is rebuilt exactly, ``opponent_ckpt`` loads any
        checkpoint (e.g. a self-play sentinel) as the opponent, else the trainee's own model stands in
        (a flagged self-play approximation). **Requires the trace's ``*_reconstruction.json`` sibling.**"""
        from sb3_contrib import MaskablePPO
        from poke_env.ps_client import LocalhostServerConfiguration
        from agents.observation.state_encoder import load_mappings
        from main.prober.replay import replay_counterfactual_battle
        from utils.bridge.reconstruction import ReconstructionRecord

        b = self._battle(battle_id)
        recon_path = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon_path):
            raise FileNotFoundError(
                f"no reconstruction record next to this trace ({recon_path}) — counterfactual replay "
                "needs the re-roll layer's replay data, which only bridge-eval traces carry")
        record = ReconstructionRecord.load(recon_path)

        choice = self._resolve(b)
        if choice.path is None:
            raise FileNotFoundError(
                f"no checkpoint resolved for the counterfactual trainee: {choice.detail}")
        if self._cf_mappings is None:
            self._cf_mappings = load_mappings()

        def _load(path):
            m = self._play_models.get(path)
            if m is None:
                m = MaskablePPO.load(path, env=None, device="cpu")
                m.policy.set_training_mode(False)
                # A `--log-level periodic` checkpoint carries an ObservationDebugger that print()s a
                # "DEEP TRACE" banner on every forward — it would corrupt the CLI's JSON stdout and the
                # TUI screen. Silence it on the replay players, exactly as ProbeModel.load does.
                for mod in m.policy.modules():
                    if hasattr(mod, "_debugger"):
                        mod._debugger = None
                self._play_models[path] = m
            return m

        play_model = _load(choice.path)
        opp_model = _load(opponent_ckpt) if opponent_ckpt else None

        return replay_counterfactual_battle(
            record, self._summary(b), self._npz(b), int(inv), int(action),
            play_model=play_model, opp_name=b.opponent, mappings=self._cf_mappings,
            server_config=LocalhostServerConfiguration, opponent_ckpt=opponent_ckpt,
            opp_model=opp_model, opponent_source=opponent_source, n_rollouts=n_rollouts, narrate=narrate)

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
                    n_alts=n_alts, followup=followup)
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
