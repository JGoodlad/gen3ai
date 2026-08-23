"""The representation-probe TARGET table.

Each target maps a decision (its `_probe_ctx`) to a LABEL (the derived quantity to recover from
the model's activations; None = skip), a GROUP (easy vs contested — the real signal is whether the
rep knows X on the HARD cases), and the PROVIDED obs/belief feature we already hand the model (the
baseline the representation probe is compared against — "is the rep more than the feature?").
"""

from __future__ import annotations


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
