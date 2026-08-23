"""The engine's DATA MODEL — every frozen dataclass one analysis returns.

`analyze_invocation` builds a tree of these and nothing else, which is what makes the whole
engine renderable by three surfaces (the web app, the JSON CLI, `probe_replay`) without any of
them reaching into the analysis. They are pure declarations: no numpy, no torch, no IO.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TraceMeta:
    step: int
    battle_id: str
    result: str          # "WIN" | "LOSS" | "TIE"
    turns: int
    n_invocations: int
    summary_path: str
    npz_path: "str | None"


@dataclass(frozen=True)
class ActionRow:
    label: str           # "switch:Dragonite" / "thunderbolt" / "struggle"
    valid: bool
    recorded: float      # fraction in [0, 1] (parsed from "92.1%")
    rerun: float         # live re-run prob, fraction
    is_chosen: bool


@dataclass(frozen=True)
class MatchupView:
    multipliers: "tuple[float, ...]"   # 4 values, ×4-denormalised (obs[mm:mm+4]*4)
    move_labels: "tuple[str, ...]"     # request-slot order (the 4 move actions)
    # The obs computes a type multiplier for EVERY request slot — including non-damaging moves
    # (Spikes/Toxic/Protect/Recover), where it is meaningless: base_power 0 routes through no
    # damage calc, so a "2.00×" on Spikes is a misleading artefact, not a signal. `applicable[i]`
    # is False for those slots so a UI can render "n/a" instead of the phantom multiplier; an
    # unknown/empty slot label is False too. Applicability is the BROAD "does the type chart matter"
    # sense (`_multiplier_meaningful`): a positive-BP move, a fixed-damage move (Seismic Toss/Night
    # Shade — base_power 0 in the dex but type IMMUNITY still applies), a variable-power move
    # (Return/Frustration), or Hidden Power (poke-env reveals it as the bare id but it's a typed
    # ~70-BP attack) — NOT the bare `gen3_data.moves.is_damaging` (= base_power>0), which would
    # wrongly hide our own Hidden Power and the fixed/variable-power attacks.
    applicable: "tuple[bool, ...]"     # per-slot: does the type multiplier mean anything?


@dataclass(frozen=True)
class InterventionRow:
    multiplier: float                  # 0, 1, 2, 4
    p_chosen: float
    p_switches: float


@dataclass(frozen=True)
class InterventionSweep:
    chosen_label: str
    request_slot: int                  # -1 when the chosen action is not a move
    baseline_p_switches: float
    rows: "tuple[InterventionRow, ...]"  # empty when chosen is not a move

    @property
    def applicable(self) -> bool:
        return self.request_slot >= 0


@dataclass(frozen=True)
class SaliencyBlock:
    name: str
    mean_abs: float      # |grad| / dim
    total_abs: float     # sum |grad|


@dataclass(frozen=True)
class Saliency:
    overall_mean_abs: float
    blocks: "tuple[SaliencyBlock, ...]"


@dataclass(frozen=True)
class ThreatView:
    """Incoming type-effectiveness the OPPONENT threatens, decoded from the obs
    `their_matchups` block (opp-mon × move-slot × our-mon, ×4-denormalised).

    The decisive bit is `present`/`revealed_frac`: the block is **all-zeros for an
    opponent mon whose moves are unrevealed** (`get_sorted_moves` reads only revealed
    `mon.moves`), so a just-switched-in threat contributes *nothing* here — the policy
    must price it from the species embedding alone. A low `revealed_frac` means the
    model is flying on priors, not an explicit incoming-effectiveness signal. Note this
    is *effectiveness*, not *damage* (no base-power / Atk·Def / HP) — an OHKO still has
    to be inferred from this × the per-move power × per-mon stats elsewhere in the obs."""
    present: bool                          # any nonzero incoming-effectiveness cell
    revealed_frac: float                   # fraction of cells > 0 (how much opp coverage is revealed)
    max_incoming: float                    # worst incoming effectiveness anywhere on the board (×4)
    per_our_slot_max: "tuple[float, ...]"  # worst incoming effectiveness vs each of our 6 team slots (×4)


@dataclass(frozen=True)
class IncomingBeliefView:
    """The opponent active's incoming-damage / OHKO **belief** vs our team, decoded from the
    `incoming_damage` obs block (incoming_damage_v1).

    Where ThreatView is raw type-*effectiveness* (no power/Atk/Def/HP), this is the calibrated
    P(KO) / expected-chip belief the feature was added to provide — it already prices base-power ×
    Atk·Def × HP × the damage roll. So it is the direct lens on "did the critic-tail-blindness obs
    gap get filled": at a value cliff, ``active_pko`` near 1.0 means the obs DID contain the OHKO
    signal (any remaining mistake is downstream policy/critic usage, not a missing input); a low
    ``active_pko`` where the active then faints to a direct hit means the BELIEF itself is
    mis-calibrated (an encoder bug to chase). Per our 6 slots the crit-split block holds
    ``[phys_exp, spec_exp, phys_pko_nocrit, spec_pko_nocrit, phys_crit_delta, spec_crit_delta,
    p_outspeed, threat_revealed]`` (the crit risk is the DELTA, pko_crit − pko_nocrit). ``active_pko``
    is the RECONSTRUCTED crit-inclusive ``max(nocrit+delta)`` (preserving the old meaning),
    ``active_pko_nocrit`` the modal line, and ``threat_revealed`` the dominant KO threat's provenance
    (1.0 revealed / <1 prior-guess / 0.0 when no candidate can KO). Explicit 5-field ObsOffsets decode
    with ``active_pko_nocrit``/``threat_revealed`` = None.
    ``active_*`` pull our on-field mon's slot (located via its per-mon active flag)."""
    present: bool                          # any nonzero KO/chip belief on the board
    max_pko: float                         # worst P(KO) across our team (crit-incl. max(phys,spec) per slot)
    active_pko: "float | None"             # crit-inclusive max(phys,spec) P(KO) for our ACTIVE slot
    active_exp: "float | None"             # max(phys,spec) expected-damage-fraction, active slot
    active_outspeed: "float | None"        # P(our active outspeeds the opp), [0,1]
    per_slot_pko: "tuple[float, ...]"      # crit-inclusive max(phys,spec) P(KO) per our team slot
    recovery_rate: float                   # opp recovery-move belief (Suicune-Rest discriminator)
    cures_status: float                    # P(opp Rest) — cures its own status clock
    recovery_known: float                  # 1.0 once a recovery move is revealed
    active_pko_nocrit: "float | None" = None   # modal-line (no-crit) P(KO), active slot — None on old traces
    threat_revealed: "float | None" = None     # dominant-threat provenance, active slot — None on old traces


@dataclass(frozen=True)
class ValueView:
    """The critic's read on this state: recorded V(s), the loaded model's re-run
    V(s) (critic faithfulness), V at the next captured decision, and ΔV (how the
    value moved). ΔV spikes flag where the critic mis-valued / got surprised."""
    recorded: float
    rerun: "float | None"
    next_recorded: "float | None"
    delta: "float | None"
    # PopArt: the V's above are DE-normalized (real return units). On a `--use-popart` run these
    # carry the running (mu, sigma) and the normalized V = (V - mu)/sigma — the critic's own
    # learning scale (~[-1,1], comparable across return-scale drift). All None on a no-PopArt run.
    popart_mu: "float | None" = None
    popart_sigma: "float | None" = None
    normalized_recorded: "float | None" = None
    normalized_rerun: "float | None" = None


@dataclass(frozen=True)
class WinProbView:
    """The win-probability head's read: recorded P(win|s) and ΔP(win) to the next captured decision —
    "how much this turn moved the win odds". The calibrated [0,1] analog of `ValueView`'s recorded V /
    ΔV (the shaped critic's V is expected RETURN, not win odds). None unless the run trained with
    ``--win-prob-mode != none`` (the trace's recorded P(win) is NaN otherwise)."""
    recorded: float
    next_recorded: "float | None"
    delta: "float | None"


@dataclass(frozen=True)
class ValueDistView:
    """The distributional value head's predicted RETURN DISTRIBUTION at this state (v29) — the per-atom
    softmax + its shape stats. The interpretability read the scalar V collapses: a sharp spike =
    confident, a wide spread = uncertain, a bimodal shape = the critic sees a coinflip (e.g. "I win if
    this move hits, else I lose"). Stats are in the head's SUPPORT space (PopArt-normalized on a
    ``--use-popart`` run — the critic's own learning scale); ``mean_real`` de-normalizes E[Z] to real
    return units when PopArt stats are available. None unless the run trained ``--value-dist-mode``."""
    probs: "tuple[float, ...]"             # the per-atom distribution (the histogram bars)
    support: "tuple[float, ...]"           # atom centers (the x-axis), in support space
    mean: float                            # E[Z] = Σ atomsᵢ·probsᵢ (support space)
    std: float                             # spread = the critic's own uncertainty
    p10: float
    p50: float
    p90: float
    entropy: float                         # nats — sharpening over training = the critic committing
    bimodality: float                      # mass OUTSIDE the dominant peak's ±2-bin neighborhood (coinflip ⇒ high)
    mean_real: "float | None" = None       # de-normalized E[Z] (real return) when PopArt present


@dataclass(frozen=True)
class MonState:
    species: str
    hp: str           # "100%" / "75%" / "faint"
    fainted: bool
    status: str = ""  # bundled status+volatiles, e.g. "TOX(5)", "PAR|SUB" ("" when none)
    item: str = ""    # held item (e.g. "choiceband"; "" when unknown/none)
    moves: "tuple[str, ...]" = ()   # known moveset (ours full; opp's revealed-only) — decoded from obs


@dataclass(frozen=True)
class SideBoard:
    active_species: str
    active_hp: str
    status: str                      # "" when none
    boosts: str                      # "" when none
    moves: "tuple[str, ...]"         # active mon's moves (our side only; opp unknown)
    bench: "tuple[MonState, ...]"    # the rest of the (revealed) team
    item: str = ""                   # active mon's held item (our side only; "" when unknown/none)


@dataclass(frozen=True)
class BoardView:
    """The board at this decision (model-free, from the summary): each side's
    active mon (species/hp/status/boosts) + revealed bench, and our moveset."""
    ours: SideBoard
    opp: SideBoard


@dataclass(frozen=True)
class OppFullMon:
    """One opponent mon in the PRIVILEGED full-team view, each fact tagged seen-in-battle or not:
    the team is known from the reconstruction record, but only some of it has been REVEALED on field."""
    species: str
    revealed: bool                                  # this mon has appeared on the field
    active: bool                                    # it's the current opp active
    hp: str                                         # live HP when revealed, else "" (unknown)
    status: str
    item: str                                       # the TRUE held item (privileged)
    item_revealed: bool                             # the item has been shown in-battle
    moves: "tuple[tuple[str, bool], ...]"           # every true move + whether it's been revealed


@dataclass(frozen=True)
class OppFullTeamView:
    """The opponent's WHOLE team (all 6) from the reconstruction record, with each mon / item / move
    tagged revealed-or-not — the Summary OPP TEAM panel. `None` (→ revealed-only panel) when there's
    no privileged team (websocket/older traces without a `reconstruction.json`)."""
    mons: "tuple[OppFullMon, ...]"


@dataclass(frozen=True)
class BeliefSlotView:
    """One still-HIDDEN opponent slot's species belief: the model's top-k guesses,
    `(species, prob)` descending. `slot` is the encoder opp-slot index it filled."""
    slot: int
    top: "tuple[tuple[str, float], ...]"


@dataclass(frozen=True)
class BeliefView:
    """The hidden-opponent species belief at this decision — what the model thinks each STILL-HIDDEN
    opponent mon is (decoded from the belief head's per-slot species logits, either re-computed from
    the loaded model or read from the summary's `belief` block). The slots are ANONYMOUS (a set
    prediction); `BeliefTruthView` is the privileged, slot-MATCHED version. Present ONLY when the
    hidden-opponent belief was enabled (`--opp-belief-aux-coef>0`); `None` otherwise."""
    slots: "tuple[BeliefSlotView, ...]"


@dataclass(frozen=True)
class ExclusiveSlotView:
    """One hidden slot read under the SPECIES CLAUSE — the exclusivity-adjusted counterpart of a
    `BeliefSlotView`, carried BESIDE it and never in place of it.

    `top` is the adjusted top-k. `raw_top1` / `adj_top1` are the model's own most-likely species and
    the adjusted one; `differs` is the only case worth drawing, and `total_variation` says how far
    the row actually moved (½·Σ|adjusted − raw|), so a surface can stay silent on a 0.001 shift.
    `hypothesis` is this slot's entry in the greedy no-duplicates POINT assignment, which can differ
    from `adj_top1` even when the distributions barely move — two slots may legally share an expected
    count below 1 while still both NAMING the same mon."""
    slot: int
    top: "tuple[tuple[str, float], ...]"
    raw_top1: str
    raw_top1_prob: float
    adj_top1: str
    adj_top1_prob: float
    differs: bool
    total_variation: float
    hypothesis: "str | None"
    hypothesis_differs: bool


@dataclass(frozen=True)
class ExclusiveBeliefView:
    """The species-clause READING of the hidden-opponent belief — a display aid, NOT the belief.

    ⚠️ **The model's belief is `BeliefView`, the raw per-slot marginals.** The belief head publishes
    one independent softmax per hidden slot, so nothing in it can express "at most one of you is
    Salamence"; this view applies that constraint after the fact
    (`agents.inference.species_exclusivity`) so a reader is not handed a team no gen3 rule allows. It
    is always shown ALONGSIDE the raw view — replacing the raw view would substitute our arithmetic
    for the model's actual state, which is the interpretability failure this exists to fix, one level
    up.

    `team_hypothesis` is the single most likely hidden team consistent with the clause (greedy
    no-duplicates assignment — `coherent_team_hypothesis` documents why greedy and not Hungarian).
    `max_expected_count` / `illegal_mass` are the RAW belief's incoherence headline: the peak
    per-species expected count over the hidden slots, and Σ_s max(0, E[count(s)] − 1).
    `duplicate_top1` counts hidden slots sharing a top-1 species — a weaker and (measured) far more
    common defect than `max_expected_count > 1`, and the one a reader actually sees.
    `revealed_leak_max` is the largest per-slot mass sitting on an ALREADY-REVEALED species, a
    different and flatly-wrong failure rather than a marginal-vs-joint subtlety.
    `converged` False means the constraint set was unreachable and the adjusted rows do NOT satisfy
    the column cap — a surface must not then present them as clause-consistent."""
    slots: "tuple[ExclusiveSlotView, ...]"
    team_hypothesis: "tuple[str, ...]"
    revealed: "tuple[str, ...]"
    max_expected_count: float
    illegal_mass: float
    duplicate_top1: int
    revealed_leak_max: float
    converged: bool
    iterations: int
    # True when the RAW belief already satisfied the clause — nothing was adjusted, so a surface can
    # say so in one line instead of drawing an identical second table. A stored FIELD rather than a
    # property on purpose: `asdict` drops properties, so the CLI JSON would omit it and every
    # consumer would re-derive the same three-way test — a view deriving a number.
    coherent: bool = True


@dataclass(frozen=True)
class OppIntentOption:
    """One option `α` put mass on: a NAMED believed move of the opponent's, or the `SWITCH` option.

    `is_switch` is carried rather than left for a caller to infer by comparing the name, because two
    surfaces comparing a magic string against a move dex is how one of them ends up wrong."""
    name: str
    p: float
    is_switch: bool
    #: Did the opponent actually DO this? Carried on the option rather than left to a surface to
    #: match, for the same reason as `is_switch`: the recorded action is an id (`drillpeck`) and the
    #: option is a display name (`Drill Peck`), so the comparison needs normalization and a Hidden
    #: Power rule — two surfaces doing that separately is two chances to get it wrong.
    was_actual: bool = False


#: The caveat every surface must print beside a `β` slot name that came from the model's species
#: POSTERIOR rather than from the board. It lives here, not in a template, for the same reason
#: `timeline_entry_text` does: a sentence one surface learns must not go missing on another.
#:
#: Why it exists: `β`'s candidate mask is alive-and-not-active, which INCLUDES revealed bench mons,
#: while the species aux only supervises the *believed* slots — so on a revealed slot the posterior
#: is un-trained. Measured over a 843-battle sentinel sweep (2026-08-19), the posterior-decoded name
#: was a mon not on the opponent's team **at all** in 73.3% of 6,876 pivots (88.3% on revealed
#: slots), and one such label was read as "β predicts porygon2" on a turn where β's slot was the
#: revealed Salamence and β was CORRECT. Recorders from `gen3_beta_revealed_naming_v1` on name a
#: revealed slot off the board and flag it (`revealed: true`); EVERY trace written before that
#: baked the posterior name, and no read-time re-derivation can recover the board — so the honest
#: repair for an old trace is to say what the name is.
BELIEF_NAME_CAVEAT = "believed (posterior decode)"


@dataclass(frozen=True)
class OppIntentCandidate:
    """One `β` candidate: a slot the opponent could bring in, and how much mass `β` puts there.

    `species` names the slot, and WHERE the name came from is the load-bearing part:

    - `revealed=True` — the slot was already on the board at that decision, and the recorder named
      it from the opponent's revealed team (the encoder's own opp-slot order). A fact.
    - `revealed=False` — the model's OWN species posterior
      (`belief_decode.top_species_per_slot`), which is un-supervised on a revealed slot. A guess,
      and `caveat` carries the sentence saying so.

    Every trace written before `gen3_beta_revealed_naming_v1` carries no `revealed` key at all, so
    it reads as `False` — correct, because those names ARE all posterior decodes.

    `caveat` is `None` when there is nothing to qualify: a revealed name, or a slot with no species
    head to name it (which renders as a bare index and is already honest)."""
    slot: int
    p: float
    species: "str | None"
    #: Did the RECORDER name this slot off the board? False on every pre-`gen3_beta_revealed_naming_v1`
    #: trace, which is the honest reading of one.
    revealed: bool = False
    #: `BELIEF_NAME_CAVEAT` when the name is a posterior decode, else None. A stored FIELD rather
    #: than a property, so `asdict` carries it into the CLI JSON and no surface re-derives the test.
    caveat: "str | None" = None


@dataclass(frozen=True)
class OppIntentView:
    """What the model expected the OPPONENT to do at this decision — the v67 `α`/`β` heads
    (`gen3_opp_intent_v1`), model-free from the trace's `opp_intent` block.

    This is the whole interpretability case for the intent heads, and it is the one thing no other
    view carries: a turn where the model played AROUND a Fire Blast and one where it never saw the
    move coming are identical in the board, the timeline and the critic's numbers.

    `alpha` is the ranked distribution over [their believed moves] + SWITCH — already sorted by the
    recorder (`opp_intent.render_alpha`), and NOT re-sorted here. `beta` answers the follow-up: if
    they do switch, who comes in. `None` on any run that did not train the heads
    (`--opp-intent-coef 0`), which is every trace written before v67."""
    alpha: "tuple[OppIntentOption, ...]"
    beta: "tuple[OppIntentCandidate, ...]"
    top: "OppIntentOption | None"       # the single most-expected option
    switch_p: "float | None"            # P(they SWITCH) — None when α did not carry the option
    #: What the opponent ACTUALLY did this turn, as a display string ("Drill Peck", "SWITCH"), or
    #: None when the trace did not record it. The prediction is only readable next to the outcome:
    #: α saying "Drill Peck 41%" is a different fact depending on whether Drill Peck is what came.
    actual: "str | None" = None
    #: True when `actual` is a move α never named — the interesting miss, and NOT the same as α
    #: simply ranking it low.
    actual_unlisted: bool = False


@dataclass(frozen=True)
class OppMoveBelief:
    """One REVEALED opponent mon's MOVE belief, each entry `(move, P(in set))` from the multi-label
    move-belief posterior: `revealed` = the moves it has already shown (their belief should be pinned
    ≈100% under `--move-prior-fusion`, so showing it CONFIRMS the belief is pinning known moves), and
    `believed` = the model's top guesses at its still-UNSEEN moves (the revealed ones filtered out).
    `slot` is the encoder opp-slot index."""
    slot: int
    species: str
    revealed: "tuple[tuple[str, float], ...]"
    believed: "tuple[tuple[str, float], ...]"


@dataclass(frozen=True)
class MoveBeliefView:
    """The model's MOVE belief at this decision (`--move-belief-mode != off`): per REVEALED opponent mon,
    what it thinks the still-UNSEEN moves are (`opp`). Paired in the TUI with the DamageOperator's
    per-our-mon incoming damage; `our_labels` = `(team_slot, species, is_active)` in TEAM-SLOT order so
    those op rows (also team-slot order) can be labeled by species. `None` on a move-belief-off run."""
    opp: "tuple[OppMoveBelief, ...]"
    our_labels: "tuple[tuple[int, str, bool], ...]" = ()


@dataclass(frozen=True)
class OppMonTruth:
    """One opponent mon's PRIVILEGED truth + the model's matched guess. `revealed` = seen by this
    decision (we already know it). For a still-HIDDEN mon, `guess` is the top-k species of the
    Hungarian-matched believed slot, `guessed_right` = our top-1 == this true mon, and `true_rank` is
    the 1-based rank the model gave the TRUE species (−1 if revealed / unmatched)."""
    species: str
    revealed: bool
    guess: "tuple[tuple[str, float], ...]" = ()
    guessed_right: bool = False
    true_rank: int = -1


@dataclass(frozen=True)
class BeliefTruthView:
    """PRIVILEGED belief-vs-truth: the opponent's FULL team (from the trace's `reconstruction.json`
    referee record), each mon tagged revealed/hidden, and for each HIDDEN mon the model's species
    guess — its believed slot Hungarian-matched to the true hidden mons by the SAME species-CE cost
    the training aux loss assigns on. Built only when BOTH the privileged team AND a belief-on
    checkpoint are available; `None` otherwise (then the anonymous `BeliefView` is shown instead)."""
    mons: "tuple[OppMonTruth, ...]"
    n_hidden: int = 0
    n_correct: int = 0     # hidden mons whose true species was the model's top-1 guess


@dataclass(frozen=True)
class SpreadStatRow:
    """One derived stat for one opp slot: the model's BELIEVED value (the DamageOperator's input), the
    TRUE value (L100, from `team_details()` base + IV + EV + nature), and the Smogon usage-prior mean.
    `err` (believed − true) is the otherwise-invisible damage root-cause — a wrong Atk/SpA mis-prices
    every incoming hit. `true`/`prior` are `None` when unavailable (no ground truth / no prior data)."""
    stat: str                       # atk / def / spa / spd / spe
    believed: float
    true: "float | None" = None
    prior: "float | None" = None

    @property
    def err(self) -> "float | None":
        return (self.believed - self.true) if self.true is not None else None


@dataclass(frozen=True)
class SpreadSlotBelief:
    """The model's believed spread for ONE REVEALED opp slot, matched to its TRUE mon by species (exact —
    species is known for a revealed mon, unique under the species clause). `rows` = the 5 derived stats;
    `nature`/`ev_note` annotate the true EV source (e.g. ``adamant · atk252/spe252/hp4``)."""
    slot: int
    species: str
    rows: "tuple[SpreadStatRow, ...]" = ()
    nature: str = ""
    ev_note: str = ""
    matched: bool = False           # True iff the true mon (EVs/nature) was found in team_details


@dataclass(frozen=True)
class SpreadBeliefView:
    """PRIVILEGED spread-belief-vs-truth: per REVEALED opp mon the model's believed DERIVED stats (what the
    `DamageOperator` actually consumes for that mon) next to the true derived stats + the Smogon prior. The
    head predicts spreads for SEEN mons (known species, unknown EVs), so hidden mons are out of scope here.
    `None` unless ``--spread-belief`` AND privileged truth (`reconstruction.json`) are both available;
    without truth the believed column still renders (the `true`/`err`/`prior` cells are then `None`)."""
    slots: "tuple[SpreadSlotBelief, ...]"
    n_slots: int = 0
    mean_abs_err: "float | None" = None   # mean |believed − true| over matched stats (the headline number)


@dataclass(frozen=True)
class BeliefTrajectoryPoint:
    """One decision in a battle's belief REFINEMENT TRAJECTORY (axis B, across-battle turns): the per-hidden-
    slot top-1 species confidence + whether the model's top-1 was the true species, at turn `turn`. Built
    model-free from each decision's summary `belief` block matched to the privileged team.
    `move_entropy` / `believed_atk` / `believed_spe` are populated ONLY when the trace npz carries the
    captured `move_logits` / `spread_belief` arrays (future runs) — the opp-active move-belief Bernoulli
    entropy (should DECAY) + the believed opp-active Atk/Spe — else `None` (species-only trajectory)."""
    inv_index: int
    turn: int
    n_hidden: int
    n_correct: int                  # hidden mons whose top-1 == a STILL-HIDDEN true mon (one-time consumption)
    mean_top1_conf: float           # mean top-1 species confidence over hidden slots
    move_entropy: "float | None" = None     # opp-active move-belief entropy (npz move_logits), axis B
    believed_atk: "float | None" = None     # believed opp-active Atk (npz spread_belief active row)
    believed_spe: "float | None" = None     # believed opp-active Spe (npz spread_belief active row)


@dataclass(frozen=True)
class BeliefTrajectoryView:
    """A battle's belief sharpening over its decisions (axis B): as reveals accumulate the hidden count
    falls and correctness/confidence should rise. `points` is decision-ordered. `None` without a belief-on
    trace + privileged team."""
    points: "tuple[BeliefTrajectoryPoint, ...]"


@dataclass(frozen=True)
class InvocationAnalysis:
    meta: TraceMeta
    inv_index: int
    turn: int
    phase: str
    our_species: str
    opp_species: str
    chosen: str
    has_state: bool
    actions: "tuple[ActionRow, ...]"
    matchups: "MatchupView | None"
    sweep: "InterventionSweep | None"
    saliency: "Saliency | None"
    value_saliency: "Saliency | None"          # |d V(s)/d obs| per block (critic lens)
    threats: "ThreatView | None"
    incoming: "IncomingBeliefView | None"      # the P(KO)/chip belief block (incoming_damage_v1)
    warnings: "tuple[str, ...]"
    # Outcome / value / disagreement (added for the Outcome panel + agent API).
    outcome: dict = field(default_factory=dict)   # raw {our, opp, reward, events}
    value: "ValueView | None" = None
    win_prob: "WinProbView | None" = None          # P(win|s) + ΔP(win) (None unless --win-prob-mode != none)
    value_dist: "ValueDistView | None" = None       # predicted return DISTRIBUTION (None unless --value-dist-mode)
    rerun_argmax: "str | None" = None              # the loaded model's top valid action
    agrees: bool = True                            # rerun_argmax == chosen
    flags: "tuple[str, ...]" = ()                  # switch/uncertain/faint/disagree/cure-skipped
    cure_options: "tuple[str, ...]" = ()           # LEGAL status cures while statused (see `self_cure_options`)
    board: "BoardView | None" = None               # board state at this decision
    next_board: "BoardView | None" = None          # board at the NEXT decision = the resolved "after" state
    obs_mismatch: "tuple[int, int] | None" = None  # (trace_obs_dim, encoder_dim) when they differ → obs-offset
    #                                                 panels (incoming/threat/crit/saliency) are UNRELIABLE
    field: "dict | None" = None                    # weather/spikes/screens (decoded from obs)
    belief: "BeliefView | None" = None             # hidden-opp species belief (anonymous slots) — THE MODEL'S ACTUAL BELIEF
    exclusive_belief: "ExclusiveBeliefView | None" = None  # the SPECIES-CLAUSE reading of `belief` — a display
    #                                                 aid computed at read time, never a replacement (see the dataclass)
    opp_intent: "OppIntentView | None" = None      # α/β: what the model expected THEM to do; None unless --opp-intent-coef>0
    belief_truth: "BeliefTruthView | None" = None  # privileged truth + slot-matched guess (None unless recon+belief)
    opp_full_team: "OppFullTeamView | None" = None  # WHOLE opp team + revealed-or-not tags (None w/o reconstruction)
    damage_op: "dict | None" = None                # unified DamageOperator view (incoming + outgoing); None unless --damage-op
    move_belief: "MoveBeliefView | None" = None     # what the model thinks the revealed opp's UNSEEN moves are; None unless --move-belief-mode
    spread_belief: "SpreadBeliefView | None" = None  # believed vs true opp DERIVED stats; None unless --spread-belief
    switch_in_outgoing: "SwitchInOutgoingView | None" = None  # forced-switch: each alive candidate's best move vs the opp active (📋); None off a forced switch / w/o recon
    opp_switched_to: "str | None" = None             # species the opp VOLUNTARILY pivoted in this turn (our move resolved vs it, not the active)


@dataclass(frozen=True)
class SwitchInOutgoingRow:
    species: str
    hp: str            # the candidate's live HP ("100%")
    move: str          # its best (highest-P(KO), then highest-damage) BP-damaging move
    low: float         # low-roll damage as a % of the opp active's MAX HP
    high: float        # high-roll (R=100) damage %
    pko: float         # P(KO) this hit, vs the opp active's REMAINING HP
    type_mult: float   # the move's type effectiveness vs the opp active
    outspeed: float    # P(this candidate outspeeds the opp active)


@dataclass(frozen=True)
class SwitchInOutgoingView:
    """Per ALIVE switch-in candidate, its best damaging move's expected damage to the opp active —
    the forced-switch counterpart to the op's all-zero OUTGOING block. CPU-computed from the
    privileged true spreads (needs a `reconstruction.json` sibling); rows sorted best-KO first."""
    opp_species: str
    opp_hp: str
    rows: "tuple[SwitchInOutgoingRow, ...]"
