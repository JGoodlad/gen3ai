"""Pure forensic-replay analysis — the single source of truth for both the
``probe_replay.py`` CLI and the Textual prober TUI.

No printing, no Textual, no file IO beyond reading the already-loaded summary
dict / npz arrays. Every torch call goes through the injected ``model`` object
(see ``model.ProbeModel``), so the whole engine is testable with a fake.

The analysis for one invocation (decision point) mirrors what the original CLI
computed inline:
  - faithfulness: recorded action probs (from the summary) vs a live re-run,
  - matchups: the active mon's per-move type multipliers the model saw,
  - intervention: sweep the chosen move's matchup 0×→4× and watch P shift,
  - saliency: |d logit(chosen) / d obs| aggregated per obs block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from agents import gen3_data
from agents.action.constants import MOVE_END, MOVE_START
from agents.inference.belief_decode import BELIEF_TOPK
from agents.observation import incoming_damage as _inc

# Sweep points for the intervention (×-multipliers); stored /4 into the obs.
_SWEEP_MULTIPLIERS = (0.0, 1.0, 2.0, 4.0)
# A decision is "uncertain" when even the chosen (top) action's recorded prob is below
# this — a genuine tossup (≈3-way), not merely a non-dominant pick (common in Pokémon).
UNCERTAIN_THRESHOLD = 0.34


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


# The matchup matrices are 6×4×6 (opp/our mon × move slot × the other side) — mirrors
# reactive.py's `their_matchups`/`our_matchups` 144-dim blocks.
_TEAM_SIZE = 6
_MATCHUP_DIM = _TEAM_SIZE * 4 * _TEAM_SIZE   # 144


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


def _dist_quantile(support, cdf, t: float) -> float:
    idx = int(np.searchsorted(cdf, t))
    return float(support[min(idx, len(support) - 1)])


def build_value_dist(npz, i: int, support, popart=None) -> "ValueDistView | None":
    """The distributional value head's per-decision return distribution at decision ``i``. None when the
    array is absent (old trace / the run had no value-dist head → the KeyError "unavailable" path), this
    row wasn't captured (all-NaN), or the support/trace bin counts disagree (config drift). ``support`` =
    ``(vmin, vmax, bins)`` from the loaded model; ``popart`` = optional ``(mu, sigma)`` to denormalize
    E[Z]. Pure (numpy only) — the single source the app histogram + the ``analyze`` CLI both render."""
    try:
        arr = npz["value_dist"]
    except KeyError:
        return None
    if not (0 <= i < len(arr)):
        return None
    probs = np.asarray(arr[i], dtype=np.float64)
    if probs.size == 0 or np.isnan(probs).any():
        return None
    vmin, vmax, bins = support
    if int(bins) != probs.size:
        return None
    z = np.linspace(float(vmin), float(vmax), int(bins))
    p = probs / max(float(probs.sum()), 1e-8)
    mean = float((p * z).sum())
    std = float(np.sqrt(max(float((p * (z - mean) ** 2).sum()), 0.0)))
    cdf = np.cumsum(p)
    peak = int(np.argmax(p))
    lo, hi = max(0, peak - 2), min(len(p), peak + 3)
    bimodality = float(max(0.0, 1.0 - float(p[lo:hi].sum())))
    mean_real = None
    if popart is not None and popart[1]:
        mean_real = mean * float(popart[1]) + float(popart[0])
    return ValueDistView(
        probs=tuple(float(x) for x in p), support=tuple(float(x) for x in z),
        mean=mean, std=std,
        p10=_dist_quantile(z, cdf, 0.10), p50=_dist_quantile(z, cdf, 0.50),
        p90=_dist_quantile(z, cdf, 0.90),
        entropy=float(-(p * np.log(p + 1e-12)).sum()), bimodality=bimodality, mean_real=mean_real,
    )


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
class OppIntentOption:
    """One option `α` put mass on: a NAMED believed move of the opponent's, or the `SWITCH` option.

    `is_switch` is carried rather than left for a caller to infer by comparing the name, because two
    surfaces comparing a magic string against a move dex is how one of them ends up wrong."""
    name: str
    p: float
    is_switch: bool


@dataclass(frozen=True)
class OppIntentCandidate:
    """One `β` candidate: a slot the opponent could bring in, and how much mass `β` puts there.

    `species` is the model's OWN species posterior for that slot (`belief_decode.top_species_per_slot`
    at capture time) — the same content-addressing `β`'s target uses, so the slot reads as a mon. It
    is `None` on a checkpoint with no species head, and it is a BELIEF even when the slot is
    revealed: `β`'s candidate mask is alive-and-not-active, which includes mons already seen, and
    the species aux only supervises the believed slots."""
    slot: int
    p: float
    species: "str | None"


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
    belief: "BeliefView | None" = None             # hidden-opp species belief (anonymous slots)
    opp_intent: "OppIntentView | None" = None      # α/β: what the model expected THEM to do; None unless --opp-intent-coef>0
    belief_truth: "BeliefTruthView | None" = None  # privileged truth + slot-matched guess (None unless recon+belief)
    opp_full_team: "OppFullTeamView | None" = None  # WHOLE opp team + revealed-or-not tags (None w/o reconstruction)
    damage_op: "dict | None" = None                # unified DamageOperator view (incoming + outgoing); None unless --damage-op
    move_belief: "MoveBeliefView | None" = None     # what the model thinks the revealed opp's UNSEEN moves are; None unless --move-belief-mode
    spread_belief: "SpreadBeliefView | None" = None  # believed vs true opp DERIVED stats; None unless --spread-belief
    switch_in_outgoing: "SwitchInOutgoingView | None" = None  # forced-switch: each alive candidate's best move vs the opp active (📋); None off a forced switch / w/o recon
    opp_switched_to: "str | None" = None             # species the opp VOLUNTARILY pivoted in this turn (our move resolved vs it, not the active)


# ---------------------------------------------------------------------------
# Small parsing / npz helpers
# ---------------------------------------------------------------------------

def parse_pct(s: str) -> float:
    """'92.1%' -> 0.921. Tolerates a missing '%'."""
    return float(str(s).rstrip("%")) / 100.0


def _has_state(npz, i: int) -> bool:
    try:
        flags = npz["has_state"]
    except KeyError:
        return True
    return bool(flags[i])


def _npz_value(npz, i: int) -> "float | None":
    try:
        vals = npz["values"]
    except KeyError:
        return None
    return float(vals[i]) if 0 <= i < len(vals) else None


def _npz_array(npz, key: str):
    """A captured per-decision array (e.g. `move_logits` / `spread_belief`) or None when the key is absent
    (the head was off / an older trace). `npz` may be None (no captured state)."""
    if npz is None:
        return None
    try:
        return npz[key]
    except (KeyError, TypeError, IndexError):
        return None


def _npz_win_prob(npz, i: int) -> "float | None":
    """Recorded P(win) at decision i. None when the array is absent (old trace) or NaN (the run had
    no win-prob head / this decision wasn't captured) — so the prober shows P(win) only when real."""
    try:
        vals = npz["win_probs"]
    except KeyError:
        return None
    if not (0 <= i < len(vals)):
        return None
    v = float(vals[i])
    return None if np.isnan(v) else v


# A real STATUS (curable by Refresh / Heal Bell), as opposed to a VOLATILE. The recorder bundles both
# into one display string ("TOX(2)|TAUNT"), so a cure check must split them — Taunt is not curable, and
# it also makes every status move ILLEGAL, which the action-validity check below picks up on its own.
_CURABLE_STATUSES = frozenset({"TOX", "PSN", "BRN", "PAR", "SLP", "FRZ"})


def has_curable_status(status: str) -> bool:
    """True when a recorder status string ("TOX(2)|TAUNT", "PAR", "") carries a real status (not just
    volatiles). Pure — the recorder's own bundling format is the only contract."""
    return any(tok.split("(")[0].strip().upper() in _CURABLE_STATUSES
               for tok in str(status or "").split("|"))


def is_status_cure(move_id: str) -> bool:
    """True for a move that CLEARS status — Refresh (self) or Heal Bell / Aromatherapy (team).
    Read from the DATA facade's `curesSelfStatus` / `curesTeamStatus`, never a hardcoded id list
    (`data/` is the source of truth). Rest is NOT one: it cures by *inflicting* sleep."""
    md = gen3_data.moves.get(str(move_id or "").split(":")[0])
    return bool(md and (md.cures_self_status or md.cures_team_status))


def self_cure_options(inv: dict) -> "tuple[str, ...]":
    """The LEGAL status-cure moves at this decision, but ONLY when our active actually carries a
    curable status — i.e. the cures that would have *done something*. Empty otherwise.

    Model-free (summary only). This is the "was a cure on the table" question the raw action list
    can't answer: an illegal (Taunted / no-PP) cure and a cure with nothing to cure both read as a
    move label, and neither is a real option."""
    if not has_curable_status((inv.get("our") or {}).get("status")):
        return ()
    return tuple(lbl for lbl, a in (inv.get("actions") or {}).items()
                 if a.get("valid") and is_status_cure(lbl))


def summary_flags(inv: dict, uncertain_threshold: float = UNCERTAIN_THRESHOLD) -> "tuple[str, ...]":
    """Cheap, model-free per-invocation flags (used by the TUI list + agent API to
    jump to interesting decisions): a switch, an uncertain (low top-prob) call, or a
    turn whose recorded events include a faint."""
    flags = []
    chosen = inv.get("chosen", "")
    if chosen.startswith("switch"):
        flags.append("switch")
    acts = inv.get("actions", {})
    if chosen in acts and parse_pct(acts[chosen].get("prob", "0%")) < uncertain_threshold:
        flags.append("uncertain")
    events = (inv.get("outcome") or {}).get("events") or []
    if any("faint" in str(e).lower() for e in events):
        flags.append("faint")
    # The OPPONENT voluntarily pivoted this turn → our move RESOLVED against a switch-IN, not the active we
    # computed damage against (the "computed-vs ≠ resolved-vs" trap — see `opp_voluntary_switch`).
    if opp_voluntary_switch(inv):
        flags.append("opp-switch")
    # Statused, a LEGAL cure was on the table, and we did something else (Recover heals HP but never
    # clears status — a distinction that is invisible in a move list).
    if self_cure_options(inv) and not is_status_cure(chosen):
        flags.append("cure-skipped")
    return tuple(flags)


# Recorder bench format: "species(hp%)" / "species(hp%,STATUS)" / "species(faint)"
# where STATUS bundles status+volatiles (e.g. "TOX(5)", "PAR|SUB") — see battle_recorder.
_BENCH_RE = re.compile(r"^(.+?)\((.+)\)$")          # "metagross(100%)" / "tyranitar(faint)"
_MOVE_PLACEHOLDER_RE = re.compile(r"move\d$")        # "move0".."move3" filler labels


def _norm_species(s: str) -> str:
    """Lenient species key — lowercase, alnum only — so an item map keyed by an obs-decoded
    display name ('Tyranitar') matches a board id ('tyranitar') regardless of source form."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _team_entry(team: dict, species: str) -> dict:
    """Per-mon {item, moves} from a leniently-keyed team map ({} when absent)."""
    return team.get(_norm_species(species), {}) if team else {}


_LOG_BLOCK_RE = re.compile(r'class="battle-log-data"[^>]*>(.*?)</script>', re.DOTALL)


def parse_protocol_log(html: str) -> "tuple[str, ...]":
    """Extract the raw Showdown protocol lines (``|move|…`` / ``|-damage|…`` / ``|turn|N``) from a
    ``replay.html``'s ``battle-log-data`` script block — the same browser-watchable log, surfaced in
    the prober so each decision's raw events are visible. Returns the ``|``-lines in order (the whole
    body if the expected block isn't found, so a format tweak degrades rather than blanks)."""
    m = _LOG_BLOCK_RE.search(html or "")
    body = m.group(1) if m else (html or "")
    return tuple(ln.rstrip("\r\n") for ln in body.splitlines() if ln.startswith("|"))


def protocol_for_turn(lines: "tuple[str, ...]", turn: int) -> "tuple[str, ...]":
    """The protocol slice for one decision's turn: every line from ``|turn|N`` up to (not incl.)
    ``|turn|N+1``. Pure — pairs `parse_protocol_log` (the caller does the file IO) with a decision's
    ``turn`` so the raw events between this decision and the next are shown in order."""
    out, cur = [], 0
    for ln in lines or ():
        if ln.startswith("|turn|"):
            try:
                n = int(ln.split("|")[2])
            except (IndexError, ValueError):
                n = cur
            if n > turn:
                break
            cur = n
        if cur == turn:
            out.append(ln)
    return tuple(out)


def build_our_hp_types(team_details: "list[dict] | None") -> "dict[str, str]":
    """`{norm_species: 'hiddenpower(bug)'}` from a reconstruction `team_details` list — the typed
    Hidden Power display id for each own mon, so a bare own HP (all the request carries before reveal)
    can be typed. Pure; the caller does the reconstruction-file IO (mirrors the opp-team load)."""
    out: "dict[str, str]" = {}
    for m in team_details or []:
        for mv in m.get("moves", ()) or ():
            s = str(mv)
            if s.startswith("hiddenpower") and s != "hiddenpower":
                out[_norm_species(m.get("species", ""))] = f"hiddenpower({s[len('hiddenpower'):]})"
    return out


def _retype_hp(moves: "tuple[str, ...]", species: str, hp_map: "dict | None") -> "tuple[str, ...]":
    """Replace a bare ``hiddenpower`` in a KNOWN-OWN moveset with its true typed display form
    (``hiddenpower(bug)``) from ``hp_map`` (norm-species → formatted id, built from the reconstruction
    record). Showdown's request carries only the bare id for an UNREVEALED own Hidden Power (the type
    is IV-derived, not in the request), so without this our own mons show an untyped HP until they use
    it. ``hp_map`` is None for the OPPONENT side and for non-reconstruction traces — an opponent's
    un-revealed HP MUST stay bare (no leak), so the retype only ever runs on our own team."""
    typed = (hp_map or {}).get(_norm_species(species))
    if not typed:
        return tuple(moves)
    return tuple(typed if str(m) == "hiddenpower" else m for m in moves)


def _parse_bench(s: str, team: "dict | None" = None, hp_map: "dict | None" = None) -> "tuple[MonState, ...]":
    team = team or {}
    out = []
    for chunk in (s or "").split(", "):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _BENCH_RE.match(chunk)
        if m:
            species, inside = m.group(1), m.group(2)
            e = _team_entry(team, species)
            item, moves = e.get("item", ""), _retype_hp(tuple(e.get("moves", ())), species, hp_map)
            if "faint" in inside.lower():
                out.append(MonState(species, "faint", True, "", item, moves))
            else:
                # "hp%[,STATUS]" — the status tail (incl. any volatiles) is comma-separated.
                hp, _, status = inside.partition(",")
                out.append(MonState(species, hp.strip(), False, status.strip(), item, moves))
        else:
            e = _team_entry(team, chunk)
            out.append(MonState(chunk, "?", False, "", e.get("item", ""),
                                _retype_hp(tuple(e.get("moves", ())), chunk, hp_map)))
    return tuple(out)


def _side_board(side: dict, moves: "tuple[str, ...]", team: "dict | None" = None,
                hp_map: "dict | None" = None) -> SideBoard:
    team = team or {}
    species = side.get("species", "")
    e = _team_entry(team, species)
    return SideBoard(
        active_species=species,
        active_hp=side.get("hp", "?"),
        status=side.get("status", "") or "",
        boosts=side.get("boosts", "") or "",
        # our active's moves come from the trace actions; the opp active's (and any side with no
        # trace moves) fall back to the obs-decoded revealed moveset. `hp_map` (our side only) types
        # a bare own Hidden Power the request couldn't.
        moves=_retype_hp(moves or tuple(e.get("moves", ())), species, hp_map),
        bench=_parse_bench(side.get("bench", ""), team, hp_map),
        item=e.get("item", ""),
    )


def _our_items(summary: dict) -> "dict[str, dict]":
    """species → {item, moves:()} from the summary's top-level teams block (our side only; opp
    items aren't in it; moves aren't in it at all). Items recorded at battle end — held items
    (Choice Band / Leftovers) are exact; a consumed berry reads 'none' (dropped to ''). The obs
    decode (`describe_team`) supersedes this per-turn when a state is captured — no-state fallback."""
    out = {}
    for m in (summary.get("teams", {}) or {}).get("ours", []) or []:
        item = str(m.get("item", "") or "")
        if item and item.lower() != "none":
            out[str(m.get("species", ""))] = {"item": item, "moves": ()}
    return out


def _merge_team(base: dict, obs_team: dict) -> "dict[str, dict]":
    """Overlay the per-turn obs decode onto the summary-derived team, keyed leniently by species.
    The obs only OVERRIDES a field when it carries info — an empty obs item must NOT erase a known
    item from the summary teams block (the bug where our own bench mon showed no item because its
    obs slot decoded blank). Returns ``{norm_species: {item, moves}}``."""
    out: "dict[str, dict]" = {}
    for sp, e in (base or {}).items():
        out[_norm_species(sp)] = {"item": e.get("item", ""), "moves": tuple(e.get("moves", ()))}
    for sp, e in (obs_team or {}).items():
        cur = out.setdefault(_norm_species(sp), {"item": "", "moves": ()})
        if e.get("item"):
            cur["item"] = e["item"]
        if e.get("moves"):
            cur["moves"] = tuple(e["moves"])
    return out


def build_board(inv: dict, team: "dict | None" = None,
                our_hp_types: "dict | None" = None) -> BoardView:
    """Board state at a decision — model-free, parsed from the summary invocation. ``team``
    (species → {item, moves}) annotates BOTH sides; keys are matched leniently (:func:`_norm_species`)
    so an obs-decoded name resolves against a board id. ``our_hp_types`` (norm-species → typed HP
    display id, from the reconstruction record) types a bare own Hidden Power — OUR side only."""
    norm = {_norm_species(k): v for k, v in (team or {}).items()}
    labels = list(inv.get("actions", {}).keys())
    moves = tuple(k for k in labels[MOVE_START:MOVE_END] if not _MOVE_PLACEHOLDER_RE.fullmatch(k))
    return BoardView(ours=_side_board(inv.get("our", {}), moves, norm, our_hp_types),
                     opp=_side_board(inv.get("opp", {}), (), norm))


# ── Result timeline (the RESULT panel's data model) ──────────────────────────────────────────────
# The recorder stores each side's action + that mon's OWN net HP change ("hp_delta"). Rendered
# naively ("we icebeam (-72%)") it misreads: a mon's HP loss is dealt by the OPPONENT's move, not its
# own, so the damage shows on the wrong line and move order needs a confusing "«1st»" tag. This builds
# an ordered, one-line-per-action timeline that RE-ATTRIBUTES each mon's HP loss to the opponent's
# move that caused it, Showdown-battle-log style — "opp rockslide did 73% (salamence 100% → 27%)" —
# so the lines read top-to-bottom in execution order.

_SENT_IN = "_sent_in"
_SEP = " → "                       # the " → " the recorder uses to pack a forced replacement
_FAINT_EVENT_RE = re.compile(r"^(our|opp):(.+):fainted$")
_STATUS_EVENT_RE = re.compile(r"^(our|opp):(.+):([A-Z]{3})$")   # our:milotic:PAR, opp:swampert:TOX
_SWITCH_IN_HIT_MAX_HP = 90.0       # a switched-in mon below this clearly took our hit (vs a sand tick)


def _is_attack(move_id: "str | None") -> bool:
    """Whether a recorded move id is a damaging ATTACK — positive-BP, fixed-damage (Seismic Toss /
    Night Shade), variable-power (Return), or Hidden Power (the bare id reads BP 0 but is a real
    attack). Uses the same ``_multiplier_meaningful`` predicate the matchup panel does, so a real hit
    is never dropped from the attribution and a whiffed attack can be flagged 'missed'/'no effect'."""
    return _multiplier_meaningful((move_id or "").lower())


def _no_effect_reason(move_id: "str | None", effectiveness: "str | None",
                      outcome: "str | None" = None) -> "str | None":
    """Why a move that produced NO visible effect did nothing — 'immune' / 'missed' / 'failed' — or
    None for a move whose effect is legitimately invisible here (hazards / heal / boost / Protect),
    which must NOT be flagged. Only an ATTACK (should deal damage) or a status-inflicting move (should
    apply a status) is annotated. ``outcome`` is the RECORDED move fate (gen3_move_outcome_v1:
    'hit'/'miss'/'fail') — preferred when present, so 'missed'/'failed' is a fact; a type immunity
    (a hit that did nothing) reads 'immune' first; only when the outcome wasn't decoded (model-free /
    older trace) do we fall back to inferring a miss from the move's accuracy."""
    mid = (move_id or "").lower()
    md = gen3_data.moves.get(mid)
    if not (_is_attack(mid) or (md and md.status_inflicted)):
        return None
    if effectiveness == "immune":
        return "immune"
    if outcome == "miss":
        return "missed"
    if outcome == "fail":
        return "failed"
    if outcome == "hit":                 # connected but no damage/status landed — not a miss
        return "failed"
    if md and md.accuracy and md.accuracy < 100 and not md.never_miss:   # fallback: no recorded fate
        return "missed"
    return "failed"


def _parse_outcome_action(action: "str | None") -> dict:
    """Structure a recorded action string: a bare move (``icebeam``), a voluntary switch
    (``switched_to:skarmory``), a move that ended in a forced replacement
    (``hiddenpower → metagross_sent_in``), a bare post-faint replacement with no move
    (``claydol_sent_in`` — the mon fainted before acting), or a no-op (``none``/``unknown``)."""
    a = (action or "").strip()
    if a in ("", "none", "unknown", "-"):
        return {"kind": "none"}
    if a.startswith("switched_to:"):
        return {"kind": "switch", "switch_to": a.split(":", 1)[1]}
    if a.endswith(_SENT_IN):
        head, sep, tail = a.partition(_SEP)
        if sep:                                   # "<move> → <X>_sent_in"
            return {"kind": "move", "move": head, "sent_in": tail[:-len(_SENT_IN)]}
        return {"kind": "send_in", "sent_in": a[:-len(_SENT_IN)]}   # bare "<X>_sent_in"
    return {"kind": "move", "move": a}


def opp_voluntary_switch(inv: dict) -> "str | None":
    """The species the OPPONENT VOLUNTARILY switched IN on this decision's turn (model-free, from the
    recorded opp action), else `None`. A voluntary pivot means our move RESOLVED against this switch-in —
    NOT the active mon we computed our damage against — the source of the "computed-vs ≠ resolved-vs"
    confusion when reading a decision (e.g. Earthquake computed vs Snorlax, then resolved into a Levitate
    Claydol the opp pivoted in). A forced post-faint replacement (`_sent_in`) is NOT a voluntary pivot."""
    opp_action = ((inv.get("outcome") or {}).get("opp") or {}).get("action")
    pa = _parse_outcome_action(opp_action)
    return pa.get("switch_to") if pa.get("kind") == "switch" else None


def _pct(hp: "str | float | None") -> "float | None":
    """Parse a recorded HP string (``'28%'`` → 28.0, ``'faint'`` → 0.0) to a percent, else None."""
    if isinstance(hp, (int, float)):
        return float(hp)
    s = str(hp or "").strip().lower()
    if not s:
        return None
    if "faint" in s:
        return 0.0
    try:
        return float(s.rstrip("%"))
    except ValueError:
        return None


def _loss_pct(hp_delta: "str | None") -> "float | None":
    """Magnitude of a NEGATIVE recorded ``hp_delta`` (``'-72%'`` → 72.0); None for a gain/zero."""
    v = _pct(str(hp_delta or "").lstrip("+"))
    return -v if (v is not None and v < 0) else None


def build_result_timeline(outcome: dict, our_species: str, opp_species: str, phase: str = "",
                          our_hp_before=None, opp_hp_before=None,
                          our_hp_after=None, opp_hp_after=None) -> "list[dict]":
    """Ordered, one-line-per-action model of what HAPPENED after a decision (the RESULT panel). Pure.

    Re-attributes each side's HP loss to the OPPONENT's move that dealt it, and pairs it with the
    target's before→after HP (``before = after + damage``), so each line reads like a battle log:
    ``{side, kind, move, damage, target, hp_before, hp_after, crit, boost, cant, status, resulting,
    no_effect, switch_to, sent_in}``. ``resulting`` marks a hit on a switch-IN where only the after-HP
    is known (the recorded delta can't price it across the switch); ``no_effect`` (``immune`` /
    ``missed`` / ``failed``) explains a move that did NOTHING visible. Ordered by execution: a voluntary
    switch resolves before a move; otherwise
    the TurnDelta ``move_order`` (folded from the real event-log sequence). When BOTH sides moved but
    ``move_order`` is absent (a no-state / model-free decision) the order is unknown — the move entries
    carry ``order_certain=False`` so the renderer drops the implied sequence rather than guessing."""
    out = outcome or {}
    our, opp = out.get("our") or {}, out.get("opp") or {}
    if not (our or opp):
        return []

    events = [str(e) for e in (out.get("events") or [])]
    faints = {(m.group(1), _norm_species(m.group(2)))
              for m in (_FAINT_EVENT_RE.match(e) for e in events) if m}
    statuses = {(m.group(1), _norm_species(m.group(2))): m.group(3)
                for m in (_STATUS_EVENT_RE.match(e) for e in events) if m}
    consumed: set = set()

    pa_our, pa_opp = _parse_outcome_action(our.get("action")), _parse_outcome_action(opp.get("action"))
    # A move lands on the opponent's active — redirected to its switch-IN when the opponent
    # VOLUNTARILY switched (that resolves first); a forced "_sent_in" does NOT redirect (the fainting
    # mon ate the hit, the replacement is a separate line).
    our_target = pa_opp.get("switch_to") if pa_opp["kind"] == "switch" else opp_species
    opp_target = pa_our.get("switch_to") if pa_our["kind"] == "switch" else our_species

    def _move_entry(side, pa, crit, boost, cant, target, delta, before, after, switched_in, eff, fate):
        e = {"side": side, "kind": "move", "move": pa.get("move", ""), "crit": bool(crit),
             "boost": boost or "", "cant": cant, "target": "", "damage": "", "hp_before": "",
             "hp_after": "", "status": "", "resulting": False, "no_effect": "",
             "switch_to": "", "sent_in": ""}
        if cant:
            return e
        recip_side = "opp" if side == "we" else "our"
        key = (recip_side, _norm_species(target))
        fainted = key in faints or str(delta).strip() == "-100%"
        dmg = _loss_pct(delta)
        if _is_attack(pa.get("move")) and (fainted or dmg is not None):
            e["target"] = target
            if fainted:
                consumed.add(key)
                b = dmg if dmg is not None else before
                e["damage"] = f"{dmg:.0f}%" if dmg is not None else ""
                e["hp_before"] = f"{b:.0f}%" if isinstance(b, (int, float)) else ""
                e["hp_after"] = "faint"
            else:
                aft = after if after is not None else max(0.0, (before or 0.0) - dmg)
                e["damage"], e["hp_after"] = f"{dmg:.0f}%", f"{aft:.0f}%"
                e["hp_before"] = f"{min(100.0, aft + dmg):.0f}%"
        elif _is_attack(pa.get("move")) and switched_in and after is not None and after < _SWITCH_IN_HIT_MAX_HP:
            # The opponent VOLUNTARILY switched, so the recorded hp_delta (≈0, it compares the mon
            # that LEFT) can't price the hit on the switch-IN. The next board's HP is still truth —
            # show the RESULTING hp ("→ celebi (now 11%)") rather than dropping the attack entirely.
            e["target"], e["hp_after"], e["resulting"] = target, f"{after:.0f}%", True
        if key in statuses:                          # a status this move applied (own line or alongside dmg)
            e["target"], e["status"] = target, statuses[key]
        # A move that produced NOTHING visible: say WHY (missed / no effect / immune) so a blank line
        # never reads as "data missing". Silent for utility moves (hazards/heal/boost — reason None).
        if not (e["damage"] or e["status"] or e["hp_after"]):
            reason = _no_effect_reason(pa.get("move"), eff, fate)
            # ...UNLESS the target SWITCHED IN and nothing recorded the move's fate. Then the
            # recorded hp_delta compares the mon that LEFT, so it cannot price the hit either way —
            # and `_no_effect_reason`'s last resort is to infer a miss from the move's accuracy,
            # which turns "we have no evidence" into the confident claim "it missed". MEASURED: a
            # Meteor Mash into a Jirachi switch-in read `— missed` while the battle's own protocol
            # log showed -resisted / -damage 84/100 / -boost atk (the switch-in then healed to
            # exactly 90% on Leftovers, so the resulting-HP branch above just missed its threshold).
            # An absent explanation is honest; a wrong one is worse than none.
            if switched_in and not fate and reason == "missed":
                reason = None
            e["no_effect"] = reason or ""
        return e

    def _entry_for(side):
        if side == "we":
            pa, crit, boost, cant = pa_our, out.get("our_crit"), out.get("our_boost", ""), out.get("our_cant")
            actor, target, delta = our_species, our_target, opp.get("hp_delta")
            before, after, eff = _pct(opp_hp_before), _pct(opp_hp_after), out.get("our_effectiveness")
            switched_in, fate = pa_opp["kind"] == "switch", out.get("our_move_outcome")  # our move lands on the opp's switch-IN
        else:
            pa, crit, boost, cant = pa_opp, out.get("opp_crit"), out.get("opp_boost", ""), out.get("opp_cant")
            actor, target, delta = opp_species, opp_target, our.get("hp_delta")
            before, after, eff = _pct(our_hp_before), _pct(our_hp_after), out.get("opp_effectiveness")
            switched_in, fate = pa_our["kind"] == "switch", out.get("opp_move_outcome")
        if pa["kind"] == "none":
            return None
        if pa["kind"] == "switch":
            return {"side": side, "kind": "switch", "actor": actor, "switch_to": pa.get("switch_to", "")}
        if pa["kind"] == "send_in":
            return {"side": side, "kind": "send_in", "sent_in": pa.get("sent_in", "")}
        e = _move_entry(side, pa, crit, boost, cant, target, delta, before, after, switched_in, eff, fate)
        e["actor"] = actor
        return e

    entries: "list[dict]" = []
    if phase == "forced_switch":
        # Our post-faint replacement choice; the opponent doesn't act → just the send-in.
        if pa_our["kind"] == "switch":
            entries.append({"side": "we", "kind": "send_in", "sent_in": pa_our.get("switch_to", "")})
    else:
        we_e, opp_e = _entry_for("we"), _entry_for("opp")
        # Execution order: a voluntary switch precedes a move; else the TurnDelta move_order; else ours.
        we_first = True
        if pa_opp["kind"] == "switch" and pa_our["kind"] == "move":
            we_first = False
        elif pa_our["kind"] != "switch" and out.get("move_order") == "opp_first":
            we_first = False
        entries = [e for e in ([we_e, opp_e] if we_first else [opp_e, we_e]) if e is not None]
        # Order certainty: top-to-bottom is REAL only when a voluntary switch fixes it (switches
        # resolve first) or the TurnDelta recorded move_order. If BOTH sides actually moved (a canted
        # side didn't) and move_order is absent — a no-state / model-free decision — we don't know who
        # went first, so flag it and let the renderer drop the implied sequence instead of guessing.
        def _moved(e):
            return bool(e and e.get("kind") == "move" and not e.get("cant"))
        order_certain = (out.get("move_order") in ("we_first", "opp_first")) or not (
            _moved(we_e) and _moved(opp_e))
        for e in entries:
            if e.get("kind") == "move":
                e["order_certain"] = order_certain
        # A move that ended in a forced replacement → surface the send-in as its own trailing line.
        for pa, sd in ((pa_our, "we"), (pa_opp, "opp")):
            if pa["kind"] == "move" and pa.get("sent_in"):
                entries.append({"side": sd, "kind": "send_in", "sent_in": pa["sent_in"]})

    # Any faint NOT already shown as a move's target (self-KO, residual, opp action unknown).
    for sd, sp in faints:
        if (sd, sp) not in consumed:
            entries.append({"side": ("we" if sd == "our" else "opp"), "kind": "faint", "actor": sp})
    return entries


def _timeline_for(inv: dict, next_board: "BoardView | None", outcome: dict) -> "list[dict]":
    """`build_result_timeline` wired to a decision: the actives' HP this turn (before) + the resolved
    HP at the next decision (after, from ``next_board``)."""
    return build_result_timeline(
        outcome, inv.get("our", {}).get("species", ""), inv.get("opp", {}).get("species", ""),
        inv.get("phase", ""),
        our_hp_before=inv.get("our", {}).get("hp"), opp_hp_before=inv.get("opp", {}).get("hp"),
        our_hp_after=(next_board.ours.active_hp if next_board else None),
        opp_hp_after=(next_board.opp.active_hp if next_board else None),
    )


# Plain-language for a "couldn't move" (cant) reason decoded from the TurnDelta, and for a move that
# produced nothing visible. Both live HERE rather than in a renderer: the TUI paints them with Rich
# styles, the web/CLI want the same words as plain text, and two copies of this vocabulary would
# drift the moment one surface learns a new reason.
CANT_PHRASE = {"slp": "asleep", "frz": "frozen", "par": "fully paralyzed", "flinch": "flinched",
               "recharge": "recharging", "nopp": "no PP", "truant": "loafing",
               "attract": "immobilized", "taunt": "taunted", "disable": "disabled",
               "flinched": "flinched"}
NO_EFFECT_TEXT = {"immune": "no effect (immune)", "missed": "missed", "failed": "no effect"}


def cant_phrase(cant: str) -> str:
    return CANT_PHRASE.get(str(cant).lower(), str(cant))


def surprise_phrase(td: float) -> str:
    """Plain-language reading of the TD-surprise δ = r + γV(s′) − V(s), so the ML term is
    self-explaining wherever it is shown: negative δ = the turn went worse than the critic predicted.

    Lives HERE, beside `timeline_entry_text` and `CANT_PHRASE`, for the same reason they do — it is
    VOCABULARY, and every surface must say it the same way. (It began as a TUI-local helper; a web
    view re-wording "much worse than the critic expected" is precisely the drift the engine/renderer
    split exists to prevent.)"""
    mag = abs(td)
    if mag < 0.5:
        return "about what the critic expected"
    much = "much " if mag >= 3.0 else ""
    return f"{much}{'better' if td > 0 else 'worse'} than the critic expected"


def timeline_entry_text(e: dict) -> str:
    """One :func:`build_result_timeline` entry as a PLAIN-TEXT battle-log line — the unstyled
    sibling of the TUI's Rich renderer (``app._append_timeline_entry``), with the same wording:

        ``we thunderbolt did 31% (suicune 31% → faint)`` · ``opp rockslide did 73% (salamence 100%
        → 27%)`` · ``we switch tyranitar → skarmory`` · ``opp sends in metagross`` ·
        ``we hypnosis — missed``

    Pure. It exists so the JSON CLI and the web view render the timeline without either of them
    re-deriving the sentence (the engine is the single source of truth for what happened, including
    how it reads)."""
    side = str(e.get("side", ""))
    kind = e.get("kind")
    if kind == "switch":
        return f"{side} switch {e.get('actor', '')} → {e.get('switch_to', '')}".strip()
    if kind == "send_in":
        verb = "send in" if side == "we" else "sends in"
        return f"{side} {verb} {e.get('sent_in', '')}".strip()
    if kind == "faint":
        return f"{side} {e.get('actor', '')} fainted".strip()

    parts = [side, str(e.get("move") or "?")]
    text = " ".join(p for p in parts if p)
    if e.get("cant"):
        text += f" — couldn't move ({cant_phrase(e['cant'])})"
    elif e.get("damage"):
        text += (f" did {e['damage']}  ({e.get('target', '')} "
                 f"{e.get('hp_before', '')} → {e.get('hp_after', '')})")
    elif e.get("resulting"):        # hit a switch-IN; only the resulting HP is known
        text += f" → {e.get('target', '')} (now {e.get('hp_after', '')})"
    elif e.get("status"):
        text += f" → {e.get('target', '')} {e['status']}"
    elif e.get("no_effect"):        # nothing happened — say WHY, never leave it blank
        text += f" — {NO_EFFECT_TEXT.get(e['no_effect'], 'no effect')}"
    if e.get("boost"):
        text += f"  ·  {e['boost']}"
    if e.get("crit"):
        text += "  ⚡CRIT"
    return text


def build_belief(inv: dict) -> "BeliefView | None":
    """The hidden-opponent species belief at a decision — model-free, parsed from the summary
    invocation's ``belief`` block (the recorder writes it only when the belief was enabled). Each
    entry is ``{"slot": int, "top": [{"species": str, "prob": "NN.N%"}, ...]}``. Returns ``None`` when
    the block is absent (belief off) or empty (no hidden slot this turn) so off-runs show nothing."""
    raw = inv.get("belief")
    if not raw:
        return None
    slots = []
    for entry in raw:
        top = tuple((str(t.get("species", "?")), parse_pct(t.get("prob", "0%")))
                    for t in entry.get("top", []))
        if top:
            slots.append(BeliefSlotView(slot=int(entry.get("slot", -1)), top=top))
    return BeliefView(slots=tuple(slots)) if slots else None


# The name `α` uses for "none of these moves" — set by `opp_intent.render_alpha`, matched here.
SWITCH_OPTION = "SWITCH"


def build_opp_intent(inv: dict) -> "OppIntentView | None":
    """What the model expected the OPPONENT to do — model-free, from the summary invocation's
    `opp_intent` block (`RLPlayer._opp_intent` → `BattleRecorder.record`).

    `None` when the block is absent (the `α`/`β` heads were off, i.e. every trace before v67) or
    holds no named option. **The `alpha` order is the recorder's** — `render_alpha` already sorted it
    highest-first, and re-sorting a list that is already ordered is how the action-label scramble
    (see this package's CLAUDE.md gotcha) happened; the entries are passed through as given."""
    raw = inv.get("opp_intent")
    if not raw:
        return None
    alpha = []
    for entry in raw.get("alpha") or []:
        name = str(entry.get("name", "?"))
        alpha.append(OppIntentOption(name=name, p=float(entry.get("p", 0.0)),
                                     is_switch=name == SWITCH_OPTION))
    if not alpha:
        return None
    beta = tuple(
        OppIntentCandidate(slot=int(e.get("slot", -1)), p=float(e.get("p", 0.0)),
                           species=(str(e["species"]) if e.get("species") else None))
        for e in (raw.get("beta") or [])
    )
    switch = next((o.p for o in alpha if o.is_switch), None)
    return OppIntentView(alpha=tuple(alpha), beta=beta, top=alpha[0], switch_p=switch)


def opp_intent_text(view: "OppIntentView | None", top_n: int = 3) -> str:
    """The one-line rendering of `α` (+ `β` when a switch is expected) — the shared vocabulary, so
    the TUI, the JSON CLI and the web replay all say the same sentence about the same numbers.

    `expects fireblast 41% · SWITCH 22% · icebeam 12%`, and when SWITCH leads, the `β` follow-up:
    `expects SWITCH 52% · fireblast 20% → in: blissey 61%`. Empty string on `None`."""
    if view is None or not view.alpha:
        return ""
    parts = [f"{o.name} {o.p * 100:.0f}%" for o in view.alpha[:top_n]]
    text = "expects " + " · ".join(parts)
    if view.beta and view.top is not None and view.top.is_switch:
        best = view.beta[0]
        who = best.species or f"slot {best.slot}"
        text += f" → in: {who} {best.p * 100:.0f}%"
    return text


def awareness_text(aw: "dict | None") -> str:
    """The one-line rendering of a battle's 'did it KNOW?' verdict (`main/prober/awareness.py`),
    so the CLI and the web replay say the same sentence about the same fold.

    `never saw it coming — P(loss) never held above 50% to the end` ·
    `knew by turn 34 — 12 turns of warning` · and, when the stall signature fired, the clause that
    names it: `· stall signature: 41% tail mass at turn 28 while the mean still read positive`.
    Empty string on `None` (no dist head / fewer than 2 recorded distributions)."""
    if not aw:
        return ""
    knew, lead = aw.get("knew_by_turn"), aw.get("lead_time")
    if aw.get("blind_loss"):
        text = "never saw it coming — P(loss) never held above 50% to the end"
    elif knew is None:
        # Not a loss, and P(loss) never sustained: the ordinary shape of a win.
        text = "P(loss) never held above 50% to the end"
    elif aw.get("outcome") == "loss":
        turns = "turn" if lead == 1 else "turns"
        text = f"knew by turn {knew} — {lead} {turns} of warning"
    else:
        text = f"P(loss) held above 50% from turn {knew} to the end"
    div, div_turn = aw.get("mean_tail_divergence") or 0.0, aw.get("divergence_turn")
    # ≥0.5% is a RENDERING floor, not a semantic threshold: below it the clause prints
    # "0% tail mass", which reads as a finding when it is rounding noise. Which values count as
    # "the signature" stays the caller's bar to set (`stall_bar`), and it sits far above this.
    if div >= 0.005 and div_turn is not None:
        # The stall signature: catastrophic-band mass piling up while the distribution MEAN — the
        # only thing a scalar critic reads — still looked healthy. Reported as a FACT, with no
        # judgement applied.
        text += (f" · stall signature: {div * 100:.0f}% tail mass at turn {div_turn} "
                 "while the mean still read positive")
    return text


_SPECIES_MAPS = None
_MOVE_NUM_TO_ID = None
_MOVE_ID_TO_NUM = None


def _species_maps():
    """Cached ``({num->id}, {id->num})`` over the gen3 species vocab — the inverse of the belief
    labels' species_to_num, used to decode the species head (index == national-dex num) and to look
    up a true mon's num for the match cost."""
    global _SPECIES_MAPS
    if _SPECIES_MAPS is None:
        from agents.gen3_data import species as _sp
        raw = _sp.raw()
        num_to_id = {int(v["num"]): sid for sid, v in raw.items() if v.get("num")}
        id_to_num = {sid: int(v["num"]) for sid, v in raw.items() if v.get("num")}
        _SPECIES_MAPS = (num_to_id, id_to_num)
    return _SPECIES_MAPS


def _softmax(row) -> np.ndarray:
    r = np.asarray(row, dtype=np.float64)
    e = np.exp(r - r.max())
    return e / e.sum()


def _move_maps():
    """Cached ``{move_num -> move_id}`` over the gen3 move vocab — the inverse of the move-belief head's
    axis (index == gen3 move num). All 16 Hidden Powers share ONE num and the belief axis is
    type-collapsed there, so that num maps to the bare canonical ``hiddenpower`` (the op prices HP typing
    separately) — which then normalises to match a revealed ``hiddenpower(grass)``."""
    global _MOVE_NUM_TO_ID
    if _MOVE_NUM_TO_ID is None:
        raw = gen3_data.moves.raw()
        m = {int(v["num"]): mid for mid, v in raw.items() if v.get("num")}
        for mid, v in raw.items():               # collapse every Hidden Power num → bare "hiddenpower"
            if mid.startswith("hiddenpower") and v.get("num"):
                m[int(v["num"])] = "hiddenpower"
        _MOVE_NUM_TO_ID = m
    return _MOVE_NUM_TO_ID


def _move_id_to_num():
    """Cached ``{normalised_move_id -> move_num}`` — the forward map, so a REVEALED move name (from
    `describe_vector`, e.g. ``hiddenpower(fire)``) can be looked up on the move-belief axis to read its
    pinned belief. Every Hidden-Power variant + the bare ``hiddenpower`` resolve to the shared HP num."""
    global _MOVE_ID_TO_NUM
    if _MOVE_ID_TO_NUM is None:
        raw = gen3_data.moves.raw()
        d = {_norm_move(mid): int(v["num"]) for mid, v in raw.items() if v.get("num")}
        for mid, v in raw.items():
            if mid.startswith("hiddenpower") and v.get("num"):
                d["hiddenpower"] = int(v["num"])
        _MOVE_ID_TO_NUM = d
    return _MOVE_ID_TO_NUM


def _norm_move(m: str) -> str:
    """Normalise a move name for revealed-vs-believed comparison: lowercase, drop a trailing
    ``(type)`` (so a revealed ``hiddenpower(fire)`` matches the believed bare ``hiddenpower``)."""
    return (m or "").split("(")[0].strip().lower()


_MAX_MOVES = 4   # a gen3 mon carries at most 4 moves → only (4 − revealed) slots can still be unseen


def move_belief_view(raw, top_k: int = 4, prob_floor: float = 0.10) -> "MoveBeliefView | None":
    """Decode `ProbeModel.move_belief`'s raw output into a `MoveBeliefView`: per REVEALED opponent mon,
    each already-`revealed` move WITH its (pinned ≈100%) belief, plus the believed STILL-UNSEEN moves
    (multi-label sigmoid posterior, already-revealed filtered out, kept if `P ≥ prob_floor`). The unseen
    list is CAPPED at the number of OPEN move slots — `min(top_k, 4 − n_revealed)` — since a mon with k
    known moves can have at most `4 − k` more (the multi-label head itself doesn't enforce that 4-move
    constraint, so its raw top-K over-shows). Also carries the team-slot→species labels for the op's
    per-our-mon damage rows. Pure (no torch). `None` when the model has no move-belief head."""
    if not raw:
        return None
    probs = np.asarray(raw.get("opp_probs"), dtype=np.float64)        # [6, n_moves]
    num_to_id = _move_maps()
    id_to_num = _move_id_to_num()
    opp = []
    for i, slot in enumerate(raw.get("opp_slots", ()) or ()):
        if not slot.get("known") or i >= probs.shape[0]:
            continue
        revealed_names = tuple(slot.get("revealed_moves", ()) or ())
        revealed_norm = {_norm_move(m) for m in revealed_names}
        p = probs[i]
        # Revealed moves WITH their belief — look up each name's num on the belief axis (the model PINS
        # these ≈1.0 under prior fusion, so this confirms the belief tracks the known moveset).
        revealed = []
        for m in revealed_names:
            num = id_to_num.get(_norm_move(m))
            revealed.append((m, float(p[num]) if (num is not None and num < p.shape[0]) else 0.0))
        n_unseen = min(top_k, max(0, _MAX_MOVES - len(revealed_norm)))   # only the OPEN move slots
        believed = []
        for n in (np.argsort(p)[::-1] if n_unseen else ()):
            pv = float(p[n])
            if pv < prob_floor:
                break
            name = num_to_id.get(int(n))
            if not name or _norm_move(name) in revealed_norm:
                continue
            believed.append((name, pv))
            if len(believed) >= n_unseen:
                break
        opp.append(OppMoveBelief(slot=i, species=slot.get("species", ""),
                                 revealed=tuple(revealed), believed=tuple(believed)))
    our_labels = tuple((i, s.get("species", ""), bool(s.get("active")))
                       for i, s in enumerate(raw.get("our_slots", ()) or ()))
    return MoveBeliefView(opp=tuple(opp), our_labels=our_labels) if (opp or our_labels) else None


def belief_view_from_logits(species_logits, believed_mask, top_k: int = BELIEF_TOPK,
                            num_to_id=None) -> "BeliefView | None":
    """A `BeliefView` (anonymous per-slot top-k) decoded straight from the model's stashed species
    logits — the re-computed counterpart of `build_belief` (which parses the summary). `[6,n_species]`
    logits indexed by national-dex num; only believed slots (mask True) are decoded."""
    num_to_id = num_to_id or _species_maps()[0]
    logits = np.asarray(species_logits, dtype=np.float64)
    mask = np.asarray(believed_mask, dtype=bool)
    k = max(0, min(top_k, logits.shape[1]))
    slots = []
    for i in range(logits.shape[0]):
        if i >= mask.shape[0] or not bool(mask[i]):
            continue
        p = _softmax(logits[i])
        order = np.argsort(p)[::-1][:k]
        top = tuple((num_to_id.get(int(n), f"num_{int(n)}"), float(p[n])) for n in order)
        if top:
            slots.append(BeliefSlotView(slot=i, top=top))
    return BeliefView(slots=tuple(slots)) if slots else None


def revealed_opp_species(board: "BoardView | None") -> "tuple[str, ...]":
    """The opponent species REVEALED by this decision (active + revealed bench), from the board."""
    if board is None:
        return ()
    seen = [board.opp.active_species] + [m.species for m in board.opp.bench]
    return tuple(s for s in seen if s and s != "NONE")


def _norm_move(move: str) -> str:
    """Move id normalised for the revealed-vs-truth compare: lowercase, alnum-only, so the revealed
    display form (`hiddenpower(bug)`) matches the truth id (`hiddenpowerbug`) — both collapse to the
    bare `hiddenpower` (the opp's HP type stays unrevealed until it fires, so we compare the base)."""
    s = re.sub(r"[^a-z0-9]", "", str(move).lower())
    return "hiddenpower" if s.startswith("hiddenpower") else s


def build_opp_full_team(opp_team_details: "list | None",
                        board: "BoardView | None") -> "OppFullTeamView | None":
    """Merge the opponent's PRIVILEGED full team (`opp_team_details` — all 6 mons + moves + item from the
    `reconstruction.json`) with what's been REVEALED on field (`board.opp`), tagging each mon / item /
    move seen-or-not. `None` when there's no privileged team (websocket/older traces)."""
    if not opp_team_details:
        return None
    # Revealed lookup: norm-species → (hp, status, item, {norm-revealed-move}, active).
    rev: dict = {}
    if board is not None:
        o = board.opp
        if o.active_species and o.active_species != "NONE":
            rev[_norm_species(o.active_species)] = (o.active_hp, o.status, o.item,
                                                    {_norm_move(m) for m in o.moves}, True)
        for m in o.bench:
            rev.setdefault(_norm_species(m.species),
                           (m.hp, m.status, m.item, {_norm_move(x) for x in m.moves}, False))
    mons = []
    for d in opp_team_details:
        sp = str(d.get("species", ""))
        r = rev.get(_norm_species(sp))
        hp, status, ritem, rmoves, active = r if r else ("", "", "", set(), False)
        moves = tuple((str(mv), _norm_move(mv) in rmoves) for mv in (d.get("moves", ()) or ()))
        mons.append(OppFullMon(
            species=sp, revealed=r is not None, active=active, hp=hp, status=status,
            item=str(d.get("item", "") or ""), item_revealed=bool(r is not None and ritem), moves=moves))
    return OppFullTeamView(mons=tuple(mons))


def build_belief_truth(species_logits, believed_mask, revealed_species, true_team,
                       top_k: int = BELIEF_TOPK, maps=None) -> "BeliefTruthView | None":
    """Match the model's per-slot species belief to the opponent's TRUE hidden mons (privileged, from
    `reconstruction.json`) and tag every true mon revealed/hidden.

    The believed slots are anonymous, so they are **Hungarian-assigned** to the still-hidden true mons
    by minimum total ``-log P(true species | slot)`` — the SAME species-CE cost the training aux loss
    matches on (`instrumented_ppo._belief_aux_loss`), so the displayed correspondence is how the model
    itself aligns the slots. Returns `None` when there's no privileged team."""
    if true_team is None or len(true_team) == 0:
        return None
    from scipy.optimize import linear_sum_assignment
    num_to_id, id_to_num = maps or _species_maps()
    logits = np.asarray(species_logits, dtype=np.float64)
    mask = np.asarray(believed_mask, dtype=bool)
    revealed = {_norm_species(s) for s in (revealed_species or ())}
    true_ids = [_norm_species(s) for s in true_team]
    hidden_true = [s for s in true_ids if s not in revealed]
    believed_slots = [i for i in range(logits.shape[0]) if i < mask.shape[0] and bool(mask[i])]
    probs = {i: _softmax(logits[i]) for i in believed_slots}

    slot_for_hidden: "dict[int, int]" = {}     # hidden_true index -> believed slot index
    if believed_slots and hidden_true:
        cost = np.full((len(believed_slots), len(hidden_true)), 50.0)   # large finite default
        for a, i in enumerate(believed_slots):
            for b, sp in enumerate(hidden_true):
                num = id_to_num.get(sp)
                if num is not None and num < logits.shape[1]:
                    cost[a, b] = -np.log(max(float(probs[i][num]), 1e-12))
        rows, cols = linear_sum_assignment(cost)
        for a, b in zip(rows, cols):
            slot_for_hidden[int(b)] = believed_slots[int(a)]

    hidden_idx_of = {sp: b for b, sp in enumerate(hidden_true)}   # species unique under the species clause
    k = max(0, min(top_k, logits.shape[1]))
    mons, n_correct = [], 0
    for sp in true_ids:
        if sp in revealed:
            mons.append(OppMonTruth(species=sp, revealed=True))
            continue
        slot = slot_for_hidden.get(hidden_idx_of.get(sp))
        if slot is None:
            mons.append(OppMonTruth(species=sp, revealed=False))
            continue
        p = probs[slot]
        order = np.argsort(p)[::-1]
        top = tuple((num_to_id.get(int(n), f"num_{int(n)}"), float(p[n])) for n in order[:k])
        num = id_to_num.get(sp)
        right = num is not None and int(order[0]) == num
        rank = (int(np.where(order == num)[0][0]) + 1) if num is not None else -1
        n_correct += int(right)
        mons.append(OppMonTruth(species=sp, revealed=False, guess=top,
                                guessed_right=right, true_rank=rank))
    n_hidden = sum(1 for m in mons if not m.revealed)
    return BeliefTruthView(mons=tuple(mons), n_hidden=n_hidden, n_correct=n_correct)


# ---------------------------------------------------------------------------
# Spread belief vs truth (the DamageOperator's stat input — gen3_unified_spread_belief_v1)
# ---------------------------------------------------------------------------

_SPREAD_COLS = ("atk", "def", "spa", "spd", "spe")   # the order of last_spread_belief's 5 columns
_SPREAD_BASE_KEY = {"atk": "atk", "def": "def", "spa": "spa", "spd": "spd", "spe": "spe"}
_SPREAD_PRIOR_CACHE: "dict[str, tuple[float, ...]] | None" = None


def _derived_stat(base: int, iv: int, ev: int, mult: float) -> int:
    """The gen3 non-HP derived stat at level 100 with the mon's ACTUAL IV (poke-env/team_details give the
    real IV — e.g. a Hidden-Power mon isn't IV31 everywhere), EV, and nature multiplier. Exact integer
    math, mirroring `gen3_data.priors.gen3_stat` but parameterized on IV (that helper hardcodes IV31)."""
    pre = 2 * int(base) + int(iv) + int(ev) // 4 + 5
    if mult > 1.0:
        return pre * 11 // 10
    if mult < 1.0:
        return pre * 9 // 10
    return pre


def _spread_prior_means(species_id: str) -> "tuple[float, ...] | None":
    """Usage-weighted mean DERIVED stat per {atk,def,spa,spd,spe} for ``species_id`` — the Smogon spread
    PRIOR the SpreadBelief head corrects (the same quantity `damage_tables.build_opp_spread_prior`'s mean
    column holds). `None` when the species has no spread data. Cached per species (cheap, data-only)."""
    from agents import gen3_data
    sp = gen3_data.species.get(species_id)
    if sp is None:
        return None
    spreads = gen3_data.priors.spreads(species_id)
    if not spreads:
        return None
    ev_idx = {"atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}   # index into [hp,atk,def,spa,spd,spe]
    out = []
    for stat in _SPREAD_COLS:
        base = int(sp.base_stats.get(stat, 0))
        m1 = wsum = 0.0
        for nature, evs, w in spreads:
            nd = gen3_data.natures.get(str(nature).lower())
            mult = nd.multipliers.get(stat, 1.0) if nd is not None else 1.0
            m1 += float(w) * float(gen3_data.priors.gen3_stat(base, int(evs[ev_idx[stat]]), mult))
            wsum += float(w)
        out.append(m1 / wsum if wsum > 0 else float(gen3_data.priors.gen3_stat(base, 0, 1.0)))
    return tuple(out)


def _true_derived_spread(detail: dict) -> "tuple[tuple[float, ...], str, str] | None":
    """From one `team_details()` entry → (the 5 true DERIVED stats {atk,def,spa,spd,spe}, nature, ev_note).
    Uses the mon's REAL base/IV/EV/nature (privileged). `None` if the species is unknown to the dex."""
    from agents import gen3_data
    sp = gen3_data.species.get(detail.get("species", ""))
    if sp is None:
        return None
    evs = detail.get("evs") or {}
    ivs = detail.get("ivs") or {}
    nature = str(detail.get("nature", "") or "").lower()
    nd = gen3_data.natures.get(nature)
    vals = []
    for stat in _SPREAD_COLS:
        base = int(sp.base_stats.get(stat, 0))
        iv = int(ivs.get(stat, 31))
        ev = int(evs.get(stat, 0))
        mult = nd.multipliers.get(stat, 1.0) if nd is not None else 1.0
        vals.append(float(_derived_stat(base, iv, ev, mult)))
    ev_note = "/".join(f"{s}{int(evs[s])}" for s in ("hp",) + _SPREAD_COLS
                       if int(evs.get(s, 0)) > 0) or "0 EVs"
    return tuple(vals), nature, ev_note


def build_spread_belief(raw, opp_team_details, top_revealed_only: bool = True) -> "SpreadBeliefView | None":
    """Match the model's believed opp spread (`ProbeModel.spread_belief_view` raw) to the TRUE mons from
    `reconstruction.json`'s `team_details()` and compare the believed vs true DERIVED stats per REVEALED opp
    slot (the head predicts spreads for seen mons — species known, EVs unknown — so revealed slots are the
    meaningful ones; hidden mons have no spread prediction). Match is by SPECIES id (exact — a revealed mon's
    species is known + unique). `opp_team_details` is the list of `{species, evs, ivs, nature, …}` dicts (or
    `None`/`()` → believed-only, no truth columns). Returns `None` when the run trained `--spread-belief` off
    (raw is `None`) or no revealed slot has a believed spread."""
    if not raw:
        return None
    spread = np.asarray(raw.get("spread"), dtype=np.float64)          # [6, 5]
    bmask = raw.get("believed_mask")                                  # [6] bool (True = HIDDEN) or None
    opp_species = list(raw.get("opp_species") or [])
    # Index the privileged truth by normalized species id (revealed species are unique under the clause).
    truth_by_species: "dict[str, dict]" = {}
    for d in (opp_team_details or ()):
        sid = _norm_species(d.get("species", ""))
        if sid:
            truth_by_species[sid] = d

    slots, abs_errs = [], []
    for i in range(min(6, spread.shape[0])):
        sid = (opp_species[i] if i < len(opp_species) else "").strip()
        hidden = bool(bmask[i]) if (bmask is not None and i < len(bmask)) else (not sid)
        if top_revealed_only and (hidden or not sid):
            continue                                                  # hidden slot → no spread prediction
        believed = spread[i]                                          # [5]
        prior = _spread_prior_means(sid)
        truth = truth_by_species.get(_norm_species(sid))
        td = _true_derived_spread(truth) if truth is not None else None
        true_vals, nature, ev_note = (td if td is not None else (None, "", ""))
        rows = []
        for j, stat in enumerate(_SPREAD_COLS):
            tv = float(true_vals[j]) if true_vals is not None else None
            pv = float(prior[j]) if prior is not None else None
            rows.append(SpreadStatRow(stat=stat, believed=float(believed[j]), true=tv, prior=pv))
            if tv is not None:
                abs_errs.append(abs(float(believed[j]) - tv))
        slots.append(SpreadSlotBelief(slot=i, species=sid, rows=tuple(rows),
                                      nature=nature, ev_note=ev_note, matched=td is not None))
    if not slots:
        return None
    mae = (sum(abs_errs) / len(abs_errs)) if abs_errs else None
    return SpreadBeliefView(slots=tuple(slots), n_slots=len(slots), mean_abs_err=mae)


# ---------------------------------------------------------------------------
# Switch-in outgoing damage (forced-switch panel — what each candidate would DO)
# ---------------------------------------------------------------------------
# On a forced switch the DamageOperator's OUTGOING block is all-zero (it prices the fainted
# active only), so the model picks a switch-in from INCOMING threat alone, with no estimate of
# what each candidate would then DO to the opp active. This CPU-side panel fills that view
# (prober-only, no model change): each ALIVE bench candidate → its best damaging move vs the opp
# active → [low–high %, →KO, ×mult, P(outspeed)], from the privileged true spreads (📋).

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


def _derived_hp(base: int, iv: int, ev: int) -> int:
    """Gen3 HP derived stat at level 100 (distinct from _derived_stat: +level+10, no nature)."""
    return 2 * int(base) + int(iv) + int(ev) // 4 + 110


def _hp_frac_from_str(hp: str) -> "float | None":
    """Board HP string ('31%' / '100%' / 'faint') → fraction in [0,1]; None if fainted/unknown."""
    s = (hp or "").strip().lower()
    if not s or s == "faint" or s.startswith("0%"):
        return None
    try:
        return max(0.0, min(1.0, float(s.rstrip("%")) / 100.0))
    except ValueError:
        return None


def _as_ptype(name: str):
    """A type string ('WATER') → poke-env PokemonType (species.types are UPPERCASE enum names)."""
    try:
        from poke_env.battle.pokemon_type import PokemonType
        return PokemonType[str(name).upper()]
    except (KeyError, AttributeError, ImportError):
        return None


def build_switch_in_outgoing(board, our_team_details, opp_team_details) -> "SwitchInOutgoingView | None":
    """Each ALIVE bench candidate's best damaging move's expected damage to the opp ACTIVE — the
    forced-switch panel. Model-free / privileged (true spreads). None when no reconstruction
    (no team_details), no opp active, or no candidate has a BP move."""
    if not our_team_details or not opp_team_details or board is None:
        return None
    from agents import gen3_data
    from agents.observation.incoming_damage import (
        gen3_damage_max, p_ko, p_outspeed, type_is_physical)
    from agents.gen3_mechanics import effective_multiplier_by_types

    opp_species = _norm_species(getattr(board.opp, "active_species", "") or "")
    if not opp_species:
        return None
    opp_detail = next((d for d in opp_team_details
                       if _norm_species(d.get("species", "")) == opp_species), None)
    opp_sp = gen3_data.species.get(opp_species)
    od = _true_derived_spread(opp_detail) if opp_detail is not None else None
    if opp_sp is None or od is None:
        return None
    opp_stats, _, _ = od                                   # (atk,def,spa,spd,spe)
    opp_def, opp_spd, opp_spe = opp_stats[1], opp_stats[3], opp_stats[4]
    ivs, evs = (opp_detail.get("ivs") or {}), (opp_detail.get("evs") or {})
    opp_max_hp = _derived_hp(opp_sp.base_stats.get("hp", 0), int(ivs.get("hp", 31)), int(evs.get("hp", 0)))
    opp_hp_frac = _hp_frac_from_str(getattr(board.opp, "active_hp", ""))
    if opp_max_hp <= 0 or opp_hp_frac is None:
        return None
    opp_remaining = max(1, int(round(opp_max_hp * opp_hp_frac)))
    opp_t = [t for t in (_as_ptype(x) for x in opp_sp.types) if t is not None]
    if not opp_t:
        return None
    opp_t1, opp_t2 = opp_t[0], (opp_t[1] if len(opp_t) > 1 else None)

    by_species = {_norm_species(d.get("species", "")): d for d in our_team_details}
    rows = []
    for cand in getattr(board.ours, "bench", ()):          # the switch-in candidates the board lists
        if _hp_frac_from_str(getattr(cand, "hp", "")) is None:
            continue                                        # fainted / unavailable
        sid = _norm_species(getattr(cand, "species", ""))
        d, sp = by_species.get(sid), gen3_data.species.get(sid)
        cd = _true_derived_spread(d) if d is not None else None
        if sp is None or cd is None:
            continue
        c_stats, _, _ = cd
        our_atk, our_spa, our_spe = c_stats[0], c_stats[2], c_stats[4]
        our_types = {t for t in (_as_ptype(x) for x in sp.types) if t is not None}
        best = None
        for mid in (d.get("moves") or ()):
            mv = gen3_data.moves.get(mid)
            if mv is None or int(getattr(mv, "base_power", 0)) <= 0:
                continue                                    # status / fixed-damage (v1: BP moves only)
            if mid in ("explosion", "selfdestruct"):
                continue                                    # KOs but self-KOs — not a switch-in's sustainable offense
            phys = type_is_physical(mv.type)
            eff = effective_multiplier_by_types(mv.type, opp_t1, opp_t2)
            dmax = gen3_damage_max(int(mv.base_power), int(our_atk if phys else our_spa),
                                   int(opp_def if phys else opp_spd),
                                   stab=(mv.type in our_types), type_eff=eff)
            high = 100.0 * dmax / opp_max_hp
            pko = p_ko(dmax, opp_remaining)
            if best is None or (pko, high) > (best[3], best[2]):
                best = (mid, eff, high, pko)
        if best is None:
            continue
        mid, eff, high, pko = best
        rows.append(SwitchInOutgoingRow(
            species=getattr(cand, "species", ""), hp=getattr(cand, "hp", ""), move=mid,
            low=high * 0.85, high=high, pko=pko, type_mult=eff,
            outspeed=p_outspeed(int(our_spe), [(int(opp_spe), 1.0)])))
    if not rows:
        return None
    rows.sort(key=lambda r: (r.pko, r.high), reverse=True)
    return SwitchInOutgoingView(opp_species=getattr(board.opp, "active_species", ""),
                                opp_hp=getattr(board.opp, "active_hp", ""), rows=tuple(rows))


# ---------------------------------------------------------------------------
# Move-belief entropy helper
# ---------------------------------------------------------------------------

def _entropy_bits(logits: np.ndarray) -> float:
    """Bernoulli entropy (nats) summed over a multi-label move-belief logit row — the uncertainty
    of the believed moveset. Should DECAY across a battle as reveals accumulate."""
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).sum())


# ---------------------------------------------------------------------------
# Belief refinement trajectory (axis B — across-battle turns, model-free from the summary)
# ---------------------------------------------------------------------------

def build_belief_trajectory(summary: dict, opp_team: "tuple[str, ...] | None",
                            npz=None) -> "BeliefTrajectoryView | None":
    """A battle's belief sharpening across its decisions (axis B): for each decision with a summary `belief`
    block, the per-hidden-slot top-1 species confidence + how many top-1 guesses correctly named a STILL-
    HIDDEN true mon (needs the privileged `opp_team`). Model-free over the summary `belief` blocks
    (`build_belief`), so it works on ANY belief-on trace without re-running the model.

    Correctness scoring mirrors `build_belief_truth`'s precision: per decision the true HIDDEN set is
    `opp_team` minus the species REVEALED by then (decoded model-free from the inv board), and each believed
    slot's top-1 is matched against that set with **one-time consumption** — so guessing an already-revealed
    species, or two slots guessing the same single hidden mon, can't double-count (the set-membership bug).
    Scored only when `opp_team` is given (else `n_correct`=0, confidence-only — the websocket/no-truth case).

    When `npz` carries the captured `move_logits` / `spread_belief` arrays (future runs), each point also gets
    the opp-active move-belief Bernoulli `move_entropy` (should DECAY as reveals accumulate) + the believed
    opp-active `believed_atk`/`believed_spe` — the move/spread analog of the species trajectory, decoded
    WITHOUT re-running the model. Absent/NaN on older traces ⇒ those stay `None`. `None` when no decision
    carries a belief block."""
    invs = summary.get("invocations", []) or []
    points = []
    team = _our_items(summary)                                       # for the model-free opp-revealed board decode
    move_logits = _npz_array(npz, "move_logits")                     # [T, n_moves] or None
    spread_arr = _npz_array(npz, "spread_belief")                    # [T, 5] (opp-active row) or None
    for idx, inv in enumerate(invs):
        bv = build_belief(inv)                                       # BeliefView | None (summary fallback)
        if bv is None or not bv.slots:
            continue
        # The still-HIDDEN true multiset = opp_team minus the species revealed by this decision (the believed
        # slots ARE the hidden mons; species are unique under the clause). Match top-1 with consumption.
        hidden_remaining = None
        if opp_team:
            revealed = set()
            try:
                revealed = {_norm_species(s) for s in revealed_opp_species(build_board(inv, team))}
            except Exception:  # noqa: BLE001 — model-free board decode is best-effort; degrade to no-reveal
                revealed = set()
            from collections import Counter
            hidden_remaining = Counter(_norm_species(s) for s in opp_team if _norm_species(s) not in revealed)
        confs, n_correct = [], 0
        for s in bv.slots:
            if not s.top:
                continue
            top_sp, top_p = s.top[0]
            confs.append(float(top_p))
            key = _norm_species(top_sp)
            if hidden_remaining is not None and hidden_remaining.get(key, 0) > 0:
                n_correct += 1
                hidden_remaining[key] -= 1
        if not confs:
            continue
        m_ent = b_atk = b_spe = None
        if move_logits is not None and 0 <= idx < len(move_logits) and np.isfinite(move_logits[idx]).any():
            m_ent = _entropy_bits(move_logits[idx])
        if spread_arr is not None and 0 <= idx < len(spread_arr) and np.isfinite(spread_arr[idx]).all():
            row = np.asarray(spread_arr[idx], dtype=np.float64)      # opp-active believed [atk,def,spa,spd,spe]
            if row.shape[0] >= 5:
                b_atk, b_spe = float(row[0]), float(row[4])
        points.append(BeliefTrajectoryPoint(
            inv_index=idx, turn=int(inv.get("turn", 0)), n_hidden=len(confs),
            n_correct=n_correct, mean_top1_conf=float(np.mean(confs)),
            move_entropy=m_ent, believed_atk=b_atk, believed_spe=b_spe))
    return BeliefTrajectoryView(points=tuple(points)) if points else None


def build_meta(summary: dict, summary_path: str = "", npz_path: "str | None" = None) -> TraceMeta:
    m = summary.get("meta", {})
    return TraceMeta(
        step=int(m.get("step", 0)),
        battle_id=str(m.get("battle_id", "")),
        result=str(m.get("result", "")),
        turns=int(m.get("turns", 0)),
        n_invocations=int(m.get("invocations", len(summary.get("invocations", [])))),
        summary_path=summary_path,
        npz_path=npz_path,
    )


# ---------------------------------------------------------------------------
# Analysis units (each pure; fed the loaded model + arrays)
# ---------------------------------------------------------------------------

def _faithfulness(probs: np.ndarray, labels: list, acts: dict, chosen: str,
                  mask: np.ndarray) -> "tuple[ActionRow, ...]":
    rows = []
    for j, k in enumerate(labels):
        rows.append(ActionRow(
            label=k,
            valid=bool(mask[j]),
            recorded=parse_pct(acts[k]["prob"]),
            rerun=float(probs[j]),
            is_chosen=(k == chosen),
        ))
    return tuple(rows)


# Moves that read base_power 0 in the dex but whose type multiplier is still real (fixed/variable
# power — type IMMUNITY applies even when the amount is constant). Reuses the canonical sets the
# incoming-damage belief prices from, so this stays in sync with the rest of the codebase.
_BP0_DAMAGING = set(_inc.FIXED_DAMAGE) | {"return", "frustration"}


def _multiplier_meaningful(move_id: str) -> bool:
    """Does the active-move type multiplier carry signal for ``move_id``? True for any damaging move
    — positive-BP, fixed-/variable-power, or Hidden Power (revealed as the bare id but a typed
    attack). False for a genuine status/self/field move (Spikes/Recover/Protect), where the obs
    still computes a phantom multiplier the UI should render as n/a. See MatchupView.applicable."""
    if not move_id:
        return False
    if gen3_data.moves.is_damaging(move_id) or move_id.startswith("hiddenpower"):
        return True
    return move_id in _BP0_DAMAGING


def _display_hp(move_id: str, our_species: str, hp_map: "dict | None") -> str:
    """Render OUR move label's Hidden Power with its TYPE (``hiddenpower(grass)``), since we always
    know our own HP type. Handles BOTH trace vintages: the recorder's typed id ``hiddenpowergrass``
    (recovered straight from the id) AND a bare own ``hiddenpower`` from an OLDER trace (recovered
    from the reconstruction ``hp_map``, our side only — matches how the Board/move-belief retype).
    A non-HP label, or a bare HP with no reconstruction record, is returned unchanged (no leak: an
    opponent's un-revealed HP never reaches this — these are OUR action labels)."""
    if not move_id.startswith("hiddenpower"):
        return move_id
    if move_id != "hiddenpower":                       # typed id (new traces) → from the id itself
        return f"hiddenpower({move_id[len('hiddenpower'):]})"
    return (hp_map or {}).get(_norm_species(our_species)) or move_id   # bare own HP → reconstruction


def _matchups(obs: np.ndarray, labels: list, off, our_species: str = "",
              hp_map: "dict | None" = None) -> MatchupView:
    # gen3_cpu_damage_deleted_v1: mm_off==0 ⇒ the block is absent from this obs layout.
    mults = (tuple(float(x) * 4.0 for x in obs[off.mm_off:off.mm_off + 4])
             if off.mm_off > 0 else (0.0, 0.0, 0.0, 0.0))
    raw = tuple(labels[MOVE_START:MOVE_END])  # the 4 move actions, request order
    # Show OUR Hidden Power typed (hiddenpower(grass)); the multiplier IS already typed in the obs.
    move_labels = tuple(_display_hp(lbl, our_species, hp_map) for lbl in raw)
    # Flag which slots' multipliers are meaningful vs a phantom artefact on a status/self move (from
    # the RAW move id, so is_damaging is exact) — see MatchupView.applicable. _multiplier_meaningful
    # tolerates unknown/empty ids → False.
    applicable = tuple(_multiplier_meaningful(lbl) for lbl in raw)
    return MatchupView(multipliers=mults, move_labels=move_labels, applicable=applicable)


def _switch_prob_sum(p: np.ndarray, labels: list) -> float:
    return float(sum(p[j] for j, l in enumerate(labels) if l.startswith("switch")))


def _intervention_sweep(model, obs: np.ndarray, mask: np.ndarray, labels: list,
                        chosen: str, off) -> InterventionSweep:
    cj = labels.index(chosen)
    if not (MOVE_START <= cj < MOVE_END):
        return InterventionSweep(chosen, -1, 0.0, ())
    slot = cj - MOVE_START
    probs, _ = model.action_dist(obs, mask)
    baseline_switches = _switch_prob_sum(probs, labels)
    rows = []
    for mult in _SWEEP_MULTIPLIERS:
        o = obs.copy()
        if off.mm_off > 0:                      # no-op when the block was deleted from the obs
            o[off.mm_off + slot] = mult / 4.0
        p, _ = model.action_dist(o, mask)
        rows.append(InterventionRow(
            multiplier=mult,
            p_chosen=float(p[cj]),
            p_switches=_switch_prob_sum(p, labels),
        ))
    return InterventionSweep(chosen, slot, baseline_switches, tuple(rows))


def _saliency_from_grad(g: np.ndarray, off) -> Saliency:
    """Aggregate a per-dim gradient into the named obs blocks. Shared by the policy-logit
    saliency and the critic value saliency so both report the SAME regions (incl. the new
    `incoming_damage` block) — the only difference is which head's gradient is fed in."""
    def block(name: str, lo: int, hi: int) -> SaliencyBlock:
        seg = g[lo:hi]
        return SaliencyBlock(name=name, mean_abs=float(seg.mean()), total_abs=float(seg.sum()))

    blocks = [

        *((block("our_matchups(144)", off.om_off, off.om_off + 144),
           block("their_matchups(144)", off.tm_off, off.tm_off + 144))
          if off.om_off > 0 else ()),  # gen3_entity_rehome_v1: absent from re-homed layouts
    ]
    if off.incoming_dim > 0:  # incoming-damage / OHKO belief block (incoming_damage_v1)
        blocks.append(block(f"incoming_damage({off.incoming_dim})",
                            off.incoming_off, off.incoming_off + off.incoming_dim))
    blocks += [
        block("our active pokemon block(99)", 0, off.active_block_dim),
        block("turn-history block", off.turn_history_offset,
              off.turn_history_offset + off.turn_history_dim),
    ]
    return Saliency(overall_mean_abs=float(g.mean()), blocks=tuple(blocks))


def history_slot_saliency(g: np.ndarray, off) -> "list[float]":
    """Per-turn-slot mean|grad| within the turn-history block — splits the one 'turn-history block'
    saliency into its N_HISTORY_TURNS TurnDelta slots, so we can see whether the OLDER turns carry
    little signal (a candidate to shorten N_HISTORY_TURNS and reclaim obs/attention compute). ``g``
    is an already-abs per-dim gradient (policy-logit or value). Slot i is the i-th TurnDelta in obs
    order; the transformer's positional embedding learns recency, so the caller labels recent/old."""
    td = getattr(off, "turn_delta_dim", 0) or 0
    if off.turn_history_dim <= 0 or td <= 0:
        return []
    base = off.turn_history_offset
    n = off.turn_history_dim // td
    return [float(g[base + i * td: base + (i + 1) * td].mean()) for i in range(n)]


def _saliency(model, obs: np.ndarray, mask: np.ndarray, chosen_idx: int, off) -> Saliency:
    """Policy saliency: |d logit(chosen) / d obs| aggregated per block (what the ACTOR reads)."""
    return _saliency_from_grad(model.logit_grad(obs, mask, chosen_idx), off)


def _value_saliency(model, obs: np.ndarray, mask: np.ndarray, off) -> "Saliency | None":
    """Critic saliency: |d V(s) / d obs| aggregated per block (what the VALUE head reads) — the
    relevant lens for OHKO tail-blindness. None when the model can't expose a value gradient."""
    vg = getattr(model, "value_grad", None)
    if vg is None:
        return None
    try:
        return _saliency_from_grad(vg(obs, mask), off)
    except Exception:  # noqa: BLE001 — value-grad is best-effort (stub models / odd heads)
        return None


def _threats(obs: np.ndarray, off) -> "ThreatView | None":
    """Decode the opponent's incoming type-effectiveness from `their_matchups`.

    Returns None if the obs is too short to hold the block (tiny synthetic test obs) —
    so the engine never crashes on a malformed/old trace."""
    if off.tm_off <= 0:
        return None  # gen3_entity_rehome_v1: the block no longer exists in the live layout
    seg = obs[off.tm_off:off.tm_off + _MATCHUP_DIM]
    if seg.shape[0] != _MATCHUP_DIM:
        return None
    m = seg.reshape(_TEAM_SIZE, 4, _TEAM_SIZE) * 4.0   # [opp_mon, move_slot, our_mon], ×4-denormalised
    return ThreatView(
        present=bool((m > 0).any()),
        revealed_frac=float((m > 0).mean()),
        max_incoming=float(m.max()),
        per_our_slot_max=tuple(float(m[:, :, j].max()) for j in range(_TEAM_SIZE)),
    )


def _active_slot(obs: np.ndarray, off) -> "int | None":
    """Our on-field mon's team-slot index, read from the per-mon active flag (the last dim of each
    `pokemon_full_dim`-wide our-team block). The incoming-damage block is slot-aligned to the same
    team list, so this index selects the active mon's belief. None if no flag is set / obs too short."""
    stride = off.pokemon_full_dim
    best = None
    best_v = 0.5
    for i in range(_TEAM_SIZE):
        idx = i * stride + stride - 1
        if idx < obs.shape[0] and float(obs[idx]) > best_v:
            best_v, best = float(obs[idx]), i
    return best


def decode_incoming_belief(obs: np.ndarray, off) -> "IncomingBeliefView | None":
    """Decode the incoming-damage / OHKO belief block from a raw obs vector.

    Pure + model-free (reads the saved obs the trace recorded), so it is exact for the model that
    produced the trace and is reusable by both `analyze_invocation` and the model-free `scan`.

    ``off`` (the obs offsets) is resolved from the CURRENT encoder, so it is only valid for traces of
    the current arch. A trace whose obs length differs (e.g. an archived old-arch run) would be
    **mis-sliced** by these offsets — so we REFUSE it (return None) on a length mismatch rather than
    silently decoding garbage. ``pm>=8`` is the crit-split layout (the live arch); the ``pm==5`` branch
    serves explicit/synthetic 5-field ObsOffsets (tests) — it is NOT a path for archived old-arch traces
    (``resolve()`` always yields the current 8-field offsets, and the length guard rejects the old obs).
    Returns None when the block is absent (dim 0) or the obs length doesn't match the current arch."""
    if off.incoming_dim <= 0:
        return None
    if getattr(off, "total_dim", 0) and obs.shape[0] != off.total_dim:
        return None   # wrong-length trace (old/foreign arch) — refuse, don't mis-slice
    seg = obs[off.incoming_off:off.incoming_off + off.incoming_dim]
    if seg.shape[0] != off.incoming_dim:
        return None
    pm, rec_dim = off.incoming_per_mon, off.incoming_recovery
    n_slots = (off.incoming_dim - rec_dim) // pm
    per_slot_pko, per_slot_exp, outspeeds, per_slot_nc, reveals = [], [], [], [], []
    for i in range(n_slots):
        b = i * pm
        if pm >= 8:   # crit-split: phys/spec exp, phys/spec pko_nocrit, phys/spec CRIT_DELTA, outspeed, revealed
            phys_exp, spec_exp, phys_nc, spec_nc, phys_d, spec_d, outspeed, revealed = (
                float(x) for x in seg[b:b + 8])
            # reconstruct the crit-inclusive P(KO) (= nocrit + delta) so active_pko keeps its meaning
            per_slot_pko.append(max(phys_nc + phys_d, spec_nc + spec_d))
            per_slot_nc.append(max(phys_nc, spec_nc))
            reveals.append(revealed)
        else:         # explicit/synthetic 5-field: phys_exp, spec_exp, phys_pko, spec_pko, outspeed
            phys_exp, spec_exp, phys_pko, spec_pko, outspeed = (float(x) for x in seg[b:b + 5])
            per_slot_pko.append(max(phys_pko, spec_pko))
        per_slot_exp.append(max(phys_exp, spec_exp))
        outspeeds.append(outspeed)
    rec = seg[n_slots * pm:]
    a = _active_slot(obs, off)
    in_range = a is not None and a < n_slots
    split = pm >= 8
    return IncomingBeliefView(
        present=bool(any(v > 0 for v in per_slot_pko) or any(v > 0 for v in per_slot_exp)),
        max_pko=max(per_slot_pko) if per_slot_pko else 0.0,
        active_pko=per_slot_pko[a] if in_range else None,
        active_exp=per_slot_exp[a] if in_range else None,
        active_outspeed=outspeeds[a] if in_range else None,
        per_slot_pko=tuple(per_slot_pko),
        recovery_rate=float(rec[0]) if rec.shape[0] > 0 else 0.0,
        cures_status=float(rec[1]) if rec.shape[0] > 1 else 0.0,
        recovery_known=float(rec[2]) if rec.shape[0] > 2 else 0.0,
        active_pko_nocrit=(per_slot_nc[a] if (split and in_range) else None),
        threat_revealed=(reveals[a] if (split and in_range) else None),
    )


# NOTE: a former `_reorder_move_labels` helper re-sorted the recorded move labels to the per-mon block's
# MOVESET order — it was REMOVED because the recorded `actions` dict is already in ACTION-INDEX / request-slot
# order (see `analyze_invocation`); reordering it scrambled correct labels. The recorder↔prober alignment
# invariant is pinned by `engine_test.test_recorded_actions_are_action_index_aligned`.


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def analyze_invocation(model, summary: dict, npz, inv_index: int,
                       summary_path: str = "", npz_path: "str | None" = None,
                       opp_team: "tuple[str, ...] | None" = None,
                       our_hp_types: "dict | None" = None,
                       opp_team_details: "list | None" = None,
                       our_team_details: "list | None" = None) -> InvocationAnalysis:
    """Analyze a single decision point. Pure given ``model`` (the torch boundary).

    ``opp_team`` is the opponent's PRIVILEGED full team (species ids from the trace's
    `reconstruction.json` sibling, loaded by the caller — kept out of this pure engine); when given
    AND the model exposes the belief, the result carries the slot-MATCHED `belief_truth`.
    ``opp_team_details`` is the richer per-mon `team_details()` list ({species, evs, ivs, nature, …}) from
    the same `reconstruction.json`; when given AND the model exposes the spread belief, the result carries
    the believed-vs-true `spread_belief`."""
    meta = build_meta(summary, summary_path, npz_path)
    inv = summary["invocations"][inv_index]
    chosen = inv["chosen"]
    common = dict(
        meta=meta, inv_index=inv_index, turn=int(inv["turn"]), phase=inv.get("phase", ""),
        our_species=inv["our"]["species"], opp_species=inv["opp"]["species"], chosen=chosen,
    )

    outcome = inv.get("outcome", {}) or {}
    # Team (item + moveset): our side (items only, exact, end-of-battle) from the summary; superseded
    # /extended per-turn by the obs decode below once a state is captured — which also surfaces the
    # OPP's revealed items + both sides' revealed movesets.
    team = _our_items(summary)
    board = build_board(inv, team, our_hp_types)   # model-free; available even without captured state
    # The NEXT decision's board is the RESOLVED "after" state — read it (model-free) so the UI can
    # show before→after HP. None on the last decision (no following invocation).
    invs = summary["invocations"]
    next_board = (build_board(invs[inv_index + 1], team, our_hp_types)
                  if inv_index + 1 < len(invs) else None)
    belief = build_belief(inv)       # model-free summary fallback (re-computed below when a model + state exist)
    belief_truth = None
    # α/β — model-free from the trace, and it STAYS model-free (unlike `belief`, which prefers a
    # re-computed read): the intent heads are supervised against what the opponent then did, so the
    # honest question is what THIS decision's model expected, not what a later checkpoint would.
    opp_intent = build_opp_intent(inv)
    outcome = {**outcome, "timeline": _timeline_for(inv, next_board, outcome)}   # model-free RESULT lines
    opp_full_team = build_opp_full_team(opp_team_details, board)   # model-free; available without state
    # Forced-switch panel: what each ALIVE candidate would DO to the opp active (the op's outgoing is
    # all-zero here). Model-free / privileged — available even without captured state.
    switch_in_outgoing = None
    if common["phase"] == "forced_switch":
        try:
            switch_in_outgoing = build_switch_in_outgoing(board, our_team_details, opp_team_details)
        except Exception:  # noqa: BLE001 — never break the analysis
            switch_in_outgoing = None
    if not _has_state(npz, inv_index):
        return InvocationAnalysis(
            **common, has_state=False, actions=(), matchups=None, sweep=None,
            saliency=None, value_saliency=None, threats=None, incoming=None,
            warnings=(f"invocation {inv_index} has no captured state",), opp_full_team=opp_full_team,
            outcome=outcome, flags=summary_flags(inv), cure_options=self_cure_options(inv),
            board=board, next_board=next_board, belief=belief, opp_intent=opp_intent,
            switch_in_outgoing=switch_in_outgoing, opp_switched_to=opp_voluntary_switch(inv),
        )

    obs = npz["obs"][inv_index].astype(np.float32)
    # Obs-version guard: if the trace's obs is a different length than the CURRENT encoder, every
    # obs-OFFSET decode past the divergence point (incoming/threat/matchups/turn-history crit etc.)
    # is misaligned. Flag it so the UI warns instead of showing garbage. (Team blocks + global are
    # at the front and still decode; the model forward still runs on the trace's native obs.)
    enc_dim = getattr(model.offsets, "total_dim", 0)
    obs_mismatch = (int(obs.shape[0]), int(enc_dim)) if enc_dim and obs.shape[0] != enc_dim else None
    decode_team = getattr(model, "describe_team", None)
    if decode_team is not None:
        obs_team = decode_team(obs)
        if obs_team:
            board = build_board(inv, _merge_team(team, obs_team), our_hp_types)   # both sides, per-turn revealed
    # What actually happened on THIS turn (crit + couldn't-move reason): the realized events are
    # recorded in the NEXT decision's most-recent TurnDelta (turn T's events land in decision T+1's
    # obs). Adds our_crit/opp_crit + our_cant/opp_cant to the outcome so the RESULT/happened line can
    # tag '⚡crit' and 'couldn't move (asleep)'.
    decode_turn = getattr(model, "describe_turn_outcome", None)
    if decode_turn is not None:
        obs_all = npz["obs"]
        nxt = inv_index + 1
        if nxt < len(obs_all) and _has_state(npz, nxt):
            to = decode_turn(obs_all[nxt].astype(np.float32))
            if to:
                outcome = {**outcome, **to}
    # Rebuild the RESULT timeline now that crit / move_order / cant are merged in (the no-state path
    # above already built a model-free one without them).
    outcome = {**outcome, "timeline": _timeline_for(inv, next_board, outcome)}
    acts = inv["actions"]
    # The recorded `actions` dict is ALREADY in ACTION-INDEX order: `BattleRecorder._all_action_labels`
    # iterates action index 0..10 and keys move slot m (action 6+m) on `legal.move_ids[m]` — the SAME
    # request-slot order the action mask, the DamageOperator's per-move blocks, and the policy logits
    # (action 6+k) use. So `labels[i]` ↔ action index i ↔ `probs[i]` directly; every action-6+k-indexed
    # consumer below (matchups ×mult, op outgoing, the re-run argmax) is correctly paired with NO realign.
    # (A prior `_reorder_move_labels` step re-sorted these to the per-mon block's MOVESET order via
    # `our_active_move_slots`, which differs from request order after a server reorder — that SCRAMBLED the
    # already-correct labels and produced spurious `disagree` flags + transposed matchup/op labels. Removed.)
    labels = list(acts.keys())
    mask = np.array([1 if acts[k]["valid"] else 0 for k in labels], dtype=np.int8)

    # Hidden-opp belief: prefer the loaded model's RE-COMPUTED belief (works for ANY belief-on
    # checkpoint, incl. runs whose recorder predates the summary `belief` block) over the summary
    # fallback; with the privileged opponent team, also build the slot-MATCHED truth view.
    belief_fn = getattr(model, "belief", None)
    if belief_fn is not None:
        raw = belief_fn(obs, mask)
        if raw is not None:
            sp_logits, bmask = raw
            mb = belief_view_from_logits(sp_logits, bmask)
            if mb is not None:
                belief = mb
            if opp_team:
                belief_truth = build_belief_truth(sp_logits, bmask, revealed_opp_species(board), opp_team)

    probs, _ = model.action_dist(obs, mask)
    actions = _faithfulness(probs, labels, acts, chosen, mask)
    matchups = _matchups(obs, labels, model.offsets, inv["our"].get("species", ""), our_hp_types)
    sweep = _intervention_sweep(model, obs, mask, labels, chosen, model.offsets)
    saliency = _saliency(model, obs, mask, labels.index(chosen), model.offsets)
    value_saliency = _value_saliency(model, obs, mask, model.offsets)
    threats = _threats(obs, model.offsets)
    incoming = decode_incoming_belief(obs, model.offsets)

    # Value: recorded V(s), the loaded model's re-run V(s), V at the next captured
    # decision, and ΔV. (A big ΔV is where the critic's expectation shifted.)
    recorded_v = _npz_value(npz, inv_index)
    value = None
    if recorded_v is not None:
        n = len(summary["invocations"])
        nxt = inv_index + 1
        next_v = _npz_value(npz, nxt) if (nxt < n and _has_state(npz, nxt)) else None
        rerun_v = model.value(obs, mask)
        # PopArt-normalized companions (the critic's learning scale), when the model exposes stats.
        mu = sigma = norm_rec = norm_rerun = None
        pa = getattr(model, "popart_stats", lambda: None)()
        if pa is not None and pa[1]:
            mu, sigma = pa
            norm_rec = (recorded_v - mu) / sigma
            norm_rerun = (rerun_v - mu) / sigma if rerun_v is not None else None
        value = ValueView(
            recorded=recorded_v, rerun=rerun_v, next_recorded=next_v,
            delta=(next_v - recorded_v) if next_v is not None else None,
            popart_mu=mu, popart_sigma=sigma,
            normalized_recorded=norm_rec, normalized_rerun=norm_rerun,
        )

    # Win probability (--win-prob-mode): recorded P(win|s) + ΔP(win) to the next captured decision —
    # the calibrated analog of the value/ΔV above. None on a run without the head (win_probs NaN/absent).
    recorded_wp = _npz_win_prob(npz, inv_index)
    win_prob = None
    if recorded_wp is not None:
        n = len(summary["invocations"])
        nxt = inv_index + 1
        next_wp = _npz_win_prob(npz, nxt) if (nxt < n and _has_state(npz, nxt)) else None
        win_prob = WinProbView(
            recorded=recorded_wp, next_recorded=next_wp,
            delta=(next_wp - recorded_wp) if next_wp is not None else None,
        )

    # Distributional value head (--value-dist-mode): the predicted RETURN DISTRIBUTION at this state —
    # the interpretability read the scalar V collapses (sharp=confident, wide=uncertain, bimodal=coinflip).
    # Model-free from the trace's per-atom `value_dist` array; the support (atoms) + PopArt denorm come
    # from the loaded model. None on a run without the head (array absent / NaN).
    value_dist = None
    _vds = getattr(model, "value_dist_support", lambda: None)()
    if _vds is not None:
        value_dist = build_value_dist(
            npz, inv_index, _vds, getattr(model, "popart_stats", lambda: None)())

    # Does the loaded model still pick what was recorded? (Exact tier ≈ always; on
    # nearest/recent a disagreement is the interesting case.)
    rerun_argmax = labels[int(np.argmax(probs))]
    agrees = rerun_argmax == chosen
    flags = summary_flags(inv) + (() if agrees else ("disagree",))

    # Field state (weather/spikes/screens) decoded from the obs global block, when
    # the model can decode it (a fake/stub model simply won't have describe_global).
    field = None
    describe = getattr(model, "describe_global", None)
    if describe is not None:
        try:
            field = describe(obs)
        except Exception:  # noqa: BLE001 — field decode is best-effort
            field = None

    # Unified DamageOperator view (per-mon incoming + outgoing per-move), re-computed from the loaded
    # model — None when the checkpoint has no damage op or the model can't expose it (a fake/stub model).
    damage_op = None
    dop = getattr(model, "damage_op_view", None)
    if dop is not None:
        try:
            damage_op = dop(obs, mask)
        except Exception:  # noqa: BLE001 — best-effort, never break the analysis
            damage_op = None

    # Move belief: what the model thinks the REVEALED opponent's still-UNSEEN moves are (+ the per-our-mon
    # labels for the op damage rows). Best-effort, like damage_op; None on a move-belief-off checkpoint.
    move_belief = None
    mbfn = getattr(model, "move_belief", None)
    if mbfn is not None:
        try:
            move_belief = move_belief_view(mbfn(obs, mask))
        except Exception:  # noqa: BLE001 — never break the analysis
            move_belief = None

    # Spread belief: the believed opp DERIVED stats (the op's stat input) vs the true derived stats from the
    # privileged team_details. Best-effort; None on a --spread-belief-off checkpoint.
    spread_belief = None
    sbfn = getattr(model, "spread_belief_view", None)
    if sbfn is not None:
        try:
            spread_belief = build_spread_belief(sbfn(obs, mask), opp_team_details)
        except Exception:  # noqa: BLE001 — never break the analysis
            spread_belief = None

    # Rebuilt from the FINAL board (with the obs-decoded revealed moves), so a move shows revealed the
    # moment it fires.
    opp_full_team = build_opp_full_team(opp_team_details, board)
    return InvocationAnalysis(
        **common, has_state=True, actions=actions, matchups=matchups, sweep=sweep,
        saliency=saliency, value_saliency=value_saliency, threats=threats, incoming=incoming,
        warnings=(), outcome=outcome, value=value, win_prob=win_prob, value_dist=value_dist,
        rerun_argmax=rerun_argmax, agrees=agrees, flags=flags, cure_options=self_cure_options(inv),
        board=board, next_board=next_board,
        obs_mismatch=obs_mismatch, field=field, belief=belief, belief_truth=belief_truth,
        opp_intent=opp_intent,
        damage_op=damage_op, move_belief=move_belief, spread_belief=spread_belief,
        opp_full_team=opp_full_team,
        switch_in_outgoing=switch_in_outgoing, opp_switched_to=opp_voluntary_switch(inv),
    )


# ---------------------------------------------------------------------------
# Loss attribution — categorize the DECISIVE turning point of a loss into a
# fixed taxonomy, so a whole run's losses can be RANKED by which lever (obs /
# reward / self-play / critic / upstream) would recover the most rating.
# ---------------------------------------------------------------------------
# The taxonomy is the single place to extend: add one `_Cat` entry (a name, the
# LEVER it implicates, a one-line blurb, and a predicate over the feature dict).
# `attribute_turning_point` assigns the FIRST matching category (ordered
# most-diagnostic-first), so the buckets are non-overlapping by construction.
#
# A feature dict (built model-free by ProbeSession._turning_point_features from
# the saved summary + npz at the worst-ΔV decision) carries:
#   turns:int|None  is_switch:bool  is_setup:bool  our_hp:float|None
#   our_hp_delta:float|None  faint:bool  active_pko:float|None
#   active_outspeed:float|None  max_pko:float|None  n_healthy_bench:int
#   min_other_pko:float|None  delta_v:float|None  td:float|None  v_at:float|None
#   wp_at:float|None        recorded P(win) at the cliff (win-prob head); the CALIBRATED winning-vs-
#                           losing signal that re-centers the grind/throw split (see `_was_winning`)
#   wp_even / v_even        the winning thresholds triage stamps on (defaults WP_EVEN_DEFAULT / 0.0)
# Every field is optional-tolerant: a predicate must treat None as "unknown"
# (never assume), so a trace missing the belief block still categorizes (it
# just falls through to a coarser bucket) rather than crashing.

STALL_NEAR_CAP = 220          # a loss at/near the 250-turn forfeit cap = a no-progress timeout, not combat
BELIEF_FIRED_PKO = 0.6        # active_pko at/above this = the OHKO belief WARNED of the threat
BELIEF_UNDERREAD_PKO = 0.3    # active_pko below this on a healthy-mon death = the belief MISSED it
HEALTHY_HP = 0.6              # our active counts as "healthy" (a real mon thrown away) above this HP
FAINTED_HP = 0.02            # our active's PRE-decision HP at/below this = already fainted (forced replacement)
CRITIC_CONFIDENT_V = 0.0     # FALLBACK winning threshold on V when no win-prob recorded. NB: V's zero is NOT
                             # "even" — V is a shaped/discounted RETURN with a structural NEGATIVE offset (a
                             # measured self-mirror 50/50 reads V≈−6.5; PopArt μ≈−3.6), so V>0 OVER-counts grinds.
                             # Prefer the calibrated P(win) split (`_was_winning`); re-center this only via v_even.
WP_EVEN_DEFAULT = 0.5        # PRIMARY winning threshold: P(win) at the cliff ≥ this = the model rated the position
                             # WINNING. Calibrated win-odds (the win-prob head), so it is correctly centered at 0.5
                             # — unlike V's sign. (The head can carry a small absolute optimism bias; pass wp_even to
                             # de-bias by a per-checkpoint self-mirror offset if you have one.)

# Common gen3 boosting/setup moves — used at the decisive turn = "set up into a threat" (greedy).
SETUP_MOVES = frozenset({
    "dragondance", "swordsdance", "calmmind", "nastyplot", "bulkup", "curse", "agility",
    "irondefense", "amnesia", "growth", "meditate", "sharpen", "acidarmor", "barrier",
    "cosmicpower", "bellydrum", "tailglow", "rockpolish", "shellsmash", "workup",
})


@dataclass(frozen=True)
class _Cat:
    name: str
    lever: str          # which system axis a fix would touch (the prioritization output)
    blurb: str
    test: "callable"    # (feat) -> bool


def _f(feat, k):
    """feat[k] or None — tolerant of missing keys."""
    return feat.get(k)


def _was_winning(f) -> bool:
    """Did the critic rate the position as WINNING right before the value cratered? This is the
    grind-vs-throw boundary, and it must NOT be the sign of V: V is a shaped/discounted return with a
    structural negative offset (a self-mirror 50/50 reads V≈−6.5), so V>0 systematically UNDER-counts
    "was winning" and over-attributes losses to `positional_grind`. So PREFER the calibrated win-prob
    head — P(win) ≥ wp_even (default 0.5) — and fall back to V > v_even (default 0, re-centerable via the
    structural even-point) only when no win-prob was recorded. Returns False on unknown (no signal)."""
    wp = _f(f, "wp_at")
    if wp is not None:
        return wp >= f.get("wp_even", WP_EVEN_DEFAULT)
    v = _f(f, "v_at")
    return v is not None and v > f.get("v_even", CRITIC_CONFIDENT_V)


# Ordered most-specific / most-diagnostic first. First match wins. The death buckets
# (surprise / ignored / doomed / attrition) are split to separate the LEVERS: a death the
# belief UNDER-read is an OBS gap; a death the belief FIRED on (we had a pivot, didn't take it)
# is a POLICY/REWARD gap; a death with no pivot left is UPSTREAM; the rest is attrition.
LOSS_TAXONOMY = (
    _Cat("stall_timeout", "self-play / stall reward (Φ-price heal moves in the mirror)",
         "lost at/near the 250-turn cap — a no-progress timeout, not a combat loss",
         lambda f: (_f(f, "turns") or 0) >= STALL_NEAR_CAP),

    _Cat("post_faint_replacement",
         "MEASUREMENT/UPSTREAM (worst-ΔV is a forced post-faint pick — the causal turn is earlier; re-scan turn N-1)",
         "our active had ALREADY fainted — this is a forced replacement, not the decision that lost the mon",
         lambda f: _f(f, "our_hp") is not None and f["our_hp"] <= FAINTED_HP),

    _Cat("surprise_ohko", "obs (surprise-OHKO coverage — price unrevealed/just-switched threats)",
         "a HEALTHY mon DIED but the incoming belief UNDER-READ it (unseen / just-switched attacker)",
         lambda f: _f(f, "faint")
                   and (_f(f, "our_hp") is None or f["our_hp"] >= HEALTHY_HP)
                   and _f(f, "active_pko") is not None and f["active_pko"] < BELIEF_UNDERREAD_PKO),

    _Cat("ignored_threat_death",
         "reward/policy (belief FIRED but the policy didn't switch out — the under-switch / doomed_stay target)",
         "the incoming belief FIRED (high P(KO)) and we had a healthy pivot, yet our mon DIED (stayed or pivoted into it)",
         lambda f: _f(f, "faint")
                   and _f(f, "active_pko") is not None and f["active_pko"] >= BELIEF_FIRED_PKO
                   and (_f(f, "n_healthy_bench") or 0) >= 1),

    _Cat("doomed_already", "UPSTREAM (the loss was decided earlier — sequencing/material, look back)",
         "the belief fired high but NO healthy mon left to switch to — the position was already lost",
         lambda f: _f(f, "faint")
                   and _f(f, "active_pko") is not None and f["active_pko"] >= BELIEF_FIRED_PKO
                   and (_f(f, "n_healthy_bench") or 0) == 0),

    _Cat("greedy_setup", "reward/critic (anti-greedy: setup-into-threat + critic tail-blindness)",
         "the decisive move was a SETUP/boost move that got punished",
         lambda f: bool(_f(f, "is_setup"))),

    _Cat("attrition_death", "obs/critic (chip / partial-belief death — a worn-down mon died, belief only partly fired)",
         "our mon DIED with the belief only PARTLY fired (mid P(KO)) or already chipped below healthy — attrition, not a clean surprise",
         lambda f: bool(_f(f, "faint"))),

    _Cat("critic_blindspot", "critic capacity / obs (the critic rated the position WINNING then it craters — confident-wrong: more value capacity / a missing obs feature)",
         "no death this turn, but right before the cliff the model rated the position WINNING — P(win)≥0.5 "
         "(or, no win-prob head, V above its even-point) — a confident-wrong critic miss (a THROW, coachable)",
         lambda f: not _f(f, "faint") and _was_winning(f)),

    _Cat("positional_grind", "UPSTREAM / material (the model already knew it was behind — a slow positional / material loss, not a critic miss)",
         "no death this turn and the model ALREADY rated itself behind (P(win)<0.5, or V below its even-point) "
         "right before the cliff — a gradual positional / material grind (was never ahead to throw)",
         lambda f: not _f(f, "faint")),

    _Cat("other", "unattributed (drill in with analyze)", "did not match a known failure pattern",
         lambda f: True),
)


def attribute_turning_point(feat: dict) -> dict:
    """Assign one loss's decisive turning point to the FIRST matching taxonomy bucket.

    Returns ``{category, lever, blurb}``. Pure + total (the final ``other`` rule matches
    anything), so it never raises on a partial feature dict."""
    for cat in LOSS_TAXONOMY:
        try:
            if cat.test(feat):
                return {"category": cat.name, "lever": cat.lever, "blurb": cat.blurb}
        except Exception:  # noqa: BLE001 — a predicate must never crash the scan; treat as no-match
            continue
    return {"category": "other", "lever": LOSS_TAXONOMY[-1].lever, "blurb": LOSS_TAXONOMY[-1].blurb}


# ---------------------------------------------------------------------------
# Representation probing — fit a small LINEAR probe on the model's INTERNAL
# activations to predict a derived game quantity (is-faster, damage-taken,
# faint-soon). The decisive test of "is X already in the representation": if a
# linear probe recovers X from the trunk embedding, the model HAS computed it
# (so handing X over as a feature is redundant); if a linear probe CAN'T, X is
# an EXTRACTION gap — a real obs lever (per the provide-vs-learn rule, "let it
# learn" has hit this small net's capacity wall for X, so provide it).
#
# Pure numpy (no sklearn). Standardized ridge (regression) / logistic
# (classification) with k-fold OUT-OF-FOLD predictions, scored overall AND per
# group — so we see whether the representation knows X on the HARD/contested
# cases, not just on average (a model can encode speed on an obvious matchup yet
# fail the Leftovers/Sandstorm-timing inference exactly where it's decision-relevant).
# ---------------------------------------------------------------------------

def _kfold_indices(n: int, k: int, seed: int) -> "list[np.ndarray]":
    """k interleaved test folds over a seeded permutation (deterministic)."""
    perm = np.random.default_rng(seed).permutation(n)
    return [perm[i::k] for i in range(min(k, n))]


def _standardize(train: np.ndarray, test: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    mu = train.mean(0)
    sd = train.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + l2 * np.eye(d), X.T @ y)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _logistic_fit(X: np.ndarray, y: np.ndarray, l2: float,
                  iters: int = 400, lr: float = 0.5) -> "tuple[np.ndarray, float]":
    """L2-regularized logistic regression by full-batch gradient descent on
    standardized inputs (robust + dependency-free; converges in a few hundred steps)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(iters):
            p = _sigmoid(X @ w + b)
            g = p - y
            w_new = w - lr * (X.T @ g / n + l2 * w / n)
            b -= lr * g.mean()
            if not np.isfinite(w_new).all():    # a weak-l2 grid point diverged — keep the last finite w
                break
            w = w_new
    return w, b


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann–Whitney); nan if a class is absent."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allp = np.concatenate([pos, neg])
    ranks = allp.argsort().argsort().astype(float) + 1.0  # average-ish ranks (ties rare on floats)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


_L2_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _oof_predict(X, y, task, l2, folds, seed) -> np.ndarray:
    """Out-of-fold predictions for one l2 (every row scored by a model that didn't see it)."""
    n = len(y)
    oof = np.full(n, np.nan)
    for te in _kfold_indices(n, folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        if len(tr) < 2:
            continue
        Xtr, Xte = _standardize(X[tr], X[te])
        if task == "classification":
            w, b = _logistic_fit(Xtr, y[tr], l2)
            oof[te] = _sigmoid(Xte @ w + b)
        else:
            yc = y[tr].mean()
            oof[te] = Xte @ _ridge_fit(Xtr, y[tr] - yc, l2) + yc
    return oof


def _selection_score(y, oof, task) -> float:
    """The scalar used to pick l2: AUC (classification) / R² (regression), over the usable rows."""
    ok = ~np.isnan(oof)
    yy, pp = y[ok], oof[ok]
    if len(yy) < 5:
        return -np.inf
    if task == "classification":
        a = _auc(yy, pp)
        return a if not np.isnan(a) else -np.inf
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    return 1.0 - float(((yy - pp) ** 2).sum()) / ss_tot if ss_tot > 0 else -np.inf


def fit_probe(X, y, task: str, groups=None, seed: int = 0, folds: int = 5, l2=None) -> dict:
    """Fit a cross-validated linear probe and score its OUT-OF-FOLD predictions.

    ``task='classification'`` → logistic, reports accuracy / AUC / base_rate / lift (accuracy −
    majority-class). ``task='regression'`` → ridge, reports r2 / rmse. ``groups`` (per-sample
    labels) adds a per-group breakdown — the easy-vs-contested contrast is the real signal.

    ``l2=None`` (default) **auto-tunes** the ridge/logistic penalty over a grid by the OOF
    selection score — essential because the activation probe is high-dim (≈512) and a fixed weak
    penalty overfits when d≈n (negative OOF R²). Both the representation probe and the 1-D provided
    baseline get the SAME grid, so the comparison stays fair. Pure; no torch, no sklearn. ``overall``
    is None when n is too small."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    grid = (float(l2),) if l2 is not None else _L2_GRID
    chosen_l2, oof = grid[0], np.full(n, np.nan)
    if n >= folds:
        best = -np.inf
        for cand in grid:
            cand_oof = _oof_predict(X, y, task, cand, folds, seed)
            score = _selection_score(y, cand_oof, task)
            if score > best:
                best, chosen_l2, oof = score, cand, cand_oof

    def _metrics(mask) -> "dict | None":
        yy, pp = y[mask], oof[mask]
        ok = ~np.isnan(pp)
        yy, pp = yy[ok], pp[ok]
        if len(yy) < 5:
            return None
        if task == "classification":
            base = float(max(yy.mean(), 1.0 - yy.mean()))
            acc = float(((pp >= 0.5).astype(float) == yy).mean())
            return {"n": int(len(yy)), "accuracy": round(acc, 4), "auc": round(_auc(yy, pp), 4),
                    "base_rate": round(base, 4), "lift": round(acc - base, 4),
                    "pos_rate": round(float(yy.mean()), 4)}
        ss_res = float(((yy - pp) ** 2).sum())
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        return {"n": int(len(yy)), "r2": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
                "rmse": round(float(np.sqrt(ss_res / len(yy))), 4),
                "target_std": round(float(yy.std()), 4)}

    out = {"task": task, "n": n, "l2": chosen_l2, "overall": _metrics(np.ones(n, dtype=bool))}
    if groups is not None:
        g = np.asarray(groups)
        out["by_group"] = {str(k): _metrics(g == k) for k in sorted({str(v) for v in g.tolist()})}
    return out
