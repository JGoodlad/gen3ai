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

from agents.action.constants import MOVE_END, MOVE_START
from agents.inference.belief_decode import BELIEF_TOPK

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
    rerun_argmax: "str | None" = None              # the loaded model's top valid action
    agrees: bool = True                            # rerun_argmax == chosen
    flags: "tuple[str, ...]" = ()                  # switch/uncertain/faint/disagree
    board: "BoardView | None" = None               # board state at this decision
    field: "dict | None" = None                    # weather/spikes/screens (decoded from obs)
    belief: "BeliefView | None" = None             # hidden-opp species belief (anonymous slots)
    belief_truth: "BeliefTruthView | None" = None  # privileged truth + slot-matched guess (None unless recon+belief)


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


def _parse_bench(s: str, team: "dict | None" = None) -> "tuple[MonState, ...]":
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
            item, moves = e.get("item", ""), tuple(e.get("moves", ()))
            if "faint" in inside.lower():
                out.append(MonState(species, "faint", True, "", item, moves))
            else:
                # "hp%[,STATUS]" — the status tail (incl. any volatiles) is comma-separated.
                hp, _, status = inside.partition(",")
                out.append(MonState(species, hp.strip(), False, status.strip(), item, moves))
        else:
            e = _team_entry(team, chunk)
            out.append(MonState(chunk, "?", False, "", e.get("item", ""), tuple(e.get("moves", ()))))
    return tuple(out)


def _side_board(side: dict, moves: "tuple[str, ...]", team: "dict | None" = None) -> SideBoard:
    team = team or {}
    species = side.get("species", "")
    e = _team_entry(team, species)
    return SideBoard(
        active_species=species,
        active_hp=side.get("hp", "?"),
        status=side.get("status", "") or "",
        boosts=side.get("boosts", "") or "",
        # our active's moves come from the trace actions; the opp active's (and any side with no
        # trace moves) fall back to the obs-decoded revealed moveset.
        moves=moves or tuple(e.get("moves", ())),
        bench=_parse_bench(side.get("bench", ""), team),
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


def build_board(inv: dict, team: "dict | None" = None) -> BoardView:
    """Board state at a decision — model-free, parsed from the summary invocation. ``team``
    (species → {item, moves}) annotates BOTH sides; keys are matched leniently (:func:`_norm_species`)
    so an obs-decoded name resolves against a board id."""
    norm = {_norm_species(k): v for k, v in (team or {}).items()}
    labels = list(inv.get("actions", {}).keys())
    moves = tuple(k for k in labels[MOVE_START:MOVE_END] if not _MOVE_PLACEHOLDER_RE.fullmatch(k))
    return BoardView(ours=_side_board(inv.get("our", {}), moves, norm),
                     opp=_side_board(inv.get("opp", {}), (), norm))


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


_SPECIES_MAPS = None


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


def _matchups(obs: np.ndarray, labels: list, off) -> MatchupView:
    mults = tuple(float(x) * 4.0 for x in obs[off.mm_off:off.mm_off + 4])
    move_labels = tuple(labels[MOVE_START:MOVE_END])  # the 4 move actions, request order
    return MatchupView(multipliers=mults, move_labels=move_labels)


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
        block("active move_multipliers(4)", off.mm_off, off.mm_off + 4),
        block("our_matchups(144)", off.om_off, off.om_off + 144),
        block("their_matchups(144)", off.tm_off, off.tm_off + 144),  # raw incoming type-effectiveness
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


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def analyze_invocation(model, summary: dict, npz, inv_index: int,
                       summary_path: str = "", npz_path: "str | None" = None,
                       opp_team: "tuple[str, ...] | None" = None) -> InvocationAnalysis:
    """Analyze a single decision point. Pure given ``model`` (the torch boundary).

    ``opp_team`` is the opponent's PRIVILEGED full team (species ids from the trace's
    `reconstruction.json` sibling, loaded by the caller — kept out of this pure engine); when given
    AND the model exposes the belief, the result carries the slot-MATCHED `belief_truth`."""
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
    board = build_board(inv, team)   # model-free; available even without captured state
    belief = build_belief(inv)       # model-free summary fallback (re-computed below when a model + state exist)
    belief_truth = None
    if not _has_state(npz, inv_index):
        return InvocationAnalysis(
            **common, has_state=False, actions=(), matchups=None, sweep=None,
            saliency=None, value_saliency=None, threats=None, incoming=None,
            warnings=(f"invocation {inv_index} has no captured state",),
            outcome=outcome, flags=summary_flags(inv), board=board, belief=belief,
        )

    obs = npz["obs"][inv_index].astype(np.float32)
    decode_team = getattr(model, "describe_team", None)
    if decode_team is not None:
        obs_team = decode_team(obs)
        if obs_team:
            board = build_board(inv, {**team, **obs_team})   # both sides, per-turn revealed
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
    acts = inv["actions"]
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
    matchups = _matchups(obs, labels, model.offsets)
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
        value = ValueView(
            recorded=recorded_v, rerun=model.value(obs, mask), next_recorded=next_v,
            delta=(next_v - recorded_v) if next_v is not None else None,
        )

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

    return InvocationAnalysis(
        **common, has_state=True, actions=actions, matchups=matchups, sweep=sweep,
        saliency=saliency, value_saliency=value_saliency, threats=threats, incoming=incoming,
        warnings=(), outcome=outcome, value=value,
        rerun_argmax=rerun_argmax, agrees=agrees, flags=flags, board=board, field=field,
        belief=belief, belief_truth=belief_truth,
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
# Every field is optional-tolerant: a predicate must treat None as "unknown"
# (never assume), so a trace missing the belief block still categorizes (it
# just falls through to a coarser bucket) rather than crashing.

STALL_NEAR_CAP = 220          # a loss at/near the 250-turn forfeit cap = a no-progress timeout, not combat
BELIEF_FIRED_PKO = 0.6        # active_pko at/above this = the OHKO belief WARNED of the threat
BELIEF_UNDERREAD_PKO = 0.3    # active_pko below this on a healthy-mon death = the belief MISSED it
HEALTHY_HP = 0.6              # our active counts as "healthy" (a real mon thrown away) above this HP
FAINTED_HP = 0.02            # our active's PRE-decision HP at/below this = already fainted (forced replacement)
CRITIC_CONFIDENT_V = 0.0     # V(s)>this at a value-cliff = the critic rated the position WINNING (sign-based, scale-invariant)

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
         "no death this turn, but the critic had V(s)>0 (thought we were WINNING) right before the value cratered — a confident-wrong critic miss",
         lambda f: not _f(f, "faint") and _f(f, "v_at") is not None and f["v_at"] > CRITIC_CONFIDENT_V),

    _Cat("positional_grind", "UPSTREAM / material (the critic already knew it was losing — a slow positional / material loss, not a critic miss)",
         "no death this turn and the critic ALREADY had V(s)≤0 (it knew it was losing) — a gradual positional / material grind",
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
