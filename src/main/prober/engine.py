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
    mis-calibrated (an encoder bug to chase). Per our 6 slots the block holds
    ``[phys_exp, spec_exp, phys_pko, spec_pko, p_outspeed]``; ``*_pko`` here is ``max(phys,spec)``.
    ``active_*`` pull our on-field mon's slot (located via its per-mon active flag)."""
    present: bool                          # any nonzero KO/chip belief on the board
    max_pko: float                         # worst P(KO) across our team (max(phys,spec) per slot)
    active_pko: "float | None"             # max(phys,spec) P(KO) for our ACTIVE slot
    active_exp: "float | None"             # max(phys,spec) expected-damage-fraction, active slot
    active_outspeed: "float | None"        # P(our active outspeeds the opp), [0,1]
    per_slot_pko: "tuple[float, ...]"      # max(phys,spec) P(KO) per our team slot
    recovery_rate: float                   # opp recovery-move belief (Suicune-Rest discriminator)
    cures_status: float                    # P(opp Rest) — cures its own status clock
    recovery_known: float                  # 1.0 once a recovery move is revealed


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


@dataclass(frozen=True)
class SideBoard:
    active_species: str
    active_hp: str
    status: str                      # "" when none
    boosts: str                      # "" when none
    moves: "tuple[str, ...]"         # active mon's moves (our side only; opp unknown)
    bench: "tuple[MonState, ...]"    # the rest of the (revealed) team


@dataclass(frozen=True)
class BoardView:
    """The board at this decision (model-free, from the summary): each side's
    active mon (species/hp/status/boosts) + revealed bench, and our moveset."""
    ours: SideBoard
    opp: SideBoard


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


_BENCH_RE = re.compile(r"^(.+?)\((.+)\)$")          # "metagross(100%)" / "tyranitar(faint)"
_MOVE_PLACEHOLDER_RE = re.compile(r"move\d$")        # "move0".."move3" filler labels


def _parse_bench(s: str) -> "tuple[MonState, ...]":
    out = []
    for chunk in (s or "").split(", "):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _BENCH_RE.match(chunk)
        if m:
            species, inside = m.group(1), m.group(2)
            fainted = "faint" in inside.lower()
            out.append(MonState(species, "faint" if fainted else inside, fainted))
        else:
            out.append(MonState(chunk, "?", False))
    return tuple(out)


def _side_board(side: dict, moves: "tuple[str, ...]") -> SideBoard:
    return SideBoard(
        active_species=side.get("species", ""),
        active_hp=side.get("hp", "?"),
        status=side.get("status", "") or "",
        boosts=side.get("boosts", "") or "",
        moves=moves,
        bench=_parse_bench(side.get("bench", "")),
    )


def build_board(inv: dict) -> BoardView:
    """Board state at a decision — model-free, parsed from the summary invocation."""
    labels = list(inv.get("actions", {}).keys())
    moves = tuple(k for k in labels[MOVE_START:MOVE_END] if not _MOVE_PLACEHOLDER_RE.fullmatch(k))
    return BoardView(ours=_side_board(inv.get("our", {}), moves),
                     opp=_side_board(inv.get("opp", {}), ()))


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
    """Decode the incoming-damage / OHKO belief block (incoming_damage_v1) from a raw obs vector.

    Pure + model-free (reads the saved obs the trace recorded), so it is exact for the model that
    produced the trace and is reusable by both `analyze_invocation` and the model-free `scan`.
    Returns None when the block isn't present (dim 0, or a too-short synthetic obs)."""
    if off.incoming_dim <= 0:
        return None
    seg = obs[off.incoming_off:off.incoming_off + off.incoming_dim]
    if seg.shape[0] != off.incoming_dim:
        return None
    pm, rec_dim = off.incoming_per_mon, off.incoming_recovery
    n_slots = (off.incoming_dim - rec_dim) // pm
    per_slot_pko, per_slot_exp, outspeeds = [], [], []
    for i in range(n_slots):
        b = i * pm
        phys_exp, spec_exp, phys_pko, spec_pko, outspeed = (float(x) for x in seg[b:b + pm])
        per_slot_pko.append(max(phys_pko, spec_pko))
        per_slot_exp.append(max(phys_exp, spec_exp))
        outspeeds.append(outspeed)
    rec = seg[n_slots * pm:]
    a = _active_slot(obs, off)
    in_range = a is not None and a < n_slots
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
    )


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def analyze_invocation(model, summary: dict, npz, inv_index: int,
                       summary_path: str = "", npz_path: "str | None" = None) -> InvocationAnalysis:
    """Analyze a single decision point. Pure given ``model`` (the torch boundary)."""
    meta = build_meta(summary, summary_path, npz_path)
    inv = summary["invocations"][inv_index]
    chosen = inv["chosen"]
    common = dict(
        meta=meta, inv_index=inv_index, turn=int(inv["turn"]), phase=inv.get("phase", ""),
        our_species=inv["our"]["species"], opp_species=inv["opp"]["species"], chosen=chosen,
    )

    outcome = inv.get("outcome", {}) or {}
    board = build_board(inv)   # model-free; available even without captured state
    if not _has_state(npz, inv_index):
        return InvocationAnalysis(
            **common, has_state=False, actions=(), matchups=None, sweep=None,
            saliency=None, value_saliency=None, threats=None, incoming=None,
            warnings=(f"invocation {inv_index} has no captured state",),
            outcome=outcome, flags=summary_flags(inv), board=board,
        )

    obs = npz["obs"][inv_index].astype(np.float32)
    acts = inv["actions"]
    labels = list(acts.keys())
    mask = np.array([1 if acts[k]["valid"] else 0 for k in labels], dtype=np.int8)

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
    )
