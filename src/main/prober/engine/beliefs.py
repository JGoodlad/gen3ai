"""The opponent-model views: species / move / exclusive-species beliefs, and their trajectory.

Every builder here takes RAW head output (or the summary's recorded belief block) and returns a
`views` dataclass. `build_exclusive_belief` additionally applies the species clause at READ time —
a reading aid published BESIDE the raw marginals, never in place of them.
"""

from __future__ import annotations

import re

import numpy as np

from agents import gen3_data
from agents.inference.belief_decode import BELIEF_TOPK

from main.prober.engine.board import _our_items, build_board
from main.prober.engine.util import _norm_species, _npz_array, parse_pct
from main.prober.engine.views import (BeliefSlotView, BeliefTrajectoryPoint, BeliefTrajectoryView,
    BeliefTruthView, BeliefView, BoardView, ExclusiveBeliefView, ExclusiveSlotView,
    MoveBeliefView, OppFullMon, OppFullTeamView, OppMonTruth, OppMoveBelief)


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


def _norm_move(move: str) -> str:
    """The ONE move-name normalisation in this module: lowercase, **alnum-only**, with every
    Hidden-Power variant collapsed to the bare ``hiddenpower``.

    Alnum-only is the substantive half and it is easy to misread: ``"Rock Slide"`` normalises to
    ``rockslide``, NOT ``rock slide`` — spaces, hyphens and brackets all go. That is what lets a
    revealed DISPLAY form (`describe_vector`'s ``hiddenpower(bug)``) meet a truth/dex ID
    (``hiddenpowerbug``) on one key. The HP collapse is deliberate on top of it: the opponent's
    Hidden Power TYPE stays unrevealed until it fires, so believed-vs-revealed can only ever be
    compared on the base move.

    Serves both consumers here — `_move_id_to_num`'s key space and `move_belief_view` /
    `build_opp_full_team`'s revealed-vs-believed compares — so the two cannot drift apart.
    (Until 2026-08-23 there were literally TWO definitions of this name in this module, the
    second shadowing the first; the first promised a `split("(")` normalisation that nothing had
    ever run, and ruff's F811 stayed silent because the name is used BETWEEN the definitions.)
    """
    s = re.sub(r"[^a-z0-9]", "", str(move).lower())
    return "hiddenpower" if s.startswith("hiddenpower") else s


def _move_id_to_num():
    """Cached ``{normalised_move_id -> move_num}`` — the forward map, so a REVEALED move name (from
    `describe_vector`, e.g. ``hiddenpower(fire)``) can be looked up on the move-belief axis to read its
    pinned belief. Keys are `_norm_move`'s, so every Hidden-Power variant already collapses onto the
    single ``hiddenpower`` key and the explicit loop below just pins which num that key carries."""
    global _MOVE_ID_TO_NUM
    if _MOVE_ID_TO_NUM is None:
        raw = gen3_data.moves.raw()
        d = {_norm_move(mid): int(v["num"]) for mid, v in raw.items() if v.get("num")}
        for mid, v in raw.items():
            if mid.startswith("hiddenpower") and v.get("num"):
                d["hiddenpower"] = int(v["num"])
        _MOVE_ID_TO_NUM = d
    return _MOVE_ID_TO_NUM


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


def build_exclusive_belief(species_logits, believed_mask, revealed_species,
                           top_k: int = BELIEF_TOPK, maps=None) -> "ExclusiveBeliefView | None":
    """The SPECIES-CLAUSE reading of the belief head's per-slot species logits.

    Same inputs as `belief_view_from_logits` plus the opponent's REVEALED species (names, from
    `revealed_opp_species`); returns the adjusted per-slot distributions, the point team hypothesis,
    and the raw belief's incoherence headline. `None` when no slot is believed — the same "nothing to
    show" contract as `belief_view_from_logits`, so a surface's two blocks appear and disappear
    together.

    **This does not change the belief.** It is computed at READ time from the published marginals and
    is carried beside them; see `ExclusiveBeliefView` for why both are always shown. Nothing here
    feeds a model, a loss, or an obs — wiring an aggregate consumer is a separate, evidence-gated
    decision (the `tmp/species_exclusivity_measure.py` artifact is that evidence)."""
    from agents.inference.species_exclusivity import (
        coherent_team_hypothesis, exclusive_team_posterior_info)

    num_to_id, id_to_num = maps or _species_maps()
    logits = np.asarray(species_logits, dtype=np.float64)
    mask = np.asarray(believed_mask, dtype=bool)
    hidden = [i for i in range(logits.shape[0]) if i < mask.shape[0] and bool(mask[i])]
    if not hidden:
        return None

    # Revealed NAMES → the belief axis's NUMS. A name the vocab does not know is dropped rather than
    # guessed: a wrong num would zero an innocent species' column, which is a worse answer than one
    # missing constraint.
    revealed_names = tuple(revealed_species or ())
    revealed_nums = sorted({n for n in (id_to_num.get(_norm_species(s)) for s in revealed_names)
                            if n is not None})

    raw = np.stack([_softmax(logits[i]) for i in hidden])                   # [H, S]
    adj, info = exclusive_team_posterior_info(raw, revealed_species=revealed_nums)
    hyp = coherent_team_hypothesis(adj, revealed_species=revealed_nums,
                                   num_to_name=num_to_id, slot_ids=hidden)

    k = max(0, min(top_k, logits.shape[1]))
    raw_top1 = [int(np.argmax(raw[h])) for h in range(len(hidden))]
    slots = []
    for h, slot in enumerate(hidden):
        order = np.argsort(adj[h])[::-1][:k]
        a1 = int(order[0]) if order.size else -1
        r1 = raw_top1[h]
        slots.append(ExclusiveSlotView(
            slot=slot,
            top=tuple((num_to_id.get(int(n), f"num_{int(n)}"), float(adj[h][n])) for n in order),
            raw_top1=num_to_id.get(r1, f"num_{r1}"),
            raw_top1_prob=float(raw[h][r1]),
            adj_top1=num_to_id.get(a1, f"num_{a1}") if a1 >= 0 else "",
            adj_top1_prob=float(adj[h][a1]) if a1 >= 0 else 0.0,
            differs=bool(a1 >= 0 and a1 != r1),
            total_variation=float(info.total_variation[h]),
            hypothesis=hyp[h]["species"],
            hypothesis_differs=bool(hyp[h]["differs"]),
        ))

    return ExclusiveBeliefView(
        slots=tuple(slots),
        team_hypothesis=tuple(h["species"] for h in hyp if h["species"]),
        revealed=revealed_names,
        max_expected_count=float(info.max_expected_count_before),
        illegal_mass=float(info.illegal_mass_before),
        duplicate_top1=len(raw_top1) - len(set(raw_top1)),
        revealed_leak_max=float(info.revealed_leak_before.max()) if len(hidden) else 0.0,
        converged=bool(info.converged),
        iterations=int(info.iterations),
        # The leak bar is 1e-4, not 0: `--species-prior-fusion` floors a revealed species at
        # SPECIES_CLAUSE_LOGIT (1e-6) rather than at -inf, so a few 1e-6 entries are the FLOOR
        # doing its job, not a defect. Measured on 3000 gen-15 decisions the leak never exceeded
        # 3.2e-4, so a 0.0 bar would have called every decision incoherent.
        coherent=bool(float(info.max_expected_count_before) <= 1.0 + 1e-9
                      and len(raw_top1) == len(set(raw_top1))
                      and float(info.revealed_leak_before.max()) <= 1e-4),
    )


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
