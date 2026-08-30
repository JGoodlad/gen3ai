"""harvest — the STALL-TAIL LABEL FACTORY for the win-prob head repair.

    python -m main.harvest --subject models/<run>/final_model.zip \\
        [--states 300] [--rollouts 32] [--min-turn 60] [--workers 2] [--out DIR]

What this is for
----------------
Probe O (``designs/research_state/measurements/stall_tail_head_reading_2026-08-29.md``) measured a
**conditional** defect in the trained win-probability head: on games that run to the 250-turn
forfeit deadline — which are losses by construction — the head ends above 0.5 on **34.8%** of tails
and above 0.98 on 4.4%, at 4.3x the ordinary-loss rate. The worst specimen held phi = 0.999 across
five consecutive decisions into a -30 forfeit.

The banked account of WHY (ledger ``b63a96f``) is that all three mechanisms are **data-shaped**, not
conceptual: the labels are verified correct, but (1) a sigmoid over additively-combined features
builds a slope where the rule is a multiplicative veto, (2) BCE optimizes the average and cap-game
final turns are epsilon of the buffer, and (3) strong positions finish early, so
``dominant position x turn ~= 249`` barely exists in training at all. Mechanisms (2) and (3) are
exactly what a purpose-built census of that joint attacks.

So this module is a **census instrument, not a propagation one**. The head has no
credit-assignment-through-time problem: Monte-Carlo labels already stamp the terminal outcome onto
every step, so turn-100 of a capped game is labelled 0 directly. What it lacks is *discrimination
mass at the late time-slices*. This manufactures that mass.

The three mechanisms, as built
------------------------------
1. **MID-GAME RE-SEED + MULTI-ROLLOUT** (:func:`label_one`). A recorded late-game state is picked
   up through the reconstruction layer, the recorded action is replayed, and the rest of the battle
   is played live ``R`` times with fresh post-divergence dice against the RELOADED real opponent.
   The result is ``k`` wins of ``n`` adjudicated rollouts — a dense, low-variance per-state win
   RATE where the on-policy stream sees one bit. Its own sampling floor is ``sqrt(p(1-p)/n)``,
   which is 0.088 at ``p=0.5, n=32`` against the 0.5 of a single bit.

2. **PRIORITIZED SELECTION** (:func:`priority_of`). Candidates are ranked by the subject's own
   confessed uncertainty (``CfEvidentialHead``'s Beta width) blended with the head-vs-realized gap
   and a drag-signature bonus. This is semantics-free active learning: the over-confident tail
   probe O convicted gets hammered without anyone hand-writing what "over-confident" means.

3. **SLICE RE-WEIGHTING** happens in the CONSUMER (``agents.training.winprob_finetune``), not here.
   This module's job is to make the rows and to record honestly what selected them; re-weighting a
   sample is the fitter's decision and it needs the ``priority`` field to make it, which is why
   ``priority`` is a pinned schema column rather than a scratch variable.

The subject checkpoint is re-scored, ALWAYS
-------------------------------------------
A trace's recorded ``win_probs`` came from whichever policy played it, and the harvest mines many
runs' traces to feed ONE subject. Measured on this box: the subject's re-scored phi differs from the
recorded phi by up to 0.135 on that subject's OWN traces (the trace was written by an earlier
checkpoint of the same run). So ``phi_head`` is always a fresh forward of the subject
(:func:`score_candidates`), never the recorded column. The rollouts likewise run the SUBJECT as the
trainee via ``ProbeSession(ckpt_override=...)``, which is what makes ``k/n`` an estimate of *the
subject's* value at that state rather than of the run that happened to record it.

Holdout is decided HERE, before a single label is bought
--------------------------------------------------------
The meter (``main.harvest_meter``) scores held-out stall tails pre/post fine-tune, and the split
must be at the **battle** level — states inside one battle are not independent, and a state-level
split would put turn 118 in train and turn 119 in test and call the result a generalization. Rather
than trust three modules to agree about that, the split is computed here from ``--seed``, written to
``holdout.json``, and the harvest **refuses to draw a candidate from a held-out battle**. Leakage is
made unrepresentable instead of forbidden.

Adjudication and the timeout bucket
-----------------------------------
A rollout is scored only if it reached a terminal — a win, a loss, or a TIE (the 250-turn cap
forfeit is recorded as a LOSS, so cap endings adjudicate correctly and need no special case).
Anything else — a transport error, a bridge timeout, a horizon overrun — lands in
``provenance["n_timeout"]`` and is excluded from BOTH ``n_wins`` and ``n_rollouts``. A timeout is
never a semantic outcome: folding one into the denominator would make a busy box read as a losing
position, which is the exact error that once let a starved parity run report 39/40 timeouts as a
clean pass.

⚠️ A TIE, by contrast, IS a semantic outcome and belongs in the denominator as a non-win — it went
into the timeout bucket until the R1 adversarial review, which biased ``k/n`` upward on every state
where a game can end even. It is also counted on its own in ``provenance["n_tie"]``, because a
denominator that silently absorbed a second outcome class is a denominator nobody can audit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.training.harvest_schema import (
    HarvestRow, obs_b64, obs_digest, write_rows,
)

#: Declared, versioned selection weights. A silent priority change is a distribution-shift
#: confound for every downstream readout, so this string is written into every manifest and the
#: weights below are the only place it is decided.
SAMPLER_VERSION = "harvest_stall_tail_v1"

#: Blend weights for :func:`priority_of`. They sum to 1 so ``priority`` stays in [0, 1] and is
#: comparable across harvests at the same version.
W_GAP, W_UNCERTAINTY, W_DRAG = 0.5, 0.3, 0.2

#: The lowest turn the offline replay driver will anchor at. Matches ``cf_audit``'s sampler bound:
#: turn 1 became openable on both impls in 2026-08, but lowering the two independently would change
#: the audited population, so they move together or not at all.
MIN_LABELABLE_TURN = 2

#: A handful of very long games must not carry a whole harvest. Same value ``cf_audit`` uses.
MAX_PER_BATTLE = 12

#: Probe O's classes, verbatim, so a harvest and the probe that motivated it partition the corpus
#: the same way. ``turns >= CAP_TURNS`` is the ``MAX_TURNS`` forfeit deadline.
CAP_TURNS = 250
DRAG_TAIL = 20          # the trailing window probe O counts faints in
DRAG_MIN_TURNS = 100    # "long" in probe O's taxonomy

#: How many decisions from the END of a battle count as its TAIL. Matches the meter's ``K_TAIL``
#: (and probe O's K = 5) DELIBERATELY: the meter scores the last five decisions of a doomed game,
#: so that is the region the head must be supervised on. Fitting the same REGION on DIFFERENT
#: battles is generalization; the battles themselves are held out, so this is not leakage.
TAIL_K = 5


# ---------------------------------------------------------------------------
# The candidate frame
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One reconstructable late-game decision, before any model has looked at it."""
    run: str
    battle_tag: str          # trace prefix RELATIVE to the models root — the join key
    abs_prefix: str          # the same prefix, absolute (not serialized)
    decision_idx: int        # the INVOCATION index == the states.npz row
    turn: int
    action: int
    opponent: str
    outcome: str             # "win" | "loss"
    battle_turns: int
    faints_tail: int
    is_cap: bool
    recorded_phi: float
    tail_rank: int = 99      # 0 = the battle's LAST move decision

    # filled by score_candidates
    phi_head: Optional[float] = None
    beta_alpha: Optional[float] = None
    beta_beta: Optional[float] = None
    priority: float = 0.0

    @property
    def realized(self) -> float:
        """The battle's actual outcome as a 0/1 — ONE sample of the terminal, not a label."""
        return 1.0 if self.outcome == "win" else 0.0

    @property
    def is_stall(self) -> bool:
        """Probe O's stall signature: a long game whose tail stopped producing faints. The probe
        measured that this and 'cap ending' are ONE population, and that replicates here — the
        current-arch corpus holds **zero** non-capping stall losses. So this is not an independent
        class; it is kept because it names the thing, and because a future corpus may separate."""
        return self.is_cap or (self.battle_turns >= DRAG_MIN_TURNS
                               and self.outcome == "loss" and self.faints_tail == 0)

    @property
    def meter_class(self) -> Optional[str]:
        """``"cap"`` | ``"long_loss"`` | ``None`` — which doomed-tail population this belongs to.

        **Why this is wider than "cap", and why widening is faithful rather than convenient.**
        The obvious target is the cap ending, because that is where probe O's headline lives
        (34.8% of cap tails end phi_T >= 0.5). But only **8 of 48** current-arch cap battles carry a
        replayable record (see :func:`cap_record_replayable`), and a meter over 8 battles minus a
        holdout reads nothing at all.

        The defect itself is not "caps". Probe O's own framing is a **heavy right tail on the doomed
        side** — the head is over-confident on states whose game is lost — and it demonstrated the
        failure is NOT a length effect by showing LONG_WIN reads 0.986 at 128 median turns. Cap
        endings are where that tail *concentrates* (4.3x the ordinary-loss rate), not the only place
        it exists. Long losses carry the same failure at lower density, and there are 245 of them.

        So the doomed-tail population is caps PLUS long losses, the two are stratified separately
        everywhere they are sampled or reported, and the meter never pools them into one number —
        because the head reads them very differently (probe O: detect_le05 0.652 on caps vs
        0.94-0.95 on long losses) and an average over both would hide exactly the cell of interest.
        """
        if self.is_cap:
            return "cap"
        if self.battle_turns >= DRAG_MIN_TURNS and self.outcome == "loss":
            return "long_loss"
        return None

    @property
    def is_drag(self) -> bool:
        """Membership in the doomed-tail population the harvest over-samples and the meter scores."""
        return self.meter_class is not None

    @property
    def is_tail(self) -> bool:
        """One of the battle's last :data:`TAIL_K` decisions — the region the meter actually reads.

        **This distinction was earned by a failed pilot and it is the single most important thing
        the sampler knows.** A 200-state harvest ranked purely on priority drew states at turns
        60-152 (p50 90) with a mean MC label of 0.621 — mid-game positions the subject WINS most of
        the time. The meter, meanwhile, reads the last five decisions of doomed games at turns
        96-239, and **29.3% of those turns were past the harvest's maximum turn entirely.** The head
        was therefore fit on a population it wins 62% of, never shown a losing tail, and collapsed
        to a near-constant ~0.6: on the held-out long losses `phi_T` went 0.070 -> 0.607 and on the
        untouched long-WIN control 0.943 -> 0.567. Both directions worse, from one cause.

        A label factory that never samples the region its meter scores is extrapolating, and
        extrapolation from a 62%-win sample to a 0%-win population is exactly the regression to the
        sample mean that was measured.
        """
        return self.tail_rank < TAIL_K

    @property
    def beta_evidence(self) -> Optional[float]:
        """The Beta's precision alpha+beta. LOW precision is high confessed uncertainty."""
        if self.beta_alpha is None or self.beta_beta is None:
            return None
        return float(self.beta_alpha + self.beta_beta)

    @property
    def beta_mean(self) -> Optional[float]:
        if self.beta_alpha is None or self.beta_beta is None:
            return None
        return float(self.beta_alpha / (self.beta_alpha + self.beta_beta))

    @property
    def beta_width(self) -> Optional[float]:
        """``CfEvidentialHead.epistemic_std`` in closed form — the head's own confessed uncertainty.
        Computed here rather than called through the module so the harvest stays importable without
        torch on the selection path."""
        if self.beta_alpha is None or self.beta_beta is None:
            return None
        a, b = float(self.beta_alpha), float(self.beta_beta)
        p = a + b
        return float(np.sqrt(a * b / (p * p * (p + 1.0))))


def cap_record_replayable(prefix: str) -> bool:
    """Does this CAP battle's reconstruction record carry the terminal that ends it?

    A 250-turn game ends by FORFEIT — ``Gen3Env.action_to_order`` returns a ``ForfeitBattleOrder``
    at ``MAX_TURNS`` — and the bridges log that as a ``["forcelose", side]`` command. When the entry
    is missing, the record describes a battle that never finishes, and the offline replay driver
    refuses it in those words: *"replayed all N commands but battle has not ended (turn 250) —
    corrupt or truncated record?"*. Both impls assert it, in the ``replay`` verb that
    ``materialize_from_record`` depends on, so EVERY model-scored offline path is blocked on such a
    record, not just this one.

    **Measured over the whole archive: 689 cap records, 543 (79%) carry the forfeit, 146 do not.**
    The 21% are skipped and counted (``cap_record_unterminated``). Since cap battles are the
    scarcest and most valuable population here — 48 in the entire current-arch corpus — it is worth
    the one extra JSON read per cap battle to find this out before the sampler spends budget on
    states that can never be labelled.

    **Why the missing forfeit is NOT synthesized.** It would be easy to append
    ``["forcelose", trainee_side]`` in memory and make every one of these replay: the battle is a
    LOSS at exactly 250 turns, so a trainee forfeit is the overwhelmingly likely terminal. It is
    not done, because "overwhelmingly likely" is not a basis on which a LABEL FACTORY may
    manufacture an ending. A record that lacks its terminal may lack it because the opponent
    forfeited, or because the recorder was cut off mid-write; inventing our own loss in either case
    would fabricate exactly the outcome the harvest is trying to measure. Skipping is honest and
    the count is published.

    Non-cap battles are NOT pre-checked — they terminate naturally, the check costs a JSON read per
    battle across ~1,400 of them, and the labeler already counts and rejects a failure it hits.
    """
    path = prefix + "_reconstruction.json"
    try:
        with open(path) as fh:
            cmds = json.load(fh).get("commands") or []
    except (OSError, ValueError):
        return False
    return any(isinstance(c, (list, tuple)) and c and c[0] == "forcelose" for c in cmds)


def _faints_in_tail(invs: Sequence[dict], tail: int = DRAG_TAIL) -> int:
    """Probe O's ``faints_tail``, verbatim — faints recorded in the last ``tail`` game turns."""
    if not invs:
        return 0
    last = max((i.get("turn") or 0) for i in invs)
    return sum(
        sum(1 for e in ((i.get("outcome") or {}).get("events") or []) if ":fainted" in e)
        for i in invs if (i.get("turn") or 0) >= last - tail)


def current_arch_runs(models_root: str) -> List[str]:
    """Run names under ``models_root`` whose ``model_config.json`` matches the CURRENT
    ``ARCH_SIGNATURE``. A trace from any other run has a different obs family and cannot be scored
    by the subject at all, so this is a hard filter, not a preference."""
    from agents.model.model_version import ARCH_SIGNATURE
    out = []
    for name in sorted(os.listdir(models_root)):
        cfg = os.path.join(models_root, name, "model_config.json")
        if not os.path.exists(cfg):
            continue
        try:
            with open(cfg) as fh:
                c = json.load(fh)
        except (OSError, ValueError):
            continue
        if str(c.get("arch_signature")) == str(ARCH_SIGNATURE):
            out.append(name)
    return out


def build_candidates(models_root: str, runs: Sequence[str], *, min_turn: int = 60,
                     ) -> "Tuple[List[Candidate], Counter]":
    """Every labelable late-game decision under ``runs``, plus a census of what was skipped.

    A decision qualifies when it is a ``move_selection`` round with a recorded state, its battle has
    a ``*_reconstruction.json`` sibling, and either the decision is at ``turn >= min_turn`` OR the
    battle is a **cap ending**, in which case every one of its decisions qualifies. Caps are the
    scarce population probe O convicted (48 battles in the whole current-arch corpus), so they are
    swept whole rather than filtered by turn.

    A skip that is not counted is a skip that is not known, so every rejection increments a named
    counter and the census is printed and written to the manifest.
    """
    import glob

    rows: List[Candidate] = []
    skipped: Counter = Counter()
    for run in runs:
        pat = os.path.join(models_root, run, "eval_traces", "*", "*", "*_summary.json")
        for sp in sorted(glob.glob(pat)):
            base = sp[: -len("_summary.json")]
            if not os.path.exists(base + "_reconstruction.json"):
                skipped["no_reconstruction_sibling"] += 1
                continue
            try:
                with open(sp) as fh:
                    summ = json.load(fh)
                with np.load(base + "_states.npz") as z:
                    files = set(z.files)
                    if "win_probs" not in files or "actions" not in files:
                        skipped["no_win_prob_head"] += 1
                        continue
                    wps = np.asarray(z["win_probs"], dtype=float)
                    has = np.asarray(z["has_state"], dtype=int)
                    acts = np.asarray(z["actions"], dtype=int)
            except Exception as exc:                                    # noqa: BLE001
                skipped[f"load:{type(exc).__name__}"] += 1
                continue

            meta = summ.get("meta") or {}
            outcome = str(meta.get("result") or "").lower()
            if outcome not in ("win", "loss"):
                skipped[f"outcome:{outcome or 'none'}"] += 1
                continue
            battle_turns = int(meta.get("turns") or 0)
            is_cap = battle_turns >= CAP_TURNS
            if battle_turns < min_turn and not is_cap:
                skipped["battle_too_short"] += 1
                continue
            if is_cap and not cap_record_replayable(base):
                # The record has no terminal, so no offline path can open it. Counted, never
                # silent — this is 21% of cap battles archive-wide and it is the scarcest
                # population the harvest has.
                skipped["cap_record_unterminated"] += 1
                continue

            invs = summ.get("invocations", [])
            faints = _faints_in_tail(invs)
            opp = os.path.basename(os.path.dirname(sp))
            # Rank every move decision from the END, so `is_tail` can name the region the meter reads.
            move_idx = [i for i, iv in enumerate(invs) if iv.get("phase") == "move_selection"]
            rank_from_end = {i: (len(move_idx) - 1 - k) for k, i in enumerate(move_idx)}
            kept_here = 0
            for i, iv in enumerate(invs):
                if iv.get("phase") != "move_selection":
                    skipped["forced_switch_round"] += 1
                    continue
                turn = int(iv.get("turn", -1))
                if turn < MIN_LABELABLE_TURN:
                    skipped["turn_below_driver_bound"] += 1
                    continue
                if not is_cap and turn < min_turn:
                    skipped["decision_too_early"] += 1
                    continue
                if i >= len(has) or not has[i]:
                    skipped["no_recorded_state"] += 1
                    continue
                wp = float(wps[i]) if i < len(wps) else float("nan")
                if not np.isfinite(wp):
                    skipped["nan_win_prob"] += 1
                    continue
                rows.append(Candidate(
                    run=run,
                    battle_tag=os.path.relpath(base, models_root),
                    abs_prefix=base,
                    decision_idx=i,
                    turn=turn,
                    action=int(acts[i]) if i < len(acts) else -1,
                    opponent=opp,
                    outcome=outcome,
                    battle_turns=battle_turns,
                    faints_tail=faints,
                    is_cap=is_cap,
                    recorded_phi=wp,
                    tail_rank=rank_from_end.get(i, 99),
                ))
                kept_here += 1
            if kept_here:
                skipped["battles_kept"] += 1
    return rows, skipped


# ---------------------------------------------------------------------------
# Holdout — decided before a single label is bought
# ---------------------------------------------------------------------------

def meter_battles(cands: Sequence[Candidate]) -> "Dict[str, List[str]]":
    """The doomed-tail battle tags the meter can score, keyed by class (``cap`` / ``long_loss``).

    Kept as a mapping rather than a flat list so every consumer downstream is forced to say which
    class it means. The two read very differently and pooling them averages the cell of interest
    away — see :attr:`Candidate.meter_class`.
    """
    out: Dict[str, List[str]] = {}
    seen: Dict[str, str] = {}
    for c in cands:
        cls = c.meter_class
        if cls is None or c.battle_tag in seen:
            continue
        seen[c.battle_tag] = cls
        out.setdefault(cls, []).append(c.battle_tag)
    return {k: sorted(v) for k, v in out.items()}


def battle_holdout(cands: Sequence[Candidate], frac: float, seed: int) -> List[str]:
    """A deterministic battle-level holdout drawn from the METER's population, STRATIFIED BY CLASS.

    Returns the held-out battle tags. Two properties matter and neither is incidental:

    * Drawing from the meter population rather than from every battle is what makes a small
      ``frac`` still yield a scorable test set — a holdout spread over all 1,400 long games would
      put almost no doomed tails in it.
    * Stratifying **within class** is what stops the scarce population from landing entirely on one
      side. There are 8 replayable cap battles; an unstratified 35% draw over the pooled doomed
      tails can easily take 0 of them, or all 8, and either way one arm of the meter loses the
      class the whole exercise is about. Each class contributes ``ceil`` of its own share.
    """
    pools = meter_battles(cands)
    if not pools or frac <= 0:
        return []
    rng = random.Random(seed)
    held: List[str] = []
    for cls in sorted(pools):
        shuffled = list(pools[cls])
        rng.shuffle(shuffled)
        n = max(1, int(round(len(shuffled) * frac))) if shuffled else 0
        held.extend(shuffled[:n])
    return sorted(held)


# ---------------------------------------------------------------------------
# Scoring — one batched CPU forward of the SUBJECT over every candidate
# ---------------------------------------------------------------------------

def score_candidates(subject_ckpt: str, cands: Sequence[Candidate], *, models_root: str,
                     batch_size: int = 256, verbose: bool = True) -> dict:
    """Fill ``phi_head`` / ``beta_alpha`` / ``beta_beta`` in place, batched, on CPU.

    Measured on this box: ~1.4 ms per state for the joint win-prob + evidential read, so scoring a
    50,000-candidate frame is about a minute — cheap enough that the whole frame is scored and the
    priority is computed over all of it, rather than scoring a pre-filtered subset and thereby
    letting a cheap heuristic decide what the expensive one gets to see.

    The evidential columns are BEST-EFFORT: a subject with no ``cf_evid_head`` (``--cf-evidential``
    off, or pre-v98) leaves them ``None`` and :func:`priority_of` falls back to the gap term alone,
    which is stated in the manifest. Absent is not zero — "this checkpoint has no head" and "this
    head claims no uncertainty" are different facts.
    """
    import torch

    from main.prober.model import ProbeModel

    torch.set_num_threads(1)
    pm = ProbeModel.load(subject_ckpt, "cpu")
    have_evid = getattr(getattr(pm._policy, "features_extractor", None), "cf_evid_head", None) is not None

    # Group by npz so each file is opened once. Candidates from one battle are contiguous already,
    # but a run's battles interleave across step dirs after sorting, so group explicitly.
    by_npz: Dict[str, List[Candidate]] = {}
    for c in cands:
        by_npz.setdefault(c.abs_prefix + "_states.npz", []).append(c)

    n_scored, n_failed = 0, 0
    t0 = time.time()
    for npz_path, group in by_npz.items():
        try:
            with np.load(npz_path) as z:
                obs = np.asarray(z["obs"], dtype=np.float32)
        except Exception:                                               # noqa: BLE001
            n_failed += len(group)
            continue
        for start in range(0, len(group), batch_size):
            chunk = group[start:start + batch_size]
            idx = [c.decision_idx for c in chunk]
            if max(idx) >= len(obs):
                n_failed += len(chunk)
                continue
            batch = obs[idx]
            try:
                phi = _win_prob_batch(pm, batch)
                ev = pm.cf_evidential_batch(batch) if have_evid else None
            except Exception as exc:                                    # noqa: BLE001
                if verbose:
                    print(f"  score: batch failed ({type(exc).__name__}: {str(exc)[:100]})",
                          flush=True)
                n_failed += len(chunk)
                continue
            for j, c in enumerate(chunk):
                c.phi_head = float(phi[j]) if phi is not None else None
                if ev is not None:
                    c.beta_alpha, c.beta_beta = float(ev[0][j]), float(ev[1][j])
            n_scored += len(chunk)
    dt = time.time() - t0
    if verbose:
        print(f"  scored {n_scored} candidates in {dt:.1f}s "
              f"({1000 * dt / max(1, n_scored):.2f} ms/state), {n_failed} failed, "
              f"evidential={'on' if have_evid else 'ABSENT (columns omitted)'}", flush=True)
    return {"n_scored": n_scored, "n_failed": n_failed, "have_evidential": have_evid,
            "score_seconds": round(dt, 1)}


def _win_prob_batch(pm, obs: np.ndarray) -> Optional[np.ndarray]:
    """The batched sibling of ``ProbeModel.win_prob_at``.

    Rides the extractor's ``last_win_prob_logits`` stash exactly as the single-state method does.
    The mask is all-legal because the extractor forward reads only ``"observation"`` — that is the
    key going unread, not an approximation (``ProbeModel._value_pooled_batch`` documents the same
    thing for the cf heads).
    """
    import torch

    from agents.action.constants import ACTION_SPACE_SIZE

    ex = getattr(pm._policy, "features_extractor", None)
    if ex is None:
        return None
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    pm._check_obs_dim(obs)
    ot = torch.as_tensor(obs)
    mt = torch.ones(ot.shape[0], ACTION_SPACE_SIZE)
    with torch.no_grad():
        pm._policy.extract_features({"observation": ot, "action_mask": mt})
    logits = ex.last_win_prob_logits
    if logits is None:
        return None
    return torch.sigmoid(logits[:, 0]).cpu().numpy()


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

def priority_of(c: Candidate, mode: str = "blend") -> float:
    """The selection score, in [0, 1]. Higher is harvested first.

    ``blend`` (default) = ``0.5*gap + 0.3*uncertainty + 0.2*drag``:

    * **gap** ``= |phi_head - realized|``. ``realized`` is the battle's actual 0/1 outcome, so this
      is a NOISY proxy — a state legitimately worth 0.7 in a game that was then lost scores high on
      it. That is acceptable and deliberate: this is a **selection rule, not a measurement**, and it
      points exactly at the over-confident-into-a-loss population probe O convicted. It is
      corrected for downstream by ``priority`` travelling in the row, so a consumer can re-weight
      or ablate against it instead of inheriting it silently.
    * **uncertainty** ``= min(1, 2*beta_width)``. The Beta's epistemic std is at most 0.5, so the
      doubling maps a maximally-uncertain head to 1.0. This is the semantics-free half: the model
      nominates its own weak spots.
    * **drag** ``= 1`` on a cap ending or a zero-faint long tail, else 0 — the population the meter
      scores, given a floor so it is never crowded out entirely by a merely-surprising early state.

    Other modes exist so the default is not the only option and an ablation is one flag away:
    ``gap`` / ``evidence`` / ``drag`` isolate a term, ``random`` is the control that answers "did
    prioritizing buy anything at all?".
    """
    gap = abs((c.phi_head if c.phi_head is not None else c.recorded_phi) - c.realized)
    width = c.beta_width
    unc = min(1.0, 2.0 * width) if width is not None else 0.0
    drag = 1.0 if c.is_drag else 0.0
    if mode == "gap":
        return gap
    if mode == "evidence":
        return unc
    if mode == "drag":
        return drag
    if mode == "random":
        return 0.0
    if width is None:
        # No evidential head: renormalize over the two terms that exist rather than silently
        # scoring every candidate 0.3 lower than a run that has one.
        return (W_GAP * gap + W_DRAG * drag) / (W_GAP + W_DRAG)
    return W_GAP * gap + W_UNCERTAINTY * unc + W_DRAG * drag


def _rank(pool: List[Candidate], mode: str, rng: random.Random) -> List[Candidate]:
    if mode == "random":
        out = list(pool)
        rng.shuffle(out)
        return out
    # Deterministic tie-break: the jitter is drawn from a seeded RNG so two runs at one seed
    # produce the same order, but ties do not systematically favour whichever battle sorted
    # first on disk.
    return sorted(pool, key=lambda c: (-c.priority, rng.random(), c.battle_tag, c.decision_idx))


def _take(ranked: Sequence[Candidate], budget: int, per_battle: Counter,
          max_per_battle: int, taken: set) -> List[Candidate]:
    out: List[Candidate] = []
    for c in ranked:
        if len(out) >= budget:
            break
        key = (c.battle_tag, c.decision_idx)
        if key in taken or per_battle[c.battle_tag] >= max_per_battle:
            continue
        per_battle[c.battle_tag] += 1
        taken.add(key)
        out.append(c)
    return out


def select(cands: Sequence[Candidate], n_states: int, *, mode: str = "blend", seed: int = 0,
           max_per_battle: int = MAX_PER_BATTLE, drag_frac: float = 0.6,
           general_win_frac: float = 0.5, tail_frac: float = 0.5,
           exclude_battles: Sequence[str] = ()) -> List[Candidate]:
    """Rank by :func:`priority_of` and take ``n_states``, capped per battle, in THREE STRATA.

    ``exclude_battles`` is the meter's holdout and is applied FIRST — a held-out battle never even
    enters the ranking, so no amount of later slicing can readmit it.

    **Why strata and not one ranking.** Two measurements on the real frame forced this, in order:

    1. The blend priority, left alone, fills the whole sample from drag tails — a 40-state draw
       came from 4 battles, all doomed. A head fine-tuned only on doomed late states has a trivial
       way to score perfectly: say 0 at every late turn. That is a BIAS, not a repair, and it would
       wreck the thing probe O found the head already does RIGHT — ``LONG_WIN`` at 128 median turns
       reads phi = 0.986, so the failure is a heavy right tail on the doomed side, not a length
       effect. Hence ``drag_frac`` (default 0.6) caps the doomed share.

    2. Capping it was **not enough**. With the general stratum also ranked by priority, a 300-state
       draw took **1** state from a won battle. The gap term is ``|phi - realized|`` and on a won
       battle a confident head has ``phi ~ 1, realized = 1``, so a correctly-read win scores ~0 and
       is never drawn. The control stratum was selected out of existence by the very rule that
       makes the doomed stratum good.

    So the general stratum is additionally balanced on the realized OUTCOME
    (``general_win_frac``, default 0.5): won and lost battles each get half of it, ranked by
    priority within their own side. That guarantees the fit sees both directions of late-game
    truth, and it is what makes the meter's long-win control an honest test rather than a
    formality.
    """
    excl = set(exclude_battles)
    pool = [c for c in cands if c.battle_tag not in excl]
    rng = random.Random(seed)
    for c in pool:
        c.priority = priority_of(c, mode)

    drag_budget = int(round(n_states * max(0.0, min(1.0, drag_frac))))
    per_battle: Counter = Counter()
    taken: set = set()

    drag_pool = [c for c in pool if c.is_drag]
    gen_pool = [c for c in pool if not c.is_drag]
    gen_win = [c for c in gen_pool if c.outcome == "win"]
    gen_loss = [c for c in gen_pool if c.outcome != "win"]

    # CAPS FIRST inside the doomed stratum. Measured: without this, a 240-state pilot drew
    # **zero** cap states — there are only 8 replayable cap battles against 245 long losses, and a
    # single blended ranking lets the plentiful class outrank the scarce one on priority alone.
    # Since the cap ending IS probe O's headline class, losing it entirely to a ranking artifact
    # would leave the harvest unable to supervise the very cell it was built for. Caps are capped
    # by `max_per_battle` like everything else, so "first" cannot mean "all of it".
    cap_pool = [c for c in drag_pool if c.meter_class == "cap"]
    rest_drag = [c for c in drag_pool if c.meter_class != "cap"]
    # TAILS FIRST, then caps, then the rest — the order the failed pilot bought. `tail_frac` of the
    # doomed budget is reserved for the last TAIL_K decisions of a doomed battle, because that is
    # the region the meter reads and the region a purely priority-ranked draw never reaches (see
    # `Candidate.is_tail` for the measurement).
    tail_budget = int(round(drag_budget * max(0.0, min(1.0, tail_frac))))
    out: List[Candidate] = []
    for group in ([c for c in cap_pool if c.is_tail], [c for c in rest_drag if c.is_tail]):
        out += _take(_rank(group, mode, rng), tail_budget - len(out), per_battle,
                     max_per_battle, taken)
    out += _take(_rank(cap_pool, mode, rng), drag_budget - len(out), per_battle,
                 max_per_battle, taken)
    out += _take(_rank(rest_drag, mode, rng), drag_budget - len(out), per_battle,
                 max_per_battle, taken)
    # Whatever the drag stratum could not supply (caps are scarce — 48 battles in the whole
    # current-arch corpus) rolls over to the general strata rather than shrinking the harvest.
    remaining = n_states - len(out)
    win_budget = int(round(remaining * max(0.0, min(1.0, general_win_frac))))
    out += _take(_rank(gen_win, mode, rng), win_budget, per_battle, max_per_battle, taken)
    out += _take(_rank(gen_loss, mode, rng), n_states - len(out), per_battle,
                 max_per_battle, taken)
    # Backfill in a fixed order if any stratum ran dry, so the harvest is never short by accident.
    for backfill in (gen_win, drag_pool, gen_loss):
        if len(out) >= n_states:
            break
        out += _take(_rank(backfill, mode, rng), n_states - len(out), per_battle,
                     max_per_battle, taken)
    return out


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _worker_init(subject_ckpt: str, impl: str) -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    import torch
    torch.set_num_threads(1)
    _WORKER["subject"] = subject_ckpt
    _WORKER["impl"] = impl
    _WORKER["sessions"] = {}
    _WORKER["models"] = {}


def _session_for(abs_prefix: str):
    """A ``ProbeSession`` for the traces dir this battle lives in, cached per worker.

    The model cache is SHARED across sessions on purpose: every session in a worker resolves to the
    same subject checkpoint via ``ckpt_override``, and a fresh ``MaskablePPO.load`` per traces dir
    would pay ~1.5 s per step directory for a model that is already in memory.
    """
    from main.prober.session import ProbeSession

    traces = os.path.dirname(os.path.dirname(abs_prefix))
    sess = _WORKER["sessions"].get(traces)
    if sess is None:
        sess = ProbeSession(traces, impl=_WORKER["impl"],
                            ckpt_override=_WORKER["subject"], compile_extractor=False)
        sess._play_models = _WORKER["models"]
        _WORKER["sessions"][traces] = sess
    return sess


def label_one(sess, abs_prefix: str, decision_idx: int, action: int, n_rollouts: int) -> dict:
    """One re-seeded multi-rollout label: replay the recorded action, roll to a terminal ``R`` times.

    Returns ``{n, k, n_tie, n_timeout, outcomes, opponent_source, seconds}``. ``n`` counts
    ADJUDICATED rollouts — a rollout that reached a terminal, which is win, loss OR TIE; everything
    else (``unfinished``: a transport error, a bridge timeout, a horizon overrun) is ``n_timeout``.
    See the module header on why a timeout is its own bucket rather than a loss.

    ⚠️ THE TIE IS IN THE DENOMINATOR, and it was not until the R1 adversarial review.
    ``counterfactual._battle_outcome`` emits four values — ``win`` / ``loss`` / ``tie`` /
    ``unfinished`` — and ``n = win + loss`` with ``n_timeout = total - n`` filed a tie as a
    timeout, which is none of the three things the module header says that bucket counts. The cost
    is a biased label, not just a mislabelled counter: every dropped rollout is a NON-win, so
    ``k/n`` overstates P(win) on exactly the states where a game can end even. The owner's
    clean-world ruling says the same thing in the reward's language — a draw scores ``-victory``,
    i.e. as a loss. ``n_tie`` is carried separately so a label's composition stays auditable.
    """
    t0 = time.time()
    out = sess.replay_counterfactual(abs_prefix + "_summary.json", decision_idx, action,
                                     n_rollouts=n_rollouts)
    outcomes = out.get("outcomes") or {}
    k = int(outcomes.get("win", 0))
    losses = int(outcomes.get("loss", 0))
    ties = int(outcomes.get("tie", 0))
    n = k + losses + ties
    total = int(out.get("n_rollouts") or sum(outcomes.values()))
    return {
        "n": n, "k": k, "n_tie": ties, "n_timeout": max(0, total - n),
        "outcomes": {str(a): int(b) for a, b in outcomes.items()},
        "opponent_source": out.get("opponent_source"),
        "seconds": round(time.time() - t0, 2),
    }


def _label_task(task: dict) -> dict:
    """Executed in a worker process. Never raises: a failure is a reported row, not a dead pool."""
    try:
        sess = _session_for(task["abs_prefix"])
        res = label_one(sess, task["abs_prefix"], task["decision_idx"],
                        task["action"], task["n_rollouts"])
        res["ok"] = True
    except Exception as exc:                                            # noqa: BLE001
        res = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
               "n": 0, "k": 0, "n_timeout": task["n_rollouts"], "seconds": 0.0}
    res["key"] = task["key"]
    return res


def build_row(c: Candidate, res: dict, *, subject_ckpt: str, sampler_version: str, seed: int,
              inline_obs: bool, models_root: str) -> Optional[HarvestRow]:
    """Turn a candidate plus its rollout result into a schema row, or ``None`` if nothing
    adjudicated (a row with ``n_rollouts == 0`` is not a label and the schema refuses it)."""
    if not res.get("ok") or res["n"] <= 0:
        return None
    npz_path = c.abs_prefix + "_states.npz"
    with np.load(npz_path) as z:
        obs = np.asarray(z["obs"], dtype=np.float32)[c.decision_idx]
    return HarvestRow(
        run=c.run,
        battle_tag=c.battle_tag,
        decision_idx=c.decision_idx,
        turn=c.turn,
        n_rollouts=int(res["n"]),
        n_wins=int(res["k"]),
        phi_head=c.phi_head,
        beta_evidence=c.beta_evidence,
        beta_mean=c.beta_mean,
        priority=round(float(c.priority), 6),
        provenance={
            "opponent": c.opponent,
            "opponent_source": res.get("opponent_source"),
            "recorded_outcome": c.outcome,
            "recorded_phi": round(c.recorded_phi, 6),
            "battle_turns": c.battle_turns,
            "is_cap": c.is_cap,
            "is_drag": c.is_drag,
            "faints_tail20": c.faints_tail,
            "action": c.action,
            "subject_ckpt": os.path.relpath(subject_ckpt, models_root)
                            if subject_ckpt.startswith(models_root) else subject_ckpt,
            "sampler_version": sampler_version,
            "seed": seed,
            "n_timeout": int(res.get("n_timeout", 0)),
            # Adjudicated but not a win — inside `n_rollouts`, unlike `n_timeout`. Recorded so a
            # label's composition is auditable without re-running it (see `label_one`).
            "n_tie": int(res.get("n_tie", 0)),
            "outcomes": res.get("outcomes", {}),
            "label_seconds": res.get("seconds"),
        },
        obs_npz=os.path.relpath(npz_path, models_root),
        obs_sha1=obs_digest(obs),
        obs_inline=obs_b64(obs) if inline_obs else None,
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def render_manifest(m: dict) -> str:
    """The human-readable sibling of the shards. Every harvest ships one; a directory of gzipped
    JSONL with no statement of what selected it is an artifact nobody can audit later."""
    L: List[str] = []
    a = L.append
    a(f"# Harvest — {m['sampler_version']}")
    a("")
    a(f"**Produced** {m['finished_iso']} · **subject** `{m['subject_ckpt']}` · "
      f"**seed** {m['seed']} · **impl** `{m['impl']}`")
    a("")
    a("## Supply")
    a("")
    a("| quantity | n |")
    a("|---|---:|")
    for k in ("runs_scanned", "candidate_battles", "candidate_decisions",
              "cap_battles_replayable", "long_loss_battles", "drag_battles", "holdout_battles",
              "selected_states", "labelled_rows", "rejected_rows"):
        if k in m["supply"]:
            a(f"| {k.replace('_', ' ')} | {m['supply'][k]} |")
    a("")
    a("## Labels")
    a("")
    a(f"- requested rollouts per state: **{m['rollouts']}**")
    a(f"- adjudicated rollouts: **{m['labels']['total_adjudicated']}** "
      f"(mean {m['labels']['mean_n']:.1f}/state)")
    a(f"- timeout bucket: **{m['labels']['total_timeout']}** "
      f"({m['labels']['timeout_frac']:.2%} of attempted) — never scored as losses")
    a(f"- mean MC label k/n: **{m['labels']['mean_label']:.4f}**; "
      f"mean subject phi: **{m['labels']['mean_phi']:.4f}**")
    a(f"- mean |phi - k/n|: **{m['labels']['mean_abs_gap']:.4f}**")
    a(f"- wall clock: **{m['labels']['seconds'] / 60:.1f} min** over {m['workers']} worker(s)")
    a("")
    a("## Selection")
    a("")
    share = (f"{m['drag_frac']:.0%}" if isinstance(m.get("drag_frac"), float) else "?")
    a(f"Mode `{m['priority_mode']}`, weights gap={W_GAP} uncertainty={W_UNCERTAINTY} "
      f"drag={W_DRAG}, max {m['max_per_battle']} states/battle, drag share capped at {share}.")
    if m.get("finalized_from_shards"):
        a("")
        a("⚠️ **FINALIZED FROM SHARDS after an interrupted run** — the sampler settings above are "
          "partly unrecoverable (`?`). The LABEL statistics are exact, recomputed from the rows.")
    comp = m.get("composition", {})
    if comp:
        a("")
        a(f"Composition: **{comp['drag_states']} doomed-tail** / **{comp['general_states']} "
          f"general late-game** states over {comp['battles']} battles; "
          f"{comp['from_won_battles']} came from battles that were WON. The general stratum is "
          f"the negative control — a head that learned 'late means lost' scores worse on it, and "
          f"the meter reports long-win reading beside the detection rate for exactly that reason.")
    a("")
    a(f"Evidential head on subject: **{'yes' if m['have_evidential'] else 'NO — columns omitted'}**.")
    a("")
    a("## Holdout")
    a("")
    a(f"{m['supply'].get('holdout_battles', 0)} battles held out at the BATTLE level from the DOOMED-TAIL "
      f"population (cap endings + long losses), STRATIFIED BY CLASS so the scarce cap class reaches "
      f"both arms, and listed in `holdout.json`. No held-out battle contributed a single labelled "
      f"state — the exclusion is applied before ranking, so no later slicing can readmit one.")
    a("")
    a(f"⚠️ Cap endings are scarce because a record with no terminal `forcelose` cannot be replayed "
      f"at all: `cap_record_unterminated` = **{m['skipped'].get('cap_record_unterminated', 0)}** "
      f"battles skipped here (unknown on a finalized run). That is the rust `sim_bridge` recorder gap fixed on 2026-08-24, so "
      f"the shortage shrinks with every post-fix run.")
    a("")
    a("## Frame census (what was skipped, and why)")
    a("")
    a("| reason | n |")
    a("|---|---:|")
    for k, v in sorted(m["skipped"].items(), key=lambda kv: -kv[1]):
        a(f"| `{k}` | {v} |")
    a("")
    a("## Caveats")
    a("")
    for c in m["caveats"]:
        a(f"- {c}")
    a("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def finalize(out_dir: str) -> int:
    """Rebuild ``manifest.json`` / ``manifest.md`` from the shards of an INTERRUPTED harvest.

    Earned by an observed failure: a 240-state pilot was killed externally at ~200 states, and
    because the manifest is written last, **10 shards of perfectly good labels were left with
    nothing describing them**. The shard flush had done its job; the artifact contract had not.

    A harvest is hours long, so being interrupted is a normal event rather than an exception, and
    an artifact that is only self-describing when the job runs to completion is self-describing
    exactly when you need it least. This reads the rows back, recomputes the label statistics from
    them, and marks the result ``stopped_early`` + ``finalized_from_shards`` so a partial harvest
    can never be mistaken for a complete one.
    """
    from agents.training.harvest_schema import read_dir

    rows = read_dir(out_dir)
    if not rows:
        print(f"harvest --finalize: no shards under {out_dir}", file=sys.stderr)
        return 2
    hpath = os.path.join(out_dir, "holdout.json")
    holdout = {}
    if os.path.exists(hpath):
        with open(hpath) as fh:
            holdout = json.load(fh)

    n = np.array([r["n_rollouts"] for r in rows], dtype=float)
    to = np.array([r["provenance"].get("n_timeout", 0) for r in rows], dtype=float)
    lab = np.array([r["n_wins"] / r["n_rollouts"] for r in rows], dtype=float)
    phi = np.array([r["phi_head"] if r["phi_head"] is not None else np.nan for r in rows])
    pops = holdout.get("meter_population") or {}
    prov0 = rows[0]["provenance"]
    manifest = {
        "sampler_version": prov0.get("sampler_version", SAMPLER_VERSION),
        "schema": 1,
        "finished_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "subject_ckpt": prov0.get("subject_ckpt", holdout.get("subject_ckpt", "?")),
        "models_root": holdout.get("models_root", "?"),
        "runs": sorted({r["run"] for r in rows}),
        "seed": prov0.get("seed", 0), "impl": "?", "workers": "?",
        "rollouts": int(n.max()) if len(n) else 0,
        "min_turn": int(min(r["turn"] for r in rows)),
        "priority_mode": "?", "max_per_battle": MAX_PER_BATTLE,
        "drag_frac": None, "general_win_frac": None,
        "have_evidential": rows[0].get("beta_evidence") is not None,
        "stopped_early": True,
        "finalized_from_shards": True,
        "composition": {
            "drag_states": int(sum(1 for r in rows if r["provenance"].get("is_drag"))),
            "general_states": int(sum(1 for r in rows if not r["provenance"].get("is_drag"))),
            "from_won_battles": int(sum(1 for r in rows
                                        if r["provenance"].get("recorded_outcome") == "win")),
            "battles": len({r["battle_tag"] for r in rows}),
        },
        "harvested_battles": sorted({r["battle_tag"] for r in rows}),
        "supply": {
            "runs_scanned": len({r["run"] for r in rows}),
            "cap_battles_replayable": len(pops.get("cap", [])),
            "long_loss_battles": len(pops.get("long_loss", [])),
            "holdout_battles": len(holdout.get("holdout_battles", [])),
            "labelled_rows": len(rows), "rejected_rows": 0,
        },
        "labels": {
            "total_adjudicated": int(n.sum()), "total_timeout": int(to.sum()),
            "timeout_frac": float(to.sum() / max(1.0, n.sum() + to.sum())),
            "mean_n": float(n.mean()), "mean_label": float(lab.mean()),
            "mean_phi": float(np.nanmean(phi)),
            "mean_abs_gap": float(np.nanmean(np.abs(phi - lab))),
            "seconds": float(sum(r["provenance"].get("label_seconds") or 0 for r in rows)),
        },
        "skipped": {},
        "caveats": [
            "⚠️ FINALIZED FROM SHARDS — this harvest did not run to completion, so the frame "
            "census and the sampler settings are NOT recoverable and are reported as '?'. The "
            "label statistics ARE exact: they are recomputed from the rows themselves.",
            "A timeout is excluded from both numerator and denominator and counted separately.",
        ],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    with open(os.path.join(out_dir, "manifest.md"), "w") as fh:
        fh.write(render_manifest(manifest))
    print(f"harvest --finalize: rebuilt manifest from {len(rows)} rows in {out_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.harvest",
        description="Harvest stall-tail win-probability labels by mid-game re-seed multi-rollout.")
    p.add_argument("--subject", default=None,
                   help="the checkpoint whose head is being repaired: it scores every candidate "
                        "AND plays the trainee side of every rollout")
    p.add_argument("--models-root", default=None,
                   help="the run archive (default: utils.paths.main_models_dir())")
    p.add_argument("--runs", nargs="*", default=None,
                   help="run names to mine (default: every CURRENT-ARCH run in the archive)")
    p.add_argument("--out", default=None,
                   help="output directory (default: <harvest_dir>/<subject stem>_<timestamp>)")
    p.add_argument("--states", type=int, default=300, help="how many states to label")
    p.add_argument("--rollouts", type=int, default=32, help="R — rollouts per state")
    p.add_argument("--min-turn", type=int, default=60,
                   help="the mid-game floor; cap games are swept whole regardless")
    p.add_argument("--priority", default="blend",
                   choices=("blend", "gap", "evidence", "drag", "random"))
    p.add_argument("--holdout-frac", type=float, default=0.35,
                   help="fraction of the METER population held out at the battle level")
    p.add_argument("--max-per-battle", type=int, default=MAX_PER_BATTLE)
    p.add_argument("--drag-frac", type=float, default=0.6,
                   help="cap on the doomed-tail share of the sample; the rest is drawn from the "
                        "general late-game population as the negative control (see select())")
    p.add_argument("--tail-frac", type=float, default=0.5,
                   help="share of the DOOMED budget reserved for a battle's last TAIL_K decisions "
                        "— the region the meter reads. Measured: without it the harvest tops out "
                        "at turn 152 while 29%% of meter-eval turns are above that")
    p.add_argument("--general-win-frac", type=float, default=0.5,
                   help="outcome balance INSIDE the general stratum. Without it the gap term "
                        "selects won battles out of existence (measured: 1 of 300)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--impl", default="rust", choices=("rust", "node"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inline-obs", action="store_true",
                   help="embed each obs in its row (self-contained, ~13 KB/row before gzip)")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="wall-clock budget for the LABELING phase; on expiry the harvest keeps "
                        "every label it bought and records `stopped_early` in the manifest")
    p.add_argument("--shard-every", type=int, default=50,
                   help="flush a shard every N labels, so an interrupted harvest keeps its work")
    p.add_argument("--finalize", metavar="DIR", default=None,
                   help="rebuild manifest.json/.md from the shards of an INTERRUPTED harvest and "
                        "exit; the label statistics are recomputed exactly from the rows")
    p.add_argument("--dry-run", action="store_true",
                   help="build + score + select and write the manifest, but buy no labels")
    return p


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    args = build_parser().parse_args(argv)
    if args.finalize:
        return finalize(args.finalize)

    from utils.paths import harvest_dir, main_models_dir, models_skip_reason

    models_root = args.models_root or (str(main_models_dir()) if main_models_dir() else None)
    if not models_root:
        print(f"harvest: {models_skip_reason()}", file=sys.stderr)
        return 2
    if not args.subject:
        print("harvest: --subject is required (except with --finalize)", file=sys.stderr)
        return 2
    subject = os.path.abspath(args.subject)
    if not os.path.exists(subject):
        print(f"harvest: no such checkpoint: {subject}", file=sys.stderr)
        return 2

    runs = args.runs or current_arch_runs(models_root)
    if not runs:
        print("harvest: no current-arch runs in the archive — nothing to mine", file=sys.stderr)
        return 2

    stem = os.path.basename(os.path.dirname(subject)) or "subject"
    out_dir = args.out or os.path.join(str(harvest_dir(create=True)),
                                       f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    print(f"harvest: {len(runs)} current-arch run(s), subject={os.path.relpath(subject, models_root)}",
          flush=True)
    cands, skipped = build_candidates(models_root, runs, min_turn=args.min_turn)
    if not cands:
        print(f"harvest: no labelable decisions — skipped {dict(skipped)}", file=sys.stderr)
        return 2
    battles = {c.battle_tag for c in cands}
    caps = {c.battle_tag for c in cands if c.is_cap}
    pools = meter_battles(cands)
    drags = {t for v in pools.values() for t in v}
    print(f"  frame: {len(cands)} decisions / {len(battles)} battles "
          f"(doomed tails: " + ", ".join(f"{len(v)} {k}" for k, v in sorted(pools.items()))
          + ")", flush=True)

    holdout = battle_holdout(cands, args.holdout_frac, args.seed)
    print(f"  holdout: {len(holdout)} battles reserved for the meter (never harvested)", flush=True)

    score_info = score_candidates(subject, cands, models_root=models_root)
    chosen = select(cands, args.states, mode=args.priority, seed=args.seed,
                    max_per_battle=args.max_per_battle, drag_frac=args.drag_frac,
                    general_win_frac=args.general_win_frac, tail_frac=args.tail_frac,
                    exclude_battles=holdout)
    n_drag = sum(1 for c in chosen if c.is_drag)
    n_win = sum(1 for c in chosen if c.outcome == "win")
    print(f"  selected {len(chosen)} states over {len({c.battle_tag for c in chosen})} battles "
          f"(priority={args.priority}, mean {np.mean([c.priority for c in chosen]):.3f}; "
          f"{n_drag} drag / {len(chosen) - n_drag} general, {n_win} from won battles)",
          flush=True)

    with open(os.path.join(out_dir, "holdout.json"), "w") as fh:
        json.dump({"holdout_battles": holdout, "seed": args.seed,
                   "frac": args.holdout_frac, "meter_population": {k: v for k, v in sorted(pools.items())},
                   "harvested_battles": sorted({c.battle_tag for c in chosen}),
                   "models_root": models_root,
                   "subject_ckpt": os.path.relpath(subject, models_root),
                   "sampler_version": SAMPLER_VERSION}, fh, indent=1)

    rows: List[HarvestRow] = []
    stats = {"total_adjudicated": 0, "total_timeout": 0, "attempted": 0,
             "rejected": 0, "seconds": 0.0, "stopped_early": False,
             "planned_states": len(chosen)}
    if not args.dry_run and chosen:
        rows, stats = _run_labeling(chosen, args, subject=subject, models_root=models_root,
                                    out_dir=out_dir)

    labels = _label_stats(rows, stats, args.rollouts)
    manifest = {
        "sampler_version": SAMPLER_VERSION,
        "schema": 1,
        "finished_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "subject_ckpt": os.path.relpath(subject, models_root),
        "models_root": models_root,
        "runs": runs, "seed": args.seed, "impl": args.impl, "workers": args.workers,
        "rollouts": args.rollouts, "min_turn": args.min_turn,
        "priority_mode": args.priority, "max_per_battle": args.max_per_battle,
        "drag_frac": args.drag_frac, "general_win_frac": args.general_win_frac,
        "tail_frac": args.tail_frac,
        "composition": {
            "drag_states": int(sum(1 for c in chosen if c.is_drag)),
            "general_states": int(sum(1 for c in chosen if not c.is_drag)),
            "from_won_battles": int(sum(1 for c in chosen if c.outcome == "win")),
            "tail_states": int(sum(1 for c in chosen if c.is_tail)),
            "max_turn": int(max(c.turn for c in chosen)) if chosen else 0,
            "battles": len({c.battle_tag for c in chosen}),
        },
        "harvested_battles": sorted({c.battle_tag for c in chosen}),
        "have_evidential": score_info["have_evidential"],
        "score": score_info,
        "supply": {
            "runs_scanned": len(runs),
            "candidate_battles": len(battles),
            "candidate_decisions": len(cands),
            "cap_battles": len(caps),
            "drag_battles": len(drags),
            "cap_records_unterminated": int(skipped.get("cap_record_unterminated", 0)),
            "cap_battles_replayable": len(pools.get("cap", [])),
            "long_loss_battles": len(pools.get("long_loss", [])),
            "holdout_battles": len(holdout),
            "selected_states": len(chosen),
            "labelled_rows": len(rows),
            "rejected_rows": stats["rejected"],
        },
        "labels": labels,
        "stopped_early": stats.get("stopped_early", False),
        "planned_states": stats.get("planned_states", len(chosen)),
        "skipped": dict(skipped),
        "caveats": [
            "`priority`'s gap term uses the battle's REALIZED 0/1 outcome as a proxy for the "
            "state's true value — one sample, deliberately noisy. It is a selection rule, not a "
            "measurement; it travels in every row so a consumer can re-weight or ablate it.",
            "Rollouts play the SUBJECT as the trainee against the reloaded recorded opponent. A "
            "bot is rebuilt exactly; a pool sentinel is reloaded from its pinned snapshot; "
            "anything unresolved falls back to the subject standing in for itself, which is "
            "flagged per row in `provenance.opponent_source`.",
            "The recorded battle prefix was played by whichever policy wrote the trace, so a "
            "harvested state may be off-distribution for the subject. That is intended: the "
            "over-confident tail is exactly the region the subject rarely reaches on-policy.",
            "A timeout is excluded from both numerator and denominator and counted separately. A "
            "harvest whose timeout fraction is large was measured on a busy box and its labels "
            "are thinner than its `--rollouts` suggests.",
        ],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    with open(os.path.join(out_dir, "manifest.md"), "w") as fh:
        fh.write(render_manifest(manifest))

    print(f"harvest: {len(rows)} rows → {out_dir} in {(time.time() - t_start) / 60:.1f} min",
          flush=True)
    return 0


def _label_stats(rows: Sequence[HarvestRow], stats: dict, rollouts: int) -> dict:
    n = len(rows)
    if not n:
        return {"total_adjudicated": stats["total_adjudicated"],
                "total_timeout": stats["total_timeout"],
                "timeout_frac": 0.0, "mean_n": 0.0, "mean_label": float("nan"),
                "mean_phi": float("nan"), "mean_abs_gap": float("nan"),
                "seconds": stats["seconds"]}
    labels = np.array([r.label for r in rows])
    phis = np.array([r.phi_head if r.phi_head is not None else np.nan for r in rows])
    attempted = max(1, stats["attempted"])
    return {
        "total_adjudicated": stats["total_adjudicated"],
        "total_timeout": stats["total_timeout"],
        "timeout_frac": stats["total_timeout"] / attempted,
        "mean_n": float(np.mean([r.n_rollouts for r in rows])),
        "mean_label": float(np.mean(labels)),
        "mean_phi": float(np.nanmean(phis)),
        "mean_abs_gap": float(np.nanmean(np.abs(phis - labels))),
        "seconds": round(stats["seconds"], 1),
        "requested_rollouts": rollouts,
    }


def _run_labeling(chosen: Sequence[Candidate], args, *, subject: str, models_root: str,
                  out_dir: str) -> "Tuple[List[HarvestRow], dict]":
    """Buy the labels, in ``--workers`` processes, flushing a shard every ``--shard-every``.

    Tasks are ordered by battle so a worker reuses one ``ProbeSession`` (and therefore one warm
    bridge child) across a battle's states instead of paying the open cost per label.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    by_key = {f"{c.battle_tag}#{c.decision_idx}": c for c in chosen}
    tasks = [{"key": k, "abs_prefix": c.abs_prefix, "decision_idx": c.decision_idx,
              "action": c.action, "n_rollouts": args.rollouts}
             for k, c in sorted(by_key.items(), key=lambda kv: (kv[1].battle_tag,
                                                                kv[1].decision_idx))]
    rows: List[HarvestRow] = []
    stats = {"total_adjudicated": 0, "total_timeout": 0, "attempted": 0,
             "rejected": 0, "seconds": 0.0}
    t0 = time.time()
    shard, pending = 0, []
    errors: Counter = Counter()

    def _flush() -> None:
        nonlocal shard, pending
        if not pending:
            return
        write_rows(pending, os.path.join(out_dir, f"labels_{shard:04d}.jsonl.gz"))
        shard += 1
        pending = []

    ctx = mp.get_context("spawn")
    n_workers = max(1, int(args.workers))
    budget_s = float(args.max_minutes) * 60.0 if args.max_minutes else None
    stopped_early = False
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_worker_init,
                             initargs=(subject, args.impl)) as pool:
        # `submit` + `as_completed` rather than `map`, because `map` yields IN ORDER: one slow
        # state (a turn-100 drag rolled 32x to a 250-turn cap is minutes) would block every
        # finished result behind it, starving both the progress readout and the shard flush that
        # makes an interrupted harvest keep its work.
        futures = {pool.submit(_label_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            c = by_key[res["key"]]
            stats["attempted"] += args.rollouts
            stats["total_adjudicated"] += res.get("n", 0)
            stats["total_timeout"] += res.get("n_timeout", 0)
            if not res.get("ok"):
                errors[res.get("error", "?").split(":")[0]] += 1
            row = build_row(c, res, subject_ckpt=subject, sampler_version=SAMPLER_VERSION,
                            seed=args.seed, inline_obs=args.inline_obs, models_root=models_root)
            if row is None:
                stats["rejected"] += 1
            else:
                rows.append(row)
                pending.append(row)
            if len(pending) >= args.shard_every:
                _flush()
            if i % 10 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  [{i}/{len(tasks)}] {len(rows)} rows · {el / 60:.1f} min · "
                      f"{el / max(1, i):.1f} s/state · eta "
                      f"{(len(tasks) - i) * el / max(1, i) / 60:.0f} min", flush=True)
            if budget_s is not None and (time.time() - t0) > budget_s:
                # A time-boxed harvest stops on the clock and KEEPS what it bought. The manifest
                # records `stopped_early`, so a short run can never be mistaken for a complete one.
                stopped_early = True
                print(f"  --max-minutes reached at {i}/{len(tasks)} — cancelling the rest and "
                      f"writing what is labelled", flush=True)
                for f in futures:
                    f.cancel()
                break
    _flush()
    stats["stopped_early"] = stopped_early
    stats["attempted_states"] = len(rows) + stats["rejected"]
    stats["planned_states"] = len(tasks)
    stats["seconds"] = time.time() - t0
    if errors:
        print(f"  label errors: {dict(errors)}", flush=True)
    return rows, stats


if __name__ == "__main__":
    raise SystemExit(main())
