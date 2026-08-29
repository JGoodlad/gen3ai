"""`--search-teacher-mode` — the ONE place that knows which teacher a run is running.

Two modes today, and the dispatch lives here rather than in each worker because there are THREE
call sites (the per-cycle worker, the persistent worker, and the callback's own selection) and a
mode string validated in three places is a mode string that will eventually mean three things.

| mode | selection | production | asks |
|---|---|---|---|
| `crater` (default) | `selection.select_candidates` — value craters, falsify-gated to reducible mistakes | `produce.produce_correction` — a depth-2 beam over the CRITIC, Wilson-gated on rollout-confirmed win rate | *where did the model lose the most value, and is there a strictly better LINE?* |
| `winprob_oneply` | `winprob_oneply.select_winprob_candidates` — CONTESTED decisions (the H rule), model-free | `winprob_oneply.produce_winprob_correction` — one-ply win-prob ranking, margin floor, paired-rollout confirmation | *at a decision the head calls contested, does a one-ply read prefer something else by a margin that survives confirmation?* |

Both produce the SAME `Correction` record, so everything downstream of the ring buffer — the shard
format, `CorrectionBuffer`, `_searchteacher_loss`, `--search-teacher-coef` — is untouched and cannot
tell them apart. That is the point: `winprob_oneply` is a new *supply* of targets, not a new pipeline.

ai_v12 routes 2 + 3 — `designs/ai_v12/design_winprob_behavior_coupling.md`. **Nothing has run
`winprob_oneply`; no arm is registered.** `crater` is the default and is byte-identical to the
behaviour before this module existed.
"""

from __future__ import annotations

from typing import List

MODE_CRATER = "crater"
MODE_WINPROB_ONEPLY = "winprob_oneply"
#: The legal set, and the argparse `choices`. `crater` is FIRST because it is the default and the
#: historical behaviour.
TEACHER_MODES = (MODE_CRATER, MODE_WINPROB_ONEPLY)


def validate_mode(mode: str) -> str:
    """Normalize + validate. An unknown mode RAISES rather than falling back to `crater`: a silent
    fall-back would run a different teacher than the operator asked for, at the same coefficient,
    with nothing in the logs to say so."""
    m = str(mode or MODE_CRATER)
    if m not in TEACHER_MODES:
        raise ValueError(f"unknown --search-teacher-mode {m!r}; expected one of {TEACHER_MODES}")
    return m


def select_for_mode(mode: str, run_dir: str, *, budget: int, scan_limit: int,
                    falsify_gate: bool, window: int, wp_band: float,
                    step=None, session=None) -> List:
    """Dispatch to the mode's candidate selector. Both return ``List[Candidate]``.

    The keyword set is the UNION of both selectors' knobs, so a caller passes its whole config and
    the dispatcher drops what the chosen mode does not use. That is deliberate: the alternative is
    each call site knowing which knobs belong to which mode, which is the drift this module exists
    to prevent.
    """
    m = validate_mode(mode)
    if m == MODE_WINPROB_ONEPLY:
        from agents.training.teacher.winprob_oneply import select_winprob_candidates
        return select_winprob_candidates(
            run_dir, budget=budget, band=wp_band, step=step, session=session)
    from agents.training.teacher.selection import select_candidates
    return select_candidates(run_dir, budget=budget, scan_limit=scan_limit,
                             falsify_gate=falsify_gate, window=window, step=step)


def produce_for_mode(mode: str, session, candidate, *, opponent_ckpt, opponent_source,
                     confirm_rollouts: int, depth: int, beam: int, top_k: int,
                     margin_min: float, wp_margin: float, search_session=None,
                     build_pi_target: bool = False, opd_beta: float = 1.0):
    """Dispatch to the mode's producer. Both return ``(Correction | None, status)``.

    Note the two DIFFERENT margins, which is why they are two parameters and not one:
    ``margin_min`` gates the `crater` mode's Wilson lower bound against the played loss line (units:
    win RATE, floor 0.0), while ``wp_margin`` gates `winprob_oneply`'s one-ply Δφ (units: win
    PROBABILITY from the head, default 0.02). Collapsing them would silently re-purpose whichever
    value a run happened to set.
    """
    m = validate_mode(mode)
    if m == MODE_WINPROB_ONEPLY:
        from agents.training.teacher.winprob_oneply import produce_winprob_correction
        return produce_winprob_correction(
            session, candidate, opponent_ckpt=opponent_ckpt, opponent_source=opponent_source,
            margin_min=wp_margin, confirm_rollouts=confirm_rollouts)
    from agents.training.teacher.produce import produce_correction
    return produce_correction(
        session, candidate, opponent_ckpt=opponent_ckpt, opponent_source=opponent_source,
        confirm_rollouts=confirm_rollouts, depth=depth, beam=beam, top_k=top_k,
        margin_min=margin_min, search_session=search_session,
        build_pi_target=build_pi_target, opd_beta=opd_beta)
