"""The α/β OPPONENT-INTENT read: what the model expected the opponent to do."""

from __future__ import annotations

from main.prober.engine.util import _norm_species
from main.prober.engine.views import (BELIEF_NAME_CAVEAT, OppIntentCandidate, OppIntentOption,
    OppIntentView)


# The name `α` uses for "none of these moves" — set by `opp_intent.render_alpha`, matched here.
SWITCH_OPTION = "SWITCH"


def _opp_actual_action(inv: dict) -> "tuple[str | None, bool]":
    """`(move_id, was_a_voluntary_switch)` for what the OPPONENT actually did this turn.

    The recorder writes one string and it carries more than one shape: a bare move id
    (`drillpeck`), a move whose result forced a replacement (`dragonclaw → skarmory_sent_in` — that
    is still a MOVE, the send-in is its consequence), a voluntary pivot (`switched_to:blissey`),
    and a post-faint replacement (`swampert_sent_in`, which is not a chosen action at all).
    """
    action = str(((inv.get("outcome") or {}).get("opp") or {}).get("action") or "").strip()
    if not action or action == "none":
        return None, False
    if action.startswith("switched_to:"):
        return None, True
    head = action.split("→")[0].strip()          # drop a forced-replacement consequence
    if not head or head.endswith("_sent_in"):
        return None, False                        # a replacement is not a choice
    return head, False


def _matches_move(display_name: str, move_id: str) -> bool:
    """Does an α option name the move the opponent used? Compared on normalized ids, because α
    carries display names (`Drill Peck`) and the recorder an id (`drillpeck`).

    Hidden Power needs its own rule in ONE direction only: α names the model's BELIEVED type
    (`Hidden Power Grass`) while an opponent's un-revealed HP is recorded bare (`hiddenpower`), so a
    bare id matches any typed HP option. The reverse is not allowed — a specific recorded type must
    not match a different believed one.
    """
    a, b = _norm_species(display_name), _norm_species(move_id)
    if not a or not b:
        return False
    if a == b:
        return True
    return b == "hiddenpower" and a.startswith("hiddenpower")


def _move_display(move_id: str) -> str:
    """A move id as the card should show it when α never listed it (`drillpeck` → `Drill Peck`),
    via the data facade so the spelling matches everywhere else. Falls back to the raw id."""
    try:
        from agents.gen3_data import moves as _moves
        name = (_moves.raw().get(move_id) or {}).get("name")
        if name:
            return str(name)
    except Exception:                     # noqa: BLE001 — a display name is never worth raising for
        pass
    return move_id


def _beta_candidate(e: dict) -> OppIntentCandidate:
    """One recorded `β` row → a candidate, with the naming PROVENANCE resolved once.

    Names are NOT re-derived here, deliberately: an old trace baked the posterior name and the
    board it should have been read from is not in the trace at all, so the only honest repair at
    read time is to attach `BELIEF_NAME_CAVEAT` — never to invent a replacement name."""
    species = str(e["species"]) if e.get("species") else None
    revealed = bool(e.get("revealed"))
    return OppIntentCandidate(
        slot=int(e.get("slot", -1)), p=float(e.get("p", 0.0)), species=species,
        revealed=revealed,
        caveat=(BELIEF_NAME_CAVEAT if (species is not None and not revealed) else None))


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
    did_move, did_switch = _opp_actual_action(inv)
    alpha, matched = [], False
    for entry in raw.get("alpha") or []:
        name = str(entry.get("name", "?"))
        is_switch = name == SWITCH_OPTION
        hit = (did_switch and is_switch) or (
            not is_switch and did_move is not None and _matches_move(name, did_move))
        matched = matched or hit
        alpha.append(OppIntentOption(name=name, p=float(entry.get("p", 0.0)),
                                     is_switch=is_switch, was_actual=hit))
    if not alpha:
        return None
    beta = tuple(_beta_candidate(e) for e in (raw.get("beta") or []))
    switch = next((o.p for o in alpha if o.is_switch), None)
    # What they actually picked, named the way α names things. When α listed it, reuse ITS spelling
    # so the card reads "Drill Peck" in both places rather than "Drill Peck" and "drillpeck".
    if did_switch:
        actual = SWITCH_OPTION
    elif did_move is not None:
        actual = next((o.name for o in alpha if o.was_actual), _move_display(did_move))
    else:
        actual = None
    return OppIntentView(alpha=tuple(alpha), beta=beta, top=alpha[0], switch_p=switch,
                         actual=actual, actual_unlisted=bool(actual and not matched))


def opp_intent_text(view: "OppIntentView | None", top_n: int = 3) -> str:
    """The one-line rendering of `α` (+ `β` when a switch is expected) — the shared vocabulary, so
    the TUI, the JSON CLI and the web replay all say the same sentence about the same numbers.

    `expects fireblast 41% · SWITCH 22% · icebeam 12%`, and when SWITCH leads, the `β` follow-up:
    `expects SWITCH 52% · fireblast 20% → in: blissey 61%`. Empty string on `None`.

    When the `β` name is a POSTERIOR DECODE rather than a mon read off the board, the sentence
    carries `BELIEF_NAME_CAVEAT` — the sentence is the shared vocabulary, so the qualifier has to
    ride IN it or the surfaces that print only the sentence lose it silently."""
    if view is None or not view.alpha:
        return ""
    parts = [f"{o.name} {o.p * 100:.0f}%" for o in view.alpha[:top_n]]
    text = "expects " + " · ".join(parts)
    if view.beta and view.top is not None and view.top.is_switch:
        best = view.beta[0]
        who = best.species or f"slot {best.slot}"
        text += f" → in: {who} {best.p * 100:.0f}%"
        if best.caveat:
            text += f" · {best.caveat}"
    return text


def awareness_text(aw: "dict | None") -> str:
    """The one-line rendering of a battle's 'did it KNOW?' verdict (`main/prober/awareness.py`),
    so the CLI and the web replay say the same sentence about the same fold.

    `never saw it coming — P(win) never fell below 50% to the end` ·
    `knew by turn 34 — 12 turns of warning` · and, when the stall signature fired, the clause that
    names it: `· stall signature: 41% tail mass at turn 28 while the mean still read positive`.
    Empty string on `None` (no dist head / fewer than 2 recorded distributions).

    Phrased in P(WIN) throughout, matching the strip and the win-prob head beside it: one direction
    on one card, where higher always reads as better. The underlying test is unchanged (it is
    defined on `p_loss > 0.5` sustained) — `P(win) below 50%` is the same crossing said the other
    way up."""
    if not aw:
        return ""
    knew, lead = aw.get("knew_by_turn"), aw.get("lead_time")
    if aw.get("blind_loss"):
        text = "never saw it coming — P(win) never fell below 50% to the end"
    elif knew is None:
        # Not a loss, and it never sustained: the ordinary shape of a win.
        text = "P(win) never fell below 50% to the end"
    elif aw.get("outcome") == "loss":
        turns = "turn" if lead == 1 else "turns"
        text = f"knew by turn {knew} — {lead} {turns} of warning"
    else:
        text = f"P(win) held below 50% from turn {knew} to the end"
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
