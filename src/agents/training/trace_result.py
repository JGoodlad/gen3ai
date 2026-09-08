"""THE TRACE-RESULT VOCABULARY — the one declaration of what an eval trace's outcome can BE.

``gen3_trace_result_v2``.

WHY THIS MODULE EXISTS. Until 2026-09-07 an eval trace could only say WIN or LOSS. The recorder's
filename stem was ``win_``/``loss_``, and the eval quota had exactly two buckets, so:

* a **250-turn TIMEOUT** — which the training reward itself scores as a DRAW (`reward_manager`'s
  terminal fold pays ``draw_penalty``, detected by the turn count because the trainee FORFEITS at
  the cap and poke-env therefore reports ``lost=True``) — was written to disk as an ordinary LOSS,
  indistinguishable from a decisive one except by re-deriving ``meta.turns >= MAX_TURNS``; and
* a true **TIE** (the sim's ``|tie|``, ``won is None``) matched NEITHER quota branch and its
  buffered capture was silently dropped — no file, no count, nothing.

The consequence, recorded 2026-09-07: **"0 draws in every eval trace" was what the instrument
could express, not what happened** — the absence-is-not-a-zero class. 145,173 traces across the
whole archive, every one of them ``win_*`` or ``loss_*``.

THE VOCABULARY. Three buckets, and the bucket is what the filename prefix and ``meta.result``
both say:

    WIN · LOSS · DRAW

``DRAW`` carries a ``meta.draw_kind`` naming WHICH draw it was — ``"tie"`` or ``"timeout"``. The
two are distinguishable at the battle layer and are NOT the same event:

===========  =============================================  ==================================
draw_kind    how the battle layer reports it                what it means
===========  =============================================  ==================================
``timeout``  ``lost`` is true AND ``turn >= turn_cap``      the trainee forfeited at the cap
                                                            (``inference/player._handle_stall``
                                                            → ``ForfeitBattleOrder``); a STALL
``tie``      ``won``/``lost`` both falsy, ``finished``      the sim emitted ``|tie|``
                                                            (``abstract_battle.tied()`` leaves
                                                            ``_won`` None); genuinely drawn
===========  =============================================  ==================================

Keeping them apart is the point: a timeout is a POLICY failure mode (the stall rate G7 kills on),
a tie is a rules outcome. Collapsing them would rebuild the same absence the bucket exists to fix.

🚨 **AN UNKNOWN RESULT IS REFUSED, NEVER COERCED.** :func:`check_result` raises. A result string
this build does not know means the writer and the reader disagree about the vocabulary, and the
one thing that must not happen then is a quiet fallback to LOSS — that is exactly how the timeout
spent five months wearing a loss's clothes.

READING AN OLD TRACE. Every trace written before this landed carries no ``result_vocabulary`` key.
:func:`result_era` names the era from the meta block alone, and :data:`PRE_DRAW_BUCKET_NOTE` is the
one sentence a consumer prints beside any statistic computed over such a tree. A v1 trace stays
readable and is reported **as it was written** — nothing rewrites history — but a v1 ``LOSS``
cannot be split into "decisive" and "timeout" from the result alone (``meta.turns >= MAX_TURNS``
is the only handle, which is why G7 keys on the turn count), and a v1 tree contains no ties at
all because a tie was never persisted.

PURE STDLIB, NO TORCH, no import of the observation/training stack (the turn cap is a PARAMETER,
never imported here), and it lives in a namespace package — so the prober, the harvesters and the
gauge read the same declaration the recorder writes.
"""

from __future__ import annotations

#: The version tag written into every trace's ``meta.result_vocabulary``. Bump when the SET of
#: results changes, so a reader can tell a vocabulary it understands from one it does not.
RESULT_VOCABULARY = "gen3_trace_result_v2"

#: What a trace written before the draw bucket landed is called. Never written — only inferred,
#: by the ABSENCE of ``meta.result_vocabulary``.
RESULT_VOCABULARY_V1 = "gen3_trace_result_v1"

#: The key the vocabulary tag rides under, inside the summary's ``meta`` block.
VOCABULARY_KEY = "result_vocabulary"
#: The key naming WHICH draw a DRAW was. Present only on a DRAW.
DRAW_KIND_KEY = "draw_kind"

WIN = "WIN"
LOSS = "LOSS"
DRAW = "DRAW"

#: The whole vocabulary, in the order a report lists it.
RESULTS = (WIN, LOSS, DRAW)

DRAW_TIE = "tie"
DRAW_TIMEOUT = "timeout"
DRAW_KINDS = (DRAW_TIE, DRAW_TIMEOUT)

#: The filename prefix / filter token for each result. Lowercase, because that is what
#: ``trace_filename_stem`` has always emitted and what ``discovery._FNAME_RE`` inverts.
_PREFIX = {WIN: "win", LOSS: "loss", DRAW: "draw"}

#: The filter tokens, in report order — the single source for every ``--outcome`` choice list and
#: every web query pattern, so a surface cannot offer a bucket the recorder never writes (or miss
#: one it does).
OUTCOMES = tuple(_PREFIX[r] for r in RESULTS)

#: The one sentence a consumer prints beside a statistic computed over a pre-draw-bucket tree.
PRE_DRAW_BUCKET_NOTE = (
    "PRE-DRAW-BUCKET TRACES (gen3_trace_result_v1) — this tree was written before the DRAW bucket "
    "existed. Its LOSS traces mix decisive losses with 250-turn TIMEOUTS (which the training "
    "reward itself scores as draws) and cannot be separated by result alone; use "
    "meta.turns >= MAX_TURNS, the way the G7 kill clause does. Its TIES are not merely absent from "
    "the counts — a tie was never persisted at all, so a draw rate is NOT MEASURABLE here and a "
    "zero must never be read as one."
)


class UnknownTraceResult(ValueError):
    """A trace result string this build's vocabulary does not contain.

    A hard failure on purpose (the GIGO rule): a coerced unknown result is a silent
    misclassification, and the last one cost five months of invisible timeouts.
    """


def check_result(result: "str | None") -> str:
    """Return ``result`` if it is in the vocabulary; otherwise RAISE.

    The throwing guard the summary writer and every strict reader go through. Case is normalised
    (the on-disk form is upper) but nothing else is: an unrecognised token is refused, never
    mapped to the nearest neighbour.
    """
    token = str(result).strip().upper() if result is not None else ""
    if token not in RESULTS:
        raise UnknownTraceResult(
            f"unknown trace result {result!r} — the {RESULT_VOCABULARY} vocabulary is "
            f"{', '.join(RESULTS)}. A result outside it means the writer and this reader "
            f"disagree about the vocabulary; refusing rather than guessing."
        )
    return token


def check_draw_kind(kind: "str | None") -> "str | None":
    """``None`` (not a draw) or a known draw kind; anything else RAISES."""
    if kind is None:
        return None
    token = str(kind).strip().lower()
    if token not in DRAW_KINDS:
        raise UnknownTraceResult(
            f"unknown draw kind {kind!r} — expected one of {', '.join(DRAW_KINDS)}."
        )
    return token


def classify_result(*, won, lost, finished, turn, turn_cap: int) -> "tuple[str, str | None]":
    """``(result, draw_kind)`` for one finished battle.

    The ONE place the (won, lost, finished, turn) → bucket mapping is written. Callers pass the
    battle layer's own three flags plus the turn count and the forfeit deadline; the cap is a
    PARAMETER so this module stays free of the observation/training import graph (the producer
    passes ``StallConfig().threshold``, which IS ``MAX_TURNS`` — `gen3_deadline_clock_v1`).

    Order matters. A TIMEOUT is checked BEFORE the plain loss, because a timeout arrives wearing a
    loss's flags: the trainee forfeits at the cap, so poke-env reports ``lost=True``. That is the
    exact confusion this function exists to end.
    """
    if won:
        return WIN, None
    turn_val = int(turn or 0)
    if turn_val >= int(turn_cap):
        # At or past the forfeit deadline. Whether the layer calls it a loss (our forfeit) or a
        # tie (both sides out of time), the reward pays `draw_penalty` and so do we.
        return DRAW, DRAW_TIMEOUT
    if lost:
        return LOSS, None
    if finished:
        # Finished, not won, not lost, before the cap: the sim's `|tie|`.
        return DRAW, DRAW_TIE
    # Not finished. Reached only by a caller finalising an abandoned battle; a LOSS is what the
    # old two-bucket code produced for it and it is the honest read (we did not win).
    return LOSS, None


def outcome_prefix(result: str) -> str:
    """The lowercase filename / filter token for a result (``"win"``/``"loss"``/``"draw"``)."""
    return _PREFIX[check_result(result)]


def result_of(meta: "dict | None") -> "str | None":
    """The result recorded in a summary's ``meta`` block, checked; ``None`` when absent.

    Absent means absent — a summary with no ``result`` is not a loss.
    """
    if not isinstance(meta, dict):
        return None
    raw = meta.get("result")
    if raw is None or str(raw).strip() == "":
        return None
    return check_result(raw)


def draw_kind_of(meta: "dict | None") -> "str | None":
    """The recorded draw kind, checked; ``None`` when the trace is not a draw or predates the key."""
    if not isinstance(meta, dict):
        return None
    return check_draw_kind(meta.get(DRAW_KIND_KEY))


def result_era(meta: "dict | None") -> str:
    """Which vocabulary a trace was WRITTEN under — read from the meta block, never assumed.

    A trace carrying no ``result_vocabulary`` predates the draw bucket
    (:data:`RESULT_VOCABULARY_V1`); one carrying an unrecognised tag is returned VERBATIM so a
    reader reports "a vocabulary I do not know" rather than silently claiming it is the current
    one.
    """
    if not isinstance(meta, dict):
        return RESULT_VOCABULARY_V1
    tag = meta.get(VOCABULARY_KEY)
    if not isinstance(tag, str) or not tag.strip():
        return RESULT_VOCABULARY_V1
    return tag.strip()


def is_pre_draw_bucket(meta: "dict | None") -> bool:
    """True when this trace was written before the DRAW bucket existed."""
    return result_era(meta) == RESULT_VOCABULARY_V1


def era_note(metas) -> "str | None":
    """The one sentence to print beside a statistic over ``metas``, or ``None`` when every trace
    in the sample carries the current vocabulary.

    ``metas`` is any iterable of ``meta`` dicts (or of era strings). A MIXED tree — a run that
    restarted onto new code mid-flight — is the interesting case, and it gets the note too: the
    pre-bucket half is still unsplittable.
    """
    eras = {m if isinstance(m, str) else result_era(m) for m in metas}
    if not eras or eras == {RESULT_VOCABULARY}:
        return None
    if RESULT_VOCABULARY_V1 in eras:
        note = PRE_DRAW_BUCKET_NOTE
        if len(eras) > 1:
            note = ("MIXED RESULT VOCABULARIES in this sample "
                    f"({', '.join(sorted(eras))}). " + note)
        return note
    return ("UNKNOWN RESULT VOCABULARY in this sample "
            f"({', '.join(sorted(eras))}) — this build writes {RESULT_VOCABULARY}. Read every "
            "outcome count as a statement about a vocabulary this reader does not define.")
