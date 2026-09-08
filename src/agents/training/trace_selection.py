"""THE TRACE-SELECTION CONTRACT — the one declaration the recorder writes and every consumer reads.

`gen3_trace_selection_manifest_v1`.

WHY THIS MODULE EXISTS. Eval traces are written under a quota that PREFERS LOSSES (by design — the
prober is a loss-forensics tool), and until this shipped nothing in the trace tree recorded that.
So every consumer that averages over traces — the prober's `calibration` and `falsify_scan`, the
scaffolding gauge, and the win-prob baseline — silently inherited a loss-enriched sample. Measured
on `ai_v9_59_R2ACTION_0827` (2026-09-06): the captured slice's outcome rate is **0.46** against the
same cycles' own recorded **0.901 vs bots / 0.702 vs pool**, and reading the raw table first
inverts the verdict on the head under test (ECE 0.237/0.281 raw vs 0.025/0.035 corrected; skill
+0.071/−0.080 raw vs +0.336/+0.265 corrected).

The fix is not a correction factor, it is a RECORD: the per-cycle `eval_manifest.json` now states,
per opponent, how many battles were PLAYED, how many were WON, how many were TRACED and how many of
those traces were wins — plus the two derived capture rates and the rule in words. A consumer can
then reweight from the manifest alone instead of knowing to join `eval_results.jsonl` by hand.

🚨 **ABSENT IS "UNKNOWN", NEVER "UNIFORM".** A legacy manifest (or a cycle that crashed before
collecting) carries no `selection` block, and `read_selection` returns ``None`` there. Every
consumer must render that as SELECTION UNKNOWN and label its curve accordingly — treating a missing
record as an unbiased sample is exactly the defect this module exists to close, and it would be a
worse defect for being invisible.

PURE STDLIB, NO TORCH, and it lives in a namespace package (`agents/training/` has no
``__init__.py``), so the prober imports it without pulling in the training stack — which is what
lets the producer and every consumer read ONE declaration rather than three copies of it.
"""

from __future__ import annotations

# The version tag of the contract below. Bump when the recorder's rule changes SHAPE, so a
# consumer can tell a rule it understands from one it does not.
#
# 2 (2026-09-07) added the DRAW bucket: `battles_drawn` / `traces_drawn` / `capture_rate_draw`
# and the `draw_quota`. Purely ADDITIVE — every schema-1 key kept its name and its meaning — so
# schema 1 stays READABLE rather than being demoted to SELECTION UNKNOWN, which would have
# silently thrown away the recorded selection of every cycle written before today.
SELECTION_SCHEMA = 2

#: Every schema this build can read. A manifest at an OLDER known schema is read as it was
#: written (its draw keys simply absent); an UNKNOWN schema is SELECTION UNKNOWN, never guessed.
KNOWN_SELECTION_SCHEMAS = (1, 2)

#: The key the per-cycle manifest carries the per-opponent record under.
SELECTION_KEY = "selection"
#: The key carrying the rule in words.
SELECTION_RULE_KEY = "selection_rule"

#: What a consumer prints for a tree that records no selection. One string, so the prober, the
#: gauge and the CLI cannot describe the same state three different ways.
UNKNOWN_LABEL = (
    "SELECTION UNKNOWN — this trace tree records no capture quota, so the sample's win/loss mix "
    "is unknown and may be LOSS-ENRICHED (the recorder's quota prefers losses by design). Read "
    "this curve as a statement about the CAPTURED slice, not about the eval population."
)


def forensic_selection_rule(win_quota: int, loss_quota: int,
                            draw_quota: "int | None" = None) -> str:
    """The ONE sentence describing which battles get a forensic trace.

    Spelled out rather than named: a reader of a two-year-old trace tree has the recorder's
    constants nowhere to hand, and the numbers are what make the capture rates interpretable.

    ``draw_quota`` is ``None`` only for a caller reconstructing the PRE-DRAW-BUCKET rule; the live
    recorder always passes it, and the sentence then states where draws sit.
    """
    draws = (f"the first {draw_quota} DRAWS (a tie, or a 250-turn timeout — its OWN bucket, so a "
             f"stall storm cannot evict the decisive losses), "
             if draw_quota is not None else
             "NO DRAW BUCKET AT ALL — a timeout was booked as a loss and a tie was dropped "
             "unwritten, so this tree's draw count is UNKNOWN and not zero; ")
    return (f"per-opponent per-cycle OUTCOME QUOTA: the first {loss_quota} losses, {draws}"
            f"and the first {win_quota} wins of each opponent are persisted as traces; every "
            f"later battle is played (and counts toward the win rate) but is NOT traced. Under "
            f"battle-level work-stealing each shard unit carries max(1, ceil(quota / n_shards)), "
            f"so the per-opponent totals are approximately these. LOSS-ENRICHED BY DESIGN — the "
            f"traces are a loss-forensics sample, never a random subsample of the cycle.")


def _rate(num: int, den: int) -> "float | None":
    """``num/den``, or ``None`` when the denominator is zero.

    ``None``, never 0.0: an opponent that lost no battles has an UNDEFINED loss-capture rate, and
    a 0.0 there would read as "we captured none of its losses" — a different, wrong claim.
    """
    return (num / den) if den > 0 else None


def selection_entry(*, battles_played: int, battles_won: int,
                    traces_written: int, traces_won: int,
                    battles_drawn: int = 0, traces_drawn: int = 0) -> dict:
    """One opponent's selection record, with the three derived capture rates.

    ``capture_rate_win``  = traces_won  / battles_won        (traces per WON battle played)
    ``capture_rate_loss`` = traces_lost / battles_lost       (traces per LOST battle played)
    ``capture_rate_draw`` = traces_drawn / battles_drawn     (traces per DRAWN battle played)

    All three are ``None`` when their denominator is zero. Counts are CLAMPED into consistency
    (`traces_won <= battles_won`, `traces_lost <= battles_lost`) rather than trusted: a partial
    eval cycle can report a shard's traces while its battle counts came from elsewhere, and a
    capture rate above 1 is an arithmetic impossibility that would silently produce a negative
    importance weight downstream.

    🚨 **A DRAW IS SUBTRACTED FROM THE LOSSES, NOT ADDED TO THE PLAYED COUNT.** poke-env counts a
    250-turn timeout as a loss and a tie as neither a win nor a loss, so ``battles_played`` (its
    ``n_finished_battles``) already contains every draw. ``lost = played - won - drawn`` is what
    makes the loss capture rate a statement about DECISIVE losses; treating draws as extra
    battles would inflate the denominator and quietly deflate the rate.
    """
    played = max(0, int(battles_played))
    won = min(max(0, int(battles_won)), played)
    drawn = min(max(0, int(battles_drawn)), played - won)
    written = min(max(0, int(traces_written)), played)
    t_won = min(max(0, int(traces_won)), won, written)
    t_drawn = min(max(0, int(traces_drawn)), drawn, written - t_won)
    lost = played - won - drawn
    t_lost = min(written - t_won - t_drawn, lost)
    return {
        "battles_played": played,
        "battles_won": won,
        "battles_drawn": drawn,
        "traces_written": written,
        "traces_won": t_won,
        "traces_drawn": t_drawn,
        "capture_rate_win": _rate(t_won, won),
        "capture_rate_loss": _rate(t_lost, lost),
        "capture_rate_draw": _rate(t_drawn, drawn),
    }


def build_selection(per_opponent: "dict[str, dict]", *, win_quota: int, loss_quota: int,
                    draw_quota: "int | None" = None) -> dict:
    """The whole `selection` block: the schema tag, the quotas, and one entry per opponent.

    ``per_opponent`` maps an opponent key to the raw counts (the keyword names of
    :func:`selection_entry`).
    """
    block = {
        "schema": SELECTION_SCHEMA,
        "win_quota": int(win_quota),
        "loss_quota": int(loss_quota),
        "opponents": {str(k): selection_entry(**v) for k, v in sorted(per_opponent.items())},
    }
    if draw_quota is not None:
        block["draw_quota"] = int(draw_quota)
    return block


def read_selection(manifest: "dict | None") -> "dict | None":
    """The `selection` block of a manifest, or ``None`` when the tree does not record one.

    ``None`` covers every way the record can be missing — no manifest, a legacy manifest, a cycle
    that crashed before collecting (the block is written as ``null`` at launch), a schema this
    build does not know, or an empty opponent map. All of them mean the same thing to a consumer:
    the selection is UNKNOWN. Never inferred, never defaulted to uniform.

    An OLDER KNOWN schema is not missing — it is read as it was written. Schema 1 predates the
    draw bucket, so its entries simply carry no ``battles_drawn`` / ``traces_drawn`` /
    ``capture_rate_draw``; a consumer must treat those as ABSENT (unknown), never as zero.
    """
    if not isinstance(manifest, dict):
        return None
    sel = manifest.get(SELECTION_KEY)
    if not isinstance(sel, dict):
        return None
    if sel.get("schema") not in KNOWN_SELECTION_SCHEMAS:
        return None
    opponents = sel.get("opponents")
    if not isinstance(opponents, dict) or not opponents:
        return None
    return sel


def selection_rule(manifest: "dict | None") -> "str | None":
    """The rule in words, or ``None`` when the tree does not record one."""
    if not isinstance(manifest, dict):
        return None
    rule = manifest.get(SELECTION_RULE_KEY)
    return rule if isinstance(rule, str) and rule else None


def capture_rates(manifest: "dict | None") -> "dict[str, dict] | None":
    """``{opponent: entry}`` from a manifest, or ``None`` when the selection is unknown."""
    sel = read_selection(manifest)
    return dict(sel["opponents"]) if sel else None


def manifest_win_rates(manifest: "dict | None") -> "dict[str, float]":
    """``{opponent: battles_won / battles_played}`` — the cycle's OWN recorded rate per opponent.

    This is the population a selection correction targets, taken from the cycle that produced the
    traces rather than joined in from `eval_results.jsonl` by a reader who knew to. An opponent
    with no played battles is omitted (its rate is undefined, and a 0.0 would be a claim).
    """
    out: dict[str, float] = {}
    for name, e in (capture_rates(manifest) or {}).items():
        played = e.get("battles_played") or 0
        if played > 0:
            out[str(name)] = float(e.get("battles_won", 0)) / float(played)
    return out


def describe_selection(manifest: "dict | None") -> str:
    """A one-line human label for a curve computed over this tree's traces.

    Either the recorded capture rates in brief, or :data:`UNKNOWN_LABEL`. Consumers print this
    BESIDE their numbers — the numbers themselves are unchanged by the presence of a record.
    """
    rates = capture_rates(manifest)
    if not rates:
        return UNKNOWN_LABEL
    wins = [e["capture_rate_win"] for e in rates.values() if e["capture_rate_win"] is not None]
    losses = [e["capture_rate_loss"] for e in rates.values() if e["capture_rate_loss"] is not None]
    played = sum(e["battles_played"] for e in rates.values())
    traced = sum(e["traces_written"] for e in rates.values())
    # Schema 1 has no draw keys at all — ABSENT, not zero. `.get(...)` with an `is None` filter is
    # what keeps a pre-draw-bucket cycle from reporting "0 draws" as if it had counted them.
    has_draws = any("battles_drawn" in e for e in rates.values())
    drawn = sum(e.get("battles_drawn") or 0 for e in rates.values()) if has_draws else None
    draw_note = (f"; {drawn} of those battles were DRAWS (tie or 250-turn timeout)"
                 if has_draws else
                 "; DRAWS NOT COUNTED (pre-draw-bucket cycle — a timeout was booked as a loss "
                 "and a tie was dropped unwritten, so a zero here would be a claim)")

    def _mean(xs: "list[float]") -> str:
        return f"{sum(xs) / len(xs):.3f}" if xs else "n/a"

    # The two rates are REPORTED, never compared for the reader: which one is higher depends on
    # the win rate as well as the quota (against a bot the model loses to, the loss quota binds
    # first and the loss rate reads LOWER), so asserting a direction here would be wrong half the
    # time. What is always true is that they differ when the quota binds unevenly.
    return (f"SELECTION RECORDED — {traced} traces of {played} battles played over "
            f"{len(rates)} opponents; mean capture rate {_mean(wins)} per WIN vs "
            f"{_mean(losses)} per LOSS. The two differ wherever the quota binds unevenly, and "
            f"that difference IS the sample's skew — compare them, never assume they match"
            f"{draw_note}.")
