"""THE TRAINING-SIDE CALIBRATION READ — `python -m main.ops.value_sidecar_read <run>`.

The counterpart to `main.ops.critic_read`, and the distinction between them is the whole point:

* **`critic_read` reads EVAL battles** — a greedy trainee against a fixed roster, sampled under a
  quota that prefers losses. It answers *"is this critic calibrated on the eval distribution?"*
* **this reads the TRAINING buffer** — the stochastic policy against the live self-play curriculum,
  scored against `win_target`, the label the BCE actually minimises. It answers *"is this critic
  calibrated on the distribution it is being fit to?"*

A critic can pass one and fail the other, and until `gen3_value_sidecar_v1` only the first was
measurable. Neither supersedes the other; a disagreement between them is a finding about
generalisation, not a defect in either instrument.

**WHAT IT REPORTS.** Mean V against mean target, the Murphy decomposition of the Brier score
(reliability / resolution / uncertainty) and skill, sliced four ways — by TURN bucket, by OPPONENT
CLASS, by episode OUTCOME and by training STEP — each with a battle-clustered bootstrap CI.

🚨 **`critic_resolution` IS THE METER, `critic_reliability` IS NOT** — the same rule the training
leaf states for the live TB family. A base-rate forecaster scores a perfect 0 reliability and a
useless 0 resolution, so a change that improves reliability while resolution stays flat has moved
the number that was never the disease. The report prints resolution first and says so.

🚨 **THE BOOTSTRAP CLUSTERS BY EPISODE, NEVER BY ROW.** Decisions inside one battle share a board,
a team matchup and a dice stream; the sidecar samples uniformly over buffer cells, so a long
episode contributes proportionally more rows and a row-level interval would understate its width by
exactly that correlation. `agents.training.stats.cluster_bootstrap_ci` is imported, never
re-implemented.

🚨 **SLICING BY OUTCOME IS NOT A CALIBRATION CHECK AND THE REPORT SAYS SO.** Conditioning on the
outcome and then asking whether V predicted it is circular — within the won slice the target is 1.0
by construction, so "mean V" there is a statement about how confident the critic was on states that
happened to win, which is a resolution component, not an error. It is reported because the
asymmetry between the two slices is informative (an optimistic critic reads high in BOTH), and
labelled so nobody quotes it as a bias.

🚨 **`target` HAS ONE NAME AND TWO MEANINGS, AND THE HEADER IS THE ONLY THING THAT SAYS WHICH**
(`gen3_winprob_lambda_v1`, schema 2). At `--win-prob-lambda 1.0` — every arm before arm 8, and
every schema-1 file — it is the episode's terminal 0/1 OUTCOME. Below 1.0 it is the λ-RETURN: a
per-state soft probability blending that outcome with the collector's own recorded `V(s)`, which
varies WITHIN an episode and, under the default `bootstrap` truncation, covers trailing rows that
have no outcome at all. **This reader reads the header FIRST**, states in words what `target` is
for the file in hand, LABELS every calibration table with the quantity it scored, and — on a λ
file — adds a second set of tables scored against the `outcome` column so the two readings sit
side by side instead of being confused for one another. Three arms have each landed on exactly
155,137 rows; nothing about matching row counts makes two files interchangeable.

**REFUSALS OVER SILENCE.** A run with no sidecar, a sidecar with no header (so the critic mode —
and therefore whether `v` is a probability at all — is unknown), a `shaped` sidecar read as if it
were calibration, a sidecar whose `target` changes meaning MID-FILE (a resume across the flag
boundary), a `--compare` across two files that disagree on schema or λ, and a slice below the cell
floor: each refuses or is marked, never averaged into a confident number.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

from agents.model.critic_mode import is_winprob
from agents.training.stats import MIN_CELL_N, cluster_bootstrap_ci, cluster_bootstrap_diff_ci
from agents.training.value_sidecar import (
    SCHEMA_EQUIVALENCE, describe_target, read_sidecar_segments, same_quantity, sidecar_path,
    target_identity, target_is_outcome,
)
from agents.training.wrappers import MaskableAgentWrapper as _W
from main.ops.run_ref import refuse, resolve_run_dir

TOOL = "value_sidecar_read"
TOOL_VERSION = 2

#: The opponent-class codes the env tags rows with, by name. Imported from the wrapper that
#: DEFINES them rather than re-listed, so a new class cannot leave this reader silently mislabelling.
OPP_CLASS_NAMES = {
    _W.OPP_CLASS_BOT: "bot",
    _W.OPP_CLASS_POOL: "pool",
    _W.OPP_CLASS_STABLE: "stable",
    _W.OPP_CLASS_EXPLOITER: "exploiter",
}

#: Turn buckets. Coarse and fixed rather than quantile-derived: a quantile bucketing moves with the
#: run's own episode-length distribution, so two runs' "early game" would not be the same states —
#: and episode length is itself a primary endpoint on a win-prob arm.
TURN_BUCKETS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 250))

HEADLINE = (
    "critic_resolution is the meter (HIGHER is better); reliability alone is not — "
    "a base-rate forecaster scores a perfect 0 reliability and a useless 0 resolution."
)


# ── the statistics ────────────────────────────────────────────────────────────────────────────
def murphy(preds, targets):
    """The Brier score and its Murphy decomposition: BS = reliability - resolution + uncertainty.

    Binned on the forecast into 10 equal-width bins, which is the same binning
    `scaffolding.reliability_table` uses for the live `win_prob/*` family — so a sidecar number and
    a TB number are the same statistic and not two neighbouring ones.

    Returns a dict, or ``None`` when there is nothing to decompose. **`None`, never zeros**: a
    zero reliability and an unmeasurable one look identical in a table and mean opposite things.
    """
    n = len(preds)
    if n < MIN_CELL_N:
        return None
    base = sum(targets) / n
    brier = sum((p - y) ** 2 for p, y in zip(preds, targets)) / n
    bins = defaultdict(list)
    for p, y in zip(preds, targets):
        bins[min(9, int(p * 10))].append((p, y))
    reliability = resolution = 0.0
    for cell in bins.values():
        k = len(cell)
        p_bar = sum(p for p, _ in cell) / k
        y_bar = sum(y for _, y in cell) / k
        reliability += k * (p_bar - y_bar) ** 2
        resolution += k * (y_bar - base) ** 2
    reliability /= n
    resolution /= n
    uncertainty = base * (1.0 - base)
    return {
        "n": n,
        "base_rate": base,
        "mean_v": sum(preds) / n,
        "mean_target": base,
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        # Murphy's identity. It must close to ~0; a residual that does not is an arithmetic bug in
        # this function, not a property of the critic, and it is printed rather than asserted away.
        "decomp_residual": brier - (reliability - resolution + uncertainty),
        # Brier skill against the base rate: 1 = perfect, 0 = no better than always forecasting the
        # base rate, NEGATIVE = worse than that. `None` when there is no uncertainty to beat (a
        # slice that was won or lost every time), because the ratio is undefined, not infinite.
        "skill": (1.0 - brier / uncertainty) if uncertainty > 0 else None,
    }


def _pairs(rows, target_key, known_key):
    """``(preds, targets, clusters)`` for the rows this scoring column can actually score.

    🚨 **THE KNOWN-FLAG IS PER COLUMN, not per row.** Under `--win-prob-lambda-truncated
    bootstrap` a trailing in-progress row has `target_known: true` (the λ-return covers it) and
    `outcome_known: false` (its episode never finished), so the two tables are scored on
    DIFFERENT, correctly different, row sets — and neither may borrow the other's mask.
    """
    usable = [r for r in rows if r.get(known_key) and r.get(target_key) is not None]
    return ([float(r["v"]) for r in usable],
            [float(r[target_key]) for r in usable],
            [str(r["episode"]) for r in usable])


def _cell(rows, *, draws, seed, target_key="target", known_key="target_known"):
    """One slice's statistics plus a battle-clustered CI on the mean signed error (V - target).

    ``target_key`` selects WHICH quantity `v` is scored against — `target` (the training target,
    whatever the header says that is) or `outcome` (the terminal 0/1 bit, when the file carries
    it). The caller is responsible for LABELLING the result; this function does the arithmetic and
    knows nothing about which reading is which.
    """
    preds, targets, clusters = _pairs(rows, target_key, known_key)
    usable = preds
    if len(usable) < MIN_CELL_N:
        return {"n": len(usable), "under_floor": True, "floor": MIN_CELL_N}
    out = murphy(preds, targets) or {"n": len(usable), "under_floor": True}
    errs = [p - y for p, y in zip(preds, targets)]
    lo, hi = cluster_bootstrap_ci(errs, clusters, draws=draws, seed=seed)
    out["mean_error"] = sum(errs) / len(errs)
    out["mean_error_ci"] = [lo, hi]
    # The sign convention, stated once so the report never has to hedge: POSITIVE = the critic is
    # OPTIMISTIC on this slice (it forecast more win probability than the outcomes delivered).
    out["direction"] = (
        "UNRESOLVED (the CI straddles 0)" if lo is None or hi is None or (lo <= 0.0 <= hi)
        else ("OPTIMISTIC" if out["mean_error"] > 0 else "PESSIMISTIC"))
    return out


def _slices(rows, *, draws, seed, target_key="target", known_key="target_known",
            split_key="target"):
    """The four registered slicings. A slice with no rows is ABSENT, never an empty cell.

    ``target_key`` / ``known_key`` say what `v` is SCORED against; ``split_key`` says what the
    by-OUTCOME table SPLITS on, and the two are separate because a λ-return cannot split anything.

    🚨 **A SOFT TARGET HAS NO `won` / `lost` SLICE.** The split used to be `target == 1.0` /
    `target == 0.0`, which under `--win-prob-lambda < 1` selects nothing except the handful of rows
    whose λ-return happened to land exactly on an endpoint — a table that would look like a
    thin-but-real reading of the win/loss asymmetry and would in fact be a reading of the rows
    nearest their own terminals. ``split_key=None`` OMITS the table, and the report says why.
    """
    by_turn, by_opp, by_outcome, by_step = {}, {}, {}, {}

    def cell(sel):
        return _cell(sel, draws=draws, seed=seed,
                     target_key=target_key, known_key=known_key)

    for lo, hi in TURN_BUCKETS:
        sel = [r for r in rows if lo <= float(r.get("turn", -1)) < hi]
        if sel:
            by_turn[f"{lo}-{hi}"] = cell(sel)

    for code, name in OPP_CLASS_NAMES.items():
        sel = [r for r in rows if r.get("opp_class") == code]
        if sel:
            by_opp[name] = cell(sel)

    if split_key is not None:
        split_known = "target_known" if split_key == "target" else f"{split_key}_known"
        for label, want in (("won", 1.0), ("lost", 0.0)):
            sel = [r for r in rows if r.get(split_known) and r.get(split_key) is not None
                   and float(r[split_key]) == want]
            if sel:
                by_outcome[label] = cell(sel)

    # Training step, bucketed per 1M — the same bucketing every other ops instrument uses, so a
    # sidecar trend lines up with a TB trend row for row.
    buckets = defaultdict(list)
    for r in rows:
        buckets[int(r.get("step", 0)) // 1_000_000].append(r)
    for m, sel in sorted(buckets.items()):
        by_step[f"{m}M-{m + 1}M"] = cell(sel)

    return {"by_turn": by_turn, "by_opponent_class": by_opp,
            "by_outcome": by_outcome, "by_step": by_step}


# ── the report ────────────────────────────────────────────────────────────────────────────────
def _f(x, nd=4):
    return "     —" if x is None else f"{x:.{nd}f}"


def _ci(pair):
    if not pair or pair[0] is None or pair[1] is None:
        return "[      —,       —]"
    return f"[{pair[0]:+.4f}, {pair[1]:+.4f}]"


def _table(title, cells, note=""):
    if not cells:
        return [f"### {title}", "", "_(no rows in any cell)_", ""]
    out = [f"### {title}", ""]
    if note:
        out += [note, ""]
    out += ["| slice | n | mean V | mean target | resolution | reliability | brier | skill "
            "| mean error | 95% CI (episode-clustered) | reading |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for name, c in cells.items():
        if c.get("under_floor"):
            out.append(f"| `{name}` | {c['n']} | — | — | — | — | — | — | — | — | "
                       f"UNDER THE CELL FLOOR (n < {c.get('floor', MIN_CELL_N)}) — not reported |")
            continue
        out.append(
            f"| `{name}` | {c['n']} | {_f(c['mean_v'])} | {_f(c['mean_target'])} | "
            f"**{_f(c['resolution'])}** | {_f(c['reliability'])} | {_f(c['brier'])} | "
            f"{_f(c['skill'])} | {c['mean_error']:+.4f} | {_ci(c['mean_error_ci'])} | "
            f"{c['direction']} |")
    return out + [""]


def _reading_block(reading) -> list:
    """One complete set of tables, under the LABEL that says what they scored against."""
    L = [f"## {reading['label']}", ""]
    if reading.get("note"):
        L += [reading["note"], ""]
    L += _table("pooled", {"all": reading["overall"]})
    sl = reading["slices"]
    L += _table("By TURN bucket", sl["by_turn"],
                "Game phase. ⚠️ A battle turn is not a decision index — one turn can carry several "
                "decisions, so consecutive rows can share a turn.")
    L += _table("By OPPONENT CLASS", sl["by_opponent_class"],
                "⚠️ The env tags the CLASS (bot / pool / stable / exploiter), never the archetype "
                "name or the snapshot step — those are not reachable per-step today.")
    if sl["by_outcome"]:
        L += _table("By OUTCOME", sl["by_outcome"],
                    "🚨 **NOT a calibration check.** Conditioning on the outcome makes the target "
                    "constant within each slice by construction, so `mean error` here is a "
                    "resolution component, not a bias. Read the ASYMMETRY between the two rows — "
                    "an optimistic critic reads high in both.")
    elif reading.get("no_outcome_split"):
        L += ["### By OUTCOME", "", reading["no_outcome_split"], ""]
    L += _table("By TRAINING STEP (1M buckets)", sl["by_step"],
                "The trend. Compare against the run's own `win_prob/critic_resolution` series — "
                "same statistic, different distribution (training vs eval).")
    return L


def render_md(doc) -> str:
    h = doc["header"]
    idn = doc["target_identity"]
    L = [f"# Training-side value sidecar — `{doc['run']}`", "",
         f"_{doc['tool']} v{doc['tool_version']}, generated {doc['generated_at']}_", "",
         f"`{doc['invocation']}`", "",
         "## 🚨 What `target` IS in this file", "",
         "The header is read FIRST and decides everything below it: the column has one NAME and "
         "two MEANINGS, and only these four fields say which one is on disk.", "",
         "| header field | value |", "|---|---|",
         f"| `schema` | **{idn['schema']}** |",
         f"| `critic_mode` | `{idn['critic_mode']}` "
         f"({'V is a PROBABILITY' if h.get('v_is_probability') else 'V is a SHAPED RETURN'}) |",
         f"| `win_prob_lambda` | **{idn['win_prob_lambda']:g}**"
         f"{' (OFF — the identity)' if idn['win_prob_lambda'] >= 1.0 else ' (λ-RETURN TARGETS)'} |",
         f"| `win_prob_lambda_truncated` | `{idn['win_prob_lambda_truncated']}`"
         f"{' — INERT at λ = 1.0' if idn['win_prob_lambda'] >= 1.0 else ''} |",
         f"| writer segments in this file | {doc['n_segments']} |", "",
         f"**`target` is {doc['target_description']}.**", "",
         f"🚨 {HEADLINE}", "",
         "## What this is", "",
         "The critic scored against **its own training target** — the `win_target` label the BCE "
         "minimises, on the states the rollout buffer actually held. This is NOT "
         "`main.ops.critic_read`, which reads EVAL battles (a greedy trainee, a fixed roster, a "
         "loss-preferring quota). A critic can pass one and fail the other; a disagreement is a "
         "finding about generalisation, not a defect in either.", "",
         "## The sample", "",
         f"| rows | {doc['n_rows']:,} |", "|---|---|",
         f"| labelled (`target_known`) | {doc['n_labelled']:,} |",
         ("| carrying an OUTCOME (`outcome_known`) | "
          + (f"{doc['n_outcome']:,} |" if doc["n_outcome"] is not None
             else "**NONE — UNRECOVERABLE** |")),
         f"| episodes | {doc['n_episodes']:,} |",
         f"| sampling fraction | {h.get('fraction')} (seed {h.get('seed')}) |",
         f"| step range | {doc['step_lo']:,} … {doc['step_hi']:,} |",
         f"| completed episodes | {doc['n_complete']:,} |",
         f"| timeouts (clock reached the cap) | {doc['n_timeout']:,} |", ""]

    if doc.get("notes"):
        L += ["## Notes", ""] + [f"- {n}" for n in doc["notes"]] + [""]

    for reading in doc["readings"]:
        L += _reading_block(reading)
    return "\n".join(L) + "\n"


# ── the CLI ───────────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.ops.value_sidecar_read",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Calibration of the critic against its OWN TRAINING TARGET, from a run's "
                    "value sidecar (gen3_value_sidecar_v1).",
        epilog=(
            "Refusals over silence: a run with no sidecar, a sidecar with no header (the critic "
            "mode — and so whether V is a probability at all — would be unknown), a sidecar in "
            "which nothing is labelled, and a slice under the cell floor are each refused or "
            "marked, never averaged into a confident number.\n\n"
            "This is NOT main.ops.critic_read. That one reads EVAL battles; this reads the "
            "TRAINING buffer. Neither supersedes the other."))
    p.add_argument("run", help="a run NAME under models/ or a run DIRECTORY")
    p.add_argument("--out", default=None,
                   help="directory for value_sidecar_read.{json,md} (default: print only)")
    p.add_argument("--bootstrap-draws", type=int, default=2000,
                   help="episode-clustered bootstrap resamples (default 2000)")
    p.add_argument("--seed", type=int, default=7, help="bootstrap seed (default 7)")
    p.add_argument("--allow-shaped", action="store_true",
                   help="read a --critic shaped sidecar anyway. REFUSED by default: under shaped "
                        "V is a PopArt-normalised shaped return whose scale moves over the run, "
                        "so a Brier decomposition of it is a category error, not a loose reading.")
    p.add_argument("--compare", default=None, metavar="RUN2",
                   help="a SECOND run to read beside the first. REFUSED when the two sidecars "
                        "disagree on schema, critic mode or --win-prob-lambda: `target` then "
                        "holds a different QUANTITY on each side and a delta between them is "
                        "mostly the difference between the two definitions.")
    p.add_argument("--quiet", action="store_true", help="suppress progress lines")
    return p


# ── the schema guard ──────────────────────────────────────────────────────────────────────────
def _load(run_dir, *, role="the run"):
    """``(header, rows, segments)`` for one run, refusing every way the file can be unreadable.

    ⚠️ The file is parsed ONCE. A production sidecar is a few hundred MB — reading it twice to get
    the segment list beside the pooled rows would double the cost of every read for a count.

    🚨 **A MIXED FILE IS REFUSED BY ROW INDEX.** The sidecar writes a header per writer session,
    so a run resumed across the `gen3_winprob_lambda_v1` boundary carries terminal 0/1 outcomes in
    its head and λ-returns in its tail. Pooling them averages two different quantities under one
    column name and produces a number that is neither.
    """
    try:
        segments = read_sidecar_segments(str(run_dir))
    except FileNotFoundError:
        refuse(f"REFUSING: no value sidecar at {sidecar_path(str(run_dir))}",
               f"  {role} was trained without one (--value-sidecar off, or before",
               "  gen3_value_sidecar_v1 landed). It cannot be reconstructed after the fact — the",
               "  rollout buffer it read is long gone. Nothing is read and nothing is concluded.")
        raise AssertionError("unreachable")  # pragma: no cover
    except ValueError as e:
        refuse(f"REFUSING: {e}")
        raise AssertionError("unreachable")  # pragma: no cover

    first = segments[0]["header"]
    for seg in segments[1:]:
        if same_quantity(first, seg["header"]):
            continue
        segs = segments
        lines = [f"REFUSING: {role}'s sidecar changes what `target` MEANS at row "
                 f"{seg['row_index']:,}.",
                 "  The file was written by more than one process and they did not agree:"]
        for i, seg in enumerate(segs):
            idn = target_identity(seg["header"])
            lines.append(
                f"    segment {i} (rows {seg['row_index']:,}+, {len(seg['rows']):,} rows): "
                f"schema {idn['schema']}, critic {idn['critic_mode']}, "
                f"win_prob_lambda {idn['win_prob_lambda']:g}, "
                f"truncated {idn['win_prob_lambda_truncated']} — target is "
                f"{'the terminal 0/1 OUTCOME' if idn['win_prob_lambda'] >= 1.0 else 'a λ-RETURN'}")
        lines += [
            "  Averaging a 0/1 outcome with a soft λ-return under one column name produces a "
            "number that is neither, and it would carry a CI describing neither.",
            "  THE FIX: read each segment on its own. Split the file at the header rows "
            "(`kind == \"header\"`) into one `rows.jsonl` per segment and read each; compare only "
            "the OUTCOME-based tables across them, which are the same quantity on both sides. "
            "Nothing is read and nothing is concluded."]
        refuse(*lines)
    return first, [r for sg in segments for r in sg["rows"]], segments


def check_comparable(a_dir, a_header, b_dir, b_header) -> None:
    """REFUSE a two-run read whose two `target` columns hold different QUANTITIES.

    🚨 THIS IS THE WHOLE POINT OF THE HEADER. Three arms have each landed on exactly 155,137 rows,
    which invites treating the files as interchangeable; arm 8 writes schema 2 with a λ-return in
    the same column name. A pooled or differenced read across that boundary is mostly the
    difference between the two DEFINITIONS of `target`, and nothing in the row shape says so.

    A schema-1 file and a schema-2 file at λ = 1.0 compare EQUAL here on purpose — they are
    byte-identical apart from two header fields, and refusing them would be a false alarm.
    """
    a_idn, b_idn = target_identity(a_header), target_identity(b_header)
    if same_quantity(a_header, b_header):
        return
    differing = [k for k in a_idn if a_idn[k] != b_idn[k]]
    if differing == ["schema"]:
        differing = [f"schema (and {a_idn['schema']} ↔ {b_idn['schema']} is not a declared "
                     f"equivalence — {sorted(SCHEMA_EQUIVALENCE)} is)"]
    lines = ["REFUSING: the two sidecars do not mean the same thing by `target`.",
             f"  differing header field(s): {', '.join(differing)}", ""]
    for role, d, idn in (("run  ", a_dir, a_idn), ("--compare", b_dir, b_idn)):
        lines += [f"  {role} {d}",
                  f"      schema {idn['schema']} · critic {idn['critic_mode']} · "
                  f"win_prob_lambda {idn['win_prob_lambda']:g} · "
                  f"truncated {idn['win_prob_lambda_truncated']}",
                  f"      target = {describe_target(idn)}"]
    lines += [
        "",
        "  Pooling or differencing these two columns averages two different quantities under one "
        "name. Matching ROW COUNTS do not make two files interchangeable.",
        "  THE FIX: read each run ON ITS OWN (`python -m main.ops.value_sidecar_read <run>`), "
        "which labels each table with the quantity it scored. Compare only the OUTCOME-based "
        "tables, which are the terminal 0/1 bit on both sides — and only where the λ file carries "
        "an `outcome` column at all. Nothing is read and nothing is concluded."]
    refuse(*lines)


def build_readings(rows, header, *, draws, seed):
    """The one or two labelled table-sets this file supports, and WHY the second may be missing.

    * At **λ = 1.0** there is exactly ONE reading: `target` is the outcome, so a second table
      against the outcome would be the identical numbers under a second heading.
    * Below 1.0 the PRIMARY reading is against the λ-return — the quantity the BCE actually
      minimised — and a SECOND reading against the `outcome` column is added when the file carries
      one, so "is the critic calibrated to its objective" and "is the critic calibrated to
      winning" are read separately instead of being confused.
    * Below 1.0 with **no** `outcome` column the second reading is UNRECOVERABLE, and the report
      says so in place of the table rather than leaving a silence a reader fills in.
    """
    lam_off = target_is_outcome(header)
    readings = []
    if lam_off:
        readings.append({
            "key": "outcome",
            "label": "Calibration against the TERMINAL 0/1 OUTCOME (`target`)",
            "note": "`win_prob_lambda` is 1.0, so `target` IS the episode's outcome. There is one "
                    "reading here and it is both the objective's own residual and the outcome "
                    "calibration.",
            "overall": _cell(rows, draws=draws, seed=seed),
            "slices": _slices(rows, draws=draws, seed=seed),
        })
        return readings

    readings.append({
        "key": "lambda",
        "label": "Calibration against the λ-RETURN target (NOT the outcome)",
        "note": "🚨 **These tables score `v` against the λ-return** — the soft, per-state target "
                "the BCE actually minimised, which blends the outcome with the collector's own "
                "recorded `V(s)`. They are the objective's residual and they are **not** "
                "comparable, row for row, with any λ = 1.0 run's tables, whose target is the "
                "terminal bit. Nothing in the column name says so, which is why this heading "
                "does.",
        "overall": _cell(rows, draws=draws, seed=seed),
        "slices": _slices(rows, draws=draws, seed=seed, split_key=None),
        "no_outcome_split": "_Omitted: `target` is a SOFT λ-return, so a `won` / `lost` split on "
                            "it selects only the rows whose return happened to land exactly on an "
                            "endpoint — a reading of proximity to a terminal, dressed as a "
                            "reading of the win/loss asymmetry. The outcome split lives in the "
                            "outcome reading below, where it is well defined._",
    })

    if any(r.get("outcome_known") for r in rows):
        readings.append({
            "key": "outcome",
            "label": "Calibration against the OUTCOME (`outcome`, the terminal 0/1 bit)",
            "note": "The generalisation question, and the ONE reading that compares across the λ "
                    "boundary: this column is the terminal bit on every file that carries it. "
                    "⚠️ Its row set is SMALLER than the λ table's — under "
                    "`--win-prob-lambda-truncated bootstrap` a trailing in-progress row has a "
                    "λ-target but no outcome, so it is scored above and absent here. That is "
                    "correct, and it means the two `n` columns are not meant to match.",
            "overall": _cell(rows, draws=draws, seed=seed,
                             target_key="outcome", known_key="outcome_known"),
            "slices": _slices(rows, draws=draws, seed=seed,
                              target_key="outcome", known_key="outcome_known",
                              split_key="outcome"),
        })
    return readings


UNRECOVERABLE_NOTE = (
    "🚨 **THE OUTCOME IS UNRECOVERABLE ON THIS FILE, so there is no outcome-based table.** "
    "`--win-prob-lambda < 1` makes `WinProbLabelCallback` OVERWRITE `win_target` in place with the "
    "λ-return, and this file carries no `outcome` column to preserve what was there. It cannot be "
    "reconstructed from the rows: (1) the λ-return holds the outcome at weight λ^d for a distance "
    "`d` to the terminal that no column records; (2) `win_margin` is the per-turn MATERIAL margin "
    "(a by-product of Φ_mat, ∈ [−1,1]) — its sign is a material lead at that turn, not a win, and "
    "this project's own rule forbids reading a material state as an outcome; (3) `target_known` "
    "under the default `bootstrap` truncation marks rows whose episode never finished at all. "
    "Every reading below is against the λ-RETURN. A file written after "
    "`gen3_winprob_lambda_v1`'s outcome column carries both."
)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    say = (lambda _m: None) if args.quiet else (lambda m: print(f"[{TOOL}] {m}", flush=True))

    run_dir = resolve_run_dir(args.run)
    # 🚨 THE HEADER IS READ FIRST, before a single statistic is computed. Everything the report
    # says about `target` — and whether a second run may be compared to this one at all — is
    # decided here.
    header, rows, segments = _load(run_dir, role="the run")
    idn = target_identity(header)

    cmp_dir = cmp_header = cmp_rows = None
    if args.compare:
        cmp_dir = resolve_run_dir(args.compare)
        cmp_header, cmp_rows, _ = _load(cmp_dir, role="--compare")
        check_comparable(run_dir, header, cmp_dir, cmp_header)

    mode = header.get("critic_mode")
    if not is_winprob(mode) and not args.allow_shaped:
        refuse(f"REFUSING: this sidecar was written under --critic {mode!r}, so `v` is a "
               "PopArt-normalised SHAPED RETURN,",
               "  not a probability. A Brier decomposition of it is a category error rather than "
               "a loose reading:",
               "  its scale moves over the run, so the same number means different things at 2M "
               "and 20M.",
               "  Pass --allow-shaped if you want the raw pairs anyway. Nothing is concluded.")

    labelled = [r for r in rows if r.get("target_known")]
    if not labelled:
        refuse(f"REFUSING: {len(rows):,} sidecar rows and NONE of them is labelled.",
               "  Every row's episode was still in progress at its rollout's end, or the sidecar",
               "  ran before WinProbLabelCallback's back-fill (a callback-ORDER defect — the",
               "  sidecar must be registered after it). Nothing is read and nothing is concluded.")
    say(f"{len(rows):,} rows ({len(labelled):,} labelled) from {sidecar_path(str(run_dir))}")
    say(f"schema {idn['schema']} · critic {idn['critic_mode']} · "
        f"win_prob_lambda {idn['win_prob_lambda']:g} · "
        f"truncated {idn['win_prob_lambda_truncated']} · target = {describe_target(header)}")

    n_outcome = sum(1 for r in rows if r.get("outcome_known"))
    notes = []
    if not target_is_outcome(header) and n_outcome == 0:
        notes.append(UNRECOVERABLE_NOTE)
    if any(r.get("opp_class") is None for r in rows):
        notes.append("Some rows carry NO opponent class — the buffer had no `opp_class` obs key. "
                     "Since `gen3_value_sidecar_v1` every win-prob run emits it, so this means a "
                     "PRE-WIDENING checkpoint was resumed under its own saved observation space. "
                     "Those rows are ABSENT from the by-class table, never pooled into `bot` "
                     "(0 is a real class, so a default would invent a curriculum).")
    notes.append("The opponent is identified by CLASS only. The bot's archetype name and the pool "
                 "snapshot's step are chosen per episode and never reach the observation; an "
                 "opponent LADDER RATING does not exist at training time at all (a snapshot's Elo "
                 "is derived post-hoc by `main.elo` from the finished run's ladder). Look a rating "
                 "up by snapshot afterwards — it was never a per-step fact.")
    if len(segments) > 1:
        notes.append(
            f"This sidecar was written by {len(segments)} writer SESSIONS (a run with restarts). "
            "Every one declares the same schema, critic mode and λ — that is checked, and a "
            "disagreement is refused by row index rather than pooled.")
    incomplete = sum(1 for r in rows if not r.get("ep_complete"))
    if incomplete:
        notes.append(f"{incomplete:,} row{'' if incomplete == 1 else 's'} "
                     f"belong{'s' if incomplete == 1 else ''} to an episode that straddles a "
                     "rollout boundary; its `ep_len` is the length WITHIN the buffer, not the "
                     "episode's. Every length statistic filters on `ep_complete`.")

    steps = [int(r.get("step", 0)) for r in rows]
    doc = {
        "tool": TOOL, "tool_version": TOOL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "invocation": "python -m main.ops.value_sidecar_read " + " ".join(sys.argv[1:]),
        "run": run_dir.name, "run_dir": str(run_dir), "header": header,
        "target_identity": idn, "target_description": describe_target(header),
        "target_is_outcome": target_is_outcome(header),
        "n_segments": len(segments),
        "n_rows": len(rows), "n_labelled": len(labelled),
        "n_outcome": (n_outcome if n_outcome else None),
        "n_episodes": len({r.get("episode") for r in labelled}),
        "n_complete": sum(1 for r in rows if r.get("ep_complete")),
        "n_timeout": sum(1 for r in rows if r.get("timeout")),
        "step_lo": min(steps) if steps else 0, "step_hi": max(steps) if steps else 0,
        "headline": HEADLINE, "notes": notes,
        "readings": build_readings(rows, header, draws=args.bootstrap_draws, seed=args.seed),
    }
    # Back-compat for a consumer of v1's document shape: the PRIMARY reading's tables, under the
    # names they had. They are the λ-return's tables on a λ file, which is why every renderer
    # above reads `readings` and labels them instead.
    doc["overall"] = doc["readings"][0]["overall"]
    doc["slices"] = doc["readings"][0]["slices"]

    md = render_md(doc)
    if cmp_rows is not None:
        doc["compare"] = _compare_doc(rows, cmp_rows, cmp_dir, cmp_header,
                                      draws=args.bootstrap_draws, seed=args.seed)
        md += _render_compare(doc["compare"])
    print(md)
    if args.out:
        import os
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "value_sidecar_read.json"), "w") as f:
            json.dump(doc, f, indent=1, default=float)
        with open(os.path.join(args.out, "value_sidecar_read.md"), "w") as f:
            f.write(md)
        say(f"wrote {args.out}/value_sidecar_read.{{json,md}}")
    return 0


def _compare_doc(rows_a, rows_b, b_dir, b_header, *, draws, seed):
    """The two runs' pooled cells and the DELTA with its own episode-clustered CI.

    🎯 **THE DELTA CARRIES ITS OWN CI, not two overlapping ones.** Two intervals that overlap say
    nothing about whether the difference is resolved; a delta CI that straddles 0 reads NOT
    DETECTED, and only a delta CI inside a pre-stated bar supports equivalence.
    """
    ea = [p - y for p, y, _ in zip(*_pairs(rows_a, "target", "target_known"))]
    ca = _pairs(rows_a, "target", "target_known")[2]
    eb = [p - y for p, y, _ in zip(*_pairs(rows_b, "target", "target_known"))]
    cb = _pairs(rows_b, "target", "target_known")[2]
    point, lo, hi = cluster_bootstrap_diff_ci(ea, ca, eb, cb, draws=draws, seed=seed)
    return {
        "compare_run": b_dir.name, "compare_run_dir": str(b_dir),
        "compare_identity": target_identity(b_header),
        "a": _cell(rows_a, draws=draws, seed=seed),
        "b": _cell(rows_b, draws=draws, seed=seed),
        "mean_error_delta": point, "mean_error_delta_ci": [lo, hi],
        "verdict": ("UNRESOLVED (the delta CI straddles 0)"
                    if lo is None or hi is None or (lo <= 0.0 <= hi) else "RESOLVED"),
    }


def _render_compare(c) -> str:
    L = ["", f"## Compared with `{c['compare_run']}`", "",
         "Both sidecars declare the SAME schema, critic mode and `win_prob_lambda`, so `target` is "
         "the same quantity on both sides — which is the only condition under which these two "
         "columns may be differenced at all.", "",
         "| side | n | mean V | mean target | resolution | brier | mean error |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for name, cell in (("this run", c["a"]), (c["compare_run"], c["b"])):
        if cell.get("under_floor"):
            L.append(f"| `{name}` | {cell['n']} | — | — | — | — | UNDER THE CELL FLOOR |")
        else:
            L.append(f"| `{name}` | {cell['n']} | {_f(cell['mean_v'])} | "
                     f"{_f(cell['mean_target'])} | **{_f(cell['resolution'])}** | "
                     f"{_f(cell['brier'])} | {cell['mean_error']:+.4f} |")
    L += ["",
          f"**Δ mean error (this − compare): {_f(c['mean_error_delta'])} "
          f"{_ci(c['mean_error_delta_ci'])}** — {c['verdict']}.", "",
          "🎯 The DELTA carries its own episode-clustered CI. Two overlapping per-side intervals "
          "would say nothing about whether the difference is resolved, and an equivalence claim "
          "needs this interval inside a pre-stated bar — never a bar against a point estimate.",
          ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
