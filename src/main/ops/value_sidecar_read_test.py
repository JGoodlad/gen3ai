"""THE SCHEMA GUARD on the training-side calibration read (`main.ops.value_sidecar_read`).

🚨 **THE DEFECT THESE TESTS STAND FOR.** Every arm before arm 8 wrote schema 1, where the sidecar's
`target` column is the episode's terminal 0/1 OUTCOME — and three of them landed on exactly 155,137
rows, which invites treating the files as interchangeable. Arm 8 (`--win-prob-lambda 0.9`) writes
schema 2, where the SAME column holds a soft λ-return. A reader that pools the two, or that prints
a calibration table without saying which quantity it scored, has produced a number nobody can
interpret and everybody can compare.

So: the header is read FIRST, the report says in words what `target` IS, every table carries the
quantity it scored in its heading, and every way of mixing the two is a refusal that names both
headers.
"""

import json
import os

import pytest

from agents.training.value_sidecar import SIDECAR_DIRNAME, SIDECAR_FILENAME
from main.ops import value_sidecar_read as R

# Comfortably over `stats.MIN_CELL_N` (12) so nothing under test lands under the cell floor for a
# reason that has nothing to do with the schema.
N_EPISODES, EP_LEN = 24, 6


def _header(schema=1, lam=None, truncated=None, critic="winprob", rollout=None):
    h = {"kind": "header", "schema": schema, "tag": "gen3_value_sidecar_v1",
         "critic_mode": critic, "v_is_probability": critic == "winprob",
         "fraction": 1.0, "seed": 0, "max_turns": 250, "started_at": "2026-09-09T00:00:00+00:00"}
    if lam is not None:
        h["win_prob_lambda"] = lam
        h["win_prob_lambda_truncated"] = truncated or "bootstrap"
    if rollout is not None:
        h["win_prob_rollout_target"] = rollout
        h["win_prob_rollout_r"] = 8
        h["win_prob_rollout_mode"] = "replace"
    return h


def _rows(*, lam=1.0, with_outcome=True, bootstrap_tail=0, step=1_000_000, seed=0):
    """A synthetic rollout: `N_EPISODES` episodes of `EP_LEN` states, half won.

    Under λ < 1 the `target` column is a λ-return that DECAYS from the outcome toward 0.5 as the
    state gets further from its terminal — the shape the real recursion produces — while `outcome`
    keeps the terminal bit. ``bootstrap_tail`` marks that many trailing episodes as unmasked-by-
    bootstrap: a λ target but NO outcome, which is exactly the row set difference the two tables
    are supposed to have.
    """
    out = []
    for e in range(N_EPISODES):
        y = float(e % 2)
        boot = e >= N_EPISODES - bootstrap_tail
        for t in range(EP_LEN):
            d = EP_LEN - 1 - t                      # distance to the terminal
            if lam >= 1.0:
                target = y
            else:
                w = lam ** d
                target = w * y + (1.0 - w) * 0.5
            # A critic that is right, plus a small deterministic wobble so `resolution` is not 0.
            v = min(0.98, max(0.02, target + (0.05 if (e + t) % 2 else -0.05)))
            row = {"step": step, "rollout": 0, "env": e % 4, "episode": f"{step}:{e}:0",
                   "t": t, "turn": float(t * 3), "v": v, "win_logit": 0.0,
                   "target": target, "target_known": True,
                   "opp_class": 1, "win_margin": 0.0,
                   "ep_len": EP_LEN, "ep_complete": not boot, "timeout": False}
            if with_outcome:
                row["outcome"] = None if boot else y
                row["outcome_known"] = not boot
            out.append(row)
    return out


def _write(tmp_path, name, *segments):
    """One sidecar file. Each segment is `(header, rows)` — more than one means a RESUME."""
    run = tmp_path / name
    d = run / SIDECAR_DIRNAME
    os.makedirs(d, exist_ok=True)
    with open(d / SIDECAR_FILENAME, "w") as f:
        for header, rows in segments:
            f.write(json.dumps(header) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return run


def _read(*argv):
    """Run the CLI, returning its markdown. A refusal surfaces as `SystemExit(2)`."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        R.main(list(argv) + ["--quiet", "--bootstrap-draws", "40"])
    return buf.getvalue()


def _refusal(*argv):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as e:
            R.main(list(argv) + ["--quiet", "--bootstrap-draws", "40"])
    assert e.value.code == 2
    return buf.getvalue()


# ── what `target` IS, said in words, before any statistic ──────────────────────────────────────
def test_a_SCHEMA_1_file_says_target_is_the_TERMINAL_OUTCOME(tmp_path):
    run = _write(tmp_path, "v1", (_header(1), _rows(lam=1.0, with_outcome=False)))
    md = _read(str(run))
    assert "TERMINAL 0/1 OUTCOME" in md
    assert "| `schema` | **1** |" in md
    assert "λ-RETURN" not in md
    # One reading, and it is the outcome one — a second table would be the same numbers twice.
    assert md.count("## Calibration against") == 1
    assert "Calibration against the TERMINAL 0/1 OUTCOME" in md


def test_a_SCHEMA_2_LAMBDA_file_says_target_is_the_LAMBDA_RETURN_and_labels_its_tables(tmp_path):
    run = _write(tmp_path, "v2", (_header(2, 0.9), _rows(lam=0.9)))
    md = _read(str(run))
    assert "λ-RETURN, λ = 0.9" in md
    assert "| `schema` | **2** |" in md
    assert "| `win_prob_lambda` | **0.9** (λ-RETURN TARGETS) |" in md
    assert "truncated rows" in md or "BOOTSTRAPPED" in md
    # 🚨 The heading itself must carry the quantity — that is the whole fix.
    assert "## Calibration against the λ-RETURN target (NOT the outcome)" in md
    assert "## Calibration against the OUTCOME (`outcome`, the terminal 0/1 bit)" in md


def test_a_SCHEMA_2_file_at_LAMBDA_1_reads_exactly_like_a_SCHEMA_1_one(tmp_path):
    """`gen3_winprob_lambda_v1` says the two are byte-identical but for the header fields."""
    run = _write(tmp_path, "v2off", (_header(2, 1.0), _rows(lam=1.0)))
    md = _read(str(run))
    assert "TERMINAL 0/1 OUTCOME" in md and "λ-RETURN" not in md
    assert "INERT at λ = 1.0" in md


# ── the second table, and what happens when it cannot exist ────────────────────────────────────
def test_the_OUTCOME_table_is_scored_on_its_OWN_row_set_not_the_lambda_one(tmp_path):
    """🚨 A bootstrap-unmasked row has a λ target and NO outcome. The two `n`s must differ."""
    rows = _rows(lam=0.9, bootstrap_tail=4)
    run = _write(tmp_path, "boot", (_header(2, 0.9, "bootstrap"), rows))
    doc_rows = rows
    lam_n = sum(1 for r in doc_rows if r["target_known"])
    out_n = sum(1 for r in doc_rows if r["outcome_known"])
    assert out_n < lam_n                                   # the fixture is the case under test

    readings = R.build_readings(doc_rows, _header(2, 0.9, "bootstrap"), draws=40, seed=7)
    by_key = {r["key"]: r for r in readings}
    assert by_key["lambda"]["overall"]["n"] == lam_n
    assert by_key["outcome"]["overall"]["n"] == out_n
    # And the outcome table is scored against a 0/1 bit, so its mean target is a WIN RATE.
    assert by_key["outcome"]["overall"]["mean_target"] == pytest.approx(0.5, abs=0.06)
    md = _read(str(run))
    assert "the two `n` columns are not meant to match" in md


def test_a_LAMBDA_file_with_NO_outcome_column_says_UNRECOVERABLE_and_why(tmp_path):
    """The honest answer, not a table built out of `win_margin`'s sign."""
    run = _write(tmp_path, "noout", (_header(2, 0.9), _rows(lam=0.9, with_outcome=False)))
    md = _read(str(run))
    assert "UNRECOVERABLE" in md
    # Each of the three reasons must be NAMED — a bare "unrecoverable" invites someone to try.
    assert "weight λ^d" in md
    assert "win_margin" in md and "MATERIAL margin" in md
    assert "bootstrap" in md
    assert "## Calibration against the OUTCOME" not in md
    assert "**NONE — UNRECOVERABLE** |" in md


def test_the_LAMBDA_reading_OMITS_the_won_lost_split_rather_than_faking_one(tmp_path):
    """A soft target has no `won` / `lost` rows; splitting on it reads proximity, not outcome."""
    run = _write(tmp_path, "split", (_header(2, 0.9), _rows(lam=0.9, with_outcome=False)))
    md = _read(str(run))
    assert "_Omitted: `target` is a SOFT λ-return" in md


# ── the refusals ───────────────────────────────────────────────────────────────────────────────
def test_a_MIXED_SCHEMA_file_is_REFUSED_with_the_ROW_INDEX_of_the_change(tmp_path):
    """A resume across the 28ece02a boundary: outcomes in the head, λ-returns in the tail."""
    head = _rows(lam=1.0, with_outcome=False, step=1_000_000)
    tail = _rows(lam=0.9, step=2_000_000)
    run = _write(tmp_path, "mixed", (_header(1), head), (_header(2, 0.9), tail))
    out = _refusal(str(run))
    assert "changes what `target` MEANS at row" in out
    assert f"{len(head):,}" in out                      # the exact index of the change
    assert "segment 0" in out and "segment 1" in out
    assert "the terminal 0/1 OUTCOME" in out and "a λ-RETURN" in out
    assert "Nothing is read and nothing is concluded." in out


def test_COMPARE_across_SCHEMAS_is_REFUSED_naming_BOTH_headers(tmp_path):
    a = _write(tmp_path, "arm7", (_header(1), _rows(lam=1.0, with_outcome=False)))
    b = _write(tmp_path, "arm8", (_header(2, 0.9), _rows(lam=0.9)))
    out = _refusal(str(a), "--compare", str(b))
    assert "do not mean the same thing by `target`" in out
    assert "win_prob_lambda" in out
    assert "arm7" in out and "arm8" in out
    assert "Matching ROW COUNTS do not make two files interchangeable." in out
    assert "Nothing is read and nothing is concluded." in out


def test_COMPARE_across_TRUNCATION_MODES_is_REFUSED_too(tmp_path):
    """Same λ, different boundary convention — a different row POPULATION under one column name."""
    a = _write(tmp_path, "boot", (_header(2, 0.9, "bootstrap"), _rows(lam=0.9)))
    b = _write(tmp_path, "mask", (_header(2, 0.9, "mask"), _rows(lam=0.9)))
    out = _refusal(str(a), "--compare", str(b))
    assert "win_prob_lambda_truncated" in out


def test_COMPARE_of_a_SCHEMA_1_and_a_SCHEMA_2_at_LAMBDA_1_is_ALLOWED(tmp_path):
    """🚨 NOT a false alarm: the two are the same quantity, and refusing them would be wrong."""
    a = _write(tmp_path, "one", (_header(1), _rows(lam=1.0, with_outcome=False)))
    b = _write(tmp_path, "two", (_header(2, 1.0), _rows(lam=1.0)))
    md = _read(str(a), "--compare", str(b))
    assert "## Compared with `two`" in md
    assert "Δ mean error" in md
    # 🎯 The DELTA gets its own CI — two overlapping per-side intervals would say nothing.
    assert "carries its own episode-clustered CI" in md


def test_a_SHAPED_sidecar_is_still_REFUSED_without_allow_shaped(tmp_path):
    run = _write(tmp_path, "shaped",
                 (_header(1, critic="shaped"), _rows(lam=1.0, with_outcome=False)))
    out = _refusal(str(run))
    assert "SHAPED RETURN" in out


def test_a_HEADERLESS_file_is_REFUSED_rather_than_attributed_to_a_later_header(tmp_path):
    run = tmp_path / "hless"
    d = run / SIDECAR_DIRNAME
    os.makedirs(d)
    with open(d / SIDECAR_FILENAME, "w") as f:
        for r in _rows(with_outcome=False):
            f.write(json.dumps(r) + "\n")
    out = _refusal(str(run))
    assert "UNKNOWN" in out


# ── the identity helpers, pinned directly ──────────────────────────────────────────────────────
def test_the_target_identity_fills_the_pre_lambda_DEFAULTS():
    """A schema-1 header carries neither λ field; the defaults are the flag's OFF position."""
    from agents.training.value_sidecar import target_identity

    v1 = target_identity(_header(1))
    assert v1["win_prob_lambda"] == 1.0 and v1["win_prob_lambda_truncated"] == "bootstrap"
    assert v1["schema"] == 1


def test_SAME_QUANTITY_ignores_the_VERSION_but_never_the_MEANING():
    """🚨 The split the whole guard rests on: a moved version number is not a moved meaning."""
    from agents.training.value_sidecar import same_quantity

    # v1 and v2-at-λ-1.0 are the same quantity — refusing them would be a false alarm.
    assert same_quantity(_header(1), _header(2, 1.0, "bootstrap")) is True
    # λ moves the meaning; so does the truncation convention; so does the critic mode.
    assert same_quantity(_header(1), _header(2, 0.9)) is False
    assert same_quantity(_header(2, 0.9, "bootstrap"), _header(2, 0.9, "mask")) is False
    assert same_quantity(_header(1), _header(1, critic="shaped")) is False
    # v3 (gen3_winprob_rollout_target_v1) JOINED the declared set, for v2's measured reason: a v3
    # file at `win_prob_rollout_target` 0.0 is byte-identical to a v2 one apart from three header
    # fields. The DISTINCTION moved into QUANTITY_FIELDS, where it belongs.
    assert same_quantity(_header(2, 1.0), _header(3, 1.0)) is True
    assert same_quantity(_header(3, 1.0), _header(3, 1.0, rollout=0.0012)) is False, (
        "a rollout-labelled file's `target` is a MEASURED win fraction on part of its rows; "
        "pooling it with a terminal-bit file is the defect this guard exists for")
    # An UNDECLARED schema is still refused even at an identical λ — the direction this errs in.
    assert same_quantity(_header(3, 1.0), _header(4, 1.0)) is False
