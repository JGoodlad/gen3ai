"""`--v-column` REACHES THE QUOTA-MATCHED ROWS — it used to be silently ignored on every one.

The defect this file pins (tech-debt P1, 2026-09-16). `critic_read --v-column values` read the
flag at the AS-TRACED call site and `main.ops.quota_match` did not read it at all: the matched
path extracts its OWN frame (`CM.extract_cycle(d["trace_dir"])`, no kwarg), so a
`--v-column values` report's matched rows came out byte-identical to the `win_probs` run while
its as-traced table read the shaped column. Two halves of one report disagreeing about which
tensor `V` is, with nothing on the page saying so (`flywheel_pair_read_2026-09-15/` H-L).

🚨 THE SUBTLETY THAT MAKES A NAIVE FIX WRONG: `conditioning_block(..., frame=…)` SELECTS NOTHING
with its `v_column` argument, because the column was fixed when the frame was extracted. Passing
the flag there and nowhere else would have looked like a fix and changed no number. So the column
is applied where the frame is BUILT, both paths read it through one accessor
(`conditioning_meters.v_column_of`), and `conditioning_block` REFUSES a frame whose recorded
column disagrees with the one it was asked for.

The fixture plants a cycle where `values` and `win_probs` are DIFFERENT TENSORS — which is the
real state of a `--critic shaped` run — so a matched row that ignores the flag is visibly equal to
the wrong one.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, Optional

import numpy as np
import pytest

from main.ops import conditioning_meters as CM
from main.ops import quota_match as QM

STEP = 100
_TURNS = (1, 2, 3, 6, 12, 30)


def _plant(root, *, quota_w: int, quota_l: int, n_battles: int = 120, n_teams: int = 10,
           seed: int = 0):
    """A cycle whose two V columns carry DIFFERENT signals.

    ``win_probs`` is a function of the OPPONENT only; ``values`` adds a strong per-TEAM term. The
    own-team decoder (`cond.own_team_r2.*`, frame-sensitive and therefore quota-matched) can find
    something in ``values`` and almost nothing in ``win_probs``, so the two columns give visibly
    different numbers on the same tree.
    """
    rng = np.random.default_rng(seed)
    cyc = root / "eval_traces" / f"step_{STEP}"
    sel: Dict[str, dict] = {}
    for oi, (opp, wr) in enumerate((("heuristic", 0.8), ("staller", 0.6), ("aggressive", 0.5),
                                    ("sentinel_0", 0.35), ("sentinel_1", 0.2))):
        odir = cyc / opp
        odir.mkdir(parents=True)
        team_bias = {t: float(rng.normal(0, 0.30)) for t in range(n_teams)}
        wins = int(round(wr * n_battles))
        w_written = l_written = 0
        for b in range(n_battles):
            y = 1.0 if b < wins else 0.0
            if y and w_written >= quota_w:
                continue
            if not y and l_written >= quota_l:
                continue
            w_written += int(y)
            l_written += int(not y)
            team = (b * 7 + oi * 3) % n_teams
            wp = float(np.clip(wr + rng.normal(0, 0.04), 0.01, 0.99))
            val = float(np.clip(wr + team_bias[team] + rng.normal(0, 0.02), 0.01, 0.99))
            n = len(_TURNS)
            np.savez(odir / f"b{b}_states.npz", values=np.full(n, val),
                     win_probs=np.full(n, wp), has_state=np.ones(n, dtype=np.int64))
            (odir / f"b{b}_summary.json").write_text(json.dumps({
                "meta": {"result": "WIN" if y else "LOSS", "step": STEP},
                "teams": {"ours": [{"species": f"m{team}_{k}"} for k in range(6)]},
                "invocations": [{"i": i, "turn": t} for i, t in enumerate(_TURNS)]}))
        sel[opp] = {"battles_played": n_battles, "battles_won": wins, "battles_drawn": 0,
                    "traces_written": w_written + l_written, "traces_won": w_written,
                    "traces_drawn": 0,
                    "capture_rate_win": w_written / wins,
                    "capture_rate_loss": l_written / (n_battles - wins)}
    (cyc / "eval_manifest.json").write_text(json.dumps(
        {"step": STEP, "selection_schema": 1, "opponents": sorted(sel),
         "selection": {"opponents": sel, "win_quota": quota_w, "loss_quota": quota_l}}))
    return str(cyc)


def _args(v_column: Optional[str]):
    return argparse.Namespace(no_quota_match=False, quota_match_seeds=5, quota_match_boot=120,
                              cond_boot=120, seed=0, v_column=v_column)


def _side(root, trace_dir):
    return {"run": root.name, "run_dir": str(root), "step": STEP, "trace_dir": trace_dir,
            "conditioning": {"points": {"cond.own_team_r2.t1": 0.1}}}


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("vcol")
    rich_root, poor_root = root / "rich", root / "poor"
    return {"rich_root": rich_root, "poor_root": poor_root,
            "rich_td": _plant(rich_root, quota_w=40, quota_l=40, seed=11),
            "poor_td": _plant(poor_root, quota_w=6, quota_l=9, seed=12)}


def _matched_points(pair, v_column):
    doc = QM.build_quota_match(_side(pair["rich_root"], pair["rich_td"]),
                               _side(pair["poor_root"], pair["poor_td"]),
                               _args(v_column), say=lambda _m: None)
    assert doc is not None and doc.get("rungs"), "the planted pair must actually be matched"
    out = {}
    for side, rungs in doc["rungs"].items():
        for rname, r in rungs.items():
            if not isinstance(r, dict) or "rows" not in r:
                continue          # `poorer_decoder_battles` and friends are scalars
            for k, row in r["rows"].items():
                out[(side, rname, k)] = row["point"]
    return doc, out


# ── THE REGRESSION ────────────────────────────────────────────────────────────────────────────

def test_the_matched_rows_CHANGE_when_the_v_column_does(pair):
    """🚨 On the reverted code these two are byte-identical: `build_quota_match` extracted both
    frames at the default `win_probs` whatever the flag said."""
    doc_wp, wp = _matched_points(pair, "win_probs")
    doc_v, val = _matched_points(pair, "values")
    assert doc_wp["v_column"] == "win_probs" and doc_v["v_column"] == "values"
    assert set(wp) == set(val) and wp, "the two runs must produce the same ROW SET"
    differing = [k for k in wp
                 if not (np.isnan(wp[k]) and np.isnan(val[k])) and wp[k] != val[k]]
    assert differing, (
        "every quota-MATCHED row came out identical across --v-column win_probs and values. "
        "That is the silent-ignore: the matched path extracted its own frame at the default "
        f"while the as-traced table read the other column. rows={sorted(wp)[:6]}")


def test_the_matched_frame_is_extracted_with_the_requested_column(pair, monkeypatch):
    """The column must be applied where the FRAME IS BUILT — `conditioning_block`'s `v_column`
    selects nothing once a frame is injected, so a fix that only threaded it there would have
    changed no number at all."""
    seen = []
    real = CM.extract_cycle

    def _spy(trace_dir, *, v_column=CM.DEFAULT_V_COLUMN):
        seen.append(v_column)
        return real(trace_dir, v_column=v_column)

    monkeypatch.setattr(CM, "extract_cycle", _spy)
    _matched_points(pair, "values")
    assert seen and set(seen) == {"values"}, seen


def test_the_two_paths_read_the_flag_through_ONE_accessor():
    """`critic_read` and `quota_match` both call `v_column_of`, so they cannot disagree."""
    import inspect

    from main.ops import critic_read as CR

    for mod in (CR, QM):
        src = inspect.getsource(mod)
        assert 'getattr(args, "v_column"' not in src, (
            f"{mod.__name__} reads --v-column with its own getattr instead of "
            "conditioning_meters.v_column_of — that is the second opinion this closed.")
    assert CM.v_column_of(_args("values")) == "values"
    assert CM.v_column_of(_args(None)) == CM.DEFAULT_V_COLUMN
    assert CM.v_column_of(argparse.Namespace()) == CM.DEFAULT_V_COLUMN
    with pytest.raises(CM.ConditioningRefusal):
        CM.v_column_of(_args("advantages"))


def test_a_frame_whose_column_DISAGREES_with_the_request_is_REFUSED(pair):
    """The structural guard: a silent no-op becomes a loud refusal naming the fix."""
    arr, meta = CM.extract_cycle(pair["poor_td"], v_column="win_probs")
    with pytest.raises(CM.ConditioningRefusal) as exc:
        CM.conditioning_block(str(pair["poor_root"]), STEP, boot=0, seed=0, ladder="off",
                              frame=(arr, meta), v_column="values")
    msg = str(exc.value)
    assert "INJECTED frame" in msg and "v_column='win_probs'" in msg and "FIX:" in msg
    # and the agreeing case is fine
    CM.conditioning_block(str(pair["poor_root"]), STEP, boot=0, seed=0, ladder="off",
                          frame=(arr, meta), v_column="win_probs")


def test_a_pre_v_column_frame_is_trusted_as_the_DEFAULT_not_refused(pair):
    """A frame built before `meta['v_column']` existed carries no column; it was `win_probs`."""
    arr, meta = CM.extract_cycle(pair["poor_td"])
    legacy = {k: v for k, v in meta.items() if k != "v_column"}
    CM.conditioning_block(str(pair["poor_root"]), STEP, boot=0, seed=0, ladder="off",
                          frame=(arr, legacy), v_column=CM.DEFAULT_V_COLUMN)


def test_the_subsample_carries_the_column_into_every_matched_block(pair):
    """`rung` reads the column OFF THE FRAME, so it agrees with the data by construction even if
    a caller forgets — the property that makes the refusal above unreachable in normal use."""
    arr, meta = CM.extract_cycle(pair["rich_td"], v_column="values")
    sarr, smeta = QM.subsample(arr, meta, (6, 9, CM.STATES_PER_BATTLE_CAP), seed=0)
    assert smeta["v_column"] == "values"
