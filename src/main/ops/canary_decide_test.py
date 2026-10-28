"""``main.ops.canary_decide``: the canary-FATAL resume rule, on synthetic records (no models/, no GPU).

Cases mirror the ones Training Oct tested the job-directory version on (C0 2.581 RESUME / 2.667 HOLD, X47 2.375
RESUME / 2.475 HOLD), plus the holes it had: an empty judgement, a missing field, a typed bar."""
from __future__ import annotations

import json

import pytest

from main.ops import canary_decide as cd

SWITCH0 = "region R1: the compiled arm moved a fixture row's log-prob by up to 0.9 and switched 0 row(s)' PPO clip branch"


def _record(*, verdict="fatal", compiled=0.5, eager=0.25, arms=("scheduled",), update=700):
    watch = {a: {"rule": "bf16", "clip_margin": {"max_abs_dlogp": {"compiled": compiled, "eager": eager},
                                                  "same_precision": {"max_abs_dlogp": 0.3}}} for a in arms}
    return {"verdict": verdict, "update": update, "watch": watch}


def _run(tmp_path, name, records, disagreement=SWITCH0):
    d = tmp_path / name
    d.mkdir()
    (d / "canary_verdicts.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    if disagreement is not None:
        (d / "canary_disagreements.jsonl").write_text(json.dumps({"update": 700, "verdict": disagreement}) + "\n")
    return d


def _go(tmp_path, name, **kw):
    rec = _record(**{k: kw.pop(k) for k in list(kw) if k in ("verdict", "compiled", "eager", "arms")})
    return cd.decide_run(_run(tmp_path, name, [rec], **kw))


def test_the_registered_family_bars():
    assert cd.FAMILY_BARS == {"c0": 2.64, "x47": 2.40}


@pytest.mark.parametrize("name,fam", [("rb_x35t_c0_s3601", "c0"), ("rb_x35_c0_s3505", "c0"),
                                      ("rb_x35t_x47d4_s3602", "x47"), ("rb_x35_d1wide_s3505", "c0"),
                                      ("/a/b/rb_x35t_c0_s3601/", "c0")])
def test_the_family_is_read_from_the_run_name(name, fam):
    assert cd.family_of(name) == fam


#: Every X35 run name that exists under models/ (2026-10-26) plus the planned lrfloor arm: production = C0 (a fork of
#: P* at production pins), toy C0 = C0, toy X47 = X47. X48 (per-kind FFN experts, a P*-fork) takes C0's bar by its
#: PREREG's ruling (x48_kffn_production_2026-10-27 §7). X63-C (the X63 control replicate, X26c's argv from X26 @380M)
#: is production C0's argv and architecture, so it takes C0's bar too.
PRODUCTION_ARMS = ["rb_x35_c0_s3501", "rb_x35_a1arch_s3501", "rb_x35_sprt_s3503", "rb_x35_d1wide_s3505",
                   "rb_x35_d1wide_s3501.crashed-20261022", "rb_x35_x52prac_s3502", "rb_x35_d3mut_s3503",
                   "rb_x35_n1p_s3502", "rb_x35_lrfloor_s3501", "rb_x35_x48kffn_s3501", "rb_x63c_s6302"]


@pytest.mark.parametrize("name", PRODUCTION_ARMS)
def test_every_production_x35_arm_is_judged_at_the_c0_bar(tmp_path, name):
    assert cd.family_of(name) == "c0"
    ok = _go(tmp_path, name, compiled=2.63, eager=1.0)
    held = _go(tmp_path, name + "_b", compiled=2.65, eager=1.0)
    assert ok.verdict == cd.RESUME and ok.bar == 2.64, ok.reasons
    assert held.verdict == cd.HOLD and any("> 2.64" in r for r in held.reasons)


@pytest.mark.parametrize("name", ["rb_x35_r0_s3531", "rb_x35t_r0_s3631", "rb_x36_c0_s3601", "rb_x26_s2601",
                                  "xrb_x35_c0_s3501", "rb_x35_c0", "rb_x35_zzz_s3501", "rb_x35t_a1arch_s3601"])
def test_an_unknown_run_name_holds_with_no_bar(tmp_path, name):
    assert cd.family_of(name) is None
    d = _go(tmp_path, name, compiled=0.1, eager=1.0)
    assert d.verdict == cd.HOLD and d.bar is None and any("no family bar" in r for r in d.reasons)


@pytest.mark.parametrize("name,compiled,verdict", [
    ("rb_x35t_c0_s3601", 2.581, cd.RESUME), ("rb_x35t_c0_s3601", 2.667, cd.HOLD),
    ("rb_x35t_x47d4_s3601", 2.375, cd.RESUME), ("rb_x35t_x47d4_s3601", 2.475, cd.HOLD),
    ("rb_x35t_x47d4_s3601", 2.64, cd.HOLD),          # the C0 bar must NOT leak into X47
    ("rb_x35_c0_s3505", 2.64 * 1.0, cd.RESUME)])     # exactly AT the bar: <=, so a resume (deterministic: 2.64 / 1.0)
def test_the_ratio_is_judged_against_its_own_family_bar(tmp_path, name, compiled, verdict):
    d = _go(tmp_path, name, compiled=compiled, eager=1.0)
    assert d.verdict == verdict, d.reasons
    assert cd.EXIT[d.verdict] == (0 if verdict == cd.RESUME else 3)


def test_a_typed_bar_wins_and_is_stated_as_this_run_only(tmp_path):
    rec = [_record(compiled=2.5, eager=1.0)]
    held = cd.decide_run(_run(tmp_path, "rb_x35t_x47d4_s3601", rec))
    ok = cd.decide_run(_run(tmp_path, "rb_x35_r0_s3531", rec), bar=2.64)
    assert held.verdict == cd.HOLD and ok.verdict == cd.RESUME and "typed --bar 2.64 (this run only)" in ok.bar_source


def test_a_run_of_no_family_and_no_typed_bar_holds_and_says_why(tmp_path):
    d = _go(tmp_path, "rb_x35_r0_s3531")
    assert d.verdict == cd.HOLD and any("no family bar" in r for r in d.reasons)


def test_only_a_FATAL_is_judged(tmp_path):
    d = _go(tmp_path, "rb_x35t_c0_s3601", verdict="pass", compiled=0.1)
    assert d.verdict == cd.HOLD and "not fatal" in d.reasons[0]


def test_any_clip_switch_holds_even_with_a_tiny_ratio(tmp_path):
    d = _go(tmp_path, "rb_x35t_c0_s3601", compiled=0.1, eager=1.0,
            disagreement=SWITCH0.replace("switched 0", "switched 3"))
    assert d.verdict == cd.HOLD and d.clip_switches == [3] and "clip branches switched" in d.reasons[0]


def test_a_missing_switch_count_is_zero_only_in_the_eager_yardstick_class(tmp_path):
    eager = cd.decide_run(_run(tmp_path, "rb_x35t_c0_s3601", [_record()],
                               disagreement=f"--compile-trainer region R1 (bf16 rule): the {cd.EAGER_DRIFT} reference"))
    other = cd.decide_run(_run(tmp_path, "rb_x35t_c0_s3602", [_record()], disagreement="some other class, no count"))
    assert eager.verdict == cd.RESUME and eager.clip_switches == [0] and eager.notes
    assert other.verdict == cd.HOLD and "clip-switch count missing" in other.reasons[0]


def test_a_missing_disagreements_file_and_missing_fields_hold(tmp_path):
    assert cd.decide_run(_run(tmp_path, "rb_x35t_c0_s3601", [_record()], disagreement=None)).verdict == cd.HOLD
    bad = _record()
    del bad["watch"]["scheduled"]["clip_margin"]["max_abs_dlogp"]["eager"]
    d = cd.decide_run(_run(tmp_path, "rb_x35t_c0_s3602", [bad]))
    assert d.verdict == cd.HOLD and "field missing" in d.reasons[0]


def test_one_arm_over_the_bar_holds_the_run_and_a_zero_eager_holds(tmp_path):
    rec = _record(arms=("scheduled", "confirm"))
    rec["watch"]["confirm"]["clip_margin"]["max_abs_dlogp"] = {"compiled": 3.0, "eager": 1.0}
    d = cd.decide_run(_run(tmp_path, "rb_x35t_c0_s3601", [rec]))
    assert d.verdict == cd.HOLD and len(d.reasons) == 1 and d.reasons[0].startswith("confirm:")
    assert _go(tmp_path, "rb_x35t_c0_s3602", eager=0.0).verdict == cd.HOLD


def test_a_record_that_judges_no_arm_holds_instead_of_resuming(tmp_path):
    """The job-directory version RESUMED on an empty judgement (a record with no clip_margin). Revert-proof."""
    d = cd.decide_run(_run(tmp_path, "rb_x35t_c0_s3601", [{"verdict": "fatal", "update": 700, "watch": {"scheduled": {}}}]))
    assert d.verdict == cd.HOLD and "nothing was judged" in d.reasons[-1]


def test_an_unreadable_verdicts_file_holds(tmp_path):
    d = tmp_path / "rb_x35t_c0_s3601"
    d.mkdir()
    (d / "canary_verdicts.jsonl").write_text("{not json\n")
    out = cd.decide_run(d)
    assert out.verdict == cd.HOLD and "unreadable" in out.reasons[0]


def test_only_the_LAST_record_counts(tmp_path):
    run = _run(tmp_path, "rb_x35t_c0_s3601", [_record(compiled=9.0, eager=1.0, update=600), _record(update=700)])
    assert cd.decide_run(run).verdict == cd.RESUME


def test_the_cli_exit_codes_the_log_and_the_text(tmp_path, capsys):
    run = _run(tmp_path, "rb_x35t_x47d4_s3601", [_record(compiled=2.0, eager=1.0)])
    log = tmp_path / "decisions.jsonl"
    assert cd.main([str(run), "--log", str(log)]) == 0
    assert "RESUME" in capsys.readouterr().out
    assert cd.main([str(run), "--bar", "1.5", "--log", str(log), "--json"]) == 3
    rows = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert [r["verdict"] for r in rows] == ["RESUME", "HOLD"] and rows[1]["bar"] == 1.5
    assert json.loads(capsys.readouterr().out)["verdict"] == "HOLD"
    with pytest.raises(SystemExit):
        cd.main([str(run), "--family", "c0", "--bar", "1"])      # a family and a typed bar are exclusive
