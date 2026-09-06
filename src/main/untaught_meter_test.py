"""CLI tests for ``python -m main.untaught_meter`` — resolution, ``--check``, and the BANKED
cross-check that proves the aggregation reproduces a number already in the ledger.

Nothing here plays a battle. The battle path's gate is
``untaught_meter_reproducibility_integration_test.py`` (marked ``sim``).
"""
from __future__ import annotations

import json
import os

import pytest

from agents.training import untaught_meter as engine
from main import untaught_meter as cli

BANKED_DIR = engine.repo_path(
    "designs/research_state/measurements/teacher_content_2x2_2026-09-04")


# ---------------------------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------------------------

def test_every_help_string_renders():
    """A single unescaped ``%`` in a help string raises at render time and nothing else reads them
    (the ``checkargs`` lesson: ``--help`` was itself broken by one ``"~0.6% of"``)."""
    text = cli.build_parser().format_help()
    assert "--baseline" in text and "--control" in text and "--from-rows" in text


def test_a_repeated_control_flag_accumulates_rather_than_overwriting():
    """``nargs='+'`` with the DEFAULT store action silently drops the earlier group, which would
    turn a three-arm control into a one-arm control with no floor."""
    args = cli.build_parser().parse_args(["R", "--control", "A", "--control", "B", "C"])
    assert args.control == ["A", "B", "C"]


def test_label_prefix_splits_only_on_a_leading_name():
    assert cli._split_label("FUND=models/x") == ("FUND", "models/x")
    assert cli._split_label("models/a=b/x.zip") == (None, "models/a=b/x.zip")
    assert cli._split_label("plain/ref") == (None, "plain/ref")


def test_duplicate_labels_are_uniquified_rather_than_colliding():
    assert cli._uniquify(["A", "A", "B", "A"]) == ["A", "A#1", "B", "A#2"]


def test_teams_are_sharded_round_robin_and_empty_shards_are_dropped():
    teams = [engine.TeamSlice(i, f"U_{i}", "/x", "s", "p", "t") for i in range(5)]
    shards = cli._shard_teams(teams, 2)
    assert [[t.index for t in s] for s in shards] == [[0, 2, 4], [1, 3]]
    assert len(cli._shard_teams(teams, 99)) == 5          # never more shards than teams


# ---------------------------------------------------------------------------------------------
# --check / --dry-run on a SYNTHETIC tree (no models/, no battles)
# ---------------------------------------------------------------------------------------------

def _fake_run(tmp_path, name, *, arch="sig") -> str:
    d = tmp_path / name
    d.mkdir()
    (d / "final_model.zip").write_bytes(b"not really a zip")
    (d / "model_config.json").write_text(json.dumps({"arch_signature": arch}))
    return str(d)


def _fake_teams(tmp_path, n=2) -> str:
    rels = []
    for i in range(n):
        f = tmp_path / f"team_{i}.txt"
        f.write_text(f"Snorlax @ Leftovers  # {i}\n")
        rels.append(str(f))
    p = tmp_path / "teams.json"
    p.write_text(json.dumps({"untaught": rels}))
    return str(p)


def _base_argv(tmp_path):
    return [_fake_run(tmp_path, "arm"), "--baseline", _fake_run(tmp_path, "parent"),
            "--teams", _fake_teams(tmp_path),
            "--opponent", _fake_run(tmp_path, "opp"), "--config", "auto"]


def test_check_resolves_a_synthetic_tree_and_exits_zero(tmp_path, capsys):
    assert cli.main(_base_argv(tmp_path) + ["--check"]) == 0
    out = capsys.readouterr().out
    assert "every ref, team and opponent resolved — OK" in out
    assert "final_model.zip" in out                       # the RESOLVED FILE is printed per ref
    assert "rung=" in out and "rule=" in out              # …and HOW it was chosen


def test_check_exits_non_zero_on_a_ref_that_does_not_resolve(tmp_path, capsys):
    argv = _base_argv(tmp_path)
    argv[0] = str(tmp_path / "no_such_run")
    assert cli.main(argv + ["--check"]) == 1
    assert "no_such_run" in capsys.readouterr().err


def test_check_exits_non_zero_on_a_missing_team_file(tmp_path, capsys):
    manifest = tmp_path / "bad_teams.json"
    manifest.write_text(json.dumps({"untaught": [str(tmp_path / "gone.txt")]}))
    argv = _base_argv(tmp_path)
    argv[argv.index("--teams") + 1] = str(manifest)
    assert cli.main(argv + ["--check"]) == 1
    assert "gone.txt" in capsys.readouterr().err


def test_check_plays_nothing_even_when_the_models_are_unloadable(tmp_path):
    """The synthetic zips are not real SB3 archives — ``--check`` must never open them."""
    assert cli.main(_base_argv(tmp_path) + ["--check"]) == 0


def test_dry_run_prints_the_battle_budget_and_the_missing_control_warning(tmp_path, capsys):
    assert cli.main(_base_argv(tmp_path) + ["--dry-run", "--games-per-team", "50"]) == 0
    out = capsys.readouterr().out
    assert "battles      200" in out                       # 2 refs x 2 teams x 50
    assert "control      NONE" in out
    assert "nothing played" in out


def test_dry_run_with_controls_names_them_and_drops_the_warning(tmp_path, capsys):
    argv = _base_argv(tmp_path) + ["--control", _fake_run(tmp_path, "c1"),
                                   _fake_run(tmp_path, "c2"), "--dry-run"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.count("control      ") == 2
    assert "control      NONE" not in out


def test_concurrency_above_one_is_refused_before_anything_is_resolved(tmp_path, capsys):
    assert cli.main(_base_argv(tmp_path) + ["--concurrency", "4"]) == 1
    assert "REFUSING concurrency=4" in capsys.readouterr().err


def test_no_refs_is_a_usage_error(capsys):
    assert cli.main(["--baseline", "x"]) == 2
    assert "no refs given" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# --from-rows: THE BANKED CROSS-CHECK
# ---------------------------------------------------------------------------------------------

def _banked(name: str) -> str:
    return str(BANKED_DIR / name)


@pytest.mark.parametrize("leg,expected", [
    (("TCFUNDA", "TCUNFA"), -4.50),          # tc_readout.py: end TCFUNDA-TCUNFA  -4.50
    (("TCFUNDB", "TCUNFB"), -4.25),          # tc_readout.py: end TCFUNDB-TCUNFB  -4.25
])
def test_the_meter_reproduces_the_banked_2x2_endpoint_legs(leg, expected, tmp_path, capsys):
    """The ledger's ``funded − unfunded`` untaught endpoint is −4.37 = the mean of these two legs
    (2026-09-06 · CELL notes / the 2×2 entry). Point estimates are seed-free and must be EXACT."""
    arm, base = leg
    out = tmp_path / "o.json"
    rc = cli.main(["--from-rows", f"ARM={_banked(f'untaught_{arm}_end.json')}",
                   "--baseline", f"BASE={_banked(f'untaught_{base}_end.json')}",
                   "--floor", "1.66", "--quiet", "--json", str(out)])
    assert rc == 0
    doc = json.loads(out.read_text())
    d = doc["result"]["contrasts"][0]["vs_baseline"]
    assert d["delta_pp"] == pytest.approx(expected, abs=0.005)
    assert d["verdict"] == "SIGNIFICANT"
    assert doc["result"]["timeouts"]["inconclusive"] is False


def test_the_two_legs_average_to_the_ledgers_banked_minus_4_37(tmp_path):
    deltas = []
    for arm, base in (("TCFUNDA", "TCUNFA"), ("TCFUNDB", "TCUNFB")):
        out = tmp_path / f"{arm}.json"
        cli.main(["--from-rows", f"ARM={_banked(f'untaught_{arm}_end.json')}",
                  "--baseline", f"BASE={_banked(f'untaught_{base}_end.json')}",
                  "--quiet", "--json", str(out)])
        deltas.append(json.loads(out.read_text())["result"]["contrasts"][0]["vs_baseline"]["delta_pp"])
    assert sum(deltas) / 2 == pytest.approx(-4.37, abs=0.006)


def test_the_control_floor_reproduces_the_banked_UNF_end_replicate_draw(tmp_path):
    """``tc_readout.py`` prints ``UNF/end  +0.06pp`` — the two unfunded arms differ by 0.06pp at
    the endpoint, and that IS the max-pairwise floor of a two-arm control."""
    out = tmp_path / "o.json"
    cli.main(["--from-rows", f"ARM={_banked('untaught_TCFUNDA_end.json')}",
              "--baseline", f"BASE={_banked('untaught_TCUNFA_end.json')}",
              "--control", f"UNF_A={_banked('untaught_TCUNFA_end.json')}",
              f"UNF_B={_banked('untaught_TCUNFB_end.json')}",
              "--quiet", "--json", str(out)])
    doc = json.loads(out.read_text())
    assert doc["result"]["control"]["replicate_floor_pp"] == pytest.approx(0.0625, abs=0.005)


def test_from_rows_ignores_the_POOLED_summary_row_on_the_real_artifacts(tmp_path):
    out = tmp_path / "o.json"
    cli.main(["--from-rows", f"A={_banked('untaught_TCUNFA_end.json')}",
              "--quiet", "--json", str(out)])
    doc = json.loads(out.read_text())
    assert len(doc["result"]["teams"]) == 8
    assert "POOLED" not in doc["result"]["teams"]


def test_from_rows_check_reports_the_shared_team_count(capsys):
    rc = cli.main(["--from-rows", _banked("untaught_TCUNFA_end.json"),
                   "--baseline", _banked("untaught_TCUNFB_end.json"), "--check"])
    assert rc == 0
    assert "8 shared team key(s)" in capsys.readouterr().out


def test_from_rows_check_exits_non_zero_on_a_missing_artifact(capsys):
    assert cli.main(["--from-rows", "/nope/untaught_X.json", "--check"]) == 1
    assert "missing artifact" in capsys.readouterr().err


def test_the_markdown_report_is_written_and_carries_both_columns(tmp_path):
    md = tmp_path / "o.md"
    cli.main(["--from-rows", f"ARM={_banked('untaught_TCFUNDA_end.json')}",
              "--baseline", f"BASE={_banked('untaught_TCUNFA_end.json')}",
              "--control", f"UNF_A={_banked('untaught_TCUNFA_end.json')}",
              f"UNF_B={_banked('untaught_TCUNFB_end.json')}",
              "--quiet", "--md", str(md)])
    text = md.read_text()
    assert "Δ vs baseline" in text and "Δ vs continuation control" in text
    assert "Per-team win rate" in text
    assert os.path.getsize(md) > 500
