"""Unit tests for the best-response-gap engine — synthetic run dirs, no models, no battles."""
from __future__ import annotations

import json
import os

import pytest

from agents.training import best_response_gap as brg
from agents.training.best_response_gap import (
    RunReadError, SeriesError, UnmatchedBudgetError, UnmatchedDoseError, UnmatchedRegimeError,
)

TEAM_TEXT = "Zapdos @ Leftovers\nAbility: Pressure\n- Thunderbolt\n"

TEAMSETS = {
    "offense": {"anchor": "aaaa1111", "hashes": ["aaaa1111", "aaaa2222"]},
    "stall": {"anchor": "bbbb1111", "hashes": ["bbbb1111", "bbbb2222"]},
}


def _teamsets_file(tmp_path) -> str:
    path = tmp_path / "teamsets.json"
    path.write_text(json.dumps(TEAMSETS))
    return str(path)


def _team_files(tmp_path, stems):
    out = []
    d = tmp_path / "teams"
    d.mkdir(exist_ok=True)
    for stem in stems:
        f = d / f"{stem}.txt"
        f.write_text(TEAM_TEXT)
        out.append(str(f))
    return out


def make_run(tmp_path, name, *, target_run, target_step, fork_step, num_timesteps,
             cycles, team_stems, lr=2.5e-4, batch_size=2048, grad_accum=32, n_epochs=10,
             bot_fraction=0.5, external=None, role="exploiter", with_target=True):
    """One synthetic exploiter run directory: metadata.json + sidecars + eval_results.jsonl.

    ``cycles`` is ``[(step, wins, games), …]``; ``external`` overrides the ``ext_*`` label so a
    target/series disagreement can be exercised.
    """
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    teams = _team_files(tmp_path, team_stems)
    target_block = {
        "path": f"models/{target_run}/final_model.zip",
        "resolved_path": f"/models/{target_run}/final_model.zip",
        "run_dir": f"/models/{target_run}", "run_name": target_run,
        "resolved_file": f"/models/{target_run}/final_model.zip",
        "resolved_num_timesteps": target_step, "num_timesteps": target_step,
        "resolution_rung": "explicit_zip", "resolution_rule": "explicit_zip",
    }
    meta = {
        "num_timesteps": num_timesteps,
        "cli_args": {"trainee_teams": ",".join(teams), "eval_sentinel_greedy": True,
                     "exploiter_keep_bots": True, "exploiter_bot_fraction": bot_fraction,
                     "exploiter_temp_mode": "fixed", "exploiter_temp_end": 1.0},
        "lineage": {"schema": 1, "role": role, "fork_step": fork_step,
                    "fork_parent": target_block, "teachers": [],
                    "exploiter_target": target_block if with_target else None,
                    "ancestry": []},
    }
    (run / "metadata.json").write_text(json.dumps(meta))
    for i, (step, _w, _g) in enumerate(cycles):
        (run / "checkpoints" / f"checkpoint_{step}_steps.json").write_text(json.dumps(
            {"num_timesteps": step, "lr": lr, "batch_size": batch_size,
             "grad_accum_steps": grad_accum, "n_epochs": n_epochs}))
    label = external or f"ext_{target_run}"
    with open(run / "eval_results.jsonl", "w") as fh:
        for step, w, g in cycles:
            fh.write(json.dumps({"step": step, "n_games": g, "bots": {},
                                 "externals": {label: {"win_rate": w / g, "counts": [w, g]}}})
                     + "\n")
    return str(run)


# ------------------------------------------------------------------ reading one run

def test_reads_target_budget_dose_and_series(tmp_path):
    d = make_run(tmp_path, "r1", target_run="gen1", target_step=75_000_000,
                 fork_step=75_000_000, num_timesteps=83_000_000,
                 cycles=[(76_000_000, 64, 100), (82_000_000, 74, 100)],
                 team_stems=["aaaa1111"])
    run = brg.read_exploiter(d, TEAMSETS)
    assert run.target_run == "gen1"
    assert run.target_step == 75_000_000
    assert run.budget == 8_000_000
    assert run.archetype == "offense" and run.membership == "1/2"
    assert [p.rate for p in run.series] == [0.64, 0.74]
    assert run.endpoint().rate == 0.74
    assert run.pooled() == (138, 200)
    # dose = lr_median * n_epochs / (batch * grad_accum)
    assert run.dose_rate == pytest.approx(2.5e-4 * 10 / (2048 * 32))


def test_pre_fork_rows_are_the_parents_and_are_excluded(tmp_path):
    """A row at or below fork_step belongs to the PARENT. Crediting it to the best responder
    would report the parent's rate as a best-response gap."""
    d = make_run(tmp_path, "r1", target_run="gen1", target_step=75_000_000,
                 fork_step=75_000_000, num_timesteps=83_000_000,
                 cycles=[(74_000_000, 50, 100), (75_000_000, 51, 100), (82_000_000, 74, 100)],
                 team_stems=["aaaa1111"])
    run = brg.read_exploiter(d, TEAMSETS)
    assert [p.post_fork for p in run.series] == [False, False, True]
    assert run.pooled() == (74, 100)
    assert run.endpoint().step == 82_000_000


def test_all_rows_pre_fork_refuses(tmp_path):
    d = make_run(tmp_path, "r1", target_run="gen1", target_step=75_000_000,
                 fork_step=90_000_000, num_timesteps=99_000_000,
                 cycles=[(76_000_000, 64, 100)], team_stems=["aaaa1111"])
    with pytest.raises(SeriesError, match="PARENT"):
        brg.read_exploiter(d, TEAMSETS)


def test_series_naming_a_different_external_refuses(tmp_path):
    """The recorded target and the evaluated external must be the SAME opponent — reading
    whichever ext_* happens to be present would silently measure something else."""
    d = make_run(tmp_path, "r1", target_run="gen1", target_step=75_000_000,
                 fork_step=75_000_000, num_timesteps=83_000_000,
                 cycles=[(82_000_000, 74, 100)], team_stems=["aaaa1111"],
                 external="ext_someone_else")
    with pytest.raises(SeriesError, match="ext_someone_else"):
        brg.read_exploiter(d, TEAMSETS)


def test_a_non_exploiter_run_refuses(tmp_path):
    d = make_run(tmp_path, "r1", target_run="gen1", target_step=75_000_000,
                 fork_step=75_000_000, num_timesteps=83_000_000,
                 cycles=[(82_000_000, 74, 100)], team_stems=["aaaa1111"],
                 role="fork", with_target=False)
    with pytest.raises(RunReadError, match="not an exploiter run"):
        brg.read_exploiter(d, TEAMSETS)


def test_missing_run_dir_refuses_by_name(tmp_path):
    with pytest.raises(RunReadError, match="no such run directory"):
        brg.read_exploiter(str(tmp_path / "nope"), TEAMSETS)


# ------------------------------------------------------------------ archetypes / grouping

def test_archetype_membership_and_unassigned():
    assert brg.archetype_of(["x/aaaa1111.txt"], TEAMSETS) == ("offense", "1/2")
    assert brg.archetype_of(["x/aaaa1111.txt", "x/aaaa2222.txt"], TEAMSETS) == ("offense", "2/2")
    name, why = brg.archetype_of(["x/aaaa1111.txt", "x/bbbb1111.txt"], TEAMSETS)
    assert name is None and "none" in why
    assert brg.archetype_of([], TEAMSETS)[0] is None


def _two_rounds(tmp_path):
    """Round 1 (target @75M) and round 2 (target @95M), two archetypes each, matched dose."""
    runs = []
    for step, fork, end, tag, cyc in (
            (75_000_000, 75_000_000, 83_000_000, "gen1",
             {"offense": (74, 100), "stall": (72, 100)}),
            (95_000_000, 95_000_000, 103_000_000, "gen2",
             {"offense": (66, 100), "stall": (46, 100)})):
        for arch, (w, g) in cyc.items():
            stem = "aaaa1111" if arch == "offense" else "bbbb1111"
            runs.append(make_run(tmp_path, f"{tag}_{arch}", target_run=tag, target_step=step,
                                 fork_step=fork, num_timesteps=end,
                                 cycles=[(end - 1_000_000, w, g)], team_stems=[stem]))
    return [brg.read_exploiter(d, TEAMSETS) for d in runs]


def test_rounds_are_inferred_from_the_targets_step(tmp_path):
    runs = _two_rounds(tmp_path)
    rounds = brg.assign_rounds(runs)
    by_run = {r.run: rounds[brg.target_key(r)] for r in runs}
    assert by_run == {"gen1_offense": 1, "gen1_stall": 1, "gen2_offense": 2, "gen2_stall": 2}


def test_rounds_override_wins(tmp_path):
    runs = _two_rounds(tmp_path)
    rounds = brg.assign_rounds(runs, {"gen2": 7})
    assert {rounds[brg.target_key(r)] for r in runs if r.target_run == "gen2"} == {7}


def test_target_key_separates_two_steps_of_one_run(tmp_path):
    """A bare run dir resolves to the run's LAST SNAPSHOT, which moves; the FILE is the identity."""
    a = brg.read_exploiter(make_run(tmp_path, "a", target_run="gen", target_step=10,
                                    fork_step=10, num_timesteps=20,
                                    cycles=[(15, 6, 10)], team_stems=["aaaa1111"]), TEAMSETS)
    b = brg.read_exploiter(make_run(tmp_path, "b", target_run="gen", target_step=30,
                                    fork_step=30, num_timesteps=40,
                                    cycles=[(35, 6, 10)], team_stems=["bbbb1111"]), TEAMSETS)
    assert brg.target_key(a) != brg.target_key(b)
    assert brg.assign_rounds([a, b])[brg.target_key(b)] == 2


# ------------------------------------------------------------------ the matched gate

def test_matched_rounds_pass(tmp_path):
    assert brg.check_matched(_two_rounds(tmp_path)) == []


def test_dose_mismatch_refuses(tmp_path):
    a = brg.read_exploiter(make_run(tmp_path, "a", target_run="g1", target_step=10, fork_step=10,
                                    num_timesteps=20, cycles=[(15, 7, 10)],
                                    team_stems=["aaaa1111"], lr=2.5e-4), TEAMSETS)
    b = brg.read_exploiter(make_run(tmp_path, "b", target_run="g2", target_step=30, fork_step=30,
                                    num_timesteps=40, cycles=[(35, 5, 10)],
                                    team_stems=["bbbb1111"], lr=5.5e-5), TEAMSETS)
    with pytest.raises(UnmatchedDoseError, match="4.5[0-9]x apart"):
        brg.check_matched([a, b])
    found = brg.check_matched([a, b], allow_unmatched=True)
    assert [m.kind for m in found] == ["dose"]
    assert "dose" in found[0].message().lower()


def test_budget_mismatch_refuses(tmp_path):
    a = brg.read_exploiter(make_run(tmp_path, "a", target_run="g1", target_step=10, fork_step=10,
                                    num_timesteps=1_000_010, cycles=[(500_000, 7, 10)],
                                    team_stems=["aaaa1111"]), TEAMSETS)
    b = brg.read_exploiter(make_run(tmp_path, "b", target_run="g2", target_step=30, fork_step=30,
                                    num_timesteps=2_000_030, cycles=[(1_000_000, 5, 10)],
                                    team_stems=["bbbb1111"]), TEAMSETS)
    with pytest.raises(UnmatchedBudgetError, match="post-fork steps"):
        brg.check_matched([a, b])


def test_regime_mismatch_refuses(tmp_path):
    a = brg.read_exploiter(make_run(tmp_path, "a", target_run="g1", target_step=10, fork_step=10,
                                    num_timesteps=20, cycles=[(15, 7, 10)],
                                    team_stems=["aaaa1111"], bot_fraction=0.5), TEAMSETS)
    b = brg.read_exploiter(make_run(tmp_path, "b", target_run="g2", target_step=30, fork_step=30,
                                    num_timesteps=40, cycles=[(35, 5, 10)],
                                    team_stems=["bbbb1111"], bot_fraction=0.9), TEAMSETS)
    with pytest.raises(UnmatchedRegimeError, match="exploiter_bot_fraction"):
        brg.check_matched([a, b])


def test_cycle_sample_size_is_part_of_the_regime(tmp_path):
    a = brg.read_exploiter(make_run(tmp_path, "a", target_run="g1", target_step=10, fork_step=10,
                                    num_timesteps=20, cycles=[(15, 70, 100)],
                                    team_stems=["aaaa1111"]), TEAMSETS)
    b = brg.read_exploiter(make_run(tmp_path, "b", target_run="g2", target_step=30, fork_step=30,
                                    num_timesteps=40, cycles=[(35, 20, 40)],
                                    team_stems=["bbbb1111"]), TEAMSETS)
    with pytest.raises(UnmatchedRegimeError, match="cycle_games"):
        brg.check_matched([a, b])


# ------------------------------------------------------------------ statistics

def test_newcombe_brackets_the_point_and_is_asymmetric_near_the_edge():
    d, lo, hi = brg.newcombe_diff_ci(74, 100, 50, 100)
    assert lo < d < hi
    assert d == pytest.approx(0.24)
    # at the boundary a Wald interval would run past 1; Newcombe's does not.
    d2, lo2, hi2 = brg.newcombe_diff_ci(100, 100, 50, 100)
    assert hi2 <= 1.0 + 1e-9 and lo2 < d2


def test_paired_delta_point_is_the_mean_of_the_pairs():
    pairs = [((66, 100), (74, 100)), ((46, 100), (72, 100))]
    point, lo, hi = brg.paired_delta_ci(pairs, draws=4000)
    assert point == pytest.approx(((0.66 - 0.74) + (0.46 - 0.72)) / 2)
    assert lo < point < hi


def test_paired_delta_refuses_a_single_pair():
    assert brg.paired_delta_ci([((6, 10), (7, 10))]) == (None, None, None)


def test_paired_delta_is_wider_than_the_within_cell_interval_alone():
    """The outer level (resampling archetypes) is what a pooled binomial CI would omit; a
    two-archetype delta with a big spread must not read as tight as a 200-battle binomial."""
    spread = brg.paired_delta_ci([((90, 100), (50, 100)), ((10, 100), (50, 100))], draws=4000)
    tight = brg.paired_delta_ci([((50, 100), (50, 100)), ((50, 100), (50, 100))], draws=4000)
    assert (spread[2] - spread[1]) > (tight[2] - tight[1])


# ------------------------------------------------------------------ the report

def test_report_gap_delta_and_verdict(tmp_path):
    runs = _two_rounds(tmp_path)
    doc = brg.build_report(runs, stat="pooled", draws=4000)
    assert [b["round"] for b in doc["rounds"]] == [1, 2]
    r1 = {row["archetype"]: row for row in doc["rounds"][0]["rows"]}
    assert r1["offense"]["gap"] == pytest.approx(0.24)
    assert r1["offense"]["gap_lo"] < r1["offense"]["gap"] < r1["offense"]["gap_hi"]
    d = doc["deltas"][0]
    assert d["earlier_round"] == 1 and d["later_round"] == 2 and d["n_pairs"] == 2
    per = {p["archetype"]: p for p in d["per_archetype"]}
    assert per["offense"]["delta"] == pytest.approx(-0.08)
    assert per["stall"]["delta"] == pytest.approx(-0.26)
    assert d["mean_delta"] == pytest.approx(-0.17)
    assert d["hi"] < 0 and "GAP FELL" in d["verdict"]


def test_report_not_detected_when_the_interval_straddles(tmp_path):
    runs = _two_rounds(tmp_path)
    for r in runs:                      # flatten both rounds onto the same rate
        r.series = [brg.SeriesPoint(step=p.step, wins=50, games=100, post_fork=True)
                    for p in r.series]
    doc = brg.build_report(runs, stat="pooled", draws=4000)
    assert "NOT DETECTED" in doc["deltas"][0]["verdict"]


def test_endpoint_stat_uses_the_last_cycle_only(tmp_path):
    d = make_run(tmp_path, "r1", target_run="gen1", target_step=10, fork_step=10,
                 num_timesteps=20, cycles=[(12, 10, 100), (15, 74, 100)],
                 team_stems=["aaaa1111"])
    run = brg.read_exploiter(d, TEAMSETS)
    doc = brg.build_report([run], stat="endpoint")
    row = doc["rounds"][0]["rows"][0]
    assert (row["wins"], row["games"]) == (74, 100)
    assert row["pooled_rate"] == pytest.approx(0.42)
    assert doc["rounds"][0]["rows"][0]["gap"] == pytest.approx(0.24)


def test_report_carries_the_mismatches_and_the_teamset_size_caveat(tmp_path):
    """A permitted-but-unmatched comparison must carry its confounds INTO the artifact."""
    a = brg.read_exploiter(make_run(tmp_path, "a", target_run="g1", target_step=10, fork_step=10,
                                    num_timesteps=20, cycles=[(15, 74, 100)],
                                    team_stems=["aaaa1111"], lr=2.5e-4), TEAMSETS)
    b = brg.read_exploiter(make_run(tmp_path, "b", target_run="g2", target_step=30, fork_step=30,
                                    num_timesteps=40, cycles=[(35, 66, 100)],
                                    team_stems=["aaaa1111", "aaaa2222"], lr=5.5e-5), TEAMSETS)
    found = brg.check_matched([a, b], allow_unmatched=True)
    doc = brg.build_report([a, b], mismatches=found, draws=2000)
    assert doc["unmatched"] is True
    assert any("dose" == m["kind"] for m in doc["mismatches"])
    assert any("TEAMSET SIZE changed" in c for c in doc["caveats"])


def test_unassigned_archetype_is_reported_and_not_paired(tmp_path):
    a = brg.read_exploiter(make_run(tmp_path, "a", target_run="g1", target_step=10, fork_step=10,
                                    num_timesteps=20, cycles=[(15, 74, 100)],
                                    team_stems=["aaaa1111", "bbbb1111"]), TEAMSETS)
    doc = brg.build_report([a])
    assert doc["rounds"][0]["unassigned"][0]["run"] == "a"
    assert doc["rounds"][0]["rows"][0]["archetype"] is None


def test_bad_stat_refuses(tmp_path):
    with pytest.raises(brg.BestResponseGapError, match="pooled"):
        brg.build_report(_two_rounds(tmp_path), stat="median")


def test_default_teamsets_file_is_committed_and_loads():
    sets = brg.load_teamsets()
    assert set(sets) == {"offense", "balance", "stall"}
    for name, block in sets.items():
        assert len(block["hashes"]) == 5, name
    assert os.path.isfile(str(brg.DEFAULT_TEAMSETS))


def test_malformed_teamsets_refuse(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"offense": {"anchor": "x"}}))
    with pytest.raises(RunReadError, match="hashes"):
        brg.load_teamsets(str(bad))


# ------------------------------------------------------------------ the CLI

def test_cli_end_to_end_on_the_fixture(tmp_path, capsys):
    from main import best_response_gap as cli
    runs = []
    for step, fork, end, tag, arch, w in ((75_000_000, 75_000_000, 83_000_000, "gen1",
                                           "aaaa1111", 74),
                                          (75_000_000, 75_000_000, 83_000_000, "gen1",
                                           "bbbb1111", 72),
                                          (95_000_000, 95_000_000, 103_000_000, "gen2",
                                           "aaaa1111", 66),
                                          (95_000_000, 95_000_000, 103_000_000, "gen2",
                                           "bbbb1111", 46)):
        runs.append(make_run(tmp_path, f"{tag}_{arch}", target_run=tag, target_step=step,
                             fork_step=fork, num_timesteps=end,
                             cycles=[(end - 1_000_000, w, 100)], team_stems=[arch]))
    out = str(tmp_path / "out.json")
    rc = cli.main([*runs, "--teamsets", _teamsets_file(tmp_path), "--json", out, "--draws", "2000"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "THE GAP FELL" in printed
    assert "greedy-vs-greedy" in printed
    doc = json.load(open(out))
    assert doc["deltas"][0]["mean_delta"] == pytest.approx(-0.17)
    assert doc["_meta"]["tool"] == "main.best_response_gap"


def test_cli_exits_2_on_a_refusal(tmp_path, capsys):
    from main import best_response_gap as cli
    a = make_run(tmp_path, "a", target_run="g1", target_step=10, fork_step=10, num_timesteps=20,
                 cycles=[(15, 74, 100)], team_stems=["aaaa1111"], lr=2.5e-4)
    b = make_run(tmp_path, "b", target_run="g2", target_step=30, fork_step=30, num_timesteps=40,
                 cycles=[(35, 66, 100)], team_stems=["bbbb1111"], lr=5.5e-5)
    rc = cli.main([a, b, "--teamsets", _teamsets_file(tmp_path), "--no-json"])
    assert rc == 2
    assert "unmatched_dose" in capsys.readouterr().err


def test_cli_check_mode_plays_and_writes_nothing(tmp_path):
    from main import best_response_gap as cli
    d = make_run(tmp_path, "a", target_run="g1", target_step=10, fork_step=10, num_timesteps=20,
                 cycles=[(15, 74, 100)], team_stems=["aaaa1111"])
    ts = _teamsets_file(tmp_path)
    before = set(os.listdir(tmp_path))
    assert cli.main([d, "--teamsets", ts, "--check", "--quiet"]) == 0
    assert set(os.listdir(tmp_path)) == before


def test_cli_rounds_parse_error(tmp_path):
    from main import best_response_gap as cli
    with pytest.raises(brg.BestResponseGapError, match="NAME=ROUND"):
        cli.parse_rounds(["gen2"])
    with pytest.raises(brg.BestResponseGapError, match="not an integer"):
        cli.parse_rounds(["gen2=late"])


def test_play_refuses_a_run_with_no_pinned_teams(tmp_path):
    d = make_run(tmp_path, "a", target_run="g1", target_step=10, fork_step=10, num_timesteps=20,
                 cycles=[(15, 74, 100)], team_stems=["aaaa1111"])
    run = brg.read_exploiter(d, TEAMSETS)
    run.teams = []
    with pytest.raises(brg.BestResponseGapError, match="NO pinned trainee teams"):
        brg.play_head_to_head(run, games=4)


def test_play_refuses_when_the_recorded_target_file_is_gone(tmp_path):
    d = make_run(tmp_path, "a", target_run="g1", target_step=10, fork_step=10, num_timesteps=20,
                 cycles=[(15, 74, 100)], team_stems=["aaaa1111"])
    run = brg.read_exploiter(d, TEAMSETS)
    with pytest.raises(brg.BestResponseGapError, match="not on disk"):
        brg.play_head_to_head(run, games=4)


def test_play_refuses_nonpositive_games(tmp_path):
    d = make_run(tmp_path, "a", target_run="g1", target_step=10, fork_step=10, num_timesteps=20,
                 cycles=[(15, 74, 100)], team_stems=["aaaa1111"])
    with pytest.raises(brg.BestResponseGapError, match="must be positive"):
        brg.play_head_to_head(brg.read_exploiter(d, TEAMSETS), games=0)


# ------------------------------------------------------------------ REPLICATES (finding F3)
#
# Population-loop round 1 has two offense exploiters of ONE target: arm A and its seed-1002 twin
# A2. Until 2026-09-23 the round-over-round delta was keyed on archetype in a dict comprehension,
# so the later-sorted run silently REPLACED the other (demonstrated in
# designs/research_state/measurements/population_loop_r1_2026-09-23/validation/
# brgap_standin_same_round_collapse.txt). Each test below fails on that code.

def _replicate_fixture(tmp_path, *, a2_wins=70, a2_lr=2.5e-4, a2_stems=("aaaa1111",),
                       a2_target=("gen1", 75_000_000)):
    """Round 1: ``ai_A`` (64/100) and ``ai_A2`` of the SAME target; round 2: one reader."""
    a = make_run(tmp_path, "ai_A", target_run="gen1", target_step=75_000_000,
                 fork_step=75_000_000, num_timesteps=83_000_000,
                 cycles=[(82_000_000, 64, 100)], team_stems=["aaaa1111"])
    a2 = make_run(tmp_path, "ai_A2", target_run=a2_target[0], target_step=a2_target[1],
                  fork_step=a2_target[1], num_timesteps=a2_target[1] + 8_000_000,
                  cycles=[(a2_target[1] + 7_000_000, a2_wins, 100)],
                  team_stems=list(a2_stems), lr=a2_lr)
    rd = make_run(tmp_path, "gen2_reader", target_run="gen2", target_step=95_000_000,
                  fork_step=95_000_000, num_timesteps=103_000_000,
                  cycles=[(102_000_000, 66, 100)], team_stems=["aaaa1111"])
    return [a, a2, rd]


def test_same_target_same_archetype_exploiters_are_pooled_replicates_not_collapsed(tmp_path):
    runs = [brg.read_exploiter(d, TEAMSETS) for d in _replicate_fixture(tmp_path)]
    doc = brg.build_report(runs, stat="pooled", draws=2000)
    r1 = doc["rounds"][0]
    # both replicates keep their own row, labelled
    reps = {row["run"]: row.get("replicate") for row in r1["rows"]}
    assert reps == {"ai_A": "1/2", "ai_A2": "2/2"}
    # ... and the round gains ONE pooled row: 64 + 70 over 100 + 100
    assert len(r1["pooled"]) == 1
    pooled = r1["pooled"][0]
    assert (pooled["wins"], pooled["games"]) == (134, 200)
    assert pooled["replicates"] == ["ai_A", "ai_A2"]
    assert pooled["gap"] == pytest.approx(0.17)
    # the delta reads the POOLED cell — not A2's +0.20 (the old silent winner), nor A's +0.14
    off = doc["deltas"][0]["per_archetype"][0]
    assert off["archetype"] == "offense"
    assert off["gap_earlier"] == pytest.approx(0.17)
    assert off["delta"] == pytest.approx(0.66 - 0.67)
    assert off["runs_earlier"] == ["ai_A", "ai_A2"] and off["runs_later"] == ["gen2_reader"]
    # ... beside a per-replicate delta for EACH replicate
    per = {q["earlier_run"]: q for q in off["per_replicate"]}
    assert set(per) == {"ai_A", "ai_A2"}
    assert per["ai_A"]["delta"] == pytest.approx(0.02)
    assert per["ai_A2"]["delta"] == pytest.approx(-0.04)
    # the between-replicate spread (the design's reader floor) is carried, with its interval
    spread = pooled["replicate_spread"]
    assert (spread["a"], spread["b"]) == ("ai_A", "ai_A2")
    assert spread["delta"] == pytest.approx(0.06) and spread["lo"] < 0 < spread["hi"]
    assert any("2 REPLICATES pooled" in c and "ai_A2" in c for c in doc["caveats"])
    assert not any("DISAGREE" in c for c in doc["caveats"])


def test_replicates_that_disagree_beyond_binomial_noise_are_flagged(tmp_path):
    runs = [brg.read_exploiter(d, TEAMSETS)
            for d in _replicate_fixture(tmp_path, a2_wins=92)]
    doc = brg.build_report(runs, draws=2000)
    spread = doc["rounds"][0]["pooled"][0]["replicate_spread"]
    assert spread["lo"] > 0
    assert any("DISAGREE" in c and "UNDERSTATES" in c for c in doc["caveats"])


def test_same_round_same_archetype_different_targets_refuses_naming_both(tmp_path):
    """A --rounds override can put two DIFFERENT generalists in one round; one archetype cell
    cannot hold both, and pooling them would average two different quantities."""
    runs = [brg.read_exploiter(d, TEAMSETS)
            for d in _replicate_fixture(tmp_path, a2_target=("genX", 76_000_000))]
    rounds = brg.assign_rounds(runs, {"gen1": 1, "genX": 1, "gen2": 2})
    with pytest.raises(brg.ReplicateCollisionError, match="DIFFERENT generalists") as ei:
        brg.build_report(runs, rounds=rounds, draws=500)
    assert "ai_A (target" in str(ei.value) and "ai_A2 (target" in str(ei.value)
    with pytest.raises(brg.ReplicateCollisionError):
        brg.replicate_groups(runs, rounds)


def test_same_target_different_teamset_sizes_are_not_replicates(tmp_path):
    runs = [brg.read_exploiter(d, TEAMSETS)
            for d in _replicate_fixture(tmp_path, a2_stems=("aaaa1111", "aaaa2222"))]
    with pytest.raises(brg.ReplicateCollisionError, match="teamset sizes"):
        brg.build_report(runs, draws=500)


def test_matched_refusals_still_fire_between_replicates(tmp_path):
    """Pooling replicates must not bypass the matched gate: a replicate at a different DOSE is
    refused exactly like a cross-round comparison is."""
    runs = [brg.read_exploiter(d, TEAMSETS)
            for d in _replicate_fixture(tmp_path, a2_lr=5.5e-5)]
    with pytest.raises(UnmatchedDoseError, match="ai_A vs ai_A2"):
        brg.check_matched(runs)
    # and a matched replicate set passes the gate
    ok_dir = tmp_path / "ok"
    ok_dir.mkdir()
    ok = [brg.read_exploiter(d, TEAMSETS) for d in _replicate_fixture(ok_dir)]
    assert brg.check_matched(ok) == []


def test_cli_prints_both_replicates_and_the_pooled_row(tmp_path, capsys):
    from main import best_response_gap as cli
    dirs = _replicate_fixture(tmp_path)
    out = str(tmp_path / "out.json")
    rc = cli.main([*dirs, "--teamsets", _teamsets_file(tmp_path), "--json", out,
                   "--draws", "500"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "offense [rep 1/2]" in printed and "offense [rep 2/2]" in printed
    assert "offense POOLED" in printed
    assert "└ gen2_reader − ai_A " in printed and "└ gen2_reader − ai_A2" in printed
    assert "REPLICATES pooled" in printed
    doc = json.load(open(out))
    assert doc["deltas"][0]["per_archetype"][0]["gap_earlier"] == pytest.approx(0.17)
    md = cli.render_markdown(doc)
    assert "offense POOLED" in md and "`ai_A2`" in md


def test_cli_exits_2_on_a_replicate_collision(tmp_path, capsys):
    from main import best_response_gap as cli
    dirs = _replicate_fixture(tmp_path, a2_target=("genX", 76_000_000))
    argv = [*dirs, "--teamsets", _teamsets_file(tmp_path), "--rounds", "gen1=1", "genX=1",
            "gen2=2"]
    assert cli.main([*argv, "--no-json"]) == 2
    assert "replicate_collision" in capsys.readouterr().err
    assert cli.main([*argv, "--check", "--quiet"]) == 2      # --check reads the same gate


def test_cli_exits_2_on_an_unmatched_replicate(tmp_path, capsys):
    from main import best_response_gap as cli
    dirs = _replicate_fixture(tmp_path, a2_lr=5.5e-5)
    assert cli.main([*dirs, "--teamsets", _teamsets_file(tmp_path), "--no-json"]) == 2
    assert "unmatched_dose" in capsys.readouterr().err
