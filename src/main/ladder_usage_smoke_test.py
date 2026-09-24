"""The full Metamon smoke's INCREMENTAL contract (``designs/ops/ORCHESTRATOR_SOP.md`` §2): the
registered battle list covers every team once, a unit on disk is never re-run, a unit is written
atomically, and the verdict is refused before the registered n."""
import json

from main import ladder_usage_smoke as S


def test_the_pairing_plays_every_team_exactly_once():
    pairs = S.pairing(22862)
    assert len(pairs) == 11431
    flat = [i for p in pairs for i in p]
    assert sorted(flat) == list(range(22862))
    assert S.pairing(22862) == pairs, "seeded"
    assert len(S.pairing(7)) == 3, "an odd team out is dropped"


def test_units_tile_the_registered_battles():
    n, size = 11431, 50
    keys = [k for u in range(S.n_units(n, size)) for k in S.unit_keys(u, n, size)]
    assert keys == list(range(n))


def _fake_src(tmp_path, n=10):
    src = tmp_path / "teams"
    src.mkdir()
    for i in range(n):
        (src / f"team_{i:06d}.gen3ou_team").write_text(f"team {i}\n")
    return src


def test_resume_runs_only_the_units_not_on_disk(tmp_path, monkeypatch):
    """A driver restarted after a kill re-launches ONLY the missing units."""
    src = _fake_src(tmp_path)
    out = tmp_path / "out"
    S.write_registration(out, src, unit_size=1)            # 5 battles -> 5 units
    (out / "units").mkdir()
    for u in (0, 2):                                        # "finished before the kill"
        S.unit_path(out, u).write_text(json.dumps({"key": u}) + "\n")
    launched = []

    class FakeUnit:
        def __init__(self, argv, **kw):
            u = int(argv[argv.index("--unit") + 1])
            launched.append(u)
            S.unit_path(out, u).write_text(json.dumps({"key": u}) + "\n")

        def poll(self):
            return 0

        returncode = 0

    monkeypatch.setattr(S.subprocess, "Popen", FakeUnit)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    S.drive(out, src, workers=2, unit_size=1, units=None, idle_budget_s=1, total_budget_s=1)
    assert sorted(launched) == [1, 3, 4]
    assert S.done_units(out) == {0, 1, 2, 3, 4}
    launched.clear()
    S.drive(out, src, workers=2, unit_size=1, units=None, idle_budget_s=1, total_budget_s=1)
    assert launched == [], "a complete run re-launches nothing"


def test_a_changed_recipe_is_refused(tmp_path):
    src = _fake_src(tmp_path)
    out = tmp_path / "out"
    S.write_registration(out, src, unit_size=1)
    try:
        S.write_registration(out, src, unit_size=2)
    except SystemExit as e:
        assert "different recipe" in str(e)
    else:
        raise AssertionError("a resume with a different unit size must refuse")


def test_the_verdict_is_refused_before_the_registered_n(tmp_path):
    src = _fake_src(tmp_path)
    out = tmp_path / "out"
    S.write_registration(out, src, unit_size=2)
    (out / "units").mkdir()
    row = {"key": 0, "ok": True, "p1_file": "a", "p2_file": "b", "decisions": 3, "finished": True}
    S.unit_path(out, 0).write_text(json.dumps(row) + "\n")
    assert S.main(["report", "--out", str(out), "--src", str(src), "--unit-size", "2"]) == 2
    assert S.main(["status", "--out", str(out), "--src", str(src), "--unit-size", "2"]) == 0


def test_summaries_group_failures_by_class():
    rows = [{"key": 0, "ok": True, "called": {"Metronome": 2}},
            {"key": 1, "ok": False, "stage": "battle", "error_class": "ValueError: x", "error": "x",
             "p1_file": "a", "p2_file": "b"},
            {"key": 2, "ok": False, "stage": "battle", "error_class": "ValueError: x", "error": "y",
             "p1_file": "c", "p2_file": "d"}]
    s = S.summarize(rows)
    assert s["failed"] == 2 and s["failure_classes"] == {"[battle] ValueError: x": 2}
    assert s["called_move_lines"] == {"Metronome": 2}
    assert S.classify(ValueError("Unhandled move message format - ['a'] in battle battle-gen3ou-7 turn 12")) == \
        "ValueError: Unhandled move message format - […] in battle <battle> turn N"


def test_a_unit_file_is_written_atomically(tmp_path, monkeypatch):
    """`run_unit` writes through a temp file + rename, so a killed unit leaves no partial file."""
    src = _fake_src(tmp_path, n=4)
    out = tmp_path / "out"

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(S, "play_battle", boom)
    try:
        S.run_unit(out, 0, src, unit_size=2, idle_budget_s=1, total_budget_s=1)
    except KeyboardInterrupt:
        pass
    assert S.done_units(out) == set()
