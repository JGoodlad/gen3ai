"""The training-fps reader and the pre-registered throttle."""
from pathlib import Path

from main.rust_core_cutover import governor as G


def _table(it, elapsed, steps):
    return (f"| time/          |      |\n|    fps         | 500  |\n|    iterations  | {it}  |\n"
            f"|    time_elapsed | {elapsed} |\n|    total_timesteps | {steps} |\n")


def _log(child, rows, attach=True):
    head = f"===== child attached 2026-09-24 20:47:52 (pid {child}) =====\n" if attach else ""
    return head + "".join(_table(*r) for r in rows)


def test_parse_log_tags_each_child_and_marginals_restart_with_it():
    text = _log(11, [(1, 100, 98304), (2, 300, 196608), (3, 500, 294912), (4, 700, 393216), (5, 896, 491520)])
    text += _log(22, [(1, 50, 98304), (2, 250, 196608), (3, 450, 294912), (4, 650, 393216)])
    its = G.parse_log(text)
    assert [(i.child, i.iteration) for i in its][:2] == [(11, 1), (11, 2)]
    m = G.marginals(its)
    # the first SKIP_FIRST iterations of EACH child are dropped; no pair spans a restart
    assert [(c, e0, e1) for c, e0, e1, _ in m] == [(11, 500, 700), (11, 700, 896), (22, 450, 650)]
    assert abs(m[1][3] - 98304 / 196) < 1e-9


def test_classify_off_on_mixed():
    tl = [(0.0, 0), (100.0, 3), (200.0, 0)]
    assert G.classify(10, 90, tl) == "off"
    assert G.classify(110, 190, tl) == "on"
    assert G.classify(50, 150, tl) == "mixed"
    assert G.classify(250, 300, tl) == "off"
    assert G.classify(10, 20, []) == "off"


def _gov(tmp_path, monkeypatch, run, rows):
    monkeypatch.setattr(G, "live_runs", lambda: {run: 1})
    g = G.Governor(tmp_path)
    g.state[run] = G.RunState(rows=rows)
    return g


def test_the_throttle_cuts_on_a_drop_of_more_than_15_percent_and_never_raises(tmp_path, monkeypatch):
    run = str(tmp_path / "run")
    off = [[1, i, float(i), 500.0, "off"] for i in range(6)]
    on_ok = [[1, 10 + i, 10.0 + i, 430.0, "on"] for i in range(5)]      # 0.86 x: no cut
    g = _gov(tmp_path, monkeypatch, run, off + on_ok)
    assert g.baseline(run) == 500.0 and g.on_reading(run) == 430.0
    assert g.decide(6, now=100.0) == 6
    on_bad = [[1, 20 + i, 20.0 + i, 420.0, "on"] for i in range(5)]     # 0.84 x: cut by 2
    g = _gov(tmp_path, monkeypatch, run, off + on_bad)
    assert g.decide(6, now=100.0) == 4
    assert g.events[-1]["cap_to"] == 4
    # judged again only on ON iterations completed AFTER the cut
    assert g.on_reading(run) is None and g.decide(4, now=101.0) == 4
    # the floor
    g = _gov(tmp_path, monkeypatch, run, off + on_bad)
    assert g.decide(2, now=100.0) == 2


def test_no_baseline_means_no_throttle_and_a_baseline_is_requested(tmp_path, monkeypatch):
    run = str(tmp_path / "run")
    g = _gov(tmp_path, monkeypatch, run, [[1, i, float(i), 100.0, "on"] for i in range(9)])
    assert g.baseline(run) is None
    assert g.decide(6, now=5.0) == 6
    assert g.needs_baseline() == [run]


def test_poll_places_iterations_on_the_wall_clock_from_the_rows_it_saw(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    rows = [(1, 100, 98304), (2, 300, 196608), (3, 500, 294912), (4, 700, 393216), (5, 900, 491520)]
    (run / "launcher_child.log").write_text(_log(7, rows))
    monkeypatch.setattr(G, "live_runs", lambda: {str(run): 1})
    g = G.Governor(tmp_path)
    g.poll(timeline=[(0.0, 0), (10_000.5, 2)], now=10_000.0)   # origin = 10_000 - 900 = 9_100
    st = g.state[str(run)]
    assert st.origin[7] == 9100.0
    assert [(r[1], r[2], r[4]) for r in st.rows] == [(700, 9800.0, "off"), (900, 10000.0, "off")]
    # a row first seen LATE is re-placed once the origin estimate improves
    (run / "launcher_child.log").write_text(_log(7, rows + [(6, 1100, 589824)]))
    g.poll(timeline=[(0.0, 0), (10_000.5, 2)], now=10_150.0)   # row 1100 seen 50 s after 10_100
    assert st.origin[7] == 9050.0
    assert [(r[1], r[2]) for r in st.rows] == [(700, 9750.0), (900, 9950.0), (1100, 10150.0)]
    assert st.rows[-1][4] == "mixed"          # the stress started inside [9950, 10150]
    g.save()
    assert G.Governor(tmp_path).state[str(run)].rows == st.rows


def test_live_runs_reads_only_train_rl_agent_processes_with_a_run_dir():
    for rd, pid in G.live_runs().items():
        assert Path(rd).is_absolute() and pid > 0


def test_rows_of_a_finished_child_are_never_placed(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    old = [(1, 100, 98304), (2, 300, 196608), (3, 500, 294912), (4, 700, 393216)]
    new = [(1, 50, 98304), (2, 250, 196608), (3, 450, 294912), (4, 650, 393216)]
    (run / "launcher_child.log").write_text(_log(7, old) + _log(8, new))
    monkeypatch.setattr(G, "live_runs", lambda: {str(run): 1})
    g = G.Governor(tmp_path)
    g.poll(timeline=[], now=5_000.0)
    assert {r[0] for r in g.state[str(run)].rows} == {8}
    assert set(g.state[str(run)].origin) == {8}
