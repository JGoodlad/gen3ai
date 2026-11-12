"""BOUNDARY-1's looks: the checkpoint rule, the per-run guards, the cells and the read (``main.b1.look``)."""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from main.b1 import arms as A
from main.b1 import cross as X
from main.b1 import look as K

PIN = "a" * 40
OTHER_PIN = "b" * 40
#: What each arm's argv resolves to on ``K.CONFIG_KEYS`` (pinned against the real resolver in the last test).
LEVERS: Dict[str, Dict[str, Any]] = {
    "C": {"d_model": 128, "team_mix": None, "team_mix_sides": None, "trunk_ffn_dim": 256},
    "W": {"d_model": 256, "team_mix": None, "team_mix_sides": None, "trunk_ffn_dim": 512},
    "Br": {"d_model": 128, "team_mix": "n1_proc_v1@ALL:0.5", "team_mix_sides": "both", "trunk_ffn_dim": 256},
    "WB": {"d_model": 256, "team_mix": "n1_proc_v1@ALL:0.5", "team_mix_sides": "both", "trunk_ffn_dim": 512},
}


def _levers(seeds: Optional[Dict[str, Tuple[int, ...]]] = None) -> Dict[Tuple[str, int], Dict[str, Any]]:
    seeds = seeds or K.design_seeds()
    return {(arm, s): LEVERS[arm] for arm in X.ARMS for s in seeds[arm]}


#: The module's own K9(b) reader (TensorBoard rows + the child log), before the fixture below stands in for it.
REAL_K9B_READ = K.k9b_read
K9B_STUB = "k9b_stub.json"


def _stub_k9b_read(run_dir: Path, upto: Optional[int]) -> Dict[str, Any]:
    """The K9(b) record ``make_run`` planted (``k9b=``), else a PASS: the guard runs on every plan and check here
    without an event file a run; the real reader has its own tests below."""
    rec = {"schema": K.K9B_SCHEMA, "run": run_dir.name, "verdict": "PASS", "upto": upto, "fires": [], "reasons": []}
    planted = run_dir / K9B_STUB
    return {**rec, **(json.loads(planted.read_text()) if planted.is_file() else {})}


@pytest.fixture(autouse=True)
def _k9b_from_the_fixture(monkeypatch):
    monkeypatch.setattr(K, "k9b_read", _stub_k9b_read)


def make_run(models: Path, arm: str, seed: int, *, steps: Tuple[int, ...] = (50_000_081, 51_000_012),
             pin: str = PIN, last_step: int = 100_050_000, target: int = 50_000_000,
             k9b: Optional[Dict[str, Any]] = None, **over: Any) -> Path:
    """A run folder as the trainer leaves it: ``metadata.json`` and ``checkpoints/checkpoint_<N>_steps.zip`` + sidecar.
    ``k9b``: what the stubbed K9(b) reader reports for this run (default: PASS)."""
    kit = K.KIT_ARM[arm]
    run = models / A.run_name(kit, seed)
    (run / "checkpoints").mkdir(parents=True)
    history = [{"git_hash": pin, "pin_source": "pin_commit", "first_step": 0, "last_step": last_step, "torch": "2.8"}]
    md: Dict[str, Any] = {
        "cli_args": {"seed": seed, "n_epochs": 10, "ent_coef": 0.0125, **LEVERS[arm]},
        "original_command": " ".join(["/x/src/main/launcher/__main__.py", *A.build_argv(kit, seed, pin, target)]),
        "ent_coef": 0.0125, "pin_history": history, "lineage": {"role": "fresh", "fork_parent": None},
        "torch_version": "2.8.0+cu126", "init_num_threads": 8}
    side_over = over.pop("side", {})
    cli_over = over.pop("cli", {})
    md.update(over)
    md["cli_args"].update(cli_over)
    (run / "metadata.json").write_text(json.dumps(md))
    if k9b is not None:
        (run / K9B_STUB).write_text(json.dumps(k9b))
    for n in steps:
        z = run / "checkpoints" / f"checkpoint_{n}_steps.zip"
        z.write_bytes(f"{run.name}@{n}".encode())
        side = {"num_timesteps": n, "ent_coef": 0.0125,
                "pin_history": [{**history[0], "last_step": n}], **side_over}
        z.with_suffix(".json").write_text(json.dumps(side))
    return run


def make_all(models: Path, skip: Tuple[int, ...] = (), **kw: Any) -> None:
    for arm in X.ARMS:
        for seed in A.SEEDS[K.KIT_ARM[arm]]:
            if seed not in skip:
                make_run(models, arm, seed, **kw)


def plan_of(models: Path, look: str = "50M", **kw: Any) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    seeds = K.design_seeds(kw.get("replace"), K.look_arms(look, kw.get("survivor")))
    kw.setdefault("levers", {(arm, s): LEVERS[arm] for arm, ss in seeds.items() for s in ss})
    return K.plan(models, look, PIN, **kw)


# ------------------------------------------------------------------------------------------- names and steps
def test_one_family_a_panel_and_one_request_a_look_and_panel():
    fams = {K.family(p) for p in X.PANELS}
    reqs = {K.request(look, p, eq_step=61_000_000 if look == "EQ" else None) for look in X.LOOKS for p in X.PANELS}
    assert fams == {"b1_scratch_h2h_home72", "b1_scratch_h2h_heldout72_hpfix", "b1_scratch_h2h_forum72",
                    "b1_scratch_h2h_p72"}
    assert len(reqs) == len(X.LOOKS) * len(X.PANELS) and K.request("50M", "forum72") == "b1_50m_forum72"
    for bad in (lambda: K.family("u72"), lambda: K.request("60M", "home72"), lambda: K.request("50M", "u72")):
        with pytest.raises(K.LookError):
            bad()


def test_the_eq_request_carries_its_step_and_a_survivor_request_its_factor():
    # a plan at another EQ step, or after a stop, can never resume into (or be read from) another look's games
    assert K.request("EQ", "p72", eq_step=61_000_000) == "b1_eq61m_p72"
    assert K.request("EQ", "p72", eq_step=52_000_000) == "b1_eq52m_p72"
    assert K.request("100M", "forum72", survivor="breadth") == "b1_100m_breadthonly_forum72"
    assert K.request("EQ", "home72", eq_step=61_000_000, survivor="width") == "b1_eq61m_widthonly_home72"
    every = {K.request(look, "home72", eq_step=61_000_000 if look == "EQ" else None, survivor=sv)
             for look in ("75M", "100M", "EQ") for sv in (None, "width", "breadth") if (look, sv) != ("EQ", "breadth")}
    assert len(every) == 8
    for bad in (lambda: K.request("EQ", "p72"), lambda: K.request("50M", "p72", eq_step=61_000_000),
                lambda: K.request("100M", "p72", survivor="depth"),
                lambda: K.request("EQ", "p72", eq_step=61_000_000, survivor="breadth")):  # no wide arm is left
        with pytest.raises(K.LookError):
            bad()
    assert "§5" in K.family_rule("home72", PIN) and "1000 mirrored pairs" in K.family_rule("home72", PIN)
    assert "500 mirrored pairs" in K.family_rule("p72", PIN)


def test_a_matched_look_takes_every_arm_at_one_step_and_eq_takes_the_wide_arms_earlier():
    assert K.look_steps("50M") == {a: 50_000_000 for a in X.ARMS}
    assert K.look_steps("100M") == {a: 100_000_000 for a in X.ARMS}
    assert K.look_steps("EQ", 61_000_000) == {"C": 100_000_000, "Br": 100_000_000, "W": 61_000_000,
                                              "WB": 61_000_000}
    for bad in (lambda: K.look_steps("EQ"), lambda: K.look_steps("EQ", 61_500_000),
                lambda: K.look_steps("EQ", 100_000_000), lambda: K.look_steps("EQ", 0),
                lambda: K.look_steps("50M", 61_000_000), lambda: K.look_steps("60M")):
        with pytest.raises(K.LookError):
            bad()


def test_the_spare_seed_replaces_a_registered_seed_in_its_own_arm_only():
    assert K.design_seeds() == {"C": (5101, 5102, 5110), "W": (5103, 5104, 5111), "Br": (5105, 5106, 5112),
                                "WB": (5107, 5108, 5113)}
    assert K.design_seeds(5106)["Br"] == (5105, 5109, 5112) and K.design_seeds(5113)["WB"] == (5107, 5108, 5109)
    assert {a: s for a, s in K.design_seeds(5106).items() if a != "Br"} == \
        {a: s for a, s in K.design_seeds().items() if a != "Br"}
    # --replace names the REPLACED registered seed; the spare's own number is refused, and says what to give
    with pytest.raises(K.LookError, match="give the seed the spare \\(5109\\) REPLACED"):
        K.design_seeds(5109)


# ------------------------------------------------------------------------------------------ the checkpoint
def test_a_looks_checkpoint_is_the_first_one_at_or_after_its_step(tmp_path):
    run = make_run(tmp_path, "C", 5101, steps=(49_000_090, 50_000_081, 51_000_012, 100_000_044))
    (run / "final_model.zip").write_bytes(b"never read")
    z, n, why = K.find_checkpoint(run, 50_000_000)
    assert (z.name if z else None, n, why) == ("checkpoint_50000081_steps.zip", 50_000_081, None)
    assert K.find_checkpoint(run, 100_000_000)[1] == 100_000_044
    assert K.find_checkpoint(run, 50_000_081)[1] == 50_000_081
    assert K.find_checkpoint(run, 50_000_082)[1] == 51_000_012


def test_a_checkpoint_more_than_one_interval_late_or_missing_is_not_a_looks_checkpoint(tmp_path):
    run = make_run(tmp_path, "C", 5101, steps=(49_000_090, 51_000_000))
    assert K.find_checkpoint(run, 50_000_000)[2] == \
        "its first checkpoint at or after 50,000,000 is 51,000,000, more than one interval (1,000,000) late"
    assert K.find_checkpoint(run, 50_000_001)[1] == 51_000_000
    assert K.find_checkpoint(run, 75_000_000) == (None, None, "no checkpoint at or after 75,000,000")
    assert K.find_checkpoint(tmp_path / "absent", 1)[2] == "no checkpoint at or after 1"


# ---------------------------------------------------------------------------------------------- the guards
def test_twelve_clean_runs_plan_the_fifty_four_cells_the_later_arm_playing(tmp_path):
    make_all(tmp_path)
    doc, probs = plan_of(tmp_path)
    assert probs == [] and doc is not None
    assert len(doc["cells"]) == 54 and len({tuple(c) for c in doc["cells"]}) == 54
    arm_of = {f["checkpoint"]: a for a in X.ARMS for f in doc["runs"][a]}
    for player, opponent in doc["cells"]:
        assert X.ARMS.index(arm_of[player]) > X.ARMS.index(arm_of[opponent])
        assert player.endswith("checkpoint_50000081_steps.zip") and opponent.endswith("checkpoint_50000081_steps.zip")
    assert doc["names"]["home72"] == {"family": "b1_scratch_h2h_home72", "request": "b1_50m_home72", "pairs": 1000,
                                      "rule": K.family_rule("home72", PIN)}
    assert doc["survivor"] is None
    assert doc["names"]["forum72"]["pairs"] == 500
    shas = [f["sha256"] for a in X.ARMS for f in doc["runs"][a]]
    assert len(set(shas)) == 12 and all(len(s) == 64 for s in shas)


def _one_bad(tmp_path: Path, seed: int = 5105, **kw: Any) -> str:
    arm = next(a for a in X.ARMS if seed in A.SEEDS[K.KIT_ARM[a]])
    make_all(tmp_path, skip=(seed,))
    make_run(tmp_path, arm, seed, **kw)
    doc, probs = plan_of(tmp_path)
    assert doc is None and len(probs) >= 1
    assert all(p.startswith(A.run_name(K.KIT_ARM[arm], seed) + ":") for p in probs)
    return " | ".join(probs)


def test_a_missing_run_folder_refuses_the_plan(tmp_path):
    make_all(tmp_path, skip=(5108,))
    doc, probs = plan_of(tmp_path)
    assert doc is None and probs == [f"rb_b1wb_s5108: no run folder under {tmp_path}"]


def test_a_run_that_changed_pin_or_did_not_start_at_step_zero_is_refused(tmp_path):
    assert "is not one entry at aaaaaaaa" in _one_bad(tmp_path / "a", pin_history=[
        {"git_hash": PIN, "first_step": 0, "last_step": 30_000_000},
        {"git_hash": OTHER_PIN, "first_step": 30_000_000, "last_step": 60_000_000}])
    assert "is not one entry at aaaaaaaa" in _one_bad(tmp_path / "b", pin_history=[
        {"git_hash": OTHER_PIN, "first_step": 0, "last_step": 60_000_000}])
    # a run that STARTED elsewhere and ends at the pin is still two pins: the last entry alone proves nothing
    assert "is not one entry at aaaaaaaa" in _one_bad(tmp_path / "b2", pin_history=[
        {"git_hash": OTHER_PIN, "first_step": 0, "last_step": 30_000_000},
        {"git_hash": PIN, "first_step": 30_000_000, "last_step": 60_000_000}])
    assert "starts at 5, not step 0" in _one_bad(tmp_path / "c", pin_history=[
        {"git_hash": PIN, "first_step": 5, "last_step": 60_000_000}])
    assert "before the checkpoint's 50,000,081" in _one_bad(tmp_path / "d", pin_history=[
        {"git_hash": PIN, "first_step": 0, "last_step": 50_000_080}])
    assert "checkpoint_50000081_steps.zip: pin_history" in _one_bad(tmp_path / "e", side={"pin_history": [
        {"git_hash": OTHER_PIN, "first_step": 0, "last_step": 50_000_081}]})


def test_a_fork_is_not_a_fresh_run(tmp_path):
    assert "not a FRESH run (role 'fork')" in _one_bad(tmp_path / "a", lineage={"role": "fork", "fork_parent": {}})
    assert "not a FRESH run" in _one_bad(tmp_path / "b", lineage={"role": "fresh", "fork_parent": {"run_name": "x"}})


def test_a_run_not_launched_by_its_arms_registered_argv_is_refused(tmp_path):
    def cmd(arm: str, seed: int, extra: str = "", steps: int = 50_000_000) -> str:
        return " ".join(["/x/__main__.py", *A.build_argv(arm, seed, PIN, steps)]) + extra
    for i, bad in enumerate((cmd("c", 5101), cmd("wb", 5107), cmd("br", 5106), cmd("br", 5105, " --ent-coef 0.05"),
                             cmd("br", 5105).replace("@ALL:0.5", "@ALL:0.8"), "", None)):
        assert "original_command is not the registered argv of arm Br seed 5105" in \
            _one_bad(tmp_path / str(i), original_command=bad)


def test_the_phased_and_the_fallback_launch_are_both_the_registered_argv(tmp_path):
    make_all(tmp_path, skip=(5101, 5103))
    make_run(tmp_path, "C", 5101, target=100_000_000)
    make_run(tmp_path, "W", 5103, target=50_000_000)
    assert plan_of(tmp_path)[1] == []


def test_a_resume_that_lost_the_lever_is_refused_by_the_last_processs_resolved_keys(tmp_path):
    assert "cli_args.team_mix None is not the arm's 'n1_proc_v1@ALL:0.5'" in \
        _one_bad(tmp_path / "a", cli={"team_mix": None})
    assert "cli_args.team_mix_sides" in _one_bad(tmp_path / "b", cli={"team_mix_sides": "p1"})
    assert "cli_args.d_model 128 is not the arm's 256" in _one_bad(tmp_path / "c", seed=5107, cli={"d_model": 128})
    assert "cli_args.trunk_ffn_dim" in _one_bad(tmp_path / "d", seed=5104, cli={"trunk_ffn_dim": 256})
    assert "cli_args.d_model 256 is not the arm's 128" in _one_bad(tmp_path / "e", seed=5102, cli={"d_model": 256})


def test_another_entropy_coefficient_or_epoch_count_is_refused(tmp_path):
    """The registered base is ``ent_coef`` 0.0125 (PREREG §18 Amendment 2): production's own 0.05 is refused too."""
    assert K.ENT_COEF == 0.0125 == float(A.ENT_COEF)
    assert "metadata.ent_coef 0.05 != 0.0125" in _one_bad(tmp_path / "a", ent_coef=0.05)
    assert "cli_args.ent_coef 0.05 != 0.0125" in _one_bad(tmp_path / "b", cli={"ent_coef": 0.05})
    assert "checkpoint_50000081_steps.zip ent_coef 0.01 != 0.0125" in _one_bad(tmp_path / "c", side={"ent_coef": 0.01})
    assert "n_epochs 4 != 10" in _one_bad(tmp_path / "d", cli={"n_epochs": 4})
    assert "metadata.ent_coef None != 0.0125" in _one_bad(tmp_path / "e", ent_coef=None)


# --------------------------------------------------------------------------------- K9(b) at a look (Amendment 2)
def test_a_run_whose_k9b_read_is_not_pass_refuses_the_plan_and_a_pass_is_recorded_in_it(tmp_path):
    """PREREG §10 (Amendment 2): the runs train under ``--behaviour-check warn``, so the look is where K9(b) is
    judged; VOID and INCONCLUSIVE both refuse, and the plan carries every run's record."""
    void = {"verdict": "VOID", "fires": ["judged_max 0.002 >= 0.001 at step 31000000"]}
    assert ("K9(b) through step 50000081 is VOID, not PASS: judged_max 0.002 >= 0.001 at step 31000000"
            in _one_bad(tmp_path / "a", k9b=void))
    absent = {"verdict": "INCONCLUSIVE", "reasons": ["behaviour/max_abs_dlogp_judged: no row"]}
    assert "is INCONCLUSIVE, not PASS: behaviour/max_abs_dlogp_judged: no row" in _one_bad(tmp_path / "b", k9b=absent)
    make_all(tmp_path / "c")
    doc, probs = plan_of(tmp_path / "c")
    assert probs == [] and doc is not None
    recs = [f["k9b"] for a in X.ARMS for f in doc["runs"][a]]
    assert len(recs) == 12 and all(r["verdict"] == "PASS" and r["upto"] == 50_000_081 for r in recs)
    # the reader is asked about the look checkpoint's own step, per arm (the EQ look: the wide arms' earlier step)
    asked: List[Tuple[str, int]] = []
    make_all(tmp_path / "d", steps=(61_000_007, 100_000_033))

    def spy(run_dir: Path, upto: int) -> Dict[str, Any]:
        asked.append((run_dir.name, upto))
        return _stub_k9b_read(run_dir, upto)
    assert plan_of(tmp_path / "d", "EQ", eq_step=61_000_000, k9b=spy)[1] == []
    assert len(asked) == 12 and {u for r, u in asked if "_b1w" in r} == {61_000_007}
    assert {u for r, u in asked if "_b1w" not in r} == {100_000_033}


def _write_events(path: Path, rows) -> None:
    from torch.utils.tensorboard import SummaryWriter

    path.mkdir(parents=True, exist_ok=True)
    w = SummaryWriter(log_dir=str(path))
    for tag, step, v in rows:
        w.add_scalar(tag, v, step)
    w.close()


JUDGED, P99 = "behaviour/max_abs_dlogp_judged", "behaviour/p99_abs_dlogp_current"


def test_the_k9b_reader_judges_a_fresh_runs_own_rows_through_the_look_checkpoint(tmp_path):
    """The real reader on event files: every row in (0, upto] is judged on X35's ceilings (1e-3 / 1e-4); a row after
    the checkpoint is not this look's; no row, or no child log, is INCONCLUSIVE, never a pass; the child log's
    non-finite FATAL fires from any rotation of the log."""
    run = tmp_path / "rb_b1w_s5103"
    _write_events(run / "tb", [(JUDGED, 200_000, 4e-4), (P99, 200_000, 3e-5), (JUDGED, 900_000, 5e-5),
                               (P99, 900_000, 2e-5), (JUDGED, 60_000_000, 2e-3), (P99, 60_000_000, 2e-5)])
    no_log = REAL_K9B_READ(run, 1_000_040)
    assert no_log["verdict"] == "INCONCLUSIVE" and "no launcher_child.full.log" in no_log["reasons"][0]
    assert K.k9b_problem(no_log).startswith("K9(b) through step 1000040 is INCONCLUSIVE, not PASS: no launcher_")
    (run / "launcher_child.full.log").write_text("update 1\nupdate 2\n")
    ok = REAL_K9B_READ(run, 1_000_040)
    assert ok["verdict"] == "PASS" and ok["rows"] == {"judged_max": 2, "p99": 2} and K.k9b_problem(ok) is None
    assert ok["schema"] == K.K9B_SCHEMA and ok["upto"] == 1_000_040 and ok["required_family"] == "behaviour"
    assert ok["max"]["judged_max"] == pytest.approx(4e-4) and ok["ceilings"] == {"judged_max": 1e-3, "p99": 1e-4}
    assert REAL_K9B_READ(run, 59_999_999)["verdict"] == "PASS", "the 60M row is not the 50M look's"
    late = REAL_K9B_READ(run, 60_000_000)
    assert late["verdict"] == "VOID" and "judged_max" in late["fires"][0] and "at step 60000000" in late["fires"][0]
    assert REAL_K9B_READ(run, 100_000)["verdict"] == "INCONCLUSIVE", "no row yet is not a pass"
    assert ok["last_row_step"] == 900_000 and late["last_row_step"] == 60_000_000
    # the ARM-END read (upto None) judges every row the run has written, and says how far that is
    end = REAL_K9B_READ(run, None)
    assert end["verdict"] == "VOID" and end["upto"] is None and end["last_row_step"] == 60_000_000
    assert end["rows"] == {"judged_max": 3, "p99": 3}
    assert K.k9b_problem(end).startswith("K9(b) through the run's last row (step 60000000) is VOID, not PASS: ")
    # a second process's file (a restart) is read too: a p99 at the ceiling's wrong side fires
    _write_events(run / "tb" / "restart", [(P99, 950_000, 1.5e-4)])
    for f in (run / "tb" / "restart").iterdir():
        f.rename(run / "tb" / (f.name + ".restart"))
    (run / "tb" / "restart").rmdir()
    assert "p99" in REAL_K9B_READ(run, 1_000_040)["fires"][0]
    # the non-finite FATAL in a ROTATED generation of the child log fires, at any look
    clean = tmp_path / "rb_b1c_s5101"
    _write_events(clean / "tb", [(JUDGED, 200_000, 4e-4), (P99, 200_000, 3e-5)])
    (clean / "launcher_child.full.log").write_text("fine\n")
    assert REAL_K9B_READ(clean, 1_000_000)["verdict"] == "PASS"
    (clean / "launcher_child.full.log.1").write_text("x\nFATAL: a NON-FINITE |d log pi| reached the judge\n")
    fired = REAL_K9B_READ(clean, 1_000_000)
    assert fired["verdict"] == "VOID" and "NON-FINITE" in fired["fires"][0]


def test_a_run_short_of_the_look_or_with_a_bad_sidecar_is_refused(tmp_path):
    assert "no checkpoint at or after 50,000,000" in _one_bad(tmp_path / "a", steps=(49_000_090,))
    assert "sidecar says num_timesteps 7" in _one_bad(tmp_path / "b", side={"num_timesteps": 7})
    make_all(tmp_path / "c")
    (tmp_path / "c" / "rb_b1c_s5101" / "checkpoints" / "checkpoint_50000081_steps.json").unlink()
    assert plan_of(tmp_path / "c")[1] == ["rb_b1c_s5101: checkpoint_50000081_steps.zip's sidecar cannot be read "
                                          "(FileNotFoundError)"]
    make_all(tmp_path / "d")
    (tmp_path / "d" / "rb_b1c_s5101" / "metadata.json").write_text("{")
    assert plan_of(tmp_path / "d")[1] == ["rb_b1c_s5101: metadata.json cannot be read (JSONDecodeError)"]


def test_runs_on_different_torch_or_thread_counts_refuse_the_plan(tmp_path):
    make_all(tmp_path / "a", skip=(5104,))
    make_run(tmp_path / "a", "W", 5104, torch_version="2.9.0")
    assert plan_of(tmp_path / "a")[1] == ["torch_version differs across the runs: ['2.8.0+cu126', '2.9.0']"]
    make_all(tmp_path / "b", skip=(5104,))
    make_run(tmp_path / "b", "W", 5104, init_num_threads=4)
    assert plan_of(tmp_path / "b")[1] == ["init_num_threads differs across the runs: ['4', '8']"]


def test_any_knob_that_moved_in_one_run_and_not_in_the_control_refuses_the_plan(tmp_path):
    """The last process's whole public namespace is compared with the first control run's: a recipe flag typed on
    ONE run's resume (here a learning rate and a batch size no lever owns) is named, whatever the knob."""
    make_all(tmp_path / "a", skip=(5106,))
    make_run(tmp_path / "a", "Br", 5106, cli={"lr": 1e-4, "vf_coef": 0.25})
    assert plan_of(tmp_path / "a")[1] == ["rb_b1br_s5106: its last process differs from the control's in "
                                          "['lr', 'vf_coef'] (beyond its own keys and its arm's lever)"]
    # the same knob on the FIRST control run names every other run: the control is the reference
    make_all(tmp_path / "b", skip=(5101,))
    make_run(tmp_path / "b", "C", 5101, cli={"gamma": 0.99})
    probs = plan_of(tmp_path / "b")[1]
    assert len(probs) == 11 and all("differs from the control's in ['gamma']" in x for x in probs)
    assert not any(x.startswith("rb_b1c_s5101") for x in probs)


def test_a_runs_own_keys_and_its_arms_lever_keys_are_not_drift(tmp_path):
    for i, (arm, seed) in enumerate((a, s) for a in X.ARMS for s in A.SEEDS[K.KIT_ARM[a]]):
        own = {"model": f"models/x{i}/checkpoints/c.zip", "run_dir": f"/m/x{i}", "run_name": f"x{i}",
               "steps": 50_000_000 + i, "lr": 2.5e-4, "_explicit_flags": [f"--seed{i}"]}
        if arm in ("W", "WB"):
            own["allow_nonproduction_arch"] = True
        make_run(tmp_path, arm, seed, cli=own)
    assert plan_of(tmp_path)[1] == []
    assert K.RUN_OWN_KEYS == ("model", "run_dir", "run_name", "seed", "steps")
    assert K.recipe_drift("W", {"d_model": 256, "team_mix": "x", "lr": 1}, {"d_model": 128, "lr": 1}) == ["team_mix"]
    assert K.recipe_drift("Br", {"d_model": 256, "team_mix": "x"}, {"d_model": 128}) == ["d_model"]
    assert K.recipe_drift("C", {"lr": 1}, {"lr": 1, "seed": 3, "vf_coef": 0.5}) == ["vf_coef"]


#: What the kit's stop-and-resume smoke RECORDED (2026-11-09, arm WB, ``cli_args`` of the stop against the phased
#: run's): the keys a typed ``--model`` resume rewrote, as (fresh process, resume) — batch_size and n_steps at
#: production's values in place of the smoke's typed shape.
FRESH = {"allow_nonproduction_arch": False, "arch": "production", "batch_size": 2048, "n_steps": 384,
         "eval_mirrored_pairs_source": "default", "eval_sentinel_greedy_source": "default",
         "promote_threshold_source": "default", "promotion_sprt_source": "default", "unified_damage": "both",
         "unified_moves": "both", "model": None}
RESUMED = {"allow_nonproduction_arch": False, "arch": None, "batch_size": 4096, "n_steps": 2048,
           "eval_mirrored_pairs_source": "inherited", "eval_sentinel_greedy_source": "inherited",
           "promote_threshold_source": "inherited", "promotion_sprt_source": "inherited", "unified_damage": "off",
           "unified_moves": "off", "model": "models/x/final_model.zip", "run_name": None}
#: the launcher's own six-hour restart child: the fresh argv plus --model, so --run-name and the batch size stay
RESTARTED = {**RESUMED, "batch_size": 2048, "run_name": "x"}
TRUE = {"batch_size": 2048, "n_steps": 384}


def _kinds(models: Path, kind_of: Dict[int, Dict[str, Any]], default: Dict[str, Any], **over: Any) -> List[str]:
    for arm in X.ARMS:
        for seed in A.SEEDS[K.KIT_ARM[arm]]:
            make_run(models, arm, seed, cli=dict(kind_of.get(seed, default)), **{**TRUE, **over.get(str(seed), {})})
    return plan_of(models)[1]


def test_a_resume_rewrites_some_recorded_keys_and_runs_of_different_kinds_are_not_drift(tmp_path):
    """The recorded differences between a fresh process, a restart child and a typed resume are the KIND's, not the
    run's: a replacement run still on its restart children beside seven phase-2 resumes plans, and so does a run
    that never restarted."""
    assert [K.process_kind(c) for c in (FRESH, RESTARTED, RESUMED)] == ["fresh", "restart", "resume"]
    assert set(k for k in FRESH if FRESH[k] != RESUMED.get(k)) - {"model"} <= set(K.RESUME_REWRITES)
    assert _kinds(tmp_path / "a", {5106: RESTARTED, 5103: FRESH}, RESUMED) == []
    assert _kinds(tmp_path / "b", {5101: FRESH}, RESUMED) == [], "the control is the odd one out"


def test_a_rewritten_key_that_differs_between_two_processes_of_one_kind_is_still_drift(tmp_path):
    probs = _kinds(tmp_path, {5106: {**RESUMED, "unified_moves": "both"}}, RESUMED)
    assert probs == ["rb_b1br_s5106: its last process differs from the control's in ['unified_moves'] "
                     "(beyond its own keys and its arm's lever)"]


def test_the_true_rollout_shape_is_compared_at_the_sidecars_top_level_whatever_the_kind(tmp_path):
    """``cli_args.batch_size`` is inert after a resume; the run's own value is metadata.json's top-level field."""
    probs = _kinds(tmp_path, {5106: RESTARTED}, RESUMED, **{"5106": {"batch_size": 4096}})
    assert probs == ["rb_b1br_s5106: metadata.json's ['batch_size'] differ from the control's "
                     "({'batch_size': 4096} against {'batch_size': 2048})"]
    assert K.TRUE_KEYS == ("batch_size", "n_steps")


def test_two_runs_with_the_same_checkpoint_bytes_refuse_the_plan(tmp_path):
    make_all(tmp_path)
    src = tmp_path / "rb_b1c_s5101" / "checkpoints" / "checkpoint_50000081_steps.zip"
    (tmp_path / "rb_b1w_s5103" / "checkpoints" / "checkpoint_50000081_steps.zip").write_bytes(src.read_bytes())
    assert plan_of(tmp_path) == (None, ["two runs' checkpoints have the same sha256"])


def test_the_replacement_plan_reads_the_spare_seeds_run_in_the_replaced_runs_arm(tmp_path):
    make_all(tmp_path, skip=(5106,))
    make_run(tmp_path, "Br", 5109)
    assert plan_of(tmp_path)[0] is None
    doc, probs = plan_of(tmp_path, replace=5106)
    assert probs == [] and doc is not None and doc["replaced_seed"] == 5106
    assert [f["run"] for f in doc["runs"]["Br"]] == ["rb_b1br_s5105", "rb_b1br_s5109", "rb_b1br_s5112"]


def test_the_equal_compute_look_takes_the_wide_arms_at_the_earlier_step(tmp_path):
    make_all(tmp_path, steps=(61_000_033, 100_000_044))
    doc, probs = plan_of(tmp_path, "EQ", eq_step=61_000_000)
    assert probs == [] and doc is not None
    got = {a: {f["steps"] for f in doc["runs"][a]} for a in X.ARMS}
    assert got == {"C": {100_000_044}, "Br": {100_000_044}, "W": {61_000_033}, "WB": {61_000_033}}
    assert doc["names"]["p72"]["request"] == "b1_eq61m_p72"


# ------------------------------------------------------------------------------------- after a STOP-HARM
def test_a_survivor_plan_is_the_surviving_arm_against_the_control_and_needs_only_those_runs(tmp_path):
    for arm, seeds in (("C", A.SEEDS["c"]), ("Br", A.SEEDS["br"])):  # the wide arms were stopped: no folder at 100M
        for seed in seeds:
            make_run(tmp_path, arm, seed, steps=(50_000_081, 100_000_044))
    assert plan_of(tmp_path, "100M")[0] is None  # the full look still needs all twelve
    doc, probs = plan_of(tmp_path, "100M", survivor="breadth")
    assert probs == [] and doc is not None and doc["survivor"] == "breadth" and set(doc["runs"]) == {"C", "Br"}
    arm_of = {f["checkpoint"]: a for a in doc["runs"] for f in doc["runs"][a]}
    assert len(doc["cells"]) == 9 and all((arm_of[p], arm_of[o]) == ("Br", "C") for p, o in doc["cells"])
    assert doc["names"]["forum72"]["request"] == "b1_100m_breadthonly_forum72"
    assert plan_of(tmp_path, "100M", survivor="width")[0] is None  # the other survivor's runs are not there
    with pytest.raises(K.LookError, match="every registered run"):
        plan_of(tmp_path, "50M", survivor="breadth")
    with pytest.raises(K.LookError, match="not a registered seed of"):
        plan_of(tmp_path, "100M", survivor="breadth", replace=5103)  # a stopped arm's seed is not replaced


def test_a_survivor_read_reports_the_simple_effect_and_classes_nothing(tmp_path):
    for arm, seeds in (("C", A.SEEDS["c"]), ("Br", A.SEEDS["br"])):
        for seed in seeds:
            make_run(tmp_path, arm, seed, steps=(50_000_081, 100_000_044))
    doc, _ = plan_of(tmp_path, "100M", survivor="breadth")
    cells = {p: _cells(doc, p, {("Br", "C"): 0.70}) for p in X.PANELS}
    v = K.read(doc, cells)
    assert v["outcome"] == X.NO_CLASS and v["to_owner"] is True and v["survivor"] == "breadth"
    assert v["views"]["forum72"]["estimate_pp"] == pytest.approx(20.0) and v["views"]["N"]["df"] == 4
    assert "factors" not in v and len(v["checkpoints"]) == 6
    with pytest.raises(K.LookError, match="full 100M look alone"):
        K.read(doc, cells, {"width": X.KAKUNA_POSITIVE, "breadth": X.KAKUNA_POSITIVE})
    # the full look's 54 cells are not a survivor look's: every extra cell is named, and nothing is read
    full = dict(cells)
    full["home72"] = cells["home72"] + [SimpleNamespace(player="f" * 64, opponent="e" * 64, score=0.5, n_pairs=1000,
                                                        verdict="OK", reasons=(), d=0, games=2000)]
    assert K.read(doc, full)["outcome"] == K.INCONCLUSIVE


# ------------------------------------------------------------------------------------------------ the read
def _cells(doc: Dict[str, Any], panel: str, score: Dict[Tuple[str, str], float], default: float = 0.5,
           n_pairs: Optional[int] = None, draws: int = 0) -> List[SimpleNamespace]:
    """One panel's ledger cells; ``score[(player arm, opponent arm)]`` in [0, 1]."""
    sha = {f["checkpoint"]: (f["sha256"], a) for a in doc["runs"] for f in doc["runs"][a]}
    n = X.PAIRS[panel] if n_pairs is None else n_pairs
    return [SimpleNamespace(player=sha[p][0], opponent=sha[o][0], score=score.get((sha[p][1], sha[o][1]), default),
                            n_pairs=n, verdict="OK", reasons=(), d=draws, games=2 * n)
            for p, o in doc["cells"]]


def _played(tmp_path: Path, look: str, by_panel: Dict[str, Dict[Tuple[str, str], float]]
            ) -> Tuple[Dict[str, Any], Dict[str, List[SimpleNamespace]]]:
    make_all(tmp_path, steps=(50_000_081, 75_000_007, 100_000_044))
    doc, _ = plan_of(tmp_path, look)
    assert doc is not None
    return doc, {p: _cells(doc, p, by_panel.get(p, {})) for p in X.PANELS}


BREADTH_WINS = {("Br", "C"): 0.58, ("WB", "C"): 0.58, ("Br", "W"): 0.58, ("WB", "W"): 0.58}
BREADTH_LOSES = {k: 0.42 for k in BREADTH_WINS}


def test_the_100m_read_classes_width_on_n_and_breadth_on_forum72_with_the_net_rule(tmp_path):
    # ("W", "Br") is stored as Br (the later arm) playing W: width winning there is Br scoring 0.42.
    wide = {("W", "C"): 0.58, ("WB", "C"): 0.58, ("Br", "W"): 0.42, ("WB", "Br"): 0.58}
    doc, cells = _played(tmp_path, "100M", {p: wide for p in X.PANELS})
    v = K.read(doc, cells)
    assert v["look"] == "100M" and v["factors"]["width"]["outcome"] == X.WIN
    assert v["factors"]["width"]["read_on"] == "N" and v["factors"]["breadth"]["read_on"] == "forum72"
    assert v["factors"]["breadth"]["outcome"] in (X.NO_USEFUL_GAIN, X.UNDECIDED)
    assert set(v["checkpoints"]) == {A.run_name(k, s) for k, seeds in A.SEEDS.items() for s in seeds}
    assert {c["steps"] for c in v["checkpoints"].values()} == {100_000_044}


def test_breadth_winning_on_forum72_is_kept_only_while_home72_pays_no_more_than_it_gains(tmp_path):
    gain = {("Br", "C"): 0.58, ("WB", "C"): 0.58, ("Br", "W"): 0.58, ("WB", "W"): 0.58}
    for i, (home, want) in enumerate(((0.44, X.WIN), (0.40, X.WIN_NET_NEGATIVE))):
        cost = {k: home for k in gain}
        doc, cells = _played(tmp_path / str(i), "100M", {"forum72": gain, "home72": cost})
        v = K.read(doc, cells)
        assert v["factors"]["breadth"]["outcome"] == want
        assert v["factors"]["breadth"]["class_before_net_rule"] == X.WIN
        assert v["factors"]["breadth"]["net"]["net_pp"] == pytest.approx(100 * (home - 0.5) + 8.0)


def test_the_confirming_meter_labels_a_100m_outcome_and_never_changes_it(tmp_path):
    doc, cells = _played(tmp_path, "100M", {"forum72": BREADTH_WINS})
    plain = K.read(doc, cells)
    v = K.read(doc, cells, {"width": X.KAKUNA_SPANS_ZERO, "breadth": X.KAKUNA_NEGATIVE})
    assert v["factors"]["breadth"]["outcome"] == plain["factors"]["breadth"]["outcome"] == X.WIN
    assert v["factors"]["breadth"]["confirming"] == {"contrast": "breadth_F", "kakuna_class": "NEGATIVE",
                                                     "label": X.IN_HOUSE_ONLY, "to_owner": True}
    assert v["factors"]["width"]["confirming"]["label"] == X.NO_CONTRADICTION
    assert "confirming" not in plain["factors"]["width"]
    with pytest.raises(K.LookError):
        K.read(doc, cells, {"width": X.KAKUNA_POSITIVE})
    with pytest.raises(X.CrossInputError):
        K.read(doc, cells, {"width": "GOOD", "breadth": X.KAKUNA_POSITIVE})


def test_the_50m_read_stops_breadth_on_the_net_view_and_takes_no_confirming_meter(tmp_path):
    doc, cells = _played(tmp_path, "50M", {"home72": {k: 0.36 for k in BREADTH_WINS}})
    v = K.read(doc, cells)
    assert v["factors"]["breadth"]["outcome"] == X.STOP_HARM and v["factors"]["breadth"]["read_on"] == "HF"
    assert v["factors"]["width"]["outcome"] == X.CONTINUE
    with pytest.raises(K.LookError):
        K.read(doc, cells, {"width": X.KAKUNA_POSITIVE, "breadth": X.KAKUNA_POSITIVE})


def test_the_75m_look_reports_and_classes_nothing(tmp_path):
    doc, cells = _played(tmp_path, "75M", {p: BREADTH_LOSES for p in X.PANELS})
    v = K.read(doc, cells)
    assert {v["factors"][f]["outcome"] for f in X.FACTORS} == {X.REPORTED}


def test_a_short_missing_or_foreign_cell_on_any_panel_makes_the_look_inconclusive(tmp_path):
    doc, cells = _played(tmp_path, "100M", {})
    short = dict(cells)
    short["p72"] = _cells(doc, "p72", {}, n_pairs=499)
    v = K.read(doc, short)
    assert v["outcome"] == K.INCONCLUSIVE and "factors" not in v
    assert len(v["reasons"]) == 54 and all(r.startswith("p72: cell ") and "499 completed pairs < the registered 500"
                                           in r for r in v["reasons"])
    over = dict(cells)  # games added to a look after it was planned are no part of the registered read
    over["p72"] = _cells(doc, "p72", {}, n_pairs=1000)
    assert all("1000 completed pairs > the registered 500" in r for r in K.read(doc, over)["reasons"])
    gone = dict(cells)
    gone["forum72"] = cells["forum72"][:-1]
    assert [r[:22] for r in K.read(doc, gone)["reasons"]] == ["forum72: missing cell "]
    assert len(K.read(doc, {"home72": cells["home72"]})["reasons"]) == 3 * 54
    foreign = dict(cells)
    foreign["home72"] = cells["home72"] + [SimpleNamespace(player="f" * 64, opponent="e" * 64, score=0.5,
                                                         n_pairs=1000, verdict="OK", reasons=(), d=0, games=2000)]
    assert "outside the look" in K.read(doc, foreign)["reasons"][0]
    aborted = dict(cells)
    aborted["home72"] = cells["home72"][:-1] + [SimpleNamespace(
        player=cells["home72"][-1].player, opponent=cells["home72"][-1].opponent, score=0.5, n_pairs=1000,
        verdict="INCONCLUSIVE", reasons=("30 % of its games aborted",), d=0, games=2000)]
    assert K.read(doc, aborted)["reasons"] == ["home72: cell rb_b1wb_s5113 vs rb_b1br_s5112: 30 % of its games aborted"]


def test_drawn_games_are_reported_a_panel_and_more_than_a_quarter_of_a_cell_is_inconclusive(tmp_path):
    doc, cells = _played(tmp_path, "100M", {})
    ok = dict(cells)
    ok["forum72"] = _cells(doc, "forum72", {}, draws=250)  # exactly 25 % of 1,000 games: read, and reported
    v = K.read(doc, ok)
    assert "factors" in v and v["draws"]["forum72"] == {"games": 54_000, "draws": 13_500, "share": 0.25,
                                                         "worst_cell_share": 0.25}
    assert v["draws"]["home72"]["draws"] == 0 and v["draws"]["home72"]["games"] == 108_000
    bad = dict(cells)
    bad["forum72"] = _cells(doc, "forum72", {}, draws=251)
    v = K.read(doc, bad)
    assert v["outcome"] == K.INCONCLUSIVE and len(v["reasons"]) == 54
    assert all("251 of 1000 games drawn (the turn limit) > 25 %" in r for r in v["reasons"])


def _rows(pin: str = PIN, team_set: str = "ts_home", device: str = "cpu", backend: str = "eager", n: int = 2
          ) -> List[Dict[str, Any]]:
    return [{"commit": pin, "regime": {"team_set": team_set}, "compute": {"device": device, "backend": backend}}
            for _ in range(n)]


TEAM_SETS = {p: "ts_" + p.split("7")[0] for p in X.PANELS}


def test_rows_at_another_commit_team_set_or_a_second_device_are_named():
    good = {p: _rows(team_set=TEAM_SETS[p]) for p in X.PANELS}
    assert K.row_problems(good, PIN, TEAM_SETS) == []
    assert K.row_problems({}, PIN, TEAM_SETS) == []  # no row is a missing cell's business, not this check's
    at_head = {**good, "p72": good["p72"] + _rows(pin=OTHER_PIN, team_set=TEAM_SETS["p72"], n=1)}
    assert K.row_problems(at_head, PIN, TEAM_SETS) == ["p72: rows played at ['bbbbbbbbbbbb'], not the pin aaaaaaaaaaaa"]
    wrong_teams = {**good, "forum72": _rows(team_set=TEAM_SETS["home72"])}
    assert K.row_problems(wrong_teams, PIN, TEAM_SETS)[0].startswith("forum72: rows on team set(s) ['ts_home']")
    mixed = {**good, "home72": _rows(team_set=TEAM_SETS["home72"], device="cuda", backend="graph")}
    assert K.row_problems(mixed, PIN, TEAM_SETS)[0].startswith("the look was played on more than one device/backend")
    eager_and_graph = {**good, "home72": _rows(team_set=TEAM_SETS["home72"], backend="graph")}
    assert len(K.row_problems(eager_and_graph, PIN, TEAM_SETS)) == 1
    all_cuda = {p: _rows(team_set=TEAM_SETS[p], device="cuda", backend="graph") for p in X.PANELS}
    assert K.row_problems(all_cuda, PIN, TEAM_SETS) == []


def test_a_row_problem_makes_the_look_inconclusive_whatever_the_cells_say(tmp_path):
    doc, cells = _played(tmp_path, "100M", {"forum72": BREADTH_WINS})
    assert "factors" in K.read(doc, cells)
    v = K.read(doc, cells, None, ["p72: rows played at ['bbbbbbbbbbbb'], not the pin aaaaaaaaaaaa"])
    assert v["outcome"] == K.INCONCLUSIVE and "factors" not in v and v["reasons"][0].startswith("p72: rows played at")


def _write_look(root: Path, doc: Dict[str, Any], *, commit: str = PIN, device: str = "cpu",
                team_set_of: Any = None, skip_panel: Optional[str] = None) -> None:
    """A survivor look's rows in a REAL count ledger, as ``main.h2h`` builds them: a registered family a panel, the
    look's request, one 2-pair batch a cell, the panel's own team set in the regime."""
    import types

    from agents.training import eval_ledger as L
    from agents.training.trace_result import LOSS, WIN
    from main.h2h import panel as PN
    from main.h2h import play as PL
    from utils.rust_env import episode as EP

    teams = [f"packed team {i}" for i in range(4)]
    games = [{"game": g, "result": WIN if g % 2 == 0 else LOSS, "swapped": g % 2 == 1,
              "teams": [g // 2, 3 - g // 2] if g % 2 == 0 else [3 - g // 2, g // 2], "near_ties": 0,
              "near_ties_wide": 0} for g in range(4)]
    ref = {f["checkpoint"]: PL.PlayerRef(f["sha256"][:8], f["checkpoint"], str(root / "cfg.json"), "/r", f["run"],
                                         "explicit_zip", f["steps"], f["sha256"])
           for a in doc["runs"] for f in doc["runs"][a]}
    (root).mkdir(parents=True, exist_ok=True)
    (root / "cfg.json").write_text(json.dumps({"progress_decision_tense": False, "progress_switch_freeze": False}))
    w = L.LedgerWriter(root, producer="h2h")
    for panel in X.PANELS:
        if panel == skip_panel:
            continue
        nm = doc["names"][panel]
        w.register_family(nm["family"], decision_kind="ab_verdict", rule=nm["rule"], protocol=K.PROTOCOL)
        req = w.open_request(nm["request"], kind="adhoc", purpose="ab", family=nm["family"])
        ts = (team_set_of or PN.panel_team_set_id)(panel)
        for player, opponent in doc["cells"]:
            a, b = ref[player], ref[opponent]
            eng = types.SimpleNamespace(compute=PL.Compute(device=device), regime=PL.regime_for(EP.stall_threshold(), ts),
                                        torch_version="t", core_stamp="s", team_check={"a": "x"},
                                        historical={"a": []}, mode="off", levels=("off", "off"), oracle=None)
            key = f"h2h:{a.sha256}:{b.sha256}"
            row = PL.build_row(writer=w, run_label=nm["request"], commit=commit, a=a, b=b, eng=eng, purpose="ab",
                               key=key, sched_seed=0, batch=0, cseed=PL.cycle_seed(0, key, 0),
                               score=PL.score_games(games, teams, 2), t_start=L.utc_now(), t_end=L.utc_now(),
                               wall_s=2.0, load={"start": [0, 0, 0], "end": [0, 0, 0], "contention": 1.0},
                               executor={}, request=req)
            assert L.validate_row(row) == []
            claim = w.claim(req["request_id"], batch=0, player=a.sha256, opponent=b.sha256,
                            regime_id=row["regime"]["regime_id"], expected_wall_s=10)
            w.append_row(row, claim)
    w.close()


@pytest.fixture()
def survivor_doc(tmp_path):
    for arm, seeds in (("C", A.SEEDS["c"]), ("Br", A.SEEDS["br"])):
        for seed in seeds:
            make_run(tmp_path / "models", arm, seed, steps=(50_000_081, 100_000_044))
    doc, probs = plan_of(tmp_path / "models", "100M", survivor="breadth")
    assert probs == [] and doc is not None
    return doc


@pytest.mark.integration
def test_the_ledger_read_maps_real_rows_to_the_looks_cells_and_passes_rows_played_as_registered(tmp_path, survivor_doc):
    """The whole read path on a real ledger: ``main.h2h``'s row builder, the family read, the request's cells, the
    row checks. Reverting any field name the checks read (``commit``, ``regime.team_set``, ``compute.device``,
    ``request.id``) fails here."""
    root = tmp_path / "ledger"
    _write_look(root, survivor_doc)
    cells, row_reasons = K.ledger_cells(root, survivor_doc)
    assert row_reasons == []
    shas = {f["sha256"] for a in survivor_doc["runs"] for f in survivor_doc["runs"][a]}
    for panel in X.PANELS:
        assert len(cells[panel]) == 9 and {c.player for c in cells[panel]} | {c.opponent for c in cells[panel]} == shas
        assert all(c.n_pairs == 2 and c.games == 4 and c.d == 0 and c.score == 0.5 for c in cells[panel])
    v = K.read(survivor_doc, cells, None, row_reasons)  # 2 pairs a cell is not the registered count: said, not read
    assert v["outcome"] == K.INCONCLUSIVE and all("< the registered" in r for r in v["reasons"])
    assert v["draws"]["p72"] == {"games": 36, "draws": 0, "share": 0.0, "worst_cell_share": 0.0}


@pytest.mark.integration
@pytest.mark.parametrize("kw,want", [
    ({"commit": OTHER_PIN}, "rows played at ['bbbbbbbbbbbb'], not the pin aaaaaaaaaaaa"),
    # forum72's rows played on p72's teams and p72's on forum72's (one team set twice is the ledger's own refusal)
    ({"team_set_of": lambda panel: __import__("main.h2h.panel", fromlist=["x"]).panel_team_set_id(
        {"forum72": "p72", "p72": "forum72"}.get(panel, panel))}, "rows on team set(s)"),
])
def test_real_rows_at_another_commit_or_on_another_panels_teams_are_named(tmp_path, survivor_doc, kw, want):
    root = tmp_path / "ledger"
    _write_look(root, survivor_doc, **kw)
    _cells_read, row_reasons = K.ledger_cells(root, survivor_doc)
    assert row_reasons and all(want in r for r in row_reasons)
    assert len(row_reasons) == (4 if "commit" in kw else 2)


@pytest.mark.integration
def test_a_look_with_an_unplayed_family_reads_no_cells_there_and_is_inconclusive(tmp_path, survivor_doc):
    root = tmp_path / "ledger"
    _write_look(root, survivor_doc, skip_panel="forum72")
    cells, row_reasons = K.ledger_cells(root, survivor_doc)
    assert cells["forum72"] == [] and row_reasons == []
    assert any(r.startswith("forum72: missing cell") for r in K.read(survivor_doc, cells, None, row_reasons)["reasons"])


# ------------------------------------------------------------------------------------------------- the CLI
def test_plan_writes_the_cells_file_play_many_takes_and_exits_three_when_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(K, "expected_levers", lambda arm, seed, pin: LEVERS[arm])
    make_all(tmp_path / "models", skip=(5108,))
    assert K.main(["plan", "--look", "50M", "--pin", PIN, "--out", str(tmp_path / "w")]) == 3
    assert "REFUSED: rb_b1wb_s5108: no run folder" in capsys.readouterr().err
    assert not (tmp_path / "w").exists()
    make_run(tmp_path / "models", "WB", 5108)
    assert K.main(["plan", "--look", "50M", "--pin", PIN, "--out", str(tmp_path / "w")]) == 0
    assert capsys.readouterr().out.strip() == "54"
    cells = json.loads((tmp_path / "w" / "cells_plan.json").read_text())
    assert len(cells) == 54 and all(len(c) == 2 and Path(c[0]).is_file() and Path(c[1]).is_file() for c in cells)
    assert json.loads((tmp_path / "w" / "look.json").read_text())["pin"] == PIN


def test_the_early_check_reads_one_run_at_its_first_checkpoint_and_exits_three_on_a_problem(tmp_path, monkeypatch,
                                                                                            capsys):
    """PREREG §10 (Amendment 1): a run is checked hours in, alone, with the plan's own per-run guard."""
    models = tmp_path / "models"
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(models))
    monkeypatch.setattr(K, "expected_levers", lambda arm, seed, pin: LEVERS[arm])
    first = dict(steps=(1_000_040,), last_step=1_000_040)
    make_run(models, "W", 5103, **first)
    assert K.main(["check-run", "--arm", "W", "--seed", "5103", "--pin", PIN]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["ok"] is True and got["problems"] == [] and got["run"] == "rb_b1w_s5103" and got["steps"] == 1_000_040
    assert got["checkpoint"].endswith("checkpoint_1000040_steps.zip") and got["init_num_threads"] == 8
    assert got["k9b"]["verdict"] == "PASS" and got["k9b"]["upto"] == 1_000_040, "the record carries the K9(b) read"
    assert got["arm_end"] is False
    # --arm-end: the same guards, K9(b) on every row the run has written (no step bound; PREREG §10).
    assert K.main(["check-run", "--arm", "W", "--seed", "5103", "--pin", PIN, "--arm-end"]) == 0
    end = json.loads(capsys.readouterr().out)
    assert end["arm_end"] is True and end["k9b"]["upto"] is None and end["steps"] == 1_000_040
    # A run whose K9(b) read is not PASS at its first checkpoint is a problem (PREREG §10, Amendment 2).
    make_run(models, "W", 5111, **first, k9b={"verdict": "VOID", "fires": ["p99 0.0002 >= 0.0001 at step 700000"]})
    assert K.main(["check-run", "--arm", "W", "--seed", "5111", "--pin", PIN]) == 3
    void = json.loads(capsys.readouterr().out)
    assert void["k9b"]["verdict"] == "VOID" and void["problems"] == [
        "K9(b) through step 1000040 is VOID, not PASS: p99 0.0002 >= 0.0001 at step 700000"]
    # The wide run typed WITHOUT its lever: the launch argv and the resolved width both name it.
    make_run(models, "W", 5104, **first, cli={"d_model": 128, "trunk_ffn_dim": 256},
             original_command=" ".join(["/x/launcher", *A.build_argv("c", 5101, PIN)]))
    assert K.main(["check-run", "--arm", "W", "--seed", "5104", "--pin", PIN]) == 3
    bad = json.loads(capsys.readouterr().out)
    assert bad["ok"] is False and any("original_command is not the registered argv" in p for p in bad["problems"])
    assert any(p.startswith("cli_args.d_model 128") for p in bad["problems"])
    # Another pin, no checkpoint yet, no folder: each is a problem, never a pass.
    assert K.main(["check-run", "--arm", "W", "--seed", "5103", "--pin", OTHER_PIN]) == 3
    assert any("is not one entry at" in p for p in json.loads(capsys.readouterr().out)["problems"])
    assert K.main(["check-run", "--arm", "W", "--seed", "5103", "--pin", PIN, "--step", "2000000"]) == 3
    assert json.loads(capsys.readouterr().out)["checkpoint"] is None
    assert K.main(["check-run", "--arm", "C", "--seed", "5101", "--pin", PIN]) == 3
    assert json.loads(capsys.readouterr().out)["problems"] == [f"no run folder under {models}"]
    # The spare is checked under the arm it replaced in; a seed of another arm is refused, not checked.
    make_run(models, "Br", A.SPARE_SEED, **first)
    assert K.main(["check-run", "--arm", "Br", "--seed", str(A.SPARE_SEED), "--pin", PIN]) == 0
    capsys.readouterr()
    assert K.main(["check-run", "--arm", "C", "--seed", "5103", "--pin", PIN]) == 2
    assert "REFUSED: seed 5103 is not arm 'c'" in capsys.readouterr().err


def test_names_prints_what_the_driver_registers_and_plays(capsys):
    assert K.main(["names", "--look", "100M", "--pin", PIN]) == 0
    got = json.loads(capsys.readouterr().out)
    assert list(got) == list(X.PANELS)
    assert got["home72"]["request"] == "b1_100m_home72" and got["home72"]["pairs"] == 1000
    assert got["forum72"]["rule"] == K.family_rule("forum72", PIN)
    assert K.main(["names", "--look", "EQ", "--pin", PIN, "--eq-step", "61000000", "--survivor", "width"]) == 0
    assert json.loads(capsys.readouterr().out)["p72"]["request"] == "b1_eq61m_widthonly_p72"
    assert K.main(["names", "--look", "EQ", "--pin", PIN]) == 2  # an EQ look without its step has no name
    assert K.main(["names", "--look", "50M", "--pin", PIN, "--survivor", "width"]) == 2


def test_the_arms_real_argvs_resolve_to_the_lever_keys_the_guard_compares():
    for arm in X.ARMS:
        seed = A.SEEDS[K.KIT_ARM[arm]][0]
        assert K.expected_levers(arm, seed, PIN) == LEVERS[arm], arm


def test_the_looks_declared_read_is_the_h2h_family_read_under_its_own_name():
    """The look reads what ``main.h2h`` wrote, on ``main.h2h.cross``'s own terms: only the reader's name differs."""
    import dataclasses

    from main.h2h import cross as CX
    assert K.LOOK_READ.name == "main.b1.look.family"
    assert dataclasses.replace(K.LOOK_READ, name=CX.FAMILY_READ_OFF.name) == CX.FAMILY_READ_OFF
