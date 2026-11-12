"""BOUNDARY-1's stop-and-resume check (``main.b1.resume_check``): the static invariants, fast; the two-segment smoke
and the optimizer comparison, ``slow``."""
import os
import re
import subprocess
import sys

import pytest

from main.b1 import arms as A
from main.b1 import resume_check as R
from utils.paths import repo_path, repo_root

PIN = "0" * 40          # any full sha: the argv and the resolved config do not depend on which
RECORD = "designs/research_state/measurements/boundary1_scratch_2026-11-09"


def _record_pin():
    """The pin the committed launch block names (the kit carries none), or skip."""
    block = repo_path(RECORD, "launch_block.txt")
    if not block.exists():
        pytest.skip("the registration record is not in this tree")
    m = re.search(r"^# pin ([0-9a-f]{40});", block.read_text(), re.M)
    assert m, "the launch block names its pin"
    if subprocess.run(["git", "-C", str(repo_root()), "cat-file", "-e", m.group(1) + "^{commit}"],
                      capture_output=True).returncode != 0:
        pytest.skip("the pin commit is not in this clone")
    return m.group(1)


def _diffs(rep):
    return "\n".join(rep.differences)


# ------------------------------------------------------------------------------------------------ the report
def test_a_report_passes_until_a_difference_is_recorded_and_then_exits_nonzero():
    rep = R.Report()
    rep.ok("a")
    rep.note("b")
    assert rep.expect(True, "c") is True
    assert rep.exit_code == 0 and rep.render().endswith("PASS: no difference")
    assert rep.expect(False, "the thing", "why") is False
    rep.diff("another")
    assert rep.exit_code == 1 and rep.differences == ["the thing: why", "another"]
    assert rep.render().splitlines()[-3:] == ["FAIL: 2 difference(s)", "  1. the thing: why", "  2. another"]


# ------------------------------------------------------------------------------------------------ the resolved config
def test_the_static_check_passes_for_all_four_arms():
    rep = R.static_report(pin=PIN)
    assert rep.differences == [], rep.render()
    text = rep.render()
    for arm in A.ARMS:
        assert f"arm {arm}: the resolved config at 50,000,000 and at 100,000,000 differs in 'steps' only" in text
        for dest, (want, _why) in R.TARGET_KNOBS.items():
            assert f"arm {arm}: {dest} = {want!r}" in text
    assert "all 12 phase-2 commands: a same-run RESTART" in text


def test_the_target_knobs_are_the_registered_recipes_values():
    """The registered recipe facts the check leans on: production's, with the base every run types (PREREG §18
    Amendment 2: ent_coef 0.0125), ten epochs, 1M checkpoints."""
    knobs = {k: v for k, (v, _why) in R.TARGET_KNOBS.items()}
    assert knobs == {"anneal_lr_start_steps": None, "adaptive_batch": "off", "clip_range": 0.15,
                     "clip_range_vf": None, "ent_coef": 0.0125, "n_epochs": 10, "overlap_collect": False,
                     "checkpoint_every_steps": 1_000_000}


def test_a_knob_that_moves_with_the_step_target_is_a_named_difference(monkeypatch):
    real = A.resolved_namespace

    def moving(argv):
        ns = real(argv)
        ns["anneal_lr_start_steps"] = ns["steps"] // 2          # a schedule keyed to the target
        return ns
    monkeypatch.setattr(A, "resolved_namespace", moving)
    rep = R.Report()
    R.check_resolved(rep, PIN)
    for arm in A.ARMS:
        assert f"arm {arm}: the resolved config at 50,000,000 and at 100,000,000 differs in 'steps' only" in _diffs(rep)
        assert f"arm {arm}: anneal_lr_start_steps = None: resolved 25000000 / 50000000" in _diffs(rep)
    assert "differs in ['anneal_lr_start_steps', 'steps']" in _diffs(rep)


@pytest.mark.parametrize("dest,value", [("anneal_lr_start_steps", 40_000_000), ("adaptive_batch", "on"),
                                        ("clip_range", 0.2), ("clip_range_vf", 0.15), ("ent_coef", 0.01),
                                        ("n_epochs", 4), ("overlap_collect", True),
                                        ("checkpoint_every_steps", 3_000_000)])
def test_each_target_knob_off_its_value_is_a_named_difference(monkeypatch, dest, value):
    real = A.resolved_namespace

    def off(argv):
        ns = real(argv)
        assert dest in ns, f"{dest} is not a key of the resolved namespace"
        ns[dest] = value
        return ns
    monkeypatch.setattr(A, "resolved_namespace", off)
    rep = R.Report()
    R.check_resolved(rep, PIN)
    assert len(rep.differences) == len(A.ARMS), rep.render()         # that knob, once an arm; nothing else
    assert all(f": {dest} = " in d and f"resolved {value!r} / {value!r}" in d for d in rep.differences)


# ------------------------------------------------------------------------------------------------ the phase-2 command
def test_the_phase_two_commands_pass_the_role_check():
    rep = R.Report()
    R.check_phase2_role(rep, PIN)
    assert rep.differences == [] and len(rep.lines) == 1


@pytest.mark.parametrize("edit,said", [
    (lambda v: v + ["--run-name", "rb_b1c_s5101"], "names a run dir (a fork)"),
    (lambda v: v + ["--arch", "production"], "carries a fresh-only flag"),
    (lambda v: [t for t in v if t not in ("--pin-commit", PIN)], "does not hold the pin"),
    (lambda v: [("1" * 40 if t == PIN else t) for t in v], "does not hold the pin"),
    (lambda v: v + ["--seed", "5101"], "types a lever or the seed"),
    (lambda v: v + ["--d-model", "256"], "types a lever or the seed"),
    (lambda v: [("models/rb_other_s1/final_model.zip" if t.endswith("final_model.zip") else t) for t in v],
     "not a same-run checkpoint"),
])
def test_a_phase_two_command_that_is_not_a_plain_same_run_resume_is_named(monkeypatch, edit, said):
    real = A.resume_argv
    monkeypatch.setattr(A, "resume_argv", lambda arm, seed, pin=None: edit(real(arm, seed, pin)))
    rep = R.Report()
    R.check_phase2_role(rep, PIN)
    assert len(rep.differences) == 12 and all(said in d for d in rep.differences), rep.render()
    assert sorted(d.split(":")[0] for d in rep.differences) == sorted(A.run_name(a, s) for a, s in A.LAUNCH_ORDER)


# ------------------------------------------------------------------------------------------------ the inventory
def test_the_readers_of_the_target_are_classified_by_name_and_tests_are_ignored():
    files = [
        ("src/main/train/new_schedule.py", "x = warmup(args.steps)\n"),
        ("src/main/train/loopish.py", "def learn(self, total_timesteps):\n    pass\n"),
        ("src/main/train/ns_reader.py", "n = getattr(ns, 'steps', 0)\n"),
        ("src/agents/training/sched.py", "p = self._current_progress_remaining\n"),
        ("src/agents/training/sched_test.py", "p = m._current_progress_remaining; args.steps\n"),
        ("src/agents/conftest.py", "args.steps\n"),
        ("src/agents/training/pool.py", "kept = set(self.steps())\nmx.steps[i]\nnum_timesteps = 3\n"),
        ("src/agents/training/notes.md", "args.steps\n"),
    ]
    assert R.classify_readers(files) == {
        "target": ["src/main/train/loopish.py", "src/main/train/new_schedule.py", "src/main/train/ns_reader.py"],
        "progress": ["src/agents/training/sched.py"]}


def test_an_unaudited_reader_and_a_vanished_audited_one_are_both_differences():
    inv = {"target": sorted(R.TARGET_READERS) + ["src/main/train/new_schedule.py"],
           "progress": sorted(set(R.PROGRESS_READERS) - {"src/agents/training/instrumented_ppo/ppo.py"})}
    rep = R.Report()
    R.check_inventory(inv, rep)
    assert len(rep.differences) == 2, rep.render()
    assert "NOT audited: src/main/train/new_schedule.py" in rep.differences[0]
    assert "no longer a reader" in rep.differences[1] and "instrumented_ppo/ppo.py" in rep.differences[1]


def test_at_the_pin_the_readers_are_exactly_the_audited_ones():
    """Read from the pin's OBJECTS, so this holds in any checkout, however far main has moved."""
    inv = R.classify_readers(R.pin_files(str(repo_root()), _record_pin()))
    assert inv == {"target": sorted(R.TARGET_READERS), "progress": sorted(R.PROGRESS_READERS)}


def test_every_pattern_match_contains_a_prefilter_needle():
    for text in ("args.steps", "a.steps", "ns.steps", "total_timesteps", "_total_timesteps",
                 "getattr(ns, 'steps', 0)", 'getattr(args, "steps")', "_current_progress_remaining"):
        assert R._TARGET_RE.search(text) or R._PROGRESS_RE.search(text), text
        assert any(n in text for n in R._NEEDLES), text


def test_pin_files_refuses_a_commit_it_cannot_read():
    with pytest.raises(RuntimeError, match="git grep"):
        R.pin_files(str(repo_root()), "f" * 40)


# ------------------------------------------------------------------------------------------------ the command line
def _cli(*argv):
    env = dict(os.environ, PYTHONPATH=str(repo_path("src")), CUDA_VISIBLE_DEVICES="")
    env.pop("GEN3AI_TEST_ALLOW_GPU", None)
    return subprocess.run([sys.executable, "-m", "main.b1.resume_check", *argv], capture_output=True, text=True,
                          env=env, cwd=str(repo_root()), timeout=600)


def test_the_static_command_fails_on_a_tree_it_cannot_read_and_says_so(tmp_path):
    """The PREREG gates the phased block on this command: a tree that is not there must never read as a pass (the
    module once had no entry point, printed nothing and exited 0)."""
    r = _cli("static", "--tree", str(tmp_path / "nonexistent"), "--pin", PIN)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "PASS" not in r.stdout and "FAIL:" in r.stdout
    assert "the source inventory could not be read, so no reader was audited" in r.stdout


def test_the_static_command_needs_a_tree_and_a_pin():
    for argv in (["static"], ["static", "--tree", str(repo_root())], ["static", "--pin", PIN], []):
        r = _cli(*argv)
        assert r.returncode == 2 and "PASS" not in r.stdout, argv
    r = _cli("static", "--tree", str(repo_root()), "--pin", "0220f840")
    assert r.returncode == 2 and "REFUSED" in r.stderr and "PASS" not in r.stdout


def test_the_static_command_prints_its_evidence_then_pass_at_the_registered_pin():
    """PASS needs the importing checkout's training code to BE the pin's (true of a worktree at the pin, and of
    main until a training file moves); where it has moved the command must say which files, and fail."""
    pin = _record_pin()
    r = _cli("static", "--tree", str(repo_root()), "--pin", pin)
    moved = subprocess.run(["git", "-C", str(repo_root()), "diff", "--name-only", pin, "--", *R._CODE_PATHS],
                           capture_output=True, text=True, check=True).stdout.split()
    dirty = subprocess.run(["git", "-C", str(repo_root()), "status", "--porcelain", "--untracked-files=all", "--",
                            *R._CODE_PATHS], capture_output=True, text=True, check=True).stdout.strip()
    assert "arm wb: the resolved config at 50,000,000 and at 100,000,000 differs in 'steps' only" in r.stdout
    assert "every module reading the step target (args.steps) is an audited one (8 found)" in r.stdout
    if moved or dirty:
        assert r.returncode == 1 and "file(s) differ from" in r.stdout and "PASS" not in r.stdout
    else:
        assert r.returncode == 0 and r.stdout.rstrip().endswith("PASS: no difference"), r.stdout + r.stderr


def test_compare_fails_on_run_dirs_and_logs_that_are_not_there(tmp_path):
    r = _cli("compare", "--run-through", str(tmp_path / "a"), "--phased", str(tmp_path / "b"),
             "--stop", str(tmp_path / "s"), "--logs", str(tmp_path / "logs"))
    assert r.returncode == 1 and "PASS" not in r.stdout and "missing" in r.stdout


# ------------------------------------------------------------------------------------------------ the child argvs
def test_the_six_hour_restart_child_drops_arch_keeps_the_base_and_the_levers_and_names_the_checkpoint_and_the_run_dir():
    """The launcher's own restart child re-types the fresh argv's base (``--ent-coef 0.0125 --behaviour-check
    warn``, PREREG §18 Amendment 2) and levers; the typed phase-2 resume types neither and restores them."""
    fresh = A.build_argv("wb", 5107, PIN, 100_000_000)
    child = R.restart_child_argv(fresh, "/m/rb_b1wb_s5107/final_model_interrupted.zip", "/m/rb_b1wb_s5107")
    assert "--arch" not in child and "--pin-commit" not in child and "--restart-interval-hours" not in child
    assert child[:len(A.BASE) + len(A.WIDE_LEVER) + len(A.BREADTH_LEVER)] == [*A.BASE, *A.WIDE_LEVER, *A.BREADTH_LEVER]
    assert child[-4:] == ["--model", "/m/rb_b1wb_s5107/final_model_interrupted.zip", "--run-dir", "/m/rb_b1wb_s5107"]
    assert child[child.index("--steps") + 1] == "100000000" and child[child.index("--seed") + 1] == "5107"


def test_a_recorded_namespace_is_compared_whole_but_for_private_keys():
    reg = {"steps": 50, "model": None, "team_mix": ["n1_proc_v1@ALL:0.5"], "d_model": 256, "seed": 5107}
    rec = {"steps": 100, "model": "/m/x.zip", "team_mix": ["n1_proc_v1@ALL:0.5"], "d_model": 256, "seed": 5107,
           "_explicit_flags": ["--steps"]}
    assert R.recorded_vs_registered(rec, reg) == ["model", "steps"]
    assert R.recorded_vs_registered(dict(rec, d_model=128), reg) == ["d_model", "model", "steps"]
    assert R.recorded_vs_registered({k: v for k, v in rec.items() if k != "seed"}, reg) == ["model", "seed", "steps"]
    assert R.recorded_vs_registered(dict(rec, ent_coef=0.02), reg) == ["ent_coef", "model", "steps"], \
        "a key only the record has is a difference too"
    from main.train.launch_flags import INHERITED_DESTS, OWN_ROUTE, PER_LAUNCH
    from main.train.recipe_surface import INERT_ON_RESUME
    assert set(R.INERT_KEYS) == set(INERT_ON_RESUME), "the trainer's own list of flags a resume ignores has moved"
    # Every volatile key is one the TRAINER classes as not carried by a --model launch, or a provenance label.
    for key in R.CLI_ARGS_VOLATILE:
        assert key in PER_LAUNCH or key in INERT_ON_RESUME or key.endswith("_source"), key
        assert key not in INHERITED_DESTS and key not in OWN_ROUTE, key
    assert set(R.CLI_ARGS_VOLATILE) == {
        "steps", "model", "run_dir", "run_name", "arch", "unified_moves", "unified_damage",
        "allow_nonproduction_arch", "eval_mirrored_pairs_source", "eval_sentinel_greedy_source",
        "promote_threshold_source", "promotion_sprt_source", "batch_size", "n_steps"}
    # The arms' levers stay comparable after any resume, all but the wide arms' one-launch consent.
    levers = {d for dests in A.LEVER_DESTS.values() for d in dests}
    assert levers & set(R.CLI_ARGS_VOLATILE) == {"allow_nonproduction_arch"}
    assert set(R.FRESH_VOLATILE) < set(R.CLI_ARGS_VOLATILE)


def test_the_smokes_commands_are_the_kits_with_the_listed_substitutions_only():
    fresh = R.smoke_fresh_argv("wb", "rb_b1smoke_b_wb", 4500)
    kit = A.build_argv("wb", 5107, PIN)
    assert [t for t in kit if t not in fresh] == ["--pin-commit", PIN, "1000000", "50000000", "cuda", "rb_b1wb_s5107"]
    assert fresh[-8:] == ["--no-pin", "--debug", *R.SMOKE_SHAPE]
    assert [t for t in fresh[:-6] if t not in kit] == ["2000", "4500", "cpu", "rb_b1smoke_b_wb", "--no-pin", "--debug"]
    assert R.SMOKE_SHAPE == ("--rollout-target-samples", "4096", "--batch-size", "256", "--n-epochs", "1")
    assert int(R.SMOKE_SHAPE[1]) % 4096 == 0, "a resume's parse-time quantum is lcm(the parser's 4,096, 256 envs)"
    resume = R.smoke_resume_argv("wb", "/s/rb_b1smoke_b_wb", 9000)
    assert resume == ["--restart-interval-hours", "6", "--model", "/s/rb_b1smoke_b_wb/final_model.zip",
                      "--steps", "9000", "--device", "cpu", "--log-level", "periodic", "--no-pin", "--debug"]


# ------------------------------------------------------------------------------------------------ the smoke
@pytest.mark.slow
def test_the_two_segment_smoke_finds_no_difference_between_phased_and_run_through(tmp_path):
    """Three CPU ``--debug`` launches of the WB arm and one resume (minutes): run A to 8,000 steps, run B to 4,000
    then resumed to 8,000 by the kit's phase-2 command form, then every comparison ``compare`` makes. Also the
    comparison's own sensitivity: a record whose lever moved and whose lineage says fork."""
    import json
    import shutil
    paths = R.run_smoke(str(tmp_path))
    rep = R.compare_report(**paths)
    assert rep.differences == [], rep.render()
    assert "tensor-equal" in rep.render() and "the file the look reads exists before the run stops" in rep.render()

    bad = tmp_path / "tampered"
    shutil.copytree(paths["phased"], bad / "phased", symlinks=True)
    meta = json.loads((bad / "phased" / "metadata.json").read_text())
    meta["cli_args"]["team_mix_sides"] = "trainee"
    meta["lineage"]["role"] = "fork"
    (bad / "phased" / "metadata.json").write_text(json.dumps(meta))
    rep = R.compare_report(paths["run_through"], str(bad / "phased"), paths["stop"], paths["logs"])
    said = "\n".join(rep.differences)
    assert "team_mix_sides" in said and "lineage role of B" in said, rep.render()
