"""BOUNDARY-1's arms: eight fresh production runs, each one lever from the control, at one pin (``main.b1.arms``)."""
import json
import re
import shlex
import subprocess

import pytest

from main.b1 import arms as A
from utils.era import CURRENT_ERA, era_of_name, run_name_warning
from utils.paths import repo_path, repo_root

PIN = "0" * 40
E = CURRENT_ERA.code
RECORD = "designs/research_state/measurements/boundary1_scratch_2026-11-09"


def test_the_design_is_a_two_by_two_with_two_seeds_a_cell_and_one_seed_per_run():
    assert list(A.ARMS) == ["c", "w", "br", "wb"]
    assert A.ARMS["c"] == () and A.ARMS["wb"] == A.ARMS["w"] + A.ARMS["br"]
    assert A.ARMS["w"] == ("--d-model", "256", "--trunk-ffn-dim", "512", "--allow-nonproduction-arch")
    assert A.ARMS["br"] == ("--team-mix", "n1_proc_v1@ALL:0.5", "--team-mix-sides", "both")
    assert A.SEEDS == {"c": (5101, 5102), "w": (5103, 5104), "br": (5105, 5106), "wb": (5107, 5108)}
    seeds = [s for ss in A.SEEDS.values() for s in ss]
    assert len(set(seeds)) == len(seeds) == 8, "no two runs share a seed, within or across arms"
    assert A.SPARE_SEED == 5109 and A.SPARE_SEED not in seeds
    assert A.all_seeds() == tuple(range(5101, 5110))
    assert A.PHASE_STEPS == (50_000_000, 100_000_000) and A.CHECKPOINT_EVERY == 1_000_000


def test_the_launch_order_runs_one_seed_of_every_arm_before_any_arms_second():
    assert A.LAUNCH_ORDER == (("c", 5101), ("w", 5103), ("br", 5105), ("wb", 5107),
                              ("c", 5102), ("w", 5104), ("br", 5106), ("wb", 5108))
    assert sorted(A.LAUNCH_ORDER) == sorted((a, s) for a, ss in A.SEEDS.items() for s in ss)


def test_the_argv_is_the_shared_head_the_arms_lever_and_the_shared_tail():
    for arm, lever in A.ARMS.items():
        seed = A.SEEDS[arm][0]
        argv = A.build_argv(arm, seed, PIN)
        head = ["--restart-interval-hours", "6", "--pin-commit", PIN, "--arch", "production"]
        tail = ["--checkpoint-every-steps", "1000000", "--steps", "50000000", "--seed", str(seed),
                "--device", "cuda", "--log-level", "periodic", "--run-name", f"{E}_b1{arm}_s{seed}"]
        assert argv == head + list(lever) + tail
        assert "--model" not in argv, "a FRESH run"
        assert A.build_argv(arm, seed, PIN, 100_000_000) == head + list(lever) + \
            tail[:3] + ["100000000"] + tail[4:], "the fallback differs in the step target alone"


def test_each_arm_is_one_lever_from_the_control_in_the_argv_and_in_the_resolved_config():
    for arm, lever in A.ARMS.items():
        assert A.lever_diff(arm, PIN) == lever
    assert A.check_levers(PIN) == []
    base = A.resolved_namespace(A.build_argv("c", 5101, PIN))
    wb = A.resolved_namespace(A.build_argv("wb", 5107, PIN))
    assert A.namespace_diff(base, wb) == [
        "allow_nonproduction_arch", "d_model", "run_name", "seed", "team_mix", "team_mix_sides", "trunk_ffn_dim"]
    assert (base["d_model"], base["trunk_ffn_dim"], wb["d_model"], wb["trunk_ffn_dim"]) == (128, 256, 256, 512)
    assert base["team_mix"] is None and wb["team_mix"] == "n1_proc_v1@ALL:0.5" and wb["team_mix_sides"] == "both"
    assert (base["ent_coef"], base["n_epochs"], wb["ent_coef"], wb["n_epochs"]) == (0.05, 10, 0.05, 10)


def test_a_lever_that_leaks_into_another_arm_is_named(monkeypatch):
    monkeypatch.setitem(A.ARMS, "br", A.BREADTH_LEVER + ("--ent-coef", "0.02"))
    bad = A.check_levers(PIN)
    assert bad and all(b.startswith("br @") for b in bad) and any("ent_coef" in b for b in bad)


def test_run_names_carry_the_current_eras_prefix():
    names = A.all_run_names()
    assert names[:8] == tuple(f"{E}_b1{a}_s{s}" for a, s in A.LAUNCH_ORDER)
    assert names[8:] == tuple(f"{E}_b1{a}_s5109" for a in A.ARMS)
    assert len(set(names)) == 12
    for n in names:
        assert era_of_name(n) == CURRENT_ERA and run_name_warning(n) is None
        assert re.fullmatch(r"[a-z]{2}_b1(c|w|br|wb)_s51\d\d", n)


def test_an_unknown_arm_seed_or_target_is_refused():
    with pytest.raises(A.ArmError):
        A.build_argv("x", 5101, PIN)
    with pytest.raises(A.ArmError, match="seed"):
        A.build_argv("c", 5103, PIN)          # another arm's seed
    with pytest.raises(A.ArmError, match="target"):
        A.build_argv("c", 5101, PIN, 75_000_000)
    with pytest.raises(A.ArmError, match="target"):
        A.resume_argv("c", 5101, PIN, 50_000_000)
    assert A.build_argv("w", A.SPARE_SEED, PIN)[-1] == f"{E}_b1w_s5109", "the spare seed serves any arm"


def test_the_kit_carries_no_default_pin_and_every_entry_point_refuses_without_one(capsys):
    """The pin is registered in the PREREG, not in the tree it names: a constant here would be the wrong sha in the
    registration commit itself."""
    assert not hasattr(A, "PIN")
    for call in (lambda: A.build_argv("c", 5101), lambda: A.resume_argv("c", 5101), A.launch_block, A.check_levers,
                 lambda: A.build_argv("c", 5101, ""), lambda: A._check(str(repo_root()), None, params=False, roots=[])):
        with pytest.raises(A.ArmError, match="PREREG.md §3; pass --pin"):
            call()
    with pytest.raises(A.ArmError, match="full 40-hex"):
        A.build_argv("c", 5101, "0220f840")
    for argv in (["describe"], ["describe", "--block", "fallback"], ["describe", "--arm", "c", "--seed", "5101"],
                 ["check", "--tree", str(repo_root())]):
        with pytest.raises(SystemExit) as e:
            A.main(argv)
        assert e.value.code == 2 and "--pin" in capsys.readouterr().err, argv
    assert A.main(["describe", "--pin", "0220f840"]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_the_phase_two_command_is_a_same_run_resume_to_the_second_target():
    for arm, seed in A.LAUNCH_ORDER:
        argv = A.resume_argv(arm, seed, PIN)
        assert argv == ["--restart-interval-hours", "6", "--pin-commit", PIN,
                        "--model", f"models/{E}_b1{arm}_s{seed}/final_model.zip", "--steps", "100000000",
                        "--device", "cuda", "--log-level", "periodic"]
        assert "--run-name" not in argv, "a --run-name beside --model is a FORK into a new folder"
        assert "--arch" not in argv, "--arch is fresh-only: refused beside --model"
        assert not set(argv) & set(A.WIDE_LEVER[::2] + A.BREADTH_LEVER[::2] + ("--seed",)), \
            "the levers and the seed come back from the run's own record"


def test_the_phase_two_command_is_classified_a_restart_never_a_fork():
    """The launcher's own predicates on the phase-2 argv: the one fork-vs-restart predicate, and no fresh-only flag."""
    from main.train.combination_checks import fresh_only_flags
    from main.train.fork_lr import is_same_run_checkpoint

    for arm, seed in A.LAUNCH_ORDER:
        argv = A.resume_argv(arm, seed, PIN)
        assert not set(argv) & set(fresh_only_flags())
        model = argv[argv.index("--model") + 1]
        assert is_same_run_checkpoint(model, f"models/{A.run_name(arm, seed)}")
        assert not is_same_run_checkpoint(model, f"models/{A.run_name(arm, A.SPARE_SEED)}")


def test_describe_prints_the_block_in_launch_order(capsys):
    assert A.main(["describe", "--pin", PIN]) == 0
    out = capsys.readouterr().out
    assert out == A.launch_block(PIN)
    lines = out.splitlines()
    cmds = [ln for ln in lines if ln.startswith("python -m main.launcher ")]
    dry = [ln for ln in lines if ln.startswith("#     dry run: python -m main.launcher --dry-run ")]
    assert len(cmds) == len(dry) == 16
    for c, d in zip(cmds, dry):
        assert d == "#     dry run: " + c.replace("main.launcher ", "main.launcher --dry-run ", 1)
    assert [shlex.split(c)[3:] for c in cmds[:8]] == [A.build_argv(a, s, PIN) for a, s in A.LAUNCH_ORDER]
    assert [shlex.split(c)[3:] for c in cmds[8:]] == [A.resume_argv(a, s, PIN) for a, s in A.LAUNCH_ORDER]
    assert cmds[3] == (
        f"python -m main.launcher --restart-interval-hours 6 --pin-commit {PIN} --arch production --d-model 256 "
        "--trunk-ffn-dim 512 --allow-nonproduction-arch --team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both "
        f"--checkpoint-every-steps 1000000 --steps 50000000 --seed 5107 --device cuda --log-level periodic "
        f"--run-name {E}_b1wb_s5107")
    assert cmds[8] == (
        f"python -m main.launcher --restart-interval-hours 6 --pin-commit {PIN} --model "
        f"models/{E}_b1c_s5101/final_model.zip --steps 100000000 --device cuda --log-level periodic")
    look = lines.index("# THE 50,000,000 LOOK (the harm stop) is read here, before any phase-2 command.")
    assert lines.index(cmds[7]) < look < lines.index(cmds[8]), "the look sits between the phases"
    assert f"# {E}_b1c_s5101, {E}_b1c_s5102" not in out and f"# C control          {E}_b1c_s5101, {E}_b1c_s5102" in out


def test_the_fallback_block_is_eight_fresh_launches_straight_to_the_second_target(capsys):
    assert A.main(["describe", "--block", "fallback", "--pin", PIN]) == 0
    out = capsys.readouterr().out
    cmds = [ln for ln in out.splitlines() if ln.startswith("python -m main.launcher ")]
    assert [shlex.split(c)[3:] for c in cmds] == [A.build_argv(a, s, PIN, 100_000_000) for a, s in A.LAUNCH_ORDER]
    assert "--model" not in out and "PHASE" not in out


def test_describe_one_run_prints_one_argv_line(capsys):
    assert A.main(["describe", "--arm", "br", "--seed", "5106", "--pin", PIN]) == 0
    assert shlex.split(capsys.readouterr().out) == A.build_argv("br", 5106, PIN)
    assert A.main(["describe", "--arm", "br", "--seed", "5106", "--phase", "2", "--pin", PIN]) == 0
    assert shlex.split(capsys.readouterr().out) == A.resume_argv("br", 5106, PIN)
    assert A.main(["describe", "--arm", "br", "--pin", PIN]) == 2
    assert A.main(["describe", "--arm", "br", "--seed", "5101", "--pin", PIN]) == 2
    assert "REFUSED" in capsys.readouterr().err


def _record_pin(block_text: str) -> str:
    """The pin the committed launch block was printed at (its second line says it)."""
    m = re.search(r"^# pin ([0-9a-f]{40});", block_text, re.M)
    assert m, "the launch block names its pin on a '# pin <sha>;' line"
    return m.group(1)


def test_the_registered_transcripts_are_what_describe_prints_at_the_pin_they_name():
    """The committed launch block and argv files are this module's output AT THE PIN THE BLOCK NAMES: an edit to
    either is a visible diff, and the four argv files hold the block's pin, not another."""
    block = repo_path(RECORD, "launch_block.txt")
    if not block.exists():
        pytest.skip("the registration record is not in this tree")
    if E != "rb":
        pytest.skip("the record was registered in the Rustboro era")
    pin = _record_pin(block.read_text())
    assert block.read_text() == A.launch_block(pin)
    for arm in A.ARMS:
        seed = A.SEEDS[arm][0]
        line = repo_path(RECORD, f"argv_{A.run_name(arm, seed)}.txt").read_text()
        assert shlex.split(line) == A.build_argv(arm, seed, pin)


# --- the collision scan -------------------------------------------------------------------------------------------

def _run(root, name, seed=None, text=None):
    d = root / name
    d.mkdir(parents=True)
    if text is not None:
        (d / "metadata.json").write_text(text)
    elif seed is not None:
        (d / "metadata.json").write_text(json.dumps({"cli_args": {"seed": seed}}))
    return d


def test_the_scan_passes_a_tree_of_other_runs_and_counts_what_it_read(tmp_path):
    _run(tmp_path, "rb_x60t_s3507", seed=3507)
    _run(tmp_path, "rb_x1_s51010", seed=51010)             # 5101 is a PREFIX of this seed: not ours
    _run(tmp_path / "archive", "rb_toy_s4002", seed=4002)
    _run(tmp_path, "no_sidecar")
    _run(tmp_path, "odd", text=json.dumps({"cli_args": {"seed": None}}))
    _run(tmp_path, "odder", text=json.dumps(["not", "a", "dict"]))
    scan = A.scan_collisions([tmp_path])
    assert scan["hits"] == [] and scan["sidecars"] == 5 and scan["dirs"] == 7


@pytest.mark.parametrize("seed", [5101, 5108, 5109])
def test_the_scan_refuses_a_sidecar_that_recorded_one_of_our_seeds(tmp_path, seed):
    _run(tmp_path / "state" / "ckpts", "some_run", seed=seed)   # three directories down, as the archive nests them
    hits = A.scan_collisions([tmp_path])["hits"]
    assert len(hits) == 1 and hits[0].startswith(f"SEED {seed} is taken")


def test_the_scan_refuses_an_existing_run_folder_a_seed_in_a_name_and_an_unreadable_sidecar(tmp_path):
    _run(tmp_path, f"{E}_b1br_s5105")
    _run(tmp_path, "rb_other_s5103")
    _run(tmp_path, "torn", text="{not json")
    hits = A.scan_collisions([tmp_path])["hits"]
    assert sorted(h.split(":")[0].rsplit(" ", 1)[0] for h in hits) == [
        f"RUN NAME {E}_b1br_s5105", "SEED in a run", "UNREADABLE sidecar"]
    assert A.scan_collisions([tmp_path]) == A.scan_collisions([tmp_path]), "the same tree, the same answer"


def test_check_exits_nonzero_on_a_planted_collision_and_zero_on_a_clean_tree(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(A, "tree_head", lambda tree: PIN)
    clean = tmp_path / "clean"
    _run(clean, "rb_x60t_s3507", seed=3507)
    assert A._check(str(repo_root()), PIN, params=False, roots=[clean]) == 0
    assert "CHECK PASSED" in capsys.readouterr().out
    _run(clean, "planted", seed=5104)
    assert A._check(str(repo_root()), PIN, params=False, roots=[clean]) == 1
    out = capsys.readouterr().out
    assert "REFUSED: SEED 5104 is taken" in out and "CHECK FAILED" in out


def test_check_refuses_a_tree_that_is_not_at_the_pin(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(A, "tree_head", lambda tree: "f" * 40)
    assert A._check(str(repo_root()), PIN, params=False, roots=[tmp_path]) == 1
    assert "HEAD is not the pin" in capsys.readouterr().out


def _own(root, arm, seed, pin=PIN, steps=50_000_000, recorded_seed=None, extra=""):
    """A run folder as the launcher leaves it: the registered fresh argv behind the launcher's script path."""
    cmd = " ".join(["/x/src/main/launcher/__main__.py", *A.build_argv(arm, seed, pin, steps)]) + extra
    return _run(root, A.run_name(arm, seed), text=json.dumps(
        {"cli_args": {"seed": seed if recorded_seed is None else recorded_seed}, "original_command": cmd}))


def test_after_the_first_launch_the_scan_lists_the_experiments_own_runs_and_still_passes(tmp_path, capsys, monkeypatch):
    """Re-run at launch, the scan meets rb_b1c_s5101 itself: the run whose immutable original_command is the
    registered argv at the pin is OURS (listed, not a hit), at either registered target."""
    _own(tmp_path, "c", 5101)
    _own(tmp_path / "archive", "wb", 5107, steps=100_000_000)       # the fallback block's form, in the archive
    _own(tmp_path, "w", A.SPARE_SEED)                               # a replacement run
    scan = A.scan_collisions([tmp_path], PIN)
    assert scan["hits"] == [] and len(scan["own"]) == 3
    assert A.scan_collisions([tmp_path])["hits"], "with no pin nothing can be recognised as ours: every one is a hit"
    monkeypatch.setattr(A, "tree_head", lambda tree: PIN)
    assert A._check(str(repo_root()), PIN, params=False, roots=[tmp_path]) == 0
    out = capsys.readouterr().out
    assert out.count("    own: ") == 3 and "(3 of our own exist)" in out and "CHECK PASSED" in out


@pytest.mark.parametrize("plant", [
    lambda root: _own(root, "c", 5101, pin="1" * 40),                       # launched at another pin
    lambda root: _own(root, "c", 5101, extra=" --ent-coef 0.02"),           # a flag the registration does not type
    lambda root: _own(root, "c", 5101, recorded_seed=5102),                 # the record's seed is not the name's
    lambda root: _run(root, f"{E}_b1c_s5101", seed=5101),                   # no original_command at all
    lambda root: _run(root, f"{E}_b1c_s5101"),                              # an empty folder under our name
    lambda root: _run(root, "somebody_else", text=json.dumps({              # our argv under a foreign name
        "cli_args": {"seed": 5101},
        "original_command": " ".join(["/x/__main__.py", *A.build_argv("c", 5101, PIN)])})),
])
def test_a_foreign_run_under_one_of_our_names_or_seeds_is_still_refused(tmp_path, plant):
    plant(tmp_path)
    scan = A.scan_collisions([tmp_path], PIN)
    assert scan["hits"] and scan["own"] == []


def test_the_default_scan_roots_are_the_main_models_dir_and_the_archive(tmp_path, monkeypatch):
    import utils.paths as P

    models, archive = tmp_path / "models", tmp_path / "archive"
    models.mkdir()
    monkeypatch.setattr(P, "main_models_dir", lambda: models)
    monkeypatch.setattr(P, "gen3ai_archive_dir", lambda: archive)
    assert A.scan_roots() == [models], "a root that does not exist is not scanned (and is not listed as scanned)"
    archive.mkdir()
    assert A.scan_roots() == [models, archive]


# --- the corpus and the pin ---------------------------------------------------------------------------------------

def test_the_breadth_lever_names_the_registered_corpus_at_share_one_half():
    from agents.training.matchup_spec import parse_team_mix
    from utils.team_corpora import CORPORA

    assert A.CORPUS in CORPORA
    corpus, tier, share = parse_team_mix(A.BREADTH_LEVER[1])
    assert (corpus, tier, share) == (A.CORPUS, A.CORPUS_TIER, 0.5)


def test_the_corpus_check_loads_this_trees_corpus_and_refuses_another_identity(monkeypatch):
    ident = A.check_corpus_at(str(repo_root()))
    assert {k: ident[k] for k in A.CORPUS_IDENT} == A.CORPUS_IDENT and ident["n"] == 12000
    monkeypatch.setitem(A.CORPUS_IDENT, "data_sha256", "0" * 64)
    with pytest.raises(A.ArmError, match="not the registered corpus"):
        A.check_corpus_at(str(repo_root()))


def test_the_pin_the_record_names_carries_the_width_flag_and_the_corpus():
    block = repo_path(RECORD, "launch_block.txt")
    if not block.exists():
        pytest.skip("the registration record is not in this tree")
    pin = _record_pin(block.read_text())
    git = ["git", "-C", str(repo_root())]
    if subprocess.run(git + ["cat-file", "-e", pin + "^{commit}"], capture_output=True).returncode != 0:
        pytest.skip("the pin commit is not in this clone")
    tree = subprocess.run(git + ["ls-tree", "--name-only", pin, "src/utils/team_corpora/n1_proc_v1/"],
                          capture_output=True, text=True, check=True).stdout.split()
    assert {"manifest.json", "teams.jsonl.gz"} <= {t.rsplit("/", 1)[-1] for t in tree}
    grep = subprocess.run(git + ["grep", "-c", "--", "--d-model", pin, "--", "src/main/train"],
                          capture_output=True, text=True)
    assert grep.returncode == 0, "the pin's trainer has --d-model"


@pytest.mark.slow
def test_the_fresh_learners_have_the_registered_parameter_counts():
    """A CPU build of the four arms' learners (about fifteen seconds)."""
    assert A.param_counts(PIN) == A.EXPECTED_PARAMS == {
        "c": 2_609_183, "w": 6_258_079, "br": 2_609_183, "wb": 6_258_079}
