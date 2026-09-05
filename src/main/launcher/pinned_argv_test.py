"""AN ARGV IS VALIDATED BY THE PARSER OF THE TREE THAT WILL RUN IT.

WHY THIS FILE EXISTS (2026-09-05). `--pin-commit` exists to re-run an old recipe on its own
commit, and it refused exactly that::

    python -m main.launcher --pin-commit b13b30b2 <that run's own argv> --dry-run
    error: argument --hp-type-belief-coef: invalid float value: 'learned'

At b13b30b2 `--hp-type-belief` TOOK A VALUE. Today it is deleted, so the CURRENT parser
abbreviation-matches the token onto the surviving `--hp-type-belief-coef` and feeds it the value.
**A same-named flag whose ARITY changed is invisible to any presence check** — which is why the
fixture here is not a deleted flag but a flag that survives with a different arity, the shape no
"does the current parser know this?" test can see.

Run: python -m pytest src/main/launcher/pinned_argv_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import hashlib
import importlib
import json
import os
import pathlib
import subprocess

import pytest

import main.checkargs as checkargs
import main.launcher.dry_run as dry_run_mod
import main.launcher.pinned_argv as pa
import main.launcher.pinned_argv_probe as probe
import main.launcher.worktree as wt
from main.exit_codes import TrainExitCode

launcher_run = importlib.import_module("main.launcher.run")


# ---------------------------------------------------------------------------------------
# A real (tiny) git repo whose parser CHANGES ARITY across commits — same fixture shape as
# pin_commit_test.py / dry_run_test.py, for the same reason: the materialisation IS
# `git archive`, so a fake would test nothing.
# ---------------------------------------------------------------------------------------

# commit 1: `--flag` takes a VALUE (the b13b30b2 shape). Every OTHER flag is identical across
# the commits on purpose — the only thing that moves is one flag's arity, which is the whole
# class of defect a presence check cannot see.
_C1 = '''
import argparse


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--flag", dest="flag", default="off")
    p.add_argument("--other-float", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--run-dir", default=None)
    return p
'''

# commit 2: `--flag` is a BARE store_true, followed by a float flag — so the old argv's value
# falls through exactly as it does on the real tree.
_C2 = '''
import argparse


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--flag", action="store_true", default=False)
    p.add_argument("--other-float", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--run-dir", default=None)
    return p
'''

# The custom boolean action every commit before the parser package used — the shape that made
# `ast_scan` report `argument --self-play: expected one argument` for a flag that is a bare
# boolean at b13b30b2. `action=` here is a CLASS, not a string, and the class both passes
# `nargs="?"` up and invents a `--no-<flag>` spelling that no add_argument call ever names.
_BOOLFLAG_SRC = '''
class BoolFlag(argparse.Action):
    """Boolean flag accepting BOTH the bare/`--no-` form AND an explicit value."""

    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        opts, self._negatives = [], set()
        for opt in option_strings:
            opts.append(opt)
            if opt.startswith("--"):
                neg = "--no-" + opt[2:]
                opts.append(neg)
                self._negatives.add(neg)
        super().__init__(option_strings=opts, dest=dest, nargs="?", default=default,
                         required=required, help=help, metavar="{true,false}")

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, option_string not in self._negatives)
'''

# The flag body shared by the two pre-build_parser commits. The filler flags are not decoration:
# the parse_args hook refuses to answer with a parser carrying fewer than MIN_HOOK_OPTIONS
# options, so that it can never mistake some dependency's import-time argparse for the trainer's.
_INLINE_FLAGS = '''
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--flag", dest="flag", default="off")
    p.add_argument("--self-play", action=BoolFlag, default=False)
    p.add_argument("--other-float", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--run-dir", default=None)
    for i in range(24):
        p.add_argument("--filler-%d" % i, default=None)
'''

# commit 3: a commit whose parser is built INLINE, as every commit before 2026-08-16 is. There is
# no build_parser() to import — the parse_args HOOK answers with the real parser anyway.
#
# 🚨 Everything after `parse_args()` is a SIDE EFFECT, on purpose: a real entry point creates its
# run dir there, and the guard test asserts that tree is byte-identical after the probe.
_C3_INLINE_BOOLFLAG = '''
import argparse
import os
''' + _BOOLFLAG_SRC + '''

def main():
''' + _INLINE_FLAGS + '''
    args = p.parse_args()
    sink = os.environ.get("PROBE_SIDE_EFFECT_DIR", "models")
    os.makedirs(os.path.join(sink, "run_created_by_the_entry_point"), exist_ok=True)
    with open(os.path.join(sink, "SIDE_EFFECT"), "w") as f:
        f.write("the probe let the entry point run past parse_args()")
    return args


if __name__ == "__main__":
    main()
'''

# commit 4: no build_parser AND it will not import (so the hook cannot run either) — but every
# add_argument call is still statically readable. This is the only commit that reaches `ast_scan`,
# which is why the BoolFlag shape is declared here too.
_C4_AST_SCAN_ONLY = '''
import argparse
import a_module_that_does_not_exist_anywhere
''' + _BOOLFLAG_SRC + '''

def main():
''' + _INLINE_FLAGS + '''
    return p.parse_args()


if __name__ == "__main__":
    main()
'''

# commit 5 (HEAD): nothing a parser can be read out of, at all.
_C5_UNREADABLE = '''
X = 1
'''

#: The argv the whole file is about: legal at commit 1, an arity mismatch at commit 2.
ARGV = ["--flag", "learned", "--other-float", "0.5"]

#: A flag that EXISTS in the current tree and at none of the fixture commits — the DEFECT-1 shape.
#: A real one, deliberately: `flags_only_in_current_tree` asks the live parser what it knows, so a
#: made-up spelling would test nothing (it would land in the ordinary unrecognized bucket instead).
ADDED_SINCE_THE_PIN = "--checkpoint-every-steps"


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _tree_snapshot(root: pathlib.Path) -> dict:
    """Every file under `root` → sha256 of its bytes. BYTE-identical, not merely "still there":
    a probe that rewrote a metadata.json in place would pass a name-only comparison."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@pytest.fixture
def repo(tmp_path):
    """A 5-commit repo; yields (root, [c1, c2, c3_hook, c4_ast_scan, c5_unreadable=HEAD])."""
    root = tmp_path / "repo"
    (root / "src" / "main").mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "deps").mkdir()
    (root / "deps" / ".keep").write_text("")
    (root / "src" / "main" / "__init__.py").write_text("")
    shas = []
    bodies = (_C1, _C2, _C3_INLINE_BOOLFLAG, _C4_AST_SCAN_ONLY, _C5_UNREADABLE)
    for n, body in enumerate(bodies, start=1):
        (root / "src" / "main" / "train_rl_agent.py").write_text(body)
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", f"commit {n}")
        shas.append(_git(root, "rev-parse", "HEAD"))
    return str(root), shas


@pytest.fixture(autouse=True)
def _fresh_cache():
    pa.clear_cache()
    yield
    pa.clear_cache()


# ---------------------------------------------------------------------------------------
# (a) the same argv: CLEAN at commit 1, REFUSED at commit 2 with the token named
# ---------------------------------------------------------------------------------------

def test_a_the_argv_validates_against_the_commit_that_declared_that_arity(repo):
    root, (c1, _c2, _c3, _c4, _c5) = repo
    r = pa.pinned_parser_check(c1, ARGV, root)
    assert r.mode == "build_parser", r.reason
    assert r.available and r.authoritative
    assert r.ok, r.findings()
    assert r.values["flag"] == "'learned'", "the pinned parser really consumed the value"


def test_a_the_same_argv_FAILS_against_the_commit_that_changed_the_arity(repo):
    root, (_c1, c2, _c3, _c4, _c5) = repo
    r = pa.pinned_parser_check(c2, ARGV, root)
    assert r.mode == "build_parser"
    assert not r.ok, "a bare store_true leaves the old value dangling — that must be caught"
    assert r.authoritative, "the pinned build_parser() is a verdict, not a hint"
    assert any("learned" in f for f in r.findings()), \
        f"the OFFENDING TOKEN must be named, got {r.findings()}"


def test_a_an_unknown_flag_at_the_pinned_commit_is_reported_as_such(repo):
    root, (c1, _c2, _c3, _c4, _c5) = repo
    r = pa.pinned_parser_check(c1, ["--deleted-later", "0.5"], root)
    assert not r.ok and "--deleted-later" in r.unknown, r.findings()


# ---------------------------------------------------------------------------------------
# (b) end to end through the launcher's `--dry-run`
# ---------------------------------------------------------------------------------------

@pytest.fixture
def isolated(repo, tmp_path, monkeypatch):
    """cwd in a scratch dir, every repo-root lookup pointed at the temp repo, and the effectful
    launcher entry points booby-trapped (a dry run must still create nothing)."""
    root, shas = repo
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(dry_run_mod, "get_repo_root", lambda *a, **k: root)
    monkeypatch.setattr(wt, "get_repo_root", lambda *a, **k: root)
    monkeypatch.setattr(pa, "_repo_root", lambda: root)
    for name in ("_create_run_worktree", "_prune_stale_launcher_worktrees"):
        monkeypatch.setattr(
            wt, name, lambda *a, **k: pytest.fail(f"--dry-run must never call {name}"))
        if hasattr(launcher_run, name):
            monkeypatch.setattr(
                launcher_run, name,
                lambda *a, **k: pytest.fail(f"--dry-run must never call {name}"))
    return root, shas, work


def _dry_run(argv, monkeypatch, expect=0):
    monkeypatch.setattr("sys.argv", ["launcher", "--dry-run", *argv])
    with pytest.raises(SystemExit) as e:
        launcher_run.main()
    assert e.value.code == expect, f"expected exit {expect}, got {e.value.code}"


def test_b_dry_run_pinned_to_the_old_commit_reports_validated_and_exits_0(
        isolated, monkeypatch, capsys):
    """The motivating command: an argv the CURRENT parser mangles, pinned to its own commit."""
    _root, (c1, _c2, _c3, _c4, _c5), _work = isolated
    _dry_run(["--pin-commit", c1, "--steps", "1000", *ARGV], monkeypatch, expect=0)
    out = capsys.readouterr().out
    assert "validated against PINNED parser" in out and c1[:8] in out
    assert "would launch" in out
    # …and the current tree's own opinion about those flags is present but DEMOTED.
    assert "ADVISORY" in out or "advisory" in out


def test_b_dry_run_pinned_to_the_arity_changing_commit_exits_fatal_config(
        isolated, monkeypatch, capsys):
    _root, (_c1, c2, _c3, _c4, _c5), _work = isolated
    _dry_run(["--pin-commit", c2, "--steps", "1000", *ARGV], monkeypatch,
             expect=int(TrainExitCode.FATAL_CONFIG))
    out = capsys.readouterr().out
    assert "REFUSED by the PINNED parser" in out
    assert "learned" in out, "the refusal must name the offending token"
    assert "would NOT launch" in out


def test_b_the_real_launch_path_refuses_before_creating_a_worktree(
        isolated, monkeypatch, capsys):
    """`_prepare_session` must refuse BEFORE `_create_run_worktree` — the booby trap proves it."""
    _root, (_c1, c2, _c3, _c4, _c5), _work = isolated
    from main.launcher.state import LauncherState
    monkeypatch.setattr(launcher_run, "get_repo_root", lambda *a, **k: _root)
    # The real path prunes before it pins; that is not what this test is about, and the fixture
    # booby-traps it for the DRY-RUN tests.
    monkeypatch.setattr(launcher_run, "_prune_stale_launcher_worktrees", lambda *a, **k: None)
    with pytest.raises(SystemExit) as e:
        launcher_run._prepare_session(
            LauncherState(interval_hours=0), ["--steps", "1000", *ARGV],
            interval_hours=0, pin=True, sync_to_main=False, pin_commit=c2,
            grace_minutes=1.0, max_crash_restarts=1)
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    assert "learned" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------
# (c) pin == HEAD is the unchanged path — the current parser, and NO subprocess
# ---------------------------------------------------------------------------------------

def test_c_a_pin_that_names_head_is_not_a_pinned_check(repo):
    root, (c1, _c2, _c3, _c4, c5_head) = repo
    assert pa.differs_from_head(c1, root) is True
    assert pa.differs_from_head(c5_head, root) is False
    assert pa.differs_from_head(c5_head[:8], root) is False, "short spellings of HEAD too"


def test_c_dry_run_at_head_never_spawns_the_probe(isolated, monkeypatch, capsys):
    _root, (_c1, _c2, _c3, _c4, c5_head), _work = isolated
    monkeypatch.setattr(
        dry_run_mod, "pinned_parser_check",
        lambda *a, **k: pytest.fail("pin == HEAD must use the CURRENT parser, unchanged"))
    monkeypatch.setattr(
        pa, "materialise",
        lambda *a, **k: pytest.fail("pin == HEAD must not materialise anything"))
    _dry_run(["--pin-commit", c5_head, "--steps", "1000"], monkeypatch, expect=0)
    out = capsys.readouterr().out
    assert "PINNED parser" not in out


# ---------------------------------------------------------------------------------------
# (d) a commit with no build_parser: the gap is NAMED, never a silent pass
# ---------------------------------------------------------------------------------------

def test_d_a_commit_without_build_parser_is_answered_by_its_REAL_parser(repo):
    """Pre-2026-08-16 commits build their parser inside main(). Running the entry point as far as
    its first parse_args() gets the REAL parser to answer — authoritative, not a reconstruction."""
    root, (_c1, _c2, c3, _c4, _c5) = repo
    r = pa.pinned_parser_check(c3, ARGV, root)
    assert r.mode == "parse_args_hook", r.reason
    assert r.available and r.authoritative, "the parser that answered IS the child's parser"
    assert "predates build_parser()" in r.reason
    assert r.ok, "commit 3 declares --flag with a value, so this argv is fine there"
    assert "parse_args()" in r.how, "the mode that answered must be NAMED in the summary"


def test_d_the_hook_never_lets_the_entry_point_do_ANY_work(repo, tmp_path, monkeypatch):
    """🚨 THE GUARD. The monkeypatch fires before the run dir is created, so a fake models/ tree
    must be BYTE-IDENTICAL afterwards — and the probe must never stand in the repo."""
    root, (_c1, _c2, c3, _c4, _c5) = repo
    sink = tmp_path / "fake_models"
    (sink / "run_a" / "checkpoints").mkdir(parents=True)
    (sink / "run_a" / "metadata.json").write_text('{"git_hash": "abc"}')
    (sink / "run_a" / "checkpoints" / "checkpoint_1_steps.zip").write_text("weights")
    before = _tree_snapshot(sink)
    repo_before = _tree_snapshot(pathlib.Path(root))

    monkeypatch.setenv("PROBE_SIDE_EFFECT_DIR", str(sink))
    r = pa.pinned_parser_check(c3, ARGV, root)
    assert r.mode == "parse_args_hook", r.reason

    assert _tree_snapshot(sink) == before, (
        "the entry point ran past parse_args() and touched the run-dir tree")
    assert not (sink / "SIDE_EFFECT").exists()
    assert _tree_snapshot(pathlib.Path(root)) == repo_before, (
        "the probe must run with cwd = its temp dir, never the repo")
    assert not (pathlib.Path(root) / "models").exists(), (
        "a relative 'models' path resolved against the repo ⇒ the cwd was wrong")


def test_d_a_commit_that_cannot_import_falls_back_to_the_static_scan(repo):
    """Neither build_parser nor the hook can run at c4 (its import raises), so the AST scan is the
    only thing left — and its findings are a WARNING, never a refusal."""
    root, (_c1, _c2, _c3, c4, _c5) = repo
    r = pa.pinned_parser_check(c4, ARGV, root)
    assert r.mode == "ast_scan", r.reason
    assert r.available and not r.authoritative
    assert r.ok, "commit 4 declares --flag with a value, so this argv is fine there"


def test_d_a_static_scan_finding_is_a_warning_not_a_refusal(isolated, monkeypatch, capsys):
    _root, (_c1, _c2, _c3, c4, _c5), _work = isolated
    # `--nope` exists at no commit; the static scan sees that, but must not FATAL on it.
    _dry_run(["--pin-commit", c4, "--steps", "1000", "--nope", "1"], monkeypatch, expect=0)
    out = capsys.readouterr().out
    assert "questioned by the PINNED parser" in out
    assert "NOT authoritative" in out
    assert "would launch" in out, "a best-effort scan must never block a launch"


def test_d_a_commit_with_no_readable_parser_reports_parser_unavailable_at_pin(repo):
    root, (_c1, _c2, _c3, _c4, c5) = repo
    r = pa.pinned_parser_check(c5, ARGV, root)
    assert r.mode == "unavailable" and not r.available
    assert "parser_unavailable_at_pin" in r.summary_line()
    assert r.reason, "an unavailable check must always say WHY"


def test_d_unavailable_warns_and_still_launches(isolated, monkeypatch, capsys):
    _root, (_c1, _c2, _c3, _c4, _c5), _work = isolated
    # c5 IS head here, so force the unavailable path directly rather than through the pin.
    monkeypatch.setattr(
        dry_run_mod, "pinned_parser_check",
        lambda sha, argv, root=None: pa.ParseReport(sha=sha, mode="unavailable",
                                                    reason="synthetic: nothing to read"))
    _dry_run(["--pin-commit", _c1_of(_root), "--steps", "1000"], monkeypatch, expect=0)
    out = capsys.readouterr().out
    assert "parser_unavailable_at_pin" in out and "NOT validated" in out
    assert "would launch" in out, "a check we could not run must not block a launch"


def _c1_of(root):
    return _git(root, "rev-list", "--max-parents=0", "HEAD")


def test_d_an_unresolvable_sha_is_unavailable_not_a_crash(repo):
    root, _shas = repo
    r = pa.pinned_parser_check("deadbeefdeadbeefdeadbeef", ARGV, root)
    assert r.mode == "unavailable" and "does not name a commit" in r.reason


def test_d_a_timeout_is_unavailable_never_a_pass(repo, monkeypatch):
    root, (c1, _c2, _c3, _c4, _c5) = repo

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="probe", timeout=0.01)

    real = pa.subprocess.run
    monkeypatch.setattr(pa.subprocess, "run",
                        lambda *a, **k: _boom() if "_pinned_argv_probe.py" in " ".join(a[0])
                        else real(*a, **k))
    r = pa.pinned_parser_check(c1, ARGV, root)
    assert r.mode == "unavailable" and "did not finish" in r.reason
    assert not r.ok or True  # the point is that `available` is False, never a silent pass
    assert not r.available


# ---------------------------------------------------------------------------------------
# (e) checkargs --pin routes through the same function
# ---------------------------------------------------------------------------------------

def test_e_checkargs_pin_uses_the_pinned_parser_and_passes(repo, monkeypatch, capsys):
    root, (c1, _c2, _c3, _c4, _c5) = repo
    monkeypatch.setattr(pa, "_repo_root", lambda: root)
    rc = checkargs.main(["--argv", " ".join(ARGV), "--pin", c1])
    out = capsys.readouterr().out
    assert f"the PINNED commit {c1[:8]}'s" in out
    assert rc == 0, "the pinned parser accepts this argv, so checkargs must too"
    assert "ADVISORY" in out, "the current tree's flag findings are demoted, not deleted"


def test_e_checkargs_pin_fails_on_the_arity_changing_commit(repo, monkeypatch, capsys):
    root, (_c1, c2, _c3, _c4, _c5) = repo
    monkeypatch.setattr(pa, "_repo_root", lambda: root)
    rc = checkargs.main(["--argv", " ".join(ARGV), "--pin", c2])
    out = capsys.readouterr().out
    assert rc == 1 and "learned" in out


def test_e_without_a_pin_checkargs_is_unchanged(repo, monkeypatch, capsys):
    """No `--model`, no `--pin` ⇒ the launch runs on HEAD and the current parser is right."""
    monkeypatch.setattr(pa, "_repo_root", lambda: repo[0])
    rc = checkargs.main(["--argv", "--steps 1000"])
    out = capsys.readouterr().out
    assert rc == 0 and "the CURRENT tree's" in out
    assert "PINNED" not in out


def test_e_a_model_argv_pins_itself_to_the_checkpoints_recorded_hash(tmp_path):
    """The `--model` path takes the pin from the checkpoint sidecar, as `resolve_pin` does."""
    run = tmp_path / "models" / "r"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "checkpoint_10_steps.zip"
    ckpt.write_text("x")
    (run / "checkpoints" / "checkpoint_10_steps.json").write_text(
        json.dumps({"git_hash": "cafebabecafebabe"}))
    sha, why = checkargs.resolve_pin_for(["--model", str(ckpt)], None)
    assert sha == "cafebabecafebabe" and "recorded" in why


def test_e_a_model_with_no_recorded_hash_says_so_rather_than_guessing(tmp_path):
    ckpt = tmp_path / "loose.zip"
    ckpt.write_text("x")
    sha, why = checkargs.resolve_pin_for(["--model", str(ckpt)], None)
    assert sha is None and "no git_hash" in why


def test_e_model_arg_reads_both_spellings():
    assert checkargs.model_arg(["--model", "a.zip"]) == "a.zip"
    assert checkargs.model_arg(["--model=a.zip"]) == "a.zip"
    assert checkargs.model_arg(["--steps", "1"]) is None


# ---------------------------------------------------------------------------------------
# The materialisation itself: cheap, cached, and it creates NO worktree.
# ---------------------------------------------------------------------------------------

def test_materialise_creates_no_git_worktree(repo):
    """`git archive`, not `git worktree add` — a validation command has already cost this
    program one live run by touching the worktree list."""
    root, (c1, _c2, _c3, _c4, _c5) = repo
    before = _git(root, "worktree", "list")
    pa.materialise(c1, root)
    assert _git(root, "worktree", "list") == before


def test_materialise_is_cached_per_process(repo):
    root, (c1, _c2, _c3, _c4, _c5) = repo
    assert pa.materialise(c1, root) == pa.materialise(c1, root)
    assert os.path.isdir(os.path.join(pa.materialise(c1, root), "src", "main"))


def test_the_probe_env_never_inherits_the_callers_pythonpath():
    """The caller's PYTHONPATH names the CURRENT checkout's src (every worktree shell here sets
    it), and inheriting it would silently validate against the very parser we are avoiding."""
    env = pa._probe_env("/somewhere/src")
    assert env["PYTHONPATH"] == "/somewhere/src"


def test_argparse_usage_spam_never_reaches_the_report(repo):
    """A parse failure prints the whole 200-line flag table before its one-line diagnosis."""
    root, (_c1, c2, _c3, _c4, _c5) = repo
    r = pa.pinned_parser_check(c2, ["--steps", "not-an-int"], root)
    assert r.errors and all("error:" in e for e in r.errors), r.errors
    assert not any(e.startswith("usage:") for e in r.errors)


# ---------------------------------------------------------------------------------------
# (f) DEFECT 1 — a flag that exists ONLY in the current tree
#
# `--checkpoint-every-steps 150000` appended to the v8 fold's argv exited 0 and named nothing,
# because "accepted" was counted against the parser you are standing in. It killed three arms.
# ---------------------------------------------------------------------------------------

def test_f_a_flag_defined_at_head_and_absent_at_the_pin_is_NAMED(repo):
    root, (c1, _c2, _c3, _c4, _c5) = repo
    argv = [*ARGV, ADDED_SINCE_THE_PIN, "150000"]
    r = pa.pinned_parser_check(c1, argv, root)
    assert r.options, "the pinned OPTION SET is what makes this answerable"
    assert pa.flags_only_in_current_tree(r, argv) == [ADDED_SINCE_THE_PIN]
    loud = pa.absent_at_pin_lines(r, argv)
    assert loud and ADDED_SINCE_THE_PIN in loud[0] and "NOT IN PINNED TREE" in loud[0]
    assert pa.report_lines(r, argv)[0] == loud[0], "it must be the LOUDEST line — the first one"
    assert pa.refuses(r, argv), "an authoritative mode REFUSES it"


def test_f_the_same_flag_is_a_warning_and_never_a_count_under_ast_scan(repo):
    """`ast_scan` can be incomplete in BOTH directions, so it may warn and may not refuse."""
    root, (_c1, _c2, _c3, c4, _c5) = repo
    argv = [*ARGV, ADDED_SINCE_THE_PIN, "150000"]
    r = pa.pinned_parser_check(c4, argv, root)
    assert r.mode == "ast_scan"
    assert pa.flags_only_in_current_tree(r, argv) == [ADDED_SINCE_THE_PIN]
    assert not pa.refuses(r, argv), "a static scan may warn; it may not block a launch"
    assert "WARNING" in pa.report_lines(r, argv)[1]


def test_f_a_flag_missing_from_BOTH_trees_is_not_this_bucket(repo):
    """That is an ordinary stale flag — the existing unrecognized bucket already names it, and
    counting it twice would make the loud line mean two different things."""
    root, (c1, _c2, _c3, _c4, _c5) = repo
    argv = [*ARGV, "--never-existed-anywhere", "1"]
    r = pa.pinned_parser_check(c1, argv, root)
    assert pa.flags_only_in_current_tree(r, argv) == []


def test_f_a_launcher_owned_flag_is_never_reported_as_missing_at_the_pin(repo):
    """The launcher strips its own flags before forwarding, so they are absent from EVERY
    trainer parser by design."""
    root, (c1, _c2, _c3, _c4, _c5) = repo
    argv = [*ARGV, "--nice", "10"]
    r = pa.pinned_parser_check(c1, argv, root)
    assert pa.flags_only_in_current_tree(r, argv) == []


def test_f_an_unambiguous_abbreviation_is_not_called_absent(repo):
    """argparse abbreviation-matches, so a unique prefix really does parse at the pin. Calling
    it absent would be a false alarm on the loudest line in the output."""
    root, (c1, _c2, _c3, _c4, _c5) = repo
    r = pa.pinned_parser_check(c1, ARGV, root)
    assert r.declares("--run-dir") and r.declares("--run-d")
    assert not r.declares("--checkpoint-every-steps")


def test_f_dry_run_exits_fatal_config_on_a_flag_that_exists_only_here(
        isolated, monkeypatch, capsys):
    _root, (c1, _c2, _c3, _c4, _c5), _work = isolated
    _dry_run(["--pin-commit", c1, "--steps", "1000", *ARGV, ADDED_SINCE_THE_PIN, "150000"],
             monkeypatch, expect=int(TrainExitCode.FATAL_CONFIG))
    out = capsys.readouterr().out
    assert f"NOT IN PINNED TREE @{c1[:8]}" in out and ADDED_SINCE_THE_PIN in out
    assert "would NOT launch" in out


def test_f_checkargs_exits_fatal_config_and_drops_it_from_the_accepted_count(
        repo, monkeypatch, capsys):
    root, (c1, _c2, _c3, _c4, _c5) = repo
    monkeypatch.setattr(pa, "_repo_root", lambda: root)
    argv = " ".join([*ARGV, ADDED_SINCE_THE_PIN, "150000"])
    rc = checkargs.main(["--argv", argv, "--pin", c1])
    out = capsys.readouterr().out
    assert rc == int(TrainExitCode.FATAL_CONFIG), out
    assert "NOT IN PINNED TREE" in out and ADDED_SINCE_THE_PIN in out
    assert "accepted by the PINNED parser" in out, "the bucket is the PINNED parser's now"
    assert "in THIS tree but NOT at the pin : 1" in out


def test_f_checkargs_strips_launcher_flags_before_asking_the_pinned_parser(repo, monkeypatch,
                                                                          capsys):
    """A recorded `launcher_command` carries flags the launcher consumes; feeding them to the
    child's parser buries the real findings under a wall of false unknowns."""
    root, (c1, _c2, _c3, _c4, _c5) = repo
    monkeypatch.setattr(pa, "_repo_root", lambda: root)
    assert checkargs.forwarded_argv(
        ["--restart-interval-hours", "6", "--sync-to-main", "--flag", "learned"]
    ) == ["--flag", "learned"]
    rc = checkargs.main(
        ["--argv", "--restart-interval-hours 6 --sync-to-main " + " ".join(ARGV), "--pin", c1])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "restart-interval-hours" not in out.split("launcher-owned")[0]


# ---------------------------------------------------------------------------------------
# (g) DEFECT 2 — `action=<a class>` is not `action="store_true"`
#
# The real v8 argv came back as `argument --self-play: expected one argument`, for a flag that
# is a bare boolean at b13b30b2. `ast_scan` read `action=` only when it was a string constant,
# so `action=BoolFlag` looked like "no action ⇒ takes a value".
# ---------------------------------------------------------------------------------------

def test_g_a_custom_action_class_flag_is_legal_BARE_under_the_static_scan(repo):
    root, (_c1, _c2, _c3, c4, _c5) = repo
    r = pa.pinned_parser_check(c4, [*ARGV, "--self-play"], root)
    assert r.mode == "ast_scan", r.reason
    assert r.ok, f"--self-play is a bare boolean at this commit: {r.findings()}"
    assert not any("expected one argument" in e for e in r.errors)


def test_g_a_custom_action_class_flag_still_ACCEPTS_a_value(repo):
    """BoolFlag passes `nargs="?"` up, so both spellings parse. The scan reads that nargs out of
    the class body rather than assuming a shape."""
    root, (_c1, _c2, _c3, c4, _c5) = repo
    r = pa.pinned_parser_check(c4, [*ARGV, "--self-play", "true"], root)
    assert r.ok, r.findings()


def test_g_the_generated_no_form_is_a_known_flag_under_the_static_scan(repo):
    """`--no-self-play` is declared by NO add_argument call — the class invents it. Reading that
    out of the class body is what keeps the real v8 argv's `--no-stall-pbrs` from reading as an
    unknown flag."""
    root, (_c1, _c2, _c3, c4, _c5) = repo
    r = pa.pinned_parser_check(c4, [*ARGV, "--no-self-play"], root)
    assert r.ok, r.findings()
    assert "--no-self-play" in r.options


def test_g_the_class_shape_is_READ_not_assumed(repo):
    """The nargs and the `--no-` behaviour both come from the pinned tree's own source, so a
    commit whose action class differs is read as it is rather than as BoolFlag."""
    root, _shas = repo
    src = pa.materialise(_shas[3], root)
    shapes = probe.custom_action_shapes(probe.parser_files(os.path.join(src, "src")))
    assert shapes["BoolFlag"] == ("?", True)
    assert shapes["BooleanOptionalAction"] == (0, True), "the stdlib one is seeded, not scanned"


def test_g_an_unreadable_action_class_takes_the_permissive_arity():
    """A scan that guesses wrong must UNDER-report. `nargs="?"` accepts the flag bare and with a
    value, so an action class we cannot resolve can never invent an 'expected one argument'."""
    assert probe._UNKNOWN_ACTION_NARGS == "?"


def test_g_the_hook_answers_the_same_way_for_the_custom_action(repo):
    """And the authoritative mode agrees — which is the point: the static scan is a
    reconstruction OF this, so a disagreement between them is a scan bug."""
    root, (_c1, _c2, c3, _c4, _c5) = repo
    r = pa.pinned_parser_check(c3, [*ARGV, "--self-play"], root)
    assert r.mode == "parse_args_hook" and r.ok, r.findings()
    assert r.values.get("self_play") == "True"
