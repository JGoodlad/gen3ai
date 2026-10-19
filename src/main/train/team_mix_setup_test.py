"""``--team-mix`` at the trainer's surfaces (``gen3_team_mix_v1``; designs/training/team_curriculum.md "The team
mix"): the parser, ``main.checkargs``, the combination checks, the startup guard (held-out contamination, validator
drops, an unverifiable corpus), the same-run-restore / fork-does-not-inherit rule and the launcher's restart argv.

Pure unit tests over a tiny corpus in ``tmp_path`` (``agents.training.team_mix_testkit``); the Node validator is
mocked where a builder is built. Each test names the revert it fails on.
"""
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from agents.training.matchup_spec import MatchupSpec
from agents.training.team_mix_testkit import TEST_CORPUS, clear_corpus_cache, install_corpus
from main.train import matchup_setup as MS
from main.train.combination_checks import failing_checks
from utils.teambuilder_test import TEAM_A, TEAM_B


@pytest.fixture(autouse=True)
def _fresh_corpus_cache():
    clear_corpus_cache()
    yield
    clear_corpus_cache()


@pytest.fixture
def all_valid():
    with mock.patch("utils.bridge.team_validator.validate_teams_locally",
                    side_effect=lambda fmt, ts: [{"valid": True} for _ in ts]):
        yield


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    return install_corpus(monkeypatch, tmp_path, [TEAM_B, TEAM_A], tiers={"A": 1, "AB": 2})


def _parse(argv):
    from main.train.parser import build_parser

    return build_parser().parse_args(argv)


def _args(**kw):
    base = dict(trainee_team=None, trainee_teams=None, exploiter=None, self_play=False, bot_weights=None,
                exploiter_keep_bots=False, exploiter_bot_fraction=0.5, stable_opponent_temp=1.0,
                team_mix=None, team_mix_sides=None, model=None, run_dir=None, run_name=None)
    base.update(kw)
    return SimpleNamespace(**base)


# ── the parser and checkargs ───────────────────────────────────────────────────────────────────────────────────

def test_parser_takes_the_flag_as_typed_and_defaults_to_no_mix():
    ns = _parse(["--team-mix", "w_forum_v1@ABC:0.5", "--team-mix-sides", "opponent"])
    assert ns.team_mix == "w_forum_v1@ABC:0.5" and ns.team_mix_sides == "opponent"
    ns = _parse([])
    assert ns.team_mix is None and ns.team_mix_sides is None


@pytest.mark.parametrize("argv", [["--team-mix", "nope_v9:0.5"], ["--team-mix", "w_forum_v1:2"],
                                  ["--team-mix", "w_forum_v1"], ["--team-mix-sides", "left"]])
def test_parser_refuses_a_bad_value(argv, capsys):
    """Fails on revert of the ``type=`` syntax check (a bad value would parse and die only at startup)."""
    with pytest.raises(SystemExit) as e:
        _parse(argv)
    assert e.value.code == 2


def test_checkargs_accepts_the_flag_and_refuses_a_bad_value_offline():
    from main.checkargs import check

    good = check(["--arch", "production", "--team-mix", "w_forum_v1:0.5", "--self-play"])
    assert good["unknown"] == [] and good["ns"] is not None and good["ns"].team_mix == "w_forum_v1:0.5"
    assert not [c for c, _ in good["combinations"] if c.name.startswith("team_mix")]
    bad = check(["--arch", "production", "--team-mix", "w_forum_v1:1.5"])
    assert bad["ns"] is None and "--team-mix" in " ".join(bad["resolution"]["parse_error"])
    combo = check(["--arch", "production", "--team-mix", "w_forum_v1:0.5", "--exploiter", "models/x"])
    assert "team_mix_trainee_side_vs_exploiter" in [c.name for c, _ in combo["combinations"]]


# ── the combination checks ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw, refused", [
    (dict(team_mix_sides="opponent"), "team_mix_sides_needs_team_mix"),
    (dict(team_mix="w_forum_v1:0.5", exploiter="models/x"), "team_mix_trainee_side_vs_exploiter"),
    (dict(team_mix="w_forum_v1:0.5", exploiter="models/x", team_mix_sides="trainee"),
     "team_mix_trainee_side_vs_exploiter"),
    (dict(team_mix="w_forum_v1:0.5", trainee_team="x.txt"), "team_mix_trainee_side_vs_pinned_trainee"),
    (dict(team_mix="w_forum_v1:0.5", trainee_teams="x.txt,y.txt", team_mix_sides="both"),
     "team_mix_trainee_side_vs_pinned_trainee"),
    (dict(team_mix="w_forum_v1:0.5", exploiter="models/x", trainee_team="x.txt", team_mix_sides="opponent"), None),
    (dict(team_mix="w_forum_v1:0.5"), None),
    (dict(), None),
])
def test_combination_checks(kw, refused):
    ns = _parse([])
    vars(ns).update(kw)
    names = [c.name for c in failing_checks(ns) if c.name.startswith("team_mix")]
    assert names == ([refused] if refused else [])


# ── the startup guard ──────────────────────────────────────────────────────────────────────────────────────────

def _builders(spec):
    return spec.trainee_teams.build([TEAM_A], [TEAM_A]), spec.opponent_teams.build([TEAM_A], [TEAM_A])


def test_guard_is_silent_without_a_mix(all_valid):
    spec = MatchupSpec.from_args(_args())
    assert MS.team_mix_guard(spec, *_builders(spec)) == []


def test_guard_echoes_a_clean_mix(corpus, all_valid):
    spec = MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@AB:0.5"))
    lines = MS.team_mix_guard(spec, *_builders(spec))
    assert lines[0].startswith(f"🧬 [TEAM-MIX] {TEST_CORPUS}@AB (2 teams, tier sha ")
    assert "share 0.50 on both sides; eval stays on the pool" in lines[0]
    assert "held-out panels: 0 hits" in lines[1] and "1 corpus team(s) pack identically to a pool team" in lines[1]
    spec = MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@AB:0.5", team_mix_sides="opponent"))
    assert "on the opponent side only" in MS.team_mix_guard(spec, *_builders(spec))[0]


def test_guard_refuses_a_corpus_that_hits_a_held_out_panel(monkeypatch, tmp_path, all_valid):
    """A corpus holding a held-out panel's team is FATAL_CONFIG, naming the panel. Fails on revert of the
    contamination check."""
    from main import heldout_panels as HP
    from main.exit_codes import TrainExitCode, exit_code_for

    leaked = HP.panel_team_texts("heldout72")[0][1]
    install_corpus(monkeypatch, tmp_path, [TEAM_B, leaked])
    spec = MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}:0.5"))
    with pytest.raises(MS.TeamMixRefused, match=r"1 corpus team\(s\) .* hit a HELD-OUT panel.*heldout72") as e:
        MS.team_mix_guard(spec, *_builders(spec))
    assert exit_code_for(e.value) == int(TrainExitCode.FATAL_CONFIG)


def test_guard_refuses_a_validator_drop(corpus):
    """A corpus team the validator rejects is FATAL (the record would name teams the run never draws). Fails on
    revert of the ``mix_dropped`` check."""
    def verdicts(fmt, ts):
        return [{"valid": not (len(ts) == 2 and i == 1), "errors": ["banned"]} for i, _ in enumerate(ts)]

    spec = MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@AB:0.5", team_mix_sides="opponent"))
    with mock.patch("utils.bridge.team_validator.validate_teams_locally", side_effect=verdicts):
        tt, ot = _builders(spec)
    assert ot.mix_dropped == [(1, ["banned"])]
    with pytest.raises(MS.TeamMixRefused, match="DROPPED 1 of 2 corpus team"):
        MS.team_mix_guard(spec, tt, ot)


def test_startup_turns_an_unverifiable_corpus_into_fatal_config(corpus):
    """``build_matchup_and_opponents`` wraps the loader's ``CorpusError`` (here: an unknown tier) into
    FATAL_CONFIG before any builder exists. Fails on revert of the wrap (a bare ``CorpusError`` exits CRASH, which
    the launcher restarts into the same error)."""
    from main.exit_codes import TrainExitCode, exit_code_for

    with pytest.raises(MS.TeamMixRefused, match="tier 'Q'") as e:
        MS.build_matchup_and_opponents(_args(team_mix=f"{TEST_CORPUS}@Q:0.5"))
    assert exit_code_for(e.value) == int(TrainExitCode.FATAL_CONFIG)


# ── restart vs fork ────────────────────────────────────────────────────────────────────────────────────────────

def _run(tmp_path, name, cli):
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "ckpt_100.zip"
    ckpt.write_bytes(b"")
    (run / "metadata.json").write_text(json.dumps({"cli_args": cli}))
    return run, ckpt


def test_a_same_run_resume_restores_the_recorded_mix(tmp_path):
    """Fails on revert of ``inherit_team_mix`` (a hand-typed resume would train the rest of a mix arm on the
    pool)."""
    run, ckpt = _run(tmp_path, "rb_mix", {"team_mix": "w_forum_v1@ABC:0.5", "team_mix_sides": "opponent"})
    args, lines = _args(model=str(ckpt), run_dir=str(run)), []
    assert MS.inherit_team_mix(args, emit=lines.append) == "restored"
    assert args.team_mix == "w_forum_v1@ABC:0.5" and args.team_mix_sides == "opponent"
    assert "restoring the run's recorded --team-mix" in lines[0]


def test_a_fork_does_not_inherit_the_mix(tmp_path):
    run, ckpt = _run(tmp_path, "rb_mix", {"team_mix": "w_forum_v1@ABC:0.5", "team_mix_sides": None})
    args, lines = _args(model=str(ckpt), run_dir=str(tmp_path / "rb_fork")), []
    assert MS.inherit_team_mix(args, emit=lines.append) == "fork_not_inherited"
    assert args.team_mix is None and args.team_mix_sides is None
    assert "does NOT inherit" in lines[0]


def test_a_typed_mix_wins_and_a_parent_without_one_gives_nothing(tmp_path):
    run, ckpt = _run(tmp_path, "rb_mix", {"team_mix": "w_forum_v1@ABC:0.5"})
    args = _args(model=str(ckpt), run_dir=str(run), team_mix="w_forum_v1@A:0.25")
    assert MS.inherit_team_mix(args, emit=lambda s: None) is None and args.team_mix == "w_forum_v1@A:0.25"
    # a fork of a parent that never had the flag (X26's case) and a same-run resume of one: nothing changes
    run2, ckpt2 = _run(tmp_path, "rb_x26_like", {"self_play": True})
    for rd in (str(run2), str(tmp_path / "rb_fork2")):
        a = _args(model=str(ckpt2), run_dir=rd)
        assert MS.inherit_team_mix(a, emit=lambda s: None) is None and a.team_mix is None


def test_the_launchers_restart_argv_keeps_the_flag():
    """A same-run restart re-launches the RESUME-role argv; ``--team-mix`` is not fresh-only, so it survives
    verbatim (with ``--arch`` stripped and ``--model`` / ``--run-dir`` set)."""
    from main.launcher.checkpoint import resume_child_args

    argv = ["--arch", "production", "--team-mix", "w_forum_v1@ABC:0.5", "--team-mix-sides", "both", "--steps", "9"]
    out, removed = resume_child_args(argv, "/m/rb_x/checkpoints/c.zip", "/m/rb_x")
    assert out[out.index("--team-mix") + 1] == "w_forum_v1@ABC:0.5"
    assert out[out.index("--team-mix-sides") + 1] == "both"
    assert "--arch" not in out and removed
