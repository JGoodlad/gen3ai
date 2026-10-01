"""Configuration errors found AFTER argument parsing exit ``FATAL_CONFIG`` (3), never ``CRASH`` (1).

`main.exit_codes.FatalConfigError` is mapped by NAME through `exit_code_for`, so a refusal raised
anywhere on the startup path reaches the trainer's fail-fast handlers as exit 3 — which the launcher
does not restart. Before it (verified 2026-09-30 on the parent commit):

* a `--bot-weights` typo exited 1 — the launcher restarted it into the same typo until its
  rapid-crash breaker gave up;
* a failed `--warmstart-consensus` raised an uncaught exception → 1, and because the warm-start is
  rebuilt from scratch on each restart and can run longer than the breaker's 10-minute window, the
  loop had no bound;
* a `--distill-teacher` team that fails gen3ou validation was DROPPED silently by the teambuilder
  (pinned with the real validator and the real trainer in `fatal_config_exits_integration_test.py`).
"""
from __future__ import annotations

import ast

import pytest

from main.exit_codes import FatalConfigError, TrainExitCode, exit_code_for


def test_a_FatalConfigError_maps_to_FATAL_CONFIG_through_any_wrap():
    assert exit_code_for(FatalConfigError("x")) == int(TrainExitCode.FATAL_CONFIG) == 3
    try:
        try:
            raise FatalConfigError("inner")
        except FatalConfigError as e:
            raise RuntimeError("outer") from e
    except RuntimeError as outer:
        assert exit_code_for(outer) == 3
    assert exit_code_for(ValueError("x")) == int(TrainExitCode.CRASH)


def test_the_launcher_stops_on_FATAL_CONFIG_and_restarts_a_CRASH(tmp_path):
    """Why the mapping matters: 1 is restarted into the same failure; 3 is not."""
    from main.launcher.nonfinite_exit_test import _drive
    code, spawned, _ev = _drive(tmp_path / "fatal", int(TrainExitCode.FATAL_CONFIG))
    assert (code, spawned) == (3, 1)
    code, spawned, _ev = _drive(tmp_path / "crash", int(TrainExitCode.CRASH))
    assert spawned == 2


def test_every_bot_weights_refusal_is_FATAL_CONFIG():
    from main.train.matchup_setup import BotWeightsRejected, resolve_bot_weights
    roster = ["heuristic", "staller", "aggressive"]
    assert resolve_bot_weights("staller=3", roster) == [1.0, 3.0, 1.0]
    for bad in ("stallr=3", "staller", "staller=abc", "staller=-1", "staller=nan",
                "heuristic=0,staller=0,aggressive=0"):
        with pytest.raises(BotWeightsRejected) as ei:
            resolve_bot_weights(bad, roster)
        assert exit_code_for(ei.value) == 3, bad


def test_the_bot_weights_path_no_longer_exits_1():
    """`sys.exit(1)` read as CRASH and the launcher restarted into the same typo. The resolver is
    the ONLY bot-weights parse in the trainer's startup."""
    import main.train.matchup_setup as ms
    src = open(ms.__file__, encoding="utf-8").read()
    assert "--bot-weights token" not in src.split("def resolve_bot_weights")[0]
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "build_matchup_and_opponents")
    calls = {getattr(c.func, "id", None) for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert "resolve_bot_weights" in calls


def test_a_failed_warmstart_is_FATAL_CONFIG_and_removes_a_partial_artifact(tmp_path):
    from agents.training.warmstart import WarmstartFailed, warmstart_failed
    partial = tmp_path / "warmstart_consensus.zip"
    partial.write_text("half")
    err = warmstart_failed(FileNotFoundError("no teacher at models/x"), str(partial))
    assert isinstance(err, WarmstartFailed) and exit_code_for(err) == 3
    assert "models/x" in str(err) and not partial.exists()
    # the pre-fix exit for the same underlying error: CRASH, i.e. a restart
    assert exit_code_for(FileNotFoundError("no teacher")) == int(TrainExitCode.CRASH)


def test_the_warmstart_build_is_wrapped_in_the_typed_fatal():
    """`run_consensus_warmstart` (and the teacher resolution before it) must sit inside a `try`
    whose handler raises `warmstart_failed(...)` — revert the wrap and this fails."""
    import main.train.model_build as mb
    tree = ast.parse(open(mb.__file__, encoding="utf-8").read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body_calls = {getattr(getattr(c, "func", None), "id", None)
                      for b in node.body for c in ast.walk(b) if isinstance(c, ast.Call)}
        if "run_consensus_warmstart" not in body_calls or "_ws_resolve" not in body_calls:
            continue
        raised = [r for h in node.handlers for r in ast.walk(h) if isinstance(r, ast.Raise)]
        assert any(isinstance(r.exc, ast.Call) and getattr(r.exc.func, "id", "") ==
                   "warmstart_failed" for r in raised)
        return
    pytest.fail("run_consensus_warmstart is not inside a try that raises warmstart_failed")
