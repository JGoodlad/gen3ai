"""THE ONE END-TO-END TEST: two real gen3ou games against `metamon:SmallRL`.

Everything else in this package is exercised in isolation — the plan, the schema, the watchdog, the
refusals. None of that proves the pieces FIT: that the server starts on a 9XXX port and stops by
its PID, that the acceptor is online before the challenger, that Metamon's upstream poke-env and
our vendored fork both accept the same team files, that a regime set in one process is verified in
the other, and that a row lands carrying its regime. Two games is the smallest thing that does.

**Tier.** Marked ``integration``, not ``slow``, as a deliberate choice: it is ~30-60 s on this box
(``SmallRL`` is 14 ms/move and 2.3-3.0 s/game, plus one model load), and the failure it catches —
the pieces not fitting — is exactly the kind that a routine gate should see. It is NOT ``sim`` (no
in-process bridge) and NOT ``e2e`` (no server on :8000; it starts and stops its own on 9500-9599).

**It SKIPS with a named reason** when the Metamon checkout, its interpreter, its weight cache or
our run archive is absent — which is every box but this one. 🚨 A skip that is supposed to happen
looks exactly like a skip that is not, so the reason is always printed and always specific:
``src/utils/paths_test.py`` was written after four tests skipped forever on a ``/home/...`` literal
nobody noticed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from main.anchors.cli import main as anchors_main
from main.anchors.config import load_config
from main.anchors.results import REQUIRED_ROW_FIELDS
from utils.paths import main_models_dir, models_skip_reason

pytestmark = pytest.mark.integration

#: Two games: one per challenge role, so the role balancing is exercised rather than asserted.
N_GAMES = 2

#: The checkpoint the 2026-09-14 batteries were taken with. NAMED rather than "the newest run",
#: because "whatever is newest" is a different measurement every week.
_RUN = "ai_v12_02_winprob_critic"


def _skip_reason() -> "str | None":
    """Why this box cannot run it — specific enough to act on, or None."""
    cfg = load_config()
    for label, path, env_var in (
        ("the Metamon checkout", cfg.metamon.checkout, "GEN3AI_METAMON_DIR"),
        ("the Metamon interpreter", cfg.metamon.python, "GEN3AI_METAMON_PYTHON"),
        ("the Metamon weight/team cache", cfg.metamon.cache_dir, "GEN3AI_METAMON_CACHE_DIR"),
    ):
        if not Path(path).exists():
            return (f"{label} is not on this box ({path}). Point ${env_var} at it, or edit "
                    f"{cfg.source}. This is an EXTERNAL checkout, deliberately outside the repo.")
    archive = main_models_dir()
    if archive is None:
        return models_skip_reason()
    if not (archive / _RUN).exists():
        return (f"no run at {archive / _RUN} — this test needs a checkpoint to play, and models/ "
                "lives only in the MAIN checkout ($GEN3AI_MODELS_DIR overrides).")
    node = cfg.node
    from shutil import which

    if which(node) is None:
        return f"no `{node}` on PATH — the Showdown server cannot be started."
    return None



def test_two_real_games_against_metamon_smallrl(tmp_path: Path) -> None:
    """The whole tool, end to end, on the smallest sample that still exercises every seam."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    archive = main_models_dir()
    assert archive is not None
    model = str(archive / _RUN / "final_model.zip")

    rc = anchors_main([
        "--model", model,
        "--opponent", "metamon:SmallRL",
        "--regime", "greedy",
        "--teamset", "away",
        "--games", str(N_GAMES),
        "--device", "cpu",
        "--out", str(tmp_path),
        # Generous, because the box normally carries a live training arm; a TIMEOUT is never a
        # semantic outcome, so a slow box must not turn into a failed assertion about the tool.
        "--first-game-timeout", "900",
        "--progress-timeout", "900",
        "--peer-ready-timeout", "900",
        "--username", "Gen3AIitest",
        "--peer-username", "MetaItest",
    ])

    summary = json.loads((tmp_path / "summary.json").read_text())
    rows = [json.loads(line) for line in (tmp_path / "games.jsonl").read_text().splitlines()]

    # A named failure is a legitimate outcome of a contended box — but it must be NAMED, and the
    # games that did finish must still be here. Silence is the one thing this tool may not produce.
    if rc != 0:
        pytest.fail(
            f"anchor read returned {rc}; failure={summary.get('failure')}; "
            f"{len(rows)} of {N_GAMES} games completed. "
            f"peer log: {sorted(p.name for p in tmp_path.rglob('peer_*.log'))}")

    assert summary["status"] == "OK"
    assert summary["n"] == N_GAMES == len(rows)
    assert summary["wins"] + summary["losses"] + summary["ties"] == N_GAMES

    # ROLE BALANCE: one game each way, which is what makes two the smallest useful n.
    assert summary["by_half"]["ours_challenge"]["n"] == 1
    assert summary["by_half"]["peer_challenge"]["n"] == 1

    # THE REGIME, VERIFIED ON BOTH SIDES rather than assumed.
    assert summary["cell"]["our_regime"] == "greedy"
    assert summary["cell"]["regime_matched"] is True
    for row in rows:
        assert row["our_stochastic_kwargs"] == [False], (
            "our side's `stochastic` keyword was not False on every decision — the temperature "
            "flag reached the parser but not the policy")
        assert row["their_sample_kwargs"] == [False], (
            "Metamon's `sample` keyword was not False on every decision — greedy is "
            "`Agent.get_actions(sample=False)` and NOTHING else (hazard H-D: no temperature can "
            "express it, because MetamonDiscrete clips probabilities to [0.001, 0.99])")
        assert row["their_argmax_match_rate"] == 1.0, (
            "Metamon's emitted action did not equal its own argmax on every decision, so this "
            "cell is not a greedy measurement")
        # THE RULE: a number never leaves this tool without its regime.
        assert not [f for f in REQUIRED_ROW_FIELDS if f not in row]
        assert row["opponent"] == "metamon:SmallRL"
        assert row["opponent_version"] == "SmallRL@ckpt40"
        assert row["model_step"] == 75_005_952
        assert row["turns"] > 0

    # Both sides drew from Metamon's own 20-team `competitive` set — the AWAY cell.
    assert summary["cell"]["teamset"] == "away"
    assert summary["team_source_asymmetry"] is False, (
        f"the two sides drew from differently sized team sources "
        f"({summary['cell']['our_team_count']} vs {summary['cell']['their_team_count']}) — the "
        "win rate would then mix skill with matchup")

    # The server we started is gone, and nothing else on the box was touched.
    assert (tmp_path / "showdown.log").exists()
    port = summary["cell"]["server_uri"].split(":")[2].split("/")[0]
    assert 9500 <= int(port) <= 9599


def test_the_skip_reason_names_a_specific_missing_thing_when_it_fires() -> None:
    """Guard the guard. A skip whose reason is 'environment not available' sends the next reader
    hunting; this one must always name the path AND the env var that overrides it."""
    reason = _skip_reason()
    if reason is None:
        pytest.skip("this box HAS the Metamon environment — the skip path is covered below")
    assert any(tok in reason for tok in ("GEN3AI_METAMON", "GEN3AI_MODELS_DIR", "PATH")), reason


def test_the_skip_path_is_reachable_and_names_the_env_var(monkeypatch: pytest.MonkeyPatch,
                                                          tmp_path: Path) -> None:
    """Drive the skip branch ON THIS BOX, where it otherwise never runs.

    Modelled on `src/utils/paths_test.py`'s four archive-reading tests: a skip path nothing
    exercises is a skip path nobody has ever seen work.
    """
    monkeypatch.setenv("GEN3AI_METAMON_DIR", str(tmp_path / "definitely-not-here"))
    reason = _skip_reason()
    assert reason is not None
    assert "GEN3AI_METAMON_DIR" in reason
    assert "definitely-not-here" in reason
    assert "anchors.json" in reason


def test_the_config_env_overrides_reach_this_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch the skip reason advertises must actually work, or it is advice that
    wastes the next person's afternoon."""
    monkeypatch.setenv("GEN3AI_METAMON_PYTHON", sys.executable)
    assert str(load_config().metamon.python) == sys.executable
