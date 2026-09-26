"""THE ONE END-TO-END TEST: two real gen3ou games against `metamon:SmallRL` — on BOTH transports.

Everything else in this package is exercised in isolation — the plan, the schema, the watchdog, the
refusals. None of that proves the pieces FIT: that the server — the RUST websocket front end by
default, the Node server under `--server node` — starts on a 9XXX port and stops by its PID, that the acceptor is online before the challenger, that Metamon's upstream poke-env and
our vendored fork both accept the same team files, that a regime set in one process is verified in
the other, and that a row lands carrying its regime. Two games is the smallest thing that does.

**Tier.** Marked ``integration``, not ``slow``, as a deliberate choice: it is ~30-60 s on this box
(``SmallRL`` is 14 ms/move and 2.3-3.0 s/game, plus one model load), and the failure it catches —
the pieces not fitting — is exactly the kind that a routine gate should see. It is NOT ``sim`` (no
in-process bridge) and NOT ``e2e`` (no server on :8000; it starts and stops its own on 9500-9599).

**Both transports, one test, parametrized.** The default (`rust`) is the one the owner's
direction put in the hot path and the one every read now takes unless it opts out; `node` is kept
green because it is the reference a transport differential is taken against, and a reference
nobody runs rots. The Node case additionally needs `node` on PATH and the submodule's build
artifacts; the rust case needs neither.

**Our side is a FRESHLY BUILT current-architecture checkpoint**, saved to ``tmp_path`` through the
project's own save path (a real ``MaskablePPO.save`` + ``save_model_snapshot``). It used to play the
named archived run ``ai_v12_02_winprob_critic``; gen3_event_record_v2 (the observation-architecture
batch, v121) put every archived run behind the PRE-GENERATION wall, and this test is about the
pieces FITTING, not about any one run's strength — so it builds the one thing that is guaranteed
to load on the tree under test, and no longer needs the run archive at all.

**It SKIPS with a named reason** when the Metamon checkout, its interpreter or its weight cache is
absent — which is every box but this one. 🚨 A skip that is supposed to happen
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
pytestmark = pytest.mark.integration

#: Two games: one per challenge role, so the role balancing is exercised rather than asserted.
N_GAMES = 2

#: The step the freshly built checkpoint DECLARES in its zip — non-zero and distinctive, so the
#: row's `model_step` is checked against a value the tool could only have read from our file.
_FRESH_STEP = 4_096


def _save_current_generation_checkpoint(run_dir: Path) -> str:
    """A current-architecture checkpoint built fresh (untrained weights) and saved through the
    project's own path: a real `MaskablePPO.save` plus `save_model_snapshot`'s
    model_config.json / metadata.json beside it. Returns the `.zip` path."""
    import gymnasium as gym
    import numpy as np
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.model_version import ModelVersion
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.model.snapshot import save_model_snapshot
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    total_dim = layout["total_dim"]
    obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (total_dim,), np.float32),
        "action_mask": gym.spaces.MultiBinary(11)})

    class _E(gym.Env):
        observation_space = obs_space
        action_space = gym.spaces.Discrete(11)

        def reset(self, **kwargs):
            return {"observation": np.zeros(total_dim, np.float32),
                    "action_mask": np.ones(11, np.int8)}, {}

        def step(self, action):
            return self.reset()[0], 0.0, False, False, {}

    pk = {"features_extractor_class": Gen3FeaturesExtractor,
          "features_extractor_kwargs": {"layout": layout, "mappings": mappings},
          "net_arch": [512, 512]}
    model = MaskablePPO(Gen3DualHeadMaskablePolicy, DummyVecEnv([_E]), policy_kwargs=pk,
                        verbose=0, device="cpu")
    model.num_timesteps = _FRESH_STEP
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "final_model"))
    save_model_snapshot(str(run_dir),
                        ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]}),
                        git_hash="test")
    return str(run_dir / "final_model.zip")


def _skip_reason(server: str = "rust") -> "str | None":
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
    if server == "node":
        # Only the OPT-OUT transport needs Node. The default one is the point of this switch:
        # a box with no `node` can still take an anchor read.
        from shutil import which

        if which(cfg.node) is None:
            return (f"no `{cfg.node}` on PATH — the --server node opt-out cannot be started. "
                    "The default --server rust transport needs none.")
    return None



@pytest.mark.parametrize("server", ["rust", "node"])
def test_two_real_games_against_metamon_smallrl(tmp_path: Path, server: str) -> None:
    """The whole tool, end to end, on the smallest sample that still exercises every seam — once
    per transport, because "it works" on one of them says nothing about the other."""
    reason = _skip_reason(server)
    if reason:
        pytest.skip(reason)

    model = _save_current_generation_checkpoint(tmp_path / "current_gen_run")
    out = tmp_path / "out"

    rc = anchors_main([
        "--model", model,
        "--opponent", "metamon:SmallRL",
        "--regime", "greedy",
        "--teamset", "away",
        "--games", str(N_GAMES),
        "--device", "cpu",
        "--out", str(out),
        # Generous, because the box normally carries a live training arm; a TIMEOUT is never a
        # semantic outcome, so a slow box must not turn into a failed assertion about the tool.
        "--first-game-timeout", "900",
        "--progress-timeout", "900",
        "--peer-ready-timeout", "900",
        # 18 characters is Showdown's ceiling and a 19th is a HANG, not an error. The transport
        # goes in the name so the two parametrizations cannot collide on a live login.
        "--username", f"Gen3AIit{server}",
        "--peer-username", f"MetaIt{server}",
        "--server", server,
    ])

    summary = json.loads((out / "summary.json").read_text())
    rows = [json.loads(line) for line in (out / "games.jsonl").read_text().splitlines()]

    # A named failure is a legitimate outcome of a contended box — but it must be NAMED, and the
    # games that did finish must still be here. Silence is the one thing this tool may not produce.
    if rc != 0:
        pytest.fail(
            f"anchor read returned {rc}; failure={summary.get('failure')}; "
            f"{len(rows)} of {N_GAMES} games completed. "
            f"peer log: {sorted(p.name for p in out.rglob('peer_*.log'))}")

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
        assert row["model_step"] == _FRESH_STEP
        assert row["turns"] > 0

    # Both sides drew from Metamon's own 20-team `competitive` set — the AWAY cell.
    assert summary["cell"]["teamset"] == "away"
    assert summary["team_source_asymmetry"] is False, (
        f"the two sides drew from differently sized team sources "
        f"({summary['cell']['our_team_count']} vs {summary['cell']['their_team_count']}) — the "
        "win rate would then mix skill with matchup")

    # THE TRANSPORT, on every row — the same rule the regime has, for the same reason.
    assert summary["cell"]["server_impl"] == server
    assert all(row["server_impl"] == server for row in rows)
    assert summary["cell"]["server_version"]

    # The server we started is gone, its log is named after it, and nothing else was touched.
    log = "ws_frontend.log" if server == "rust" else "showdown.log"
    assert (out / log).exists()
    assert not (out / ("showdown.log" if server == "rust" else "ws_frontend.log")).exists(), (
        "the other transport's log is here — this read started a server it was not asked for")
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
