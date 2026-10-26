"""The weight pins belong to the OPPONENT: a config that carries none resolves them from the canonical registry.

REGRESSION (a260e9c3). ``main.x35.kakuna`` passes each panel's sha256-pinned, committed anchors config through
``$GEN3AI_ANCHORS_CONFIG``; those configs predate the pins, so every Kakuna unit exited rc 3 with
``opponents.metamon.hf_repo is not set`` and played nothing. The panel files cannot be edited (their sha256 is pinned in
``PanelCell`` and in regime ids), so ``load_config`` fills the pins from ``designs/ops/anchors.json`` by agent name.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from main.anchors.config import AnchorConfigError, load_config
from main.anchors.peers import MetamonPeer
from main.reads import assemble as AS
from utils.paths import repo_path

REGISTRY = repo_path("designs", "ops", "anchors.json")


def committed_anchors_configs() -> list:
    """EVERY committed anchors config under ``designs/`` (the registry and the panels' sha-pinned copies)."""
    found = [p for p in repo_path("designs").rglob("anchors*.json") if "opponents" in json.loads(p.read_text())]
    assert len(found) >= 15, f"the scan found only {len(found)} configs; the glob is broken: {found}"
    return sorted(found)


@pytest.mark.parametrize("path", committed_anchors_configs(), ids=lambda p: p.name)
def test_every_committed_config_resolves_its_agents_pins_from_the_registry(path: Path) -> None:
    """Fails on revert: without the registry fallback a panel config has no ``hf_repo`` and every agent is refused."""
    reg, cfg = load_config(REGISTRY).metamon, load_config(path).metamon
    assert cfg.agents, path
    for name in cfg.agents:
        pin = cfg.pin(name)                       # raises AnchorConfigError if not pinned
        assert pin == reg.pin(name), f"{path.name}:{name} does not carry the registry's pin"
        assert pin.digest == AS.LEGACY_WEIGHTS[name], "same bytes -> the banked regime id (LEGACY_WEIGHTS rule)"


def _panel_config() -> Path:
    from main.x35 import kakuna as KA

    return KA.PANEL_CELLS["heldout72_hpfix"].path


def test_the_committed_panel_config_carries_no_pins_and_stays_byte_identical() -> None:
    """The premise of the fallback, and the guard that nobody 'fixes' it by editing the sha-pinned file."""
    from main.x35 import kakuna as KA

    path = _panel_config()
    mm = json.loads(path.read_text())["opponents"]["metamon"]
    assert "hf_repo" not in mm and "hf_revision" not in mm and not any("weights" in a for a in mm["agents"].values())
    assert KA._sha(path) == KA.PANEL_CELLS["heldout72_hpfix"].sha256


def _write(tmp_path: Path, agents: dict, **meta) -> Path:
    blob = json.loads(REGISTRY.read_text())
    blob["opponents"]["metamon"]["agents"] = agents
    for k in ("hf_repo", "hf_revision"):
        blob["opponents"]["metamon"].pop(k, None)
    blob["opponents"]["metamon"].update(meta)
    p = tmp_path / "panel.json"
    p.write_text(json.dumps(blob))
    return p


def test_an_agent_is_borrowed_a_pin_only_for_the_same_policy_and_checkpoint(tmp_path) -> None:
    reg = load_config(REGISTRY).metamon
    cfg = load_config(_write(tmp_path, {"Kakuna": {"checkpoint": 34}, "Kakuna28": {"policy": "Kakuna", "checkpoint": 28},
                                        "Elsewhere": {"checkpoint": 1}, "SmallRL": {"checkpoint": 39}})).metamon
    assert cfg.pin("Kakuna") == reg.pin("Kakuna") and cfg.pin("Kakuna28") == reg.pin("Kakuna28")
    with pytest.raises(AnchorConfigError, match="not pinned there"):
        cfg.pin("Elsewhere")                                     # not in the registry
    with pytest.raises(AnchorConfigError):
        cfg.pin("SmallRL")                                       # the registry's SmallRL is checkpoint 40


def test_a_pin_the_config_carries_itself_wins_over_the_registry(tmp_path) -> None:
    mine = {"kakuna/ckpts/policy_weights/policy_epoch_34.pt": "1" * 64}
    cfg = load_config(_write(tmp_path, {"Kakuna": {"checkpoint": 34, "weights": mine}},
                             hf_repo="o/r", hf_revision="c" * 40)).metamon
    pin = cfg.pin("Kakuna")
    assert (pin.repo, pin.revision, pin.weights) == ("o/r", "c" * 40, mine)


def test_the_resolved_peer_command_for_a_panel_config_carries_the_registry_pin(tmp_path) -> None:
    cfg = load_config(_panel_config())
    plan = MetamonPeer().plan(cfg=cfg.metamon, agent="Kakuna", regime="greedy", teamset="away", battle_format="gen3ou",
                              server_uri="ws://localhost:9597/showdown/websocket", username="a", opponent_username="b",
                              role="acceptor", n_games=1, team_seed=1, out_dir=tmp_path)
    want = load_config(REGISTRY).metamon.pin("Kakuna")
    argv = plan.argv
    assert argv[argv.index("--hf-revision") + 1] == want.revision
    assert json.loads(argv[argv.index("--weights-json") + 1]) == want.weights


@pytest.mark.integration
def test_a_real_kakuna_unit_plan_on_a_panel_config_gets_past_config_load() -> None:
    """``main.anchors`` as ``main.x35.kakuna`` launches it (the panel config via ``$GEN3AI_ANCHORS_CONFIG``), dry-run."""
    from main.x35 import kakuna as KA

    for panel in ("heldout72_hpfix", "forum72", "p72"):
        cell = KA.PANEL_CELLS[panel]
        if not cell.path.is_file():
            continue
        out = subprocess.run(
            [sys.executable, "-m", "main.anchors", "--model", "", "--opponent", "metamon:Kakuna", "--regime", "greedy",
             "--teamset", cell.teamset, "--games", "2", "--dry-run"],
            env=KA.unit_env(config=cell.path), capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, (panel, out.stderr[-1500:])
        assert "--weights-json" in out.stdout and "hf_repo is not set" not in out.stdout + out.stderr


@pytest.mark.integration
@pytest.mark.slow  # ship gate: starts the real Metamon peer (torch + metamon import), ~15 s
def test_the_real_peer_started_from_a_panel_config_verifies_its_weights() -> None:
    """The peer process itself, on a CLOSED local port (nothing listens, nothing is bound): it must get through the
    pinned-weights step and log it before it fails to connect. Stopped by its own PID."""
    from main.anchors.windowed_attention_integration_test import _metamon_or_skip
    import tempfile

    _metamon_or_skip()
    cfg = load_config(_panel_config())
    with tempfile.TemporaryDirectory() as tmp:
        plan = MetamonPeer().plan(cfg=cfg.metamon, agent="SmallRL", regime="greedy", teamset="away",
                                  battle_format="gen3ou", server_uri="ws://localhost:9597/showdown/websocket",
                                  username="PinSmokeA", opponent_username="PinSmokeB", role="acceptor", n_games=1,
                                  team_seed=1, out_dir=Path(tmp))
        log = Path(tmp) / "peer.log"
        with open(log, "w") as fh:
            proc = subprocess.Popen(plan.argv, cwd=plan.cwd, env=plan.env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            deadline = time.time() + 240
            while time.time() < deadline and proc.poll() is None and "weights pinned" not in log.read_text():
                time.sleep(2)
            text = log.read_text()
        finally:
            proc.terminate()
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()
        assert "weights pinned" in text and "sha256-verified" in text, text[-2500:]
        assert "REFUSING" not in text and "hf_repo is not set" not in text
