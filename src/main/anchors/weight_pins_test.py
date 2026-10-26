"""FR-1 FINDING 3 — the Metamon weights are pinned: ONE hub revision, a sha256 per file, OFFLINE, refuse otherwise.

THE DEFECT. ``main.anchors`` loaded Metamon weights through ``huggingface_hub.hf_hub_download(repo_id, filename,
cache_dir)`` with no ``revision`` and online: a read could silently fetch newer upstream weights if the hub's ``main``
moved, changing the anchor under a regime id that claimed the same policy. Every test here fails if its half of the fix
is reverted (the sha check, the offline switch, the revision forcing, the fail-closed report read, the regime input).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from main.anchors import runner as runner_mod
from main.anchors.config import AnchorConfigError, load_config
from main.anchors.peer_scripts import weight_pins as wp
from main.anchors.peers import MetamonPeer
from main.anchors.peers_test import _metamon_plan, pin_report_fields
from main.anchors.results import REQUIRED_ROW_FIELDS
from utils.paths import src_path

REPO = "owner/models"
REV = "a" * 40
REL = "small-rl/ckpts/policy_weights/policy_epoch_40.pt"
BODY = b"the weights"
SHA = hashlib.sha256(BODY).hexdigest()


@pytest.fixture()
def cfg():
    return load_config()


def _cache_file(tmp_path: Path, body: bytes = BODY) -> Path:
    p = tmp_path / "blob"
    p.write_bytes(body)
    return p


class FakeHub:
    """``hf_hub_download`` stand-in that records how it was called and serves one local file."""

    def __init__(self, path: Path | None) -> None:
        self.path, self.calls = path, []

    def __call__(self, repo_id: str, filename: str, **kw: Any) -> str:
        self.calls.append((repo_id, filename, kw))
        if self.path is None:
            raise FileNotFoundError("cannot find the file in the local cache and offline mode is on")
        return str(self.path)


def _downloader(hub: FakeHub, weights: dict | None = None):
    loaded: list = []
    return wp.pinned_downloader(hub, repo=REPO, revision=REV, weights=weights or {REL: SHA}, loaded=loaded), loaded


# ------------------------------------------------------------------ a wrong sha refuses; a missing file refuses
def test_a_wrong_sha_refuses_and_the_right_sha_serves(tmp_path) -> None:
    hub = FakeHub(_cache_file(tmp_path, b"NEWER UPSTREAM WEIGHTS"))
    dl, loaded = _downloader(hub)
    with pytest.raises(wp.WeightPinError, match="hashes to"):
        dl(repo_id=REPO, filename=REL, cache_dir="/c")
    assert loaded == [], "a refused file is never reported as loaded"
    good, loaded = _downloader(FakeHub(_cache_file(tmp_path)))
    assert good(repo_id=REPO, filename=REL, cache_dir="/c").endswith("blob")
    assert loaded == [REL]


def test_a_file_changed_after_the_first_hash_is_still_a_new_path_only_if_it_is_a_new_path(tmp_path) -> None:
    """The hash is by REAL path: a symlink to a different blob is a different file and is hashed on its own."""
    a, b = _cache_file(tmp_path), tmp_path / "other"
    b.write_bytes(b"x")
    link = tmp_path / "snap"
    link.symlink_to(a)
    hub = FakeHub(link)
    dl, _ = _downloader(hub)
    dl(repo_id=REPO, filename=REL)
    link.unlink()
    link.symlink_to(b)
    with pytest.raises(wp.WeightPinError, match="hashes to"):
        dl(repo_id=REPO, filename=REL)


def test_a_missing_file_refuses_rather_than_downloading(tmp_path) -> None:
    dl, _ = _downloader(FakeHub(None))
    with pytest.raises(wp.WeightPinError, match="not in the local cache"):
        dl(repo_id=REPO, filename=REL)


def test_an_unpinned_file_or_repo_or_revision_refuses(tmp_path) -> None:
    dl, _ = _downloader(FakeHub(_cache_file(tmp_path)))
    with pytest.raises(wp.WeightPinError, match="not a pinned weight file"):
        dl(repo_id=REPO, filename="kakuna/ckpts/policy_weights/policy_epoch_34.pt")
    with pytest.raises(wp.WeightPinError, match="only 'owner/models' is pinned"):
        dl(repo_id="someone/else", filename=REL)
    with pytest.raises(wp.WeightPinError, match="the pin is"):
        dl(repo_id=REPO, filename=REL, revision="main")
    with pytest.raises(wp.WeightPinError, match="force_download"):
        dl(repo_id=REPO, filename=REL, force_download=True)


def test_the_hub_is_called_at_the_pinned_revision_from_the_local_cache_only(tmp_path) -> None:
    """The call Metamon makes names no revision; the replacement adds the pin and forbids the network."""
    hub = FakeHub(_cache_file(tmp_path))
    dl, _ = _downloader(hub)
    dl(REPO, REL, cache_dir="/c")                      # positional, exactly as huggingface_hub accepts it
    dl(repo_id=REPO, filename=REL, cache_dir="/c")
    for repo, filename, kw in hub.calls:
        assert (repo, filename) == (REPO, REL)
        assert kw["revision"] == REV and kw["local_files_only"] is True and kw["cache_dir"] == "/c"


def test_verify_all_hashes_every_pinned_file_before_play(tmp_path) -> None:
    other = "kakuna/ckpts/policy_weights/policy_epoch_34.pt"
    hub = FakeHub(_cache_file(tmp_path))
    dl, loaded = _downloader(hub, {REL: SHA, other: "0" * 64})
    with pytest.raises(wp.WeightPinError, match="hashes to"):
        wp.verify_all(dl, REPO, {REL: SHA, other: "0" * 64}, "/c")
    assert wp.verify_all(*_va(tmp_path)) == {REL: SHA}


def _va(tmp_path):
    dl, _ = _downloader(FakeHub(_cache_file(tmp_path)))
    return dl, REPO, {REL: SHA}, "/c"


@pytest.mark.parametrize("revision", ["main", "ac8e88e7", "A" * 40, "", None, "a" * 41])
def test_a_moving_or_short_revision_is_refused(revision) -> None:
    with pytest.raises(wp.WeightPinError, match="40-hex"):
        wp.check_pins(revision, {REL: SHA})


@pytest.mark.parametrize("weights", [{}, None, {REL: "abc"}, {REL: "G" * 64}, {"/abs": SHA}, {"../x": SHA}])
def test_a_malformed_weight_map_is_refused(weights) -> None:
    with pytest.raises(wp.WeightPinError):
        wp.check_pins(REV, weights)


# ------------------------------------------------------------------ OFFLINE is enforced
def test_force_offline_overrides_an_inherited_online_setting() -> None:
    env = {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
    wp.force_offline(env)
    assert env == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_an_online_hub_library_is_refused() -> None:
    class C:
        HF_HUB_OFFLINE = False

    with pytest.raises(wp.WeightPinError, match="not in offline mode"):
        wp.assert_offline(C)
    C.HF_HUB_OFFLINE = True
    wp.assert_offline(C)


def test_the_peer_plan_runs_offline_whatever_the_caller_exported(cfg, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    env = _metamon_plan(cfg, tmp_path).env
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"


def test_the_peer_script_forces_offline_at_import_before_the_hub_library_can_load() -> None:
    """Run in a clean interpreter with the caller saying ONLINE: after the script's import the switch is on, and the
    hub library has not been imported yet (it reads the switch at import)."""
    script = src_path("main", "anchors", "peer_scripts", "metamon_side.py")
    code = ("import importlib.util, json, os, sys\n"
            f"spec = importlib.util.spec_from_file_location('ms', {str(script)!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "print(json.dumps({'off': os.environ['HF_HUB_OFFLINE'], 'hub': 'huggingface_hub' in sys.modules}))")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120,
                         env={"HF_HUB_OFFLINE": "0", "PATH": os.environ.get("PATH", "")})
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"off": "1", "hub": False}


def test_the_peer_script_installs_the_pins_before_it_builds_the_model() -> None:
    """The order is the guard: pins verified (hashed) BEFORE ``get_pretrained_model``, and the report carries them."""
    src = src_path("main", "anchors", "peer_scripts", "metamon_side.py").read_text()
    assert src.index("pin_record = install_weight_pins(args)") < src.index("model = get_pretrained_model(args.agent)")
    assert "**pin_record" in src and 'ap.add_argument("--weights-json", required=True' in src


# ------------------------------------------------------------------ the plan carries the pin; the config refuses an unpinned agent
def test_the_plan_hands_the_peer_the_pinned_revision_and_every_sha(cfg, tmp_path) -> None:
    plan = _metamon_plan(cfg, tmp_path)
    argv = plan.argv
    pin = cfg.metamon.pin("SmallRL")
    assert argv[argv.index("--hf-repo") + 1] == cfg.metamon.hf_repo == "jakegrigsby/metamon"
    assert argv[argv.index("--hf-revision") + 1] == pin.revision == cfg.metamon.hf_revision
    assert json.loads(argv[argv.index("--weights-json") + 1]) == pin.weights
    assert plan.extra["weight_pin"] == pin


def test_every_committed_metamon_agent_is_pinned_to_a_full_revision_and_a_sha_per_file(cfg) -> None:
    for name in cfg.metamon.agents:
        pin = cfg.metamon.pin(name)
        wp.check_pins(pin.revision, pin.weights, name)
        assert len(pin.digest) == 64
    # an ensemble pins every member checkpoint (Kakuna 28 / 30 / 34 and Alakazam 8)
    assert len(cfg.metamon.pin("PastaMittens").weights) == 4
    # an alias pins ITS checkpoint, not its policy's default
    assert [Path(p).name for p in cfg.metamon.pin("Kakuna28").weights] == ["policy_epoch_28.pt"]


def _with_agents(cfg, agents, **over):
    return dataclasses.replace(cfg.metamon, agents=agents, **over)


def test_an_unpinned_agent_is_refused_by_require_and_by_plan(cfg, tmp_path) -> None:
    meta = _with_agents(cfg, {"SmallRL": {"checkpoint": 40}})
    with pytest.raises(AnchorConfigError, match="weights"):
        meta.require("SmallRL")
    with pytest.raises(AnchorConfigError):
        MetamonPeer().plan(cfg=meta, agent="SmallRL", regime="greedy", teamset="away", battle_format="gen3ou",
                           server_uri="ws://localhost:9501/showdown/websocket", username="a", opponent_username="b",
                           role="acceptor", n_games=1, team_seed=1, out_dir=tmp_path)


def test_a_moving_revision_or_a_wrong_checkpoint_file_is_refused_in_the_config(cfg) -> None:
    entry = dict(cfg.metamon.agents["SmallRL"])
    with pytest.raises(AnchorConfigError, match="40-hex"):
        _with_agents(cfg, {"SmallRL": entry}, hf_revision="main").pin("SmallRL")
    with pytest.raises(AnchorConfigError, match="hf_repo"):
        _with_agents(cfg, {"SmallRL": entry}, hf_repo="").pin("SmallRL")
    wrong = {**entry, "weights": {"small-rl/ckpts/policy_weights/policy_epoch_39.pt": SHA}}
    with pytest.raises(AnchorConfigError, match="policy_epoch_40"):
        _with_agents(cfg, {"SmallRL": wrong}).pin("SmallRL")


# ------------------------------------------------------------------ the report read FAILS CLOSED
def _read(cfg, tmp_path, **override) -> dict:
    plan = _metamon_plan(cfg, tmp_path)
    blob = {"sample_kwarg_values": [False], "argmax_match_rate": 1.0, "n_decisions": 9,
            "regime_verified_decisions": True, "peer_error_free": True, **pin_report_fields(plan)}
    blob.update(override)
    blob = {k: v for k, v in blob.items() if v is not ...}
    plan.report_path.write_text(json.dumps(blob))
    return MetamonPeer.read_report(plan)


def test_a_peer_that_verified_the_pin_reads_verified_and_reports_what_it_verified(cfg, tmp_path) -> None:
    r = _read(cfg, tmp_path)
    pin = cfg.metamon.pin("SmallRL")
    assert r["regime_verified_decisions"] is True and "weights_pin_mismatch" not in r
    assert r["hf_revision"] == pin.revision and r["weights_sha256"] == pin.digest


@pytest.mark.parametrize("override, why", [
    ({"hf_revision": "b" * 40}, "revision"),
    ({"weights_verified": {REL: "0" * 64}}, "verified weights"),
    ({"weights_verified": {}}, "verified weights"),
    ({"hf_hub_offline": False}, "offline"),
    ({"hf_hub_offline": ...}, "offline"),                      # a peer script that predates the pins
    ({"hf_revision": ..., "weights_verified": ...}, "revision"),
    ({"weights_loaded": [REL, "kakuna/ckpts/policy_weights/policy_epoch_34.pt"]}, "unpinned"),
])
def test_a_report_that_does_not_prove_the_pin_voids_the_regime(cfg, tmp_path, override, why) -> None:
    r = _read(cfg, tmp_path, **override)
    assert r["regime_verified_decisions"] is False, "an unverified weight set must never count as a measurement"
    assert why in r["weights_pin_mismatch"]


# ------------------------------------------------------------------ the regime includes the revision (stamp + id)
def test_the_cell_stamps_the_verified_revision_and_weights_on_every_row(cfg, tmp_path) -> None:
    r = _read(cfg, tmp_path)
    assert "hf_revision" in REQUIRED_ROW_FIELDS and "weights_sha256" in REQUIRED_ROW_FIELDS
    from main.anchors.cli import build_parser, build_plan

    plan = build_plan(build_parser().parse_args(["--model", "", "--opponent", "metamon:SmallRL", "--dry-run"]), cfg)
    report = {**r, "version": "SmallRL@ckpt40", "our_hf_revision": "e" * 40, "our_weights_sha256": "f" * 64}
    stamp = runner_mod.cell_spec(plan, report, 20).stamp()
    assert stamp["hf_revision"] == r["hf_revision"] and stamp["weights_sha256"] == r["weights_sha256"]
    assert stamp["our_hf_revision"] == "e" * 40 and stamp["our_weights_sha256"] == "f" * 64
    unstamped = runner_mod.cell_spec(plan, {}, 20).stamp()
    assert unstamped["hf_revision"] is None and unstamped["weights_sha256"] is None, "no weights, no claim"


def test_a_repinned_revision_changes_the_regime_id() -> None:
    from main.reads import assemble as AS

    cell = {"opponent_version": "Kakuna@ckpt34", "opponent_agent": "Kakuna", "opponent": "metamon:Kakuna",
            "our_regime": "greedy", "their_regime": "greedy"}
    base = AS.anchor_regime_id({"cell": {**cell, "hf_revision": AS.LEGACY_HF_REVISION,
                                         "weights_sha256": AS.LEGACY_WEIGHTS["Kakuna"]}})
    moved = AS.anchor_regime_id({"cell": {**cell, "hf_revision": "d" * 40,
                                          "weights_sha256": AS.LEGACY_WEIGHTS["Kakuna"]}})
    assert base != moved


# ------------------------------------------------------------------ the committed pins ARE the bytes in the cache
@pytest.mark.integration
def test_the_committed_pins_match_the_files_in_the_metamon_cache() -> None:
    """Hash every pinned file at the pinned revision in the real cache (offline, no network): the pins are not a guess."""
    meta = load_config().metamon
    root = meta.cache_dir / "pretrained_models" / "models--jakegrigsby--metamon" / "snapshots" / meta.hf_revision
    if not root.is_dir():
        pytest.skip(f"no Metamon cache snapshot at {root}")
    for name in meta.agents:
        for rel, sha in meta.pin(name).weights.items():
            assert wp.sha256_file(str(root / rel)) == sha, (name, rel)
