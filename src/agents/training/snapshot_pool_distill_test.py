"""Unit tests for SnapshotPool's distilled-variant support (the disk hooks the reconcile manager
uses + the env-side distilled adapter loading). Temp dir + a real saved DistilledStudent; no teacher
needed for the file/loading logic."""
import json

import torch

from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.model_version import ModelVersion
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.snapshot_pool import SnapshotPool, SnapshotEntry
from agents.training.distill.student import DistilledStudent, save_distilled


def _pool(tmp_path):
    enc = Gen3ObservationEncoder(load_mappings())
    ek = enc.get_features_extractor_kwargs()
    layout = ek["layout"]
    ver = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"features_extractor_class": Gen3FeaturesExtractor,
                 "features_extractor_kwargs": ek, "net_arch": NET_ARCH})
    pool = SnapshotPool(pool_dir=tmp_path / "snapshots", current_version=ver, device="cpu")
    pool.set_distill_layout(layout)
    return pool, layout


def _write_distilled(pool, layout, step, passed=True):
    pool.distilled_dir.mkdir(parents=True, exist_ok=True)
    s = DistilledStudent(layout, enc_hidden=128, head_hidden=256)
    save_distilled(s, str(pool.distilled_pt(step)))
    pool.distilled_manifest(step).write_text(json.dumps(
        {"step": step, "passed": passed, "top1": 0.86, "h2h": 0.44, "speedup": 4.7}))


def test_gate_passed_and_artifact_scans(tmp_path):
    pool, layout = _pool(tmp_path)
    assert pool.gate_passed_steps() == set() and pool.distilled_artifact_steps() == set()
    _write_distilled(pool, layout, 1000, passed=True)
    _write_distilled(pool, layout, 2000, passed=False)   # distilled but failed gate
    assert pool.gate_passed_steps() == {1000}            # only passed counts as ready
    assert pool.distilled_artifact_steps() == {1000, 2000}  # both have a .pt


def test_remove_distilled(tmp_path):
    pool, layout = _pool(tmp_path)
    _write_distilled(pool, layout, 1000)
    assert pool.distilled_pt(1000).exists()
    pool.remove_distilled(1000)
    assert not pool.distilled_pt(1000).exists() and not pool.distilled_manifest(1000).exists()
    assert pool.gate_passed_steps() == set()


def test_load_distilled_opponent_is_rlplayer_shaped(tmp_path):
    pool, layout = _pool(tmp_path)
    _write_distilled(pool, layout, 1000)
    entry = SnapshotEntry(path=tmp_path / "snapshot_000000001000.zip", step=1000)
    model = pool.load_distilled_opponent(entry)
    assert hasattr(model, "device") and hasattr(model.policy, "get_distribution")
    obs = {"observation": torch.zeros(2, layout["total_dim"]),
           "action_mask": torch.ones(2, 11, dtype=torch.int8)}
    logits = model.policy.get_distribution(obs).distribution.logits
    assert logits.shape == (2, 11)
    # LRU cache: a second load returns the same object
    assert pool.load_distilled_opponent(entry) is model


def test_sample_from_restricts_to_steps(tmp_path):
    pool, _ = _pool(tmp_path)
    pool._entries = [SnapshotEntry(path=tmp_path / f"s{s}.zip", step=s) for s in (1000, 2000, 3000)]
    assert pool.sample_from({2000}).step == 2000
    assert pool.sample_from({1000, 3000}).step in {1000, 3000}
    assert pool.sample_from({9999}) is None     # none in pool
