"""Integration test of the worker->pool->manager->env contract (the disk handoff), without the
bridge: a fake ``run_distill`` writes the SAME manifest+`.pt` the real worker writes, then the
pool's real disk hooks feed the manager's reconcile, and the env loads the distilled adapter."""
import json

import torch

from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.model_version import ModelVersion
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.snapshot_pool import SnapshotPool, SnapshotEntry
from agents.training.distill.student import DistilledStudent, save_distilled
from agents.training.distill.manager import DistilledOpponentManager


def _pool(tmp_path):
    enc = Gen3ObservationEncoder(load_mappings())
    ek = enc.get_features_extractor_kwargs(); layout = ek["layout"]
    ver = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"features_extractor_class": Gen3FeaturesExtractor,
                 "features_extractor_kwargs": ek, "net_arch": NET_ARCH})
    pool = SnapshotPool(pool_dir=tmp_path / "snapshots", current_version=ver, device="cpu")
    pool.set_distill_layout(layout)
    return pool, layout


def test_pool_manager_env_contract(tmp_path):
    pool, layout = _pool(tmp_path)
    pool._entries = [SnapshotEntry(path=tmp_path / f"s{s}.zip", step=s) for s in (1000, 2000)]
    spawned_cfg = {}

    def run_distill(step, cfg):           # stands in for the subprocess
        spawned_cfg[step] = cfg
        return step                       # handle

    def poll(step):                       # simulate the worker FINISHING + writing its outputs
        pool.distilled_dir.mkdir(parents=True, exist_ok=True)
        save_distilled(DistilledStudent(layout, **spawned_cfg[step]), str(pool.distilled_pt(step)))
        pool.distilled_manifest(step).write_text(json.dumps(
            {"step": step, "passed": True, "speedup": 4.7, "top1": 0.86, "h2h": 0.44}))
        return {"passed": True, "speedup": 4.7}

    mgr = DistilledOpponentManager(
        ready_steps_fn=pool.gate_passed_steps, list_artifacts_fn=pool.distilled_artifact_steps,
        run_distill_fn=run_distill, poll_fn=poll, remove_fn=pool.remove_distilled, min_pool=1)

    active = [1000, 2000]
    r = mgr.reconcile(active)             # backfill: spawn both, not yet ready
    assert set(r.spawned) == {1000, 2000} and not r.all_distilled
    r = mgr.reconcile(active)             # poll finishes both -> on disk -> ready
    assert r.all_distilled and r.frac_distilled == 1.0
    assert pool.gate_passed_steps() == {1000, 2000}

    # the env can now load either as a drop-in opponent
    entry = next(e for e in pool._entries if e.step == 1000)
    model = pool.load_distilled_opponent(entry)
    obs = {"observation": torch.zeros(2, layout["total_dim"]),
           "action_mask": torch.ones(2, 11, dtype=torch.int8)}
    assert model.policy.get_distribution(obs).distribution.logits.shape == (2, 11)

    # eviction: snapshot 1000 slides out -> reconcile removes its artifact
    mgr.reconcile([2000])
    assert not pool.distilled_pt(1000).exists() and pool.gate_passed_steps() == {2000}
