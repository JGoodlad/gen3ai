"""The prober on the 2026-10-27 model-config moves, with REAL toy checkpoints (CPU).

v163 (`OBS_SEMANTICS_VERSION` 163, the HP-type belief / Foresight fix, `replay: warn`) and v164 (`--grow
trunk_ffn_dim=N`: every trunk round gains a `round.ffn_extra` width branch, no observation change, so no marker row).
`gap_policy_test` pins the table with stand-in verdicts and `model_test` with a stub loader; THESE load a toy through
the real `ProbeModel.load`: a plain v164, the same toy grown 256 -> 512 (the extra `ffn_extra` parameters), and the
mismatches between a record and its weights.
"""

import dataclasses
import json
import shutil
import zipfile

import numpy as np
import pytest

from agents.model import growth as G
from agents.model.model_version import MODEL_CONFIG_VERSION, OBS_SEMANTICS_VERSION
from main.prober import arch_status as A
from main.prober.model import ArchDriftError, ProbeModel

_USES = (A.USE_TRACE_REPLAY, A.USE_FRESH, A.USE_STORED_SCORE)


@pytest.fixture(scope="module")
def toys(tmp_path_factory):
    """`plain` (a toy after one AdamW step) and `grown` (the same, `--grow trunk_ffn_dim=512`), each a saved run dir."""
    from agents.model.growth_test import _parent
    from agents.model.snapshot import load_model_snapshot, recorded_model_version, save_model_snapshot
    from main.train.checkpoint_state import _validate_or_reset_optimizer_state
    from utils.torch_state_guard import torch_globals

    root = tmp_path_factory.mktemp("grown_toys")
    fields = {"trunk_ffn_dim": {"parent": 256, "child": 512}}
    with torch_globals(num_threads=1):
        model, obs, mask = _parent()
        plain = root / "plain"
        plain.mkdir()
        model.save(str(plain / "final_model"))
        version = recorded_model_version(model)
        save_model_snapshot(str(plain), version, git_hash="test")
        child = dataclasses.replace(version, trunk_ffn_dim=512)
        grown_model = load_model_snapshot(str(plain / "final_model.zip"), env=None, current_version=child,
                                          device="cpu", grow=fields)
        _validate_or_reset_optimizer_state(grown_model, str(plain / "final_model.zip"))
        G.apply_growth(grown_model, fields, rows=(obs, mask))
        grown = root / "grown"
        grown.mkdir()
        grown_model.save(str(grown / "final_model"))
        save_model_snapshot(str(grown), dataclasses.replace(child, growth={"fields": fields}), git_hash="test")
    assert version.trunk_ffn_dim == 256 and MODEL_CONFIG_VERSION >= 164
    return {"plain": plain, "grown": grown, "obs": obs[:4], "mask": mask[:4]}


def _restamp(src_dir, dst_dir, config_version):
    """A copy of a saved run whose RECORD says `config_version` (the weights are untouched)."""
    shutil.copytree(src_dir, dst_dir)
    cfg = json.loads((dst_dir / "model_config.json").read_text())
    cfg["config_version"] = config_version
    (dst_dir / "model_config.json").write_text(json.dumps(cfg))
    return str(dst_dir / "final_model.zip")


@pytest.mark.parametrize("run", ["plain", "grown"])
@pytest.mark.parametrize("use", _USES)
def test_a_v164_model_grown_or_not_loads_current_for_every_use_and_runs_every_forward(toys, run, use):
    """v164 adds no observation change (no marker row): a v164 record is CURRENT, `allow`, no badge; the grown child
    (extra `ffn_extra` parameters, a 512-wide FFN) loads strictly and answers the forward, the batch forwards, the
    saliency gradients and the attention capture exactly as its parent does."""
    z = str(toys[run] / "final_model.zip")
    pm = ProbeModel.load(z, "cpu", use=use)
    assert pm.arch_verdict.status == A.CURRENT and pm.obs_semantics["action"] == A.ALLOW
    assert pm.obs_semantics["badge"] is None and pm.obs_semantics["changes_since"] == []
    n_extra = sum(1 for k in pm._policy.state_dict() if "ffn_extra" in k)
    assert (n_extra > 0) == (run == "grown")
    obs, mask = toys["obs"], toys["mask"]
    probs, _logits = pm.action_dist(obs[0], mask[0])
    assert probs.shape == (11,) and np.isfinite(probs).all() and abs(probs.sum() - 1) < 1e-5
    assert pm.action_probs_batch(obs, mask).shape == (4, 11) and pm.values_batch(obs, mask).shape == (4,)
    assert np.abs(pm.logit_grad(obs[0], mask[0], int(np.argmax(mask[0])))).sum() > 0
    assert np.abs(pm.value_grad(obs[0], mask[0])).sum() > 0
    cap = pm.capture_battle(obs, mask)
    assert cap["attention"].shape[0] == 4 and cap["n_layers"] == 2
    assert any(r["name"] == "TeamTransformer" for r in pm.architecture())


def test_the_grown_child_is_its_parents_function_and_the_architecture_panel_names_the_width(toys):
    parent = ProbeModel.load(str(toys["plain"] / "final_model.zip"), "cpu", use=A.USE_TRACE_REPLAY)
    child = ProbeModel.load(str(toys["grown"] / "final_model.zip"), "cpu", use=A.USE_TRACE_REPLAY)
    obs, mask = toys["obs"], toys["mask"]
    assert np.array_equal(parent.action_probs_batch(obs, mask), child.action_probs_batch(obs, mask))
    assert np.array_equal(parent.values_batch(obs, mask), child.values_batch(obs, mask))
    role = {r["name"]: r["role"] for r in child.architecture()}["TeamTransformer"]
    assert "FFN 512 wide" in role
    assert "FFN" not in {r["name"]: r["role"] for r in parent.architecture()}["TeamTransformer"]


@pytest.mark.parametrize("config_version,badge_rows", [
    (162, [163]),             # trained at the v162 marker: only the Foresight fix is newer
    (161, [162, 163]),        # the current production runs (marker 151): both rows newer
])
def test_a_pre_163_record_replays_with_the_badge_naming_each_newer_row(toys, tmp_path, config_version, badge_rows):
    """The WARN across 162 -> 163: weights that fit, an observation whose meaning moved, `replay: warn` on every row,
    so a trace replay RUNS, badged with exactly the rows newer than the record's marker."""
    z = _restamp(toys["plain"], tmp_path / f"v{config_version}", config_version)
    pm = ProbeModel.load(z, "cpu", use=A.USE_TRACE_REPLAY)
    sem = pm.obs_semantics
    assert sem["action"] == A.WARN and sem["gap"] == A.GAP_SEMANTICS and sem["code_semantics"] == OBS_SEMANTICS_VERSION
    assert [c["version"] for c in sem["changes_since"]] == badge_rows and all(c["replay"] == A.WARN
                                                                              for c in sem["changes_since"])
    assert "may differ from what the model computed at play time" in sem["badge"]
    for v in badge_rows:
        assert f"v{v}:" in sem["badge"]
    assert pm.action_probs_batch(toys["obs"], toys["mask"]).shape == (4, 11)        # and it answers


def _swap_policy_weights(src_zip, donor_zip, dst_dir, record_dir):
    """`src_zip`'s zip with `donor_zip`'s weights: a record and a state_dict that disagree."""
    dst_dir.mkdir(parents=True)
    out = dst_dir / "m.zip"
    with zipfile.ZipFile(src_zip) as zs, zipfile.ZipFile(donor_zip) as zd, \
            zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
        for name in zs.namelist():
            zo.writestr(name, (zd if name == "policy.pth" else zs).read(name))
    shutil.copy(record_dir / "model_config.json", dst_dir / "model_config.json")
    return str(out)


@pytest.mark.parametrize("config,weights", [("plain", "grown"), ("grown", "plain")])
@pytest.mark.parametrize("use", _USES)
def test_weights_that_do_not_fit_their_architecture_refuse_every_use_with_a_true_sentence(toys, tmp_path, config,
                                                                                         weights, use):
    """Extra `ffn_extra` parameters under a 256-wide extractor (and the reverse): `state_dict`, refused for every use.
    The record here IS the code's (config v164), so the sentence must not say the run is `older than the code`."""
    z = _swap_policy_weights(toys[config] / "final_model.zip", toys[weights] / "final_model.zip",
                             tmp_path / "mm", toys[config])
    with pytest.raises(ArchDriftError) as ei:
        ProbeModel.load(z, "cpu", use=use)
    e = ei.value
    assert e.kind == A.KIND_STATE_DICT
    assert "older than the code" not in e.plain
    assert "the weights and the record disagree" in e.plain and f"config v{MODEL_CONFIG_VERSION}" in e.plain


def test_the_same_sentence_for_an_actually_older_record_still_says_older(toys):
    older = A.plain_sentence(A.KIND_STATE_DICT, config_version=MODEL_CONFIG_VERSION - 1,
                             arch_signature=A.ARCH_SIGNATURE)
    assert "older than the code" in older and "disagree" not in older
