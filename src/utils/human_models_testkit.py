"""A SYNTHETIC human-trained model for tests (``utils.human_models``): a tiny state dict in a checkpoint-shaped zip
under a temp ``$GEN3AI_ARCHIVE_DIR/human_models/<id>/``, and its stub in a temp registry. Never a real clone."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

TEST_HUMAN_MODEL = "hclone_t"
TEST_ROWSET = "hr_t_v1.r1"


def tiny_state(seed: int = 0) -> Dict[str, Any]:
    """Four tensors, the shape of a two-layer network's state dict."""
    import torch as th

    g = th.Generator().manual_seed(seed)
    return {"a.weight": th.randn(3, 4, generator=g), "a.bias": th.randn(3, generator=g),
            "b.weight": th.randn(2, 3, generator=g), "b.bias": th.randn(2, generator=g)}


def write_human_model(d: Path, model_id: str, state: Dict[str, Any], trained: Sequence[str], *, smoke: bool = False,
                      trained_sha: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Path:
    import torch as th

    from utils import human_models as HM

    d.mkdir(parents=True)
    buf = io.BytesIO()
    th.save(state, buf)
    with zipfile.ZipFile(d / f"{model_id}.zip", "w") as z:
        z.writestr("policy.pth", buf.getvalue())
    (d / HM.CONFIG_NAME).write_text(json.dumps(config or {}))
    (d / HM.MANIFEST_NAME).write_text(json.dumps({
        "schema": HM.SCHEMA, HM.HUMAN_DATA_KEY: True, "id": model_id, "rowset": TEST_ROWSET,
        "rows_manifest_sha256": "r" * 64, "zip": f"{model_id}.zip", "seed": 1, "smoke": smoke,
        "recipe": {"smoke": smoke}, "held_out": {"report": {"nll": 1.0, "top1": 0.5}},
        "stop": {"reason": "CONVERGED", "converged": True}, "trained_keys": sorted(trained),
        "trained_keys_sha256": trained_sha or HM.state_digest(state, sorted(trained))}, indent=1))
    return d


def install_human_model(monkeypatch, tmp_path: Path, state: Optional[Dict[str, Any]] = None,
                        model_id: str = TEST_HUMAN_MODEL, *, trained: Sequence[str] = ("a.weight", "a.bias"),
                        smoke: bool = False, status: Optional[str] = None, register: bool = True,
                        trained_sha: Optional[str] = None) -> Path:
    """Write the model under a temp archive, point ``utils.human_models`` at a temp registry, and (``register``)
    write its stub. Returns the model's directory."""
    from utils import human_models as HM

    archive = tmp_path / "archive"
    monkeypatch.setenv("GEN3AI_ARCHIVE_DIR", str(archive))
    reg = tmp_path / "human_replay_snapshots.json"
    if not reg.exists():
        reg.write_text(json.dumps({"schema": "gen3_human_replay_snapshots_v2", "models": {}}))
    monkeypatch.setattr(HM, "registry_path", lambda: reg)
    d = write_human_model(archive / "human_models" / model_id, model_id, tiny_state() if state is None else state,
                          trained, smoke=smoke, trained_sha=trained_sha)
    if register:
        HM.register(d, status or (HM.SMOKE_STATUS if smoke else "measurement"))
    return d
