"""A CPU-intended checkpoint load never unpickles the ride-along optimizers and never creates a CUDA
context (`gen3_cpu_load_no_cuda_v1`, 2026-10-06).

THE BUG: `_excluded_save_params` did not list `_ridealong_opt`, `_ridealong_opt_owner`, `_ridealong_vopts`,
`_ridealong_vopts_owner`, so every checkpoint since the ride-along heads pickled them (~100 MB of CUDA Adam state
and head copies per X5 A/B checkpoint) into its sb3 ``data``. sb3's ``json_to_data`` cloudpickle-loads every
serialized attribute onto the device it was SAVED from, whatever ``device`` the load asked for, so the prober,
h2h, belief_roles, policy_spectrum and the meters created a CUDA context (330 MiB measured) on a "CPU" load. No load
ever used them: every ``_setup_model`` re-acquires the optimizers (or sets them to None for an inference load).

No GPU is touched here: a CUDA tensor's unpickle is stood in for by a TRIPWIRE object whose unpickle counts
itself (and, for the guard, flips a stand-in ``torch.cuda.is_initialized``). Each test fails on revert:
  * the save no longer writes the attributes (revert `hparams._excluded_save_params`);
  * no loader unpickles them from an OLD checkpoint that carries them (revert `strict_load.never_unpickled`
    in `StrictCheckpointLoad.load` or in `snapshot._patch_historical_floor`);
  * a CPU load that initialises CUDA anyway is a typed `CudaContextOnCpuLoad` (revert the guard).
"""
from __future__ import annotations

import base64
import json
import zipfile
from pathlib import Path
from typing import Any, Dict

import cloudpickle
import pytest

from agents.training.instrumented_ppo import strict_load as SL

_UNPICKLED = {"n": 0}
_FAKE_CUDA = {"on": False}


def _tripwire_fired(kind: str) -> str:
    """Runs only when a tripwire is UNPICKLED (what a CUDA tensor's unpickle is in the real checkpoint)."""
    _UNPICKLED["n"] += 1
    if kind == "cuda":
        _FAKE_CUDA["on"] = True
    return "unpickled"


class _Tripwire:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def __reduce__(self) -> Any:
        return (_tripwire_fired, (self.kind,))


def _plant(src: Path, dst: Path, entries: Dict[str, Any]) -> Path:
    """A copy of ``src`` whose sb3 ``data`` member also carries ``entries`` serialized as sb3 serializes an
    attribute it cannot JSON-encode (a cloudpickle, base64)."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            raw = zin.read(item.filename)
            if item.filename == "data":
                data = json.loads(raw)
                for k, v in entries.items():
                    data[k] = {":type:": "<class 'object'>",
                               ":serialized:": base64.b64encode(cloudpickle.dumps(v)).decode()}
                raw = json.dumps(data).encode()
            zout.writestr(item, raw)
    return dst


@pytest.fixture(scope="module")
def world(tmp_path_factory: Any) -> Dict[str, Any]:
    from agents.training.opponent_inference_load_test import RIDEALONG_ON, _build, _save

    root = tmp_path_factory.mktemp("cpuload")
    model, version, _ = _build(RIDEALONG_ON, 3)
    assert model._ridealong_opt is not None                       # the precondition: a trainee with heads
    zip_path = _save(model, version, root / "run")
    legacy = _plant(zip_path, root / "run" / "legacy.zip", {k: _Tripwire("cuda") for k in SL.NEVER_UNPICKLED})
    return {"model": model, "version": version, "zip": zip_path, "legacy": legacy, "root": root}


def test_a_save_writes_no_ride_along_optimizer_state(world: Dict[str, Any]) -> None:
    data = json.loads(zipfile.ZipFile(world["zip"]).read("data"))
    assert not set(SL.NEVER_UNPICKLED) & set(data), sorted(set(SL.NEVER_UNPICKLED) & set(data))


def test_no_loader_unpickles_an_old_checkpoints_training_state(world: Dict[str, Any]) -> None:
    from agents.model.snapshot import (historical_load_kwargs, load_checkpoint_strict, load_foreign_opponent,
                                       load_model_snapshot, load_opponent_snapshot)

    z, v = str(world["legacy"]), world["version"]
    _UNPICKLED["n"] = 0
    co = historical_load_kwargs(z).get("custom_objects")
    loads = {"strict": load_checkpoint_strict(z, device="cpu", custom_objects=co),
             "opponent": load_opponent_snapshot(z, v),
             "foreign": load_foreign_opponent(z, v)[0],
             "trainee": load_model_snapshot(z, env=None, current_version=v, device="cpu")}
    assert _UNPICKLED["n"] == 0, f"{_UNPICKLED['n']} pickled training-state object(s) were unpickled by a load"
    # the trainee's resume semantics are unchanged: its ride-along optimizers are acquired fresh, for its own heads
    t = loads["trainee"]
    assert t._ridealong_opt is not None and t._ridealong_opt_owner is t.policy.ridealong
    assert t._ridealong_vopts and t._ridealong_vopts_owner is t.policy.ridealong


def test_a_cpu_load_that_creates_a_cuda_context_is_refused(world: Dict[str, Any], tmp_path: Path,
                                                           monkeypatch: Any) -> None:
    """Any OTHER pickled CUDA state a future save adds: the load raises rather than leaving a context behind."""
    import torch

    from agents.model.snapshot import load_checkpoint_strict

    z = _plant(world["zip"], tmp_path / "future.zip", {"_future_cuda_state": _Tripwire("cuda")})
    _FAKE_CUDA["on"] = False
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: _FAKE_CUDA["on"])
    with pytest.raises(SL.CudaContextOnCpuLoad, match="created a CUDA context"):
        load_checkpoint_strict(str(z), device="cpu")
    # the clean checkpoint loads under the same stand-in
    _FAKE_CUDA["on"] = False
    assert load_checkpoint_strict(str(world["zip"]), device="cpu") is not None
