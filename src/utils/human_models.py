"""REGISTERED HUMAN-TRAINED MODELS (``gen3_human_model_v1``): the checkpoints ``--human-data init:<model>@<dose>`` and
``opp:<model>@<share>`` may name, kept OUTSIDE git and outside ``models/``.

Design: ``designs/endstate/human_data_architecture.md`` §5 (snapshot → rowset → model) and §6 (the switch). A model
trained on human replay rows is human data one step removed, so it follows the human corpora's rules
(``utils.team_corpora.human``):

* the FILES live in ``utils.paths.human_models_dir() / <id>/``: the checkpoint zip, its ``model_config.json`` and
  ``metadata.json`` (the project's own save path), and ``human_model.json``, the manifest that says
  ``"human_data": true`` and names the rowset, the rows manifest, the recipe, the held-out read and the parameters
  the training moved;
* the STUB is committed: the ``models`` block of ``designs/research_state/human_replay_snapshots.json`` holds the id,
  the rowset and the sha256 of the manifest, the zip and the config. ``python -m main.human_data.clone register``
  writes it after the training.

:func:`load` verifies the directory against its stub and raises :class:`HumanModelError` on a miss or a mismatch; the
trainer turns it into ``FATAL_CONFIG``. **One entry point per model:** :func:`refuse_human_model` is the throwing
guard every OTHER way of handing a checkpoint to a run calls (``--stable-opponents``, ``--exploiter``, ``--model``):
a path that is a registered human model's, or lies under the human-model store, is refused there, so the only way in
is the switch (which records the clause in the run's lineage).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

SCHEMA = "gen3_human_model_v1"
MANIFEST_NAME = "human_model.json"
CONFIG_NAME = "model_config.json"
HUMAN_DATA_KEY = "human_data"
REGISTRY_REL = ("designs", "research_state", "human_replay_snapshots.json")
#: A registry status the trainer honours only under ``--debug``: a clone trained under ``--smoke`` overrides.
SMOKE_STATUS = "smoke"
STATUSES = ("measurement", "reference", "retired", SMOKE_STATUS)
#: The manifest fields a load must find equal to the committed stub's.
IDENTITY_FIELDS = ("rowset", "manifest_sha256", "checkpoint_sha256", "config_sha256")


class HumanModelError(ValueError):
    """A human-trained model that is unregistered, changed on disk, or handed to a run outside the switch."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def registry_path() -> Path:
    from utils.paths import repo_path

    return Path(repo_path(*REGISTRY_REL))


def models_dir() -> Path:
    from utils.paths import human_models_dir

    return human_models_dir()


def registry() -> Dict[str, Dict[str, Any]]:
    """The committed stubs, by id (``{}`` when none is registered)."""
    return dict(json.loads(registry_path().read_text()).get("models") or {})


def stub(model_id: str) -> Dict[str, Any]:
    stubs = registry()
    if model_id not in stubs:
        raise HumanModelError(f"human-trained model {model_id!r} is not registered ({registry_path().name} 'models' "
                              f"holds {sorted(stubs) or 'none'}); train and export it, then `python -m "
                              "main.human_data.clone register`")
    return dict(stubs[model_id])


def model_dir(model_id: str) -> Path:
    return models_dir() / model_id


def measure(directory: Path) -> Dict[str, Any]:
    """The stub fields of the model in ``directory``, every shape rule checked (shared by register and load)."""
    directory = Path(directory)
    mp = directory / MANIFEST_NAME
    if not mp.is_file():
        raise HumanModelError(f"{mp} is missing (a human-trained model lives outside git; train it on this machine)")
    man = json.loads(mp.read_text())
    if man.get("schema") != SCHEMA or man.get(HUMAN_DATA_KEY) is not True:
        raise HumanModelError(f"{mp}: not a {SCHEMA} manifest with \"{HUMAN_DATA_KEY}\": true")
    for key in ("id", "rowset", "rows_manifest_sha256", "zip", "trained_keys", "seed", "recipe", "held_out"):
        if key not in man:
            raise HumanModelError(f"{mp}: the manifest lacks {key!r}")
    zp, cp = directory / str(man["zip"]), directory / CONFIG_NAME
    for f in (zp, cp):
        if not f.is_file():
            raise HumanModelError(f"human-trained model {man['id']!r}: {f} is missing")
    return {"id": str(man["id"]), "rowset": str(man["rowset"]), HUMAN_DATA_KEY: True,
            "rows_manifest_sha256": str(man["rows_manifest_sha256"]), "manifest_sha256": _sha256(mp),
            "checkpoint_sha256": _sha256(zp), "config_sha256": _sha256(cp), "zip": str(man["zip"]),
            "seed": int(man["seed"]), "smoke": bool(man.get("smoke")), "recipe": man["recipe"],
            "held_out": man["held_out"], "stop": man.get("stop")}


def load(model_id: str, *, debug: bool = False) -> Dict[str, Any]:
    """The verified identity of the registered model ``model_id`` (what a clause's record carries), with the absolute
    ``zip_path`` and its ``trained_keys``. A ``smoke`` model is honoured only when ``debug`` is set."""
    want = stub(model_id)
    d = model_dir(model_id)
    got = measure(d)
    if got["id"] != model_id:
        raise HumanModelError(f"human-trained model {model_id!r} at {d}: its manifest names {got['id']!r}")
    bad = [f"{k}: {got.get(k)!r} != stub {want.get(k)!r}" for k in IDENTITY_FIELDS if got.get(k) != want.get(k)]
    if bad:
        raise HumanModelError(f"human-trained model {model_id!r} at {d} does not match its committed stub "
                              f"({registry_path().name}): " + "; ".join(bad))
    status = str(want.get("status"))
    if status == "retired":
        raise HumanModelError(f"human-trained model {model_id!r} is RETIRED in the registry")
    if (status == SMOKE_STATUS or got["smoke"]) and not debug:
        raise HumanModelError(f"human-trained model {model_id!r} is a SMOKE model (trained under overrides of the "
                              "recipe): only a --debug run may use it")
    man = json.loads((d / MANIFEST_NAME).read_text())
    return {"id": model_id, "rowset": got["rowset"], HUMAN_DATA_KEY: True, "status": status,
            "rows_manifest_sha256": got["rows_manifest_sha256"], "manifest_sha256": got["manifest_sha256"],
            "checkpoint_sha256": got["checkpoint_sha256"], "config_sha256": got["config_sha256"],
            "seed": got["seed"], "zip_path": str(d / got["zip"]),
            "trained_keys_sha256": man.get("trained_keys_sha256"), "n_trained_keys": len(man["trained_keys"])}


def trained_keys(model_id: str) -> list:
    return list(json.loads((model_dir(model_id) / MANIFEST_NAME).read_text())["trained_keys"])


def state_digest(state: Mapping[str, Any], keys: Optional[Sequence[str]] = None) -> str:
    """sha256 over the named tensors of a state dict (all of them when ``keys`` is None), keys sorted: name bytes
    then the tensor's raw bytes. The digest ``trained_keys_sha256`` is, and the one a copy is checked against."""
    import torch as th

    h = hashlib.sha256()
    for k in sorted(state if keys is None else keys):
        h.update(k.encode())
        h.update(th.as_tensor(state[k]).detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def read_policy_state(zip_path: str) -> Dict[str, Any]:
    """The policy state dict inside a checkpoint zip (the project's save path writes ``policy.pth``), on CPU."""
    import io
    import zipfile

    import torch as th

    with zipfile.ZipFile(zip_path) as z:
        return th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)


def register(directory: Path, status: str = "measurement") -> Dict[str, Any]:
    """Measure the model in ``directory`` (which must be ``model_dir(<its id>)``) and write (or replace) its stub."""
    from utils.atomic_io import atomic_write_text

    got = measure(Path(directory))
    if Path(directory).resolve() != model_dir(got["id"]).resolve():
        raise HumanModelError(f"{directory} is not the store's directory of {got['id']!r} ({model_dir(got['id'])})")
    if status not in STATUSES:
        raise HumanModelError(f"status {status!r} is not one of {STATUSES}")
    if got["smoke"] and status != SMOKE_STATUS:
        raise HumanModelError(f"{got['id']!r} was trained under --smoke overrides: it registers as "
                              f"{SMOKE_STATUS!r} only")
    entry = {k: got[k] for k in ("rowset", "rows_manifest_sha256", "manifest_sha256", "checkpoint_sha256",
                                 "config_sha256", "seed", "recipe", "held_out", "stop")}
    entry.update({HUMAN_DATA_KEY: True, "status": status,
                  "location": f"~/gen3ai_archive/human_models/{got['id']}/",
                  "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    doc = json.loads(registry_path().read_text())
    doc.setdefault("models", {})[got["id"]] = entry
    atomic_write_text(registry_path(), json.dumps(doc, indent=1) + "\n")   # the tracked file's own bytes
    return entry


def marker_of(path: Optional[str]) -> Optional[str]:
    """Why ``path`` (a checkpoint zip, a run directory, or either without its ``.zip``) is a human-trained model's,
    or None. Two rules, either suffices: the real path lies under the human-model store; a ``human_model.json``
    sits in the directory (or beside the zip)."""
    if not path:
        return None
    real = Path(os.path.realpath(os.path.expanduser(str(path))))
    store = Path(os.path.realpath(models_dir()))
    if real == store or store in real.parents:
        return f"{real} lies under the human-model store {store}"
    for d in ((real,) if real.is_dir() else ()) + (real.parent,):
        if (d / MANIFEST_NAME).is_file():
            return f"{d / MANIFEST_NAME} marks it as a human-trained model"
    return None


def refuse_human_model(path: Optional[str], where: str) -> None:
    """THE THROWING GUARD (one entry point per human-trained model): raise when ``path`` is a human-trained model's
    and the caller is not the switch."""
    why = marker_of(path)
    if why is not None:
        raise HumanModelError(f"{where} {path}: {why}; a human-trained model enters a run only through "
                              "--human-data init:<model>@<dose> or opp:<model>@<share> "
                              "(designs/endstate/human_data_architecture.md §6)")
