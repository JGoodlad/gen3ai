"""The Metamon WEIGHT PINS: one hub revision, one sha256 per weight file, offline, refuse on any difference.

STDLIB ONLY, and shared by both interpreters. ``metamon_side.py`` runs under the Metamon env (``PYTHONPATH=""``,
nothing of ours importable) and loads this file by path; ``main.anchors`` imports it as an ordinary module. One
implementation, so the digest the peer verifies and the digest the cell stamps cannot drift.

THE DEFECT THIS CLOSES (FR-1 FINDING 3). ``get_pretrained_model(...).initialize_agent`` fetches its weights with
``huggingface_hub.hf_hub_download(repo_id, filename, cache_dir)``: NO ``revision`` (so it resolves the hub's moving
``main``) and online. A read could silently load newer upstream weights if ``main`` moved, under a regime id that
claimed the same policy. So the peer runs with ``HF_HUB_OFFLINE=1``, :func:`pinned_downloader` replaces the call
(forcing ``revision=<pin>`` and ``local_files_only=True``), and every file is hashed before play.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable, Dict, List, Mapping

#: A full commit sha. A short sha or a branch name ("main") is a MOVING reference and is refused.
REVISION_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_CHUNK = 1 << 22


class WeightPinError(RuntimeError):
    """A weight file is unpinned, missing, or does not hash to its pin: the peer must not play."""


def check_pins(revision: Any, weights: Any, who: str = "") -> None:
    """Refuse a pin that is not a full commit sha plus a non-empty ``{path: sha256}`` map."""
    pre = f"{who}: " if who else ""
    if not (isinstance(revision, str) and REVISION_RE.fullmatch(revision)):
        raise WeightPinError(f"{pre}hf_revision {revision!r} is not a full 40-hex commit sha (a branch or short sha moves)")
    if not isinstance(weights, Mapping) or not weights:
        raise WeightPinError(f"{pre}weights must be a non-empty {{file path: sha256}} map, got {weights!r}")
    for rel, sha in weights.items():
        if not (isinstance(rel, str) and rel and not rel.startswith("/") and ".." not in rel.split("/")):
            raise WeightPinError(f"{pre}weight path {rel!r} must be a relative path inside the hub repo")
        if not (isinstance(sha, str) and SHA256_RE.fullmatch(sha)):
            raise WeightPinError(f"{pre}weight {rel!r}: sha256 {sha!r} is not 64 lowercase hex")


def weights_digest(weights: Mapping[str, str]) -> str:
    """The ONE id of a weight set: sha256 of the canonical JSON of ``{path: sha256}``."""
    return hashlib.sha256(json.dumps(dict(weights), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def force_offline(environ: Any = None) -> None:
    """Set the hub's offline switches, OVERRIDING whatever the caller exported. Before ``huggingface_hub`` imports:
    it reads them into constants at import."""
    env = os.environ if environ is None else environ
    for name in OFFLINE_ENV:
        env[name] = "1"


def assert_offline(constants: Any) -> None:
    """Refuse unless the imported hub library is actually in offline mode (the env var may have been read too late)."""
    if getattr(constants, "HF_HUB_OFFLINE", None) is not True:
        raise WeightPinError("huggingface_hub is not in offline mode (HF_HUB_OFFLINE did not take effect before its "
                             "import); a download of unpinned weights is possible, so the peer refuses to run")


def pinned_downloader(real: Callable[..., str], *, repo: str, revision: str, weights: Mapping[str, str],
                      loaded: List[str]) -> Callable[..., str]:
    """``hf_hub_download`` replacement. Only a pinned file of the pinned repo, only at the pinned revision, only from
    the local cache, and only if its content hashes to the pin; ``loaded`` collects the paths it served."""
    check_pins(revision, weights)
    seen: Dict[str, str] = {}

    def download(*args: Any, **kwargs: Any) -> str:
        names = ("repo_id", "filename")
        bound = {n: a for n, a in zip(names, args)}
        for n in names:
            if n in kwargs:
                bound[n] = kwargs.pop(n)
        args = args[len(names):]
        if bound.get("repo_id") != repo:
            raise WeightPinError(f"download of repo {bound.get('repo_id')!r}: only {repo!r} is pinned")
        rel = bound.get("filename")
        if rel not in weights:
            raise WeightPinError(f"download of {rel!r}: not a pinned weight file (pinned: {sorted(weights)})")
        if kwargs.get("revision") not in (None, revision):
            raise WeightPinError(f"download of {rel!r} asked for revision {kwargs['revision']!r}, the pin is {revision}")
        if kwargs.get("force_download"):
            raise WeightPinError(f"download of {rel!r} asked force_download: weights are never re-fetched")
        kwargs.update(revision=revision, local_files_only=True)
        try:
            path = real(bound["repo_id"], rel, *args, **kwargs)
        except Exception as exc:             # noqa: BLE001 - a missing file must read as a refusal, not a download
            raise WeightPinError(f"pinned weight {rel!r} @ {revision[:8]} is not in the local cache: "
                                 f"{type(exc).__name__}: {exc}") from exc
        real_path = os.path.realpath(path)
        if real_path not in seen:
            seen[real_path] = sha256_file(real_path)
        if seen[real_path] != weights[rel]:
            raise WeightPinError(f"weight {rel!r} @ {revision[:8]} hashes to {seen[real_path]}, the pin is {weights[rel]}")
        loaded.append(rel)
        return path

    return download


def verify_all(download: Callable[..., str], repo: str, weights: Mapping[str, str], cache_dir: str) -> Dict[str, str]:
    """Hash EVERY pinned file through ``download`` before play; returns ``{path: sha256}`` (== the pin, or it raised)."""
    out = {}
    for rel in sorted(weights):
        download(repo_id=repo, filename=rel, cache_dir=cache_dir)
        out[rel] = weights[rel]
    return out
