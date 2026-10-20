"""Test stand-ins for ``MaskablePPO.save(path)``.

Production saves go through `utils.atomic_io.atomic_save_model`, which hands ``model.save`` a hidden temp name
(``.<name>.<pid>.<token>.tmp``) and replaces it onto the final name only when the writer produced it, so a fake
``save`` must write the file SB3 would write for the name it is given: the name itself, with ``.zip`` appended only
to a suffix-less one (`atomic_io.model_zip_path`). A fake that appends ``.zip`` to every name writes a stray
file and the save is refused (`AtomicWriteError`: the writer produced no file).
"""
from __future__ import annotations

from typing import Callable

from utils.atomic_io import model_zip_path


def fake_save(path: str) -> None:
    """``model.save`` that writes an empty file where SB3 would."""
    with open(model_zip_path(path), "wb"):
        pass


def fake_save_bytes(payload: bytes) -> Callable[[str], None]:
    """``model.save`` that writes ``payload`` where SB3 would."""

    def save(path: str) -> None:
        with open(model_zip_path(path), "wb") as f:
            f.write(payload)

    return save
