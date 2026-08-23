"""`ModelVersion` -- the assembled record, and its JSON IO.

The class is `ModelVersionFields` plus one mixin per gate family. That trades a file-size problem
for a BASE-LIST problem (a family can drop out of the bases without any import failing), which is
why `model_version_hub_contract_test.py` pins the base list and every gate method by name.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Self

from agents.model.model_version.compat import ModelVersionCompatibility
from agents.model.model_version.construct import ModelVersionConstruction
from agents.model.model_version.migrations import _migrate_config
from agents.model.model_version.resume_checks import ModelVersionResumeChecks


@dataclass
class ModelVersion(ModelVersionConstruction,
                   ModelVersionCompatibility,
                   ModelVersionResumeChecks):
    """The weight-shape / architecture record written beside every checkpoint.

    Fields: `fields.py`. Construction: `construct.py`. The every-load gate: `compat.py`.
    The resume-only and opponent-only gates: `resume_checks.py`.
    """

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json_file(cls, path: str) -> Self:
        with open(path) as f:
            data = json.load(f)
        data = _migrate_config(data)
        return cls(**data)
