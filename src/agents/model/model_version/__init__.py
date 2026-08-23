"""`agents.model.model_version` — the version gate, as a package.

The re-export HUB. `src/agents/model/model_version.py` was 2,000 lines — exactly at the size
gate's hard bound, one line from tripping it — and is now one module per concern with this
`__init__.py` re-exporting every name it ever exported. `from agents.model.model_version import
<anything>` resolves unchanged for all ~48 import sites.

    constants.py      MODEL_CONFIG_VERSION · ARCH_SIGNATURE · ModelVersionError · the
                      reward-immutable field tables + `_reward_flag_repr`
    migrations.py     MIGRATION_FLOOR · SIGNATURE_FIRST_VERSION · `_migrate_config`, including
                      the PRE-FLOOR HISTORY archive
    fields.py         `ModelVersionFields` — the dataclass field block alone
    construct.py      `from_layout_and_policy_kwargs`
    compat.py         `check_compatible` — the gate that runs on EVERY load
    resume_checks.py  `check_opponent_compatible` + the six resume-immutable hparam gates
    spec.py           `ModelVersion` = fields + the three mixins, plus `to_json` /
                      `from_json_file`

**The import graph is a DAG rooted at `constants`**, which imports nothing from the package. No
submodule imports this hub back — that would close a cycle whose symptom is an `AttributeError`
on a name that plainly exists. `model_version_hub_contract_test.py` pins the hub's export list
(recovered by AST from the pre-split commit), the base list of `ModelVersion`, the cycle guard,
and that every submodule imports standalone.
"""
from agents.model.model_version.constants import (
    ARCH_SIGNATURE,
    MODEL_CONFIG_VERSION,
    ModelVersionError,
    _BELIEF_GRAD_MODE_EFFECT,
    _REWARD_FIELD_FLAGS,
    _REWARD_IMMUTABLE_FIELDS,
    _reward_flag_repr,
)
from agents.model.model_version.migrations import (
    MIGRATION_FLOOR,
    SIGNATURE_FIRST_VERSION,
    _migrate_config,
)
from agents.model.model_version.fields import ModelVersionFields
from agents.model.model_version.spec import ModelVersion

__all__ = [
    "ARCH_SIGNATURE",
    "MIGRATION_FLOOR",
    "MODEL_CONFIG_VERSION",
    "ModelVersion",
    "ModelVersionError",
    "ModelVersionFields",
    "SIGNATURE_FIRST_VERSION",
    "_BELIEF_GRAD_MODE_EFFECT",
    "_REWARD_FIELD_FLAGS",
    "_REWARD_IMMUTABLE_FIELDS",
    "_migrate_config",
    "_reward_flag_repr",
]
