"""`agents.model.model_version` — the version gate, as a package.

The re-export HUB. `src/agents/model/model_version.py` was 2,000 lines — exactly at the size
gate's hard bound, one line from tripping it — and is now one module per concern with this
`__init__.py` re-exporting every name it ever exported. `from agents.model.model_version import
<anything>` resolves unchanged for all ~48 import sites.

    constants.py      MODEL_CONFIG_VERSION · ARCH_SIGNATURE · OBS_SEMANTICS_VERSION · ModelVersionError ·
                      the reward-immutable field table
    migrations.py     MIGRATION_FLOOR · SIGNATURE_FIRST_VERSION · `_migrate_config`, including
                      the PRE-FLOOR HISTORY archive
    fields.py         `ModelVersionFields` — the dataclass field block alone
    construct.py      `from_layout_and_policy_kwargs`
    compat.py         `check_compatible` — the gate that runs on EVERY load
    resume_checks.py  `check_opponent_compatible` + the six resume-immutable hparam gates
    spec.py           `ModelVersion` = fields + the three mixins, plus `to_json` /
                      `from_json_file`
    shaped_reward.py  a checkpoint TRAINED WITH THE DELETED SHAPED REWARD, recognised from its
                      raw config, so a resume / fork refuses (`ShapedRewardCheckpointError`)
    retired_levers.py the DELETED Python-core levers, recognised the same way (`RETIRED`,
                      `RetiredLeverCheckpointError`) — each deletion unit appends its levers
    version_break.py  THE X5 VERSION BREAK (v144): the last blob-capable commit, the belief-specific
                      pre-floor diagnosis, the pickled `belief_tokens` judgment, `PreBreakCheckpointError`

**The import graph is a DAG rooted at `constants`**, which imports nothing from the package. No
submodule imports this hub back — that would close a cycle whose symptom is an `AttributeError`
on a name that plainly exists. `model_version_hub_contract_test.py` pins the hub's export list
(recovered by AST from the pre-split commit), the base list of `ModelVersion`, the cycle guard,
and that every submodule imports standalone.
"""
from agents.model.model_version.constants import (
    ARCH_SIGNATURE,
    MODEL_CONFIG_VERSION,
    OBS_SEMANTICS_REASON,
    OBS_SEMANTICS_VERSION,
    ModelVersionError,
    _BELIEF_GRAD_MODE_EFFECT,
    _REWARD_IMMUTABLE_FIELDS,
)
from agents.model.model_version.migrations import (
    MIGRATION_FLOOR,
    SIGNATURE_FIRST_VERSION,
    _migrate_config,
)
from agents.model.model_version.fields import ModelVersionFields
from agents.model.model_version.shaped_reward import (
    DELETED_SHAPED_REWARD_FIELDS,
    ShapedRewardCheckpointError,
)
from agents.model.model_version.retired_levers import (
    RETIRED_FIELDS,
    RetiredLeverCheckpointError,
)
from agents.model.model_version.spec import ModelVersion
from agents.model.model_version.version_break import (
    LAST_BLOB_COMMIT,
    PreBreakCheckpointError,
)

__all__ = [
    "ARCH_SIGNATURE",
    "MIGRATION_FLOOR",
    "MODEL_CONFIG_VERSION",
    "OBS_SEMANTICS_REASON",
    "OBS_SEMANTICS_VERSION",
    "ModelVersion",
    "ModelVersionError",
    "ModelVersionFields",
    "DELETED_SHAPED_REWARD_FIELDS",
    "RETIRED_FIELDS",
    "RetiredLeverCheckpointError",
    "ShapedRewardCheckpointError",
    "LAST_BLOB_COMMIT",
    "PreBreakCheckpointError",
    "SIGNATURE_FIRST_VERSION",
    "_BELIEF_GRAD_MODE_EFFECT",
    "_REWARD_IMMUTABLE_FIELDS",
    "_migrate_config",
]
