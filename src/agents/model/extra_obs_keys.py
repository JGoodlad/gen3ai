"""THE declared registry of Dict obs keys the extractor's forward reads BEYOND ``observation``.

WHY THIS MODULE EXISTS. ``Gen3FeaturesExtractor.forward`` is normally a pure function of
``obs["observation"]`` — the flat 2501-dim vector — and every synthetic-obs caller in the tree was
written against that one-key shape::

    obs = {"observation": torch.zeros(1, layout["total_dim"])}

``gen3_value_true_team_v1`` (v114) broke that assumption: the privileged value route reads its own
training-and-eval-only Dict key, ``opp_true_team``, and RAISES rather than skipping when it is
missing (a silent skip is the gen-12 dead-tail bug the seam exists to prevent). The two facts were
independently correct and had never been exercised together, so the FIRST launch that combined
``--value-true-team`` with the forkserver compile preload
(``ai_v12_14_ladder_truevalue`` @ 377a5aa1) died two minutes in at env init: the preload's
one-key trace input hit the seam and killed the forkserver bootstrap.

The bug is not "the preload forgot a key" — it is that a synthetic obs was assembled by HAND at
each site, so every future obs-key-adding flag re-arms the same trap at every one of them. This
module is the fix: the (extractor attribute -> obs key, shape) mapping is DECLARED once, here,
beside the seam that reads it, and every synthetic-obs caller builds its dict from
``synthetic_obs`` instead of spelling one out.

**The enable condition is the ATTRIBUTE, not the flag.** ``ExtraObsKey.attr`` names the extractor
attribute whose non-``None``-ness the forward itself tests (``if self.true_team_value is not
None:``), so the registry and the seam agree by construction rather than by two people reading the
same flag name. A config-derived predicate would drift the moment a flag's build condition grew a
second term.

**The drift gate is ``extra_obs_keys_test.py``**, which AST-scans ``extractor_forward`` for every
``obs.get(...)`` / ``obs[...]`` read and fails on a key that is not declared here. A new obs-key
flag therefore cannot reach a launch without a row in this table.

⚠️ **ADOPTION IS PARTIAL, and the boundary is deliberate.** Every site a TRAINING RUN reaches now
builds through this module — the forkserver preload, the round-trip smoke, ``compile_trainer``,
``compile_opponents`` and ``warmstart``. The OFFLINE audit / probe CLIs still hand-build a one-key
dict (``critic_route_audit``, ``edge_ablation_audit``, ``op_block_split_audit``,
``capacity_probes``, ``concat_readout_probe``, ``feature_coverage/_support``, ``cf_terms``,
``cf_producer_snapshot``, ``capacity_telemetry``, ``instrumented_ppo/rollout_probes``,
``teacher/buffer``, ``search_dividend/{perf,search,ab_racing}``, ``harvest``,
``visualize_arch``, ``rust_sim/harness/g1_bakeoff``). Those fail in the first second at a terminal
with the seam's own message naming the fix, which costs a retype rather than a GPU-hour — a
different severity, tracked in ``designs/ops/TECH_DEBT_BACKLOG.md``. When you touch one, route it
through ``zero_extra_obs`` rather than adding a key by hand. ``prober/model`` is NOT on that
list: it REFUSES such a checkpoint on purpose, because a V re-forwarded from the recorded
observation vector alone is V stripped of the privilege — a different quantity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from agents.observation.true_team import (
    TRUE_TEAM_KEY, TRUE_TEAM_SHAPE, empty_true_team_block,
)

#: The Dict keys every obs carries, on every path, with no flag. ``action_mask`` is consumed by the
#: POLICY (the pointer head's mask), not by the extractor forward, so a trace input may omit it —
#: but it is named here so the drift gate can tell "undeclared" from "unconditional".
BASE_OBS_KEYS: Tuple[str, ...] = ("observation", "action_mask")


@dataclass(frozen=True)
class ExtraObsKey:
    """One flag-gated Dict obs key the extractor's forward READS.

    ``attr`` is the extractor attribute the forward tests for ``None`` — the same expression, so
    the two cannot disagree. ``build_zeros`` is the canonical all-zero block builder for the key
    (the honest "this input is unavailable" encoding); it defaults to plain ``np.zeros(shape)``.
    """
    key: str
    attr: str
    shape: Tuple[int, ...]
    flag: str
    dtype: str = "float32"
    build_zeros: Optional[Callable[[], np.ndarray]] = None

    def zeros(self) -> np.ndarray:
        """One unbatched all-zero block for this key."""
        if self.build_zeros is not None:
            return self.build_zeros()
        return np.zeros(self.shape, dtype=np.dtype(self.dtype))


EXTRA_OBS_KEYS: Tuple[ExtraObsKey, ...] = (
    # gen3_value_true_team_v1 (v114). The all-zero block is not a stand-in invented for the trace:
    # it is exactly what `Gen3Env` / `RLPlayer` supply when no privileged view exists (ladder play,
    # a battle with no `_opp_player` handle), so the graph traced against it IS the graph the
    # workers run.
    ExtraObsKey(key=TRUE_TEAM_KEY, attr="true_team_value", shape=TRUE_TEAM_SHAPE,
                flag="value_true_team", build_zeros=empty_true_team_block),
)

BY_KEY: Dict[str, ExtraObsKey] = {e.key: e for e in EXTRA_OBS_KEYS}
if len(BY_KEY) != len(EXTRA_OBS_KEYS):                # a duplicate would silently shadow a row
    raise RuntimeError("EXTRA_OBS_KEYS contains a duplicate obs key")


def required_extra_obs_keys(extractor: Any) -> Tuple[ExtraObsKey, ...]:
    """Every declared extra key THIS extractor instance's forward will read.

    Reads the same attributes the forward's own guards read, so it answers for the object rather
    than for the config that (maybe) built it — which is what makes it correct on a RELOADED
    checkpoint, where no argv is in scope at all.
    """
    return tuple(e for e in EXTRA_OBS_KEYS if getattr(extractor, e.attr, None) is not None)


def zero_extra_obs(extractor: Any, batch: int = 1, device: Any = None) -> Dict[str, Any]:
    """`{key: all-zero block}` for every extra key THIS extractor's forward reads. Torch tensors.

    The all-zero block is not a stand-in invented for synthetic callers: it is the encoding a REAL
    emitter supplies when the privileged view is unavailable, so a graph traced against it — or a
    policy-only forward run over it — is the one production runs. Use it to TOP UP an obs dict
    whose `observation` came from real data (`warmstart`'s behaviour cloning is the case: the
    routes that read these keys are vf-only, so the policy logits it reads back are unaffected).
    """
    import torch                                      # local: keeps the registry import-light

    out: Dict[str, Any] = {}
    for entry in required_extra_obs_keys(extractor):
        block = np.stack([entry.zeros() for _ in range(batch)])
        tensor = torch.as_tensor(block)
        out[entry.key] = tensor.to(device) if device is not None else tensor
    return out


def synthetic_obs(extractor: Any, total_dim: int, batch: int = 1,
                  device: Any = None, action_mask: bool = False) -> Dict[str, Any]:
    """The obs dict a TRACE / SMOKE / round-trip path must hand this extractor.

    ``observation`` zeros plus one all-zero block per required extra key. Torch tensors, on
    ``device`` when given. ``action_mask`` is opt-in because the extractor never reads it; a caller
    that goes on to run the POLICY wants it.

    🚨 **Never hand-build this dict.** Every site that did is one launch away from the
    ``ai_v12_14_ladder_truevalue`` crash this module's docstring records.
    """
    import torch                                      # local: keeps the registry import-light

    obs: Dict[str, Any] = {
        "observation": torch.zeros(batch, total_dim, device=device),
    }
    if action_mask:
        obs["action_mask"] = torch.ones(batch, 11, dtype=torch.int8, device=device)
    obs.update(zero_extra_obs(extractor, batch=batch, device=device))
    return obs
