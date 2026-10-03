"""`agents.training.instrumented_ppo` — MaskablePPO + our whole training-side loss fold.

The re-export HUB. `instrumented_ppo.py` was 2,152 lines and is now one module per concern; every
name it ever exported still resolves from `agents.training.instrumented_ppo`.

    ppo.py            the class + `train()` — the vendored upstream override and THE FOLD
                      SEQUENCE, deliberately kept in ONE straight-line module (see its docstring),
                      plus `train_step_source()`: the unit a source-level pin should read
    train_setup.py    the PRE-LOOP half of `train()` — the opponent-intent label alignment, the
                      FOLD FLAGS (`FoldFlags`) and the once-per-call probes (`ProbeSetup`), both
                      unpacked back into the locals the loop was written against
    micro_step.py     K8: the LEARNER MICRO-STEP — `evaluate_actions` + fold steps 1-3a as ONE static
                      function (region R1 under --compile-trainer); `pack`/`unpack` = its one host read
    metrics_export.py the ~400-line `self.logger.record` tail — diagnostics, no gradient, one
                      method per TB prefix group
    rollout_probes.py `collect_rollouts` and the episode-start read — per-ROLLOUT work that is
                      not part of the fold at all
    hparams.py        every after-construction knob `train_rl_agent` sets, with its rationale
    noise_scale.py    the McCandlish gradient-noise-scale estimator + the NSR advisor
    value_terms.py    the win-prob BCE and its opponent-stratified weights
    aux_terms.py      the `belief_bank` / `td_aux` delegates
    capacity_terms.py the `--capacity-telemetry` delegates — the ONE `*_terms` module that folds
                      NOTHING into `loss` (see its docstring)
    constants.py      the four module-level tuning constants
    device_batches.py K8: the DEVICE-RESIDENT micro-batch — one device copy of the flattened buffer
                      per update, micro-batches gathered from it (bit-identical, same permutation)
    intent_fold.py    K8: the opponent-intent fold as ONE static, fullgraph-traceable function
                      (called by `micro_step`, region R1; its oracle is a verbatim copy of the old block)
    inference.py      `InferenceMaskablePPO` — the class an OPPONENT / reader load builds: policy
                      weights only, no optimizer of any kind (gen3_opponent_inference_load_v1)
    phase_hook.py     BENCHMARK-ONLY segment marks inside `train()` (None in production; the one
                      consumer is `agents.training.learner_benchmark`)

`_verify_upstream_unchanged` and `_EXPECTED_UPSTREAM_TRAIN_HASH` stay HERE, in the hub, on
purpose: `instrumented_ppo_test` patches that global on the module object it imports, so moving
it to a submodule would have left the patch reaching a different global than the function reads
— a test that still passes, for the wrong reason.

InstrumentedMaskablePPO — MaskablePPO with `train/clip_fraction_vf` added.

sb3_contrib's MaskablePPO logs `train/clip_fraction` (policy clip fraction) but
not the equivalent metric for value-function clipping, even though VF clipping
is applied when `clip_range_vf` is non-None. This subclass copies the upstream
`train()` method verbatim and adds three lines: a per-batch fraction
computation, accumulation, and one final `self.logger.record(...)` call.

# Drift detection

`train()` is a vendored copy of upstream code. If sb3_contrib ever updates the
method, our copy will silently become stale and we'd run different training
logic than upstream. To prevent that, `_verify_upstream_unchanged()` runs at
import time and computes the SHA256 of `inspect.getsource(MaskablePPO.train)`.
A mismatch raises immediately with both hashes and instructions.

If upstream changes (e.g. after a `pip install -U sb3_contrib`):
1. Diff the new upstream `train()` against the one this subclass was vendored
   from (last known hash is in `_EXPECTED_UPSTREAM_TRAIN_HASH`).
2. Port any non-instrumentation changes into the `train()` override below.
3. Recompute the hash and update `_EXPECTED_UPSTREAM_TRAIN_HASH`.

The same pattern guards the torch._dynamo internals the compile sentinel rests on:
`agents.model.compile_control.verify_torch_internals` (`_SOURCE_HASHES`).
"""

import hashlib
import inspect
import pathlib

from sb3_contrib import MaskablePPO

from agents.training.instrumented_ppo.constants import (
    _NOISE_SCALE_EMA_DECAY,
    _WIN_CONTESTED_TAU,
)

#: The file the drift message tells you to re-port into — `train()`'s home since the
#: decomposition. Derived, not typed, and pinned by `instrumented_ppo_test` so the message can
#: never name a file that is not there (which is what `__file__` started doing the moment the
#: override moved out of the hub).
_TRAIN_OVERRIDE_FILE = pathlib.Path(__file__).with_name("ppo.py")

# SHA256 of inspect.getsource(MaskablePPO.train) at the time this file was
# written. If sb3_contrib updates and this no longer matches, _verify_...
# raises at import time.
_EXPECTED_UPSTREAM_TRAIN_HASH = (
    "79500464b6a71d5adcfdf10028df56fbaf72b7754952e760f9e377610b9cf809"
)
# The supervised belief-head losses + their scale constants live in `belief_bank` (the
# declarative fold of design_unified_belief.md §4 — one ROW per head instead of one inline
# vertical per head). Re-exported here because tests and older call sites import them from
# this module.
from agents.training.belief_bank import (   # noqa: F401  (re-exports)
    _EV_LOSS_SCALE, _NATURE_CE_WEIGHT, _EV_LOSS_WEIGHT, _SPREAD_LOSS_SCALE,
    _LATENT_STD_TARGET, _LATENT_VICREG_WEIGHT,
)
def _verify_upstream_unchanged() -> None:
    """Fail-loud at import time if the upstream `MaskablePPO.train()` source
    has drifted from what this subclass was vendored against."""
    source = inspect.getsource(MaskablePPO.train)
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != _EXPECTED_UPSTREAM_TRAIN_HASH:
        upstream_file = inspect.getfile(MaskablePPO)
        raise RuntimeError(
            "[InstrumentedMaskablePPO] DRIFT DETECTED: upstream "
            "`sb3_contrib.MaskablePPO.train()` source has changed since this "
            "subclass was vendored.\n"
            f"  Expected SHA256: {_EXPECTED_UPSTREAM_TRAIN_HASH}\n"
            f"  Actual SHA256:   {actual}\n"
            f"  Upstream file:   {upstream_file}\n\n"
            "ACTION REQUIRED: diff the upstream train() against this subclass, "
            "port any non-instrumentation changes into the override in "
            f"{_TRAIN_OVERRIDE_FILE}, then update _EXPECTED_UPSTREAM_TRAIN_HASH "
            "to silence this check."
        )


_verify_upstream_unchanged()
from agents.training.instrumented_ppo.ppo import InstrumentedMaskablePPO   # noqa: E402
from agents.training.instrumented_ppo.inference import (   # noqa: E402
    InferenceMaskablePPO, InferenceOnlyModelError)

__all__ = [
    "InferenceMaskablePPO",
    "InferenceOnlyModelError",
    "InstrumentedMaskablePPO",
    "_EV_LOSS_SCALE",
    "_EV_LOSS_WEIGHT",
    "_EXPECTED_UPSTREAM_TRAIN_HASH",
    "_LATENT_STD_TARGET",
    "_LATENT_VICREG_WEIGHT",
    "_NATURE_CE_WEIGHT",
    "_NOISE_SCALE_EMA_DECAY",
    "_SPREAD_LOSS_SCALE",
    "_WIN_CONTESTED_TAU",
    "_verify_upstream_unchanged",
]
