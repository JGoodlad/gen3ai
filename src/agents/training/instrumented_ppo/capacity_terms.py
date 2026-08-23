"""The CAPACITY-TELEMETRY delegates (`gen3_capacity_telemetry_v1`).

Same split as `aux_terms` -> `cf_terms`: the bodies live in `agents/training/capacity_telemetry.py`
as a self-contained vertical, and these thin methods are what `train()` calls, so every call site
and every `model._capacity_*` test resolves against one name.

⚠️ **NOTHING HERE ENTERS `loss`.** This is the one difference from every other `*_terms` module in
this package, and it is the design rather than an omission. The canary trains through its OWN Adam
over its OWN parameters on a DETACHED input; the cosine probe reads gradients with
`autograd.grad` (which never writes `.grad`); the velocity probe runs under `no_grad`. So the
policy's parameter updates are bit-identical whether the flag is on or off — the flag buys scalars,
never a training change. `capacity_telemetry_test.py` asserts both halves of that on a real
`MaskablePPO`.
"""
from typing import Any, Dict, Optional

import torch as th

from agents.training import capacity_telemetry as _cap


class CapacityTerms:
    """The `--capacity-telemetry` half of the train step."""

    def _capacity(self) -> Optional[_cap.CapacityTelemetry]:
        """The live telemetry holder, or ``None`` when the flag is off.

        Built lazily and cached on the model as ``_capacity_state``. OFF costs exactly this
        boolean read: no head, no optimizer, no projection matrix, no frozen probe batch.
        """
        if not bool(getattr(self, "capacity_telemetry", False)):
            return None
        if getattr(self, "_capacity_failed", False):
            return None
        state = getattr(self, "_capacity_state", None)
        if state is None:
            state = _cap.CapacityTelemetry(
                reset_steps=int(getattr(self, "canary_reset_steps", 1_000_000)),
                cosine_every=int(getattr(self, "capacity_cosine_every", 50)),
                velocity_every=int(getattr(self, "capacity_velocity_every", 50)),
            )
            self._capacity_state = state
        return state

    @staticmethod
    def _capacity_snapshot_features(features_extractor, n_rows: int) -> Optional[th.Tensor]:
        """This minibatch's ``value_pooled``, DETACHED, or ``None`` if it is absent or stale.

        Taken immediately after `evaluate_actions` because the TD-aux / counterfactual folds each
        run their own extractor forward and REPLACE the stash. The row-count check is what makes a
        stale read a skip rather than a silent mis-pairing of features with observations.
        """
        pooled = getattr(features_extractor, "last_value_pooled", None)
        if pooled is None or int(pooled.shape[0]) != int(n_rows):
            return None
        return pooled.detach()

    def _capacity_observe(self, state, rollout_data, actions, advantages, trunk_params,
                          clip_range: float, features: Optional[th.Tensor]) -> None:
        """Per-minibatch: one canary optimizer step, plus the half-batch cosine on cadence.

        A DIAGNOSTIC must never crash the run — the same rule `rank_probe` follows — so any failure
        here disables the telemetry for the rest of the process rather than killing a 3-hour
        training window. It is loud on stderr exactly once.
        """
        try:
            state.observe(self, rollout_data, actions, advantages, trunk_params, clip_range,
                          features, int(self.num_timesteps))
        except Exception as exc:                                # pragma: no cover - defensive
            self._capacity_disable(exc)

    def _capacity_finish(self, state) -> Dict[str, float]:
        """Per-`train()`: fold the samples, run the velocity probe on cadence, return the scalars."""
        try:
            return state.finish_train(self, int(self.num_timesteps))
        except Exception as exc:                                # pragma: no cover - defensive
            self._capacity_disable(exc)
            return {}

    def _capacity_disable(self, exc: BaseException) -> None:    # pragma: no cover - defensive
        """Turn the telemetry off for the rest of the process, LOUDLY and exactly once.

        It sets a private `_capacity_failed` latch rather than clearing `capacity_telemetry`,
        deliberately: that attribute is what `model_config.json` records, and a run that was
        LAUNCHED with the flag should not later claim it was not. The recorded provenance stays
        true; the probe stops.
        """
        import sys
        if not getattr(self, "_capacity_failed", False):
            print(f"[Capacity] telemetry DISABLED after an error (training is unaffected — the "
                  f"probe carries no gradient into the policy): {exc!r}", file=sys.stderr, flush=True)
        self._capacity_failed = True
        self._capacity_state = None


def capacity_startup_banner(model: Any) -> str:
    """One line for the launch announcer, or ``""`` when the flag is off."""
    if not bool(getattr(model, "capacity_telemetry", False)):
        return ""
    return (f"🩺 [Capacity] LIVE TELEMETRY ON: plasticity canary (K={_cap.CANARY_K}, "
            f"reset every {int(getattr(model, 'canary_reset_steps', 1_000_000)):,} env steps) · "
            f"half-batch trunk cosine every {int(getattr(model, 'capacity_cosine_every', 50))} "
            f"minibatches · feature velocity every "
            f"{int(getattr(model, 'capacity_velocity_every', 50))} train() calls. "
            f"Canary state is NOT checkpointed — it re-inits on every resume/restart.")
