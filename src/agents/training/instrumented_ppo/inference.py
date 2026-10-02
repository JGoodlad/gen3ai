"""The INFERENCE-ONLY load (`gen3_opponent_inference_load_v1`): an opponent never trains, so an opponent
load never acquires training state.

THE BUG IT CLOSES (2026-10-01, the X26 ride-along baseline at N = 256): a self-play pool snapshot was
loaded through `InstrumentedMaskablePPO.load`, whose `_setup_model` ends with the ride-along heads'
STARTUP ACQUISITION (`RideAlongTerms._ridealong_acquire`): it built an Adam over the snapshot's 111
ride-along tensors and pre-stepped it (`preallocate_adam_state`). The pool loads at the first eval, AFTER
the learner froze, so K6's global optimizer-step hook FATALed the run ("an optimizer outside the frozen
set STEPPED after the freeze"). In X26 that would have run until the self-play seeding win rate, hours
in, and died.

THE PRINCIPLE, made structural rather than a skip flag: `InferenceMaskablePPO` is the class an
opponent / reader load builds. Its `_setup_model` builds the POLICY and nothing a learner needs:

  * no ride-along optimizer (the heads' weights load, so an offline reader still reads them; nothing
    steps them — a ride-along step on this model raises `RideAlongLifecycleViolation`);
  * no policy optimizer at all: the policy is constructed with an optimizer class that builds NOTHING
    (`_no_optimizer`), so no `torch.optim.Optimizer` is ever constructed, and the checkpoint's saved
    Adam moments (2x the policy's parameters) are never loaded into one;
  * no rollout buffer, no loop hooks, no clip schedules.

It refuses to `learn`, `train`, `collect_rollouts` or `save` (`InferenceOnlyModelError`): a model that
carries no optimizer state must never write a checkpoint a resume would read as a learner's.

Everything else is `MaskablePPO._setup_model` in its order, MINUS `set_random_seed(self.seed)`
(`gen3_no_global_reseed_v1`): a load must never touch the process's global RNG. sb3's setup re-seeded
Python `random`, NumPy and torch to the SNAPSHOT's saved seed — the run's own `--seed` — so every pool
refresh / sentinel / rung load mid-run rewound the trainee's global streams to their startup state,
and the PPO minibatch permutation (`np.random.permutation` in `RolloutBuffer.get`) replayed the run's
first updates' sequence after every load (measured, `designs/training/learner_lifecycle.md` "No global
reseed after the freeze"). The policy is built inside `global_rng_guard.isolated_global_rng()`: its
init draws (and the ride-along heads' constructor `torch.manual_seed`s) come from torch's CPU
generator, which the scope restores, so the global streams after a load are exactly where they were
before it. The weights the construction drew are overwritten by `set_parameters`.

The TRAINEE is unaffected: its resume / fork (`load_model_snapshot` with an env) is an
`InstrumentedMaskablePPO` that acquires every ride-along optimizer at startup, before the freeze, and is
checked strictly on every key (`ModelVersion.check_compatible`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agents.training.global_rng_guard import isolated_global_rng
from agents.training.instrumented_ppo.ppo import InstrumentedMaskablePPO


class InferenceOnlyModelError(RuntimeError):
    """An inference-only (opponent / reader) load was asked to train or to write a checkpoint."""


def _no_optimizer(*_args: Any, **_kwargs: Any) -> None:
    """The policy's `optimizer_class` on an inference load: builds nothing (SB3's `_build` assigns the
    result to `policy.optimizer`, which is therefore None)."""
    return None


#: The only saved object an inference load restores: the policy's weights.
_INFERENCE_PARAMS = ("policy",)


class InferenceMaskablePPO(InstrumentedMaskablePPO):
    """Policy weights only — the class every opponent / reader load builds (module docs)."""

    inference_only = True

    def _setup_model(self) -> None:
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

        self._setup_lr_schedule()
        # NO `self.set_random_seed(self.seed)` (module docs, gen3_no_global_reseed_v1): an opponent load
        # never seeds — nor draws from — a process-global RNG stream.
        pk = {**self.policy_kwargs, "optimizer_class": _no_optimizer, "optimizer_kwargs": {}}
        with isolated_global_rng():
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space, self.action_space, self.lr_schedule, **pk)
            self.policy = self.policy.to(self.device)
        if not isinstance(self.policy, MaskableActorCriticPolicy):
            raise ValueError("Policy must subclass MaskableActorCriticPolicy")
        # The ride-along heads' optimizer slots exist and are EMPTY: a step refuses
        # (`RideAlongLifecycleViolation`), and nothing here is ever acquired.
        self._ridealong_opt = None
        self._ridealong_opt_owner = None
        self._ridealong_vopts = {}
        self._ridealong_vopts_owner = None
        self._ridealong_variant_disabled = set()

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        return list(_INFERENCE_PARAMS), []

    def set_parameters(self, load_path_or_dict: Any, exact_match: bool = True,
                       device: Any = "auto") -> None:
        """The policy's weights ONLY: the saved optimizer state is dropped, never loaded."""
        params: Dict[str, Any]
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            from stable_baselines3.common.save_util import load_from_zip_file

            _, params, _ = load_from_zip_file(load_path_or_dict, device=device, load_data=False)
        weights = {k: v for k, v in (params or {}).items() if k in _INFERENCE_PARAMS}
        super().set_parameters(weights, exact_match=exact_match, device=device)

    def _refuse(self, what: str) -> InferenceOnlyModelError:
        return InferenceOnlyModelError(
            f"{what} on an INFERENCE-ONLY load (an opponent / reader, gen3_opponent_inference_load_v1): "
            "it carries no optimizer, no optimizer state and no rollout buffer. Load a model that "
            "trains through `load_model_snapshot` with its env (the trainee), or "
            "`load_foreign_opponent(..., inference_only=False)` for an offline tool that fits one.")

    def learn(self, *args: Any, **kwargs: Any) -> Any:
        raise self._refuse("learn()")

    def train(self) -> None:
        raise self._refuse("train()")

    def collect_rollouts(self, *args: Any, **kwargs: Any) -> Any:
        raise self._refuse("collect_rollouts()")

    def save(self, *args: Any, **kwargs: Any) -> None:
        raise self._refuse("save()")


__all__ = ["InferenceMaskablePPO", "InferenceOnlyModelError"]
