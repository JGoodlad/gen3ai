"""The owned PPO loop's contracts after stage 3 (`instrumented_ppo/loop.py`; deletion pass U4).

* **The dump comes BEFORE the update** — update k's ``train/*`` is written at the dump after rollout k+1;
* **the final dump** (P3, `gen3_final_update_dump_v1`) writes the LAST update's scalars at ``learn()``'s end
  — revert it and the last update's ``train/*`` are never written (the dump list loses its last entry);
* **the env is a `TrainerVecEnv`** — sb3's constructor would wrap anything else in a ``DummyVecEnv`` +
  ``Monitor`` that nothing steps, so `OwnedLoop._wrap_env` refuses it;
* **the callback protocol is declared** (`loop_callbacks`): an sb3 callback or a bare function is refused,
  an undeclared step local is refused, a `CallbackList` dispatches in list order and stops on any False.

The update arithmetic is the K9 learner golden's bar; the hook nesting is `loop_hooks_test`'s.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest

from agents.training import loop_callbacks as LC
from agents.training.instrumented_ppo.loop import LOOP_PHASES
from agents.training.rust_rollout.testkit import ToyVecEnv, attach_vec_collector, record_dumps

N_STEPS, N_ENVS = 8, 2


def _model() -> Any:
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from agents.training.instrumented_ppo_test import _CounterDictEnv

    env = ToyVecEnv([(lambda: _CounterDictEnv()) for _ in range(N_ENVS)])
    return attach_vec_collector(InstrumentedMaskablePPO(
        "MultiInputPolicy", env, n_steps=N_STEPS, batch_size=4, n_epochs=1, device="cpu", seed=0))


def test_the_phase_table_ends_with_the_final_dump():
    assert LOOP_PHASES[-2:] == ("training_end", "final_dump")


def test_dump_before_update_and_the_final_dump_writes_the_last_update():
    model = _model()
    dumps = record_dumps(model)
    model.learn(total_timesteps=2 * N_STEPS * N_ENVS)              # two rollouts, two updates
    assert model._n_updates == 2
    rows = N_STEPS * N_ENVS
    assert [d["_step"] for d in dumps] == [rows, 2 * rows, 2 * rows], [d["_step"] for d in dumps]
    first, second, final = dumps
    assert "train/n_updates" not in first                            # before any update
    assert second["train/n_updates"] == 1                            # update 1, at the dump after rollout 2
    assert final["train/n_updates"] == 2                             # update 2: the final dump (P3)
    assert "time/fps" in second and "time/fps" not in final          # the final dump is the PENDING values only
    assert not model.logger.name_to_value                            # nothing left pending


def test_nothing_pending_means_no_final_dump():
    model = _model()
    dumps = record_dumps(model)
    model._final_dump()
    assert dumps == []


def test_the_learners_env_must_be_a_trainer_env():
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from agents.training.instrumented_ppo_test import _CounterDictEnv

    with pytest.raises(TypeError, match="TrainerVecEnv"):
        InstrumentedMaskablePPO("MultiInputPolicy", DummyVecEnv([_CounterDictEnv]), n_steps=N_STEPS,
                                batch_size=4, device="cpu")
    assert isinstance(_CounterDictEnv().observation_space, spaces.Dict)


class _Rec(LC.BaseCallback):
    def __init__(self, log: List[str], name: str, stop: bool = False) -> None:
        super().__init__()
        self.log, self.name, self.stop = log, name, stop

    def _on_step(self) -> bool:
        self.log.append(f"{self.name}:{self.locals.get('dones')}")
        return not self.stop


def test_the_callback_protocol_is_declared():
    from stable_baselines3.common.callbacks import BaseCallback as Sb3Callback

    class _Sb3(Sb3Callback):
        def _on_step(self) -> bool:
            return True

    model = type("M", (), {"num_timesteps": 5, "get_env": lambda self: None, "logger": None})()
    with pytest.raises(LC.CallbackProtocolError, match="loop_callbacks.BaseCallback"):
        LC.init_callback(model, [_Sb3()])
    with pytest.raises(LC.CallbackProtocolError):
        LC.init_callback(model, lambda _l, _g: True)
    log: List[str] = []
    root = LC.init_callback(model, [_Rec(log, "a", stop=True), _Rec(log, "b")])
    root.on_training_start()
    root.update_locals({"infos": [{}], "dones": np.array([True])})
    assert root.on_step() is False                                    # one child stopped ...
    assert log == ["a:[ True]", "b:[ True]"]                          # ... and every child still ran, in order
    assert [c.n_calls for c in root.callbacks] == [1, 1] and root.callbacks[0].num_timesteps == 5
    with pytest.raises(LC.CallbackProtocolError, match="not declared"):
        root.update_locals({"rollout_buffer": None})
    assert isinstance(LC.init_callback(model, None), LC.CallbackList)
