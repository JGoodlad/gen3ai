"""The learner's CALLBACK PROTOCOL — ours (`gen3_owned_callbacks_v1`; `designs/endstate/design_own_ppo_loop.md` §3.3).

Until deletion pass U4 every training callback subclassed sb3's ``BaseCallback`` and the loop handed them
to sb3's ``CallbackList``. This module is that protocol, written for the loop that drives it
(`instrumented_ppo/loop.py` + the Rust collector), with the EVENTS declared:

=================  ===================================================  ==========================================
event              fired by                                             a callback overrides
=================  ===================================================  ==========================================
``init``           `init_callback(model, callback)` in ``_setup_learn``  ``_init_callback``
``training_start`` ``learn()``, once, before the first collection       ``_on_training_start``
``rollout_start``  the collector, at the top of each collection         ``_on_rollout_start``
``step``           the collector, once per ``n_envs`` trainee decisions ``_on_step`` (False stops training)
``rollout_end``    the collector, after the fill (the post-collect      ``_on_rollout_end``
                   window: the buffer is the update's)
``training_end``   ``learn()``, once, after the last update             ``_on_training_end``
=================  ===================================================  ==========================================

What a callback may read: ``self.model`` (the learner), ``self.training_env`` (its env), ``self.logger``
(the learner's logger — `train_logger`), ``self.n_calls`` (``step`` events so far), ``self.num_timesteps``
(the model's counter, refreshed at ``training_start`` and every ``step``) and, during ``step``,
``self.locals`` — exactly the keys in `STEP_LOCALS`, supplied by the collector through
``update_locals`` (an undeclared key is a typed refusal: the per-step surface is declared, not "whatever
the loop's frame held" as sb3's ``locals()`` was). ORDER: a `CallbackList` dispatches every event to its
children in list order; that order is a contract where two callbacks share the post-collect window, and
``main/train/callbacks.py`` (``build_callbacks``) is where it is declared.

Differences from sb3, each deliberate: ``on_training_start()`` takes no frame (sb3 passed ``learn()``'s
``locals()`` / ``globals()``; no callback read them), there is no ``globals``, an sb3 ``BaseCallback`` or a
bare function handed to the loop is REFUSED (`init_callback`) instead of wrapped, and there is no
progress bar.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The declared events, in the order one ``learn()`` fires them (``step`` repeats inside a rollout).
CALLBACK_EVENTS: Tuple[str, ...] = ("init", "training_start", "rollout_start", "step", "rollout_end", "training_end")

#: The per-step locals a collector supplies (``update_locals``): the vector step's per-env ``infos``
#: (``info["episode"]`` on a finished game, and its outcome keys) and ``dones``. Read by
#: ``SignalMetricsCallback``.
STEP_LOCALS: Tuple[str, ...] = ("infos", "dones")


class CallbackProtocolError(TypeError):
    """A callback, or a step's locals, outside the declared protocol (named)."""


class BaseCallback(ABC):
    """One training callback (module docstring)."""

    model: Any

    def __init__(self, verbose: int = 0):
        super().__init__()
        self.n_calls = 0
        self.num_timesteps = 0
        self.verbose = verbose
        self.locals: Dict[str, Any] = {}
        self.parent: Optional[BaseCallback] = None

    @property
    def training_env(self) -> Any:
        env = self.model.get_env()
        assert env is not None, ("`model.get_env()` returned None, you must initialize the model with an "
                                 "environment to use callbacks")
        return env

    @property
    def logger(self) -> Any:
        return self.model.logger

    # ---- init
    def init_callback(self, model: Any) -> None:
        self.model = model
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    # ---- training start / end
    def on_training_start(self) -> None:
        self.locals = {}
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    # ---- rollout
    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    def on_step(self) -> bool:
        """One ``step`` event; False stops training (the collector returns False, ``learn()`` breaks)."""
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps
        return self._on_step()

    @abstractmethod
    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    # ---- the step's locals
    def update_locals(self, locals_: Dict[str, Any]) -> None:
        bad = sorted(set(locals_) - set(STEP_LOCALS))
        if bad:
            raise CallbackProtocolError(f"step locals {bad} are not declared (loop_callbacks.STEP_LOCALS = "
                                        f"{STEP_LOCALS}); declare a new per-step key there first")
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        pass


class CallbackList(BaseCallback):
    """Children, dispatched in LIST ORDER for every event; ``step`` is False if any child's is (every child
    still runs)."""

    def __init__(self, callbacks: Sequence[BaseCallback]):
        super().__init__()
        if not isinstance(callbacks, list):
            raise CallbackProtocolError(f"CallbackList takes a list, not {type(callbacks).__name__}")
        for cb in callbacks:
            _require_ours(cb)
        self.callbacks: List[BaseCallback] = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.model)
            callback.parent = self.parent

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start()

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            continue_training = callback.on_step() and continue_training
        return continue_training

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.update_locals(locals_)


def _require_ours(cb: Any) -> None:
    if not isinstance(cb, BaseCallback):
        raise CallbackProtocolError(
            f"{type(cb).__module__}.{type(cb).__name__} is not a `agents.training.loop_callbacks.BaseCallback` "
            "— the learner's loop drives only the declared protocol (an sb3 callback or a bare function is "
            "refused, not wrapped)")


def init_callback(model: Any, callback: Any) -> BaseCallback:
    """The ``init`` event: ``None`` -> an empty `CallbackList`, a list -> a `CallbackList`, a `BaseCallback`
    as is; anything else REFUSES. Returns the root callback the loop drives."""
    if callback is None:
        callback = CallbackList([])
    elif isinstance(callback, list):
        callback = CallbackList(callback)
    _require_ours(callback)
    callback.init_callback(model)
    return callback
