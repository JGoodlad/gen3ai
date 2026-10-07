"""THE STRICT CHECKPOINT LOAD (`gen3_strict_checkpoint_load_v1`) — one mechanism, every algorithm class.

THE BUG (P10 review, F4). sb3's ``BaseAlgorithm.load`` retries with ``exact_match=False`` whenever the
strict ``set_parameters`` error mentions ``pi_features_extractor`` and ``Missing key(s) in state_dict``
(its "SB3 < 1.7.0" patch). Our extractor is registered under three aliases, so ANY extractor key missing
from a checkpoint names it, and the retry loaded every other tensor and left the missing submodule at
FRESH INIT behind one warning. A checkpoint with ``alpha_head``'s keys deleted loaded "normally".

THE MECHANISM. :class:`StrictCheckpointLoad` overrides ``set_parameters`` so that it is ALWAYS strict —
a missing key and an unexpected key both raise — and answers a non-strict request (the retry) with
:class:`StrictLoadError` carrying the strict error. It is a MIXIN with no other behaviour, so it sits on
every class a checkpoint is loaded into:

  * the learner and every opponent / reader the trainer builds — ``OwnedLoop`` inherits it, so
    ``InstrumentedMaskablePPO`` and ``InferenceMaskablePPO`` (P10 follow-up B);
  * every OTHER reader of a checkpoint (the ladder session ``play.py``, the prober, the eval worker, the
    offline meters) — :class:`StrictMaskablePPO`, a plain ``MaskablePPO`` and nothing but the strict
    ``set_parameters``, built by ``agents.model.snapshot.load_checkpoint_strict`` (P10 follow-up F1).
    It is deliberately NOT ``InferenceMaskablePPO``: that class also drops the optimizer, the rollout
    buffer and the global reseed, and a caller that fits and saves (``winprob_finetune``) or compares
    with an old measurement (the eval worker's seeding) must keep a plain ``MaskablePPO``'s behaviour.

``src/strict_checkpoint_load_gate_test.py`` keeps the class closed: a bare ``MaskablePPO.load`` /
``PPO.load`` (any class imported from sb3 / sb3_contrib), an ``exact_match=False`` request, or a
subclass of an sb3 algorithm that does not carry this mixin FAILS, with an EMPTY allowlist.

No declared non-strict exception exists. A future one is declared HERE, with its reason — never a
silent fallback.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar

from sb3_contrib import MaskablePPO


class StrictLoadError(RuntimeError):
    """A checkpoint whose state dict does not match the model it loads into."""


class CudaContextOnCpuLoad(RuntimeError):
    """A checkpoint load asked for the CPU created a CUDA context (`gen3_cpu_load_no_cuda_v1`)."""


#: PROCESS-LOCAL training state that older checkpoints PICKLED into their ``data`` member and that no
#: load may unpickle (`gen3_cpu_load_no_cuda_v1`, 2026-10-06): the ride-along heads' optimizers and the
#: heads module copies they were bound to (`RideAlongTerms._ridealong_acquire`). Their tensors were
#: CUDA tensors on a GPU run, and sb3's ``json_to_data`` cloudpickle-loads every serialized attribute
#: onto the device it was saved from — whatever ``device`` the load asked for — so a READ-ONLY CPU load
#: (the prober, h2h, belief_roles, policy_spectrum, the meters) created a CUDA context (~100 MB of the
#: ~170 MB data member of an X5 A/B checkpoint). Nothing ever used them after a load: every
#: ``_setup_model`` re-acquires the optimizers (the learner) or sets them to None (an inference load), so
#: the heads' Adam state was never restored across a restart (X26 PREREGISTRATION "Overhead"). Saves stop
#: writing them (`hparams._excluded_save_params`); this list keeps the checkpoints that already carry
#: them loadable without a GPU.
NEVER_UNPICKLED = ("_ridealong_opt", "_ridealong_opt_owner", "_ridealong_vopts", "_ridealong_vopts_owner",
                   "_ridealong_variant_disabled")


def never_unpickled(custom_objects: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``custom_objects`` for sb3's ``load`` / ``load_from_zip_file`` with every `NEVER_UNPICKLED`
    attribute replaced by None (sb3 substitutes a custom object for a ``data`` key INSTEAD of
    unpickling it); the caller's own entries win."""
    out: Dict[str, Any] = {k: None for k in NEVER_UNPICKLED}
    out.update(custom_objects or {})
    return out


def _cpu_device(device: Any) -> bool:
    import torch as th

    try:
        return th.device("cpu" if device == "auto" and not th.cuda.is_available() else device).type == "cpu"
    except (RuntimeError, TypeError):
        return False


_LoadedT = TypeVar("_LoadedT", bound="StrictCheckpointLoad")


class StrictCheckpointLoad:
    """Mixin: ``set_parameters`` is ALWAYS strict (module docs), and ``load`` never unpickles the
    `NEVER_UNPICKLED` training state and never creates a CUDA context on a CPU load. Put it BEFORE the
    sb3 algorithm class."""

    @classmethod
    def load(cls: Type[_LoadedT], path: Any, env: Any = None, device: Any = "auto",
             custom_objects: Optional[Dict[str, Any]] = None, **kwargs: Any) -> _LoadedT:
        """sb3's ``load`` with `NEVER_UNPICKLED` substituted (`never_unpickled`), THROWING
        `CudaContextOnCpuLoad` when a load asked for the CPU initialised CUDA (`gen3_cpu_load_no_cuda_v1`):
        any other pickled CUDA object a future save adds is a typed error at the first CPU read, never a
        silent GPU context in an analysis process."""
        import torch as th

        cpu = _cpu_device(device)
        before = th.cuda.is_initialized()
        model: _LoadedT = super().load(path, env=env, device=device,   # type: ignore[misc]
                                       custom_objects=never_unpickled(custom_objects), **kwargs)
        if cpu and not before and th.cuda.is_initialized():
            raise CudaContextOnCpuLoad(
                f"loading {path!r} on the CPU created a CUDA context: an attribute of the checkpoint's data "
                "member is pickled CUDA state — exclude it from the save (`_excluded_save_params`) and add it "
                "to strict_load.NEVER_UNPICKLED (gen3_cpu_load_no_cuda_v1)")
        return model

    def set_parameters(self, load_path_or_dict: Any, exact_match: bool = True,
                       device: Any = "auto") -> None:
        """sb3's ``set_parameters``, ALWAYS STRICT (`gen3_strict_checkpoint_load_v1`): every module's
        state dict must match the checkpoint key for key — a MISSING key and an UNEXPECTED key both
        raise. ``exact_match=False`` is REFUSED rather than honoured.

        Why: sb3's ``BaseAlgorithm.load`` retries with ``exact_match=False`` whenever the strict error
        mentions ``pi_features_extractor`` (module docs), so a checkpoint missing an extractor
        submodule's keys loaded with that submodule at FRESH INIT.

        No caller passes ``exact_match=False`` (survey 2026-10-03; the static gate refuses one), and the
        deleted-kwarg sanitizers in ``snapshot`` pop only kwargs that build no parameters and REFUSE the
        rest, so they never need a key dropped."""
        if exact_match:
            super().set_parameters(load_path_or_dict, exact_match=True, device=device)  # type: ignore[misc]
            return
        try:
            super().set_parameters(load_path_or_dict, exact_match=True, device=device)  # type: ignore[misc]
        except (RuntimeError, ValueError) as exc:
            # the strict error FIRST: a reader that truncates the message (the prober's drift diagnosis
            # keeps ~180 characters) still sees which keys differ
            raise StrictLoadError(
                f"the checkpoint's state dict does not match this model's (strict load): {exc} -- a "
                "NON-STRICT load was requested and REFUSED (gen3_strict_checkpoint_load_v1): it would "
                "leave every missing tensor at FRESH INIT (sb3's 'SB3 < 1.7.0' retry fires on any missing "
                "extractor key)") from exc


class StrictMaskablePPO(StrictCheckpointLoad, MaskablePPO):
    """A plain ``MaskablePPO`` whose checkpoint load is strict, and nothing else: the class every
    reader of a checkpoint builds through ``agents.model.snapshot.load_checkpoint_strict``."""


__all__ = ["CudaContextOnCpuLoad", "NEVER_UNPICKLED", "StrictCheckpointLoad", "StrictLoadError", "StrictMaskablePPO",
           "never_unpickled"]
