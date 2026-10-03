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

from typing import Any

from sb3_contrib import MaskablePPO


class StrictLoadError(RuntimeError):
    """A checkpoint whose state dict does not match the model it loads into."""


class StrictCheckpointLoad:
    """Mixin: ``set_parameters`` is ALWAYS strict (module docs). Put it BEFORE the sb3 algorithm class."""

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


__all__ = ["StrictCheckpointLoad", "StrictLoadError", "StrictMaskablePPO"]
