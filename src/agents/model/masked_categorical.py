"""FUNCTIONAL MASKING — sb3's `MaskableCategorical`, as plain tensor ops (M5 Lane K8, `gen3_functional_masking_v1`).

WHY. `sb3_contrib`'s `MaskableCategorical.apply_masking` re-runs `torch.distributions.Categorical.
__init__` after `self.__dict__.pop("probs")` — an object mutation dynamo cannot trace ("call_method
… `__dict__` pop" on torch 2.5.1, "Unsupported method call" on 2.8; 17–19 graph breaks per update
in the K8 inventory, `designs/research_state/measurements/k8_inventory/`) — and every distribution
built with torch's default `validate_args` checks its logits and every sample with a host read. The
learner's micro-step cannot be ONE `fullgraph=True` region while the masked distribution is an
object. This module is the same arithmetic with no object and no validation.

BIT-IDENTICAL BY CONSTRUCTION — the exact op sequence sb3 + torch run, spelled out:

    MaskableCategorical(logits=x)        n1 = x - x.logsumexp(-1, keepdim=True)        (torch Categorical)
                                         _original_logits = n1
      .apply_masking(None) inside init   n2 = n1 - n1.logsumexp(-1, keepdim=True)      (re-init on n1)
    .apply_masking(masks)                m  = where(masks, n1, -1e8)                   (sb3: from _original_logits)
                                         n3 = m - m.logsumexp(-1, keepdim=True)        (re-init on m)
    .log_prob(a)                         n3.gather(-1, a.long().unsqueeze(-1)).squeeze(-1)
    .entropy()  (masks given)            -(where(masks, n3 * softmax(n3), 0.0)).sum(-1)
    .entropy()  (no masks, torch)        -(clamp(n2, finfo.min) * softmax(n2)).sum(-1)
    .probs                               softmax(n)  (`logits_to_probs`)
    .sample()                            multinomial(probs.reshape(-1, A), 1, True).T.reshape(B)
    mode (deterministic)                 argmax(probs, dim=1)

`masked_categorical_test.py` pins every line against sb3's own objects (bit-for-bit, both torches),
including the unmasked double normalisation and the RNG stream of `sample`. What is NOT reproduced:
torch's argument/sample VALIDATION (`validate_args` — a host read per call, never a numeric change).
"""
from __future__ import annotations

from typing import Any, Optional

import torch

#: sb3's masked-logit floor (`MaskableCategorical.apply_masking`'s `HUGE_NEG`).
HUGE_NEG = -1e8


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x - x.logsumexp(dim=-1, keepdim=True)


def masked_logits(action_logits: torch.Tensor, masks: Any, n_actions: int) -> torch.Tensor:
    """The masked, normalised log-probabilities sb3's distribution ends with (`n3`, or `n2` when
    ``masks`` is None). ``action_logits`` is reshaped ``[-1, n_actions]`` exactly as
    `MaskableCategoricalDistribution.proba_distribution` does; ``masks`` may be a bool / float
    tensor or an array (True / non-zero = legal)."""
    n1 = _normalize(action_logits.view(-1, n_actions))
    if masks is None:
        return _normalize(n1)
    m = torch.as_tensor(masks, dtype=torch.bool, device=n1.device).reshape(n1.shape)
    huge_neg = torch.tensor(HUGE_NEG, dtype=n1.dtype, device=n1.device)
    return _normalize(torch.where(m, n1, huge_neg))


def mask_bool(masks: Any, like: torch.Tensor) -> Optional[torch.Tensor]:
    """``masks`` as the bool tensor sb3 stores (``None`` stays ``None``)."""
    if masks is None:
        return None
    return torch.as_tensor(masks, dtype=torch.bool, device=like.device).reshape(like.shape)


def log_prob(logp: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """`Categorical.log_prob` on the normalised logits (no sample validation)."""
    return logp.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)


def probs(logp: torch.Tensor) -> torch.Tensor:
    """`logits_to_probs` — ``softmax`` over the action axis."""
    return torch.nn.functional.softmax(logp, dim=-1)


def entropy(logp: torch.Tensor, masks_bool: Optional[torch.Tensor]) -> torch.Tensor:
    """sb3's masked entropy (masks given) or torch's `Categorical.entropy` (no masks)."""
    if masks_bool is None:
        min_real = torch.finfo(logp.dtype).min
        return -(torch.clamp(logp, min=min_real) * probs(logp)).sum(-1)
    p_log_p = logp * probs(logp)
    p_log_p = torch.where(masks_bool, p_log_p, torch.tensor(0.0, device=logp.device))
    return -p_log_p.sum(-1)


def sample(logp: torch.Tensor) -> torch.Tensor:
    """`Categorical.sample()` (the same `multinomial` call, so the same RNG draw)."""
    probs_2d = probs(logp).reshape(-1, logp.shape[-1])
    return torch.multinomial(probs_2d, 1, True).T.reshape(logp.shape[:-1])


def mode(logp: torch.Tensor) -> torch.Tensor:
    """sb3's deterministic action: ``argmax(probs, dim=1)``."""
    return torch.argmax(probs(logp), dim=1)


class MaskedPi:
    """The masked policy as TENSORS, for the readers of the old distribution stash
    (``policy._last_pi_distribution``: the ride-along heads),
    which read ``.distribution.logits`` / ``.distribution.probs``. ``distribution`` is the object
    itself, so both spellings resolve. Built OUTSIDE any compiled region."""

    def __init__(self, logp: torch.Tensor, masks_bool: Optional[torch.Tensor]) -> None:
        self.logits = logp
        self.masks = masks_bool
        self._probs: Optional[torch.Tensor] = None

    @property
    def distribution(self) -> "MaskedPi":
        return self

    @property
    def probs(self) -> torch.Tensor:
        if self._probs is None:
            self._probs = probs(self.logits)
        return self._probs

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return log_prob(self.logits, actions)

    def entropy(self) -> torch.Tensor:
        return entropy(self.logits, self.masks)
