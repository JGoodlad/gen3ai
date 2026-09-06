"""The masked forward KL, re-implemented for the v8 era ONLY.

The v8-era checkout (`b13b30b2`) predates `agents.training.instrumented_ppo.distill_anchor`, so the
era arm cannot import `masked_kl_rows`. This file is the era's copy. It is DELIBERATELY a verbatim
transcription of the gen-era function rather than an independent derivation, and
`kl_unit_test.py` -- which runs in the GEN tree, where BOTH are importable -- fails if the two ever
disagree on a synthetic batch.

Do not use this in the gen era. There, import the real one.
"""
import torch as th
import torch.nn.functional as F


def masked_kl_rows_era(p_logits, q_logits, action_mask):
    """Per-row forward ``KL(p || q)`` over the LEGAL actions — ``[B]``.

    Illegal logits go to -inf on BOTH sides first, so both distributions normalise over the same
    legal set and illegal actions contribute exactly 0.
    """
    neg = (action_mask.to(q_logits.dtype) - 1.0) * 1e9
    logq = F.log_softmax(q_logits + neg, dim=-1)
    p = F.softmax(p_logits + neg, dim=-1)
    return (p * (th.log(p.clamp_min(1e-9)) - logq)).sum(-1)
