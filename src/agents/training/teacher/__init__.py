"""Search-as-teacher (selective Expert Iteration) — the offline-teacher plateau-breaker.

Distils VERIFIED-better search corrections (from the prober's depth-limited beam search +
rollout-confirm tiers) into the policy via an **advantage-weighted CE aux loss** (AWR), applied to a
few high-leverage turns of recent eval-trace losses under a compute budget — NOT every turn.

The signal is the POLICY (what should it have done), NOT the value: the search's value is the
*improved-policy* value V^π*(s), so regressing the PPO critic (which must predict V^π for GAE) toward
it would bias advantages — the value term is therefore optional and off by default. The AWR weight is
the **confirmed win-rate improvement** of the better action vs the EXACT reloaded opponent (the
strictly-better CI gate), never a critic advantage.

Design: ``designs/ai_v6/design_search_teacher.md``. Off (``--search-teacher`` absent / coef 0) is
byte-identical to stock PPO. Training-only (not version-locked).
"""
