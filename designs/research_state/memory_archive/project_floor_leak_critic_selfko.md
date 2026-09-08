---
name: project-floor-leak-critic-selfko
description: "The model loses ~18% to bots a human beats 100% because the CRITIC over-values self-destruction (Explosion/full-HP heal); reward is correct, NOT ①. Floor leak = critic credit-assignment, not a frontier lever."
metadata: 
  node_type: memory
  type: project
  originSessionId: 6d3cc0b1-c1f5-48ff-b361-e28b6d0f519f
---

> **Archived 2026-09-08** — ai_v5-era critic forensics; era closed, critic replaced by the win-prob head (ai_v12). Preserved verbatim; nothing below is current.

**The floor leak (found 2026-06-12 by grounding in concrete bot losses, ai_v5_11_tail2_N_0611).** A human beats the scripted bots ~100%; the model wins only ~82%. The losses are human-obvious blunders, NOT hard positional play:

- **41% of all Explosion/Self-Destruct selections are on a HEALTHY mon (HP≥80)** — incl. turn-1 full-HP Metagross. ~38% healthy across runs. Falsifier-confirmed as +1–1.7 mon material MISTAKES.
- **13% of recovery moves at near-full HP** (incl. Blissey Soft-Boiled at 100% for 7 turns = the "PP-stall heavy tail" — a real blunder, NOT the reward/horizon artifact I first dismissed it as).
- Policy assigns the explosion **median P≈0.5** (73% are confident preferences, only 9% exploration tail) → a LEARNED preference, not entropy noise.

**Mechanism (verified, ruling out 2 wrong hypotheses):** reward CORRECTLY punishes healthy explosion (−2.67; the finishing_blow mis-credit is ALREADY guarded in `reward_manager.py:1199-1222` `_compute_finishing_blow_bonus`, suppressed when our mon faints). Exploration ruled out (P≈0.5). The **CRITIC over-values the post-self-KO state: dV ≈ +2.5 even on NON-trades where our mon died for nothing**, which neutralizes the −2.67 reward in the PPO advantage (δ≈−0.1) → policy never un-learns. A GENERAL critic credit-assignment failure: ai_v5_9 (NO `value_active_readout`/①) explodes 16.5% when-available vs ai_v5_11 (WITH ①) 13.7% → **① is NOT the cause** (exonerated; my "ai_v5_9=0 explosions" was a script bug — its older npz lacks logits/actions). Consistent with the under-spread critic (std V 10.9 vs R 28.2) not tracking team material.

**Open / next falsifiable step:** does V track material at all? Regress V vs (our_alive − opp_alive). Flat → confirmed material-blindness → fix = value-head/obs material weighting (architecture-LIGHT) → predicted to lift win_rate_vs_bots 82%→~95%. This is the FLOOR, beats every frontier lever we chased.

**META-lesson (the real answer to "how we keep getting it wrong"):** 3 confident hypotheses overturned in one session by a more careful check — (1) heavy-tail→distributional critic = outcome-conditioning artifact [[project_distributional_critic_verdict]]; (2) reward-rewards-Explosion = already fixed; (3) ①-causes-it = script artifact. Pattern: hypothesis → measurement that APPEARS to confirm → the confirmation is an artifact (selection bias / stale-data bug / confound). Discipline: distrust the first confirming result, always run the disambiguating check. And: we chased frontier architecture (distributional/forward-model/I2A) while the floor leaked. See [[project_model_frontier_roadmap]], [[project_anti_stall_fix]].
