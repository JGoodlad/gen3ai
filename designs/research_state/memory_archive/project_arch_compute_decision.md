---
name: project_arch_compute_decision
description: "Architecture/compute decision (2026-06-09, 9-agent workflow + verified in code) — do NOT restructure the transformer to \"reclaim compute\"; it's not the bottleneck. History 10→7 only as a retrain free-rider. Real critic lever = tail-weighted value loss (no arch change)"
metadata: 
  node_type: memory
  type: project
  originSessionId: b7d3c1d7-0aa2-4875-9183-6e9491acdea9
---

> **Archived 2026-09-08** — ai_v5-era 23-token transformer decision; the entity-graph generation replaced that trunk — designs/ARCHITECTURE.md is the doc of record. Preserved verbatim; nothing below is current.

2026-06-09 architecture-redesign workflow (4 analysts → adversarial verify → synthesis) + per-turn
history-saliency measurement. Built prober `history-saliency` (per-turn-slot |grad| of the turn-history
block, both heads; `ProbeSession.history_saliency` / CLI). 92 prober tests.

**TOKEN STRUCTURE (TeamTransformer):** 23 tokens = 6 our + 6 their Pokémon role tokens + **10 history
tokens** (each = one TurnDelta turn, embedded→D_MODEL + a learned positional embedding) + 1 global token.
D_MODEL=128 (=ROLE_TOKEN_SIZE), 2 layers, 4 heads, FFN=256. Token-type embeddings (4). The 3 CLS pools
(our/their/value_cls) read ONLY the 12 team tokens post-transformer; history+global influence the heads
only via attention into the team tokens. ProjectionAssembler concats pooled team tokens + active-ctx +
`non_matchup_rest` → pi/vf_combined → 512-dim heads, NET_ARCH [512,512].

**HISTORY SALIENCY @70M (normalized, 1.0 = avg obs dim; slot 9 = MOST RECENT, slot 0 = oldest):**
monotonic decay — pol slot9=.62, slot8=.17, slots0-5 each <.08; val similar. The whole 1590-dim history
block reads like ~1.25 avg dims; the model uses ~the last 1-2 turns. → older history is near-dead weight.

**THE LOAD-BEARING CORRECTION — "reclaim compute via the transformer" is FALSE:** GPU ~86% IDLE; rollout
is CPU/latency-bound on OBS BUILD (~88% of trainer CPU, state_encoder.encode ~80%); the transformer
forward is sub-1-2% of CPU. So depth / funnel / smaller D_MODEL are ~FREE on compute = reclaim ~NOTHING;
each is a QUALITY decision, never throughput. The only resource they move is rollout-buffer RAM, which is
obs_dim-driven → ONLY the history cut touches it (~0.2-0.4 GB @ n_envs 48-64), and that is NOT the binding
memory constraint (the node-server RAM creep + env-worker leak is, see [[project_showdown_server_memory_growth]]).

**DECISIONS:**
- **N_HISTORY_TURNS 10→7: DO, but ONLY as a free-rider** in a retrain already happening (the crit-split
  A/B). N=7 keeps ~88-91% of history saliency, drops the dead tail. NOT a dedicated retrain. Instrument
  per-slot saliency + ELO vs an N=10 control. N=5 = SKIP standalone (76-82% retention band where a
  >5-turn dependence could hide under γ=0.9999, and a converged-policy ablation can't falsify it). Not N=3.
- **The "funnel" (mixed-scope: deep layers attend board-only): REJECT** — optimizes a non-bottleneck AND
  masking history from deep layers starves the final layer of context = quality regression for 0 compute.
- **Add layers (2→4): SKIP** — blind depth, the exact lever the [[project_plateau_diagnosis_2026_06_09]]
  refuted (EV 0.77 flat-healthy).
- **Shrink D_MODEL 128→96/64: SKIP** — saves non-bottleneck compute, doesn't touch obs_dim/RAM, and
  narrows value_pooled (the critic's only whole-board readout) while we're fixing critic tail-blindness.
- **CODE FACT (verified, corrects an earlier claim):** `vf_combined` ALREADY concats `non_matchup_rest`,
  which includes turns_since_progress (vec[14]) AND the full 33-dim incoming-damage belief — so the critic
  RECEIVES the attrition clock + OHKO belief directly. It's present-but-LOW-SALIENCY (under-used), NOT
  invisible. The fold-Toxic/PP-into-the-incoming-group idea = raise its saliency, still valid.

**REAL LEVERS (sequencing):** (1) tail-weighted value loss — upweight high-|TD|/near-terminal targets;
PPO-objective change, NO ARCH bump, no retrain-class risk, falsifiable against the stuck td_resid_tail=-10;
the honest "fix the critic tail" (not layers). (2) fold 10→7 free-rider into the crit-split A/B. (3) defer
any transformer-body change; only if a tail-weighted loss develops an EV ceiling is body capacity worth an
A/B (prefer 2→3 all-23). (4) highest ceiling = decision-time search (PPO+MCTS), orthogonal to topology.
NOT committed (awaiting /gen3ai-ship).
