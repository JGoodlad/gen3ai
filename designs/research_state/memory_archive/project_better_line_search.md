---
name: project_better_line_search
description: "Prober \"better-line\" search + serializeBattle clone-and-branch search-server — BUILT, NOT shipped"
metadata: 
  node_type: memory
  type: project
  originSessionId: 28bb26ec-61dd-4d71-b1d4-bf2614b7ed03
---

> **Archived 2026-09-08** — BUILT+shipped; superseded operationally by project_rust_search_driver and src/main/prober/CLAUDE.md. Preserved verbatim; nothing below is current.

2026-06-23 (worktree bridge-cse, NOT shipped — awaiting /gen3ai-ship). The user wanted to **find better lines / the ideal-or-better trajectory when reviewing games**, and OK'd building a new bridge to support searching. A research+design workflow (10 agents) verified the linchpin: Showdown's **`State.serializeBattle`/`deserializeBattle`** (`dist/sim/state.js:79,99`) round-trips a battle paused with both move requests open (deserialize rebuilds requests via `getRequests`; PRNG continuity from the live counter) at **~1.7 ms/clone — ~16× cheaper than warm `buildToTurn` ~26 ms, CONSTANT in depth**. The Node-spawn (~677 ms) is the real per-reroll cost; batching/cloning is the lever, NOT serializeBattle for one-ply (so `reroll_many` got the +9× lookahead win; serializeBattle is the lever ONLY for the multi-ply TREE).

User decisions (AskUserQuestion): build the **real serializeBattle bridge** (not the reroll_many stopgap) as high-quality infra; **faithful-conditional opponent** (RECORDED move at the divergence ply = the value_crn anchor, reloaded policy at interior plies); **depth configurable**, find a useful default empirically (default=2: ~4-5s, legible; depth1~2s, depth3~6s); **build it all, present once**.

WHAT WAS BUILT (all reuses the lookahead recipe — node-expand→materialize one-sided obs→score V):
- `utils/bridge/replay_kernels.js` — shared JS sim kernels lifted verbatim out of `replay_driver.js` (one impl, trusted reroll path byte-unchanged).
- `utils/bridge/search_driver.js` — WARM persistent search-server: `open_root`/`expand_many`, node-snapshot cache, returns per-ply SUFFIX chunks (a deserialized battle can't re-emit historical `|request|` lines → Python composes root-prefix + suffixes, the `reroll_many` shape). `recorded_exact` (root) for the value_crn anchor.
- `utils/bridge/search_session.py` — `SearchSession` (one Node proc/call, bg-drained queue, timeouts, context-managed).
- `agents/training/obs_materializer.py::infer_action_indices` — recovers a side's action-index history by inverting recorded choices (the OPP's history, for interior-opp obs; trace only records the trainee's). Verified: inverting the trainee's own choices reproduces `npz["actions"]` 66/66.
- `main/prober/better_line.py` — beam-over-the-critic, depth-configurable, CRN ("original") throughout, backup=max-over-our-continuations (opp=fixed conditional response), principal-variation. ROOT ply expands ALL candidates (fair full-depth rank); beam+top_k only interior. Fixed a desync: opp plays RECORDED at divergence for EVERY candidate (not just chosen) → opp_actions must extend for all depth-1 nodes.
- `main/prober/model.py` — `values_batch` + `_check_obs_dim` on batch paths.
- `session.better_line` + `query better-line` CLI + TUI **B-key** (contrastive trajectory: divergence / line / where-went-wrong; primes C to confirm). `--confirm-rollouts N` folds rollout-to-end vs the REAL opponent (the 3-tier: search by V, report ΔP(win), confirm by rollout).

FAITHFULNESS proven (a NEW path): `search_clone_parity_fuzz_test.py` — clone obs == `reroll_many` obs BIT-FOR-BIT + `recorded_exact` == recorded next obs (value_crn anchor) + depth-2 composes + encode_only_at==full-encode bit-for-bit (24 checks/3 battles). Full suite 3003 green. Builds on [[project_counterfactual_prober]] (lookahead=depth-1 instance) + [[project_battle_reconstruction]]. The 3-tier-eval / shallow-CRN-beam / faithful-conditional design came from the design workflow's synthesis.

PERF (06-24): profiled — bottleneck is OBS MATERIALIZATION (~58%), the serializeBattle clones are only ~2%. Three levers cut depth-2 ~1.5× (5.7s→3.7s): (1) ONE shared `replay_battle` feeds both anchor choice-map + opp `infer_action_indices` history (was 2 full replays); (2) per-node policy forwards BATCHED via `action_probs_batch` (0.67s→0.08s); (3) `materialize_decisions(encode_only_at={target})` encodes obs ONLY at the decision read, prefix is TRACK-ONLY — rides new `Gen3Player.track_decision` (tracking half of embed_battle, extracted byte-identical → live obs path unchanged, obs_roundtrip 566 bit-for-bit). Lever-3 obs equivalence pinned bit-for-bit in the parity fuzz. ALSO fixed a CI-surfacing bug the live demo caught (confirm read `ci` not `win_rate_ci`). The big remaining lever (resume poke-env state at the anchor vs deep-copy) NOT done — encode_only_at got most of it without state-forking risk.
