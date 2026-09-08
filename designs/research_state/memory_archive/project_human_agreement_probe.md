---
name: project_human_agreement_probe
description: ai_v6 BC track — human-agreement probe built (replays→obs→policy match); v0 number is fidelity-confounded; spectator-log reconstruction ceiling is the real finding
metadata: 
  node_type: memory
  type: project
  originSessionId: 5295b510-4f56-4871-9eec-0c9ecfba2aff
---

Built 2026-06-12 (branch claude/trusting-cannon-9610be, NOT shipped) the **human-agreement
probe** — the first concrete slice of the ai_v6 BC track ([[project_model_frontier_roadmap]]
names "self-play treadmill" as a structural ceiling; human data is the external-distribution
answer). Purpose = *measure before you build* BC: does the policy already play like the ladder?

**What exists:** ~102k real gen3ou ladder replays at `replays/showdown/gen3ou/<date>/` (ai_v6
Step 1 `collect_replays.py`, a LIVE daemon; 98% carry both Elos but ladder is weak — median
1231, only 9%≥1500 / 3%≥1600 / ~0%≥1700). ai_v6 BC design is fully written
(`designs/ai_v6/impl_step2_bc.md`) but UNBUILT; `team_completion_model.py` exists but orphaned.

**Built:** `src/agents/bc/log_reader.py` (`SpectatorLogReader`) replays a spectator `.log` →
per-decision `(obs, human_action, mask)` by SYNTHESISING a `LegalActions` from revealed state
and calling the live `embed_battle` sequence (encoder+tracker+turn-history) unchanged — the obs
seam is `encode(..., legal=)`, the ONLY request dependency. `src/main/human_agreement.py` =
CLI aggregator (loads ckpt via `ProbeModel`, picks rating≥N sides, agreement breakdowns).
6 unit tests `log_reader_test.py`. The probe and BC SHARE this exact pipeline, so building it
validates BC's Phase-1 substrate.

**FIDELITY-V2 BUILT same day (the foundation):** `_prescan_team` collects our side's full
revealed team+movesets across the whole game; at turn 1 (nothing damaged yet) inject ONE
synthesised `|request|` via `battle.parse_request` (poke-env's native path) → creates the full
bench + adds every known moveset. Obs own-team block, move mask, switch options all FAITHFUL
(opp stays correctly partial). Switch-unrepresentable count 623→0. Residual gaps (unrecoverable
from spectator, hit BC too): own EV/IV/nature spreads, moves never used all game, unrevealed
abilities/items, no PP. `ProbeModel.action_probs_batch` added (one forward/replay). Probe is
CPU-bound on the policy forward → cap `--threads` to spare the live training run.

**FAITHFUL result (ai_v5_11 best_model, 80 replays / 114 sides≥1500 / 2874 decisions, 0
exclusions):** overall agreement 35.0% (mean p_human 0.31), move 45.3%, switch 10.9%. move-by-
#legal: 4-legal=26%, 3=33%, 2=49%, 1=90%(forced, now only 17% of moves). Declines by turn
(49%→27%) and rating (35.7%@1500→32.4%@1700+).

**THE FINDING (trustworthy, and it OVERTURNS v0): the policy UNDER-SWITCHES massively vs strong
humans — human_switch_rate 29.9% vs policy 16.5% (same states), policy switch-prob 0.28.** v0's
"modest 16 vs 19" was an ARTIFACT: 43% of human switches (to unrevealed mons) were excluded, so
v0 undercounted human switching. The fidelity fix removed that bias → strong humans switch ~2×
as often as our policy. This QUANTIFIES the long-suspected under-switching lever against actual
strong humans (research_state frontier "Under-switching policy lever"; pairs w/ existing
`--switch-bias-weight`). Overall ~35% (≈1-in-3) match also says real human-distribution signal
self-play misses (supports the external-data / treadmill-break thesis) — BUT low agreement ≠ bad
(two strong humans also disagree often; agreement = behavioural DISTANCE, not skill). The value
is the LOCALISED gap (switching), not the absolute.

**Foundation framing (user's steer = build for long-run non-MCTS inference):** the faithful
`log_reader` is the shared substrate for the whole non-search roadmap — BC warm-start, the
opponent-action/forward-model aux head (the log has BOTH sides' actions → free supervised
"what will opp do" labels = roadmap's "shared scaffold"), team-completion, offline human-eval.
Alt safe lever: ladder-team-sampling (uses only extracted TEAMS, no reconstruction). See
[[feedback_research_state]], [[project_loss_triage_tool]], [[project_model_frontier_roadmap]].

**UPDATE 2026-06-21 (ai_v6_13 re-probe — UNDER-SWITCHING IS RESOLVED, BC's headline rationale is DEAD).**
Re-ran `human_agreement.py` on `models/ai_v6_13_outgoing_dmg_0620/best_model` (400 replays ≥1600,
13,359 decisions, fidelity excellent: `excluded_switch_to_unrevealed=0` — the 43% exclusion fear is
STALE, the turn-1 full-team injection fixed it). Result: **human_switch_rate 28.0% vs policy 31.4%
(argmax) / 37.6% soft mass** — the model now switches *MORE* than strong humans (mildly reversed),
NOT the 16.5%-vs-29.9% under-switching of ai_v5_11. So the named pathology that motivated the whole
BC todo is **gone** (consistent with [[project_positional_grind_decomposition]] "under-switching
OUTGROWN at 0.30≈human"); the `--switch-bias-weight`/temperature fix is moot too. Remaining policy↔human
divergence is now MOVE-SELECTION: faithful 4-moves-revealed agreement only 30.4% (mass 0.26), worst
EARLY-game (t1-5: 33.6% vs t21+: 37.9%) and worst vs the STRONGEST humans (1700+: 34.7% < 1600s: 36.9%).
But agreement≠better (can't tell different-equally-good from worse) → a human prior risks CAPPING a
34M-step self-play model. So BC is now a WEAK bet: revised honest menu = (1) DIAGNOSTIC FIRST: bridge-roll
the elite human's move vs the policy's move at high-divergence 4-move states — the ONLY honest go/no-go
for whether BC has move headroom (if humans don't win more → BC falsified → league/exploiter); (2) best
actual BC build = a human-like PINNED league opponent (PFSP just shipped oversamples it; exposes early-game
scouting lines; doesn't touch the main policy → no ceiling risk; needs `add_from_path` pinned-kwarg patch);
(3) policy-head BC warm-start only on a FRESH lineage as convergence-speed, not a pathology fix; (4) the
KL-to-human anchor — skip unless (1) proves human moves win. Pairs w/ [[project_pfsp_phase1_built]].
