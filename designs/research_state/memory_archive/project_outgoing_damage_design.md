---
name: project_outgoing_damage_design
description: "Outgoing-KO obs feature — BUILT (gen3_outgoing_ko_v1, obs 3451) after the owner OVERRODE the Gate-0 falsifier NO-GO; Gate 6 (post-retrain POLICY-mix movement) is the decisive test"
metadata: 
  node_type: memory
  type: project
  originSessionId: 256c5a69-2b5f-47eb-9436-f6a301d8b0f2
---

> **Archived 2026-09-08** — ai_v5-era obs block; era closed. Preserved verbatim; nothing below is current.

**BUILT 2026-06-10 (branch `claude/outgoing-ko`, ships via /gen3ai-ship), owner OVERRODE Gate-0.**
ARCH `gen3_outgoing_ko_v1`, obs **3409 → 3451**: 42-dim block at reactive offset 102 (abs 1520), per
opp slot `[phys_exp, spec_exp, phys_pko_nocrit, spec_pko_nocrit, phys_crit_delta, spec_crit_delta,
defender_known]` — the crit-split mirror, uncertainty on the DEFENDER (bulky tail Q=0.85,
deliberately below incoming's 0.95). As-built: core renamed `incoming_damage.py → damage_belief.py`
(side-neutral `StatBelief(mean, conservative)`; defender ability dist marginalised at the P(KO)
level; **honest mean-exp replaced the ratio shortcut → incoming expdmg values shifted within
rounding**, golden regenerated digest 7e593a7a…, 991 dec); `gen3_hp_stat` (2·base+31+ev//4+110) +
`stat_distribution` widened to def/spd/hp; our own Hidden Power resolved EXACTLY from IVs
(`gen3_hidden_power`, pinned vs GEN3_HP_IVS — type AND 30-70 power). Gates: suite 2159; outgoing
fuzz 4391 dec / 0 violations / 100% fire-rate; benchmark **+5.1%** (7249→7620 calls/encode); golden
parity byte-exact; roundtrip + bridge smoke green (use `--device cpu` when the training run holds
the GPU). Also fixed a LATENT main bug: `incoming_damage_fuzz_test.py` still unpacked 5/mon
post-crit-split. Prober pins: om 1562, tm 1706, outgoing 1520/42/7; incoming 1469/51/8 unchanged.
**Gate 6 is the decisive test**: the bar is POLICY-side movement (futile attacks + passed KOs fall,
π-head saliency on `outgoing_damage(42)`) — a miss confirms the falsifier and the lane should be
reconsidered. Deferred to Phase 2: Gate-1 calibration oracle, joint-spread bulk belief, prober
OutgoingBeliefView decode. Docs: `impl_step9_outgoing_ko_obs.md` + reconstructed
`design_outgoing_damage_obs.md` (BUILT banner).

--- (history: the Gate-0 NO-GO record below — the override does NOT erase the evidence) ---

Original design: symmetric mirror of the shipped INCOMING-damage block
([[project_loss_analysis_run20260601]]): for OUR active vs each opp mon, a calibrated
`[phys_expdmg, spec_expdmg, phys_pko, spec_pko]` belief — "can WE KO
the opp?". (Draft-era numbers, superseded by the as-built above: obs **3390 → 3414** (+24 = 6 opp slots ×
4), new block at reactive offset 83 (between incoming-damage and the matchups → both heads via
`non_matchup_rest`).

**Status (2026-06-10): GATE-0 FALSIFIER = NO-GO — build STOPPED per the gate.** The authoritative
design moved to `designs/ai_v5/design_offense_and_opponent_belief.md` Part A (committed on main;
superseded the standalone draft). The Phase-0 falsifier
(`designs/ai_v5/falsifier_missed_ko_attribution.py`, in worktree branch `claude/outgoing-ko`) ran on
the ai_v5_6_stable_70m_0608 plateau corpus (2780 loss / 1500 win battles) with TWO instruments — the
spec'd power×eff proxy AND a REAL computed belief (gen3 damage core + spread priors + obs eff
scalars = a faithful offline prototype of the feature). Both agree:
- LOSS decisive-TP addressable (A_PASSED_KO+A2_WRONG_MOVE+B_FUTILE+C_SELF_KO) = **10.1%** (robust
  9–11%) vs the incoming falsifier's 56% analog; WIN control 9.4% → **no discrimination**.
- Decision-level (erosive frame): LOSS 8.4%/dec vs WIN 10.4%/dec → **inverted**.
- With a ≥85% kill available the policy ALREADY takes a kill move **67% (loss) vs 68% (win)** —
  identical; KO-taking does not separate outcomes at 70M.
- The decisive loss cliffs are ~90% D: incoming surprises while attacking REASONABLY, post-faint
  repricing, recovery/stall — i.e. Part B opponent-blindness / critic-tail territory.
**This OVERTURNS the frontier-roadmap ranking of outgoing-KO as "build first"** (see
[[project_model_frontier_roadmap]]) — the arch-review argument (ReLU can't multiply) remains true in
principle but the measured decision evidence says offense-blindness is NOT a binding loss cause on
the plateau corpus. Cheap re-check: re-run the falsifier on ai_v5_8 crit-split traces when they
exist (`FALSIFIER_RUN=...`). Energy goes to Part B (B1 unmask unrevealed → team belief) instead.

**Owner-resolved decisions (2026-06-06 review), folded into the doc §5/§8:**
- (1) **RENAME** `incoming_damage.py → damage_belief.py` (clarity wins; touches constants/encoder/tests/docs).
- (2) **Honest mean-exp unification, NO approximate-ratio fallback** — correctness is the core constraint;
  buy perf with caching, never a knowingly-less-correct path. (Perturbs incoming exp within rounding →
  re-validate fuzz + golden.)
- (3) **`_CONSERVATIVE_TAIL_Q = 0.85`, but built MODULAR**: `_stat_belief(species, stat, q)` is
  percentile-parametrised and returns `dist`, so adding a P50/median summary (or a 2nd median-case P(KO)
  channel) later is additive, not a refactor.
- (4) **Attacker uses the full known `active.moves` (capability set)** — "if we have the info, use it"
  (we know our IV/EV/nature/moves exactly); the mask handles per-turn legality.

**Key design decisions (in the doc):**
- **Uncertainty flips to the DEFENDER.** Our offense is exact (our stats/moves/STAB, P=1, no priors);
  the opp's def/spd/HP/ability are hidden → distributions. Tail direction flips: conservative = the
  **bulky** P85 tail (don't claim a KO a max-bulk spread survives); expected-dmg at the mean.
- **ONE side-neutral core, two thin encoders** (the win condition). Generalise the core: add
  `StatBelief(mean, conservative)` per uncertain stat (exact side sets mean==conservative) + a defender
  `ability_dist` marginalised at the P(KO)/exp level. Reuse `gen3_damage_max`/`p_ko`/`percentile`/
  `weighted_mean`/`FIXED_DAMAGE`/Explosion-Def-halve unchanged.
- **Almost no new data:** `gen3_spread_priors.json` already carries def/spd/hp EVs — only extend
  `priors.stat_distribution` to def/spd/hp + add `gen3_hp_stat` (HP formula `2*base+31+ev//4+110`, no
  nature). Opp HP is a FRACTION (poke-env `stats["hp"]=100` for opp); absolute HP = fraction × HP belief.
- **Dropped (justified):** p_outspeed (vs opp active already in incoming; vs bench meaningless), recovery
  scalars (our recovery is exact, already in move-effects). Unrevealed/fainted opp slot → zero;
  opp-active-behind-Sub → zero its pko.

**Why it's worth doing:** ReLU body can't multiply; we precompute the opp's offense (incoming) but not our
own → the net re-derives `power·A/Def·eff·STAB·roll≥HP` from raw linear inputs. Directly serves documented
pathologies (FUTILE_ATTACK ~12, passed-lethal-KO, SELF_KO-into-immune, walked-into-wall). Stronger than
incoming on the policy-side caveat: "I have a KO / this is futile" speaks to the ARGMAX, not just the critic.

Verdict in doc: **PRELIMINARY-GO** (medium-high) pending the Phase-0 falsifier
(`falsifier_missed_ko_attribution.py`, model-free, zero-build — decodes `our_matchups` + move powers +
opp HP from the existing run_20260601 loss corpus; on that arch our_matchups @ 1468). See
[[reference_wang2024_thesis]] for the MCTS lever (complementary, ai_v6).
