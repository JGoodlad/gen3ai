---
name: project_refresh_status_cure_gap
description: 2026-06-14 prober-verified — policy does NOT understand Refresh cures status (obs gap); fix = per-move cures_status feature
metadata: 
  node_type: memory
  type: project
  originSessionId: ecffe614-2730-4139-b9f2-b5932b18b063
---

> **Archived 2026-09-08** — ai_v5/v6-era obs bits, shipped 92cf1ca/4ce3c72; era closed. Preserved verbatim; nothing below is current.

Verified on **ai_v6_01_belief_53m_0613** (prober, ckpt 14002863): the policy **does NOT
understand that Refresh cures status** — confirming the obs gap. Milotic (Refresh+Recover) is the
test case; it Recovers through Toxic and lets it stack, observed ~5% Refresh use.

**Model-free split (n=1173 TOX milotic+refresh decisions):** recorded mean P(refresh)=0.141, CHOSE
refresh **1.4%** vs CHOSE recover **57.5%**. P(refresh) is **flat ~0.14–0.18 across ALL status
conditions** (no-status=0.160 and CHOSE refresh **19%** — i.e. it uses Refresh MORE when there's
nothing to cure). P(refresh) does **not rise** with the toxic counter: ctr~1→ctr5+ stays ~0.13
while recover climbs to **96%** at ctr5+.

**Causal intervention (toggle the status one-hot + toxic-counter + reactive active-status flag,
re-forward):** removing the Toxic status moves **P(recover) −11pp and switch-mass −11pp** but
**P(refresh) only −1.5pp** (n=1173). Reverse (add tox to no-status decisions, n=405): P(refresh)
0.158→**0.099** (goes DOWN). ⇒ status IS visible to the head (it drives Recover/switch strongly),
but **Refresh is decoupled from it** — the model learned "statused → heal/run," NOT "→ cure."

**Mechanism:** the action-aligned move-effect block (`reactive.py`, 9 feats `gen3_move_effects_v1`)
has **no cures_status feature**. Refresh reads as base_power 0, neutral mult, ALL effect flags 0
(`gen3_moves.json`: isHeal:false, status:null) — indistinguishable from an inert move at the policy
head. (Refresh in gen3 is a PURE status cure, no HP.) Same GIGO class as [[project_model_frontier_roadmap]]'s
move_slot_align.

**Fix BUILT 2026-06-14 (branch worktree-bridge-cse, NOT shipped/run — `gen3_status_cure_moves_v1`,
obs 3409→3417):** added TWO static per-move bits to the action-aligned move-effect block (user chose
static-fact-let-it-learn over my live-resolved cures_current_status) — **`cures_self_status`**
(Refresh) + **`cures_team_status`** (Heal Bell / Aromatherapy, party-scoped so the model values it
off BENCH statuses). `MOVE_EFFECT_FEATURES` 9→11 cascades all reactive offsets via named constants.
Cure lives in an onHit callback (invisible declaratively) → curated override in the acquisition tool
`_CURES_SELF_STATUS`/`_CURES_TEAM_STATUS` (like Belly Drum); regenerated `gen3_moves.json`
(only-additions diff-checked). Rest EXCLUDED (heal+sleep, not a clear). Files: sync.py, gen3_data/moves.py,
constants.py, reactive.py (eff[9]/eff[10]), model_version.py ARCH bump, golden_obs_fixture regen (3417),
+ moves_test cure coverage + pinned-offset/forensics-dim test updates + 4 CLAUDE.md. Verified: 947 unit +
obs-roundtrip fuzz (2992 decisions bit-identical) + move-alignment fuzz (1937, 0 fails) + obs benchmark
(7196 calls/encode vs 7269 baseline, no regression) + debug smoke (Round-trip PASSED, pi+vf (1,512)) +
87 worktree focused. HONESTY GATE: wired+learnable but UNMEASURED if it HELPS → fresh-run A/B (Refresh
use vs status ↑ AND toxic-stack craters ↓; risk = learnable-but-inconsequential, the incoming-belief
precedent). GOTCHA: first edited MAIN checkout by mistake ([[feedback_edit_in_worktree_path]]) — caught
via git status, patch-moved to worktree (ff'd to main e1b09b3), main restored clean.

**Optional complement (NOT built):** a telescoping residual-status PBRS potential ranking cure>switch.
Obs feature is the primary lever; PBRS gives gradient the head still needs a feature to key on.
**PROBER GAP (NOT built):** no "available-move X + active-status Y" filter and no arbitrary obs-feature
toggle intervention (only the move-mult sweep `_intervention_sweep`) — both worth adding to engine/CLI.

**Sleep WAKE belief BUILT 2026-06-14 (same branch/unshipped, `gen3_sleep_wake_belief_v1`, obs
3417→3453):** user said "do BOTH (source bit + P(wake)), don't make it learn edge cases, research the
rates + FUZZ to confirm." 3-dim per-mon block in the slot (`POKEMON_VECTOR_DIM` 106→109):
`[sleep_is_deterministic (Rest), p_wake (COMPUTED), sleep_counter_reliable]`, zeros unless asleep. RATES
(ultracode research+adversarial-verify workflow, re-simulated bit-for-bit + Bulbapedia cross-check): opp
`time=random(2,6)`={2,3,4,5}, Rest `time=3`, Early Bird halves (d=2). **P(wake|counter K)** opp-noEB=
[0,¼,⅓,½,1], opp-EB=[¼,⅔,1], rest-noEB=[0,0,1], rest-EB=[0,1]; closed form P(T=K+1)/P(T≥K+1),
c_wake=ceil(T/d). Opp Early-Bird MARGINALISED over Smogon prior (collapses 0/1 if revealed/own). Source
(Rest) read from EVENT LOG `[from]` (poke-env discards it). KEY edge case: Sleep Talk/Snore ticks
poke-env counter **+3** (empirically verified) → `sleep_counter_reliable`=0 instead of reconstructing
Showdown skippedTime switch-refund (user's "kick the can"; verifier: not worth it, opp's true time hidden
anyway). Module `observation/sleep_belief.py` (pure tables + `build_sleep_sources` event-log fold +
Early-Bird prior); threaded via state_encoder gated on any-asleep (~0 cost when nobody sleeps). Verified:
12 unit + **calibration FUZZ** (`sleep_wake_fuzz_test.py`, 150 curated sleep-heavy battles, turn-capped
120 to dodge the bridge __RECON__ 64KB readline overflow on 250-turn stalls): **32521 obs-wiring checks
EXACT + empirical wake-freq == computed table** (opp K1 .247/.250, K2 .326/.333, K3 .571/.500, K4 1.0;
rest exact) — the real sim RNG matches our tables. + obs-roundtrip bit-identical + 2513 unit + golden
3453 + smoke roundtrip. Fixed distill student hardcoded per-mon 107→POKEMON_FULL_DIM. HONESTY GATE:
built+validated-CORRECT but UNMEASURED if it HELPS the policy → same fresh-run A/B as the cure bits (both
ride one ARCH bump). REUSABLE METHOD: compute the aleatoric-RNG belief, fuzz-CALIBRATE against the sim
(predicted prob == empirical freq), not just unit-test.

**SHIPPED THREAD (this branch landed on main across 3 ships 2026-06-15):** cure bits + sleep belief =
`92cf1ca` (obs 3453, reconciled w/ remote `gen3_protect_odds_v1` → obs 3455 final). Then user: "add a
dim for wish floating, don't wire it up... one for us and our opponent" → `gen3_wish_reserve_v1`
(`4ce3c72`, 2 reactive scalars vec[17]/vec[18], unwired/always-0, REACTIVE_SCALAR_DIM 17→19, obs
3455→3457) — reserve-now-so-wiring-is-values-only. **Then user: "robustly implement wish floating...
read the showdown impl... GIGO is terrible" → `gen3_wish_wired_v1` (BUILT+validated, NOT yet shipped at
this write):** KEY FINDING (wish research workflow, gen3 inherits the GEN4 wish condition NOT base) =
heal is the **RECIPIENT's floor(maxhp/2)** at END of cast+1 (NOT the wisher's, NOT captured at cast) →
heal-as-fraction is ALWAYS ≈0.5 → encode flat `WISH_HEAL_FRACTION=0.5` when pending, 0 else → **NO
max-HP read, GIGO-IMPOSSIBLE**. SLOT-keyed (survives faint/Roar-phaze/switch/self-KO — gen3 sends
replacements mid-turn before residuals), duration 2, double-Wish on occupied slot FAILS, full-HP resolve
SILENT. poke-env tracks NONE of it → reconstructed from event log (`observation/wish_belief.py`
`build_wish_pending`: pending iff side successfully cast Wish at turn-1, double-Wish guard = cast at T
fails iff successful cast at T-1; set-membership pending check). FUZZ (`wish_floating_fuzz_test.py`,
curated Wish+Roar+Spikes team to force wish-passing/phaze-drag/hazard-sac edge cases): ground truth from
RAW protocol (`|move|Wish` use + `|-heal|[from] move: Wish` resolve = NON-circular vs the event log the
obs uses) → 120 battles **1086 actual resolves, 0 completeness/soundness/timing failures**; 7 unit +
2588 suite + obs-roundtrip bit-identical + golden 3457 + smoke. 6-point adversarial code review (its own
200+ battle probes: double-wish seqs, forced-switch turn-arith, Snatch-vs-Wish [gen3 sim doesn't
redirect self-Wish], Healing-Wish no collision) found NO bug. REUSABLE: gen3 INHERITS gen4 wish (check
the mod chain, not base moves.ts); the recipient-based heal is what makes it GIGO-proof.
