# model.md — Running log of the Gen3AI model

**What this is.** A curated, dated journal of the *model's* architecture + research state — the
"what's going on with the network" that's otherwise scattered across `CLAUDE.md` version notes,
`src/agents/model/CLAUDE.md`, the per-feature `designs/ai_v6/design_*.md`, the memory files, and
`designs/research_state/`. This is the **single narrative timeline**: what the current model is, how it
got here (the `MODEL_CONFIG_VERSION` ladder), what just changed, and what's still an open bet.

**How to keep it.** Manually curated (it's a *log*, not auto-generated). **Append a dated entry under
"Log entries" on every `MODEL_CONFIG_VERSION` bump or significant architecture/training change**, newest
first. Keep entries short — what changed, why, how it was verified, and the honesty gate (learns ≠ helps).
Deep mechanics stay in the design docs / leaf CLAUDE.md; link to them. (Not in the auto-maintained
doc set — update it when asked or when you bump a version.)

---

## Current state (2026-06-17)

| | |
|---|---|
| `ARCH_SIGNATURE` | `gen3_wish_wired_v1` (obs layout/meaning; bumps only on an obs-vector change) |
| obs vector | **3457-dim** float32 (`Gen3ObservationEncoder.dimension`) |
| `MODEL_CONFIG_VERSION` | **37** on this worktree (HEAD/`main` is **36** after the v36 ship `4ad37da`; v37 = the status-into-trunk work below) |
| Live training run | `ai_v6_09_dmg_reattend_N_0617` — launched on `main`@`bbe321a` (**v35**): `--unified-moves both --spread-belief --damage-topk 5 --move-candidate-floor 0.02 --damage-refine-rounds 2 --damage-reattend --damage-matrices both`, self-play + PopArt + async-rollout, CPU damage obs **visible** (not masked) |
| Frontier | the in-trunk threat field is now **complete both directions + all signal classes** (damage v36, status v37); the next big run is **the full `--unified-obs` deprecation** A/B — see the open bets below |

### Architecture at a glance
`Gen3FeaturesExtractor` (`src/agents/model/features_extractor.py`) — a phase pipeline feeding **two**
projection heads (policy + value), each `pre_proj_norm → projection → ReLU`:

```
ObsUnpack → PokemonEncoder(+MoveLatentEncoder) → [BeliefSlots] → TeamTransformer (2 layers,
  between-layers refine: incoming→OUR tokens, [#1 outgoing→OPP tokens]) → [BeliefHead/MoveBelief]
  → CLSPool (policy/value-dedicated CLS) → [DamageOperator] → [damage-reattend] → ProjectionAssembler → (pi, vf)
```
Bracketed phases are flag-gated; with all off the chain is the byte-for-byte baseline. Side readouts
(never in pi/vf): `WinProbHead` (P(win)), `ValueDistHead` (return distribution). The architecture-constant
SoT is the module-level constants in `features_extractor.py`; `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` in
`model_version.py`. Full phase contract: `src/agents/model/CLAUDE.md`.

---

## `MODEL_CONFIG_VERSION` ladder (compact)

Each is OFF-byte-identical unless noted; only an obs-vector change bumps `ARCH_SIGNATURE`.

| v | tag / flag | what it added |
|---|---|---|
| 16 | `opp_belief_slots` (`--opp-belief-aux-coef`) | hidden-opp **belief slots** + species/moves aux head |
| 17 | `move_belief_mode` (`--move-belief-mode`) | **move-belief** reinjection into opp tokens |
| 18 | `opp_belief_latent` (`--opp-belief-latent-coef`) | SimSiam **latent** belief (role-token regression) |
| 19 | `damage_op` (`--damage-op`) | differentiable **DamageOperator** (believed-move incoming damage) |
| 20 | `move_prior_fusion` | unified move posterior = Smogon prior ⊕ learned delta, revealed pinned |
| 21 | `mask_incoming_damage_obs` | A/B knob: zero the CPU incoming-damage obs from the model's view |
| 22 | `win_prob_mode` | calibrated **P(win)** side head (none/read_only/shaping) |
| 23 | `gen3_unified_damage` | OUTGOING per-move direction + `move_candidate_floor` legality gate |
| 24 | `gen3_unified_move_system_v1` | `MoveLatentEncoder` + per-status secondary probs + latent grading |
| 25 | `gen3_unified_spread_belief_v1` | **SpreadBelief** (believed opp stats) + `--unified-obs` master mask |
| 26 | `gen3_unified_op_physics_v1` | op folds boosts/burn/weather/paralysis + fixed-damage |
| 27 | `gen3_unified_status_landing_v1` | op outgoing status-landing (Toxic/WoW/TWave/Spore/Leech Seed) |
| 28 | `gen3_unified_choice_band_v1` | op prices Choice Band (×1.5 phys) + CB-conditional tail belief |
| 29 | `value_dist_mode` | **distributional value** side head (interpretability; HL-Gauss aux) |
| 30 | `damage_topk_k` (`--damage-topk`) | DISCRETE top-K incoming move-space (anticipate the move, pick the immune pivot) |
| 31 | `damage_reattend` / `damage_refine_rounds` | re-attend over computed physics / **iterative refine** (incoming → OUR tokens, between layers) |
| 32 | `move_belief_prefuse` | move belief reinjected PRE-transformer (co-refines through attention) |
| 33 | `damage_refine_rounds` | (the iterative-refine int; sequenced after the v31/v32 collision) |
| 34 | `damage_matrices_outgoing` | OUTGOING per-move matrix (our 4 moves × opp 6, revealed-gated) |
| 35 | `damage_matrices_incoming` | INCOMING per-move matrix (enriched top-K; reuses `--damage-topk` K) |
| **36** | **`gen3_bidir_threat_trunk_v1`** | **bidirectional in-trunk threat** (outgoing→trunk, expected-latent defender, prob-outspeed) |
| **37** | **`gen3_status_trunk_v1`** | **status-landing into the trunk** (both directions) — the last CPU-obs deprecation gap |

(Older v1–v15: the ai_v3/ai_v4 obs-richness + reward + dual-head + strict-API era — see `designs/`
version map.)

---

## Log entries (newest first)

### 2026-06-17 — Prober GPU-obs observability (tooling, NOT a model change; in worktree, not shipped)
**A lens to SEE the model's world-model — not a new head.** As the architecture moved the belief/physics
signals into the trunk (v30→v37) and we're about to A/B the full `--unified-obs` CPU deprecation, we built
first-class observability into the **prober TUI** so the owner can watch what the model believes vs ground
truth and how it refines. No forward change — the ONE model edit is a **prober-only** capture stash
(`features_extractor.capture_refine_rounds`, default False → `last_refine_rounds` None → byte-identical, zero
training cost). What landed (all in `src/main/prober/` + a small capture in `player.py`/`battle_recorder.py`):
- **Beliefs section** (new, key `b`): species belief vs TRUE team (✓/≈/✗), move belief (✓ revealed · ≈
  unseen), **believed SPREAD vs true DERIVED stats** (the DamageOperator's stat input — surfaces a wrong
  spread as a damage root-cause, e.g. "believes Metagross Atk 385 vs true 305"), the **across-battle
  refinement trajectory** (axis B — species-confidence sparkline + ✓/✗ as reveals accumulate, model-free),
  the **within-forward refine rounds** (axis A — per-`--damage-refine-rounds` belief-entropy↓ + physics↑),
  and a value-dist × belief cross-read.
- **Threats section** (renamed from *Matchups*): reordered **GPU-first** — the `🔷` DamageOperator physics
  primary, the `📋` CPU obs decodes dim/subsumed below (full-styled only when no op). Provenance tags
  (`🔷 GPU` / `📋 CPU-obs`) everywhere + Flow `🔷 GPU-computed` callouts on the learned phases.
- **Capture (axis B beyond species):** `move_logits` (opp-active posterior) + `spread_belief` `[6,5]` ride
  the trace npz (OMITTED when off, NaN when headless) so move/spread trajectories decode without re-running.
- Pure engine builders (`build_spread_belief` / `build_refine_trajectory` / `build_belief_trajectory`) +
  `ProbeModel.spread_belief_view` / `refine_rounds_view`; new `analyze`-JSON fields. Tests: +10 engine
  units, +2 app-render +1 recorder, the renamed/moved-content app tests updated; a bridge-backed **prober
  fuzz** (`belief_obs_fuzz_test.py`, ~280 live decisions — stashes finite + engine decoders populate); a
  real-run `ProbeSession.analyze` + headless-TUI smoke on `ai_v6_06_unified_all` (species belief 4/5 top-1 @
  turn 1, spread MAE ≈53). **4-lens adversarial review** (engine-math / leak-byte-identity / app-render /
  integration-contract → 1 adjudicator): byte-identity-when-off + no-leak + section-key + graceful-off all
  held; 2 minor bugs FIXED — (a) `build_belief_trajectory` `n_correct` now consumes the still-hidden
  multiset (was set-membership → could over-count a revealed/duplicate guess), (b) the captured
  `move_logits`/`spread_belief` npz arrays (were written-but-unread) are now consumed by the trajectory's
  `Hmv`/`bAtk` sparklines (capture switched to the opp-active row). Design:
  `designs/ai_v6/design_gpu_obs_observability.md`; prober internals: `src/main/prober/CLAUDE.md` →
  *Beliefs / Threats (GPU-first observability)*.

### 2026-06-17 — v37 status-landing into the trunk (`gen3_status_trunk_v1`, shipped)
**The last CPU-obs deprecation gap.** The move-effect block's board-conditional `status_will_land` was
heads-only; status immunity (type × ability × already-statused × Sleep-Clause × Substitute) is a computed
MECHANICS fact (same class as type effectiveness), and *learning* it would force attention to correlate
non-local info (the move's status intent on one token, the defender's types+ability on another). So we
**compute it and hand it to the trunk**, both directions, via `--threat-status-refine` (one flag):

- **INCOMING** (`discrete_incoming_status`): the opp active's top-K believed status moves → per OUR mon
  `[P(major), P(immobilize=para/frz/slp)]`, injected onto OUR tokens (the "will I get statused" signal).
- **OUTGOING** (`discrete_outgoing_status`): our active's status moves → per opp mon (revealed-gated),
  injected onto OPP tokens (the in-trunk home for the masked `status_will_land`).

Both reuse the v27 `_status_landing` immunity physics + buffers; two zero-init residuals on the refine
loop. The **major-vs-immobilize split** makes the signal self-contained (the policy needn't cross-reference
which move). STRUCTURAL (v37), OFF byte-identical, requires `--damage-op` + `--damage-refine-rounds>0`.

**Verified:** 17 unit (T-Wave→Ground=0 both ways, immobilize⊆major, revealed-gating, identity-at-init,
grad); full suite 2817 passed; v37 smoke (all flags) roundtrip + train PASSED; **extensive fuzz — 1783
live bridge decisions, all invariants held, 687 priced a status-landing**; a 5-agent adversarial review
with an **exhaustive CPU-obs deprecation-gap audit** → **0 real bugs, 0 blocking gaps** (all 16 flags
dismissed: PP verified present in the per-mon move slot, the 7 effect/cure flags verified in `MOVE_ATTR`,
the CLI hard-requires the GPU replacement before any `--mask-*-obs`).

**Deprecation verdict — the full `--unified-obs` is now FAIR to A/B:** every CPU-obs signal has a GPU home
(damage→trunk via refine; status→trunk via v37; effects→move latent; PP→per-mon slot; provenance/p_outspeed/
crit→explicit op channels; per-move status_will_land + known→v27 heads block). **Honest residuals** (minor,
documented, not blockers — watch them in the A/B): the opp-recovery scalars are heads-only (op effect column)
and coarsen the Rest-specific self-status-CURE nuance into a generic "has recovery"; the crit-delta and
threat-provenance become implicit/op-channel rather than explicit decorrelated scalars.

### 2026-06-17 — v36 bidirectional in-trunk threat field (shipped `4ad37da`)
**Goal (owner):** make the model's threat *both directions* dynamic (known ⊕ believed) and **infused into
the trunk** so attention reasons over it — not just the projection heads. Three independent toggles:

- **#1 `--threat-refine-outgoing`** (STRUCTURAL) — the symmetric mirror of the incoming refine. New
  `DamageOperator.discrete_outgoing` (our active's 4 known moves → each opp mon → `[phys_high, spec_high,
  phys_pko, spec_pko]`) injected onto the **OPP token slice** via a zero-init `outgoing_proj`, riding the
  same `--damage-refine-rounds` between-layers loop. Requires `--damage-op` + `--damage-refine-rounds>0`.
- **#2 `--threat-unrevealed-outgoing`** (forward-behavior) — the **expected-latent defender**: keep an
  unrevealed opp mon *latent* and marginalize the move-belief's `P(species)` (read per-round from the
  factored `BeliefHead.species_logits`) through `SPECIES_EXP_MULT` (type chart × per-species expected
  ability immunity — Levitate/Water&Volt Absorb/Flash Fire/Thick Fat, folded from `gen3_ability_priors`) +
  `SPECIES_SPREAD_PRIOR` (E[bulk]/E[maxhp]). **P(KO) NULLED** (a full-HP switch-in is ~never OHKO'd — owner
  call; drops the Jensen-threshold complexity, keeps the expected magnitude). Requires #1 + a belief head.
- **#3 `--threat-prob-outspeed`** (forward-behavior) — uncertainty-aware `P(outspeed)`: ÷ the believed
  speed std (sigmoid≈normal-CDF) instead of a fixed scale.

**Data prerequisite unblocked:** there was **no species→types** in the data layer (the obs reads revealed
types live). Added `types` to the extractor → `gen3_species.json` → `SpeciesData.types` (386 species,
parity-green).

**Why "into the trunk" matters (the load-bearing finding):** the CPU incoming-damage obs reaches the trunk
only as **one diluted global token**; the DamageOperator output reached **only the projection heads**. So
masking the CPU obs without an in-trunk injector starved the trunk of all damage signal (the earlier
`--unified-obs` regression). `--damage-refine-rounds` put *incoming* damage in the trunk; v36 puts
*outgoing* there too. The belief heads already **warm-start to the priors** (zero-init delta over a prior
buffer), so the only cold spot at init is the zero-init injector ramp — not the belief.

**Verification:** `bidir_threat_test.py` (10 unit — kernel, immunity, expected-latent Levitate→0, pko-null,
prob-outspeed bounds, identity-at-init, grad→P(species)); full suite **2810 passed**; serverless smoke
(all 3 flags) **`[ModelVersion] Round-trip smoke test PASSED` + Training complete**; **real-battle fuzz**
`bidir_threat_fuzz_test.py` (481 live decisions, all invariants held, **188 priced an unrevealed
defender**); 4-lens adversarial review (physics/versioning/threading/leak-gradient + skeptical verify) —
**0 confirmed findings**. OFF byte-identical, version-gated, threaded through all opp-load + extractor sites.

**Honesty gate (learns ≠ helps):** wired + differentiable + exercised on real boards, but **UNMEASURED**
whether it improves the policy → needs a fresh-run A/B (`--threat-refine-outgoing` on/off). Risk =
"learnable-but-inconsequential" (the incoming-belief precedent). Design + diagrams:
`designs/ai_v6/design_bidirectional_threat_trunk.md`, `gen3ai/tmp/model_v36_full.png`.

### 2026-06-16 → 06-17 — v30–v35 damage-system enrichment (shipped to main @ `bbe321a`)
top-K incoming move-space (v30); reattend + iterative refine + move-prefuse (v31–v33); per-move
outgoing/incoming damage matrices (v34–v35). The **live run** `ai_v6_09_dmg_reattend_N_0617` exercises
refine 2 + reattend + matrices both, CPU damage obs still visible.

---

## Open bets / honesty gates (the frontier)

- **Does v36 help?** Fresh-run A/B on `--threat-refine-outgoing` (± `--threat-unrevealed-outgoing` /
  `--threat-prob-outspeed`). Watch: belief precision↑, surprise-OHKO crater share↓, win-rate non-regress,
  `grad/value_share` (PopArt). Same gate as every prior belief/damage feature.
- **Deprecate the CPU obs — the next big run.** As of v37 the architecture is in place for the FULL
  `--unified-obs` (all three CPU blocks): damage + status, both directions, are in the trunk; effects in
  the move latent; PP in the per-mon slot. The A/B: a fresh run with `--unified-moves both
  --damage-refine-rounds 2 --threat-refine-outgoing --threat-unrevealed-outgoing --threat-prob-outspeed
  --threat-status-refine` (the GPU carries everything) — control = CPU obs visible, treatment =
  `--unified-obs`. If treatment ≥ control → deprecate. Watch the documented residuals (opp-recovery
  heads-only + Rest-cure coarsening; crit-delta/provenance now implicit). Warm-start lever if an early dip
  shows: init the refine/outgoing/status projections small-normal vs zero (the belief heads already init at
  the prior, so only the injector ramps).
- **Broader frontier** (see `designs/research_state/` + memory): the ceiling is structural (offense/
  opponent-blind obs being closed by these belief/damage features; scalar PPO teacher; self-play
  treadmill), not capacity. Levers still open: PFSP-league/exploiter, offline teacher, tail-calibrated
  critic. MCTS is ruled out by the owner.

## Pointers
- Architecture detail: `src/agents/model/CLAUDE.md` · obs layout: `src/agents/observation/CLAUDE.md`
- Per-feature designs: `designs/ai_v6/design_*.md` (damage op, unified move/damage, matrices, refine,
  distributional critic, **bidirectional threat**)
- Diagrams: `designs/ai_v6/threat_{now,future,delta}.svg` + `gen3ai/tmp/{model_v36_full,
  latent_expected_defender,warm_start,slow_start_recap,incoming_cpu_vs_gpu}.png`
- Research state / levers ledger: `designs/research_state/` · pathology register: `designs/design_pathologies.md`
