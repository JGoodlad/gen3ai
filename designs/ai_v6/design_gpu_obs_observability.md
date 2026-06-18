# Design — First-class GPU-obs observability in the prober TUI (ai_v6)

**Status:** DESIGN — owner-approved, scheduled for an autonomous overnight build (start 2026-06-17 23:00 PDT).
**This doc is the self-contained spec the 11pm session implements.** Diagram: `gen3ai/tmp/observability_design.png`.

## Goal (owner)
See the model's GPU-side beliefs (species / moves / spread / status) + the computed damage physics, **compared
to ground truth**, and **watch the model refine over time** — and make the TUI **first-class for the GPU obs**, not
the CPU obs we're deprecating. The owner will "hunt strange things"; this is the lens to see how the model reasons
about the state space.

## Locked decisions (from the owner Q&A)
- **Home:** extend the EXISTING prober TUI (`src/main/prober/`) — it owns trace discovery, per-battle model reload,
  the pure `engine.py`/`model.py` seam, and the `reconstruction.json` ground-truth loader. NOT a new tool, NOT the
  launcher (the launcher is live-training aggregates; the prober has per-decision state + ground truth).
- **Over-time axes:** build **B (across-battle turns)** AND **A (within-forward refine rounds)**. NOT C
  (across-training) — that's a separate training-callback metric, out of scope here.
- **CPU-obs panels:** **demote to dim fallback** — render dim/secondary, only fully-styled when the GPU op is
  absent (non-GPU checkpoint), and struck-through / "masked from model" when `--unified-obs` zeroed them.
- **Spread truth:** show the true **DERIVED stat** (head-comparable) with raw EVs/nature as a small annotation.
- **First-class:** every datum tagged `🔷 GPU` (learned op/belief) vs `📋 CPU-obs` (decoded obs region being
  subsumed); GPU views render PRIMARY; CPU decodes render last/dim.
- **Capture:** ship the views on live re-computation (works on existing checkpoints) AND add the small npz capture
  for the move/spread trajectory on future runs (axis B beyond species). Ground-truth EVs/moves are already in
  `reconstruction.team_details()` — just consume them.

## Architecture (engine pure, app thin — preserve the seam)
All new analysis lives in the **pure `engine.py`** (no Textual/IO) as new `InvocationAnalysis` fields; `app.py`
stays a thin renderer; the `analyze` JSON CLI inherits every field via `asdict`. New `model.py`/`ProbeModel`
decoders for the stashes the engine reads.

### Sections (`app.py` `_SECTIONS` is the single source — bindings + titles auto-regenerate)
- **NEW `sec-beliefs` "Beliefs"** — the model's world-model vs ground truth (the 4 belief legs + the over-time
  views). Insert after Faithfulness.
- **RENAME `sec-matchups` → `sec-threats` "Threats"** — reorder GPU-first (DamageOperator primary; CPU decodes
  dim/last).
- **Section budget:** 11 sections vs 10 digit hotkeys (1-9,0). Resolution: keep digits for the first 10 by panel
  order; the 11th binds a **letter key** (e.g. `b` for Beliefs) — extend the binding generator in `app.py` to fall
  back to a letter when the digit pool is exhausted (don't silently drop the 11th binding). Keep
  `_OPEN_BY_DEFAULT` sensible (add `sec-beliefs`).

## Views to build
### Beliefs section
1. **Belief-vs-truth matrix** (top panel) — one row per opp slot: believed species top-k (🔷) | TRUE species (📋
   `team_details()`) | ✓/≈/✗ + rank; EXTEND with per-slot believed MOVES (from `move_belief_view`) vs true
   moveset, and believed SPREAD vs true. Reuse/extend the existing `build_belief_truth` Hungarian match
   (`engine.py:~985`).
2. **Believed spread vs true** — `ProbeModel.spread_belief_view()` reads `last_spread_belief[B,6,5]`
   (atk/def/spa/spd/spe) → E[stat]±std per hidden slot; truth = the DERIVED stat from `team_details()` EVs/IV/nature
   (compute via the same L100 formula the op uses), with raw EVs/nature annotated; Smogon prior as a third column.
   This is the DamageOperator's input — a wrong spread is an otherwise-invisible damage root-cause.
3. **Refinement trajectory (axis B, across-battle)** — for the navigated battle, per hidden slot: top-1 species
   confidence sparkline + ✓/✗ correctness dots across the battle's decisions, + move-belief entropy decay. Species
   reads from the per-decision summary `belief` blocks already on disk; move/spread need the npz capture (below).
4. **Within-forward refine rounds (axis A)** — for the selected decision, show the per-round (`--damage-refine-rounds
   N`) sharpening: move-belief entropy + the lean incoming-damage `[phys_high,spec_high,phys_pko,spec_pko]` per
   round 1..N, so you can SEE the belief/physics tighten across the transformer layers. Needs the refine-round
   side-stash (below) + a `ProbeModel.refine_rounds_view()`.
5. **value-dist × belief cross-read** (footer) — the `value_dist` histogram (already in Outcome) next to belief
   confidence: does critic bimodality co-occur with low belief confidence?

### Threats section (promoted GPU-first)
6. **GPU damage physics primary** — `decode_damage_block` incoming/outgoing per-mon `[low–high·crit·acc·→KO]` 🔷;
   the **outgoing damage matrix** (v34, our 4 moves × opp 6 → "KO a switch-in?"); the **top-K discrete incoming**
   move-space + per-pivot safe-switch + **status-landing (v37)**. The CPU `their_matchups`/`incoming_damage` decodes
   drop to a single dim 📋 "obs-static (subsumed)" line, struck-through when `--unified-obs` masked them.
7. **Provenance tags everywhere** — a `_prov(gpu: bool)` helper (🔷/📋 + style) used across panels + the Flow
   diagram phase rows (`DamageOperator`/`MoveBelief`/`SpreadBelief`/`BeliefHead` get `🔷 GPU-computed` callouts).

## Capture additions (model + trace)
- **Axis A — refine-round side-stash (opt-in, zero training cost):** add an extractor attr `capture_refine_rounds`
  (default False); when set, the `between_layers` refine_cb appends `(layer_idx, move_logits.detach(),
  discrete_incoming.detach())` (+ the status block if `threat_status_refine`) to `self.last_refine_rounds`. The
  PROBER sets `capture_refine_rounds=True` before its re-run forward (no rollout path touches it → no training
  cost). ~10KB/decision, prober-only.
- **Axis B — trace npz (future runs):** thread `move_logits` (sigmoid posterior, the opp-active row) + `spread_belief`
  ([6,5]) through `RLPlayer._last_prediction` (`inference/player.py`) → `BattleRecorder.record(state=…)` →
  `states_arrays` as new NaN-filled npz arrays (parallel to `value_dist`). Lets move/spread trajectory decode
  WITHOUT re-running + survives arch drift. Species trajectory needs nothing new (already in the summary `belief`).
- **Ground truth:** consume `reconstruction.team_details()` (species/moves/evs/ivs/nature/ability/item) — already
  present; today only species is read. Add ability + item to the truth column where useful.

## Robustness / graceful degradation (mandatory)
- Belief off (`--opp-belief-aux-coef==0`) → Beliefs section shows "belief head not enabled" (no crash).
- `damage_op` off → Threats falls back to the CPU decodes at full styling (the demote only applies when the op is
  present). `--unified-obs` checkpoint → CPU decodes struck-through "masked from model".
- Websocket trace (no `reconstruction.json`) → truth columns show "no ground truth (websocket)"; the trajectory
  still shows confidence/entropy **uncolored** (the sharpening is informative without a scorecard).
- All decoders return None/empty gracefully; the `analyze` CLI emits the new fields (None when off).

## Test + fuzz plan (extensive — the owner will rely on this)
- **Engine unit tests** (`src/main/prober/engine_test.py` or a new `belief_obs_test.py`): `build_belief_full`
  species/moves/spread vs a synthetic truth (✓/✗ + rank correct); `spread_belief_view` derived-stat math;
  `refine_rounds_view` shape + monotone-entropy on a synthetic sharpening; `build_belief_trajectory` over synthetic
  per-turn beliefs; the graceful-None paths (belief off, op off, no truth).
- **Prober integration/fuzz** (bridge-backed, no server — the project's fuzz pattern): play N real battles with a
  belief-on + damage-op + refine + status model, capture traces + reconstruction, run the engine decode on each
  decision, and assert: belief-vs-truth aligns to the actual hidden team; the trajectory is populated + monotone-ish;
  the refine-rounds stash has N rounds; the spread/move npz arrays round-trip; the CPU-demote/masked rendering is
  correct. Reuse `bidir_threat_fuzz_test.py` / `obs_roundtrip_fuzz_test.py` harness patterns.
- **Smoke:** `python -m main.prober <run_dir>` renders without error on a real run; `python -m main.prober.query
  analyze …` emits the new fields.

## Build order (phased — for the overnight session)
1. `ProbeModel` decoders: `spread_belief_view`, `refine_rounds_view`, extend belief/move decode + the truth join
   (consume `team_details()` moves/evs/ability/item). Engine `build_belief_full` / `build_belief_trajectory`.
2. Model side: `capture_refine_rounds` opt-in stash in the refine_cb (prober-only).
3. Trace capture: `move_logits` + `spread_belief` npz arrays (player → recorder → states_arrays), NaN when off.
4. `app.py`: new Beliefs section + render methods; rename/reorder Matchups→Threats GPU-first; `_prov` tag helper;
   binding-gen letter fallback for the 11th section; Flow phase callouts.
5. Tests (engine units) + the prober fuzz + smoke.
6. Adversarial review (workflow). Update `src/main/prober/CLAUDE.md` + `designs/model.md` log.

## Constraints for the autonomous session
- **DO NOT `/gen3ai-ship` and DO NOT commit/push** — the owner reviews + ships. Leave the work in the worktree
  tree. (The prober runs fine from the worktree: `PYTHONPATH=src python -m main.prober <run_dir>`.)
- ultracode is on — use workflows for research/review; adversarially verify findings.
- Run the full unit suite + the prober fuzz before declaring done; report a morning summary (what built, test
  results, how to run it, any residuals).
