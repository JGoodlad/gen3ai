# designs/ — Version Map

This file tells Claude which `ai_vN` folder is relevant when reading or writing design
docs. Read it whenever you're about to touch anything in `designs/`.

---

## Critical: Training run ≠ Code version

**These two are almost always at different versions simultaneously.** A training run lasts
weeks; code changes happen daily. When the user says "update the doc" or "record what we
built", figure out which version applies to *what was just implemented*, not to *what is
currently training*.

To orient yourself:

- `git log --oneline -10 -- designs/ src/` — which `ai_vN` folder was most recently
  touched by commits? That's the version the code changes belong to.
- `designs/ai_vN/todo.md` — the in-progress version's todo has the most recent `✓ DONE`
  entries and open items; the running training run's todo is mostly done.
- When in doubt, ask: "is this an implementation doc for new code, or a record of what
  a running experiment does?"

**Current state as of 2026-05-31:**

| What | Version | Notes |
|------|---------|-------|
| **Active training run** | **ai_v4** | Fresh run started 2026-05-31 (`run_20260531_182804`, git `63220dd`, 300M-step target) on the v4 arch (`gen3_trapping_signals_v1`, obs 3321); fixed-bot eval pool — **pathology-hunting phase before self-play**. The long ai_v3 ~350M run is retired. |
| **Code on main** | **ai_v4 (closed out)** | Event-sourced battle layer / strict battle-API / observation richness / obs-build perf — landed across impl_step1–9. Open tail: **pathology hunting** (eval-replay analysis). Self-play code has landed (`selfplay_callback.py`, `--self-play`) but is gated behind pathology cleanup; self-play/league as a chapter is ai_v5. |

---

## Version summaries

### ai_v1
Initial end-to-end PPO pipeline. Basic observation encoding, action masking, first working
training loop. Mostly design/analysis docs — no stable training run yet.

### aI_v2 (note: mixed case in filesystem)
Feature extractor redesign. Shared move processor, role encoder, team attention heads.
First architecture that learned meaningful strategy beyond random.

### ai_v3
Stability and signal hardening. Goals: clean the pipeline, encode richer state, get to
a stable 60–70% win rate against fixed bots.

Key milestones in order: clean pipeline (`impl_step1`), observation features (`impl_step2`),
architecture improvements (`impl_step3`), reward shaping (`impl_step4`), hyperparameters
(`impl_step5`), active state signals (`impl_step6`), effectiveness + move order
(`impl_step7`), item consumption (`impl_step8`), reward overhaul (`impl_step9`), adaptive
training infrastructure (`impl_step10`). Also: launcher with restart loop, spectator mode.

**Training run:** The long-running v3 experiment (350M+ steps) is the most mature model.
It reached ~70–75% vs Heuristic, limited by the fixed-bot ceiling — the policy fights
entropy collapse (ent_coef rose 0.029→0.055) rather than improving further.

### ai_v4
Event-sourced battle layer, strict battle-API, observation richness, and obs-build
performance. *(Originally planned as the self-play/league chapter; that work was deferred to
ai_v5, and ai_v4 became the data-quality + encapsulation chapter that has to come first.)*

Key milestones in order (impl_step1–9): own-team IV/EV/nature spread (`impl_step1`), opponent
Hidden Power type inference (`impl_step2`), damaging-event attribution (`impl_step3`), unified
L=2 transformer feature extractor (`impl_step4`), move-outcome reporting (`impl_step5`), the
next-run bundle — accuracy + modular extractor + dual-head value + reward overhaul
(`impl_step6`), adaptive-LR KL band (`impl_step7`), strict battle-API + event-sourced TurnDelta
fold (`impl_step8`), and strict-API completion + trapping signals + the ~2× obs-build perf pass
(`impl_step9`). Net obs **3321-dim**, `ARCH_SIGNATURE = gen3_trapping_signals_v1`.

**Open tail:** pathology hunting (eval-replay analysis); plus the one unscheduled strict-API
sub-item, Phase 5b (true `LiveView` current-board event-fold — `todo_live_battle.md`). The
first v4-obs run is now live (the fresh fixed-bot run started 2026-05-31, see the state table
above) — the retired v3 run was on an older arch that can't load the v4 obs.

### ai_v5
Self-play / league play, then MCTS at inference + behavioural cloning from human replays.
**Self-play and league** (relocated here from the original ai_v4 plan) lead the version: the
agent trains against frozen copies of itself (snapshot pool, win-rate gating, sentinel
monotonicity — Step 1), then league play with exploiters (Step 2). A prerequisite for league
play is reward annealing ≥ 50% complete so the value head learns win probability. With that in
place: BC pre-training from the (already-implemented) replay collection pipeline, then MCTS
integration — Wang (2024) found MCTS gave 78.6% → 90.8% vs Heuristic, the biggest single
untapped lever.

### ai_v6
Specialisation and ladder play. Train team-specific models from the v3–v5 generalist,
then take them to the ranked ladder. Also integrates MCTS into the training loop.

### ai_v7
Rust battle simulator (via PyO3). Replaces the Node.js MCTS bridge for ~50× more MCTS
rollouts per turn. Pure performance work — game mechanics unchanged.

---

## Folder conventions

Each version folder has:
- `todo.md` — in-progress checklist; `✓ DONE` marks completed steps
- `impl_step*.md` — post-implementation records (what was built, constants set, files
  changed); these are the primary targets for `gen3ai-update-design-docs`
- `design_*.md` — forward-looking design docs written before implementation

When writing a new `impl_step*.md`, match the existing docs in that folder exactly —
heading levels, table style, and level of detail vary between versions.
