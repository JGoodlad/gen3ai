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

**Current state as of 2026-05:**

| What | Version | Notes |
|------|---------|-------|
| **Active training run** | **ai_v3 end** | ~350M steps, fixed bot pool (Random/Heuristic/Aggressive/Staller/SetupSweep), no self-play; AdamW and adaptive LR added recently |
| **Code being changed** | **ai_v4** | Self-play not yet implemented — currently in prep (AdamW optimizer, adaptive KL, reward tuning); self-play is the next thing to add |

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
Self-play and league play. Agent trains against frozen copies of itself (snapshot pool,
win-rate gating, sentinel monotonicity). Step 1 (self-play) is the current milestone —
**not yet implemented**. Step 2 (league play with exploiters) follows after.

**Prerequisite for league play:** reward annealing ≥ 50% complete so the value head learns
win probability (needed for MCTS in v5).

### ai_v5
MCTS at inference + behavioural cloning from human replays. Replay collection pipeline
already implemented. BC pre-training (Step 2) and MCTS integration (Step 5) are the main
goals. Wang (2024) found MCTS gave 78.6% → 90.8% vs Heuristic — this is the biggest
single untapped lever.

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
