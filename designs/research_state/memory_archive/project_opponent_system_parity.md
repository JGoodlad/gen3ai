---
name: project_opponent_system_parity
description: "Opponent-system cleanup audit (self-play/stable/exploiter): mostly defensible, 2 low-severity debts; DEFER refactor while ai_v7_05 live — design note written"
metadata:
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

**2026-07-03. Opponent-flag cleanup audit (6-agent workflow, adversarially critiqued). NOT built — design
note `designs/ai_v7/design_opponent_system_parity.md`.** Trigger: launching ai_v7_05 exploiter-mode
specialist we hit `--exploiter <T>` NOT auto-evaluating its target (must ALSO pass `--stable-opponents <T>`
for a `win_rate_vs_ext_<T>` verdict). Owner asked: is the opponent-flag surface accreting non-orthogonal
edge cases, should we unify?

**DIAGNOSIS (owner's instinct was right — a cleaner design existed):** "opponent" = TWO orthogonal axes the
code only unified in SELF-PLAY: **Axis A = training MIXTURE** (weighted sources: bots/self-pool/frozen-external,
each a selection prob) vs **Axis B = eval SET** (which frozen refs to SCORE, independent of who you train on).
`--stable-opponents` wires BOTH but only under `--self-play` (`if self_play and stable_opponents:`
train_rl_agent ~2273); `--exploiter` (added later) wires only Axis A → never eval-registers its target. One
training-selection fn `wrappers.py:185-207` (B1 exploiter short-circuit / B2 challenge / B3 floor);
`_fixed_opponents` (eval) comes ONLY from `resolve_stable_opponents(--stable-opponents)`.

**VERDICT = ~80% DEFENSIBLE separation-of-concerns, ~20% real-but-LOW-severity debt.** Defensible (leave, add
docs): two loader gates (check_compatible vs check_opponent_compatible — full-shape for pool/sentinel vs
obs-family for foreign; unifying false-rejects league), mastery flip, `--exploiter` XOR `--self-play`,
eval-class-by-mode, stable-trains-only-under-selfplay, the mix-fraction mirror (test-guarded). GENUINE debt
(2): (1) exploiter target not auto-evaluated [the footgun], (2) stable share/mastery/temp silently no-op
without self-play. Both = additive ≤5-line fixes. (Fact: `--exploiter-pool`/`--trainee-archetype` are NOT on
main — branch-built, unshipped.)

**PLAN — Proposal A (minimal, recommended) for the NEXT run's branch:** (1) append `_exploiter_entry` to
`_fixed_opponents` — DEDUP-GUARDED `any(e.label==...)` [MANDATORY: `--exploiter T --stable-opponents T`
double-shards eval + last-writer-wins the `ext_T` key without it; `is_external` keeps it out of
bot_wr/ELO/best-model so training is safe]; (2) `sys.argv`-scan warning for inert stable knobs. No version
bump, byte-identical-off. **Proposal B (full two-axis `--opponents`/`--eval-opponents`) DEFERRED** — clean +
training-only (no arch/version/checkpoint break) BUT rewrites the selection hot path + RNG STREAM (weighted
`rng.choices` draws an extra per-episode → training-dist drift on resume; the "skip degenerate single-source"
fix MISSES the dominant early self-play `[bots,pool≈0]` 2-source case) + the telemetry mirror; caught by no
existing test. Only worth it if a real multi-target league appears, RNG-sequence test written FIRST.

**HARD CONSTRAINT (why not now):** ai_v7_05 runs `--sync-to-main` → 6h restarts pull origin/main HEAD, so ANY
opponent-system commit hits the LIVE run at next restart (not just flag changes). DON'T ship opponent-system
changes to main while a --sync-to-main run is live. ai_v7_05 already has its verdict via the added
`--stable-opponents ai_v7_02` → even Change 1 is a no-op for it (dedup skips). Ties to
[[project_tss_specialist_poc]]; the exploiter/stable mechanics extend [[project_exploiter_league_tooling]].
