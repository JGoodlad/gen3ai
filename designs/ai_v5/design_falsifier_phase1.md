# Dice-Attribution Falsifier (Phase 1) — design

**Status: BUILT + SHIPPED (`2236b76`, 2026-06-11).** The first consumer of the Phase-0
battle-reconstruction layer (`design_battle_reconstruction.md`): per decision in
a recorded battle, separate **luck** (the dice broke bad) from **reducible
mistake** (a better action existed). Lives in the prober ecosystem:
`src/main/prober/falsifier.py` + `ProbeSession.falsify` +
`python -m main.prober.query falsify`. **As-built record (files, gates, module
map): `impl_step9_battle_reconstruction_and_falsifier.md`.**

## Why (the question it answers)

The plateau diagnosis showed the critic's td-tail stuck at ≈−10 with ~49% of
losses passing through value craters. The unresolved confound: a crater can be
**aleatoric** (a fair coinflip lost — irreducible, the fix is a *distributional*
critic that prices the variance) or **epistemic/policy** (a better line existed —
the fix is obs/reward/policy). These pull toward different levers; eyeballing
replays can't separate them. The falsifier measures the split.

## Probe decisions (the Phase-0 deferred choices, now locked)

- **Luck axis = fix-both re-roll.** Both players' actions held at the recorded
  picks; the turn re-rolled under N fresh PRNG seeds. The REALIZED line comes
  from the special `"original"` seed (no PRNG swap + exact recorded follow-ups —
  a Phase-1 driver extension), scored through the same outcome pipeline, so its
  margin is directly comparable. `luck_percentile` = midrank of realized within
  the fresh-seed margins.
- **Mistake axis = paired alternative sweep.** Top-k alternative LEGAL actions
  (ranked by the saved policy logits in `states.npz`), each replayed under the
  SAME N seeds — **common random numbers**, so the advantage estimate is a mean
  of *paired* per-seed differences (the dice noise cancels within pairs; SE is
  typically ~5–10× tighter than two independent arms). Action index → sim choice
  string via the REAL mapper: `obs_materializer(map_actions_at=i)` replays to
  the decision and runs `action_to_order` against the rebuilt state — zero
  mapping reimplementation, legality = the live mask.
- **Metric = omniscient material margin** `(our_alive − opp_alive) +
  (our_hp_frac − opp_hp_frac)`. Referee-side ANALYSIS is allowed to be
  omniscient — the one-sided wall constrains what feeds the encoder/critic, not
  what a forensic report reads off the board. A critic-scored variant (V(s′)
  over materialized one-sided obs — the distributional-critic calibration probe)
  is the designed follow-up, NOT built: it needs a checkpoint forward per seed
  and answers a different question (is V calibrated to the dice distribution).
- **Anchors**: the `worst` most-negative-δ decisions (δ = r + γV(s′) − V(s), the
  prober's formula) on distinct turns, `move_selection` only — re-rolls anchor
  at start-of-turn rounds (Phase-0 limit); a forced-switch crater attributes to
  its turn's move decision. Explicit `--inv` overrides.
- **Verdicts** (v1 thresholds, centralized in `falsifier.py`, echoed in output):
  `MISTAKE` = best paired advantage > 0.5 material AND > 1.96·SE;
  `LUCK` = realized percentile ≤ 0.20 and no such alt; `MIXED` = both;
  `NEUTRAL` = neither. An alt the sim refuses (maybe-trapped switch — detected
  via `[Unavailable choice]` in the one-sided suffix) on >50% of seeds was never
  really available → excluded from the verdict, reported with `refused_frac`.
- **New mid-turn decisions in re-rolled timelines** (a forced switch the
  original never had): `--followup random|default` (default random, uniform
  legal via an aux PRNG derived from the re-roll seed — independent of the sim
  dice under study).

## Phase-0 API extensions added for this

- `replay_driver.js`: seed `"original"` (no PRNG swap; with both sides recorded
  → `resolveTurnExact` feeds the remaining recorded commands verbatim);
  `recorded` action source became a per-side QUEUE (a live refused-then-corrected
  sequence replays faithfully under fresh seeds — pre-state identical ⇒ identical
  refusal ⇒ recorded correction). Routing invariant in `resolveTurn`: within a
  turn a side gets at most one `'move'` request (refusals re-ask as `'move'`,
  follow-ups ask as `'switch'`), so `'move'` → source, else → followup.
- `obs_materializer`: `map_actions_at=i` (legal action idx → sim choice string
  at decision i, via the real mapper) and `stop_after_decision=i` (cheap prefix
  replay).

## Validation (green, 2026-06-11)

- `falsifier_test.py` — 14 pure tests: margin antisymmetry / alive-dominates,
  midrank percentile, paired stats (1 pair ⇒ ∞ SE ⇒ never significant), the
  verdict matrix incl. threshold edges, seed determinism, δ-anchor ranking +
  forced-switch remap + guards.
- `falsifier_integration_test.py` — real bridge battle → full pipeline:
  realized line exists, dice distribution sampled, choices mapped by the real
  mapper, verdicts valid, **and the whole attribution is deterministic on
  re-run** (same seeds, same replay ⇒ identical output).
- Sanity property on the demo: with a RANDOM "policy" as trainee, decisions
  should read MISTAKE often — they do (e.g. chosen `switch:tyranitar` vs
  `move hydropump`: paired advantage +0.89 ± 0.11 over 24 seeds → MISTAKE),
  while a genuine bottom-decile dice outcome read LUCK (percentile 0.083).
- Phase-0 regression after the driver rewrite: `reconstruction_fuzz_test` and
  `obs_roundtrip_fuzz_test` (339/339 bit-identical) both pass.

## Cost

~1–2 s per arm at 40 seeds (fresh Node replay per seed + rebuild-to-turn); a
decision with 3 alts ≈ 5–10 s; a battle at `--worst 3` ≈ 20–30 s. Fine for
forensics; the `State.serializeBattle` clone fast path remains the known ~2–3×
optimization if a sweep over hundreds of decisions is ever needed.

## Deferred

- **Run-level aggregation** (`falsify-scan`): falsify the worst decision of
  EVERY loss at a step → the run's luck-vs-mistake share — the number that
  decides how much of the td-tail is irreducible. Needs traces from a live run
  (no run has reconstruction records yet; the live run picks the layer up at its
  next restart).
- Critic-scored arm (V(s′) over materialized obs) — distributional-critic
  calibration.
- Opponent re-sample arm (game-theoretic caveat: fixing the opponent's recorded
  action leaks what they did; re-sampling needs an opponent policy source) and
  the 3-way determinization (dice vs hidden-team vs critic-bias) — needs
  team-pool sampling.
- TUI parity (a Falsify panel) once the CLI shape settles in real use.
