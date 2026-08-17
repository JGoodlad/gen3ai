# Opponent-PP observability (the stall-class blind spot)

**Status:** 🔬 open — probe pre-registered, not yet run · **Ledger id:** C3

One-line claim: *the critic is blind on stall losses because the opponent's PP ledger — the
quantity that decides a Gen-3 stall war — is structurally absent from the observation.*

## Known (established, cleared the honesty gates)

- The obs encodes opponent `current_pp` as ALWAYS FULL: `src/agents/observation/moves.py:129-130`
  ("Opponent moves in Gen 3 always show full PP since Showdown doesn't track opponent PP for
  Gen 3"). Verified 2026-08-17: no usage-count tracker exists anywhere in
  `observation/`/`battle/`/`training/`, and the 7-turn TurnDelta window cannot span a stall war.
  Our own PP is real (it rides the request).
- The critic's failure is specifically the stall/losing tail: LEVEL-calibrated vs pool (C2,
  rollout-PIT gap −0.011, Spearman +0.66) yet win_prob 0.7–0.98 on stall-loss decisions whose
  resampled-dice win-rate is 0.0–0.4 (gen-12 @24M probes).
- The win-prob head is Monte-Carlo supervised (undiscounted terminal outcome, BCE —
  `win_prob_callback.py`), so bootstrapped self-confirmation CANNOT explain its blindness: with
  ground-truth labels, a persistent class-conditional miss means the states are off-distribution
  or the signal is absent from the input. This lever is the "absent from the input" branch.
- Opponent move usage is PUBLIC (every `|move|` protocol line) → a tracker is "provide raw known
  facts", not a prior. No Smogon-sourcing issue.

## Not-known (the open questions — what would resolve this)

- Whether PP/usage features actually carry outcome signal beyond what the critic already reads —
  the pre-registered probe (`gen13_endofrun_runbook.md` §8): gen-12 traces, battles ≥50 turns
  (608 battles, 253 L / 355 W), decisions at turn ≥30, ΔAUC of win_prob+turn vs +PP features,
  battle-grouped CV + battle-bootstrap + permutation null.
- If convicted: how much of the blind-loss mass the production feature recovers (a training-run
  question, not answerable offline).
- The alternative suspect if the probe nulls: the training DISTRIBUTION of stall games.

## Pros (why it might be the lever)

- Coheres from three independent angles: blind exactly on the game class PP decides; calibrated
  everywhere PP never binds; MC supervision rules out the objective-side story for the win-prob
  head.
- Cheap at every stage: the probe is offline on existing traces (no model loading — recorded
  `win_probs` + summary action streams); the production feature is an event-log fold (the
  TurnDelta/tracker pattern already exists) feeding the per-move slot `pp_fraction` lane that
  already exists for our side.
- Feeds policy too, not just critic: `pp_fraction` rides the move tokens the pointer head scores
  ("their Recover is nearly dry — keep clicking").

## Cons (why it might not work or not matter)

- Stall losses recoverable-early means the POLICY walks in; seeing the collapse is necessary but
  the payoff chain (critic sees → advantage punishes stall-entering lines → policy avoids) is two
  steps long and only the first is tested here.
- The usage counts the probe can build UNDERCOUNT (decision-point outcomes only; Pressure and
  PP-Ups unmodeled) — a null with poor count-coverage is weak evidence against.
- Archetype confound: recovery counts encode "opponent is a staller". Acceptable for an obs
  feature (it would carry that signal in production too), but it means a positive ΔAUC is
  "PP-ledger + archetype beyond win_prob", not PP alone — the secondary conditional slice is the
  sharper reading.
- Stall is a minority of battles; even a full fix moves the blind-loss fraction, not the ELO
  headline, on its own.

## Next test

- Run `gen13_endofrun_runbook.md` §8 exactly as pre-registered. GO gate: primary ΔAUC CI
  excluding 0 + permutation p<0.05 → build the tracker feature as a gen-14 rider candidate.
  KILL: both primary and secondary null with count-coverage ≥ ~0.7 → close this lever, open the
  training-distribution investigation.
