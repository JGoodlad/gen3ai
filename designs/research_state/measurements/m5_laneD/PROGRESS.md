# M5 Lane D — episodes and reward in the Rust env core: PROGRESS (resume point)

Lane D of `designs/endstate/program_rust_core.md` §2 M5. It owns:

- `src/rust_env/src/episode.rs` (RESET / STEP / QUARANTINE, the terminal reward, `terminated` /
  `truncated`, the stall forfeit, ties, the refused-start PARK);
- `src/utils/rust_env/episode.py` (the spec's `terminal` object, the production default, the
  stall threshold);
- `src/rust_env/tests/episode_test.rs`, `src/agents/training/rust_env_episode_parity_test.py`;
- hand-offs: the `reward` / `terminated` / `truncated` rows of `columns.py`, the `terminal` row of
  `protocol.SPEC_KEYS` + `spec.rs`, `pool.rs`'s env ops moved out (fields `pub(crate)`, the
  `parked` flag, `EnvReport.quarantined` a `Vec`), `dispatch.rs` banking a `Vec`, gate ①'s
  forfeit tail-frame tolerance removed (`sim_bridge_parity_test.rs`), `tests/common` digest covers
  the two new columns.

**Gate:** slice N `reward` / `terminated` / `truncated`, the stall forfeit at
`StallConfig().threshold`, ties, the terminal observation.

## Units

| # | unit | status |
|---|---|---|
| 1 | `episode.rs` + columns + spec `terminal` + F-L0-6 + F-L0-2 + Rust pins | LANDED (see git log, `M5 Lane D`) |
| 2 | slice N episode parity through both front ends (COMMIT routine, MILESTONE slow) | LANDED (same commit) |

## Semantics (all equal to `Gen3Env`, read from the Python code)

- **reward** — `Gen3RewardManager.process_turn_reward`'s terminal on p1's reading: `victory_value`
  on a win; else 0 (indicator) or `draw_penalty` if the end turn `>= timeout_turn_cap`
  (`reward_weights._TIMEOUT_TURN_CAP`), else `-victory_value` (decisive loss / pre-cap tie).
- **terminated / truncated** — `PokeEnv.calc_term_trunc` (p1's own team size for both sides).
  RAW env flags; `wrappers.resolve_episode_end` (winprob ⇒ terminal) stays the host's.
- **stall forfeit** — `Gen3Env.action_to_order`: at a p1 decision with `turn >= turn_limit`
  (`StallConfig().threshold`), p1 forfeits instead of acting; p2 is not fed (`PokeEnv.step`
  skips agent2 after a forfeit). Only p1 ever forfeits. Decided BEFORE any feed.
- **tie** — no winner, truncated; 0 (indicator) / `-victory_value` (signed).
- **terminal observation** — NOT produced (Decision; see the program doc). Only SB3's truncation
  bootstrap reads it, which production never takes.
- **refused start** — banked, `refused` = 1, env PARKED (`need` = 0 0); the ended episode's
  outcome stands; the next op starts from the staged inputs.

## Findings closed

- **F-L0-6** (the forfeit checked after the feeds, `>` not `>=`): fixed; pinned by
  `episode_test.rs::the_stall_forfeit_is_decided_before_any_feed` (mutation-checked: the Lane-0
  placeholder fails it on "a row was encoded and not exposed").
- **F-L0-2** (an ended episode's reward lost to a refused next start): fixed; pinned by
  `episode::tests::f_l0_2_…` (mutation-checked). Lane 0's description was also inexact: with a
  deterministic refusal the quarantine's in-op retry failed again, so the batch FAULTED (poisoning
  the pool) rather than reading `refused = 1, reward = 0`.

## Gate results (2026-09-29)

- Rust: 15 lib tests + `episode_test.rs` (2) + the crate's other tests PASS; gate ① (via
  `core_cargo_test.py`) PASS with the forfeit tolerance removed.
- Slice N COMMIT (routine): natural endings 6 episodes (2 wins, all terminated), stall forfeits at
  threshold 5 (4/4 truncated), signed terminal at threshold/cap 45 (rewards +30 ×2, −30 ×1, −35 ×5),
  one TIE (Explosion vs a last mon, truncated, reward 0), teeth (a shifted Python terminal FAILS).
  FFI and process recordings identical.
- Slice N MILESTONE (`slow`): 60 pool + 60 ladder episodes, production terminal and threshold — PASS.

## Findings for Lane G (and later lanes)

- **F-LD-1 (Lane G):** the columns carry RAW `terminated` / `truncated`; the rust vec env must
  apply `resolve_episode_end` exactly where `MaskableAgentWrapper.step` does today.
- **F-LD-2 (Lane G):** the core produces NO terminal observation. A host whose learner would
  bootstrap a truncation (a non-winprob critic ⇒ `TimeLimit.truncated`) must REFUSE the Rust env at
  startup, not synthesize one.
- **F-LD-3 (Lane G):** a PARKED env (`done` = 1 or 0, `refused` = 1, `need` = 0 0) has no row for
  the learner this op; the rollout loop must handle an env with no decision (step again) — rare
  (a seed-specific start refusal on startup-validated teams), but typed.
- **F-LD-4 (Lane G):** the host's per-episode info (`win_outcome`, `win_draw`, `opponent_class`,
  team WR / PFSP records) was read from `battle1.won`; the core exposes only reward / terminated /
  truncated. Under the production indicator a win is `reward == victory_value`, and a TIE is
  `truncated` without a stall forfeit — which the columns do NOT separate from a forfeit
  (`win_draw` needs a tie flag or the end turn). Not built: flag it if Lane G needs `win_draw`.
- **F-LD-5 (spec):** the key stays named `turn_limit` (Lane C's harness calls `spec_json(turn_limit=…)`)
  but now MEANS the stall threshold; a rename is a cross-lane change for later.
- **F-LD-6 (UNVERIFIED on a live-length battle):** no parity battle reached the production 250-turn
  forfeit; the forfeit rule is verified at thresholds 5 and 45 (same code path, `>=` semantics).
