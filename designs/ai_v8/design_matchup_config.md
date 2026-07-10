# Design — MatchupSpec: one explicit config for training/eval matchups (teams, opponents, play modes)

**Status:** SCOPED 2026-07-09 (user-requested), not built. Priority order at the bottom.
**Motivation:** one week, four independent failures, one shared root: *the matchup a run plays is
assembled implicitly across seams that nothing forces to agree.*

## The failure catalog (all observed 2026-07-02 → 07-09)

| # | failure | seam that drifted |
|---|---------|-------------------|
| 1 | **Eval-OOD teams** — `--trainee-team` runs were evaluated piloting random pool teams (ai_v7_05–08 "plateau at 7–16%" was measurement, not training) | `eval_worker` rebuilt its own default teambuilder; the run's pin never reached the worker cfg (fixed `182bb4d`) |
| 2 | **The training mirror** — `--trainee-team` silently pinned the OPPONENTS too; every specialist run trained a single-team mirror vs bot pilots (genuine ~100% training WR, fake curriculum) | PokeEnv's single `team=` feeds BOTH env agents; the rotated opponent Players are decision-functions whose own builders are inert (fixed `0f16bcd` + hotfix) |
| 3 | **Noise-farming** — exploiter "beat" its frozen target 95% in training vs 27% greedy-greedy (~20pts from the stochastic target, the rest was #2) | training plays stochastic@temp, eval plays greedy; controllers (the temp ratchet) keyed on the stochastic number |
| 4 | **Crash-restart resolved a FRESH run's "latest checkpoint" to `models/_goldens/ai_v3_…`** (observed at the ai_v7_10 launch crash) | launcher checkpoint discovery is not strictly run-dir-scoped when the run has no checkpoint yet |

Plus the meta-failure: a shipped `SyntaxError` in `train_rl_agent.py` reached main because entry-point
modules are imported by no test (now gated by `src/main/syntax_test.py`).

## The core idea: declare the matchup once, thread it whole, verify it realized

A single frozen dataclass, built ONCE in `train_rl_agent.py` from the CLI and then **consumed**
(never re-derived) by every party — the `plan.json` pattern the eval sharding already uses ("the
parent writes the single source of truth; workers read it, so they can't drift").

```python
@dataclass(frozen=True)
class MatchupSpec:                       # serializable; hash goes into provenance
    # -- teams --
    trainee_teams: TeamSource            # pool | pinned(file) | biased(pool, pin, prob)  ← team-prob lives here
    opponent_teams: TeamSource           # independent of trainee_teams BY CONSTRUCTION
    # -- training opponent mix --
    mix: OpponentMix                     # bots(weights) | self_play(shares…) | exploiter(target, keep_bots, frac)
    opponent_play: PlayMode              # greedy | stochastic(temp | schedule)   ← the ratchet/anneal plugs in here
    # -- eval --
    eval_trainee_teams: TeamSource       # defaults to trainee_teams (the #1 fix, made structural)
    eval_opponents: tuple[EvalOpponent]  # bots + sentinels + ext targets, EACH with its own PlayMode
```

Threading: `MatchupSpec` → env factory (`team=` + `opponent_team=` both from the spec), wrapper
(mix), both eval callbacks → worker cfg (`spec.eval_*`), and `metadata.json` (full spec + hash).
CLI flags become *constructors of the spec*, not free-floating knobs — `--trainee-team`,
`--trainee-team-prob`, `--exploiter*`, `--stable-opponent-temp`, `--bot-weights` all funnel in.

## The guards (what makes it robust rather than just tidier)

1. **Realized-matchup fuzz (P0, the #2-class killer).** Generalize this week's win-attribution probe
   into a permanent fuzz test: play real bridge episodes through the REAL factory+wrapper, parse the
   protocol stream, and assert the REALIZED p1/p2 teams + opponent identity match the spec
   (`matchup_spec_fuzz_test.py`). This is the test that could not have existed against implicit
   config — the spec gives it something to assert against.
2. **Startup echo (P0).** `emit()` the resolved spec at launch (one block: trainee teams / opponent
   teams / mix / play modes / eval sources) so a mis-assembled run is visible in the Events panel
   before burning GPU-days. The current per-feature emit lines (`🎯 [SPECIALIST]`, `🥊 [EXPLOITER]`)
   fold into it.
3. **Provenance + regime tags (P1).** Stamp `spec_hash` into `metadata.json:latest_eval` and every
   `eval_results.jsonl` row. ELO/history aggregation only pools rows with compatible measurement
   regimes — the #1 bug also poisoned weeks of recorded ELO; tags make stale history self-identifying.
4. **Controllers key on eval, not training (P1, the #3 fix).** `ExploiterTempRatchetCallback` (and
   any future WR-keyed gate: promotion, mastery) gains a `signal: training|eval_greedy` knob,
   defaulting new runs to `eval_greedy` — controllers should ratchet on the yardstick, not the
   noise-farmable training number. Metric names grow explicit mode suffixes where ambiguous
   (`…_wr_greedy` vs `…_wr_stoch`).
5. **Launcher checkpoint discovery hardening (P1, the #4 fix).** `find_latest_checkpoint` must be
   strictly run-dir-scoped; a fresh run that crashes pre-checkpoint is FATAL (propagate), never
   resolved to anything outside `<run_dir>/checkpoints/`. Add a regression test with an empty run
   dir + a decoy `models/_goldens/`.
6. **Already landed this week:** eval-team threading (`182bb4d`), the `opponent_team` env seam
   (`0f16bcd`+hotfix), `eval_worker_test` / `gen3_env_test` pins, the repo-wide syntax gate.

## Non-goals (for now)
- No behavior changes hidden inside the refactor: the spec must reproduce today's runs byte-identically
  from the same CLI (pin with a golden-args → spec snapshot test).
- Per-opponent team pools / archetype-conditional sampling — the spec's `TeamSource` leaves room, not built.
- Migrating poke-env's `PokeEnv(team=…)` upstream signature — the post-init seam is sufficient.

## Priorities
- **P0** (before the next multi-day run beyond ai_v7_10): the dataclass + threading + realized-matchup
  fuzz + startup echo. ~1 day of focused work, mostly mechanical consolidation.
- **P1**: regime tags, controller keying, launcher discovery hardening. ~half day.
- **P2**: metric renames, golden-args snapshots, per-opponent team sources.
