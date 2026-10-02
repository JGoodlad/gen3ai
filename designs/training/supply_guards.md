# Supply guards — every live lever's supply is a DECLARED resource (`gen3_supply_guard_v2`)

**Owner of:** `agents/training/lever_supply.py` (the shared guard), `agents/training/selfplay_supply.py`
(the self-play pool + PFSP mixin), the guard wiring in `team_pfsp_callback.py`, `win_prob_callback.py`,
`fork_callback.py` and `teacher/callback.py`, the `--bot-weights` / distill-team / warm-start refusals
in `main/train/matchup_setup.py` and `agents/training/warmstart.py`, and `main.exit_codes.FatalConfigError`.
The cf label producer's own guard (`gen3_supply_guard_v1`) is
[`cf_grounding.md`](cf_grounding.md) § *THE SUPPLY IS A DECLARED RESOURCE*; this doc is the rest.

## Why

The supply inventory of 2026-09-30 looked for one shape across the trainer: **a flag is set, the
mechanism behind it silently delivers nothing, and the run then reads as a result about that
lever.** v1 closed the cf label producer (`ai_v12_12_ladder_cflabels`: 10M steps at
`--cf-winprob-coef 0.5`, zero labels). v2 closes the rest of that inventory (the `win_prob_rollout` lever it also covered was deleted with the R-rollout target, deletion pass L2):

| lever (`--supply-starve-cycles` key) | what used to happen silently | declared floor |
|---|---|---|
| `self_play_pool` — `--self-play` | a pool that never seeds → every episode falls back to the BOT pool | 3 eval cycles |
| `pfsp` — `--pfsp-scale` | no measured sentinel win-rate → the pool sample stays uniform (or stale) | 3 eval cycles |
| `team_pfsp` — `--team-pfsp` | no self-play / exploiter team game → team sampling stays uniform | 5 updates |
| `fork` — `--fork-fraction` | the arm DISABLED itself with a print (no fork buffer / missing obs key / no handle / no ring), or every pass failed | 5 rollouts |
| `search_teacher` — `--search-teacher` | selection returns no candidate (no loss traces) → the AWR / OPD terms fold nothing | 3 teacher cycles |

Plus three STARTUP refusals that exited 1 (CRASH, which the launcher restarts into the same error)
or did not exit at all:

| refusal | before | now |
|---|---|---|
| `--bot-weights` typo / non-number / negative / all-zero | `sys.exit(1)` or a bare `ValueError` → CRASH; the launcher restarted it until its rapid-crash breaker (3) | `BotWeightsRejected` → `FATAL_CONFIG` (3) |
| `--warmstart-consensus` failure (teacher resolution, battles, BC) | an uncaught exception → CRASH. The warm-start is rebuilt from scratch on every restart, and one that failed AFTER the launcher's 10-minute rapid-crash window reset the breaker each time — **an unbounded crash loop** | `WarmstartFailed` → 3; a partial `warmstart_consensus.zip` is removed |
| a `--distill-teacher` team that fails gen3ou validation | `Gen3Teambuilder` DROPPED it silently: never drawn, its `distill_mask` never fires, its `distill/*` scalars never appear. Measured on the parent commit: the run went to completion, rc 0 | `DistillTeamRejected` → 3, naming the file and the validator's reason |

And two startup refusals for a PFSP flag with no possible supply (`combination_checks`, so
`checkargs` sees them): `pfsp_scale_needs_self_play`, `team_pfsp_needs_self_play_or_exploiter`.

## The mechanism

**Two failure classes, two exit codes, neither restarted by the launcher.**

* **A deterministic mis-wiring** — the same on every restart — raises `LeverConfigError`
  (a `main.exit_codes.FatalConfigError`) → **`FATAL_CONFIG` (3)** the first time it is seen.
  `FatalConfigError` is mapped by NAME in `exit_codes._FATAL_BY_NAME`, so it can be raised from
  anywhere (a callback, the startup path) and both fail-fast handlers and the FRESH-path `learn()`
  handler exit 3 through `exit_code_for`.
* **A dry streak** — the lever is LIVE (supposed to be delivering) and delivered ZERO units for its
  floor of consecutive cycles — raises `LeverStarvedError` (a `SupplyStarvedError`) →
  **`FATAL_SUPPLY` (5)**.

`DryStreakGuard` is the one state machine: `observe(delivered, live=..., why=...)` once per cycle;
a delivery resets the streak; a cycle with `live=False` is counted (and reported) but neither arms
nor breaks a streak. `--supply-starve-cycles key=N[,key=N]` overrides a floor; `key=0` disables
that lever's FATAL, which is **announced at training start**, and the end-of-segment summary is
still LOUD at zero. A typo in the spec is a parser error (`supply_starve_cycles_parses`).

**Every supply line reaches the run's own log.** `lever_supply.loud()` prints to stdout (the
launcher's child log) AND sends a launcher event. `main.launcher.ipc.emit` alone only does the
second under the launcher — which is why the old "pool is EMPTY at startup" warning never appeared
in any run directory (audit 2026-09-30, finding F5; that warning now uses `loud` too).

**End of every segment:** each guard prints `🚨🚨 [SUPPLY] ZERO … This segment is NOT evidence
about that lever.` when a live lever delivered nothing, else its total.

## What "live" means, per lever

| lever | one cycle | delivered | live when |
|---|---|---|---|
| `self_play_pool` | one COLLECTED eval cycle (`SelfPlayCallback._collect_pending`), **failed cycles included** | 1 if the pool is non-empty after the cycle | always (`--self-play` is the declaration) |
| `pfsp` | the same cycle | sentinel win-rates measured | the cycle launched ≥ 1 sentinel (an empty pool is the pool guard's job) |
| `team_pfsp` | one `TeamPFSPCallback` update (every 3 rollouts) | team games counted on POOL teams | `--exploiter`: always. `--self-play`: the pool is seeded AND the persisted `self_play_fraction` > 0 (`callbacks.team_pfsp_live_probe`) |
| `fork` | one rollout | fork rows injected (a failed pass = 0) | `--fork-fraction` > 0 |
| `search_teacher` | one selection (crater / winprob_oneply); one refresh window in `--teacher-persistent` | candidates selected (a failed selection = 0); corrections ingested (persistent) | always |

**Why the self-play floor counts a gate that never opens.** The pool seeds only when
`win_rate_vs_bots` ≥ `--self-play-start-wr` (0.55). A run that never clears it is not starved by a
broken supplier — the bots-only regime is ENDOGENOUS (audit 2026-09-30: `ai_v12_27`'s bots win rate
read 0.148 → 0.531 over its five cycles). It is still a run whose argv says self-play and whose every
game was against bots, which is exactly what a reader of its result will not assume. The FATAL's
message names the gate and both remedies: lower `--self-play-start-wr`, or drop `--self-play` when a
bots-only run IS the experiment (or `--supply-starve-cycles self_play_pool=0`, announced). Every v9+
fresh run in the archive seeded at its FIRST eval cycle (2M), which is what sizes the floor at 3.

**Run-level, not segment-level, for the long cycles.** An eval cycle and a teacher cycle are ~2M
steps; a launcher segment is ~3 h. A per-process streak would reset at every restart before a floor
of 3 could trip. So the self-play / PFSP counters ride the pool's `summary.json` (key
`supply_guard`, stamped with THIS run dir — a fork's pool is seeded with its parent's summary, and
inheriting the parent's streak would judge the fork by cycles it never ran), and the teacher's ride
`<run>/teacher_cycle/supply_state.json`. Both are written BEFORE the judge raises. The rollout-cycle
guards (`team_pfsp`, `fork`) are per process: their floors are ~0.5–1.5M steps.

**The FATAL is never raised from a graceful drain.** `SelfPlayCallback._on_training_end` drains the
in-flight eval cycle with `_draining` set: the cycle is counted and the summary is loud, but a
completed run is not turned into exit 5 from inside its own shutdown path.

## TensorBoard

`supply/selfplay_pool_dry_streak`, `supply/pfsp_dry_streak`, `eval/failed_cycles_total`,
`supply/team_pfsp_dry_streak`, `supply/fork_dry_streak`,
`supply/search_teacher_dry_streak`.

## Honest limits

* **Arrival, not quality.** A guard checks that the lever delivered SOMETHING; a fork arm injecting
  one row per rollout passes. The levers' own meters (`fork/*`,
  `teacher/yield`, `eval/pfsp_*`, `team_pfsp/n_measured`) still own quality.
* **The search teacher guards CANDIDATES, not corrections.** A cycle that selects candidates and
  confirms none is a fact about the policy (the composition test measures ~15–25% per-candidate
  conversion on a small budget, so three dry cycles there would be a ~5% flake); a zero-correction
  segment is still LOUD in the end-of-segment summary.
* **A pool that seeds and then regresses below the gate** (non-empty pool, `self_play_fraction` 0)
  is the curriculum working as designed and is NOT a FATAL; it is not live for `team_pfsp` either.
* **`--debug` without `--debug-eval` runs no eval**, so no `SelfPlayCallback` and no pool guard.

## Tests

`agents/training/lever_supply_test.py` (the guard, the exit mapping, the launcher on 1/3/5, the pool
and PFSP guards through a real `_collect_pending`, run-level persistence across a simulated restart,
a fork's inherited counters ignored, failed cycles, the drain, team-PFSP liveness, the teacher on
empty and failed selections, `--bot-weights`, the warm-start wrap); the fork guard in
`fork_callback_test.py`; `agents/training/lever_supply_integration_test.py`
with REAL processes — the trainer exits 3 on a `--bot-weights` typo (rc 1 on the parent commit) and
on an illegal teacher team (the real Node validator; rc 0 on the parent commit), and (slow, sim) the
`--debug` trainer exits 5 on a self-play run that can never seed and on a search teacher with no
eval traces.
