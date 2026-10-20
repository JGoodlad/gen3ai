# STAGE 1c: more or team-level proposals, importance-resampled by the opponent's own choices (2026-10-20)

Stage 1c of [`design_world_sampler.md`](../../../endstate/design_world_sampler.md) §6.3, registered in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before any world of a registered state was built. The numbers and their
reading go to the design doc, `search_ledger.md`, `UNDERSTANDING.md` and the ledger once read.

**Status: REGISTERED; phase A crawls detached (§2).** No number yet.

| file | what |
|---|---|
| `PREREGISTRATION.md`, `c1c_registration.json` | the arms, the meters, the rules, the bar, the stops; the registered files' sha256 |
| `c1c_common.py` | the constants, the arms as (set, first n) slices, the per-world facts meter (registered) |
| `run_1c.py` | PHASE A's driver: per state 128 F + 128 C_ladder worlds (`dice="rolls"`), the opponent-choice likelihood, the facts, each weighted arm's resample (`opp_likelihood.resample`); `--one-chunk`, `--recheck` (G3), `--smoke`, `--status` |
| `play_1c.py`, `run_play.sh` | PHASE B (only when the bar is met): the identity tooth, then every played arm's 8 worlds through the pilot's driver at the pin checkout |
| `loop_1c.sh`, `launch_1c.sh` | the detached runner: one heavy slot per bounded chunk; the registered stops |
| `read_1c.py` | the registered read: `a` (phase A, `--interim`), `b` (phase B), `bcheck` (phase B's plumbing), `selfcheck` |

## 1. Checkouts and archive

| | |
|---|---|
| DRIVER | `~/.cache/gen3ai/search_1c/driver`: a detached checkout of the commit that ships this unit's code (every row's `head`). Keep it until the read |
| PIN (phase B) | `~/.cache/gen3ai/search_1c/pin_rolls`: X26's pin `9c033489` + the damage-roll control `fc010d7d3` cherry-picked (its tree equals stage 1's play commit `13ffd55c`), its own `src/rust_env` release build |
| archive | `~/gen3ai_archive/search_1c/` (`rows/`, `worlds/`, `recheck/`, `play/<arm>/`, `ws1/` (the corpus cache, a read-only copy of stage 1's), `logs/`, `STOP_AT`, `A_DONE` / `A_STOPPED`, `STOP_B`, `B_DONE` / `B_STOPPED`) |
| inputs (read only) | `~/gen3ai_archive/hidden_info_maturity/` (the bank, the 225M checkpoint copy, roots, truth and banked F rows); `~/gen3ai_archive/search_l3/` (G4); stage 1's corpus cache and `alpha_fit.json` |
| smoke | `~/gen3ai_archive/search_1c_smoke/` (4 non-LEAN states at N = 64; never read as output) |

## 2. Operate

```bash
D=~/.cache/gen3ai/search_1c/driver/designs/research_state/measurements/search_1c_2026-10-20
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
$PY -I $D/run_1c.py --status                   # progress counts only (no meter)
tail -f ~/gen3ai_archive/search_1c/logs/queue.log
bash $D/launch_1c.sh a                          # (re)start phase A from the DRIVER checkout; STOP_AT kept
cd $D && $PY -I read_1c.py a --out result_1c_a.json        # THE READ, once A_DONE / A_STOPPED and the recheck exist
bash $D/launch_1c.sh play                       # phase B, only when result_1c_a.json's bar is met
cd $D && $PY -I read_1c.py b --out result_1c_b.json
```

CPU only (`CUDA_VISIBLE_DEVICES=""`), SCHED_IDLE, `nice -n 19`, `taskset -c 5-7,13-15`, one heavy slot per ≤ 45-min
chunk (root rule 5b).
