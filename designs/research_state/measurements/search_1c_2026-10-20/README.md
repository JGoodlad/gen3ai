# STAGE 1c: more or team-level proposals, importance-resampled by the opponent's own choices (2026-10-20)

Stage 1c of [`design_world_sampler.md`](../../../endstate/design_world_sampler.md) §6.3, registered in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before any world of a registered state was built. The numbers and their
reading go to the design doc, `search_ledger.md`, `UNDERSTANDING.md` and the ledger once read.

**Status: REGISTERED, with AMENDMENT A (PREREGISTRATION §11: the orchestrator's ruling on the independent review; a STRICT corpus read, the L-vs-dice attribution co-primaries, a permuted-L placebo that stops the read, PROVISIONAL-POOL / SELF-PLAY-ONLY labels, Holm, NOT-DETECTED's meaning, an ESS split); phase A crawls detached (§2).** An INTERIM read exists (§4); no verdict yet.

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

## 3. The run (operational log; none of it changes the design)

- **Launched** 2026-10-20 04:41 PDT from `664385776`; `STOP_AT` = 2026-10-21 10:41 PDT (registered, kept by every
  relaunch). Stopped by its owner at 05:22 while still waiting for its first heavy slot (no state built) to register
  AMENDMENT A; **relaunched** 05:29 from `a144482b0`.
- **First chunk** (06:26–06:59): 27 states, ≈ 90 s a state (27–245 s), peak 6.3 GB of the 16-GB cap, 0 defects, 0
  exclusions, 0 parked.
- **THE ORCHESTRATOR'S RULING (b), 2026-10-20 08:36 PDT: the chunks run BLOCKING** (`GEN3AI_BLOCKING="stage 1c
  (P0.1)"`, bypassing the heavy-slot queue, then 17 deep). Stage 1c is the owner's P0.1 study, and most of the queue
  was lower-priority work. It is still `nice -n 19`, `taskset -c 5-7,13-15`, SCHED_IDLE, under `mem_cap`, ONE chunk at a
  time, with the 45-min chunk shape and the registered stop. It changes throughput, not the design: the done states are
  still a prefix of the registered order. The waiting loop was stopped between chunks (no state in flight) and
  relaunched with the mark at 08:36.
- **10:24: the chunks' scheduling class, SCHED_IDLE → nice 19 (SCHED_OTHER).** `loop_1c.sh` wraps each chunk in
  `chrt --idle` (stage 1's habit), which yields even to other agents' nice-19 jobs: wall time ran 2.5–3× CPU time
  (≈ 4–6 min a state). The ruling asks for `nice -n 19` on the agent cores, still far below the trainer (nice 10). The
  loop's bash was stopped between states (its chunk left to finish, still SCHED_IDLE: an unprivileged process cannot
  leave it), and an operational wrapper, `~/gen3ai_archive/search_1c/ops/loop_a_nice19.sh` (the same loop and recheck,
  the same driver checkout's `run_1c.py`, so every row's head stays `a144482b0`), runs the next chunks at nice 19.

## 4. THE INTERIM READ (registered §7: about 12 h after launch, descriptive; NO verdict, NO bar; it changes no rule, n or stop)

Made 2026-10-20 16:42 PDT on the done prefix: **236 of 287 rows, 234 read** (2 exclusions, 0 defects, 0 parked).
[`result_1c_a_interim.json`](result_1c_a_interim.json), by `read_1c.py a --interim` from the driver checkout. Gates:
G1 (a prefix) and G2 pass; G5 recomputes all 1,872 stored resamples exactly; G3 (the recheck) runs after phase A.
**G4 (descriptive): 0 of 232 truth likelihoods equal L3's exactly, and all differ negligibly** (median |Δ| 4e-6 nats,
max 6e-5; the same scored steps). That is the expected trace of the fp32_alt τ solve the driver commit inherited (config
v160, `d4af9f1e2`; X26's pre-v160 checkpoint migrates to it). The worlds' own log-likelihoods differ by ≈ 1 nat.

Pooled, the amended reading (corpus arms STRICT), Δ against F8 (95 % CI, n = 234):

| arm | ΔH (binding facts held) | ΔM_hit (unrevealed moves) | ΔU_hit (never-seen species) | q_SM (exact-world mass) |
|---|---|---|---|---|
| F128L | +0.007 [−0.000, +0.015] | +0.010 [−0.000, +0.019] | +0.007 [−0.008, +0.023] | 0.0000 |
| C128L | +0.018 [+0.010, +0.027] | +0.003 [−0.007, +0.013] | +0.100 [+0.075, +0.125] | 0.0001 |
| M256L | +0.013 [+0.006, +0.020] | +0.007 [−0.002, +0.016] | +0.051 [+0.032, +0.070] | 0.0000 |

- **The corpus proposals carry the species, not the likelihood:** C8 − F8 alone (8 uniform corpus worlds) is ΔH
  +0.019 [+0.012, +0.025].
- **The likelihood's own selection is small:**
  - F128L − F128d +0.007 [+0.001, +0.013];
  - C128L − C128d +0.004 [−0.002, +0.010].
- **The exact-world mass stays ≈ 0 in every arm** (≤ 0.0003). The every-move-exact share rises 0.012 (F8) → 0.018
  (F128L).
- **The ESS of the likelihood-weighted 128** is ≈ 18–19 (9–13 % of states below 2).
- **The N-scaling (128 − 64) is ≈ 0.**
- **By stratum, every lift is in the SNAPSHOT stratum:**

  | | snapshot | bot |
  |---|---|---|
  | F128L | +0.019 | −0.005 |
  | C128L | +0.035 | +0.002 |
  | the likelihood's selection, F | +0.014 [+0.008, +0.020] | −0.000 |
- **The leak (A1) is small:**
  - 10 % of states hold a near4 C proposal;
  - L puts 0.0014 of its weight on them;
  - C128L's ΔH without those states is +0.017.
- **The A3 placebo** reads not NEARER on every arm (F128L's +0.004 [−0.001, +0.009] is the largest).

Read as an interim only: the registered verdict and the bar are made once, at `A_DONE` or the stop, after G3.
