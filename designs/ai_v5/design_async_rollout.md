# Design — Non-barrier Async Rollout Collection (ai_v5, `--async-rollout`)

> Status: **implemented + benchmarked** (2026-06-05). Flag-guarded (`--async-rollout`, default off →
> unchanged stock `SubprocVecEnv` + `MaskablePPO.collect_rollouts`). Build log:
> `designs/ai_v5/async_rollout_BUILD_LOG.md`.

## 1. Problem — the per-step barrier

PPO collects rollouts via `env.step(actions)`, which on `SubprocVecEnv` is a **barrier**:
`step_async` sends actions to all N workers, then `step_wait` blocks until *every* worker returns
(`[remote.recv() for remote in self.remotes]`, `subproc_vec_env.py:137`). Consequences on this box
(16 cores, n_envs up to 64; py-spy on a live run: ~86 % of wall in rollout, GPU ~86 % idle):

- **Straggler-gated.** Each step costs `max` over N env latencies, not the mean. A slow battle turn,
  a heavy self-play opponent forward (~70 % of env-worker CPU), or oversubscription jitter stalls
  the whole batch every step.
- **No overlap.** The loop alternates policy-forward (GPU) → `env.step` (CPU) → forward → step; the
  GPU sits idle while envs step and vice-versa. They never run concurrently.

The rollout is **latency-bound**, and the trainer spends most wall-clock waiting. (Box: Ryzen
9800X3D, **8 cores / 16 threads**; `nproc`=16. "core count" below means the 16 logical threads.)

## 2. Key insight — async collection is ON-POLICY for PPO

PPO **freezes the policy during `collect_rollouts`** (no gradient step until the buffer is full —
`ppo_mask.py:221-272`, `train()` only runs after). So stepping envs asynchronously and acting on
whichever are ready does **not** change the policy any action sees — every transition in a rollout
still comes from the *same* frozen policy. This is a pure **scheduling** change (overlap GPU forward
with CPU stepping; drop the max-latency-per-step barrier), **not** an algorithm change like
APPO/IMPALA — those let a learner update while actors run ahead, introducing policy-lag that needs
V-trace/importance correction. Here there is no lag, no correction, no math change. Per-env
trajectories stay contiguous, so per-column GAE is identical.

## 3. Architecture

### 3.1 `AsyncSubprocVecEnv(SubprocVecEnv)` (`async_vec_env.py`)
Adds per-env async stepping + drain-safe RPC; the stock barrier `step_async`/`step_wait` still work.
- `send_step(i, action)` — dispatch ONE env's step (non-blocking), mark it in-flight.
- `poll_ready(idxs, timeout)` — `multiprocessing.connection.wait` on the in-flight pipes → the
  subset ready to read (never blocks on the slowest).
- `recv_step(i)` — read one env's result (from the stash if drained early, else the pipe).
- **Drain-safe `env_method`/`get_attr`/`set_attr`** — the eval callback calls
  `env_method("set_self_play_target"/"set_distill_active"/"opponent_default_stats")` from inside
  `on_step`, which fires **mid-collection** while some pipes hold un-recv'd step results. Sending an
  RPC command then would interleave with those results and desync. The override first **stashes**
  every in-flight step result so the barrier RPC sees clean pipes; the collector reads stashed
  results transparently. (Infrequent — eval collect — so the drain barrier is off the hot path.)

### 3.2 `collect_rollouts_async` (`async_vec_env.py`)
A drop-in for `MaskablePPO.collect_rollouts`, dispatched by `InstrumentedMaskablePPO.collect_rollouts`
when `self._async_rollout` is set and the env is `AsyncSubprocVecEnv`. The loop:

1. **READY phase** — every env that's READY and still needs transitions: batch the obs (dynamic
   size — whatever's ready), one policy forward, store `(obs, action, value, log_prob, mask,
   episode_start)` as pending, `send_step`, mark STEPPING.
2. **WAIT phase** — `poll_ready` for ANY in-flight env (non-barrier). For each ready env: recv,
   write its transition into **its own buffer column** at `filled[i]`, advance its current
   obs/episode-start. A "wave" ≈ a macro-step.
3. End when every column has `n_steps` transitions; set `pos`/`full`; carry `cur_obs`/`cur_estart`
   into `_last_obs`/`_last_episode_starts`; per-column `compute_returns_and_advantage`.

Mirrors the stock loop's bookkeeping exactly: `num_timesteps += wave_size`, the GH-#633 timeout
bootstrap, `_update_info_buffer(wave_infos, wave_dones)`, `_last_*` carry-over, and the final GAE.

### 3.3 Masks ride in the obs (no wrapper change)
`Gen3Env`'s obs is a **Dict** `{"observation": Box(3357), "action_mask": Box(11)}` — the
per-decision mask is *in the obs*, equal to what the stock path reads via `get_action_masks(env)`
(both are `last_ctx.mask` for the obs being acted on). So the collector reads each env's mask from
`obs["action_mask"]` — no per-env `env_method` round-trip, no wrapper change. Writes go to a
`MaskableDictRolloutBuffer` (dict-of-arrays observations, per-column).

## 4. Flag-guard
`--async-rollout` (default **off**). On → `EnvClass = AsyncSubprocVecEnv` (not `--debug`, which is
single-env `DummyVecEnv`) + `model._async_rollout = True` (routes `collect_rollouts`). Off → the
stock `SubprocVecEnv` + `MaskablePPO.collect_rollouts`, byte-for-byte unchanged. Resumed runs get it
too (`load_model_snapshot` returns an `InstrumentedMaskablePPO`).

## 5. Results — FPS (bridge transport, GPU forward, steady-state Δts/Δte)

**Round 1 — equal-envs + right-sizing (16-core box):**

| Config | n_envs | steady FPS | |
|---|---|---|---|
| sync | 16 | 1024 | baseline |
| **async** | 16 | **1229** | **1.20× vs sync@16** |
| sync | 32 | 1365 | |
| **async** | 24 | **1536** | 1.12× vs sync@32; **fastest of the four** |

At n_envs = core count (16) async is **+20 %** — it reclaims the per-step straggler bubble and
overlaps the GPU forward with CPU stepping. `async@24` is the fastest config overall and beats
`sync@32` with fewer envs (less RAM, fewer bridge children, less jitter).

**Round 2 — production-relevant (vs the current `--n-envs 64` sync = 1489 FPS):**

| Config | n_envs | steady FPS | vs sync@64 |
|---|---|---|---|
| sync | 64 | 1489 | baseline (current production) |
| **async** | 32 | **1638** | **1.10×** — matches/beats production with **half the envs** |
| **async** | 48 | **1676** | 1.12× |
| **async** | 64 | **1695** | **1.14×** — highest FPS measured |

Async beats sync at **every** comparable point — **+20 %** at n_envs = core count (16), **+10–14 %**
at production scale. Two ways to bank it:
- **Max FPS:** flip the flag on the current config → `--async-rollout --n-envs 64` is **+14 %**
  (1489→1695) for one flag, no other change.
- **Same FPS, half the footprint:** `--async-rollout --n-envs 32` already **matches/beats** production
  `sync@64` (1638 vs 1489) with **half the env workers + bridge children** → roughly half the
  per-worker RAM and far less oversubscription jitter. (The 64→128-process oversubscription and its
  RAM are real operational costs — see the throughput/RAM notes in `src/agents/training/CLAUDE.md`.)

Since async@N ≥ sync@N at equal n_envs (identical `train()` time cancels in the comparison), the
ratios are a clean read of the **rollout** speedup diluted by the shared gradient-update time — the
rollout-only speedup is larger; end-to-end FPS (what's tabulated) is the number that matters.

Method: bridge transport (serverless, deterministic), GPU policy forward, `n_steps=256`, `n_epochs=4`,
8 rollouts/config, steady-state FPS = Δtimesteps/Δtime over the post-warmup rollouts (SB3's cumulative
`time/fps` is dragged down by the ~one-time bridge warmup). Single runs (no repeats) at integer-second
log resolution — treat ±~5 % as noise; the ordering (async > sync everywhere) is consistent across
both rounds.

> **Regime caveat (important).** These runs used **heuristic** opponents (no `--self-play`), so the
> per-step cost is light and the workload is comparatively CPU-bound (more envs kept helping:
> sync 16→32→64 = 1024→1365→1489). The **production self-play** regime is heavier per step (the
> opponent NN forward is ~70 % of env-worker CPU) and the py-spy profile found it **latency-bound
> with ~24 % idle CPU at 48 envs and an n_envs plateau** (`project_throughput_profile`). That idle,
> straggler-induced CPU is *exactly* what async reclaims, so the self-play gain is plausibly **at
> least** as large as measured here — but it is **unconfirmed**; re-run this benchmark with
> `--self-play` (seeded pool) before quoting a production self-play number.

## 6. Limitations / honest notes
- **Not a free lunch at heavy oversubscription.** Both sync and async are ultimately CPU-bound at
  `~cores / E[step]`; async reclaims the *barrier bubbles* (straggler variance + forward/step
  serialization), which is largest near n_envs ≈ cores and shrinks as cores saturate. The win is
  real but bounded — the bigger CPU lever is reducing per-step cost (opponent **distillation**,
  already in the codebase).
- **Dynamic-size forwards.** Async forwards whatever subset is ready, so batch size varies (the
  benchmark confirms the net effect is still a win — waves are typically many envs, not batch-1).
- **`train()` idle is unaddressed.** Env workers still idle during the gradient update; overlapping
  the next rollout's collection with `train()` (double-buffering) is a separate, larger change.
- **Targets the Dict-obs + MaskableDictRolloutBuffer path** (Gen3Env). The collector asserts that
  shape; other obs layouts fall back via the flag-guard (default sync).

## 7. Recommendation
- **Turn it on.** `--async-rollout` is a strict FPS win at every n_envs tested and is fully
  flag-guarded (default off; zero change to the sync path), on-policy-correct, and regression-clean
  (110 targeted + full project unit suite green; real-`Gen3Env` smoke EXIT=0).
- **Sweet spot:** `--async-rollout --n-envs 32` gives ~production FPS (1638 ≥ sync@64's 1489) at
  **half the env/bridge footprint** — the best FPS-per-RAM. For **max throughput**, `--async-rollout
  --n-envs 48–64` (+12–14 %). Avoid pushing n_envs far past cores: returns diminish and RAM/jitter
  grow.
- **Compounds with distillation.** Async attacks the *scheduling* loss (barrier bubbles); opponent
  distillation attacks the *per-step CPU* (the ~70 % opponent forward). They stack — run both for the
  largest gain.
- **Next lever (out of scope here):** overlap the next rollout's collection with `train()`
  (double-buffering) to reclaim the gradient-update idle — a larger change, tracked separately.
