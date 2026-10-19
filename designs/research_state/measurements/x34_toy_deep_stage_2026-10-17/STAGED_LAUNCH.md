# X34, the toy deep run: STAGED launch (2026-10-17)

**Status: STAGED, not launched.** It needs no GPU until launch. It launches when X26 ends or is stopped and the GPU
is free. The spec is [`design_compute_efficiency.md`](../../../endstate/design_compute_efficiency.md) §4, and its
backlog row is `EXPERIMENT_BACKLOG.md` X34. The Training session runs [`launch_x34.sh`](launch_x34.sh), which repeats
every pre-check below and stops with HOLD (exit 7) before creating anything if one fails.

**What runs.** A FORK of `rb_toy_ent_s4002`:
- **From:** `final_model.zip`, its 100M cap, at step 100,081,428 (sha256 `60af6b88…5648fb`).
- **Into:** a NEW run directory, `rb_toy_ent_s4002_deep`.
- **Code:** PINNED at main, not at the toy's pin `3899277a`. The run must carry T36's canary fix `5236eb22` (F10;
  the pin draws the false-alarm lottery I30) and the memory fix `9c033489`.
- **Stop:** X26's own Tier-1 rule (Δ 10M, W 50M, two consecutive FLAT checks), with a CAP at 500M.

## 1. The network is the same network at HEAD (PROVEN)

The toy's argv typed 17 legacy OFF flags that U1 and U3 deleted. At HEAD those flags make `checkargs` exit 1, with 17
`unrecognized` ([`checkargs_pin_argv_at_head.txt`](checkargs_pin_argv_at_head.txt)). Dropped:
`--move-resolution off --speed-physics off --op-reduction max --switch-hazard-cost off --eot-residual off
--move-resolution-facts off --status-facts off --ko-ramp ramp --g-ledger coarse --effective-stats off
--no-intent-move-cell --no-intent-threshold --no-intent-conditional --no-pair-outcome-cell --no-pair-outcome-switch
--no-switch-branch-cell --no-conditional-threat-cell`.

Every one of them is VACUOUS for this network. The toy recorded `damage_op` false and `opp_intent` false, so under
`adopted_levers` D-LD2 none of those levers was ever built. What is left is
[`toy_argv_head_fresh.txt`](toy_argv_head_fresh.txt), and `checkargs` passes it at HEAD (exit 0,
[`checkargs_head_fresh.txt`](checkargs_head_fresh.txt)). Its arch flags equal `adopted_levers_test.TOY_ARGV`.

How identity was proven, with [`identity.py`](identity.py), on CPU, one torch thread, rule 5b:

| check | HEAD `6dcc769c` (reduced argv) | pin `3899277a` (`git archive` export, the toy's full argv) |
|---|---|---|
| network built from the argv vs `final_model.zip` `policy.pth` | 316 / 316 keys, no key only on one side, no shape difference, 3,121,590 elements (1,423,748 parameters) | the same |
| strict `load_state_dict` of the checkpoint | OK | OK |
| the trainer's own fork loader (`load_model_snapshot`, version check and sanitizer) | loads; weights byte-equal to the zip | the same |
| forward on 191 real Rust-core observation rows (`play_rows(6, seed=34)`), logits (191 × 11) and values | — | **bit-identical to HEAD** (max \|d\| 0.0; logits sha `82daa135…` on both) |
| the same on the 64-row fixture | — | bit-identical (sha `fad5fccd…`) |

The raw outputs are in [`identity_head.txt`](identity_head.txt) and [`identity_pin.txt`](identity_pin.txt). The forward
is the fp32 eager CPU forward. The training forward (CUDA, compiled, a bf16 trunk in R1) is judged at startup by the
compile gate on the trained weights, and by the canary after that.

**A regression test pins it:** `src/agents/model/toy_network_identity_test.py`, CPU, about 2 s.
- (1) HEAD's toy argv builds exactly the checkpoint's key and shape manifest
  ([`toy_s4002_state_dict_manifest.json`](toy_s4002_state_dict_manifest.json)).
- (2) The real checkpoint, loaded strictly at HEAD, reproduces the pin's logits and values on 64 recorded rows
  ([`toy_s4002_forward_ref.npz`](toy_s4002_forward_ref.npz); bar 1e-5, measured 0.0). It is skipped where the
  archive is absent.
- **It fails on revert:** with `--trunk-layers 3` in `TOY_ARGV`, both tests fail (the key set, and the strict load).

## 2. The argv

The launch argv is [`fork_argv_minimal.txt`](fork_argv_minimal.txt). It types nothing that is inherited:

```
--model models/rb_toy_ent_s4002/final_model.zip --steps 500000000 --seed 4002 --checkpoint-every-steps 1000000 \
  --snapshot-ladder-games 0 --device cuda --run-name rb_toy_ent_s4002_deep --supply-starve-cycles self_play_pool=0
```

The launcher flags are `--restart-interval-hours 24 --pin-commit <PIN>` (24 h is the X26 decision of 2026-10-17).
- **The architecture** is INHERITED from the parent's `model_config.json`. On a fork, the launcher passes the arch
  surface through untouched.
- **The recipe** is RESTORED from the parent's record by the fork-recipe fix: 12 knobs passed TYPED (`n_envs=256
  rollout_target_samples=98304 trainee_slots=1 t2_lanes=0 grad_accum_steps=32 n_epochs=10 min_lr=1e-05
  weight_decay=1e-05 clip_range=0.15 clip_range_vf=None ent_coef=0.05 self_play=True`).
- **Inherited:** the seven zero supervision coefficients and `trunk_precision bf16`.
- **SB3 restores** `n_steps 384` and `batch 2048` from the checkpoint. The parser's 2048 / 4096 is INERT, as on
  s4001b, whose metadata recorded 384 / 2048.
- **The typed variant** [`fork_argv_typed.txt`](fork_argv_typed.txt) (the toy's arch flags and coefficients typed, the
  s4001b style) resolves to the SAME effective config. The dry-runs differ only in "37 vs 69 unset flags INHERITED"
  and in "(TYPED)" vs "(inherited)" on the seven coefficients.
- **The shared compile cache is ON by construction.** A pinned worktree is a clean tree. The dry-run counts the
  shared cache (7.77 GiB headroom of 20 GiB).

`checkargs --pin 6dcc769c` (exit 0, [`checkargs_fork_6dcc769c.txt`](checkargs_fork_6dcc769c.txt)):

```
  parser                         : the CURRENT tree's — --pin names HEAD (6dcc769c)
  accepted by the trainer parser : 8
  resolved against the FORK PARENT: /home/goodlad/dev/gen3ai/models/rb_toy_ent_s4002/model_config.json
      69 unset flag(s) INHERITED from it — the checks below read the EFFECTIVE config, not the argv
  [ForkLR] ✓ the fork parent (rb_toy_ent_s4002) did not run at a frozen LR — the inherited rate is an ordinary annealed one
  retired-lever parent             : none (no --model, or no deleted lever was ON)
  ✓ this command still launches
```

`launcher --dry-run --restart-interval-hours 24 --pin-commit 6dcc769c` (exit 0,
[`dryrun_fork_6dcc769c.txt`](dryrun_fork_6dcc769c.txt)):

```
  role        : FORK of /home/goodlad/dev/gen3ai/models/rb_toy_ent_s4002
  recipe      : 12 untyped recipe knob(s) restored from the PARENT's record and passed to the child TYPED:
                n_envs=256  rollout_target_samples=98304  trainee_slots=1  t2_lanes=0  grad_accum_steps=32  n_epochs=10  min_lr=1e-05  weight_decay=1e-05  clip_range=0.15  clip_range_vf=None  ent_coef=0.05  self_play=True
  run dir     : /home/goodlad/dev/gen3ai/models/rb_toy_ent_s4002_deep   [would be created]
  pin         : 6dcc769c4539096d6544683fa262ecc980b60029  (source: pin_commit)
  steps       : --steps 500,000,000 vs checkpoint at 100,081,428 (sidecar) → +399,918,572 steps
  interpreter : /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3  (torch 2.8.0+cu126) — carries the run's recorded torch
  restarts    : every 24.0h, grace 20.0 min, max 3 crash restart(s), nice 10
    trunk       : bf16 — the learner's trunk rounds run under bf16 autocast in R1 (T2 / eval / the K9(b) probe stay fp32)
  disk space  : ✓ 319.46 GiB free on /home/goodlad/dev/gen3ai/models vs 48.27 GiB REQUIRED
  ✓ DRY RUN — this command would launch. Nothing was created or modified.
```

## 3. Pre-checks (`launch_x34.sh` runs them all; any failure is a HOLD)

1. X26 has ended or been stopped, and the GPU is free under the Training session's own lease and window rules. This
   one is the session's call; the script does not check it.
2. The PIN contains `5236eb22` (T36) and `9c033489` (the memory fix).
3. `models/rb_toy_ent_s4002_deep` does not exist yet.
4. The parent zip's sha256 is unchanged.
5. `checkargs --pin PIN` exits 0.
6. `launcher --dry-run` says "would launch", the role is FORK of s4002, and the delta is +399,918,572 steps.
7. `toy_network_identity_test.py` passes at main with 2 passed (a skip is not a pass).

The default PIN is the validated `6dcc769c`. A newer main is fine once the same checks pass on it, so pass it as
`$1`.

## 4. The startup banners to expect (from s4001b's fork and X26's resume)

- `🧮 [TRUNK PRECISION] bf16: …`
- `🧊 [CompileCache] SHARED-NEW …/compile_shared/<key> — clean tree, … code <PIN> …`. The first launch at a new code
  and config is cold. **`NOT SHARED` means the tree is not clean: investigate.**
- `🌱 [SELFPLAY] [pool] seeded 20 snapshots + metadata from rb_toy_ent_s4002 …`. **A fork without this line is a
  FATAL_SUPPLY risk.**
- `▶️ Resuming at LR …, epochs 10`. The parent ended at an lr_now of 3.0e-4.
- `[CompileRegions] parity PASS (bf16 trunk: the bf16 rule) — R1 [trained weights] … per-param worst ratio … <= 12 …;
  clip margin: N/2048 boundary rows relocated …`. **The `clip margin` clause is T36's fix (`gen3_r1_clip_margin_v1`).
  Its absence means the pin lacks T36.** A per-param refusal (> 12) follows the startup-gate refusal procedure
  (`bc5a91fd`, `design_x26_readiness.md`).
- `🐤 [CompileCanary] armed: … at update 10, then every 100 updates …; a disagreement is confirmed before it FATALs`
- `🛡️ [SUPPLY] --self-play: …`, then `🧊 [LEARNER FREEZE] the first rollout of learn(): …`. The toy has no ride-along
  heads, so there is 1 optimizer.
- The first `🧪 Self-play eval @ 102,…`. s4002 ended at ≈ 87–94 % against the bots.

## 5. Watch and tick plan

- **Throughput:** ≈ 4,800–5,100 fps, so ≈ 35 min per 10M and ≈ 23 GPU-h to the cap (§4.4, ESTIMATE).
- **The watcher:** `watch_run.sh` (layer 1) starts from the script, with status in
  `~/.claude/jobs/x34/rb_toy_ent_s4002_deep.repo.watch.txt`. The Training session's 55-min quiet cron reads it.
- **The Tier-1 tick:** after each 10M checkpoint lands, on the agent cores (rule 5b, ONE heavy job):
  `nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh <GB> python -m main.plateau tick rb_toy_ent_s4002_deep
  --lineage-run models/rb_toy_ent_s4002 --delta-steps 10000000 --lag 5`, then `python -m main.plateau status …` with
  the same flags. The W-back nodes for the checks from 110M to 150M are the parent's checkpoints, hence
  `--lineage-run`.
- **What a tick costs:** a toy GAIN check took 1–7 min of wall time in T33. A check near the plateau runs up to the
  cap of 6,000 pairs.
- **STOP** at `TIER1_PLATEAU` (two consecutive FLAT checks). The plateau checkpoint is the newest node of the second
  FLAT. A stop at 500M is recorded as `CAP`, never as a plateau.
- **K9(b)** stays at the toy's default, `--behaviour-check fatal`, with the closing test's Amendment 1 / 2 handling
  (§4.3, "as for the toys").
- **Restart telemetry:** the restart interval is 24 h. Watch VmRSS growth, as X26 does (> 2 GB ⇒ a manual restart).

## 6. Findings recorded at staging

- **F1. A fork pins the PARENT's commit unless told otherwise.** Without `--pin`, `checkargs` judges the argv by the
  parent's recorded `git_hash` (`3899277a`), and a launch without `--pin-commit` would run there: the pre-T36 canary.
  `--pin-commit` is MANDATORY for X34, and the script always passes it.
- **F2. The plateau driver has no "from step" option.** `tick … --lineage-run rb_toy_ent_s4002` also lists the
  checks from 60M to 100M as due, on the parent's nodes (a dry-run on s4002 at Δ 10M / lag 5 lists 60M–100M as
  NOT_STARTED). They are played oldest first before the first post-fork check. They are real points on the same
  curve under the same rule. If the 90M and 100M checks both read FLAT, Tier 1 already reads `TIER1_PLATEAU` at
  100M. That contradicts the readiness doc's GAIN reading at the toy's own Δ 5M rule, and it would be a finding for
  the orchestrator, not a reason to stop.
- **F3. The K9(b) exposure over 400M steps.** X26 hit a rounding-class K9(b) stop at u13900 and resumed under `warn`
  with fixed ceilings (Amendment 3, which predicts ≈ 110 rounding stops by 500M under `fatal`). The toy kept `fatal`
  through 100M with no stop, but X34 is 4× longer. If one is expected, a `warn`-with-ceilings amendment for X34 must
  be registered BEFORE the first Tier-1 check (§4.3's amendment rule). This is the orchestrator's call, and the
  staging changes nothing.
- **F4. The fork keeps `--seed 4002`**, the parent's seed, as s4001b kept 4001. It is unverified whether a fork
  re-draws the parent's opening episode seeds. This is inherited practice, not checked here.
- **F5. The identity proof is the fp32 eager CPU forward.** The CUDA compiled bf16 path is judged by the startup
  compile gate and the canary on the trained weights at launch, not here.
- **F6. The toy stays loadable through U2 and U4.** The toy recorded the ADOPTED value of every remaining U2 and U4
  lever (`obs_facts v1`, `drop_progress_clock on`, `move_target_state on`, `token_encoding static`,
  `move_actor_state on`). Its two OFF values, `mon_hazard_cost off` and `move_set_closure off`, are vacuous (they
  need `damage_op` and `opp_intent`). The structure test above fails if a later unit moves the toy network anyway.

## 7. LAUNCH DECISION + K9(b) amendment (orchestrator, 2026-10-19 ≈ 02:00; registered BEFORE launch)

- **X34 is the NEXT GPU run.** It launches after X26's Amendment-5 stop at 380M (≈ 10-19 11:30–12:30). The order follows
  the owner's 10-16 plan: X26 plateau → toy plateau → breakers on both. X34's plateau checkpoint T\* is also the
  missing second X-block member of X35's frozen R (`d172c10e` F1).
- **K9(b) (resolves F3):** X34 runs `--behaviour-check warn`, with Amendment-3-style GIGO ceilings read at full precision
  at every Tier-1 tick, restart and daily read:
  - **STOP** on any update with judged max ≥ 1e-3, or p99 ≥ 1e-4, or any non-finite value.
  - Rule 8: within 1e-9 relative of a ceiling = recorded, no fire.
  - X26's Amendment 6 (STOP at 1e-2 for the τ class) does NOT apply. The toy has no belief tokens
    (`--move-belief-mode off --no-t0-species-prior --opp-belief-cls-k 0`), so the τ mechanism cannot occur. A crossing
    is a §A2.7(3)-style HOLD + diagnosis.
- **Pin:** `--pin-commit` = main at launch (≥ `d172c10e`; it contains T36 `5236eb22`, memfix `9c033489`, T54
  `cdab9cfe`).
- **Restart interval:** 24 h, with the VmRSS guard.
- **F2 handling:** if the inherited 90M / 100M nodes both read FLAT at the first tick, that is a FINDING to the
  orchestrator, not a stop.
