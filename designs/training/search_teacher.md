# Training — search-as-teacher

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## Search-as-teacher (`--search-teacher`, `teacher/` package)

Selective **Expert Iteration** — the offline-teacher plateau-breaker (design:
`designs/ai_v6/design_search_teacher.md`). Each cycle, **search + rollout-confirm the worst loss
craters** of recent eval traces and distil the VERIFIED-better action into the policy via an
**advantage-weighted CE aux loss (AWR)**. Off by default (`--search-teacher` absent / coef 0 ⇒
byte-identical). The "expert" is the prober's `better_line` beam + the rollout-confirm tiers
(`src/main/prober/`); this wires them into training. Package `src/agents/training/teacher/`:

- **`selection.py`** (`select_candidates`, Phase 0, model-free) — the two-stage funnel:
  `ProbeSession.scan` ranks the worst-ΔV loss craters → `falsifier.falsify_battle` gates to *reducible
  MISTAKEs* (not aleatoric LUCK — don't teach against dice) → expand to the crater **±window** (the
  cause is usually 1–2 turns BEFORE the value crater). Ranks by |δ|, caps at the budget.
- **`opponent_resolver.py`** (`resolve_opponent`) — the EXACT opponent: a `sentinel_<i>` trace → its
  `models/<run>/snapshots/snapshot_<step>.zip` (the positional index→step map is in
  `metadata.json:latest_eval.pool.sentinels[i].snapshot`, valid only for the latest cycle — which is
  what the teacher runs on); a bot → reproducible from its name; anything else → **`'unresolved'` →
  SKIPPED, never approximated** (distilling "A* beats a proxy" is a soundness failure, not a degrade).
- **`produce.py`** (`produce_correction`) — the 3-tier strictly-better gate: SEARCH (`session.better_line`
  with `interior_opponent='ckpt'`, the exact opp) → CONFIRM (rollout-to-end vs the same exact opp,
  Wilson CI) → GATE (keep only if the Wilson LOWER bound beats the played loss rate). Distils the
  **CONFIRMED** win-rate improvement (`confirmed − played`), never the critic's optimistic backed-up
  value (the Spore 95%-vs-62% lesson). Staleness re-verify: if the frozen trainee already argmaxes A*,
  skip (`already_known`).
- **`buffer.py`** (`Correction`, `CorrectionBuffer`) — a bounded recency RING of corrections, sampled
  (with its own forward) on each rollout minibatch inside `train()`. **STANDALONE, not the rollout buffer** — the searched states are
  off-policy (older eval traces), so they must never enter GAE / the clip objective. Lives on
  `model._correction_buffer`.
- **`callback.py`** (`SearchTeacherCallback`) + **`src/main/search_teacher_select_worker.py`** +
  **`src/main/search_teacher_worker.py`** — the non-blocking driver mirrors the eval cadence.
  🚨 **A CYCLE HAS TWO PHASES AND BOTH RUN IN CHILDREN** (`_pending["phase"]`): **select** spawns
  `search_teacher_select_worker`, which runs `select_for_mode` over the run's eval traces and writes
  `teacher_cycle/candidates.json`; **search** then freezes the trainee and spawns the per-cycle
  worker subprocesses (own POKE_LOOP, spare cores — the live trunk mutates, so a thread is unsafe;
  isolation is why eval uses subprocesses too), each running the search + confirm over a candidate
  slice (ONE warm `SearchSession` reused → the Node spawn is amortized) and publishing a shard (obs
  `.npz` + scalars `.json`); the parent polls and fills the buffer. Skip-while-running, a per-phase
  watchdog, crash-logged.
  - 🚨 **SELECTION USED TO RUN INLINE IN `_on_step`, and "non-blocking" was half true for a year.**
    `_launch` called `select_for_mode` on the training step, falsify-gating every loss trace of the
    newest eval cycle through the re-roll driver — measured by the composition gate on 2026-09-07 at
    **48.1 s over 9 traces and 350.2 s over 60** (2026-09-08, a loaded 16-core box, over a copied
    123-loss-trace directory; ~5.5 s per trace, so it scales with exactly the thing a long run has
    most of — the gate's own smaller run measured ~30 s / ~100 s on 9 traces of a 6k-step policy) — while the search workers it then spawned
    cost the loop nothing. Selection is **pure over files** (its `ProbeSession` loads no model; it
    reads `eval_traces/` and the `*_reconstruction.json` siblings and nothing from the live process),
    which is what makes a child the right shape rather than a thread. The training step now returns
    in **at most 3.9 ms** at a select launch, 83.5 ms at the worker spawn and 1.0 ms at the collect
    (composition gate, 2026-09-08, worst of a whole 30,000-step run: **0.26 s of training-step time
    for the entire run**, against 30-350 s PER CYCLE before). The 83.5 ms is `model.save` freezing
    the trainee, and it is the same save the cycle always paid.
    **`teacher/step_block_ms` / `teacher/step_block_ms_total` record it**, so the claim is a series
    in the run's own TensorBoard rather than a comment. The selected candidate list is byte-identical
    before and after (proved on a copied 123-loss-trace directory at `scan_limit` 9 and 60).
  - **A FAILED SELECTION IS A REPORTED CYCLE, not a silence.** It used to be a bare
    `except Exception` + an optional print, so a teacher that had selected nothing for a week looked
    exactly like a policy with no craters. A selection error (or a child that dies without writing a
    result) now emits BOTH the `[SearchTeacher] selection failed:` marker and a collect marker
    carrying `status={'error:selection': 1}`, plus `teacher/selection_failures_total` — the same two
    artifacts a worker crash uses, which is what lets the composition gate see it. An EMPTY candidate
    list is not an error and deliberately emits no `cycle @ N: M candidates` marker.
  - ⚠️ **The selection half's re-rolls run on NODE even on a `--use-bridge rust` run** —
    `select_candidates` calls `falsify_battle` with no `impl`. Left exactly as it was (changing the
    engine changes which craters survive the falsify gate, which is a different claim needing its own
    measurement) and recorded in `designs/ops/TECH_DEBT_BACKLOG.md`. It is why the select child's
    config is `select_config.json` and NOT `config_*.json`: the composition gate globs the latter and
    asserts every match records `"impl": "rust"`, which is true of the SEARCH workers only.
- **SUPPLY+POOL mode (`--teacher-persistent`)** — `teacher/generate.py` +
  `src/main/search_teacher_persistent_worker.py`. The per-cycle mode reads eval traces (a trickle every
  ~2M steps); the persistent mode is a LONG-LIVED worker pool that GENERATES its own fresh losses (the
  frozen trainee vs sampled current opponents — the recent pool snapshots + bots — recorded via the
  eval forensic path `begin_forensic_cycle` + `run_local_battles`) and searches them CONTINUOUSLY,
  dripping corrections into the buffer instead of a 2M-step burst. The parent RE-FREEZES the snapshot
  every `--teacher-refresh-steps` (default 500k, written to a polled `control.json`) so long-lived
  workers track the moving policy, and `_ingest`s correction shards incrementally each `_on_step`.
  Because the worker CHOSE the opponent, the exact-opponent is KNOWN directly (no sentinel-resolution
  fragility); `falsify_gate=False` here (supply is plentiful → the CONFIRM is the gate). Never touches
  the training hot path (a frozen-snapshot side activity, like eval). Validated end-to-end: one worker
  published 8 verified-better corrections from self-generated battles in ~150 s. Flags:
  `--teacher-persistent`, `--teacher-refresh-steps`, `--teacher-gen-battles`.
  - **Lifecycle hardening** (a long-lived, multi-process pool must self-heal — an adversarial review
    surfaced these): the parent `_reap_and_respawn`s a crashed worker on a step-backoff (so a dead
    worker can't silently drain the pool to zero — `teacher/workers_alive`/`worker_respawns_total`);
    snapshot pruning keeps the latest **three** (numeric `_version_key`, not lexical — `v10 > v9`) and
    the worker re-checks `os.path.exists` before every snapshot/opponent load + wraps both in try/except
    (a pruned/corrupt file SKIPS the iteration, never crashes); `_spawn_persistent` wipes stale shards +
    `gen_*` dirs from a prior crash/restart so a fresh pool never double-ingests; `_ingest` CONSUMES
    (deletes) a shard BEFORE buffering it (a delete failure DROPS it rather than re-globbing it into a
    duplicate); the worker's per-iteration `ProbeSession` is a context manager (drops its cached models)
    and the warm `SearchSession` recycles every `recycle_every` (Node V8-heap backstop; the launcher's
    3 h restart owns the rest). **The `_correction_buffer` is `_excluded_save_params` from the SB3 save**
    — it holds a `threading.Lock` that cloudpickle can't serialize (it would crash `model.save()` at the
    pre-train roundtrip smoke for EVERY `--search-teacher` run), and it's transient scaffolding like the
    rollout buffer (re-created empty on resume; keeps checkpoints small).

**The AWR aux loss** (`InstrumentedMaskablePPO._searchteacher_loss` + the `train()` fold): `coef ·
advantage-weighted CE(π(·|s), A*)` over a minibatch sampled from `_correction_buffer` with its OWN
policy forward (`get_distribution`); weight `w = clamp(exp(advantage/β), w_clip)`. The advantage is the
CONFIRMED win-rate improvement (NOT a critic advantage — the soundness point). The shared-trunk pull
rides `grad/searchteacher_share` / `_policy_cosine` (the live "is the teacher fighting the actor"
signal). `teacher/*` metrics: `agree_rate` (π ↔ A*, should RISE), `mean_adv`, `mean_w`, `loss`, `n`,
`buffer_size`, `corrections_per_cycle`, `yield`, `mean_confirmed_dwin`, `selection_failures_total`,
`step_block_ms` / `step_block_ms_total` (what the teacher cost the TRAINING STEP — see the two-phase
cycle above).

**On-policy self-distillation (OPD) — the KL upgrade of AWR (`--opd-coef`).** AWR distils only the
single verified-better action A*; OPD upgrades the distillation TARGET to the FULL improved distribution
**π'** via `opd_coef · KL(π' ‖ π_student)` (`InstrumentedMaskablePPO._opd_loss` + its own `train()` fold,
modelled EXACTLY on the AWR fold). π' is the softmax over LEGAL actions of the beam's per-action
**backed-up** values `(v(a) − max_legal_v) / opd_beta`, with a COMPLETED-Q floor (min legal value) for a
legal-but-unsearched slot and 0 on illegal slots — built worker-side in `produce.py` (`_build_pi_target`,
only when `build_pi_target`, so no cost off) and carried on the `Correction` as a NEW `pi_target [11]`
field (appended LAST, default None → an AWR-only run is backward-compatible). It travels the worker shard
`.npz` (like obs/mask, a NaN row = None) and `CorrectionBuffer.to_tensors` stacks it (all-present → a
tensor; **any-None → the key is None** so the KL None-guards — never a partial batch). The OPD fold
samples the **SAME** `_correction_buffer` (its own `get_distribution` forward), so a Correction carries
BOTH targets and a run can **A/B AWR vs KL** by which coef is set. `opd/*` metrics: `kl` (should FALL),
`agree_rate` (student ↔ π' mode, should RISE), `pi_target_entropy` (π' sharpness), `n`; the shared-trunk
pull rides `grad/opd_share` / `_policy_cosine`. **Training-only** (0 = byte-identical, NOT in
ModelVersion / `check_compatible` / any `check_*` → both A/B arms resume a pre-OPD checkpoint with zero
FATAL risk; coefs `_resolve`-inherited on a flagless resume). **Requires `--search-teacher`** (it fills
the buffer + its workers build π'; a `parser.error` guards `--opd-coef>0` without it).

**Why NOT value-only:** the search VALUE is the *improved-policy* value V^π*(s); regressing the PPO
critic (which must predict V^π for GAE) toward it biases advantages. So the signal is the **policy**
(AWR); the off-policy value term is wired but `--search-teacher-value-coef 0` by default (the
joint-ExIt A/B). **All training-only** (no `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` bump; coefs
`_resolve`-inherited on a flagless resume, operational knobs forwarded by the launcher). **Honesty
gate:** the search *finding* a better line ≠ it *helping* — validate `eval/td_resid_tail` /
calibration / ELO on a `coef=0` A/B. (The old "~⅔ of grind losses are matchup-lost / UNCOACHABLE"
caveat is **RETRACTED** — model-judged recoverability is circular; treat those losses as headroom.)

| Flag | default | role |
|---|---|---|
| `--search-teacher` | off | master enable (constructs the callback + buffer) |
| `--search-teacher-coef` | `0.0` | AWR policy CE weight (0 = byte-identical) |
| `--search-teacher-value-coef` | `0.0` | off-policy value term (OFF — soundness) |
| `--search-teacher-beta` | `1.0` | AWR temperature β |
| `--opd-coef` | `0.0` | OPD KL(π' ‖ π_student) weight (0 = byte-identical; requires `--search-teacher`) |
| `--opd-beta` | `1.0` | OPD softmax temperature β for π' |
| `--teacher-search-budget` | `200` | candidates searched per cycle |
| `--teacher-confirm-rollouts` | `8` | Monte-Carlo confirm games (the CI gate) |
| `--teacher-search-workers` | `3` | worker subprocesses per cycle |
| `--teacher-search-freq` | `0` | steps between cycles (0 = eval freq) |
| `--teacher-scan-limit` | `60` | loss traces the SELECTION child falsify-gates per cycle — its cost (~4.2 s/trace, in the child) **and** its supply (the crater pool). Config **v113**: recorded in `model_config.json` and `_resolve`-inherited on a flagless resume, so a launcher restart cannot silently reset a run's scan width. Ignored by `--search-teacher-mode winprob_oneply` |

**Sim engine (`impl`, no flag of its own).** Every child the teacher spawns — the generation
battles (`teacher/generate.py` → `run_local_battles`), the searches (`SearchSession`) and the
replay/re-roll driver (`ProbeSession`) — takes its engine from `SearchTeacherCallback(impl=…)`,
which `train_rl_agent` sources from the existing **`args.bridge_impl`** (so there is no new
user-facing flag; `"node"` when `--use-bridge` is off, which is the historical behavior). It rides
each worker's config JSON as an `"impl"` key. This closed a real silent gap:
`teacher/generate.py`'s `run_local_battles` call had **no** `impl=`, so on a `--use-bridge=rust`
run its battles would have been generated on node regardless.

**`--use-bridge=rust` + `--search-teacher` now RUNS** — the old hard `parser.error` is deleted
(`gen3_rust_search_driver_v1` / `gen3_rust_replay_driver_v1`: one `search_driver` binary serves both
offline verb families). Each LEG is gated on rust — `better_line` node≡rust candidate V (an
obs-level bit-identity claim), `search_clone_parity` (clone ≡ `reroll_many` at the obs), and the
counterfactual confirm leg — and since **2026-09-07 the COMPOSITION is gated too**:
`src/main/train/search_teacher_composition_test.py` (`gen3_search_teacher_composition_rust_v1`,
`sim` + `slow`, ~11 min) launches the real `train_rl_agent.py` at smoke scale under
`--use-bridge rust` and asserts on the ARTIFACTS of **>= 2 cycles** — the per-cycle markers, the
worker config's `"impl": "rust"`, the worker status histogram, and the TB scalars for both the
cross-process facts (`teacher/corrections_per_cycle`) and the in-process fold (`teacher/loss`,
`teacher/n`, `grad/searchteacher_share`). **The first multi-cycle rust run found no seam.**
🚨 **The one knob that had to be MEASURED rather than guessed is `--teacher-confirm-rollouts`**: the
Wilson strictly-better gate needs the alternative line to win at least one confirm game, and a
6k-step policy does that at p ~= 0.028, so at 2 rollouts a whole cycle yields nothing about 90% of
the time (measured: 0 corrections over 12 candidates and 3 cycles) while at 16 the yield is 4 of 11.
A cycle reporting `status={'gate_failed': N}` has still exercised every join — the yield is a
property of the POLICY, not of the composition. Fall back to `--use-bridge=node` if a cycle
misbehaves. That guard's OLD stated reason — the record's `input_log` being
replay-equivalent rather than byte-identical — was **wrong and is retracted**: no consumer reads the
committed-choice lines, so do not re-derive a plan from it. See `src/utils/bridge/README.md` →
*Offline driver transport* for the seam and the full gate table.

**Tests** (`src/agents/training/teacher/*_test.py`): `buffer_test` (ring/sample/stack), `awr_loss_test`
(AWR math, masking, grad), `opponent_resolver_test` (bot/sentinel/unresolved, tmp metadata),
`produce_test` (the 3-tier gate with a fake session), `selection_test` (the funnel with a fake
ProbeSession + monkeypatched falsify), `callback_test` (shard→buffer collect + crash-graceful); plus
`instrumented_ppo_test.py::test_search_teacher_*` (the AWR fold in a real `train()` moves the policy
toward A*; off-by-default no-op). **OPD tests:** `instrumented_ppo_test.py::test_opd_*` (the `_opd_loss`
KL — 0 at the fixed point / >0 otherwise / None-guards / illegal-action masking — plus the real-`train()`
fold moving the policy toward π', off-byte-identical even with a populated buffer, and the AWR-only
π'-less buffer being skipped), `teacher/buffer_test` (`pi_target` roundtrip: all-present → tensor,
any-None → None), `teacher/produce_test::test_pi_target_*` (π' sums to 1 over legal / 0 illegal / peaks
A* / temperature flattens / completed-Q floor). End-to-end pipeline (selection → exact-opp search →
confirm → gate → Correction) validated against a real run.

### The WIN-PROB ONE-PLY teacher (`--search-teacher-mode winprob_oneply`, ai_v12 routes 2+3)

**`--search-teacher-mode` defaults to `crater` — everything above — and that default is
byte-identical to the behaviour before the flag existed. Nothing has run `winprob_oneply`; no arm is
registered.** Design:
[`designs/ai_v12/design_winprob_behavior_coupling.md`](../../../designs/ai_v12/design_winprob_behavior_coupling.md).

A new **SUPPLY** of corrections on this exact seam, not a new pipeline. It produces the SAME
`Correction` record, so the shard format, `CorrectionBuffer`, `_searchteacher_loss` and
`--search-teacher-coef` are all untouched and cannot tell the two modes apart. Only the SELECTION and
PRODUCTION halves are swapped, and the dispatch lives in ONE place (`teacher/modes.py`) because there
are three call sites and a mode string validated in three places will eventually mean three things.

| | `crater` (default) | `winprob_oneply` |
|---|---|---|
| asks | *where did the model lose the most value, and is there a strictly better LINE?* | *at a decision the head calls CONTESTED, does a one-ply read prefer another action by a margin that survives confirmation?* |
| selection | `select_candidates` — value craters, falsify-gated to reducible mistakes, ±window | `select_winprob_candidates` — the H rule, **model-free** off the trace's recorded `win_probs` / `action_mask` |
| production | `produce_correction` — a depth-2 beam over the **critic**, Wilson-gated | `produce_winprob_correction` — one-ply **win-prob** ranking → margin floor → paired rollouts |
| battles used | LOSSES only | **every outcome** — a whiff in a won game is still a whiff, and the head's self-referential labels are exactly why it never noticed |

The pipeline, which is the design doc's "3 filters → 2 transplants" as code:

1. **CONTESTED gate** — `n_legal ≥ 2` AND `|P(win|s) − 0.5| < --winprob-teacher-band` (default
   `0.15`). **Imported from `main.search_dividend.defensive.gate`**, not re-typed: two definitions of
   "contested" that could drift apart while both looked right is a failure this tree has paid for,
   and the teacher's band IS `DefensiveConfig.wp_margin`. A decision with no recorded win-prob (NaN)
   is never contested and is **never imputed** — one we cannot judge is one we do not teach from.
2. **ONE-PLY read** — `ProbeSession.lookahead` re-rolls the turn under each legal action (opponent
   plays its RECORDED move), materializes the successor through the real encoder, reads the heads.
   We take the **win-prob** read, not V: under `--critic shaped` the critic estimates shaped return
   in PopArt units (under `winprob` the two are the same readout, so the choice is free), and
   probe G measured the win-prob head beating the played action on exactly this job. A candidate with
   no win-prob read is **dropped, never scored from the critic** — a fall-back would silently run a
   different teacher under the same flag (the confusion `defensive.check_leaf` exists to prevent).
3. **MARGIN gate** — `--winprob-teacher-margin` (default `0.02`), against the **PLAYED** action, not
   the runner-up: the target exists to move probability OFF what the policy did.
4. **CONFIRMATION** — `--teacher-confirm-rollouts` (the **existing** flag, default 8) paired
   `replay_counterfactual` rollouts to a terminal for A\* and for the played action. A rollout
   contains the opponent response the one-ply leaf structurally lacks. The test is **asymmetric on
   purpose** — A\*'s Wilson LOWER bound against the played action's POINT rate — because the failure
   it catches is a flattering estimate of the challenger.

⚠️ **STEP 4 IS A REQUIREMENT, NOT A REFINEMENT — the WINNER'S CURSE.** Defensive-search iter 2
(`designs/research_state/measurements/defensive_search_iter2_2026-08-29.md`) un-throttled its
allocator, produced **13× more evidence-certified overrules (1.8% → 5.82%)** and landed the win rate
on **0.5003 [0.4803, 0.5203] — the point estimate IS the null**. CRN pairing removes dice noise *and*
the shared offset, so what a separation procedure certifies is the leaf's residual **differential**
bias (RMS 0.122, larger than most true gaps) as much as signal. **Statistical separation of a biased
reader is not correctness**, and unlike route 1's PBRS a distillation target has **no invariance
shield** — a wrong target simply trains the policy to be wrong. `--teacher-confirm-rollouts 0` exists
only because the design doc's **E2** needs an undisciplined control arm to demonstrate this.

The counter-evidence that keeps the mode alive: **probe K** re-judged iter 2's 3,531 overrules under
opponent-MARGINALIZED ground truth and found **+0.0474 [+0.0216, +0.0730] per decision — REAL**. The
overrules were right; the per-decision → per-episode TRANSFER failed (+4.7pp × ~2.2 overrules/game
bought +0.0003). A **training** target changes the policy everywhere the network generalizes, not
only at the 2.2 decisions per game where a searcher intervened — which is why the response to probe K
is route 2 rather than a fourth iteration of route 3 as an inference lever.

**Why `--winprob-teacher-margin` defaults to 0.02 and not 0.122.** 0.122 is the *measured* leaf-bias
RMS, and running there collapses target volume by roughly an order of magnitude before any arm has
asked whether it should. E4 is the arm that measures the volume/quality trade; E2 runs at the working
default. ⚠️ If the head's differential bias is ever fixed at source (the empowerment program's
contrastive marginalized labels), **this default and E4's whole premise need re-measuring** — they
are keyed to a bias that would no longer exist.

**What was reused from `search_dividend/` and what was not.** `defensive.gate` + `DefensiveConfig`:
imported. `defensive.verdict` / `resolve_action`: NOT — they answer "which action do I PLAY", and the
teacher answers "is this a target". `racing.Racer` and the budget/deadline machinery: NOT — they are
the *allocator*, racing arms against a wall clock inside a battle in flight, and the teacher works
offline from a recorded reconstruction with no clock to race. `playoff.PlayoffRunner`: NOT — it needs
a live `SearchEngine` and a shared `Deadline`; the confirmation goes through
`ProbeSession.replay_counterfactual`, the same offline primitive `produce_correction` already uses.
The residual duplication is the paired-margin arithmetic, a handful of lines, and it is deliberate.

**Flags** (all OPERATIONAL — re-pass on resume, like `--search-teacher` itself; not `_resolve`d, not
on `ModelVersion`, recorded in `metadata.json`'s `cli_args` like the rest of this family):
`--search-teacher-mode {crater,winprob_oneply}` (default `crater`), `--winprob-teacher-band` (0.15),
`--winprob-teacher-margin` (0.02). The confirm count is the **existing** `--teacher-confirm-rollouts`
— adding a second spelling for one number is how a flag surface rots.

**Config gates** (the only gates there are): `winprob_oneply` without `--search-teacher` is refused
(no teacher would run at all); without `--win-prob-mode read_only|shaping` it is refused (the ranking
IS the head, and falling back to the critic would run a different teacher under the same flag); the
band must be in `(0, 0.5]` and the margin in `[0, 1)`. An unknown mode string **raises** at callback
construction rather than falling back to `crater` — and a worker config with no `mode` key defaults
to `crater`, so an older parent's config still runs exactly as it did.

**Tests.** `teacher/winprob_oneply_test.py` (40): every gate as a pure function (contested / ranking /
margin / Wilson / paired confirmation, including the synthetic winner's-curse rejection and the
asymmetry of the test); the mode seam (default, unknown-mode raise, both dispatch pairs, the two
margins staying separate parameters, both workers' `crater` fall-back, callback-time validation); the
consumer contract (a winprob `Correction` runs through the real `_searchteacher_loss`); crater-path
argument identity; and all five config gates.

