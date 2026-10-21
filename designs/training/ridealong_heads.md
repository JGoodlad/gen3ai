# Training — the detached ride-along heads, and STRICT checkpoint loading

> Lifted out of `src/agents/training/CLAUDE.md` on 2026-10-10. Always-current, like the leaf: update it in the same
> pass as the code. Lifecycle detail: [`learner_lifecycle.md`](learner_lifecycle.md).

## The DETACHED RIDE-ALONG heads (`--ridealong-ensemble` · `--ridealong-rnd` · `--ridealong-adv` · `--ridealong-opp` · `--ridealong-rnd-variants`)

`gen3_ridealong_heads_v1` (v126; `instrumented_ppo/ridealong_terms.py`, heads in
`agents/model/ridealong_heads.py`). Baselines that OBSERVE: a V ensemble (epistemic uncertainty),
RND novelty on the raw observation, and the Q = V + A + B main effects on PPO's own labels.
`_ridealong_update` runs right after each minibatch's `evaluate_actions` and BEFORE PPO's loss is
built: detached stashes → the heads' losses → their own backward, clip and Adam step → their grads
back to None. So it sits OUTSIDE the fold order above, and nothing in the fold can see it.
`ridealong_update_test` pins one real update ON vs OFF as bit-identical (params, PPO optimizer
state, PPO scalars, RNG) — at fp32 AND with the learner's trunk at bf16, production's recipe
(`test_BF16_one_update_is_BIT_IDENTICAL_with_the_heads_on_and_off`, T32: rollout fp32, learner bf16; it also
fails if the heads' path leaves the trunk at another precision, which the fp32 test cannot see). Two things to know when reading `ridealong/*`. **The heads TRAIN and READ
on epoch 0 only** (`RIDEALONG_EPOCHS` = 1: each rollout row seen once, scored before the heads train
on it). All 10 epochs cost +13 % of a GPU update on the learner benchmark; one pass MEASURED +0.59 s = 0.88 % of a 67 s update (`ridealong_step_benchmark.py`; X26's
`PREREGISTRATION.md` "Overhead"; on the 2026-09-30 fp32 legacy network). On the end-state bf16 update (`train_ms` 42.1 s)
the cost is **UNVERIFIED** — ≈ 2.3 % if the 0.974 s of heads + variants carried over; it is inside every measured
end-state throughput (Amendment 2 clause 6).
And **the heads' Adam state is not checkpointed** (a restart resumes their weights with a fresh Adam: `_setup_model` re-acquires it). 🚨 **Nor is it SAVED or UNPICKLED any more** (`gen3_cpu_load_no_cuda_v1`): older checkpoints pickled the optimizers (CUDA tensors) into sb3's `data`, and a CPU READ of one created a CUDA context. `strict_load.NEVER_UNPICKLED` is excluded from every save and substituted on every load (`StrictCheckpointLoad.load`), and a CPU load that initialises CUDA anyway raises `CudaContextOnCpuLoad` (`designs/training/learner_lifecycle.md`).
**Every ride-along optimizer is ACQUIRED AT STARTUP** (`RideAlongTerms._setup_model` →
`_ridealong_acquire`, Adam state pre-allocated, bit-identical to lazy init). There is NO lazy build:
a step that finds an optimizer missing, or bound to other heads, raises
`RideAlongLifecycleViolation`, as K6.1's freeze guard would. Tooling that swaps `policy.ridealong`
must call `_ridealong_acquire()` again (both benchmarks do). 🚨 **Only the TRAINEE acquires: an
OPPONENT load acquires nothing** (`gen3_opponent_inference_load_v1`). The self-play pool and the eval
sentinels load through `snapshot.load_opponent_snapshot`, and every `load_foreign_opponent` (stable
opponents, exploiter targets, `main.anchors`, the offline readers) is an
`InferenceMaskablePPO` by default: policy weights only, with no optimizer, no ride-along optimizer and
no rollout buffer. It refuses `learn` / `train` / `save`. Before this fix a pool load after the freeze
pre-stepped a ride-along Adam, and K6 FATALed the X26 launch at its first pool seeding. An opponent may
differ from the trainee ONLY in the declared ride-along keys (`RIDEALONG_FLAGS`), in either direction
(F-MEM); the trainee's own resume stays strict. 🚨 **Every checkpoint load is STRICT on state-dict
keys** (`gen3_strict_checkpoint_load_v1`): `StrictCheckpointLoad.set_parameters`
(`instrumented_ppo/strict_load.py`; `OwnedLoop` inherits it, so the learner and the opponent classes
carry it) refuses sb3's non-strict "SB3 < 1.7.0" retry (`StrictLoadError`), which used to load a
checkpoint missing an extractor submodule with that submodule at fresh init. The READERS (the ladder
session `play.py`, the prober, the offline meters, `winprob_finetune`) load through
`agents.model.snapshot.load_checkpoint_strict` — a `StrictMaskablePPO`, a plain `MaskablePPO` with only
that `set_parameters` — and `src/strict_checkpoint_load_gate_test.py` fails a bare `MaskablePPO.load` /
`PPO.load`, an `exact_match=False`, or an sb3-algorithm subclass without the mixin (EMPTY allowlist).
Detail: `designs/training/learner_lifecycle.md`. The step is
K8's candidate compile region R-ride; it stays eager. **The RND variants**
(`--ridealong-rnd-variants all`, v127: `fast` / `decay` / `small` / `feat`, beside the unchanged base
RND) each step on their own Adam after the four heads. A non-finite variant disables ITSELF
(`ridealong/rndv_<name>_disabled`). `decay`'s pull toward init runs once per update, at the first step
(the accumulator marks the boundary). Their series are `ridealong/rndv_<name>_*`: loss, error mean /
median / IQR / `rel_spread` (saturation), z by class, the V-error meters, and `*_ident_ratio` (block
chimera ÷ real-row error, the identification monitor). Base logs the same `rnd_err_median` /
`rnd_err_rel_spread` / `rnd_ident_ratio`. With every variant on, the heads add 0.97 s
(1.45 %) to a 67 s update (`designs/research_state/measurements/ridealong_baseline/overhead_variants_2026-09-30.json`). B trains
only where `--opp-intent-coef > 0`, because that is what aligns the one-ahead opponent labels. The
meters are disagreement / novelty vs |V − z| (`*_auroc_err`, `*_spearman_err`, top vs bottom decile),
`rnd_z_<class>`, `adv_corr_logit`, `adv_std_starved` vs `adv_std_fed`, `q_out_of_range`. These are
MONITORING (rows the heads just trained on). No offline reader of a checkpoint's heads exists in the tree: a
verdict read (the heads on the Lane S bank and the X4 pre-read truth turns, the untrained-head floor, the RND
variants' coverage) is built on demand from this module's heads and `main.policy_spectrum`'s reader. The pre-registered run is
EXPERIMENT_BACKLOG X26.

### The micro-batch path makes ONE host read (T25 next lever 1, 2026-10-10)

`_ridealong_update` used to make ~120 device reads per epoch-0 micro-batch (CPU trace on the golden learner with
every head and variant on: 490 over 4 micro-batches, Adam's host-side `step` reads excluded; ~5,800 per production
update at 48 micro-batches). On CUDA each is a host sync: the first drains the deep queue (window C found
`RndNovelty.update_obs_stats`'s `obs_count.item()` blocking 1.75 s of the update's 2.2 s of host-blocked time) and
the rest serialise the eager tail behind the GPU, so epoch 0 ran 85.6 % GPU-busy against ~95 % later. Now:

| was | is |
|---|---|
| `obs_count.item()` (`update_obs_stats`, `normalise`) and `err_seen.item()` (`update_err_stats`), per RND head per micro-batch | `ridealong_heads._HostMirror`: the host holds the value (both buffers are written only by their own update, with host-known numbers) and reads the device ONCE per process. It is valid only while the buffer is the same tensor object, at the same storage address, at the same `Tensor._version`, so an outside `zero_()` / `fill_()`, a `load_state_dict`, a `.to(...)` or an `inference_mode` tensor falls back to the one read; a `deepcopy` / pickle starts with an empty mirror |
| `bool((w.sum(0) > 0).any())` and `bool(ok.any())`, the two label guards in `losses` | `RideAlongHeads.losses_gated`: every loss plus a 0-d device GATE for `ens` and `opp`; `losses` is its host-filtered form (the old API, for the tests and tools) |
| `RideAlongAccumulator.observe` / `observe_identification`: ~100 `float(t)`, `bool(t.any())`, boolean-mask indexes and `.cpu()` per micro-batch | DEFERRED and READ ONCE: they snapshot (device copies of the batch fields and `opp_class`, detached `out` and losses), and `flush()` folds the snapshots in call order — at the first micro-batch past the ride-along epoch (`_ridealong_update` with `epoch >= RIDEALONG_EPOCHS`), else in `metrics()`. The fold computes every statistic on the device, QUEUES each scalar on a `HostReadQueue` with the host arithmetic that consumes it, and reads the whole fold back in ONE transfer; a masked set is compacted by one shared `nonzero` + `index_select` (the elements `t[mask]` selects, in its order), and an empty selection's mean is NaN, which `add` omits — what HEAD's `.any()` guards did. Every logged value is the value it was, bit for bit |
| a `bool(isfinite(gn))` per head set and per variant, and a `float(gn)` per logged norm | `_ridealong_step`: variants' backward and clips, the heads' backward and clip, then ONE transfer of every norm and every gate, then the verdicts and the Adam steps. Nothing irreversible happens before the read |

**The rare path.** A head with no usable label in a WHOLE micro-batch is not a loss there (HEAD omitted its key, so
its parameters saw a `None` gradient and Adam skipped them; a zero gradient would still move them). The step keeps the
graph of the heads' backward (`retain_graph=True`, only when a gate exists), and when the one read shows a False gate
it zeroes the grads, redoes the backward over the valid keys and takes the old host verdict (one extra read), and
`acc.discard_losses` takes the key out of the micro-batch's snapshot so no `<key>_loss` is logged for it.

**Proof.** `ridealong_host_sync_test.py` keeps HEAD's `_ridealong_update`, its variant step, its three RND methods and
the accumulator's two bodies as the ORACLE and compares bit for bit (heads, buffers, both Adam states, every `ridealong/*` scalar, the disabled
flags, the RNG) on a clean run, a micro-batch with no win labels (the rare path), a NaN observation (fail-closed), and
a full `InstrumentedMaskablePPO.train()` (policy, trunk, V, PPO's optimizer, all 278 logged scalars). The same
comparison against a copy of the HEAD tree held on `ridealong_update_test._build` for the all-heads and four-heads
surfaces. The fold's rewrite is also compared with HEAD's body on seven hand-built edges (an empty class, no
win-labelled row, a zero advantage variance, no starved action, a zero correlation denominator), scalars, columns
(byte for byte) and identification sums. `HostSyncTrace` counts one host read per ride-along micro-batch (HEAD's path:
over a hundred), and its `train()`-level test puts every meter read after the last micro-batch read and bounds the
fold at ten reads per micro-batch plus one transfer. The golden plain update with every head on: 1,082 syncs -> 50.

**What remains.** (1) One transfer per micro-batch, the verdicts: a non-finite norm must not be stepped, which is a
host decision (a device-conditional Adam step would remove it; fused Adam's `found_inf` is CUDA-only, so the CPU tests
could not hold it). (2) The fold's compaction: ~8 `nonzero` / boolean-index reads per micro-batch (the win-labelled
rows, the four opponent classes, the starved / fed / legal sets) and its one transfer, once, past epoch 0 — ~440 per
production update where the old path made ~5,800 mid-epoch; one sort could cut the four class reads to one. GPU wall,
GPU-busy % and the host-blocked time are NOT measured here (no lease): the bench command is in the 2026-10-10
CHANGELOG entry.
