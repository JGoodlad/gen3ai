# Training — THE TRAILING MAGNET (`--magnet-kl-coef`, `gen3_magnet_kl_v1`)

**Owner of:** `src/agents/training/magnet.py` (the term, the magnet's state, the sidecar, the collector's side), the
two flags, and the magnet's seams in the collector, the checkpoint commit and R1. Experiment CE-3d / X35-A4, task T56.
**OFF by default; no run has trained with it.**

```
loss += β · mean over rows of KL(π_θ(·|s) ‖ π_mag(·|s))       β = --magnet-kl-coef (unset = 0 = OFF)
```

`π_mag` (the magnet) is a frozen copy of the policy's own weights. It is replaced by the current weights at the first
update boundary at or after `--magnet-refresh-steps` env steps since it was last set (unset = 2,000,000). The entropy
bonus is untouched: the term sits beside it.

## 1. Where it comes from, and what it is not

- **Magnetic Mirror Descent** (Sokota et al., ICLR 2023, arXiv:2206.05825) regularises every policy step toward a
  magnet with the REVERSE KL, KL(π ‖ ρ). A uniform magnet is exactly the entropy bonus PPO already has; a magnet that
  trails the policy moves the fixed point from a regularised (quantal-response) equilibrium toward a Nash equilibrium.
  Its deep form is PPO plus a reverse-KL loss term, which is what this is.
- **R-NaD / DeepNash** (Perolat et al., *Science* 2022, arXiv:2206.15378) is the same idea as a reward transform,
  −η · log(π / π_reg), with π_reg replaced every Δm steps.
- **Not taken from R-NaD:** the interpolation between two consecutive regularisation policies (the magnet here jumps
  at a refresh), and the opponent-side term of the reward transform (the loss regularises the learner's own
  distribution only). Neither paper's convergence result is claimed for this spelling; **UNVERIFIED:** that a
  trailing magnet damps our treadmill is CE-3d's question, not a property of the build.

## 2. The direction, and what β does to the entropy bonus

The term is KL(π ‖ π_mag), the expectation under the LEARNER's distribution (MMD's and R-NaD's direction). Over a
row's legal actions

```
KL(π ‖ π_mag) = −H(π) − Σ_a π(a) · log π_mag(a)
```

so β is two things at once: **β more weight on the entropy bonus's own gradient**, and a cross-entropy pull toward the
magnet. At the card's β = 0.05 beside `ent_coef` 0.05 the entropy gradient's weight is 0.10. An arm that reads the
pull alone needs an entropy-matched control (`--ent-coef 0.10` and no magnet), or the swap `--ent-coef 0
--magnet-kl-coef 0.05` (MMD's own form: the uniform magnet replaced by the trailing one at the same weight).
`magnet_learner_test.py` pins the identity: against a masked-uniform magnet the term's gradient is the entropy
bonus's at `ent_coef + β`.

## 3. Legal actions only

Both distributions are the masked ones over the row's own legal set, the mask the row was played under: the
learner's from R1's own forward (`masked_categorical`), the magnet's from T2 on the same observation and mask.
`reverse_kl_rows` sums legal entries only; an illegal entry contributes exactly 0 to the value and (two `where`s, so
no 0 · ∞) to the gradient, whatever the column holds there. A non-finite value on a LEGAL entry of the magnet's column
is `NonFiniteLearnerError` at the fill (FATAL_NONFINITE), never repaired.

## 4. Where the magnet's log-probabilities come from

T2, at the fill. With β > 0 `RustEnvDecl.magnet` declares ONE more T2 slot: the last declared (after eval's), in the
trainee's slot group, at the trainee's full bucket. At each fill the collector serves every row of the filled buffer
through it once (`RustCollector._fill` → `magnet.fill_column`, in flushes of at most the declared arena) and writes
`RolloutBuffer.magnet_logp`, `[n_steps, n_envs, n_actions]` float32 with 0.0 on illegal entries. The column rides the
buffer's `get` beside its rows in every device-batch mode, and R1 reads it as a tenth argument.

**R1 never runs a second forward.** The alternative, a frozen-copy forward per micro-batch inside the update, costs a
whole extra forward of every epoch (ten per rollout at production's `n_epochs`); one T2 pass per rollout is the cheap
spelling. Cost on the GPU, **UNVERIFIED (no lease at the build):** one more full-bucket slot of CUDA-graph captures
and one more weight copy on the card, and one extra inference pass over the buffer per rollout (the task card's
estimate is ≈ +3 % of the update). Read `magnet/forward_ms` on the first GPU run.

T2 serves the fp32 function; production's R1 runs a bf16 trunk. So with the magnet equal to the current weights the
term is a small positive floor, not 0, on a bf16 run (**UNVERIFIED size**). On an fp32 learner it is exactly 0.0
(`magnet_learner_test.py`).

**Compile.** β = 0 keeps R1's nine-argument call, so the compiled program, its cache key and the gate fixture are
unchanged. β > 0 appends the column, and `magnet_kl_coef` joins the compile key only when set
(`COMPILE_ARG_KEYS_WHEN_SET`). The compile gate's fixture column is the masked uniform, not a real magnet.

## 5. The cadence

An update boundary is the top of the next collection: the loop calls `RustCollector.after_update` once the update
counter has moved, at the env step the last fill ended on. There `magnet.refresh_if_due` compares the magnet's age
(`num_timesteps` minus the step it was set at) with the cadence. At or past it, the magnet becomes the weights the
learner just stepped to, and T2's magnet slot is reloaded with them (`magnet@<step>`).

- The cadence counts total env steps, and a collection's count is the decisions the core played, not
  `n_envs × n_steps`. A refresh therefore lands up to one rollout after the nominal step.
- **A restart can delay one refresh by one rollout.** The last update of a process has no boundary after it. A
  refresh due exactly there waits for the first boundary of the next process.
- A static magnet (R-NaD's anchor, or "P\* forever" on a fork) is a cadence the run never reaches.

## 6. The state, the sidecar, a resume and a fork

The magnet is ONE host (CPU) copy of the served weights plus the env step it was set at (`Magnet`, held as
`model._magnet`). It is not a module of the policy (every reader and strict load would carry a second network) and it
is not in the checkpoint zip. It is a **sidecar**, `<checkpoint stem>.magnet.pt`, written atomically by
`run_io.commit_checkpoint` right after the zip and before the caller's sidecars and `latest.txt`. So every resumable
checkpoint of a magnet run carries its magnet, at the cost of one more weight copy per checkpoint on disk.

What a `--model` launch does (`magnet.attach`, called before the Rust env's startup because the collector declares
the slot iff the learner holds a magnet):

| the launch | the magnet |
|---|---|
| fresh | the initial weights, set at step 0 |
| the checkpoint has a sidecar | the sidecar's weights and step (its age survives the restart) |
| a same-run resume, the run recorded β > 0, no sidecar | **REFUSED** (`MagnetError`, FATAL_CONFIG): the file was lost, and re-seeding would change the regulariser mid-run |
| a fork, or the lever turned on at this launch, no sidecar | the loaded weights at the loaded step, announced on the `[MAGNET]` line |

A sidecar is also refused when it was set after the checkpoint's own step, when its keys are not this policy's, or
when it cannot be read. **Snapshots and `best_model` files carry no sidecar** (they are not written by the commit
protocol), so a fork from one seeds a new magnet.

Both flags are recorded in `model_config.json` (`magnet_kl_coef`, `magnet_refresh_steps`) and inherited on a flagless
resume or fork; a typed value wins. **At this pin (CE-3d's backport of main's `251bbfb87` onto `f4b25238`) the two
fields are UN-VERSIONED:** `MODEL_CONFIG_VERSION` stays 161, a v161 config without them loads at the defaults (0.0,
2,000,000), and a config that carries them is refused by the plain `f4b25238` (unknown key) and loads on main (v170's
migration keeps a recorded value). "Config v170" in a comment or a help text of this pin names main's version.

## 7. Refusals

| combination | why | where |
|---|---|---|
| β > 0 with `--fork-fraction > 0` | the column is served for the buffer's played rows only; the fork arm appends successor rows | `magnet.validate_decl` (`RustEnvDecl`), and `combination_checks` `magnet_kl_vs_fork_arm` |
| β > 0 with `--overlap-collect` | the magnet's pass runs on the main thread at the fill, while the overlap's worker owns the T2 flush | the same two places, `magnet_kl_vs_overlap_collect` |
| β > 0 and no column in the buffer | a run that would train without its regulariser | `MagnetError` in `PPO._magnet_args`, `ValueError` in `micro_step` |
| β < 0, or a cadence ≤ 0 | not a regulariser | the parser |

## 8. Telemetry

| tag | producer | reads |
|---|---|---|
| `train/magnet_kl` | `ppo.train` | the term's mean KL (nats) over the update's micro-batches, before β |
| `magnet/age_steps` | `magnet.fill` | env steps since the magnet was set, at this fill |
| `magnet/refreshes` | `magnet.fill` | refreshes this process has made |
| `magnet/forward_ms` | `magnet.fill` | wall time of the magnet's pass over the buffer |

None is emitted with the magnet off. `train/magnet_kl` rising toward the cadence and dropping at a refresh is the
expected saw-tooth; a flat ≈ 0 between refreshes means the policy is not moving.

## 9. Tests (each fails on revert)

- `agents/training/magnet_test.py`: the term against a hand sum; illegal entries inert in value and gradient; the
  direction and the entropy split; the cadence; the sidecar's place in the commit order; the resume and fork table of
  §6; the refusals; the flags, the version fields and the migration; the compile key.
- `agents/training/magnet_learner_test.py`: β = 0 is R1's nine-argument call and reproduces the recorded fp32 learner
  golden bit for bit even with a column in the buffer; β > 0 against the learner's own log-probs is exactly 0; against
  the masked uniform it is `log n_legal − H` and the entropy bonus's gradient.
- `agents/training/rust_rollout/magnet_integration_test.py` (`slow`, real core and real T2): the slot is last and
  never the trainee's; the column is the magnet's legal-only log-probs; frozen between refreshes; refreshed at the
  boundary whose age reaches the cadence; a resume into a new learner and collector serves the same column bit for
  bit; a whole `learn()` logs the meters with the replayed ages.
