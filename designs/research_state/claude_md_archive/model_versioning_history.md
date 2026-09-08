# Model versioning / provenance — the incidents behind the guards

Lifted verbatim from the root `CLAUDE.md` § *Model Versioning* on **2026-09-07**, when that section
was cut from 104 lines to ~50. The live rules stayed in `CLAUDE.md`; what follows is the evidence:
the sidecar git-hash mis-stamping, the `best_model`-first resolution defect that silently loaded
~0.93M-step teachers, and the TensorBoard-inheritance measurements and backfill census.

---

## Model Versioning

Every model save writes two **run-level** files at the run root:

- `model_config.json` — all weight-shape-relevant architecture params (embedding dims, layer sizes, obs dim, etc.)
- `metadata.json` — git hash, timestamp, SB3/Python versions (+ `snapshot_history`, `latest_eval`,
  and run provenance: `cli_args` [the latest process's full argparse namespace], `launcher_command`,
  and `original_command` — the **immutable** original invocation that created the model, written once
  at creation and preserved verbatim across every restart/checkpoint, unlike `cli_args` which is
  overwritten by the resuming process. Provenance lives ONLY here, never in `model_config.json`, which
  is the weight-shape/arch record used for `check_compatible`. It also carries a top-level
  **`num_timesteps`** — how far the run has trained — written by the same save path into the
  run-level file and into every per-checkpoint sidecar, "latest" rather than immutable, and ABSENT
  (= unknown, never 0) on a run that predates the key)

**LINEAGE — who forked whom** (`gen3_run_lineage_v1`). `metadata.json` carries a top-level
`lineage` block with the SAME immutability contract as `original_command` — written ONCE at fork
creation, preserved verbatim across every restart and every checkpoint, because
`save_model_snapshot` reads the existing value first and the existing value always wins. It records
the `fork_parent` (path as given, resolved realpath, run dir + run name, git hash, arch signature,
model-config version, the parent checkpoint's `num_timesteps` and its sha256), the `fork_step` this
run started at, the `role` (`fresh` / `fork` / `fold` / `exploiter`), every `--distill-teacher`, the
`--exploiter` target, and the `ancestry` chain walked through each parent's own block. A FRESH run
writes the explicit null form (`fork_parent: null, role: "fresh"`) — "no block" and "no parent" are
different facts. Fork-vs-restart is decided by `main.train.fork_lr.is_same_run_checkpoint`,
IMPORTED, never re-derived: a `--model` outside the run dir is a fork, a `<run>/checkpoints/*.zip`
is a restart, so the launcher swapping `--model` to the fork's own drifted checkpoint can never
re-point the recorded parent. **Read it through the ONE accessor**,
`agents.training.lineage.fork_parent(run_dir)` / `.ancestry(run_dir)`, which falls back to DERIVING
the parent from `original_command` for every pre-`lineage` run on disk and prints
`[lineage] WARNING: derived from original_command (legacy run, pre-lineage)` when it does — so the
argv regex lives in exactly one place and is marked as the legacy path.
Every model reference in the block — the parent, each teacher, the exploiter target — additionally
records **WHICH FILE it resolved to and HOW** (`gen3_last_snapshot_resolution_v1`): `resolved_file`,
`resolved_num_timesteps`, `resolution_rung` (`explicit_step` / `explicit_zip` / `latest_txt` /
`highest_checkpoint` / `final_model` / `best_model_fallback`) and `resolution_rule` (that rung's
coarse class). A reference is usually a run DIRECTORY, and a directory is not a file — before this
it silently meant the BOT-WIN-RATE `best_model` export, which for 2 of 8 R5F teachers was a
~0.93M-step checkpoint rather than the ~2.93M final with nothing on disk saying so. **A run recorded
before the change has none of these keys, and `main.lineage` prints `resolved file not recorded (pre
gen3_last_snapshot_resolution_v1)` rather than re-resolving it under today's rule** — every teacher
loaded before 2026-09-06 went through the OLD (best_model-first) order, and a current answer
presented as history would be worse than no answer.

**`python -m main.lineage <run>…`** prints the ancestry tree (`--json` for scripts), the resolved
file behind every reference, and flags a broken link: a missing parent dir, a sha256 that no longer
matches the file on disk, an `arch_signature` that changed across a link. `--backfill` writes a
derived block into a legacy run's metadata (marked `"derived": true`), dry-run unless `--apply`.

🚨 **The `git_hash` is the HEAD of the checkout the code was IMPORTED from — never the process
cwd — and a disagreement with `$LAUNCHER_GIT_HASH` RAISES at the write** (`gen3_sidecar_git_hash_v1`,
2026-09-05). `utils.git.get_git_hash()` asks git about `utils.paths.repo_root()`, so a launcher
child — which imports the pinned worktree via `PYTHONPATH` but is spawned with no `cwd=` and
therefore *stands in* the un-pinned main checkout — records the pin rather than main's ambient HEAD;
one resolver (`snapshot.resolve_git_hash`) serves the run-level metadata and every checkpoint
sidecar, and it throws `GitHashMismatchError` when the launcher's pin and the imported tree name
different commits. Before the fix a run pinned to `eb5261ff` stamped `fff95a16` into every sidecar,
and its resume pinned the wrong commit.

**`pin_history` — WHICH COMMIT RAN WHICH STEPS.** The scalar `git_hash` is rewritten on every save,
so on a run that restarts every 3 h it names only the LAST code to touch the run. `metadata.json`
therefore also carries an **append-only** `pin_history` beside `lineage` and under the same
immutability contract: one `{git_hash, pin_source, first_step, last_step}` entry per contiguous
commit span, written by the same save path, an existing entry never rewritten except to advance its
`last_step`, and a legacy run with no history seeded with a single `derived: true` span from its
scalar hash (so *absent* never reads as *one commit*). Every checkpoint sidecar stamps the history
as of its write. Read it with **`python -m main.sidecar_audit <models_dir_or_run>…`**, which flags a
run with >1 span as PIN-SPLIT and separates an explained sidecar hash from a misattributed one.

**A FORK INHERITS ITS PARENT'S TENSORBOARD CURVES** (`gen3_tb_inherit_v1`, 2026-09-06). A fork's
global step CONTINUES the parent's (`reset_num_timesteps=False`), so its `tb/` used to open mid-air
at `fork_step` — measured: `ai_v8_03_zarch_control_0718`'s first logged step is **148,401,356**,
exactly the `fork_step` its lineage block records. At fork creation, right where the `lineage` block
is written, `<fork>/tb/` now also gets a TRUNCATED copy of the parent's **scalar** events at steps
**≤ `fork_step`** (the parent usually trained past the fork, and that tail would draw parent-only
progress inside the fork's own step range). TensorBoard merges every event file in a run dir into one
series by step, so the fork's charts read as one continuous curve from step 0. A fork-of-a-fork
composes for free — the parent's `tb/` already carries its own prefix, so re-truncating it yields the
grandparent's span plus the parent's. Cost is a few hundred KB against a 262 MB archive.
`<fork>/tb/INHERITED_FROM.json` records the parent, the truncation step, the tags and the sha256 —
a curve a run did not train must say so where a reader will find it — and its EXISTENCE is the
idempotency key, so a launcher restart that still names the parent as `--model` cannot append a
second copy. `--no-tb-inherit` opts out; worth it for a large sibling FLEET under an UNCURATED
logdir, where 8 exploiters off one target draw 8 identical prefixes (under `main.tb_curate` it is
exactly what you want). Existing forks are NOT backfilled: **137 would gain a prefix** —
`python -m main.tb_inherit --list`, then `--backfill --all` (dry-run unless `--apply`). ⚠️ 105 of
those name a **derived** parent (regexed out of `original_command`, not recorded at fork time) and
the derivation can be wrong — `ai_v8_01_zarch_film_0717` claims `role=fresh, fork_step=0` while its
own tb opens at step 148,401,356 — so the backfill prints that flag per row and is dry-run by
default. Engine: `agents/training/tb_inherit.py`.

These are run-level (one per run), NOT per-checkpoint: a periodic checkpoint `.zip` lives one
level down in `checkpoints/` beside its own per-checkpoint `.json` sidecar, so
`load_model_snapshot()` searches the zip's dir **and its parent** (the run root) for
`model_config.json`. `load_model_snapshot()` in `src/agents/model/snapshot.py` checks it before
calling `MaskablePPO.load()`. A mismatch causes a hard `[ModelVersion] FATAL` error at startup,
not a silent wrong-output bug later. `model_config.json` additionally records `vf_coef` (`--vf-coef`,
the PPO value-loss coefficient): it is **fixed for a run's lifetime**, so resuming with a
different value is a FATAL error — enforced resume-only (frozen eval/pool/distill opponents are
exempt, since vf_coef doesn't affect a forward pass). See `src/agents/model/CLAUDE.md` →
resume-immutable training hparams. `_run_roundtrip_test()` in `train_rl_agent.py` runs automatically
before every `model.learn()` (save → reload → zero forward pass), so serialization breakage
crashes in seconds rather than hours.
