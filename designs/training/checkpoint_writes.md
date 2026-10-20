# Checkpoint and run-state writes: atomic, and ordered

Always-current (the training leaf `src/agents/training/CLAUDE.md` points here). The rule, the one helper, the commit
order, and the audit of every write a resume, a restart or a reader may load. Pinned by `main/train/checkpoint_commit_test.py`,
`agents/training/atomic_writers_test.py` and `utils/atomic_io_test.py`, each test failing on a revert.

## The rule

**A name a reader loads holds the previous complete file or the new complete one, never a partial one, and the pointer
(`latest.txt`) moves only after the zip AND its sidecars are in place.** Origin: the T46 adversarial review of the GPU
window (finding 8, `dccb920e`). The interrupted save wrote `model.save("final_model_interrupted")` straight to its
fixed name; a save that failed or was killed mid-write left a TORN zip, `latest.txt` still named it (it had named that
file after any earlier interrupt), and its fresh mtime passed the window's freshness check
(`gpu_window/controller.py`, `CHECKPOINT_SLACK_S`), so a resume loaded it. The same shape sat under the periodic
checkpoint at a re-reached step, the final model and every `json.dump` into `metadata.json` (a truncated one crashes
the NEXT save, which read-modify-writes it).

## The helper: `src/utils/atomic_io.py`

`atomic_write(path, write_fn)` hands `write_fn` a temp path in the SAME directory, then fsyncs it, `os.replace`s it
onto `path` and fsyncs the directory; a writer that raises (or produces no file) removes its temp and re-raises with the
previous file untouched. On top: `atomic_write_text`, `atomic_write_json` (the bytes `open` + `json.dump` always
wrote), `atomic_write_bytes`, and `atomic_save_model(model, path)`, which gives `model.save` the temp name (it carries a
`.tmp` suffix, so SB3 writes exactly there) and returns the `.zip` path SB3 would have written. File names and
formats are the callers' own, so no reader changed and old runs load as before.

- **The temp name** is `.<final name>.<pid>.<8 hex>.tmp`: hidden and ending `.tmp`, so no `*.zip` / `snapshot_*.zip`
  glob, `endswith(".zip")` test or step regex (`find_latest_checkpoint`, the pool's `_scan`) can take it for a file.
- **A stray temp** (a SIGKILL mid-save) is removed by the next atomic write into that directory when its owner pid is
  dead (`sweep_stale_temps`; a live owner's temp, in this or another process, is never touched; a doubt keeps the file).
  Until then it is inert.
- **An atomic replace gives the name a new inode**, so a save over a name that is hard-linked to a checkpoint (a
  best-model / pool / eval-snapshot name, `weights_link.link_or_copy`) never writes through the link, and a failed save
  leaves every name's bytes whole. This is why `weights_link.unshare` is gone.

## The commit order: `run_io.commit_checkpoint`

Every path that saves a resumable zip runs the same three steps; an exception anywhere stops before the pointer moves:

1. the zip, `atomic_save_model`;
2. its sidecars (`record_checkpoint`: the `<ckpt>.json` and the `snapshot_history` entry; `save_model_snapshot`:
   `model_config.json` and `metadata.json`), each atomic;
3. `latest.txt`, run-relative, LAST.

A failure after step 1 leaves a complete but unreferenced zip and the pointer on the previous target. For the
interrupted save that makes the GPU window HOLD on "no checkpoint saved by the pause" (it reads the pointer's target and
its mtime), which is the loud outcome; before, a torn file passed.

## Audit: every write, before and after (2026-10-20, `dccb920e`)

| Write | Final name | Before | After |
|---|---|---|---|
| periodic checkpoint | `checkpoints/checkpoint_<N>_steps.zip` | `unshare` + in-place `model.save`; pointer BEFORE the sidecars | `commit_checkpoint` |
| forced checkpoint (SIGUSR1) | `checkpoints/checkpoint_forced_*.zip` | in place; pointer before the sidecar | `commit_checkpoint` |
| interrupted save | `final_model_interrupted.zip` (fixed name) | **in place: the defect**; pointer before the sidecars | `commit_checkpoint` |
| final model | `final_model.zip` (fresh and resume paths) | in place; pointer before the sidecars | `commit_checkpoint` |
| crash save | `final_model_exception.zip` | in place | `commit_checkpoint` (no sidecars) |
| the pointer | `latest.txt` | temp `latest.txt.tmp` + `os.replace`, no fsync, moved too early | helper; moved last |
| per-checkpoint sidecar | `<ckpt>.json` | plain `open("w")` | helper |
| run config and metadata | `model_config.json`, `metadata.json` (4 writers in `snapshot.py`) | plain `open("w")` | helper |
| pool snapshot | `snapshots/snapshot_<step>.zip` | `unshare` + in-place | `atomic_save_model` |
| pool records | `snapshots/model_config.json`, `summary.json`, `win_rate_vs_bots.txt` | plain `write_text` | helper |
| fork pool seeding | the copied `snapshot_*.zip`, metadata, `pool_seed.json` | plain `shutil.copy2` / `open` | helper |
| growth of a seeded pool | the grown `snapshot_*.zip`, `model_config.json` | temp + replace, no fsync; plain write | helper |
| eval scratch snapshot | `eval_traces/step_<N>/snapshot.zip` (both eval callbacks) | in place | `atomic_save_model` |
| best-model / pool / kept-snapshot weights | `link_or_copy` targets | temp + replace, no fsync | helper |
| eval records | `eval_manifest.json` (+ its two rewrites in `eval_collect`), `plan.json`, shard results, `ladder.json` | plain `open`, or temp + replace with no fsync | helper |

**Left as they are, with the reason.** `eval_results.jsonl` and the TensorBoard event files are append-only (one
`write` per row; a torn last line is the reader's to skip). The eval ledger, `tb_inherit`, the compile cache and the
GPU-window protocol files are already temp + fsync + replace (or fsync'd appends). `command.txt`,
`cuda_ledger.json` and `update_fit.json` are written once, for a human, and no resume reads them. The forensic
`torch.save` dumps (`compile_canary`, `rust_rollout/consistency`) are never loaded by a run. Offline tools that write a
checkpoint into a fresh directory (`main/fresh_checkpoint.py`, `rust_eval/offline.py`, `winprob_finetune.py`) and the
separate `main/train_team_completion.py` trainer are outside the PPO training process and were not changed.

## Residual hazards (read before trusting a fixed-name pair)

- **The zip and its sidecar are not ONE atomic unit.** `final_model_interrupted.zip` and `final_model_interrupted.json`
  are two renames; a kill between them leaves the new zip beside the previous interrupt's sidecar (a stale LR / epochs /
  handoff LR on resume). The pointer cannot make this worse (it has not moved), but when it already names the fixed name
  it names a mixed pair until the next save. The window is the time of one small JSON write.
- **The temp file lives in the same directory as the target**, so a run directory on a filesystem that cannot hold a
  second checkpoint fails the save (ENOSPC) instead of overwriting in place; the disk guard (`utils/disk_guard.py`) already
  budgets one extra checkpoint.
- **A pinned older run keeps its pin's writers** (the in-place save). The fix reaches a run at its next pin.
