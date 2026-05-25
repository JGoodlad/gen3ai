---
description: Create a resume branch pinned to the exact git hash a model checkpoint was saved on, then print the training command to continue from it. Takes the model directory path as an argument.
---

# /gen3ai-resume-run

Given a model directory, create a branch at the exact git hash it was saved on and print the command to resume training.

## Steps

### 1. Read the metadata

```bash
cat <model_dir>/metadata.json
```

Extract:
- `git_hash` — the commit to pin the branch to
- `git_branch` — the branch it was originally trained on (if present)
- `training_args` — the hyperparameters used (if present)

If `metadata.json` is missing, stop and tell the user.

### 2. Determine the branch name

Use `run/resume-<model_dir_name>` where `model_dir_name` is the last path component of the model directory (e.g. `models/gen3ou_ppo_new_20260517_172316` → `run/resume-gen3ou_ppo_new_20260517_172316`).

If that branch already exists locally or on remote, append `-2`, `-3`, etc.

### 3. Create and push the branch

```bash
git -C /home/goodlad/dev/gen3ai branch <branch_name> <git_hash>
git -C /home/goodlad/dev/gen3ai push origin <branch_name>
```

### 4. Find the checkpoint file

Look for the best available checkpoint in the model directory, in this priority order:
1. `final_model_interrupted.zip`
2. `final_model.zip`
3. `best_model/best_model.zip`
4. Most recent `checkpoint_*.zip` by filename

Report which one will be used.

### 5. Print the resume command

Print the exact command the user should run. Use `training_args` from metadata if present; otherwise use the defaults from CLAUDE.md. Always ask the user how many `--steps` they want for this continuation run — do not guess.

```bash
cd /home/goodlad/dev/gen3ai
git checkout <branch_name>

export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py \
  --model <checkpoint_path> \
  --steps <ASK_USER> \
  --n-envs <from metadata or default 96> \
  --batch-size <from metadata or default 16384> \
  --n-epochs <from metadata or default 10> \
  --ent-coef <from metadata or default 0.02> \
  --n-steps <from metadata or default 2048> \
  --lr <from metadata or default 0.00015> \
  --device cuda \
  --log-level periodic
```

### 6. Confirm

One line summary, e.g.:
> Branch `run/resume-gen3ou_ppo_new_20260517_172316` pinned to `673827b` — use `final_model_interrupted.zip`
