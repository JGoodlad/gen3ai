---
description: Syncs a designs/ai_vN/impl_step*.md document with recent code changes. Use when the user wants to update or create a design doc after implementing something, sync docs with git history, or capture what was actually built. Triggers on "update design docs", "sync the doc", "update the step doc", "write up what we built", or any request to keep the designs/ folder current with recent commits.
---

# /gen3ai-update-design-docs

Keep a `designs/ai_vN/impl_step*.md` document in sync with what was actually built. This is a post-implementation record — it captures decisions made, constants set, and files changed. It does not speculate about future plans.

## Step 1 — Identify the active version and target doc

You may receive a path argument (e.g. `designs/ai_v3/impl_step4_reward_shaping.md`). If so, use it and skip the rest of this step.

Otherwise, figure it out from git:

```bash
git log --oneline -20 -- designs/ src/
```

Look at the commit messages. The most recently touched `ai_vN` folder is the active version. Then:

```bash
ls designs/<active_version>/
```

Read the folder listing. Look for:
- The highest-numbered `impl_step*.md` that already exists — that's likely the current doc to update
- A `todo.md` — read it to see which step is marked ✓ DONE most recently and what's in progress

If you're updating an existing doc, read it in full so you know what's already covered and don't duplicate. If you're creating a new one, read 2–3 existing `impl_step*.md` files in that folder to understand the conventions — heading structure, table format, section order, and level of detail vary by version.

## Step 2 — Find what changed since the last doc update

Get the commit hash when the target doc was last touched:

```bash
git log --oneline -1 -- <doc_path>
```

Then get all commits since then that touched relevant src/ paths:

```bash
git log --oneline <last_hash>..HEAD -- src/agents/ src/main/ src/utils/ src/poke_env/
```

For each commit that looks relevant (reward, training, model, observation, environment, launcher), get the actual diff:

```bash
git show <hash> --stat
git diff <last_hash>..HEAD -- <relevant_file>
```

Focus on: `reward_manager.py`, `features_extractor.py`, `battle_context.py`, `state_encoder.py`, `train_rl_agent.py`, `adaptive_lr_callback.py`, and any observation encoder files. But read whatever the commits actually touched.

## Step 3 — Write or update the doc

Add only what's in the git diff — do not speculate or add aspirational content.

For an **existing doc**, add new sections or extend existing ones for:
- New constants (name, value, purpose)
- Changed values (before → after, reason if in the commit message)
- New methods or callbacks (what they detect, when they fire, edge cases skipped)
- Updated reward signal table if rewards changed
- Updated files-changed table

For a **new doc**, follow the pattern from existing docs in the same folder:
- Start with a one-paragraph summary of what this step accomplishes and why
- List the key design decisions (the "why" not just the "what")
- Include a reward signal table or architecture diagram if applicable
- End with a files-changed table listing every file meaningfully modified

Match the heading levels, table style, and tone of the existing docs in that folder exactly — each version folder has its own conventions that emerged from the work done in it.

## Step 4 — Stage the doc (do not push)

```bash
git add <doc_path>
git commit -m "docs(<version>): sync <step_name> doc with recent changes"
```

Do not push — let the user `/gen3ai-ship` when ready.

## Step 5 — Summarise

Tell the user:
- Which commits were incorporated (list them by hash + message)
- The key changes documented
- One line on anything skipped (e.g. "skipped 2 minor refactor commits — no observable behaviour change")
