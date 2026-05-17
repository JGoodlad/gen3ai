# Sync Step Design Doc

Keeps a `designs/ai_v3/impl_step*.md` document in sync with recent commits.

## Usage

```
/sync-step-doc [step_doc_path]
```

- `step_doc_path` — relative path to the doc, e.g. `designs/ai_v3/impl_step4_reward_shaping.md`.  
  If omitted, infer from context (ask if ambiguous).

## What to do

1. **Find the last doc update commit** — run `git log --oneline -- <step_doc_path>` and take the most recent hash.

2. **Get all reward/training changes since then** — run:
   ```
   git log --oneline <last_hash>..HEAD -- src/agents/training/ src/agents/observation/ src/agents/model/
   ```
   Then for each commit that touches reward_manager.py, battle_context.py, features_extractor.py, or observation files, run `git show <hash> --stat` and `git diff <last_hash>..HEAD -- <relevant_file>` to read the actual diff.

3. **Read the current doc** — so you know what's already covered and don't duplicate.

4. **Update the doc** with every change not yet reflected:
   - New constants (value + purpose)
   - Changed scalar values (before → after, reason)
   - New methods (what they detect, when they fire, edge cases skipped)
   - Updated reward signal summary table
   - Updated files-changed table

5. **Commit the doc update** on the current branch (do not push — let the user /ship when ready):
   ```
   git add <step_doc_path> && git commit -m "docs(ai_v3): sync step4 doc with recent reward changes"
   ```

## Rules

- Only add what's in the git diff — don't speculate about future plans.
- Don't remove existing content unless it's now wrong.
- Keep the same section structure and table formatting as the existing doc.
- After updating, briefly summarise to the user: which commits were added and what the key changes were.
