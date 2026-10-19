---
description: Syncs a designs/ai_vN/impl_step*.md document with recent code changes. Use when the user wants to update or create a design doc after implementing something, sync docs with git history, or capture what was actually built. Triggers on "update design docs", "sync the doc", "update the step doc", "write up what we built", or any request to keep the designs/ folder current with recent commits.
---

# /gen3ai-update-design-docs

Sync one `designs/ai_vN/impl_step*.md` with what was actually built: a post-implementation record of decisions, constants and files changed, never future plans. These docs are explicit-only (root `CLAUDE.md` § Documentation Maintenance); the always-current docs (`designs/endstate/`, `ARCHITECTURE.md`, the eight owned trees) are updated with their code and are not this command's job.

1. **Target.** Use the path argument if given. Otherwise `git log --oneline -20 -- designs/ src/` names the active `ai_vN` folder; take its highest-numbered `impl_step*.md`, or the step its `todo.md` marks in progress. Read the target in full (and, for a new doc, two existing ones for headings and table style: each folder has its own conventions).
2. **What changed.** `git log --oneline -1 -- <doc>` gives the last sync point; `git log --oneline <hash>..HEAD -- src/` lists later commits. Read the diffs of the relevant ones (`git show <hash> --stat`, `git diff <hash>..HEAD -- <file>`), whatever files they touched.
3. **Write.** Add only what the diff shows: new constants (name, value, purpose), changed values (before, after, the reason if the commit says), new methods or callbacks, the files-changed table. A new doc opens with a one-paragraph summary and the design decisions (the why), and ends with a files-changed table.
4. **Commit, do not push.** `git add <doc> && git commit -m "docs(<version>): sync <step> doc with recent changes"` in a worktree; landing is `/gen3ai-ship`.
5. **Report** the commits incorporated (hash and message), the key changes, and any commits skipped as behaviour-neutral.
