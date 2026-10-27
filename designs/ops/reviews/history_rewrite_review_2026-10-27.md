# Adversarial review: the git-history rewrite runbook and kit (2026-10-27)

**Subject.** `designs/ops/history_rewrite.md`, `src/utils/history_rewrite/` (`39c60355c`), `src/utils/git_sha.py` and the
call sites converted in `bd0860c8b`, and TECH_DEBT P1 (g). Reviewed at `main` = `e52dc800e`. Nothing was rewritten,
pushed or deleted, and no ref of the real repository was touched. The checks ran read-only against the real
repository, plus a synthetic end-to-end run on a throwaway repository (described next). No account name appears below:
an identity is `#NN`, as the kit prints it.

**Verdict: do not run it yet. 2 BLOCKERs, 9 MAJORs, 10 MINORs.** The core purge works: filter-repo with these rules
changes exactly the descendants of 14ae4203, and `--preserve-commit-hashes` does what the kit says. But the dry run
rehearses only steps 1-6. Phase A2 (moving the local refs onto the new history) fails part-way when executed. And
nothing stops the old history from coming back with an ordinary fast-forward push. The same gap hides two Phase B
failures. Every fix below is small. The highest-leverage one is an end-to-end test that EXECUTES `push-plan` against
a bare origin and `local-plan` on the source. It would have caught B1, M3 and M8.

## How it was attacked (what was executed)

- **Synthetic end to end** (in the job's scratch dir, deleted afterwards). `synthetic_repo_testkit.make_repo`, plus a
  bare `origin` with `main` and the tag pushed, a local-only branch, a detached pin worktree, and an agent branch
  worktree with one unshipped commit. Then the kit's own CLI: `names`, `rewrite`, `map`, `push-plan` (EXECUTED against
  the bare origin), `local-plan` (EXECUTED), a stale branch rebased and merged onto the new `origin/main`, the
  runbook's bundle-rollback command, and `local-plan`'s fetch with a local-only tag present.
- **Real repository, read-only.** The changed set recomputed today. Every tracked citation classified by what reaches
  the cited commit. Every compressed blob in history (633: `.gz`, `.zip`, `.npz`) decompressed and scanned for the
  names. Out-of-git sha256 pins of the JSON files the citation pass would edit. Byte copies of tracked read specs in
  read states. Run sidecars recounted. Other clones on the box. `git ls-remote origin`. The public GitHub API
  (unauthenticated). Software Heritage. The kit's 97 tests (pass).

## BLOCKERs

**B1. Phase A2 cannot complete: `local-plan` moves refs to commits that exist only in the throwaway clone.**
`local_plan` fetches from `origin` only. But the rewritten commits of every LOCAL-ONLY branch exist nowhere except
`$WORK/clone`: the agent and pin branches (`x35-*-pin`, `x59s-pin-a5`, the 14 `.claude/worktrees/*` branches; `origin`
has `main` and 11 unrelated branches only), and the keep-ref'd commits too. Executed on the synthetic repository, the
generated script moved `main` and one branch, then died at the agent worktree (`fatal: Could not parse object
<new sha>`, exit 128, `set -e`). It never reached the pin worktree. That leaves a half-moved repository during the
ship freeze, with `main` failing the static tier. Precondition 3 says agent branches are to be rebased `--onto` the
new base afterwards. `local-plan` instead emits `reset --keep <new>` for them: the two contradict each other.
*Fix:* before moving anything, fetch the clone's rewritten heads and keep-refs into a private namespace (`git fetch
$WORK/clone '+refs/heads/*:refs/history-rewrite/new/heads/*' '+refs/history-rewrite/keep/*:refs/history-rewrite/new/keep/*'`).
Emit `rebase --onto` for unshipped agent branches, as precondition 3 says. Extend `rewrite_test` so it EXECUTES both
generated scripts.

**B2. Nothing refuses the old history once the purge is done; one ordinary push restores the names.** After Phase A,
every pre-rewrite clone still carries the old commits: the ~14 agent worktrees' branches, and any cloud session's
container. Executed on the synthetic repository:
- `git merge origin/main` on a stale branch succeeds silently. The fast-forward push would carry the whole old
  history, with 15 name hits in `git log -p origin/main..HEAD`.
- `git rebase origin/main` replays the introducing commit and stops on add/add conflicts in the account-data files.
  `/gen3ai-ship` step 6 then tells the agent to "keep both intents".

`utils.push_guard` compares trees, so it cannot see history (the merged tree equals `main`'s). The ship freeze is an
instruction, not a mechanism, and it ends when Phase B lands, while stale worktrees live for days. *Fix:*
- `push_guard` refuses a push when the OLD full sha of 14ae4203 (a sha is not a name) or any old sha in the map is an
  ancestor of what is pushed.
- A freeze marker that `push_guard` honours from Phase A until Phase B.
- On GitHub: "require linear history" on `main`, and the owner stops or re-clones the cloud sessions before Phase A
  (their `claude/*` branches are pushed to `origin`).
- Precondition 3 becomes a check: no worktree branch below the old 14ae4203 survives A2 un-rebased.

## MAJORs

**M1. The repository is PUBLIC, and the runbook never says so.** `api.github.com/repos/JGoodlad/gen3ai` answers
unauthenticated: `visibility: public`, `forks_count: 0`, 1 star. The names have been fetchable since the push of
2026-10-25 11:40 PDT, and the old commits stay fetchable by SHA after the force-push, until GitHub Support acts.
Software Heritage has no origin for the repository (checked), and `origin` advertises no `refs/pull/*` (no PR was
ever opened). *Fix:*
- Say it in "Cannot be rewritten".
- Have the owner rule on an interim flip to private: it is reversible, stops anonymous fetch-by-SHA, and costs the
  star and watcher. First check what clones the repository anonymously (the cloud sessions; the "public token"
  memory).
- Run the real rewrite with filter-repo's `--sensitive-data-removal` flag (it ships in the pinned 2.47.0). Its "First
  Changed Commit(s)" output is the list GitHub's purge procedure asks for. Name every pushed old commit in the ticket,
  not only 14ae4203 and 742decb2.

**M2. The waived set is not pinned, so one stray mention silently un-purges an identity.** `build_rules` marks an
identity SCOPED whenever the audit finds it standing alone outside the account-data directory (`is_scoped = ... or
bool(outside)`). Say an agent quotes a flagged account in any doc before the rewrite. That identity then:
- stops being in the `--replace-message` rules, so its commit and tag messages stay;
- is left everywhere outside the account-data directory;
- has the new doc's path written into `rules.meta.json`, which is the allow-list `leakcheck` reads. Phase B's
  `leakcheck` then passes the leak.

The owner's ruling covers three identities. The kit re-derives the set on every run and only prints it. *Fix:* store
the ruled scoped indices and waived paths in `names.json` (or a ruling file beside it), and have `rules` refuse when
the audit's set differs. `leakcheck` allows only the ruled paths.

**M3. Most of the "pin-only" commits are garbage. Their new SHAs live only in the clone, and Phase B's `cite` will
fail against the real repository.** The runbook says 295 commits are reached "only by a detached worktree pin". Today
the descendants of 14ae4203 number 637: 246 reachable from a branch or tag, 40 only from a reflog, 351 dangling
(pre-rebase and amended agent commits), and 0 reachable only from a worktree HEAD (every pin worktree is also under a
tag or branch). 85 tracked citations in 10 files name a dangling or reflog-only commit: `src/main/reads/engine.py`,
`engine_test.py`, `designs/ops/slow_tier_status.json`, `designs/tools/reads.md`, two PREREGs, TECH_DEBT and three
measurement docs. `cite` re-points them to new SHAs that are never pushed and never fetched. The dry run's "all 191
tokens resolve" ran in the CLONE, where the keep-refs hold them. In Phase B, `verify_edits` resolves in the real
repository and returns 1, after `--apply` has already written. *Fix:* keep-ref only the commits a ref, a worktree HEAD
or `origin` reaches. Leave citations of garbage commits on their old SHAs, counted in the report. Run `verify_edits`
against the repository the edits ship in.

**M4. The citation pass rewrites a frozen lookup table that out-of-git records are matched against.**
`src/main/reads/engine.py` `LEGACY_ENGINE` maps the playing commit that `server_version` stamps to an engine. 6 of its
16 keys are rewritten (`46b9e572`, `de15dfc7`, `2cc864f2`, `d32ffcee`, `4ee65564`, and the dangling `033fcbcd`). The
banked units in `~/gen3ai_archive` (for example `x61p_n1p_read/look1`) keep stamping the OLD sha. After Phase B,
`legacy_engine(old)` returns `None` for N1-P look 1, D3 look 1 and the belief-prior-mix units. TECH_DEBT (g) says the
opposite: "NOT converted on purpose ... it already answers an old sha". That holds only if `cite` leaves the table
alone. Whether a `None` engine refuses loudly or degrades silently is UNVERIFIED. *Fix:* add `engine.py` and its test
to `cite.FROZEN` (the table is "FROZEN: never edited" by its own comment), or compare through `git_sha.current` on
both sides.

**M5. Live read states hold byte copies of specs the citation pass edits.** `find_pinned` only sees TRACKED citers.
`~/gen3ai_archive/su_headroom_2026-10-27/spec.json` and `~/gen3ai_archive/x41_d3_read/look2/spec.json` are
byte-identical today to `su_headroom_2026-10-27/read_spec.json` and `x41_d3_setmut_2026-10-25/read/read_spec_look2.json`.
Both are rewritten (1 and 3 citations). After Phase B, `main.reads` `plan` on those states refuses ("holds another
spec: a state is ONE read"), so the read is stranded or must be re-frozen. No out-of-git file cites the sha256 of any
of the 25 rewritable JSON files (scanned). *Fix:* freeze every `read_spec*.json`, plus every tracked file
byte-identical to a `<state>/spec.json` under the scan roots, and list them in `report`.

**M6. The recorded-SHA sites are gated only on Phase C, and the list is incomplete.**
- `main.ops.arm_chain.window` (`merge-base --is-ancestor <require_commit> HEAD`) and `x35.batch` (`is-ancestor` of a
  required commit against a pin) break as soon as one side is new and the other old: after A2 moves a worktree, or
  after Phase B re-points a spec. That is well before Phase C.
- Not on any list:
  - `main/reads/rules.py` (the VOID gate added in `e52dc800e` today). It compares `pin_history` spans to the
    registered pin by text prefix, so any run that spans the rewrite VOIDs.
  - `x35.batch.recorded_pins`.
  - `agents/model/snapshot.py`: a resume after the rewrite appends a second `pin_history` span, with the new sha, for
    identical code. `main.dose` and "replicate = same pin + seed" judgments then read it as a code change.

The runbook's precondition 2 allows a "parked" chain, which un-parks into these. *Fix:* convert every site before
Phase A. Make the closed TECH_DEBT row a precondition. Add one fake-map test each, `git_sha.current` on both sides of
every textual pin comparison, and a `snapshot.py` comparison through `git_sha.current`.

**M7. The quiet window and the ship freeze are not enforceable.** No script checks preconditions 2-6, and some lines
are judgments ("no lease that matters"). `worktree_busy` only reads `/proc/*/cwd`. The user timers
`gen3ai-replay-daily` and `gen3ai-prober-web-watchdog` run from the main checkout and are not mentioned. *Fix:* a
`preflight` subcommand that refuses on:
- a live chain (`main.ops.arm_chain live` non-empty), or a `main.reads` / `main.anchors` job or supervisor;
- a process whose cwd, exe or environment points into a worktree on a rewritten commit;
- a dirty worktree, or `main` != `origin/main`;
- a local-only tag (M8), a missing freeze marker (B2), or a missing bundle.

**M8. `local-plan`'s fetch deletes local-only tags.** `git fetch --force --prune origin ... '+refs/tags/*:refs/tags/*'`
prunes every local tag that `origin` lacks. Executed: `- [deleted] (none) -> local-only-play-tag`. Tags are
never-delete read pins (TECH_DEBT P3), and agents create them daily. Today local and origin agree (12 = 12), so this
is latent. *Fix:* fetch tags without `--prune`, and have preflight refuse on local-only tags.

**M9. Rollback is real only up to step 9.**
- The after-Phase-C command `git fetch $BK/pre-rewrite.bundle '+refs/*:refs/*'` fails (executed: `fatal: refusing to
  fetch into branch 'refs/heads/main' checked out`).
- Its refspec skips the bundle's `worktrees/<name>/HEAD` entries. The bundle does hold them: `git bundle create --all`
  on git 2.53 lists `worktrees/<name>/HEAD` and carries the detached pin (verified).
- No row covers "after Phase B, or after new commits land on the new history". A ref reset discards them, and their
  citations need the inverse map.

*Fix:* fetch into `refs/rollback/*` and `update-ref` per `refs.txt`. Add a row for after Phase B (rebase the new
commits `--onto` the old tip, then `cite` with the inverted map). State the point after which a rollback re-publishes
the names and loses work.

## MINORs

1. **The dry-run figures are stale within hours.** Changed commits: 480 then, 637 now. Tags containing the introducing
   commit: 5 then, 7 now. Citations: 2,614 / 191 distinct then, 3,118 / 281 now. Say "as of" and take the real
   numbers from the real run's own dry run on the frozen `main`.
2. **The determinism claim omits two conditions.** It does not mention the git version (2.53.0 here; fast-export and
   signature handling change across versions). The rules bytes are re-derived per run (see M2). Record `git --version`
   and the sha256 of `rules.txt` and `callback.py` in the map header. Refuse when they differ from the dry run, and
   diff the common map rows.
3. **`verify names` does not decompress.** Measured: of 633 compressed blobs in history, only identity `#69` stands
   alone (4 hits, all in `src/utils/team_corpora/w_forum_v1`, consistent with the waived forum curator). Add a
   decompressing pass (`.gz`, `.zip`, `.npz` with UTF-32 `.npy` members).
4. **The citation pass has no guard for non-SHA hex.** A 7+ digit number, or a float's fraction after `.`, can prefix
   a rewritten commit. Measured today: 0 (the two all-digit hits are real backticked SHAs). List all-digit tokens in
   the report for review.
5. **`worktree_busy` misses** processes that run a worktree's code from another cwd. The risk is low, because only the
   account-data files differ between the old and new commits.
6. **Phase C is incomplete.**
   - Its cleanup omits `rules.txt`, `callback.py` and `rules.meta.json`, which sit beside `names.json` and hold the
     names.
   - It never re-audits the real repository after `gc`.
   - `gc --prune=now` is unsafe beside concurrent git writers, so it needs the quiet window too.
   - Out of git and out of scope, but worth the owner knowing: 32 Claude session transcripts under
     `~/.claude/projects/` hold a name form.
7. **`push_plan` pushes every local tag whose target moved**, including one `origin` lacks. Its lease then fails and
   `--atomic` refuses the whole push. This is safe but surprising, and the docstring says "only refs on origin".
8. **`utils.git_sha` reads the CURRENT checkout's map.** A pinned worktree whose code postdates `bd0860c8` but
   predates Phase B carries an empty map, and cannot resolve an old sha after Phase C. Fall back to the main
   checkout's map.
9. **Not verifiable here:** issue and comment text, and branch protection (`gh` is not authenticated; the protection
   API answers 401). `open_issues_count` is 0.
10. **Decision 4's premise is stale.** `claude/elegant-faraday-5mmpgi` is now in the local object database. It is
    still not a descendant, so the conclusion holds.

## Checked and holding

| claim | how it was checked | result |
|---|---|---|
| `--preserve-commit-hashes` keeps a non-descendant whose message cites a rewritten commit | synthetic e2e (`side2`) | unchanged; changed set = descendants + unreachable only |
| the purge leaves no non-waived name | kit tests (97 pass); compressed-blob scan (above) | holds |
| the derivation found every name the account files held | strings present in any old version of the 23 account-data files, gone from HEAD, not among the 77 | 5 such strings, all code or schema strings |
| run sidecars are unaffected today | recounted `models/*/metadata.json` `git_hash` + `pin_history` | 0 runs on a rewritten commit (any run launched before the rewrite from `main` will be) |
| no other clone on the box | `.git` dirs to depth 6 under `~` and `/tmp` | only the main repository holds 14ae4203; every `~/dev/gen3ai-reads/*` is its worktree |
| `designs/baselines.json` | field walk | only `pstar_plateau.notes` cites a rewritten commit |
| refs | `for-each-ref`, `ls-remote` | no notes, no stash, no other namespaces; local tags = origin tags (12), 7 contain 14ae4203 |
| chain specs | `~/.claude/jobs/**/*.json` `pin` / `require_commit` | none names a rewritten commit today |
