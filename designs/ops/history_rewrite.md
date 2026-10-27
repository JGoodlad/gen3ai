# Purging account names from git history: the runbook

**Status: tooling and a dry run only. Nothing has been rewritten, pushed or deleted.** The real rewrite needs the
owner's go and the rulings in "Decisions first". This page is the procedure. The code is
`python -m utils.history_rewrite` (`src/utils/history_rewrite/`); the resolver is `python -m utils.git_sha`
(`src/utils/git_sha.py`); the map is `designs/ops/history_rewrite_sha_map.tsv` (header-only until the rewrite lands).

**Do not run it yet: the adversarial review of 2026-10-27 found 2 BLOCKERs and 9 MAJORs**
(`designs/ops/reviews/history_rewrite_review_2026-10-27.md`). `local-plan` fails part-way, because the rewritten
local-only branches exist only in the clone. And nothing refuses a stale branch that pushes the old history back.

**Why this is a procedure and not a command.** `git filter-repo` gives every commit from the first changed one onward a
new SHA, and this repository cites SHAs everywhere: ~500 design docs, the slow-tier rows, tag names, read records, run
sidecars in `models/` (immutable), `baselines.json`, `ledger.md`, the live pinned worktrees. The tooling therefore does
five things, each separately checkable: purge the names, record the old-to-new map, re-point the citations that can
be re-pointed, report the ones that cannot, and keep the old SHAs resolvable (`utils.git_sha`).

## What gets rewritten (the design, short)

| piece | rule |
|---|---|
| **the names** | derived from the results file commit 14ae4203 introduced (`accounts.flagged` keys and names, `accounts.top25` names) into a file OUTSIDE the repo (`~/.cache/gen3ai/history_rewrite/names.json`, mode 0600; the tool refuses a path inside a work tree that does not ignore it). Every message the tool prints speaks of `#NN`. |
| **global vs scoped** | an identity that stands alone only in the account-data files the two commits created (`designs/research_state/measurements/human_replay_quality_2026-10-25/`) is replaced EVERYWHERE (plain `literal:` rules from the exact spellings found). An identity that ALSO stands alone elsewhere (a plain English phrase, an older public replay fixture, a Smogon forum curator) is replaced ONLY in those account-data files, by a boundary regex in a `--file-info-callback`; its other occurrences are WAIVED and listed by index and path. |
| **what changes SHA** | exactly the descendants of 14ae4203 (`verify` proves it: a changed commit outside that set is named). `--prune-empty never` keeps the graph isomorphic; `--preserve-commit-encoding` keeps the one pre-existing commit with an `encoding` header from shifting every later SHA. |
| **commits no ref reaches** | a detached worktree pin descending from 14ae4203 is fetched into `refs/history-rewrite/keep/<sha>` before the rewrite so it gets a new SHA and a map row. Those refs are local bookkeeping and are never pushed. |
| **the citation pass** | a token of 7-40 lowercase hex not touching a letter or digit is rewritten ONLY when it is a prefix of exactly one commit of the OLD universe (every commit object of the original repo) and that commit is in the map. The new token keeps the length (a byte-budgeted doc keeps its size) unless that prefix is ambiguous among the new commits, when it is lengthened. |

## Citations the pass leaves on OLD SHAs (counted in `report`; resolve with `utils.git_sha`)

| left alone | why |
|---|---|
| `designs/research_state/ledger.md`, `ledger_index.md` (generated), `claude_md_archive/`, `designs/CHANGELOG.md` | append-only or generated; the pass counts what it would have written. The owner may overrule per file (`cite.FROZEN`), then `python -m main.ledger_index --write`. |
| `designs/ops/history_rewrite_sha_map.tsv` | its old column IS the map. |
| a file whose sha256 or git blob id another tracked file cites | editing it would break that pin (listed with its citers). |
| a token touching `_` or `-` | an identifier: a tag NAME (`x41-d3-look2-play-c4a5c37b`), a worktree or directory name (`pin_9c033489`). Tag names are KEPT; so are directory names until the worktree is moved. |
| a prefix of two or more old commits, one rewritten | AMBIGUOUS: refused and listed with `path:line`; cite more digits (or add the file to the allow list) and rerun. |
| `models/` sidecars (`original_command`, `pin_history`, `lineage`, `git_hash`), `~/gen3ai_archive`, any file outside git | immutable or out of reach; `report` counts them by JSON field. |

## Decisions first (owner)

1. **Tiers.** The brief names 61 accounts (`flagged`). The same results file lists 16 MORE accounts only in `top25`,
   ordinary Showdown ids that are just as much account names. `--tiers flagged` purges the 61 and leaves the 16 in
   history; the default purges all 77. Recommended: all 77.
2. **Waived occurrences.** Three identities (the scoped ones) also occur outside the account-data files:
   one is an ordinary English phrase that turns up in design prose (`design_human_replay_lever.md`,
   `learner_lifecycle.md`, `RESPONSE.md`), one is the player in a committed public replay fixture
   (`src/main/x50_r1/testdata/human_mimic_2679271755.log`), one is a Smogon forum curator in the forum-team corpora.
   They are NOT replaced there. If the fixture or the corpus must be scrubbed too, that is a content change (the
   fixture is a test input), not a history purge: decide, then edit it in a normal unit before the rewrite.
3. **Frozen files.** Leave `ledger.md` and friends on old SHAs (default; `utils.git_sha` covers them) or rewrite them.
4. **Signed commits.** filter-repo 2.47 strips a commit signature (and so changes the SHA). Four SSH-signed
   cloud-agent commits exist, all only on the remote-only branch `claude/elegant-faraday-5mmpgi`, which contains neither
   introducing commit. It is not in the local clone and not pushed; leave it.
5. **Local copies.** The old objects stay in the local repository (and the backup bundle) until the owner says
   `git gc --prune=now`; until then the names are still on this disk.

## Preconditions: the quiet window

Do not start until every line is true. All are read-only checks.

1. The owner's explicit go, with the decisions above.
2. **No live run, no pin mid-flight.** `python -m main.ops.arm_chain live` lists none (or the orchestrator has parked the
   chain: `python -m main.ops.arm_chain status`), `scripts/ops/gpu_lease.sh status` shows no lease that matters,
   `scripts/ops/mem_cap.sh --status` shows no `main.anchors` / `main.reads` job in flight, and no process stands in a
   `~/dev/gen3ai-reads/*` worktree. A resume or a read STARTED in the window would cite a SHA that is about to move.
   (Runs already live keep running: their pinned worktree is a directory, and the old objects stay.)
3. **No in-flight branches on old history.** The orchestrator freezes `/gen3ai-ship` from Phase A until Phase B lands; every agent
   worktree either ships first or is rebased afterwards (`git rebase --onto <new base> <old base>`, the old base
   from the map). `git worktree list` is the census; `local-plan` marks the ones that sit on a rewritten commit.
4. **main is clean and equals `origin/main`**, after `git fetch origin --prune`: `git -C /home/goodlad/dev/gen3ai
   status --porcelain` empty, `git rev-parse main origin/main` equal.
5. **Branch protection** on `main` allows the owner's force-push (a rewrite needs it).
6. **A backup that holds the names, kept private:** `git bundle create $BK/pre-rewrite.bundle --all`, plus
   `git for-each-ref > $BK/refs.txt` and `git worktree list --porcelain > $BK/worktrees.txt`, in an out-of-repo mode 700
   directory ($BK). Delete it only on the owner's word.
7. The tools: `python -m utils.history_rewrite fetch-filter-repo` (pinned 2.47.0, sha256 checked); the names file
   present (`names` re-derives it from the repository).

## Steps

`PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3`, `export PYTHONPATH=$PWD/src` from a worktree of the
tooling, and rule 5b while a run is live: `--cpus 5-7,13-15` (the tool sets nice 19 itself). `$WORK` is a NEW directory
outside the repository (it will hold a full clone, ~0.6 GB, and the names' rules).

```bash
$PY -m utils.history_rewrite names --out ~/.cache/gen3ai/history_rewrite/names.json         # counts only
$PY -m utils.history_rewrite audit                                                          # where every name stands, by #NN
$PY -m utils.history_rewrite --cpus 5-7,13-15 dryrun --work $WORK --gates                   # the whole rehearsal on a clone
```

The dry run IS steps 1-5 below on a clone; the real rewrite repeats the same commands, so nothing new happens on the
day. filter-repo is deterministic: the same repository, names file, tiers and filter-repo version give the same new
SHAs, so a dry run on the same `main` previews the real map exactly (new commits on `main` only append rows).

1. **Rewrite a clone.** `rewrite --work $WORK` = `git clone --no-local`, the keep-refs, one repack, `git filter-repo`
   (`plan` prints the exact commands). The real repository is only read.
2. **Map.** `map --work $WORK --out $WORK/history_rewrite_sha_map.tsv` (changed commits only; a pruned or duplicated
   row is refused).
3. **Verify.** `verify --work $WORK`: `names` (no identity stands alone in any blob, commit or tag message, ref name or
   path name, except the waived ones, which are listed), `variants` (a normalised search of every version of the files
   the two commits touched), `graph` (the commit count and every changed commit's authors, dates and parent count),
   `changed_set` (changed = descendants of 14ae4203), `tip_trees` (the tip tree of each shared branch is
   byte-identical: the names were already gone from the current files). Any FAIL stops the procedure.
4. **Citations on the clone.** `cite --tree $WORK/clone --map $WORK/history_rewrite_sha_map.tsv --apply`; it checks
   every distinct new token against git's own resolution in the clone and a second pass must rewrite 0.
5. **Gates.** `gates --clone $WORK/clone` (the static tier) before and after step 4.
6. **The report.** `report --work $WORK --tree $WORK/clone --out $WORK/cannot_rewrite_report.md --scan
   archive=/home/goodlad/gen3ai_archive` (the archive scan reads ~13 GB of text: ten-odd minutes of one core).
7. **Phase A, the force-push (the owner, by hand).** `push-plan --map $WORK/history_rewrite_sha_map.tsv --clone
   $WORK/clone` prints ONE `git push --atomic` with `--force-with-lease=<ref>:<old value>` for `main` and every tag
   on origin that contains the first changed commit (five of ten here), pushing nothing under
   `refs/history-rewrite/`. Lease values are the refs as `git fetch origin` last saw them: fetch again first. Do it
   from the CLONE; the push has no `/gen3ai-ship`, no gate, and that is deliberate (the citation unit ships through
   the gate afterwards).
8. **Check the remote.** A fresh `git clone` of origin into scratch, then `audit --source <it>`: the same waived list
   and nothing else; `git fsck`. Tell GitHub Support (see "Cannot be rewritten").
9. **Phase A2, this repository.** `local-plan --map ...` prints the commands that bring this repository onto the new
   SHAs: `git fetch --force --prune origin`, `update-ref <ref> <new> <old>` for each branch (the old value is the
   lease), and for each worktree `reset --keep <new>` (a branch) or `checkout --detach <new>` (a pin), IN PLACE, behind
   `test -z "$(git status --porcelain)"`. **It never removes a worktree**: a removal destroyed run data before, and
   a `models/<run>` symlink or a process cwd inside makes a worktree BUSY (`report` and `local-plan` flag both). Only
   files that differ between the old and the new commit change on disk. Review, then run.
10. **Phase B, the citations (a normal unit, in the same freeze).** Between A and B `main` FAILS the static tier
    (`status_doc_gate`: STATUS.md cites SHAs that are no longer in its history; measured below), so the ship freeze
    ends only when B lands. In a worktree of the new `main`: `map --work $WORK --out
    designs/ops/history_rewrite_sha_map.tsv --status "REWRITTEN <date>"`, then `cite --tree . --map
    designs/ops/history_rewrite_sha_map.tsv --apply`, `leakcheck --tree .`, the routine gate, `/gen3ai-ship`. Add a
    row to the root `CLAUDE.md` table: a cited SHA that no longer resolves -> this page and `python -m utils.git_sha`.

## Resolving an old SHA afterwards

`python -m utils.git_sha <sha-or-prefix>...` or `from utils.git_sha import new_sha, old_sha, resolve`. The map is
consulted FIRST: while the old objects still exist locally, git alone would answer with the commit that carries the
names. `mapped` = the new full SHA; `git` = a live commit (unchanged by the rewrite, or cited after it); `ambiguous`
(exit 2) names the candidates; `unknown` (exit 1) is not a commit here. Prefixes of several rewritten commits are
ambiguous, never guessed. Tag NAMES keep their old SHA text by design: the tag points at the new commit.

**Code that reads a RECORDED sha routes through it** (`commit_of` = `rev-parse --verify <x>^{commit}` through the map,
`current` = the new sha of a rewritten commit or the token unchanged, `is_ancestor`), so Phase C can collect the old
objects: `main.launcher.worktree` (`_rev_parse`; a resume pins to the sidecar's `git_hash`), `main.launcher.pinned_argv`,
`main.checkargs.pin_predates_commit`, `main.live.halt` (`--fixed-by`), `main.reads.playpath.resolve` (so `repin` and
`speccheck`), `main.reads.runner` (`check_code`, `_code_holds`) and `main.reads.repin`; each has a test that feeds an
old sha through a fake map (`utils.git_sha_testkit.use_map`) and fails on revert. **Still on git alone, deferred while the
arm chain is live (`TECH_DEBT_BACKLOG.md` row in (g))**: `main.ops.arm_chain.window` / `.spec`, `main.x35.batch`,
`main.x35.kit`: do not start Phase C before they are converted. `sidecar_audit` compares hashes textually among sidecars
and is unaffected. A new site that reads a recorded sha uses these helpers, never a bare `git rev-parse`.

## Cannot be rewritten

The `report` lists it with counts; the standing items are these.

- **GitHub.** The old commits stay fetchable BY SHA (web UI, API, `git fetch <sha>`) and in cached views until GitHub's
  own garbage collection; only GitHub Support can purge them. Open a ticket naming the old SHAs of 14ae4203 and
  742decb2 and the force-pushed refs.
- **Forks and other clones**, of anyone who ever cloned this repository.
- **`models/` sidecars** (`original_command`, `pin_history`, `lineage`, `git_hash`): immutable by rule, resolved by
  `utils.git_sha`. **`~/gen3ai_archive`** and any file outside git: counted, not edited.
- **The 16 or 3 waived occurrences** (decisions 1 and 2), the frozen files, the content-pinned ones, the identifier-embedded
  tokens (tag names, directory names).
- **This disk.** The backup bundle, the reflog and any worktree still on an old SHA hold the old objects.

## Rollback

| when | how |
|---|---|
| before step 7 | nothing to undo: the clone is throwaway and no ref of the real repository moved. `rm -rf $WORK`. |
| after the push, before step 9 | `git push --force-with-lease=refs/heads/main:<new main> origin <old main>:refs/heads/main`, and the same per tag with the old values from `$BK/refs.txt`. The old objects are in this repository and in the bundle. |
| after step 9 | `git update-ref <ref> <old> <new>` per `$BK/refs.txt`; `git -C <wt> checkout --detach <old>` per `$BK/worktrees.txt`. Possible until Phase C. |
| after Phase C | `git fetch $BK/pre-rewrite.bundle '+refs/*:refs/*'` restores every old object; then as above. |

## Phase C (later, on the owner's word)

All sidecar-reading call sites route through `utils.git_sha`; the GitHub ticket is answered; then `git reflog expire
--expire=now --all && git gc --prune=now` in this repository (and in each stray clone), `rm` the bundle, the dry-run
clones and the names file. Until the last of those, the names are on this disk.

## The dry run (measured 2026-10-27 on a `--no-local` clone of the main checkout, `dryrun --gates`)

Inputs: 3,709 commits reachable from the clone's heads and tags plus 295 that only a detached worktree pin reaches
= 4,004; 77 identities (61 `flagged` + 16 `top25`), of which 3 are scoped; git-filter-repo 2.47.0. One core at nice 19:
audit 85 s, clone + keep-refs + repack 15 s, filter-repo 4 min, verify 3 min; **12 min** end to end.

| check | result |
|---|---|
| names | PASS. 40,157 blobs (3,643 MB), commit and tag messages, ref names and tree paths: no identity stands alone anywhere except the 3 scoped ones at 7 paths (the old public replay fixture, 3 design-prose files, 3 forum-corpus files) and 2 commit messages. Before: 54 blobs held a name, 39 of them blobs that predate the introducing commit (all 3 scoped identities); the global ones sat only in `results.json`, the data-card `README.md` and `REVIEW_A_data.md`. |
| graph | PASS. 4,004 commits in, 4,004 out (nothing pruned); **480 changed** = 185 reachable from a head or tag + 295 that only a pin reaches; author, committer, dates and parent count equal on all 480. |
| changed set | PASS. The 480 are exactly the descendants of 14ae4203: 0 changed outside, 0 descendants unchanged. (Without `--preserve-commit-hashes` one more commit changed, the 2026-10-25 N1-P pin, because its message cites a later commit.) |
| tip trees | PASS. `main`'s tip tree is byte-identical (the names were already gone from the current files). |
| variants | REVIEW. One scoped identity (an ordinary phrase) matches normalised text in 5 touched design files: expected. |
| citations | 17,573 tracked text files scanned (556 binary skipped); **144 changed, 2,614 citations rewritten** (191 distinct tokens, 0 lengthened, 0 ambiguous). Left alone: 19 identifier-embedded, 4 frozen files (42 citations), 7 content-pinned read specs (7 citations; their sha256 is cited by the read's `cells.json` / `record.json` / PREREG). By extension: .md 60, .py 49, .json 22, .txt 6, .out 3, .sh 2, .jsonl 1, .rs 1; the code ones are mostly comments, and some are FUNCTIONAL (`src/main/x61/arms.py` `PIN`, the epoch table in `src/main/reads/engine.py`), which is why the pass does not skip code. |
| resolves | PASS. All 191 distinct new tokens resolve in the clone, through git's own `cat-file --batch-check`, to exactly the mapped commit; a second pass rewrites 0. |
| static tier | Before the citation pass 2 failures: `status_doc_gate` (STATUS.md cites SHAs no longer in its history: the predicted break) and the freshness gate's `data/mods/gen3` (an artifact of ANY fresh clone: the submodule is not checked out; the same failure on an un-rewritten clone). After: only the artifact; 276 pass. |
| determinism | A second rewrite of a clone taken 16 min later (18 new commits on `main` meanwhile) gave the identical new SHA for all 4,004 common commits. A dry run on the same `main` previews the real map exactly. |

Measured by `report` on the same clone (the archive scan read 205,465 text files in 6 min):

| outside the tracked tree | count |
|---|---|
| `models/` sidecars (`*/metadata.json`, `model_config.json`, `checkpoints/*.json`) | 4,535 files scanned, **0** cite a rewritten commit: with `--preserve-commit-hashes` every run pinned today sits on a commit the rewrite does not change (an earlier dry run without it moved the N1-P pin and put 66 sidecars, 516 citations in `original_command`, `pin_history`, `git_hash`, `core_stamp`, on a changed commit). A run launched in the window from a post-14ae4203 pin WOULD be affected: hence the quiet window. |
| `~/gen3ai_archive` text files | 2,078 of 205,465 cite a rewritten commit: 18,043 citations of 33 distinct commits, all read outputs (`cell.server_version`, `provenance.anchors_config`, `parts[].commit`, `commit`, `git_hash`, `pin`; under `x61p_n1p_read`, `belief_prior_mix_*`, `hctx_rc72_*`, `x41_d3_read`, `x59s`, `su_*`). Unchanged, resolved by `utils.git_sha`. |
| refs | 13 local branches and 5 tags move; 13 remote-tracking refs go stale (`git fetch --prune`); 63 local refs are untouched and 8 were created or moved after the clone (the report calls them stale: `main` moves all day, which is why the real run freezes shipping). The 5 tag NAMES embed an old SHA and are kept. |
| worktrees | 42 (primary included); 16 sit on a rewritten commit and are moved in place; 4 were BUSY during the report (three live reads' play worktrees, and the tooling's own). |
| read specs | 7 content-pinned `read_spec*.json` keep their one old SHA each (their sha256 is the pin): a consumer that resolves `pin` / `play_commit` from a spec must go through `utils.git_sha` or the read is re-frozen as a new record. |
