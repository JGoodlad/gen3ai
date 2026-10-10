# Pinning, worktrees and which commit a run records

> Lifted out of `src/main/launcher/CLAUDE.md` (2026-10-10). Always-current: update it with the code.

## Worktree isolation and the startup prune

- **Worktree isolation** — at startup, creates a detached git worktree pinned to the commit
  `worktree.resolve_pin` chooses (`--pin-commit` > the checkpoint's recorded `metadata.json`
  hash on a resume > HEAD). Agent pushes to `main` never affect a running session.

  🚨 **THE STARTUP PRUNE REMOVES ONLY A DEAD LAUNCHER'S WORKTREE, and that is a 2026-09-05
  fix paid for with a run.** `_prune_stale_launcher_worktrees` used to force-remove EVERY
  `launcher-*` worktree it found, on the assumption that such a directory can only be a
  crashed session's debris. It is not: it is also every LIVE run's isolated checkout. A
  one-second validation command — `python -m main.launcher --pin-commit deadbeef --steps 1`,
  which exits `FATAL_CONFIG` before creating a worktree of its own — deleted a live
  production run's. The run kept going on its already-open file descriptors and looked
  healthy for hours; it died at its next 3 h periodic restart, when the launcher re-exec'd
  the child out of a directory that no longer existed (**exit 2, no `final_model`**). The
  resume then surfaced a second defect (see *Which commit a checkpoint records*).

  **The ownership record.** `_create_run_worktree` now writes a claim naming this process:
  `<worktree>.owner.json` — pid, its `/proc/<pid>/stat` **start time** (the pid-reuse guard;
  a pid alone is not an identity, `(pid, starttime)` is), the run dir, and `sys.argv`. It
  lives **BESIDE** the worktree, not inside it, so it cannot appear in that worktree's
  `git status`: the pinned tree is a real checkout of a real commit, and an ignore rule added
  today does not exist in a checkout of a commit from last month, while the child itself runs
  `git` in there. Being a `tempfile.mkdtemp` sibling, it is collected by the same /tmp
  cleanup, and `cleanup()` removes both.

  **The prune rule**, per `launcher-*` worktree — and every ambiguity resolves to KEEP,
  because a stale directory in /tmp costs disk and a deleted live one costs a run:

  | state | verdict |
  |---|---|
  | owner file present, pid alive (`os.kill(pid, 0)`) **and** its starttime matches | **KEEP** — a live run |
  | owner alive but starttime unrecorded / `/proc` unreadable | **KEEP** — cannot verify reuse |
  | owner pid gone, or alive with a DIFFERENT starttime (pid reuse) | remove |
  | no owner file (pre-fix worktree), mtime < 24 h | **KEEP** — may be a live pre-fix run |
  | no owner file, mtime > 24 h (`_LEGACY_ORPHAN_MAX_AGE_S`) | remove — abandoned |
  | the directory no longer exists | remove — the case git's own prune handles |
  | any "remove" above, but the tree holds **RUN DATA** (`utils.worktree_guard`) | **KEEP**, and say what it holds |

  🚨 **The last row, and the atexit `cleanup()`, run the SAME guard `scripts/land.sh` runs**
  (`_run_data_held`, 2026-09-23): refuse when a main-checkout `models/` symlink resolves into the
  tree, or when its untracked + ignored content outside the build/cache allowlist exceeds 50 MiB.
  A pin never holds a run — the child gets an ABSOLUTE `--run-dir` in the archive — but a launcher
  started from INSIDE a worktree used to put eight v9 runs in theirs (a cwd-relative `models/<run>`), and a
  forced removal of those worktrees destroyed them (ledger 2026-09-23); that path is now closed at its
  source (`run_archive_dir` / `checked_run_dir`), and this guard stays as the backstop. A guard that
  cannot run keeps the tree. Gate: `worktree_prune_test.py` (e), `src/utils/run_archive_test.py`.

  It **reports every decision** through a `report` callable (`state.add_event` from the
  launcher, `print` standalone), naming the owning pid on each skip — a startup that leaves
  debris behind must say so rather than look like a no-op. Gate:
  `worktree_prune_test.py`, over a real temp git repo: a current-pid worktree survives, a
  dead-pid one is removed, a reused pid is not mistaken for the owner, an unverifiable live
  owner is kept, legacy fresh/old split correctly, a non-`launcher-*` worktree is untouched,
  and the claim provably does not change the worktree's `git status`.

## An argv is validated by the parser of the tree that will RUN it — `pinned_argv.py`

🚨 **`--pin-commit` refused the exact command it exists for, and the reason is a class of drift no
presence check can see.** On 2026-09-05::

    python -m main.launcher --pin-commit b13b30b2 <the argv that run recorded> --dry-run
    error: argument --hp-type-belief-coef: invalid float value: 'learned'

At `b13b30b2` **`--hp-type-belief` TOOK A VALUE** (`learned`). Today that flag is deleted, so
argparse — which abbreviation-matches by default — resolved the token onto the surviving
`--hp-type-belief-coef` and handed it the value. (HEAD's parsers no longer abbreviation-match — deletion pass P11 —
so today that token is `unrecognized`; a PINNED commit's parser is judged by its own setting, which the probe now
reports as `ParseReport.allow_abbrev`.) Every argv check the launcher performed (its own
`--dry-run`, and `main.checkargs`) read the **CURRENT** tree's `build_parser()`, while the child
runs the **PINNED** tree's. **A same-named flag whose ARITY or TYPE changed is invisible to a "does
the parser still know this flag?" test**, because the current parser thinks it does — and the
result was that re-running an old recipe on its own commit, the one thing `--pin-commit` is for,
could not be validated at all.

**The rule now: when the resolved pin names a commit other than the HEAD of the checkout the
launcher is running from, every argv validation runs against the PINNED tree's parser.** Pin ==
HEAD is unchanged in every respect (no subprocess, no archive, the current parser) — pinned to your
own tree, the current parser IS the right one.

**And "the resolved pin" means `resolve_pin`'s answer, which `main.checkargs` now CALLS rather than
re-deriving** — explicit `--pin-commit` > `--sync-to-main` ⇒ HEAD > the checkpoint's recorded
`git_hash` > HEAD, with the chosen rule printed on the parser line: `checkargs` used to take the
checkpoint's hash whenever a `--model` was present, so every `--sync-to-main` fork was judged at its
parent's commit and exited 3 naming the HEAD-only flags it legitimately carries (reproduced on
`models/ai_v9_162_TCUNFA_0903`'s recorded command, a run that trained to completion).

🚨 **AND THE "ACCEPTED" BUCKET IS THE PINNED PARSER'S, NOT THIS TREE'S — the second half, and it
cost three arms on 2026-09-05.** The inverse of a deleted flag is a flag ADDED since the pin: it
parses cleanly against the parser you are standing in, so every "is this flag still known?" check
passes it, and the child — which runs the pinned tree — dies at startup on a flag its parser never
had. `python -m main.checkargs --pin b13b30b2 "<the v8 argv> --checkpoint-every-steps 150000"`
exited **0** and named nothing, because `--checkpoint-every-steps` is real *here*.

Every mode now reports its **OPTION SET**, and `flags_only_in_current_tree` names each argv flag
this tree knows and the pin does not. It is printed FIRST, above the summary, because it is the
finding a reader must not scroll past:

```
✗ NOT IN PINNED TREE @b13b30b2: --checkpoint-every-steps (exists only in the current tree)
```

In an authoritative mode that is a **refusal** — `FATAL_CONFIG` from the launcher and from
`checkargs` alike, one finding with one exit code. Under `ast_scan` it is a named **WARNING** and
never a `+1` in a count, because a reconstruction can be incomplete in both directions. An
unambiguous ABBREVIATION is not called absent (argparse abbreviation-matches, so a unique prefix
really does parse at the pin), and a launcher-owned flag never is either — `checkargs.forwarded_argv`
strips the launcher's own flags before asking, since a recorded `launcher_command` carries
`--restart-interval-hours 6` and friends that the child's parser has never heard of.

**How the pinned parser is obtained** (`pinned_argv.pinned_parser_check`, one subprocess):
`git archive <sha> -- src/main src/agents src/utils src/poke_env data designs/baselines.json designs/production_config.json` (`pinned_argv._ARCHIVE_PATHS`; a path the pin lacks is dropped — `src/poke_env` exists only at a pin before T27 P6) into a temp dir, copy
`pinned_argv_probe.py` beside it, run it with a **clean environment** (the caller's `PYTHONPATH`
names the *current* `src` in every worktree shell on this box, so inheriting it would silently
validate against the parser we are trying not to use). **Measured on this repo (2026-09-05, box
under a live run): `b13b30b2` 3.25 s cold / 2.89 s warm (`ast_scan`, 171 options); `HEAD~30`
3.33 s cold / 2.84 s warm (`build_parser`, 579 options).** The archive+extract is only ~0.4 s of
that — the rest is the probe subprocess importing the pinned tree — so the per-sha cache saves
little and the whole check is ~3 s either way. `data/` is 18 MB of the archive and `src/` 20 MB;
`src/rust_sim`'s 66 MB is excluded, since nothing on the parser path imports it. Time-boxed at
`DEFAULT_TIMEOUT_S` (180 s), and a timeout is UNAVAILABLE, never a pass.

**`git archive`, never `git worktree add`.** A worktree is a durable, registered, prunable object,
and a one-second validation command that touched the worktree list has already cost this program a
live production run (see the prune incident above). An archive is a read of the object database
and leaves nothing registered anywhere. `pinned_argv_test.py` asserts `git worktree list` is
unchanged.

**Four outcomes, and TWO of them are a verdict. Which one answered is printed on every run**
(`[mode=…]` in the summary line) — a reader must never have to guess whether the verdict came from
the real parser or a reconstruction of it:

| mode | what it is | a failure means |
|---|---|---|
| `build_parser` | the pinned tree's own `build_parser()` — the parser the child constructs | **`FATAL_CONFIG` (3)**, naming the offending token. The child would die on it ~40 s later, with a run dir already on disk |
| **`parse_args_hook`** | for every commit BEFORE `build_parser()` existed: run the pinned `main/train_rl_agent.py` as `__main__` with `argparse.ArgumentParser.parse_args` MONKEYPATCHED, so the first call hands us the fully-built REAL parser, answers, and exits | **`FATAL_CONFIG` (3)** — also AUTHORITATIVE, because the parser answering IS the parser the child builds. The only static thing about it is that the entry point is never allowed to run |
| `ast_scan` | a STATIC read of every `…add_argument(…)` call in the pinned `train_rl_agent.py` (+ `main/train/parser/*.py`), replayed into a synthetic parser carrying only each option's SPELLING and ARITY | a **WARNING** — the fallback when the hook times out or the pinned tree will not import on this box. A reconstruction can be incomplete, so it may not refuse |
| `unavailable` | `parser_unavailable_at_pin` — git failed, nothing was readable at all, or the probe timed out | a **WARNING** naming the reason, and the launch proceeds with the argv marked UNVALIDATED. **Never a silent pass** — but also never a refusal on a check we could not run |

**The `parse_args_hook`, and why it is safe to point at a real training entry point on a box
carrying a live run.** `build_parser()` landed 2026-08-16 (`26b28509`); every commit before it —
`b13b30b2` included — builds its parser inline inside `main()`, and the old answer was "that cannot
be called without starting a training job". It can: at b13b30b2 the first ~1080 lines of `main()`
are `add_argument` calls and `parse_args()` is the next statement, so a monkeypatch installed
**before the module is executed** intercepts the real parser before one line of work. The hook
writes its report and calls **`os._exit(0)`** from inside the call — not `sys.exit`, because a
`SystemExit` can be caught or wrapped by whatever the entry point had running (at b13b30b2 the call
sits inside `asyncio.run`), and "probably unwinds cleanly" is not a guarantee worth having. It runs
in a **child of the probe** with **cwd = the temp dir**, so a hang, a hard crash, or an
incompatibility with today's site-packages degrades to `ast_scan` instead of taking the probe down,
and a relative `models/` in the entry point could not reach the repo even if it ran. A parser
carrying fewer than `MIN_HOOK_OPTIONS` (20) option strings is declined and passed through — that is
some dependency's import-time argparse, not the trainer's. Time-boxed at
`HOOK_TIMEOUT_S = 120 s` (a pre-2026-08 tree imports torch on the way to its parser); the outer
probe box is 180 s so a hook timeout still reaches the fallback it exists for.
**Measured on the real `b13b30b2` (2026-09-05, box under a live run): 2.5 s, 369 options** — the
torch import is cheap enough that the hook, not the scan, is what answers in practice.
`pinned_argv_test.py` asserts a fake run-dir tree is **byte-identical** (sha256 per file) after the
probe, that no `models/` appears in the repo, and that the repo tree is unchanged.

⚠️ **A CUSTOM ACTION CLASS IS NOT A STRING** — `ast_scan` read `action=` only when it was an
`ast.Constant`, so `action=BoolFlag` (38 flags at b13b30b2) parsed as "no action ⇒ takes one value"
and the real v8 argv came back as `argument --self-play: expected one argument`, for a flag that is
a bare boolean there. A non-string action is now resolved from the pinned tree's OWN `class` body:
the `nargs` it passes to `super().__init__` **and** whether it generates `--no-<flag>` spellings
(`--no-stall-pbrs` is a real flag at that commit, declared by no `add_argument` call) are both read
out of the AST, so the reconstruction follows the pinned commit rather than a class name frozen into
the probe. An action class that cannot be resolved takes `nargs="?"` — the only arity that accepts
the flag bare AND with a value, so a scan that guesses wrong under-reports rather than inventing a
refusal.

`data/` is in the archive and that is not optional: `utils.paths.repo_root()` is
`__file__`-relative, so a pinned tree looks for its data beside itself and the `gen3_data` facade
raises `FileNotFoundError` at import — which would silently demote every recent pin from
`build_parser` to the static scan.

**Only the PARSER is pinned, and the other checks say so.** The extractor dependency graph
(`agents.model.flag_registry`) and the value-conditional refusals (`main.train.combination_checks`)
are still read from the current tree, so whenever a pinned check ran their findings print as
`ℹ️ ADVISORY — the CURRENT tree, not the pinned parser` and do **not** fail the dry run. That
inversion is the fix: judging a pinned argv by today's flag set is precisely what made
`--pin-commit` unusable.

**Where it is wired.** `run._prepare_session` (the real launch — checked **before**
`_create_run_worktree`, so a refusal creates nothing, and against `child_args` **with `--run-dir`
injected**, because that is the argv the child receives), `launcher/dry_run.py`, and
`main.checkargs --pin <sha>`. `checkargs` defaults the pin to the git_hash **recorded by the argv's
`--model` checkpoint** — the commit `worktree.resolve_pin` would pin a resume to — and always
prints which parser it used.

```bash
python -m main.checkargs --pin b13b30b2 --argv "--steps 1000 --hp-type-belief learned"
python -m main.checkargs models/<run>          # pins itself to that checkpoint's git_hash
```

Gate: `pinned_argv_test.py`, over a real **5-commit** temp repo whose `--flag` **changes arity**
across commits (a deleted flag would test the easy half) and two of whose commits build their parser
inline with a `BoolFlag`-shaped custom action: the same argv validates clean at commit 1 and is
refused at commit 2 with the token named, `--dry-run` exits 0 and 3 respectively, a pin naming HEAD
spawns no probe at all, commit 3 is answered by the `parse_args_hook` **and leaves a fake run-dir
tree byte-identical**, commit 4 (which will not import) degrades to the static scan where
`--self-play` / `--self-play true` / `--no-self-play` are all legal, and commit 5 reports
`parser_unavailable_at_pin` and still launches. Section (f) is DEFECT 1 — a REAL current flag
(`--checkpoint-every-steps`) absent at the pin must be named, must lead the block, must refuse
authoritatively, must only warn under `ast_scan`, and must not fire for a launcher-owned flag, a
flag missing from both trees, or an unambiguous abbreviation.

### Which commit a checkpoint records

🚨 **The pin only works if the recorded hash IS the code that ran, and until 2026-09-05 it was
not.** The run whose worktree the prune deleted (above) then failed to resume correctly,
because its checkpoint **sidecar** recorded `fff95a16` — the ambient HEAD of the main checkout
— while the run-level `metadata.json` recorded the actual pin `eb5261ff`. `resolve_pin` reads
the sidecar first, so the resume pinned the wrong commit. Two independent causes:

1. `agents.model.snapshot.record_checkpoint` resolved `git_hash or get_git_hash()`, never
   consulting `$LAUNCHER_GIT_HASH`. Worse, the truthy value it produced then **won** the
   `git_hash or env or …` chain inside `_build_snapshot_entry`, so that function's env
   fallback was dead code for the whole checkpoint path.
2. `utils.git.get_git_hash()` ran `git rev-parse HEAD` **in the process cwd**. The launcher
   puts the pinned worktree on the child's `PYTHONPATH` but spawns it with **no `cwd=`** (see
   the `PYTHONPATH` note below — that split is deliberate, so `models/` lands in the main
   checkout), so the child *imports* the pin while *standing in* un-pinned `main`.

Fixed at the root: `get_git_hash()` is anchored at `utils.paths.repo_root()`, the checkout the
code was **imported from** — in a detached launcher worktree that is the pin, in the main
checkout nothing changes. And **one resolver**, `snapshot.resolve_git_hash`, now serves the
run-level metadata and every sidecar: explicit argument → `$LAUNCHER_GIT_HASH` → the imported
checkout's HEAD, **raising `GitHashMismatchError` when the launcher's pin and the imported
tree name different commits** (a producer-side GIGO throw — a warning in a training child's
stdout is a line in a 1 MiB ring buffer nobody reads). Gate:
`src/agents/model/snapshot_git_hash_test.py`.

**`pin_history` — the scalar `git_hash` is "current", not "the code that ran".** It is
rewritten on every save, so on a run that restarts every 3 h it names the LAST code to touch
the run (observed on `ai_v9_171`: `eb5261ff`, then `fff95a16` after one resume). `metadata.json`
therefore also carries an **append-only** `pin_history` — `{git_hash, pin_source, first_step,
last_step}` per contiguous commit span, written by the same save path, with the same
immutability contract as `lineage` (an existing entry is never rewritten except to advance its
`last_step`). A legacy run with no history is seeded with one `derived: true` span from its
scalar hash, so *absent* never reads as *one commit*. Every checkpoint sidecar stamps the
history as of its write.

**`python -m main.sidecar_audit <models_dir_or_run>…`** is the offline reader (JSON only, no
torch, no `.zip` opened): per run it prints the run-level pin, the `pin_history` spans, and
every sidecar's hash, flags a run with >1 span as **PIN-SPLIT**, and separates a sidecar whose
hash *is* a recorded span (explained — a restart) from one that appears nowhere (misattributed
— the shape this defect leaves). `--json`, `-v`, and `--strict` (exit 1 on any unexplained
hash).

🚨 **THE PIN HAS FOUR SOURCES AND ONE DECISION FUNCTION** (`worktree.resolve_pin`, returning a
`PinDecision(sha, source, subject)`; every refusal is a `PinRefused` carrying the exit code to
leave with). In precedence order: an explicit **`--pin-commit`**, the resumed checkpoint's
recorded **`git_hash`**, **HEAD** under `--sync-to-main`, **HEAD** on a fresh run. The chosen
source is exported to the child as `LAUNCHER_PIN_SOURCE` and recorded as `metadata.json`'s
top-level **`pin_source`** (`"pin_commit"` / `"checkpoint"` / `"sync_to_main"` / `"head"`) beside
the `git_hash` it chose — so a finished run can say whether its commit was NAMED or inherited.

**Why `--pin-commit` exists, measured.** A batch of arms launched sequentially under
`--sync-to-main` each pins to HEAD *at its own launch*, so a commit landing mid-batch splits the
batch across two commits and nothing in any run's output says so — that happened on 2026-09-04
(arm 1 on `0c76e2ee`, arms 2-4 on `52ab5914`). Naming the commit on every arm removes HEAD from
the decision. It is the launcher half of the fix; the chain script's half is to record the pin
once and refuse to launch when HEAD has moved off it.

**A RESTART may never MOVE the pin, and that is the one case `--pin-commit` does not win.** The
launcher re-invokes the identical argv into the identical run dir every
`--restart-interval-hours`, so a `--pin-commit` that differs from the resumed checkpoint's
recorded `git_hash` would silently walk a live run onto other code every few hours. That is
`FATAL_CONFIG`, naming both commits and the three ways out. Fork-vs-restart is
`main.train.fork_lr.is_same_run_checkpoint`, **IMPORTED** — the same predicate `--fork-lr`, the
pool seeding and `resolve_fork_resume_model` key on. The **fork swap runs first** on purpose:
once an idempotent fork has its own progress, re-running its launch command is a RESTART of that
fork, so the guard is checked against the fork's own checkpoint rather than the source it was
originally forged from. Gate: `pin_commit_test.py`.

**The child's `PYTHONPATH` is what makes that pin real, and it must never be "cleaned up."** The
spawn passes no `cwd=`, so `PYTHONPATH=<worktree>/src` is the only thing making a resumed run
*import* the code its checkpoint was saved on. Measured (2026-08-22 scope survey, Finding B): with
an editable install present and no `PYTHONPATH`, a pinned old-commit child imports `agents` from the
**main checkout** — an old checkpoint silently resuming on current HEAD, the arch-drift disaster
class. `PYTHONPATH` entries land in `sys.path` *before* a `.pth`'s, so the pin and an editable
install coexist correctly exactly as long as that line stays. Note the deliberate split it creates:
the child **imports from the worktree** while the run it writes is the ABSOLUTE `--run-dir` the launcher
hands it (in the archive — main's `models/`), so a pinned child running an OLD commit that knows nothing of
`run_archive_dir` still lands there from any launcher cwd (verified 2026-10-02: a launcher in a worktree,
child pinned to HEAD, wrote only into the archive; `run_archive_test.py` pins the absolute `--run-dir` and
the no-`cwd=` spawn).

`models/` exists **only in the main checkout** and never in a worktree —
so anything else that needs the run archive must reach across rather than look beside itself.
`utils.paths.main_models_dir()` is that reach (via git's shared `--git-common-dir`, the same fact
`utils.git.get_main_repo_root()` reads); see the root `CLAUDE.md` § *Path discovery*.
