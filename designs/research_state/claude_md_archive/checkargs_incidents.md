# `checkargs` / launch-validation — the incidents behind the guards

Lifted verbatim from the root `CLAUDE.md` § *Will this command still launch?* on **2026-09-07**,
when that section was cut from 188 lines to ~55. Every guard it describes is LIVE; what follows is
the evidence — the G5 combination-check migration, the C1 inherited-config defect, the pinned-parser
arity defect, the relative-`models/` worktree defect, and the 2026-09-06 arch-drift incident.

The operating rules stayed in `CLAUDE.md`; the per-flag detail lives in `src/main/launcher/CLAUDE.md`.

---

### Will this command still launch? — `python -m main.checkargs`

A run's recorded `launcher_command` outlives the flags in it, and **argparse reports only the FIRST
unrecognized flag** — so relaunching an old argv after a deletion is a launch-crash-fix loop, ~40 s
and a stray run dir per stale flag. `checkargs` answers offline, in one pass, without touching
`models/`:

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.checkargs models/<run>                 # validate that run's recorded command
python -m main.checkargs --argv "--steps 1 --device cuda"
```

Exit 0 = every flag is accepted; exit 1 names each one that would fail (with its value, so you can
see whether it mattered). It checks **three** ways a command fails: a flag the parser no longer
knows; a COMBINATION the extractor refuses (`agents.model.flag_registry`'s `requires` graph, e.g.
`--intent-conditional` without `--damage-outgoing`); and a combination `resolve_config` refuses —
**every** value-conditional rule that is not `requires`-shaped, which lives in
**`main.train.combination_checks`** and is read by the launch path and by `checkargs` from that one
declaration. The last two crash later and dearer than an argparse error: the run dir exists and the
child starts.

🚨 **"EVERY" IS ENFORCED, NOT INTENDED — and it was not true until 2026-09-06.** The list held four
rules (the `--distill-target action` / top-K / gate / tau family from C1) while ~90 siblings stayed
as bare `parser.error` lines inside `resolve_config`, invisible here; G5's control arm was refused
three times in a row on three of them (`--distill-anchor-monitor` at `--distill-coef 0`,
`--distill-team-bias` with no teacher, then the inherited target) after `checkargs` printed
"✓ this command still launches". A partial single-source is one nobody can trust, because the output
does not distinguish "checked and clean" from "never asked". So `resolve_config` now evaluates the
whole list at one point (`refuse_first`, preserving each rule's own message and exit style —
`parser.error`, the two `--anneal-lr-*` prints, the CF duty-cycle `FATAL_CONFIG`), and
**`main/train/combination_checks_test.py` AST-scans `config.py` — resolving local aliases, which is
how three of them hid — and FAILS naming file:line if a cross-flag `parser.error` reappears.** Its
allowlist is EMPTY. Two consequences worth knowing: `checkargs` now runs the combination half on an
argv with **no `--model`** too (it used to skip it entirely, which is what let G5's fresh control arm
pass), and it runs `config.desugar_umbrella_flags` first so `--unified-moves`' default-'both' is
resolved the way a launch resolves it. A single-value RANGE check (`--distill-topk >= 1`) is out of
scope and stays put.
It also reports a **`--distill-teacher` spec that would teach NOTHING** — the grammar is
`<run|zip>[@<step>]:<teams|*>`, the `@step` is part of the SPEC and is split by the one canonical
`agents.training.run_spec.split_run_spec` (a `<run>@<step>:*` teacher used to resolve to zero teams
in silence, `gen3_run_spec_split_v1`), and `agents.training.distill_spec.check_teacher_spec` is the
single declaration both this tool and `resolve_config`'s refusal read.

🚨 **A BARE RUN DIRECTORY MEANS THE RUN'S LAST SNAPSHOT** (`gen3_last_snapshot_resolution_v1`,
2026-09-06). `--distill-teacher`, `--stable-opponents`, `--exploiter`, `--exploiter-ladder`,
`--warmstart-consensus`, `--distill-anchor-parent` and `--win-prob-pbrs-source` all resolve their
model file through the ONE choke point `agents.training.fixed_opponent_pool.resolve_model_ref`,
whose rungs for a bare dir are **`latest.txt` → the highest-step `checkpoints/` zip →
`final_model*.zip` → `best_model/best_model.zip` LAST** (a fallback for a run with nothing else,
printed as such), with the higher `num_timesteps` winning when the first three disagree. `<run>@<step>`
and an explicit `.zip` path — `best_model/best_model.zip` included — are still used verbatim, so
naming the file is how you pin it. It changes which FILE a run loads, never a weight shape, so it is
absent from `check_compatible` by design. Detail — the owner ruling, the measured 47,424-step gap
that makes the disagreement rule necessary, and what every teacher loaded BEFORE this change went
through — is in `src/agents/training/CLAUDE.md` → *WHICH FILE a run spec names*.

🚨 **AN ARGV IS NOT A CONFIG, and `checkargs` was argv-only until 2026-09-03.** With `--model`,
every flag the argv does not name is INHERITED from the checkpoint's recorded `model_config.json`
(`main.train.config`'s `_resolve`), so the thing that launches is the argv OVERLAID ON THE PARENT.
C1 is the third instance of the resulting class — its parent recorded `distill_target="action"`, the
argv said `--distill-coef 0` and named no target, `_resolve` inherited `action`, and the launch died
while `checkargs` had printed "✓ this command still launches". So with a `--model` present it now
builds that effective namespace — parsed argv, then every unset value filled through
`config.inherit_saved_flag` (the launch path's own function, called rather than re-implemented) —
and runs both combination checks on it, printing per finding whether the value came from the command
line or was `INHERITED`. The run dir the argv would write into is resolved the same way
`train_rl_agent` resolves it, and `fork_lr.is_same_run_checkpoint` labels the source a **FORK
PARENT** or a **same-run RESTART checkpoint**; both are read, because `_resolve` reads the recorded
config on every `--model`. A parent config that cannot be read is a **WARNING naming every path
tried**, never a silent pass, and the argv-only checks still run. A relative `models/...` path —
the `--model`, the run dir, the `<run_dir>` positional, a `--distill-teacher` run, the checkpoint
the pin is auto-derived from — resolves through `utils.paths.main_models_dir()` when it does not
exist relative to the cwd, so the inherited half works in a WORKTREE (where `models/` does not
exist and the whole check silently degraded to argv-only); an absolute or cwd-existing path is
untouched, no archive on the box still means the WARNING, and a REPEATED flag reads its LAST value
the way argparse does (a recorded command carries `--model` twice — reading the first pinned the
check to the fork PARENT's commit and refused four flags on a command that trained to completion).

**The old "absence carries no information" rule survives only where it is still true** — an argv
with no `--model` (or whose parent config is unreadable). There, the dependency half stays
conservative: it fires only when the argv enables a flag AND explicitly names a dependency with a
disabled value. Once the parent is read, an unset flag has a known value and an unsatisfiable
dependency is reported whether or not the argv mentions it. **It reports; it does not repair** — a
deleted flag may have a replacement, so dropping one silently could change the run. Launcher-owned
flags
(`--restart-interval-hours`, `--nice`, `--sync-to-main`, …) are recognised as not-forwarded rather
than reported as stale. **Run it after deleting flags**, over the recorded commands of any run you
might still relaunch or fork — that is what it is for.

🚨 **AND IT NOW ASKS THE OTHER QUESTION: IS THIS THE ARCHITECTURE YOU MEANT?**
(`gen3_arch_surface_guard_v1`, 2026-09-06.) **"it launches" and "it is the experiment" are
INDEPENDENT checks, and only the RESOLVED-CONFIG DIFF tests the second.** On **2026-09-06** the
first win-prob-critic arm was launched from a design document's 38-token command block — the critic
flags and the PPO knobs, none of the production feature flags, so every architecture flag silently
took its OFF default. Three gates ran and **all three passed**: `checkargs` exit 0, `resolve_config`
accepted, `--dry-run` clean. All three were right. The run trained a near-bare network for
**25,131 s / 24.4M steps / 6 checkpoints** — still holding the GPU when it was discovered a second
time — and **31 keys of its `model_config.json` differ from `designs/production_config.json`**
(every edge family off, zero entity seats, no belief slots, no event window, no intent heads).
Every number taken off it measures a different model. The gate that would have caught it
(`arch_tables_test`'s drift gate) fires only when someone runs the suite. The gap was never a
missing check — it was a missing QUESTION. Ledger: `2026-09-06 · INCIDENT`.

So `checkargs`, `--dry-run` and the launcher's own `_prepare_session` all print an **ARCH SURFACE
vs designs/production_config.json** block, from ONE function (`main.train.arch_surface.report` —
three copies of a guard is three things to keep in step). Its key set is DERIVED, never hand-listed:
`flag_registry.arch_surface_flags()` = the `structural` × `family=arch` rows, the toggles whose
mismatch means a DIFFERENT NETWORK. Everything else drops out by its own declaration —
`training_coef` and `runtime` are not architecture, `resume_immutable` leaves the forward identical,
and `family=critic` marks the readouts an experiment deliberately VARIES (`--critic winprob`
IMPLIES `win_prob_mode` and REFUSES `--value-dist-mode`, so gating them would refuse every critic
arm). The count is RECONCILED rather than merely smaller — every block prints
`39 arch + 7 critic + 3 non-structural = 49 registry rows`, because a guard that compares fewer keys
than a reader's own count leaves them unable to tell an excluded row from a forgotten one. On a
**FRESH** argv a non-empty diff **REFUSES** (exit 1 / `FATAL_CONFIG`), naming every key
with both values, unless the argv carries **`--allow-nonproduction-arch`**. A **FORK or RESTART is
exempt but still printed** — it INHERITS its parent's surface through `config.inherit_saved_flag`,
so its silence is the parent's architecture, not a bare one. A **PINNED** launch is ADVISORY for
`gen3_pinned_argv_parser_v1`'s exact reason: the mirror is THIS tree's, that commit has its own
registry and its own `production_config.json`, and `--arch` does not exist before 2026-09-06 — so
refusing there would be the same false POSITIVE that rule already fixed. **The diff is printed
either way**; dropping it on the pinned path is how the guard would silently stop working the day a
batch pins.

🚨 **AN ARCH-SURFACE REFUSAL AND A REFUSED FLAG COMBINATION ARE DIFFERENT FAILURES AND SHARE NO
MESSAGE, NO SUMMARY LINE AND NO REFUSAL PATH.** Rebuilding that arm from an older generation's
recorded `original_command` also fails — on nine flags the win-prob critic SUBSUMES (`--use-popart`,
`--value-from-dist`, the four `--value-dist-*`, `--value-dist-coef`, `--win-prob-coef`,
`--value-tail-weight`). That failure is **LOUD and PRE-launch**: `checkargs` names it, nothing
starts, it is fixed in a minute. Arch drift is **SILENT and POST-launch**: everything parses, the
run starts, and seven GPU-hours later the config diff is the only thing that would have told you. A
guard that catches the first is no protection against the second, so `checkargs` closes an
arch-only failure with its own verdict — *"✗ this command LAUNCHES — and builds the wrong
architecture"* — which fires only when no combination also failed.

**`--arch production` is the remedy and makes the runbook's `$ARCH` block one token**: it applies
every ARCH-surface key from `designs/production_config.json` as if typed, inside
`desugar_umbrella_flags` (so an explicitly-typed flag still wins and the C1-class inheritance rules
are untouched), and records `arch_source: "production_config@<12 hex of the mirror's git blob
hash>"` in `model_config.json` (config **v111**, provenance only — recorded, never gated, no
`ARCH_SIGNATURE` bump). It is refused on a resume, and it deliberately does NOT set the CRITIC
readouts, `--belief-grad-mode`, or the six **SUPERVISION DOSES** (`--move-belief-coef` ·
`--move-belief-latent-coef` · `--spread-belief-coef` · `--item-belief-coef` ·
`--hp-type-belief-coef` · `--intent-label-bot-weight`, declared by `ModelFlag.coef_arg` on the
toggle each supervises) — the block lists all of them, every time, so its silence is never read as
coverage. Measured against the incident's own config: of its 31 differing keys, **26 are refused on
the surface, 4 are named as doses, and the last is the enable coefficient of a refused surface
row** — not one can pass unmentioned. The mirror is read through
`agents.training.baselines.production_config_path()`, the registry's own accessor, so the guard and
`designs/baselines.json` cannot disagree about what "production" is. The launch path (`resolve_config`) prints the block and records
`arch_source` but does NOT refuse: it runs in the child, after the worktree and the run dir already
exist, so its refusal would be both late and a duplicate of `_prepare_session`'s.

🚨 **A PINNED argv is judged by the PINNED commit's parser** (`gen3_pinned_argv_parser_v1`,
2026-09-05). `--pin <sha>` validates against that commit's `build_parser()` instead of this tree's,
and with a `--model` present it does so automatically using the checkpoint's recorded `git_hash` —
printing which parser it used either way. **Which commit that is comes from the launcher's own
resolver** (`main.launcher.worktree.resolve_pin`, CALLED not re-derived: explicit `--pin-commit` >
`--sync-to-main` ⇒ HEAD > the checkpoint's hash), because a `--sync-to-main` fork runs the child at
HEAD and judging it at the parent's pin refused every HEAD-only flag (`--fork-lr`,
`--distill-anchor-monitor`, …) on commands that launch — a false POSITIVE, where this tool's three
earlier defects were false negatives. The launcher does the same for `--pin-commit`. **A FRESH
argv — `--pin-commit <sha>` with no `--model`, i.e. every arm of a batch that starts a new run on
one commit — is pinned too** (2026-09-06): the derivation used to be gated on a `--model`, so the
first win-prob arm's `--pin-commit e798c13a` was ignored and the command was judged at HEAD,
harmless only because HEAD *was* the pin that afternoon. The defect:
a flag whose **ARITY** changed is invisible to any presence check, so `--pin-commit b13b30b2` died on
`--hp-type-belief-coef: invalid float value: 'learned'` — the current parser abbreviation-matched a
deleted flag onto a surviving one. Detail (including what is NOT pinned): `src/main/launcher/CLAUDE.md`
→ *An argv is validated by the parser of the tree that will RUN it*.

**`python -m main.launcher --dry-run` is its EXECUTING complement, and it is the one that is safe
on a same-run RESTART** — `checkargs` answers "do these flags still parse and cohere?" from an argv
anywhere, while `--dry-run` resolves the actual launch on this box (role FRESH/FORK/RESTART, the run
dir, the pin + its subject, `+X steps` against the checkpoint, the INHERITED-vs-argv config, the
pool) and exits without creating a run dir, a worktree or a child. Reach for it instead of
"launching the real command and killing it after the startup lines": that habit is harmless on a
fork and DESTRUCTIVE on a restart, which is the 2026-09-05 incident `src/main/launcher/CLAUDE.md` →
*Validating a launch without launching* records.

It reads the parser's own `_actions` via **`train_rl_agent.build_parser()`** (extracted from
`main()` so the parser can be inspected without running a training job), not scraped `--help` text.
That distinction is load-bearing: `--help` was itself broken by one unescaped `%` — `"~0.6% of"`
renders as a space-flag `%o` conversion and raises `TypeError: %o format: an integer is required,
not dict` — and nothing rendered the help strings, so nothing caught it.
`checkargs_test.py::test_every_help_string_renders` is now that guard.
