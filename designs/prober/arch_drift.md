# Architecture drift — every way a run can be incompatible with HEAD is ONE typed diagnosis

Owned by this tree. The RULE is in `src/main/prober/CLAUDE.md`: a model-loading probe works only on a
run at the CURRENT architecture (owner decision) — bar the one class the GAP POLICY below decides by use (weights
that fit, an observation whose meaning moved) — the failure is an `ArchDriftError`, and a surface that
loads a model renders its **plain sentence** with the diagnosis folded under it rather than collapsing it
to "analysis failed". This file holds what the diagnosis is made of and where each kind is decided.

## The verdict: `main/prober/arch_status.py`

One pure module (stdlib + the model-version constants; no torch) answers "is this run / step /
checkpoint at the architecture THIS code builds?" for three callers: the run PICKER (cheaply, from JSON),
the per-step marks (`ProbeSession.model_status`), and `ProbeModel.load` (before the load, and by classifying
a failed one). A verdict is `status` (`current` · `incompatible` · `unrecorded`), a `kind` when incompatible,
and `plain` — ONE plain-language sentence a page shows verbatim:

> This run's architecture (config v143, signature gen3_event_record_v2) is older than the code (config
> v151, signature gen3_mon_tied_gain_v1); model views need a current-architecture checkpoint. *(+ what is
> specific to the kind)*

| kind | what it is | decided by |
|---|---|---|
| `arch_signature` | the network family differs, or the recorded config is below `MIGRATION_FLOOR` — the weights have no home | the record (pre-load) |
| `obs_dim` | trained on a different observation width than the encoder writes (`OFFSET_OBS_FACTS + OBS_FACTS_DIM`) | the zip's own layout (`peek_checkpoint`) or `total_dim` in the record (pre-load) |
| `obs_semantics` | **the silent class**: the weights fit, but the observation changed MEANING since `OBS_SEMANTICS_VERSION` (v151's Toxic cell n/8 → n/15, the Wish flag; v162's consumed-item op reads and stale volatile cells). The ONLY kind the gap policy lets a use through: a trace replay is refused, a fresh probe / stored score runs badged | the recorded `config_version` alone (pre-load) |
| `newer_than_code` | recorded by NEWER code than the one running (a stale server) | the record (pre-load) |
| `no_checkpoint` | nothing loadable — a trimmed skeleton, or the file is gone | the ladder finding no path / `os.path.exists` (`NoCheckpointError`, also a `FileNotFoundError`) |
| `unreadable` | the file is there but is not a zip | `zipfile.is_zipfile` (pre-load) |
| `state_dict` | the strict load rejected missing / unexpected / mis-shaped parameters | the failed load (`StrictLoadError`, "state_dict", "size mismatch", "shapes cannot be multiplied") |
| `config_value` | the code now rejects a value the checkpoint recorded, or no longer has a flag it names | the failed load (`TypeError` / `ValueError` / `ModelVersionError`) |
| `load_failed` | any other failure of the load itself — the catch-all, so nothing escapes untyped | the failed load |

Precedence when a record is wrong several ways: family → newer-than-code → width → meaning. Recorded evidence
beats the exception's shape (`ProbeModel._classify_failure`).

**Where the record comes from.** For a checkpoint FILE: the `eval_manifest.json` in its own directory (the
per-cycle record of the model that played; an eval `snapshot.zip` sits beside it), else the nearest
`model_config.json` going up (its directory, then the run root — three levels). For a RUN or a STEP: the
(newest / named) step's manifest, with `model_config.json` for anything it lacks. A record with nothing in it is
`unrecorded`: the views may be TRIED (and a failure is classified), but it is never counted as current — and a
signature without a version is `unrecorded` too, because only the version can vouch for the observation's meaning.

**From a shell:** `python -m main.prober.arch_status <run_dir | checkpoint.zip> [--step N] [--json]` prints the
verdict (status, kind, the plain sentence, recorded vs code identity; for a run, its traces and picker tier).
Exit 0 current, 1 incompatible, 3 unrecorded, 2 a path or step that does not exist. Records only: a run is judged
at its newest (or named) step, a checkpoint by its sidecars plus the width its zip declares; no torch import.

**Two sides, and the worse one wins.** The model views re-run a CHECKPOINT on a trace's STORED OBSERVATIONS.
`model_status(step)` is the worse of the checkpoint's verdict and the trace's own (`arch_status.worst`), and
`ProbeSession._model_for` refuses on the trace's verdict before loading when the gap policy refuses it for the
declared use — a FRESH probe feeds no stored observation, so a trace's semantics gap does not refuse it (an injected
test loader skips the trace check): on a
`nearest` / `recent` rung the checkpoint is not the model that played the step, and a run that resumed across an
observation change holds old-meaning traces beside a new-meaning checkpoint.

## The gap policy — version gap × use ⇒ allow / warn / refuse (`arch_status.decide`, 2026-10-26)

The verdict says WHETHER a record is HEAD's; what a caller may then DO depends on what it feeds the network. Every
`ProbeModel.load(path, device, *, use)` and `ProbeSession._model_for(battle, *, use)` / `probe_model` DECLARES a use
(a required keyword; an unknown one is a `ValueError`), and one pure function, `decide(verdict, use)`, reads the
table `GAP_POLICY`:

| gap (`gap_of(verdict)`) | `trace_replay` | `fresh` | `stored_score` |
|---|---|---|---|
| `none` — a CURRENT verdict | allow | allow | allow |
| `unrecorded` — no version record | warn | warn | warn |
| `obs_semantics` — weights fit, meaning moved | **refuse** | warn | warn |
| `weights` — every other incompatible kind | refuse | refuse | refuse |

- **`trace_replay`**: the checkpoint re-run on a trace's STORED observations and read against what it recorded. The
  forward runs HEAD's model-internal inputs (the op) over observations encoded when the trace was written, so across a
  gap it no longer reproduces what the model saw.
- **`fresh`**: positions encoded NOW by HEAD's encoder — the model reads inputs whose meaning moved since it trained;
  it runs, and says so.
- **`stored_score`**: stored observations scored with no faithfulness claim (a paired pre/post read moves both arms
  by the same gap).

A WARN carries the badge `⚠ trained on obs_semantics vN, probed at vM (<use>)`, with `changes_since` (every
`OBS_SEMANTICS_HISTORY` row newer than N). `GapDecision.as_dict()` is the `obs_semantics` block a caller records in
its output (`trained_semantics`, `code_semantics`, `action`, `gap`, `use`, `badge`, `changes_since`); `ProbeModel`
carries it as `.obs_semantics` and the verdict as `.arch_verdict`, and logs the badge at WARNING on load. The session
re-judges a CACHED model per use, so a model a fresh probe loaded is still refused to a trace replay.

| caller | use | output that records `obs_semantics` |
|---|---|---|
| `analyze`, `find disagree`, `battle_readout` / `decision_attention` (`_capture`), `probe`, `history_saliency` | trace_replay | `analyze`, `battle_readout`, `probe`, `history_saliency` |
| `main.probe_replay` | trace_replay | prints the badge |
| `lookahead`, `better_line` (+ its `--interior-opponent ckpt` load) | fresh | `obs_semantics` (+ `opponent_obs_semantics`) |
| `replay_counterfactual` (bare rollout models, decided from the records by `_play_semantics`) | fresh | `obs_semantics` (+ `opponent_obs_semantics`) |
| `main.harvest` scoring / its rollouts | stored_score / fresh | manifest `score.obs_semantics` / `rollout_obs_semantics` |
| `main.harvest_meter`, `winprob_finetune` | stored_score | `report.json` `obs_semantics` |

`gap_policy_test.py` pins the table cell by cell and, by AST over `src/`, the use every call site declares (a new
caller fails until it is classified). **Surfaces:** a WARN renders as a `p.banner.warn[data-obs-semantics]` on the
counterfactual job results, `/analyze` and `/game`'s model panel (`_macros.html` `obs_semantics_badge`); an
`obs_semantics` model-unavailable card names the fresh probes that still run (`data-fresh-probes`). ⚠ On `/analyze`
the counterfactual FORMS sit inside the analyze result, which a gap run refuses; the fresh probes are reached from
`python -m main.prober.query lookahead|better-line|replay-counterfactual` and the `/api/jobs/*` routes.

**Why a trace replay is refused although its stored obs ARE what the model saw.** Nothing re-encodes a stored trace;
the refusal is earned by the MODEL-INTERNAL inputs the forward computes at HEAD (v162: the op's consumed-item reads).
A future marker row that is purely an ENCODER change (stored obs unchanged, no model-internal input) would leave a
model's replay of its OWN traces exact; the table does not yet split `obs_semantics` that way.

## `OBS_SEMANTICS_VERSION` — the marker that makes the silent class detectable

`model_version/constants.py`. The last row of `OBS_SEMANTICS_HISTORY` — `(config version, one-clause reason)` per
version whose observation (and model-input) values mean something different, with or without a shape change: **151**
(the Toxic / Wish fix; v148 / v149's `x`-cell input fixes precede it, so one row covers them) and **162** (the
2026-10-26 GIGO audit: the op reads a consumed item as no item and the stale Destiny Bond / Grudge / Rage / Endure /
Snatch / Helping Hand / Mean Look `trapped` cells clear, `d8d674b9`; the lock-in and recharge counters end when
Showdown's do, `f9978586`). A checkpoint "trained on obs_semantics N" has N = the newest row at or below its recorded
config version (`arch_status.obs_semantics_of`). `ARCH_SIGNATURE` / `MIGRATION_FLOOR` only see weights. A commit that
re-scales or re-defines a cell without a shape change appends a row for the version it stamps. Nothing gates a resume
or an opponent load on it — `check_compatible` is unchanged; it is a READER's marker, acted on through the gap policy.
`agents/model/obs_semantics_test.py` pins the golden-vs-marker coupling: the obs golden's content hash is pinned
beside the marker, so re-recording the golden fails until the author has decided whether a cell changed meaning.
⚠ The pin sees what the golden battles' rows see (991 decisions); a change to a model-INTERNAL input they do not
move (an op edge cell) must be raised by hand.

## What the error carries

`ArchDriftError(message, kind=, plain=, saved_version=, current_version=, saved_obs_dim=, current_obs_dim=,
saved_arch=, current_arch=, git_hash=, dropped_kwargs=)`. `str(exc)` is the full diagnosis (the plain sentence
first, then the evidence — config version trained vs code, obs dim, arch_signature, dropped flags, the underlying
cause, **the exact `git checkout <hash>`** from the run's `metadata.json`, and which model-free views still work);
`exc.detail` is that message without the sentence it opens with (what a card folds UNDER the sentence).
`NoCheckpointError(ArchDriftError, FileNotFoundError)` is the `no_checkpoint` kind: typed for every surface, and
still a `FileNotFoundError` for the probes that treat "no model" as an answer (`probe`, `history_saliency`).

Surfaces: `/game`'s fragment renders `plain` with `detail` folded (`data-model-reason="<kind>"`);
`/partials/analyze` leads with `plain` (`data-arch-kind`) over the whole diagnosis (`.err` is pre-wrap);
the JSON routes (`/api/analyze`, `/api/game/readout`, `/api/game/attention`) answer 400 with `{error, kind,
plain}`; `python -m main.prober.query` prints `{error, kind, plain}`. **First paint:** `/game` and `/analyze` ask
`ProbeSession.model_status(battle)` (model-free) and, when the views cannot run, render the reason straight
away (`partials/model_unavailable.html`) — no loader to wait on, and no password prompt for a view the password
cannot unlock. The run picker, the context strip, the steps table, the step dropdowns (only a step that differs
from the newest is marked) and the checkpoint table all read the same verdicts.

## Measured (2026-10-26, the marker raised to 162)

Every current run records config v161 (`rb_x35_*` incl. N1-P, `rb_x26c_s2601`) or older (P\* =
`rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`, v158): all read `obs_semantics` (trained v151, code v162), the
picker files them under "older", and the views over their traces are refused (`query analyze` / `find … disagree` on
`rb_x35_d3mut_s3501` and on P\*'s own step: `{error, kind: obs_semantics, plain}`, before any load). Model-free
`query triage rb_x35_d3mut_s3501` is unchanged (98 losses, step 300000193). The fresh path on P\*: `query lookahead
<step_280000170/heuristic2/loss_003> --worst 1 --ckpt <P*>` ran (86 s, 0.86 GB peak, CPU) with `obs_semantics.action
= warn` and the badge on stderr; the web app on a 9XXX port rendered the `obs_semantics` card with `data-fresh-probes`
on `/game` and the badge banner on the `/api/jobs/lookahead` result. Readers that never load through the prober
(`main.anchors`, `main.h2h`, `main.reads`, `main.x35` incl. Kakuna, `main.team_strength`) import neither
`arch_status` nor `ProbeModel` and are untouched.

## The three walls behind the load-time classes (measured 2026-08-13)

Each used to surface as a raw error from inside SB3:

1. a DELETED flag still baked into the zip's `features_extractor_kwargs` → `TypeError: unexpected keyword
   argument 'spread_belief_nature_marginalize'`. **Recovered**: unknown kwargs are dropped
   (`_accepted_extractor_kwargs` introspects the live `__init__` signature) and *which* ones is reported — a
   dropped flag means the rebuilt extractor is not the one that played. ⚠ Recovery rate on today's archive is
   ZERO — every run carrying a deleted flag also differs in obs dim, so the drop alone never rescues one. It is
   unit-tested, not archive-proven, and exists for the *next* pure-flag deletion (v66 was exactly that shape).
2. a value the code now VALIDATES → `move_candidate_floor=0.0`, legal when trained, rejected since the v65
   legality guard (kind `config_value`). **Not** recovered: relaxing a correctness guard to probe would answer
   with a different model than the one under the microscope.
3. weight SHAPES that no longer fit (kind `state_dict`, or `arch_signature` / `obs_dim` when the record already
   said so). Not recoverable in principle.

`peek_checkpoint` reads the arch fingerprint from the zip's JSON `data` member in **~5 ms**, which is what makes
diagnosing cheap enough to do before the load is even attempted. `analyze`'s `model_resolution.dropped_kwargs`
carries a drop to a surface.

## Measured (2026-10-09, the real archive, 323 runs)

All 38 runs with eval traces are v136–v143 (`rb_st_*`, `rb_x5ab_*`): `arch_signature` (a v121–v143 config is
`gen3_event_record_v2`, the code's is `gen3_mon_tied_gain_v1`, floor 145) — **zero** are at HEAD, so the picker's
"Current architecture" group is empty until a v151 run captures traces. The 285 others are skeletons (weights +
logs, no `eval_traces/`) or launches that have not finished a first eval cycle. A HEAD-architecture CPU smoke
(`--debug --arch production`, config v151) loads and renders; the same weights re-stamped v150 are refused
(`obs_semantics`) — before this change they loaded without a word.

Tests: `main/prober/arch_status_test.py` (the verdicts and the record readers), `gap_policy_test.py` (the table, the
badge, every call site's declared use), `model_test.py` (every class end to end through `ProbeModel.load`, the
semantics-only case with a loader that WOULD succeed — refused for a trace replay, loaded badged for a fresh probe or
a stored score — and the session refusing / allowing per use on a gap run),
`web/runs_test.py` + `web/app_test.py` ("the run picker, by architecture"), `agents/model/obs_semantics_test.py`.

## From the prober leaf (moved 2026-10-10)

Moved verbatim-ish out of `src/main/prober/CLAUDE.md` when that leaf was cut to rules, commands and the map.

### The model-free / model-loading tiers, and the measurement behind them

**MEASURED 2026-08-13 over every run in `models/`: 79 runs carry a checkpoint, and 0 of them load
under current code.** Not one archived run is even at the current obs dim — the closest is 2667
against the code's 2669 (the v65 deadline clock's +2). What they were trained on: `2992` ×42 ·
`3409` ×8 · `3457` ×8 · `3469` ×6 · `2667` ×5 · `3390` ×3 · `3391` ×3 · `2889` ×3 · `2925` ×1.

This is **by design, not a bug** — the root `CLAUDE.md` says *"checkpoint compatibility is not a
concern"* and `ARCH_SIGNATURE` exists to reject stale checkpoints. But it decides how to read this
whole tool:

| tier | works on | why |
|---|---|---|
| **model-free** — `scan` · `triage` · `turns` · `battle_story` · `loops` · `overview` · `find` (bar `disagree`) · `falsify` · `falsify_scan` · `calibration` · `decision_table` | **every run, forever** | reads the trace on disk; no checkpoint |
| **model-loading** — `analyze` · `battle_readout` · `decision_attention` · `probe` · `lookahead` · `better_line` · `replay_counterfactual` · `history_saliency` · `find disagree` | **only a run at the CURRENT arch** (across an `obs_semantics` gap the three counterfactuals still run, badged: the gap policy above) | re-runs the policy under today's code |

So the durable surface is the model-free one, and it is not a coincidence that the web front end was
built there first. **The owner's ruling (2026-10-08): the prober supports only the CURRENT architecture**
("we are still rapidly iterating") — there is no pinned-checkout worker; on an older run `/game`'s
model panels render one plain sentence with the `ArchDriftError` diagnosis folded under it. A model-loading view is worth having for the run you are *currently training* and
stops working the day the obs layout moves.

🚨 **EVERY way a checkpoint can be unusable is ONE typed diagnosis (2026-10-09), and the model slots say it BEFORE
trying** — the owner: *"clean up the model selection; 99 % are irrelevant, and it doesn't even detect the new error
correctly."* Two defects behind that: a checkpoint whose weights FIT but whose observation changed meaning (v151's
Toxic cell re-scale) loaded without a word — nothing but `OBS_SEMANTICS_VERSION` (`model_version/constants.py`,
compared with the recorded `config_version` by `arch_status`) can see it; and a run with traces but no weights died as
a raw `FileNotFoundError`. Now `ArchDriftError.kind` is one of `arch_signature · obs_dim · obs_semantics ·
state_dict · config_value · newer_than_code · no_checkpoint · unreadable · load_failed`, its `plain` is one sentence
("This run's architecture (config vN, signature S) is older than the code (config vM…); model views need a
current-architecture checkpoint."), and `/game` / `/analyze` render that reason on first paint from
`ProbeSession.model_status` (model-free). The picker is the same classification (`web_service.md` §3).

**The three walls behind the diagnosis** — a deleted flag still baked into the zip's
`features_extractor_kwargs` (recovered by dropping unknown kwargs, and *which* ones is reported),
a value the code now validates (deliberately NOT recovered), and weight shapes that no longer fit
(not recoverable in principle) — plus what the error message names and the ~5 ms `peek_checkpoint`
that makes diagnosing cheap: `designs/prober/arch_drift.md`. Tests: `model_test.py`.
