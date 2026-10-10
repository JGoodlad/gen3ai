# Architecture drift — every way a run can be incompatible with HEAD is ONE typed diagnosis

Owned by this tree. The RULE is in `src/main/prober/CLAUDE.md`: a model-loading probe works only on a
run at the CURRENT architecture (owner decision), the failure is an `ArchDriftError`, and a surface that
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
| `obs_semantics` | **the silent class**: the weights fit, but the observation changed MEANING since `OBS_SEMANTICS_VERSION` (v151's Toxic cell n/8 → n/15, the Wish flag) | the recorded `config_version` alone (pre-load) |
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

**Two sides, and the worse one wins.** The model views re-run a CHECKPOINT on a trace's STORED OBSERVATIONS.
`model_status(step)` is the worse of the checkpoint's verdict and the trace's own (`arch_status.worst`), and
`ProbeSession._model_for` refuses on the trace's verdict before loading (an injected test loader skips it): on a
`nearest` / `recent` rung the checkpoint is not the model that played the step, and a run that resumed across an
observation change holds old-meaning traces beside a new-meaning checkpoint.

## `OBS_SEMANTICS_VERSION` — the marker that makes the silent class detectable

`model_version/constants.py`. The first config version whose observation (and model-input) values mean something
different, with or without a shape change; set to 151 by the Toxic / Wish fix (v148 / v149's `x`-cell input fixes
precede it, so one marker covers them). `ARCH_SIGNATURE` / `MIGRATION_FLOOR` only see weights. A commit that
re-scales or re-defines a cell without a shape change sets the marker to the version it stamps. Nothing gates a
resume or an opponent load on it — `check_compatible` is unchanged; it is a READER's marker.
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

Tests: `main/prober/arch_status_test.py` (the verdicts and the record readers), `model_test.py` (every class end
to end through `ProbeModel.load`, the semantics-only case with a loader that WOULD succeed),
`web/runs_test.py` + `web/app_test.py` ("the run picker, by architecture"), `agents/model/obs_semantics_test.py`.
