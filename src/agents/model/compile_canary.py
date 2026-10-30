"""The IN-RUN PARITY CANARY (K6, `gen3_compile_canary_v1`; persistence `gen3_compile_canary_v2`) — the compiled learner is still the same
function at update N.

THE GAP IT CLOSES. The startup region gate (`compile_regions.gate_regions`) proves the
compiled graph equals eager at t = 0, on the K9 learner golden's real labelled rows. Nothing proved it
afterwards: a graph that changes behaviour mid-run (a silent recompile into a miscompiling variant, a
cache artifact served under a key that omitted a setting — the K1b class — or a kernel that goes bad
on weights it never saw at t = 0) would train on, and the K9(b) behaviour gate cannot see it when the
rollout and the learner share the wrong function (the 2026-09-28 miscompile). The owner (2026-09-28):
"I don't review the code or the logs, I trust agents — stopping GIGO is so powerful."

WHAT RUNS. Every `CANARY_EVERY` updates (after `train()` returns, between updates), on the SAME
golden rows as the startup gate (`compile_regions.r1_batch`), through the DECLARED signature only (the
canary must never be an undeclared compile itself): the TRAIN graph at the UPDATE signature — train /
grad / batch `batch_size` — compiled vs eager: region R1's loss and every policy gradient (the
per-parameter bar chosen by `compile_regions.weights_regime`). It is the only compiled region (the
compiled rollout region R0 and its decision-readout arm were deleted, P10-E: the learner process
never ran R0, so its verdict checked a graph nothing used), and EVERY canary runs it — there is no
decision-only canary, which would check nothing.

The bars are the startup gate's (`train_verdict`), at fp32 matmul precision 'highest' — the one
precision. The weights are the live, trained ones, so vacuity is REPORTED, never a refusal.

PERSISTENCE, NOT A SINGLE SHOT (owner, 2026-10-01: "implement the consecutive check, up it to 100").
A real miscompile is deterministic and disagrees every time; a healthy graph's rare exceedance is a
property of one batch on one weight state. So a disagreement is CONFIRMED IN THE SAME UPDATE:

  1. warn, dump the verdict (`<run_dir>/canary_disagreements.jsonl`), and re-run the whole check —
     compiled AND eager — on the two DISJOINT halves of the golden rows (`compile_trainer.FIXTURE_HALVES`:
     slice 2 = the first half, slice 1 = the second half, each tiled to the same declared shape);
  2. CONFIRMED (it disagrees on BOTH halves) — checkpoint (`final_model_canary_fatal.zip`), then
     `CompileCanaryError` (FATAL_CONFIG, not restarted);
  3. not confirmed — `compile/canary_unconfirmed_disagreements` counts it, training continues, and
     the NEXT scheduled canary decides: a disagreement at TWO CONSECUTIVE scheduled canaries is
     FATAL even when neither confirmed.

WHY TWO DISJOINT HALVES (T36, 2026-10-15). Until T36 the confirmation re-ran the SAME rows (the same deterministic
arithmetic: it reproduces whatever the first check read) and "slice 1", the second half — a SUBSET of the scheduled
check's rows. A disagreement carried by ONE row (the `rb_toy_ent_s4001` FATAL: a row at the PPO clip boundary, row 35,
in both) therefore "confirmed" itself. The halves share no row, so a confirmation needs two row-disjoint sets to
disagree: independence by construction, not by chance. The scheduled check keeps ALL rows (a half can leave a
parameter below the per-parameter floor — the moves head on slice 1 since the 2026-10-12 golden rebuild — so the
halves alone would judge less); a miscompile only one half judges is FATAL by the consecutive rule below. Every check
also runs on the fixture moved off the PPO clip boundary (`compile_regions.relocate_clip_boundary`, the startup gate's
rows exactly), so no verdict depends on a row's 0/1 clip switch (standing rule 8).

THE YARDSTICK IS NOT THE COMPILED ARM (`gen3_r1_yardstick_margin_v1`, 2026-10-19; the X34 u1700 UNCONFIRMED). Eager and
the fp32 reference are what the compiled arm is measured against: one past the clip margin is a recorded
`yardstick_out_of_margin` event (`compile/canary_yardstick_events`), never counted toward confirm / FATAL — the rung
re-runs once at M' (`compile_regions.YARDSTICK_ARMS`) and is judged there. The log says "compiled DISAGREES with eager"
only when the compiled arm failed; a `compile_regions.YardstickRefusal` is worded as the yardstick. Every record names
each arm's worst row (batch and fixture row) and the same-precision compiled-vs-eager gap; an UNCONFIRMED or FATAL
record dumps the policy weights (`DUMP_WEIGHTS_MAX` per process). So does a PASS that read any judged quantity at >= 0.9
of its bar (`gen3_canary_near_margin_dump_v1`, the X26c u1000 PASS at 0.712 / 0.72: `NEAR_MARGIN_FRACTION`, rotated to
the newest `NEAR_MARGIN_KEEP` per run); every record carries each judged quantity's `bar_fractions`.

THE PER-ROW BAR, STAGE 1 (`gen3_canary_row_bar_v1`, ruled 2026-10-28; `canary_row_bar`). The per-row dither null
(`canary_row_null`, computed before the check on the live weights) is BINDING: the compiled arm's shift on each fixture
row is held to min(6 x max(null, 0.02), 0.72) in the scheduled check and both confirmation halves, the yardsticks keep
the scalar margin, and a null that cannot bind (off, skipped, truncated, in error) puts every row at the cap (the
global rule at M). Every record carries ``canary_rule_version`` and ``canary_rule`` (the constants, the bars' source,
the bf16 loss envelope's K), so a reader tells this regime from a record written before it (no field: the global M).

THE PER-PARAMETER DITHER NULL (`gen3_canary_param_null_v1`, 2026-10-30; `canary_param_null`). Under the bf16 rule a
judged parameter's bar is K x max(eager bf16's error, its fp32 dither null, the floor), K 12: the null (16 seeds of the
fp32 reference on the check's own rows at half-ulp-dithered weights) replaces ONE eager draw as the denominator's
floor, which a parameter carried by one to three fixture rows reads as a lottery (the slice-2 refusals; the 2026-10-30
GPU replay read all five states CONDITIONING). It runs only when the eager bar alone would refuse (the new bar is never
lower, so the verdict is the same as computing it always), in the scheduled check and both halves, and falls back to
the eager bar when it cannot bind. ``canary_rule_version`` `per_row_stage1_param_null`; ``canary_rule.param_null``.

Every verdict (update, step, pass / unconfirmed / fatal) is appended to
`<run_dir>/canary_verdicts.jsonl` WITH THE WATCH-LIST RECORD (`WATCH_PREFIXES`, `gen3_canary_watch_v1`): every
watched parameter's per-check numbers — under the bf16 rule its ratio, the compiled distance and eager bf16's
yardstick; under the fp32 rule its compiled-vs-eager error and bar — plus the check's worst parameter and global
numbers, for the scheduled check and each confirmation, so a parameter family's growth curve is recoverable from the
run (the 2026-10-15 `rb_toy_ent_s4001` FATAL had logged only each check's worst parameter). Every record also carries
`canary_shift_max` {value, source, margin}: the shift the clip margin was built from (`--canary-shift-max`, T37: the
code's `R1_LOGPROB_SHIFT_MAX`, or a RAISED value a same-run resume typed), so a record read alone says what bar it
judged by. The FATAL names the SAFE
ROLLBACK POINT: the newest checkpoint
written at or before the last PASSING canary's step (`rollback_point`; restarts included — the file
is the run's, not the process's).

WHAT IT DOES NOT DO: change training. It runs under `torch.random.fork_rng`, restores the policy's
training mode, and leaves every `.grad` as it found it (None between updates); the check reads
gradients through `.backward()` on R1's micro-step loss and clears them. Cost (measured on the
production surface by the K6 smoke, see `designs/training/learner_lifecycle.md`): one eager + one
compiled forward+backward at `batch_size` per canary (0.65-0.71 s at B = 2048, K6).
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from agents.model import canary_param_null
from agents.model import canary_row_bar
from agents.model import canary_row_null
from agents.model import compile_trainer as ct

#: Updates between canaries (owner, 2026-10-01: 100; a production update is ~36 s on K8, so ~1 h).
CANARY_EVERY = 100
#: The FIRST canary runs at this update, then every `CANARY_EVERY` (orchestrator, 2026-10-01): the
#: earliest point where training has moved the weights off the startup gate's state, so no run of
#: any length goes unchecked (sizing arm A — 82 updates — never reached update 100).
CANARY_FIRST = 10
CANARY_TAG = "[CompileCanary] FATAL"
#: The run-dir files the canary appends to (one JSON object per line).
VERDICTS_FILE = "canary_verdicts.jsonl"
DISAGREEMENTS_FILE = "canary_disagreements.jsonl"
FATAL_CHECKPOINT = "final_model_canary_fatal"
#: The DECLARED WATCH-LIST (`gen3_canary_watch_v1`, orchestrator 2026-10-15): parameter-name prefixes whose per-check
#: numbers every canary record carries (`compile_regions.GradWatch`). The pointer head is the family the
#: `rb_toy_ent_s4001` FATAL named; the record costs no extra compute (the rule computes these numbers anyway).
WATCH_PREFIXES: Tuple[str, ...] = ("pointer_head.",)
#: TEST HOOK (gen3_canary_memory_v1, 2026-10-15): run the FIRST scheduled canary at this update instead of
#: `CANARY_FIRST` — a GPU smoke reads the canary's memory and verdict at update 2 instead of 10. The cadence after it
#: is unchanged; a check never changes training (module docstring), so this moves only WHEN one runs. Unset in every
#: production launch.
FIRST_ENV = "GEN3AI_CANARY_FIRST"
#: THE WEIGHT DUMP (gen3_r1_yardstick_margin_v1, 2026-10-19): an UNCONFIRMED or FATAL verdict writes the policy's
#: `state_dict` (``canary_u<update>_s<steps>_<verdict>_policy.pt``, ~40 MB) beside its record, so the weights a
#: disagreement was read on can be re-read offline (the X34 u1700 weights were never saved, and its row is
#: unrecoverable). NAMED BY THE GLOBAL STEP and NEVER OVERWRITTEN (`gen3_canary_dump_no_clobber_v1`, 2026-10-25): the
#: update number restarts at 0 in every process, so a post-restart dump used to overwrite an earlier one of the same
#: update; a name already taken (a resume from the same checkpoint reaches the same update AND step) gets ``_r<n>``
#: (`_save_new`). At most this many per PROCESS (`_WEIGHT_DUMPS`, a module counter: no checkpoint carries it), like
#: K9(b)'s cap (`rust_rollout.consistency._DUMP_WEIGHTS_MAX`, T61).
DUMP_WEIGHTS_MAX = 3
_WEIGHT_DUMPS = 0
#: THE NEAR-MARGIN DUMP (`gen3_canary_near_margin_dump_v1`, 2026-10-25; the X26 continuation's u1000 PASS at compiled
#: 0.712 against M 0.72, `designs/research_state/measurements/x26c_canary_margin_2026-10-25/` F1: a PASS dumped nothing,
#: so the weights a 1 %-headroom read was taken on were never saved and the CUDA read cannot be replayed exactly). A
#: PASSING canary whose scheduled check read ANY judged quantity at >= this fraction of the bar it was judged at
#: (`bar_fractions`: the clip margin, the bf16 drift (loss, gradient), the bf16 envelope (loss, gradient) and the
#: per-parameter ratio) writes the same `state_dict` an UNCONFIRMED does, as ``canary_u<update>_s<steps>_near_policy.pt``
#: (the step makes the name unique across restarts: `self.updates` restarts at 0 in every process). A diagnostic
#: trigger, never a verdict: the canary's PASS stands whatever it reads.
NEAR_MARGIN_FRACTION = 0.9
#: ROTATED, per RUN (the files are the run dir's, so the cap spans restarts): after each near-margin write only the
#: newest this many ``*_near_policy.pt`` survive (~40 MB each), so a run whose healthy reads sit near a bar for a whole
#: era cannot fill the disk. A record naming a rotated-away file keeps its name (the record is history, the file is not).
#: Independent of `DUMP_WEIGHTS_MAX` (an UNCONFIRMED / FATAL dump is never rotated away by a near-margin one).
NEAR_MARGIN_KEEP = 3
NEAR_MARGIN_SUFFIX = "_near_policy.pt"
#: THE SHADOW PER-ROW NULL (`gen3_canary_row_null_shadow_v1`, 2026-10-25, orchestrator-approved; `canary_row_null`):
#: every canary records each golden row's fp32 dither null and each arm's per-row |Δ log π| / null as ``row_null``,
#: judging NOTHING — a CUDA census row for the per-row margin's k at every production canary. A test seam turns it
#: off per canary (`CompileCanary(row_null=False)`); ``$GEN3AI_CANARY_ROW_NULL_CAP_S`` 0 turns it off for a process.
ROW_NULL_ON = True
#: THE PER-PARAMETER DITHER NULL (`gen3_canary_param_null_v1`; `canary_param_null`): the bf16 per-parameter bar takes
#: max(eager, null, floor) in every check of a scheduled canary. A test / replay seam turns it off per canary
#: (`CompileCanary(param_null=False)`: the eager bar alone, a record before this rule); ``$GEN3AI_CANARY_PARAM_NULL_CAP_S``
#: 0 turns it off for a process.
PARAM_NULL_ON = True


def _first_from_env() -> Optional[int]:
    v = os.environ.get(FIRST_ENV, "").strip()
    if not v:
        return None
    try:
        n = int(v)
    except ValueError as exc:
        raise ct.CompileTrainerError(f"${FIRST_ENV}={v!r} is not a positive integer") from exc
    if n < 1:
        raise ct.CompileTrainerError(f"${FIRST_ENV}={v!r} is not a positive integer")
    return n


class CompileCanaryError(ct.CompileTrainerError):
    """The compiled learner no longer equals eager on the fixture at the startup gate's bars."""


def _yardstick_refused(exc: BaseException) -> bool:
    """The refusal's failing arm is the canary's YARDSTICK (eager / the fp32 reference) or the bf16 DRIFT rule, not
    the compiled arm (`compile_regions.YardstickRefusal`)."""
    from agents.model.compile_regions import YardstickRefusal
    return isinstance(exc, YardstickRefusal)


def _what(exc: BaseException) -> str:
    """The log's name for a refusal: "compiled DISAGREES with eager" ONLY when the compiled arm failed."""
    return ("the canary's YARDSTICK refused (eager / the fp32 reference, NOT the compiled arm)"
            if _yardstick_refused(exc) else "compiled DISAGREES with eager")


#: The most ``_r<n>`` suffixes `_save_new` tries before it gives up (diagnostics: the verdict stands either way).
_SAVE_NEW_TRIES = 100


def _save_new(model: Any, run_dir: str, stem: str, tag: str) -> Optional[str]:
    """Write the policy's `state_dict` as ``<stem>_<tag>_policy.pt`` in ``run_dir`` — or, when that name is TAKEN, as
    ``<stem>_r2_<tag>_policy.pt``, ``_r3`` … — and NEVER over an existing file (`gen3_canary_dump_no_clobber_v1`): a
    temp file, then `os.link` (which refuses an existing target), then the temp file removed. Returns the name written,
    None on an OSError or when every try is taken."""
    sd = {k: v.detach().cpu() for k, v in model.policy.state_dict().items()}
    tmp = os.path.join(run_dir, f".{stem}_{tag}_policy.pt.{os.getpid()}.tmp")
    try:
        torch.save(sd, tmp)
        for i in range(1, _SAVE_NEW_TRIES + 1):
            name = f"{stem}_{tag}_policy.pt" if i == 1 else f"{stem}_r{i}_{tag}_policy.pt"
            try:
                os.link(tmp, os.path.join(run_dir, name))
            except FileExistsError:
                continue
            return name
        return None
    except OSError:
        return None
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)


def _dump_weights(model: Any, run_dir: Optional[str], update: int, verdict: str, steps: int = 0) -> Optional[str]:
    """Write the policy's `state_dict` for an unconfirmed / FATAL verdict (`DUMP_WEIGHTS_MAX` per process) as
    ``canary_u<update>_s<steps>_<verdict>_policy.pt``, never over an existing file (`_save_new`). Returns the file
    name written, ``"cap"`` when this process has used its cap, None with no run directory or on an OSError
    (diagnostics: the verdict stands either way)."""
    global _WEIGHT_DUMPS
    if not run_dir:
        return None
    if _WEIGHT_DUMPS >= DUMP_WEIGHTS_MAX:
        return "cap"
    name = _save_new(model, run_dir, f"canary_u{int(update)}_s{int(steps)}", str(verdict))
    if name is None:
        return None
    _WEIGHT_DUMPS += 1
    return name


def _frac(value: Any, bar: Any) -> float:
    """``value / bar`` as a float; a non-finite value or a non-positive / non-finite bar reads +inf (it would not have
    passed, and an unreadable quantity is treated as at its bar, never as far from it)."""
    try:
        v, b = float(value), float(bar)
    except (TypeError, ValueError):
        return float("inf")
    if not (v == v and abs(v) != float("inf") and b > 0 and b != float("inf")):
        return float("inf")
    return v / b


def bar_fractions(watch: Any) -> Dict[str, Dict[str, Any]]:
    """EVERY JUDGED QUANTITY of one check, as the fraction of the bar it was judged at (`gen3_canary_near_margin_dump_v1`),
    from the live rung's `compile_regions.GradWatch` (its UNROUNDED numbers, so the 0.9 trigger is not a rounding read):

      * ``clip_margin`` — the largest over the arms of |Δ log π| from the fp32 reference / the bar it was judged at:
        a yardstick's against the margin (M, or the yardstick re-run's M′), the judged arm's worst row against that
        row's own bar (`canary_row_bar`); ``arm`` names the arm that carries it;
      * bf16 rule only (`_r1_verdict_bf16`): ``drift_loss`` / ``drift_grad`` (eager bf16 vs the reference, /
        `BF16_DRIFT_BARS`), ``envelope_loss`` / ``envelope_grad`` (compiled vs the reference, / `BF16_ENVELOPE_K` x
        max(eager's, the floor)), ``param_ratio`` (the worst per-parameter ratio, / `BF16_ENVELOPE_K["param"]`).

    A quantity the rung never reached (it refused before filling it, or the fp32 rule, which records no loss / cosine
    here) is absent. Only the FIRST route's live rung fills a watch (`r1_rungs`), so the other rungs are not read."""
    from agents.model import compile_regions as cr
    out: Dict[str, Dict[str, Any]] = {}
    m = getattr(watch, "margin", None)
    if m and m.get("max_abs_dlogp"):                  # gen3_canary_row_bar_v1: a judged arm against its row's own bar
        rb = (m.get("row_bar") or {}).get("arms") or {}
        cand = {a: ((rb[a]["fraction"], rb[a]["shift"], rb[a]["bar"]) if a in rb
                    else (_frac(v, m["margin"]), v, m["margin"])) for a, v in m["max_abs_dlogp"].items()}
        arm, (f, v, b) = max(cand.items(), key=lambda kv: kv[1][0])
        out["clip_margin"] = {"fraction": float(f), "value": float(v), "bar": float(b), "arm": arm}
    g = getattr(watch, "globals", None) or {}
    if getattr(watch, "rule", None) == "bf16" and cr.BF16_DRIFT_BARS is not None and cr.BF16_ENVELOPE_K is not None:
        drift = {"loss": ("drift_loss", cr.BF16_LOSS_FLOOR), "grad_rel_l2": ("drift_grad", cr.BF16_GRAD_FLOOR)}
        for what, (key, floor) in drift.items():
            if key in g:
                bar = cr.BF16_DRIFT_BARS[what]
                out[key] = {"fraction": _frac(g[key], bar), "value": float(g[key]), "bar": float(bar)}
            env = "envelope_" + key.split("_", 1)[1]
            if env in g and key in g:
                bar = cr.BF16_ENVELOPE_K[what] * max(float(g[key]), floor)
                out[env] = {"fraction": _frac(g[env], bar), "value": float(g[env]), "bar": float(bar)}
        w = getattr(watch, "worst", None)
        if w and "ratio" in w:
            k = cr.BF16_ENVELOPE_K["param"]
            out["param_ratio"] = {"fraction": _frac(w["ratio"], k), "value": float(w["ratio"]), "bar": float(k),
                                  "param": w.get("param")}
    return out


def _r4(v: Any) -> Any:
    """A float to 4 significant digits for the JSONL record (a non-finite one as its string, as `GradWatch.record`)."""
    if not isinstance(v, float):
        return v
    return float(f"{v:.4g}") if v == v and abs(v) != float("inf") else str(v)


def near_margin(fractions: Dict[str, Dict[str, Any]]) -> List[str]:
    """The quantities of `bar_fractions` at >= `NEAR_MARGIN_FRACTION` of their bar (in its order)."""
    return [q for q, d in fractions.items() if not d["fraction"] < NEAR_MARGIN_FRACTION]


def _dump_near_margin(model: Any, run_dir: Optional[str], update: int, steps: int) -> Optional[str]:
    """Write the policy's `state_dict` for a near-margin PASS (atomically, never over an existing file: `_save_new`),
    then ROTATE: keep only the newest `NEAR_MARGIN_KEEP` ``canary_*_near_policy.pt`` of the run dir (by write time,
    then name). Returns the file name written, None with no run directory or on an OSError (diagnostics: the PASS
    stands either way)."""
    if not run_dir:
        return None
    name = _save_new(model, run_dir, f"canary_u{int(update)}_s{int(steps)}", NEAR_MARGIN_SUFFIX[1:-len("_policy.pt")])
    if name is None:
        return None
    with contextlib.suppress(OSError):
        kept = []
        for f in os.listdir(run_dir):
            if f.startswith("canary_") and f.endswith(NEAR_MARGIN_SUFFIX):
                with contextlib.suppress(OSError):
                    kept.append((os.stat(os.path.join(run_dir, f)).st_mtime_ns, f))
        kept.sort()
        for _t, f in kept[:-NEAR_MARGIN_KEEP]:
            if f != name:
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(run_dir, f))
    return name


def _run_dir(model: Any) -> Optional[str]:
    """The run directory the trainer declared (`model.behaviour_dump_dir`, set by `model_build` for
    K9(b)'s dumps), else None (a test learner: nothing is written)."""
    d = getattr(model, "behaviour_dump_dir", None)
    return str(d) if d else None


def _append(path: Optional[str], name: str, record: Dict[str, Any]) -> None:
    if not path:
        return
    with contextlib.suppress(OSError):
        with open(os.path.join(path, name), "a") as f:
            f.write(json.dumps(record) + "\n")


_CKPT_STEP = re.compile(r"(\d+)_steps\.zip$")


def rollback_point(run_dir: Optional[str]) -> str:
    """The SAFE ROLLBACK POINT for a canary FATAL: the newest checkpoint (`checkpoints/*_<N>_steps.zip`)
    written at or before the last PASSING canary's step in `<run_dir>/canary_verdicts.jsonl`. A
    one-line human answer, never an exception."""
    if not run_dir:
        return "unknown (no run directory declared)"
    last_pass: Optional[int] = None
    with contextlib.suppress(OSError, ValueError):
        with open(os.path.join(run_dir, VERDICTS_FILE)) as f:
            for line in f:
                r = json.loads(line)
                if r.get("verdict") == "pass":
                    last_pass = int(r["num_timesteps"])
    if last_pass is None:
        return "none: no canary of this run has PASSED (the startup gate's weights are the only known-good state)"
    best: Optional[int] = None
    best_path = None
    ck = os.path.join(run_dir, "checkpoints")
    with contextlib.suppress(OSError):
        for name in os.listdir(ck):
            m = _CKPT_STEP.search(name)
            if m and int(m.group(1)) <= last_pass and (best is None or int(m.group(1)) > best):
                best, best_path = int(m.group(1)), os.path.join(ck, name)
    if best_path is None:
        return f"none: no checkpoint at or before the last passing canary (step {last_pass:,})"
    return f"{best_path} (step {best:,}; the last passing canary was at step {last_pass:,})"


class CompileCanary:
    """Runs the canary on its cadence; see the module docstring. ``saver(path)`` writes the FATAL's
    checkpoint (default `model.save`; a test seam)."""

    def __init__(self, model: Any, *, batch_size: int,
                 emit: Optional[Callable[[str], None]] = None,
                 every: int = CANARY_EVERY,
                 saver: Optional[Callable[[str], None]] = None,
                 first: Optional[int] = None,
                 row_null: Optional[bool] = None, param_null: Optional[bool] = None) -> None:
        self.model = model
        self.batch_size = int(batch_size)
        self.every = max(1, int(every))
        # the first canary: `CANARY_FIRST` at the production cadence; a custom cadence (tests) starts
        # at its own period unless told otherwise; `$GEN3AI_CANARY_FIRST` (a smoke's test hook) wins
        if first is None:
            first = _first_from_env()
        self.first = int(first) if first is not None else (CANARY_FIRST if self.every == CANARY_EVERY
                                                            else self.every)
        self._emit = emit
        self._saver = saver
        self.updates = 0
        self.runs = 0
        self.unconfirmed = 0            # cumulative: disagreements that did not reproduce
        self.yardstick_events = 0       # cumulative: yardstick-out-of-margin events (recorded, never counted)
        self.memory: Dict[str, float] = {}   # the last check's card (gen3_canary_memory_v1; CUDA only)
        self.prev_disagreed = False     # the previous SCHEDULED canary disagreed (unconfirmed)
        self.last: Dict[str, float] = {}
        #: gen3_canary_row_null_shadow_v1: compute + record the shadow per-row null at every canary (`ROW_NULL_ON`
        #: unless the caller says; `canary_row_null`'s cap can still skip it, recorded)
        self.row_null_on = ROW_NULL_ON if row_null is None else bool(row_null)
        if self.row_null_on:
            try:                                        # a malformed cap refuses at STARTUP, never at a canary
                canary_row_null.cap_seconds()
            except ValueError as exc:
                raise ct.CompileTrainerError(str(exc)) from exc
        #: gen3_canary_param_null_v1: the bf16 per-parameter bar takes the dither null (`PARAM_NULL_ON` unless told)
        self.param_null_on = PARAM_NULL_ON if param_null is None else bool(param_null)
        if self.param_null_on:
            try:                                        # likewise: a malformed cap refuses at STARTUP
                canary_param_null.cap_seconds()
            except ValueError as exc:
                raise ct.CompileTrainerError(str(exc)) from exc
        self._row_null_raw: Optional[Dict[str, Any]] = None
        self._row_bars: Optional[canary_row_bar.RowBars] = None     # gen3_canary_row_bar_v1: this canary's bars
        self._w0: Optional[Any] = None

    def after_update(self) -> Dict[str, float]:
        """Count one update; run the canary when due. Returns the TB scalars it produced (``{}`` on
        a skipped update — the cadence writes nothing, never a stale value)."""
        self.updates += 1
        if self.updates != self.first and self.updates % self.every:
            return {}
        return self.run()

    # ---------------------------------------------------------------------------- one verdict
    def _check(self, slice_: int, out: Dict[str, float], watch: Optional[Any] = None,
               row_bars: Optional[canary_row_bar.RowBars] = None) -> List[str]:
        """One full comparison (compiled AND eager re-computed) on fixture slice ``slice_``. Raises
        `CompileTrainerError` on a disagreement; restores training mode and `.grad` either way. ``watch``: a
        `compile_regions.GradWatch` the live rung fills (whatever the verdict). ``row_bars``: the per-row clip-margin
        bars the compiled arm is judged by (`canary_row_bar`); None = the scalar margin (a replay of a global-M record,
        the dry update's memory read)."""
        from agents.model import compile_regions as cr
        model = self.model
        policy = model.policy
        was_training = bool(policy.training)
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        mem = cr.GateMemory(model)
        try:
            with torch.random.fork_rng(devices=devices), cr.gate_memory(mem):
                return self._regions(model, out, slice_, watch, row_bars)
        finally:
            policy.set_training_mode(was_training)
            for p in policy.parameters():
                p.grad = None
            cr.release_gate_memory(model)       # gen3_gate_releases_memory_v1: no stashed graph outlives the check
            if mem.rows:                        # gen3_canary_memory_v1: the check's card, with training state resident
                self.memory = {"start_alloc_mib": round(mem.start[0], 1),
                               "transient_mib": round(mem.peak_above_start(), 1),
                               "peak_alloc_mib": round(mem.start[0] + mem.peak_above_start(), 1),
                               "reserved_mib": round(max(r[3] for r in mem.rows), 1)}

    def measure(self) -> Optional[str]:
        """The scheduled check's comparison ONCE, for its MEMORY only (`update_fit.dry_update`'s dry canary,
        gen3_update_fit_v2): the same arms on the same rows as `run`, but no verdict is recorded, nothing is counted
        and no state changes (`_check` restores training mode and `.grad`, under `fork_rng`). Returns None, or the
        disagreement's text (a startup check on the weights the gate just passed — reported, never a verdict here)."""
        try:
            self._check(0, {})
        except ct.CompileTrainerError as exc:
            return str(exc)[:500]
        return None

    def _record(self, verdict: str, detail: str = "", watch: Optional[Dict[str, Any]] = None,
                fractions: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        rec: Dict[str, Any] = {"verdict": verdict, "update": self.updates,
                               "num_timesteps": int(getattr(self.model, "num_timesteps", 0) or 0),
                               "time": time.time(), "detail": detail[:4000],
                               "canary_shift_max": self._shift_max_record(),
                               "canary_rule_version": self.rule_version,
                               "canary_rule": self._rule_record()}
        if fractions is not None:                           # gen3_canary_near_margin_dump_v1: every judged quantity
            rec["bar_fractions"] = {q: {k: _r4(v) for k, v in d.items()} for q, d in fractions.items()}
        if self._row_null_raw is not None:                  # gen3_canary_row_null_shadow_v1: recorded, never judged
            rec["row_null"] = canary_row_null.record(self._row_null_raw, getattr(self._w0, "row_dlogp", None),
                                                     judged=self._bars_live())
        if verdict != "pass":                               # gen3_r1_yardstick_margin_v1: the weights it was read on
            rec["policy_pt"] = _dump_weights(self.model, _run_dir(self.model), self.updates, verdict,
                                             rec["num_timesteps"])
        elif fractions and near_margin(fractions):          # a PASS near its bar: the weights, rotated per run
            rec["near_margin"] = near_margin(fractions)
            rec["policy_pt"] = _dump_near_margin(self.model, _run_dir(self.model), self.updates, rec["num_timesteps"])
            self._say(f"📐 [CompileCanary] update {self.updates}: PASS, but "
                      + ", ".join(f"{q} at {fractions[q]['fraction']:.3f} of its bar ({fractions[q]['value']:.4g} / "
                                  f"{fractions[q]['bar']:.4g})" for q in rec["near_margin"])
                      + f" (>= {NEAR_MARGIN_FRACTION:g}) — policy weights "
                      + (f"dumped to {rec['policy_pt']} (the newest {NEAR_MARGIN_KEEP} per run kept)"
                         if rec["policy_pt"] else "NOT dumped (no run directory, or the write failed)"))
        if self.memory:
            rec["memory"] = dict(self.memory)
        if watch:
            rec["watch"] = watch
        _append(_run_dir(self.model), VERDICTS_FILE, {k: v for k, v in rec.items() if k != "detail"}
                if verdict == "pass" else rec)
        return rec

    @property
    def rule_version(self) -> str:
        """The rule this canary judges by (``canary_rule_version``): the per-parameter null on, or stage 1's."""
        return (canary_row_bar.CANARY_RULE_VERSION if self.param_null_on
                else canary_row_bar.STAGE1_RULE_VERSION)

    def _bars_live(self) -> bool:
        return self._row_bars is not None and self._row_bars.live

    def _rule_record(self) -> Dict[str, Any]:
        """THE RULE every record names (`gen3_canary_row_bar_v1`): the per-row bar's constants, where this canary's bars
        came from, and the bf16 loss envelope's K."""
        from agents.model import compile_regions as cr
        rb = self._row_bars or canary_row_bar.fallback("no per-row null was computed (outside a scheduled canary)")
        return {**rb.record(), "version": self.rule_version, "envelope_loss_k": cr.BF16_ENVELOPE_K["loss"] if cr.BF16_ENVELOPE_K else None,
                "param_null": canary_param_null.rule_record() if self.param_null_on else "off"}

    def _shift_max_record(self) -> Dict[str, Any]:
        """THE JUDGEMENT INPUT every record carries (`--canary-shift-max`, T37): the shift the margin was built from, where
        it came from (``default`` | ``--canary-shift-max``) and the margin (`R1_BAR_K` x it) — so a record read alone
        says what bar the check judged by, whichever verdict it carries and whether or not it reached the margin rule."""
        from agents.model import compile_regions as cr
        return {"value": cr.shift_max_of(self.model), "source": cr.shift_max_source(self.model),
                "margin": cr.clip_margin_of(self.model)}

    @staticmethod
    def _new_watch() -> Any:
        from agents.model.compile_regions import GradWatch
        return GradWatch(WATCH_PREFIXES)

    def _say(self, line: str) -> None:
        print(line, flush=True)
        if self._emit is not None:
            with contextlib.suppress(Exception):
                self._emit(line[:500])

    def _fatal(self, why: str, first: BaseException, *, compared: bool = True,
               watch: Optional[Dict[str, Any]] = None,
               fractions: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        run_dir = _run_dir(self.model)
        saved = "not written (no run directory declared)"
        if run_dir:
            path = os.path.join(run_dir, FATAL_CHECKPOINT)
            try:
                (self._saver or self.model.save)(path)
                saved = path + ".zip (forensics only: these weights trained on the graph in question)"
            except Exception as exc:                          # the FATAL stands either way
                saved = f"FAILED to write ({type(exc).__name__}: {exc})"
        self._record("fatal", f"{why} — {first}", watch, fractions)
        verdict = ("NO comparison was made, so this says nothing about the compiled arithmetic — the "
                   "learner's compiled route is not the one the gate certified" if not compared else
                   "The canary's YARDSTICK (eager / the fp32 reference) refused, NOT the compiled arm: there is "
                   "nothing left to judge the compiled learner by on these weights" if _yardstick_refused(first) else
                   "The COMPILED learner no longer equals eager on the committed real-obs fixture at the "
                   "startup gate's bars")
        raise CompileCanaryError(
            f"{CANARY_TAG} after update {self.updates}: {why}. {verdict} — {first}\n"
            f"Checkpoint: {saved}.\nSAFE ROLLBACK POINT: {rollback_point(run_dir)}.\n"
            f"Training on a wrong function is GIGO: fatal by design, the launcher does NOT restart it "
            f"(FATAL_CONFIG). See designs/training/learner_lifecycle.md (the canary).") from first

    def run(self) -> Dict[str, float]:
        # the per-row null on these weights, BEFORE the check and outside its timer and memory card; it sets the compiled
        # arm's per-row bars (gen3_canary_row_bar_v1), or the fallback at the cap when it cannot bind
        self._row_null_raw = None
        if self.row_null_on:
            try:
                self._row_null_raw = canary_row_null.compute(self.model)
            except Exception as exc:                        # noqa: BLE001 — recorded; the bars fall back to the cap
                self._row_null_raw = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        self._row_bars = canary_row_bar.from_null(self._row_null_raw)
        out: Dict[str, float] = {}
        t0 = time.perf_counter()
        w0 = self._new_watch()
        self._w0 = w0
        try:
            rules = self._check(0, out, w0, self._row_bars)
        except ct.GateArmsNotIndependentError as first:
            # Not a disagreement: the two arms were not independent (the compiled route ran eager, or
            # the eager reference ran the compiled graph), so there is no verdict to CONFIRM. A dispatch
            # fault is FATAL at once — never "compiled DISAGREES … CONFIRMED" (F-XC-4 FINDING 6).
            self._fatal("the canary's two arms are NOT independent (a dispatch fault, not a "
                        "disagreement)", first, compared=False)
            raise                                            # unreachable: _fatal raises
        except ct.CompileTrainerError as first:
            # (1) warn + dump, then CONFIRM in the same update on the two DISJOINT halves of the rows,
            # compiled and eager both re-computed (no row is in both, so no single row confirms itself).
            _append(_run_dir(self.model), DISAGREEMENTS_FILE,
                    {"update": self.updates, "num_timesteps": int(getattr(self.model, "num_timesteps", 0) or 0),
                     "verdict": str(first)[:4000]})
            self._say(f"⚠️  [CompileCanary] update {self.updates}: {_what(first)} — "
                      f"confirming now (the two disjoint halves of the rows). {str(first)[:300]}")
            again = []
            watches: Dict[str, Any] = {"scheduled": w0.record()}
            for sl in ct.FIXTURE_HALVES:
                wc = self._new_watch()
                try:
                    self._check(sl, {}, wc, self._row_bars)
                except ct.CompileTrainerError as exc:
                    again.append((sl, exc))
                watches[f"confirm_slice{sl}"] = wc.record()
            if len(again) == len(ct.FIXTURE_HALVES):        # (2) confirmed on both disjoint halves
                self._fatal("the disagreement is CONFIRMED (it reproduced on BOTH disjoint halves of the "
                            "fixture rows in the same update)", first, watch=watches,
                            fractions=bar_fractions(w0))
            if self.prev_disagreed:                          # (4) two consecutive scheduled canaries
                self._fatal("the previous scheduled canary disagreed too — two CONSECUTIVE "
                            "disagreements, each unconfirmed", first, watch=watches,
                            fractions=bar_fractions(w0))
            # (3) not confirmed: count it, keep training, the next scheduled canary decides
            self.unconfirmed += 1
            self.prev_disagreed = True
            self.runs += 1
            halves = list(ct.FIXTURE_HALVES)
            self._record("unconfirmed", f"reproduced on slice(s) {[sl for sl, _ in again]} of {halves}; {first}",
                         watches, bar_fractions(w0))
            self._say(f"⚠️  [CompileCanary] update {self.updates}: the disagreement did NOT confirm "
                      f"(reproduced on slice(s) {[sl for sl, _ in again]} of the halves {halves}) — training "
                      f"continues; {self.unconfirmed} unconfirmed so far; a disagreement at the NEXT "
                      f"scheduled canary is FATAL.")
            self._note_yardstick(watches)
            out = {"compile/canary_ok": 0.0,
                   "compile/canary_unconfirmed_disagreements": float(self.unconfirmed),
                   "compile/canary_yardstick_events": float(self.yardstick_events),
                   "compile/canary_seconds": time.perf_counter() - t0}
            out.update(self._row_null_scalars())
            self.last = out
            return out
        self.prev_disagreed = False
        self.runs += 1
        watches = {"scheduled": w0.record()}
        self._record("pass", watch=watches, fractions=bar_fractions(w0))
        self._note_yardstick(watches)
        out["compile/canary_ok"] = 1.0
        out["compile/canary_unconfirmed_disagreements"] = float(self.unconfirmed)
        out["compile/canary_yardstick_events"] = float(self.yardstick_events)
        out["compile/canary_seconds"] = time.perf_counter() - t0
        out.update(self._row_null_scalars())
        if self.memory:
            out["compile/canary_peak_alloc_mib"] = float(self.memory["peak_alloc_mib"])
            out["compile/canary_reserved_mib"] = float(self.memory["reserved_mib"])
        self.last = out
        from agents.model.trunk_precision import trunk_precision_of
        memo = (f"; card: peak allocated {self.memory['peak_alloc_mib']:,.0f} MiB (training state "
                f"{self.memory['start_alloc_mib']:,.0f} + the check's transient {self.memory['transient_mib']:,.0f}), "
                f"reserved {self.memory['reserved_mib']:,.0f}" if self.memory else "")
        self._say(f"🐤 [CompileCanary] update {self.updates}: compiled == eager on the fixture "
                  f"({trunk_precision_of(self.model.policy)}, {out['compile/canary_seconds']:.1f} s{memo}) — "
                  + " | ".join(rules) + self._row_null_line())
        return out

    def _row_null_scalars(self) -> Dict[str, float]:
        """The null's TB scalars: its wall time (absent when it was off) and whether the per-row bars came from it (1)
        or fell back to the cap (0) — gen3_canary_row_bar_v1."""
        raw = self._row_null_raw
        out = {"compile/canary_row_null_seconds": float(raw["seconds"])} if raw and "seconds" in raw else {}
        out["compile/canary_row_bars_live"] = float(self._bars_live())
        return out

    def _row_null_line(self) -> str:
        """The PASS line's null fragment: the null's worst row and each arm's worst shift / null (the bars themselves
        are on the R1 rule line), or why the per-row bars fell back to the cap."""
        raw = self._row_null_raw
        if raw is None:
            return f" | row null: OFF, {self._row_bars.source if self._row_bars else 'no bars'}"
        rec = canary_row_null.record(raw, getattr(self._w0, "row_dlogp", None))
        if "null" not in rec:
            why = rec.get("skipped") or rec.get("error") or "no null"
            return f" | row null: {why} (per-row bars at the cap {canary_row_bar.ROW_BAR_CAP:g})"
        arms = ", ".join(f"{a} ratio max {d['ratio']['max']} (row {d['ratio']['argmax_row']})"
                         for a, d in rec.get("arms", {}).items() if "ratio" in d)
        return (f" | row null (K {rec['seeds_done']}, {rec['seconds']:.2f} s; "
                f"{self._row_bars.source if self._row_bars else 'no bars'}): max "
                f"{rec['null']['max']} (row {rec['null']['argmax_row']})" + (f"; {arms}" if arms else ""))

    def _note_yardstick(self, watches: Dict[str, Any]) -> None:
        """Count and LOG the yardstick-out-of-margin events of this canary's checks (gen3_r1_yardstick_margin_v1):
        recorded, never counted toward confirm / FATAL — the rung re-ran at M' and was judged there."""
        from agents.model.compile_regions import yardstick_line
        for name, w in watches.items():
            for ev in (w or {}).get("yardstick_out_of_margin", []):
                self.yardstick_events += 1
                self._say(f"📏 [CompileCanary] update {self.updates} ({name}): {yardstick_line(ev)}")

    # ------------------------------------------------------------------ the compiled region
    def _regions(self, model: Any, out: Dict[str, float], slice_: int = 0, watch: Optional[Any] = None,
                 row_bars: Optional[canary_row_bar.RowBars] = None) -> list:
        """K8's declared region: R1 (compiled micro-step vs eager: loss + every policy gradient) on the
        same real labelled batch the startup gate uses (``slice_`` 2 / 1 = the disjoint halves). ``watch`` is filled
        by the FIRST route's live rung (`compile_regions.r1_rungs`)."""
        from agents.model import compile_regions as cr
        policy = model.policy
        policy.set_training_mode(True)
        b = cr.r1_batch(model, self.batch_size, slice_)
        args = cr._r1_args(model, b)
        if watch is not None and b.fixture_rows is not None:      # worst rows recorded as FIXTURE rows too
            watch.fixture_rows = [int(x) for x in b.fixture_rows]
        names = [n for n, _ in ct.grad_parameters(model, policy.features_extractor)]
        if row_bars is not None and b.fixture_rows is None:
            row_bars = canary_row_bar.fallback("the batch names no fixture rows")
        bb = None if row_bars is None else row_bars.for_batch(b.fixture_rows)     # gen3_canary_row_bar_v1
        # the startup gate's rungs (`compile_regions.r1_rungs`): the live weights at the regime's bar,
        # an unmoved zero-init parameter at the FRESH bar there and on its own perturbed rung at the
        # TRAINED bar (`gen3_r1_unmoved_param_v1`)
        regime = cr.weights_regime(model)
        if not cr.learner_cudagraphs_on(model):
            eager, comp, rules = cr.r1_rungs(model, args, names, regime, watch=watch, row_bars=bb,
                                             param_null=self.param_null_on)
            out["compile/canary_grad_cosine"] = ct._cos(comp["grad"], eager["grad"])
            out["compile/canary_unmoved_params"] = float(len(cr.unmoved_parameters(model)))
            return rules
        # `--learner-cudagraphs on` (gen3_r1_graphed_route_v1): EVERY declared route — the GRAPHED one
        # first (it serves every plain micro-step) and its cosine is the tag's; then the compiled one (the
        # diagnostic micro-steps' route, the same graph without cudagraph trees). One more compiled + eager
        # pair per canary (~0.35 s at B = 2048, every 100 updates).
        rules = []
        for i, (route, fn) in enumerate(cr.r1_routes(model)):
            eager, comp, r = cr.r1_rungs(model, args, names, regime, fn=fn, route=route,
                                         watch=watch if i == 0 else None, row_bars=bb, param_null=self.param_null_on)
            rules.extend(r)
            if i == 0:
                out["compile/canary_grad_cosine"] = ct._cos(comp["grad"], eager["grad"])
        out["compile/canary_unmoved_params"] = float(len(cr.unmoved_parameters(model)))
        return rules
