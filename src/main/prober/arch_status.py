"""Is a run / a step / a checkpoint at the architecture THIS code builds? — answered from RECORDS, never a load.

The prober's model views (`/game`'s model panels, `/analyze`, `battle_readout`, the counterfactual probes) re-run
the policy under CURRENT code, so they work on a checkpoint at HEAD's architecture and on nothing else (owner
decision). This module is the ONE place that says which, in one typed, plain-language verdict, for three callers:

* the run PICKER (`web/runs.py`) classifies every run cheaply from `model_config.json` / the newest eval manifest —
  no checkpoint is opened, so listing 300+ runs costs a few JSON reads, cached by mtime;
* the per-STEP marks (`ProbeSession.model_status`, the step dropdowns, the run summary);
* `ProbeModel.load` — the model seam — which pre-checks a checkpoint against the same verdict BEFORE the load and
  maps every load FAILURE onto the same kinds, so no incompatibility class reaches a surface as a raw error.

Pure stdlib plus the model-version constants: importing this loads no torch.

THE KINDS (`KINDS`; one typed diagnosis each — a surface branches on the kind, never on message text):

  arch_signature   the network family differs from the code's, or the run predates `MIGRATION_FLOOR`
                   (weights have no home)
  obs_dim          trained on a different observation width than the code's encoder writes
  obs_semantics    weights WOULD load, but the observation changed meaning since `OBS_SEMANTICS_VERSION`
                   (a re-scaled / re-defined cell) — the silent class: the shapes all fit
  state_dict       the strict load found missing / unexpected / mis-shaped parameters
  config_value     the code now rejects a value (or no longer has a flag) the checkpoint recorded
  newer_than_code  recorded by NEWER code than the one running (a stale server)
  no_checkpoint    nothing loadable (a trimmed "skeleton" run, or the file is gone)
  unreadable       the file is there but is not a readable checkpoint
  load_failed      any other failure of the load itself (never leaked raw)

Recorded versions come from, in order: the eval manifest beside the snapshot (`eval_manifest.json` — the
per-cycle record of the model that played), else the nearest `model_config.json` (the checkpoint's own dir, then up
to the run root). A checkpoint with NO record is `unrecorded`: the model views may be TRIED (and their failure is
classified), but it is never counted as current.
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Optional

from agents.model.model_version.constants import (
    ARCH_SIGNATURE,
    MODEL_CONFIG_VERSION,
    OBS_SEMANTICS_REASON,
    OBS_SEMANTICS_VERSION,
)
from agents.model.model_version.migrations import MIGRATION_FLOOR

# -- statuses -------------------------------------------------------------------------------------

CURRENT = "current"                 # recorded at HEAD's architecture AND observation semantics: model views run
INCOMPATIBLE = "incompatible"       # model views cannot run (the kind says why)
UNRECORDED = "unrecorded"           # no version record: the views may be tried, never counted as current

# -- kinds ----------------------------------------------------------------------------------------

KIND_ARCH_SIGNATURE = "arch_signature"
KIND_OBS_DIM = "obs_dim"
KIND_OBS_SEMANTICS = "obs_semantics"
KIND_STATE_DICT = "state_dict"
KIND_CONFIG_VALUE = "config_value"
KIND_NEWER = "newer_than_code"
KIND_NO_CHECKPOINT = "no_checkpoint"
KIND_UNREADABLE = "unreadable"
KIND_LOAD_FAILED = "load_failed"
KINDS = (KIND_ARCH_SIGNATURE, KIND_OBS_DIM, KIND_OBS_SEMANTICS, KIND_STATE_DICT, KIND_CONFIG_VALUE,
         KIND_NEWER, KIND_NO_CHECKPOINT, KIND_UNREADABLE, KIND_LOAD_FAILED)

# -- the picker's tiers ----------------------------------------------------------------------------

TIER_CURRENT = "current"            # has eval traces AND model views can run: "Current architecture"
TIER_OLDER = "older"                # has eval traces, model views cannot (or cannot be shown to) run
TIER_NO_TRACES = "no_traces"        # nothing to inspect: hidden behind "show all"
TIERS = (TIER_CURRENT, TIER_OLDER, TIER_NO_TRACES)


def current_obs_dim() -> int:
    """The observation width the CURRENT encoder writes — from the layout constants, so no encoder (and no
    vocabulary load) is built to answer it. `Gen3ObservationEncoder.dimension` is exactly this sum."""
    import agents.observation.constants as C
    return int(C.OFFSET_OBS_FACTS + C.OBS_FACTS_DIM)


@dataclass(frozen=True)
class ArchVerdict:
    status: str                         # CURRENT | INCOMPATIBLE | UNRECORDED
    kind: Optional[str]                 # one of KINDS when INCOMPATIBLE, else None
    plain: str                          # ONE plain-language sentence a page shows verbatim
    config_version: Optional[int] = None
    arch_signature: Optional[str] = None
    obs_dim: Optional[int] = None
    source: Optional[str] = None        # which record said so: "eval_manifest" | "model_config" | "checkpoint"

    @property
    def current(self) -> bool:
        return self.status == CURRENT

    @property
    def model_views(self) -> bool:
        """May the model views be TRIED (only an INCOMPATIBLE verdict says they cannot run)."""
        return self.status != INCOMPATIBLE

    def as_dict(self) -> dict:
        return {"status": self.status, "kind": self.kind, "plain": self.plain,
                "model_views": self.model_views, "config_version": self.config_version,
                "arch_signature": self.arch_signature, "obs_dim": self.obs_dim, "source": self.source,
                "current_config_version": MODEL_CONFIG_VERSION, "current_arch_signature": ARCH_SIGNATURE}


# -- the plain-language sentences ------------------------------------------------------------------

def _identity(config_version: Optional[int], arch_signature: Optional[str]) -> str:
    parts = []
    if config_version is not None:
        parts.append(f"config v{config_version}")
    if arch_signature:
        parts.append(f"signature {arch_signature}")
    return ", ".join(parts)


def plain_sentence(kind: str, *, config_version: Optional[int] = None, arch_signature: Optional[str] = None,
                   obs_dim: Optional[int] = None, cause: Optional[str] = None) -> str:
    """The one sentence for a kind. For every drift kind it opens with the SAME clause —
    "This run's architecture (config vN, signature S) is older than the code (config vM, signature S'); model
    views need a current-architecture checkpoint." — and adds what is specific to the kind."""
    saved = _identity(config_version, arch_signature)
    code = _identity(MODEL_CONFIG_VERSION, ARCH_SIGNATURE)
    if kind == KIND_NEWER:
        return (f"This run was recorded by NEWER code ({saved or 'a later config'}) than the code running here "
                f"({code}); restart the prober from a newer checkout to open its model views.")
    if kind == KIND_NO_CHECKPOINT:
        return ("This run has no loadable checkpoint (only its trace files were kept; the weights are gone), so "
                "there is nothing to run the model on. The turn story and every model-free view still work.")
    if kind == KIND_UNREADABLE:
        return ("This run's checkpoint file is not a readable checkpoint (corrupt or truncated), so the model "
                "views cannot load it. The turn story and every model-free view still work.")
    older = (f"This run's architecture ({saved}) is older than the code ({code}); "
             if saved else f"This run's architecture is older than the code ({code}); ")
    tail = "model views need a current-architecture checkpoint."
    if kind == KIND_ARCH_SIGNATURE:
        return older + tail + " Its network was restructured since, so its weights have no home in the current one."
    if kind == KIND_OBS_DIM:
        trained = f"{obs_dim} inputs" if obs_dim else "a different number of inputs"
        return (older + tail + f" It was trained on {trained}; the code's observation is {current_obs_dim()} wide.")
    if kind == KIND_OBS_SEMANTICS:
        return (older + tail + " Its weights would load, but the observation changed meaning at config "
                f"v{OBS_SEMANTICS_VERSION} ({OBS_SEMANTICS_REASON}), so the model would read inputs it never "
                "trained on.")
    if kind == KIND_STATE_DICT:
        return (older + tail + " Its saved weights do not fit the network the code builds for it "
                "(missing, extra or differently-shaped parameters).")
    if kind == KIND_CONFIG_VALUE:
        return (older + tail + " The code now rejects a setting it recorded (or no longer has a flag it names)"
                + (f" — {cause}." if cause else "."))
    return (older + tail + " The checkpoint could not be loaded under the current code"
            + (f" ({cause})." if cause else "."))


def _int(v) -> Optional[int]:
    try:
        return int(v) if v is not None and not isinstance(v, bool) else None
    except (TypeError, ValueError):
        return None


def verdict_from_record(config_version, arch_signature, obs_dim=None, *,
                        source: Optional[str] = None) -> ArchVerdict:
    """The verdict a RECORDED (config_version, arch_signature, observation width) earns against this code.

    Pure and ordered, most fundamental first, so a record that is wrong in several ways reports the deepest one:
    the network family (or a config below `MIGRATION_FLOOR`, which is the same fact in numbers) → recorded by
    newer code → the observation width → the observation's MEANING (`OBS_SEMANTICS_VERSION`). A record carrying
    nothing at all is `unrecorded`; one that carries a matching signature but no version cannot vouch for the
    observation's meaning and is `unrecorded` too."""
    cv = _int(config_version)
    sig = str(arch_signature) if arch_signature else None
    od = _int(obs_dim)
    if cv is None and sig is None and od is None:
        return ArchVerdict(UNRECORDED, None, "This run records no architecture version, so whether the model "
                           "views can load its checkpoint is unknown; they will be tried.", source=source)

    def bad(kind: str) -> ArchVerdict:
        return ArchVerdict(INCOMPATIBLE, kind, plain_sentence(kind, config_version=cv, arch_signature=sig,
                                                             obs_dim=od), cv, sig, od, source)

    if (sig is not None and sig != ARCH_SIGNATURE) or (cv is not None and cv < MIGRATION_FLOOR):
        return bad(KIND_ARCH_SIGNATURE)
    if cv is not None and cv > MODEL_CONFIG_VERSION:
        return bad(KIND_NEWER)
    if od is not None and od != current_obs_dim():
        return bad(KIND_OBS_DIM)
    if cv is not None and cv < OBS_SEMANTICS_VERSION:
        return bad(KIND_OBS_SEMANTICS)
    if cv is None:
        return ArchVerdict(UNRECORDED, None, "This run records its network family but not its config version, "
                           "so whether the observation still means what it trained on is unknown; the model "
                           "views will be tried.", None, sig, od, source)
    return ArchVerdict(CURRENT, None, "This run is at the current architecture: the model views can run.",
                       cv, sig, od, source)


_RANK = {CURRENT: 0, UNRECORDED: 1, INCOMPATIBLE: 2}


def worst(a: ArchVerdict, b: ArchVerdict) -> ArchVerdict:
    """The more damning of two verdicts (incompatible > unrecorded > current); `a` wins a tie. The model views
    re-run a CHECKPOINT on a trace's STORED OBSERVATIONS, and either side can be the one that is not HEAD's — the
    checkpoint the ladder picked (a `nearest` / `recent` rung is not the model that played the step) or the
    trace itself (its observations were encoded under an older meaning)."""
    if _RANK[b.status] != _RANK[a.status]:
        return b if _RANK[b.status] > _RANK[a.status] else a
    # Two incompatible verdicts: a RECORDED reason (a different family, width or meaning) explains more than the
    # state of the file ("unreadable" / "no checkpoint"), which is often just the symptom.
    if a.status == INCOMPATIBLE and a.kind in (KIND_NO_CHECKPOINT, KIND_UNREADABLE) \
            and b.kind not in (KIND_NO_CHECKPOINT, KIND_UNREADABLE):
        return b
    return a


def no_checkpoint_verdict(detail: Optional[str] = None) -> ArchVerdict:
    return ArchVerdict(INCOMPATIBLE, KIND_NO_CHECKPOINT, plain_sentence(KIND_NO_CHECKPOINT), source=detail)


# -- reading the records ----------------------------------------------------------------------------

_STEP_DIR = re.compile(r"^step_(\d+)$")
_MANIFEST = "eval_manifest.json"


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def record_of(d: Optional[dict]) -> "tuple[Optional[int], Optional[str], Optional[int]]":
    """`(config_version, arch_signature, obs_dim)` out of a manifest / model_config dict (`total_dim` is the
    model_config's name for the observation width)."""
    if not d:
        return None, None, None
    return _int(d.get("config_version")), (d.get("arch_signature") or None), _int(d.get("total_dim"))


def step_dirs(run_dir: str) -> "list[tuple[int, str]]":
    """`(step, dir)` for every `eval_traces/step_<N>/` of a run, ascending. Names only: no file is opened."""
    base = os.path.join(run_dir, "eval_traces")
    out = []
    try:
        for entry in os.scandir(base):
            m = _STEP_DIR.match(entry.name)
            if m and entry.is_dir():
                out.append((int(m.group(1)), entry.path))
    except OSError:
        return []
    return sorted(out)


def run_has_traces(run_dir: str) -> bool:
    """At least one `eval_traces/step_<N>/` — an empty or absent `eval_traces/` is a run with nothing to inspect."""
    base = os.path.join(run_dir, "eval_traces")
    try:
        return any(_STEP_DIR.match(e.name) and e.is_dir() for e in os.scandir(base))
    except OSError:
        return False


def verdict_for_step(run_dir: str, step: Optional[int] = None) -> ArchVerdict:
    """The verdict for one eval step of a run (the newest when `step` is None): the step's own manifest first
    (what the model that played RECORDED), the run's `model_config.json` for anything the manifest lacks. Reads
    two small JSON files; never a checkpoint."""
    cfg = _read_json(os.path.join(run_dir, "model_config.json"))
    man = None
    if step is None:
        dirs = step_dirs(run_dir)
        if dirs:
            step = dirs[-1][0]
    if step is not None:
        man = _read_json(os.path.join(run_dir, "eval_traces", f"step_{step}", _MANIFEST))
    mcv, msig, _ = record_of(man)
    ccv, csig, cdim = record_of(cfg)
    cv = mcv if mcv is not None else ccv
    sig = msig or csig
    src = ("eval_manifest" if (mcv is not None or msig) else "model_config") if (cv is not None or sig) else None
    return verdict_from_record(cv, sig, cdim, source=src)


# The picker's cache: run dir -> (signature of what the verdict was read from, verdict). The signature is the
# mtime of `model_config.json` and of the `eval_traces/` directory (a new step directory moves it), so a run that
# gains a cycle or is re-saved is re-read and an idle one costs two `stat`s. Bounded: the archive is ~350 runs.
_RUN_CACHE: "dict[str, tuple[tuple, ArchVerdict]]" = {}
_RUN_CACHE_CAP = 4096


def _mtime(path: str) -> Optional[int]:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def run_verdict(run_dir: str) -> ArchVerdict:
    """`verdict_for_step(run_dir)` for the run's newest step, cached by mtime (see `_RUN_CACHE`)."""
    sig = (_mtime(os.path.join(run_dir, "model_config.json")), _mtime(os.path.join(run_dir, "eval_traces")))
    hit = _RUN_CACHE.get(run_dir)
    if hit is not None and hit[0] == sig:
        return hit[1]
    v = verdict_for_step(run_dir)
    if len(_RUN_CACHE) >= _RUN_CACHE_CAP:
        _RUN_CACHE.clear()
    _RUN_CACHE[run_dir] = (sig, v)
    return v


def tier_of(has_traces: bool, verdict: ArchVerdict) -> str:
    """The picker tier: no traces = nothing to inspect; traces + a CURRENT verdict = full model views; every other
    run with traces (older, unrecorded, newer-than-code) = turn story only."""
    if not has_traces:
        return TIER_NO_TRACES
    return TIER_CURRENT if verdict.current else TIER_OLDER


def checkpoint_record(ckpt_path: str) -> "tuple[Optional[int], Optional[str], Optional[int], Optional[str]]":
    """`(config_version, arch_signature, obs_dim, source)` recorded for ONE checkpoint file: the
    `eval_manifest.json` in its own directory (an eval `snapshot.zip` sits beside the cycle's manifest), else the
    nearest `model_config.json` going up (its own directory, then to the run root — three levels, because an
    eval snapshot is a grandchild of the run root)."""
    d = os.path.dirname(os.path.abspath(ckpt_path))
    man = _read_json(os.path.join(d, _MANIFEST))
    cv, sig, dim = record_of(man)
    if cv is not None or sig:
        return cv, sig, dim, "eval_manifest"
    for _ in range(3):
        cfg = _read_json(os.path.join(d, "model_config.json"))
        if cfg is not None:
            cv, sig, dim = record_of(cfg)
            return cv, sig, dim, "model_config"
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None, None, None, None


def checkpoint_verdict(ckpt_path: str, *, peek_obs_dim: Optional[int] = None) -> ArchVerdict:
    """The verdict for a checkpoint FILE, before any load: its recorded identity, with the observation width the
    zip itself declares (`peek_checkpoint`, ~5 ms) taking precedence over the sidecar's. A path that does not exist
    is `no_checkpoint`."""
    if not os.path.exists(ckpt_path):
        return no_checkpoint_verdict(os.path.basename(ckpt_path))
    if not zipfile.is_zipfile(ckpt_path):
        return ArchVerdict(INCOMPATIBLE, KIND_UNREADABLE, plain_sentence(KIND_UNREADABLE),
                           source=os.path.basename(ckpt_path))
    cv, sig, dim, src = checkpoint_record(ckpt_path)
    return verdict_from_record(cv, sig, peek_obs_dim if peek_obs_dim is not None else dim, source=src)
