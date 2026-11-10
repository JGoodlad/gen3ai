"""``--human-data`` AT THE TRAINER (``human_data_v1``): the one decision the trainer, ``main.checkargs`` and the
launcher's ``--dry-run`` share. Grammar and record: ``agents.info_levers.human_data``; the corpus behind a ``teams``
clause: ``utils.team_corpora.human``; the model behind an ``init`` / ``opp`` clause: ``utils.human_models``; design:
``designs/endstate/human_data_architecture.md`` §6.

:func:`decide` (pure reads: the source run's ``metadata.json``, the corpus's or the model's files) answers:

1. **The resume / fork refusal.** A ``--model`` launch whose SOURCE run carries a clause other than ``off`` (its
   ``metadata.json`` ``cli_args``, or its lineage, or an ancestor's recorded in that lineage) REFUSES unless
   ``--human-data`` is typed (the clause or ``off``): nothing inherits or drops a lever silently. The launcher's
   restarts re-launch the RESUME-role argv, which keeps the typed flag (it is not fresh-only), so a 6-h restart
   neither trips the refusal nor drops the lever.
2. **What the surface honours.** Training honours every head of the grammar (``teams``, ``init``, ``opp``).
3. **One lever per run.** A ``teams`` clause beside a ``--team-mix`` (typed, or restored by a same-run resume) is
   refused; an ``opp`` clause beside a ``--stable-opponents`` other than the one it derives is refused.
4. **The corpus.** ``teams:<rowset>@<share>`` loads and verifies the corpus against its committed stub.
5. **The model.** ``init:<model>@<dose>`` and ``opp:<model>@<share>`` load and verify the registered human-trained
   model against its committed stub (``utils.human_models.load``).

   * ``init`` is a FRESH run's lever: :func:`apply_init` copies the parameters the clone's training moved into the
     freshly built learner (dose 1; dose 0 copies nothing). On a ``--model`` launch the weights come from the
     checkpoint, so the clause may only be re-typed to KEEP the source's own (a restart, a resume, a fork that
     keeps it); typed on a source that does not carry it, it is refused.
   * ``opp`` DERIVES the stable-opponent flags: :func:`resolve` sets ``--stable-opponents <zip>:<id>`` and
     ``--stable-opponent-selfplay-share <share>`` on the namespace, so the frozen clone enters through the one
     cross-run opponent path (and is recorded and restored like any stable opponent). Share 0 installs nothing.

6. **One entry point per human-trained model.** A ``--model`` that IS a human-trained model is refused here, and
   ``--stable-opponents`` / ``--exploiter`` refuse one in ``fixed_opponent_pool.resolve_stable_opponents`` unless it
   is the zip this clause derived (:func:`allowed_stable_zip`).

Every refusal is ``FATAL_CONFIG`` at the trainer (:class:`HumanDataRefused`). ``off`` opens no corpus or model path.
:func:`lineage_keys` is what the immutable lineage block of a NEW run (fresh or fork) carries.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents.info_levers import human_data as HD
from main.exit_codes import FatalConfigError


class HumanDataRefused(FatalConfigError):
    """A ``--human-data`` launch the trainer will not run → ``FATAL_CONFIG`` (3): a restart would hit it identically."""


@dataclass
class Decision:
    record: Optional[Dict[str, Any]]            # None when refused
    refusal: Optional[str] = None
    source: Optional[Dict[str, Any]] = None     # what the --model source run carries (None: nothing / no --model)
    lines: List[str] = field(default_factory=list)
    stable: Optional[Tuple[str, float]] = None  # an opp clause's derived (--stable-opponents spec, self-play share)


def _metadata(model_path: Optional[str]) -> Dict[str, Any]:
    from agents.training.lineage import run_dir_of
    from utils.paths import resolve_archive_ref

    try:
        run_dir = run_dir_of(resolve_archive_ref(model_path)) if model_path else None
        with open(os.path.join(run_dir, "metadata.json")) as f:
            meta = json.load(f)
    except (OSError, TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _canonical(value: Any) -> Optional[str]:
    """A recorded clause (a record block or the typed text) → its canonical text, or None for off / absent / junk."""
    if isinstance(value, dict):
        return str(value["canonical"]) if HD.is_on(value) else None
    if isinstance(value, str):
        try:
            c = HD.parse(value)
        except HD.HumanDataError:
            return value          # unparseable here (a later grammar?): still a clause, never treated as off
        return None if c.is_off else c.canonical
    return None


def source_clause(model_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """The human-data clause the ``--model`` source run carries (``{"canonical", "where"}``), or None. Read from
    its ``metadata.json``: the last process's resolved record (``cli_args._human_data``) and typed text
    (``cli_args.human_data``), the lineage's own clause, and the first human-data ancestor the lineage names."""
    meta = _metadata(model_path)
    cli = meta.get("cli_args") if isinstance(meta.get("cli_args"), dict) else {}
    lin = meta.get("lineage") if isinstance(meta.get("lineage"), dict) else {}
    anc = lin.get("human_data_ancestor") if isinstance(lin.get("human_data_ancestor"), dict) else None
    for where, value in (("metadata.json cli_args._human_data", cli.get("_human_data")),
                         ("metadata.json cli_args.human_data", cli.get("human_data")),
                         ("metadata.json lineage.human_data", lin.get("human_data")),
                         ("metadata.json lineage.human_data_ancestor", (anc or {}).get("record"))):
        c = _canonical(value)
        if c is not None:
            return {"canonical": c, "where": where, "ancestor": anc}
    return None


def _model_is_human(model: Optional[str]) -> Optional[str]:
    """Why ``--model``'s target is a human-trained model (``utils.human_models.marker_of``), or None."""
    from utils import human_models as HM
    from utils.paths import resolve_archive_ref

    for cand in (model, resolve_archive_ref(model) if model else None):
        why = HM.marker_of(str(cand)) if cand else None
        if why is not None:
            return why
    return None


def _stable_of(args, clause: HD.Clause, hm: Dict[str, Any]) -> "Tuple[Optional[Tuple[str, float]], Optional[str]]":
    """An ``opp`` clause's derived ``(--stable-opponents spec, share)`` and the refusal (either may be None)."""
    from main.train.recipe_surface import value_source

    have = getattr(args, "stable_opponents", None)
    if clause.param == 0.0:
        if have:
            return None, (f"--human-data {clause.canonical} beside --stable-opponents {have}: one stable-opponent "
                          "lever per run (an opp clause owns the stable-opponent flags, also at the null dose)")
        return None, None
    zp = str(hm["zip_path"])
    if any(c in zp for c in "@:,"):
        return None, f"the model's path {zp!r} holds a character the --stable-opponents grammar reserves (@ : ,)"
    spec = f"{zp}:{hm['id']}"
    if have and have != spec:
        return None, (f"--human-data {clause.canonical} beside --stable-opponents {have}: one stable-opponent lever "
                      f"per run (the clause derives --stable-opponents {spec}; type neither stable-opponent flag)")
    share = getattr(args, "stable_opponent_selfplay_share", None)
    if value_source(args, "stable_opponent_selfplay_share")[0] != "default" and share is not None \
            and abs(float(share) - float(clause.param)) > 1e-9:
        return None, (f"--human-data {clause.canonical} beside --stable-opponent-selfplay-share {share}: the "
                      "clause's share IS that flag; type the clause alone")
    return (spec, float(clause.param)), None


def decide(args, *, load_corpus: bool = True) -> Decision:
    """The launch's human-data decision over a resolved (or checkargs-effective) namespace. Pure but for reads."""
    typed = getattr(args, "human_data", None)
    model = getattr(args, "model", None)
    why = _model_is_human(model)
    if why is not None:
        return Decision(None, (f"--model {model}: {why}; a human-trained model is never resumed or forked: it enters "
                               "a run only through --human-data init:<model>@<dose> or opp:<model>@<share>"))
    src = source_clause(model) if model else None
    if typed is None and src is not None:
        return Decision(None, (f"--model {model}: its run carries the human-data clause {src['canonical']} "
                               f"({src['where']}); a resume or fork must TYPE --human-data (that clause to keep it, or "
                               "off to drop it), so no lever is inherited or dropped silently"), src)
    try:
        clause = HD.honoured_or_refuse(HD.parse(typed), "train")
    except HD.HumanDataError as e:
        return Decision(None, str(e), src)
    if clause.lever == "teams" and getattr(args, "team_mix", None):
        return Decision(None, (f"--human-data {clause.canonical} beside --team-mix {args.team_mix}: one team lever "
                               "per run (the clause IS a team mix, sides both)"), src)
    corpus = None
    if clause.lever == "teams" and load_corpus:
        from utils.team_corpora import CorpusError
        from utils.team_corpora import human as H

        try:
            corpus = H.identity(clause.rowset, H.load(clause.rowset))
        except CorpusError as e:
            return Decision(None, f"--human-data {clause.canonical}: {e}", src)
    hm, stable, notes = None, None, []
    if clause.lever == "init" and model and (src is None or src["canonical"] != clause.canonical):
        return Decision(None, (f"--human-data {clause.canonical} on --model {model}: init is a FRESH run's lever (the "
                               "weights of a --model launch come from its checkpoint). It may only be re-typed to "
                               "keep the source run's own clause, and the source carries "
                               f"{src['canonical'] if src else 'none'}"), src)
    if clause.lever in HD.MODEL_LEVERS and load_corpus:
        from utils import human_models as HM

        try:
            hm = HM.load(clause.model, debug=bool(getattr(args, "debug", False)))
        except HM.HumanModelError as e:
            return Decision(None, f"--human-data {clause.canonical}: {e}", src)
        if clause.lever == "opp":
            stable, refusal = _stable_of(args, clause, hm)
            if refusal is not None:
                return Decision(None, refusal, src)
            if stable is not None:
                notes.append(f"   opp: derives --stable-opponents {stable[0]} --stable-opponent-selfplay-share "
                             f"{HD.format_param(stable[1])} (frozen; eval opponent ext_{hm['id']}; the training mix "
                             "needs --self-play)")
        elif clause.param == 1.0:
            notes.append("   init: the weights come from the --model checkpoint; nothing is copied" if model else
                         f"   init: the fresh learner takes {hm['n_trained_keys']} trained tensors from {hm['id']}; "
                         "every other parameter keeps this run's own fresh init")
    rec = HD.record(clause, corpus, hm)
    lines = [HD.stamp_line(rec)] + notes
    if src is not None and src["canonical"] != clause.canonical:
        lines.append(f"   ⚠️ the --model source carried {src['canonical']} ({src['where']}); this launch TYPED "
                     f"{clause.canonical}: the lever changes here")
    return Decision(rec, None, src, lines, stable)


def resolve(args, emit=print) -> Dict[str, Any]:
    """The trainer's call (before the matchup is declared): refuse, or stamp the record on ``args._human_data`` (so
    ``metadata.json:cli_args`` carries it) and emit the startup stamp."""
    d = decide(args)
    if d.refusal is not None:
        raise HumanDataRefused(f"[HUMAN-DATA] FATAL: {d.refusal}")
    args._human_data = d.record
    if d.stable is not None:
        args.stable_opponents, args.stable_opponent_selfplay_share = d.stable
    for ln in d.lines:
        emit(ln)
    return d.record


def allowed_stable_zip(args) -> Optional[str]:
    """The ONE human-trained checkpoint the stable-opponent path may load in this run: the zip an ``opp`` clause at a
    non-zero share resolved to (:func:`resolve` stamped it), else None."""
    rec = getattr(args, "_human_data", None)
    if not HD.is_on(rec) or rec.get("lever") != "opp" or HD.is_null_dose(rec):
        return None
    return (rec.get("human_model") or {}).get("zip_path")


def apply_init(model, args, version=None, emit=print) -> Optional[Dict[str, Any]]:
    """``init:<model>@1`` on a FRESH learner (``model_build``, right after the fresh build and its init record):
    copy the tensors the clone's training moved into ``model.policy``, in place; every other parameter keeps this
    run's own seeded init, and the optimizer starts fresh. Refuses (``FATAL_CONFIG``) a clone of another
    architecture family, a tensor the learner lacks or holds at another shape, and bytes that are not the
    manifest's. No-op (None) for any other clause, the null dose, and a ``--model`` launch."""
    rec = getattr(args, "_human_data", None)
    if not HD.is_on(rec) or rec.get("lever") != "init" or HD.is_null_dose(rec) or getattr(args, "model", None):
        return None
    import torch as th

    from utils import human_models as HM

    hm = rec["human_model"]
    keys = HM.trained_keys(hm["id"])
    src = HM.read_policy_state(hm["zip_path"])
    if HM.state_digest(src, keys) != hm["trained_keys_sha256"]:
        raise HumanDataRefused(f"[HUMAN-DATA] FATAL: init {hm['id']}: the checkpoint's trained tensors are not the "
                               "ones its manifest recorded")
    if version is not None:
        from agents.model.model_version import ModelVersion, ModelVersionError

        try:
            version.check_opponent_compatible(ModelVersion.from_json_file(
                os.path.join(os.path.dirname(hm["zip_path"]), HM.CONFIG_NAME)))
        except ModelVersionError as e:
            raise HumanDataRefused(f"[HUMAN-DATA] FATAL: init {hm['id']}: {e}") from e
    dst = model.policy.state_dict()
    bad = [k for k in keys if k not in dst or tuple(dst[k].shape) != tuple(src[k].shape)]
    if bad:
        raise HumanDataRefused(f"[HUMAN-DATA] FATAL: init {hm['id']}: {len(bad)} of its {len(keys)} trained tensors "
                               f"are absent from this run's network or have another shape (first: {bad[0]}); a clone "
                               "initialises only a run of its own architecture and width")
    with th.no_grad():
        for k in keys:
            dst[k].copy_(src[k].to(device=dst[k].device, dtype=dst[k].dtype))
    if HM.state_digest(model.policy.state_dict(), keys) != hm["trained_keys_sha256"]:
        raise HumanDataRefused(f"[HUMAN-DATA] FATAL: init {hm['id']}: the copy did not land bit for bit")
    out = {"model": hm["id"], "tensors": len(keys), "parameters": int(sum(src[k].numel() for k in keys)),
           "of_tensors": len(dst)}
    rec["init_applied"] = out
    emit(f"[HUMAN-DATA] init: copied {out['tensors']} of {out['of_tensors']} tensors ({out['parameters']:,} values) "
         f"from {hm['id']}; every other parameter keeps this run's fresh init; the optimizer starts fresh")
    return out


def lineage_keys(args, model: Optional[str]) -> Dict[str, Any]:
    """The sibling keys a NEW run's immutable lineage block carries: ``human_data`` (this run's clause, when not
    ``off``) and ``human_data_ancestor`` (the nearest ancestor that trained under a clause: the fork parent itself,
    else the one the parent's lineage names), so the refusal follows a lever down every fork. ``model`` is the fork
    parent AS PARSED (``run_io._run_lineage``'s ``model_path``; None for a fresh run)."""
    out: Dict[str, Any] = {}
    rec = getattr(args, "_human_data", None)
    if HD.is_on(rec):
        out["human_data"] = dict(rec)
    if model:
        meta = _metadata(model)
        cli = meta.get("cli_args") if isinstance(meta.get("cli_args"), dict) else {}
        lin = meta.get("lineage") if isinstance(meta.get("lineage"), dict) else {}
        own = cli.get("_human_data") if HD.is_on(cli.get("_human_data")) else lin.get("human_data")
        if HD.is_on(own):
            from agents.training.lineage import run_dir_of
            out["human_data_ancestor"] = {"run": os.path.basename(run_dir_of(model) or ""), "record": dict(own)}
        elif isinstance(lin.get("human_data_ancestor"), dict):
            out["human_data_ancestor"] = dict(lin["human_data_ancestor"])
    return out


def report_lines(args) -> "tuple[List[str], Optional[str]]":
    """``main.checkargs`` / ``--dry-run``: the lines to print and the refusal (or None)."""
    d = decide(args)
    if d.refusal is not None:
        return [f"human data                     : ✗ REFUSED — {d.refusal}"], d.refusal
    rec = d.record or {}
    head = ("off (no human data)" if not HD.is_on(rec)
            else f"{rec['canonical']}  hash {rec['hash']}  (a training lever; eval / play stay off)")
    return [f"human data                     : {head}"] + [f"      {ln.strip()}" for ln in d.lines], None
