"""CLI for the X5 belief readers (U7). See ``main/belief_roles/__init__.py``.

  python -m main.belief_roles roles                                   # the Smogon role set + pairs
  python -m main.belief_roles read --out <dir> \\
        --ckpt <run>/checkpoints/<ckpt>.zip=<label> [...] \\
        [--reference <label>=<blob>.erow.npz[,<blob>.erow.npz...]]    # CPU forwards on the Lane S bank
  python -m main.belief_roles infer --treat <X5 read>.json ... --control <blob read>.json ... \\
        --boundary 2.683 [--metric intent_logloss] [--margin 0]      # §7.4's across-seed t

``--out`` under ``models/`` is REFUSED (read-only). A bare run directory is REFUSED as a checkpoint.

``read`` reads X5 checkpoints (the hypothesis tokens, the only belief representation since the X5
version break, config v144). A PRE-BREAK checkpoint — a blob one, or a pre-break fixed_mass one — is
REFUSED with the loader's typed reason (run this reader PINNED to the checkpoint's own commit, where the
blob read arm still exists). A run's conditional metric (the adoption gate) is scored on EVERY blob run
of the look through ``--reference``: the BANKED ``<label>.erow.npz`` named sets the X5 A/B's blob reads
wrote (its value is the MEAN of the conditional log loss over those sets). Without any, it is None.
``infer`` compares finished read JSONs, the X5 A/B's blob reads among them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

READ_SCHEMA = "gen3_belief_purpose_read_v2"


def default_bank() -> Path:
    from utils.paths import repo_path

    return repo_path("designs", "research_state", "measurements", "m5_laneS", "bank_v1")


def _commit() -> str:
    from utils.git import get_git_hash

    try:
        return get_git_hash()
    except Exception:                                    # noqa: BLE001 - a read still runs
        return "unknown"


def refuse_under_models(path: Path) -> None:
    """``models/`` is read-only (standing rule 6); this reader never writes there."""
    from utils.paths import main_models_dir

    md = main_models_dir()
    if md is not None:
        rp, mr = Path(path).resolve(), md.resolve()
        if rp == mr or mr in rp.parents:
            raise SystemExit(f"[belief_roles] REFUSED: {path} is under models/ (read-only)")


def read_one(br, roles, zip_path: Path, label: str, threads: int, commit: str,
             reencode_s: float, references=()) -> dict:
    """One checkpoint's read. ``references``: banked blob runs' ``ERow``s (the run is scored on each,
    its value their mean). A pre-break checkpoint raises ``forward.ReadRefused``."""
    from main.belief_roles.forward import arm_of, file_sha256, load_strict, read_columns
    from main.belief_roles.metrics import read_all
    from main.policy_spectrum.reader import inference_globals

    references = list(references)
    for reference in references:
        if reference.meta.get("bank_sha256") != br.bank.manifest["content_sha256"]:
            raise SystemExit(f"[belief_roles] REFUSED: the reference E_row {reference.meta.get('label')} was "
                             f"read on bank {reference.meta.get('bank_sha256')}, this read is on "
                             f"{br.bank.manifest['content_sha256']}")
    ref_shas = [r.meta.get("checkpoint_sha256") for r in references]
    if len(set(ref_shas)) != len(ref_shas):
        raise SystemExit("[belief_roles] REFUSED: a blob checkpoint is named twice as a reference")
    t0 = time.time()
    with inference_globals(threads):
        model = load_strict(zip_path)
    t_load = time.time() - t0
    arm = arm_of(model)
    t1 = time.time()
    cols = read_columns(model, br, roles, threads=threads, references=references)
    t_fwd = time.time() - t1
    del model
    sha = file_sha256(Path(zip_path))
    bank_sha = br.bank.manifest["content_sha256"]
    if references:
        eref = {"mode": "all_blob_mean", "rule": "the mean over every blob run of the look (§7.7(b))",
                "references": [{"reference_label": r.meta.get("label"),
                                "checkpoint_sha256": r.meta.get("checkpoint_sha256"),
                                "erow_sha256": r.content_sha256()} for r in references]}
    else:
        eref = {"mode": "none"}
    t2 = time.time()
    body = read_all(br, cols, roles)
    t_met = time.time() - t2
    return {
        "schema": READ_SCHEMA, "label": label, "arm": arm,
        "checkpoint": {"path": str(zip_path), "sha256": sha},
        "bank_sha256": bank_sha, "eset_reference": eref,
        "bank": {"decisions": br.n, "battles": len(br.bank.battles),
                 "on_pool_rows": int(br.on_pool.sum()), "labeled_rows": int((br.event >= 0).sum())},
        "reencode": {k: v for k, v in br.gate.items() if k != "encoder"},
        "encoder_sha256": br.gate["encoder"]["core_events_sha256"],
        "role_set_sha256": roles.sha256(), "role_set": roles.to_json(),
        "reader_commit": commit, "threads": threads,
        "timing_s": {"reencode_shared": round(reencode_s, 2), "load": round(t_load, 2),
                     "forward": round(t_fwd, 2), "metrics": round(t_met, 2)},
        **body,
    }


def cmd_roles(a) -> int:
    from main.belief_roles.roles import derive

    print(json.dumps(derive().to_json(), indent=1))
    return 0


def cmd_read(a) -> int:
    from main.belief_roles.bank_rows import load_bank_rows
    from main.belief_roles.roles import derive

    out = Path(a.out)
    refuse_under_models(out)
    specs = []
    for spec in a.ckpt:
        path, _, label = spec.partition("=")
        if not label:
            sys.exit(f"[belief_roles] --ckpt {spec!r}: name it as <path.zip>=<label>")
        p = Path(path)
        if p.is_dir() or p.suffix != ".zip":
            sys.exit(f"[belief_roles] {p}: name the checkpoint .zip — a bare run directory resolves "
                     "to the run's LAST snapshot and moves")
        specs.append((p, label))
    # a PRE-BREAK checkpoint (blob, or a pre-break fixed_mass) is refused here, before the bank's re-encode
    from main.belief_roles.forward import ReadRefused, refuse_pre_break
    for p, _ in specs:
        try:
            refuse_pre_break(p)
        except ReadRefused as e:
            sys.exit(f"[belief_roles] REFUSED: {e}")
    labels = [lb for _, lb in specs]
    if len(set(labels)) != len(labels):
        sys.exit("[belief_roles] two --ckpt share a label")
    refs: dict = {}
    for spec in a.reference or []:
        lb, _, ref = spec.partition("=")
        names = [x for x in ref.split(",") if x]
        if lb not in labels or not names:
            sys.exit(f"[belief_roles] --reference {spec!r}: name it <label>=<blob>.erow.npz[,...], <label> "
                     "a --ckpt label of this invocation")
        refs.setdefault(lb, []).extend(names)
    for lb, names in refs.items():
        if len(set(names)) != len(names):
            sys.exit(f"[belief_roles] --reference {lb}: a blob set is named twice")
        for r in names:
            if r in labels:
                sys.exit(f"[belief_roles] --reference {lb}={r}: a reference is a BANKED blob named set "
                         "(<blob>.erow.npz) — a checkpoint this commit reads is an X5 one, never a blob")
    t = time.time()
    br = load_bank_rows(Path(a.bank), workers=a.workers)
    t_re = time.time() - t
    print(f"[belief_roles] re-encoded {br.n} decisions in {t_re:.1f} s; obs as recorded: "
          f"{br.gate['obs_as_recorded']} ({br.gate['byte_equal']}/{br.gate['recorded_rows_checked']}); "
          f"labelled {int((br.event >= 0).sum())}, on-pool {int(br.on_pool.sum())}", flush=True)
    roles = derive()
    commit = _commit()
    out.mkdir(parents=True, exist_ok=True)
    from main.belief_roles.eset import ERow

    for p, label in specs:
        references = []
        for r in refs.get(label, []):
            rp = Path(r)
            if not rp.is_file():
                sys.exit(f"[belief_roles] --reference {label}={r}: no E_row file at {rp}")
            references.append(ERow.load(rp))
        safe = label.replace("/", "_").replace(" ", "_")
        try:
            res = read_one(br, roles, p, label, a.threads, commit, t_re, references=references)
        except ReadRefused as e:
            sys.exit(f"[belief_roles] REFUSED: {e}")
        (out / f"{safe}.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(f"[belief_roles] {label} ({res['arm']}): {json.dumps(res['per_run'])}  "
              f"{res['timing_s']}", flush=True)
    return 0


def cmd_infer(a) -> int:
    from main.belief_roles.infer import compare_reads, summarize
    from main.belief_roles.metrics import PER_RUN_DIRECTION

    metrics = a.metric or sorted(PER_RUN_DIRECTION)
    res = {m: compare_reads([Path(p) for p in a.treat], [Path(p) for p in a.control], m,
                            boundary=a.boundary, margin=a.margin, stratum=a.stratum)
           for m in metrics}
    print(summarize(res))
    if a.json:
        print(json.dumps({m: r.to_json() for m, r in res.items()}, indent=1))
    return 0


def main(argv=None) -> int:
    from main.policy_spectrum.reader import default_threads

    ap = argparse.ArgumentParser(prog="python -m main.belief_roles", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roles", help="print the Smogon-derived role set and substitute pairs")
    r.set_defaults(fn=cmd_roles)
    r = sub.add_parser("read", help="the purpose + role reads of checkpoints on the bank (CPU)")
    r.add_argument("--bank", default=str(default_bank()))
    r.add_argument("--out", required=True)
    r.add_argument("--ckpt", action="append", required=True, help="<path.zip>=<label> (repeatable)")
    r.add_argument("--threads", type=int, default=default_threads())
    r.add_argument("--workers", type=int, default=2, help="re-encoding core processes")
    r.add_argument("--reference", action="append",
                   help="<label>=<blob>.erow.npz[,...] — EVERY blob run of the look, as the BANKED named "
                        "sets its blob reads wrote; the run's conditional metric is the MEAN over them "
                        "(repeatable; names accumulate)")
    r.set_defaults(fn=cmd_read)
    r = sub.add_parser("infer", help="§7.4's across-seed two-sample t on read JSONs")
    r.add_argument("--treat", nargs="+", required=True, help="the X5 (fixed_mass) runs' read JSONs")
    r.add_argument("--control", nargs="+", required=True,
                   help="the blob runs' read JSONs (banked: a HEAD build reads no blob checkpoint)")
    r.add_argument("--boundary", type=float, required=True,
                   help="the stopping look's t-boundary (§7.4: 5.761 / 2.683 / 1.874)")
    r.add_argument("--metric", action="append", help="a per_run metric (repeatable; default: all)")
    r.add_argument("--margin", type=float, default=0.0)
    r.add_argument("--stratum", choices=("on_pool", "off_pool"), default="on_pool")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_infer)
    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
