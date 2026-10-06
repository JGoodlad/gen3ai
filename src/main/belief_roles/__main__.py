"""CLI for the X5 belief readers (U7). See ``main/belief_roles/__init__.py``.

  python -m main.belief_roles roles                                   # the Smogon role set + pairs
  python -m main.belief_roles read --out <dir> \\
        --ckpt <run>/checkpoints/<ckpt>.zip=<label> [...] \\
        [--reference <fixed_mass label>=<blob label | <blob>.erow.npz>]  # CPU forwards on the Lane S bank
  python -m main.belief_roles infer --treat <X5 read>.json ... --control <blob read>.json ... \\
        --boundary 2.683 [--metric intent_logloss] [--margin 0]      # §7.4's across-seed t

``--out`` under ``models/`` is REFUSED (read-only). A bare run directory is REFUSED as a checkpoint.

A blob read writes its named set E_row beside its JSON (``<label>.erow.npz``, Amendment 3(b)); a
fixed_mass read is scored on its PAIRED blob run's E_row through ``--reference`` (the registered pairing:
fixed_mass seed s ↔ blob seed s), naming either a blob label read in the same invocation or a saved
``.erow.npz``. Without one its conditional metric (the adoption gate) is None.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

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
             reencode_s: float, reference=None, erow_out: Optional[Path] = None) -> dict:
    """One checkpoint's read. ``reference``: a blob run's ``ERow`` (a fixed_mass run's E_row);
    ``erow_out``: where a blob run's own E_row is saved."""
    from main.belief_roles.eset import ERow_SCHEMA
    from main.belief_roles.forward import arm_of, file_sha256, load_strict, read_columns
    from main.belief_roles.metrics import read_all
    from main.policy_spectrum.reader import inference_globals

    if reference is not None and reference.meta.get("bank_sha256") != br.bank.manifest["content_sha256"]:
        raise SystemExit(f"[belief_roles] REFUSED: the reference E_row was read on bank "
                         f"{reference.meta.get('bank_sha256')}, this read is on "
                         f"{br.bank.manifest['content_sha256']}")
    t0 = time.time()
    with inference_globals(threads):
        model = load_strict(zip_path)
    t_load = time.time() - t0
    arm = arm_of(model)
    t1 = time.time()
    cols = read_columns(model, br, roles, threads=threads, reference=reference)
    t_fwd = time.time() - t1
    del model
    sha = file_sha256(Path(zip_path))
    bank_sha = br.bank.manifest["content_sha256"]
    if cols.own_erow is not None:
        cols.own_erow.meta = {"schema": ERow_SCHEMA, "label": label, "checkpoint_sha256": sha,
                              "bank_sha256": bank_sha}
        eref = {"mode": "own", "reference_label": label, "checkpoint_sha256": sha,
                "erow_sha256": cols.own_erow.content_sha256()}
        if erow_out is not None:
            cols.own_erow.save(erow_out)
            eref["erow_path"] = str(erow_out)
    elif reference is not None:
        eref = {"mode": "paired", "reference_label": reference.meta.get("label"),
                "checkpoint_sha256": reference.meta.get("checkpoint_sha256"),
                "erow_sha256": reference.content_sha256(), "pairing_rule": "fixed_mass seed s ↔ blob seed s"}
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
    labels = [lb for _, lb in specs]
    if len(set(labels)) != len(labels):
        sys.exit("[belief_roles] two --ckpt share a label")
    refs = {}
    for spec in a.reference or []:
        fm, _, ref = spec.partition("=")
        if fm not in labels or not ref:
            sys.exit(f"[belief_roles] --reference {spec!r}: name it <fixed_mass label>=<blob label | "
                     "path.erow.npz>, the label one of this invocation's --ckpt")
        refs[fm] = ref
    # a blob named as a reference is read FIRST, so its E_row exists when its fixed_mass partner is read
    ref_labels = set(refs.values())
    specs.sort(key=lambda x: x[1] not in ref_labels)
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

    def safe_of(lb: str) -> str:
        return lb.replace("/", "_").replace(" ", "_")

    for p, label in specs:
        reference = None
        if label in refs:
            r = refs[label]
            rp = out / f"{safe_of(r)}.erow.npz" if r in labels else Path(r)
            if not rp.is_file():
                sys.exit(f"[belief_roles] --reference {label}={r}: no E_row file at {rp}")
            reference = ERow.load(rp)
        safe = safe_of(label)
        res = read_one(br, roles, p, label, a.threads, commit, t_re, reference=reference,
                       erow_out=out / f"{safe}.erow.npz")
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
                   help="<fixed_mass label>=<blob label | path.erow.npz> — the paired blob run whose named "
                        "set E_row the fixed_mass run's conditional metric is scored on (repeatable)")
    r.set_defaults(fn=cmd_read)
    r = sub.add_parser("infer", help="§7.4's across-seed two-sample t on read JSONs")
    r.add_argument("--treat", nargs="+", required=True, help="the X5 (fixed_mass) runs' read JSONs")
    r.add_argument("--control", nargs="+", required=True, help="the blob runs' read JSONs")
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
