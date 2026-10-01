"""CLI for the ride-along heads' offline reader. See ``main/ridealong_read/__init__.py``.

  python -m main.ridealong_read --checkpoint models/<run>/<ckpt>.zip[=<label>] [--checkpoint ...] \\
      --out <dir> [--threads N] [--limit-battles N] \\
      [--drift-pair <A.zip> <B.zip> [<B2.zip> ...]]     # the RND input-choice drift read

Forward passes only, CPU. REFUSES an output (or archive) directory under ``models/`` and a bare run
directory as a checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _commit() -> str:
    from utils.git import get_git_hash

    try:
        return get_git_hash()
    except Exception:                                    # noqa: BLE001 - a read still runs
        return "unknown"


def _dirty() -> Optional[bool]:
    import subprocess

    from utils.paths import repo_root

    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                             cwd=str(repo_root()), capture_output=True, text=True, timeout=30)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:                                    # noqa: BLE001
        return None


def _ckpt(spec: str) -> Tuple[Path, str]:
    path, _, label = spec.partition("=")
    p = Path(path)
    if not label:
        label = f"{p.parent.name}__{p.stem}" if p.parent.name != "checkpoints" else \
            f"{p.parent.parent.name}__{p.stem}"
    return p, label


def _models_root() -> Optional[Path]:
    from utils.paths import main_models_dir

    return main_models_dir()


def run_rnd_choice(bank: Any, rows: np.ndarray, masks: np.ndarray, threads: int,
                   epochs: Sequence[int], drift: Optional[List[Path]],
                   pooled_cache: Dict[str, np.ndarray],
                   log: Callable[..., None] = print) -> Tuple[Optional[dict], dict]:
    """``(obs_rnd, file_body)``: obs-RND's novelty reads, and the ``rnd_input_choice.json`` body
    (obs-RND, and — with ``drift`` — feature-RND trained on A's features, scored through A and each B)."""
    from main.policy_spectrum.reader import inference_globals
    from main.ridealong_read import reader as RD
    from main.ridealong_read import rnd_choice as R

    sp = R.split(bank.decisions)
    battles = np.array([d["battle"] for d in bank.decisions])
    counts = {k: int(v.sum()) for k, v in sp.items()}
    if not (sp["train"].any() and sp["heldout"].any()):
        return None, {"skipped": "the read's rows do not span the train / held-out split", "split": counts}
    t = time.time()
    with inference_globals(threads):
        obs = R.train_and_read(rows[sp["train"]],
                               lambda rnd: R.novelty_reads(R.rnd_errors(rnd, rows), sp, battles),
                               epochs_read=epochs)
    obs.pop("_rnd")
    log(f"[ridealong_read] obs-RND trained + read ({time.time() - t:.1f} s)")
    body: dict = {"split": {"rule": f"train = sources {list(R.TRAIN_SOURCES)} (battle-level, by "
                                    "source); held-out = every other source", **counts},
                  "epochs_read": list(epochs), "obs_rnd": obs,
                  "obs_rnd_drift": "zero by construction: the observation does not depend on the "
                                   "checkpoint, so every score is bit-identical under any checkpoint"}
    if drift:
        a_path, b_paths = drift[0], drift[1:]

        def pooled(p: Path) -> np.ndarray:
            k = str(p.resolve())
            if k not in pooled_cache:
                pooled_cache[k] = RD.pooled_only(p, rows, masks, threads)
            return pooled_cache[k]

        fa = pooled(a_path)
        fbs = {str(b): pooled(b) for b in b_paths}
        t = time.time()

        def read(rnd: Any) -> dict:
            ea = R.rnd_errors(rnd, fa)
            res = {"novelty_through_A": R.novelty_reads(ea, sp, battles), "through_B": {}}
            for name, fb in fbs.items():
                eb = R.rnd_errors(rnd, fb)
                ebn = R.rnd_errors(R.renormalised(rnd, fb[sp["train"]]), fb)
                res["through_B"][name] = {
                    "drift": R.drift_reads(ea, eb, sp, battles),
                    "novelty_through_B": R.novelty_reads(eb, sp, battles),
                    "drift_renormalised": R.drift_reads(ea, ebn, sp, battles),
                    "novelty_through_B_renormalised": R.novelty_reads(ebn, sp, battles)}
            return res

        with inference_globals(threads):
            fr = R.train_and_read(fa[sp["train"]], read, epochs_read=epochs)
        fr.pop("_rnd")
        body["feature_rnd_drift"] = {
            "A": {"path": str(a_path), "sha256": RD.file_sha256(a_path)},
            "B": [{"path": str(b), "sha256": RD.file_sha256(b)} for b in b_paths],
            "rule": "feature-RND (the heads' RndNovelty shapes over value_pooled) trained on A's "
                    "features of the TRAIN rows (normalisation fitted there too); the SAME states "
                    "then scored through A and through each B — with A's frozen normalisation "
                    "(drift) and with the normalisation re-fitted on B's TRAIN-row features "
                    "(drift_renormalised: a pure shift / rescale of the features removed)",
            **fr}
        log(f"[ridealong_read] feature-RND drift read ({time.time() - t:.1f} s)")
    return obs, body


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.ridealong_read")
    ap.add_argument("--checkpoint", action="append", required=True, help="<path.zip>[=<label>]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2, help="re-encode workers")
    ap.add_argument("--limit-battles", type=int, default=None)
    ap.add_argument("--bank", default=None, help="default: the committed bank_v1")
    ap.add_argument("--truth-dir", default=None, help="default: the committed truth_v2")
    ap.add_argument("--archive", default=str(Path.home() / "gen3ai_archive" / "ridealong_read"),
                    help="per-row arrays (<label>.rows.npz); '' = none")
    ap.add_argument("--drift-pair", nargs="+", default=None, metavar="ZIP",
                    help="A B [B2 ...]: feature-RND trained on A's features, scored through each B")
    ap.add_argument("--rnd-epochs", default="3,10,30")
    ap.add_argument("--no-rnd-choice", action="store_true")
    ap.add_argument("--fresh-heads", action="store_true",
                    help="read FRESH (prior-only) heads even when the checkpoint carries trained "
                         "ones — the pre-registered floor (X26)")
    a = ap.parse_args(argv)

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import reencode, refuse_under_models
    from main.ridealong_read import reader as RD

    out = Path(a.out)
    refuse_under_models(out)
    archive = Path(a.archive) if a.archive else None
    if archive is not None:
        refuse_under_models(archive)
    ckpts = [_ckpt(s) for s in a.checkpoint]
    for p, _ in ckpts:
        RD.check_checkpoint_arg(p)
    drift = [Path(p) for p in a.drift_pair] if a.drift_pair else None
    if drift is not None:
        if len(drift) < 2:
            sys.exit("[ridealong_read] --drift-pair needs A and at least one B")
        for p in drift:
            RD.check_checkpoint_arg(p)
    epochs = tuple(sorted({int(x) for x in a.rnd_epochs.split(",") if x}))

    bank = RD.limit_bank(load_bank(Path(a.bank) if a.bank else RD.bank_dir()), a.limit_battles)
    t = time.time()
    rows, masks, gate = reencode(bank, workers=a.workers)
    print(f"[ridealong_read] re-encoded {len(rows)} decisions / {len(bank.battles)} battles in "
          f"{time.time() - t:.1f} s; obs as recorded: {gate['obs_as_recorded']} "
          f"({gate['byte_equal']}/{gate['recorded_rows_checked']})", flush=True)
    truth = RD.load_truth(Path(a.truth_dir) if a.truth_dir else RD.truth_dir())
    commit = _commit()
    dirty = _dirty()
    commit_s = commit + ("+dirty" if dirty else "")
    models_root = _models_root()

    obs_rnd = None
    rc_body: Optional[dict] = None
    pooled_cache: Dict[str, np.ndarray] = {}
    if not a.no_rnd_choice:
        obs_rnd, rc_body = run_rnd_choice(bank, rows, masks, a.threads, epochs, drift, pooled_cache)
        rc_body.update({"schema": RD.READ_SCHEMA + ":rnd_input_choice", "reader_commit": commit_s,
                        "bootstrap": RD.BOOT_METHOD,
                        "bank": {"content_sha256": bank.manifest["content_sha256"],
                                 "decisions": len(bank.decisions), "battles": len(bank.battles)},
                        "reencode": {k: v for k, v in gate.items() if k != "encoder"}})
        RD.write_json(out, "rnd_input_choice", rc_body)
    for path, label in ckpts:
        t = time.time()
        res, cols = RD.read_checkpoint(bank, rows, masks, gate, path, label, threads=a.threads,
                                       commit=commit_s, truth=truth, models_root=models_root,
                                       rnd_obs=obs_rnd, epochs_read=epochs, archive=archive,
                                       fresh_heads=bool(a.fresh_heads))
        RD.write_json(out, RD.safe_label(label), res)
        u = res["i_uncertainty_vs_v_error"]["all"]
        print(f"[ridealong_read] {label} ({res['heads']}): AUROC(|V-z|>0.5) ens_std "
              f"{u['ens_std']['auroc']} rnd_z {u['rnd_z']['auroc']} ref_v_entropy "
              f"{u['ref_v_entropy']['auroc']} ({time.time() - t:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
