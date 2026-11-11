"""CE-3d, the trailing-magnet screen (PREREG §5.2): the PLAN of the three head-to-head crosses.

* ``deciding``: rows = the three MAGNET forks ``rb_x35_ce3dmag_s7301..s7303``, columns = the three CONTROL forks
  ``rb_x35_ce3dent_s7311..s7313``. All six are REQUIRED.
* ``mag_vs_c0`` / ``ent_vs_c0`` (REPORTED): the same rows / the control forks as rows, against the C0 finals
  ``rb_x35_c0_s3501..s3508``: the five banked (3501-3505) are REQUIRED; a top-up seed (3506-3508) with no
  ``final_model.zip`` is left out and named, and one that has a final but fails a guard REFUSES the plan.

Every run in the plan must have: ``final_model.zip``; a ``pin_history`` of exactly ONE entry at ITS pin (the CE-3d
backport pin for the six forks, the production side pin for C0), from the fork step 280,000,170 to at least
300,000,170; the lineage record of a FORK of P* (the parent checkpoint's sha256); its own seed; ``n_epochs`` 10; its
arm's ``ent_coef`` (magnet 0.05, control 0.1, C0 0.05) in ``metadata.json`` and its ``cli_args``; its arm's
``magnet_kl_coef`` in ``model_config.json`` (magnet 0.05 with ``magnet_refresh_steps`` 2,000,000; control 0.0; C0:
the key ABSENT, a v161 config written at the production pin); for a magnet fork, the committed final's magnet
sidecar ``final_model.magnet.pt``; K9(b) PASS over its own rows (``main.x35.batch.k9b_read``, AM-6). All runs must
share one ``torch_version`` and one ``init_num_threads``. REFUSES (exit 3) otherwise.

Writes ``<out-dir>/cells_<cross>.json`` (the [player .zip, opponent .zip] lists ``main.h2h play-many --cells`` reads)
and ``<out-dir>/shas.json`` (per cross: row and column sha256 lists, seeds and paths, for the reader). Stdout: the
three cell counts. ``--columns-only`` checks the C0 columns alone and writes nothing (the registration's own check,
run before any CE-3d fork exists).

Run with ``PYTHONPATH=<a checkout>/src`` (``main.x35.batch`` supplies the K9(b) read).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MODELS = Path(os.environ.get("GEN3AI_MODELS_DIR") or "/home/goodlad/dev/gen3ai/models")
SIDE_PIN = "f4b25238def57900c0305bb2a3a5a8f9edaef285"
CE3D_PIN = "333a73ddda7829b9ccdf9e17ef04624eb40edeba"
PSTAR_SHA256 = "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"
FORK_STEP, MIN_LAST_STEP = 280_000_170, 300_000_170
N_EPOCHS = 10
REFRESH_STEPS = 2_000_000
#: arm -> (run-name format, seeds, required seeds, pin, ent_coef, magnet_kl_coef or None = the key must be absent)
ARMS: Dict[str, Tuple[str, Tuple[int, ...], Tuple[int, ...], str, float, Optional[float]]] = {
    "mag": ("rb_x35_ce3dmag_s{}", (7301, 7302, 7303), (7301, 7302, 7303), CE3D_PIN, 0.05, 0.05),
    "ent": ("rb_x35_ce3dent_s{}", (7311, 7312, 7313), (7311, 7312, 7313), CE3D_PIN, 0.1, 0.0),
    "c0": ("rb_x35_c0_s{}", (3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508), (3501, 3502, 3503, 3504, 3505),
           SIDE_PIN, 0.05, None),
}
#: cross -> (row arm, column arm)
CROSSES: Dict[str, Tuple[str, str]] = {"deciding": ("mag", "ent"), "mag_vs_c0": ("mag", "c0"),
                                       "ent_vs_c0": ("ent", "c0")}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_run(run: Path, seed: int, pin: str, ent: float, magnet: Optional[float]) -> Tuple[List[str], Dict[str, Any]]:
    """``(problems, facts)`` for one finished run; the caller has checked ``final_model.zip`` exists."""
    import main.x35.batch as B

    probs: List[str] = []
    md = json.loads((run / "metadata.json").read_text())
    ph = md.get("pin_history") or []
    seen = [(str(p.get("git_hash"))[:8], p.get("first_step"), p.get("last_step")) for p in ph]
    if len(ph) != 1 or ph[0].get("git_hash") != pin:
        probs.append(f"pin_history {seen} is not one entry at {pin[:8]}")
    else:
        if int(ph[0].get("first_step") or -1) != FORK_STEP:
            probs.append(f"its pin entry starts at {ph[0].get('first_step')}, not the fork step {FORK_STEP}")
        if int(ph[0].get("last_step") or 0) < MIN_LAST_STEP:
            probs.append(f"last_step {ph[0].get('last_step')} < {MIN_LAST_STEP}")
    lin = md.get("lineage") or {}
    parent = lin.get("fork_parent") or {}
    if lin.get("role") != "fork" or parent.get("sha256") != PSTAR_SHA256:
        probs.append(f"lineage is not a FORK of P* (role {lin.get('role')!r}, parent sha {str(parent.get('sha256'))[:12]})")
    cli = md.get("cli_args") or {}
    if cli.get("seed") != seed:
        probs.append(f"seed {cli.get('seed')!r} != {seed}")
    if cli.get("n_epochs") != N_EPOCHS:
        probs.append(f"n_epochs {cli.get('n_epochs')!r} != {N_EPOCHS}")
    for where, got in (("metadata.ent_coef", md.get("ent_coef")), ("cli_args.ent_coef", cli.get("ent_coef"))):
        if not isinstance(got, (int, float)) or abs(float(got) - ent) > 1e-12:
            probs.append(f"{where} {got!r} != {ent}")
    cfg = json.loads((run / "model_config.json").read_text())
    got_mag = cfg.get("magnet_kl_coef", None)
    if magnet is None:
        if "magnet_kl_coef" in cfg or "magnet_refresh_steps" in cfg:
            probs.append(f"model_config.json carries a magnet field ({got_mag!r}); a run at {pin[:8]} cannot")
    else:
        if not isinstance(got_mag, (int, float)) or abs(float(got_mag) - magnet) > 1e-12:
            probs.append(f"model_config.magnet_kl_coef {got_mag!r} != {magnet}")
        if magnet > 0.0 and cfg.get("magnet_refresh_steps") != REFRESH_STEPS:
            probs.append(f"model_config.magnet_refresh_steps {cfg.get('magnet_refresh_steps')!r} != {REFRESH_STEPS}")
    side = run / "final_model.magnet.pt"
    if magnet is not None and magnet > 0.0 and not side.is_file():
        probs.append("no final_model.magnet.pt beside the final (a magnet run commits one with every checkpoint)")
    if (magnet is None or magnet == 0.0) and side.is_file():
        probs.append("a final_model.magnet.pt sits beside the final of a run with no magnet")
    k = B.k9b_read(str(run))
    if k.verdict != "PASS":
        probs.append(f"K9(b) {k.verdict}: {'; '.join(k.reasons + k.fires) or 'no reason given'}")
    return probs, {"torch_version": md.get("torch_version"), "init_num_threads": md.get("init_num_threads"),
                   "num_timesteps": md.get("num_timesteps")}


def collect(arm: str, fails: List[str], notes: List[str]) -> List[Tuple[int, Path, Dict[str, Any]]]:
    fmt, seeds, required, pin, ent, magnet = ARMS[arm]
    out = []
    for s in seeds:
        run = MODELS / fmt.format(s)
        fz = run / "final_model.zip"
        if not fz.is_file():
            (fails if s in required else notes).append(f"{run.name}: no final_model.zip"
                                                       + ("" if s in required else " (left out of the columns)"))
            continue
        probs, facts = check_run(run, s, pin, ent, magnet)
        fails.extend(f"{run.name}: {p}" for p in probs)
        out.append((s, fz, facts))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir")
    ap.add_argument("--columns-only", action="store_true")
    a = ap.parse_args()
    if not a.columns_only and not a.out_dir:
        ap.error("--out-dir is required unless --columns-only")
    fails: List[str] = []
    notes: List[str] = []
    got = {arm: collect(arm, fails, notes) for arm in (("c0",) if a.columns_only else ("mag", "ent", "c0"))}
    for what in ("torch_version", "init_num_threads"):
        seen = sorted({str(f[what]) for runs in got.values() for _, _, f in runs})
        if len(seen) > 1:
            fails.append(f"{what} differs across the runs: {seen}")
    for n in notes:
        print("NOTE: " + n, file=sys.stderr)
    if fails:
        print("REFUSED: " + "; ".join(fails), file=sys.stderr)
        return 3
    if a.columns_only:
        print(f"columns OK: {[s for s, _, _ in got['c0']]} ({len(got['c0'])} C0 finals pass every guard)")
        return 0
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shas = {arm: {"sha256": [sha(p) for _, p, _ in runs], "seeds": [s for s, _, _ in runs],
                  "runs": [str(p) for _, p, _ in runs]} for arm, runs in got.items()}
    doc: Dict[str, Any] = {"left_out": notes, "crosses": {}}
    counts = []
    for cross, (ra, ca) in CROSSES.items():
        cells = [[r, c] for r in shas[ra]["runs"] for c in shas[ca]["runs"]]
        (out / f"cells_{cross}.json").write_text(json.dumps(cells, indent=1))
        doc["crosses"][cross] = {"rows": shas[ra]["sha256"], "cols": shas[ca]["sha256"],
                                 "row_seeds": shas[ra]["seeds"], "col_seeds": shas[ca]["seeds"],
                                 "row_runs": shas[ra]["runs"], "col_runs": shas[ca]["runs"]}
        counts.append(f"{cross}={len(cells)}")
    (out / "shas.json").write_text(json.dumps(doc, indent=1))
    print(" ".join(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
