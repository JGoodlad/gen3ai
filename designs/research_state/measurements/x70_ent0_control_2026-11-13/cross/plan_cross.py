"""X70, the ent-0 control of the trailing-magnet screen (PREREG §5): the PLAN of a head-to-head cross.

THREE crosses share this plan, chosen by ``--cross``:

* ``anchor`` (the default; the REGISTERED cross, which decides): rows = the three banked SWAP forks
  ``rb_x35_ce3dswap_s7321..s7323``, columns = the four X70 forks ``rb_x35_x70ent0_s7331..s7334``; 4,000 pairs a cell.
* ``c0`` (REPORTED, decides nothing): rows = the four X70 forks, columns = the eight C0 finals
  ``rb_x35_c0_s3501..s3508``; 1,000 pairs a cell.
* ``x67`` (REPORTED, decides nothing): rows = the four X70 forks, columns = X67's three finals
  ``rb_x35_x67ent_s6701..s6703``; 1,000 pairs a cell.

EVERY run of a cross is REQUIRED: a missing final or a failed guard REFUSES the plan (exit 3). A run that exists and
is not clean is a question for the orchestrator, never a silent drop; an X70 fork that is VOID (PREREG §8) is
replaced only by a dated amendment, before any cell is played. CE-3d's plan
(``../../ce3d_magnet_screen_2026-11-11/cross/plan_cross.py``) with this screen's arms; the guards are that file's.

Every run in the plan must have: ``final_model.zip``; a ``pin_history`` of exactly ONE entry at ITS pin (the CE-3d
TRAINING pin ``9e97cad5`` for a swap fork and for an X70 fork; the production side pin ``f4b25238`` for C0 and X67),
from the fork step 280,000,170 to at least 300,000,170; the lineage record of a FORK of P* (the parent checkpoint's
sha256); its own seed; ``n_epochs`` 10; its arm's ``ent_coef`` (swap 0.0, X70 0.0, C0 0.05, X67 0.0125) in
``metadata.json`` and its ``cli_args``; its arm's magnet record in ``model_config.json`` (swap: ``magnet_kl_coef``
0.05 with ``magnet_refresh_steps`` 2,000,000; X70: ``magnet_kl_coef`` PRESENT and 0.0, the pin writes the field for
every run; C0 and X67: both keys ABSENT, a v161 config written at the production pin); the committed final's magnet
sidecar ``final_model.magnet.pt`` for a swap fork and NONE beside any other final; K9(b) PASS over its own rows
(``main.x35.batch.k9b_read``, AM-6). All runs must share one ``torch_version`` and one ``init_num_threads``.

Writes ``--out`` (the [player .zip, opponent .zip] list ``main.h2h play-many --cells`` reads) and ``--shas`` (row and
column sha256 lists and seeds for the reader, ``cross`` and ``pairs``). Stdout: the cell count. ``--banked-only``
checks the runs that exist at the registration (the swap forks, the C0 finals, X67's finals) and writes nothing.

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
#: The TRAINING pin of the swap forks and of the X70 forks (``main.x35.batch.CE3D_PIN``, tag ``ce3d-pin-9e97cad5``).
#: The games are PLAYED at the first CE-3d pin ``333a73dd`` (``drive_cross.sh``), as CE-3d's were.
CE3D_PIN = "9e97cad5709da21a1f9d4d8846096c87eb17b1d0"
PSTAR_SHA256 = "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"
FORK_STEP, MIN_LAST_STEP = 280_000_170, 300_000_170
N_EPOCHS = 10
REFRESH_STEPS = 2_000_000
#: arm -> (run-name format, seeds (all REQUIRED), pin, ent_coef, magnet_kl_coef or None = the key must be absent)
ARMS: Dict[str, Tuple[str, Tuple[int, ...], str, float, Optional[float]]] = {
    "swap": ("rb_x35_ce3dswap_s{}", (7321, 7322, 7323), CE3D_PIN, 0.0, 0.05),
    "x70": ("rb_x35_x70ent0_s{}", (7331, 7332, 7333, 7334), CE3D_PIN, 0.0, 0.0),
    "c0": ("rb_x35_c0_s{}", (3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508), SIDE_PIN, 0.05, None),
    "x67": ("rb_x35_x67ent_s{}", (6701, 6702, 6703), SIDE_PIN, 0.0125, None),
}
#: cross -> (row arm, column arm, pairs a cell). ``anchor`` DECIDES; ``c0`` and ``x67`` are REPORTED.
CROSSES: Dict[str, Tuple[str, str, int]] = {"anchor": ("swap", "x70", 4000), "c0": ("x70", "c0", 1000),
                                            "x67": ("x70", "x67", 1000)}
BANKED_ARMS = ("swap", "c0", "x67")


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
        if isinstance(got, bool) or not isinstance(got, (int, float)) or abs(float(got) - ent) > 1e-12:
            probs.append(f"{where} {got!r} != {ent}")
    cfg = json.loads((run / "model_config.json").read_text())
    got_mag = cfg.get("magnet_kl_coef", None)
    if magnet is None:
        if "magnet_kl_coef" in cfg or "magnet_refresh_steps" in cfg:
            probs.append(f"model_config.json carries a magnet field ({got_mag!r}); a run at {pin[:8]} cannot")
    else:
        if isinstance(got_mag, bool) or not isinstance(got_mag, (int, float)) or abs(float(got_mag) - magnet) > 1e-12:
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


def collect(arm: str, fails: List[str]) -> List[Tuple[int, Path, Dict[str, Any]]]:
    fmt, seeds, pin, ent, magnet = ARMS[arm]
    out = []
    for s in seeds:
        run = MODELS / fmt.format(s)
        fz = run / "final_model.zip"
        if not fz.is_file():
            fails.append(f"{run.name}: no final_model.zip")
            continue
        probs, facts = check_run(run, s, pin, ent, magnet)
        fails.extend(f"{run.name}: {p}" for p in probs)
        out.append((s, fz, facts))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--shas")
    ap.add_argument("--banked-only", action="store_true")
    ap.add_argument("--cross", choices=tuple(CROSSES), default="anchor")
    a = ap.parse_args()
    if not a.banked_only and not (a.out and a.shas):
        ap.error("--out and --shas are required unless --banked-only")
    fails: List[str] = []
    if a.banked_only:
        got = {arm: collect(arm, fails) for arm in BANKED_ARMS}
        runs = [x for v in got.values() for x in v]
    else:
        row_arm, col_arm, pairs = CROSSES[a.cross]
        rows, cols = collect(row_arm, fails), collect(col_arm, fails)
        runs = rows + cols
    for what in ("torch_version", "init_num_threads"):
        seen = sorted({str(f[what]) for _, _, f in runs})
        if len(seen) > 1:
            fails.append(f"{what} differs across the runs: {seen}")
    if fails:
        print("REFUSED: " + "; ".join(fails), file=sys.stderr)
        return 3
    if a.banked_only:
        print("banked runs OK: " + "; ".join(f"{arm} {[s for s, _, _ in v]}" for arm, v in got.items())
              + f" ({len(runs)} finals pass every guard)")
        return 0
    cells = [[str(r), str(c)] for _, r, _ in rows for _, c, _ in cols]
    Path(a.out).write_text(json.dumps(cells, indent=1))
    Path(a.shas).write_text(json.dumps({
        "cross": a.cross, "pairs": pairs, "row_arm": row_arm, "col_arm": col_arm,
        "rows": [sha(p) for _, p, _ in rows], "cols": [sha(p) for _, p, _ in cols],
        "row_seeds": [s for s, _, _ in rows], "col_seeds": [s for s, _, _ in cols],
        "row_runs": [str(p) for _, p, _ in rows], "col_runs": [str(p) for _, p, _ in cols]}, indent=1))
    print(len(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
