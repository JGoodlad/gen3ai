"""Static-token screen LOOK 1: the PLAN — the seeds, their preconditions, and the 3 × 3 cells, resolved at PLAY time.

``designs/endstate/design_static_tokens.md`` §8.1 / §8.2 (registered 2026-10-07, re-registered at P_st ``6c6d2e09``):
each static seed's 15M ``final_model.zip`` (rows) plays each legacy seed's (columns), 1,000 mirrored pairs per cell,
on the Rust eval core through ``main.h2h play-many``. h_ij = static_i's score against legacy_j (pp).

    PYTHONPATH=<6c6d2e09 checkout>/src python plan_look1.py --out cells.json [--models M] [--no-pin-check]

Exit 0 = ``--out`` holds the 9 cells (a JSON list of [player .zip, opponent .zip], the form ``main.h2h play-many
--cells`` reads) and stdout is their count. Exit 3 = REFUSED: a run missing, short of 15M by its final's
``num_timesteps``, without ``final_model.zip``, not at the registered training pin (every ``pin_history`` entry must be
6c6d2e09), the wrong ``token_encoding`` / ``belief_tokens`` / ``policy_readout``, or ``init_num_threads`` unequal
across the six (§8.1's preconditions). A missing / short seed is a refusal, never a dropped seed.

THE DEVIATIONS (Decision record 2026-10-08, amendments 1 and 2; README): S1 and S2 each had one tie-only K9(b) resume;
S3 had five tie-only stops, then finished under ``--behaviour-check warn`` from ``checkpoint_7000118``. S3 counts only
if ``s3_validity.py`` reads VALID (every post-switch probe |d log π| < 1e-4); ``--require-s3-valid`` (default on)
re-runs that check here and refuses on anything else."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
ARCHIVE_MODELS = Path("/home/goodlad/dev/gen3ai/models")
#: P_st, the re-registration commit (§8.2 Decision row 2026-10-07): the training pin AND the play / read pin
PIN = "6c6d2e0942e2111703a7e6d79bfadb31c8e51f01"
SEEDS = (1001, 1002, 1003)
ARMS = ("static", "legacy")
FINAL = "final_model.zip"
STEPS_15M = 15_000_000
#: §8.2: both arms on the adopted belief arm and production's readout
WANT = {"belief_tokens": "fixed_mass", "policy_readout": "tower"}


class Refused(Exception):
    pass


def run_dir(models: Path, arm: str, seed: int) -> Path:
    return models / f"rb_st_{arm}_s{seed}"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def final_steps(p: Path) -> int:
    """``num_timesteps`` recorded INSIDE the final's sb3 ``data`` member (the file, not the sidecar)."""
    with zipfile.ZipFile(p) as z:
        data = json.loads(z.read("data"))
    return int(data["num_timesteps"])


def resolve(models: Path, pin_check: bool = True) -> Tuple[Dict[str, Dict[int, Path]], List[str], Dict[str, dict]]:
    fails: List[str] = []
    got: Dict[str, Dict[int, Path]] = {a: {} for a in ARMS}
    facts: Dict[str, dict] = {}
    threads = set()
    for arm in ARMS:
        for s in SEEDS:
            run = run_dir(models, arm, s)
            if not run.is_dir():
                fails.append(f"{run.name}: no run directory")
                continue
            meta = json.loads((run / "metadata.json").read_text())
            cfg = json.loads((run / "model_config.json").read_text())
            pins = [str(p.get("git_hash", "")) for p in meta.get("pin_history") or [] if isinstance(p, dict)]
            if pin_check and (not pins or any(p != PIN for p in pins)):
                fails.append(f"{run.name}: pin_history {pins} is not the registered P_st {PIN[:8]}")
            if cfg.get("token_encoding") != arm:
                fails.append(f"{run.name}: token_encoding {cfg.get('token_encoding')!r} != {arm!r}")
            for k, v in WANT.items():
                if cfg.get(k) != v:
                    fails.append(f"{run.name}: {k} {cfg.get(k)!r} != {v!r}")
            threads.add(meta.get("init_num_threads"))
            f = run / FINAL
            if not f.is_file():
                fails.append(f"{run.name}: no {FINAL}")
                continue
            n = final_steps(f)
            if n < STEPS_15M:
                fails.append(f"{run.name}: {FINAL} at {n} steps < 15M")
            got[arm][s] = f
            facts[run.name] = {"final": str(f), "final_num_timesteps": n, "pin_history": pins,
                               "init_num_threads": meta.get("init_num_threads"),
                               "torch_version": meta.get("torch_version"),
                               "token_encoding": cfg.get("token_encoding"),
                               "belief_tokens": cfg.get("belief_tokens"), "policy_readout": cfg.get("policy_readout"),
                               "behaviour_check_warn_in_metadata": "behaviour-check warn" in json.dumps(meta)}
    if len(threads) != 1:
        fails.append(f"init_num_threads differ across the six: {sorted(map(str, threads))}")
    return got, fails, facts


def s3_valid() -> Tuple[bool, str]:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "s3.json"
        r = subprocess.run([sys.executable, str(HERE / "s3_validity.py"), "--out", str(out)], capture_output=True,
                           text=True)
        try:
            d = json.loads(out.read_text())
        except (OSError, ValueError):
            return False, f"no result (exit {r.returncode})"
    line = (f"{d.get('verdict')}: {d.get('post_switch_probes')} probes after the {d['switch_attach']['time']} "
            f"switch, max |d log pi| (every current row) {d.get('post_max_abs_dlogp_current_max')} < {d.get('bar')}")
    return r.returncode == 0, line


def cells(got: Dict[str, Dict[int, Path]]) -> List[List[str]]:
    return [[str(got["static"][i]), str(got["legacy"][j])] for i in SEEDS for j in SEEDS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=str(ARCHIVE_MODELS))
    ap.add_argument("--no-pin-check", action="store_true")
    ap.add_argument("--no-s3-check", action="store_true")
    a = ap.parse_args()
    got, fails, facts = resolve(Path(a.models), not a.no_pin_check)
    if not a.no_s3_check:
        ok, line = s3_valid()
        if not ok:
            fails.append(f"S3 (rb_st_static_s1003) amendment-2 validity check did not read VALID ({line})")
    if fails:
        for f in fails:
            print(f"REFUSED: {f}", file=sys.stderr)
        return 3
    cs = cells(got)
    Path(a.out).write_text(json.dumps(cs, indent=1) + "\n")
    print(len(cs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
