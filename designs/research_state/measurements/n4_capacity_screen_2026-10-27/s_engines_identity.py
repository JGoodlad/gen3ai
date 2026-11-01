"""N4 AM-N4-PIN item 2 / AM-N4-FIXES item 6: does the S lane's architecture-aware ``runner.s_engines`` reproduce
banked toy S rows EXACTLY? X48 PREREG NOTE 5 (a)'s pattern, on CPU.

The bank: X47 look 1 (``~/gen3ai_archive/x47_look1_2026-10-21/ledger``), toy cells of 500 mirrored pairs in ONE
batch, each played ALONE in its own engine, at ``f4b0a74f2`` on engine E1 (core ``7c53280566d666ae/125``), regime
``adfdbee824c9eb0a``, ``--torch-threads 2 --threads 2``. The replay plays at ``86cfdf7ef`` (the X35-frame shared pin,
the same engine E1 and regime; the clean pin worktree ``--pin-root``) with the bank's compute flags, and groups the
cells into engines with THIS checkout's ``runner.s_engines`` and ``runner.arch_key_of_config``:

* ``grouped``: label X47 s3601's final (``trunk_layers`` 4, a third architecture) with the toy R members ``toy_080m``,
  ``toy_150m`` and ``pstar_x26`` (production) pending, 2 workers. The old stripe would put X47 + toy + production on
  one engine (``CellArchMismatch``); the new grouping gives ``x47_s3601:toy_080m,toy_150m`` and
  ``x47_s3601:pstar_x26``.
* ``stripe``: label toy C0 s3601's final with ``toy_080m`` and ``pstar_x26`` pending, 1 worker: two architectures,
  so the stripe is kept, ONE mixed engine ``c0_s3601:toy_080m,pstar_x26`` (as the runner always played them).

Every engine of a case runs at once in ONE short-lane heavy slot (``mem_cap.sh --slot --max-s 600``), nice 19, agent
cores, into a scratch ledger. Then every scratch row is compared with the bank's row of the same (player sha256,
opponent sha256, batch): regime id, seed block, counts and both outcome digests, exact. Exit 0 iff all are equal.

    PYTHONPATH=<this checkout>/src python .../s_engines_identity.py --case grouped \\
        --pin-root /home/goodlad/dev/gen3ai-reads/lrf_play_86cfdf7e --scratch ~/gen3ai_archive/n4_s_engines_identity
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from agents.training.eval_ledger.store import scan_rows
from main.reads import runner as R
from utils.paths import main_models_dir, repo_path

BANK = Path.home() / "gen3ai_archive/x47_look1_2026-10-21/ledger"
PIN = "86cfdf7efbe21b4ff3382541e6d78390df7d9966"
FIELDS = ("regime_id", "seed", "counts", "outcome_digest", "outcome_digest_all")
REFPOP = "designs/research_state/measurements/x35_breakers_stage_2026-10-17/reference_population_toy.json"
CASES: Dict[str, Dict[str, Any]] = {
    "grouped": {"label": "x47_s3601", "run": "rb_x35t_x47d4_s3601", "members": ["toy_080m", "toy_150m", "pstar_x26"],
                "workers": 2, "expect": ["x47_s3601:toy_080m,toy_150m", "x47_s3601:pstar_x26"]},
    "stripe": {"label": "c0_s3601", "run": "rb_x35t_c0_s3601", "members": ["toy_080m", "pstar_x26"], "workers": 1,
               "expect": ["c0_s3601:toy_080m,pstar_x26"]},
}


def _config(zip_path: Path) -> Path:
    return next(d / "model_config.json" for d in (zip_path.parent, zip_path.parent.parent)
                if (d / "model_config.json").is_file())


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _view(r: Dict[str, Any]) -> Dict[str, Any]:
    return {"regime_id": r["regime"]["regime_id"], "seed": r["seed"], "counts": r["counts"],
            "outcome_digest": r["compute"].get("outcome_digest"),
            "outcome_digest_all": r["compute"].get("outcome_digest_all"), "commit": r.get("commit"),
            "core_stamp_src": [x for x in r["compute"].get("core_stamp", "").split(";") if x.startswith("src=")],
            "torch_threads": r["compute"].get("torch_threads"), "threads": r["compute"].get("threads")}


def _key(r: Dict[str, Any]) -> tuple:
    return (r["player"]["sha256"], r["opponent"]["sha256"], int(r["seed"]["batch"]))


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", choices=sorted(CASES), required=True)
    ap.add_argument("--pin-root", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    a = ap.parse_args(argv)
    c = CASES[a.case]
    pin_root, scratch = a.pin_root.resolve(), a.scratch.expanduser() / a.case
    head = subprocess.run(["git", "-C", str(pin_root), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(pin_root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if head != PIN or dirty:
        raise SystemExit(f"{pin_root} is at {head[:12]}{' (dirty)' if dirty else ''}, not the clean pin {PIN[:12]}")
    if (scratch / "ledger").exists():
        raise SystemExit(f"{scratch}/ledger exists: a replay is played once into a fresh ledger")
    refpop = {m["label"]: m for m in json.loads(repo_path(REFPOP).read_text())["members"]}
    player = main_models_dir() / c["run"] / "final_model.zip"
    paths = {c["label"]: player, **{m: Path(refpop[m]["path"]) for m in c["members"]}}
    for m in c["members"]:
        if _sha(paths[m]) != refpop[m]["sha256"]:
            raise SystemExit(f"{m}: {paths[m]} no longer hashes to R's recorded sha256")
    keys = {lab: tuple(map(tuple, R.arch_key_of_config(json.loads(_config(p).read_text())))) for lab, p in paths.items()}
    arch_ids: Dict[tuple, int] = {}
    arch_of = {lab: arch_ids.setdefault(k, len(arch_ids)) for lab, k in keys.items()}
    pending = [(c["label"], list(c["members"]))]
    engines = R.s_engines(pending, c["workers"], label_arch=lambda lab: keys[lab], member_arch=lambda m: keys[m])
    old = R.s_engines(pending, c["workers"])
    if engines != c["expect"]:
        raise SystemExit(f"s_engines gave {engines}, not the expected {c['expect']}")
    ledger = scratch / "ledger"
    ledger.mkdir(parents=True)
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=str(pin_root / "src"), CUDA_VISIBLE_DEVICES="")
    subprocess.run([py, "-m", "main.eval_ledger", "family-register", "--root", str(ledger), "--id", "x35_x47",
                    "--decision-kind", "ab_verdict", "--rule", R.FAMILY_RULE, "--protocol",
                    "gen3_eval_protocol_v1_h2h"], check=True, env=env, cwd=pin_root, capture_output=True)
    shell: List[str] = []
    for e in engines:
        lab, ms = e.split(":", 1)
        av = [py, "-m", "main.h2h", "play-many", "--players", str(paths[lab]), "--opponents",
              *[str(paths[m]) for m in ms.split(",")], "--pairs", "500", "--batch-pairs", "500", "--purpose", "ab",
              "--family", "x35_x47", "--request", f"n4_s_engines_identity_{a.case}", "--out", str(ledger),
              "--label", "n4_s_engines_identity", "--device", "cpu", "--torch-threads", "2", "--threads", "2"]
        log = scratch / (e.replace(":", "_").replace(",", "+") + ".log")
        shell.append(" ".join(json.dumps(x) for x in av) + f" > {json.dumps(str(log))} 2>&1 &")
    script = f"cd {json.dumps(str(pin_root))}\n" + "\n".join(shell) + \
        "\nrc=0; for p in $(jobs -p); do wait $p || rc=1; done; exit $rc\n"
    cmd = [str(repo_path("scripts/ops/mem_cap.sh")), "--slot", "--max-s", "600", "--name",
           f"n4_s_engines_identity_{a.case}", "12", "nice", "-n", "19", "taskset", "-c", R.CORES, "bash", "-c", script]
    rc = subprocess.run(cmd, check=False, env=env).returncode
    problems: list = []
    bank = {_key(s.row): s.row for s in scan_rows(BANK, problems)}
    rows = []
    for s in scan_rows(ledger, problems):
        r, b = s.row, bank.get(_key(s.row))
        row: Dict[str, Any] = {"player": r["player"]["id"], "opponent": r["opponent"]["id"],
                               "batch": int(r["seed"]["batch"]), "replay": _view(r)}
        if b is None:
            row.update(equal=False, why="no banked row for this cell and batch")
        else:
            diff = [f for f in FIELDS if _view(r)[f] != _view(b)[f]]
            row.update(bank=_view(b), bank_row_id=b.get("row_id"), differing=diff, equal=not diff)
        rows.append(row)
    ok = rc == 0 and len(rows) == len(c["members"]) and all(x["equal"] for x in rows) and not problems
    print(json.dumps({"schema": "n4_s_engines_identity_v1", "case": a.case, "pin": PIN, "bank": str(BANK),
                      "arch_of": arch_of, "engines": engines, "old_stripe": old, "play_rc": rc,
                      "fields": list(FIELDS), "rows": rows, "ledger_problems": problems, "identical": ok},
                     indent=1, sort_keys=True, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
