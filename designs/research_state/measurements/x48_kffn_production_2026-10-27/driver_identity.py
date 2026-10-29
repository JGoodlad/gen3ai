"""X48 PREREG NOTE 5 (a): does the architecture-pure driver reproduce the runner's banked S rows exactly?

One already-banked C0 S cell pair, regrouped as the driver groups it: ``c0_s3501`` against ``x26_pstar`` and against
``tstar``, which the runner played in MIXED engines. `x48_s_driver.pure_engines` splits them into two engines (one per
architecture). Each engine's argv is the runner's own `play_many_argv` at the pin (what the `s-unit` verb runs), with
two changes that do not enter a row's identity: ``--pairs 100`` (batch 0 only; LR-F's identity showed a batch's seed
block does not depend on the cell's pair count) and ``--out`` a scratch ledger (its family registered as the
runner does). Both engines run in ONE short-lane heavy slot (`mem_cap.sh --slot --max-s 600`), nice 19, agent cores.
Then every scratch row is compared with the read ledger's banked row of the same (player sha256, opponent sha256,
batch): regime id, seed block, counts and both outcome digests, exact. Exit 0 iff all are equal. Run it from the
clean pin worktree with its ``src`` on PYTHONPATH.

    python .../driver_identity.py --state ~/gen3ai_archive/x35_x48_read/look1 \
        --scratch ~/gen3ai_archive/x35_x48_read/driver_identity > .../driver_identity.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from agents.training.eval_ledger.store import scan_rows
from main.reads import layout as LY
from main.reads import runner as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x48_s_driver as X  # noqa: E402

LABEL, MEMBERS, GROUP = "c0_s3501", ["x26_pstar", "tstar"], "pool"
FIELDS = ("regime_id", "seed", "counts", "outcome_digest", "outcome_digest_all")


def _view(r: Dict[str, Any]) -> Dict[str, Any]:
    return {"regime_id": r["regime"]["regime_id"], "seed": r["seed"], "counts": r["counts"],
            "outcome_digest": r["compute"].get("outcome_digest"),
            "outcome_digest_all": r["compute"].get("outcome_digest_all"), "commit": r.get("commit")}


def _key(r: Dict[str, Any]) -> tuple:
    return (r["player"]["sha256"], r["opponent"]["sha256"], int(r["seed"]["batch"]))


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    a = ap.parse_args(argv)
    state, scratch = a.state.expanduser(), a.scratch.expanduser()
    spec = R.load_state(state)
    models = R._load_models(state)
    members = json.loads((LY.s_dir(state, GROUP) / "state.json").read_text())["members"]
    engines = X.pure_engines([(LABEL, MEMBERS)], members, workers=len(MEMBERS))
    if len(engines) != 2:
        raise SystemExit(f"the driver did not split {MEMBERS} into two engines: {engines}")
    ledger = scratch / "ledger"
    ledger.mkdir(parents=True, exist_ok=True)
    fam = spec.s[GROUP].family
    subprocess.run([sys.executable, "-m", "main.eval_ledger", "family-register", "--root", str(ledger), "--id", fam,
                    "--decision-kind", "ab_verdict", "--rule", R.FAMILY_RULE, "--protocol",
                    "gen3_eval_protocol_v1_h2h"], check=True, env=R.child_env(), capture_output=True)
    shell: List[str] = []
    for e in engines:
        lab, ms = e.split(":", 1)
        av = R.play_many_argv(spec, state, GROUP, lab, ms.split(","), models, members, threads=R.S_THREADS)
        av[av.index("--pairs") + 1] = "100"
        av[av.index("--out") + 1] = str(ledger)
        shell.append(" ".join(json.dumps(x) for x in av) + f" > {json.dumps(str(scratch / (e.replace(':', '_') + '.log')))} 2>&1 &")
    script = "\n".join(shell) + "\nrc=0; for p in $(jobs -p); do wait $p || rc=1; done; exit $rc\n"
    cmd = [str(Path(R.__file__).resolve().parents[3] / "scripts/ops/mem_cap.sh"), "--slot", "--max-s", "600",
           "--name", "x48_driver_identity", "12", "nice", "-n", "19", "taskset", "-c", R.CORES, "bash", "-c", script]
    rc = subprocess.run(cmd, check=False, env=R.child_env()).returncode
    problems: list = []
    bank = {_key(s.row): s.row for s in scan_rows(LY.s_ledger(state, GROUP), problems)}
    rows = []
    for s in scan_rows(ledger, problems):
        r, b = s.row, bank.get(_key(s.row))
        row: Dict[str, Any] = {"player": r["player"]["id"], "opponent": r["opponent"]["id"],
                               "batch": int(r["seed"]["batch"]), "replay": _view(r)}
        if b is None:
            row.update(equal=False, why="no banked row for this cell and batch")
        else:
            diff = [f for f in FIELDS if _view(r)[f] != _view(b)[f]]
            row.update(bank=_view(b), differing=diff, equal=not diff)
        rows.append(row)
    ok = rc == 0 and len(rows) == len(MEMBERS) and all(x["equal"] for x in rows) and not problems
    print(json.dumps({"schema": "x48_driver_identity_v1", "engines": engines, "play_rc": rc, "fields": list(FIELDS),
                      "rows": rows, "ledger_problems": problems, "identical": ok}, indent=1, sort_keys=True,
                     default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
