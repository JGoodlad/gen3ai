"""P5 identity capture: every prober JSON-CLI command on a FIXED trace set, before vs after.

``gen3_pokeenv_p5_identity_v1``. The poke-env retirement's P5 moves the prober's reconstruction of a
Rust-eval core trace off poke-env (``main.prober.core_walk``). Its gate: on a fixed set of banked
eval traces every command's JSON output is IDENTICAL before vs after, or each difference is
explained. This script is the instrument; ``README.md`` holds the read.

    # 1. a CLOSED fixture: the declared battles COPIED read-only out of the archive into mini runs
    python capture.py fixture --dir /tmp/p5_fixture
    # 2. capture (CPU only), BOTH sides pinned: PYTHONHASHSEED=0 (the live recorder's multi-faint order is a
    #    frozenset's — README F-P5-1) and --threads 4 (analyze's floats move with torch's thread count — F-P5-6);
    #    BEFORE runs the base commit from a detached worktree (--src), AFTER installs utils.poke_env_blocker FIRST
    PYTHONHASHSEED=0 python capture.py run --fixture /tmp/p5_fixture --out /tmp/p5_before --src <base>/src --threads 4
    PYTHONHASHSEED=0 python capture.py run --fixture /tmp/p5_fixture --out /tmp/p5_after --blocked --threads 4
    # 3. the identity table
    python capture.py diff /tmp/p5_before /tmp/p5_after

Run it with the checkout's own ``src`` importable (it inserts ``<git toplevel>/src`` itself). Every
command runs IN PROCESS through ``main.prober.query``'s own parser and dispatcher, so the JSON is
exactly what ``python -m main.prober.query <argv>`` prints (``json.dumps(out, indent=2)``); an error
is captured as the CLI's ``{"error": …}`` object, never dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True,
                          text=True, check=True).stdout.strip())
ARCHIVE = Path(os.environ.get("GEN3AI_MODELS_DIR") or "/home/goodlad/dev/gen3ai/models")

#: The fixed trace set: run -> trace prefixes (``step_N/<opponent>/<outcome>_sK_NNN``). Two Rust-eval
#: CORE-trace runs (``sizing_C`` is at the current architecture — the model-loading run) and one
#: poke-env-eval run (full summaries + ``_replay.html``: unaffected by P5, the control).
BATTLES = {
    "sizing_C_n256_e5_s1001": [
        "step_8000030/sentinel_0/loss_s0_005", "step_8000030/sentinel_0/win_s1_001",
        "step_8000030/sentinel_1/loss_s2_003", "step_8000030/aggressive/win_s2_002",
        "step_8000030/heuristic/loss_s0_004", "step_8000030/staller_v2/loss_s0_004",
        "step_6000215/sentinel_0/draw_s3_005", "step_2000128/aggressive/draw_s0_006",
        "step_4000000/setup_sweep_v2/win_s2_005",
    ],
    "rb_x5ab_blob_s1008": [
        "step_10000076/aggressive/loss_s0_003", "step_12000041/staller_v2/draw_s1_003",
        "step_12000041/sentinel_1/draw_s1_007", "step_14000098/sentinel_4/loss_s0_003",
        "step_14000098/heuristic/win_s2_001", "step_8000179/sentinel_0/loss_s0_004",
        "step_8000179/staller/win_s3_002",
    ],
    "ai_v14_01_base": [
        "step_74000016/heuristic/win_s3_002", "step_74000016/sentinel_2/loss_s2_004",
        "step_74000016/setup_sweep_v2/loss_s3_003", "step_74000016/sentinel_0/loss_s0_002",
        "step_36000000/aggressive/draw_s1_006",
    ],
}
#: The run whose checkpoints load under the current code (the model-loading commands run there).
MODEL_RUN = "sizing_C_n256_e5_s1001"
#: Files and directories LINKED (never copied, never written) so model resolution finds the weights.
_LINKS = ("checkpoints", "snapshots", "final_model.zip", "best_model")
_COPIES = ("model_config.json", "metadata.json")
FIND_MODEL_FREE = ("switch", "uncertain", "faint", "opp-switch", "cure-skipped", "value_drop",
                   "low_value", "high_value")
_RR = ["--seeds", "4", "--alts", "1", "--worst", "1"]


def build_fixture(root: Path) -> None:
    if root.exists():
        raise SystemExit(f"{root} exists — the fixture is built once; remove it first")
    for run, prefixes in BATTLES.items():
        src, dst = ARCHIVE / run, root / run
        dst.mkdir(parents=True)
        for name in _COPIES:
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        for name in _LINKS:
            if (src / name).exists():
                os.symlink(src / name, dst / name)
        for pre in prefixes:
            step, opp, stem = pre.split("/")
            sd = dst / "eval_traces" / step
            (sd / opp).mkdir(parents=True, exist_ok=True)
            man = src / "eval_traces" / step / "eval_manifest.json"
            if man.exists() and not (sd / "eval_manifest.json").exists():
                shutil.copy2(man, sd / "eval_manifest.json")
            snap = src / "eval_traces" / step / "snapshot.zip"
            if snap.exists() and not (sd / "snapshot.zip").exists():
                os.symlink(snap, sd / "snapshot.zip")
            files = sorted((src / "eval_traces" / step / opp).glob(stem + "[._]*"))
            if not files:
                raise SystemExit(f"{run}/{pre}: no files")
            for f in files:
                shutil.copy2(f, sd / opp / f.name)
    print(f"fixture at {root}")


def commands(fx: Path, model: bool) -> list:
    """``(key, argv)`` for every capture, in a fixed order."""
    out = []
    for run, prefixes in BATTLES.items():
        r = str(fx / run)
        out += [
            (f"summary__{run}", ["summary", r]),
            (f"list__{run}", ["list", r]),
            (f"scan__{run}", ["scan", r, "--metric", "value_drop"]),
            (f"scan_td__{run}", ["scan", r, "--metric", "td_residual"]),
            (f"awareness__{run}", ["awareness", r, "--outcome", "all"]),
            (f"loops__{run}", ["loops", r]),
            (f"triage__{run}", ["triage", r]),
            (f"switch_vs_info__{run}", ["switch-vs-info", r]),
            (f"decision_table__{run}", ["decision-table", r]),
            (f"decision_table_rows__{run}", ["decision-table", r, "--limit", "100000"]),
            (f"calibration__{run}", ["--impl", "rust", "calibration", r, "--limit", "3", *_RR,
                                     "--concurrency", "1"]),
            (f"falsify_scan__{run}", ["--impl", "rust", "falsify-scan", r, "--limit", "3", *_RR,
                                      "--concurrency", "1"]),
        ]
        for pre in prefixes:
            b = str(fx / run / "eval_traces" / (pre + "_summary.json"))
            k = f"{run}__{pre.replace('/', '__')}"
            out += [(f"turns__{k}", ["turns", b]), (f"overview__{k}", ["overview", b])]
            out += [(f"find_{c}__{k}", ["find", b, c]) for c in FIND_MODEL_FREE]
            if not pre.split("/")[-1].startswith("win"):
                out.append((f"falsify__{k}", ["--impl", "rust", "falsify", b, *_RR]))
    if model:
        r = str(fx / MODEL_RUN)
        out += [(f"probe__{MODEL_RUN}", ["probe", r, "is_faster", "--max-decisions", "200"]),
                (f"history_saliency__{MODEL_RUN}", ["history-saliency", r, "--max-decisions", "60"])]
        for pre in BATTLES[MODEL_RUN][:4]:
            b = str(fx / MODEL_RUN / "eval_traces" / (pre + "_summary.json"))
            k = f"{MODEL_RUN}__{pre.replace('/', '__')}"
            out += [(f"find_disagree__{k}", ["find", b, "disagree"])]
            out += [(f"analyze_{i}__{k}", ["analyze", b, str(i)]) for i in (0, 3, 7)]
        # (battle, a move_selection invocation) — better-line / replay-counterfactual anchor there
        for pre, inv in zip(BATTLES[MODEL_RUN][:2], ("4", "3")):
            b = str(fx / MODEL_RUN / "eval_traces" / (pre + "_summary.json"))
            k = f"{MODEL_RUN}__{pre.replace('/', '__')}"
            out += [(f"lookahead__{k}", ["--impl", "rust", "lookahead", b, "--worst", "1"]),
                    (f"better_line__{k}", ["--impl", "rust", "better-line", b, inv, "--depth", "1",
                                           "--beam", "2", "--top-k", "2"])]
        b = str(fx / MODEL_RUN / "eval_traces" / (BATTLES[MODEL_RUN][0] + "_summary.json"))
        out.append((f"replay_counterfactual__{MODEL_RUN}", ["--impl", "rust", "replay-counterfactual",
                                                             b, "4", "6", "--rollouts", "1"]))
    return out


def run_all(fx: Path, out: Path, *, blocked: bool, model: bool, only: str = "", threads: int = 0) -> None:
    if blocked:
        from utils import poke_env_blocker
        poke_env_blocker.install()
    if threads:
        # a DECLARED intra-op thread count, the same for both sides: a CPU reduction may round with the
        # thread count, and on a contended box an oversubscribed default stalls torch for an hour
        import torch
        torch.set_num_threads(threads)
    import main.prober as _mp
    from main.prober import query

    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    # an `--only` re-capture MERGES into the directory's manifest (the other keys keep their rows)
    manifest = json.loads(mpath.read_text()) if (only and mpath.exists()) else {}
    manifest["__run__"] = {"src": str(Path(_mp.__file__).resolve().parents[2]), "threads": threads or None,
                           "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "blocked": blocked}
    parser = query._build_parser()
    for key, argv in commands(fx, model):
        if only and only not in key:
            continue
        t0 = time.time()
        try:
            res = query._run(parser.parse_args(argv))
            text = json.dumps(res, indent=2)
        except Exception as e:  # noqa: BLE001 — the CLI's own contract
            text = json.dumps({"error": f"{type(e).__name__}: {e}"}, indent=2)
        (out / f"{key}.json").write_text(text + "\n")
        manifest[key] = {"argv": argv, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "seconds": round(time.time() - t0, 2), "error": text.startswith('{\n  "error"')}
        print(f"{key}: {manifest[key]['seconds']}s{' ERROR' if manifest[key]['error'] else ''}",
              flush=True)
        mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")   # durable per capture
    if blocked:
        from utils import poke_env_blocker
        manifest["__poke_env_attempts__"] = [n for n, _ in poke_env_blocker.ATTEMPTS]
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def _first_diff(pa: Path, pb: Path) -> str:
    la = pa.read_text().splitlines() if pa.exists() else []
    lb = pb.read_text().splitlines() if pb.exists() else []
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return f"line {i + 1}: `{x.strip()[:90]}` → `{y.strip()[:90]}`"
    return f"length {len(la)} → {len(lb)} lines"


def diff(before: Path, after: Path, md: "Path | None" = None) -> int:
    """The identity table: per COMMAND, how many captures are byte-identical; each difference named by its
    first differing line. ``md`` also writes the table as markdown (the README's table)."""
    a = json.loads((before / "manifest.json").read_text())
    b = json.loads((after / "manifest.json").read_text())
    keys = sorted(k for k in set(a) | set(b) if not k.startswith("__"))
    same = [k for k in keys if k in a and k in b and a[k]["sha256"] == b[k]["sha256"]]
    differ = [k for k in keys if k not in same]
    by_cmd: dict = {}
    for k in keys:
        cmd = k.split("__")[0]
        cmd = "find_<criterion>" if cmd.startswith("find_") and cmd != "find_disagree" else cmd
        cmd = "analyze_<inv>" if cmd.startswith("analyze_") else cmd
        row = by_cmd.setdefault(cmd, [0, 0])
        row[0] += 1
        row[1] += k in same
    out = [f"{len(same)} identical / {len(differ)} differ / {len(keys)} captures", "",
           "| command | captures | byte-identical |", "|---|---|---|"]
    out += [f"| `{c}` | {n} | {s} |" for c, (n, s) in sorted(by_cmd.items())]
    if differ:
        out += ["", "| differing capture | first difference |", "|---|---|"]
        out += [f"| `{k}` | {_first_diff(before / f'{k}.json', after / f'{k}.json')} |" for k in differ]
    out += ["", f"poke-env import attempts in the AFTER process: {b.get('__poke_env_attempts__')}"]
    text = "\n".join(out)
    print(text)
    if md is not None:
        md.write_text(text + "\n")
    return 0 if not differ else 1


def main() -> int:
    # `--src DIR` runs another checkout's code (the BEFORE side: a detached worktree at the base commit)
    src = sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else str(TOP / "src")
    sys.path.insert(0, src)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fixture")
    f.add_argument("--dir", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fixture", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--blocked", action="store_true")
    r.add_argument("--no-model", action="store_true", help="skip the model-loading captures")
    r.add_argument("--only", default="", help="capture only keys containing this substring")
    r.add_argument("--src", default=None, help="the checkout's src/ to import (default: this one's)")
    r.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = torch's default)")
    d = sub.add_parser("diff")
    d.add_argument("before")
    d.add_argument("after")
    d.add_argument("--md", default=None, help="also write the identity table as markdown here")
    a = p.parse_args()
    if a.cmd == "fixture":
        build_fixture(Path(a.dir))
        return 0
    if a.cmd == "run":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        run_all(Path(a.fixture), Path(a.out), blocked=a.blocked, model=not a.no_model, only=a.only,
                threads=a.threads)
        return 0
    return diff(Path(a.before), Path(a.after), Path(a.md) if a.md else None)


if __name__ == "__main__":
    sys.exit(main())
