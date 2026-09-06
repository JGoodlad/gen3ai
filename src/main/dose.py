"""What DOSE did this run actually train at? — offline, model-free, no torch.

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.dose models/<run> [models/<run2> ...]
python -m main.dose models/a models/b --reference models/<baseline> --md
```
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

WHY. `--lr` is INERT on a resume: `main.train.model_build`'s resume path restores the checkpoint's
optimizer LR and prints `(arg --lr=… ignored on resume)`, so a distillation fold inherits whatever
rate the PARENT's KL controller had annealed to. Three folds launched with the same `--lr` therefore
ran at three different rates, and the quantity that predicts a fold's collateral is not the rate
alone but the DOSE:

    updates_per_env_step = n_epochs / (batch_size * grad_accum_steps)
    dose_rate            = lr_median * updates_per_env_step

`--fork-lr` makes that chosen and `metadata.json`'s `dose` block makes it recorded — for runs from
here on. This tool answers the same question for every run ALREADY on disk, from what those runs
already wrote down.

WHERE THE NUMBERS COME FROM, in preference order, and why the first one wins:

  1. the per-checkpoint SIDECARS (`<run>/checkpoints/*.json`) — one `lr` per checkpoint, so the
     MEDIAN is over the run's own recorded trajectory rather than over its endpoints;
  2. `metadata.json`'s `snapshot_history` — the same rows, but CAPPED (a long run keeps ~15 while
     its sidecars keep every un-groomed checkpoint), so it is the fallback;
  3. `metadata.json`'s top-level `current_lr` — one point, reported as such.

The `steps` column is `metadata.json`'s top-level `num_timesteps` (else the last sidecar's) — how
far the run trained, which is the exposure the dose RATE was applied for. A run that predates the
key shows `—`: this tool never opens a checkpoint zip, so unknown stays unknown.

Everything else (`batch_size`, `grad_accum_steps`, `n_epochs`) comes from the SAME rows, so the
dose is computed from one consistent record rather than from a shape read out of `cli_args` and an
LR read out of somewhere else. When a run CHANGED shape mid-flight the table says so
(`shape_stable: false`) and uses the LAST row's shape — a run whose effective batch moved has no
single dose, and averaging one is worse than flagging it.

Torch is never imported and no checkpoint zip is opened, so this works on a run whose architecture
has drifted past the current code — which is most of `models/`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

from agents.training.dose import effective_batch, updates_per_env_step

#: The v8 fold whose dose every gen-era fold is compared against (ledger M7). Used only when it
#: exists under the resolved models dir; a missing reference is omitted, never faked.
DEFAULT_REFERENCE = "ai_v8_14_distill3_0725"


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            obj = json.load(f)
    except Exception:  # noqa: BLE001 — absent, truncated, or not ours
        return None
    return obj if isinstance(obj, dict) else None


def _row_step(row: Dict[str, Any], base: str) -> int:
    """The step a sidecar sits at: its RECORDED `num_timesteps`, else the filename's own number.

    The recorded key wins because it is what the save path wrote down
    (`agents.model.snapshot`), while the filename is an inference that only holds for the
    `checkpoint_<N>_steps` convention — `final_model.json` and `best_model.json` carry no number
    at all and sort first at -1, unchanged.
    """
    recorded = row.get("num_timesteps")
    if isinstance(recorded, (int, float)):
        return int(recorded)
    step = -1
    for part in base.replace(".json", "").split("_"):
        if part.isdigit():
            step = int(part)
    return step


def _sidecar_rows(run_dir: str) -> List[Dict[str, Any]]:
    """Every per-checkpoint sidecar, oldest first by recorded step.

    Sorted by STEP rather than by name so `checkpoint_9_steps.json` cannot sort after
    `checkpoint_10_steps.json`; a row that states neither a `num_timesteps` nor a parseable name
    sorts last by name.
    """
    rows = []
    for path in glob.glob(os.path.join(run_dir, "checkpoints", "*.json")):
        row = _read_json(path)
        if row is None:
            continue
        base = os.path.basename(path)
        rows.append((_row_step(row, base), base, row))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows]


def _history_rows(run_dir: str) -> List[Dict[str, Any]]:
    meta = _read_json(os.path.join(run_dir, "metadata.json")) or {}
    hist = meta.get("snapshot_history") or {}
    if not isinstance(hist, dict):
        return []
    return [v for _, v in sorted(hist.items()) if isinstance(v, dict)]


def _shape(row: Dict[str, Any]) -> Optional[tuple]:
    """`(batch_size, grad_accum_steps, n_epochs)` if the row states all three, else None."""
    bs, ga, ne = row.get("batch_size"), row.get("grad_accum_steps", 1), row.get("n_epochs")
    if bs is None or ne is None:
        return None
    try:
        return (int(bs), max(1, int(ga or 1)), int(ne))
    except (TypeError, ValueError):
        return None


def read_run(run_dir: str) -> Dict[str, Any]:
    """Everything `main.dose` knows about one run. Pure over the filesystem; unit-tested directly.

    `error` is set (and every numeric field left None) when the run states no usable row — a run
    that recorded nothing must read as UNKNOWN, never as a dose of 0.
    """
    out: Dict[str, Any] = {
        "run": os.path.basename(os.path.normpath(run_dir)), "dir": run_dir,
        "source": None, "n_lr": 0, "lr_median": None, "lr_min": None, "lr_max": None,
        "batch_size": None, "grad_accum_steps": None, "effective_batch": None, "n_epochs": None,
        "updates_per_env_step": None, "dose_rate": None, "shape_stable": None,
        "recorded_dose": None, "fork_lr": None, "lr_frozen": None, "num_timesteps": None,
        "error": None,
    }
    if not os.path.isdir(run_dir):
        out["error"] = "no such run directory"
        return out

    rows = _sidecar_rows(run_dir)
    source = "sidecars"
    if not rows:
        rows, source = _history_rows(run_dir), "snapshot_history"
    meta = _read_json(os.path.join(run_dir, "metadata.json")) or {}
    if not rows and meta:
        # Last resort: the run-level record is ONE point. Reported as such (`n_lr` = 1), because a
        # median over one sample and a median over forty are different claims about the same run.
        rows, source = [meta], "metadata"

    # The RECORDED dose block, when the run is new enough to have written one. Reported beside the
    # derived numbers rather than instead of them: a recorded value and a derived one that disagree
    # is a finding, and only showing one of them hides it.
    dose = meta.get("dose") if isinstance(meta.get("dose"), dict) else None
    if dose:
        out["recorded_dose"] = dose.get("dose_rate_now")
        out["fork_lr"] = dose.get("fork_lr")
        out["lr_frozen"] = dose.get("lr_frozen")
    # HOW FAR THE RUN TRAINED — the run-level `num_timesteps`, else the LAST row's. A dose is a
    # rate, so the steps it was applied for is the other half of the exposure; a run that recorded
    # neither reads None => `—`, never 0.
    steps = meta.get("num_timesteps")
    if not isinstance(steps, (int, float)) and rows:
        steps = rows[-1].get("num_timesteps")
    if isinstance(steps, (int, float)):
        out["num_timesteps"] = int(steps)

    lrs = [float(r["lr"]) for r in rows
           if isinstance(r.get("lr"), (int, float)) and float(r["lr"]) > 0]
    if not lrs:
        lrs = [float(r["current_lr"]) for r in rows
               if isinstance(r.get("current_lr"), (int, float)) and float(r["current_lr"]) > 0]
    shapes = [sh for sh in (_shape(r) for r in rows) if sh is not None]
    if not lrs or not shapes:
        out["error"] = f"no lr/shape rows recorded ({source})"
        out["source"] = source
        return out

    bs, ga, ne = shapes[-1]        # the LAST row's shape — see the module docstring
    ups = updates_per_env_step(batch_size=bs, grad_accum_steps=ga, n_epochs=ne)
    med = statistics.median(lrs)
    out.update({
        "source": source, "n_lr": len(lrs), "lr_median": med,
        "lr_min": min(lrs), "lr_max": max(lrs),
        "batch_size": bs, "grad_accum_steps": ga, "effective_batch": effective_batch(bs, ga),
        "n_epochs": ne, "updates_per_env_step": ups, "dose_rate": med * ups,
        "shape_stable": len(set(shapes)) == 1,
    })
    return out


def _fmt(value: Any, spec: str = "") -> str:
    return "—" if value is None else (format(value, spec) if spec else str(value))


def render(rows: List[Dict[str, Any]], reference: Optional[Dict[str, Any]],
           markdown: bool = False) -> str:
    """The table. `reference` is a `read_run` result; its own row shows a ratio of 1.00x."""
    ref_dose = (reference or {}).get("dose_rate")
    header = ["run", "steps", "eff.batch", "epochs", "lr_median", "updates/step", "dose_rate",
              "vs ref"]
    body = []
    for r in rows:
        ratio = ("—" if (ref_dose in (None, 0) or r.get("dose_rate") is None)
                 else f"{r['dose_rate'] / ref_dose:.2f}x")
        flags = []
        if r.get("shape_stable") is False:
            flags.append("SHAPE MOVED")
        if r.get("lr_frozen"):
            flags.append("FROZEN")
        if r.get("fork_lr") is not None:
            flags.append(f"pinned {r['fork_lr']:.2e}")
        if r.get("error"):
            flags.append(r["error"])
        name = r["run"] + (f"  [{'; '.join(flags)}]" if flags else "")
        body.append([
            name,
            _fmt(r.get("num_timesteps"), ","),
            _fmt(r.get("effective_batch"), ","),
            _fmt(r.get("n_epochs")),
            _fmt(r.get("lr_median"), ".4g"),
            _fmt(r.get("updates_per_env_step"), ".4g"),
            _fmt(r.get("dose_rate"), ".4g"),
            ratio,
        ])
    if markdown:
        out = ["| " + " | ".join(header) + " |",
               "|" + "|".join("---" for _ in header) + "|"]
        out += ["| " + " | ".join(c) for c in ([*b, ""] for b in body)]
        return "\n".join(out)
    widths = [max(len(header[i]), *(len(b[i]) for b in body)) if body else len(header[i])
              for i in range(len(header))]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.ljust(w) for c, w in zip(b, widths)) for b in body]
    return "\n".join(lines)


def _resolve(name: str) -> str:
    """A run dir, or a bare run NAME resolved against the main checkout's `models/`.

    `models/` is not committed and exists only in the MAIN checkout, so a worktree must reach across
    (`utils.paths.main_models_dir`). A name that resolves nowhere is returned unchanged and
    `read_run` reports it as missing — a resolver that guessed would be worse than one that says so.
    """
    if os.path.isdir(name):
        return name
    from utils.paths import main_models_dir
    root = main_models_dir()
    if root is not None and os.path.isdir(os.path.join(root, name)):
        return os.path.join(root, name)
    return name


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m main.dose", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs (or bare run names under models/)")
    ap.add_argument("--reference", default=None,
                    help=f"run to take the ratio against (default: {DEFAULT_REFERENCE} if present; "
                         f"'none' to omit the column)")
    ap.add_argument("--md", action="store_true", help="emit a markdown table")
    ap.add_argument("--json", action="store_true", help="emit the raw rows as JSON")
    args = ap.parse_args(argv)

    rows = [read_run(_resolve(r)) for r in args.runs]

    reference = None
    if args.reference != "none":
        ref_name = args.reference or DEFAULT_REFERENCE
        ref_dir = _resolve(ref_name)
        if os.path.isdir(ref_dir):
            reference = read_run(ref_dir)
        elif args.reference:
            print(f"[dose] reference {ref_name!r} not found — omitting the ratio column",
                  file=sys.stderr)

    if args.json:
        print(json.dumps({"runs": rows, "reference": reference}, indent=2))
        return 0
    if reference is not None:
        print(f"reference: {reference['run']}  dose_rate={_fmt(reference.get('dose_rate'), '.4g')}")
    print(render(rows, reference, markdown=args.md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
