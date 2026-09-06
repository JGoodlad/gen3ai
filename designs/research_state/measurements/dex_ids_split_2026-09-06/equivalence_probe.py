#!/usr/bin/env python3
"""Did `gen3_dex_ids_split_v1` change anything? — the before/after equivalence probe.

Run:
    python designs/research_state/measurements/dex_ids_split_2026-09-06/equivalence_probe.py \
        --baseline <the pre-cut commit>
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

The cut moved `build_move_prior_logits`, `sanitize_historical_move_floor` and their floor
constants, plus `build_species_cooccur_prior`, out of `damage_tables.py` into `belief_tables.py`,
and the dex-identity facts those builders share with the op's physics (`HIDDEN_POWER_NUM`,
`_belief_num`, `_hp_typed_nums`, `build_species_usage_prior`) into a new neutral `dex_ids.py`.

A refactor's equivalence is a ONE-TIME measurement — the permanent invariants live in
`belief_tables_test.py` — so this script exists to be re-runnable rather than collected, and it
answers two questions:

  1. **AST** — is every moved definition's EXECUTABLE body identical to the one at the BASELINE
     commit? Docstrings are stripped before comparison (each moved definition gains one origin
     line, the only intended text change) and the comparison is `ast.dump` on the stripped tree,
     so a reflowed expression or a renamed local WOULD show up.
  2. **THE BUILT MODEL** — does the literal production-config extractor come out identical? The
     baseline tree is materialised with `git archive`, both arms are built in their OWN
     subprocess under a fixed torch seed, and every `state_dict` entry and every registered
     BUFFER is compared by sha256 over its raw bytes — a checkable digest rather than a boolean.

The seed matters, and its absence is itself informative: the previous round of this cut also
measured the UNSEEDED arms, where 107 `state_dict` keys differ (random init) and ZERO buffers do —
which is the point, since every relocated table is a `persistent=False` data-derived buffer and so
contributes no `state_dict` key at all.

Pass `--baseline` the commit BEFORE the cut (this tree is the "after" arm, read from disk, so it
works on an uncommitted working tree).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

# The names this cut moved, and where each one now lives.
MOVED = {
    "agents.model.dex_ids": [
        "HIDDEN_POWER_NUM", "_belief_num", "_hp_typed_nums",
        "_USAGE_PRIOR_FLOOR", "build_species_usage_prior",
    ],
    "agents.model.belief_tables": [
        "_PRIOR_FLOOR", "_ILLEGAL_PROB", "_MIN_PRIOR_FLOOR",
        "sanitize_historical_move_floor", "build_move_prior_logits",
        "_SPECIES_PRIOR_FLOOR", "_SPECIES_CLAUSE_PROB", "SPECIES_CLAUSE_LOGIT",
        "_COOCCUR_LIFT_CLAMP", "build_species_cooccur_prior",
    ],
}
_MOD_PATH = {"agents.model.dex_ids": "src/agents/model/dex_ids.py",
             "agents.model.belief_tables": "src/agents/model/belief_tables.py",
             "agents.model.damage_tables": "src/agents/model/damage_tables.py"}

# The child that builds one arm's extractor and prints a digest of it. Kept as source text so the
# BASELINE tree runs nothing of ours except its own package.
_CHILD = r'''
import hashlib, json, sys
import torch
torch.manual_seed(1234567)
from agents.model.delivery_graph import build_extractor
fe, _cfg, _layout = build_extractor()
def dig(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
out = {
    "state_dict": {k: dig(v) for k, v in fe.state_dict().items()},
    "buffers":    {k: dig(v) for k, v in fe.named_buffers()},
    "module_of":  {k: type(m).__name__ for k, m in fe.named_modules()},
}
json.dump(out, sys.stdout)
'''


def _strip_docstrings(node: ast.AST) -> ast.AST:
    """Delete every docstring Expr, so the comparison is of EXECUTABLE code only."""
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(n, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                del body[0]
    return node


def _defs(path: Path) -> dict:
    """`{name: executable-AST dump}` for every top-level def and simple assignment in `path`."""
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.dump(_strip_docstrings(node))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = ast.dump(node.value)
    return out


def _materialise_baseline(rev: str, dest: Path) -> None:
    tar = subprocess.run(["git", "archive", rev], cwd=REPO, check=True,
                         stdout=subprocess.PIPE).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar, check=True)


def _build_arm(src_root: Path) -> dict:
    """Build the production extractor under `src_root`, in a fresh clean-PYTHONPATH subprocess."""
    r = subprocess.run([sys.executable, "-c", _CHILD], cwd=str(src_root.parent),
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(src_root),
                            "HOME": str(Path.home()), "OMP_NUM_THREADS": "1"},
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"arm at {src_root} failed:\n{r.stderr[-4000:]}")
    return json.loads(r.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description="dex_ids-split equivalence probe")
    ap.add_argument("--baseline", default="HEAD", help="commit to compare against (default HEAD)")
    args = ap.parse_args()
    ok = True

    with tempfile.TemporaryDirectory(prefix="dex_ids_baseline_") as td:
        base = Path(td)
        _materialise_baseline(args.baseline, base)
        print(f"baseline tree: {args.baseline} -> {base}")
        print(f"this tree:     {REPO}")

        # ---------------------------------------------------------- 1. executable-AST identity
        base_dt = _defs(base / _MOD_PATH["agents.model.damage_tables"])
        print("\n=== 1. executable-AST identity of every moved name (docstrings stripped) ===")
        for mod, names in MOVED.items():
            here = _defs(REPO / _MOD_PATH[mod])
            for name in names:
                if name not in base_dt:
                    print(f"  ?? {name:34s} not found in the baseline damage_tables")
                    ok = False
                elif name not in here:
                    print(f"  !! {name:34s} MISSING from {mod}")
                    ok = False
                else:
                    same = here[name] == base_dt[name]
                    ok &= same
                    print(f"  {'==' if same else '!!'} {name:34s} {mod}")

        # ---------------------------------------------------------- 2. the built model
        print("\n=== 2. the production-config extractor, seeded, baseline vs this tree ===")
        a = _build_arm(base / "src")
        b = _build_arm(REPO / "src")
        for field in ("state_dict", "buffers"):
            ka, kb = set(a[field]), set(b[field])
            if ka != kb:
                ok = False
                print(f"  !! {field}: key sets differ"
                      f" (+{sorted(kb - ka)[:6]} -{sorted(ka - kb)[:6]})")
                continue
            diff = [k for k in sorted(ka) if a[field][k] != b[field][k]]
            ok &= not diff
            print(f"  {'==' if not diff else '!!'} {field}: {len(ka)} entries, "
                  f"{len(diff)} differing{'' if not diff else ' -> ' + str(diff[:8])}")
        mods_same = a["module_of"] == b["module_of"]
        ok &= mods_same
        print(f"  {'==' if mods_same else '!!'} module tree: {len(b['module_of'])} modules")
        roll = hashlib.sha256(
            json.dumps({"sd": b["state_dict"], "bufs": b["buffers"]}, sort_keys=True).encode()
        ).hexdigest()
        print(f"  digest over this tree's state_dict+buffers: {roll}")

    print("\nVERDICT:", "IDENTICAL" if ok else "DIVERGED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
