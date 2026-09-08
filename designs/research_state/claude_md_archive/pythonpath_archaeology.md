# `PYTHONPATH` vs the editable install — the archaeology

Lifted verbatim from the root `CLAUDE.md` on **2026-09-07**, when that section was cut from 67 lines
to ~40. The rules stayed in `CLAUDE.md`; this is the derivation — how the `.pth` ordering was proved
from first principles, the `environment.yml` history, and the `asyncio==4.0.0` removal.

---

### The `export PYTHONPATH` prefix is now OPTIONAL on this box — and still correct everywhere

`pyproject.toml` + `pip install -e .` (run once, from the **main checkout**) puts `src/` on the
import path permanently, so `import agents` / `import poke_env` resolve from any directory in any
shell with nothing exported. The incantation stays in the commands **throughout this file** for one
specific reason: **this file's reader is usually an agent, and an agent is usually in a worktree**,
where the export is not optional (below). It is also load-bearing in a fresh clone before
`pip install -e .`, in CI, in a container, in a bare venv. Both paths work; neither is retired.

🚨 **IN A GIT WORKTREE THE EXPORT IS STILL MANDATORY, and this is the one that will bite an
agent.** The install names ONE absolute path — the main checkout's `src/` — so a worktree that
runs `pytest` with no `PYTHONPATH` collects *its own* test files and imports *main's* code. Every
result is then about a tree you did not edit. `src/packaging_gate_test.py` fails loudly on exactly
this (it compares the resolved `agents` against its own `__file__`), so it is caught rather than
believed — but the fix is to export, every time, in a worktree. Optional means optional **in the
main checkout**.

**The `Run:` headers in run-directly scripts no longer carry the export** (swept 2026-08-22, one
reviewed pass over ~65 files). A fuzz test's / benchmark's / harness's header is now the bare
command plus one standard closing line — `(in a linked worktree, first: export
PYTHONPATH=$PYTHONPATH:src)` — so the block above it is copy-pasteable as-is in the main checkout
and the worktree case is stated where the reader already is. The parenthetical is prose, never a
command; do not "tidy" it into the block.

**The two mechanisms coexist, and the ORDER between them is load-bearing.** `PYTHONPATH` entries
land in `sys.path` *before* site-packages; an editable install's `.pth` lands *after*. That
asymmetry is what lets the launcher pin a resumed run to its checkpoint's commit
(`PYTHONPATH=<pinned worktree>/src` beats the install, so an old run cannot silently resume on
current HEAD). Both orderings are re-proved on every test run against real `.pth` files by
**`src/packaging_gate_test.py`**, which also fails if `child.py`'s PYTHONPATH export is ever
"cleaned up" now that an install exists.

`pyproject.toml` **declares no dependencies, deliberately**: `environment.yml` is the single owner
of what is installed, so `pip install -e .` writes a `.pth` plus a `dist-info` and can never
resolve or replace a pinned wheel in a working env. Install from the **main checkout only** — a
`.pth` made inside a git worktree points at a directory that later gets deleted, and Python skips
a missing `.pth` entry in silence (guarded, with the fix in the message).

**That absolute path is THIS box's env, not a requirement of the code.** Nothing in the tree
hardcodes an interpreter any more: every process the project spawns — the launcher's training
child, the eval workers, the snapshot-ladder, the search-teacher workers — runs under
**`sys.executable`**, i.e. whatever interpreter started the parent. So on another machine, or in
another env name, `conda activate gen3ai_stable && python <script>` is enough; the path above is
just how this box's commands are written so they work from any shell. The launcher additionally
takes **`$GEN3AI_PYTHON`** to pin its child's interpreter explicitly — see
`src/main/launcher/CLAUDE.md` → Which interpreter the child runs.

**Two things `environment.yml` says that are load-bearing, and neither is a preference:**

- The pip block opens with `--extra-index-url https://download.pytorch.org/whl/cu121`. The torch
  pins are LOCAL-VERSION builds (`2.5.1+cu121`) and local versions are **not published on PyPI**,
  so without that line `conda env create` fails outright on any fresh machine at the very first
  step. Verified 2026-08-22 (PyPI offers `2.5.1`, never `2.5.1+cu121`).
- `poke-env` is **deliberately absent**. This repo vendors the fork at `src/poke_env/` (57 modules
  to upstream 0.15.0's 45) and the battle layer depends on the additions. A second installed copy
  makes `import poke_env` depend on `sys.path` ORDER — and the failure is **silent**: upstream
  imports cleanly and behaves subtly differently. Under PYTHONPATH the fork wins; under an
  editable install's `.pth`, which lands *after* site-packages, **upstream wins** — reproduced
  from first principles and kept executable by `src/packaging_gate_test.py`. The live
  `gen3ai_stable` env's PyPI copy was uninstalled on 2026-08-22, so there is no longer a second
  copy and no ordering to lose. `src/poke_env_fork_gate_test.py` is the permanent guard
  (unmarked, 0.03 s, in every tier); it also fails if the pin is re-added to `environment.yml`.
  `asyncio==4.0.0` was removed for the same class of reason — a deprecated backport of a stdlib
  module.

---
