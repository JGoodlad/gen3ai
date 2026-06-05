# CLAUDE.md — `src/main/tui/` (shared Textual base)

A deliberately **thin** shared base for the project's Textual UIs. It has **two
consumers** — the prober (`src/main/prober/`) and the launcher's UI
(`src/main/launcher/app.py`, run via `python -m main.launcher`). It stays minimal:
only what both genuinely share.

## What's here

- **`app_base.py` — `Gen3App(App)`**: shared chrome (`Header` + `Footer`), the
  `q` quit binding, and the shared theme loaded via the absolute `THEME_PATH`.
  Subclasses implement `compose_body()` (the chrome wraps it) and append their
  own `.tcss` with `CSS_PATH = [str(THEME_PATH), "my_app.tcss"]` (the first entry
  is absolute so it resolves to *this* package's theme regardless of where the
  subclass lives; the second resolves relative to the subclass module).
- **`theme.tcss`**: the Gen3AI palette + cross-app layout primitives. Per-app
  layout belongs in that app's own `.tcss`, not here.
- **`colors.py` — `gradient_color(t)`**: red→yellow→green hex used for both UIs'
  gradient cells (launcher win-rate/reward cells; prober faithfulness-drift
  cells). Plus the named palette constants.

## What is intentionally NOT shared

The launcher's runtime — IPC, child subprocess, SIGTERM/restart, worktree
pinning, the lock+snapshot `LauncherState` — is launcher-specific and stays in
`src/main/launcher/`. Its UI (`app.py`, the second consumer of this base) uses
exactly the genuinely-shared pieces here (`Gen3App`, the theme, `gradient_color`,
the key-hint `Footer`) and keeps its own runtime/loop in `src/main/launcher/run.py`.

## Tests

`base_test.py` — `gradient_color` endpoints + a `Gen3App` mount smoke test.

```bash
export PYTHONPATH=$PYTHONPATH:src && python3 -m pytest src/main/tui -q
```
