# CLAUDE.md — `src/main/tui/` (shared Textual base)

A deliberately **thin** shared base for the project's Textual UIs. Today it has
exactly one consumer — the prober (`src/main/prober/`). It exists now so a future
**launcher Textual port** has a ready seam, without prematurely abstracting more
than is genuinely shared.

## What's here

- **`app_base.py` — `Gen3App(App)`**: shared chrome (`Header` + `Footer`), the
  `q` quit binding, and the shared theme loaded via the absolute `THEME_PATH`.
  Subclasses implement `compose_body()` (the chrome wraps it) and append their
  own `.tcss` with `CSS_PATH = [str(THEME_PATH), "my_app.tcss"]` (the first entry
  is absolute so it resolves to *this* package's theme regardless of where the
  subclass lives; the second resolves relative to the subclass module).
- **`theme.tcss`**: the Gen3AI palette + cross-app layout primitives. Per-app
  layout belongs in that app's own `.tcss`, not here.
- **`colors.py` — `gradient_color(t)`**: red→yellow→green hex, lifted verbatim
  from `launcher/ui.py`'s `_gradient_color` so both UIs draw the same gradient
  (launcher win-rate cells; prober faithfulness-drift cells). Plus the named
  palette constants.

## What is intentionally NOT shared

The launcher's runtime — IPC, child subprocess, SIGTERM/restart, worktree
pinning, the lock+snapshot `LauncherState` — is launcher-specific and stays in
`src/main/launcher/`. It is also still **Rich**, not Textual; porting it is a
separate, larger effort. When that port happens, the genuinely-shared pieces
(an app base, the theme, the gradient helper, a key-hint footer) are already
here; extend this package then, not before.

## Tests

`base_test.py` — `gradient_color` endpoints + a `Gen3App` mount smoke test.

```bash
export PYTHONPATH=$PYTHONPATH:src && python3 -m pytest src/main/tui -q
```
