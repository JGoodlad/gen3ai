# CLAUDE.md — `src/main/tui/` (shared Textual base)

A deliberately **thin** shared base for the project's Textual UIs. It has **two
consumers** — the prober (`src/main/prober/`) and the launcher's UI
(`src/main/launcher/app.py`, run via `python -m main.launcher`). It stays minimal:
only what both genuinely share.

## What's here

- **`app_base.py` — `Gen3App(App)`**: shared chrome (`Header` + `Footer`), the
  `q` quit binding, two copy paths (`super+c` and `v` copy mode — see **Copying
  text** below), and the shared theme loaded via the absolute `THEME_PATH`.
  Subclasses implement `compose_body()` (the chrome wraps it) and append their
  own `.tcss` with `CSS_PATH = [str(THEME_PATH), "my_app.tcss"]` (the first entry
  is absolute so it resolves to *this* package's theme regardless of where the
  subclass lives; the second resolves relative to the subclass module).
- **`theme.tcss`**: the Gen3AI palette + cross-app layout primitives. Per-app
  layout belongs in that app's own `.tcss`, not here.
- **`colors.py` — `gradient_color(t)`**: red→yellow→green hex used for both UIs'
  gradient cells (launcher win-rate/reward cells; prober faithfulness-drift
  cells). Plus the named palette constants.

## Copying text

A Textual app puts the terminal in application mode with mouse tracking on, so the
terminal never sees a click-drag and its native select-and-copy never fires. Two
shared paths, both in `Gen3App` (inherited by the launcher + prober):

- **`super+c` → `screen.copy_text`** — macOS ⌘C copying the Textual-native mouse
  selection. Needs a terminal that **forwards ⌘C** (kitty keyboard protocol) **and**
  honours **OSC 52** (the clipboard-write escape): kitty, Ghostty, WezTerm, iTerm2.
  Dead on **Terminal.app**, which does neither — the keystroke never reaches the app
  and the clipboard write is ignored.
- **`v` → copy mode** (`action_toggle_copy_mode` / `copy_mode` reactive) — the
  **portable** path that works even on Terminal.app, because it uses zero app/clipboard
  escapes. It calls the driver's `_disable_mouse_support()` (mouse handed back to the
  terminal → native click-drag selection returns) and **freezes live updates** via the
  `_pause_live_updates()` / `_resume_live_updates()` hooks (so a 2 Hz repaint can't wipe
  the selection mid-drag). You then select + copy with the **terminal's own** mechanism
  (e.g. ⌘C in Terminal.app); the same `v` **or `Escape`** resumes (`action_exit_copy_mode`,
  a no-op when copy mode is off so it never steals Escape from anything else). The base hooks
  are no-ops (static apps like the prober need no freeze); the launcher overrides them to
  pause/resume its `set_interval(0.5, …)` refresh `Timer`.

## What is intentionally NOT shared

The launcher's runtime — IPC, child subprocess, SIGTERM/restart, worktree
pinning, the lock+snapshot `LauncherState` — is launcher-specific and stays in
`src/main/launcher/`. Its UI (`app.py`, the second consumer of this base) uses
exactly the genuinely-shared pieces here (`Gen3App`, the theme, `gradient_color`,
the key-hint `Footer`) and keeps its own runtime/loop in `src/main/launcher/run.py`.

## Tests

`base_test.py` — `gradient_color` endpoints, a `Gen3App` mount smoke test, the
`super+c` copy binding (present + harmless with no selection), and `v` copy mode
(toggles + drives the freeze hooks). `launcher_app_test.py` additionally guards that
the launcher's fresh `BINDINGS` don't shadow `super+c`, and that `v` pauses the real
refresh `Timer` (and is handled app-locally, not routed to the supervisor).

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python3 -m pytest src/main/tui -q
```
