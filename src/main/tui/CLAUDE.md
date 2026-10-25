# CLAUDE.md — `src/main/tui/` (shared Textual base)

A thin base (`Gen3App`, `theme.tcss`, `colors.gradient_color`) for the launcher's Textual UI
(`src/main/launcher/app.py`), its one consumer. Keep it thin: only presentation goes here; the
launcher's runtime stays in `src/main/launcher/`.

## Copying text

`super+c` copies the Textual selection (needs a terminal that forwards ⌘C and honours OSC 52). `v`
copy mode is the portable path: it hands the mouse back to the terminal and calls
`_pause_live_updates()` / `_resume_live_updates()`. The base hooks are no-ops, so an app with a live
refresh timer must override them, or a repaint wipes the selection. A subclass's `BINDINGS` must
never shadow `super+c` (`src/main/launcher_app_test.py`).
