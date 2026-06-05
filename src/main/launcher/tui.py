"""Back-compat alias: ``python -m main.launcher.tui`` → the launcher.

The launcher's only UI is now Textual, so ``python -m main.launcher`` is the canonical
command; this module is kept so existing scripts/muscle-memory using ``.tui`` still work.
"""

from main.launcher import main

if __name__ == "__main__":
    main()
