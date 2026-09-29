"""Pins for `utils.headless_chrome` that need no browser (unmarked — they run in every tier).

The browser tier itself is where the driver is EXERCISED; these pin the one line whose loss would
not fail anything there, only make it ~40x slower: `--password-store=basic`. Measured 2026-09-29,
without it every page that touches chrome's network stack pays a 25 s D-Bus keyring timeout
(25.5 s → 0.65 s per page), which is what the tier's old "~25 s chrome cold start" really was.
A timing assertion would be contention-sensitive, so the pin is on the flag and on every browser
launch in the tier going through `BASE_FLAGS`.
"""

from __future__ import annotations

from utils import headless_chrome
from utils.paths import src_path


def test_the_keyring_flag_is_in_the_base_flags():
    assert "--password-store=basic" in headless_chrome.BASE_FLAGS


def test_every_browser_test_launches_through_the_base_flags():
    for rel in ("main/prober/web/render_integration_test.py",
                "agents/model/build_arch_viewer_render_integration_test.py"):
        text = src_path(*rel.split("/")).read_text()
        assert "headless_chrome" in text, f"{rel} no longer goes through utils.headless_chrome"
        assert '"--headless"' not in text, (
            f"{rel} spells its own chrome argv — route it through BASE_FLAGS so the keyring "
            "flag cannot be dropped")
