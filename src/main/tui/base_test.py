"""Smoke tests for the shared Textual base (Gen3App + colors)."""

from textual.widgets import Footer, Header

from main.tui import Gen3App, gradient_color


def test_gradient_color_endpoints_and_midpoint():
    assert gradient_color(0.0) == "#ff0000"   # full red
    assert gradient_color(1.0) == "#00ff00"   # full green
    assert gradient_color(0.5) == "#ffff00"   # yellow
    # clamps out-of-range
    assert gradient_color(-1.0) == "#ff0000"
    assert gradient_color(2.0) == "#00ff00"


async def test_base_app_mounts_chrome():
    app = Gen3App()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(Header) is not None
        assert app.query_one(Footer) is not None
