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


async def test_super_c_copy_binding():
    """⌘C (super+c) copies the Textual selection; harmless when nothing is selected."""
    app = Gen3App()
    keys = {b.key for b in app.BINDINGS}
    assert "super+c" in keys
    async with app.run_test() as pilot:
        await pilot.press("super+c")  # no selection → SkipAction → must not crash/quit
        await pilot.pause()
        assert app.is_running


async def test_copy_mode_toggles_and_calls_freeze_hooks():
    """`v` toggles copy mode, driving the live-update freeze hooks; same key resumes."""
    paused, resumed = [], []
    app = Gen3App()
    app._pause_live_updates = lambda: paused.append(True)   # type: ignore[method-assign]
    app._resume_live_updates = lambda: resumed.append(True)  # type: ignore[method-assign]
    assert "v" in {b.key for b in app.BINDINGS}
    async with app.run_test() as pilot:
        await pilot.press("v")               # enter copy mode
        await pilot.pause()
        assert app.copy_mode is True
        assert paused == [True] and resumed == []
        await pilot.press("v")               # same key resumes
        await pilot.pause()
        assert app.copy_mode is False
        assert resumed == [True]
        assert app.is_running
