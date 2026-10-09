"""Do the pages actually RENDER? — a headless-browser gate on the web views.

`app_test.py` asserts properties of the emitted TEXT and never executes a line of the JavaScript.
That gap is not theoretical here: every chart on these pages is a Vega-Lite spec that Python emits
and a browser compiles, and a spec can be perfectly well-formed JSON, embed without throwing, and
plot **nothing** — a mistyped field name produces an empty chart and a green text suite. Likewise
the scan/triage/battles tables do not exist in the page source at all; they arrive through an HTMX
request that the text tests cannot make.

So the page publishes a small machine-readable record at the end of init (`static/app.js` →
`document.body.dataset`) and this test reads it back out of a real browser:

  * `ready`        the bootstrap ran to completion (any throw leaves it unset)
  * `htmx`/`vega`  the vendored bundles defined their globals — "missing" if a <script> 404'd
  * `charts`       how many specs the server embedded
  * `chart-marks`  how many mark elements Vega ACTUALLY DREW. This is the value no text check can
                   see, and the one that goes to zero for a chart that compiles but plots nothing
  * `rows`         data rows present in the DOM
  * `swaps`        completed HTMX swaps — on /scan and /triage the entire table AND its chart
                   arrive this way, so a non-zero count with marks > 0 proves the whole round trip

RESPONSIVE, MEASURED RATHER THAN ASSERTED. The record also carries what the laid-out page measured
of itself — `vw` / `docw` / `narrow` / `headerh` / `ctlfont` / `overflowby` + `overflowwhat` /
`scrollers` + `scrollingwrappers` — and the layout tests read those at two viewports. Checking the
CSS instead would not do: a media query can be present and still lose on specificity (the arch
viewer's bottom-sheet close bar was defined after the query that showed it, so `display:none` won
and the sheet had no way out). Only the laid-out page knows.

NO NETWORK, DELIBERATELY. The arch viewer's render test skips itself when its CDN is unreachable,
which turns the strongest gate in that suite into a no-op offline — the exact trap this package
vendors its JavaScript to avoid. So chrome is launched with every non-loopback host mapped to a
dead address: if any asset were still remote, `vega` would read "missing" and the assertions would
FAIL rather than quietly skip.

It skips, naming which, only when there is genuinely no browser on the box.

CONTENTION. Every wall-clock bound here goes through `scale_timeout`, per the project's timeout
doctrine: a page that settles in ~0.1 s idle takes several times that beside a live trainer, and a
raw constant would report the load average as a rendering failure. Nothing here ASSERTS on time —
the measurements are layout and mark counts, which contention cannot move — so scaling the bounds
costs no signal. (Idle box ⇒ factor exactly 1.0 ⇒ no change.)

HOW IT DRIVES THE BROWSER (2026-09-29). One headless chrome per module over the DevTools pipe
(`utils/headless_chrome.py`), a fresh browser CONTEXT per probe (per-test isolation of cookies,
storage and cache), and a wait on the page's own SETTLED signal (`data-busy` == "0", set by
`static/app.js`) instead of a fixed `--virtual-time-budget`. The CLICK-level tests at the end
dispatch real mouse events — the coverage `--dump-dom` never had.
"""

from __future__ import annotations

import html as _html
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from main.prober.web import fixture_run
from utils.contention import describe_contention, scale_timeout
from utils.headless_chrome import LOOPBACK_ONLY, Chrome, ChromeTimeout, find_chrome
from utils.paths import repo_root

# MEASURED 2026-08-14: this file alone was 1280 s, blamed on "a chrome cold start is ~25 s".
# RE-MEASURED 2026-09-29 — that attribution was WRONG: a bare headless chrome starts in 0.37 s, and
# the virtual-time budget (20,000 → 2,000 → 500 ms → none) moved nothing. The 25 s was a D-Bus
# keyring timeout paid by every page that touched the network stack; `--password-store=basic`
# removes it. See `utils/headless_chrome.py`. The file now runs ONE browser per module over the
# DevTools pipe, with a fresh browser context per probe — ~40 probes in well under a minute.
# `slow` stays for now (the tier also needs chrome, which the routine gate must not require);
# `browser` is what lets you select or skip it by what it needs.
pytestmark = [pytest.mark.integration, pytest.mark.browser]

_REPO = str(repo_root())


def _chrome_binary() -> str:
    path = find_chrome()
    if not path:
        pytest.skip("no chrome/chromium on PATH — cannot render the prober web views")
    return path


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """The REAL app under uvicorn, on a loopback port nobody else uses.

    A `TestClient` would exercise the handlers but not the socket, and it is the socket the browser
    talks to. `arch_viewer_serve`'s own test makes the same argument: byte-identical generated
    output says nothing about whether the server answers.
    """
    run = fixture_run.build(str(tmp_path_factory.mktemp("renderrun")))
    port = _free_port()
    env = dict(os.environ, PYTHONPATH=os.path.join(_REPO, "src"))
    proc = subprocess.Popen(
        [sys.executable, "-m", "main.prober.web", run, "--port", str(port)],
        cwd=_REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        last = None
        for _ in range(int(200 * scale_timeout(1.0))):   # wait for the socket; never guess a sleep
            if proc.poll() is not None:
                pytest.fail(f"the server exited early:\n{proc.stdout.read()[-1500:]}")
            try:
                urllib.request.urlopen(base + "/api/health", timeout=1).read()
                break
            except Exception as exc:              # noqa: BLE001 — not up yet
                last = exc
                time.sleep(0.1)
        else:
            pytest.fail(f"the server never came up on {port}: {last}\n{describe_contention()}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


class _Browser:
    """The module's one chrome, relaunched if it dies — so one crash costs one probe, not the
    rest of the file. Isolation between probes is a fresh BROWSER CONTEXT each (own cookies,
    storage and cache), which is what the old fresh-process-per-probe bought, minus the process."""

    def __init__(self, binary: str) -> None:
        self.binary = binary
        self._chrome: Chrome | None = None

    def get(self) -> Chrome:
        if self._chrome is None or self._chrome.proc.poll() is not None:
            # Everything the page needs is served from loopback. Mapping every other host to a
            # dead address turns "the assets are vendored" from a claim in a docstring into a
            # property the browser enforces: a remote <script> would simply not load.
            self._chrome = Chrome(self.binary, LOOPBACK_ONLY)
        return self._chrome

    def close(self) -> None:
        if self._chrome is not None:
            self._chrome.close()


@pytest.fixture(scope="module")
def browser():
    holder = _Browser(_chrome_binary())
    try:
        yield holder
    finally:
        holder.close()


# Viewport widths under test. The layout viewport is now set exactly over CDP
# (`Emulation.setDeviceMetricsOverride`), so the old 500 px floor of `--window-size` no longer
# binds; 500 is KEPT so the measurements below stay comparable with their history. The layout's
# breakpoint is 720, so every phone rule IS exercised (the record's `narrow` flag asserts the
# media query actually matched).
NARROW = (500, 900)
DESKTOP = (1280, 900)


# A page is SETTLED when app.js says so (`data-busy` == "0": loaded, every HTMX request answered,
# every swap re-embedded and re-recorded — see the header of `static/app.js`). `data-ready` alone
# is NOT enough: it lands before HTMX issues its `load` requests. This replaces the fixed
# `--virtual-time-budget` wait.
_SETTLED = ("document.readyState === 'complete' && !!document.body && "
            "document.body.dataset.busy === '0'")
_SETTLE_TIMEOUT = 30.0      # a settled fixture page takes ~0.1 s idle; scaled at CALL time

# 🚨 QUIET IS DECIDED ON THE RAW LOAD RATIO, NOT ON `cpu_contention_factor`.
#
# `cpu_contention_factor` is a SCALING metric and is clamped to a floor of 1.0 — by construction it
# cannot tell a half-loaded box from an idle one, because both scale by 1.0. Using it as a
# CLASSIFIER is the mistake this constant exists to record: measured 2026-09-06 at load 12.89 on 16
# cores, with chrome unable to render a page inside 180 s, the factor read exactly **1.0** — so a
# 1.05 "quiet" bar called an 80%-utilised box idle and turned every starvation timeout into a hard
# FAILURE. That is strictly worse than the raw `TimeoutExpired` it replaced.
#
# The raw ratio has no floor and separates the cases: 3.89/16 = 0.24 (genuinely idle) vs
# 12.89/16 = 0.81 (a live trainer). The bar sits at 0.5 — half the cores actually free — which is
# well clear of both.
_QUIET_LOAD = 0.5


def _load_ratio() -> float:
    """1-minute load average per available CPU, UNFLOORED. See `_QUIET_LOAD`."""
    try:
        return os.getloadavg()[0] / max(1, len(os.sched_getaffinity(0)))
    except (OSError, AttributeError):       # not Linux / no affinity — assume busy, never idle
        return 1.0


def _with_timeout_doctrine(what: str, attempt_fn):
    """Run `attempt_fn()` under the project's timeout doctrine and return its result.

    🚨 **A TIMEOUT IS NEVER A SEMANTIC OUTCOME** (root CLAUDE.md → *Running Tests*), and until
    2026-09-06 this suite broke that rule: a starved chrome raised `TimeoutExpired`, pytest recorded
    a FAILURE, and a load average was reported as a rendering bug. Measured on the win-prob-era
    landing: three failures at load 36.8/16 cores, every one a timeout and not one a layout
    assertion — and on a re-run the failing SET MOVED, the signature no real regression has.

    * **quiet box** (`_load_ratio()` < `_QUIET_LOAD`) — RETRY once, then a hard FAILURE. A page that
      cannot settle on an idle machine is genuinely broken (a JS error before the record hook, or a
      request that never answered, both leave `data-busy` above 0).
    * **contended box** — SKIP immediately, loudly, with `describe_contention()`. INCONCLUSIVE.

    ⚠️ **The retry is QUIET-ONLY, and that asymmetry is measured rather than tidy**: a
    two-attempt-everywhere draft turned a 45-test tier into 7 tests in 50 minutes and was killed
    mid-line by its own wrapper, destroying the pytest summary and every failure message with it.
    """
    quiet = _load_ratio() < _QUIET_LOAD
    attempts = 2 if quiet else 1
    last: BaseException | None = None
    for _attempt in range(1, attempts + 1):
        try:
            return attempt_fn()
        except (subprocess.TimeoutExpired, ChromeTimeout) as e:
            last = e
    ratio = _load_ratio()
    detail = f"{what} did not settle on {attempts} attempt(s): {last}"
    if ratio < _QUIET_LOAD:
        raise AssertionError(
            f"{detail} — and THE BOX WAS QUIET (load/cpu {ratio:.2f} < {_QUIET_LOAD}), so this is a "
            f"real rendering failure, not starvation.\n    {describe_contention()}")
    pytest.skip(
        f"INCONCLUSIVE (not a failure): {detail}. The box is BUSY (load/cpu {ratio:.2f}), and a "
        f"timeout is never a semantic outcome — this says nothing about the page.\n"
        f"    {describe_contention()}\n    Re-run on a quiet box for a verdict; below "
        f"load/cpu {_QUIET_LOAD} a timeout here FAILS.")


def _dump_dom(browser, url: str, *, size=DESKTOP, dark: bool = False) -> str:
    """Render `url` in a fresh browser context, wait for the page to SETTLE, return its DOM."""
    return _with_timeout_doctrine(
        url, lambda: _run_chrome(browser, url, size=size, dark=dark))


def _open_settled(browser, url: str, *, size, dark: bool):
    """A settled `Page` in its own context. The CALLER closes it (the click tests keep it open)."""
    page = browser.get().page(width=size[0], height=size[1], dark=dark)
    try:
        page.goto(url)
        try:
            page.wait_for(_SETTLED, timeout=scale_timeout(_SETTLE_TIMEOUT))
        except ChromeTimeout as exc:
            state = page.eval("document.body ? JSON.stringify(document.body.dataset) : null")
            exc.args = (f"{exc} — the page's record at the deadline: {state}",)
            raise
    except BaseException:
        page.close()
        raise
    return page


def _run_chrome(browser, url: str, *, size, dark: bool) -> str:
    page = _open_settled(browser, url, size=size, dark=dark)
    try:
        return page.dom()
    finally:
        page.close()


def _body_data(dom: str) -> dict:
    """The `data-*` attributes off <body>. DOM keys are hyphenated: JS `dataset.chartMarks`
    serialises as `data-chart-marks`."""
    body = re.search(r"<body[^>]*>", dom)
    assert body, "no <body> in the dumped DOM"
    return {k: _html.unescape(v)
            for k, v in re.findall(r'data-([a-z-]+)="([^"]*)"', body.group(0))}


def _probe(browser, base: str, path: str, size=DESKTOP, dark: bool = False) -> dict:
    data = _body_data(_dump_dom(browser, base + path, size=size, dark=dark))
    assert data.get("ready") == "1", (
        f"{path}: the bootstrap never completed — a JS error before the record hook. "
        f"chartError={data.get('chart-error')!r}, record={data}")
    assert not data.get("chart-error"), f"{path}: chart embed failed: {data['chart-error']}"
    return data


# -- the gate ---------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/", "/battles", "/scan", "/triage", "/battle", "/game", "/analyze",
                                  "/falsify", "/calibration"])
def test_every_page_boots_with_its_vendored_libraries(server, browser, path):
    """Both bundles must define their globals with the network taken away.

    This is the assertion the arch viewer cannot make: its equivalent skips when the CDN is
    unreachable, so offline it proves nothing at all.
    """
    data = _probe(browser, server, path)
    assert data["htmx"] == "1", f"{path}: htmx did not load from /static/vendor"
    assert data["vega"] != "missing", f"{path}: the vega runtime did not load"
    assert data["vega-lite"] != "missing", f"{path}: the vega-lite compiler did not load"
    assert data["page"] == path.strip("/") or (path == "/" and data["page"] == "run")


def test_the_run_page_draws_its_chart(server, browser):
    """The spec is embedded server-side, so this is the pure "does Vega draw it" case."""
    data = _probe(browser, server, "/")
    assert int(data["charts"]) == 1
    assert int(data["chart-marks"]) > 0, (
        "the outcome chart embedded but drew ZERO marks — a spec can compile cleanly and plot "
        "nothing (a mistyped field does exactly that), which the text tests cannot see")
    assert int(data["rows"]) > 0, "the steps table rendered no rows"


def test_the_scan_page_fetches_its_table_and_chart_over_htmx(server, browser):
    """The strongest single assertion here.

    Nothing on /scan's table or chart exists in the page source: the filter form fires
    `hx-trigger="load"`, HTMX fetches `/partials/scan`, the fragment carries a Vega-Lite spec, and
    `app.js` re-embeds it on `htmx:afterSwap`. A non-zero mark count therefore proves the whole
    chain — server fragment, swap, re-embed, draw — end to end.
    """
    data = _probe(browser, server, "/scan")
    assert int(data["swaps"]) >= 1, (
        "the HTMX load trigger never completed a swap — /scan is the one view still fetched "
        "asynchronously, because the scan itself measured ~2 s on a real run")
    assert int(data["rows"]) > 0, "the scan table swapped in with no rows"
    assert int(data["charts"]) == 1, "the swapped fragment did not bring its chart"
    assert int(data["chart-marks"]) > 0, "the swapped-in chart drew nothing"
    assert not data.get("htmx-error")


def test_the_run_page_fetches_its_awareness_panel_over_htmx(server, browser):
    """The run summary paints server-side, but the 'did it know?' panel reads every captured
    loss's npz — so it arrives the way /scan's table does, and the run summary is not held behind
    it. Nothing in the panel exists in the page source; a completed swap is what proves the whole
    chain (fragment, swap, render) rather than a route that merely returns 200 to a test client."""
    data = _probe(browser, server, "/")
    assert int(data["swaps"]) >= 1, (
        "the awareness panel's hx-trigger=load never completed a swap — the run page would show "
        "its spinner forever")
    assert not data.get("htmx-error")
    assert int(data["rows"]) > 0


def test_the_triage_page_arrives_already_populated(server, browser):
    """Triage renders SERVER-SIDE on first paint (measured 436 ms on a real run) rather than
    landing empty and fetching, so the assertion inverts: the rows and the chart must be there
    with NO swap having happened."""
    data = _probe(browser, server, "/triage")
    assert int(data["rows"]) > 0, "the triage table did not render on first paint"
    assert int(data["chart-marks"]) > 0, "the lever chart drew nothing on first paint"
    assert int(data["swaps"]) == 0, (
        "triage made an HTMX round trip on load — the first paint should already carry the data")


def test_the_battles_page_arrives_already_populated(server, browser):
    """Also server-rendered (119 ms). No chart here, so it still pins that a chart-free page
    reaches `ready` — an embed loop that throws on zero specs would break exactly here."""
    data = _probe(browser, server, "/battles")
    assert int(data["rows"]) > 0, "the battles table did not render on first paint"
    assert int(data["charts"]) == 0
    assert int(data["swaps"]) == 0


@pytest.mark.parametrize("path", ["/falsify", "/calibration"])
def test_the_job_pages_render_before_anything_is_submitted(server, browser, path):
    """These start empty by design — the probes take minutes. Reaching `ready` with no charts is
    the correct state, and it must not read as a broken page."""
    data = _probe(browser, server, path)
    assert int(data["charts"]) == 0
    assert int(data["chart-marks"]) == 0


def test_a_404_page_still_renders_the_shell(server, browser):
    data = _body_data(_dump_dom(browser, server + "/no-such-page"))
    assert data.get("ready") == "1", "the error page must be a real page, not a bare JSON body"
    assert data["page"] == "error"


def test_the_pages_reference_no_remote_asset(server, browser):
    """Belt and braces on the host-resolver block: assert the markup itself is CDN-free, so this
    fails as a clear "someone added a CDN tag" rather than as a mysterious missing global."""
    for path in ("/", "/scan", "/calibration"):
        dom = _dump_dom(browser, server + path)
        assert "cdn." not in dom and "https://" not in re.sub(r'"https://vega\.github\.io[^"]*"',
                                                              "", dom), (
            f"{path} pulls a remote asset; the render gate would then depend on the network")


# -- responsive layout --------------------------------------------------------------------------
# Every assertion below reads a MEASUREMENT the page took of itself, not a CSS rule. A media query
# can be present and still be beaten on specificity (the arch viewer's bottom-sheet close bar was
# defined after the query that showed it, so `display:none` won and the sheet had no way out) —
# only asking the laid-out page catches that.

@pytest.mark.parametrize("path", ["/", "/battles", "/scan", "/triage", "/battle", "/game", "/analyze",
                                  "/falsify", "/calibration"])
def test_no_page_scrolls_sideways_on_a_narrow_viewport(server, browser, path):
    """The single rule the whole responsive layout serves.

    Wide content — the forensic tables, an oversized chart — must scroll inside its own
    `.scroll-x` / `.chart` box. If the PAGE scrolls instead, columns run off the right edge and
    the nav goes with them. `overflowWhat` names the offending element so this fails with the fix
    in the message.
    """
    data = _probe(browser, server, path, size=NARROW)
    assert data["narrow"] == "1", (
        f"the max-width:720px rules did not match at {data['vw']}px — the viewport meta tag "
        "or the breakpoint is wrong")
    assert data["overflowby"] == "0", (
        f"{path} overflows the viewport by {data['overflowby']}px — widest offender: "
        f"{data['overflowwhat']}")
    assert int(data["docw"]) <= int(data["vw"]) + 1, (
        f"{path}: document scrollWidth {data['docw']} > viewport {data['vw']}")


def test_a_wide_table_scrolls_inside_its_wrapper_not_the_page(server, browser):
    """The mechanism behind the rule above, asserted directly.

    `scan` is the widest table in the app (14 columns of forensic numbers) and it deliberately
    does NOT reflow into cards on a phone — a row of these numbers read out of column order is
    worse than one you scroll. So at least one wrapper must actually be scrolling here; if none
    is, the table is fitting by luck and the rule is untested.
    """
    data = _probe(browser, server, "/scan", size=NARROW)
    assert int(data["rows"]) > 0
    assert int(data["scrollers"]) >= 1, "the scan table lost its .scroll-x wrapper"
    assert int(data["scrollingwrappers"]) >= 1, (
        "no wrapper is scrolling at 500px — either the table got narrower or, more likely, the "
        "wrapper is not constraining it and the page is absorbing the width")
    assert data["overflowby"] == "0"


# The replay is the one view that REFLOWS instead of scrolling, so it gets its own pair of
# measurements. `_REPLAY` pins the fixture battle that actually has a battle log — the default
# battle has none, and a layout test over an empty card proves very little.
_REPLAY = "/battle?battle=step_4000000%2Fheuristic2%2Floss_003"


def test_the_battle_replay_stacks_on_a_phone_and_scrolls_nowhere(server, browser):
    """Everywhere else on this site the phone answer is "scroll the table, don't reflow it",
    because a row of forensic numbers read out of column order is worse than one you scroll. A
    TURN is different — it is a short narrative — so here the answer IS to reflow, and the claim
    that it does is a measurement rather than a screenshot someone looked at once.
    """
    data = _probe(browser, server, _REPLAY, size=NARROW)
    assert data["narrow"] == "1"
    assert int(data["rows"]) > 0, "no turn cards rendered at 500px"
    assert data["monstack"] == "1", (
        "the two mons are still side by side on a phone — the board did not reflow, so the "
        "species and HP are being squeezed into half a screen each")
    assert data["overflowby"] == "0", (
        f"the replay overflows by {data['overflowby']}px — widest offender: {data['overflowwhat']}")
    assert int(data["scrollers"]) == 0, (
        "the replay grew a .scroll-x wrapper — if it now needs one, it stopped being a reflowing "
        "card layout and the phone rules above need rethinking, not a scrollbar")


def test_the_replay_metrics_are_tappable_in_the_browser(server, browser):
    """The `title` tooltips have no TOUCH equivalent, so each metric is also tappable and the page
    carries a no-JS anchor to its legend. What a `--dump-dom` browser can prove is that the markup
    reached the live DOM and that app.js's own selector matches it — `metrics` is counted by that
    selector, not by the test.

    The tap BEHAVIOUR (the panel opening, its toggle, the legend jump) is gated by the CLICK-level
    tests at the end of this file, which dispatch real mouse events through the DevTools pipe.
    """
    data = _probe(browser, server, _REPLAY)
    assert int(data["metrics"]) >= 6, (
        "no tappable metrics in the live DOM — app.js's `.metric[title]` selector matches nothing, "
        "so a tap would explain nothing on a phone")
    dom = _dump_dom(browser, server + _REPLAY)
    assert 'id="turncard-legend"' in dom, "the legend has no anchor for the no-JS fallback to reach"
    assert 'class="whatsthis"' in dom, "no per-row link to the legend"


def test_the_battle_replay_is_side_by_side_on_a_desktop(server, browser):
    """The other half of the same claim: reflowing on a phone must not mean a phone layout on a
    1280px screen, where the two boards belong on one line for read-across."""
    data = _probe(browser, server, _REPLAY, size=DESKTOP)
    assert data["narrow"] == "0"
    assert data["monstack"] == "0", "the mons stacked at desktop width — the phone rule leaked up"
    assert data["overflowby"] == "0"


# -- the dark palette ---------------------------------------------------------------------------
# The stylesheet ships TWO palettes and only one is ever on screen, so any review by screenshot
# covers exactly half of it. These read what the page painted instead.
#
# `--bg` tokens, restated here on purpose: the point is to catch the palette silently NOT being
# applied, which a test that re-derives the expected value from the same stylesheet cannot do.
_DARK_BG, _LIGHT_BG = "rgb(22, 22, 26)", "rgb(251, 251, 249)"
_DARK_ACCENT, _LIGHT_ACCENT = "rgb(88, 166, 200)", "rgb(42, 111, 151)"
_UA_LINK_BLUE = "rgb(0, 0, 238)"


@pytest.mark.parametrize("path", ["/", "/battles", "/battle"])
def test_the_dark_palette_is_actually_painted(server, browser, path):
    data = _probe(browser, server, path, dark=True)
    assert data["scheme"] == "dark", "prefers-color-scheme did not match — the probe is not testing dark"
    assert data["bg"] == _DARK_BG, (
        f"{path} painted {data['bg']} in dark mode — the dark palette is defined but not applied")


@pytest.mark.parametrize("path", ["/", "/battles", "/battle"])
def test_links_follow_the_theme_rather_than_the_browser_default(server, browser, path):
    """MEASURED before the fix: an unclassed <a> was the UA default `rgb(0, 0, 238)` in BOTH
    schemes — about 2.4:1 against the dark background, under the 4.5:1 floor. Styling only the
    specific link classes left every plain link behind."""
    for dark, expected in ((True, _DARK_ACCENT), (False, _LIGHT_ACCENT)):
        data = _probe(browser, server, path, dark=dark)
        assert data["linkcolor"] != _UA_LINK_BLUE, f"{path}: a link kept the browser default blue"
        assert data["linkcolor"] == expected, f"{path}: link is {data['linkcolor']}, want {expected}"


def test_the_browsers_own_widgets_are_told_which_palette_is_live(server, browser):
    """`color-scheme` is what makes a <select>, a scrollbar and the canvas behind the page follow
    the theme. Undeclared, a dark-mode visitor gets a WHITE dropdown on a dark page — and a
    `--force-dark-mode` screenshot cannot show it, because that flag darkens UA widgets anyway."""
    for dark in (True, False):
        data = _probe(browser, server, "/battles", dark=dark)
        assert data["colorscheme"] == "light dark", (
            f"color-scheme is {data['colorscheme']!r} — the browser will render its own widgets "
            "in the light style regardless of the page")


def test_chart_text_is_legible_on_whichever_palette_is_live(server, browser):
    """Vega-Lite's default text is near-black and `_BASE` makes the chart background transparent,
    so on the dark palette the axis labels, titles and legends are black-on-#16161a. `app.js`
    themes each spec at embed time from the stylesheet's own custom properties."""
    dark = _probe(browser, server, "/", dark=True)
    light = _probe(browser, server, "/", dark=False)
    assert int(dark["chart-marks"]) > 0 and int(light["chart-marks"]) > 0
    assert dark["axistext"] == "rgb(154, 154, 149)", (
        f"dark axis text is {dark['axistext']} — Vega's default would be near-black and invisible")
    assert light["axistext"] == "rgb(107, 107, 102)"
    assert dark["axistext"] != light["axistext"], "the chart is not following the theme at all"


def test_form_controls_are_large_enough_not_to_trap_ios_zoom(server, browser):
    """Below 16px, iOS zooms the whole page when a <select> takes focus and does not zoom back
    out — one tap on the opponent filter and the page is stuck magnified."""
    for path in ("/battles", "/scan", "/falsify"):
        data = _probe(browser, server, path, size=NARROW)
        assert data["ctlfont"] != "none", f"{path} has no filter control to measure"
        assert float(data["ctlfont"].rstrip("px")) >= 16.0, (
            f"{path} controls are {data['ctlfont']} on a phone — iOS will zoom and stay zoomed")


def test_the_narrow_header_stays_compact_and_hides_nothing(server, browser):
    """The nav WRAPS rather than becoming a horizontal strip.

    The arch viewer shipped that strip: six controls in a scrollable bar of which exactly one was
    ever visible, with nothing to say the rest existed. Wrapping costs a line and hides nothing —
    so the header is allowed to be tallish, but must not be a scroll.
    """
    data = _probe(browser, server, "/scan", size=NARROW)
    # 160px buys three wrapped rows at this width: brand + auth badge, six nav tabs, and the run
    # picker. That is more furniture than the 130px this allowed before the picker existed, and
    # the trade is deliberate — the header is NOT sticky on narrow, so its height costs one scroll
    # rather than a permanent slice of the screen, whereas the ways of making it shorter all mean
    # hiding a control behind something (the exact regression the docstring above is about).
    assert int(data["headerh"]) <= 160, (
        f"the header is {data['headerh']}px of a phone screen — too much furniture")
    assert data["overflowby"] == "0", "the nav is overflowing rather than wrapping"


def test_charts_still_draw_at_a_narrow_width(server, browser):
    """A chart that silently collapses to zero marks when the container narrows is a chart that
    only works on the machine it was built on."""
    for path, expect_marks in (("/", True), ("/scan", True), ("/triage", True)):
        data = _probe(browser, server, path, size=NARROW)
        assert int(data["charts"]) == 1, path
        if expect_marks:
            assert int(data["chart-marks"]) > 0, f"{path}: the chart drew nothing at 500px"


@pytest.mark.parametrize("path", ["/", "/scan", "/calibration"])
def test_the_desktop_layout_does_not_inherit_the_phone_rules(server, browser, path):
    """The narrow rules must not leak upward: full-size controls, an inline run path, and no
    page-level horizontal scroll there either."""
    data = _probe(browser, server, path, size=DESKTOP)
    assert data["narrow"] == "0", "the max-width:720px rules matched at 1280px"
    assert int(data["vw"]) == DESKTOP[0]
    assert data["overflowby"] == "0", (
        f"{path} overflows at desktop width — widest offender: {data['overflowwhat']}")
    assert int(data["headerh"]) <= 64, (
        f"the desktop header is {data['headerh']}px — it should be a single row")


def test_the_api_answers_over_the_real_socket(server, browser):
    """The handlers are covered in-process; this covers the socket, which is a separate program
    concern — `arch_viewer_serve`'s split shipped two 500ing routes behind a healthy /healthz."""
    health = json.loads(urllib.request.urlopen(server + "/api/health", timeout=10).read())
    assert health["ok"] and health["n_runs"] >= 1
    assert health["jobs_unlocked"] is False, "an anonymous socket client must not be unlocked"
    scan = json.loads(urllib.request.urlopen(server + "/api/scan?outcome=loss", timeout=30).read())
    assert scan and scan[0]["worst"]["delta_v"] < 0
    try:
        urllib.request.urlopen(server + "/api/nope", timeout=10)
        pytest.fail("an unknown API route must 404")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
        assert "error" in json.loads(exc.read())


def test_the_copy_button_actually_copies(server, browser):
    """The scan table's whole point after this change is that a row can be taken onward — the id
    and the CLI command are the handoff, and a copy button that silently does nothing would be
    indistinguishable from one that works, in the DOM."""
    dom = _dump_dom(browser, server + "/scan?run=run_fixture", size=DESKTOP)
    assert 'class="linky copybtn"' in dom, "the scan rows lost their take-it-onward buttons"
    assert "main.prober.query analyze" in dom, "the copyable command is missing"
    assert "data-battle-id=" in dom, "rows no longer carry the id a reader needs"


# -- CLICK level ---------------------------------------------------------------------------------
# Everything above reads what a page DREW. These read what it DOES when a reader acts on it — the
# coverage `--dump-dom` could never give. Each one dispatches a real mouse press/release at the
# element's centre (`Page.click`), so the event goes through hit testing: a target covered by
# another element, or collapsed to zero size, misses exactly as a finger would. They are written
# against the prober's DOM for the fixture trace, never against the Python that produced it, so
# they hold whichever engine serves the page.

def _clicked(browser, server, path, size, fn):
    """Open `path` settled, run `fn(page)`, always close the context."""
    def attempt():
        page = _open_settled(browser, server + path, size=size, dark=False)
        try:
            return fn(page)
        finally:
            page.close()
    return _with_timeout_doctrine(server + path, attempt)


_METRIC_STATE = """(() => {
  const el = document.querySelectorAll('.metric[title]')[%d];
  const row = el.closest('p'), next = row.nextElementSibling;
  const panels = document.querySelectorAll('p.metric-help');
  return {label: (el.textContent || '').trim().split(/\\s+/)[0], title: el.getAttribute('title'),
          panels: panels.length,
          nextIsPanel: !!(next && next.classList.contains('metric-help')),
          panelText: next && next.classList.contains('metric-help') ? next.textContent : null,
          panelFor: next && next.dataset ? (next.dataset.forMetric || null) : null,
          record: document.body.dataset.metrichelp};
})()"""


@pytest.mark.parametrize("size", [NARROW, DESKTOP], ids=["phone", "desktop"])
def test_tapping_a_metric_opens_its_explanation_under_the_row_and_tapping_again_closes_it(
        server, browser, size):
    """THE TAP PATH, driven. `title` has no touch equivalent, so a tap on a turn-card metric must
    render that metric's OWN title into a panel directly under its row, and a second tap on the
    same metric must close it (a toggle, not a stack). Tapping a DIFFERENT metric moves the one
    panel rather than adding a second."""
    def drive(page):
        n = page.eval("document.querySelectorAll('.metric[title]').length")
        assert n >= 2, f"need two tappable metrics on the replay, found {n}"
        before = page.eval(_METRIC_STATE % 0)
        assert before["panels"] == 0, "a help panel is open before anything was tapped"

        page.click(".metric[title]", 0)
        opened = page.wait_for(
            "document.querySelectorAll('p.metric-help').length === 1 && (%s)" % (_METRIC_STATE % 0),
            timeout=scale_timeout(10.0))
        assert opened["nextIsPanel"], "the panel did not open DIRECTLY under the tapped row"
        assert opened["panelFor"] == opened["label"]
        assert opened["record"] == opened["label"], (
            f"app.js recorded {opened['record']!r} for a tap on {opened['label']!r}")
        assert opened["panelText"] == f"{opened['label']} {opened['title']}", (
            "the panel does not carry the metric's own title — the tooltip and the tap disagree")

        # The same metric again: a toggle, so the panel goes and the record clears.
        page.click(".metric[title]", 0)
        closed = page.wait_for(
            "document.querySelectorAll('p.metric-help').length === 0 && (%s)" % (_METRIC_STATE % 0),
            timeout=scale_timeout(10.0))
        assert closed["record"] == "", "the record still names a metric after the panel closed"

        # Two DIFFERENT metrics in turn: still exactly one panel, now for the second.
        page.click(".metric[title]", 0)
        page.wait_for("document.querySelectorAll('p.metric-help').length === 1",
                      timeout=scale_timeout(10.0))
        page.click(".metric[title]", 1)
        second = page.eval(_METRIC_STATE % 1)
        assert second["panels"] == 1, f"tapping a second metric stacked {second['panels']} panels"
        assert second["record"] == second["label"]
    _clicked(browser, server, _REPLAY, size, drive)


def test_the_whatsthis_link_takes_a_reader_to_the_legend(server, browser):
    """THE LEGEND PATH, driven. The per-row `?` is the no-JS route to what the numbers mean; a
    click on it must land the reader AT the legend — the hash set, and the legend scrolled to the
    top of a phone-sized viewport rather than left 50 turns away."""
    def drive(page):
        page.eval("window.scrollTo(0, document.body.scrollHeight)")      # start far from it
        n = page.eval("document.querySelectorAll('a.whatsthis').length")
        assert n >= 1, "no `?` link on any turn card"
        page.click("a.whatsthis", n - 1)                                  # the LAST row's link
        state = page.wait_for(
            "location.hash === '#turncard-legend' && (() => {"
            " const r = document.getElementById('turncard-legend').getBoundingClientRect();"
            " return {top: r.top, vh: window.innerHeight}; })()",
            timeout=scale_timeout(10.0))
        assert 0 <= state["top"] < state["vh"] / 2, (
            f"the hash moved but the legend is at {state['top']:.0f}px of a {state['vh']}px "
            "viewport — the reader was not taken to it")
    _clicked(browser, server, _REPLAY, NARROW, drive)


def test_clicking_a_copy_button_copies_its_command(server, browser):
    """The copy button, CLICKED. The DOM test above proves the button exists; only a click proves
    it does something — and "does nothing" is exactly the failure a reader would report. app.js
    records what it copied on success; the clipboard itself is read back where the browser allows.
    """
    def drive(page):
        chrome = browser.get()
        origin = server.rstrip("/")
        chrome.send("Browser.grantPermissions", {
            "origin": origin, "browserContextId": page.ctx,
            "permissions": ["clipboardReadWrite", "clipboardSanitizedWrite"]})
        want = page.wait_for(
            "(() => { const b = document.querySelector('.copybtn'); return b && b.dataset.copy; })()",
            timeout=scale_timeout(10.0))
        page.click(".copybtn", 0)
        got = page.wait_for("document.body.dataset.copied", timeout=scale_timeout(10.0))
        assert got == want, f"the button copied {got!r}, its row offers {want!r}"
        clip = page.eval("navigator.clipboard.readText()")
        assert clip == want, f"the clipboard holds {clip!r}, not the row's command"
    _clicked(browser, server, "/scan?run=run_fixture", DESKTOP, drive)
