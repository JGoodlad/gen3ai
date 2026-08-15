"""Does the emitted page actually RENDER? — a headless-browser gate on the viewer.

Every other test in `build_arch_viewer_test.py` asserts properties of the emitted TEXT. None of
them executes a line of the JavaScript, and that gap is not theoretical: the viewer shipped with a
`#theme=dark` deep link that painted every node in the LIGHT palette on a dark canvas, because
cytoscape resolves the CSS variables once at construction and owns them thereafter on its own
canvas. The text was perfectly well-formed. Only looking at it found the bug.

So the page publishes a small machine-readable record at the end of init (`document.body.dataset`)
and this test reads it back out of a real browser:

  * `ready`      — the script ran to completion (any throw leaves this unset)
  * `nodes`      — cytoscape received every node
  * `positioned` — the hand-placed layout actually assigned positions
  * `node-bg`    — the node fill AS CYTOSCAPE COMPUTED IT, which is the value the CSS-only checks
                   cannot see, and the one the dark-mode bug got wrong

SKIPS rather than fails when there is no browser or no network — the browser is not a project
dependency and the page loads cytoscape from a CDN. Offline is detected by a `data-cytoscape`
attribute the page sets on failure, NOT by looking for the error text: that text also appears
verbatim in the script's own source, so a substring check reports every healthy page as offline
(it did, and it turned three real assertions into silent skips).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

import pytest

from agents.model import build_arch_viewer as B

# MEASURED 2026-08-14: 147 s for 9 tests — a fresh headless chrome per test at ~25 s each. Second
# only to the prober's own render suite. `browser` = what it needs, `slow` = what it costs.
pytestmark = [pytest.mark.integration, pytest.mark.browser, pytest.mark.slow]

# Both themes' `--surface-2`, which is what a node is filled with. These are the literal values in
# the template's `:root` / `html[data-theme=dark]` blocks; a palette change should update them
# here deliberately, because "the node matches the page" is the property under test.
_SURFACE_2 = {"light": (242, 242, 239), "dark": (36, 36, 35)}


def _chrome() -> str:
    for name in ("chrome", "google-chrome", "google-chrome-stable", "chromium",
                 "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    for path in (os.path.expanduser("~/.local/bin/chrome"),
                 "/usr/bin/google-chrome", "/usr/bin/chromium"):
        if os.path.exists(path):
            return path
    pytest.skip("no chrome/chromium on PATH — cannot render the viewer")


# A phone viewport, via an iframe. Headless chrome clamps its window to ~500px wide on Linux, so
# `--window-size=390,844` silently renders a 500px page and crops the screenshot — which reads as
# "the layout is broken" when it is the harness that is. An iframe gets the width it is told.
_PHONE_HARNESS = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{{margin:0}}iframe{{width:{w}px;height:{h}px;border:0;display:block}}</style>
</head><body><iframe id="f" src="{url}"></iframe><script>
/* Report on the iframe's LOAD event, not on a timer.
   A fixed delay reads `d.body` while the frame is still parsing and gets null. A fast poll loop
   is worse: under --virtual-time-budget a setTimeout chain advances virtual time immediately, so
   300 x 50ms burns the entire budget without ever yielding to the CDN fetch the frame is waiting
   on — the harness then reports "not ready" for a page that is perfectly fine. */
function report() {{
  var out;
  try {{
    var d = document.getElementById('f').contentDocument, w = d.defaultView;
    out = {{}};
    for (var k in d.body.dataset) out[k] = d.body.dataset[k];
    out.innerW = w.innerWidth; out.innerH = w.innerHeight;
    out.headerH = d.querySelector('header').offsetHeight;
    out.cyH = d.getElementById('cy').clientHeight;
    out.ckFont = w.getComputedStyle(d.getElementById('ck')).fontSize;
    out.sheetbar = w.getComputedStyle(d.getElementById('sheetbar')).display;
    var bar = d.querySelector('header');
    out.barScrollW = bar.scrollWidth; out.barClientW = bar.clientWidth;
    out.ctlBtn = w.getComputedStyle(d.getElementById('ctlbtn')).display;
    out.ctlsPos = w.getComputedStyle(d.getElementById('ctls')).position;
    out.ctlsDisplay = w.getComputedStyle(d.getElementById('ctls')).display;
    out.asidePos = w.getComputedStyle(d.querySelector('aside')).position;
  }} catch (e) {{ out = {{err: String(e)}}; }}
  document.body.dataset.probe = JSON.stringify(out);
}}
document.getElementById('f').addEventListener('load', function () {{ setTimeout(report, 600); }});
setTimeout(function () {{                       // the frame never loaded at all
  if (!document.body.dataset.probe) document.body.dataset.probe =
    JSON.stringify({{err: 'iframe load event never fired'}});
}}, 20000);
</script></body></html>"""


def _probe_at(binary: str, page, width: int, height: int) -> dict:
    """Render the viewer at an exact viewport and read measurements back out of it."""
    with tempfile.TemporaryDirectory() as d:
        harness = os.path.join(d, "harness.html")
        with open(harness, "w") as fh:
            fh.write(_PHONE_HARNESS.format(w=width, h=height, url=f"file://{page}"))
        proc = subprocess.run(
            [binary, "--headless", "--no-sandbox", "--disable-gpu",
             "--allow-file-access-from-files", "--virtual-time-budget=25000", "--dump-dom",
             f"--window-size={max(width, 520)},{max(height, 560)}", f"file://{harness}"],
            capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        pytest.skip(f"headless chrome failed ({proc.returncode}): {proc.stderr[-300:]}")
    m = re.search(r'data-probe="([^"]*)"', proc.stdout)
    if not m:
        pytest.skip("the harness never reported — cytoscape probably could not be fetched")
    import html as _html
    probe = json.loads(_html.unescape(m.group(1)))
    if "err" in probe:                    # never let a broken harness read as a passing layout
        pytest.fail(f"the phone harness failed at {width}x{height}: {probe['err']}")
    return probe


def _dump_dom(binary: str, url: str) -> str:
    """Load the page in headless chrome and return the DOM after the script has run."""
    proc = subprocess.run(
        [binary, "--headless", "--no-sandbox", "--disable-gpu", "--virtual-time-budget=15000",
         "--dump-dom", url],
        capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        pytest.skip(f"headless chrome failed ({proc.returncode}): {proc.stderr[-400:]}")
    return proc.stdout


def _body_data(dom: str) -> dict:
    """`data-*` attributes off <body>. Note the DOM keys are hyphenated: JS `dataset.nodeBg`
    serialises as `data-node-bg`."""
    body = re.search(r"<body[^>]*>", dom)
    assert body, "no <body> in the dumped DOM"
    return dict(re.findall(r'data-([a-z-]+)="([^"]*)"', body.group(0)))


def _rgb(value: str) -> tuple:
    nums = re.findall(r"[\d.]+", value)
    assert len(nums) >= 3, f"unparseable colour {value!r}"
    return tuple(round(float(n)) for n in nums[:3])


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """The viewer as it would be committed, written somewhere a file:// URL can reach."""
    path = tmp_path_factory.mktemp("viewer") / "architecture_viewer.html"
    path.write_text(B.render(B.build_payload(B._load_graph(None))))
    return path


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_page_renders_and_the_theme_reaches_the_canvas(page, theme):
    """The regression gate: the cytoscape canvas must agree with the CSS about the theme."""
    binary = _chrome()
    dom = _dump_dom(binary, f"file://{page}#theme={theme}")
    data = _body_data(dom)
    if data.get("cytoscape") == "missing":
        pytest.skip("cytoscape could not be fetched from the CDN — no network")

    assert data.get("ready") == "1", (
        "the viewer script did not run to completion — a JS error before the status hook")
    payload = B.build_payload(B._load_graph(None))
    assert data.get("nodes") == str(len(payload["nodes"]))
    assert data.get("positioned") == str(len(payload["nodes"])), (
        "some node was left at the origin — the hand-placed layout did not cover it")

    assert f'data-theme="{theme}"' in dom, "the requested theme was not applied to <html>"
    assert _rgb(data["node-bg"]) == _SURFACE_2[theme], (
        f"nodes are painted {data['node-bg']} while the page is in {theme} mode — cytoscape "
        "resolved the palette at construction and never picked the theme up")


def test_dark_is_the_default_with_no_hash(page):
    """Dark is the default; a bare open must not need a hash to get there."""
    binary = _chrome()
    dom = _dump_dom(binary, f"file://{page}")
    data = _body_data(dom)
    if data.get("cytoscape") == "missing":
        pytest.skip("cytoscape could not be fetched from the CDN — no network")
    assert 'data-theme="dark"' in dom
    assert _rgb(data["node-bg"]) == _SURFACE_2["dark"]


def test_vendored_copy_renders_with_no_network():
    """`--vendor` must produce a page that works with the network taken away entirely."""
    binary = _chrome()
    try:
        html = B.vendor(B.render(B.build_payload(B._load_graph(None))))
    except SystemExit as exc:
        pytest.skip(f"--vendor needs network once: {exc}")
    assert "cdn.jsdelivr.net" not in html, "a vendored copy must not keep a remote script"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vendored.html")
        with open(path, "w") as fh:
            fh.write(html)
        # `--disable-features=NetworkService` is not enough to prove offline; the absence of any
        # remote URL above is the actual proof. This asserts the inlined bundle still executes.
        dom = _dump_dom(binary, f"file://{path}")
    data = _body_data(dom)
    assert data.get("cytoscape") != "missing", (
        "the inlined bundle did not define cytoscape — vendoring is broken, not offline")
    assert data.get("ready") == "1"


def test_phone_viewport_fits_and_is_usable(page):
    """The iPhone layout: nothing below the fold, no iOS zoom trap, and a way out of the sheet.

    Three regressions this pins, all of which shipped at least once:

    * `#wrap` was `calc(100vh - 52px)` — a hardcoded header height. The phone header is not 52px
      (it measured 81px before the buttons were told not to wrap their own text), so the bottom
      of the graph sat below the fold. Now the column is flexed and header + canvas must equal
      the viewport exactly.
    * The control font was 12px. Below 16px, iOS zooms the whole page when a `<select>` takes
      focus and does not zoom back out, so one tap on the checkpoint picker left the page stuck.
    * `#sheetbar` — the bottom sheet's only close affordance — was defined AFTER the media query
      that shows it, so `display:none` won on equal specificity and the sheet had no way out.
    """
    binary = _chrome()
    p = _probe_at(binary, page, 390, 844)
    if p.get("cytoscape") == "missing":
        pytest.skip("cytoscape could not be fetched from the CDN — no network")

    assert p["innerW"] == 390, f"harness did not give a 390px viewport: {p['innerW']}"
    assert p.get("ready") == "1"
    assert p.get("positioned") == p.get("nodes"), "a node was left unplaced on the phone layout"

    # +-1px: offsetHeight/clientHeight are integers over a fractional layout, so an exact
    # equality fails on a sub-pixel header. One pixel is rounding; the bug this guards against
    # was 35.
    assert abs(p["headerH"] + p["cyH"] - p["innerH"]) <= 1, (
        f"header {p['headerH']} + canvas {p['cyH']} != viewport {p['innerH']} — the graph runs "
        "below the fold; the column must be flexed, not offset by a hardcoded header height")
    assert p["headerH"] <= 64, f"header is {p['headerH']}px of a phone screen — too tall"
    assert p["ckFont"] == "16px", (
        f"controls are {p['ckFont']}; below 16px iOS zooms the page on focus and stays zoomed")
    assert p["sheetbar"] == "flex", "the bottom sheet has no visible close bar"
    assert p["asidePos"] == "fixed", "the side panel is still in flow — it would eat the canvas"

    # The complaint this replaced: the bar was a horizontal scroll holding six controls, of which
    # exactly one (a ~330px checkpoint picker) was ever visible, with nothing to say the rest
    # existed. Everything left in the bar must FIT in the bar; the controls live in a sheet.
    assert p["barScrollW"] <= p["barClientW"] + 1, (
        f"the top bar overflows ({p['barScrollW']}px of content in {p['barClientW']}px) — "
        "something is hidden behind a scroll nobody can see")
    assert p["ctlBtn"] != "none", "no controls button on a phone, so the controls are unreachable"
    assert p["ctlsPos"] == "fixed", "the controls did not become a sheet"


def test_landscape_phone_keeps_the_compact_chrome(page):
    """844x390 is a phone too, and too short to hand 352px to a side panel.

    The breakpoint is width OR short height for exactly this case; a width-only rule leaves a
    landscape iPhone with the desktop layout.
    """
    binary = _chrome()
    p = _probe_at(binary, page, 844, 390)
    if p.get("cytoscape") == "missing":
        pytest.skip("cytoscape could not be fetched from the CDN — no network")
    assert p["asidePos"] == "fixed", "landscape phone still uses the in-flow desktop panel"
    assert p["ckFont"] == "16px"
    assert abs(p["headerH"] + p["cyH"] - p["innerH"]) <= 1


def test_desktop_keeps_the_side_panel_in_flow(page):
    """The phone rules must not leak upward: on a desktop the panel is a column, not an overlay."""
    binary = _chrome()
    p = _probe_at(binary, page, 1280, 900)
    if p.get("cytoscape") == "missing":
        pytest.skip("cytoscape could not be fetched from the CDN — no network")
    assert p["asidePos"] != "fixed", "the bottom-sheet layout leaked into the desktop breakpoint"
    assert abs(p["headerH"] + p["cyH"] - p["innerH"]) <= 1
    # `display:contents` is what keeps the desktop header identical now that the controls are
    # wrapped in #ctls — the wrapper must stay invisible to layout here.
    assert p["ctlsDisplay"] == "contents", "the #ctls wrapper became a box on desktop"
    assert p["ctlBtn"] == "none", "the phone controls button leaked onto the desktop"


def test_serve_answers_every_route():
    """`--serve` must actually serve — all three routes, not just the cheap one.

    This exists because splitting the server out of the generator moved
    `_snapshot_provenance` and left two routes calling it through the hot-reloaded GENERATOR
    module, where it no longer was. `/healthz` does not touch it and kept returning 200, so the
    endpoint looked healthy while `/` and `/graph.json` were both 500ing.

    Byte-identical generated output — which the refactor did prove — says nothing about any of
    this. The generator is a pure function; the server is a separate program that happens to call
    it. Only opening the socket tests the socket.
    """
    import socket
    import sys
    import time
    import urllib.error
    import urllib.request

    with socket.socket() as probe:                      # borrow a free port, then let it go
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    env = dict(os.environ, PYTHONPATH=os.path.join(B._REPO, "src"))
    proc = subprocess.Popen(
        [sys.executable, "-m", "agents.model.build_arch_viewer", "--serve",
         "--port", str(port)],
        cwd=B._REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        base, last = f"http://127.0.0.1:{port}", None
        for _ in range(100):                            # wait for the socket, do not guess a sleep
            if proc.poll() is not None:
                pytest.fail(f"--serve exited early:\n{proc.stdout.read()[-800:]}")
            try:
                urllib.request.urlopen(base + "/healthz", timeout=1).read()
                break
            except Exception as exc:                    # noqa: BLE001 - not up yet
                last = exc
                time.sleep(0.1)
        else:
            pytest.fail(f"--serve never came up on {port}: {last}")

        health = urllib.request.urlopen(base + "/healthz", timeout=10).read().decode()
        assert health.startswith("ok "), health

        page = urllib.request.urlopen(base + "/", timeout=20).read().decode()
        assert page.lstrip().startswith("<!DOCTYPE html>")
        assert "__PAYLOAD__" not in page

        graph = json.loads(urllib.request.urlopen(base + "/graph.json", timeout=20).read())
        assert graph["nodes"] and graph["edges"]
        assert graph["graph_sha256"]
        assert graph["snapshot_built"] is None or "days" in graph["snapshot_built"], (
            "the served payload lost its provenance block")

        try:
            urllib.request.urlopen(base + "/nope", timeout=10)
            pytest.fail("an unknown route must 404, not 200")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
