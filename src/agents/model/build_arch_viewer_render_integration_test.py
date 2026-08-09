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

import os
import re
import shutil
import subprocess
import tempfile

import pytest

from agents.model import build_arch_viewer as B

pytestmark = pytest.mark.integration

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
