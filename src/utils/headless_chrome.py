"""One headless chrome, driven over the DevTools protocol on a PIPE — no Playwright, no websocket.

Used by the browser tier (`pytest -m browser`). It exists for two reasons, both measured on
2026-09-29 (Chrome for Testing 152, this box, load/cpu ~0.1):

1. **The ~25 s per test the docs blamed on "chrome cold start" was a D-Bus timeout.** A bare
   `chrome --headless --dump-dom about:blank` returns in 0.37 s, and a `file://` page in 0.40 s —
   but ANY page that touches the network stack (http://127.0.0.1, or a `file://` page pulling a
   CDN script) took 25.5 s, at every `--virtual-time-budget` from 20,000 down to 500 ms and with
   no budget at all. 25 s is libdbus's default reply timeout: the network service initialises
   cookie encryption through the desktop keyring over D-Bus, and nothing answers.
   **`--password-store=basic` removes it (25.5 s → 0.65 s per page).** It is in `BASE_FLAGS`, and
   every browser launch in the suite goes through them.

2. **`--dump-dom` cannot click.** It renders, fast-forwards virtual time and prints the DOM. A
   tap, a toggle, an anchor jump — anything a reader DOES — was ungated. This driver keeps one
   browser per test module and gives every probe a fresh BROWSER CONTEXT (its own cookies,
   storage and cache — the per-test isolation a fresh process used to buy), waits on a signal
   the PAGE sets rather than on a fixed budget, and dispatches real mouse events through the
   compositor's hit test (`Input.dispatchMouseEvent`), so a covered or zero-size target misses
   exactly as a finger would.

Transport: `--remote-debugging-pipe` makes chrome read NUL-terminated JSON commands on fd 3 and
write responses/events on fd 4. That is the whole protocol surface needed here, with no
dependency beyond the standard library.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from typing import Any

# Every launch in the browser tier uses these. See the module docstring for `--password-store`.
BASE_FLAGS = (
    "--headless",
    "--no-sandbox",
    "--disable-gpu",
    "--password-store=basic",       # 🚨 without it every networked page costs a 25 s D-Bus timeout
    "--use-mock-keychain",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-component-update",
    "--mute-audio",
)

# Everything the prober serves comes from loopback. Mapping every other host to a dead address
# turns "the assets are vendored" into a property the browser enforces.
LOOPBACK_ONLY = "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1"


def find_chrome() -> str | None:
    for name in ("chrome", "google-chrome", "google-chrome-stable", "chromium",
                 "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    for path in (os.path.expanduser("~/.local/bin/chrome"),
                 "/usr/bin/google-chrome", "/usr/bin/chromium"):
        if os.path.exists(path):
            return path
    return None


class ChromeError(RuntimeError):
    """The browser answered a command with an error, or went away."""


class ChromeTimeout(TimeoutError):
    """A wait ran out. NEVER a semantic outcome on its own — the caller decides, by load."""


class Chrome:
    """One browser process. `page(...)` hands out isolated tabs; `close()` stops it by its PID."""

    def __init__(self, binary: str, *extra_flags: str) -> None:
        self._to_chrome_r, self._to_chrome_w = os.pipe()
        self._from_chrome_r, self._from_chrome_w = os.pipe()
        # chrome expects its command pipe on fd 3 and its reply pipe on fd 4. Dup both HIGH
        # first: either pipe end may itself be fd 3 or 4, and a direct dup2 would clobber it.
        r, w = self._to_chrome_r, self._from_chrome_w

        def _wire_fds() -> None:
            hi_r, hi_w = os.dup(r), os.dup(w)
            os.dup2(hi_r, 3)
            os.dup2(hi_w, 4)
        self.proc = subprocess.Popen(
            [binary, *BASE_FLAGS, *extra_flags, "--remote-debugging-pipe", "about:blank"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            pass_fds=(3, 4), preexec_fn=_wire_fds)
        os.close(self._to_chrome_r)
        os.close(self._from_chrome_w)
        self._writer = os.fdopen(self._to_chrome_w, "wb", buffering=0)
        self._lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, "queue.Queue[dict]"] = {}
        self._events: dict[str, "queue.Queue[dict]"] = {}
        self._dead: str | None = None
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.version = self.send("Browser.getVersion", timeout=60.0)

    # -- transport -------------------------------------------------------------------------
    def _read_loop(self) -> None:
        buf = b""
        with os.fdopen(self._from_chrome_r, "rb", buffering=0) as reader:
            while True:
                chunk = reader.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                while b"\0" in buf:
                    raw, buf = buf.split(b"\0", 1)
                    msg = json.loads(raw)
                    if "id" in msg:
                        q = self._pending.pop(msg["id"], None)
                        if q is not None:
                            q.put(msg)
                    elif "sessionId" in msg:
                        self._events.setdefault(msg["sessionId"], queue.Queue()).put(msg)
        self._dead = "the browser closed its DevTools pipe (crashed or was killed)"
        for q in list(self._pending.values()):
            q.put({"error": {"message": self._dead}})

    def send(self, method: str, params: dict | None = None, *, session: str | None = None,
             timeout: float = 30.0) -> dict:
        if self._dead:
            raise ChromeError(self._dead)
        with self._lock:
            self._next_id += 1
            mid = self._next_id
            q: "queue.Queue[dict]" = queue.Queue()
            self._pending[mid] = q
            msg: dict[str, Any] = {"id": mid, "method": method, "params": params or {}}
            if session:
                msg["sessionId"] = session
            self._writer.write(json.dumps(msg).encode() + b"\0")
        try:
            reply = q.get(timeout=timeout)
        except queue.Empty:
            self._pending.pop(mid, None)
            raise ChromeTimeout(f"{method} got no reply in {timeout:.0f}s") from None
        if "error" in reply:
            raise ChromeError(f"{method}: {reply['error'].get('message')}")
        return reply.get("result", {})

    def page(self, *, width: int, height: int, dark: bool = False) -> "Page":
        """A tab in its OWN browser context — fresh cookies, storage and cache."""
        ctx = self.send("Target.createBrowserContext", {"disposeOnDetach": True})["browserContextId"]
        target = self.send("Target.createTarget", {"url": "about:blank",
                                                   "browserContextId": ctx})["targetId"]
        session = self.send("Target.attachToTarget", {"targetId": target,
                                                      "flatten": True})["sessionId"]
        return Page(self, ctx, target, session, width=width, height=height, dark=dark)

    def close(self) -> None:
        try:
            self.send("Browser.close", timeout=5.0)
        except (ChromeError, ChromeTimeout, OSError):
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()                      # this process only — never a name-based kill
            self.proc.wait(timeout=10)
        try:
            self._writer.close()
        except OSError:
            pass


class Page:
    def __init__(self, chrome: Chrome, ctx: str, target: str, session: str, *,
                 width: int, height: int, dark: bool) -> None:
        self.chrome, self.ctx, self.target, self.session = chrome, ctx, target, session
        self.send("Page.enable")
        self.send("Runtime.enable")
        # A true layout viewport of exactly this size (the old `--window-size` path was clamped
        # to >= 500 px by headless chrome on Linux).
        self.send("Emulation.setDeviceMetricsOverride",
                  {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
        # `prefers-color-scheme` as the page sees it. Unlike `--force-dark-mode` this does NOT
        # darken the browser's own widgets — which is what a real dark-mode visitor gets.
        self.send("Emulation.setEmulatedMedia", {"features": [
            {"name": "prefers-color-scheme", "value": "dark" if dark else "light"}]})

    def send(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        return self.chrome.send(method, params, session=self.session, timeout=timeout)

    def goto(self, url: str, timeout: float = 30.0) -> None:
        res = self.send("Page.navigate", {"url": url}, timeout=timeout)
        if res.get("errorText"):
            raise ChromeError(f"navigation to {url} failed: {res['errorText']}")

    def eval(self, expression: str, timeout: float = 30.0) -> Any:
        res = self.send("Runtime.evaluate", {"expression": expression, "returnByValue": True,
                                             "awaitPromise": True}, timeout=timeout)
        if "exceptionDetails" in res:
            raise ChromeError(f"page threw evaluating {expression[:80]!r}: "
                              f"{res['exceptionDetails'].get('text')} "
                              f"{res['exceptionDetails'].get('exception', {}).get('description', '')}")
        return res.get("result", {}).get("value")

    def wait_for(self, expression: str, timeout: float) -> Any:
        """Poll `expression` until truthy. Raises `ChromeTimeout` — never returns a stale False."""
        deadline = time.monotonic() + timeout
        last: Any = None
        while True:
            try:
                last = self.eval(expression, timeout=max(1.0, deadline - time.monotonic()))
            except ChromeError as exc:           # mid-navigation: the context was replaced
                if "context" not in str(exc).lower() and "navigat" not in str(exc).lower():
                    raise
                last = None
            if last:
                return last
            if time.monotonic() >= deadline:
                raise ChromeTimeout(f"waited {timeout:.0f}s for {expression!r}")
            time.sleep(0.02)

    def dom(self) -> str:
        return self.eval("document.documentElement.outerHTML")

    def click(self, selector: str, index: int = 0) -> dict:
        """A REAL click: scroll the element into view, then press and release the mouse at its
        centre. The event goes through hit testing, so whatever is actually on top at that point
        receives it — a click that a covering element would swallow is swallowed here too."""
        box = self.eval(
            "(() => { const el = document.querySelectorAll(%s)[%d]; if (!el) return null;"
            " el.scrollIntoView({block: 'center', inline: 'center'});"
            " const r = el.getBoundingClientRect();"
            " return {x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height}; })()"
            % (json.dumps(selector), index))
        if not box:
            raise ChromeError(f"nothing matches {selector!r}[{index}] to click")
        if box["w"] <= 0 or box["h"] <= 0:
            raise ChromeError(f"{selector!r}[{index}] has no area ({box}) — it cannot be tapped")
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            self.send("Input.dispatchMouseEvent",
                      {"type": kind, "x": box["x"], "y": box["y"], "button": "left",
                       "buttons": 0 if kind == "mouseMoved" else 1, "clickCount": 1})
        return box

    def close(self) -> None:
        try:
            self.chrome.send("Target.closeTarget", {"targetId": self.target}, timeout=10.0)
            self.chrome.send("Target.disposeBrowserContext", {"browserContextId": self.ctx},
                             timeout=10.0)
        except (ChromeError, ChromeTimeout):
            pass

    def __enter__(self) -> "Page":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
