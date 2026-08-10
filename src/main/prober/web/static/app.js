/* Page bootstrap + the machine-readable render record.
 *
 * WHY THE RECORD EXISTS. Every server-side test here asserts properties of the emitted TEXT, and
 * none of them executes a line of this file. The arch viewer shipped a bug of exactly that shape:
 * perfectly well-formed markup whose canvas came out in the wrong palette, because only a browser
 * could see what the library actually drew. So at the end of init this publishes what happened
 * into `document.body.dataset`, and `render_integration_test.py` reads it back out of headless
 * chrome:
 *
 *   ready        the script ran to completion (any throw leaves it unset)
 *   page         which view this is
 *   htmx / vega  the vendored libraries are present ("missing" if a <script> did not load)
 *   charts       how many Vega-Lite specs the server embedded
 *   chartMarks   how many mark elements Vega ACTUALLY DREW — the number no text check can see,
 *                and the one that goes to zero when a spec compiles but plots nothing
 *   rows         data rows in the DOM
 *   swaps        completed HTMX swaps (0 on first paint; >0 proves an interaction re-rendered)
 *   chartError   the first embed failure, if any
 *
 * It also publishes LAYOUT measurements, so "it works on a phone" is a checkable claim rather
 * than a screenshot someone looked at once: the viewport width, whether the narrow breakpoint is
 * active, the header height, the control font size (the iOS zoom trap lives below 16px), whether
 * any element overflows the viewport AND WHICH ONE, and how many table wrappers are scrolling
 * internally (which is what must absorb a wide table instead of the page).
 *
 * The SVG renderer is not a preference — it is what makes `chartMarks` a real measurement. Under
 * the canvas renderer Vega draws pixels and leaves no DOM behind, so the count would be 0 for a
 * healthy chart and 0 for a broken one, and the best gate on this page would silently be a no-op.
 */
(function () {
  "use strict";

  var swaps = 0;

  function specNodes(root) {
    return Array.prototype.slice.call(root.querySelectorAll(".chart[data-chart]"));
  }

  /* Count what Vega put on the page. Its SVG renderer wraps each mark set in a
   * `<g class="mark-rect role-mark ...">`, so the leaf shapes under those groups are the marks
   * the spec's data produced. */
  function markCount(root) {
    return root.querySelectorAll('g[class*="role-mark"] > path, ' +
                                'g[class*="role-mark"] > rect, ' +
                                'g[class*="role-mark"] > line, ' +
                                'g[class*="role-mark"] > text').length;
  }

  function rowCount(root) {
    return root.querySelectorAll("tbody tr[data-row]").length;
  }

  /* The widest element that is NOT allowed to scroll on its own.
   *
   * `document.documentElement.scrollWidth > innerWidth` says the page scrolls sideways but not
   * WHAT did it, and on a layout of tables and charts that is the difference between a one-line
   * fix and an afternoon. So this walks the elements and names the widest offender, skipping the
   * containers whose whole job is to scroll internally (`.scroll-x`, `.chart`) and anything
   * inside one. */
  function widestOverflow() {
    var vw = document.documentElement.clientWidth, worst = null;
    var all = document.body.querySelectorAll("*");
    for (var i = 0; i < all.length; i++) {
      var el = all[i];
      if (el.closest(".scroll-x, .chart")) { continue; }
      var r = el.getBoundingClientRect();
      if (r.right > vw + 1 && (!worst || r.right > worst.right)) {
        worst = { right: r.right,
                  what: el.tagName.toLowerCase() +
                        (el.className && typeof el.className === "string"
                          ? "." + el.className.trim().split(/\s+/).join(".") : "") };
      }
    }
    return worst;
  }

  function record(err) {
    var d = document.body.dataset;
    d.page = document.body.getAttribute("data-page-name") || "?";
    d.htmx = window.htmx ? "1" : "missing";
    d.vega = (window.vega && window.vega.version) ? window.vega.version : "missing";
    d.vegaLite = (window.vegaLite && window.vegaLite.version) ? window.vegaLite.version : "missing";
    d.charts = String(specNodes(document).length);
    d.chartMarks = String(markCount(document));
    d.rows = String(rowCount(document));
    d.swaps = String(swaps);

    /* Layout measurements — the responsive claims, made checkable.
     *
     * These keys are deliberately ALL LOWERCASE. `dataset.innerW` serialises to the attribute
     * `data-inner-w`, so a camelCase key here and a naive `data["innerw"]` in the test silently
     * miss each other. Lowercase round-trips unchanged. */
    d.vw = String(window.innerWidth);
    d.docw = String(document.documentElement.scrollWidth);
    d.narrow = window.matchMedia("(max-width: 720px)").matches ? "1" : "0";
    d.headerh = String(Math.round(document.querySelector("header.top").offsetHeight));
    var ctl = document.querySelector("form.filters select, form.filters input");
    d.ctlfont = ctl ? window.getComputedStyle(ctl).fontSize : "none";
    var over = widestOverflow();
    d.overflowby = over
      ? String(Math.round(over.right - document.documentElement.clientWidth)) : "0";
    d.overflowwhat = over ? over.what : "";
    /* A table must scroll INSIDE its wrapper rather than stretch the page. Reporting whether any
       wrapper is actually scrolling distinguishes "fits" from "the rule is not wired up". */
    var wrappers = document.querySelectorAll(".scroll-x");
    var scrolling = 0;
    for (var j = 0; j < wrappers.length; j++) {
      if (wrappers[j].scrollWidth > wrappers[j].clientWidth + 1) { scrolling += 1; }
    }
    d.scrollers = String(wrappers.length);
    d.scrollingwrappers = String(scrolling);

    if (err) { d.chartError = String(err); }
    d.ready = "1";
  }

  function embedAll(root) {
    var nodes = specNodes(root).filter(function (n) { return !n.dataset.embedded; });
    if (!nodes.length) { return Promise.resolve(); }
    if (!window.vegaEmbed) {
      return Promise.reject(new Error("vega-embed did not load — the vendored bundle is missing"));
    }
    return Promise.all(nodes.map(function (node) {
      var holder = node.querySelector("script.vega-spec");
      if (!holder) { return null; }
      var spec;
      try {
        spec = JSON.parse(holder.textContent);
      } catch (e) {
        node.textContent = "chart spec is not valid JSON: " + e;
        throw e;
      }
      node.dataset.embedded = "1";
      return window.vegaEmbed(node, spec, {
        actions: false,
        renderer: "svg",          /* see the header — canvas would make chartMarks meaningless */
        config: { background: null }
      }).catch(function (e) {
        node.textContent = "chart failed to render: " + e;
        throw e;
      });
    }));
  }

  function boot() {
    embedAll(document).then(function () { record(null); },
                            function (e) { record(e && e.message ? e.message : e); });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  /* An HTMX swap brings in fresh markup that may carry its own spec. Re-embed, then refresh the
   * record so the test can assert the interaction actually re-rendered rather than just returning
   * 200 with markup nobody drew. */
  document.body.addEventListener("htmx:afterSwap", function () {
    swaps += 1;
    embedAll(document).then(function () { record(null); },
                            function (e) { record(e && e.message ? e.message : e); });
  });

  /* Copy-to-clipboard for the "take it onward" buttons.
   *
   * The tables identify the exact decision that lost a battle, but the analysis that explains it
   * (`analyze`, which loads a checkpoint) is deliberately not a web view — so the handoff is the
   * command line. Delegated from body so it keeps working on HTMX-swapped rows.
   *
   * `navigator.clipboard` needs a secure context; over plain http on a LAN address it is simply
   * absent. Falling back to a hidden textarea + execCommand keeps the button working there rather
   * than silently doing nothing, which is the failure a user would report as "the button is
   * broken". */
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy") ? resolve() : reject(new Error("execCommand refused"));
      } catch (e) { reject(e); } finally { document.body.removeChild(ta); }
    });
  }

  document.body.addEventListener("click", function (evt) {
    var btn = evt.target.closest && evt.target.closest(".copybtn");
    if (!btn) { return; }
    evt.preventDefault();
    var original = btn.textContent;
    copyText(btn.dataset.copy).then(function () {
      btn.textContent = "copied";
      document.body.dataset.copied = btn.dataset.copy;   /* the render test reads this back */
    }, function () {
      btn.textContent = "press ⌘/ctrl-C";
      /* Leave the text selected so the keyboard path still works. */
    });
    setTimeout(function () { btn.textContent = original; }, 1600);
  });

  /* A failed fetch must be visible on the page, not only in the console. */
  document.body.addEventListener("htmx:responseError", function (evt) {
    var t = evt.detail && evt.detail.target;
    if (t) { t.innerHTML = '<p class="err">request failed: ' +
                           (evt.detail.xhr ? evt.detail.xhr.status : "?") + "</p>"; }
    document.body.dataset.htmxError = "1";
  });
})();
