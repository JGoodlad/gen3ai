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
 *   rows         data rows in the DOM (table rows AND battle-replay turn cards)
 *   monstack     on the battle replay: did the two mons stack (phone) or sit side by side?
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

  /* Data rows in the DOM. Deliberately `[data-row]` rather than `tbody tr[data-row]`: the battle
   * replay's rows are turn CARDS, not table rows, and a selector that only understands tables
   * would report 0 for a perfectly rendered page. Every table row already carries the attribute,
   * so the count is unchanged everywhere else. */
  function rowCount(root) {
    return root.querySelectorAll("[data-row]").length;
  }

  /* Did the two mons on a battle-replay board STACK, or are they side by side?
   *
   * This is the one place the layout genuinely reflows rather than scrolls, so "it works on a
   * phone" here means "the boards stacked" — and that is a fact only the laid-out page knows.
   * Absent on every page that has no board. */
  function monStacking() {
    var board = document.querySelector(".board");
    if (!board) { return null; }
    var mons = board.querySelectorAll(".mon");
    if (mons.length !== 2) { return null; }
    var a = mons[0].getBoundingClientRect(), b = mons[1].getBoundingClientRect();
    return b.top >= a.bottom - 1 ? "1" : "0";
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
    d.metrics = String(document.querySelectorAll(".metric[title]").length);
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
    var stacked = monStacking();
    if (stacked !== null) { d.monstack = stacked; }

    /* THEME, measured. The page ships two palettes and only one is ever on screen, so every
     * screenshot review covers exactly half of it. These make the other half checkable:
     *   scheme     which palette the browser asked for
     *   bg         what the page actually painted (proves the palette is applied, not just defined)
     *   axistext   the fill Vega gave its axis labels — the value that is near-black by default
     *              and therefore invisible on the dark background unless themeConfig() reached it
     */
    d.scheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    d.bg = window.getComputedStyle(document.body).backgroundColor;
    /* `normal` here means the browser renders its own widgets (selects, scrollbars) in the LIGHT
     * style regardless of the page — a white dropdown on a dark page. Not visible in a
     * --force-dark-mode screenshot, which darkens UA widgets anyway, so it has to be read. */
    d.colorscheme = window.getComputedStyle(document.documentElement).colorScheme || "normal";
    var anyLink = document.querySelector("main a");
    d.linkcolor = anyLink ? window.getComputedStyle(anyLink).color : "";
    var axisLabel = document.querySelector('.chart g[class*="role-axis-label"] text')
                 || document.querySelector(".chart .role-axis text")
                 || document.querySelector(".chart text");
    d.axistext = axisLabel ? window.getComputedStyle(axisLabel).fill : "";

    if (err) { d.chartError = String(err); }
    d.ready = "1";
  }

  /* Vega-Lite's default text is near-black, and `_BASE` makes the chart background transparent so
   * it sits on the page. On the DARK palette that is black-on-#16161a: axis labels, titles and
   * legends effectively vanish. The specs cannot know the theme — they are static JSON emitted by
   * Python — so the theme is applied here, at embed time, which is the one place that knows what
   * the viewer's browser asked for.
   *
   * The values are READ BACK OUT OF THE STYLESHEET rather than restated here. app.css already
   * defines both palettes as custom properties on :root with a prefers-color-scheme override, so
   * this returns whichever one is live and there is no second copy of the colours to drift.
   */
  function themeConfig() {
    var css = getComputedStyle(document.documentElement);
    function tok(name, fallback) {
      return (css.getPropertyValue(name) || "").trim() || fallback;
    }
    var text = tok("--text", "#1d1d1b");
    var dim = tok("--dim", "#6b6b66");
    var line = tok("--line", "#dcdcd6");
    return {
      axis: {labelColor: dim, titleColor: text, gridColor: line, domainColor: line,
             tickColor: line},
      legend: {labelColor: dim, titleColor: text},
      title: {color: text, subtitleColor: dim},
      header: {labelColor: dim, titleColor: text},
      view: {stroke: null}
    };
  }

  /* Theme UNDER the spec's own config, never over it: a chart that deliberately sets something
   * (the reliability curve's identity line, the fixed lever colours) must keep winning. */
  function themed(spec) {
    var cfg = themeConfig(), own = spec.config || {}, key;
    for (key in own) {
      if (Object.prototype.hasOwnProperty.call(own, key)) {
        cfg[key] = typeof own[key] === "object" && own[key] && !Array.isArray(own[key])
          ? Object.assign({}, cfg[key] || {}, own[key])
          : own[key];
      }
    }
    spec.config = cfg;
    return spec;
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
      return window.vegaEmbed(node, themed(spec), {
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

  /* Whole-row navigation for tables whose rows ARE a destination (the battle list).
   *
   * The row's id cell is a real <a> carrying the same href, so this is pure convenience layered on
   * top of working markup — no-JS, keyboard tabbing and open-in-new-tab all keep working without
   * it. Clicks that landed on a control (the copy button, the link itself) are left alone, or the
   * copy button would navigate away instead of copying. Delegated from body so HTMX-swapped rows
   * are covered too.
   */
  document.body.addEventListener("click", function (evt) {
    var row = evt.target.closest && evt.target.closest("tr[data-href]");
    if (!row || evt.target.closest("a, button, input, select, label")) { return; }
    if (evt.defaultPrevented || evt.metaKey || evt.ctrlKey || evt.shiftKey) { return; }
    if (window.getSelection && String(window.getSelection())) { return; }  /* selecting text */
    window.location.href = row.dataset.href;
  });

  /* TAP A NUMBER, GET ITS EXPLANATION — because `title` does not exist on touch.
   *
   * Every metric on a turn card carries a `title` that says what it is and how to read it. That
   * covers desktop hover and it covers nothing else: a tooltip has no tap equivalent, and this is
   * the one view built to be read on a phone. So a tap on a `.metric` renders its OWN title into a
   * panel directly under that row — at the point of use, rather than sending the reader back to
   * the legend at the top of a 50-turn page.
   *
   * The `title` stays the single source of the text: nothing is duplicated into a data attribute
   * that could drift from the tooltip, and with JS off the hover path and the page legend both
   * still work. Tapping the same metric again closes it (a toggle, not a stack of panels).
   */
  function metricHelp(el) {
    var row = el.closest("p");
    if (!row) { return; }
    var label = (el.textContent || "").trim().split(/\s+/)[0] || "this";
    var existing = row.nextElementSibling;
    var isPanel = existing && existing.classList.contains("metric-help");
    if (isPanel && existing.dataset.forMetric === label) {
      existing.remove();                     /* same metric tapped twice = close */
      document.body.dataset.metrichelp = "";
      return;
    }
    if (isPanel) { existing.remove(); }
    var panel = document.createElement("p");
    panel.className = "metric-help";
    panel.dataset.forMetric = label;
    var strong = document.createElement("strong");
    strong.textContent = label + " ";
    panel.appendChild(strong);
    panel.appendChild(document.createTextNode(el.getAttribute("title") || ""));
    row.parentNode.insertBefore(panel, row.nextSibling);
    /* The render test reads this back to prove the whole chain ran, not merely that a class exists. */
    document.body.dataset.metrichelp = label;
  }

  document.body.addEventListener("click", function (evt) {
    var el = evt.target.closest && evt.target.closest(".metric[title]");
    if (!el) { return; }
    if (evt.target.closest("a, button")) { return; }   /* the analyze link / copy button win */
    evt.preventDefault();
    metricHelp(el);
  });

  /* A failed fetch must be visible on the page, not only in the console. */
  document.body.addEventListener("htmx:responseError", function (evt) {
    var t = evt.detail && evt.detail.target;
    if (t) { t.innerHTML = '<p class="err">request failed: ' +
                           (evt.detail.xhr ? evt.detail.xhr.status : "?") + "</p>"; }
    document.body.dataset.htmxError = "1";
  });
})();
