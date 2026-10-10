# The prober web app's TESTS — the headless-browser gate, the palettes, the layout, the suite

Owned by this tree (always current), lifted out of `src/main/prober/web/CLAUDE.md` on 2026-10-10.
The engine/session suite is [`tests.md`](tests.md); the pages these tests pin are
[`web_pages.md`](web_pages.md) and [`web_service.md`](web_service.md).

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python3 -m pytest src/main/prober/web -q -m "not browser"   # unit + the snapshot gate
python3 -m pytest src/main/prober/web -q -m browser         # the headless-chrome tier (needs a browser)
```

## 1. What the render test actually verifies (`render_integration_test.py`, `@integration @browser`)

`static/app.js` publishes a record into `document.body.dataset` at the end of init, and the test reads
it back out of headless chrome (skipping, loudly, only when there is no browser).

**How it drives chrome** (2026-09-29): ONE browser per test module over the DevTools pipe
(`src/utils/headless_chrome.py` — stdlib only, no Playwright), a fresh **browser context** per probe, and
a wait on the page's own **`busy` == "0"** settled signal. The whole `-m browser` tier ran in **~19 s** on
a quiet box (2026-09-29). 🚨 **`ready` alone is NOT a settled signal**: it is set by the first-paint
record, which on a chart-free page lands BEFORE HTMX has issued its `load`-triggered request — swap the
wait to `ready` and `/scan`'s swap test fails (measured). 🚨 **Every launch goes through
`headless_chrome.BASE_FLAGS`, whose `--password-store=basic` is load-bearing**: without it every page
that touches chrome's network stack pays a 25 s D-Bus keyring timeout (bare chrome starts in 0.37 s).
`src/utils/headless_chrome_test.py` pins it.

**The CLICK-level tests** dispatch real mouse press/release at an element's centre (`Page.click`,
through hit testing): a tap on one of the viewer's decision-header metrics opens ITS title in a panel
under the row and a second tap closes it (phone and desktop); the `?` link lands at `#glossary`, OPENED
(`app.js` opens a collapsed `<details>` before the hash moves); the viewer's turn-rail drawer starts
closed on a phone and opens on a tap; a copy button puts its row's command on the clipboard. Each was
proven to FAIL with the behaviour broken in `app.js`, and each is written against the DOM for the
fixture trace, never the Python that produced it.

| key | proves |
|---|---|
| `ready` | the bootstrap ran to completion — any throw leaves it unset |
| `htmx` / `vega` / `vega-lite` | the vendored bundles defined their globals **with the network blocked** |
| `charts` | how many specs the server embedded |
| `chart-marks` | how many mark elements **Vega actually drew** — a spec can compile cleanly and plot nothing |
| `rows` | data rows present in the DOM (`[data-row]` table rows) |
| `monstack` | on `/game`: whether the board's two SIDES stacked (phone) or sat side by side (desktop); on `/analyze` the two mons |
| `tapmin` + `tapwhat` | the smallest side of every visible control the viewer is driven with, and WHICH one — the T29 ≥ 44px gate |
| `vpanels` | which of the viewer's panels are on the page (`rail,board,options,action,story,scouting,model`) |
| `scheme` · `bg` · `colorscheme` · `linkcolor` · `axistext` | the PALETTE, measured (§2) |
| `swaps` | completed HTMX swaps |
| `busy` | outstanding work (page load, in-flight HTMX requests, embed passes); **"0" = SETTLED** |
| `metrichelp` · `copied` | set by the tap and copy handlers — what the click-level tests read back |
| `chart-error` / `htmx-error` | a failure surfaced on the page rather than only in the console |
| `vw` · `docw` · `narrow` · `headerh` · `ctlfont` · `overflowby` + `overflowwhat` · `scrollers` + `scrollingwrappers` | the LAYOUT, measured (§3) |

The strongest single assertion is on `/scan`: nothing in its table or chart exists in the page source.
The filter form fires `hx-trigger="load"`, HTMX fetches `/partials/scan`, the fragment carries a
Vega-Lite spec, and `app.js` re-embeds it on `htmx:afterSwap`; a non-zero `chart-marks` proves the whole
chain end to end. **The SVG renderer is not a preference**: under canvas Vega leaves no DOM, so
`chart-marks` would read 0 for a healthy and a broken chart alike. `app.js` pins `renderer: "svg"`.

## 2. The two palettes, and why the dark one needs its own gate

Dark is the BASE (`:root`) and light a `prefers-color-scheme` override. **Only one palette is ever on
screen**, so review-by-screenshot covers half the stylesheet. Three defects lived in the unseen half
until the record measured it:

- **Chart text was Vega's near-black on `#16161a`.** `charts._BASE` makes the chart background
  transparent and a static spec cannot know the theme, so `app.js` applies the theme at EMBED time
  (`themeConfig` → `themed`), reading the live values from the stylesheet's custom properties.
  Spec-provided config still wins.
- **Unclassed `<a>` kept the browser default `#0000EE`** (~2.4:1 on dark, under the 4.5:1 floor). Fixed
  with a base `a { color: var(--accent) }` rule.
- **`color-scheme` was never declared**, so the browser drew its OWN widgets (`<select>`, scrollbars) in
  the light style.

**A dark screenshot FLATTERS the page**: `--force-dark-mode` also darkens UA widgets, which a real
dark-mode visitor does not get. The render test sets the media feature per context over CDP
(`Emulation.setEmulatedMedia`), which does NOT touch the widgets; `colorscheme` stays the gate. The
expected colours are written out literally — a test that derives its expectation from the stylesheet it
checks cannot catch the palette failing to apply.

## 3. Responsive layout (desktop + phone)

**The one rule: the PAGE never scrolls sideways.** Wide content scrolls inside its own container —
`.scroll-x` for tables, `.chart` for an oversized SVG. Tables of forensic numbers do **not** reflow into
cards on a phone (a `scan` row read out of column order is worse than one you scroll). **`/game` is the
one exception**: a turn is a short narrative, so it REFLOWS — three columns at ≥ 1280px (rail · decision
· the model's read), two below, one on a phone, where the rail becomes a drawer of the SAME rows and a
sticky bar carries prev / next as 44px buttons (T29, `82aae1aa`).

Under `@media (max-width: 720px)`: the sticky header drops and **the nav and the run picker fold behind
ONE labelled 44px menu button** (brand · `☰ <this page>` · unlock in one row, ≤ 80px measured, where the
wrapped tabs + picker had cost 150–230px). Opened, every tab is a 44px cell of a 4-column grid and the
picker its own row. It is a CSS-only toggle (a checkbox + its label), so it works with JavaScript off,
and the button NAMES the page you are on. Each filter control gets its own full-width row at **≥16px**
(below that, iOS zooms the page when a `<select>` takes focus).

Two things bite:

- **Vega does not wrap title text.** `charts._subtitle()` pre-wraps anything long into the list-of-lines
  form Vega-Lite accepts, and `.chart { overflow-x: auto }` catches the rest.
- **Checking the CSS is not checking the layout.** `app.js` publishes what the laid-out page *measured of
  itself*: `narrow` (the query matched), `overflowby` + `overflowwhat` (**which element** overflows),
  `ctlfont`, `headerh`, and `scrollingwrappers` ("the table fits" vs "the wrapper is not constraining it").

**The widths.** The narrow case is **500×900** (`NARROW`) — once forced by headless chrome's 500px
`--window-size` floor; the CDP driver now sets the viewport exactly, and 500 is KEPT so every layout
measurement stays comparable with its history. **The TRUE phone widths are gated beside it** (`PHONES`,
T29): every page at 360 and 430 (no sideways scroll, the folded header ≤ 80px, every measured control
≥ 44px — `tapmin`), and the battle viewer at 360, 390 and 430 (every panel present, the board stacked,
the rail drawer).

## 4. A CHROME TIMEOUT IS NEVER A SEMANTIC OUTCOME

`_dump_dom` used to let a starved chrome raise `TimeoutExpired`, which pytest recorded as a FAILURE.
Measured 2026-09-06: three failures at load 36.8 on 16 cores, every one a timeout, and on a re-run the
failing SET MOVED. A timeout now gets its own bucket:

| box | a timeout means | why |
|---|---|---|
| **quiet** (`_load_ratio()` < `_QUIET_LOAD` = **0.5**) | retry once, then hard **FAILURE** | a page that cannot SETTLE within the (scaled) 30 s bound on an idle machine is broken — a JS error before the record hook, or a request that never answered, both leave `busy` above 0 |
| **busy** | **SKIP** immediately, loudly, with `describe_contention()` | INCONCLUSIVE about the page — a different claim from "broken" |

🚨 **QUIET IS THE RAW LOAD RATIO, NOT `cpu_contention_factor`** — the first version of this fix shipped
the factor, which is clamped to a floor of 1.0 and so cannot tell a half-loaded box from an idle one
(measured at load 12.89/16 it read exactly 1.0, turning every starvation timeout into a FAILURE). The
unfloored ratio separates them: 0.24 idle vs 0.81 with a live trainer, the bar at 0.5. ⚠️ **The retry is
QUIET-ONLY** (a two-attempt-everywhere draft turned the tier into 7 tests in 50 minutes), and **the bound
was deliberately NOT inflated** — scaling does not rescue a cap.

The asymmetry is pinned in **`app_test.py`** (unmarked, stubbed chrome, milliseconds —
`test_a_timeout_on_a_QUIET_box_STILL_FAILS` and its siblings), so it runs in every tier. ⚠️ `Skipped`
derives from `BaseException`, so `pytest.raises(Exception)` does NOT catch it.

## 5. The suite, file by file

- `runs_test.py` — **path confinement**, written as a list of ATTACKS: every traversal string a visitor
  could type, plus the symlink cases (top-level followed and marked; one inside a run refuses the run; a
  pinned run cannot reach its siblings). Also the picker's classification on a constructed archive (a
  HEAD-architecture run, an older one with traces, a skeleton, a launch with no traces yet): the three
  tiers, the default-run rule, hidden runs still resolve, and that a listing never opens a zip. Its
  sibling in `app_test.py` ("the run picker, by architecture") pins the HTML: the optgroups, show-all,
  the selected run staying in the list, the model slots' first-paint reasons and the
  `{error, kind, plain}` envelope. `fixture_run.build(root, identity=…)` stamps an architecture record
  (`head_identity()`) and a valid husk zip; the default records none.
- `gate_guard_test.py` — the CLASS GUARD for the unlock gate ([`web_service.md`](web_service.md) §4).
- `auth_test.py` — fails closed with no password, the cookie is a signature and not the secret, a
  tampered expiry is rejected, throttling is per client AND globally capped, the signing key outlives a
  restart.
- **The 2026-08-09 review's fixes each have a regression test PROVEN to fail when its fix is reverted:**
  `test_a_rotating_client_identity_cannot_brute_force_the_password`, `test_the_failure_map_is_bounded`,
  `test_a_spoofed_forwarding_header_is_ignored_from_an_untrusted_peer`,
  `test_an_unreadable_subdirectory_refuses_the_run_rather_than_passing_it`,
  `test_a_pinned_run_whose_name_fails_the_enumeration_pattern_still_resolves`.
- `charts_test.py` — pure spec assertions: the field a chart plots, the fixed lever order, the
  reliability curve's identity rule, that `critic_headroom_upper_bound` is **not** stacked as a fifth lever.
- `app_test.py` — `TestClient` over `fixture_run.build()`: every endpoint against a direct `ProbeSession`
  call, the HTMX fragments, the job lifecycle (session method replaced — the re-roll machinery is
  `falsifier_integration_test.py`'s job), and that a failing probe renders as a message.
- `openapi_snapshot_test.py` — the committed contract, plus a **proof the gate fails on drift**.
- `staleness_test.py` — the one-revision contract ([`web_service.md`](web_service.md) §2).
- `game_test.py` — `/game`'s degraded paths on the synthetic fixture (see [`tests.md`](tests.md)).
- `game_viewer_test.py` — the viewer's READING contract on the fixture's story battle: ordered beats with
  their phases, "moved first", forced replacements linked to their decision, "never got to move", the
  side class + chip on every beat and effect, our options and the model's α as the same sorted
  component, locked vs unlocked, the picker never naming another battle; and the SCOUTING notes from a
  readout in the shipped shape (every truth-only judgement absent from the model view and MARKED under
  + Truth).
- `perspective_guard_test.py` — the INFORMATION-PERSPECTIVE class guard: the story battle's
  reconstruction plants facts the protocol never reveals; none may reach the `model` / `public` page,
  our private ones never the `public` page, and under `truth` every planted fact appears only MARKED.
- `render_integration_test.py` — §1–§4.

### `fixture_run.py` — the one synthetic run

Both the unit and the render test build it, so they cannot test different data. Its LOSS battles have
**no `reconstruction.json`**, so the heavy probes are expected to fail on them — and `app_test.py`
asserts that failure renders. The newest step carries ONE WIN with a real gen-3 protocol and a
reconstruction record — **the story battle** (`fixture_story.py`): every beat shape (a voluntary switch,
move order, a hazard, a mid-turn forced replacement on each side, residuals, "never got to move", the
end) plus planted hidden facts for the perspective guard; `/game` opens on it by default. Its
`opp_intent` blocks come in **two shapes on purpose** (α expects an ATTACK at one decision, a SWITCH at
another so `β`'s named mon is promoted onto the card), on only ONE battle, so the `β` path and the
heads-off path are both gated. Its `β` candidates hold **one of each name provenance**: one
`"revealed": true` and the rest with the key **ABSENT** (the literal shape of every trace before
`gen3_beta_revealed_naming_v1`).

**Its distributions follow the same one-of-each rule** (the `value_dist` arrays OLD traces carry — the
head is deleted, L1, but the awareness fold still reads archived ones): one **blind** loss that also
carries the **stall signature**, one loss it **called**, one trace with **no `value_dist` at all**, and
win-prob on exactly one battle. ⚠ **A distribution must agree with its own recorded V**: the awareness
fold denormalizes the support by a least-squares fit over `(dist mean, recorded V)` pairs, so rows are
built as a SHAPE centred on the recorded value, and a requested tail mass the mean cannot carry is a
hard error.

**`build_winprob()` is the SECOND era's fixture**, separate because the two differ in what EXISTS:
`values` equals `win_probs` exactly, no dist head, γ = 1.0 with a terminal win indicator, and
`model_config.json` carries `"critic": "winprob"` — while `build()` carries NO `critic` key ("absent means
shaped" keeps archived runs readable). Its loss is valued high and its win low, so `V − G` spans about
±0.7: above the winprob τ (≈0.083) and far below the shaped one (5.0) — the discrimination the
unreachable-threshold guard has to make.
