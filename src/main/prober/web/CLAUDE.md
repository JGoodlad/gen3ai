# CLAUDE.md — `src/main/prober/web/` (browser front end for the prober)

A **third sibling** over the analysis engine. `engine.py` is the analysis; the Textual TUI
(`app.py`) and the JSON CLI (`query.py`) are two independent callers of it. This is a third — a
FastAPI app whose handlers are a thin adapter over **`ProbeSession`**, the same facade `query.py`
uses.

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.prober.web models/                         # a models ROOT -> pick any run in it
python -m main.prober.web models/run_<timestamp>          # one run -> the picker offers only it
python -m main.prober.web models/ --port 6108 --job-workers 1 --open
python -m main.prober.web models/ --impl rust             # the probes spawn the rust driver
python -m main.prober.web --openapi                       # regenerate the committed contract
python -m main.prober.web --check-openapi                 # the staleness gate (exit 1 if stale)
```

**`ProbeSession` is unmodified and the TUI is untouched.** Nothing here imports `main.prober.app`.

## How to actually look at it

**It is LOCAL ONLY — there is no g5d.io hostname for it.** TensorBoard (`:6006`) and the model
viewer (`:6007`) are systemd units behind a Cloudflare tunnel; this is not. It binds `127.0.0.1`
on **6008** and you start it when you want it.

```bash
# on the workstation
export PYTHONPATH=$PYTHONPATH:src && python -m main.prober.web /home/goodlad/dev/gen3ai/models/run_<timestamp>

# from anywhere else, over the workstation SSH tunnel
ssh -p 2222 -L 6008:localhost:6008 goodlad@workstation.g5d.io   # then http://localhost:6008
```

`models/` exists only in the **main checkout**, never in a worktree — pass an absolute
`models/...` path when serving from one.

What promoting it to `prober.g5d.io` would take (and the one design question still open — a
service is pointed at ONE run) is written up in
`scripts/workstation/GCP_INFRASTRUCTURE.md` → *Prober web views*. **No Cloudflare Access**: the
owner's decision (2026-08-09) is that this is an open-source model whose outcomes and traces are
meant to be public, same posture as `model.g5d.io`.

## The one rule

**Every number rendered here comes back from a `ProbeSession` method verbatim.** This package
reshapes nothing, derives nothing, and rounds nothing the session did not already round. If a page
wants a figure the session does not return, the fix is a session method — not a computation in a
handler. Otherwise the web view and the CLI start disagreeing about the same run, which is exactly
the failure the engine/TUI/CLI split exists to prevent. `app_test.py` enforces it by comparing each
JSON endpoint against a direct `ProbeSession` call on the same run rather than against
hand-written expectations.

## Run picker + path confinement (`runs.py`)

Point the app at a **models directory** and it enumerates the runs inside it; point it at a
**single run** and the picker offers exactly that one (the root is NOT widened to the parent —
pointing at one run must never make its siblings reachable).

**The security property: no client string is ever joined to a path.** The server enumerates the
children of the models root, and a request's `run` value is only tested for MEMBERSHIP in that
enumerated set. A traversal string cannot appear in a directory listing, so it cannot select
anything. This is deliberately not input sanitisation — sanitising is a blocklist you can be wrong
about; membership is an allowlist you cannot. `runs_test.py` is written as a list of attacks
(`../../etc`, `%2e%2e%2f`, `run_a/../../x`, NUL bytes, absolute paths…) and each asserts that
nothing resolves.

**Symlinks are asymmetric, by the owner's decision (2026-08-09).** The launcher isolates each run
in a git worktree and surfaces it into `models/` as a symlink — on this box the five newest runs,
the whole current generation, are links into `.claude/worktrees/<name>/models/run_<ts>`. So:

- a **direct child** of the models root may be a symlink; it is resolved ONCE at enumeration and
  the resolved path becomes that run's canonical root;
- **nothing deeper** may be — a run containing a symlink at any depth is REFUSED outright (loudly,
  not skipped: a link planted inside a run is the one remaining way to read a file outside it).
  The audit is a `followlinks=False` walk, cached per run (~7k entries on a real run).

Errors never echo the rejected input — a rendered message must not become an oracle for mapping
the filesystem.

## Access: reading is anonymous, spending CPU is not (`auth.py`)

The model is open source and its outcomes are meant to be public, so **every read view is
anonymous**. What is gated is not the DATA but the WORK: `falsify_scan` and `calibration` spawn
Node re-rolls for minutes each beside a live trainer, so a public endpoint that starts them is a
free CPU-burn button.

One **shared password**, no usernames, no email — handed out in Discord. It unlocks the two job
endpoints and nothing else.

```bash
# the secret never goes in argv (a command line is world-readable in `ps`)
export GEN3AI_PROBER_PASSWORD_FILE=~/.config/gen3ai/prober-password   # preferred
export GEN3AI_PROBER_PASSWORD='…'                                     # or inline
python -m main.prober.web models/ --open      # laptop mode: no password, jobs open
```

On this box the secret lives at `~/.config/gen3ai/prober-password` (mode 0600) and only the *path*
is exported — from `~/.config/environment.d/60-gen3ai-prober.conf` (systemd user services, at
boot), `~/.profile` (login shells, including non-interactive) and `~/.bashrc` (interactive
non-login shells). None of those files contains the password. `.profile` is not redundant:
Ubuntu's `.bashrc` returns early when not interactive, so an export appended there is invisible to
`bash -lc`. Operational detail: `scripts/workstation/GCP_INFRASTRUCTURE.md`.

**The password is never written into this repository** — not in a doc, not as a test fixture. The
tests use an obviously-fake `test-only-password`; a committed test file publishes a secret exactly
as effectively as a committed doc does.

What keeps a low-entropy shared password honest: constant-time comparison; an HMAC-signed cookie
rather than the password itself (HttpOnly, SameSite=Lax — which is also the CSRF story for the job
POSTs); a signed expiry; a per-process signing key so a restart logs everyone out; and **two**
rate limits. **It fails CLOSED** — with no password configured the probes are off, not open, so an
operator who forgets the secret publishes a read-only site rather than a CPU-burn button.

**Why TWO rate limits, and why `_client` is fussy about headers.** The first version keyed the
throttle on `CF-Connecting-IP` (falling back to `X-Forwarded-For`) read unconditionally. An
adversarial review defeated it completely: rotating the header per request gave **500/500 guesses
with the cooldown never firing**, and spoofing someone else's address could lock THEM out. Two
changes followed, and the second is the one that matters:

1. `app._client` honours the forwarding header **only from a trusted peer** (loopback — which is
   where `cloudflared` sits, and Cloudflare's edge overwrites a client-supplied `CF-Connecting-IP`).
   `X-Forwarded-For` was dropped entirely rather than trust-gated: it is the append-friendly one.
   When this is wrong it now over-throttles rather than under-throttles.
2. `auth.py` adds a **global** cap (`_GLOBAL_MAX_FAILURES` in `_GLOBAL_WINDOW_SECONDS`) on top of
   the per-client one. A keying scheme is exactly the thing that gets quietly weakened again, so
   "you cannot brute-force this at request rate" must not depend on keying being right. The global
   cap holds even if every request presents a fresh identity, and a successful login deliberately
   does **not** reset it.

The failure map is also bounded (`_MAX_TRACKED_CLIENTS`, LRU-evicted) — it is written by anonymous
requests, and unbounded it was an attacker-controlled memory leak (3000 spoofed identities left
3000 permanent entries).

This is a speed bump, not a security boundary, and it is sized for what is behind it: "may spend
some CPU", not "may read private data".

## Information flow (why the pages are shaped the way they are)

The TUI's standing complaint was that *the information didn't flow*. These are the deliberate
answers, each pinned by a test in `app_test.py` → "usability / information flow":

- **The nav follows the documented investigation recipe**, not the order the views were written:
  `run → triage → scan → battles → falsify → calibration`. `src/main/prober/CLAUDE.md` says of
  triage *"start here for 'what next'"*, and `_NAV` now agrees with it.
- **`/` carries a "where to start" card** — the five views in recipe order, each with the QUESTION
  it answers (`VIEW_QUESTIONS` in `app.py`). Six equal tabs answering six different questions and
  saying so nowhere is the flow complaint restated.
- **A context strip on every page** names the run, its step count and its W/L. Without it the run
  being viewed lives only inside a dropdown, and a screenshot of a chart records nothing.
- **Scan rows are not dead ends.** Each row carries `data-battle-id` and copy buttons for the id
  and for the exact `python -m main.prober.query analyze <id> <inv>` line. The table names the
  decision that lost the battle; `analyze` is deliberately not a web view (checkpoint + state), so
  the honest continuation is the command.
- **First paint is populated where that is cheap.** Measured on a real run: battles **119 ms**,
  triage **436 ms**, scan **2006 ms**. Battles and triage render server-side (HTMX only fires on
  `change`); scan stays async because 2 s of white page is worse than a page that fills in — and
  its waiting state says what it is doing rather than "loading…".
- **The battles table is capped at `_BATTLE_PAGE` (200) and says so.** Uncapped it emitted 2397
  rows / 545 KB, which is a download rather than a table.
- **The run picker is grouped by generation** (`_group_runs`): 79 flat near-identical names like
  `ai_v9_06_gen5_no_concat_0809` is a scanning task, not a choice.
- **Cryptic columns explain themselves** via `title=` (`inv`, `ΔV`, `TD δ`), and `wp_coverage=0`
  is rendered as prose rather than a bare stat — it means the winning/behind split fell back to
  `V > 0`, which the project's own docs call systematically wrong, so it belongs next to the two
  categories it distorts.

## Which sim the probes spawn (`--impl`)

`falsify_scan` and `calibration` re-roll turns through an offline replay/search driver, and since
`gen3_search_driver_impl_seam_v1` that driver can be **node** (default) or **rust**.
`ProbeSession` treats the choice as **session-wide** — *"two probes of the same run answering
under different engines would not be comparable"* — so this is a **startup flag**, not a query
parameter. A per-request knob would invite exactly the incomparable mix the seam exists to
prevent, and the app caches one `ProbeSession` per run, which the session-wide reading matches
exactly.

`/api/health` reports the active engine, so a falsify result can always be traced to the sim that
produced it.

## Scope (deliberately read-only)

| View | Session call | Notes |
|---|---|---|
| `/` run | `run_summary()` | steps · per-step identity · opponents · checkpoints · γ |
| `/battles` | `battles()` | outcome / opponent / step filters |
| `/scan` | `scan()` | each battle's worst turning point, ranked (model-free) |
| `/triage` | `triage()` | failure categories ranked by recoverable win-rate |
| `/falsify` | `falsify_scan()` | the crater bracket — **a background job** |
| `/calibration` | `calibration()` | the reliability curve — **a background job** |

**Deferred, on purpose:** `analyze`, `lookahead`, `better_line`, `replay_counterfactual`. Those
load a checkpoint and hold expensive per-decision state; they are the stateful tier and want a
different session model than "one cached `ProbeSession` per run". The TUI remains the surface for
them.

## The session cache is BOUNDED (`_MAX_CACHED_SESSIONS`)

One `ProbeSession` is cached per run, and a `scan` of one run costs **~430 MB** of cached summaries
and value arrays (measured on the real `models/`: 6 runs → 3.0 GB, growing monotonically). The
picker offers **81** runs, so an unbounded dict is an anonymous visitor's lever to ~35 GB — on a box
whose day job is training.

So `app.state.sessions` is an LRU bounded at `_MAX_CACHED_SESSIONS`, and an evicted session gets
`.close()` called (dropping its `_summaries` / `_models` caches) rather than being left to the
garbage collector's discretion. Same defect class as the auth failure map, found the same way:
*anything an anonymous request can make grow must have a bound*.

## Architecture decisions, and why

- **FastAPI + uvicorn**, not `http.server` (which is what `arch_viewer_serve.py` uses). Two
  reasons, both load-bearing: the heavy probes need to run off the request thread, and
  `/openapi.json` is a machine-readable description of the surface that can be snapshot-committed
  and `--check`ed like `delivery_graph_snapshot.json`.
- **Server-rendered Jinja2 + HTMX**, no build step, no `node_modules`. The server owns the state
  (the loaded `ProbeSession`), so the client has nothing to hold; a SPA would add a build system to
  a repo whose root `package.json` has zero dependencies.
- **Charts are Vega-Lite specs emitted from Python** (`charts.py`) — dicts, so they diff, snapshot
  and unit-test. No template writes plotting code; `_macros.html`'s `chart()` macro is the one
  place a spec reaches the page.
- **The JS is VENDORED** (`static/vendor/`), never CDN-linked. This is a direct lesson from
  `build_arch_viewer_render_integration_test.py`, whose strongest assertions **skip** when the CDN
  is unreachable — offline, the best gate in that suite is a no-op. Here the render test launches
  chrome with every non-loopback host mapped to a dead address, so a remote asset would make the
  test **fail**, not skip.

Rejected: Streamlit (rerun-per-interaction fights an expensive server-side session),
React/Vite (the repo's first JS build), Dash/Panel/Gradio.

## Background jobs (`jobs.py`)

`falsify_scan` and `calibration` spawn Node per re-roll and take minutes. Running one in a handler
would stall the event loop and therefore every other page. So a submit returns **202 + a job id**,
the work runs on a small `ThreadPoolExecutor`, and the page polls `/partials/job/{id}` every 2s —
the poll trigger is emitted only while the job is unfinished, so a finished page stops hitting the
server. A failure is captured as `status="error"`, because a probe raising on a run whose traces
have no `*_reconstruction.json` sibling is an ordinary state of the data, not a 500.

`--job-workers` defaults to **2**. Each concurrent job spawns its own Node processes and the box
normally carries a live training run — see the contention notes in the root `CLAUDE.md`.

**Handlers are `def`, not `async def`, on purpose.** FastAPI runs a sync handler on a worker thread
and an async one *on the event loop*; the read-only session calls do real file IO (they open every
trace's npz), so making them async would let a slow `scan` stall a poll.

## The OpenAPI snapshot

`openapi.json` is **committed** and gated by `openapi_snapshot_test.py`. It is generated from
`create_app(None)` — an app with no run directory — so the contract cannot vary with what happens
to be in someone's `models/`. **The HTML routes are in the schema too**: the snapshot is the route
inventory, and hiding the pages would let one be added, renamed or deleted without the gate
noticing.

Regenerate deliberately: `python -m main.prober.web --openapi`.

## What the render test actually verifies

`static/app.js` publishes a record into `document.body.dataset` at the end of init, and
`render_integration_test.py` reads it back out of headless chrome (skipping, loudly, only when
there is no browser):

| key | proves |
|---|---|
| `ready` | the bootstrap ran to completion — any throw leaves it unset |
| `htmx` / `vega` / `vega-lite` | the vendored bundles defined their globals **with the network blocked** |
| `charts` | how many specs the server embedded |
| `chart-marks` | how many mark elements **Vega actually drew** — a spec can compile cleanly and plot nothing |
| `rows` | data rows present in the DOM |
| `swaps` | completed HTMX swaps |
| `chart-error` / `htmx-error` | a failure surfaced on the page rather than only in the console |
| `vw` · `docw` · `narrow` · `headerh` · `ctlfont` · `overflowby` + `overflowwhat` · `scrollers` + `scrollingwrappers` | the LAYOUT, measured (see below) |

The strongest single assertion is on `/scan`: nothing in its table or chart exists in the page
source. The filter form fires `hx-trigger="load"`, HTMX fetches `/partials/scan`, the fragment
carries a Vega-Lite spec, and `app.js` re-embeds it on `htmx:afterSwap`. A non-zero
`chart-marks` there proves the whole chain — fragment, swap, re-embed, draw — end to end.

**The SVG renderer is not a preference.** Under the canvas renderer Vega leaves no DOM behind, so
`chart-marks` would read 0 for a healthy chart and 0 for a broken one, and the best gate on the
page would silently be a no-op. `app.js` pins `renderer: "svg"` for exactly that reason.

## Responsive layout (desktop + phone), and how it is gated

**The one rule: the PAGE never scrolls sideways.** Wide content scrolls inside its own container —
`.scroll-x` for tables, `.chart` for an oversized SVG. Everything else follows from that.

These are tables of forensic numbers, so the phone answer is deliberately **not** to reflow every
table into cards: a `scan` row read out of column order is worse than one you scroll. Under
`@media (max-width: 720px)` the layout instead: drops the sticky header and truncates the run path,
**wraps** the nav (never a horizontal strip — the arch viewer shipped that strip and five of its
six controls were invisible), gives each filter control its own full-width row at **≥16px** (below
that, iOS zooms the page when a `<select>` takes focus and does not zoom back out), and tightens
the paddings.

Two things bite that are not obvious:

- **Vega does not wrap title text.** A 150-character subtitle renders as one ~1000px line and drags
  the chart past its container. `charts._subtitle()` pre-wraps anything long into the list-of-lines
  form Vega-Lite accepts, and `.chart { overflow-x: auto }` catches whatever is left.
- **Checking the CSS is not checking the layout.** A media query can be present and still lose on
  specificity. So `app.js` publishes what the laid-out page *measured of itself* and the tests read
  that: `narrow` (the query actually matched), `overflowby` + `overflowwhat` (**which element**
  overflows, so the failure carries its own fix), `ctlfont`, `headerh`, and
  `scrollingwrappers` — the last distinguishing "the table fits" from "the wrapper is not
  constraining it and the page is absorbing the width".

**Headless chrome clamps its window to 500px wide** — `--window-size=390,844` silently renders a
500px page (measured, both `--headless` and `--headless=new`). So the narrow case is **500×900**,
labelled as such rather than claiming an iPhone width it is not testing; the 720px breakpoint sits
comfortably above it, so every phone rule is exercised. The arch viewer gets a true 390 with a
same-origin `file://` iframe; that does not transfer here (these pages are served over http, and
forcing it with `--disable-web-security --user-data-dir` hung chrome outright).

## Tests

```bash
export PYTHONPATH=$PYTHONPATH:src
python3 -m pytest src/main/prober/web -q                       # unit + the snapshot gate
python3 -m pytest src/main/prober/web -q -m integration        # + headless chrome (needs a browser)
```

- `runs_test.py` — **path confinement**, written as a list of ATTACKS rather than behaviour
  checks: every traversal string a visitor could type, plus the symlink cases (top-level followed
  and marked; one inside a run refuses the run; a pinned run cannot reach its siblings). These
  should read as boring — that is the point of membership-over-sanitisation.
- `auth_test.py` — the gate's properties: fails closed with no password, the cookie is a
  signature and not the secret, a tampered expiry is rejected, throttling is per client AND
  globally capped.

**The 2026-08-09 review's four fixes each have a regression test, and each was PROVEN to fail when
its fix is reverted** (a security test that passes either way is worse than none — the reversion
was applied to a copy of the tree and the matching test confirmed red):
`test_a_rotating_client_identity_cannot_brute_force_the_password`,
`test_the_failure_map_is_bounded`,
`test_a_spoofed_forwarding_header_is_ignored_from_an_untrusted_peer`,
`test_an_unreadable_subdirectory_refuses_the_run_rather_than_passing_it`,
`test_a_pinned_run_whose_name_fails_the_enumeration_pattern_still_resolves`.
- `charts_test.py` — pure spec assertions: the field a chart plots, the fixed lever order, that
  the reliability curve keeps its identity rule, that the derived
  `critic_headroom_upper_bound` is **not** stacked as a fifth lever.
- `app_test.py` — `TestClient` over `fixture_run.build()`: every endpoint against a direct
  `ProbeSession` call, the HTMX fragments, the job lifecycle (with the session method replaced —
  the re-roll machinery is `falsifier_integration_test.py`'s job), and that a failing probe renders
  as a message.
- `openapi_snapshot_test.py` — the committed contract, plus a **proof the gate fails on drift**
  (a `--check` that always passes is worse than none).
- `render_integration_test.py` — `@integration`; the headless-browser gate above.
- `fixture_run.py` — the synthetic run both the unit and the render test build, so they cannot end
  up testing different data. It has **no `reconstruction.json`**, so the heavy probes are expected
  to fail on it — and `app_test.py` asserts that failure renders.

## Gotchas

- **Register error handlers on `starlette.exceptions.HTTPException`, not FastAPI's.** Starlette
  dispatches by walking `type(exc).__mro__`, and an unmatched route raises the *starlette* class —
  the PARENT of `fastapi.HTTPException`, not a subclass. Handling only the FastAPI one misses every
  404 (it did: `/nope` returned the stock `{"detail": ...}` JSON to a browser).
- **`build_trace_tree` tolerates a path with no traces** — it returns an empty tree rather than
  raising. On the CLI that reads as an empty JSON object; on a web page it renders as a confident
  "0 battles captured", which looks like a finding about the run rather than a typo. So
  `__main__` rejects a nonexistent path at **startup**; the app itself does not invent an error the
  session did not report.
- **`models/` is gitignored and lives only in the main checkout**, not in a worktree — pass an
  absolute `models/...` path when serving from one.
- Port **6008** by default, beside tensorboard (6006) and the arch viewer (6007). Bind loopback and
  put a tunnel in front, as `arch_viewer_serve.py` does for model.g5d.io — do not bind a public
  interface, and never touch **:8001** (live training).
