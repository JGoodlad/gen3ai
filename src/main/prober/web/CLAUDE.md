# CLAUDE.md — `src/main/prober/web/` (browser front end for the prober)

The prober's **only human-facing surface** (the Textual TUI is retired): a FastAPI app whose handlers
are a thin adapter over **`ProbeSession`**, the same facade the JSON CLI `query.py` uses. Server-rendered
Jinja2 + HTMX, charts as Vega-Lite specs emitted from Python, all JS vendored (no CDN, no build step).

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python -m main.prober.web models/                         # a models ROOT -> pick any run in it
python -m main.prober.web models/run_<timestamp>          # one run -> the picker offers only it
python -m main.prober.web models/ --port 6108 --job-workers 1 --open   # --open = laptop mode, no password
python -m main.prober.web models/ --impl rust             # the probes spawn the rust driver
python -m main.prober.web --openapi                       # regenerate the committed contract
python -m main.prober.web --check-openapi                 # the staleness gate (exit 1 if stale)
```

**The detail lives in `designs/prober/`** (always current, updated in the same pass as the code):

| I am about to… | Read |
|---|---|
| deploy / restart the service, touch the run picker, auth, the gate, jobs, the session cache, the OpenAPI snapshot | `designs/prober/web_service.md` |
| change what a page shows, its information flow, the critic-currency copy, `/analyze`, awareness | `designs/prober/web_pages.md` |
| change `/game`, THE battle viewer | `designs/prober/battle_view_v2.md` (field map) + `designs/prober/battle_viewer_ux_2026-10-09.md` (why) |
| touch the render test, the palettes, the phone layout, the fixture, any web test | `designs/prober/web_tests.md` |

The pre-cleanup leaf (history, not current): `designs/research_state/claude_md_archive/src_main_prober_web_CLAUDE_2026-10-10.md`.

## The map

| file | does |
|---|---|
| `__main__.py` | CLI: `--host` (loopback) · `--port` (6008) · `--job-workers` (2) · `--impl` · `--open` · `--openapi` / `--check-openapi`; rejects a nonexistent path at startup; passes the cookie-key file when a password is configured |
| `app.py` | `create_app` (templates pinned: `auto_reload = False`), every non-`/game` route, `_NAV`, the picker (`_picker`), `_client`, `battle_row`, the bounded session LRU (`_MAX_CACHED_SESSIONS`) |
| `game.py` | `/game` (THE battle viewer), its `/api/game/*` + `/partials/game/*`, and `/battle` → 307 to `/game` (`start=N` → `turn=N`) |
| `runs.py` | run enumeration + MEMBERSHIP resolution, the symlink audit, the three-group classification (`tier`, `default_run`) |
| `auth.py` | the shared password, signed cookie, persisted signing key, two rate limits |
| `gate.py` | `model_gate(next_url)` / `job_gate(next_url)` — the ONLY way a route is gated |
| `jobs.py` | the background-job registry (`ThreadPoolExecutor`, 202 + job id, polled) |
| `charts.py` | Vega-Lite specs as dicts |
| `templates/`, `static/` | Jinja2 + HTMX; `static/app.js` publishes the measured record the render test reads; `static/vendor/` the vendored JS |
| `openapi.json` | the committed route inventory (HTML routes included) |
| `fixture_run.py`, `fixture_story.py` | the one synthetic run every web test builds |

## Rules

1. **Every number rendered here comes back from a `ProbeSession` method verbatim.** This package
   reshapes, derives and rounds nothing the session did not. A page that wants a figure the session
   does not return gets a session METHOD, never a computation in a handler. `app_test.py` compares each
   JSON endpoint against a direct `ProbeSession` call.
2. 🚨 **A route that loads a checkpoint, runs the model forward or starts a job carries a gate in its
   `dependencies=[...]` — `model_gate(...)` or `job_gate(...)` from `gate.py` — and nothing else checks
   `unlocked()` by hand** (`31277fb7`). The model-free turn story stays open. The class guard
   `gate_guard_test.py` DERIVES the route set from the code (AST over `session/` + an anonymous
   behavioural sweep), so a new route is covered the moment it exists. Locked `/partials/...` answer a 200
   fragment (HTMX swallows a non-2xx); locked `/api` and pages answer 403.
3. **No client string is ever joined to a path.** A `run` (and a battle `short_id`) is only tested for
   MEMBERSHIP in the server's own listing — `runs.py`, and `app.battle_row()` one level down (without it
   a single-run instance served a sibling run's trace with a 200). Errors never echo the rejected input.
   A symlink is allowed only as a direct child of the models root; a run containing one deeper is REFUSED.
   `interior_opponent="ckpt"` (a client-supplied path) stays CLI-only.
4. **Anything an anonymous request can make grow has a bound** — the session cache
   (`_MAX_CACHED_SESSIONS`, evictees `.close()`d), the auth failure map (`_MAX_TRACKED_CLIENTS`), plus a
   GLOBAL failure cap that a login does not reset. Auth FAILS CLOSED: no password configured ⇒ probes and
   model views off.
5. **The secret never goes in argv or in the repo** — `GEN3AI_PROBER_PASSWORD_FILE` (preferred) or
   `GEN3AI_PROBER_PASSWORD`; tests use the fake `test-only-password`. The signing key is persisted
   (mode 0600, `$GEN3AI_PROBER_COOKIE_KEY_FILE`) and bound to the password, so the unlock survives a
   restart (`9c397a5e`) and changing the password still ends every session.
6. **Every model-view incompatibility is ONE typed diagnosis** (`main/prober/arch_status.py`, `kind` +
   one `plain` sentence): pages render it on first paint from `ProbeSession.model_status` (model-free)
   instead of a loader or a password prompt; JSON answers 400 `{error, kind, plain}`. Never collapse it
   to "analysis failed"; branch on `kind`, never on message text. `obs_mismatch` and
   `model_resolution.dropped_kwargs` banners are never dropped.
7. **Long work never runs in a request**: `falsify_scan`, `calibration` and the counterfactual tier go
   through `jobs.py`. Handlers are `def`, not `async def` (the session does real file IO; async would
   stall the event loop). `--impl` is a STARTUP flag because `ProbeSession` treats it as session-wide.
8. **`/game` and `/analyze` are plain GETs** (a decision and its perspective are things you link to; it
   works with JavaScript off); the model half is an HTMX fragment. A run with no traces is an EMPTY
   STATE, not a 404; an unknown battle is a 404 that never echoes the token. The information perspective
   is applied only through `engine/perspective.shown` via the `fv` macro (`perspective_guard_test.py`).
9. **Never contradict the critic's currency.** Thresholds (`v_even`, `overvalue_tau`) default to
   `None`/blank so the session resolves them per currency — a default typed into a handler is an
   EXPLICIT argument. Copy and chart axes branch on `critic_currency.is_probability`.
10. **The run picker is three groups** — current architecture · older (turn story only) · no traces
    (hidden behind `?all_runs=1`, the selected run always kept). Hiding is presentation; `resolve()` is
    still membership in every enumerated run, and listing never opens a checkpoint.
11. **The JS is vendored and the page never scrolls sideways** — wide content scrolls in its own
    container; `/game` is the one page that reflows (T29 phone layout, `82aae1aa`). Check the measured
    record, not the CSS.
12. **A template edit needs a restart** (templates are pinned to the process). The deployed service is
    replaced automatically when main's HEAD moves; an uncommitted edit under it is not.

## Commands a template prints (kept here so the freshness gate resolves their flags)

`/` links the era's offline instruments instead of re-implementing them (they were checked against the
live parsers — `main.untaught_meter` takes its refs POSITIONALLY):

```bash
python -m main.critic_gate <run> --parent <ref> --control <refs…>        # the pre-registered read
python -m main.scaffolding_gauge <run> --reliability --reliability-reweight
python -m main.untaught_meter <run> --control <continuation arms…>       # refs are POSITIONAL
python -m main.elo <run>
```

## Deployment

**Live at https://prober.g5d.io** — the `gen3ai-prober-web.service` systemd user unit (reference copy
`scripts/workstation/gen3ai-prober-web.service`) binds `127.0.0.1:6008`, `cloudflared` in front, pointed
at `models/`. No Cloudflare Access (owner, 2026-08-09: outcomes are public). A watchdog
(`scripts/workstation/prober_web_watchdog.sh`, a 2-minute timer) restarts it when `/api/health`'s
`revision` falls behind HEAD, deferring while jobs run. Ad hoc / over SSH:

```bash
python -m main.prober.web /home/goodlad/dev/gen3ai/models/
ssh -p 2222 -L 6008:localhost:6008 goodlad@workstation.g5d.io   # then http://localhost:6008
```

Detail: `designs/prober/web_service.md`; operations: `scripts/workstation/GCP_INFRASTRUCTURE.md`.

## Tests

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python3 -m pytest src/main/prober/web -q -m "not browser"   # unit + the OpenAPI snapshot gate
python3 -m pytest src/main/prober/web -q -m browser         # headless chrome, network blocked (~19 s, 2026-09-29)
```

Load-bearing guards: `gate_guard_test.py` (the unlock gate), `runs_test.py` (path confinement, written
as attacks), `auth_test.py`, `staleness_test.py` (one revision), `openapi_snapshot_test.py`,
`perspective_guard_test.py`, `render_integration_test.py` (`@integration @browser`: reads `app.js`'s
measured record — wait on `busy == "0"`, never `ready`; a chrome TIMEOUT on a busy box is a SKIP, on a
quiet one a FAILURE). A security or regression test must be PROVEN to fail with its fix reverted. Every
test builds `fixture_run.py`. What each file pins: `designs/prober/web_tests.md`.

## Gotchas

- **Register error handlers on `starlette.exceptions.HTTPException`, not FastAPI's** — an unmatched
  route raises the starlette PARENT class, so FastAPI's handler misses every 404.
- **An HTMX `<form hx-post>` sends its fields in the BODY — declare them `Form(...)`, never
  `Query(...)`** (`/falsify` and `/calibration` once silently ran at their defaults). `run` stays a
  `Query`: it rides the URL.
- **`build_trace_tree` tolerates a path with no traces** (empty tree), so `__main__` rejects a
  nonexistent path at startup rather than rendering "0 battles captured".
- **`models/` exists only in the main checkout** — pass an absolute `models/...` path when serving from a
  worktree.
- **Ports:** 6008 by default (beside TensorBoard 6006 and the arch viewer 6007); bind loopback and put a
  tunnel in front, never a public interface. A Showdown server of your own (only the probes' tooling
  might want one) binds a `9XXX` port; never touch a server you did not start. 8000/8001 are reserved and
  refused in code — nothing of ours listens there.
