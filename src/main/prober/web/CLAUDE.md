# CLAUDE.md — `src/main/prober/web/` (the prober's browser front end)

The prober's only human-facing surface: FastAPI handlers over **`ProbeSession`** (the facade the JSON
CLI uses), server-rendered Jinja2 + HTMX, Vega-Lite specs emitted from Python, all JS vendored.
The engine rules are the parent leaf's (`src/main/prober/CLAUDE.md`).

```bash
python -m main.prober.web models/ --port 6108 --job-workers 1 --open   # --open: laptop mode, no password
python -m main.prober.web --check-openapi    # exit 1 if openapi.json is stale; --openapi regenerates it
```

| I am about to… | Read |
|---|---|
| deploy or restart the service; the run picker, auth, the gate, jobs, the session cache, OpenAPI | `designs/prober/web_service.md` |
| change what a page shows, the critic-currency copy, `/analyze` | `designs/prober/web_pages.md` |
| change `/game` | `designs/prober/battle_view_v2.md` + `battle_viewer_ux_2026-10-09.md` |
| touch any web test, the render test, palettes, the phone layout, the fixture | `designs/prober/web_tests.md` |

## Rules

1. **Every number rendered comes back from a `ProbeSession` method verbatim.** A page that needs a
   figure the session does not return gets a new session method, never a computation in a handler
   (`app_test.py` compares each JSON endpoint with a direct session call).
2. **A route that loads a checkpoint, runs the model or starts a job carries `model_gate(...)` or
   `job_gate(...)` in its `dependencies=[...]`**, and nothing checks `unlocked()` by hand.
   `gate_guard_test.py` derives the route set from the code, so a new route is covered at once. A
   locked `/partials/...` answers a 200 fragment (HTMX swallows a non-2xx); a locked `/api` or page, 403.
3. **No client string is ever joined to a path.** A `run` or battle `short_id` is only tested for
   membership in the server's own listing (`runs.py`, and `battle_row` inside `create_app`); errors never echo the input.
   A symlink is allowed only as a direct child of the models root. `interior_opponent="ckpt"` (a
   client-supplied path) stays CLI-only.
4. Anything an anonymous request can grow is bounded (the session LRU, the auth failure map, a
   global failure cap a login does not reset). Auth fails closed: no password configured ⇒ probes and
   model views off.
5. **The secret never goes in argv or the repo**: `GEN3AI_PROBER_PASSWORD_FILE` (preferred) or
   `GEN3AI_PROBER_PASSWORD`; tests use the fake `test-only-password`.
6. A model-view failure renders the `ArchDriftError` diagnosis on first paint, instead of a loader or a
   password prompt (JSON: 400 `{error, kind, plain}`), and the `obs_mismatch` and `model_resolution.dropped_kwargs` banners are never dropped.
7. **Long work never runs in a request**: `falsify_scan`, `calibration` and the counterfactual tier go
   through `jobs.py`. Handlers are `def`, not `async def` (the session does blocking file IO).
8. `/game` and `/analyze` are plain GETs that work with JavaScript off; the model half is an HTMX
   fragment. A run with no traces is an empty state, not a 404. The information perspective is applied
   only through `engine/perspective.shown` via the `fv` macro (`perspective_guard_test.py`).
9. Thresholds (`v_even`, `overvalue_tau`) default to `None` so the session resolves them per critic
   currency; a default typed into a handler becomes an explicit argument. Copy and axes branch on
   `critic_currency.is_probability`.
10. The page never scrolls sideways: wide content scrolls in its own container. Check the render
    test's measured record, not the CSS.
11. A template edit needs a server restart (templates are pinned to the process).

A security or regression test must be shown to fail with its fix reverted. Every web test builds
`fixture_run.py`.

## Gotchas

- Register error handlers on `starlette.exceptions.HTTPException`, not FastAPI's: an unmatched route
  raises the starlette parent class, so FastAPI's handler misses every 404.
- An HTMX `<form hx-post>` sends its fields in the body: declare them `Form(...)`, never `Query(...)`,
  or the handler silently runs at its defaults. `run` stays a `Query`.
- The commands `/` prints (`templates/index.html`) are checked by no gate; re-run each one's `--help`
  when you edit that card.
