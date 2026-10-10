# The prober web app as a SERVICE — deployment, staleness, the run picker, access, jobs, the contract

Owned by this tree (always current), lifted out of `src/main/prober/web/CLAUDE.md` on 2026-10-10.
The leaf keeps the rules and the map; this doc keeps the reasons and the measurements. Pages and
what they render: [`web_pages.md`](web_pages.md). Tests and the browser gate: [`web_tests.md`](web_tests.md).

## 1. Deployment

**DEPLOYED at https://prober.g5d.io** — a `gen3ai-prober-web.service` systemd user unit binding
`127.0.0.1:6008`, with `cloudflared` forwarding to it, exactly like TensorBoard (`:6006`) and the
model viewer (`:6007`). The unit is reference-copied at `scripts/workstation/gen3ai-prober-web.service`;
it is pointed at **`models/`** (not one run) so a new generation needs no restart — the header carries
the run picker. **No Cloudflare Access**: the owner's decision (2026-08-09) is that this is an
open-source model whose outcomes and traces are meant to be public, same posture as `model.g5d.io`.
(The leaf once said "LOCAL ONLY — there is no g5d.io hostname"; true when written, false by
2026-08-18 — deployment facts belong next to the thing deployed.)

```bash
# ad hoc, on the workstation (the deployed instance is the systemd unit above)
python -m main.prober.web /home/goodlad/dev/gen3ai/models/
# from anywhere else without the tunnel, over SSH
ssh -p 2222 -L 6008:localhost:6008 goodlad@workstation.g5d.io   # then http://localhost:6008
```

Port **6008** by default, beside TensorBoard (6006) and the arch viewer (6007). Bind loopback and put
a tunnel in front, as `src/agents/model/arch_viewer_serve.py` does for model.g5d.io — never
bind a public interface. Operational detail: `scripts/workstation/GCP_INFRASTRUCTURE.md` → *Prober web
views*.

## 2. ONE REVISION, or a 500 — the staleness contract

**Jinja reloads a changed template from disk. Python cannot reload a changed module.** So a long-lived
server drifts into serving NEW templates against OLD code — not a stale page but a broken one.

**Measured 2026-08-18:** the service had been up **5 days**; every `/battle` for a current run returned
**HTTP 500** — `UndefinedError: 'dict object' has no attribute 'win_prob'` — because a template shipped
two days earlier read a key the running `session.py` predated. `Restart=always` never fired: **nothing
had crashed.** systemd called the unit healthy, the tunnel agreed, and the only symptom was a user
saying they could not load a game.

Two mechanisms close it, and both are needed:

1. **Templates are pinned to the process** (`templates.env.auto_reload = False`, set in `create_app`).
   A stale process serves a COHERENT old page instead of a hybrid. The cost is real and accepted:
   editing a template locally needs a restart.
2. **A watchdog replaces the process when it falls behind** — `scripts/workstation/prober_web_watchdog.sh`,
   driven by a systemd `.timer` every 2 minutes. It compares `/api/health`'s **`revision`** (the git sha
   of the source THIS PROCESS imported, captured once, never re-read) against the repo's HEAD, and
   restarts on a mismatch. It **defers while `jobs_running > 0`** (a restart kills a multi-minute
   `falsify_scan`) and **verifies the replacement came up on the new revision**, because "restarted
   successfully" into a crash-loop is the same class of lie it exists to catch.

`revision` is keyed on the SOURCE directory, not the process CWD: those coincide today and would
silently diverge the moment anyone served a worktree. Known limit: an **uncommitted** edit does not
move HEAD, so it does not trigger a restart — hand-edit under the service and you restart it yourself.

Gate: `staleness_test.py` — the pin asserted AND behaviourally proven by editing a template
mid-process, each verified to fail when `auto_reload` is put back; the watchdog driven end to end
through the real script with a stubbed `systemctl` (current / stale / no-revision / job-deferred /
unit-stopped / unreachable).

## 3. Run picker + path confinement (`runs.py`)

Point the app at a **models directory** and it enumerates the runs inside it; point it at a **single
run** and the picker offers exactly that one (the root is NOT widened to the parent — pointing at one
run must never make its siblings reachable).

**No client string is ever joined to a path.** The server enumerates the children of the models root,
and a request's `run` value is only tested for MEMBERSHIP in that enumerated set. A traversal string
cannot appear in a directory listing, so it cannot select anything. Deliberately not input
sanitisation — sanitising is a blocklist you can be wrong about; membership is an allowlist you cannot.
`runs_test.py` is written as a list of attacks (`../../etc`, `%2e%2e%2f`, `run_a/../../x`, NUL bytes,
absolute paths…), each asserting that nothing resolves.

**Symlinks are asymmetric, by the owner's decision (2026-08-09).** The launcher used to surface
worktree-isolated runs into `models/` as symlinks (since 2026-10-02 every run lands in the main
archive directly, but old links remain in the archive). So:

- a **direct child** of the models root may be a symlink; it is resolved ONCE at enumeration and the
  resolved path becomes that run's canonical root;
- **nothing deeper** may be — a run containing a symlink at any depth is REFUSED outright (loudly, not
  skipped: a link planted inside a run is the one remaining way to read a file outside it). The audit is
  a `followlinks=False` walk, cached per run (~7k entries on a real run).

Errors never echo the rejected input — a rendered message must not become an oracle for mapping the
filesystem.

### The picker is three groups (2026-10-09)

The owner: *"clean up the model selection; 99 % are irrelevant."* On the real archive (2026-10-09) 38
of 323 runs had eval traces (all v136–v143, none at HEAD) and 285 were trimmed skeletons or launches
mid-first-cycle. `list_runs()` rows carry `tier` / `model_views` / `arch` beside `has_traces`, and the
picker (`app._picker`) offers:

- **Current architecture · full model views** — traces AND the model views can run;
- **Older architecture · turn story only** — traces; every model-free view works, the model views are
  refused with the typed reason (`main/prober/arch_status.py`; an `unrecorded` run lands here too,
  labelled "architecture not recorded");
- **No eval traces · nothing to inspect** — HIDDEN behind "show all N" (`?all_runs=1`, carried through
  the picker form), except the run currently selected, which always stays in the list. Hiding is
  presentation: `resolve()` is still membership in EVERY enumerated run, so a deep link to a skeleton
  opens (as an empty state).

`has_traces` means "at least one `eval_traces/step_<N>/`" (an empty directory is not traces). **The
default run** (`RunStore.default_run`) is the newest CURRENT run with traces, else the newest run with
traces, else the newest run. Each option reads `name · v<config_version>`. Classification is two small
JSON reads per run (`model_config.json` + the newest eval manifest; **no checkpoint is opened to list**
— `runs_test` booby-traps the zip machinery), cached by mtime. `/api/runs` returns EVERY run with its
`tier` and `arch` block (clients filter). None of this touches the security properties above: no client
string is joined to a path, `_SANE_NAME` still guards enumeration, the symlink audit still runs in
`resolve()`.

## 4. Access: reading is anonymous, spending CPU is not (`auth.py`, `gate.py`)

Every read view is anonymous. What is gated is the WORK: **background probes** (`falsify_scan`,
`calibration`, the per-decision counterfactuals — minutes of re-rolls) and **anything that loads a
checkpoint or runs the model forward** (`/analyze`, `/game`'s model panels: seconds of CPU and a
resident model, with the step chosen by an anonymous query string). A public endpoint that starts
either is a free CPU-burn button.

**The route rule (2026-10-09, `31277fb7`)** — stated in the leaf. Mechanics: a locked `/api` or page
route answers a plain 403; a locked `/partials/...` HTMX target answers a 200 FRAGMENT
(`partials/locked.html` for a job, `partials/model_locked.html` for the model) because HTMX swallows a
non-2xx into a generic error. The fragment's unlock link returns to the page it was embedded in
(`HX-Current-URL`, accepted only for `/game` and `/analyze` and passed through `_safe_next`). `/game` and
`/analyze` render the model slot as that same card on first paint when locked, and the turn story
renders either way. The gated routes document `403` in the OpenAPI contract.

**The class guard `gate_guard_test.py` derives the route set from the CODE.** Leg 1 (static): the
`ProbeSession` methods that reach model-loading code are computed by AST over `session/` (calls
`_model_for` / `ProbeModel` / `capture_battle` / …, transitively through `self.`), each route handler is
walked (its lambdas and the web helpers it calls), and a handler that calls a model-reaching method or
`jobs.submit` MUST be gated in the route table — and a gated route MUST be expensive by that derivation.
Leg 2 (behavioural): every route is requested anonymously with `ProbeSession._model_for` and
`JobRegistry.submit` replaced by recorders (nothing may be recorded; the expensive routes must answer
locked), then again unlocked (each must reach a recorder — the proof the first sweep is not vacuous).
Proven to fail with a gate removed from `/api/game/readout` or `/api/analyze`.

**The secret.** One shared password, no usernames — handed out in Discord. Never in argv:

```bash
export GEN3AI_PROBER_PASSWORD_FILE=~/.config/gen3ai/prober-password   # preferred
export GEN3AI_PROBER_PASSWORD='…'                                     # or inline
python -m main.prober.web models/ --open      # laptop mode: no password, jobs open
```

On this box the secret lives at `~/.config/gen3ai/prober-password` (mode 0600) and only the *path* is
exported — from `~/.config/environment.d/60-gen3ai-prober.conf` (systemd user services),
`~/.profile` (login shells, incl. non-interactive — Ubuntu's `.bashrc` returns early when not
interactive, so `bash -lc` would not see an export there) and `~/.bashrc`. None contains the password.
**The password is never written into this repository** — tests use the obviously-fake
`test-only-password`.

**What keeps a low-entropy shared password honest:** constant-time comparison; an HMAC-signed cookie
rather than the password (HttpOnly, SameSite=Lax — also the CSRF story for the job POSTs); a signed
14-day expiry; a signing key that survives restarts but is bound to the password; **two** rate limits.
**It fails CLOSED** — with no password configured the probes and model views are off, not open.

**The unlock outlasts a restart (2026-10-09, `9c397a5e`).** `create_app(..., cookie_key_file=...)`
(passed by `__main__` whenever a password is configured; `None` — unit tests, `--open` — keeps a
per-process key) loads a 32-byte key from a mode-0600 file, `$GEN3AI_PROBER_COOKIE_KEY_FILE` else
`~/.local/state/gen3ai/prober_cookie_key`, created atomically on first start (`load_or_create_key`:
temp file + `link()`, race-safe). The SIGNING key is `HMAC(file_key, SHA256(password))`, so **changing
the shared password still ends every session** (and the key file alone forges nothing). A missing or
wrong-size file is regenerated; a file looser than 0600, owned by another user, a symlink or not a
regular file is REFUSED (`CookieKeyError`) and `Auth` falls back to a per-process key with a logged
warning (`Auth.key_persisted` False). Pinned by `auth_test.py`'s "signing key outlives a restart" block.

**Why TWO rate limits, and why `_client` is fussy about headers.** The first version keyed the throttle
on `CF-Connecting-IP` (falling back to `X-Forwarded-For`) read unconditionally; rotating the header per
request gave **500/500 guesses with the cooldown never firing**, and spoofing someone else's address
could lock THEM out. So:

1. `app._client` honours the forwarding header **only from a trusted peer** (loopback — where
   `cloudflared` sits; Cloudflare's edge overwrites a client-supplied `CF-Connecting-IP`).
   `X-Forwarded-For` is dropped entirely. When this is wrong it over-throttles rather than under.
2. `auth.py` adds a **global** cap (`_GLOBAL_MAX_FAILURES` in `_GLOBAL_WINDOW_SECONDS`) on top of the
   per-client one, so "you cannot brute-force this at request rate" does not depend on keying being
   right. A successful login deliberately does **not** reset it.

The failure map is bounded (`_MAX_TRACKED_CLIENTS`, LRU-evicted) — unbounded, 3000 spoofed identities
left 3000 permanent entries. This is a speed bump sized for "may spend some CPU", not a boundary around
private data.

## 5. The session cache is BOUNDED (`_MAX_CACHED_SESSIONS`)

One `ProbeSession` is cached per run, and a `scan` of one run costs **~430 MB** of cached summaries and
value arrays (measured on the real `models/`: 6 runs → 3.0 GB, growing monotonically). With ~80 runs on
offer then, an unbounded dict was an anonymous visitor's lever to ~35 GB on a box whose day job is
training. So `app.state.sessions` is an LRU bounded at `_MAX_CACHED_SESSIONS`, and an evicted session
gets `.close()` called (dropping its `_summaries` / `_models` caches). Same class as the auth failure
map: *anything an anonymous request can make grow must have a bound*.

## 6. Background jobs (`jobs.py`)

`falsify_scan` and `calibration` spawn a re-roll child per seed per arm (Node under the default
`--impl node`) and take minutes. Running one in a handler would stall every other page. So a submit
returns **202 + a job id**, the work runs on a small `ThreadPoolExecutor`, and the page polls
`/partials/job/{id}` every 2 s — the poll trigger is emitted only while the job is unfinished. A
failure is captured as `status="error"`, because a probe raising on a run whose traces have no
`*_reconstruction.json` sibling is an ordinary state of the data, not a 500.

`--job-workers` defaults to **2**: each concurrent job spawns its own sim processes on a box that
normally trains.

**Handlers are `def`, not `async def`, on purpose.** FastAPI runs a sync handler on a worker thread and
an async one *on the event loop*; the read-only session calls do real file IO (every trace's npz), so
making them async would let a slow `scan` stall a poll.

## 7. Which sim the probes spawn (`--impl`)

`falsify_scan` and `calibration` re-roll turns through an offline replay/search driver: **node**
(default) or **rust**. `ProbeSession` treats the choice as **session-wide** — two probes of the same run
answering under different engines would not be comparable — so this is a **startup flag**, not a query
parameter; the app caches one `ProbeSession` per run, which matches. `/api/health` reports the active
engine. (`replay_counterfactual` plays out in process on the Rust core whatever `--impl` says —
`designs/prober/sim_impl.md`.)

## 8. The OpenAPI snapshot

`openapi.json` is **committed** and gated by `openapi_snapshot_test.py`. It is generated from
`create_app(None)` — an app with no run directory — so the contract cannot vary with what is in
someone's `models/`. **The HTML routes are in the schema too**: the snapshot is the route inventory,
so a page cannot be added, renamed or deleted without the gate noticing. Regenerate deliberately:
`python -m main.prober.web --openapi`.

## 9. Architecture decisions, and why

- **FastAPI + uvicorn**, not `http.server` (what `arch_viewer_serve.py` uses): the heavy probes need to run off the request thread, and
  `/openapi.json` is a machine-readable surface that can be snapshot-committed and `--check`ed like
  `src/agents/model/delivery_graph_snapshot.json`.
- **Server-rendered Jinja2 + HTMX**, no build step, no `node_modules`. The server owns the state (the
  loaded `ProbeSession`); a SPA would add a build system to a repo whose root `package.json` has zero
  dependencies.
- **Charts are Vega-Lite specs emitted from Python** (`charts.py`) — dicts, so they diff, snapshot and
  unit-test. `_macros.html`'s `chart()` macro is the one place a spec reaches the page.
- **The JS is VENDORED** (`static/vendor/`), never CDN-linked — a lesson from
  `build_arch_viewer_render_integration_test.py`, whose strongest assertions **skip** when the CDN is
  unreachable. Here the render test maps every non-loopback host to a dead address, so a remote asset
  makes the test **fail**, not skip.

Rejected: Streamlit (rerun-per-interaction fights an expensive server-side session), React/Vite (the
repo's first JS build), Dash/Panel/Gradio.

## 10. Gotchas (detail)

- **Register error handlers on `starlette.exceptions.HTTPException`, not FastAPI's.** Starlette
  dispatches by walking `type(exc).__mro__`, and an unmatched route raises the *starlette* class — the
  PARENT of `fastapi.HTTPException`. Handling only the FastAPI one missed every 404 (`/nope` returned
  the stock `{"detail": ...}` JSON to a browser).
- **`build_trace_tree` tolerates a path with no traces** — it returns an empty tree. On a web page that
  renders as a confident "0 battles captured", which looks like a finding about the run rather than a
  typo. So `__main__` rejects a nonexistent path at **startup**; the app itself does not invent an error
  the session did not report.
