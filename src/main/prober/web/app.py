"""The FastAPI app — a THIN adapter over `ProbeSession`, plus server-rendered Jinja2 + HTMX.

THE ONE RULE. Every number rendered here comes back from a `ProbeSession` method verbatim. This
module reshapes nothing, derives nothing, and rounds nothing that the session did not already
round. If a page wants a figure the session does not return, the fix is a session method, not a
computation here — otherwise the web view and the CLI would start disagreeing about the same run,
which is precisely the failure the engine/TUI/CLI split exists to prevent.

WHY SYNC HANDLERS. The read-only session calls are file IO (they open every trace's npz), and
FastAPI runs a `def` handler on a worker thread while an `async def` one runs ON the event loop.
So these are deliberately `def`: a slow `scan` over a thousand-battle run must not stall the page
that is polling a job. The genuinely expensive probes (`falsify_scan`, `calibration` — minutes,
Node per re-roll) do not even get a thread per request; they go through `jobs.JobRegistry`.

WHY THE HTML ROUTES ARE IN THE OPENAPI SCHEMA. `/openapi.json` is snapshot-committed and gated by
`openapi_snapshot_test.py`, so the snapshot is the route inventory. Hiding the HTML routes would
mean adding, renaming or deleting a page slipped past the gate — the drift the gate exists to
catch. They are documented as `text/html` responses, which is what they are.

TWO POLICIES LIVE OUTSIDE THIS FILE, on purpose:
  * WHICH RUNS are openable, and the path confinement that decides it — `runs.py`.
  * WHO MAY SPEND CPU — `auth.py`. Reading is anonymous; the shared password gates only the
    job endpoints.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from main.prober.web import charts
from main.prober.web.auth import COOKIE, Auth
from main.prober.web.jobs import JobRegistry
from main.prober.web.runs import RunAccessError, RunStore

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES = os.path.join(_HERE, "templates")
_STATIC = os.path.join(_HERE, "static")

TITLE = "gen3ai prober"
DESCRIPTION = (
    "Read-only browser views over the prober's analysis engine, adapted from `ProbeSession` — "
    "the same facade `python -m main.prober.query` uses. No analysis lives here. Reading is "
    "anonymous; the two expensive probe endpoints require the shared password."
)
VERSION = "1.1.0"


# --- response models -------------------------------------------------------------------
# Only the shapes this module actually OWNS are modelled. The session's results are deep,
# heterogeneous dicts documented by `src/main/prober/CLAUDE.md`; declaring a Pydantic model for
# them would put a second, drifting spec of the engine's output in the contract. `dict` is the
# honest annotation, and the snapshot still pins routes, parameters and status codes.

class Health(BaseModel):
    ok: bool
    models_root: "str | None" = Field(None, description="The root the run picker enumerates.")
    n_runs: int = Field(..., description="Selectable runs under that root.")
    jobs_unlocked: bool = Field(..., description="May THIS request start an expensive probe?")
    auth_required: bool = Field(..., description="False when started with --open.")
    impl: str = Field(..., description="Offline replay/search engine the probes spawn: node | rust.")
    version: str


class RunRow(BaseModel):
    name: str
    has_traces: bool
    linked: bool = Field(..., description="Reached via an owner-placed symlink in the models root.")
    mtime: float


class JobRef(BaseModel):
    id: str
    kind: str
    status: str = Field(..., description="pending | running | done | error")
    done: bool


class JobState(JobRef):
    params: dict
    error: "str | None" = None
    result: "dict | None" = None
    submitted: float
    started: "float | None" = None
    finished: "float | None" = None
    elapsed_sec: float


class _NoBattles(HTTPException):
    """The run is fine, it just has no captured traces yet.

    Distinct from "no such battle" (a bad token, a real 404) because it is the ORDINARY state of the
    newest run — the one the app opens by default — until its first eval cycle writes a trace. As a
    bare 404 the two battle-addressed pages greeted a fresh run with what looked like a broken link;
    the page handlers turn this into an empty state instead, and only the JSON API still sees a
    status code."""

    def __init__(self) -> None:
        super().__init__(status_code=404, detail="this run has captured no battle traces yet")


def create_app(root: "str | None" = None, *, max_job_workers: int = 2,
               password: "str | None" = None, open_access: bool = False,
               impl: str = "node") -> FastAPI:
    """Build the app.

    `root` is a models directory (the picker enumerates its runs) or a single run directory (the
    picker then offers exactly that one — pointing at a run must never make its siblings
    reachable). `root=None` builds a fully-formed app with no data, which is what `openapi.json`
    is generated from, so the committed contract cannot depend on a machine's `models/`.
    """
    app = FastAPI(title=TITLE, description=DESCRIPTION, version=VERSION)
    app.state.root = root
    app.state.runs = RunStore(root) if root else None
    app.state.auth = Auth(password, open_access=open_access)
    # WHICH offline replay/search engine the re-roll-backed probes spawn. `ProbeSession` treats
    # this as SESSION-WIDE on purpose ("two probes of the same run answering under different
    # engines would not be comparable"), so it is a startup flag here rather than a query param —
    # a per-request knob would invite exactly the incomparable mix the seam exists to prevent.
    app.state.impl = impl
    # LRU, NOT a plain dict — see _MAX_CACHED_SESSIONS. A scan of one run caches ~430 MB, and an
    # anonymous visitor can walk every run in the picker.
    app.state.sessions = OrderedDict()            # resolved run path -> ProbeSession
    app.state.session_lock = threading.Lock()
    app.state.jobs = JobRegistry(max_workers=max_job_workers)
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    templates = Jinja2Templates(directory=_TEMPLATES)

    # -- plumbing ------------------------------------------------------------------------

    def store() -> RunStore:
        if app.state.runs is None:
            raise HTTPException(
                status_code=503,
                detail="no models directory: start with `python -m main.prober.web <models_dir>`")
        return app.state.runs

    def pick(name: "str | None") -> str:
        """The run this request is about. A bad name is a 404 with a message safe to render —
        never an echo of what was asked for."""
        runs = store()
        try:
            return runs.resolve(name or runs.default_run())
        except RunAccessError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def session(run_path: str):
        """The cached `ProbeSession` for one resolved run, as a BOUNDED LRU.

        Built lazily and once per run — constructing it walks the trace tree, and every page would
        otherwise pay that walk. But the cache must also be bounded: a `scan` of one run leaves
        ~430 MB of cached summaries and value arrays behind (measured on the real `models/`:
        6 runs → 3.0 GB, monotonic), and the picker offers 81 runs, so an unbounded dict hands an
        anonymous visitor a ~35 GB lever on a box that is training. Evicting closes the session so
        its caches are dropped now rather than whenever the collector notices.
        """
        sessions = app.state.sessions
        with app.state.session_lock:
            got = sessions.get(run_path)
            if got is not None:
                sessions.move_to_end(run_path)            # most-recently-used last
                return got
            from main.prober.session import ProbeSession
            try:
                got = ProbeSession(run_path, impl=app.state.impl)
            except Exception as exc:              # noqa: BLE001 — a broken run is a 400, not a 500
                raise HTTPException(
                    status_code=400,
                    detail=f"cannot open that run: {type(exc).__name__}: {exc}") from exc
            sessions[run_path] = got
            while len(sessions) > _MAX_CACHED_SESSIONS:
                _, evicted = sessions.popitem(last=False)  # least-recently-used
                try:
                    evicted.close()
                except Exception:                 # noqa: BLE001 — eviction must never fail a request
                    pass
        return got

    def battle_row(sess, token: "str | None") -> dict:
        """The battle this request is about, as a row the SESSION ITSELF enumerated.

        THE SAME RULE AS `runs.py`, one level down: a client's `battle` value is only tested for
        MEMBERSHIP in the run's own battle listing — it is never joined to a path and never handed
        to the session as one. That distinction is load-bearing here, because
        `ProbeSession._battle` falls back to `build_trace_tree(battle_id)` for an unknown id, which
        would happily open a `*_summary.json` belonging to a DIFFERENT run (or anywhere else the
        process can read). So the URL carries the opaque `short_id`
        (`step_<N>/<opponent>/<outcome>_<idx>`), the server matches it against its own listing, and
        the PATH it then passes to the session is one the server produced.

        A miss is a 404 that does not echo the token — a rendered message must not become an oracle.
        """
        rows, err = guarded(lambda: sess.battles())
        if err:
            raise HTTPException(status_code=400, detail=err)
        if not rows:
            # A run with no traces is a NORMAL state, not a bad URL — and it is the state of the
            # run this app opens by DEFAULT, because the default is the newest run and a fresh one
            # has captured nothing until its first eval cycle. The page handlers catch this and
            # render an empty state; only the API surfaces it as a status code.
            raise _NoBattles()
        if not token:
            # The NEWEST checkpoint's first battle. `battles()` is ordered by step ASCENDING, so
            # `rows[0]` is the oldest eval cycle in the run — landing a visitor on a battle played
            # by a months-old checkpoint, which is never what "open the replay" means.
            return _newest_first(rows)[0]
        for r in rows:
            if r["short_id"] == token:
                return r
        raise HTTPException(status_code=404, detail="no such battle in this run")

    def unlocked(request: Request) -> bool:
        return app.state.auth.unlocked(request.cookies.get(COOKIE))

    def require_unlocked(request: Request) -> None:
        if not unlocked(request):
            raise HTTPException(
                status_code=403,
                detail="this probe spends minutes of CPU beside a live training run — "
                       "unlock it with the shared password at /login")

    def shell(request: Request, run_name: "str | None") -> dict:
        rows = store().list_runs() if app.state.runs else []
        return {"nav": _NAV, "runs": rows, "run_groups": _group_runs(rows), "run": run_name,
                "models_root": app.state.root,
                "unlocked": unlocked(request),
                "auth_required": app.state.auth.required,
                "auth_configured": app.state.auth.configured}

    def page(request: Request, template: str, name: str, run_name: "str | None",
             **ctx) -> HTMLResponse:
        # `run_meta` is what the persistent context strip renders. Without it a screenshot of a
        # chart carries no record of which run it came from.
        summary = ctx.get("summary") or ctx.get("data")
        meta = None
        if summary:
            meta = {"run": run_name, "n_steps": summary.get("n_steps"),
                    "battles": (summary.get("totals") or {}).get("battles"),
                    "wins": (summary.get("totals") or {}).get("win"),
                    "losses": (summary.get("totals") or {}).get("loss")}
        return templates.TemplateResponse(
            request=request, name=template,
            context={"page_name": name, "run_meta": meta, "views": VIEW_QUESTIONS,
                     **shell(request, run_name), **ctx})

    def fragment(request: Request, template: str, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name=template, context=ctx)

    def guarded(fn):
        """Run a session call, turning its failure into a rendered message rather than a 500.

        A probe raising on a run with no `reconstruction.json` siblings, or on an empty
        `eval_traces/`, is an ordinary state of the data. The page must say so."""
        try:
            return fn(), None
        except HTTPException:
            raise
        except Exception as exc:                  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    # -- JSON API: meta ------------------------------------------------------------------

    @app.get("/api/health", response_model=Health, tags=["meta"],
             summary="Liveness, how many runs are selectable, and whether jobs are unlocked")
    def api_health(request: Request) -> Health:
        n = len(app.state.runs.list_runs()) if app.state.runs else 0
        return Health(ok=True, models_root=app.state.root, n_runs=n,
                      jobs_unlocked=unlocked(request), impl=app.state.impl,
                      auth_required=app.state.auth.required, version=VERSION)

    @app.get("/api/runs", response_model=list[RunRow], tags=["read-only"],
             summary="Selectable runs, newest first — the picker's contents")
    def api_runs() -> list:
        return store().list_runs()

    # -- JSON API: read-only views -------------------------------------------------------

    @app.get("/api/run", tags=["read-only"], response_model=dict,
             summary="ProbeSession.run_summary() — steps, per-step identity, opponents, checkpoints")
    def api_run(run: "str | None" = Query(None, description="run name; default = newest")) -> dict:
        return session(pick(run)).run_summary()

    @app.get("/api/battles", tags=["read-only"], response_model=list,
             summary="ProbeSession.battles() — the filtered battle list")
    def api_battles(
        run: "str | None" = Query(None),
        outcome: "str | None" = Query(None, pattern="^(win|loss)$"),
        opponent: "str | None" = Query(None),
        step: "int | None" = Query(None),
    ) -> list:
        return session(pick(run)).battles(outcome=outcome, opponent=opponent, step=step)

    @app.get("/api/battle-turns", tags=["read-only"], response_model=dict,
             summary="ProbeSession.battle_turns() — one battle's decisions grouped by game turn")
    def api_battle_turns(
        run: "str | None" = Query(None),
        battle: "str | None" = Query(
            None, description="the battle's short_id (step_<N>/<opponent>/<outcome>_<idx>); "
                              "default = the run's first captured battle"),
    ) -> dict:
        sess = session(pick(run))
        return sess.battle_turns(battle_row(sess, battle)["id"])

    @app.get("/api/analyze", tags=["read-only"], response_model=dict,
             summary="ProbeSession.analyze() — one decision, fully analyzed (LOADS THE CHECKPOINT)")
    def api_analyze(
        run: "str | None" = Query(None),
        battle: "str | None" = Query(
            None, description="the battle's short_id (step_<N>/<opponent>/<outcome>_<idx>); "
                              "default = the newest checkpoint's first captured battle"),
        inv: int = Query(0, ge=0, description="invocation index — the Nth recorded decision"),
    ) -> dict:
        # THE ONE MODEL-LOADING VIEW. It raises `ArchDriftError` on any run that is not at the
        # current architecture (measured 2026-08-13: 79/79 archived runs), which is an ordinary
        # state of the data, not a server fault — so it comes back as the usual `{"error": ...}`
        # envelope carrying the diagnosis VERBATIM (multi-line, ending in the `git checkout` to
        # re-probe from), which is the same text the CLI prints and the page renders.
        sess = session(pick(run))
        row = battle_row(sess, battle)
        data, err = guarded(lambda: sess.analyze(row["id"], inv))
        if err:
            raise HTTPException(status_code=400, detail=err)
        return data

    @app.get("/api/scan", tags=["read-only"], response_model=list,
             summary="ProbeSession.scan() — each battle's worst turning point, ranked (model-free)")
    def api_scan(
        run: "str | None" = Query(None),
        outcome: "str | None" = Query(None, pattern="^(win|loss)$"),
        opponent: "str | None" = Query(None),
        step: "int | None" = Query(None),
        metric: str = Query("value_drop", pattern="^(value_drop|td_residual)$"),
        limit: "int | None" = Query(None, ge=1, le=1000),
    ) -> list:
        return session(pick(run)).scan(outcome=outcome, opponent=opponent, step=step,
                                       metric=metric, limit=limit)

    @app.get("/api/triage", tags=["read-only"], response_model=dict,
             summary="ProbeSession.triage() — failure categories ranked by recoverable win-rate")
    def api_triage(
        run: "str | None" = Query(None),
        step: "int | None" = Query(None),
        opponent: "str | None" = Query(None),
        wp_even: float = Query(0.5, description="P(win) threshold for the winning/behind split"),
        v_even: float = Query(0.0, description="V fallback threshold when no win-prob head"),
    ) -> dict:
        return session(pick(run)).triage(step=step, opponent=opponent,
                                         wp_even=wp_even, v_even=v_even)

    @app.get("/api/awareness", tags=["read-only"], response_model=dict,
             summary="ProbeSession.awareness_scan() — the 'did it KNOW?' verdicts (model-free)")
    def api_awareness(
        run: "str | None" = Query(None),
        # An EMPTY outcome means "every battle", and it has to be reachable: the quantile-coverage
        # half of this probe is only comparable to the published baseline unfiltered (the loss
        # filter biases PIT low by construction, which the result's own caveats state).
        outcome: "str | None" = Query("loss", pattern="^(win|loss|draw|)$"),
        opponent: "str | None" = Query(None),
        step: "int | None" = Query(None),
        lead_bar: int = Query(5, ge=0, description="turns of warning that count as 'aware'"),
        cap_turn: int = Query(240, ge=1, description="last decision turn ≥ this = a CAP loss"),
        stall_bar: float = Query(0.25, ge=0.0, le=1.0,
                                 description="tail-divergence that counts as the stall signature"),
    ) -> dict:
        return session(pick(run)).awareness_scan(
            outcome=outcome or None, opponent=opponent, step=step,
            lead_bar=lead_bar, cap_turn=cap_turn, stall_bar=stall_bar)

    # -- JSON API: the expensive probes, as jobs (PASSWORD REQUIRED) ----------------------

    @app.post("/api/jobs/falsify-scan", tags=["jobs"], response_model=JobRef, status_code=202,
              summary="Submit ProbeSession.falsify_scan() — minutes; needs the shared password")
    def api_job_falsify(
        request: Request,
        run: "str | None" = Query(None),
        outcome: str = Query("loss", pattern="^(win|loss)$"),
        opponent: "str | None" = Query(None),
        step: "int | None" = Query(None),
        limit: int = Query(20, ge=1, le=200),
        worst: int = Query(2, ge=1, le=10),
        seeds: int = Query(32, ge=2, le=200),
        alts: int = Query(2, ge=1, le=8),
        concurrency: int = Query(1, ge=1, le=8,
                                 description="battles falsified in parallel; each re-roll spawns "
                                             "Node, so raise it only on an idle box"),
    ) -> JobRef:
        require_unlocked(request)
        sess = session(pick(run))
        params = {"run": run, "outcome": outcome, "opponent": opponent, "step": step,
                  "limit": limit, "worst": worst, "seeds": seeds, "alts": alts,
                  "concurrency": concurrency}
        job = app.state.jobs.submit(
            "falsify_scan", params,
            lambda: sess.falsify_scan(outcome=outcome, opponent=opponent, step=step, limit=limit,
                                      worst=worst, n_seeds=seeds, n_alts=alts,
                                      concurrency=concurrency))
        return JobRef(**{k: job.as_dict()[k] for k in ("id", "kind", "status", "done")})

    @app.post("/api/jobs/calibration", tags=["jobs"], response_model=JobRef, status_code=202,
              summary="Submit ProbeSession.calibration() — minutes; needs the shared password")
    def api_job_calibration(
        request: Request,
        run: "str | None" = Query(None),
        outcome: str = Query("loss", pattern="^(win|loss)$"),
        opponent: "str | None" = Query(None),
        step: "int | None" = Query(None),
        limit: int = Query(20, ge=1, le=200),
        worst: int = Query(2, ge=1, le=10),
        seeds: int = Query(32, ge=2, le=200),
        alts: int = Query(2, ge=1, le=8),
        concurrency: int = Query(8, ge=1, le=16),
        bins: int = Query(10, ge=2, le=50),
        overvalue_tau: float = Query(5.0),
    ) -> JobRef:
        require_unlocked(request)
        sess = session(pick(run))
        params = {"run": run, "outcome": outcome, "opponent": opponent, "step": step,
                  "limit": limit, "worst": worst, "seeds": seeds, "alts": alts,
                  "concurrency": concurrency, "bins": bins, "overvalue_tau": overvalue_tau}
        job = app.state.jobs.submit(
            "calibration", params,
            lambda: sess.calibration(outcome=outcome, opponent=opponent, step=step, limit=limit,
                                     worst=worst, n_seeds=seeds, n_alts=alts,
                                     concurrency=concurrency, n_bins=bins,
                                     overvalue_tau=overvalue_tau))
        return JobRef(**{k: job.as_dict()[k] for k in ("id", "kind", "status", "done")})

    # -- JSON API: the three PER-DECISION counterfactual probes (PASSWORD REQUIRED) -------
    #
    # Same registry, same gate, and for the same reason as the two run-level probes above: each of
    # these spawns Node and runs for seconds to minutes (≈2 s for a depth-1 search, ≈4 s at depth 2,
    # seconds per replay rollout), so they are WORK, not reading. They differ only in being anchored
    # on ONE decision — hence `battle` + `inv`, and the battle token goes through `battle_row`, so a
    # client string is matched against the run's own listing rather than joined to a path.
    #
    # All three ALSO need the trace's `*_reconstruction.json` sibling and load the checkpoint, so
    # their two ordinary failures — a websocket/older trace, and `ArchDriftError` on any run that is
    # not at the current architecture (measured 79/79 archived runs) — arrive as `status="error"`
    # with the message intact. `partials/job.html` renders that message whole; see the note there.

    @app.post("/api/jobs/lookahead", tags=["jobs"], response_model=JobRef, status_code=202,
              summary="Submit ProbeSession.lookahead() — one-ply V(s′) per legal action")
    def api_job_lookahead(
        request: Request,
        run: "str | None" = Query(None),
        battle: "str | None" = Query(None, description="battle short_id; default = the newest"),
        inv: int = Query(0, ge=0, description="invocation index (the Nth decision)"),
        seeds: int = Query(0, ge=0, le=64,
                           description="0 = the CRN headline alone (the session default); >0 also "
                                       "dice-averages V(s′) ± std over that many fresh seeds"),
        followup: str = Query("random", pattern="^(random|default)$"),
    ) -> JobRef:
        require_unlocked(request)
        sess = session(pick(run))
        row = battle_row(sess, battle)
        params = {"run": run, "battle": row["short_id"], "inv": inv, "seeds": seeds,
                  "followup": followup}
        job = app.state.jobs.submit(
            "lookahead", params,
            lambda: sess.lookahead(row["id"], inv=inv, n_seeds=seeds, followup=followup))
        return JobRef(**{k: job.as_dict()[k] for k in ("id", "kind", "status", "done")})

    @app.post("/api/jobs/better-line", tags=["jobs"], response_model=JobRef, status_code=202,
              summary="Submit ProbeSession.better_line() — a CRN-anchored beam over the critic")
    def api_job_better_line(
        request: Request,
        run: "str | None" = Query(None),
        battle: "str | None" = Query(None, description="battle short_id; default = the newest"),
        inv: int = Query(0, ge=0),
        depth: int = Query(2, ge=1, le=4, description="OUR plies searched; 1 == lookahead"),
        beam: int = Query(3, ge=1, le=8),
        top_k: int = Query(4, ge=1, le=11),
        interior_opponent: str = Query("self", pattern="^(self|none)$",
                                       description="who reacts at INTERIOR plies: the trainee as a "
                                                   "flagged self-proxy, or the sim default. "
                                                   "('ckpt' needs a path — CLI only.)"),
        confirm_rollouts: int = Query(0, ge=0, le=16,
                                      description="ground-truth Monte-Carlo confirm of the "
                                                  "recommendation; each rollout is a full game"),
    ) -> JobRef:
        require_unlocked(request)
        sess = session(pick(run))
        row = battle_row(sess, battle)
        params = {"run": run, "battle": row["short_id"], "inv": inv, "depth": depth, "beam": beam,
                  "top_k": top_k, "interior_opponent": interior_opponent,
                  "confirm_rollouts": confirm_rollouts}
        job = app.state.jobs.submit(
            "better_line", params,
            lambda: sess.better_line(row["id"], inv, depth=depth, beam=beam, top_k=top_k,
                                     interior_opponent=interior_opponent,
                                     confirm_rollouts=confirm_rollouts))
        return JobRef(**{k: job.as_dict()[k] for k in ("id", "kind", "status", "done")})

    @app.post("/api/jobs/replay-counterfactual", tags=["jobs"], response_model=JobRef,
              status_code=202,
              summary="Submit ProbeSession.replay_counterfactual() — substitute an action, "
                      "play the rest to a win/loss")
    def api_job_replay_counterfactual(
        request: Request,
        run: "str | None" = Query(None),
        battle: "str | None" = Query(None, description="battle short_id; default = the newest"),
        inv: int = Query(0, ge=0),
        action: int = Query(..., ge=0, le=10,
                            description="the action INDEX to substitute for ours at this decision "
                                        "(the position in `analyze`'s `actions` list)"),
        rollouts: int = Query(1, ge=1, le=32,
                             description="1 = the single realized-dice line, which is NOT a "
                                         "probability; >1 resamples the post-divergence dice"),
        opponent_source: str = Query("auto", pattern="^(auto|bot|self)$"),
        narrate: bool = Query(True, description="capture the first recovered win/loss play-by-play"),
    ) -> JobRef:
        require_unlocked(request)
        sess = session(pick(run))
        row = battle_row(sess, battle)
        params = {"run": run, "battle": row["short_id"], "inv": inv, "action": action,
                  "rollouts": rollouts, "opponent_source": opponent_source, "narrate": narrate}
        job = app.state.jobs.submit(
            "replay_counterfactual", params,
            lambda: sess.replay_counterfactual(row["id"], inv, action, n_rollouts=rollouts,
                                               opponent_source=opponent_source, narrate=narrate))
        return JobRef(**{k: job.as_dict()[k] for k in ("id", "kind", "status", "done")})

    @app.get("/api/jobs", tags=["jobs"], response_model=list,
             summary="Every submitted job, newest first (no results — those are per-id)")
    def api_jobs() -> list:
        return app.state.jobs.list()

    @app.get("/api/jobs/{job_id}", tags=["jobs"], response_model=JobState,
             summary="One job's state, with its result once done. Readable without the password.")
    def api_job(job_id: str) -> JobState:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        return JobState(**job.as_dict())

    # -- login ---------------------------------------------------------------------------

    @app.get("/login", response_class=HTMLResponse, tags=["auth"],
             summary="The shared-password form")
    def page_login(request: Request, next: str = Query("/falsify")) -> HTMLResponse:
        return page(request, "login.html", "login", None, next_url=_safe_next(next), message=None)

    @app.post("/login", tags=["auth"], summary="Exchange the shared password for a session cookie")
    def do_login(request: Request, password: str = Form(""),
                 next: str = Form("/falsify")):
        auth, client = app.state.auth, _client(request)
        target = _safe_next(next)
        if not auth.configured:
            return page(request, "login.html", "login", None, next_url=target,
                        message="No password is configured on this instance, so the expensive "
                                "probes are switched off entirely.")
        wait = auth.throttled(client)
        if wait > 0:
            return page(request, "login.html", "login", None, next_url=target,
                        message=f"Too many attempts — try again in {int(wait) + 1}s.")
        if not auth.check(password, client):
            return page(request, "login.html", "login", None, next_url=target,
                        message="That is not the password.")
        resp = RedirectResponse(target, status_code=303)
        resp.set_cookie(COOKIE, auth.issue(), httponly=True, samesite="lax",
                        secure=_is_https(request), max_age=14 * 24 * 3600, path="/")
        return resp

    @app.post("/logout", tags=["auth"], summary="Drop the session cookie")
    def do_logout(request: Request):
        resp = RedirectResponse("/", status_code=303)
        resp.delete_cookie(COOKIE, path="/")
        return resp

    # -- HTML pages ------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, tags=["pages"],
             summary="Run summary: steps, identity, opponents, checkpoints")
    def page_index(request: Request, run: "str | None" = Query(None)) -> HTMLResponse:
        name, data, err = _load(pick, session, guarded, run, store, lambda s: s.run_summary())
        spec = charts.outcome_by_step_spec(data["steps"]) if data and data["steps"] else None
        return page(request, "index.html", "run", name, data=data, error=err, spec=spec)

    @app.get("/battles", response_class=HTMLResponse, tags=["pages"],
             summary="Battle list with outcome/opponent/step filters")
    def page_battles(request: Request, run: "str | None" = Query(None)) -> HTMLResponse:
        # DEFAULTS TO THE NEWEST STEP, not to every step. A run holds thousands of traces across
        # every eval cycle it ever ran, and the question behind opening this page is essentially
        # always "what is the CURRENT model doing" — an all-steps default answered that with a
        # 200-row cap sliced out of an arbitrary mixture of checkpoints, oldest included. "All
        # steps" is still one selection away.
        name, data, err = _load(pick, session, guarded, run, store, lambda s: s.run_summary())
        step = _newest_step(data)
        rows, rerr = guarded(lambda: session(pick(run)).battles(step=step))
        shown, total = _cap(rows, _BATTLE_PAGE)
        return page(request, "battles.html", "battles", name, summary=data,
                    error=err or rerr, opponents=_opponents(data), selected_step=step,
                    rows=shown, total=total, capped=_BATTLE_PAGE)

    @app.get("/battle", response_class=HTMLResponse, tags=["pages"],
             summary="Turn-by-turn replay of one battle: board, what happened, the critic's read")
    def page_battle(
        request: Request,
        run: "str | None" = Query(None),
        battle: "str | None" = Query(None, description="battle short_id; default = the first"),
        start: "str | None" = Query(None, description="first game turn to show (windowing)"),
    ) -> HTMLResponse:
        # DELIBERATELY NOT HTMX. Every other page refilters a table in place; this one is a thing
        # you read, link to and send to someone ("look at turn 47"). So the battle picker and the
        # turn window are plain GET forms and links, which makes every view of it a shareable URL
        # and leaves it fully working with JavaScript off.
        name, data, err = _load(pick, session, guarded, run, store, lambda s: s.run_summary())
        sess = session(pick(run))
        try:
            row = battle_row(sess, battle)
        except _NoBattles as empty:
            return page(request, "battle.html", "battle", name, summary=data, error=err,
                        battles=[], selected=None, turns=None, window=[],
                        first_turn=None, last_turn=None, empty=empty.detail)
        turns, terr = guarded(lambda: sess.battle_turns(row["id"]))
        window, first, last = _turn_window(turns, _form_int(start))
        return page(request, "battle.html", "battle", name, summary=data, error=err or terr,
                    battles=_picker_rows(sess.battles(), row), selected=row,
                    turns=turns, window=window, first_turn=first, last_turn=last)

    @app.get("/analyze", response_class=HTMLResponse, tags=["pages"],
             summary="One decision, fully analyzed: faithfulness, beliefs, threats, saliency")
    def page_analyze(
        request: Request,
        run: "str | None" = Query(None),
        battle: "str | None" = Query(None, description="battle short_id; default = the newest"),
        inv: "str | None" = Query(None, description="invocation index (the Nth decision)"),
    ) -> HTMLResponse:
        # The battle + decision are PLAIN GET params, like `/battle` and unlike the filter tables:
        # "look at this decision" is a thing you link to and send someone, and it must survive a
        # reload. But the ANALYSIS itself arrives via HTMX (`hx-trigger="load"`, the `/scan`
        # pattern) because it loads a checkpoint — 0.5–3 s on a run that loads at all, and a
        # checkpoint load is not something to spend a white page on.
        name, data, err = _load(pick, session, guarded, run, store, lambda s: s.run_summary())
        sess = session(pick(run))
        try:
            row = battle_row(sess, battle)
        except _NoBattles as empty:
            return page(request, "analyze.html", "analyze", name, summary=data, error=err,
                        battles=[], selected=None, inv=0, empty=empty.detail)
        return page(request, "analyze.html", "analyze", name, summary=data, error=err,
                    battles=_picker_rows(sess.battles(), row), selected=row,
                    inv=_form_int(inv, 0))

    @app.get("/scan", response_class=HTMLResponse, tags=["pages"],
             summary="Cross-battle worst-turning-point scan")
    def page_scan(request: Request, run: "str | None" = Query(None)) -> HTMLResponse:
        name, data, err = _load(pick, session, guarded, run, store, lambda s: s.run_summary())
        return page(request, "scan.html", "scan", name, summary=data, error=err,
                    opponents=_opponents(data))

    @app.get("/triage", response_class=HTMLResponse, tags=["pages"],
             summary="Loss-lever triage: failure categories ranked by recoverable win-rate")
    def page_triage(request: Request, run: "str | None" = Query(None)) -> HTMLResponse:
        name, data, err = _load(pick, session, guarded, run, store, lambda s: s.run_summary())
        tri, terr = guarded(lambda: session(pick(run)).triage())
        cats = (tri or {}).get("categories") or []
        return page(request, "triage.html", "triage", name, summary=data,
                    error=err or terr, opponents=_opponents(data),
                    data=tri, spec=charts.triage_lever_spec(cats) if cats else None)

    @app.get("/falsify", response_class=HTMLResponse, tags=["pages"],
             summary="falsify_scan's crater bracket (submits a background job)")
    def page_falsify(request: Request, run: "str | None" = Query(None)) -> HTMLResponse:
        name, data, err = _load(pick, session, guarded, run, store, lambda s: s.run_summary())
        latest = app.state.jobs.latest("falsify_scan")
        return page(request, "falsify.html", "falsify", name, summary=data, error=err,
                    opponents=_opponents(data), job=latest.as_dict() if latest else None)

    @app.get("/calibration", response_class=HTMLResponse, tags=["pages"],
             summary="Critic reliability curve (submits a background job)")
    def page_calibration(request: Request, run: "str | None" = Query(None)) -> HTMLResponse:
        name, data, err = _load(pick, session, guarded, run, store, lambda s: s.run_summary())
        latest = app.state.jobs.latest("calibration")
        return page(request, "calibration.html", "calibration", name, summary=data, error=err,
                    opponents=_opponents(data), job=latest.as_dict() if latest else None)

    # -- HTMX fragments ----------------------------------------------------------------------
    # Each is the SAME markup the full page embeds on first load, so a filtered swap and a fresh
    # page render cannot drift apart.
    #
    # THEIR NUMERIC PARAMETERS ARE DECLARED AS STRINGS, on purpose. An HTML `<select>` whose "all"
    # option has `value=""` submits `step=`, and an empty string is not an int — so a strictly
    # typed `int | None` 422s the moment the user picks "all steps". That is not hypothetical: it
    # shipped, every unit test passed (they never sent the empty field), and only the headless
    # browser caught it. The `/api/*` routes stay strictly typed — a machine client should be told
    # its int is malformed; a browser form legitimately sends "".

    @app.get("/partials/battles", response_class=HTMLResponse, tags=["partials"],
             summary="Battle table fragment (HTMX target)")
    def partial_battles(
        request: Request,
        run: "str | None" = Query(None),
        outcome: "str | None" = Query(None),
        opponent: "str | None" = Query(None),
        step: "str | None" = Query(None),
    ) -> HTMLResponse:
        sess = session(pick(run))
        rows, err = guarded(lambda: sess.battles(
            outcome=outcome or None, opponent=opponent or None, step=_form_int(step)))
        shown, total = _cap(rows, _BATTLE_PAGE)
        # `run` rides along because each row links into /battle, which resolves a battle WITHIN a
        # run — a link that dropped it would silently open the default run's battle instead.
        return fragment(request, "partials/battles_table.html", rows=shown, error=err,
                        total=total, capped=_BATTLE_PAGE, run=run)

    @app.get("/partials/scan", response_class=HTMLResponse, tags=["partials"],
             summary="Scan table + chart fragment (HTMX target)")
    def partial_scan(
        request: Request,
        run: "str | None" = Query(None),
        outcome: "str | None" = Query("loss"),
        opponent: "str | None" = Query(None),
        step: "str | None" = Query(None),
        metric: str = Query("value_drop", pattern="^(value_drop|td_residual)$"),
        limit: "str | None" = Query("25"),
    ) -> HTMLResponse:
        sess = session(pick(run))
        rows, err = guarded(lambda: sess.scan(
            outcome=outcome or None, opponent=opponent or None, step=_form_int(step),
            metric=metric, limit=_form_int(limit, 25)))
        spec = charts.scan_drop_spec(rows, metric=metric) if rows else None
        return fragment(request, "partials/scan_table.html", rows=rows or [], error=err,
                        spec=spec, metric=metric, run=run)

    @app.get("/partials/analyze", response_class=HTMLResponse, tags=["partials"],
             summary="One decision's full analysis (HTMX target; loads the checkpoint)")
    def partial_analyze(
        request: Request,
        run: "str | None" = Query(None),
        battle: "str | None" = Query(None),
        inv: "str | None" = Query("0"),
    ) -> HTMLResponse:
        sess = session(pick(run))
        row = battle_row(sess, battle)
        i = _form_int(inv, 0) or 0
        data, err = guarded(lambda: sess.analyze(row["id"], i))
        # `ArchDriftError` is the EXPECTED outcome on an archived run (79/79 measured), and its
        # message is a multi-line DIAGNOSIS written for a human that ends with the exact
        # `git checkout` to re-probe from. `guarded` hands it over verbatim and `.err` is
        # `white-space: pre-wrap`, so the page renders the whole thing rather than collapsing it
        # to "analysis failed" — which would throw away the only part that says what to do next.
        dist = (data or {}).get("value_dist")
        return fragment(request, "partials/analyze_result.html", a=data, error=err,
                        spec=_value_dist_spec(dist) if dist else None,
                        run=run, battle=row["short_id"], battle_path=row["id"], inv=i)

    @app.get("/partials/awareness", response_class=HTMLResponse, tags=["partials"],
             summary="Run-level 'did it KNOW?' panel (HTMX target)")
    def partial_awareness(
        request: Request,
        run: "str | None" = Query(None),
        outcome: "str | None" = Query("loss"),
        step: "str | None" = Query(None),
    ) -> HTMLResponse:
        # Loaded async on `/`, like `/scan`: this reads every matching battle's npz, so it is
        # seconds on a real run and must not sit in front of the run summary's first paint.
        sess = session(pick(run))
        data, err = guarded(lambda: sess.awareness_scan(
            outcome=outcome or None, step=_form_int(step)))
        # `awareness_scan` reports a missing dist head as an `error` KEY rather than by raising —
        # an ordinary state of most runs, not a failure, so it renders as a note either way.
        return fragment(request, "partials/awareness_panel.html", data=data, error=err,
                        outcome=outcome, run=run)

    @app.get("/partials/triage", response_class=HTMLResponse, tags=["partials"],
             summary="Triage table + lever chart fragment (HTMX target)")
    def partial_triage(
        request: Request,
        run: "str | None" = Query(None),
        step: "str | None" = Query(None),
        opponent: "str | None" = Query(None),
    ) -> HTMLResponse:
        sess = session(pick(run))
        data, err = guarded(lambda: sess.triage(step=_form_int(step), opponent=opponent or None))
        cats = (data or {}).get("categories") or []
        spec = charts.triage_lever_spec(cats) if cats else None
        return fragment(request, "partials/triage_table.html", data=data, error=err, spec=spec)

    @app.get("/partials/job/{job_id}", response_class=HTMLResponse, tags=["partials"],
             summary="Poll target: renders a job's progress, then its result view")
    def partial_job(request: Request, job_id: str) -> HTMLResponse:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        return fragment(request, "partials/job.html", job=job.as_dict(),
                        **_job_view(job.kind, job.result))

    @app.post("/partials/job/falsify-scan", response_class=HTMLResponse, tags=["partials"],
              summary="Submit a falsify_scan from the page (needs the shared password)")
    def partial_submit_falsify(
        request: Request,
        # FORM, not Query. An HTMX `<form hx-post>` serializes its fields into the request BODY, so
        # `Query(...)` here read the (empty) URL and every control on the page was silently ignored:
        # measured, a submit of outcome=win/limit=3/seeds=7/concurrency=4 reached the session as
        # loss/20/32/1. The `outcome` one is the dangerous member — asking for WINS quietly scanned
        # LOSSES and returned a confident answer to a question nobody asked. `run` stays a Query
        # because it rides the URL (the run picker is a link, not a field).
        run: "str | None" = Query(None),
        outcome: str = Form("loss"),
        opponent: "str | None" = Form(None),
        step: "str | None" = Form(None),
        limit: "str | None" = Form("20"),
        worst: "str | None" = Form("2"),
        seeds: "str | None" = Form("32"),
        alts: "str | None" = Form("2"),
        concurrency: "str | None" = Form("1"),
    ) -> HTMLResponse:
        if not unlocked(request):
            return fragment(request, "partials/locked.html", next_url="/falsify")
        sess = session(pick(run))
        params = {"run": run, "outcome": outcome, "opponent": opponent or None,
                  "step": _form_int(step), "limit": _form_int(limit, 20),
                  "worst": _form_int(worst, 2), "seeds": _form_int(seeds, 32),
                  "alts": _form_int(alts, 2), "concurrency": _form_int(concurrency, 1)}
        job = app.state.jobs.submit(
            "falsify_scan", params,
            lambda: sess.falsify_scan(outcome=params["outcome"], opponent=params["opponent"],
                                      step=params["step"], limit=params["limit"],
                                      worst=params["worst"], n_seeds=params["seeds"],
                                      n_alts=params["alts"],
                                      concurrency=params["concurrency"]))
        return fragment(request, "partials/job.html", job=job.as_dict(), **_job_view(job.kind, None))

    @app.post("/partials/job/calibration", response_class=HTMLResponse, tags=["partials"],
              summary="Submit a calibration from the page (needs the shared password)")
    def partial_submit_calibration(
        request: Request,
        run: "str | None" = Query(None),      # rides the URL; see the note on the falsify submit
        outcome: str = Form("loss"),
        opponent: "str | None" = Form(None),
        step: "str | None" = Form(None),
        limit: "str | None" = Form("20"),
        worst: "str | None" = Form("2"),
        seeds: "str | None" = Form("32"),
        alts: "str | None" = Form("2"),
        concurrency: "str | None" = Form("8"),
        bins: "str | None" = Form("10"),
        overvalue_tau: "str | None" = Form("5.0"),
    ) -> HTMLResponse:
        if not unlocked(request):
            return fragment(request, "partials/locked.html", next_url="/calibration")
        sess = session(pick(run))
        params = {"run": run, "outcome": outcome, "opponent": opponent or None,
                  "step": _form_int(step), "limit": _form_int(limit, 20),
                  "worst": _form_int(worst, 2), "seeds": _form_int(seeds, 32),
                  "alts": _form_int(alts, 2), "concurrency": _form_int(concurrency, 8),
                  "bins": _form_int(bins, 10),
                  "overvalue_tau": _form_float(overvalue_tau, 5.0)}
        job = app.state.jobs.submit(
            "calibration", params,
            lambda: sess.calibration(outcome=params["outcome"], opponent=params["opponent"],
                                     step=params["step"], limit=params["limit"],
                                     worst=params["worst"], n_seeds=params["seeds"],
                                     n_alts=params["alts"], concurrency=params["concurrency"],
                                     n_bins=params["bins"],
                                     overvalue_tau=params["overvalue_tau"]))
        return fragment(request, "partials/job.html", job=job.as_dict(), **_job_view(job.kind, None))

    # -- the three PER-DECISION counterfactual probes, submitted from /analyze ------------
    #
    # WHY THESE READ `Form` AND THE TWO ABOVE READ `Query`. They are submitted by real <form>s in
    # `analyze_result.html`, and an HTMX POST puts the form's fields in the request BODY — a
    # `Query(...)` parameter reads the URL only. For a run-level scan a field that silently falls
    # back to its default is a wrong knob; here `action` says WHICH move to substitute, so the same
    # slip would probe a different decision than the one the reader asked about. Every field,
    # `run`/`battle`/`inv` included, therefore rides the body (the forms carry them as hidden
    # inputs), and the JSON `/api/jobs/...` twins above stay the query-string surface for scripts.

    def _cf_locked(request: Request) -> HTMLResponse:
        # Deliberately NOT echoing the run/battle back into the login `next`: this branch runs
        # BEFORE any membership check, so those are still raw client strings.
        return fragment(request, "partials/locked.html", next_url="/analyze")

    @app.post("/partials/job/lookahead", response_class=HTMLResponse, tags=["partials"],
              summary="Submit a one-ply lookahead from /analyze (needs the shared password)")
    def partial_submit_lookahead(
        request: Request,
        run: "str | None" = Form(None),
        battle: "str | None" = Form(None),
        inv: "str | None" = Form("0"),
        seeds: "str | None" = Form("0"),
        followup: "str | None" = Form("random"),
    ) -> HTMLResponse:
        if not unlocked(request):
            return _cf_locked(request)
        sess = session(pick(run))
        row = battle_row(sess, battle)
        params = {"run": run, "battle": row["short_id"], "inv": _form_int(inv, 0) or 0,
                  "seeds": _form_int(seeds, 0) or 0,
                  "followup": followup if followup in ("random", "default") else "random"}
        job = app.state.jobs.submit(
            "lookahead", params,
            lambda: sess.lookahead(row["id"], inv=params["inv"], n_seeds=params["seeds"],
                                   followup=params["followup"]))
        return fragment(request, "partials/job.html", job=job.as_dict(), **_job_view(job.kind, None))

    @app.post("/partials/job/better-line", response_class=HTMLResponse, tags=["partials"],
              summary="Submit a better-line search from /analyze (needs the shared password)")
    def partial_submit_better_line(
        request: Request,
        run: "str | None" = Form(None),
        battle: "str | None" = Form(None),
        inv: "str | None" = Form("0"),
        depth: "str | None" = Form("2"),
        beam: "str | None" = Form("3"),
        top_k: "str | None" = Form("4"),
        interior_opponent: "str | None" = Form("self"),
        confirm_rollouts: "str | None" = Form("0"),
    ) -> HTMLResponse:
        if not unlocked(request):
            return _cf_locked(request)
        sess = session(pick(run))
        row = battle_row(sess, battle)
        params = {"run": run, "battle": row["short_id"], "inv": _form_int(inv, 0) or 0,
                  "depth": _form_int(depth, 2) or 2, "beam": _form_int(beam, 3) or 3,
                  "top_k": _form_int(top_k, 4) or 4,
                  "interior_opponent": interior_opponent if interior_opponent in ("self", "none")
                  else "self",
                  "confirm_rollouts": _form_int(confirm_rollouts, 0) or 0}
        job = app.state.jobs.submit(
            "better_line", params,
            lambda: sess.better_line(row["id"], params["inv"], depth=params["depth"],
                                     beam=params["beam"], top_k=params["top_k"],
                                     interior_opponent=params["interior_opponent"],
                                     confirm_rollouts=params["confirm_rollouts"]))
        return fragment(request, "partials/job.html", job=job.as_dict(), **_job_view(job.kind, None))

    @app.post("/partials/job/replay-counterfactual", response_class=HTMLResponse, tags=["partials"],
              summary="Substitute an action and replay to a win/loss (needs the shared password)")
    def partial_submit_replay_counterfactual(
        request: Request,
        run: "str | None" = Form(None),
        battle: "str | None" = Form(None),
        inv: "str | None" = Form("0"),
        action: "str | None" = Form(None),
        rollouts: "str | None" = Form("1"),
        opponent_source: "str | None" = Form("auto"),
        # An UNCHECKED checkbox sends no field at all, so the default has to be "absent", not "on"
        # — otherwise un-ticking the play-by-play would silently keep capturing it.
        narrate: "str | None" = Form(None),
    ) -> HTMLResponse:
        if not unlocked(request):
            return _cf_locked(request)
        act = _form_int(action)
        if act is None or act < 0:
            # There is no sensible default here — "replay something else" has to say WHAT. The form
            # marks the picker `required`, so this is the hand-rolled-request path; it renders as a
            # message rather than a 400 the HTMX swap would drop on the floor.
            return fragment(request, "partials/job.html", job=None, specs=[],
                            kind="replay_counterfactual",
                            message="pick which action to substitute — the counterfactual replay "
                                    "has no default alternative to run.")
        sess = session(pick(run))
        row = battle_row(sess, battle)
        params = {"run": run, "battle": row["short_id"], "inv": _form_int(inv, 0) or 0,
                  "action": act, "rollouts": _form_int(rollouts, 1) or 1,
                  "opponent_source": opponent_source if opponent_source in ("auto", "bot", "self")
                  else "auto",
                  "narrate": narrate is not None}
        job = app.state.jobs.submit(
            "replay_counterfactual", params,
            lambda: sess.replay_counterfactual(row["id"], params["inv"], params["action"],
                                               n_rollouts=params["rollouts"],
                                               opponent_source=params["opponent_source"],
                                               narrate=params["narrate"]))
        return fragment(request, "partials/job.html", job=job.as_dict(), **_job_view(job.kind, None))

    # Registered on STARLETTE's HTTPException, not FastAPI's. Starlette dispatches by walking
    # `type(exc).__mro__`, and an unmatched route raises the starlette class — which is the PARENT
    # of `fastapi.HTTPException`, not a subclass. Handling only the FastAPI one therefore misses
    # every 404 (it did: `/nope` returned the stock `{"detail": ...}` JSON to a browser).
    @app.exception_handler(StarletteHTTPException)
    def _http_error(request: Request, exc: StarletteHTTPException):
        """JSON for the API, a rendered page for a browser — an agent hitting `/api/...` always
        gets parseable output, mirroring the CLI's `{"error": ...}` envelope."""
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
        rows = app.state.runs.list_runs() if app.state.runs else []
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=exc.status_code,
            context={"page_name": "error", "nav": _NAV, "runs": rows, "run": None,
                     "models_root": app.state.root, "unlocked": unlocked(request),
                     "auth_required": app.state.auth.required,
                     "auth_configured": app.state.auth.configured,
                     "status": exc.status_code, "detail": exc.detail})

    return app


# Ordered by the investigation recipe in src/main/prober/CLAUDE.md — "triage: start here for
# 'what next'" — not by when each view happened to be written. Six equal tabs in an arbitrary
# order is exactly the "information doesn't flow" complaint the TUI earned.
_NAV = [("/", "run"), ("/triage", "triage"), ("/scan", "scan"), ("/battles", "battles"),
        ("/battle", "battle"), ("/analyze", "analyze"),
        ("/falsify", "falsify"), ("/calibration", "calibration")]

# What each view ANSWERS, in recipe order. Rendered as the "where to start" card on `/` and as the
# one-line subtitle on each page, so a newcomer never has to guess which tab holds their question.
VIEW_QUESTIONS = [
    ("/triage", "triage", "Which failure lever would recover the most rating?",
     "Start here. Every loss is attributed to its worst turning point and the causes are ranked."),
    ("/scan", "scan", "Where exactly did each loss go wrong?",
     "The single worst decision in every matching battle, ranked globally. Model-free."),
    ("/battles", "battles", "Which battles were captured?",
     "The raw trace list, filterable — the ids you hand to the CLI or the TUI."),
    ("/battle", "battle", "How did one game actually go?",
     "Turn by turn: the board, what each side did, and what the critic made of it. Model-free."),
    ("/analyze", "analyze", "Why did it choose that, and what did it believe?",
     "One decision, all the way down — faithfulness, beliefs, threat tables, saliency. The only "
     "view that LOADS the checkpoint, so it works on a current-architecture run and diagnoses "
     "the drift on every other."),
    ("/falsify", "falsify", "Was that crater bad luck or a reducible mistake?",
     "Re-rolls the dice on the worst craters. Minutes of work; needs the shared password."),
    ("/calibration", "calibration", "Was the critic wrong, or was the position lost?",
     "Splits falsify's unattributed bucket against a reliability curve. Needs the password."),
]


def _load(pick, session, guarded, run, store, fn):
    """Resolve the run, then run a read-only session call under `guarded`.

    Returns `(run_name, data, error)`. The run NAME (not the path) goes back to the template, so
    every link and form on the page carries the same opaque token the client sent — a page never
    round-trips a filesystem path through the browser.
    """
    path = pick(run)
    name = run or store().default_run()
    data, err = guarded(lambda: fn(session(path)))
    return name, data, err


def _value_dist_spec(dist: dict) -> dict:
    """The distributional critic's predicted RETURN DISTRIBUTION as a real chart.

    This is the single biggest thing a browser buys over the terminal on this view: the TUI can
    only draw the histogram as a one-line eighth-block sparkline, where "sharp vs wide vs BIMODAL"
    — the whole interpretability point of the head — is a judgement call about eight characters.
    Plotted, the shape IS the reading.

    The spec is built from `charts`' own base so it carries the same transparent background and
    fit-autosize as every other chart on the site (`charts.py` is the one place chart encoding
    lives and it is not editable in this change; duplicating its constants here would be the drift
    that module exists to prevent). Pure: dict in, dict out.
    """
    values = [{"z": z, "p": p} for z, p in zip(dist.get("support") or [], dist.get("probs") or [])]
    return charts._spec(
        title=charts._title(
            "Critic's predicted return distribution",
            "sharp = confident · wide = uncertain · two humps = the critic sees a coinflip"),
        data={"values": values},
        width="container", height=150,
        mark={"type": "area", "interpolate": "step-after", "tooltip": True,
              "color": "#2a6f97", "opacity": 0.85},
        encoding={
            "x": {"field": "z", "type": "quantitative",
                  "axis": {"title": "return (head support space)"}},
            "y": {"field": "p", "type": "quantitative", "axis": {"title": "probability"}},
            "tooltip": [{"field": "z", "type": "quantitative", "title": "return"},
                        {"field": "p", "type": "quantitative", "title": "P", "format": ".3f"}],
        },
    )


def _safe_next(target: str) -> str:
    """Only ever redirect within this app.

    An open redirect on a login form is the classic phishing primitive: `/login?next=https://evil`
    turns our hostname into the bait. A single leading slash and no scheme is the whole rule, and
    `//evil.com` is caught because it is protocol-relative."""
    if not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target


def _is_https(request: Request) -> bool:
    """Should the session cookie carry `Secure`?

    Behind the Cloudflare tunnel the app itself speaks http, so the scheme alone would say "no"
    on exactly the deployment where it matters. `X-Forwarded-Proto` is what the tunnel sets."""
    forwarded = request.headers.get("x-forwarded-proto", "")
    return request.url.scheme == "https" or forwarded.split(",")[0].strip() == "https"


# Peers whose forwarding header we believe. Behind the Cloudflare tunnel, `cloudflared` runs on
# this same box, so every request genuinely arrives from loopback and `CF-Connecting-IP` is set by
# Cloudflare's edge (which overwrites any client-supplied value). From anywhere else the header is
# just a string the client typed.
_TRUSTED_PEERS = frozenset({"127.0.0.1", "::1", "localhost"})
_FORWARD_HEADER = "cf-connecting-ip"


def _client(request: Request) -> str:
    """A stable-enough identity for login throttling.

    ONLY honours the forwarding header when the immediate peer is trusted. The first version read
    `CF-Connecting-IP` (falling back to `X-Forwarded-For`) unconditionally, which an adversarial
    review defeated completely: rotating the header per request gave **500/500 password guesses
    with the cooldown never firing**, and spoofing someone else's address could lock THEM out.

    `X-Forwarded-For` is gone entirely rather than trust-gated — it is the append-friendly one, so
    even behind a proxy a client-supplied value can survive in it. One deployment, one header.

    Note the failure mode when this is wrong is now over-throttling (every visitor behind an
    untrusted proxy shares the peer's identity), not under-throttling — and the global cap in
    `auth.py` holds regardless of how identity is derived.
    """
    peer = request.client.host if request.client else "?"
    if peer in _TRUSTED_PEERS:
        value = request.headers.get(_FORWARD_HEADER)
        if value:
            return value.split(",")[0].strip()
    return peer


def _form_int(value: "str | None", default: "int | None" = None) -> "int | None":
    """Parse a form field that may legitimately be blank. `""` (the "all" option) and an
    unparseable value both fall back to `default` — a browser control must not be able to 422 a
    page fragment."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _form_float(value: "str | None", default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


# A run has thousands of traces; 2397 rows rendered to 545 KB of HTML is not a table anyone reads,
# it is a download. Cap it and say so — the filters are how you find a specific battle.
_BATTLE_PAGE = 200

# How many battles the /battle page's picker <select> offers, and how many game turns one view of a
# battle shows. MEASURED on a real run: the longest battle is 249 turns / 522 KB of session JSON, so
# an unwindowed replay is the same "download, not a page" failure `_BATTLE_PAGE` exists to prevent —
# and it lands on a phone. The window is navigated by plain prev/next links (and jumped into by the
# `notable` shortcuts), so nothing is unreachable.
#
# The picker is 100, not 300, because it is a CONVENIENCE, not the way you find a battle: `battles`
# and `scan` are, and both now link straight into the replay. At 300 the dropdown was 40 KB of a
# 154 KB page (measured) — a quarter of the payload, on a phone, for a list nobody scrolls.
_BATTLE_PICK = 100
_TURN_PAGE = 50


def _newest_first(rows: "list[dict]") -> "list[dict]":
    """Battles by most recent eval step first. `ProbeSession.battles()` orders by step ASCENDING
    (the trace tree's natural order), which is the opposite of what a reader wants offered first.
    Sorted STABLY, so the tree's within-step order is preserved."""
    return sorted(rows, key=lambda r: r["step"], reverse=True)


def _picker_rows(rows: "list[dict]", selected: dict) -> "list[dict]":
    """The capped picker list — NEWEST first, with the battle actually being shown GUARANTEED to
    be in it.

    Without that guarantee a `<select>` whose options don't contain the selected value silently
    falls back to displaying its FIRST option — so arriving from a `scan` deep link to an old
    battle would name one battle in the picker while rendering another below it. A dropdown that
    lies about what you are looking at is worse than a short dropdown.
    """
    shown = _newest_first(rows)[:_BATTLE_PICK]
    if selected and not any(r["short_id"] == selected["short_id"] for r in shown):
        shown = [selected] + shown[:_BATTLE_PICK - 1]
    return shown


def _turn_window(turns: "dict | None", start: "int | None"):
    """The slice of `battle_turns()["turns"]` to render, plus the first/last GAME TURN either side of
    it (`None` when there is nothing further that way — the template renders prev/next off that).

    `start` is a game turn NUMBER, not an index: it arrives from a `notable` jump link ("the biggest
    value drop was at turn 47"), and turn numbers are what a reader and a CLI both speak. Turns
    whose number is missing (older recorders bucket them under `None`) sort last and are reachable
    by paging to the end rather than being silently dropped.
    """
    rows = (turns or {}).get("turns") or []
    if not rows:
        return [], None, None
    begin = 0
    if start is not None:
        begin = next((i for i, t in enumerate(rows)
                      if t["turn"] is not None and t["turn"] >= start), max(0, len(rows) - 1))
    begin = max(0, min(begin, len(rows) - 1))
    end = min(begin + _TURN_PAGE, len(rows))
    prev_turn = rows[max(0, begin - _TURN_PAGE)]["turn"] if begin > 0 else None
    next_turn = rows[end]["turn"] if end < len(rows) else None
    return rows[begin:end], prev_turn, next_turn

# How many runs' `ProbeSession`s stay cached. MEASURED: a scan of one run leaves ~430 MB behind
# (real `models/`: 1→466 MB, 6→3.0 GB RSS, monotonic), and the picker offers 81 runs — so an
# unbounded cache is an anonymous-visitor lever on ~35 GB, next to a live trainer. Three keeps
# "compare two runs" instant at roughly 1.3 GB worst case.
_MAX_CACHED_SESSIONS = 3


def _cap(rows, limit):
    """(first `limit` rows, true total). `rows` may be None when the call failed."""
    rows = rows or []
    return rows[:limit], len(rows)


def _group_runs(rows: "list[dict]") -> "list[tuple]":
    """Group the picker by generation so 79 near-identical names become a handful of choices.

    `ai_v9_06_gen5_...` -> "ai_v9"; `run_20260808_...` -> "run 2026-08"; anything else -> "other".
    Groups keep the newest-first order of `rows`, so the group you are working in is at the top.
    """
    import re as _re
    groups: "dict[str, list]" = {}
    for r in rows:
        name = r["name"]
        m = _re.match(r"^(ai_v\d+)", name)
        if m:
            key = m.group(1)
        elif _re.match(r"^run_(\d{4})(\d{2})\d{2}", name):
            m2 = _re.match(r"^run_(\d{4})(\d{2})\d{2}", name)
            key = f"run {m2.group(1)}-{m2.group(2)}"
        else:
            key = "other"
        groups.setdefault(key, []).append(r)
    return list(groups.items())


def _newest_step(summary: "dict | None") -> "int | None":
    """The most recent eval step in the run — the checkpoint whose battles you almost always mean.

    `max` rather than `steps[-1]`, so this does not silently depend on the tree's ordering."""
    steps = (summary or {}).get("steps") or []
    return max((s["step"] for s in steps), default=None)


def _opponents(summary: "dict | None") -> "list[str]":
    """Every opponent name in the run, for the filter dropdowns. Sorted and de-duplicated across
    steps, because an opponent appears once per step it was evaluated at."""
    if not summary:
        return []
    return sorted({o["name"] for s in summary.get("steps", []) for o in s.get("opponents", [])})


def _job_view(kind: str, result: "dict | None") -> dict:
    """The chart specs a finished job's result renders with. Empty until it is done."""
    if not result:
        return {"specs": [], "kind": kind}
    if kind == "falsify_scan":
        return {"specs": [("bracket", charts.crater_bracket_spec(
            result["gate"], weighting=result.get("weighting", "delta")))], "kind": kind}
    if kind == "calibration":
        bins = result.get("reliability_curve") or []
        specs = []
        if bins:
            specs.append(("reliability", charts.reliability_curve_spec(bins)))
            specs.append(("gap", charts.reliability_gap_spec(bins)))
        specs.append(("bracket", charts.crater_bracket_spec(result["falsify_gate"])))
        return {"specs": specs, "kind": kind}
    return {"specs": [], "kind": kind}
