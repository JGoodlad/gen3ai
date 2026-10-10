"""`/game` — THE battle viewer (`designs/prober/battle_viewer_ux_2026-10-09.md`, field map
`designs/prober/battle_view_v2.md`): its page, its two HTMX fragments, its JSON endpoints, and the
`/battle` redirect (the classic replay was merged into this page on 2026-10-09). Registered onto the
app by `create_app` (kept out of `app.py`, which is past the 1,000-line report line).

The one rule holds: every number comes from a `ProbeSession` method (`battle_story`, `battle_board`,
`battle_readout`, `decision_attention`) verbatim; this module only picks the decision, builds the
chart specs (`charts.py`) and renders.

The page is a plain GET (`/game?run=…&battle=…&inv=N&view=model|truth|public`): a position in a
battle, and the information PERSPECTIVE it is read under, are things you link to. The model panels
arrive as an HTMX fragment because they load a checkpoint, and on a run whose architecture is older
than the code they render one plain sentence (plus the `ArchDriftError` diagnosis folded under it) —
never a 500, never a blank panel.

ACCESS (`web/gate.py`): the turn story and the board are model-free and open; every route that loads a
checkpoint or runs the model forward (`/api/game/readout`, `/api/game/attention`, the two
`/partials/game/*`) carries `model_gate`. A locked visitor's page renders the story, the board and
the PUBLIC half of the scouting notes, and an unlock card in each model slot (the HTMX fragments answer
the same card, never a 403 the swap would swallow). `gate_guard_test.py` derives the gated set from the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from main.prober.engine.perspective import PERSPECTIVE_WORDS, PERSPECTIVES, normalize
from main.prober.web import charts
from main.prober.web.gate import model_gate

ARCH_OLDER_TEXT = ("This run's architecture is older than the code, so the model views (intent, "
                   "beliefs, attention, damage physics) cannot load its checkpoint — they need a "
                   "current-architecture checkpoint. The turn story and the board are model-free and still work.")

@dataclass(frozen=True)
class GameHelpers:
    """The app's request plumbing, handed in by `create_app` (they close over its state)."""

    pick: Callable[..., Any]
    session: Callable[..., Any]
    guarded: Callable[..., Any]
    battle_row: Callable[..., Any]
    page: Callable[..., Any]
    fragment: Callable[..., Any]
    load: Callable[..., Any]
    no_battles: type


def _int(v: "str | None", default: "int | None" = None) -> "int | None":
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default


def _model_error(exc: Exception) -> dict:
    """An exception from a model view, as what the page shows: a typed `ArchDriftError` gets its ONE plain
    sentence (`exc.plain`; the generic `ARCH_OLDER_TEXT` only for an error built without one) with the
    diagnosis folded under it, and `reason` carries its `arch_status` kind; anything else its message."""
    from main.prober.model import ArchDriftError

    if isinstance(exc, ArchDriftError):
        return {"kind": "arch_drift", "reason": exc.kind, "text": exc.plain or ARCH_OLDER_TEXT,
                "detail": exc.detail}
    return {"kind": "error", "reason": None, "text": f"{type(exc).__name__}: {exc}", "detail": None}


def model_http_error(exc: Exception) -> HTTPException:
    """The 400 a JSON route raises when a MODEL call failed. The `error` string is what it always was
    (`Type: message`); a typed `ArchDriftError` also carries `kind` (an `arch_status` kind) and `plain` (the one
    sentence), which `app._http_error` copies into the response body, so an agent can branch on the kind instead
    of parsing prose."""
    detail = f"{type(exc).__name__}: {exc}"
    from main.prober.model import ArchDriftError

    if isinstance(exc, ArchDriftError):
        err = HTTPException(status_code=400, detail=detail)
        err.extra = {"kind": exc.kind, "plain": exc.plain}          # type: ignore[attr-defined]
        return err
    return HTTPException(status_code=400, detail=detail)


def _neighbours(rows: "list[dict]", short_id: str) -> "tuple[str | None, str | None]":
    ids = [r["short_id"] for r in rows]
    if short_id not in ids:
        return None, None
    i = ids.index(short_id)
    return (ids[i - 1] if i > 0 else None), (ids[i + 1] if i + 1 < len(ids) else None)


def _decision_for_turn(story: dict, turn: "int | None") -> "int | None":
    """The first decision at or after game turn ``turn`` (how `/battle?start=N` lands)."""
    if turn is None:
        return None
    for d in story.get("decisions") or []:
        if int(d.get("turn") or 0) >= int(turn):
            return int(d["inv"])
    return None


def game_url(run: "str | None", battle: str, inv: "int | None" = None, view: str = "model") -> str:
    q: dict = {"run": run or "", "battle": battle}
    if inv is not None:
        q["inv"] = int(inv)
    if view and view != "model":
        q["view"] = view
    return "/game?" + urlencode(q)


def register_game_routes(app: FastAPI, h: GameHelpers, newest_first: Callable[[list], list],
                         picker: Callable[[list, dict], list]) -> None:
    """`picker` is the app's ONE battle picker (`app._picker_rows`: newest first, capped, and ALWAYS
    containing the battle shown — `/game` used to slice its own list and named a different battle
    as selected whenever the shown one fell outside it, measured 2026-10-09)."""
    @app.get("/api/game/story", tags=["read-only"], response_model=dict,
             summary="One battle's turn story: per turn the protocol events, the ordered beats, the board; "
                     "the decisions (model-free)")
    def api_game_story(run: "str | None" = Query(None), battle: "str | None" = Query(None)) -> dict:
        sess = h.session(h.pick(run))
        row = h.battle_row(sess, battle)
        data, err = h.guarded(lambda: sess.battle_story(row["id"]))
        if err:
            raise HTTPException(status_code=400, detail=err)
        return data

    @app.get("/api/game/board", tags=["read-only"], response_model=dict,
             summary="The board one decision was made on, every fact tagged public / ours / hidden (model-free)")
    def api_game_board(run: "str | None" = Query(None), battle: "str | None" = Query(None),
                       inv: int = Query(0, ge=0)) -> dict:
        sess = h.session(h.pick(run))
        row = h.battle_row(sess, battle)
        data, err = h.guarded(lambda: sess.battle_board(row["id"], inv))
        if err:
            raise HTTPException(status_code=400, detail=err)
        return data

    @app.get("/api/game/readout", tags=["read-only"], response_model=dict, dependencies=[model_gate("/game")],
             summary="One battle's model panels: intent, beliefs, attention summary, pointer scores, "
                     "operator facts (LOADS the checkpoint; current architecture only)")
    def api_game_readout(run: "str | None" = Query(None), battle: "str | None" = Query(None)) -> dict:
        sess = h.session(h.pick(run))
        row = h.battle_row(sess, battle)
        try:
            return sess.battle_readout(row["id"])
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — a model that cannot load is a typed 400, not a 500
            raise model_http_error(exc) from exc

    @app.get("/api/game/attention", tags=["read-only"], response_model=dict, dependencies=[model_gate("/game")],
             summary="One decision's full attention map (query × key), one layer/head or their average")
    def api_game_attention(run: "str | None" = Query(None), battle: "str | None" = Query(None),
                           inv: int = Query(0, ge=0), layer: "int | None" = Query(None, ge=0),
                           head: "int | None" = Query(None, ge=0)) -> dict:
        sess = h.session(h.pick(run))
        row = h.battle_row(sess, battle)
        try:
            return sess.decision_attention(row["id"], inv, layer, head)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise model_http_error(exc) from exc

    @app.get("/battle", response_class=RedirectResponse, status_code=307, tags=["pages"],
             summary="The classic replay, MERGED into /game (2026-10-09): redirects there, keeping the "
                     "battle and mapping start=N to turn=N")
    def page_battle_redirect(run: "str | None" = Query(None), battle: "str | None" = Query(None),
                             start: "str | None" = Query(None, description="game turn (mapped to turn=)")
                             ) -> RedirectResponse:
        q: dict = {}
        if run:
            q["run"] = run
        if battle:
            q["battle"] = battle
        if _int(start) is not None:
            q["turn"] = _int(start)
        return RedirectResponse("/game" + (("?" + urlencode(q)) if q else ""), status_code=307)

    @app.get("/game", response_class=HTMLResponse, tags=["pages"],
             summary="The battle viewer: the turn story, the board under an information perspective, our "
                     "options vs their action, and (unlocked) the model's beliefs, intent and attention")
    def page_game(request: Request, run: "str | None" = Query(None), battle: "str | None" = Query(None),
                  inv: "str | None" = Query(None, description="decision index; default = the first"),
                  turn: "str | None" = Query(None, description="game turn; selects its first decision"),
                  view: "str | None" = Query(None, description="perspective: model (default) | truth | public")
                  ) -> HTMLResponse:
        persp = normalize(view)
        name, data, err = h.load(run, lambda s: s.run_summary())
        sess = h.session(h.pick(run))
        try:
            row = h.battle_row(sess, battle)
        except h.no_battles as empty:
            return h.page(request, "game.html", "game", name, summary=data, error=err, story=None,
                          empty=empty.detail, battles=[], selected=None, view=persp)
        story, serr = h.guarded(lambda: sess.battle_story(row["id"]))
        # Can the MODEL panels run on this battle's step? Model-free (recorded identity + a checkpoint that
        # resolves; nothing is loaded). When they cannot, the model slots render the plain reason on first
        # paint — no loader to wait on, and no password prompt for a view the password cannot unlock.
        model_status = sess.model_status(row["id"])
        listing = newest_first(sess.battles())
        prev_b, next_b = _neighbours(listing, row["short_id"])
        n = (story or {}).get("n_decisions") or 0
        want = _int(inv)
        if want is None and story is not None:
            want = _decision_for_turn(story, _int(turn))
        i = min(max(want or 0, 0), max(n - 1, 0))
        dec = (story or {}).get("decisions", [None])[i] if n else None
        turn_no = int((dec or {}).get("turn") or 0)
        cur_turn = next((t for t in (story or {}).get("turns", []) if int(t["turn"]) == turn_no), None)
        board, berr = (h.guarded(lambda: sess.battle_board(row["id"], i)) if dec is not None else (None, None))
        nxt = (story or {}).get("decisions", [])[i + 1] if (dec is not None and i + 1 < n) else None
        wp_spec = (charts.game_pwin_strip_spec(story["decisions"], selected=i,
                                               href=lambda k: game_url(run, row["short_id"], k, persp))
                   if story and any(d.get("p_win") is not None for d in story["decisions"]) else None)
        return h.page(request, "game.html", "game", name, summary=data, error=err or serr or berr, story=story,
                      selected=row, battles=picker(listing, row), inv=i, dec=dec, next_dec=nxt,
                      turn=cur_turn, board=board, view=persp, perspectives=PERSPECTIVES,
                      model_status=model_status,
                      perspective_words=PERSPECTIVE_WORDS, wp_spec=wp_spec,
                      prev_battle=prev_b, next_battle=next_b, empty=None)

    @app.get("/partials/game/model", response_class=HTMLResponse, tags=["partials"],
             dependencies=[model_gate("/game")],
             summary="/game's model panels for one decision (HTMX target; loads the checkpoint)")
    def partial_game_model(request: Request, run: "str | None" = Query(None),
                           battle: "str | None" = Query(None), inv: "str | None" = Query("0"),
                           view: "str | None" = Query(None)) -> HTMLResponse:
        persp = normalize(view)
        sess = h.session(h.pick(run))
        row = h.battle_row(sess, battle)
        i = _int(inv, 0) or 0
        board, _berr = h.guarded(lambda: sess.battle_board(row["id"], i))
        story, _serr = h.guarded(lambda: sess.battle_story(row["id"]))
        try:
            r = sess.battle_readout(row["id"])
        except Exception as exc:  # noqa: BLE001 — an older arch is the ordinary case on an archived run
            return h.fragment(request, "partials/game_model.html", r=None, error=_model_error(exc),
                              run=run, battle=row["short_id"], inv=i, board=board, story=story, view=persp)
        n = int(r.get("n_decisions") or 0)
        i = min(max(i, 0), max(n - 1, 0))
        d = r["decisions"][i] if n else None
        bel = charts.game_belief_spec(r.get("belief_evolution"), selected=i)
        return h.fragment(request, "partials/game_model.html", r=r, d=d, error=None, run=run,
                          battle=row["short_id"], inv=i, belief_spec=bel, board=board, story=story, view=persp)

    @app.get("/partials/game/attention", response_class=HTMLResponse, tags=["partials"],
             dependencies=[model_gate("/game")],
             summary="/game's attention heat map for one decision (HTMX target)")
    def partial_game_attention(request: Request, run: "str | None" = Query(None),
                               battle: "str | None" = Query(None), inv: "str | None" = Query("0"),
                               layer: "str | None" = Query(None), head: "str | None" = Query(None)
                               ) -> HTMLResponse:
        sess = h.session(h.pick(run))
        row = h.battle_row(sess, battle)
        i, lay, hd = _int(inv, 0) or 0, _int(layer), _int(head)
        try:
            att = sess.decision_attention(row["id"], i, lay, hd)
        except Exception as exc:  # noqa: BLE001
            return h.fragment(request, "partials/game_attention.html", att=None, spec=None,
                              error=_model_error(exc), run=run, battle=row["short_id"], inv=i)
        return h.fragment(request, "partials/game_attention.html", att=att, spec=charts.game_attention_spec(att),
                          error=None, run=run, battle=row["short_id"], inv=i)
