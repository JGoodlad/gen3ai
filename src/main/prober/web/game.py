"""`/game` — the battle viewer (`designs/prober/battle_view_v2.md`): its page, its two HTMX fragments
and its three JSON endpoints. Registered onto the app by `create_app` (kept out of `app.py`, which is
already past the 1,000-line report line).

The one rule holds: every number comes from a `ProbeSession` method (`battle_story`,
`battle_readout`, `decision_attention`) verbatim; this module only picks the decision, builds the
chart specs (`charts.py`) and renders.

The page is a plain GET (`/game?run=…&battle=…&inv=N`), like `/battle`: a position in a battle is a
thing you link to. The model panels arrive as an HTMX fragment because they load a checkpoint, and
on a run whose architecture is older than the code they render one plain sentence (plus the
`ArchDriftError` diagnosis folded under it) — never a 500, never a blank panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from main.prober.web import charts

ARCH_OLDER_TEXT = ("This run's architecture is older than the code, so the model views (intent, "
                   "attention, operator facts) cannot load its checkpoint — they need a "
                   "current-architecture checkpoint. The turn story above is model-free and still works.")


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
    """An exception from a model view, as what the page shows: the arch-drift case gets the plain
    sentence + its diagnosis, anything else its message."""
    from main.prober.model import ArchDriftError

    if isinstance(exc, ArchDriftError):
        return {"kind": "arch_drift", "text": ARCH_OLDER_TEXT, "detail": str(exc)}
    return {"kind": "error", "text": f"{type(exc).__name__}: {exc}", "detail": None}


def _neighbours(rows: "list[dict]", short_id: str) -> "tuple[str | None, str | None]":
    ids = [r["short_id"] for r in rows]
    if short_id not in ids:
        return None, None
    i = ids.index(short_id)
    return (ids[i - 1] if i > 0 else None), (ids[i + 1] if i + 1 < len(ids) else None)


def register_game_routes(app: FastAPI, h: GameHelpers, newest_first: Callable[[list], list]) -> None:
    @app.get("/api/game/story", tags=["read-only"], response_model=dict,
             summary="One battle's turn story: per turn the protocol events + the board; the decisions (model-free)")
    def api_game_story(run: "str | None" = Query(None), battle: "str | None" = Query(None)) -> dict:
        sess = h.session(h.pick(run))
        row = h.battle_row(sess, battle)
        data, err = h.guarded(lambda: sess.battle_story(row["id"]))
        if err:
            raise HTTPException(status_code=400, detail=err)
        return data

    @app.get("/api/game/readout", tags=["read-only"], response_model=dict,
             summary="One battle's model panels: intent, hypotheses, attention summary, pointer scores, "
                     "operator facts (LOADS the checkpoint; current architecture only)")
    def api_game_readout(run: "str | None" = Query(None), battle: "str | None" = Query(None)) -> dict:
        sess = h.session(h.pick(run))
        row = h.battle_row(sess, battle)
        data, err = h.guarded(lambda: sess.battle_readout(row["id"]))
        if err:
            raise HTTPException(status_code=400, detail=err)
        return data

    @app.get("/api/game/attention", tags=["read-only"], response_model=dict,
             summary="One decision's full attention map (query × key), one layer/head or their average")
    def api_game_attention(run: "str | None" = Query(None), battle: "str | None" = Query(None),
                           inv: int = Query(0, ge=0), layer: "int | None" = Query(None, ge=0),
                           head: "int | None" = Query(None, ge=0)) -> dict:
        sess = h.session(h.pick(run))
        row = h.battle_row(sess, battle)
        data, err = h.guarded(lambda: sess.decision_attention(row["id"], inv, layer, head))
        if err:
            raise HTTPException(status_code=400, detail=err)
        return data

    @app.get("/game", response_class=HTMLResponse, tags=["pages"],
             summary="Battle viewer: the turn story, opponent intent, attention, operator facts")
    def page_game(request: Request, run: "str | None" = Query(None), battle: "str | None" = Query(None),
                  inv: "str | None" = Query(None, description="decision index; default = the first")
                  ) -> HTMLResponse:
        name, data, err = h.load(run, lambda s: s.run_summary())
        sess = h.session(h.pick(run))
        try:
            row = h.battle_row(sess, battle)
        except h.no_battles as empty:
            return h.page(request, "game.html", "game", name, summary=data, error=err, story=None,
                          empty=empty.detail, battles=[], selected=None)
        story, serr = h.guarded(lambda: sess.battle_story(row["id"]))
        listing = newest_first(sess.battles())
        prev_b, next_b = _neighbours(listing, row["short_id"])
        n = (story or {}).get("n_decisions") or 0
        i = min(max(_int(inv, 0) or 0, 0), max(n - 1, 0))
        dec = (story or {}).get("decisions", [None])[i] if n else None
        turn_no = int((dec or {}).get("turn") or 0)
        turn = next((t for t in (story or {}).get("turns", []) if int(t["turn"]) == turn_no), None)
        prev_turn = next((t for t in reversed((story or {}).get("turns", [])) if int(t["turn"]) < turn_no), None)
        return h.page(request, "game.html", "game", name, summary=data, error=err or serr, story=story,
                      selected=row, battles=listing[:200], inv=i, dec=dec, turn=turn,
                      board_before=(prev_turn or {}).get("board"),
                      prev_battle=prev_b, next_battle=next_b, empty=None)

    @app.get("/partials/game/model", response_class=HTMLResponse, tags=["partials"],
             summary="/game's model panels for one decision (HTMX target; loads the checkpoint)")
    def partial_game_model(request: Request, run: "str | None" = Query(None),
                           battle: "str | None" = Query(None), inv: "str | None" = Query("0")) -> HTMLResponse:
        sess = h.session(h.pick(run))
        row = h.battle_row(sess, battle)
        i = _int(inv, 0) or 0
        try:
            r = sess.battle_readout(row["id"])
        except Exception as exc:  # noqa: BLE001 — an older arch is the ordinary case on an archived run
            return h.fragment(request, "partials/game_model.html", r=None, error=_model_error(exc),
                              run=run, battle=row["short_id"], inv=i)
        n = int(r.get("n_decisions") or 0)
        i = min(max(i, 0), max(n - 1, 0))
        d = r["decisions"][i] if n else None
        wp = charts.game_win_prob_spec(r["win_prob_series"], selected=i) if r.get("win_prob_series") else None
        bel = charts.game_belief_spec(r.get("belief_evolution"), selected=i)
        return h.fragment(request, "partials/game_model.html", r=r, d=d, error=None, run=run,
                          battle=row["short_id"], inv=i, wp_spec=wp, belief_spec=bel)

    @app.get("/partials/game/attention", response_class=HTMLResponse, tags=["partials"],
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
