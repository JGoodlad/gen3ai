# The prober web app's PAGES — information flow, the critic's currency, `/game`, `/analyze`, awareness

Owned by this tree (always current), lifted out of `src/main/prober/web/CLAUDE.md` on 2026-10-10.
The battle viewer's field map is [`battle_view_v2.md`](battle_view_v2.md) and its reasons
[`battle_viewer_ux_2026-10-09.md`](battle_viewer_ux_2026-10-09.md); the service side (deployment,
picker, access, jobs) is [`web_service.md`](web_service.md); the tests [`web_tests.md`](web_tests.md).

## 1. Information flow (why the pages are shaped the way they are)

**`/game` is THE battle viewer** (2026-10-09, `2f8197ea`): the classic `/battle` replay was MERGED into
it and `/battle` answers 307 to `/game` (`start=N` → `turn=N`), so every old link lands on the same
battle and turn ([`web_views.md`](web_views.md)). `/analyze` has no nav tab — it is a per-DECISION page,
opened from every decision in the viewer and every scan row.

The TUI's standing complaint was that *the information didn't flow*. The deliberate answers, each
pinned in `app_test.py` → "usability / information flow":

- **The nav follows the investigation recipe**: `_NAV` = `run → triage → scan → battles → game →
  falsify → calibration` (triage is "start here for 'what next'", `src/main/prober/CLAUDE.md`).
- **`/` carries a "where to start" card** — the views in recipe order, each with the QUESTION it answers
  (`VIEW_QUESTIONS` in `app.py`).
- **A context strip on every page** names the run, its step count and its W/L, so a screenshot of a
  chart records what it is of.
- **Scan rows are not dead ends.** Each row links to `/analyze` for its worst decision, carries
  `data-battle-id` and copy buttons for the id and for the exact `python -m main.prober.query analyze
  <id> <inv>` line (the CLI stays offered as a real second surface).
- **First paint is populated where that is cheap.** Measured on a real run: battles **119 ms**, triage
  **436 ms**, scan **2006 ms**. Battles and triage render server-side; scan stays async (2 s of white
  page is worse than a page that fills in) and its waiting state says what it is doing.
- **The battles table is capped at `_BATTLE_PAGE` (200) and says so.** Uncapped it emitted 2397 rows /
  545 KB.
- **`/battles` preselects the NEWEST eval step, and its rows open the viewer.** "All steps" stays one
  selection away. Each row carries `data-href` (a delegated handler in `app.js` navigates on a row click,
  ignoring clicks on a button or link) **and** a real `<a>` in the id cell — the `<a>` works with
  JavaScript off, gives the row a keyboard tab stop, and lets open-in-new-tab behave.
- **SENTINELS COME FIRST on `/battles`** — in the opponent dropdown (`_opponents`) and in the rows
  (`_by_opponent_strength`), both through the one shared key `engine.opponent_rank`. A sentinel game says
  most about where the model is now, and with the 200-row cap an alphabetical sort could cut one
  entirely. ⚠ **`sentinel_0` is the STRONGEST, not the oldest** — the index is a strength rank and the
  labels FLOAT (a promotion re-seats every sentinel); `engine_test.py::test_sentinel_zero_is_the_STRONGEST_and_sorts_first`.
  The fixed bots move as a BLOCK in their existing order (ranking them by strength is the ELO ladder's
  claim). `/api/battles` is deliberately NOT reordered — row order is a presentation choice.
- **The run picker is grouped by architecture** (`_picker`): current · older · hidden-no-traces
  ([`web_service.md`](web_service.md) §3).
- **Cryptic columns explain themselves** via `title=` (`inv`, `ΔV`, `TD δ`), and `wp_coverage=0` is
  rendered as prose — it means the winning/behind split fell back to `V > 0`, which the project's own
  docs call systematically wrong.

## 2. The critic's CURRENCY, and the thresholds that depend on it

The win-prob critic (the only critic) makes V a **probability in [0,1]** instead of a shaped return of
roughly ±30, and `values` then EQUALS `win_probs`. `ProbeSession.critic_currency()` computes it (riding
`run_summary()`, `battle_turns()`, `analyze()`, `calibration()`); this package only has to stop
CONTRADICTING it. Archived runs with no `critic` key read as shaped (`src/main/prober/CLAUDE.md`).

**A "default" typed into a handler is not a default.** `/api/triage` once declared
`v_even: float = Query(0.0)` and `/calibration` `overvalue_tau = Form("5.0")`, so both reached the
session as EXPLICIT arguments and defeated its per-currency resolution — the engine's `if v_even is None`
branch was dead on every web request. Both are now `None`/blank by default (`_form_opt_float` keeps
"unset" and "typed a number" tellable apart, and treats an unparseable value as unset). The calibration
form's placeholder shows the run's own `default_overvalue_tau`.

On a winprob run a shaped τ of 5.0 exceeds the whole range a probability gap can occupy, so
`critic_overvalued` read a confident **0%**. `calibration_result.html` renders the engine's
**`threshold_warning`** ABOVE the numbers it invalidates.

`/` names the mode in its stat strip and spells out the units. `/analyze`'s critic card and the
counterfactual job tables branch on `is_probability`: under winprob they say P(win) **is** V and mark
the P(win) tiles `= V` — two panels showing one number while the copy promises two estimators is a
reader trap.

**The reliability charts follow the currency.** On a win-prob critic the curve is a PROBABILITY
CALIBRATION — axes `predicted P(win)` / `realized win rate`, the gap chart's y-axis `gap (pp)` — not
`recorded V(s)` / `realized return G(s)`. `charts_test.py` pins both eras, incl. that the winprob spec
contains "realized return" nowhere.

**DEAD under this era — a census candidate for deletion, deliberately NOT deleted here.** The
distributional value head is gone (L1; no new trace carries `value_dist`), so every awareness surface
degrades to empty. Nothing 500s:

| surface | state |
|---|---|
| `index.html` "Did it know?" card | heading + blurb always render; the swap fills with `{{ data.error }}` — a permanently empty card |
| `scan_table.html` `knew @` / `lead` columns | headers always render, every cell `—` |
| `triage_table.html` `blind` / `median lead` columns | same |
| `/api/awareness`'s `lead_bar` / `cap_turn` / `stall_bar` | inert (their chart was deleted in L1) |

**The era's offline instruments are LINKED from `/`, not re-implemented.** Each owns statistics this app
does not (a Bradley-Terry ladder, a cluster bootstrap, a selection reweighting that REFUSES), and
re-deriving one in a handler breaks *the one rule*. So `/` carries a card with the commands
(`critic_gate` only on a winprob run). ⚠ **A command in a template is a command someone will paste** —
they were checked against the live parsers (which caught `main.untaught_meter` taking its refs
POSITIONALLY), and the leaf keeps them so the `CLAUDE.md` freshness gate resolves their flags.

## 3. `/game` — THE battle viewer

The field map is **`battle_view_v2.md`**; the reasons — the owner's complaints, the audit, every removal
— are **`battle_viewer_ux_2026-10-09.md`**. The rules: the page is a plain GET
(`/game?run=…&battle=…&inv=N&view=…` — a decision and its perspective are things you link to, and it
works with JavaScript off); the model's half is an HTMX fragment because it loads a checkpoint; the
battle picker is the app's ONE `_picker_rows` (newest first, and ALWAYS containing the battle shown —
`/game` used to slice its own list and name a different battle as selected); the INFORMATION
PERSPECTIVE is decided per field by `engine/perspective.shown`, injected into Jinja as the global `shown`
and applied by the one macro `fv` — a board fact rendered any other way is what
`perspective_guard_test.py` catches. Routes live in `game.py`.

**A run with no traces is an EMPTY STATE, not a 404.** The app opens a run by default and a
freshly-launched run has no traces until its first eval cycle. `_NoBattles` is caught by the two page
handlers (`/game`, `/analyze`) and rendered as a message pointing at the run summary; the JSON API still
returns the status code, and an unknown battle token is still a real 404 that never echoes the token
(`test_a_run_with_no_traces_is_an_empty_state_not_a_404`, `test_a_battle_that_does_not_exist_is_still_a_404`).

**A battle is named by its `short_id`, checked for MEMBERSHIP** — `runs.py`'s rule one level down.
`ProbeSession._battle` falls back to `build_trace_tree(battle_id)` for an id it does not recognise, which
will happily open a `*_summary.json` belonging to **another run**; `app.battle_row()` matches the token
against the run's own listing and passes the session a path the *server* produced. Without it a pinned
single-run instance served a sibling run's trace with a **200**.

## 4. "Did it know?" — the awareness layer

One fold (`main/prober/awareness.py`), model-free, `None` throughout on runs with no distributional head
(every run since L1):

| where | what it adds |
|---|---|
| `/scan` | `knew @` and `lead` columns beside each crater — `BLIND` badged when it never saw it coming |
| `/triage` | a `blind` / `median lead` column per category, **beside** the lever, never folded into it |
| `/` | the run-level panel: the aggregate against the published **gen-10 baseline** |

**The baseline is data, not copy** — `awareness.AWARENESS_BASELINES`, riding `awareness_scan()`'s payload
as `aggregate.baseline`. Two honesty rules in the markup: **cap-aware@5 prints its `n` on both sides**
("a direction, not a rate"; the baseline is over **12** cap losses), and **the two coverage rows are NOT
comparable at the default filter** (the baseline was over ALL outcomes, the panel defaults to losses, so
a filtered PIT is biased low by construction — the page says so and links the unfiltered read; the CLI
gained `--outcome all` for the same reason). It loads **async on `/`** (`hx-trigger="load"`): measured
on gen-11 (1292 losses), `awareness_scan()` **5.4 s** cold vs `scan()`'s 2.0 s warm, beside a live trainer.

## 5. `/analyze` — a view that loads a checkpoint (password-gated)

The per-decision forensic read: page shell + an HTMX fragment (`hx-trigger="load"`), the battle and
`inv` plain GET params. Every hand-off (the viewer's decisions, every `scan` row) LINKS to
`/analyze?run=…&battle=…&inv=N`; until 2026-08-18 the copy around it pointed at the TUI retired on
2026-08-13 — a retirement must sweep the user-facing copy, not just the module
(`app_test.py::test_every_hand_off_goes_to_a_web_view_not_a_retired_terminal` asserts "the TUI" appears
in neither view).

**Arch drift is its ordinary output on an old run** (`designs/prober/arch_drift.md`; measured 2026-10:
none of the 38 archived runs with traces can load). On first paint the page asks
`ProbeSession.model_status` (model-free) and, when the views cannot run, renders the reason at once
(`partials/model_unavailable.html`, `data-model-reason="<kind>"`) instead of a loader — and instead of a
password prompt for a view the password cannot unlock. The fragment route renders the same sentence
over the whole diagnosis (obs dim, `arch_signature`, dropped flags, the exact `git checkout`) in `.err`
(`white-space: pre-wrap`); the JSON routes answer 400 `{error, kind, plain}`.

Two faithfulness banners must never be dropped: `obs_mismatch` (every obs-offset decode below reads past
a divergence) and `model_resolution.dropped_kwargs` (flags the current code no longer accepts were
dropped to make the load possible, so the rebuilt extractor is not the one that played). Panels
self-hide per flag-gated head, so an absent panel reads as "that head was off".

The operator's `incoming_matrix` is a real **heatmap table** (opp candidate move × our mon).

**The beliefs section carries TWO readings of the species belief, in a load-bearing order**
(`#beliefs-exclusive`, `a.exclusive_belief`): the raw per-slot marginals first — what the model actually
believes — and the **species-clause reading** below it, QUIET when the belief was already coherent and
saying in its own copy that it changes nothing (gen-15: **14.2%** of decisions displayed two hidden slots
naming the same mon). See `src/main/prober/CLAUDE.md` → *The SPECIES-CLAUSE reading*.

**The counterfactual tier rides `/analyze`** — `lookahead`, `better_line` and `replay_counterfactual`
launch from the bottom of the page pre-filled with the battle + inv, through the **job registry** (submit
→ job id → poll `/partials/job/{id}`), **password-gated**. A finished `lookahead` offers its best
non-chosen alternative straight to the replay probe; `better_line` renders the interior-opponent
provenance with a **self-proxy banner** at depth ≥ 2; `replay_counterfactual` at `n_rollouts=1` is a
single realized-dice line and **not** a probability — the session says so in `caveats` and the page
renders them. `interior_opponent="ckpt"` is NOT exposed: it takes a filesystem path from the client,
which is what `runs.py`'s membership rule exists to prevent. CLI-only.

⚠ **A COVERAGE HOLE.** The fixture run has no loadable checkpoint, so the browser gate measures
`/analyze`'s shell and its arch-drift state only; the POPULATED markup (the heatmap, the wide
faithfulness/threat tables) is covered by unit tests and an eyeball, **not** the measured layout gate.
Until a fixture checkpoint at the current architecture exists, "the heatmap does not overflow on a
phone" is unproven.

## 6. The `/falsify` and `/calibration` forms

⚠ **An HTMX `<form hx-post>` sends its fields in the BODY — declare them `Form(...)`, never `Query(...)`.**
Both pages shipped with `Query`, so every control was ignored: a submit of
`outcome=win/limit=3/seeds=7/concurrency=4` reached the session as `loss/20/32/1` — asking for wins
quietly scanned losses. Pinned by `test_the_page_form_fields_actually_reach_the_probe`. `run` stays a
`Query`: it rides the URL, because the run picker is a link.
