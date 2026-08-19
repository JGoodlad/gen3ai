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

**This is the only human-facing surface** — the Textual TUI (`main.prober.app`) was retired
2026-08-13. The rule it was built under still holds and matters more now: nothing here contains
analysis. Every number comes from `ProbeSession`, so the browser and `query.py` cannot disagree
about a run.

## How to actually look at it

**It is DEPLOYED at https://prober.g5d.io** — a `gen3ai-prober-web.service` systemd user unit
binding `127.0.0.1:6008`, with `cloudflared` forwarding to it, exactly like TensorBoard (`:6006`)
and the model viewer (`:6007`). The unit is reference-copied at
`scripts/workstation/gen3ai-prober-web.service`; it is pointed at **`models/`** (not one run) so a
new generation needs no restart — the header carries the run picker. **No Cloudflare Access**: the
owner's decision (2026-08-09) is that this is an open-source model whose outcomes and traces are
meant to be public, same posture as `model.g5d.io`.

*(This section previously said "LOCAL ONLY — there is no g5d.io hostname for it". That was true
when written and false by 2026-08-18, which is a good illustration of why deployment facts belong
next to the thing deployed.)*

```bash
# ad hoc, on the workstation (the deployed instance is the systemd unit above)
export PYTHONPATH=$PYTHONPATH:src && python -m main.prober.web /home/goodlad/dev/gen3ai/models/

# from anywhere else without the tunnel, over SSH
ssh -p 2222 -L 6008:localhost:6008 goodlad@workstation.g5d.io   # then http://localhost:6008
```

`models/` exists only in the **main checkout**, never in a worktree — pass an absolute
`models/...` path when serving from one.

Operational detail is in `scripts/workstation/GCP_INFRASTRUCTURE.md` → *Prober web views*.

### ⚠ ONE REVISION, or a 500 — the staleness contract

**Jinja reloads a changed template from disk. Python cannot reload a changed module.** So a
long-lived server drifts into serving NEW templates against OLD code, and that is not a stale page
but a broken one.

**Measured 2026-08-18**, and the shape is worth remembering: the service had been up **5 days**;
every `/battle` for a current run returned **HTTP 500** —
`UndefinedError: 'dict object' has no attribute 'win_prob'` — because a template shipped two days
earlier read a key the running `session.py` predated. `Restart=always` never fired: **nothing had
crashed.** systemd called the unit healthy, the tunnel agreed, and the only symptom was a user
saying they could not load a game.

Two mechanisms close it, and both are needed:

1. **Templates are pinned to the process** (`templates.env.auto_reload = False`, set in
   `create_app`). A stale process now serves a COHERENT old page instead of a hybrid. The cost is
   real and accepted: editing a template locally needs a restart.
2. **A watchdog replaces the process when it falls behind** —
   `scripts/workstation/prober_web_watchdog.sh`, driven by a systemd `.timer` every 2 minutes. It
   compares `/api/health`'s **`revision`** (the git sha of the source THIS PROCESS imported,
   captured once, never re-read) against the repo's HEAD, and restarts on a mismatch. It **defers
   while `jobs_running > 0`** — a restart kills a multi-minute `falsify_scan` — and it **verifies
   the replacement actually came up on the new revision**, because "restarted successfully" into a
   crash-loop is the same class of lie it exists to catch.

`revision` is keyed on the SOURCE directory, not the process CWD: those coincide today and would
silently diverge the moment anyone served a worktree. Known limit, stated rather than papered
over: an **uncommitted** edit does not move HEAD, so it does not trigger a restart — correct on
this box, where main only advances by commit, but hand-edit under the service and you restart it
yourself.

Gates: `staleness_test.py` (both halves — the pin asserted AND behaviourally proven by editing a
template mid-process, each verified to fail when `auto_reload` is put back; the watchdog driven end
to end through the real script with a stubbed `systemctl`, covering current / stale / no-revision /
job-deferred / unit-stopped / unreachable).

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
- **`/battles` preselects the NEWEST eval step, and its rows open the replay.** A run holds every
  cycle it ever ran; the question behind opening the page is essentially always "what is the
  *current* model doing", and an all-steps default answered that with a 200-row cap sliced out of
  an arbitrary mixture of checkpoints. "All steps" stays one selection away. Each row then carries
  `data-href` (a delegated handler in `app.js` navigates on a row click, ignoring clicks that land
  on a button or link) **and** a real `<a>` in the id cell — the `<a>` is what makes it work with
  JavaScript off, gives the row a keyboard tab stop, and lets open-in-new-tab behave; a JS-only row
  click has none of those.
- **SENTINELS COME FIRST on `/battles`** — in the opponent dropdown (`_opponents`) and in the rows
  themselves (`_by_opponent_strength`), both through the one shared key `engine.opponent_rank`. A
  sentinel is the trainee against a recent SELF, so those games say most about where the model is
  now; scattered alphabetically among nine fixed bots they were something to hunt for, and with the
  200-row cap a sentinel game could be cut entirely by an alphabetical accident.
  ⚠ **`sentinel_0` is the STRONGEST, not the oldest.** The index is a strength rank and the labels
  FLOAT — a promotion re-seats every sentinel — so ascending index is descending strength. Getting
  that backwards would put the weakest opponent at the top of the list while looking equally
  deliberate, which is why `test_sentinel_zero_is_the_STRONGEST_and_sorts_first` says so in its
  name. The fixed bots are moved as a BLOCK in their existing order rather than re-sorted: ranking
  them by strength is a different claim, and the ELO ladder owns it.
  `/api/battles` is deliberately NOT reordered — row order is a presentation choice about which
  rows a human meets first, and a machine client asked for the run's battles, not for this page's
  opinion about them.
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
| `/` run | `run_summary()` + `awareness_scan()` | steps · per-step identity · opponents · checkpoints · γ, **plus the "did it know?" panel** (async, see below) |
| `/battles` | `battles()` | outcome / opponent / step filters |
| `/scan` | `scan()` | each battle's worst turning point, ranked (model-free) |
| `/triage` | `triage()` | failure categories ranked by recoverable win-rate |
| `/battle` | `battle_turns()` | **one game, turn by turn** — board · expected opponent intent (α/β) · battle log · critic · **P(win) and the P(loss) strip** (model-free) |
| `/analyze` | `analyze()` | **one decision, all the way down** — faithfulness · beliefs · threats · intervention · saliency. **LOADS THE CHECKPOINT** (see below) |
| `/falsify` | `falsify_scan()` | the crater bracket — **a background job** |
| `/calibration` | `calibration()` | the reliability curve — **a background job** |
| (on `/analyze`) | `lookahead()` · `better_line()` · `replay_counterfactual()` | the counterfactual tier — **background jobs**, password-gated |

⚠ **An HTMX `<form hx-post>` sends its fields in the BODY — declare them `Form(...)`, never
`Query(...)`.** `/falsify` and `/calibration` shipped with `Query`, so every control on both pages
was silently ignored and the probes ran at their defaults: a submit of
`outcome=win/limit=3/seeds=7/concurrency=4` reached the session as `loss/20/32/1`. The `outcome`
one made it a CORRECTNESS bug rather than an inconvenience — asking for wins quietly scanned losses
and returned a confident answer to a question nobody asked. Fixed, and pinned by
`test_the_page_form_fields_actually_reach_the_probe` (verified to fail on the reverted code).
`run` stays a `Query` on purpose: it rides the URL, because the run picker is a link.

### `/battle` — the turn-by-turn replay

The read-it-like-a-game view: decisions grouped by GAME TURN, each with the board it was made on,
the ordered battle log of what then happened, and V / ΔV / TD δ / reward. `scan` answers *which*
decision lost the battle; this answers *how the game went* around it.

Three things about it are deliberate:

- **It is NOT HTMX.** Every other page refilters a table in place; this one is a thing you read and
  LINK to ("look at turn 47"). So the battle picker and the turn window are plain GET forms and
  links — every position in the replay is a shareable URL that also works with JavaScript off.
  `scan` and `battles` rows carry a **turns** link, and scan's opens the replay *windowed on the
  losing turn and anchored on the exact decision*.
- **It is windowed at `_TURN_PAGE` (50 turns).** Measured: the longest real battle is **249 turns /
  821 KB** of session JSON — the same "download, not a page" failure `_BATTLE_PAGE` exists to
  prevent, except this one lands on a phone. Nothing is unreachable: prev/next links plus the
  session's own `notable` jump targets (worst value drops, faints) reach any turn.
- **Each decision has a collapsed drop-down** carrying the TUI's per-decision detail, restricted to
  what needs **no checkpoint**: the full recorded action distribution (bars, chosen marked, illegal
  actions *grey* — never red, mirroring the TUI's `_DISABLED_GREY`: "unavailable" must not read as
  "dangerous"), the full `α`/`β` distributions, both benches with items/movesets, the reward
  component breakdown, the events, and the **raw Showdown protocol for that turn**. The footer names
  what is missing and where it lives, rather than leaving the reader to wonder — beliefs, threat
  tables, saliency and the re-run distribution all need the model, so they stay in `analyze` / the
  TUI.
- **The card carries what the model expected the OPPONENT to do** (`opp_intent`, the v67 `α`/`β`
  heads), between the board and our choice. That placement is the whole point: it is the only line
  separating a turn the model played AROUND a Fire Blast from one where it never saw the move
  coming — the board, the battle log and the critic's numbers read identically in both. `α`'s top
  four sit on the card (`SWITCH` in the accent every switch on this page already uses); the full
  distribution and `β` — *if they switch, who comes in* — live in the drop-down beside our own
  policy distribution, which is the honest pairing: two distributions, ours over our actions and
  `α` over theirs. `β` names a slot by the model's own species posterior, and the panel SAYS so
  rather than letting a believed mon read as the board. Absent entirely on a run without the heads
  (every trace before v67) — no line, never an empty one and never a fabricated 0%.
- **It says whether the model SAW THE LOSS COMING, twice.** Above the replay, the battle-level
  verdict — a `blind loss` / `knew @ turn N` badge and `engine.awareness_text`'s sentence,
  printed, never re-worded here. Then under each decision's critic row, a **`P(win) · dist` strip**
  with the 50% bar drawn on it, tinted and railed from the sustained onset on, so scrolling the
  replay SHOWS where the read turned rather than asking the reader to trust the badge. The strip is
  a bar and not a number because the fact is a CROSSING, which a column of percentages hides.
  Beside it sits **`tail`** — the catastrophic-band mass — because tail mass piling up under a
  still-positive mean is the stall signature, and it is precisely what the scalar V on the same
  line cannot show. Both are suppressed below a **0.5% legibility floor**, the same one
  `awareness_text` applies: "tail 0%" reads as a finding when it is rounding noise. Absent entirely
  on a run with no dist head — never a 0%, which would be a claim the trace cannot support.

  **It reads as P(WIN), not P(loss) — ONE direction per card**, matching the win-prob head on the
  line above, so higher always means better and the fill shrinks as the position sours (a SHORT bar
  is the dangerous one, the same association `.hpfill` already builds on this page). Two different
  quantities are therefore both called P(win) here, and they must stay distinguishable: the head is
  a **calibrated classifier**, the strip is the **return distribution's own mass above zero**. The
  strip carries `· dist` for exactly that reason. `p_win` is computed in `awareness.py` and shipped
  on the payload rather than flipped in the template — `1 - x` in a view is a view deriving a
  number. Every THRESHOLD stays defined on `p_loss > 0.5`: this is a presentation of one crossing,
  not a second definition of it.
- **It says WHAT THEY PICKED, on the card.** `expect` is a prediction, and a prediction is only
  readable next to its outcome — "Drill Peck 41%" means one thing when Drill Peck is what came and
  another when it was not. So the option the opponent actually took is marked in the `α` line, and
  a `they` line under it names the pick outright. Until 2026-08-18 that comparison needed either
  expanding `details` or reading the move back out of the battle log.
  **`not expected` is the case worth seeing from across the page**: `α` never listed the move at
  all, which is a different failure from ranking it low (measured on a real turn — the model
  expected a SWITCH plus four moves; they used Dragon Claw). The match is
  `engine.build_opp_intent`'s, not a view's: `α` carries display names (`Drill Peck`) and the
  recorder an id (`drillpeck`), so it needs normalizing plus a Hidden Power rule — a bare
  `hiddenpower` (an opponent's un-revealed HP) matches any typed HP option, but never the reverse,
  since a specific recorded type must not match a different believed one.
- **Every number in the critic row explains itself, in two places.** Each carries a `title` that
  says what it IS and how to read it (V's zero is not "even"; ΔP is percentage POINTS, not a "%";
  TD δ is the critic's surprise), and because **a tooltip does not exist on a touch device** — and
  this is the one view built to be read on a phone — the same explanations are collected once in a
  collapsed **legend** at the top of the page rather than repeated under fifty turn cards.
- **P(win) sits beside V, not instead of it** (`win_prob`/`delta_win_prob`, in percentage POINTS
  via the `signed_pp` macro — a difference of probabilities is not a "%"). V is a shaped,
  discounted return whose zero is not "even" (a measured self-mirror 50/50 reads about −6.5), so
  the two disagree in sign routinely and only the calibrated one reads as odds. Absent on the
  great majority of traces, which have no such head.
- **The default battle, and the picker, are NEWEST-first** (`_newest_first`). `ProbeSession.battles()`
  is ordered by step *ascending*, so a naive `rows[0]` default landed visitors on a battle played by
  the run's oldest checkpoint.
- **It renders server-side on first paint** — measured **17–20 ms** for `battle_turns()` on that
  249-turn battle, ~110 ms for the whole page. Nothing here needs to be async.

**A run with no traces is an EMPTY STATE, not a 404.** Both battle-addressed pages (`/battle`,
`/analyze`) resolve a battle before rendering, and a run that has captured nothing has none — which
surfaced as a 404, indistinguishable from a bad link on a perfectly healthy run. It is also not an
edge case: the app opens the NEWEST run by default and a freshly-launched run has no traces until
its first eval cycle (gen-9 sat exactly there for hours on 2026-08-13). `_NoBattles` is caught by
the two page handlers and rendered as a message pointing at the run summary; the JSON API still
returns the status code, and an unknown battle token is still a real 404 that never echoes the
token. Pinned by `test_a_run_with_no_traces_is_an_empty_state_not_a_404` +
`test_a_battle_that_does_not_exist_is_still_a_404`.

**A battle is named by its `short_id`, and the name is checked for MEMBERSHIP** — `runs.py`'s rule
one level down, and load-bearing for the same reason: `ProbeSession._battle` falls back to
`build_trace_tree(battle_id)` for an id it does not recognise, which will happily open a
`*_summary.json` belonging to **another run**. `app.battle_row()` therefore matches the token
against the run's own battle listing and passes the session a path the *server* produced. The
reversion test proved it: without that check a pinned single-run instance served a sibling run's
trace with a **200**.

### "Did it know?" — the awareness layer, on three views

One fold (`main/prober/awareness.py`), surfaced wherever it changes a reading. It is **model-free**,
so unlike `/analyze` it works on every run at any architecture — and `None` throughout on the runs
with no distributional head, which is most of them.

| where | what it adds |
|---|---|
| `/battle` | the battle verdict above the replay + a per-decision **`P(win) · dist` strip** (see above) |
| `/scan` | `knew @` and `lead` columns beside each crater — `BLIND` badged when it never saw it coming |
| `/triage` | a `blind` / `median lead` column per category, **beside** the lever, never folded into it |
| `/` | the run-level panel: the aggregate against the published **gen-10 baseline** |

**The baseline is data, not copy.** The gen-10 figures live in `awareness.AWARENESS_BASELINES` and
ride `awareness_scan()`'s own payload as `aggregate.baseline`, so this page renders them the same
way it renders the live numbers — the one rule applied to a reference point. A baseline typed into
a template is one the CLI would eventually disagree with.

**Two honesty rules the panel enforces in the markup**, because a comparison table invites reading
every row as a like-for-like verdict:

- **cap-aware@5 prints its `n` on both sides** and says "a direction, not a rate". The baseline is
  over **12** cap losses; at that n the fraction moves in quarter-steps.
- **The two coverage rows are NOT comparable at the default filter.** The baseline was measured
  over ALL outcomes; the panel defaults to losses, which are the low-outcome tail, so a filtered
  PIT is biased low *by construction*. The page says so next to those rows and links the
  unfiltered read, rather than leaving it in the caveats fold. Same fix landed in the CLI, where
  `--outcome` had only `win`/`loss` — the probe's own caveat was prescribing a reading its
  interface could not produce (`--outcome all` now does).

It loads **async on `/`** (`hx-trigger="load"`, the `/scan` pattern): it reads every matching
battle's npz, so the run summary must not sit behind it. Measured on a real run (gen-11, **1292
losses**): `awareness_scan()` **5.4 s** cold, against `scan()`'s 2.0 s warm — the same order, and
both taken on a box carrying a live trainer, so treat them as upper bounds.
`render_integration_test.py` requires a completed swap there — a route returning 200 to a test
client would not catch the panel spinning forever.

### Every hand-off points HERE, not at a retired terminal

`/analyze` is a web view. It loads the checkpoint and renders faithfulness, beliefs, threat
tables, intervention and saliency in the browser — but until 2026-08-18 the replay's drop-down
footer, its per-decision button, the page lede and the `scan` table all told the reader to go and
run a CLI command *"or the TUI"*, a surface **retired on 2026-08-13**. The feature was built,
shipped and reachable, and the copy around it said it lived somewhere that no longer existed.

Both the replay and every `scan` row now LINK straight to `/analyze?run=…&battle=…&inv=N`; the
CLI equivalent stays offered beside it, because that is a real second surface. Pinned by
`test_every_hand_off_goes_to_the_web_view_not_a_retired_terminal`, which also asserts the string
"the TUI" appears in neither view — the cheapest possible guard against the same copy drifting
back.

**The lesson is about docs, not code:** a retirement has to sweep the *user-facing copy*, not just
the module. Deleting `app.py` left ten pointers to it in templates, and every one of them read as
an instruction.

### `/analyze` — the one view that loads a checkpoint

The per-decision forensic read, ported here as part of retiring the TUI. Page shell + an HTMX
fragment (`hx-trigger="load"`, the `/scan` pattern) because it deserializes a checkpoint; the
battle and `inv` stay plain GET params, because "look at this decision" is a thing you link to.

**Read the arch-drift section in `src/main/prober/CLAUDE.md` before trusting anything here.**
Re-running the model needs a checkpoint at the CURRENT architecture, and measured over `models/`,
**79 of 79 archived runs cannot load**. So this view's ordinary output on an old run is an
`ArchDriftError` diagnosis — obs dim, `arch_signature`, dropped flags, and the exact `git checkout`
— which the fragment renders whole (`white-space: pre-wrap`) rather than collapsing to "analysis
failed". Every other view is model-free and unaffected; that asymmetry is why they were built first.

Two faithfulness banners are load-bearing and must never be dropped: `obs_mismatch` (every
obs-offset decode below is reading past a divergence) and `model_resolution.dropped_kwargs` (flags
the current code no longer accepts were dropped to make the load possible, so the rebuilt extractor
is not the one that played). The panels self-hide per flag-gated head, so an absent panel reads as
"that head was off", never as a broken probe.

**The genuine upgrades over the terminal**, both because a browser can draw: the distributional
critic's return distribution is a real **chart** (the TUI could only manage a one-line eighth-block
sparkline, where "sharp vs wide vs bimodal" — the entire point of the head — was a judgement call
about eight characters), and the operator's `incoming_matrix` is a real **heatmap table** (opp
candidate move × our mon), where the terminal had to fake a 2-D grid as an indented list.

**The beliefs section carries TWO readings of the species belief, and the order is load-bearing**
(`#beliefs-exclusive`, `a.exclusive_belief`). The raw per-slot marginals come first — that is what
the model actually believes — and the **species-clause reading** sits below it: the most likely
hidden team gen3 would allow, plus the slots where the two readings disagree. The panel is
deliberately QUIET when the belief was already coherent (one line, no second table) and it says in
its own copy that it changes nothing. Measured on gen-15, **14.2% of decisions display two hidden
slots naming the same mon**, which is the case it exists for; see `src/main/prober/CLAUDE.md` →
*The SPECIES-CLAUSE reading* for what it is not.

**The counterfactual tier rides `/analyze`** — `lookahead`, `better_line` and
`replay_counterfactual` are per-DECISION probes, so they launch from the bottom of this page
pre-filled with the current battle + inv, rather than getting a page of their own. They spawn Node
and run for seconds to minutes, so they go through the **job registry** exactly like `falsify_scan`
(submit → job id → poll `/partials/job/{id}`) and they are **password-gated** by the same rule:
reading is anonymous, spending CPU is not.

Two handoffs are ported from the TUI's `L`→`C` flow: a finished `lookahead` offers its best
non-chosen alternative straight to the replay probe, and `better_line` renders the interior-opponent
provenance with a **self-proxy banner** at depth ≥ 2 (the trainee standing in for the opponent is an
approximation, and a contrastive line that does not say so reads as fact). `replay_counterfactual`
at `n_rollouts=1` is a single realized-dice line and **not** a probability — the session says so in
its `caveats` and the page renders them.

`interior_opponent="ckpt"` is deliberately NOT exposed: it takes a filesystem path from the client,
which is the one thing `runs.py`'s membership rule exists to prevent. CLI-only.

⚠ **A COVERAGE HOLE, stated rather than papered over.** `/analyze` is in the headless render test's
page list, but the fixture run has no loadable checkpoint, so what the browser actually measures is
the page shell and the **arch-drift error state**. The POPULATED markup — the incoming-matrix
heatmap and the wide faithfulness/threat tables, i.e. exactly the widest things on the site — is
covered by unit tests (strings in the HTML) and by an eyeball, **not** by the measured layout gate
(`overflowby` / `scrollers` / `monstack`). Closing it needs a fixture checkpoint at the current
architecture; until one exists, treat "the heatmap does not overflow on a phone" as unproven.

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
| `rows` | data rows present in the DOM (table rows **and** battle-replay turn cards) |
| `monstack` | on `/battle`: whether the two mons stacked (phone) or sat side by side (desktop) |
| `scheme` · `bg` · `colorscheme` · `linkcolor` · `axistext` | the PALETTE, measured (see below) |
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

## The two palettes, and why the dark one needs its own gate

Dark is the BASE (`:root`) and light is a `prefers-color-scheme` override, so the default open on a
dark desktop needs no toggle and no flash. But **only one palette is ever on screen**, which makes
review-by-screenshot cover exactly half the stylesheet. Three defects lived in the unseen half
until the record started measuring it (`scheme` · `bg` · `colorscheme` · `linkcolor` · `axistext`):

- **Chart text was Vega's near-black on `#16161a`.** `charts._BASE` makes the chart background
  transparent so it sits on the page, and a static spec emitted by Python cannot know the theme —
  so `app.js` applies the theme at EMBED time (`themeConfig` → `themed`), reading the live values
  back out of the stylesheet's own custom properties rather than restating them. Spec-provided
  config still wins over the theme, so a chart that deliberately sets a colour keeps it.
- **Unclassed `<a>` kept the browser default `#0000EE`** — about **2.4:1** against the dark
  background, under the 4.5:1 floor. Only the link *classes* were styled, so any plain link was
  missed. Fixed with a base `a { color: var(--accent) }` rule, which a later link cannot undo.
- **`color-scheme` was never declared** (measured `normal`), so the browser rendered its OWN
  widgets — `<select>` menus, scrollbars — in the light style: a white dropdown on a dark page.

**A dark screenshot FLATTERS the page, which is why the third one hid.** `--force-dark-mode` is
the only way to make `prefers-color-scheme: dark` match in headless chrome (there is no CLI switch
for the media feature), and it *also* darkens UA widgets — which a real dark-mode visitor does not
get. So the widget defect was invisible in every screenshot and only the `colorscheme` reading
catches it. The expected colours in `render_integration_test.py` are written out literally rather
than re-derived from the stylesheet: a test that computes its expectation from the same source it
is checking cannot catch the palette failing to apply.

## Responsive layout (desktop + phone), and how it is gated

**The one rule: the PAGE never scrolls sideways.** Wide content scrolls inside its own container —
`.scroll-x` for tables, `.chart` for an oversized SVG. Everything else follows from that.

These are tables of forensic numbers, so the phone answer is deliberately **not** to reflow every
table into cards: a `scan` row read out of column order is worse than one you scroll. **`/battle`
is the one exception, and it earns it** — a turn is a short narrative (board → choice → what
happened → what the critic thought), not a row whose columns carry the meaning, so it is built as
cards that reflow: the two mons stack, the log wraps, and the page needs no scroll container at
all. That claim is measured, not asserted — `app.js` publishes **`monstack`** (did the two boards
stack, or are they side by side?) and the render test requires `1` at 500px, `0` at 1280px, plus
`scrollers == 0` so the view cannot quietly regress into "grew a scrollbar".

Under `@media (max-width: 720px)` the layout instead: drops the sticky header and truncates the run path,
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
  to fail on it — and `app_test.py` asserts that failure renders. Its `opp_intent` blocks come in
  **two shapes on purpose** (one decision where `α` expects an ATTACK, one where it expects a SWITCH
  so `β`'s named mon is promoted onto the card) and on only ONE of its battles — a fixture carrying
  a single shape would leave the `β` path and the heads-off path ungated. Because the fixture
  carries them, the measured layout gate (`overflowby` / `scrollers` / `monstack`) covers the new
  markup with no new assertion.

  **Its distributions follow the same one-of-each rule**, and the `model_config.json` that declares
  the atom support: one **blind** loss that also carries the **stall signature** (both recorded
  values positive, so P(loss) never crosses the bar, while 30% of the second decision's mass sits
  in the bottom atoms — the exact pathology the head exists to make visible), one loss it **called**
  (values under water, so the onset marker and the `knew` strip render), one trace with **no
  `value_dist` at all** (the counted-but-never-judged path), and win-prob on exactly one battle so
  the far more common no-win-prob rendering stays gated.

  ⚠ **A distribution must agree with its own recorded V.** The awareness fold denormalizes the
  support by a least-squares fit over the trace's `(dist mean, recorded V)` pairs, so a row built
  to a chosen P(loss) is silently relocated by that fit — a first draft did exactly that and read
  back `p_loss = 0.00` on every decision. The rows are therefore built as a SHAPE centred on the
  recorded value, and P(loss) falls out of `(V, spread)` the way it does in a real trace. A
  requested tail mass the mean cannot carry is a hard error, never a quiet degradation: on
  `[-12, 12]` no distribution with mean 10 holds 30% at −10.

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
