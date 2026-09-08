# `/battle` — the turn-by-turn replay, field by field

Owned by this tree (always current, updated in the same pass as the code). The rules the page rests
on — it is not HTMX, a run with no traces is an EMPTY STATE and not a 404, a battle is named by its
`short_id` and that name is checked for MEMBERSHIP, every number comes back from a `ProbeSession`
method verbatim — stay in `src/main/prober/web/CLAUDE.md`. This file is what each element carries.

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
  `α` over theirs. Absent entirely on a run without the heads (every trace before v67) — no line,
  never an empty one and never a fabricated 0%.

  🚨 **A `β` name says WHERE IT CAME FROM, per row.** A name the recorder read off the board renders
  plain; a name the model's species POSTERIOR produced renders with the engine's caveat tag
  (`engine.BELIEF_NAME_CAVEAT`, injected as a Jinja global so the tag on a row and the legend
  explaining it are one string). That distinction is not decoration: the posterior is un-supervised
  on a slot the board already revealed, and over a 843-battle sweep (2026-08-19) it named a mon not
  on the opponent's team at all in 73.3% of 6,876 pivots — one such bare label was read as *"β
  predicts porygon2"* on a turn where `β`'s slot held the revealed Salamence and `β` was CORRECT.
  **Every pre-fix trace is entirely posterior-named**, so on an archived run every `β` name carries
  the tag; the page never repairs one (see `src/main/prober/CLAUDE.md`). A row with no species at all
  renders as a bare `slot 4` and is NOT tagged — a caveat needs a name to qualify.
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
- **Every number in the critic row explains itself, THREE ways** — because the first two were not
  enough on the device this view is built for. Each carries a `title` (what it IS and how to read
  it: V's zero is not "even"; ΔP is percentage POINTS, not a "%"; TD δ is the critic's surprise),
  and the same explanations are collected once in a collapsed **legend** at the top of the page.
  **A `title` has NO touch equivalent**, though, and the legend is at the top of a 50-turn page, so
  a reader at turn 13 on a phone had no way to ask — which is exactly how it was reported. So:
  - **tap a number** and `app.js` renders THAT metric's own `title` into a panel directly under its
    row (`.metric` + the delegated handler; the title stays the single source, so nothing is
    duplicated into a data attribute that could drift from the tooltip);
  - a **`?` link on every row** anchors to `#turncard-legend` — the no-JS, always-works route.
  The dotted underline is the discoverability half: without a visible affordance nobody learns a
  number can be tapped. ⚠ `--dump-dom` cannot dispatch a click, so the browser gate proves the
  markup and app.js's own selector match the live DOM (`metrics`), NOT the panel opening — stated
  in the test rather than implied by a green tick.
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
