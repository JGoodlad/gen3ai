# `/game` — the battle viewer (v2): story, opponent intent, attention, operator facts

Owned by this tree (always current, updated in the same pass as the code). Owner request,
2026-10-08: *"The prober seems very outdated … make it better, especially now that we have the full
Rust protocol parsing so each turn has its details. Improve the battle prober including opponent
intent, and show where the model spent its attention, like a heat map."*

`/game` is THE battle viewer: the classic `/battle` replay was MERGED into it on 2026-10-09 and
redirects there (307, `start=N` → `turn=N`). Why it is shaped as below, the owner's complaints that
drove it and everything removed: [`battle_viewer_ux_2026-10-09.md`](battle_viewer_ux_2026-10-09.md).
Every number comes back from a `ProbeSession` method verbatim (web `CLAUDE.md`'s one rule); the
engine does the arithmetic, the page only draws.

**The page** (`web/game.py`, `templates/game.html`, the components in `_game_macros.html`), one battle
at a time, one decision selected (`/game?run=…&battle=…&inv=N&view=model|truth|public`, or `turn=N`
for that turn's first decision):

- **the battle bar** — the result, opponent, step and length; ‹ › to the neighbouring battles; the
  picker (the app's one `_picker_rows`, always containing the battle shown); the PERSPECTIVE switch
  (§0); the **P(win) strip** — the critic's RECORDED P(win) at every decision (model-free: under the
  win-prob critic `values` IS P(win)), the current decision marked, each point a link to it;
- **the turn rail** (left; a drawer on a phone) — ONE row component per turn, a fixed grid of at least
  44px: the turn number, then our line and their line (`us` / `them` + the side's primary action:
  `▸` a move, `⇄` a switch, `⏸` can't move, `—` none; `✕` a faint, `↳` a forced replacement), and on
  the right the recorded P(win) at the turn's decision, `▼` on the three biggest drops (the session's
  `notable`) and the result word on the last turn. The current turn is filled and `aria-current`;
- **the decision** (centre) — the header (turn, decision k of n, "forced replacement" when it is one,
  P(win) before → after in points, what we chose at what probability, the glossary `?`, the `/analyze`
  hand-off); **the board it was made on** (`battle_board`: the start-of-turn board, or the MID-turn board
  after the faint for a forced replacement), us left, them right, the same row format on both sides,
  the active mon's moves / item / ability, and OUR TEAM SHEET folded under it; **what we could do vs what
  they did** — the same sorted distribution component on both sides (our recorded policy; their α when
  the model is loaded, else just what they did), the actual choice marked `✔` / `◀`, unavailable options
  listed once in grey; **what happened** — the turn's ordered BEATS (§1);
- **the model's read** (right column ≥ 1280px, below otherwise) — the SCOUTING NOTES on their team
  (§6; their public half renders without the model), then **under the hood**, collapsed: attention,
  damage physics, the raw pointer scores, the raw protocol of the turn;
- **the step bar** (phone only) — ‹ prev · turn · next › as 44px buttons, sticky at the bottom.

### 0. The information perspective (model-free)

Every board fact is a field `{"v": value, "vis": "public" | "ours" | "hidden"}` built by
`engine/perspective.perspective_board` from the protocol fold (what was announced) and both teams'
reconstruction sheets (`_our_team_details` / `_opp_team_details`): `public` = a protocol line announced
it (a mon that appeared, its HP %, the moves it used, an item / ability a `[from]` / `-item` /
`-enditem` / `-ability` line named, or an ability the species can only have one of — the Smogon table);
`ours` = known to our agent only (our six mons from turn 0, our exact HP, our full sets); `hidden` =
unknown to our agent at that point (their unrevealed mons, moves, items, abilities). The page's
perspective decides what renders, through ONE rule, `perspective.shown(vis, view)`, injected into Jinja
as `shown` and applied by the macro `fv`: `public` everywhere, `ours` in `model` and `truth`, `hidden`
only in `truth` and always MARKED (`◇`, a dashed underline, the class `truth`, `data-vis="hidden"`).
The model's choices and beliefs are its mind, not the board, so they show in every perspective.
`perspective_guard_test.py` is the class guard. Without a reconstruction record there is no truth and
the switch offers only the other two.

## The scope ruling — CURRENT ARCHITECTURE ONLY (owner, 2026-10-08)

*"The prober only needs to support the current version; totally fine with that since we are still
rapidly iterating."* So there is NO arch-drift worker (a pinned-checkout forward was designed in the
first brief and DROPPED by this ruling before any of it was built):

- the MODEL-FREE half of `/game` (the story: turns, events, board, our choice vs the legal set)
  works on **every** run with traces, at any architecture — it reads the trace and the protocol;
- the MODEL half (intent, hypotheses, attention, pointer scores, operator facts, the re-run win
  probability) loads the battle's checkpoint through the session's ONE resolution ladder
  (exact → nearest → recent) and works only when that checkpoint is at HEAD's architecture. On an
  older run the model panels render the typed `ArchDriftError` as one plain sentence —
  *"this run's architecture (config vN, signature S) is older than the code (…); model views need a
  current-architecture checkpoint"* (the sentence is `exc.plain`; its `kind` — `arch_signature`, `obs_dim`,
  `obs_semantics`, `no_checkpoint`, … — is on the card as `data-model-reason`) — with the full diagnosis folded
  under it, and the story above still renders. The page asks `ProbeSession.model_status` and shows the reason on
  FIRST PAINT when the views cannot run (no loader, no password prompt). Never a 500, never a blank panel.

## Where each view's data comes from, and what it costs

| view | session method | source | cost (measured, CPU, 2026-10-08) |
|---|---|---|---|
| turn list + turn story (events, board, HP) | `battle_story(battle_id)` | `battle_turns()` (the core walk's expansion) + the battle's protocol log, folded by `engine/turn_events.py` | one protocol fold over the battle; the core walk is the existing (cached) expansion |
| our choice vs the legal set | `battle_story` | the recorded action distribution + mask | free |
| win-prob over the battle, opponent intent, hypotheses, pointer scores, operator facts, attention summary | `battle_readout(battle_id)` | ONE batched eager CPU forward over every recorded decision's stored obs (`states.npz`), with read-only forward hooks (`model_capture.py`) | ~0.1 s per 64 decisions on the production arch, plus the checkpoint load once per session (cached by path) |
| the full attention heat map for one decision | `decision_attention(battle_id, inv, layer, head)` | the cached capture of the same forward | free after `battle_readout` |

The readout is cached IN MEMORY per (checkpoint path, battle summary path) — a bounded LRU on the
session (`_READOUT_CACHE_CAP`), dropped by `close()`. Nothing is written anywhere: not under
`models/` (read-only) and not to disk, the prober's standing convention for derived data.

### Access: the story is open, the model panels need the shared password

`/game` is public at prober.g5d.io, so the split above is also the access split (`web/gate.py`,
`web/CLAUDE.md` "Access"): the turn story (`/api/game/story`, the page's story half) is model-free
and stays anonymous — measured 2026-10-09 on a real 1,448-battle run, ~20 ms of Python plus ~30 ms of
the `core_events --walk` child per UNCACHED battle (~50 ms CPU, under the 100 ms bar), 4 ms warm. Every
route that loads the checkpoint or runs the forward (`/api/game/readout`, `/api/game/attention`,
`/partials/game/model`, `/partials/game/attention`) carries `model_gate`: a locked visitor's page
renders the story and, in the model slot, one "Unlock to view the model's analysis" card
(`partials/model_locked.html`, `data-model-state="locked"`) whose link returns to the same battle and
decision; no HTMX request is made that would be refused, and a stale-cookie request to a fragment
answers the same card as a 200 (HTMX swallows a 403). The JSON endpoints answer a plain 403.
`web/gate_guard_test.py` is the class guard.

### 1. The turn story (model-free)

**The turn is read as ordered BEATS** (`engine/turn_beats.py`): one actor's action plus every
consequence it caused, in the order the sim ran them, each with its PHASE — `lead` (turn 0), `start`
(before the first action), `switch` (by choice), `move` (a `|move|` or `|cant|`, numbered in execution
order; the first is "moved first"), `replace` (a FORCED replacement — gen 3 sends it straight after the
action that caused the faint, so it can sit between two moves, or after `upkeep` for a residual KO),
`residual` (from the residual action's opening blank line to `|upkeep|`) and `end`. The phase is read off
the protocol's own framing (`sim/battle.ts` `runAction`), never guessed from wording. Inside a beat:
damage is an HP-bar DELTA (before → after, the lost slice hatched) with its cause — the move, or the
`[from]` source in display case — the effectiveness / critical-hit lines that preceded it are TAGS on it,
the hits of a multi-hit move merge into one line, and a faint carries its cause ("our Skarmory's Drill
Peck", "its own Explosion", "Sandstorm"). A forced-replacement beat links the decision it was; a
move-selection decision whose mon fainted before it moved adds a "never got to use X — it fainted
first" line. Every mon is named by its SPECIES (a nickname never reaches the page — the HEAD smoke's
French nicknames, and one mojibake nickname in `data/teams/`, did). The flat `events` list stays in the
JSON, each event stamped with its `beat` and `phase`; `turns[i].summary` is the rail's per-side line.

Per game turn: the protocol events in the order the sim emitted them, each a typed row
(`kind` ∈ move · damage · heal · switch · drag · faint · status · cure · boost · unboost ·
crit · supereffective · resisted · immune · miss · fail · cant · weather · hazard · screen ·
item · ability · other) with the SIDE it happened to (`we` / `opp`), the mon, the amount in % of
max HP where the line carries HP, the SOURCE (`[from] item: Leftovers`, Sandstorm, recoil, Spikes, …)
and a plain-English sentence. The board at the end of each turn is a fold over the same lines: both
sides' six mons (species, HP %, status, fainted, which one is in), the active mons' stat stages,
each side's hazards and screens, and the weather. Our side's HP comes in exact points
(`202/301`) and theirs in hundredths (`92/100`); both are shown as % of max, and the fold says which
it saw. The decisions of that turn sit under it with our choice against the legal set (illegal
options grey, never red).

Why the protocol and not the recorded summary alone: the summary's per-turn `timeline` carries
one line per action and drops everything that is not an action — the `-resisted`, the recoil, the
Leftovers, the sand chip, the hazard set. That is exactly the "details" the owner asked for, and
the Rust core now emits the full protocol for every core trace (`core_trace.protocol_log`).

### 2. Opponent intent per decision (model)

The flat opponent pointer (`flat_intent_head`, ARCHITECTURE §2.1) is ONE softmax over: the
opponent active's K move seats (revealed moves first), OTHER_move ("some move not in the seats"),
a switch to each of their six slots, and OTHER_species ("a switch to a mon we have not guessed").
The page draws every live candidate as a bar — SORTED by probability, beside our own options in the
same component (swapped in out of band, `#game-intent`), the action they actually took marked `◀` —
each labelled in words
(`Earthquake (seen)`, `Ice Beam (guess)`, `any other move`, `switch → Skarmory (seen)`,
`switch → Blissey (guess, 51% present)`, `switch → a mon not on our list`).

Beside it, the opponent's ACTUAL next action — the label — mapped onto the same columns by the
same rule training's intent label uses: a move beyond the seats is OTHER_move, a switch-in that is
not among the slots' species is OTHER_species (a bare `hiddenpower` matches any Hidden Power seat,
never the reverse). A post-faint replacement is not a choice and gets no label.

**Calibration of α across the battle** (battle level): over the decisions with a label, the mean
probability the model gave the action that came, the top-1 hit rate, the mean log loss (nats), and a
coarse reliability table (the model's top pick's probability, binned, against how often that top
pick came). Small n — a single battle — so it is DESCRIPTIVE and the page prints the n.

**The hypothesis tokens** (`HypothesisBuilder`, X5): per opponent slot, either the REVEALED mon
(named off the board) or the hypothesis that hidden slot holds, with its presence π (the model's
probability that species is on their team); OTHER_species' mass and its five likeliest members;
and, where the reconstruction carries the opponent's true team, whether each guess is ON their team.
**Beliefs over time** is a battle-level chart: the presence of every species that was ever a
hypothesis, decision by decision, with each reveal marked.

### 3. Attention heat maps (model)

Captured with forward PRE-hooks on each trunk round (the post-LN `BiasedEncoderLayer`s, then — under
`--trunk-layers 3/4` — the pre-LN `IdentityInitRound`s appended after them), in eager mode, with no
change to model code: the hook receives the round's input `x` and its additive bias, and recomputes
`softmax(q kᵀ / √d_head + bias)` from the round's own `in_proj` (over `norm1(x)` for a pre-LN round, whose
attention reads the normed input) — exactly the logits
`scaled_dot_product_attention` consumes (the bias already carries the key-padding addend and every
edge family). A unit test pins the recomputation against the layer's own output (re-deriving the
layer's output from the captured weights). Hooks are installed for the duration of one capture and
removed in a `finally`; nothing in the model changes, so the cost when off is zero.
`PolicyStateQuery` (built only under `--policy-readout trunk`, OFF in production) is captured the
same way when it exists, and shown as "what the decision summary looked at"; absent otherwise.

Shape: layer × head × query token × key token (production: 2 × 4 × 62 × 62). Token labels per
decision, in plain words: our six mons (by species), their six mons (revealed species, or the
hypothesis a hidden slot holds, marked *guess*), the board token(s) (one `board` token under
`legacy`, `our side` / `their side` / `field` under `static`), E3 `our move: <name>`, E4
`their move: <name>`, E5 `their other moves (<mon>)` (the active's is `any other move`),
OTHER_species `a mon not on our list`, and the event seats `event −N` (most recent last). Padded
tokens (a fainted mon, an empty event row) are masked and drawn blank.

The default view: the AVERAGE over layers and heads, plus the two rows that matter as ranked bars —
**what the chosen action's token attended to** (a move's E3 seat, or the switch target's team token;
the pointer head scores each action from its own token) and **what our active mon attended to**.
A layer / head picker re-draws one map. Beside it: the pointer head's score (logit) per legal action
and the resulting probability, and on the battle strip the critic's win probability decision by
decision (re-run on the stored observation — a core trace stores no win-prob column).

Attention is a reading aid, not an explanation: a large weight says the token's VALUE was mixed in,
not that it caused the choice. The page says so once, under the map.

### 4. Operator facts per action (model)

From the `DamageOperator`'s pre-gain stash after the same forward (`decode_damage_block`), for each
of our legal moves against their active: the damage range (low–high, % of their max HP), P(KO),
the crit roll's damage, P(we move first), and P(lands) for a status move. The move-resolution family
(`--move-resolution on`, OFF in production) adds P(resolve) and P(KO first) per move; when it is not
built those two columns render GREY with "not built in this run's model" — present-at-this-architecture
features show, absent ones are greyed, nothing crashes.

### 5. Navigation

`←` / `→` (or `j` / `k`) step decision by decision, `[` / `]` step battle by battle within the run's
listing; every position is a URL (`/game?run=…&battle=…&inv=N&view=…`) that works with JavaScript off
(the rail is links). `/battles` and `/scan` rows and `/analyze` link here, anchored on the decision; the
old `/battle` URLs redirect here. A link to a collapsed `<details>` (`#glossary`) opens it.

A trace with no protocol log (an old python-era trace without its `_replay.html`) still lists every
decision's turn — an empty row that says so — so no decision is orphaned; its board panels are absent.

## Measured (CPU, 2026-10-08, a 65-decision production-arch battle from a `--debug --arch production` run)

`battle_story` 0.6 s cold (the core walk's expansion; cached after), `battle_readout` 1.75 s cold
including the checkpoint load (the capture itself 0.12 s), `decision_attention` < 1 ms from the cache;
`/game` 0.76 s cold, the model fragment 1.9 s cold. The readout JSON is ~690 KB (mostly the per-decision
attention summaries); the page renders one decision at a time, so it never ships it whole.

## Abbreviations

Every abbreviation on the page carries a `title` that expands it (`α` = the model's prediction of
the opponent's action, `π` = presence, `P(KO)` = probability of a knock-out, `E3`/`E4`/`E5` = the
model's move tokens, …), and the panels are titled in plain words first and the model's term second.
