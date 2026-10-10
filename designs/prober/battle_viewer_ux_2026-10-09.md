# The battle viewer, redesigned (2026-10-09): audit, design, and what was removed

**Status:** design of record for the prober's battle viewer overhaul. The owner handed the redesign
over end to end on 2026-10-09 ("let the prober agent drive the change"), with latitude to remove
and consolidate. This note is the record: what was wrong, what the viewer is now, and what was
deleted and why. The always-current field map stays in [`battle_view_v2.md`](battle_view_v2.md);
this note explains the decisions behind it. Owner's words, collected over the evening:

> "The switch, faint etc. is so hard to understand … it isn't colour-coded for we vs opp, it
> doesn't have good parity, no order of the uncertain. And I would like it if we could load the
> model for battle if logged in, to at least include opponent intent too … what is the difference
> between battle and game? Game has the better left-side UI. The pills on the left aren't
> consistent etc. I feel like UX is very poor right now."
>
> "battle and game have this annoying difference of public information versus what we can see."
>
> "I feel like 'just ugh' with it right now. Please give the model latitude to remove things,
> really focus on UX. I am not looking for a few bug fixes, I want this doing better."

**The bar:** a strong competitive player opens a battle and immediately understands what happened
and how the model thought — and can judge "how is the model doing, compared to how I'd be
reasoning?".

Screenshots (before / after, desktop 1440 px and phone 390 px, locked and unlocked) live in
`~/gen3ai_archive/prober_ux_2026-10-09/{before,after}/`. Audit inputs: the real run
`rb_st_static_s1008` (Rustboro era, 1,448 traced battles; its checkpoints predate the X5 break, so
its model views are locked out as an older architecture) and a HEAD-architecture CPU smoke
(`--debug --arch production --debug-eval`, config v151, 54 traced battles) served from a scratch
models directory outside `models/`.

---

## 1. Audit: what a reader met before this change

### 1.1 Two viewers that disagreed about the same battle

| | `/battle` ("classic replay") | `/game` ("battle viewer") |
|---|---|---|
| layout | one long page of turn cards, 50 per window | turn list on the left, one decision on the right |
| what happened | the summary's one-line-per-action timeline (`we explosion — outcome unrecorded`, `opp sends in metagross`) | every protocol line as a typed row (`SWITCH`, `MOVE`, `RESISTED`, `CRIT`, `DAMAGE`, `FAINT` …) |
| our team | **all six from the start** (the core walk's own-team view) | **only the mons that had appeared** (the protocol fold): "4 not seen yet" about OUR OWN team |
| our HP | rounded % | exact-derived %, no points |
| names | lower-case ids (`switch:metagross`, `hydropump`) | display names, but **nicknames** ("our Leuphorie switched in for Tyranocif") |
| side colour | ours = body text, theirs = dim grey | ours = green, theirs = red (and a `-supereffective` / `-crit` row overrode the side colour with amber) |
| opponent intent | the RECORDED α/β line — absent from every Rust-core trace, so absent on every current run | the re-run α — model-gated, and an arch-drift sentence on every archived run |
| the critic | V · ΔV · TD δ · r on every card | a re-run P(win) chart |

Two pages, two data paths, two perspectives on who knows what, two vocabularies, two colour
schemes. The owner's "public information versus what we can see" complaint is exactly the our-team
row: one page showed the agent's knowledge of its own team, the other a spectator's.

### 1.2 The turn story could not be read at a glance

On real battles (`rb_st_static_s1008` `step_14000018/sentinel_4/loss_3005`, the HEAD smoke's
`step_8000/heuristic2/loss_001`):

- **Consequences were separate rows from their cause.** `MOVE our Metagross used Explosion` was
  followed by four unattached rows (`RESISTED`, `CRIT`, `DAMAGE their Metagross lost 100% — now 0%`,
  two `FAINT`s). Nothing said the faint was BECAUSE of the Explosion, or that our Metagross fainted
  from its own move.
- **A forced replacement read as a voluntary switch.** "our Charizard switched in for Metagross"
  — after Metagross fainted — is the same sentence, glyph and colour as a deliberate switch on the
  previous turn. It is also a separate DECISION (`forced_switch`), and nothing linked the two.
- **No order.** The rows were in emission order but nothing marked who moved first, where the
  moves ended and the end-of-turn residuals began, or that Gen 3 sends a replacement in straight
  after the move that caused the faint (verified at the source:
  `deps/pokemon-showdown/sim/battle.ts` `runAction`, "in gen 3 or earlier, switching in fainted
  pokemon is done after every move, rather than only at the end of the turn").
- **Damage was a sentence, not a picture.** "their Metagross lost 100% — now 0%"; no HP bar, and
  the cause (the move, Spikes, Sandstorm, Leftovers) sat in free text with inconsistent case
  (`spikes` vs `Sandstorm`).
- **Wrong words.** `|move|…|Protect||[still]` rendered as "used Protect (charging)" — `[still]`
  only suppresses the animation; Protect does not charge.
- **Nicknames.** The HEAD smoke's teams carry French nicknames, so the story read "our Leuphorie
  switched in for Tyranocif" (Blissey for Tyranitar), and one nickname is mojibake in the source
  team files themselves (`MÃ©talosse`, 13 occurrences under `data/teams/` — a data finding, §17).

### 1.3 The turn list ("the pills on the left")

Each row showed the first two move-or-switch events of the turn as free text, in side colours, plus
a `× Metagross` faint badge with **no side** (turn 2's `× Metagross × Metagross` is one of each).
Rows were 24–45 px tall on a phone depending on how the text wrapped (measured, 390 px), 11 px
type, and the only state was "current" (an outline). Nothing distinguished a forced switch, the
turn the game turned, or the result.

### 1.4 Uncertainty had no order and no single format

`/game`'s α bars were sorted, but the team-guess table was in SLOT order, the action list in ACTION
order, the operator table in move order. Probabilities appeared as `1%`, `35%`, `0.29`, `51%
present`, `3.11 expected mons`, `+0.01` (raw scores) and `11.2%` on one screen, with bars on three
different scales (the attention bars were ×4).

### 1.5 The model views

They worked (on a current-architecture run, unlocked) but sat BELOW the story, so the model's read
of the opponent was a screen away from the board it was about; the model's beliefs were a
hypothesis-token table (slot, species, π) rather than anything a player would recognise as
scouting; the per-slot move / item / spread beliefs the network computes every forward were not
shown at all (only the active's E4 move seats).

### 1.6 The phone

At 390 px the decision panel started **905 px** down the page (header, lede, picker, the 170 px turn
box); nav tabs were 26 px tall, the prev/next buttons 35 px, the "deep analysis" / "classic replay"
links 18 px — all under the 44 px touch floor.

### 1.7 Defects found on the way

- **The battle picker LIED**: `/game` sliced the listing to 200 rows before rendering the `<select>`,
  so a battle outside the newest 200 (here `sentinel_4/loss_3005`) showed a DIFFERENT battle as
  selected (`random/win_001`) — a picker that names the wrong game. (`/battle`'s `_picker_rows`
  already guarded this; `/game` did not use it.)
- "(charging)" on `[still]` (above).
- Hazard / weather source names in mixed case.

---

## 2. Principles

1. **One viewer, one vocabulary, one data path per fact.** Every surface that shows a fact gets it
   from the same session field and renders it through the same macro.
2. **Perspective is explicit.** Every board fact carries WHO COULD KNOW IT; one control on the page
   decides what is shown, and ground truth is never mixed in silently.
3. **Read a turn like a player does:** board → options on both sides → what each side did → what
   happened, in the order it happened → what the model believed.
4. **Parity.** Our side and theirs use the same component, the same scale, the same marks, mirrored
   (ours left, theirs right).
5. **Uncertainty is ordered and uniform:** every distribution sorted by probability, one percent
   format, one bar scale (linear 0–100 %), and the thing that actually happened marked in place.
6. **Colour is never the only carrier:** a side chip says `us` / `them`, a glyph says what an event
   is, a word says what a mark means.
7. **Remove what does not earn its place** (§13).

---

## 3. Information architecture

**ONE battle viewer at `/game`.** `/battle` answers **307** to `/game` (its `battle` / `run` kept,
`start=N` mapped to `turn=N`), so every old link — scan rows, analyze's footer, bookmarks — lands
in the viewer. `/api/battle-turns` stays (a JSON contract, used by the CLI's `turns`). The
`battle.html` / `turns_list.html` templates are deleted.

The page, top to bottom (desktop):

1. **Battle bar** — result, opponent, step, length; ‹ › to the neighbouring battles; the battle
   picker (readable labels, always containing the battle shown); the **perspective control**
   (§4); the **P(win) strip** — the critic's recorded win probability at every decision of this
   battle (model-free: under the win-prob critic V *is* P(win) and the trace records it), the
   current decision marked, each point a link to its decision.
2. **Turn rail** (left, sticky; a drawer on a phone) — the one pill component (§9).
3. **The decision** (centre) — turn and decision header with P(win) before → after; the **board at
   the start of the turn**, ours and theirs mirrored; **our options vs their next action** (§6);
   **what happened**, as ordered beats (§5).
4. **The model's read** (right column on a wide screen, below on a narrow one) — **scouting notes**
   on their team (§7), then **under the hood** (attention, damage physics, raw scores, the raw
   protocol), collapsed.

A locked visitor gets 1–3 in full (they are model-free) and, in 4 and in the "their next action"
column, ONE compact unlock card each — no request that would be refused, exactly the gate's
contract.

---

## 4. Information perspective — a first-class concept

Three perspectives, defined once (engine: `main/prober/engine/perspective.py`):

| perspective | contains | when |
|---|---|---|
| **Spectator** (`public`) | what anyone watching the battle saw: mons that appeared, HP as %, moves used, items / abilities the protocol announced, status, boosts, hazards, weather | `view=public` |
| **What the model saw** (`model`, the DEFAULT) | spectator + OUR OWN team's private facts: all six of our mons from turn 0, our exact HP (`180/343`), our full movesets and items | default — the viewer exists to judge the model's decisions, so it shows what the model observed |
| **+ Truth** (`truth`) | the model's view + the opponent's HIDDEN facts from the battle's reconstruction record (their unrevealed mons, moves, items, abilities, spreads), each one MARKED | `view=truth` |

**Per-field visibility is computed in the engine, never in a template.** Every board field is a
`{"v": value, "vis": "public" | "ours" | "hidden"}` pair: `public` = announced on the protocol;
`ours` = known to our agent only; `hidden` = unknown to our agent at that point (ground truth).
ONE Jinja macro (`fv`) renders a field under the page's perspective:

- `public` renders in every perspective;
- `ours` renders in `model` and `truth`, never in `public`;
- `hidden` renders ONLY in `truth`, always wrapped in the truth marker — `◇` + a dashed underline +
  the class `truth` + `title="ground truth — hidden from the model at this point"` — never colour
  alone.

Every panel that shows a board fact (the board, both teams, our team sheet, the scouting notes'
"actually" column) goes through `fv`. The control is one labelled three-way switch in the battle
bar ("Show: What the model saw · + Truth · Spectator") with a one-line legend, and it is a URL
parameter, so a perspective is linkable and works with JavaScript off.

**The class guard** (`web/perspective_guard_test.py`): a fixture battle whose reconstruction plants
unique hidden values (an opponent item, move and mon that the protocol never reveals) is rendered
in every perspective; the `model` and `public` pages must not contain any planted value at all, the
`public` page must not contain our private values, and on the `truth` page every planted value must
sit inside a `data-vis="hidden"` element carrying the truth marker. A panel that renders a hidden
field without going through `fv` fails it.

What the model OBSERVED is the agent perspective — but the model's BELIEFS (intent, team guesses,
sets) are its mind, not the board, so they show in every perspective; only their "actually"
comparison is a truth-overlay field.

---

## 5. The turn story — ordered beats

The engine (`engine/turn_events.py`) now groups a turn's protocol lines into **beats**: one actor's
action plus every consequence it caused, in the order the simulator ran them. Each beat carries its
**phase**:

| phase | what | how it is recognised (the protocol's own framing) |
|---|---|---|
| `start` | start-of-turn effects (Focus Punch tightening) | lines before the turn's first action |
| `switch` | a voluntary switch | a `|switch|` that does not follow its side's faint |
| `move` | a move (or "can't move"), with its ORDER in the turn — the first is marked **moved first** | `|move|` / `|cant|`; a `[from]` move (Sleep Talk's call) stays in its caller's beat |
| `replace` | a FORCED replacement after a faint — Gen 3 sends it straight after the action that caused the faint, so it can sit between two moves | a `|switch|` for a side whose active fainted since its last switch |
| `residual` | end of turn: weather, Leftovers, poison, Leech Seed … | from the residual action's opening blank line to `|upkeep|` (and the faints after it) |

Inside a beat: **damage is an HP-bar delta** (before → after, the lost portion hatched, the cause
named — the move, or the `[from]` source in display case), effectiveness / critical hit / miss /
immune / fail are TAGS on the line they qualify, and a **faint carries its cause** ("KO'd by our
Metagross's Explosion", "its own Explosion", "Sandstorm"). Every mon is named by its SPECIES (a
nickname never reaches the page), with the side chip `us` / `them`. A forced-replacement beat links
the decision it was (`↳ Charizard — forced, decision 3`).

Glyph legend (one legend, used by the rail, the story and the board): `▸` a move · `⇄` a switch by
choice · `↳` sent in after a faint · `✕` fainted · `⏸` couldn't move · `◇` ground truth.

```
What happened in turn 2                                         (in the order the game ran it)
 SWITCH   them  Milotic ⇄ Metagross                       came in at 100%
 MOVE 1st us    Metagross ▸ Explosion
                  them Metagross  [##########··········]  100% → 0%  −100%  not very effective · critical hit
                  ✕ them Metagross fainted — KO'd by Explosion
                  ✕ us Metagross fainted — its own Explosion
 REPLACE  us    ↳ Charizard — sent in after Metagross fainted          (decision 3 ›)
          them  ↳ Snorlax — sent in after Metagross fainted
```

The flat `events` list stays in the JSON (additive: each event gains `phase`, `beat` and the mon's
`species`), so the CLI contract does not move.

---

## 6. The decision: our options vs their next action (parity)

Two columns, the SAME component, mirrored:

```
 US — what we could do (the policy)          THEM — what we expected them to do (α)
 ✔ Explosion          ███████░░░░░  35%       ◀ Surf (seen)          ████████░░░░  41%  what they did
   switch → Celebi    ███░░░░░░░░░  17%         switch → Metagross   ████░░░░░░░░  20%
   Psychic            ███░░░░░░░░░  16%         Ice Beam (guess)     ██░░░░░░░░░░  12%
   Meteor Mash        ██░░░░░░░░░░  11%         any other move       █░░░░░░░░░░░   5%
   …                                           …
   not available: switch → Metagross (in battle) · Struggle
                                               over this battle: top guess right 7 of 12 · gave
                                               what they did 41% on average (descriptive, n = 12)
```

- Both sorted by probability, one bar scale, one format: integer percent, `<1%` for a non-zero
  probability below half a percent, `>99%` below one, `0%` / `100%` only when exact.
- The actual choice is marked in place on both sides: `✔` + "we chose", `◀` + "what they did".
- Unavailable options leave the ranking and are listed once underneath, grey — "unavailable" must
  never read as "dangerous".
- Our side is model-FREE (the recorded policy at play time). Their side needs the model: a locked
  visitor sees "what they did" (public, from the story) and the compact unlock card; an older
  architecture shows the one plain sentence.

---

## 7. Scouting notes — the model's beliefs, the way a player tracks the opponent

Per opponent mon, a card — the active first, then the others in reveal order, then the slots the
model has not seen:

```
 them ▶ Skarmory  · 100% · seen turn 1                      [truth: moves 3/4 · item ✓]
   Moves   Spikes (seen) · Protect (seen) · Drill Peck 87% · Roar 64% · Toxic 31%
           ◇ actually: Spikes, Protect, Drill Peck, Toxic       ✗ Roar 64% is not in its set
   Item    Leftovers 71% · Shell Bell 9% · Lum Berry 6%          ◇ actually: Leftovers ✓
   Spread  slow, physically bulky — Spe 176 (bottom of 176–262) · Def invested · Impish 41%
           ◇ actually: Impish, 252 HP / 252 Def
   Since the last decision: Drill Peck 62% → 87% (after: it used Spikes, Protect)

 their unseen mons — who the model thinks is still in the back
   Tyranitar 49% · Blissey 39% · Swampert 38% (✗) · Gengar 37% · Starmie 27% (✗)
   someone not on this list: 97% at least one
   ◇ actually: Starmie, Gengar, Dugtrio
```

- **Moves**: the network's per-slot move posterior (`last_move_belief_logits`, sigmoid presence),
  seen moves pinned and labelled `seen`, the rest ranked; Hidden Power carries its believed type
  (`last_hp_type_logits`).
- **Item**: the item posterior (`last_item_logits`), top three.
- **Spread**: the SpreadBelief's derived stats, described as a player would ("fast", "bulky")
  from where each believed stat sits in the species' possible range, with the top nature when the
  nature head is built. The thresholds are a DISPLAY rule (top / middle / bottom third of the
  range), stated in the glossary.
- **Ability**: shown when the protocol revealed it, or when the species has only one; never
  invented.
- **Since the last decision**: the moves / items whose probability moved by 10 points or more, and
  the public reveals in between that touched this mon — "after: it used Spikes" — so a reader sees
  the update AND its evidence. Correlational, and it says "after", not "because".
- **Unseen slots**: the hypothesis tokens' species with presence, sorted, plus OTHER_species' "at
  least one" mass.
- **Truth overlay** (`view=truth`): an "actually" line per field, and a per-mon verdict chip —
  moves right / missed / false (a true move is "believed" at ≥ 50 % presence), item ✓ / ✗ (the top
  item is the true one), the unseen-team guesses ✓ / ✗.
- **Locked**: the card still renders its PUBLIC half (seen moves, revealed item, HP, status) — the
  same scouting a spectator could do — with the model columns replaced by one unlock card.

---

## 8. Under the hood (collapsed, model views)

- **Attention**: the two ranked lists first — what the chosen option's token looked at, what our
  active mon looked at (top 8, one bar scale) — then the full heat map with its layer / head
  pickers inside a `<details>`, in its own horizontal scroller.
- **Damage physics** — our moves against their active (range, crit, P(KO), P(lands), P(we move
  first)), sorted by expected damage.
- **Raw pointer scores** per option.
- **Raw protocol** of the turn (model-free; also shown to a locked visitor).
- A link to `/analyze` for the decision (faithfulness, saliency, threat tables, counterfactual
  probes) — it stays the deep, per-decision forensic page.

---

## 9. The turn rail — one component

```
 ┌──────────────────────────────────────────┐
 │ 12 │ us   ▸ Thunderbolt              ✕   │  ← our line: blue tick, glyph, label, marks
 │    │ them ⇄ Heracross                    │  ← their line: orange tick
 │    │                          62% ▼      │  ← P(win) at the turn's decision; ▼ = a big drop
 └──────────────────────────────────────────┘
```

- Every row is the SAME grid: turn number · our line · their line · P(win) — fixed 44 px minimum
  height, ellipsised labels (full text in `title`), so rows never wrap into different heights.
- Each line shows that side's primary action this turn (`▸ move`, `⇄ switch`, `⏸ can't move`,
  `—` none) and its marks (`✕` fainted, `↳` sent in after a faint).
- States, each with a non-colour cue: **current** (filled accent background + bold + a left bar,
  `aria-current`), **a forced-switch decision** (`↳` on our line and a second decision link),
  **a big P(win) drop** (`▼`, the three largest, from the session's `notable` block), **the last
  turn** carries the result word (WIN / LOSS / DRAW), **no decision** (turn 0's leads: dim, not a
  link).
- It IS the navigation: each row is a link to its turn's first decision; the keyboard keeps
  `←` / `→` (and `j` / `k`) for decisions and `[` / `]` for battles.

---

## 10. The phone (T29)

At 360–430 px: the header shrinks to brand + a wrapped nav (the `analyze` tab is gone, §13); the
battle bar becomes two rows; **the rail becomes a drawer** — a `<details>` "Turn 13 of 50 · all
turns" holding the SAME row component, closed by default; **a sticky bottom bar** carries
‹ previous · Turn 13 · 2/3 · next › as 44 px buttons; the board stacks (us, then them); every
distribution keeps one column; the heat map stays behind its `<details>` with the two ranked lists
first. Gated in the browser tier at 360 × 800, 390 × 844 and 430 × 932: no horizontal page scroll,
the key panels present, and every tap target in the rail, the decision nav and the perspective
control at least 44 × 44 px.

---

## 11. Accessibility

- **Side colours**: blue for us, orange for them — the colour-blind-safe pair — on both palettes,
  every one measured against both backgrounds (WCAG contrast): dark `#6cb2ff` 8.1:1 / `#ffa057`
  9.0:1, light `#1a5fb4` 6.1:1 / `#a8500f` 5.3:1; the truth marker `#c99cff` 8.3:1 / `#7b3fbf`
  6.1:1. Win / loss keep green / red and are now the ONLY use of them (the old ours = green
  conflated "ours" with "good").
- Never colour alone: side chips carry the words `us` / `them`; events carry glyphs and phase
  words; the truth marker carries `◇` and a dashed underline.
- Visible focus rings on every link and button (`:focus-visible`), keyboard navigation kept,
  `aria-current` on the current turn, `aria-label`s on the bars.

---

## 12. Wireframes

### Desktop, unlocked, a current-architecture run (≥ 1280 px)

```
 gen3ai prober   run · triage · scan · battles · game · falsify · calibration   [run ▾] unlocked
 rb_ux_head_smoke · 2 eval steps · 54 battles · 1W 52L
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ ‹ LOSS  vs heuristic2 · step 8,000 · 50 turns · 55 decisions  ›           [pick a battle ▾] │
│ Show: (•) What the model saw  ( ) + Truth  ( ) Spectator   — public + our own team          │
│ P(win) ▁▂▂▃▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁│▁▁▁▁▁▁▁▁                                                          │
├──────────────┬──────────────────────────────────────────┬──────────────────────────────────┤
│ TURNS        │ Turn 13 · decision 13 of 55   ‹ prev next ›│ THEIR TEAM — scouting notes       │
│ 0 │us Tyranit│ P(win) 9% → 8% (−1 pt)                    │ them ▶ Skarmory 100% · seen T1    │
│   │them Skarm│ ┌ us ──────────────┐┌ them ──────────────┐│  Moves  Spikes·Protect (seen) …   │
│ 1 │us ⇄ Bliss│ │▶Blissey ██▌ 38%  ││▶Skarmory ████ 100% ││  Item   Leftovers 71% …           │
│ … │          │ │ Tyranitar ███ 63%││ ? 5 not yet seen   ││  Spread slow, bulky …             │
│13▐│us ⇄ Bliss│ │ … 4 more         ││                    ││ their unseen mons: Tyranitar 49%… │
│   │them▸Prote│ │ Spikes ×1        ││                    ││                                   │
│ … │          │ └──────────────────┘└────────────────────┘│ UNDER THE HOOD ▸                  │
│              │ US — options            THEM — next action │  attention · damage · scores ·    │
│              │ ✔ ⇄ Blissey 11% …       ◀ Protect 12% …   │  raw protocol                     │
│              │ WHAT HAPPENED (in order)                  │                                   │
│              │ SWITCH us Tyranitar ⇄ Blissey             │                                   │
│              │        Blissey [####····] 100→88 Spikes   │                                   │
│              │ MOVE 1st them Skarmory ▸ Protect — failed │                                   │
│              │ END OF TURN Sandstorm · Leftovers …       │                                   │
└──────────────┴──────────────────────────────────────────┴──────────────────────────────────┘
```

Between 721 and 1279 px the right column drops below the centre one (two columns: rail + main).

### Desktop, locked

The same page; the THEM column of "next action" shows "what they did" + a one-line unlock card,
the scouting notes show their public half + one unlock card, under the hood shows only the raw
protocol + the unlock card. No request is made that would be refused.

### Phone (390 px)

```
 gen3ai prober                         unlock
 run triage scan battles game falsify calibration
 [run ▾                                      ]
┌──────────────────────────────────────────────┐
│ ‹ LOSS vs heuristic2 · step 8,000          › │
│ 50 turns · [pick ▾]                           │
│ Show: [model] [+truth] [spectator]            │
│ P(win) ▁▂▂▃▃▂▂▁▁▁▁▁▁▁│▁▁▁▁▁                    │
├──────────────────────────────────────────────┤
│ ▸ Turn 13 of 50 · all turns          (drawer) │
├──────────────────────────────────────────────┤
│ Turn 13 · decision 13/55 · P(win) 9% → 8%     │
│ us   ▶ Blissey ██▌ 38% · Tyranitar 63% …      │
│ them ▶ Skarmory ████ 100% · 5 not seen        │
│ US — options          (sorted bars)           │
│ THEM — next action    (sorted bars / unlock)  │
│ WHAT HAPPENED                                 │
│ … beats …                                     │
│ THEIR TEAM — scouting notes                   │
│ UNDER THE HOOD ▸                              │
├──────────────────────────────────────────────┤
│ [‹ prev]     Turn 13 · 13/55      [next ›]    │  ← sticky, 44 px buttons
└──────────────────────────────────────────────┘
```

---

## 13. Removed, and why

| removed | why |
|---|---|
| the `/battle` page (classic replay) and its templates | a second viewer with a second data path and a second perspective; every element that earns its place moved into `/game` (rows below). The URL now redirects (307), so no link breaks |
| `/battle`'s critic row: TD δ and r | under the win-prob critic γ = 1 and r = 0 until the end, so TD δ = ΔV on every row and r printed `0.00` fifty times; V is P(win) and now lives in the decision header and the P(win) strip |
| `/battle`'s "did it know?" awareness verdict and `P(win) · dist` strip | the distributional head was deleted (L1); they rendered nothing on any current run |
| `/battle`'s recorded `expect` / `they` α/β line | no Rust-core trace records `opp_intent` (null on every battle of every current run); the model-backed intent (§6) replaces it on current-architecture runs |
| the per-card reward breakdown | the win-indicator terminal has no shaping terms |
| the turn window (`start=`, 50 turns per page) | the rail lists every turn and the page shows one decision; `start=` is mapped to `turn=` by the redirect |
| `/game`'s lede paragraph | a paragraph of prose above the content on every visit; the key hints live in the rail's footer and the glossary |
| the hypothesis-token TABLE | superseded by the scouting notes' unseen-mons line (same data, sorted, in words) |
| the re-run P(win) chart | on the `exact` tier the re-run equals the recorded value; the recorded strip is model-free, so everyone sees it |
| the separate "Our options, as the action head scored them" table | a duplicate of the policy list; the raw scores moved under the hood |
| the "board after the turn" fold | the story now shows every change; the next decision's board is the after-board |
| the `analyze` nav tab | `/analyze` is a per-DECISION page; opened from the nav it guessed a decision. It is reached from every decision in the viewer (and the scan table); one tab fewer is one row fewer on a phone header |
| green = ours / red = theirs | conflated "ours" with "good" and failed red–green colour blindness |

---

## 14. Engine, session and API changes

All additive to the JSON (`battle_story` / `/api/game/story`, `battle_readout` / `/api/game/readout`);
the CLI-facing `battle_turns` / `/api/battle-turns` is untouched.

- `engine/turn_events.py` — beats + phases, species on every event, display-case sources, faint
  causes, the `[still]` fix; `fold_turns` keeps its signature and output keys.
- `engine/perspective.py` (new) — the per-field visibility board from the protocol fold + both
  teams' reconstruction details; `battle_story` returns it per turn (`turns[i].view`).
- `session/game.py` — `battle_story` adds `beats`, the perspective board, the recorded P(win)
  series and `notable`; `battle_readout` adds `scouting` per decision (beliefs + their change +
  truth).
- `model_capture.py` — captures the per-slot move, item, spread / nature and Hidden-Power-type
  posteriors (top-k per slot, so the readout stays bounded).
- `web/game.py` — `/game` takes `turn` and `view`; `/battle` redirects.

## 15. Tests that fail on revert

- `/battle` → 307 to `/game` with the battle kept and `start` → `turn`; `/game` accepts `turn`.
- every story event and beat carries `side` ∈ {we, opp, null} and the rendered rows carry the side
  class + the `us` / `them` chip.
- the phase ORDER within a turn on a real battle (`sim` tier: a Rust-core battle with an Explosion
  double-KO and a residual turn) and on a synthetic protocol (unit).
- the intent candidates rendered in descending probability, the actual one marked.
- locked vs unlocked rendering (the gate's contract, `gate_guard_test.py` unchanged and green).
- the perspective guard (§4).
- the browser tier at 1440 / 430 / 390 / 360 px: no horizontal overflow, the rail / board / story /
  options present, tap targets ≥ 44 px.

## 16. Slices

1. this note; 2. the merged viewer + the story redesign + the perspective model + the rail;
3. the model views on the battle (intent column, scouting notes, under the hood); 4. nav;
5. the phone. Each ships on its own with the prober suite, the browser tier and the static gates.

## 17. Findings from the audit (not fixed here unless stated)

- **Every archived run's model views are locked out by architecture**, including the Rustboro
  `rb_st_*` runs (trained at obs 2761 / `gen3_event_record_v2`, before the X5 break). The model
  half of the viewer works on runs trained at HEAD's architecture; until a current run exists the
  owner will see the one-sentence arch notice there. (The parallel run-picker / arch-drift work
  owns the diagnosis.)
- **`data/teams/` carries mojibake nicknames** (`MÃ©talosse`, 13 occurrences across
  `data/teams/sample/` and `data/teams/others/mcmegan/`) — UTF-8 read as Latin-1 at import. The
  viewer no longer prints nicknames, so it is invisible here; whether a nickname reaches anything
  the model reads is a separate question (not checked).
- The `/game` battle picker named the wrong battle (fixed in slice 2).
