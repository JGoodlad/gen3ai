# Gen-3 Protocol Line Inventory

The exhaustive catalogue of every distinct `|...|` protocol line **type** that appears
in `tests/vectors/protocol_capture_golden.txt` (the captured omniscient stream from
`harness/gen_protocol_capture.js`), with its token grammar, an example, the fiddly
formatting rules, and — for each — whether **our poke-env fork parses or ignores it**.

This is the emission target list for the level-2 port: `protocol.rs` must reproduce
these bytes exactly. The design for hooking the engine to emit them is in
`src/rust_sim/PROTOCOL_EMISSION_DESIGN.md`.

## What "the stream" is (read first)

We capture the **omniscient** (referee / spectator) stream — the *clean* full stream:
- **No `|request|` blocks.** The per-player streams carry `|request|{json}` (the legal-
  action + side JSON); the omniscient stream does not. `|request|` is emitted downstream
  by the bridge's `getPlayerStreams` privacy fold, NOT by the engine's core stream. The
  Rust engine emits the omniscient stream; the bridge (`local_sim_bridge.js`) keeps doing
  the per-side fold. So `|request|` is **out of scope for engine emission**.
- **No `|split|` lines.** `getPlayerStreams` uses `|split|<side>` internally to fold
  opponent HP to a percentage per viewer; the omniscient stream never contains it and
  shows full `x/y` HP for **both** sides.
- **Full `x/y` HP both sides.** The percentage fold (`x/100`) is the per-side viewer's
  job, not the engine's.

So the Rust engine's job is the omniscient stream; the bridge already produces the
per-side streams poke-env actually consumes. **This is why the golden captures the
omniscient stream: it is exactly the byte set the engine owns.**

## The parse/ignore split (our poke-env fork)

Two layers decide what poke-env does with a line
(`src/poke_env/player/player.py::_handle_battle_message` →
`src/poke_env/battle/abstract_battle.py::parse_message`):

1. **Player layer** (`_handle_battle_message`) peels off the battle-control lines
   BEFORE `parse_message`: `request`, `win`, `tie`, `error`, `bigerror`, `showteam`,
   `init` (multiline-battle detection). Everything else → `parse_message`.
2. **Battle layer** (`parse_message`) first checks `MESSAGES_TO_IGNORE`
   (`abstract_battle.py:47`) — if `event[1]` is in that set it **returns immediately**;
   otherwise it dispatches on `event[1]`.

> **Hard constraint for the port — every emitted type must be handled or ignored.**
> `parse_message`'s final `else` branch is `raise NotImplementedError(event)` — it does
> NOT silently drop unknown types. So any `|...|` line the engine emits that is neither
> in `MESSAGES_TO_IGNORE` nor explicitly dispatched **crashes poke-env**. The port's
> emission set must therefore be a subset of {ignored} ∪ {explicitly-handled}. Emitting
> a `debug`/`t:` line is safe (ignored); emitting a *new unhandled* type is not.

`MESSAGES_TO_IGNORE` (the ones that appear in our capture): `-anim`, `-block`, `-burst`,
`-center`, `-combine`, `-fieldactivate`, `-hint`, `-hitcount`, `-ohko`, `-waiting`,
`-zbroken`, `J`, `L`, `askreg`, `badge`, `c`, `chat`, `crit` (the legacy no-prefix
duplicate — NOT `-crit`), `debug`, `deinit`, `gametype`, `hidelines`, `html`, `immune`,
`inactiveoff`, `init`, `j`, `join`, `l`, `leave`, `n`, `name`, `rated`, `resisted` (the
no-prefix duplicate — NOT `-resisted`), `sentchoice`, `split`, `supereffective` (no-
prefix duplicate), `teampreview`, `upkeep`, `uhtml`, `uhtmlchange`, `zbroken`, `:`,
**`t:`**, `""`.

> **The ONE documented exception the task names: `|t:|` wall-clock lines are IGNORED**
> by poke-env (`t:` ∈ `MESSAGES_TO_IGNORE`). The engine still emits a `|t:|` line
> (Showdown does), but its value is a Unix timestamp — inherently un-reproducible AND
> unparsed — so the capture **normalizes it to `|t:|<NORMALIZED>`** and the byte-
> comparison test excludes it. See `PROTOCOL_EMISSION_DESIGN.md` §c.

## Inventory

Legend for the **poke-env** column:
- **PARSE** — `parse_message` dispatches on it (drives battle state / trackers).
- **PLAYER** — handled at the player layer, never reaches `parse_message`.
- **IGNORE** — in `MESSAGES_TO_IGNORE` (engine emits it; poke-env drops it).

### Pokémon identifier grammar (used everywhere)

A mon reference is `p<N><slot>: <Nickname>` — e.g. `p1a: Tyranitar`, `p2a: Snorlax`.
`p1`/`p2` = the side; `a` = the active position (gen-3 singles is always `a`);
`Nickname` = the display name (species name unless a custom nickname). A **side**
reference (side conditions) is `p<N>: <PlayerName>` — e.g. `p2: P2` (no position
letter). poke-env keys mons by the `p1a:`/`p2a:` ident and reads only the leading `p1`/
`p2` for side attribution.

### HP-fraction formatting (the #1 fiddly rule)

The HP field in `switch`/`drag`/`-damage`/`-heal`/`-sethp` is **`currentHP/maxHP`** on
the omniscient stream (e.g. `224/341`). Three variants the port MUST reproduce exactly:
- **Healthy, no status:** `224/341`
- **With a major status appended (space-separated):** `116/524 slp` — the status token
  (`brn`/`par`/`slp`/`frz`/`psn`/`tox`) is appended after a **single space**, e.g.
  `|-damage|p1a: Snorlax|116/524 slp|[from] Sandstorm` or
  `|-heal|p1a: Skarmory|102/334 par|[from] item: Leftovers`.
- **Fainted:** `0 fnt` — when HP hits 0 the field is the literal `0 fnt` (NOT `0/341`),
  e.g. `|-damage|p1a: Snorlax|0 fnt`. poke-env's `Pokemon.damage`/`heal` parse this form.

(On the **per-side** stream the OPPONENT's `x/y` is folded to `x/100` percent — that is
the bridge's fold, not the engine's; the omniscient golden keeps `x/y`.)

### Tag suffixes (`[from]` / `[of]` / `[still]` / `[miss]` / `[msg]` / …)

Trailing bracketed tags carry provenance/flags; the port emits them verbatim in the
sim's order. The ones seen in the capture:
- `[from] <cause>` — the effect that caused this line. Forms:
  `[from] item: Leftovers`, `[from] ability: Sand Stream`, `[from] move: Rest`,
  `[from] Sandstorm` (a bare field cause — no `item:`/`ability:`/`move:` prefix),
  `[from] psn`/`[from] brn` (residual DoT source, bare status token).
- `[of] <ident>` — the SOURCE mon of the effect, e.g.
  `|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar`. poke-env reads
  `[of]` in the `-damage`/`-heal` item/ability checks.
- `[still]` — a move that did nothing observable animates in place:
  `|move|p1a: Skarmory|Protect||[still]` (note the **empty target field** — two pipes).
- `[miss]` — a `|move|` whose accuracy roll failed:
  `|move|p2a: Tyranitar|Rock Slide|p1a: Skarmory|[miss]` (paired with a `|-miss|` line).
- `[upkeep]` — the end-of-turn weather tick: `|-weather|Sandstorm|[upkeep]`.
- `[msg]` — a `-curestatus` that shows a message: `|-curestatus|p1a: Snorlax|slp|[msg]`.
- `[damage]` — a Substitute that absorbed damage:
  `|-activate|p1a: Suicune|Substitute|[damage]`.
- `[weak]` — a Substitute that failed because HP was too low:
  `|-fail|p1a: Suicune|move: Substitute|[weak]`.
- `[silent]` — (not in this capture; Showdown suppresses the client message — the port
  should still emit the line where the sim does, e.g. some `-damage` residuals).

---

### Battle-init / framing lines (emitted once at battle start)

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|player\|` | PARSE | `\|player\|p<N>\|<name>\|<avatar>\|<rating>` | `\|player\|p1\|P1\|\|` | avatar+rating empty here (two trailing pipes). poke-env reads name→role. |
| `\|gametype\|` | IGNORE | `\|gametype\|singles` | `\|gametype\|singles` | in `MESSAGES_TO_IGNORE`. |
| `\|gen\|` | PARSE | `\|gen\|<n>` | `\|gen\|3` | poke-env asserts it equals the battle's gen. |
| `\|tier\|` | PARSE | `\|tier\|<display>` | `\|tier\|[Gen 3] Custom Game` | poke-env slugs it into `_format`. |
| `\|rule\|` | PARSE | `\|rule\|<text>` | `\|rule\|HP Percentage Mod: HP is shown in percentages` | recorded as a rule. |
| `\|teamsize\|` | PARSE | `\|teamsize\|p<N>\|<count>` | `\|teamsize\|p1\|2` | sets `_team_size`. |
| `\|start` | PARSE | `\|start` | `\|start` | ends team-preview (`in_team_preview=False`). No args. |
| `\|` (blank) | PARSE→no-op | `\|` | `\|` | a bare separator line; `event[1]==""` → `parse_message` treats it as a no-op. Emitted between phases (turn boundary, upkeep). The port MUST emit these. |
| `\|t:\|` | **IGNORE** | `\|t:\|<unixtime>` | `\|t:\|<NORMALIZED>` | **the documented exception.** Wall-clock; normalized in the golden. |

### Turn / phase lines

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|turn\|` | PARSE | `\|turn\|<n>` | `\|turn\|1` | increments the turn counter; a decision boundary. |
| `\|upkeep\|` | IGNORE | `\|upkeep` | `\|upkeep` | in `MESSAGES_TO_IGNORE`; still emitted (end-of-turn marker). No args. |

### Move / action lines

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|move\|` | PARSE | `\|move\|<user>\|<MoveName>\|<target>[\|tags]` | `\|move\|p1a: Suicune\|Surf\|p2a: Tyranitar` | Move display name (Title Case, spaced). Tags: `[miss]`, `[still]` (with an EMPTY target: `\|move\|…\|Protect\|\|[still]`), `[from]`, `[notarget]`, `[spread]`. The heavy poke-env handler (reveal, dancer, sleep-talk). |
| `\|cant\|` | PARSE | `\|cant\|<mon>\|<reason>[\|<MoveName>]` | `\|cant\|p1a: Tyranitar\|par` | reason ∈ `par`/`slp`/`frz`/`flinch`/`recharge`/… A blocked action (full-para, asleep, flinch). |
| `\|switch\|` | PARSE | `\|switch\|<mon>\|<Details>\|<HP>` | `\|switch\|p1a: Snorlax\|Snorlax\|524/524` | Details = `Species, L<lvl>, <gender>` (here just `Snorlax` — L100/genderless omitted). Handled with `drag`. |
| `\|drag\|` | PARSE | `\|drag\|<mon>\|<Details>\|<HP>` | `\|drag\|p2a: Swampert\|Swampert\|404/404` | identical grammar to `switch`; the FORCED (Roar/Whirlwind) entry. Same poke-env handler. |
| `\|faint\|` | PARSE | `\|faint\|<mon>` | `\|faint\|p2a: Tyranitar` | marks the mon fainted. |

### Damage / heal / HP lines

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|-damage\|` | PARSE | `\|-damage\|<mon>\|<HP>[\|[from] <cause>][\|[of] <src>]` | `\|-damage\|p2a: Tyranitar\|97/404` | HP variants: `x/y`, `x/y status`, `0 fnt`. Residual forms carry `[from] Sandstorm` / `[from] psn` / `[from] item:`. poke-env `_check_damage_message_for_{item,ability}` reads the `[from]`/`[of]`. |
| `\|-heal\|` | PARSE | `\|-heal\|<mon>\|<HP>[\|[from] <cause>][\|[of] <src>]` | `\|-heal\|p1a: Suicune\|308/404\|[from] item: Leftovers` | Leftovers/Rest/Wish/drain. poke-env `_check_heal_message_for_{item,ability}`. |
| `\|-sethp\|` | PARSE | `\|-sethp\|<mon>\|<HP>` | (not in capture — Pain Split) | poke-env sets HP directly. Emit where the sim does. |

### Status lines

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|-status\|` | PARSE | `\|-status\|<mon>\|<status>[\|[from] <cause>]` | `\|-status\|p1a: Snorlax\|slp\|[from] move: Rest` | status ∈ `brn`/`par`/`slp`/`frz`/`psn`/`tox`. Self-inflict (Rest) carries `[from] move: Rest`; a foe-inflicted status has no `[from]`. |
| `\|-curestatus\|` | PARSE | `\|-curestatus\|<mon>\|<status>[\|[msg]]` | `\|-curestatus\|p1a: Snorlax\|slp\|[msg]` | wake-up / thaw / Heal Bell. `[msg]` = show the message. |
| `\|-cureteam\|` | PARSE | `\|-cureteam\|<mon>` | (not in capture — Heal Bell) | team-wide cure. |

### Boost / stat lines

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|-boost\|` | PARSE | `\|-boost\|<mon>\|<stat>\|<amount>` | `\|-boost\|p2a: Metagross\|atk\|1` | stat ∈ `atk`/`def`/`spa`/`spd`/`spe`/`accuracy`/`evasion`. amount = stages (unsigned). |
| `\|-unboost\|` | PARSE | `\|-unboost\|<mon>\|<stat>\|<amount>` | `\|-unboost\|p1a: Tyranitar\|atk\|1` | the stat DROP (Intimidate, Crunch −SpD). |
| `\|-setboost\|` | PARSE | `\|-setboost\|<mon>\|<stat>\|<amount>` | (not in capture) | Belly Drum / Bulk-set. |
| `\|-clearboost\|` etc. | PARSE | `\|-clearboost\|<mon>` (+ `-clearallboost`/`-clearnegativeboost`/`-clearpositiveboost`/`-invertboost`/`-copyboost`/`-swapboost`) | (not in capture — Haze) | boost-table manipulations. |

### Effectiveness / immunity lines

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|-supereffective\|` | PARSE | `\|-supereffective\|<mon>` | `\|-supereffective\|p2a: Tyranitar` | `<mon>` is the DEFENDER; poke-env sets effectiveness 2.0 keyed on the defender's side prefix. |
| `\|-resisted\|` | PARSE | `\|-resisted\|<mon>` | `\|-resisted\|p1a: Aerodactyl` | effectiveness 0.5. (Note `resisted` — no dash — IS ignored.) |
| `\|-immune\|` | PARSE | `\|-immune\|<mon>[\|[from] ability:<A>]` | `\|-immune\|p1a: Skarmory` | effectiveness 0.0; an ability form reveals the ability. |
| `\|-crit\|` | PARSE | `\|-crit\|<mon>` | `\|-crit\|p2a: Skarmory` | a critical hit on the defender. (Bare `crit` IS ignored — only `-crit` is parsed; it feeds the crit tracker.) |
| `\|-miss\|` | PARSE | `\|-miss\|<user>[\|<target>]` | `\|-miss\|p1a: Cloyster\|p2a: Blissey` | a missed move (feeds the miss tracker). Paired with the `\|move\|…\|[miss]`. |
| `\|-fail\|` | PARSE | `\|-fail\|<mon>[\|<detail>][\|[weak]]` | `\|-fail\|p2a: Blissey\|par` / `\|-fail\|p1a: Suicune\|move: Substitute\|[weak]` | a move/effect that failed. detail = `par` (already statused), `move: Substitute`, etc. |

### Weather / field lines

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|-weather\|` | PARSE | `\|-weather\|<Weather>[\|[from] ability:<A>\|[of] <src>][\|[upkeep]]` | `\|-weather\|Sandstorm\|[from] ability: Sand Stream\|[of] p1a: Tyranitar` | the SET form carries `[from] ability:`+`[of]`; the per-turn TICK is `\|-weather\|Sandstorm\|[upkeep]`. `\|-weather\|none` clears it. |
| `\|-fieldstart\|` / `\|-fieldend\|` | PARSE | `\|-fieldstart\|<Effect>` | (not in capture — Trick Room etc., not gen3) | pseudo-weather. |

### Side-condition lines

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|-sidestart\|` | PARSE | `\|-sidestart\|<side>\|<Effect>` | `\|-sidestart\|p2: P2\|Spikes` | `<side>` is `p<N>: <PlayerName>` (no position letter). Spikes/Reflect/Light Screen. poke-env adds the side condition. |
| `\|-sideend\|` | PARSE | `\|-sideend\|<side>\|<Effect>` | (not in capture — Rapid Spin, not modeled) | removes a side condition. |

### Volatile / activate lines

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|-start\|` | PARSE | `\|-start\|<mon>\|<Effect>` | `\|-start\|p1a: Suicune\|Substitute` | a volatile begins (Substitute, Leech Seed, confusion, Flash Fire, typechange, Mimic). Heavy poke-env handler. |
| `\|-end\|` | PARSE | `\|-end\|<mon>\|<Effect>` | `\|-end\|p1a: Suicune\|Substitute` | the volatile ends (Substitute breaks). |
| `\|-activate\|` | PARSE | `\|-activate\|<mon>\|<Effect>[\|<detail>]` | `\|-activate\|p1a: Skarmory\|Protect` / `\|-activate\|p1a: Suicune\|Substitute\|[damage]` | an effect fires without start/end — Protect blocking, a Substitute absorbing (`[damage]`), Trick/Mimic/Leppa. |
| `\|-singleturn\|` | PARSE | `\|-singleturn\|<mon>\|<Effect>` | `\|-singleturn\|p1a: Skarmory\|Protect` | a one-turn effect announced (Protect, Focus Punch, Snatch). Explicit handler (`-singleturn`/`-singlemove`) → `start_effect(effect.replace("move: ",""))`. |
| `\|-ability\|` | PARSE | `\|-ability\|<mon>\|<Ability>[\|<detail>]` | `\|-ability\|p2a: Salamence\|Intimidate\|boost` | an ability reveals/fires. `boost` detail = Intimidate's boost trigger. poke-env sets the mon's ability. |
| `\|-endability\|` | PARSE | `\|-endability\|<mon>` | (not in capture) | ability suppressed. |
| `\|-item\|` / `\|-enditem\|` | PARSE | `\|-item\|<mon>\|<Item>[\|[from] …]` / `\|-enditem\|<mon>\|<Item>[\|[from] …]` | (not in capture — berry/Frisk/Knock Off) | item reveal / consume. poke-env reads Frisk/Pickpocket/Magician `[from]` forms. |

### End-of-battle lines (PLAYER layer — never reach `parse_message`)

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|win\|` | **PLAYER** | `\|win\|<PlayerName>` | `\|win\|P1` | `_handle_battle_message` calls `battle.won_by(name)` then ends the battle. NOT via `parse_message`. |
| `\|tie\|` | **PLAYER** | `\|tie\|` | `\|tie\|` | gen-3 double-KO tie; `battle.tied()`. No name. |

### Debug / diagnostic lines (IGNORED — but the engine still emits them)

| Type | poke-env | Grammar | Example | Notes |
|---|---|---|---|---|
| `\|debug\|` | IGNORE | `\|debug\|<text>` | `\|debug\|natural status immunity` | free-form sim debug (`move failed because it did nothing`, `natural status immunity`, `weather immunity`). In `MESSAGES_TO_IGNORE`. **The engine emits them** (Showdown does under `debug`/`-debug`) → the port must reproduce them for byte-equality, but the byte-comparison test may allow-list them like `t:` if the port chooses not to emit debug (see `PROTOCOL_EMISSION_DESIGN.md` §d — debug is a phasing decision). |

---

## Summary counts (this capture)

38 distinct line types captured across 11 scenarios × 6 seeds (9740 raw lines):

**High-frequency core (emit first):** `-damage`, `move`, `-heal`, `turn`, `upkeep`,
`switch`, `-weather`, `faint`, the blank `|` separator, `-resisted`/`-supereffective`,
`-crit`, `cant`, `-status`, `win`/`tie`, and the init framing (`player`/`gen`/`tier`/
`rule`/`teamsize`/`start`/`gametype`).

**Mid-tail (per-mechanic):** `-curestatus`, `-boost`/`-unboost`, `-immune`, `-miss`,
`-fail`, `-activate`, `-singleturn`, `-sidestart`, `drag`, `-start`/`-end`, `-ability`.

**Debug / normalized (special):** `debug` (ignored), `t:` (ignored + normalized).

## Line types NOT in this capture the port will eventually need

The capture covers the *modeled* gen-3 OU mechanics (matching the engine's current
coverage). These types exist in Showdown / are parsed by poke-env and will appear once
their mechanic is emitted, but are absent here:
`-sethp` (Pain Split), `-cureteam` (Heal Bell), `-setboost` (Belly Drum),
`-clearboost`/`-clearallboost`/`-clearnegativeboost`/`-clearpositiveboost`/
`-invertboost`/`-copyboost`/`-swapboost` (Haze / boost manip), `-sideend` (Rapid Spin),
`-item`/`-enditem` (berries / Knock Off / Frisk), `-prepare` (two-turn moves —
Solar Beam / Sky Attack), `-mustrecharge` (Hyper Beam), `-transform` (Ditto),
`-fieldstart`/`-fieldend` (not gen3-relevant), `-notarget`, `-hitcount` (multi-hit,
ignored), and the team-preview lines `clearpoke`/`poke` (gen-3 has no team preview → not
emitted). Each maps to a deferred engine layer; add its inventory row when that layer
lands (mirroring the phasing in `PROTOCOL_EMISSION_DESIGN.md` §d).
